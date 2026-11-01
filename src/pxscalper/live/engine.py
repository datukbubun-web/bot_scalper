"""Live scalping loop.

One loop, one instrument per iteration, one bar per decision -- scalping
discipline in code:

    1. wait for the bar to close (+ a small settle delay)
    2. pull quotes, positions, orders, account metrics      (3-4 REST calls)
    3. build features from the rolling buffer, normalise      (no look-ahead)
    4. primary model -> direction, RL -> veto/size            (microseconds)
    5. risk engine -> ALLOW / FLAT / BLOCK + size             (guard rails)
    6. impact -> place -> verify -> log                       (only if allowed)
    7. manage open positions: stop, target, time stop, trailing

Safety: ``PX_LIVE_TRADING=1`` and ``--i-understand-the-risk`` are both required
to send an order. Otherwise the loop runs in *paper* mode against live prices --
identical code path, no money.

State is durable: risk state, an append-only JSONL decision log and Prometheus
metrics survive restarts, and every decision carries the reason it was taken.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..exchange.models import Position, SymbolSettings
from ..exchange.rest import ApiError, PrimeXBTClient
from ..execution.router import ExecutionResult, OrderRouter
from ..features.pipeline import FeatureEngineer, FeatureConfig
from ..inference.predictor import Predictor
from ..risk.basket import BasketTarget
from ..risk.engine import RiskEngine
from ..strategy.decision import ScalpingStrategy
from ..utils import JsonlLogger, bar_ms
from .metrics import Metrics

log = logging.getLogger(__name__)

__all__ = ["LiveConfig", "LiveEngine"]


@dataclass
class LiveConfig:
    symbols: list[str] = field(default_factory=lambda: ["EUR/USD"])
    bar_size: str = "1M"
    lookback: int = 64
    warmup_bars: int = 900
    settle_ms: int = 1200            # let the feed print the first ticks of a bar
    loop_sleep_s: float = 2.5
    max_hold_bars: int = 45
    trailing_stop_atr: float = 0.0
    paper: bool = True
    require_confirmation: bool = True
    log_path: str = "logs/decisions.jsonl"
    state_path: str = "state/live.json"
    refresh_settings_s: float = 3600.0
    refresh_account_s: float = 5.0
    # ---- basket profit target ------------------------------------------------
    # Close ALL positions once the combined unrealised PnL reaches this amount,
    # even if some legs are still losing. Intended for scaled entry (several
    # positions in one symbol): the account target is what the operator asked
    # for, and waiting for every leg's own take-profit means giving the profit
    # back while the losers stay open.
    close_all_on_profit: float = 0.0     # 0 disables
    close_all_enforce: bool = False      # False = report only (paper observation)
    close_all_cooldown_s: float = 300.0  # re-arm delay after firing
    # ---- synthetic L2 depth probing -----------------------------------------
    l2_probe_interval_s: float = 0.0     # >0 probes /orders/market/impact ladder
    l2_probe_ladder: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class LiveEngine:
    def __init__(
        self,
        client: PrimeXBTClient,
        predictor: Predictor,
        risk: RiskEngine,
        router: OrderRouter,
        strategy: ScalpingStrategy,
        config: LiveConfig,
        feature_config: FeatureConfig | None = None,
        metrics: Metrics | None = None,
        clock=time.time,
        quote_recorder=None,
    ) -> None:
        self.client = client
        self.predictor = predictor
        self.risk = risk
        self.router = router
        self.strategy = strategy
        self.config = config
        self.features = FeatureEngineer(feature_config or FeatureConfig())
        self.metrics = metrics or Metrics()
        self.audit = JsonlLogger(config.log_path)
        self._clock = clock
        self.quote_recorder = quote_recorder

        self.settings: dict[str, SymbolSettings] = {}
        self.buffers: dict[str, pd.DataFrame] = {}
        self.context_buffers: dict[str, pd.DataFrame] = {}
        self.htf_buffers: dict[tuple[str, str], pd.DataFrame] = {}
        self.htf_context_buffers: dict[tuple[str, str], pd.DataFrame] = {}
        self._last_decision_time: dict[str, int] = {}
        self._last_l2_probe: dict[str, float] = {}
        self._known_positions: dict[int | str, Position] = {}
        self._seen_fill_ids: set[int] = set()
        self._position_hold_mult: dict[int | str, float] = {}
        self._last_trade_s = 0.0
        self._last_loss_s = 0.0
        self._last_settings_refresh = 0.0
        self._running = False
        self._basket = BasketTarget(
            config.close_all_on_profit,
            cooldown_s=config.close_all_cooldown_s,
            clock=self._clock,
        )
        self._consecutive_errors = 0

    def _context_symbols(self) -> list[str]:
        if getattr(self.features, "_context_cfg", None) is not None:
            return list(self.features._context_cfg.symbols.keys())
        return []

    def _htf_context_symbols(self) -> list[str]:
        syms = self._context_symbols()
        if syms:
            return syms
        mtf_cfg = getattr(self.features, "_mtf_cfg", None)
        if mtf_cfg is not None and getattr(mtf_cfg, "context_symbols", None):
            return list(mtf_cfg.context_symbols.keys())
        return []

    def _htf_bar_sizes(self) -> list[str]:
        mtf_cfg = getattr(self.features, "_mtf_cfg", None)
        if mtf_cfg is None:
            return []
        ltf_ms = bar_ms(self.config.bar_size)
        return [str(tf).upper() for tf in mtf_cfg.htf_bar_sizes if bar_ms(tf) > ltf_ms]

    def _htf_need_bars(self, htf_bar_size: str) -> int:
        mtf_cfg = getattr(self.features, "_mtf_cfg", None)
        tf_up = str(htf_bar_size).upper()
        z_w = int(getattr(mtf_cfg, "zscore_by_tf", {}).get(tf_up, 96)) if mtf_cfg else 96
        rv_w = int(getattr(mtf_cfg, "rv_norm_by_tf", {}).get(tf_up, 48)) if mtf_cfg else 48
        ltf_span_ms = max(self.config.warmup_bars, 1500) * bar_ms(self.config.bar_size)
        span_htf_bars = int(ltf_span_ms // max(bar_ms(tf_up), 60_000)) + 64
        return max(500, z_w + rv_w + span_htf_bars)

    @staticmethod
    def _closed_htf_slice(
        frame: pd.DataFrame | None, htf_bar_size: str, ltf_close_ms: int
    ) -> pd.DataFrame | None:
        if frame is None or frame.empty:
            return None
        htf_ms = bar_ms(htf_bar_size)
        closed = frame[frame["time"].astype(np.int64) + htf_ms <= ltf_close_ms]
        return closed.reset_index(drop=True) if not closed.empty else None

    def _load_symbol_history(self, symbol: str, target_bar_size: str | None = None, need_bars: int | None = None) -> pd.DataFrame:
        bs = target_bar_size or self.config.bar_size
        bars = self.client.bars(symbol, bs, limit=500)
        if not bars:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume"])
        collected = list(reversed(bars))
        need = need_bars if need_bars is not None else self.config.warmup_bars
        oldest = collected[0].time
        while len(collected) < need:
            chunk = self.client.bars(
                symbol, bs, limit=500, end_ms=oldest - 1
            )
            if not chunk:
                break
            collected = chunk + collected
            oldest = chunk[0].time
        return _bars_to_frame(collected)

    # ---------------------------------------------------------------- lifecycle
    def prepare(self) -> None:
        """Load metadata, history and warm the model. Never trade before this."""
        log.info("preparing live engine for %s (paper=%s)", self.config.symbols, self.config.paper)
        self.client.sync_clock()
        self._refresh_settings(force=True)
        for symbol in self.config.symbols:
            frame = self._load_symbol_history(symbol)
            self.buffers[symbol] = frame
            if len(frame):
                log.info("%s: %d bars loaded (last %s)", symbol, len(frame),
                         pd.to_datetime(frame["time"].iloc[-1], unit="ms", utc=True))
            for htf_bs in self._htf_bar_sizes():
                hframe = self._load_symbol_history(
                    symbol, target_bar_size=htf_bs, need_bars=self._htf_need_bars(htf_bs)
                )
                self.htf_buffers[(symbol, htf_bs)] = hframe
        for ctx_sym in self._context_symbols():
            cframe = self._load_symbol_history(ctx_sym)
            self.context_buffers[ctx_sym] = cframe
            if len(cframe):
                log.info("context %s: %d bars loaded", ctx_sym, len(cframe))
        for ctx_sym in self._htf_context_symbols():
            for htf_bs in self._htf_bar_sizes():
                hcframe = self._load_symbol_history(
                    ctx_sym, target_bar_size=htf_bs, need_bars=self._htf_need_bars(htf_bs)
                )
                self.htf_context_buffers[(ctx_sym, htf_bs)] = hcframe
        latency = self.predictor.warmup(100)
        self.metrics.set_inference_latency(latency)
        log.info("model warm; last inference %.0f us", latency)
        account = self.client.metrics()
        try:
            init_pos = self.client.positions()
            self._known_positions = {p.id: p for p in init_pos}
        except Exception:  # noqa: BLE001
            self._known_positions = {}
        if hasattr(self.client, "fills"):
            try:
                self._seen_fill_ids = {f.id for f in self.client.fills(limit=50)}
            except Exception:  # noqa: BLE001
                self._seen_fill_ids = set()
        log.info(
            "account %s | %s | equity %.2f | positions %d | %s mode",
            account.account_id, "LIVE" if account.real_money else "DEMO",
            account.equity, account.open_positions, account.position_mode,
        )
        self.audit.write("engine_start", config=self.config.to_dict(), account=account.account_id)

    def run(self, max_bars: int | None = None) -> None:
        self._running = True
        processed = 0
        last_bar_time: dict[str, int] = {}
        log.info("live loop started (press Ctrl-C to stop)")

        while self._running:
            try:
                cycle_start = time.perf_counter()
                if self.risk.kill.engaged():
                    log.critical("kill switch engaged -> flattening and exiting")
                    self._emergency_flatten()
                    break

                now = self._clock()
                bar = bar_ms(self.config.bar_size)
                current_bar = int(now * 1000) // bar * bar

                account = self.client.metrics()
                self.risk.on_account(account.equity)
                self.metrics.set_equity(account.equity)
                positions = self.client.positions()
                self._reconcile_closed_positions(positions)

                if now - self._last_settings_refresh >= self.config.refresh_settings_s:
                    self._refresh_settings(force=True)

                # Basket target is checked every cycle, not per bar: profit can
                # peak between bars and giving it back while waiting for a bar
                # close is exactly what this rule exists to prevent.
                if self._check_basket_target(positions):
                    positions = self.client.positions()

                for symbol in self.config.symbols:
                    if last_bar_time.get(symbol) == current_bar:
                        continue
                    if int(now * 1000) - current_bar < self.config.settle_ms:
                        continue  # bar just opened; wait for the feed to settle
                    last_bar_time[symbol] = current_bar
                    self._process_symbol(symbol, account, positions)
                    processed += 1

                self.metrics.observe_cycle((time.perf_counter() - cycle_start) * 1000)
                self._consecutive_errors = 0
                if max_bars is not None and processed >= max_bars:
                    log.info("max_bars=%d reached, stopping", max_bars)
                    break
                time.sleep(self.config.loop_sleep_s)
            except KeyboardInterrupt:
                log.warning("interrupted by user")
                break
            except ApiError as exc:
                self._consecutive_errors += 1
                log.error("API error in loop (%d consecutive): %s", self._consecutive_errors, exc)
                if exc.is_auth:
                    log.warning("auth failure -> re-syncing clock; if this repeats, rotate the key")
                    try:
                        self.client.sync_clock()
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(min(2 ** self._consecutive_errors, 30))
            except Exception as exc:  # noqa: BLE001
                self._consecutive_errors += 1
                log.exception("unexpected loop error: %s", exc)
                time.sleep(min(2 ** self._consecutive_errors, 30))

        self.audit.write("engine_stop", processed_bars=processed)
        log.info("live loop stopped after %d decisions", processed)

    def stop(self) -> None:
        self._running = False

    # ----------------------------------------------------------------- one bar
    def _process_symbol(self, symbol: str, account, positions: list[Position]) -> None:
        bar = bar_ms(self.config.bar_size)

        # ---- refresh the newest bars (the closed bar may not be in the buffer yet
        fresh = self.client.bars(symbol, self.config.bar_size, limit=2)
        if fresh:
            frame = _bars_to_frame(fresh)
            self.buffers[symbol] = (
                pd.concat([self.buffers[symbol], frame], ignore_index=True)
                .drop_duplicates("time", keep="last")
                .sort_values("time")
                .reset_index(drop=True)
                .tail(self.config.warmup_bars + 200)
            )

        quotes = self.client.quotes([symbol])
        if not quotes:
            log.warning("%s: no quote", symbol)
            return
        quote = quotes[0]
        if self.quote_recorder is not None:
            try:
                book_snap = None
                now_s = self._clock()
                if (
                    self.config.l2_probe_interval_s > 0
                    and now_s - self._last_l2_probe.get(symbol, 0.0) >= self.config.l2_probe_interval_s
                ):
                    from ..exchange.orderbook import probe_synthetic_orderbook

                    book_snap = probe_synthetic_orderbook(
                        self.client, symbol, size_ladder=self.config.l2_probe_ladder, quote=quote
                    )
                    self._last_l2_probe[symbol] = now_s
                self.quote_recorder.record_quote(quote, book=book_snap)
            except Exception as exc:  # noqa: BLE001
                log.debug("quote_recorder failed: %s", exc)
        if not quote.market_open:
            log.info("%s: market closed", symbol)
            return
        mid = quote.price or (quote.bid + quote.ask) / 2
        spread_bps = (quote.ask - quote.bid) / max(mid, 1e-12) * 1e4
        self.metrics.set_spread(symbol, spread_bps)

        settings = self.settings.get(symbol)
        if settings is None:
            log.warning("%s: missing settings, skipping", symbol)
            return

        # ---- position management first: exits outrank entries --------------
        symbol_positions = [p for p in positions if p.symbol == symbol]
        atr = self._atr(symbol)
        closed_any = False
        for pos in symbol_positions:
            if self._manage_position(pos, mid, atr, symbol):
                closed_any = True
        if closed_any:
            return  # closed at least one position this bar; do not re-enter on the same bar

        # ---- refresh context buffers when cross-asset features are active --
        for ctx_sym in self._context_symbols():
            cfresh = self.client.bars(ctx_sym, self.config.bar_size, limit=2)
            if cfresh:
                cframe = _bars_to_frame(cfresh)
                prev_ctx = self.context_buffers.get(ctx_sym)
                merged_ctx = (
                    pd.concat([prev_ctx, cframe], ignore_index=True)
                    if prev_ctx is not None and not prev_ctx.empty
                    else cframe
                )
                self.context_buffers[ctx_sym] = (
                    merged_ctx.drop_duplicates("time", keep="last")
                    .sort_values("time")
                    .reset_index(drop=True)
                    .tail(self.config.warmup_bars + 200)
                )

        # ---- refresh causal higher-timeframe buffers for Hybrid Intraday Bias
        now_ms = int(self._clock() * 1000)
        ltf_buf = self.buffers[symbol]
        closed_ltf = ltf_buf[ltf_buf["time"] + bar <= now_ms]
        if len(closed_ltf) >= self.config.lookback + 40:
            ltf_buf = closed_ltf.reset_index(drop=True)
        ltf_close_ms = int(ltf_buf["time"].iloc[-1]) + bar
        ctx_arg: dict[str, pd.DataFrame] | None = None
        if self.context_buffers:
            ctx_arg = {}
            for csym, cdf in self.context_buffers.items():
                closed_c = cdf[cdf["time"] + bar <= now_ms]
                ctx_arg[csym] = (
                    closed_c.reset_index(drop=True)
                    if len(closed_c) >= self.config.lookback + 40
                    else cdf
                )
        htf_bars_arg: dict[str, pd.DataFrame] | None = None
        htf_ctx_arg: dict[str, dict[str, pd.DataFrame]] | None = None
        active_htfs = self._htf_bar_sizes()
        if active_htfs:
            htf_bars_arg = {}
            htf_ctx_arg = {}
            for htf_bs in active_htfs:
                hfresh = self.client.bars(symbol, htf_bs, limit=2)
                if hfresh:
                    hframe = _bars_to_frame(hfresh)
                    prev_h = self.htf_buffers.get((symbol, htf_bs))
                    merged_h = (
                        pd.concat([prev_h, hframe], ignore_index=True)
                        if prev_h is not None and not prev_h.empty
                        else hframe
                    )
                    self.htf_buffers[(symbol, htf_bs)] = (
                        merged_h.drop_duplicates("time", keep="last")
                        .sort_values("time")
                        .reset_index(drop=True)
                        .tail(self._htf_need_bars(htf_bs) + 200)
                    )
                closed_h = self._closed_htf_slice(
                    self.htf_buffers.get((symbol, htf_bs)), htf_bs, ltf_close_ms
                )
                if closed_h is not None:
                    htf_bars_arg[htf_bs] = closed_h

                cmap_closed: dict[str, pd.DataFrame] = {}
                for ctx_sym in self._htf_context_symbols():
                    hcfresh = self.client.bars(ctx_sym, htf_bs, limit=2)
                    if hcfresh:
                        hcframe = _bars_to_frame(hcfresh)
                        prev_hc = self.htf_context_buffers.get((ctx_sym, htf_bs))
                        merged_hc = (
                            pd.concat([prev_hc, hcframe], ignore_index=True)
                            if prev_hc is not None and not prev_hc.empty
                            else hcframe
                        )
                        self.htf_context_buffers[(ctx_sym, htf_bs)] = (
                            merged_hc.drop_duplicates("time", keep="last")
                            .sort_values("time")
                            .reset_index(drop=True)
                            .tail(self._htf_need_bars(htf_bs) + 200)
                        )
                    closed_hc = self._closed_htf_slice(
                        self.htf_context_buffers.get((ctx_sym, htf_bs)), htf_bs, ltf_close_ms
                    )
                    if closed_hc is not None:
                        cmap_closed[ctx_sym] = closed_hc
                if cmap_closed:
                    htf_ctx_arg[htf_bs] = cmap_closed

        # ---- features ------------------------------------------------------
        window = self.features.last_window(
            ltf_buf,
            self.config.lookback,
            context=ctx_arg,
            htf_bars=htf_bars_arg or None,
            htf_context=htf_ctx_arg or None,
        )
        if window is None:
            log.warning("%s: not enough history for a window yet", symbol)
            return
        prediction = self.predictor.predict(window)
        self.metrics.set_inference_latency(prediction.latency_us)

        rl_obs = None
        if self.strategy.rl_policy is not None:
            flat = window[0, :, -1]
            pos_state = np.zeros(7, dtype=np.float32)
            if symbol_positions:
                p = symbol_positions[0]
                pos_state[0] = 1.0 if p.side == "BUY" else -1.0
                pos_state[1] = p.upl / max(account.equity, 1e-9)
            rl_obs = np.concatenate([flat, pos_state]).astype(np.float32)

        decision = self.strategy.decide(
            window,
            rl_observation=rl_obs,
            position=1.0 if symbol_positions else 0.0,
            now_s=self._clock(),
            last_trade_s=self._last_trade_s,
            last_loss_s=self._last_loss_s,
        )

        # ---- risk ----------------------------------------------------------
        verdict = self.risk.evaluate(
            symbol=symbol,
            direction=decision.intent.direction,
            confidence=decision.confidence,
            edge=decision.edge,
            price=mid,
            equity=account.equity,
            atr=atr,
            spread_bps=spread_bps,
            market_open=quote.market_open,
            min_qty=settings.min_qty,
            qty_step=settings.qty_step,
            price_step=settings.min_price_step,
            # Pass the *notional* of every open position: the risk engine sums
            # these against the exposure ceiling. Sending only symbol/side/qty
            # made every position look free, so each new order could reuse the
            # full ceiling -- harmless with one position, 10x leverage with ten.
            open_positions=[
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "qty": p.qty,
                    "mark_price": p.mark_price or p.current_price or p.open_price,
                    "exposure": p.exposure,
                    "open_time": p.open_time,
                }
                for p in positions
            ],
            gross_exposure=sum((p.exposure or 0.0) for p in positions),
            # Pass the clock explicitly: the macro blackout and the cooldown
            # compare against wall time, and defaulting to None silently used a
            # different time source than the rest of the loop.
            now_ms=now_ms,
        )

        record = {
            "symbol": symbol,
            "bar_time": int(ltf_buf["time"].iloc[-1]),
            "mid": mid,
            "spread_bps": spread_bps,
            "atr": atr,
            "prediction": {
                "direction": prediction.direction,
                "confidence": prediction.confidence,
                "edge": prediction.edge,
                "probs": prediction.probs.tolist(),
                "latency_us": prediction.latency_us,
            },
            "decision": decision.to_dict(),
            "risk": verdict.to_dict(),
        }

        if not decision.is_trade or not verdict.allowed:
            self.audit.write("no_trade", **record)
            log.info(
                "%s | %s | %s", symbol,
                decision.intent.value if decision.is_trade else "FLAT",
                verdict.reason,
            )
            return

        # ---- execute -------------------------------------------------------
        qty = settings.floor_qty(verdict.qty * decision.size_multiplier)
        if qty < settings.min_qty:
            log.info("%s: scaled qty %.8f below min, skipping", symbol, qty)
            self.audit.write("no_trade", reason="qty below min after scaling", **record)
            return

        # Enforce allow_hedging=False in HEDGE/NETTING accounts: close any
        # opposite-side open positions before placing the new entry, fulfilling
        # RiskEngine's "reverse handled by execution layer" contract.
        if symbol_positions and not self.risk.config.allow_hedging:
            want_buy = decision.intent.direction > 0
            opposite_positions = [
                p for p in symbol_positions
                if (want_buy and p.side.upper() == "SELL")
                or (not want_buy and p.side.upper() == "BUY")
            ]
            for opp in opposite_positions:
                close_res = self.router.close_position(opp, reason="reverse before opposite entry")
                self._after_close(opp, close_res)
                if not close_res.ok:
                    log.error(
                        "%s: aborting reversal entry because closing opposite position %s failed: %s",
                        symbol, opp.id, close_res.error,
                    )
                    self.audit.write("trade_failed", reason="opposite close failed", **record)
                    return

        tp_price = verdict.take_profit
        tp_mult = float(decision.meta.get("tp_atr_mult", 1.0)) if decision.meta else 1.0
        if tp_price is not None and abs(tp_mult - 1.0) > 1e-6 and atr > 0:
            tp_dist = self.risk.config.tp_atr * tp_mult * atr
            tp_price = settings.round_price(mid + decision.intent.direction * tp_dist)

        hold_mult = float(decision.meta.get("max_hold_mult", 1.0)) if decision.meta else 1.0
        result = self.router.open_position(
            symbol=symbol,
            intent=decision.intent.value,
            qty=qty,
            stop_loss=verdict.stop_loss,
            take_profit=tp_price,
            reference_price=mid,
            bar_time_ms=int(ltf_buf["time"].iloc[-1]),
            spread_bps=spread_bps,
        )
        record["execution"] = result.to_dict()
        if result.ok:
            self._last_trade_s = self._clock()
            if not result.dry_run:
                try:
                    for p in self.client.positions():
                        if p.id not in self._known_positions and p.symbol == symbol:
                            self._position_hold_mult[p.id] = hold_mult
                        self._known_positions[p.id] = p
                except Exception:  # noqa: BLE001
                    pass
            self.metrics.inc_trades(symbol)
            self.audit.write("trade_opened", **record)
            log.info(
                "OPEN %s %s %.6f @ %.5f sl=%.5f tp=%s (%s) in %.0fms",
                decision.intent.value, symbol, result.qty, result.price,
                result.stop_loss or 0.0, result.take_profit, result.client_order_id,
                result.latency_ms,
            )
        else:
            self.metrics.inc_rejects(symbol)
            self.audit.write("trade_failed", **record)
            log.warning("order failed: %s", result.error)

    # -------------------------------------------------------------- management
    def _manage_position(self, pos: Position, mid: float, atr: float, symbol: str) -> bool:
        """Return True when the position was closed or should be left alone."""
        cfg = self.config
        current_ms = int(self._clock() * 1000)
        held_bars = max(int((current_ms - pos.open_time) / bar_ms(cfg.bar_size)), 0)
        side = 1 if pos.side == "BUY" else -1
        pnl_price = side * (mid - pos.open_price)

        # time stop: a scalp that stops being a scalp is exited at breakeven-ish
        hold_mult = self._position_hold_mult.get(pos.id, 1.0)
        eff_max_hold = max(1, int(round(cfg.max_hold_bars * hold_mult))) if cfg.max_hold_bars else 0
        if eff_max_hold and held_bars >= eff_max_hold:
            result = self.router.close_position(pos, reason=f"time stop {held_bars} bars")
            self._after_close(pos, result)
            return True

        # trailing stop: only ever tightens, never widens
        if cfg.trailing_stop_atr > 0 and atr > 0 and pos.stop_loss_price:
            trail = mid - side * cfg.trailing_stop_atr * atr
            improves = (side > 0 and trail > pos.stop_loss_price) or (
                side < 0 and trail < pos.stop_loss_price
            )
            if improves:
                self.router.tighten_stop(pos, self.settings[symbol].round_price(trail))

        # emergency: unrealised loss beyond 1.5x the intended stop (feed gap / slippage)
        if atr > 0 and pnl_price < -1.5 * self.risk.config.sl_atr * atr:
            log.warning("%s: catastrophic move, closing early (%.5f)", symbol, pnl_price)
            result = self.router.close_position(pos, reason="emergency exit")
            self._after_close(pos, result)
            return True
        return False

    def _after_close(
        self,
        pos: Position,
        result: ExecutionResult,
        pnl_override: float | None = None,
    ) -> None:
        if not result.ok:
            self.audit.write(
                "close_failed",
                position_id=pos.id,
                symbol=pos.symbol,
                error=result.error,
                result=result.to_dict(),
            )
            return
        self._known_positions.pop(pos.id, None)
        self._position_hold_mult.pop(pos.id, None)
        pnl = float(pnl_override if pnl_override is not None else pos.upl)
        self.risk.on_trade_closed(pnl)
        if pnl < 0:
            self._last_loss_s = self._clock()
        self.metrics.inc_closed(pnl)
        self.audit.write(
            "trade_closed", position_id=pos.id, symbol=pos.symbol, pnl=pnl,
            side=pos.side, qty=pos.qty, open_price=pos.open_price,
            current_price=result.price or pos.current_price, result=result.to_dict(),
        )

    def _reconcile_closed_positions(self, current_positions: list[Position]) -> None:
        """Detect positions closed server-side (SL / TP / liquidation / manual) between polls."""
        current_by_id = {p.id: p for p in current_positions}
        vanished = [
            prev for pid, prev in self._known_positions.items() if pid not in current_by_id
        ]
        if vanished:
            fills_by_sym: dict[str, list[Any]] = {}
            if hasattr(self.client, "fills"):
                for sym in {v.symbol for v in vanished}:
                    try:
                        fills_by_sym[sym] = list(self.client.fills(symbol=sym, limit=25))
                    except Exception:  # noqa: BLE001
                        fills_by_sym[sym] = []
            for prev in vanished:
                close_side = "SELL" if prev.side.upper() == "BUY" else "BUY"
                matched_pnl: float | None = None
                exit_px = prev.current_price
                for f in sorted(fills_by_sym.get(prev.symbol, ()), key=lambda x: x.time):
                    if f.id in self._seen_fill_ids:
                        continue
                    if f.side.upper() == close_side and f.time >= prev.open_time:
                        self._seen_fill_ids.add(f.id)
                        matched_pnl = (matched_pnl or 0.0) + float(f.rpl) - float(f.fee)
                        exit_px = float(f.price or exit_px)
                        if abs(f.qty - prev.qty) <= max(prev.qty * 0.05, 1e-9):
                            break
                realized_pnl = matched_pnl if matched_pnl is not None else float(prev.upl)
                syn_res = ExecutionResult(
                    ok=True,
                    intent="EXCHANGE_CLOSE",
                    symbol=prev.symbol,
                    qty=prev.qty,
                    price=exit_px,
                )
                log.info(
                    "detected exchange close for position %s (%s %s) pnl=%.2f",
                    prev.id, prev.side, prev.symbol, realized_pnl,
                )
                self._after_close(prev, syn_res, pnl_override=realized_pnl)
        self._known_positions = current_by_id

    # ------------------------------------------------------- basket target
    def _check_basket_target(self, positions: list[Position]) -> bool:
        """Close every position when the combined PnL reaches the target.

        Returns True when a close-all was issued. Closing is done position by
        position on purpose: the exchange's `close/all` endpoint leaves no
        per-position audit trail, and an audit trail is the only way to explain
        afterwards why a losing leg was closed.
        """
        if not positions or self.config.close_all_on_profit <= 0:
            return False
        verdict = self._basket.evaluate(
            positions, enforce=self.config.close_all_enforce
        )
        if verdict.triggered and not self.config.close_all_enforce:
            self.audit.write("basket_target_report_only", **verdict.to_dict())
            return False
        if not verdict.triggered:
            return False

        log.warning("%s", verdict.reason)
        self.audit.write("basket_target_hit", **verdict.to_dict())
        closed = 0
        for pos in positions:
            result = self.router.close_position(pos, reason="basket profit target")
            self._after_close(pos, result)
            if result.ok:
                closed += 1
            else:
                # Do NOT mark the cycle as done if a close failed: the guard must
                # keep trying, otherwise a failed close silently disarms it.
                log.error("basket close failed for %s: %s", pos.id, result.error)
                return False
        self._basket.mark_fired()
        self.audit.write(
            "basket_target_closed_all", closed=closed, total_pnl=verdict.total_pnl
        )
        return closed > 0

    # ------------------------------------------------------------------ helpers
    def _atr(self, symbol: str) -> float:
        frame = self.buffers[symbol]
        if len(frame) < 20:
            return 0.0
        high, low, close = frame["high"], frame["low"], frame["close"]
        prev = close.shift(1)
        tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
        return float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])

    def _refresh_settings(self, force: bool = False) -> None:
        if not force and self._clock() - self._last_settings_refresh < self.config.refresh_settings_s:
            return
        rows = self.client.settings(symbols=self.config.symbols)
        self.settings = {s.symbol: s for s in rows}
        self.router.settings.update(self.settings)
        self._last_settings_refresh = self._clock()
        for symbol, s in self.settings.items():
            log.info(
                "%s: minQty %s step %s | taker %.3f%% | maxLev %dx | hours %s",
                symbol, s.min_qty, s.qty_step, s.taker_fee_pct, s.max_leverage,
                s.trading_hours or "n/a",
            )
            self.metrics.set_fee(symbol, s.taker_fee_pct)

    def _emergency_flatten(self) -> None:
        try:
            positions = self.client.positions()
            for pos in positions:
                res = self.router.close_position(pos, reason="kill switch")
                self._after_close(pos, res)
        except Exception as exc:  # noqa: BLE001
            log.error("emergency flatten failed: %s", exc)


def _bars_to_frame(bars) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"time": b.time, "open": b.open, "high": b.high, "low": b.low,
             "close": b.close, "volume": b.volume}
            for b in bars
        ]
    )

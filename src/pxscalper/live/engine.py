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
from ..risk.engine import RiskEngine, RiskVerdict
from ..strategy.decision import Decision, Intent, ScalpingStrategy
from ..utils import JsonlLogger, bar_ms, now_ms
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

        self.settings: dict[str, SymbolSettings] = {}
        self.buffers: dict[str, pd.DataFrame] = {}
        self._last_decision_time: dict[str, int] = {}
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

    # ---------------------------------------------------------------- lifecycle
    def prepare(self) -> None:
        """Load metadata, history and warm the model. Never trade before this."""
        log.info("preparing live engine for %s (paper=%s)", self.config.symbols, self.config.paper)
        self.client.sync_clock()
        self._refresh_settings(force=True)
        for symbol in self.config.symbols:
            bars = self.client.bars(symbol, self.config.bar_size, limit=500)
            collected = list(reversed(bars))
            need = self.config.warmup_bars
            oldest = collected[0].time
            while len(collected) < need:
                chunk = self.client.bars(
                    symbol, self.config.bar_size, limit=500, end_ms=oldest - 1
                )
                if not chunk:
                    break
                collected = chunk + collected
                oldest = chunk[0].time
            frame = _bars_to_frame(collected)
            self.buffers[symbol] = frame
            log.info("%s: %d bars loaded (last %s)", symbol, len(frame),
                     pd.to_datetime(frame["time"].iloc[-1], unit="ms", utc=True))
        latency = self.predictor.warmup(100)
        self.metrics.set_inference_latency(latency)
        log.info("model warm; last inference %.0f us", latency)
        account = self.client.metrics()
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
        last_time = int(self.buffers[symbol]["time"].iloc[-1])
        need_next = last_time + bar

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
        for pos in symbol_positions:
            if self._manage_position(pos, mid, atr, symbol):
                return  # closed this bar; do not re-enter on the same bar

        # ---- features ------------------------------------------------------
        window = self.features.last_window(self.buffers[symbol], self.config.lookback)
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
                }
                for p in positions
            ],
            gross_exposure=sum((p.exposure or 0.0) for p in positions),
            # Pass the clock explicitly: the macro blackout and the cooldown
            # compare against wall time, and defaulting to None silently used a
            # different time source than the rest of the loop.
            now_ms=int(self._clock() * 1000),
        )

        record = {
            "symbol": symbol,
            "bar_time": int(self.buffers[symbol]["time"].iloc[-1]),
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
        qty = verdict.qty * decision.size_multiplier
        qty = settings.round_qty(qty)
        if qty < settings.min_qty:
            log.info("%s: scaled qty %.8f below min, skipping", symbol, qty)
            self.audit.write("no_trade", reason="qty below min after scaling", **record)
            return

        result = self.router.open_position(
            symbol=symbol,
            intent=decision.intent.value,
            qty=qty,
            stop_loss=verdict.stop_loss,
            take_profit=verdict.take_profit,
            reference_price=mid,
            bar_time_ms=int(self.buffers[symbol]["time"].iloc[-1]),
        )
        record["execution"] = result.to_dict()
        if result.ok:
            self._last_trade_s = self._clock()
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
        held_bars = max(int((now_ms() - pos.open_time) / bar_ms(cfg.bar_size)), 0)
        side = 1 if pos.side == "BUY" else -1
        pnl_price = side * (mid - pos.open_price)

        # time stop: a scalp that stops being a scalp is exited at breakeven-ish
        if cfg.max_hold_bars and held_bars >= cfg.max_hold_bars:
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

    def _after_close(self, pos: Position, result: ExecutionResult) -> None:
        pnl = pos.upl
        self.risk.on_trade_closed(pnl)
        if pnl < 0:
            self._last_loss_s = self._clock()
        self.metrics.inc_closed(pnl)
        self.audit.write(
            "trade_closed", position_id=pos.id, symbol=pos.symbol, pnl=pnl,
            side=pos.side, qty=pos.qty, open_price=pos.open_price,
            current_price=pos.current_price, result=result.to_dict(),
        )

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
        if not force and time.time() - self._last_settings_refresh < self.config.refresh_settings_s:
            return
        rows = self.client.settings(symbols=self.config.symbols)
        self.settings = {s.symbol: s for s in rows}
        self._last_settings_refresh = time.time()
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
                self.router.close_position(pos, reason="kill switch")
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

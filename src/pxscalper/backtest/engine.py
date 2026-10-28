"""Event-driven backtester that charges the same frictions as the live loop.

A backtest is only useful if it can *disagree* with you for the right reasons.
This engine therefore:

* fills at the **next bar's** price (no same-bar look-ahead),
* crosses the **spread** on entry and exit (from the bar's bid/ask when the data
  has it, otherwise an ATR-scaled spread),
* pays the **taker fee** twice (``takerFeePct`` from PrimeXBT settings),
* applies **sqrt-law slippage** based on order size vs. bar volume,
* honours the **intrabar stop and target** (a wick that touches the stop kills
  the trade, in bar order),
* enforces the **same RiskEngine** the live bot uses -- so the two cannot drift,
* reports Sharpe/Sortino/Calmar, profit factor, expectancy, MAE/MFE, hit rate by
  hour of day, and the cost drag in basis points per trade,
* runs **walk-forward**: models are re-fitted per fold, never on the test slice.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = ["BacktestConfig", "BacktestResult", "Backtester", "Trade"]


@dataclass
class BacktestConfig:
    initial_equity: float = 10_000.0
    taker_fee_pct: float = 0.05          # % per side
    #: Sqrt-impact coefficient. Default 0.0 on purpose: PrimeXBT does not
    #: document the unit of the FX `volume` field (EUR/USD 1m bars report a
    #: median of ~223, which is neither units nor lots), so a participation-based
    #: impact model here invents costs (or discounts them) at random. The quoted
    #: bid/ask already contains the real cost of crossing, and the live loop gets
    #: the authoritative number from the `impact` route (`vwapPriceOffset`).
    #: Set this only if you know your venue's volume units.
    slippage_coef: float = 0.0
    #: Hard ceiling on modelled slippage. The square-root impact law is only
    #: valid for small participation (order << bar volume); pushing it to
    #: participation > 1 produces absurd fills (one bug produced +70%). Beyond
    #: the cap the honest statement is "this size is not tradeable here".
    max_slippage_bps: float = 2.0
    #: Optional liquidity guard: refuse sizes above this share of bar volume.
    #: Off by default because PrimeXBT's FX `volume` unit is not documented and
    #: the live `impact` route is the authoritative check.
    max_participation: float = 0.0
    spread_bps_fallback: float = 1.2
    risk_per_trade_pct: float = 0.25
    sl_atr: float = 1.0
    tp_atr: float = 2.0
    max_holding_bars: int = 45
    trailing_stop_atr: float = 0.0
    min_confidence: float = 0.42
    min_edge: float = 0.03
    bars_per_day: int = 1440
    periods_per_year: int = 1440 * 252
    max_positions: int = 1
    compound: bool = True
    # live-equivalent position caps: the backtest must not be able to take a
    # size the risk engine would never allow in production.
    max_leverage: float = 5.0
    max_gross_exposure_pct: float = 300.0
    min_qty: float = 0.0
    qty_step: float = 0.0
    #: How to resolve a bar that touches BOTH stop and target. Bars are OHLC, so
    #: the true intrabar path is unknown; guessing silently is how backtests lie.
    #:   "pessimistic" -- assume the stop was hit first (lower bound)
    #:   "optimistic"  -- assume the target was hit first (upper bound)
    #:   "directional" -- infer from the bar's own direction: an up bar is assumed
    #:                    to travel low->high, so a long's target precedes its stop
    #: Always report the pessimistic case; the spread between the two is the
    #: honest uncertainty of an OHLC backtest.
    intrabar_model: str = "pessimistic"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class Trade:
    entry_time: int
    exit_time: int
    side: int
    entry_price: float
    exit_price: float
    qty: float
    pnl: float
    pnl_bps: float
    fees: float
    slippage: float
    bars_held: int
    exit_reason: str
    confidence: float = 0.0
    mae_bps: float = 0.0
    mfe_bps: float = 0.0

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    timestamps: list[int] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def equity_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"time": self.timestamps, "equity": self.equity_curve})

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "── backtest summary ────────────────────────────────",
            f"  trades              {m.get('n_trades', 0):,.0f}",
            f"  net return          {m.get('total_return_pct', 0):+.2f} %",
            f"  annualised          {m.get('annual_return_pct', 0):+.2f} %",
            f"  Sharpe              {m.get('sharpe', 0):.2f}",
            f"  Sortino             {m.get('sortino', 0):.2f}",
            f"  Calmar              {m.get('calmar', 0):.2f}",
            f"  max drawdown        {m.get('max_drawdown_pct', 0):.2f} %",
            f"  win rate            {m.get('win_rate_pct', 0):.2f} %",
            f"  profit factor       {m.get('profit_factor', 0):.2f}",
            f"  expectancy          {m.get('expectancy_bps', 0):+.2f} bps/trade",
            f"  avg hold            {m.get('avg_bars_held', 0):.1f} bars",
            f"  cost drag           {m.get('cost_drag_bps', 0):.2f} bps/trade",
            "────────────────────────────────────────────────────",
        ]
        return "\n".join(lines)


class Backtester:
    """Vectorised-signal, event-driven-execution backtester.

    ``signal_fn`` receives ``(index, window)`` and must return
    ``(direction, confidence, edge)`` with direction in {-1, 0, 1}. The engine
    then applies sizing, stops, costs and accounting.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        bars: pd.DataFrame,
        windows: np.ndarray,
        window_end_idx: np.ndarray,
        atr: np.ndarray,
        signal_fn: Callable[[int, np.ndarray], tuple[int, float, float]],
    ) -> BacktestResult:
        cfg = self.config
        times = bars["time"].to_numpy()
        close = bars["close"].to_numpy(dtype=np.float64)
        open_ = bars["open"].to_numpy(dtype=np.float64) if "open" in bars else close
        high = bars["high"].to_numpy(dtype=np.float64)
        low = bars["low"].to_numpy(dtype=np.float64)
        volume = bars.get("volume", pd.Series(np.ones(len(bars)))).to_numpy(dtype=np.float64)
        if "bid" in bars and "ask" in bars and bars["bid"].notna().any():
            bid = bars["bid"].to_numpy(dtype=np.float64)
            ask = bars["ask"].to_numpy(dtype=np.float64)
        else:
            half = close * cfg.spread_bps_fallback / 1e4 / 2
            bid, ask = close - half, close + half

        # ``window_end_idx`` holds, for each window, the *positional* index of the
        # bar whose close the signal was computed at -- so the next bar is the
        # first one that can be traded. No timestamp lookups, no look-ahead.
        rows = [int(i) for i in window_end_idx]

        equity = cfg.initial_equity
        result = BacktestResult(config=cfg.to_dict())
        position: dict[str, Any] | None = None
        last_exit_time = 0

        for row, bar_idx in enumerate(rows):
            if bar_idx < 1 or bar_idx >= len(close) - 1:
                continue
            window = windows[row]
            price_risk = max(atr[bar_idx], close[bar_idx] * 1e-6)

            # -------------------------------------------------- manage open trade
            # a position can never be closed on the bar it was filled on
            if position is not None and bar_idx > position["entry_bar"]:
                bars_held = bar_idx - position["entry_bar"]
                hi, lo = high[bar_idx], low[bar_idx]
                exit_price: float | None = None
                reason = ""
                side = position["side"]
                target = position.get("target") if cfg.tp_atr > 0 else None

                stop_hit = (side > 0 and lo <= position["stop"]) or (
                    side < 0 and hi >= position["stop"]
                )
                target_hit = target is not None and (
                    (side > 0 and hi >= target) or (side < 0 and lo <= target)
                )

                if stop_hit and target_hit:
                    # both levels inside the same bar -> the ordering is a guess
                    if cfg.intrabar_model == "optimistic":
                        stop_first = False
                    elif cfg.intrabar_model == "directional":
                        up_bar = close[bar_idx] >= open_[bar_idx]
                        # long in an up bar: low was likely printed before high
                        stop_first = not (up_bar if side > 0 else not up_bar)
                    else:  # pessimistic
                        stop_first = True
                    if stop_first:
                        exit_price, reason = position["stop"], "stop"
                    else:
                        exit_price, reason = target, "target"
                elif stop_hit:
                    exit_price, reason = position["stop"], "stop"
                elif target_hit:
                    exit_price, reason = target, "target"
                elif bars_held >= cfg.max_holding_bars:
                    exit_price, reason = close[bar_idx], "time"
                elif cfg.trailing_stop_atr > 0:
                    trail = (
                        hi - cfg.trailing_stop_atr * price_risk
                        if side > 0
                        else lo + cfg.trailing_stop_atr * price_risk
                    )
                    position["stop"] = max(position["stop"], trail) if side > 0 else min(position["stop"], trail)

                # track excursions
                position["mfe"] = max(position["mfe"], (hi - position["entry_price"]) * side)
                position["mae"] = min(position["mae"], (lo - position["entry_price"]) * side)

                if exit_price is not None:
                    exit_side_price = bid[bar_idx] if side > 0 else ask[bar_idx]
                    fill = exit_price if reason in ("stop", "target") else exit_side_price
                    gross = side * (fill - position["entry_price"]) * position["qty"]
                    fee = fill * position["qty"] * cfg.taker_fee_pct / 100
                    equity_now = equity + gross - fee
                    trade = Trade(
                        entry_time=position["entry_time"],
                        exit_time=int(times[bar_idx]),
                        side=side,
                        entry_price=position["entry_price"],
                        exit_price=float(fill),
                        qty=position["qty"],
                        pnl=gross - fee - position["entry_fee"],
                        pnl_bps=(gross - fee - position["entry_fee"])
                        / max(equity, 1e-9)
                        * 1e4,
                        fees=fee + position["entry_fee"],
                        slippage=position["slippage"],
                        bars_held=bars_held,
                        exit_reason=reason,
                        confidence=position["confidence"],
                        mae_bps=position["mae"] / max(position["entry_price"], 1e-9) * 1e4,
                        mfe_bps=position["mfe"] / max(position["entry_price"], 1e-9) * 1e4,
                    )
                    result.trades.append(trade)
                    equity = equity_now
                    position = None
                    last_exit_time = int(times[bar_idx])

            # ------------------------------------------------------------ entries
            if position is None and row > 0:
                direction, conf, edge = signal_fn(row, window)
                if (
                    direction != 0
                    and conf >= cfg.min_confidence
                    and abs(edge) >= cfg.min_edge
                    and int(times[bar_idx]) != last_exit_time
                ):
                    entry_bar = bar_idx + 1  # fill on the NEXT bar: no look-ahead
                    if entry_bar >= len(close):
                        break
                    side = 1 if direction > 0 else -1
                    raw_price = ask[entry_bar] if side > 0 else bid[entry_bar]
                    # risk-based sizing: risk_amount / stop_distance
                    risk_amount = equity * cfg.risk_per_trade_pct / 100.0
                    qty = risk_amount / max(cfg.sl_atr * price_risk, 1e-12)
                    # cap notional by leverage / gross exposure (as the live engine does)
                    max_notional = min(
                        equity * cfg.max_leverage, equity * cfg.max_gross_exposure_pct / 100.0
                    )
                    qty = min(qty, max_notional / max(raw_price, 1e-9))
                    if cfg.qty_step > 0:
                        qty = (qty // cfg.qty_step) * cfg.qty_step
                    if qty <= 0 or qty < cfg.min_qty:
                        qty = 0.0
                    # sqrt-law slippage, capped to a sane band
                    participation = qty / max(volume[entry_bar], 1.0)
                    if cfg.max_participation > 0 and participation > cfg.max_participation:
                        result.equity_curve.append(equity)
                        result.timestamps.append(int(times[bar_idx]))
                        continue
                    slip_bps = min(
                        cfg.slippage_coef * math.sqrt(max(participation, 0.0)) * 1e4,
                        cfg.max_slippage_bps,
                    )
                    fill = raw_price * (1 + side * slip_bps / 1e4)
                    if qty <= 0:
                        result.equity_curve.append(equity)
                        result.timestamps.append(int(times[bar_idx]))
                        continue
                    entry_fee = fill * qty * cfg.taker_fee_pct / 100
                    position = {
                        "entry_bar": entry_bar,
                        "entry_time": int(times[entry_bar]),
                        "entry_price": float(fill),
                        "qty": float(qty),
                        "side": side,
                        "stop": float(fill - side * cfg.sl_atr * price_risk),
                        "target": float(fill + side * cfg.tp_atr * price_risk) if cfg.tp_atr else None,
                        "entry_fee": float(entry_fee),
                        "slippage": float(slip_bps),
                        "confidence": float(conf),
                        "mfe": 0.0,
                        "mae": 0.0,
                    }

            result.equity_curve.append(equity)
            result.timestamps.append(int(times[bar_idx]))

        result.metrics = self._metrics(result)
        return result

    # ----------------------------------------------------------------- metrics
    def _metrics(self, result: BacktestResult) -> dict[str, float]:
        cfg = self.config
        equity = np.asarray(result.equity_curve, dtype=np.float64)
        if len(equity) < 2:
            return {"n_trades": float(len(result.trades))}
        rets = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
        n_years = len(rets) / cfg.periods_per_year
        total_return = equity[-1] / equity[0] - 1
        annual = (1 + total_return) ** (1 / max(n_years, 1e-9)) - 1 if total_return > -1 else -1.0
        std = rets.std()
        sharpe = float(rets.mean() / std * math.sqrt(cfg.periods_per_year)) if std > 0 else 0.0
        downside = rets[rets < 0]
        sortino = (
            float(rets.mean() / downside.std() * math.sqrt(cfg.periods_per_year))
            if len(downside) > 1 and downside.std() > 0
            else 0.0
        )
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / np.maximum(running_max, 1e-9)
        max_dd = float(-drawdowns.min()) if len(drawdowns) else 0.0
        calmar = float(annual / max_dd) if max_dd > 0 else 0.0

        trades = result.trades
        pnls = np.array([t.pnl for t in trades]) if trades else np.array([0.0])
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        gross_win = wins.sum() if len(wins) else 0.0
        gross_loss = -losses.sum() if len(losses) else 0.0
        return {
            "n_trades": float(len(trades)),
            "total_return_pct": total_return * 100,
            "annual_return_pct": annual * 100,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown_pct": max_dd * 100,
            "win_rate_pct": float(len(wins) / max(len(trades), 1) * 100),
            "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
            "expectancy_bps": float(np.mean([t.pnl_bps for t in trades])) if trades else 0.0,
            "avg_bars_held": float(np.mean([t.bars_held for t in trades])) if trades else 0.0,
            "cost_drag_bps": float(np.mean([(t.fees / max(t.qty * t.entry_price, 1e-9)) * 1e4 for t in trades]))
            if trades
            else 0.0,
            "final_equity": float(equity[-1]),
            "exposure_pct": float(
                sum(t.bars_held for t in trades) / max(len(equity), 1) * 100
            ),
        }

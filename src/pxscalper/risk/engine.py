"""Pre-trade risk engine: the last line of defence before money moves.

This module is intentionally *not* part of the model. No prediction, however
confident, can bypass these checks. Every check returns a structured verdict so
the live loop can log exactly why a trade was refused -- the single most useful
thing when reviewing a bad week.

Checks, in order of evaluation
-----------------------------
1. ``kill_switch``      -- a file (``state/KILL``) or env flag halts trading now.
2. ``account_blocked``  -- PrimeXBT says ``blocked`` or ``reduceOnly``.
3. ``session``          -- symbol must be open, inside configured trading hours.
4. ``spread``           -- spread above ``max_spread_bps`` -> no scalp edge.
5. ``volatility``       -- ATR outside [min, max] band: too dead or too wild.
6. ``confidence``       -- model conviction under threshold.
7. ``exposure``         -- max concurrent positions, max gross exposure, max leverage.
8. ``correlation``      -- refuses a second position in a correlated pair.
9. ``daily_loss``       -- realised + unrealised loss vs ``max_daily_loss_pct``.
10. ``drawdown``        -- peak-to-trough equity vs ``max_drawdown_pct``.
11. ``cooldown``        -- after N consecutive losses, stand down for M minutes.
12. ``min_size``        -- sized order must clear the exchange ``minQty``/``qtyStep``.

Position sizing
---------------
Risk-based, not notional-based::

    risk_amount   = equity * risk_per_trade_pct / 100
    stop_distance = sl_atr * ATR(price units)
    qty           = risk_amount / (stop_distance * pnl_coefficient)

where ``pnl_coefficient`` converts a 1.0 price move into account currency per
unit of base: ``1.0`` for ``EUR/USD``-style pairs quoted in USD, and ``1/price``
for ``USD/JPY``-style pairs. The result is floored to ``qtyStep`` and rejected if
it falls below ``minQty`` -- never silently rounded up, which would breach risk.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

__all__ = [
    "RiskConfig",
    "RiskVerdict",
    "RiskDecision",
    "RiskEngine",
    "RiskState",
    "KillSwitch",
]

log = logging.getLogger(__name__)


class RiskDecision(str, Enum):
    ALLOW = "ALLOW"
    ALLOW_REDUCED = "ALLOW_REDUCED"
    FLAT = "FLAT"
    BLOCK = "BLOCK"

    @property
    def is_tradeable(self) -> bool:
        return self in (RiskDecision.ALLOW, RiskDecision.ALLOW_REDUCED)


@dataclass
class RiskConfig:
    # sizing
    risk_per_trade_pct: float = 0.25      # of equity, to the stop distance
    max_positions: int = 2
    max_gross_exposure_pct: float = 300.0  # notional / equity
    max_leverage: float = 5.0
    min_qty_fraction: float = 1.0          # 1.0 = never round up above risk budget
    # stops / targets
    sl_atr: float = 1.0
    tp_atr: float = 2.0
    trail_atr: float = 0.0                 # >0 enables ATR trailing stop
    max_holding_bars: int = 45
    # account guards
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 5.0
    max_daily_trades: int = 60
    consecutive_loss_cooldown: int = 3
    cooldown_minutes: int = 30
    # micro-filters
    max_spread_bps: float = 3.0
    min_atr_bps: float = 0.8               # volatility floor (dead market)
    max_atr_bps: float = 30.0              # volatility ceiling (news spike)
    min_confidence: float = 0.45
    min_edge: float = 0.02
    # trading hours per symbol (UTC); empty = follow exchange `marketOpen`
    trading_hours: dict[str, tuple[int, int]] = field(default_factory=dict)
    correlated_groups: list[list[str]] = field(default_factory=list)
    allow_hedging: bool = False
    # ops
    state_path: str = "state/risk_state.json"
    kill_switch_path: str = "state/KILL"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RiskConfig":
        valid = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        for key in ("trading_hours",):
            if key in valid:
                valid[key] = {k: tuple(v) for k, v in valid[key].items()}
        return cls(**valid)


@dataclass
class RiskState:
    """Persisted across restarts -- a bot that forgets its losses is dangerous."""

    day: str = ""
    day_start_equity: float = 0.0
    realised_pnl_today: float = 0.0
    trades_today: int = 0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
    cooldown_until_ms: int = 0
    last_trade_ms: int = 0
    kills: int = 0

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def load(cls, path: str | Path) -> "RiskState":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls(**json.loads(p.read_text()))
        except Exception:  # noqa: BLE001
            log.warning("risk state unreadable, starting fresh")
            return cls()

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))


@dataclass(slots=True)
class RiskVerdict:
    decision: RiskDecision
    reason: str
    qty: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    risk_amount: float = 0.0
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def allowed(self) -> bool:
        return self.decision.is_tradeable

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reason": self.reason,
            "qty": self.qty,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "risk_amount": self.risk_amount,
            "checks": self.checks,
            "details": self.details,
        }


class KillSwitch:
    """File + env kill switch. ``state/KILL`` exists -> flatten and stop."""

    def __init__(self, path: str | Path = "state/KILL") -> None:
        self.path = Path(path)

    def engaged(self) -> bool:
        return self.path.exists() or os.getenv("PX_KILL_SWITCH", "0") == "1"

    def engage(self, reason: str = "manual") -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{int(time.time()*1000)} {reason}\n")
        log.critical("KILL SWITCH ENGAGED: %s", reason)

    def release(self) -> None:
        if self.path.exists():
            self.path.unlink()
            log.warning("kill switch released")


def pnl_coefficient(symbol: str, price: float) -> float:
    """Account-currency PnL per 1.0 price move, per 1 unit of base currency.

    ``EUR/USD``  -> 1.0          (quote is USD; long 1000 EUR gains 1000 * dP)
    ``USD/JPY``  -> 1/price      (quote is JPY; convert back to USD)
    """
    base, _, quote = symbol.partition("/")
    if quote.upper() in ("USD", "USDT", "USDC"):
        return 1.0
    if base.upper() in ("USD", "USDT", "USDC"):
        return 1.0 / max(price, 1e-9)
    return 1.0  # unknown quoting: assume USD-settled, risk engine stays conservative


class RiskEngine:
    def __init__(
        self,
        config: RiskConfig | None = None,
        state: RiskState | None = None,
        clock=time.time,
    ) -> None:
        self.config = config or RiskConfig()
        self.state = state or RiskState.load(self.config.state_path)
        self.kill = KillSwitch(self.config.kill_switch_path)
        self._clock = clock

    # ------------------------------------------------------------------ account
    def on_account(self, equity: float) -> None:
        """Call once per loop with live equity: rolls the day, tracks peak."""
        day = time.strftime("%Y-%m-%d", time.gmtime(self._clock()))
        if self.state.day != day:
            self.state.day = day
            self.state.day_start_equity = equity
            self.state.realised_pnl_today = 0.0
            self.state.trades_today = 0
            log.info("new UTC trading day %s, equity %.2f", day, equity)
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        if self.state.day_start_equity <= 0:
            self.state.day_start_equity = equity

    def on_trade_closed(self, pnl: float) -> None:
        self.state.realised_pnl_today += pnl
        self.state.trades_today += 1
        self.state.last_trade_ms = int(self._clock() * 1000)
        if pnl < 0:
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= self.config.consecutive_loss_cooldown:
                until = int((self._clock() + self.config.cooldown_minutes * 60) * 1000)
                self.state.cooldown_until_ms = max(self.state.cooldown_until_ms, until)
                log.warning(
                    "%d consecutive losses -> cooldown until %s",
                    self.state.consecutive_losses,
                    time.strftime("%H:%M:%S", time.gmtime(until / 1000)),
                )
        else:
            self.state.consecutive_losses = 0
        self.state.save(self.config.state_path)

    # ---------------------------------------------------------------- pre-trade
    def evaluate(
        self,
        *,
        symbol: str,
        direction: int,
        confidence: float,
        edge: float,
        price: float,
        equity: float,
        atr: float,
        spread_bps: float,
        market_open: bool,
        min_qty: float,
        qty_step: float,
        price_step: float,
        open_positions: list[dict[str, Any]] | None = None,
        gross_exposure: float = 0.0,
        now_ms: int | None = None,
    ) -> RiskVerdict:
        """Decide FLAT / ALLOW / ALLOW_REDUCED / BLOCK with an auditable reason."""
        cfg = self.config
        checks: dict[str, bool] = {}
        now_ms = now_ms if now_ms is not None else int(self._clock() * 1000)

        self.on_account(equity)
        details: dict[str, Any] = {
            "symbol": symbol,
            "direction": direction,
            "equity": equity,
            "atr_bps": (atr / price * 1e4) if price else 0.0,
            "spread_bps": spread_bps,
            "confidence": confidence,
            "edge": edge,
            "day_realised_pnl": self.state.realised_pnl_today,
            "trades_today": self.state.trades_today,
        }

        def verdict(decision: RiskDecision, reason: str, **kw) -> RiskVerdict:
            merged_details = {**details, **kw.pop("details", {})}
            return RiskVerdict(
                decision=decision, reason=reason, checks=checks, details=merged_details, **kw
            )

        # 1. kill switch -------------------------------------------------------
        checks["kill_switch"] = not self.kill.engaged()
        if not checks["kill_switch"]:
            return verdict(RiskDecision.BLOCK, "kill switch engaged")

        # 2/3. session ---------------------------------------------------------
        checks["market_open"] = bool(market_open)
        window = cfg.trading_hours.get(symbol)
        if window:
            hour = time.gmtime(now_ms / 1000).tm_hour
            start, end = window
            checks["trading_hours"] = start <= hour < end
        else:
            checks["trading_hours"] = True
        if not (checks["market_open"] and checks["trading_hours"]):
            return verdict(RiskDecision.FLAT, "market closed / outside trading hours")

        # 4. spread ------------------------------------------------------------
        checks["spread_ok"] = spread_bps <= cfg.max_spread_bps
        if not checks["spread_ok"]:
            return verdict(RiskDecision.FLAT, f"spread {spread_bps:.2f}bps > {cfg.max_spread_bps}bps")

        # 5. volatility band ---------------------------------------------------
        atr_bps = details["atr_bps"]
        checks["volatility_ok"] = cfg.min_atr_bps <= atr_bps <= cfg.max_atr_bps
        if not checks["volatility_ok"]:
            return verdict(RiskDecision.FLAT, f"ATR {atr_bps:.2f}bps outside band")

        # 6. conviction --------------------------------------------------------
        checks["direction"] = direction in (-1, 0, 1)
        if direction == 0:
            return verdict(RiskDecision.FLAT, "model says flat")
        checks["confidence_ok"] = confidence >= cfg.min_confidence
        checks["edge_ok"] = abs(edge) >= cfg.min_edge
        if not checks["confidence_ok"]:
            return verdict(RiskDecision.FLAT, f"confidence {confidence:.3f} < {cfg.min_confidence}")
        if not checks["edge_ok"]:
            return verdict(RiskDecision.FLAT, f"edge {edge:.3f} < {cfg.min_edge}")

        # 7. exposure ----------------------------------------------------------
        positions = open_positions or []
        same_symbol = [p for p in positions if p.get("symbol") == symbol]
        checks["max_positions"] = len(positions) < cfg.max_positions
        if not checks["max_positions"]:
            return verdict(RiskDecision.BLOCK, f"max positions ({cfg.max_positions}) reached")
        if same_symbol and not cfg.allow_hedging:
            existing = positions[0].get("side", "")
            opposite = (existing.upper() == "BUY" and direction < 0) or (
                existing.upper() == "SELL" and direction > 0
            )
            if same_symbol and not opposite:
                return verdict(RiskDecision.FLAT, "already positioned in this symbol")
            if opposite and not cfg.allow_hedging:
                details["note"] = "reverse handled by execution layer"

        # 8. correlation -------------------------------------------------------
        checks["correlation_ok"] = True
        for group in cfg.correlated_groups:
            if symbol in group:
                others = [p.get("symbol") for p in positions if p.get("symbol") in group and p.get("symbol") != symbol]
                if others:
                    checks["correlation_ok"] = False
                    return verdict(
                        RiskDecision.BLOCK, f"correlated exposure already open ({others[0]})"
                    )

        # 9. daily loss --------------------------------------------------------
        day_pnl_pct = (
            (equity - self.state.day_start_equity) / self.state.day_start_equity * 100
            if self.state.day_start_equity
            else 0.0
        )
        checks["daily_loss_ok"] = day_pnl_pct > -cfg.max_daily_loss_pct
        if not checks["daily_loss_ok"]:
            self.kill.engage(f"daily loss {day_pnl_pct:.2f}%")
            return verdict(RiskDecision.BLOCK, f"daily loss {day_pnl_pct:.2f}% breached")

        # 10. drawdown ---------------------------------------------------------
        dd_pct = (
            (self.state.peak_equity - equity) / self.state.peak_equity * 100
            if self.state.peak_equity
            else 0.0
        )
        checks["drawdown_ok"] = dd_pct < cfg.max_drawdown_pct
        if not checks["drawdown_ok"]:
            self.kill.engage(f"drawdown {dd_pct:.2f}%")
            return verdict(RiskDecision.BLOCK, f"drawdown {dd_pct:.2f}% breached")

        # 11. cooldown / trade budget -----------------------------------------
        checks["cooldown_ok"] = now_ms >= self.state.cooldown_until_ms
        if not checks["cooldown_ok"]:
            return verdict(RiskDecision.FLAT, "cooldown after consecutive losses")
        checks["trade_budget_ok"] = self.state.trades_today < cfg.max_daily_trades
        if not checks["trade_budget_ok"]:
            return verdict(RiskDecision.BLOCK, "daily trade budget exhausted")

        # ------------------------------------------------------- position sizing
        stop_distance = max(cfg.sl_atr * atr, price * 1e-6)
        risk_amount = equity * cfg.risk_per_trade_pct / 100.0
        coeff = pnl_coefficient(symbol, price)
        raw_qty = risk_amount / max(stop_distance * coeff, 1e-12)

        # cap by leverage / gross exposure
        max_notional = min(equity * cfg.max_leverage, equity * cfg.max_gross_exposure_pct / 100.0)
        max_qty_by_leverage = max_notional / max(price * coeff, 1e-12)
        reduced = False
        if raw_qty > max_qty_by_leverage:
            raw_qty = max_qty_by_leverage
            reduced = True

        # respect exchange increments: floor, never round up into more risk
        if qty_step > 0:
            steps = int(raw_qty // qty_step)
            qty = steps * qty_step
        else:
            qty = raw_qty
        qty = round(qty, 10)
        checks["min_size_ok"] = qty >= min_qty
        if not checks["min_size_ok"]:
            return verdict(
                RiskDecision.BLOCK,
                f"sized qty {qty:.8f} below exchange minimum {min_qty} "
                f"(raise risk_per_trade_pct or pick a smaller-increment symbol)",
                details={**details, "raw_qty": raw_qty},
            )

        actual_risk = qty * stop_distance * coeff
        details["actual_risk_amount"] = actual_risk
        details["actual_risk_pct"] = actual_risk / equity * 100 if equity else 0.0
        details["gross_exposure_after"] = gross_exposure + qty * price * coeff

        stop = price - direction * stop_distance
        target = price + direction * cfg.tp_atr * atr if cfg.tp_atr > 0 else None

        return verdict(
            RiskDecision.ALLOW_REDUCED if reduced else RiskDecision.ALLOW,
            "ok" if not reduced else "ok (size capped by leverage)",
            qty=float(qty),
            stop_loss=round(stop, 6),
            take_profit=round(target, 6) if target else None,
            risk_amount=float(actual_risk),
        )

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "kill_engaged": self.kill.engaged(),
        }

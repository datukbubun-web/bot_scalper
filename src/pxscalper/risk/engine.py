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
from typing import Any, ClassVar, get_origin, get_type_hints

__all__ = [
    "RiskConfig",
    "RiskVerdict",
    "RiskDecision",
    "RiskEngine",
    "RiskState",
    "KillSwitch",
]

log = logging.getLogger(__name__)


def _normalise_position(raw: Any) -> dict[str, Any]:
    """Accept dicts from tests or exchange ``Position`` objects from the loop.

    The notional of an open position is what the exposure ceiling must count.
    Without it every position looks free and `max_gross_exposure_pct` is a
    decoration -- which is exactly the bug that let N positions each fill to the
    full ceiling.
    """
    if isinstance(raw, dict):
        out = dict(raw)
    else:
        out = {
            "symbol": getattr(raw, "symbol", ""),
            "side": getattr(raw, "side", ""),
            "qty": getattr(raw, "qty", 0.0),
            "exposure": getattr(raw, "exposure", None),
            "mark_price": getattr(raw, "mark_price", None)
            or getattr(raw, "current_price", None)
            or getattr(raw, "open_price", None),
        }
    if not out.get("notional"):
        exposure = out.get("exposure")
        if exposure:
            out["notional"] = abs(float(exposure))
        else:
            price = out.get("mark_price") or out.get("open_price") or 0.0
            out["notional"] = abs(float(out.get("qty") or 0.0) * float(price or 0.0))
    return out


def _iso(ms: float) -> str:
    """UTC timestamp for log/verdict messages (never used for logic)."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ms / 1000.0))


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
    # Ten open positions in ONE symbol is not diversification, it is the same
    # bet at 10x the size. Cap per symbol as well as in total.
    max_positions_per_symbol: int = 1
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
    # Intraday brake. FX trades 24h, so "daily loss" alone is too coarse: this
    # flattens and pauses after a small intraday drawdown, then resumes when the
    # UTC day rolls. 0 disables it.
    max_intraday_loss_pct: float = 0.75
    max_daily_trades: int = 60
    consecutive_loss_cooldown: int = 3
    cooldown_minutes: int = 30
    # ---- scheduled-event blackout -------------------------------------------
    # Macro releases make a 1-minute bar ~22x wider than normal (measured on
    # XAU/USD: 87.21 bps vs 3.96). A stop sized from pre-event ATR sits inside
    # the noise, so the bot stops opening trades around a release and resumes by
    # itself afterwards. Nothing is closed: this gates ENTRY only.
    blackout_enabled: bool = True
    blackout_pre_min: float = 15.0        # minutes before the release
    blackout_post_min: float = 15.0       # minutes after the release
    blackout_kinds: tuple[str, ...] = ("nfp", "cpi", "fomc")
    # optional explicit override: [(start_ms, end_ms), ...] merged with the calendar
    blackout_windows: list[tuple[int, int]] = field(default_factory=list)

    # ---- lifetime capital guards --------------------------------------------
    # A daily brake is useless if the account dies in a week. For a small account
    # the total drawdown limit is the one that actually protects the capital, so
    # it is checked first and it latches the kill switch.
    max_total_loss_pct: float = 15.0       # from the initial deposit; 0 disables
    # Refuse to trade an account so small that one qty step exceeds the risk
    # budget -- at that size the "risk per trade" setting is a fiction.
    min_equity: float = 0.0

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

    # Resolved once at import: with `from __future__ import annotations` the
    # dataclass field types are strings, so coercion needs the real classes.
    _field_types: ClassVar[dict[str, Any]] = {}

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, allow_env: bool = True) -> "RiskConfig":
        valid = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        for key in ("trading_hours",):
            if key in valid:
                valid[key] = {k: tuple(v) for k, v in valid[key].items()}
        valid = cls._coerce(valid)
        if allow_env:
            valid.update(cls._coerce(cls._env_overrides()))
        return cls(**valid)

    @classmethod
    def _coerce(cls, raw: dict[str, Any]) -> dict[str, Any]:
        """Turn strings ("15", "true", "nfp,cpi") into the declared field types.

        Values arrive as strings from env vars, from some YAML parsers and from
        human typos. Coercing in one place means a typo fails loudly here
        instead of producing a silently wrong guard.
        """
        out = dict(raw)
        for key, value in raw.items():
            if not isinstance(value, str):
                continue
            hint = cls._field_types.get(key)
            origin = get_origin(hint) or hint
            text = value.strip()
            if origin in (int, float) and text:
                out[key] = origin(float(text))
            elif origin is bool:
                out[key] = text.lower() in ("1", "true", "yes", "on")
            elif origin in (tuple, list):
                items = tuple(p.strip() for p in text.split(",") if p.strip())
                out[key] = items if origin is tuple else list(items)
        return out

    @classmethod
    def _env_overrides(cls) -> dict[str, Any]:
        """``PX_RISK_<FIELD>=value`` overrides the YAML, for ops without a deploy.

        Deliberately narrow: only fields that exist on RiskConfig, and every
        value goes through the same coercion as the YAML. An override is logged,
        because a guard that differs from the committed config must not be
        invisible.
        """
        out: dict[str, Any] = {}
        prefix = "PX_RISK_"
        for name, value in os.environ.items():
            if not name.startswith(prefix):
                continue
            key = name[len(prefix):].lower()
            if key not in cls.__dataclass_fields__:
                log.warning("ignoring unknown risk override %s", name)
                continue
            out[key] = value
        if out:
            log.warning("risk config overridden from environment: %s", sorted(out))
        return out


@dataclass
class RiskState:
    """Persisted across restarts -- a bot that forgets its losses is dangerous."""

    day: str = ""
    day_start_equity: float = 0.0
    intraday_peak_equity: float = 0.0
    realised_pnl_today: float = 0.0
    trades_today: int = 0
    peak_equity: float = 0.0
    initial_equity: float = 0.0
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


USD_STABLE = {"USD", "USDT", "USDC", "USDE"}


def pnl_coefficient(symbol: str, price: float, quote_usd_rate: float | None = None) -> float:
    """USD PnL per 1.0 price move, per 1 unit of base currency.

    Three genuinely different cases -- conflating them mis-sizes risk by orders of
    magnitude (an early version of this function priced USD/JPY at 157x too high):

    ``EUR/USD``  quote is USD      -> 1.0
    ``USD/JPY``  base is USD       -> 1/price   (PnL lands in JPY, convert back)
    ``XAU/JPY``  neither is USD    -> quote_usd_rate (e.g. JPY->USD = 1/USDJPY)

    ``quote_usd_rate`` must come from the live catalogue (USD/<quote> price). When
    it is unknown the caller must refuse the trade rather than guess: a guessed
    coefficient silently multiplies the risk budget.
    """
    base, _, quote = symbol.partition("/")
    quote, base = quote.upper(), base.upper()
    if quote in USD_STABLE:
        return 1.0
    if base in USD_STABLE:
        return 1.0 / max(price, 1e-9)
    if quote_usd_rate is not None:
        return float(quote_usd_rate)
    raise ValueError(
        f"{symbol}: PnL coefficient needs the {quote}->USD rate; pass quote_usd_rate "
        f"(the live catalogue provides it). Refusing to guess -- a wrong coefficient "
        f"mis-sizes risk."
    )


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
            self.state.intraday_peak_equity = equity
            self.state.realised_pnl_today = 0.0
            self.state.trades_today = 0
            log.info("new UTC trading day %s, equity %.2f", day, equity)
        if equity > self.state.peak_equity:
            self.state.peak_equity = equity
        if equity > self.state.intraday_peak_equity:
            self.state.intraday_peak_equity = equity
        if self.state.day_start_equity <= 0:
            self.state.day_start_equity = equity
        if self.state.intraday_peak_equity <= 0:
            self.state.intraday_peak_equity = equity
        # The first equity we ever see is the deposit this account was funded
        # with; total-loss limits are measured from here, not from a rolling peak.
        if self.state.initial_equity <= 0:
            self.state.initial_equity = equity

    # -------------------------------------------------------------- blackout
    def blackout_status(self, now_ms: int) -> tuple[bool, str, float]:
        """Is trading blacked out right now, and for how much longer?

        Combines the macro calendar (known months in advance) with any explicit
        windows from the config. Returns ``(blocked, reason, seconds_left)``.
        Empty/disabled config returns ``(False, "", 0.0)``.
        """
        cfg = self.config
        if not cfg.blackout_enabled:
            return False, "", 0.0

        windows: list[tuple[int, int]] = [(int(a), int(b)) for a, b in cfg.blackout_windows]
        if cfg.blackout_kinds:
            try:
                from ..features.calendar import MacroCalendar

                cal = MacroCalendar()
                pre = int(cfg.blackout_pre_min * 60_000)
                post = int(cfg.blackout_post_min * 60_000)
                for kind in cfg.blackout_kinds:
                    for ev in cal.times(kind):
                        windows.append((int(ev) - pre, int(ev) + post))
            except Exception as exc:  # noqa: BLE001 - calendar must never kill the loop
                log.warning("blackout calendar unavailable (%s); using explicit windows only", exc)

        for start, end in windows:
            if start <= now_ms < end:
                left = (end - now_ms) / 1000.0
                return True, f"macro blackout until {_iso(end)}", left
        return False, "", 0.0

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
        quote_usd_rate: float | None = None,
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
        # 0. lifetime capital guard ------------------------------------------
        # Checked before everything: on a small account the total-loss limit is
        # the guard that actually keeps the capital alive, and once breached it
        # latches the kill switch rather than pausing.
        if cfg.max_total_loss_pct > 0 and self.state.initial_equity > 0:
            total_loss_pct = (self.state.initial_equity - equity) / self.state.initial_equity * 100
            details["total_loss_pct"] = total_loss_pct
            details["initial_equity"] = self.state.initial_equity
            checks["total_loss_ok"] = total_loss_pct < cfg.max_total_loss_pct
            if not checks["total_loss_ok"]:
                self.kill.engage(
                    f"total loss {total_loss_pct:.2f}% >= {cfg.max_total_loss_pct}% "
                    f"(initial {self.state.initial_equity:.2f})"
                )
                return verdict(
                    RiskDecision.BLOCK,
                    f"TOTAL LOSS LIMIT: down {total_loss_pct:.2f}% from an initial "
                    f"{self.state.initial_equity:.2f} (limit {cfg.max_total_loss_pct}%) "
                    f"-> kill switch engaged, manual review required",
                )

        # 0b. account minimum -------------------------------------------------
        if cfg.min_equity > 0:
            checks["min_equity_ok"] = equity >= cfg.min_equity
            if not checks["min_equity_ok"]:
                return verdict(
                    RiskDecision.BLOCK,
                    f"equity {equity:.2f} below the configured minimum {cfg.min_equity:.2f}",
                )

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
        # 3b. scheduled-event blackout ---------------------------------------
        if cfg.blackout_enabled:
            clock_ms = int((now_ms if now_ms is not None else self._clock() * 1000))
            blocked, why, seconds_left = self.blackout_status(clock_ms)
            checks["no_blackout"] = not blocked
            details["blackout_seconds_left"] = seconds_left
            if blocked:
                return verdict(
                    RiskDecision.FLAT,
                    f"{why} ({seconds_left / 60:.0f} min left) -> entry paused, "
                    f"resumes automatically",
                )

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
        positions = [_normalise_position(p) for p in (open_positions or [])]
        same_symbol = [p for p in positions if p.get("symbol") == symbol]
        checks["max_positions"] = len(positions) < cfg.max_positions
        if not checks["max_positions"]:
            return verdict(RiskDecision.BLOCK, f"max positions ({cfg.max_positions}) reached")
        checks["max_positions_per_symbol"] = (
            len(same_symbol) < cfg.max_positions_per_symbol
        )
        if not checks["max_positions_per_symbol"]:
            return verdict(
                RiskDecision.BLOCK,
                f"already {len(same_symbol)} position(s) in {symbol}, "
                f"limit is {cfg.max_positions_per_symbol}",
            )
        # A *single* position per symbol is a "one bet only" rule, so it lives
        # with the per-symbol cap above. Here we only forbid the genuinely
        # dangerous combination: piling a same-side bet onto an existing one
        # beyond the cap, or hedging when hedging is off.
        if same_symbol and not cfg.allow_hedging:
            sides = {str(p.get("side", "")).upper() for p in same_symbol}
            opposite = ("BUY" in sides and direction < 0) or ("SELL" in sides and direction > 0)
            if opposite:
                details["note"] = "reverse handled by execution layer"
            elif len(same_symbol) >= cfg.max_positions_per_symbol:
                return verdict(
                    RiskDecision.FLAT,
                    f"already {len(same_symbol)} position(s) in this symbol "
                    f"(cap {cfg.max_positions_per_symbol})",
                )

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

        # 8b. TOTAL exposure across every open position -----------------------
        # `max_gross_exposure_pct` used to cap a single order only: it was never
        # compared against what was already open. With one position that is
        # invisible; with ten it means 10x the intended leverage, and with a
        # small account that is the difference between a bad day and a dead
        # account. Sum first, then decide.
        open_notional = sum(p.get("notional", 0.0) for p in positions)
        ceiling_notional = equity * cfg.max_gross_exposure_pct / 100.0
        details["open_notional"] = open_notional
        details["gross_exposure_ceiling"] = ceiling_notional
        checks["gross_exposure_ok"] = open_notional < ceiling_notional
        if not checks["gross_exposure_ok"]:
            return verdict(
                RiskDecision.BLOCK,
                f"gross exposure ${open_notional:,.0f} already at/above the "
                f"${ceiling_notional:,.0f} ceiling "
                f"({cfg.max_gross_exposure_pct:g}% of equity) across "
                f"{len(positions)} position(s)",
            )

        # 9. intraday brake ----------------------------------------------------
        if cfg.max_intraday_loss_pct > 0 and self.state.intraday_peak_equity > 0:
            intraday_dd = (
                (self.state.intraday_peak_equity - equity)
                / self.state.intraday_peak_equity
                * 100
            )
            checks["intraday_ok"] = intraday_dd < cfg.max_intraday_loss_pct
            details["intraday_drawdown_pct"] = intraday_dd
            if not checks["intraday_ok"]:
                return verdict(
                    RiskDecision.FLAT,
                    f"intraday drawdown {intraday_dd:.2f}% >= "
                    f"{cfg.max_intraday_loss_pct}% -> pause until UTC rollover",
                )

        # 10. daily loss -------------------------------------------------------
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
        try:
            coeff = pnl_coefficient(symbol, price, quote_usd_rate)
        except ValueError as exc:
            checks["pnl_coefficient"] = False
            return verdict(RiskDecision.BLOCK, str(exc))
        raw_qty = risk_amount / max(stop_distance * coeff, 1e-12)

        # cap by leverage / gross exposure, minus whatever is already open:
        # ten positions each sized against the full ceiling is not a limit.
        max_notional = min(
            equity * cfg.max_leverage,
            max(ceiling_notional - open_notional, 0.0),
        )
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

        # one step of the instrument must be affordable within the risk budget
        step_notional = qty_step * price * coeff
        max_notional_room = max_notional
        checks["step_feasible"] = step_notional <= max_notional_room if qty_step > 0 else True
        details["qty_step"] = qty_step
        details["notional_per_step"] = step_notional
        details["max_notional_room"] = max_notional_room
        if not checks["step_feasible"]:
            return verdict(
                RiskDecision.BLOCK,
                f"instrument too coarse: 1 step ({qty_step:g} units) = "
                f"${step_notional:,.0f} notional > ${max_notional_room:,.0f} allowed "
                f"(raise capital/leverage, or trade a finer-grained symbol)",
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


# ---------------------------------------------------------------- type resolver
def _resolve_risk_field_types() -> None:
    hints = get_type_hints(RiskConfig)
    RiskConfig._field_types = {name: hints.get(name) for name in RiskConfig.__dataclass_fields__}


try:  # pragma: no cover - import-time wiring
    _resolve_risk_field_types()
except Exception as _exc:  # noqa: BLE001 - never break import over a convenience
    log.warning("risk config type resolution skipped: %s", _exc)

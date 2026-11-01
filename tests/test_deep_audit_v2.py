"""Regression suite for the deep repository forensic audit (v2).

Covers:
1. Latched daily profit target across post-harvest spread slippage & mid-day restarts.
2. RiskState persistence in on_account() before any trade closes + forward-compatible load().
3. Opposite-side reversal unblocking on max_positions=1 presets + post-reversal risk re-check.
4. OrderRouter strict floor_qty, attach_protection=False, dry_run close/tighten isolation,
   and market-order timeout reconciliation via positions()/fills().
5. Reconciler dry_run safety, allowed_symbols isolation, and tick-rounded emergency stops.
6. LiveEngine symbol-scope isolation, harvest same-bar entry skip, and paper position lifecycle.
7. CLI `pxs trade --mode paper|live` validation.
8. FeatureEngineer.make_windows zero-copy view parity, 2027 MacroCalendar blackout coverage,
   IntradayBiasEngine.evaluate_bars dynamic htf_rv_col, and XAU preset Hybrid consistency.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from pxscalper.backtest.engine import BacktestConfig, Backtester
from pxscalper.cli import app
from pxscalper.data.pipeline import DataPipeline
from pxscalper.exchange.models import (
    AccountMetrics,
    Order,
    OrderCost,
    Position,
    Quote,
    SymbolSettings,
)
from pxscalper.execution.router import ExecutionConfig, OrderRouter
from pxscalper.features.calendar import BUILTIN_SCHEDULE, MacroCalendar, _et
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer
from pxscalper.inference.predictor import Prediction
from pxscalper.live.engine import LiveConfig, LiveEngine
from pxscalper.live.metrics import Metrics
from pxscalper.live.reconcile import Reconciler
from pxscalper.risk.engine import RiskConfig, RiskDecision, RiskEngine, RiskState
from pxscalper.strategy.decision import Decision, Intent, ScalpingStrategy, StrategyConfig
from pxscalper.strategy.intraday_bias import IntradayBiasConfig, IntradayBiasEngine

XAU_SETTINGS = SymbolSettings(
    symbol="XAU/USD",
    min_qty=0.01,
    max_qty=500.0,
    qty_step=0.01,
    min_price_step=0.01,
    price_scale=2,
    max_leverage=100,
    default_leverage=20,
    taker_fee_pct=0.0,
    maker_fee_pct=0.0,
)


def _acc(equity: float = 1000.0, upl: float = 0.0, open_positions: int = 0) -> AccountMetrics:
    return AccountMetrics.from_api(
        {
            "accountId": "acc",
            "currency": "USD",
            "balance": equity - upl,
            "equity": equity,
            "realEquity": equity - upl,
            "upl": upl,
            "availableMargin": equity,
            "openPositions": open_positions,
            "positionMode": "NETTING",
        }
    )


def _pos(
    pid: int = 1,
    symbol: str = "XAU/USD",
    side: str = "BUY",
    qty: float = 0.10,
    open_price: float = 4000.0,
    current_price: float = 4010.0,
    upl: float = 1.0,
    sl: float | None = 3990.0,
    tp: float | None = None,
    open_time: int = 1_780_000_000_000,
) -> Position:
    return Position(
        id=pid,
        symbol=symbol,
        side=side,
        qty=qty,
        open_price=open_price,
        current_price=current_price,
        mark_price=current_price,
        upl=upl,
        roe_pct=0.0,
        liq_price=None,
        margin=20.0,
        client_leverage=20.0,
        stop_loss_price=sl,
        take_profit_price=tp,
        open_time=open_time,
        exposure=abs(qty * current_price),
    )


# ── 1 & 2. Latched daily profit target + RiskState persistence ──────────────
def test_daily_profit_harvest_latches_even_after_spread_slip_and_restarts(tmp_path: Path):
    now_s = [1_780_050_000.0]  # inside one UTC day
    state_file = tmp_path / "risk_state.json"
    cfg = RiskConfig(
        state_path=str(state_file),
        kill_switch_path=str(tmp_path / "KILL"),
        daily_profit_target_pct=3.0,
        max_intraday_loss_pct=0.0,
        blackout_enabled=False,
    )
    risk = RiskEngine(cfg, RiskState.load(state_file), clock=lambda: now_s[0])
    risk.on_account(1000.0)

    # RiskState must already be persisted before any trade closes!
    assert state_file.exists()
    persisted_0 = json.loads(state_file.read_text())
    assert persisted_0["initial_equity"] == pytest.approx(1000.0)
    assert persisted_0["day_start_equity"] == pytest.approx(1000.0)

    class _Client:
        def __init__(self) -> None:
            self.closed_ids: list[int] = []

        def close_position(self, pid: int, qty=None):
            self.closed_ids.append(pid)
            return {"accepted": True}

    class _Pred:
        config = type("C", (), {"n_features": 38, "lookback": 64, "backend": "onnx"})()

    client = _Client()
    router = OrderRouter(client, ExecutionConfig(dry_run=False, verify_fills=False), {"XAU/USD": XAU_SETTINGS})  # type: ignore[arg-type]
    eng = LiveEngine(
        client=client,  # type: ignore[arg-type]
        predictor=_Pred(),  # type: ignore[arg-type]
        risk=risk,
        router=router,
        strategy=ScalpingStrategy(None, StrategyConfig(require_rl_agreement=False)),  # type: ignore[arg-type]
        config=LiveConfig(
            symbols=["XAU/USD"],
            harvest_daily_profit_target=True,
            log_path=str(tmp_path / "decisions.jsonl"),
            state_path=str(tmp_path / "live_positions.json"),
        ),
        metrics=Metrics(enable_server=False),
        clock=lambda: now_s[0],
    )

    # Floating equity touches $1,030.20 (+3.02%), triggering harvest
    open_p = _pos(101, upl=30.20)
    assert eng._check_daily_profit_harvest([open_p], equity=1030.20) is True
    assert client.closed_ids == [101]
    assert risk.state.daily_target_locked is True

    # Market close pays spread/slippage so settled cash equity is $1,029.60 (+2.96% < 3.00%)
    verdict_same_day = risk.evaluate(
        symbol="XAU/USD",
        direction=1,
        confidence=0.65,
        edge=0.15,
        price=4000.0,
        equity=1029.60,
        atr=2.0,
        spread_bps=0.5,
        market_open=True,
        min_qty=0.01,
        qty_step=0.01,
        price_step=0.01,
        now_ms=int(now_s[0] * 1000),
    )
    assert verdict_same_day.decision is RiskDecision.FLAT
    assert verdict_same_day.checks["daily_profit_target_ok"] is False

    # Simulate process restart on the same UTC day + extra unknown schema key in JSON
    raw_json = json.loads(state_file.read_text())
    raw_json["future_extra_field"] = "ignored_safely"
    state_file.write_text(json.dumps(raw_json))

    restarted_risk = RiskEngine(cfg, RiskState.load(state_file), clock=lambda: now_s[0])
    assert restarted_risk.state.daily_target_locked is True
    assert restarted_risk.state.day_start_equity == pytest.approx(1000.0)
    verdict_restarted = restarted_risk.evaluate(
        symbol="XAU/USD",
        direction=1,
        confidence=0.65,
        edge=0.15,
        price=4000.0,
        equity=1029.60,
        atr=2.0,
        spread_bps=0.5,
        market_open=True,
        min_qty=0.01,
        qty_step=0.01,
        price_step=0.01,
        now_ms=int(now_s[0] * 1000),
    )
    assert verdict_restarted.decision is RiskDecision.FLAT

    # Advance past 00:00 UTC -> lock resets automatically and new day starts at $1,029.60
    now_s[0] += 86_400.0
    verdict_next_day = restarted_risk.evaluate(
        symbol="XAU/USD",
        direction=1,
        confidence=0.65,
        edge=0.15,
        price=4000.0,
        equity=1029.60,
        atr=2.0,
        spread_bps=0.5,
        market_open=True,
        min_qty=0.01,
        qty_step=0.01,
        price_step=0.01,
        now_ms=int(now_s[0] * 1000),
    )
    assert verdict_next_day.allowed
    assert restarted_risk.state.daily_target_locked is False
    assert restarted_risk.state.day_start_equity == pytest.approx(1029.60)


def test_backtester_latches_daily_target_after_harvest_with_fees():
    # Single UTC day with 16 hourly bars; first trade hits daily target price,
    # exit fee settles net slightly under 3.0% -- second entry on same day must stay locked.
    n = 16
    day0_ms = 1_780_012_800_000  # 00:00 UTC
    times = day0_ms + np.arange(n, dtype=np.int64) * 3_600_000
    base = 1000.0 + np.arange(n, dtype=np.float64) * 5.0
    df = pd.DataFrame(
        {
            "time": times,
            "open": base,
            "high": base + 20.0,
            "low": base - 0.5,
            "close": base + 5.0,
            "volume": np.full(n, 1000.0),
        }
    )
    windows = np.zeros((n, 2, 4), dtype=np.float32)
    idx = np.arange(n, dtype=np.int64)
    atr = np.full(n, 2.0, dtype=np.float64)

    res = Backtester(
        BacktestConfig(
            initial_equity=1000.0,
            taker_fee_pct=0.05,
            spread_bps_fallback=0.5,
            risk_per_trade_pct=5.0,
            max_leverage=20.0,
            max_gross_exposure_pct=2000.0,
            sl_atr=2.0,
            tp_atr=0.0,
            max_holding_bars=10,
            compound=True,
            daily_profit_target_pct=3.0,
            harvest_daily_profit_target=True,
        )
    ).run(df, windows, idx, atr, lambda r, _w: (1, 0.8, 0.2))
    assert len(res.trades) == 1
    assert res.trades[0].exit_reason == "target"


# ── 3. Reversal unblocking on max_positions=1 + post-reversal risk check ────
def test_single_slot_reversal_is_allowed_while_same_side_blocks(tmp_path: Path):
    risk = RiskEngine(
        RiskConfig(
            state_path=str(tmp_path / "r.json"),
            kill_switch_path=str(tmp_path / "K"),
            max_positions=1,
            max_positions_per_symbol=1,
            allow_hedging=False,
            max_gross_exposure_pct=500.0,
            max_leverage=5.0,
            risk_per_trade_pct=0.5,
            max_intraday_loss_pct=0.0,
            blackout_enabled=False,
        ),
        RiskState(),
    )
    risk.on_account(1000.0)
    open_buy = [_pos(1, side="BUY", qty=1.0, current_price=4500.0)]  # $4,500 notional (90% of $5k cap)

    # Same-side BUY must still be blocked by max_positions=1
    same_side = risk.evaluate(
        symbol="XAU/USD",
        direction=1,
        confidence=0.65,
        edge=0.15,
        price=4500.0,
        equity=1000.0,
        atr=3.0,
        spread_bps=0.5,
        market_open=True,
        min_qty=0.01,
        qty_step=0.01,
        price_step=0.01,
        open_positions=open_buy,
    )
    assert same_side.decision is RiskDecision.BLOCK
    assert "max positions" in same_side.reason

    # Opposite-side SELL must be ALLOWED because BUY will be closed before SELL opens
    rev = risk.evaluate(
        symbol="XAU/USD",
        direction=-1,
        confidence=0.65,
        edge=0.15,
        price=4500.0,
        equity=1000.0,
        atr=3.0,
        spread_bps=0.5,
        market_open=True,
        min_qty=0.01,
        qty_step=0.01,
        price_step=0.01,
        open_positions=open_buy,
    )
    assert rev.allowed
    assert rev.details.get("note") == "reverse handled by execution layer"
    assert rev.details["open_notional"] == pytest.approx(0.0)
    assert rev.details["reversing_notional"] == pytest.approx(4500.0)


def test_live_engine_post_reversal_close_respects_newly_triggered_cooldown(tmp_path: Path):
    now_s = [1_780_050_000.0]

    class _Client:
        def __init__(self) -> None:
            self.pos = [_pos(77, side="BUY", qty=0.10, upl=-12.0, open_time=int(now_s[0] * 1000) - 60_000)]
            self.opened: list[dict] = []
            self.closed: list[int] = []

        def bars(self, symbol, bar_size, limit=2):
            return []

        def quotes(self, symbols):
            return [Quote("XAU/USD", bid=4000.0, ask=4000.2, price=4000.1, timestamp_ms=int(now_s[0] * 1000))]

        def metrics(self):
            return _acc(988.0)

        def positions(self):
            return list(self.pos)

        def close_position(self, pid: int, qty=None):
            self.closed.append(pid)
            self.pos = [p for p in self.pos if p.id != pid]
            return {"accepted": True}

        def market_order_cost(self, symbol, side, qty, tp=None, sl=None):
            return OrderCost("USD", 4000.0, qty * 4000.0, 20.0, 0.0, 0.0, 20.0, None, 0.01, "OK", "INFO", 0.0)

        def place_market(self, symbol, side, qty, tp=None, sl=None, cid=None):
            self.opened.append({"symbol": symbol, "side": side, "qty": qty})
            return {"orderId": 888}

    class _Pred:
        config = type("C", (), {"n_features": 38, "lookback": 64, "backend": "onnx"})()

        def predict(self, window):
            return Prediction(np.array([0.75, 0.10, 0.15], dtype=np.float32), -1, 0.75, 10.0)

    class _ForceShortStrategy(ScalpingStrategy):
        def decide(self, *args, **kwargs):
            return Decision(Intent.SHORT, confidence=0.75, edge=-0.60, reason="reverse short")

    client = _Client()
    risk = RiskEngine(
        RiskConfig(
            state_path=str(tmp_path / "r.json"),
            kill_switch_path=str(tmp_path / "K"),
            max_positions=1,
            max_positions_per_symbol=1,
            allow_hedging=False,
            risk_per_trade_pct=2.0,
            max_leverage=20.0,
            max_gross_exposure_pct=2000.0,
            min_atr_bps=0.05,
            max_atr_bps=500.0,
            consecutive_loss_cooldown=1,  # closing losing BUY immediately triggers cooldown!
            cooldown_minutes=30,
            max_intraday_loss_pct=0.0,
            blackout_enabled=False,
        ),
        RiskState(),
        clock=lambda: now_s[0],
    )
    risk.on_account(1000.0)
    router = OrderRouter(client, ExecutionConfig(dry_run=False, verify_fills=False), {"XAU/USD": XAU_SETTINGS})  # type: ignore[arg-type]
    eng = LiveEngine(
        client=client,  # type: ignore[arg-type]
        predictor=_Pred(),  # type: ignore[arg-type]
        risk=risk,
        router=router,
        strategy=_ForceShortStrategy(None, StrategyConfig(require_rl_agreement=False)),  # type: ignore[arg-type]
        config=LiveConfig(
            symbols=["XAU/USD"],
            bar_size="5M",
            max_hold_bars=99,
            log_path=str(tmp_path / "decisions.jsonl"),
            state_path=str(tmp_path / "live_positions.json"),
        ),
        metrics=Metrics(enable_server=False),
        clock=lambda: now_s[0],
    )
    eng.settings["XAU/USD"] = XAU_SETTINGS
    eng.buffers["XAU/USD"] = DataPipeline.synthetic(symbol="XAU/USD", n_bars=900, bar_size="5M", start_price=4000.0)

    eng._process_symbol("XAU/USD", _acc(1000.0, upl=-12.0, open_positions=1), client.positions())
    assert client.closed == [77], "opposite BUY position was closed"
    assert client.opened == [], "new SHORT entry was vetoed because closing BUY engaged loss cooldown"


# ── 4. OrderRouter floor_qty, attach_protection=False, dry_run & timeout ────
def test_order_router_never_rounds_up_honors_attach_protection_and_dry_run_close():
    class _Client:
        def __init__(self) -> None:
            self.placed: list[dict] = []
            self.closed: list[int] = []
            self.protectives: list[tuple] = []

        def positions(self):
            return []

        def active_orders(self):
            return []

        def market_order_cost(self, symbol, side, qty, take_profit=None, stop_loss=None):
            return OrderCost("USD", 4000.0, qty * 4000.0, 4.0, 0.0, 0.0, 10.0, None, 0.01, "OK", "INFO", 0.0)

        def place_market(self, symbol, side, qty, take_profit=None, stop_loss=None, client_order_id=None):
            self.placed.append({"qty": qty, "sl": stop_loss, "tp": take_profit, "cid": client_order_id})
            return {"orderId": 10}

        def close_position(self, pid, qty=None):
            self.closed.append(pid)
            return {"accepted": True}

        def set_protective(self, pid, kind, price):
            self.protectives.append((pid, kind, price))
            return {"accepted": True}

    c = _Client()
    r_no_prot = OrderRouter(
        c,  # type: ignore[arg-type]
        ExecutionConfig(dry_run=False, verify_fills=False, attach_protection=False),
        {"XAU/USD": XAU_SETTINGS},
    )
    res = r_no_prot.open_position("XAU/USD", "LONG", 0.019, 3990.0, 4020.0, 4000.0)
    assert res.ok
    assert c.placed[0]["qty"] == pytest.approx(0.01)  # floored, NEVER rounded up to 0.02!
    assert c.placed[0]["sl"] is None
    assert c.placed[0]["tp"] is None

    # In dry_run=True, close_position / close_all / tighten_stop must NEVER call live broker writes
    r_dry = OrderRouter(c, ExecutionConfig(dry_run=True), {"XAU/USD": XAU_SETTINGS})  # type: ignore[arg-type]
    p = _pos(55)
    close_res = r_dry.close_position(p, reason="paper test")
    assert close_res.ok and close_res.dry_run
    assert r_dry.tighten_stop(p, 3995.0) is True
    r_dry.close_all()
    assert c.closed == []
    assert c.protectives == []


def test_order_router_reconciles_filled_market_order_after_read_timeout():
    """When place_market times out on read AFTER executing on the server, the
    order is absent from active_orders() and lives in positions()/fills()."""

    class _TimeoutAfterFillClient:
        def __init__(self) -> None:
            self._pos: list[Position] = []

        def positions(self):
            return list(self._pos)

        def active_orders(self):
            return []  # market order filled immediately -> NOT in active_orders!

        def market_order_cost(self, symbol, side, qty, take_profit=None, stop_loss=None):
            return OrderCost("USD", 4005.0, qty * 4005.0, 8.0, 0.0, 0.0, 10.0, None, 0.01, "OK", "INFO", 0.0)

        def place_market(self, symbol, side, qty, take_profit=None, stop_loss=None, client_order_id=None):
            # Server executes the market order, THEN network socket drops before response
            self._pos.append(_pos(777, symbol=symbol, side=side, qty=qty, open_price=4005.25))
            raise TimeoutError("HTTPSConnectionPool read timed out")

    client = _TimeoutAfterFillClient()
    router = OrderRouter(
        client,  # type: ignore[arg-type]
        ExecutionConfig(dry_run=False, verify_fills=True, verify_timeout_s=0.5),
        {"XAU/USD": XAU_SETTINGS},
    )
    res = router.open_position("XAU/USD", "LONG", 0.05, 3995.0, 4025.0, 4005.0)
    assert res.ok, "filled market order must be reconciled as ok=True, preventing duplicate retry"
    assert res.order_id == 777
    assert res.price == pytest.approx(4005.25)
    assert res.qty == pytest.approx(0.05)


# ── 5. Reconciler dry_run, allowed_symbols isolation & rounded emergency SL ──
def test_reconciler_dry_run_and_symbol_isolation_and_tick_rounded_stop(tmp_path: Path):
    class _RecClient:
        def __init__(self) -> None:
            self.protectives: list[tuple] = []
            self.cancelled: list[int] = []

        def positions(self):
            return [
                # Unprotected managed XAU/USD position already underwater (open 4010, current 4002)
                _pos(11, symbol="XAU/USD", side="BUY", open_price=4010.0, current_price=4002.0, sl=None),
                # Unrelated manual BTC/USD position without stop -> MUST NOT BE TOUCHED
                _pos(22, symbol="BTC/USD", side="BUY", open_price=90000.0, current_price=90100.0, sl=None),
            ]

        def active_orders(self):
            return [
                Order(
                    id=101, symbol="XAU/USD", side="BUY", type="LIMIT", status="NEW",
                    qty=0.01, executed_qty=0.0, executed_price=None, trigger_price=3950.0,
                    protective=False, position_id=None, client_order_id=None, placed_at=0,
                ),
                # Manual resting BTC/USD limit order -> MUST NOT BE CANCELLED!
                Order(
                    id=202, symbol="BTC/USD", side="BUY", type="LIMIT", status="NEW",
                    qty=0.01, executed_qty=0.0, executed_price=None, trigger_price=88000.0,
                    protective=False, position_id=None, client_order_id=None, placed_at=0,
                ),
            ]

        def set_protective(self, pid: int, kind: str, price: float):
            self.protectives.append((pid, kind, price))
            return {"accepted": True}

        def cancel_order(self, oid: int):
            self.cancelled.append(oid)
            return {"accepted": True}

    journal_path = tmp_path / "live_positions.json"
    c = _RecClient()

    # 1) dry_run=True -> zero writes to exchange or journal
    rep_dry = Reconciler(
        c,  # type: ignore[arg-type]
        state_path=journal_path,
        atr_by_symbol={"XAU/USD": 3.333333},
        sl_atr=2.0,
        dry_run=True,
        allowed_symbols=["XAU/USD"],
        settings_by_symbol={"XAU/USD": XAU_SETTINGS},
    ).run()
    assert rep_dry.unprotected_positions == [11]
    assert rep_dry.orphan_orders == [101]
    assert c.protectives == []
    assert c.cancelled == []
    assert not journal_path.exists()

    # 2) dry_run=False with allowed_symbols=["XAU/USD"] -> fixes XAU/USD ONLY, rounds stop to 0.01
    rep_live = Reconciler(
        c,  # type: ignore[arg-type]
        state_path=journal_path,
        atr_by_symbol={"XAU/USD": 3.333333},
        sl_atr=2.0,  # open_price - 6.666 = 4003.33 > current_price 4002.0 -> clamps to 4002.0 - 6.666 = 3995.33!
        dry_run=False,
        allowed_symbols=["XAU/USD"],
        settings_by_symbol={"XAU/USD": XAU_SETTINGS},
    ).run()
    assert rep_live.ok
    assert rep_live.stops_attached == [11]
    assert rep_live.cancelled == [101]
    assert c.protectives == [(11, "STOP_LOSS", pytest.approx(3995.33))]
    assert 202 not in c.cancelled, "manual BTC/USD order must NEVER be cancelled"


# ── 6 & 7. Paper lifecycle, symbol isolation in LiveEngine & CLI --mode ─────
def test_paper_mode_tracks_simulated_position_lifecycle_and_leaves_other_symbols_alone(tmp_path: Path):
    now_s = [1_780_050_000.0]
    quote_holder = [Quote("XAU/USD", bid=4000.0, ask=4000.2, price=4000.1, timestamp_ms=int(now_s[0] * 1000))]

    class _PaperClient:
        def __init__(self) -> None:
            # Unrelated real BTC/USD position on account
            self.btc_pos = _pos(999, symbol="BTC/USD", side="BUY", upl=500.0)
            self.closed_ids: list[int] = []

        def bars(self, symbol, bar_size, limit=2):
            return []

        def quotes(self, symbols):
            return [quote_holder[0]]

        def metrics(self):
            return _acc(1000.0, upl=500.0, open_positions=1)

        def positions(self):
            return [self.btc_pos]

        def close_position(self, pid: int, qty=None):
            self.closed_ids.append(pid)
            return {"accepted": True}

        def market_order_cost(self, symbol, side, qty, tp=None, sl=None):
            return OrderCost("USD", quote_holder[0].ask, qty * quote_holder[0].ask, 8.0, 0.0, 0.0, 10.0, None, 0.01, "OK", "INFO", 0.0)

    class _Pred:
        config = type("C", (), {"n_features": 38, "lookback": 64, "backend": "onnx"})()

        def predict(self, window):
            return Prediction(np.array([0.1, 0.1, 0.8], dtype=np.float32), 1, 0.8, 10.0)

    class _OneLongStrategy(ScalpingStrategy):
        def decide(self, *args, **kwargs):
            return Decision(Intent.LONG, confidence=0.80, edge=0.70, reason="paper buy")

    client = _PaperClient()
    risk = RiskEngine(
        RiskConfig(
            state_path=str(tmp_path / "r.json"),
            kill_switch_path=str(tmp_path / "K"),
            risk_per_trade_pct=2.0,
            max_leverage=20.0,
            max_gross_exposure_pct=2000.0,
            min_atr_bps=0.05,
            max_atr_bps=500.0,
            tp_atr=2.0,
            sl_atr=2.0,
            max_intraday_loss_pct=0.0,
            blackout_enabled=False,
        ),
        RiskState(),
        clock=lambda: now_s[0],
    )
    router = OrderRouter(client, ExecutionConfig(dry_run=True), {"XAU/USD": XAU_SETTINGS})  # type: ignore[arg-type]
    eng = LiveEngine(
        client=client,  # type: ignore[arg-type]
        predictor=_Pred(),  # type: ignore[arg-type]
        risk=risk,
        router=router,
        strategy=_OneLongStrategy(None, StrategyConfig(require_rl_agreement=False)),  # type: ignore[arg-type]
        config=LiveConfig(
            symbols=["XAU/USD"],
            bar_size="5M",
            log_path=str(tmp_path / "decisions.jsonl"),
        ),
        metrics=Metrics(enable_server=False),
        clock=lambda: now_s[0],
    )
    eng.settings["XAU/USD"] = XAU_SETTINGS
    eng.buffers["XAU/USD"] = DataPipeline.synthetic(symbol="XAU/USD", n_bars=900, bar_size="5M", start_price=4000.0)

    # Bar 1: opens simulated paper LONG position
    paper_list: list[Position] = []
    eng._process_symbol("XAU/USD", client.metrics(), paper_list)
    assert len(eng._paper_positions) == 1
    paper_pos = next(iter(eng._paper_positions.values()))
    assert paper_pos.take_profit_price is not None

    # Bar 2: price rallies past take_profit_price -> paper TP closes position and books paper PnL
    now_s[0] += 300.0
    tp_hit_px = float(paper_pos.take_profit_price) + 1.0
    quote_holder[0] = Quote("XAU/USD", bid=tp_hit_px, ask=tp_hit_px + 0.2, price=tp_hit_px + 0.1, timestamp_ms=int(now_s[0] * 1000))
    eng._process_symbol("XAU/USD", client.metrics(), list(eng._paper_positions.values()))
    assert len(eng._paper_positions) == 0
    assert risk.state.trades_today == 1
    assert risk.state.realised_pnl_today > 0
    assert client.closed_ids == [], "unrelated BTC/USD live position must NEVER be closed"


def test_cli_trade_mode_flag_guards(monkeypatch):
    captured: dict = {}

    def _fake_start(cfg_path, allow_live, paths=None, max_bars=None):
        captured["allow_live"] = allow_live
        captured["max_bars"] = max_bars

    import pxscalper.ops.live_runner as lr

    monkeypatch.setattr(lr, "start_from_config", _fake_start)
    runner = CliRunner()

    # --mode paper forces allow_live=False even when PX_LIVE_TRADING=1
    monkeypatch.setenv("PX_LIVE_TRADING", "1")
    res_paper = runner.invoke(app, ["trade", "--mode", "paper", "--max-bars", "3"])
    assert res_paper.exit_code == 0
    assert captured["allow_live"] is False
    assert captured["max_bars"] == 3

    # --mode live WITHOUT --i-understand-the-risk must fail hard (exit 2), never silently paper
    res_live_missing = runner.invoke(app, ["trade", "--mode", "live"])
    assert res_live_missing.exit_code == 2

    # --mode live WITH PX_LIVE_TRADING=1 AND --i-understand-the-risk succeeds
    res_live_ok = runner.invoke(app, ["trade", "--mode", "live", "--i-understand-the-risk"])
    assert res_live_ok.exit_code == 0
    assert captured["allow_live"] is True


# ── 8. Zero-copy make_windows, 2027 blackout, dynamic bias & preset check ───
def test_make_windows_zero_copy_view_matches_contiguous_exact():
    bars = DataPipeline.synthetic(symbol="XAU/USD", n_bars=600, bar_size="1M", start_price=4000.0)
    eng = FeatureEngineer(FeatureConfig())
    norm = eng.transform_normalized(bars)
    w_contig, idx_contig = eng.make_windows(norm, lookback=64, contiguous=True)
    w_view, idx_view = eng.make_windows(norm, lookback=64, contiguous=False)
    assert np.array_equal(idx_contig, idx_view)
    assert np.array_equal(w_contig, w_view)
    assert not w_view.flags["C_CONTIGUOUS"] or w_view.base is not None


def test_macro_calendar_covers_2027_and_blackout_fires_in_2027(tmp_path: Path):
    cal = MacroCalendar()
    years = {pd.Timestamp(e.time_ms, unit="ms", tz="UTC").year for e in BUILTIN_SCHEDULE}
    assert {2026, 2027}.issubset(years)
    nfp_2027 = _et(2027, 1, 8, 8, 30)  # First NFP of 2027 (Jan 1 is federal holiday)
    assert nfp_2027 in set(cal.times("nfp"))

    risk = RiskEngine(
        RiskConfig(
            state_path=str(tmp_path / "r.json"),
            kill_switch_path=str(tmp_path / "K"),
            blackout_enabled=True,
            blackout_pre_min=15.0,
            blackout_post_min=15.0,
        ),
        RiskState(),
    )
    blocked, why, left = risk.blackout_status(nfp_2027 - 5 * 60_000)
    assert blocked is True
    assert "macro blackout" in why
    assert left == pytest.approx(20 * 60.0)


def test_intraday_bias_evaluate_bars_honors_configured_htf_rv_col():
    bars_1m = DataPipeline.synthetic(symbol="XAU/USD", n_bars=900, bar_size="1M", start_price=4000.0)
    ctx_1m = DataPipeline.synthetic(symbol="XAG/USD", n_bars=900, bar_size="1M", seed=11, start_price=32.0)
    engine = IntradayBiasEngine(IntradayBiasConfig(htf_bar_size="5M", htf_rv_col="htf_5m_xag_rv_z_2"))
    snap = engine.evaluate_bars(bars_1m, ltf_bar_size="1M", context={"XAG/USD": ctx_1m})
    assert np.isfinite(snap.htf_fv_score)
    assert np.isfinite(snap.bias_score)


@pytest.mark.parametrize(
    "preset",
    [
        "configs/live.hybrid.yaml",
        "configs/live.hybrid.1m.yaml",
        "configs/live.mini100.yaml",
        "configs/live.mini100.hybrid.yaml",
        "configs/live.mini100.multi.yaml",
        "configs/live.mini100.compound.yaml",
        "configs/live.hybrid.compound.yaml",
        "configs/live.hybrid.daily3pct_lock.yaml",
    ],
)
def test_all_xau_presets_use_hybrid_architecture(preset: str):
    raw = yaml.safe_load(Path(preset).read_text())
    assert raw["strategy"]["kind"] == "hybrid", f"{preset} must use Hybrid Architecture"
    assert raw.get("hybrid", {}).get("enabled") is True, f"{preset} must enable hybrid block"


def test_live_engine_run_skips_same_bar_reentry_after_basket_harvest(tmp_path: Path):
    now_s = [1_780_050_010.0]  # 10s after 5M bar open (past settle_ms=1200)
    process_calls: list[str] = []

    class _Client:
        def __init__(self) -> None:
            self.pos = [_pos(301, symbol="XAU/USD", side="BUY", upl=35.0)]
            self.closed: list[int] = []

        def metrics(self):
            return _acc(1035.0, upl=35.0, open_positions=len(self.pos))

        def positions(self):
            return list(self.pos)

        def close_position(self, pid: int, qty=None):
            self.closed.append(pid)
            self.pos = [p for p in self.pos if p.id != pid]
            return {"accepted": True}

        def settings(self, symbols=None):
            return [XAU_SETTINGS]

    class _Pred:
        config = type("C", (), {"n_features": 38, "lookback": 64, "backend": "onnx"})()

    client = _Client()
    risk = RiskEngine(
        RiskConfig(state_path=str(tmp_path / "r.json"), kill_switch_path=str(tmp_path / "K")),
        RiskState(),
        clock=lambda: now_s[0],
    )
    router = OrderRouter(client, ExecutionConfig(dry_run=False, verify_fills=False), {"XAU/USD": XAU_SETTINGS})  # type: ignore[arg-type]
    eng = LiveEngine(
        client=client,  # type: ignore[arg-type]
        predictor=_Pred(),  # type: ignore[arg-type]
        risk=risk,
        router=router,
        strategy=ScalpingStrategy(None, StrategyConfig(require_rl_agreement=False)),  # type: ignore[arg-type]
        config=LiveConfig(
            symbols=["XAU/USD"],
            bar_size="5M",
            loop_sleep_s=0.0,
            close_all_on_profit_pct=3.0,
            close_all_enforce=True,
            log_path=str(tmp_path / "decisions.jsonl"),
            state_path=str(tmp_path / "live_positions.json"),
        ),
        metrics=Metrics(enable_server=False),
        clock=lambda: now_s[0],
    )
    eng._process_symbol = lambda sym, acc, pos: process_calls.append(sym)  # type: ignore[method-assign]
    eng.run(max_bars=1)

    assert client.closed == [301], "basket target closed the winning position"
    assert process_calls == [], "must NOT immediately re-open on the exact same bar right after basket harvest"
    assert eng._last_trade_s == pytest.approx(now_s[0])


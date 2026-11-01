"""Regression tests covering all 10 audited infrastructure, execution, risk,
feature, backtest, strategy, trainer, and calendar bugs."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pxscalper.backtest.engine import BacktestConfig, Backtester
from pxscalper.data.pipeline import DataPipeline
from pxscalper.exchange.catalogue import Instrument
from pxscalper.exchange.models import OrderCost, Position, Quote, SymbolSettings
from pxscalper.execution.router import ExecutionConfig, ExecutionResult, OrderRouter
from pxscalper.features.calendar import CalendarConfig, calendar_features
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer
from pxscalper.features.targets import BarrierConfig, barrier_y_to_dir, directed_barrier_labels
from pxscalper.inference.predictor import Calibration, Prediction, PredictorConfig, load_predictor
from pxscalper.live.engine import LiveConfig, LiveEngine
from pxscalper.models.sequence import ModelConfig, TCN
from pxscalper.models.trainer import TrainConfig, Trainer, WindowDataset, class_weights
from pxscalper.ops.backtest_runner import run_backtest_from_config
from pxscalper.risk.engine import RiskConfig, RiskEngine, RiskState
from pxscalper.strategy.decision import Intent, ScalpingStrategy, StrategyConfig


def _make_settings(symbol: str = "XAU/USD", min_qty: float = 0.01, max_qty: float = 100.0) -> SymbolSettings:
    return SymbolSettings(
        symbol=symbol,
        min_qty=min_qty,
        max_qty=max_qty,
        qty_step=0.01,
        min_price_step=0.01,
        price_scale=2,
        max_leverage=20,
        default_leverage=10,
        taker_fee_pct=0.0,
        maker_fee_pct=0.0,
    )


def _make_cost(price: float = 4000.0) -> OrderCost:
    return OrderCost(
        currency="USD",
        price=price,
        order_cost=price * 0.01,
        margin_impact=4.0,
        fee=0.0,
        fee_pct=0.0,
        leverage=10.0,
        liq_price=3600.0,
        min_order_size=0.01,
        resolution="OK",
        resolution_level="INFO",
        vwap_price_offset=0.0,
    )


def _make_position(pid: int, qty: float = 0.01, open_price: float = 4000.0, open_time: int = 0) -> Position:
    return Position(
        id=pid,
        symbol="XAU/USD",
        side="BUY",
        qty=qty,
        open_price=open_price,
        current_price=open_price,
        mark_price=open_price,
        upl=0.0,
        roe_pct=0.0,
        liq_price=None,
        margin=4.0,
        client_leverage=10.0,
        stop_loss_price=None,
        take_profit_price=None,
        open_time=open_time,
    )


# ── Bug #1: Manifest portability & cross-symbol model/calibration guards ────
def test_manifest_resolves_stale_foreign_paths_and_blocks_wrong_symbol(tmp_path):
    src_dir = Path("artifacts/models/latest")
    for fn in ("tcn.onnx", "tcn.ts", "tcn.manifest.json", "calibration.json"):
        if (src_dir / fn).exists():
            (tmp_path / fn).write_bytes((src_dir / fn).read_bytes())

    manifest = json.loads((tmp_path / "tcn.manifest.json").read_text())
    manifest["onnx_path"] = "/Users/foreign_machine/nonexistent/tcn.onnx"
    manifest["torchscript_path"] = "/Users/foreign_machine/nonexistent/tcn.ts"
    manifest["symbol"] = "EUR/USD"
    (tmp_path / "tcn.manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="trained for EUR/USD"):
        PredictorConfig.from_manifest(tmp_path / "tcn.manifest.json", expected_symbol="XAU/USD")

    pred = load_predictor(tmp_path, arch="tcn", expected_symbol="EUR/USD")
    assert Path(pred.config.onnx_path).exists()

    cal_path = tmp_path / "foreign_cal.json"
    Calibration(temperature=3.5, symbol="EUR/USD", arch="tcn").save(cal_path)
    loaded = Calibration.load(cal_path, expected_symbol="XAU/USD")
    assert loaded.temperature == pytest.approx(1.0)


# ── Bug #2: FeatureEngineer ret_1 non-NaN & strict causal microstructure/RV ─
def test_ret_1_is_populated_and_microstructure_and_rv_are_strictly_causal():
    bars = DataPipeline.synthetic(symbol="XAU/USD", n_bars=800).drop(columns=["bid", "ask"], errors="ignore")
    ctx_bars = DataPipeline.synthetic(symbol="XAG/USD", n_bars=800).drop(columns=["bid", "ask"], errors="ignore")
    ctx_bars["time"] = bars["time"].to_numpy()

    fcfg = FeatureConfig(
        synthetic_microstructure=True,
        include_gold_session_features=True,
        include_context_features=True,
        context_config={
            "symbols": {"XAG/USD": "xag"},
            "include_rv_features": True,
            "rv_windows": [1, 4],
            "rv_norm_window": 60,
            "zscore_window": 60,
        },
    )
    eng = FeatureEngineer(fcfg)
    raw_f = eng.transform(bars, context={"XAG/USD": ctx_bars})

    assert raw_f["ret_1"].iloc[25:].notna().all()
    assert raw_f["ret_1"].iloc[25:].std() > 0.0
    assert raw_f["spread_bps"].iloc[30:].gt(0).any()
    assert raw_f["quote_imbalance"].iloc[30:].abs().gt(0).any()

    norm_a = eng.transform_normalized(bars, context={"XAG/USD": ctx_bars})
    bars_mut = bars.copy()
    ctx_mut = ctx_bars.copy()
    bars_mut.loc[501:, ["open", "high", "low", "close"]] *= 1.35
    ctx_mut.loc[501:, ["open", "high", "low", "close"]] *= 0.65
    norm_b = eng.transform_normalized(bars_mut, context={"XAG/USD": ctx_mut})
    np.testing.assert_allclose(
        norm_a.iloc[:501].to_numpy(),
        norm_b.iloc[:501].to_numpy(),
        atol=1e-12,
    )


# ── Bug #3: ExecutionConfig YAML mapping & sub-min_qty router safety ────────
def test_execution_config_from_dict_and_router_rejects_sub_min_qty():
    ecfg = ExecutionConfig.from_dict({"order_type": "market", "max_slippage_bps": 3.2, "dry_run": True})
    assert ecfg.use_limit_orders is False
    assert ecfg.max_impact_vwap_bps == pytest.approx(3.2)

    class StubClient:
        def market_order_cost(self, *args, **kwargs):
            raise AssertionError("market_order_cost must never fire for sub-min_qty")

    router = OrderRouter(StubClient(), ExecutionConfig(verify_fills=False), {"XAU/USD": _make_settings()})
    res = router.open_position("XAU/USD", "LONG", qty=0.004, stop_loss=None, take_profit=None, reference_price=4000.0)
    assert res.ok is False
    assert "below minimum" in (res.error or "")


# ── Bug #6: OrderRouter fill verification on scaled entries #2..#10 ─────────
@pytest.mark.parametrize("mode", ["HEDGE", "NETTING"])
def test_router_verify_fills_detects_second_position_in_hedge_and_netting(mode):
    pos1 = _make_position(101, qty=0.01, open_price=4000.0)
    pos2 = _make_position(102, qty=0.01, open_price=4001.0)

    class ScalingStubClient:
        def __init__(self):
            self.pos_calls = 0

        def positions(self):
            self.pos_calls += 1
            # Call 1 = pre-order snapshot; Call 2 = first post-order poll (still stale);
            # Call 3 = venue reflects the new fill!
            if self.pos_calls <= 2:
                return [pos1]
            if mode == "HEDGE":
                return [pos1, pos2]
            return [_make_position(101, qty=0.02, open_price=4000.5)]

        def market_order_cost(self, *args, **kwargs):
            return _make_cost(4001.0)

        def place_market(self, *args, **kwargs):
            return {"orderId": 999}

    client = ScalingStubClient()
    router = OrderRouter(
        client,
        ExecutionConfig(verify_fills=True, verify_timeout_s=1.0, attach_protection=False),
        {"XAU/USD": _make_settings()},
    )
    out = router.open_position("XAU/USD", "LONG", qty=0.01, stop_loss=None, take_profit=None, reference_price=4001.0)
    assert out.ok is True
    assert client.pos_calls >= 3  # did NOT prematurely accept stale pos1 on call 2!


# ── Bug #4: Backtester pnl_coefficient, open-bar fill & IEEE-754 qty_step ───
def test_backtester_pnl_coefficient_and_qty_step_precision():
    bars = DataPipeline.synthetic(symbol="XAU/USD", n_bars=300).drop(columns=["bid", "ask"], errors="ignore")
    windows = np.zeros((250, 4, 8), dtype=np.float32)
    idx = np.arange(20, 270, dtype=np.int64)
    atr = np.full(len(bars), 2.0, dtype=np.float64)

    def sig(row: int, _w: np.ndarray):
        return (1, 0.8, 0.2) if row == 10 else (0, 0.0, 0.0)

    cfg = BacktestConfig(
        initial_equity=100.0,
        risk_per_trade_pct=0.6,
        sl_atr=1.0,
        tp_atr=0.0,
        max_holding_bars=2,
        spread_bps_fallback=0.0,
        taker_fee_pct=0.0,
        qty_step=0.01,
        min_qty=0.01,
        pnl_coefficient=10.0,
    )
    res = Backtester(cfg).run(bars, windows, idx, atr, sig)
    assert len(res.trades) == 1
    tr = res.trades[0]
    assert tr.qty == pytest.approx(0.03)
    entry_bar = int(idx[10]) + 1
    assert tr.entry_price == pytest.approx(float(bars["open"].iloc[entry_bar]))
    expected_pnl = (tr.exit_price - tr.entry_price) * tr.qty * 10.0
    assert tr.pnl == pytest.approx(expected_pnl)


def test_backtest_runner_runs_cleanly_on_synthetic_config(tmp_path):
    cfg_file = tmp_path / "bt.yaml"
    cfg_file.write_text(Path("configs/live.mini100.yaml").read_text())
    res = run_backtest_from_config(cfg_file, synthetic=True, oos_frac=0.3)
    assert len(res.equity_curve) > 100


# ── Bug #5: LiveEngine multi-position exit loop & clock consistency ─────────
def test_live_engine_closes_all_slots_on_time_stop_not_just_first(tmp_path):
    closed_ids: list[int] = []
    pos_list = [
        _make_position(1, open_time=100_000),
        _make_position(2, open_time=100_000),
    ]

    class FakeClient:
        def bars(self, symbol, bar_size, limit=2):
            return []

        def quotes(self, symbols):
            return [Quote("XAU/USD", bid=4000.0, ask=4000.2, price=4000.1, timestamp_ms=900_000)]

    class FakeRouter:
        def __init__(self):
            self.settings = {}

        def close_position(self, pos: Position, reason: str = ""):
            closed_ids.append(pos.id)
            return ExecutionResult(ok=True, intent="CLOSE", symbol=pos.symbol, qty=pos.qty, price=4000.0)

    risk = RiskEngine(
        RiskConfig(state_path=str(tmp_path / "r.json"), kill_switch_path=str(tmp_path / "K")),
        RiskState(),
    )
    eng = LiveEngine(
        client=FakeClient(),
        predictor=load_predictor(Path("artifacts/models/latest"), arch="tcn", expected_symbol="XAU/USD"),
        risk=risk,
        router=FakeRouter(),
        strategy=ScalpingStrategy(None, StrategyConfig()),
        config=LiveConfig(
            symbols=["XAU/USD"], bar_size="1M", max_hold_bars=5, log_path=str(tmp_path / "audit.jsonl")
        ),
        clock=lambda: 900.0,  # 900_000 ms -> (900_000 - 100_000)/60_000 = 13 bars >= 5!
    )
    eng.settings["XAU/USD"] = _make_settings()
    eng.buffers["XAU/USD"] = DataPipeline.synthetic(symbol="XAU/USD", n_bars=120)
    eng._process_symbol("XAU/USD", account=type("Acc", (), {"equity": 100.0})(), positions=pos_list)
    assert closed_ids == [1, 2]  # BOTH positions closed in the same bar!


# ── Bug #7: Catalogue metals recognition & SymbolSettings.floor_qty ─────────
def test_instrument_is_fx_recognizes_metals_and_floor_qty():
    inst_metal = Instrument("XAU/USD", "CFD", "metals", "XAU", "USD", 4000.0, 3999.9, 4000.1, True)
    inst_comm = Instrument("XAU/USD", "CFD", "commodities", "XAU", "USD", 4000.0, 3999.9, 4000.1, True)
    inst_oil = Instrument("BRENT/USD", "CFD", "commodities", "BRENT", "USD", 75.0, 74.9, 75.1, True)
    assert inst_metal.is_fx is True
    assert inst_comm.is_fx is True
    assert inst_oil.is_fx is False

    s = _make_settings(min_qty=0.01, max_qty=50.0)
    assert s.floor_qty(0.008) == 0.0
    assert s.floor_qty(0.039) == pytest.approx(0.03)
    assert s.round_qty(999.0) == pytest.approx(50.0)


# ── Bug #8: Target class alignment & discrete tuple RL decoding ─────────────
def test_barrier_y_dir_and_window_dataset_guard_and_discrete_rl():
    bars = DataPipeline.synthetic(symbol="XAU/USD", n_bars=300)
    out = directed_barrier_labels(bars, BarrierConfig(max_holding=10))
    assert "y_dir" in out.columns
    y_np = out["y"].to_numpy()
    yd_np = out["y_dir"].to_numpy()
    assert np.all(yd_np[y_np == 0] == 0)
    assert np.all(yd_np[y_np == 1] == -1)
    assert np.all(yd_np[y_np == 2] == 1)
    assert np.array_equal(barrier_y_to_dir(y_np), yd_np)

    w = np.zeros((10, 4, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="directional labels in \\{-1, 0, \\+1\\}"):
        WindowDataset(w, np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]))
    with pytest.raises(ValueError, match="directional labels"):
        class_weights(np.array([0, 1, 2, 0]))

    class StubPred:
        config = type("C", (), {"backend": "onnx"})()

        def predict(self, _w):
            return Prediction(
                direction=-1, probs=np.array([0.65, 0.20, 0.15]), confidence=0.65, latency_us=10.0
            )

    class DiscreteTuplePPO:
        continuous = False

        def act(self, _obs):
            return (2, -0.4, 0.1)  # discrete 2 == SHORT (-1.0) in FXScalpEnv!

    strat = ScalpingStrategy(
        StubPred(), rl_policy=DiscreteTuplePPO(), config=StrategyConfig(require_rl_agreement=True)
    )
    dec = strat.decide(np.zeros((4, 8), dtype=np.float32), rl_observation=np.zeros(10, dtype=np.float32))
    assert dec.intent == Intent.SHORT
    assert dec.rl_action == pytest.approx(-1.0)


# ── Bug #9: Trainer EMA warmup wiring & overlap-scaled Sharpe ───────────────
def test_trainer_wires_ema_warmup_and_bars_per_year():
    model = TCN(ModelConfig(n_features=4, lookback=8, channels=(8,)))
    t_warm = Trainer(model, TrainConfig(epochs=1, ema_warmup=True))
    t_cold = Trainer(model, TrainConfig(epochs=1, ema_warmup=False))
    assert t_warm.ema.warmup is True
    assert t_cold.ema.warmup is False

    w = np.random.default_rng(0).standard_normal((120, 4, 8)).astype(np.float32)
    y = np.zeros(120, dtype=np.int64)
    y_ret = np.linspace(0.001, 0.002, 120)
    loader = torch.utils.data.DataLoader(WindowDataset(w, y), batch_size=64)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.head.net[-1].bias[:] = torch.tensor([-5.0, -5.0, 5.0])
    t1 = Trainer(model, TrainConfig(epochs=1, bars_per_year=252 * 288), window_overlap=1)
    t12 = Trainer(model, TrainConfig(epochs=1, bars_per_year=252 * 288), window_overlap=12)
    s1 = t1._directional_sharpe(model, loader, y_ret)
    s12 = t12._directional_sharpe(model, loader, y_ret)
    assert s12 < s1


# ── Bug #10: MacroCalendar out-of-range warning & CSV override ──────────────
def test_calendar_warns_out_of_range_and_loads_schedule_path(tmp_path, caplog):
    future_2028_ms = int(pd.Timestamp("2028-06-15T12:00:00Z").timestamp() * 1000)
    with caplog.at_level(logging.WARNING):
        calendar_features(np.array([future_2028_ms], dtype=np.int64))
    assert any("outside schedule span" in rec.message for rec in caplog.records)

    csv_path = tmp_path / "sched_2028.csv"
    csv_path.write_text("kind,iso_utc\ncpi,2028-06-15T12:30:00Z\n")
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        df = calendar_features(
            np.array([future_2028_ms], dtype=np.int64),
            config=CalendarConfig(schedule_path=str(csv_path)),
        )
    assert not any("outside schedule span" in rec.message for rec in caplog.records)
    assert df["cal_cpi_pre_60"].iloc[0] == pytest.approx(1.0)

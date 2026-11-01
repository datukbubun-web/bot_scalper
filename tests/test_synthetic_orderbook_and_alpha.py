"""Tests for Synthetic L2 Order Book reconstruction, causal quote recording,
and shipped XAU/USD OOS Positive Alpha models."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pxscalper.exchange.models import OrderCost, Quote
from pxscalper.exchange.orderbook import (
    QuoteBookRecorder,
     aggregate_quotes_to_bars,
    invert_vwap_ladder,
    probe_synthetic_orderbook,
)
from pxscalper.features.pipeline import FeatureConfig
from pxscalper.inference.predictor import load_predictor


def test_invert_vwap_ladder_recovers_exact_marginal_book_slices():
    # True ask book: 1 oz @ 4000.10, next 4 oz @ 4000.30, next 5 oz @ 4000.80
    # Cumulative VWAPs:
    # Q=1  -> 4000.10
    # Q=5  -> (1*4000.10 + 4*4000.30) / 5 = 4000.26
    # Q=10 -> (20001.30 + 5*4000.80) / 10 = 4000.53
    rungs = [(1.0, 4000.10), (5.0, 4000.26), (10.0, 4000.53)]
    levels = invert_vwap_ladder(rungs, mid=4000.00, side="BUY")
    assert len(levels) == 3
    assert levels[0].marginal_price == pytest.approx(4000.10, abs=1e-6)
    assert levels[0].slice_qty == pytest.approx(1.0)
    assert levels[1].marginal_price == pytest.approx(4000.30, abs=1e-6)
    assert levels[1].slice_qty == pytest.approx(4.0)
    assert levels[2].marginal_price == pytest.approx(4000.80, abs=1e-6)
    assert levels[2].slice_qty == pytest.approx(5.0)


def test_probe_synthetic_orderbook_computes_imbalance_and_slippage():
    # Thicker bid book (tiny impact) vs thinner ask book (steeper impact)
    # -> positive L2 imbalance? Wait: thinner asks mean ask_cost > bid_cost -> positive upward ease!
    class ImpactClient:
        def market_order_cost(self, symbol: str, side: str, qty: float):
            if side == "BUY":
                px = 4000.10 + 0.05 * qty  # steeper ask impact
            else:
                px = 3999.90 - 0.005 * qty  # deep, tight bid book
            return OrderCost(
                currency="USD",
                price=px,
                order_cost=px * qty,
                margin_impact=px * qty * 0.1,
                fee=0.0,
                fee_pct=0.0,
                leverage=10.0,
                liq_price=None,
                min_order_size=0.01,
                resolution="OK",
                resolution_level="INFO",
            )

    q = Quote(symbol="XAU/USD", bid=3999.90, ask=4000.10, price=4000.00, timestamp_ms=1_700_000_000_000)
    snap = probe_synthetic_orderbook(ImpactClient(), "XAU/USD", size_ladder=(1.0, 10.0, 50.0), quote=q)
    assert snap is not None
    assert snap.spread_bps == pytest.approx(0.5, abs=1e-3)
    assert snap.ask_slope_bps > snap.bid_slope_bps
    assert snap.l2_imbalance > 0.0
    assert snap.estimate_slippage_bps("BUY", 25.0) > snap.estimate_slippage_bps("SELL", 25.0)


def test_quote_book_recorder_and_strict_causal_bar_aggregation(tmp_path):
    rec = QuoteBookRecorder(tmp_path)
    # Bar 0: [100_000, 160_000), Bar 1: [160_000, 220_000)
    rec.record_quote(Quote("XAU/USD", bid=4000.0, ask=4000.2, price=4000.1, timestamp_ms=130_000))
    # Future quote inside Bar 1 MUST NEVER leak into Bar 0!
    rec.record_quote(Quote("XAU/USD", bid=4010.0, ask=4010.2, price=4010.1, timestamp_ms=165_000))

    quotes = rec.load_quotes("XAU/USD")
    bars = pd.DataFrame(
        {
            "time": [40_000, 100_000, 160_000],
            "open": [3999.0, 4000.0, 4005.0],
            "high": [3999.5, 4000.5, 4011.0],
            "low": [3998.5, 3999.5, 4004.0],
            "close": [3999.2, 4000.1, 4010.1],
            "volume": [10.0, 12.0, 15.0],
        }
    )
    joined = aggregate_quotes_to_bars(bars, quotes, bar_ms=60_000)
    assert np.isnan(joined["bid"].iloc[0])  # no quote existed during [40_000, 100_000]
    assert joined["bid"].iloc[1] == pytest.approx(4000.0)  # strictly Bar 0's quote, NOT 4010.0!
    assert joined["bid"].iloc[2] == pytest.approx(4010.0)


@pytest.mark.parametrize("model_subpath", ["models/latest", "xau_5M", "xau_15M"])
def test_shipped_xau_models_are_antisymmetric_roll_gated_and_positive_oos(model_subpath):
    mdir = Path("artifacts") / model_subpath
    pred = load_predictor(
        mdir,
        arch="tcn",
        expected_symbol="XAU/USD",
        confidence_threshold=0.45,
        edge_threshold=0.03,
    )
    fcfg = FeatureConfig.load(mdir / "features.json")
    from pxscalper.features.pipeline import FeatureEngineer

    names = list(FeatureEngineer(fcfg).feature_names)
    assert len(names) == pred.config.n_features
    roll_ch = names.index("gold_roll_blackout")
    rv_cols = [i for i, c in enumerate(names) if c.startswith("ctx_xag_rv_z_")]
    assert len(rv_cols) >= 2

    # 1. Zero window -> exact FLAT (direction == 0, edge == 0)
    w_zero = np.zeros((pred.config.n_features, pred.config.lookback), dtype=np.float32)
    p0 = pred.predict(w_zero)
    assert p0.direction == 0
    assert p0.edge == pytest.approx(0.0, abs=1e-6)

    # 2. Exact antisymmetry: P(up | -x) == P(down | +x)
    w_pos = np.zeros_like(w_zero)
    for idx in rv_cols:
        w_pos[idx, -1] = 2.8  # +2.8 sigma XAU overshoot vs XAG -> fade SHORT (-1)
    p_pos = pred.predict(w_pos)
    p_neg = pred.predict(-w_pos)
    assert p_pos.direction == -1
    assert p_neg.direction == 1
    assert p_pos.probs[0] == pytest.approx(p_neg.probs[2], abs=1e-5)

    # 3. Built-in 21:00-23:00 UTC maintenance roll blackout gate
    w_roll = w_pos.copy()
    w_roll[roll_ch, -1] = 1.0
    assert pred.predict(w_roll).direction == 0

    # 4. Verified positive OOS net return & Sharpe across ALL cost regimes including 2x stress
    report = json.loads((mdir / "training_report.json").read_text())
    assert report["symbol"] == "XAU/USD"
    for slice_key in ("validation_60_80", "blind_test_80_100", "full_unseen_oos_60_100"):
        for cost_key in ("live_0.49bps", "realistic_0.65bps", "stress_2x_1.30bps"):
            m = report["oos_verification"][slice_key][cost_key]
            assert m["total_return_pct"] > 0.0, f"{model_subpath} {slice_key} {cost_key} negative return"
            assert m["sharpe"] > 1.0, f"{model_subpath} {slice_key} {cost_key} Sharpe <= 1.0"
            assert m["profit_factor"] > 1.15


def test_live_engine_probes_and_records_synthetic_l2(tmp_path):
    from pxscalper.data.pipeline import DataPipeline
    from pxscalper.exchange.models import SymbolSettings
    from pxscalper.execution.router import ExecutionConfig, OrderRouter
    from pxscalper.live.engine import LiveConfig, LiveEngine
    from pxscalper.risk.engine import RiskConfig, RiskEngine, RiskState
    from pxscalper.strategy.decision import ScalpingStrategy, StrategyConfig

    class ProbeStubClient:
        def bars(self, symbol, bar_size, limit=2):
            return []

        def quotes(self, symbols):
            return [Quote("XAU/USD", bid=3999.90, ask=4000.10, price=4000.00, timestamp_ms=500_000)]

        def market_order_cost(self, symbol: str, side: str, qty: float):
            px = 4000.10 + 0.08 * qty if side == "BUY" else 3999.90 - 0.01 * qty
            return OrderCost(
                currency="USD",
                price=px,
                order_cost=px * qty,
                margin_impact=px * qty * 0.1,
                fee=0.0,
                fee_pct=0.0,
                leverage=10.0,
                liq_price=None,
                min_order_size=0.01,
                resolution="OK",
                resolution_level="INFO",
            )

    recorder = QuoteBookRecorder(tmp_path / "quotes")
    client = ProbeStubClient()
    pred = load_predictor(Path("artifacts/models/latest"), arch="tcn", expected_symbol="XAU/USD")
    eng = LiveEngine(
        client=client,
        predictor=pred,
        risk=RiskEngine(
            RiskConfig(state_path=str(tmp_path / "r.json"), kill_switch_path=str(tmp_path / "K")),
            RiskState(),
        ),
        router=OrderRouter(client, ExecutionConfig(dry_run=True)),
        strategy=ScalpingStrategy(pred, config=StrategyConfig()),
        config=LiveConfig(
            symbols=["XAU/USD"],
            bar_size="5M",
            l2_probe_interval_s=30.0,
            l2_probe_ladder=(1.0, 10.0),
            log_path=str(tmp_path / "decisions.jsonl"),
        ),
        feature_config=FeatureConfig.load("artifacts/models/latest/features.json"),
        clock=lambda: 100.0,
        quote_recorder=recorder,
    )
    eng.settings["XAU/USD"] = SymbolSettings(
        "XAU/USD", 0.01, 100.0, 0.01, 0.01, 2, 20, 10, 0.0, 0.0
    )
    b = DataPipeline.synthetic("XAU/USD", n_bars=150)
    c = DataPipeline.synthetic("XAG/USD", n_bars=150)
    c["time"] = b["time"].to_numpy()
    eng.buffers["XAU/USD"] = b
    eng.context_buffers["XAG/USD"] = c

    eng._process_symbol("XAU/USD", account=type("Acc", (), {"equity": 100.0})(), positions=[])
    recorded = recorder.load_quotes("XAU/USD")
    assert len(recorded) == 1
    assert recorded["l2_imbalance"].iloc[0] > 0.0
    assert recorded["ask_slope_bps"].iloc[0] > recorded["bid_slope_bps"].iloc[0]


def test_cli_backtest_oos_and_risk_show():
    from typer.testing import CliRunner

    from pxscalper.cli import app

    runner = CliRunner()
    r1 = runner.invoke(app, ["risk", "show", "--config", "configs/live.mini100.yaml"])
    assert r1.exit_code == 0
    assert "max_intraday_loss_pct" in r1.stdout

    r2 = runner.invoke(app, ["backtest", "--config", "configs/live.mini100.yaml", "--oos-frac", "0.40"])
    assert r2.exit_code == 0
    assert "trades=" in r2.stdout


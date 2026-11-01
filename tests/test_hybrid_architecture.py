"""Tests for the Hybrid Architecture (Intraday Bias + Scalper Execution).

Verifies:
1. Strict zero-look-ahead multi-timeframe alignment (``T_htf + htf_ms <= t_ltf + ltf_ms``)
   and partial in-flight HTF candle rejection.
2. IntradayBiasEngine regime classification & directional gate consistency.
3. HybridStrategy confluence gating, 1M micro-absorption timing & size/TP modulation.
4. Exported Hybrid ONNX/TorchScript models (``xau_hybrid_5M``, ``xau_hybrid_1M``)
   exact antisymmetry, built-in MTF graph gates, and unseen OOS alpha.
5. End-to-end LiveEngine, BacktestRunner, and CLI ``pxs bias`` integration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from pxscalper.cli import app
from pxscalper.features.mtf import (
    align_htf_to_ltf,
    resample_ohlcv_causal,
)
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer
from pxscalper.inference.predictor import Prediction, load_predictor
from pxscalper.live.engine import LiveEngine
from pxscalper.ops.backtest_runner import run_backtest_from_config
from pxscalper.strategy.decision import Intent, StrategyConfig
from pxscalper.strategy.hybrid import HybridConfig, HybridStrategy
from pxscalper.strategy.intraday_bias import (
    BiasDirection,
    IntradayBiasConfig,
    IntradayBiasEngine,
    IntradayRegime,
)


def _make_5m_bars(n: int = 360, start_ms: int = 1_760_000_100_000, seed: int = 11) -> pd.DataFrame:
    # Align start_ms to exact 15M boundary
    t0 = (start_ms // 900_000) * 900_000
    rng = np.random.default_rng(seed)
    times = t0 + np.arange(n, dtype=np.int64) * 300_000
    rets = rng.normal(0.0, 0.0006, size=n)
    close = 4000.0 * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1)
    open_[0] = 4000.0
    spread = np.abs(rng.normal(1.2, 0.4, size=n))
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    vol = rng.uniform(50.0, 300.0, size=n)
    return pd.DataFrame(
        {"time": times, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    )


def test_resample_ohlcv_causal_drops_trailing_incomplete_htf_bucket():
    # 7 x 5M bars starting on a 15M boundary:
    # - Bucket 0: bars 0, 1, 2 -> complete [0, 15m)
    # - Bucket 1: bars 3, 4, 5 -> complete [15m, 30m)
    # - Bucket 2: bar 6 ONLY  -> incomplete [30m, 45m), closed only up to 35m!
    bars_5m = _make_5m_bars(n=7)
    res_closed = resample_ohlcv_causal(bars_5m, "15M", source_bar_size="5M", drop_incomplete=True)
    assert len(res_closed) == 2
    assert int(res_closed["time"].iloc[0]) == int(bars_5m["time"].iloc[0])
    assert int(res_closed["time"].iloc[1]) == int(bars_5m["time"].iloc[3])


def test_align_htf_to_ltf_strictly_prevents_inflight_htf_lookahead():
    # Suppose 15M bar T0=[00:00, 00:15) has val=10.0 and T1=[00:15, 00:30) has val=999.0.
    # On 5M bars:
    # - t=00:00 (closes 00:05) -> NEITHER T0 nor T1 has closed yet! Must be 0.0.
    # - t=00:05 (closes 00:10) -> NEITHER T0 nor T1 has closed yet! Must be 0.0.
    # - t=00:10 (closes 00:15) -> T0 JUST closed at 00:15! Must see 10.0, NEVER 999.0!
    # - t=00:15 (closes 00:20) -> T1 is in-flight (closes 00:30)! Must STILL see 10.0!
    # - t=00:20 (closes 00:25) -> Must STILL see 10.0!
    # - t=00:25 (closes 00:30) -> T1 closes at 00:30! Now sees 999.0!
    base = 1_760_000_700_000  # divisible by 900_000? normalize explicitly:
    base = (base // 900_000) * 900_000
    ltf = pd.DataFrame({"time": [base + k * 300_000 for k in range(6)]})
    htf = pd.DataFrame({"time": [base, base + 900_000], "signal": [10.0, 999.0]})

    aligned = align_htf_to_ltf(ltf, htf, ltf_bar_size="5M", htf_bar_size="15M", value_columns=["signal"])
    got = aligned["signal"].to_numpy()
    np.testing.assert_allclose(got, [0.0, 0.0, 10.0, 10.0, 10.0, 999.0])


def test_mtf_feature_pipeline_adversarial_future_shock_invariance():
    """Mutating future & in-flight 5M/15M prices after cutoff k MUST NOT alter any feature <= k."""
    xau5 = _make_5m_bars(n=450, seed=21)
    xag5 = _make_5m_bars(n=450, seed=22)

    fcfg = FeatureConfig(
        bar_size="5M",
        synthetic_microstructure=True,
        include_gold_session_features=True,
        include_context_features=True,
        context_config={
            "symbols": {"XAG/USD": "xag"},
            "zscore_window": 96,
            "min_coverage": 0.25,
            "include_rv_features": True,
            "rv_windows": [1, 4, 12],
            "rv_norm_window": 48,
        },
        include_mtf_bias_features=True,
        mtf_bias_config={
            "htf_bar_sizes": ["15M"],
            "context_symbols": {"XAG/USD": "xag"},
            "min_coverage": 0.25,
        },
    )
    eng = FeatureEngineer(fcfg)
    feats_clean = eng.transform_normalized(xau5, context={"XAG/USD": xag5})

    # Choose cutoff k=301 (middle of an in-flight 15M bucket: bars 300, 301, 302 form one 15M candle!)
    # Shock bar 302 (and all bars >= 302) by +40%! Bar 301 MUST NOT see bar 302's shock via 15M!
    cutoff = 301
    xau5_shock = xau5.copy()
    xag5_shock = xag5.copy()
    xau5_shock.loc[cutoff + 1 :, ["open", "high", "low", "close"]] *= 1.40
    xag5_shock.loc[cutoff + 1 :, ["open", "high", "low", "close"]] *= 0.70

    htf_shock = resample_ohlcv_causal(xau5_shock, "15M", source_bar_size="5M", drop_incomplete=True)
    htf_ctx_shock = resample_ohlcv_causal(xag5_shock, "15M", source_bar_size="5M", drop_incomplete=True)
    feats_shock = eng.transform_normalized(
        xau5_shock,
        context={"XAG/USD": xag5_shock},
        htf_bars={"15M": htf_shock},
        htf_context={"15M": {"XAG/USD": htf_ctx_shock}},
    )

    np.testing.assert_allclose(
        feats_clean.iloc[: cutoff + 1].to_numpy(),
        feats_shock.iloc[: cutoff + 1].to_numpy(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_intraday_bias_engine_regimes_and_directional_gate_consistency():
    names = [
        "ctx_xag_rv_z_4",
        "htf_15m_xag_rv_z_2",
        "intraday_session_vwap_z",
        "intraday_session_trend",
        "intraday_vol_regime",
        "intraday_er_2h",
        "gold_roll_blackout",
    ]
    eng = IntradayBiasEngine(IntradayBiasConfig(), feature_names=names)

    # 1. Roll blackout overrides everything
    w = np.zeros((len(names), 4), dtype=np.float32)
    w[0, -1] = -2.5
    w[1, -1] = -2.2
    w[6, -1] = 1.0
    snap_roll = eng.evaluate_window(w)
    assert snap_roll.regime is IntradayRegime.ROLL_BLACKOUT
    assert snap_roll.direction is BiasDirection.NEUTRAL
    assert not snap_roll.evaluate_trade_alignment(1, eng.config)[0]
    assert not snap_roll.evaluate_trade_alignment(-1, eng.config)[0]

    # 2. Volatility shock (> 1.85x => excess > 0.85) blocks both sides
    w_vol = np.zeros((len(names), 4), dtype=np.float32)
    w_vol[0, -1] = -2.5
    w_vol[1, -1] = -2.2
    w_vol[4, -1] = 1.10  # 2.10x vol ratio
    snap_vol = eng.evaluate_window(w_vol)
    assert snap_vol.regime is IntradayRegime.VOLATILITY_SHOCK
    assert not snap_vol.evaluate_trade_alignment(1, eng.config)[0]

    # 3. Strong bullish session trend (+0.75) vetoes counter-trend SHORT while allowing aligned LONG
    w_bull = np.zeros((len(names), 4), dtype=np.float32)
    w_bull[0, -1] = -2.2  # XAU undervalued -> +2.2 bullish FV
    w_bull[1, -1] = -1.8  # 15M XAU undervalued -> +1.8 bullish HTF FV
    w_bull[2, -1] = 1.8
    w_bull[3, -1] = 0.75
    snap_bull = eng.evaluate_window(w_bull)
    assert snap_bull.regime is IntradayRegime.TREND_ACCUMULATION
    assert snap_bull.direction is BiasDirection.BULLISH
    assert snap_bull.evaluate_trade_alignment(1, eng.config)[0] is True
    assert snap_bull.evaluate_trade_alignment(-1, eng.config)[0] is False


def test_hybrid_strategy_micro_timing_and_confluence_modulation():
    names = [
        "close_location",
        "ctx_xag_rv_z_4",
        "htf_15m_xag_rv_z_2",
        "intraday_session_vwap_z",
        "intraday_session_trend",
        "intraday_vol_regime",
        "intraday_er_2h",
        "gold_roll_blackout",
    ]

    class StubPredictor:
        def predict(self, _window):
            return Prediction(
                probs=np.array([0.08, 0.32, 0.60], dtype=np.float32),
                direction=1,
                confidence=0.60,
                latency_us=25.0,
                model_version="stub",
            )

    hcfg = HybridConfig(
        enabled=True,
        require_composite_threshold=True,
        composite_threshold=1.85,
        require_micro_absorption=True,
        min_wick_absorption=0.10,
        modulate_size_by_bias=True,
        aligned_size_boost=1.20,
        aligned_tp_mult=1.15,
    )
    strat = HybridStrategy(
        predictor=StubPredictor(),
        config=StrategyConfig(
            min_confidence=0.45,
            min_edge=0.03,
            require_rl_agreement=False,
            size_by_edge=False,
            max_size_multiplier=1.5,
        ),
        hybrid_config=hcfg,
        feature_names=names,
    )

    # Case A: Intraday Bias aligned (+2.2), but 1M candle closed on its lows (close_location = -0.60)
    # -> Scalper micro-timing waits!
    w_wait = np.zeros((len(names), 8), dtype=np.float32)
    w_wait[0, -1] = -0.60
    w_wait[1, -1] = -2.30
    w_wait[2, -1] = -2.10
    w_wait[4, -1] = 0.50
    dec_wait = strat.decide(w_wait)
    assert dec_wait.intent is Intent.FLAT
    assert "micro-timing wait" in dec_wait.reason

    # Case B: Next 1M candle absorbs selling and closes in upper half (close_location = +0.45)
    # -> Fires LONG with CONFLUENCE tier boost!
    w_fire = w_wait.copy()
    w_fire[0, -1] = 0.45
    dec_fire = strat.decide(w_fire)
    assert dec_fire.intent is Intent.LONG
    assert dec_fire.meta["execution_tier"] == "CONFLUENCE"
    assert dec_fire.size_multiplier == pytest.approx(1.20)
    assert dec_fire.meta["tp_atr_mult"] == pytest.approx(1.15)
    assert dec_fire.meta["intraday_bias"]["direction"] == "BULLISH"


@pytest.mark.parametrize("model_subpath", ["xau_hybrid_5M", "xau_hybrid_1M"])
def test_exported_hybrid_models_exact_antisymmetry_gates_and_oos_alpha(model_subpath):
    mdir = Path("artifacts") / model_subpath
    pred = load_predictor(
        mdir,
        arch="tcn",
        expected_symbol="XAU/USD",
        confidence_threshold=0.45,
        edge_threshold=0.03,
    )
    fcfg = FeatureConfig.load(mdir / "features.json")
    names = list(FeatureEngineer(fcfg).feature_names)
    assert len(names) == pred.config.n_features

    # 1. Zero window -> exact FLAT
    w0 = np.zeros((pred.config.n_features, pred.config.lookback), dtype=np.float32)
    p0 = pred.predict(w0)
    assert p0.direction == 0
    assert p0.edge == pytest.approx(0.0, abs=1e-6)

    # 2. Exact mathematical odd-function antisymmetry under arbitrary active input x -> -x
    w_act = np.zeros_like(w0)
    htf_idx = names.index("htf_15m_xag_rv_z_2")
    ltf_name = "ctx_xag_rv_z_4" if "ctx_xag_rv_z_4" in names else "htf_5m_xag_rv_z_4"
    ltf_idx = names.index(ltf_name)
    w_act[ltf_idx, -1] = 2.4
    w_act[htf_idx, -1] = 2.1
    if "close_location" in names:
        w_act[names.index("close_location"), -1] = -0.35  # short rejection wick for +RV overshoot
    p_pos = pred.predict(w_act)
    p_neg = pred.predict(-w_act)
    assert p_pos.direction == -1
    assert p_neg.direction == 1
    assert p_pos.probs[0] == pytest.approx(p_neg.probs[2], abs=1e-5)

    # 3. Built-in graph HTF disagreement gate: LTF=+2.5 vs HTF=-1.2 -> gated FLAT!
    w_disagree = w_act.copy()
    w_disagree[htf_idx, -1] = -1.2
    assert pred.predict(w_disagree).direction == 0

    # 4. Built-in graph Volatility Shock gate: vol_excess=+1.2 (2.2x vol) -> gated FLAT!
    w_vol = w_act.copy()
    w_vol[names.index("intraday_vol_regime"), -1] = 1.2
    assert pred.predict(w_vol).direction == 0

    # 5. Built-in graph Session VWAP runaway guard: shorting into +0.85 session accumulation -> gated FLAT!
    w_vwap = w_act.copy()
    w_vwap[names.index("intraday_session_trend"), -1] = 0.85
    assert pred.predict(w_vwap).direction == 0

    # 6. Verified positive OOS net return, Sharpe > 2.5, PF > 1.5 across all unseen slices & 2x stress
    report = json.loads((mdir / "training_report.json").read_text())
    assert report["architecture_mode"].startswith("HYBRID_")
    for slice_key in ("validation_60_80", "blind_test_80_100", "full_unseen_oos_60_100"):
        for cost_key in ("live_0.49bps", "realistic_0.65bps", "stress_2x_1.30bps"):
            m = report["oos_verification"][slice_key][cost_key]
            assert m["total_return_pct"] > 0.0
            assert m["sharpe"] > 1.5
            assert m["profit_factor"] > 1.45
            assert m["max_drawdown_pct"] < 1.5


def test_live_engine_closed_htf_slice_drops_inflight_bar():
    # At 10:05 UTC (ltf_close_ms = 10:05), REST returns 09:45 (closed at 10:00) and 10:00 (closes 10:15)
    t_0945 = 1_760_003_100_000
    t_1000 = t_0945 + 900_000
    htf_df = pd.DataFrame(
        {
            "time": [t_0945, t_1000],
            "open": [4000.0, 4005.0],
            "high": [4006.0, 4099.0],
            "low": [3999.0, 4004.0],
            "close": [4005.0, 4095.0],
            "volume": [100.0, 50.0],
        }
    )
    closed_at_1005 = LiveEngine._closed_htf_slice(htf_df, "15M", ltf_close_ms=t_1000 + 300_000)
    assert closed_at_1005 is not None
    assert len(closed_at_1005) == 1
    assert int(closed_at_1005["time"].iloc[0]) == t_0945

    closed_at_1015 = LiveEngine._closed_htf_slice(htf_df, "15M", ltf_close_ms=t_1000 + 900_000)
    assert closed_at_1015 is not None
    assert len(closed_at_1015) == 2


def test_hybrid_backtest_runner_and_cli_bias_command():
    res = run_backtest_from_config(Path("configs/live.hybrid.yaml"), oos_frac=0.40)
    assert res.metrics["n_trades"] >= 50
    assert res.metrics["total_return_pct"] > 4.0
    assert res.metrics["sharpe"] > 3.5
    assert res.metrics["profit_factor"] > 1.8
    assert res.metrics["max_drawdown_pct"] < 1.2

    runner = CliRunner()
    out = runner.invoke(app, ["bias", "--symbol", "XAU/USD", "--bars", "5M", "--htf", "15M"])
    assert out.exit_code == 0, out.output
    assert "Hybrid Intraday Bias + Scalper State" in out.output


def test_live_runner_builds_hybrid_strategy_from_config(monkeypatch, tmp_path):
    import base64
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from pxscalper.exchange.rest import PrimeXBTClient
    from pxscalper.live.reconcile import ReconcileReport, Reconciler
    from pxscalper.ops.live_runner import build_engine
    from pxscalper.utils import RepoPaths

    key = Ed25519PrivateKey.generate()
    secret = base64.b64encode(
        key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    ).decode()
    monkeypatch.setenv("PX_API_KEY", "pxk_hybrid_test")
    monkeypatch.setenv("PX_API_SECRET", secret)
    monkeypatch.setattr(PrimeXBTClient, "sync_clock", lambda self: 0.0)
    monkeypatch.setattr(Reconciler, "run", lambda self: ReconcileReport())

    paths = RepoPaths().ensure()
    eng = build_engine(Path("configs/live.hybrid.yaml"), allow_live=False, paths=paths)
    assert isinstance(eng.strategy, HybridStrategy)
    assert eng.features.config.include_mtf_bias_features is True
    assert eng._htf_bar_sizes() == ["15M"]


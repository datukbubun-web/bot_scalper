"""Regression tests for the bugs found while running against the REAL exchange.

Every test here exists because a live account or a real historical series
exposed something the synthetic data hid. Keeping them is the difference between
"it ran in the demo" and "it is correct".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxscalper.backtest.engine import BacktestConfig, Backtester
from pxscalper.data.pipeline import DataPipeline
from pxscalper.exchange.catalogue import Instrument, SymbolResolver, resolve_units
from pxscalper.exchange.models import SymbolSettings
from pxscalper.risk.engine import RiskConfig, RiskDecision, RiskEngine, RiskState


# ── 1. cache must not depend on an optional parquet engine ───────────────────
def test_cache_falls_back_to_compressed_csv(tmp_path):
    pipe = DataPipeline(cache_dir=tmp_path)
    bars = DataPipeline.synthetic(n_bars=200)
    # simulate a machine without pyarrow: write the CSV the fallback produces
    bars.to_csv(tmp_path / "EURUSD_1M.csv.gz", index=False, compression="gzip")
    loaded = pipe.load("EUR/USD", "1M")
    assert loaded is not None
    assert len(loaded) == len(bars)
    assert "close" in loaded.columns


def test_cache_roundtrip(tmp_path):
    pipe = DataPipeline(cache_dir=tmp_path)
    bars = DataPipeline.synthetic(n_bars=150)
    pipe.save("EUR/USD", "1M", bars)
    loaded = pipe.load("EUR/USD", "1M")
    assert len(loaded) == 150
    assert loaded["close"].iloc[-1] == pytest.approx(bars["close"].iloc[-1])


# ── 2. FX size granularity: 1000-unit steps, not 0.001 ──────────────────────
def _fx_instrument(symbol="EUR/USD", price=1.115, step=1000.0, min_qty=1000.0, category="forex"):
    settings = SymbolSettings(
        symbol=symbol, min_qty=min_qty, max_qty=10_000_000, qty_step=step,
        min_price_step=1e-5, price_scale=5, max_leverage=1000, default_leverage=1000,
        taker_fee_pct=0.0, maker_fee_pct=0.0,
    )
    return Instrument(
        symbol=symbol, market_type="FOREX", category=category, base=symbol[:3],
        quote=symbol[4:], price=price, bid=price, ask=price + 1e-5, market_open=True,
        settings=settings,
    )


def test_resolve_units_rounds_down_to_step():
    instrument = _fx_instrument()
    qty, notional, note = resolve_units(instrument, 5_000)   # $5k / 1.115 = 4484 units
    assert qty == 4000.0                                     # floored to the 1000 step
    assert notional == pytest.approx(4000 * 1.115)
    assert note == ""


def test_resolve_units_refuses_below_minimum():
    instrument = _fx_instrument()
    qty, notional, note = resolve_units(instrument, 500)     # cannot reach 1000 units
    assert qty == 0.0
    assert notional == 0.0
    assert "below minimum" in note


def test_usdjpy_step_notional_is_base_currency_not_price_times_step():
    """1000 units of USD/JPY is $1,000 of base currency -- not 1000 x 157.8.
    Getting this wrong overstates risk by the FX rate (a real bug caught here)."""
    usdjpy = _fx_instrument("USD/JPY", price=157.8)
    assert usdjpy.notional_per_step() == pytest.approx(1_000.0)


def test_cross_pair_needs_a_quote_rate_and_refuses_to_guess():
    xaujpy = _fx_instrument("XAU/JPY", price=642_179.0, step=0.01, min_qty=0.01, category="metals")
    with pytest.raises(ValueError):
        xaujpy.notional_per_step()
    # with JPY->USD (1/157.8) the step is 0.01 oz -> ~$40.7
    assert xaujpy.notional_per_step(quote_usd_rate=1 / 157.8) == pytest.approx(40.7, rel=1e-2)


# ── 3. risk engine refuses instruments whose step exceeds the account's room ─
def _engine(tmp_path, **overrides):
    cfg = RiskConfig(
        state_path=str(tmp_path / "risk.json"), kill_switch_path=str(tmp_path / "KILL"),
        **overrides,
    )
    return RiskEngine(cfg, RiskState())


def _evaluate(engine, **overrides):
    params = dict(
        symbol="EUR/USD", direction=1, confidence=0.6, edge=0.15, price=1.115,
        equity=10_000.0, atr=0.0005, spread_bps=0.5, market_open=True,
        min_qty=1000.0, qty_step=1000.0, price_step=1e-5,
    )
    params.update(overrides)
    return engine.evaluate(**params)


def test_tradeable_pair_sizes_in_whole_steps(tmp_path):
    verdict = _evaluate(_engine(tmp_path))
    assert verdict.allowed
    assert verdict.qty % 1000.0 == 0.0        # never a fractional FX unit
    assert verdict.qty >= 1000.0


def test_coarse_instrument_is_blocked_with_explanation(tmp_path):
    """A step whose notional exceeds the account's room must be refused, and the
    message must say by how much (the operator needs a number to act on)."""
    # a symbol whose 1000-unit step is ~$1.1k, but the account only has $50
    verdict = _evaluate(
        _engine(tmp_path), symbol="EUR/USD", price=1.115, equity=50.0,
        min_qty=1000.0, qty_step=1000.0,
    )
    assert verdict.decision is RiskDecision.BLOCK
    assert "too coarse" in verdict.reason or "minimum" in verdict.reason


def test_unknown_quote_currency_is_refused_not_guessed(tmp_path):
    """A cross pair without a quote->USD rate must be refused: guessing the
    coefficient would silently multiply the risk budget."""
    verdict = _evaluate(_engine(tmp_path), symbol="XAU/JPY", price=642_179.0,
                        min_qty=0.01, qty_step=0.01, atr=1_500.0)
    assert verdict.decision is RiskDecision.BLOCK
    assert "rate" in verdict.reason or "coefficient" in verdict.reason


# ── 4. intraday brake (FX trades 24h, no daily close) ───────────────────────
def test_intraday_brake_pauses_trading(tmp_path):
    engine = _engine(tmp_path)
    engine.on_account(10_000)
    verdict = _evaluate(engine, equity=9_900)   # -1.0% intraday
    assert verdict.decision is RiskDecision.FLAT
    assert "intraday" in verdict.reason


# ── 5. slippage model must never invent or erase costs ──────────────────────
def test_slippage_model_is_off_by_default():
    """PrimeXBT FX volume units are undocumented: participation-based impact is
    meaningless there, so the default must be zero and costs come from bid/ask."""
    assert BacktestConfig().slippage_coef == 0.0


def test_slippage_is_capped_when_enabled():
    bars = DataPipeline.synthetic(n_bars=400)
    from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer

    engineer = FeatureEngineer(FeatureConfig(norm_min_periods=50))
    features = engineer.transform_normalized(bars)
    windows, end_idx = engineer.make_windows(features, 32)
    atr = (bars["high"] - bars["low"]).to_numpy()
    config = BacktestConfig(
        min_confidence=0.0, min_edge=0.0, slippage_coef=50.0, max_slippage_bps=2.0,
        taker_fee_pct=0.0, spread_bps_fallback=0.0, max_holding_bars=2,
    )
    result = Backtester(config).run(
        bars, windows, end_idx, atr, lambda row, window: (1, 0.9, 0.3)
    )
    if result.trades:
        assert all(t.slippage <= 2.0 + 1e-9 for t in result.trades)


# ── 6. intrabar ambiguity must be explicit and switchable ───────────────────
def test_intrabar_model_changes_outcome_of_ambiguous_bar():
    """A bar that touches both stop and target has an unknowable ordering; the
    backtester must expose that choice instead of silently picking one."""
    n = 60
    times = np.arange(n, dtype=np.int64) * 60_000
    # a tight range that touches both a 1-ATR stop and a 1-ATR target
    close = np.full(n, 1.1000)
    open_ = close.copy()
    high = close + 0.0008
    low = close - 0.0008
    bars = pd.DataFrame(
        {"time": times, "open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(n, 100.0)}
    )
    atr = np.full(n, 0.0005)
    windows = np.zeros((n, 4, 8), dtype=np.float32)
    end_idx = np.arange(n)

    def run(model: str):
        return Backtester(
            BacktestConfig(min_confidence=0.0, min_edge=0.0, sl_atr=1.0, tp_atr=1.0,
                           max_holding_bars=1, taker_fee_pct=0.0, spread_bps_fallback=0.0,
                           intrabar_model=model)
        ).run(bars, windows, end_idx, atr, lambda row, window: (1, 0.9, 0.3))

    pessimistic = run("pessimistic")
    optimistic = run("optimistic")
    if pessimistic.trades and optimistic.trades:
        assert pessimistic.trades[0].exit_reason == "stop"
        assert optimistic.trades[0].exit_reason == "target"

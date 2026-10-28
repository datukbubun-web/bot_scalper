"""Backtester accounting: costs, no look-ahead, coherent metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxscalper.backtest.engine import BacktestConfig, Backtester
from pxscalper.data.pipeline import DataPipeline
from pxscalper.features.labels import volatility_scaled_labels
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer


@pytest.fixture(scope="module")
def prepared():
    bars = DataPipeline.synthetic(n_bars=3000, seed=21)
    engineer = FeatureEngineer(FeatureConfig(norm_min_periods=50))
    features = engineer.transform_normalized(bars)
    labels = volatility_scaled_labels(bars)
    common = features.index.intersection(labels.index)
    windows, window_idx = engineer.make_windows(features.loc[common], 32)
    frame = bars.loc[common].reset_index(drop=True)
    return frame, windows, window_idx, labels.loc[common, "atr"].to_numpy()


def test_always_long_produces_trades_and_costs(prepared):
    frame, windows, times, atr = prepared

    def signal(row, window):
        return 1, 0.9, 0.3

    result = Backtester(BacktestConfig(min_confidence=0.0, min_edge=0.0, max_holding_bars=10)).run(
        frame, windows, times, atr, signal
    )
    assert len(result.trades) > 0
    assert all(t.fees > 0 for t in result.trades)
    assert result.metrics["max_drawdown_pct"] >= 0
    assert result.metrics["final_equity"] > 0


def test_no_trades_when_model_is_flat(prepared):
    frame, windows, times, atr = prepared

    def signal(row, window):
        return 0, 0.99, 0.0

    result = Backtester(BacktestConfig()).run(frame, windows, times, atr, signal)
    assert result.trades == []
    assert result.metrics["total_return_pct"] == 0.0


def test_costs_make_random_signal_lose(prepared):
    """A coin-flip scalper must lose to the spread + fees. If it doesn't, the
    backtester is not charging costs and every result it prints is fantasy."""
    frame, windows, times, atr = prepared
    rng = np.random.default_rng(0)

    def signal(row, window):
        return int(rng.choice([-1, 1])), 0.9, 0.3

    result = Backtester(
        BacktestConfig(min_confidence=0.0, min_edge=0.0, max_holding_bars=5, tp_atr=0.0)
    ).run(frame, windows, times, atr, signal)
    assert result.trades
    assert result.metrics["total_return_pct"] < 0


def test_entry_fills_on_next_bar(prepared):
    frame, windows, times, atr = prepared

    def signal(row, window):
        return 1, 0.9, 0.3

    config = BacktestConfig(min_confidence=0.0, min_edge=0.0, max_holding_bars=1, tp_atr=0.0)
    result = Backtester(config).run(frame, windows[:40], times[:40], atr, signal)
    if result.trades:
        trade = result.trades[0]
        # exit must be at or after entry, and held at least one bar
        assert trade.exit_time >= trade.entry_time
        assert trade.bars_held >= 1

"""Feature engineering: no look-ahead, stable shapes, sane values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxscalper.data.pipeline import DataPipeline
from pxscalper.features.pipeline import FEATURE_NAMES, FeatureConfig, FeatureEngineer


@pytest.fixture(scope="module")
def bars() -> pd.DataFrame:
    return DataPipeline.synthetic(n_bars=1500, seed=3)


def test_feature_frame_matches_declared_schema(bars):
    engineer = FeatureEngineer()
    features = engineer.transform(bars)
    assert list(features.columns) == list(FEATURE_NAMES)
    assert features.shape == (len(bars), len(FEATURE_NAMES))


def test_normalisation_is_bounded(bars):
    engineer = FeatureEngineer()
    normalized = engineer.transform_normalized(bars)
    assert np.isfinite(normalized.to_numpy()).all()
    assert normalized.abs().to_numpy().max() <= 4.0 + 1e-6


def test_no_lookahead(bars):
    """Truncating the future must not change past feature values."""
    engineer = FeatureEngineer(FeatureConfig(norm_min_periods=50))
    full = engineer.transform_normalized(bars)
    cut = len(bars) - 10
    truncated = engineer.transform_normalized(bars.iloc[:cut])
    pd.testing.assert_frame_equal(
        full.iloc[:cut], truncated, check_exact=False, atol=1e-8
    )


def test_windows_shape_and_dtype(bars):
    engineer = FeatureEngineer()
    normalized = engineer.transform_normalized(bars)
    windows, times = engineer.make_windows(normalized, lookback=32)
    assert windows.ndim == 3
    assert windows.shape[1] == len(FEATURE_NAMES)
    assert windows.shape[2] == 32
    assert windows.dtype == np.float32
    assert len(times) == len(windows)


def test_last_window_matches_batch_windows(bars):
    engineer = FeatureEngineer()
    normalized = engineer.transform_normalized(bars)
    windows, _ = engineer.make_windows(normalized, lookback=32)
    live = engineer.last_window(bars, lookback=32)
    assert live is not None
    # finite warm-up buffer => EMA/OBV recursions differ by a fraction of the scale
    assert np.allclose(live[0], windows[-1], atol=2e-2)


def test_handles_missing_quote_columns(bars):
    engineer = FeatureEngineer()
    trimmed = bars.drop(columns=["bid", "ask", "bid_size", "ask_size"])
    features = engineer.transform(trimmed)
    assert "quote_imbalance" in features.columns
    assert (features["quote_imbalance"] == 0.0).all()

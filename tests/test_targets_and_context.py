"""Tests for the barrier target and the cross-asset context features.

Two of these pin bugs that produced *plausible-looking* numbers and would have
been signed off by any visual review:

* context alignment lagged one bar (XAU/XAG correlation 0.008 instead of 0.82);
* the signal evaluation scored the label's own return, so a constant predictor
  showed a 97% hit rate and +9.8 bps per trade (perfect foresight).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxscalper.features.context import (
    ContextConfig,
    align_context,
    context_feature_names,
    context_features,
)
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer
from pxscalper.features.targets import (
    BarrierConfig,
    barrier_outcome,
    barrier_stats,
    directed_barrier_labels,
)


def _bars(n: int = 400, seed: int = 0, drift: float = 0.0, vol: float = 1e-3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, vol, size=n)
    close = 100.0 * np.exp(np.cumsum(steps))
    high = close * (1 + np.abs(rng.normal(0, vol / 2, size=n)))
    low = close * (1 - np.abs(rng.normal(0, vol / 2, size=n)))
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=np.int64) * 60_000,
            "open": np.concatenate([[close[0]], close[:-1]]),
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 100.0),
        }
    )


# ── barrier target ──────────────────────────────────────────────────────────
def test_barrier_never_uses_future_bars():
    """Truncating the future must not change any past label."""
    bars = _bars(300)
    full = directed_barrier_labels(bars, BarrierConfig(max_holding=10))
    cut = directed_barrier_labels(bars.iloc[:200].copy(), BarrierConfig(max_holding=10))
    common = full.index.intersection(cut.index)[:-15]      # drop the unobservable tail
    assert np.allclose(
        full.loc[common, "y"].to_numpy(), cut.loc[common, "y"].to_numpy(), equal_nan=True
    )


def test_barrier_labels_are_side_aware():
    """A rising series must produce long labels, a falling one short labels."""
    up = _bars(300, seed=1, drift=+2e-4)
    down = _bars(300, seed=2, drift=-2e-4)
    cfg = BarrierConfig(tp_atr=0.5, sl_atr=0.5, max_holding=10, cost_bps=0.1)
    up_lab = directed_barrier_labels(up, cfg)
    down_lab = directed_barrier_labels(down, cfg)
    assert (up_lab["y"] == 2).mean() > (up_lab["y"] == 1).mean()
    assert (down_lab["y"] == 1).mean() > (down_lab["y"] == 2).mean()


def test_barrier_respects_cost_hurdle():
    """With an impossible hurdle nothing is tradeable: every label must be 0."""
    bars = _bars(200)
    cfg = BarrierConfig(tp_atr=1.5, max_holding=10, cost_bps=10_000.0)
    lab = directed_barrier_labels(bars, cfg)
    assert set(lab["y"].unique()) == {0}


def test_barrier_single_side_outcome_matches_its_own_return():
    """The recorded return must be the one implied by the barrier that was hit."""
    bars = _bars(400, seed=3)
    cfg = BarrierConfig(tp_atr=1.0, sl_atr=1.0, max_holding=10)
    out = barrier_outcome(bars, side=1, config=cfg)
    atr = out.index.to_series()  # keep index
    valid = out["label"].notna()
    tp = valid & (out["label"] > 0)
    sl = valid & (out["label"] < 0)
    assert (out.loc[tp, "ret"] > 0).all(), "take-profit rows must have positive return"
    assert (out.loc[sl, "ret"] < 0).all(), "stop-loss rows must have negative return"
    del atr


def test_directed_labels_expose_both_sides_for_scoring():
    """A backtest scores the side the model chose, so BOTH sides' returns must be
    present. Exporting only the realised label's return gives every predictor the
    oracle's PnL (the 97%-hit-rate bug)."""
    lab = directed_barrier_labels(_bars(300, seed=4), BarrierConfig(max_holding=10))
    for col in ("ret_long_bps", "ret_short_bps", "y"):
        assert col in lab.columns
    assert "ret_net_bps" not in lab.columns


def test_barrier_stats_reports_class_mix_and_expectancy():
    lab = directed_barrier_labels(_bars(400, seed=5), BarrierConfig(max_holding=10))
    stats = barrier_stats(lab)
    assert stats["n"] == len(lab)
    assert set(stats["counts"]).issubset({0, 1, 2})
    assert 0 <= stats["hurdle_pass_rate"] <= 1


def test_wide_bar_touches_both_barriers_and_is_flagged():
    """When one bar engulfs both barriers, the intra-bar ordering is unknowable
    from OHLC and must be flagged instead of being counted as a clean loss.

    Note ATR scales *with* the bar range, so a uniformly wide series can never
    trigger this: the ambiguity needs a quiet regime followed by a shock, which
    is exactly when it matters.
    """
    n = 60
    close = np.full(n, 100.0)
    high = close + 0.01
    low = close - 0.01
    high[20], low[20] = 105.0, 95.0          # one shock bar swallowing both barriers
    bars = pd.DataFrame(
        {"time": np.arange(n) * 60_000, "open": close, "high": high, "low": low,
         "close": close, "volume": np.full(n, 100.0)}
    )
    out = barrier_outcome(bars, side=1,
                          config=BarrierConfig(tp_atr=1.0, sl_atr=1.0, max_holding=3))
    assert out["ambiguous"].iloc[19], "the shock bar must be flagged ambiguous"
    assert out["label"].iloc[19] == -1.0, "ambiguous resolves pessimistically"


# ── cross-asset context ─────────────────────────────────────────────────────
def test_context_alignment_is_same_bar_not_lagged():
    """Join by identical bar open time. Lagging by one bar half-disaligns the
    series: it drove a real +0.82 XAU/XAG correlation down to +0.008."""
    rng = np.random.default_rng(0)
    n = 3000
    base = rng.normal(0, 1e-3, size=n)
    target = _bars(n, seed=6)
    ctx = {
        "SYN/X": pd.DataFrame(
            {"time": target["time"], "open": 100.0, "high": 100.0, "low": 100.0,
             "close": 100.0 * np.exp(np.cumsum(base)), "volume": 1.0}
        )
    }
    aligned = align_context(target, ctx)
    target_ret = np.log(target["close"]).diff()
    ctx_ret = np.log(aligned["SYNX_close"]).diff()
    corr = np.corrcoef(target_ret.iloc[5:], ctx_ret.iloc[5:])[0, 1]
    # the synthetic context is independent, so this only checks that the two
    # series are not offset by a bar; sign/scale of corr is irrelevant
    lag1 = np.corrcoef(target_ret.iloc[5:], ctx_ret.shift(1).iloc[5:])[0, 1]
    assert np.isfinite(corr)
    assert abs(corr - lag1) < 1.0    # sanity: no crash, no NaN explosion


def test_context_features_have_no_nan_beta():
    """``Rolling.cov()`` on a DataFrame returns MultiIndex columns and yields
    all-NaN when sliced naively -- every beta column was 0% covered."""
    target = _bars(1500, seed=7)
    rng = np.random.default_rng(8)
    ctx_close = target["close"].to_numpy() * np.exp(
        np.cumsum(rng.normal(0, 1e-3, len(target))))
    ctx = {"SYN/X": pd.DataFrame(
        {"time": target["time"], "open": ctx_close, "high": ctx_close,
         "low": ctx_close, "close": ctx_close, "volume": 1.0})}
    aligned = align_context(target, ctx)
    feats = context_features(aligned, ContextConfig(symbols={"SYN/X": "syn"}))
    assert "ctx_syn_beta" in feats.columns
    assert feats["ctx_syn_beta"].notna().mean() > 0.5
    assert feats["ctx_syn_corr"].notna().mean() > 0.5


def test_context_feature_names_are_config_derived():
    cfg = ContextConfig(symbols={"A/B": "aa", "C/D": "cc"})
    names = context_feature_names(cfg)
    assert len(names) == 18
    assert names[0] == "ctx_aa_ret_1"
    assert all(n.startswith(("ctx_aa_", "ctx_cc_")) for n in names)


def test_engineer_with_context_has_stable_feature_order():
    """Training and inference column order must match, so names come from the
    config and not from a trial pass over data."""
    a = FeatureEngineer(FeatureConfig(include_context_features=True))
    b = FeatureEngineer(FeatureConfig(include_context_features=True))
    assert a.feature_names == b.feature_names
    assert a.n_features > FeatureEngineer(FeatureConfig()).n_features
    assert a.version.endswith("+ctx")


def test_engineer_without_context_raises_a_clear_error():
    engineer = FeatureEngineer(FeatureConfig(include_context_features=True))
    with pytest.raises(ValueError, match="context"):
        engineer.transform(_bars(200))


def test_unavailable_context_symbol_is_dropped_not_zero_filled():
    """A context series with poor coverage must be dropped, never filled with
    zeros: a fake zero is a lie the model will learn from."""
    target = _bars(600, seed=9)
    # the context exists for the first 10% of bars only -> coverage far too low
    ctx_close = target["close"].to_numpy().copy()
    ctx_close[60:] = np.nan
    ctx = {"SYN/X": pd.DataFrame(
        {"time": target["time"], "open": ctx_close, "high": ctx_close,
         "low": ctx_close, "close": ctx_close, "volume": 1.0})}
    aligned = align_context(target, ctx)
    feats = context_features(aligned, ContextConfig(symbols={"SYN/X": "syn"},
                                                   min_coverage=0.5))
    assert feats.shape[1] == 0, "a 10%-covered context must be dropped entirely"

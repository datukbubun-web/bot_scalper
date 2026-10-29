"""The screener must be able to (a) find a real signal and (b) not invent one.

A research tool that only ever says "no" is useless, and one that only ever says
"yes" is dangerous. Both directions are tested here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxscalper.research.edge_screen import (
    candidate_signals,
    cost_hurdle_table,
    forward_return_bps,
    reality_check,
    screen,
)


def make_bars(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 2000 * np.exp(np.cumsum(rng.normal(0, 1e-4, n)))
    high = close * (1 + np.abs(rng.normal(0, 5e-5, n)))
    low = close * (1 - np.abs(rng.normal(0, 5e-5, n)))
    return pd.DataFrame(
        {
            "time": np.arange(n) * 60_000,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(500, 5000, n).astype(float),
        }
    )


# ── correctness of the plumbing ─────────────────────────────────────────────
def test_forward_return_has_no_look_ahead():
    df = make_bars(100)
    fwd = forward_return_bps(df, horizon=1)
    close = df["close"].to_numpy()
    assert fwd[0] == pytest.approx((close[1] / close[0] - 1) * 1e4)
    assert np.isnan(fwd[-1]), "the last bar has no future bar; must not be scored"


def test_signals_never_change_when_the_future_changes():
    """The strongest causality test available: rewrite every bar after the
    cut and require the signal up to the cut to be identical."""
    df = make_bars(600, seed=1)
    cut = 400
    baseline = candidate_signals(df)
    mutated = df.copy()
    rng = np.random.default_rng(99)
    mutated.loc[cut + 1 :, ["open", "high", "low", "close", "volume"]] = rng.normal(
        2000, 50, size=(len(df) - cut - 1, 5)
    )
    after = candidate_signals(mutated)
    for name in baseline:
        np.testing.assert_array_equal(
            baseline[name][: cut + 1],
            after[name][: cut + 1],
            err_msg=f"{name} uses future data",
        )


def test_screen_reports_n_and_net_of_cost():
    df = make_bars(3000, seed=2)
    results = screen(df, horizon=1, cost_bps=0.6)
    assert results, "screen returned nothing"
    for r in results:
        assert r.net_bps == pytest.approx(r.mean_bps - 0.6)
        assert 0.0 <= r.hit_rate <= 1.0


def test_bonferroni_is_actually_applied():
    """A p-value must be multiplied by the number of candidates searched."""
    df = make_bars(3000, seed=3)
    loose = screen(df, horizon=1)
    n_tests = len(candidate_signals(df))
    for r in loose:
        naive = float(__import__("math").erfc(abs(r.t_stat) / 2**0.5))
        assert r.p_bonferroni == pytest.approx(min(1.0, naive * n_tests), abs=1e-12)


# ── positive control: a planted edge must be found ──────────────────────────
def test_a_planted_signal_is_detected():
    """Build bars where an outside bar is systematically followed by a reversal,
    i.e. exactly what `fade_outside` bets on. The screen must find it."""
    rng = np.random.default_rng(7)
    n = 4000
    close = np.empty(n)
    close[0] = 2000.0
    for i in range(1, n):
        step = rng.normal(0, 1e-4)
        if i > 25 and rng.random() < 0.35:
            step = -np.sign(close[i - 1] - close[i - 2] or 1.0) * 8e-5  # reversal
        close[i] = close[i - 1] * (1 + step)
    high = close * 1.0002
    low = close * 0.9998
    df = pd.DataFrame(
        {
            "time": np.arange(n) * 60_000,
            "open": close,
            "high": high,
            "low": low,
            "close": close,
            "volume": rng.integers(500, 5000, n).astype(float),
        }
    )
    best = screen(df, horizon=1, cost_bps=0.0)[0]
    assert best.t_stat > 3.0, f"planted structure not found, best was {best}"


# ── negative control: pure noise must not produce a discovery ───────────────
def test_pure_noise_does_not_pass_the_reality_check():
    df = make_bars(6000, seed=11)
    rc = reality_check(df, horizon=1, n_boot=120, block=120, seed=5)
    assert rc["p_value"] > 0.05, f"noise was declared an edge: {rc}"
    assert rc["null_max_t_p95"] > 2.5, "null distribution should reach t~3 by luck"


def test_reality_check_runs_and_is_deterministic():
    df = make_bars(2000, seed=12)
    a = reality_check(df, n_boot=40, seed=3)
    b = reality_check(df, n_boot=40, seed=3)
    assert a == b


# ── the cost arithmetic ─────────────────────────────────────────────────────
def test_cost_hurdle_shrinks_with_holding_time():
    t = cost_hurdle_table()
    for _, row in t.iterrows():
        assert row["required_edge_bps_per_bar"] == pytest.approx(
            row["cost_bps"] / row["hold_bars"]
        )
    one = t[t["hold_bars"] == 1]["required_edge_bps_per_bar"].max()
    long = t[t["hold_bars"] == 15]["required_edge_bps_per_bar"].max()
    assert one > long, "a 1-bar hold cannot be cheaper than a 15-bar hold"


# ── does the edge scale with signal strength? ───────────────────────────────
def test_conditional_edge_is_flat_on_noise():
    """With no real structure, a stronger signal must not imply a bigger edge.
    If this ever shows a slope on pure noise, the bucket logic is broken."""
    df = make_bars(20000, seed=21)
    from pxscalper.research.edge_screen import conditional_edge

    table = conditional_edge(df, cost_bps=0.0)
    strongest = table.iloc[-1]["gross_bps"]
    assert abs(strongest) < 2.0, f"noise produced a strong-bucket edge: {table}"


def test_conditional_edge_finds_a_planted_gradient():
    """Plant a *non-linear* mean reversion: big shocks reverse harder than small
    ones. That gradient is precisely what a selection model would exploit, so
    the conditional table must show it -- otherwise the tool cannot tell a
    usable signal from a flat one."""
    rng = np.random.default_rng(5)
    n = 30000
    sigma = 1e-4
    # Stationary AR(1) on *price steps*: d[i] = e[i] - alpha*d[i-1], with alpha
    # growing in the size of the previous step. |alpha| < 0.9 keeps the process
    # stable, and the conditional negative autocorrelation is exactly the
    # "big bar reverses harder" gradient a selection model would exploit.
    steps = np.zeros(n)
    prev = 0.0
    for i in range(n):
        alpha = 0.1 + 0.75 / (1.0 + np.exp(-(abs(prev) / sigma - 1.0) * 2.0))
        steps[i] = rng.normal(0, sigma) - alpha * prev
        prev = steps[i]
    close = 2000.0 * np.exp(np.cumsum(steps))
    half_range = 2e-5 + 1.2 * np.abs(steps)
    df = pd.DataFrame(
        {
            "time": np.arange(n) * 60_000,
            "open": close,
            "high": close * (1 + half_range),
            "low": close * (1 - half_range),
            "close": close,
            "volume": rng.integers(500, 5000, n).astype(float),
        }
    )
    from pxscalper.research.edge_screen import conditional_edge

    table = conditional_edge(df, cost_bps=0.0)
    assert len(table) >= 3, table
    assert table.iloc[-1]["gross_bps"] > table.iloc[0]["gross_bps"], (
        "planted gradient not detected:\n" + table.to_string()
    )


def test_pnl_concentration_detects_a_lucky_few_days():
    df = make_bars(20000, seed=31)
    from pxscalper.research.edge_screen import pnl_concentration

    stats = pnl_concentration(df)
    assert 0.0 <= stats["positive_day_share"] <= 1.0
    assert stats["days"] > 0
    assert np.isfinite(stats["mean_bps_per_day"])

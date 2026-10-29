"""Tests for the event-study barrier simulator.

These exist because the first version of this code lived in a throwaway script
and shipped a bug that produced a **+73 bps per event "edge"**. The take-profit
and stop levels were computed once from the long side, so every short was
evaluated with its barriers swapped: it hit "profit" on almost any bar. The
output looked like the best result in the whole project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxscalper.backtest.event_study import simulate_barrier_trade


def _bars_from(close: np.ndarray, spread: float = 1e-6) -> pd.DataFrame:
    n = len(close)
    return pd.DataFrame(
        {
            "time": np.arange(n, dtype=np.int64) * 60_000,
            "open": close,
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": np.full(n, 1.0),
        }
    )


def _times(bars: pd.DataFrame) -> np.ndarray:
    return bars["time"].to_numpy(dtype=np.int64)


# ── the swapped-barrier bug ─────────────────────────────────────────────────
def test_short_takes_profit_when_price_falls():
    close = 1.115 * (1 - np.arange(50) * 1e-4)      # steady decline
    bars = _bars_from(close)
    trade = simulate_barrier_trade(bars, _times(bars), 0, side=-1, sl_bps=2.0,
                                   tp_bps=2.0, max_holding=40)
    assert trade is not None
    assert trade.exit_reason == "target"
    assert trade.gross_bps == pytest.approx(+2.0, abs=0.05)


def test_short_stops_out_when_price_rises():
    close = 1.115 * (1 + np.arange(50) * 1e-4)
    bars = _bars_from(close)
    trade = simulate_barrier_trade(bars, _times(bars), 0, side=-1, sl_bps=2.0,
                                   tp_bps=2.0, max_holding=40)
    assert trade is not None
    assert trade.exit_reason == "stop"
    assert trade.gross_bps == pytest.approx(-2.0, abs=0.05)


def test_long_mirrors_the_short():
    up = _bars_from(1.115 * (1 + np.arange(50) * 1e-4))
    down = _bars_from(1.115 * (1 - np.arange(50) * 1e-4))
    long_up = simulate_barrier_trade(up, _times(up), 0, 1, 2.0, 2.0, 40)
    short_down = simulate_barrier_trade(down, _times(down), 0, -1, 2.0, 2.0, 40)
    assert long_up.exit_reason == short_down.exit_reason == "target"
    assert long_up.gross_bps == pytest.approx(short_down.gross_bps, abs=0.05)


def test_barriers_are_side_specific_not_shared():
    """For the same flat market, a long and a short must NOT report the same
    outcome -- sharing barrier levels is exactly the bug."""
    rng = np.random.default_rng(0)
    close = 1.115 * np.exp(np.cumsum(rng.normal(0, 3e-5, 300)))
    bars = _bars_from(close)
    trades = [
        simulate_barrier_trade(bars, _times(bars), i, side, 5.0, 5.0, 30)
        for i in range(0, 250, 10) for side in (1, -1)
    ]
    trades = [t for t in trades if t is not None]
    longs = [t.gross_bps for t in trades if t.side > 0]
    shorts = [t.gross_bps for t in trades if t.side < 0]
    assert len(longs) == len(shorts) > 5
    assert np.mean(longs) != pytest.approx(np.mean(shorts), abs=1e-9)


# ── contract behaviour ──────────────────────────────────────────────────────
def test_return_is_what_the_chosen_side_earned():
    close = 1.115 * (1 - np.arange(50) * 1e-4)
    bars = _bars_from(close)
    long_trade = simulate_barrier_trade(bars, _times(bars), 0, 1, 2.0, 2.0, 40)
    short_trade = simulate_barrier_trade(bars, _times(bars), 0, -1, 2.0, 2.0, 40)
    assert long_trade.gross_bps < 0 < short_trade.gross_bps


def test_ambiguous_bar_resolves_pessimistically():
    """A bar touching both barriers must not be booked as a win."""
    n = 20
    close = np.full(n, 1.115)
    bars = _bars_from(close, spread=0.0)
    bars.loc[2, "high"] = close[2] * (1 + 50e-4)     # engulfs 2 bps barriers
    bars.loc[2, "low"] = close[2] * (1 - 50e-4)
    trade = simulate_barrier_trade(bars, _times(bars), 0, 1, 2.0, 2.0, 10)
    assert trade.exit_reason == "ambiguous_sl"
    assert trade.gross_bps < 0


def test_vertical_wall_closes_the_trade():
    close = np.full(60, 1.115)                        # no movement at all
    bars = _bars_from(close, spread=0.0)
    trade = simulate_barrier_trade(bars, _times(bars), 0, 1, 5.0, 5.0, 20)
    assert trade.exit_reason == "time"
    assert trade.bars_held == 20
    assert trade.gross_bps == pytest.approx(0.0, abs=1e-9)


def test_no_trade_when_series_ends():
    bars = _bars_from(np.full(10, 1.115))
    assert simulate_barrier_trade(bars, _times(bars), 9, 1, 2.0, 2.0, 5) is None


def test_invalid_barrier_is_rejected():
    bars = _bars_from(np.full(20, 1.115))
    assert simulate_barrier_trade(bars, _times(bars), 0, 1, 0.0, 2.0, 5) is None
    assert simulate_barrier_trade(bars, _times(bars), 0, 1, 2.0, -1.0, 5) is None


def test_bars_held_and_timestamps_are_consistent():
    close = 1.115 * (1 + np.arange(50) * 1e-4)
    bars = _bars_from(close)
    trade = simulate_barrier_trade(bars, _times(bars), 5, 1, 2.0, 2.0, 40)
    assert trade.entry_ms == int(bars["time"].iloc[5])
    assert trade.exit_ms > trade.entry_ms
    assert trade.bars_held == (trade.exit_ms - trade.entry_ms) // 60_000

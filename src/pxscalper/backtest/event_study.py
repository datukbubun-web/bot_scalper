"""Event-study simulation: one barrier trade per event, barrier-correct for both sides.

This exists as library code rather than a script because the first version of it
lived in a script and shipped a bug that manufactured a +73 bps "edge": the
take-profit and stop-loss levels were computed once from the long side, so every
**short** was evaluated with its barriers swapped -- it took profit on almost any
bar. Nothing about the output flagged it.

Rules encoded here
------------------
1. Barriers are side-specific. For a long the target is above entry and the stop
   below; for a short it is the mirror image. They are never shared.
2. A bar that touches both barriers is unresolvable from OHLC and resolves
   **pessimistically** (stop).
3. A trade that hits neither exits at the vertical wall, marked ``time``.
4. The simulated return is what the *chosen side* actually earned -- never the
   return of the side that happened to win.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["BarrierTrade", "simulate_barrier_trade"]


@dataclass(frozen=True)
class BarrierTrade:
    side: int
    entry_ms: int
    exit_ms: int
    gross_bps: float
    bars_held: int
    exit_reason: str


def simulate_barrier_trade(
    bars: pd.DataFrame,
    times: np.ndarray,
    entry_idx: int,
    side: int,
    sl_bps: float,
    tp_bps: float,
    max_holding: int,
) -> BarrierTrade | None:
    """Enter at ``close[entry_idx]`` and exit at the first barrier touched.

    ``sl_bps`` and ``tp_bps`` are distances from entry, always positive.
    Returns ``None`` when there is no room left in the series to simulate.
    """
    if entry_idx >= len(bars) - 1 or entry_idx < 0:
        return None
    entry_px = float(bars["close"].iloc[entry_idx])
    if entry_px <= 0 or sl_bps <= 0 or tp_bps <= 0:
        return None

    if side > 0:
        tp_px = entry_px * (1.0 + tp_bps / 1e4)
        sl_px = entry_px * (1.0 - sl_bps / 1e4)
    else:
        tp_px = entry_px * (1.0 - tp_bps / 1e4)
        sl_px = entry_px * (1.0 + sl_bps / 1e4)

    end = min(entry_idx + max_holding, len(bars) - 1)
    for j in range(entry_idx + 1, end + 1):
        hi = float(bars["high"].iloc[j])
        lo = float(bars["low"].iloc[j])
        hit_tp = hi >= tp_px if side > 0 else lo <= tp_px
        hit_sl = lo <= sl_px if side > 0 else hi >= sl_px
        if hit_tp and hit_sl:
            px, reason = sl_px, "ambiguous_sl"
        elif hit_sl:
            px, reason = sl_px, "stop"
        elif hit_tp:
            px, reason = tp_px, "target"
        else:
            continue
        return BarrierTrade(
            side=side,
            entry_ms=int(times[entry_idx]),
            exit_ms=int(times[j]),
            gross_bps=side * (px / entry_px - 1.0) * 1e4,
            bars_held=j - entry_idx,
            exit_reason=reason,
        )

    j = end
    px = float(bars["close"].iloc[j])
    return BarrierTrade(
        side=side,
        entry_ms=int(times[entry_idx]),
        exit_ms=int(times[j]),
        gross_bps=side * (px / entry_px - 1.0) * 1e4,
        bars_held=j - entry_idx,
        exit_reason="time",
    )

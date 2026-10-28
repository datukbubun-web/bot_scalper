"""Tradeable targets: what happens *first*, not where price ends up.

Why the old target was the wrong question
-----------------------------------------
``volatility_scaled_labels`` asks "is the forward return over the next 5 bars
positive or negative", i.e. where price *ends*. A scalper never holds to the end:
it exits at a barrier. Worse, that target counts a move that would have stopped
out first as a win, so a model can score well on the label and still lose money.

What replaces it
----------------
A **triple-barrier** target (López de Prado, *Advances in Financial Machine
Learning*, ch. 3) labelled from the *trade's* point of view:

For a long candidate at bar ``t`` with volatility-scaled barriers::

    upper = entry * (1 + tp_atr  * ATR/entry)      <- take-profit
    lower = entry * (1 - sl_atr  * ATR/entry, 0)   <- stop-loss
    wall  = entry * (1 + 0)                        <- vertical barrier at t+max_holding

    y = +1   upper touched before lower        (a win *as traded*)
    y = -1   lower touched before upper        (a loss *as traded*)
    y =  0   neither: exit at the wall         (flat/timeout)

Directed (side-aware)
---------------------
The honest version for a bot that chooses direction. For each bar we ask "if I
went long here, what happened?" and "if I went short here, what happened?" and
the three classes become:

    0 = no trade           (neither side clears its costs)
    1 = short is the trade (short TP hit first AND nets > costs)
    2 = long is the trade  (long TP hit first AND nets > costs)

Bars where *both* sides would have won (a whipsaw that touches both barriers) are
excluded rather than guessed: the intra-bar ordering is unknowable from OHLC.

Anti-leakage
------------
Every label at ``t`` is computed only from bars ``> t``. The final
``max_holding`` bars have an unobservable outcome and are dropped. Shifted
statistics (``rvol_mean``, ``rvol_std``) use only past bars.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "BarrierConfig",
    "barrier_outcome",
    "directed_barrier_labels",
    "barrier_stats",
]


@dataclass
class BarrierConfig:
    """Barrier geometry, expressed in ATR multiples (scale-free across symbols).

    ``cost_bps`` is not a detail: a +1 bps scalp on a 0.49 bps spread is a real
    edge, a +1 bps scalp on a 4 bps spread is none. It is subtracted from every
    winning outcome, so barriers that cannot pay for themselves are labelled
    "no trade" instead of "win".
    """

    tp_atr: float = 1.5
    sl_atr: float = 1.0
    max_holding: int = 30
    atr_window: int = 14
    cost_bps: float = 0.6        # round-trip: spread + fees, measured per venue
    min_reward_bps: float = 0.0  # optional extra hurdle on a winning trade
    allow_both_win: bool = False  # if both sides clear, label as ambiguous


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def barrier_outcome(
    bars: pd.DataFrame, side: int, config: BarrierConfig | None = None
) -> pd.DataFrame:
    """First-touch outcome of a single trade, vectorised over bars.

    Returns a frame indexed like ``bars`` with

    ``label``       +1 take-profit first, -1 stop-loss first, 0 vertical wall
    ``bars_held``   bars until the exit (``max_holding`` when it times out)
    ``ret``         gross return at the exit, before costs
    ``hit_upper``   the upper barrier was touched at any point (for diagnostics)
    ``hit_lower``   the lower barrier was touched at any point
    ``ambiguous``   both barriers touched inside the same bar -- unknowable
    """
    cfg = config or BarrierConfig()
    close = bars["close"].to_numpy(dtype=np.float64)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)
    atr = _atr(bars["high"], bars["low"], bars["close"], cfg.atr_window).to_numpy()

    n = len(bars)
    label = np.full(n, np.nan)
    held = np.full(n, float(cfg.max_holding))
    ret = np.full(n, np.nan)
    hit_up = np.zeros(n, dtype=bool)
    hit_dn = np.zeros(n, dtype=bool)
    ambiguous = np.zeros(n, dtype=bool)

    for i in range(n - 1):
        a = atr[i]
        entry = close[i]
        if not np.isfinite(a) or a <= 0 or entry <= 0:
            continue
        dist_tp = cfg.tp_atr * a
        dist_sl = cfg.sl_atr * a
        end = min(i + cfg.max_holding, n - 1)
        if end <= i:
            continue
        upper = entry + dist_tp
        lower = entry - dist_sl

        exit_price = close[end]
        outcome = 0.0
        exit_idx = end
        for j in range(i + 1, end + 1):
            up_touch = high[j] >= upper
            dn_touch = low[j] <= lower
            if up_touch:
                hit_up[i] = True
            if dn_touch:
                hit_dn[i] = True
            if up_touch and dn_touch:
                # Both barriers inside one bar: OHLC cannot tell us the order.
                ambiguous[i] = True
                outcome = -1.0 if side > 0 else 1.0   # pessimistic convention
                exit_idx = j
                exit_price = lower if side > 0 else upper
                break
            if side > 0:
                if dn_touch:
                    outcome, exit_idx, exit_price = -1.0, j, lower
                    break
                if up_touch:
                    outcome, exit_idx, exit_price = 1.0, j, upper
                    break
            else:
                if up_touch:
                    outcome, exit_idx, exit_price = -1.0, j, upper
                    break
                if dn_touch:
                    outcome, exit_idx, exit_price = 1.0, j, lower
                    break
        label[i] = outcome
        held[i] = exit_idx - i
        ret[i] = side * (exit_price / entry - 1.0)

    tail = cfg.max_holding
    label[max(n - tail, 0) :] = np.nan
    ret[max(n - tail, 0) :] = np.nan
    return pd.DataFrame(
        {
            "label": label,
            "bars_held": held,
            "ret": ret,
            "hit_upper": hit_up,
            "hit_lower": hit_dn,
            "ambiguous": ambiguous,
        },
        index=bars.index,
    )


def directed_barrier_labels(
    bars: pd.DataFrame, config: BarrierConfig | None = None
) -> pd.DataFrame:
    """Side-aware 3-class target: 0 = no trade, 1 = short, 2 = long.

    A side is only a candidate if its take-profit beats the round-trip cost::

        tp_bps = tp_atr * ATR / price * 1e4
        tradable = tp_bps > cost_bps + min_reward_bps

    Both sides tradable is a whipsaw bar: excluded, because we cannot know which
    barrier came first.
    """
    cfg = config or BarrierConfig()
    long_out = barrier_outcome(bars, side=1, config=cfg)
    short_out = barrier_outcome(bars, side=-1, config=cfg)
    close = bars["close"]
    atr = _atr(bars["high"], bars["low"], close, cfg.atr_window)
    tp_bps = (cfg.tp_atr * atr / close) * 1e4
    hurdle = cfg.cost_bps + cfg.min_reward_bps
    tradable = tp_bps > hurdle

    long_win = (long_out["label"] > 0) & tradable
    short_win = (short_out["label"] > 0) & tradable

    both = long_win & short_win
    none = ~long_win & ~short_win
    y = pd.Series(np.nan, index=bars.index)
    y[long_win & ~both] = 2
    y[short_win & ~both] = 1
    y[both & cfg.allow_both_win] = 2     # documented, not the default
    y[none] = 0

    # Return of *each* side, for every bar, net of costs.
    #
    # Both are needed because a backtest must score the side the model CHOSE, not
    # the side the future happened to favour. Using only the realised label
    # return gives every model -- including a constant one -- the oracle's PnL;
    # that bug produced a 97% "hit rate" and +9.8 bps per trade for a model that
    # predicted nothing.
    ret_long_bps = long_out["ret"].to_numpy() * 1e4 - cfg.cost_bps
    ret_short_bps = short_out["ret"].to_numpy() * 1e4 - cfg.cost_bps
    # a side that cannot clear its costs is not a tradeable side at all
    ret_long_bps = np.where(tradable.to_numpy(), ret_long_bps, np.nan)
    ret_short_bps = np.where(tradable.to_numpy(), ret_short_bps, np.nan)

    out = pd.DataFrame(
        {
            "y": y,
            "ret_long_bps": ret_long_bps,
            "ret_short_bps": ret_short_bps,
            "tp_bps": tp_bps,
            "atr": atr,
            "long_win": long_win,
            "short_win": short_win,
            "ambiguous": both,
            "bars_to_exit": np.where(y == 2, long_out["bars_held"],
                                     np.where(y == 1, short_out["bars_held"], 0.0)),
        },
        index=bars.index,
    )
    return out[out["y"].notna()]


def barrier_stats(labels: pd.DataFrame) -> dict:
    """Class balance, hurdle pass rate and the honest expectation per class.

    ``expectancy_bps_classK`` is the mean net return of that side on the bars
    where it was the correct target -- perfect foresight, so an upper bound and
    explicitly not a forecast.
    """
    y = labels["y"]
    counts = {int(k): int(v) for k, v in y.value_counts().sort_index().items()}
    total = max(len(y), 1)
    stats: dict = {
        "n": int(len(y)),
        "counts": counts,
        "share": {k: round(v / total, 4) for k, v in counts.items()},
        "hurdle_pass_rate": round(float((labels["tp_bps"] > 0).mean()), 4),
        "median_tp_bps": round(float(labels["tp_bps"].median()), 3),
        "mean_bars_to_exit": round(float(labels["bars_to_exit"].mean()), 2),
    }
    for cls, col in ((1, "ret_short_bps"), (2, "ret_long_bps")):
        mask = y == cls
        if mask.any() and col in labels:
            stats[f"expectancy_bps_class{cls}"] = round(
                float(labels.loc[mask, col].mean()), 3
            )
    return stats

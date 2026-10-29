"""A pre-registered search for an exploitable edge, and an honest verdict.

Doing statistics on markets without multiple-testing control is how people
"discover" a strategy that loses money. Twenty candidate signals tested at
p<0.05 naive will produce one false positive *by construction*. So this module
does three things that most research code does not:

1. **Pre-registers the candidates.** The list is fixed in code before results are
   seen; nothing is added because it "looks promising".
2. **Reality-checks the maximum.** Under the null, the best of N candidates has a
   high t-stat by luck. A block bootstrap gives the null distribution of the
   *maximum* t across all candidates, so the comparison is against the right
   benchmark (White's Reality Check), not against zero.
3. **Prices the hurdle in the answer.** A signal whose mean edge is 0.3 bps is
   not an edge when the round trip costs 0.6 bps, however significant it is.

The forward return is always measured from the *close of the bar where the signal
is computable* to the close h bars later. A signal may only use data up to and
including its own bar; nothing is allowed to peek.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

BPS = 1e4


# --------------------------------------------------------------------- returns
def forward_return_bps(df: pd.DataFrame, horizon: int = 1) -> np.ndarray:
    """Close(t) -> Close(t+h), in bps. The last `horizon` entries are NaN."""
    close = df["close"].to_numpy(dtype=float)
    fwd = np.full(close.shape, np.nan)
    if horizon < len(close):
        fwd[:-horizon] = (close[horizon:] / close[:-horizon] - 1.0) * BPS
    return fwd


# -------------------------------------------------------------------- signals
# Each signal returns an array of -1/0/+1 aligned to df rows and uses ONLY data
# at or before that row. The list below is the pre-registered set.
def _flat_like(df: pd.DataFrame) -> np.ndarray:
    return np.zeros(len(df), dtype=float)


def _runs(df: pd.DataFrame, k: int = 3) -> np.ndarray:
    """+k / -k for k consecutive up/down closes (else 0). Causal by construction."""
    up = (df["close"].diff() > 0).to_numpy()
    run = np.zeros(len(up), dtype=float)
    for i in range(1, len(up)):
        if up[i]:
            run[i] = max(1.0, run[i - 1] + 1) if run[i - 1] > 0 else 1.0
        elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
            run[i] = min(-1.0, run[i - 1] - 1) if run[i - 1] < 0 else -1.0
    return run


def _rolling_max_prev(s: pd.Series, window: int) -> np.ndarray:
    """Highest value in the *previous* `window` bars (shifted: no peek)."""
    return s.rolling(window).max().shift(1).to_numpy()


def candidate_signals(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """The pre-registered candidates. Names are stable so results are comparable."""
    out: dict[str, np.ndarray] = {}
    close = df["close"]
    high, low, vol = df["high"], df["low"], df["volume"]

    # 1-24: drift by UTC hour (long during that hour only)
    hours = pd.to_datetime(df["time"], unit="ms", utc=True).dt.hour
    for h in range(24):
        sig = _flat_like(df)
        sig[(hours == h).to_numpy()] = 1.0
        out[f"long_hour_{h:02d}"] = sig

    # 25-29: drift by day of week
    dow = pd.to_datetime(df["time"], unit="ms", utc=True).dt.dayofweek
    for d in range(5):
        sig = _flat_like(df)
        sig[(dow == d).to_numpy()] = 1.0
        out[f"long_dow_{d}"] = sig

    # 30-33: serial dependency -- follow or fade a run of 3
    run3 = _runs(df, 3)
    out["fade_run3"] = np.where(run3 >= 3, -1.0, np.where(run3 <= -3, 1.0, 0.0))
    out["follow_run3"] = np.where(run3 >= 3, 1.0, np.where(run3 <= -3, -1.0, 0.0))
    run5 = _runs(df, 5)
    out["fade_run5"] = np.where(run5 >= 5, -1.0, np.where(run5 <= -5, 1.0, 0.0))
    out["follow_run5"] = np.where(run5 >= 5, 1.0, np.where(run5 <= -5, -1.0, 0.0))

    # 34-37: volume spike -> follow or fade the bar's own direction
    vma = vol.rolling(50).median().shift(1)
    spike = (vol > 3 * vma).to_numpy()
    bar_dir = np.sign(close.diff()).to_numpy()
    out["follow_volspike"] = np.where(spike, bar_dir, 0.0)
    out["fade_volspike"] = np.where(spike, -bar_dir, 0.0)

    # 38-41: range breakout / breakdown over the previous 20 bars
    hi20 = _rolling_max_prev(high, 20)
    lo20 = _rolling_max_prev(-low, 20) * -1.0
    c = close.to_numpy()
    out["breakout_20"] = np.where(c > hi20, 1.0, 0.0)
    out["breakdown_20"] = np.where(c < lo20, -1.0, 0.0)

    # 42-45: mean reversion after a stretched move (z of 20-bar return)
    r20 = (close / close.shift(20) - 1.0)
    z = ((r20 - r20.rolling(200).mean()) / r20.rolling(200).std()).to_numpy()
    out["fade_z2"] = np.where(z > 2, -1.0, np.where(z < -2, 1.0, 0.0))
    out["follow_z2"] = np.where(z > 2, 1.0, np.where(z < -2, -1.0, 0.0))

    # 46-47: inside-bar / outside-bar follow-through
    rng = (high - low)
    outside = (rng > rng.rolling(20).mean().shift(1)).to_numpy()
    out["follow_outside"] = np.where(outside, bar_dir, 0.0)
    out["fade_outside"] = np.where(outside, -bar_dir, 0.0)

    return out


# ------------------------------------------------------------------- statistics
@dataclass
class ScreenResult:
    name: str
    n: int
    mean_bps: float
    t_stat: float
    hit_rate: float
    net_bps: float          # mean minus round-trip cost
    p_bonferroni: float

    def __str__(self) -> str:
        return (
            f"{self.name:22s} n={self.n:6d} mean={self.mean_bps:+7.3f}bps "
            f"t={self.t_stat:+6.2f} hit={self.hit_rate:5.1%} "
            f"net={self.net_bps:+7.3f} p_bonf={self.p_bonferroni:.3f}"
        )


def _two_sided_p(t: float, dof: int) -> float:
    """Normal approximation; exact enough for n in the thousands."""
    if dof <= 0:
        return 1.0
    z = abs(t)
    # Abramowitz-Stegun style erfc via numpy
    return float(np.math.erfc(z / np.sqrt(2.0))) if hasattr(np, "math") else 1.0


def _p_from_t(t: float) -> float:
    from math import erfc, sqrt

    return erfc(abs(t) / sqrt(2.0))


def screen(
    df: pd.DataFrame,
    horizon: int = 1,
    cost_bps: float = 0.6,
    min_trades: int = 100,
) -> list[ScreenResult]:
    """Evaluate every pre-registered candidate against forward returns."""
    fwd = forward_return_bps(df, horizon)
    candidates = candidate_signals(df)
    n_tests = len(candidates)
    results: list[ScreenResult] = []

    for name, sig in candidates.items():
        mask = (sig != 0) & np.isfinite(fwd)
        n = int(mask.sum())
        if n < min_trades:
            continue
        pnl = sig[mask] * fwd[mask]
        mean = float(pnl.mean())
        sd = float(pnl.std(ddof=1))
        t = mean / (sd / np.sqrt(n)) if sd > 0 else 0.0
        results.append(
            ScreenResult(
                name=name,
                n=n,
                mean_bps=mean,
                t_stat=t,
                hit_rate=float((pnl > 0).mean()),
                net_bps=mean - cost_bps,
                p_bonferroni=min(1.0, _p_from_t(t) * n_tests),
            )
        )
    results.sort(key=lambda r: -r.t_stat)
    return results


# ---------------------------------------------------------------- reality check
def reality_check(
    df: pd.DataFrame,
    horizon: int = 1,
    n_boot: int = 300,
    block: int = 120,
    seed: int = 7,
    min_trades: int = 100,
) -> dict:
    """White's Reality Check: is the BEST candidate better than luck?

    Shifting the forward-return series in circular blocks preserves the
    autocorrelation and volatility clustering of the real data while destroying
    any genuine signal-to-return relationship. Running every candidate on each
    shuffled series gives the distribution of the *maximum* t-statistic under
    the null. If the observed maximum sits inside that distribution, the search
    found nothing.
    """
    fwd = forward_return_bps(df, horizon)
    candidates = candidate_signals(df)
    rng = np.random.default_rng(seed)
    n = len(df)

    def max_t(fwd_series: np.ndarray) -> float:
        best = -np.inf
        for sig in candidates.values():
            mask = (sig != 0) & np.isfinite(fwd_series)
            k = int(mask.sum())
            if k < min_trades:
                continue
            pnl = sig[mask] * fwd_series[mask]
            sd = float(pnl.std(ddof=1))
            if sd <= 0:
                continue
            best = max(best, float(pnl.mean()) / (sd / np.sqrt(k)))
        return float(best)

    observed = max_t(fwd)

    null: list[float] = []
    starts = np.arange(0, n, block)
    for _ in range(n_boot):
        idx = np.concatenate(
            [np.arange(s, s + block) % n for s in rng.permutation(starts)]
        )[:n]
        null.append(max_t(fwd[idx]))
    null_arr = np.asarray(null)
    p_value = float((null_arr >= observed).mean())
    return {
        "observed_max_t": observed,
        "null_max_t_p95": float(np.quantile(null_arr, 0.95)),
        "null_max_t_median": float(np.median(null_arr)),
        "p_value": p_value,
        "n_boot": n_boot,
        "block": block,
        "n_candidates": len(candidates),
    }


def cost_hurdle_table(horizons=(1, 3, 5, 15)) -> pd.DataFrame:
    """How much edge a trade must carry to beat its own cost, by holding time.

    A scalper's cost is paid per trade, so the shorter the hold, the higher the
    required per-bar edge. This is the arithmetic that decides whether a signal
    is worth pursuing at all.
    """
    rows = []
    for h in horizons:
        for cost in (0.3, 0.6, 1.5):
            rows.append(
                {
                    "hold_bars": h,
                    "cost_bps": cost,
                    "required_edge_bps_per_bar": cost / h,
                    "required_hit_rate_if_1to1": 0.5 + cost / (2 * h * 8.0),
                }
            )
    return pd.DataFrame(rows)


# ------------------------------------------------- does the edge scale with it?
def conditional_edge(
    df: pd.DataFrame,
    horizon: int = 1,
    cost_bps: float = 0.6,
    buckets: tuple[tuple[float, float], ...] = (
        (1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, float("inf")),
    ),
) -> pd.DataFrame:
    """Does `fade_outside` earn more on more extreme bars?

    This is the question that decides whether a model can help. If the edge is
    flat across signal strength, selection adds nothing and a model is
    decoration. If it grows, selection has something to work with -- but the
    result still has to clear the cost, which is the harder test.

    t is computed per DAY, not per bar: bars inside a multi-bar hold overlap,
    and counting each one inflates t by 2-4x (seen: +12.0 vs +2.47).
    """
    close = df["close"]
    fwd = forward_return_bps(df, horizon)
    rng = df["high"] - df["low"]
    ratio = (rng / rng.rolling(20).mean().shift(1)).to_numpy()
    bar_dir = np.sign(close.diff()).to_numpy()
    signal = np.where(np.isfinite(ratio), -bar_dir, 0.0)
    pnl = signal * fwd

    if "time" in df.columns:
        day = pd.to_datetime(df["time"], unit="ms", utc=True).dt.date.to_numpy()
    else:
        day = np.arange(len(df)) // 1440

    rows = []
    for lo, hi in buckets:
        mask = np.isfinite(pnl) & (ratio >= lo) & (ratio < hi)
        n = int(mask.sum())
        if n < 300:
            continue
        per_day = pd.DataFrame({"day": day[mask], "p": pnl[mask]}).groupby("day")["p"].mean()
        sd = float(per_day.std(ddof=1))
        t = float(per_day.mean() / (sd / np.sqrt(len(per_day)))) if sd > 0 else 0.0
        rows.append(
            {
                "range_x_avg": f"{lo:g}-{hi if np.isfinite(hi) else 'inf'}",
                "n": n,
                "trades_per_day": round(n / max(len(per_day), 1), 1),
                "days": len(per_day),
                "gross_bps": float(pnl[mask].mean()),
                "t_per_day": t,
                "net_bps": float(pnl[mask].mean()) - cost_bps,
                "hit_rate": float((pnl[mask] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def pnl_concentration(df: pd.DataFrame, top_n: int = 10, horizon: int = 1) -> dict:
    """How much of the result comes from a handful of days?

    A real edge shows up across many days. An edge that lives in 10 days out of
    174 is a handful of lucky afternoons, and it will not repeat.
    """
    close = df["close"]
    fwd = forward_return_bps(df, horizon)
    rng = df["high"] - df["low"]
    ratio = (rng / rng.rolling(20).mean().shift(1)).to_numpy()
    signal = np.where(np.isfinite(ratio) & (ratio >= 3.0), -np.sign(close.diff()).to_numpy(), 0.0)
    pnl = signal * fwd
    mask = np.isfinite(pnl) & (signal != 0)
    day = pd.to_datetime(df["time"], unit="ms", utc=True).dt.date.to_numpy()
    per_day = pd.DataFrame({"day": day[mask], "p": pnl[mask]}).groupby("day")["p"].mean()
    total = float(per_day.sum())
    best = float(per_day.nlargest(top_n).sum())
    return {
        "days": len(per_day),
        "mean_bps_per_day": float(per_day.mean()),
        "positive_day_share": float((per_day > 0).mean()),
        "top_days_share_of_total": best / total if total else float("nan"),
        "mean_without_top_days": (total - best) / len(per_day),
    }

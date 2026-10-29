"""Event breakout: trade the volatility expansion after a macro release.

Previous finding (docs/CALENDAR_REPORT.md): the calendar predicts *when*
volatility arrives (pre-NFP vol 9.4x normal) but not direction. So the honest
hypothesis to test is not "predict the number" but "let the release pick the
direction, then follow the expansion".

Design decisions that keep this from being a fishing expedition
--------------------------------------------------------------
1. **A small, pre-registered variant set** (4 variants, listed below). All are
   reported, including the ones that lose. Adding variants after seeing the
   result is how a backtest lies.
2. **The unit of observation is the event, not the bar.** Bars inside one release
   are one outcome; treating them as independent would multiply n by ~60 and
   manufacture significance.
3. **A placebo control.** The same procedure applied to random times matched on
   hour-of-day and weekday, excluding real events. If the release matters, the
   real arm must beat the placebo arm -- a positive number alone proves nothing,
   because any breakout system on a trending series can look fine.
4. **Cost sensitivity** at 0 / 0.6 / 2.0 bps round trip.

    python scripts/training/eval_event_breakout.py --symbol XAU/USD
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.backtest.event_study import simulate_barrier_trade  # noqa: E402
from pxscalper.features.calendar import MacroCalendar  # noqa: E402
from pxscalper.utils import RepoPaths, seed_everything, setup_logging  # noqa: E402

BAR_MS_1M = 60_000


@dataclass(frozen=True)
class TradeResult:
    variant: str
    event_kind: str
    event_ms: int
    side: int
    entry_ms: int
    exit_ms: int
    gross_bps: float
    bars_held: int
    exit_reason: str


def _true_range_bps(bars: pd.DataFrame) -> np.ndarray:
    prev_close = bars["close"].shift(1)
    tr = np.maximum(
        bars["high"] - bars["low"],
        np.maximum((bars["high"] - prev_close).abs(), (bars["low"] - prev_close).abs()),
    )
    return (tr / bars["close"] * 1e4).to_numpy()


def _simulate(bars, times, entry_idx, side, sl_bps, tp_bps, max_holding):
    """Thin wrapper over the library simulator (side-correct barriers)."""
    trade = simulate_barrier_trade(bars, times, entry_idx, side, sl_bps, tp_bps,
                                   max_holding)
    if trade is None:
        return None
    return trade.gross_bps, trade.bars_held, trade.exit_ms, trade.exit_reason


def run_momentum(bars, times, event_ms, kind, follow: bool, sl_mult: float = 1.0,
                 tp_mult: float = 1.5, max_holding: int = 60) -> TradeResult | None:
    """Trade the direction of the first bar that contains the release."""
    idx = int(np.searchsorted(times, event_ms, side="left"))
    if idx >= len(bars) - 2:
        return None
    first = bars.iloc[idx]
    move = float(first["close"]) - float(first["open"])
    if move == 0:
        return None
    side = int(np.sign(move)) * (1 if follow else -1)
    # scale the barrier to the release bar itself, not to the pre-event ATR:
    # that is the whole point of an event strategy
    bar_range_bps = max((float(first["high"]) - float(first["low"])) / float(first["close"]) * 1e4, 1.0)
    out = _simulate(bars, times, idx, side, sl_mult * bar_range_bps,
                    tp_mult * bar_range_bps, max_holding)
    if out is None:
        return None
    gross, held, exit_ms, reason = out
    return TradeResult("momentum" if follow else "reversal", kind, event_ms, side,
                       int(times[idx]), exit_ms, gross, held, reason)


def run_breakout(bars, times, event_ms, kind, box_minutes: int = 5,
                 sl_mult: float = 1.0, tp_mult: float = 1.5,
                 max_holding: int = 60) -> TradeResult | None:
    """Wait for the release to define a box, then trade the break of it."""
    idx = int(np.searchsorted(times, event_ms, side="left"))
    box_end = int(np.searchsorted(times, event_ms + box_minutes * BAR_MS_1M, side="left"))
    if idx >= len(bars) - 2 or box_end <= idx or box_end >= len(bars) - 2:
        return None
    box = bars.iloc[idx:box_end]
    hi, lo = float(box["high"].max()), float(box["low"].min())
    height_bps = (hi - lo) / hi * 1e4
    if height_bps <= 0:
        return None
    horizon_end = min(box_end + max_holding, len(bars) - 1)
    for j in range(box_end, horizon_end):
        px = float(bars["close"].iloc[j])
        if px > hi:
            side = 1
        elif px < lo:
            side = -1
        else:
            continue
        out = _simulate(bars, times, j, side, sl_mult * height_bps,
                        tp_mult * height_bps, max_holding)
        if out is None:
            return None
        gross, held, exit_ms, reason = out
        return TradeResult(f"breakout{box_minutes}", kind, event_ms, side,
                           int(times[j]), exit_ms, gross, held, reason)
    return None


VARIANTS = {
    # pre-registered: four, all reported
    "momentum": lambda b, t, e, k: run_momentum(b, t, e, k, follow=True),
    "reversal": lambda b, t, e, k: run_momentum(b, t, e, k, follow=False),
    "breakout5": lambda b, t, e, k: run_breakout(b, t, e, k, box_minutes=5),
    "breakout15": lambda b, t, e, k: run_breakout(b, t, e, k, box_minutes=15),
}


def placebo_times(bars, calendar, events: list[int], rng, days_options=(7, 14, 21, 28)) -> list[int]:
    """Same hour-of-day and weekday, shifted by whole weeks, away from any event."""
    lo, hi = int(bars["time"].iloc[0]), int(bars["time"].iloc[-1])
    all_events = np.array([e.time_ms for e in calendar.events], dtype=np.int64)
    out: list[int] = []
    for e in events:
        for shift in rng.permutation(days_options):
            cand = int(e + shift * 86_400_000)
            if not (lo + 86_400_000 < cand < hi - 86_400_000):
                continue
            # reject if a real release lands within 3 hours
            if np.min(np.abs(all_events - cand)) < 3 * 3_600_000:
                continue
            out.append(cand)
            break
    return out


def summarise(pnls: list[float]) -> dict:
    arr = np.array([p for p in pnls if np.isfinite(p)], dtype=float)
    if arr.size < 3:
        return {"n": int(arr.size), "mean_bps": float("nan"), "t": float("nan"),
                "p": float("nan"), "hit": float("nan")}
    t, p = stats.ttest_1samp(arr, 0.0)
    return {
        "n": int(arr.size),
        "mean_bps": float(arr.mean()),
        "median_bps": float(np.median(arr)),
        "std_bps": float(arr.std(ddof=1)),
        "hit": float((arr > 0).mean()),
        "t": float(t),
        "p": float(p),
        "total_bps": float(arr.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="XAU/USD")
    ap.add_argument("--bars", default="1M")
    ap.add_argument("--cost-bps", type=float, default=0.6)
    ap.add_argument("--out", default="artifacts/event")
    args = ap.parse_args()

    setup_logging("WARNING")
    seed_everything(11)
    rng = np.random.default_rng(11)
    paths = RepoPaths().ensure()
    bars = DataPipeline(cache_dir=paths.data).load(args.symbol, args.bars)
    if bars is None:
        print(f"no cached data for {args.symbol} {args.bars}")
        return 1
    bars = bars.reset_index(drop=True)
    times = bars["time"].to_numpy(dtype=np.int64)
    calendar = MacroCalendar()
    lo, hi = int(times[0]), int(times[-1])
    events = [e.time_ms for e in calendar.events if lo < e.time_ms < hi]
    span_days = (hi - lo) / 86_400_000
    print(f"=== {args.symbol} {args.bars}: {len(bars)} bar, {span_days:.0f} hari ===")
    print(f"event dalam rentang: {len(events)} "
          f"({', '.join(f'{k}={calendar.count(k, lo, hi)}' for k in calendar.kinds())})")
    print(f"biaya: {args.cost_bps} bps round-trip\n")

    results: dict = {}
    rows: list[TradeResult] = []
    for name, fn in VARIANTS.items():
        real, placebo = [], []
        for e in events:
            kind = next(x.kind for x in calendar.events if x.time_ms == e)
            r = fn(bars, times, e, kind)
            if r is not None:
                real.append(r.gross_bps - args.cost_bps)
                rows.append(r)
        placebos = placebo_times(bars, calendar, events, rng)
        for p_ms in placebos:
            r = fn(bars, times, p_ms, "placebo")
            if r is not None:
                placebo.append(r.gross_bps - args.cost_bps)
        s_real, s_place = summarise(real), summarise(placebo)
        diff = float("nan")
        p_diff = float("nan")
        if len(real) >= 3 and len(placebo) >= 3:
            tt = stats.ttest_ind(real, placebo, equal_var=False)
            diff, p_diff = float(s_real["mean_bps"] - s_place["mean_bps"]), float(tt.pvalue)
        results[name] = {"real": s_real, "placebo": s_place,
                         "diff_bps": diff, "p_diff": p_diff}
        print(f"{name:11s} | event n={s_real['n']:2d} mean {s_real['mean_bps']:+7.2f} bps "
              f"hit {s_real['hit']:.2f} t={s_real['t']:+5.2f} p={s_real['p']:.3f} "
              f"|| plasebo n={s_place['n']:2d} mean {s_place['mean_bps']:+7.2f} "
              f"|| selisih {diff:+7.2f} p={p_diff:.3f}")

    # ---------------------------------------------------------- cost sensitivity
    best = max(results, key=lambda k: results[k]["real"]["mean_bps"])
    print(f"\nSensitivitas biaya untuk varian terbaik ({best}):")
    print(f"  {'biaya':>8s} {'mean bps/event':>15s} {'total bps':>10s}")
    for cost in (0.0, 0.6, 1.0, 2.0, 5.0):
        pnls = [r.gross_bps - cost for r in rows if r.variant == best]
        if pnls:
            print(f"  {cost:8.1f} {np.mean(pnls):+15.2f} {np.sum(pnls):+10.1f}")

    # ------------------------------------------------------------- per event kind
    print("\nRincian per jenis event (variant terbaik):")
    for kind in ("nfp", "cpi", "fomc"):
        pnls = [r.gross_bps - args.cost_bps for r in rows
                if r.variant == best and r.event_kind == kind]
        if pnls:
            print(f"  {kind:5s} n={len(pnls):2d} mean {np.mean(pnls):+7.2f} bps | "
                  f"per event: {' '.join(f'{p:+.0f}' for p in pnls)}")

    # --------------------------------------------------------------- power check
    s = results[best]["real"]
    if np.isfinite(s.get("std_bps", np.nan)) and s["mean_bps"] != 0:
        need = int((s["std_bps"] * 3 / abs(s["mean_bps"])) ** 2)
        print(f"\nPower: untuk mean {s['mean_bps']:+.2f} bps dan std {s['std_bps']:.1f}, "
              f"butuh n≈{need} event agar 3-sigma. Tersedia {s['n']}.")
        if need > s["n"]:
            print(f"       ~{need * 30 / 12:.0f} bulan data (30 hari/event/bulan) — "
                  f"tersedia {span_days / 30.4:.1f} bulan.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.symbol.replace('/', '')}_{args.bars}"
    (out_dir / f"{tag}.json").write_text(json.dumps(
        {"symbol": args.symbol, "bars": args.bars, "span_days": span_days,
         "n_events": len(events), "cost_bps": args.cost_bps, "variants": results},
        indent=2))
    pd.DataFrame([r.__dict__ for r in rows]).to_csv(out_dir / f"{tag}_trades.csv", index=False)
    print(f"\nwritten -> {out_dir / (tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

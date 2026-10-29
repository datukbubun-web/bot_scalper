"""Does the macro calendar carry *tradeable* information for XAU/USD?

The calendar tells you **when** volatility arrives, not which way price goes, so
testing it as a direction predictor is testing the wrong hypothesis. This script
tests the three things it can actually claim:

1. **Regime** -- is volatility around releases measurably different?
2. **Tradability** -- does expectancy change inside those windows, per side?
3. **Enough data?** -- how many observations would be needed before the effect
   could be told apart from noise. A release window is a handful of bars; the
   answer is usually "more history than exists".

    python scripts/training/eval_calendar.py --symbol XAU/USD --bars 5M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.features.calendar import MacroCalendar, calendar_features  # noqa: E402
from pxscalper.features.targets import BarrierConfig, directed_barrier_labels  # noqa: E402
from pxscalper.utils import RepoPaths, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="XAU/USD")
    ap.add_argument("--bars", default="5M")
    ap.add_argument("--n-bars", type=int, default=20000)
    ap.add_argument("--horizon", type=int, default=40)
    ap.add_argument("--cost-bps", type=float, default=0.6)
    args = ap.parse_args()

    setup_logging("WARNING")
    paths = RepoPaths().ensure()
    bars = DataPipeline(cache_dir=paths.data).load(args.symbol, args.bars)
    if bars is None:
        print(f"no cached data for {args.symbol} {args.bars}")
        return 1
    bars = bars.tail(args.n_bars).reset_index(drop=True)
    cal = MacroCalendar()
    feats = calendar_features(bars["time"].to_numpy())
    labels = directed_barrier_labels(
        bars, BarrierConfig(max_holding=args.horizon, cost_bps=args.cost_bps))
    df = labels.join(feats)
    span_days = (bars["time"].iloc[-1] - bars["time"].iloc[0]) / 86_400_000
    print(f"=== {args.symbol} {args.bars}: {len(bars)} bar, {span_days:.0f} hari ===\n")
    print(f"event dalam rentang: {cal.count('nfp', bars['time'].iloc[0], bars['time'].iloc[-1])} NFP, "
          f"{cal.count('cpi', bars['time'].iloc[0], bars['time'].iloc[-1])} CPI, "
          f"{cal.count('fomc', bars['time'].iloc[0], bars['time'].iloc[-1])} FOMC\n")

    fwd_vol = np.log(bars["close"]).diff().rolling(5).std().shift(-5) * 1e4

    def window(kind: str, side: str, minutes: int) -> pd.Series:
        return df[f"cal_{kind}_{side}_{minutes}"] > 0

    # ---------------------------------------------------------------- regime
    print("1. REZIM: volatilitas 5 bar setelahnya (median bps)")
    any_post = np.zeros(len(df), dtype=bool)
    for kind in ("nfp", "cpi", "fomc"):
        for side, minutes in (("pre", 15), ("post", 15), ("post", 60)):
            mask = window(kind, side, minutes).to_numpy()
            any_post |= mask
            ev = fwd_vol[mask].dropna()
            ref = fwd_vol[~mask].dropna()
            if len(ev) < 5:
                continue
            ratio = ev.median() / ref.median()
            u = stats.mannwhitneyu(ev, ref, alternative="greater")
            print(f"   {kind:5s} {side}-{minutes:<3d} n={len(ev):4d} | vol {ev.median():6.2f} "
                  f"vs {ref.median():5.2f} = {ratio:.2f}x | p={u.pvalue:.4f}")

    # ------------------------------------------------------------ tradability
    print("\n2. TRADABILITY: expectancy per sisi (target TP 1.5xATR)")
    print(f"   {'rezim':24s} {'n':>6s} {'short bps':>10s} {'long bps':>9s} {'p(short)':>9s}")
    base = ~any_post
    rows = [("normal (tanpa event)", base), ("jendela event", any_post)]
    out = {}
    for name, mask in rows:
        s = df.loc[mask, "ret_short_bps"].dropna()
        l = df.loc[mask, "ret_long_bps"].dropna()
        p = stats.ttest_1samp(s, 0.0).pvalue if len(s) > 2 else float("nan")
        print(f"   {name:24s} {int(mask.sum()):6d} {s.mean():+10.2f} {l.mean():+9.2f} {p:9.3f}")
        out[name] = {"n": int(mask.sum()), "short_bps": float(s.mean()),
                     "long_bps": float(l.mean()), "p_short": float(p)}

    # ------------------------------------------------------- statistical power
    print("\n3. POWER: berapa data yang dibutuhkan agar efek ini bisa dipercaya?")
    for kind in ("nfp", "cpi", "fomc"):
        for side, minutes in (("post", 15), ("post", 60)):
            mask = window(kind, side, minutes).to_numpy()
            s = df.loc[mask, "ret_short_bps"].dropna()
            if len(s) < 5 or s.mean() == 0:
                continue
            needed = int((s.std() * 3 / abs(s.mean())) ** 2)
            ratio = needed / max(len(s), 1)
            print(f"   {kind:5s} {side}-{minutes:<3d}: n={len(s):4d} mean {s.mean():+6.2f} bps "
                  f"-> butuh n≈{needed} ({ratio:.0f}x lebih banyak)")
            if ratio > 1 and span_days > 0:
                print(f"        ~{span_days * ratio:.0f} hari data pada timeframe ini "
                      f"(tersedia {span_days:.0f})")

    # --------------------------------------------------------- avoid-the-event
    print("\n4. FILTER: apakah menghindari event memperbaiki expectancy?")
    baseline = df["ret_short_bps"].mean()
    avoid_all = df.loc[~any_post, "ret_short_bps"].mean()
    print(f"   dasar (short selalu)         : {baseline:+.3f} bps")
    print(f"   hindari semua jendela event  : {avoid_all:+.3f} bps "
          f"({avoid_all - baseline:+.3f} vs dasar)")
    event_day = ((df["cal_nfp_is_day"] > 0) | (df["cal_cpi_is_day"] > 0)
                 | (df["cal_fomc_is_day"] > 0)).to_numpy()
    no_event_day = df.loc[~event_day, "ret_short_bps"].mean()
    print(f"   hindari seluruh hari event   : {no_event_day:+.3f} bps "
          f"({no_event_day - baseline:+.3f} vs dasar)")
    print("\n   Catatan: 'lebih baik' di sini tetap NEGATIF. Filter yang mengurangi")
    print("   kerugian bukan edge; ia hanya mengurangi frekuensi.")

    out_dir = Path("artifacts/calendar")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.symbol.replace('/', '')}_{args.bars}_h{args.horizon}"
    (out_dir / f"{tag}.json").write_text(json.dumps(
        {"symbol": args.symbol, "bars": args.bars, "span_days": span_days,
         "events_in_range": {k: cal.count(k, bars["time"].iloc[0], bars["time"].iloc[-1])
                             for k in cal.kinds()},
         "regimes": out}, indent=2))
    print(f"\nwritten -> {out_dir / (tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

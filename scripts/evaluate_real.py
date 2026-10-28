#!/usr/bin/env python3
"""Evaluate a trained model on REAL cached bars, with cost sensitivity.

Why this script exists: a single backtest number is a marketing figure. A desk
asks "how much of the result survives costs?" -- so we run the same signal under
three cost assumptions:

    demo       : PrimeXBT demo FX (taker 0%, spread ~0.2 bps, measured)
    realistic  : live-like FX (spread 0.9 bps, 0.01%/side)
    punitive   : spread 2 bps, 0.05%/side (crypto-grade costs)

A signal that only works in the first column does not exist.

    python scripts/evaluate_real.py --arch tcn --symbol EUR/USD --bars 1M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pxscalper.backtest.engine import BacktestConfig, Backtester  # noqa: E402
from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.features.labels import LabelConfig, volatility_scaled_labels  # noqa: E402
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer  # noqa: E402
from pxscalper.inference.predictor import load_predictor  # noqa: E402
from pxscalper.utils import RepoPaths, setup_logging  # noqa: E402

COST_SCENARIOS = {
    "demo (taker 0%, spread 0.2bps)": dict(taker_fee_pct=0.0, spread_bps_fallback=0.2),
    "realistic (spread 0.9bps, 0.01%/side)": dict(taker_fee_pct=0.01, spread_bps_fallback=0.9),
    "punitive (spread 2.0bps, 0.05%/side)": dict(taker_fee_pct=0.05, spread_bps_fallback=2.0),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="tcn")
    ap.add_argument("--symbol", default="EUR/USD")
    ap.add_argument("--bars", default="1M")
    ap.add_argument("--version", default="latest")
    ap.add_argument("--out", default="artifacts/eval")
    ap.add_argument("--horizon", type=int, default=5, help="must match training horizon")
    ap.add_argument("--n-bars", type=int, default=0,
                    help="use only the last N cached bars (0 = all). Windows are "
                         "38x64 float32 each, so 200k bars needs ~1.9 GB of RAM")
    ap.add_argument("--oos-frac", type=float, default=0.25,
                    help="evaluate only the last N%% of windows (held-out). "
                         "0 = all windows (in-sample, optimistic -- not a result)")
    ap.add_argument("--live-spread-bps", type=float, default=0.2,
                    help="spread you actually measured on the venue for this symbol")
    ap.add_argument("--sl-atr", type=float, default=1.5)
    ap.add_argument("--tp-atr", type=float, default=1.5)
    ap.add_argument("--max-holding", type=int, default=24)
    args = ap.parse_args()

    setup_logging()
    paths = RepoPaths().ensure()
    bars = DataPipeline(cache_dir=paths.data).load(args.symbol, args.bars)
    if bars is None:
        print(f"no cached data for {args.symbol}; run `pxs data` first")
        return 1

    if args.n_bars and len(bars) > args.n_bars:
        bars = bars.tail(args.n_bars).reset_index(drop=True)
    engineer = FeatureEngineer(FeatureConfig())
    features = engineer.transform_normalized(bars)
    labels = volatility_scaled_labels(bars, LabelConfig(horizon=args.horizon))
    common = features.index.intersection(labels.index)
    windows, end_idx = engineer.make_windows(features.loc[common], 64)
    frame = bars.loc[common].reset_index(drop=True)
    atr = labels.loc[common, "atr"].to_numpy()
    n = min(len(windows), len(end_idx))
    windows, end_idx = windows[:n], end_idx[:n]

    predictor = load_predictor(
        paths.model_dir(args.version), arch=args.arch,
        confidence_threshold=0.0, edge_threshold=0.0,
    )

    # held-out slice only: training used the first 75%, evaluating the whole
    # sample reports the model's memory, not its skill.
    oos_start = int(len(windows) * (1.0 - args.oos_frac)) if args.oos_frac > 0 else 0
    live = args.live_spread_bps
    scenarios = {
        f"measured venue (spread {live:.2f}bps, taker 0%)": dict(
            taker_fee_pct=0.0, spread_bps_fallback=live),
        f"venue + 0.01%/side (spread {live:.2f}bps)": dict(
            taker_fee_pct=0.01, spread_bps_fallback=live),
        f"2x cost (spread {2 * live:.2f}bps, 0.01%/side)": dict(
            taker_fee_pct=0.01, spread_bps_fallback=2 * live),
    }
    if not args.oos_frac:
        print("WARNING: --oos-frac 0 evaluates in-sample; the numbers are optimistic",
              file=sys.stderr)

    def signal(row: int, window: np.ndarray):
        if row < oos_start:
            return 0, 0.0, 0.0
        pred = predictor.predict(window)
        return pred.direction, pred.confidence, pred.edge

    results = {}
    for name, costs in scenarios.items():
        config = BacktestConfig(
            initial_equity=10_000.0,
            min_confidence=0.42,
            min_edge=0.03,
            sl_atr=args.sl_atr,
            tp_atr=args.tp_atr,
            max_holding_bars=args.max_holding,
            **costs,
        )
        result = Backtester(config).run(frame, windows, end_idx, atr, signal)
        results[name] = result
        m = result.metrics
        print(
            f"{name:42s} trades {m.get('n_trades', 0):5.0f} | "
            f"return {m.get('total_return_pct', 0):+7.2f}% | "
            f"sharpe {m.get('sharpe', 0):6.2f} | "
            f"win {m.get('win_rate_pct', 0):5.1f}% | "
            f"PF {m.get('profit_factor', 0):4.2f} | "
            f"DD {m.get('max_drawdown_pct', 0):5.2f}%"
        )

    # ---- signal quality, independent of trading rules ----------------------
    y_true, y_pred = [], []
    for row in range(oos_start, len(windows)):
        pred = predictor.predict(windows[row])
        y_true.append(labels.loc[common, "y_dir"].to_numpy()[end_idx[row]])
        y_pred.append(pred.direction)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    nonflat = y_true != 0
    hit = (y_pred == y_true)[nonflat].mean() if nonflat.any() else float("nan")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "symbol": args.symbol,
        "arch": args.arch,
        "bars": int(len(bars)),
        "windows": int(len(windows)),
        "period": {
            "from_ms": int(bars["time"].iloc[0]),
            "to_ms": int(bars["time"].iloc[-1]),
        },
        "evaluation": {
            "oos_frac": args.oos_frac,
            "oos_windows": int(len(windows) - oos_start),
            "horizon": args.horizon,
            "live_spread_bps": live,
        },
        "signal_quality": {
            "accuracy_all": float((y_pred == y_true).mean()),
            "accuracy_nonflat": float(hit),
            "share_flat_predictions": float((y_pred == 0).mean()),
            "share_flat_labels": float((y_true == 0).mean()),
        },
        "cost_scenarios": {name: r.metrics for name, r in results.items()},
    }
    (out_dir / f"{args.symbol.replace('/', '')}_{args.arch}_eval.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("\nsignal quality:", json.dumps(summary["signal_quality"], indent=2))
    for name, result in results.items():
        safe = name.split(" ")[0]
        result.to_frame().to_csv(out_dir / f"{safe}_trades.csv", index=False)
    print(f"written -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

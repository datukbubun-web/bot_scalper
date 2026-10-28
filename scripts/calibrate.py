#!/usr/bin/env python3
"""Fit temperature scaling on a held-out slice and write ``calibration.json``.

Why bother: a deep classifier trained on financial data is systematically
over-confident. Because the bot sizes positions and gates trades on
``confidence`` and ``edge``, miscalibration directly translates into over-sized
trades. Temperature scaling is one scalar, order-preserving, and fitted on data
the model never saw.

    python scripts/calibrate.py --arch tcn --symbol EUR/USD
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.exchange.rest import PrimeXBTClient  # noqa: E402
from pxscalper.features.labels import LabelConfig, volatility_scaled_labels  # noqa: E402
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer  # noqa: E402
from pxscalper.inference.export import load_model_from_checkpoint  # noqa: E402
from pxscalper.utils import RepoPaths, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="tcn")
    ap.add_argument("--symbol", default="EUR/USD")
    ap.add_argument("--bars", default="1M")
    ap.add_argument("--n-bars", type=int, default=60_000)
    ap.add_argument("--tail-frac", type=float, default=0.15, help="unseen tail used for fitting")
    ap.add_argument("--version", default="latest")
    args = ap.parse_args()

    setup_logging()
    paths = RepoPaths().ensure()

    client = PrimeXBTClient()
    pipeline = DataPipeline(cache_dir=paths.data, client=client)
    bars = pipeline.load(args.symbol, args.bars)
    if bars is None or len(bars) < args.n_bars // 2:
        bars = pipeline.download(args.symbol, args.bars, args.n_bars)

    engineer = FeatureEngineer(FeatureConfig())
    features = engineer.transform_normalized(bars)
    labels = volatility_scaled_labels(bars, LabelConfig())
    common = features.index.intersection(labels.index)
    model, mcfg = load_model_from_checkpoint(paths.model_dir(args.version) / f"{args.arch}.pt", args.arch)
    windows, window_times = engineer.make_windows(features.loc[common], mcfg.lookback)
    pos = {int(t): i for i, t in enumerate(common)}
    rows = np.array([pos[int(t)] for t in window_times if int(t) in pos], dtype=int)
    n = min(len(windows), len(rows))
    windows, rows = windows[:n], rows[:n]
    y = labels["y_dir"].to_numpy()[rows] + 1  # 0/1/2

    # fit only on the unseen tail
    cut = int(n * (1 - args.tail_frac))
    xb = torch.from_numpy(windows[cut:])
    model.eval()
    with torch.no_grad():
        logits = model(xb).numpy()
    yb = y[cut:]
    mask = (yb >= 0) & (yb <= 2)
    logits, yb = logits[mask], yb[mask]

    def nll(temperature: float) -> float:
        z = logits / max(temperature, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(len(yb)), yb].mean())

    before = nll(1.0)
    result = minimize_scalar(nll, bounds=(0.2, 10.0), method="bounded")
    after = float(result.fun)
    payload = {
        "temperature": float(result.x),
        "nll_before": before,
        "nll_after": after,
        "n_samples": int(len(yb)),
        "arch": args.arch,
        "symbol": args.symbol,
    }
    out = paths.model_dir(args.version) / "calibration.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"temperature {result.x:.3f} | NLL {before:.4f} -> {after:.4f} | written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

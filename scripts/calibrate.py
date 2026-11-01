#!/usr/bin/env python3
"""Fit temperature scaling on a held-out slice and write ``calibration.json``.

Why bother: a deep classifier trained on financial data is systematically
over-confident. Because the bot sizes positions and gates trades on
``confidence`` and ``edge``, miscalibration directly translates into over-sized
trades. Temperature scaling is one scalar, order-preserving, and fitted on data
the model never saw.

    python scripts/calibrate.py --arch tcn --symbol XAU/USD --bars 5M
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pxscalper.data.pipeline import DataPipeline
from pxscalper.exchange.rest import PrimeXBTClient
from pxscalper.features.labels import LabelConfig, volatility_scaled_labels
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer
from pxscalper.inference.export import load_model_from_checkpoint
from pxscalper.inference.predictor import Calibration
from pxscalper.ops.backtest_runner import load_context_frames, load_htf_frames
from pxscalper.utils import RepoPaths, setup_logging


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="tcn")
    ap.add_argument("--symbol", default="XAU/USD")
    ap.add_argument("--bars", default="5M")
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--n-bars", type=int, default=60_000)
    ap.add_argument("--val-start", type=float, default=0.60, help="validation slice start fraction")
    ap.add_argument("--val-end", type=float, default=0.80, help="validation slice end fraction")
    ap.add_argument("--version", default="latest")
    ap.add_argument("--model-dir", default=None)
    args = ap.parse_args()

    setup_logging()
    paths = RepoPaths().ensure()
    model_dir = Path(args.model_dir) if args.model_dir else paths.model_dir(args.version)

    pipeline = DataPipeline(cache_dir=paths.data)
    bars = pipeline.load(args.symbol, args.bars)
    if bars is None or len(bars) < 500:
        client = PrimeXBTClient()
        pipeline = DataPipeline(cache_dir=paths.data, client=client)
        bars = pipeline.download(args.symbol, args.bars, args.n_bars)

    feat_json = model_dir / "features.json"
    engineer = FeatureEngineer(FeatureConfig.load(feat_json) if feat_json.exists() else FeatureConfig())
    context_map = load_context_frames(pipeline, engineer, bar_size=args.bars)
    htf_bars_map, htf_ctx_map = load_htf_frames(
        pipeline, engineer, symbol=args.symbol, ltf_bar_size=args.bars
    )
    span_mins: list[int] = []
    if context_map:
        span_mins.extend(int(cf["time"].min()) for cf in context_map.values() if not cf.empty)
    if htf_ctx_map:
        for cmap in htf_ctx_map.values():
            span_mins.extend(int(cf["time"].min()) for cf in cmap.values() if not cf.empty)
    if span_mins:
        bars = bars[bars["time"] >= max(span_mins)].reset_index(drop=True)

    features = engineer.transform_normalized(
        bars, context=context_map, htf_bars=htf_bars_map, htf_context=htf_ctx_map
    )
    labels = volatility_scaled_labels(bars, LabelConfig(horizon=args.horizon))
    common = features.index.intersection(labels.index)
    model, mcfg = load_model_from_checkpoint(model_dir / f"{args.arch}.pt", args.arch)
    windows, end_idx = engineer.make_windows(features.loc[common], mcfg.lookback)
    n = min(len(windows), len(end_idx))
    windows, end_idx = windows[:n], end_idx[:n]
    y_all = labels.loc[common, "y_dir"].to_numpy()[end_idx] + 1  # 0/1/2

    # fit strictly on the validation slice [val_start, val_end)
    lo = int(n * args.val_start)
    hi = int(n * args.val_end)
    xb = torch.from_numpy(windows[lo:hi])
    model.eval()
    with torch.no_grad():
        logits = model(xb).numpy()
    yb = y_all[lo:hi]
    mask = (yb >= 0) & (yb <= 2)
    if getattr(mcfg, "antisymmetric_head", False):
        # Antisymmetric gated models output exact [0, 0, 0] logits on quiet/gated bars;
        # fit temperature strictly on active bars and cap T <= 1.05 so selective
        # dislocation confidence is not crushed below threshold.
        active = np.abs(logits).max(axis=1) > 1e-4
        if int((mask & active).sum()) >= 10:
            mask = mask & active
        bounds = (0.5, 1.05)
    else:
        bounds = (0.2, 10.0)
    logits, yb = logits[mask], yb[mask]

    def nll(temperature: float) -> float:
        z = logits / max(temperature, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        log_probs = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-log_probs[np.arange(len(yb)), yb].mean())

    before = nll(1.0)
    result = minimize_scalar(nll, bounds=bounds, method="bounded")
    after = float(result.fun)
    out = model_dir / "calibration.json"
    Calibration(
        temperature=float(result.x),
        symbol=args.symbol,
        arch=args.arch,
    ).save(
        out,
        nll_before=before,
        nll_after=after,
        n_samples=int(len(yb)),
        slice=[args.val_start, args.val_end],
    )
    print(f"temperature {result.x:.3f} | NLL {before:.4f} -> {after:.4f} | written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

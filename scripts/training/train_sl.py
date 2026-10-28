#!/usr/bin/env python3
"""Train a supervised sequence model (TCN / LSTM / GRU / Transformer).

Example
-------
    python scripts/training/train_sl.py --arch tcn --symbol EUR/USD --bars 1M \
        --n-bars 60000 --epochs 40 --export

The script is walk-forward aware: validation is the last ``val_frac`` of the
sample with an embargo gap, and the shipped checkpoint is the EMA of the weights
at the best validation loss.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.exchange.rest import PrimeXBTClient  # noqa: E402
from pxscalper.features.labels import LabelConfig, volatility_scaled_labels  # noqa: E402
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer  # noqa: E402
from pxscalper.models.sequence import ModelConfig, build_sequence_model  # noqa: E402
from pxscalper.models.trainer import TrainConfig, Trainer, WindowDataset, class_weights  # noqa: E402
from pxscalper.utils import RepoPaths, load_dotenv, seed_everything, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="tcn", choices=["tcn", "lstm", "gru", "transformer"])
    ap.add_argument("--symbol", default="EUR/USD")
    ap.add_argument("--bars", default="1M", help="bar size: 1M 5M 15M 1H ...")
    ap.add_argument("--n-bars", type=int, default=60_000)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--embargo", type=int, default=25)
    ap.add_argument("--synthetic", action="store_true", help="train on synthetic data (smoke)")
    ap.add_argument("--export", action="store_true", help="export TorchScript + ONNX after training")
    ap.add_argument("--version", default="latest", help="artifact folder name")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    load_dotenv(".env")
    setup_logging()
    seed_everything(args.seed)
    paths = RepoPaths().ensure()

    # ------------------------------------------------------------------ data
    if args.synthetic:
        bars = DataPipeline.synthetic(symbol=args.symbol, n_bars=args.n_bars)
    else:
        client = PrimeXBTClient()  # bars are unsigned
        pipeline = DataPipeline(cache_dir=paths.data, client=client)
        cached = pipeline.load(args.symbol, args.bars)
        bars = cached if cached is not None and len(cached) >= args.n_bars else pipeline.download(
            args.symbol, args.bars, args.n_bars
        )
    print(f"{args.symbol} {args.bars}: {len(bars)} bars "
          f"({bars['time'].iloc[0]} -> {bars['time'].iloc[-1]})")

    # -------------------------------------------------------------- features
    fcfg = FeatureConfig()
    engineer = FeatureEngineer(fcfg)
    features = engineer.transform_normalized(bars)
    labels = volatility_scaled_labels(bars, LabelConfig(horizon=args.horizon))
    common = features.index.intersection(labels.index)
    windows, window_times = engineer.make_windows(features.loc[common], args.lookback)
    pos = {int(t): i for i, t in enumerate(common)}
    rows = np.array([pos[int(t)] for t in window_times if int(t) in pos], dtype=int)
    n = min(len(windows), len(rows))
    windows, rows = windows[:n], rows[:n]
    y = labels["y_dir"].to_numpy()[rows]
    y_ret = labels["y_ret"].to_numpy()[rows]
    print(f"windows: {windows.shape} | class balance: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ----------------------------------------------------------------- model
    mcfg = ModelConfig(
        n_features=engineer.n_features, lookback=args.lookback, n_outputs=3,
        channels=(64, 64, 64, 64), hidden_size=64, d_model=64, n_heads=4, n_layers=2,
    )
    model = build_sequence_model(args.arch, mcfg)

    split = int(len(windows) * (1 - args.val_frac))
    train_end = max(split - args.embargo, 1)
    train_ds = WindowDataset(windows[:train_end], y[:train_end])
    val_ds = WindowDataset(windows[split:], y[split:])
    weights = class_weights(y[:train_end])

    trainer = Trainer(
        model,
        TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, class_balance=True),
        weights,
    )
    history = trainer.fit(train_ds, val_ds, y_ret=y_ret[split:])

    # ------------------------------------------------------------- artifacts
    model_dir = paths.model_dir(args.version)
    ckpt = model_dir / f"{args.arch}.pt"
    meta = {
        "arch": args.arch,
        "symbol": args.symbol,
        "bar_size": args.bars,
        "lookback": args.lookback,
        "horizon": args.horizon,
        "feature_version": engineer.version,
        "features": list(engineer.feature_names),
        "history": history.to_dict(),
        "windows": int(len(windows)),
        "class_balance": {str(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
        "trained_at_ms": int(__import__("time").time() * 1000),
    }
    trainer.save(ckpt, meta)
    fcfg.save(model_dir / "features.json")
    (model_dir / "training_report.json").write_text(json.dumps(meta, indent=2))
    print(f"saved {ckpt}")
    print(f"best epoch {history.best_epoch} | val loss {history.best_val_loss:.4f} | "
          f"acc(non-flat) {history.val_acc_nonflat[history.best_epoch]:.3f} | "
          f"dir. sharpe {history.val_sharpe[history.best_epoch]:.2f}")

    if args.export:
        from pxscalper.inference.export import export_all

        result = export_all(ckpt, args.arch, model_dir, iterations=2000)
        print(json.dumps(result.latency_us, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

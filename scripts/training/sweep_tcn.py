"""Capacity/regularisation sweep for the TCN, with a linear ceiling to beat.

The TCN overfits within one epoch on FX/gold bars and ends up worse than a
logistic regression on the same features. That is a capacity/regularisation
problem, not a data problem, so sweep the knobs that matter and keep the linear
model in the table as the bar to clear.

    python scripts/training/sweep_tcn.py --symbol XAU/USD --bars 1M --n-bars 16000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.features.labels import LabelConfig, volatility_scaled_labels  # noqa: E402
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer  # noqa: E402
from pxscalper.models.sequence import ModelConfig, build_sequence_model  # noqa: E402
from pxscalper.models.trainer import TrainConfig, Trainer, WindowDataset, class_weights  # noqa: E402
from pxscalper.utils import RepoPaths, seed_everything, setup_logging  # noqa: E402


def held_out_metrics(model, x_va, y_va, r_va) -> dict:
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(x_va), 512):
            logits.append(model(torch.from_numpy(x_va[i : i + 512])).numpy())
    logits = np.concatenate(logits)
    prob = torch.softmax(torch.from_numpy(logits), -1).numpy()
    pred = logits.argmax(1) - 1
    nf = y_va != 0
    edge = prob[:, 2] - prob[:, 0]
    return {
        "log_loss": float(-np.log(np.clip(prob[np.arange(len(y_va)), y_va + 1], 1e-9, 1)).mean()),
        "acc_nonflat": float((pred[nf] == y_va[nf]).mean()) if nf.any() else float("nan"),
        "ic": float(np.corrcoef(edge, r_va)[0, 1]) if edge.std() > 0 else 0.0,
        "pnl_bps": float((pred * r_va).mean() * 1e4),
        "p_max": float(prob.max(1).mean()),
    }


def logistic_baseline(x_tr, y_tr, x_va, y_va, r_va) -> dict:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=3000, C=0.1).fit(x_tr[:, :, -1], y_tr)
    prob = clf.predict_proba(x_va[:, :, -1])
    pred = clf.predict(x_va[:, :, -1])
    nf = y_va != 0
    edge = prob[:, 2] - prob[:, 0]
    return {
        "log_loss": float(-np.log(np.clip(prob[np.arange(len(y_va)), y_va + 1], 1e-9, 1)).mean()),
        "acc_nonflat": float((pred[nf] == y_va[nf]).mean()) if nf.any() else float("nan"),
        "ic": float(np.corrcoef(edge, r_va)[0, 1]) if edge.std() > 0 else 0.0,
        "pnl_bps": float((pred * r_va).mean() * 1e4),
        "p_max": float(prob.max(1).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="XAU/USD")
    ap.add_argument("--bars", default="1M")
    ap.add_argument("--n-bars", type=int, default=16000)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=5)
    args = ap.parse_args()

    setup_logging("WARNING")
    paths = RepoPaths().ensure()
    bars = DataPipeline(cache_dir=paths.data).load(args.symbol, args.bars).tail(args.n_bars)
    bars = bars.reset_index(drop=True)
    engineer = FeatureEngineer(FeatureConfig())
    features = engineer.transform_normalized(bars)
    labels = volatility_scaled_labels(bars, LabelConfig(horizon=args.horizon))
    common = features.index.intersection(labels.index)
    windows, times = engineer.make_windows(features.loc[common], args.lookback)
    pos = {int(t): i for i, t in enumerate(common)}
    rows = np.array([pos[int(t)] for t in times], dtype=int)
    y = labels["y_dir"].to_numpy()[rows]
    y_ret = labels["y_ret"].to_numpy()[rows]

    purge = args.lookback + args.horizon
    split = int(len(windows) * 0.8)
    train_end = split - purge
    x_tr, y_tr = windows[:train_end], y[:train_end]
    x_va, y_va, r_va = windows[split:], y[split:], y_ret[split:]
    cw = class_weights(y_tr)
    print(f"{args.symbol} {args.bars}: {len(bars)} bars -> train {len(x_tr)} / val {len(x_va)}")
    print(f"acak ln(3) = {np.log(3):.4f}\n")

    base = logistic_baseline(x_tr, y_tr, x_va, y_va, r_va)
    print(f"{'logistic (batas bawah yg harus dikalahkan)':44s} "
          f"log-loss {base['log_loss']:.4f} | acc(nf) {base['acc_nonflat']:.3f} | "
          f"IC {base['ic']:+.4f} | PnL {base['pnl_bps']:+7.2f}bps")

    configs = {
        "kecil 16ch, do .3, wd 1e-2": dict(channels=(16, 16), dropout=0.3, wd=1e-2, lr=3e-4, ep=6),
        "sedang 32ch, do .3, wd 1e-2": dict(channels=(32, 32), dropout=0.3, wd=1e-2, lr=3e-4, ep=6),
        "besar 64ch x4, do .3, wd 1e-2": dict(channels=(64, 64, 64, 64), dropout=0.3, wd=1e-2, lr=1e-4, ep=6),
        "kecil 16ch, do .5, wd 5e-2": dict(channels=(16, 16), dropout=0.5, wd=5e-2, lr=3e-4, ep=8),
    }
    out = {"logistic": base}
    best = None
    for name, spec in configs.items():
        seed_everything(7)
        mcfg = ModelConfig(
            n_features=engineer.n_features, lookback=args.lookback, n_outputs=3,
            channels=spec["channels"], dropout=spec["dropout"],
            hidden_size=32, d_model=32, head_hidden=32, n_heads=4, n_layers=2,
        )
        model = build_sequence_model("tcn", mcfg)
        trainer = Trainer(
            model,
            TrainConfig(epochs=spec["ep"], batch_size=256, lr=spec["lr"],
                        weight_decay=spec["wd"], patience=2, ema_warmup=True),
            cw, window_overlap=args.lookback,
        )
        hist = trainer.fit(WindowDataset(x_tr, y_tr), WindowDataset(x_va, y_va),
                           y_ret=r_va)
        metrics = held_out_metrics(trainer.ema.shadow, x_va, y_va, r_va)
        metrics["params"] = int(sum(p.numel() for p in model.parameters()))
        metrics["beat_baseline"] = bool(hist.beat_baseline)
        out[name] = metrics
        print(f"{name:44s} log-loss {metrics['log_loss']:.4f} | "
              f"acc(nf) {metrics['acc_nonflat']:.3f} | IC {metrics['ic']:+.4f} | "
              f"PnL {metrics['pnl_bps']:+7.2f}bps | {metrics['params'] / 1000:.0f}k param "
              f"| {'LULUS' if hist.beat_baseline else 'gagal'}")
        if best is None or metrics["log_loss"] < best[1]["log_loss"]:
            best = (name, metrics)

    Path("artifacts/sweep_tcn.json").write_text(json.dumps(out, indent=2))
    print(f"\nterbaik: {best[0]} (log-loss {best[1]['log_loss']:.4f} vs "
          f"logistic {base['log_loss']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

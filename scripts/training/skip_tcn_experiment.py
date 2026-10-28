"""Can the TCN be made to at least match the linear baseline?

Two measured facts drive this experiment:

1. The stock TCN needs far more optimizer steps than a 12-epoch run provides
   (in the overfit probe it reached 0.867 train accuracy on 1200 samples with
   120 full-batch steps -- so the architecture and gradients are fine).
2. A logistic regression on the same 38 features beats every TCN variant tried.
   The features are standardised and the relation is largely linear, so the
   sequence stack is being asked to rediscover linear structure through
   convolutions -- an inefficient path.

So: give the head a direct linear skip (pooled conv output concatenated with the
raw last-step features) and train with more optimizer steps (smaller batch).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.features.labels import LabelConfig, volatility_scaled_labels  # noqa: E402
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer  # noqa: E402
from pxscalper.models.sequence import ModelConfig, TCN, build_sequence_model  # noqa: E402
from pxscalper.models.trainer import TrainConfig, Trainer, WindowDataset, class_weights  # noqa: E402
from pxscalper.utils import RepoPaths, seed_everything, setup_logging  # noqa: E402


class SkipTCN(nn.Module):
    """TCN head fed with the pooled conv summary *and* the raw last-step features.

    The skip is the sequence-model equivalent of the linear term in a GLM: it lets
    the network read off any linear relation directly instead of approximating it
    with dilated convolutions, while the conv stack still models interactions.
    """

    def __init__(self, config: ModelConfig, pool: int = 8) -> None:
        super().__init__()
        self.config = config
        self.tcn = TCN(config)
        self.pool = pool
        self.head = nn.Sequential(
            nn.Linear(config.channels[-1] + config.n_features, config.head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.head_hidden, config.n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn.blocks(x)                       # (B, C, L)
        pooled = h[:, :, -self.pool :].mean(dim=-1)
        raw = x[:, :, -1]                            # (B, n_features)
        return self.head(torch.cat([self.tcn.norm(pooled), raw], dim=-1))


def metrics(model, x_va, y_va, r_va) -> dict:
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
        "p_max": float(prob.max(1).mean()),
        "n_down": int((pred == -1).sum()), "n_flat": int((pred == 0).sum()),
        "n_up": int((pred == 1).sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="XAU/USD")
    ap.add_argument("--bars", default="1M")
    ap.add_argument("--n-bars", type=int, default=16000)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--only", default="", help="substring filter on the run names")
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
    x_tr, y_tr = windows[: split - purge], y[: split - purge]
    x_va, y_va, r_va = windows[split:], y[split:], y_ret[split:]
    cw = class_weights(y_tr)
    print(f"{args.symbol} {args.bars}: train {len(x_tr)} / val {len(x_va)}")

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=3000, C=0.1).fit(x_tr[:, :, -1], y_tr)
    prob = clf.predict_proba(x_va[:, :, -1])
    pred = clf.predict(x_va[:, :, -1])
    nf = y_va != 0
    maj = -1 if np.mean(y_tr == -1) > np.mean(y_tr == 1) else 1
    print(f"\n{'drift floor (tebak arah mayoritas train)':46s} acc(nf) {np.mean(y_va[nf] == maj):.3f}")
    print(f"{'logistic 38 fitur':46s} log-loss "
          f"{-np.log(np.clip(prob[np.arange(len(y_va)), y_va + 1], 1e-9, 1)).mean():.4f} | "
          f"acc(nf) {np.mean(pred[nf] == y_va[nf]):.3f}")

    runs = {
        "stock 32ch, batch 256, 12 ep": (
            lambda: build_sequence_model("tcn", ModelConfig(
                n_features=engineer.n_features, lookback=args.lookback, n_outputs=3,
                channels=(32, 32), hidden_size=32, d_model=32, head_hidden=32, dropout=0.1)),
            dict(batch_size=256, epochs=12, lr=3e-4)),
        "skip 32ch, batch 128, 30 ep": (
            lambda: SkipTCN(ModelConfig(
                n_features=engineer.n_features, lookback=args.lookback, n_outputs=3,
                channels=(32, 32), hidden_size=32, d_model=32, head_hidden=32, dropout=0.1)),
            dict(batch_size=128, epochs=30, lr=3e-4)),
        "skip 64ch, batch 128, 30 ep": (
            lambda: SkipTCN(ModelConfig(
                n_features=engineer.n_features, lookback=args.lookback, n_outputs=3,
                channels=(64, 64), hidden_size=64, d_model=64, head_hidden=64, dropout=0.1)),
            dict(batch_size=128, epochs=30, lr=2e-4)),
    }

    out = {}
    for name, (build, spec) in runs.items():
        if args.only and args.only.lower() not in name.lower():
            continue
        seed_everything(7)
        model = build()
        trainer = Trainer(model, TrainConfig(epochs=spec["epochs"], batch_size=spec["batch_size"],
                                            lr=spec["lr"], patience=5, ema_warmup=True),
                          cw, window_overlap=args.lookback)
        hist = trainer.fit(WindowDataset(x_tr, y_tr), WindowDataset(x_va, y_va), y_ret=r_va)
        m = metrics(trainer.ema.shadow, x_va, y_va, r_va)
        m["beat_baseline"] = bool(hist.beat_baseline)
        m["steps_per_epoch"] = int(np.ceil(len(x_tr) / spec["batch_size"]))
        out[name] = m
        print(f"{name:46s} log-loss {m['log_loss']:.4f} | acc(nf) {m['acc_nonflat']:.3f} | "
              f"IC {m['ic']:+.4f} | pred(-1/0/+1) {m['n_down']}/{m['n_flat']}/{m['n_up']} | "
              f"{'LULUS' if hist.beat_baseline else 'gagal'}")

    Path("artifacts/skip_tcn.json").write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

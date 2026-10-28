"""Ablation: which bug actually breaks the TCN?

Trains the same architecture on the same data under several fixes, one at a
time, and reports held-out metrics. Run it when the model looks "worse than
random" -- architecture arguments are guesses, this is evidence.

    python scripts/training/ablate_tcn.py --symbol XAU/USD --bars 1M --n-bars 24000
"""

from __future__ import annotations

import argparse
import copy
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
from pxscalper.models.sequence import ModelConfig, build_sequence_model  # noqa: E402
from pxscalper.utils import RepoPaths, seed_everything, setup_logging  # noqa: E402


class MeanPoolTCN(nn.Module):
    """TCN whose head sees the mean over the last k positions, not just the last.

    The stock model does ``h[:, :, -1]`` and so exposes the head to a single
    noisy bar, even though the 4-block/dilation-8 stack has a 61-bar receptive
    field to draw on. Pooling over the same span keeps the head's input variance low.
    """

    def __init__(self, config: ModelConfig, pool: int = 8) -> None:
        super().__init__()
        from pxscalper.models.sequence import TCN

        self.tcn = TCN(config)
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn.blocks(x)                     # (B, C, L)
        pooled = h[:, :, -self.pool :].mean(dim=-1)
        return self.tcn.head(self.tcn.norm(pooled))


def evaluate(model: nn.Module, windows: np.ndarray, y: np.ndarray, y_ret: np.ndarray,
             batch: int = 512) -> dict:
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(windows), batch):
            logits.append(model(torch.from_numpy(windows[i : i + batch])).numpy())
    logits = np.concatenate(logits)
    prob = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    pred = logits.argmax(1) - 1
    nonflat = y != 0
    edge = prob[:, 2] - prob[:, 0]
    pnl = pred * y_ret
    stats = {
        "log_loss": float(-np.log(np.clip(prob[np.arange(len(y)), y + 1], 1e-9, 1)).mean()),
        "acc": float((pred == y).mean()),
        "acc_nonflat": float((pred[nonflat] == y[nonflat]).mean()) if nonflat.any() else float("nan"),
        "ic": float(np.corrcoef(edge, y_ret)[0, 1]) if edge.std() > 0 else 0.0,
        "pnl_bps": float(pnl.mean() * 1e4),
        "max_prob": float(prob.max(1).mean()),
    }
    return stats


def train_one(model: nn.Module, x_tr, y_tr, x_va, y_va, *, epochs: int, lr: float,
              weight_decay: float, class_weight: torch.Tensor, dropout_ok: bool = True,
              batch: int = 256, use_ema: bool = True, label_smoothing: float = 0.0,
              cosine: bool = True) -> nn.Module:
    loss_fn = nn.CrossEntropyLoss(weight=class_weight, label_smoothing=label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = max((len(x_tr) + batch - 1) // batch, 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 0.5 * (1 + np.cos(np.pi * min(s / max(steps * epochs, 1), 1.0)))
        if cosine else 1.0
    )
    ema = copy.deepcopy(model).eval() if use_ema else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)
    yt = torch.from_numpy(y_tr + 1)
    xt = torch.from_numpy(x_tr)
    best = (float("inf"), None)
    n = len(xt)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            sel = perm[i : i + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[sel]), yt[sel])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for a, b in zip(ema.state_dict().values(), model.state_dict().values()):
                        if a.dtype.is_floating_point:
                            a.mul_(0.995).add_(b.detach(), alpha=0.005)
                        else:
                            a.copy_(b)
        candidate = ema if ema is not None else model
        stats = evaluate(candidate, x_va, y_va, np.zeros(len(y_va)))
        if stats["log_loss"] < best[0]:
            best = (stats["log_loss"], copy.deepcopy(candidate.state_dict()))
    model.load_state_dict(best[1])
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="XAU/USD")
    ap.add_argument("--bars", default="1M")
    ap.add_argument("--n-bars", type=int, default=24000)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()

    setup_logging("WARNING")
    seed_everything(7)
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
    print(f"{args.symbol} {args.bars}: {len(bars)} bars -> {windows.shape} windows")

    split = int(len(windows) * 0.8)
    x_tr, x_va = windows[:split], windows[split:]
    y_tr, y_va = y[:split], y[split:]
    r_va = y_ret[split:]
    counts = np.bincount(y_tr + 1, minlength=3)
    cw = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    print(f"train {len(x_tr)} / val {len(x_va)} | kelas train {counts.tolist()}")
    print(f"baseline acak: log-loss {np.log(3):.4f}, acc {1 / 3:.4f}\n")

    mcfg = ModelConfig(n_features=engineer.n_features, lookback=args.lookback, n_outputs=3,
                       channels=(64, 64, 64, 64), hidden_size=64, d_model=64, n_heads=4,
                       n_layers=2)

    runs = {
        "stock (EMA, wd=1e-4, ls=0.02)": dict(
            build=lambda: build_sequence_model("tcn", mcfg),
            train=dict(lr=1e-3, weight_decay=1e-4, use_ema=True, label_smoothing=0.02)),
        "tanpa EMA (bobot akhir)": dict(
            build=lambda: build_sequence_model("tcn", mcfg),
            train=dict(lr=1e-3, weight_decay=1e-4, use_ema=False, label_smoothing=0.02)),
        "lr 1e-3 -> 1e-4": dict(
            build=lambda: build_sequence_model("tcn", mcfg),
            train=dict(lr=1e-4, weight_decay=1e-4, use_ema=True, label_smoothing=0.02)),
        "tanpa weight decay (1e-4 -> 0)": dict(
            build=lambda: build_sequence_model("tcn", mcfg),
            train=dict(lr=1e-3, weight_decay=0.0, use_ema=True, label_smoothing=0.02)),
        "mean-pool 8 bar terakhir": dict(
            build=lambda: MeanPoolTCN(mcfg, pool=8),
            train=dict(lr=1e-4, weight_decay=0.0, use_ema=True, label_smoothing=0.0)),
        "gabungan (pool + lr 1e-4 + no wd + no ls)": dict(
            build=lambda: MeanPoolTCN(mcfg, pool=8),
            train=dict(lr=1e-4, weight_decay=0.0, use_ema=False, label_smoothing=0.0)),
    }

    results = {}
    for name, spec in runs.items():
        seed_everything(7)
        model = spec["build"]()
        model = train_one(model, x_tr, y_tr, x_va, y_va, epochs=args.epochs,
                          class_weight=cw, **spec["train"])
        results[name] = evaluate(model, x_va, y_va, r_va)
        r = results[name]
        print(f"{name:42s} log-loss {r['log_loss']:.4f} | acc {r['acc']:.3f} | "
              f"acc(non-flat) {r['acc_nonflat']:.3f} | IC {r['ic']:+.4f} | "
              f"PnL {r['pnl_bps']:+6.2f}bps | Pmax {r['max_prob']:.3f}")

    out = Path("artifacts/ablation_tcn.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

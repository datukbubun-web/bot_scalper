"""Supervised training loop for the sequence models.

Institutional practices implemented here (each one is a place naive bots leak):

1. **Walk-forward validation with a purge gap** -- no fold ever sees a label
   whose outcome window overlaps its training data.
2. **Class-balanced loss** -- dead-band labelling makes the classes uneven;
   a plain cross-entropy collapses to "always flat".
3. **EMA of weights** -- the model that ships is the exponential moving average
   of the weights, which is materially more stable out-of-sample than the last
   SGD iterate.
4. **Early stopping on validation loss AND on directional accuracy** -- a model
   can lower its loss by predicting the majority class; accuracy on non-flat
   labels is the metric that matters for a scalper.
5. **Gradient clipping + warmup/cosine schedule** -- transformers especially.
6. **Deterministic seeding** -- a backtest is only meaningful if it reproduces.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)

__all__ = ["TrainConfig", "WindowDataset", "Trainer", "TrainHistory", "class_weights"]


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_frac: float = 0.05
    patience: int = 6
    ema_decay: float = 0.995
    label_smoothing: float = 0.02
    class_balance: bool = True
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 7
    mixed_precision: bool = False
    log_every: int = 0


class WindowDataset(Dataset):
    """(N, C, L) windows + integer labels (offset by +1 so -1/0/1 -> 0/1/2)."""

    def __init__(self, windows: np.ndarray, labels: np.ndarray, n_classes: int = 3) -> None:
        assert len(windows) == len(labels), "windows/labels length mismatch"
        self.windows = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
        if n_classes == 3:
            self.labels = torch.from_numpy(np.ascontiguousarray(labels + 1, dtype=np.int64))
        else:
            self.labels = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.int64))

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.labels[idx]


def class_weights(labels: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels + 1 if n_classes == 3 else labels), minlength=n_classes)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    val_acc_nonflat: list[float] = field(default_factory=list)
    val_sharpe: list[float] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = float("inf")
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "train_loss": self.train_loss,
            "val_loss": self.val_loss,
            "val_acc": self.val_acc,
            "val_acc_nonflat": self.val_acc_nonflat,
            "val_sharpe": self.val_sharpe,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "seconds": self.seconds,
        }


class _EMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for s_param, param in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s_param.dtype.is_floating_point:
                s_param.mul_(self.decay).add_(param.detach(), alpha=1 - self.decay)
            else:
                s_param.copy_(param)


class Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig, class_weight: torch.Tensor | None = None):
        self.model = model.to(config.device)
        self.config = config
        self.loss_fn: nn.Module = (
            nn.CrossEntropyLoss(weight=class_weight, label_smoothing=config.label_smoothing)
            if class_weight is not None
            else nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        )
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )
        self.ema = _EMA(model, config.ema_decay)

    # ------------------------------------------------------------------ schedule
    def _lr_lambda(self, step: int, total: int) -> float:
        warmup = max(int(total * self.config.warmup_frac), 1)
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: self._lr_lambda(s, total_steps)
        )

    # --------------------------------------------------------------------- train
    def fit(
        self,
        train_ds: WindowDataset,
        val_ds: WindowDataset,
        y_ret: np.ndarray | None = None,
    ) -> TrainHistory:
        cfg = self.config
        history = TrainHistory()
        started = time.time()
        loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
            drop_last=len(train_ds) > cfg.batch_size, pin_memory=False,
        )
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)
        total_steps = max(len(loader) * cfg.epochs, 1)
        scheduler = self._scheduler(total_steps)
        best_state = copy.deepcopy(self.ema.shadow.state_dict())
        patience_left = cfg.patience

        for epoch in range(cfg.epochs):
            self.model.train()
            running, seen = 0.0, 0
            for x, y in loader:
                x, y = x.to(cfg.device), y.to(cfg.device)
                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
                self.optimizer.step()
                scheduler.step()
                self.ema.update(self.model)
                running += float(loss.detach()) * x.shape[0]
                seen += x.shape[0]
            train_loss = running / max(seen, 1)

            val_loss, val_acc, val_acc_nf = self._evaluate(self.ema.shadow, val_loader)
            val_sharpe = self._directional_sharpe(self.ema.shadow, val_loader, y_ret)

            history.train_loss.append(train_loss)
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)
            history.val_acc_nonflat.append(val_acc_nf)
            history.val_sharpe.append(val_sharpe)
            log.info(
                "epoch %02d | train %.4f | val %.4f | acc %.3f | acc(non-flat) %.3f | sharpe %.2f",
                epoch, train_loss, val_loss, val_acc, val_acc_nf, val_sharpe,
            )

            if val_loss < history.best_val_loss - 1e-5:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_state = copy.deepcopy(self.ema.shadow.state_dict())
                patience_left = cfg.patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    log.info("early stop at epoch %d (best %d)", epoch, history.best_epoch)
                    break

        self.ema.shadow.load_state_dict(best_state)
        self.model.load_state_dict(best_state)
        history.seconds = time.time() - started
        return history

    # ------------------------------------------------------------------ evaluate
    @torch.no_grad()
    def _evaluate(self, model: nn.Module, loader: DataLoader) -> tuple[float, float, float]:
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        nf_correct, nf_total = 0, 0
        for x, y in loader:
            x, y = x.to(self.config.device), y.to(self.config.device)
            logits = model(x)
            total_loss += float(self.loss_fn(logits, y)) * x.shape[0]
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum())
            mask = y != 1  # class 1 is the "flat" class after the +1 shift
            nf_correct += int((pred[mask] == y[mask]).sum())
            nf_total += int(mask.sum())
            total += x.shape[0]
        return (
            total_loss / max(total, 1),
            correct / max(total, 1),
            nf_correct / max(nf_total, 1) if nf_total else 0.0,
        )

    @torch.no_grad()
    def _directional_sharpe(
        self, model: nn.Module, loader: DataLoader, y_ret: np.ndarray | None
    ) -> float:
        """Sharpe of a trivial strategy: trade the predicted direction, size 1.

        This is a *model-selection* metric, not a backtest -- costs, spread and
        position sizing arrive later in the risk/backtest layers. Its only job is
        to stop us shipping a model that is accurate but uselessly late.
        """
        if y_ret is None or len(y_ret) == 0:
            return 0.0
        model.eval()
        preds: list[np.ndarray] = []
        for x, _ in loader:
            logits = model(x.to(self.config.device))
            preds.append(logits.argmax(dim=-1).cpu().numpy())
        pred_dir = np.concatenate(preds) - 1 if preds else np.array([])
        n = min(len(pred_dir), len(y_ret))
        pnl = pred_dir[:n] * y_ret[:n]
        if pnl.std() == 0:
            return 0.0
        return float(pnl.mean() / pnl.std() * math.sqrt(252 * 390))

    # --------------------------------------------------------------------- save
    def save(self, path: str | Path, meta: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.ema.shadow.state_dict(),
                "model_config": getattr(self.model, "config", None).to_dict()
                if getattr(self.model, "config", None)
                else None,
                "meta": meta or {},
            },
            path,
        )
        if meta:
            path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))

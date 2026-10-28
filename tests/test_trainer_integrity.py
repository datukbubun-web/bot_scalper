"""Tests for the training-loop bugs found while chasing "worse than random".

Each test pins a specific defect that made a shipped checkpoint score no better
than a coin flip, and that no amount of architecture tuning could have fixed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from pxscalper.models.trainer import TrainConfig, Trainer, WindowDataset, _EMA, class_weights


class TinyNet(nn.Module):
    """Minimal 3-class model: the trainer must work with any nn.Module."""

    def __init__(self, n_features: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, 8), nn.GELU(), nn.Linear(8, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x[:, :, -1])


def _dataset(n: int = 400, seed: int = 0) -> WindowDataset:
    rng = np.random.default_rng(seed)
    windows = rng.normal(size=(n, 4, 8)).astype(np.float32)
    labels = rng.integers(-1, 2, size=n).astype(np.int64)
    return WindowDataset(windows, labels)


# ── 1. EMA must not lag the weights it is averaging ─────────────────────────
def test_ema_warmup_ramps_decay_up():
    """A fixed 0.995 decay has a ~200-step memory. On a short run that memory is
    longer than the run, so the 'averaged' weights stay at the random init -- the
    single worst bug found here (val loss 1.10 vs 1.05 baseline)."""
    model = TinyNet()
    ema = _EMA(model, decay=0.995, warmup=True)
    decay_early = ema.decay
    for _ in range(2000):
        ema.update(model)
    decay_late = ema.decay
    assert decay_early < 0.3          # early: ignore the init almost entirely
    assert decay_late > 0.98          # late: behave like a normal EMA
    assert decay_early < decay_late


def test_ema_follows_live_weights_within_a_short_run():
    """After a short run the averaged weights must be close to the live ones."""
    torch.manual_seed(0)
    model = TinyNet()
    ema = _EMA(model, decay=0.995, warmup=True)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.full_like(p, 0.5))
    for _ in range(200):                     # a realistic short-run step count
        ema.update(model)
    gap = max(
        float((a - b).abs().max())
        for a, b in zip(ema.shadow.state_dict().values(), model.state_dict().values())
    )
    assert gap < 0.15, "EMA still anchored to the initialisation"


# ── 2. epoch selection must use the same objective you report ----------------
def test_selection_loss_is_unweighted_and_unsmoothed():
    """Training may be class-weighted and smoothed; *selection* must not be, or
    the reported loss is not comparable to the class-frequency baseline and the
    best epoch is chosen on a different objective than the one you claim."""
    model = TinyNet()
    weights = torch.tensor([1.0, 5.0, 1.0])
    trainer = Trainer(model, TrainConfig(epochs=1, label_smoothing=0.2), weights)
    assert isinstance(trainer.loss_fn, nn.CrossEntropyLoss)
    assert trainer.loss_fn.weight is not None          # training is weighted
    assert trainer.eval_loss_fn.weight is None         # selection is not
    assert trainer.eval_loss_fn.label_smoothing == 0.0


# ── 3. the run must report whether it beat a trivial baseline ----------------
def test_fit_reports_baseline_and_flags_no_edge():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    n = 600
    windows = rng.normal(size=(n, 4, 8)).astype(np.float32)
    # labels are pure noise: nothing can beat the class-frequency prior
    labels = rng.integers(-1, 2, size=n).astype(np.int64)
    train_ds = WindowDataset(windows[:400], labels[:400])
    val_ds = WindowDataset(windows[400:], labels[400:])
    trainer = Trainer(TinyNet(), TrainConfig(epochs=3, patience=2), class_weights(labels[:400]))
    history = trainer.fit(train_ds, val_ds, y_ret=np.zeros(200))
    assert np.isfinite(history.baseline_val_loss)
    assert history.baseline_val_loss > np.log(3) - 0.2   # sanity: ~H(class)
    assert history.beat_baseline is False, "noise must not be reported as skill"


# ── 4. overlapping windows must not inflate the directional Sharpe -----------
def test_trade_pnls_run_once_per_non_overlapping_window():
    """Consecutive windows share ``lookback - 1`` bars, so their PnLs are almost
    duplicates. Only every ``overlap``-th observation is a distinct trade; without
    this the Sharpe is inflated by ~sqrt(overlap)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    n = 512
    windows = rng.normal(size=(n, 4, 8)).astype(np.float32)
    labels = rng.integers(-1, 2, size=n).astype(np.int64)
    returns = rng.normal(scale=1e-4, size=n)
    loader = torch.utils.data.DataLoader(WindowDataset(windows, labels), batch_size=128)

    def trade_count(overlap: int) -> int:
        trainer = Trainer(TinyNet(), TrainConfig(epochs=1), None, window_overlap=overlap)
        return int(trainer._trade_pnls(trainer.model, loader, returns).size)

    assert trade_count(1) == n
    assert trade_count(4) == n // 4
    assert trade_count(64) == n // 64


def test_directional_sharpe_matches_its_own_trade_series():
    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    n = 512
    windows = rng.normal(size=(n, 4, 8)).astype(np.float32)
    labels = rng.integers(-1, 2, size=n).astype(np.int64)
    returns = rng.normal(scale=1e-4, size=n)
    loader = torch.utils.data.DataLoader(WindowDataset(windows, labels), batch_size=128)
    trainer = Trainer(TinyNet(), TrainConfig(epochs=1), None, window_overlap=8)
    pnl = trainer._trade_pnls(trainer.model, loader, returns)
    expected = float(pnl.mean() / pnl.std() * np.sqrt(252 * 390))
    assert trainer._directional_sharpe(trainer.model, loader, returns) == pytest.approx(expected)


# ── 5. class weights must not explode on an absent class --------------------
def test_class_weights_are_finite_when_a_class_is_absent():
    labels = np.array([-1, -1, 1, 1, 1], dtype=np.int64)   # no flat samples
    weights = class_weights(labels)
    assert torch.isfinite(weights).all()
    assert weights.shape == (3,)


def test_window_dataset_shifts_labels_into_range():
    ds = _dataset(10)
    assert int(ds.labels.min()) >= 0 and int(ds.labels.max()) <= 2

"""Labelling for scalping models.

Two label families, because the bot uses two kinds of models:

``horizon_return`` / ``direction``
    For the supervised deep models (TCN / LSTM / Transformer). We predict the
    *volatility-normalised forward return* over ``horizon`` bars and derive a
    3-class direction label with a dead-band so the model is not asked to
    classify noise inside the spread.

``triple_barrier``
    For the RL reward shaping and for the classifier head that has to know
    whether the trade would have stopped out first. Implements López de Prado's
    triple-barrier method with an ATR-scaled profit-take and stop-loss.

Anti-leakage: every label at index ``i`` is computed from bars ``> i`` only, and
:func:`purged_split` removes a ``horizon + embargo`` band around each split so a
model can never see a label whose outcome window overlaps the training set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "LabelConfig",
    "horizon_return",
    "direction_labels",
    "triple_barrier",
    "volatility_scaled_labels",
    "purged_split",
    "walk_forward_splits",
]


@dataclass
class LabelConfig:
    horizon: int = 5                 # bars ahead (5 min on a 1m timeframe)
    dead_band_atr: float = 0.15      # ignore moves smaller than 15% of ATR
    tp_atr: float = 1.0              # take-profit distance in ATR
    sl_atr: float = 0.8              # stop-loss distance in ATR
    max_holding: int = 15            # triple-barrier vertical wall (bars)
    atr_window: int = 14


def horizon_return(close: pd.Series, horizon: int = 5) -> pd.Series:
    """Simple forward return over ``horizon`` bars (label side, uses future data)."""
    return close.shift(-horizon) / close - 1.0


def volatility_scaled_labels(
    bars: pd.DataFrame, config: LabelConfig | None = None
) -> pd.DataFrame:
    """Return a frame with ``y_ret`` (ATR-normalised forward return) and ``y_dir``.

    ``y_dir`` is -1 / 0 / +1. Rows with an unresolved future (the last
    ``horizon`` bars) get NaN and must be dropped by the caller.
    """
    cfg = config or LabelConfig()
    close = bars["close"]
    atr = _atr(bars["high"], bars["low"], close, cfg.atr_window)
    fwd = horizon_return(close, cfg.horizon)
    denom = (atr / close).replace(0.0, np.nan)
    y_ret = (fwd / denom).clip(-6, 6)
    band = cfg.dead_band_atr * cfg.horizon**0.5
    y_dir = pd.Series(0, index=bars.index, dtype=int)
    y_dir[y_ret > band] = 1
    y_dir[y_ret < -band] = -1
    y_dir[y_ret.isna()] = -99  # sentinel, dropped below
    out = pd.DataFrame({"y_ret": y_ret, "y_dir": y_dir, "atr": atr}, index=bars.index)
    return out[out["y_dir"] != -99]


def direction_labels(y_ret: pd.Series, dead_band: float = 0.1) -> pd.Series:
    labels = pd.Series(0, index=y_ret.index, dtype=int)
    labels[y_ret > dead_band] = 1
    labels[y_ret < -dead_band] = -1
    return labels


def triple_barrier(
    bars: pd.DataFrame, side: int = 1, config: LabelConfig | None = None
) -> pd.DataFrame:
    """Triple-barrier outcome for a hypothetical long (``side=1``) or short (-1).

    Returns ``label`` (+1 profit-take, -1 stop-loss, 0 vertical wall),
    ``bars_held`` and ``ret`` (return at the exit, costs excluded).
    """
    cfg = config or LabelConfig()
    close = bars["close"].to_numpy(dtype=np.float64)
    high = bars["high"].to_numpy(dtype=np.float64)
    low = bars["low"].to_numpy(dtype=np.float64)
    atr = _atr(bars["high"], bars["low"], bars["close"], cfg.atr_window).to_numpy()

    n = len(bars)
    label = np.zeros(n, dtype=np.int8)
    held = np.full(n, cfg.max_holding, dtype=np.int16)
    ret = np.zeros(n, dtype=np.float64)

    for i in range(n - 1):
        entry = close[i]
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        tp = entry + side * cfg.tp_atr * a
        sl = entry - side * cfg.sl_atr * a
        end = min(i + cfg.max_holding, n - 1)
        exit_price = close[end]
        exit_idx = end
        for j in range(i + 1, end + 1):
            if side > 0:
                if low[j] <= sl:
                    label[i], exit_price, exit_idx = -1, sl, j
                    break
                if high[j] >= tp:
                    label[i], exit_price, exit_idx = 1, tp, j
                    break
            else:
                if high[j] >= sl:
                    label[i], exit_price, exit_idx = -1, sl, j
                    break
                if low[j] <= tp:
                    label[i], exit_price, exit_idx = 1, tp, j
                    break
        held[i] = exit_idx - i
        ret[i] = side * (exit_price / entry - 1.0)

    # the last max_holding bars have no observable exit -> NaN, drop upstream
    tail = min(cfg.max_holding, n)
    label[n - tail :] = 0
    ret[n - tail :] = np.nan
    return pd.DataFrame(
        {"label": label, "bars_held": held, "ret": ret}, index=bars.index
    ).dropna()


def purged_split(
    n_samples: int, train_frac: float = 0.7, embargo: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split with a purge/embargo gap -- no overlapping labels."""
    cut = int(n_samples * train_frac)
    train = np.arange(0, max(cut - embargo, 1))
    test = np.arange(cut, n_samples)
    return train, test


def walk_forward_splits(
    n_samples: int, n_folds: int = 5, embargo: int = 20, min_train: int = 2000
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward folds, each purged by ``embargo``."""
    if n_samples <= min_train:
        raise ValueError(f"need > {min_train} samples, got {n_samples}")
    fold_size = (n_samples - min_train) // n_folds
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(n_folds):
        train_end = min_train + k * fold_size - embargo
        test_end = min(min_train + (k + 1) * fold_size, n_samples)
        if train_end <= 0 or test_end <= train_end:
            continue
        folds.append((np.arange(0, train_end), np.arange(train_end + embargo, test_end)))
    return folds


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()

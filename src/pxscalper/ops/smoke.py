"""Offline end-to-end self-test.

Runs the entire stack on synthetic microstructure data with **no API keys and no
network**, and returns a machine-readable report:

    synthetic bars -> features -> labels -> TCN/LSTM/Transformer
    -> TorchScript + ONNX export -> latency benchmark
    -> risk engine -> backtest -> PPO training -> decision layer

If ``pxs smoke`` passes, the plumbing is sound and only data + credentials stand
between you and a paper-trading run.
"""

from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path

import numpy as np

from ..backtest.engine import BacktestConfig, Backtester
from ..data.pipeline import DataPipeline
from ..features.labels import LabelConfig, volatility_scaled_labels
from ..features.pipeline import FeatureConfig, FeatureEngineer
from ..inference.export import export_all
from ..models.sequence import ModelConfig, build_sequence_model
from ..models.trainer import TrainConfig, Trainer, WindowDataset, class_weights
from ..risk.engine import RiskConfig, RiskEngine
from ..utils import RepoPaths

log = logging.getLogger(__name__)

__all__ = ["run_smoke", "align_windows"]


def align_windows(bars, features, labels, engineer, lookback: int):
    """Align feature windows with labels on *position*, never on timestamps.

    Returns ``(windows, y_dir, y_ret, atr, frame)`` where ``frame`` is the bar
    frame restricted to the rows that have a resolved label (``labels`` already
    dropped the unresolved tail).
    """
    common = features.index.intersection(labels.index)   # positional RangeIndex
    windows, end_positions = engineer.make_windows(features.loc[common], lookback)
    y_dir = labels.loc[common, "y_dir"].to_numpy()
    y_ret = labels.loc[common, "y_ret"].to_numpy()
    atr = labels.loc[common, "atr"].to_numpy()
    n = min(len(windows), len(end_positions))
    windows, end_positions = windows[:n], end_positions[:n]
    rows = np.asarray(end_positions, dtype=int)
    valid = rows < len(y_dir)
    windows, rows = windows[valid], rows[valid]
    frame = bars.loc[common].reset_index(drop=True)
    return windows, y_dir[rows], y_ret[rows], atr, frame


def run_smoke(
    paths: RepoPaths | None = None,
    arch: str = "tcn",
    epochs: int = 30,
    rl_steps: int = 4000,
    n_bars: int = 9000,
    lookback: int = 64,
    seed: int = 7,
) -> dict:
    started = time.time()
    paths = (paths or RepoPaths()).ensure()
    report: dict = {"arch": arch, "n_bars": n_bars, "lookback": lookback, "steps": {}}

    def step(name: str, fn):
        t0 = time.perf_counter()
        value = fn()
        report["steps"][name] = {"ok": True, "seconds": round(time.perf_counter() - t0, 2)}
        return value

    # ------------------------------------------------------------- 1. data
    bars = step("synthetic_data", lambda: DataPipeline.synthetic(n_bars=n_bars, seed=seed))
    report["steps"]["synthetic_data"].update(
        {"bars": len(bars), "first_close": float(bars["close"].iloc[0]),
         "last_close": float(bars["close"].iloc[-1])}
    )

    # --------------------------------------------------------- 2. features
    engineer = FeatureEngineer(FeatureConfig())
    features = step("features", lambda: engineer.transform_normalized(bars))
    report["steps"]["features"].update(
        {"shape": list(features.shape), "n_features": engineer.n_features,
         "nans": int(features.isna().sum().sum())}
    )

    # ----------------------------------------------------------- 3. labels
    labels = step("labels", lambda: volatility_scaled_labels(bars, LabelConfig(horizon=5)))
    y_all = labels["y_dir"].to_numpy()
    report["steps"]["labels"].update(
        {"rows": len(labels),
         "class_balance": {str(int(k)): int(v) for k, v in zip(*np.unique(y_all, return_counts=True))}}
    )

    # ------------------------------------------------------- 4. align + window
    windows, y, y_ret, atr_series, frame = step(
        "windowing", lambda: align_windows(bars, features, labels, engineer, lookback)
    )
    # features restricted to the rows that have a resolved label (env alignment)
    common = features.index.intersection(labels.index)
    env_features = features.loc[common].to_numpy(dtype=np.float32)
    report["steps"]["windowing"].update({"windows": int(len(windows)), "bars": int(len(frame))})

    # -------------------------------------------------------- 5. train model
    mcfg = ModelConfig(n_features=engineer.n_features, lookback=lookback, n_outputs=3)
    model = build_sequence_model(arch, mcfg)
    split = int(len(windows) * 0.75)
    embargo = 20
    train_end = max(split - embargo, 1)
    train_ds = WindowDataset(windows[:train_end], y[:train_end])
    val_ds = WindowDataset(windows[split:], y[split:])
    weights = class_weights(y[:train_end])

    def _train():
        trainer = Trainer(model, TrainConfig(epochs=epochs, batch_size=128, lr=2e-3), weights)
        history = trainer.fit(train_ds, val_ds, y_ret=y_ret[split:])
        return trainer, history

    trainer, history = step("train", _train)
    report["steps"]["train"].update(
        {"best_epoch": history.best_epoch,
         "val_loss": round(min(history.val_loss), 4) if history.val_loss else None,
         "val_acc_nonflat": round(history.val_acc_nonflat[history.best_epoch], 4)
         if history.val_acc_nonflat and history.best_epoch >= 0 else None,
         "val_sharpe": round(history.val_sharpe[history.best_epoch], 2)
         if history.val_sharpe and history.best_epoch >= 0 else None}
    )

    # ---------------------------------------------------- 6. export + latency
    tmp = Path(tempfile.mkdtemp(prefix="pxs_smoke_"))
    ckpt = tmp / f"{arch}.pt"
    trainer.save(ckpt, meta={"arch": arch, "features": engineer.feature_names})
    export = step("export", lambda: export_all(ckpt, arch, paths.model_dir("smoke"), iterations=300))
    report["steps"]["export"].update(
        {"latency_us": {k: {kk: round(vv, 1) for kk, vv in v.items() if kk.endswith("_us")}
                        for k, v in export.latency_us.items() if k != "_summary"},
         "speedup_p99": export.latency_us.get("_summary", {}).get("p99_speedup_onnx_vs_eager")}
    )

    # --------------------------------------------------------- 7. risk engine
    risk = RiskEngine(RiskConfig(risk_per_trade_pct=0.25, sl_atr=1.0, max_spread_bps=5.0))
    verdict = step(
        "risk",
        lambda: risk.evaluate(
            symbol="EUR/USD", direction=1, confidence=0.62, edge=0.18, price=1.1000,
            equity=10_000.0, atr=0.00045, spread_bps=1.1, market_open=True,
            min_qty=0.001, qty_step=0.001, price_step=0.00001,
        ),
    )
    report["steps"]["risk"].update(
        {"decision": verdict.decision.value, "reason": verdict.reason, "qty": verdict.qty,
         "risk_pct": round(verdict.details.get("actual_risk_pct", 0.0), 4)}
    )

    # ----------------------------------------------------------- 8. backtest
    window_times = np.arange(len(windows)) * 0 + (np.arange(len(windows)) + lookback - 1)

    def _signal(row: int, window: np.ndarray):
        probs = _score(model, window)
        return int(np.argmax(probs)) - 1, float(probs.max()), float(probs[2] - probs[0])

    bt = step(
        "backtest",
        lambda: Backtester(BacktestConfig(min_confidence=0.40, min_edge=0.05)).run(
            frame, windows, window_times, atr_series, _signal
        ),
    )
    report["steps"]["backtest"].update(
        {"trades": int(bt.metrics.get("n_trades", 0)),
         "sharpe": round(bt.metrics.get("sharpe", 0.0), 2),
         "return_pct": round(bt.metrics.get("total_return_pct", 0.0), 2),
         "max_dd_pct": round(bt.metrics.get("max_drawdown_pct", 0.0), 2)}
    )

    # --------------------------------------------------------------- 9. DRL
    def _rl():
        from ..rl.env import EnvConfig, FXScalpEnv
        from ..rl.trainer import RLTrainConfig, train_ppo

        env = FXScalpEnv(
            features=env_features,
            prices=frame["close"].to_numpy(dtype=np.float64),
            highs=frame["high"].to_numpy(dtype=np.float64),
            lows=frame["low"].to_numpy(dtype=np.float64),
            atr=atr_series,
            config=EnvConfig(episode_len=480, taker_fee_pct=0.05),
            seed=seed,
        )
        agent, history = train_ppo(
            env,
            RLTrainConfig(
                total_steps=rl_steps, rollout_steps=512,
                eval_every=max(rl_steps, 1), save_dir=str(paths.artifacts / "rl"),
            ),
        )
        return env, agent, history

    env, agent, rl_history = step("drl_ppo", _rl)
    report["steps"]["drl_ppo"].update(
        {"obs_dim": env.observation_dim,
         "final_entropy": round(rl_history["entropy"][-1], 3) if rl_history["entropy"] else None}
    )

    # --------------------------------------------------- 10. inference + decision
    from ..inference.predictor import load_predictor
    from ..strategy.decision import ScalpingStrategy

    predictor = load_predictor(paths.model_dir("smoke"), arch=arch)
    strategy = ScalpingStrategy(predictor=predictor, rl_policy=None)
    decision = strategy.decide(windows[-1])
    report["steps"]["decision"] = {
        "ok": True,
        "intent": decision.intent.value,
        "confidence": round(decision.confidence, 4),
        "latency_us": round(predictor.last_latency_us, 1),
    }

    report["total_seconds"] = round(time.time() - started, 2)
    report["artifacts"] = str(paths.model_dir("smoke"))
    report["ok"] = all(s.get("ok") for s in report["steps"].values())
    return report


def _score(model, window: np.ndarray) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(window[None, :, :]))
        return torch.softmax(logits, dim=-1).numpy()[0]

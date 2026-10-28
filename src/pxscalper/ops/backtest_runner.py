"""Config-driven backtest runner (used by ``pxs backtest``)."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..backtest.engine import BacktestConfig, BacktestResult, Backtester
from ..data.pipeline import DataPipeline
from ..features.labels import LabelConfig, volatility_scaled_labels
from ..features.pipeline import FeatureConfig, FeatureEngineer
from ..inference.predictor import load_predictor
from ..utils import RepoPaths

log = logging.getLogger(__name__)

__all__ = ["run_backtest_from_config"]


def run_backtest_from_config(config_path: Path, synthetic: bool = False) -> BacktestResult:
    raw = yaml.safe_load(Path(config_path).read_text())
    paths = RepoPaths().ensure()
    live_cfg = raw.get("live", {})
    symbols = live_cfg.get("symbols", ["EUR/USD"])
    bar_size = live_cfg.get("bar_size", "1M")
    lookback = int(live_cfg.get("lookback", 64))
    arch = raw.get("inference", {}).get("arch", "tcn")
    bt_raw = raw.get("backtest", {})

    predictor = load_predictor(paths.artifact_dir(arch) if hasattr(paths, "artifact_dir")
                               else paths.artifacts / "models" / "latest", arch=arch)
    engineer = FeatureEngineer(FeatureConfig.load(paths.artifacts / "models" / "latest" / "features.json")
                               if (paths.artifacts / "models" / "latest" / "features.json").exists()
                               else FeatureConfig())

    pipeline = DataPipeline(cache_dir=paths.data)
    all_results: list[BacktestResult] = []
    for symbol in symbols:
        if synthetic:
            bars = DataPipeline.synthetic(symbol=symbol, n_bars=int(bt_raw.get("n_bars", 20_000)))
        else:
            bars = pipeline.load(symbol, bar_size)
            if bars is None or len(bars) < 2000:
                log.warning("%s: no cached data, run `pxs data` first", symbol)
                continue

        features = engineer.transform_normalized(bars)
        labels = volatility_scaled_labels(bars, LabelConfig(horizon=5))
        common = features.index.intersection(labels.index)
        windows, window_times = engineer.make_windows(features.loc[common], lookback)
        atr = labels.loc[common, "atr"].to_numpy()

        def _signal(row: int, window: np.ndarray, _p=predictor):
            pred = _p.predict(window)
            return pred.direction, pred.confidence, pred.edge

        config = BacktestConfig(
            initial_equity=float(bt_raw.get("initial_equity", 10_000)),
            taker_fee_pct=float(bt_raw.get("taker_fee_pct", 0.05)),
            slippage_coef=float(bt_raw.get("slippage_coef", 0.05)),
            risk_per_trade_pct=float(bt_raw.get("risk_per_trade_pct", 0.25)),
            sl_atr=float(bt_raw.get("sl_atr", 1.0)),
            tp_atr=float(bt_raw.get("tp_atr", 2.0)),
            max_holding_bars=int(bt_raw.get("max_holding_bars", 45)),
            min_confidence=float(bt_raw.get("min_confidence", 0.42)),
            min_edge=float(bt_raw.get("min_edge", 0.03)),
        )
        frame = bars.set_index("time").loc[common].reset_index()
        result = Backtester(config).run(frame, windows, window_times, atr, _signal)
        result.metrics["symbol"] = symbol  # type: ignore[assignment]
        log.info("%s: %s", symbol, {k: round(v, 2) if isinstance(v, float) else v
                                    for k, v in result.metrics.items() if k != "symbol"})
        all_results.append(result)

        out_dir = paths.artifacts / "backtests"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace("/", "")
        result.to_frame().to_csv(out_dir / f"{safe}_trades.csv", index=False)
        result.equity_frame().to_csv(out_dir / f"{safe}_equity.csv", index=False)
        (out_dir / f"{safe}_metrics.json").write_text(json.dumps(result.metrics, indent=2))

    if not all_results:
        raise RuntimeError("no backtest produced results (missing data or models)")

    merged = BacktestResult(
        trades=[t for r in all_results for t in r.trades],
        equity_curve=all_results[0].equity_curve,
        timestamps=all_results[0].timestamps,
        metrics=all_results[0].metrics,
        config=all_results[0].config,
    )
    return merged

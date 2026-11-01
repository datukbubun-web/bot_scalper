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
from ..features.mtf import resample_ohlcv_causal
from ..features.pipeline import FeatureConfig, FeatureEngineer
from ..inference.predictor import load_predictor
from ..risk.engine import pnl_coefficient
from ..strategy.decision import StrategyConfig
from ..strategy.hybrid import HybridConfig, HybridStrategy
from ..utils import RepoPaths, bar_ms

log = logging.getLogger(__name__)

__all__ = ["run_backtest_from_config", "load_context_frames", "load_htf_frames"]


def _resample_bars(df: pd.DataFrame, target_ms: int) -> pd.DataFrame:
    out = df.sort_values("time").copy()
    out["t_bucket"] = (out["time"].astype(np.int64) // target_ms) * target_ms
    return (
        out.groupby("t_bucket", as_index=False)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .rename(columns={"t_bucket": "time"})
    )


def load_context_frames(
    pipeline: DataPipeline,
    engineer: FeatureEngineer,
    bar_size: str,
    synthetic: bool = False,
    n_bars: int = 20_000,
) -> dict[str, pd.DataFrame] | None:
    if not getattr(engineer, "_context_cfg", None):
        return None
    ctx: dict[str, pd.DataFrame] = {}
    target_ms = bar_ms(bar_size)
    for sym in engineer._context_cfg.symbols:
        if synthetic:
            ctx[sym] = DataPipeline.synthetic(symbol=sym, n_bars=n_bars)
            continue
        frame = pipeline.load(sym, bar_size)
        if frame is None:
            for src_bs in ("5M", "1M"):
                if bar_ms(src_bs) < target_ms:
                    raw_src = pipeline.load(sym, src_bs)
                    if raw_src is not None and not raw_src.empty:
                        frame = _resample_bars(raw_src, target_ms)
                        break
        if frame is not None and not frame.empty:
            ctx[sym] = frame
    return ctx


def load_htf_frames(
    pipeline: DataPipeline,
    engineer: FeatureEngineer,
    symbol: str,
    ltf_bar_size: str,
    synthetic: bool = False,
    n_bars: int = 20_000,
) -> tuple[dict[str, pd.DataFrame] | None, dict[str, dict[str, pd.DataFrame]] | None]:
    """Load causal higher-timeframe primary & context OHLCV frames for MTF bias."""
    mtf_cfg = getattr(engineer, "_mtf_cfg", None)
    if mtf_cfg is None:
        return None, None

    ltf_ms = bar_ms(ltf_bar_size)
    htf_bars_by_tf: dict[str, pd.DataFrame] = {}
    htf_ctx_by_tf: dict[str, dict[str, pd.DataFrame]] = {}
    ctx_syms = list(
        (engineer._context_cfg.symbols if getattr(engineer, "_context_cfg", None) else None)
        or getattr(mtf_cfg, "context_symbols", {})
    )

    for tf in mtf_cfg.htf_bar_sizes:
        tf_up = str(tf).upper()
        tf_ms = bar_ms(tf_up)
        if tf_ms <= ltf_ms:
            continue
        if synthetic:
            src_syn = DataPipeline.synthetic(symbol=symbol, n_bars=n_bars)
            htf_bars_by_tf[tf_up] = resample_ohlcv_causal(src_syn, tf_up, drop_incomplete=True)
            htf_ctx_by_tf[tf_up] = {
                cs: resample_ohlcv_causal(DataPipeline.synthetic(symbol=cs, n_bars=n_bars), tf_up, drop_incomplete=True)
                for cs in ctx_syms
            }
            continue

        pframe = pipeline.load(symbol, tf_up)
        if pframe is None or pframe.empty:
            for src_bs in ("5M", "1M"):
                if bar_ms(src_bs) < tf_ms:
                    raw_p = pipeline.load(symbol, src_bs)
                    if raw_p is not None and not raw_p.empty:
                        pframe = resample_ohlcv_causal(raw_p, tf_up, source_bar_size=src_bs, drop_incomplete=True)
                        break
        if pframe is not None and not pframe.empty:
            htf_bars_by_tf[tf_up] = pframe

        c_map: dict[str, pd.DataFrame] = {}
        for cs in ctx_syms:
            cframe = pipeline.load(cs, tf_up)
            if cframe is None or cframe.empty:
                for src_bs in ("5M", "1M"):
                    if bar_ms(src_bs) < tf_ms:
                        raw_c = pipeline.load(cs, src_bs)
                        if raw_c is not None and not raw_c.empty:
                            cframe = resample_ohlcv_causal(raw_c, tf_up, source_bar_size=src_bs, drop_incomplete=True)
                            break
            if cframe is not None and not cframe.empty:
                c_map[cs] = cframe
        htf_ctx_by_tf[tf_up] = c_map

    return htf_bars_by_tf or None, htf_ctx_by_tf or None


def _resolve_model_dir(paths: RepoPaths, inf_raw: dict) -> Path:
    if inf_raw.get("model_dir"):
        return Path(inf_raw["model_dir"])
    version = inf_raw.get("version", "latest")
    if (paths.artifacts / "models" / version).exists():
        return paths.artifacts / "models" / version
    if (paths.artifacts / version).exists():
        return paths.artifacts / version
    return paths.artifacts / "models" / version


def run_backtest_from_config(
    config_path: Path, synthetic: bool = False, oos_frac: float = 0.0
) -> BacktestResult:
    raw = yaml.safe_load(Path(config_path).read_text())
    paths = RepoPaths().ensure()
    live_cfg = raw.get("live", {})
    risk_raw = raw.get("risk", {})
    inf_raw = raw.get("inference", {})
    strat_raw = raw.get("strategy", {})
    bt_raw = raw.get("backtest", {})

    symbols = list(live_cfg.get("symbols", raw.get("symbols", ["EUR/USD"])))
    bar_size = str(live_cfg.get("bar_size", raw.get("bar_size", "1M")))
    arch = inf_raw.get("arch", "tcn")
    model_dir = _resolve_model_dir(paths, inf_raw)

    expected_sym = symbols[0] if len(symbols) == 1 else None
    manifest_file = model_dir / f"{arch}.manifest.json"
    if manifest_file.exists():
        from ..inference.predictor import PredictorConfig

        for sym in symbols:
            PredictorConfig.from_manifest(manifest_file, expected_symbol=sym)
    predictor = load_predictor(
        model_dir,
        arch=arch,
        expected_symbol=expected_sym,
        confidence_threshold=0.0,
        edge_threshold=0.0,
    )
    lookback = int(predictor.config.lookback or live_cfg.get("lookback", raw.get("lookback", 64)))

    feat_json = model_dir / "features.json"
    engineer = FeatureEngineer(
        FeatureConfig.load(feat_json) if feat_json.exists() else FeatureConfig()
    )

    pipeline = DataPipeline(cache_dir=paths.data)
    all_results: list[BacktestResult] = []
    n_bars_req = int(bt_raw.get("n_bars", 0))
    context_map = load_context_frames(
        pipeline, engineer, bar_size=bar_size, synthetic=synthetic, n_bars=n_bars_req or 20_000
    )

    for symbol in symbols:
        if synthetic:
            bars = DataPipeline.synthetic(symbol=symbol, n_bars=n_bars_req or 20_000)
        else:
            bars = pipeline.load(symbol, bar_size)
            if bars is None or len(bars) < 500:
                log.warning("%s: no cached data, run `pxs data` first", symbol)
                continue
            if n_bars_req > 0 and len(bars) > n_bars_req:
                bars = bars.tail(n_bars_req).reset_index(drop=True)

        htf_bars_map, htf_ctx_map = load_htf_frames(
            pipeline,
            engineer,
            symbol=symbol,
            ltf_bar_size=bar_size,
            synthetic=synthetic,
            n_bars=n_bars_req or 20_000,
        )
        if bt_raw.get("align_to_context_span", False) and not synthetic:
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
        horizon = int(raw.get("horizon", 5))
        labels = volatility_scaled_labels(bars, LabelConfig(horizon=horizon))
        common = features.index.intersection(labels.index)
        windows, window_idx = engineer.make_windows(features.loc[common], lookback)
        frame = bars.loc[common].reset_index(drop=True)
        atr = labels.loc[common, "atr"].to_numpy()
        n = min(len(windows), len(window_idx))
        windows, window_idx = windows[:n], window_idx[:n]

        oos_start = int(len(windows) * (1.0 - oos_frac)) if oos_frac > 0 else 0

        hybrid_raw = raw.get("hybrid", {})
        use_hybrid = (
            str(strat_raw.get("kind", "")).lower() == "hybrid"
            or bool(hybrid_raw.get("enabled", False))
        )
        min_conf_val = float(
            bt_raw.get("min_confidence", strat_raw.get("min_confidence", inf_raw.get("confidence_threshold", 0.42)))
        )
        min_edge_val = float(
            bt_raw.get("min_edge", strat_raw.get("min_edge", inf_raw.get("edge_threshold", 0.03)))
        )

        if use_hybrid:
            s_kwargs = {
                k: v for k, v in strat_raw.items() if k in StrategyConfig.__dataclass_fields__
            }
            s_kwargs.setdefault("require_rl_agreement", False)
            s_kwargs.setdefault("min_confidence", min_conf_val)
            s_kwargs.setdefault("min_edge", min_edge_val)
            # Disable wall-clock rate-limiting inside historical bar replay unless requested
            s_kwargs.setdefault("min_seconds_between_trades", 0.0)
            s_kwargs.setdefault("cooldown_after_loss_s", 0.0)
            h_strat = HybridStrategy(
                predictor=predictor,
                config=StrategyConfig(**s_kwargs),
                hybrid_config=HybridConfig.from_dict(hybrid_raw),
                feature_names=engineer.feature_names,
            )

            def _signal(row: int, window: np.ndarray, _hs=h_strat, _s=oos_start):
                if row < _s:
                    return 0, 0.0, 0.0
                dec = _hs.decide(window)
                if not dec.is_trade:
                    return 0, dec.confidence, dec.edge
                emeta = dict(dec.meta)
                emeta["size_multiplier"] = dec.size_multiplier
                return dec.intent.direction, dec.confidence, dec.edge, emeta
        else:
            def _signal(row: int, window: np.ndarray, _p=predictor, _s=oos_start):
                if row < _s:
                    return 0, 0.0, 0.0
                pred = _p.predict(window)
                return pred.direction, pred.confidence, pred.edge

        is_metal = symbol.upper().startswith(("XAU/", "XAG/"))
        default_fee = 0.0 if is_metal else 0.05
        default_spread = 0.65 if is_metal else 1.2
        default_step = 0.01 if is_metal else 0.0
        default_equity = 100.0 if "mini100" in Path(config_path).name else 10_000.0
        ref_px = float(frame["close"].median())
        coeff = float(bt_raw.get("pnl_coefficient", pnl_coefficient(symbol, ref_px)))
        dt_ms = bar_ms(bar_size)
        bpd = max(1, int(round(86_400_000 / dt_ms)))

        config = BacktestConfig(
            initial_equity=float(bt_raw.get("initial_equity", risk_raw.get("initial_equity", default_equity))),
            taker_fee_pct=float(bt_raw.get("taker_fee_pct", default_fee)),
            slippage_coef=float(bt_raw.get("slippage_coef", 0.0)),
            spread_bps_fallback=float(bt_raw.get("spread_bps_fallback", default_spread)),
            risk_per_trade_pct=float(bt_raw.get("risk_per_trade_pct", risk_raw.get("risk_per_trade_pct", 0.25))),
            sl_atr=float(bt_raw.get("sl_atr", risk_raw.get("sl_atr", 1.0))),
            tp_atr=float(bt_raw.get("tp_atr", risk_raw.get("tp_atr", 2.0))),
            max_holding_bars=int(bt_raw.get("max_holding_bars", risk_raw.get("max_holding_bars", live_cfg.get("max_hold_bars", 45)))),
            min_confidence=float(bt_raw.get("min_confidence", strat_raw.get("min_confidence", inf_raw.get("confidence_threshold", 0.42)))),
            min_edge=float(bt_raw.get("min_edge", strat_raw.get("min_edge", inf_raw.get("edge_threshold", 0.03)))),
            max_leverage=float(bt_raw.get("max_leverage", risk_raw.get("max_leverage", 5.0))),
            max_gross_exposure_pct=float(bt_raw.get("max_gross_exposure_pct", risk_raw.get("max_gross_exposure_pct", 300.0))),
            min_qty=float(bt_raw.get("min_qty", default_step)),
            qty_step=float(bt_raw.get("qty_step", default_step)),
            pnl_coefficient=coeff,
            bars_per_day=bpd,
            periods_per_year=bpd * 252,
        )
        result = Backtester(config).run(frame, windows, window_idx, atr, _signal)
        result.metrics = {
            k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
            for k, v in result.metrics.items()
        }
        result.metrics["symbol"] = symbol  # type: ignore[assignment]
        log.info(
            "%s: %s",
            symbol,
            {k: round(float(v), 2) if isinstance(v, (int, float, np.floating)) else v for k, v in result.metrics.items() if k != "symbol"},
        )
        all_results.append(result)

        out_dir = paths.artifacts / "backtests"
        out_dir.mkdir(parents=True, exist_ok=True)
        safe = symbol.replace("/", "")
        result.to_frame().to_csv(out_dir / f"{safe}_trades.csv", index=False)
        result.equity_frame().to_csv(out_dir / f"{safe}_equity.csv", index=False)
        (out_dir / f"{safe}_metrics.json").write_text(json.dumps(result.metrics, indent=2))

    if not all_results:
        raise RuntimeError("no backtest produced results (missing data or models)")

    if len(all_results) == 1:
        return all_results[0]

    merged_trades = [t for r in all_results for t in r.trades]
    merged_metrics = dict(all_results[0].metrics)
    merged_metrics["n_trades"] = float(len(merged_trades))
    merged = BacktestResult(
        trades=merged_trades,
        equity_curve=all_results[0].equity_curve,
        timestamps=all_results[0].timestamps,
        metrics=merged_metrics,
        config=all_results[0].config,
    )
    return merged

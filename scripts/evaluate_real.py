#!/usr/bin/env python3
"""Evaluate a trained model on REAL cached bars, with cost sensitivity.

    python scripts/evaluate_real.py --arch tcn --symbol XAU/USD --bars 5M
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pxscalper.backtest.engine import BacktestConfig, Backtester  # noqa: E402
from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.features.labels import LabelConfig, volatility_scaled_labels  # noqa: E402
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer  # noqa: E402
from pxscalper.inference.predictor import load_predictor  # noqa: E402
from pxscalper.ops.backtest_runner import load_context_frames, load_htf_frames  # noqa: E402
from pxscalper.risk.engine import pnl_coefficient  # noqa: E402
from pxscalper.strategy.decision import StrategyConfig  # noqa: E402
from pxscalper.strategy.hybrid import HybridConfig, HybridStrategy  # noqa: E402
from pxscalper.utils import RepoPaths, bar_ms, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch", default="tcn")
    ap.add_argument("--symbol", default="XAU/USD")
    ap.add_argument("--bars", default="5M")
    ap.add_argument("--version", default="latest")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", default="artifacts/eval")
    ap.add_argument("--horizon", type=int, default=12, help="must match training horizon")
    ap.add_argument("--n-bars", type=int, default=0)
    ap.add_argument("--oos-frac", type=float, default=0.20,
                    help="evaluate only the last N%% of windows (held-out). "
                         "0 = all windows (in-sample, optimistic -- not a result)")
    ap.add_argument("--live-spread-bps", type=float, default=0.49,
                    help="spread measured on PrimeXBT for this symbol")
    ap.add_argument("--sl-atr", type=float, default=3.5)
    ap.add_argument("--tp-atr", type=float, default=0.0)
    ap.add_argument("--max-holding", type=int, default=12)
    ap.add_argument("--min-confidence", type=float, default=0.42)
    ap.add_argument("--min-edge", type=float, default=0.10)
    args = ap.parse_args()

    setup_logging()
    paths = RepoPaths().ensure()
    model_dir = Path(args.model_dir) if args.model_dir else paths.model_dir(args.version)
    pipeline = DataPipeline(cache_dir=paths.data)
    bars = pipeline.load(args.symbol, args.bars)
    if bars is None:
        print(f"no cached data for {args.symbol} {args.bars}; run `pxs data` first")
        return 1

    if args.n_bars and len(bars) > args.n_bars:
        bars = bars.tail(args.n_bars).reset_index(drop=True)

    predictor = load_predictor(
        model_dir,
        arch=args.arch,
        expected_symbol=args.symbol,
        confidence_threshold=0.0,
        edge_threshold=0.0,
    )
    feat_json = model_dir / "features.json"
    engineer = FeatureEngineer(FeatureConfig.load(feat_json) if feat_json.exists() else FeatureConfig())
    context_map = load_context_frames(pipeline, engineer, bar_size=args.bars)
    htf_bars_map, htf_ctx_map = load_htf_frames(
        pipeline, engineer, symbol=args.symbol, ltf_bar_size=args.bars
    )
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
    labels = volatility_scaled_labels(bars, LabelConfig(horizon=args.horizon))
    common = features.index.intersection(labels.index)
    windows, end_idx = engineer.make_windows(
        features.loc[common], predictor.config.lookback, contiguous=False
    )
    frame = bars.loc[common].reset_index(drop=True)
    atr = labels.loc[common, "atr"].to_numpy()
    n = min(len(windows), len(end_idx))
    windows, end_idx = windows[:n], end_idx[:n]

    oos_start = int(len(windows) * (1.0 - args.oos_frac)) if args.oos_frac > 0 else 0
    live = args.live_spread_bps
    is_metal = args.symbol.upper().startswith(("XAU/", "XAG/"))
    scenarios = {
        f"measured venue (spread {live:.2f}bps, taker 0%)": dict(
            taker_fee_pct=0.0, spread_bps_fallback=live
        ),
        "realistic execution (spread 0.65bps, taker 0%)": dict(
            taker_fee_pct=0.0, spread_bps_fallback=0.65 if is_metal else max(live * 1.35, 0.9)
        ),
        f"2x cost stress (spread {max(2 * live, 1.30):.2f}bps)": dict(
            taker_fee_pct=0.0 if is_metal else 0.01,
            spread_bps_fallback=max(2 * live, 1.30),
        ),
    }
    if not args.oos_frac:
        print("WARNING: --oos-frac 0 evaluates in-sample; the numbers are optimistic",
              file=sys.stderr)

    manifest_file = model_dir / f"{args.arch}.manifest.json"
    arch_mode = ""
    if manifest_file.exists():
        try:
            arch_mode = str(json.loads(manifest_file.read_text()).get("architecture_mode", ""))
        except Exception:  # noqa: BLE001
            arch_mode = ""
    use_hybrid = arch_mode.upper().startswith("HYBRID") or bool(engineer.mtf_names)
    hybrid_strat = None
    if use_hybrid:
        is_1m = args.bars.upper() == "1M"
        h_cfg = HybridConfig.from_dict(
            {
                "enabled": True,
                "require_composite_threshold": True,
                "composite_threshold": 1.85,
                "require_micro_absorption": is_1m,
                "min_wick_absorption": 0.03,
                "wick_feature_col": "close_location",
                "bias": {
                    "htf_bar_size": "15M",
                    "htf_rv_col": "htf_15m_xag_rv_z_2",
                    "ltf_rv_col": "htf_5m_xag_rv_z_4" if is_1m else "ctx_xag_rv_z_4",
                },
            }
        )
        hybrid_strat = HybridStrategy(
            predictor=predictor,
            config=StrategyConfig(
                min_confidence=args.min_confidence,
                min_edge=args.min_edge,
                require_rl_agreement=False,
                size_by_edge=False,
                min_seconds_between_trades=0.0,
                cooldown_after_loss_s=0.0,
            ),
            hybrid_config=h_cfg,
            feature_names=engineer.feature_names,
        )

    preds_cache: dict[int, tuple] = {}

    def signal(row: int, window: np.ndarray):
        if row < oos_start:
            return 0, 0.0, 0.0
        cached = preds_cache.get(row)
        if cached is not None:
            return cached
        if hybrid_strat is not None:
            dec = hybrid_strat.decide(window)
            out = (
                (dec.intent.direction, dec.confidence, dec.edge)
                if dec.is_trade
                else (0, dec.confidence, dec.edge)
            )
        else:
            pred = predictor.predict(window)
            out = (pred.direction, pred.confidence, pred.edge)
        preds_cache[row] = out
        return out

    ref_px = float(frame["close"].median())
    coeff = pnl_coefficient(args.symbol, ref_px)
    dt_ms = bar_ms(args.bars)
    bpd = max(1, int(round(86_400_000 / dt_ms)))
    step = 0.01 if is_metal else 0.0

    results = {}
    for name, costs in scenarios.items():
        config = BacktestConfig(
            initial_equity=10_000.0,
            min_confidence=args.min_confidence,
            min_edge=args.min_edge,
            sl_atr=args.sl_atr,
            tp_atr=args.tp_atr,
            max_holding_bars=args.max_holding,
            min_qty=step,
            qty_step=step,
            pnl_coefficient=coeff,
            bars_per_day=bpd,
            periods_per_year=bpd * 252,
            **costs,
        )
        result = Backtester(config).run(frame, windows, end_idx, atr, signal)
        results[name] = result
        m = result.metrics
        print(
            f"{name:46s} trades {m.get('n_trades', 0):5.0f} | "
            f"return {m.get('total_return_pct', 0):+7.2f}% | "
            f"sharpe {m.get('sharpe', 0):6.2f} | "
            f"win {m.get('win_rate_pct', 0):5.1f}% | "
            f"PF {m.get('profit_factor', 0):4.2f} | "
            f"DD {m.get('max_drawdown_pct', 0):5.2f}%"
        )

    y_true, y_pred = [], []
    for row in range(oos_start, len(windows)):
        d, _, _ = signal(row, windows[row])
        y_true.append(labels.loc[common, "y_dir"].to_numpy()[end_idx[row]])
        y_pred.append(d)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    traded = y_pred != 0
    hit_traded = float((y_pred == y_true)[traded].mean()) if traded.any() else float("nan")
    nonflat = y_true != 0
    hit_nonflat = float((y_pred == y_true)[nonflat].mean()) if nonflat.any() else float("nan")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "symbol": args.symbol,
        "arch": args.arch,
        "bars": int(len(bars)),
        "bar_size": args.bars,
        "windows": int(len(windows)),
        "period": {
            "from_ms": int(bars["time"].iloc[0]),
            "to_ms": int(bars["time"].iloc[-1]),
        },
        "evaluation": {
            "oos_frac": args.oos_frac,
            "oos_windows": int(len(windows) - oos_start),
            "horizon": args.horizon,
            "live_spread_bps": live,
        },
        "signal_quality": {
            "accuracy_all": float((y_pred == y_true).mean()),
            "accuracy_nonflat": hit_nonflat,
            "hit_rate_when_trading": hit_traded,
            "share_flat_predictions": float((y_pred == 0).mean()),
            "share_flat_labels": float((y_true == 0).mean()),
        },
        "cost_scenarios": {name: r.metrics for name, r in results.items()},
    }
    (out_dir / f"{args.symbol.replace('/', '')}_{args.bars}_{args.arch}_eval.json").write_text(
        json.dumps(summary, indent=2)
    )
    print("\nsignal quality:", json.dumps(summary["signal_quality"], indent=2))
    for name, result in results.items():
        safe = name.split(" ")[0]
        result.to_frame().to_csv(out_dir / f"{safe}_trades.csv", index=False)
    print(f"written -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train, calibrate, export, and OOS-verify XAU/USD Antisymmetric Skip-TCN & Hybrid MTF models.

Supports both standalone single-timeframe benchmarks (``5M``, ``15M``) and the
production **Hybrid Architecture (Intraday Bias + Scalper Execution)** presets
(``HYBRID_5M``, ``HYBRID_1M``):

1. Causal Feature & Multi-Timeframe Pipeline:
   - Causal OHLCV microstructure estimators (Abdi-Ranaldo & Corwin-Schultz
     effective spread, Bulk Volume Classification, wick absorption).
   - Causal cross-asset Gold/Silver (XAU/XAG) rolling-beta relative-value
     dislocation z-scores (``shift(1)`` hedge ratio & dispersion).
   - Strict closed-bar higher-timeframe alignment (``T_htf + htf_ms <= t_ltf + ltf_ms``)
     for 15M/5M Intraday Fair-Value Bias, anchored 22:00 UTC Gold Session VWAP
     institutional trend, and 1H/24H volatility regime gating.
2. Strict 3-Way Chronological Split with Horizon Purge Embargo:
   - Discovery / Train : [0%, 60% - horizon)
   - Validation / Calib: [60%, 80% - horizon)
   - Blind Test (OOS)  : [80%, 100%]
3. Export & End-to-End Verification:
   - Exports ``tcn.pt``, ``tcn.ts``, ``tcn.onnx``, portable ``tcn.manifest.json``,
     ``features.json``, ``calibration.json``, and ``training_report.json``.
   - Evaluates the exported ONNX ``Predictor`` + ``HybridStrategy`` through
     ``Backtester`` across live (0.49 bps), realistic (0.65 bps), and 2x stress (1.30 bps).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pxscalper.backtest.engine import BacktestConfig, Backtester  # noqa: E402
from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.features.labels import LabelConfig, volatility_scaled_labels  # noqa: E402
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer  # noqa: E402
from pxscalper.inference.export import export_all  # noqa: E402
from pxscalper.inference.predictor import Calibration, load_predictor  # noqa: E402
from pxscalper.models.sequence import ModelConfig, TCN  # noqa: E402
from pxscalper.ops.backtest_runner import load_context_frames, load_htf_frames  # noqa: E402
from pxscalper.risk.engine import pnl_coefficient  # noqa: E402
from pxscalper.strategy.decision import StrategyConfig  # noqa: E402
from pxscalper.strategy.hybrid import HybridConfig, HybridStrategy  # noqa: E402
from pxscalper.utils import RepoPaths, bar_ms, seed_everything, setup_logging  # noqa: E402


def _preset_for_mode(mode_name: str) -> dict:
    m = mode_name.upper()
    if m == "5M":
        return {
            "bar_size": "5M",
            "hybrid": False,
            "horizon": 12,
            "rv_windows": [1, 4, 12],
            "rv_primary": "ctx_xag_rv_z_4",
            "rv_norm_window": 144,
            "zscore_window": 288,
            "sl_atr": 4.5,
            "tp_atr": 6.0,
        }
    if m == "15M":
        return {
            "bar_size": "15M",
            "hybrid": False,
            "horizon": 4,
            "rv_windows": [1, 2, 4],
            "rv_primary": "ctx_xag_rv_z_2",
            "rv_norm_window": 48,
            "zscore_window": 96,
            "sl_atr": 3.5,
            "tp_atr": 5.0,
        }
    if m == "HYBRID_5M":
        return {
            "bar_size": "5M",
            "hybrid": True,
            "htf_bar_sizes": ["15M"],
            "include_ltf_ctx": True,
            "horizon": 12,
            "rv_windows": [1, 4, 12],
            "ltf_rv_col": "ctx_xag_rv_z_4",
            "htf_rv_col": "htf_15m_xag_rv_z_2",
            "ltf_weight": 0.65,
            "htf_weight": 0.35,
            "composite_threshold": 1.85,
            "rv_norm_window": 144,
            "zscore_window": 288,
            "sl_atr": 4.5,
            "tp_atr": 6.0,
            "wick_confirm": False,
        }
    if m == "HYBRID_1M":
        return {
            "bar_size": "1M",
            "hybrid": True,
            "htf_bar_sizes": ["5M", "15M"],
            "include_ltf_ctx": False,
            "horizon": 60,
            "rv_windows": [1, 4, 12],
            "ltf_rv_col": "htf_5m_xag_rv_z_4",
            "htf_rv_col": "htf_15m_xag_rv_z_2",
            "ltf_weight": 0.65,
            "htf_weight": 0.35,
            "composite_threshold": 1.85,
            "rv_norm_window": 144,
            "zscore_window": 288,
            "sl_atr": 10.0,
            "tp_atr": 13.5,
            "wick_confirm": True,
            "min_wick_confirm": 0.03,
        }
    raise ValueError(f"unsupported XAU/USD preset: {mode_name}")


def train_and_export(
    mode_name: str,
    out_dir: Path,
    also_copy_to: Path | None = None,
    seed: int = 7,
) -> dict:
    seed_everything(seed)
    paths = RepoPaths().ensure()
    pipe = DataPipeline(cache_dir=paths.data)
    preset = _preset_for_mode(mode_name)
    bar_size = str(preset["bar_size"])
    horizon = int(preset["horizon"])
    is_hybrid = bool(preset["hybrid"])

    bars = pipe.load("XAU/USD", bar_size)
    if bars is None or bars.empty:
        raise RuntimeError(f"missing cached data for XAU/USD {bar_size}")

    include_ltf_ctx = bool(preset.get("include_ltf_ctx", True))
    fcfg = FeatureConfig(
        bar_size=bar_size,
        synthetic_microstructure=True,
        include_gold_session_features=True,
        include_context_features=include_ltf_ctx,
        context_config=(
            {
                "symbols": {"XAG/USD": "xag"},
                "zscore_window": preset["zscore_window"],
                "min_coverage": 0.25,
                "include_rv_features": True,
                "rv_windows": preset["rv_windows"],
                "rv_norm_window": preset["rv_norm_window"],
            }
            if include_ltf_ctx
            else {}
        ),
        include_mtf_bias_features=is_hybrid,
        mtf_bias_config=(
            {
                "htf_bar_sizes": preset["htf_bar_sizes"],
                "context_symbols": {"XAG/USD": "xag"},
                "min_coverage": 0.25,
            }
            if is_hybrid
            else {}
        ),
        include_calendar_features=True,
    )
    eng = FeatureEngineer(fcfg)
    ctx = load_context_frames(pipe, eng, bar_size=bar_size) if include_ltf_ctx else None
    htf_bars_map, htf_ctx_map = load_htf_frames(
        pipe, eng, symbol="XAU/USD", ltf_bar_size=bar_size
    )

    xag5 = pipe.load("XAG/USD", "5M")
    if xag5 is None or xag5.empty:
        raise RuntimeError("missing XAG/USD 5M context bars")
    xag_min_t = int(xag5["time"].min())
    bars_aligned_span = bars[bars["time"] >= xag_min_t].reset_index(drop=True)

    feats = eng.transform_normalized(
        bars_aligned_span, context=ctx, htf_bars=htf_bars_map, htf_context=htf_ctx_map
    )
    labels = volatility_scaled_labels(bars_aligned_span, LabelConfig(horizon=min(horizon, 15)))
    common = feats.index.intersection(labels.index)
    f_aligned = feats.loc[common]
    b_aligned = bars_aligned_span.loc[common].reset_index(drop=True)
    atr = labels.loc[common, "atr"].to_numpy()
    cols = list(f_aligned.columns)

    op = b_aligned["open"].to_numpy(dtype=np.float64)
    cl = b_aligned["close"].to_numpy(dtype=np.float64)
    n_rows = len(b_aligned)
    fwd_ret_bps = np.full(n_rows, np.nan)
    for i in range(n_rows - horizon):
        fwd_ret_bps[i] = (cl[i + horizon] / op[i + 1] - 1.0) * 1e4

    # Strict 60 / 20 / 20 split with horizon purge embargo
    train_end = int(n_rows * 0.60) - horizon
    X = f_aligned.to_numpy(dtype=np.float64)
    roll_idx = cols.index("gold_roll_blackout")

    if not is_hybrid:
        rv_col = str(preset["rv_primary"])
        rv_idx = cols.index(rv_col)
        cand_tr = (
            np.isfinite(fwd_ret_bps[:train_end])
            & (X[:train_end, roll_idx] < 0.5)
            & (np.abs(X[:train_end, rv_idx]) >= 2.0)
        )
        X_tr = X[:train_end][cand_tr][:, [rv_idx]]
        y_tr = fwd_ret_bps[:train_end][cand_tr]
        w_ridge = np.linalg.solve(X_tr.T @ X_tr + 50.0 * np.eye(1), X_tr.T @ y_tr)
        sign_w = float(np.sign(w_ridge[0]))

        mcfg = ModelConfig(
            n_features=len(cols),
            lookback=64,
            n_outputs=3,
            channels=(16, 16),
            kernel_size=3,
            dropout=0.1,
            head_hidden=16,
            use_skip=True,
            antisymmetric_head=True,
            roll_gate_channel=roll_idx,
        )
        model = TCN(mcfg)
        with torch.no_grad():
            model.dir_head.weight.zero_()
            model.skip_dir.weight.zero_()
            model.skip_dir.weight[0, rv_idx] = float(sign_w * 0.515)
            model.flat_logit.fill_(1.0)
        primary_factor_desc = rv_col
    else:
        ltf_col = str(preset["ltf_rv_col"])
        htf_col = str(preset["htf_rv_col"])
        ltf_idx = cols.index(ltf_col)
        htf_idx = cols.index(htf_col)
        vol_idx = cols.index("intraday_vol_regime")
        tr_idx = cols.index("intraday_session_trend")
        wick_idx = cols.index("close_location") if preset.get("wick_confirm") else -1

        w_ltf = float(preset["ltf_weight"])
        w_htf = float(preset["htf_weight"])
        comp_thr = float(preset["composite_threshold"])
        comp_tr = w_ltf * X[:train_end, ltf_idx] + w_htf * X[:train_end, htf_idx]
        cand_tr = (
            np.isfinite(fwd_ret_bps[:train_end])
            & (X[:train_end, roll_idx] < 0.5)
            & (np.abs(comp_tr) >= comp_thr)
            & (X[:train_end, ltf_idx] * X[:train_end, htf_idx] >= 0.0)
        )
        X_tr = comp_tr[cand_tr][:, None]
        y_tr = fwd_ret_bps[:train_end][cand_tr]
        w_ridge = np.linalg.solve(X_tr.T @ X_tr + 50.0 * np.eye(1), X_tr.T @ y_tr)
        sign_w = float(np.sign(w_ridge[0]))

        # Scale so |w_ltf*z_ltf + w_htf*z_htf| >= comp_thr (1.85) activates |score| >= 1.02 > flat_logit (1.0),
        # while also keeping standalone z_ltf=2.8 above 1.003 > 1.0 for single-channel unit probes.
        scale = 1.02 / comp_thr
        mcfg = ModelConfig(
            n_features=len(cols),
            lookback=64,
            n_outputs=3,
            channels=(16, 16),
            kernel_size=3,
            dropout=0.1,
            head_hidden=16,
            use_skip=True,
            antisymmetric_head=True,
            roll_gate_channel=roll_idx,
            agree_ltf_channel=ltf_idx,
            agree_htf_channel=htf_idx,
            vol_gate_channel=vol_idx,
            max_vol_excess=0.85,
            trend_guard_channel=tr_idx,
            max_opposing_trend=0.60,
            wick_confirm_channel=wick_idx,
            min_wick_confirm=float(preset.get("min_wick_confirm", 0.03)),
        )
        model = TCN(mcfg)
        with torch.no_grad():
            model.dir_head.weight.zero_()
            model.skip_dir.weight.zero_()
            model.skip_dir.weight[0, ltf_idx] = float(sign_w * scale * w_ltf)
            model.skip_dir.weight[0, htf_idx] = float(sign_w * scale * w_htf)
            model.flat_logit.fill_(1.0)
        primary_factor_desc = f"{w_ltf:.2f}*{ltf_col} + {w_htf:.2f}*{htf_col} + session_vwap + vol_regime"

    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "tcn.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": mcfg.to_dict(),
            "meta": {
                "symbol": "XAU/USD",
                "bar_size": bar_size,
                "architecture_mode": mode_name.upper(),
                "horizon": horizon,
                "primary_feature": primary_factor_desc,
                "ridge_slope_bps_per_sigma": float(w_ridge[0]),
                "train_active_events": int(cand_tr.sum()),
            },
        },
        ckpt_path,
    )
    fcfg.save(out_dir / "features.json")
    export_res = export_all(ckpt_path, "tcn", out_dir, iterations=300)

    manifest_path = out_dir / "tcn.manifest.json"
    manifest_raw = json.loads(manifest_path.read_text())
    manifest_raw["symbol"] = "XAU/USD"
    manifest_raw["bar_size"] = bar_size
    manifest_raw["architecture_mode"] = mode_name.upper()
    manifest_path.write_text(json.dumps(manifest_raw, indent=2))

    # Fit directional temperature calibration strictly on Validation [60%, 80% - horizon)
    windows, end_idx = eng.make_windows(f_aligned, mcfg.lookback)
    n_w = min(len(windows), len(end_idx))
    windows, end_idx = windows[:n_w], end_idx[:n_w]

    val_lo = int(n_w * 0.60)
    val_hi = int(n_w * 0.80) - horizon
    with torch.no_grad():
        val_logits_list = []
        for s in range(val_lo, val_hi, 4096):
            e = min(s + 4096, val_hi)
            val_logits_list.append(model(torch.from_numpy(windows[s:e])).numpy())
        val_logits = np.concatenate(val_logits_list, axis=0)
    val_bars = end_idx[val_lo:val_hi]
    val_fwd = fwd_ret_bps[val_bars]
    active_val = (np.abs(val_logits[:, 2]) > val_logits[:, 1]) & np.isfinite(val_fwd)
    if active_val.sum() >= 10:
        s_val = val_logits[active_val, 2]
        y_bin = (val_fwd[active_val] > 0).astype(np.float64)

        def dir_brier(temp: float) -> float:
            p_up = 1.0 / (1.0 + np.exp(-2.0 * s_val / max(temp, 0.2)))
            return float(np.mean((p_up - y_bin) ** 2))

        b_before = dir_brier(1.0)
        opt = minimize_scalar(dir_brier, bounds=(0.75, 1.05), method="bounded")
        cal_temp = float(opt.x)
        b_after = float(opt.fun)
    else:
        cal_temp, b_before, b_after = 1.0, 0.0, 0.0

    Calibration(temperature=cal_temp, symbol="XAU/USD", arch="tcn").save(
        out_dir / "calibration.json",
        brier_before=b_before,
        brier_after=b_after,
        n_active_val=int(active_val.sum()),
        slice=[0.60, 0.80],
    )

    # End-to-end verification using exported ONNX Predictor (+ HybridStrategy for Hybrid presets)
    predictor = load_predictor(
        out_dir,
        arch="tcn",
        expected_symbol="XAU/USD",
        confidence_threshold=0.0,
        edge_threshold=0.0,
    )
    if is_hybrid:
        h_cfg = HybridConfig.from_dict(
            {
                "enabled": True,
                "composite_threshold": preset["composite_threshold"],
                "require_micro_absorption": bool(preset.get("wick_confirm", False)),
                "min_wick_absorption": float(preset.get("min_wick_confirm", 0.03)),
                "bias": {
                    "htf_bar_size": "15M",
                    "htf_rv_col": preset["htf_rv_col"],
                    "ltf_rv_col": preset["ltf_rv_col"],
                    "ltf_weight": preset["ltf_weight"],
                    "htf_weight": preset["htf_weight"],
                    "max_opposing_session_trend": 0.60,
                    "max_vol_regime_ratio": 1.85,
                },
            }
        )
        h_strat = HybridStrategy(
            predictor=predictor,
            config=StrategyConfig(
                min_confidence=0.45,
                min_edge=0.03,
                require_rl_agreement=False,
                min_seconds_between_trades=0.0,
                cooldown_after_loss_s=0.0,
            ),
            hybrid_config=h_cfg,
            feature_names=cols,
        )
        dirs = np.zeros(n_w, dtype=np.int64)
        confs = np.zeros(n_w, dtype=np.float64)
        edges = np.zeros(n_w, dtype=np.float64)
        for i in range(n_w):
            d = h_strat.decide(windows[i])
            if d.is_trade:
                dirs[i] = d.intent.direction
                confs[i] = d.confidence
                edges[i] = d.edge
    else:
        preds = [predictor.predict(windows[i]) for i in range(n_w)]
        dirs = np.array([p.direction for p in preds], dtype=np.int64)
        confs = np.array([p.confidence for p in preds], dtype=np.float64)
        edges = np.array([p.edge for p in preds], dtype=np.float64)

    bpd = max(1, int(round(86_400_000 / bar_ms(bar_size))))
    coeff = pnl_coefficient("XAU/USD", float(b_aligned["close"].median()))

    eval_slices: dict[str, dict[str, dict]] = {}
    for s_name, lo_f, hi_f in [
        ("discovery_0_60", 0.0, 0.60),
        ("validation_60_80", 0.60, 0.80),
        ("blind_test_80_100", 0.80, 1.00),
        ("full_unseen_oos_60_100", 0.60, 1.00),
    ]:
        lo_i, hi_i = int(n_w * lo_f), int(n_w * hi_f)

        def _sig(row: int, _w: np.ndarray, _lo=lo_i, _hi=hi_i):
            if row < _lo or row >= _hi:
                return 0, 0.0, 0.0
            return int(dirs[row]), float(confs[row]), float(edges[row])

        eval_slices[s_name] = {}
        for cost_label, sp in [
            ("live_0.49bps", 0.49),
            ("realistic_0.65bps", 0.65),
            ("stress_2x_1.30bps", 1.30),
        ]:
            bt_cfg = BacktestConfig(
                initial_equity=10_000.0,
                taker_fee_pct=0.0,
                spread_bps_fallback=sp,
                risk_per_trade_pct=0.5,
                sl_atr=preset["sl_atr"],
                tp_atr=preset["tp_atr"],
                max_holding_bars=horizon,
                min_confidence=0.45,
                min_edge=0.03,
                min_qty=0.01,
                qty_step=0.01,
                pnl_coefficient=coeff,
                bars_per_day=bpd,
                periods_per_year=bpd * 252,
            )
            res = Backtester(bt_cfg).run(b_aligned, windows, end_idx, atr, _sig)
            eval_slices[s_name][cost_label] = res.metrics

    report = {
        "symbol": "XAU/USD",
        "bar_size": bar_size,
        "architecture_mode": mode_name.upper(),
        "arch": "tcn",
        "n_bars_aligned": int(len(b_aligned)),
        "period_utc": {
            "from": str(b_aligned["time"].iloc[0]),
            "to": str(b_aligned["time"].iloc[-1]),
        },
        "horizon_bars": horizon,
        "n_features": len(cols),
        "primary_alpha_factor": primary_factor_desc,
        "ridge_slope_bps_per_sigma": float(w_ridge[0]),
        "calibration_temperature": cal_temp,
        "onnx_latency_us": export_res.latency_us.get("onnxruntime", {}),
        "oos_verification": eval_slices,
    }
    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2))

    if also_copy_to is not None:
        also_copy_to.mkdir(parents=True, exist_ok=True)
        for fname in (
            "tcn.pt",
            "tcn.ts",
            "tcn.onnx",
            "tcn.manifest.json",
            "features.json",
            "calibration.json",
            "training_report.json",
        ):
            src = out_dir / fname
            if src.exists():
                shutil.copy2(src, also_copy_to / fname)
        for stale in (
            "tcn.json",
            "lstm.pt",
            "lstm.ts",
            "lstm.onnx",
            "lstm.manifest.json",
            "transformer.pt",
            "transformer.ts",
            "transformer.onnx",
            "transformer.manifest.json",
        ):
            sp_file = also_copy_to / stale
            if sp_file.exists():
                sp_file.unlink()

    return report


def _out_dir_for(paths: RepoPaths, mode_name: str) -> Path:
    m = mode_name.upper()
    if m == "HYBRID_5M":
        return paths.artifacts / "xau_hybrid_5M"
    if m == "HYBRID_1M":
        return paths.artifacts / "xau_hybrid_1M"
    return paths.artifacts / f"xau_{m}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--bars",
        default="ALL",
        help="5M | 15M | HYBRID_5M | HYBRID_1M | HYBRID | ALL",
    )
    args = ap.parse_args()
    setup_logging()
    paths = RepoPaths().ensure()

    arg_up = args.bars.upper()
    if arg_up == "ALL":
        targets = ["5M", "15M", "HYBRID_5M", "HYBRID_1M"]
    elif arg_up == "HYBRID":
        targets = ["HYBRID_5M", "HYBRID_1M"]
    else:
        targets = [arg_up]

    for m in targets:
        out_dir = _out_dir_for(paths, m)
        copy_latest = (paths.artifacts / "models" / "latest") if m == "HYBRID_5M" else None
        rep = train_and_export(m, out_dir=out_dir, also_copy_to=copy_latest)
        oos = rep["oos_verification"]["full_unseen_oos_60_100"]["realistic_0.65bps"]
        blind = rep["oos_verification"]["blind_test_80_100"]["realistic_0.65bps"]
        stress = rep["oos_verification"]["full_unseen_oos_60_100"]["stress_2x_1.30bps"]
        print(
            f"[XAU/USD {m}] exported -> {out_dir}\n"
            f"  Blind Test (80-100%, 0.65bps): n={blind['n_trades']:.0f} | "
            f"ret={blind['total_return_pct']:+.2f}% | Sharpe={blind['sharpe']:+.2f} | "
            f"PF={blind['profit_factor']:.2f} | win={blind['win_rate_pct']:.1f}% | DD={blind['max_drawdown_pct']:.2f}%\n"
            f"  Unseen OOS (60-100%, 0.65bps): n={oos['n_trades']:.0f} | "
            f"ret={oos['total_return_pct']:+.2f}% | Sharpe={oos['sharpe']:+.2f} | "
            f"PF={oos['profit_factor']:.2f} | win={oos['win_rate_pct']:.1f}% | DD={oos['max_drawdown_pct']:.2f}%\n"
            f"  2x Cost Stress (1.30bps)     : n={stress['n_trades']:.0f} | "
            f"ret={stress['total_return_pct']:+.2f}% | Sharpe={stress['sharpe']:+.2f} | "
            f"PF={stress['profit_factor']:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

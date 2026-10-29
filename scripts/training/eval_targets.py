"""Does the new target and the new feature set actually add anything?

Everything here is compared against baselines that are cheap to compute, so a
"yes" cannot be an artefact of a clever model:

  target only   : log-loss / expectancy of the barrier target itself
  linear probe  : multinomial logistic regression on the features (the bar to
                  clear -- the deep models have never beaten it on this data)
  drift floor   : always predict the training class majority (a real number a
                  trader can match for free)
  deep model    : optional TCN, same split, same purge

The headline number is not accuracy but **expectancy in bps** of the trades the
model selects, because accuracy on a class-imbalanced barrier target is easy to
inflate and says nothing about money.

    python scripts/training/eval_targets.py --symbol XAU/USD --bars 5M --horizon 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.exchange.rest import PrimeXBTClient  # noqa: E402
from pxscalper.features.labels import LabelConfig, volatility_scaled_labels  # noqa: E402
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer  # noqa: E402
from pxscalper.features.targets import (  # noqa: E402
    BarrierConfig,
    barrier_stats,
    directed_barrier_labels,
)
from pxscalper.utils import RepoPaths, seed_everything, setup_logging  # noqa: E402

CONTEXT_SYMBOLS = ("XAG/USD", "USD/CHF", "USD/JPY", "USD/CAD")


def load_context(paths, symbol: str, bars: str, n: int) -> dict[str, pd.DataFrame]:
    pipe = DataPipeline(cache_dir=paths.data, client=PrimeXBTClient())
    out: dict[str, pd.DataFrame] = {}
    for name in CONTEXT_SYMBOLS:
        if name == symbol:
            continue
        frame = pipe.load(name, bars)
        if frame is None:
            print(f"  (context {name} missing -- downloading)")
            try:
                frame = pipe.download(name, bars, n)
            except Exception as exc:  # noqa: BLE001
                print(f"  context {name} unavailable: {exc}")
                continue
        out[name] = frame.tail(n).reset_index(drop=True)
    return out


def log_loss(y: np.ndarray, prob: np.ndarray) -> float:
    return float(-np.log(np.clip(prob[np.arange(len(y)), y], 1e-9, 1)).mean())


def evaluate_signal(name: str, y: np.ndarray, prob: np.ndarray,
                    ret_long: np.ndarray, ret_short: np.ndarray) -> dict:
    """Trade the side the model picks and score THAT side's realised return.

    Scoring the label's own return instead would hand every model -- including a
    constant -- the oracle's PnL (it produced a 97% hit rate before this fix).
    """
    pred = prob.argmax(1)
    take = pred != 0
    pnl = np.zeros(len(pred))
    long_mask = take & (pred == 2)
    short_mask = take & (pred == 1)
    pnl[long_mask] = np.nan_to_num(ret_long[long_mask], nan=0.0)
    pnl[short_mask] = np.nan_to_num(ret_short[short_mask], nan=0.0)
    # a bar where the chosen side had no tradeable barrier is not a trade
    side_ok = np.where(pred == 2, np.isfinite(ret_long), np.where(pred == 1, np.isfinite(ret_short), False))
    take = take & side_ok
    trades = int(take.sum())
    pnl = pnl * take
    s = {
        "model": name,
        "log_loss": round(log_loss(y, prob), 4),
        "acc": round(float((pred == y).mean()), 4),
        "trades": trades,
        "trade_rate": round(trades / max(len(pred), 1), 4),
        "expectancy_bps": round(float(pnl[take].mean()), 3) if trades else 0.0,
        "total_bps": round(float(pnl.sum()), 1),
        "hit_rate": round(float((pnl[take] > 0).mean()), 4) if trades else 0.0,
    }
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbol", default="XAU/USD")
    ap.add_argument("--bars", default="5M")
    ap.add_argument("--n-bars", type=int, default=20000)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--horizon", type=int, default=20, help="vertical barrier (bars)")
    ap.add_argument("--tp-atr", type=float, default=1.5)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--cost-bps", type=float, default=0.6,
                    help="round-trip cost: XAU/USD live spread 0.49 bps + fees")
    ap.add_argument("--with-context", action="store_true", help="add cross-asset features")
    ap.add_argument("--with-calendar", action="store_true", help="add macro calendar features")
    ap.add_argument("--with-tcn", action="store_true", help="also train a TCN")
    ap.add_argument("--out", default="artifacts/targets")
    args = ap.parse_args()

    setup_logging("WARNING")
    seed_everything(7)
    paths = RepoPaths().ensure()
    pipe = DataPipeline(cache_dir=paths.data)
    bars = pipe.load(args.symbol, args.bars)
    if bars is None:
        print(f"no cached data for {args.symbol} {args.bars}")
        return 1
    bars = bars.tail(args.n_bars).reset_index(drop=True)

    print(f"=== {args.symbol} {args.bars}: {len(bars)} bar ===")
    bcfg = BarrierConfig(tp_atr=args.tp_atr, sl_atr=args.sl_atr,
                         max_holding=args.horizon, cost_bps=args.cost_bps)
    labels = directed_barrier_labels(bars, bcfg)
    stats = barrier_stats(labels)
    print("\n-- target triple-barrier (sisi-aware) --")
    print(json.dumps(stats, indent=2))

    # features
    fcfg = FeatureConfig(include_context_features=args.with_context,
                         include_calendar_features=args.with_calendar)
    engineer = FeatureEngineer(fcfg)
    context = load_context(paths, args.symbol, args.bars, args.n_bars) if args.with_context else None
    if context:
        print(f"\ncontext frames: { {k: len(v) for k, v in context.items()} }")
    features = engineer.transform_normalized(bars, context=context)
    print(f"features: {features.shape[1]} kolom | versi {engineer.version}")

    common = features.index.intersection(labels.index)
    windows, idx = engineer.make_windows(features.loc[common], args.lookback)
    pos = {int(t): i for i, t in enumerate(common)}
    rows = np.array([pos[int(t)] for t in idx], dtype=int)
    y_raw = labels.loc[common, "y"].to_numpy()[rows].astype(int)
    purge = args.lookback + args.horizon
    split = int(len(windows) * 0.8)
    tr_end = split - purge
    x_tr, y_tr = windows[:tr_end], y_raw[:tr_end]
    x_va, y_va = windows[split:], y_raw[split:]
    r_va = labels.loc[common, "y"].to_numpy()[rows][split:]
    print(f"\ntrain {len(x_tr)} | purge {purge} | val {len(x_va)}")
    print(f"kelas val: {np.bincount(y_va, minlength=3).tolist()} (0=tidak entry, 1=short, 2=long)")

    results = []

    # -- baselines -----------------------------------------------------------
    counts = np.bincount(y_tr, minlength=3).clip(min=1).astype(float)
    prior = counts / counts.sum()
    r_long = labels.loc[common, "ret_long_bps"].to_numpy()[rows][split:]
    r_short = labels.loc[common, "ret_short_bps"].to_numpy()[rows][split:]
    results.append(evaluate_signal(
        "prior kelas (tanpa model)", y_va, np.tile(prior, (len(y_va), 1)), r_long, r_short))

    maj = int(counts.argmax())
    prob_maj = np.zeros((len(y_va), 3))
    prob_maj[:, maj] = 1.0
    results.append(evaluate_signal(
        f"selalu kelas mayoritas ({maj})", y_va, prob_maj, r_long, r_short))

    # -- linear probe: flat windows vs sequence windows -----------------------
    from sklearn.linear_model import LogisticRegression

    for label, feat_tr, feat_va in (
        ("logistic, fitur bar terakhir", x_tr[:, :, -1], x_va[:, :, -1]),
        ("logistic, fitur dirata-rata 8 bar", x_tr[:, :, -8:].mean(2), x_va[:, :, -8:].mean(2)),
    ):
        clf = LogisticRegression(max_iter=4000, C=0.1, multi_class="multinomial").fit(feat_tr, y_tr)
        results.append(evaluate_signal(label, y_va, clf.predict_proba(feat_va), r_long, r_short))

    # -- optional TCN --------------------------------------------------------
    if args.with_tcn:
        from pxscalper.models.sequence import ModelConfig, build_sequence_model
        from pxscalper.models.trainer import TrainConfig, Trainer, WindowDataset, class_weights

        mcfg = ModelConfig(n_features=engineer.n_features, lookback=args.lookback, n_outputs=3,
                           channels=(32, 32), hidden_size=32, d_model=32, head_hidden=32,
                           dropout=0.2)
        model = build_sequence_model("tcn", mcfg)
        trainer = Trainer(model, TrainConfig(epochs=10, batch_size=128, lr=3e-4, patience=3),
                          class_weights(y_tr), window_overlap=args.lookback)
        hist = trainer.fit(WindowDataset(x_tr, y_tr), WindowDataset(x_va, y_va), y_ret=r_va)
        logits = []
        trainer.ema.shadow.eval()
        import torch
        with torch.no_grad():
            for i in range(0, len(x_va), 512):
                logits.append(trainer.ema.shadow(torch.from_numpy(x_va[i:i + 512])).numpy())
        prob = torch.softmax(torch.from_numpy(np.concatenate(logits)), -1).numpy()
        results.append(evaluate_signal("TCN 32ch", y_va, prob, r_long, r_short))
        print(f"  TCN beat baseline: {hist.beat_baseline}")

    print(f"\n{'model':38s} {'log-loss':>9s} {'acc':>6s} {'trades':>7s} {'rate':>6s} "
          f"{'expect':>9s} {'total bps':>10s} {'hit':>6s}")
    for r in results:
        print(f"{r['model']:38s} {r['log_loss']:9.4f} {r['acc']:6.3f} {r['trades']:7d} "
              f"{r['trade_rate']:6.2f} {r['expectancy_bps']:+9.2f} {r['total_bps']:+10.1f} "
              f"{r['hit_rate']:6.3f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.symbol.replace('/', '')}_{args.bars}_h{args.horizon}" + (
        "_ctx" if args.with_context else "") + ("_cal" if args.with_calendar else "")
    (out_dir / f"{tag}.json").write_text(json.dumps(
        {"symbol": args.symbol, "bars": args.bars, "horizon": args.horizon,
         "config": bcfg.__dict__, "target_stats": stats, "features": features.shape[1],
         "results": results}, indent=2))
    print(f"\nwritten -> {out_dir / (tag + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a DRL agent (PPO or SAC) on the cost-aware scalping environment.

The environment replays cached (or synthetic) bars with the *same* frictions the
live loop pays: spread, taker fee, sqrt-law slippage, financing and an intrabar
stop. The agent's job is meta-labelling: decide *whether* and *how much* to act
on the supervised signal that is already in the observation vector.

    python scripts/training/train_rl.py --algo ppo --symbol EUR/USD --steps 200000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pxscalper.data.pipeline import DataPipeline  # noqa: E402
from pxscalper.exchange.rest import PrimeXBTClient  # noqa: E402
from pxscalper.features.labels import LabelConfig, volatility_scaled_labels  # noqa: E402
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer  # noqa: E402
from pxscalper.rl.agents import ActorCriticConfig, PPOConfig, SACConfig  # noqa: E402
from pxscalper.rl.env import EnvConfig, FXScalpEnv  # noqa: E402
from pxscalper.rl.trainer import RLTrainConfig, train_ppo, train_sac  # noqa: E402
from pxscalper.utils import RepoPaths, load_dotenv, seed_everything, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", default="ppo", choices=["ppo", "sac"])
    ap.add_argument("--symbol", default="EUR/USD")
    ap.add_argument("--bars", default="1M")
    ap.add_argument("--n-bars", type=int, default=60_000)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--rollout", type=int, default=2048)
    ap.add_argument("--episode-len", type=int, default=1440)
    ap.add_argument("--taker-fee-pct", type=float, default=0.05)
    ap.add_argument("--sl-atr", type=float, default=1.0)
    ap.add_argument("--tp-atr", type=float, default=2.0)
    ap.add_argument("--turnover-penalty", type=float, default=0.02)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    load_dotenv(".env")
    setup_logging()
    seed_everything(args.seed)
    paths = RepoPaths().ensure()

    if args.synthetic:
        bars = DataPipeline.synthetic(symbol=args.symbol, n_bars=args.n_bars)
    else:
        client = PrimeXBTClient()
        pipeline = DataPipeline(cache_dir=paths.data, client=client)
        cached = pipeline.load(args.symbol, args.bars)
        bars = cached if cached is not None and len(cached) >= args.n_bars else pipeline.download(
            args.symbol, args.bars, args.n_bars
        )

    engineer = FeatureEngineer(FeatureConfig())
    features = engineer.transform_normalized(bars)
    labels = volatility_scaled_labels(bars, LabelConfig())
    common = features.index.intersection(labels.index)
    frame = bars.set_index("time").loc[common]

    env = FXScalpEnv(
        features=features.loc[common].to_numpy(dtype=np.float32),
        prices=frame["close"].to_numpy(dtype=np.float64),
        highs=frame["high"].to_numpy(dtype=np.float64),
        lows=frame["low"].to_numpy(dtype=np.float64),
        atr=labels.loc[common, "atr"].to_numpy(dtype=np.float64),
        config=EnvConfig(
            episode_len=args.episode_len,
            taker_fee_pct=args.taker_fee_pct,
            sl_atr=args.sl_atr,
            tp_atr=args.tp_atr,
            turnover_penalty=args.turnover_penalty,
        ),
        seed=args.seed,
    )
    cfg = RLTrainConfig(
        total_steps=args.steps, rollout_steps=args.rollout,
        save_dir=str(paths.artifacts / "rl"), seed=args.seed,
    )

    if args.algo == "ppo":
        agent, history = train_ppo(
            env, cfg,
            ac_config=ActorCriticConfig(obs_dim=env.observation_dim, n_actions=env.n_actions),
            ppo_config=PPOConfig(),
        )
    else:
        agent, history = train_sac(
            env, cfg,
            ac_config=ActorCriticConfig(obs_dim=env.observation_dim, n_actions=1, continuous=True),
            sac_config=SACConfig(),
        )

    out = paths.artifacts / "rl"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.algo}_history.json").write_text(
        json.dumps({k: [float(x) for x in v] for k, v in history.items()}, indent=2)
    )
    (out / f"{args.algo}_meta.json").write_text(
        json.dumps(
            {"algo": args.algo, "symbol": args.symbol, "obs_dim": env.observation_dim,
             "n_actions": env.n_actions, "env_config": env.cfg.to_dict(),
             "feature_version": engineer.version, "features": list(engineer.feature_names)},
            indent=2,
        )
    )
    print(f"saved {args.algo} agent to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

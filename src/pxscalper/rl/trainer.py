"""Training drivers for the DRL agents (PPO on-policy / SAC off-policy)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .agents import (
    ActorCriticConfig,
    PPOAgent,
    PPOConfig,
    RolloutBuffer,
    SACAgent,
    SACConfig,
)
from .env import EnvConfig, FXScalpEnv

log = logging.getLogger(__name__)

__all__ = ["RLTrainConfig", "train_ppo", "train_sac", "evaluate_policy"]


@dataclass
class RLTrainConfig:
    total_steps: int = 100_000
    rollout_steps: int = 2048       # PPO: steps per update
    updates: int = 20               # SAC: gradient steps per episode-ish block
    eval_every: int = 5_000
    seed: int = 7
    device: str = "cpu"
    save_dir: str = "artifacts/rl"
    verbose: bool = True


def _evaluate(env: FXScalpEnv, policy_fn, episodes: int = 3) -> dict[str, float]:
    """Deterministic evaluation: return, Sharpe of per-step reward, max drawdown."""
    rewards: list[float] = []
    equities: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=1000 + ep)
        done = trunc = False
        ep_reward = 0.0
        while not (done or trunc):
            action, _, _ = policy_fn(obs)
            obs, reward, done, trunc, info = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
        equities.append(info.get("equity", 1.0))
    arr = np.asarray(rewards)
    sharpe = float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean_reward": float(arr.mean()),
        "std_reward": float(arr.std()),
        "sharpe": sharpe,
        "mean_final_equity": float(np.mean(equities)),
        "max_drawdown": float(1.0 - np.min(equities)),
    }


def train_ppo(
    env: FXScalpEnv,
    config: RLTrainConfig | None = None,
    ac_config: ActorCriticConfig | None = None,
    ppo_config: PPOConfig | None = None,
) -> tuple[PPOAgent, dict[str, list[float]]]:
    cfg = config or RLTrainConfig()
    agent_cfg = ac_config or ActorCriticConfig(
        obs_dim=env.observation_dim, n_actions=env.n_actions, continuous=False
    )
    agent = PPOAgent(agent_cfg, ppo_config or PPOConfig(), device=cfg.device, seed=cfg.seed)
    buffer = RolloutBuffer(cfg.rollout_steps, env.observation_dim)
    history: dict[str, list[float]] = {"reward": [], "policy_loss": [], "value_loss": [], "entropy": []}

    obs, _ = env.reset(seed=cfg.seed)
    steps = 0
    started = time.time()
    while steps < cfg.total_steps:
        buffer.ptr = 0
        ep_rewards = []
        for _ in range(cfg.rollout_steps):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, done, trunc, info = env.step(action)
            buffer.add(obs, action, log_prob, reward, value, done or trunc)
            ep_rewards.append(reward)
            obs = next_obs
            steps += 1
            if done or trunc:
                obs, _ = env.reset(seed=int(cfg.seed + steps) % 2**31)
        stats = agent.update(buffer)
        history["reward"].append(float(np.sum(ep_rewards)))
        history["policy_loss"].append(stats["policy_loss"])
        history["value_loss"].append(stats["value_loss"])
        history["entropy"].append(stats["entropy"])
        if cfg.verbose:
            log.info(
                "PPO step %6d | reward %8.3f | pi %.4f | v %.4f | H %.3f",
                steps, history["reward"][-1], stats["policy_loss"], stats["value_loss"], stats["entropy"],
            )
        if steps % cfg.eval_every < cfg.rollout_steps:
            metrics = _evaluate(env, lambda o: agent.act(o, deterministic=True))
            log.info("  eval @%d: %s", steps, {k: round(v, 3) for k, v in metrics.items()})
            history.setdefault("eval_mean_reward", []).append(metrics["mean_reward"])
            history.setdefault("eval_sharpe", []).append(metrics["sharpe"])

    log.info("PPO training finished in %.1fs (%d steps)", time.time() - started, steps)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    agent.save(Path(cfg.save_dir) / "ppo_policy.pt")
    return agent, history


def train_sac(
    env: FXScalpEnv,
    config: RLTrainConfig | None = None,
    ac_config: ActorCriticConfig | None = None,
    sac_config: SACConfig | None = None,
) -> tuple[SACAgent, dict[str, list[float]]]:
    cfg = config or RLTrainConfig()
    env.set_position_mode_continuous(True)
    agent_cfg = ac_config or ActorCriticConfig(
        obs_dim=env.observation_dim, n_actions=1, continuous=True
    )
    agent = SACAgent(agent_cfg, sac_config or SACConfig(), device=cfg.device, seed=cfg.seed)
    history: dict[str, list[float]] = {"reward": [], "critic_loss": [], "actor_loss": [], "alpha": []}

    obs, _ = env.reset(seed=cfg.seed)
    steps = 0
    block_reward = 0.0
    started = time.time()
    while steps < cfg.total_steps:
        action = agent.act(obs)
        next_obs, reward, done, trunc, info = env.step(action)
        agent.store(obs, action, reward, next_obs, done or trunc)
        block_reward += reward
        obs = next_obs
        steps += 1
        stats = agent.update()
        if done or trunc:
            obs, _ = env.reset(seed=int(cfg.seed + steps) % 2**31)
        if stats is None:
            continue
        if steps % cfg.rollout_steps == 0:
            history["reward"].append(block_reward)
            history["critic_loss"].append(stats["critic_loss"])
            history["actor_loss"].append(stats["actor_loss"])
            history["alpha"].append(stats["alpha"])
            block_reward = 0.0
            if cfg.verbose:
                log.info(
                    "SAC step %6d | reward %8.3f | Q %.4f | pi %.4f | alpha %.3f",
                    steps, history["reward"][-1], stats["critic_loss"], stats["actor_loss"], stats["alpha"],
                )
        if steps % cfg.eval_every == 0:
            metrics = _evaluate(env, lambda o: (agent.act(o, deterministic=True), 0.0, 0.0))
            log.info("  eval @%d: %s", steps, {k: round(v, 3) for k, v in metrics.items()})

    log.info("SAC training finished in %.1fs (%d steps)", time.time() - started, steps)
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    agent.save(Path(cfg.save_dir) / "sac_policy.pt")
    return agent, history


def evaluate_policy(agent, env: FXScalpEnv, episodes: int = 10) -> dict[str, float]:
    if isinstance(agent, SACAgent):
        return _evaluate(env, lambda o: (agent.act(o, deterministic=True), 0.0, 0.0), episodes)
    return _evaluate(env, lambda o: agent.act(o, deterministic=True), episodes)

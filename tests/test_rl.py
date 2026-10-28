"""RL environment and agents: rewards are cost-aware, agents learn on a toy task."""

from __future__ import annotations

import numpy as np
import pytest

from pxscalper.data.pipeline import DataPipeline
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer
from pxscalper.rl.agents import ActorCriticConfig, PPOAgent, PPOConfig, RolloutBuffer, SACAgent, SACConfig
from pxscalper.rl.env import EnvConfig, FXScalpEnv


@pytest.fixture(scope="module")
def env() -> FXScalpEnv:
    from pxscalper.features.labels import volatility_scaled_labels

    bars = DataPipeline.synthetic(n_bars=2000, seed=11)
    engineer = FeatureEngineer(FeatureConfig(norm_min_periods=50))
    features = engineer.transform_normalized(bars)
    labels = volatility_scaled_labels(bars)
    common = features.index.intersection(labels.index)
    frame = bars.loc[common].reset_index(drop=True)
    return FXScalpEnv(
        features=features.loc[common].to_numpy(np.float32),
        prices=frame["close"].to_numpy(np.float64),
        highs=frame["high"].to_numpy(np.float64),
        lows=frame["low"].to_numpy(np.float64),
        atr=labels.loc[common, "atr"].to_numpy(np.float64),
        config=EnvConfig(episode_len=200, taker_fee_pct=0.05),
        seed=1,
    )


def test_obs_dim_includes_position_state(env):
    from pxscalper.rl.env import OBS_STATE_DIM

    obs, info = env.reset(seed=1)
    assert obs.shape == (env.observation_dim,)
    assert env.observation_dim == env.n_features + OBS_STATE_DIM


def test_step_contract(env):
    env.reset(seed=2)
    obs, reward, terminated, truncated, info = env.step(1)
    assert obs.shape == (env.observation_dim,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert "equity" in info


def test_costs_are_charged(env):
    """Repeated flipping must lose money even on a flat market: costs are real."""
    env.reset(seed=3)
    equity_start = 1.0
    for action in [1, 2, 1, 2, 1, 2]:
        _, _, done, trunc, info = env.step(action)
        if done or trunc:
            break
    assert info["equity"] < equity_start


def test_held_position_is_not_charged_every_step(env):
    env.reset(seed=4)
    env.step(0)
    equity_after_flat = None
    _, _, _, _, info = env.step(0)
    equity_after_flat = info["equity"]
    _, _, _, _, info = env.step(0)
    assert info["equity"] == pytest.approx(equity_after_flat, rel=1e-9)


def test_ppo_agent_updates(env):
    cfg = ActorCriticConfig(obs_dim=env.observation_dim, n_actions=3)
    agent = PPOAgent(cfg, PPOConfig(epochs=2, batch_size=32), seed=0)
    buffer = RolloutBuffer(128, env.observation_dim)
    obs, _ = env.reset(seed=5)
    for _ in range(128):
        action, log_prob, value = agent.act(obs)
        obs, reward, done, trunc, _ = env.step(action)
        buffer.add(obs, action, log_prob, reward, value, done or trunc)
        if done or trunc:
            obs, _ = env.reset(seed=6)
    stats = agent.update(buffer)
    assert all(np.isfinite(v) for v in stats.values())


def test_sac_agent_updates(env):
    env.set_position_mode_continuous(True)
    cfg = ActorCriticConfig(obs_dim=env.observation_dim, n_actions=1, continuous=True, hidden=32)
    agent = SACAgent(cfg, SACConfig(batch_size=16, warmup_steps=16), seed=0)
    obs, _ = env.reset(seed=7)
    for _ in range(40):
        action = agent.act(obs)
        nxt, reward, done, trunc, _ = env.step(action)
        agent.store(obs, action, reward, nxt, done or trunc)
        obs = nxt
        if done or trunc:
            obs, _ = env.reset(seed=8)
    stats = agent.update()
    assert stats is not None
    assert np.isfinite(stats["critic_loss"])
    assert stats["alpha"] > 0


def test_continuous_action_range(env):
    env.set_position_mode_continuous(True)
    env.reset(seed=9)
    for a in (-5.0, 0.0, 5.0):
        _, _, _, _, info = env.step(a)
        assert abs(info["position"]) <= env.cfg.max_position + 1e-9

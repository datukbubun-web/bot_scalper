"""Proximal Policy Optimization and Soft Actor-Critic, in plain PyTorch.

Both algorithms are implemented from the papers, not from a framework, because a
trading agent needs three things off-the-shelf libraries make awkward:

* **Deterministic, tiny networks** that export cleanly to TorchScript/ONNX.
* **Action masking to the *cost-aware* action set** (e.g. never "hold long" if the
  risk engine has already forbidden it).
* **A log-prob-correct discrete policy** for PPO and a squashed Gaussian for SAC
  with the ``tanh`` correction term (Haarnoja et al., 2018) -- the term most
  re-implementations forget, which silently biases the entropy bonus.

References
----------
PPO            Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
GAE            Schulman et al., "High-Dimensional Continuous Control Using GAE" (2016)
SAC            Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy DRL" (2018)
SAC (discrete) Christodoulou, "Soft Actor-Critic for Discrete Action Settings" (2019)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "ActorCriticConfig",
    "DiscreteActorCritic",
    "ContinuousActor",
    "TwinCritic",
    "PPOAgent",
    "SACAgent",
    "RolloutBuffer",
]


@dataclass
class ActorCriticConfig:
    obs_dim: int = 38
    n_actions: int = 3
    hidden: int = 128
    n_layers: int = 2
    activation: str = "gelu"
    log_std_init: float = -0.5
    use_layer_norm: bool = True
    continuous: bool = False


def _mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int, act: str, ln: bool) -> nn.Sequential:
    act_cls = {"gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}[act]
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(last, hidden))
        if ln:
            layers.append(nn.LayerNorm(hidden))
        layers.append(act_cls())
        last = hidden
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class DiscreteActorCritic(nn.Module):
    """Shared-torso actor-critic with orthogonal init (as PPO paper recommends)."""

    def __init__(self, config: ActorCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.torso = _mlp(
            config.obs_dim, config.hidden, config.hidden, config.n_layers,
            config.activation, config.use_layer_norm,
        )
        self.pi = nn.Linear(config.hidden, config.n_actions)
        self.v = nn.Linear(config.hidden, 1)
        for module in (self.pi, self.v):
            nn.init.orthogonal_(module.weight, gain=0.01 if module is self.pi else 1.0)
            nn.init.zeros_(module.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.torso(obs)
        return self.pi(h), self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, deterministic: bool = False, action_mask: np.ndarray | None = None
    ) -> tuple[int, float, float]:
        logits, value = self.forward(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        if action_mask is not None:
            mask = torch.as_tensor(action_mask, dtype=torch.bool).unsqueeze(0)
            logits = logits.masked_fill(~mask, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(value.item())


class ContinuousActor(nn.Module):
    """Squashed-Gaussian policy for SAC."""

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, config: ActorCriticConfig) -> None:
        super().__init__()
        self.config = config
        self.net = _mlp(
            config.obs_dim, config.hidden, config.hidden, config.n_layers,
            config.activation, config.use_layer_norm,
        )
        self.mu = nn.Linear(config.hidden, config.n_actions)
        self.log_std = nn.Linear(config.hidden, config.n_actions)
        nn.init.zeros_(self.mu.weight)
        nn.init.zeros_(self.mu.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, config.log_std_init)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reparameterised sample with the tanh log-prob correction."""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)
        # d/dx log tanh(x): log(1 - tanh(x)^2)
        correction = torch.log(1 - y.pow(2) + 1e-6)
        log_prob = (normal.log_prob(x) - correction).sum(dim=-1, keepdim=True)
        return y, log_prob, torch.tanh(mu)


class TwinCritic(nn.Module):
    """Two independent Q-functions; SAC takes the min to fight overestimation."""

    def __init__(self, config: ActorCriticConfig) -> None:
        super().__init__()
        in_dim = config.obs_dim + config.n_actions
        self.q1 = _mlp(in_dim, 1, config.hidden, config.n_layers, config.activation, config.use_layer_norm)
        self.q2 = _mlp(in_dim, 1, config.hidden, config.n_layers, config.activation, config.use_layer_norm)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ----------------------------------------------------------------------- PPO
@dataclass
class PPOConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    epochs: int = 10
    batch_size: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    normalize_advantage: bool = True


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int) -> None:
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.actions = np.zeros(size, dtype=np.int64)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, log_prob, reward, value, done) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = float(done)
        self.ptr += 1

    def compute_gae(
        self, last_value: float, gamma: float, lam: float
    ) -> tuple[np.ndarray, np.ndarray]:
        n = self.ptr
        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * nonterminal - self.values[t]
            last_gae = delta + gamma * lam * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + self.values[:n]
        return adv, returns


class PPOAgent:
    def __init__(
        self,
        config: ActorCriticConfig,
        ppo: PPOConfig | None = None,
        device: str = "cpu",
        seed: int = 7,
    ) -> None:
        torch.manual_seed(seed)
        self.config = config
        self.ppo = ppo or PPOConfig()
        self.device = device
        self.model = DiscreteActorCritic(config).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.ppo.lr, eps=1e-5)

    def act(self, obs: np.ndarray, deterministic: bool = False, action_mask=None):
        action, log_prob, value = self.model.act(obs, deterministic, action_mask)
        return action, log_prob, value

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        cfg = self.ppo
        adv, returns = buffer.compute_gae(
            last_value=0.0 if buffer.dones[buffer.ptr - 1] else float(buffer.values[buffer.ptr - 1]),
            gamma=cfg.gamma,
            lam=cfg.gae_lambda,
        )
        obs = torch.as_tensor(buffer.obs[: buffer.ptr], device=self.device)
        actions = torch.as_tensor(buffer.actions[: buffer.ptr], device=self.device)
        old_log_probs = torch.as_tensor(buffer.log_probs[: buffer.ptr], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_t = torch.as_tensor(returns, device=self.device)
        if cfg.normalize_advantage and adv_t.numel() > 1:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = obs.shape[0]
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}
        iters = 0
        for _ in range(cfg.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, cfg.batch_size):
                idx = perm[start : start + cfg.batch_size]
                logits, values = self.model(obs[idx])
                dist = torch.distributions.Categorical(logits=logits)
                log_probs = dist.log_prob(actions[idx])
                entropy = dist.entropy().mean()
                ratio = torch.exp(log_probs - old_log_probs[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret_t[idx])
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = float(((ratio - 1) - (log_probs - old_log_probs[idx])).mean())
                    clipfrac = float(((ratio - 1).abs() > cfg.clip_ratio).float().mean())
                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy)
                stats["approx_kl"] += approx_kl
                stats["clipfrac"] += clipfrac
                iters += 1
            if iters and stats["approx_kl"] / iters > cfg.target_kl:
                break
        for key in stats:
            stats[key] /= max(iters, 1)
        return stats

    def save(self, path) -> None:
        torch.save({"model": self.model.state_dict(), "config": self.config.__dict__}, path)

    def load(self, path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model"])


# ----------------------------------------------------------------------- SAC
@dataclass
class SACConfig:
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 200_000
    warmup_steps: int = 1000
    target_entropy: float | None = None
    reward_scale: float = 1.0


class SACAgent:
    """Off-policy maximum-entropy agent (continuous, squashed Gaussian)."""

    def __init__(
        self,
        config: ActorCriticConfig,
        sac: SACConfig | None = None,
        device: str = "cpu",
        seed: int = 7,
    ) -> None:
        torch.manual_seed(seed)
        config.continuous = True
        self.config = config
        self.cfg = sac or SACConfig()
        self.device = device
        self.actor = ContinuousActor(config).to(device)
        self.critic = TwinCritic(config).to(device)
        self.critic_target = TwinCritic(config).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic)
        self.target_entropy = (
            self.cfg.target_entropy
            if self.cfg.target_entropy is not None
            else -float(config.n_actions)
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.cfg.lr_alpha)

        self._obs = np.zeros((self.cfg.buffer_size, config.obs_dim), dtype=np.float32)
        self._act = np.zeros((self.cfg.buffer_size, config.n_actions), dtype=np.float32)
        self._rew = np.zeros((self.cfg.buffer_size, 1), dtype=np.float32)
        self._next = np.zeros((self.cfg.buffer_size, config.obs_dim), dtype=np.float32)
        self._done = np.zeros((self.cfg.buffer_size, 1), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        if deterministic:
            _, _, action = self.actor.sample(o)
        else:
            action, _, _ = self.actor.sample(o)
        return action.squeeze(0).cpu().numpy()

    def store(self, obs, action, reward, next_obs, done) -> None:
        i = self._ptr
        self._obs[i] = obs
        self._act[i] = action
        self._rew[i] = reward * self.cfg.reward_scale
        self._next[i] = next_obs
        self._done[i] = float(done)
        self._ptr = (self._ptr + 1) % self.cfg.buffer_size
        self._size = min(self._size + 1, self.cfg.buffer_size)

    def update(self) -> dict[str, float] | None:
        if self._size < max(self.cfg.batch_size, self.cfg.warmup_steps):
            return None
        idx = np.random.randint(0, self._size, size=self.cfg.batch_size)
        obs = torch.as_tensor(self._obs[idx], device=self.device)
        act = torch.as_tensor(self._act[idx], device=self.device)
        rew = torch.as_tensor(self._rew[idx], device=self.device)
        nxt = torch.as_tensor(self._next[idx], device=self.device)
        done = torch.as_tensor(self._done[idx], device=self.device)

        with torch.no_grad():
            next_action, next_log_prob, _ = self.actor.sample(nxt)
            q_next = self.critic_target.q_min(nxt, next_action) - self.alpha * next_log_prob
            target_q = rew + self.cfg.gamma * (1.0 - done) * q_next

        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        new_action, log_prob, _ = self.actor.sample(obs)
        q_new = self.critic.q_min(obs, new_action)
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.mul_(1 - self.cfg.tau).add_(p, alpha=self.cfg.tau)

        return {
            "critic_loss": float(critic_loss),
            "actor_loss": float(actor_loss),
            "alpha": float(self.alpha),
            "q_mean": float(q1.mean()),
        }

    def save(self, path) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic"])
        self.log_alpha.data = payload["log_alpha"].to(self.device)

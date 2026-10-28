"""A cost-aware FX scalping environment for reinforcement learning.

Why a custom env instead of gym-retro/backtrader
------------------------------------------------
The agent must learn *the same* frictions the live loop pays, otherwise the
policy is optimal in simulation and loss-making in production. This environment
models, at every step:

* **Spread**: half-spread is paid on entry *and* exit, taken from the bar's
  bid/ask when the data carries it (synthetic data does; historical bars fall
  back to an ATR-scaled spread).
* **Taker fee**: PrimeXBT publishes ``takerFeePct`` per instrument; passed in.
* **Slippage**: impact proportional to ``sqrt(participation)`` (a square-root law,
  the standard empirical market-impact form).
* **Financing**: charged when a position is held past ``financingInterval``.
* **Risk limit**: an intrabar stop-loss at ``sl_atr * ATR`` terminates the
  episode-fragment with the loss realised -- no survivorship from ignoring wicks.
* **Turnover penalty**: discourages the degenerate "flip every bar" policy that
  looks great gross and dies on costs.

Observation vector (``OBS_DIM``): the model features at bar ``t`` **plus** the
agent's own state -- position, unrealised PnL in risk units, bars in trade,
distance to stop, and whether a stop/target is near. Without the position block
the agent cannot learn to hold winners.

Compatibility: implements the tiny subset of the Gymnasium API the trainers use
(``reset`` -> obs, ``step`` -> obs, reward, terminated, truncated, info) without
depending on gymnasium, so the repo installs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["EnvConfig", "FXScalpEnv", "OBS_STATE_DIM"]

#: size of the agent-state block appended to every observation
OBS_STATE_DIM = 7


@dataclass
class EnvConfig:
    # episode structure (in bars)
    episode_len: int = 1440          # one trading day of 1m bars
    random_start: bool = True
    # costs
    taker_fee_pct: float = 0.05      # % per side (PrimeXBT BTC/USD example)
    default_spread_bps: float = 1.2
    slippage_coef: float = 0.1       # sqrt-impact coefficient
    financing_pct_per_8h: float = 0.0
    # risk
    sl_atr: float = 1.0              # intrabar stop distance in ATR
    tp_atr: float = 2.0              # optional target; 0 disables
    max_holding_bars: int = 60       # scalps die fast or become swings
    # reward
    risk_unit_atr: float = 1.0       # normalise PnL by this much ATR
    turnover_penalty: float = 0.02
    drawdown_penalty: float = 0.5
    reward_scale: float = 1.0
    # action space
    continuous: bool = False         # True -> SAC style target position
    max_position: float = 1.0        # continuous: [-max_position, +max_position]

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class FXScalpEnv:
    """Feature-matrix driven scalping environment.

    Parameters
    ----------
    features:
        ``(N, C)`` array of **normalised** features, one row per bar.
    prices:
        ``(N,)`` close prices aligned with ``features``.
    highs, lows:
        optional intrabar extremes for stop/target simulation.
    atr:
        ``(N,)`` average true range in price units (for stop sizing + reward norm).
    spread_bps:
        ``(N,)`` realised spread per bar (falls back to ``cfg.default_spread_bps``).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        highs: np.ndarray | None = None,
        lows: np.ndarray | None = None,
        atr: np.ndarray | None = None,
        spread_bps: np.ndarray | None = None,
        config: EnvConfig | None = None,
        seed: int = 7,
    ) -> None:
        self.cfg = config or EnvConfig()
        self.features = np.asarray(features, dtype=np.float32)
        self.prices = np.asarray(prices, dtype=np.float64)
        n = len(self.prices)
        if len(self.features) != n:
            raise ValueError("features and prices must be aligned row-wise")
        self.highs = np.asarray(highs, dtype=np.float64) if highs is not None else self.prices
        self.lows = np.asarray(lows, dtype=np.float64) if lows is not None else self.prices
        if atr is None:
            # fall back to a robust rolling range estimate in price units
            rng = np.maximum(self.highs - self.lows, 1e-9)
            self.atr = np.convolve(rng, np.ones(14) / 14, mode="same")
        else:
            self.atr = np.asarray(atr, dtype=np.float64)
        self.atr = np.maximum(self.atr, 1e-9)
        self.spread_bps = (
            np.asarray(spread_bps, dtype=np.float64)
            if spread_bps is not None
            else np.full(n, self.cfg.default_spread_bps)
        )

        self.n_features = self.features.shape[1]
        self.obs_dim = self.n_features + OBS_STATE_DIM
        self.n_actions = 3 if not self.cfg.continuous else 1
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._start = 0
        self._end = 0
        self._position = 0.0
        self._entry_price = 0.0
        self._entry_t = 0
        self._equity = 1.0
        self._peak_equity = 1.0
        self._last_financing_t = 0

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        cfg = self.cfg
        length = min(cfg.episode_len, len(self.prices) - 2)
        if cfg.random_start:
            hi = len(self.prices) - length - 1
            self._start = int(self.rng.integers(0, max(hi, 1)))
        else:
            self._start = 0
        self._end = min(self._start + length, len(self.prices) - 1)
        self._t = self._start
        self._position = 0.0
        self._entry_price = 0.0
        self._entry_t = self._t
        self._equity = 1.0
        self._peak_equity = 1.0
        self._last_financing_t = self._t
        return self._obs(), {"start": self._start}

    def step(
        self, action: Any
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.cfg
        target = self._decode_action(action)
        price = self.prices[self._t]
        atr = self.atr[self._t]
        reward = 0.0
        info: dict[str, Any] = {"t": self._t, "position": self._position, "equity": self._equity}

        # ---------------------------------------------------------- rebalance
        delta = target - self._position
        if abs(delta) > 1e-9:
            notional = abs(delta)
            cost = self._trade_cost(notional, atr, price)
            reward -= cost
            self._equity -= cost
            if abs(self._position) < 1e-9 and abs(target) > 1e-9:
                self._entry_price = price * (1 + np.sign(target) * self._half_spread_bps() / 1e4)
                self._entry_t = self._t
            elif abs(target) < 1e-9:
                self._entry_price = 0.0
            elif np.sign(target) != np.sign(self._position):
                self._entry_price = price * (1 + np.sign(target) * self._half_spread_bps() / 1e4)
                self._entry_t = self._t
            self._position = float(np.clip(target, -cfg.max_position, cfg.max_position))
            reward -= cfg.turnover_penalty * notional
            info["traded"] = notional

        # ------------------------------------------------------ mark to market
        nxt = min(self._t + 1, len(self.prices) - 1)
        next_price = self.prices[nxt]
        pnl = self._position * (next_price - price) / max(self.atr[self._t], 1e-9) * cfg.risk_unit_atr
        reward += pnl
        self._equity += pnl
        self._peak_equity = max(self._peak_equity, self._equity)

        # ---------------------------------------------------------- financing
        if abs(self._position) > 1e-9 and cfg.financing_pct_per_8h:
            bars_per_interval = 480
            if self._t - self._last_financing_t >= bars_per_interval:
                charge = abs(self._position) * cfg.financing_pct_per_8h / 100.0
                reward -= charge
                self._equity -= charge
                self._last_financing_t = self._t

        # ------------------------------------------------- stop / target / time
        terminated = False
        hit_stop = False
        if abs(self._position) > 1e-9:
            if cfg.sl_atr > 0 and np.sign(self._position) > 0 and self.lows[nxt] <= price - cfg.sl_atr * atr:
                hit_stop = True
            elif cfg.sl_atr > 0 and np.sign(self._position) < 0 and self.highs[nxt] >= price + cfg.sl_atr * atr:
                hit_stop = True
            if cfg.tp_atr > 0 and not hit_stop:
                if np.sign(self._position) > 0 and self.highs[nxt] >= price + cfg.tp_atr * atr:
                    info["target_hit"] = True
                elif np.sign(self._position) < 0 and self.lows[nxt] <= price - cfg.tp_atr * atr:
                    info["target_hit"] = True
            if hit_stop:
                # realise the stop loss (plus exit cost) and flatten
                loss = cfg.sl_atr * cfg.risk_unit_atr
                exit_cost = self._trade_cost(abs(self._position), atr, next_price)
                reward -= loss + exit_cost
                self._equity -= loss + exit_cost
                self._position = 0.0
                self._entry_price = 0.0
                terminated = True
                info["stopped"] = True
            elif self._t - self._entry_t >= cfg.max_holding_bars:
                exit_cost = self._trade_cost(abs(self._position), atr, next_price)
                reward -= exit_cost
                self._equity -= exit_cost
                self._position = 0.0
                self._entry_price = 0.0
                info["timed_out"] = True

        # ---------------------------------------------------- drawdown penalty
        drawdown = (self._peak_equity - self._equity) / max(abs(self._peak_equity), 1e-9)
        reward -= cfg.drawdown_penalty * max(drawdown, 0.0)

        self._t = nxt
        truncated = self._t >= self._end
        info["equity"] = self._equity
        info["drawdown"] = float(drawdown)
        return self._obs(), float(reward * cfg.reward_scale), terminated, truncated, info

    # ------------------------------------------------------------------ helpers
    def _decode_action(self, action: Any) -> float:
        if self.cfg.continuous:
            a = float(np.clip(np.asarray(action).reshape(-1)[0], -1.0, 1.0))
            return a * self.cfg.max_position
        a = int(np.asarray(action).reshape(-1)[0])
        return {0: 0.0, 1: 1.0, 2: -1.0}.get(a, 0.0)

    def _half_spread_bps(self) -> float:
        return float(self.spread_bps[self._t]) / 2.0

    def _trade_cost(self, notional: float, atr: float, price: float) -> float:
        """Cost of trading ``notional`` (in position units) at ``price``, in equity units."""
        spread = self.spread_bps[self._t] / 1e4
        fee = 2.0 * self.cfg.taker_fee_pct / 100.0  # entry + exit
        impact = self.cfg.slippage_coef * np.sqrt(max(notional, 0.0)) * (self.spread_bps[self._t] / 1e4)
        return float(notional * (spread + fee + impact))

    def _obs(self) -> np.ndarray:
        t = self._t
        atr = max(self.atr[t], 1e-9)
        price = self.prices[t]
        if abs(self._position) > 1e-9:
            unreal = self._position * (price - self._entry_price) / atr
            dist_stop = (price - (self._entry_price - np.sign(self._position) * self.cfg.sl_atr * atr)) / atr
            bar_in_trade = float(min(self._t - self._entry_t, self.cfg.max_holding_bars)) / max(
                self.cfg.max_holding_bars, 1
            )
        else:
            unreal = 0.0
            dist_stop = 0.0
            bar_in_trade = 0.0
        equity_state = float(np.tanh(self._equity - 1.0))
        dd = float(np.clip((self._peak_equity - self._equity) / max(abs(self._peak_equity), 1e-9), 0, 1))
        state = np.array(
            [self._position, unreal, bar_in_trade, np.tanh(dist_stop), equity_state, dd, 1.0 if self._position else 0.0],
            dtype=np.float32,
        )
        return np.concatenate([self.features[t], state])

    @property
    def observation_dim(self) -> int:
        return self.obs_dim

    def set_position_mode_continuous(self, flag: bool) -> None:
        self.cfg.continuous = flag
        self.n_actions = 1 if flag else 3

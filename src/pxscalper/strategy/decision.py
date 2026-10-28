"""Decision layer: turns model outputs into a single, auditable intent.

The layout is *meta-labelling* (López de Prado, 2018), which is how a scalping
desk actually stacks models:

* the **primary model** (TCN / LSTM / Transformer) proposes a *direction*;
* the **secondary model** (PPO/SAC DRL agent) decides whether to **act on it**
  and **how much** -- it sees the primary signal plus the microstructure and the
  current inventory, so it learns "this looks like a long, but flow says the
  spread is about to widen, stand down".

Both are optional: with one model the bot degrades gracefully to pure
direction-following, which keeps the repo runnable end-to-end after a single
training job.

Nothing here touches the exchange or the risk limits -- :mod:`pxscalper.risk`
has the final say. Keeping the three layers separate is what makes the system
auditable: for every order you can print *direction -> veto -> size -> verdict*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

__all__ = ["Intent", "Decision", "StrategyConfig", "ScalpingStrategy"]


class Intent(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

    @property
    def direction(self) -> int:
        return {"LONG": 1, "SHORT": -1, "FLAT": 0}[self.value]


@dataclass
class Decision:
    intent: Intent
    size_multiplier: float = 1.0     # 0..1, scales the risk engine's size
    confidence: float = 0.0
    edge: float = 0.0
    reason: str = ""
    primary_prob: float = 0.0
    rl_action: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def is_trade(self) -> bool:
        return self.intent is not Intent.FLAT and self.size_multiplier > 0

    def to_dict(self) -> dict:
        return {
            "intent": self.intent.value,
            "size_multiplier": self.size_multiplier,
            "confidence": self.confidence,
            "edge": self.edge,
            "reason": self.reason,
            "primary_prob": self.primary_prob,
            "rl_action": self.rl_action,
            **self.meta,
        }


@dataclass
class StrategyConfig:
    min_confidence: float = 0.42
    min_edge: float = 0.03
    require_rl_agreement: bool = True
    rl_neutral_band: float = 0.15     # |rl action| below this = RL says "no trade"
    size_by_edge: bool = True
    max_size_multiplier: float = 1.0
    uncertainty_penalty: float = 0.5  # shrink size when models disagree
    allow_reversal: bool = True
    min_seconds_between_trades: float = 5.0
    cooldown_after_loss_s: float = 30.0
    max_hold_bars: int = 45


class ScalpingStrategy:
    """Combines the supervised predictor with the DRL agent."""

    def __init__(
        self,
        predictor=None,
        rl_policy=None,
        config: StrategyConfig | None = None,
    ) -> None:
        self.predictor = predictor
        self.rl_policy = rl_policy
        self.config = config or StrategyConfig()

    # ------------------------------------------------------------------ decide
    def decide(
        self,
        window: np.ndarray,
        rl_observation: np.ndarray | None = None,
        position: float = 0.0,
        now_s: float | None = None,
        last_trade_s: float = 0.0,
        last_loss_s: float = 0.0,
    ) -> Decision:
        cfg = self.config
        meta: dict[str, Any] = {}

        # 1. primary direction ------------------------------------------------
        if self.predictor is None:
            return Decision(Intent.FLAT, reason="no primary model loaded")

        pred = self.predictor.predict(window)
        meta["latency_us"] = round(pred.latency_us, 1)
        meta["probs"] = np.round(pred.probs, 4).tolist()

        if pred.direction == 0 or pred.confidence < cfg.min_confidence:
            return Decision(
                Intent.FLAT,
                confidence=pred.confidence,
                edge=pred.edge,
                primary_prob=pred.confidence,
                reason=f"primary flat (conf {pred.confidence:.3f} < {cfg.min_confidence})",
                meta=meta,
            )
        if abs(pred.edge) < cfg.min_edge:
            return Decision(
                Intent.FLAT,
                confidence=pred.confidence,
                edge=pred.edge,
                primary_prob=pred.confidence,
                reason=f"edge {pred.edge:.3f} below {cfg.min_edge}",
                meta=meta,
            )

        intent = Intent.LONG if pred.direction > 0 else Intent.SHORT

        # 2. rate limiting / loss cooldown -------------------------------------
        if now_s is not None:
            if cfg.cooldown_after_loss_s and last_loss_s and now_s - last_loss_s < cfg.cooldown_after_loss_s:
                return Decision(
                    Intent.FLAT,
                    confidence=pred.confidence,
                    edge=pred.edge,
                    reason="cooldown after loss",
                    meta=meta,
                )
            if cfg.min_seconds_between_trades and last_trade_s and now_s - last_trade_s < cfg.min_seconds_between_trades:
                return Decision(
                    Intent.FLAT,
                    confidence=pred.confidence,
                    edge=pred.edge,
                    reason="rate limited",
                    meta=meta,
                )

        # 3. secondary model: act / size --------------------------------------
        size_mult = 1.0
        rl_action: float | None = None
        if self.rl_policy is not None and rl_observation is not None:
            rl_action = float(self.rl_policy.act(rl_observation))
            meta["rl_action"] = round(rl_action, 4)
            agrees = np.sign(rl_action) == np.sign(pred.direction)
            if not agrees or abs(rl_action) < cfg.rl_neutral_band:
                if cfg.require_rl_agreement:
                    return Decision(
                        Intent.FLAT,
                        confidence=pred.confidence,
                        edge=pred.edge,
                        rl_action=rl_action,
                        primary_prob=pred.confidence,
                        reason=f"RL veto (action {rl_action:+.2f})",
                        meta=meta,
                    )
                size_mult *= cfg.uncertainty_penalty
                meta["rl_veto_soft"] = True
            else:
                size_mult *= min(abs(rl_action), 1.0)

        # 4. edge-proportional sizing ----------------------------------------
        if cfg.size_by_edge:
            edge_scale = min(abs(pred.edge) / 0.25, 1.0)   # full size at |edge| >= 0.25
            size_mult *= max(edge_scale, 0.25)
        size_mult = float(np.clip(size_mult, 0.0, cfg.max_size_multiplier))
        if size_mult <= 0:
            return Decision(Intent.FLAT, reason="size multiplier zero", meta=meta)

        # 5. reversal handling --------------------------------------------------
        if position and np.sign(position) != np.sign(pred.direction) and not cfg.allow_reversal:
            return Decision(
                Intent.FLAT, reason="reversal disabled", meta=meta,
                confidence=pred.confidence, edge=pred.edge,
            )

        return Decision(
            intent=intent,
            size_multiplier=size_mult,
            confidence=pred.confidence,
            edge=pred.edge,
            reason="signal",
            primary_prob=pred.confidence,
            rl_action=rl_action,
            meta=meta,
        )

    # --------------------------------------------------------------- lifecycle
    def attach_rl(self, rl_policy) -> None:
        self.rl_policy = rl_policy

    def describe(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "has_primary": self.predictor is not None,
            "has_rl": self.rl_policy is not None,
            "predictor_backend": getattr(self.predictor, "config", None).backend
            if self.predictor is not None
            else None,
        }

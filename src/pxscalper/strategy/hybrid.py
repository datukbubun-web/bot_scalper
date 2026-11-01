"""Hybrid Strategy Coordinator: Intraday Bias (Layer 1) + Scalper Execution (Layer 2).

Architecture
------------
Pure scalpers fail when they fade institutional session trends or trade inside
chaotic volatility shocks; pure higher-timeframe systems suffer wide stops and
coarse entry timing. :class:`HybridStrategy` unifies both layers:

1. **Intraday Bias Layer** (:class:`IntradayBiasEngine`):
   - Evaluates closed higher-timeframe (15M/1H) cross-metal fair-value equilibrium,
     anchored Gold Session VWAP institutional flow (22:00 UTC reset), 1H/24H
     volatility regime, and 2H Kaufman Efficiency Ratio.
   - Emits an auditable :class:`BiasSnapshot` that gates direction and classifies
     the market into ``TREND_ACCUMULATION``, ``TREND_DISTRIBUTION``,
     ``BALANCED_MEAN_REVERSION``, ``VOLATILITY_SHOCK``, or ``ROLL_BLACKOUT``.

2. **Scalper Execution Layer** (:class:`ScalpingStrategy` + Micro-Timing):
   - Uses the sub-millisecond ONNX/TorchScript sequence model (and optional DRL
     meta-policy) on 5M or 1M bars for precise entry timing.
   - Optionally enforces 1M candle wick-absorption confirmation
     (``require_micro_absorption=True``) so entries fire only once adverse
     intrabar order flow stalls.
   - Adapts position sizing and target/holding multipliers according to the
     degree of confluence with Intraday Bias.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .decision import Decision, Intent, ScalpingStrategy, StrategyConfig
from .intraday_bias import BiasSnapshot, IntradayBiasConfig, IntradayBiasEngine

__all__ = ["HybridConfig", "HybridStrategy"]


@dataclass
class HybridConfig:
    """Configuration for the Hybrid (Intraday Bias + Scalper Execution) coordinator."""

    enabled: bool = True
    bias: IntradayBiasConfig = field(default_factory=IntradayBiasConfig)
    require_composite_threshold: bool = True
    composite_threshold: float = 1.85
    require_micro_absorption: bool = False
    min_wick_absorption: float = 0.05
    wick_feature_col: str = "close_location"
    modulate_size_by_bias: bool = False
    aligned_size_boost: float = 1.15
    cautious_size_cut: float = 0.85
    aligned_tp_mult: float = 1.0
    cautious_tp_mult: float = 1.0
    max_hold_aligned_mult: float = 1.0

    @classmethod
    def from_dict(cls, raw: dict | None) -> "HybridConfig":
        if not raw:
            return cls()
        data = dict(raw)
        bias_raw = data.pop("bias", None)
        bias_cfg = IntradayBiasConfig.from_dict(bias_raw if isinstance(bias_raw, dict) else data)
        valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(bias=bias_cfg, **valid)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["bias"] = self.bias.to_dict()
        return d


class HybridStrategy(ScalpingStrategy):
    """Combines causal Intraday Bias gating/modulation with Scalper Execution."""

    def __init__(
        self,
        predictor=None,
        rl_policy=None,
        config: StrategyConfig | None = None,
        hybrid_config: HybridConfig | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        super().__init__(predictor=predictor, rl_policy=rl_policy, config=config)
        self.hybrid_config = hybrid_config or HybridConfig()
        self.feature_names: tuple[str, ...] = tuple(feature_names or ())
        self.bias_engine = IntradayBiasEngine(
            config=self.hybrid_config.bias,
            feature_names=self.feature_names,
        )
        self._col_map = {n: i for i, n in enumerate(self.feature_names)}

    def bind_features(self, feature_names: Sequence[str]) -> None:
        self.feature_names = tuple(feature_names)
        self._col_map = {n: i for i, n in enumerate(self.feature_names)}
        self.bias_engine.bind_features(self.feature_names)

    def _extract_feature(self, window: np.ndarray, col_name: str, default: float = 0.0) -> float:
        idx = self._col_map.get(col_name)
        if idx is None:
            return default
        arr = np.asarray(window, dtype=np.float64)
        x_last = arr[0, :, -1] if arr.ndim == 3 else (arr[:, -1] if arr.ndim == 2 else arr.reshape(-1))
        if idx < 0 or idx >= len(x_last):
            return default
        val = float(x_last[idx])
        return val if np.isfinite(val) else default

    def decide(
        self,
        window: np.ndarray,
        rl_observation: np.ndarray | None = None,
        position: float = 0.0,
        now_s: float | None = None,
        last_trade_s: float = 0.0,
        last_loss_s: float = 0.0,
        bias_snapshot: BiasSnapshot | None = None,
    ) -> Decision:
        base_dec = super().decide(
            window=window,
            rl_observation=rl_observation,
            position=position,
            now_s=now_s,
            last_trade_s=last_trade_s,
            last_loss_s=last_loss_s,
        )
        hcfg = self.hybrid_config
        if not hcfg.enabled:
            return base_dec

        snap = bias_snapshot or self.bias_engine.evaluate_window(window, self.feature_names)
        meta = dict(base_dec.meta)
        meta["architecture"] = "hybrid_intraday_scalper"
        meta["intraday_bias"] = snap.to_dict()

        if not base_dec.is_trade:
            base_dec.meta = meta
            return base_dec

        trade_dir = base_dec.intent.direction

        # 1. Intraday Bias Gate (Roll Blackout, Vol Regime, HTF Fair-Value, Session VWAP)
        allowed, align_score, align_reason = snap.evaluate_trade_alignment(trade_dir, hcfg.bias)
        meta["bias_alignment"] = round(float(align_score), 4)
        if not allowed:
            return Decision(
                Intent.FLAT,
                size_multiplier=0.0,
                confidence=base_dec.confidence,
                edge=base_dec.edge,
                primary_prob=base_dec.primary_prob,
                rl_action=base_dec.rl_action,
                reason=f"hybrid bias veto: {align_reason}",
                meta=meta,
            )

        # 2. Composite MTF Conviction check (when MTF channels are bound)
        has_mtf_cols = (
            hcfg.bias.htf_rv_col in self._col_map or "htf_15m_xag_rv_z_2" in self._col_map
        )
        if hcfg.require_composite_threshold and has_mtf_cols:
            if trade_dir * snap.bias_score < hcfg.composite_threshold:
                return Decision(
                    Intent.FLAT,
                    size_multiplier=0.0,
                    confidence=base_dec.confidence,
                    edge=base_dec.edge,
                    primary_prob=base_dec.primary_prob,
                    rl_action=base_dec.rl_action,
                    reason=(
                        f"hybrid composite score {trade_dir * snap.bias_score:.2f} "
                        f"< {hcfg.composite_threshold:.2f}"
                    ),
                    meta=meta,
                )

        # 3. Micro-Scalper Candle Absorption Confirmation (1M precision entry timing)
        if hcfg.require_micro_absorption and hcfg.wick_feature_col in self._col_map:
            wick_val = self._extract_feature(window, hcfg.wick_feature_col, 0.0)
            meta["wick_absorption"] = round(wick_val, 4)
            if trade_dir * wick_val < hcfg.min_wick_absorption:
                return Decision(
                    Intent.FLAT,
                    size_multiplier=0.0,
                    confidence=base_dec.confidence,
                    edge=base_dec.edge,
                    primary_prob=base_dec.primary_prob,
                    rl_action=base_dec.rl_action,
                    reason=(
                        f"scalper micro-timing wait: {hcfg.wick_feature_col}="
                        f"{wick_val:+.2f} below {hcfg.min_wick_absorption:+.2f}"
                    ),
                    meta=meta,
                )

        # 4. Bias-Adaptive Execution Modulation (Size / TP / Hold tier)
        size_mult = float(base_dec.size_multiplier)
        tp_mult = 1.0
        hold_mult = 1.0
        if align_score >= 0.25:
            exec_tier = "CONFLUENCE"
            if hcfg.modulate_size_by_bias:
                size_mult = min(self.config.max_size_multiplier, size_mult * hcfg.aligned_size_boost)
            tp_mult = float(hcfg.aligned_tp_mult)
            hold_mult = float(hcfg.max_hold_aligned_mult)
        elif align_score <= -0.20:
            exec_tier = "CAUTIOUS_SCALP"
            if hcfg.modulate_size_by_bias:
                size_mult = max(0.1, size_mult * hcfg.cautious_size_cut)
            tp_mult = float(hcfg.cautious_tp_mult)
        else:
            exec_tier = "STANDARD_HYBRID"

        meta["execution_tier"] = exec_tier
        meta["tp_atr_mult"] = round(tp_mult, 4)
        meta["max_hold_mult"] = round(hold_mult, 4)

        return Decision(
            intent=base_dec.intent,
            size_multiplier=float(size_mult),
            confidence=base_dec.confidence,
            edge=base_dec.edge,
            primary_prob=base_dec.primary_prob,
            rl_action=base_dec.rl_action,
            reason=f"{base_dec.reason} | {exec_tier} ({align_reason})",
            meta=meta,
        )

"""Layer 1 of the Hybrid Architecture: Causal Intraday Bias Engine.

Separates **Intraday Macro / Fair-Value Bias** (15M–4H horizon, anchored session
VWAP institutional flow, and intraday volatility regime) from **Scalper
Execution** (1M/5M short-horizon entry timing, microstructure absorption, and
tight stop/target management).

Every quantity in :class:`BiasSnapshot` is strictly causal (``T_htf + htf_ms <=
t_ltf + ltf_ms``) and auditable in JSONL decision logs and CLI telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..features.mtf import MTFBiasConfig, compute_mtf_bias_features

__all__ = [
    "BiasDirection",
    "IntradayRegime",
    "BiasSnapshot",
    "IntradayBiasConfig",
    "IntradayBiasEngine",
]


class BiasDirection(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"

    @property
    def sign(self) -> int:
        return {"BULLISH": 1, "BEARISH": -1, "NEUTRAL": 0}[self.value]


class IntradayRegime(str, Enum):
    TREND_ACCUMULATION = "TREND_ACCUMULATION"
    TREND_DISTRIBUTION = "TREND_DISTRIBUTION"
    BALANCED_MEAN_REVERSION = "BALANCED_MEAN_REVERSION"
    VOLATILITY_SHOCK = "VOLATILITY_SHOCK"
    ROLL_BLACKOUT = "ROLL_BLACKOUT"


@dataclass
class IntradayBiasConfig:
    """Thresholds & weights governing the Intraday Bias state machine."""

    htf_bar_size: str = "15M"
    htf_rv_col: str = "htf_15m_xag_rv_z_2"
    ltf_rv_col: str = "ctx_xag_rv_z_4"
    ltf_weight: float = 0.65
    htf_weight: float = 0.35
    active_bias_threshold: float = 1.0
    strong_bias_threshold: float = 1.85
    trend_regime_threshold: float = 0.45
    max_opposing_session_trend: float = 0.60
    max_vol_regime_ratio: float = 1.85
    min_vol_regime_ratio: float = 0.15
    require_htf_non_opposing: bool = True

    @classmethod
    def from_dict(cls, raw: dict | None) -> "IntradayBiasConfig":
        if not raw:
            return cls()
        valid = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class BiasSnapshot:
    """Point-in-time causal readout of Intraday Bias & Market Regime."""

    direction: BiasDirection
    regime: IntradayRegime
    bias_score: float = 0.0         # composite fair-value score (+ = bullish, - = bearish)
    htf_fv_score: float = 0.0       # -htf_rv_z (+ = undervalued on HTF)
    ltf_fv_score: float = 0.0       # -ltf_rv_z (+ = undervalued on LTF)
    session_vwap_z: float = 0.0
    session_trend: float = 0.0      # [-1, +1] institutional session VWAP flow
    vol_regime_ratio: float = 1.0   # 1.0 = normal 24h median volatility
    efficiency_ratio: float = 0.25  # [0, 1] 2H Kaufman ER
    roll_blackout: bool = False

    def evaluate_trade_alignment(
        self, trade_dir: int, cfg: IntradayBiasConfig
    ) -> tuple[bool, float, str]:
        """Check whether ``trade_dir`` in ``{-1, +1}`` is allowed by Intraday Bias.

        Returns ``(allowed, alignment_score, reason)`` where ``alignment_score``
        measures directional confluence between the trade and session/HTF structure.
        """
        if trade_dir == 0:
            return False, 0.0, "flat intent"
        if self.roll_blackout or self.regime is IntradayRegime.ROLL_BLACKOUT:
            return False, 0.0, "intraday roll blackout (21:00-23:00 UTC)"
        if (
            self.regime is IntradayRegime.VOLATILITY_SHOCK
            or self.vol_regime_ratio > cfg.max_vol_regime_ratio
            or self.vol_regime_ratio < cfg.min_vol_regime_ratio
        ):
            return (
                False,
                0.0,
                f"intraday vol regime {self.vol_regime_ratio:.2f}x outside "
                f"[{cfg.min_vol_regime_ratio:.2f}, {cfg.max_vol_regime_ratio:.2f}]",
            )
        if cfg.require_htf_non_opposing and (trade_dir * self.htf_fv_score < -1e-6):
            return (
                False,
                float(trade_dir * self.htf_fv_score),
                f"opposes {cfg.htf_bar_size} fair-value bias ({self.htf_fv_score:+.2f})",
            )
        sess_align = float(trade_dir * self.session_trend)
        if sess_align < -cfg.max_opposing_session_trend:
            return (
                False,
                sess_align,
                f"opposes runaway session VWAP trend ({self.session_trend:+.2f})",
            )
        alignment = 0.50 * np.tanh(trade_dir * self.htf_fv_score / 2.0) + 0.50 * sess_align
        return True, float(alignment), f"aligned ({self.regime.value}, bias={self.bias_score:+.2f})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "regime": self.regime.value,
            "bias_score": round(float(self.bias_score), 4),
            "htf_fv_score": round(float(self.htf_fv_score), 4),
            "ltf_fv_score": round(float(self.ltf_fv_score), 4),
            "session_vwap_z": round(float(self.session_vwap_z), 4),
            "session_trend": round(float(self.session_trend), 4),
            "vol_regime_ratio": round(float(self.vol_regime_ratio), 4),
            "efficiency_ratio": round(float(self.efficiency_ratio), 4),
            "roll_blackout": bool(self.roll_blackout),
        }


class IntradayBiasEngine:
    """Evaluates causal Intraday Bias snapshots from feature windows or OHLCV frames."""

    def __init__(
        self,
        config: IntradayBiasConfig | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self.config = config or IntradayBiasConfig()
        self._col_map: dict[str, int] = {}
        if feature_names:
            self.bind_features(feature_names)

    def bind_features(self, feature_names: Sequence[str]) -> None:
        self._col_map = {str(name): idx for idx, name in enumerate(feature_names)}

    def _get_feat(self, x_last: np.ndarray, name: str, default: float = 0.0) -> float:
        idx = self._col_map.get(name)
        if idx is None or idx < 0 or idx >= len(x_last):
            return default
        val = float(x_last[idx])
        return val if np.isfinite(val) else default

    def evaluate_window(
        self, window: np.ndarray, feature_names: Sequence[str] | None = None
    ) -> BiasSnapshot:
        """Extract causal :class:`BiasSnapshot` from a ``(C, L)`` or ``(1, C, L)`` window."""
        if feature_names is not None and not self._col_map:
            self.bind_features(feature_names)
        arr = np.asarray(window, dtype=np.float64)
        if arr.ndim == 3:
            x_last = arr[0, :, -1]
        elif arr.ndim == 2:
            x_last = arr[:, -1]
        else:
            x_last = arr.reshape(-1)

        cfg = self.config
        # Resolve HTF & LTF relative-value channels gracefully across 5M and 1M presets
        htf_rv = self._get_feat(x_last, cfg.htf_rv_col, default=np.nan)
        if not np.isfinite(htf_rv):
            for fallback in ("htf_15m_xag_rv_z_2", "ctx_xag_rv_z_12", "ctx_xag_rv_z_2"):
                if fallback in self._col_map:
                    htf_rv = self._get_feat(x_last, fallback, 0.0)
                    break
            else:
                htf_rv = 0.0

        ltf_rv = self._get_feat(x_last, cfg.ltf_rv_col, default=np.nan)
        if not np.isfinite(ltf_rv):
            for fallback in ("ctx_xag_rv_z_4", "htf_5m_xag_rv_z_4", "ctx_xag_rv_z_2"):
                if fallback in self._col_map:
                    ltf_rv = self._get_feat(x_last, fallback, 0.0)
                    break
            else:
                ltf_rv = 0.0

        vwap_z = self._get_feat(x_last, "intraday_session_vwap_z", 0.0)
        sess_trend = self._get_feat(x_last, "intraday_session_trend", 0.0)
        vol_excess = self._get_feat(x_last, "intraday_vol_regime", 0.0)
        vol_ratio = float(max(0.05, 1.0 + vol_excess))
        er_2h = self._get_feat(x_last, "intraday_er_2h", 0.25)
        roll = self._get_feat(x_last, "gold_roll_blackout", 0.0) > 0.5

        return self._build_snapshot(
            htf_rv=htf_rv,
            ltf_rv=ltf_rv,
            vwap_z=vwap_z,
            sess_trend=sess_trend,
            vol_ratio=vol_ratio,
            er_2h=er_2h,
            roll=roll,
        )

    def evaluate_bars(
        self,
        ltf_bars: pd.DataFrame,
        ltf_bar_size: str = "5M",
        context: dict[str, pd.DataFrame] | None = None,
        htf_bars: dict[str, pd.DataFrame] | None = None,
        htf_context: dict[str, dict[str, pd.DataFrame]] | None = None,
        context_symbols: dict[str, str] | None = None,
    ) -> BiasSnapshot:
        """Compute causal :class:`BiasSnapshot` directly from OHLCV frames."""
        if ltf_bars is None or ltf_bars.empty:
            return BiasSnapshot(BiasDirection.NEUTRAL, IntradayRegime.BALANCED_MEAN_REVERSION)

        syms = context_symbols or ({"XAG/USD": "xag"} if context and "XAG/USD" in context else {})
        bs_up = ltf_bar_size.upper()
        htf_tfs = ("5M", "15M") if bs_up == "1M" else (self.config.htf_bar_size.upper(),)
        mtf_cfg = MTFBiasConfig(htf_bar_sizes=htf_tfs)
        mtf_df = compute_mtf_bias_features(
            ltf_bars=ltf_bars,
            ltf_bar_size=bs_up,
            cfg=mtf_cfg,
            context_symbols=syms,
            ltf_context=context,
            htf_bars_by_tf=htf_bars,
            htf_context_by_tf=htf_context,
        )
        last_row = mtf_df.iloc[-1]
        rv_col = self.config.htf_rv_col
        if rv_col not in last_row:
            for candidate in ("htf_15m_xag_rv_z_2", "htf_5m_xag_rv_z_2"):
                if candidate in last_row:
                    rv_col = candidate
                    break
        htf_rv = float(last_row.get(rv_col, 0.0))

        ltf_rv = 0.0
        if bs_up == "1M" and "htf_5m_xag_rv_z_4" in last_row:
            ltf_rv = float(last_row["htf_5m_xag_rv_z_4"])
        elif context and syms:
            from ..features.context import ContextConfig, align_context, context_features

            c_cfg = ContextConfig(
                symbols=syms,
                zscore_window=288 if bs_up == "5M" else 96,
                min_coverage=0.25,
                include_rv_features=True,
                rv_windows=(1, 4, 12) if bs_up == "5M" else (1, 2, 4),
                rv_norm_window=144 if bs_up == "5M" else 48,
            )
            c_df = context_features(align_context(ltf_bars, context), c_cfg)
            target_col = "ctx_xag_rv_z_4" if "ctx_xag_rv_z_4" in c_df.columns else "ctx_xag_rv_z_2"
            if target_col in c_df.columns:
                ltf_rv = float(c_df[target_col].iloc[-1])

        ts_last = pd.to_datetime(int(ltf_bars["time"].iloc[-1]), unit="ms", utc=True)
        roll = bool(21 <= ts_last.hour < 23)
        return self._build_snapshot(
            htf_rv=htf_rv,
            ltf_rv=ltf_rv,
            vwap_z=float(last_row["intraday_session_vwap_z"]),
            sess_trend=float(last_row["intraday_session_trend"]),
            vol_ratio=float(max(0.05, 1.0 + float(last_row["intraday_vol_regime"]))),
            er_2h=float(last_row["intraday_er_2h"]),
            roll=roll,
        )

    def _build_snapshot(
        self,
        htf_rv: float,
        ltf_rv: float,
        vwap_z: float,
        sess_trend: float,
        vol_ratio: float,
        er_2h: float,
        roll: bool,
    ) -> BiasSnapshot:
        cfg = self.config
        htf_fv = -float(htf_rv)
        ltf_fv = -float(ltf_rv)
        bias_score = cfg.ltf_weight * ltf_fv + cfg.htf_weight * htf_fv

        if roll:
            regime = IntradayRegime.ROLL_BLACKOUT
        elif vol_ratio > cfg.max_vol_regime_ratio or vol_ratio < cfg.min_vol_regime_ratio:
            regime = IntradayRegime.VOLATILITY_SHOCK
        elif sess_trend >= cfg.trend_regime_threshold:
            regime = IntradayRegime.TREND_ACCUMULATION
        elif sess_trend <= -cfg.trend_regime_threshold:
            regime = IntradayRegime.TREND_DISTRIBUTION
        else:
            regime = IntradayRegime.BALANCED_MEAN_REVERSION

        if roll or regime is IntradayRegime.VOLATILITY_SHOCK:
            direction = BiasDirection.NEUTRAL
        elif (bias_score >= cfg.active_bias_threshold or htf_fv >= cfg.active_bias_threshold) and htf_fv >= -1e-6 and sess_trend >= -cfg.max_opposing_session_trend:
            direction = BiasDirection.BULLISH
        elif (bias_score <= -cfg.active_bias_threshold or htf_fv <= -cfg.active_bias_threshold) and htf_fv <= 1e-6 and sess_trend <= cfg.max_opposing_session_trend:
            direction = BiasDirection.BEARISH
        elif sess_trend >= cfg.trend_regime_threshold and htf_fv >= -1e-6:
            direction = BiasDirection.BULLISH
        elif sess_trend <= -cfg.trend_regime_threshold and htf_fv <= 1e-6:
            direction = BiasDirection.BEARISH
        else:
            direction = BiasDirection.NEUTRAL

        return BiasSnapshot(
            direction=direction,
            regime=regime,
            bias_score=float(bias_score),
            htf_fv_score=float(htf_fv),
            ltf_fv_score=float(ltf_fv),
            session_vwap_z=float(vwap_z),
            session_trend=float(sess_trend),
            vol_regime_ratio=float(vol_ratio),
            efficiency_ratio=float(er_2h),
            roll_blackout=bool(roll),
        )

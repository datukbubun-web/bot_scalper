"""Causal multi-timeframe (MTF) alignment & intraday session structure features.

Zero Look-Ahead Invariant
-------------------------
Let an execution (LTF) bar open at ``t`` with duration ``ltf_ms`` so its close
is ``t_close = t + ltf_ms``. Let a higher-timeframe (HTF) bar open at ``T`` with
duration ``htf_ms`` so its close is ``T_close = T + htf_ms``.

An HTF bar ``T`` is observable at LTF decision time ``t_close`` **if and only if**:

    T + htf_ms <= t + ltf_ms

Aligning on open timestamps (``T <= t``) is a classic multi-timeframe look-ahead
bug because at ``t = 10:00`` (close ``10:05``) the ``15M`` bar opening at
``T = 10:00`` will not close until ``10:15`` -- joining it onto ``10:00`` leaks
10 minutes of future prices (`10:05..10:15`). Every function in this module
enforces ``T_close <= t_close`` strictly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..utils import bar_ms
from .context import ContextConfig, align_context, context_features

__all__ = [
    "MTFBiasConfig",
    "resample_ohlcv_causal",
    "align_htf_to_ltf",
    "intraday_session_features",
    "mtf_feature_names",
    "compute_mtf_bias_features",
]


@dataclass
class MTFBiasConfig:
    """Configuration for causal HTF Intraday Bias + Session features."""

    htf_bar_sizes: tuple[str, ...] = ("15M",)
    context_symbols: dict[str, str] = field(default_factory=dict)
    rv_windows_by_tf: dict[str, tuple[int, ...]] = field(
        default_factory=lambda: {
            "5M": (1, 4, 12),
            "15M": (1, 2, 4),
            "1H": (1, 2, 4),
        }
    )
    rv_norm_by_tf: dict[str, int] = field(
        default_factory=lambda: {"5M": 144, "15M": 48, "1H": 24}
    )
    zscore_by_tf: dict[str, int] = field(
        default_factory=lambda: {"5M": 288, "15M": 96, "1H": 48}
    )
    min_coverage: float = 0.25
    session_open_hour_utc: int = 22
    min_session_minutes: int = 60
    vol_fast_minutes: int = 60
    vol_baseline_minutes: int = 1440
    er_window_minutes: int = 120

    def __post_init__(self) -> None:
        self.htf_bar_sizes = tuple(str(s).upper() for s in self.htf_bar_sizes)
        self.rv_windows_by_tf = {
            str(k).upper(): tuple(int(x) for x in v)
            for k, v in self.rv_windows_by_tf.items()
        }
        self.rv_norm_by_tf = {str(k).upper(): int(v) for k, v in self.rv_norm_by_tf.items()}
        self.zscore_by_tf = {str(k).upper(): int(v) for k, v in self.zscore_by_tf.items()}

    def to_dict(self) -> dict:
        return {
            "htf_bar_sizes": list(self.htf_bar_sizes),
            "context_symbols": dict(self.context_symbols),
            "rv_windows_by_tf": {k: list(v) for k, v in self.rv_windows_by_tf.items()},
            "rv_norm_by_tf": dict(self.rv_norm_by_tf),
            "zscore_by_tf": dict(self.zscore_by_tf),
            "min_coverage": float(self.min_coverage),
            "session_open_hour_utc": int(self.session_open_hour_utc),
            "min_session_minutes": int(self.min_session_minutes),
            "vol_fast_minutes": int(self.vol_fast_minutes),
            "vol_baseline_minutes": int(self.vol_baseline_minutes),
            "er_window_minutes": int(self.er_window_minutes),
        }

    @classmethod
    def from_dict(cls, raw: dict | None) -> "MTFBiasConfig":
        if not raw:
            return cls()
        valid = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


def _infer_bar_ms(df: pd.DataFrame, fallback_bar_size: str | None = None) -> int:
    if fallback_bar_size:
        return bar_ms(fallback_bar_size)
    if len(df) >= 2:
        diffs = np.diff(np.sort(df["time"].to_numpy(dtype=np.int64)))
        pos = diffs[diffs > 0]
        if len(pos):
            return int(np.median(pos))
    return 60_000


def resample_ohlcv_causal(
    bars: pd.DataFrame,
    target_bar_size: str,
    source_bar_size: str | None = None,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Aggregate lower-timeframe OHLCV bars into closed higher-timeframe bars.

    Each output bar at ``time = T`` covers ``[T, T + target_ms)``. When
    ``drop_incomplete=True`` (default), a trailing HTF bucket whose constituent
    source bars have not yet reached ``T + target_ms`` is excluded so in-flight
    partial candles never masquerade as closed HTF bars.
    """
    cols = ["time", "open", "high", "low", "close", "volume"]
    if bars is None or bars.empty:
        return pd.DataFrame(columns=cols)

    target_ms = bar_ms(target_bar_size)
    src_ms = _infer_bar_ms(bars, source_bar_size)
    if target_ms < src_ms:
        raise ValueError(
            f"cannot resample coarser bars ({src_ms} ms) to finer target {target_bar_size} ({target_ms} ms)"
        )
    if target_ms == src_ms:
        out = bars.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
        if "volume" not in out.columns:
            out["volume"] = 0.0
        return out[cols].copy()

    src = bars.sort_values("time").drop_duplicates("time", keep="last").copy()
    if "volume" not in src.columns:
        src["volume"] = 0.0
    t_arr = src["time"].to_numpy(dtype=np.int64, copy=True)
    src["_bucket"] = (t_arr // target_ms) * target_ms
    src["_src_close"] = t_arr + src_ms

    grouped = (
        src.groupby("_bucket", as_index=False, sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            last_src_close=("_src_close", "max"),
        )
        .rename(columns={"_bucket": "time"})
    )
    if drop_incomplete and not grouped.empty:
        # An HTF bucket [T, T + target_ms) is closed iff we observed a source bar
        # closing at T + target_ms (or a later bucket has already started).
        max_seen_close = int(grouped["last_src_close"].max())
        is_closed = (grouped["time"] + target_ms <= grouped["last_src_close"]) | (
            grouped["time"] + target_ms <= max_seen_close
        )
        grouped = grouped.loc[is_closed].reset_index(drop=True)

    return grouped[cols].copy()


def align_htf_to_ltf(
    ltf_bars: pd.DataFrame,
    htf_frame: pd.DataFrame,
    ltf_bar_size: str,
    htf_bar_size: str,
    value_columns: list[str] | tuple[str, ...],
) -> pd.DataFrame:
    """Causally project ``value_columns`` from ``htf_frame`` onto ``ltf_bars``.

    Guarantees ``T_htf + htf_ms <= t_ltf + ltf_ms`` for every aligned row.
    """
    n = len(ltf_bars)
    out = pd.DataFrame(index=ltf_bars.index)
    if n == 0 or htf_frame is None or htf_frame.empty or not value_columns:
        for col in value_columns:
            out[col] = 0.0
        return out

    ltf_ms = bar_ms(ltf_bar_size)
    htf_ms = bar_ms(htf_bar_size)

    left = pd.DataFrame(
        {
            "_orig_pos": np.arange(n, dtype=np.int64),
            "_ltf_close": ltf_bars["time"].to_numpy(dtype=np.int64, copy=True) + ltf_ms,
        }
    ).sort_values("_ltf_close", kind="mergesort")

    avail_cols = [c for c in value_columns if c in htf_frame.columns]
    right = htf_frame[["time"] + avail_cols].copy()
    right["_htf_close"] = right["time"].to_numpy(dtype=np.int64, copy=True) + htf_ms
    right = (
        right.drop(columns=["time"])
        .sort_values("_htf_close", kind="mergesort")
        .drop_duplicates("_htf_close", keep="last")
    )

    merged = pd.merge_asof(
        left,
        right,
        left_on="_ltf_close",
        right_on="_htf_close",
        direction="backward",
    ).sort_values("_orig_pos", kind="mergesort")

    # Anti-look-ahead invariant assertion
    valid = merged["_htf_close"].notna()
    if valid.any():
        leaked = merged.loc[valid, "_htf_close"] > merged.loc[valid, "_ltf_close"]
        if bool(leaked.any()):
            raise RuntimeError("Causal MTF alignment invariant violated: HTF close > LTF close")

    for col in value_columns:
        if col in merged.columns:
            vals = merged[col].to_numpy(dtype=np.float64, copy=True)
            out[col] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            out[col] = 0.0
    return out


def intraday_session_features(
    bars: pd.DataFrame,
    bar_size: str | None = None,
    session_open_hour_utc: int = 22,
    min_session_minutes: int = 60,
    vol_fast_minutes: int = 60,
    vol_baseline_minutes: int = 1440,
    er_window_minutes: int = 120,
) -> pd.DataFrame:
    """Compute strictly causal intraday session VWAP, volatility regime & efficiency."""
    idx = bars.index
    n = len(bars)
    cols = (
        "intraday_session_vwap_z",
        "intraday_session_trend",
        "intraday_vol_regime",
        "intraday_er_2h",
    )
    if n == 0:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in cols}, index=idx)

    dt_ms = _infer_bar_ms(bars, bar_size)
    bar_min = max(1.0, dt_ms / 60_000.0)
    bars_1h = max(2, int(round(vol_fast_minutes / bar_min)))
    bars_24h = max(bars_1h * 4, int(round(vol_baseline_minutes / bar_min)))
    bars_er = max(4, int(round(er_window_minutes / bar_min)))
    min_sess_bars = max(2, int(round(min_session_minutes / bar_min)))

    times = bars["time"].to_numpy(dtype=np.int64, copy=True)
    high = bars["high"].to_numpy(dtype=np.float64, copy=True)
    low = bars["low"].to_numpy(dtype=np.float64, copy=True)
    close = bars["close"].to_numpy(dtype=np.float64, copy=True)
    vol = (
        bars["volume"].fillna(0.0).to_numpy(dtype=np.float64, copy=True)
        if "volume" in bars.columns
        else np.ones(n, dtype=np.float64)
    )

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr_14 = pd.Series(tr).ewm(alpha=1.0 / 14.0, adjust=False).mean().to_numpy(dtype=np.float64, copy=True)

    # Shift timestamps so `session_open_hour_utc` (22:00 UTC for Gold/Globex) starts a new session day
    shift_ms = (24 - int(session_open_hour_utc)) % 24 * 3_600_000
    sess_day = (times + shift_ms) // 86_400_000
    typical_px = (high + low + close) / 3.0

    vwap_z = np.zeros(n, dtype=np.float64)
    sess_cnt = np.zeros(n, dtype=np.int64)
    cum_pv = 0.0
    cum_v = 0.0
    cur_day = -1
    cnt = 0
    scale_1h = math.sqrt(float(bars_1h))
    for i in range(n):
        d = int(sess_day[i])
        if d != cur_day:
            cur_day = d
            cum_pv = 0.0
            cum_v = 0.0
            cnt = 0
        cnt += 1
        v = max(float(vol[i]), 1.0)
        cum_pv += float(typical_px[i]) * v
        cum_v += v
        vwap = cum_pv / cum_v
        denom = max(float(atr_14[i]) * scale_1h, float(close[i]) * 1e-6, 1e-9)
        vwap_z[i] = np.clip((float(close[i]) - vwap) / denom, -5.0, 5.0)
        sess_cnt[i] = cnt

    sess_trend = np.where(sess_cnt >= min_sess_bars, np.tanh(vwap_z / 1.5), 0.0)

    # Intraday Volatility Regime: centered at 0.0 (= normal 1.0x median 24h volatility)
    # Centering at 0.0 preserves clean zero-baseline semantics for sequence models while
    # keeping exact recovery `vol_ratio = 1.0 + intraday_vol_regime`.
    atr_fast = pd.Series(high - low).rolling(bars_1h, min_periods=max(2, bars_1h // 3)).mean().bfill()
    atr_base = atr_fast.rolling(bars_24h, min_periods=max(8, bars_24h // 6)).median().replace(0.0, np.nan)
    vol_ratio = (atr_fast / atr_base).fillna(1.0).to_numpy(dtype=np.float64, copy=True)
    vol_excess = np.clip(vol_ratio - 1.0, -0.9, 4.0)

    # 2H Kaufman Efficiency Ratio in [0, 1]
    cl_s = pd.Series(close)
    net_move = cl_s.diff(bars_er).abs()
    path_move = cl_s.diff(1).abs().rolling(bars_er, min_periods=max(3, bars_er // 4)).sum().replace(0.0, np.nan)
    er_2h = (net_move / path_move).clip(0.0, 1.0).fillna(0.25).to_numpy(dtype=np.float64, copy=True)

    return pd.DataFrame(
        {
            "intraday_session_vwap_z": vwap_z,
            "intraday_session_trend": sess_trend,
            "intraday_vol_regime": vol_excess,
            "intraday_er_2h": er_2h,
        },
        index=idx,
    )


def mtf_feature_names(
    cfg: MTFBiasConfig, context_symbols: dict[str, str] | None = None
) -> tuple[str, ...]:
    names: list[str] = []
    labels = list((context_symbols or {}).values())
    for tf in cfg.htf_bar_sizes:
        tf_low = tf.lower()
        rv_ws = cfg.rv_windows_by_tf.get(tf, (1, 2, 4))
        for label in labels:
            for k in rv_ws:
                names.append(f"htf_{tf_low}_{label}_rv_z_{k}")
    names.extend(
        (
            "intraday_session_vwap_z",
            "intraday_session_trend",
            "intraday_vol_regime",
            "intraday_er_2h",
        )
    )
    return tuple(names)


def compute_mtf_bias_features(
    ltf_bars: pd.DataFrame,
    ltf_bar_size: str,
    cfg: MTFBiasConfig,
    context_symbols: dict[str, str] | None = None,
    ltf_context: dict[str, pd.DataFrame] | None = None,
    htf_bars_by_tf: dict[str, pd.DataFrame] | None = None,
    htf_context_by_tf: dict[str, dict[str, pd.DataFrame]] | None = None,
) -> pd.DataFrame:
    """Compute causal HTF relative-value bias + intraday session regime features.

    Supports both explicit higher-timeframe bar frames (``htf_bars_by_tf``) AND
    automatic causal resampling from ``ltf_bars`` / ``ltf_context`` when separate
    HTF frames are not supplied, guaranteeing zero train/serve skew.
    """
    out = pd.DataFrame(index=ltf_bars.index)
    ltf_ms = bar_ms(ltf_bar_size)
    symbols_map = dict(context_symbols or {})

    for tf in cfg.htf_bar_sizes:
        tf_up = tf.upper()
        tf_low = tf.lower()
        tf_ms = bar_ms(tf_up)
        rv_ws = list(cfg.rv_windows_by_tf.get(tf_up, (1, 2, 4)))
        expected_cols = [
            f"htf_{tf_low}_{label}_rv_z_{k}"
            for label in symbols_map.values()
            for k in rv_ws
        ]
        if not expected_cols or tf_ms < ltf_ms:
            for col in expected_cols:
                out[col] = 0.0
            continue

        # 1. Obtain closed HTF primary bars
        htf_bars = (htf_bars_by_tf or {}).get(tf_up)
        if htf_bars is None or htf_bars.empty:
            htf_bars = resample_ohlcv_causal(
                ltf_bars, tf_up, source_bar_size=ltf_bar_size, drop_incomplete=True
            )
        else:
            htf_bars = htf_bars.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)

        # 2. Obtain closed HTF context bars
        htf_ctx: dict[str, pd.DataFrame] = {}
        explicit_ctx = (htf_context_by_tf or {}).get(tf_up) or {}
        for sym in symbols_map:
            cframe = explicit_ctx.get(sym)
            if cframe is None or cframe.empty:
                src_ctx = (ltf_context or {}).get(sym)
                if src_ctx is not None and not src_ctx.empty:
                    cframe = resample_ohlcv_causal(src_ctx, tf_up, drop_incomplete=True)
            if cframe is not None and not cframe.empty:
                htf_ctx[sym] = cframe

        if htf_bars.empty or len(htf_ctx) < len(symbols_map):
            for col in expected_cols:
                out[col] = 0.0
            continue

        # 3. Compute causal HTF relative-value features on the HTF grid
        htf_ctx_cfg = ContextConfig(
            symbols=symbols_map,
            zscore_window=int(cfg.zscore_by_tf.get(tf_up, 96)),
            min_coverage=float(cfg.min_coverage),
            include_rv_features=True,
            rv_windows=rv_ws,
            rv_norm_window=int(cfg.rv_norm_by_tf.get(tf_up, 48)),
        )
        aligned_ctx = align_context(htf_bars, htf_ctx)
        htf_feats = context_features(aligned_ctx, htf_ctx_cfg)

        htf_pack = pd.DataFrame({"time": htf_bars["time"].to_numpy(dtype=np.int64, copy=True)})
        for label in symbols_map.values():
            for k in rv_ws:
                src_col = f"ctx_{label}_rv_z_{k}"
                dst_col = f"htf_{tf_low}_{label}_rv_z_{k}"
                if src_col in htf_feats.columns:
                    htf_pack[dst_col] = htf_feats[src_col].to_numpy(dtype=np.float64, copy=True)
                else:
                    htf_pack[dst_col] = 0.0

        # 4. Project closed HTF features onto LTF execution grid (T_close <= t_close)
        projected = align_htf_to_ltf(
            ltf_bars=ltf_bars,
            htf_frame=htf_pack,
            ltf_bar_size=ltf_bar_size,
            htf_bar_size=tf_up,
            value_columns=expected_cols,
        )
        for col in expected_cols:
            out[col] = projected[col]

    sess_df = intraday_session_features(
        ltf_bars,
        bar_size=ltf_bar_size,
        session_open_hour_utc=cfg.session_open_hour_utc,
        min_session_minutes=cfg.min_session_minutes,
        vol_fast_minutes=cfg.vol_fast_minutes,
        vol_baseline_minutes=cfg.vol_baseline_minutes,
        er_window_minutes=cfg.er_window_minutes,
    )
    for col in sess_df.columns:
        out[col] = sess_df[col]
    return out

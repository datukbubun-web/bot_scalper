"""Product-grade feature engineering for FX scalping.

Why not raw OHLCV
-----------------
A 1-minute bar of EUR/USD carries almost no stationary information: price level
is non-stationary, and the raw candle is a 4-number summary that destroys the
path inside the bar. Institutional scalpers feed the model **stationary,
scale-free, and microstructure-aware** features:

======================  =====================================================
group                   features
======================  =====================================================
returns                 normalised log-returns over 1, 3, 5, 13 bars
momentum                EMA fast/slow spread, MACD histogram, RSI(7), Stoch %K
volatility              ATR(14)/price, Garman-Klass, Parkinson, realised vol,
                        Bollinger bandwidth, range expansion
order flow              bar-level order-flow imbalance (tick-rule proxy),
                        quote imbalance (bid/ask size), microprice deviation,
                        signed-volume z-score, OBV slope, Kyle lambda proxy
microstructure          spread in bps, roll measure, quote intensity
session / seasonality   hour-of-day (sin/cos), minute-of-hour, weekday,
                        Asia / London / New York session overlap flags
regime                  rolling skew, kurtosis, lag-1 autocorrelation
======================  =====================================================

Stationarity & leakage rules
----------------------------
1. Every feature at bar ``t`` uses only data with ``timestamp <= t``.
2. Normalisation is a **causal rolling** robust z-score (median / MAD), never a
   global mean/std fitted on the full sample -- that is the classic silent
   look-ahead bug in crypto/FX notebooks.
3. Labels are built with an explicit horizon and a purge gap; see ``labels.py``.

The same code path is used for training, backtesting and live inference: live
inference calls :meth:`FeatureEngineer.transform` on a rolling buffer of the
last ``lookback + warmup`` bars, so there is no train/serve skew.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["FeatureConfig", "FeatureEngineer", "FEATURE_GROUPS"]


@dataclass
class FeatureConfig:
    """All windows in **bars**. Defaults assume the 1-minute scalping timeframe."""

    short_returns: tuple[int, ...] = (1, 3, 5, 13)
    atr_window: int = 14
    vol_window: int = 20
    ema_fast: int = 8
    ema_slow: int = 21
    rsi_window: int = 7
    stoch_window: int = 14
    bb_window: int = 20
    ofi_window: int = 20
    regime_window: int = 60
    norm_window: int = 500          # causal robust-z lookback
    norm_min_periods: int = 100
    clip_sigma: float = 8.0
    include_time_features: bool = True
    include_quote_features: bool = True
    #: When historical bars lack ``bid``/``ask``, populate ``spread_bps``,
    #: ``quote_imbalance`` and ``microprice_dev`` with causal OHLCV estimators
    #: (Abdi-Ranaldo + Corwin-Schultz spread, BVC + wick imbalance).
    synthetic_microstructure: bool = False
    #: Attach cross-asset context features (silver, dollar proxies). Requires the
    #: context bars to be passed to ``transform``; changes n_features, so it is
    #: recorded in the model manifest and must match at inference time.
    include_context_features: bool = False
    context_config: dict = field(default_factory=dict)
    #: Attach scheduled macro-release features (NFP / CPI / FOMC). Only the
    #: *schedule* is used -- release times are public months ahead, the content
    #: is never touched, so this is not look-ahead.
    include_calendar_features: bool = False
    calendar_config: dict = field(default_factory=dict)
    #: Attach institutional Gold session/fixing interaction features (SGE AM
    #: post-fix unwind, post-LBMA-AM / pre-COMEX inventory reversion, daily roll).
    include_gold_session_features: bool = False
    #: Attach causal multi-timeframe (HTF) Intraday Bias & Session Regime features
    #: (T_htf + htf_ms <= t_ltf + ltf_ms). Used by the Hybrid Architecture.
    include_mtf_bias_features: bool = False
    mtf_bias_config: dict = field(default_factory=dict)
    bar_size: str = ""
    #: Hours (UTC) of the London/NY overlap, where FX liquidity is deepest.
    session_windows: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {
            "asia": (0, 8),
            "london": (7, 16),
            "newyork": (12, 21),
        }
    )

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["short_returns"] = list(self.short_returns)
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if "short_returns" in raw:
            raw["short_returns"] = tuple(raw["short_returns"])
        if "session_windows" in raw:
            raw["session_windows"] = {k: tuple(v) for k, v in raw["session_windows"].items()}
        valid = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "returns": ("ret_1", "ret_3", "ret_5", "ret_13"),
    "momentum": ("ema_spread", "macd_hist", "rsi_7", "stoch_k"),
    "volatility": ("atr_14", "garman_klass", "parkinson", "realized_vol", "bb_bandwidth"),
    "structure": ("range_expansion", "close_location", "bb_z", "ema_dist"),
    "orderflow": ("ofi", "signed_volume_z", "obv_slope", "kyle_lambda"),
    "microstructure": ("spread_bps", "quote_imbalance", "microprice_dev", "roll_measure"),
    "session": ("hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos",
                "sess_asia", "sess_london", "sess_newyork", "sess_overlap"),
    "regime": ("skew", "kurtosis", "autocorr_1"),
}

#: Context columns are generated per symbol label, so they are not in the frozen
#: FEATURE_NAMES tuple; ``FeatureEngineer`` appends them and freezes the result.
CONTEXT_GROUP = "context"

#: Columns the model consumes, in a frozen order (bump MODEL_FEATURE_VERSION on change).
FEATURE_NAMES: tuple[str, ...] = tuple(n for group in FEATURE_GROUPS.values() for n in group)
MODEL_FEATURE_VERSION = "fx-scalp-1.0.0"


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> pd.Series:
    a = pd.Series(a) if not isinstance(a, pd.Series) else a
    b = pd.Series(b) if not isinstance(b, pd.Series) else b
    return a / b.replace(0.0, np.nan)


class FeatureEngineer:
    """Stateless-by-design transformer: fit() only freezes column order/metadata."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.feature_names: tuple[str, ...] = FEATURE_NAMES
        if self.config.include_context_features:
            from .context import ContextConfig, context_feature_names

            cfg = ContextConfig(**self.config.context_config)
            # Column order must be identical for training and inference, so the
            # names come from the config, never from a trial pass over data.
            self.context_names = context_feature_names(cfg)
            self.feature_names = FEATURE_NAMES + self.context_names
            self._context_cfg = cfg
        else:
            self.context_names = ()
            self._context_cfg = None

        if self.config.include_calendar_features:
            from .calendar import CalendarConfig, calendar_feature_names

            self._calendar_cfg = CalendarConfig(**self.config.calendar_config)
            self.calendar_names = calendar_feature_names(self._calendar_cfg)
            self.feature_names = tuple(self.feature_names) + self.calendar_names
        else:
            self.calendar_names = ()
            self._calendar_cfg = None

        if self.config.include_gold_session_features:
            self.gold_session_names: tuple[str, ...] = (
                "gold_ny12_rev",
                "gold_sge02_unwind",
                "gold_roll_blackout",
            )
            self.feature_names = tuple(self.feature_names) + self.gold_session_names
        else:
            self.gold_session_names = ()

        if self.config.include_mtf_bias_features:
            from .mtf import MTFBiasConfig, mtf_feature_names

            self._mtf_cfg = MTFBiasConfig.from_dict(self.config.mtf_bias_config)
            ctx_syms = (
                self._context_cfg.symbols if self._context_cfg is not None else None
            ) or self._mtf_cfg.context_symbols
            self.mtf_names = mtf_feature_names(self._mtf_cfg, context_symbols=ctx_syms)
            self.feature_names = tuple(self.feature_names) + self.mtf_names
        else:
            self.mtf_names = ()
            self._mtf_cfg = None

        suffix = (
            ("+ctx" if self.context_names else "")
            + ("+cal" if self.calendar_names else "")
            + ("+synmicro" if self.config.synthetic_microstructure else "")
            + ("+gold" if self.gold_session_features_active else "")
            + ("+mtf" if self.mtf_names else "")
        )
        self.version = f"{MODEL_FEATURE_VERSION}{suffix}"

    @property
    def gold_session_features_active(self) -> bool:
        return bool(self.gold_session_names)

    # ------------------------------------------------------------------ public
    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def transform(
        self,
        bars: pd.DataFrame,
        context: dict[str, pd.DataFrame] | None = None,
        htf_bars: dict[str, pd.DataFrame] | None = None,
        htf_context: dict[str, dict[str, pd.DataFrame]] | None = None,
    ) -> pd.DataFrame:
        """Raw bars -> feature frame (un-normalised), index aligned to ``bars``.

        ``bars`` needs columns: ``time, open, high, low, close, volume`` and
        optionally ``bid, ask, bid_size, ask_size`` when quote data is replayed.
        ``context`` maps symbol -> OHLCV frame of the same bar size and is
        required when ``include_context_features`` is on.
        """
        df = self._validate(bars)
        cfg = self.config
        out = pd.DataFrame(index=df.index)

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        volume = df["volume"].fillna(0.0)

        # ---------------------------------------------------------- returns
        log_ret = np.log(close).diff()
        vol_base = log_ret.rolling(cfg.vol_window, min_periods=max(2, cfg.vol_window // 2)).std()
        for w in cfg.short_returns:
            scale = log_ret.rolling(w).std() if w >= 3 else vol_base
            out[f"ret_{w}"] = np.log(close).diff(w) / (scale * np.sqrt(w) + 1e-12)

        # --------------------------------------------------------- momentum
        ema_f = close.ewm(span=cfg.ema_fast, adjust=False).mean()
        ema_s = close.ewm(span=cfg.ema_slow, adjust=False).mean()
        out["ema_spread"] = _safe_div(ema_f - ema_s, close)
        macd = ema_f - ema_s
        signal = macd.ewm(span=9, adjust=False).mean()
        out["macd_hist"] = _safe_div(macd - signal, close)
        out["rsi_7"] = _rsi(close, cfg.rsi_window) / 50.0 - 1.0
        out["stoch_k"] = _stochastic(high, low, close, cfg.stoch_window)

        # -------------------------------------------------------- volatility
        atr = _atr(high, low, close, cfg.atr_window)
        out["atr_14"] = _safe_div(atr, close)
        out["garman_klass"] = _garman_klass(open_, high, low, close).rolling(
            cfg.vol_window, min_periods=cfg.vol_window // 2
        ).mean()
        out["parkinson"] = _parkinson(high, low).rolling(
            cfg.vol_window, min_periods=cfg.vol_window // 2
        ).mean()
        out["realized_vol"] = log_ret.rolling(cfg.vol_window).std()
        bb_mid = close.rolling(cfg.bb_window).mean()
        bb_std = close.rolling(cfg.bb_window).std()
        out["bb_bandwidth"] = _safe_div(4.0 * bb_std, bb_mid)
        out["bb_z"] = _safe_div(close - bb_mid, bb_std)
        out["ema_dist"] = _safe_div(close - ema_s, ema_s)

        # --------------------------------------------------------- structure
        out["range_expansion"] = _safe_div(high - low, close)
        out["close_location"] = _safe_div(close - low, (high - low).replace(0.0, np.nan)).fillna(0.5) * 2 - 1

        # --------------------------------------------------------- order flow
        # Bar-level order-flow imbalance using the tick rule: a bar closing in
        # the upper half is treated as buy-initiated. This is a proxy -- the
        # exchange does not stream L2 on API v3 -- and it is deliberately kept
        # scale-free so it survives volatility regime changes.
        direction = np.sign(close.diff().fillna(0.0))
        signed_vol = direction * volume
        ofi = _safe_div(
            signed_vol.rolling(cfg.ofi_window).sum(),
            volume.rolling(cfg.ofi_window).sum().replace(0.0, np.nan),
        )
        out["ofi"] = ofi
        out["signed_volume_z"] = _robust_z(signed_vol, cfg.vol_window)
        obv = (direction * volume).cumsum()
        out["obv_slope"] = _safe_div(obv.diff(10), volume.rolling(10).sum().replace(0.0, np.nan))
        # Kyle's lambda: price impact per unit of signed volume.
        out["kyle_lambda"] = _safe_div(log_ret.abs(), np.sqrt(volume + 1.0)).rolling(
            cfg.ofi_window
        ).mean()

        # ---------------------------------------------------- microstructure
        if {"bid", "ask"}.issubset(df.columns) and df["bid"].notna().any():
            mid = (df["bid"] + df["ask"]) / 2.0
            out["spread_bps"] = _safe_div(df["ask"] - df["bid"], mid) * 1e4
            bid_size = df.get("bid_size", pd.Series(1.0, index=df.index)).fillna(1.0)
            ask_size = df.get("ask_size", pd.Series(1.0, index=df.index)).fillna(1.0)
            out["quote_imbalance"] = _safe_div(bid_size - ask_size, bid_size + ask_size).fillna(0.0)
            microprice = _safe_div(df["bid"] * ask_size + df["ask"] * bid_size, bid_size + ask_size)
            out["microprice_dev"] = _safe_div(microprice - mid, mid) * 1e4
            out["roll_measure"] = np.sqrt(
                (-log_ret * log_ret.shift(1)).rolling(cfg.ofi_window).mean().clip(lower=0)
            )
        else:
            if cfg.synthetic_microstructure:
                from .microstructure import (
                    bvc_order_imbalance,
                    synthetic_microprice_dev_bps,
                    synthetic_spread_bps,
                )

                out["spread_bps"] = synthetic_spread_bps(high, low, close, window=cfg.ofi_window)
                out["quote_imbalance"] = bvc_order_imbalance(
                    open_, high, low, close, volume, window=cfg.ofi_window
                )
                out["microprice_dev"] = synthetic_microprice_dev_bps(open_, high, low, close)
            else:
                # honest fallback: no quote data -> neutral, not fake
                out["spread_bps"] = np.nan
                out["quote_imbalance"] = 0.0
                out["microprice_dev"] = 0.0
            out["roll_measure"] = np.sqrt(
                (-log_ret * log_ret.shift(1)).rolling(cfg.ofi_window).mean().clip(lower=0)
            )

        # ----------------------------------------------------------- session
        if cfg.include_time_features:
            ts = pd.to_datetime(df["time"], unit="ms", utc=True)
            hours = ts.dt.hour + ts.dt.minute / 60.0
            out["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
            out["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
            out["minute_sin"] = np.sin(2 * np.pi * ts.dt.minute / 60.0)
            out["minute_cos"] = np.cos(2 * np.pi * ts.dt.minute / 60.0)
            out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 5.0)
            out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 5.0)
            for name, (start, end) in cfg.session_windows.items():
                out[f"sess_{name}"] = ((ts.dt.hour >= start) & (ts.dt.hour < end)).astype(float)
            out["sess_overlap"] = out.get("sess_london", 0.0) * out.get("sess_newyork", 0.0)

        # ------------------------------------------------------------ regime
        out["skew"] = log_ret.rolling(cfg.regime_window).skew()
        out["kurtosis"] = log_ret.rolling(cfg.regime_window).kurt()
        out["autocorr_1"] = log_ret.rolling(cfg.regime_window).apply(
            _lag1_autocorr, raw=True
        )

        # ------------------------------------------------------ cross-asset
        if self.context_names:
            if not context:
                raise ValueError(
                    "include_context_features=True but no context frames were passed; "
                    "call transform(bars, context={symbol: frame})"
                )
            from .context import align_context, context_features

            aligned = align_context(df, context)
            ctx = context_features(aligned, self._context_cfg)
            out = out.join(ctx)

        # ------------------------------------------------ macro schedule
        if getattr(self, "calendar_names", ()):
            from .calendar import calendar_features

            cal = calendar_features(df["time"].to_numpy(), config=self._calendar_cfg)
            cal.index = out.index
            out = out.join(cal)

        # ------------------------------------ gold session interactions
        if getattr(self, "gold_session_names", ()):
            ts_g = pd.to_datetime(df["time"], unit="ms", utc=True)
            tod = ts_g.dt.hour * 60 + ts_g.dt.minute
            bar_min = int(max(1, round(float(df["time"].diff().median() or 300_000) / 60_000.0)))
            bars_2h = max(1, int(round(120 / bar_min)))
            rvol_g = log_ret.rolling(max(24, bars_2h * 6), min_periods=10).std().replace(0.0, np.nan)
            mv_2h = (np.log(close).diff(bars_2h) / (rvol_g * np.sqrt(bars_2h))).clip(-3.5, 3.5).fillna(0.0)
            # Trigger on the bar closing at 12:00 UTC (pre-COMEX / post-LBMA-AM inventory fade)
            out["gold_ny12_rev"] = np.where(tod == 12 * 60 - bar_min, -mv_2h, 0.0)
            # Trigger on the bar closing at 02:00 UTC (post-SGE AM Benchmark unwind)
            out["gold_sge02_unwind"] = np.where(tod == 2 * 60 - bar_min, -1.0, 0.0)
            # Daily COMEX/CFD maintenance roll window (21:00 - 22:59 UTC)
            out["gold_roll_blackout"] = ((ts_g.dt.hour >= 21) & (ts_g.dt.hour < 23)).astype(float)

        # ------------------------------ multi-timeframe intraday bias & regime
        if getattr(self, "mtf_names", ()) and self._mtf_cfg is not None:
            from .mtf import compute_mtf_bias_features

            ltf_bs = cfg.bar_size
            if not ltf_bs:
                dt_med = int(max(60_000, round(float(df["time"].diff().median() or 60_000))))
                if dt_med >= 3_600_000:
                    ltf_bs = "1H"
                elif dt_med >= 900_000:
                    ltf_bs = "15M"
                elif dt_med >= 300_000:
                    ltf_bs = "5M"
                else:
                    ltf_bs = "1M"
            ctx_syms = (
                self._context_cfg.symbols if self._context_cfg is not None else None
            ) or self._mtf_cfg.context_symbols
            mtf_df = compute_mtf_bias_features(
                ltf_bars=df,
                ltf_bar_size=ltf_bs,
                cfg=self._mtf_cfg,
                context_symbols=ctx_syms,
                ltf_context=context,
                htf_bars_by_tf=htf_bars,
                htf_context_by_tf=htf_context,
            )
            mtf_df.index = out.index
            out = out.join(mtf_df)

        out = out.reindex(columns=list(self.feature_names))
        return out.replace([np.inf, -np.inf], np.nan)

    def normalize(self, features: pd.DataFrame) -> pd.DataFrame:
        """Causal robust z-score per column, clipped. Safe for live use."""
        cfg = self.config
        normalized = pd.DataFrame(index=features.index)
        for col in features.columns:
            series = features[col]
            if (
                col.startswith("sess_")
                or col.startswith("cal_")     # see below
                or col.startswith("gold_")
                or col.startswith("htf_")
                or col.startswith("intraday_")
                or (col.startswith("ctx_") and "_rv_z_" in col)
                or col in {
                    "hour_sin", "hour_cos", "minute_sin", "minute_cos", "dow_sin", "dow_cos",
                }
            ):
                # Already bounded and stationary; pass through unscaled.
                #
                # Calendar columns MUST bypass the robust z-score. A binary flag
                # is 1 on a handful of bars out of a 500-bar window, so its
                # rolling MAD is 0, the z-score becomes NaN and ``fillna(0.0)``
                # silently deletes the feature -- the model would never see a
                # release, while the column still existed and looked fine.
                normalized[col] = series.fillna(0.0)
                continue
            median = series.rolling(cfg.norm_window, min_periods=cfg.norm_min_periods).median()
            mad = (series - median).abs().rolling(
                cfg.norm_window, min_periods=cfg.norm_min_periods
            ).median()
            z = (series - median) / (1.4826 * mad.replace(0.0, np.nan) + 1e-9)
            normalized[col] = z.clip(-cfg.clip_sigma, cfg.clip_sigma) / cfg.clip_sigma * 4.0
        return normalized.fillna(0.0)

    def transform_normalized(
        self,
        bars: pd.DataFrame,
        context: dict[str, pd.DataFrame] | None = None,
        htf_bars: dict[str, pd.DataFrame] | None = None,
        htf_context: dict[str, dict[str, pd.DataFrame]] | None = None,
    ) -> pd.DataFrame:
        return self.normalize(
            self.transform(bars, context=context, htf_bars=htf_bars, htf_context=htf_context)
        )

    def make_windows(
        self, normalized: pd.DataFrame, lookback: int, drop_warmup: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """(N, C, L) float32 windows + the bar timestamp of each window's last row."""
        values = normalized.to_numpy(dtype=np.float32, copy=False)
        n = len(values)
        if n < lookback:
            return np.empty((0, values.shape[1], lookback), np.float32), np.empty((0,), dtype=np.int64)
        windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=lookback, axis=0)
        last_ts = np.arange(lookback - 1, n, dtype=np.int64)  # positional bar index
        if drop_warmup:
            # rows before warmup are filled with zeros by normalize(); drop them
            warmup = min(self.config.norm_min_periods + max(self.config.atr_window, self.config.regime_window), n)
            keep = last_ts >= warmup
            windows, last_ts = windows[keep], last_ts[keep]
        return np.ascontiguousarray(windows, dtype=np.float32), last_ts

    def last_window(
        self,
        bars: pd.DataFrame,
        lookback: int,
        context: dict[str, pd.DataFrame] | None = None,
        htf_bars: dict[str, pd.DataFrame] | None = None,
        htf_context: dict[str, dict[str, pd.DataFrame]] | None = None,
    ) -> np.ndarray | None:
        """(1, C, L) window for the newest bar -- the live inference path.

        The tail must be long enough that the causal rolling normalisation is
        identical to the one used at training time, otherwise the model sees
        slightly different inputs live than in the backtest (train/serve skew).
        """
        min_mtf_bars = 0
        if getattr(self, "_mtf_cfg", None) is not None:
            dt_ms = (
                int(max(60_000, round(float(bars["time"].diff().median() or 60_000))))
                if len(bars) >= 2 and "time" in bars.columns
                else 60_000
            )
            bar_min = max(1.0, dt_ms / 60_000.0)
            vol_base_bars = int(round(self._mtf_cfg.vol_baseline_minutes / bar_min))
            vol_fast_bars = int(round(self._mtf_cfg.vol_fast_minutes / bar_min))
            min_mtf_bars = vol_base_bars + vol_fast_bars + 200
        buffer = max(lookback + self.config.norm_window + 700, lookback + min_mtf_bars)
        bars_tail = bars.tail(buffer)
        if context:
            if len(bars_tail) and "time" in bars_tail.columns:
                min_t = int(bars_tail["time"].iloc[0]) - 7 * 86_400_000
                ctx_tail = {
                    k: (
                        v[v["time"] >= min_t]
                        if "time" in v.columns
                        else v.tail(buffer + 400)
                    )
                    for k, v in context.items()
                    if v is not None
                }
            else:
                ctx_tail = {
                    k: v.tail(buffer + 400) for k, v in context.items() if v is not None
                }
        else:
            ctx_tail = None
        normalized = self.transform_normalized(
            bars_tail,
            context=ctx_tail,
            htf_bars=htf_bars,
            htf_context=htf_context,
        )
        if len(normalized) < lookback:
            return None
        values = normalized.to_numpy(dtype=np.float32)[-lookback:]
        return np.transpose(values, (1, 0))[None, :, :]

    # ------------------------------------------------------------------ private
    @staticmethod
    def _validate(bars: pd.DataFrame) -> pd.DataFrame:
        required = {"time", "open", "high", "low", "close"}
        missing = required - set(bars.columns)
        if missing:
            raise ValueError(f"bars missing columns: {sorted(missing)}")
        df = bars.copy()
        if "volume" not in df.columns:
            df["volume"] = 0.0
        df = df.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        # Integer positional index: every row *is* a bar, and downstream code
        # (labels, windowing, the RL env) aligns on position. The bar timestamp
        # stays available in the ``time`` column.
        df.index = pd.RangeIndex(len(df))
        return df


# ---------------------------------------------------------------- indicators
def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / window, adjust=False).mean()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    lowest = low.rolling(window).min()
    highest = high.rolling(window).max()
    return _safe_div(close - lowest, (highest - lowest).replace(0.0, np.nan)) * 2.0 - 1.0


def _garman_klass(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    hl = np.log(_safe_div(high, low)) ** 2
    co = np.log(_safe_div(close, open_)) ** 2
    return np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * co).clip(lower=0.0))


def _parkinson(high: pd.Series, low: pd.Series) -> pd.Series:
    return np.sqrt((np.log(_safe_div(high, low)) ** 2) / (4 * np.log(2)))


def _robust_z(series: pd.Series, window: int) -> pd.Series:
    median = series.rolling(window, min_periods=max(3, window // 2)).median()
    mad = (series - median).abs().rolling(window, min_periods=max(3, window // 2)).median()
    return (series - median) / (1.4826 * mad.replace(0.0, np.nan) + 1e-12)


def _lag1_autocorr(values: np.ndarray) -> float:
    if len(values) < 3 or np.isnan(values).any():
        return 0.0
    x = values - values.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return 0.0
    return float((x[:-1] * x[1:]).sum() / denom)

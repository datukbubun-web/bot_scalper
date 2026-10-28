"""Cross-asset context features.

Gold does not move on its own. A 1-minute gold bar is largely a read-through of
the dollar and of silver, and a metal model that cannot see them is being asked
to infer the dollar from gold alone. This module joins *aligned* context series
onto the target's timeline and derives causal, scale-free features from them.

Alignment rules (all causal, all explicit)
------------------------------------------
1. A context bar is joined to the target bar with the **same open time**. Both
   bars close at the same instant, so at that instant both are fully known: this
   is the alignment a bar-close feature vector uses, in training and live alike.
   Lagging the context by one bar instead (an easy mistake: "make sure it is
   already closed") half-disaligns the series and destroys the signal -- it drove
   the XAU/XAG return correlation from a real +0.82 down to +0.008. Set
   ``lag_bars=1`` only if the venue genuinely publishes the context late.
   Bars without a matching context bar are dropped, never forward-filled.
2. Every derived feature is a return or a z-score, never a price level: levels
   are non-stationary and would let the model memorise a regime.
3. ``lead_*`` features are correlation/lead-lag estimates computed on a rolling
   window of *past* bars only.

If a context symbol is missing, the feature columns are dropped rather than
filled with zeros -- a fake zero is a lie the model will learn from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "ContextConfig",
    "align_context",
    "context_features",
    "context_feature_names",
    "CONTEXT_FEATURES",
    "PER_SYMBOL_SUFFIXES",
]


@dataclass
class ContextConfig:
    """Which context series to use and over what windows."""

    #: symbol -> short label used in column names
    symbols: dict[str, str] = field(
        default_factory=lambda: {
            "XAG/USD": "xag",     # silver leads gold
            "USD/CHF": "chf",     # dollar proxy (negative correlation to gold)
            "USD/JPY": "jpy",     # funding / risk appetite
            "USD/CAD": "cad",     # commodity-dollar, oil-linked
        }
    )
    windows: tuple[int, ...] = (1, 5, 20)
    corr_window: int = 120
    zscore_window: int = 120
    #: minimum fraction of non-NaN context bars required to keep a symbol
    min_coverage: float = 0.5


def align_context(
    target: pd.DataFrame,
    contexts: dict[str, pd.DataFrame],
    bar_ms: int = 60_000,
    lag_bars: int = 0,
) -> pd.DataFrame:
    """Join context series onto the target timeline by identical bar open time.

    ``contexts`` maps symbol -> OHLCV frame **of the same bar size**. Rows whose
    context bar is missing stay NaN; they are dropped downstream rather than
    filled, so a gap in one market cannot silently invent a value in another.
    """
    out = target.copy()
    key = out["time"].to_numpy(dtype=np.int64)
    for symbol, frame in contexts.items():
        if frame is None or frame.empty:
            continue
        ctx = frame.sort_values("time").copy()
        if lag_bars:
            ctx["time"] = ctx["time"] + lag_bars * bar_ms
        safe = symbol.replace("/", "")
        mapping = {
            "open": f"{safe}_open", "high": f"{safe}_high",
            "low": f"{safe}_low", "close": f"{safe}_close",
        }
        aligned = pd.DataFrame({"time": key}).merge(
            ctx[["time", "open", "high", "low", "close"]], on="time", how="left"
        )
        for src, dst in mapping.items():
            out[dst] = aligned[src].to_numpy()
    return out


def context_features(
    aligned: pd.DataFrame, config: ContextConfig | None = None
) -> pd.DataFrame:
    """Derive causal, scale-free features from already-aligned context columns."""
    cfg = config or ContextConfig()
    out = pd.DataFrame(index=aligned.index)
    target_ret = np.log(aligned["close"]).diff()

    for symbol, label in cfg.symbols.items():
        safe = symbol.replace("/", "")
        col = f"{safe}_close"
        if col not in aligned.columns:
            continue
        series = aligned[col]
        coverage = float(series.notna().mean())
        if coverage < cfg.min_coverage:
            log.warning("context %s dropped: only %.0f%% coverage", symbol, coverage * 100)
            continue
        ctx_ret = np.log(series).diff()

        for w in cfg.windows:
            out[f"ctx_{label}_ret_{w}"] = np.log(series).diff(w)
        # relative strength vs the target itself: does gold lead or lag?
        out[f"ctx_{label}_rel"] = ctx_ret - target_ret
        out[f"ctx_{label}_vol"] = ctx_ret.rolling(cfg.zscore_window).std()
        # is the context currently stretched? (mean reversion in the driver)
        out[f"ctx_{label}_z"] = (
            series - series.rolling(cfg.zscore_window).mean()
        ) / series.rolling(cfg.zscore_window).std().replace(0.0, np.nan)

        # Rolling correlation and beta of the target on the context. Written out
        # in explicit rolling means because ``Rolling.cov()`` on a DataFrame
        # returns MultiIndex columns and silently yields all-NaN if you slice it
        # naively (that bug cost 0% coverage on every beta column).
        w = cfg.corr_window
        mean_t = target_ret.rolling(w).mean()
        mean_c = ctx_ret.rolling(w).mean()
        dev_t = target_ret - mean_t
        dev_c = ctx_ret - mean_c
        cov_tc = (dev_t * dev_c).rolling(w).mean()
        var_c = (dev_c * dev_c).rolling(w).mean()
        var_t = (dev_t * dev_t).rolling(w).mean()
        out[f"ctx_{label}_corr"] = (cov_tc / np.sqrt(var_t * var_c)).to_numpy()
        out[f"ctx_{label}_beta"] = (cov_tc / var_c.replace(0.0, np.nan)).to_numpy()

        # Lead-lag, causally: how strongly did the target follow the *previous*
        # context bar over the last window? Using ctx_ret (same bar) against the
        # target's next return would be look-ahead.
        out[f"ctx_{label}_lead"] = (
            target_ret.rolling(w).corr(ctx_ret.shift(1))
        ).to_numpy()

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


PER_SYMBOL_SUFFIXES: tuple[str, ...] = (
    "ret_1", "ret_5", "ret_20", "rel", "vol", "z", "corr", "beta", "lead",
)


def context_feature_names(config: ContextConfig | None = None) -> tuple[str, ...]:
    """Column names ``context_features`` will produce, derived from config only.

    Deriving the names from the config (not from a trial run on data) is what
    makes the training and inference column order identical.
    """
    cfg = config or ContextConfig()
    return tuple(
        f"ctx_{label}_{suffix}"
        for label in cfg.symbols.values()
        for suffix in PER_SYMBOL_SUFFIXES
    )


CONTEXT_FEATURES: tuple[str, ...] = (
    "ctx_{sym}_ret_1", "ctx_{sym}_ret_5", "ctx_{sym}_ret_20",
    "ctx_{sym}_rel", "ctx_{sym}_vol", "ctx_{sym}_z",
    "ctx_{sym}_corr", "ctx_{sym}_beta", "ctx_{sym}_lead",
)

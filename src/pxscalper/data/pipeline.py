"""Market data acquisition and caching.

* :meth:`DataPipeline.download` pulls bars from the public (unauthenticated)
  PrimeXBT ``/v3/trading/bars`` route and caches them as Parquet/CSV so training
  is reproducible offline.
* :meth:`DataPipeline.synthetic` builds a realistic microstructure simulation
  (regime-switching volatility, fat tails, mean reversion + trend, spread) so the
  whole repo can be smoke-tested with no credentials and no network.

Bars are the *feed bid*; the bot quotes ask from the live ``quotes`` route and
models the spread explicitly rather than assuming mid-price fills.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils import BAR_MS, now_ms

log = logging.getLogger(__name__)

__all__ = ["DataPipeline", "BAR_SIZES"]

BAR_SIZES = tuple(BAR_MS.keys())


@dataclass
class DataPipeline:
    cache_dir: Path = Path("data")
    client: object | None = None  # PrimeXBTClient, injected to avoid a hard import

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ caching
    def _path(self, symbol: str, bar_size: str, suffix: str = ".parquet") -> Path:
        safe = symbol.replace("/", "")
        return self.cache_dir / f"{safe}_{bar_size}{suffix}"

    def load(self, symbol: str, bar_size: str) -> pd.DataFrame | None:
        """Read the cache, whatever format it was written in."""
        for suffix in (".parquet", ".csv.gz", ".csv"):
            path = self._path(symbol, bar_size, suffix)
            if path.exists():
                frame = (
                    pd.read_parquet(path)
                    if suffix == ".parquet"
                    else pd.read_csv(path, compression="infer")
                )
                return frame
        return None

    def save(self, symbol: str, bar_size: str, bars: pd.DataFrame) -> Path:
        """Parquet when pyarrow is available, else gzipped CSV (same API)."""
        bars = bars.sort_values("time").drop_duplicates("time").reset_index(drop=True)
        try:
            path = self._path(symbol, bar_size, ".parquet")
            bars.to_parquet(path, index=False)
        except ImportError:
            path = self._path(symbol, bar_size, ".csv.gz")
            bars.to_csv(path, index=False, compression="gzip")
            log.info("pyarrow absent -> cached as compressed CSV")
        log.info("cached %d bars -> %s", len(bars), path)
        return path

    # ---------------------------------------------------------------- download
    def download(
        self,
        symbol: str,
        bar_size: str = "1M",
        n_bars: int = 5000,
        end_ms: int | None = None,
        pause_s: float = 0.05,
    ) -> pd.DataFrame:
        """Download ``n_bars`` of history, paging backwards (500 bars per call max)."""
        if self.client is None:
            raise RuntimeError("DataPipeline.download needs a PrimeXBTClient")
        end = int(end_ms if end_ms is not None else now_ms())
        collected: list[dict] = []
        # page backwards: ask for bars up to `cursor`, take the oldest time as next cursor
        while len(collected) < n_bars:
            want = min(500, n_bars - len(collected))
            chunk = self.client.bars(symbol, bar_size=bar_size, limit=want, end_ms=end)
            if not chunk:
                break
            rows = [
                {"time": b.time, "open": b.open, "high": b.high, "low": b.low,
                 "close": b.close, "volume": b.volume}
                for b in chunk
            ]
            collected = rows + collected
            oldest = min(r["time"] for r in rows)
            if oldest >= end:
                break
            end = oldest - 1
            time.sleep(pause_s)
        bars = pd.DataFrame(collected).drop_duplicates("time").sort_values("time").reset_index(drop=True)
        if bars.empty:
            raise RuntimeError(f"no bars returned for {symbol} {bar_size}")
        self.save(symbol, bar_size, bars)
        return bars

    def update(self, symbol: str, bar_size: str = "1M", n_bars: int = 2000) -> pd.DataFrame:
        """Incremental update of the cache (only fetches what is missing)."""
        cached = self.load(symbol, bar_size)
        if cached is None or cached.empty:
            return self.download(symbol, bar_size, n_bars)
        last = int(cached["time"].iloc[-1])
        fresh = self.download(symbol, bar_size, n_bars, end_ms=now_ms())
        merged = (
            pd.concat([cached, fresh], ignore_index=True)
            .drop_duplicates("time")
            .sort_values("time")
            .reset_index(drop=True)
        )
        if len(merged) > len(cached):
            self.save(symbol, bar_size, merged)
        else:
            log.info("%s %s already up to date (last bar %s)", symbol, bar_size, last)
        return merged

    def load_multi(
        self, symbols: Iterable[str], bar_size: str = "1M", min_bars: int = 500
    ) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            df = self.load(symbol, bar_size)
            if df is None or len(df) < min_bars:
                log.warning("insufficient cached data for %s (%s bars)", symbol,
                            0 if df is None else len(df))
                continue
            out[symbol] = df
        return out

    # ---------------------------------------------------------------- synthetic
    @staticmethod
    def synthetic(
        symbol: str = "EUR/USD",
        n_bars: int = 20_000,
        bar_size: str = "1M",
        seed: int = 7,
        start_price: float = 1.1000,
        start_ms: int | None = None,
    ) -> pd.DataFrame:
        """Regime-switching synthetic FX series with realistic microstructure.

        Not a toy random walk: volatility clusters (2-state Markov), returns have
        fat tails (Student-t innovations), there is weak short-horizon mean
        reversion inside a slow trend, and the spread widens when volatility
        rises -- exactly the conditions a scalper is punished by.
        """
        rng = np.random.default_rng(seed)
        step = BAR_MS[bar_size]

        # --- volatility regime (calm / stressed) -----------------------------
        states = np.zeros(n_bars, dtype=int)
        p_stay = [0.995, 0.97]
        for i in range(1, n_bars):
            stay = p_stay[states[i - 1]]
            states[i] = states[i - 1] if rng.random() < stay else 1 - states[i - 1]
        vol = np.where(states == 0, 6e-5, 2.6e-4)

        # --- slow trend + intraday seasonality -------------------------------
        t = np.arange(n_bars)
        trend = 2e-7 * np.sin(2 * np.pi * t / (1440 * 3)) + 5e-8 * np.sin(2 * np.pi * t / 480)
        hour = ((t * step) // 3_600_000) % 24
        session_mult = 1.0 + 0.6 * ((hour >= 7) & (hour < 17)) + 0.3 * ((hour >= 12) & (hour < 16))

        # --- fat-tailed innovations with slight mean reversion ---------------
        nu = 4.0
        innov = rng.standard_t(nu, size=n_bars) / np.sqrt(nu / (nu - 2))
        rets = trend + vol * session_mult * innov
        rets[1:] -= 0.02 * rets[:-1]  # weak anti-persistence (scalp-friendly)

        close = start_price * np.exp(np.cumsum(rets))
        # build OHLC consistent with the close path
        noise = np.abs(rng.normal(0, 1, size=n_bars))
        open_ = np.concatenate([[start_price], close[:-1]])
        hi_ext = noise * vol * session_mult * 0.9
        lo_ext = np.abs(rng.normal(0, 1, size=n_bars)) * vol * session_mult * 0.9
        high = np.maximum(open_, close) + hi_ext
        low = np.minimum(open_, close) - lo_ext
        volume = (1.0 + 60 * session_mult * vol / vol.mean()) * rng.lognormal(0, 0.35, n_bars)

        spread_bps = 0.6 + 35 * vol  # spread widens with volatility
        half = close * spread_bps / 1e4 / 2

        t0 = int(start_ms if start_ms is not None else now_ms() - n_bars * step)
        times = (t0 // step) * step + t * step
        return pd.DataFrame(
            {
                "time": times.astype(np.int64),
                "open": open_.round(5),
                "high": high.round(5),
                "low": low.round(5),
                "close": close.round(5),
                "volume": volume.round(2),
                "bid": (close - half).round(5),
                "ask": (close + half).round(5),
                "bid_size": (rng.lognormal(0, 0.5, n_bars) * 100).round(0),
                "ask_size": (rng.lognormal(0, 0.5, n_bars) * 100).round(0),
            }
        )

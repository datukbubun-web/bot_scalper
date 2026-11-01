"""Synthetic L2 Order Book reconstruction & live quote/depth recorder.

Why this module exists
----------------------
PrimeXBT Trading API v3 does not expose a public WebSocket Level-2 order-book
stream, and its ``GET /bars`` historical endpoint returns OHLCV without
``bid`` / ``ask`` columns. However, PrimeXBT *does* expose two authoritative
real-time liquidity endpoints:

1. ``GET /quotes`` -- live top-of-book ``bid``, ``ask`` and ``mid``.
2. ``GET /orders/market/impact`` -- exact VWAP execution price across PrimeXBT's
   internal matching engine / liquidity-provider book for any requested order
   size ``Q`` on either ``BUY`` (ask book) or ``SELL`` (bid book).

Inverting the cumulative VWAP ladder ``Q_1 < Q_2 < ... < Q_K``::

    p_marginal(k) = (Q_k * VWAP(Q_k) - Q_{k-1} * VWAP(Q_{k-1})) / (Q_k - Q_{k-1})

reconstructs the venue's true marginal L2 price slices, depth-weighted book
asymmetry, and Kyle price-impact slope (bps per unit size) before sending a
single order. Meanwhile :class:`QuoteBookRecorder` persists live quotes/depth
to disk and causally aggregates them onto OHLCV bars so live collection builds
a genuine historical bid/ask + L2 dataset over time.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .models import Quote
from .rest import ApiError, PrimeXBTClient

log = logging.getLogger(__name__)

__all__ = [
    "OrderBookLevel",
    "SyntheticOrderBookSnapshot",
    "probe_synthetic_orderbook",
    "invert_vwap_ladder",
    "QuoteBookRecorder",
    "aggregate_quotes_to_bars",
]


@dataclass(slots=True)
class OrderBookLevel:
    """One reconstructed slice of the venue's internal L2 order book."""

    marginal_price: float
    slice_qty: float
    cumulative_qty: float
    vwap_price: float
    offset_bps: float  # signed distance from mid in bps (+ for asks, - for bids)


@dataclass
class SyntheticOrderBookSnapshot:
    """Reconstructed bidirectional L2 snapshot from ``/quotes`` + ``/impact``."""

    symbol: str
    timestamp_ms: int
    bid: float
    ask: float
    mid: float
    spread_bps: float
    bids: list[OrderBookLevel] = field(default_factory=list)
    asks: list[OrderBookLevel] = field(default_factory=list)
    l2_imbalance: float = 0.0
    bid_slope_bps: float = 0.0
    ask_slope_bps: float = 0.0

    def estimate_slippage_bps(self, side: str, qty: float) -> float:
        """Interpolate expected VWAP slippage (above half-spread) for ``qty``."""
        if qty <= 0 or self.mid <= 0:
            return 0.0
        ladder = self.asks if side.upper() in ("BUY", "LONG") else self.bids
        if not ladder:
            return 0.0
        top = self.ask if side.upper() in ("BUY", "LONG") else self.bid
        for prev, curr in zip([None, *ladder[:-1]], ladder):
            if qty <= curr.cumulative_qty:
                if prev is None or curr.cumulative_qty <= prev.cumulative_qty:
                    vwap = curr.vwap_price
                else:
                    w = (qty - prev.cumulative_qty) / (curr.cumulative_qty - prev.cumulative_qty)
                    vwap = prev.vwap_price + w * (curr.vwap_price - prev.vwap_price)
                return abs(vwap - top) / self.mid * 1e4
        # Extrapolate linearly using measured book slope beyond deepest rung
        last = ladder[-1]
        extra = qty - last.cumulative_qty
        slope = self.ask_slope_bps if side.upper() in ("BUY", "LONG") else self.bid_slope_bps
        base_slip = abs(last.vwap_price - top) / self.mid * 1e4
        return float(max(0.0, base_slip + max(slope, 0.0) * extra))

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp_ms": self.timestamp_ms,
            "bid": self.bid,
            "ask": self.ask,
            "mid": self.mid,
            "spread_bps": self.spread_bps,
            "l2_imbalance": self.l2_imbalance,
            "bid_slope_bps": self.bid_slope_bps,
            "ask_slope_bps": self.ask_slope_bps,
            "bids": [asdict(b) for b in self.bids],
            "asks": [asdict(a) for a in self.asks],
        }


def invert_vwap_ladder(
    rungs: Sequence[tuple[float, float]],
    mid: float,
    side: str,
) -> list[OrderBookLevel]:
    """Invert ``(cumulative_qty, vwap_price)`` pairs into marginal L2 slices.

    Enforces monotonicity along the book (asks non-decreasing, bids
    non-increasing) so micro-jitter between sequential REST calls cannot invent
    crossed synthetic slices.
    """
    if mid <= 0 or not rungs:
        return []
    is_buy = side.upper() in ("BUY", "LONG", "ASK")
    ordered = sorted((float(q), float(p)) for q, p in rungs if q > 0 and p > 0)
    levels: list[OrderBookLevel] = []
    prev_q = 0.0
    prev_notional = 0.0
    prev_marg = ordered[0][1] if ordered else mid

    for cum_q, vwap in ordered:
        dq = cum_q - prev_q
        if dq <= 1e-12:
            continue
        notional = cum_q * vwap
        marg_px = (notional - prev_notional) / dq
        if is_buy:
            marg_px = max(marg_px, prev_marg)
        else:
            marg_px = min(marg_px, prev_marg)
        offset_bps = (marg_px - mid) / mid * 1e4
        levels.append(
            OrderBookLevel(
                marginal_price=float(marg_px),
                slice_qty=float(dq),
                cumulative_qty=float(cum_q),
                vwap_price=float(vwap),
                offset_bps=float(offset_bps),
            )
        )
        prev_q = cum_q
        prev_notional = notional
        prev_marg = marg_px
    return levels


def _compute_book_metrics(
    bids: list[OrderBookLevel],
    asks: list[OrderBookLevel],
    mid: float,
) -> tuple[float, float, float]:
    """Compute depth-weighted L2 imbalance in [-1, 1] and bid/ask slopes (bps/unit)."""
    if not bids or not asks or mid <= 0:
        return 0.0, 0.0, 0.0

    # Liquidity per bps of price displacement: tighter/deeper side has smaller
    # |offset_bps| for the same cumulative quantity -> stronger book support.
    bid_cost = sum(abs(lvl.offset_bps) * lvl.slice_qty for lvl in bids) / max(bids[-1].cumulative_qty, 1e-12)
    ask_cost = sum(abs(lvl.offset_bps) * lvl.slice_qty for lvl in asks) / max(asks[-1].cumulative_qty, 1e-12)
    # If ask_cost > bid_cost, asks are thinner (easier to push UP, or bids are thicker):
    # positive imbalance = stronger bid support relative to ask resistance.
    denom = bid_cost + ask_cost
    imbalance = float((ask_cost - bid_cost) / denom) if denom > 1e-12 else 0.0

    def _slope(levels: list[OrderBookLevel]) -> float:
        if len(levels) < 2:
            return 0.0
        dq = levels[-1].cumulative_qty - levels[0].cumulative_qty
        if dq <= 1e-12:
            return 0.0
        d_bps = abs(levels[-1].vwap_price - levels[0].vwap_price) / mid * 1e4
        return float(max(0.0, d_bps / dq))

    return imbalance, _slope(bids), _slope(asks)


def probe_synthetic_orderbook(
    client: PrimeXBTClient,
    symbol: str,
    size_ladder: Sequence[float] = (0.01, 0.1, 1.0, 10.0, 50.0),
    quote: Quote | None = None,
) -> SyntheticOrderBookSnapshot | None:
    """Reconstruct a live bidirectional L2 snapshot via ``/quotes`` + ``/impact``."""
    ts_ms = int(time.time() * 1000)
    if quote is None:
        quotes = client.quotes([symbol])
        if not quotes:
            return None
        quote = quotes[0]
    mid = quote.price or (quote.bid + quote.ask) / 2.0
    if mid <= 0:
        return None
    spread_bps = max(quote.ask - quote.bid, 0.0) / mid * 1e4

    buy_rungs: list[tuple[float, float]] = []
    sell_rungs: list[tuple[float, float]] = []
    for qty in size_ladder:
        if qty <= 0:
            continue
        try:
            c_buy = client.market_order_cost(symbol, "BUY", float(qty))
            if c_buy.ok and c_buy.price > 0:
                buy_rungs.append((float(qty), float(c_buy.price)))
        except ApiError as exc:
            log.debug("impact BUY %s qty=%s skipped: %s", symbol, qty, exc)
        try:
            c_sell = client.market_order_cost(symbol, "SELL", float(qty))
            if c_sell.ok and c_sell.price > 0:
                sell_rungs.append((float(qty), float(c_sell.price)))
        except ApiError as exc:
            log.debug("impact SELL %s qty=%s skipped: %s", symbol, qty, exc)

    asks = invert_vwap_ladder(buy_rungs, mid=mid, side="BUY")
    bids = invert_vwap_ladder(sell_rungs, mid=mid, side="SELL")
    imbalance, bid_slope, ask_slope = _compute_book_metrics(bids, asks, mid)

    return SyntheticOrderBookSnapshot(
        symbol=symbol,
        timestamp_ms=quote.timestamp_ms or ts_ms,
        bid=float(quote.bid),
        ask=float(quote.ask),
        mid=float(mid),
        spread_bps=float(spread_bps),
        bids=bids,
        asks=asks,
        l2_imbalance=imbalance,
        bid_slope_bps=bid_slope,
        ask_slope_bps=ask_slope,
    )


class QuoteBookRecorder:
    """Append-only recorder for live top-of-book quotes and synthetic L2 probes."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, symbol: str) -> Path:
        safe = symbol.replace("/", "")
        return self.out_dir / f"{safe}_quotes.jsonl"

    def record_quote(
        self,
        quote: Quote,
        book: SyntheticOrderBookSnapshot | None = None,
    ) -> None:
        ts = quote.timestamp_ms or int(time.time() * 1000)
        mid = quote.price or (quote.bid + quote.ask) / 2.0
        # Derive effective bid_size / ask_size weights from L2 imbalance so
        # FeatureEngineer's standard quote_imbalance & microprice formulas work
        # directly on recorded snapshots.
        imb = book.l2_imbalance if book is not None else 0.0
        bid_size = float(max(1.0 + imb, 1e-4))
        ask_size = float(max(1.0 - imb, 1e-4))
        row = {
            "time": int(ts),
            "symbol": quote.symbol,
            "bid": float(quote.bid),
            "ask": float(quote.ask),
            "mid": float(mid),
            "spread_bps": float(max(quote.ask - quote.bid, 0.0) / max(mid, 1e-12) * 1e4),
            "bid_size": bid_size,
            "ask_size": ask_size,
            "l2_imbalance": float(imb),
            "bid_slope_bps": float(book.bid_slope_bps) if book is not None else 0.0,
            "ask_slope_bps": float(book.ask_slope_bps) if book is not None else 0.0,
        }
        with self._path_for(quote.symbol).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    def load_quotes(self, symbol: str) -> pd.DataFrame:
        p = self._path_for(symbol)
        if not p.exists():
            return pd.DataFrame(
                columns=["time", "bid", "ask", "bid_size", "ask_size", "l2_imbalance"]
            )
        rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            return pd.DataFrame(
                columns=["time", "bid", "ask", "bid_size", "ask_size", "l2_imbalance"]
            )
        return pd.DataFrame(rows).sort_values("time").reset_index(drop=True)


def aggregate_quotes_to_bars(
    bars: pd.DataFrame,
    quotes: pd.DataFrame,
    bar_ms: int = 60_000,
) -> pd.DataFrame:
    """Causally join recorded ``quotes`` onto OHLCV ``bars`` without look-ahead.

    A bar stamped with open time ``T`` closes at ``T + bar_ms``. Only quotes
    with ``timestamp <= T + bar_ms`` (the bar's close instant) may contribute to
    bar ``T``; later quotes belong to future bars and are strictly excluded.
    """
    out = bars.copy().sort_values("time").reset_index(drop=True)
    if quotes is None or quotes.empty:
        return out
    q = quotes.sort_values("time").copy()
    q_cols = [c for c in ("bid", "ask", "bid_size", "ask_size", "l2_imbalance") if c in q.columns]
    if not {"bid", "ask"}.issubset(q_cols):
        return out

    bar_close = pd.DataFrame({"bar_idx": out.index, "close_time": out["time"].astype(np.int64) + int(bar_ms)})
    merged = pd.merge_asof(
        bar_close,
        q[["time", *q_cols]].rename(columns={"time": "quote_time"}).astype({"quote_time": np.int64}),
        left_on="close_time",
        right_on="quote_time",
        direction="backward",
    )
    # Reject stale quotes recorded strictly before the bar even opened
    stale = merged["quote_time"] < out["time"].to_numpy(dtype=np.int64)
    for col in q_cols:
        vals = merged[col].to_numpy(dtype=np.float64, copy=True)
        vals[stale.to_numpy()] = np.nan
        out[col] = vals
    return out

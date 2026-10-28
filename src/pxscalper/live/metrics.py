"""Prometheus metrics + a graceful no-op fallback.

A scalping bot that cannot answer "how long did the last decision take, how wide
was the spread, how many orders were rejected?" is not deployable. These are the
ten metrics an on-call operator actually looks at.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

try:  # pragma: no cover - optional dependency
    from prometheus_client import Counter, Gauge, Histogram, start_http_server

    _AVAILABLE = True
except Exception:  # noqa: BLE001
    _AVAILABLE = False


class _Null:
    def labels(self, *_, **__):
        return self

    def inc(self, *_, **__):
        return None

    def observe(self, *_, **__):
        return None

    def set(self, *_, **__):
        return None


class Metrics:
    """Thin façade so the engine code never imports prometheus directly."""

    def __init__(self, port: int = 9101, enable_server: bool = False) -> None:
        self.enabled = _AVAILABLE
        if not self.enabled:
            self._noop = _Null()
            log.warning("prometheus_client not installed; metrics disabled")
            return

        self.equity = Gauge("pxs_equity", "Account equity")
        self.upl = Gauge("pxs_upl", "Unrealised PnL")
        self.spread = Gauge("pxs_spread_bps", "Current spread", ["symbol"])
        self.fee = Gauge("pxs_taker_fee_pct", "Taker fee", ["symbol"])
        self.inference_us = Gauge("pxs_inference_us", "Last inference latency (us)")
        self.cycle_ms = Histogram(
            "pxs_cycle_ms", "Decision cycle time (ms)",
            buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500),
        )
        self.trades = Counter("pxs_trades_total", "Orders opened", ["symbol"])
        self.rejects = Counter("pxs_order_rejects_total", "Orders refused", ["symbol"])
        self.closed = Counter("pxs_positions_closed_total", "Positions closed")
        self.pnl = Histogram(
            "pxs_trade_pnl", "Realised PnL per trade",
            buckets=(-100, -50, -20, -10, -5, -1, 0, 1, 5, 10, 20, 50, 100),
        )
        self.decisions = Counter(
            "pxs_decisions_total", "Decisions by outcome", ["outcome"]
        )
        if enable_server:
            start_http_server(port)
            log.info("prometheus metrics on :%d", port)

    # ------------------------------------------------------------------ setters
    def set_equity(self, value: float) -> None:
        if self.enabled:
            self.equity.set(value)

    def set_upl(self, value: float) -> None:
        if self.enabled:
            self.upl.set(value)

    def set_spread(self, symbol: str, bps: float) -> None:
        if self.enabled:
            self.spread.labels(symbol).set(bps)

    def set_fee(self, symbol: str, pct: float) -> None:
        if self.enabled:
            self.fee.labels(symbol).set(pct)

    def set_inference_latency(self, us: float) -> None:
        if self.enabled:
            self.inference_us.set(us)

    def observe_cycle(self, ms: float) -> None:
        if self.enabled:
            self.cycle_ms.observe(ms)

    def inc_trades(self, symbol: str) -> None:
        if self.enabled:
            self.trades.labels(symbol).inc()

    def inc_rejects(self, symbol: str) -> None:
        if self.enabled:
            self.rejects.labels(symbol).inc()

    def inc_closed(self, pnl: float) -> None:
        if self.enabled:
            self.closed.inc()
            self.pnl.observe(pnl)

    def inc_decision(self, outcome: str) -> None:
        if self.enabled:
            self.decisions.labels(outcome).inc()

"""Async-free, low-latency REST client for the PrimeXBT Trading API v3.

Design notes
------------
* **One session, keep-alive, no redirects.** A fresh TLS handshake per tick is
  the single biggest avoidable latency in a scalping loop; ``requests.Session``
  with an ``HTTPAdapter`` pool removes it.
* **Re-sign on retry.** A signature is valid for ``recvWindowMs`` (5 s) and is
  the whole replay protection. Writes are therefore never blindly retried: see
  :meth:`PrimeXBTClient.place_market` and the reconciler in ``live/reconcile.py``.
* **Impact before action.** Every placement has an ``impact`` twin that costs the
  same order without sending it. The execution layer always costs first -- that
  is how we detect ``INSUFFICIENT_FUNDS`` / ``MARGIN_CALL_RISK`` before money
  moves.
* **No websocket exists (yet) on v3**, so the live loop polls ``quotes`` /
  ``positions`` / ``orders/active`` at a bounded rate and never faster than it acts.

Endpoints used
--------------
GET  /v3/trading/time                     (unsigned)
GET  /v3/trading/bars                     (unsigned)
GET  /v3/trading/metrics                  read
POST /v3/trading/markets                  read
POST /v3/trading/quotes                   read
POST /v3/trading/settings                 read
GET  /v3/trading/positions                read
GET  /v3/trading/orders/active            read
POST /v3/trading/orders/market/impact     read
POST /v3/trading/orders/market/place      trade
POST /v3/trading/orders/limit/impact      read
POST /v3/trading/orders/limit/place       trade
POST /v3/trading/orders/cancel            trade
POST /v3/trading/orders/cancel-all        trade
POST /v3/trading/positions/id/close       trade
POST /v3/trading/positions/id/protective-order/set     trade
POST /v3/trading/positions/protective-order/set        trade
POST /v3/trading/margins                   read
POST /v3/trading/margins/change            trade
POST /v3/trading/report/fills             read
POST /v3/trading/report/orders            read
"""

from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import quote, urlencode

import requests
from requests.adapters import HTTPAdapter

from .models import (
    AccountMetrics,
    Bar,
    Fill,
    Market,
    Order,
    OrderCost,
    Position,
    Quote,
    SymbolSettings,
)
from .signing import ApiCredentials, sign_request

log = logging.getLogger(__name__)

BASE_URL = "https://api.primexbt.com"
TRADING_PREFIX = "/v3/trading"

__all__ = ["PrimeXBTClient", "ApiError", "RateLimiter", "RateLimitError"]


class ApiError(RuntimeError):
    """A non-2xx answer from PrimeXBT, carrying the documented error envelope."""

    def __init__(self, status: int, code: str, description: str, rid: str | None = None):
        super().__init__(f"[{status} {code}] {description}" + (f" (rid={rid})" if rid else ""))
        self.status = status
        self.code = code
        self.description = description
        self.rid = rid

    @property
    def is_auth(self) -> bool:
        return self.status == 401

    @property
    def is_permission(self) -> bool:
        return self.status == 403

    @property
    def is_engine_refusal(self) -> bool:
        """A trading-engine refusal: INSUFFICIENT_FUNDS, TOO_LOW_AMOUNT, ..."""
        return self.status == 400

    @property
    def retryable(self) -> bool:
        # 503 = server still loading keys; 429 = we were too eager.
        return self.status in (429, 502, 503, 504)


class RateLimitError(RuntimeError):
    pass


@dataclass
class RateLimiter:
    """Token bucket. PrimeXBT publishes no per-route weights for v3, so the bot
    budgets itself: a scalper needs quotes + one position poll per second and
    must never become the reason the account gets throttled."""

    rate_per_sec: float = 12.0
    burst: int = 24
    _tokens: float = 0.0
    _last: float = 0.0
    _lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self._tokens = float(self.burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(self.burst, self._tokens + (now - self._last) * self.rate_per_sec)
                self._last = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                deficit = (tokens - self._tokens) / self.rate_per_sec
                time.sleep(min(deficit, 0.25))


class PrimeXBTClient:
    """Thin, explicit wrapper around the endpoints the scalper needs."""

    def __init__(
        self,
        credentials: ApiCredentials | None = None,
        read_credentials: ApiCredentials | None = None,
        base_url: str = BASE_URL,
        timeout: tuple[float, float] = (3.0, 10.0),
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 3,
        pool_size: int = 16,
        recv_window_ms: int | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.creds = credentials
        self.read_creds = read_credentials or credentials
        self.timeout = timeout
        self.rate_limiter = rate_limiter or RateLimiter()
        self.max_retries = max_retries
        self.recv_window_ms = recv_window_ms
        self._clock_offset_ms = 0.0

        self._session = requests.Session()
        adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=0)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        # A signature is bound to the exact bytes sent -> never let a redirect
        # rewrite our request.
        self._session.max_redirects = 0

    # ------------------------------------------------------------------ plumbing
    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "PrimeXBTClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _url(self, path: str, params: dict[str, Any] | None = None) -> tuple[str, str, str]:
        """Return (full_url, path, query_string) with the query exactly as sent."""
        if params:
            items: list[tuple[str, str]] = []
            for key, value in params.items():
                if value is None:
                    continue
                items.append((key, str(value)))
            query = urlencode(items, quote_via=quote, safe="/")
        else:
            query = ""
        full = f"{self.base_url}{path}" + (f"?{query}" if query else "")
        return full, path, query

    def _raw(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        signed: bool = True,
        read_only: bool = False,
        retries: int | None = None,
    ) -> Any:
        url, p, query = self._url(path, params)
        payload = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
        attempts = (retries if retries is not None else self.max_retries) + 1
        last_exc: Exception | None = None

        for attempt in range(attempts):
            self.rate_limiter.acquire()
            headers: dict[str, str] = {"Accept": "application/json"}
            if signed:
                creds = self.read_creds if read_only else self.creds
                if creds is None:
                    raise ApiError(401, "NO_CREDENTIALS", f"{method} {path} needs an API key")
                headers = sign_request(
                    creds,
                    method,
                    p,
                    query,
                    payload,
                    timestamp_ms=int(time.time() * 1000 + self._clock_offset_ms),
                    recv_window_ms=self.recv_window_ms,
                )
            elif payload:
                headers["Content-Type"] = "application/json"

            try:
                resp = self._session.request(
                    method, url, data=payload, headers=headers, timeout=self.timeout
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    backoff = min(2**attempt * 0.15, 2.0) * (0.5 + random.random())
                    log.warning("%s %s network error (%s); retry in %.2fs", method, path, exc, backoff)
                    time.sleep(backoff)
                    continue
                raise

            if resp.status_code == 200:
                if not resp.content:
                    return None
                return resp.json()

            # ---- documented error envelope: {code, description, rid}
            try:
                err = resp.json()
            except ValueError:
                err = {"code": f"HTTP_{resp.status_code}", "description": resp.text[:300]}
            api_err = ApiError(
                resp.status_code,
                str(err.get("code", "UNKNOWN")),
                str(err.get("description", "")),
                err.get("rid"),
            )
            if resp.status_code == 401:
                # almost always clock drift or a stale key; self-heal the clock once
                if attempt == 0:
                    try:
                        self.sync_clock()
                    except Exception:  # noqa: BLE001
                        pass
            if api_err.retryable and attempt < attempts - 1:
                last_exc = api_err
                backoff = min(2**attempt * 0.25, 3.0) * (0.5 + random.random())
                log.warning("%s %s -> %s; retry in %.2fs", method, path, api_err, backoff)
                time.sleep(backoff)
                continue
            raise api_err

        raise last_exc if last_exc else RuntimeError("unreachable")

    # ---------------------------------------------------------------- unsigned
    def server_time(self) -> dict[str, Any]:
        return self._raw("GET", f"{TRADING_PREFIX}/time", signed=False, retries=2)

    def sync_clock(self) -> float:
        """Estimate the offset between local clock and PrimeXBT's.

        A signature older than 5 s (or more than 1 s in the future) is refused
        with a deliberately opaque 401, so the bot keeps this offset fresh.
        """
        t0 = time.time() * 1000
        answer = self.server_time()
        t1 = time.time() * 1000
        rtt = t1 - t0
        server = float(answer["serverTime"])
        self._clock_offset_ms = server + rtt / 2 - t1
        if abs(self._clock_offset_ms) > 500:
            log.warning("clock offset vs PrimeXBT is %.0f ms", self._clock_offset_ms)
        return self._clock_offset_ms

    @property
    def clock_offset_ms(self) -> float:
        return self._clock_offset_ms

    def bars(
        self,
        symbol: str,
        bar_size: str = "1M",
        limit: int = 500,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[Bar]:
        """Price history (feed bid). ``1M`` is one minute, ``1MO`` one month."""
        params: dict[str, Any] = {"symbol": symbol, "barSize": bar_size, "limit": min(limit, 500)}
        if start_ms is not None:
            params["from"] = int(start_ms)
        if end_ms is not None:
            params["to"] = int(end_ms)
        answer = self._raw("GET", f"{TRADING_PREFIX}/bars", params=params, signed=False)
        return [Bar.from_api(b) for b in answer.get("bars", [])]

    def bars_range(
        self,
        symbol: str,
        bar_size: str,
        start_ms: int,
        end_ms: int,
        page: int = 500,
    ) -> list[Bar]:
        """Page forward through history: the API answers the first ``limit`` bars from ``from``."""
        out: list[Bar] = []
        cursor = start_ms
        while cursor < end_ms:
            chunk = self.bars(symbol, bar_size, limit=page, start_ms=cursor, end_ms=end_ms)
            if not chunk:
                break
            out.extend(chunk)
            next_cursor = chunk[-1].time + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor
            if len(chunk) < page:
                break
        return [b for b in out if b.time <= end_ms]

    # ------------------------------------------------------------------- account
    def metrics(self) -> AccountMetrics:
        return AccountMetrics.from_api(self._raw("GET", f"{TRADING_PREFIX}/metrics", read_only=True))

    def positions(self) -> list[Position]:
        answer = self._raw("GET", f"{TRADING_PREFIX}/positions", read_only=True)
        return [Position.from_api(p) for p in answer.get("positions", [])]

    def active_orders(self) -> list[Order]:
        answer = self._raw("GET", f"{TRADING_PREFIX}/orders/active", read_only=True)
        return [Order.from_api(o) for o in answer.get("orders", [])]

    def margins(self, symbols: Iterable[str] | None = None) -> list[dict[str, Any]]:
        body = {"symbols": list(symbols)} if symbols else {}
        return self._raw("POST", f"{TRADING_PREFIX}/margins", body=body, read_only=True)["margins"]

    def fills(self, symbol: str | None = None, limit: int = 100, offset: int = 0) -> list[Fill]:
        body: dict[str, Any] = {"limit": min(limit, 1000), "offset": offset}
        if symbol:
            body["symbol"] = symbol
        answer = self._raw("POST", f"{TRADING_PREFIX}/report/fills", body=body, read_only=True)
        return [Fill.from_api(f) for f in answer.get("fills", [])]

    # --------------------------------------------------------------- market data
    def markets(
        self,
        symbols: Iterable[str] | None = None,
        market_type: str | None = None,
        category: str | None = None,
    ) -> list[Market]:
        body: dict[str, Any] = {}
        if symbols:
            body["symbols"] = list(symbols)
        if market_type:
            body["marketType"] = market_type
        if category:
            body["category"] = category
        answer = self._raw("POST", f"{TRADING_PREFIX}/markets", body=body, read_only=True)
        return [Market.from_api(m) for m in answer.get("markets", [])]

    def quotes(self, symbols: Iterable[str]) -> list[Quote]:
        ts = int(time.time() * 1000 + self._clock_offset_ms)
        answer = self._raw(
            "POST", f"{TRADING_PREFIX}/quotes", body={"symbols": list(symbols)}, read_only=True
        )
        return [Quote.from_api(q, ts) for q in answer.get("quotes", [])]

    def settings(self, market_type: str | None = None, symbols: Iterable[str] | None = None):
        body: dict[str, Any] = {}
        if symbols:
            body["symbols"] = list(symbols)
        if market_type:
            body["marketType"] = market_type
        answer = self._raw("POST", f"{TRADING_PREFIX}/settings", body=body, read_only=True)
        return [SymbolSettings.from_api(s) for s in answer.get("symbols", [])]

    # -------------------------------------------------------------------- orders
    def market_order_cost(
        self, symbol: str, side: str, qty: float, take_profit: float | None = None,
        stop_loss: float | None = None,
    ) -> OrderCost:
        body: dict[str, Any] = {"symbol": symbol, "side": side.upper(), "qty": qty}
        if take_profit is not None:
            body["takeProfit"] = take_profit
        if stop_loss is not None:
            body["stopLoss"] = stop_loss
        return OrderCost.from_api(
            self._raw("POST", f"{TRADING_PREFIX}/orders/market/impact", body=body, read_only=True)
        )

    def place_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        """Send a market order. **Not idempotent**: caller must reconcile on timeout."""
        body: dict[str, Any] = {"symbol": symbol, "side": side.upper(), "qty": qty}
        if take_profit is not None:
            body["takeProfit"] = take_profit
        if stop_loss is not None:
            body["stopLoss"] = stop_loss
        if client_order_id:
            body["clientOrderId"] = client_order_id
        return self._raw(
            "POST", f"{TRADING_PREFIX}/orders/market/place", body=body, retries=0
        )

    def place_limit(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        take_profit: float | None = None,
        stop_loss: float | None = None,
        client_order_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "symbol": symbol, "side": side.upper(), "qty": qty, "price": price
        }
        if take_profit is not None:
            body["takeProfit"] = take_profit
        if stop_loss is not None:
            body["stopLoss"] = stop_loss
        if client_order_id:
            body["clientOrderId"] = client_order_id
        return self._raw("POST", f"{TRADING_PREFIX}/orders/limit/place", body=body, retries=0)

    def cancel_order(self, order_id: int) -> dict[str, Any]:
        return self._raw(
            "POST", f"{TRADING_PREFIX}/orders/cancel", body={"orderId": int(order_id)}
        )

    def cancel_all(self, symbol: str | None = None, including_protective: bool = False):
        body: dict[str, Any] = {"includingProtective": bool(including_protective)}
        if symbol:
            body["symbol"] = symbol
        return self._raw("POST", f"{TRADING_PREFIX}/orders/cancel-all", body=body)

    def close_position(self, position_id: int, qty: float | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"positionId": int(position_id)}
        if qty is not None:
            body["qty"] = qty
        return self._raw("POST", f"{TRADING_PREFIX}/positions/id/close", body=body, retries=0)

    def close_all_positions(self) -> dict[str, Any]:
        return self._raw("POST", f"{TRADING_PREFIX}/positions/close/all", body={}, retries=0)

    def set_protective(
        self, position_id: int, kind: str, price: float
    ) -> dict[str, Any]:
        """``kind`` is ``STOP_LOSS`` or ``TAKE_PROFIT``."""
        return self._raw(
            "POST",
            f"{TRADING_PREFIX}/positions/id/protective-order/set",
            body={"positionId": int(position_id), "type": kind.upper(), "price": float(price)},
        )

    def set_symbol_protective(self, symbol: str, kind: str, price: float) -> dict[str, Any]:
        return self._raw(
            "POST",
            f"{TRADING_PREFIX}/positions/protective-order/set",
            body={"symbol": symbol, "type": kind.upper(), "price": float(price)},
        )

    def set_leverage(self, symbol: str, leverage: int, margin_type: str = "CROSS"):
        return self._raw(
            "POST",
            f"{TRADING_PREFIX}/margins/change",
            body={"symbol": symbol, "marginType": margin_type, "leverage": int(leverage)},
        )

    def change_netting(self, netting: bool) -> dict[str, Any]:
        return self._raw(
            "POST", f"{TRADING_PREFIX}/netting/change", body={"netting": bool(netting)}
        )

    # ------------------------------------------------------------------ helpers
    def snapshot(self, symbols: Iterable[str]) -> dict[str, Any]:
        """One round of everything the live loop needs, in three calls."""
        symbols = list(symbols)
        return {
            "quotes": self.quotes(symbols),
            "positions": self.positions(),
            "orders": self.active_orders(),
            "metrics": self.metrics(),
        }

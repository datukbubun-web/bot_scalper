"""Order execution: impact -> place -> verify -> reconcile.

The rules this module enforces, because each one has bankrupted a bot somewhere:

1. **Cost first.** Every placement is preceded by the matching ``*/impact`` call.
   If the engine answers ``resolutionLevel=ERROR`` (INSUFFICIENT_FUNDS,
   TOO_LOW_AMOUNT, MARGIN_CALL_RISK) we never send the order.
2. **Never blindly retry a write.** PrimeXBT: *"No request is safe to repeat. A
   placed order sent twice is two orders."* On a timeout we reconcile by reading
   ``positions`` / ``orders/active`` and matching our ``clientOrderId``.
3. **Protection travels with the order.** Stop-loss and take-profit are attached
   at placement (``stopLoss`` / ``takeProfit`` fields), not "added later" -- the
   network can die between the two calls.
4. **Idempotency key.** A ``clientOrderId`` derived from (symbol, bar time, intent)
   lets the reconciler recognise our own order after a crash.
5. **Post-trade verification.** Positions are read back; if the expected position
   is absent, the loop treats it as a rejection and raises, it does not proceed
   as if the trade existed.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from ..exchange.models import OrderCost, Position, SymbolSettings
from ..exchange.rest import ApiError, PrimeXBTClient

log = logging.getLogger(__name__)

__all__ = ["ExecutionResult", "ExecutionConfig", "OrderRouter", "make_client_order_id"]


def make_client_order_id(symbol: str, bar_time_ms: int, intent: str, tag: str = "pxs") -> str:
    """Deterministic, <=36 chars, charset [#. _ : -] as the API requires."""
    raw = f"{symbol}|{bar_time_ms}|{intent}".encode()
    digest = hashlib.sha1(raw).hexdigest()[:16]
    base = symbol.replace("/", "")
    return f"{tag}.{base}.{bar_time_ms}.{digest}"[:36]


@dataclass
class ExecutionConfig:
    use_limit_orders: bool = False        # resting limit = maker fee, but may not fill
    limit_offset_bps: float = 0.2         # passive offset inside the spread
    max_spread_bps: float = 3.0
    attach_protection: bool = True
    verify_fills: bool = True
    verify_timeout_s: float = 3.0
    max_impact_vwap_bps: float = 2.5      # refuse orders that walk the book
    dry_run: bool = False
    retry_after_timeout: bool = False     # keep False unless you enjoy double fills


@dataclass
class ExecutionResult:
    ok: bool
    intent: str
    symbol: str
    qty: float
    price: float
    order_id: int | None = None
    client_order_id: str | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    cost: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    latency_ms: float = 0.0
    dry_run: bool = False

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class OrderRouter:
    """Turns a risk-approved intent into an exchange position, or refuses to."""

    def __init__(
        self,
        client: PrimeXBTClient,
        config: ExecutionConfig | None = None,
        settings: dict[str, SymbolSettings] | None = None,
    ) -> None:
        self.client = client
        self.config = config or ExecutionConfig()
        self.settings = settings or {}

    # ------------------------------------------------------------------ public
    def open_position(
        self,
        symbol: str,
        intent: str,
        qty: float,
        stop_loss: float | None,
        take_profit: float | None,
        reference_price: float,
        bar_time_ms: int = 0,
    ) -> ExecutionResult:
        """Market (or passive limit) entry with protection attached."""
        started = time.perf_counter()
        side = "BUY" if intent.upper() in ("LONG", "BUY") else "SELL"
        settings = self.settings.get(symbol)
        qty = settings.round_qty(qty) if settings else qty
        cid = make_client_order_id(symbol, bar_time_ms or int(time.time() * 1000), intent)

        if stop_loss and settings:
            stop_loss = settings.round_price(stop_loss)
        if take_profit and settings:
            take_profit = settings.round_price(take_profit)

        # ---- 1. cost it before sending it --------------------------------
        try:
            cost = self.client.market_order_cost(symbol, side, qty, take_profit, stop_loss)
        except ApiError as exc:
            return self._fail(intent, symbol, qty, reference_price, f"impact failed: {exc}", started)
        if not cost.ok:
            return self._fail(
                intent, symbol, qty, reference_price,
                f"engine refuses order: {cost.resolution}", started, cost,
            )
        if abs(cost.vwap_price_offset / max(reference_price, 1e-12) * 1e4) > self.config.max_impact_vwap_bps:
            return self._fail(
                intent, symbol, qty, reference_price,
                f"book impact {cost.vwap_price_offset:.5f} beyond {self.config.max_impact_vwap_bps}bps",
                started, cost,
            )

        if self.config.dry_run:
            log.info("[dry-run] %s %s %.6f @ ~%.5f sl=%s tp=%s", side, symbol, qty,
                     cost.price, stop_loss, take_profit)
            return ExecutionResult(
                ok=True, intent=intent, symbol=symbol, qty=qty, price=cost.price,
                client_order_id=cid, stop_loss=stop_loss, take_profit=take_profit,
                cost=asdict(cost), latency_ms=(time.perf_counter() - started) * 1000,
                dry_run=True,
            )

        # ---- 2. send ------------------------------------------------------
        try:
            if self.config.use_limit_orders:
                half_spread = max(cost.price * self.config.limit_offset_bps / 1e4, 0.0)
                px = cost.price - half_spread if side == "BUY" else cost.price + half_spread
                px = settings.round_price(px) if settings else px
                answer = self.client.place_limit(
                    symbol, side, qty, px, take_profit, stop_loss, cid
                )
            else:
                answer = self.client.place_market(
                    symbol, side, qty, take_profit, stop_loss, cid
                )
        except ApiError as exc:
            if exc.code in ("POSITION_NOT_FOUND", "ORDER_NOT_FOUND"):
                return self._fail(intent, symbol, qty, reference_price, str(exc), started, cost)
            # network / timeout: the order may or may not exist -> reconcile
            log.error("placement uncertain for %s (%s); reconciling", cid, exc)
            found = self._find_by_client_id(cid)
            if found:
                return ExecutionResult(
                    ok=True, intent=intent, symbol=symbol, qty=qty,
                    price=found.get("executedPrice") or cost.price,
                    order_id=found.get("id"), client_order_id=cid,
                    stop_loss=stop_loss, take_profit=take_profit, cost=asdict(cost),
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            return self._fail(intent, symbol, qty, reference_price, f"placement failed: {exc}", started, cost)
        except Exception as exc:  # noqa: BLE001 -- sockets, TLS, DNS
            log.error("placement ambiguous (%s); reconciling by clientOrderId", exc)
            found = self._find_by_client_id(cid)
            if found:
                return ExecutionResult(
                    ok=True, intent=intent, symbol=symbol, qty=qty,
                    price=found.get("executedPrice") or cost.price,
                    order_id=found.get("id"), client_order_id=cid,
                    stop_loss=stop_loss, take_profit=take_profit, cost=asdict(cost),
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
            return self._fail(intent, symbol, qty, reference_price, f"placement ambiguous: {exc}", started, cost)

        order_id = int(answer.get("orderId", 0)) or None
        result = ExecutionResult(
            ok=True, intent=intent, symbol=symbol, qty=qty, price=cost.price,
            order_id=order_id, client_order_id=cid, stop_loss=stop_loss,
            take_profit=take_profit, cost=asdict(cost),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

        # ---- 3. verify ----------------------------------------------------
        if self.config.verify_fills and self.config.verify_timeout_s > 0:
            pos = self.wait_for_position(symbol, timeout_s=self.config.verify_timeout_s)
            if pos is None:
                result.ok = False
                result.error = "order accepted but no position appeared (unfilled or rejected)"
                log.error("%s", result.error)
            else:
                result.price = pos.open_price
                result.qty = pos.qty
        return result

    def close_position(self, position: Position, reason: str = "") -> ExecutionResult:
        started = time.perf_counter()
        try:
            self.client.close_position(position.id)
        except ApiError as exc:
            return ExecutionResult(
                ok=False, intent="CLOSE", symbol=position.symbol, qty=position.qty,
                price=position.current_price, error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000,
            )
        log.info("closed position %s (%s) pnl %.2f %s", position.id, position.symbol,
                 position.upl, reason)
        return ExecutionResult(
            ok=True, intent="CLOSE", symbol=position.symbol, qty=position.qty,
            price=position.current_price,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def close_all(self) -> None:
        try:
            self.client.close_all_positions()
        except ApiError as exc:
            log.error("close_all failed: %s", exc)

    def tighten_stop(self, position: Position, new_stop: float) -> bool:
        """Trailing-stop update. Replacing protection is a trade-permission route."""
        try:
            self.client.set_protective(position.id, "STOP_LOSS", new_stop)
            return True
        except ApiError as exc:
            log.warning("could not move stop on %s: %s", position.id, exc)
            return False

    def wait_for_position(
        self, symbol: str, timeout_s: float = 3.0, poll_s: float = 0.25
    ) -> Position | None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                for pos in self.client.positions():
                    if pos.symbol == symbol:
                        return pos
            except ApiError as exc:
                log.warning("positions poll failed: %s", exc)
            time.sleep(poll_s)
        return None

    # ----------------------------------------------------------------- private
    def _find_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        """Reconciliation after an ambiguous write: did our order actually land?"""
        try:
            for order in self.client.active_orders():
                if order.client_order_id == client_order_id:
                    return {
                        "id": order.id,
                        "executedPrice": order.executed_price,
                    }
        except ApiError as exc:
            log.warning("reconcile: orders/active unavailable (%s)", exc)
        return None

    @staticmethod
    def _fail(intent, symbol, qty, price, error, started, cost: OrderCost | None = None) -> ExecutionResult:
        log.warning("execution refused: %s", error)
        return ExecutionResult(
            ok=False, intent=intent, symbol=symbol, qty=qty, price=price,
            error=error, cost=asdict(cost) if cost else {},
            latency_ms=(time.perf_counter() - started) * 1000,
        )

"""Execution router behaviour with a fake exchange client.

These tests encode the rules that protect capital: never place an order the
engine would refuse, never double-send after a timeout, attach protection.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pxscalper.exchange.models import OrderCost, Position, SymbolSettings
from pxscalper.exchange.rest import ApiError
from pxscalper.execution.router import ExecutionConfig, OrderRouter


def _cost(level: str = "INFO", resolution: str = "OK", price: float = 1.1000) -> OrderCost:
    return OrderCost(
        currency="USD", price=price, order_cost=110.0, margin_impact=22.0, fee=0.055,
        fee_pct=0.05, leverage=5.0, liq_price=0.9, min_order_size=0.001,
        resolution=resolution, resolution_level=level, vwap_price_offset=0.0,
    )


class FakeClient:
    def __init__(self, cost: OrderCost | None = None, place_error: Exception | None = None):
        self._cost = cost or _cost()
        self._place_error = place_error
        self.placed: list[dict] = []
        self.closed: list[int] = []
        self._positions: list[Position] = []
        self.cancelled: list[int] = []

    def market_order_cost(self, symbol, side, qty, take_profit=None, stop_loss=None):
        return self._cost

    def place_market(self, symbol, side, qty, take_profit=None, stop_loss=None, client_order_id=None):
        if self._place_error:
            raise self._place_error
        self.placed.append(
            dict(symbol=symbol, side=side, qty=qty, tp=take_profit, sl=stop_loss, cid=client_order_id)
        )
        self._positions.append(
            Position(
                id=12345, symbol=symbol, side=side, qty=qty, open_price=self._cost.price,
                current_price=self._cost.price, mark_price=self._cost.price, upl=0.0,
                roe_pct=0.0, liq_price=0.9, margin=22.0, client_leverage=5.0,
                stop_loss_price=stop_loss, take_profit_price=take_profit, open_time=0,
            )
        )
        return {"orderId": 999}

    def place_limit(self, symbol, side, qty, price, take_profit=None, stop_loss=None, client_order_id=None):
        return self.place_market(symbol, side, qty, take_profit, stop_loss, client_order_id)

    def positions(self):
        return self._positions

    def active_orders(self):
        return []

    def close_position(self, position_id, qty=None):
        self.closed.append(position_id)
        self._positions = [p for p in self._positions if p.id != position_id]
        return {"accepted": True}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)
        return {"accepted": True}


SETTINGS = {
    "EUR/USD": SymbolSettings(
        symbol="EUR/USD", min_qty=0.001, max_qty=1000, qty_step=0.001, min_price_step=0.00001,
        price_scale=5, max_leverage=100, default_leverage=20, taker_fee_pct=0.05,
        maker_fee_pct=0.02,
    )
}


def test_places_order_with_protection():
    client = FakeClient()
    router = OrderRouter(client, ExecutionConfig(verify_fills=True, verify_timeout_s=0.2), SETTINGS)
    result = router.open_position("EUR/USD", "LONG", 0.01, 1.095, 1.110, 1.10, bar_time_ms=1)
    assert result.ok
    assert result.order_id == 999
    assert len(client.placed) == 1
    assert client.placed[0]["sl"] == 1.095
    assert client.placed[0]["tp"] == 1.11
    assert client.placed[0]["cid"].startswith("pxs.")


def test_refuses_when_engine_says_error():
    client = FakeClient(cost=_cost("ERROR", "INSUFFICIENT_FUNDS"))
    router = OrderRouter(client, ExecutionConfig(verify_fills=False), SETTINGS)
    result = router.open_position("EUR/USD", "LONG", 0.01, 1.095, None, 1.10)
    assert not result.ok
    assert "INSUFFICIENT_FUNDS" in (result.error or "")
    assert client.placed == []


def test_refuses_excessive_book_impact():
    client = FakeClient(cost=OrderCost(
        currency="USD", price=1.10, order_cost=110, margin_impact=1, fee=0.05, fee_pct=0.05,
        leverage=5, liq_price=None, min_order_size=0.001, resolution="OK",
        resolution_level="INFO", vwap_price_offset=0.01,
    ))
    router = OrderRouter(client, ExecutionConfig(verify_fills=False, max_impact_vwap_bps=1.0), SETTINGS)
    result = router.open_position("EUR/USD", "BUY", 0.01, None, None, 1.10)
    assert not result.ok
    assert client.placed == []


def test_timeout_is_reconciled_not_duplicated():
    """A network timeout must not produce a second order."""
    client = FakeClient(place_error=TimeoutError("socket timeout"))
    router = OrderRouter(client, ExecutionConfig(verify_fills=False), SETTINGS)
    result = router.open_position("EUR/USD", "LONG", 0.01, 1.095, 1.11, 1.10)
    assert not result.ok
    assert len(client.placed) == 0  # no retry attempted


def test_api_error_insufficient_funds_is_reported():
    client = FakeClient(place_error=ApiError(400, "INSUFFICIENT_FUNDS", "not enough margin"))
    router = OrderRouter(client, ExecutionConfig(verify_fills=False), SETTINGS)
    result = router.open_position("EUR/USD", "LONG", 0.01, None, None, 1.10)
    assert not result.ok
    assert "INSUFFICIENT_FUNDS" in (result.error or "")


def test_qty_floored_to_step():
    client = FakeClient()
    router = OrderRouter(client, ExecutionConfig(verify_fills=False), SETTINGS)
    router.open_position("EUR/USD", "LONG", 0.0129, None, None, 1.10)
    assert client.placed[0]["qty"] == pytest.approx(0.013)


def test_dry_run_sends_nothing():
    client = FakeClient()
    router = OrderRouter(client, ExecutionConfig(dry_run=True), SETTINGS)
    result = router.open_position("EUR/USD", "LONG", 0.01, 1.09, 1.11, 1.10)
    assert result.ok and result.dry_run
    assert client.placed == []


def test_close_position_calls_api():
    client = FakeClient()
    router = OrderRouter(client, ExecutionConfig(verify_fills=False), SETTINGS)
    router.open_position("EUR/USD", "LONG", 0.01, 1.095, 1.11, 1.10)
    pos = client.positions()[0]
    result = router.close_position(pos, reason="test")
    assert result.ok
    assert client.closed == [pos.id]

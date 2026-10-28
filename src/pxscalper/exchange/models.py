"""Typed models for PrimeXBT Trading API v3 payloads (the subset the bot uses).

Field names mirror the wire format exactly (snake_case where the API uses it).
Unknown fields are allowed so an additive API change never breaks the bot --
PrimeXBT's version policy is additive-only for /v3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Bar",
    "Market",
    "Quote",
    "SymbolSettings",
    "AccountMetrics",
    "Position",
    "Order",
    "Fill",
    "OrderCost",
]


def _d(obj: dict[str, Any], key: str, default: Any = None) -> Any:
    return obj.get(key, default)


@dataclass(slots=True)
class Bar:
    time: int  # epoch ms, open time
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @classmethod
    def from_api(cls, o: dict[str, Any]) -> "Bar":
        return cls(
            time=int(o["time"]),
            open=float(o["open"]),
            high=float(o["high"]),
            low=float(o["low"]),
            close=float(o["close"]),
            volume=float(o.get("volume") or 0.0),
        )


@dataclass(slots=True)
class Market:
    symbol: str
    price: float
    bid: float
    ask: float
    market_type: str = ""
    category: str = ""
    price_scale: int = 5
    qty_scale: int = 4
    change_24h_pct: float = 0.0
    high_24h: float = 0.0
    low_24h: float = 0.0
    volume_24h: float = 0.0
    market_open: bool = True
    tradable: bool = True

    @property
    def spread(self) -> float:
        return max(self.ask - self.bid, 0.0)

    @property
    def spread_bps(self) -> float:
        mid = (self.ask + self.bid) / 2 if (self.ask + self.bid) else self.price
        return (self.spread / mid * 1e4) if mid else 0.0

    @classmethod
    def from_api(cls, o: dict[str, Any]) -> "Market":
        return cls(
            symbol=o["symbol"],
            price=float(o.get("price") or 0.0),
            bid=float(o.get("bid") or 0.0),
            ask=float(o.get("ask") or 0.0),
            market_type=o.get("marketType", "") or "",
            category=o.get("category", "") or "",
            price_scale=int(o.get("priceScale") or 5),
            qty_scale=int(o.get("qtyScale") or 4),
            change_24h_pct=float(o.get("change24hPct") or 0.0),
            high_24h=float(o.get("high24h") or 0.0),
            low_24h=float(o.get("low24h") or 0.0),
            volume_24h=float(o.get("volume24h") or 0.0),
            market_open=bool(o.get("marketOpen", True)),
            tradable=bool(o.get("tradable", True)),
        )


@dataclass(slots=True)
class Quote:
    symbol: str
    bid: float
    ask: float
    price: float
    market_open: bool = True
    timestamp_ms: int = 0

    @classmethod
    def from_api(cls, o: dict[str, Any], ts: int = 0) -> "Quote":
        return cls(
            symbol=o["symbol"],
            bid=float(o.get("bid") or 0.0),
            ask=float(o.get("ask") or 0.0),
            price=float(o.get("price") or (o.get("bid", 0) + o.get("ask", 0)) / 2),
            market_open=bool(o.get("marketOpen", True)),
            timestamp_ms=ts,
        )


@dataclass(slots=True)
class SymbolSettings:
    symbol: str
    min_qty: float
    max_qty: float
    qty_step: float
    min_price_step: float
    price_scale: int
    max_leverage: int
    default_leverage: int
    taker_fee_pct: float
    maker_fee_pct: float
    financing_long_pct: float = 0.0
    financing_short_pct: float = 0.0
    financing_interval: str = "8h"
    trading_hours: str = ""
    market_type: str = ""
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def round_qty(self, qty: float) -> float:
        if self.qty_step <= 0:
            return max(qty, self.min_qty)
        steps = round(qty / self.qty_step)
        return round(max(steps * self.qty_step, self.min_qty), 10)

    def round_price(self, price: float) -> float:
        if self.min_price_step <= 0:
            return round(price, self.price_scale)
        steps = round(price / self.min_price_step)
        return round(steps * self.min_price_step, 10)

    @classmethod
    def from_api(cls, o: dict[str, Any]) -> "SymbolSettings":
        return cls(
            symbol=o["symbol"],
            min_qty=float(o.get("minQty") or 0.0),
            max_qty=float(o.get("maxQty") or 1e12),
            qty_step=float(o.get("qtyStep") or 0.0),
            min_price_step=float(o.get("minPriceStep") or 0.0),
            price_scale=int(o.get("priceScale") or 5),
            max_leverage=int(o.get("maxLeverage") or 1),
            default_leverage=int(o.get("defaultLeverage") or 1),
            taker_fee_pct=float(o.get("takerFeePct") or 0.0),
            maker_fee_pct=float(o.get("makerFeePct") or 0.0),
            financing_long_pct=float(o.get("financingLongPct") or 0.0),
            financing_short_pct=float(o.get("financingShortPct") or 0.0),
            financing_interval=str(o.get("financingInterval") or "8h"),
            trading_hours=str(o.get("tradingHours") or ""),
            market_type=str(o.get("marketType") or ""),
            raw=o,
        )


@dataclass(slots=True)
class AccountMetrics:
    account_id: str
    currency: str
    balance: float
    equity: float
    real_equity: float
    upl: float
    available_margin: float
    used_margin: float
    maintenance_margin: float
    margin_ratio: float
    margin_level: str | None
    position_mode: str
    open_positions: int
    open_orders: int
    real_money: bool
    blocked: bool
    reduce_only: bool
    server: str

    @classmethod
    def from_api(cls, o: dict[str, Any]) -> "AccountMetrics":
        return cls(
            account_id=str(o.get("accountId", "")),
            currency=str(o.get("currency", "USD")),
            balance=float(o.get("balance") or 0.0),
            equity=float(o.get("equity") or 0.0),
            real_equity=float(o.get("realEquity") or 0.0),
            upl=float(o.get("upl") or 0.0),
            available_margin=float(o.get("availableMargin") or 0.0),
            used_margin=float(o.get("usedMargin") or 0.0),
            maintenance_margin=float(o.get("maintenanceMargin") or 0.0),
            margin_ratio=float(o.get("marginRatio") or 0.0),
            margin_level=o.get("marginLevel"),
            position_mode=str(o.get("positionMode", "NETTING")),
            open_positions=int(o.get("openPositions") or 0),
            open_orders=int(o.get("openOrders") or 0),
            real_money=bool(o.get("realMoney", False)),
            blocked=bool(o.get("blocked", False)),
            reduce_only=bool(o.get("reduceOnly", False)),
            server=str(o.get("server", "")),
        )


@dataclass(slots=True)
class Position:
    id: int
    symbol: str
    side: str  # BUY | SELL
    qty: float
    open_price: float
    current_price: float
    mark_price: float
    upl: float
    roe_pct: float
    liq_price: float | None
    margin: float
    client_leverage: float
    stop_loss_price: float | None
    take_profit_price: float | None
    open_time: int
    closeable: bool = True
    exposure: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def signed_qty(self) -> float:
        return self.qty if self.side.upper() == "BUY" else -self.qty

    @classmethod
    def from_api(cls, o: dict[str, Any]) -> "Position":
        return cls(
            id=int(o["id"]),
            symbol=o["symbol"],
            side=str(o.get("side", "BUY")),
            qty=float(o.get("qty") or 0.0),
            open_price=float(o.get("openPrice") or 0.0),
            current_price=float(o.get("currentPrice") or 0.0),
            mark_price=float(o.get("markPrice") or 0.0),
            upl=float(o.get("upl") or 0.0),
            roe_pct=float(o.get("roePct") or 0.0),
            liq_price=(float(o["liqPrice"]) if o.get("liqPrice") is not None else None),
            margin=float(o.get("margin") or 0.0),
            client_leverage=float(o.get("clientLeverage") or 0.0),
            stop_loss_price=(
                float(o["stopLossPrice"]) if o.get("stopLossPrice") is not None else None
            ),
            take_profit_price=(
                float(o["takeProfitPrice"]) if o.get("takeProfitPrice") is not None else None
            ),
            open_time=int(o.get("openTime") or 0),
            closeable=bool(o.get("closeable", True)),
            exposure=float(o.get("exposure") or 0.0),
            raw=o,
        )


@dataclass(slots=True)
class Order:
    id: int
    symbol: str
    side: str
    type: str
    status: str
    qty: float
    executed_qty: float
    executed_price: float | None
    trigger_price: float | None
    protective: bool
    position_id: int | None
    client_order_id: str | None
    placed_at: int
    source: str = ""

    @classmethod
    def from_api(cls, o: dict[str, Any]) -> "Order":
        return cls(
            id=int(o["id"]),
            symbol=o["symbol"],
            side=str(o.get("side", "")),
            type=str(o.get("type", "")),
            status=str(o.get("status", "")),
            qty=float(o.get("qty") or 0.0),
            executed_qty=float(o.get("executedQty") or 0.0),
            executed_price=(
                float(o["executedPrice"]) if o.get("executedPrice") is not None else None
            ),
            trigger_price=(
                float(o["triggerPrice"]) if o.get("triggerPrice") is not None else None
            ),
            protective=bool(o.get("protective", False)),
            position_id=(int(o["positionId"]) if o.get("positionId") is not None else None),
            client_order_id=o.get("clientOrderId"),
            placed_at=int(o.get("placedAt") or 0),
            source=str(o.get("source", "")),
        )


@dataclass(slots=True)
class Fill:
    id: int
    order_id: int
    symbol: str
    side: str
    type: str  # MAKER | TAKER
    order_type: str
    qty: float
    price: float
    fee: float
    rpl: float
    time: int

    @classmethod
    def from_api(cls, o: dict[str, Any]) -> "Fill":
        return cls(
            id=int(o["id"]),
            order_id=int(o.get("orderId") or 0),
            symbol=o["symbol"],
            side=str(o.get("side", "")),
            type=str(o.get("type", "")),
            order_type=str(o.get("orderType", "")),
            qty=float(o.get("qty") or 0.0),
            price=float(o.get("price") or 0.0),
            fee=float(o.get("fee") or 0.0),
            rpl=float(o.get("rpl") or 0.0),
            time=int(o.get("time") or 0),
        )


@dataclass(slots=True)
class OrderCost:
    """Answer of any ``*/impact`` route: what an order would cost, without placing it."""

    currency: str
    price: float
    order_cost: float
    margin_impact: float
    fee: float
    fee_pct: float
    leverage: float
    liq_price: float | None
    min_order_size: float
    resolution: str
    resolution_level: str
    vwap_price_offset: float = 0.0
    take_profit_pl: float | None = None
    stop_loss_pl: float | None = None

    @property
    def ok(self) -> bool:
        return self.resolution_level != "ERROR"

    @classmethod
    def from_api(cls, o: dict[str, Any]) -> "OrderCost":
        return cls(
            currency=str(o.get("currency", "")),
            price=float(o.get("price") or 0.0),
            order_cost=float(o.get("orderCost") or 0.0),
            margin_impact=float(o.get("marginImpact") or 0.0),
            fee=float(o.get("fee") or 0.0),
            fee_pct=float(o.get("feePct") or 0.0),
            leverage=float(o.get("leverage") or 0.0),
            liq_price=(float(o["liqPrice"]) if o.get("liqPrice") is not None else None),
            min_order_size=float(o.get("minOrderSize") or 0.0),
            resolution=str(o.get("resolution", "")),
            resolution_level=str(o.get("resolutionLevel", "")),
            vwap_price_offset=float(o.get("vwapPriceOffset") or 0.0),
            take_profit_pl=(
                float(o["takeProfitPl"]) if o.get("takeProfitPl") is not None else None
            ),
            stop_loss_pl=(float(o["stopLossPl"]) if o.get("stopLossPl") is not None else None),
        )

"""Exchange models: parsing and rounding must match PrimeXBT's increments."""

from __future__ import annotations

from pxscalper.exchange.models import Bar, Market, Position, Quote, SymbolSettings


def test_symbol_settings_rounding_floors_to_step():
    settings = SymbolSettings.from_api(
        {
            "symbol": "EUR/USD", "minQty": 0.001, "maxQty": 100, "qtyStep": 0.001,
            "minPriceStep": 0.00001, "priceScale": 5, "maxLeverage": 100,
            "defaultLeverage": 20, "takerFeePct": 0.05, "makerFeePct": 0.02,
        }
    )
    assert settings.round_qty(0.0129) == 0.013
    assert settings.round_qty(0.0004) == 0.001  # never below minimum
    assert settings.round_price(1.100007) == 1.10001


def test_market_spread_helpers():
    market = Market.from_api(
        {"symbol": "EUR/USD", "price": 1.1, "bid": 1.0999, "ask": 1.1001,
         "priceScale": 5, "marketOpen": True}
    )
    assert market.spread > 0
    assert market.spread_bps < 2.0


def test_position_signed_qty():
    position = Position.from_api(
        {"id": 1, "symbol": "EUR/USD", "side": "SELL", "qty": 0.5, "openPrice": 1.1,
         "currentPrice": 1.09, "markPrice": 1.09, "upl": 5.0, "roePct": 1.0,
         "margin": 10.0, "clientLeverage": 5, "openTime": 0}
    )
    assert position.signed_qty == -0.5


def test_quote_falls_back_to_mid():
    quote = Quote.from_api({"symbol": "EUR/USD", "bid": 1.0999, "ask": 1.1001})
    assert quote.price == 1.1


def test_bar_parsing_ignores_missing_volume():
    bar = Bar.from_api({"time": 1, "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05})
    assert bar.volume == 0.0

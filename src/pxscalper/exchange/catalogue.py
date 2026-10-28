"""Symbol catalogue helpers: resolve what the account can actually trade.

PrimeXBT's catalogue is heterogeneous and the obvious filters lie:

* ``markets(marketType="FOREX")`` returns *everything* on this endpoint --
  crypto included -- so filtering by ``marketType`` alone is not enough.
* FX symbols are named ``EUR/USD`` but the category field is ``forex``, and
  instrument sizes are quoted in **units of base currency** with a very coarse
  step (``qtyStep = 1000`` for EUR/USD), which changes position sizing entirely
  compared with a crypto pair (``minQty = 0.0001``).
* Indices/shares/commodities appear by bare ticker (``AAPL``, ``US500``) with no
  ``/`` at all, so "contains a slash" is also not a reliable filter.

Everything downstream (live loop, backtest, risk sizing) reads the catalogue
from here instead of guessing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from ..exchange.models import Market, SymbolSettings
from ..exchange.rest import ApiError, PrimeXBTClient

log = logging.getLogger(__name__)

__all__ = ["Instrument", "SymbolResolver", "resolve_units"]

#: categories that are genuinely FX
FX_CATEGORIES = {"forex"}
#: quote currencies that settle in USD (PnL per unit = 1.0)
USD_QUOTES = {"USD", "USDT", "USDC"}


@dataclass(slots=True)
class Instrument:
    symbol: str
    market_type: str
    category: str
    base: str
    quote: str
    price: float
    bid: float
    ask: float
    market_open: bool
    settings: SymbolSettings | None = None

    @property
    def spread_bps(self) -> float:
        mid = (self.bid + self.ask) / 2 if (self.bid + self.ask) else self.price
        return (self.ask - self.bid) / mid * 1e4 if mid else 0.0

    @property
    def is_fx(self) -> bool:
        return self.category.lower() in FX_CATEGORIES or (
            self.market_type.upper() == "FOREX" and self.base and self.quote
        )

    @property
    def qty_step(self) -> float:
        return self.settings.qty_step if self.settings else 0.0

    @property
    def min_qty(self) -> float:
        return self.settings.min_qty if self.settings else 0.0

    def notional_per_step(self, quote_usd_rate: float | None = None) -> float:
        """USD notional of one quantity step -- the real granularity of risk.

        ``EUR/USD``: step 1000 units x 1.115 = $1,115.
        ``USD/JPY``: step 1000 units are 1000 USD of base, so $1,000 -- *not*
        1000 x 157.8. Using ``price`` alone overstates it by the FX rate.
        """
        base, quote = self.base.upper(), self.quote.upper()
        if quote in USD_QUOTES:
            return self.qty_step * self.price
        if base in USD_QUOTES:
            return self.qty_step
        if quote_usd_rate is not None:
            return self.qty_step * self.price * quote_usd_rate
        raise ValueError(f"{self.symbol}: needs {quote}->USD rate to size in USD")

    def describe(self) -> str:
        s = self.settings
        fee = f"taker {s.taker_fee_pct:.3f}%" if s else "fees unknown"
        return (
            f"{self.symbol:10s} {self.category:12s} px {self.price:>12.5f} "
            f"spread {self.spread_bps:6.2f}bps step {self.qty_step:g} {fee} "
            f"{'OPEN' if self.market_open else 'CLOSED'}"
        )


class SymbolResolver:
    """Ask the exchange what exists, then answer questions about it."""

    def __init__(self, client: PrimeXBTClient) -> None:
        self.client = client
        self._instruments: dict[str, Instrument] = {}

    def refresh(self, symbols: Iterable[str] | None = None) -> dict[str, Instrument]:
        wanted = list(symbols) if symbols else None
        markets: list[Market] = self.client.markets(symbols=wanted)
        try:
            settings = {s.symbol: s for s in self.client.settings(symbols=wanted)}
        except ApiError as exc:  # settings needs `read`; markets may not
            log.warning("settings unavailable (%s); sizing data will be missing", exc)
            settings = {}
        for market in markets:
            self._instruments[market.symbol] = Instrument(
                symbol=market.symbol,
                market_type=market.market_type,
                category=market.category,
                base=market.symbol.partition("/")[0] or market.symbol,
                quote=market.symbol.partition("/")[2],
                price=market.price,
                bid=market.bid,
                ask=market.ask,
                market_open=market.market_open,
                settings=settings.get(market.symbol),
            )
        return self._instruments

    # ------------------------------------------------------------------ queries
    def get(self, symbol: str) -> Instrument | None:
        if symbol not in self._instruments:
            self.refresh([symbol])
        return self._instruments.get(symbol)

    def fx_instruments(self) -> list[Instrument]:
        if not self._instruments:
            self.refresh()
        return [i for i in self._instruments.values() if i.is_fx]

    def open_fx(self) -> list[Instrument]:
        return [i for i in self.fx_instruments() if i.market_open]

    def quote_usd_rates(self) -> dict[str, float]:
        """Quote currency -> USD rate, derived from the catalogue (USD/<quote>)."""
        rates: dict[str, float] = {}
        for instrument in self._instruments.values():
            base, quote = instrument.base.upper(), instrument.quote.upper()
            if base in USD_QUOTES and quote and instrument.price > 0:
                rates[quote] = 1.0 / instrument.price   # USD/JPY 157.8 -> JPY = 0.00634
        return rates

    def notional_per_step(self, instrument: Instrument) -> float | None:
        try:
            return instrument.notional_per_step(self.quote_usd_rates().get(instrument.quote.upper()))
        except ValueError:
            return None

    def scalping_candidates(self, max_spread_bps: float = 3.0) -> list[Instrument]:
        """FX instruments that are open and tight enough to scalp."""
        return sorted(
            (i for i in self.open_fx() if i.spread_bps <= max_spread_bps),
            key=lambda i: i.spread_bps,
        )


def resolve_units(
    instrument: Instrument, usd_notional: float
) -> tuple[float, float, str]:
    """Convert a desired USD notional into exchange units, respecting the step.

    FX on PrimeXBT is sized in base-currency units with a coarse step (1000 EUR
    for EUR/USD ⇒ ~$1.1k of notional per step). Risk sizing must therefore be
    *quantised*: we round **down** to the step and report what the trade
    actually risks, instead of pretending to trade a size the exchange cannot
    represent.

    Returns ``(qty, actual_notional_usd, note)``.
    """
    if instrument.price <= 0:
        return 0.0, 0.0, "no price"
    raw_units = usd_notional / instrument.price
    step = instrument.qty_step or 0.0
    min_qty = instrument.min_qty or 0.0
    if step > 0:
        qty = int(raw_units // step) * step
    else:
        qty = raw_units
    note = ""
    if qty < min_qty:
        qty = 0.0
        note = (
            f"below minimum: need {min_qty:g} units ≈ ${min_qty * instrument.price:,.0f} "
            f"notional, budget was ${usd_notional:,.0f}"
        )
    return float(qty), float(qty * instrument.price), note

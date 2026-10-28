"""Real XAU/USD order-cost ladder: what does size actually cost on the venue?

Read-only (`/market-order-cost`), sends no orders. This is the authoritative
slippage number -- the price PrimeXBT would actually fill at, plus the offset
from the quoted price.
"""
import dataclasses
import os

from pxscalper.exchange.rest import PrimeXBTClient
from pxscalper.exchange.signing import ApiCredentials

client = PrimeXBTClient(
    credentials=ApiCredentials(os.environ["PX_API_KEY"], os.environ["PX_API_SECRET"])
)
QUOTE = 4067.26

header = f"{'qty (oz)':>9s} {'fill':>10s} {'offset bps':>11s} {'order cost':>12s} {'margin':>9s} {'fee%':>5s}"
print(f"spot bid ~{QUOTE:,.2f} | spread live 0.49 bps\n{header}")
print("-" * len(header))
rows = []
for qty in (0.01, 0.1, 1, 10, 100, 500):
    try:
        data = dataclasses.asdict(
            client.market_order_cost(symbol="XAU/USD", side="BUY", qty=qty)
        )
    except Exception as exc:  # noqa: BLE001 - live error reporting is the point
        print(f"{qty:9.2f} {type(exc).__name__}: {str(exc)[:60]}")
        continue
    offset_bps = (data["price"] / QUOTE - 1.0) * 1e4
    rows.append((qty, offset_bps))
    print(
        f"{qty:9.2f} {data['price']:10,.2f} {offset_bps:11.2f} "
        f"{data['order_cost']:12,.2f} {data['margin_impact']:9,.2f} {data['fee_pct']:5.2f}"
    )

print("\nmarginal impact vs size (penentu apakah scalping bisa menutup biaya):")
for qty, offset in rows:
    notional = qty * QUOTE
    print(f"  {qty:8.2f} oz (${notional:10,.0f} notional) -> {offset:+.2f} bps, "
          f"sama dengan {abs(offset) / 0.49:.1f}x spread")

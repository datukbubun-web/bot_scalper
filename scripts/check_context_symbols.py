"""Read-only: which cross-asset context symbols does the venue actually offer?"""
import os

from pxscalper.exchange.catalogue import SymbolResolver
from pxscalper.exchange.rest import PrimeXBTClient
from pxscalper.exchange.signing import ApiCredentials

client = PrimeXBTClient(
    credentials=ApiCredentials(os.environ["PX_API_KEY"], os.environ["PX_API_SECRET"])
)
resolver = SymbolResolver(client)
resolver.refresh()
syms = {i.symbol: i for i in resolver._instruments.values()}

for want in ("XAG/USD", "DXY", "USD/JPY", "USD/CHF", "EUR/USD", "USD/CAD", "XAU/EUR"):
    inst = syms.get(want)
    if inst:
        print(f"{want:10s} ADA      open={inst.market_open} price={inst.price:,.4f} "
              f"spread={inst.spread_bps:.2f}bps step={inst.qty_step:g}")
    else:
        print(f"{want:10s} TIDAK ADA")

print("\nkandidat XAG:", sorted(s for s in syms if "XAG" in s))
print("proxy dolar (USD/X):", sorted(s for s in syms if s.startswith("USD/"))[:14])
print("metals:", sorted(i.symbol for i in syms.values() if i.category == "metals"))

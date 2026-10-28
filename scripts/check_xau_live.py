"""Read-only live checks for XAU/USD: real impact, sizing, market status."""
import dataclasses
import json
import os
import tempfile

from pxscalper.exchange.rest import PrimeXBTClient
from pxscalper.exchange.signing import ApiCredentials
from pxscalper.risk.engine import RiskConfig, RiskEngine, RiskState

client = PrimeXBTClient(
    credentials=ApiCredentials(os.environ["PX_API_KEY"], os.environ["PX_API_SECRET"])
)

print("=== 1. impact nyata (read-only, tidak mengirim order) ===")
for qty in (0.01, 0.1, 1, 10, 100):
    try:
        answer = client.market_order_cost(symbol="XAU/USD", side="BUY", qty=qty)
        data = answer if isinstance(answer, dict) else dataclasses.asdict(answer)
        print(f"  BUY {qty:>6} oz -> {json.dumps(data)[:240]}")
    except Exception as exc:  # noqa: BLE001 - reporting live errors is the point
        print(f"  BUY {qty:>6} oz -> {type(exc).__name__}: {exc}")

print("\n=== 2. sizing lewat risk engine (equity 10k USD) ===")
tmp = tempfile.mkdtemp()
engine = RiskEngine(
    RiskConfig(state_path=f"{tmp}/r.json", kill_switch_path=f"{tmp}/K", min_atr_bps=0.4),
    RiskState(),
)
verdict = engine.evaluate(
    symbol="XAU/USD", direction=1, confidence=0.55, edge=0.12, price=4067.18,
    equity=10000.0, atr=1.5, spread_bps=0.49, market_open=True,
    min_qty=0.01, qty_step=0.01, price_step=0.01,
)
print(f"  keputusan: {verdict.decision.value} | {verdict.reason}")
print(f"  qty {verdict.qty} oz (notional ${verdict.qty * 4067.18:,.2f})")
print(f"  stop {verdict.stop_loss} | tp {verdict.take_profit}")
keys = ("atr_bps", "notional_per_step", "max_notional_room", "risk_amount")
print("  detail:", json.dumps({k: verdict.details.get(k) for k in keys}, default=str))

print("\n=== 3. status pasar ===")
market = client.markets(symbols=["XAU/USD"])[0]
print(f"  XAU/USD open={market.market_open} bid={market.bid} ask={market.ask} "
      f"spread={market.spread_bps:.2f}bps")
print("  jam: 22:01 Sun - 20:59 Fri (break harian 20:59-22:01 UTC)")

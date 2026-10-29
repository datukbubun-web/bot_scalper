"""Demo: 10 staged entries in XAU/USD, then close-all when basket profit hits $3.

Runs against the real preset, the real RiskEngine guards and the real basket
rule -- no mocks, no fake numbers. Nothing is sent to an exchange.

    python3 scripts/demo_staged_basket.py
    python3 scripts/demo_staged_basket.py --config configs/live.mini100.yaml --target 5
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml

from pxscalper.exchange.models import Position
from pxscalper.features.calendar import _et
from pxscalper.risk.basket import BasketTarget
from pxscalper.risk.engine import RiskConfig, RiskEngine, RiskState

PRICE = 4067.18
NOW = _et(2026, 10, 2) + 3 * 3_600_000  # a quiet Thursday, outside any blackout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/live.mini100.yaml")
    ap.add_argument("--target", type=float, default=None, help="override close_all_on_profit")
    ap.add_argument("--equity", type=float, default=100.0)
    args = ap.parse_args()

    raw = yaml.safe_load(Path(args.config).read_text())
    risk_raw = raw["risk"]
    target = args.target if args.target is not None else float(raw.get("close_all_on_profit", 0))
    enforce = bool(raw.get("close_all_enforce", False))

    print(f"config : {args.config}")
    print(f"equity : ${args.equity:.2f}")
    print(f"slots  : max_positions={risk_raw.get('max_positions')} "
          f"per_symbol={risk_raw.get('max_positions_per_symbol')} "
          f"symbols={risk_raw.get('allowed_symbols')}")
    print(f"target : close_all_on_profit=${target:.2f}  enforce={enforce}\n")

    # ---------------------------------------------------------- staged entries
    print("── masuk bertahap ─────────────────────────────────────────────")
    book: list[dict] = []
    total_notional = 0.0
    for n in range(1, 12):
        d = tempfile.mkdtemp()
        engine = RiskEngine(
            RiskConfig.from_dict({**risk_raw, "state_path": f"{d}/r", "kill_switch_path": f"{d}/K"},
                                 allow_env=False),
            RiskState(),
        )
        engine.on_account(args.equity)
        verdict = engine.evaluate(
            symbol="XAU/USD", direction=1, confidence=0.6, edge=0.1, price=PRICE,
            equity=args.equity, atr=1.5, spread_bps=0.49, market_open=True,
            min_qty=0.01, qty_step=0.01, price_step=0.01, now_ms=NOW,
            open_positions=list(book),
        )
        if verdict.decision.is_tradeable and verdict.qty > 0:
            notional = verdict.qty * PRICE
            book.append({"symbol": "XAU/USD", "side": "BUY", "qty": verdict.qty,
                         "mark_price": PRICE, "notional": notional})
            total_notional += notional
            print(f"  entry {n:2d}: {verdict.qty:.2f} oz  ${notional:6.0f}  "
                  f"| total ${total_notional:6.0f}  | sisa anggaran/slot "
                  f"${verdict.details.get('budget_per_slot', 0):5.0f}")
        else:
            print(f"  entry {n:2d}: DITOLAK — {verdict.reason}")
    print(f"\n  --> {len(book)} posisi terbuka, total ${total_notional:,.0f} notional "
          f"(plafon ${args.equity * float(risk_raw.get('max_gross_exposure_pct', 0)) / 100:,.0f})\n")

    # ------------------------------------------------------- basket behaviour
    print("── target profit gabungan ─────────────────────────────────────")
    basket = BasketTarget(target)

    def leg(i: int, upl: float) -> Position:
        return Position(
            id=i, symbol="XAU/USD", side="BUY", qty=0.02, open_price=PRICE,
            current_price=PRICE, mark_price=PRICE, upl=upl, roe_pct=0.0,
            liq_price=None, margin=20.0, client_leverage=10.0, stop_loss_price=None,
            take_profit_price=None, open_time=0, closeable=True,
            exposure=PRICE * 0.02, raw={},
        )

    scenarios = [
        ("semua baru buka", [0.0] * len(book)),
        ("sebagian untung", [1.2, 0.9] + [-0.4] * (len(book) - 2)),
        ("hampir mencapai", [1.8, 1.1] + [-0.2] * (len(book) - 2)),
        ("TEPAT target", [2.0, 1.4] + [-0.1] * (len(book) - 2)),
        ("target lewat, 8 rugi", [4.2, 2.8] + [-0.5] * (len(book) - 2)),
    ]
    for label, pnls in scenarios:
        positions = [leg(i + 1, p) for i, p in enumerate(pnls)]
        result = basket.evaluate(positions, enforce=True)
        action = "TUTUP SEMUA" if result.triggered else "tahan"
        print(f"  {label:22s} total ${result.total_pnl:+6.2f}  "
              f"{result.winners} naik / {result.losers} turun  -> {action}")

    print("\n  Catatan: kalau close_all_enforce=false, bot hanya mencatat target")
    print("  tercapai di log dan TIDAK menutup apa pun. Berguna untuk paper run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

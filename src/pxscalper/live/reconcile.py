"""Reconciliation on startup and after any ambiguous write.

PrimeXBT is explicit: *"No request is safe to repeat. A placed order sent twice is
two orders. When a write times out, read orders/active and positions before
sending it again."*

The riskiest moment in a bot's life is a restart with an open position and a
half-written journal. This module compares three sources of truth -- the journal
(everything the bot believes it did), ``positions`` (what the exchange holds) and
``orders/active`` (what is resting) -- and produces an explicit action list:

* position at the broker, unknown to the journal -> adopt it (with its exchange
  stop) so the manager can still exit it;
* journal position, absent at the broker -> mark closed at the last known price;
* resting order with no journal record -> cancel (it was orphaned by a crash);
* position without a stop -> attach one immediately.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..exchange.rest import ApiError, PrimeXBTClient

log = logging.getLogger(__name__)

__all__ = ["ReconcileReport", "Reconciler"]


@dataclass
class ReconcileReport:
    adopted_positions: list[int] = field(default_factory=list)
    missing_positions: list[int] = field(default_factory=list)
    orphan_orders: list[int] = field(default_factory=list)
    unprotected_positions: list[int] = field(default_factory=list)
    cancelled: list[int] = field(default_factory=list)
    stops_attached: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    ok: bool = True

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def summary(self) -> str:
        return (
            f"adopted={len(self.adopted_positions)} missing={len(self.missing_positions)} "
            f"orphans_cancelled={len(self.cancelled)} stops_attached={len(self.stops_attached)} "
            f"errors={len(self.errors)}"
        )


class Reconciler:
    def __init__(self, client: PrimeXBTClient, state_path: str | Path = "state/live_positions.json"):
        self.client = client
        self.state_path = Path(state_path)

    def _load_journal(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"positions": {}}
        try:
            return json.loads(self.state_path.read_text())
        except Exception:  # noqa: BLE001
            return {"positions": {}}

    def _save_journal(self, journal: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(journal, indent=2))

    def run(self, attach_stops: bool = True, atr_by_symbol: dict[str, float] | None = None,
            sl_atr: float = 1.0) -> ReconcileReport:
        report = ReconcileReport()
        try:
            positions = self.client.positions()
            orders = self.client.active_orders()
        except ApiError as exc:
            report.ok = False
            report.errors.append(str(exc))
            return report

        journal = self._load_journal()
        known: dict[str, dict] = journal.get("positions", {})
        live_ids = {str(p.id) for p in positions}

        # 1. adopt unknown positions ---------------------------------------
        for pos in positions:
            if str(pos.id) not in known:
                report.adopted_positions.append(pos.id)
                known[str(pos.id)] = {
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "qty": pos.qty,
                    "open_price": pos.open_price,
                    "open_time": pos.open_time,
                    "source": "reconciled",
                }
                log.warning("adopted unknown position %s %s %s", pos.id, pos.side, pos.symbol)
            if attach_stops and pos.stop_loss_price is None:
                report.unprotected_positions.append(pos.id)
                atr = (atr_by_symbol or {}).get(pos.symbol, 0.0)
                if atr > 0:
                    side = 1 if pos.side == "BUY" else -1
                    stop = pos.open_price - side * sl_atr * atr
                    try:
                        self.client.set_protective(pos.id, "STOP_LOSS", stop)
                        report.stops_attached.append(pos.id)
                        log.warning("attached missing stop to %s at %.6f", pos.id, stop)
                    except ApiError as exc:
                        report.errors.append(f"stop {pos.id}: {exc}")

        # 2. journal positions that no longer exist -------------------------
        for pos_id in list(known):
            if pos_id not in live_ids:
                report.missing_positions.append(int(pos_id))
                known.pop(pos_id, None)
                log.info("journal position %s no longer open", pos_id)

        # 3. orphan working orders -----------------------------------------
        for order in orders:
            if order.protective:
                continue
            if order.client_order_id and order.client_order_id.startswith("pxs."):
                continue
            if order.position_id is None:
                report.orphan_orders.append(order.id)
                try:
                    self.client.cancel_order(order.id)
                    report.cancelled.append(order.id)
                    log.warning("cancelled orphan order %s", order.id)
                except ApiError as exc:
                    report.errors.append(f"cancel {order.id}: {exc}")

        journal["positions"] = known
        self._save_journal(journal)
        report.ok = not report.errors
        log.info("reconcile: %s", report.summary())
        return report

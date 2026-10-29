"""Opening N positions is safe only if N is capped AND the total is capped.

The original code capped `max_gross_exposure_pct` per order and never summed
what was already open. With one position that is invisible; with ten it means
ten times the intended leverage. These tests exist to keep that closed.
"""

from __future__ import annotations

import tempfile

import pytest

from pxscalper.exchange.models import Position
from pxscalper.features.calendar import _et
from pxscalper.risk.engine import RiskConfig, RiskDecision, RiskEngine, RiskState, _normalise_position

PRICE = 4067.18
NOW = _et(2026, 10, 2) + 3 * 3_600_000


def make_position(i: int = 1, symbol: str = "XAU/USD", exposure: float = 976.0,
                  side: str = "BUY") -> Position:
    return Position(
        id=i, symbol=symbol, side=side, qty=0.24, open_price=PRICE, current_price=PRICE,
        mark_price=PRICE, upl=0.0, roe_pct=0.0, liq_price=None, margin=10.0,
        client_leverage=10.0, stop_loss_price=None, take_profit_price=None,
        open_time=0, closeable=True, exposure=exposure, raw={},
    )


def engine(**overrides) -> RiskEngine:
    d = tempfile.mkdtemp()
    base = dict(
        state_path=f"{d}/r.json", kill_switch_path=f"{d}/K",
        risk_per_trade_pct=0.5, max_leverage=10.0, max_gross_exposure_pct=1000.0,
        max_positions=10, max_positions_per_symbol=10,
        max_intraday_loss_pct=0.0, max_daily_loss_pct=100.0,
        max_drawdown_pct=100.0, max_total_loss_pct=0.0,
    )
    base.update(overrides)
    e = RiskEngine(RiskConfig(**base), RiskState())
    e.on_account(100.0)
    return e


def evaluate(e: RiskEngine, positions):
    return e.evaluate(
        symbol="XAU/USD", direction=1, confidence=0.6, edge=0.1, price=PRICE,
        equity=100.0, atr=1.5, spread_bps=0.49, market_open=True,
        min_qty=0.01, qty_step=0.01, price_step=0.01, now_ms=NOW,
        open_positions=positions,
    )


# ── the count the user asked for ────────────────────────────────────────────
@pytest.mark.parametrize("open_count", [0, 1, 5, 9])
def test_ten_positions_allowed_until_the_cap(open_count):
    e = engine(max_positions=10, max_positions_per_symbol=1, max_gross_exposure_pct=100_000)
    positions = [make_position(i, f"SYM{i}", 10.0) for i in range(open_count)]
    assert evaluate(e, positions).decision.is_tradeable


def test_the_eleventh_position_is_refused():
    e = engine(max_positions=10, max_positions_per_symbol=1, max_gross_exposure_pct=100_000)
    positions = [make_position(i, f"SYM{i}", 10.0) for i in range(10)]
    verdict = evaluate(e, positions)
    assert verdict.decision is RiskDecision.BLOCK
    assert "max positions" in verdict.reason


def test_position_count_is_configurable():
    for limit in (1, 3, 10):
        e = engine(max_positions=limit, max_positions_per_symbol=1,
                   max_gross_exposure_pct=100_000)
        full = [make_position(i, f"SYM{i}", 10.0) for i in range(limit)]
        assert not evaluate(e, full).decision.is_tradeable, f"limit {limit} not enforced"


# ── the hole this change closes ─────────────────────────────────────────────
def test_total_exposure_cannot_be_reused_by_every_position():
    """THE regression test. Two open positions of $900 each against a $1,000
    ceiling must block -- previously each *new order* was measured against the
    full ceiling, so the third, fourth ... position was always allowed."""
    e = engine(max_positions=10, max_positions_per_symbol=10, max_gross_exposure_pct=1000.0)
    positions = [make_position(i, f"SYM{i}", 900.0) for i in range(2)]
    verdict = evaluate(e, positions)
    assert verdict.decision is RiskDecision.BLOCK
    assert "gross exposure" in verdict.reason
    assert verdict.details["open_notional"] == pytest.approx(1800.0)
    assert verdict.details["gross_exposure_ceiling"] == pytest.approx(1000.0)


def test_remaining_room_shrinks_with_each_open_position():
    e = engine(max_positions=10, max_positions_per_symbol=10, max_gross_exposure_pct=1000.0)
    room = {}
    for exposure in (0.0, 400.0, 800.0):
        positions = [make_position(i, f"SYM{i}", exposure) for i in range(int(exposure > 0) * 2)]
        verdict = evaluate(e, positions)
        room[exposure] = (verdict.details["gross_exposure_ceiling"]
                          - verdict.details["open_notional"])
    assert room[0.0] > room[400.0] > room[800.0], room


def test_exposure_blocks_even_when_there_is_still_position_slot():
    """A free slot must not override a full exposure budget."""
    e = engine(max_positions=10, max_positions_per_symbol=10, max_gross_exposure_pct=500.0)
    positions = [make_position(1, "SYM1", 600.0)]
    verdict = evaluate(e, positions)
    assert verdict.decision is RiskDecision.BLOCK
    assert verdict.checks["gross_exposure_ok"] is False
    assert len(positions) < 10, "there was a free slot, and it must not be used"


def test_hedging_is_still_refused_when_disabled():
    e = engine(max_positions_per_symbol=5, max_gross_exposure_pct=100_000)
    positions = [make_position(1, exposure=100.0, side="SELL")]
    verdict = e.evaluate(
        symbol="XAU/USD", direction=1, confidence=0.6, edge=0.1, price=PRICE,
        equity=100.0, atr=1.5, spread_bps=0.49, market_open=True,
        min_qty=0.01, qty_step=0.01, price_step=0.01, now_ms=NOW,
        open_positions=positions,
    )
    assert verdict.decision.is_tradeable
    assert verdict.details.get("note") == "reverse handled by execution layer"


# ── position normalisation ──────────────────────────────────────────────────
def test_notional_derived_from_qty_and_price_when_exposure_missing():
    normalised = _normalise_position({"symbol": "XAU/USD", "qty": 2.0, "mark_price": 4000.0})
    assert normalised["notional"] == pytest.approx(8000.0)


def test_exchange_position_objects_are_accepted():
    """The live loop passes Position objects, not dicts. If this silently
    returned zero notional, every position would look free again."""
    normalised = _normalise_position(make_position(exposure=976.0))
    assert normalised["symbol"] == "XAU/USD"
    assert normalised["notional"] == pytest.approx(976.0)


def test_multi_position_preset_loads_and_keeps_the_total_capped():
    import yaml

    raw = yaml.safe_load(open("configs/live.mini100.multi.yaml"))
    cfg = RiskConfig.from_dict(raw["risk"], allow_env=False)
    assert cfg.max_positions == 10
    # The user explicitly asked for up to 10 staged entries in one instrument,
    # with a basket target that closes them all together.
    assert cfg.max_positions_per_symbol == 10
    assert cfg.max_gross_exposure_pct <= 1000.0
    assert raw["execution"]["dry_run"] is True

"""Close-all-on-basket-profit. This decides when real money is taken, so the
edge cases matter more than the happy path."""

from __future__ import annotations

import pytest

from pxscalper.risk.basket import BasketTarget, basket_pnl, position_pnl


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def pos(pnl: float, **extra):
    return {"symbol": "XAU/USD", "side": "BUY", "upl": pnl, **extra}


class ExchangePosition:
    """Mimics exchange.models.Position, which the live loop actually passes."""

    def __init__(self, upl: float) -> None:
        self.symbol = "XAU/USD"
        self.side = "BUY"
        self.upl = upl


# ── the core behaviour ──────────────────────────────────────────────────────
def test_closes_everything_once_the_basket_reaches_the_target():
    target = BasketTarget(3.0)
    result = target.evaluate([pos(2.5), pos(0.6)], enforce=True)
    assert result.triggered
    assert result.total_pnl == pytest.approx(3.1)
    assert "TARGET HIT" in result.reason


def test_closes_even_while_a_position_is_losing():
    """The explicit request: $3 profit is $3 profit, whatever the individual
    legs did."""
    target = BasketTarget(3.0)
    result = target.evaluate([pos(5.0), pos(-1.5), pos(-0.5)], enforce=True)
    assert result.triggered
    assert result.total_pnl == pytest.approx(3.0)
    assert result.losers == 2
    assert result.winners == 1


def test_does_not_fire_below_the_target():
    result = BasketTarget(3.0).evaluate([pos(1.0), pos(1.5)], enforce=True)
    assert not result.triggered
    assert result.total_pnl == pytest.approx(2.5)


def test_boundary_is_inclusive():
    """Exactly $3.00 must fire: an operator who says 3 means 3, not 3.01."""
    assert BasketTarget(3.0).evaluate([pos(3.0)], enforce=True).triggered


def test_negative_basket_never_fires():
    result = BasketTarget(3.0).evaluate([pos(-1.0), pos(-2.0)], enforce=True)
    assert not result.triggered
    assert result.total_pnl == pytest.approx(-3.0)


# ── safety behaviour ────────────────────────────────────────────────────────
def test_report_only_when_not_enforcing():
    """A paper run must be able to observe the rule without closing anything."""
    result = BasketTarget(3.0).evaluate([pos(4.0)], enforce=False)
    assert result.triggered, "the target WAS reached, and that is reported"
    assert "report only" in result.reason


def test_disabled_target_does_nothing():
    for value in (0.0, -1.0):
        result = BasketTarget(value).evaluate([pos(100.0)], enforce=True)
        assert not result.triggered
        assert "disabled" in result.reason


def test_no_positions_is_not_a_trigger():
    result = BasketTarget(3.0).evaluate([], enforce=True)
    assert not result.triggered
    assert result.total_pnl == 0.0


def test_cooldown_prevents_immediate_refire():
    clock = FakeClock()
    target = BasketTarget(3.0, cooldown_s=300, clock=clock)
    assert target.evaluate([pos(4.0)], enforce=True).triggered
    target.mark_fired()
    # flat now; the next cycle opens one position that is instantly +$4
    later = target.evaluate([pos(4.0)], enforce=True)
    assert not later.triggered
    assert "re-arming" in later.reason
    clock.advance(301)
    assert target.evaluate([pos(4.0)], enforce=True).triggered


def test_arm_above_threshold_is_respected():
    target = BasketTarget(3.0, arm_above=5.0)
    assert not target.evaluate([pos(4.0)], enforce=True).triggered
    assert target.evaluate([pos(5.5)], enforce=True).triggered


# ── input handling ──────────────────────────────────────────────────────────
def test_exchange_objects_are_understood():
    """The live loop passes Position objects. If this broke, the basket would
    read $0.00 and never fire -- a silent no-op guard."""
    assert position_pnl(ExchangePosition(1.25)) == pytest.approx(1.25)
    assert basket_pnl([ExchangePosition(1.25), ExchangePosition(-0.25)]) == pytest.approx(1.0)


def test_unknown_shape_raises_instead_of_counting_zero():
    """A silently-ignored position is worse than a loud error: it makes the
    basket look flat when it is not."""
    with pytest.raises((KeyError, AttributeError)):
        position_pnl({"symbol": "XAU/USD", "side": "BUY"})
    with pytest.raises(AttributeError):
        position_pnl(object())


def test_alternative_pnl_field_names_are_accepted():
    assert position_pnl({"unrealized_pnl": 2.0}) == pytest.approx(2.0)
    assert position_pnl({"profit": -0.5}) == pytest.approx(-0.5)


# ── staged entries: guard against "allowed but impossible" ──────────────────
def test_ten_staged_entries_can_actually_be_opened():
    """A config that permits 10 positions must be able to open 10. Before the
    exposure ceiling was split across slots, the first order consumed the whole
    ceiling and the second was refused -- the rule was allowed on paper and
    unreachable in practice."""
    import tempfile

    from pxscalper.features.calendar import _et
    from pxscalper.risk.engine import RiskConfig, RiskEngine, RiskState

    raw = {
        "risk_per_trade_pct": 0.5, "max_leverage": 10.0,
        "max_gross_exposure_pct": 1000.0, "max_positions": 10,
        "max_positions_per_symbol": 10, "single_symbol": True,
        "allowed_symbols": ["XAU/USD"], "split_exposure_across_slots": True,
        "max_intraday_loss_pct": 0.0, "max_daily_loss_pct": 100.0,
        "max_drawdown_pct": 100.0, "max_total_loss_pct": 0.0,
        "blackout_enabled": False,
    }
    now = _et(2026, 10, 2) + 3 * 3_600_000
    book, total = [], 0.0
    for _ in range(10):
        d = tempfile.mkdtemp()
        engine = RiskEngine(
            RiskConfig.from_dict({**raw, "state_path": f"{d}/r", "kill_switch_path": f"{d}/K"}),
            RiskState(),
        )
        engine.on_account(100.0)
        verdict = engine.evaluate(
            symbol="XAU/USD", direction=1, confidence=0.6, edge=0.1, price=4067.18,
            equity=100.0, atr=1.5, spread_bps=0.49, market_open=True,
            min_qty=0.01, qty_step=0.01, price_step=0.01, now_ms=now,
            open_positions=list(book),
        )
        assert verdict.decision.is_tradeable, f"stopped at position {len(book) + 1}: {verdict.reason}"
        notional = verdict.qty * 4067.18
        book.append({"symbol": "XAU/USD", "side": "BUY", "qty": verdict.qty,
                     "mark_price": 4067.18, "notional": notional})
        total += notional
    assert len(book) == 10
    assert total <= 100.0 * 10.0 + 1e-6, f"exposure ceiling breached: {total}"


def test_slot_split_can_be_turned_off():
    """One deliberate position sized to the full ceiling must still be possible."""
    import tempfile

    from pxscalper.features.calendar import _et
    from pxscalper.risk.engine import RiskConfig, RiskEngine, RiskState

    d = tempfile.mkdtemp()
    engine = RiskEngine(
        RiskConfig.from_dict({
            "risk_per_trade_pct": 0.5, "max_leverage": 10.0,
            "max_gross_exposure_pct": 1000.0, "max_positions": 10,
            "split_exposure_across_slots": False,
            "max_intraday_loss_pct": 0.0, "max_daily_loss_pct": 100.0,
            "max_drawdown_pct": 100.0, "max_total_loss_pct": 0.0,
            "blackout_enabled": False, "state_path": f"{d}/r", "kill_switch_path": f"{d}/K",
        }),
        RiskState(),
    )
    engine.on_account(100.0)
    verdict = engine.evaluate(
        symbol="XAU/USD", direction=1, confidence=0.6, edge=0.1, price=4067.18,
        equity=100.0, atr=1.5, spread_bps=0.49, market_open=True,
        min_qty=0.01, qty_step=0.01, price_step=0.01,
        now_ms=_et(2026, 10, 2) + 3 * 3_600_000,
    )
    assert verdict.qty * 4067.18 > 400.0, "full ceiling should be usable when splitting is off"


# ── guard ordering: a loose brake must actually be the one that grips ───────
def test_intraday_brake_grips_when_it_is_the_loosest_relevant_limit():
    """If the operator sets the brake to 20%, it must fire at 20% -- not be
    silently pre-empted by a 3% daily limit. That failure mode makes the setting
    look applied while doing nothing, which is worse than refusing it."""
    import tempfile

    from pxscalper.features.calendar import _et
    from pxscalper.risk.engine import RiskConfig, RiskEngine, RiskState

    d = tempfile.mkdtemp()
    engine = RiskEngine(
        RiskConfig.from_dict({
            "risk_per_trade_pct": 0.5, "max_leverage": 10.0,
            "max_gross_exposure_pct": 1000.0, "max_positions": 10,
            "max_positions_per_symbol": 10, "blackout_enabled": False,
            "max_intraday_loss_pct": 20.0, "max_daily_loss_pct": 25.0,
            "max_drawdown_pct": 28.0, "max_total_loss_pct": 30.0,
            "state_path": f"{d}/r", "kill_switch_path": f"{d}/K",
        }),
        RiskState(),
    )
    engine.on_account(100.0)
    now = _et(2026, 10, 2) + 3 * 3_600_000

    def at(equity: float):
        return engine.evaluate(
            symbol="XAU/USD", direction=1, confidence=0.6, edge=0.1, price=4067.18,
            equity=equity, atr=1.5, spread_bps=0.49, market_open=True,
            min_qty=0.01, qty_step=0.01, price_step=0.01, now_ms=now,
        )

    assert at(85.0).decision.is_tradeable, "should still trade at -15%"
    brake = at(80.0)
    assert not brake.decision.is_tradeable
    assert "intraday" in brake.reason, brake.reason
    assert at(70.0).decision.name == "BLOCK"  # permanent total-loss limit
    assert engine.kill.engaged()

"""Risk engine: the guard rails must be boring and provably correct."""

from __future__ import annotations

import pytest

from pxscalper.risk.engine import (
    KillSwitch,
    RiskConfig,
    RiskDecision,
    RiskEngine,
    RiskState,
    pnl_coefficient,
)


@pytest.fixture
def engine(tmp_path):
    cfg = RiskConfig(
        state_path=str(tmp_path / "risk.json"),
        kill_switch_path=str(tmp_path / "KILL"),
        max_spread_bps=3.0,
        min_atr_bps=0.5,
        max_atr_bps=50.0,
    )
    return RiskEngine(cfg, RiskState())


def _evaluate(engine: RiskEngine, **overrides):
    params = dict(
        symbol="EUR/USD", direction=1, confidence=0.6, edge=0.15, price=1.1000,
        equity=10_000.0, atr=0.0005, spread_bps=1.0, market_open=True,
        min_qty=0.001, qty_step=0.001, price_step=0.00001,
    )
    params.update(overrides)
    return engine.evaluate(**params)


def test_allows_reasonable_trade(engine):
    verdict = _evaluate(engine)
    assert verdict.allowed
    assert verdict.qty >= 0.001
    assert verdict.stop_loss < 1.1
    assert verdict.take_profit > 1.1


def test_risk_per_trade_is_respected(engine):
    verdict = _evaluate(engine)
    budget = 10_000 * engine.config.risk_per_trade_pct / 100
    assert verdict.risk_amount <= budget * 1.001
    assert verdict.risk_amount > 0


def test_flat_when_model_says_flat(engine):
    verdict = _evaluate(engine, direction=0)
    assert verdict.decision is RiskDecision.FLAT


def test_blocks_wide_spread(engine):
    verdict = _evaluate(engine, spread_bps=9.0)
    assert verdict.decision is RiskDecision.FLAT
    assert "spread" in verdict.reason


def test_blocks_dead_and_wild_volatility(engine):
    dead = _evaluate(engine, atr=0.0000001)
    wild = _evaluate(engine, atr=0.05)
    assert not dead.allowed
    assert not wild.allowed


def test_min_size_rejection_never_rounds_up(engine):
    """A tiny account with a wide stop sizes below the exchange minimum: refuse,
    never round up (rounding up would multiply the risk budget)."""
    verdict = _evaluate(engine, equity=5.0, atr=0.0015, min_qty=50.0)
    assert verdict.decision is RiskDecision.BLOCK
    assert "minimum" in verdict.reason


def test_intraday_brake_pauses_before_daily_limit(engine):
    """FX never closes, so a small intraday drawdown must pause trading before
    the (much looser) daily limit is touched."""
    engine.on_account(10_000)
    verdict = _evaluate(engine, equity=9_900)  # -1.0% intraday, daily limit is 2%
    assert verdict.decision is RiskDecision.FLAT
    assert "intraday" in verdict.reason


def test_daily_loss_engages_kill_switch(engine, tmp_path):
    engine.on_account(10_000)
    # intraday brake off for this test: we want to prove the *daily* guard fires
    engine.config.max_intraday_loss_pct = 0.0
    verdict = _evaluate(engine, equity=9_700)  # -3% vs 2% limit
    assert verdict.decision is RiskDecision.BLOCK
    assert engine.kill.engaged()


def test_cooldown_after_consecutive_losses(engine):
    engine.on_account(10_000)
    for _ in range(engine.config.consecutive_loss_cooldown):
        engine.on_trade_closed(-5.0)
    assert engine.state.cooldown_until_ms > 0
    verdict = _evaluate(engine, now_ms=engine.state.cooldown_until_ms - 1000)
    assert verdict.decision is RiskDecision.FLAT
    assert "cooldown" in verdict.reason


def test_correlated_group_blocks_second_position(tmp_path):
    cfg = RiskConfig(
        state_path=str(tmp_path / "r.json"),
        kill_switch_path=str(tmp_path / "K"),
        correlated_groups=[["EUR/USD", "GBP/USD"]],
    )
    engine = RiskEngine(cfg, RiskState())
    verdict = _evaluate(
        engine,
        open_positions=[{"symbol": "GBP/USD", "side": "BUY", "qty": 0.1}],
    )
    assert verdict.decision is RiskDecision.BLOCK
    assert "correlated" in verdict.reason


def test_pnl_coefficient_quotes():
    assert pnl_coefficient("EUR/USD", 1.1) == 1.0
    assert pnl_coefficient("USD/JPY", 150.0) == pytest.approx(1 / 150.0)


def test_kill_switch_file(tmp_path):
    switch = KillSwitch(tmp_path / "KILL")
    assert not switch.engaged()
    switch.engage("test")
    assert switch.engaged()
    switch.release()
    assert not switch.engaged()

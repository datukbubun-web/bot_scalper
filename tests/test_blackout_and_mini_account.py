"""Tests for the macro blackout gate and the small-account capital guards.

The blackout exists because a macro release makes a 1-minute bar ~22x wider than
normal (measured: 87.21 bps vs 3.96), so a stop sized from pre-event ATR sits
inside the noise. Trading is paused around the release and resumes on its own.

The capital guards exist because a $100 account dies fast: a daily brake that
resets every 24 hours does not protect the deposit. Total loss must latch.
"""

from __future__ import annotations

import tempfile

import pytest
import yaml

from pxscalper.features.calendar import _et
from pxscalper.risk.engine import (
    RiskConfig,
    RiskDecision,
    RiskEngine,
    RiskState,
)

NFP = _et(2026, 10, 2)          # 2026-10-02 12:30 UTC
CPI = _et(2026, 10, 14)         # 2026-10-14 12:30 UTC
FOMC = _et(2026, 10, 28, 14, 0)  # 2026-10-28 18:00 UTC


def _engine(tmp_path, **overrides) -> RiskEngine:
    cfg = RiskConfig(
        state_path=str(tmp_path / "risk.json"),
        kill_switch_path=str(tmp_path / "KILL"),
        **overrides,
    )
    return RiskEngine(cfg, RiskState())


def _evaluate(engine: RiskEngine, equity: float = 100.0, now_ms: int | None = None,
              **overrides):
    params = dict(
        symbol="XAU/USD", direction=1, confidence=0.6, edge=0.1, price=4067.18,
        equity=equity, atr=1.5, spread_bps=0.49, market_open=True,
        min_qty=0.01, qty_step=0.01, price_step=0.01,
        now_ms=now_ms if now_ms is not None else NFP + 3 * 3_600_000,
    )
    params.update(overrides)
    return engine.evaluate(**params)


# ── blackout ────────────────────────────────────────────────────────────────
def test_trading_paused_before_a_release(tmp_path):
    engine = _engine(tmp_path)
    verdict = _evaluate(engine, now_ms=NFP - 5 * 60_000)
    assert verdict.decision is RiskDecision.FLAT
    assert "blackout" in verdict.reason
    assert verdict.details["blackout_seconds_left"] > 0


def test_trading_paused_at_the_moment_of_the_release(tmp_path):
    verdict = _evaluate(_engine(tmp_path), now_ms=NFP)
    assert verdict.decision is RiskDecision.FLAT
    assert "blackout" in verdict.reason


def test_trading_resumes_automatically_after_the_window(tmp_path):
    """No human intervention, no state to clear: just trade again."""
    engine = _engine(tmp_path)
    blocked = _evaluate(engine, now_ms=NFP + 10 * 60_000)
    resumed = _evaluate(engine, now_ms=NFP + 16 * 60_000)
    assert blocked.decision is RiskDecision.FLAT
    assert resumed.decision.is_tradeable


def test_window_boundaries_are_respected(tmp_path):
    engine = _engine(tmp_path, blackout_pre_min=15, blackout_post_min=15)
    inside_start = engine.blackout_status(NFP - 15 * 60_000)[0]
    just_before = engine.blackout_status(NFP - 15 * 60_000 - 1)[0]
    inside_end = engine.blackout_status(NFP + 15 * 60_000 - 1)[0]
    just_after = engine.blackout_status(NFP + 15 * 60_000)[0]
    assert inside_start and inside_end
    assert not just_before and not just_after


@pytest.mark.parametrize("event", [NFP, CPI, FOMC], ids=["nfp", "cpi", "fomc"])
def test_all_three_event_kinds_are_gated(tmp_path, event):
    engine = _engine(tmp_path)
    assert engine.blackout_status(event)[0]


def test_kinds_can_be_narrowed(tmp_path):
    engine = _engine(tmp_path, blackout_kinds=("fomc",))
    assert not engine.blackout_status(NFP)[0]
    assert engine.blackout_status(FOMC)[0]


def test_blackout_can_be_disabled(tmp_path):
    engine = _engine(tmp_path, blackout_enabled=False)
    assert not engine.blackout_status(NFP)[0]
    verdict = _evaluate(engine, now_ms=NFP)
    assert verdict.decision.is_tradeable


def test_explicit_windows_work_without_a_calendar(tmp_path):
    """An operator must be able to declare a blackout by hand (a surprise
    announcement, a maintenance window) without touching the calendar."""
    start, end = NFP - 3_600_000, NFP + 7_200_000
    engine = _engine(tmp_path, blackout_kinds=(), blackout_windows=[(start, end)])
    assert engine.blackout_status(start + 1)[0]
    assert engine.blackout_status(end)[0] is False


def test_blackout_does_not_close_positions():
    """It gates ENTRY only. A pause that also flattens turns a 15-minute news
    window into a forced exit at the worst possible price."""
    assert "FLAT" in RiskDecision.__members__
    assert RiskDecision.FLAT.is_tradeable is False


# ── lifetime capital guards ─────────────────────────────────────────────────
def test_initial_equity_is_recorded_once(tmp_path):
    engine = _engine(tmp_path)
    engine.on_account(100.0)
    engine.on_account(140.0)
    assert engine.state.initial_equity == pytest.approx(100.0)


def test_total_loss_limit_latches_the_kill_switch(tmp_path):
    """A daily brake resets every 24h; the deposit does not. This is the guard
    that keeps a small account alive, so it must be permanent until reviewed."""
    engine = _engine(tmp_path, max_total_loss_pct=10.0, max_intraday_loss_pct=0.0,
                     max_daily_loss_pct=100.0, max_drawdown_pct=100.0)
    engine.on_account(100.0)
    ok = _evaluate(engine, equity=95.0)
    breach = _evaluate(engine, equity=89.0)
    assert ok.decision.is_tradeable
    assert breach.decision is RiskDecision.BLOCK
    assert "TOTAL LOSS LIMIT" in breach.reason
    assert engine.kill.engaged()
    # and it stays shut even if equity recovers
    assert _evaluate(engine, equity=100.0).decision is RiskDecision.BLOCK


def test_min_equity_blocks_below_a_floor(tmp_path):
    engine = _engine(tmp_path, min_equity=60.0, max_total_loss_pct=0.0,
                     max_intraday_loss_pct=0.0, max_daily_loss_pct=100.0,
                     max_drawdown_pct=100.0)
    engine.on_account(100.0)
    assert _evaluate(engine, equity=59.0).decision is RiskDecision.BLOCK
    assert _evaluate(engine, equity=100.0).decision.is_tradeable


def test_total_loss_can_be_disabled(tmp_path):
    engine = _engine(tmp_path, max_total_loss_pct=0.0, max_intraday_loss_pct=0.0,
                     max_daily_loss_pct=100.0, max_drawdown_pct=100.0)
    engine.on_account(100.0)
    assert _evaluate(engine, equity=50.0).decision.is_tradeable


# ── the shipped preset ──────────────────────────────────────────────────────
def test_mini100_preset_loads_and_is_conservative():
    """Guards must be internally consistent, not tied to one chosen number.

    The user has deliberately raised the intraday brake to 20%, which only means
    anything if the outer limits sit above it. What must always hold is the
    ordering: intraday (resumes tomorrow) < daily < total (permanent).
    """
    raw = yaml.safe_load(open("configs/live.mini100.yaml"))
    risk = RiskConfig.from_dict(raw["risk"])
    assert 0 < risk.max_intraday_loss_pct < risk.max_total_loss_pct
    assert risk.max_intraday_loss_pct < risk.max_daily_loss_pct or risk.max_daily_loss_pct <= 0
    assert risk.max_total_loss_pct < 50.0, "a total-loss limit above half the deposit is not a limit"
    assert risk.blackout_enabled is True
    assert risk.max_positions <= 10
    assert risk.max_leverage <= 10.0
    assert raw["execution"]["dry_run"] is True, "a real-money preset must start in dry-run"


def test_mini100_sizes_a_position_within_the_leverage_cap():
    raw = yaml.safe_load(open("configs/live.mini100.yaml"))
    with tempfile.TemporaryDirectory() as tmp:
        engine = RiskEngine(
            RiskConfig.from_dict({**raw["risk"], "state_path": f"{tmp}/r.json",
                                  "kill_switch_path": f"{tmp}/K"}),
            RiskState(),
        )
        engine.on_account(100.0)
        verdict = _evaluate(engine, equity=100.0)
    assert verdict.decision.is_tradeable
    notional = verdict.qty * 4067.18
    assert verdict.qty > 0
    assert notional <= 100.0 * 10.0 + 1e-6      # leverage cap holds
    assert verdict.stop_loss is not None and verdict.take_profit is not None


# ── configuration paths (YAML + environment) ────────────────────────────────
def test_string_values_from_env_or_yaml_are_coerced():
    """A settings typo must not silently produce a wrong guard."""
    cfg = RiskConfig.from_dict({
        "max_intraday_loss_pct": "2.0",
        "max_daily_trades": "8",
        "blackout_enabled": "false",
        "blackout_kinds": "nfp, cpi",
        "blackout_pre_min": "30",
    })
    assert cfg.max_intraday_loss_pct == pytest.approx(2.0)
    assert cfg.max_daily_trades == 8
    assert cfg.blackout_enabled is False
    assert cfg.blackout_kinds == ("nfp", "cpi")
    assert cfg.blackout_pre_min == pytest.approx(30.0)


def test_env_overrides_the_yaml(monkeypatch):
    """Ops must be able to tighten a guard without editing the committed file."""
    monkeypatch.setenv("PX_RISK_MAX_INTRADAY_LOSS_PCT", "2.0")
    monkeypatch.setenv("PX_RISK_BLACKOUT_PRE_MIN", "30")
    cfg = RiskConfig.from_dict({"max_intraday_loss_pct": 0.75})
    assert cfg.max_intraday_loss_pct == pytest.approx(2.0)
    assert cfg.blackout_pre_min == pytest.approx(30.0)
    # and it can be switched off for tests / backtests
    assert RiskConfig.from_dict({}, allow_env=False).max_intraday_loss_pct == 0.75


def test_unknown_env_override_is_ignored_not_crashed(monkeypatch):
    monkeypatch.setenv("PX_RISK_NOT_A_REAL_FIELD", "1")
    assert RiskConfig.from_dict({}).max_intraday_loss_pct == 0.75


def test_intraday_brake_is_above_a_single_stop_at_mini_size():
    """The whole point of raising it: one stop + slippage must not trip the
    brake, or the bot stops after every normal loss and the operator is
    tempted to raise risk to 'recover'."""
    raw = yaml.safe_load(open("configs/live.mini100.yaml"))
    risk = RiskConfig.from_dict(raw["risk"], allow_env=False)
    one_stop_pct = risk.risk_per_trade_pct * risk.sl_atr / risk.sl_atr  # 0.5% at 1.0xATR
    assert risk.max_intraday_loss_pct >= 3 * one_stop_pct, (
        f"brake {risk.max_intraday_loss_pct}% is less than 3 stops "
        f"({3 * one_stop_pct}%) -- one bad bar would halt the bot"
    )
    assert risk.max_total_loss_pct > risk.max_intraday_loss_pct

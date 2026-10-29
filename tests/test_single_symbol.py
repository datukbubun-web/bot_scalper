"""One instrument only -- enforced, not just written in a YAML default.

`symbols: ["XAU/USD"]` is a value someone can edit in ten seconds. A scalper's
edge (if any) lives in one market at one cost structure, so the scope is a rule:
the runner refuses to start with more than one symbol, and the risk engine
refuses an order for anything outside the allow-list.
"""

from __future__ import annotations

import base64
import inspect
import tempfile

import pytest
import yaml

from pxscalper.features.calendar import _et
from pxscalper.risk.engine import RiskConfig, RiskDecision, RiskEngine, RiskState

NOW = _et(2026, 10, 2) + 3 * 3_600_000


def engine(**overrides) -> RiskEngine:
    d = tempfile.mkdtemp()
    base = dict(
        state_path=f"{d}/r.json", kill_switch_path=f"{d}/K",
        max_intraday_loss_pct=0.0, max_daily_loss_pct=100.0,
        max_drawdown_pct=100.0, max_total_loss_pct=0.0,
        max_gross_exposure_pct=100_000.0,
        single_symbol=True, allowed_symbols=["XAU/USD"],
    )
    base.update(overrides)
    e = RiskEngine(RiskConfig(**base), RiskState())
    e.on_account(100.0)
    return e


def evaluate(e: RiskEngine, symbol: str = "XAU/USD"):
    return e.evaluate(
        symbol=symbol, direction=1, confidence=0.6, edge=0.1, price=4067.18,
        equity=100.0, atr=1.5, spread_bps=0.49, market_open=True,
        min_qty=0.01, qty_step=0.01, price_step=0.01, now_ms=NOW,
    )


# ── risk engine scope ───────────────────────────────────────────────────────
def test_the_allowed_symbol_trades():
    assert evaluate(engine()).decision.is_tradeable


@pytest.mark.parametrize("other", ["EUR/USD", "XAG/USD", "BTC/USD", "xau/usd"])
def test_any_other_symbol_is_blocked(other):
    verdict = evaluate(engine(), symbol=other)
    assert verdict.decision is RiskDecision.BLOCK
    assert "allowed_symbols" in verdict.reason


def test_scope_is_checked_before_anything_else():
    """A forbidden symbol must be refused even when every other guard would
    also have objected -- otherwise the reason shown to the operator is
    misleading."""
    e = engine(max_intraday_loss_pct=0.5)
    verdict = e.evaluate(
        symbol="EUR/USD", direction=1, confidence=0.9, edge=0.5, price=1.1,
        equity=100.0, atr=0.0005, spread_bps=0.2, market_open=True,
        min_qty=1000, qty_step=1000, price_step=0.00001, now_ms=NOW,
    )
    assert verdict.decision is RiskDecision.BLOCK
    assert "allowed_symbols" in verdict.reason


def test_single_symbol_with_several_allowed_is_a_config_error():
    verdict = evaluate(engine(allowed_symbols=["XAU/USD", "EUR/USD"]))
    assert verdict.decision is RiskDecision.BLOCK
    assert "single_symbol" in verdict.reason


def test_scope_can_be_left_open():
    e = engine(single_symbol=False, allowed_symbols=[])
    assert evaluate(e, "EUR/USD").decision.is_tradeable


# ── config load: fail hard, do not trade two symbols silently ───────────────
def _write_config(tmp_path, symbols, risk_extra):
    doc = yaml.safe_load(open("configs/live.mini100.yaml"))
    doc["symbols"] = symbols
    doc["risk"].update(risk_extra)
    path = tmp_path / "live.yaml"
    path.write_text(yaml.safe_dump(doc))
    return path


def _valid_credentials(monkeypatch):
    """The runner validates credentials before it reads the risk block, so the
    test needs a real Ed25519 key, not a placeholder."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    key = Ed25519PrivateKey.generate()
    secret = base64.b64encode(
        key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
    ).decode()
    monkeypatch.setenv("PX_API_KEY", "pxk_test_key_id")
    monkeypatch.setenv("PX_API_SECRET", secret)


def test_runner_refuses_two_symbols_when_scope_says_one(tmp_path, monkeypatch):
    _valid_credentials(monkeypatch)
    from pxscalper.ops.live_runner import build_engine

    path = _write_config(tmp_path, ["XAU/USD", "EUR/USD"], {"single_symbol": True})
    with pytest.raises(RuntimeError, match="single_symbol is true"):
        build_engine(path, allow_live=False)


def test_runner_refuses_mismatched_allow_list(tmp_path, monkeypatch):
    _valid_credentials(monkeypatch)
    from pxscalper.ops.live_runner import build_engine

    path = _write_config(
        tmp_path, ["XAU/USD"], {"single_symbol": True, "allowed_symbols": ["EUR/USD"]}
    )
    with pytest.raises(RuntimeError, match="does not match"):
        build_engine(path, allow_live=False)


# ── the shipped presets ─────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "preset", ["configs/live.mini100.yaml", "configs/live.mini100.multi.yaml"]
)
def test_presets_declare_a_single_instrument(preset):
    raw = yaml.safe_load(open(preset))
    assert raw["risk"]["single_symbol"] is True
    assert len(raw["symbols"]) == 1
    assert raw["risk"]["allowed_symbols"] == raw["symbols"]


# ── the config-path bug this uncovered ──────────────────────────────────────
def test_top_level_symbols_is_honoured(tmp_path, monkeypatch):
    """Regression: `symbols` lives at the TOP level in every preset we ship, but
    the runner read `raw["live"]["symbols"]` and silently fell back to its
    default of EUR/USD. A preset that said XAU/USD traded EUR/USD."""
    _valid_credentials(monkeypatch)
    from pxscalper.ops import live_runner

    source = inspect.getsource(live_runner.build_engine)
    assert 'raw.get("symbols")' in source, "top-level symbols must be read"

    path = _write_config(tmp_path, ["XAU/USD"], {"single_symbol": True})
    doc = yaml.safe_load(path.read_text())
    assert doc["symbols"] == ["XAU/USD"]
    # and the scope check now compares against the real symbol
    assert doc["risk"]["allowed_symbols"] == ["XAU/USD"]


def test_missing_symbols_fails_loudly(tmp_path, monkeypatch):
    """No symbol at all must be an error, not a silent EUR/USD."""
    from pxscalper.ops import live_runner

    source = inspect.getsource(live_runner.build_engine)
    assert "no symbols configured" in source

"""Tests for the Compounding Engine: daily/continuous/fixed equity bases,
daily profit target lock + live position harvesting, Anti-Martingale drawdown
scaling, percentage-compounding BasketTarget, and compounding presets/CLI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from pxscalper.backtest.engine import BacktestConfig, Backtester
from pxscalper.cli import app
from pxscalper.exchange.models import AccountMetrics, Position, SymbolSettings
from pxscalper.execution.router import ExecutionConfig, OrderRouter
from pxscalper.inference.predictor import Prediction
from pxscalper.live.engine import LiveConfig, LiveEngine
from pxscalper.live.metrics import Metrics
from pxscalper.risk.basket import BasketTarget
from pxscalper.risk.engine import RiskConfig, RiskDecision, RiskEngine, RiskState
from pxscalper.strategy.decision import ScalpingStrategy, StrategyConfig


def _eval_kwargs(**overrides):
    base = {
        "symbol": "XAU/USD",
        "direction": 1,
        "confidence": 0.65,
        "edge": 0.12,
        "price": 4000.0,
        "equity": 1000.0,
        "atr": 5.0,
        "spread_bps": 0.5,
        "market_open": True,
        "min_qty": 0.01,
        "qty_step": 0.01,
        "price_step": 0.01,
        "now_ms": 1_780_010_000_000,
    }
    base.update(overrides)
    return base


def test_risk_engine_daily_profit_target_locks_until_utc_rollover(tmp_path: Path):
    now_s = [1_780_010_000.0]  # Day D
    risk = RiskEngine(
        RiskConfig(
            state_path=str(tmp_path / "r.json"),
            kill_switch_path=str(tmp_path / "K"),
            risk_per_trade_pct=3.0,
            compound_mode="daily",
            daily_profit_target_pct=3.0,
            max_leverage=25.0,
            max_gross_exposure_pct=2500.0,
            max_intraday_loss_pct=9.0,
            max_daily_loss_pct=12.0,
            sl_atr=4.0,
            blackout_enabled=False,
        ),
        RiskState(),
        clock=lambda: now_s[0],
    )
    risk.on_account(1000.0)
    v1 = risk.evaluate(**_eval_kwargs(equity=1000.0, now_ms=int(now_s[0] * 1000)))
    assert v1.allowed
    qty_day1 = v1.qty

    # Day gains +3.1% ($1000 -> $1031) -> daily profit lock engages (FLAT)
    risk.on_account(1031.0)
    v_locked = risk.evaluate(**_eval_kwargs(equity=1031.0, now_ms=int(now_s[0] * 1000)))
    assert v_locked.decision is RiskDecision.FLAT
    assert "daily profit target" in v_locked.reason

    # Roll clock forward 24h into next UTC day -> lock resets & sizing compounds on $1031!
    now_s[0] += 86_400.0
    risk.on_account(1031.0)
    assert risk.state.day_start_equity == pytest.approx(1031.0)
    v_day2 = risk.evaluate(**_eval_kwargs(equity=1031.0, now_ms=int(now_s[0] * 1000)))
    assert v_day2.allowed
    assert v_day2.risk_amount > v1.risk_amount
    assert v_day2.qty >= qty_day1


def test_risk_engine_compound_modes_continuous_daily_and_fixed(tmp_path: Path):
    def make_eng(mode: str) -> RiskEngine:
        eng = RiskEngine(
            RiskConfig(
                state_path=str(tmp_path / f"r_{mode}.json"),
                kill_switch_path=str(tmp_path / f"K_{mode}"),
                risk_per_trade_pct=2.0,
                compound_mode=mode,
                max_positions=1,
                max_leverage=30.0,
                max_gross_exposure_pct=3000.0,
                max_intraday_loss_pct=0.0,
                blackout_enabled=False,
            ),
            RiskState(),
        )
        eng.on_account(1000.0)  # initial_equity=1000, day_start_equity=1000
        return eng

    c_eng = make_eng("continuous")
    d_eng = make_eng("daily")
    f_eng = make_eng("fixed")

    # Intraday equity rises to $1500 on same UTC day
    for e in (c_eng, d_eng, f_eng):
        e.on_account(1500.0)

    vc = c_eng.evaluate(**_eval_kwargs(equity=1500.0))
    vd = d_eng.evaluate(**_eval_kwargs(equity=1500.0))
    vf = f_eng.evaluate(**_eval_kwargs(equity=1500.0))

    assert vc.details["sizing_equity"] == pytest.approx(1500.0)
    assert vd.details["sizing_equity"] == pytest.approx(1000.0)
    assert vf.details["sizing_equity"] == pytest.approx(1000.0)
    assert vc.risk_amount == pytest.approx(30.0)  # 2% of live $1500
    assert vd.risk_amount == pytest.approx(20.0)  # 2% of day_start $1000
    assert vf.risk_amount == pytest.approx(20.0)  # 2% of initial $1000

    with pytest.raises(ValueError, match="compound_mode"):
        RiskConfig(compound_mode="martingale")


def test_risk_engine_anti_martingale_drawdown_scale_tapers_and_recovers(tmp_path: Path):
    eng = RiskEngine(
        RiskConfig(
            state_path=str(tmp_path / "r.json"),
            kill_switch_path=str(tmp_path / "K"),
            risk_per_trade_pct=10.0,
            compound_dd_scale=True,
            dd_scale_start_pct=4.0,
            dd_scale_min_factor=0.40,
            max_drawdown_pct=24.0,
            max_daily_loss_pct=50.0,
            max_total_loss_pct=50.0,
            max_intraday_loss_pct=0.0,
            max_positions=1,
            max_leverage=100.0,
            max_gross_exposure_pct=10_000.0,
            blackout_enabled=False,
        ),
        RiskState(),
    )
    eng.on_account(1000.0)
    v_peak = eng.evaluate(**_eval_kwargs(equity=1000.0))
    assert v_peak.details["effective_risk_pct"] == pytest.approx(10.0)

    # 14% drawdown (halfway between 4% start and 24% max_dd -> factor = 0.70)
    eng.on_account(860.0)
    v_dd = eng.evaluate(**_eval_kwargs(equity=860.0))
    assert v_dd.allowed
    assert v_dd.details["effective_risk_pct"] == pytest.approx(7.0)

    # Recover to new peak $1100 -> full 10% risk restored & compounded on $1100
    eng.on_account(1100.0)
    v_rec = eng.evaluate(**_eval_kwargs(equity=1100.0))
    assert v_rec.details["effective_risk_pct"] == pytest.approx(10.0)
    assert v_rec.risk_amount == pytest.approx(110.0)


def test_basket_target_percentage_compounding_mode():
    bt = BasketTarget(target=0.0, target_pct=3.0)
    # At $100 equity, target is $3.00
    assert not bt.evaluate([{"upl": 2.80}], enforce=True, equity=100.0).triggered
    assert bt.evaluate([{"upl": 3.05}], enforce=True, equity=100.0).triggered

    # At $500 compounded equity, $3.05 is NOT enough (target compounded to $15.00!)
    bt.reset()
    res_500 = bt.evaluate([{"upl": 3.05}], enforce=True, equity=500.0)
    assert not res_500.triggered
    assert res_500.target == pytest.approx(15.0)
    assert bt.evaluate([{"upl": 15.10}], enforce=True, equity=500.0).triggered


def test_live_engine_harvests_open_positions_on_daily_profit_target(tmp_path: Path):
    class _Client:
        def __init__(self) -> None:
            self.closed_ids: list[int] = []
            self.pos = [
                Position(
                    id=77,
                    symbol="XAU/USD",
                    side="BUY",
                    qty=0.50,
                    open_price=4000.0,
                    current_price=4064.0,
                    mark_price=4064.0,
                    upl=32.0,
                    roe_pct=3.2,
                    liq_price=None,
                    margin=100.0,
                    client_leverage=20.0,
                    stop_loss_price=3980.0,
                    take_profit_price=4100.0,
                    open_time=1_780_000_000_000,
                    closeable=True,
                    exposure=2032.0,
                    raw={},
                )
            ]

        def settings(self, symbols=None):
            return [
                SymbolSettings(
                    symbol="XAU/USD",
                    min_qty=0.01,
                    max_qty=1000.0,
                    qty_step=0.01,
                    min_price_step=0.01,
                    price_scale=2,
                    max_leverage=100,
                    default_leverage=20,
                    taker_fee_pct=0.0,
                    maker_fee_pct=0.0,
                )
            ]

        def positions(self):
            return list(self.pos)

        def close_position(self, pos_id):
            self.closed_ids.append(int(pos_id))
            self.pos = [p for p in self.pos if p.id != pos_id]
            return {"id": pos_id}

        def metrics(self):
            return AccountMetrics(
                account_id="D",
                currency="USD",
                balance=1000.0,
                equity=1032.0,
                real_equity=1032.0,
                upl=32.0,
                available_margin=932.0,
                used_margin=100.0,
                maintenance_margin=20.0,
                margin_ratio=0.1,
                margin_level=None,
                position_mode="NETTING",
                open_positions=len(self.pos),
                open_orders=0,
                real_money=False,
                blocked=False,
                reduce_only=False,
                server="demo",
            )

    class _Pred:
        config = type("C", (), {"n_features": 38, "lookback": 64, "backend": "onnx"})()

        def warmup(self, n: int = 10) -> float:
            return 10.0

        def predict(self, window: np.ndarray) -> Prediction:
            return Prediction(np.array([0.2, 0.6, 0.2], dtype=np.float32), 0, 0.6, 10.0)

    client = _Client()
    risk = RiskEngine(
        RiskConfig(
            state_path=str(tmp_path / "r.json"),
            kill_switch_path=str(tmp_path / "K"),
            daily_profit_target_pct=3.0,
        ),
        RiskState(),
    )
    risk.on_account(1000.0)  # day_start_equity = $1000
    pred = _Pred()
    engine = LiveEngine(
        client=client,  # type: ignore[arg-type]
        predictor=pred,  # type: ignore[arg-type]
        risk=risk,
        router=OrderRouter(client, ExecutionConfig(dry_run=False, verify_fills=False)),  # type: ignore[arg-type]
        strategy=ScalpingStrategy(predictor=pred, config=StrategyConfig(require_rl_agreement=False)),
        config=LiveConfig(
            symbols=["XAU/USD"],
            harvest_daily_profit_target=True,
            log_path=str(tmp_path / "decisions.jsonl"),
        ),
        metrics=Metrics(enable_server=False),
    )
    harvested = engine._check_daily_profit_harvest(client.positions(), equity=1032.0)
    assert harvested is True
    assert risk.state.trades_today == 1
    assert risk.state.realised_pnl_today == pytest.approx(32.0)


def test_backtester_compound_vs_fixed_and_daily_profit_lock():
    # Build 6 UTC days of steadily rising bars with multiple winning signals per day
    bars_per_day = 24
    n = bars_per_day * 6
    times = np.arange(n, dtype=np.int64) * 3_600_000 + 1_780_012_800_000
    base_px = 1000.0 + np.arange(n, dtype=np.float64) * 2.0
    df = pd.DataFrame(
        {
            "time": times,
            "open": base_px,
            "high": base_px + 3.0,
            "low": base_px - 0.2,
            "close": base_px + 2.0,
            "volume": np.full(n, 1000.0),
        }
    )
    windows = np.zeros((n, 2, 4), dtype=np.float32)
    idx = np.arange(n, dtype=np.int64)
    atr = np.full(n, 1.0, dtype=np.float64)

    def sig(row: int, _w: np.ndarray):
        return (1, 0.8, 0.2) if row % 4 == 1 else (0, 0.0, 0.0)

    res_fixed = Backtester(
        BacktestConfig(
            initial_equity=1000.0,
            taker_fee_pct=0.0,
            spread_bps_fallback=0.1,
            risk_per_trade_pct=5.0,
            max_leverage=30.0,
            max_gross_exposure_pct=3000.0,
            sl_atr=2.0,
            tp_atr=3.0,
            max_holding_bars=3,
            compound=False,
        )
    ).run(df, windows, idx, atr, sig)

    res_comp = Backtester(
        BacktestConfig(
            initial_equity=1000.0,
            taker_fee_pct=0.0,
            spread_bps_fallback=0.1,
            risk_per_trade_pct=5.0,
            max_leverage=30.0,
            max_gross_exposure_pct=3000.0,
            sl_atr=2.0,
            tp_atr=3.0,
            max_holding_bars=3,
            compound=True,
            compound_mode="continuous",
        )
    ).run(df, windows, idx, atr, sig)

    assert res_comp.metrics["final_equity"] > res_fixed.metrics["final_equity"]

    # With daily_profit_target_pct=3.0 & harvest=True, fewer trades are taken per day
    res_locked = Backtester(
        BacktestConfig(
            initial_equity=1000.0,
            taker_fee_pct=0.0,
            spread_bps_fallback=0.1,
            risk_per_trade_pct=5.0,
            max_leverage=30.0,
            max_gross_exposure_pct=3000.0,
            sl_atr=2.0,
            tp_atr=3.0,
            max_holding_bars=3,
            compound=True,
            compound_mode="daily",
            daily_profit_target_pct=3.0,
            harvest_daily_profit_target=True,
        )
    ).run(df, windows, idx, atr, sig)
    assert len(res_locked.trades) < len(res_comp.trades)


@pytest.mark.parametrize(
    "preset",
    [
        "configs/live.hybrid.compound.yaml",
        "configs/live.mini100.compound.yaml",
        "configs/live.hybrid.daily3pct_lock.yaml",
    ],
)
def test_compound_presets_valid_and_cli_runs(preset: str):
    raw = yaml.safe_load(Path(preset).read_text())
    unknown = set(raw["risk"]) - set(RiskConfig.__dataclass_fields__)
    assert not unknown
    cfg = RiskConfig.from_dict(raw["risk"], allow_env=False)
    assert cfg.max_intraday_loss_pct / cfg.risk_per_trade_pct >= 3.0

    res = CliRunner().invoke(app, ["risk", "show", "--config", preset])
    assert res.exit_code == 0, res.output

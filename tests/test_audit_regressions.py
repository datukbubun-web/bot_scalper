"""Regression tests for deep repository audit fixes (live SL/TP accounting,
train/serve MTF buffer parity, multi-symbol manifest guard, export metadata
preservation, intraday brake sizing, and config/.env hygiene).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from pxscalper.data.pipeline import DataPipeline
from pxscalper.exchange.models import AccountMetrics, Fill, Position, SymbolSettings
from pxscalper.execution.router import ExecutionConfig, ExecutionResult, OrderRouter
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer
from pxscalper.inference.export import export_all
from pxscalper.inference.predictor import Prediction
from pxscalper.live.engine import LiveConfig, LiveEngine
from pxscalper.live.metrics import Metrics
from pxscalper.ops.backtest_runner import (
    load_context_frames,
    load_htf_frames,
    run_backtest_from_config,
)
from pxscalper.ops.live_runner import build_engine
from pxscalper.risk.engine import RiskConfig, RiskDecision, RiskEngine, RiskState
from pxscalper.strategy.decision import ScalpingStrategy, StrategyConfig
from pxscalper.utils import RepoPaths


class _DummyPredictor:
    def __init__(self, n_features: int = 38, lookback: int = 64) -> None:
        self.config = type("C", (), {"n_features": n_features, "lookback": lookback, "backend": "onnx"})()

    def warmup(self, n: int = 10) -> float:
        return 50.0

    def predict(self, window: np.ndarray) -> Prediction:
        return Prediction(
            probs=np.array([0.2, 0.6, 0.2], dtype=np.float32),
            direction=0,
            confidence=0.6,
            latency_us=42.0,
        )


class _FakeExchangeClient:
    def __init__(self) -> None:
        self.live_positions: list[Position] = []
        self.fill_rows: list[Fill] = []

    def sync_clock(self) -> float:
        return 0.0

    def settings(self, symbols=None):
        return [
            SymbolSettings(
                symbol="XAU/USD",
                min_qty=0.01,
                max_qty=1000.0,
                qty_step=0.01,
                min_price_step=0.01,
                price_scale=2,
                max_leverage=20,
                default_leverage=5,
                taker_fee_pct=0.0,
                maker_fee_pct=0.0,
            )
        ]

    def bars(self, symbol: str, bar_size: str = "5M", limit: int = 500, end_ms=None):
        return []

    def metrics(self) -> AccountMetrics:
        return AccountMetrics(
            account_id="DEMO-1",
            currency="USD",
            balance=10_000.0,
            equity=10_000.0,
            real_equity=10_000.0,
            upl=0.0,
            available_margin=10_000.0,
            used_margin=0.0,
            maintenance_margin=0.0,
            margin_ratio=0.0,
            margin_level=None,
            position_mode="NETTING",
            open_positions=len(self.live_positions),
            open_orders=0,
            real_money=False,
            blocked=False,
            reduce_only=False,
            server="demo",
        )

    def positions(self) -> list[Position]:
        return list(self.live_positions)

    def fills(self, symbol: str | None = None, limit: int = 25, offset: int = 0) -> list[Fill]:
        if symbol:
            return [f for f in self.fill_rows if f.symbol == symbol][:limit]
        return self.fill_rows[:limit]


def _pos(pid: int = 991, upl: float = -8.0, open_time: int = 1_780_000_000_000) -> Position:
    return Position(
        id=pid,
        symbol="XAU/USD",
        side="BUY",
        qty=0.10,
        open_price=4050.0,
        current_price=4040.0,
        mark_price=4040.0,
        upl=upl,
        roe_pct=-1.0,
        liq_price=None,
        margin=81.0,
        client_leverage=5.0,
        stop_loss_price=4037.5,
        take_profit_price=4070.0,
        open_time=open_time,
        closeable=True,
        exposure=404.0,
        raw={},
    )


def test_live_engine_reconciles_exchange_closed_sl_tp_positions(tmp_path: Path):
    client = _FakeExchangeClient()
    client.live_positions = [_pos(991, upl=-8.0)]
    now_holder = [1_780_000_600.0]

    risk = RiskEngine(
        RiskConfig(
            state_path=str(tmp_path / "risk.json"),
            kill_switch_path=str(tmp_path / "KILL"),
            consecutive_loss_cooldown=2,
            cooldown_minutes=30,
        ),
        RiskState(),
        clock=lambda: now_holder[0],
    )
    pred = _DummyPredictor()
    engine = LiveEngine(
        client=client,  # type: ignore[arg-type]
        predictor=pred,  # type: ignore[arg-type]
        risk=risk,
        router=OrderRouter(client, ExecutionConfig(dry_run=True)),  # type: ignore[arg-type]
        strategy=ScalpingStrategy(predictor=pred, config=StrategyConfig(require_rl_agreement=False)),
        config=LiveConfig(
            symbols=["XAU/USD"],
            bar_size="5M",
            warmup_bars=10,
            log_path=str(tmp_path / "decisions.jsonl"),
        ),
        metrics=Metrics(enable_server=False),
        clock=lambda: now_holder[0],
    )
    engine.prepare()
    assert 991 in engine._known_positions

    # Broker triggers Stop-Loss between cycles: position disappears & closing Fill lands
    client.live_positions = []
    client.fill_rows = [
        Fill(
            id=5001,
            order_id=7001,
            symbol="XAU/USD",
            side="SELL",
            type="TAKER",
            order_type="STOP_LOSS",
            qty=0.10,
            price=4037.5,
            fee=0.0,
            rpl=-12.5,
            time=1_780_000_300_000,
        )
    ]
    engine._reconcile_closed_positions(client.positions())

    assert risk.state.trades_today == 1
    assert risk.state.realised_pnl_today == pytest.approx(-12.5)
    assert risk.state.consecutive_losses == 1
    assert engine._last_loss_s == pytest.approx(now_holder[0])

    # Subsequent poll with empty positions must NOT double-count
    engine._reconcile_closed_positions(client.positions())
    assert risk.state.trades_today == 1

    events = [json.loads(line)["event"] for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert events.count("trade_closed") == 1


def test_after_close_ignores_failed_close_execution(tmp_path: Path):
    client = _FakeExchangeClient()
    p = _pos(992, upl=-15.0)
    client.live_positions = [p]
    risk = RiskEngine(
        RiskConfig(
            state_path=str(tmp_path / "risk.json"),
            kill_switch_path=str(tmp_path / "KILL"),
        ),
        RiskState(),
    )
    pred = _DummyPredictor()
    engine = LiveEngine(
        client=client,  # type: ignore[arg-type]
        predictor=pred,  # type: ignore[arg-type]
        risk=risk,
        router=OrderRouter(client, ExecutionConfig(dry_run=True)),  # type: ignore[arg-type]
        strategy=ScalpingStrategy(predictor=pred, config=StrategyConfig(require_rl_agreement=False)),
        config=LiveConfig(symbols=["XAU/USD"], log_path=str(tmp_path / "decisions.jsonl")),
        metrics=Metrics(enable_server=False),
    )
    engine._known_positions[p.id] = p
    failed = ExecutionResult(
        ok=False, intent="CLOSE", symbol="XAU/USD", qty=0.10, price=4040.0, error="503 timeout"
    )
    engine._after_close(p, failed)

    assert risk.state.trades_today == 0
    assert risk.state.consecutive_losses == 0
    assert p.id in engine._known_positions


def test_risk_engine_daily_trade_budget_counts_intraday_open_positions(tmp_path: Path):
    now_ms = 1_780_050_000_000
    day_start_ms = (now_ms // 86_400_000) * 86_400_000
    risk = RiskEngine(
        RiskConfig(
            state_path=str(tmp_path / "r.json"),
            kill_switch_path=str(tmp_path / "K"),
            max_daily_trades=4,
            max_positions=10,
            max_positions_per_symbol=10,
            max_gross_exposure_pct=10_000.0,
            max_intraday_loss_pct=0.0,
            blackout_enabled=False,
        ),
        RiskState(),
        clock=lambda: now_ms / 1000.0,
    )
    risk.on_account(10_000.0)
    risk.on_trade_closed(10.0)
    risk.on_trade_closed(10.0)  # 2 closed today

    # Carry-over position opened yesterday should NOT consume today's budget
    yesterday_pos = _pos(1, open_time=day_start_ms - 3_600_000)
    today_pos_1 = _pos(2, open_time=day_start_ms + 60_000)
    v_allow = risk.evaluate(
        symbol="XAU/USD",
        direction=1,
        confidence=0.6,
        edge=0.1,
        price=4050.0,
        equity=10_000.0,
        atr=2.0,
        spread_bps=0.5,
        market_open=True,
        min_qty=0.01,
        qty_step=0.01,
        price_step=0.01,
        open_positions=[yesterday_pos, today_pos_1],  # type: ignore[list-item]
        now_ms=now_ms,
    )
    assert v_allow.allowed  # 2 closed + 1 opened today = 3 < 4

    # Adding a 2nd position opened today hits 2 closed + 2 open today == 4 -> BLOCK
    today_pos_2 = _pos(3, open_time=day_start_ms + 120_000)
    v_block = risk.evaluate(
        symbol="XAU/USD",
        direction=1,
        confidence=0.6,
        edge=0.1,
        price=4050.0,
        equity=10_000.0,
        atr=2.0,
        spread_bps=0.5,
        market_open=True,
        min_qty=0.01,
        qty_step=0.01,
        price_step=0.01,
        open_positions=[yesterday_pos, today_pos_1, today_pos_2],  # type: ignore[list-item]
        now_ms=now_ms,
    )
    assert v_block.decision is RiskDecision.BLOCK
    assert "daily trade budget exhausted" in v_block.reason


def test_hybrid_1m_and_5m_live_buffers_match_batch_features():
    pipe = DataPipeline(cache_dir=Path("data"))
    client = _FakeExchangeClient()
    risk = RiskEngine(RiskConfig())
    pred = _DummyPredictor()

    for mdir, bs, warmup, tol in [
        (Path("artifacts/xau_hybrid_5M"), "5M", 1200, 0.02),
        (Path("artifacts/xau_hybrid_1M"), "1M", 2200, 1e-5),
    ]:
        fcfg = FeatureConfig.load(mdir / "features.json")
        eng = FeatureEngineer(fcfg)
        bars = pipe.load("XAU/USD", bs).tail(4000).reset_index(drop=True)
        ctx = load_context_frames(pipe, eng, bar_size=bs)
        htf_b, htf_c = load_htf_frames(pipe, eng, "XAU/USD", ltf_bar_size=bs)

        norm_batch = eng.transform_normalized(bars, context=ctx, htf_bars=htf_b, htf_context=htf_c)
        w_batch = np.transpose(norm_batch.to_numpy(dtype=np.float32)[-64:], (1, 0))

        le = LiveEngine(
            client=client,  # type: ignore[arg-type]
            predictor=pred,  # type: ignore[arg-type]
            risk=risk,
            router=OrderRouter(client, ExecutionConfig(dry_run=True)),  # type: ignore[arg-type]
            strategy=ScalpingStrategy(predictor=pred),
            config=LiveConfig(symbols=["XAU/USD"], bar_size=bs, warmup_bars=warmup),
            feature_config=fcfg,
            metrics=Metrics(enable_server=False),
        )
        live_bars = bars.tail(warmup + 200).reset_index(drop=True)
        live_ctx = (
            {k: v.tail(warmup + 200).reset_index(drop=True) for k, v in (ctx or {}).items()}
            if ctx
            else None
        )
        live_htf_b = (
            {tf: df.tail(le._htf_need_bars(tf)).reset_index(drop=True) for tf, df in htf_b.items()}
            if htf_b
            else None
        )
        live_htf_c = (
            {
                tf: {s: df.tail(le._htf_need_bars(tf)).reset_index(drop=True) for s, df in m.items()}
                for tf, m in htf_c.items()
            }
            if htf_c
            else None
        )
        w_live = eng.last_window(
            live_bars, 64, context=live_ctx, htf_bars=live_htf_b, htf_context=live_htf_c
        )
        assert w_live is not None
        max_diff = float(np.max(np.abs(w_batch - w_live[0])))
        assert max_diff <= tol, f"{bs} train/serve max_diff={max_diff} exceeded {tol}"


def test_multi_symbol_config_cannot_bypass_single_symbol_manifest_guard(
    tmp_path: Path, monkeypatch
):
    cfg_path = tmp_path / "bad_multi.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "symbols": ["EUR/USD", "GBP/USD"],
                "bar_size": "5M",
                "inference": {"arch": "tcn", "backend": "onnx", "model_dir": "artifacts/xau_hybrid_5M"},
                "risk": {"single_symbol": False},
            }
        )
    )
    with pytest.raises(ValueError, match="trained for XAU/USD"):
        run_backtest_from_config(cfg_path, synthetic=True)

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
    import base64

    pk = Ed25519PrivateKey.generate().private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
    monkeypatch.setenv("PX_API_KEY", "pxk_test123456")
    monkeypatch.setenv("PX_API_SECRET", base64.b64encode(pk).decode())
    with pytest.raises(ValueError, match="trained for XAU/USD"):
        build_engine(cfg_path, allow_live=False, paths=RepoPaths(root=tmp_path))


def test_export_all_preserves_symbol_and_architecture_mode_in_manifest(tmp_path: Path):
    src = Path("artifacts/xau_hybrid_5M")
    dst = tmp_path / "model_out"
    dst.mkdir(parents=True)
    for fname in ("tcn.pt", "tcn.manifest.json", "training_report.json"):
        shutil.copy(src / fname, dst / fname)

    export_all(dst / "tcn.pt", "tcn", dst, iterations=5)
    manifest = json.loads((dst / "tcn.manifest.json").read_text())
    assert manifest["symbol"] == "XAU/USD"
    assert manifest["bar_size"] == "5M"
    assert manifest["architecture_mode"] == "HYBRID_5M"


@pytest.mark.parametrize(
    "preset",
    [
        "configs/live.hybrid.yaml",
        "configs/live.hybrid.1m.yaml",
        "configs/live.mini100.hybrid.yaml",
        "configs/live.mini100.yaml",
    ],
)
def test_all_hybrid_presets_have_safe_intraday_brake_and_valid_risk_keys(preset: str):
    raw = yaml.safe_load(Path(preset).read_text())
    risk_dict = raw["risk"]
    unknown = set(risk_dict) - set(RiskConfig.__dataclass_fields__)
    assert not unknown, f"{preset} contains unknown RiskConfig keys: {unknown}"

    cfg = RiskConfig.from_dict(risk_dict, allow_env=False)
    stops = cfg.max_intraday_loss_pct / cfg.risk_per_trade_pct
    assert stops >= 3.0, f"{preset} intraday brake is {stops:.2f} stops (< 3.0)"
    assert cfg.max_intraday_loss_pct < cfg.max_daily_loss_pct


def test_dotenv_example_contains_no_active_or_invalid_risk_overrides():
    lines = Path(".env.example").read_text().splitlines()
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key = s.split("=", 1)[0].strip()
        assert not key.startswith("PX_RISK_"), (
            f".env.example should keep {key} commented out so copying .env.example "
            "does not silently override YAML risk presets"
        )


def test_backtester_equity_reconciles_entry_and_exit_fees_identically():
    import pandas as pd
    from pxscalper.backtest.engine import BacktestConfig, Backtester

    n = 40
    df = pd.DataFrame(
        {
            "time": np.arange(n, dtype=np.int64) * 300_000 + 1_780_000_000_000,
            "open": np.linspace(100.0, 104.0, n),
            "high": np.linspace(100.3, 104.3, n),
            "low": np.linspace(99.7, 103.7, n),
            "close": np.linspace(100.1, 104.1, n),
            "volume": np.full(n, 1000.0),
        }
    )
    windows = np.zeros((n, 2, 4), dtype=np.float32)
    idx = np.arange(n, dtype=np.int64)
    atr = np.full(n, 0.5, dtype=np.float64)

    def sig(row: int, _w: np.ndarray):
        return (1, 0.8, 0.2) if row in (3, 18) else (0, 0.0, 0.0)

    cfg = BacktestConfig(
        initial_equity=10_000.0,
        taker_fee_pct=0.05,
        max_holding_bars=4,
        sl_atr=4.0,
        tp_atr=8.0,
    )
    res = Backtester(cfg).run(df, windows, idx, atr, sig)
    assert len(res.trades) == 2
    sum_pnl = sum(t.pnl for t in res.trades)
    assert (res.equity_curve[-1] - cfg.initial_equity) == pytest.approx(sum_pnl, rel=1e-9, abs=1e-9)


def test_order_router_enforces_max_spread_bps():
    client = _FakeExchangeClient()
    router = OrderRouter(
        client,  # type: ignore[arg-type]
        ExecutionConfig(dry_run=True, max_spread_bps=1.5),
    )
    res = router.open_position(
        symbol="XAU/USD",
        intent="LONG",
        qty=0.1,
        stop_loss=4035.0,
        take_profit=4070.0,
        reference_price=4050.0,
        spread_bps=2.2,
    )
    assert not res.ok
    assert "max_spread_bps" in (res.error or "")


def test_edge_screen_two_sided_p_works_under_numpy2():
    from pxscalper.research.edge_screen import _two_sided_p

    p = _two_sided_p(3.29, 1000)
    assert 0.0 < p < 0.002


def test_cli_risk_set_dry_run_does_not_crash(tmp_path: Path):
    from typer.testing import CliRunner
    from pxscalper.cli import app

    cfg = tmp_path / "live.hybrid.yaml"
    shutil.copy("configs/live.hybrid.yaml", cfg)
    res = CliRunner().invoke(
        app, ["risk", "set", "--config", str(cfg), "--dry-run", "risk_per_trade_pct=0.25"]
    )
    assert res.exit_code == 0, res.output
    assert "dry run" in res.output



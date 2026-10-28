"""Assemble the live stack from a YAML config + environment.

Order of assembly matters: credentials -> client -> clock sync -> reconcile ->
models -> risk -> router -> engine. The engine is only started after the
reconciler has told us the truth about the account.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from ..exchange.rest import PrimeXBTClient
from ..exchange.signing import ApiCredentials
from ..execution.router import ExecutionConfig, OrderRouter
from ..features.pipeline import FeatureConfig
from ..inference.predictor import PredictorConfig, Predictor, Calibration
from ..live.engine import LiveConfig, LiveEngine
from ..live.metrics import Metrics
from ..live.reconcile import Reconciler
from ..risk.engine import RiskConfig, RiskEngine, RiskState
from ..strategy.decision import ScalpingStrategy, StrategyConfig
from ..utils import RepoPaths

log = logging.getLogger(__name__)

__all__ = ["start_from_config", "build_engine"]


def start_from_config(config_path: Path, allow_live: bool, paths: RepoPaths | None = None) -> None:
    engine = build_engine(config_path, allow_live=allow_live, paths=paths)
    engine.prepare()
    engine.run()


def build_engine(
    config_path: Path, allow_live: bool = False, paths: RepoPaths | None = None
) -> LiveEngine:
    raw = yaml.safe_load(Path(config_path).read_text())
    paths = (paths or RepoPaths()).ensure()

    # ---------------------------------------------------------- credentials
    key = os.getenv("PX_API_KEY", "").strip()
    secret = os.getenv("PX_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError("PX_API_KEY / PX_API_SECRET missing (see .env.example)")
    creds = ApiCredentials(key, secret, permission="trade" if allow_live else "read")
    read_key = os.getenv("PX_API_KEY_READ", "").strip()
    read_secret = os.getenv("PX_API_SECRET_READ", "").strip()
    read_creds = (
        ApiCredentials(read_key, read_secret, permission="read")
        if read_key and read_secret
        else None
    )

    client = PrimeXBTClient(credentials=creds, read_credentials=read_creds)
    offset = client.sync_clock()
    log.info("clock offset %+.0f ms", offset)

    # --------------------------------------------------------------- config
    live_raw = raw.get("live", {})
    live_cfg = LiveConfig(
        symbols=live_raw.get("symbols", ["EUR/USD"]),
        bar_size=live_raw.get("bar_size", "1M"),
        lookback=int(live_raw.get("lookback", 64)),
        warmup_bars=int(live_raw.get("warmup_bars", 900)),
        settle_ms=int(live_raw.get("settle_ms", 1200)),
        loop_sleep_s=float(live_raw.get("loop_sleep_s", 2.5)),
        max_hold_bars=int(live_raw.get("max_hold_bars", 45)),
        trailing_stop_atr=float(live_raw.get("trailing_stop_atr", 0.0)),
        paper=not allow_live,
        log_path=str(paths.logs / "decisions.jsonl"),
    )
    risk_cfg = RiskConfig.from_dict(raw.get("risk", {}))
    exec_cfg = ExecutionConfig(
        dry_run=not allow_live,
        **{k: v for k, v in raw.get("execution", {}).items()
           if k in ExecutionConfig.__dataclass_fields__ and k != "dry_run"},
    )
    strat_cfg = StrategyConfig(
        **{k: v for k, v in raw.get("strategy", {}).items()
           if k in StrategyConfig.__dataclass_fields__}
    )

    # --------------------------------------------------------------- models
    inf_raw = raw.get("inference", {})
    arch = inf_raw.get("arch", "tcn")
    model_dir = paths.artifacts / "models" / inf_raw.get("version", "latest")
    manifest = model_dir / f"{arch}.manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} missing. Run `make train-{arch}` then `make export` first."
        )
    predictors = []
    for i in range(int(inf_raw.get("ensemble_size", 1)) or 1):
        cfg = PredictorConfig.from_manifest(
            manifest,
            backend=inf_raw.get("backend", os.getenv("PX_INFERENCE_BACKEND", "onnx")),
            calibration_path=str(model_dir / "calibration.json")
            if (model_dir / "calibration.json").exists()
            else None,
            confidence_threshold=float(inf_raw.get("confidence_threshold", 0.45)),
            edge_threshold=float(inf_raw.get("edge_threshold", 0.0)),
            intra_op_threads=int(inf_raw.get("threads", 1)),
            model_version=f"{arch}-{i}",
        )
        predictors.append(Predictor(cfg))
    primary = predictors[0]

    # ----------------------------------------------------------------- risk
    risk = RiskEngine(risk_cfg, RiskState.load(risk_cfg.state_path))

    # -------------------------------------------------------------- reconcile
    report = Reconciler(client, state_path=paths.root / "state" / "live_positions.json").run()
    if not report.ok:
        log.warning("reconcile reported issues: %s", report.errors)

    # -------------------------------------------------------------- assembly
    strategy = ScalpingStrategy(predictor=primary, config=strat_cfg)
    router = OrderRouter(client, exec_cfg)
    metrics = Metrics(port=int(os.getenv("PX_METRICS_PORT", "9101")), enable_server=allow_live)
    engine = LiveEngine(
        client=client,
        predictor=primary,
        risk=risk,
        router=router,
        strategy=strategy,
        config=live_cfg,
        feature_config=FeatureConfig(),
        metrics=metrics,
    )
    log.info(
        "engine assembled | symbols=%s | paper=%s | backend=%s | arch=%s",
        live_cfg.symbols, live_cfg.paper, primary.config.backend, arch,
    )
    return engine

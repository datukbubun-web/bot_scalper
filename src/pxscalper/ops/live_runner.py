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
from ..inference.predictor import PredictorConfig, Predictor
from ..live.engine import LiveConfig, LiveEngine
from ..live.metrics import Metrics
from ..live.reconcile import Reconciler
from ..risk.engine import RiskConfig, RiskEngine, RiskState
from ..strategy.decision import ScalpingStrategy, StrategyConfig
from ..strategy.hybrid import HybridConfig, HybridStrategy
from ..utils import RepoPaths

log = logging.getLogger(__name__)

__all__ = ["start_from_config", "build_engine"]


def start_from_config(
    config_path: Path,
    allow_live: bool,
    paths: RepoPaths | None = None,
    max_bars: int | None = None,
) -> None:
    engine = build_engine(config_path, allow_live=allow_live, paths=paths)
    engine.prepare()
    engine.run(max_bars=max_bars)


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

    # --------------------------------------------------------------- config
    live_raw = raw.get("live", {})
    risk_raw = raw.get("risk", {})
    data_raw = raw.get("data", {})
    # The examples (and every preset we ship) put `symbols` at the TOP level, but
    # this read from `raw["live"]["symbols"]` -- so every preset silently fell
    # back to its default of EUR/USD. Configuring the bot with a preset traded a
    # different instrument than the preset named. Support both layouts, top level
    # first, and never guess silently.
    symbols_raw = live_raw.get("symbols", raw.get("symbols"))
    if not symbols_raw:
        raise RuntimeError(
            "no symbols configured: set top-level `symbols: [\"XAU/USD\"]` "
            "or `live.symbols`"
        )
    bar_size_val = live_raw.get("bar_size", raw.get("bar_size", data_raw.get("bar_size", "1M")))
    lookback_val = int(live_raw.get("lookback", raw.get("lookback", 64)))
    max_hold_val = int(
        live_raw.get(
            "max_hold_bars",
            raw.get("max_hold_bars", risk_raw.get("max_holding_bars", 45)),
        )
    )
    live_cfg = LiveConfig(
        symbols=list(symbols_raw),
        bar_size=str(bar_size_val),
        lookback=lookback_val,
        warmup_bars=int(live_raw.get("warmup_bars", raw.get("warmup_bars", 900))),
        settle_ms=int(live_raw.get("settle_ms", 1200)),
        loop_sleep_s=float(live_raw.get("loop_sleep_s", 2.5)),
        max_hold_bars=max_hold_val,
        trailing_stop_atr=float(live_raw.get("trailing_stop_atr", 0.0)),
        close_all_on_profit=float(
            live_raw.get("close_all_on_profit", raw.get("close_all_on_profit", 0.0))
        ),
        close_all_on_profit_pct=float(
            live_raw.get("close_all_on_profit_pct", raw.get("close_all_on_profit_pct", 0.0))
        ),
        close_all_enforce=bool(
            live_raw.get("close_all_enforce", raw.get("close_all_enforce", False))
        ),
        close_all_cooldown_s=float(
            live_raw.get("close_all_cooldown_s", raw.get("close_all_cooldown_s", 300.0))
        ),
        harvest_daily_profit_target=bool(
            live_raw.get(
                "harvest_daily_profit_target",
                raw.get("harvest_daily_profit_target", False),
            )
        ),
        l2_probe_interval_s=float(
            live_raw.get("l2_probe_interval_s", raw.get("l2_probe_interval_s", 60.0))
        ),
        l2_probe_ladder=tuple(
            float(x)
            for x in live_raw.get(
                "l2_probe_ladder", raw.get("l2_probe_ladder", (0.01, 0.1, 1.0, 10.0))
            )
        ),
        paper=not allow_live,
        log_path=str(paths.logs / "decisions.jsonl"),
    )
    risk_cfg = RiskConfig.from_dict(risk_raw)
    symbols = list(live_cfg.symbols)

    # ---- instrument scope -------------------------------------------------
    # `symbols: [XAU/USD]` is a default, not a rule: the next person to add a
    # second instrument changes the bot's behaviour by accident. If the risk
    # block says one instrument, refuse to start with more than one instead of
    # quietly trading both.
    if risk_cfg.single_symbol:
        if len(symbols) != 1:
            raise RuntimeError(
                f"risk.single_symbol is true but live.symbols has {len(symbols)} "
                f"entries {symbols}. Leave exactly one, or set single_symbol: false."
            )
        if not risk_cfg.allowed_symbols:
            risk_cfg.allowed_symbols = [symbols[0]]
        elif risk_cfg.allowed_symbols != symbols:
            raise RuntimeError(
                f"risk.allowed_symbols {risk_cfg.allowed_symbols} does not match "
                f"live.symbols {symbols}"
            )
    elif risk_cfg.allowed_symbols:
        outside = [s for s in symbols if s not in risk_cfg.allowed_symbols]
        if outside:
            raise RuntimeError(
                f"live.symbols contains {outside}, which risk.allowed_symbols "
                f"{risk_cfg.allowed_symbols} forbids"
            )
    exec_cfg = ExecutionConfig.from_dict(raw.get("execution", {}), dry_run=not allow_live)
    strat_raw = dict(raw.get("strategy", {}))
    if "require_rl_agreement" not in strat_raw:
        strat_raw["require_rl_agreement"] = False
    strat_cfg = StrategyConfig(
        **{k: v for k, v in strat_raw.items()
           if k in StrategyConfig.__dataclass_fields__}
    )

    # --------------------------------------------------------------- models
    inf_raw = raw.get("inference", {})
    arch = inf_raw.get("arch", "tcn")
    version = inf_raw.get("version", "latest")
    if inf_raw.get("model_dir"):
        model_dir = Path(inf_raw["model_dir"])
    elif (paths.artifacts / "models" / version).exists():
        model_dir = paths.artifacts / "models" / version
    elif (paths.artifacts / version).exists():
        model_dir = paths.artifacts / version
    else:
        model_dir = paths.artifacts / "models" / version
    manifest = model_dir / f"{arch}.manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} missing. Run `make train-{arch}` then `make export` first."
        )
    # Validate every configured symbol against the manifest's symbol scope so a
    # single-instrument model (e.g. XAU/USD) can never silently trade EUR/USD
    # when len(symbols) > 1.
    for sym in symbols:
        PredictorConfig.from_manifest(manifest, expected_symbol=sym)
    expected_sym = symbols[0] if len(symbols) == 1 else None
    predictors = []
    for i in range(int(inf_raw.get("ensemble_size", 1)) or 1):
        cfg = PredictorConfig.from_manifest(
            manifest,
            backend=inf_raw.get("backend", os.getenv("PX_INFERENCE_BACKEND", "onnx")),
            calibration_path=str(model_dir / "calibration.json")
            if (model_dir / "calibration.json").exists()
            else None,
            expected_symbol=expected_sym,
            confidence_threshold=float(inf_raw.get("confidence_threshold", 0.45)),
            edge_threshold=float(inf_raw.get("edge_threshold", 0.0)),
            intra_op_threads=int(inf_raw.get("threads", 1)),
            model_version=f"{arch}-{i}",
        )
        predictors.append(Predictor(cfg))
    primary = predictors[0]
    live_cfg.lookback = int(primary.config.lookback)

    feat_json = model_dir / "features.json"
    feat_cfg = FeatureConfig.load(feat_json) if feat_json.exists() else FeatureConfig()

    # Sync clock & reconcile only AFTER local config/model validation succeeds
    offset = client.sync_clock()
    log.info("clock offset %+.0f ms", offset)

    # ----------------------------------------------------------------- risk
    risk = RiskEngine(risk_cfg, RiskState.load(risk_cfg.state_path))

    # -------------------------------------------------------------- reconcile
    settings_by_symbol = {}
    atr_by_symbol: dict[str, float] = {}
    if allow_live:
        try:
            settings_by_symbol = {s.symbol: s for s in client.settings(symbols=symbols)}
        except Exception as exc:  # noqa: BLE001
            log.warning("startup settings fetch skipped during reconcile: %s", exc)
        for sym in symbols:
            try:
                recent = client.bars(sym, live_cfg.bar_size, limit=30)
                if recent and len(recent) >= 5:
                    import pandas as pd

                    high = pd.Series([b.high for b in recent], dtype=float)
                    low = pd.Series([b.low for b in recent], dtype=float)
                    close = pd.Series([b.close for b in recent], dtype=float)
                    prev = close.shift(1)
                    tr = pd.concat(
                        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
                    ).max(axis=1)
                    atr_by_symbol[sym] = float(tr.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1])
            except Exception as exc:  # noqa: BLE001
                log.debug("startup ATR fetch skipped for %s: %s", sym, exc)

    state_file = (
        Path(live_cfg.state_path)
        if Path(live_cfg.state_path).is_absolute()
        else (paths.root / live_cfg.state_path)
    )
    live_cfg.state_path = str(state_file)
    report = Reconciler(
        client,
        state_path=state_file,
        attach_stops=exec_cfg.attach_protection,
        atr_by_symbol=atr_by_symbol,
        sl_atr=risk_cfg.sl_atr,
        dry_run=not allow_live,
        allowed_symbols=symbols,
        settings_by_symbol=settings_by_symbol,
    ).run()
    if not report.ok:
        log.warning("reconcile reported issues: %s", report.errors)

    # -------------------------------------------------------------- assembly
    from ..exchange.orderbook import QuoteBookRecorder
    from ..features.pipeline import FeatureEngineer

    hybrid_raw = raw.get("hybrid", {})
    use_hybrid = (
        str(strat_raw.get("kind", "")).lower() == "hybrid"
        or bool(hybrid_raw.get("enabled", False))
    )
    if use_hybrid:
        strategy = HybridStrategy(
            predictor=primary,
            config=strat_cfg,
            hybrid_config=HybridConfig.from_dict(hybrid_raw),
            feature_names=FeatureEngineer(feat_cfg).feature_names,
        )
    else:
        strategy = ScalpingStrategy(predictor=primary, config=strat_cfg)
    router = OrderRouter(client, exec_cfg)
    metrics = Metrics(port=int(os.getenv("PX_METRICS_PORT", "9101")), enable_server=allow_live)
    recorder = QuoteBookRecorder(paths.data / "quotes")
    engine = LiveEngine(
        client=client,
        predictor=primary,
        risk=risk,
        router=router,
        strategy=strategy,
        config=live_cfg,
        feature_config=feat_cfg,
        metrics=metrics,
        quote_recorder=recorder,
    )
    log.info(
        "engine assembled | symbols=%s | paper=%s | backend=%s | arch=%s",
        live_cfg.symbols, live_cfg.paper, primary.config.backend, arch,
    )
    return engine

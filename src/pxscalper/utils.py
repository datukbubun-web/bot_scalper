"""Shared utilities: config loading, structured logging, time helpers, seeds."""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "RepoPaths",
    "load_dotenv",
    "setup_logging",
    "now_ms",
    "floor_to_bar",
    "bar_ms",
    "seed_everything",
    "JsonlLogger",
    "deep_merge",
]

MS_MINUTE = 60_000
BAR_MS = {
    "1M": MS_MINUTE,
    "5M": 5 * MS_MINUTE,
    "15M": 15 * MS_MINUTE,
    "30M": 30 * MS_MINUTE,
    "1H": 60 * MS_MINUTE,
    "2H": 120 * MS_MINUTE,
    "4H": 240 * MS_MINUTE,
    "1D": 1440 * MS_MINUTE,
    "1W": 7 * 1440 * MS_MINUTE,
}


def now_ms() -> int:
    return int(time.time() * 1000)


def bar_ms(bar_size: str) -> int:
    return BAR_MS[bar_size]


def floor_to_bar(ts_ms: int, bar_size: str) -> int:
    step = BAR_MS[bar_size]
    return (ts_ms // step) * step


@dataclass
class RepoPaths:
    root: Path = field(default_factory=lambda: Path(os.getenv("PX_REPO_ROOT", ".")).resolve())

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> "RepoPaths":
        for path in (self.data, self.artifacts, self.logs):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def model_dir(self, version: str = "latest") -> Path:
        directory = self.artifacts / "models" / version
        directory.mkdir(parents=True, exist_ok=True)
        return directory


def load_dotenv(path: str | Path = ".env", override: bool = False) -> dict[str, str]:
    """Minimal .env loader (no dependency, never logs values)."""
    loaded: dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return loaded
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.split(" #")[0].strip().strip('"').strip("'")
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = value
    return loaded


def setup_logging(level: str = "INFO", json_mode: bool = False) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_mode:
        fmt = '{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}'
        formatter = logging.Formatter(fmt)
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)-24s %(message)s", datefmt="%H:%M:%S"
        )
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)


def seed_everything(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)  # speed > bit-exact for this workload
    except Exception:  # noqa: BLE001
        pass


class JsonlLogger:
    """Append-only JSONL log: every decision, order and fill is auditable."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **payload: Any) -> None:
        record = {"ts": now_ms(), "event": event, **_jsonable(payload)}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))  # type: ignore[arg-type]
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(dict(out[key]), value)  # type: ignore[arg-type]
        else:
            out[key] = value
    return out

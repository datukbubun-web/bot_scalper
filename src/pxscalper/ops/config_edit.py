"""Edit the risk block of a YAML config in place, without destroying it.

Why not `yaml.safe_load` + `yaml.dump`: that loses every comment and every bit of
formatting in the file, and those comments are the reasoning behind numbers that
protect the deposit. So the values are rewritten as *text* — the key you touch
changes, everything else in the file is byte-identical.

Every edit is validated before it is written: the file is re-parsed and pushed
through ``RiskConfig.from_dict``, so a typo fails here instead of at 3 a.m. when
the loop starts.
"""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..risk.engine import RiskConfig

RISK_FIELDS = {f.name: f for f in fields(RiskConfig)}


class ConfigEditError(ValueError):
    """The edit was refused; the file on disk is untouched."""


# --------------------------------------------------------------------- reading
def read_risk(path: str | Path) -> dict[str, Any]:
    """The effective risk section of a config file."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return dict(raw.get("risk") or {})


def read_effective(path: str | Path, *, allow_env: bool = True) -> RiskConfig:
    """The config as the engine would build it (YAML + env overrides)."""
    return RiskConfig.from_dict(read_risk(path), allow_env=allow_env)


# -------------------------------------------------------------------- coercion
def coerce(key: str, value: Any) -> Any:
    """Turn a CLI/YAML string into the type the dataclass declares."""
    if key not in RISK_FIELDS:
        known = ", ".join(sorted(RISK_FIELDS))
        raise ConfigEditError(f"unknown risk setting '{key}'.\nknown settings: {known}")
    hint = RiskConfig._field_types.get(key)
    origin = getattr(hint, "__origin__", hint)
    if isinstance(value, str):
        text = value.strip()
        try:
            if origin is bool:
                low = text.lower()
                if low not in ("true", "false", "1", "0", "yes", "no", "on", "off"):
                    raise ValueError(text)
                return low in ("true", "1", "yes", "on")
            if origin is int:
                return int(float(text))
            if origin is float:
                return float(text)
            if origin in (tuple, list):
                items = [p.strip() for p in text.split(",") if p.strip()]
                return tuple(items) if origin is tuple else items
        except ValueError as exc:
            raise ConfigEditError(
                f"{key}: '{value}' is not a valid {origin.__name__}"
            ) from exc
    return value


def format_value(value: Any) -> str:
    """Render a value for YAML, inline and unquoted where possible."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = repr(value)
        if "e" in text or "E" in text:  # YAML 1.1 reads 1e-05 as a string
            text = f"{value:.10f}".rstrip("0").rstrip(".")
        return text
    if isinstance(value, int):
        return str(value)
    if isinstance(value, (tuple, list)):
        if not value:
            return "[]"
        return "[" + ", ".join(format_value(v) if not isinstance(v, str) else v for v in value) + "]"
    if value is None:
        return "null"
    return str(value)


# ---------------------------------------------------------------------- editing
def _risk_block(lines: list[str]) -> tuple[int, int, str]:
    """(start, end, indent) of the `risk:` block. start is the header line."""
    start = -1
    for i, line in enumerate(lines):
        if re.match(r"^risk\s*:\s*(#.*)?$", line):
            start = i
            break
    if start < 0:
        raise ConfigEditError("this file has no top-level 'risk:' section")

    indent = "  "
    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        if line[0] not in (" ", "\t") and not line.lstrip().startswith("#"):
            end = j
            break
        if line[0] in (" ", "\t") and not line.lstrip().startswith("#"):
            indent = line[: len(line) - len(line.lstrip())]
    return start, end, indent


_KEY_LINE = r"^(?P<indent>[ \t]+)(?P<key>{key})\s*:(?P<value>.*?)(?P<comment>\s+#.*)?$"


def set_values(
    path: str | Path, updates: dict[str, Any], *, validate: bool = True
) -> list[tuple[str, Any, Any]]:
    """Set one or more risk values by editing text. Returns (key, old, new)."""
    p = Path(path)
    original = p.read_text()
    lines = original.splitlines(keepends=True)
    before = read_risk(p)

    coerced = {k: coerce(k, v) for k, v in updates.items()}
    changes: list[tuple[str, Any, Any]] = []

    for key, new in coerced.items():
        start, end, indent = _risk_block(lines)
        pattern = re.compile(_KEY_LINE.format(key=re.escape(key)))
        for i in range(start + 1, end):
            m = pattern.match(lines[i].rstrip("\n"))
            if not m:
                continue
            comment = m.group("comment") or ""
            eol = "\n" if lines[i].endswith("\n") else ""
            head = f"{m.group('indent')}{key}: {format_value(new)}"
            if comment:
                # keep the comment in the column it was written in
                column = len(lines[i].rstrip("\n")) - len(comment)
                head = head.ljust(column)
            lines[i] = f"{head}{comment}{eol}"
            break
        else:
            # key absent -> insert at the end of the block, same indent
            insert_at = end
            while insert_at > start + 1 and not lines[insert_at - 1].strip():
                insert_at -= 1
            if not lines[insert_at - 1].endswith("\n"):
                lines[insert_at - 1] += "\n"
            lines.insert(insert_at, f"{indent}{key}: {format_value(new)}\n")
        changes.append((key, before.get(key), new))

    updated = "".join(lines)

    if validate:
        try:
            parsed = yaml.safe_load(updated) or {}
            RiskConfig.from_dict(parsed.get("risk") or {}, allow_env=False)
        except Exception as exc:  # noqa: BLE001
            raise ConfigEditError(f"refusing to write, result would not parse: {exc}") from exc

    p.write_text(updated)
    return changes


# ------------------------------------------------------------------- explaining
def describe(cfg: RiskConfig, stops_at_pct: float | None = None) -> list[tuple[str, str, str]]:
    """(name, value, meaning) rows for the guards a user actually tunes."""
    rows: list[tuple[str, str, str]] = []
    risk = cfg.risk_per_trade_pct

    def guard(name: str, value: float, meaning: str) -> None:
        rows.append((name, f"{value:g}%", meaning))

    guard(
        "max_intraday_loss_pct",
        cfg.max_intraday_loss_pct,
        "pause until UTC rollover"
        + (
            f" (~{cfg.max_intraday_loss_pct / risk:.1f} stops)" if risk > 0 and cfg.max_intraday_loss_pct else " (disabled)"
        ),
    )
    guard(
        "max_daily_loss_pct",
        cfg.max_daily_loss_pct,
        "kill switch for the day",
    )
    guard(
        "max_total_loss_pct",
        cfg.max_total_loss_pct,
        "PERMANENT kill switch, measured from the deposit"
        if cfg.max_total_loss_pct
        else "(disabled)",
    )
    guard("max_drawdown_pct", cfg.max_drawdown_pct, "kill switch from the peak")
    rows.append(("risk_per_trade_pct", f"{risk:g}%", "size per trade, to the stop"))
    rows.append(("max_leverage", f"{cfg.max_leverage:g}x", "position ceiling"))
    rows.append(("max_daily_trades", str(cfg.max_daily_trades), "trade budget per UTC day"))
    rows.append(
        (
            "max_positions",
            str(cfg.max_positions),
            f"open at once (max {cfg.max_positions_per_symbol} per symbol)",
        )
    )
    rows.append(
        (
            "max_gross_exposure_pct",
            f"{cfg.max_gross_exposure_pct:g}%",
            f"total notional ceiling = ${cfg.max_gross_exposure_pct / 100 * 100:,.0f} per $100",
        )
    )
    rows.append(
        (
            "blackout",
            f"{cfg.blackout_pre_min:g}/{cfg.blackout_post_min:g} min",
            f"{'on' if cfg.blackout_enabled else 'off'} — {', '.join(cfg.blackout_kinds) or 'no kinds'}",
        )
    )
    return rows


def summarise(changes: Iterable[tuple[str, Any, Any]]) -> str:
    return "\n".join(f"  {k}: {o!r} -> {n!r}" for k, o, n in changes)

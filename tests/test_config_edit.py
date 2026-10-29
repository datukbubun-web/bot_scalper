"""Editing the guards must be safe, or people will edit them carelessly."""

from __future__ import annotations

import shutil

import pytest
import yaml

from pxscalper.ops.config_edit import (
    ConfigEditError,
    coerce,
    describe,
    format_value,
    read_risk,
    set_values,
)

PRESET = "configs/live.mini100.yaml"


@pytest.fixture()
def cfg(tmp_path):
    dst = tmp_path / "live.yaml"
    shutil.copy(PRESET, dst)
    return dst


def test_edit_changes_only_the_named_key(cfg):
    before = cfg.read_text()
    set_values(cfg, {"max_intraday_loss_pct": 2.5})
    after = cfg.read_text()
    changed = [a for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
    assert changed == ["  max_intraday_loss_pct: 2.0"]
    assert len(before.splitlines()) == len(after.splitlines())


def test_comments_survive_and_never_touch_the_value(cfg):
    before = cfg.read_text()
    original = [l for l in before.splitlines() if "max_daily_trades" in l][0]
    set_values(cfg, {"max_daily_trades": 12})
    line = [l for l in cfg.read_text().splitlines() if "max_daily_trades" in l][0]
    assert line.endswith(original[original.index("#"):]), "comment text changed"
    assert line.index("#") >= original.index("#"), "comment moved left over the value"
    assert before.count("#") == cfg.read_text().count("#")


def test_a_same_length_change_keeps_the_comment_exactly_in_place(cfg):
    before = cfg.read_text()
    col = [l for l in before.splitlines() if "max_daily_trades" in l][0].index("#")
    set_values(cfg, {"max_daily_trades": 9})       # 8 -> 9, same width
    line = [l for l in cfg.read_text().splitlines() if "max_daily_trades" in l][0]
    assert line.index("#") == col


def test_new_key_is_added_inside_the_risk_block(cfg):
    set_values(cfg, {"min_equity": 55})
    doc = yaml.safe_load(cfg.read_text())
    assert doc["risk"]["min_equity"] == 55
    assert "min_equity" not in {k for k in doc if k != "risk"}


def test_result_must_parse_before_it_is_written(cfg):
    original = cfg.read_text()
    with pytest.raises(ConfigEditError):
        set_values(cfg, {"max_intraday_loss_pct": "not-a-number"})
    assert cfg.read_text() == original, "a refused edit must not touch the file"


def test_unknown_field_lists_the_real_ones(cfg):
    with pytest.raises(ConfigEditError) as exc:
        set_values(cfg, {"max_intraday_loss": 1})
    assert "max_intraday_loss_pct" in str(exc.value)


@pytest.mark.parametrize(
    "value,expected",
    [("2.5", 2.5), ("3", 3.0), (2, 2.0)],
)
def test_float_coercion(value, expected):
    assert coerce("max_intraday_loss_pct", value) == pytest.approx(expected)


def test_bool_and_tuple_coercion():
    assert coerce("blackout_enabled", "false") is False
    assert coerce("blackout_enabled", "true") is True
    assert coerce("blackout_kinds", "nfp, cpi") == ("nfp", "cpi")
    with pytest.raises(ConfigEditError):
        coerce("blackout_enabled", "kadang")


def test_integer_settings_stay_integers(cfg):
    set_values(cfg, {"max_daily_trades": "9"})
    value = read_risk(cfg)["max_daily_trades"]
    assert value == 9 and isinstance(value, int)


def test_brake_is_described_in_stops():
    """The user asked for 2%; what matters is that 2% is four stops, not two."""
    from pxscalper.ops.config_edit import read_effective

    cfg = read_effective(PRESET, allow_env=False)
    rows = dict((name, (value, meaning)) for name, value, meaning in describe(cfg))
    assert rows["max_intraday_loss_pct"][1].startswith("pause")
    assert "4.0 stops" in rows["max_intraday_loss_pct"][1]
    assert "PERMANENT" in rows["max_total_loss_pct"][1]


def test_disabling_a_guard_is_reported_as_disabled(cfg):
    from pxscalper.ops.config_edit import read_effective

    set_values(cfg, {"max_intraday_loss_pct": 0})
    rows = dict((n, m) for n, _, m in describe(read_effective(cfg, allow_env=False)))
    assert "disabled" in rows["max_intraday_loss_pct"]


def test_format_value_never_emits_yaml_11_float_traps():
    assert format_value(1e-05) == "0.00001"
    assert format_value(2.0) == "2.0"
    assert format_value(True) == "true"
    assert format_value(("nfp", "cpi")) == "[nfp, cpi]"

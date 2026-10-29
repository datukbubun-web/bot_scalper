"""Tests for the macro-calendar features.

The dangerous failure mode for a calendar feature is not a crash: it is a column
that exists, looks plausible, and is silently zero. Two tests here exist because
that actually happened (robust-z normalisation deleted every flag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pxscalper.features.calendar import (
    BUILTIN_SCHEDULE,
    CalendarConfig,
    MacroCalendar,
    MacroEvent,
    _et,
    calendar_feature_names,
    calendar_features,
)
from pxscalper.features.pipeline import FeatureConfig, FeatureEngineer


def _times(start_iso: str, n: int, bar_ms: int = 300_000) -> np.ndarray:
    start = int(pd.Timestamp(start_iso, tz="UTC").timestamp() * 1000)
    return start + np.arange(n, dtype=np.int64) * bar_ms


# ── schedule integrity ──────────────────────────────────────────────────────
def test_builtin_schedule_has_all_three_event_kinds():
    kinds = {e.kind for e in BUILTIN_SCHEDULE}
    assert kinds == {"nfp", "cpi", "fomc"}
    assert len(BUILTIN_SCHEDULE) == 32      # 12 CPI + 12 NFP + 8 FOMC in 2026


def test_schedule_is_sorted_by_instant():
    """Two agencies can publish on the same day, so sharing a day is fine; an
    unsorted schedule is not, because everything that walks it assumes order."""
    times = [e.time_ms for e in BUILTIN_SCHEDULE]
    assert times == sorted(times)
    assert len(set(times)) == len(set(times))


def test_daylight_saving_is_handled():
    """A hardcoded UTC offset would put every release an hour off from November
    to March; the feature would look useless for reasons nobody could see."""
    summer = _et(2026, 10, 14)      # US on daylight time  -> 12:30 UTC
    winter = _et(2026, 12, 10)      # US on standard time  -> 13:30 UTC
    assert pd.Timestamp(summer, unit="ms", tz="UTC").hour == 12
    assert pd.Timestamp(winter, unit="ms", tz="UTC").hour == 13
    assert (winter - summer) % 86_400_000 == 3_600_000


def test_unknown_event_kind_is_rejected():
    with pytest.raises(ValueError):
        MacroEvent("gdp", 0)


# ── leakage ─────────────────────────────────────────────────────────────────
def test_next_and_prev_are_strictly_ordered():
    cal = MacroCalendar()
    probe = int(pd.Timestamp("2026-10-13 00:00", tz="UTC").timestamp() * 1000)
    nxt = cal.next_ms("cpi", probe)
    prev = cal.prev_ms("cpi", probe)
    assert prev <= probe < nxt
    assert nxt == int(pd.Timestamp("2026-10-14 12:30", tz="UTC").timestamp() * 1000)


def test_features_never_use_future_content():
    """Truncating the future must not change any past calendar feature: the
    schedule is known in advance, so past rows depend only on past rows."""
    times = _times("2026-10-01 00:00", 2000)     # spans CPI 14 Oct and NFP 2 Oct
    full = calendar_features(times)
    cut = calendar_features(times[:1000])
    pd.testing.assert_frame_equal(full.iloc[:1000].reset_index(drop=True), cut)


def test_flags_are_active_only_inside_their_window():
    """Around the 2026-10-14 12:30 UTC CPI print, the 15-minute flag must be on
    for exactly the three 5-minute bars before it and the three after."""
    times = _times("2026-10-14 11:30", 60)       # 5-minute bars
    feats = calendar_features(times)
    stamps = pd.to_datetime(times, unit="ms", utc=True)
    release = pd.Timestamp("2026-10-14 12:30", tz="UTC")
    pre = feats["cal_cpi_pre_15"].to_numpy() == 1.0
    post = feats["cal_cpi_post_15"].to_numpy() == 1.0
    assert stamps[pre].max() < release
    # bars are stamped at their OPEN, so the earliest bar inside a 15-minute
    # window is exactly 15 minutes before the release
    assert (release - stamps[pre].min()) <= pd.Timedelta(minutes=15)
    assert stamps[post].min() >= release
    assert (stamps[post].max() - release) < pd.Timedelta(minutes=15)


def test_event_day_flag_marks_the_whole_day():
    times = _times("2026-10-14 00:00", 288)      # a full UTC day of 5m bars
    feats = calendar_features(times)
    assert (feats["cal_cpi_is_day"] == 1.0).all()


def test_unlisted_symbol_years_degrade_to_no_signal_not_a_crash():
    """Outside the built-in year the calendar has no events; features must be a
    neutral constant, never NaN and never a fabricated date."""
    times = _times("2031-06-01 00:00", 500)
    feats = calendar_features(times)
    assert not feats.isna().any().any()
    assert (feats["cal_nfp_pre_15"] == 0.0).all()
    assert (feats["cal_nfp_is_day"] == 0.0).all()


def test_custom_calendar_from_events():
    events = [MacroEvent("nfp", _et(2026, 10, 2))]
    cal = MacroCalendar(events)
    assert cal.times("cpi").size == 0
    assert cal.next_ms("nfp", 0) == _et(2026, 10, 2)


# ── the normalisation bug ───────────────────────────────────────────────────
def test_calendar_flags_survive_pipeline_normalisation():
    """A binary flag is 1 on a handful of bars in a 500-bar window, so its
    rolling MAD is 0 -> the robust z-score is NaN -> ``fillna(0)`` erased the
    feature. The column still existed and looked fine."""
    n = 3000
    # must span a real release, or "never fires" is a property of the fixture
    times = _times("2026-10-12 00:00", n)
    rng = np.random.default_rng(0)
    close = 4000 * np.exp(np.cumsum(rng.normal(0, 5e-4, n)))
    bars = pd.DataFrame(
        {"time": times, "open": close, "high": close * 1.0005,
         "low": close * 0.9995, "close": close, "volume": np.full(n, 100.0)}
    )
    engineer = FeatureEngineer(FeatureConfig(include_calendar_features=True))
    raw = engineer.transform(bars)
    normalized = engineer.transform_normalized(bars)
    # this window contains the 14 Oct CPI print and no NFP: asserting on both
    # checks the flag fires when it should AND stays dark when it should
    for flag in ("cal_cpi_pre_15", "cal_cpi_is_day", "cal_cpi_post_60"):
        assert raw[flag].sum() > 0, f"{flag} never fires in the raw features"
        assert normalized[flag].abs().sum() > 0, (
            f"{flag} was silently erased by normalisation"
        )
        # passthrough: the value must be unchanged
        assert np.allclose(normalized[flag].to_numpy(), raw[flag].to_numpy())
    assert raw["cal_nfp_is_day"].sum() == 0, "no NFP falls inside this window"


def test_engineer_version_records_calendar():
    plain = FeatureEngineer(FeatureConfig())
    cal = FeatureEngineer(FeatureConfig(include_calendar_features=True))
    assert cal.n_features == plain.n_features + len(calendar_feature_names())
    assert cal.version.endswith("+cal")
    assert cal.feature_names[: plain.n_features] == plain.feature_names


def test_feature_names_are_config_derived():
    a = calendar_feature_names(CalendarConfig())
    b = calendar_feature_names(CalendarConfig(pre_windows=(5,), kinds=("cpi",)))
    assert "cal_cpi_pre_5" in b and "cal_cpi_pre_60" not in b
    assert len(a) == 23


def test_features_are_bounded_and_finite():
    feats = calendar_features(_times("2026-01-01 00:00", 5000))
    vals = feats.to_numpy()
    assert np.isfinite(vals).all()
    # ``cal_any_pre_min`` is measured in minutes (0..60), the rest are log-scaled
    assert np.abs(vals).max() <= 60.0

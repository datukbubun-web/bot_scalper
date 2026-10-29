"""Macro calendar features: what the market knows *before* a scheduled release.

Why this is legitimate and not look-ahead
-----------------------------------------
A macro release has two halves:

* the **schedule** (date + time). It is published months in advance by the
  statistical agency. Using it at bar ``t`` is not leakage -- a trader looking at
  a screen in July knows CPI lands on 14 October at 08:30 ET.
* the **content** (actual, consensus, surprise). Using any of it at bar ``t``
  would be leakage, because it does not exist until the release fires.

This module only ever uses the first half. It answers questions a trader really
asks: *how long until the next print, how long since the last one, is a release
sitting inside the holding window of the trade I am about to take.*

Why it can matter for gold
--------------------------
NFP / CPI / FOMC produce the largest scheduled volatility in the majors and in
metals. A 1-minute gold bar is mostly a read-through of the dollar; an FOMC
statement at 14:00 ET is the one time of the month when that read-through is
repriced. Even without seeing the numbers, "a CPI print is 6 minutes away" is
information about the *distribution* of the next few minutes.

Sourcing rules
--------------
``BUILTIN_SCHEDULE`` holds the official 2026 BLS and Federal Reserve dates. It
is data, not a guess, and every entry carries its source in ``SCHEDULE_SOURCES``.
When a year rolls over, fetch the next schedule and either extend the table or
point ``MacroCalendar.from_csv`` at an export -- never extrapolate dates, because
agencies move them (holidays, shutdowns, data delays).

Anti-leakage rules enforced here
--------------------------------
1. Only timestamps are consumed; no actual/consensus values enter.
2. A bar at ``t`` sees events with ``time > t`` (known in advance) and events
   with ``time <= t`` (already happened). Neither path peeks at the content.
3. All times are converted from the announcement timezone (US/Eastern) to UTC
   with ``zoneinfo``, so DST is handled. A hardcoded UTC offset would put every
   release an hour off from November to March -- the kind of silent bug that
   makes a feature look useless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

__all__ = [
    "MacroEvent",
    "MacroCalendar",
    "CalendarConfig",
    "calendar_features",
    "calendar_feature_names",
    "BUILTIN_SCHEDULE",
    "SCHEDULE_SOURCES",
]

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

SCHEDULE_SOURCES = {
    "cpi": "U.S. BLS CPI release schedule 2026 (bls.gov/schedule/news_release/cpi.htm), 08:30 ET",
    "nfp": "U.S. BLS Employment Situation schedule 2026, 08:30 ET",
    "fomc": "Federal Reserve FOMC calendar 2026, statement 14:00 ET on the final day",
}


@dataclass(frozen=True)
class MacroEvent:
    kind: str          # nfp | cpi | fomc
    time_ms: int       # UTC epoch milliseconds of the release

    def __post_init__(self) -> None:
        if self.kind not in SCHEDULE_SOURCES:
            raise ValueError(f"unknown event kind {self.kind!r}")


def _et(y: int, m: int, d: int, hour: int = 8, minute: int = 30) -> int:
    """US/Eastern wall time -> UTC epoch ms (DST handled by zoneinfo)."""
    return int(datetime(y, m, d, hour, minute, tzinfo=ET).astimezone(UTC).timestamp() * 1000)


#: Official 2026 releases. Times are announcement times in US/Eastern, sorted by
#: instant: an unsorted schedule is a landmine for anything that walks it.
BUILTIN_SCHEDULE: tuple[MacroEvent, ...] = tuple(sorted(
    [MacroEvent("cpi", _et(2026, m, d)) for m, d in (
        (1, 13), (2, 13), (3, 11), (4, 10), (5, 12), (6, 10),
        (7, 14), (8, 12), (9, 11), (10, 14), (11, 10), (12, 10),
    )]
    + [MacroEvent("nfp", _et(2026, m, d)) for m, d in (
        (1, 9), (2, 11), (3, 6), (4, 3), (5, 8), (6, 5),
        (7, 2), (8, 7), (9, 4), (10, 2), (11, 6), (12, 4),
    )]
    # FOMC: policy statement at 14:00 ET on the second day
    + [MacroEvent("fomc", _et(2026, m, d, 14, 0)) for m, d in (
        (1, 28), (3, 18), (4, 29), (6, 17), (7, 29),
        (9, 16), (10, 28), (12, 9),
    )],
    key=lambda e: e.time_ms,
))


@dataclass
class CalendarConfig:
    """Knobs for the calendar features.

    ``pre_windows`` / ``post_windows`` are minutes. The defaults are deliberately
    short: at a 5-minute bar a 60-minute pre-window is 12 bars, which is enough
    to learn from without drowning the sample.
    """

    kinds: tuple[str, ...] = ("nfp", "cpi", "fomc")
    pre_windows: tuple[int, ...] = (15, 60)
    post_windows: tuple[int, ...] = (15, 60)
    #: cap for |minutes to next| so the feature stays bounded far from events
    cap_minutes: float = 1440.0
    include_event_day: bool = True


class MacroCalendar:
    """A list of scheduled macro releases, queried without any leakage."""

    def __init__(self, events: list[MacroEvent] | tuple[MacroEvent, ...] | None = None):
        evs = sorted(events if events is not None else BUILTIN_SCHEDULE,
                     key=lambda e: e.time_ms)
        self.events: tuple[MacroEvent, ...] = tuple(evs)
        self._by_kind: dict[str, np.ndarray] = {}
        for kind in {e.kind for e in self.events}:
            self._by_kind[kind] = np.array(
                [e.time_ms for e in self.events if e.kind == kind], dtype=np.int64
            )

    # ------------------------------------------------------------------ loaders
    @classmethod
    def from_csv(cls, path: str) -> "MacroCalendar":
        """Load ``kind,iso_utc`` rows. Use this when a new year's schedule lands."""
        frame = pd.read_csv(path)
        missing = {"kind", "iso_utc"} - set(frame.columns)
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        events = []
        for row in frame.itertuples():
            ts = pd.Timestamp(row.iso_utc, tz="UTC")
            events.append(MacroEvent(str(row.kind), int(ts.timestamp() * 1000)))
        return cls(events)

    # ------------------------------------------------------------------ queries
    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_kind))

    def times(self, kind: str) -> np.ndarray:
        return self._by_kind.get(kind, np.empty(0, dtype=np.int64))

    def next_ms(self, kind: str, t_ms: int) -> float:
        """First scheduled release strictly after ``t_ms`` (or NaN)."""
        arr = self.times(kind)
        if arr.size == 0:
            return float("nan")
        idx = np.searchsorted(arr, t_ms, side="right")
        return float(arr[idx]) if idx < arr.size else float("nan")

    def prev_ms(self, kind: str, t_ms: int) -> float:
        """Most recent release at or before ``t_ms`` (or NaN)."""
        arr = self.times(kind)
        if arr.size == 0:
            return float("nan")
        idx = np.searchsorted(arr, t_ms, side="right") - 1
        return float(arr[idx]) if idx >= 0 else float("nan")

    def count(self, kind: str, start_ms: int, end_ms: int) -> int:
        arr = self.times(kind)
        if arr.size == 0:
            return 0
        lo = np.searchsorted(arr, start_ms, side="left")
        hi = np.searchsorted(arr, end_ms, side="right")
        return int(hi - lo)


def calendar_feature_names(config: CalendarConfig | None = None) -> tuple[str, ...]:
    """Column names produced by :func:`calendar_features`, derived from config."""
    cfg = config or CalendarConfig()
    names: list[str] = []
    for kind in cfg.kinds:
        names += [f"cal_{kind}_mins_to", f"cal_{kind}_mins_since"]
        names += [f"cal_{kind}_pre_{w}" for w in cfg.pre_windows]
        names += [f"cal_{kind}_post_{w}" for w in cfg.post_windows]
        if cfg.include_event_day:
            names.append(f"cal_{kind}_is_day")
    names += ["cal_any_mins_to", "cal_any_pre_min"]
    return tuple(names)


def calendar_features(
    times_ms: np.ndarray | pd.Series,
    calendar: MacroCalendar | None = None,
    config: CalendarConfig | None = None,
) -> pd.DataFrame:
    """Per-bar calendar features. Vectorised; no future *content* is touched.

    For each event kind::

        cal_<kind>_mins_to     minutes until the next release (capped, log-scaled)
        cal_<kind>_mins_since  minutes since the last release (capped, log-scaled)
        cal_<kind>_pre_<w>     1 inside the ``w``-minute window before a release
        cal_<kind>_post_<w>    1 inside the ``w``-minute window after a release
        cal_<kind>_is_day      1 on the calendar day of the release (UTC day of it)

    plus ``cal_any_mins_to`` (nearest of any kind) and ``cal_any_pre_min`` (the
    shortest active pre-window in minutes, 0 when none).
    """
    cfg = config or CalendarConfig()
    cal = calendar or MacroCalendar()
    t = np.asarray(times_ms, dtype=np.int64)
    out = pd.DataFrame(index=pd.RangeIndex(len(t)))
    cap = float(cfg.cap_minutes)

    def signed_minutes(delta_ms: np.ndarray, forward: bool) -> np.ndarray:
        """log1p-scaled magnitude with an explicit sign convention.

        Scaling keeps the feature in a sane range far from events while staying
        monotone close to them, which is where the information is.
        """
        minutes = delta_ms / 60_000.0
        mag = np.log1p(np.clip(minutes, 0, cap))
        return mag if forward else -mag

    best_to = np.full(len(t), np.inf)
    best_pre = np.zeros(len(t))

    for kind in cfg.kinds:
        arr = cal.times(kind)
        if arr.size == 0:
            for name in calendar_feature_names(CalendarConfig(kinds=(kind,),
                                                              pre_windows=cfg.pre_windows,
                                                              post_windows=cfg.post_windows,
                                                              include_event_day=cfg.include_event_day)):
                out[name] = 0.0
            continue

        idx_next = np.searchsorted(arr, t, side="right")
        next_ms = np.where(idx_next < arr.size, arr[np.clip(idx_next, 0, arr.size - 1)],
                           np.iinfo(np.int64).max)
        idx_prev = idx_next - 1
        prev_ms = np.where(idx_prev >= 0, arr[np.clip(idx_prev, 0, arr.size - 1)],
                           np.iinfo(np.int64).min)

        d_to = np.where(np.isfinite(next_ms), next_ms - t, cap * 60_000).astype(np.float64)
        d_to = np.clip(d_to, 0.0, cap * 60_000)
        d_since = np.clip(t - prev_ms, 0.0, cap * 60_000).astype(np.float64)
        has_prev = idx_prev >= 0

        out[f"cal_{kind}_mins_to"] = signed_minutes(d_to, forward=True)
        out[f"cal_{kind}_mins_since"] = np.where(
            has_prev, signed_minutes(d_since, forward=False), 0.0
        )
        for w in cfg.pre_windows:
            flag = (d_to <= w * 60_000) & (d_to >= 0)
            out[f"cal_{kind}_pre_{w}"] = flag.astype(float)
            best_pre = np.where(flag, np.minimum(best_pre + (best_pre == 0) * w, w), best_pre)
        for w in cfg.post_windows:
            # half-open [release, release + w): a 15-minute window must not
            # count the bar that falls exactly 15 minutes later, or every window
            # is one bar wider than it says it is.
            out[f"cal_{kind}_post_{w}"] = (
                has_prev & (d_since < w * 60_000)
            ).astype(float)
        if cfg.include_event_day:
            day_of_bar = (t // 86_400_000)
            event_days = set((arr // 86_400_000).tolist())
            out[f"cal_{kind}_is_day"] = np.array(
                [1.0 if d in event_days else 0.0 for d in day_of_bar]
            )

        best_to = np.minimum(best_to, d_to)

    out["cal_any_mins_to"] = signed_minutes(
        np.where(np.isfinite(best_to), best_to, cap * 60_000), forward=True
    )
    out["cal_any_pre_min"] = best_pre.astype(float)

    # freeze the column order so train and inference agree
    return out.reindex(columns=list(calendar_feature_names(cfg)))

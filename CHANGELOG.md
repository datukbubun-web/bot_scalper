# Changelog

## [0.1.0] — 2026-10-29

Rilis pertama: kerangka lengkap, diuji end-to-end.

### Ditambahkan
- **Exchange**: klien PrimeXBT Trading API **v3** dengan signing **Ed25519** (PKCS#8 base64), sinkronisasi clock otomatis, rate limiter token-bucket, typed models untuk bar/quote/position/order/fill/impact.
- **Data**: download & cache Parquet dari `/v3/trading/bars` (endpoint publik) + generator data sintetis regime-switching untuk uji offline.
- **Fitur**: 38 fitur stasioner (normalized returns 1/3/5/13, ATR, Garman-Klass, Parkinson, realized vol, OFI, signed-volume z, Kyle's lambda, quote imbalance, microprice deviation, Roll measure, sesi, regime) dengan normalisasi causal robust-z.
- **Model**: TCN (causal dilated conv), LSTM/GRU, PatchTST-lite; trainer dengan EMA, embargo, class balancing, early stopping, walk-forward.
- **DRL**: environment scalping sadar-biaya (spread, fee, slippage sqrt-law, financing, stop intrabar) + **PPO** dan **SAC** murni PyTorch.
- **Inferensi**: ekspor TorchScript & ONNX Runtime, benchmark p50/p90/p99, hot path dengan `io_binding` dan buffer pre-alokasi. Terukur **3.8× p99** lebih cepat vs eager (250 µs vs 1242 µs di container 2 vCPU).
- **Strategi**: meta-labelling (arah supervised × veto/ukuran DRL).
- **Risiko**: 12 guard rail, sizing berbasis ATR, pembulatan ke bawah, limit harian, drawdown, cooldown, kill switch berbasis file & env.
- **Eksekusi**: `impact` → `place` → `verify` → reconcile via `clientOrderId`; tidak ada retry buta pada write.
- **Backtest**: event-driven, fill di bar berikutnya, spread/fee/slippage, kap leverage setara live, metrik Sharpe/Sortino/Calmar/PF/expectancy/cost drag.
- **Live**: loop bar-close, manajemen posisi (time stop, trailing, emergency exit), rekonsiliasi saat start, Prometheus metrics, JSONL audit log.
- **CLI**: `pxs doctor | data | smoke | backtest | trade | reconcile | kill`.
- **Test**: 68 test (tanpa network, tanpa API key) termasuk uji anti-look-ahead, kausalitas arsitektur, dan uji bahwa backtester benar-benar membebankan biaya.

### Catatan
- API v3 belum menyediakan websocket → bot polling; cocok untuk scalping 1 menit, bukan HFT.
- Fitur order-flow memakai proxy tick-rule karena L2 book belum tersedia.

## [Unreleased] — 2026-10-29

### Fixed (training pipeline — each one made a shipped model worse than random)
- **EMA anchored to the random initialisation.** A fixed `decay=0.995` has a
  ~200-step memory; a 62-step epoch never reaches the live weights. Measured
  cost: non-flat accuracy 0.446 → 0.306, log-loss 1.0960 → 1.1075. EMA decay now
  ramps in (`min(decay, (1+step)/(10+step))`).
- **Selection objective ≠ reported objective.** Epoch selection used the
  class-weighted, label-smoothed loss but was compared to an unweighted
  class-frequency baseline. Selection now uses plain unweighted NLL.
- **Directional Sharpe inflated by √overlap.** Overlapping windows (63 shared
  bars) were counted as independent trades. Now sampled once per non-overlapping
  window.
- **Validation contained training bars.** The purge gap ignored the input window;
  now `lookback + horizon + embargo`.
- **No "did it learn anything" gate.** `Trainer.fit` now reports the
  class-frequency baseline and `history.beat_baseline`;
  `train_sl.py --fail-if-no-edge` exits 3 when the model loses to it.

### Changed
- `train_sl.py`: `--no-download` (never overwrite a large cache) and
  `--no-ema-warmup` (reproduce the old behaviour for comparison).
- `evaluate_real.py`: `--n-bars` to bound RAM (windows are 38×64 float32 each;
  200k bars needs ~1.9 GB).

### Added
- `docs/TCN_FIX_REPORT.md` — evidence, ablation tables, and the (negative) result.
- `scripts/training/{ablate_tcn,sweep_tcn,skip_tcn_experiment}.py` — the
  experiments, re-runnable.
- `tests/test_trainer_integrity.py` — 8 tests pinning the fixes above.

## [Unreleased] — 2026-10-29 (target + context features)

### Added
- `features/targets.py` — **side-aware triple-barrier target** replacing "sign of
  the forward return". Classes: 0 = no trade, 1 = short, 2 = long. Barriers scale
  with ATR, `cost_bps` is subtracted from every winner so an unprofitable barrier
  is labelled "no trade", and bars touching both barriers are flagged
  `ambiguous` instead of guessed. Exports `ret_long_bps` **and** `ret_short_bps`
  so a backtest scores the side the model chose.
- `features/context.py` — cross-asset features (XAG/USD, USD/CHF, USD/JPY,
  USD/CAD): return/rel-strength/vol/z/corr/beta/lead-lag per symbol.
  `FeatureConfig(include_context_features=True)` takes the model from 38 to 74
  features; column order is config-derived so train/serve match.
- `scripts/training/eval_targets.py` — evaluates the new target against prior,
  drift floor, linear probe and TCN, reporting expectancy in bps.
- `tests/test_targets_and_context.py` — 13 tests.

### Fixed
- **Context alignment lagged one bar.** `merge_asof` on close times made the
  context one bar stale, which collapsed the real XAU/XAG return correlation
  from +0.82 to +0.008 and turned the context features into noise. Now joined on
  identical bar open time (`lag_bars` available for genuinely late venues).
- **Evaluation gave every model the oracle's PnL.** Scoring the label's own
  return instead of the predicted side's return showed a 97% hit rate and
  +9.8 bps/trade for a constant predictor. Now scores `ret_long_bps` /
  `ret_short_bps` per predicted class; the same model reads −0.44 bps.
- `Rolling.cov()` on a DataFrame returns MultiIndex columns and yields all-NaN
  when sliced, so every context `beta` column had 0% coverage. Rewritten with
  explicit rolling moments (coverage 89.8–100%).

## [Unreleased] — 2026-10-29 (macro calendar)

### Added
- `features/calendar.py` — scheduled macro-release features for **NFP, CPI and
  FOMC**. Uses only the *schedule* (published months ahead), never the content,
  so it is not look-ahead. Times are converted from US/Eastern with `zoneinfo`,
  so DST is handled (CPI 14 Oct → 12:30 UTC, 10 Nov → 13:30 UTC).
  `BUILTIN_SCHEDULE` holds the 32 official 2026 dates with sources in
  `SCHEDULE_SOURCES`; `MacroCalendar.from_csv` for future years.
  Columns: `cal_<kind>_mins_to`, `_mins_since`, `_pre_15/_pre_60`,
  `_post_15/_post_60`, `_is_day`, plus `cal_any_mins_to`/`cal_any_pre_min`.
  `FeatureConfig(include_calendar_features=True)`: 38 → 61 features.
- `scripts/training/eval_calendar.py` — tests the calendar where it can actually
  help: regime volatility, per-side expectancy, and the sample size a claim
  would need.
- `tests/test_calendar.py` — 14 tests.

### Fixed
- **Normalisation silently erased every calendar flag.** A binary flag is 1 on a
  handful of bars out of a 500-bar window, so its rolling MAD is 0, the robust
  z-score becomes NaN and `fillna(0.0)` replaced it — the column existed, looked
  populated, and was entirely zero. `cal_*` columns now pass through unscaled.
- Post-event windows were one bar too wide (`<=` should be `<`): a "15-minute
  window" spanned 16 minutes.
- `BUILTIN_SCHEDULE` is now sorted by instant; an unsorted schedule is a
  landmine for anything that walks it.

### Findings (docs/CALENDAR_REPORT.md)
- Volatility around releases is **9.43×** normal for pre-NFP, 5.18× CPI, 5.33×
  FOMC (p < 0.0001), and the median 5m bar range goes 8.79 → 26.10 bps.
- Directional expectancy is unchanged inside those windows, and adding the
  features makes classification slightly worse. The calendar predicts *when*,
  not *which way*.
- Publishing that honestly needs power analysis, not intuition: the attractive
  cells need 112–5,279 days of data and we have 101.

## [Unreleased] — 2026-10-29 (event breakout study)

### Added
- `backtest/event_study.py` — `simulate_barrier_trade`, a barrier simulator that
  is correct per side, resolves both-barrier bars pessimistically, and returns
  what the *chosen* side earned.
- `scripts/training/eval_event_breakout.py` — event study with four
  pre-registered variants (momentum, reversal, breakout 5m, breakout 15m), a
  matched placebo arm, per-event aggregation and a power check.
- `tests/test_event_study.py` — 10 tests.

### Fixed
- **Shorts were evaluated with their target and stop swapped**, which produced a
  +73.42 bps per event "edge" with p=0.013 that beat the placebo arm and
  survived 5 bps of costs. The barriers were computed once from the long side
  and reused for both. Corrected, the same variants are +2.9 / +2.8 / +7.3 /
  −1.1 bps with p ≥ 0.59: nothing. The simulator now lives in the package with
  tests instead of a one-shot script, because a script that produced a
  plausible-looking 73 bps would have been believed.

### Findings (docs/EVENT_BREAKOUT_REPORT.md)
- The mechanism is real: the median release-bar range is **87.2 bps vs 3.96 bps**
  normally (22×), matching the calendar's volatility signal.
- The direction is not: 56% of post-release bars move up, indistinguishable from
  a coin at n=18.
- Four directional variants, none significant, none distinguishable from the
  placebo arm. The best needs ≈876 events (~2,190 months of data); 18 exist.

### Added — macro blackout gate
- `RiskConfig.blackout_enabled / blackout_pre_min / blackout_post_min /
  blackout_kinds / blackout_windows` and `RiskEngine.blackout_status(now_ms)`.
  Entry is paused from 15 min before an NFP/CPI/FOMC release to 15 min after,
  and resumes **on its own** — nothing to clear, no operator action. Open
  positions are *not* touched: this is an entry gate, not a flat-everything
  switch, because force-closing into a 22×-range bar is the worst trade of the
  day. The calendar is wrapped in a try/except so a calendar failure can never
  take down the trading loop; explicit `blackout_windows` cover surprise events
  and maintenance.
- `RiskDecision.details["blackout_seconds_left"]`, and a human-readable message
  (`macro blackout until 2026-10-02T12:45:00Z (20 min left)`).

### Added — capital guards for a small account
- `max_total_loss_pct`: measured from `initial_equity` and it **latches the kill
  switch**. A daily brake resets every 24 h; the deposit does not, and a $100
  account can lose 10% in a single day.
- `min_equity`: absolute floor, below which trading stops because the position
  granularity itself becomes unworkable.
- `RiskState.initial_equity`, recorded once and only once in `on_account()`
  (previously three scattered assignments, which is how a limit silently drifts).

### Added — `configs/live.mini100.yaml`
A $100 preset sized to what the venue can actually fill: 1 qtyStep = 0.01 oz ≈
$40.67 notional, spread 0.49 bps, `max_leverage 10` → a $1,000 position ceiling
and 0.24 oz ≈ $976 notional per trade. `max_daily_trades 8`, `max_positions 1`,
`dry_run: true`. Slippage on a 100 oz ($407k) order measures **0**, so size is
not the constraint — the absence of edge is.

### Fixed
- `live/engine.py` passed `now_ms=None` into `risk.evaluate()`, so every
  time-based guard silently fell back to the engine's own clock. Blackout and
  cooldown now receive the same clock the loop uses.

### Tests
- `tests/test_blackout_and_mini_account.py` — 17 tests: pause before/at/after a
  release, automatic resume, half-open window boundaries, per-kind filtering,
  disabled flag, hand-declared windows, kill-switch latching, `initial_equity`
  write-once, `min_equity`, and the shipped preset's sizing under the leverage
  cap. Suite: **144 passed**.

### Added — `pxs risk show` / `pxs risk set`
Tuning the guards no longer means editing YAML by hand or remembering variable
names. `pxs risk show` explains what each limit does and how many stops the
intraday brake is worth; `pxs risk set max_intraday_loss_pct=2.5` rewrites just
that line.

`ops/config_edit.py` edits the file as **text**, because `yaml.safe_load` +
`yaml.dump` deletes every comment — and in `live.mini100.yaml` the comments are
the reasoning behind the numbers that protect the deposit. Every edit is
re-parsed and pushed through `RiskConfig.from_dict` *before* it is written, so a
typo is refused at the terminal instead of at 3 a.m. Tests (14) cover: one-line
diff, comment preservation, insert-into-block, refused-edit-leaves-file-untouched,
unknown-field listing, float/bool/tuple coercion, integer settings staying ints,
and the "2% is four stops" explanation.

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

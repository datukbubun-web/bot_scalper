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

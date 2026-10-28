# Model Card — pxscalper sequence models

## Ringkasan

| Field | Nilai |
|---|---|
| Nama | pxscalper supervised direction model (`tcn` / `lstm` / `gru` / `transformer`) |
| Versi fitur | `fx-scalp-1.0.0` (38 fitur, lihat `features/pipeline.py::FEATURE_GROUPS`) |
| Tugas | Klasifikasi 3 kelas: **turun (−1) / datar (0) / naik (+1)** atas return forward yang dinormalisasi ATR |
| Input | Tensor `(1, 38, L)` dengan `L` default 64 bar (1 menit) |
| Output | Tiga probabilitas (softmax) + confidence; dikalibrasi dengan temperature scaling |
| Horizon label | 5 bar ke depan, dead-band `0.15 × √horizon × ATR` |
| Loss | Cross-entropy berbobot kelas + label smoothing 0.02 |
| Optimizer | AdamW, warmup + cosine decay, grad-clip 1.0 |
| Artefak yang dikirim | `model.pt` (EMA weights), `.ts` TorchScript, `.onnx`, `features.json`, `calibration.json`, `*.manifest.json` |

## Data

- **Sumber produksi**: `GET /v3/trading/bars` (harga bid feed PrimeXBT), bar 1 menit.
- **Sumber uji**: `DataPipeline.synthetic()` — regime-switching (2-state Markov), inovasi Student-t (ν=4), mean reversion lemah, spread melebar saat volatilitas naik. Dipakai **hanya** untuk menguji plumbing; metrik di data sintetis tidak boleh dibaca sebagai edge.
- **Periode**: ditentukan pemanggilan; disarankan ≥ 60.000 bar per simbol (~6 minggu) dan selalu lintas rezim (tren, range, news).

## Metrik yang dilaporkan saat training

`Trainer.fit()` mengembalikan:

| Metrik | Arti | Ambang kelayakan (saran) |
|---|---|---|
| `val_loss` | cross-entropy validasi | menurun stabil |
| `val_acc` | akurasi keseluruhan (termasuk kelas datar) | menyesatkan bila kelas datar dominan |
| `val_acc_nonflat` | akurasi pada label ±1 saja | **> 52%** berarti ada sinyal |
| `val_sharpe` | Sharpe strategi "ikuti prediksi, size 1", sebelum biaya | > 0.5 setelah kalibrasi |

> **Penting:** `val_sharpe` di trainer adalah metrik *pemilihan model*, bukan backtest. Backtest sebenarnya ada di `backtest/engine.py` dengan spread, fee, dan slippage.

## Validasi

- Split kronologis dengan **embargo 20–25 bar** (label horizon 5 bar tidak boleh tumpang tindih).
- Walk-forward tersedia lewat `features/labels.py::walk_forward_splits`.
- Model yang disimpan = **EMA** dari bobot pada `val_loss` terbaik (bukan iterasi terakhir).

## Keterbatasan & risiko

1. **Non-stationeritas.** Distribusi FX berubah; model yang dilatih pada satu kuartal bisa usang. Latih ulang berkala dan pantau `val_acc_nonflat`.
2. **Label horizon pendek** → rasio sinyal/noise rendah. Akurasi 55% sudah "baik" sekaligus rapuh.
3. **Tidak ada L2 book** → fitur order-flow adalah proxy tick-rule, bukan imbalance riil.
4. **Data sintetis tidak mewakili FX nyata**; jangan menyimpulkan edge dari `pxs smoke`.
5. **Overfitting laten** mungkin terjadi bila `min_edge`/`confidence_threshold` disetel berulang kali di data uji yang sama (multiple testing). Simpan satu periode buta untuk evaluasi akhir.
6. **Kalibrasi wajib** sebelum confidence dipakai untuk sizing. Jalankan `scripts/calibrate.py`.

## Penggunaan yang tidak diizinkan

- Menjalankan dengan `risk_per_trade_pct` besar "karena backtest bagus".
- Menonaktifkan guard rail (`risk/engine.py`) atau `PX_LIVE_TRADING` gate.
- Menaikkan leverage karena model "sangat yakin" — keyakinan model bukan estimasi probabilitas sebelum dikalibrasi.

## Catatan etika & kepatuhan

Perdagangan leveraged dapat menyebabkan kerugian melebihi ekspektasi. Pastikan Anda memenuhi syarat hukum di yurisdiksi Anda, bahwa akun PrimeXBT Anda memang boleh dipakai bot, dan bahwa Anda memahami risiko likuidasi. Repo ini tidak memberi nasihat keuangan.

## Ship criteria (enforced, not aspirational)

A checkpoint may only be described as an "edge" if it passes ALL of these:

1. `history.beat_baseline == True` — held-out NLL must beat the class-frequency
   prior (not just a coin flip, which is a weaker bar: `ln(3) = 1.0986`).
2. Validation is purged: `purge = lookback + horizon + embargo`.
3. Reported directional Sharpe is de-overlapped (`window_overlap = lookback`).

`python3 scripts/training/train_sl.py ... --fail-if-no-edge` exits 3 when (1)
fails. As of 2026-10-29 the TCN fails this bar on XAU/USD 5M/15M and EUR/USD 5M;
see `docs/TCN_FIX_REPORT.md`.

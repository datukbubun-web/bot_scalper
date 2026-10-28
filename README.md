# pxscalper — FX Scalping Stack for PrimeXBT

Bot scalping forex (dan CFD kripto) untuk PrimeXBT: **TCN / LSTM-GRU / PatchTST-style Transformer + PPO/SAC**, dengan inferensi **TorchScript / ONNX Runtime** di jalur panas, risk engine yang tidak bisa ditawar model, dan eksekusi yang memeriksa biaya sebelum mengirim order.

> ⚠️ **Baca ini dulu.** Trading dengan leverage bisa menghabiskan seluruh modal Anda. Repo ini adalah *engine* — bukan jaminan profit, bukan nasihat keuangan. Tidak ada model deep learning yang bisa memprediksi FX dengan akurasi tinggi secara konsisten; edge yang realistis di scalping berasal dari **eksekusi, biaya, dan disiplin risiko**, bukan dari keajaiban arsitektur. Jalankan `paper` minimal beberapa minggu dan uji di **demo account** PrimeXBT sebelum uang sungguhan.

---

## 1. Status & fakta penting (baca sebelum setup)

| Hal | Kenyataan |
|---|---|
| **API PrimeXBT** | Trading API **v3** di `https://api.primexbt.com/v3/trading` — REST, **Ed25519** signing. Dokumentasi: <https://api.primexbt.com/docs/trading.html> |
| **API key `pxk_...`** | Itu **key ID** (publik). Yang dipakai menandatangani request adalah **secret Ed25519 (PKCS#8, base64)** yang hanya ditampilkan sekali saat key dibuat. Tanpa secret, tidak ada request bertanda tangan yang bisa dibuat. |
| **Websocket** | Belum ada di v3. Bot *polling* `quotes` / `positions` / `orders/active` dan tidak pernah lebih cepat dari ritme aksinya. |
| **Order tidak idempoten** | Dokumentasi PrimeXBT: *"No request is safe to repeat. A placed order sent twice is two orders."* Karena itu tidak ada retry buta — lihat `execution/router.py` + `live/reconcile.py`. |
| **Margin/hedge** | Mendukung `NETTING` dan `HEDGE`. Bot menangani keduanya, tetapi default-nya netting (satu posisi per instrumen). |
| **Bars** | `GET /v3/trading/bars` tidak butuh key (harga feed bid) → bisa dipakai untuk download data historis & training tanpa kredensial. |

---

## 2. Arsitektur

```
exchange/    PrimeXBT v3: Ed25519 signing, REST client, typed models
data/        download + cache Parquet, generator data sintetis (regime-switching)
features/    fitur stasioner + order-flow + microstructure, labelling anti-leakage
models/      TCN (causal dilated conv), LSTM/GRU, PatchTST-lite, trainer (EMA, embargo)
rl/          environment scalping sadar-biaya, PPO & SAC (PyTorch murni)
inference/   ekspor TorchScript + ONNX Runtime, hot path tanpa alokasi
strategy/    meta-labelling: arah (supervised) × aksi/size (DRL)
risk/        sizing berbasis risiko, limit harian, drawdown, kill switch
execution/   impact → place → verify → reconcile (idempoten via clientOrderId)
backtest/    event-driven, friksi identik dengan live
live/        loop, rekonsiliasi, metrik Prometheus
```

**Kenapa berlapis?** Supaya setiap order bisa diaudit: `arah → veto DRL → ukuran → verdict risiko → eksekusi`. Setiap keputusan ditulis ke `logs/decisions.jsonl` beserta alasannya.

### Model prediksi harga (Deep Learning & Time-Series)

| Arsitek | Peran | Catatan |
|---|---|---|
| **TCN** | prediksi arah (supervised) | causal dilated conv, receptive field `1 + 2(k−1)(2ⁿ−1)`; ~250 µs p50 di ONNX Runtime |
| **LSTM / GRU** | alternatif long-memory | satu layer `hidden=64`, cocok bila rezim bertahan lama |
| **PatchTST-lite** | Transformer kecil | patching 8 bar → attention hanya 8 token, bukan 64 |
| **PPO / SAC** | agen keputusan (DRL) | meta-labelling: *apakah* bertindak & *seberapa besar* |

Semua model memakai kontrak yang sama: `x: (B, C, L) float32 → logits (B, n_outputs)`, sepenuhnya **kausal** (diuji oleh `tests/test_models.py::test_causality`).

### Fitur input (bukan OHLC mentah)

`features/pipeline.py` menghasilkan 38 fitur stasioner & scale-free:

- **Normalized returns** pada window 1, 3, 5, 13 bar
- **Volatilitas**: ATR(14)/price, **Garman-Klass**, Parkinson, realized vol, Bollinger bandwidth
- **Order flow / imbalance**: bar-level OFI (tick-rule proxy), signed-volume z-score, slope OBV, **Kyle's lambda**
- **Microstructure**: spread bps, quote imbalance (bid/ask size), deviasi microprice, Roll measure
- **Momentum**: EMA spread, MACD histogram, RSI(7), Stochastic %K
- **Session/seasonality**: sin/cos jam & menit, weekday, flag sesi Asia/London/NY + overlap
- **Regime**: skew, kurtosis, autokorelasi lag-1

Normalisasi memakai **causal rolling robust z-score** (median/MAD), bukan mean/std global — ini menghilangkan *look-ahead* yang paling sering lolos di notebook.

### Optimasi latensi (TorchScript / ONNX)

Diukur di container 2 vCPU, `TCN(38×64)`, batch 1, 2000 iterasi:

| Backend | p50 | p99 |
|---|---|---|
| eager PyTorch | 1123 µs | 1242 µs |
| TorchScript | 891 µs | 986 µs |
| **ONNX Runtime** | **250 µs** | **330 µs** (~3.8× p99) |

Angka absolut sangat bergantung mesin — **selalu** jalankan `python scripts/export_models.py --all` di box yang akan trading. Selain backend, `inference/predictor.py` memakai buffer input pre-alokasi, `io_binding`, dan ambang `confidence`/`edge` (di bawah ambang → FLAT) sehingga tidak ada keputusan "maksa" saat model ragu.

---

## 3. Instalasi

```bash
git clone https://github.com/datukbubun-web/bot_scalper.git
cd bot_scalper

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[train,test]"                        # torch, onnxruntime, dll.

cp .env.example .env      # lalu isi PX_API_KEY + PX_API_SECRET
make test                 # 68 test, tanpa network
make smoke                # pipeline end-to-end dengan data sintetis
```

### Kredensial

1. PrimeXBT → **API keys** (2FA wajib aktif) → buat key: satu `read`, satu `trade`.
2. Salin **key id** (`pxk_...`) dan **secret Ed25519** (base64 PKCS#8, tampil sekali).
3. Isi `.env`:

```ini
PX_API_KEY=pxk_...
PX_API_SECRET=MC4CAQAwBQYDK2Vw...      # <- BUKAN nilai pxk_...
PX_API_KEY_READ=pxk_...                 # opsional, key read-only
PX_SYMBOLS=EUR/USD,GBP/USD
PX_LIVE_TRADING=0
```

4. Verifikasi:

```bash
pxs doctor
```

`doctor` memeriksa versi paket, format secret, **clock skew** (signature hanya valid 5 detik), izin key, akun (LIVE/DEMO), dan metadata simbol (`minQty`, `qtyStep`, `takerFeePct`, jam trading).

---

## 4. Alur kerja

```bash
# 1) data historis (tanpa API key — endpoint bars publik)
pxs data --bars 1M --limit 50000

# 2) latih model supervised (TCN / lstm / transformer) + ekspor ONNX
python scripts/training/train_sl.py --arch tcn --symbol EUR/USD --n-bars 60000 --epochs 40 --export

# 3) kalibrasi probabilitas (temperature scaling) di slice yang belum pernah dilihat
python scripts/calibrate.py --arch tcn

# 4) (opsional) latih agen DRL untuk meta-labelling
python scripts/training/train_rl.py --algo ppo --symbol EUR/USD --steps 200000

# 5) backtest dengan friksi setara live
pxs backtest --config configs/live.example.yaml

# 6) paper trading (harga live, tanpa uang)
pxs trade --config configs/live.example.yaml

# 7) live — butuh DUA izin eksplisit
PX_LIVE_TRADING=1 pxs trade --config configs/live.yaml --i-understand-the-risk
```

Mode aman: `PX_LIVE_TRADING=1` **tanpa** `--i-understand-the-risk` tetap berjalan sebagai paper. Untuk live, `state/risk_state.json` menyimpan limit harian & cooldown, `state/KILL` menghentikan semuanya (`pxs kill`).

---

## 5. Risiko & guard rail

`risk/engine.py` mengevaluasi 12 pemeriksaan berurutan sebelum uang bergerak: kill switch → akun diblokir → sesi/jam trading → spread → band volatilitas → keyakinan model → exposure/leverage → korelasi → limit rugi harian → drawdown → cooldown setelah 3 loss beruntun → ukuran minimum exchange.

Sizing berbasis risiko, bukan notional:

```
risk_amount   = equity × risk_per_trade_pct / 100
stop_distance = sl_atr × ATR
qty           = risk_amount / (stop_distance × pnl_coefficient)   # JPY: 1/price
```

`qty` **dibulatkan ke bawah** ke `qtyStep`; bila hasilnya < `minQty` → **BLOCK**, tidak pernah dibulatkan ke atas (itu akan melipatgandakan risiko diam-diam).

Default: risiko 0.25% per trade, stop 1×ATR, target 2×ATR, maksimum 2 posisi, limit rugi harian 2%, drawdown maksimum 5%.

---

## 6. Struktur repo

```
src/pxscalper/
  cli.py               pxs doctor | data | smoke | backtest | trade | reconcile | kill
  exchange/            signing.py (Ed25519), rest.py, models.py
  data/pipeline.py     download/cache + sintetis
  features/            pipeline.py (38 fitur), labels.py (horizon & triple-barrier)
  models/              sequence.py (TCN, LSTM/GRU, PatchTST), trainer.py
  rl/                  env.py, agents.py (PPO, SAC), trainer.py
  inference/           export.py (TorchScript+ONNX+benchmark), predictor.py
  strategy/decision.py meta-labelling
  risk/engine.py       sizing + guard rail + kill switch
  execution/router.py  impact → place → verify → reconcile
  backtest/engine.py   event-driven, friksi setara live
  live/                engine.py, reconcile.py, metrics.py
scripts/training/      train_sl.py, train_rl.py
scripts/               export_models.py, calibrate.py
tests/                 68 test (tanpa network, tanpa API key)
docs/                  API_NOTES.md, MODEL_CARD.md, RUNBOOK.md
```

---

## 7. Batasan yang jujur

- **Belum ada L2 order book** di API v3 → fitur order-flow memakai proxy tick-rule dari bar, bukan true imbalance.
- **Belum ada websocket** → latensi dibatasi polling REST (~ratusan ms). Cocok untuk scalping 1-menit, **bukan** HFT sub-detik.
- **Model tidak dijamin profitable.** Konfigurasi default sengaja konservatif; backtest pada data sintetis hanya menguji *plumbing*, bukan edge.
- **Biaya adalah musuh utama scalping.** Taker fee 0.05%/sisi + spread berarti butuh pergerakan > ~0.1% untuk break-even; uji apakah `min_edge` Anda realistis.
- **Demo dulu.** `pxs doctor` menunjukkan apakah akun Anda `DEMO` atau `LIVE`.

## 8. Lisensi

MIT — lihat `LICENSE`. Penggunaan sepenuhnya tanggung jawab Anda.

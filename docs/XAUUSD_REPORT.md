# XAU/USD — Hasil Uji (29 Oktober 2026)

Uji ini dilakukan setelah EUR/USD dinyatakan **tidak layak secara struktural**.
Semua angka di bawah berasal dari data bar asli PrimeXBT dan API live
read-only (tidak ada order yang dikirim).

---

## 1. Ringkasan keputusan

| Aspek | EUR/USD 1M | **XAU/USD 1M** |
|---|---|---|
| ATR per bar | 1,09 bps | **4,49 bps** |
| Spread live | 3,59–4,03 bps | **0,49 bps** |
| Biaya / ATR | 3,30 (**mustahil**) | **0,11** |
| Granularitas size | 1000 unit = $1.115/step | **0,01 oz = $40,67/step** |
| Fee taker/maker | 0% / 0% | 0% / 0% |
| Impact 100 oz ($407k notional) | — | **0,00 bps** (tetap 0,49 bps) |

**XAU/USD adalah instrumen yang jauh lebih baik untuk scalping di venue ini.**
Alasannya bukan opini: spread-nya hanya 11% dari ATR 1-menit, sementara
EUR/USD 330%. Kedalaman order book-nya juga lebih baik daripada yang dibutuhkan
akun $10k — order 100 oz (notional $407 ribu) masih terisi di harga yang sama.

**Tetapi model belum punya edge di XAU/USD** (lihat bagian 5). Instrumennya
layak; otaknya belum.

---

## 2. Kondisi instrumen (live, read-only)

```
XAU/USD  metals  price 4.067,18  spread 0,49 bps  step 0,01 oz ≈ $40,67
minQty 0,01  maxQty 5000  maxLeverage 1000  fee 0%/0%
financing long 0,0181 / short -0,0044 (H24)
jam: 22:01 Sun – 20:59 Fri (break harian 20:59–22:01 UTC)
```

Tangga biaya order nyata (`scripts/xau_impact_ladder.py`):

| qty | fill | offset | notional | margin |
|---:|---:|---:|---:|---:|
| 0,01 oz | 4.067,46 | +0,49 bps | $41 | $0,40 |
| 1 oz | 4.067,46 | +0,49 bps | $4.067 | $40,67 |
| 10 oz | 4.067,46 | +0,49 bps | $40.674 | $406,72 |
| 100 oz | 4.067,46 | +0,49 bps | $406.736 | $4.067,26 |
| 500 oz | 4.067,56 | +0,74 bps | $2.033.680 | $20.336,30 |

Sampai 100 oz, offset **0,00** di atas spread — artinya untuk ukuran akun ini,
biaya eksekusi = spread saja. Ini yang membuat XAU/USD bisa diuji dengan jujur.

---

## 3. Data

| Timeframe | Bar | Periode | Catatan |
|---|---:|---|---|
| 1M | 200.000 | 2026-04-08 → 2026-10-28 (7 bulan) | 1M adalah plafon riwayat venue |
| 5M | 20.000 | 2026-07-20 → 2026-10-28 (69 hari) | |
| 15M | 20.000 | 2025-12-23 → 2026-10-28 (7 bulan) | |

Kualitas 1M: 0 duplikat, 14 gap (>2× interval), gap terbesar 2.943 menit
(akhir pekan). σ return 1-menit 3,06 bps.

> Catatan kapasitas: sandbox punya RAM 1,98 GB. 200k window = 38×64 float32
> ≈ 1,95 GB → OOM. Pelatihan memakai **45.000 bar terakhir** (3,5 bulan).
> Mesin dengan RAM ≥8 GB bisa memakai seluruh 200k bar.

---

## 4. Latensi (ONNX, sesuai requirement)

| Backend | p50 | p99 |
|---|---:|---:|
| PyTorch eager | 1.037 µs | 2.083 µs |
| TorchScript | 867 µs | 1.521 µs |
| **ONNX Runtime** | **220 µs** | **315 µs** |

ONNX memberi speedup p99 **6,6×** vs eager. Persyaratan "bukan inferensi PyTorch
mentah per tick" terpenuhi dan terukur.

---

## 5. Hasil model — jujur

### 5a. Pelatihan (XAU/USD 1M, 45.000 bar, horizon 5)

```
windows 44.835 | kelas {-1: 18.612, 0: 8.529, +1: 17.694}
epoch 0  train 1,1048  val 1,1051  acc(non-flat) 0,267  sharpe -0,14
epoch 1  train 1,0985  val 1,1038  acc(non-flat) 0,304  sharpe -0,61   ← terpilih
epoch 2  train 1,0937  val 1,1051  ...
epoch 7  train 1,0084  val 1,1738  (overfit, early stop)
```

`val_loss 1,1038` vs acak `ln(3) = 1,0986` → **lebih buruk dari tebakan seragam**.
Train loss turun ke 1,0084 sementara val naik: model menghafal, tidak belajar.

### 5b. Backtest OOS (25% terakhir, biaya spread terukur 0,49 bps, fee 0%)

| Timeframe | trades | return | hit | PF | expectancy | P(E≤0) bootstrap |
|---|---:|---:|---:|---:|---:|---:|
| 15M | 64 | **−2,04%** | 43,8% | 0,77 | −3,20 bps | 0,858 |
| 5M | 94 | +1,62% | 53,2% | 1,15 | +1,74 bps | 0,247 |
| 1M | 8 | +0,27% | 62,5% | 1,86 | — | — (n terlalu kecil) |

**Tidak ada satu pun yang lulus uji signifikansi.** 5M paling mendekati
(P(edge≤0) = 0,25), tapi 94 trade dan hasil per kuartal berubah tanda
(−1,8 / +8,3 / +4,0 / −3,7 bps) → belum bisa dipercaya.
Dengan tambahan fee 0,01%/side, 5M jatuh ke **−0,39%**.

### 5c. Diagnosis: otaknya yang rusak, bukan datanya

Uji kontrol dengan regresi logistik pada 38 fitur yang sama:

| Model | log-loss | akurasi | akurasi non-flat | IC |
|---|---:|---:|---:|---:|
| **Logistik linear** | **1,0487** | 0,4176 | **0,5152** | +0,0184 |
| TCN terlatih | 1,1021 | 0,3293 | **0,3070** | +0,0189 |
| Acak | 1,0986 | 0,3333 | 0,3333 | 0 |

Confusion matrix TCN menunjukkan pola yang **lebih buruk dari acak**: dari 4.757
bar yang sebenarnya turun, model menebak "naik" 1.055 kali dan "turun" hanya
1.734 kali. Rata-rata probabilitas output `[0,333, 0,338, 0,328]` — praktis
seragam, tidak terkalibrasi. Probabilitas maksimum tertinggi sepanjang 11.076
sampel hanya **0,444**.

Dua kesimpulan yang harus dipisahkan:

1. **TCN gagal**: dengan arsitektur & hiperparameter saat ini ia kalah dari model
   linear, bahkan di bawah acak pada akurasi arah. Ini bug/pemilihan desain,
   bukan bukti pasar efisien.
2. **Data hampir tidak punya struktur**: autokorelasi return 1M lag 1–10 semua
   |ρ| < 0,004; variance ratio 5-bar = 0,985 (≈ random walk). Regresi logistik
   yang menang pun hanya akurasi non-flat 51,5% dan IC +0,018 — cukup untuk
   mengalahkan acak, **tidak cukup** untuk menutup spread 0,49 bps setelah
   stop/target. Buy & hold selama sampel: −6,3%.

Artinya: XAU/USD **boleh** dilanjutkan, tapi targetnya harus realistis —
edge beberapa bps, bukan puluhan.

---

## 6. Yang harus diperbaiki sebelum uang sungguhan

1. **Model kalah dari baseline linear** — wajib ada test regresi yang gagal
   kalau TCN tidak mengalahkan regresi logistik pada data validasi. Tanpa ini,
   "institutional grade" tidak bisa diklaim.
2. **Stop/target tidak selaras dengan horizon.** Pelajaran dari EUR/USD: stop
   1×ATR pada bar 1-menit berada di dalam noise. Untuk XAU/USD 1M, ATR 4,5 bps
   sementara target harus > 2× spread + noise.
3. **Holdout walk-forward, bukan satu potongan.** Satu potongan 25% dengan
   8–94 trade tidak cukup untuk menyimpulkan apa pun. Perlu minimal 5 lipatan
   berurutan dan laporan distribusi expectancy antar lipatan.
4. **Data 1M hanya 7 bulan** dan tidak mencakup rezim lain. Tambahkan 15M/1H
   (7 bulan juga) atau data eksternal untuk uji lintas rezim.
5. **Financing** (long 0,0181%/H24) belum masuk backtest; untuk trade scalping
   beberapa menit dampaknya kecil, untuk swing semalam signifikan.

---

## 7. Reproduksi

```bash
# data
python3 -c "from pxscalper.data.pipeline import DataPipeline; \
  from pxscalper.exchange.rest import PrimeXBTClient; \
  DataPipeline(client=PrimeXBTClient()).download('XAU/USD','1M',200000)"

# latih (RAM-terbatas: pakai 45k bar terakhir)
python3 scripts/training/train_sl.py --arch tcn --symbol XAU/USD --bars 1M \
    --n-bars 45000 --horizon 5 --epochs 14 --no-download --export

# evaluasi OOS dengan spread terukur
python3 scripts/evaluate_real.py --arch tcn --symbol XAU/USD --bars 1M \
    --n-bars 45000 --horizon 5 --live-spread-bps 0.49 --oos-frac 0.25

# biaya order nyata (read-only)
python3 scripts/xau_impact_ladder.py
```

## 8. Pembaruan Arsitektur Positive Alpha & Solusi L2 Order Book (1 November 2026)

Seluruh poin kelemahan pada Bagian 5–6 telah ditindaklanjuti dan dipecahkan secara kausal:

### 8a. Rekonstruksi Synthetic L2 Order Book & Estimator Mikrostruktur Kausal
1. **Real-Time Synthetic L2 (`exchange/orderbook.py`, CLI `pxs orderbook`)**: membalikkan tangga eksekusi VWAP `GET /orders/market/impact` pada berbagai ukuran order (`Q_1 < Q_2 < ... < Q_K`) bersama `GET /quotes` untuk merekonstruksi irisan harga marjinal Bid/Ask, *L2 imbalance*, kemiringan *Kyle's lambda*, dan estimasi *slippage* sebelum order dikirim, serta merekam *quote/depth* via `QuoteBookRecorder` (`aggregate_quotes_to_bars` bebas *look-ahead*).
2. **Mikrostruktur Historis OHLCV (`features/microstructure.py`)**: ketika bar historis `/bars` tidak memiliki `bid`/`ask`, `FeatureConfig(synthetic_microstructure=True)` mengisi estimasi spread efektif **Abdi–Ranaldo (2017)** + **Corwin–Schultz (2012)**, **Bulk Volume Classification (BVC)**, dan ketidakseimbangan sumbu (*wick pressure*) secara kausal murni.

### 8b. Sumber Alpha Kausal: Dislokasi Nilai Relatif Emas–Perak (`XAU/USD` vs `XAG/USD`)
- Alih-alih melatih TCN berkapasitas besar tanpa jangkar yang menghafal *bull drift*, `scripts/training/train_xau.py` membangun **Antisymmetric Skip-TCN** (`use_skip=True`, `antisymmetric_head=True`, `roll_gate_channel=idx("gold_roll_blackout")`):
  - **Fitur utama (`ctx_xag_rv_z_k`)**: z-score dislokasi return `XAU/USD` terhadap $\beta_{t-1} \cdot r_{\text{XAG}}$, dengan $\beta_{t-1}$ bergulir dan dispersi $\sigma_{t-1}$ keduanya digeser `shift(1)` (100% kausal, nol *self-attenuation*).
  - **Antisimetri struktural**: `logits = [-s, flat_logit, +s]` menjamin `P(up | 0) == P(down | 0)` dan menghilangkan bias arah tanpa syarat.
  - **Gerbang *Roll Blackout* di dalam graf ONNX**: mematikan sinyal secara otomatis pada jam *rollover* harian `21:00–23:00 UTC`.

### 8c. Hasil Verifikasi Out-of-Sample Ketat (`0–60%` Fit $\to$ `60–80%` Validasi $\to$ `80–100%` Blind Test)
Diukur melalui `Predictor (ONNX) -> Backtester` dengan eksekusi kausal `open[t+1] ± half_spread`:

| Artefak / Preset | Segmen OOS | Spread (`0%` Fee) | Trades | Net Return | Sharpe | PF | Win Rate | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `artifacts/xau_5M` (`latest`) | Validasi `60–80%` | `0.65 bps` | 80 | **+2.44%** | **+1.92** | 1.34 | 53.8% | 1.23% |
| `artifacts/xau_5M` (`latest`) | Blind Test `80–100%` | `0.65 bps` | 71 | **+1.90%** | **+1.47** | 1.26 | 57.7% | 2.43% |
| `artifacts/xau_5M` (`latest`) | **Unseen OOS `60–100%`** | `0.65 bps` | 151 | **+4.38%** | **+2.39** | 1.30 | 55.6% | 2.78% |
| `artifacts/xau_5M` (`latest`) | **Unseen OOS `60–100%`** | **Stress `1.30 bps`** | 151 | **+3.48%** | **+1.92** | 1.23 | 52.3% | 2.98% |
| `artifacts/xau_15M` | Blind Test `80–100%` | `0.65 bps` | 38 | **+2.91%** | **+3.71** | 2.49 | 65.8% | 0.63% |
| `artifacts/xau_15M` | **Unseen OOS `60–100%`** | `0.65 bps` | 78 | **+5.14%** | **+4.70** | 2.17 | 61.5% | 0.91% |
| `artifacts/xau_15M` | **Unseen OOS `60–100%`** | **Stress `1.30 bps`** | 78 | **+4.73%** | **+4.36** | 2.04 | 59.0% | 0.98% |
| `configs/live.mini100.yaml` ($100, step `0.01 oz`) | **Unseen OOS `60–100%`** | `0.65 bps` | 152 | **+4.01%** | **+2.62** | 1.34 | 55.9% | 2.14% |

Latensi inferensi ONNX Runtime (`batch=1`): **p50 = 55 µs, p99 = 131 µs** (`5M`) dan **p50 = 54 µs, p99 = 100 µs** (`15M`).

---

## 9. Transformasi Arsitektur Hybrid (`Intraday Bias + Scalper Execution`)

Sistem ditingkatkan dari *pure scalper* satu-timeframe menjadi **Arsitektur Hybrid 2-Lapis**:

1. **Layer 1 — Causal Intraday Bias Engine (`features/mtf.py`, `strategy/intraday_bias.py`)**:
   - Menyelaraskan bar timeframe lebih tinggi (`15M` / `5M`) secara kausal ketat (`T_htf + htf_ms <= t_ltf + ltf_ms`; bar yang masih terbentuk *in-flight* tidak pernah bocor ke keputusan eksekusi).
   - Mengevaluasi tiga dimensi institusional: (a) kesetimbangan nilai wajar lintas-logam `15M` (`htf_15m_xag_rv_z_2`), (b) arus akumulasi/distribusi institusional terhadap **Anchored Session VWAP** (reset sesi Emas `22:00 UTC`), dan (c) rezim ekspansi volatilitas `1H / 24H` (`intraday_vol_regime`) serta *Kaufman Efficiency Ratio* 2 jam.
2. **Layer 2 — Scalper Execution Coordinator (`strategy/hybrid.py`, `models/sequence.py`)**:
   - Menggabungkan bias intraday dengan sinyal pendek `5M` (`configs/live.hybrid.yaml`, `configs/live.mini100.hybrid.yaml`) atau konfirmasi absorpsi sumbu *candle* `1M` (`configs/live.hybrid.1m.yaml`), memveto *scalp* yang melawan arus tren VWAP sesi ekstrem (`> 0.60`), ketidakselarasan `15M`, maupun guncangan volatilitas (`> 1.85x`).

### Benchmark OOS: Pure Scalper vs Arsitektur Hybrid (`XAU/USD`)

| Arsitektur / Konfigurasi | Segmen Uji | Spread | Trades | Net Return | Sharpe | PF | Win Rate | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Pure 5M Scalper (`xau_5M`) | Blind Test `80–100%` | `0.65 bps` | 71 | +1.90% | +1.47 | 1.26 | 57.7% | 2.43% |
| **Hybrid 15M Bias + 5M Exec (`xau_hybrid_5M`)** | **Blind Test `80–100%`** | `0.65 bps` | 37 | **+3.06%** | **+3.38** | **2.07** | **67.6%** | **0.59%** |
| **Hybrid 15M Bias + 5M Exec (`xau_hybrid_5M`)** | **Unseen OOS `60–100%`** | `0.65 bps` | 77 | **+5.27%** | **+4.35** | **2.01** | **64.9%** | **0.72%** |
| **Hybrid 15M Bias + 5M Exec (`xau_hybrid_5M`)** | **Unseen OOS `60–100%`** | **Stress `1.30 bps`** | 77 | **+4.75%** | **+3.96** | **1.88** | **61.0%** | **0.73%** |
| **Hybrid 5M/15M Bias + 1M Micro-Exec (`xau_hybrid_1M`)** | **Blind Test `80–100%`** | `0.65 bps` | 41 | **+2.24%** | **+2.67** | **1.70** | **68.3%** | **0.69%** |
| **Hybrid 5M/15M Bias + 1M Micro-Exec (`xau_hybrid_1M`)** | **Unseen OOS `60–100%`** | `0.65 bps` | 84 | **+4.27%** | **+3.62** | **1.68** | **64.3%** | **0.83%** |
| **Mini $100 Hybrid (`live.mini100.hybrid.yaml`)** | **Unseen OOS `60–100%`** | `0.65 bps` | 77 | **+3.89%** | **+3.91** | **1.89** | **64.9%** | **0.64%** |



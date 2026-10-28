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

Hasil mentah: `artifacts/eval/XAUUSD_tcn_eval.json`, `artifacts/xau_1M_200k/`.

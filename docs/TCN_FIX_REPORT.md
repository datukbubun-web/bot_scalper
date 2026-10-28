# Perbaikan TCN & Audit Bug Training — 29 Oktober 2026

Dokumen ini mencatat apa yang **terbukti** rusak, buktinya, perbaikannya, dan
keputusan yang sekarang dipaksakan oleh kode.

Ringkas: model tidak punya edge, tetapi yang lebih penting — **training pipeline
punya 5 bug nyata yang membuat setiap model yang di-ship lebih buruk dari
tebakan trivial, dan melaporkan Sharpe besar sebagai bukti palsu.** Semuanya
sudah diperbaiki dan dikunci dengan test.

---

## 1. Lima bug yang ditemukan

### Bug 1 — EMA menahan bobot di inisialisasi acak (paling parah)

`_EMA` memakai `decay=0,995` tetap. Memori efektifnya ~200 langkah. Dengan
batch 256 dan 15.774 sampel, satu epoch hanya **62 langkah** — jadi bobot yang
di-*ship* (rata-rata bergerak) masih sangat dekat ke inisialisasi acak,
sementara bobot live sudah belajar.

Bukti dari ablation (`scripts/training/ablate_tcn.py`, data identik, seed sama):

| Varian | log-loss | acc non-flat |
|---|---:|---:|
| stok (EMA tetap) | 1,1075 | **0,306** |
| tanpa EMA | **1,0960** | **0,446** |

Akurasi non-flat naik **46%** hanya dengan mematikan EMA.

**Perbaikan:** EMA dengan *warmup debiasing* `decay = min(0,995, (1+step)/(10+step))`
— standar timm/HuggingFace. Tanpa perbaikan ini, tuning arsitektur apa pun akan
sia-sia karena bobot yang dikirim tidak pernah sampai ke model yang terlatih.

### Bug 2 — Objective seleksi tidak sama dengan objective yang dilaporkan

Model dilatih dengan `CrossEntropyLoss(weight=class_weight, label_smoothing=0.02)`
tetapi validator memakai loss berbobot itu juga, lalu dibandingkan dengan
baseline yang **tidak berbobot**. Perbandingan apel-vs-jeruk: sistem menghukum
model yang benar.

**Perbaikan:** `eval_loss_fn = nn.CrossEntropyLoss()` (polos, tanpa bobot, tanpa
smoothing) khusus untuk seleksi epoch dan perbandingan baseline. Setelah
perbaikan, model skip 32ch: log-loss 1,1057 → **1,0984**, acc non-flat
0,277 → **0,364**.

### Bug 3 — Tidak ada gate "apakah model belajar apa pun"

Ini bug kebijakan, bukan bug kode, dan yang paling berbahaya. Model di-ship
dengan:
- `val_loss 1,1038` padahal baseline kelas-frekuensi `1,05` dan acak `1,0986`
- melaporkan `dir. sharpe 6,77` yang sebenarnya mengukur noise

Tidak ada satu pun mekanisme yang menolak model seperti itu. Itu sumber klaim
"institutional grade" yang tidak berdasar.

**Perbaikan:** `Trainer.fit()` kini menghitung NLL baseline kelas-frekuensi pada
data validasi dan menyetel `history.beat_baseline`. `train_sl.py` mencetak
LULUS/GAGAL dan memasang log WARNING. Flag `--fail-if-no-edge` membuat proses
keluar dengan kode 3 (siap dipakai di CI).

### Bug 4 — Sharpe arah digelembungkan akar-overlap

Window bersambung berbagi `lookback-1 = 63` bar, jadi PnL dua window berturut
hampir identik. Menghitungnya sebagai trade independen menggelembungkan Sharpe
hingga ~√64 = **8×**.

**Perbaikan:** `_trade_pnls()` mengambil setiap `overlap`-te sample, dengan
`window_overlap=lookback`. Test mengunci jumlah trade = `n/overlap`.

### Bug 5 — Purge gap tidak menutup lookback window

`train_end = split - embargo` dengan `embargo=25`, padahal window validasi
melihat mundur `lookback=64` bar. Artinya data validasi **berisi bar training** —
kontaminasi halus yang membuat metrik validasi terlalu optimistis.

**Perbaikan:** `purge = lookback + horizon + embargo` (default 94 bar).

---

## 2. Eksperimen arsitektur (semua gagal mengalahkan baseline)

Setelah lima bug di atas diperbaiki, TCN diuji dari berbagai sisi. Data:
XAU/USD 1M (12.599 train / 3.167 val, purged) dan EUR/USD 5M.

| Percobaan | log-loss | acc non-flat | Hasil |
|---|---:|---:|---|
| Acak seragam `ln(3)` | 1,0986 | 0,333 | — |
| Baseline kelas-frekuensi | **1,0585** | — | ambang |
| TCN 16ch, do 0,3, wd 1e-2 | 1,1054 | 0,199 | gagal |
| TCN 32ch, do 0,3, wd 1e-2 | 1,1008 | 0,323 | gagal |
| TCN 64ch×4, do 0,3, wd 1e-2 | 1,1075 | 0,142 | gagal |
| TCN 16ch, do 0,5, wd 5e-2 | 1,1040 | 0,236 | gagal |
| Skip-TCN 32ch, 30 epoch | 1,0984 | 0,364 | gagal |
| Skip-TCN 64ch, 30 epoch | 1,1017 | 0,343 | gagal |
| Skip-TCN 64ch, **35.799 sampel** | 1,0980 | 0,350 | gagal |
| Logistic regression (38 fitur) | 1,0606 | 0,536 | (lihat §3) |

**Uji overfit** memastikan ini bukan bug arsitektur: TCN 32ch berhasil menghapal
1.200 sampel (loss 1,1043 → **0,3438**, akurasi 0,361 → **0,867**). Jadi gradien
dan arsitektur sehat; TCN hanya butuh jauh lebih banyak langkah optimizer
daripada yang tersedia, lalu overfit sebelum generalisasi.

**Kesimpulan:** kapasitas, regularisasi, skip connection, dan jumlah data (2,8×)
semuanya sudah dicoba. Tidak ada varian yang melewati baseline.

---

## 3. Temuan yang lebih penting: "edge" baseline linear itu drift, bukan skill

Regresi logistik terlihat paling baik (akurasi non-flat 0,536). Diuji lebih
teliti, ternyata sebagian besar adalah **drift rezim**, bukan prediksi:

| Ukuran | Nilai |
|---|---:|
| Lantai drift (selalu tebak arah mayoritas train) | **0,507** |
| Logistic 38 fitur | 0,536 |
| Skill di atas lantai drift | **+0,029** |

Stabilitas per kuartal validasi: `0,588 · 0,445 · 0,536 · 0,579` — **runtuh ke
0,445** saat drift berbalik arah. Ketika sampel diperbesar (35.799 train),
akurasinya justru turun ke **0,501** — praktis setara lantai drift.

Probe horizon (skill di atas drift):

| horizon | 5 mnt | 15 mnt | 30 mnt | 60 mnt | 120 mnt |
|---|---:|---:|---:|---:|---:|
| skill vs drift | +0,029 | +0,016 | +0,002 | +0,008 | **−0,084** |

Terbaik di 5 menit, dan itu pun tidak cukup untuk menutup spread 0,49 bps
setelah stop/target. Autokorelasi return 1M |ρ| < 0,004; variance ratio 5-bar
0,985 → datanya mendekati random walk.

---

## 4. Hasil final (setelah semua perbaikan)

| Dataset | purge | train/val | val loss | baseline | Verdict |
|---|---|---:|---:|---:|---|
| XAU/USD 5M | 94 | 15.774/3.967 | 1,0987 | 1,0541 | **GAGAL** |
| EUR/USD 5M | 94 | 15.774/3.967 | 1,0979 | 1,0494 | **GAGAL** |
| XAU/USD 15M | 94 | 15.774/3.967 | 1,0850 | 1,0617 | **GAGAL** (terdekat) |

XAU/USD 15M adalah yang terbaik (1,0850 < 1,0986 acak) tetapi tetap di atas
baseline 1,0617. Artinya model menangkap sedikit informasi, tidak cukup untuk
dipakai.

---

## 5. Yang sekarang dipaksakan oleh kode

1. Model **tidak bisa** dianggap selesai tanpa melewati baseline
   (`--fail-if-no-edge` → exit 3).
2. Sharpe arah tidak lagi menggelembung (de-overlap).
3. Validasi tidak lagi mengandung bar training (purge = lookback + horizon).
4. Bobot yang di-ship adalah bobot yang benar-benar terlatih (EMA warmup).
5. Objective seleksi = objective yang dilaporkan.

Test baru mengunci kelimanya: `tests/test_trainer_integrity.py` (8 test),
`tests/test_real_data_contracts.py` (14 test). Total suite: **90 test lulus**.

---

## 6. Langkah berikutnya (belum dikerjakan)

Perbaikan pipeline sudah selesai; edge-nya belum ada. Arah yang masuk akal,
berurut dari yang paling mungkin memberi hasil:

1. **Fitur baru, bukan model baru.** 38 fitur saat ini didominasi statistik
   return/volatilitas. Yang belum ada dan relevan untuk scalping XAU/USD:
   microstructure lintas-venue (DXY, yield riil AS, XAG/USD sebagai lead),
   imbalance volume per level, dan fitur sesi/jam yang lebih halus.
2. **Ganti target.** Memprediksi arah 3 kelas dengan dead-band terbukti lemah
   (68% label non-flat). Target yang lebih cocok: *apakah harga akan menyentuh
   X bps lebih dulu sebelum −Y bps* (triple-barrier), yang langsung
   mengoptimalkan apa yang dipertaruhkan.
3. **Model linear sebagai komponen, bukan pesaing.** Karena logistic mengalahkan
   TCN, mulai dari model linear teregularisasi + fitur interaksi, lalu tambahkan
   kapasitas hanya jika terbukti menambah.
4. **Purged walk-forward 5 lipatan** dengan laporan distribusi expectancy antar
   lipatan — bukan satu potongan validasi.
5. **RL (PPO/SAC)** baru layak dicoba setelah ada sinyal dasar; memberi RL ruang
   aksi di atas sinyal nol hanya akan menghasilkan eksploitasi biaya/noise.

---

## 7. Reproduksi

```bash
# ablation bug EMA (bukti bug 1)
python3 scripts/training/ablate_tcn.py --n-bars 24000 --epochs 5

# sweep kapasitas + batas bawah linear
python3 scripts/training/sweep_tcn.py --n-bars 16000

# skip connection + lebih banyak langkah
python3 scripts/training/skip_tcn_experiment.py --n-bars 16000
python3 scripts/training/skip_tcn_experiment.py --n-bars 45000 --only "skip 64"

# training final dengan gate (exit 3 kalau tidak mengalahkan baseline)
python3 scripts/training/train_sl.py --arch tcn --symbol XAU/USD --bars 5M \
    --n-bars 20000 --horizon 5 --epochs 16 --no-download --fail-if-no-edge
```

Artefak: `artifacts/ablation_tcn.json`, `artifacts/sweep_tcn.json`,
`artifacts/skip_tcn.json`.

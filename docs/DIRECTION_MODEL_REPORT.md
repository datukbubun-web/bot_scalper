# Kalau kita tambahkan model peramal arah dari 38 fitur?

*29 Oktober 2026. Dijawab dengan menjalankan kodenya, bukan dengan pendapat.*

---

## Jawaban singkat

Model itu **sudah ada di repo ini** dan sudah pernah dilatih: TCN, LSTM, GRU,
PatchTST-lite, 38 fitur, 3 kelas (turun/datar/naik). Saya melatih ulang hari ini
pada 45.000 bar XAU/USD 1M untuk menjawab pertanyaan Anda secara langsung.

**Hasilnya: model tidak mengalahkan baseline, dan tidak akan menutup biaya.**

```
epoch 00 | val 1.1055 | acc(non-flat) 0.223
epoch 01 | val 1.1040 | acc(non-flat) 0.356   <- terbaik, lalu berhenti
epoch 07 | val 1.2246 | acc(non-flat) 0.370   <- early stop
WARNING model did NOT beat the class-frequency baseline
        (val 1.1040 vs baseline 1.0478): it has not learned the signal
FAIL: model tidak mengalahkan baseline; tidak ada sinyal yang bisa dikirim.
```

Perhatikan pola yang khas: `val_loss` **naik** dari epoch 1 ke 7 (1.1040 → 1.2246)
sementara akurasi latih membaik. Itu overfitting murni — model menghafal noise
latih dan tidak menggeneralisasi.

---

## Kenapa ini bukan soal arsitektur

### 1. Angka yang harus dilampaui

Label di repo: horizon 5 bar, dead-band `0.15 × √5 × ATR`. Diukur pada data nyata:

| Besaran | Nilai |
|---|---|
| Bar yang berlabel ±1 (bukan datar) | 81.1% |
| Pergerakan absolut saat berlabel ±1 | median 5.19 bps, rata-rata 6.86 bps |
| Menang rata-rata (setelah biaya 0.6 bps) | **+6.26 bps** |
| Kalah rata-rata | **−7.46 bps** |
| **Akurasi break-even** | **54.4%** |

Jadi model peramal arah harus benar **54.4%** hanya untuk impas. Coin flip 50%.
Buffer realistis untuk noise sampling sekitar 52–53%, artinya target praktisnya
**di atas 55%** — dan itu harus stabil di luar sampel, lintas rezim.

### 2. Yang benar-benar dicapai

Saya jalankan uji terpisah yang lebih murah dan lebih langsung: regresi
logistik pada fitur sejenis 38 (return 1/3/5/13, range, volume-z, body),
split 70/30 kronologis, 162.236 bar:

| Metrik | Hasil | Pembanding |
|---|---|---|
| Akurasi OOS (3 kelas) | **0.4205** | baseline kelas-mayoritas 0.4182 |
| Selisih dari baseline | **+0.24 poin persentase** | — |
| Akurasi arah (2 arah) | **0.5159** | coin flip 0.5000 |
| Break-even yang dibutuhkan | 0.5440 | **jarak: +2.81 poin** |
| Bobot terbesar yang dipelajari | 0.028 | praktis nol |

Model belajar **hampir tidak ada apa-apa**. Bobot terbesarnya 0.028 — kalau ada
sinyal nyata, bobotnya akan jauh lebih besar.

### 3. Kontrol: apakah kodenya yang buta, atau memang datanya kosong?

Ini bagian yang paling sering dilewati orang. Saya buat target dari fitur yang
sama ditambah noise, dengan kekuatan menurun, lalu jalankan prosedur identik:

| Kekuatan sinyal | Akurasi OOS | Baseline |
|---|---|---|
| 3.0 (kuat) | **0.843** | 0.380 |
| 2.0 (sedang) | **0.654** | 0.365 |
| 1.0 (lemah) | 0.501 | 0.392 |
| 0.5 (sangat lemah) | 0.469 | 0.442 |
| **data nyata** | **0.4205** | 0.4182 |

Kode ini **bisa** menemukan sinyal kalau ada — sampai ke tingkat yang lemah
sekalipun. Pada data nyata ia tidak menemukan apa pun. Jadi kesimpulannya bukan
"modelnya kurang besar", melainkan **tidak ada pola arah yang bisa dipelajari
dari 38 fitur itu pada data ini.**

Konsisten dengan hasil pencarian 41 kandidat di `docs/EDGE_REPORT.md`: tidak ada
sinyal arah di OHLCV XAU/USD 1M yang muat membayar biayanya.

---

## Jadi kalau tetap mau model, apa yang harus berubah?

Model adalah **lapisan terakhir**, bukan lapisan pertama. Empat hal ini harus
berubah lebih dulu, dan tanpa salah satunya, mengganti TCN dengan Transformer
tidak akan mengubah apa pun:

**1. Targetnya harus berubah.**
"Prediksi arah harga 5 bar ke depan" adalah target yang paling sulit dan paling
banyak dipakai orang. Target yang lebih mudah dan lebih berguna:
*akankah trade ini kena stop atau target lebih dulu?* (triple-barrier), atau
*akankah biaya trade ini tertutup?* Itu pertanyaan yang langsung terkait uang.

**2. Sinyal dasarnya harus ada lebih dulu.**
Model dilatih untuk memilih di antara sinyal yang sudah lolos gerbang biaya
(`pxs edge`). Kalau gerbangnya kosong — dan sekarang kosong — model hanya belajar
mengurutkan derau. Ini yang saya jelaskan di `docs/EDGE_EXPLAINED.md` dengan
angka: seleksi menambah 0.8–1.4 bps, tapi hanya kalau rata-rata dasarnya positif.

**3. Datanya harus lebih banyak dan lebih beragam.**
45.000 bar = ~31 hari. Itu terlalu sedikit untuk 38 fitur × jaringan dengan
puluhan ribu parameter. Untuk 38 fitur dan model sekuensial, saya akan menuntut
**minimal 6–12 bulan** dan **lintas rezim** (tren, range, krisis), bukan 1 bulan.

**4. Biayanya harus lebih kecil, atau waktu tahan lebih panjang.**
Dari `docs/EDGE_REPORT.md`: hold 1 menit butuh edge 0.60 bps, hold 15 menit butuh
0.04 bps — 15× lebih mudah. Model yang menahan posisi 15–60 menit punya peluang
jauh lebih besar daripada model scalping 1 menit.

---

## Yang akan saya kerjakan kalau Anda mau lanjut

Bukan "latih model lebih besar". Urutan yang bisa dipertanggungjawabkan:

1. **`pxs edge --bars 5M --horizon 3`** pada semua instrumen yang datanya ada —
   cari di mana sinyalnya muat membayar biaya, kalau ada. Murah, 1 menit.
2. Kalau ada kandidat yang lolos: **latih model untuk memilih di antara kandidat
   itu** (meta-labelling), dengan target triple-barrier.
3. Ukur dengan **backtester berbiaya** (`backtest/engine.py`), bukan `val_acc`.
4. Gerbang terakhir tetap `pxs edge` pada sinyal yang dihasilkan model. Kalau
   net-nya negatif, model tidak dikirim.

Kalau tidak ada yang lolos di langkah 1 — dan berdasarkan bukti sejauh ini saya
menduga begitu — maka jawaban jujurnya adalah: **bot ini tidak bisa dibuat
menguntungkan dengan data dan biaya yang sekarang.** Bukan karena modelnya kurang
canggih, tapi karena tidak ada informasi yang cukup untuk dieksploitasi. Uang
Anda lebih aman di tempat lain, dan itu kesimpulan yang lebih berharga daripada
backtest cantik yang akan menghabiskan deposit Anda dalam 3 bulan.

---

## Cara mereproduksi

```bash
# latih ulang model yang sudah ada, dengan gerbang anti-omong-kosong
python3 scripts/training/train_sl.py --arch tcn --symbol XAU/USD --bars 1M \
    --n-bars 45000 --epochs 12 --no-download --fail-if-no-edge

# cari sinyal yang muat membayar biaya
pxs edge --symbol XAU/USD --bars 1M --horizon 5

# cek biaya sebenarnya
pxs doctor
```

`--fail-if-no-edge` sudah ada di repo sejak perbaikan pipeline: skrip keluar
dengan status gagal bila model tidak mengalahkan baseline kelas-frekuensi. Itu
sengaja, supaya model yang tidak belajar apa pun tidak bisa lolos diam-diam.

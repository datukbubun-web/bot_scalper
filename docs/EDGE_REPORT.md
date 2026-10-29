# Bagaimana supaya bot punya edge — dan mengapa yang kita punya belum

*29 Oktober 2026. Semua angka di bawah bisa Anda reproduksi sendiri dengan
`pxs edge --symbol XAU/USD --bars 1M`.*

## Ringkasan jujur

Saya mencari edge dengan 41 hipotesis yang didaftarkan **sebelum** melihat hasil.
Hasilnya:

| Pertanyaan | Jawaban |
|---|---|
| Ada sinyal yang **nyata** secara statistik? | **Ya** — `fade_outside`, t=+4.45, Reality Check p=0.005 |
| Sinyal itu **cukup besar** untuk diperdagangkan? | **Tidak** — +0.064 bps vs biaya 0.6 bps (**9× terlalu kecil**) |
| Ada sinyal lain yang menutup biaya? | **Tidak satu pun** dari 41 kandidat |
| Ada lead-lag dari instrumen lain? | **Tidak** — prediksi terkuat 0.05 bps, 12× di bawah biaya |
| Apakah "jam 06:00 naik" itu edge? | **Bukan** — artefak overlap; t menyusut +12 → +2.47 saat dihitung per hari |

**Kesimpulan:** masalahnya bukan kurang model, dan bukan kurang fitur. Tidak ada
sinyal pada data ini yang muat membayar biayanya sendiri. Menambah kompleksitas
(TCN lebih besar, Transformer, PPO) tidak akan mengubah aritmatika itu.

---

## 1. Kerangka berpikir yang benar

Edge bukan "model yang akurasinya tinggi". Edge adalah:

```
E[edge per trade] > biaya per trade
```

Pada XAU/USD di venue ini, biaya round-trip ≈ **0.6 bps** (spread 0.49 + slippage).
Maka:

| Waktu tahan | Edge minimum per bar | Artinya |
|---|---|---|
| 1 menit | **0.600 bps** | harus menang 53.8% dengan RR 1:1 |
| 5 menit | 0.120 bps | harus menang ~50.8% |
| 15 menit | 0.040 bps | harus menang ~50.3% |

Bar 1 menit XAU punya simpangan baku **7.78 bps**. Jadi untuk hold 1 menit kita
butuh edge 0.6 bps = **7.7% dari satu simpangan baku**. Itu bukan mustahil, tapi
jelas bukan sesuatu yang muncul dari indikator biasa.

## 2. Apa yang saya uji (41 kandidat, terdaftar lebih dulu)

1. **Drift per jam** (24 kandidat) — long hanya pada jam UTC itu.
2. **Drift per hari** (5) — efek hari-dalam-minggu.
3. **Serial dependency** (4) — follow/fade setelah 3 atau 5 close searah.
4. **Volume spike** (2) — follow/fade bar dengan volume > 3× median.
5. **Breakout 20 bar** (2) — tembus high/low sebelumnya.
6. **Mean reversion z≥2** (2) — fade/follow pergerakan 20 bar yang ekstrem.
7. **Outside bar** (2) — follow/fade bar yang range-nya di atas rata-rata.

Semua sinyal hanya memakai data sampai bar-nya sendiri (diuji: mengubah seluruh
masa depan tidak mengubah satu pun nilai sinyal — `test_signals_never_change_when_the_future_changes`).

## 3. Hasil, dan cara membacanya dengan benar

`pxs edge --symbol XAU/USD --bars 1M`:

```
candidate    n        gross bps  t      hit    net bps  p (Bonf.)
fade_outside 81620    +0.064     +4.45  51.6%  -0.536   0.000
fade_run3    46730    +0.062     +3.40  51.7%  -0.538   0.028
long_hour_11  8760    +0.068     +1.95  50.6%  -0.532   1.000
...
reality check: best t=4.45 vs null median 1.85 / p95 2.81 -> p=0.005
no candidate beats its cost. Best gross +0.064 bps is 9x too small for a
0.6 bps round trip.
```

Tiga lapis koreksi yang dipakai, karena tanpa itu kita akan menipu diri sendiri:

- **Bonferroni** — 41 tes, jadi p dikali 41. Hanya dua kandidat yang lolos.
- **Reality Check (White)** — block bootstrap 200× memberi distribusi t
  maksimum *andai tidak ada sinyal sama sekali*. Median 1.85, p95 2.81. Jadi
  "t=2.5" pada pencarian 41 kandidat adalah **kebetulan biasa**, bukan temuan.
- **Net of cost** — kolom paling kanan yang menentukan, bukan kolom t.

`fade_outside` lolos dua lapis pertama — sinyalnya nyata:
range bar di atas rata-rata → arah bar itu cenderung berbalik di bar berikutnya.
Hit rate 51.6% dari 81.620 trade. Stabil di kedua paruh data (+0.090 vs +0.067
bps/hari), t per hari +3.09.

**Tapi tetap tidak bisa dipakai.** +0.064 bps per trade, biaya 0.6 bps. Anda
membayar ~9× lebih besar daripada yang Anda peroleh. Sinyal nyata yang terlalu
kecil adalah **kebisingan yang mahal**, bukan edge.

## 4. Dua jebakan yang hampir menipu saya

**Jebakan 1 — overlap.** Dengan hold 30 menit, "long jam 06:00 UTC" tampak luar
biasa: rata-rata +2.07 bps, t=+12.01, p=0.000. Tapi 30 bar yang saling tumpang
tindih bukan 30 observasi independen; Anda hanya punya **146 hari**, bukan 8.760
bar. Dihitung per hari, **t=+2.47**. Dan paruh kedua data melemah (+2.74 →
+1.40 bps/hari). Ini artefak, bukan edge.

**Jebakan 2 — edge yang tidak terakumulasi.** `fade_outside` terlihat menjanjikan
di hold 1 menit dan naik sampai 15 menit (+0.064 → +0.157 bps), lalu **berhenti**.
Kalau ini alpha sungguhan, ia akan tumbuh sebanding waktu. Karena mentok, ia
adalah mean reversion mikro yang habis dalam beberapa menit — mustahil menutup
biaya round-trip.

Sebagai pembanding, `long_hour_06` naik **linear** (0.064 → 2.07 bps dari 1 ke 30
menit). Itu tanda bahaya, bukan tanda kualitas: pola seperti itu berarti kita
sedang mengukur **drift harga**, bukan prediksi.

## 5. Lead-lag dari instrumen lain: juga nol

Kalau XAU bergerak mengikuti USD atau perak dengan jeda beberapa menit, itu edge
klasik yang bisa dieksekusi. Diuji pada 19.156 menit irisan 5 instrumen
(XAU, XAG, USD/JPY, USD/CAD, USD/CHF):

| Prediktor | Lag terbaik | Korelasi | Edge dalam bps |
|---|---|---|---|
| Dolar sintetis (JPY+CAD+CHF) | 1 menit | −0.006 | **0.018 bps** |
| Perak (XAG) | 3 menit | −0.018 | **0.054 bps** |

Keduanya 12–33× di bawah biaya 0.6 bps. Pasar ini efisien pada skala menit.

## 6. Jadi bagaimana supaya punya edge?

Bukan dengan menambah lapisan model. Urutannya begini, dan urutan ini penting:

**Langkah 1 — pilih tempat biaya lebih kecil dari sinyal.**
Edge adalah *rasio* terhadap biaya. Biaya Anda 0.6 bps; di pasar yang lebih tidak
efisien (crypto altcoin, sesi buka pasar tertentu, instrumen dengan spread lebih
tipis) rasio itu bisa berubah drastis dengan sinyal yang sama. Ini cara termurah
menaikkan edge: **turunkan biaya, atau pindah ke tempat yang lebih mahal
ketidakefisienannya.**

**Langkah 2 — perpanjang waktu tahan sampai edge muat menutup biaya.**
Dari tabel di §1: hold 1 menit butuh 0.6 bps; hold 15 menit butuh 0.04 bps —
**15× lebih mudah**. Tapi ini mengubah strategi dari scalping menjadi intraday.
Itu keputusan bisnis, bukan keputusan teknis.

**Langkah 3 — cari data yang tidak dimiliki orang lain.**
Semua yang saya uji di atas memakai OHLCV publik yang sama untuk semua peserta
pasar. Edge yang bertahan biasanya berasal dari informasi yang mahal atau sulit:
buku order L2 (venue ini belum menyediakannya — lihat `docs/API_NOTES.md`),
data posisi/funding, kalender revisi, atau eksekusi yang lebih cepat. Bukan dari
arsitektur jaringan yang lebih dalam.

**Langkah 4 — baru model, dan hanya untuk mengurutkan, bukan menemukan.**
Kalau suatu saat ada sinyal yang lolos uji biaya, gunakan model untuk memilih
*di antara* sinyal itu (meta-labelling), bukan untuk menciptakan sinyal dari nol.

**Langkah 5 — uji kualitas, bukan kuantitas.** Satu hipotesis, satu uji,
didaftarkan lebih dulu. Kalau Anda menguji 50 ide per minggu, Anda akan selalu
"menemukan" sesuatu — dan itu akan menjadi kebetulan yang menghabiskan uang Anda.

## 7. Yang harus Anda tolak dari siapa pun (termasuk saya)

- Backtest tanpa biaya, atau dengan biaya yang dipilih setelah melihat hasil.
- Sharpo tinggi pada data < 1 tahun. 204 hari XAU = ~146 hari efektif.
- "Akurasi 70%" tanpa menyebut base rate dan biaya.
- Hasil yang tidak menyebut **jumlah hipotesis yang diuji**.
- Strategi yang hanya diuji pada satu simbol, satu rezim, satu periode.

## 8. Alat yang sekarang Anda punya

```bash
pxs edge --symbol XAU/USD --bars 1M --horizon 1     # cari edge, jujur
pxs edge --bars 5M --horizon 3 --cost-bps 1.5       # uji asumsi biaya lain
pxs edge --bars 1M --no-reality-check               # lebih cepat, kurang ketat
```

Keluar dengan kode 1 kalau tidak ada kandidat yang menutup biaya — sehingga bisa
dipakai di CI: *jangan pernah men-deploy model yang belum lolos gerbang ini.*

Kode: `src/pxscalper/research/edge_screen.py` (41 kandidat terdaftar,
reality check, tabel hurdle biaya). Tes: `tests/test_edge_screen.py` (8 tes),
termasuk **positive control** (edge yang ditanam buatan harus ditemukan) dan
**negative control** (derau murni tidak boleh dinyatakan sebagai temuan).

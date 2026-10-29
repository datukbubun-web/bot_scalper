# Fitur Kalender Makro (NFP / CPI / FOMC) — 29 Oktober 2026

Fitur terpasang, 14 test lulus, dan hasilnya jelas dibedakan menjadi dua klaim
yang selama ini sering dicampur:

- **Kalender memberi tahu KAPAN volatilitas datang** → terbukti, efeknya besar
  dan sangat signifikan.
- **Kalender tidak memberi tahu ARAH harga** → juga terbukti, dan
  ketidakpastiannya terlalu besar untuk diperdagangkan.

---

## 1. Kenapa ini bukan look-ahead

Satu rilis makro punya dua bagian:

| Bagian | Kapan diketahui | Boleh dipakai di bar `t`? |
|---|---|---|
| **Jadwal** (tanggal + jam) | dipublikasikan berbulan-bulan sebelumnya | **ya** |
| **Isi** (aktual, konsensus, kejutan) | baru ada saat rilis | **tidak** |

Modul ini **hanya** memakai bagian pertama. Ia menjawab pertanyaan yang benar-
benar ditanyakan trader: *berapa menit lagi rilis berikutnya, berapa menit sejak
rilis terakhir, apakah ada rilis di dalam jendela trade yang akan saya ambil.*
Tidak ada satu pun nilai aktual/konsensus yang masuk.

Semua waktu dikonversi dari `US/Eastern` ke UTC dengan `zoneinfo`, jadi DST
tertangani: CPI 14 Oktober → **12:30 UTC**, CPI 10 November → **13:30 UTC**.
Offset UTC yang di-hardcode akan membuat setiap rilis meleset satu jam dari
November sampai Maret — bug senyap yang membuat fitur tampak tidak berguna.

## 2. Sumber jadwal (bukan karangan)

`BUILTIN_SCHEDULE` memuat **32 rilis resmi 2026**, masing-masing dengan sumbernya
di `SCHEDULE_SOURCES`:

- **CPI** (12 tanggal, 08:30 ET): BLS `bls.gov/schedule/news_release/cpi.htm`
- **NFP** (12 tanggal, 08:30 ET): BLS Employment Situation
- **FOMC** (8 tanggal, statement 14:00 ET hari kedua): kalender Federal Reserve

Untuk tahun berikutnya: jangan mengekstrapolasi tanggal. Ambil jadwal resmi dan
perluas tabel, atau pakai `MacroCalendar.from_csv("jadwal.csv")` dengan kolom
`kind,iso_utc`.

## 3. Fitur yang dihasilkan (23 kolom, 38 → 61 fitur)

Per jenis event: `mins_to`, `mins_since` (log-skalas, bertanda), flag
`pre_15`/`pre_60`/`post_15`/`post_60`, dan `is_day`. Ditambah `cal_any_mins_to`
dan `cal_any_pre_min`.

Verifikasi langsung di data: di sekitar FOMC 28 Okt 18:00 UTC, flag `pre_15`
menyala tepat pada tiga bar 5-menit sebelumnya, `post_15` pada tiga bar
sesudahnya. Jendela post bersifat half-open `[rilis, rilis+w)` — versi pertama
memakai `<=` sehingga setiap jendela satu bar lebih lebar daripada namanya.

## 4. Bug yang ditemukan: normalisasi menghapus fitur kalender

Ini yang paling berbahaya karena **tidak terlihat**. Flag biner bernilai 1 hanya
pada 3 dari 500 bar dalam jendela normalisasi, sehingga MAD bergulir = 0, lalu:

```
z = (x - median) / (1.4826 * 0 + 1e-9)  ->  NaN  ->  fillna(0.0)
```

Hasilnya: kolom tetap ada, tetap "terisi", tetapi **seluruhnya nol** — model
tidak akan pernah melihat satu rilis pun, dan tidak ada error apa pun.

**Perbaikan:** kolom `cal_*` melewati normalisasi apa adanya (seperti `sess_*`),
karena sudah bounded dan stasioner. Test
`test_calendar_flags_survive_pipeline_normalisation` mengunci ini.

## 5. Hasil: KAPAN — terbukti kuat

XAU/USD 5M, 20.000 bar, **101 hari** (3 NFP, 3 CPI, 3 FOMC dalam rentang).

Volatilitas 5 bar setelahnya, dibanding bar normal (median 5,50 bps):

| Event | Jendela | n | Vol (bps) | Rasio | p |
|---|---|---:|---:|---:|---:|
| NFP | pre-15 | 9 | 51,91 | **9,43×** | <0,0001 |
| CPI | pre-15 | 9 | 28,54 | **5,18×** | <0,0001 |
| FOMC | pre-15 | 9 | 29,34 | **5,33×** | <0,0001 |
| FOMC | post-60 | 36 | 23,08 | **4,20×** | <0,0001 |
| NFP | post-60 | 36 | 16,56 | **3,01×** | <0,0001 |

Median range bar 5M: **8,79 bps normal → 26,10 bps saat event (3,0×)**.

Kalender memang memprediksi lonjakan volatilitas, dengan sangat meyakinkan.

## 6. Hasil: ARAH — tidak

Expectancy per sisi pada target TP 1,5×ATR (biaya 0,6 bps):

| Rezim | n | short | long | p(short) |
|---|---:|---:|---:|---:|
| normal | 19.865 | −0,42 bps | −0,87 bps | 0,000 |
| jendela event | 135 | −0,42 bps | −1,22 bps | 0,865 |

Tidak ada perbedaan expectancy yang bisa dipakai. Dan menambahkan 23 fitur
kalender ke model klasifikasi tidak menolong — hasilnya sedikit **lebih buruk**
(h=20: log-loss 0,7462 → 0,7509; h=40: 0,7153 → 0,7280), konsisten dengan
menambah fitur tanpa informasi arah.

### Berapa data yang dibutuhkan?

Untuk setiap efek per-event, saya hitung n yang diperlukan agar 3σ tercapai
(rumus `(σ·3/μ)²`) lalu diterjemahkan ke jumlah hari pada timeframe ini:

| Event | Jendela | n tersedia | mean | Butuh n | Perlu hari |
|---|---|---:|---:|---:|---:|
| NFP | post-15 | 9 | +15,71 bps | 10 | 112 |
| FOMC | post-60 | 24 | +10,10 bps | 77 | **323** |
| CPI | post-60 | 36 | −4,11 bps | 441 | 1.232 |
| FOMC | post-15 | 6 | +4,50 bps | 315 | 5.279 |

Kita punya **101 hari**. Satu-satunya angka yang mendekati (NFP post-15, butuh
112 hari) adalah satu sampel; begitu estimasi variansnya dibuat lebih jujur
(t-distribution, bukan `(σ·3/μ)²` yang mengasumsikan σ diketahui), efeknya tidak
lagi terpisah dari nol. **Tidak ada satu pun yang bisa dipercaya.**

## 7. Yang benar-benar berguna dari kalender

Bukan sebagai prediktor, melainkan sebagai **koreksi risiko**. Di jendela event,
ATR dihitung dari bar-bar tenang sebelumnya, sementara pergerakan sebenarnya 3×
lebih besar. Konsekuensinya untuk barrier fixed-ATR:

| Rezim | n | Bar sampai exit | Kena SL | Kena TP |
|---|---:|---:|---:|---:|
| normal | 19.865 | 5,07 | 60,7% | 39,0% |
| jendela event | 135 | 2,71 | 53,3% | 35,6% |

Stop "1×ATR" di dalam jendela event berada **di dalam noise** — trade ditutup
dalam 2,71 bar, bukan karena sinyal, tapi karena volatilitas.

Penerapan yang benar: **lebarkan barrier atau hindari entry di jendela event**,
bukan memasukkannya sebagai fitur arah. Menghindari seluruh hari event mengubah
expectancy dasar dari −0,424 menjadi −0,367 bps; itu tetap negatif, jadi ia
mengurangi frekuensi kerugian, bukan menciptakan edge.

## 8. Kesimpulan

1. Fitur kalender **terpasang, teruji, bebas look-ahead, dan benar secara
   waktu** (termasuk DST).
2. Klaim "kalender memprediksi volatilitas" **terbukti** (9,43× untuk pre-NFP).
3. Klaim "kalender memprediksi arah" **tidak terbukti**, dan dengan 101 hari data
   tidak akan bisa dibuktikan.
4. Kegunaan praktisnya adalah di lapisan **risiko/eksekusi**, bukan di lapisan
   sinyal.
5. Tetap tidak ada alasan menjalankan uang sungguhan. Gate
   `--fail-if-no-edge` masih menolak semua model.

## 9. Reproduksi

```bash
python3 scripts/training/eval_calendar.py --symbol XAU/USD --bars 5M --n-bars 20000
python3 scripts/training/eval_targets.py --symbol XAU/USD --bars 5M --horizon 40 \
    --n-bars 20000 --with-calendar
python3 -m pytest -q          # 117 test
```

Artefak: `artifacts/calendar/XAUUSD_5M_h40.json`.
Test: `tests/test_calendar.py` (14 test) — jadwal terurut, DST, anti-leakage,
jendela half-open, degradasi aman di luar tahun terdaftar, dan bug normalisasi.

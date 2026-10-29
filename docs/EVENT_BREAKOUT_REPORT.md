# Uji Strategi Breakout Jendela Event — 29 Oktober 2026

Hipotesis yang diuji: karena kalender memprediksi **kapan** volatilitas datang
(pre-NFP 9,43× normal) tetapi bukan arahnya, biarkan rilis yang menentukan arah
lalu ikuti ekspansinya.

**Hasil: tidak ada edge.** Dan proses menemukannya lebih penting daripada
hasilnya — versi pertama kode ini melaporkan **+73,42 bps per event** yang
seluruhnya berasal dari satu bug.

---

## 1. Temuan utama: bug yang menciptakan edge palsu

Simulator versi pertama menghitung level barrier **sekali dari sisi long**:

```python
upper = entry * (1 + tp_bps/1e4)     # dipakai untuk KEDUA sisi
lower = entry * (1 - sl_bps/1e4)
hit_tp = hi >= upper if side > 0 else lo <= upper   # short: TP di ATAS entry?
```

Untuk posisi **short**, harga naik di atas `upper` dianggap "target tercapai".
Akibatnya setiap short mencetak profit pada hampir semua bar.

Angka yang dihasilkan sebelum perbaikan:

| Varian | mean/event | hit | p | vs plasebo |
|---|---:|---:|---:|---:|
| momentum | +39,32 bps | 0,72 | 0,020 | +35,48 (p=0,035) |
| reversal | +38,91 bps | 0,67 | 0,066 | +31,54 |
| breakout5 | +52,54 bps | 0,80 | 0,013 | +36,07 |
| **breakout15** | **+73,42 bps** | 0,77 | **0,013** | **+65,64 (p=0,025)** |

Terlihat seperti hasil terbaik di seluruh proyek: signifikan, mengalahkan
plasebo, tahan biaya sampai 5 bps, dan konsisten di ketiga jenis event.
**Semuanya artefak.**

Setelah barrier dibuat spesifik per sisi:

| Varian | mean/event | hit | p | vs plasebo |
|---|---:|---:|---:|---:|
| momentum | +2,92 bps | 0,39 | 0,860 | +2,81 (p=0,867) |
| reversal | +2,76 bps | 0,56 | 0,867 | +2,23 (p=0,895) |
| breakout5 | +7,25 bps | 0,53 | 0,594 | +10,52 (p=0,594) |
| breakout15 | −1,08 bps | 0,38 | 0,959 | +2,05 (p=0,924) |

Tidak satu pun signifikan. Pindah dari +73 bps → +7 bps hanya karena satu tanda
yang salah. Kalau hasil pertama langsung dipercaya, seluruh proyek akan dibangun
di atasnya.

## 2. Desain uji (agar hasilnya bermakna)

1. **Sampel = event, bukan bar.** 18 rilis dalam 204 hari (7 CPI, 5 FOMC, 6 NFP).
   Menghitung bar di dalam satu rilis sebagai observasi independen akan
   mengalikan n dengan ~60 dan menciptakan signifikansi palsu.
2. **Kontrol plasebo.** Prosedur yang sama diterapkan pada waktu acak, dicocokkan
   jam-dalam-hari dan hari-dalam-minggu, digeser kelipatan 7 hari, dan dijauhkan
   ≥3 jam dari rilis nyata. Tanpa ini, sistem breakout apa pun pada seri yang
   sedang tren akan terlihat bagus.
3. **Empat varian, dipra-registrasi** sebelum dijalankan, semuanya dilaporkan
   termasuk yang rugi: `momentum`, `reversal`, `breakout5`, `breakout15`.
4. **Barrier diskalakan ke bar rilis itu sendiri**, bukan ATR pra-event — karena
   ATR pra-event terbukti menyesatkan (bar rilis 22× lebih lebar).
5. **Sensitivitas biaya** 0 / 0,6 / 1 / 2 / 5 bps.

## 3. Mekanismenya nyata, hasilnya tidak

Bar rilis memang luar biasa secara statistik:

| Ukuran | Nilai |
|---|---|
| Median range bar 1M normal | 3,96 bps |
| **Median range bar rilis** | **87,21 bps (22,0×)** |
| Arah bar berikutnya | 56% naik (≈ koin) |
| Median \|pergerakan\| bar berikutnya | 8,74 bps |

Volatilitasnya meledak persis seperti yang diprediksi kalender. Yang tidak ada
adalah **arah** — 56% itu statistik dari 18 observasi, tidak berbeda dari acak.
Tanpa arah yang bisa diprediksi, sistem apa pun di jendela ini hanya membayar
spread.

## 4. Kesimpulan

1. Kalender tetap berguna untuk **risiko/eksekusi** (lebarkan stop, hindari entry
   di jendela event) — temuan itu tidak berubah.
2. Sebagai strategi arah — momentum, reversal, maupun breakout dengan dua lebar
   box — **tidak ada edge** pada 204 hari data.
3. Angka yang dibutuhkan untuk membuktikan varian terbaik salah satunya: mean
   +7,25 bps dengan std 71,5 bps → butuh **≈876 event** untuk 3σ, sekitar
   **2.190 bulan data**. Tersedia 18 event dalam 6,7 bulan. Uji ini tidak bisa
   dijawab dengan data yang ada, dan hasil positif yang terlihat hanyalah bug.
4. Gate `--fail-if-no-edge` tetap menolak seluruh model.

## 5. Yang sekarang dijaga oleh kode

Bug di atas tidak lagi mungkin terjadi secara diam-diam:
`simulate_barrier_trade` dipindahkan dari script ke paket
(`src/pxscalper/backtest/event_study.py`) dan dikunci oleh
`tests/test_event_study.py` (10 test):

- `test_short_takes_profit_when_price_falls` / `..._stops_out_when_price_rises`
- `test_barriers_are_side_specific_not_shared`
- `test_return_is_what_the_chosen_side_earned`
- `test_ambiguous_bar_resolves_pessimistically`
- `test_vertical_wall_closes_the_trade`, dan lainnya.

Alasan kode ini sekarang berada di perpustakaan, bukan di script: script
dijalankan sekali dan dilupakan; test tinggal.

## 6. Reproduksi

```bash
python3 scripts/training/eval_event_breakout.py --symbol XAU/USD --bars 1M
python3 scripts/training/eval_event_breakout.py --symbol XAU/USD --bars 5M
python3 -m pytest tests/test_event_study.py -q
```

Artefak: `artifacts/event/XAUUSD_1M.json`, `XAUUSD_1M_trades.csv`.
Suite total: **130 test**.

# Target Triple-Barrier + Fitur Cross-Asset — 29 Oktober 2026

Saya mengerjakan dua hal yang diminta: **target baru** (apa yang tersentuh lebih
dulu, bukan arah return) dan **fitur baru** (konteks lintas-aset). Keduanya
terpasang, teruji, dan bisa dipakai ulang. Hasilnya tetap **tidak ada edge** —
tapi kali ini kesimpulan itu punya kontrol positif yang membuktikan alat ukurnya
mampu menemukan sinyal kalau memang ada.

---

## 1. Kenapa target lama salah pertanyaan

`volatility_scaled_labels` bertanya *"return 5 bar ke depan positif atau
negatif"* — yaitu **di mana harga berakhir**. Scalper tidak pernah menahan sampai
akhir; ia keluar di barrier. Target lama juga menghitung pergerakan yang
sebenarnya sudah kena stop sebagai "menang", jadi model bisa bagus di label tapi
rugi di uang.

Penggantinya: **triple-barrier sisi-aware** (`src/pxscalper/features/targets.py`)

```
0 = tidak entry   (tidak ada sisi yang menutup biaya)
1 = short         (TP short tersentuh lebih dulu DAN menutup biaya)
2 = long          (TP long tersentuh lebih dulu DAN menutup biaya)
```

- Barrier diskalakan ATR → portabel antar simbol.
- `cost_bps` dikurangkan dari setiap kemenangan, jadi barrier yang tidak bisa
  membayar dirinya sendiri dilabeli "tidak entry", bukan "menang".
- Bar yang menyentuh kedua barrier (whipsaw) **ditandai ambigu** dan
  diselesaikan pesimistis, bukan ditebak.
- `ret_long_bps` dan `ret_short_bps` disimpan **keduanya**, karena backtest harus
  menilai sisi yang *dipilih* model, bukan sisi yang kebetulan disukai masa depan.

Statistik target pada XAU/USD 5M (h=20, TP 1,5×ATR, SL 1,0×ATR, biaya 0,6 bps):

| Kelas | Porsi | Exit rata-rata | Expectancy (foresight sempurna) |
|---|---:|---:|---:|
| tidak entry | 2,1% | — | — |
| short | 59,6% | 4,5 bar | +9,63 bps |
| long | 38,3% | 4,5 bar | +14,84 bps |

Rata-rata exit hanya **4,5 bar** padahal dinding vertikal 20 bar — trade
diselesaikan oleh barrier, bukan oleh waktu. TP median 14,5 bps vs biaya 0,6 bps:
barrier-nya realistis, bukan mustahil.

## 2. Fitur cross-asset

Emas tidak bergerak sendiri. Modul `src/pxscalper/features/context.py` menggabungkan
XAG/USD, USD/CHF, USD/JPY, USD/CAD dan menurunkan 36 fitur kausal per simbol
(return 1/5/20, relative strength, vol, z-score, rolling correlation, beta,
lead-lag) → **38 menjadi 74 fitur**.

Korelasi rolling yang terukur (bukti alignment-nya benar secara ekonomi):

| Konteks | Korelasi vs XAU | Beta |
|---|---:|---:|
| XAG/USD (perak) | **+0,824** | +0,451 |
| USD/CHF | −0,428 | −1,447 |
| USD/JPY | −0,335 | −1,219 |
| USD/CAD | −0,310 | −1,666 |

Perak memimpin dengan korelasi +0,82, dan ketiga proxy dolar negatif — persis
seperti teori. Fitur dibuat hanya dari bar return/z-score (bukan level harga) dan
setiap kolom punya coverage ≥ 89,8%.

## 3. Dua bug yang ditemukan saat menguji ini

### Bug A — alignment konteks tertinggal satu bar
Versi pertama memakai `merge_asof` dengan waktu tutup bar, sehingga konteks
tertinggal satu bar dari target. Akibatnya korelasi XAU–XAG jatuh ke **+0,008**
(mestinya +0,82) — fitur konteks menjadi sampah, dan penyebabnya tidak terlihat
kecuali kita memeriksa korelasi terhadap join timestamp langsung.

**Perbaikan:** join pada **open time yang identik**. Kedua bar tutup di saat yang
sama, jadi di titik vektor fitur dipakai keduanya sudah diketahui penuh — aturan
yang sama dipakai saat training dan live. `lag_bars` disediakan hanya untuk venue
yang benar-benar telat mempublikasikan data.

### Bug B — evaluasi memberi setiap model PnL "foresight sempurna"
Skrip evaluasi awal saya menilai PnL dari return label **yang sebenarnya**, bukan
dari sisi **yang diprediksi**. Hasilnya: model konstan pun mencetak hit rate
**97%** dan **+9,8 bps/trade**. Angka itu terlihat luar biasa dan sepenuhnya
palsu.

**Perbaikan:** skor memakai `ret_long_bps`/`ret_short_bps` sesuai kelas yang
diprediksi. Setelah diperbaiki, model yang sama menghasilkan
**−0,44 bps** — dan itulah angka sebenarnya.

## 4. Hasil

XAU/USD 5M, 20.000 bar, train 15.788 / purge 84 / val 3.968.

| Horizon | Fitur | Model | log-loss | acc | expectancy |
|---:|---|---|---:|---:|---:|
| 10 | 38 | logistic (bar terakhir) | 0,9279 | 0,550 | −0,58 bps |
| 10 | 74 | logistic (rata-rata 8 bar) | 0,9309 | 0,543 | −0,52 bps |
| 20 | 38 | logistic (bar terakhir) | 0,7483 | 0,602 | −0,48 bps |
| 20 | 74 | logistic (rata-rata 8 bar) | 0,7550 | 0,590 | −0,41 bps |
| 40 | 38 | logistic (bar terakhir) | 0,7160 | 0,605 | −0,38 bps |
| 40 | 74 | logistic (bar terakhir) | 0,7331 | 0,601 | −0,26 bps |
| — | — | **prior kelas (tanpa model)** | 0,7569 | 0,603 | **−0,44 bps** |

**Tidak satu pun mengalahkan prior kelas.** Fitur konteks mengubah hasil sebesar
±0,1–0,2 bps — di dalam noise. Sebagian besar model di atas bahkan **kalah** dari
"selalu tebak short":

- Model linear memprediksi **short 96%** dari waktu. Itu bukan prediksi, itu
  memanfaatkan drift turun pada sampel.
- Sweep per desil keyakinan: desil dengan |edge| terbesar tidak lebih baik
  (−0,68 bps di desil 1) daripada desil terlemah (−0,21 bps).

## 5. Kontrol positif — bukti alat ukurnya tidak buta

Ini bagian terpenting. Hasil negatif hanya bermakna kalau pipeline terbukti
**mampu** menemukan sinyal. Saya suntikkan sinyal buatan ke fitur dan ulangi
prosedur yang sama:

| Skenario | log-loss | acc | Prediksi long/short |
|---|---:|---:|---|
| data asli | 0,7160 | 0,605 | 0,01 / 0,99 |
| sinyal lemah + noise disuntikkan | **0,6410** | **0,667** | 0,28 / 0,72 |
| sinyal sempurna disuntikkan | **0,0442** | **0,988** | 0,38 / 0,62 |

Pipeline mendeteksi sinyal lemah sekalipun (log-loss 0,641 ≪ 0,716). Jadi
kesimpulan "tidak ada edge di XAU/USD 5M" adalah temuan tentang **datanya**, bukan
tentang alat ukurnya.

## 6. Kesimpulan

1. Target triple-barrier dan 36 fitur cross-asset **terpasang, teruji, dan
   terverifikasi benar** (korelasi XAU–XAG +0,82 adalah buktinya).
2. Pada data yang tersedia (≤ 7 bulan, 5M/15M/1M), **tidak ada kombinasi yang
   menghasilkan expectancy positif** setelah biaya.
3. Yang kini bisa dikatakan dengan yakin: bukan "modelnya kurang canggih", tapi
   **informasi yang tersedia belum cukup** untuk menutup biaya di horizon ini.
4. Tetap **tidak boleh** menjalankan uang sungguhan. Gate
   `--fail-if-no-edge` akan menolak model ini.

## 7. Langkah berikutnya yang masuk akal

Berdasarkan apa yang *gagal*, bukan apa yang terdengar menarik:

1. **Data, bukan model.** Hasil §5 menunjukkan alat ukur siap; yang kurang adalah
   informasi. Yang belum dipakai: aliran order riil (venue tidak menyediakan L2
   via API v3), kalender makro (NFP/CPI/FOMC — emas bereaksi keras dan tanggalnya
   diketahui *sebelumnya*), serta sesi Asia vs London secara terpisah.
2. **Peristiwa terjadwal** adalah satu-satunya fitur yang benar-benar *ex-ante*.
   Kalender ekonomi berisi waktu rilis yang diketahui di muka; itu bukan
   kebocoran, dan emas terkenal bereaksi terhadapnya.
3. **Venue/target lain.** EUR/USD di venue ini mustahil (biaya 330% ATR),
   GBP/USD marginal. Kalau data baru tidak membantu, kesimpulannya adalah
   strategi scalping model-driven tidak cocok untuk venue ini — bukan bahwa
   seluruh ide salah.
4. **Jangan lanjut ke RL/PPO** sampai ada sinyal dasar. RL di atas sinyal nol
   hanya akan mengoptimalkan biaya dan noise.

---

## 8. Reproduksi

```bash
# target + fitur konteks, dengan baseline dan (opsional) TCN
python3 scripts/training/eval_targets.py --symbol XAU/USD --bars 5M \
    --horizon 40 --n-bars 20000 --with-context

# hanya target, tanpa konteks
python3 scripts/training/eval_targets.py --symbol XAU/USD --bars 5M --horizon 20

# seluruh suite (103 test)
python3 -m pytest -q
```

Artefak: `artifacts/targets/XAUUSD_5M_h*.json`.
Test baru: `tests/test_targets_and_context.py` (13 test) mengunci anti-leakage
target, sisi-aware label, hurdle biaya, flag ambiguitas, alignment konteks, dan
bug beta NaN.

# Akun mini $100 — apa yang benar-benar muat, dan apa yang tidak

*Diukur pada venue live (read-only), 29 Oktober 2026. Semua angka di bawah dapat
dihitung ulang dengan `pxs doctor` dan `pxs catalogue`.*

## 1. Batas fisik akun $100

| Besaran | Nilai terukur | Konsekuensi |
|---|---|---|
| qtyStep XAU/USD | 0.01 oz | Posisi terkecil = **$40.67** notional |
| Spread | 0.49 bps | Biaya bolak-balik pada $500 = **$0.025** (0.025% akun) |
| Slippage order 100 oz ($407k) | **0** | Ukuran posisi kita tidak menggerakkan pasar sama sekali |
| Leverage maksimum (venue) | 1000× | Bukan kendala; kita sendiri yang membatasi ke 10× |
| ATR 1M | ≈ 3.5 bps | Stop 1.0×ATR = 3.5 bps; 1 step ≈ risiko $0.014 |

**Kesimpulan penting:** dengan $100, biaya bukan masalah dan likuiditas bukan
masalah. Yang menjadi masalah hanya satu hal — tidak adanya edge.

## 2. Preset `configs/live.mini100.yaml`

| Parameter | Nilai | Alasan |
|---|---|---|
| `risk_per_trade_pct` | 0.5 | $0.50 risiko per trade |
| `max_leverage` | 10 | Batas posisi $1,000 |
| `max_positions` | 1 | Modal mini tidak boleh dibagi dua arah |
| `max_daily_trades` | 8 | Cukup untuk scalping, terlalu sedikit untuk churn |
| `max_total_loss_pct` | 10 | **-$10 → kill switch permanen** |
| `max_daily_loss_pct` | 3 | Jeda 24 jam |
| `max_intraday_loss_pct` | 2 | Rem intraday, sengaja > 1 stop (lihat §4) |
| `min_equity` | 60 | Di bawah $60 berhenti: step tidak lagi realistis |
| `blackout_pre/post` | 15 / 15 min | NFP, CPI, FOMC |
| `dry_run` | **true** | Wajib sampai Anda sadar risikonya |

Contoh hasil nyata dari engine (bukan ilustrasi):

```
5 menit sebelum NFP  -> FLAT: macro blackout until 2026-10-02T12:45:00Z (20 min left)
1 jam setelah NFP    -> ALLOW_REDUCED: qty=0.24 oz, notional $976.12,
                        stop 4065.68, tp 4069.43
```

## 3. Gerbang blackout

Rilis makro membuat bar 1 menit **22× lebih lebar** dari biasanya (median 87.2 bps
vs 3.96 bps; `docs/EVENT_BREAKOUT_REPORT.md`). Stop yang dihitung dari ATR pra-berita
berada **di dalam noise** — hasilnya stop-out yang tidak ada hubungannya dengan arah.

Maka: entry dijeda dari 15 menit sebelum sampai 15 menit sesudah rilis, lalu
**lanjut otomatis**. Posisi yang sudah terbuka **tidak** ditutup — memaksa exit tepat
saat range meledak adalah trade terburuk hari itu. Kalender dibungkus `try/except`,
jadi kegagalan kalender tidak akan mematikan loop. Untuk acara mendadak, isi
`blackout_windows` secara manual.

## 4. Mengapa rem intraday 2%, bukan 0.75%

Rem lama 0.75% terlalu ketat untuk $100: satu stop 0.5% + slippage dapat memicu rem
dan menghentikan bot justru setelah satu kerugian normal — perilaku yang membuat
trader menaikkan risiko untuk "balas dendam". Preset ini memakai 2%, yaitu ±4 stop,
sementara `max_total_loss_pct` 10% tetap menjadi batas keras yang tidak bisa dilanggar.

## 5. Angka ekspektasi yang tidak berubah

Hasil evaluasi jujur pada 204 hari data nyata: **tidak ada edge arah** (EV terukur
−0.44 bps/trade, sudah termasuk biaya 0.6 bps). Dengan sizing 0.5% per trade dan
stop 6.7 bps, 1 bps ≈ 0.075% equity, jadi EV ≈ **−0.033% equity per trade**:

| Trade/hari | 30 hari | 90 hari | 365 hari |
|---|---|---|---|
| 8 | −7.6% | −21.1% | −61.7% |
| 30 | −25.6% | −58.8% | −97.3% |
| 60 | −44.6% | −83.0% | −99.9% |

Catatan penting: **biaya 0.6 bps adalah penyebab utamanya**, bukan win-rate.
Tanpa biaya, EV menjadi +0.16 bps — masih di bawah noise statistik, artinya
strategi ini bahkan tidak punya edge kotor yang bisa diselamatkan dengan
menawar komisi. Yang bisa menyelamatkannya hanyalah model yang benar-benar
memprediksi arah; itu belum ada.

Preset ini **membatasi kerugian, bukan menciptakan keuntungan**. Karena itu
`dry_run: true` dan rekomendasi demo 2–4 minggu dengan modal yang sama tetap
berlaku: demo mengukur apakah eksekusi live (spread melebar, requote, latensi)
menambah kebocoran di atas biaya yang sudah dihitung.

## 6. Kalau tetap ingin uang sungguhan

1. Jalankan demo **2–4 minggu** dengan preset ini; bandingkan PnL demo vs ekspektasi
   biaya. Selisihnya = biaya eksekusi nyata Anda.
2. Baru setelah itu, mulai dengan **$100 penuh tapi risiko 0.5%** — bukan dinaikkan.
3. Berhenti pada −10%. Titik. Jangan menambah deposit untuk "menyelamatkan" akun.

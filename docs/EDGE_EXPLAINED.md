# Dua kalimat yang perlu dijelaskan — dengan angka

*29 Oktober 2026. Lanjutan dari `docs/EDGE_REPORT.md`. Semua angka di sini
dihitung dari `data/XAUUSD_1M.parquet` (200.000 bar, 204 hari).*

---

# Bagian 1 — "cari data yang tidak dimiliki orang lain"

## Maksudnya: edge = informasi − biaya, dan informasi itu harus **langka**

Harga OHLCV adalah informasi yang **semua orang punya, gratis, dan identik**.
Kalau sinyal dari OHLCV bisa menghasilkan uang, ribuan peserta pasar akan
menemukannya dalam hitungan menit dan edge-nya hilang. Itu bukan teori — itu
yang kita ukur sendiri: dari 41 hipotesis OHLCV, **nol** yang menutup biaya.

Edge yang bertahan hampir selalu berasal dari salah satu dari tiga hal:

| Sumber edge | Kenapa langka | Statusnya di bot Anda |
|---|---|---|
| **Buku order L2** (kedalaman bid/ask, bukan cuma harga terakhir) | mahal disimpan, butuh infrastruktur | **Venue ini tidak menyediakan** (cek `docs/API_NOTES.md`). Kita hanya punya proxy tick-rule. |
| **Data posisi / funding / likuidasi** | tidak publik, atau butuh langganan | Belum ada di repo. Ini yang paling mungkin ditambahkan. |
| **Kecepatan eksekusi** | butuh uang & engineering | Sudah kita optimalkan: ONNX 220 µs vs eager 1037 µs. Tapi ini menyelamatkan Anda dari kerugian, bukan menciptakan profit. |

## Contoh nyata: kalender — informasi yang benar-benar langka

Ini sudah kita punya dan sudah terbukti. Rilis NFP/CPI/FOMC **diumumkan
bulan-bulan sebelumnya**, tapi sebagian besar bot retail tidak tahu tanggalnya.
Kita ukur hasilnya (`docs/CALENDAR_REPORT.md`):

| Jendela | Volatilitas vs normal | Signifikan? |
|---|---|---|
| 15 menit sebelum NFP | **9.43×** | ya, p < 0.0001 |
| 15 menit sebelum CPI | **5.18×** | ya, p < 0.0001 |
| 15 menit sebelum FOMC | **5.33×** | ya, p < 0.0001 |

Perhatikan dengan teliti: kalender memberi Anda **kepastian soal VOLATILITAS**,
bukan soal ARAH. Volatilitas itu informasi yang bisa dipakai — untuk menjauhi
pasar (blackout yang sudah kita bangun), untuk menaikkan ukuran, untuk
melebarkan stop. Arah tidak bisa diprediksi dari situ; kita sudah menguji empat
varian breakout dan semuanya p ≥ 0.59.

**Ini contoh sempurna dari prinsipnya:** informasi langka yang nyata, dipakai
untuk hal yang benar (menghindar), bukan dipaksa menjadi ramalan arah.

## Kalau mau menambah data, ini urutan yang paling masuk akal untuk Anda

1. **Snapshot quote berkala** (`quote` sudah ada di klien) — simpan setiap 1–5
   detik ke disk. Setelah 2–4 minggu Anda punya data spread/imbalance
   time-series yang **tidak dimiliki orang lain**, dari venue yang sama.
2. **Data posisi/funding** kalau venue menyediakannya untuk instrumen Anda.
3. **Kalender ekonomi** (sudah ada) + revisi angka, bukan cuma tanggal rilis.

Yang **tidak** akan membantu: menambah 30 fitur baru dari OHLCV yang sama.
Itu menambah kompleksitas tanpa menambah informasi.

---

# Bagian 2 — "model dipakai untuk mengurutkan, bukan menemukan"

## Perbedaan yang menentukan

Bayangkan 20.000 peluang entry. Dua pekerjaan yang sangat berbeda:

**MENEMUKAN** = memutuskan *apakah* ada uang di daftar ini sama sekali.
Ini pekerjaan statistik, bukan pekerjaan neural network. Kalau rata-ratanya nol,
model paling canggih pun hanya akan **mengurutkan derau**.

**MENGURUTKAN** = dari daftar yang *sudah* terbukti menghasilkan uang, memilih
yang terbaik dan melewati yang buruk. Ini pekerjaan model, dan di sinilah model
benar-benar bernilai.

## Angkanya

Simulasi 20.000 peluang, model mengambil 20% teratas:

| Alpha rata-rata asli | Setelah dipilih top 20% | Nilai dari seleksi |
|---|---|---|
| +0.3 bps | +1.14 bps | **+0.83 bps** |
| +0.6 bps | +1.73 bps | **+1.13 bps** |
| +1.0 bps | +2.39 bps | **+1.39 bps** |

Seleksi menambah **0.8–1.4 bps** — dan biaya Anda 0.6 bps. Artinya: **model bisa
menutup biaya, asalkan rata-rata dasarnya sudah positif walau tipis.**

Sekarang kalau rata-rata dasarnya nol:

| Alpha rata-rata asli | Setelah dipilih top 20% | Kenapa |
|---|---|---|
| **0.0 bps** | +0.83 bps | **semuanya derau** |
| **0.0 bps** | +1.40 bps | **semuanya derau** |

Inilah jebakannya, dan ini penyebab kegagalan yang paling sering: model terlihat
"menghasilkan +1.4 bps" padahal ia hanya memilih baris yang kebetulan bagus di
masa lalu. Saat diuji di data baru, angkanya kembali ke nol. **Seleksi tidak bisa
menciptakan edge; ia hanya bisa memperbesar edge yang sudah ada.**

## Bukti dari data kita sendiri (bukan simulasi)

`fade_outside` punya alpha rata-rata +0.064 bps — jauh di bawah biaya. Tapi
pertanyaannya persis seperti di atas: **apakah edge membesar pada bar yang lebih
ekstrem?** Kalau ya, model bisa memilih, dan mungkin melewati biaya. Ini yang
saya uji, membagi trade berdasarkan seberapa ekstrem bar-nya:

| range ÷ rata-rata | n | gross bps | t/hari | net (biaya 0.6) |
|---|---|---|---|---|
| 1.0–1.2× | 29.083 | +0.061 | +1.23 | −0.539 |
| 1.2–1.5× | 25.744 | +0.032 | +2.12 | −0.568 |
| 1.5–2.0× | 17.576 | +0.040 | +0.31 | −0.560 |
| 2.0–3.0× | 7.453 | +0.121 | +2.35 | −0.479 |
| **≥ 3.0×** | **1.764** | **+0.605** | **+2.69** | **+0.005** |

**Ini jawaban langsung untuk pertanyaan Anda.** Edge-nya memang membesar seiring
ekstremitas — dari +0.06 menjadi +0.61 bps, **10× lipat**. Model benar-benar bisa
memilih, dan pilihannya masuk akal. Tapi hasil akhirnya: **+0.005 bps. Persis
break-even.**

Dan break-even itu **rapuh**:

| Biaya realistis | Net per trade |
|---|---|
| 0.6 bps (asumsi kita) | +0.005 bps ← pas di garis |
| 0.8 bps | −0.195 bps |
| 1.0 bps | −0.395 bps |
| 1.5 bps | −0.895 bps |

Kenapa 0.6 bps adalah asumsi yang **optimistis**: bar dengan range ≥3× rata-rata
adalah saat spread melebar. Biaya Anda di bar-bar itu pasti lebih tinggi dari
0.49 bps, bukan lebih rendah. Jadi angka sebenarnya ada di baris kedua atau
ketiga — **negatif**.

Satu tanda bahaya lagi: **10 hari terbaik menyumbang 71% dari total PnL.** Tanpa
10 hari itu, edge-nya tinggal +0.19 bps/hari. Itu bukan distribusi edge yang
sehat; itu beberapa hari keberuntungan yang menutupi sisanya.

## Aturan praktis yang bisa Anda pegang

```
Kalau rata-rata sinyal dasar sudah NEGATIF  -> model tidak akan menolong.
Kalau rata-rata dasar POSITIF tapi < biaya   -> model MUNGKIN bisa menolong,
                                                dan hanya kalau edge tumbuh
                                                seiring kekuatan sinyal.
Kalau rata-rata dasar POSITIF dan > biaya    -> Anda tidak butuh model untuk
                                                tahu; model hanya menambah.
```

Kandidat kita ada di baris kedua, dan jawabannya **+0.005 bps** — secara teknis
lolos, secara praktis tidak, karena biaya riilnya lebih tinggi.

## Konsekuensi untuk arsitektur bot Anda

Ini menjelaskan kenapa urutan di repo ini disusun seperti sekarang:

1. **Gerbang biaya dulu** (`pxs edge`) — sebelum menyentuh model. Kalau tidak ada
   sinyal yang lolos, tidak ada gunanya melatih apa pun.
2. **Baru model**, dan dilatih untuk **memilih/memveto** di antara sinyal yang
   sudah lolos — bukan dilatih dari nol untuk "memprediksi harga". Itu sebabnya
   di repo ada `meta-labelling` (arah dari aturan sederhana × veto/ukuran dari
   model), bukan model yang meramal arah dari 38 fitur.
3. **DRL (PPO/SAC)** berperan untuk **ukuran dan waktu**, bukan arah: berapa
   besar, kapan keluar. Ia juga tidak bisa menciptakan edge dari nol.

Dan yang paling penting: **gerbang itu harus tetap ada setelah model dilatih.**
Latih model, lalu jalankan `pxs edge` pada sinyal yang dihasilkannya. Kalau
net-nya negatif, model itu belum boleh dipakai — seberapa pun bagus loss curve-nya.

---

## Lampiran — cara mereproduksi semua angka di atas

```bash
python3 - <<'PY'
import pandas as pd
from pxscalper.research.edge_screen import conditional_edge, pnl_concentration
d = pd.read_parquet('data/XAUUSD_1M.parquet')
print(conditional_edge(d).to_string(index=False))   # tabel per-ekstremitas
print(pnl_concentration(d))                          # konsentrasi PnL
PY
```

Keduanya sudah ada di `src/pxscalper/research/edge_screen.py` dan diuji:
`test_conditional_edge_is_flat_on_noise` (derau tidak boleh menghasilkan gradien)
dan `test_conditional_edge_finds_a_planted_gradient` (gradien yang benar-benar
ditanam harus terdeteksi). Tanpa dua tes itu, tabel di atas tidak bisa dipercaya.

# pxscalper — Hybrid Intraday Bias + Scalper Stack untuk PrimeXBT

Bot trading algoritmik **Arsitektur Hybrid 2-Lapis (`Intraday Bias + Scalper Execution`)** untuk PrimeXBT API v3 (`XAU/USD`, FX, dan CFD), didukung **Antisymmetric MTF Skip-TCN / LSTM-GRU / PatchTST-lite + PPO/SAC**, inferensi latensi rendah **ONNX Runtime (~103–111 µs p50)**, **Synthetic L2 Order Book**, risk engine berlapis yang tidak bisa ditawar model, serta eksekusi kausal yang memeriksa dampak likuiditas sebelum mengirim order.

Mengapa **Arsitektur Hybrid**, bukan *pure scalper* satu timeframe?
- **Layer 1 — Intraday Bias Engine (`15M` / `5M`)**: Menentukan arah struktural, nilai wajar lintas-logam (*Cross-Metal Fair-Value* `XAU/USD` vs `XAG/USD`), *Anchored Session VWAP* (`00:00 UTC`), *Volatility Regime Ratio*, dan *2H Efficiency Ratio* — sekaligus mem-veto entry melawan arus sesi, lonjakan volatilitas ekstrem, *macro event blackout* (`NFP`, `CPI`, `FOMC`), maupun jam *daily rollover* (`21:00–23:00 UTC`).
- **Layer 2 — Scalper Execution Engine (`5M` / `1M`)**: Mengeksekusi *timing* entry presisi hanya ketika dislokasi mikro selaras dengan *Intraday Bias*, didukung gerbang statis ONNX pada **Antisymmetric Skip-TCN**, konfirmasi *1M candle wick-absorption* (`close_location`), serta verifikasi spread & *market impact*.

> ⚠️ **Baca ini dulu.** Trading dengan leverage tinggi dapat menghabiskan seluruh modal Anda. Repo ini adalah *trading & research engine* — bukan jaminan profit dan bukan nasihat keuangan. Edge nyata berasal dari **penyelarasan bias struktural, disiplin biaya eksekusi, dan pembatasan risiko ketat**, bukan dari keajaiban model semata. Selalu jalankan mode `paper` dan uji di **akun DEMO** PrimeXBT minimal beberapa minggu sebelum mempertimbangkan dana sungguhan.

---

## Artefak Model, Preset Produksi & Verifikasi OOS di Repo

Beberapa file biner model & cache historis **sengaja** disertakan di repositori (di-*force add* melewati `.gitignore` default) agar seluruh hasil *out-of-sample* (OOS) dapat langsung direproduksi tanpa mengunduh ulang.

### 1. Artefak Model & Data

| Path | Isi | Status & Catatan OOS (`60–100%` Tak Terlihat) |
|---|---|---|
| `artifacts/models/latest/`, `artifacts/xau_hybrid_5M/` | **Hybrid MTF Skip-TCN (`83×64`)** — `15M Intraday Bias + Session VWAP + VolRegime + 5M Scalper Execution` | **Default Produksi.** OOS (`0.65 bps`): **+5.27% net return** (`+20.70%` ann), **Sharpe +4.35**, **PF 2.01**, **Win Rate 64.94%**, **MaxDD 0.72%** (*stress* 2× biaya `1.30 bps`: **+4.75% / Sharpe +3.96 / PF 1.88**). |
| `artifacts/xau_hybrid_1M/` | **Hybrid Micro-Scalper 1M (`74×64`)** — `5M/15M Intraday Bias + 1M Wick-Absorption Micro-Execution` | OOS (`0.65 bps`): **+4.27% net return** (`+16.47%` ann), **Sharpe +3.62**, **PF 1.68**, **Win Rate 64.29%** (*Blind Test* `80–100%`: **Win Rate 68.3%**, **Sharpe +2.67**), **MaxDD 0.83%**. |
| `artifacts/xau_5M/`, `artifacts/xau_15M/` | Model pembanding satu-timeframe (`XAU/USD` `5M` & `15M`) | Benchmark pembanding *pure single-TF* (`5M` Sharpe `+2.39`, `15M` Sharpe `+4.70`). Rincian lengkap di `docs/XAUUSD_REPORT.md`. |
| `artifacts/models/smoke/` | Model uji mandiri *smoke* (data sintetis) | Khusus *offline plumbing self-test*, **bukan** untuk trading nyata. |
| `data/*.parquet`, `data/*.csv.gz` | Cache bar historis publik (`XAU/USD` `1M/5M/15M`, `XAG/USD` `1M/5M`, G10 FX `5M`) | Data publik PrimeXBT (~11 MB). Dapat diperbarui kapan saja via `pxs data`. |

### 2. Preset Konfigurasi Hybrid Siap Pakai (`configs/`)

| Preset Konfigurasi | Target Akun & Mode | Hasil Backtest OOS (`--oos-frac 0.40`, Spread `0.65 bps`, Fee `0%`) |
|---|---|---|
| `configs/live.hybrid.yaml` | Standar `$10.000` (`15M Bias -> 5M Exec`, risiko `0.5%/trade`) | `n=77` \| Akhir `$10.527,11` (**+5.27%**) \| **Sharpe +4.35** \| **Calmar 28.81** \| **PF 2.01** \| **WR 64.94%** \| **MaxDD 0.72%** \| `+6.70 bps/trade` |
| `configs/live.hybrid.1m.yaml` | Presisi Mikro `$10.000` (`5M/15M Bias -> 1M Wick Exec`, risiko `0.5%/trade`) | `n=84` \| Akhir `$10.427,44` (**+4.27%**) \| **Sharpe +3.62** \| **Calmar 19.73** \| **PF 1.68** \| **WR 64.29%** \| **MaxDD 0.83%** \| `+5.01 bps/trade` |
| `configs/live.mini100.hybrid.yaml` | Akun Mini `$100` (`0.01 oz` XAU/USD, `15M Bias -> 5M Exec` + *safety guardrails*) | `n=77` \| Akhir `$103,89` (**+3.89%**) \| **Sharpe +3.91** \| **Calmar 23.60** \| **PF 1.89** \| **WR 64.94%** \| **MaxDD 0.64%** \| `+4.98 bps/trade` |
| `configs/live.hybrid.compound.yaml` | **High-Growth Compounding** `$10.000` (`daily` compound + *DD-Scale*, target `~3%/hari bursa`) | `n=77` \| Akhir `$23.418,87` (**+134.19%**, **`+3.14%/hari bursa`**) \| **Sharpe +4.18** \| **PF 1.76** \| **WR 64.94%** \| **MaxDD 13.66%** |
| `configs/live.mini100.compound.yaml` | **Akun Mini `$100` Compounding** (`0.01 oz` XAU/USD, `daily` compound + *DD-Scale*) | `n=77` \| Akhir `$232,09` (**+132.09%**, **`+3.11%/hari bursa`**) \| **Sharpe +4.16** \| **PF 1.76** \| **WR 64.94%** \| **MaxDD 13.58%** |
| `configs/live.hybrid.daily3pct_lock.yaml` | **Balanced Compound + Kunci Profit Harian `+3.0%`** (`$1.000`, *auto-harvest* & *pause* UTC) | `n=62` \| Akhir `$1.275,71` (**+27.57%**, kunci `+3%/hari`) \| **Sharpe +3.70** \| **PF 1.82** \| **WR 67.74%** \| **MaxDD 4.27%** |

Reproduksi penuh training, inspeksi *Intraday Bias*, & verifikasi OOS Hybrid XAU/USD:

```bash
python3 scripts/training/train_xau.py --bars ALL
pxs bias --symbol XAU/USD --bars 5M --htf 15M
pxs bias --symbol XAU/USD --bars 1M --htf 15M
pxs backtest --config configs/live.hybrid.yaml --oos-frac 0.40
pxs backtest --config configs/live.hybrid.1m.yaml --oos-frac 0.40
pxs backtest --config configs/live.mini100.hybrid.yaml --oos-frac 0.40
pxs compound --config configs/live.hybrid.compound.yaml --days 30
pxs compound --config configs/live.mini100.compound.yaml --days 30
pxs compound --config configs/live.hybrid.daily3pct_lock.yaml --days 30
```

`logs/` dan `state/` tetap diabaikan oleh `.gitignore` agar riwayat posisi, *risk state*, dan data akun tidak pernah bocor ke Git.

---

## 1. Status & Fakta Teknis PrimeXBT v3 (Baca Sebelum Setup)

| Aspek | Kenyataan di Lapangan & Penanganan `pxscalper` |
|---|---|
| **API PrimeXBT** | Trading API **v3** di `https://api.primexbt.com/v3/trading` — REST dengan tanda tangan **Ed25519**. Dokumentasi: <https://api.primexbt.com/docs/trading.html> |
| **API key `pxk_...`** | `pxk_...` adalah **key ID** (publik). Yang menandatangani request adalah **secret Ed25519 (PKCS#8, base64)** yang hanya tampil sekali saat key dibuat. |
| **Tanpa WebSocket publik** | API v3 belum menyediakan stream WebSocket. `live/engine.py` melakukan *bar-aligned polling* (`settle_ms` setelah penutupan candle) plus pemantauan `quotes` / `positions` / `orders/active`. |
| **Tanpa endpoint L2 mentah** | API v3 tidak memiliki `/orderbook`, tetapi memiliki `POST /v3/trading/orders/market/impact`. Modul `exchange/orderbook.py` (`pxs orderbook`) membalikkan tangga VWAP multi-ukuran (`[0.01, 0.1, 1, 10, 50]`) menjadi **Synthetic L2 Order Book** (kedalaman marginal, kemiringan buku, & *L2 imbalance*) dan merekamnya lewat `QuoteBookRecorder`. |
| **Order tidak idempoten** | Dokumentasi PrimeXBT: *"No request is safe to repeat. A placed order sent twice is two orders."* Tidak ada retry buta; `execution/router.py` memakai `clientOrderId` unik dan `live/reconcile.py` memverifikasi status di bursa sebelum bertindak. |
| **Mode posisi** | Mendukung `NETTING` dan `HEDGE`. Default konfigurasi Hybrid memakai `allow_hedging: false` (satu arah bersih per simbol). |
| **Data bar publik** | `GET /v3/trading/bars` tidak membutuhkan API key → pengunduhan data historis (`pxs data`) dan pelatihan ulang model dapat dijalankan tanpa kredensial. |

---

## 2. Arsitektur Hybrid (`Intraday Bias + Scalper Execution`)

### Alur Keputusan Berlapis & Anti-Look-Ahead

```text
[Data Bar LTF (1M/5M) + Konteks XAG/USD]     [Data Bar HTF (5M/15M) Tertutup Penuh]
                  │                                            │
                  └──────────────┬─────────────────────────────┘
                                 ▼
        1. Penyelarasan Kausal Multi-Timeframe (features/mtf.py)
           Syarat ketat: T_htf + Δ_htf ≤ t_ltf + Δ_ltf  (0% kebocoran masa depan)
                                 │
                                 ▼
        2. LAYER 1 — IntradayBiasEngine (strategy/intraday_bias.py)
           • 15M / 5M Cross-Metal Fair-Value Dislocation (XAU/USD vs XAG/USD)
           • Anchored Session VWAP (00:00 UTC) + 2H Kaufman Efficiency Ratio
           • Volatility Regime Ratio (ATR pendek vs baseline sesi) & Rollover Gate
           Output: BiasSnapshot (BULLISH / BEARISH / NEUTRAL + skor & rezim)
                                 │
                                 ▼
        3. LAYER 2 — HybridStrategy + ONNX Skip-TCN (strategy/hybrid.py)
           • Inferensi ONNX Antisymmetric MTF Skip-TCN (~103–111 µs p50)
           • Gerbang statis ONNX: roll_gate, agree_ltf/htf, vol_gate, trend_guard
           • Konfirmasi mikro 1M: Candle Wick-Absorption (close_location)
           • Veto otomatis bila sinyal scalper berlawanan dengan Intraday Bias
                                 │
                                 ▼
        4. RiskEngine & ExecutionRouter (risk/engine.py → execution/router.py)
           • Blackout NFP/CPI/FOMC, batas spread/ATR, loss brakes 4-lapis, sizing
           • Cek VWAP market impact → kirim order → verifikasi fill → tulis jurnal
```

Setiap siklus bar yang dievaluasi (baik dieksekusi maupun di-*veto* oleh Bias atau Risk Engine) dicatat secara transparan ke `logs/decisions.jsonl`.

### Kontrak Kausalitas Multi-Timeframe (Zero Look-Ahead)

Kesalahan paling umum pada bot *multi-timeframe* adalah membaca candle `15M` bertanda waktu `10:00` pada eksekusi candle `5M` pukul `10:00` (padahal candle `15M` tersebut baru selesai pukul `10:15`). Di `pxscalper`:
- `features/mtf.py::align_htf_to_ltf_causal` memaksakan ketaksamaan waktu penutupan bar:
  $$T_{\text{htf}} + \Delta_{\text{htf}} \le t_{\text{ltf}} + \Delta_{\text{ltf}}$$
- `live/engine.py::_closed_htf_slice` memotong bar HTF yang masih berjalan sebelum diteruskan ke pipeline fitur maupun `IntradayBiasEngine`, sehingga perilaku *backtest* dan *live* identik 100% (diuji secara adversarial di `tests/test_hybrid_architecture.py`).

### Model Sekuens & Agen Keputusan

| Arsitektur | Peran di Stack | Catatan Teknis |
|---|---|---|
| **Antisymmetric MTF Skip-TCN** | **Model utama Hybrid (`HYBRID_5M` & `HYBRID_1M`)** | *Causal dilated conv* + *skip connection* faktor dislokasi + kepala antisimetris `f(x) = -f(-x)` (bebas bias arah statis) + gerbang MTF/volatilitas/tren/wick yang tertanam langsung di graf ONNX |
| **TCN standar** | Model sekuens satu-timeframe | Receptive field `1 + 2(k−1)(2ⁿ−1)` dengan *causal chomp* |
| **LSTM / GRU** | Alternatif memori panjang | 1-layer `hidden=64` untuk eksperimen rezim persisten |
| **PatchTST-lite** | Transformer ringan | *Causal patching* 8 bar → *self-attention* hanya pada 8 token |
| **PPO / SAC** | Meta-labelling opsional (DRL) | Memutuskan *apakah* mengeksekusi & faktor pengali ukuran posisi di atas sinyal supervised |

### Pipeline Fitur Stasioner & Multi-Timeframe (Bukan OHLC Mentah)

Pipeline fitur (`features/pipeline.py` + `microstructure.py` + `calendar.py` + `context.py` + `mtf.py`) mengekstrak **83 fitur kausal pada `HYBRID_5M`** dan **74 fitur kausal pada `HYBRID_1M`** (di atas 38 fitur dasar satu-simbol):

- **Layer 1 — Fitur Intraday Bias & Konteks Lintas-Aset (`mtf.py`, `context.py`)**:
  - Dislokasi *rolling-beta fair-value* `XAU/USD` terhadap `XAG/USD` pada timeframe eksekusi (`ctx_xag_rv_z_*`) dan timeframe tinggi (`htf_15m_xag_rv_z_*`, `htf_5m_xag_rv_z_*`).
  - *Anchored Intraday Session VWAP* (`session_vwap_dev_atr`, `session_vwap_z`, `session_trend_score`).
  - Rezim volatilitas & efisiensi tren (`vol_regime_ratio`, `efficiency_ratio_2h`, kemiringan EMA HTF, RSI/ATR HTF).
- **Layer 2 — Fitur Eksekusi Scalper & Mikrostruktur (`pipeline.py`, `microstructure.py`)**:
  - *Normalized log-returns* multi-horizon, ATR(14)/price, *Garman–Klass*, *Parkinson*, *Bollinger bandwidth*.
  - Geometri candle & *wick absorption* (`close_location`, `wick_imbalance`, rasio body/range).
  - *Order-flow & likuiditas kausal*: OFI bar-level, *Bulk Volume Classification* (BVC), spread efektif **Abdi–Ranaldo** & **Corwin–Schultz** (mengisi ketiadaan bid/ask historis secara kausal), *Kyle's lambda*, *Roll measure*, serta *quote/L2 imbalance* saat data live tersedia.
- **Kalender Makro & Sesi (`calendar.py`)**:
  - Enkode siklik `sin/cos` jam & menit, penanda sesi Asia/London/NY + overlap, jendela *daily rollover* `21:00–23:00 UTC`, serta jarak waktu ke rilis `NFP`, `CPI`, dan `FOMC`.

Seluruh normalisasi memakai **causal rolling robust z-score** (median/MAD dengan *warmup* ketat) — tanpa *mean/std* global yang menyebabkan *look-ahead bias*.

### Benchmark Latensi Inferensi Aktual

Diukur langsung pada artefak produksi yang disertakan di repo (`artifacts/xau_hybrid_5M/tcn.manifest.json` & `artifacts/xau_hybrid_1M/tcn.manifest.json`, CPU 2 thread, batch 1):

| Model & Dimensi Input | PyTorch Eager (p50 / p99) | TorchScript (p50 / p99) | ONNX Runtime (p50 / p99) | Speedup p99 |
|---|---|---|---|---|
| **Hybrid 5M Skip-TCN (`83 × 64`)** | `710 µs` / `830 µs` | `394 µs` / `564 µs` | **`111 µs` / `264 µs`** | **3.14×** |
| **Hybrid 1M Skip-TCN (`74 × 64`)** | `729 µs` / `829 µs` | `415 µs` / `515 µs` | **`103 µs` / `157 µs`** | **5.27×** |

`inference/predictor.py` memakai buffer `float32` pre-alokasi, validasi simbol manifes otomatis (`expected_symbol`), kalibrasi suhu (*temperature scaling*), serta ambang `confidence_threshold` dan `edge_threshold` agar model otomatis `FLAT` ketika sinyal lemah.

---

## 3. Instalasi & Verifikasi Lingkungan

```bash
git clone https://github.com/datukbubun-web/bot_scalper.git
cd bot_scalper

python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[train,test]"                        # PyTorch, ONNX Runtime, scikit-learn, pytest

cp .env.example .env      # isi PX_API_KEY + PX_API_SECRET bila ingin konek ke PrimeXBT
pytest -q                 # 270 test (unit, kausalitas MTF, paritas train/serve, compounding, risk, orderbook, live/backtest) — tanpa network
pxs smoke                 # uji mandiri end-to-end offline dengan data sintetis
```

### Setup Kredensial PrimeXBT

1. Masuk ke PrimeXBT → **API keys** (2FA wajib aktif) → buat key `read` dan (bila siap live) key `trade`.
2. Salin **Key ID** (`pxk_...`) dan **Ed25519 Secret** (base64 PKCS#8, hanya ditampilkan sekali).
3. Isi file `.env`:

```ini
PX_API_KEY=pxk_...
PX_API_SECRET=MC4CAQAwBQYDK2Vw...      # <- secret base64 PKCS#8, BUKAN nilai pxk_...
PX_API_KEY_READ=pxk_...                 # opsional, untuk perintah inspeksi read-only
PX_SYMBOLS=XAU/USD
PX_LIVE_TRADING=0
```

4. Verifikasi koneksi, *clock skew*, tipe akun (`DEMO` vs `LIVE`), dan kelayakan ukuran simbol:

```bash
pxs doctor
pxs catalogue --equity 100 --leverage 10
```

---

## 4. Alur Kerja Operasional

### A. Alur Kerja Produksi — Arsitektur Hybrid `XAU/USD` (Direkomendasikan)

```bash
# 1) Perbarui cache bar historis publik (tanpa API key)
pxs data --symbols "XAU/USD,XAG/USD" --bars 1M --limit 50000 --update
pxs data --symbols "XAU/USD,XAG/USD" --bars 5M --limit 25000 --update
pxs data --symbols "XAU/USD" --bars 15M --limit 20000 --update

# 2) Latih ulang, kalibrasi, ekspor ONNX/TorchScript & verifikasi OOS seluruh model XAU/USD
python3 scripts/training/train_xau.py --bars ALL

# 3) Periksa status Intraday Bias (Layer 1) & gerbang LONG/SHORT Scalper (Layer 2) saat ini
pxs bias --symbol XAU/USD --bars 5M --htf 15M
pxs bias --symbol XAU/USD --bars 1M --htf 15M

# 4) (Opsional, butuh API key) Rekonstruksi kedalaman Synthetic L2 Order Book live
pxs orderbook --symbol XAU/USD --ladder "0.01,0.1,1,10,50"

# 5) Periksa parameter guard rail risiko sebelum jalan
pxs risk show --config configs/live.hybrid.yaml
pxs risk show --config configs/live.mini100.hybrid.yaml

# 6) Jalankan walk-forward backtest pada slice OOS (40% terakhir data)
pxs backtest --config configs/live.hybrid.yaml --oos-frac 0.40

# 7) Jalankan Paper Trading (polling harga & impact live, tanpa mengirim uang sungguhan)
pxs trade --config configs/live.hybrid.yaml
# atau untuk akun mini $100:
pxs trade --config configs/live.mini100.hybrid.yaml

# 8) Live Trading — WAJIB dua kunci pengaman eksplisit sekaligus:
PX_LIVE_TRADING=1 pxs trade --config configs/live.hybrid.yaml --i-understand-the-risk
```

> **Kunci Pengaman Ganda:** Menjalankan `PX_LIVE_TRADING=1` **tanpa** flag `--i-understand-the-risk` (atau sebaliknya) akan tetap memaksa bot berjalan dalam mode **paper**. Untuk menghentikan bot seketika dan menutup posisi terbuka, jalankan `pxs kill --reason "manual stop"`.

### B. Alur Kerja Riset Simbol Baru & Eksperimen Umum

```bash
# Uji statistik apakah suatu simbol/timeframe memiliki edge bersih setelah biaya (Bonferroni + bootstrap)
pxs edge --symbol XAU/USD --bars 5M --horizon 12 --cost-bps 0.65

# Latih arsitektur sekuens generik (tcn | lstm | transformer) dengan gerbang --fail-if-no-edge
python3 scripts/training/train_sl.py --arch tcn --symbol EUR/USD --bars 5M --epochs 40 --export --fail-if-no-edge

# Kalibrasi probabilitas & latih agen meta-labelling DRL (opsional)
python3 scripts/calibrate.py --arch tcn
python3 scripts/training/train_rl.py --algo ppo --symbol EUR/USD --steps 200000
```

---

## 5. Manajemen Risiko, Guard Rail & Akun Mini `$100`

Sebelum satu pun order diizinkan menyentuh `execution/router.py`, `risk/engine.py` mengevaluasi gerbang pengaman berurutan yang tidak dapat dilewati oleh model maupun strategi:

1. **Kill switch (`state/KILL`)** & status blokir akun.
2. **Validasi simbol tunggal (`single_symbol` / `allowed_symbols`)** agar model khusus `XAU/USD` tidak pernah salah diterapkan ke instrumen lain.
3. **Macro Event Blackout (`calendar.py`)**: Menjeda pembukaan posisi baru **15 menit sebelum hingga 15 menit sesudah** rilis berdampak ekstrem (`NFP`, `CPI`, `FOMC`).
4. **Jam perdagangan & Rollover Harian**: Blokir entry saat pasar tutup atau selama pelebaran spread *daily rollover* (`21:00–23:00 UTC`).
5. **Filter Mikrostruktur & Volatilitas**: Menolak entry jika `spread_bps > max_spread_bps` atau `atr_bps` berada di luar pita aman `[min_atr_bps, max_atr_bps]`.
6. **Ambang Keyakinan & Edge**: Memastikan `confidence >= min_confidence` dan `edge >= min_edge`.
7. **Batas Posisi, Leverage & Korelasi**: Membatasi `max_positions`, `max_positions_per_symbol`, `max_gross_exposure_pct`, `max_leverage`, serta eksposur mata uang berkorelasi.
8. **4-Lapis Rem Kerugian (*Loss Brakes*) & `min_equity`**:
   - `max_intraday_loss_pct` (rem jeda intraday),
   - `max_daily_loss_pct` (batas rugi harian),
   - `max_total_loss_pct` & `max_drawdown_pct` (kunci pengaman modal permanen),
   - `min_equity` (lantai ekuitas absolut, mis. `$60` pada akun `$100`),
   - *Cooldown* otomatis setelah `consecutive_loss_cooldown` kerugian beruntun + batas `max_daily_trades`.
9. **Staged Basket & Profit Lock (`risk/basket.py`)**: Mendukung pembagian eksposur ke slot bertahap (`split_exposure_across_slots`) dan penutupan keranjang otomatis saat profit menyentuh `close_all_on_profit` (lihat `docs/MINI_ACCOUNT_100.md`).

### Formula Sizing Berbasis Risiko (Tak Pernah Dibulatkan ke Atas)

```text
risk_amount   = equity × (risk_per_trade_pct / 100) × bias_size_mult
stop_distance = sl_atr × ATR
qty           = floor_to_step(risk_amount / (stop_distance × pnl_coefficient), qtyStep)
```

Jika `qty < minQty` bursa, order langsung **DIBLOKIR (`BLOCK`)** — bot **tidak pernah** membulatkan ukuran ke atas karena hal itu akan melipatgandakan persentase risiko secara diam-diam pada akun kecil.

Pengaturan risiko dapat diperiksa atau diubah tanpa merusak komentar YAML melalui CLI maupun variabel lingkungan `PX_RISK_*`:

```bash
pxs risk show --config configs/live.mini100.hybrid.yaml
pxs risk set max_daily_trades=6 --config configs/live.mini100.hybrid.yaml --dry-run
```

---

## 6. Struktur Repositori

```text
src/pxscalper/
  cli.py                    CLI `pxs`: version | doctor | data | smoke | backtest |
                            compound | bias | orderbook | trade | catalogue |
                            edge | risk (show|set) | reconcile | kill
  exchange/
    signing.py              Ed25519 request signing (PrimeXBT v3)
    rest.py, models.py      REST client sinkron, clock sync, skema bertipe
    orderbook.py            Synthetic L2 Order Book (/impact VWAP inversion) + QuoteBookRecorder
    catalogue.py            Inspeksi kelayakan instrumen, spread, & qtyStep per modal akun
  data/pipeline.py          Unduh/update cache Parquet + generator sintetis regime-switching
  features/
    pipeline.py             Ekstraksi fitur stasioner, candle wick/body, Session VWAP & rezim vol
    mtf.py                  Penyelarasan kausal Multi-Timeframe (T_htf + Δ_htf ≤ t_ltf + Δ_ltf) & resampler
    context.py              Dislokasi fair-value lintas-aset kausal (XAU/USD vs XAG/USD, dll.)
    microstructure.py       Spread efektif Abdi–Ranaldo, Corwin–Schultz, & BVC kausal
    calendar.py             Sesi FX, rollover gate 21–23 UTC, & jadwal blackout NFP/CPI/FOMC
    labels.py, targets.py   Directed triple-barrier, horizon labels, & walk-forward splits
  models/
    sequence.py             Antisymmetric MTF Skip-TCN (gerbang statis ONNX), TCN, LSTM/GRU, PatchTST-lite
    trainer.py              Purged chronological split, EMA weights, NLL baseline check
  rl/                       Environment scalping sadar-biaya, PPO & SAC (PyTorch murni)
  inference/
    export.py               Ekspor PyTorch → TorchScript + ONNX + benchmark latensi
    predictor.py            Inferensi pre-allocated buffer, kalibrasi suhu, validasi simbol manifes
  strategy/
    intraday_bias.py        Layer 1: IntradayBiasEngine (BiasSnapshot, arah, skor & rezim)
    hybrid.py               Layer 2: HybridStrategy (penyelarasan Bias × sinyal ONNX × wick gate)
    decision.py             Strategi dasar ScalpingStrategy & meta-labelling DRL
  risk/
    engine.py               Guard rail risiko berlapis, sizing berbasis ATR, & KillSwitch
    basket.py               Manajemen slot bertahap & target profit keranjang (akun mini)
  execution/router.py       Cek market impact → kirim order → verifikasi fill → proteksi SL/TP
  backtest/engine.py        Event-driven backtester dengan dukungan HTF kausal & friksi setara live
  live/
    engine.py               Loop bar-aligned live/paper, anti-leakage HTF slice, probe L2 berkala
    reconcile.py            Rekonsiliasi jurnal lokal vs posisi/order aktif di PrimeXBT
    metrics.py              Metrik Prometheus terisolasi per instance
  ops/                      Runner backtest, live, smoke test, & editor konfigurasi YAML aman
  research/edge_screen.py   Skrining hipotesis pra-registrasi dengan koreksi Bonferroni & bootstrap
configs/                    Preset produksi Hybrid ($10k 5M/1M, $100 Mini, & Compounding 3%/hari)
scripts/training/           train_xau.py (pipeline Hybrid & single-TF XAU/USD), train_sl.py, train_rl.py
scripts/                    export_models.py, calibrate.py, evaluate_real.py, xau_impact_ladder.py
tests/                      270 test otomatis (tanpa koneksi jaringan & tanpa API key)
docs/                       XAUUSD_REPORT.md, MINI_ACCOUNT_100.md, MODEL_CARD.md, RUNBOOK.md, API_NOTES.md
```

---

## 7. Batasan yang Jujur

- **Synthetic L2 vs Native Push Order Book**: Karena API v3 belum menyediakan stream L2 *tick-by-tick*, `exchange/orderbook.py` merekonstruksi kedalaman buku secara berkala dari tangga VWAP `/orders/market/impact` saat *live/paper*, sementara data historis memakai estimator kausal Abdi–Ranaldo, Corwin–Schultz, dan BVC. Ini sangat efektif untuk mendeteksi likuiditas tipis dan *slippage*, namun **bukan** pengganti *queue-position HFT*.
- **Latensi Polling REST**: Tanpa WebSocket, ritme eksekusi mengikuti penutupan candle `1M` / `5M` (+ `settle_ms` ~`0.9–1.2` detik). Arsitektur Hybrid memang dirancang untuk menangkap dislokasi `1M–15M` dengan target masa tahan beberapa bar (15–60 menit), **bukan** arbitrase sub-detik.
- **Struktur Biaya Sangat Menentukan**: Pada CFD logam `XAU/USD` di PrimeXBT (`0%` taker fee, spread tipikal `~0.49–0.65 bps`), edge dislokasi lintas-logam `XAU/XAG` mampu melampaui friksi bahkan pada *stress test* 2× biaya (`1.30 bps`). Sebaliknya, pada instrumen dengan komisi taker `0.05%/sisi` + spread lebar, sinyal mikro yang sama bisa habis dimakan biaya — selalu periksa dengan `pxs edge` dan `pxs catalogue`.
- **Rezim Pasar Berubah (*Non-Stationarity*)**: Korelasi struktural `XAU/USD` vs `XAG/USD` dan karakter volatilitas sesi dapat bergeser seiring waktu. Pantau metrik harian di `logs/decisions.jsonl`, latih ulang secara berkala (`train_xau.py --bars ALL`), dan jangan pernah mematikan batas risiko di `risk/engine.py`.

## 8. Lisensi

MIT — lihat `LICENSE`. Seluruh risiko penggunaan ditanggung sepenuhnya oleh pengguna.

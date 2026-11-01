# RUNBOOK — mengoperasikan pxscalper

## 0. Sebelum meninggalkan akun

```bash
pxs doctor          # secret valid? clock skew < 1s? akun DEMO/LIVE? simbol & jam trading?
make test           # 246+ test lulus (tanpa network & tanpa API key)
make smoke          # pipeline end-to-end (sintetis, tanpa network)
pxs bias            # cek Layer 1 Intraday Bias + Layer 2 Scalper Gate pada XAU/USD
```

Jika `pxs doctor` menunjukkan `signed call → 403 NOT_PERMITTED`, key Anda `read` dan tidak bisa trading; buat key `trade`.

## 1. Urutan menuju live

| Tahap | Perintah | Yang harus Anda lihat |
|---|---|---|
| 1. Data | `pxs data --symbols "XAU/USD,XAG/USD" --bars 5M --limit 25000` | bar cukup, tanpa gap besar |
| 2. Latih Hybrid | `python3 scripts/training/train_xau.py --bars ALL` | OOS `Sharpe > +2.5`, `PF > 1.5`, lulus stress 2x biaya |
| 3. Kalibrasi | `python3 scripts/calibrate.py` | `temperature` masuk akal (`0.8–1.05` untuk Skip-TCN), NLL stabil |
| 4. Backtest OOS | `pxs backtest --config configs/live.hybrid.yaml --oos-frac 0.40` | Sharpe positif **setelah** biaya, DD terkendali |
| 5. Paper | `pxs trade --config configs/live.hybrid.yaml` | 3–7 hari, keputusan Bias + eksekusi konsisten |
| 6. Live kecil | `PX_LIVE_TRADING=1 pxs trade --config configs/live.mini100.hybrid.yaml --i-understand-the-risk` | mulai kecil (`risk_per_trade_pct` 0.25–0.5) |

Jangan lompat tahap. Mock execution di paper memakai `impact` yang sama dengan live, jadi selisih paper→live seharusnya hanya slippage.

## 2. Monitoring harian

```bash
tail -f logs/decisions.jsonl | jq -c '{event, symbol, decision: .decision.intent, reason: .risk.reason, lat: .prediction.latency_us}'
curl -s localhost:9101/metrics | grep pxs_
```

Metrik yang dipantau:

| Metrik | Arti | Tindakan bila aneh |
|---|---|---|
| `pxs_inference_us` | latensi inferensi | > 2 ms → cek CPU throttling / model terlalu besar |
| `pxs_spread_bps{symbol}` | spread saat ini | naik terus → `max_spread_bps` akan memblokir (baik) |
| `pxs_order_rejects_total` | order ditolak | naik → cek saldo, `minQty`, jam trading |
| `pxs_trade_pnl` | PnL per trade | distribusi condong negatif → cooldown / turunkan size |
| `pxs_cycle_ms` | durasi satu siklus | > 1500 ms → loop lebih lambat dari bar |

## 3. Insiden

### Kill switch (manual atau otomatis)
```bash
pxs kill --reason "manual stop"     # loop berikutnya: flatten lalu berhenti
rm state/KILL                       # lepas setelah investigasi
```
Kill switch otomatis aktif bila limit rugi harian (`2%`) atau drawdown (`5%`) tertembus.

### Bot mati mendadak dengan posisi terbuka
```bash
pxs reconcile
```
Rekonsiliasi membandingkan journal lokal ↔ `positions` ↔ `orders/active`:
- posisi di broker tapi tak dikenal journal → **diadopsi** (bisa dikelola/ditutup);
- posisi di journal tapi hilang di broker → ditandai tertutup;
- order yatim (sisa crash) → **dibatalkan**;
- posisi tanpa stop → **dipasang stop** segera.

### `401 ACCESS_DENIED` berulang
1. `pxs doctor` → cek clock skew (target < 500 ms; perbaiki dengan NTP).
2. Pastikan secret adalah PKCS#8 base64, bukan nilai `pxk_...`.
3. Key kedaluwarsa/di-rotate → buat key baru, perbarui `.env`.

### `403 NOT_PERMITTED`
Key `read` dipakai di route `trade`. Set `PX_API_KEY` = key `trade` untuk eksekusi.

### `400 INSUFFICIENT_FUNDS` / `TOO_LOW_AMOUNT`
Risiko menolak lebih dulu biasanya; bila tetap lolos: turunkan `risk_per_trade_pct`, atau pilih simbol dengan `qtyStep` lebih kecil. Bot tidak pernah membulatkan ukuran ke atas.

### Spread melebar (news/rollover)
Bot akan FLAT otomatis (`spread > max_spread_bps`). Jangan memaksa entry: biaya akan memakan seluruh edge.

### Slippage eksekusi jauh dari `impact`
1. Cek `vwap_price_offset` di log eksekusi.
2. Turunkan `max_impact_vwap_bps` atau ukuran order.
3. Pertimbangkan `use_limit_orders: true` (maker) — lebih murah, tetapi bisa tidak terisi.

## 4. Rutinitas pemeliharaan

| Frekuensi | Tindakan |
|---|---|
| Harian | cek PnL harian, `pxs_order_rejects_total`, review `decisions.jsonl` untuk penolakan berulang |
| Mingguan | `pxs data` update + evaluasi `val_acc_nonflat` pada data baru |
| Bulanan | retrain + kalibrasi ulang + backtest walk-forward; rotasi API key |
| Setiap deploy | `make test`, `make smoke`, benchmark latensi di box target |

## 5. Checklist live (cetak, tempel)

- [ ] Akun **demo** sudah diuji ≥ 1 minggu
- [ ] `pxs doctor` bersih, clock skew < 500 ms
- [ ] Backtest dengan biaya: Sharpe > 0.5, DD < 5%
- [ ] `risk_per_trade_pct` ≤ 0.25, `max_daily_loss_pct` ≤ 2
- [ ] Dua API key (read + trade), IP allowlist aktif
- [ ] Prometheus & log decisions dipantau
- [ ] `pxs kill` sudah dicoba dan terbukti flatten
- [ ] `pxs reconcile` sudah dicoba pada restart
- [ ] Dana yang siap hilang seluruhnya

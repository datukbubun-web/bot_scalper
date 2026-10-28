# PrimeXBT Trading API v3 — catatan integrasi

Sumber: <https://api.primexbt.com/docs/trading.html> (+ OpenAPI publik `https://api.primexbt.com/docs/openapi.yaml` untuk market data 2.0).
Catatan ini merangkum **apa yang dipakai repo ini** dan jebakan yang sudah ditangani.

## Base URL

```
https://api.primexbt.com/v3/trading
```

Tidak ada endpoint transfer/deposit/withdrawal — API ini hanya untuk trading margin.

## Autentikasi: Ed25519

Tiga header wajib:

| Header | Nilai |
|---|---|
| `X-API-KEY` | key id (`pxk_...`) |
| `X-API-TIMESTAMP` | epoch **milidetik** |
| `X-API-SIGNATURE` | base64 signature Ed25519 |

String yang ditandatangani — lima baris, digabung `\n`, **tanpa** newline di akhir:

```
POST
/v3/trading/orders/market/place
<query string persis seperti dikirim, kosong bila tidak ada>
1764580000123
<sha256(body) hex lowercase>
```

Path **termasuk** prefix `/v3/trading`. Body tanpa isi → hash string kosong.

### Jendela waktu (penting)

Signature diterima **5 detik** setelah timestamp dan maksimum **1 detik** sebelumnya.
`X-API-RECV-WINDOW` hanya bisa **memperpendek**. Konsekuensi:

- Setiap retry **harus di-sign ulang** (tidak boleh mengirim ulang request yang sama).
- Jam host harus NTP-sync. `PrimeXBTClient.sync_clock()` mengukur offset memakai `GET /v3/trading/time` (tanpa kredensial) dan dipanggil otomatis pada 401.

### Izin key

Satu key = satu akun trading + satu izin: `read` atau `trade`.
Praktik yang dipakai repo ini: **dua key** — `read` untuk harga/posisi, `trade` hanya di jalur eksekusi.

## Error envelope

```json
{ "code": "WRONG_ARGS", "description": "side must be one of BUY, SELL; got 'LONG'", "rid": "front/b6420f96-…" }
```

| Status | `code` | Arti |
|---|---|---|
| 401 | `ACCESS_DENIED` | tidak ditandatangani / salah tanda tangan / di luar jendela / key tidak dikenal — sengaja disamakan semua |
| 403 | `NOT_PERMITTED` | key `read` memanggil route `trade` |
| 400 | `WRONG_ARGS` | body salah, field disebutkan |
| 400 | `POSITION_NOT_FOUND`, `ORDER_NOT_FOUND` | id bukan milik akun ini |
| 503 | `SERVICE_UNAVAILABLE` | server masih memuat key → retry, key baik-baik saja |

Penolakan dari trading engine (400) mempertahankan kode engine: `INSUFFICIENT_FUNDS`, `TOO_LOW_AMOUNT`, `TRADING_SESSION_CLOSED`, dst.

## Endpoint yang dipakai

| Method | Path | Izin |
|---|---|---|
| GET | `/time` | – (tanpa key) |
| GET | `/bars?symbol&barSize&from&to&limit` | – (tanpa key) |
| GET | `/metrics` | read |
| GET | `/positions` | read |
| GET | `/orders/active` | read |
| POST | `/markets` | read |
| POST | `/quotes` | read |
| POST | `/settings` | read |
| POST | `/margins` | read |
| POST | `/orders/market/impact` | read |
| POST | `/orders/market/place` | trade |
| POST | `/orders/limit/impact` \| `/place` | read \| trade |
| POST | `/orders/cancel`, `/orders/cancel-all` | trade |
| POST | `/positions/id/close` | trade |
| POST | `/positions/id/protective-order/set` | trade |
| POST | `/positions/protective-order/set` | trade |
| POST | `/margins/change` | trade |
| POST | `/report/fills`, `/report/orders` | read |

`barSize`: `1M, 5M, 15M, 30M, 1H, 2H, 4H, 1D, 1W, 1MO` — **`1M` = satu menit**, `1MO` = satu bulan. `limit` maksimum 500.

## Aturan teknis yang mengubah desain bot

1. **Tidak ada yang aman diulang.** Order yang dikirim dua kali adalah dua order. → `OrderRouter` tidak retry `place`; setelah timeout ia merekonsiliasi lewat `clientOrderId`.
2. **`impact` sebelum `place`.** Setiap order diprakirakan biayanya tanpa dikirim. `resolutionLevel=ERROR` → order tidak dikirim.
3. **`orders/modify` mengembalikan `orderId` baru** (order diganti, bukan diedit).
4. **`cancel-all` tidak menyentuh TP/SL** kecuali `includingProtective=true`.
5. **Hedge mode**: order berlawanan membuka posisi kedua; route by-instrument tidak menjangkau semua leg → alamatkan posisi via `id`.
6. **Harga trigger/resting** ada di field `triggerPrice`.
7. **Tidak ada stream (v3).** Poll `positions`, `orders/active`, `quotes` — "tidak lebih sering daripada Anda bertindak".
8. **`bars` = harga bid feed.** Ask dimodelkan dari `quotes` / spread, bukan diasumsikan sama.

## Contoh: order market dengan proteksi

```bash
POST /v3/trading/orders/market/place
{
  "symbol": "EUR/USD",
  "side": "BUY",
  "qty": 0.01,
  "takeProfit": 1.1100,
  "stopLoss": 1.0950,
  "clientOrderId": "pxs.EURUSD.1793217900000.ab12cd34ef56"
}
```

`clientOrderId`: 1–36 karakter, hanya huruf/angka/`. _ : -`. Dipakai untuk rekonsiliasi setelah crash.

## Simbol FX

Nama instrumen mengikuti format `BASE/QUOTE` (contoh `EUR/USD`, `USD/JPY`). Karena akun PrimeXBT diselesaikan dalam USD, `${BASE}` (EUR) adalah mata uang base untuk ukuran order; untuk pasangan ber-quote JPY, konversi P&L memakai `1/price` — lihat `risk.engine.pnl_coefficient`.

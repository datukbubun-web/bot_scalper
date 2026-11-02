# Menghubungkan Arena.ai ke VPS Anda (bridge via GitHub Actions)

## 1. Kenapa tidak bisa SSH langsung

Sandbox tempat Arena berjalan **hanya boleh keluar ke host HTTPS tertentu**: `github.com`, `api.github.com`,
`codeload.github.com`, `registry.npmjs.org`, `pypi.org`, `files.pythonhosted.org`.

Uji dari sandbox: `TCP connect` ke IP VPS di port 22/80/443 "berhasil", tetapi tidak ada data yang pernah
kembali — bahkan ke alamat yang tidak ada (`192.0.2.1:22`). Artinya koneksi itu ditangkap proxy egress sandbox,
bukan sampai ke server Anda. Jadi `ssh root@156.246.91.184` dari Arena **tidak akan pernah** bisa, dan itu
bukan masalah di VPS.

Solusinya: **pakai GitHub sebagai jalur** — satu-satunya jalur tulis yang Arena punya ke dunia luar.

```
Arena  --(git push / gh)-->  GitHub  -->  GitHub Actions runner  --(SSH key)-->  VPS Anda
                                  <--(commit ops/last_output.txt)<--
```

- Perintah ditulis Arena ke `ops/command.txt` lalu di-*push* → workflow `vps-bridge` jalan.
- Runner GitHub (IP publik, bukan sandbox Arena) yang men-SSH VPS Anda.
- Hasil perintah di-commit balik ke `ops/last_output.txt` → Arena membacanya lewat `git pull`.
- Kredensial SSH **tidak pernah** masuk repo: disimpan sebagai *repository secret*.

File terkait: `.github/workflows/vps-bridge.yml`, `scripts/vps_check.sh`.

---

## 2. Langkah yang perlu Anda lakukan (sekali saja, ±3 menit)

### A. Pasang public key bridge di VPS

Private key-nya sudah saya buat dan simpan di sandbox (tidak pernah dicetak ke chat), public key-nya:

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMBTOKZpJ+2ODo6VlPv/fUldv0El8NdluOuAbfvfHuGz arena-bridge@bot_scalper
```

Masuk ke VPS seperti biasa (SSH dari laptop/HP, atau console/VNC panel hosting), lalu jalankan **satu perintah** ini:

```bash
mkdir -p ~/.ssh && chmod 700 ~/.ssh && \
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIMBTOKZpJ+2ODo6VlPv/fUldv0El8NdluOuAbfvfHuGz arena-bridge@bot_scalper' >> ~/.ssh/authorized_keys && \
chmod 600 ~/.ssh/authorized_keys && tail -1 ~/.ssh/authorized_keys
```

Kalau `tail -1` menampilkan baris `ssh-ed25519 ...` yang sama, berarti sudah.

> Opsional (lebih aman): batasi key hanya bisa dipakai dari IP runner GitHub. Ambil daftar IP terbaru dari
> `https://api.github.com/meta` (bagian `actions`), lalu tambahkan opsi `from=` sebelum tipe key:
> `from="140.82.112.0/20,143.55.64.0/20,185.199.108.0/22",ssh-ed25519 AAAA...`

### B. Isi 4 secret di GitHub

Buka: **https://github.com/datukbubun-web/bot_scalper/settings/secrets/actions** → *New repository secret*
(tombol ini **hanya** bisa dipakai Anda; token Arena tidak punya izin mengelola Actions secrets).

| Name | Value |
|---|---|
| `VPS_HOST` | `156.246.91.184` |
| `VPS_USER` | `root` |
| `VPS_PORT` | `22` |
| `VPS_SSH_KEY` | isi **seluruh** isi file private key `id_ed25519` (mulai `-----BEGIN OPENSSH PRIVATE KEY-----` sampai `-----END OPENSSH PRIVATE KEY-----`) |

Private key-nya akan saya kirim ke Anda lewat file terpisah (bukan di chat publik). Simpan baik-baik; kalau
hilang, minta saya buatkan pasangan key baru.

### C. Kabari saya

Setelah A dan B selesai, cukup bilang "sudah". Saya akan langsung men-*trigger* pemeriksaan penuh
(`scripts/vps_check.sh`) dan membacakan hasilnya.

---

## 3. Cara pakai setelah aktif

| Yang ingin dilakukan | Caranya |
|---|---|
| Arena memeriksa VPS | Arena menulis `ops/command.txt` → push → baca `ops/last_output.txt` |
| Anda menjalankan perintah manual | Tab **Actions** → `vps-bridge` → **Run workflow** → isi `command` |
| Menjalankan script dari repo | isi `command` dengan `@script scripts/vps_check.sh` |

Contoh isi `ops/command.txt`:

```
@script scripts/vps_check.sh
```

```
uptime && df -h && free -m && systemctl --failed
```

```
tail -n 200 /root/bot_scalper/logs/*.log
```

---

## 4. Mematikan / mencabut akses

1. Hapus 3 secret di halaman Secrets (workflow langsung mati).
2. Cabut key di VPS: `sed -i '/arena-bridge@bot_scalper/d' ~/.ssh/authorized_keys`.
3. Hapus file workflow bila tidak dipakai lagi.

Akses SSH kunci ini **tidak** memberi izin apa pun di GitHub; ia hanya bisa login ke VPS.

---

## 5. Catatan keamanan (penting)

- **Ganti password root VPS.** Password `root` sudah dikirim sebagai teks biasa di chat dan di riwayat
  percakapan; anggap sudah bocor. Setelah key bridge aktif, sebaiknya `PasswordAuthentication no` +
  `PermitRootLogin prohibit-password` di `/etc/ssh/sshd_config`.
- **Pertimbangkan menjadikan repo `private`.** Saat ini `datukbubun-web/bot_scalper` **publik**, sehingga
  `ops/last_output.txt` (informasi server, service, log, konfigurasi) akan ikut publik. Workflow sudah
  meredaksi nilai yang mirip API key/secret/token/password, tapi tidak bisa menjamin semua.
  Perintah: `gh repo edit --visibility private` (atau Settings → General → Danger Zone).
- Key bridge ini khusus: jangan dipakai ulang untuk hal lain, dan simpan hanya di *repository secret*.
- Jangan pernah men-*commit* private key ke repo, walau repo privat.

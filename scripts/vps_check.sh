#!/usr/bin/env bash
# =============================================================================
#  vps_check.sh — Diagnostik READ-ONLY untuk VPS pxscalper (PrimeXBT)
# -----------------------------------------------------------------------------
#  Tidak mengubah apa pun: tidak install paket, tidak restart service,
#  tidak menulis file, tidak pernah mencetak isi API key/secret.
#
#  Cara pakai (dari laptop Anda):
#      ssh root@156.246.91.184 'bash -s' < scripts/vps_check.sh | tee vps_report.txt
#
#  Atau salin file ini ke VPS lalu:
#      bash vps_check.sh                      # auto-deteksi repo
#      bash vps_check.sh /root/bot_scalper    # tentukan path repo manual
# =============================================================================

set -uo pipefail
export LC_ALL=C

REPO_ARG="${1:-}"
FAILS=(); WARNS=(); OKS=()

C_RESET=$'\033[0m'; C_OK=$'\033[32m'; C_WARN=$'\033[33m'; C_FAIL=$'\033[31m'; C_HEAD=$'\033[1;36m'
if [ ! -t 1 ]; then C_RESET=""; C_OK=""; C_WARN=""; C_FAIL=""; C_HEAD=""; fi

head_() { printf '\n%s==================================================================%s\n' "$C_HEAD" "$C_RESET"; printf '%s %s%s\n' "$C_HEAD" "$*" "$C_RESET"; printf '%s==================================================================%s\n' "$C_HEAD" "$C_RESET"; }
ok()   { printf '%s[ OK ]%s %s\n'   "$C_OK"   "$C_RESET" "$*"; OKS+=("$*"); }
warn() { printf '%s[WARN]%s %s\n'   "$C_WARN" "$C_RESET" "$*"; WARNS+=("$*"); }
fail() { printf '%s[FAIL]%s %s\n'   "$C_FAIL" "$C_RESET" "$*"; FAILS+=("$*"); }
info() { printf '       %s\n' "$*"; }
has()  { command -v "$1" >/dev/null 2>&1; }

printf 'pxscalper VPS check — %s — host %s — kernel %s\n' \
  "$(date -u '+%Y-%m-%d %H:%M:%SZ')" "$(hostname 2>/dev/null || echo '?')" "$(uname -r)"

# ---------------------------------------------------------------- 1. SISTEM ---
head_ "1. IDENTITAS & SUMBER DAYA SISTEM"
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  info "OS          : ${PRETTY_NAME:-unknown}"
fi
info "Hostname    : $(hostname 2>/dev/null)  ($(hostname -I 2>/dev/null | awk '{print $1}'))"
info "Kernel/arch : $(uname -r) / $(uname -m)"
info "Uptime      : $(uptime -p 2>/dev/null || uptime)"
info "Virtualisasi: $(systemd-detect-virt 2>/dev/null || echo n/a)"

if [ -r /proc/loadavg ]; then
  read -r l1 l5 l15 _ < /proc/loadavg
  cores=$(nproc 2>/dev/null || echo 1)
  info "Load (1/5/15m): $l1 / $l5 / $l15  di $cores core"
  # bandingkan load1 dengan jumlah core (tanpa bc)
  if [ "${l1%%.*}" -ge "$cores" ] 2>/dev/null; then
    warn "Load 1 menit (${l1}) >= jumlah core (${cores}) — server sibuk/overload"
  else
    ok "Load normal (${l1} < ${cores} core)"
  fi
fi

if has free; then
  info "RAM         : $(free -h | awk '/^Mem:/{print $3" terpakai / "$2" total ("$7" tersedia)"}')"
  info "Swap        : $(free -h | awk '/^Swap:/{print $3" / "$2}')"
fi
info "Disk (/)    : $(df -h / | awk 'NR==2{print $3" / "$2" ("$5" dipakai), sisa "$4}')"
diskpct=$(df -P / | awk 'NR==2{gsub("%","",$5); print $5}')
if [ "${diskpct:-0}" -ge 90 ]; then fail "Disk / terpakai ${diskpct}% — kritis, log/model bisa gagal tulis"
elif [ "${diskpct:-0}" -ge 80 ]; then warn "Disk / terpakai ${diskpct}% — bersihkan log/artefak lama"
else ok "Disk / sehat (${diskpct}% terpakai)"; fi
inodepct=$(df -Pi / | awk 'NR==2{gsub("%","",$5); print $5}')
[ "${inodepct:-0}" -ge 85 ] && warn "Inode / terpakai ${inodepct}%"

# ------------------------------------------------------------------- 2. JAM ---
head_ "2. JAM SISTEM & SINKRONISASI NTP  (kritis: signature API PrimeXBT ditolak bila clock skew)"
info "Waktu lokal : $(date '+%Y-%m-%d %H:%M:%S %Z')"
info "Waktu UTC   : $(date -u '+%Y-%m-%d %H:%M:%SZ')"
info "Timezone    : $(timedatectl show -p Timezone --value 2>/dev/null || cat /etc/timezone 2>/dev/null || echo n/a)"

if has timedatectl; then
  sync_state=$(timedatectl show -p NTPSynchronized --value 2>/dev/null)
  case "$sync_state" in
    yes) ok "NTP tersinkron (timedatectl)" ;;
    no)  warn "NTP belum tersinkron — aktifkan: timedatectl set-ntp true" ;;
    *)   info "Status NTP: ${sync_state:-tidak diketahui}" ;;
  esac
  if has chronyc; then
    info "chrony      : $(chronyc tracking 2>/dev/null | awk -F': ' '/System time/{print $2}')"
  elif has ntpq; then
    info "ntpq offset : $(ntpq -pn 2>/dev/null | awk 'NR>2 && $1 ~ /^\*/ {print $9" ms ("$1")"}')"
  fi
fi

if has curl; then
  srv_date=$(curl -sI --max-time 8 https://api.primexbt.com/ 2>/dev/null | tr -d '\r' | awk 'BEGIN{IGNORECASE=1}/^date:/{sub(/^[Dd]ate: */,""); print; exit}')
  if [ -n "$srv_date" ]; then
    srv_epoch=$(date -u -d "$srv_date" +%s 2>/dev/null)
    loc_epoch=$(date -u +%s)
    if [ -n "$srv_epoch" ]; then
      skew=$(( loc_epoch - srv_epoch )); abs=$(( skew < 0 ? -skew : skew ))
      info "Selisih jam vs api.primexbt.com: ${skew} detik"
      if   [ "$abs" -le 2 ]; then ok "Clock skew aman (<= 2s)"
      elif [ "$abs" -le 10 ]; then warn "Clock skew ${abs}s — sebaiknya < 2s untuk signing API"
      else fail "Clock skew ${abs}s — request bertanda tangan berisiko ditolak"; fi
    fi
  else
    warn "Tidak bisa membaca header Date dari api.primexbt.com"
  fi
fi

# -------------------------------------------------------------- 3. JARINGAN ---
head_ "3. JARINGAN KELUAR (VPS <-> PrimeXBT)"
if has getent; then
  if getent hosts api.primexbt.com >/dev/null 2>&1; then
    ok "DNS resolve api.primexbt.com -> $(getent hosts api.primexbt.com | awk '{print $1}' | tr '\n' ' ')"
  else
    fail "DNS gagal resolve api.primexbt.com — cek /etc/resolv.conf"
  fi
fi
if has curl; then
  res=$(curl -s -o /dev/null --max-time 10 -w '%{http_code} %{time_namelookup} %{time_connect} %{time_appconnect} %{time_total}' https://api.primexbt.com/ 2>/dev/null)
  set -- ${res:-000 0 0 0 0}
  info "HTTP ${1}  dns=${2}s tcp=${3}s tls=${4}s total=${5}s"
  if [ "${1:-000}" = "000" ]; then
    fail "Tidak bisa mencapai https://api.primexbt.com — cek firewall keluar / koneksi provider"
  else
    ok "api.primexbt.com dapat dihubungi (HTTP ${1})"
  fi
  # latensi total: bandingkan dengan skala integer ms
  tot_ms=$(awk -v t="${5:-0}" 'BEGIN{printf "%d", t*1000}')
  if   [ "$tot_ms" -ge 800 ]; then warn "Latensi ke PrimeXBT ${tot_ms} ms — cukup tinggi untuk scalper"
  elif [ "$tot_ms" -ge 300 ]; then info "Latensi ${tot_ms} ms (dapat diterima, tapi bukan ideal)"
  else ok "Latensi ke PrimeXBT ${tot_ms} ms"; fi

  # cek proxy env yang tak sengaja ter-set
  if [ -n "${http_proxy:-}${https_proxy:-}${HTTPS_PROXY:-}${HTTP_PROXY:-}" ]; then
    warn "Variabel proxy ter-set (${https_proxy:-${HTTPS_PROXY:-${http_proxy:-${HTTP_PROXY}}}}} ) — pastikan disengaja"
  fi
fi
info "IPv6 default route aktif: $(ip -6 route show default 2>/dev/null | head -1 | grep -q . && echo ya || echo tidak)"
info "Interface utama          : $(ip -4 route show default 2>/dev/null | awk '{print $5; exit}')"

# ------------------------------------------------------------------ 4. REPO ---
head_ "4. LOKASI REPO, PYTHON & ARTEFAK MODEL"
REPO=""
for cand in "$REPO_ARG" /root/bot_scalper /opt/bot_scalper /home/*/bot_scalper /srv/bot_scalper "$PWD"; do
  [ -n "$cand" ] && [ -d "$cand" ] && [ -f "$cand/pyproject.toml" ] && { REPO=$(cd "$cand" && pwd); break; }
done
if [ -z "$REPO" ]; then
  REPO=$(find / -maxdepth 4 -type f -name pyproject.toml -path '*bot_scalper*' 2>/dev/null | head -1 | xargs -r dirname)
fi

if [ -n "$REPO" ] && [ -d "$REPO" ]; then
  ok "Repo ditemukan: $REPO"
  if [ -d "$REPO/.git" ] && has git; then
    info "Git branch  : $(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)"
    info "Commit      : $(git -C "$REPO" log -1 --format='%h %s (%cr)' 2>/dev/null | cut -c1-90)"
    if [ -n "$(git -C "$REPO" status --porcelain 2>/dev/null | head -1)" ]; then
      warn "Working tree repo kotor (ada perubahan belum di-commit) — $(git -C "$REPO" status --porcelain | wc -l) file"
    else ok "Working tree bersih"; fi
    if has git && [ -n "$(git -C "$REPO" log -1 --format=%H 2>/dev/null)" ]; then
      behind=$(git -C "$REPO" rev-list --count HEAD..@{u} 2>/dev/null)
      [ "${behind:-0}" -gt 0 ] 2>/dev/null && warn "Repo tertinggal ${behind} commit dari remote"
    fi
  fi

  # .env — nama key saja, nilai di-mask
  if [ -f "$REPO/.env" ]; then
    perm=$(stat -c '%a %U:%G' "$REPO/.env" 2>/dev/null)
    info ".env        : ada (perm ${perm})"
    others=$(stat -c '%a' "$REPO/.env" 2>/dev/null | cut -c3)
    [ "${others:-0}" != "0" ] && warn "Permission .env ${perm} — seharusnya 600 (chmod 600 $REPO/.env)"
    while IFS= read -r line; do
      case "$line" in ''|'#'*) continue;; esac
      key=${line%%=*}
      case "$key" in
        *KEY*|*SECRET*|*TOKEN*|*PASS*) info "  ${key}=<disembunyikan>" ;;
        *) info "  ${line}" ;;
      esac
    done < "$REPO/.env"
    lt=$(grep -E '^PX_LIVE_TRADING=' "$REPO/.env" 2>/dev/null | cut -d= -f2 | tr -d ' "'"'"'')
    case "$lt" in
      1|true|TRUE|yes) fail "PX_LIVE_TRADING=$lt di .env — bot berada di mode UANG NYATA" ;;
      0|false|FALSE|no) ok "PX_LIVE_TRADING=${lt:-0} — mode paper/demo" ;;
      *) info "PX_LIVE_TRADING tidak eksplisit di .env (default paper)" ;;
    esac
  else
    warn ".env tidak ditemukan di $REPO — bot tidak akan punya kredensial"
  fi

  # python env
  py=""
  if [ -x "$REPO/.venv/bin/python" ]; then py="$REPO/.venv/bin/python"; info "Venv        : $REPO/.venv"
  elif [ -x "$REPO/venv/bin/python" ]; then py="$REPO/venv/bin/python"; info "Venv        : $REPO/venv"
  elif has python3; then py=$(command -v python3); warn "Tidak ada .venv/venv — memakai python3 sistem ($py)"; fi
  if [ -n "$py" ]; then
    info "Python      : $("$py" -V 2>&1)"
    if "$py" -c 'import pxscalper, sys; print("pxscalper", getattr(pxscalper,"__version__","?"), "di", pxscalper.__file__)' 2>/dev/null; then
      ok "Paket pxscalper terpasang"
    else
      warn "Paket pxscalper TIDAK import-able dari $py — jalankan: pip install -e ."
    fi
    for mod in numpy pandas torch onnxruntime; do
      v=$("$py" -c "import $mod, sys; print(getattr($mod,'__version__','?'))" 2>/dev/null)
      [ -n "$v" ] && info "  $mod: $v" || warn "  modul $mod tidak ada di environment ini"
    done
  fi

  # artefak model
  if [ -d "$REPO/artifacts" ]; then
    n_onnx=$(find "$REPO/artifacts" -type f \( -name '*.onnx' -o -name '*.pt' -o -name '*.ts' \) 2>/dev/null | wc -l)
    size=$(du -sh "$REPO/artifacts" 2>/dev/null | awk '{print $1}')
    [ "$n_onnx" -gt 0 ] && ok "Artefak model: ${n_onnx} file biner (${size})" \
                        || fail "Tidak ada file model (.onnx/.pt/.ts) di artifacts/ — bot tidak bisa inferensi"
    info "Layout model: $(find "$REPO/artifacts" -maxdepth 2 -mindepth 1 -type d 2>/dev/null | sed "s|$REPO/||" | head -8 | tr '\n' ' ')"
  else
    fail "Direktori artifacts/ tidak ada di $REPO"
  fi

  # state & logs
  for d in state logs; do
    if [ -d "$REPO/$d" ]; then
      info "$d/        : $(ls -1 "$REPO/$d" 2>/dev/null | wc -l) item, ukuran $(du -sh "$REPO/$d" 2>/dev/null | awk '{print $1}')"
    fi
  done
else
  fail "Repo pxscalper tidak ditemukan — jalankan ulang: bash vps_check.sh /path/ke/bot_scalper"
fi

# --------------------------------------------------------------- 5. PROSES ---
head_ "5. PROSES & SERVICE BOT"
found_proc=0
if has pgrep; then
  procs=$(pgrep -af 'pxscalper|pxs (trade|live)|cli trade' 2>/dev/null | grep -v 'vps_check.sh' | grep -v pgrep)
  if [ -n "$procs" ]; then found_proc=1; ok "Proses bot berjalan:"; printf '%s\n' "$procs" | sed 's/^/         /' ; fi
fi
if has systemctl; then
  units=$(systemctl list-units --type=service --state=running --no-legend --no-pager 2>/dev/null | awk '{print $1}' | grep -iE 'pxs|scale?lper|trade|bot' || true)
  if [ -n "$units" ]; then
    found_proc=1; ok "Systemd service aktif: $(echo "$units" | tr '\n' ' ')"
    for u in $units; do
      info "  $u: $(systemctl show "$u" -p ActiveEnterTimestamp --value 2>/dev/null) | restart $(systemctl show "$u" -p NRestarts --value 2>/dev/null)x"
      if systemctl is-enabled --quiet "$u" 2>/dev/null; then info "  $u: enabled (auto-start saat boot)"
      else warn "  $u: tidak enabled — mati setelah reboot, jalankan: systemctl enable $u"; fi
    done
  fi
  failed=$(systemctl list-units --type=service --state=failed --no-legend --no-pager 2>/dev/null | awk '{print $1}' | tr '\n' ' ')
  [ -n "$failed" ] && warn "Service dalam status failed: $failed"
fi
if has tmux;  then s=$(tmux ls 2>/dev/null); [ -n "$s" ] && { found_proc=1; ok "Sesi tmux: $(echo "$s" | tr '\n' '; ')"; }; fi
if has screen; then s=$(screen -ls 2>/dev/null | grep -E '^\s+[0-9]+\.'); [ -n "$s" ] && { found_proc=1; ok "Sesi screen aktif"; }; fi
if has docker; then
  d=$(docker ps --format '{{.Names}} ({{.Status}})' 2>/dev/null)
  [ -n "$d" ] && { found_proc=1; ok "Container Docker: $(echo "$d" | tr '\n' '; ')"; }
fi
[ "$found_proc" -eq 0 ] && fail "TIDAK ada proses/service bot yang terdeteksi berjalan"

# ----------------------------------------------------------------- 6. PORT ---
head_ "6. PORT TERBUKA & FIREWALL"
if has ss; then
  listening=$(ss -ltnp 2>/dev/null)
  printf '%s\n' "$listening" | head -15 | sed 's/^/       /'
  met=$(printf '%s\n' "$listening" | grep -cE ':(8000|9090|9100|3000|8080)\b' || true)
  [ "${met:-0}" -gt 0 ] && info "Port metrics/app terdeteksi — pastikan hanya untuk localhost/VPN"
  pub=$(printf '%s\n' "$listening" | grep -E '0\.0\.0\.0:|\*:|\[::\]:' | grep -vE ':(22|80|443)\b' || true)
  [ -n "$pub" ] && warn "Ada port mendengarkan di semua interface selain 22/80/443" || ok "Tidak ada port sensitif terbuka ke publik"
fi
if has ufw && ufw status 2>/dev/null | grep -qi 'status: active'; then
  ok "UFW aktif"; info "$(ufw status numbered 2>/dev/null | head -12 | tr '\n' '|' | sed 's/|/\n       /g')"
elif has iptables; then
  n=$(iptables -S 2>/dev/null | grep -c '^-A' || true); info "iptables rules: ${n:-0}"
fi
if has fail2ban-client; then
  info "fail2ban    : $(fail2ban-client status 2>/dev/null | awk -F': ' '/Jail list/{print $2}')"
  banned=$(fail2ban-client status sshd 2>/dev/null | awk -F'\t' '/Currently banned/{print $2}')
  info "  banned sshd: ${banned:-n/a}"
fi

# ----------------------------------------------------------------- 7. LOG ---
head_ "7. LOG TERAKHIR & ERROR"
if [ -n "$REPO" ] && [ -d "$REPO/logs" ]; then
  latest=$(find "$REPO/logs" -type f -name '*.log*' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | awk '{print $2}')
  if [ -n "$latest" ]; then
    info "Log terbaru: $latest ($(du -h "$latest" | awk '{print $1}'), mtime $(date -r "$latest" '+%Y-%m-%d %H:%M' 2>/dev/null))"
    errs=$(grep -iE 'traceback|error|exception|rejected|insufficient|unauthorized|signature' "$latest" 2>/dev/null | tail -15)
    if [ -n "$errs" ]; then warn "Baris error terakhir pada log:"; printf '%s\n' "$errs" | cut -c1-200 | sed 's/^/         /'
    else ok "Tidak ada baris error/traceback di log terbaru"; fi
    info "--- 10 baris terakhir ---"
    tail -10 "$latest" 2>/dev/null | cut -c1-200 | sed 's/^/         /'
  else
    warn "Folder logs/ ada tapi tidak berisi file .log"
  fi
else
  info "Tidak ada folder logs/ di repo"
fi
if has journalctl; then
  j=$(journalctl -u '*pxs*' -u '*scalper*' --since '24 hours ago' -p err --no-pager -n 15 2>/dev/null | tail -15)
  if [ -n "$j" ] && ! printf '%s' "$j" | grep -q 'No entries'; then
    warn "journalctl error 24 jam terakhir:"; printf '%s\n' "$j" | cut -c1-200 | sed 's/^/         /'
  fi
fi

# ------------------------------------------------------------ 8. KEAMANAN ---
head_ "8. KEAMANAN DASAR VPS"
if [ -r /etc/ssh/sshd_config ]; then
  eff() { ss -ltn 2>/dev/null >/dev/null; grep -hiE "^[[:space:]]*$1[[:space:]]" /etc/ssh/sshd_config /etc/ssh/sshd_config.d/*.conf 2>/dev/null | awk '{print $2}' | tail -1; }
  prl=$(eff PermitRootLogin); pw=$(eff PasswordAuthentication); sport=$(eff Port)
  info "SSH port            : ${sport:-22}"
  info "PermitRootLogin     : ${prl:-tidak diset (default prohibit-password)}"
  info "PasswordAuthentication: ${pw:-tidak diset (default yes)}"
  [ "${pw:-yes}" = "yes" ] && warn "Login SSH dengan password masih aktif — pakai SSH key, lalu PasswordAuthentication no"
  [ "${prl:-}" = "yes" ] && warn "Root login dengan password diizinkan — sangat berisiko"
  [ -f "$HOME/.ssh/authorized_keys" ] && ok "SSH key terpasang untuk user ini ($(wc -l < "$HOME/.ssh/authorized_keys") key)"
fi
if has lastb; then
  n=$(lastb 2>/dev/null | grep -c . || true)
  [ "${n:-0}" -gt 0 ] && warn "${n} percobaan login SSH gagal tercatat (lastb) — pertimbangkan fail2ban + ganti port/key-only"
fi
if has last; then
  info "Login sukses terakhir:"; last -n 5 -w 2>/dev/null | head -5 | sed 's/^/         /'
fi
if has apt-get && [ -f /var/run/reboot-required ]; then
  warn "Reboot tertunda: $(cat /var/run/reboot-required 2>/dev/null) — ada update kernel/libc"
fi
if has crontab; then
  c=$(crontab -l 2>/dev/null | grep -vE '^\s*#|^\s*$' | head -6)
  [ -n "$c" ] && { info "Crontab root:"; printf '%s\n' "$c" | sed 's/^/         /'; }
fi

# ------------------------------------------------------------- 9. RINGKASAN ---
head_ "9. RINGKASAN TEMUAN"
printf '  OK   : %d\n' "${#OKS[@]}"
printf '  WARN : %d\n' "${#WARNS[@]}"
printf '  FAIL : %d\n' "${#FAILS[@]}"
if [ "${#FAILS[@]}" -gt 0 ]; then
  printf '\n%sPERLU DIPERBAIKI SEGERA:%s\n' "$C_FAIL" "$C_RESET"
  for f in "${FAILS[@]}"; do printf '  - %s\n' "$f"; done
fi
if [ "${#WARNS[@]}" -gt 0 ]; then
  printf '\n%sPERLU DIPERHATIKAN:%s\n' "$C_WARN" "$C_RESET"
  for w in "${WARNS[@]}"; do printf '  - %s\n' "$w"; done
fi
printf '\nSelesai %s. Tempel seluruh output ini bila minta bantuan analisis.\n' "$(date -u '+%Y-%m-%d %H:%M:%SZ')"
exit 0

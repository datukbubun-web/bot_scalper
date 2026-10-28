#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  Push pxscalper ke https://github.com/datukbubun-web/bot_scalper
#
#  Pakai (dari folder repo, di mesin Anda):
#     GITHUB_TOKEN=ghp_xxx bash scripts/push_to_github.sh
#
#  Token butuh scope:  repo  (classic)  atau  Contents: Read and write (fine-grained)
#  Tanpa token, script akan memakai kredensial git yang sudah ada (SSH/helper).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

OWNER="${GITHUB_OWNER:-datukbubun-web}"
REPO="${GITHUB_REPO:-bot_scalper}"
BRANCH="${GITHUB_BRANCH:-main}"
REMOTE_NAME="${REMOTE_NAME:-origin}"

if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  REMOTE_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${OWNER}/${REPO}.git"
  echo "→ memakai GITHUB_TOKEN"
else
  REMOTE_URL="https://github.com/${OWNER}/${REPO}.git"
  echo "→ tanpa token, mengandalkan kredensial git yang ada (SSH / credential helper)"
fi

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
echo "→ repo lokal: $ROOT"

# ---------------------------------------------------------------- sanity check
if [[ ! -f pyproject.toml || ! -f src/pxscalper/cli.py ]]; then
  echo "✗ Ini bukan root repo pxscalper. Jalankan script dari dalam repo." >&2
  exit 1
fi

for f in .env .env.local secrets; do
  if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
    echo "✗ '$f' terlacak git. Hentikan: hapus dari index dulu (git rm --cached $f)." >&2
    exit 1
  fi
done
echo "✓ tidak ada file kredensial yang terlacak"

# ------------------------------------------------------------------ git basics
if [[ ! -d .git ]]; then
  git init -q
  git symbolic-ref HEAD "refs/heads/${BRANCH}"
fi

git config user.name  >/dev/null 2>&1 || git config user.name  "${GIT_AUTHOR_NAME:-datukbubun-web}"
git config user.email >/dev/null 2>&1 || git config user.email "${GIT_AUTHOR_EMAIL:-datukbubun-web@users.noreply.github.com}"

git add -A
if git diff --cached --quiet; then
  echo "→ tidak ada perubahan untuk di-commit"
else
  git commit -q -m "${COMMIT_MSG:-feat: pxscalper v0.1.0 — FX scalping stack for PrimeXBT (TCN/LSTM/Transformer + PPO/SAC, ONNX low-latency inference)}"
  echo "✓ commit dibuat"
fi

# --------------------------------------------------------------------- remote
if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
  git remote set-url "$REMOTE_NAME" "$REMOTE_URL"
else
  git remote add "$REMOTE_NAME" "$REMOTE_URL"
fi

echo "→ push ke ${OWNER}/${REPO}:${BRANCH}"
git branch -M "$BRANCH"
git push -u "$REMOTE_NAME" "$BRANCH"

echo
echo "✓ selesai: https://github.com/${OWNER}/${REPO}"
echo "  Langkah berikutnya di repo:"
echo "    Settings → Secrets and variables → Actions → tambahkan (opsional) untuk live runner"
echo "    Ingat: JANGAN commit .env. Isi kredensial lewat environment/secret manager."

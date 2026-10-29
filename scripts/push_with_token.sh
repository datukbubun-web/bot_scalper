#!/usr/bin/env bash
# Push the local commits, then immediately scrub the token from the repo config.
#
# The token is read from a FILE, never from an argument or an environment
# variable that would land in shell history or in `ps`. It is used only to build
# a one-shot push URL and is never written to .git/config.
#
#   printf '%s' 'github_pat_xxx' > /tmp/gh_token && chmod 600 /tmp/gh_token
#   bash scripts/push_with_token.sh
#
# Afterwards: revoke the token on GitHub. Do not rely on deleting the file.
set -euo pipefail

REPO="datukbubun-web/bot_scalper"
TOKEN_FILE="${1:-/tmp/gh_token}"
BRANCH="main"
CLEAN_URL="https://github.com/${REPO}.git"

if [[ ! -f "$TOKEN_FILE" ]]; then
  echo "error: token file $TOKEN_FILE not found" >&2
  exit 1
fi
TOKEN="$(tr -d ' \t\r\n' < "$TOKEN_FILE")"
if [[ -z "$TOKEN" ]]; then
  echo "error: $TOKEN_FILE is empty" >&2
  exit 1
fi

cd "$(dirname "$0")/.."

# Never carry a token in the config: build the URL for this one call only.
AUTH_URL="https://x-access-token:${TOKEN}@github.com/${REPO}.git"

echo "== commits to push =="
git log --oneline "origin/${BRANCH}..${BRANCH}" 2>/dev/null || git log --oneline -6

echo
echo "== push =="
# -c http.extraheader is not used: the URL form works with fine-grained tokens.
if git push "$AUTH_URL" "refs/heads/${BRANCH}:refs/heads/${BRANCH}" 2>&1 | sed -E 's#https://[^@]*@#https://***@#g'; then
  echo "push OK"
else
  status=${PIPESTATUS[0]}
  echo "push FAILED (exit $status)" >&2
  exit "$status"
fi

echo
echo "== verify remote head =="
git fetch "$AUTH_URL" "${BRANCH}" 2>/dev/null || true
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git ls-remote "$AUTH_URL" "refs/heads/${BRANCH}" | awk '{print $1}')"
echo "local  HEAD: $LOCAL_SHA"
echo "remote head: $REMOTE_SHA"
if [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
  echo "VERIFIED: remote matches local"
  VERIFIED=1
else
  echo "MISMATCH: push did not land as expected" >&2
  VERIFIED=0
fi

# Belt and braces: make sure the token is not sitting in the config.
git remote set-url origin "$CLEAN_URL"
if git config --get-regexp '^remote\.origin\.' | grep -q "@github.com"; then
  echo "WARNING: token still present in remote config -- cleaning" >&2
  git remote set-url origin "$CLEAN_URL"
fi
echo "remote url: $(git remote get-url origin)"

rm -f "$TOKEN_FILE"
echo "token file removed: $TOKEN_FILE"
echo
echo "NEXT STEP (manual, do not skip): revoke the token at"
echo "  https://github.com/settings/tokens"
exit $((1 - VERIFIED))

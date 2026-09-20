#!/usr/bin/env bash
# Runner preflight self-check: run once after the runner is installed to confirm the
# environment a PR review needs is present. All green = @aiter-bot review runs end to end
# on trigger. Fix any red per its hint. Read-only, changes nothing.
#   bash .claude/skills/review-pr/preflight.sh
# Run it as the same user the runner runs as (claude-glm config is per-user).
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # the review-pr skill dir
ok=0; bad=0
chk() { if eval "$2" >/dev/null 2>&1; then echo "  ✅ $1"; ok=$((ok+1)); else echo "  ❌ $1 — $3"; bad=$((bad+1)); fi; }

echo "=== aiter-review-bot runner preflight (user=$(whoami)) ==="

echo "[skill scripts present]"
for s in fetch.sh triage.py render.sh run_one.sh _lib.py _gates.py _collect.py _publish.py; do
  chk "$s present" "[ -f '$S/$s' ]" "missing $S/$s"
done

echo "[runtime]"
chk "python3 available" "command -v python3" "install python3"
chk "git available" "command -v git" "install git"
chk "curl available" "command -v curl" "install curl"

echo "[headless GLM]"
chk "claude-glm on PATH" "command -v claude-glm" "install/symlink claude-glm"
chk "claude-glm has endpoint config" "[ -r \"\${XDG_CONFIG_HOME:-\$HOME/.config}/claude-glm/endpoints.conf\" ]" "this user is missing ~/.config/claude-glm/ (endpoints + ssh key)"
if command -v claude-glm >/dev/null 2>&1; then
  where="$(timeout 40 claude-glm --where 2>/dev/null | head -1)"
  chk "GLM endpoint resolves" "[ -n '$where' ]" "no endpoint in endpoints.conf can generate a token"
  [ -n "$where" ] && echo "     -> ${where%%$'\t'*}"
fi

echo "[publish identity]"
tok=""
[ -n "${AITER_BOT_TOKEN:-}" ] && tok="$AITER_BOT_TOKEN"
[ -z "$tok" ] && [ -n "${AITER_BOT_TOKEN_FILE:-}" ] && tok="$(grep -oE '(ghp_|github_pat_)[A-Za-z0-9_]+' "${AITER_BOT_TOKEN_FILE}" 2>/dev/null | tail -1)"
if [ -n "$tok" ]; then
  who="$(curl -s -m 12 -H "Authorization: token $tok" https://api.github.com/user 2>/dev/null | python3 -c 'import sys,json;print(json.load(sys.stdin).get("login",""))' 2>/dev/null)"
  chk "bot token valid (identity=${who:-?})" "[ -n '$who' ]" "AITER_BOT_TOKEN(_FILE) invalid"
  [ "$who" = "aiter-bot" ] || echo "     ⚠ identity is '$who', not aiter-bot — comments would post as $who"
else
  echo "  ⚠ AITER_BOT_TOKEN / AITER_BOT_TOKEN_FILE not set — the workflow injects it from a secret; export one to verify locally"
fi

echo "=== $ok green / $bad red ==="
[ "$bad" -eq 0 ] && echo "runner ready: @aiter-bot review can run end to end." || echo "fix the red items before triggering."
exit "$bad"

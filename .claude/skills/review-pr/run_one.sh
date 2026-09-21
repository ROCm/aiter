#!/usr/bin/env bash
# Headless review of one PR: fetch -> GLM worker -> GLM refuter -> gates -> collect.
# Lives in the review-pr skill dir and drives the rest of the skill (calls its fetch.sh and
# triage.py). Pure GLM on the box (claude-glm), no claude-mix. Publishing is a separate step
# (_publish.py); this script does not publish.
#
# This is the backend of the @aiter-bot review workflow, and can also be run by hand.
#
# ⚠️ Must run in a plain terminal / the runner, NOT inside an auto-mode Claude session — the
#    auto-mode classifier blocks a headless sub-agent launched with
#    --dangerously-skip-permissions as "Create Unsafe Agents". A runner is a plain shell.
#
# ⚠️ Cross-family strength: worker and refuter are both GLM (same family) here. The refuter's
#    value comes from being a different family (Opus) catching the GLM worker's mistakes. Pure
#    GLM is weaker adversarially; point REFUTER_CMD at Opus to restore the cross-family check.
set -euo pipefail

PR="${1:?usage: run_one.sh <pr> [owner/repo]}"
REPO="${2:-ROCm/aiter}"
SKILL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .claude/skills/review-pr
PROJ="$(git -C "$SKILL" rev-parse --show-toplevel)"     # repo root

# GLM-5.3 is not in Claude's model catalog; disable the unknown-model window enforcement.
# The container runs as root; declare the docker sandbox so headless tools work.
export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1 CLAUDE_GLM_QUIET=1 IS_SANDBOX=1 PYTHONUTF8=1
# Kernel validation off unless a GPU of the matching arch is present; report the gap honestly.
export REVIEW_AUTO_VALIDATE="${REVIEW_AUTO_VALIDATE:-0}"

# gh's own login may be broken (GraphQL 401 observed) while ~/.git-credentials is fine.
if ! gh api repos/"$REPO" --jq .full_name >/dev/null 2>&1; then
  TOK=$(sed -n 's#https://\([^:]*\):\([^@]*\)@github.com#\2#p' ~/.git-credentials | head -1)
  [ -n "$TOK" ] && export GH_TOKEN="$TOK" GITHUB_TOKEN="$TOK"
fi

# The box's Claude entrypoint. Default is claude-glm (resolves a remote GLM over a tunnel);
# on a box that hosts GLM itself, set AITER_REVIEW_AGENT to a local `claude` pointed at the
# on-box endpoint (ANTHROPIC_BASE_URL=http://localhost:<port>), so no tunnel/wrapper is needed.
AGENT="${AITER_REVIEW_AGENT:-claude-glm}"
WORKER_CMD=("$AGENT" -p --dangerously-skip-permissions)
# Cross-family (restores the refuter's value): set AITER_REVIEW_REFUTER_AGENT to an Opus entrypoint.
REFUTER_CMD=("${AITER_REVIEW_REFUTER_AGENT:-$AGENT}" -p --dangerously-skip-permissions)
say() { echo "[run_one #$PR] $*"; }
# Classify a failed review and expose it so the notify step @-mentions the RIGHT owner:
#   atom -> GLM/backend (honglie) | flow -> this review pipeline (bot owner) | env -> runner box.
# Writes <class>TAB<message> to the status file the workflow reads, plus an ::error annotation.
fail() {  # <class> <exit-code> <message...>
  local cls="$1" code="$2"; shift 2
  say "$*"
  echo "::error title=aiter-bot::[$cls] $*"
  printf '%s\t%s\n' "$cls" "$*" > "${GITHUB_WORKSPACE:-$PROJ}/.aiter-review-status"
  exit "$code"
}

# Fail fast if the prompts have drifted from SKILL.md (they quote it verbatim).
python3 "$SKILL/check_prompts.py" >/dev/null || fail flow 4 "prompts drifted from SKILL.md -- regenerate the prompt copies (check_prompts.py)"

# GLM health-gate: a review is worthless if the backend is down, and a dead GLM otherwise
# hangs ~40s per agent call and dies silently mid-review. When a direct endpoint is configured
# (ANTHROPIC_BASE_URL, i.e. an on-box GLM), deep-probe it with a REAL 1-token inference — not
# the shallow /health, which stays green while inference is wedged — and fail fast with a clear
# signal. claude-glm mode resolves its own endpoint, so there is nothing to probe here.
if [ -n "${ANTHROPIC_BASE_URL:-}" ]; then
  say "GLM health probe..."
  _code=$(curl -s -m "${AITER_GLM_PROBE_TIMEOUT:-30}" --noproxy '*' \
    "$ANTHROPIC_BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"${ANTHROPIC_MODEL:-/models/GLM-5.3}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
    -o /dev/null -w '%{http_code}' 2>/dev/null || true)
  if [ "$_code" != "200" ]; then
    fail glm 5 "GLM backend unavailable at $ANTHROPIC_BASE_URL (deep inference probe returned '$_code')"
  fi
  say "GLM ok"
fi

# 1) fetch (the skill's own Step-1 fetcher) -> WORK dir
say "fetch..."
FL="$(mktemp)"
(cd "$PROJ" && bash "$SKILL/fetch.sh" "$PR" "$REPO") 2>&1 | tee "$FL"
W="$(grep -oE 'WORK=[^[:space:]]+/review-pr-[A-Za-z0-9]+' "$FL" | tail -1 | cut -d= -f2)"
rm -f "$FL"
[ -n "$W" ] && [ -d "$W" ] || fail env 1 "fetch produced no WORK dir -- check gh auth/token, network, git and disk on the runner"

# applies.txt fix: the skill's fetch.sh checks `git apply` against PROJECT_ROOT's worktree but
# reports it as the merge target; re-check on the merge-target worktree checked out at BASE_SHA.
if [ -d "$W/merge-target" ]; then
  BASE=$(grep -oE '[0-9a-f]{40}' "$W/merge_target.txt" 2>/dev/null | head -1)
  if git -C "$W/merge-target" -c core.fileMode=false apply --check "$W/pr.diff" 2>"$W/.err2"; then
    echo "APPLIES: the diff still applies to merge target $BASE (rechecked on the merge-target worktree)" > "$W/applies.txt"
  else
    { echo "STALE: no longer applies to merge target $BASE -- the PR needs a rebase,"
      echo "  and any CI result on it describes a tree that has moved"
      sed 's/^/  /' "$W/.err2"; } > "$W/applies.txt"
  fi
fi
say "WORK=$W"

# The GLM backend can be slow or time out on a shared box; a single request timeout must not
# kill the whole review. Retry the agent up to AITER_REVIEW_RETRIES (default 3) with backoff,
# requiring its output file to exist and be non-empty before counting the attempt as success.
run_agent() {  # <label> <prompt-file> <out-file> <cmd...>
  local label="$1" pf="$2" out="$3"; shift 3
  local n=0 max="${AITER_REVIEW_RETRIES:-3}"
  while :; do
    n=$((n + 1)); rm -f "$out"
    if (cd "$PROJ" && "$@" "$(cat "$pf")") && [ -s "$out" ]; then return 0; fi
    if [ "$n" -ge "$max" ]; then say "$label failed after $max attempts (GLM error/timeout?)"; return 1; fi
    say "$label attempt $n failed (GLM slow/timeout?); retrying in $((n * 10))s"; sleep $((n * 10))
  done
}

# 2) worker (headless GLM), with retry on GLM timeout
say "worker (GLM)..."
bash "$SKILL/render.sh" worker "$W" > "$W/_pw.txt"
run_agent "worker" "$W/_pw.txt" "$W/card.md" "${WORKER_CMD[@]}" || fail glm 2 "the GLM worker failed after retries -- the backend is timing out or down"

# 3) refuter (headless GLM) -- Step 7.7; or the NONE line for a 0-finding card
say "refuter..."
if grep -qiE '(NO FINDINGS|✅)' "$W/card.md" && ! grep -qE '^(🔴|⚠️|📝)' "$W/card.md"; then
  printf 'NONE AVAILABLE -- 0 findings on the card (NO FINDINGS); nothing for an independent reader to refute\n' > "$W/independent.txt"
else
  bash "$SKILL/render.sh" refuter "$W" "$W/card.md" > "$W/_prf.txt"
  run_agent "refuter" "$W/_prf.txt" "$W/independent.txt" "${REFUTER_CMD[@]}" || fail glm 3 "the GLM refuter failed after retries -- the backend is timing out or down"
fi

# 4) gates + collect (call the python directly; no thin shell wrappers)
say "gates..."
python3 "$SKILL/_gates.py" "$W" "$PROJ"
say "collect..."
python3 "$SKILL/_collect.py" "$W"

# Clean up the scratch WORK dir + its worktrees on success (set -e keeps it on failure for debug).
for wt in "$W/merge-target" "$W/head"; do [ -d "$wt" ] && git -C "$PROJ" worktree remove --force "$wt" 2>/dev/null || true; done
rm -rf "$W"
git -C "$PROJ" worktree prune 2>/dev/null || true

say "done. report in $SKILL/reports/PR-$PR/ (uncommitted, unpublished)"

#!/usr/bin/env bash
# Headless review of one PR: fetch -> GLM worker -> GLM refuter -> gates -> collect.
# Pure GLM on the box (claude-glm), no claude-mix. Publishing is a separate step (see the
# end); this script does not publish.
#
# This is the "real run" backend of the @aiter-bot flow: the CI workflow / daemon calls it
# after a trigger. It can also be run by hand as `bash run_one.sh <pr>`.
#
# ⚠️ Must run in a plain terminal / the runner, NOT inside an auto-mode Claude session —
#    the auto-mode classifier blocks a headless sub-agent launched with
#    --dangerously-skip-permissions as "Create Unsafe Agents" (observed 2026-09-18). A
#    runner is a plain shell with no such restriction.
#
# ⚠️ Cross-family strength is lost: worker and refuter are both GLM (same family) here. The
#    refuter's value comes from being a *different family* (Opus) catching the GLM worker's
#    mistakes (it killed ~a dozen false positives in testing). Pure GLM = same-family self-
#    review, weaker adversarially. To keep the cross-family check, point REFUTER_CMD at Opus
#    (needs mix/gateway). The current choice is "pure GLM, no mix", so the default is same-
#    family; switch back by editing the one REFUTER_CMD line.
set -euo pipefail

PR="${1:?usage: run_one.sh <pr> [owner/repo]}"
REPO="${2:-ROCm/aiter}"
RL="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$RL/.." && pwd)"

# GLM-5.3 is not in Claude's model catalog; disable the unknown-model context-window
# enforcement (otherwise it warns every run and can stall).
export CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1
export CLAUDE_GLM_QUIET=1
# The container runs as root; Claude Code refuses --dangerously-skip-permissions as root
# unless a sandbox is declared. This is a controlled docker sandbox (isolated from the host),
# so declare it and let the headless sub-agent use tools autonomously.
export IS_SANDBOX=1

WORKER_CMD=(claude-glm -p --dangerously-skip-permissions)
# Cross-family version (keeps the refuter's value): point at an Opus entrypoint, e.g.
#   REFUTER_CMD=(claude -p --dangerously-skip-permissions)      # default subscription = Opus
# Pure-GLM version (the current choice):
REFUTER_CMD=(claude-glm -p --dangerously-skip-permissions)

say() { echo "[run_one #$PR] $*"; }

# 1) fetch -> WORK dir
say "fetch..."
FETCH_LOG="$(mktemp)"
bash "$RL/fetch.sh" "$PR" "$REPO" 2>&1 | tee "$FETCH_LOG"
W="$(grep -oE 'WORK=/tmp/review-pr-[A-Za-z0-9]+' "$FETCH_LOG" | tail -1 | cut -d= -f2)"
rm -f "$FETCH_LOG"
[ -n "$W" ] && [ -d "$W" ] || { say "fetch produced no WORK dir, aborting"; exit 1; }
say "WORK=$W"

# 2) worker (headless GLM) -- run from the project root; the prompt uses absolute paths
say "worker (GLM) ..."
bash "$RL/render.sh" worker "$W" > "$W/_pw.txt"
( cd "$PROJ" && "${WORKER_CMD[@]}" "$(cat "$W/_pw.txt")" ) || { say "worker failed"; exit 2; }
[ -s "$W/card.md" ] || { say "worker produced no card.md, aborting"; exit 2; }

# 3) refuter (headless, GLM by default) -- Step 7.7 independent refutation, writes independent.txt
say "refuter ..."
if grep -qiE '(NO FINDINGS|✅)' "$W/card.md" && ! grep -qE '^(🔴|⚠️|📝)' "$W/card.md"; then
  # 0-finding card: nothing to refute, write the NONE line directly (matches the interactive flow)
  printf 'NONE AVAILABLE -- 0 findings on the card (NO FINDINGS); nothing for an independent reader to refute\n' > "$W/independent.txt"
  say "0-finding, wrote NONE independent"
else
  bash "$RL/render.sh" refuter "$W" "$W/card.md" > "$W/_prf.txt"
  ( cd "$PROJ" && "${REFUTER_CMD[@]}" "$(cat "$W/_prf.txt")" ) || { say "refuter failed"; exit 3; }
  [ -s "$W/independent.txt" ] || { say "refuter wrote no independent.txt, aborting"; exit 3; }
fi

# 4) gates + collect
say "gates..."
bash "$RL/gates.sh" "$W"
say "collect..."
bash "$RL/collect.sh" "$W"

# 5) Publishing is not done here. Once gates are 7-green, the report is in reports/PR-$PR/.
#    Publishing is a separate step (the phase-2 gate: auto-post non-🔴, hold 🔴), e.g.:
#      bash "$RL/publish.sh" "$PR" --post
say "done. report in $PROJ/.review-loop/reports/PR-$PR/ (uncommitted, unpublished)"

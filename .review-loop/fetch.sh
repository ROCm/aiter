#!/usr/bin/env bash
# Stage P: fetch the PR and produce the full Step 1 artifact set.
# Verified on ROCm/aiter#2510 and #2912, both passing first try on Linux + Claude Code.
set -euo pipefail
PR="${1:?usage: fetch.sh <PR> [owner/repo]}"
REPO="${2:-ROCm/aiter}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Pitfall #1: gh's own login state may be broken (GraphQL 401 Bad credentials observed),
# while the credential in ~/.git-credentials is fine. Feed that to gh; no gh auth login needed.
if ! gh auth status >/dev/null 2>&1 || ! gh api repos/"$REPO" --jq .full_name >/dev/null 2>&1; then
  TOK=$(sed -n 's#https://\([^:]*\):\([^@]*\)@github.com#\2#p' ~/.git-credentials | head -1)
  [ -n "$TOK" ] || { echo "no token: ~/.git-credentials has no github.com entry" >&2; exit 1; }
  export GH_TOKEN="$TOK" GITHUB_TOKEN="$TOK"
fi

# Pitfall #2: REVIEW_AUTO_VALIDATE=1 runs validate-kernel-pr, which needs a GPU of the
# matching arch. Without one, keep it 0 and report the gap honestly on the card — do not
# pass off "not run" as "no problem".
export REVIEW_AUTO_VALIDATE="${REVIEW_AUTO_VALIDATE:-0}" PYTHONUTF8=1

cd "$ROOT"
OUT=$(bash "$ROOT/.claude/skills/review-pr/fetch.sh" "$PR" "$REPO" | tee /dev/stderr)
W=$(printf '%s\n' "$OUT" | sed -n 's/^WORK=//p' | tail -1)

# Pitfall #3 (observed 2026-09-14; a skill bug, wrong on two of three PRs):
# the skill's applies.txt runs `git apply --check` against $PROJECT_ROOT's worktree but
# reports the result as "applies to merge target <BASE_SHA>". Whatever commit PROJECT_ROOT
# sits at is the tree the verdict is based on — and it errs both ways:
#   #2912 reported APPLIES, but actually FAILS on the true merge-target (a miss: the card
#         would drop the "needs a rebase" note)
#   #5339 reported STALE,   but actually APPLIES on the true merge-target (a false alarm:
#         it would wrongly send the author to do a pointless rebase)
# Here we re-test on the merge-target worktree that is actually checked out at BASE_SHA and
# overwrite applies.txt.
if [ -n "$W" ] && [ -d "$W/merge-target" ]; then
  BASE=$(grep -oE "[0-9a-f]{40}" "$W/merge_target.txt" 2>/dev/null | head -1)
  if git -C "$W/merge-target" -c core.fileMode=false apply --check "$W/pr.diff" 2>"$W/.err2"; then
    echo "APPLIES: the diff still applies to merge target $BASE (rechecked on the merge-target worktree)" > "$W/applies.txt"
  else
    { echo "STALE: no longer applies to merge target $BASE -- the PR needs a rebase,"
      echo "  and any CI result on it describes a tree that has moved"
      sed "s/^/  /" "$W/.err2"; } > "$W/applies.txt"
  fi
  echo
  echo "[.review-loop] applies.txt rechecked and overwritten on the merge-target worktree:"
  head -1 "$W/applies.txt" | sed "s/^/  /"
fi

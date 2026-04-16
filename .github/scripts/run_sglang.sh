#!/usr/bin/env bash
# =============================================================================
# run_sglang.sh — Trigger SGLang AITER Scout and wait for completion
#
# Usage:
#   run_sglang.sh <aiter_sha> [job_filter]
#
# Required env:
#   GH_TOKEN — PAT with workflow scope for sgl-project/sglang
#
# Optional env:
#   SGLANG_REPO        — default: sgl-project/sglang
#   SGLANG_SCOUT_WF    — default: amd-aiter-scout.yml
#   SCOUT_TIMEOUT      — max wait seconds (default: 50400 = 14h)
#   SCOUT_POLL_INTERVAL — poll interval seconds (default: 600 = 10m)
# =============================================================================
set -euo pipefail

AITER_SHA="${1:?Usage: run_sglang.sh <aiter_sha> [job_filter]}"
JOB_FILTER="${2:-all}"

SGLANG_REPO="${SGLANG_REPO:-sgl-project/sglang}"
SGLANG_SCOUT_WF="${SGLANG_SCOUT_WF:-amd-aiter-scout.yml}"
SCOUT_TIMEOUT="${SCOUT_TIMEOUT:-50400}"
SCOUT_POLL_INTERVAL="${SCOUT_POLL_INTERVAL:-600}"

echo "=== SGLang AITER Scout ==="
echo "AITER SHA:   ${AITER_SHA}"
echo "Job filter:  ${JOB_FILTER}"
echo "Repo:        ${SGLANG_REPO}"
echo ""

# ── Dispatch ──
gh workflow run "${SGLANG_SCOUT_WF}" \
  --repo "${SGLANG_REPO}" \
  -f aiter_ref="${AITER_SHA}" \
  -f job_filter="${JOB_FILTER}" \
  -f continue_on_error=true

sleep 15

RUN_ID=$(gh run list --repo "${SGLANG_REPO}" \
  --workflow "AMD AITER Scout" --limit 1 \
  --json databaseId --jq '.[0].databaseId')

echo "Scout dispatched: https://github.com/${SGLANG_REPO}/actions/runs/${RUN_ID}"
echo ""

# ── Export for GitHub Actions ──
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "run_id=${RUN_ID}" >> "$GITHUB_OUTPUT"
fi

# ── Wait ──
elapsed=0
while [ "$elapsed" -lt "$SCOUT_TIMEOUT" ]; do
  STATUS=$(gh run view "${RUN_ID}" --repo "${SGLANG_REPO}" \
    --json status,conclusion \
    --jq '{status: .status, conclusion: .conclusion}' 2>/dev/null || echo "")

  if [ -n "$STATUS" ]; then
    RUN_STATUS=$(echo "$STATUS" | jq -r '.status')
    CONCLUSION=$(echo "$STATUS" | jq -r '.conclusion // empty')

    if [ "$RUN_STATUS" = "completed" ]; then
      echo "SGLang Scout completed: ${CONCLUSION}"
      if [ -n "${GITHUB_OUTPUT:-}" ]; then
        echo "result=${CONCLUSION}" >> "$GITHUB_OUTPUT"
      fi
      [ "$CONCLUSION" = "success" ] && exit 0 || exit 1
    fi
  fi

  echo "  Waiting... (${elapsed}s / ${SCOUT_TIMEOUT}s)"
  sleep "$SCOUT_POLL_INTERVAL"
  elapsed=$((elapsed + SCOUT_POLL_INTERVAL))
done

echo "::warning::SGLang Scout timed out after ${SCOUT_TIMEOUT}s"
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "result=timeout" >> "$GITHUB_OUTPUT"
fi
exit 1

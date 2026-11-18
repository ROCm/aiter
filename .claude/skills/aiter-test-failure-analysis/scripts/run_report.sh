#!/usr/bin/env bash
set -euo pipefail

REPO="${1:-${GITHUB_REPOSITORY:-ROCm/aiter}}"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${AITER_TEST_FAILURE_WORKDIR:-$(mktemp -d "${RUNNER_TEMP:-/tmp}/aiter-test-failure.XXXXXX")}"
SUMMARY_FILE="${GITHUB_STEP_SUMMARY:-${WORK_DIR}/summary.md}"
AGENT="${AITER_TEST_FAILURE_AGENT:-claude-glm}"
AGENT_TIMEOUT="${AITER_TEST_FAILURE_AGENT_TIMEOUT:-1800}"
LOOKBACK="${AITER_TEST_FAILURE_LOOKBACK:-120}"
RUN_ID="${AITER_TEST_FAILURE_RUN_ID:-}"

mkdir -p "$WORK_DIR"

args=(
  --repo "$REPO"
  --lookback "$LOOKBACK"
  --out "$WORK_DIR"
)
if [ -n "$RUN_ID" ]; then
  args+=(--run-id "$RUN_ID")
fi

python3 "$SKILL_DIR/scripts/collect_aiter_test_failure.py" "${args[@]}"

REPORT="$WORK_DIR/fallback_report.md"
if [ -s "$WORK_DIR/prompt.md" ] && command -v "$AGENT" >/dev/null 2>&1; then
  set +e
  env -u GH_TOKEN -u GITHUB_TOKEN timeout "$AGENT_TIMEOUT" \
    "$AGENT" -p --dangerously-skip-permissions "$(cat "$WORK_DIR/prompt.md")" \
    > "$WORK_DIR/agent_report.md"
  rc=$?
  set -e
  if [ "$rc" -eq 0 ] && [ -s "$WORK_DIR/agent_report.md" ]; then
    REPORT="$WORK_DIR/agent_report.md"
  else
    {
      echo "> The configured analysis agent (${AGENT}) failed or timed out with exit code ${rc}; using the deterministic fallback report."
      echo
      cat "$WORK_DIR/fallback_report.md"
    } > "$WORK_DIR/agent_fallback_report.md"
    REPORT="$WORK_DIR/agent_fallback_report.md"
  fi
else
  {
    echo "> The configured analysis agent (${AGENT}) is not available; using the deterministic fallback report."
    echo
    cat "$WORK_DIR/fallback_report.md"
  } > "$WORK_DIR/agent_fallback_report.md"
  REPORT="$WORK_DIR/agent_fallback_report.md"
fi

{
  cat "$REPORT"
  echo
  echo "---"
  echo
  echo "Evidence directory on runner: \`$WORK_DIR\`"
} >> "$SUMMARY_FILE"

echo "Aiter Test failure report written to $SUMMARY_FILE"

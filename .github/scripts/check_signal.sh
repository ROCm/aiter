#!/bin/bash

# Gate a downstream workflow on the Checks workflow run for the current commit.
# Exit 0 on success, 78 (neutral/skip) on any other conclusion, 1 if the run
# cannot be located within the retry budget.

set -euo pipefail

CHECKS_WORKFLOW_NAME="${CHECKS_WORKFLOW_NAME:-Checks}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_INTERVAL_SECONDS="${RETRY_INTERVAL_SECONDS:-30}"
REPO="${GITHUB_REPOSITORY:-}"

get_target_branch() {
  if [ -n "${GITHUB_HEAD_REF:-}" ]; then
    printf '%s\n' "${GITHUB_HEAD_REF}"
    return
  fi

  if [ -n "${GITHUB_REF_NAME:-}" ]; then
    printf '%s\n' "${GITHUB_REF_NAME}"
    return
  fi

  python3 - <<'PY'
import json
import os

event_path = os.environ.get("GITHUB_EVENT_PATH")
if event_path and os.path.exists(event_path):
    with open(event_path, encoding="utf-8") as fh:
        data = json.load(fh)
    print(data.get("pull_request", {}).get("head", {}).get("ref", ""))
else:
    print("")
PY
}

get_target_head_sha() {
  case "${GITHUB_EVENT_NAME:-}" in
    pull_request|pull_request_target)
      python3 - <<'PY'
import json
import os

event_path = os.environ.get("GITHUB_EVENT_PATH")
if event_path and os.path.exists(event_path):
    with open(event_path, encoding="utf-8") as fh:
        data = json.load(fh)
    print(data.get("pull_request", {}).get("head", {}).get("sha", ""))
else:
    print("")
PY
      ;;
    *)
      printf '%s\n' "${GITHUB_SHA:-}"
      ;;
  esac
}

find_checks_run_id() {
  local target_branch target_head_sha
  target_branch="$(get_target_branch)"
  target_head_sha="$(get_target_head_sha)"

  if [ -z "${REPO}" ]; then
    echo "GITHUB_REPOSITORY is required to locate the Checks workflow run." >&2
    return 1
  fi

  if [ -z "${target_head_sha}" ]; then
    echo "Could not determine the target head SHA for the Checks workflow run." >&2
    return 1
  fi

  local -a gh_args=(
    run list
    --repo "${REPO}"
    --workflow "${CHECKS_WORKFLOW_NAME}"
    --limit 20
    --json databaseId,headSha,headBranch,event,createdAt,status
  )

  if [ -n "${target_branch}" ]; then
    gh_args+=(--branch "${target_branch}")
  fi

  # Nightly workflows reuse the Checks result from the push run on the same SHA.
  if [ -n "${GITHUB_EVENT_NAME:-}" ] && [ "${GITHUB_EVENT_NAME}" != "schedule" ]; then
    gh_args+=(--event "${GITHUB_EVENT_NAME}")
  fi

  gh "${gh_args[@]}" \
    --jq "(map(select(.headSha == \"${target_head_sha}\")) | first | .databaseId) // empty"
}

# Echoes "<status>\t<conclusion>" for the given run.
fetch_run_state() {
  local run_id="$1"
  gh api "repos/${REPO}/actions/runs/${run_id}" \
    --jq '[.status, (.conclusion // "")] | @tsv'
}

print_failed_jobs() {
  local run_id="$1"
  gh api -X GET "repos/${REPO}/actions/runs/${run_id}/jobs" --paginate \
    --jq '.jobs[] | select(.conclusion != null and .conclusion != "success" and .conclusion != "skipped") | "FAILED: \(.name) (\(.conclusion))"' \
    || true
}

for i in $(seq 1 "${MAX_RETRIES}"); do
  echo "Attempt ${i}: Locating ${CHECKS_WORKFLOW_NAME} workflow run..."

  RUN_ID="$(find_checks_run_id || true)"
  if [ -z "${RUN_ID}" ]; then
    echo "Attempt ${i}: Matching ${CHECKS_WORKFLOW_NAME} run not found yet."
  else
    STATE="$(fetch_run_state "${RUN_ID}" || true)"
    STATUS="${STATE%%$'\t'*}"
    CONCLUSION="${STATE#*$'\t'}"
    # If STATE is unexpected treat that as "no conclusion yet".
    [ "${CONCLUSION}" = "${STATE}" ] && CONCLUSION=""

    echo "Attempt ${i}: run ${RUN_ID} status=${STATUS:-unknown} conclusion=${CONCLUSION:-<none>}"

    if [ "${STATUS}" = "completed" ]; then
      if [ "${CONCLUSION}" = "success" ]; then
        echo "Pre-checks passed, continuing workflow."
        exit 0
      fi

      echo "Pre-checks did not pass (conclusion: ${CONCLUSION:-unknown}), skipping workflow."
      print_failed_jobs "${RUN_ID}"
      exit 78  # 78 = neutral/skip
    fi
  fi

  echo "Pre-checks not ready yet, retrying in ${RETRY_INTERVAL_SECONDS}s..."
  sleep "${RETRY_INTERVAL_SECONDS}"
done

echo "Failed to read pre-checks status after ${MAX_RETRIES} attempts. Exiting workflow."
exit 1

#!/bin/bash

# This script downloads the pre-checks artifact produced by the Checks workflow.
# It scopes artifact lookup to the matching workflow run for the current branch and
# head SHA, which avoids repo-wide artifact pagination and GitHub API rate limits.

set -euo pipefail

ARTIFACT_NAME="checks-signal-${GITHUB_SHA:-${1:-}}"
CHECKS_WORKFLOW_NAME="${CHECKS_WORKFLOW_NAME:-Checks}"
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_INTERVAL_SECONDS="${RETRY_INTERVAL_SECONDS:-30}"
REPO="${GITHUB_REPOSITORY:-}"

if [ -z "${ARTIFACT_NAME#checks-signal-}" ]; then
  echo "GITHUB_SHA or an explicit SHA argument is required."
  exit 1
fi

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

for i in $(seq 1 "${MAX_RETRIES}"); do
  echo "Attempt ${i}: Locating ${CHECKS_WORKFLOW_NAME} workflow run..."
  rm -f checks_signal.txt

  RUN_ID="$(find_checks_run_id || true)"
  if [ -n "${RUN_ID}" ]; then
    echo "Attempt ${i}: Downloading artifact '${ARTIFACT_NAME}' from run ${RUN_ID}..."
    if gh run download "${RUN_ID}" --repo "${REPO}" --name "${ARTIFACT_NAME}"; then
      if [ -f checks_signal.txt ]; then
        echo "Artifact ${ARTIFACT_NAME} downloaded successfully."
        SIGNAL="$(head -n 1 checks_signal.txt)"
        if [ "${SIGNAL}" = "success" ]; then
          echo "Pre-checks passed, continuing workflow."
          exit 0
        fi

        echo "Pre-checks failed, skipping workflow. Details:"
        tail -n +2 checks_signal.txt
        exit 78  # 78 = neutral/skip
      fi
    fi
  else
    echo "Attempt ${i}: Matching ${CHECKS_WORKFLOW_NAME} run not found yet."
  fi

  echo "Artifact not ready yet, retrying in ${RETRY_INTERVAL_SECONDS}s..."
  sleep "${RETRY_INTERVAL_SECONDS}"
done

echo "Failed to download pre-checks artifact after ${MAX_RETRIES} attempts. Exiting workflow."
exit 1

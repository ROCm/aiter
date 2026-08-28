#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <image>" >&2
  exit 2
fi

IMAGE="$1"
MAX_ATTEMPTS="${DOCKER_PULL_MAX_ATTEMPTS:-3}"
RETRY_DELAY_SECONDS="${DOCKER_PULL_RETRY_DELAY_SECONDS:-10}"

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
  echo "Pulling Docker image '${IMAGE}' (attempt ${attempt}/${MAX_ATTEMPTS})"
  if docker pull "${IMAGE}"; then
    echo "Docker pull succeeded for '${IMAGE}' on attempt ${attempt}"
    exit 0
  fi

  if [ "${attempt}" -lt "${MAX_ATTEMPTS}" ]; then
    echo "Docker pull failed for '${IMAGE}' on attempt ${attempt}; retrying in ${RETRY_DELAY_SECONDS}s..."
    sleep "${RETRY_DELAY_SECONDS}"
  fi
done

echo "Docker pull failed for '${IMAGE}' after ${MAX_ATTEMPTS} attempts" >&2
exit 1

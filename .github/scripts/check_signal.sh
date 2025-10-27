#!/bin/bash

set -e

ARTIFACT_NAME="checks-signal-${GITHUB_SHA:-${1}}"
MAX_RETRIES=5

for i in $(seq 1 $MAX_RETRIES); do
  echo "Attempt $i: Downloading artifact..."
  if gh run download --name "$ARTIFACT_NAME"; then
    if [ -f checks_signal.txt ]; then
      echo "Artifact $ARTIFACT_NAME downloaded successfully."
      SIGNAL=$(head -n 1 checks_signal.txt)
      if [ "$SIGNAL" = "success" ]; then
        echo "Pre-checks passed, continuing workflow."
        exit 0
      else
        echo "Pre-checks failed, skipping workflow. Details:"
        tail -n +2 checks_signal.txt
        exit 78  # 78 = neutral/skip
      fi
    fi
  fi
  echo "Artifact not found, retrying in 30s..."
  sleep 30
done

echo "Failed to download pre-checks artifact after $MAX_RETRIES attempts. Exiting workflow."
exit 1

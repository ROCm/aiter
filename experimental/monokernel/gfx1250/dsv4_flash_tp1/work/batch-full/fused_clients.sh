#!/bin/bash
set -euo pipefail
for con in 2 4 8; do
 bash /work/concurrency-rollout/clients.sh batch-full-fused 11600 "$con"
done

#!/bin/bash
set -euo pipefail
export PYTHONPATH=/work/rollout-c1248:/app/aiter:/app/ATOM TEST_BATCHES=${1:?batch}
while [ ! -f /work/rollout-c1248/variants-c${TEST_BATCHES}.done ]; do sleep 2; done
python /work/rollout-c1248/paired_swa.py
RATIO=4 python /work/rollout-c1248/paired_compressed.py
RATIO=128 python /work/rollout-c1248/paired_compressed.py
touch /work/rollout-c1248/paired-c${TEST_BATCHES}.done

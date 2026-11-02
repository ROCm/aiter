#!/bin/bash
set -euo pipefail
export PYTHONPATH=/work/rollout-c1248:/app/aiter:/app/ATOM TEST_BATCHES=${1:?batch}
python /work/rollout-c1248/final_swa.py
RATIO=4 python /work/rollout-c1248/final_compressed.py
RATIO=128 python /work/rollout-c1248/final_compressed.py
touch /work/rollout-c1248/final-c${TEST_BATCHES}.done

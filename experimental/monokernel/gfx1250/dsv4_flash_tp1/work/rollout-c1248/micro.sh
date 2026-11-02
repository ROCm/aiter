#!/bin/bash
set -euo pipefail
export PYTHONPATH=/work/rollout-c1248:/app/aiter:/app/ATOM
export TEST_BATCHES=${1:?batch} REP=30
python /work/rollout-c1248/micro_swa.py
RATIO=4 python /work/rollout-c1248/micro_compressed.py
RATIO=128 python /work/rollout-c1248/micro_compressed.py
touch /work/rollout-c1248/micro-c${TEST_BATCHES}.done

#!/bin/bash
# first_touch.sh — the minimal-first-touch ladder, run INSIDE the container.
#   bash first_touch.sh [timeout_s] [tag]
# Every step is its own process with a hard wall-clock timeout (default 60 s) and
# PYTHONFAULTHANDLER=1, so a hang names itself and keeps its log.
set -uo pipefail
TIMEOUT=${1:-60}
TAG=${2:-ft}
export PYTHONPATH=/work/pylibs:/work/work/qsa-v0
export QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline
export QSA_RESULTS=${QSA_RESULTS:-/work/results/g2/builder}
export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-$QSA_RESULTS/flydsl_cache_$TAG}
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
OUT="$QSA_RESULTS/first_touch_$TAG"
mkdir -p "$QSA_RESULTS" "$FLYDSL_RUNTIME_CACHE_DIR" "$OUT"
cd /work/work/qsa-v0/aiter/ops/flydsl/kernels/mqa_logits/_g2 || exit 9
echo "== first touch: timeout=${TIMEOUT}s out=$OUT =="
python3 -u first_touch.py --timeout "$TIMEOUT" --out "$OUT"
RC=$?
echo "== first touch rc=$RC (summary: $OUT/first_touch_summary.json) =="
exit "$RC"

#!/bin/bash
# bisect.sh - the construct-bisect ladder, run INSIDE the container (thin wrapper).
#
#   bash bisect.sh [timeout_s] [tag] [extra bisect.py args...]
#   bash bisect.sh 10 b1 --include-orig
#   bash bisect.sh 10 b1 --only store_pred_cop,store_uncond_cop
#
# Every rung is its own process with PYTHONFAULTHANDLER=1 and a phase-aware hard
# timeout: --ready-timeout for imports, --compile-timeout for the JIT/launch, and
# timeout_s (default 10) for the part that can wedge the GPU, i.e. from
# `dispatched` to the process exit.
#
# A TIMEOUT in the kernel phase means the queue never completed: STOP. Do not
# retry, do not run the next rung, capture dmesg and ask for a node reset.
# See BISECT.md for the decision table.
set -uo pipefail
TIMEOUT=${1:-10}
TAG=${2:-b1}
shift 2 2>/dev/null || true

export PYTHONPATH=${PYTHONPATH:-/work/pylibs:/work/work/qsa-v0}
export QSA_GOLDEN_DIR=${QSA_GOLDEN_DIR:-/work/task/MI455_QSA_INDEX_SCORER/golden_baseline}
export QSA_RESULTS=${QSA_RESULTS:-/work/results/g2/builder}
export FLYDSL_RUNTIME_CACHE_DIR=${FLYDSL_RUNTIME_CACHE_DIR:-$QSA_RESULTS/flydsl_cache_$TAG}
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
OUT="$QSA_RESULTS/bisect_$TAG"
mkdir -p "$QSA_RESULTS" "$FLYDSL_RUNTIME_CACHE_DIR" "$OUT" || exit 9
cd /work/work/qsa-v0/aiter/ops/flydsl/kernels/mqa_logits/_g2 || exit 9

echo "== bisect: kernel-timeout=${TIMEOUT}s tag=$TAG out=$OUT =="
echo "== access layer: QSA_G2_ACCESS=${QSA_G2_ACCESS:-(module default = raw)} =="
python3 -u bisect.py --timeout "$TIMEOUT" --out "$OUT" "$@"
RC=$?
echo "== bisect rc=$RC (summary: $OUT/bisect_summary.json) =="
if [ "$RC" = "3" ]; then
  echo "== READ THE LOGS BEFORE ANY RETRY: $OUT/bs_*.log =="
fi
exit "$RC"

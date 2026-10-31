#!/bin/bash
# CPU-only contract test (runs inside the container, no GPU device needed).
set -euo pipefail
export PYTHONPATH=/work/pylibs:/work/work/qsa-v0
export QSA_GOLDEN_DIR=/work/task/MI455_QSA_INDEX_SCORER/golden_baseline
export QSA_RESULTS=${QSA_RESULTS:-/work/results/g2/builder}
mkdir -p "$QSA_RESULTS"
cd /work/work/qsa-v0/aiter/ops/flydsl/kernels/mqa_logits/_g2
python3 test_contract_cpu.py 2>&1 | tee "$QSA_RESULTS/cpu_contract.log"

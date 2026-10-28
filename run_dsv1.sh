#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

flydsl_wheel="./flydsl-0.3.3.dev882-cp312-cp312-linux_x86_64.whl"
if [[ ! -f "$flydsl_wheel" ]]; then
  printf 'Required wheel not found: %s\n' "$flydsl_wheel" >&2
  exit 1
fi
python3 -m pip install --no-deps --force-reinstall "$flydsl_wheel"

python3 - <<'PY'
from importlib.metadata import version

installed = version("flydsl")
expected = "0.3.3.dev882"
if installed.split("+", 1)[0] != expected:
    raise SystemExit(f"Expected FlyDSL {expected}, found {installed}; benchmark not started.")
print(f"FlyDSL version: {installed}")
PY

AITER_USE_GROUPED_GEMM=1 \
AITER_GROUPED_DEBUG=0 \
ENABLE_CK=0 \
FLYDSL_DUMP_IR=1 \
AITER_LOG_MORE=1 \
AITER_MOE_EXPERT_BALANCE=true \
AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1 \
AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1 \
python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
  --scenario kernel \
  --data-format a4w4 \
  --experts 64 \
  --tokens 1536 \
  --topk 8 \
  --model-dim 7168 \
  --inter-dim 2048 \
  --act silu \
  --no-bias \
  --iters 128 \
  --no-check-aot-cache \
  --data-init "${AITER_DATA_INIT:-uniform}" \
  --scale-init "${AITER_SCALE_INIT:-uniform}"

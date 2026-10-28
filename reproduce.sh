#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

# Reproduce the ~510 us GEMM1 zero-input workload on gfx1250.
# Previously measured with FlyDSL 0.3.3.dev882; timing depends on GPU state.
python3 - <<'PY'
from importlib.metadata import version

installed = version("flydsl")
if installed != "0.3.3.dev882":
    raise SystemExit(
        f"Expected flydsl==0.3.3.dev882, found {installed}. "
        "Install the validated version before reproducing the timing."
    )
PY

export AITER_USE_GROUPED_GEMM=1
export AITER_GROUPED_DEBUG=0
export ENABLE_CK=0
export FLYDSL_DUMP_IR=0
export AITER_LOG_MORE=1
export AITER_MOE_EXPERT_BALANCE=true

# Workload-specific configuration: GEMM1 WPT2, GEMM2 WPT1, rolled drain off.
export AITER_CONFIG_GROUPED_FMOE="$script_dir/aiter/configs/benchmarks/a4w4_conc16384.csv"
# Validated for this balanced workload, not a general cluster-sync default.
export AITER_FLYDSL_TDM_EARLY_TIMEOUT=0
export AITER_FLYDSL_DISABLE_CLUSTER_SYNC=1
export AITER_FLYDSL_QUAD_PIPELINE=1
export AITER_FLYDSL_MXFP4_CLUSTER_N=4
export AITER_FLYDSL_MXFP4_CLUSTER_M=2

# Keep A-preshuffled inputs, with GEMM1 producing a BF16 intermediate and
# a separate quant/preshuffle kernel preparing GEMM2's input.
export AITER_GROUPED_A4W4_A_PRESHUFFLE=1
export AITER_FLYDSL_DISABLE_GEMM1_REQUANT=1
# Pin the measured schedule independently of the calling shell.
export AITER_FLYDSL_MMA_FIRST_GROUP=4
export AITER_FLYDSL_MMA_GROUP=4
export AITER_FLYDSL_DS_FIRST_N=8
export AITER_FLYDSL_EXPLICIT_VGPR_PARTITION=1
# export AITER_FLYDSL_PLANAR_LDS=1
# export AITER_FLYDSL_INTERLEAVED_LDS_LOAD=1
export AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE=1
export AITER_FLYDSL_DISABLE_GEMM1_ACT=1
export AITER_FLYDSL_LDS_WAR_ORDER=1

# Never start GPU work while another process is reported. The version check
# above uses package metadata only and does not initialize the GPU.
last_report=0
while true; do
  gpu_state=$(rocm-smi --showpids 2>&1) || {
    printf '%s\n' "$gpu_state" >&2
    exit 1
  }
  if [[ "$gpu_state" == *"No KFD PIDs currently running"* ]]; then
    break
  fi
  now=$(date +%s)
  if (( now - last_report >= 30 )); then
    printf 'Waiting for GPU idle...\n%s\n' "$gpu_state"
    last_report=$now
  fi
  sleep 1
done

exec python3 -u op_tests/flydsl_tests/test_flydsl_grouped_gemm.py \
  --scenario kernel \
  --data-format a4w4 \
  --experts 1 \
  --tokens 16384 \
  --topk 1 \
  --model-dim 16384 \
  --inter-dim 8192 \
  --act silu \
  --no-bias \
  --iters 3 \
  --no-check-aot-cache \
  "$@"

#!/usr/bin/env bash
# Roofline profiling for the jagged_dense_bmm backward kernels with rocprof-compute.
#
# Overcomes the two practical problems with running rocprof-compute on a FlyDSL
# (Python) workload that imports PyTorch:
#
#   1. rocprof-compute is itself a Python app (pandas/dash/matplotlib/...). We do
#      NOT install those into flydsl_venv. Instead the profiler DRIVER runs in a
#      dedicated rocprof_venv, while the profiled APP is launched explicitly with
#      flydsl_venv's interpreter. Two interpreters, zero dependency cross-talk.
#
#   2. The PyTorch wheel bundles its own copies of the ROCm profiling runtime
#      (librocprofiler-register.so, librocprofiler-sdk.so, libroctracer64.so) in
#      torch/lib and loads them via DT_RPATH=$ORIGIN -- which the dynamic linker
#      searches BEFORE LD_LIBRARY_PATH/LD_PRELOAD. With rocprofv3's tool active
#      this yields a SECOND rocprofiler stack in the process, so the runtime
#      registers outside the tool's configuration window and aborts:
#        "api registration failed with error code 16: Configuration request
#         occurred outside of valid rocprofiler configuration period".
#      Because these bundled libs are the exact same version (7.2.70200) as the
#      system ROCm, the fix is to make torch use the SINGLE system stack. The
#      only way (RPATH beats LD_LIBRARY_PATH) is for those files to be absent
#      from torch/lib at load time, so we move them aside for the duration of
#      the run and ALWAYS restore them on exit. The venv is byte-for-byte
#      unchanged afterwards.
#
# Usage:
#   bash profile_roofline.sh [--only djagged|ddense|dbias|all] [-b N] [-m N]
#                            [--regime uniform|skew] [--iters N] [--warmup N]
#                            [--roof-only|--full] [--name NAME]
# Examples:
#   bash profile_roofline.sh --only ddense
#   bash profile_roofline.sh --only all --full -b 64 -m 512
set -euo pipefail

ROCM=/opt/rocm
KDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# aiter repo root is four levels up: kernels -> flydsl -> ops -> aiter -> <root>
ROOT="$(cd "$KDIR/../../../.." && pwd)"
FLYDSL_PY="$ROOT/flydsl_venv/bin/python"
ROCPROF_PY="$ROOT/rocprof_venv/bin/python"
ROCPROF_COMPUTE="$ROCM/libexec/rocprofiler-compute/rocprof-compute"
HARNESS="$KDIR/profile_jagged_dense_bmm_bwd.py"

# Defaults
ONLY=all; NGROUPS=64; MAXSEQ=512; REGIME=uniform; ITERS=50; WARMUP=10
ROOF_MODE="--roof-only"; NAME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --only) ONLY="$2"; shift 2;;
    -b|--n-groups) NGROUPS="$2"; shift 2;;
    -m|--max-seq-len) MAXSEQ="$2"; shift 2;;
    --regime) REGIME="$2"; shift 2;;
    --iters) ITERS="$2"; shift 2;;
    --warmup) WARMUP="$2"; shift 2;;
    --roof-only) ROOF_MODE="--roof-only"; shift;;
    --full) ROOF_MODE=""; shift;;          # full counter collection (many passes)
    --name) NAME="$2"; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done
[[ -z "$NAME" ]] && NAME="bwd_${ONLY}_b${NGROUPS}_m${MAXSEQ}_${REGIME}"

for p in "$FLYDSL_PY" "$ROCPROF_PY" "$ROCPROF_COMPUTE" "$HARNESS"; do
  [[ -e "$p" ]] || { echo "missing: $p" >&2; exit 1; }
done

# --- PyTorch clash mitigation: locate torch/lib and the bundled rocprofiler libs.
TORCH_LIB="$("$FLYDSL_PY" -c 'import torch,os;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
SHIM_LIBS=(librocprofiler-register.so librocprofiler-sdk.so libroctracer64.so)
SUFFIX=".rocprof-disabled"

restore_torch_libs() {
  for n in "${SHIM_LIBS[@]}"; do
    if [[ -e "$TORCH_LIB/$n$SUFFIX" ]]; then
      mv -f "$TORCH_LIB/$n$SUFFIX" "$TORCH_LIB/$n"
    fi
  done
}
disable_torch_libs() {
  # Restore any leftover from a previous interrupted run first (idempotent).
  restore_torch_libs
  for n in "${SHIM_LIBS[@]}"; do
    if [[ -e "$TORCH_LIB/$n" ]]; then
      mv "$TORCH_LIB/$n" "$TORCH_LIB/$n$SUFFIX"
    fi
  done
}
trap restore_torch_libs EXIT INT TERM

echo "== rocprof-compute roofline =="
echo "   workload name : $NAME"
echo "   torch/lib     : $TORCH_LIB"
echo "   mode          : ${ROOF_MODE:-full-counters}"
echo "   harness args  : --only $ONLY -b $NGROUPS -m $MAXSEQ --regime $REGIME --iters $ITERS --warmup $WARMUP"

disable_torch_libs

OUTDIR="$ROOT/workloads/$NAME"
rm -rf "${OUTDIR:?}"
mkdir -p "$OUTDIR"
# Reuse the GPU-wide empirical roofline (roofline.csv) across runs if we already
# measured it once -- it depends only on the device, not the workload, and costs
# ~60s to regenerate.
CACHED_ROOF="$ROOT/workloads/roofline.csv"
[[ -f "$CACHED_ROOF" ]] && cp "$CACHED_ROOF" "$OUTDIR/roofline.csv"

# System ROCm libs first so torch's NEEDED rocprofiler/HSA deps resolve to the
# single system stack (torch/lib copies are temporarily absent).
export LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}"
export PATH="$ROCM/bin:$PATH"
export FLYDSL_RUNTIME_ENABLE_CACHE=1   # reuse JIT artifacts across profiler replays

set +e
"$ROCPROF_PY" "$ROCPROF_COMPUTE" profile -n "$NAME" -p "$OUTDIR" $ROOF_MODE -- \
  "$FLYDSL_PY" "$HARNESS" --mode profile --only "$ONLY" \
  -b "$NGROUPS" -m "$MAXSEQ" --regime "$REGIME" --iters "$ITERS" --warmup "$WARMUP"
RC=$?
set -e

# Cache the GPU-wide empirical roofline for future runs.
[[ -f "$OUTDIR/roofline.csv" ]] && cp "$OUTDIR/roofline.csv" "$CACHED_ROOF"

# Overlay our kernels onto the empirical roofline (no GPU work; safe without the
# lib swap). Produces empirRoof_*.pdf with the application's points.
restore_torch_libs
"$ROCPROF_PY" "$ROCPROF_COMPUTE" analyze -p "$OUTDIR" --roof-only >/dev/null 2>&1 || true

echo "== profile rc=$RC; workload at $OUTDIR =="
echo "   roofline PDFs : $OUTDIR/empirRoof_*.pdf"
echo "   app counters  : $OUTDIR/pmc_perf.csv"
exit $RC

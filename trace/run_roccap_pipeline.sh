#!/usr/bin/env bash
#
# Runs the roccap capture/play/extract pipeline, then sp3disasm and amtool
# using the SHA extracted from the roccap extract output.
#
# The runnable entry point is gemm_afp4wfp4.py (runs __main__, uses config from
# _get_config, launches Gluon gemm_mxfp4_preshuffle_gfx1250 on gfx1250).
# Requires PYTHONPATH to include the AITER repo root so "aiter" and "op_tests"
# resolve (the CLI imports generate_gemm_afp4wfp4_inputs from op_tests).
#
# Usage (from this trace/ directory):
#   export PYTHONPATH=/path/to/aiter/repo/root
#   ./run_roccap_pipeline.sh <kernel_disp_name> <path_to_python_script>
#
# Example:
#   ./run_roccap_pipeline.sh gemm_mxfp4_preshuffle_gfx1250 \
#     ../aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py
#
# If capture finds no kernel, list available names from a trial run or adjust
# --disp to match how roccap labels the Gluon kernel on your stack.

# Do not abort on failures; continue through the pipeline.
set +e

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <kernel_disp_name> <path_to_python_script>" >&2
  echo "Set PYTHONPATH to AITER repo root (for aiter + op_tests)." >&2
  echo "Example: $0 gemm_mxfp4_preshuffle_gfx1250 ../aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py" >&2
  exit 1
fi

KERNEL_NAME="$1"
KERNEL_FILE="$2"
CAP_BASE="gemm"
ROCCAP="${TRITON_GFX1250_MODEL_PATH:?TRITON_GFX1250_MODEL_PATH must be set}/tools/roccap/bin/roccap"
FFMLITE_ENV="$TRITON_GFX1250_MODEL_PATH/ffmlite_env.sh"
AM_ENV="$TRITON_GFX1250_MODEL_PATH/am_env.sh"
SP3DISASM="$TRITON_GFX1250_MODEL_PATH/ffm-lite/sp3disasm"
AMTOOL="$TRITON_GFX1250_MODEL_PATH/tools/rcv/amtool"

clear || true
source "$FFMLITE_ENV" || true
"$ROCCAP" capture --loglevel trace --disp "${KERNEL_NAME}/0" --file "${CAP_BASE}.cap" python3 "$KERNEL_FILE" || true

source "$AM_ENV" || true
export DtifFbBaseLocation=0x200000000
"$ROCCAP" play -r "0x200000000-0xF00000000" "./${CAP_BASE}_0002.cap" || true
grep -A1 "WGP00" xcc0se0sa0_itrace_emu.mon > wgp0.txt || true
python3 ./gen_perfetto.py wgp0.txt out.json || true

# Run extract and capture output to parse SHA from "Wrote roc-dump-<SHA>-isa-data.bin"
EXTRACT_OUT=$("$ROCCAP" extract --sp3 0- "./${CAP_BASE}_0002.cap" 2>&1) || true
SHA=$(echo "$EXTRACT_OUT" | sed -n 's/.*roc-dump-\([0-9]*-[0-9]*\)-isa-data\.bin.*/\1/p' | tail -1)

if [[ -z "$SHA" ]]; then
  echo "Warning: could not find SHA in roccap extract output. Last lines:" >&2
  echo "$EXTRACT_OUT" | tail -20 >&2
else
  ROC_DUMP="roc-dump-${SHA}-isa-data.bin"
  echo "Using SHA: $SHA -> $ROC_DUMP"
  "$SP3DISASM" "./$ROC_DUMP" gemm.sp3 || true
  "$AMTOOL" gemm/ *.mon gemm.sp3 || true
fi

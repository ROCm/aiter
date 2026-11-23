#!/usr/bin/env bash
# Run the Flash Attention Triton tests of aiter CI. CI and a local container run the same lines.
#
# Usage: test.sh <v2-pr | v2-pr-rdna3 | v2-full | v3-pr | v3-full>
#   Run it in a container that has flash-attention at /flash-attention.
#
# Each pytest line is its own process. A failure or abort does not stop the next line, and the
# script fails if a line fails. Skip a known failure with a --deselect line under its test.
# Within a line, the tests run one at a time in a pytest-xdist worker: a test that crashes it
# (abort, GPU memory fault) or hangs past --timeout (the thread method exits the worker) is
# reported as failed, and the rest of the line runs on a fresh worker.
# A --deselect ID is relative to pytest's rootdir: tests/ for v2 (tests/pyproject.toml).
set -euo pipefail
cd /flash-attention || exit 2
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=0
V2=tests/test_flash_attn_triton_amd.py
V3=hopper/test_flash_attn_triton_amd.py
status=0
PYTEST=(pytest -v -rfE --reruns 2 --timeout=120 --timeout-method=thread)
if python -c "import xdist" 2>/dev/null; then
    PYTEST+=(-n 1 --max-worker-restart=100000)
else
    echo "pytest-xdist not installed: a crashing test stops the rest of its pytest line" >&2
fi
# Triton 3.9: these fault or hang in bwd_kernel_fused_causal (LLVM miscompile, fix pending in
# https://github.com/llvm/llvm-project/pull/226689).
OUTPUT_KNOWN_FAIL=(--deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-2048-2048-128-True-False-False-False-mha-dtype0-True]" --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-2048-2048-128-True-False-False-False-mha-dtype0-False]")
VARLEN_KNOWN_FAIL=(--deselect "test_flash_attn_triton_amd.py::test_flash_attn_varlen_output[0.0-0.17-1024-1024-32-True-False-True-False-mha-dtype0-False]")

case "${1:-}" in
v2-pr)
    "${PYTEST[@]}" $V2::test_flash_attn_output -k 2048-2048 "${OUTPUT_KNOWN_FAIL[@]}" || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_varlen_output -k 1024-1024 "${VARLEN_KNOWN_FAIL[@]}" || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_kvcache -k 16-20000 || status=1
    ;;
v2-pr-rdna3)
    # Triton 3.9: fail on RDNA3 (gfx1100); the dtype0-False case crashes the process.
    "${PYTEST[@]}" $V2::test_flash_attn_output -k 2048-2048 \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-2048-2048-96-False-False-False-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-2048-2048-96-False-False-False-False-mha-dtype0-False]" || status=1
    ;;
v2-full)
    "${PYTEST[@]}" $V2::test_flash_attn_qkvpacked || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_varlen_qkvpacked || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_output "${OUTPUT_KNOWN_FAIL[@]}" || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_varlen_output "${VARLEN_KNOWN_FAIL[@]}" || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_causal || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_varlen_causal || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_splitkv || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_kvcache || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_race_condition || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_bwd_overflow || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_bwd_transpose || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_bwd_varlen_overflow || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_deterministic || status=1
    "${PYTEST[@]}" $V2::test_flash_attn_varlen_deterministic || status=1
    ;;
v3-pr)
    export PYTHONPATH=/flash-attention/hopper:${PYTHONPATH:-}
    "${PYTEST[@]}" $V3::test_flash_attn_output -k 4096-4096 || status=1
    "${PYTEST[@]}" $V3::test_flash_attn_varlen_output -k 1024-1024 || status=1
    "${PYTEST[@]}" $V3::test_flash_attn_kvcache -k 16-20000 || status=1
    ;;
v3-full)
    export PYTHONPATH=/flash-attention/hopper:${PYTHONPATH:-}
    "${PYTEST[@]}" $V3::test_flash_attn_output || status=1
    "${PYTEST[@]}" $V3::test_flash_attn_varlen_output || status=1
    "${PYTEST[@]}" $V3::test_flash_attn_kvcache || status=1
    "${PYTEST[@]}" $V3::test_flash_attn_cluster || status=1
    "${PYTEST[@]}" $V3::test_flash_attn_race_condition || status=1
    "${PYTEST[@]}" $V3::test_flash_attn_combine || status=1
    "${PYTEST[@]}" $V3::test_flash3_bw_compatibility || status=1
    ;;
*)
    echo "usage: $0 <v2-pr | v2-pr-rdna3 | v2-full | v3-pr | v3-full>" >&2
    exit 2
    ;;
esac
exit $status

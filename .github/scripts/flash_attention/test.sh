#!/usr/bin/env bash
# Run the Flash Attention Triton tests of aiter CI. CI and a local container run the same lines.
#
# Usage: test.sh <v2-pr | v2-pr-rdna3 | v2-full | v3-pr | v3-full>
#   Run it in a container that has flash-attention at /flash-attention.
#
# Each pytest line is its own process. A failure or abort does not stop the next line, and the
# script fails if a line fails. Skip a known failure with a --deselect line under its test.
# A --deselect ID is relative to pytest's rootdir: tests/ for v2 (tests/pyproject.toml).
set -euo pipefail
cd /flash-attention || exit 2
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
export FLASH_ATTENTION_TRITON_AMD_AUTOTUNE=0
V2=tests/test_flash_attn_triton_amd.py
V3=hopper/test_flash_attn_triton_amd.py
status=0

case "${1:-}" in
v2-pr)
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_output -k 2048-2048 || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_varlen_output -k 1024-1024 || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_kvcache -k 16-20000 || status=1
    ;;
v2-pr-rdna3)
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_output -k 2048-2048 || status=1
    ;;
v2-full)
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_qkvpacked || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_varlen_qkvpacked || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_output \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-1023-1024-59-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-1023-1024-59-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-1023-1024-64-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-1023-1024-64-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-1024-1023-40-False-False-True-False-gqa-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-1024-1023-40-False-False-True-False-gqa-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-1024-1023-59-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-1024-1023-59-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-1024-1023-64-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-1024-1023-64-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-108-256-59-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-108-256-59-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-108-256-64-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-108-256-64-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-113-203-59-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-113-203-59-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-113-203-64-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-113-203-64-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-113-211-59-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-113-211-59-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-113-211-64-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-113-211-64-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-128-217-40-False-False-True-False-gqa-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-128-217-40-False-False-True-False-gqa-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-128-217-59-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-128-217-59-True-False-True-False-mha-dtype0-True]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-128-217-64-True-False-True-False-mha-dtype0-False]" \
        --deselect "test_flash_attn_triton_amd.py::test_flash_attn_output[0.0-0.17-128-217-64-True-False-True-False-mha-dtype0-True]" \
        || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_varlen_output || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_causal || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_varlen_causal || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_splitkv || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_kvcache || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_race_condition || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_bwd_overflow || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_bwd_transpose || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_bwd_varlen_overflow || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_deterministic || status=1
    pytest -v --reruns 2 --timeout=120 $V2::test_flash_attn_varlen_deterministic || status=1
    ;;
v3-pr)
    export PYTHONPATH=/flash-attention/hopper:${PYTHONPATH:-}
    pytest -v --reruns 2 --timeout=120 $V3::test_flash_attn_output -k 4096-4096 || status=1
    pytest -v --reruns 2 --timeout=120 $V3::test_flash_attn_varlen_output -k 1024-1024 || status=1
    pytest -v --reruns 2 --timeout=120 $V3::test_flash_attn_kvcache -k 16-20000 || status=1
    ;;
v3-full)
    export PYTHONPATH=/flash-attention/hopper:${PYTHONPATH:-}
    pytest -v --reruns 2 --timeout=120 $V3::test_flash_attn_output || status=1
    pytest -v --reruns 2 --timeout=120 $V3::test_flash_attn_varlen_output || status=1
    pytest -v --reruns 2 --timeout=120 $V3::test_flash_attn_kvcache || status=1
    pytest -v --reruns 2 --timeout=120 $V3::test_flash_attn_cluster || status=1
    pytest -v --reruns 2 --timeout=120 $V3::test_flash_attn_race_condition || status=1
    pytest -v --reruns 2 --timeout=120 $V3::test_flash_attn_combine || status=1
    pytest -v --reruns 2 --timeout=120 $V3::test_flash3_bw_compatibility || status=1
    ;;
*)
    echo "usage: $0 <v2-pr | v2-pr-rdna3 | v2-full | v3-pr | v3-full>" >&2
    exit 2
    ;;
esac
exit $status

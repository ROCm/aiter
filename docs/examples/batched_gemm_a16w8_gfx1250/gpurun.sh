#!/bin/bash
# Run one command on one GPU, holding that GPU's lock for the whole command.
#
#   ./gpurun.sh <gpu_id> <cmd...>
#
# Every GPU job in this suite goes through here, so no GPU ever has more than
# one process in flight (two concurrent jobs on one GPU can take the machine
# down). A second job for a busy GPU blocks until the first one exits.
#
# AITER_ROOT: checkout to run against (must contain batched_gemm_a16w8);
#             defaults to the checkout this script lives in.
# AITER_GPU_LOCK_DIR: where the per-GPU lock files live (default /tmp/aiter-gpu-locks).
set -u
GPU="$1"
shift
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "bad gpu id: $GPU" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AITER_ROOT="${AITER_ROOT:-$(cd "$HERE/../../.." && pwd)}"
LOCK_DIR="${AITER_GPU_LOCK_DIR:-/tmp/aiter-gpu-locks}"
mkdir -p "$LOCK_DIR"
exec 9>"$LOCK_DIR/gpu${GPU}.lock"
flock 9
export HIP_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$AITER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$AITER_ROOT" || exit 2
echo "[gpurun] gpu=$GPU start $(date +%T)" >&2
"$@"
rc=$?
echo "[gpurun] gpu=$GPU end $(date +%T) rc=$rc" >&2
exit $rc

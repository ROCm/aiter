#!/bin/bash
# usage: mono_run.sh <name> <c> [K=V ...]   (inside qwen36g) - one point on the gdn1112 trees, extra env as EXTRA
n=$1; c=$2; shift 2
MONO="ATOM_MTP_MONO_DIR=/work/mono/dk ATOM_MTP_MONO_HEAD=/data/Qwen3.6-35B-A3B-MXFP4-experts-MXFP8-gemm-lmhead"
ulimit -c 0
while pgrep -f "[a]tom.entrypoints.openai_server|[i]nference-endpoint" > /dev/null; do sleep 10; done
EXTRA="$MONO $*" AITER_DIR=/work/push/gdn1112_aiter R=/work/qwen35_ep /work/push/gdn1112_atom/recipes/mlperf_endpoints_qwen35/run_point_20261107.sh $c /runs/$n > /runs/$n.run.log 2>&1

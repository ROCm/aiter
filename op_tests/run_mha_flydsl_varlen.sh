#!/bin/bash
set -e

FFM_DIR=$(ls -d /data/docker/overlay2/*/diff/home/user/ffm-env/rocdtif-7.13-am+ffmlite-mi400.*-rel-* 2>/dev/null | head -1)
if [ -z "$FFM_DIR" ]; then
    echo "ERROR: rocdtif-7.13+ ffm-lite not found" >&2
    exit 1
fi

source "$FFM_DIR/ffmlite_env.sh"
export FLYDSL_ROOT=/data/zanzhang/FlyDSL-main
export FLYDSL_RUNTIME_ENABLE_CACHE=0

cd "$(dirname "$0")"

echo "=== FlyDSL MHA Varlen Test (causal + non-causal) ==="
python3 -u test_mha_flydsl_varlen.py

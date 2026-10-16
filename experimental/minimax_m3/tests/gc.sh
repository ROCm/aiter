#!/bin/bash
# compile-only Gluon build in the container (GPU 4, no launch)
docker exec -e VLLM_M3_GLUON_TRACE=${TRACE:-0} -e STAGES=${STAGES:-511} -e LOOP=${LOOP:-0} -e HIP_VISIBLE_DEVICES=4 -e TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton ix_dsv41 timeout 900 python3 /home/vigokhal/m3_monokernel/tests/gluon_compile.py "$@" 2>&1 | grep -v "^\[aiter\]\|queue-intercept"

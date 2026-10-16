#!/bin/bash
# per-stage compile probe: sp.sh S "stages" EXTRA
docker exec -e S=$1 -e EXTRA=$3 -e HIP_VISIBLE_DEVICES=4 -e TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton ix_dsv41 timeout 900 python3 /home/vigokhal/m3_monokernel/tests/probes/stage_probe.py $2 2>&1 | grep -v "^\[aiter\]\|queue-intercept"

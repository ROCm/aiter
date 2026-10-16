#!/bin/bash
docker exec -e S=${S:-1} -e DBG=${DBG:-3584} -e HIP_VISIBLE_DEVICES=4 -e TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton ix_dsv41 bash -c "ulimit -c 0; timeout 300 python3 /home/vigokhal/m3_monokernel/tests/probes/down_run.py" 2>&1 | grep -v "^\[aiter\]\|queue-intercept" | grep "ok\|rror\|fault" | head -3

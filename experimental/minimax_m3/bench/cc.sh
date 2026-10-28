#!/bin/bash
# cc.sh "ENV=.." : compile-only check of the Gluon kernel (GPU 3, no launch, no lock needed)
E=$1
docker exec ix_dsv41 bash -c "cd /home/vigokhal/m3_monokernel && export HIP_VISIBLE_DEVICES=4 TRITON_CACHE_DIR=/home/vigokhal/m3_bench/cache/triton AITER_JIT_DIR=/home/vigokhal/m3_bench/cache/aiter_jit $E; timeout 900 python3 tests/gluon_compile2.py 2>&1" | grep -v "aiter\] import\|Warning" | tail -25

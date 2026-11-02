#!/bin/bash
set -euo pipefail
root=/home/vigokhal/dsv4-monokernel/flash
while [ ! -f "$root/candidate.status" ]; do sleep 10; done
[ "$(cat "$root/candidate.status")" = 0 ]
docker stop -t 3 vigokhal_dsv4_tp4
docker start vigokhal_dsv4_tp4
docker exec vigokhal_dsv4_tp4 rocm-smi --showpids > "$root/pre-validation-processes.txt"
grep -q 'No KFD PIDs currently running' "$root/pre-validation-processes.txt"
docker exec -d -w /work/flash -e HIP_VISIBLE_DEVICES=0 -e ATOM_DSV4_HYBRID=1 -e ATOM_DSV4_HYBRID_GEMM=0 -e ATOM_DSV4_HYBRID_ATTN=1 -e ATOM_DSV4_HYBRID_ATTN_CONFIG=/app/ATOM/atom/model_ops/dsv4_flash_attention_configs.json -e ATOM_DSV4_HYBRID_CHECK=1 -e ATOM_DSV4_HYBRID_HC_CHECK_LIMIT=100000 -e ATOM_DSV4_HYBRID_ATTN_CHECK_LIMIT=100000 -e FAKE_EPLB=0 -e TRACE=0 -e TRITON_CACHE_DIR=/work/flash/triton-cache vigokhal_dsv4_tp4 bash -c 'ulimit -c 0; bash /work/flash/serve_validation.sh > /work/flash/validation-launch.log 2>&1'
docker exec -w /work/flash vigokhal_dsv4_tp4 bash -c '
set -euo pipefail
trap '\''echo $? > /work/flash/real-validation.status'\'' EXIT
for i in $(seq 1 180); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 10
done
python /work/flash/validate_real_batches.py eager_validation
' > "$root/validation-client.log" 2>&1

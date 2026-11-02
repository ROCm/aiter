#!/bin/bash
set -euo pipefail
root=/home/vigokhal/dsv4-monokernel
while [ ! -f "$root/round2/candidate-repeats.status" ]; do sleep 10; done
[ "$(cat "$root/round2/candidate-repeats.status")" = 0 ]
sleep 3
docker stop -t 3 vigokhal_dsv4_tp4
docker start vigokhal_dsv4_tp4
if docker exec -w /work -e ENABLE_CK=0 -e TRITON_CACHE_DIR=/work/round2/triton-cache vigokhal_dsv4_tp4 bash -c 'python /work/round2/semaphore_probe_scoped_batched.py > /work/round2/semaphore_probe_scoped_batched.log 2>&1'; then
 echo 0 > "$root/round2/semaphore-batched.status"
else
 echo 1 > "$root/round2/semaphore-batched.status"
fi
docker exec -d -w /work -e ATOM_DSV4_HYBRID=1 -e ATOM_DSV4_HYBRID_GEMM=1 -e ATOM_DSV4_HYBRID_ATTN=1 -e ATOM_DSV4_HYBRID_CHECK=1 -e ATOM_DSV4_HYBRID_ATTN_CHECK_LIMIT=100000 -e ATOM_FORCE_ATTN_TRITON=1 -e FAKE_EPLB=0 -e TRACE=0 -e TRITON_CACHE_DIR=/work/triton-cache vigokhal_dsv4_tp4 bash -c 'bash /work/round2/serve_validation.sh > /work/round2/validation-launch.log 2>&1'
docker exec -w /work vigokhal_dsv4_tp4 bash -c '
set -euo pipefail
trap '\''echo $? > /work/round2/real-validation.status'\'' EXIT
for i in $(seq 1 180); do
 if curl -sf --max-time 2 http://localhost:8000/health >/dev/null; then break; fi
 sleep 10
done
python /work/round2/validate_real_batches.py round2_eager_validation
' > "$root/round2/validation-client.log" 2>&1

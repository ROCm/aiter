#!/usr/bin/env bash
set -euo pipefail
container=vigokhal_dsv4_tp4
host_root=/home/vigokhal/dsv4-monokernel/tp4-c16-rerun-20261019
root=/work/tp4-c16-rerun-20261019
mkdir -p "$host_root/results" "$host_root/logs"
for mode in baseline hybrid; do
 case "$mode" in baseline) enabled=0;; hybrid) enabled=1;; esac
 docker stop -t 3 "$container" >/dev/null
 docker start "$container" >/dev/null
 docker exec "$container" rocm-smi --showpids > "$host_root/$mode-gpus-before-server.txt"
 grep -q 'No KFD PIDs currently running' "$host_root/$mode-gpus-before-server.txt"
 docker exec "$container" bash -c 'if [ -d /root/.cache/atom ]; then mv /root/.cache/atom /root/.cache/atom-before-c16-rerun-$(date +%s%N); fi'
 docker exec -d -w /work -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -e PYTHONPATH=/opt/atom-gfx1250-integration -e AITER_LOG_LEVEL=WARNING \
  -e ATOM_DSV4_MONO="$enabled" -e ATOM_MONO_ENABLE=1 -e ATOM_MONO_CHECK=0 \
  -e ATOM_DSV4_FLASH_MOE=0 -e ATOM_DSV4_HYBRID=0 -e ATOM_DSV4_HYBRID_ATTN=0 \
  -e TRACE=0 -e TRITON_CACHE_DIR=/work/flash/triton-cache \
  -e TORCHINDUCTOR_CACHE_DIR="$root/inductor-$mode" \
  "$container" bash -c 'mode=$1; exec bash /app/scripts/serve_dsv4_mono_gfx1250.sh pro > "/work/tp4-c16-rerun-20261019/$mode-server.log" 2>&1' bash "$mode"
 echo "Started $mode server once" | tee "$host_root/status.txt"
 ready=0
 for attempt in $(seq 1 900); do
  if grep -q 'Server started successfully and ready to accept requests' "$host_root/$mode-server.log" && curl -sf http://127.0.0.1:8000/v1/models | grep -q DeepSeek-V4-Pro; then ready=1;break;fi
  if grep -Eq 'EXECUTOR_FAILED|ProcessExitedException|EngineCore failed' "$host_root/$mode-server.log";then tail -70 "$host_root/$mode-server.log";exit 1;fi
  sleep 1
 done
 [[ "$ready" == 1 ]]
 docker exec "$container" rocm-smi --showmemuse > "$host_root/$mode-memory-ready.txt"
 docker exec "$container" python "$root/server_identity.py" > "$host_root/$mode-server-before-client1.json"
 for client in 1 2; do
  echo "$mode client $client starting (same server)" | tee "$host_root/status.txt"
  docker exec "$container" rocm-smi --showpids > "$host_root/$mode-gpus-before-client$client.txt"
  docker exec -w /work -e HIP_VISIBLE_DEVICES=0,1,2,3 -e PYTHONPATH=/opt/atom-gfx1250-integration \
   "$container" python -m atom.benchmarks.benchmark_serving \
   --model=/data/DeepSeek-V4-Pro --backend=vllm --base-url=http://localhost:8000 \
   --dataset-name=random --random-input-len=1024 --random-output-len=1024 \
   --random-range-ratio=1 --num-prompts=64 --max-concurrency=16 \
   --request-rate=inf --ignore-eos --seed=0 --percentile-metrics=ttft,tpot,itl,e2el \
   --save-result --result-dir="$root/results" --result-filename="${mode}_client${client}_c16.json" \
   > "$host_root/logs/${mode}_client${client}_c16.log" 2>&1
  docker exec "$container" python "$root/server_identity.py" > "$host_root/$mode-server-after-client$client.json"
 done
 python3 - "$host_root" "$mode" <<'PY'
import json,sys
from pathlib import Path
r=Path(sys.argv[1]);m=sys.argv[2]
ids=[json.loads((r/f'{m}-server-{x}.json').read_text()) for x in ['before-client1','after-client1','after-client2']]
assert len({(x['pid'],x['start_ticks']) for x in ids})==1,'Server identity changed between clients'
for c in (1,2):
 x=json.loads((r/f'results/{m}_client{c}_c16.json').read_text());assert x['completed']==64,x
print(m,'both clients complete; server PID/start-time unchanged')
PY
 echo "$mode complete; report client2 only" | tee "$host_root/status.txt"
done
docker stop -t 3 "$container" >/dev/null
echo 'Both configurations completed; server stopped; report only client2' | tee "$host_root/status.txt"

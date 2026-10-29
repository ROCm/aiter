#!/usr/bin/env bash
# GPU run in the patched-Triton image on the given rocm-smi cards (comma list),
# mapped to HIP indices by PCI bus (HIP order differs from rocm-smi's on this
# box). Upstream's DS V4.1 mono tree at the commit the port follows (8321d11)
# is mounted over the image's older copy.
# Usage: CARDS=0,1,2,3 ./gpu_run.sh torchrun --nproc-per-node=4 -m deepseek_v4_gluon.tests.gpu_ffn_vs_flydsl 1 16 48
set -euo pipefail
IMG=${IMG:-vigokhal/vllm-nightly-tritonfix:00b6c4c5-v2}
CARDS=${CARDS:?rocm-smi card indices}
# a vLLM checkout at 8321d11: its vllm/models/deepseek_v41/amd/mono
UPSTREAM=${UPSTREAM:-/tmp/vllm-src/vllm/vllm/models/deepseek_v41/amd/mono}
X=$(cd "$(dirname "$0")/.." && pwd)  # the experimental/ dir
mkdir -p "$HOME/dsv4-gluon-cache/triton"
buses=$(rocm-smi --showbus --json 2>/dev/null | sed -n '/^{/,$p' | python3 -c '
import json, sys
d = json.load(sys.stdin)
print(",".join(d[f"card{c}"]["PCI Bus"].split(":")[1].lower() for c in sys.argv[1].split(",")))' "$CARDS")
VP=/usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v41/amd/mono
exec docker run --rm --name "vigokhal-dsv4g-$$" \
  --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined \
  -e PYTHONPATH=/x -e TRITON_CACHE_DIR=/cache/triton -e BUSES="$buses" -e BENCH="${BENCH:-}" -e DUMP="${DUMP:-}" -e RATIO="${RATIO:-1}" \
  -v "$X":/x:ro -v "$HOME/dsv4-gluon-cache":/cache \
  -v "$UPSTREAM":"$VP":ro \
  -v /data/DeepSeek-V4.1-Flash:/data/DeepSeek-V4.1-Flash:ro \
  -w /x --entrypoint bash "$IMG" -c '
hip=$(python3 -c "
import os, torch
want = os.environ[\"BUSES\"].split(\",\")
got = {f\"{torch.cuda.get_device_properties(i).pci_bus_id:02x}\": i for i in range(torch.cuda.device_count())}
print(\",\".join(str(got[b]) for b in want))")
echo "cards '"$CARDS"' -> PCI $BUSES -> HIP $hip"
HIP_VISIBLE_DEVICES=$hip exec "$@"' bash "$@"

#!/usr/bin/env bash
set -euo pipefail

# This script is intended to run as a multi-node Spur/sbatch job with one task
# per node. It avoids Slurm-only helpers such as "scontrol show hostnames" and
# allocation-internal "srun --nodelist".

require_env() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "ERROR: ${name} is required" >&2
    exit 2
  fi
}

require_env AITER_CI_CELL_ID
require_env DOCKER_IMAGE
require_env MODEL_NAME
require_env MODEL_PATH
require_env LOG_ROOT

SPUR_JOB_ID="${SPUR_JOB_ID:-${SLURM_JOB_ID:-local}}"
NODE_RANK="${NODE_RANK:-${SPUR_NODE_RANK:-${SPUR_TASK_OFFSET:-${SLURM_NODEID:-0}}}}"
xP="${xP:-${PREFILL_WORKERS:-1}}"
yD="${yD:-${DECODE_WORKERS:-1}}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-${PREFILL_TP:-8}}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-${DECODE_TP:-8}}"

SCRIPT_ROOT="${AITER_CI_SCRIPTS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SERVER_SCRIPT="${AITER_CI_SERVER_SCRIPT:-${SCRIPT_ROOT}/pd_server_atom.sh}"
if [[ ! -f "${SERVER_SCRIPT}" ]]; then
  echo "ERROR: pd_server_atom.sh not found at ${SERVER_SCRIPT}" >&2
  echo "Set AITER_CI_SCRIPTS_ROOT to a shared path visible on Spur compute nodes." >&2
  exit 2
fi

derive_ipaddrs() {
  python3 - <<'PY'
import os
import socket

def host_from_peer(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if value.startswith("[") and "]" in value:
        return value[1:value.index("]")]
    if value.count(":") == 1:
        return value.rsplit(":", 1)[0]
    return value

raw = os.environ.get("AITER_CI_IPADDRS") or os.environ.get("IPADDRS") or ""
items = [x.strip() for x in raw.split(",") if x.strip()]
if not items:
    peers = os.environ.get("SPUR_PEER_NODES") or ""
    items = [host_from_peer(x) for x in peers.split(",") if host_from_peer(x)]
if not items:
    try:
        items = [socket.gethostbyname(socket.gethostname())]
    except OSError:
        items = ["127.0.0.1"]
print(",".join(items))
PY
}

IPADDRS="$(derive_ipaddrs)"
if [[ -z "${IPADDRS}" ]]; then
  echo "ERROR: failed to derive IPADDRS. Set AITER_CI_IPADDRS manually." >&2
  exit 2
fi
NODE0_ADDR="${NODE0_ADDR:-${IPADDRS%%,*}}"

IFS=',' read -r -a IP_ARRAY <<< "${IPADDRS}"
expected_nodes=$((xP + yD))
if [[ "${#IP_ARRAY[@]}" -lt "${expected_nodes}" ]]; then
  echo "ERROR: IPADDRS has ${#IP_ARRAY[@]} entries, expected at least ${expected_nodes}: ${IPADDRS}" >&2
  exit 2
fi

RUN_DIR_HOST="${LOG_ROOT%/}/spur_job-${SPUR_JOB_ID}"
RANK_DIR="${RUN_DIR_HOST}/rank-${NODE_RANK}"
mkdir -p "${RANK_DIR}" "${RUN_DIR_HOST}/logs" "${RUN_DIR_HOST}/benchmark_results" "${RUN_DIR_HOST}/eval_results"
chmod a+rwx "${RUN_DIR_HOST}" "${RANK_DIR}" "${RUN_DIR_HOST}/logs" "${RUN_DIR_HOST}/benchmark_results" "${RUN_DIR_HOST}/eval_results" 2>/dev/null || true

ENV_FILE="${RANK_DIR}/docker.env"
python3 - <<'PY' > "${ENV_FILE}"
import os

allow = (
    "AITER_CI_",
    "MODEL_",
    "BACKEND",
    "PRECISION",
    "TOPOLOGY",
    "DISPLAY_TOPOLOGY",
    "ISL_LIST",
    "OSL",
    "CONC_LIST",
    "BENCH_",
    "RANDOM_RANGE_RATIO",
    "REQUEST_RATE",
    "WAIT_",
    "PREFILL_",
    "DECODE_",
    "ROUTER_",
    "PROMETHEUS_PORT",
    "KV_CACHE_DTYPE",
    "BLOCK_SIZE",
    "MEM_FRACTION",
    "MAX_NUM_SEQS",
    "EXTRA_SERVER_ARGS",
    "RUN_EVAL",
    "RUN_BENCHMARK",
    "EVAL_",
    "AITER_",
    "GPU_MAX_HW_QUEUES",
    "ATOM_",
    "NCCL_",
    "RCCL_",
    "HIP_",
    "HSA_",
    "ROCR_",
    "UCX_",
    "MOONCAKE_",
)
for key, value in sorted(os.environ.items()):
    if key.startswith(allow):
        print(f"{key}={value}")
PY

container="aiter-ci-${AITER_CI_CELL_ID}-${SPUR_JOB_ID}-${NODE_RANK}"

pre_cleanup_local() {
  [[ "${AITER_CI_PRE_CLEANUP:-1}" == "1" ]] || return 0
  local prefix="aiter-ci-${AITER_CI_CELL_ID}-"
  local ids=()
  while read -r id name; do
    [[ -n "${id}" && "${name}" == "${prefix}"* ]] && ids+=("${id}")
  done < <(docker ps -a --format '{{.ID}} {{.Names}}' 2>/dev/null || true)
  if [[ "${#ids[@]}" -gt 0 ]]; then
    echo "[cleanup] removing stale AITER CI containers for cell=${AITER_CI_CELL_ID}: ${#ids[@]}"
    docker rm -f "${ids[@]}" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  docker stop -t 0 "${container}" >/dev/null 2>&1 || true
  docker rm -f "${container}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "=== AITER CI Spur rank ==="
echo "cell=${AITER_CI_CELL_ID}"
echo "rank=${NODE_RANK}"
echo "host=$(hostname)"
echo "ipaddrs=${IPADDRS}"
echo "node0_addr=${NODE0_ADDR}"
echo "xP=${xP} yD=${yD}"
echo "run_dir=${RUN_DIR_HOST}"

pre_cleanup_local
if [[ "${AITER_CI_DOCKER_PULL:-0}" == "1" ]]; then
  docker pull "${DOCKER_IMAGE}"
fi

video_gid="$(getent group video | cut -d: -f3 || true)"
render_gid="$(getent group render | cut -d: -f3 || true)"
host_ionic="$(readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1 2>/dev/null || true)"
user_name="$(id -un)"

mount_args=(
  -v "${SCRIPT_ROOT}:/workspace/aiter_ci:ro"
  -v "${RUN_DIR_HOST}:/run_logs/spur_job-${SPUR_JOB_ID}"
  -v /mnt:/mnt
  -v /data:/data
)
[[ -d /it-share ]] && mount_args+=(-v /it-share:/it-share)
if [[ -n "${ATOM_REPO_ROOT:-}" && -d "${ATOM_REPO_ROOT}" ]]; then
  mount_args+=(-v "${ATOM_REPO_ROOT}:/workspace/ATOM:ro")
fi
[[ -n "${host_ionic}" && -e "${host_ionic}" ]] && mount_args+=(-v "${host_ionic}:/usr/lib/x86_64-linux-gnu/libionic.so.1:ro")
[[ -e /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so ]] && mount_args+=(-v /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:/usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav34.so:ro)
[[ -e /etc/libibverbs.d/ionic.driver ]] && mount_args+=(-v /etc/libibverbs.d/ionic.driver:/etc/libibverbs.d/ionic.driver:ro)

docker_args=(
  run --rm --name "${container}"
  --network host --ipc host --privileged
  --device /dev/kfd --device /dev/dri --device /dev/infiniband
  --user "$(id -u):$(id -g)"
  --cap-add IPC_LOCK --cap-add NET_ADMIN
  --ulimit memlock=-1 --ulimit stack=67108864 --ulimit nofile=65536:524288
  --shm-size 128G
  --env-file "${ENV_FILE}"
  -e SPUR_JOB_ID="${SPUR_JOB_ID}"
  -e SLURM_JOB_ID="${SPUR_JOB_ID}"
  -e NODE_RANK="${NODE_RANK}"
  -e NODE0_ADDR="${NODE0_ADDR}"
  -e IPADDRS="${IPADDRS}"
  -e xP="${xP}"
  -e yD="${yD}"
  -e PREFILL_TP_SIZE="${PREFILL_TP_SIZE}"
  -e DECODE_TP_SIZE="${DECODE_TP_SIZE}"
  -e RUN_DIR="/run_logs/spur_job-${SPUR_JOB_ID}"
  -e USER="${user_name}"
  -e LOGNAME="${user_name}"
  -e HOME="/tmp/aiter-ci-home-${SPUR_JOB_ID}-${NODE_RANK}"
  -e XDG_CACHE_HOME="/tmp/aiter-ci-cache-${SPUR_JOB_ID}-${NODE_RANK}"
  -e TORCHINDUCTOR_CACHE_DIR="/tmp/aiter-ci-cache-${SPUR_JOB_ID}-${NODE_RANK}/torchinductor"
  -e AITER_CACHE_DIR="/tmp/aiter-ci-cache-${SPUR_JOB_ID}-${NODE_RANK}/aiter"
  -e AITER_JIT_DIR="/tmp/aiter-ci-cache-${SPUR_JOB_ID}-${NODE_RANK}/aiter/jit"
  -e FLYDSL_RUNTIME_CACHE_DIR="/tmp/aiter-ci-cache-${SPUR_JOB_ID}-${NODE_RANK}/flydsl"
)
[[ -n "${video_gid}" ]] && docker_args+=(--group-add "${video_gid}")
[[ -n "${render_gid}" ]] && docker_args+=(--group-add "${render_gid}")
docker_args+=(
  "${mount_args[@]}"
  "${DOCKER_IMAGE}"
  bash -lc 'if [[ -d /workspace/ATOM ]]; then cd /workspace/ATOM; fi; bash /workspace/aiter_ci/pd_server_atom.sh'
)

docker "${docker_args[@]}" 2>&1 | tee "${RANK_DIR}/container.log"

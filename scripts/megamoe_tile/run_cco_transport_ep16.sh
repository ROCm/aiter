#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Two-node x eight-process launcher for the AITER-local CCO EP16 smoke.
# Safe by default: DRY_RUN=1 performs only read-only preflight/hash checks.

set -euo pipefail

SSH_CONFIG="${SSH_CONFIG:-/home/zihuang/work/.ssh/config}"
HOST0="${HOST0:-mi355-gpu-46}"
HOST1="${HOST1:-mi355-gpu-50}"
CONTAINER="${CONTAINER:-hzm_work}"
TEST_REL="${TEST_REL:-op_tests/multigpu_tests/test_megamoe_tile_cco_ep16_transport.py}"

MASTER_ADDR="${MASTER_ADDR:-10.2.80.17}"
MASTER_PORT="${MASTER_PORT:-29561}"
IFACE="${MORI_SOCKET_IFNAME:-enp193s0f1np1}"
DEVICE_NIC="${MORI_DEVICE_NIC:-ionic}"
RDMA_DEVICES="${MORI_RDMA_DEVICES:-rocep121s0}"
GID_INDEX="${MORI_IB_GID_INDEX:-1}"

DRY_RUN="${DRY_RUN:-1}"
TIMEOUT_SECS="${TIMEOUT_SECS:-1200}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="${LOG_DIR:-/tmp/megamoe_cco_ep16_${RUN_ID}}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-/home/hzm/cco_runs/${RUN_ID}}"
PASS_REGEX="${EP16_PASS_REGEX:-MEGAMOE_CCO_EP16_PASS}"
TEST_ARGS="${EP16_TEST_ARGS:-}"

# Optional test geometry, forwarded as environment variables. The EP16 test can
# ignore fields it does not use.
QP="${MEGAMOE_CCO_QP:-4}"
BATCH="${MEGAMOE_CCO_BATCH:-2}"
CHUNK="${MEGAMOE_CCO_CHUNK:-1024}"
EPOCHS="${MEGAMOE_CCO_EPOCHS:-2}"
TEAM="${MEGAMOE_CCO_TEAM:-rail}"

mkdir -p "${LOG_DIR}"

ssh_host() {
  local host="$1"
  shift
  ssh -F "${SSH_CONFIG}" \
    -o ControlMaster=no -o ControlPath=none -o ControlPersist=no \
    "${host}" "$@"
}

remote_preflight() {
  local host="$1"
  ssh_host "${host}" "podman exec \
    -e MORI_DEVICE_NIC=${DEVICE_NIC} \
    -e MORI_RDMA_DEVICES=${RDMA_DEVICES} \
    -e MORI_IB_GID_INDEX=${GID_INDEX} \
    ${CONTAINER} bash -lc '
    set -e
    echo HOST=\$(hostname)
    test -f /home/hzm/aiter/${TEST_REL}
    test \"\$(python -c \"import torch; print(torch.cuda.device_count())\")\" = 8
    cd /home/hzm/aiter
    echo AITER_HEAD=\$(git rev-parse HEAD)
    cd /home/hzm/mori
    echo MORI_HEAD=\$(git rev-parse HEAD)
    python -c \"import mori.cco; from mori.jit.config import get_mori_source_root,detect_nic_type,detect_build_config; assert detect_nic_type() == '${DEVICE_NIC}'; print('MORI=',mori.__version__); print('JIT_ROOT=',get_mori_source_root()); print('NIC=',detect_nic_type()); print('BUILD=',detect_build_config())\"
    sha256sum \
      /opt/venv/lib/python3.12/site-packages/mori/libmori_cco.so \
      /opt/venv/lib/python3.12/site-packages/mori/_jit-sources/include/mori/cco/cco.hpp \
      /opt/venv/lib/python3.12/site-packages/mori/_jit-sources/include/mori/cco/cco_scale_out.hpp \
      /home/hzm/aiter/${TEST_REL} \
      /home/hzm/aiter/aiter/ops/flydsl/kernels/megamoe_tile/cco/cco_device_bridge.cpp
    ip -br addr show ${IFACE}
    rdma link | grep -E \"^link ${RDMA_DEVICES}/1 state ACTIVE physical_state LINK_UP\"
  '"
}

echo "[preflight] EP16 host0=${HOST0} host1=${HOST1} master=${MASTER_ADDR}:${MASTER_PORT}"
remote_preflight "${HOST0}" | tee "${LOG_DIR}/preflight_${HOST0}.log" &
p0=$!
remote_preflight "${HOST1}" | tee "${LOG_DIR}/preflight_${HOST1}.log" &
p1=$!
rc=0
wait "${p0}" || rc=1
wait "${p1}" || rc=1
if (( rc != 0 )); then
  echo "[blocked] EP16 preflight failed" >&2
  exit 2
fi

for revision in AITER_HEAD MORI_HEAD; do
  r0="$(grep -F "${revision}=" "${LOG_DIR}/preflight_${HOST0}.log" | tail -n 1 | cut -d= -f2-)"
  r1="$(grep -F "${revision}=" "${LOG_DIR}/preflight_${HOST1}.log" | tail -n 1 | cut -d= -f2-)"
  if [[ -z "${r0}" || "${r0}" != "${r1}" ]]; then
    echo "[blocked] ${revision} mismatch: ${r0:-missing} != ${r1:-missing}" >&2
    exit 3
  fi
  echo "[preflight] ${revision}=${r0}"
done

hash_for() {
  local pattern="$1"
  local log="$2"
  grep -F "${pattern}" "${log}" | awk '{print $1}'
}

for artifact in \
  "${TEST_REL##*/}" \
  cco_device_bridge.cpp \
  libmori_cco.so \
  cco.hpp \
  cco_scale_out.hpp; do
  h0="$(hash_for "${artifact}" "${LOG_DIR}/preflight_${HOST0}.log")"
  h1="$(hash_for "${artifact}" "${LOG_DIR}/preflight_${HOST1}.log")"
  if [[ -z "${h0}" || "${h0}" != "${h1}" ]]; then
    echo "[blocked] ${artifact} hash mismatch: ${h0:-missing} != ${h1:-missing}" >&2
    exit 3
  fi
  echo "[preflight] ${artifact}=${h0}"
done

echo "[config] nic=${DEVICE_NIC} rdma=${RDMA_DEVICES} gid=${GID_INDEX} iface=${IFACE}"
echo "[config] qps=${QP} batch=${BATCH} chunk=${CHUNK} epochs=${EPOCHS} team=${TEAM}"
echo "[logs] ${LOG_DIR}"

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "[dry-run] no torchrun/CCO communicator started"
  echo "[dry-run] set DRY_RUN=0 after review to execute EP16"
  exit 0
fi

run_node() {
  local host="$1"
  local node_rank="$2"
  ssh_host "${host}" "podman exec \
    -e PYTHONPATH=/home/hzm/aiter \
    -e PYTHONUNBUFFERED=1 \
    -e OMP_NUM_THREADS=1 \
    -e TORCH_DISTRIBUTED_DEBUG=DETAIL \
    -e GLOO_SOCKET_IFNAME=${IFACE} \
    -e MORI_SOCKET_IFNAME=${IFACE} \
    -e MORI_DEVICE_NIC=${DEVICE_NIC} \
    -e MORI_RDMA_DEVICES=${RDMA_DEVICES} \
    -e MORI_IB_GID_INDEX=${GID_INDEX} \
    -e MEGAMOE_CCO_QP=${QP} \
    -e MEGAMOE_CCO_BATCH=${BATCH} \
    -e MEGAMOE_CCO_CHUNK=${CHUNK} \
    -e MEGAMOE_CCO_EPOCHS=${EPOCHS} \
    -e MEGAMOE_CCO_TEAM=${TEAM} \
    ${CONTAINER} bash -lc '
      cd /home/hzm/aiter
      mkdir -p ${REMOTE_RUN_DIR}/torchrun_node${node_rank}
      timeout --signal=TERM --kill-after=30s ${TIMEOUT_SECS}s \
        torchrun \
          --nnodes=2 \
          --node-rank=${node_rank} \
          --nproc-per-node=8 \
          --master-addr=${MASTER_ADDR} \
          --master-port=${MASTER_PORT} \
          --max-restarts=0 \
          --monitor-interval=5 \
          --log-dir=${REMOTE_RUN_DIR}/torchrun_node${node_rank} \
          --redirects=3 \
          --tee=3 \
          ${TEST_REL} ${TEST_ARGS}
    '"
}

echo "[launch] EP16 node 0 on ${HOST0}; node 1 on ${HOST1}"
run_node "${HOST0}" 0 >"${LOG_DIR}/node0_${HOST0}.log" 2>&1 &
n0=$!
sleep 1
run_node "${HOST1}" 1 >"${LOG_DIR}/node1_${HOST1}.log" 2>&1 &
n1=$!

rc0=0
rc1=0
wait "${n0}" || rc0=$?
wait "${n1}" || rc1=$?

cat "${LOG_DIR}/node0_${HOST0}.log"
cat "${LOG_DIR}/node1_${HOST1}.log"
echo "[result] node0_rc=${rc0} node1_rc=${rc1} logs=${LOG_DIR}"

if (( rc0 != rc1 )); then
  echo "[error] asymmetric node exits: ${rc0} != ${rc1}" >&2
  exit 4
fi
if (( rc0 != 0 )); then
  echo "[error] both nodes failed with rc=${rc0}" >&2
  exit 5
fi
if grep -q 'MEGAMOE_CCO_.*FAIL' "${LOG_DIR}"/node*.log; then
  echo "[error] EP16 test reported FAIL" >&2
  exit 6
fi
if ! grep -qE "${PASS_REGEX}" "${LOG_DIR}"/node*.log; then
  echo "[error] EP16 PASS marker not found: ${PASS_REGEX}" >&2
  exit 7
fi

echo "MEGAMOE_CCO_EP16_WORLD_PASS qps=${QP} batch=${BATCH} chunk=${CHUNK} epochs=${EPOCHS}"

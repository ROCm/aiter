#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Two-node launcher for the AITER-local CCO transport smoke.
#
# Safe by default: DRY_RUN=1 performs only read-only environment/hash checks.
# Set DRY_RUN=0 after the AITER test/bridge files have been synchronized to
# both hzm_work containers.

set -euo pipefail

SSH_CONFIG="${SSH_CONFIG:-/home/zihuang/work/.ssh/config}"
HOST0="${HOST0:-mi355-gpu-46}"
HOST1="${HOST1:-mi355-gpu-50}"
CONTAINER="${CONTAINER:-hzm_work}"
TEST_REL="${TEST_REL:-op_tests/multigpu_tests/test_megamoe_tile_cco_transport.py}"
IFACE="${MORI_SOCKET_IFNAME:-enp193s0f1np1}"
DEVICE_NIC="${MORI_DEVICE_NIC:-ionic}"
RDMA_DEVICES="${MORI_RDMA_DEVICES:-rocep121s0}"
GID_INDEX="${MORI_IB_GID_INDEX:-1}"
DRY_RUN="${DRY_RUN:-1}"
TIMEOUT_SECS="${TIMEOUT_SECS:-900}"

QP="${MEGAMOE_CCO_QP:-4}"
BATCH="${MEGAMOE_CCO_BATCH:-2}"
CHUNK="${MEGAMOE_CCO_CHUNK:-1024}"
GENERATION="${MEGAMOE_CCO_GENERATION:-7}"
GPU="${CCO_GPU:-0}"
H1_WORKERS="${MEGAMOE_CCO_H1_WORKERS:-240}"
H1_WARMUP="${MEGAMOE_CCO_H1_WARMUP:-0}"
H1_ITERS="${MEGAMOE_CCO_H1_ITERS:-1}"
PASS_MARKER="${MEGAMOE_CCO_PASS_MARKER:-MEGAMOE_CCO_TRANSPORT_PASS}"
FAIL_MARKER="${MEGAMOE_CCO_FAIL_MARKER:-MEGAMOE_CCO_TRANSPORT_FAIL}"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
REMOTE_RUN_DIR="${REMOTE_RUN_DIR:-/home/hzm/cco_runs/${RUN_ID}}"
UID_FILE="${REMOTE_RUN_DIR}/uid.bin"
LOG_DIR="${LOG_DIR:-/tmp/megamoe_cco_world2_${RUN_ID}}"
mkdir -p "${LOG_DIR}"

# Persisted ControlMaster sockets may belong to another execution context.
# Disable multiplexing on both hosts for deterministic non-interactive access.
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
    cd /home/hzm/aiter
    echo AITER_HEAD=\$(git rev-parse HEAD)
    cd /home/hzm/mori
    echo MORI_HEAD=\$(git rev-parse HEAD)
    python -c \"import mori.cco; from mori.jit.config import get_mori_source_root,detect_nic_type,detect_build_config; print(\\\"MORI=\\\",mori.__version__); print(\\\"JIT_ROOT=\\\",get_mori_source_root()); print(\\\"NIC=\\\",detect_nic_type()); print(\\\"BUILD=\\\",detect_build_config())\"
    sha256sum \
      /opt/venv/lib/python3.12/site-packages/mori/libmori_cco.so \
      /opt/venv/lib/python3.12/site-packages/mori/_jit-sources/include/mori/cco/cco.hpp \
      /opt/venv/lib/python3.12/site-packages/mori/_jit-sources/include/mori/cco/cco_scale_out.hpp \
      /home/hzm/aiter/${TEST_REL} \
      /home/hzm/aiter/aiter/ops/flydsl/kernels/megamoe_tile/cco/cco_device_bridge.cpp
    ip -br addr show ${IFACE}
    rdma link | grep -E "^link ${RDMA_DEVICES}/1 state ACTIVE physical_state LINK_UP"
  '"
}

echo "[preflight] host0=${HOST0} host1=${HOST1} container=${CONTAINER}"
remote_preflight "${HOST0}" | tee "${LOG_DIR}/preflight_${HOST0}.log" &
p0=$!
remote_preflight "${HOST1}" | tee "${LOG_DIR}/preflight_${HOST1}.log" &
p1=$!
rc=0
wait "${p0}" || rc=1
wait "${p1}" || rc=1
if (( rc != 0 )); then
  echo "[blocked] preflight failed; synchronize files/runtime before WORLD2" >&2
  exit 2
fi

test_hash_0="$(grep "${TEST_REL##*/}" "${LOG_DIR}/preflight_${HOST0}.log" | awk '{print $1}')"
test_hash_1="$(grep "${TEST_REL##*/}" "${LOG_DIR}/preflight_${HOST1}.log" | awk '{print $1}')"
bridge_hash_0="$(grep 'cco_device_bridge.cpp' "${LOG_DIR}/preflight_${HOST0}.log" | awk '{print $1}')"
bridge_hash_1="$(grep 'cco_device_bridge.cpp' "${LOG_DIR}/preflight_${HOST1}.log" | awk '{print $1}')"
if [[ -z "${test_hash_0}" || "${test_hash_0}" != "${test_hash_1}" || \
      -z "${bridge_hash_0}" || "${bridge_hash_0}" != "${bridge_hash_1}" ]]; then
  echo "[blocked] AITER test/bridge hashes differ across nodes" >&2
  exit 3
fi

echo "[preflight] synchronized test=${test_hash_0} bridge=${bridge_hash_0}"
echo "[config] nic=${DEVICE_NIC} rdma=${RDMA_DEVICES} gid=${GID_INDEX} qps=${QP} batch=${BATCH} chunk=${CHUNK} generation=${GENERATION} gpu=${GPU} h1_workers=${H1_WORKERS} h1_warmup=${H1_WARMUP} h1_iters=${H1_ITERS}"
echo "[markers] pass=${PASS_MARKER} fail=${FAIL_MARKER}"
echo "[logs] ${LOG_DIR}"

if [[ "${DRY_RUN}" != "0" ]]; then
  echo "[dry-run] no UID generated and no CCO communicator started"
  echo "[dry-run] set DRY_RUN=0 after review to execute WORLD2"
  exit 0
fi

# Generate the rendezvous token on rank 0 after MORI_SOCKET_IFNAME is set.
# ccoGetUniqueId encodes the rank-0 bootstrap address; Communicator.init later
# creates the actual socket bootstrap endpoint.
uid_hex="$(
  ssh_host "${HOST0}" \
    "podman exec -e MORI_SOCKET_IFNAME=${IFACE} -e MORI_DEVICE_NIC=${DEVICE_NIC} -e MORI_RDMA_DEVICES=${RDMA_DEVICES} -e MORI_IB_GID_INDEX=${GID_INDEX} ${CONTAINER} python -c 'from mori.cco import get_unique_id; print(bytes(get_unique_id()).hex())'" |
    tail -n 1 |
    tr -d '\r\n'
)"
if [[ "${#uid_hex}" -ne 256 ]]; then
  echo "[error] expected a 128-byte CCO UID, got ${#uid_hex} hex chars" >&2
  exit 4
fi

install_uid() {
  local host="$1"
  ssh_host "${host}" "podman exec -e CCO_UID_HEX=${uid_hex} -e CCO_UID_FILE=${UID_FILE} ${CONTAINER} python -c '
import os
from pathlib import Path
p = Path(os.environ[\"CCO_UID_FILE\"])
p.parent.mkdir(parents=True, exist_ok=True)
tmp = p.with_suffix(\".tmp\")
tmp.write_bytes(bytes.fromhex(os.environ[\"CCO_UID_HEX\"]))
os.replace(tmp, p)
'"
}

install_uid "${HOST0}"
install_uid "${HOST1}"

run_rank() {
  local host="$1"
  local rank="$2"
  ssh_host "${host}" "podman exec \
    -e PYTHONPATH=/home/hzm/aiter \
    -e PYTHONUNBUFFERED=1 \
    -e MORI_DEVICE_NIC=${DEVICE_NIC} \
    -e MORI_RDMA_DEVICES=${RDMA_DEVICES} \
    -e MORI_IB_GID_INDEX=${GID_INDEX} \
    -e MORI_SOCKET_IFNAME=${IFACE} \
    -e CCO_RANK=${rank} -e CCO_WORLD=2 -e CCO_UID_FILE=${UID_FILE} \
    -e CCO_GPU=${GPU} \
    -e MEGAMOE_CCO_QP=${QP} \
    -e MEGAMOE_CCO_BATCH=${BATCH} \
    -e MEGAMOE_CCO_CHUNK=${CHUNK} \
    -e MEGAMOE_CCO_GENERATION=${GENERATION} \
    -e MEGAMOE_CCO_H1_WORKERS=${H1_WORKERS} \
    -e MEGAMOE_CCO_H1_WARMUP=${H1_WARMUP} \
    -e MEGAMOE_CCO_H1_ITERS=${H1_ITERS} \
    ${CONTAINER} bash -lc '
      cd /home/hzm/aiter
      timeout --signal=TERM --kill-after=30s ${TIMEOUT_SECS}s \
        python ${TEST_REL}
    '"
}

echo "[launch] starting rank 0 on ${HOST0} and rank 1 on ${HOST1}"
run_rank "${HOST0}" 0 >"${LOG_DIR}/rank0_${HOST0}.log" 2>&1 &
r0=$!
# Give rank 0 a short head start to enter Communicator.init.
sleep 1
run_rank "${HOST1}" 1 >"${LOG_DIR}/rank1_${HOST1}.log" 2>&1 &
r1=$!

rc0=0
rc1=0
wait "${r0}" || rc0=$?
wait "${r1}" || rc1=$?

cat "${LOG_DIR}/rank0_${HOST0}.log"
cat "${LOG_DIR}/rank1_${HOST1}.log"
echo "[result] rank0_rc=${rc0} rank1_rc=${rc1} logs=${LOG_DIR}"

if (( rc0 != 0 || rc1 != 0 )); then
  exit 5
fi
for rank_log in \
  "${LOG_DIR}/rank0_${HOST0}.log" \
  "${LOG_DIR}/rank1_${HOST1}.log"; do
  if grep -q "${FAIL_MARKER}" "${rank_log}" || \
     ! grep -q "${PASS_MARKER}" "${rank_log}"; then
    echo "[error] both bidirectional ranks must report PASS: ${rank_log}" >&2
    exit 6
  fi
done

echo "MEGAMOE_CCO_WORLD2_PASS qps=${QP} batch=${BATCH} chunk=${CHUNK}"

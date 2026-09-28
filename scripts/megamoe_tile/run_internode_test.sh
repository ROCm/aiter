#!/usr/bin/env bash
# Launch test_megamoe_tile_internode.py on one node of an EP16 (2 x 8 GPU) job.
#
#   run_internode_test.sh NODE_RANK MASTER_ADDR MASTER_PORT TAG -- [test args]
#
# Start it on both nodes with the same MASTER_ADDR/PORT/TAG.  The log goes to
# trace_data/internode/TAG/node<NODE_RANK>.log and the exit code to
# node<NODE_RANK>.status; results (rank 0) to trace_data/internode/TAG/results.json.
# Examples (test args):
#   --mode perf --network kimi_k3 --tpr-list 128,256,512,1024
#   --mode perf --network dsv4 --skip-fused --tpr-list 512
#   --mode func --network kimi_k3 --tpr-list 128,512 --fixtures eplb,permuted
#   --mode op   --network kimi_k3 --tpr-list 512 --part stage1
# ATT_WRAP=<wrapper script>: torchrun --no-python wrapper (e.g. one that runs
# rocprofv3 --att on a single rank); the test script path and args follow it.
set -euo pipefail
node_rank="${1:?node rank (0 or 1) required}"
master_addr="${2:?master address required}"
master_port="${3:?master port required}"
tag="${4:?run tag required}"
shift 4
[[ "${1:-}" == "--" ]] && shift
case "${node_rank}" in 0|1) ;; *) echo "node rank must be 0 or 1" >&2; exit 64 ;; esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
out_dir="${repo_root}/trace_data/internode/${tag}"
mkdir -p "${out_dir}"
if [[ -e "${out_dir}/node${node_rank}.status" ]]; then
  echo "run tag already exists: ${tag}" >&2
  exit 64
fi
rm -f "${out_dir}/node${node_rank}.log"

# FLYDSL_PYTHONPATH: optional pinned FlyDSL install to put ahead of site-packages.
export PYTHONPATH="${FLYDSL_PYTHONPATH:+${FLYDSL_PYTHONPATH}:}${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export GLOO_SOCKET_IFNAME="${SOCKET_IFNAME:-enp193s0f1np1}"
export MORI_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
export NCCL_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}"
export MORI_DEVICE_NIC="${MORI_DEVICE_NIC:-ionic}"
export MORI_RDMA_DEVICES="${MORI_RDMA_DEVICES:-^rocep193s0f0,rocep193s0f1}"
export MORI_IB_GID_INDEX="${MORI_IB_GID_INDEX:-1}"
export MORI_NUM_QP_PER_PE=2
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-40G}"
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export AITER_MOE_EXPERT_BALANCE=true
export AMD_SERIALIZE_KERNEL=0
export FLYDSL_RUNTIME_ENABLE_CACHE=1
export MEGAMOE_TILE_PROFILE_REGIONS=0
export MEGAMOE_MASTER_ADDR="${master_addr}"

set +e
timeout --signal=TERM --kill-after=30s "${MEGAMOE_RUN_TIMEOUT:-3600s}" \
  python3 -u -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="${node_rank}" \
  --master-addr="${master_addr}" --master-port="${master_port}" --max-restarts=0 \
  ${ATT_WRAP:+--no-python "${ATT_WRAP}"} \
  op_tests/multigpu_tests/test_megamoe_tile_internode.py \
  --out-dir trace_data/internode --tag "${tag}" "$@" >"${out_dir}/node${node_rank}.log" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" >"${out_dir}/node${node_rank}.status"
exit "${rc}"

#!/usr/bin/env bash
set -uo pipefail

node_rank="${1:?node rank required}"
master_port="${2:?master port required}"
tag="${3:?log tag required}"
node_reduce_vec_bytes="${4:-8}"
mode="${5:-route}"
node_reduce_load_schedule="${6:-interleaved}"
node_reduce_work_schedule="${7:-static_strided}"
node_reduce_rejoin_blocks="${8:-0}"
rank_epilogue_lds_addressing="${9:-expanded}"

case "${mode}" in
  route) stage2_mode_flag=--route-store-stage2 ;;
  rank) stage2_mode_flag=--rank-local-stage2 ;;
  *)
    printf 'mode must be route or rank, got %s\n' "${mode}" >&2
    exit 2
    ;;
esac

log_dir=/home/hzm/logs/megamoe_route_store_validation_20260826
log_file="${log_dir}/${tag}_stress_node${node_rank}.log"
status_file="${log_dir}/${tag}_stress_node${node_rank}.status"

mkdir -p "${log_dir}"
cd /home/hzm/aiter || exit 97
export PYTHONPATH=/home/hzm/aiter
export OMP_NUM_THREADS=1
export GLOO_SOCKET_IFNAME=enp193s0f1np1
export MORI_SOCKET_IFNAME=enp193s0f1np1
export NCCL_SOCKET_IFNAME=enp193s0f1np1
export MORI_DEVICE_NIC=ionic
export MORI_RDMA_DEVICES=rocep121s0
export MORI_IB_GID_INDEX=1
export MORI_NUM_QP_PER_PE=4
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-40G}"
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export AMD_SERIALIZE_KERNEL=0
export FLYDSL_RUNTIME_ENABLE_CACHE=0

timeout --signal=TERM --kill-after=30s 1800s \
  python3 -u -m torch.distributed.run \
    --nnodes=2 \
    --nproc-per-node=8 \
    --node-rank="${node_rank}" \
    --master-addr=10.2.80.17 \
    --master-port="${master_port}" \
    --max-restarts=0 \
    --monitor-interval=5 \
    op_tests/multigpu_tests/stress_megamoe_tile_ep16_sparse_routes.py \
    --counts '' \
    --hot-ranks '' \
    --single-expert-rank -1 \
    --stage2-expected-sweep \
    --poison-stage2 \
    "${stage2_mode_flag}" \
    --stage2-node-reduce-vec-bytes "${node_reduce_vec_bytes}" \
    --stage2-node-reduce-load-schedule "${node_reduce_load_schedule}" \
    --stage2-node-reduce-work-schedule "${node_reduce_work_schedule}" \
    --stage2-node-reduce-rejoin-blocks "${node_reduce_rejoin_blocks}" \
    --stage2-rank-epilogue-lds-addressing "${rank_epilogue_lds_addressing}" \
    >"${log_file}" 2>&1
rc=$?
printf '%s\n' "${rc}" >"${status_file}"
exit "${rc}"

#!/usr/bin/env bash
set -uo pipefail

node_rank="${1:?node rank required}"
master_port="${2:?master port required}"
mode="${3:?direct or route required}"
route_pattern="${4:?route pattern required}"
tag="${5:?log tag required}"
generations="${6:-4}"
node_reduce_vec_bytes="${7:-8}"
node_reduce_load_schedule="${8:-interleaved}"
node_reduce_work_schedule="${9:-static_strided}"
node_reduce_rejoin_blocks="${10:-0}"
rank_epilogue_lds_addressing="${11:-expanded}"
rank_accumulation_mode="${12:-atomic}"

if [[ "${mode}" != "direct" && "${mode}" != "route" && "${mode}" != "rank" ]]; then
  printf 'mode must be direct, route, or rank, got %s\n' "${mode}" >&2
  exit 2
fi
if [[ "${route_pattern}" != "paired-rank-half-remote" && \
      "${route_pattern}" != "permuted-arbitrary-topk" ]]; then
  printf 'unsupported route pattern: %s\n' "${route_pattern}" >&2
  exit 2
fi
if (( generations < 4 )); then
  printf 'generations must be at least 4, got %s\n' "${generations}" >&2
  exit 2
fi

log_dir=/home/hzm/logs/megamoe_route_store_validation_20260826
reference_dir="${log_dir}/${tag}_references"
log_file="${log_dir}/${tag}_${mode}_${route_pattern}_node${node_rank}.log"
status_file="${log_dir}/${tag}_${mode}_${route_pattern}_node${node_rank}.status"

mkdir -p "${log_dir}" "${reference_dir}"
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
# Reuse the generated FlyDSL/LLVM artifact across ranks and repeated
# validation runs.  Set FLYDSL_RUNTIME_ENABLE_CACHE=0 explicitly when a
# cache-bypass diagnostic is required.
export FLYDSL_RUNTIME_ENABLE_CACHE="${FLYDSL_RUNTIME_ENABLE_CACHE:-1}"

timeout --signal=TERM --kill-after=30s 1800s \
  python3 -u -m torch.distributed.run \
    --nnodes=2 \
    --nproc-per-node=8 \
    --node-rank="${node_rank}" \
    --master-addr=10.2.80.17 \
    --master-port="${master_port}" \
    --max-restarts=0 \
    --monitor-interval=5 \
    op_tests/multigpu_tests/validate_megamoe_tile_route_store_ep16.py \
    --mode "${mode}" \
    --route-pattern "${route_pattern}" \
    --generations "${generations}" \
    --node-reduce-vec-bytes "${node_reduce_vec_bytes}" \
    --node-reduce-load-schedule "${node_reduce_load_schedule}" \
    --node-reduce-work-schedule "${node_reduce_work_schedule}" \
    --node-reduce-rejoin-blocks "${node_reduce_rejoin_blocks}" \
    --rank-epilogue-lds-addressing "${rank_epilogue_lds_addressing}" \
    --rank-accumulation-mode "${rank_accumulation_mode}" \
    --rel-l2-threshold 0.05 \
    --output-dir "${reference_dir}" \
    >"${log_file}" 2>&1
rc=$?
printf '%s\n' "${rc}" >"${status_file}"
exit "${rc}"

#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -uo pipefail

node_rank="${1:?node rank required}"
master_port="${2:?master port required}"
tag="${3:?comparison tag required}"
warmup="${4:-20}"
iters="${5:-100}"
tail_iters="${6:-50}"
order="${7:-baseline-first}"
operator_factory="${8:-aiter.ops.flydsl.kernels.megamoe_tile:HierarchicalMegaMoEV2}"
direct_packed_weights="${9:-0}"

case_label=TPR128_TopK16_E896_H7168_I3072_EP16_A4W4
log_dir=/home/hzm/logs/megamoe_final_comparison
log_file="${log_dir}/${case_label}_${tag}_e2e_node${node_rank}.log"
status_file="${log_dir}/${case_label}_${tag}_e2e_node${node_rank}.status"

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
export MORI_SHMEM_HEAP_SIZE=40G
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export AMD_SERIALIZE_KERNEL=0
export FLYDSL_RUNTIME_ENABLE_CACHE=0

weight_args=()
if [[ "${direct_packed_weights}" == "1" ]]; then
  weight_args=(--direct-packed-weights)
fi

timeout --signal=TERM --kill-after=30s 1800s \
  python3 -u -m torch.distributed.run \
    --nnodes=2 \
    --nproc-per-node=8 \
    --node-rank="${node_rank}" \
    --master-addr=10.2.80.17 \
    --master-port="${master_port}" \
    --max-restarts=0 \
    --monitor-interval=5 \
    op_tests/multigpu_tests/bench_megamoe_tile_ep16_two_kernel.py \
    --paths both \
    --order "${order}" \
    --operator-factory "${operator_factory}" \
    --candidate-stage1-transport sparse_wqe \
    --route-pattern paired-rank-half-remote \
    "${weight_args[@]}" \
    --warmup "${warmup}" \
    --iters "${iters}" \
    --tail-iters "${tail_iters}" \
    --rel-l2-threshold 5e-2 \
    >"${log_file}" 2>&1
rc=$?
printf '%s\n' "${rc}" >"${status_file}"
exit "${rc}"

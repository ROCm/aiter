#!/usr/bin/env bash
set -euo pipefail

node_rank="${1:?node rank required}"
master_port="${2:?master port required}"
profile_mode="${3:-trace}"
target_cu="${4:-1}"
tag="${5:-ranklocal_baseline_r16_f4}"
shader_engine_mask="${6:-0x1}"
candidate_mode="${7:-full}"

cd /home/hzm/aiter
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
export FLYDSL_RUNTIME_ENABLE_CACHE="${FLYDSL_RUNTIME_ENABLE_CACHE:-1}"
# profile_rank0_worker.sh enables debug info only for PROFILE_GLOBAL_RANK.
unset FLYDSL_DEBUG_ENABLE_DEBUG_INFO

export PROFILE_GLOBAL_RANK=0
export PROFILE_MODE="${profile_mode}"
export PROFILE_ROOT="/home/hzm/profiles/${tag}_cu${target_cu}"
export KERNEL_RE='.*megamoe_tile_ep16_stage2.*narank_local.*ramatomic.*'
export ATT_TARGET_CU="${target_cu}"
export ATT_GPU_INDEX=0
export ATT_SHADER_ENGINE_MASK="${shader_engine_mask}"
export ATT_SIMD_SELECT=0xF
export ATT_ITERATION_RANGE="${ATT_ITERATION_RANGE:-1-1}"
export ATT_LIBRARY_PATH="${ATT_LIBRARY_PATH:-/home/hzm/rocprof-trace-decoder/releases/linux_glibc_2_28_x86_64}"
export MEGAMOE_TILE_PROFILE_REGIONS=1

python3 -u -m torch.distributed.run \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank="${node_rank}" \
  --master-addr=10.2.80.17 \
  --master-port="${master_port}" \
  --max-restarts=0 \
  --monitor-interval=5 \
  --no-python \
  scripts/megamoe_tile/profile_rank0_worker.sh \
  op_tests/multigpu_tests/bench_megamoe_tile_ep16_stage2_breakdown.py \
  --path candidate \
  --candidate-mode "${candidate_mode}" \
  --direct-packed-weights \
  --stage2-workers 176 \
  --candidate-accumulator bf16 \
  --candidate-final-combine-blocks 4 \
  --candidate-gmm-schedule persistent_queue \
  --candidate-return-chunk-tokens 8 \
  --candidate-bf16-atomic-kind buffer \
  --candidate-node-accumulation-mode rank_local \
  --candidate-node-reduce-blocks 16 \
  --candidate-node-reduce-vec-bytes 8 \
  --candidate-node-reduce-schedule token \
  --candidate-node-reduce-load-schedule load_first \
  --candidate-node-reduce-work-schedule dynamic_head \
  --candidate-node-reduce-rejoin-blocks 0 \
  --candidate-rank-epilogue-lds-addressing expanded \
  --candidate-rank-accumulation-mode atomic \
  --candidate-rail-return-schedule compact \
  --candidate-epilogue-schedule lane32_meta \
  --candidate-n-tile-group 2 \
  --candidate-group-pipeline-schedule a_double_buffer \
  --candidate-scoreboard-schedule wave0 \
  --candidate-atomic-issue-schedule interleaved \
  --candidate-waves-per-eu-hint 2 \
  --warmup 1 \
  --iters 3 \
  --tail-iters 2

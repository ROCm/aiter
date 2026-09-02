#!/usr/bin/env bash
set -uo pipefail

node_rank="${1:?node rank required}"
master_port="${2:?master port required}"
path="${3:?candidate or mori required}"
mode="${4:?breakdown mode required}"
tag="${5:?log tag required}"
warmup="${6:-10}"
iters="${7:-30}"
tail_iters="${8:-20}"
stage2_workers="${9:-160}"
candidate_accumulator="${10:-bf16}"
candidate_final_combine_blocks="${11:-14}"
candidate_gmm_schedule="${12:-persistent_queue}"
candidate_return_chunk_tokens="${13:-8}"
candidate_bf16_atomic_kind="${14:-buffer}"
device_timeline="${15:-0}"
candidate_rail_return_schedule="${16:-lockstep}"
direct_packed_weights="${17:-0}"
candidate_epilogue_schedule="${18:-lane32_meta}"
candidate_n_tile_group="${19:-2}"
candidate_scoreboard_schedule="${20:-wave0}"
candidate_atomic_issue_schedule="${21:-interleaved}"
candidate_waves_per_eu_hint="${22:-2}"
candidate_group_pipeline_schedule="${23:-a_double_buffer}"
candidate_node_accumulation_mode="${24:-direct_atomic}"
candidate_node_reduce_blocks="${25:-32}"
candidate_node_reduce_vec_bytes="${26:-4}"
candidate_node_reduce_schedule="${27:-token}"
candidate_node_reduce_load_schedule="${28:-interleaved}"
candidate_node_reduce_work_schedule="${29:-static_strided}"
candidate_node_reduce_rejoin_blocks="${30:-0}"
candidate_rank_epilogue_lds_addressing="${31:-expanded}"
candidate_rank_accumulation_mode="${32:-atomic}"
tokens="${33:-128}"
max_routes_per_token_per_rank="${34:-}"
stage1_diagnostic_phase="${35:-full}"
mori_combine_quant_type="${36:-none}"
candidate_rail_quant_type="${37:-none}"
candidate_gmm_work_swizzle="${38:-token_major}"
candidate_window_n_groups="${39:-2}"
candidate_ready_granularity="${40:-token}"

log_dir=/home/hzm/logs/megamoe_stage2_breakdown_20260824
log_file="${log_dir}/${tag}_node${node_rank}.log"
status_file="${log_dir}/${tag}_node${node_rank}.status"

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
# Performance runs should compile once and let the remaining ranks reuse the
# generated artifact.  Set this to 0 explicitly only for a cold-compile
# diagnostic; forcing it off here makes all 16 ranks compile independently.
export FLYDSL_RUNTIME_ENABLE_CACHE="${FLYDSL_RUNTIME_ENABLE_CACHE:-1}"

if [[ "${path}" == "candidate" ]]; then
  mode_args=(--candidate-mode "${mode}")
elif [[ "${path}" == "mori" ]]; then
  mode_args=(--mori-mode "${mode}")
else
  printf 'invalid path: %s\n' "${path}" >&2
  exit 2
fi
timeline_args=()
if [[ "${device_timeline}" == "1" ]]; then
  timeline_args=(--device-timeline)
fi
weight_args=()
if [[ "${direct_packed_weights}" == "1" ]]; then
  weight_args=(--direct-packed-weights)
fi
route_capacity_args=()
if [[ -n "${max_routes_per_token_per_rank}" ]]; then
  route_capacity_args=(
    --max-routes-per-token-per-rank "${max_routes_per_token_per_rank}"
  )
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
    op_tests/multigpu_tests/bench_megamoe_tile_ep16_stage2_breakdown.py \
    --path "${path}" \
    "${mode_args[@]}" \
    "${timeline_args[@]}" \
    "${weight_args[@]}" \
    "${route_capacity_args[@]}" \
    --stage1-diagnostic-phase "${stage1_diagnostic_phase}" \
    --mori-combine-quant-type "${mori_combine_quant_type}" \
    --tokens "${tokens}" \
    --stage2-workers "${stage2_workers}" \
    --candidate-accumulator "${candidate_accumulator}" \
    --candidate-final-combine-blocks "${candidate_final_combine_blocks}" \
    --candidate-gmm-schedule "${candidate_gmm_schedule}" \
    --candidate-return-chunk-tokens "${candidate_return_chunk_tokens}" \
    --candidate-bf16-atomic-kind "${candidate_bf16_atomic_kind}" \
    --candidate-node-accumulation-mode "${candidate_node_accumulation_mode}" \
    --candidate-node-reduce-blocks "${candidate_node_reduce_blocks}" \
    --candidate-node-reduce-vec-bytes "${candidate_node_reduce_vec_bytes}" \
    --candidate-node-reduce-schedule "${candidate_node_reduce_schedule}" \
    --candidate-node-reduce-load-schedule "${candidate_node_reduce_load_schedule}" \
    --candidate-node-reduce-work-schedule "${candidate_node_reduce_work_schedule}" \
    --candidate-node-reduce-rejoin-blocks "${candidate_node_reduce_rejoin_blocks}" \
    --candidate-rank-epilogue-lds-addressing "${candidate_rank_epilogue_lds_addressing}" \
    --candidate-rank-accumulation-mode "${candidate_rank_accumulation_mode}" \
    --candidate-rail-return-schedule "${candidate_rail_return_schedule}" \
    --candidate-rail-quant-type "${candidate_rail_quant_type}" \
    --candidate-gmm-work-swizzle "${candidate_gmm_work_swizzle}" \
    --candidate-window-n-groups "${candidate_window_n_groups}" \
    --candidate-ready-granularity "${candidate_ready_granularity}" \
    --candidate-epilogue-schedule "${candidate_epilogue_schedule}" \
    --candidate-n-tile-group "${candidate_n_tile_group}" \
    --candidate-group-pipeline-schedule "${candidate_group_pipeline_schedule}" \
    --candidate-scoreboard-schedule "${candidate_scoreboard_schedule}" \
    --candidate-atomic-issue-schedule "${candidate_atomic_issue_schedule}" \
    --candidate-waves-per-eu-hint "${candidate_waves_per_eu_hint}" \
    --warmup "${warmup}" \
    --iters "${iters}" \
    --tail-iters "${tail_iters}" \
    >"${log_file}" 2>&1
rc=$?
printf '%s\n' "${rc}" >"${status_file}"
exit "${rc}"

#!/usr/bin/env bash
# One shape per process: small-token tune lookup must use that shape's capacity.
# Design/parameter rationale: scripts/megamoe_tile/FUSED_STAGE2_DESIGN_20260909.md
set -euo pipefail
node_rank="${1:?node rank required}"
master_port="${2:?unique master port required}"
bench_path="${3:?candidate or mori required}"
tag="${4:?unique run tag required}"
shift 4
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
run_dir="${repo_root}/trace_data/stage2_graph_20260909/${tag}/node${node_rank}"
mkdir -p "${run_dir}"
if [[ -e "${run_dir}/run.log" || -e "${run_dir}/run.status" ]]; then
  printf 'Run tag already exists: %s\n' "${tag}" >&2
  exit 64
fi
if [[ -f stage2_graph_overlay_sha256.json ]]; then
  cp stage2_graph_overlay_sha256.json "${run_dir}/source_manifest.json"
fi
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
# One host thread per rank keeps 16 launch processes from oversubscribing CPUs.
export OMP_NUM_THREADS=1
# Use the shared data interface for rendezvous and all transport libraries;
# successful torchrun rendezvous alone does not select Gloo's advertised NIC.
export GLOO_SOCKET_IFNAME=enp193s0f1np1
export MORI_SOCKET_IFNAME=enp193s0f1np1
export NCCL_SOCKET_IFNAME=enp193s0f1np1
export MORI_DEVICE_NIC=ionic
# Exclude the two control/BNXT devices so MORI sees all eight Ionic rails.
export MORI_RDMA_DEVICES='^rocep193s0f0,rocep193s0f1'
export MORI_IB_GID_INDEX=1
# MORI InterNodeV1LL baseline settings verified on 46/50. These 2 QPs are
# separate from fused CCO's Stage1/Stage2 arena, which requires 4 QPs.
# 40G is the current runtime heap budget, not a Stage2 tensor-size formula.
export MORI_NUM_QP_PER_PE=2
export MORI_SHMEM_HEAP_SIZE=40G
export MORI_EP_LAUNCH_CONFIG_MODE=AUTO
export AITER_MOE_EXPERT_BALANCE=true
# Kimi K3 comparison uses SiTUv2 A4W4; both paths use beta=linear_beta=1.
# The task-specific tune table preserves the reference small-token selection.
export AITER_SITUV2_A4W4=1
export AITER_SITUV2_A8W4=0
export AITER_CONFIG_FMOE="${AITER_CONFIG_FMOE:-${repo_root}/scripts/megamoe_tile/kimik3_stage2_graph_tune.csv}"
export AMD_SERIALIZE_KERNEL=0
export FLYDSL_RUNTIME_ENABLE_CACHE=1
export MEGAMOE_TILE_PROFILE_REGIONS=0
# Full forward per replay, warmup10/profile40, exact iterations20..39. All
# ranks replay without per-iteration barriers; GPU traces attribute Stage2.
# The 2026-09-10 EPLB comparison uses the shortest complete Stage2 span across
# 16 ranks x tail20 replays, as requested. Both paths use the same statistic;
# the report still retains per-rank minima and means for later inspection.
# H3584/I3072/E896/EP16/TopK16 is the current comparison shape; the legacy
# driver's H7168/SiLU defaults describe a different workload.
# cross_node has one route/rank/token, so capacity=1 suffices. Override it to
# 2 for paired routes or 4 for the arbitrary fixture; it is not the TopK value.
# This capacity is a fixture setting, not an EPLB prerequisite of the kernel.
# General-route correctness gates must pass before accepting an optimization.
# Performance tuning is cross_node EPLB only. The Graph driver hashes every
# byte of each rank's inputs/packed weights outside capture/timing and records
# actual route counts plus BM32 padding. compare_stage2_graph.py refuses A/B
# summaries with mismatched identities, tune/precision, or Graph protocol.
# The paired/arbitrary/skew overrides below are correctness diagnostics only.
# reduce_push replaces GEMM payload atomics with private route stores and rank
# reducers. Only metadata is cleared. vec16 covers 512 columns per wave64; node reducers
# read only their local inbox while rank reducers own the peer payload pushes.
# Q8/R32/F14 leaves 201 GEMM CTAs in a 256-CTA resident grid. n_major_window=2
# balances earlier group readiness with reuse of nearby M/N work. These are
# starting tune values, not a measured optimum. Trailing "$@" overrides flags.
# B1/per_row/cache-default retain the conservative configuration. B8, NT and
# per_tile are separate pending GPU ablations after general-route validation.
# Full design and parameter costs: FUSED_STAGE2_REDUCE_PUSH_DESIGN_20260910.md.
set +e
timeout --signal=TERM --kill-after=30s 1800s \
  python3 -u -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="${node_rank}" \
  --master-addr=10.2.80.17 --master-port="${master_port}" --max-restarts=0 \
  op_tests/multigpu_tests/bench_megamoe_tile_ep16_stage2_breakdown.py \
  --path "${bench_path}" --candidate-mode full --mori-mode gmm2_combine \
  --cuda-graph --hidden 3584 --activation situv2 --tokens 128 \
  --graph-comparison-statistic pooled_min \
  --warmup 10 --iters 40 --tail-iters 20 --seed 123 --route-pattern cross_node \
  --max-routes-per-token-per-rank 1 \
  --stage1-workers 256 --stage2-workers 256 \
  --candidate-node-accumulation-mode rank_local \
  --candidate-rank-accumulation-mode reduce_push --candidate-rank-reduce-blocks 8 \
  --candidate-node-reduce-vec-bytes 16 --candidate-node-reduce-load-schedule load_first \
  --candidate-gmm-work-swizzle n_major_window --candidate-window-n-groups 2 \
  --candidate-ready-granularity group \
  --torch-profiler-dir "${run_dir}" "$@" >"${run_dir}/run.log" 2>&1
rc=$?
rocm-smi --showuse --showmeminfo vram --json \
  >"${run_dir}/occupancy_postflight.json" 2>"${run_dir}/occupancy_postflight.stderr"
set -e
printf '%s\n' "${rc}" >"${run_dir}/run.status"
exit "${rc}"

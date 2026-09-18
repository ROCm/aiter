#!/usr/bin/env bash
# Run one frozen v60 Stage2 Graph case on one EP16 node.
set -euo pipefail

node_rank="${1:?node rank (0 or 1) required}"
master_port="${2:?unique master port required}"
bench_path="${3:?candidate or mori required}"
tokens="${4:?tokens per rank required}"
tag="${5:?unique run tag required}"

case "${node_rank}" in 0|1) ;; *) echo "node rank must be 0 or 1" >&2; exit 64 ;; esac
case "${bench_path}" in candidate|mori) ;; *) echo "path must be candidate or mori" >&2; exit 64 ;; esac
case "${tokens}" in 1|2|4|8|16|32|64|128|256|512|1024|2048) ;; *) echo "unsupported TPR ${tokens}" >&2; exit 64 ;; esac

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
run_dir="${repo_root}/trace_data/stage2_graph_20260909/${tag}/node${node_rank}"
mkdir -p "${run_dir}"
if [[ -e "${run_dir}/launch.json" || -e "${run_dir}/run.log" || -e "${run_dir}/run.status" ]]; then
  echo "run tag already exists on node ${node_rank}: ${tag}" >&2
  exit 64
fi

export AITER_CONFIG_FMOE="${repo_root}/scripts/megamoe_tile/kimik3_stage2_graph_tune.csv"
python3 - "${run_dir}/launch.json" "${node_rank}" "${master_port}" "${bench_path}" "${tokens}" "${tag}" <<'PY'
import json, os, socket, sys
out, node_rank, port, path, tokens, tag = sys.argv[1:]
document = {
    "schema": "best_v60_ep16_launch_v1",
    "hostname": socket.gethostname(),
    "node_rank": int(node_rank),
    "master_addr": "10.2.80.17",
    "master_port": int(port),
    "path": path,
    "tokens_per_rank": int(tokens),
    "tag": tag,
    "shape": {
        "hidden": 3584, "inter": 3072, "experts": 896,
        "topk": 16, "ep_size": 16, "activation": "situv2",
    },
    "sampling": {"warmup": 10, "iterations": 40, "tail_iterations": 20, "seed": 123},
    "routing": {"pattern": "cross_node", "max_routes_per_token_per_rank": 1},
    "candidate": {
        "stage1_workers": 256, "stage2_workers": 256,
        "rank_reduce_blocks": 56, "node_reduce_blocks": 16,
        "final_combine_blocks": 14, "window_n_groups": 1,
        "rank_push_batch_size": 64, "rank_push_batch_invariants": True,
        "rank_push_acquire_cohort": 2,
        "node_reduce_token_owner_fastpath": True,
        "rank_push_publication": "per_tile_counter",
        "rank_epilogue_barrier": "per_row",
        "all_ready_diagnostic": False,
    },
    "mori": {"mode": "gmm2_combine", "block_num": 96, "rdma_block_num": 64, "exact_capacity": True},
    "environment": {
        "GLOO_SOCKET_IFNAME": "enp193s0f1np1",
        "MORI_SOCKET_IFNAME": "enp193s0f1np1",
        "NCCL_SOCKET_IFNAME": "enp193s0f1np1",
        "MORI_DEVICE_NIC": "ionic",
        "MORI_RDMA_DEVICES": "^rocep193s0f0,rocep193s0f1",
        "MORI_IB_GID_INDEX": "1", "MORI_NUM_QP_PER_PE": "2",
        "MORI_SHMEM_HEAP_SIZE": "40G", "MORI_EP_LAUNCH_CONFIG_MODE": "AUTO",
        "AITER_CONFIG_FMOE": os.environ["AITER_CONFIG_FMOE"],
    },
}
with open(out, "w") as f:
    json.dump(document, f, indent=2, sort_keys=True)
    f.write("\n")
PY

exec bash "${repo_root}/scripts/megamoe_tile/run_stage2_graph_ep16.sh" \
  "${node_rank}" "${master_port}" "${bench_path}" "${tag}" \
  --tokens "${tokens}" \
  --graph-check-routing-replay \
  --stage1-workers 256 \
  --stage2-workers 256 \
  --candidate-final-combine-blocks 14 \
  --candidate-rank-reduce-blocks 56 \
  --candidate-node-reduce-blocks 16 \
  --candidate-node-reduce-token-owner-fastpath \
  --candidate-rank-push-batch-size 64 \
  --candidate-rank-push-batch-invariants \
  --candidate-rank-push-acquire-cohort 2 \
  --candidate-rank-epilogue-barrier per_row \
  --candidate-gmm-work-swizzle n_major_window \
  --candidate-window-n-groups 1 \
  --graph-rel-l2-threshold 1.0 "${@:6}"

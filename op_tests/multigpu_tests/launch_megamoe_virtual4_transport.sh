#!/usr/bin/env bash
# Launch the same physical 2x8 CCO team on both hosts for the logical 4x4 probe.
set -uo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 NODE_RANK MASTER_PORT TAG [transport probe args...]" >&2
  exit 2
fi

node_rank="$1"
master_port="$2"
tag="$3"
if [[ ! "$tag" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
  echo "TAG must be a simple filename component" >&2
  exit 2
fi
shift 3
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export GLOO_SOCKET_IFNAME=enp193s0f1np1
export MORI_SOCKET_IFNAME=enp193s0f1np1
export NCCL_SOCKET_IFNAME=enp193s0f1np1
export MORI_DEVICE_NIC=ionic
export MORI_ENABLE_RAIL_ONLY=1
export MORI_RDMA_DEVICES='^rocep193s0f0,rocep193s0f1'
export MORI_IB_GID_INDEX=1
export MORI_NUM_QP_PER_PE=2
export CCO_GDR_CAPABLE="${CCO_GDR_CAPABLE:-0}"
unset MORI_VPOD_ID

if [[ -d "$task_dir/runtime/mori" ]]; then
  export PYTHONPATH="$task_dir/runtime${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$task_dir/runtime/mori${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "$task_dir/flydsl_runtime" ]]; then
  export PYTHONPATH="$task_dir/flydsl_runtime${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$task_dir/flydsl_runtime/flydsl.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [[ -d "$task_dir/flydsl_51_runtime" ]]; then
  export PYTHONPATH="$task_dir/flydsl_51_runtime${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$task_dir/flydsl_51_runtime/flydsl.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

log_path="$task_dir/${tag}_node${node_rank}.log"
rc_path="$task_dir/${tag}_node${node_rank}.rc"
rm -f -- "$rc_path"
timeout --signal=TERM --kill-after=10s 300s \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$node_rank" \
  --master_addr=10.2.80.18 --master_port="$master_port" \
  "$task_dir/probe_megamoe_virtual4_transport.py" "$@" \
  >"$log_path" 2>&1
rc=$?
printf '%s\n' "$rc" >"$rc_path"
exit "$rc"

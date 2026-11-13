#!/usr/bin/env bash
# Run inside a CCO-enabled container on each physical host.
set -uo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 NODE_RANK MASTER_PORT TAG MR_SIZES_MIB GDR_CAPABLE [ALLOCATION_SIZES_MIB]" >&2
  exit 2
fi

node_rank="$1"
master_port="$2"
tag="$3"
mr_sizes_mib="$4"
gdr_capable="$5"
allocation_args=()
if [[ $# -eq 6 ]]; then
  allocation_args=(--allocation-sizes-mib="$6")
fi
task_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export GLOO_SOCKET_IFNAME=enp193s0f1np1
export MORI_SOCKET_IFNAME=enp193s0f1np1
export NCCL_SOCKET_IFNAME=enp193s0f1np1
export MORI_DEVICE_NIC=ionic
export MORI_ENABLE_RAIL_ONLY=1
export MORI_RDMA_DEVICES='^rocep193s0f0,rocep193s0f1'
export MORI_IB_GID_INDEX=1
export MORI_NUM_QP_PER_PE=2
export CCO_GDR_CAPABLE="$gdr_capable"
unset MORI_VPOD_ID

# A matching MORI build can be unpacked beside this probe on a host whose
# container ships an older build. Keep the container installation untouched.
if [[ -d "$task_dir/mori" ]]; then
  export PYTHONPATH="$task_dir${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$task_dir/mori${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ -d "$task_dir/flydsl_51_runtime" ]]; then
  export PYTHONPATH="$task_dir/flydsl_51_runtime${PYTHONPATH:+:$PYTHONPATH}"
  export LD_LIBRARY_PATH="$task_dir/flydsl_51_runtime/flydsl.libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

log_path="$task_dir/${tag}_node${node_rank}.log"
rc_path="$task_dir/${tag}_node${node_rank}.rc"
rm -f -- "$rc_path"
timeout --signal=TERM --kill-after=10s 300s \
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank="$node_rank" \
  --master_addr=10.2.80.18 --master_port="$master_port" \
  "$task_dir/probe_megamoe_virtual4_mr.py" \
  --mr-sizes-mib="$mr_sizes_mib" "${allocation_args[@]}" >"$log_path" 2>&1
rc=$?
printf '%s\n' "$rc" >"$rc_path"
exit "$rc"

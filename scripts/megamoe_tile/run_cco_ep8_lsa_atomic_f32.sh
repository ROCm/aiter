#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-$(pwd)}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-enp193s0f1np1}"
export MORI_SOCKET_IFNAME="${MORI_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

exec torchrun --standalone --nproc-per-node=8 \
  op_tests/multigpu_tests/test_megamoe_tile_cco_ep8_lsa_atomic_f32.py "$@"

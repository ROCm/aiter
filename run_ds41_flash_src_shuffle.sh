#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Experimental M256 source-rank shuffle; see README_dispatch_pull.md.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export AITER_MEGA_PULL_PLAN_PACK=1 AITER_MEGA_PULL_SRC_SHUFFLE=1
export AITER_MOE_S1F_READY_FIRST=0 AITER_MOE_S1F_TILE_READY=0
export AITER_MOE_S1F_PRODUCER_GROUP=1 AITER_MOE_S1F_AFFINITY=1
unset AITER_FLYDSL_INST_PREFETCH
exec bash run_ds41_flash_best.sh

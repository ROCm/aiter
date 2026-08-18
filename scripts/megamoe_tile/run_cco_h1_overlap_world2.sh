#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Thin launcher for the WORLD2 64-KiB CCO/ready-H1 overlap benchmark.  It uses
# the reviewed CCO rendezvous/preflight launcher and keeps its safe DRY_RUN=1
# default.  Set DRY_RUN=0 only after this script and the driver are synchronized
# to both AITER worktrees.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TEST_REL="scripts/megamoe_tile/bench_cco_h1_overlap_world2.py"
export MEGAMOE_CCO_QP=4
export MEGAMOE_CCO_BATCH=8
export MEGAMOE_CCO_CHUNK=65536

exec "${SCRIPT_DIR}/run_cco_transport_world2.sh"

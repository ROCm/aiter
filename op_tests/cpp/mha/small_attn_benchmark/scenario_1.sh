#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Scenario 1: sq, skv <= 16 — packed varlen (fwd group); bwd batch uniform P.
# Results: results/scenario1/fwd.csv, bwd.csv
#
#   cd op_tests/cpp/mha/small_attn_benchmark && ./scenario_1.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_common.sh
source "$SCRIPT_DIR/bench_common.sh"
bench_run_scenario 1

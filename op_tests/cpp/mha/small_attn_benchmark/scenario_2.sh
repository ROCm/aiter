#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Scenario 2: sq=1, skv<=16 — packed varlen KV (fwd group); bwd batch s_q=1, s_kv=P.
# Results: results/scenario2/fwd.csv, bwd.csv
#
#   cd op_tests/cpp/mha/small_attn_benchmark && ./scenario_2.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_common.sh
source "$SCRIPT_DIR/bench_common.sh"
bench_run_scenario 2

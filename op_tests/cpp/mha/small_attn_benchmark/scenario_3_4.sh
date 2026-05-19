#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Scenarios 3+4: fixed self-attn, batch mode — sq=skv sweep SEQ_MIN..SEQ_MAX (default 2..17).
# Results: results/scenario3_4/fwd.csv, bwd.csv
#
#   cd op_tests/cpp/mha/small_attn_benchmark && ./scenario_3_4.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bench_common.sh
source "$SCRIPT_DIR/bench_common.sh"
bench_run_scenario_3_4

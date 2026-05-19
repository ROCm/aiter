#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Run small-attn scenarios (CK ck_pr_6764 only). Build once from parent mha/.
#
#   cd op_tests/cpp/mha && bash build_mha.sh
#   cd small_attn_benchmark && ./run_all.sh
#
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for script in scenario_1.sh scenario_2.sh scenario_3_4.sh; do
    echo "################ ${script%.sh} ################"
    "$DIR/$script"
done
echo "done — results under ${RESULTS_DIR:-$DIR/results}/scenario{1,2,3_4}/fwd.csv and bwd.csv"

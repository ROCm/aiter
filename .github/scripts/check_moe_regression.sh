#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Wrap scripts/compare_benchmark.py for the bench_moe CI job.
# Prints a comparison table to stdout (captured by the job step into
# $GITHUB_STEP_SUMMARY). Exits 0 unless --fail-on-regress is passed and
# at least one REGRESS row is found.
#
# Usage: check_moe_regression.sh <baseline_csv> <current_csv> [extra args...]
set -euo pipefail

BASE=${1:?baseline csv path required}
CURR=${2:?current csv path required}
shift 2

BASE_LABEL="baseline"
CURR_LABEL="current"
if [[ -n "${BASE_SHA:-}" ]]; then
    BASE_LABEL="main(${BASE_SHA:0:7})"
fi
if [[ -n "${CURR_SHA:-}" ]]; then
    CURR_LABEL="PR(${CURR_SHA:0:7})"
elif [[ -n "${GITHUB_SHA:-}" ]]; then
    CURR_LABEL="${GITHUB_SHA:0:7}"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

python3 "${REPO_ROOT}/scripts/compare_benchmark.py" \
    "$BASE" "$CURR" \
    --baseline-label "$BASE_LABEL" \
    --current-label "$CURR_LABEL" \
    --warn 1.10 \
    --fail 1.15 \
    "$@"

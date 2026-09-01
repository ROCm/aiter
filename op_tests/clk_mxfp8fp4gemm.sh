#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# DPM clock-sampling driver for the gfx1250 mxfp8fp4 F8GEMM UT.
#
# The UT's `--mode clk` runs a short torch-profiled burst (for device-time
# TFLOPS/BW) and then an UNPROFILED back-to-back soak for --clk-seconds. This
# script runs it in the background and points clk_trace.py --wait-pid at its PID,
# so clocks are sampled from sysfs at 1ms for the whole run (the burst + soak);
# --settle drops the DPM/thermal ramp, and the median over the soak window is the
# steady clock. clk_trace.py reads the real sclk from hwmon/freq1_input and the
# fclk from the starred pp_dpm_fclk entry (pp_dpm_sclk lists DPM levels, not the
# measured clock, so it is NOT usable).
#
# Usage:
#   ./clk_mxfp8fp4gemm.sh
#   INTYPE=a8w4 SHAPE=2,1048576,16384 CLK_SECONDS=15 ./clk_mxfp8fp4gemm.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PY:-python3}"
CLK_TRACE="${CLK_TRACE:-/home/yayang/dpm_exp/probe/opus_mem_bw_probe/clk_trace.py}"
INTYPE="${INTYPE:-a8w8}"
SHAPE="${SHAPE:-2,1048576,16384}"   # memory-bound decode representative
CLK_SECONDS="${CLK_SECONDS:-8}"    # short on purpose: avoid heating into throttle
PERIOD_MS="${PERIOD_MS:-1}"         # sysfs sample period (clk_trace default 1ms)
SETTLE_S="${SETTLE_S:-0.5}"         # seconds dropped at the start (skip the ramp)
CSV="${CSV:-clk_trace_${INTYPE}.csv}"
METRICS="${METRICS:-clk_metrics_${INTYPE}.json}"

if [ ! -f "${CLK_TRACE}" ]; then
    echo "clk_trace.py not found at ${CLK_TRACE}; set CLK_TRACE=<path>" >&2
    exit 1
fi

"${PY}" test_mxfp8fp4gemm.py --mode clk --intype "${INTYPE}" -s "${SHAPE}" \
    --clk-seconds "${CLK_SECONDS}" --metrics-json "${METRICS}" &
pid=$!

# CLK_CARD_DEV lets clk_trace.py find the amdgpu sysfs node if it is not card1.
"${PY}" "${CLK_TRACE}" --wait-pid "${pid}" \
    --period "${PERIOD_MS}" --settle "${SETTLE_S}" \
    --csv "${CSV}" --summary

wait "${pid}"
echo "clk trace written to ${CSV}"

# Merge the per-sample clocks with the UT's soak throughput into one DPM table.
"${PY}" clk_merge.py --clk-csv "${CSV}" --metrics-json "${METRICS}" \
    --settle "${SETTLE_S}"

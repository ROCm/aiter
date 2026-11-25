#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# FlyDSL >= ASM shapes (large / high-K)
# Measured FlyDSL/ASM: 101%, 101%, 103%, 100%, 112%, 101%.
# Thin wrapper around reproduce.sh (same env vars; SHAPES/MODES/OUT_DIR can still
# be overridden). Default: blockscale method, uniform data, 1 round.
#
# Usage: op_tests/flydsl_tests/mxfp8_gemm_vs_asm/reproduce-1.sh
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SHAPES=${SHAPES:-"8192,8192,8192 16384,4608,5120 16384,5120,8192 16384,16384,16384 16384,25600,6144 16384,32768,8192"}
export MODES=${MODES:-"uniform:auto"}
export OUT_DIR=${OUT_DIR:-/tmp/mxfp8_repro_1}
exec "$HERE/reproduce.sh" "$@"

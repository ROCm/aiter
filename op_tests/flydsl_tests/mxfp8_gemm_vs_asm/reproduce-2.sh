#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Shapes where FlyDSL is slightly behind ASM (small shape / small-medium K)
# Measured FlyDSL/ASM: 90%, 95%, 92%, 89%, 97%, 88% (K=1280 uses ps4).
# Thin wrapper around reproduce.sh (same env vars; SHAPES/MODES/OUT_DIR can still
# be overridden). Default: blockscale method, uniform data, 1 round.
#
# Usage: op_tests/flydsl_tests/mxfp8_gemm_vs_asm/reproduce-2.sh
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SHAPES=${SHAPES:-"4096,4096,4096 16384,1792,5120 16384,5120,2304 16384,5120,4224 16384,5120,4352 16384,32768,1280"}
export MODES=${MODES:-"uniform:auto"}
export OUT_DIR=${OUT_DIR:-/tmp/mxfp8_repro_2}
exec "$HERE/reproduce.sh" "$@"

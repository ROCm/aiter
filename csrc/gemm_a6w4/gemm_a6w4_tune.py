# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune registered gfx950 A6W4 mixed-MXFP assembly kernels."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from csrc.gemm_mixed_mxfp.gemm_mixed_mxfp_tune import run_tuner

if __name__ == "__main__":
    run_tuner("a6w4")

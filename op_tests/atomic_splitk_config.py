# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Atomic split-K tuned rows, read only by the tuner and the ubench.

The ``_atomic`` infix misses ``get_config_file``'s
``*a8w8_blockscale_[a]bpreshuffle_tuned_gemm*.csv`` glob, so mainline dispatch
never merges these and keeps resolving to the fused rows.
"""

import os

from aiter.jit.core import AITER_ROOT_DIR

CONFIG_DIR = os.path.join(AITER_ROOT_DIR, "aiter", "configs", "model_configs")
BPRE = "dsv4_a8w8_blockscale_bpreshuffle_atomic_tuned_gemm.csv"
ABPRE = "dsv4_a8w8_blockscale_abpreshuffle_atomic_tuned_gemm.csv"


def atomic_config_path(apre: bool) -> str:
    return os.path.join(CONFIG_DIR, ABPRE if apre else BPRE)

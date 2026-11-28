# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Helpers shared by the fused MoE tuner backends."""

import os

import pandas as pd

_TUNE_EXCLUDE_KERNEL_PATTERNS = [
    p.strip()
    for p in os.environ.get("AITER_FMOE_TUNE_EXCLUDE_KERNELS", "").split(",")
    if p.strip()
]


def _is_tune_excluded_kernel(kernel_name) -> bool:
    """True if ``kernel_name`` matches any excluded-kernel pattern."""
    name = str(kernel_name or "")
    return any(pat in name for pat in _TUNE_EXCLUDE_KERNEL_PATTERNS)


def get_kernels_dict(file, key="tile_m"):
    if not os.path.exists(file):
        print(f"ASM kernel list file not exist: {file}")
        return {}
    df = pd.read_csv(file)
    kernel_dict = df.groupby(key)["knl_name"].apply(list).to_dict()
    return kernel_dict

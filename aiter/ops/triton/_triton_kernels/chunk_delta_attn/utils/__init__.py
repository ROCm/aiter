# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from .cumsum import chunk_gate_cumsum
from .index import prepare_chunk_indices
from .l2norm import l2norm_fwd
from .utils import (
    IS_GATHER_SUPPORTED,
    IS_TF32_SUPPORTED,
    RCP_LN2,
    autotune_cache_kwargs,
    check_shared_mem,
    chunk_delta_attn_autotune_configs,
    exp,
    exp2,
    input_guard,
    softplus,
)

__all__ = [
    "IS_GATHER_SUPPORTED",
    "IS_TF32_SUPPORTED",
    "RCP_LN2",
    "autotune_cache_kwargs",
    "check_shared_mem",
    "chunk_delta_attn_autotune_configs",
    "chunk_gate_cumsum",
    "exp",
    "exp2",
    "input_guard",
    "l2norm_fwd",
    "prepare_chunk_indices",
    "softplus",
]

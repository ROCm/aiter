# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Gated Delta Rule Operations (Forward Only).

This module provides optimized Triton kernels for gated delta rule computations.

Available operations:
- Fused recurrent forward: _fused_recurrent_gated_delta_rule_fwd_kernel
- Chunk-based forward: chunk_gated_delta_rule_fwd
- Hidden state computation: chunk_gated_delta_rule_fwd_h
- Output computation: chunk_fwd_o

Note: Only forward pass is implemented. Backward pass is not supported in aiter.
      For training with gradients, please use the flash-linear-attention library.
"""

from .decode.fused_recurrent import _fused_recurrent_gated_delta_rule_fwd_kernel
from .prefill.chunk import (
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_fwd_opt,
    chunk_gated_delta_rule_fwd_opt_vk,
)
from .prefill.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_h_opt,
    chunk_gated_delta_rule_fwd_h_opt_vk,
)
from .prefill.chunk_o import chunk_fwd_o, chunk_fwd_o_opt, chunk_fwd_o_opt_vk
from .prefill.fused_cumsum_kkt import fused_chunk_local_cumsum_scaled_dot_kkt_fwd
from .prefill.fused_solve_tril_recompute import fused_solve_tril_recompute_w_u
from . import gated_delta_rule_utils

__all__ = [
    "_fused_recurrent_gated_delta_rule_fwd_kernel",
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_fwd_opt",
    "chunk_gated_delta_rule_fwd_opt_vk",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_gated_delta_rule_fwd_h_opt",
    "chunk_gated_delta_rule_fwd_h_opt_vk",
    "chunk_fwd_o",
    "chunk_fwd_o_opt",
    "chunk_fwd_o_opt_vk",
    "fused_chunk_local_cumsum_scaled_dot_kkt_fwd",
    "fused_solve_tril_recompute_w_u",
    "gated_delta_rule_utils",
]

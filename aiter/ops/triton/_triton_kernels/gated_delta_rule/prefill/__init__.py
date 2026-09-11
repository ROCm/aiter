# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Gated Delta Rule Prefill Operations (Forward Only).

This module provides optimized Triton kernels for prefill/training operations.
"""

from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk import (
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_fwd_opt,
    chunk_gated_delta_rule_fwd_opt_vk,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    chunk_gated_delta_rule_fwd_h_opt,
    chunk_gated_delta_rule_fwd_h_opt_vk,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.chunk_o import (
    chunk_fwd_o,
    chunk_fwd_o_opt,
    chunk_fwd_o_opt_vk,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_cumsum_kkt import (
    fused_chunk_local_cumsum_scaled_dot_kkt_fwd,
    fused_cumsum_kkt,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_gdn_gating_prefill import (
    fused_gdn_gating_and_sigmoid,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.prefill.fused_solve_tril_recompute import (
    fused_solve_tril_recompute_w_u,
)

__all__ = [
    "chunk_fwd_o",
    "chunk_fwd_o_opt",
    "chunk_fwd_o_opt_vk",
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_fwd_h",
    "chunk_gated_delta_rule_fwd_h_opt",
    "chunk_gated_delta_rule_fwd_h_opt_vk",
    "chunk_gated_delta_rule_fwd_opt",
    "chunk_gated_delta_rule_fwd_opt_vk",
    "fused_chunk_local_cumsum_scaled_dot_kkt_fwd",
    "fused_cumsum_kkt",
    "fused_gdn_gating_and_sigmoid",
    "fused_solve_tril_recompute_w_u",
]

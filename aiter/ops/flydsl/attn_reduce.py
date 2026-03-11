# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level Python API for the FlyDSL attention reduce kernel."""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch

from aiter.jit.utils.chip_info import get_cu_num, get_lds_size_per_cu


def _next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x (for x >= 1)."""
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _get_num_work_group_per_bh(
    num_reduce_tile: int,
    max_seqlen_q: int,
    num_heads: int,
    num_cu: int,
    occupancy: int,
) -> int:
    """Compute the number of work-groups per (batch, head) tile."""
    hw_capacity = num_cu * occupancy
    num_workloads = num_reduce_tile * num_heads

    factor = 1.3
    supported = [1, 2, 4, 8, 16, 64, 128, 256]

    if hw_capacity * factor <= num_workloads:
        return 1

    wg_per_bh_hw = math.ceil(hw_capacity * factor / num_workloads)
    wg_per_bh = min(wg_per_bh_hw, max_seqlen_q)
    wg_per_bh_aligned = 1 if wg_per_bh == 1 else _next_power_of_two(wg_per_bh)
    wg_per_bh_clamped = min(wg_per_bh_aligned, supported[-1])

    for s in supported:
        if wg_per_bh_clamped <= s:
            return s

    return supported[-1]


def flydsl_attn_reduce_v1(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: Optional[torch.Tensor],
    reduce_partial_map: torch.Tensor,
    max_seqlen_q: int,
    final_output: torch.Tensor,
    final_lse: Optional[torch.Tensor] = None,
) -> None:
    """Drop-in replacement for ``mla_reduce_v1`` using FlyDSL.

    The function signature and semantics are identical to the C++/pybind11
    version in ``aiter/ops/attention.py``.
    """
    from .kernels.attn_reduce import (
        OCCUPANCY,
        launch_attn_reduce,
        launch_attn_reduce_ps,
    )

    num_heads = partial_output.size(-2)  # [*, h, dv]

    output_lse = final_lse is not None
    use_reduce_final_map = reduce_final_map is not None
    num_reduce_tile = reduce_indptr.size(0) - 1

    if num_reduce_tile <= 0:
        return

    stride_s_o = final_output.stride(-3)
    stride_h_o = final_output.stride(-2)

    device = final_output.device
    if reduce_final_map is None:
        reduce_final_map_t = torch.zeros(1, dtype=torch.int32, device=device)
    else:
        reduce_final_map_t = reduce_final_map

    if final_lse is None:
        final_lse_t = torch.zeros(1, dtype=torch.float32, device=device)
    else:
        final_lse_t = final_lse

    # --- All decision logic in pure Python ---
    num_cu = get_cu_num()
    max_lds_per_cu = get_lds_size_per_cu()
    max_splits = num_cu

    # LDS = max_splits * sizeof(i32) + max_splits * sizeof(f32)
    #      + max(0, max_splits - 256) * sizeof(f32)
    lds_size = max_splits * 4 + max_splits * 4 + max(0, max_splits - 256) * 4

    if lds_size > max_lds_per_cu:
        raise RuntimeError(
            "kn_mla_reduce_v1: The number of splits exceeds what kernel can handle."
        )
    if lds_size > max_lds_per_cu // OCCUPANCY:
        warnings.warn(
            "kn_mla_reduce_v1: The number of splits is too high, "
            "adversely affecting occupancy."
        )

    num_wg_per_bh = _get_num_work_group_per_bh(
        num_reduce_tile,
        max_seqlen_q,
        num_heads,
        num_cu,
        OCCUPANCY,
    )

    ps_grid_size = num_cu * OCCUPANCY * 2
    total_work = num_heads * num_wg_per_bh * num_reduce_tile

    common_args = (
        reduce_indptr,
        reduce_final_map_t,
        reduce_partial_map,
        final_lse_t,
        final_output,
        partial_lse,
        partial_output,
        stride_s_o,
        stride_h_o,
        num_reduce_tile,
    )

    if total_work <= ps_grid_size:
        launch_attn_reduce(
            *common_args,
            num_heads,
            num_wg_per_bh,
            lds_size,
            max_splits,
            int(output_lse),
            int(use_reduce_final_map),
        )
    else:
        launch_attn_reduce_ps(
            *common_args,
            ps_grid_size,
            lds_size,
            max_splits,
            int(output_lse),
            int(use_reduce_final_map),
        )

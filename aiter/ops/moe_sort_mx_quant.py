# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused MoE sort + MXFP8 activation quant (one launch, no grid barrier)."""

import torch

from ..jit.core import compile_ops
from ..utility import dtypes

MD_NAME = "module_moe_sort_mx_quant"

# Hidden sizes the kernel is instantiated for (cols == threads * 16).
MOE_SORT_MX_QUANT_COLS = (4096, 5120, 7168)
# One wave ranks a block's routes and the grid is capped at the CU count, so
# ceil(M / num_cu) * topk must stay within a wave.
MOE_SORT_MX_QUANT_MAX_ROUTES_PER_BLOCK = 64


def _moe_sort_mx_quant_fwd_fake(*args, **kwargs) -> None:
    return


# develop=True converts torch.Tensor -> aiter_tensor_t (the module is torch_exclude).
@compile_ops(MD_NAME, gen_fake=_moe_sort_mx_quant_fwd_fake, develop=True)
def moe_sort_mx_quant_fwd(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: torch.Tensor,
    out_scale: torch.Tensor,
    num_experts: int,
    unit_size: int,
    moe_buf: torch.Tensor | None = None,
) -> None:
    """Sort routes by expert and MXFP8-quantize ``hidden`` in one launch.

    Writes the outputs of ``moe_sorting`` followed by
    ``fused_dynamic_mx_quant_moe_sort(quant_dtype=fp8)``: the sorted routing
    map, ``out`` = per-token fp8 activations and ``out_scale`` = e8m0 scales
    swizzled per sorted row. Padding rows carry the sentinel
    ``(topk << 24) | M`` with weight 0, and the buffers past ``num_valid_ids[0]``
    are filled the same way (expert id 0). Zeroes ``moe_buf`` when given.
    """


def moe_sort_mx_quant(
    hidden: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    unit_size: int,
    moe_buf: torch.Tensor | None = None,
):
    """Allocate the outputs and run :func:`moe_sort_mx_quant_fwd`.

    Returns ``(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids,
    out, out_scale)``.
    """
    device = hidden.device
    M, cols = hidden.shape
    routes = topk_ids.numel()
    active = min(num_experts, routes)
    max_rows = (
        (routes + active * (unit_size - 1) + unit_size - 1) // unit_size * unit_size
    )
    sorted_ids = torch.empty(max_rows, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(max_rows, dtype=dtypes.fp32, device=device)
    sorted_expert_ids = torch.empty(
        max_rows // unit_size, dtype=dtypes.i32, device=device
    )
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    out = torch.empty((M, cols), dtype=dtypes.fp8, device=device)
    scale_n_pad = (cols // 32 + 7) // 8 * 8
    out_scale = torch.empty(
        ((max_rows + 31) // 32 * 32, scale_n_pad), dtype=dtypes.fp8_e8m0, device=device
    )
    moe_sort_mx_quant_fwd(
        hidden,
        topk_ids,
        topk_weights,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        out,
        out_scale,
        num_experts,
        unit_size,
        moe_buf,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, out, out_scale

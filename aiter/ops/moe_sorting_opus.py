# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from ..jit.core import compile_ops
from ..utility import dtypes

MD_NAME = "module_moe_sorting_opus"


@compile_ops("module_moe_sorting_opus", develop=True)
def moe_sorting_opus_get_workspace_size(
    tokens: int,
    num_experts: int,
    topk: int,
    dispatch_policy: int = 0,
    device_id: int = -1,
) -> int: ...


@compile_ops("module_moe_sorting_opus", develop=True)
def moe_sorting_opus_fwd(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    num_experts: int,
    unit_size: int,
    local_expert_mask: torch.Tensor | None = None,
    num_local_tokens: torch.Tensor | None = None,
    workspace: torch.Tensor | None = None,
    dispatch_policy: int = 0,
    local_topk_ids: torch.Tensor | None = None,
    m_indices: torch.Tensor | None = None,
    reverse_sorted: torch.Tensor | None = None,
) -> None: ...


@compile_ops("module_moe_sorting_opus", develop=True)
def mxfp4_moe_sort_quant_fwd(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    activation_quant: torch.Tensor,
    activation_scale_token: torch.Tensor,
    num_experts: int,
    num_local_tokens: torch.Tensor | None = None,
) -> None:
    """Fused route sort + compact MXFP4 activation quantization.

    gfx950 only, ``model_dim=4096``, ``block_m=32``, ``M`` in ``[1, 128]``.
    ``(num_experts, topk)`` must be one of ``(512, 10)`` or ``(513, 11)``
    -- the expert histogram is a kernel template parameter.
    ``num_local_tokens``: as for ``moe_sorting_opus_fwd`` -- only that many
    leading rows are sorted and quantized; ``moe_buf`` is zeroed whole.
    """


@compile_ops("module_moe_sorting_opus", develop=True)
def m3_moe_sort_quant_fwd(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_weights: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    moe_buf: torch.Tensor,
    activation_quant: torch.Tensor,
    activation_scale: torch.Tensor,
    num_local_tokens: torch.Tensor | None = None,
) -> None:
    """MiniMax-M3.1 FHMoE decode (H 6144, 129 experts, top-5, block_m 32, M <= 256): the opus
    sort, moe_buf zeroing and ``fused_dynamic_mx_quant_moe_sort`` (MXFP4 or MXFP8 by
    ``activation_quant``'s dtype) in one launch, byte-identical on every row they define.
    """


def m3_moe_sort_quant(
    x,
    topk_ids,
    topk_weights,
    q_dtype,
    num_local_tokens=None,
    moe_buf=None,
    accumulate=True,
):
    """``m3_moe_sort_quant_fwd`` with moe_sorting's allocations. Returns sorted ids, weights, expert ids,
    num_valid_ids, moe_buf (zeroed; empty when not ``accumulate``), the activation and its sorted scales.
    """
    n, dev, E, BM = x.shape[0], x.device, 129, 32
    max_sorted = (n * 5 + E * BM - 5 + BM - 1) // BM * BM
    i32, f32 = torch.int32, torch.float32
    out = (
        torch.empty(max_sorted, dtype=i32, device=dev),
        torch.empty(max_sorted, dtype=f32, device=dev),
        torch.empty(max_sorted // BM, dtype=i32, device=dev),
        torch.empty(2, dtype=i32, device=dev),
    )
    if moe_buf is None:
        moe_buf = torch.empty(
            (n, x.shape[1]) if accumulate else (0, 0), dtype=x.dtype, device=dev
        )
    cols = x.shape[1] // 2 if q_dtype == dtypes.fp4x2 else x.shape[1]
    xq = torch.empty((n, cols), dtype=q_dtype, device=dev)
    xs = torch.empty(
        ((max_sorted + 31) // 32 * 32, x.shape[1] // 32),
        dtype=dtypes.fp8_e8m0,
        device=dev,
    )
    m3_moe_sort_quant_fwd(
        x, topk_ids, topk_weights, *out, moe_buf, xq, xs, num_local_tokens
    )
    return (*out, moe_buf, xq, xs)

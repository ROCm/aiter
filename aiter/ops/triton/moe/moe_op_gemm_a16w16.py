# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a16w16 import (
    _moe_gemm_a16w16,
)
from aiter.ops.triton.moe.moe_routing.routing import ExptData, RoutingData
from aiter.ops.triton.moe.moe_routing.topk_align import align_topk_ids
from aiter.ops.triton.moe.reduce import reduce_grouped, validate_reduce_out
from aiter.ops.triton.utils._triton.arch_info import get_arch


def build_routing_from_topk(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    num_experts: int,
    block_m: int = 32,
):
    return align_topk_ids(
        topk_ids, topk_weights, num_experts=num_experts, block_m=block_m
    )


def _config(block_m: int, n: int, k: int):
    return {
        "block_m": block_m,
        "block_n": 64,
        "block_k": 128 if k % 128 == 0 else 64,
        "group_m": 4,
        "xcd_swizzle": 8 if get_arch() == "gfx1201" else 1,
        "w_cache_modifier": ".cg" if block_m <= 32 else None,
        "num_warps": 8,
        "num_stages": 2,
    }


def moe_gemm_a16w16(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    routing_data: RoutingData,
    gather_indx: torch.Tensor | None = None,
    scatter_indx: torch.Tensor | None = None,
    gammas: torch.Tensor | None = None,
    apply_separated_silu: bool = False,
    out_dtype=torch.bfloat16,
    y_out: torch.Tensor | None = None,
):
    """Grouped BF16 MoE GEMM over current RoutingData metadata.

    `w` uses AITER MoE layout [E, N, K].
    """
    assert x.dtype in (torch.bfloat16, torch.float16)
    assert w.dtype in (torch.bfloat16, torch.float16)
    assert x.is_contiguous()
    assert w.is_contiguous()

    num_rows = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    num_experts, n, k = w.shape
    assert num_experts == routing_data.n_expts_tot
    assert x.shape[-1] == k
    block_m = routing_data.block_m
    cfg = _config(block_m, n, k)
    reduction_n = 2 if apply_separated_silu else 1
    out_n = n // reduction_n
    y = torch.empty((1, num_rows, out_n), dtype=out_dtype, device=x.device)
    y_final = (
        validate_reduce_out(y_out, (num_rows // routing_data.n_expts_act, out_n), out_dtype, x.device)
        if scatter_indx is not None
        else None
    )

    expt_data = routing_data.expt_data
    grid_m = routing_data.n_blocks(num_rows, block_m)
    grid_n = triton.cdiv(out_n, cfg["block_n"])
    grid = grid_m * grid_n

    _moe_gemm_a16w16[(grid,)](
        y,
        y.stride(0),
        y.stride(1),
        y.stride(2),
        x,
        x.stride(0),
        x.stride(1),
        w,
        w.stride(0),
        w.stride(1),
        w.stride(2),
        gammas,
        n,
        k,
        gather_indx,
        expt_data.hist,
        expt_data.token_offs_raw,
        expt_data.token_offs_pad[-1],
        expt_data.block_pid_map,
        grid_m,
        grid_n,
        apply_separated_silu,
        routing_data.n_expts_act,
        cfg["block_m"],
        cfg["block_n"],
        cfg["block_k"],
        cfg["group_m"],
        cfg["xcd_swizzle"],
        k % cfg["block_k"] == 0,
        k % cfg["block_k"],
        cfg["w_cache_modifier"],
        False,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )

    if scatter_indx is None:
        return y.squeeze(0)

    group_indx = scatter_indx.view(-1, routing_data.n_expts_act)
    return reduce_grouped(
        y,
        group_indx,
        y_final,
        out_dtype=out_dtype,
    )


def fused_moe_bf16_g1u1(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
):
    assert w1.is_contiguous()
    assert w2.is_contiguous()
    experts = w1.shape[0]
    routing_data, gather_indx, scatter_indx, sorted_weights = build_routing_from_topk(
        topk_ids, topk_weight, num_experts=experts, block_m=32
    )
    stage1 = moe_gemm_a16w16(
        hidden_states.contiguous(),
        w1,
        routing_data=routing_data,
        gather_indx=gather_indx,
        apply_separated_silu=True,
        out_dtype=torch.bfloat16,
    )
    return moe_gemm_a16w16(
        stage1.contiguous(),
        w2,
        routing_data=routing_data,
        scatter_indx=scatter_indx,
        gammas=sorted_weights,
        out_dtype=torch.bfloat16,
        y_out=output,
    )

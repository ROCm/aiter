# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""W4A16 MoE for RDNA3.5 (gfx115x) on the FlyDSL WMMA kernels in
``kernels/moe_w4a16_rdna.py``.

Weights are int4 with a group size that is a multiple of 16, with or without
zero points, packed by ``pack_w4a16_weights``. Activations are bfloat16 or
float16.
"""

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kernels.moe_w4a16_rdna import (
    BLOCK_M,
    MAX_REDUCE_ROWS,
    ROW_PATH_MAX_TOKENS,
    TILE_N,
    _build_gemm,
    _build_gemm_reduce,
    _build_gemm_row,
    _check_coef,
    _expert_stride_dwords,
    gate_up_tile_order,
    pack_w4a16_weights,
    pad_expert_stride,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, current_stream_arg

__all__ = [
    "BLOCK_M",
    "MAX_REDUCE_ROWS",
    "ROW_PATH_MAX_TOKENS",
    "TILE_N",
    "flydsl_rdna_moe",
    "flydsl_rdna_moe_gemm",
    "flydsl_rdna_moe_gemm_reduce",
    "flydsl_rdna_moe_supported",
    "gate_up_tile_order",
    "moe_align_block_size_rows",
    "pack_w4a16_weights",
    "pad_expert_stride",
]


def flydsl_rdna_moe_supported() -> bool:
    return get_gfx_runtime().startswith("gfx115")


def _launch(launch, *args) -> None:
    """Run a ``@flyc.jit`` launcher through a compiled handle cached per
    device and tensor dtype, rank and unit-stride axis. The kernels read only
    the data pointers of their tensors."""
    tensors = [a for a in args if isinstance(a, torch.Tensor)]
    key = (tensors[0].device,) + tuple(
        (t.dtype, t.dim(), t.stride(-1) == 1) for t in tensors
    )
    stream = current_stream_arg(tensors[0].device)
    _run_compiled(launch, *args, stream, specialization_key=key)


def moe_align_block_size_rows(
    topk_ids: torch.Tensor, block_size: int, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group the routed rows ``r = token * top_k + k`` by expert into blocks of
    ``block_size`` rows: ``(sorted_ids, expert_ids, num_valid)``.

    Padding rows hold ``topk_ids.numel()`` and blocks past the last expert
    hold expert -1.
    """
    from aiter.ops.moe_op import moe_align_block_size

    numel = topk_ids.numel()
    max_blocks = -(-(numel + num_experts * (block_size - 1)) // block_size)
    device = topk_ids.device
    sorted_ids = torch.full(
        (max_blocks * block_size,), numel, dtype=torch.int32, device=device
    )
    expert_ids = torch.full((max_blocks,), -1, dtype=torch.int32, device=device)
    token_nums = torch.empty((max_blocks,), dtype=torch.int32, device=device)
    num_valid = torch.empty((1,), dtype=torch.int32, device=device)
    moe_align_block_size(
        topk_ids, num_experts, block_size, sorted_ids, expert_ids, token_nums, num_valid
    )
    return sorted_ids, expert_ids, num_valid


def flydsl_rdna_moe_gemm_reduce(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    coef: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    group_size: int,
    coef_pairs: bool = True,
) -> None:
    """GEMM2 with the top-k sum: ``out[t] = sum_e weights[t, e] *
    W[expert_ids[t * top_k + e]] @ x[t * top_k + e]``.

    ``weights`` is float32 ``[tokens, top_k]`` with at most
    ``MAX_REDUCE_ROWS`` experts per token. ``coef_pairs`` is False for weights
    packed without zero points.
    """
    _check_coef(coef, x.shape[1], group_size, coef_pairs)
    n_tokens, top_k = weights.shape
    launch = _build_gemm_reduce(
        x.shape[1],
        out.shape[1],
        group_size,
        top_k,
        _expert_stride_dwords(w),
        _expert_stride_dwords(coef),
        coef_pairs,
        x.dtype == torch.float16,
    )
    _launch(launch, out, x, w, coef, expert_ids, weights, n_tokens, 32)


def flydsl_rdna_moe_gemm(
    out: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    coef: torch.Tensor,
    sorted_ids: torch.Tensor | None,
    expert_ids: torch.Tensor,
    num_valid: torch.Tensor | None,
    x_row_div: int,
    group_size: int,
    silu_mul: bool = False,
    coef_pairs: bool = True,
) -> None:
    """Run ``out[r] = W[expert(r)] @ x[r // x_row_div]`` for all routed rows.

    ``sorted_ids`` / ``expert_ids`` / ``num_valid`` come from
    ``moe_align_block_size`` with block size ``BLOCK_M``. When ``sorted_ids``
    is None, ``expert_ids[r]`` is the expert of row ``r`` and no grouping is
    needed; this is the cheap routing for a few rows with distinct experts.

    With ``silu_mul`` the weights are in ``gate_up_tile_order`` and ``out``
    receives ``silu(gate) * up``, half as many columns as the weights have.
    ``coef_pairs`` is False for weights packed without zero points.
    """
    _check_coef(coef, x.shape[1], group_size, coef_pairs)
    n_rows, N = out.shape
    if silu_mul:
        N *= 2
    build_args = (
        x.shape[1],
        N,
        x_row_div,
        group_size,
        silu_mul,
        _expert_stride_dwords(w),
        _expert_stride_dwords(coef),
        coef_pairs,
        x.dtype == torch.float16,
    )
    if sorted_ids is None:
        launch = _build_gemm_row(*build_args)
        _launch(launch, out, x, w, coef, expert_ids, expert_ids.numel())
        return
    launch = _build_gemm(*build_args)
    _launch(
        launch,
        out,
        x,
        w,
        coef,
        sorted_ids,
        expert_ids,
        num_valid,
        n_rows,
        expert_ids.numel(),
    )


def flydsl_rdna_moe(
    output: torch.Tensor,
    x: torch.Tensor,
    w1: torch.Tensor,
    coef1: torch.Tensor,
    w2: torch.Tensor,
    coef2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    workspace: torch.Tensor,
    group_size: int,
    num_experts: int,
    N: int,
    activation=None,
    w1_pairs: bool = True,
    w2_pairs: bool = True,
) -> None:
    """``output[t] = sum_e topk_weights[t, e] * (W2[e] @ act(W1[e] @ x[t]))``.

    ``w1`` / ``w2`` and their coefficients are in the ``pack_w4a16_weights``
    layout. ``activation(out, gate_up)`` is applied between the GEMMs; when
    it is None, ``w1`` is in ``gate_up_tile_order`` and GEMM1 applies
    SiLU-and-multiply itself. ``N`` is the width of the activation output.
    ``workspace`` holds the intermediates, bf16 ``M * top_k * (gate_up + N +
    K)`` elements.
    """
    silu_mul = activation is None
    M, K = x.shape
    top_k = topk_ids.size(1)
    gate_up = w1.size(1) * TILE_N
    rows = M * top_k
    widths = (gate_up, N, K)
    sizes = [rows * n for n in widths]
    gate_up_out, act_out, down_out = (
        t.view(rows, n)
        for t, n in zip(workspace.view(-1)[: sum(sizes)].split(sizes), widths)
    )
    if M <= ROW_PATH_MAX_TOKENS:
        # Few tokens seldom share an expert, so every row is its own block
        # and the alignment kernel is not needed.
        route = (None, topk_ids.to(torch.int32).view(-1), None)
    else:
        route = moe_align_block_size_rows(topk_ids, BLOCK_M, num_experts)
    flydsl_rdna_moe_gemm(
        act_out if silu_mul else gate_up_out,
        x,
        w1,
        coef1,
        *route,
        top_k,
        group_size,
        silu_mul=silu_mul,
        coef_pairs=w1_pairs,
    )
    if not silu_mul:
        activation(act_out, gate_up_out)
    if (
        route[0] is None
        and top_k <= MAX_REDUCE_ROWS
        and topk_weights.dtype == torch.float32
    ):
        # GEMM2 also does the weighted sum over the experts of a token.
        flydsl_rdna_moe_gemm_reduce(
            output,
            act_out,
            w2,
            coef2,
            route[1],
            topk_weights,
            group_size,
            coef_pairs=w2_pairs,
        )
        return
    flydsl_rdna_moe_gemm(
        down_out, act_out, w2, coef2, *route, 1, group_size, coef_pairs=w2_pairs
    )
    # Rows are in routing order, so the top-k sum needs no permutation.
    weighted = down_out.view(M, top_k, K) * topk_weights.view(M, top_k, 1)
    output.copy_(weighted.sum(dim=1))

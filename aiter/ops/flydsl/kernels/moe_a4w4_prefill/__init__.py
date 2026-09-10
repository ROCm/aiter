# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fp4 x fp4 prefill MoE chain for gfx950.

Replaces the sort, stage-1 GEMM, intermediate quant and stage-2 GEMM of aiter's
``QuantType.per_1x32`` fp4 x fp4 ``fused_moe`` chain, keeping aiter's per-token
quant (``fused_dynamic_mx_quant_moe_sort``) and its top-k reduction, and
producing the same bits. Faster from ``MIN_PREFILL_TOKENS`` up; below that
aiter's small-batch configuration wins, so callers must gate on it.
"""

import functools
import os

import torch

MIN_PREFILL_TOKENS = 3072
MAX_PREFILL_TOKENS = 65536
BM256_FROM_TOKENS = 16384
GEMM2_N_SPLIT = 2
_GEMM1_BLOCK_K = 256
_GEMM2_INTERMEDIATE = 768


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """gemm1 unrolls the K loop by 4 steps of 256 after 4 peeled ones; gemm2's
    flat pipeline is written for K = 768 (3 steps)."""
    k_iters = hidden_size // _GEMM1_BLOCK_K
    return (
        hidden_size % _GEMM1_BLOCK_K == 0
        and k_iters >= 8
        and (k_iters - 4) % 4 == 0
        and intermediate_size == _GEMM2_INTERMEDIATE
    )


def block_m_for(n_tokens: int) -> int:
    return 256 if n_tokens >= BM256_FROM_TOKENS else 128


def _run_compiled(exe, *args):
    """First call compiles and runs (``flyc.compile``); later calls dispatch the
    cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is None:
        exe._cf = flyc.compile(exe, *args)
    else:
        cf(*args)


@functools.cache
def _get_sort(num_experts: int, topk: int, block_m: int):
    from .sort import compile_moe_sort

    return compile_moe_sort(E=num_experts, topk=topk, block_m=block_m)


@functools.cache
def _get_tile_map(intermediate_size: int, block_m: int):
    from .tile_map import compile_tile_map

    return compile_tile_map(I=intermediate_size, BM=block_m)


@functools.cache
def _get_reduce_bf16(hidden_size: int, topk: int):
    from .reduce_bf16 import compile_moe_reduce_bf16

    return compile_moe_reduce_bf16(H=hidden_size, topk=topk)


@functools.cache
def _get_gemm1(
    hidden_size: int, intermediate_size: int, num_experts: int, block_m: int
):
    from .gemm1 import compile_moe_gemm1

    return compile_moe_gemm1(
        H=hidden_size, I=intermediate_size, E=num_experts, BLOCK_M=block_m
    )


@functools.cache
def _get_gemm2(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    block_m: int,
    fp8_pitch: int | None = None,
):
    from .gemm2 import compile_moe_gemm2

    return compile_moe_gemm2(
        H=hidden_size,
        I=intermediate_size,
        E=num_experts,
        topk=topk,
        n_split=GEMM2_N_SPLIT,
        out_dtype="bf16" if fp8_pitch is None else "fp8",
        sort_block_m=block_m,
        fp8_pitch=fp8_pitch,
    )


def _fp8_partial_layout(hidden_size: int):
    """Row pitch of aiter's packed fp8 partial buffer when ``AITER_FLYDSL_STAGE2_FP8``
    selects the mxfp8 route-out, else None. gemm2 then writes that layout and
    aiter's own ``moe_reduction`` consumes it, halving the reduce traffic.
    gemm2's epilogue emits one e8m0 per 32 columns, so only that block works."""
    if os.environ.get("AITER_FLYDSL_STAGE2_FP8", "0") != "1":
        return None
    from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
        FP8OUT_PITCH_ALIGN,
        fp8out_row_bytes,
        fp8out_scale_blk,
    )

    if hidden_size % 32 or fp8out_scale_blk(hidden_size) != 32:
        return None
    pitch = fp8out_row_bytes(hidden_size, scale_blk=32, pitch_align=FP8OUT_PITCH_ALIGN)
    return pitch, 32, FP8OUT_PITCH_ALIGN


def _u8_flat(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.uint8).view(-1)


def a4w4_prefill_moe(
    x: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """One MoE layer, fp4 activations x fp4 weights.

    ``w13`` ``[E, 2I, H/2]`` (gate rows then up rows, ``shuffle_weight(16, 16)``),
    ``w2`` ``[E, H, I/2]`` and their ``e8m0_shuffle`` scales are the tensors
    aiter's per_1x32 fp4 path stores; only their data pointers are used.
    ``topk_ids`` / ``topk_weights`` are ``[M, topk]``. Returns ``[M, H]`` bf16.
    """
    from aiter.ops.quant import fused_dynamic_mx_quant_moe_sort

    from .gemm2 import gemm2_grid
    from .sort import SortBuffers
    from .tile_map import tile_map_grid

    n_tokens, hidden = x.shape
    assert hidden == hidden_size and x.dtype == torch.bfloat16 and x.is_contiguous()
    topk = topk_ids.shape[1]
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    device = x.device
    stream = torch.cuda.current_stream()
    bm = block_m_for(n_tokens)
    inter = intermediate_size

    # 1. routing sort (aiter moe_sorting contract, block size bm)
    bufs = SortBuffers.allocate(n_tokens, num_experts, topk, bm, device)
    _get_sort(num_experts, topk, bm)(
        *bufs.launch_args(topk_ids, topk_weights, n_tokens)
    )
    # 2. aiter's per-token fp4 quant + activation scales in sorted, shuffled order
    a_q, a_s = fused_dynamic_mx_quant_moe_sort(
        x,
        bufs.sorted_ids,
        bufs.num_valid_ids,
        token_num=n_tokens,
        topk=topk,
        block_size=bm,
    )
    num_m_blocks = bufs.max_sorted // bm
    rows = num_m_blocks * bm
    # 3. gemm1 work list
    grid1 = tile_map_grid(num_m_blocks, inter)
    tile_map = torch.empty((grid1 + 1,), dtype=torch.int32, device=device)
    _run_compiled(
        _get_tile_map(inter, bm),
        bufs.sorted_expert_ids,
        bufs.num_valid_ids,
        tile_map,
        grid1,
        stream,
    )
    # 4. gemm1: fp4 intermediate [rows, I/2] + e8m0 scales, sorted rows
    h_q = torch.empty((rows, inter // 2), dtype=torch.uint8, device=device)
    h_s = torch.empty((rows * (inter // 32),), dtype=torch.uint8, device=device)
    _run_compiled(
        _get_gemm1(hidden_size, inter, num_experts, bm),
        _u8_flat(a_q),
        _u8_flat(w13),
        h_q.view(-1),
        _u8_flat(a_s),
        _u8_flat(w13_scale),
        h_s,
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        n_tokens,
        num_m_blocks,
        int(a_s.numel() * a_s.element_size()),
        tile_map,
        grid1,
        stream,
    )
    # 5. gemm2: routing-weighted partials [M, topk, H]
    fp8 = _fp8_partial_layout(hidden_size)
    if fp8 is None:
        partial = torch.empty(
            (n_tokens * topk, hidden_size), dtype=torch.bfloat16, device=device
        )
        partial_flat = partial.view(-1)
        partial_scale = partial_flat  # unused in bf16 mode
    else:
        # one spare row: the scale view starts at +hidden_size, so its last write
        # reaches hidden_size bytes past the logical end of the last row
        partial = torch.empty(
            (n_tokens * topk + 1, fp8[0]), dtype=torch.uint8, device=device
        )
        partial_flat = partial.view(-1)
        partial_scale = partial_flat[hidden_size:]
    num_m_blocks2 = rows // 128
    grid2 = gemm2_grid(num_m_blocks2, GEMM2_N_SPLIT)
    _run_compiled(
        _get_gemm2(
            hidden_size, inter, num_experts, topk, bm, None if fp8 is None else fp8[0]
        ),
        h_q.view(-1),
        _u8_flat(w2),
        partial_flat,
        h_s,
        _u8_flat(w2_scale),
        partial_scale,
        bufs.sorted_ids,
        bufs.sorted_expert_ids,
        bufs.sorted_weights,
        bufs.num_valid_ids,
        n_tokens,
        num_m_blocks2,
        grid2,
        stream,
    )
    # 6. top-k reduction
    if out is None:
        out = torch.empty((n_tokens, hidden_size), dtype=torch.bfloat16, device=device)
    if fp8 is None:
        # fp32 sum of the bf16 rows -> bf16
        _run_compiled(
            _get_reduce_bf16(hidden_size, topk),
            partial_flat,
            out.view(-1),
            n_tokens,
            stream,
        )
    else:
        # aiter's own fp8 reduction: gemm2 wrote its packed layout and deferred
        # the routing weights, which is exactly this kernel's contract
        from aiter.ops.flydsl.moe_kernels import _run_moe_reduction

        _run_moe_reduction(
            partial_flat,
            out,
            n_tokens,
            topk,
            hidden_size,
            stream=stream,
            is_fp8=True,
            topk_weights=topk_weights,
            fp8_scale_blk=fp8[1],
            fp8_pitch_align=fp8[2],
        )
    return out


__all__ = [
    "MAX_PREFILL_TOKENS",
    "MIN_PREFILL_TOKENS",
    "a4w4_prefill_moe",
    "supports_shapes",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project (ported to ATOM)
"""FlyDSL W4A4 MoE prefill chain at TP4 on the ``AITER_MXFP4_MXFP4`` weights: sort,
aiter's scale sort, gemm1 (gate/up + SwiGLU-OAI + MXFP4 quant), gemm2
(routing-weighted partials), fixed-order reduce.
"""

import functools

import torch

from aiter.ops.flydsl.kernels.minimax_m3_moe_prefill.moe_flydsl_common.launch import (
    _get_reduce_bf16,
    _get_sort,
    _get_tile_map,
    _u8_flat,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled as run_compiled

MIN_PREFILL_TOKENS = 3072
MAX_PREFILL_TOKENS = 65536
BM256_FROM_TOKENS = 16384
GEMM2_N_SPLIT = 2
_GEMM1_BLOCK_K = 256
# gemm2's K pipeline is written for 3 K-steps of 256
_GEMM2_INTERMEDIATE = 768
# aiter's scale sort (`mxfp4_moe_sort_hip`, csrc/kernels/quant_kernels.cu) rejects wider rows
_MAX_SORT_HIDDEN = 16384
# the device of this process's first a4w4_prefill_moe
_DEVICE: list[torch.device] = []


def supports_shapes(hidden_size: int, intermediate_size: int) -> bool:
    """gemm1 peels 4 K-steps of 256 and unrolls the rest by 4; the scale sort takes
    hidden sizes up to 16384."""
    k_iters = hidden_size // _GEMM1_BLOCK_K
    return (
        hidden_size % _GEMM1_BLOCK_K == 0
        and hidden_size <= _MAX_SORT_HIDDEN
        and k_iters >= 8
        and (k_iters - 4) % 4 == 0
        and intermediate_size == _GEMM2_INTERMEDIATE
    )


def block_m_for(n_tokens: int) -> int:
    return 256 if n_tokens >= BM256_FROM_TOKENS else 128


@functools.cache
def _get_gemm1(hidden_size: int, intermediate_size: int, num_experts: int, bm: int):
    from .gemm1_prefill import compile_moe_gemm1

    return compile_moe_gemm1(
        H=hidden_size, I=intermediate_size, E=num_experts, BLOCK_M=bm
    )


@functools.cache
def _get_gemm2(
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    bm: int,
):
    from .gemm2_prefill import compile_moe_gemm2

    return compile_moe_gemm2(
        H=hidden_size,
        I=intermediate_size,
        E=num_experts,
        topk=topk,
        n_split=GEMM2_N_SPLIT,
        sort_block_m=bm,
    )


def a4w4_prefill_moe(
    x_quant: tuple[torch.Tensor, torch.Tensor],
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
    """``x_quant`` = (fp4 ``[M, H/2]``, e8m0 ``[M, H/32]``) of aiter's per-token
    MXFP4 quant -> ``[M, H]`` bf16, on ``AiterMxfp4Experts``' weights."""
    from aiter import dtypes
    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.flydsl.kernels.minimax_m3_moe_prefill.moe_flydsl_common.sort import (
        SortBuffers,
        max_sorted_rows,
    )
    from aiter.ops.flydsl.kernels.minimax_m3_moe_prefill.moe_flydsl_common.tile_map import (
        _MAX_BLOCKS,
        tile_map_grid,
    )
    from aiter.ops.quant import mxfp4_moe_sort_hip

    from .gemm2_prefill import gemm2_grid

    # the GEMM2 pipeline needs 156 KiB of LDS and gfx950's scaled FP4 MFMA; gfx942 has neither
    assert get_gfx_runtime() == "gfx950", get_gfx_runtime()
    a_q, x_s = x_quant
    n_tokens = a_q.shape[0]
    hidden = hidden_size
    assert MIN_PREFILL_TOKENS <= n_tokens <= MAX_PREFILL_TOKENS, n_tokens
    assert a_q.shape == (n_tokens, hidden // 2) and x_s.shape == (
        n_tokens,
        hidden // 32,
    )
    # byte-wise views below: reject other dtypes (a bf16 tensor would be read as raw bytes)
    # and strided views (the kernels address them flat)
    codes, scales = (torch.uint8, dtypes.fp4x2), (torch.uint8, dtypes.fp8_e8m0)
    assert all(t.dtype in codes for t in (a_q, w13, w2)), (
        a_q.dtype,
        w13.dtype,
        w2.dtype,
    )
    assert all(t.dtype in scales for t in (x_s, w13_scale, w2_scale))
    assert all(t.is_contiguous() for t in (a_q, x_s, w13, w13_scale, w2, w2_scale))
    inter = intermediate_size
    topk = topk_ids.shape[1]
    assert supports_shapes(hidden, inter), (hidden, inter)
    assert topk_ids.shape == topk_weights.shape == (n_tokens, topk)
    assert w13.shape == (num_experts, 2 * inter, hidden // 2)
    assert w2.shape == (num_experts, hidden, inter // 2)
    # the scale buffers are bounded by these sizes, not by the tensors (preshuffling may pad)
    assert w13_scale.numel() >= num_experts * 2 * inter * (hidden // 32)
    assert w2_scale.numel() >= num_experts * hidden * (inter // 32)
    assert all(
        t.is_cuda and t.device == a_q.device
        for t in (x_s, topk_weights, w13, w13_scale, w2, w2_scale)
    )
    # Buffer offsets are unsigned 32-bit and a store past 2^32 wraps to offset 0, inside the
    # buffer. The partial rows end with the padding sentinel (token n_tokens, row n_tokens *
    # topk), so the last byte is (n_tokens * topk + 1) * hidden * 2 - 1. Op test: 65536 tokens
    # at topk 4 is 3.2e9 bytes. shortcut: this only rules out the wrap; re-measure to raise it.
    assert (n_tokens * topk + 1) * hidden * 2 <= 2**32, (n_tokens, topk, hidden)
    for name, t in (
        ("w13", w13),
        ("w13_scale", w13_scale),
        ("w2", w2),
        ("w2_scale", w2_scale),
    ):
        assert t.numel() * t.element_size() < 2**32, (name, t.numel())
    # the tile map and the m-block search cover _MAX_BLOCKS blocks
    bm = block_m_for(n_tokens)
    assert max_sorted_rows(n_tokens, num_experts, topk, bm) // bm <= _MAX_BLOCKS
    if out is not None:
        assert out.shape == (n_tokens, hidden) and out.dtype == torch.bfloat16
        assert out.is_contiguous() and out.device == a_q.device
    topk_ids = topk_ids.to(torch.int32).contiguous()
    topk_weights = topk_weights.to(torch.float32).contiguous()
    device = a_q.device
    assert topk_ids.device == device
    # shortcut: FlyDSL loads a compiled kernel on the first device that runs it and a later
    # device gets hipErrorInvalidDevice, so one process drives one GPU; upgrade with FlyDSL.
    if not _DEVICE:
        _DEVICE.append(device)
    assert device == _DEVICE[0], ("device", device, "kernels loaded on", _DEVICE[0])
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream()

        bufs = SortBuffers.allocate(n_tokens, num_experts, topk, bm, device)
        run_compiled(
            _get_sort(num_experts, topk, bm),
            *bufs.launch_args(topk_ids, topk_weights, n_tokens),
        )
        a_s = torch.empty(
            ((bufs.sorted_ids.shape[0] + 31) // 32 * 32, (hidden // 32 + 7) // 8 * 8),
            dtype=dtypes.fp8_e8m0,
            device=device,
        )
        mxfp4_moe_sort_hip(
            a_s,
            x_s.view(dtypes.fp8_e8m0),
            bufs.sorted_ids,
            bufs.num_valid_ids,
            n_tokens,
            hidden,
        )
        num_m_blocks = bufs.max_sorted // bm
        rows = num_m_blocks * bm
        grid1 = tile_map_grid(num_m_blocks, inter)
        tile_map = torch.empty((grid1 + 1,), dtype=torch.int32, device=device)
        run_compiled(
            _get_tile_map(inter, bm),
            bufs.sorted_expert_ids,
            bufs.num_valid_ids,
            tile_map,
            grid1,
            stream,
        )
        h_q = torch.empty((rows, inter // 2), dtype=torch.uint8, device=device)
        h_s = torch.empty((rows * (inter // 32),), dtype=torch.uint8, device=device)
        run_compiled(
            _get_gemm1(hidden, inter, num_experts, bm),
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

        partial = torch.empty(
            (n_tokens * topk, hidden), dtype=torch.bfloat16, device=device
        )
        num_m_blocks2 = rows // 128
        run_compiled(
            _get_gemm2(hidden, inter, num_experts, topk, bm),
            h_q.view(-1),
            _u8_flat(w2),
            partial.view(-1),
            h_s,
            _u8_flat(w2_scale),
            bufs.sorted_ids,
            bufs.sorted_expert_ids,
            bufs.sorted_weights,
            bufs.num_valid_ids,
            n_tokens,
            num_m_blocks2,
            gemm2_grid(num_m_blocks2, GEMM2_N_SPLIT),
            stream,
        )
        if out is None:
            out = torch.empty((n_tokens, hidden), dtype=torch.bfloat16, device=device)
        run_compiled(
            _get_reduce_bf16(hidden, topk),
            partial.view(-1),
            out.view(-1),
            n_tokens,
            stream,
        )
        return out


def mxfp4_quant(
    x: torch.Tensor, q_out: torch.Tensor | None, scale_out: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token MXFP4 quant of bf16 ``[M, H]`` by aiter's kernel (the regular
    path's from 512 tokens) into ``q_out`` ``[M, H/2]`` / ``scale_out``
    ``[M, H/32]`` uint8."""
    from aiter import dtypes
    from aiter.ops.quant import dynamic_per_group_scaled_quant

    m, h = x.shape
    # the kernel reads x row by row through stride(-2) only
    assert x.is_cuda and x.stride(-1) == 1, ("x", x.device, x.stride())
    # the kernel sizes its stores by x, not by q_out / scale_out: a smaller caller buffer
    # is written past its end
    for name, t, cols, dts in (
        ("q_out", q_out, h // 2, (torch.uint8, dtypes.fp4x2)),
        ("scale_out", scale_out, h // 32, (torch.uint8, dtypes.fp8_e8m0)),
    ):
        if t is not None:
            assert t.shape == (m, cols) and t.dtype in dts, (name, t.shape, t.dtype)
            assert t.is_contiguous() and t.device == x.device, (name, t.device)
    if q_out is None:
        q_out = torch.empty((m, h // 2), dtype=torch.uint8, device=x.device)
    if scale_out is None:
        scale_out = torch.empty((m, h // 32), dtype=torch.uint8, device=x.device)
    dynamic_per_group_scaled_quant(
        q_out.view(dtypes.fp4x2),
        x,
        scale_out.view(dtypes.fp8_e8m0),
        32,
        shuffle_scale=False,
    )
    return q_out, scale_out

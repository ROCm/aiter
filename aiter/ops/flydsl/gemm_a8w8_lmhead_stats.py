# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The mxfp8_128 bpreshuffle GEMM of a draft LM head (gfx1250 FlyDSL, 16x256
tiles, 4 waves) with the top-k / top-p sampler's statistics in its epilogue.

Besides the bf16 logits, each 16 x 256 tile writes, from its bf16 tile in LDS,
for each row a 32-byte record (whole granules: partial-line writes shared by
neighbour tiles cost microseconds): the 32-column group maxima of the
presence-penalized logits as 16-bit ordered keys (the sampler stats kernel's
keys, bit for bit) and the tile's softmax partial at temperature t (max / t,
sum exp2((x - max) * log2(e) / t)), which the sampler's select
merges in place of its stats kernel's chunk partials
(``aiter.ops.triton.fusions.topk_topp_sample.topk_topp_sample(stats=...)``)."""

from __future__ import annotations

import torch

from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import BLOCK_K

TILE_M, TILE_N, TILE_K = 16, 256, 128
_M_WARP, _N_WARP, _NB = 1, 4, 2
_launch = None


def lmhead_stats_supported(m: int, n: int, k: int) -> bool:
    """Rows one 16-row tile holds, whole 256-column tiles, whole K tiles."""
    from aiter.jit.utils.chip_info import get_gfx

    return (
        get_gfx() == "gfx1250"
        and 0 < m <= TILE_M
        and n % TILE_N == 0
        and k % TILE_K == 0
        and k // TILE_K >= _NB
    )


def gemm_a8w8_lmhead_stats(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    out: torch.Tensor,
    temperatures: torch.Tensor,
    rec: torch.Tensor,
    presence: dict | None = None,
    rng_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """``out`` = XQ @ WQ^T (bf16) and the sampler statistics of ``out``.

    XQ [M, K] fp8 e4m3, x_scale (M, K/128) e8m0 with transposed bytes (as
    per_1x128_e8m0_quant(transpose_scale=True) writes), WQ [N, K] 16x16
    preshuffled, w_scale [N/128, K/128] e8m0. temperatures fp32 [M] (row r is
    request r). rec int32 [>= M, N/256, 8]: a 32-byte record a row and
    tile (8 int16 group keys, f32 max / t, f32 sum, 8 zero bytes). presence: the
    sampler's presence dict of a draft pass (DraftPresencePenalty: row r is
    request r, row_seq = arange, seq_start = row_seq * ctx_width; the drafts
    input_ids[seq_start + 1 .. seq_start + row_pos[r]]; no anchor). rng_state:
    int64 [1] counter, += M."""
    global _launch
    from aiter.utility import dtypes

    M, K = XQ.shape
    N = WQ.shape[0]
    assert lmhead_stats_supported(M, N, K), (M, N, K)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16 and out.stride(1) == 1
    assert x_scale.dtype == dtypes.fp8_e8m0 and tuple(x_scale.shape) == (M, K // BLOCK_K)
    assert w_scale.dtype == dtypes.fp8_e8m0
    assert tuple(w_scale.shape) == (N // 128, K // BLOCK_K)
    assert rec.dtype == torch.int32 and rec.is_contiguous()
    assert rec.shape[0] >= M and rec.shape[1:] == (N // TILE_N, 8)
    if _launch is None:
        from aiter.ops.flydsl.kernels.gemm_a8w8_stats_gfx1250 import (
            launch_gemm_a8w8_stats,
        )

        _launch = launch_gemm_a8w8_stats
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    penalty = presence is not None
    if penalty:
        assert not presence.get("anchor", False)
        mask = presence["mask"]
        assert mask.dtype == torch.uint8 and mask.stride(1) == 1
        assert mask.stride(0) % 16 == 0 and mask.data_ptr() % 16 == 0
        pen = (
            presence["input_ids"],
            presence["row_seq"],
            presence["row_pos"],
            presence["seq_start"],
            presence["slots"],
            presence["penalties"],
            mask,
        )
        mask_stride = mask.stride(0)
        max_extra = max(int(presence["max_extra"]), 1)
        # row r is request r, its drafts at input_ids[r * ctx_width + 1 ..]
        # (DraftPresencePenalty: row_seq = arange, seq_start = row_seq * width)
        ctx_width = presence["input_ids"].numel() // presence["row_seq"].numel()
        assert ctx_width * presence["row_seq"].numel() == presence["input_ids"].numel()
    else:
        pen = (temperatures,) * 7
        mask_stride = 0
        max_extra = 1
        ctx_width = 4
    rng = rng_state if rng_state is not None else temperatures
    stream = fx.Stream(torch.cuda.current_stream(device=XQ.device))
    stride_ascale_k = x_scale.stride(1) if x_scale.stride(0) == 1 else M
    spec = (
        TILE_M, TILE_N, TILE_K, _M_WARP, _N_WARP, 0, _NB, 1, 1, True, BLOCK_K,
        penalty, max_extra, rng_state is not None, ctx_width,
    )
    _run_compiled(
        _launch,
        ptr_arg(out),
        ptr_arg(XQ),
        ptr_arg(WQ),
        ptr_arg(x_scale),
        ptr_arg(w_scale),
        M,
        stream,
        N,
        K,
        stride_ascale_k,
        XQ.stride(0),
        out.stride(0),
        ptr_arg(rec),
        ptr_arg(rec),
        ptr_arg(rec),
        ptr_arg(temperatures),
        *[ptr_arg(t) for t in pen],
        ptr_arg(rng),
        mask_stride,
        *spec,
        specialization_key=spec,
    )
    return out

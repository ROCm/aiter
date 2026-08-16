# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 dense BF16 -> inline MXFP4 activation quantization x MXFP4 GEMM."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, rocdl
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch

from .mxfp4_gemm1 import _bm_constants, _gemm1_body

__all__ = ["compile_gemm_a4w4"]


@functools.cache
def compile_gemm_a4w4(*, N: int, K: int, BM: int = 16):
    """Compile a dense BN=256 inline-quantized launcher."""
    arch = str(get_rocm_arch()).split(":", 1)[0]
    if arch != "gfx950":
        raise ValueError(f"dense a4w4 FlyDSL kernel requires gfx950, got {arch!r}")
    if N <= 0 or N % 256:
        raise ValueError(f"N must be a positive multiple of 256, got {N}")
    if K <= 0 or K % 128:
        raise ValueError(f"K must be a positive multiple of 128, got {K}")
    if BM not in (16, 64):
        raise ValueError(f"BM must be 16 or 64, got {BM}")

    BN = 256
    BK = 256 if K % 256 == 0 else 128
    k_tiles = K // BK
    if k_tiles < 3:
        raise ValueError(f"K must contain at least three BK={BK} tiles, got K={K}")
    k_a_stages, k_scale_subblocks, _, lds_bytes = _bm_constants(
        BM, BN, BK // 2, k_tiles
    )
    if BM == 64:
        # Dense output is written directly from accm, so unlike the shared MoE
        # epilogue this specialization never needs a BM*BN f32 LDS accumulator.
        # Avoid reserving 64 KiB and retain only the quantization pipeline.
        lds_bytes = (
            k_a_stages * BM * (BK // 2) + k_scale_subblocks * ((K + 255) // 256) * 256
        )
    n_blocks = N // BN
    kernel_name = f"dense_a4w4_iq_n{N}_k{K}_bm{BM}_bn{BN}_bk{BK}"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[256, 1, 1])
    def kernel(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_out: fx.Int64,
        i32_m: fx.Int32,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = fx.Int32(gpu.thread_id("x"))
        block = fx.Int32(gpu.block_id("x"))
        lane = tx % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx // fx.Int32(64))

        _gemm1_body(
            lds_raw_ptr,
            arg_a,
            arg_a,
            arg_bq,
            arg_bscale,
            arg_out,
            arg_out,
            arg_out,
            arg_out,
            arg_a,
            arg_out,
            i32_m,
            block,
            lane,
            wave,
            True,
            i32_m,
            (i32_m + fx.Int32(BM - 1)) // fx.Int32(BM),
            BM=BM,
            BN=BN,
            BK=BK,
            inline_quant=True,
            K=K,
            N_OUT=N,
            NE=1,
            interleave=False,
            dense=True,
        )

    @flyc.jit
    def launch(
        arg_a: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_out: fx.Int64,
        i32_m: fx.Int32,
        stream: fx.Stream,
    ):
        grid = ((i32_m + BM - 1) // BM) * n_blocks
        kernel(arg_a, arg_bq, arg_bscale, arg_out, i32_m).launch(
            grid=(fx.Int64(grid), 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    return launch

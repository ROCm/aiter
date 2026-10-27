# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone BF16 RMSNorm for gfx1250 MegaMoE route rows."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, ReductionOp, T

from .mega_moe_tp.common import bld, bst, lds_ld, lds_st, rsrc

_WAVE = 32
_VEC = 8  # 8 BF16 values -=> one 128-bit load/store.


@functools.cache
def build_rmsnorm_gfx1250(hidden: int):
    """Build one block-per-row RMSNorm."""
    if hidden <= 0 or hidden % _VEC:
        raise ValueError(f"hidden must be a positive multiple of {_VEC}, got {hidden}")

    vectors = hidden // _VEC
    threads = 32
    while threads < vectors and threads < 256:
        threads *= 2
    waves = threads // _WAVE
    vectors_per_thread = (vectors + threads - 1) // threads

    @flyc.kernel(
        name=f"rmsnorm_bf16_h{hidden}_gfx1250",
        known_block_size=[threads, 1, 1],
    )
    def kernel(
        input_addr: Int64,
        output_addr: Int64,
        weight_addr: Int64,
        input_row_stride: Int32,
        epsilon: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(_WAVE)
        wave = tid // fx.Int32(_WAVE)
        row = fx.Int32(gpu.block_id("x"))

        input_rsrc = rsrc(input_addr)
        output_rsrc = rsrc(output_addr)
        weight_rsrc = rsrc(weight_addr)

        shared = fx.SharedAllocator(static=False).allocate((waves + 1) * 4)._ptr
        lds = fx.Int32(fx.ptrtoint(shared))

        cached = []
        local_sum = fx.Float32(0.0)
        for step in range_constexpr(vectors_per_thread):
            vector_idx = tid + fx.Int32(step * threads)
            live = vector_idx < fx.Int32(vectors)
            safe_idx = live.select(vector_idx, fx.Int32(0))
            input_byte = (
                row * input_row_stride + safe_idx * fx.Int32(_VEC)
            ) * fx.Int32(2)
            raw = (
                fx.Vector(bld(input_rsrc, input_byte, 0, T.vec(4, T.i32)))
                .bitcast(fx.BFloat16)
                .to(fx.Float32)
            )
            values = fx.Vector.from_elements(
                [live.select(raw[i], fx.Float32(0.0)) for i in range_constexpr(_VEC)],
                fx.Float32,
            )
            cached.append(values)
            local_sum = local_sum + (values * values).reduce(
                ReductionOp.ADD, fx.Float32(0.0)
            )

        for offset in (16, 8, 4, 2, 1):
            local_sum = local_sum + local_sum.shuffle_xor(offset, _WAVE)
        if lane == fx.Int32(0):
            lds_st(lds, fx.Int32(4) + wave * fx.Int32(4), local_sum)
        gpu.barrier()

        block_sum = fx.Float32(0.0)
        if wave == fx.Int32(0):
            live_wave = lane < fx.Int32(waves)  # clamp
            safe_wave = live_wave.select(lane, fx.Int32(0))
            block_sum = live_wave.select(
                fx.Float32(
                    lds_ld(lds, fx.Int32(4) + safe_wave * fx.Int32(4), T.f32)
                ),
                fx.Float32(0.0),
            )
            for offset in (16, 8, 4, 2, 1):
                block_sum = block_sum + block_sum.shuffle_xor(offset, _WAVE)
            if lane == fx.Int32(0):
                lds_st(lds, fx.Int32(0), block_sum)
        gpu.barrier()

        inverse_rms = fmath.rsqrt(
            fx.Float32(lds_ld(lds, fx.Int32(0), T.f32)) * fx.Float32(1.0 / hidden)
            + epsilon
        )
        for step in range_constexpr(vectors_per_thread):
            vector_idx = tid + fx.Int32(step * threads)
            if vector_idx < fx.Int32(vectors):
                weight_byte = vector_idx * fx.Int32(_VEC * 2)
                weight = (
                    fx.Vector(bld(weight_rsrc, weight_byte, 0, T.vec(4, T.i32)))
                    .bitcast(fx.BFloat16)
                    .to(fx.Float32)
                )
                output = (
                    (cached[step] * weight * inverse_rms)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int32)
                )
                output_byte = (
                    row * fx.Int32(hidden) + vector_idx * fx.Int32(_VEC)
                ) * fx.Int32(2)
                # Allow combine kernel to reuse results from shared memory
                bst(output, output_rsrc, output_byte, 0, 1)

    @flyc.jit
    def launch(
        input_addr: Int64,
        output_addr: Int64,
        weight_addr: Int64,
        rows: Int32,
        input_row_stride: Int32,
        epsilon: fx.Float32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            input_addr,
            output_addr,
            weight_addr,
            input_row_stride,
            epsilon,
        ).launch(
            grid=(fx.Int64(rows), 1, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return launch


def rmsnorm_flydsl_out(
    out: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> None:
    """Run gfx1250 FlyDSL RMSNorm into a caller-provided contiguous output."""
    if input.dim() != 2 or input.stride(1) != 1:
        raise ValueError("rmsnorm_flydsl_out requires a 2D input with contiguous rows")
    rows, hidden = input.shape
    if tuple(out.shape) != (rows, hidden):
        raise ValueError("rmsnorm_flydsl_out output shape must match input")
    if input.dtype != torch.bfloat16:
        raise ValueError("rmsnorm_flydsl_out supports only bfloat16")
    if out.dtype != input.dtype or not out.is_contiguous():
        raise ValueError("rmsnorm_flydsl_out requires a contiguous bfloat16 output")
    if tuple(weight.shape) != (hidden,) or weight.dtype != input.dtype:
        raise ValueError(
            "rmsnorm_flydsl_out weight must match the input width and dtype"
        )
    if not weight.is_contiguous():
        raise ValueError("rmsnorm_flydsl_out weight must be contiguous")
    if rows == 0:
        return

    build_rmsnorm_gfx1250(hidden)(
        input.data_ptr(),
        out.data_ptr(),
        weight.data_ptr(),
        rows,
        input.stride(0),
        float(epsilon),
        fx.Stream(torch.cuda.current_stream()),
    )

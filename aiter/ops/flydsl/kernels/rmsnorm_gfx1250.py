# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standalone BF16 input-scale RMSNorm for gfx1250 MoE route rows.

``y = u / sqrt(mean(u^2) + eps)`` with ``u = bf16(x * w)`` and no gain after
the norm; the scaled input is rounded to bf16 once before the RMS. With
``zero_centered`` the weight is stored as ``gamma`` and applied as ``1 + gamma``.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, ReductionOp, T

from .mega_moe_tp.common import bld, bst, lds_ld, lds_st, rsrc

_WAVE = 32
_VEC = 8  # 8 BF16 values -=> one 128-bit load/store.


@functools.cache
def build_rmsnorm_gfx1250(hidden: int, row_map: bool = False):
    """Build one block-per-row RMSNorm.

    With ``row_map`` block ``r`` normalizes row ``row_map[r]`` instead of row
    ``r``, skipping negative entries (dropped routes) and routes of tokens at or
    past ``num_valid_tokens`` (whose map entries are unwritten). A skipped block
    gets zero-length descriptors, so it neither reads nor writes memory.
    """
    if hidden <= 0 or hidden % _VEC:
        raise ValueError(f"hidden must be a positive multiple of {_VEC}, got {hidden}")

    vectors = hidden // _VEC
    threads = 32
    while threads < vectors and threads < 256:
        threads *= 2
    waves = threads // _WAVE
    vectors_per_thread = (vectors + threads - 1) // threads
    row_bytes = hidden * 2

    @flyc.kernel(
        name=f"rmsnorm_bf16_h{hidden}{'_rowmap' if row_map else ''}_gfx1250",
        known_block_size=[threads, 1, 1],
    )
    def kernel(
        input_addr: Int64,
        output_addr: Int64,
        weight_addr: Int64,
        row_map_addr: Int64,
        num_valid_tokens_addr: Int64,
        input_row_stride: Int32,
        output_row_stride: Int32,
        routes_per_token: Int32,
        epsilon: fx.Float32,
        gain_center: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % fx.Int32(_WAVE)
        wave = tid // fx.Int32(_WAVE)
        block = fx.Int32(gpu.block_id("x"))  # with map, which route.

        row = block  # no map, so block r does row r
        live_bytes = fx.Int32(row_bytes)
        if const_expr(row_map):
            # look up this route's row
            mapped = fx.Int32(bld(rsrc(row_map_addr), block * fx.Int32(4), 0, T.i32))
            valid_tokens = fx.Int32(0x7FFFFFFF)  # no num_valid_tokens => skip load
            if num_valid_tokens_addr != fx.Int64(0):
                valid_tokens = fx.Int32(bld(rsrc(num_valid_tokens_addr), 0, 0, T.i32))
            live = (block // routes_per_token < valid_tokens) & (mapped >= 0)
            row = live.select(mapped, fx.Int32(0))  # with a map, row = map[block]
            live_bytes = live.select(fx.Int32(row_bytes), fx.Int32(0))

        row_i64 = fx.Int64(row)  # allow some room to prevent overflow
        input_rsrc = rsrc(  # rsrc == buffer
            input_addr + row_i64 * fx.Int64(input_row_stride) * fx.Int64(2),
            live_bytes,
        )
        output_rsrc = rsrc(
            output_addr + row_i64 * fx.Int64(output_row_stride) * fx.Int64(2),
            live_bytes,
        )
        weight_rsrc = rsrc(weight_addr)

        shared = fx.SharedAllocator(static=False).allocate((waves + 1) * 4)._ptr
        lds = fx.Int32(fx.ptrtoint(shared))

        cached = []
        local_sum = fx.Float32(0.0)
        for step in range_constexpr(vectors_per_thread):
            vector_idx = tid + fx.Int32(step * threads)
            live_vec = vector_idx < fx.Int32(vectors)
            safe_idx = live_vec.select(vector_idx, fx.Int32(0))
            byte = safe_idx * fx.Int32(_VEC * 2)
            raw = (
                fx.Vector(bld(input_rsrc, byte, 0, T.vec(4, T.i32)))
                .bitcast(fx.BFloat16)
                .to(fx.Float32)
            )
            weight = (
                fx.Vector(bld(weight_rsrc, byte, 0, T.vec(4, T.i32)))
                .bitcast(fx.BFloat16)
                .to(fx.Float32)
            )
            # Scale before the norm and round the scaled input to bf16 once; the
            # sum of squares is taken over bf16(x * w).
            values = (
                fx.Vector.from_elements(
                    [
                        live_vec.select(
                            raw[i] * (weight[i] + gain_center), fx.Float32(0.0)
                        )
                        for i in range_constexpr(_VEC)
                    ],
                    fx.Float32,
                )
                .to(fx.BFloat16)
                .to(fx.Float32)
            )
            cached.append(values)
            local_sum = local_sum + (values * values).reduce(
                ReductionOp.ADD, fx.Float32(0.0)
            )

        # sum across waves
        for offset in (16, 8, 4, 2, 1):
            local_sum = local_sum + local_sum.shuffle_xor(offset, _WAVE)
        if lane == fx.Int32(0):  # wave 0 stores to lds
            lds_st(lds, fx.Int32(4) + wave * fx.Int32(4), local_sum)
        gpu.barrier()

        # sum across block
        block_sum = fx.Float32(0.0)
        if wave == fx.Int32(0):
            live_wave = lane < fx.Int32(waves)  # clamp
            safe_wave = live_wave.select(lane, fx.Int32(0))
            block_sum = live_wave.select(
                fx.Float32(lds_ld(lds, fx.Int32(4) + safe_wave * fx.Int32(4), T.f32)),
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
                output = (cached[step] * inverse_rms).to(fx.BFloat16).bitcast(fx.Int32)
                # Allow combine kernel to reuse results from shared memory
                bst(output, output_rsrc, vector_idx * fx.Int32(_VEC * 2), 0, 1)

    @flyc.jit
    def launch(
        input_addr: Int64,
        output_addr: Int64,
        weight_addr: Int64,
        row_map_addr: Int64,
        num_valid_tokens_addr: Int64,
        rows: Int32,
        input_row_stride: Int32,
        output_row_stride: Int32,
        routes_per_token: Int32,
        epsilon: fx.Float32,
        gain_center: fx.Float32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            input_addr,
            output_addr,
            weight_addr,
            row_map_addr,
            num_valid_tokens_addr,
            input_row_stride,
            output_row_stride,
            routes_per_token,
            epsilon,
            gain_center,
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
    *,
    zero_centered: bool = False,
    row_map: torch.Tensor | None = None,
    num_valid_tokens: torch.Tensor | None = None,
) -> None:
    """RMSNorm ``rms(bf16(x * w))`` of bf16 rows into ``out`` (gfx1250).

    With ``row_map`` (``topids_to_rows``) only the routed rows are normalized;
    ``DROPPED_ROUTE_ROW`` entries and tokens past ``num_valid_tokens`` are skipped.
    ``out`` may be ``input`` for an in-place norm if each row appears once in the map.
    """
    if input.dim() != 2 or input.stride(1) != 1:
        raise ValueError("rmsnorm_flydsl_out requires a 2D input with contiguous rows")
    rows, hidden = input.shape
    if tuple(out.shape) != (rows, hidden):
        raise ValueError("rmsnorm_flydsl_out output shape must match input")
    if input.dtype != torch.bfloat16:
        raise ValueError("rmsnorm_flydsl_out supports only bfloat16")
    if out.dtype != input.dtype or out.stride(1) != 1:
        raise ValueError(
            "rmsnorm_flydsl_out requires a bfloat16 output with contiguous rows"
        )
    if tuple(weight.shape) != (hidden,) or weight.dtype != input.dtype:
        raise ValueError(
            "rmsnorm_flydsl_out weight must match the input width and dtype"
        )
    if not weight.is_contiguous():
        raise ValueError("rmsnorm_flydsl_out weight must be contiguous")

    map_ptr = count_ptr = 0
    routes_per_token = 1
    if row_map is not None:
        if row_map.dim() != 2 or row_map.dtype != torch.int32:
            raise ValueError("rmsnorm_flydsl_out row_map must be a 2D int32 tensor")
        if not row_map.is_contiguous():
            raise ValueError("rmsnorm_flydsl_out row_map must be contiguous")
        tokens, routes_per_token = row_map.shape
        rows = tokens * routes_per_token
        map_ptr = row_map.data_ptr()
        if num_valid_tokens is not None:
            if num_valid_tokens.dtype != torch.int32 or num_valid_tokens.numel() < 1:
                raise ValueError("num_valid_tokens must be an int32 tensor")
            count_ptr = num_valid_tokens.data_ptr()
    elif num_valid_tokens is not None:
        raise ValueError("num_valid_tokens requires a row_map")
    if rows == 0:
        return

    build_rmsnorm_gfx1250(hidden, row_map is not None)(
        input.data_ptr(),
        out.data_ptr(),
        weight.data_ptr(),
        map_ptr,
        count_ptr,
        rows,
        input.stride(0),
        out.stride(0),
        routes_per_token,
        float(epsilon),
        1.0 if zero_centered else 0.0,
        fx.Stream(torch.cuda.current_stream()),
    )

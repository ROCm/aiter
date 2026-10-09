# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""K16 E4M3 NVFP4 quantization into the grouped GEMM activation layout."""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import Constexpr, T
from flydsl.expr.typing import Vector as Vec

from .quant_utils import emit_cvt_scalef32_pk8_fp4_f32, emit_nvfp4_scale


@flyc.jit
def launch_nvfp4_quant(
    source: fx.Tensor,
    payload: fx.Tensor,
    scales: fx.Tensor,
    global_scale: fx.Tensor,
    route_rows: fx.Tensor,
    route_experts: fx.Tensor,
    psum: fx.Tensor,
    rows: Constexpr[int],
    features: Constexpr[int],
    wmma_rep: Constexpr[int],
    topk: Constexpr[int],
    experts: Constexpr[int],
    stream: fx.Stream,
):
    """Quantize routed tokens (topk > 0) or contiguous expert rows (topk=0)."""
    blocks_per_row = features // 16
    row_tile = wmma_rep * 16
    scale_dwords = features // 64
    name = f"nvfp4_quant_k{features}_r{wmma_rep}_topk{topk}_e{experts}"

    @flyc.kernel(name=name, known_block_size=[128, 1, 1])
    def kernel(
        source: fx.Pointer,
        payload: fx.Pointer,
        scales: fx.Pointer,
        global_scale: fx.Pointer,
        route_rows: fx.Pointer,
        route_experts: fx.Pointer,
        psum: fx.Pointer,
    ):
        index = fx.block_idx.x * 128 + fx.thread_idx.x
        row = index // blocks_per_row
        block = index % blocks_per_row
        if row < rows:
            source_row = row // topk if topk else row
            dst_row = fx.recast_iter(fx.Int32, route_rows)[row] if topk else row
            if const_expr(topk):
                expert = fx.recast_iter(fx.Int32, route_experts)[row]
                valid = dst_row >= 0
            else:
                ends = fx.recast_iter(fx.Int32, psum)
                lo, hi = row * 0, row * 0 + experts
                for _ in range_constexpr(math.ceil(math.log2(experts + 1)) + 1):
                    mid = (lo + hi) >> 1
                    safe_mid = (mid < experts).select(mid, experts - 1)
                    right = ends[safe_mid] <= row
                    lo = right.select(mid + 1, lo)
                    hi = right.select(hi, mid)
                expert = lo
                valid = row < ends[experts - 1]
            if valid:
                values = Vec(
                    fx.ptr_load(
                        fx.recast_iter(fx.BFloat16, source)
                        + source_row * features
                        + block * 16,
                        result_type=T.vec(16, T.bf16),
                    )
                ).to(fx.Float32)
                global_value = fx.recast_iter(fx.Float32, global_scale)[expert]
                reciprocal, scale_byte = emit_nvfp4_scale(
                    [values[i] for i in range_constexpr(16)], global_value
                )
                normalized = values * reciprocal
                output = fx.recast_iter(
                    fx.PointerType.get(
                        elem_ty=T.i32,
                        address_space=fx.AddressSpace.Global,
                        alignment=4,
                    ),
                    payload,
                )
                for half in range_constexpr(2):
                    part = Vec.from_elements(
                        [normalized[half * 8 + i] for i in range_constexpr(8)],
                        fx.Float32,
                    )
                    packed = emit_cvt_scalef32_pk8_fp4_f32(
                        part.ir_value(),
                        fx.Float32(1.0),
                        i32_ty=T.i32,
                        rocdl=rocdl,
                    )
                    output[dst_row * (features // 8) + block * 2 + half] = packed
                offset = (
                    (dst_row // row_tile * scale_dwords + block // 4) * row_tile
                    + dst_row % row_tile
                ) * 4 + block % 4
                fx.recast_iter(fx.Int8, scales)[offset] = scale_byte

    kernel(
        fx.get_iter(source),
        fx.get_iter(payload),
        fx.get_iter(scales),
        fx.get_iter(global_scale),
        fx.get_iter(route_rows),
        fx.get_iter(route_experts),
        fx.get_iter(psum),
    ).launch(
        grid=((rows * blocks_per_row + 127) // 128, 1, 1),
        block=(128, 1, 1),
        stream=stream,
    )

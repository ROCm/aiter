# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block-per-route fused quant + scatter kernel (FlyDSL).

Each warp handles one route/MX-block pair: reads hidden_states[token],
quantizes to MXFP4 or MXFP8 in-place, and writes the payload + preshuffled
e8m0 scale directly to the grouped layout at the destination row given by
topids_to_rows[route].

Grid = (numel, 1, 1), Block = (256, 1, 1) = 8 warps.
Each warp (32 threads) processes one MX block (32 elements) at a time.
8 warps process 8 MX blocks per iteration, looping ceil(num_mx_blocks/8) times.

Requires topids_to_rows to be pre-computed (by moe_route_maps kernel).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.arith import _to_raw as _raw
from flydsl.compiler.kernel_function import CompilationContext

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import buffer_ops

from aiter.ops.flydsl.kernels.gemm_mxscale_gfx1250 import rocdl
from aiter.ops.flydsl.kernels.quant_utils import emit_f32_to_e2m1, emit_mx_e8m0_scale
from aiter.utility.mx_types import (
    MxDtypeInt as _D,
    MX_DEFAULT_ROUND_MODE as _DEFAULT_MODE,
)

WAVE_SIZE = 32
WARPS_PER_BLOCK = 8
BLOCK_THREADS = WARPS_PER_BLOCK * WAVE_SIZE  # 256


def _emit_e8m0_biased(
    quant_format,
    local_max,
    *,
    i32,
    c0,
    c8,
    c23,
    c254,
    c0xFF,
    c0x200000,
    c0xFF800000,
):
    """Emit the MX scale formula. The format branch is build-time only."""
    if quant_format == "fp4":
        return emit_mx_e8m0_scale(
            local_max,
            mode=_DEFAULT_MODE,
            dtype=_D.FP4_E2M1,
        )

    # Mirror dynamic_mxfp8_quant's pow2-rounded amax recipe.
    amax_i32 = local_max.bitcast(i32)
    amax_p2_i32 = (amax_i32 + c0x200000) & c0xFF800000
    biased_exp = (amax_p2_i32 >> c23) & c0xFF
    return arith.minsi(arith.maxsi(biased_exp - c8, c0), c254)


def _emit_payload_store(
    quant_format,
    *,
    scaled,
    lane,
    row,
    blk_i,
    dst_payload_base,
    payload_rsrc,
    i32,
    c0,
    c1,
    c2,
    c3,
    c4,
    c_half_mx,
    c_mx_dwords_fp8,
    c_payload_dpr,
):
    """Emit MXFP4 or MXFP8 payload store. The format branch is build-time only."""
    if quant_format == "fp4":
        nibble = emit_f32_to_e2m1(scaled)
        is_even = arith.cmpi(CmpIPredicate.eq, lane & c1, c0)

        # Even lanes write one packed FP4 byte; odd lanes only provide their
        # nibble through readlane.
        _if_even = scf.IfOp(is_even)
        with ir.InsertionPoint(_if_even.then_block):
            partner_lane = lane ^ c1
            partner_nibble = ArithValue(rocdl.readlane(i32, nibble, partner_lane))
            packed_byte = nibble | (partner_nibble << c4)
            half_lane = lane >> c1
            byte_off = dst_payload_base + blk_i * c_half_mx + half_lane
            buffer_ops.buffer_store(
                arith.trunci(T.i8, packed_byte), payload_rsrc, byte_off
            )
            scf.YieldOp([])
        return

    # Every four lanes pack four FP8 values into one dword.
    is_lane4 = arith.cmpi(CmpIPredicate.eq, lane & c3, c0)
    _if_lane4 = scf.IfOp(is_lane4)
    with ir.InsertionPoint(_if_lane4.then_block):
        lane_group = (lane >> c2) << c2
        v0 = ArithValue(rocdl.readlane(i32, scaled.bitcast(i32), lane_group)).bitcast(
            T.f32
        )
        v1 = ArithValue(
            rocdl.readlane(i32, scaled.bitcast(i32), lane_group + c1)
        ).bitcast(T.f32)
        v2 = ArithValue(
            rocdl.readlane(i32, scaled.bitcast(i32), lane_group + c2)
        ).bitcast(T.f32)
        v3 = ArithValue(
            rocdl.readlane(i32, scaled.bitcast(i32), lane_group + c3)
        ).bitcast(T.f32)
        packed_dword = c0
        packed_dword = rocdl.cvt_pk_fp8_f32(i32, v0, v1, packed_dword, 0)
        packed_dword = rocdl.cvt_pk_fp8_f32(i32, v2, v3, packed_dword, 1)
        dword_off = row * c_payload_dpr + blk_i * c_mx_dwords_fp8 + (lane >> c2)
        buffer_ops.buffer_store(packed_dword, payload_rsrc, dword_off)
        scf.YieldOp([])


def build_moe_quant_scatter_block_module(
    model_dim: int,
    wmma_rep: int,
    scale_k_per_tile: int,
    src_is_grouped: bool = False,
    quant_format: str = "fp4",
):
    """Build a block-per-route fused quant+scatter kernel for MXFP4/MXFP8.

    Parameters
    ----------
    model_dim : int      hidden dimension (must be multiple of 32).
    wmma_rep : int       warp_tile_m // 16.
    scale_k_per_tile : int  tile_k // 32.
    quant_format : str   "fp4" or "fp8".
    """
    assert model_dim % 32 == 0
    assert wmma_rep >= 1
    assert quant_format in ("fp4", "fp8")

    num_mx_blocks = model_dim // 32
    payload_bytes_per_row = model_dim // 2 if quant_format == "fp4" else model_dim
    payload_dwords_per_row = payload_bytes_per_row // 4
    scale_bytes_per_row = model_dim // 32
    scale_dwords = scale_bytes_per_row // 4
    rows_per_tile = wmma_rep * 16

    _mode_tag = "inplace" if src_is_grouped else "scatter"
    module_name = (
        f"moe_quant_{quant_format}_{_mode_tag}_block_d{model_dim}_r{wmma_rep}_k{scale_k_per_tile}"
    )

    # Total work items = numel * num_mx_blocks (one warp per item).
    # Grid = ceil(total_work / warps_per_block). Each warp independently
    # processes one (route, mx_block) pair — no loop, no barriers.
    total_warps_per_grid = "runtime"  # passed as num_total_warps

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def quant_scatter_kernel(
        hidden: fx.Tensor,            # (token_num, model_dim) as i16 (bf16 bitcast)
        topids_to_rows: fx.Tensor,    # (numel,) int32 — grouped row per route
        dst_payload: fx.Tensor,       # output payload (bytes)
        dst_scale: fx.Tensor,         # preshuffled e8m0 output (bytes)
        numel: Int32,
        topk: Int32,
        max_m: Int32,
        num_total_warps: Int32,
    ):
        i32 = T.i32
        f32 = T.f32

        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        c2 = arith.constant(2, type=i32)
        c3 = arith.constant(3, type=i32)
        c4 = arith.constant(4, type=i32)
        c8 = arith.constant(8, type=i32)
        c16 = arith.constant(16, type=i32)
        c23 = arith.constant(23, type=i32)
        c254 = arith.constant(254, type=i32)
        c0xFF = arith.constant(0xFF, type=i32)
        c0x200000 = arith.constant(0x200000, type=i32)
        c0xFF800000 = arith.constant(0xFF800000, type=i32)
        c_wave = arith.constant(WAVE_SIZE, type=i32)
        c_warps = arith.constant(WARPS_PER_BLOCK, type=i32)
        c_model_dim = arith.constant(model_dim, type=i32)
        c_payload_bpr = arith.constant(payload_bytes_per_row, type=i32)
        c_payload_dpr = arith.constant(payload_dwords_per_row, type=i32)
        c_scale_dwords = arith.constant(scale_dwords, type=i32)
        c_num_mx_blocks = arith.constant(num_mx_blocks, type=i32)
        c_mx_block_size = arith.constant(32, type=i32)
        c_half_mx = arith.constant(16, type=i32)
        c_mx_dwords_fp8 = arith.constant(8, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)

        tid = ArithValue(fx.thread_idx.x)
        bid = ArithValue(fx.block_idx.x)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave

        # Global warp id -> (route, blk_i)
        global_warp = bid * c_warps + warp_in_block
        warp_ok = arith.cmpi(CmpIPredicate.slt, global_warp, ArithValue(num_total_warps))

        _if_warp = scf.IfOp(warp_ok)
        with ir.InsertionPoint(_if_warp.then_block):
            route = global_warp // c_num_mx_blocks
            blk_i = global_warp - route * c_num_mx_blocks

            t2r_rsrc = buffer_ops.create_buffer_resource(topids_to_rows, max_size=True)
            hidden_rsrc = buffer_ops.create_buffer_resource(hidden, max_size=True)
            payload_rsrc = buffer_ops.create_buffer_resource(dst_payload, max_size=True)
            scale_rsrc = buffer_ops.create_buffer_resource(dst_scale, max_size=True)

            row = ArithValue(
                buffer_ops.buffer_load(t2r_rsrc, route, vec_width=1, dtype=i32)
            )
            token = arith.divui(route, ArithValue(topk))

            # Preshuffle coordinates
            expert = row // ArithValue(max_m)
            local_row = row - expert * ArithValue(max_m)
            tile = local_row // c_rows_per_tile
            in_tile_row = local_row - tile * c_rows_per_tile
            w_idx = in_tile_row // c16
            lane_pos = in_tile_row - w_idx * c16
            expert_scale_base = expert * (ArithValue(max_m) * c_scale_dwords)
            out_row_scale = tile * c16 + lane_pos
            dpr = c_scale_dwords * c_wmma_rep

            src_base = row * c_model_dim if src_is_grouped else token * c_model_dim
            dst_payload_base = row * c_payload_bpr

            # Load this lane's bf16 element from MX block blk_i
            src_off = src_base + blk_i * c_mx_block_size + lane
            elem_i16 = buffer_ops.buffer_load(
                hidden_rsrc, src_off, vec_width=1, dtype=T.i16
            )
            elem_f32 = elem_i16.bitcast(T.bf16).extf(T.f32)

            # Warp-wide max(abs) via readlane XOR reduction
            abs_val = arith.maximumf(elem_f32, arith.negf(elem_f32))
            local_max = abs_val
            for shift in [16, 8, 4, 2, 1]:
                other = ArithValue(
                    rocdl.readlane(
                        i32,
                        local_max.bitcast(i32),
                        (lane ^ arith.constant(shift, type=i32)),
                    )
                ).bitcast(f32)
                local_max = arith.maximumf(local_max, other)
            local_max = ArithValue(
                rocdl.readfirstlane(i32, local_max.bitcast(i32))
            ).bitcast(f32)

            # E8M0 scale + quantize.
            e8m0_biased = _emit_e8m0_biased(
                quant_format,
                local_max,
                i32=i32,
                c0=c0,
                c8=c8,
                c23=c23,
                c254=c254,
                c0xFF=c0xFF,
                c0x200000=c0x200000,
                c0xFF800000=c0xFF800000,
            )
            qscale = ((c254 - e8m0_biased) << c23).bitcast(f32)
            scaled = elem_f32 * qscale

            _emit_payload_store(
                quant_format,
                scaled=scaled,
                lane=lane,
                row=row,
                blk_i=blk_i,
                dst_payload_base=dst_payload_base,
                payload_rsrc=payload_rsrc,
                i32=i32,
                c0=c0,
                c1=c1,
                c2=c2,
                c3=c3,
                c4=c4,
                c_half_mx=c_half_mx,
                c_mx_dwords_fp8=c_mx_dwords_fp8,
                c_payload_dpr=c_payload_dpr,
            )

            # Lane 0 writes scale byte (preshuffled)
            is_lane0 = arith.cmpi(CmpIPredicate.eq, lane, c0)
            _if_lane0 = scf.IfOp(is_lane0)
            with ir.InsertionPoint(_if_lane0.then_block):
                sd = blk_i >> arith.constant(2, type=i32)
                byte_in_dw = blk_i & arith.constant(3, type=i32)
                scale_dw_off = (
                    expert_scale_base
                    + out_row_scale * dpr
                    + sd * c_wmma_rep
                    + w_idx
                )
                scale_byte_off = scale_dw_off * c4 + byte_in_dw
                buffer_ops.buffer_store(
                    arith.trunci(T.i8, e8m0_biased),
                    scale_rsrc,
                    scale_byte_off,
                )
                scf.YieldOp([])

            scf.YieldOp([])

    @flyc.jit
    def launch_quant_scatter_block(
        hidden: fx.Tensor,
        topids_to_rows: fx.Tensor,
        dst_payload: fx.Tensor,
        dst_scale: fx.Tensor,
        numel: fx.Int32,
        topk: fx.Int32,
        max_m: fx.Int32,
        num_total_warps: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        # Grid = ceil(num_total_warps / WARPS_PER_BLOCK)
        ntw = arith.index_cast(T.index, num_total_warps)
        wpb = arith.index(WARPS_PER_BLOCK)
        gx = (ntw + wpb - arith.index(1)) / wpb
        launcher = quant_scatter_kernel(
            hidden, topids_to_rows, dst_payload, dst_scale,
            numel, topk, max_m, num_total_warps,
        )
        launcher.launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_quant_scatter_block

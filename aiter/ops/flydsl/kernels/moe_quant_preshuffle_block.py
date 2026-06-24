# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block-per-row fused quant + preshuffle kernel for stage2 (FlyDSL).

Input is already in grouped layout (num_rows, dim) bf16.
Grid = num_rows (= E*max_m for stage2). Each block (256 threads = 8 warps)
processes one row: 8 warps handle 8 MX blocks per loop iteration, looping
ceil(num_mx_blocks / 8) times. Each warp does amax reduction + E8M0 scale +
FP4 quantize + pack + write payload byte + write preshuffled scale byte.

For stage2 with E*max_m=2048, grid=2048 gives full GPU occupancy.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.expr.arith import ArithValue, CmpIPredicate
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
BLOCK_THREADS = WARPS_PER_BLOCK * WAVE_SIZE


def build_moe_quant_preshuffle_block_module(
    dim: int,
    wmma_rep: int,
    scale_k_per_tile: int,
):
    assert dim % 32 == 0
    assert wmma_rep >= 1

    num_mx_blocks = dim // 32
    payload_bytes_per_row = dim // 2
    scale_dwords = dim // 32 // 4
    rows_per_tile = wmma_rep * 16
    iters_per_block = (num_mx_blocks + WARPS_PER_BLOCK - 1) // WARPS_PER_BLOCK

    module_name = (
        f"moe_quant_preshuffle_block_d{dim}_r{wmma_rep}_k{scale_k_per_tile}"
    )

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def quant_preshuffle_kernel(
        src: fx.Tensor,
        dst_payload: fx.Tensor,
        dst_scale: fx.Tensor,
        masked_m: fx.Tensor,
        num_rows: Int32,
        max_m: Int32,
        num_total_warps: Int32,
    ):
        i32 = T.i32
        f32 = T.f32

        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        c4 = arith.constant(4, type=i32)
        c16 = arith.constant(16, type=i32)
        c23 = arith.constant(23, type=i32)
        c254 = arith.constant(254, type=i32)
        c_wave = arith.constant(WAVE_SIZE, type=i32)
        c_warps = arith.constant(WARPS_PER_BLOCK, type=i32)
        c_dim = arith.constant(dim, type=i32)
        c_payload_bpr = arith.constant(payload_bytes_per_row, type=i32)
        c_scale_dwords = arith.constant(dim // 32 // 4, type=i32)
        c_num_mx_blocks = arith.constant(num_mx_blocks, type=i32)
        c_mx_block_size = arith.constant(32, type=i32)
        c_half_mx = arith.constant(16, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)

        tid = ArithValue(fx.thread_idx.x)
        row = ArithValue(fx.block_idx.x)

        row_ok = arith.cmpi(CmpIPredicate.slt, row, ArithValue(num_rows))
        _if_row = scf.IfOp(row_ok)
        with ir.InsertionPoint(_if_row.then_block):
            # Skip padding rows: check masked_m[expert] > local_row
            masked_m_rsrc = buffer_ops.create_buffer_resource(masked_m, max_size=True)
            expert_for_skip = row // ArithValue(max_m)
            local_row_for_skip = row - expert_for_skip * ArithValue(max_m)
            valid_m = ArithValue(buffer_ops.buffer_load(
                masked_m_rsrc, expert_for_skip, vec_width=1, dtype=i32
            ))
            row_valid = arith.cmpi(CmpIPredicate.slt, local_row_for_skip, valid_m)
            _if_valid = scf.IfOp(row_valid)
            with ir.InsertionPoint(_if_valid.then_block):
                warp_in_block = tid // c_wave
                lane = tid - warp_in_block * c_wave

                src_rsrc = buffer_ops.create_buffer_resource(src, max_size=True)
                payload_rsrc = buffer_ops.create_buffer_resource(dst_payload, max_size=True)
                scale_rsrc = buffer_ops.create_buffer_resource(dst_scale, max_size=True)

                expert = row // ArithValue(max_m)
                local_row = row - expert * ArithValue(max_m)
                tile = local_row // c_rows_per_tile
                in_tile_row = local_row - tile * c_rows_per_tile
                w_idx = in_tile_row // c16
                lane_pos = in_tile_row - w_idx * c16
                expert_scale_base = expert * (ArithValue(max_m) * c_scale_dwords)
                out_row_scale = tile * c16 + lane_pos
                dpr = c_scale_dwords * c_wmma_rep

                src_base = row * c_dim
                dst_payload_base = row * c_payload_bpr

                c0_idx = arith.index(0)
                c1_idx = arith.index(1)
                iters_idx = arith.index(iters_per_block)

                _loop = scf.ForOp(c0_idx, iters_idx, c1_idx)
                with ir.InsertionPoint(_loop.body):
                    iter_i = arith.index_cast(i32, _loop.induction_variable)
                    blk_i = iter_i * c_warps + warp_in_block

                    blk_ok = arith.cmpi(CmpIPredicate.slt, blk_i, c_num_mx_blocks)
                    _if_blk = scf.IfOp(blk_ok)
                    with ir.InsertionPoint(_if_blk.then_block):
                        src_off = src_base + blk_i * c_mx_block_size + lane
                        elem_i16 = buffer_ops.buffer_load(
                            src_rsrc, src_off, vec_width=1, dtype=T.i16
                        )
                        elem_f32 = elem_i16.bitcast(T.bf16).extf(T.f32)

                        abs_val = arith.maximumf(elem_f32, arith.negf(elem_f32))
                        local_max = abs_val
                        for shift in [16, 8, 4, 2, 1]:
                            other = ArithValue(
                                rocdl.readlane(
                                    i32, local_max.bitcast(i32),
                                    (lane ^ arith.constant(shift, type=i32)),
                                )
                            ).bitcast(f32)
                            local_max = arith.maximumf(local_max, other)
                        local_max = ArithValue(
                            rocdl.readfirstlane(i32, local_max.bitcast(i32))
                        ).bitcast(f32)

                        e8m0_biased = emit_mx_e8m0_scale(
                            local_max, mode=_DEFAULT_MODE, dtype=_D.FP4_E2M1
                        )
                        qscale = ((c254 - e8m0_biased) << c23).bitcast(f32)
                        nibble = emit_f32_to_e2m1(elem_f32 * qscale)

                        partner_lane = lane ^ c1
                        partner_nibble = ArithValue(
                            rocdl.readlane(i32, nibble, partner_lane)
                        )
                        is_even = arith.cmpi(CmpIPredicate.eq, lane & c1, c0)
                        packed_byte = arith.select(
                            is_even,
                            nibble | (partner_nibble << c4),
                            partner_nibble | (nibble << c4),
                        )

                        _if_even = scf.IfOp(is_even)
                        with ir.InsertionPoint(_if_even.then_block):
                            half_lane = lane >> c1
                            byte_off = dst_payload_base + blk_i * c_half_mx + half_lane
                            buffer_ops.buffer_store(
                                arith.trunci(T.i8, packed_byte), payload_rsrc, byte_off
                            )
                            scf.YieldOp([])

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
                                arith.trunci(T.i8, e8m0_biased), scale_rsrc, scale_byte_off
                            )
                            scf.YieldOp([])

                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])

    @flyc.jit
    def launch_quant_preshuffle_block(
        src: fx.Tensor,
        dst_payload: fx.Tensor,
        dst_scale: fx.Tensor,
        masked_m: fx.Tensor,
        num_rows: fx.Int32,
        max_m: fx.Int32,
        num_total_warps: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        gx = arith.index_cast(T.index, num_rows)
        launcher = quant_preshuffle_kernel(
            src, dst_payload, dst_scale, masked_m, num_rows, max_m, num_total_warps,
        )
        launcher.launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_quant_preshuffle_block

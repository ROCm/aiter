# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Warp-per-route fused kernel: route + MX FP4 quant + scatter + scale preshuffle.

Each warp handles one route atomically:
  1. Lane 0: read expert id, atomicAdd for slot, compute grouped row
  2. Broadcast row to all lanes via readfirstlane
  3. All lanes: read bf16 hidden_states row, do per-1x32 MX FP4 quantization,
     write packed fp4 payload directly to grouped layout, write preshuffled
     e8m0 scale to WMMA layout

No cross-block barrier. Decode (numel=4) completes in 1 block.
Eliminates intermediate a1_payload and a1_scale_token_u8 buffers.
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
MX_BLOCK_SIZE = 32  # elements per MX block


def build_moe_route_quant_scatter_warp_module(
    model_dim: int,
    wmma_rep: int,
    scale_k_per_tile: int,
):
    """Build a warp-per-route fused route+quant+scatter kernel for FP4.

    Parameters
    ----------
    model_dim : int      hidden dimension (must be multiple of 32).
    wmma_rep : int       warp_tile_m // 16.
    scale_k_per_tile : int  tile_k // 32.
    """
    assert model_dim % MX_BLOCK_SIZE == 0
    assert wmma_rep >= 1

    num_mx_blocks = model_dim // MX_BLOCK_SIZE  # MX blocks per token row
    payload_bytes_per_row = model_dim // 2  # fp4x2 packed
    payload_dwords = payload_bytes_per_row // 4
    scale_bytes_per_row = model_dim // MX_BLOCK_SIZE  # one e8m0 byte per block
    scale_dwords = scale_bytes_per_row // 4
    rows_per_tile = wmma_rep * 16

    module_name = (
        f"moe_route_quant_scatter_warp_d{model_dim}_r{wmma_rep}_k{scale_k_per_tile}"
    )

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def route_quant_scatter_kernel(
        topk_ids: fx.Tensor,         # (numel,) int32
        atomic_counter: fx.Tensor,   # (E,) int32 init 0
        topids_to_rows: fx.Tensor,   # (numel,) int32 out
        hidden: fx.Tensor,           # (token_num, model_dim) bf16 as i16
        dst_payload: fx.Tensor,      # (E*max_m, payload_dwords) i32
        dst_scale: fx.Tensor,        # preshuffled e8m0 output (dwords)
        numel: Int32,
        topk: Int32,
        max_m: Int32,
    ):
        i32 = T.i32
        f32 = T.f32

        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        c4 = arith.constant(4, type=i32)
        c8 = arith.constant(8, type=i32)
        c16 = arith.constant(16, type=i32)
        c23 = arith.constant(23, type=i32)
        c254 = arith.constant(254, type=i32)
        c0_f32 = arith.constant(0.0, type=f32)
        c_wave = arith.constant(WAVE_SIZE, type=i32)
        c_warps = arith.constant(WARPS_PER_BLOCK, type=i32)
        c_model_dim = arith.constant(model_dim, type=i32)
        c_model_dim_half = arith.constant(model_dim // 2, type=i32)
        c_payload_dwords = arith.constant(payload_dwords, type=i32)
        c_scale_dwords = arith.constant(scale_dwords, type=i32)
        c_num_mx_blocks = arith.constant(num_mx_blocks, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)

        tid = ArithValue(fx.thread_idx.x)
        bid = ArithValue(fx.block_idx.x)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave

        warp_route = bid * c_warps + warp_in_block
        route_ok = arith.cmpi(CmpIPredicate.slt, warp_route, ArithValue(numel))

        _if_route = scf.IfOp(route_ok)
        with ir.InsertionPoint(_if_route.then_block):
            topk_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
            t2r_rsrc = buffer_ops.create_buffer_resource(topids_to_rows, max_size=True)
            hidden_rsrc = buffer_ops.create_buffer_resource(hidden, max_size=True)
            payload_rsrc = buffer_ops.create_buffer_resource(dst_payload, max_size=True)
            scale_rsrc = buffer_ops.create_buffer_resource(dst_scale, max_size=True)

            # --- Lane 0: route -> slot -> row ---
            e_on_lane0 = buffer_ops.buffer_load(
                topk_rsrc, warp_route, vec_width=1, dtype=i32
            )

            counter_base = buffer_ops.extract_base_index(
                atomic_counter, address_space=1
            )
            e_idx = arith.index_cast(T.index, e_on_lane0)
            counter_addr = fx.Index(counter_base) + fx.Index(e_idx) * fx.Index(4)
            counter_ptr = buffer_ops.create_llvm_ptr(counter_addr, address_space=1)
            counter_ptr = (
                counter_ptr._value if hasattr(counter_ptr, "_value") else counter_ptr
            )

            slot_on_lane0 = arith.constant(0, type=i32)
            if lane == 0:
                slot_on_lane0 = ArithValue(
                    llvm.AtomicRMWOp(
                        llvm.AtomicBinOp.add,
                        counter_ptr,
                        c1,
                        llvm.AtomicOrdering.monotonic,
                        syncscope="agent",
                        alignment=4,
                    ).result
                )

            slot = ArithValue(rocdl.readfirstlane(i32, slot_on_lane0))
            e_val = ArithValue(rocdl.readfirstlane(i32, e_on_lane0))
            row = slot + e_val * ArithValue(max_m)

            if lane == 0:
                buffer_ops.buffer_store(row, t2r_rsrc, warp_route)

            token = arith.divui(warp_route, ArithValue(topk))

            # --- All lanes: quantize + scatter ---
            # Source: hidden[token, 0..model_dim-1] as bf16 (stored as i16)
            # Each MX block = 32 bf16 elements → 1 lane per element (wave=32)
            # Process num_mx_blocks iterations, each covering one 32-element block.
            src_base_i16 = token * c_model_dim  # offset in i16 elements
            dst_payload_base = row * c_payload_dwords  # offset in i32 dwords

            # Scale preshuffle coordinates for this row
            expert = row // ArithValue(max_m)
            local_row = row - expert * ArithValue(max_m)
            tile = local_row // c_rows_per_tile
            in_tile_row = local_row - tile * c_rows_per_tile
            w_idx = in_tile_row // c16
            lane_pos = in_tile_row - w_idx * c16
            expert_scale_base = expert * (ArithValue(max_m) * c_scale_dwords)
            out_row_scale = tile * c16 + lane_pos
            dpr = c_scale_dwords * c_wmma_rep

            # Runtime loop over MX blocks (compact code, icache-friendly).
            c_mx_blocks = arith.constant(num_mx_blocks, type=i32)
            c_mx_block_size = arith.constant(MX_BLOCK_SIZE, type=i32)
            c_half_mx = arith.constant(MX_BLOCK_SIZE // 2, type=i32)
            c2 = arith.constant(2, type=i32)

            c0_idx = arith.index(0)
            c1_idx = arith.index(1)
            mx_upper = arith.index(num_mx_blocks)

            _blk_loop = scf.ForOp(c0_idx, mx_upper, c1_idx)
            with ir.InsertionPoint(_blk_loop.body):
                blk_i = arith.index_cast(i32, _blk_loop.induction_variable)

                # Load this lane's bf16 element from the MX block
                src_off = src_base_i16 + blk_i * c_mx_block_size + lane
                elem_i16 = buffer_ops.buffer_load(hidden_rsrc, src_off, vec_width=1, dtype=T.i16)
                elem_f32 = elem_i16.bitcast(T.bf16).extf(T.f32)

                # Warp-wide max(abs) reduction via readlane XOR
                abs_val = arith.maximumf(elem_f32, arith.negf(elem_f32))
                local_max = abs_val
                for shift in [16, 8, 4, 2, 1]:
                    other = ArithValue(
                        rocdl.readlane(i32, local_max.bitcast(i32), (lane ^ arith.constant(shift, type=i32)))
                    ).bitcast(f32)
                    local_max = arith.maximumf(local_max, other)
                local_max = ArithValue(rocdl.readfirstlane(i32, local_max.bitcast(i32))).bitcast(f32)

                # E8M0 scale + quantize to fp4
                e8m0_biased = emit_mx_e8m0_scale(local_max, mode=_DEFAULT_MODE, dtype=_D.FP4_E2M1)
                qscale = ((c254 - e8m0_biased) << c23).bitcast(f32)
                nibble = emit_f32_to_e2m1(elem_f32 * qscale)

                # Pack adjacent lanes' nibbles into bytes
                partner_lane = lane ^ c1
                partner_nibble = ArithValue(rocdl.readlane(i32, nibble, partner_lane))
                is_even = arith.cmpi(CmpIPredicate.eq, lane & c1, c0)
                packed_byte = arith.select(
                    is_even,
                    nibble | (partner_nibble << c4),
                    partner_nibble | (nibble << c4),
                )

                # Even lanes write payload byte
                _if_even = scf.IfOp(is_even)
                with ir.InsertionPoint(_if_even.then_block):
                    half_lane = lane >> c1
                    byte_off = row * c_model_dim_half + blk_i * c_half_mx + half_lane
                    buffer_ops.buffer_store(arith.trunci(T.i8, packed_byte), payload_rsrc, byte_off)
                    scf.YieldOp([])

                # Lane 0 writes scale byte
                is_lane0 = arith.cmpi(CmpIPredicate.eq, lane, c0)
                _if_lane0 = scf.IfOp(is_lane0)
                with ir.InsertionPoint(_if_lane0.then_block):
                    sd = blk_i >> c2  # blk // 4
                    byte_in_dw = blk_i & arith.constant(3, type=i32)  # blk % 4
                    scale_dw_off = expert_scale_base + out_row_scale * dpr + sd * c_wmma_rep + w_idx
                    scale_byte_off = scale_dw_off * c4 + byte_in_dw
                    buffer_ops.buffer_store(arith.trunci(T.i8, e8m0_biased), scale_rsrc, scale_byte_off)
                    scf.YieldOp([])

                scf.YieldOp([])

            scf.YieldOp([])

    @flyc.jit
    def launch_route_quant_scatter_warp(
        topk_ids: fx.Tensor,
        atomic_counter: fx.Tensor,
        topids_to_rows: fx.Tensor,
        hidden: fx.Tensor,
        dst_payload: fx.Tensor,
        dst_scale: fx.Tensor,
        numel: fx.Int32,
        topk: fx.Int32,
        max_m: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass

        numel_idx = arith.index_cast(T.index, numel)
        warps_per_block_idx = arith.index(WARPS_PER_BLOCK)
        gx = (numel_idx + warps_per_block_idx - arith.index(1)) / warps_per_block_idx
        launcher = route_quant_scatter_kernel(
            topk_ids, atomic_counter, topids_to_rows,
            hidden, dst_payload, dst_scale,
            numel, topk, max_m,
        )
        launcher.launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_route_quant_scatter_warp

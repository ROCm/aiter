# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Warp-per-route fused kernel: route-map + payload-scatter + scale-preshuffle.

Each warp handles one route atomically:
  1. Lane 0: read expert id, atomicAdd for slot, compute grouped row, write topids_to_rows
  2. Broadcast row to all lanes via readfirstlane
  3. All lanes: copy payload row (src_payload[token] -> dst_payload[row])
  4. All lanes: preshuffle scale row (src_scale[token] -> dst_scale[WMMA layout])

No cross-block barrier needed. Decode (numel=4) completes in 1 block.
Grid = (ceil(numel / warps_per_block), 1, 1), Block = (warps_per_block * wave_size, 1, 1)
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

WAVE_SIZE = 32
WARPS_PER_BLOCK = 8
BLOCK_THREADS = WARPS_PER_BLOCK * WAVE_SIZE  # 256


def build_moe_route_scatter_warp_module(
    payload_row_bytes: int,
    scale_row_bytes: int,
    wmma_rep: int,
    scale_k_per_tile: int,
):
    """Build a warp-per-route fused route+scatter kernel.

    Parameters
    ----------
    payload_row_bytes : int  byte width of one payload row.
    scale_row_bytes : int    byte width of one scale row (K // 32).
    wmma_rep : int           warp_tile_m // 16.
    scale_k_per_tile : int   tile_k // 32.
    """
    assert payload_row_bytes > 0
    assert scale_row_bytes > 0 and scale_row_bytes % 4 == 0
    assert wmma_rep >= 1

    # Payload copy: each lane handles dwords, striding by wave_size
    payload_dwords = payload_row_bytes // 4
    assert payload_row_bytes % 4 == 0, "payload must be dword-aligned"

    # Use dwordx4 when possible for payload
    if payload_dwords % 4 == 0:
        p_vec_width = 4
    elif payload_dwords % 2 == 0:
        p_vec_width = 2
    else:
        p_vec_width = 1
    p_units_per_row = payload_dwords // p_vec_width

    # Scale geometry
    scale_dwords = scale_row_bytes // 4
    rows_per_tile = wmma_rep * 16

    module_name = (
        f"moe_route_scatter_warp_p{payload_row_bytes}_s{scale_row_bytes}"
        f"_r{wmma_rep}_k{scale_k_per_tile}"
    )

    @flyc.kernel(name=module_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def route_scatter_warp_kernel(
        topk_ids: fx.Tensor,         # (numel,) int32
        atomic_counter: fx.Tensor,   # (E,) int32 init 0
        topids_to_rows: fx.Tensor,   # (numel,) int32 out
        src_payload: fx.Tensor,      # (token_num, payload_dwords) i32
        dst_payload: fx.Tensor,      # (E*max_m, payload_dwords) i32
        src_scale: fx.Tensor,        # (token_num, scale_dwords) i32
        dst_scale: fx.Tensor,        # preshuffled output (dwords)
        numel: Int32,
        topk: Int32,
        max_m: Int32,
    ):
        i32 = T.i32

        c0 = arith.constant(0, type=i32)
        c1 = arith.constant(1, type=i32)
        c_wave = arith.constant(WAVE_SIZE, type=i32)
        c_warps = arith.constant(WARPS_PER_BLOCK, type=i32)
        c_payload_dwords = arith.constant(payload_dwords, type=i32)
        c_p_units = arith.constant(p_units_per_row, type=i32)
        c_scale_dwords = arith.constant(scale_dwords, type=i32)
        c_wmma_rep = arith.constant(wmma_rep, type=i32)
        c16 = arith.constant(16, type=i32)
        c_rows_per_tile = arith.constant(rows_per_tile, type=i32)

        tid = ArithValue(fx.thread_idx.x)
        bid = ArithValue(fx.block_idx.x)

        warp_in_block = tid // c_wave
        lane = tid - warp_in_block * c_wave

        # This warp's route index
        warp_route = bid * c_warps + warp_in_block
        route_ok = arith.cmpi(CmpIPredicate.slt, warp_route, ArithValue(numel))

        _if_route = scf.IfOp(route_ok)
        with ir.InsertionPoint(_if_route.then_block):
            topk_rsrc = buffer_ops.create_buffer_resource(topk_ids, max_size=True)
            t2r_rsrc = buffer_ops.create_buffer_resource(topids_to_rows, max_size=True)
            src_p_rsrc = buffer_ops.create_buffer_resource(src_payload, max_size=True)
            dst_p_rsrc = buffer_ops.create_buffer_resource(dst_payload, max_size=True)
            src_s_rsrc = buffer_ops.create_buffer_resource(src_scale, max_size=True)
            dst_s_rsrc = buffer_ops.create_buffer_resource(dst_scale, max_size=True)

            # --- Lane 0: route -> slot -> row ---
            e_on_lane0 = buffer_ops.buffer_load(
                topk_rsrc, warp_route, vec_width=1, dtype=i32
            )

            # atomicAdd(&counter[e], 1) -> slot
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

            # Broadcast slot from lane 0 to all lanes
            slot = ArithValue(rocdl.readfirstlane(i32, slot_on_lane0))
            e_val = ArithValue(rocdl.readfirstlane(i32, e_on_lane0))
            row = slot + e_val * ArithValue(max_m)

            # Write topids_to_rows[route] = row (lane 0 only, uniform value)
            if lane == 0:
                buffer_ops.buffer_store(row, t2r_rsrc, warp_route)

            # token = route // topk (uniform across warp)
            token = arith.divui(warp_route, ArithValue(topk))

            # --- All lanes: payload copy ---
            src_p_base = token * c_payload_dwords
            dst_p_base = row * c_payload_dwords

            for it in range_constexpr(
                (p_units_per_row + WAVE_SIZE - 1) // WAVE_SIZE
            ):
                uidx = lane + arith.constant(it * WAVE_SIZE, type=i32)
                u_ok = arith.cmpi(CmpIPredicate.ult, uidx, c_p_units)
                _if_u = scf.IfOp(u_ok)
                with ir.InsertionPoint(_if_u.then_block):
                    eidx = (
                        uidx
                        if p_vec_width == 1
                        else uidx * arith.constant(p_vec_width, type=i32)
                    )
                    v = buffer_ops.buffer_load(
                        src_p_rsrc, src_p_base + eidx,
                        vec_width=p_vec_width, dtype=i32,
                    )
                    buffer_ops.buffer_store(v, dst_p_rsrc, dst_p_base + eidx)
                    scf.YieldOp([])

            # --- All lanes: scale preshuffle ---
            # Decode row -> preshuffle coordinates
            expert = row // ArithValue(max_m)
            local_row = row - expert * ArithValue(max_m)
            tile = local_row // c_rows_per_tile
            in_tile_row = local_row - tile * c_rows_per_tile
            w_idx = in_tile_row // c16
            lane_pos = in_tile_row - w_idx * c16

            expert_dword_base = expert * (ArithValue(max_m) * c_scale_dwords)
            out_row = tile * c16 + lane_pos
            dpr = c_scale_dwords * c_wmma_rep

            src_s_base = token * c_scale_dwords

            for it in range_constexpr(
                (scale_dwords + WAVE_SIZE - 1) // WAVE_SIZE
            ):
                sd = lane + arith.constant(it * WAVE_SIZE, type=i32)
                sd_ok = arith.cmpi(CmpIPredicate.ult, sd, c_scale_dwords)
                _if_sd = scf.IfOp(sd_ok)
                with ir.InsertionPoint(_if_sd.then_block):
                    sv = buffer_ops.buffer_load(
                        src_s_rsrc, src_s_base + sd, vec_width=1, dtype=i32
                    )
                    dst_s_off = (
                        expert_dword_base + out_row * dpr + sd * c_wmma_rep + w_idx
                    )
                    buffer_ops.buffer_store(sv, dst_s_rsrc, dst_s_off)
                    scf.YieldOp([])

            scf.YieldOp([])

    @flyc.jit
    def launch_route_scatter_warp(
        topk_ids: fx.Tensor,
        atomic_counter: fx.Tensor,
        topids_to_rows: fx.Tensor,
        src_payload: fx.Tensor,
        dst_payload: fx.Tensor,
        src_scale: fx.Tensor,
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
        launcher = route_scatter_warp_kernel(
            topk_ids, atomic_counter, topids_to_rows,
            src_payload, dst_payload, src_scale, dst_scale,
            numel, topk, max_m,
        )
        launcher.launch(
            grid=(gx, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_route_scatter_warp

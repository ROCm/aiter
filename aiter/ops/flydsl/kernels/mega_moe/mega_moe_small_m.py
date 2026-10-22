# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, SIM102
"""MegaMoEV2 small-input path: quant, dispatch, GEMM1, GEMM2 and combine fused into one kernel.

Dispatch writes each routed token straight into its expert's fixed slot on the destination rank; the
runtime owns its buffers and protocol state and serves up to 1024 tokens per rank.
"""

import functools
from dataclasses import replace
from enum import IntEnum

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.ir.flydsl as mori_shmem
import mori.shmem as ms
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_gfx_runtime

from .. import communication_ops_utils as comm_ops
from ..flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
)
from ..kernels_common import ceildiv
from ..mxfp4_gemm_common import _fabs_f32 as fabs_f32
from ..mxfp4_gemm_common import lds_typed_ptr, lds_vec_load
from ..tensor_shim import _run_compiled
from .dispatch import _wave_inclusive_scan_i32
from .mega_moe_fused_gemm import (
    SMALL_M_FIXED_SLOT_MAX_MTPR,
    AddrBuffer,
    MegaMoEConfig,
    _buffer_load,
    _buffer_store,
    _LdsF32View,
    _make_buffer,
    _make_buffer_from_addr,
    build_fused_gemm1,
    gemm2_compute_v2,
    issue_a_load_lds_dt,
    kStages,
    load_at,
    nearest_token_bucket,
    select_mega_moe_config,
    store_at,
)
from .mega_moe_stage1 import _validate_dispatch_capacity
from .mega_moe_stage2 import _fp8_scale_for_leader, _stage2_lds_bytes

__all__ = ["SmallMRuntime"]


class DispatchSlot(IntEnum):
    P2P_TOKEN = 0
    P2P_SCALE = 1
    P2P_WEIGHT = 2
    P2P_SRCMAP = 3
    P2P_RUNNING = 4
    P2P_COUNT_DONE = 5
    SORTED_EXPERT = 6
    TILE_ROW_BASE = 7
    NUM_VALID = 8
    RUNNING = 9
    COUNT_DONE = 10
    ENTRY_COUNT = 11
    EPOCH_GATE = 12
    WORK_HEAD = 13
    GROUP_DONE = 14
    TILE_ROWS = 15
    # GEMM2 and combine.
    G2_AQ = 16
    G2_ASCALE = 17
    G2_BQ = 18
    G2_BSCALE = 19
    G2_EIDS = 20
    G2_STIDS = 21
    G2_SWEIGHTS = 22
    G2_TRB = 23
    G2_P2P_COMB_INP = 24
    G2_CNT = 25
    G2_READY = 26
    G2_MAX_M_BLOCKS = 27
    FC_CNT = 28
    FC_XDB = 29
    FC_P2P_XDB = 30
    FC_INP = 31
    FC_OUT = 32
    FC_TRECV = 33
    PLAN_NV = 34  # i64 (gate_epoch << 32 | num_valid): plan flag that carries num_valid
    XCD_ACQ = (
        35  # per-XCD i64 copy of PLAN_NV, republished after that XCD's L2 invalidation
    )


DISPATCH_TABLE_SIZE = max(DispatchSlot) + 1


def _quantize_row_mxfp8(row_rsrc, lane, fz_n_i32, fz_scale_n_i32):
    """Quantize one bf16 row per wave to MXFP8 in the fixed-slot payload layout.

    Chunk c of lane l holds fp8 elements [c*1024 + l*16, +16) as one i32x4, so a 32-element scale group is a lane
    pair. Lane w < fz_scale_n_i32 returns packed scale word w (groups 4w..4w+3), matching the [token, n/32] u8 scales.
    """
    assert (
        fz_scale_n_i32 * 4 * 32 == fz_n_i32 * 4
    ), "fused quant needs 1x32 scales over the whole row"
    inv_max_pos = fx.Int32(0x3B124925).bitcast(fx.Float32)  # 1/448, fp8 e4m3 max
    raw = [
        load_at(row_rsrc, lane * fx.Int32(8) + fx.Int32(c * 2 + h * 4), fx.Int32, 4)
        for c in range(0, fz_n_i32, 256)
        for h in range(2)
    ]
    row_values = []
    e8m0_chunks = []
    for c in range_constexpr(len(raw) // 2):
        values = [
            fx.Vector(raw[2 * c + h]).bitcast(fx.BFloat16).to(fx.Float32)
            for h in range(2)
        ]
        act = [values[h][e] for h in range(2) for e in range(8)]
        local_max = fx.Float32(1e-10)
        for v in act:
            local_max = local_max.maximumf(v.maximumf(fx.Float32(0.0) - v))
        group_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(1), fx.Int32(64)))
        working = (group_max * inv_max_pos).bitcast(fx.Int32)
        mantissa = working & fx.Int32(0x7FFFFF)
        biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
        e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
        e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
        quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
        scaled = [v * quant_scale for v in act]
        words = []
        for word in range_constexpr(4):
            base = word * 4
            packed = fx.rocdl.cvt_pk_fp8_f32(
                T.i32, scaled[base], scaled[base + 1], fx.Int32(0), 0
            )
            packed = fx.rocdl.cvt_pk_fp8_f32(
                T.i32, scaled[base + 2], scaled[base + 3], packed, 1
            )
            words.append(packed)
        row_values.append(fx.Vector.from_elements(words, fx.Int32))
        e8m0_chunks.append(e8m0)
    # Scale word w = groups 4w..4w+3 = chunk w//8, pair lanes 8*(w%8) + 2k.
    word_chunk = lane >> fx.Int32(3)
    scale = fx.Int32(0)
    for c in range_constexpr(len(e8m0_chunks)):
        word = fx.Int32(0)
        for k in range_constexpr(4):
            source_lane = (lane & fx.Int32(7)) * fx.Int32(8) + fx.Int32(2 * k)
            byte = fx.Int32(
                fx.rocdl.ds_bpermute(T.i32, source_lane * fx.Int32(4), e8m0_chunks[c])
            )
            word = word | (byte << fx.Int32(8 * k))
        scale = (word_chunk == fx.Int32(c)).select(word, scale)
    return row_values, scale


def prefetch_direct_fixed_slot_route(
    *,
    num_waves,
    fz_npes,
    fz_epr,
    fz_k,
    fz_mtpr,
    fz_total_experts,
    fz_nbytes,
    fz_n_i32,
    fz_scale_n_i32,
    addr_disp,
    addr_in_tok,
    addr_in_idx,
    addr_in_wts,
    addr_in_sc,
    i32_cur_tok,
    dispatch_blocks,
    producer_slot,
    route_offset=0,
):
    """Load everything the (single) route of this wave needs that does not depend on the epoch gate.

    addr_in_tok holds the bf16 rows; the wave quantizes its row to MXFP8 (1x32 E8M0 scales, bit-identical to
    quant.per_1x32_mx_quant). addr_in_sc is unused.

    Straight-line (no device branches) so it can run before the gate wait: the route's expert, the
    destination's remote pointers, the token row, its scales and its weight. Indices are clamped so waves
    without a route read valid memory; emit_direct_fixed_slot_payload ignores their values.
    """
    rdisp = AddrBuffer(addr_disp)

    def dp(i):
        return load_at(rdisp, fx.Int32(int(i)), fx.Int64, 1)

    tables = {
        "rx": dp(DispatchSlot.P2P_TOKEN),
        "sc": dp(DispatchSlot.P2P_SCALE),
        "wts": dp(DispatchSlot.P2P_WEIGHT),
        "sm": dp(DispatchSlot.P2P_SRCMAP),
        "running": dp(DispatchSlot.P2P_RUNNING),
        "source_done": dp(DispatchSlot.P2P_COUNT_DONE),
        "producer_done": dp(DispatchSlot.GROUP_DONE),
    }
    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    # route_offset selects a later round of this wave's routes (emit_direct_fixed_slot_payload route_rounds).
    route = producer_slot * fx.Int32(num_waves) + warp + fx.Int32(route_offset)
    route_limit = i32_cur_tok * fx.Int32(fz_k)
    wk = ((route >= fx.Int32(0)) & (route < route_limit)).select(route, fx.Int32(0))
    source_token = wk // fx.Int32(fz_k)
    global_expert = fx.Int32(
        fx.rocdl.readfirstlane(
            T.i32,
            load_at(AddrBuffer(addr_in_idx), wk, fx.Int32, 1),
        )
    )
    valid_expert = (global_expert >= fx.Int32(0)) & (
        global_expert < fx.Int32(fz_total_experts)
    )
    destination = valid_expert.select(global_expert, fx.Int32(0)) // fx.Int32(fz_epr)
    remote = {
        name: load_at(AddrBuffer(tables[name]), destination, fx.Int64, 1)
        for name in ("rx", "sc", "wts", "sm", "running")
    }
    assert fz_n_i32 % 256 == 0, "route prefetch needs rows of whole 1 KiB wave chunks"
    row_values, scale = _quantize_row_mxfp8(
        AddrBuffer(addr_in_tok + fx.Int64(source_token) * fx.Int64(2 * fz_nbytes)),
        lane,
        fz_n_i32,
        fz_scale_n_i32,
    )
    weight = load_at(AddrBuffer(addr_in_wts), wk, fx.Float32, 1)
    return {
        "tables": tables,
        "global_expert": global_expert,
        "remote": remote,
        "row_values": row_values,
        "scale": scale,
        "weight": weight,
    }


# fmt: off
@flyc.jit
def emit_direct_fixed_slot_payload(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_cap, fz_mtpr, fz_rank, fz_total_experts, fz_nbytes, fz_n_i32,
    fz_scale_n_i32, fz_enable_scales, addr_disp, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    i32_cur_tok, dispatch_blocks, producer_slot, parity, expected, prefetched, route_rounds, prefetch_args,
):
# fmt: on
    """Allocate and publish routes directly into destination fixed slots."""
    # prefetch_direct_fixed_slot_route loaded this wave's first len(prefetched) routes before the epoch gate.
    p_source_done = prefetched[0]["tables"]["source_done"]
    a_producer_done = prefetched[0]["tables"]["producer_done"]

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    route = producer_slot * fx.Int32(num_waves) + warp
    route_stride = fx.Int32(dispatch_blocks * num_waves)
    route_limit = i32_cur_tok * fx.Int32(fz_k)

    # A wave owns routes route + r * route_stride for r < route_rounds; each round is a loop of at most one trip,
    # and rounds past the prefetched ones load their route after the gate.
    for rr in range_constexpr(route_rounds):
        for wk in range(route + fx.Int32(rr) * route_stride, route_limit, fx.Int32(fz_mtpr * fz_k)):
            if const_expr(rr < len(prefetched)):
                rpf = prefetched[rr]
            else:
                rpf = prefetch_direct_fixed_slot_route(**prefetch_args, route_offset=rr * dispatch_blocks * num_waves)
            source_token = wk // fx.Int32(fz_k)
            topk_slot = wk - source_token * fx.Int32(fz_k)
            row_values = rpf["row_values"]
            global_expert = rpf["global_expert"]
            valid_expert = (global_expert >= fx.Int32(0)) & (global_expert < fx.Int32(fz_total_experts))
            safe_expert = valid_expert.select(global_expert, fx.Int32(0))
            destination = safe_expert // fx.Int32(fz_epr)
            local_expert = safe_expert - destination * fx.Int32(fz_epr)
            offset_lane = fx.Int32(0)
            if lane == fx.Int32(0):
                if valid_expert:
                    offset_lane = fx.Int32(
                        comm_ops.atomic_add_system(
                            rpf["remote"]["running"] + fx.Int64(local_expert) * fx.Int64(4), fx.Int32(1)
                        )
                    )
            expert_offset = fx.Int32(fx.rocdl.readlane(T.i32, offset_lane, 0))
            publish = valid_expert & (expert_offset < fx.Int32(fz_cap))
            payload_row = local_expert * fx.Int32(fz_cap) + expert_offset

            if publish:
                destination_rsrc = AddrBuffer(rpf["remote"]["rx"] + fx.Int64(payload_row) * fx.Int64(fz_nbytes))
                for i in range_constexpr(len(row_values)):
                    store_at(row_values[i], destination_rsrc, lane * fx.Int32(4) + fx.Int32(i * 256))

                if const_expr(fz_enable_scales):
                    if lane < fx.Int32(fz_scale_n_i32):
                        store_at(rpf["scale"], AddrBuffer(rpf["remote"]["sc"]),
                            payload_row * fx.Int32(fz_scale_n_i32) + lane)

                if lane == fx.Int32(0):
                    weight_bits = fx.Vector.from_elements([rpf["weight"]], fx.Float32).bitcast(fx.Int32)[0]
                    source_encoding = (fx.Int32(fz_rank * fz_mtpr) + source_token) | (topk_slot << fx.Int32(24))
                    store_at(weight_bits, AddrBuffer(rpf["remote"]["wts"]), payload_row)
                    store_at(source_encoding, AddrBuffer(rpf["remote"]["sm"]), payload_row)

    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()
    # The last producer publishes; one lane per destination, so the remote release stores overlap.
    if warp == fx.Int32(0):
        done_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            comm_ops.fence_system_release()
            done_lane = fx.Int32(comm_ops.atomic_add_agent(a_producer_done, fx.Int32(1)))
        done = fx.Int32(fx.rocdl.readlane(T.i32, done_lane, 0))
        if done == fx.Int32(dispatch_blocks - 1):
            comm_ops.fence_agent_acquire()
            if lane < fx.Int32(fz_npes):
                remote_done = load_at(AddrBuffer(p_source_done), lane, fx.Int64, 1)
                done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
                comm_ops.store_i32_system(remote_done, done_index, expected)


@flyc.jit
def emit_direct_fixed_slot_finalize(
    *, fz_npes, fz_epr, fz_cap, fz_mtpr, fz_rank, fz_tile_m, addr_disp, parity, expected, plan_nv,
):
    """Finalize local fixed slots as soon as every source publishes this destination.

    ``plan_nv = (addr, epoch)``: the local plan is published as one i64 ``epoch << 32 | num_valid`` store, so
    consumers get num_valid from the flag itself. Only this rank's consumers read it. The srcmap of padding rows
    is left unwritten: GEMM2 masks rows past the tile's valid count itself.
    """
    assert 0 < fz_epr <= 64, "direct fixed-slot finalize requires 1..64 experts per rank"
    rdisp = AddrBuffer(addr_disp)

    def dp(i):
        return load_at(rdisp, fx.Int32(int(i)), fx.Int64, 1)

    a_se = dp(DispatchSlot.SORTED_EXPERT)
    a_trb = dp(DispatchSlot.TILE_ROW_BASE)
    a_nv = dp(DispatchSlot.NUM_VALID)
    a_running = dp(DispatchSlot.RUNNING)
    a_source_done = dp(DispatchSlot.COUNT_DONE)
    a_tile_rows = dp(DispatchSlot.TILE_ROWS)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    if warp == fx.Int32(0):
        for source in range(lane, fz_npes, 64):
            done_index = parity * fx.Int32(fz_npes) + source
            mori_shmem.int32_wait_until_equals(a_source_done + fx.Int64(done_index) * fx.Int64(4), expected)
        comm_ops.fence_system_acquire()

        valid_expert = lane < fx.Int32(fz_epr)
        safe_expert = valid_expert.select(lane, fx.Int32(0))
        count = load_at(AddrBuffer(a_running), safe_expert, fx.Int32, 1)
        count = valid_expert.select(count, fx.Int32(0))
        overflow_flag = (count > fx.Int32(fz_cap)).select(fx.Int32(1), fx.Int32(0))
        overflow_prefix = _wave_inclusive_scan_i32(overflow_flag, lane)
        overflow_count = fx.Int32(fx.rocdl.readlane(T.i32, overflow_prefix, fz_epr - 1))
        no_overflow = overflow_count == fx.Int32(0)
        safe_count = (count <= fx.Int32(fz_cap)).select(count, fx.Int32(0))
        num_expert_tiles = (safe_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
        inclusive_tiles = _wave_inclusive_scan_i32(num_expert_tiles, lane)
        metadata_base = inclusive_tiles - num_expert_tiles
        total_tiles = fx.Int32(fx.rocdl.readlane(T.i32, inclusive_tiles, fz_epr - 1))

        if valid_expert:
            if no_overflow:
                global_expert = fx.Int32(fz_rank * fz_epr) + safe_expert
                payload_base = safe_expert * fx.Int32(fz_cap)
                for tile in range(fx.Int32(0), num_expert_tiles, 1):
                    metadata_index = metadata_base + tile
                    store_at(global_expert, AddrBuffer(a_se), metadata_index)
                    store_at(payload_base + tile * fx.Int32(fz_tile_m), AddrBuffer(a_trb), metadata_index)
                    rows = safe_count - tile * fx.Int32(fz_tile_m)
                    rows = (rows < fx.Int32(fz_tile_m)).select(rows, fx.Int32(fz_tile_m))
                    store_at(rows, AddrBuffer(a_tile_rows), metadata_index)
            store_at(fx.Int32(0), AddrBuffer(a_running), safe_expert)

        if lane == fx.Int32(0):
            num_valid = no_overflow.select(total_tiles * fx.Int32(fz_tile_m), fx.Int32(0))
            store_at(num_valid, AddrBuffer(a_nv), fx.Int32(0))
            # num_valid[1] is a device-visible overflow status.
            store_at(overflow_count, AddrBuffer(a_nv), fx.Int32(1))

        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        comm_ops.fence_system_release()
        if lane == fx.Int32(0):
            nv_flag = no_overflow.select(total_tiles * fx.Int32(fz_tile_m), fx.Int32(0))
            comm_ops.store_i64_global_system(
                plan_nv[0], (fx.Int64(plan_nv[1]) << fx.Int64(32)) | fx.Int64(nv_flag))
    fx.barrier()


# fmt: off
def p2p_scatter_epilog(lds_acc_base, accm, n_block_idx, wave, lane, *, N_OUT, BM, BN, npes, topk,
    log2_max_tok, mask_max_tok, recv_cap, comb_inp_nbytes, lds_packed_off, lds_weight_off,
    lds_peer_off, g2_bf16_lds=False, p2p_quant_type="none", store_cm=2):
# fmt: on
    """CShuffle one GEMM2 tile into weighted BF16 rows and scatter them to peers.

    ``store_cm`` is the cache policy of the BF16 row stores (2 = NT).
    """
    kMChunks = BM // 16
    numAccN = (BN // 4) // 16
    wave_n = BN // 4
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    lds_base_bf16 = (
        lds_typed_ptr(lds_acc_base, T.bf16, align=2)
        if const_expr(g2_bf16_lds)
        else None
    )

    quant_fp8 = p2p_quant_type == "fp8_blockwise_1x32"
    if p2p_quant_type not in ("none", "fp8_blockwise_1x32"):
        raise ValueError(f"unsupported p2p_quant_type={p2p_quant_type!r}")
    out_elem_bytes = 1 if quant_fp8 else 2
    token_nbytes = N_OUT + N_OUT // 32 if quant_fp8 else N_OUT * out_elem_bytes

    fx.barrier()

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * 4
        if const_expr(g2_bf16_lds):
            w_row = [
                fx.ptr_load(
                    lds_typed_ptr(
                        fx.Int32(lds_weight_off) + (row_base + v) * fx.Int32(4),
                        T.f32,
                        align=4,
                    )
                )
                for v in range_constexpr(4)
            ]
        for J in range_constexpr(numAccN):
            col = wave * fx.Int32(wave_n) + J * 16 + lane_mod_16
            vec = fx.Vector(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + v) * BN + col
                if const_expr(g2_bf16_lds):
                    lds_base_bf16[idx] = fx.BFloat16(fx.Float32(vec[v]) * fx.Float32(w_row[v]))
                else:
                    lds_base_fptr[idx] = fx.Float32(vec[v])

    fx.barrier()

    for row_iter in range_constexpr(BM // 4):
        row = wave + fx.Int32(row_iter * 4)
        row_byte_off = row * fx.Int32(4)
        p = fx.ptr_load(
            lds_typed_ptr(fx.Int32(lds_packed_off) + row_byte_off, T.i32, align=4)
        )
        if const_expr(not g2_bf16_lds):
            weight = fx.ptr_load(
                lds_typed_ptr(fx.Int32(lds_weight_off) + row_byte_off, T.f32, align=4)
            )
        p = rocdl.readfirstlane(T.i32, p.ir_value())
        if const_expr(not g2_bf16_lds):
            weight = rocdl.readfirstlane(T.f32, weight.ir_value())
        t = p & fx.Int32(0x00FFFFFF)
        s = p >> fx.Int32(24)
        dest_pe = t >> fx.Int32(log2_max_tok)
        dest_lid = t & fx.Int32(mask_max_tok)
        valid = (t < fx.Int32(recv_cap)) & (s < fx.Int32(topk)) & (dest_pe < fx.Int32(npes))
        dest_pe_safe = valid.select(dest_pe, fx.Int32(0))
        peer_base = fx.ptr_load(
            lds_typed_ptr(
                fx.Int32(lds_peer_off) + dest_pe_safe * fx.Int32(8),
                T.i64,
                align=8,
            )
        )
        peer_base = rocdl.readfirstlane(T.i64, peer_base.ir_value())
        rsrc_dst = AddrBuffer(peer_base, comb_inp_nbytes)
        slot = dest_lid * fx.Int32(topk) + s
        row_base = slot * fx.Int32(token_nbytes)
        row_off = row_base + n_block_idx * fx.Int32(BN * out_elem_bytes)

        # Inactive lanes read safe LDS and issue a bounded OOB store.
        active = lane < fx.Int32(BN // 8)
        col = active.select(lane * fx.Int32(8), fx.Int32(0))
        idx0 = row * fx.Int32(BN) + col
        if const_expr(g2_bf16_lds):
            pk = fx.Vector(
                lds_vec_load(
                    lds_acc_base,
                    idx0 * fx.Int32(2),
                    fx.Vector.make_type(8, fx.BFloat16),
                    fx.BFloat16,
                    align=16,
                )
            )
        else:
            v8 = fx.Vector(
                lds_vec_load(
                    lds_acc_base,
                    idx0 * fx.Int32(4),
                    fx.Vector.make_type(8, fx.Float32),
                    fx.Float32,
                    align=16,
                )
            )
            weighted_v8 = fx.Vector.from_elements([v8[i] * weight for i in range_constexpr(8)], fx.Float32)
            if const_expr(not quant_fp8):
                pk = weighted_v8.to(fx.BFloat16)
        if const_expr(quant_fp8):
            vals = [fx.Float32(weighted_v8[i]) for i in range_constexpr(8)]
            local_max = fabs_f32(vals[0])
            for q in range_constexpr(1, 8):
                local_max = local_max.maximumf(fabs_f32(vals[q]))
            max_bits = local_max.bitcast(fx.Int32)
            for xor_lane in (1, 2):
                remote_bits = rocdl.ds_bpermute(
                    T.i32,
                    (lane ^ fx.Int32(xor_lane)) * fx.Int32(4),
                    max_bits,
                )
                remote_max = fx.Int32(remote_bits).bitcast(fx.Float32)
                local_max = local_max.maximumf(remote_max)
                max_bits = local_max.bitcast(fx.Int32)
            leader_lane = lane & fx.Int32(~3)
            is_scale_leader = (lane & fx.Int32(3)) == fx.Int32(0)
            leader_e8m0 = _fp8_scale_for_leader(is_scale_leader, local_max)
            e8m0 = fx.Int32(
                rocdl.ds_bpermute(
                    T.i32,
                    leader_lane * fx.Int32(4),
                    leader_e8m0,
                )
            )
            block_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
            pk_ty = T.vec(2, T.i16)
            packed_lo = fx.Vector.filled(2, 0, fx.Int16).ir_value()
            packed_hi = fx.Vector.filled(2, 0, fx.Int16).ir_value()
            for pair in range_constexpr(4):
                if pair < 2:
                    packed_lo = rocdl.cvt_scalef32_pk_fp8_f32(
                        pk_ty,
                        packed_lo,
                        vals[pair * 2].ir_value(),
                        vals[pair * 2 + 1].ir_value(),
                        block_scale.ir_value(),
                        pair,
                    )
                else:
                    packed_hi = rocdl.cvt_scalef32_pk_fp8_f32(
                        pk_ty,
                        packed_hi,
                        vals[pair * 2].ir_value(),
                        vals[pair * 2 + 1].ir_value(),
                        block_scale.ir_value(),
                        pair - 2,
                    )
            payload = fx.Vector.from_elements(
                [
                    fx.Vector(packed_lo).bitcast(fx.Int32)[0],
                    fx.Vector(packed_hi).bitcast(fx.Int32)[0],
                ],
                fx.Int32,
            )
            scale_leader = active & ((lane & fx.Int32(3)) == fx.Int32(0))
            payload_off = (valid & active).select(
                row_off + col,
                fx.Int32(comb_inp_nbytes),
            )
            # Adjacent active lanes issue contiguous 8-byte stores without ds_bpermute gathers.
            store_at(payload.ir_value(), rsrc_dst, payload_off, 2, offset_is_bytes=True)

            @flyc.jit
            def store_scale_if_leader():
                if scale_leader:
                    scale_off = valid.select(
                        row_base
                        + fx.Int32(N_OUT)
                        + n_block_idx * fx.Int32(BN // 32)
                        + lane // fx.Int32(4),
                        fx.Int32(comb_inp_nbytes),
                    )
                    store_at(e8m0.to(fx.Int8), rsrc_dst, scale_off, 2, offset_is_bytes=True)

            store_scale_if_leader()
        else:
            off = (valid & active).select(
                row_off + col * fx.Int32(out_elem_bytes),
                fx.Int32(comb_inp_nbytes),
            )
            store_at(pk.ir_value(), rsrc_dst, off, store_cm, offset_is_bytes=True)


def make_fused_combine_tail(*, lds_fc_off, reducers, npes, rank, topk, N_OUT, flag_is_value=False, l1_acquire=False):
    """Combine after this WG's P2P scatter: grid ticket, cross-device flag, reduce by the last WGs.

    ``flag_is_value``: ``arg_flag`` is this launch's flag itself (identical on every rank) instead of the address
    of a counter the last WG advances.
    ``l1_acquire``: the reducers acquire the peers' rows by dropping only the CU's L1. Valid when the combine input
    lives in the uncached shmem heap (never held in L2) and the local rows were released by the last WG.

    Matches ep_combine_intranode(skip_stage1=True, bf16): out[t] = bf16(sum_k f32(inp[t * topk + k])).
    """
    row_i32 = N_OUT * 2 // 4
    chunks = row_i32 // 256
    assert row_i32 % 256 == 0

    # fmt: off
    @flyc.jit
    def fused_combine_tail(tx_i32: fx.Int32, lane: fx.Int32, wave: fx.Int32, arg_cnt: fx.Int64, arg_flag: fx.Int64,
        arg_xdb: fx.Int64, arg_p2p_xdb: fx.Int64, arg_inp: fx.Int64, arg_out: fx.Int64, arg_trecv: fx.Int64,
        i32_cur_tok: fx.Int32, i32_grid: fx.Int32):
    # fmt: on
        ticket_ptr = lds_typed_ptr(fx.Int32(lds_fc_off), T.i32, align=4)
        flag_ptr = lds_typed_ptr(fx.Int32(lds_fc_off + 8), T.i64, align=8)
        last = i32_grid - fx.Int32(1)
        # Every wave waits for its own P2P stores before the WG takes its ticket.
        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()
        if tx_i32 == fx.Int32(0):
            if const_expr(flag_is_value):
                flag_lane = arg_flag
            else:
                flag_lane = fx.Int64(comm_ops.load_i64_global(arg_flag))
            # The P2P rows target the uncached shmem heap, so store completion makes them visible to peers
            # (a per-WG system release would write back L2 thousands of times). The publisher releases below.
            ticket_lane = fx.Int32(comm_ops.atomic_add_agent(arg_cnt, fx.Int32(1)))
            fx.ptr_store(ticket_lane, ticket_ptr)
            fx.ptr_store(flag_lane, flag_ptr)
        fx.barrier()
        ticket = rocdl.readfirstlane(T.i32, fx.ptr_load(ticket_ptr).ir_value())
        flag = fx.ptr_load(flag_ptr)
        if fx.Int32(ticket) == last:
            if tx_i32 < fx.Int32(npes):
                # Every other WG's stores completed before its ticket; acquire them, then publish to peer tx.
                comm_ops.fence_system_acquire()
                comm_ops.fence_system_release()
                remote = load_at(AddrBuffer(arg_p2p_xdb), tx_i32, fx.Int64, 1)
                comm_ops.store_i64_global_system(fx.Int64(remote) + fx.Int64(rank * 8), flag)
            fx.barrier()
            if tx_i32 == fx.Int32(0):
                if const_expr(not flag_is_value):
                    comm_ops.store_i64_global(arg_flag, flag + fx.Int64(1))
                store_at(fx.Int32(0), AddrBuffer(arg_cnt), fx.Int32(0))
                store_at(fx.Int32(0), AddrBuffer(arg_trecv), fx.Int32(0))
        nred = (i32_grid < fx.Int32(reducers)).select(i32_grid, fx.Int32(reducers))
        first = i32_grid - nred
        if fx.Int32(ticket) >= first:
            if tx_i32 < fx.Int32(npes):
                mori_shmem.uint64_wait_until_equals(arg_xdb + fx.Int64(tx_i32) * fx.Int64(8), flag)
                if const_expr(l1_acquire):
                    comm_ops.invalidate_l1()
                else:
                    comm_ops.fence_system_acquire()
            fx.barrier()
            inp_rsrc = AddrBuffer(arg_inp)
            out_rsrc = AddrBuffer(arg_out)
            items = i32_cur_tok * fx.Int32(chunks)
            worker = (fx.Int32(ticket) - first) * fx.Int32(4) + wave
            for item in range(worker, items, nred * fx.Int32(4)):
                tok = item // fx.Int32(chunks)
                col = (item - tok * fx.Int32(chunks)) * fx.Int32(256) + lane * fx.Int32(4)
                src = tok * fx.Int32(topk * row_i32) + col
                vals = [
                    load_at(inp_rsrc, src + fx.Int32(k * row_i32), fx.Int32, 4, 2)
                    for k in range_constexpr(topk)
                ]
                acc = fx.Vector(vals[0]).bitcast(fx.BFloat16).to(fx.Float32)
                for k in range_constexpr(1, topk):
                    acc = acc + fx.Vector(vals[k]).bitcast(fx.BFloat16).to(fx.Float32)
                store_at(acc.to(fx.BFloat16).bitcast(fx.Int32).ir_value(), out_rsrc, tok * fx.Int32(row_i32) + col, 2)

    return fused_combine_tail


def g2_lds_layout(*, BM, BN, BK, a_dtype, npes, g2_bf16_lds=False):
    """Absolute LDS offsets of the stage2 tile (compute slab, packed srcmap, weights, peer table, fc scratch)."""
    compute = _stage2_lds_bytes(BM, BN, BK, a_dtype, kStages + 1, g2_bf16_lds)
    packed = compute
    weight = packed + BM * 4
    peer = weight + BM * 4
    fc = peer + npes * 8
    return {"packed": packed, "weight": weight, "peer": peer, "fc": fc, "bytes": fc + 16}


# fmt: off
def make_g2_tile_runner(*, BM, BN, BK, SBM, use_nt, INTER_MAX, a_dtype, g2_bhoist, g2_ascale_pf, expert_offset,
    npes, topk, N_OUT, max_tok, recv_cap, comb_inp_nbytes, clamp_a, store_cm=2, row_mask=False):
# fmt: on
    """One stage2 tile (m_block, n_block) as a jit helper, for kernels that run GEMM2 after their own phase.

    Uses the absolute LDS layout of ``g2_lds_layout`` (the caller's LDS pool must start at address 0); the peer
    table at ``layout['peer']`` must be loaded by the caller. Same numerics as the stage2 kernel.
    ``row_mask``: rows past the tile's valid count are dropped here, so the plan need not pad their srcmap.
    """
    assert clamp_a or not row_mask
    lay = g2_lds_layout(BM=BM, BN=BN, BK=BK, a_dtype=a_dtype, npes=npes)
    is_f8 = a_dtype == "fp8"
    aStages = kStages + 1
    KH_TILE_A = BK // (1 if is_f8 else 2)
    log2_max_tok = max_tok.bit_length() - 1

    # fmt: off
    @flyc.jit
    def g2_tile(lds_base_i32: fx.Int32, tx_i32: fx.Int32, lane: fx.Int32, wave: fx.Int32, m_block: fx.Int32,
        n_block: fx.Int32, arg_aq: fx.Int64, arg_ascale: fx.Int64, arg_bq: fx.Int64, arg_bscale: fx.Int64,
        arg_eids: fx.Int64, arg_stids: fx.Int64, arg_sweights: fx.Int64, arg_trb: fx.Int64, arg_tile_rows: fx.Int64,
        i32_max_m_blocks: fx.Int32, i32_inter: fx.Int32, i32_hidden: fx.Int32):
    # fmt: on
        num_n_blocks = fx.Int32(i32_hidden) // fx.Int32(BN)
        k_bytes = fx.Int32(i32_inter) // fx.Int32(1 if is_f8 else 2)
        vrows = None
        if const_expr(clamp_a):
            vrows = rocdl.readfirstlane(T.i32, load_at(AddrBuffer(arg_tile_rows), m_block, fx.Int32, 1))
        m_row0 = m_block * fx.Int32(BM)
        for slot in range_constexpr(kStages):
            issue_a_load_lds_dt(arg_aq, lds_base_i32, slot, slot, m_row0, wave, lane, is_f8, KH_TILE_A, k_bytes,
                BM=BM, valid_rows=vrows)
        rocdl.sched_barrier(0)
        sort_block_idx = m_row0 // fx.Int32(SBM)
        srcmap_row_base = load_at(AddrBuffer(arg_trb), sort_block_idx, fx.Int32, 1) + (m_row0 - sort_block_idx * fx.Int32(SBM))
        if tx_i32 < fx.Int32(BM):
            sorted_pos = srcmap_row_base + tx_i32
            packed = load_at(AddrBuffer(arg_stids), sorted_pos, fx.Int32, 1)
            if const_expr(row_mask):
                packed = (tx_i32 < vrows).select(packed, fx.Int32(npes * max_tok))
            weight = load_at(AddrBuffer(arg_sweights), sorted_pos, fx.Float32, 1)
            fx.ptr_store(packed, lds_typed_ptr(fx.Int32(lay["packed"]) + tx_i32 * fx.Int32(4), T.i32, align=4))
            fx.ptr_store(weight, lds_typed_ptr(fx.Int32(lay["weight"]) + tx_i32 * fx.Int32(4), T.f32, align=4))
        unit_bx = m_block * num_n_blocks + n_block
        # fmt: off
        accm_vecs, _m_row, n_block_idx, _n_out_rt = gemm2_compute_v2(lds_base_i32, arg_ascale, arg_bq,
            arg_bscale, arg_eids, arg_aq, i32_max_m_blocks, unit_bx, lane, wave, i32_inter, i32_hidden,
            fx.Int32(0), fx.Int32(0), BM=BM, BN=BN, BK=BK, use_nt=use_nt, INTER_MAX=INTER_MAX, aStages=aStages,
            a_dtype=a_dtype, has_pad=False, SBM=SBM, g2_bhoist=g2_bhoist, g2_ascale_pf=g2_ascale_pf,
            expert_offset=expert_offset, valid_rows=vrows)
        p2p_scatter_epilog(lds_base_i32, accm_vecs, n_block_idx, wave, lane, N_OUT=N_OUT, BM=BM, BN=BN, npes=npes,
            topk=topk, log2_max_tok=log2_max_tok, mask_max_tok=max_tok - 1, recv_cap=recv_cap,
            comb_inp_nbytes=comb_inp_nbytes, lds_packed_off=lay["packed"], lds_weight_off=lay["weight"],
            lds_peer_off=lay["peer"], g2_bf16_lds=False, p2p_quant_type="none", store_cm=store_cm)
        # fmt: on

    return g2_tile, lay


_SC0_SC1_CACHE = 17


def _use_direct_fixed_slot(npes, experts_per_rank, max_tokens_per_rank, cap, tile_m):
    if tile_m <= 0 or max_tokens_per_rank <= 0:
        return False
    required_cap = ((npes * max_tokens_per_rank + tile_m - 1) // tile_m) * tile_m
    return npes == 8 and experts_per_rank == 48 and cap == required_cap


# fmt: off
@functools.cache
def compile_mega_moe_stage1(
    *, model_dim: int, inter_dim: int, rank: int, experts_per_rank: int, fuse_npes: int, fuse_topk: int,
    fuse_cap: int, fuse_mtpr: int, fuse_scale_dim: int, fused: tuple, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 8,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, num_dispatch_cu: int = 32, b_nt: int = -1,
    swiglu_limit: float = 0.0,
):
    arch = get_gfx_runtime()
    if arch != "gfx950":
        raise RuntimeError(f"MegaMoE stage1 requires gfx950, got {arch}")
    NUM_WAVES = int(num_waves)
    assert NUM_WAVES > 1, "planner needs one communication wave and at least one grouping wave"
    assert 1 <= waves_per_eu_hint <= 4
    assert tile_n % NUM_WAVES == 0
    n_per_wave = tile_n // NUM_WAVES
    assert (2 * inter_dim) % tile_n == 0, "2*inter_dim must tile evenly by tile_n"
    N_TILES = (2 * inter_dim) // tile_n
    GRID_MULT_VALUES = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32)
    assert grid_mult in GRID_MULT_VALUES, "grid_mult out of range"
    dispatch_blocks = int(num_dispatch_cu)
    assert 0 < dispatch_blocks < num_cu * grid_mult - 1, "num_dispatch_cu must leave consumers"
    assert dispatch_blocks % fuse_npes == 0, "num_dispatch_cu must be divisible by fuse_npes"
    planner_blocks = 1
    # Keep the fused grid on an exact CU multiple instead of appending control/producer CTAs as a tail.
    grid_x = num_cu * grid_mult - planner_blocks - dispatch_blocks
    assert grid_x > 0, "consumer grid must remain positive"
    launch_grid_x = planner_blocks + dispatch_blocks + grid_x
    assert launch_grid_x <= num_cu * 33 + 1
    M_REPEAT = sort_block_m // 16
    NUM_ACC_N = n_per_wave // 16
    assert NUM_ACC_N % 2 == 0 and M_REPEAT % 2 == 0

    TILE_K_BYTES = tile_k // 2
    assert TILE_K_BYTES % 128 == 0
    A_K_STEP_BYTES = tile_k
    assert A_K_STEP_BYTES == 256, "MegaMoE v2 GEMM1 requires tile_k=256"
    K_ITERS = model_dim // tile_k
    TOTAL_THREADS = NUM_WAVES * 64

    a_lds_size = sort_block_m * A_K_STEP_BYTES
    a_lds_i32 = a_lds_size // 4
    cs_tile_n = tile_n // 2
    cs_size = sort_block_m * cs_tile_n
    lds_pool_bytes = max(2 * a_lds_size, cs_size * 4)
    n_scale_bytes = sort_block_m * (model_dim // 32)

    fz_npes, fz_epr, fz_k = int(fuse_npes), int(experts_per_rank), int(fuse_topk)
    fz_cap, fz_mtpr, fz_rank = int(fuse_cap), int(fuse_mtpr), int(rank)
    if fz_npes * fz_mtpr > 1 << 24:
        raise ValueError("MegaMoE v2 source-token encoding exceeds 24 bits")
    if fz_k > 1 << 8:
        raise ValueError("MegaMoE v2 top-k slot encoding exceeds 8 bits")
    fz_tile_m = int(sort_block_m)
    assert fz_cap % fz_tile_m == 0, f"fuse_cap({fz_cap}) % tile_m({fz_tile_m}) != 0"
    if not _use_direct_fixed_slot(fz_npes, fz_epr, fz_mtpr, fz_cap, fz_tile_m):
        raise ValueError("small-input stage1 needs the 8-rank, 48-expert fixed-slot layout")
    fz_total_experts = fz_npes * fz_epr
    # Producers load their route (expert, remote pointers, row, scales, weight) before the epoch gate and
    # quantize the bf16 input row to MXFP8. A producer wave sends pre_rounds routes (one per round) so
    # that npes groups of mtpr * topk routes fit.
    pre_rounds = ceildiv(fz_mtpr * fz_k, dispatch_blocks * NUM_WAVES)
    # Small batches stream B; large batches cache it across M tiles.
    b_cache_modifier = int(b_nt) if int(b_nt) >= 0 else (3 if fz_mtpr <= 512 else 0)
    fz_n_i32, fz_nbytes = model_dim // 4, model_dim
    fz_scale_bytes = int(fuse_scale_dim)
    fz_scale_n_i32 = (fz_scale_bytes + 3) // 4 if fz_scale_bytes > 0 else 0
    if fz_scale_n_i32 > 64:
        raise ValueError("direct fixed-slot dispatch supports at most 64 packed scale columns")
    fz_enable_scales = fz_scale_bytes > 0
    _validate_dispatch_capacity(
        fz_mtpr, fz_npes, fz_epr, fz_k, fz_tile_m, fz_nbytes, inter_dim, use_tile_resource
    )

    # fused = (BN, BK, recv_cap, fc_reducers): after its GEMM1 tiles every CTA runs GEMM2 + P2P scatter tiles
    # of its XCD's m-tiles (each gated on that m-tile's GEMM1 completion) and then the fused combine tail.
    # Finalize publishes per-tile valid rows that clamp the GEMM1 A loads and epilogue stores, and each XCD
    # drains a contiguous slice of the m-major work space.
    g2_bn, g2_bk, g2_recv_cap, fused_reducers = fused
    assert fused_reducers > 0, "the fused combine needs reducers"
    assert model_dim % g2_bn == 0 and NUM_WAVES == 4 and fz_npes <= 47
    g2_n_blocks = model_dim // g2_bn
    # The P2P row stores go to the uncached shmem heap; non-temporal stores there are slower (store_cm=0).
    # GEMM2 masks the padding rows of each tile itself, so the plan leaves their srcmap unwritten.
    g2_tile, g2_lay = make_g2_tile_runner(
        BM=sort_block_m, BN=g2_bn, BK=g2_bk, SBM=sort_block_m, use_nt=True, INTER_MAX=inter_dim,
        a_dtype="fp8", g2_bhoist=True, g2_ascale_pf=True, expert_offset=int(rank) * int(experts_per_rank),
        npes=int(fuse_npes), topk=int(fuse_topk), N_OUT=model_dim, max_tok=int(fuse_mtpr),
        recv_cap=int(g2_recv_cap), comb_inp_nbytes=int(fuse_mtpr) * int(fuse_topk) * model_dim * 2,
        clamp_a=True, store_cm=0, row_mask=True,
    )
    # The combine input is in the uncached shmem heap: the reducers only need to drop their L1.
    fused_combine_tail = make_fused_combine_tail(lds_fc_off=g2_lay["fc"], reducers=fused_reducers,
        npes=int(fuse_npes), rank=int(rank), topk=int(fuse_topk), N_OUT=model_dim, flag_is_value=True,
        l1_acquire=True)
    g2_scratch_off = g2_lay["bytes"]
    lds_pool_bytes = max(lds_pool_bytes, (g2_scratch_off + 16 + 15) // 16 * 16)
    # The combine reducers spin until every CTA has taken its ticket, so no CTA may still be waiting for a
    # slot: the whole grid must be co-resident (2 CTAs per CU: <= 256 VGPRs and half the 160 KiB LDS).
    assert grid_mult <= 2 and 2 * lds_pool_bytes <= 160 * 1024, (grid_mult, lds_pool_bytes)
    # Every launch ends in a cross-rank exchange, so a rank can only enter launch N+1 after every peer has
    # entered launch N (its plan waits for all sources) and finished launch N-1. All cross-rank buffers a launch
    # writes before its own plan are parity double-buffered, so there is no launch handshake:
    #  - the owner resets the counters and advances the epoch words in wave 0 lanes, with a single release;
    #  - the gate needs no acquire: the epoch words are read past L2 and the counters are only touched by atomics;
    #  - the plan flag is one i64 (gate_epoch << 32 | num_valid), and the first work fetch overlaps its wait;
    #  - only the CTA holding an XCD's first work unit invalidates that XCD's L2 for the plan; the others wait for
    #    its republished flag and drop their L1.
    # Launches of every grid size share one entry slot (each CTA adds 2 // grid_mult, so every launch adds
    # 2 * num_cu): the plan, G2-ready and combine flags compare epochs, and per-grid-size epochs would let a flag
    # left by a launch of the other grid size pass (or block) the wait.
    entry_slot = len(GRID_MULT_VALUES)
    entry_inc = 2 // grid_mult
    entry_unit = 2 * num_cu
    assert launch_grid_x * entry_inc == entry_unit

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]
        # Work-unit broadcast word, kept out of the pool: a fast wave's next A / C-shuffle store must not race the
        # slower waves' read of the unit it starts.
        work_word: fx.Array[fx.Int8, 16, 16]

    swiglu_suffix = "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    kernel_name = (
        f"megamoe_small_stage1_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_w{NUM_WAVES}_gm{grid_mult}"
        f"_dcu{dispatch_blocks}_pw{int(pipe_weights)}ma{int(mfma_amajor)}sw{int(swizzle_a)}"
        f"aa{int(async_a_copy)}"
        f"_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}_bnt{b_cache_modifier}"
        f"{swiglu_suffix}"
        f"_fused{'_'.join(str(v) for v in fused)}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
    def kernel(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        a_buf = lds.pool
        a_scale_lds = lds.A_scale
        c_tile = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
        disp_rsrc = _make_buffer_from_addr(addr_disp, fx.Int64)
        parity_rsrc = _make_buffer_from_addr(addr_parity, fx.Int32)
        expected_rsrc = _make_buffer_from_addr(addr_expected, fx.Int32)

        def _disp_ptr(slot):
            return _buffer_load(disp_rsrc, fx.Int32(int(slot)), fx.Int64)

        a_entry_count = _disp_ptr(DispatchSlot.ENTRY_COUNT)
        a_epoch_gate = _disp_ptr(DispatchSlot.EPOCH_GATE)
        a_work_head = _disp_ptr(DispatchSlot.WORK_HEAD)
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)

        ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        if tid == fx.Int32(0):
            ticket64 = fx.Int64(
                comm_ops.atomic_add_agent(a_entry_count + fx.Int64(entry_slot * 8), fx.Int64(entry_inc))
            )
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        generation = ticket64 // fx.Int64(entry_unit)
        ticket = fx.Int32(ticket64 - generation * fx.Int64(entry_unit))
        if const_expr(entry_inc > 1):
            ticket = ticket // fx.Int32(entry_inc)
        gate_addr = a_epoch_gate + fx.Int64(entry_slot * 4)
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        owner = ticket == fx.Int32(0)
        producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(dispatch_blocks))

        def _wait_gate():
            mori_shmem.int32_wait_until_equals(gate_addr, gate_epoch)

        def _load_epoch():
            # Independent loads (no parity -> expected chain); only valid after the gate.
            parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_SC1_CACHE)
            expected0 = _buffer_load(expected_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_SC1_CACHE)
            expected1 = _buffer_load(expected_rsrc, fx.Int32(1), fx.Int32, cache_modifier=_SC0_SC1_CACHE)
            return parity, (parity == fx.Int32(0)).select(expected0, expected1)
        producer_slot = ticket - fx.Int32(1)

        if owner:
            if tid < fx.Int32(64):
                work_head_rsrc = _make_buffer_from_addr(a_work_head, fx.Int32)
                if tid < fx.Int32(8):
                    # The XCD's work counter and its counts of its own GEMM1 units in the m-tiles cut at its
                    # slice ends.
                    _buffer_store(work_head_rsrc, tid * fx.Int32(16), fx.Int32(0), fx.Int32)
                    _buffer_store(work_head_rsrc, tid * fx.Int32(16) + fx.Int32(1), fx.Int32(0), fx.Int32)
                    _buffer_store(work_head_rsrc, tid * fx.Int32(16) + fx.Int32(2), fx.Int32(0), fx.Int32)
                if tid == fx.Int32(16):
                    _buffer_store(_make_buffer_from_addr(a_group_done, fx.Int32), fx.Int32(0), fx.Int32(0), fx.Int32)
                old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
                old_expected0 = _buffer_load(expected_rsrc, fx.Int32(0), fx.Int32)
                old_expected1 = _buffer_load(expected_rsrc, fx.Int32(1), fx.Int32)
                next_parity = old_parity ^ fx.Int32(1)
                next_expected = (
                    (next_parity == fx.Int32(0)).select(old_expected0, old_expected1) + fx.Int32(fz_npes)
                )
                if tid == fx.Int32(0):
                    _buffer_store(expected_rsrc, next_parity, next_expected, fx.Int32)
                    _buffer_store(parity_rsrc, fx.Int32(0), next_parity, fx.Int32)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                comm_ops.fence_agent_release()
                if tid == fx.Int32(0):
                    comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.barrier()
        else:
            # Producers wait for the gate after prefetching their route (below).
            if tid == fx.Int32(0):
                if ticket > fx.Int32(dispatch_blocks):
                    _wait_gate()
            fx.barrier()

        if producer:
            prefetch_args = dict(  # noqa: C408
                num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_mtpr=fz_mtpr,
                fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes, fz_n_i32=fz_n_i32,
                fz_scale_n_i32=fz_scale_n_i32, addr_disp=addr_disp, addr_in_tok=addr_in_tok,
                addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                i32_cur_tok=i32_cur_tok, dispatch_blocks=dispatch_blocks, producer_slot=producer_slot,
            )
            # Only the first round loads before the gate; the payload loop loads the later rounds itself.
            prefetched = [prefetch_direct_fixed_slot_route(**prefetch_args, route_offset=0)]
            if tid == fx.Int32(0):
                _wait_gate()
            fx.barrier()
            route_parity, route_expected = _load_epoch()
            emit_direct_fixed_slot_payload(
                num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_cap=fz_cap,
                fz_mtpr=fz_mtpr, fz_rank=fz_rank, fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes,
                fz_n_i32=fz_n_i32,
                fz_scale_n_i32=fz_scale_n_i32, fz_enable_scales=fz_enable_scales, addr_disp=addr_disp,
                addr_in_tok=addr_in_tok, addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                i32_cur_tok=i32_cur_tok, dispatch_blocks=dispatch_blocks, producer_slot=producer_slot,
                parity=route_parity, expected=route_expected, prefetched=prefetched, route_rounds=pre_rounds,
                prefetch_args=prefetch_args,
            )
        payload_parity, payload_expected = _load_epoch()
        if owner:
            emit_direct_fixed_slot_finalize(
                fz_npes=fz_npes, fz_epr=fz_epr, fz_cap=fz_cap, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                fz_tile_m=fz_tile_m, addr_disp=addr_disp, parity=payload_parity,
                expected=payload_expected, plan_nv=(_disp_ptr(DispatchSlot.PLAN_NV), gate_epoch),
            )
        wave_id = fx.thread_idx.x // 64

        w_rsrc = _make_buffer(w, fx.Int32, 4)
        sx_rsrc = _make_buffer(scale_x, fx.Int32, 4)
        sw_rsrc = _make_buffer(scale_w, fx.Int32)
        trb_rsrc = _make_buffer(sorted_token_ids, fx.Int32)
        expert_rsrc = _make_buffer(expert_ids, fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_nbytes = tokens * fx.Int32(scale_cols) + fx.Int32(8192)
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_nbytes = tokens * fx.Int32(inter_dim)
            out_rsrc = _make_buffer(out, fx.Int16, max_size=False, num_records_bytes=out_nbytes)
        os_rsrc = _make_buffer(out_scale, fx.Int8, max_size=False, num_records_bytes=os_nbytes)

        _, _do_scheduled_tile = build_fused_gemm1(
            x_tensor=x, w_rsrc=w_rsrc,
            sw_rsrc=sw_rsrc, sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc,
            trb_rsrc=trb_rsrc, expert_rsrc=expert_rsrc, out_tensor=out,
            a_buf=a_buf, a_scale_lds=a_scale_lds, c_tile=c_tile,
            model_dim=model_dim, inter_dim=inter_dim, sort_block_m=sort_block_m,
            tile_n=tile_n, num_waves=NUM_WAVES, n_per_wave=n_per_wave, wave_id=wave_id,
            m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_k_step_bytes=A_K_STEP_BYTES,
            total_threads=TOTAL_THREADS, k_iters=K_ITERS, a_lds_i32=a_lds_i32,
            n_tiles=N_TILES, expert_offset=fz_rank * fz_epr, b_cache_modifier=b_cache_modifier,
            swizzle_a=swizzle_a, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
            async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
            swiglu_limit=swiglu_limit,
            rows_rsrc=_make_buffer_from_addr(_disp_ptr(DispatchSlot.TILE_ROWS), fx.Int32),
            lookahead=1,
        )

        a_g2_cnt = _disp_ptr(DispatchSlot.G2_CNT)
        a_g2_ready = _disp_ptr(DispatchSlot.G2_READY)
        # GEMM2 setup is loaded before GEMM1 so the GEMM1 -> GEMM2 switch costs no memory round trip.
        g2_lane = tid % fx.Int32(64)
        g2_wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // fx.Int32(64)))
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.pool.ptr))
        g2_peer_base = _buffer_load(_make_buffer_from_addr(_disp_ptr(DispatchSlot.G2_P2P_COMB_INP), fx.Int64),
            (tid < fx.Int32(fz_npes)).select(tid, fx.Int32(0)), fx.Int64)
        g2_aq = _disp_ptr(DispatchSlot.G2_AQ)
        g2_ascale = _disp_ptr(DispatchSlot.G2_ASCALE)
        g2_bq = _disp_ptr(DispatchSlot.G2_BQ)
        g2_bscale = _disp_ptr(DispatchSlot.G2_BSCALE)
        g2_eids = _disp_ptr(DispatchSlot.G2_EIDS)
        g2_stids = _disp_ptr(DispatchSlot.G2_STIDS)
        g2_sweights = _disp_ptr(DispatchSlot.G2_SWEIGHTS)
        g2_trb = _disp_ptr(DispatchSlot.G2_TRB)
        g2_rows = _disp_ptr(DispatchSlot.TILE_ROWS)
        g2_max_m_blocks = fx.Int32(_disp_ptr(DispatchSlot.G2_MAX_M_BLOCKS))
        g2_scratch = lds_typed_ptr(fx.Int32(g2_scratch_off), T.i32, align=4)

        first_local = fx.Int32(0)
        nv_ptr = lds_typed_ptr(fx.Int32(g2_scratch_off + 4), T.i32, align=4)
        if tid == fx.Int32(0):
            xcd = fx.Int32(comm_ops.hw_xcc_id()) & fx.Int32(7)
            # The owner reset the counters before opening the gate: the first work fetch overlaps the plan wait.
            first_local = fx.Int32(
                comm_ops.atomic_add_agent(a_work_head + fx.Int64(xcd) * fx.Int64(64), fx.Int32(1)))
            plan_floor = (fx.Int64(gate_epoch) << fx.Int64(32)) - fx.Int64(1)
            a_xcd_acq = _disp_ptr(DispatchSlot.XCD_ACQ) + fx.Int64(xcd) * fx.Int64(128)
            plan_word = fx.Int64(0)
            if first_local == fx.Int32(0):
                plan_word = fx.Int64(mori_shmem.int64_wait_until_greater_than(
                    _disp_ptr(DispatchSlot.PLAN_NV), plan_floor))
                comm_ops.fence_agent_acquire()
                comm_ops.store_i64_global_system(a_xcd_acq, plan_word)
            else:
                plan_word = fx.Int64(mori_shmem.int64_wait_until_greater_than(a_xcd_acq, plan_floor))
                comm_ops.invalidate_l1()
            fx.ptr_store(fx.Int32(plan_word & fx.Int64(0xFFFFFFFF)), nv_ptr)
        fx.barrier()
        num_valid = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.ptr_load(nv_ptr).ir_value()))
        num_m_tiles = ceildiv(num_valid, fx.Int32(sort_block_m))
        total_work = num_m_tiles * fx.Int32(N_TILES)

        # Control CTAs join the work pool after dispatch.
        consumer_active = fx.Int32(1) == fx.Int32(1)
        work_scratch = fx.recast_iter(fx.Int32, lds.work_word.ptr)
        work_scratch_view = fx.make_view(work_scratch, fx.make_layout(1, 1))
        work_shard = fx.Int32(comm_ops.hw_xcc_id()) & fx.Int32(7)
        # Equal slices of the unit space, so an m-tile count that is not a multiple of 8 does not leave one XCD a
        # whole m-tile behind. GEMM2 of an m-tile reads its GEMM1 output from the L2 of the XCD that wrote it,
        # except in the m-tiles cut by a slice end (see _count_tile). Slice ends fall on groups of 4 n-tiles,
        # whose output scales fill whole 256-byte blocks: no cache line of GEMM1 output is written by two XCDs.
        assert N_TILES % 4 == 0
        shard_base = ((work_shard * total_work) // fx.Int32(8)) & fx.Int32(-4)
        shard_work = (((work_shard + fx.Int32(1)) * total_work) // fx.Int32(8) & fx.Int32(-4)) - shard_base

        # The m-tiles at this XCD slice's ends that the slice cuts (-1: the end is on an m-tile boundary).
        shard_end = shard_base + shard_work
        cut_lo = (shard_base % fx.Int32(N_TILES) != fx.Int32(0)).select(
            shard_base // fx.Int32(N_TILES), fx.Int32(-1))
        cut_hi = (shard_end % fx.Int32(N_TILES) != fx.Int32(0)).select(
            shard_end // fx.Int32(N_TILES), fx.Int32(-1))

        def _cut_own(m):
            return (m == cut_lo) | (m == cut_hi)

        pend_m = fx.Int32(-1)
        g1_fetch = fx.Int32(0)

        def _count_tile(m):
            # Count a finished tile of m-tile m (its stores completed before the loop-end barrier); the last of
            # the m-tile's N_TILES tiles raises its GEMM2 ready epoch. A cut m-tile runs on two XCDs: each counts
            # its part once, by its last own unit, after writing its L2 back for the other XCD's GEMM2 tiles.
            inc = fx.Int32(1)
            if _cut_own(m):
                side = (m == cut_lo).select(fx.Int32(4), fx.Int32(8))
                own_hi = (shard_end < (m + fx.Int32(1)) * fx.Int32(N_TILES)).select(
                    shard_end, (m + fx.Int32(1)) * fx.Int32(N_TILES))
                own_lo = (shard_base > m * fx.Int32(N_TILES)).select(shard_base, m * fx.Int32(N_TILES))
                old_own = fx.Int32(comm_ops.atomic_add_agent(
                    a_work_head + fx.Int64(work_shard) * fx.Int64(64) + fx.Int64(side), fx.Int32(1)))
                inc = (old_own + fx.Int32(1) == own_hi - own_lo).select(own_hi - own_lo, fx.Int32(0))
                if inc != fx.Int32(0):
                    comm_ops.fence_agent_release()
            if inc != fx.Int32(0):
                cnt_addr = a_g2_cnt + fx.Int64(m) * fx.Int64(4)
                old_cnt = fx.Int32(comm_ops.atomic_add_agent(cnt_addr, inc))
                if old_cnt + inc == fx.Int32(N_TILES):
                    comm_ops.atomic_umax_agent(a_g2_ready + fx.Int64(m) * fx.Int64(4), gate_epoch)
                    # All counts of this launch are in: rewind to 0, since the next launch's config may use another
                    # N_TILES (tile_n 256 / 128), and a leftover count would make its ready fire early or never.
                    comm_ops.atomic_add_agent(cnt_addr, fx.Int32(-N_TILES))
        while consumer_active:
            if tid == fx.Int32(0):
                local_work = first_local
                if g1_fetch != fx.Int32(0):
                    local_work = fx.Int32(
                        comm_ops.atomic_add_agent(
                            a_work_head + fx.Int64(work_shard) * fx.Int64(64), fx.Int32(1)
                        )
                    )
                # Count the previous tile of this CTA (its stores completed before the loop-end barrier)
                # next to the work fetch, so the two atomics share one round trip.
                if pend_m >= fx.Int32(0):
                    _count_tile(pend_m)
                work = (local_work < shard_work).select(shard_base + local_work, total_work)
                # The shard's GEMM2 units follow its GEMM1 units on the same counter: the fetch that ends
                # GEMM1 is this CTA's first GEMM2 unit.
                fx.ptr_store(local_work - shard_work, g2_scratch)
                fx.ptr_store(Vec.from_elements([work], fx.Int32), work_scratch)
            fx.barrier()
            work = Vec(work_scratch_view.load())[0]
            has_work = (work < total_work).select(fx.Int32(1), fx.Int32(0))
            if has_work != fx.Int32(0):
                _do_scheduled_tile(work)
                # Same-XCD handoff: store completion puts the tile in the shared L2; the next work fetch
                # counts it and the last of the m-tile's N_TILES CTAs raises its ready epoch.
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.barrier()
            pend_m = (has_work != fx.Int32(0)).select(work // fx.Int32(N_TILES), fx.Int32(-1))
            consumer_active = has_work != fx.Int32(0)
            g1_fetch = fx.Int32(1)

        if tid < fx.Int32(fz_npes):
            fx.ptr_store(g2_peer_base, lds_typed_ptr(fx.Int32(g2_lay["peer"]) + tid * fx.Int32(8), T.i64, align=8))
        # The XCD's slice of the GEMM2 unit space, at the same place in the m-tiles as its GEMM1 slice: an m-tile
        # that GEMM1 did not cut keeps its GEMM2 on the XCD whose L2 holds its GEMM1 output (an independent
        # equal split could move the first units of such an m-tile to the previous XCD when the GEMM1 slice end
        # was rounded down to an m-tile boundary).
        g2_base = (shard_base * fx.Int32(g2_n_blocks)) // fx.Int32(N_TILES)
        g2_units = (shard_end * fx.Int32(g2_n_blocks)) // fx.Int32(N_TILES) - g2_base
        g2_head = a_work_head + fx.Int64(work_shard) * fx.Int64(64)
        g2_active = fx.Int32(1) == fx.Int32(1)
        g2_fetch = fx.Int32(0)
        while g2_active:
            fx.barrier()
            if tid == fx.Int32(0):
                if g2_fetch != fx.Int32(0):
                    fx.ptr_store(fx.Int32(comm_ops.atomic_add_agent(g2_head, fx.Int32(1))) - shard_work,
                        g2_scratch)
            fx.barrier()
            g2_local = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.ptr_load(g2_scratch).ir_value()))
            if g2_local < g2_units:
                g2_u = g2_base + g2_local
                g2_m = g2_u // fx.Int32(g2_n_blocks)
                g2_n = g2_u - g2_m * fx.Int32(g2_n_blocks)
                if tid == fx.Int32(0):
                    mori_shmem.int32_wait_until_equals(a_g2_ready + fx.Int64(g2_m) * fx.Int64(4), gate_epoch)
                fx.barrier()
                # GEMM1 wrote this m-tile through this XCD's L2, or (in an m-tile shared by two XCDs) written back
                # to memory by the other XCD into lines this XCD has not cached, before it was counted: only the
                # CU's L1 can be stale.
                comm_ops.invalidate_l1()
                g2_tile(lds_base_i32, tid, g2_lane, g2_wave, g2_m, g2_n, g2_aq, g2_ascale, g2_bq, g2_bscale,
                    g2_eids, g2_stids, g2_sweights, g2_trb, g2_rows, g2_max_m_blocks, fx.Int32(inter_dim),
                    fx.Int32(model_dim))
            g2_active = g2_local < g2_units
            g2_fetch = fx.Int32(1)
        # The entry-ticket epoch is the same on every rank and needs no load; the high bit keeps it apart from
        # the combine op's own flag values in the shared xdb slots.
        fc_flag = fx.Int64(gate_epoch) + fx.Int64(1 << 40)
        fused_combine_tail(tid, g2_lane, g2_wave, _disp_ptr(DispatchSlot.FC_CNT), fc_flag,
            _disp_ptr(DispatchSlot.FC_XDB), _disp_ptr(DispatchSlot.FC_P2P_XDB), _disp_ptr(DispatchSlot.FC_INP),
            _disp_ptr(DispatchSlot.FC_OUT), _disp_ptr(DispatchSlot.FC_TRECV), i32_cur_tok,
            fx.Int32(launch_grid_x))

    @flyc.jit
    def launch(
        out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
        sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor,
        tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
        addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
        addr_expected: fx.Int64, stream: fx.Stream,
    ):
        kernel(
            out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, out_scale, tokens,
            addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc, addr_parity, addr_expected,
            value_attrs={
                "rocdl.waves_per_eu": waves_per_eu_hint,
                "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
            },
        ).launch(grid=(launch_grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

    return launch


def run_mega_moe_stage1(out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, out_scale,
    tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    addr_parity, addr_expected, stream, *, model_dim, inter_dim, rank, experts_per_rank, fuse_npes,
    fuse_topk, fuse_cap, fuse_mtpr, fuse_scale_dim, fused, num_cu,
    sort_block_m=32, tile_n=256, tile_k=256, num_waves=4, grid_mult=4, pipe_weights=True,
    mfma_amajor=False, swizzle_a=True, async_a_copy=False, num_dispatch_cu=32,
    use_tile_resource=True, waves_per_eu_hint=2,
    b_nt=-1, swiglu_limit=0.0):
    launch = compile_mega_moe_stage1(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank, experts_per_rank=experts_per_rank,
        fuse_npes=fuse_npes, fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim, fused=tuple(fused),
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        grid_mult=grid_mult, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, num_cu=num_cu, num_dispatch_cu=num_dispatch_cu,
        b_nt=b_nt, swiglu_limit=swiglu_limit,
    )
    _run_compiled(
        launch, out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, out_scale,
        tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
        addr_parity, addr_expected, stream,
    )
# fmt: on


# Combine reducer CTAs of the single kernel.
_COMBINE_REDUCERS = 256
# Token bucket from which the expert m-tile is 64 rows instead of 32.
_BM64_MIN_BUCKET = 256


class SmallMRuntime:
    """Small-input single-kernel MegaMoE (MTPR <= 1024) with one in-flight launch per instance.

    Owns its own dispatch/combine buffers and protocol state; weights are bound
    by pointer so one instance serves every layer.
    """

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int, experts: int, topk: int,
        quant: str, w1: torch.Tensor, w1_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor,
        max_tok_per_rank: int, swiglu_limit: float = 0.0):
    # fmt: on
        if quant != "a8w4":
            raise ValueError("SmallMRuntime currently supports quant='a8w4' only")
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
        if max_tok_per_rank <= 0 or max_tok_per_rank & (max_tok_per_rank - 1):
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two")
        if max_tok_per_rank > SMALL_M_FIXED_SLOT_MAX_MTPR:
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} > {SMALL_M_FIXED_SLOT_MAX_MTPR}")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = int(experts // world_size)
        self.topk = int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.swiglu_limit = float(swiglu_limit)
        if self.swiglu_limit < 0:
            raise ValueError("swiglu_limit must be non-negative")
        self.dev = torch.device("cuda", rank)
        # The combine reducers run at the end of the kernel on a fully resident grid, so they are not bounded by
        # one XCD's residency.
        self._reducers = _COMBINE_REDUCERS
        self.max_recv = self.world_size * self.mtpr
        self._s1_scale_dim = self.model_dim // 32
        # fmt: off
        self.comb_cfg = FlyDSLDispatchCombineConfig(rank=self.rank, world_size=self.world_size,
            hidden_dim=self.model_dim, max_num_inp_token_per_rank=self.mtpr, num_experts_per_rank=self.epr,
            num_experts_per_token=self.topk, combine_dtype=torch.bfloat16,
            dispatch_dtype=torch.float8_e4m3fn, scale_dim=self._s1_scale_dim, scale_type_size=1,
            enable_std_moe=False, enable_group_major=True, gm_unit_size=32,
            gm_scheme="fixedslot", gm_compact=False, max_total_recv_tokens=self.world_size)
        # fmt: on
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(self.comb_cfg)
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        self._build(w1, w1_scale)

    def _build(self, w1, w1_scale):
        self.sort_block_m = 32
        self._s1_w1 = w1.contiguous().view(torch.uint8)
        self._s1_w1_scale = w1_scale.contiguous().view(torch.uint8)
        op = self.comb_op._gm
        assert op is not None, "combine op was built without enable_group_major"
        self._s1_op = op
        # Payload capacity follows the largest SBM; metadata covers the smallest candidate.
        metadata_blocks = (op.num_valid_max + self.sort_block_m - 1) // self.sort_block_m
        if metadata_blocks > op.max_blocks:
            op.max_blocks = metadata_blocks
            op.sorted_expert_ids = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
            op.tile_row_base = torch.zeros(metadata_blocks, dtype=torch.int32, device=self.dev)
        self._s1_nvm = op.num_valid_max
        self._s1_cap = op.ll_cap
        self._s1_epoch_parity = torch.zeros(1, dtype=torch.int32, device=self.dev)
        self._s1_epoch_expected = torch.zeros(2, dtype=torch.int32, device=self.dev)
        self._s1_num_cu = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._allocate_dispatch_workspace(op, metadata_blocks)
        self._s1_mega = run_mega_moe_stage1

        v = op._ll_views()
        self._s1_rx = v["rx_em"]
        self._s1_scale_i32 = v["scale_em_i32"]

        inter_dim = self.inter_dim
        a2rows = self._s1_nvm
        self._s1_out = torch.zeros((a2rows, inter_dim), dtype=torch.float8_e4m3fn, device=self.dev)
        prows = ((a2rows + 255) // 256) * 256
        pcols = (((inter_dim // 32) + 7) // 8) * 8
        self._s1_osd = torch.zeros(prows * pcols + inter_dim, dtype=torch.uint8, device=self.dev)
        self._s1_disp_by_w2 = {}
        self._build_disp_table()

    def _allocate_dispatch_workspace(self, op, metadata_blocks):
        def zeros(n, dtype=torch.int32):
            return torch.zeros(n, dtype=dtype, device=self.dev)

        workspace = {
            "entry_count": zeros(11, torch.int64),
            "epoch_gate": zeros(11),
            "work_head": zeros(8 * 16),
            "group_done": zeros(1),
            "tile_rows": zeros(metadata_blocks),
            "g2_cnt": zeros(metadata_blocks),
            "g2_ready": zeros(metadata_blocks),
            "plan_nv": zeros(1, torch.int64),
            "xcd_acq": zeros(8 * 16, torch.int64),  # one 128-byte line per XCD
        }
        workspace["count_done"] = op._sym((2 * self.world_size,), torch.int32)
        ms.shmem_barrier_all()
        workspace["p2p_count_done"] = op._p2p_table(workspace["count_done"])
        self._s1_dispatch_workspace = workspace

    def _build_disp_table(self):
        op = self._s1_op
        comb_op = self.comb_op
        workspace = self._s1_dispatch_workspace
        table = [0] * DISPATCH_TABLE_SIZE
        for slot in (
            "entry_count", "epoch_gate", "work_head", "group_done", "tile_rows", "count_done", "p2p_count_done",
            "g2_cnt", "g2_ready", "plan_nv", "xcd_acq",
        ):
            table[DispatchSlot[slot.upper()]] = workspace[slot].data_ptr()
        table[DispatchSlot.P2P_TOKEN] = op.p2p_rx_em.data_ptr()
        table[DispatchSlot.P2P_SCALE] = op.p2p_scale_em.data_ptr()
        table[DispatchSlot.P2P_WEIGHT] = op.p2p_wts_em.data_ptr()
        table[DispatchSlot.P2P_SRCMAP] = op.p2p_srcmap_em.data_ptr()
        table[DispatchSlot.SORTED_EXPERT] = op.sorted_expert_ids.data_ptr()
        table[DispatchSlot.TILE_ROW_BASE] = op.tile_row_base.data_ptr()
        table[DispatchSlot.NUM_VALID] = op.num_valid.data_ptr()
        table[DispatchSlot.RUNNING] = op.running.data_ptr()
        table[DispatchSlot.P2P_RUNNING] = op.p2p_running.data_ptr()
        table[DispatchSlot.G2_AQ] = self._s1_out.data_ptr()
        table[DispatchSlot.G2_ASCALE] = self._s1_osd.data_ptr()
        table[DispatchSlot.G2_BQ] = self.w2.data_ptr()
        table[DispatchSlot.G2_BSCALE] = self.w2_scale.data_ptr()
        table[DispatchSlot.G2_EIDS] = op.sorted_expert_ids.data_ptr()
        table[DispatchSlot.G2_STIDS] = op.srcmap_em.data_ptr()
        table[DispatchSlot.G2_SWEIGHTS] = op.wts_em.data_ptr()
        table[DispatchSlot.G2_TRB] = op.tile_row_base.data_ptr()
        table[DispatchSlot.G2_P2P_COMB_INP] = comb_op._p2p_comb_inp.data_ptr()
        table[DispatchSlot.G2_MAX_M_BLOCKS] = (self._s1_nvm + self.sort_block_m - 1) // self.sort_block_m
        table[DispatchSlot.FC_CNT] = comb_op.comb_bar.data_ptr()
        table[DispatchSlot.FC_XDB] = comb_op.shmem_xdev_bar_mem.data_ptr()
        table[DispatchSlot.FC_P2P_XDB] = comb_op._p2p_xdb_mem.data_ptr()
        table[DispatchSlot.FC_INP] = comb_op.shmem_comb_inp_tok.data_ptr()
        table[DispatchSlot.FC_OUT] = comb_op.shmem_comb_out_tok.data_ptr()
        table[DispatchSlot.FC_TRECV] = comb_op.total_recv.data_ptr()
        self._s1_disp_template = table
        self._s1_disp = torch.tensor(table, dtype=torch.int64, device=self.dev)
        self._s1_disp_by_w2[(self.w2.data_ptr(), self.w2_scale.data_ptr())] = self._s1_disp

    def bind_weights(self, w1, w1_scale, w2, w2_scale) -> None:
        """Bind layer weights.

        W1 pointers are launch arguments; W2 pointers live in the dispatch table,
        so each distinct W2 gets its own table, built on first (eager) use and
        reused afterwards, including inside captured graphs.
        """
        self._s1_w1 = w1.contiguous().view(torch.uint8)
        self._s1_w1_scale = w1_scale.contiguous().view(torch.uint8)
        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()
        key = (self.w2.data_ptr(), self.w2_scale.data_ptr())
        disp = self._s1_disp_by_w2.get(key)
        if disp is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("fused dispatch table for these W2 weights was not built before graph capture")
            table = list(self._s1_disp_template)
            table[DispatchSlot.G2_BQ] = key[0]
            table[DispatchSlot.G2_BSCALE] = key[1]
            disp = torch.tensor(table, dtype=torch.int64, device=self.dev)
            self._s1_disp_by_w2[key] = disp
        self._s1_disp = disp

    def _select_config(self, tokens: int) -> MegaMoEConfig:
        config = select_mega_moe_config(
            tokens,
            self.mtpr,
            experts_per_rank=self.epr,
            model_dim=self.model_dim,
            inter_dim=self.inter_dim,
        )
        if config.stage2.persist:
            # Buckets >= 128 select a persistent GEMM2; the kernel runs GEMM2 in its own grid with NT B loads.
            config = replace(config, stage2=replace(config.stage2, block_n=256, persist=False, persist_cu=0, use_nt=True))
        config = self._bm64_override(tokens, config)
        self._active_config = config
        return config

    def _bm64_override(self, tokens: int, config: MegaMoEConfig) -> MegaMoEConfig:
        """64-row expert tiles from token bucket _BM64_MIN_BUCKET up: at ~35+ rows per expert a
        32-row tile splits every expert in two and streams its weights twice."""
        if nearest_token_bucket(max(tokens, 1)) < _BM64_MIN_BUCKET:
            return config
        stage1 = replace(config.stage1, sort_block_m=64)
        return replace(config, stage1=stage1, stage2=replace(config.stage2, block_m=64))

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = int(x_bf16.shape[0])
        if run_tokens > self.mtpr:
            raise ValueError(f"run_tokens={run_tokens} > max_tok_per_rank={self.mtpr}")
        if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if tuple(x_bf16.shape) != (run_tokens, self.model_dim):
            raise ValueError(f"x_bf16 must have shape ({run_tokens}, {self.model_dim})")
        if wts.dtype != torch.float32 or not wts.is_contiguous():
            raise ValueError("wts must be contiguous float32")
        if tuple(wts.shape) != (run_tokens, self.topk):
            raise ValueError(f"wts must have shape ({run_tokens}, {self.topk})")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if tuple(topk_ids.shape) != (run_tokens, self.topk):
            raise ValueError(f"topk_ids must have shape ({run_tokens}, {self.topk})")
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        elif hasattr(stream, "cuda_stream"):
            stream = fx.Stream(stream.cuda_stream)
        # Ranks may use different token buckets in one call; an idle rank uses the 1-token bucket.
        config = self._select_config(max(run_tokens, 1))
        stage1, stage2 = config.stage1, config.stage2
        assert config.p2p_quant == "none" and stage2.block_m == stage1.sort_block_m
        fused = (stage2.block_n, stage2.block_k, int(self.max_recv), self._reducers)
        op = self._s1_op
        # fmt: off
        self._s1_mega(
            self._s1_out, self._s1_rx, self._s1_w1, self._s1_scale_i32, self._s1_w1_scale,
            op.tile_row_base, op.sorted_expert_ids, self._s1_osd, fx.Int32(self._s1_nvm),
            fx.Int64(self._s1_disp.data_ptr()), fx.Int32(run_tokens), fx.Int64(x_bf16.data_ptr()),
            fx.Int64(topk_ids.data_ptr()), fx.Int64(wts.data_ptr()), fx.Int64(x_bf16.data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()), fx.Int64(self._s1_epoch_expected.data_ptr()),
            stream, model_dim=self.model_dim, inter_dim=self.inter_dim, rank=self.rank,
            experts_per_rank=self.epr, fuse_npes=self.world_size, fuse_topk=self.topk,
            fuse_cap=self._s1_cap, fuse_mtpr=self.mtpr, fuse_scale_dim=self._s1_scale_dim,
            fused=fused, num_cu=self._s1_num_cu,
            sort_block_m=stage1.sort_block_m, tile_n=stage1.tile_n, tile_k=stage1.tile_k,
            num_waves=stage1.num_waves, grid_mult=min(stage1.grid_mult, 2),
            pipe_weights=stage1.pipe_weights,
            mfma_amajor=stage1.mfma_amajor, swizzle_a=stage1.swizzle_a,
            async_a_copy=stage1.async_a_copy, num_dispatch_cu=stage1.num_dispatch_cu,
            use_tile_resource=stage1.use_tile_resource,
            waves_per_eu_hint=stage1.waves_per_eu_hint, b_nt=stage1.b_nt, swiglu_limit=self.swiglu_limit)
        # fmt: on
        cfg = self.comb_cfg
        out_tok = (
            self.comb_op.shmem_comb_out_tok.view(torch.int8)[: self.mtpr * cfg.combine_token_bytes]
            .view(cfg.combine_dtype)
            .view(self.mtpr, cfg.combine_token_view_dim)
        )
        return out_tok[:run_tokens] if slice_output else out_tok

    __call__ = forward

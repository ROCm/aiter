# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: B023, SIM102
"""MegaMoEV2 small-input path: quant, dispatch, GEMM1, GEMM2 and combine fused into one kernel.

Dispatch writes each routed token straight into its expert's fixed slot on the destination rank. The
runtime serves up to 512 tokens per rank and owns its protocol state and combine output; inside
MegaMoEV2 its payload, GEMM1-output and combine-input storage are views of the instance's shared arenas.
"""

import functools
from enum import IntEnum

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_gfx_runtime

from .. import communication_ops_utils as comm_ops
from ..flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
    GroupMajorPayloadArena,
    group_major_capacity,
)
from ..kernels_common import ceildiv
from ..mxfp4_gemm_common import kStages, lds_typed_ptr
from ..tensor_shim import _preload_compiled, _run_compiled, ptr_buf_tensor
from .dispatch import _wave_inclusive_scan_i32, _wave_reduce_max_i32
from .gemm1 import _LdsF32View, build_fused_gemm1
from .gemm2 import gemm2_compute_v2, issue_a_load_lds_dt
from .mega_moe_config import (
    FUSED_TOKEN_BUCKETS,
    SMALL_M_MAX_MTPR,
    MegaMoEConfig,
    fused_layout_supported,
    nearest_token_bucket,
    select_fused_config,
    small_m_cu_supported,
)
from .mega_moe_fused_gemm import (
    AddrBuffer,
    _buffer_load,
    _buffer_store,
    _make_buffer_from_addr,
    load_at,
    store_at,
)
from .mega_moe_stage1 import _validate_dispatch_capacity, stage1_output_buffers
from .mega_moe_stage2 import _stage2_lds_bytes, p2p_scatter_epilog
from .weights import (
    check_bindable,
    check_forward_inputs,
    contiguous_weights,
    weight_spec,
)

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
    PLAN_NV = 34  # i64 (gate_epoch << 32 | row-count bits | num_valid): plan flag that carries num_valid
    XCD_ACQ = (
        35  # per-XCD i64 copy of PLAN_NV, republished after that XCD's L2 invalidation
    )
    # dedup_rows: byte offsets of the token-row regions (one row per source rank and token) from the payload rows
    # and from their scales; the same on every rank (symmetric heap).
    TOKEN_ROWS = 36
    TOKEN_SCALES = 37
    # Byte offset (the same on every rank) from FC_INP to the per-token GEMM2 completion counters: two
    # parity halves of mtpr i32 (tok_done).
    TOK_DONE_OFF = 38


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


def _uniform_i64(addr):
    v = fx.Uint64(addr)
    lo = rocdl.readfirstlane(T.i32, fx.Uint32(v))
    hi = rocdl.readfirstlane(T.i32, fx.Uint32(v >> 32))
    return fx.Int64((fx.Uint64(hi) << 32) | fx.Uint64(lo))


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
    dedup=False,
    uniform=False,
):
    """Load everything the (single) route of this wave needs that does not depend on the epoch gate.

    addr_in_tok holds the bf16 rows; the wave quantizes its row to MXFP8 (1x32 E8M0 scales, bit-identical to
    quant.per_1x32_mx_quant). addr_in_sc is unused.

    Straight-line (no device branches) so it can run before the gate wait: the route's expert, the
    destination's remote pointers, the token row, its scales and its weight. Indices are clamped so waves
    without a route read valid memory; emit_direct_fixed_slot_payload ignores their values.
    ``uniform``: the table and remote pointers move to SGPRs, so their buffer stores need no waterfall loop.
    """
    rdisp = AddrBuffer(addr_disp)
    as_ptr = _uniform_i64 if const_expr(uniform) else (lambda v: v)

    def dp(i):
        return as_ptr(load_at(rdisp, fx.Int32(int(i)), fx.Int64, 1))

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
        name: as_ptr(load_at(AddrBuffer(tables[name]), destination, fx.Int64, 1))
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
    # dedup: only the token's first route to a destination sends the row, into the token-row region.
    extra = {}
    if const_expr(dedup):
        topk_slot = wk - source_token * fx.Int32(fz_k)
        duplicate = topk_slot < fx.Int32(0)
        for j in range_constexpr(fz_k - 1):
            ej = load_at(AddrBuffer(addr_in_idx), source_token * fx.Int32(fz_k) + fx.Int32(j), fx.Int32, 1)
            same = (ej >= fx.Int32(0)) & (ej < fx.Int32(fz_total_experts)) & (ej // fx.Int32(fz_epr) == destination)
            duplicate = duplicate | ((fx.Int32(j) < topk_slot) & same)
        extra = {"duplicate": duplicate, "token_rows": dp(DispatchSlot.TOKEN_ROWS),
                 "token_scales": dp(DispatchSlot.TOKEN_SCALES)}
    return {
        **extra,
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
    dedup=False, direct_done=0, running_half=None,
):
# fmt: on
    """Allocate and publish routes directly into destination fixed slots.

    ``running_half``: the destinations' expert counters are two halves of fz_epr; the routes count into this one,
    and their srcmap and weight go to the same half of the destinations' two halves of fz_epr * fz_cap slots.
    """
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
            running_slot = local_expert
            if const_expr(running_half is not None):
                running_slot = running_slot + running_half * fx.Int32(fz_epr)
            offset_lane = fx.Int32(0)
            if lane == fx.Int32(0):
                if valid_expert:
                    offset_lane = fx.Int32(
                        comm_ops.atomic_add_system(
                            rpf["remote"]["running"] + fx.Int64(running_slot) * fx.Int64(4), fx.Int32(1)
                        )
                    )
            expert_offset = fx.Int32(fx.rocdl.readlane(T.i32, offset_lane, 0))
            publish = valid_expert & (expert_offset < fx.Int32(fz_cap))
            payload_row = local_expert * fx.Int32(fz_cap) + expert_offset
            meta_row = payload_row
            if const_expr(running_half is not None):
                meta_row = meta_row + running_half * fx.Int32(fz_epr * fz_cap)

            if const_expr(dedup):
                if publish & ~rpf["duplicate"]:
                    token_row = fx.Int64(fz_rank * fz_mtpr) + fx.Int64(source_token)
                    destination_rsrc = AddrBuffer(
                        rpf["remote"]["rx"] + rpf["token_rows"] + token_row * fx.Int64(fz_nbytes))
                    for i in range_constexpr(len(row_values)):
                        store_at(row_values[i], destination_rsrc, lane * fx.Int32(4) + fx.Int32(i * 256))
                    if const_expr(fz_enable_scales):
                        if lane < fx.Int32(fz_scale_n_i32):
                            store_at(rpf["scale"], AddrBuffer(rpf["remote"]["sc"] + rpf["token_scales"]
                                + token_row * fx.Int64(fz_scale_n_i32 * 4)), lane)
            if publish:
                if const_expr(not dedup):
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
                    store_at(weight_bits, AddrBuffer(rpf["remote"]["wts"]), meta_row)
                    store_at(source_encoding, AddrBuffer(rpf["remote"]["sm"]), meta_row)

    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()
    # direct_done: producers 0..active-1 carry every route; when there are few of them, each adds its share of
    # the source flag (active - 1 ones and a remainder, all positive, summing to _DIRECT_DONE_INC) to every
    # destination right after its drain.
    publish_last = warp == fx.Int32(0)
    if const_expr(direct_done):
        active = (route_limit + fx.Int32(num_waves - 1)) // fx.Int32(num_waves)
        active = (active < fx.Int32(dispatch_blocks)).select(active, fx.Int32(dispatch_blocks))
        active = (active > fx.Int32(1)).select(active, fx.Int32(1))
        direct = active <= fx.Int32(direct_done)
        if direct:
            if warp == fx.Int32(0):
                if producer_slot < active:
                    if lane < fx.Int32(fz_npes):
                        comm_ops.fence_system_release()
                        share = (producer_slot == fx.Int32(0)).select(
                            fx.Int32(_DIRECT_DONE_INC + 1) - active, fx.Int32(1))
                        remote_done = load_at(AddrBuffer(p_source_done), lane, fx.Int64, 1)
                        done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
                        comm_ops.atomic_add_system(remote_done + fx.Int64(done_index) * fx.Int64(4), share)
        publish_last = publish_last & ~direct
    # The last producer publishes; one lane per destination, so the remote release stores overlap.
    if publish_last:
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


# Per-launch step of the expected source-flag value on a runtime whose buckets use direct_done (npes otherwise).
# direct_done reaches it with up to this many positive adds, so the flag equals the expected value only once all
# of them landed. Every bucket of such a runtime uses direct_done, as the ranks of a launch may run different
# buckets.
_DIRECT_DONE_INC = 64


def _plan_counts(a_running, lane, fz_epr, fz_cap, fz_tile_m, count_cm=0):
    """Per-expert counts (one expert per lane and 64-expert chunk) and their tile scans; the scans carry across
    chunks."""
    num_chunks = (fz_epr + 63) // 64
    chunks = []
    overflow_count = None
    for chunk in range_constexpr(num_chunks):
        last_lane = min(63, fz_epr - chunk * 64 - 1)
        expert = lane + fx.Int32(chunk * 64) if const_expr(chunk > 0) else lane
        valid_expert = lane < fx.Int32(last_lane + 1)
        safe_expert = valid_expert.select(expert, fx.Int32(0))
        count = load_at(AddrBuffer(a_running), safe_expert, fx.Int32, 1, cache_modifier=count_cm)
        count = valid_expert.select(count, fx.Int32(0))
        overflow_flag = (count > fx.Int32(fz_cap)).select(fx.Int32(1), fx.Int32(0))
        overflow_prefix = _wave_inclusive_scan_i32(overflow_flag, lane)
        chunk_overflow = fx.Int32(fx.rocdl.readlane(T.i32, overflow_prefix, last_lane))
        overflow_count = chunk_overflow if overflow_count is None else overflow_count + chunk_overflow
        if const_expr(chunk == num_chunks - 1):
            no_overflow = overflow_count == fx.Int32(0)
        safe_count = (count <= fx.Int32(fz_cap)).select(count, fx.Int32(0))
        num_expert_tiles = (safe_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
        inclusive_tiles = _wave_inclusive_scan_i32(num_expert_tiles, lane)
        metadata_base = inclusive_tiles - num_expert_tiles
        chunk_tiles = fx.Int32(fx.rocdl.readlane(T.i32, inclusive_tiles, last_lane))
        chunk_rows = fx.Int32(fx.rocdl.readlane(T.i32, _wave_inclusive_scan_i32(safe_count, lane), last_lane))
        chunk_max = _wave_reduce_max_i32(safe_count, lane)
        if const_expr(chunk == 0):
            total_tiles, routed_rows, max_count = chunk_tiles, chunk_rows, chunk_max
        else:
            metadata_base = metadata_base + total_tiles
            total_tiles = total_tiles + chunk_tiles
            routed_rows = routed_rows + chunk_rows
            max_count = (chunk_max > max_count).select(chunk_max, max_count)
        chunks.append((valid_expert, safe_expert, safe_count, num_expert_tiles, metadata_base))
    return chunks, total_tiles, routed_rows, max_count, overflow_count, no_overflow


def _plan_flag(no_overflow, total_tiles, hot_expert, routed_rows, fz_tile_m):
    """Low 32 bits of the plan word: num_valid, plus the row-count bits of _PLAN_CACHED_W_BIT /
    _PLAN_FP8_ROWS_BIT."""
    nv_flag = no_overflow.select(total_tiles * fx.Int32(fz_tile_m), fx.Int32(0))
    nv_flag = nv_flag | (hot_expert | (routed_rows >= fx.Int32(_CACHED_W_MIN_ROWS))).select(
        fx.Int32(1 << _PLAN_CACHED_W_BIT), fx.Int32(0)
    )
    return nv_flag | (hot_expert | (routed_rows >= fx.Int32(_FP8_ROWS_MIN_ROWS))).select(
        fx.Int32(1 << _PLAN_FP8_ROWS_BIT), fx.Int32(0)
    )


@flyc.jit
def emit_direct_fixed_slot_finalize(
    *, fz_npes, fz_epr, fz_cap, fz_mtpr, fz_rank, fz_tile_m, addr_disp, parity, expected, plan_nv,
    plan_wt=False, keep_running=False, uniform=False, running_half=None,
):
    """Finalize local fixed slots as soon as every source publishes this destination.

    ``plan_nv = (addr, epoch)``: the local plan is published as one i64 ``epoch << 32 | num_valid`` store, so
    consumers get num_valid from the flag itself. Only this rank's consumers read it. The srcmap of padding rows
    is left unwritten: GEMM2 masks rows past the tile's valid count itself.
    ``running_half``: the expert counters are two halves of fz_epr; this launch's routes counted into this one.
    """
    assert 0 < fz_epr <= 128, "direct fixed-slot finalize requires 1..128 experts per rank"
    rdisp = AddrBuffer(addr_disp)

    def dp(i):
        v = load_at(rdisp, fx.Int32(int(i)), fx.Int64, 1)
        if const_expr(uniform):
            v = _uniform_i64(v)
        return v

    a_se = dp(DispatchSlot.SORTED_EXPERT)
    a_trb = dp(DispatchSlot.TILE_ROW_BASE)
    a_nv = dp(DispatchSlot.NUM_VALID)
    a_running = dp(DispatchSlot.RUNNING)
    if const_expr(running_half is not None):
        a_running = a_running + fx.Int64(running_half) * fx.Int64(fz_epr * 4)
    a_source_done = dp(DispatchSlot.COUNT_DONE)
    a_tile_rows = dp(DispatchSlot.TILE_ROWS)
    wt_cm = _SC0_SC1_CACHE if plan_wt else 0

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    if warp == fx.Int32(0):
        for source in range(lane, fz_npes, 64):
            done_index = parity * fx.Int32(fz_npes) + source
            comm_ops.wait_i32_until_equals(a_source_done + fx.Int64(done_index) * fx.Int64(4), expected)
        comm_ops.fence_system_acquire()

        chunks, total_tiles, routed_rows, max_count, overflow_count, no_overflow = _plan_counts(
            a_running, lane, fz_epr, fz_cap, fz_tile_m)
        hot_expert = max_count >= fx.Int32(_HOT_EXPERT_MIN_ROWS)

        for chunk in range_constexpr(len(chunks)):
            valid_expert, safe_expert, safe_count, num_expert_tiles, metadata_base = chunks[chunk]
            if valid_expert:
                if no_overflow:
                    global_expert = fx.Int32(fz_rank * fz_epr) + safe_expert
                    payload_base = safe_expert * fx.Int32(fz_cap)
                    for tile in range(fx.Int32(0), num_expert_tiles, 1):
                        metadata_index = metadata_base + tile
                        store_at(global_expert, AddrBuffer(a_se), metadata_index, cache_modifier=wt_cm)
                        store_at(payload_base + tile * fx.Int32(fz_tile_m), AddrBuffer(a_trb), metadata_index,
                            cache_modifier=wt_cm)
                        rows = safe_count - tile * fx.Int32(fz_tile_m)
                        rows = (rows < fx.Int32(fz_tile_m)).select(rows, fx.Int32(fz_tile_m))
                        store_at(rows, AddrBuffer(a_tile_rows), metadata_index, cache_modifier=wt_cm)
                if const_expr(not keep_running):
                    store_at(fx.Int32(0), AddrBuffer(a_running), safe_expert, cache_modifier=wt_cm)

        if lane == fx.Int32(0):
            num_valid = no_overflow.select(total_tiles * fx.Int32(fz_tile_m), fx.Int32(0))
            store_at(num_valid, AddrBuffer(a_nv), fx.Int32(0), cache_modifier=wt_cm)
            # num_valid[1] is a device-visible overflow status.
            store_at(overflow_count, AddrBuffer(a_nv), fx.Int32(1), cache_modifier=wt_cm)

        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        if const_expr(not plan_wt):
            comm_ops.fence_system_release()
        if lane == fx.Int32(0):
            nv_flag = _plan_flag(no_overflow, total_tiles, hot_expert, routed_rows, fz_tile_m)
            comm_ops.store_i64_global_system(
                plan_nv[0], (fx.Int64(plan_nv[1]) << fx.Int64(32)) | fx.Int64(nv_flag))
    fx.barrier()


def make_fused_combine_tail(*, lds_fc_off, reducers, npes, rank, topk, total_experts, experts_per_rank, N_OUT,
    flag_is_value=False, l1_acquire=False, row_format=False, reset_running=0, reset_tok=0, tok_done=False,
    n_blocks=1, bf16_head_tokens=0):
    """Combine after this WG's P2P scatter: grid ticket, cross-device flag, reduce by the last WGs.

    ``flag_is_value``: ``arg_flag`` is this launch's flag itself (identical on every rank) instead of the address
    of a counter the last WG advances.
    ``l1_acquire``: the reducers acquire the peers' rows by dropping only the CU's L1. Valid when the combine input
    lives in the uncached shmem heap (never held in L2) and the local rows were released by the last WG.
    ``reset_running``: the last WG zeroes the first reset_running expert counts at ``arg_running`` before it
    publishes (every CTA read them before its ticket; the peers' next producers add to them only after the flag,
    or, when tok_done reducers skip it, to the other half of the runtime's double-buffered counters).
    ``row_format``: each rank sent its rows as BF16 or FP8 (1x32 E8M0 scales after the values, BF16 row stride),
    as ``i32_fp8_rows`` says; the choice travels as the low bit of the published flag.
    ``reset_tok``: the last WG zeroes that many per-token counters at ``arg_inp + arg_tok_reset`` (the previous
    launch's tok_done half: every add to it landed before that launch's reduce).
    ``tok_done``: a reducer takes a token once its counter at ``arg_inp + arg_tok_off`` reaches its valid routes x
    ``n_blocks`` (every sender then counted BF16 rows), or once every peer's flag is in.
    ``bf16_head_tokens`` (row_format): every sender sent the rows of the tokens below it as BF16 (see
    p2p_scatter_epilog); their reducers leave the rows poisoned (_ROW_POISON words).

    Matches ep_combine_intranode(skip_stage1=True, bf16): out[t] = bf16(sum_k f32(inp[t * topk + k])), where an
    invalid (-1) slot ``k`` of ``idx`` contributes zero.
    """
    row_i32 = N_OUT * 2 // 4
    chunks = row_i32 // 256
    assert row_i32 % 256 == 0
    assert flag_is_value or not row_format
    assert not bf16_head_tokens or row_format

    # fmt: off
    @flyc.jit
    def fused_combine_tail(tx_i32: fx.Int32, lane: fx.Int32, wave: fx.Int32, arg_cnt: fx.Int64, arg_flag: fx.Int64,
        arg_xdb: fx.Int64, arg_p2p_xdb: fx.Int64, arg_inp: fx.Int64, arg_out: fx.Int64, arg_trecv: fx.Int64,
        arg_idx: fx.Int64, i32_cur_tok: fx.Int32, i32_grid: fx.Int32, i32_fp8_rows: fx.Int32, arg_running: fx.Int64,
        arg_tok_off: fx.Int64, arg_tok_reset: fx.Int64):
    # fmt: on
        ticket_ptr = lds_typed_ptr(fx.Int32(lds_fc_off), T.i32, align=4)
        flag_ptr = lds_typed_ptr(fx.Int32(lds_fc_off + 8), T.i64, align=8)

        def fmt_ptr(peer):
            # One word per peer: its format bit, already shifted to the peer's position.
            return lds_typed_ptr(fx.Int32(lds_fc_off + 16) + peer * fx.Int32(4), T.i32, align=4)

        last = i32_grid - fx.Int32(1)
        # Every wave waits for its own P2P stores before the WG takes its ticket.
        rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()
        if tx_i32 == fx.Int32(0):
            if const_expr(flag_is_value):
                flag_lane = arg_flag
                if const_expr(row_format):
                    flag_lane = (arg_flag << fx.Int64(1)) | fx.Int64(i32_fp8_rows)
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
            if const_expr(reset_running > 0):
                if tx_i32 < fx.Int32(reset_running):
                    store_at(fx.Int32(0), AddrBuffer(arg_running), tx_i32, cache_modifier=_SC0_SC1_CACHE)
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.barrier()
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
            if const_expr(reset_tok > 0):
                for i in range_constexpr((reset_tok + 255) // 256):
                    if tx_i32 + fx.Int32(i * 256) < fx.Int32(reset_tok):
                        store_at(fx.Int32(0), AddrBuffer(arg_inp + arg_tok_reset), tx_i32 + fx.Int32(i * 256),
                            cache_modifier=_SC0_SC1_CACHE)
        nred = (i32_grid < fx.Int32(reducers)).select(i32_grid, fx.Int32(reducers))
        first = i32_grid - nred
        if fx.Int32(ticket) >= first:
            # tok_done: each reducer wave waits for its own token below instead.
            if tx_i32 < fx.Int32(0 if tok_done else npes):
                if const_expr(row_format):
                    # This launch's flag with either format bit (flag << 1 | fp8).
                    seen = fx.Int64(comm_ops.spin_until_shifted_eq_i64(
                        arg_xdb + fx.Int64(tx_i32) * fx.Int64(8), arg_flag, shift=1))
                    fx.ptr_store(fx.Int32(seen & fx.Int64(1)) << tx_i32, fmt_ptr(tx_i32))
                else:
                    comm_ops.wait_i64_until_equals(arg_xdb + fx.Int64(tx_i32) * fx.Int64(8), flag)
                if const_expr(l1_acquire):
                    comm_ops.invalidate_l1()
                else:
                    comm_ops.fence_system_acquire()
            fx.barrier()
            fp8_peers = fx.Int32(0)
            if const_expr(row_format and not tok_done):
                for peer in range_constexpr(npes):
                    fp8_peers = fp8_peers | fx.Int32(fx.ptr_load(fmt_ptr(fx.Int32(peer)), result_type=T.i32))
            inp_rsrc = AddrBuffer(arg_inp)
            idx_rsrc = AddrBuffer(arg_idx)
            out_rsrc = AddrBuffer(arg_out)
            items = i32_cur_tok * fx.Int32(chunks)
            worker = (fx.Int32(ticket) - first) * fx.Int32(4) + wave
            for item in range(worker, items, nred * fx.Int32(4)):
                tok = item // fx.Int32(chunks)
                col = (item - tok * fx.Int32(chunks)) * fx.Int32(256) + lane * fx.Int32(4)
                src = tok * fx.Int32(topk * row_i32) + col
                # The id and row loads issue together; the row of an invalid (-1) slot is never written this
                # launch, so it is zeroed after the load instead of gating the load on the id.
                eids = [load_at(idx_rsrc, tok * fx.Int32(topk) + fx.Int32(k), fx.Int32, 1) for k in range_constexpr(topk)]
                if const_expr(tok_done):
                    nvalid = fx.Int32(0)
                    for k in range_constexpr(topk):
                        nvalid = nvalid + ((eids[k] >= fx.Int32(0)) & (eids[k] < fx.Int32(total_experts))).select(
                            fx.Int32(1), fx.Int32(0))
                    target = nvalid * fx.Int32(n_blocks)
                    cnt_addr = arg_inp + arg_tok_off + fx.Int64(tok) * fx.Int64(4)

                    def flag_bits():
                        # Bit npes: every peer's flag is in; bits 0..npes-1: their format bits.
                        bits = fx.Int32(1 << npes)
                        for peer in range_constexpr(npes):
                            seen = fx.Int64(comm_ops.load_i64_acquire(arg_xdb + fx.Int64(peer * 8)))
                            bits = bits & ((seen >> fx.Int64(1)) == arg_flag).select(fx.Int32(-1), fx.Int32(0))
                            bits = bits | (fx.Int32(seen & fx.Int64(1)) << fx.Int32(peer))
                        return bits

                    cnt = fx.Int32(comm_ops.load_i32_acquire(cnt_addr))
                    fbits = flag_bits()
                    while (cnt != target) & ((fbits >> fx.Int32(npes)) == fx.Int32(0)):
                        cnt = fx.Int32(comm_ops.load_i32_acquire(cnt_addr))
                        fbits = flag_bits()
                    fp8_peers = (cnt == target).select(fx.Int32(0), fbits & fx.Int32((1 << npes) - 1))
                    comm_ops.invalidate_l1()
                fp8_tok = fp8_peers
                if const_expr(bf16_head_tokens):
                    # The first tokens' rows are BF16 from every sender.
                    fp8_tok = (tok < fx.Int32(bf16_head_tokens)).select(fx.Int32(0), fp8_peers)
                if fp8_tok == fx.Int32(0):
                    vals = [load_at(inp_rsrc, src + fx.Int32(k * row_i32), fx.Int32, 4, 2) for k in range_constexpr(topk)]
                    acc = Vec.filled(8, 0.0, fx.Float32)
                    for k in range_constexpr(topk):
                        valid = (eids[k] >= fx.Int32(0)) & (eids[k] < fx.Int32(total_experts))
                        words = fx.Vector(vals[k])
                        words = fx.Vector.from_elements(
                            [valid.select(words[w], fx.Int32(0)) for w in range_constexpr(4)], fx.Int32
                        )
                        acc = acc + words.bitcast(fx.BFloat16).to(fx.Float32)
                    store_at(acc.to(fx.BFloat16).bitcast(fx.Int32).ir_value(), out_rsrc, tok * fx.Int32(row_i32) + col, 2)
                    if const_expr(bf16_head_tokens):
                        if tok < fx.Int32(bf16_head_tokens):
                            for k in range_constexpr(topk):
                                store_at(Vec.filled(4, _ROW_POISON, fx.Int32).ir_value(), inp_rsrc,
                                    src + fx.Int32(k * row_i32))
                else:
                    # Some peer sent FP8 rows: the lane's 8 values are 2 i32 of its FP8 row (byte col * 2) plus the
                    # scale byte of their 32-value block.
                    acc = Vec.filled(8, 0.0, fx.Float32)
                    sc_byte = col // fx.Int32(16)
                    for k in range_constexpr(topk):
                        valid = (eids[k] >= fx.Int32(0)) & (eids[k] < fx.Int32(total_experts))
                        peer = valid.select(eids[k], fx.Int32(0)) // fx.Int32(experts_per_rank)
                        row_k = src - col + fx.Int32(k * row_i32)
                        contrib = Vec.filled(8, 0.0, fx.Float32)
                        if valid:
                            if ((fp8_tok >> peer) & fx.Int32(1)) != fx.Int32(0):
                                words = fx.Vector(load_at(inp_rsrc, row_k + col // fx.Int32(2), fx.Int32, 2, 2))
                                sc_word = load_at(inp_rsrc, row_k + fx.Int32(N_OUT // 4) + sc_byte // fx.Int32(4),
                                    fx.Int32, 1, 2)
                                e8m0 = (sc_word >> ((sc_byte & fx.Int32(3)) * fx.Int32(8))) & fx.Int32(0xFF)
                                scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
                                quads = []
                                for w in range_constexpr(2):
                                    lo = cvt_pk_f32_fp8(res=T.vec(2, T.f32), src=words[w], word_sel=False)
                                    hi = cvt_pk_f32_fp8(res=T.vec(2, T.f32), src=words[w], word_sel=True)
                                    quads.append(lo.shuffle(hi, [0, 1, 2, 3]) * scale)
                                contrib = quads[0].shuffle(quads[1], [0, 1, 2, 3, 4, 5, 6, 7])
                            else:
                                words = fx.Vector(load_at(inp_rsrc, row_k + col, fx.Int32, 4, 2))
                                contrib = words.bitcast(fx.BFloat16).to(fx.Float32)
                        acc = acc + contrib
                    store_at(acc.to(fx.BFloat16).bitcast(fx.Int32).ir_value(), out_rsrc, tok * fx.Int32(row_i32) + col, 2)

    return fused_combine_tail


def g2_lds_layout(*, BM, BN, BK, a_dtype, npes, g2_bf16_lds=False):
    """Absolute LDS offsets of the stage2 tile (compute slab, packed srcmap, weights, peer table, fc scratch)."""
    compute = _stage2_lds_bytes(BM, BN, BK, a_dtype, kStages + 1, g2_bf16_lds)
    packed = compute
    weight = packed + BM * 4
    peer = weight + BM * 4
    fc = peer + npes * 8
    # fc: ticket (4 B), flag (8 B at +8), one format word per peer (at +16).
    return {"packed": packed, "weight": weight, "peer": peer, "fc": fc, "bytes": fc + 16 + npes * 4}


# fmt: off
def make_g2_tile_runner(*, BM, BN, BK, SBM, use_nt, INTER_MAX, a_dtype, g2_bhoist, g2_ascale_pf, expert_offset,
    npes, topk, N_OUT, max_tok, recv_cap, comb_inp_nbytes, clamp_a, store_cm=2, row_mask=False,
    p2p_quant_type="none", uniform=False, epi_lean=0, tok_done=False, bf16_head_tokens=0):
# fmt: on
    """One stage2 tile (m_block, n_block) as a jit helper, for kernels that run GEMM2 after their own phase.

    Uses the absolute LDS layout of ``g2_lds_layout`` (the caller's LDS pool must start at address 0); the peer
    table at ``layout['peer']`` must be loaded by the caller. Same numerics as the stage2 kernel.
    ``row_mask``: rows past the tile's valid count are dropped here, so the plan need not pad their srcmap.
    ``uniform``: the pointer arguments move to SGPRs (no waterfall loops around the tile's buffer loads).
    ``epi_lean`` / ``tok_done`` (with the tile's ``i64_tok_off``) / ``bf16_head_tokens``: p2p_scatter_epilog's
    ``lean`` / ``tok_off`` / ``bf16_head_tokens``.
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
        i32_max_m_blocks: fx.Int32, i32_inter: fx.Int32, i32_hidden: fx.Int32, i32_fp8_rows: fx.Int32,
        i64_tok_off: fx.Int64):
    # fmt: on
        # The pointers come from the dispatch table (VGPR loads): keep them in SGPRs so the tile's buffer
        # resources need no waterfall loop around every load.
        if const_expr(uniform):
            arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_stids, arg_sweights, arg_trb, arg_tile_rows = (
                _uniform_i64(a) for a in (arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids, arg_stids, arg_sweights,
                    arg_trb, arg_tile_rows))
            i32_max_m_blocks = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(i32_max_m_blocks)))
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
            lds_peer_off=lay["peer"], g2_bf16_lds=False, p2p_quant_type=p2p_quant_type, store_cm=store_cm,
            fp8_store_cm=2, fp8_rows=i32_fp8_rows, elem_view_stores=True, lean=epi_lean,
            tok_off=i64_tok_off if const_expr(tok_done) else None, bf16_head_tokens=bf16_head_tokens)
        # fmt: on

    return g2_tile, lay


_SC0_SC1_CACHE = 17


def _use_direct_fixed_slot(npes, experts_per_rank, max_tokens_per_rank, cap, tile_m):
    if tile_m <= 0 or max_tokens_per_rank <= 0:
        return False
    required_cap = ((npes * max_tokens_per_rank + tile_m - 1) // tile_m) * tile_m
    return (npes, experts_per_rank) in ((8, 48), (4, 96)) and cap == required_cap


# fmt: off
@functools.cache
def compile_mega_moe_stage1(
    *, model_dim: int, inter_dim: int, rank: int, experts_per_rank: int, fuse_npes: int, fuse_topk: int,
    fuse_cap: int, fuse_mtpr: int, fuse_scale_dim: int, fused: tuple, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 8,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, num_dispatch_cu: int = 32, b_nt: int = -1,
    swiglu_limit: float = 0.0, dedup_rows: bool = False, plan_direct: bool = False, plan_wt: bool = False,
    plan_local: bool = False, g2_early: tuple = (0, 0), g2_uniform: bool = False, head_uniform: bool = False,
    g2_epi_lean: int = 0, g1_lookahead: int = 1, g1_a_ahead: int = 1, direct_done: int = 0, tok_done: bool = False,
    tok_counters: bool = False, bf16_head_tokens: int = 0, epi_evec: int = 2, epi_lean: int = 0,
    ascale_packed: bool = False,
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
        raise ValueError("small-input stage1 needs the EP8/48-expert or EP4/96-expert fixed-slot layout")
    fz_total_experts = fz_npes * fz_epr
    # Producers load their route (expert, remote pointers, row, scales, weight) before the epoch gate and
    # quantize the bf16 input row to MXFP8. A producer wave sends pre_rounds routes (one per round) so
    # that npes groups of mtpr * topk routes fit.
    pre_rounds = ceildiv(fz_mtpr * fz_k, dispatch_blocks * NUM_WAVES)
    assert 0 <= direct_done <= _DIRECT_DONE_INC
    # Per-launch step of the expected source-flag value (see _DIRECT_DONE_INC); every bucket of a runtime and every
    # rank use the same direct_done setting, so they agree on it.
    done_step = _DIRECT_DONE_INC if direct_done else fz_npes
    assert not tok_done or tok_counters, "tok_done needs the runtime's per-token counters"
    # A tok_done rank leaves without the peers' flags, so its next launch's producers can add to a peer's expert
    # counters before that peer's combine zeroes them. A runtime with tok_done buckets (tok_counters) keeps two halves
    # of the counters by launch parity: launch N counts into half N & 1 and its last WG zeroes that half; the next
    # one a peer can add to it is launch N + 2, which needs this rank's launch N + 1 sources, so launch N is over.
    # The routes' srcmap and weight slots follow the same halves: the offsets restart at 0 in every launch, so a
    # rank's launch N + 1 routes would otherwise overwrite the slots of launch N routes a peer's GEMM1 / GEMM2 still
    # reads (that peer's sources are not done with launch N).
    assert not tok_counters or plan_local, "the double-buffered expert counters are zeroed by the combine"
    # Only the dedup token rows (indexed by source token) leave the slot-indexed payload rows unused; those
    # rows are single-buffered, so a run-ahead launch would overwrite them as it did the srcmap.
    assert not tok_counters or dedup_rows, "a runtime with tok_done buckets must send dedup token rows"
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
        clamp_a=True, store_cm=0, row_mask=True, p2p_quant_type="dynamic", uniform=g2_uniform,
        epi_lean=g2_epi_lean, tok_done=tok_done, bf16_head_tokens=bf16_head_tokens,
    )
    g2_tile_cached, _ = make_g2_tile_runner(
        BM=sort_block_m, BN=g2_bn, BK=g2_bk, SBM=sort_block_m, use_nt=False, INTER_MAX=inter_dim,
        a_dtype="fp8", g2_bhoist=True, g2_ascale_pf=True, expert_offset=int(rank) * int(experts_per_rank),
        npes=int(fuse_npes), topk=int(fuse_topk), N_OUT=model_dim, max_tok=int(fuse_mtpr),
        recv_cap=int(g2_recv_cap), comb_inp_nbytes=int(fuse_mtpr) * int(fuse_topk) * model_dim * 2,
        clamp_a=True, store_cm=0, row_mask=True, p2p_quant_type="dynamic", uniform=g2_uniform,
        epi_lean=g2_epi_lean, tok_done=tok_done, bf16_head_tokens=bf16_head_tokens,
    )
    # The combine input is in the uncached shmem heap: the reducers only need to drop their L1.
    fused_combine_tail = make_fused_combine_tail(lds_fc_off=g2_lay["fc"], reducers=fused_reducers,
        npes=int(fuse_npes), rank=int(rank), topk=int(fuse_topk), total_experts=fz_total_experts,
        experts_per_rank=fz_epr, N_OUT=model_dim, flag_is_value=True, l1_acquire=True, row_format=True,
        reset_running=fz_epr if plan_local else 0, reset_tok=fz_mtpr if tok_counters else 0, tok_done=tok_done,
        n_blocks=g2_n_blocks, bf16_head_tokens=bf16_head_tokens)
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

    plan_chunks = (fz_epr + 63) // 64
    # g2_early = (K, P): one CTA in K (by in-XCD workgroup index) leaves GEMM1 for GEMM2 after a unit claimed past
    # P/16 of its XCD's GEMM1 slice, so the GEMM2 scatter uses the XGMI links the finished dispatch left idle while
    # GEMM1 still runs. GEMM2 units then take their own claim word (word 3 of the XCD's line) instead of following
    # the GEMM1 units on the GEMM1 counter.
    G2_EARLY_K, G2_EARLY_P = (int(v) for v in g2_early)
    assert G2_EARLY_K == 0 or (G2_EARLY_K >= 2 and 0 < G2_EARLY_P < 16)

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]
        # Work-unit broadcast word, kept out of the pool: a fast wave's next A / C-shuffle store must not race the
        # slower waves' read of the unit it starts.
        work_word: fx.Array[fx.Int8, 16, 16]
        if plan_local:
            # Per expert slot (64-expert chunks): its inclusive tile count, then its row count.
            plan_tab: fx.Array[fx.Int8, plan_chunks * 64 * 8, 16]

    swiglu_suffix = "" if swiglu_limit <= 0 else f"_sl{str(float(swiglu_limit)).replace('.', 'p')}"
    kernel_name = (
        f"megamoe_small_stage1_t{sort_block_m}x{tile_n}x{tile_k}"
        f"_w{NUM_WAVES}_gm{grid_mult}"
        f"_dcu{dispatch_blocks}_pw{int(pipe_weights)}ma{int(mfma_amajor)}sw{int(swizzle_a)}"
        f"aa{int(async_a_copy)}"
        f"_tr{int(use_tile_resource)}wpe{waves_per_eu_hint}_bnt{b_cache_modifier}"
        f"{swiglu_suffix}"
        f"_fused{'_'.join(str(v) for v in fused)}"
        f"{'_dd' if dedup_rows else ''}"
        f"{'_pd' if plan_direct else ''}"
        f"{'_pw' if plan_wt else ''}"
        f"{'_pl' if plan_local else ''}"
        f"{'_g2u' if g2_uniform else ''}{f'_g2l{g2_epi_lean}' if g2_epi_lean else ''}"
        f"{'_hu' if head_uniform else ''}{f'_la{g1_lookahead}' if g1_lookahead != 1 else ''}"
        f"{f'_ga{g1_a_ahead}' if g1_a_ahead != 1 else ''}{f'_dn{direct_done}' if direct_done else ''}"
        f"{'_td' if tok_done else ''}"
        f"{f'_ev{epi_evec}' if epi_evec != 2 else ''}{f'_el{epi_lean}' if epi_lean else ''}"
        f"{'_ap' if ascale_packed else ''}"
        f"{f'_ek{G2_EARLY_K}p{G2_EARLY_P}' if G2_EARLY_K else ''}"
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
            v = _buffer_load(disp_rsrc, fx.Int32(int(slot)), fx.Int64)
            if const_expr(head_uniform):
                v = _uniform_i64(v)
            return v

        a_entry_count = _disp_ptr(DispatchSlot.ENTRY_COUNT)
        a_epoch_gate = _disp_ptr(DispatchSlot.EPOCH_GATE)
        a_work_head = _disp_ptr(DispatchSlot.WORK_HEAD)
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)
        if const_expr(dedup_rows):
            token_offsets = (_disp_ptr(DispatchSlot.TOKEN_ROWS), _disp_ptr(DispatchSlot.TOKEN_SCALES))

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
            comm_ops.wait_i32_until_equals(gate_addr, gate_epoch)

        def _load_epoch():
            # Independent loads (no parity -> expected chain); only valid after the gate.
            parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_SC1_CACHE)
            expected0 = _buffer_load(expected_rsrc, fx.Int32(0), fx.Int32, cache_modifier=_SC0_SC1_CACHE)
            expected1 = _buffer_load(expected_rsrc, fx.Int32(1), fx.Int32, cache_modifier=_SC0_SC1_CACHE)
            return parity, (parity == fx.Int32(0)).select(expected0, expected1)
        producer_slot = ticket - fx.Int32(1)
        # tok_counters: this launch's half of the expert counters (every rank's launch has the same entry epoch).
        running_half = gate_epoch & fx.Int32(1) if const_expr(tok_counters) else None
        # tok_counters: byte offset of this launch's half of the srcmap and weight slots.
        meta_half = fx.Int64(running_half) * fx.Int64(fz_epr * fz_cap * 4) if const_expr(tok_counters) else None

        def _slot_meta_ptr(slot):
            v = _disp_ptr(slot)
            return v + meta_half if const_expr(tok_counters) else v

        if owner:
            if tid < fx.Int32(64):
                work_head_rsrc = _make_buffer_from_addr(a_work_head, fx.Int32)
                if tid < fx.Int32(8):
                    # The XCD's work counter and its counts of its own GEMM1 units in the m-tiles cut at its
                    # slice ends.
                    _buffer_store(work_head_rsrc, tid * fx.Int32(16), fx.Int32(0), fx.Int32)
                    _buffer_store(work_head_rsrc, tid * fx.Int32(16) + fx.Int32(1), fx.Int32(0), fx.Int32)
                    _buffer_store(work_head_rsrc, tid * fx.Int32(16) + fx.Int32(2), fx.Int32(0), fx.Int32)
                    if const_expr(G2_EARLY_K > 0):
                        _buffer_store(work_head_rsrc, tid * fx.Int32(16) + fx.Int32(3), fx.Int32(0), fx.Int32)
                if tid == fx.Int32(16):
                    _buffer_store(_make_buffer_from_addr(a_group_done, fx.Int32), fx.Int32(0), fx.Int32(0), fx.Int32)
                old_parity = _buffer_load(parity_rsrc, fx.Int32(0), fx.Int32)
                old_expected0 = _buffer_load(expected_rsrc, fx.Int32(0), fx.Int32)
                old_expected1 = _buffer_load(expected_rsrc, fx.Int32(1), fx.Int32)
                next_parity = old_parity ^ fx.Int32(1)
                next_expected = (
                    (next_parity == fx.Int32(0)).select(old_expected0, old_expected1) + fx.Int32(done_step)
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
                dedup=dedup_rows, uniform=head_uniform,
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
                prefetch_args=prefetch_args, dedup=dedup_rows, direct_done=direct_done, running_half=running_half,
            )
        payload_parity, payload_expected = _load_epoch()
        if owner:
            emit_direct_fixed_slot_finalize(
                fz_npes=fz_npes, fz_epr=fz_epr, fz_cap=fz_cap, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                fz_tile_m=fz_tile_m, addr_disp=addr_disp, parity=payload_parity,
                expected=payload_expected, plan_nv=(_disp_ptr(DispatchSlot.PLAN_NV), gate_epoch),
                plan_wt=plan_wt, keep_running=plan_local, uniform=head_uniform, running_half=running_half,
            )
        wave_id = fx.thread_idx.x // 64

        w_rsrc = ptr_buf_tensor(fx.get_iter(w), fx.Int32, unit_elems=4)
        sx_rsrc = ptr_buf_tensor(fx.get_iter(scale_x), fx.Int32, unit_elems=4)
        sw_rsrc = ptr_buf_tensor(fx.get_iter(scale_w), fx.Int32)
        trb_rsrc = ptr_buf_tensor(fx.get_iter(sorted_token_ids), fx.Int32)
        expert_rsrc = ptr_buf_tensor(fx.get_iter(expert_ids), fx.Int32)
        scale_cols = (inter_dim // 32 + 7) // 8 * 8
        os_nbytes = tokens * fx.Int32(scale_cols) + fx.Int32(8192)
        if const_expr(use_tile_resource):
            out_rsrc = None
        else:
            out_nbytes = tokens * fx.Int32(inter_dim)
            out_rsrc = ptr_buf_tensor(fx.get_iter(out), fx.Int16, num_records_bytes=out_nbytes)
        os_rsrc = ptr_buf_tensor(fx.get_iter(out_scale), fx.Int8, num_records_bytes=os_nbytes)

        # dedup_rows: the rows sit in the token-row region (row source rank * mtpr + token) and GEMM1 maps its tile
        # rows through the srcmap; the region offsets are read past the plan.
        row_map = None
        if const_expr(dedup_rows):
            row_map = {
                "srcmap": _make_buffer_from_addr(_slot_meta_ptr(DispatchSlot.G2_STIDS), fx.Int32),
                # A constant bound keeps the A buffer resource scalar.
                "x_bytes": fx.Int32(_TOKEN_REGION_END),
            }
        plan_lookup = None
        if const_expr(plan_local):
            plan_tab_base = fx.Int32(fx.ptrtoint(lds.plan_tab.ptr))

            def plan_lookup(m_tile):
                # The tile's expert = the experts whose inclusive tile count is <= m_tile.
                lane = tid & fx.Int32(63)
                expert = fx.Int32(0)
                for chunk in range_constexpr(plan_chunks):
                    incl = fx.ptr_load(lds_typed_ptr(plan_tab_base + (fx.Int32(chunk * 64) + lane) * fx.Int32(4),
                        T.i32, align=4), result_type=T.i32)
                    mask = rocdl.ballot(T.i64, fx.Int32(incl) <= m_tile)
                    expert = expert + fx.Int32(fx.Int64(fx.math.ctpop(mask)))
                expert = fx.Int32(rocdl.readfirstlane(T.i32, expert.ir_value()))
                incl_e = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lds_typed_ptr(
                    plan_tab_base + expert * fx.Int32(4), T.i32, align=4), result_type=T.i32)))
                count_e = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(lds_typed_ptr(
                    plan_tab_base + fx.Int32(plan_chunks * 256) + expert * fx.Int32(4), T.i32, align=4),
                    result_type=T.i32)))
                tiles_e = (count_e + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
                tile_rows_base = (m_tile - incl_e + tiles_e) * fx.Int32(fz_tile_m)
                rows = count_e - tile_rows_base
                rows = (rows < fx.Int32(fz_tile_m)).select(rows, fx.Int32(fz_tile_m))
                return expert, expert * fx.Int32(fz_cap) + tile_rows_base, rows
        _, _do_scheduled_tile = build_fused_gemm1(
            x_tensor=x, w_rsrc=w_rsrc,
            sw_rsrc=sw_rsrc, sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc,
            trb_rsrc=trb_rsrc, tib_rsrc=trb_rsrc, expert_rsrc=expert_rsrc, out_tensor=out,
            a_buf=a_buf, a_scale_lds=a_scale_lds, c_tile=c_tile,
            model_dim=model_dim, inter_dim=inter_dim, sort_block_m=sort_block_m,
            tile_n=tile_n, num_waves=NUM_WAVES, n_per_wave=n_per_wave, wave_id=wave_id,
            m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_k_step_bytes=A_K_STEP_BYTES,
            total_threads=TOTAL_THREADS, k_iters=K_ITERS, a_lds_i32=a_lds_i32,
            n_tiles=N_TILES, expert_offset=fz_rank * fz_epr, b_cache_modifier=b_cache_modifier,
            swizzle_a=swizzle_a, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
            async_a_copy=async_a_copy, use_tile_resource=use_tile_resource, indirect_input=False, evec=epi_evec,
            swiglu_limit=swiglu_limit,
            rows_rsrc=ptr_buf_tensor(_disp_ptr(DispatchSlot.TILE_ROWS), fx.Int32),
            lookahead=g1_lookahead, row_map=row_map, plan_lookup=plan_lookup, a_ahead=g1_a_ahead,
            fused_v8=True, epi_lean=epi_lean, ascale_packed=ascale_packed,
        )
        # A rank that receives many rows holds experts with many m-tiles: it reads W1 through the cache so they
        # reuse it, while the other ranks keep streaming W1 once per tile.
        if const_expr(b_cache_modifier != 0):
            _, _do_cached_tile = build_fused_gemm1(
                x_tensor=x, w_rsrc=w_rsrc,
                sw_rsrc=sw_rsrc, sx_rsrc=sx_rsrc, out_rsrc=out_rsrc, os_rsrc=os_rsrc,
                trb_rsrc=trb_rsrc, tib_rsrc=trb_rsrc, expert_rsrc=expert_rsrc, out_tensor=out,
                a_buf=a_buf, a_scale_lds=a_scale_lds, c_tile=c_tile,
                model_dim=model_dim, inter_dim=inter_dim, sort_block_m=sort_block_m,
                tile_n=tile_n, num_waves=NUM_WAVES, n_per_wave=n_per_wave, wave_id=wave_id,
                m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_k_step_bytes=A_K_STEP_BYTES,
                total_threads=TOTAL_THREADS, k_iters=K_ITERS, a_lds_i32=a_lds_i32,
                n_tiles=N_TILES, expert_offset=fz_rank * fz_epr, b_cache_modifier=0,
                swizzle_a=swizzle_a, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor,
                async_a_copy=async_a_copy, use_tile_resource=use_tile_resource, indirect_input=False, evec=epi_evec,
                swiglu_limit=swiglu_limit,
                rows_rsrc=ptr_buf_tensor(_disp_ptr(DispatchSlot.TILE_ROWS), fx.Int32),
                lookahead=g1_lookahead, row_map=row_map, plan_lookup=plan_lookup, a_ahead=g1_a_ahead,
                fused_v8=True, epi_lean=epi_lean, ascale_packed=ascale_packed,
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
        g2_stids = _slot_meta_ptr(DispatchSlot.G2_STIDS)
        g2_sweights = _slot_meta_ptr(DispatchSlot.G2_SWEIGHTS)
        g2_trb = _disp_ptr(DispatchSlot.G2_TRB)
        g2_rows = _disp_ptr(DispatchSlot.TILE_ROWS)
        g2_max_m_blocks = fx.Int32(_disp_ptr(DispatchSlot.G2_MAX_M_BLOCKS))
        g2_scratch = lds_typed_ptr(fx.Int32(g2_scratch_off), T.i32, align=4)

        first_local = fx.Int32(0)
        nv_ptr = lds_typed_ptr(fx.Int32(g2_scratch_off + 4), T.i32, align=4)

        def _local_plan():
            # Wave 0: wait for every source as the plan CTA does, then build this CTA's plan table in LDS. The
            # counts are final once the sources are done; they are read past L2 (the last WG of the launch
            # that used them before zeroed them). Returns the plan word's low half.
            lane = tid & fx.Int32(63)
            a_done = _disp_ptr(DispatchSlot.COUNT_DONE)
            for source in range(lane, fz_npes, 64):
                done_index = payload_parity * fx.Int32(fz_npes) + source
                comm_ops.wait_i32_until_equals(a_done + fx.Int64(done_index) * fx.Int64(4), payload_expected)
            comm_ops.invalidate_l1()
            a_running = _disp_ptr(DispatchSlot.RUNNING)
            if const_expr(tok_counters):
                a_running = a_running + fx.Int64(running_half) * fx.Int64(fz_epr * 4)
            chunks, total_tiles, routed_rows, max_count, _overflow, no_overflow = _plan_counts(
                a_running, lane, fz_epr, fz_cap, fz_tile_m, count_cm=_SC0_SC1_CACHE)
            for chunk in range_constexpr(plan_chunks):
                _valid, _expert, safe_count, num_expert_tiles, metadata_base = chunks[chunk]
                slot = (fx.Int32(chunk * 64) + lane) * fx.Int32(4)
                fx.ptr_store(metadata_base + num_expert_tiles, lds_typed_ptr(plan_tab_base + slot, T.i32, align=4))
                fx.ptr_store(safe_count, lds_typed_ptr(plan_tab_base + fx.Int32(plan_chunks * 256) + slot, T.i32,
                    align=4))
            hot_expert = max_count >= fx.Int32(_HOT_EXPERT_MIN_ROWS)
            return _plan_flag(no_overflow, total_tiles, hot_expert, routed_rows, fz_tile_m)
        if tid == fx.Int32(0):
            xcd = fx.Int32(comm_ops.hw_xcc_id()) & fx.Int32(7)
            # The owner reset the counters before opening the gate: the first work fetch overlaps the plan wait.
            first_local = fx.Int32(
                comm_ops.atomic_add_agent(a_work_head + fx.Int64(xcd) * fx.Int64(64), fx.Int32(1)))
            if const_expr(not plan_local):
                a_xcd_acq = _disp_ptr(DispatchSlot.XCD_ACQ) + fx.Int64(xcd) * fx.Int64(128)
                plan_word = fx.Int64(0)
                # Match this launch's epoch in the high word exactly: a >= wait breaks when the 32-bit epoch wraps.
                if const_expr(plan_direct):
                    plan_word = fx.Int64(comm_ops.spin_until_shifted_eq_i64(
                        _disp_ptr(DispatchSlot.PLAN_NV), gate_epoch, shift=32))
                    comm_ops.invalidate_l1()
                elif first_local == fx.Int32(0):
                    plan_word = fx.Int64(comm_ops.spin_until_shifted_eq_i64(
                        _disp_ptr(DispatchSlot.PLAN_NV), gate_epoch, shift=32))
                    comm_ops.fence_agent_acquire()
                    comm_ops.store_i64_global_system(a_xcd_acq, plan_word)
                else:
                    plan_word = fx.Int64(comm_ops.spin_until_shifted_eq_i64(a_xcd_acq, gate_epoch, shift=32))
                    comm_ops.invalidate_l1()
                fx.ptr_store(fx.Int32(plan_word & fx.Int64(0xFFFFFFFF)), nv_ptr)
        if const_expr(plan_local):
            if tid < fx.Int32(64):
                plan_low_local = _local_plan()
                if tid == fx.Int32(0):
                    fx.ptr_store(plan_low_local, nv_ptr)
        fx.barrier()
        plan_low = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.ptr_load(nv_ptr).ir_value()))
        num_valid = plan_low & fx.Int32((1 << _PLAN_CACHED_W_BIT) - 1)
        if const_expr(dedup_rows):
            row_map["x_token"] = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(token_offsets[0]).ir_value()))
            row_map["scale_token"] = fx.Int32(
                rocdl.readfirstlane(T.i32, fx.Int32(token_offsets[1] // fx.Int64(16)).ir_value()))
        cached_w = (plan_low >> fx.Int32(_PLAN_CACHED_W_BIT)) & fx.Int32(1)
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
        # (2 n-tiles of tile_n 256 cover the same 256 output columns when N_TILES is not a multiple of 4.)
        cut_tiles = 4 if N_TILES % 4 == 0 else 2
        assert N_TILES % cut_tiles == 0 and tile_n * cut_tiles >= 512
        shard_base = ((work_shard * total_work) // fx.Int32(8)) & fx.Int32(-cut_tiles)
        shard_work = (((work_shard + fx.Int32(1)) * total_work) // fx.Int32(8) & fx.Int32(-cut_tiles)) - shard_base

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
        g1_leave = fx.Int32(0)
        early_thr = shard_work
        if const_expr(G2_EARLY_K > 0):
            early_cta = ((fx.Int32(fx.block_idx.x) >> fx.Int32(3)) % fx.Int32(G2_EARLY_K)) == fx.Int32(0)
            early_thr = early_cta.select(shard_work * fx.Int32(G2_EARLY_P) // fx.Int32(16), shard_work)

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
                    # One writer per m-tile and launch; GEMM2 waits for equality, so the epoch may wrap
                    # (a max would stick at 0xffffffff).
                    comm_ops.store_i32_global_agent_release(a_g2_ready + fx.Int64(m) * fx.Int64(4), gate_epoch)
                    # All counts of this launch are in: rewind to 0, since the next launch's config may use another
                    # N_TILES (tile_n 256 / 128), and a leftover count would make its ready fire early or never.
                    comm_ops.atomic_add_agent(cnt_addr, fx.Int32(-N_TILES))
        while consumer_active:
            if tid == fx.Int32(0):
                local_work = first_local
                if g1_fetch != fx.Int32(0):
                    if const_expr(G2_EARLY_K > 0):
                        local_work = shard_work
                        if g1_leave == fx.Int32(0):
                            local_work = fx.Int32(comm_ops.atomic_add_agent(
                                a_work_head + fx.Int64(work_shard) * fx.Int64(64), fx.Int32(1)))
                    else:
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
                if const_expr(b_cache_modifier != 0):
                    if cached_w != fx.Int32(0):
                        _do_cached_tile(work)
                    else:
                        _do_scheduled_tile(work)
                else:
                    _do_scheduled_tile(work)
                # Same-XCD handoff: store completion puts the tile in the shared L2; the next work fetch
                # counts it and the last of the m-tile's N_TILES CTAs raises its ready epoch.
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.barrier()
            pend_m = (has_work != fx.Int32(0)).select(work // fx.Int32(N_TILES), fx.Int32(-1))
            consumer_active = has_work != fx.Int32(0)
            g1_fetch = fx.Int32(1)
            if const_expr(G2_EARLY_K > 0):
                g1_leave = ((has_work != fx.Int32(0)) & (work - shard_base >= early_thr)).select(
                    fx.Int32(1), fx.Int32(0))

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
        g2_claim_base = shard_work
        if const_expr(G2_EARLY_K > 0):
            g2_head = g2_head + fx.Int64(12)
            g2_fetch = fx.Int32(1)
            g2_claim_base = fx.Int32(0)
        # A rank that receives many rows sends its GEMM2 rows as FP8: they halve the traffic it would otherwise
        # spend most of its GEMM2 sending; the other ranks keep BF16 rows.
        fp8_rows = (plan_low >> fx.Int32(_PLAN_FP8_ROWS_BIT)) & fx.Int32(1)
        if const_expr(tok_done):
            fp8_rows = fx.Int32(0)
        tok_off = tok_reset = fx.Int64(0)
        if const_expr(tok_counters):
            # tok_done counters: this launch's half and the previous launch's (by the parity of the shared epoch).
            tok_base = _disp_ptr(DispatchSlot.TOK_DONE_OFF)
            tok_par = fx.Int64(gate_epoch & fx.Int32(1))
            tok_off = tok_base + tok_par * fx.Int64(fz_mtpr * 4)
            tok_reset = tok_base + (fx.Int64(1) - tok_par) * fx.Int64(fz_mtpr * 4)
        if const_expr(plan_local):
            # GEMM2 reads the plan CTA's metadata (its first unit drops the L1 after its ready wait).
            if tid == fx.Int32(0):
                comm_ops.spin_until_shifted_eq_i64(_disp_ptr(DispatchSlot.PLAN_NV), gate_epoch, shift=32)
        while g2_active:
            fx.barrier()
            if tid == fx.Int32(0):
                if g2_fetch != fx.Int32(0):
                    fx.ptr_store(fx.Int32(comm_ops.atomic_add_agent(g2_head, fx.Int32(1))) - g2_claim_base,
                        g2_scratch)
            fx.barrier()
            g2_local = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.ptr_load(g2_scratch).ir_value()))
            if g2_local < g2_units:
                g2_u = g2_base + g2_local
                g2_m = g2_u // fx.Int32(g2_n_blocks)
                g2_n = g2_u - g2_m * fx.Int32(g2_n_blocks)
                if tid == fx.Int32(0):
                    comm_ops.wait_i32_until_equals(a_g2_ready + fx.Int64(g2_m) * fx.Int64(4), gate_epoch)
                fx.barrier()
                # GEMM1 wrote this m-tile through this XCD's L2, or (in an m-tile shared by two XCDs) written back
                # to memory by the other XCD into lines this XCD has not cached, before it was counted: only the
                # CU's L1 can be stale.
                comm_ops.invalidate_l1()
                # As in GEMM1, a rank that receives many rows reads W2 through the cache.
                if cached_w != fx.Int32(0):
                    g2_tile_cached(lds_base_i32, tid, g2_lane, g2_wave, g2_m, g2_n, g2_aq, g2_ascale, g2_bq,
                        g2_bscale, g2_eids, g2_stids, g2_sweights, g2_trb, g2_rows, g2_max_m_blocks,
                        fx.Int32(inter_dim), fx.Int32(model_dim), fp8_rows, tok_off)
                else:
                    g2_tile(lds_base_i32, tid, g2_lane, g2_wave, g2_m, g2_n, g2_aq, g2_ascale, g2_bq, g2_bscale,
                        g2_eids, g2_stids, g2_sweights, g2_trb, g2_rows, g2_max_m_blocks, fx.Int32(inter_dim),
                        fx.Int32(model_dim), fp8_rows, tok_off)
            g2_active = g2_local < g2_units
            g2_fetch = fx.Int32(1)

        def reset_running_addr():
            a_running = _disp_ptr(DispatchSlot.RUNNING)
            if const_expr(tok_counters):
                # The combine's last WG zeroes this launch's half of the expert counters.
                a_running = a_running + fx.Int64(running_half) * fx.Int64(fz_epr * 4)
            return a_running

        # The entry-ticket epoch is the same on every rank and needs no load; the high bit keeps it apart from
        # the combine op's own flag values in the shared xdb slots.
        fc_flag = fx.Int64(gate_epoch) + fx.Int64(1 << 40)
        fused_combine_tail(tid, g2_lane, g2_wave, _disp_ptr(DispatchSlot.FC_CNT), fc_flag,
            _disp_ptr(DispatchSlot.FC_XDB), _disp_ptr(DispatchSlot.FC_P2P_XDB), _disp_ptr(DispatchSlot.FC_INP),
            _disp_ptr(DispatchSlot.FC_OUT), _disp_ptr(DispatchSlot.FC_TRECV), addr_in_idx, i32_cur_tok,
            fx.Int32(launch_grid_x), fp8_rows,
            reset_running_addr() if const_expr(plan_local) else fx.Int64(0), tok_off, tok_reset)

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
    b_nt=-1, swiglu_limit=0.0, preload=False, **fused_knobs):
    launch = compile_mega_moe_stage1(
        model_dim=model_dim, inter_dim=inter_dim, rank=rank, experts_per_rank=experts_per_rank,
        fuse_npes=fuse_npes, fuse_topk=fuse_topk, fuse_cap=fuse_cap, fuse_mtpr=fuse_mtpr,
        fuse_scale_dim=fuse_scale_dim, fused=tuple(fused),
        sort_block_m=sort_block_m, tile_n=tile_n, tile_k=tile_k, num_waves=num_waves,
        grid_mult=grid_mult, pipe_weights=pipe_weights, mfma_amajor=mfma_amajor, swizzle_a=swizzle_a,
        async_a_copy=async_a_copy, use_tile_resource=use_tile_resource,
        waves_per_eu_hint=waves_per_eu_hint, num_cu=num_cu, num_dispatch_cu=num_dispatch_cu,
        b_nt=b_nt, swiglu_limit=swiglu_limit, **fused_knobs,
    )
    # preload: compile and load without dispatching (AOT bundles).
    (_preload_compiled if preload else _run_compiled)(
        launch, out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, out_scale,
        tokens, addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
        addr_parity, addr_expected, stream,
    )
# fmt: on


# Combine reducer CTAs of the single kernel.
_COMBINE_REDUCERS = 256
# Routed rows a rank receives (before tile padding) from which the streaming GEMM1 / GEMM2 bodies give way to
# cached-weight ones, whose experts then span enough m-tiles to reuse their weights. Measured: cached loads
# cost uniform routing at 384 tokens (2304 rows) and gain from 448 (2688).
_CACHED_W_MIN_ROWS = 2560
# Routed rows a rank receives from which its GEMM2 sends FP8 rows (1x32 E8M0 scales, the numerics of the main
# path) to the combine instead of BF16: above the 3072 rows of uniform routing at 512 tokens.
_FP8_ROWS_MIN_ROWS = 3584
# Rows of a rank's largest expert from which the rank also takes both of the above: its rows sit in a few experts
# (skewed routing), far above the 64-row average of uniform routing at 512 tokens.
_HOT_EXPERT_MIN_ROWS = 256
# Bits of the plan word's low half that carry the two decisions above (num_valid < 2^29).
_PLAN_CACHED_W_BIT = 29
_PLAN_FP8_ROWS_BIT = 30
# The word the combine leaves in the rows of the bf16_head_tokens tokens (FusedStage1Knobs) once it summed them.
_ROW_POISON = -1
# dedup_rows: byte offset (from the payload / scale buffer base) that the token-row regions end below.
_TOKEN_REGION_END = (1 << 31) - (1 << 16)


def _kernel_knobs(config: MegaMoEConfig) -> dict:
    """compile_mega_moe_stage1's fused-kernel knobs of a config."""
    k1, k2 = config.fused_stage1, config.fused_stage2
    # fmt: off
    return dict(  # noqa: C408
        dedup_rows=k1.dedup_rows, plan_direct=k1.plan_direct, plan_wt=k1.plan_wt, plan_local=k1.plan_local,
        g2_early=(k2.early_k, k2.early_p if k2.early_k else 0), g2_uniform=k1.g2_uniform,
        head_uniform=k1.head_uniform, g2_epi_lean=k1.g2_epi_lean, g1_lookahead=k1.g1_lookahead,
        g1_a_ahead=k1.g1_a_ahead, direct_done=k1.direct_done, tok_done=k1.tok_done, tok_counters=k1.tok_counters,
        bf16_head_tokens=k1.bf16_head_tokens, epi_evec=k1.epi_evec, epi_lean=k1.epi_lean,
        ascale_packed=k1.ascale_packed,
    )
    # fmt: on


class SmallMRuntime:
    """Small-input single-kernel MegaMoE (MTPR <= 512) with one in-flight launch per instance.

    Owns its own dispatch/combine buffers and protocol state; weights are bound
    by pointer so one instance serves every layer.
    """

    # Group-major dispatch layout of the fused kernel: fixed slots of 32 rows.
    GM_UNIT_SIZE = 32
    GM_COMPACT = False

    @classmethod
    def payload_capacity(
        cls, *, world_size, model_dim, inter_dim, experts, topk, max_tok_per_rank
    ):
        """group_major_capacity() of this runtime's dispatch, for a shared payload arena.

        A layout with dedup_rows dispatch also keeps its token-row regions (one row and its scales per source
        rank and token) in the arena, right past the fixed slots.
        """
        nvm, rows, row_bytes, scale_n_i32 = group_major_capacity(
            world_size=world_size,
            hidden_dim=model_dim,
            max_tok_per_rank=max_tok_per_rank,
            experts_per_rank=experts // world_size,
            topk=topk,
            data_type=torch.float8_e4m3fn,
            unit_size=cls.GM_UNIT_SIZE,
            scale_dim=model_dim // 32,
            compact=cls.GM_COMPACT,
        )
        config = select_fused_config(
            1, max_tok_per_rank, world_size=world_size, experts_per_rank=experts // world_size,
            model_dim=model_dim, inter_dim=inter_dim,
        )
        if config.fused_stage1.dedup_rows:
            rows += world_size * max_tok_per_rank
        return nvm, rows, row_bytes, scale_n_i32

    @classmethod
    def ends_all_ranks(cls, *, world_size, model_dim, inter_dim, experts, topk, max_tok_per_rank):
        """Whether every launch ends only once every peer's grid finished (its combine waits for every peer's flag).

        A tok_done bucket's combine takes each token once its own rows are counted, so a rank can leave, and start
        its next call's dispatch into its peers, while a peer still reads its payload: such a runtime cannot share
        payload storage with another route.
        """
        return not any(
            select_fused_config(
                bucket, max_tok_per_rank, world_size=world_size, experts_per_rank=experts // world_size,
                model_dim=model_dim, inter_dim=inter_dim,
            ).fused_stage1.tok_done
            for bucket in FUSED_TOKEN_BUCKETS
            if bucket <= max_tok_per_rank
        )

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int, experts: int, topk: int,
        quant: str, w1: torch.Tensor, w1_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor,
        max_tok_per_rank: int, swiglu_limit: float = 0.0, payload_arena=None, s1_output_arena=None,
        combine_input=None):
    # fmt: on
        """``payload_arena`` / ``s1_output_arena`` / ``combine_input``: storage shared by
        the routes of one MegaMoEV2 (see GroupMajorPayloadArena, stage1_output_buffers and
        FlyDSLDispatchCombineIntraNodeOp); None allocates."""
        if quant != "a8w4":
            raise ValueError("SmallMRuntime currently supports quant='a8w4' only")
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
        if max_tok_per_rank <= 0 or max_tok_per_rank & (max_tok_per_rank - 1):
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two")
        if max_tok_per_rank > SMALL_M_MAX_MTPR:
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} > {SMALL_M_MAX_MTPR}")
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
        # Built directly, the runtime checks the layout MegaMoEV2 gates it to (the kernel splits its
        # work over 8 XCDs and bakes in the FUSED_LAYOUTS shapes), instead of failing at compile.
        if not (
            fused_layout_supported(
                arch=get_gfx_runtime(), world_size=self.world_size, model_dim=self.model_dim,
                inter_dim=self.inter_dim, experts=self.experts, topk=self.topk, quant=quant,
                swiglu_limit=self.swiglu_limit,
            )
            and small_m_cu_supported(torch.cuda.get_device_properties(self.dev).multi_processor_count)
        ):
            raise ValueError("SmallMRuntime supports the fused a8w4 layouts on 8-XCD gfx950 only")
        # The config of every token bucket. The ranks of a launch may pick different buckets, so the knobs that
        # change what ranks or launches exchange are the same for every bucket (one protocol per runtime).
        self._configs = {
            bucket: select_fused_config(
                bucket, self.mtpr, world_size=self.world_size, experts_per_rank=self.epr,
                model_dim=self.model_dim, inter_dim=self.inter_dim,
            )
            for bucket in FUSED_TOKEN_BUCKETS
            if bucket <= self.mtpr
        }
        protocols = {
            (k.dedup_rows, k.plan_local, bool(k.direct_done), k.tok_counters, k.bf16_head_tokens)
            for k in (config.fused_stage1 for config in self._configs.values())
        }
        assert len(protocols) == 1, f"buckets disagree on the launch protocol: {protocols}"
        # A tok_done reducer waits for valid routes x its own GEMM2 n-blocks on a token's counter, which the
        # senders' GEMM2 scatter waves count up one per n-block: every tok_done bucket splits model_dim alike.
        tok_done_block_n = {config.stage2.block_n for config in self._configs.values() if config.fused_stage1.tok_done}
        assert len(tok_done_block_n) <= 1, f"tok_done buckets disagree on the GEMM2 block_n: {tok_done_block_n}"
        self._protocol = self._configs[1].fused_stage1
        # The combine reducers run at the end of the kernel on a fully resident grid, so they are not bounded by
        # one XCD's residency.
        self._reducers = _COMBINE_REDUCERS
        self.max_recv = self.world_size * self.mtpr
        self._s1_scale_dim = self.model_dim // 32
        # A 0-row rank passes this 1-row stand-in (invalid expert ids): the kernel reads row 0 of its inputs
        # before checking the token count.
        self._idle_inputs = (
            torch.zeros((1, self.model_dim), dtype=torch.bfloat16, device=self.dev),
            torch.zeros((1, self.topk), dtype=torch.float32, device=self.dev),
            torch.full((1, self.topk), self.experts, dtype=torch.int32, device=self.dev),
        )
        # fmt: off
        self.comb_cfg = FlyDSLDispatchCombineConfig(rank=self.rank, world_size=self.world_size,
            hidden_dim=self.model_dim, max_num_inp_token_per_rank=self.mtpr, num_experts_per_rank=self.epr,
            num_experts_per_token=self.topk, combine_dtype=torch.bfloat16,
            dispatch_dtype=torch.float8_e4m3fn, scale_dim=self._s1_scale_dim, scale_type_size=1,
            enable_std_moe=False, enable_group_major=True, gm_unit_size=self.GM_UNIT_SIZE,
            gm_scheme="fixedslot", gm_compact=self.GM_COMPACT, max_total_recv_tokens=self.world_size)
        # fmt: on
        if payload_arena is None and self._protocol.dedup_rows:
            # The token-row regions live past the fixed slots of the payload arena (payload_capacity).
            payload_arena = GroupMajorPayloadArena([
                self.payload_capacity(world_size=self.world_size, model_dim=self.model_dim,
                    inter_dim=self.inter_dim, experts=self.experts, topk=self.topk, max_tok_per_rank=self.mtpr)
            ])
        self.comb_op = FlyDSLDispatchCombineIntraNodeOp(
            self.comb_cfg, gm_payload_arena=payload_arena, combine_input=combine_input
        )
        self._s1_output_arena = s1_output_arena
        torch.cuda.synchronize()
        ms.shmem_barrier_all()
        # Contiguous copies of strided weights live as long as the runtime: a graph captured with
        # them keeps their pointers even after another bind_weights() call.
        self._construction_weights = contiguous_weights((w1, w1_scale, w2, w2_scale))
        w1, w1_scale, w2, w2_scale = self._construction_weights
        self._weight_spec = weight_spec(self._construction_weights)
        self.w2 = w2
        self.w2_scale = w2_scale
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
        # dedup_rows dispatch sends one row per (source rank, token) and destination to the token-row regions past
        # the fixed slots of the payload arena (payload_capacity); GEMM1 reads them through u32 buffer offsets
        # from the payload / scale bases, the same on every rank.
        self._s1_token_offsets = (0, 0)
        if self._protocol.dedup_rows:
            arena = op.payload_arena
            token_rows = self.world_size * self.mtpr
            rows_off = op.rx_em.numel() * op.rx_em.element_size()
            scales_off = op.scale_em.numel() * op.scale_em.element_size()
            rows_end = rows_off + token_rows * op.row_bytes
            scales_end = scales_off + token_rows * op.scale_n_i32 * 4
            assert arena is not None and rows_end <= arena.rx_bytes and scales_end <= arena.scale_i32 * 4
            assert rows_off % 16 == 0 and scales_off % 16 == 0
            assert max(rows_end, scales_end) <= _TOKEN_REGION_END, "token rows past the u32 buffer offsets"
            self._s1_token_offsets = (rows_off, scales_off)

        self._s1_out, self._s1_osd = stage1_output_buffers(
            self._s1_nvm, self.inter_dim, self.dev, arena=self._s1_output_arena
        )
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
        if self._protocol.tok_counters:
            # Two parity halves of per-token GEMM2 completion counts, reached through each peer's FC_INP.
            workspace["tok_done"] = op._sym((2 * self.mtpr,), torch.int32)
            # Two launch-parity halves of the per-expert receive counters (compile_mega_moe_stage1), in place of
            # the combine op's single one; allocated after tok_done, so its distance from FC_INP is unchanged.
            workspace["running"] = op._sym((2 * self.epr,), torch.int32)
            # The routes' srcmap and weight slots, in the same two halves (fz_epr * fz_cap apart).
            # Half 1 starts epr * ll_cap past half 0; the non-compact slot layout (num_valid_max >= epr * ll_cap)
            # keeps it clear of half 0.
            assert op.num_valid_max >= self.epr * op.ll_cap, "double-buffered slots need the non-compact layout"
            meta_rows = op.num_valid_max + self.epr * op.ll_cap
            workspace["srcmap"] = op._sym((meta_rows,), torch.int32)
            workspace["wts"] = op._sym((meta_rows,), torch.float32)
        ms.shmem_barrier_all()
        workspace["p2p_count_done"] = op._p2p_table(workspace["count_done"])
        if self._protocol.tok_counters:
            workspace["p2p_running"] = op._p2p_table(workspace["running"])
            workspace["p2p_srcmap"] = op._p2p_table(workspace["srcmap"])
            workspace["p2p_wts"] = op._p2p_table(workspace["wts"])
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
        if self._protocol.tok_counters:
            table[DispatchSlot.RUNNING] = workspace["running"].data_ptr()
            table[DispatchSlot.P2P_RUNNING] = workspace["p2p_running"].data_ptr()
            table[DispatchSlot.P2P_SRCMAP] = workspace["p2p_srcmap"].data_ptr()
            table[DispatchSlot.P2P_WEIGHT] = workspace["p2p_wts"].data_ptr()
        table[DispatchSlot.G2_AQ] = self._s1_out.data_ptr()
        table[DispatchSlot.G2_ASCALE] = self._s1_osd.data_ptr()
        table[DispatchSlot.G2_BQ] = self.w2.data_ptr()
        table[DispatchSlot.G2_BSCALE] = self.w2_scale.data_ptr()
        table[DispatchSlot.G2_EIDS] = op.sorted_expert_ids.data_ptr()
        table[DispatchSlot.G2_STIDS] = op.srcmap_em.data_ptr()
        table[DispatchSlot.G2_SWEIGHTS] = op.wts_em.data_ptr()
        if self._protocol.tok_counters:
            table[DispatchSlot.G2_STIDS] = workspace["srcmap"].data_ptr()
            table[DispatchSlot.G2_SWEIGHTS] = workspace["wts"].data_ptr()
        table[DispatchSlot.G2_TRB] = op.tile_row_base.data_ptr()
        table[DispatchSlot.G2_P2P_COMB_INP] = comb_op._p2p_comb_inp.data_ptr()
        table[DispatchSlot.G2_MAX_M_BLOCKS] = (self._s1_nvm + self.sort_block_m - 1) // self.sort_block_m
        table[DispatchSlot.FC_CNT] = comb_op.comb_bar.data_ptr()
        table[DispatchSlot.FC_XDB] = comb_op.shmem_xdev_bar_mem.data_ptr()
        table[DispatchSlot.FC_P2P_XDB] = comb_op._p2p_xdb_mem.data_ptr()
        table[DispatchSlot.FC_INP] = comb_op.shmem_comb_inp_tok.data_ptr()
        table[DispatchSlot.FC_OUT] = comb_op.shmem_comb_out_tok.data_ptr()
        table[DispatchSlot.FC_TRECV] = comb_op.total_recv.data_ptr()
        if self._protocol.tok_counters:
            # tok_done reaches a peer's counters through the peer's FC_INP base, so the symmetric heap must keep
            # the two buffers at the same distance on every peer.
            tok_off = workspace["tok_done"].data_ptr() - comb_op.shmem_comb_inp_tok.data_ptr()
            for pe in range(self.world_size):
                peer_tok = ms.shmem_ptr_p2p(workspace["tok_done"].data_ptr(), self.rank, pe)
                peer_inp = ms.shmem_ptr_p2p(comb_op.shmem_comb_inp_tok.data_ptr(), self.rank, pe)
                assert peer_tok - peer_inp == tok_off, "tok_done is not heap-symmetric"
            table[DispatchSlot.TOK_DONE_OFF] = tok_off
        table[DispatchSlot.TOKEN_ROWS], table[DispatchSlot.TOKEN_SCALES] = self._s1_token_offsets
        self._s1_disp_template = table
        self._s1_disp = torch.tensor(table, dtype=torch.int64, device=self.dev)
        self._s1_disp_by_w2[(self.w2.data_ptr(), self.w2_scale.data_ptr())] = self._s1_disp

    def w2_dispatch_table(self, w2, w2_scale):
        """The dispatch table for these W2 weights, built on first (eager) use; binds nothing."""
        key = (w2.data_ptr(), w2_scale.data_ptr())
        disp = self._s1_disp_by_w2.get(key)
        if disp is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("fused dispatch table for these W2 weights was not built before graph capture")
            table = list(self._s1_disp_template)
            table[DispatchSlot.G2_BQ] = key[0]
            table[DispatchSlot.G2_BSCALE] = key[1]
            disp = torch.tensor(table, dtype=torch.int64, device=self.dev)
            self._s1_disp_by_w2[key] = disp
        return disp

    def bind_weights(self, w1, w1_scale, w2, w2_scale) -> None:
        """Bind layer weights.

        W1 pointers are launch arguments; W2 pointers live in the dispatch table,
        so each distinct W2 gets its own table, built on first (eager) use and
        reused afterwards, including inside captured graphs. A refused binding
        leaves the previous one in place.
        """
        check_bindable((w1, w1_scale, w2, w2_scale), self._weight_spec)
        w1_bytes, w1_scale_bytes = w1.view(torch.uint8), w1_scale.view(torch.uint8)
        disp = self.w2_dispatch_table(w2, w2_scale)
        self._s1_w1 = w1_bytes
        self._s1_w1_scale = w1_scale_bytes
        self.w2 = w2
        self.w2_scale = w2_scale
        self._s1_disp = disp

    def _select_config(self, tokens: int) -> MegaMoEConfig:
        config = self._configs[nearest_token_bucket(max(tokens, 1), FUSED_TOKEN_BUCKETS)]
        self._active_config = config
        return config

    def _launch(self, config: MegaMoEConfig, run_tokens, x_bf16, topk_ids, wts, stream, *, preload=False):
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
            waves_per_eu_hint=stage1.waves_per_eu_hint, b_nt=stage1.b_nt, swiglu_limit=self.swiglu_limit,
            preload=preload, **_kernel_knobs(config))
        # fmt: on

    def preload_aot_bundle(self):
        """Load the kernel of every token bucket this runtime selects, without GPU dispatch."""
        stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        x_bf16, wts, topk_ids = self._idle_inputs
        seen = set()
        for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= self.mtpr):
            config = self._select_config(bucket)
            key = (config.stage1, config.stage2, config.fused_stage1, config.fused_stage2)
            if key in seen:
                continue
            seen.add(key)
            self._launch(config, 1, x_bf16, topk_ids, wts, stream, preload=True)

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        run_tokens = check_forward_inputs(
            x_bf16, wts, topk_ids, max_rows=self.mtpr, model_dim=self.model_dim, topk=self.topk, device=self.dev
        )
        if run_tokens == 0:
            x_bf16, wts, topk_ids = self._idle_inputs
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        elif hasattr(stream, "cuda_stream"):
            stream = fx.Stream(stream.cuda_stream)
        # Weights may have been reassigned since the last call (see MegaMoEV2's weight attributes).
        self._s1_disp = self.w2_dispatch_table(self.w2, self.w2_scale)
        # Ranks may use different token buckets in one call; an idle rank uses the 1-token bucket.
        config = self._select_config(max(run_tokens, 1))
        self._launch(config, run_tokens, x_bf16, topk_ids, wts, stream)
        out_tok = self.comb_op.combine_output(self.mtpr)
        return out_tok[:run_tokens] if slice_output else out_tok

    __call__ = forward

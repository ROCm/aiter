# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# ruff: noqa: SIM102
"""MegaMoEV2 small-input path: quant, dispatch, GEMM1, GEMM2 and combine fused into one kernel.

Dispatch publishes each route into its expert's fixed slot on the destination rank (with dedup_rows the
row itself goes once per token and destination into a token-row region). The runtime serves up to 512
tokens per rank and owns its protocol state and combine output; inside MegaMoEV2 its payload,
GEMM1-output and combine-input storage are views of the instance's shared arenas.
"""

import functools
import os
from enum import IntEnum

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import cvt_pk_f32_fp8
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.aot_launch import aot_factory

from .. import communication_ops_utils as comm_ops
from ..flydsl_dispatch_combine_intranode_op import (
    FlyDSLDispatchCombineConfig,
    FlyDSLDispatchCombineIntraNodeOp,
    GroupMajorPayloadArena,
    _zeroed_sym,
    group_major_capacity,
)
from ..kernels_common import ceildiv
from ..mxfp4_gemm_common import kStages, lds_typed_ptr
from ..tensor_shim import (
    _preload_compiled,
    _run_compiled,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)
from .dispatch import _wave_inclusive_scan_i32, _wave_reduce_max_i32
from .gemm1 import _LdsF32View, build_fused_gemm1
from .gemm2 import gemm2_compute_v2, issue_a_load_lds_dt
from .mega_moe_config import (
    EXTRA_FIXED_SLOT_LAYOUTS,
    FUSED_TOKEN_BUCKETS,
    SMALL_M_MAX_MTPR,
    MegaMoEConfig,
    fused_bucket_override,
    fused_layout_supported,
    nearest_token_bucket,
    select_fused_config,
    shared_in_layout_enabled,
    small_m_cu_supported,
)
from .mega_moe_fused_gemm import AddrBuffer, load_at, store_at
from .mega_moe_stage1 import _validate_dispatch_capacity, stage1_output_buffers
from .mega_moe_stage2 import (
    _SC0_SC1_CACHE,
    Stage2Meta,
    _stage2_lds_bytes,
    p2p_scatter_epilog,
)
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
    # shared_in: the shared expert's MXFP8 weights and 32x32 scales (gate_up, down), the rank-local
    # activation buffers between its two GEMMs and its counters (compile_mega_moe_stage1). 0 in every other table.
    SH_GU_W = 39
    SH_GU_S = 40
    SH_DN_W = 41
    SH_DN_S = 42
    SH_ACT = 43
    SH_ACT_SCALE = 44
    SH_SYNC = 45
    # shared_in_opt bit 1: the first K halves' partials of the shared GEMM1 tiles (shared_expert.PARTIAL_UNIT_BYTES
    # a tile), which the workgroups of the second halves read.
    SH_PART = 46


DISPATCH_TABLE_SIZE = max(DispatchSlot) + 1
# shared_in: byte offsets in the SH_SYNC words (int32) -- an m-tile's GEMM1 unit count at 4 m, its ready epoch at
# _SH_READY + 4 m; the launch's GEMM2 unit count and the done epoch.
# shared_in_opt bit 1: the epoch of tile u's first-half partials at _SH_PART_READY + 4 u (u < _SH_PART_UNITS:
# the 2 m-tiles of 33..64 rows).
_SH_READY, _SH_G2_COUNT, _SH_DONE, _SH_PART_READY, _SH_PART_UNITS, _SH_SYNC_WORDS = (
    64,
    128,
    132,
    256,
    64,
    128,
)


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
    disp = ptr_buf_tensor(addr_disp, fx.Int64)
    in_idx = ptr_buf_tensor(addr_in_idx, fx.Int32)
    as_ptr = (
        (lambda v: fx.Int64(comm_ops.wave_uniform_i64(v)))
        if const_expr(uniform)
        else (lambda v: v)
    )

    def dp(i):
        return as_ptr(disp[fx.Int32(int(i))])

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
            in_idx[wk],
        )
    )
    valid_expert = (global_expert >= fx.Int32(0)) & (
        global_expert < fx.Int32(fz_total_experts)
    )
    destination = valid_expert.select(global_expert, fx.Int32(0)) // fx.Int32(fz_epr)
    remote = {
        name: as_ptr(ptr_buf_tensor(tables[name], fx.Int64)[destination])
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
            ej = in_idx[source_token * fx.Int32(fz_k) + fx.Int32(j)]
            same = (
                (ej >= fx.Int32(0))
                & (ej < fx.Int32(fz_total_experts))
                & (ej // fx.Int32(fz_epr) == destination)
            )
            duplicate = duplicate | ((fx.Int32(j) < topk_slot) & same)
        extra = {
            "duplicate": duplicate,
            "token_rows": dp(DispatchSlot.TOKEN_ROWS),
            "token_scales": dp(DispatchSlot.TOKEN_SCALES),
        }
    return {
        **extra,
        "tables": tables,
        "global_expert": global_expert,
        "remote": remote,
        "row_values": row_values,
        "scale": scale,
        "weight": weight,
    }


def _vgpr_i64(v):
    """``v`` through an empty inline asm with a VGPR result: the same value, but divergent to the compiler."""
    return fx.Int64(
        llvm.InlineAsmOp(
            T.i64, [fx.as_ir_value(fx.Int64(v))], "", "=v,0", has_side_effects=False
        ).result
    )


# fmt: off
@flyc.jit
def emit_direct_fixed_slot_payload(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_cap, fz_mtpr, fz_rank, fz_total_experts, fz_nbytes, fz_n_i32,
    fz_scale_n_i32, fz_enable_scales, addr_disp, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc,
    i32_cur_tok, dispatch_blocks, producer_slot, parity, expected, prefetched, route_rounds, prefetch_args,
    dedup=False, direct_done=0, running_half=None, late_epoch=None, active_done=False, disp_order=0,
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
    # disp_order bit 2: the destinations' source-flag pointers (one per lane) load before the routes, not behind
    # the done count.
    remote_done_pre = None
    if const_expr(disp_order & 2):
        remote_done_pre = load_at(AddrBuffer(p_source_done), (lane < fx.Int32(fz_npes)).select(lane, fx.Int32(0)),
            fx.Int64, 1)
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
            running_addr = rpf["remote"]["running"] + fx.Int64(running_slot) * fx.Int64(4)
            if const_expr(disp_order & 1):
                # A per-lane (VGPR) address keeps the RMW out of LLVM's atomic optimizer, whose readfirstlane of the
                # result would wait for the RMW right behind it, before the row stores below.
                running_addr = _vgpr_i64(running_addr)
            if lane == fx.Int32(0):
                if valid_expert:
                    offset_lane = fx.Int32(comm_ops.atomic_add_system(running_addr, fx.Int32(1)))
            if const_expr(dedup and (disp_order & 1)):
                # disp_order bit 1: the token row's place does not depend on the slot, so it goes out while the
                # slot RMW is in flight (also when the route overflows its expert; nothing reads that row then).
                if valid_expert & ~rpf["duplicate"]:
                    token_row = fx.Int64(fz_rank * fz_mtpr) + fx.Int64(source_token)
                    destination_rsrc = AddrBuffer(
                        rpf["remote"]["rx"] + rpf["token_rows"] + token_row * fx.Int64(fz_nbytes))
                    for i in range_constexpr(len(row_values)):
                        store_at(row_values[i], destination_rsrc, lane * fx.Int32(4) + fx.Int32(i * 256))
                    if const_expr(fz_enable_scales):
                        if lane < fx.Int32(fz_scale_n_i32):
                            store_at(rpf["scale"], AddrBuffer(rpf["remote"]["sc"] + rpf["token_scales"]
                                + token_row * fx.Int64(fz_scale_n_i32 * 4)), lane)
            expert_offset = fx.Int32(fx.rocdl.readlane(T.i32, offset_lane, 0))
            publish = valid_expert & (expert_offset < fx.Int32(fz_cap))
            payload_row = local_expert * fx.Int32(fz_cap) + expert_offset
            meta_row = payload_row
            if const_expr(running_half is not None):
                meta_row = meta_row + running_half * fx.Int32(fz_epr * fz_cap)

            if const_expr(dedup and not (disp_order & 1)):
                if publish & ~rpf["duplicate"]:
                    token_row = fx.Int64(fz_rank * fz_mtpr) + fx.Int64(source_token)
                    destination_row = ptr_buf_tensor(
                        rpf["remote"]["rx"] + rpf["token_rows"] + token_row * fx.Int64(fz_nbytes), fx.Int32,
                        unit_elems=4)
                    for i in range_constexpr(len(row_values)):
                        buf_copy_store(destination_row, lane + fx.Int32(i * 64), row_values[i], fx.Int32,
                            unit_elems=4)
                    if const_expr(fz_enable_scales):
                        if lane < fx.Int32(fz_scale_n_i32):
                            ptr_buf_tensor(rpf["remote"]["sc"] + rpf["token_scales"]
                                + token_row * fx.Int64(fz_scale_n_i32 * 4), fx.Int32)[lane] = rpf["scale"]
            if publish:
                if const_expr(not dedup):
                    destination_row = ptr_buf_tensor(
                        rpf["remote"]["rx"] + fx.Int64(payload_row) * fx.Int64(fz_nbytes), fx.Int32, unit_elems=4)
                    for i in range_constexpr(len(row_values)):
                        buf_copy_store(destination_row, lane + fx.Int32(i * 64), row_values[i], fx.Int32,
                            unit_elems=4)

                    if const_expr(fz_enable_scales):
                        if lane < fx.Int32(fz_scale_n_i32):
                            ptr_buf_tensor(rpf["remote"]["sc"], fx.Int32)[
                                payload_row * fx.Int32(fz_scale_n_i32) + lane] = rpf["scale"]

                if lane == fx.Int32(0):
                    weight_bits = fx.Vector.from_elements([rpf["weight"]], fx.Float32).bitcast(fx.Int32)[0]
                    source_encoding = (fx.Int32(fz_rank * fz_mtpr) + source_token) | (topk_slot << fx.Int32(24))
                    ptr_buf_tensor(rpf["remote"]["wts"], fx.Int32)[meta_row] = weight_bits
                    ptr_buf_tensor(rpf["remote"]["sm"], fx.Int32)[meta_row] = source_encoding

    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()
    if const_expr(late_epoch is not None):
        # M2 pg: the routes above went out before the epoch gate; the done flag needs this launch's parity.
        parity, expected = late_epoch()
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
                        if const_expr(not (disp_order & 4)):
                            comm_ops.fence_system_release()
                        share = (producer_slot == fx.Int32(0)).select(
                            fx.Int32(_DIRECT_DONE_INC + 1) - active, fx.Int32(1))
                        remote_done = remote_done_pre if const_expr(disp_order & 2) else ptr_buf_tensor(
                            p_source_done, fx.Int64)[lane]
                        done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
                        comm_ops.atomic_add_system(remote_done + fx.Int64(done_index) * fx.Int64(4), share)
        publish_last = publish_last & ~direct
        if const_expr(active_done):
            # M2 ad: only the producers that carry a route count in; the last of them publishes.
            publish_last = publish_last & (producer_slot < active)
    done_target = active - fx.Int32(1) if const_expr(active_done) else fx.Int32(dispatch_blocks - 1)
    # The last producer publishes; one lane per destination, so the remote release stores overlap.
    if publish_last:
        done_lane = fx.Int32(0)
        if lane == fx.Int32(0):
            # disp_order bit 4: no L2 writeback here. Every route store and RMW of this producer went to the
            # uncached symmetric heap and drained (the loop-end waitcnt) before this count.
            if const_expr(not (disp_order & 4)):
                comm_ops.fence_system_release()
            done_lane = fx.Int32(comm_ops.atomic_add_agent(a_producer_done, fx.Int32(1)))
        done = fx.Int32(fx.rocdl.readlane(T.i32, done_lane, 0))
        if done == done_target:
            comm_ops.fence_agent_acquire()
            if lane < fx.Int32(fz_npes):
                remote_done = remote_done_pre if const_expr(disp_order & 2) else load_at(
                    AddrBuffer(p_source_done), lane, fx.Int64, 1)
                done_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
                if const_expr(disp_order & 8):
                    # disp_order bit 8: the producers' rows and metadata sit drained in the uncached heap, so the
                    # source flag needs no L2 writeback before it either.
                    comm_ops.store_i32_global_system_monotonic(
                        fx.Int64(remote_done) + fx.Int64(done_index) * fx.Int64(4), expected)
                else:
                    comm_ops.store_i32_system(remote_done, done_index, expected)


# shared_in_opt bit 8: m-tiles whose ready epochs a GEMM2 claim's side read covers (one read each, one lane), and the
# byte offsets of the loop's LDS words (lean_tab): three results -- the plan CTA's flag is this launch's, the m-tile
# of the GEMM2 unit just claimed is ready, the shared expert's rows are complete -- then what the reads need: the
# launch epoch, the XCD's GEMM2 slice (first unit, unit count), the last m-tile a window may start at, and the
# addresses of the plan flag, of the ready epochs and of the shared expert's done epoch.
_LEAN_EPOCHS = 4
_LEAN_PLAN_SEEN, _LEAN_UNIT_READY, _LEAN_SHARED_DONE, _LEAN_EPOCH = 0, 4, 8, 12
_LEAN_BASE, _LEAN_UNITS, _LEAN_LIMIT = 16, 20, 24
_LEAN_PLAN_ADDR, _LEAN_READY_ADDR, _LEAN_DONE_ADDR, _LEAN_BYTES = 32, 40, 48, 64
# shared_in_opt bit 16: how far m-tile m's shared GEMM1 units start behind m-tile 0's, in s_sleep units (64 clocks) per
# m-tile: about one K-step of the tile body, so the lines the first reader of a weight slab fetched are in the XCD's L2.
_PAIR_STAGGER_UNITS = 48

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
        count = buf_copy_load(ptr_buf_tensor(a_running, fx.Int32), safe_expert, cache_modifier=count_cm)
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


def _plan_flag(no_overflow, total_tiles, hot_expert, routed_rows, fz_tile_m, stream_w=False):
    """Low 32 bits of the plan word: num_valid, plus the row-count bits of _PLAN_CACHED_W_BIT /
    _PLAN_FP8_ROWS_BIT (``stream_w``: the routed rows never pick the cached-weight bodies, only a hot expert does)."""
    nv_flag = no_overflow.select(total_tiles * fx.Int32(fz_tile_m), fx.Int32(0))
    cached_w = hot_expert if stream_w else hot_expert | (routed_rows >= fx.Int32(_CACHED_W_MIN_ROWS))
    nv_flag = nv_flag | cached_w.select(fx.Int32(1 << _PLAN_CACHED_W_BIT), fx.Int32(0))
    return nv_flag | (hot_expert | (routed_rows >= fx.Int32(_FP8_ROWS_MIN_ROWS))).select(
        fx.Int32(1 << _PLAN_FP8_ROWS_BIT), fx.Int32(0)
    )


@flyc.jit
def emit_direct_fixed_slot_finalize(
    *, fz_npes, fz_epr, fz_cap, fz_mtpr, fz_rank, fz_tile_m, addr_disp, parity, expected, plan_nv,
    plan_wt=False, keep_running=False, uniform=False, running_half=None, stream_w=False, lean_flag=False,
):
    """Finalize local fixed slots as soon as every source publishes this destination.

    ``plan_nv = (addr, epoch)``: the local plan is published as one i64 ``epoch << 32 | num_valid`` store, so
    consumers get num_valid from the flag itself. Only this rank's consumers read it. The srcmap of padding rows
    is left unwritten: GEMM2 masks rows past the tile's valid count itself.
    ``running_half``: the expert counters are two halves of fz_epr; this launch's routes counted into this one.
    """
    assert 0 < fz_epr <= 256, "direct fixed-slot finalize requires 1..256 experts per rank"
    disp = ptr_buf_tensor(addr_disp, fx.Int64)

    def dp(i):
        v = disp[fx.Int32(int(i))]
        if const_expr(uniform):
            v = fx.Int64(comm_ops.wave_uniform_i64(v))
        return v

    a_se = dp(DispatchSlot.SORTED_EXPERT)
    a_trb = dp(DispatchSlot.TILE_ROW_BASE)
    a_nv = dp(DispatchSlot.NUM_VALID)
    a_running = dp(DispatchSlot.RUNNING)
    if const_expr(running_half is not None):
        a_running = a_running + fx.Int64(running_half) * fx.Int64(fz_epr * 4)
    a_source_done = dp(DispatchSlot.COUNT_DONE)
    a_tile_rows = dp(DispatchSlot.TILE_ROWS)
    wt_cm = _SC0_SC1_CACHE if const_expr(plan_wt) else 0

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
        se_t, trb_t, nv_t, running_t, tile_rows_t = (
            ptr_buf_tensor(a, fx.Int32) for a in (a_se, a_trb, a_nv, a_running, a_tile_rows))

        for chunk in range_constexpr(len(chunks)):
            valid_expert, safe_expert, safe_count, num_expert_tiles, metadata_base = chunks[chunk]
            if valid_expert:
                if no_overflow:
                    global_expert = fx.Int32(fz_rank * fz_epr) + safe_expert
                    payload_base = safe_expert * fx.Int32(fz_cap)
                    for tile in range(fx.Int32(0), num_expert_tiles, 1):
                        metadata_index = metadata_base + tile
                        buf_copy_store(se_t, metadata_index, global_expert, cache_modifier=wt_cm)
                        buf_copy_store(trb_t, metadata_index, payload_base + tile * fx.Int32(fz_tile_m),
                            cache_modifier=wt_cm)
                        rows = safe_count - tile * fx.Int32(fz_tile_m)
                        rows = (rows < fx.Int32(fz_tile_m)).select(rows, fx.Int32(fz_tile_m))
                        buf_copy_store(tile_rows_t, metadata_index, rows, cache_modifier=wt_cm)
                if const_expr(not keep_running):
                    buf_copy_store(running_t, safe_expert, fx.Int32(0), cache_modifier=wt_cm)

        if lane == fx.Int32(0):
            num_valid = no_overflow.select(total_tiles * fx.Int32(fz_tile_m), fx.Int32(0))
            buf_copy_store(nv_t, fx.Int32(0), num_valid, cache_modifier=wt_cm)
            # num_valid[1] is a device-visible overflow status.
            buf_copy_store(nv_t, fx.Int32(1), overflow_count, cache_modifier=wt_cm)

        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        if const_expr(not plan_wt):
            comm_ops.fence_system_release()
        if lane == fx.Int32(0):
            nv_flag = _plan_flag(no_overflow, total_tiles, hot_expert, routed_rows, fz_tile_m, stream_w)
            plan_flag_value = (fx.Int64(plan_nv[1]) << fx.Int64(32)) | fx.Int64(nv_flag)
            if const_expr(lean_flag and plan_wt):
                # The plan went through to memory (plan_wt) and drained: the flag needs no L2 writeback.
                fx.generic_store(comm_ops._to_ptr_global(plan_nv[0], fx.Int64, 8), plan_flag_value,
                    memory_order=fx.AtomicOrdering.Monotonic, syncscope="one-as")
            else:
                comm_ops.store_i64_global_system(plan_nv[0], plan_flag_value)
    fx.barrier()


def _fc_out_addr(arg_out, arg_shared, shared_add):
    """Where the fused combine's reducers write a token's row: the caller's shared-output tensor (shared_add) or
    the combine buffer."""
    return arg_shared if shared_add else arg_out


def _fc_store_row(acc, out_rsrc, index, shared_add):
    """Store one lane's 8 summed F32 values of a token row as BF16 at i32 ``index`` of ``out_rsrc``. ``shared_add``:
    the row already holds the caller's BF16 values; the BF16-rounded sum is added to them and rounded again (the
    two roundings of ``bf16(sum) + shared`` on BF16 tensors)."""
    res = acc.to(fx.BFloat16)
    if shared_add:
        mine = fx.Vector(load_at(out_rsrc, index, fx.Int32, 4))
        res = (res.to(fx.Float32) + mine.bitcast(fx.BFloat16).to(fx.Float32)).to(fx.BFloat16)
    store_at(res.bitcast(fx.Int32).ir_value(), out_rsrc, index, 2)


def make_fused_combine_tail(*, lds_fc_off, reducers, npes, rank, topk, total_experts, experts_per_rank, N_OUT,
    flag_is_value=False, l1_acquire=False, row_format=False, reset_running=0, reset_tok=0, tok_done=False,
    n_blocks=1, bf16_head_tokens=0, m2_tail=frozenset(), lds_xt_off=0, shared_add=False, lean_publish=False):
    """Combine after this WG's P2P scatter: grid ticket, cross-device flag, reduce by the last WGs.

    ``shared_add``: ``arg_shared`` is the address of a caller-owned BF16 ``[cur_tok, N_OUT]`` tensor (the shared
    expert's output); the reducers write ``bf16(f32(bf16(sum)) + f32(shared))`` back into it instead of writing
    ``bf16(sum)`` to ``arg_out`` -- bit for bit the caller's ``combine_output + shared`` (two BF16 roundings), with
    the BF16 add launch gone. The tensor is the call's output; ``arg_out`` is left untouched.

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
        arg_tok_off: fx.Int64, arg_tok_reset: fx.Int64, arg_shared: fx.Int64):
    # fmt: on
        ticket_ptr = lds_typed_ptr(fx.Int32(lds_fc_off), T.i32, align=4)
        flag_ptr = lds_typed_ptr(fx.Int32(lds_fc_off + 8), T.i64, align=8)
        if const_expr("xt" in m2_tail):
            # M2 xt: "this WG is the launch's last" (the +4 word of the fc scratch is otherwise unused).
            xt_last_ptr = lds_typed_ptr(fx.Int32(lds_fc_off + 4), T.i32, align=4)

        def fmt_ptr(peer):
            # One word per peer: its format bit, already shifted to the peer's position.
            return lds_typed_ptr(fx.Int32(lds_fc_off + 16) + peer * fx.Int32(4), T.i32, align=4)

        last = i32_grid - fx.Int32(1)
        if const_expr("tl" in m2_tail):
            # M2 tl: the publisher's peer pointers, loaded under the store drain instead of after the ticket.
            remote_pre = load_at(AddrBuffer(arg_p2p_xdb), (tx_i32 < fx.Int32(npes)).select(tx_i32, fx.Int32(0)),
                fx.Int64, 1)
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
            if const_expr("xt" in m2_tail):
                # M2 xt: exit tickets per XCD (word 4 of the XCD's work-head line, zeroed by the owner at entry; the
                # grid holds i32_grid / 8 WGs per XCD, WG b on XCD b & 7), so 8 short atomic queues replace one of
                # i32_grid. The last WG of each XCD then counts in on arg_cnt; the 8th of those is the launch's last.
                # Reducers: the last reducers / 8 WGs of every XCD, with the same dense worker numbering.
                xt_xcd = fx.Int32(comm_ops.hw_xcc_id()) & fx.Int32(7)
                xt_base = fx.Int64(fx.ptr_load(lds_typed_ptr(fx.Int32(lds_xt_off), T.i64, align=8)))
                xt_per = i32_grid // fx.Int32(8)
                xt_local = fx.Int32(comm_ops.atomic_add_agent(
                    xt_base + fx.Int64(xt_xcd) * fx.Int64(64) + fx.Int64(16), fx.Int32(1)))
                xt_is_last = fx.Int32(0)
                if xt_local == xt_per - fx.Int32(1):
                    xt_glob = fx.Int32(comm_ops.atomic_add_agent(arg_cnt, fx.Int32(1)))
                    xt_is_last = (xt_glob == fx.Int32(7)).select(fx.Int32(1), fx.Int32(0))
                xt_red0 = xt_per - fx.Int32(reducers // 8)
                ticket_lane = (xt_local >= xt_red0).select(
                    i32_grid - fx.Int32(reducers) + (xt_local - xt_red0) * fx.Int32(8) + xt_xcd, fx.Int32(0))
                fx.ptr_store(xt_is_last, xt_last_ptr)
            else:
                ticket_lane = fx.Int32(comm_ops.atomic_add_agent(arg_cnt, fx.Int32(1)))
            fx.ptr_store(ticket_lane, ticket_ptr)
            fx.ptr_store(flag_lane, flag_ptr)
        fx.barrier()
        ticket = rocdl.readfirstlane(T.i32, fx.ptr_load(ticket_ptr).ir_value())
        flag = fx.ptr_load(flag_ptr)
        if const_expr("xt" in m2_tail):
            am_last = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(xt_last_ptr).ir_value())) != fx.Int32(0)
        else:
            am_last = fx.Int32(ticket) == last
        if am_last:
            if const_expr(reset_running > 0):
                if tx_i32 < fx.Int32(reset_running):
                    buf_copy_store(ptr_buf_tensor(arg_running, fx.Int32), tx_i32, fx.Int32(0),
                        cache_modifier=_SC0_SC1_CACHE)
                if const_expr("tl" not in m2_tail):
                    # (M2 tl: this half of the counters is next counted into two launches from now, and these
                    # stores complete before this kernel ends: the flag need not wait for them.)
                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    fx.barrier()
            if tx_i32 < fx.Int32(npes):
                # Every other WG's stores completed before its ticket; acquire them, then publish to peer tx.
                # lean_publish: those stores (P2P rows, counter resets) went to the uncached symmetric heap or
                # through L2 (sc0 sc1) and drained before the tickets, so the flag goes out without the L2
                # invalidate / writeback, as a relaxed system store.
                if const_expr(not lean_publish):
                    comm_ops.fence_system_acquire()
                    comm_ops.fence_system_release()
                if const_expr("tl" in m2_tail):
                    remote = remote_pre
                else:
                    remote = load_at(AddrBuffer(arg_p2p_xdb), tx_i32, fx.Int64, 1)
                if const_expr(lean_publish):
                    fx.generic_store(comm_ops._to_ptr_global(fx.Int64(remote) + fx.Int64(rank * 8), fx.Int64, 8),
                        fx.Int64(flag), memory_order=fx.AtomicOrdering.Monotonic, syncscope="one-as")
                else:
                    comm_ops.store_i64_global_system(fx.Int64(remote) + fx.Int64(rank * 8), flag)
            fx.barrier()
            if tx_i32 == fx.Int32(0):
                if const_expr(not flag_is_value):
                    comm_ops.store_i64_global(arg_flag, flag + fx.Int64(1))
                store_at(fx.Int32(0), AddrBuffer(arg_cnt), fx.Int32(0))
                store_at(fx.Int32(0), AddrBuffer(arg_trecv), fx.Int32(0))
            if const_expr(reset_tok > 0):
                tok_reset = ptr_buf_tensor(arg_inp + arg_tok_reset, fx.Int32)
                for i in range_constexpr((reset_tok + 255) // 256):
                    if tx_i32 + fx.Int32(i * 256) < fx.Int32(reset_tok):
                        buf_copy_store(tok_reset, tx_i32 + fx.Int32(i * 256), fx.Int32(0),
                            cache_modifier=_SC0_SC1_CACHE)
        nred = (i32_grid < fx.Int32(reducers)).select(i32_grid, fx.Int32(reducers))
        first = i32_grid - nred
        if fx.Int32(ticket) >= first:
            # tok_done: each reducer wave waits for its own token below instead.
            if const_expr(not tok_done):
                if tx_i32 < fx.Int32(npes):
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
            out_rsrc = AddrBuffer(_fc_out_addr(arg_out, arg_shared, shared_add))
            items = i32_cur_tok * fx.Int32(chunks)
            worker = (fx.Int32(ticket) - first) * fx.Int32(4) + wave
            if const_expr("nc" in m2_tail):
                items = fx.Int32(0)
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
                    _fc_store_row(acc, out_rsrc, tok * fx.Int32(row_i32) + col, shared_add)
                    if const_expr(bf16_head_tokens):
                        if tok < fx.Int32(bf16_head_tokens):
                            for k in range_constexpr(topk):
                                store_at(Vec.filled(4, _ROW_POISON, fx.Int32), inp_rsrc,
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
                    _fc_store_row(acc, out_rsrc, tok * fx.Int32(row_i32) + col, shared_add)

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
    p2p_quant_type="none", uniform=False, epi_lean=0, tok_done=False, bf16_head_tokens=0, g2_b2stage=False,
    rg0_only=False):
# fmt: on
    """One stage2 tile (m_block, n_block) as a jit helper, for kernels that run GEMM2 after their own phase.
    ``rg0_only``: gemm2_compute_v2's, for tiles of at most 16 valid rows.

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
                fx.Int64(comm_ops.wave_uniform_i64(a)) for a in (arg_aq, arg_ascale, arg_bq, arg_bscale, arg_eids,
                    arg_stids, arg_sweights, arg_trb, arg_tile_rows))
            i32_max_m_blocks = fx.Int32(rocdl.readfirstlane(T.i32, i32_max_m_blocks))
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
            expert_offset=expert_offset, valid_rows=vrows, **({"g2_b2stage": True} if g2_b2stage else {}),
            **({"rg0_only": True} if rg0_only else {}))
        p2p_scatter_epilog(lds_base_i32, accm_vecs, n_block_idx, wave, lane, N_OUT=N_OUT, BM=BM, BN=BN, npes=npes,
            topk=topk, log2_max_tok=log2_max_tok, mask_max_tok=max_tok - 1, recv_cap=recv_cap,
            comb_inp_nbytes=comb_inp_nbytes, meta=Stage2Meta(lds_base_i32, lay["packed"], lay["weight"], lay["peer"]),
            g2_bf16_lds=False, p2p_quant_type=p2p_quant_type, store_cm=store_cm,
            fp8_store_cm=2, fp8_rows=i32_fp8_rows, elem_view_stores=True, lean=epi_lean,
            tok_off=i64_tok_off if const_expr(tok_done) else None, bf16_head_tokens=bf16_head_tokens)
        # fmt: on

    return g2_tile, lay


def _use_direct_fixed_slot(npes, experts_per_rank, max_tokens_per_rank, cap, tile_m):
    if tile_m <= 0 or max_tokens_per_rank <= 0:
        return False
    required_cap = ((npes * max_tokens_per_rank + tile_m - 1) // tile_m) * tile_m
    layouts = ((8, 48), (4, 96), (2, 192)) + EXTRA_FIXED_SLOT_LAYOUTS  # the latter empty unless switched on
    return (npes, experts_per_rank) in layouts and cap == required_cap


# fmt: off
@aot_factory("mega_moe_small_m_stage1")
@functools.cache
def compile_mega_moe_stage1(
    *, model_dim: int, inter_dim: int, rank: int, experts_per_rank: int, fuse_npes: int, fuse_topk: int,
    fuse_cap: int, fuse_mtpr: int, fuse_scale_dim: int, fused: tuple, sort_block_m: int = 32,
    tile_n: int = 256, tile_k: int = 256, num_waves: int = 4, grid_mult: int = 8,
    pipe_weights: bool = True, mfma_amajor: bool = False, swizzle_a: bool = True,
    async_a_copy: bool = False, use_tile_resource: bool = True,
    waves_per_eu_hint: int = 2, num_cu: int = 256, num_dispatch_cu: int = 32, b_nt: int = -1,
    swiglu_limit: float = 0.0, dedup_rows: bool = False, plan_wt: bool = False,
    plan_local: bool = False, g2_early: tuple = (0, 0), g2_uniform: bool = False, head_uniform: bool = False,
    g2_epi_lean: int = 0, g1_lookahead: int = 1, g1_a_ahead: int = 1, direct_done: int = 0, tok_done: bool = False,
    tok_counters: bool = False, bf16_head_tokens: int = 0, epi_evec: int = 2, epi_lean: int = 0,
    ascale_packed: bool = False, stream_w: bool = False, g2_b2stage: bool = False, probe: str = "",
    expert_cached_w: int = 0, m2: str = "", shared_add: bool = False, disp_order: int = 0, sync_lean: int = 0,
    shared_in: int = 0, shared_in_opt: int = 0,
):
    # ``shared_in`` (default 0 = unchanged kernel; 1 / 2 = the K split of its GEMM1, see
    # shared_expert.dense_gate_up_ksplit): the kernel also computes the model's shared expert for this rank's rows
    # and its combine adds the routed sum to it (``shared_add``'s combine; ``addr_in_sc`` is the BF16 output, which
    # the kernel now writes itself). The work needs nothing from the launch protocol, so it runs as a pre-phase: the
    # first consumer workgroups (by entry ticket) each take one GEMM1 unit (32 rows x 128 activation columns) and
    # then one GEMM2 unit (32 rows x 256 output columns) before they join the routed work pool, while the grid
    # waits for the sources and the plan anyway. The host launches this kernel only for row counts with a consumer
    # per unit (shared_in_fits): a unit without a workgroup would leave its counter short and the launch waiting. Inputs: the MXFP8 pair of this call's rows through the kernel's
    # ``sorted_token_ids`` (E4M3 rows) and ``expert_ids`` (their row-major 1x32 E8M0 scales) arguments, which a
    # plan_local kernel does not read; the shared weights, the rank-local activation buffers and the counters
    # through the dispatch table (DispatchSlot.SH_*). Numerics: ops/mega_moe/shared_expert.py, the dense launches'.
    # Rank-local: no protocol word changes, ranks need not agree. Synchronization (SH_SYNC, int32): [m] counts an
    # m-tile's finished GEMM1 units and [16 + m] takes the launch epoch once all are in (each count rewinds itself);
    # [32] counts finished GEMM2 units and [33] takes the epoch once all are in. Every workgroup reads [33] before
    # its exit ticket, so the reducers add to finished rows. All of it is local to the rank and never waits on a
    # peer; a rank without rows has no units and skips the wait.
    # ``shared_in_opt`` (default 0 = the shared_in kernel as it was; bits): how the pre-phase is run.
    #   1  (shared_in == 2 only) the two K halves of a GEMM1 unit run on two consumer workgroups: consumer c takes
    #      half c & 1 of unit c >> 1. The first-half workgroup leaves its accumulators in memory (SH_PART) and raises
    #      the unit's epoch; the second-half workgroup waits for it after its own K steps, sums 0 + p0 + p1 and runs
    #      the epilogue. Same numbers as the one-workgroup tile (the same two half sums, the same merge); the GEMM1
    #      chain is half as long, so more of the pre-phase lies inside the launch's sync window. Needs a consumer
    #      workgroup per half (shared_in_fits_split).
    #   2  the pre-phase starts at entry instead of after the rank's gate: it reads nothing the gate orders (its
    #      inputs and counters are the rank's own and the previous launch on this rank has ended). Consumers wait
    #      for the gate after it.
    #   4  (not with bit 1) the units of one weight slab on one XCD: consumer c takes column tile 8 q + (c & 7) of
    #      m-tile (c >> 3) % tiles, q = (c >> 3) // tiles (shared_pair_unit), instead of unit c in m-major order. The
    #      m-tiles' units of a column tile then sit 8 workgroups apart -- with static roles (M2 st) on the same XCD
    #      -- and read the same weight lines through one L2 (the tile bodies load the shared weights through the
    #      cache), where m-major order puts them 18 / 20 workgroups = 2 / 4 XCDs apart and every XCD streams the
    #      slab itself. The same units, counters and epochs: only which workgroup runs which unit. Needs a consumer
    #      for the highest index (shared_in_fits_pair).
    #  16  (with bit 4) the GEMM1 units of m-tile m start m x _PAIR_STAGGER_UNITS x 64 clocks behind those of m-tile
    #      0, so a slab's later readers find its lines in the L2 instead of asking for them at the same moment as
    #      the first one. The GEMM2 units follow their m-tile's ready epoch and need no delay of their own. The
    #      pre-phase ends that much later; nothing waits for it before the exit tickets.
    #   8  the routed unit loop without the round trips that only observe: a GEMM1 claim and the finished tile's
    #      count go out together (through per-lane addresses, past LLVM's atomic optimizer, whose readfirstlane
    #      waits behind each one); the plan CTA's flag is read next to every GEMM1 claim and the GEMM2 loop spins on
    #      it only if the last read did not show this launch's; the ready epochs of the next _LEAN_EPOCHS m-tiles
    #      of the XCD's slice are read next to every GEMM2 claim and the unit spins on its m-tile's only if that
    #      read did not show it (a CTA's first GEMM2 unit, claimed by a GEMM1 fetch, waits as before); the shared
    #      expert's done epoch is read next to every claim too and the exit spins only if needed. Every wait is
    #      satisfied by the flag it was satisfied by before, read one round trip earlier. For kernels without an
    #      early GEMM2 start (g2_early) and without per-XCD exit tickets (M2 xt).
    assert shared_in in (0, 1, 2), shared_in
    assert 0 <= shared_in_opt <= 31 and (not shared_in_opt or shared_in), (shared_in, shared_in_opt)
    assert not (shared_in_opt & 1) or shared_in == 2, "shared_in_opt bit 1 is for the split-K variant"
    assert not (shared_in_opt & 4 and shared_in_opt & 1), "shared_in_opt bit 4 is for whole-unit workgroups"
    assert not (shared_in_opt & 16) or shared_in_opt & 4, "shared_in_opt bit 16 staggers the pairs of bit 4"
    sh_split_wg = bool(shared_in_opt & 1)
    sh_pre_gate = bool(shared_in_opt & 2)
    sh_pair = bool(shared_in_opt & 4)
    sh_lean = bool(shared_in_opt & 8)
    sh_stagger = bool(shared_in_opt & 16)
    assert not shared_in or (shared_add and plan_local and not tok_done), "shared_in needs shared_add and plan_local"
    # ``shared_add`` (default False = unchanged kernel): the launch's ``addr_in_sc`` argument (otherwise
    # unused) is the address of the caller's BF16 shared-expert output; the combine adds each token's row to it in
    # place (make_fused_combine_tail), so the caller's separate BF16 add launch goes away. Local to the rank: no
    # protocol word changes, ranks need not agree on it.
    # Experiment knobs (default off, see _kernel_knobs): ``g2_b2stage`` runs the 64-row GEMM2 tile on the 2-stage B
    # pipeline (next K-tile's W2 prefetched during this one's MFMAs) instead of the 1-stage path; ``probe`` drops
    # GEMM bodies for phase timing ("g1": no GEMM1 tile body, "g2": no GEMM2 tile body; the protocol, claims,
    # ready epochs and combine still run, the output is garbage).
    # ``expert_cached_w`` (M1-a, default 0 = unchanged kernel): an expert whose rows span more than one m-tile reads its
    # weights through the cache (the cached-weight bodies the hot ranks already use), so its second and later m-tiles
    # take the W slab from L2 instead of streaming it from HBM again; single-tile experts keep the streaming bodies.
    # 1: GEMM1 (W1) only, 2: GEMM1 and GEMM2 (W1 and W2). Only the cache policy of the weight loads changes.
    assert expert_cached_w in (0, 1, 2), expert_cached_w
    # ``disp_order`` (FusedStage1Knobs.disp_order, default 0 = unchanged kernel): order of the producers' stores, see
    # emit_direct_fixed_slot_payload. The same stores and flags, so the output is bit-identical and ranks need not
    # agree.
    assert 0 <= disp_order <= 31, disp_order
    # ``sync_lean`` (FusedStage1Knobs.sync_lean, default 0 = unchanged kernel): bit 1 GEMM2-ready epochs, bit 2 the plan
    # flag (with plan_wt) as relaxed stores without an L2 writeback. Bit-identical; ranks need not agree.
    assert 0 <= sync_lean <= 3, sync_lean
    probe_skip = frozenset(p for p in probe.split("+") if p)
    # "null" (benchmarks only): the launch runs an empty kernel of the same grid, block and arguments -- the
    # launch floor of the bench graph. It touches no protocol state, so launches of other variants stay in step.
    # "hm" / "hm2" (upper-bound probes, wrong output for tiles of more than 16 rows): every GEMM1 / GEMM2
    # tile runs only the MFMAs of its first 16-row group.
    assert probe_skip <= {"g1", "g2", "null", "hm", "hm2"}, probe
    # M2 fixed-base prototype switches (AITER_MEGA_SMALL_M2, '+'-joined, default off; every rank must set the same):
    #   st  static roles: owner / producers / consumers by block index, the launch epoch read from this rank's own
    #       combine flag of the previous launch instead of one entry-ticket atomic per CTA (the owner still advances
    #       the shared entry counter by a whole launch so launches of other variants stay in step)
    #   pg  producers send their routes before the epoch gate; they wait for the gate only before the done flag
    #   ad  the producer-done chain counts only producers that carry a route
    #   tl  combine tail: the peers' flag pointers are loaded before the exit ticket and the publisher does not wait
    #       for its counter-zeroing stores (double-buffered halves)
    #   xt  exit tickets per XCD (8 queues of grid / 8 WGs) instead of one agent-scope atomic queue of the whole grid
    #   nw / nc  probes (wrong output): no GEMM work units after the real dispatch / no combine reduce pass
    m2f = frozenset(p for p in m2.split("+") if p)
    assert m2f <= {"st", "pg", "ad", "tl", "xt", "nw", "nc"}, m2
    assert "ad" not in m2f or direct_done, "ad needs the direct_done flag arithmetic"
    assert "tl" not in m2f or (tok_counters and plan_local), "tl needs the double-buffered expert counters"
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
        raise ValueError("small-input stage1 needs the EP8/48-, EP4/96- or EP2/192-expert fixed-slot layout")
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
        epi_lean=g2_epi_lean, tok_done=tok_done, bf16_head_tokens=bf16_head_tokens, g2_b2stage=g2_b2stage,
        **({"rg0_only": True} if "hm2" in probe_skip else {}),
    )
    g2_tile_cached, _ = make_g2_tile_runner(
        BM=sort_block_m, BN=g2_bn, BK=g2_bk, SBM=sort_block_m, use_nt=False, INTER_MAX=inter_dim,
        a_dtype="fp8", g2_bhoist=True, g2_ascale_pf=True, expert_offset=int(rank) * int(experts_per_rank),
        npes=int(fuse_npes), topk=int(fuse_topk), N_OUT=model_dim, max_tok=int(fuse_mtpr),
        recv_cap=int(g2_recv_cap), comb_inp_nbytes=int(fuse_mtpr) * int(fuse_topk) * model_dim * 2,
        clamp_a=True, store_cm=0, row_mask=True, p2p_quant_type="dynamic", uniform=g2_uniform,
        epi_lean=g2_epi_lean, tok_done=tok_done, bf16_head_tokens=bf16_head_tokens, g2_b2stage=g2_b2stage,
        **({"rg0_only": True} if "hm2" in probe_skip else {}),
    )
    # The combine's last WG zeroes the expert counters one per thread.
    assert not plan_local or fz_epr <= TOTAL_THREADS, "plan_local resets at most one expert counter per thread"
    # The combine input is in the uncached shmem heap: the reducers only need to drop their L1.
    fused_combine_tail = make_fused_combine_tail(lds_fc_off=g2_lay["fc"], reducers=fused_reducers,
        npes=int(fuse_npes), rank=int(rank), topk=int(fuse_topk), total_experts=fz_total_experts,
        experts_per_rank=fz_epr, N_OUT=model_dim, flag_is_value=True, l1_acquire=True, row_format=True,
        reset_running=fz_epr if plan_local else 0, reset_tok=fz_mtpr if tok_counters else 0, tok_done=tok_done,
        n_blocks=g2_n_blocks, bf16_head_tokens=bf16_head_tokens, m2_tail=m2f & {"tl", "nc", "xt"}, lean_publish=bool(disp_order & 16),
        lds_xt_off=g2_lay["bytes"] + 8, shared_add=bool(shared_add))
    g2_scratch_off = g2_lay["bytes"]
    lds_pool_bytes = max(lds_pool_bytes, (g2_scratch_off + 16 + 15) // 16 * 16)
    # (M2 xt keeps the work-head address in the 8 bytes after the two scratch words.)
    xt_lds_off = g2_scratch_off + 8
    assert "xt" not in m2f or xt_lds_off % 8 == 0
    # The combine reducers spin until every CTA has taken its ticket, so no CTA may still be waiting for a
    # slot: the whole grid must be co-resident (2 CTAs per CU: <= 256 VGPRs and half the 160 KiB LDS).
    assert grid_mult <= 2 and 2 * lds_pool_bytes <= 160 * 1024, (grid_mult, lds_pool_bytes)
    if shared_in:
        # Imported here only: no kernel without shared_in needs the module (or the AITER helpers it imports).
        from ..tensor_shim import buf_base_i64 as _base_i64
        from . import shared_expert as _se

        plan_chunks_bytes = ((fz_epr + 63) // 64) * 64 * 8
        # The tile bodies are shared_expert's own (32 rows x 256 columns x 256 K, 4 waves), whatever this kernel's
        # routed tiles are. The host launches this kernel only for row counts whose units all find a consumer
        # workgroup (shared_in_fits); here: at least one m-tile's worth.
        sh_n1, sh_n2 = (2 * inter_dim) // 256, model_dim // 256
        # The routed tiles may be 32 or 64 rows (the 256 / 512 buckets): the pre-phase ends before the routed work
        # keeps anything in LDS, so only the budget below ties the two.
        assert NUM_WAVES == 4 and sort_block_m in (32, 64) and swiglu_limit > 0 and fz_mtpr <= 16 * 32
        assert max(sh_n1, sh_n2) <= launch_grid_x - 1 - dispatch_blocks, (
            "shared_in: too few consumer workgroups", launch_grid_x, dispatch_blocks)
        # Pool: the shared GEMM2 epilogue's 32 x 256 fp32 tile (the pre-phase ends before the kernel keeps any
        # state in the pool); A scales: the shared GEMM1's 32 x model_dim / 32 bytes and the tile-layout staging.
        lds_pool_bytes = max(lds_pool_bytes, 32 * 256 * 4)
        sh_partial_bytes = NUM_WAVES * 64 * _se.M_REPEAT * _se.NUM_ACC_N * 16 if shared_in == 2 and not sh_split_wg else 0
        if sh_pair:
            # The pair map's highest consumer of the largest call this kernel serves must exist (the host checks
            # the call's own row count: shared_in_fits_pair); here: one m-tile's worth.
            assert shared_pair_consumers(1, max(sh_n1, sh_n2)) <= launch_grid_x - 1 - dispatch_blocks, (
                "shared_in_opt 4: too few consumer workgroups", launch_grid_x, dispatch_blocks)
        if sh_split_wg:
            # 33..64 rows: 2 m-tiles, a consumer workgroup per K half of each GEMM1 unit.
            assert 2 * 2 * sh_n1 <= launch_grid_x - 1 - dispatch_blocks and 2 * sh_n1 <= _SH_PART_UNITS, (
                "shared_in_opt 1: too few consumer workgroups", launch_grid_x, dispatch_blocks)
        assert n_scale_bytes >= max(32 * (model_dim // 32), 3 * 4 * NUM_WAVES * 64)
        # Two workgroups a CU stay resident (see above): this kernel's whole LDS, not only its pool.
        assert 2 * (lds_pool_bytes + n_scale_bytes + 16 + plan_chunks_bytes + sh_partial_bytes
            + (_LEAN_BYTES if sh_lean else 0)) <= 160 * 1024
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
    assert "xt" not in m2f or (launch_grid_x % 8 == 0 and fused_reducers % 8 == 0 and fused_reducers <= launch_grid_x)

    plan_chunks = (fz_epr + 63) // 64
    # g2_early = (K, P): one CTA in K (by in-XCD workgroup index) leaves GEMM1 for GEMM2 after a unit claimed past
    # P/16 of its XCD's GEMM1 slice, so the GEMM2 scatter uses the XGMI links the finished dispatch left idle while
    # GEMM1 still runs. GEMM2 units then take their own claim word (word 3 of the XCD's line) instead of following
    # the GEMM1 units on the GEMM1 counter.
    G2_EARLY_K, G2_EARLY_P = (int(v) for v in g2_early)
    assert G2_EARLY_K == 0 or (G2_EARLY_K >= 2 and 0 < G2_EARLY_P < 16)
    assert not sh_lean or (G2_EARLY_K == 0 and "xt" not in m2f and plan_local), "shared_in_opt 8: see its comment"

    @fx.struct
    class SharedStorage:
        pool: fx.Array[fx.Int8, lds_pool_bytes, 16]
        A_scale: fx.Array[fx.Int8, n_scale_bytes, 16]
        # Work-unit broadcast word, kept out of the pool: a fast wave's next A / C-shuffle store must not race the
        # slower waves' read of the unit it starts.
        work_word: fx.Array[fx.Int8, 16, 16]
        if plan_local:
            # Per expert slot (64-expert chunks): its inclusive tile count and its row count.
            plan_incl: fx.Array[fx.Int32, plan_chunks * 64, 16]
            plan_count: fx.Array[fx.Int32, plan_chunks * 64, 16]
        if shared_in == 2 and not sh_split_wg:
            # shared_in: the first K half's partials of a split GEMM1 tile (16 bytes per accumulator and thread).
            sh_partial: fx.Array[fx.Int8, sh_partial_bytes, 16]
        if sh_lean:
            # shared_in_opt bit 8: what tid 0's side reads showed and what they need (_LEAN_*).
            lean_tab: fx.Array[fx.Int8, _LEAN_BYTES, 16]

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
        f"{'_pw' if plan_wt else ''}"
        f"{'_pl' if plan_local else ''}"
        f"{'_g2u' if g2_uniform else ''}{f'_g2l{g2_epi_lean}' if g2_epi_lean else ''}"
        f"{'_hu' if head_uniform else ''}{f'_la{g1_lookahead}' if g1_lookahead != 1 else ''}"
        f"{f'_ga{g1_a_ahead}' if g1_a_ahead != 1 else ''}{f'_dn{direct_done}' if direct_done else ''}"
        f"{'_td' if tok_done else ''}"
        f"{f'_ev{epi_evec}' if epi_evec != 2 else ''}{f'_el{epi_lean}' if epi_lean else ''}"
        f"{'_ap' if ascale_packed else ''}"
        f"{f'_ek{G2_EARLY_K}p{G2_EARLY_P}' if G2_EARLY_K else ''}"
        f"{'_stw' if stream_w else ''}"
        f"{'_g2b2' if g2_b2stage else ''}{'_probe' + ''.join(sorted(probe_skip)) if probe_skip else ''}"
        f"{'_m2' + ''.join(sorted(m2f)) if m2f else ''}"
        f"{f'_ecw{expert_cached_w}' if expert_cached_w else ''}"
        f"{'_sha' if shared_add else ''}"
        f"{f'_shi{shared_in}' if shared_in else ''}"
        f"{f'_sho{shared_in_opt}' if shared_in_opt else ''}"
        f"{f'_do{disp_order}' if disp_order else ''}"
        f"{f'_snc{sync_lean}' if sync_lean else ''}"
        f"{'' if _FP8_ROWS_MIN_ROWS == _FP8_ROWS_MIN_DEFAULT else f'_f8r{_FP8_ROWS_MIN_ROWS}'}"
    )

    if "null" in probe_skip:
        @flyc.kernel(name=kernel_name, known_block_size=[TOTAL_THREADS, 1, 1])
        def kernel_null(
            out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
            sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor,
            tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
            addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
            addr_expected: fx.Int64,
        ):
            fx.barrier()

        @flyc.jit
        def launch_null(
            out: fx.Tensor, x: fx.Tensor, w: fx.Tensor, scale_x: fx.Tensor, scale_w: fx.Tensor,
            sorted_token_ids: fx.Tensor, expert_ids: fx.Tensor, out_scale: fx.Tensor,
            tokens: fx.Int32, addr_disp: fx.Int64, i32_cur_tok: fx.Int32, addr_in_tok: fx.Int64,
            addr_in_idx: fx.Int64, addr_in_wts: fx.Int64, addr_in_sc: fx.Int64, addr_parity: fx.Int64,
            addr_expected: fx.Int64, stream: fx.Stream,
        ):
            kernel_null(
                out, x, w, scale_x, scale_w, sorted_token_ids, expert_ids, out_scale, tokens,
                addr_disp, i32_cur_tok, addr_in_tok, addr_in_idx, addr_in_wts, addr_in_sc, addr_parity, addr_expected,
                value_attrs={
                    "rocdl.waves_per_eu": waves_per_eu_hint,
                    "rocdl.flat_work_group_size": f"{TOTAL_THREADS},{TOTAL_THREADS}",
                },
            ).launch(grid=(launch_grid_x, 1, 1), block=(TOTAL_THREADS, 1, 1), stream=stream)

        return launch_null

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
        disp_t = ptr_buf_tensor(addr_disp, fx.Int64)
        parity_t = ptr_buf_tensor(addr_parity, fx.Int32)
        expected_t = ptr_buf_tensor(addr_expected, fx.Int32)

        def _disp_ptr(slot):
            v = disp_t[fx.Int32(int(slot))]
            if const_expr(head_uniform):
                v = fx.Int64(comm_ops.wave_uniform_i64(v))
            return v

        a_entry_count = _disp_ptr(DispatchSlot.ENTRY_COUNT)
        a_epoch_gate = _disp_ptr(DispatchSlot.EPOCH_GATE)
        a_work_head = _disp_ptr(DispatchSlot.WORK_HEAD)
        a_group_done = _disp_ptr(DispatchSlot.GROUP_DONE)
        if const_expr(dedup_rows):
            token_offsets = (_disp_ptr(DispatchSlot.TOKEN_ROWS), _disp_ptr(DispatchSlot.TOKEN_SCALES))

        ticket_scratch = fx.recast_iter(fx.Int64, a_buf.ptr)
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        if const_expr("st" in m2f):
            # Static roles. The previous launch's last WG left (epoch + 2^40) << 1 | fmt in this rank's own combine
            # flag slot; it is stable until this launch's own last WG, so every CTA reads the same epoch.
            if tid == fx.Int32(0):
                blk = fx.Int32(fx.block_idx.x)
                if blk == fx.Int32(0):
                    comm_ops.atomic_add_agent(a_entry_count + fx.Int64(entry_slot * 8), fx.Int64(entry_unit))
                prev_flag = fx.Int64(comm_ops.load_i64_acquire(
                    _disp_ptr(DispatchSlot.FC_XDB) + fx.Int64(fz_rank * 8)))
                prev_epoch = (prev_flag >> fx.Int64(1)) & fx.Int64(0xFFFFFFFF)
                ticket64 = prev_epoch * fx.Int64(entry_unit) + fx.Int64(blk) * fx.Int64(entry_inc)
                fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        else:
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
            parity = buf_copy_load(parity_t, fx.Int32(0), cache_modifier=_SC0_SC1_CACHE)
            expected0 = buf_copy_load(expected_t, fx.Int32(0), cache_modifier=_SC0_SC1_CACHE)
            expected1 = buf_copy_load(expected_t, fx.Int32(1), cache_modifier=_SC0_SC1_CACHE)
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
                work_head_t = ptr_buf_tensor(a_work_head, fx.Int32)
                if tid < fx.Int32(8):
                    # The XCD's work counter and its counts of its own GEMM1 units in the m-tiles cut at its
                    # slice ends.
                    work_head_t[tid * fx.Int32(16)] = fx.Int32(0)
                    work_head_t[tid * fx.Int32(16) + fx.Int32(1)] = fx.Int32(0)
                    work_head_t[tid * fx.Int32(16) + fx.Int32(2)] = fx.Int32(0)
                    if const_expr(G2_EARLY_K > 0):
                        work_head_t[tid * fx.Int32(16) + fx.Int32(3)] = fx.Int32(0)
                    if const_expr("xt" in m2f):
                        work_head_t[tid * fx.Int32(16) + fx.Int32(4)] = fx.Int32(0)
                if tid == fx.Int32(16):
                    ptr_buf_tensor(a_group_done, fx.Int32)[fx.Int32(0)] = fx.Int32(0)
                old_parity = parity_t[fx.Int32(0)]
                old_expected0 = expected_t[fx.Int32(0)]
                old_expected1 = expected_t[fx.Int32(1)]
                next_parity = old_parity ^ fx.Int32(1)
                next_expected = (
                    (next_parity == fx.Int32(0)).select(old_expected0, old_expected1) + fx.Int32(done_step)
                )
                if tid == fx.Int32(0):
                    expected_t[next_parity] = next_expected
                    parity_t[fx.Int32(0)] = next_parity
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                comm_ops.fence_agent_release()
                if tid == fx.Int32(0):
                    comm_ops.store_i32_system(gate_addr, fx.Int32(0), gate_epoch)
            fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
            fx.barrier()
        else:
            # Producers wait for the gate after prefetching their route (below).
            if const_expr(not sh_pre_gate):
                if tid == fx.Int32(0):
                    if ticket > fx.Int32(dispatch_blocks):
                        _wait_gate()
            fx.barrier()

        if const_expr(shared_in):
            # The shared expert's pre-phase (see the comment at shared_in): consumer c runs GEMM1 unit c, then
            # GEMM2 unit c, of this rank's ceil(rows / 32) m-tiles.
            def _sh_ptr(slot):
                return fx.Int64(comm_ops.wave_uniform_i64(disp_t[fx.Int32(int(slot))]))

            sh_c = fx.Int32(rocdl.readfirstlane(T.i32, (ticket - fx.Int32(dispatch_blocks + 1)).ir_value()))
            sh_epoch = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(gate_epoch).ir_value()))
            sh_sync = _sh_ptr(DispatchSlot.SH_SYNC)
            sh_act, sh_act_scale = _sh_ptr(DispatchSlot.SH_ACT), _sh_ptr(DispatchSlot.SH_ACT_SCALE)
            sh_tiles = (i32_cur_tok + fx.Int32(_se.SORT_BLOCK_M - 1)) // fx.Int32(_se.SORT_BLOCK_M)
            sh_lds_out = _LdsF32View(fx.recast_iter(fx.Float32, lds.pool.ptr))
            sh_partial_lds = None
            if const_expr(shared_in == 2 and not sh_split_wg):
                sh_partial_lds = lds.sh_partial
            # Stores past L2 (sc0 sc1): the unit that reads them may sit on another XCD. Every workgroup's stores
            # fill whole cache lines of its own (activation rows by 128 columns, tile-layout scales, 256 BF16
            # columns), and its counter is touched only after they drained.
            if const_expr(sh_split_wg):
                def _sh_wait_partials():
                    # The second-half workgroup's unit: its first half's accumulators went through to memory
                    # before the epoch; only this CU's L1 can be stale.
                    if tid == fx.Int32(0):
                        comm_ops.wait_i32_until_equals(
                            sh_sync + fx.Int64(_SH_PART_READY) + fx.Int64(sh_c >> fx.Int32(1)) * fx.Int64(4), sh_epoch)
                    fx.barrier()
                    comm_ops.invalidate_l1()

                sh_run_h0, sh_run_h1 = _se.build_shared_half_tiles(
                    x_addr=_base_i64(fx.get_iter(sorted_token_ids)), scale_x_addr=_base_i64(fx.get_iter(expert_ids)),
                    w_addr=_sh_ptr(DispatchSlot.SH_GU_W), scale_w_addr=_sh_ptr(DispatchSlot.SH_GU_S), rows=i32_cur_tok,
                    k_dim=model_dim,
                    epilogue=_se.RefActQuantEpilogue(out_addr=sh_act, out_scale_addr=sh_act_scale, rows=i32_cur_tok,
                        inter_dim=inter_dim, limit=swiglu_limit, lds_out=sh_lds_out, store_cm=_SC0_SC1_CACHE,
                        scale_layout="tile"),
                    a_buf=lds.pool, a_scale_lds=lds.A_scale, part_addr=_sh_ptr(DispatchSlot.SH_PART), n_tiles=sh_n1,
                    units=sh_tiles * fx.Int32(sh_n1), wait=_sh_wait_partials, gate_up_half=inter_dim,
                    store_cm=_SC0_SC1_CACHE)
            else:
                sh_run_g1 = _se.build_shared_tile(
                    x_addr=_base_i64(fx.get_iter(sorted_token_ids)), scale_x_addr=_base_i64(fx.get_iter(expert_ids)),
                    w_addr=_sh_ptr(DispatchSlot.SH_GU_W), scale_w_addr=_sh_ptr(DispatchSlot.SH_GU_S), rows=i32_cur_tok,
                    k_dim=model_dim,
                    epilogue=_se.RefActQuantEpilogue(out_addr=sh_act, out_scale_addr=sh_act_scale, rows=i32_cur_tok,
                        inter_dim=inter_dim, limit=swiglu_limit, lds_out=sh_lds_out, store_cm=_SC0_SC1_CACHE,
                        scale_layout="tile"),
                    a_buf=lds.pool, a_scale_lds=lds.A_scale, gate_up_half=inter_dim, ksplit=shared_in,
                    partial_lds=sh_partial_lds)
            sh_run_g2 = _se.build_shared_tile(
                x_addr=sh_act, scale_x_addr=sh_act_scale, w_addr=_sh_ptr(DispatchSlot.SH_DN_W),
                scale_w_addr=_sh_ptr(DispatchSlot.SH_DN_S), rows=i32_cur_tok, k_dim=inter_dim,
                epilogue=_se.Bf16RowsEpilogue(out_addr=addr_in_sc, rows=i32_cur_tok, n_out=model_dim,
                    lds_out=sh_lds_out, store_cm=_SC0_SC1_CACHE),
                a_buf=lds.pool, a_scale_lds=lds.A_scale, a_scale_layout="tile", a_scale_lds_bytes=n_scale_bytes)
            if const_expr(sh_split_wg):
                # Consumer c: K half c & 1 of GEMM1 unit c >> 1 (the unit's two workgroups sit on two XCDs).
                if (sh_c >= fx.Int32(0)) & (sh_c < sh_tiles * fx.Int32(2 * sh_n1)):
                    sh_u1 = sh_c >> fx.Int32(1)
                    sh_m1 = sh_u1 // fx.Int32(sh_n1)
                    if (sh_c & fx.Int32(1)) == fx.Int32(0):
                        sh_run_h0(sh_m1, sh_u1 - sh_m1 * fx.Int32(sh_n1))
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                        fx.barrier()
                        if tid == fx.Int32(0):
                            # The partials drained: the unit's second-half workgroup may read them.
                            comm_ops.store_i32_global_agent_release(
                                sh_sync + fx.Int64(_SH_PART_READY) + fx.Int64(sh_u1) * fx.Int64(4), sh_epoch)
                    else:
                        sh_run_h1(sh_m1, sh_u1 - sh_m1 * fx.Int32(sh_n1))
                        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                        fx.barrier()
                        if tid == fx.Int32(0):
                            sh_cnt = sh_sync + fx.Int64(sh_m1) * fx.Int64(4)
                            if fx.Int32(comm_ops.atomic_add_agent(sh_cnt, fx.Int32(1))) == fx.Int32(sh_n1 - 1):
                                # The m-tile's last GEMM1 unit: its GEMM2 units may start; rewind for the next launch.
                                comm_ops.store_i32_global_agent_release(sh_cnt + fx.Int64(_SH_READY), sh_epoch)
                                comm_ops.atomic_add_agent(sh_cnt, fx.Int32(-sh_n1))
            elif const_expr(sh_pair):
                # shared_in_opt bit 4 (shared_pair_unit): consumer c's place, the same for both GEMMs; a rank
                # without rows has no m-tile (the divisor is then 1 and no unit is valid).
                sh_octet = sh_c >> fx.Int32(3)
                sh_div = (sh_tiles > fx.Int32(1)).select(sh_tiles, fx.Int32(1))
                sh_q = sh_octet // sh_div
                sh_pm = sh_octet - sh_q * sh_div
                sh_pn = sh_q * fx.Int32(8) + (sh_c & fx.Int32(7))
                sh_pv = (sh_c >= fx.Int32(0)) & (sh_pm < sh_tiles)
                if sh_pv & (sh_pn < fx.Int32(sh_n1)):
                    if const_expr(sh_stagger):
                        # shared_in_opt bit 16: every wave of an m-tile m unit sleeps m beats first.
                        for _beat in range(fx.Int32(0), sh_pm, 1):
                            llvm.InlineAsmOp(None, [], f"s_sleep {_PAIR_STAGGER_UNITS}", "", has_side_effects=True)
                    sh_run_g1(sh_pm, sh_pn)
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    fx.barrier()
                    if tid == fx.Int32(0):
                        sh_cnt = sh_sync + fx.Int64(sh_pm) * fx.Int64(4)
                        if fx.Int32(comm_ops.atomic_add_agent(sh_cnt, fx.Int32(1))) == fx.Int32(sh_n1 - 1):
                            # The m-tile's last GEMM1 unit: its GEMM2 units may start; rewind for the next launch.
                            comm_ops.store_i32_global_agent_release(sh_cnt + fx.Int64(_SH_READY), sh_epoch)
                            comm_ops.atomic_add_agent(sh_cnt, fx.Int32(-sh_n1))
            else:
                if (sh_c >= fx.Int32(0)) & (sh_c < sh_tiles * fx.Int32(sh_n1)):
                    sh_m1 = sh_c // fx.Int32(sh_n1)
                    sh_run_g1(sh_m1, sh_c - sh_m1 * fx.Int32(sh_n1))
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    fx.barrier()
                    if tid == fx.Int32(0):
                        sh_cnt = sh_sync + fx.Int64(sh_m1) * fx.Int64(4)
                        if fx.Int32(comm_ops.atomic_add_agent(sh_cnt, fx.Int32(1))) == fx.Int32(sh_n1 - 1):
                            # The m-tile's last GEMM1 unit: its GEMM2 units may start; rewind for the next launch.
                            comm_ops.store_i32_global_agent_release(sh_cnt + fx.Int64(_SH_READY), sh_epoch)
                            comm_ops.atomic_add_agent(sh_cnt, fx.Int32(-sh_n1))
            if const_expr(sh_pair):
                if sh_pv & (sh_pn < fx.Int32(sh_n2)):
                    if tid == fx.Int32(0):
                        comm_ops.wait_i32_until_equals(sh_sync + fx.Int64(_SH_READY) + fx.Int64(sh_pm) * fx.Int64(4), sh_epoch)
                    fx.barrier()
                    # The activation rows went through to memory before the epoch: only this CU's L1 can be stale.
                    comm_ops.invalidate_l1()
                    sh_run_g2(sh_pm, sh_pn)
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    fx.barrier()
                    if tid == fx.Int32(0):
                        sh_total = sh_tiles * fx.Int32(sh_n2)
                        if fx.Int32(comm_ops.atomic_add_agent(sh_sync + fx.Int64(_SH_G2_COUNT), fx.Int32(1))) == sh_total - fx.Int32(1):
                            # The launch's last shared unit: the output rows are complete.
                            comm_ops.store_i32_global_agent_release(sh_sync + fx.Int64(_SH_DONE), sh_epoch)
                            comm_ops.atomic_add_agent(sh_sync + fx.Int64(_SH_G2_COUNT), fx.Int32(0) - sh_total)
            elif (sh_c >= fx.Int32(0)) & (sh_c < sh_tiles * fx.Int32(sh_n2)):
                sh_m2 = sh_c // fx.Int32(sh_n2)
                if tid == fx.Int32(0):
                    comm_ops.wait_i32_until_equals(sh_sync + fx.Int64(_SH_READY) + fx.Int64(sh_m2) * fx.Int64(4), sh_epoch)
                fx.barrier()
                # The activation rows went through to memory before the epoch: only this CU's L1 can be stale.
                comm_ops.invalidate_l1()
                sh_run_g2(sh_m2, sh_c - sh_m2 * fx.Int32(sh_n2))
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.barrier()
                if tid == fx.Int32(0):
                    sh_total = sh_tiles * fx.Int32(sh_n2)
                    if fx.Int32(comm_ops.atomic_add_agent(sh_sync + fx.Int64(_SH_G2_COUNT), fx.Int32(1))) == sh_total - fx.Int32(1):
                        # The launch's last shared unit: the output rows are complete.
                        comm_ops.store_i32_global_agent_release(sh_sync + fx.Int64(_SH_DONE), sh_epoch)
                        comm_ops.atomic_add_agent(sh_sync + fx.Int64(_SH_G2_COUNT), fx.Int32(0) - sh_total)
            if const_expr(sh_pre_gate):
                # The consumers' gate wait, after the pre-phase instead of before it.
                if ticket > fx.Int32(dispatch_blocks):
                    if tid == fx.Int32(0):
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
            late_epoch = None
            if const_expr("pg" in m2f):
                route_parity = None
                route_expected = None

                def late_epoch():
                    if tid == fx.Int32(0):
                        _wait_gate()
                    fx.barrier()
                    return _load_epoch()
            else:
                if tid == fx.Int32(0):
                    _wait_gate()
                fx.barrier()
                route_parity, route_expected = _load_epoch()
            emit_direct_fixed_slot_payload(
                late_epoch=late_epoch, active_done="ad" in m2f,
                num_waves=NUM_WAVES, fz_npes=fz_npes, fz_epr=fz_epr, fz_k=fz_k, fz_cap=fz_cap,
                fz_mtpr=fz_mtpr, fz_rank=fz_rank, fz_total_experts=fz_total_experts, fz_nbytes=fz_nbytes,
                fz_n_i32=fz_n_i32,
                fz_scale_n_i32=fz_scale_n_i32, fz_enable_scales=fz_enable_scales, addr_disp=addr_disp,
                addr_in_tok=addr_in_tok, addr_in_idx=addr_in_idx, addr_in_wts=addr_in_wts, addr_in_sc=addr_in_sc,
                i32_cur_tok=i32_cur_tok, dispatch_blocks=dispatch_blocks, producer_slot=producer_slot,
                parity=route_parity, expected=route_expected, prefetched=prefetched, route_rounds=pre_rounds,
                prefetch_args=prefetch_args, dedup=dedup_rows, direct_done=direct_done, running_half=running_half,
                disp_order=disp_order,
            )
        payload_parity, payload_expected = _load_epoch()
        if owner:
            emit_direct_fixed_slot_finalize(
                fz_npes=fz_npes, fz_epr=fz_epr, fz_cap=fz_cap, fz_mtpr=fz_mtpr, fz_rank=fz_rank,
                fz_tile_m=fz_tile_m, addr_disp=addr_disp, parity=payload_parity,
                expected=payload_expected, plan_nv=(_disp_ptr(DispatchSlot.PLAN_NV), gate_epoch),
                plan_wt=plan_wt, keep_running=plan_local, uniform=head_uniform, running_half=running_half,
                lean_flag=bool(sync_lean & 2),
                stream_w=stream_w,
            )
        wave_id = fx.thread_idx.x // 64

        w_rsrc = ptr_buf_tensor(fx.get_iter(w), fx.Int32, unit_elems=4)
        sx_rsrc = ptr_buf_tensor(fx.get_iter(scale_x), fx.Int32, unit_elems=4)
        sw_rsrc = ptr_buf_tensor(fx.get_iter(scale_w), fx.Int32)
        trb_rsrc = ptr_buf_tensor(fx.get_iter(sorted_token_ids), fx.Int32)
        expert_rsrc = ptr_buf_tensor(fx.get_iter(expert_ids), fx.Int32)
        if const_expr(shared_in):
            # These two arguments carry the shared expert's input pair: the routed tiles must not read them (a
            # plan_local kernel takes its tile metadata from its own plan; any other use fails at trace time).
            trb_rsrc = expert_rsrc = None
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
                "srcmap": ptr_buf_tensor(_slot_meta_ptr(DispatchSlot.G2_STIDS), fx.Int32),
                # A constant bound keeps the A buffer resource scalar.
                "x_bytes": fx.Int32(_TOKEN_REGION_END),
            }
        plan_lookup = None
        if const_expr(plan_local):
            # i32 LDS bases of the typed fields: indexing the fields' typed pointers directly changes the GEMM1
            # schedule of the hu/la2 tok_done kernels (+15 waits); lds_typed_ptr keeps the old addressing.
            incl_base = fx.Int32(fx.ptrtoint(lds.plan_incl.ptr))
            count_base = fx.Int32(fx.ptrtoint(lds.plan_count.ptr))

            def plan_ptr(base, i):
                return lds_typed_ptr(base + i * fx.Int32(4), T.i32, align=4)

            def plan_lookup(m_tile):
                # The tile's expert = the experts whose inclusive tile count is <= m_tile.
                lane = tid & fx.Int32(63)
                expert = fx.Int32(0)
                for chunk in range_constexpr(plan_chunks):
                    incl = fx.ptr_load(plan_ptr(incl_base, fx.Int32(chunk * 64) + lane))
                    mask = rocdl.ballot(T.i64, incl <= m_tile)
                    expert = expert + fx.Int32(fx.Int64(fx.math.ctpop(mask)))
                expert = fx.Int32(rocdl.readfirstlane(T.i32, expert))
                incl_e = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(plan_ptr(incl_base, expert))))
                count_e = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(plan_ptr(count_base, expert))))
                tiles_e = (count_e + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
                tile_rows_base = (m_tile - incl_e + tiles_e) * fx.Int32(fz_tile_m)
                rows = count_e - tile_rows_base
                rows = (rows < fx.Int32(fz_tile_m)).select(rows, fx.Int32(fz_tile_m))
                return expert, expert * fx.Int32(fz_cap) + tile_rows_base, rows
        plan_multi_tile = None
        if const_expr(expert_cached_w and plan_local):

            def plan_multi_tile(m_tile):
                # 1 when m-tile ``m_tile`` belongs to an expert with more than one m-tile (plan_lookup's expert search).
                lane = tid & fx.Int32(63)
                expert = fx.Int32(0)
                for chunk in range_constexpr(plan_chunks):
                    incl = fx.ptr_load(plan_ptr(incl_base, fx.Int32(chunk * 64) + lane))
                    mask = rocdl.ballot(T.i64, incl <= m_tile)
                    expert = expert + fx.Int32(fx.Int64(fx.math.ctpop(mask)))
                expert = fx.Int32(rocdl.readfirstlane(T.i32, expert))
                count_e = fx.Int32(rocdl.readfirstlane(T.i32, fx.ptr_load(plan_ptr(count_base, expert))))
                return (count_e > fx.Int32(fz_tile_m)).select(fx.Int32(1), fx.Int32(0))
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
            **({"mfma_m_active": 1} if "hm" in probe_skip else {}),
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
                **({"mfma_m_active": 1} if "hm" in probe_skip else {}),
            )

        a_g2_cnt = _disp_ptr(DispatchSlot.G2_CNT)
        a_g2_ready = _disp_ptr(DispatchSlot.G2_READY)
        # GEMM2 setup is loaded before GEMM1 so the GEMM1 -> GEMM2 switch costs no memory round trip.
        g2_lane = tid % fx.Int32(64)
        g2_wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // fx.Int32(64)))
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.pool.ptr))
        g2_peer_base = ptr_buf_tensor(_disp_ptr(DispatchSlot.G2_P2P_COMB_INP), fx.Int64)[
            (tid < fx.Int32(fz_npes)).select(tid, fx.Int32(0))]
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
                slot = fx.Int32(chunk * 64) + lane
                fx.ptr_store(metadata_base + num_expert_tiles, plan_ptr(incl_base, slot))
                fx.ptr_store(safe_count, plan_ptr(count_base, slot))
            hot_expert = max_count >= fx.Int32(_HOT_EXPERT_MIN_ROWS)
            return _plan_flag(no_overflow, total_tiles, hot_expert, routed_rows, fz_tile_m, stream_w)
        if tid == fx.Int32(0):
            xcd = fx.Int32(comm_ops.hw_xcc_id()) & fx.Int32(7)
            # The owner reset the counters before opening the gate: the first work fetch overlaps the plan wait.
            first_local = fx.Int32(
                comm_ops.atomic_add_agent(a_work_head + fx.Int64(xcd) * fx.Int64(64), fx.Int32(1)))
            if const_expr(not plan_local):
                a_xcd_acq = _disp_ptr(DispatchSlot.XCD_ACQ) + fx.Int64(xcd) * fx.Int64(128)
                plan_word = fx.Int64(0)
                # Match this launch's epoch in the high word exactly: a >= wait breaks when the 32-bit epoch wraps.
                if first_local == fx.Int32(0):
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
        if const_expr("nw" in m2f):
            num_valid = fx.Int32(0)
        if const_expr(dedup_rows):
            row_map["x_token"] = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(token_offsets[0])))
            row_map["scale_token"] = fx.Int32(rocdl.readfirstlane(T.i32, fx.Int32(token_offsets[1] // fx.Int64(16))))
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
        cut_tiles = 4 if const_expr(N_TILES % 4 == 0) else 2
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
            early_cta = ((fx.block_idx.x >> fx.Int32(3)) % fx.Int32(G2_EARLY_K)) == fx.Int32(0)
            early_thr = early_cta.select(shard_work * fx.Int32(G2_EARLY_P) // fx.Int32(16), shard_work)

        # shared_in_opt bit 8: the unit loop's atomics take per-lane addresses, which keeps them out of LLVM's atomic
        # optimizer (its readfirstlane of a result waits for the RMW right behind it), so that two atomics issued
        # one after the other share a round trip.
        _unit_addr = _vgpr_i64 if const_expr(sh_lean) else (lambda address: address)

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
                    _unit_addr(a_work_head + fx.Int64(work_shard) * fx.Int64(64) + fx.Int64(side)), fx.Int32(1)))
                inc = (old_own + fx.Int32(1) == own_hi - own_lo).select(own_hi - own_lo, fx.Int32(0))
                if inc != fx.Int32(0):
                    comm_ops.fence_agent_release()
            if inc != fx.Int32(0):
                cnt_addr = _unit_addr(a_g2_cnt + fx.Int64(m) * fx.Int64(4))
                old_cnt = fx.Int32(comm_ops.atomic_add_agent(cnt_addr, inc))
                if old_cnt + inc == fx.Int32(N_TILES):
                    # One writer per m-tile and launch; GEMM2 waits for equality, so the epoch may wrap
                    # (a max would stick at 0xffffffff).
                    if const_expr(sync_lean & 1):
                        # sync_lean bit 1: no L2 writeback before the ready epoch. A GEMM2 unit of an m-tile GEMM1 did
                        # not cut runs on the XCD whose L2 holds the tile (store completion), and each part of a cut
                        # m-tile was written back by its own release before it was counted.
                        fx.generic_store(comm_ops._to_ptr_global(a_g2_ready + fx.Int64(m) * fx.Int64(4)),
                            fx.Int32(gate_epoch), memory_order=fx.AtomicOrdering.Monotonic,
                            syncscope=fx.rocdl.SyncScope.AgentOneAs)
                    else:
                        comm_ops.store_i32_global_agent_release(a_g2_ready + fx.Int64(m) * fx.Int64(4), gate_epoch)
                    # All counts of this launch are in: rewind to 0, since the next launch's config may use another
                    # N_TILES (tile_n 256 / 128), and a leftover count would make its ready fire early or never.
                    comm_ops.atomic_add_agent(cnt_addr, fx.Int32(-N_TILES))
        if const_expr(sh_lean):
            # shared_in_opt bit 8: tid 0 reads the three flags next to its claims and leaves what they showed in LDS
            # (lean_tab; the exit's tid 64 reads its word behind a barrier). The fetches take their operands from
            # LDS as well, so that the loop keeps no register for them across the tile bodies.
            lean_tab = fx.Int32(fx.ptrtoint(lds.lean_tab.ptr))

            def _lean_word(offset):
                return lds_typed_ptr(lean_tab + fx.Int32(offset), T.i32, align=4)

            def _lean_i32(offset):
                return fx.Int32(fx.ptr_load(_lean_word(offset), result_type=T.i32))

            def _lean_i64(offset):
                return fx.Int64(fx.ptr_load(lds_typed_ptr(lean_tab + fx.Int32(offset), T.i64, align=8)))

            if tid == fx.Int32(0):
                lean_base = (shard_base * fx.Int32(g2_n_blocks)) // fx.Int32(N_TILES)
                lean_limit = g2_max_m_blocks - fx.Int32(_LEAN_EPOCHS)
                for offset, value in (
                    (_LEAN_PLAN_SEEN, fx.Int32(0)), (_LEAN_UNIT_READY, fx.Int32(0)), (_LEAN_SHARED_DONE, fx.Int32(0)),
                    (_LEAN_EPOCH, fx.Int32(gate_epoch)), (_LEAN_BASE, lean_base),
                    (_LEAN_UNITS, (shard_end * fx.Int32(g2_n_blocks)) // fx.Int32(N_TILES) - lean_base),
                    (_LEAN_LIMIT, (lean_limit > fx.Int32(0)).select(lean_limit, fx.Int32(0))),
                ):
                    fx.ptr_store(value, _lean_word(offset))
                for offset, value in (
                    (_LEAN_PLAN_ADDR, disp_t[fx.Int32(int(DispatchSlot.PLAN_NV))]),
                    (_LEAN_READY_ADDR, a_g2_ready), (_LEAN_DONE_ADDR, sh_sync + fx.Int64(_SH_DONE)),
                ):
                    fx.ptr_store(fx.Int64(value), lds_typed_ptr(lean_tab + fx.Int32(offset), T.i64, align=8))

            def _lean_epochs(last_index):
                # Next to a GEMM2 claim: the ready epochs of _LEAN_EPOCHS m-tiles from the one that holds the
                # slice's GEMM2 unit after ``last_index`` (the unit this CTA ran last), kept inside the epoch array
                # -- where the claim in flight most likely lands. Read as the unit's own wait reads them. Returns
                # the window's first m-tile and the epochs.
                first = (_lean_i32(_LEAN_BASE) + last_index + fx.Int32(1)) // fx.Int32(g2_n_blocks)
                limit = _lean_i32(_LEAN_LIMIT)
                first = (first < limit).select(first, limit)
                ready = _lean_i64(_LEAN_READY_ADDR)
                return first, [
                    fx.Int32(comm_ops.load_i32_acquire(ready + fx.Int64(first + fx.Int32(k)) * fx.Int64(4)))
                    for k in range_constexpr(_LEAN_EPOCHS)]

            def _lean_ready(first, epochs, g2_index):
                # 1 when ``g2_index`` is a GEMM2 unit of the slice and the window read showed this launch's epoch on
                # its m-tile.
                epoch = _lean_i32(_LEAN_EPOCH)
                rel = (_lean_i32(_LEAN_BASE) + g2_index) // fx.Int32(g2_n_blocks) - first
                hit = fx.Int32(0)
                for k in range_constexpr(_LEAN_EPOCHS):
                    hit = hit | ((rel == fx.Int32(k)) & (epochs[k] == epoch)).select(fx.Int32(1), fx.Int32(0))
                return ((g2_index >= fx.Int32(0)) & (g2_index < _lean_i32(_LEAN_UNITS))).select(hit, fx.Int32(0))

            def _lean_flag(seen, offset):
                fx.ptr_store(seen.select(fx.Int32(1), fx.Int32(0)), _lean_word(offset))
        while consumer_active:
            if const_expr(sh_lean):
                # Every fetch after the first has a finished tile to count (the loop ends with the first fetch
                # that finds no work), so the claim, the count and the two reads go out in one round trip. (No
                # epoch window here: with one, LLVM schedules the GEMM1 K loops differently; isa_loops check.)
                if tid == fx.Int32(0):
                    local_work = first_local
                    if g1_fetch != fx.Int32(0):
                        local_work = fx.Int32(comm_ops.atomic_add_agent(
                            _unit_addr(a_work_head + fx.Int64(work_shard) * fx.Int64(64)), fx.Int32(1)))
                        lean_plan = fx.Int64(comm_ops.load_i64_acquire(_lean_i64(_LEAN_PLAN_ADDR)))
                        lean_shared = fx.Int32(comm_ops.load_i32_acquire(_lean_i64(_LEAN_DONE_ADDR)))
                        _count_tile(pend_m)
                        lean_epoch = _lean_i32(_LEAN_EPOCH)
                        _lean_flag((lean_plan >> fx.Int64(32)) == fx.Int64(lean_epoch), _LEAN_PLAN_SEEN)
                        _lean_flag(lean_shared == lean_epoch, _LEAN_SHARED_DONE)
                    work = (local_work < shard_work).select(shard_base + local_work, total_work)
                    fx.ptr_store(local_work - shard_work, g2_scratch)
                    fx.ptr_store(Vec.from_elements([work], fx.Int32), work_scratch)
            elif tid == fx.Int32(0):
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
                if const_expr("g1" in probe_skip):
                    pass
                elif const_expr(b_cache_modifier != 0):
                    g1_cached = cached_w
                    if const_expr(plan_multi_tile is not None):
                        g1_cached = g1_cached | plan_multi_tile(work // fx.Int32(N_TILES))
                    if g1_cached != fx.Int32(0):
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
        if const_expr(sh_lean):
            # GEMM2 reads the plan CTA's metadata: spin only if the read next to the last GEMM1 claim did not show
            # this launch's flag.
            if tid == fx.Int32(0):
                if _lean_i32(_LEAN_PLAN_SEEN) == fx.Int32(0):
                    comm_ops.spin_until_shifted_eq_i64(_disp_ptr(DispatchSlot.PLAN_NV), gate_epoch, shift=32)
        elif const_expr(plan_local):
            # GEMM2 reads the plan CTA's metadata (its first unit drops the L1 after its ready wait).
            if tid == fx.Int32(0):
                comm_ops.spin_until_shifted_eq_i64(_disp_ptr(DispatchSlot.PLAN_NV), gate_epoch, shift=32)
        while g2_active:
            fx.barrier()
            if const_expr(sh_lean):
                if tid == fx.Int32(0):
                    if g2_fetch != fx.Int32(0):
                        lean_claim = fx.Int32(comm_ops.atomic_add_agent(_unit_addr(g2_head), fx.Int32(1)))
                        lean_shared = fx.Int32(comm_ops.load_i32_acquire(_lean_i64(_LEAN_DONE_ADDR)))
                        lean_first, lean_window = _lean_epochs(
                            fx.Int32(fx.ptr_load(g2_scratch, result_type=T.i32)))
                        fx.ptr_store(lean_claim - g2_claim_base, g2_scratch)
                        _lean_flag(lean_shared == _lean_i32(_LEAN_EPOCH), _LEAN_SHARED_DONE)
                        fx.ptr_store(_lean_ready(lean_first, lean_window, lean_claim - g2_claim_base),
                            _lean_word(_LEAN_UNIT_READY))
            elif tid == fx.Int32(0):
                if g2_fetch != fx.Int32(0):
                    fx.ptr_store(fx.Int32(comm_ops.atomic_add_agent(g2_head, fx.Int32(1))) - g2_claim_base,
                        g2_scratch)
            fx.barrier()
            g2_local = fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.ptr_load(g2_scratch).ir_value()))
            if g2_local < g2_units:
                g2_u = g2_base + g2_local
                g2_m = g2_u // fx.Int32(g2_n_blocks)
                g2_n = g2_u - g2_m * fx.Int32(g2_n_blocks)
                if const_expr(sh_lean):
                    # The epoch map read next to this unit's claim may have shown its m-tile already.
                    if tid == fx.Int32(0):
                        if _lean_i32(_LEAN_UNIT_READY) == fx.Int32(0):
                            comm_ops.wait_i32_until_equals(a_g2_ready + fx.Int64(g2_m) * fx.Int64(4), gate_epoch)
                elif tid == fx.Int32(0):
                    comm_ops.wait_i32_until_equals(a_g2_ready + fx.Int64(g2_m) * fx.Int64(4), gate_epoch)
                fx.barrier()
                # GEMM1 wrote this m-tile through this XCD's L2, or (in an m-tile shared by two XCDs) written back
                # to memory by the other XCD into lines this XCD has not cached, before it was counted: only the
                # CU's L1 can be stale.
                comm_ops.invalidate_l1()
                # As in GEMM1, a rank that receives many rows reads W2 through the cache.
                g2_cached = cached_w
                if const_expr(expert_cached_w == 2 and plan_multi_tile is not None):
                    g2_cached = g2_cached | plan_multi_tile(g2_m)
                if const_expr("g2" in probe_skip):
                    pass
                elif g2_cached != fx.Int32(0):
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

        if const_expr("xt" in m2f):
            if tid == fx.Int32(0):
                fx.ptr_store(a_work_head, lds_typed_ptr(fx.Int32(xt_lds_off), T.i64, align=8))
        # The entry-ticket epoch is the same on every rank and needs no load; the high bit keeps it apart from
        # the combine op's own flag values in the shared xdb slots.
        fc_flag = fx.Int64(gate_epoch) + fx.Int64(1 << 40)
        if const_expr(shared_in):
            # No workgroup takes its exit ticket before the shared expert's rows are complete, so the reducers
            # (the last tickets) add to finished rows. One lane of wave 1 waits; the tail's first barrier joins it.
            if const_expr(sh_lean):
                if tid == fx.Int32(64):
                    if i32_cur_tok > fx.Int32(0):
                        if _lean_i32(_LEAN_SHARED_DONE) == fx.Int32(0):
                            comm_ops.wait_i32_until_equals(sh_sync + fx.Int64(_SH_DONE), sh_epoch)
            elif tid == fx.Int32(64):
                if i32_cur_tok > fx.Int32(0):
                    comm_ops.wait_i32_until_equals(sh_sync + fx.Int64(_SH_DONE), sh_epoch)
        fused_combine_tail(tid, g2_lane, g2_wave, _disp_ptr(DispatchSlot.FC_CNT), fc_flag,
            _disp_ptr(DispatchSlot.FC_XDB), _disp_ptr(DispatchSlot.FC_P2P_XDB), _disp_ptr(DispatchSlot.FC_INP),
            _disp_ptr(DispatchSlot.FC_OUT), _disp_ptr(DispatchSlot.FC_TRECV), addr_in_idx, i32_cur_tok,
            fx.Int32(launch_grid_x), fp8_rows,
            reset_running_addr() if const_expr(plan_local) else fx.Int64(0), tok_off, tok_reset,
            addr_in_sc if const_expr(shared_add) else fx.Int64(0))

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
# Experiment switch (default off: unset keeps 3584 and the kernel unchanged): AITER_MEGA_SMALL_FP8_ROWS_MIN=<rows>
# moves that threshold, e.g. 1 makes every rank send FP8 GEMM2 rows (half the XGMI scatter bytes, FP8 transport
# rounding); the kernel name gains _f8r<rows>. Read once, at import; every rank must set the same value.
_FP8_ROWS_MIN_DEFAULT = _FP8_ROWS_MIN_ROWS
_FP8_ROWS_MIN_ROWS = int(
    os.environ.get("AITER_MEGA_SMALL_FP8_ROWS_MIN", _FP8_ROWS_MIN_DEFAULT)
)
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
    # Experiment switches, read at launch-config time (default off: unset adds no knob, so the kernel name and the
    # compile cache key are unchanged). AITER_MEGA_SMALL_G2_B2STAGE=1: 2-stage B pipeline for the 64-row GEMM2 tile
    # (kernel suffix _g2b2). AITER_MEGA_SMALL_PROBE=g1 | g2 | g1+g2: phase-timing kernels without those GEMM
    # bodies (suffix _probe*; wrong output, benchmarks only). Every rank must set the same values.
    extra = {}
    if (
        os.environ.get("AITER_MEGA_SMALL_G2_B2STAGE", "0") == "1"
        and config.stage2.block_m == 64
    ):
        extra["g2_b2stage"] = True
    if os.environ.get("AITER_MEGA_SMALL_PROBE", ""):
        extra["probe"] = os.environ["AITER_MEGA_SMALL_PROBE"]
    # M2 fixed-base prototype switches (see compile_mega_moe_stage1): AITER_MEGA_SMALL_M2=st+pg+ad+tl (default off).
    if os.environ.get("AITER_MEGA_SMALL_M2", ""):
        extra["m2"] = os.environ["AITER_MEGA_SMALL_M2"]
    # AITER_MEGA_SMALL_EXPERT_CACHED_W=1 | 2 (default off: unset adds no knob): cached weight loads per multi-tile expert
    # on every bucket (see compile_mega_moe_stage1's expert_cached_w; kernel suffix _ecw1 / _ecw2). Not a protocol knob.
    if os.environ.get("AITER_MEGA_SMALL_EXPERT_CACHED_W", "0") not in ("", "0"):
        extra["expert_cached_w"] = int(os.environ["AITER_MEGA_SMALL_EXPERT_CACHED_W"])
    elif getattr(k1, "expert_cached_w", 0):
        # The per-bucket form: FusedStage1Knobs.expert_cached_w, e.g. AITER_MEGA_SMALL_M_OVERRIDE="64:fs1.expert_cached_w=2".
        extra["expert_cached_w"] = int(k1.expert_cached_w)
    # FusedStage1Knobs.disp_order (default 0: no knob, kernel name and cache key unchanged).
    if getattr(k1, "disp_order", 0):
        extra["disp_order"] = int(k1.disp_order)
    if getattr(k1, "sync_lean", 0):
        extra["sync_lean"] = int(k1.sync_lean)
    # fmt: off
    return dict(
        **extra,
        dedup_rows=k1.dedup_rows, plan_wt=k1.plan_wt, plan_local=k1.plan_local,
        g2_early=(k2.early_k, k2.early_p if k2.early_k else 0), g2_uniform=k1.g2_uniform,
        head_uniform=k1.head_uniform, g2_epi_lean=k1.g2_epi_lean, g1_lookahead=k1.g1_lookahead,
        g1_a_ahead=k1.g1_a_ahead, direct_done=k1.direct_done, tok_done=k1.tok_done, tok_counters=k1.tok_counters,
        bf16_head_tokens=k1.bf16_head_tokens, epi_evec=k1.epi_evec, epi_lean=k1.epi_lean,
        ascale_packed=k1.ascale_packed, stream_w=k1.stream_w,
    )
    # fmt: on


def shared_in_fits(
    config: MegaMoEConfig, rows: int, num_cu: int, model_dim: int, inter_dim: int
) -> bool:
    """Whether the fused kernel of ``config`` can also run the shared expert of a ``rows``-row call
    (compile_mega_moe_stage1's shared_in): its own plan in every CTA, 4 waves, 32- or 64-row routed tiles (the
    shared units are 32-row tiles of their own either way; the kernel's LDS, two workgroups a CU, is checked where it
    is built), and a consumer workgroup for each of the call's shared GEMM1 / GEMM2 units.
    """
    stage1, knobs = config.stage1, config.fused_stage1
    consumers = num_cu * min(stage1.grid_mult, 2) - 1 - stage1.num_dispatch_cu
    return (
        rows > 0
        and knobs.plan_local
        and not knobs.tok_done
        and stage1.num_waves == 4
        and stage1.sort_block_m in (32, 64)
        and ceildiv(rows, 32) * max((2 * inter_dim) // 256, model_dim // 256)
        <= consumers
    )


def shared_in_fits_split(
    config: MegaMoEConfig, rows: int, num_cu: int, model_dim: int, inter_dim: int
) -> bool:
    """``shared_in_fits`` and room for shared_in_opt bit 1 (the K halves of a GEMM1 unit on two workgroups): a
    consumer workgroup per half and an epoch word per unit."""
    stage1 = config.stage1
    consumers = num_cu * min(stage1.grid_mult, 2) - 1 - stage1.num_dispatch_cu
    units = ceildiv(rows, 32) * ((2 * inter_dim) // 256)
    return (
        shared_in_fits(config, rows, num_cu, model_dim, inter_dim)
        and 2 * units <= consumers
        and units <= _SH_PART_UNITS
    )


def shared_pair_unit(consumer: int, tiles: int, n_tiles: int):
    """shared_in_opt bit 4: the unit ``(m_tile, n_tile)`` of consumer workgroup ``consumer`` in a call of ``tiles``
    m-tiles with ``n_tiles`` column tiles each, or None. Column tile ``8 q + r`` of m-tile ``m`` is consumer
    ``8 (q tiles + m) + r``: every unit has one consumer, and the m-tiles' units of a column tile are 8 consumers
    apart (the kernel computes the same in compile_mega_moe_stage1)."""
    if consumer < 0 or tiles <= 0:
        return None
    q, m = divmod(consumer >> 3, tiles)
    n = 8 * q + (consumer & 7)
    return (m, n) if n < n_tiles else None


def shared_pair_consumers(tiles: int, n_tiles: int) -> int:
    """shared_in_opt bit 4: consumer workgroups the pair map needs (one past its highest consumer with a unit)."""
    if tiles <= 0:
        return 0
    return 8 * (((n_tiles - 1) >> 3) * tiles + tiles - 1) + ((n_tiles - 1) & 7) + 1


def shared_in_fits_pair(
    config: MegaMoEConfig, rows: int, num_cu: int, model_dim: int, inter_dim: int
) -> bool:
    """``shared_in_fits`` and room for shared_in_opt bit 4 (the units of a weight slab on one XCD): a consumer
    workgroup at every index the pair map uses."""
    stage1 = config.stage1
    consumers = num_cu * min(stage1.grid_mult, 2) - 1 - stage1.num_dispatch_cu
    n_tiles = max((2 * inter_dim) // 256, model_dim // 256)
    return (
        shared_in_fits(config, rows, num_cu, model_dim, inter_dim)
        and shared_pair_consumers(ceildiv(rows, 32), n_tiles) <= consumers
    )


def shared_in_lean_fits(config: MegaMoEConfig) -> bool:
    """Whether shared_in_opt bit 8 (the unit loop without the observing round trips) applies to a config's kernel:
    its own plan in every CTA and no early GEMM2 start (compile_mega_moe_stage1's assertion; the per-XCD exit
    tickets of M2 xt are a process-wide experiment switch and are refused there)."""
    return bool(config.fused_stage1.plan_local) and not config.fused_stage2.early_k


def shared_in_opt_env() -> int:
    """AITER_MEGA_SHARED_IN_OPT (default 0; bits, see compile_mega_moe_stage1's shared_in_opt): 1 = the K
    halves of a split shared GEMM1 unit on two workgroups, 2 = the pre-phase before the gate, 4 = the units of one
    weight slab on one XCD, 8 = the routed unit loop without the observing round trips, 16 = bit 4's later m-tiles
    start a beat behind. Read per call."""
    value = os.environ.get("AITER_MEGA_SHARED_IN_OPT", "0").strip() or "0"
    if value not in tuple(str(v) for v in range(32)):
        raise ValueError(
            f"AITER_MEGA_SHARED_IN_OPT={value!r}: 0..31 (bits 1, 2, 4, 8, 16)"
        )
    return int(value)


def _arg_view(tensor):
    """A fixed-shape byte view of a tensor's start, for a kernel tensor argument that only carries an address."""
    return tensor.view(torch.uint8).reshape(-1)[:16]


class SmallMRuntime:
    """Small-input single-kernel MegaMoE (MTPR <= 512) with one in-flight launch per instance.

    Owns its protocol state and combine output; the payload, GEMM1-output and combine-input
    storage may be passed in as shared arenas. Weights are bound by pointer so one instance
    serves every layer.
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
            1,
            max_tok_per_rank,
            world_size=world_size,
            experts_per_rank=experts // world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
        )
        if config.fused_stage1.dedup_rows:
            rows += world_size * max_tok_per_rank
        return nvm, rows, row_bytes, scale_n_i32

    @classmethod
    def ends_all_ranks(
        cls, *, world_size, model_dim, inter_dim, experts, topk, max_tok_per_rank
    ):
        """Whether every launch ends only once every peer's grid finished (its combine waits for every peer's flag).

        A tok_done bucket's combine takes each token once its own rows are counted, so a rank can leave, and start
        its next call's dispatch into its peers, while a peer still reads its payload: such a runtime cannot share
        payload storage with another route.
        """
        return not any(
            select_fused_config(
                bucket,
                max_tok_per_rank,
                world_size=world_size,
                experts_per_rank=experts // world_size,
                model_dim=model_dim,
                inter_dim=inter_dim,
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
        # The launcher of every config, so a forward skips compile_mega_moe_stage1's cache lookup.
        self._launchers = {}
        for config in self._configs.values():
            self._launcher(config)

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
        workspace["count_done"] = _zeroed_sym((2 * self.world_size,), torch.int32)
        if self._protocol.tok_counters:
            # Two parity halves of per-token GEMM2 completion counts, reached through each peer's FC_INP.
            workspace["tok_done"] = _zeroed_sym((2 * self.mtpr,), torch.int32)
            # Two launch-parity halves of the per-expert receive counters (compile_mega_moe_stage1), in place of
            # the combine op's single one; allocated after tok_done, so its distance from FC_INP is unchanged.
            workspace["running"] = _zeroed_sym((2 * self.epr,), torch.int32)
            # The routes' srcmap and weight slots, in the same two halves (fz_epr * fz_cap apart).
            # Half 1 starts epr * ll_cap past half 0; the non-compact slot layout (num_valid_max >= epr * ll_cap)
            # keeps it clear of half 0.
            assert op.num_valid_max >= self.epr * op.ll_cap, "double-buffered slots need the non-compact layout"
            meta_rows = op.num_valid_max + self.epr * op.ll_cap
            workspace["srcmap"] = _zeroed_sym((meta_rows,), torch.int32)
            workspace["wts"] = _zeroed_sym((meta_rows,), torch.float32)
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
        config = self._configs[
            fused_bucket_override(max(tokens, 1), self.mtpr)
            or nearest_token_bucket(max(tokens, 1), FUSED_TOKEN_BUCKETS)
        ]
        self._active_config = config
        return config

    def _launcher(self, config: MegaMoEConfig, **variant):
        """compile_mega_moe_stage1's launch handle for a config (and its shared_add / shared_in variant), cached
        per runtime."""
        key = (config, tuple(sorted(variant.items())))
        launch = self._launchers.get(key)
        if launch is not None:
            return launch
        stage1, stage2 = config.stage1, config.stage2
        assert config.p2p_quant == "none" and stage2.block_m == stage1.sort_block_m
        fused = (stage2.block_n, stage2.block_k, int(self.max_recv), self._reducers)
        # fmt: off
        launch = compile_mega_moe_stage1.lazy(
            model_dim=self.model_dim, inter_dim=self.inter_dim, rank=self.rank,
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
            **_kernel_knobs(config), **variant)
        # fmt: on
        self._launchers[key] = launch
        return launch

    def _launch(self, config: MegaMoEConfig, run_tokens, x_bf16, topk_ids, wts, stream, *, preload=False,
                shared_add=None, shared_in=None, shared_in_opt=0):
        """``shared_in = (ksplit, x_fp8 view, x_scale view, dispatch table)``: the shared_in kernel of this config;
        the two views ride in the kernel's tile-metadata tensor arguments, which that kernel does not read."""
        op = self._s1_op
        tile_row_base, expert_ids, disp = op.tile_row_base, op.sorted_expert_ids, self._s1_disp
        variant = {} if shared_add is None else {"shared_add": True}
        if shared_in is not None:
            assert shared_add is not None
            variant["shared_in"], tile_row_base, expert_ids, disp = shared_in
            if shared_in_opt:
                # (Only when set: the launch of option 0 is the call it was, with the kernel and key it had.)
                variant["shared_in_opt"] = shared_in_opt
        # preload: compile and load without dispatching (AOT bundles).
        # fmt: off
        (_preload_compiled if preload else _run_compiled)(
            self._launcher(config, **variant), self._s1_out, self._s1_rx, self._s1_w1, self._s1_scale_i32,
            self._s1_w1_scale, tile_row_base, expert_ids, self._s1_osd, fx.Int32(self._s1_nvm),
            fx.Int64(disp.data_ptr()), fx.Int32(run_tokens), fx.Int64(x_bf16.data_ptr()),
            fx.Int64(topk_ids.data_ptr()), fx.Int64(wts.data_ptr()),
            fx.Int64((x_bf16 if shared_add is None else shared_add).data_ptr()),
            fx.Int64(self._s1_epoch_parity.data_ptr()), fx.Int64(self._s1_epoch_expected.data_ptr()), stream)
        # fmt: on

    # ---- shared_in: the shared expert inside the kernel ----

    def _bucket_of(self, tokens: int) -> int:
        return fused_bucket_override(max(tokens, 1), self.mtpr) or nearest_token_bucket(max(tokens, 1), FUSED_TOKEN_BUCKETS)

    def _shared_ksplit(self, tokens: int) -> int:
        """The GEMM1 K split of the shared_in kernel for a ``tokens``-row call (1 or 2), 0 when the dense launches of
        that row count are not ones the tile bodies reproduce (shared_expert.dense_gate_up_ksplit)."""
        from .shared_expert import dense_gate_up_ksplit

        return dense_gate_up_ksplit(tokens, self.model_dim, self.inter_dim)

    def _shared_opt(self, tokens: int) -> int:
        """The shared_in_opt of a ``tokens``-row call's kernel: this runtime's ``shared_in_opt`` attribute if set
        (benchmarks switch it per call), else AITER_MEGA_SHARED_IN_OPT; bit 1 only for the split-K variant
        and only where every K half finds a workgroup, so the other row counts launch the kernels they launched."""
        opt = getattr(self, "shared_in_opt", None)
        opt = shared_in_opt_env() if opt is None else int(opt)
        if opt & 1 and not (self._shared_ksplit(tokens) == 2 and shared_in_fits_split(
                self._configs[self._bucket_of(tokens)], tokens, self._s1_num_cu, self.model_dim, self.inter_dim)):
            opt &= ~1
        if opt & 28:
            # Bit 4 (the units of a weight slab on one XCD): only for calls of two or more m-tiles (a one-tile
            # call's map is the m-major one), only where the map's highest consumer exists, and not next to bit 1,
            # whose workgroups take half units. Bit 16 (its stagger): only with bit 4. Bit 8 (lean unit loop): only
            # kernels it is written for.
            config = self._configs[self._bucket_of(tokens)]
            if opt & 4 and (opt & 1 or tokens <= 32 or not shared_in_fits_pair(
                    config, tokens, self._s1_num_cu, self.model_dim, self.inter_dim)):
                opt &= ~4
            if opt & 16 and not opt & 4:
                opt &= ~16
            if opt & 8 and not shared_in_lean_fits(config):
                opt &= ~8
        return opt

    def shared_in_supported(self, tokens: int) -> bool:
        """Whether the kernel a ``tokens``-row call launches can also compute the shared expert, with the bits of
        the dense launches of that row count."""
        return 0 < tokens <= self.mtpr and self.mtpr <= 512 and self.swiglu_limit > 0 and shared_in_fits(
            self._configs[self._bucket_of(tokens)], tokens, self._s1_num_cu, self.model_dim, self.inter_dim
        ) and self._shared_ksplit(tokens) != 0

    def shared_in_prepare(self, shared) -> bool:
        """Build what a later capture of this layer needs (the buffers, and the dispatch table of the bound W2 with
        ``shared``'s weights) if it is not there yet. Called on eager calls of any route and row count, so that the
        layer's first captured small-input call can fuse; a no-op under a capture, for weights the kernel cannot
        read, and for a runtime none of whose buckets takes the shared expert."""
        if torch.cuda.is_current_stream_capturing() or self.mtpr > 512 or self.swiglu_limit <= 0:
            return False
        if not shared.weights_fit(self.model_dim, self.inter_dim):
            return False
        if getattr(self, "_sh_any_bucket", None) is None:
            self._sh_any_bucket = any(
                self.shared_in_supported(t) for t in range(1, self.mtpr + 1))
        if not self._sh_any_bucket:
            return False
        self.shared_dispatch_table(shared)
        return True

    def shared_in_ready(self, tokens: int, shared) -> bool:
        """``shared_in_supported`` and ``shared`` (a weights.SharedExpertInput) is what the kernel reads; under a
        graph capture also that this layer's table and the buffers were built by an eager call before."""
        if not (self.shared_in_supported(tokens) and shared.fits(tokens, self.model_dim, self.inter_dim)):
            return False
        return not torch.cuda.is_current_stream_capturing() or (
            getattr(self, "_sh_workspace", None) is not None and self._shared_key(shared) in self._sh_disp_by_key)

    def _shared_key(self, shared):
        return tuple(t.data_ptr() for t in (
            self.w2, self.w2_scale, shared.gate_up, shared.gate_up_scale, shared.down, shared.down_scale))

    def _shared_workspace(self):
        """The activation rows between the shared expert's two GEMMs, their scales (tile layout) and the counters."""
        workspace = getattr(self, "_sh_workspace", None)
        if workspace is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the shared_in buffers were not built before graph capture")
            from .shared_expert import PARTIAL_UNIT_BYTES

            tiles = ceildiv(self.mtpr, 32)
            workspace = self._sh_workspace = {
                "act": torch.zeros(tiles * 32 * self.inter_dim, dtype=torch.uint8, device=self.dev),
                "act_scale": torch.zeros(tiles * self.inter_dim + 64, dtype=torch.uint8, device=self.dev),
                "sync": torch.zeros(_SH_SYNC_WORDS, dtype=torch.int32, device=self.dev),
                # shared_in_opt bit 1: the first-half partials of the GEMM1 units of 2 m-tiles (33..64 rows).
                "part": torch.zeros(2 * ((2 * self.inter_dim) // 256) * PARTIAL_UNIT_BYTES, dtype=torch.uint8,
                                    device=self.dev),
            }
            self._sh_disp_by_key = {}
        return workspace

    def shared_dispatch_table(self, shared):
        """The dispatch table of the bound W2 with this shared expert's weights, built on first (eager) use."""
        workspace = self._shared_workspace()
        key = self._shared_key(shared)
        disp = self._sh_disp_by_key.get(key)
        if disp is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("the shared_in dispatch table of these weights was not built before graph capture")
            table = list(self._s1_disp_template)
            table[DispatchSlot.G2_BQ], table[DispatchSlot.G2_BSCALE] = key[0], key[1]
            table[DispatchSlot.SH_GU_W], table[DispatchSlot.SH_GU_S] = key[2], key[3]
            table[DispatchSlot.SH_DN_W], table[DispatchSlot.SH_DN_S] = key[4], key[5]
            table[DispatchSlot.SH_ACT] = workspace["act"].data_ptr()
            table[DispatchSlot.SH_ACT_SCALE] = workspace["act_scale"].data_ptr()
            table[DispatchSlot.SH_SYNC] = workspace["sync"].data_ptr()
            table[DispatchSlot.SH_PART] = workspace["part"].data_ptr()
            # The table keeps the weights it points at alive (a captured graph replays these addresses).
            disp = torch.tensor(table, dtype=torch.int64, device=self.dev)
            self._sh_disp_by_key[key] = disp
            self._sh_keepalive = getattr(self, "_sh_keepalive", [])
            self._sh_keepalive.append((shared.gate_up, shared.gate_up_scale, shared.down, shared.down_scale))
        return disp

    def _shared_in_ksplits(self, bucket: int):
        """The GEMM1 K splits (shared_expert.dense_gate_up_ksplit) of the row counts that select ``bucket`` and
        whose shared expert its kernel can run."""
        return sorted({
            self._shared_ksplit(t) for t in range(1, self.mtpr + 1)
            if self._bucket_of(t) == bucket and self.shared_in_supported(t)
        })

    def _shared_in_variants(self, bucket: int):
        """The (K split, shared_in_opt) pairs of the shared_in kernels ``bucket``'s calls launch, in the order the
        bundle preloads them. Without the option bits that depend on a call's m-tile count (4): one per K split,
        with the option of the split's first row count, as before those bits existed; with them: every pair a row
        count of the bucket takes."""
        rows = [t for t in range(1, self.mtpr + 1) if self._bucket_of(t) == bucket and self.shared_in_supported(t)]
        opt = getattr(self, "shared_in_opt", None)
        if not (shared_in_opt_env() if opt is None else int(opt)) & 4:
            return [(ksplit, self._shared_opt(next(t for t in rows if self._shared_ksplit(t) == ksplit)))
                    for ksplit in self._shared_in_ksplits(bucket)]
        return sorted({(self._shared_ksplit(t), self._shared_opt(t)) for t in rows})

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
            if os.environ.get("AITER_MEGA_SHARED_ADD", "0") == "1":
                # The shared_add variant of the same bucket (the address is not dereferenced by a preload).
                self._launch(config, 1, x_bf16, topk_ids, wts, stream, preload=True, shared_add=x_bf16)
        if shared_in_layout_enabled(self.world_size, self.epr, self.model_dim, self.inter_dim) and self.swiglu_limit > 0:
            # The shared_in variants: per bucket whose kernel fits it, the K splits its row counts take. (Buckets
            # that share a launch config share the kernel; the compile is cached.)
            stand_in = torch.zeros(16, dtype=torch.uint8, device=self.dev)
            for bucket in (b for b in FUSED_TOKEN_BUCKETS if b <= self.mtpr):
                config = self._select_config(bucket)
                for ksplit, opt in self._shared_in_variants(bucket):
                    self._launch(config, 1, x_bf16, topk_ids, wts, stream, preload=True, shared_add=x_bf16,
                        shared_in=(ksplit, stand_in, stand_in, self._s1_disp), shared_in_opt=opt)

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True, shared_add=None, shared_in=None):
        """``shared_add``: a contiguous BF16 ``[rows, model_dim]`` tensor of this call's rows (the shared expert's
        output). The combine adds each row's routed sum to it in place and it is returned as the output: bit for
        bit ``forward(...) + shared_add``, without the separate add launch.

        ``shared_in`` (instead of ``shared_add``): a weights.SharedExpertInput for which ``shared_in_ready`` holds.
        The kernel computes the shared expert's output itself and the returned tensor is ``routed + shared``, as
        above, without the shared expert's three launches."""
        run_tokens = check_forward_inputs(
            x_bf16, wts, topk_ids, max_rows=self.mtpr, model_dim=self.model_dim, topk=self.topk, device=self.dev
        )
        fused_shared, fused_opt = None, 0
        if shared_in is not None:
            if shared_add is not None:
                raise ValueError("shared_in computes the shared output: pass either shared_add or shared_in")
            # The kernel writes the shared expert's rows here, then its combine adds the routed sum to them.
            shared_add = torch.empty((run_tokens, self.model_dim), dtype=torch.bfloat16, device=x_bf16.device)
            if run_tokens:
                if not self.shared_in_ready(run_tokens, shared_in):
                    raise ValueError("shared_in: this call cannot fuse the shared expert (see shared_in_ready)")
                fused_shared = (self._shared_ksplit(run_tokens), _arg_view(shared_in.x_fp8),
                    _arg_view(shared_in.x_scale), self.shared_dispatch_table(shared_in))
                fused_opt = self._shared_opt(run_tokens)
        if shared_add is not None:
            if (shared_add.dtype != torch.bfloat16 or tuple(shared_add.shape) != (run_tokens, self.model_dim)
                    or not shared_add.is_contiguous() or shared_add.device != x_bf16.device):
                raise ValueError("shared_add must be a contiguous BF16 [rows, model_dim] tensor on the call's device")
        caller_out = shared_add
        if run_tokens == 0:
            # An idle rank still takes part in the launch (no rows, so nothing to add).
            x_bf16, wts, topk_ids = self._idle_inputs
            shared_add = None
        if stream is None:
            stream = fx.Stream(torch.cuda.current_stream().cuda_stream)
        elif hasattr(stream, "cuda_stream"):
            stream = fx.Stream(stream.cuda_stream)
        # Weights may have been reassigned since the last call (see MegaMoEV2's weight attributes).
        self._s1_disp = self.w2_dispatch_table(self.w2, self.w2_scale)
        # Ranks may use different token buckets in one call; an idle rank uses the 1-token bucket.
        config = self._select_config(max(run_tokens, 1))
        self._launch(config, run_tokens, x_bf16, topk_ids, wts, stream, shared_add=shared_add, shared_in=fused_shared,
                     **({"shared_in_opt": fused_opt} if fused_opt else {}))
        if caller_out is not None:
            return caller_out
        out_tok = self.comb_op.combine_output(self.mtpr)
        return out_tok[:run_tokens] if slice_output else out_tok

    __call__ = forward

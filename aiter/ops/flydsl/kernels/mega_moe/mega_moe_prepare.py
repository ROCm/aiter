# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Low-resource compact-dispatch prepare kernel for MegaMoE Stage1."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm  # raw asm: no fx wrapper (see _kcache_touch)
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from .. import communication_ops_utils as comm_ops
from ..dpp_utils import dpp_xor_f32, update_dpp_i32
from ..tensor_shim import (
    _preload_compiled,
    _run_compiled,
    buf_copy_load,
    buf_copy_store,
    buf_scalar_load,
    ptr_buf_tensor,
)
from .dispatch import (
    DISPATCH_TABLE_SIZE,
    DispatchSlot,
    _classify_fanout_wave_route,
    _wave_inclusive_scan_i32,
    emit_dispatch_group,
    emit_dispatch_plan,
)
from .quant import emit_per_1x32_mx_fp4_group, emit_per_1x32_mx_fp8_group

# fp32 bits of 1/448 (FP8 E4M3 max): identical to quant.py's RoundUp scale.
_FP8_E4M3_INV_MAX_POS_BITS = 0x3B124925
# Coalesced MXFP8 quant for the large-token buckets (sort_block_m >= 128):
# each lane owns one 16-byte bf16 chunk and keeps _QUANT_CHUNKS_IN_FLIGHT
# chunk loads outstanding before converting any of them.
_QUANT_CHUNKS_IN_FLIGHT = 16
# Largest prepare-group count compiled onto the fused single-CTA path.
_FUSED_MAX_P = 16
_QUANT_DPP_MIN_TILE_M = 128
# Quant CTAs for the _qdpp (tile_m >= 128) buckets.  With the short owner
# chain, quant is the m128 kernel tail, so compile raises the configured
# quant CTA count to this value per compile key (the grid stays co-resident).
_QUANT_CU_M128 = 216
# m64 P32 bucket (2048-token compile bucket, 1536-3071 live tokens): from
# _QDPP_M64_MIN_TOK live tokens on, quant is the tail, so it takes the same
# coalesced/pipelined _qdpp quant + work-steal recipe as m128 at a widened Q.
# Smaller token counts in this bucket (and P16) keep the per-lane quant; the
# host picks the variant from cur_tok.
_QDPP_M64_P = 32
_QDPP_M64_MIN_TOK = 2048
_QUANT_CU_M64 = 80
# Dynamic quant work-stealing for the _qdpp buckets: the last
# _QUANT_STEAL_ROUNDS block rounds (Q blocks each) are handed out through an
# agent-scope atomic counter instead of the static quant_slot stride, so the
# CTAs that drain early take the tail (a static partition leaves a wide
# first-to-last quant CTA spread at m128).  The counter lives in
# PREP_EPOCH_GATE slot (P = 0, Q = P): every launch asserts P >= 1, so row 0
# is never an epoch gate and no (P, Q) launch can alias the counter.  It
# returns to 0 at the end of every launch (the last grab subtracts the
# launch's total), so no host memset is needed.  Block -> output mapping is
# unchanged, so the output stays bit-identical.
_QUANT_STEAL_ROUNDS = 2


def _emit_mx_fp8_quad_chunk(raw, chunk, lane, out_buf8):
    """Quantize one 16-byte (8 x bf16) chunk; 4 adjacent lanes form a 1x32 group.

    Bit-identical to quant.emit_per_1x32_mx_fp8_group: the group amax is the
    max over the same 32 |values| plus the 1e-10 floor, reduced across the
    lane quad with DPP xor-1/xor-2; E8M0 rounding and the FP8 conversion are
    unchanged.  The 8 FP8 bytes go out as one dwordx2 store.  Returns the 4
    E8M0 scales of this 16-lane DPP row packed into one dword, valid in lane
    12 of the row (the caller stores it from that lane).
    """
    values = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
    local_max = fx.Float32(1e-10).maximumf(fmath.absf(values).reduce(ReductionOp.MAX))
    local_max = local_max.maximumf(dpp_xor_f32(local_max, 1))
    local_max = local_max.maximumf(dpp_xor_f32(local_max, 2))
    working = (
        local_max * fx.Int32(_FP8_E4M3_INV_MAX_POS_BITS).bitcast(fx.Float32)
    ).bitcast(fx.Int32)
    mantissa = working & fx.Int32(0x7FFFFF)
    biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
    e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
    quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
    scaled = [values[k] * quant_scale for k in range_constexpr(8)]
    words = []
    for word in range_constexpr(2):
        base = word * 4
        packed = rocdl.cvt_pk_fp8_f32(
            T.i32, scaled[base], scaled[base + 1], fx.Int32(0), 0
        )
        packed = rocdl.cvt_pk_fp8_f32(
            T.i32, scaled[base + 2], scaled[base + 3], packed, 1
        )
        words.append(packed)
    buf_copy_store(
        out_buf8,
        chunk,
        fx.Vector.from_elements(words, fx.Int32),
        fx.Int32,
        unit_elems=2,
    )
    # Lane q*4+j puts its scale byte at bit 8*q; row_shr:4 and row_shr:8
    # OR-reduce the four bytes into lane 12.  total_chunks is a multiple of 16
    # (model_dim % 128 == 0), so a DPP row is either fully active or inactive.
    sword = e8m0 << ((lane & fx.Int32(12)) << fx.Int32(1))
    sword = sword | fx.Int32(update_dpp_i32(fx.Int32(0), sword, 0x114, 0xF, 0xF, True))
    sword = sword | fx.Int32(update_dpp_i32(fx.Int32(0), sword, 0x118, 0xF, 0xF, True))
    return sword


# Software-pipelined quant stream (_qdpp buckets): while block k is converted,
# chunk u of block k+1 is loaded into the register slot chunk u just freed,
# so _QUANT_CHUNKS_IN_FLIGHT loads stay outstanding across block boundaries
# (instead of draining to vmcnt(0) at every block end).  The per-chunk
# conditional scale store is replaced by one all-lane dword store per block
# (lane r of a 16-lane row stores chunk r's packed scale word), which keeps
# the u-loop branch-free so LLVM's in-order vmcnt accounting stays exact
# instead of waiting on FP8 stores.
# Chunk index (within a block) at which the work-steal grab for block j+1
# (issued at the start of block j) is published by a CTA barrier; next-block
# loads for chunks below it are deferred to that point.
_QUANT_GRAB_AT = 4


def _mx_fp8_quad_convert(raw, chunk, out_buf8):
    """Quantize one 16 B chunk (same math as _emit_mx_fp8_quad_chunk).

    Stores the 8 FP8 bytes and returns this lane's E8M0 byte (the 4 lanes of
    a quad hold the same value).
    """
    values = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
    local_max = fx.Float32(1e-10).maximumf(fmath.absf(values).reduce(ReductionOp.MAX))
    local_max = local_max.maximumf(dpp_xor_f32(local_max, 1))
    local_max = local_max.maximumf(dpp_xor_f32(local_max, 2))
    working = (
        local_max * fx.Int32(_FP8_E4M3_INV_MAX_POS_BITS).bitcast(fx.Float32)
    ).bitcast(fx.Int32)
    mantissa = working & fx.Int32(0x7FFFFF)
    biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
    e8m0 = (mantissa != fx.Int32(0)).select(biased_exp + fx.Int32(1), biased_exp)
    e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
    quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(fx.Float32)
    scaled = [values[k] * quant_scale for k in range_constexpr(8)]
    words = []
    for word in range_constexpr(2):
        base = word * 4
        packed = rocdl.cvt_pk_fp8_f32(
            T.i32, scaled[base], scaled[base + 1], fx.Int32(0), 0
        )
        packed = rocdl.cvt_pk_fp8_f32(
            T.i32, scaled[base + 2], scaled[base + 3], packed, 1
        )
        words.append(packed)
    buf_copy_store(
        out_buf8,
        chunk,
        fx.Vector.from_elements(words, fx.Int32),
        fx.Int32,
        unit_elems=2,
    )
    return e8m0


def _row_scale_word(e8m0, lane):
    """Pack the 4 quad E8M0 bytes of a 16-lane row into one dword, in ALL lanes."""
    sword = e8m0 << ((lane & fx.Int32(12)) << fx.Int32(1))
    sword = sword | fx.Int32(update_dpp_i32(sword, sword, 0x124, 0xF, 0xF, False))
    sword = sword | fx.Int32(update_dpp_i32(sword, sword, 0x128, 0xF, 0xF, False))
    return sword


def _idx_bytes(i32_cur_tok, fz_k):
    """Byte extent of the live top-k rows.  Bounding the V# by it makes every
    clamped/speculative index load of a zero-token rank (whose ``topk_ids``
    is empty, data_ptr 0) a hardware OOB zero-fill instead of a fault."""
    return i32_cur_tok * fx.Int32(int(fz_k) * 4)


def _sload_table_i64(addr_table, index):
    """``table[index]`` of an i64 address table via two ``s_buffer_load``.

    The dispatch table and the P2P peer tables are host-written constants.
    Loading them with a vector buffer load leaves every derived buffer
    descriptor in VGPRs, so each later access through it becomes a
    readfirstlane waterfall loop.  Scalar loads keep descriptors in SGPRs.
    ``index`` must be wave-uniform.
    """
    table = ptr_buf_tensor(addr_table, fx.Int32)
    dword = fx.Int32(index) * fx.Int32(2)
    lo = fx.Int32(buf_scalar_load(table, dword))
    hi = fx.Int32(buf_scalar_load(table, dword + fx.Int32(1)))
    return (fx.Int64(lo) & fx.Int64(0xFFFFFFFF)) | (fx.Int64(hi) << fx.Int64(32))


def _store_i32_monotonic(addr_i64, offset, val, syncscope):
    """Monotonic i32 store at ``addr_i64 + offset*4`` in ``syncscope``."""
    off = arith.unwrap(offset)
    off64 = fx.Uint64(fx.Uint32(off)) if off.type == T.i32 else fx.Int64(off)
    addr = fx.Int64(addr_i64) + fx.Int64(off64) * 4
    fx.generic_store(
        comm_ops._to_ptr_global(addr),
        fx.Int32(val),
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=syncscope,
    )


def _store_i32_agent_relaxed(addr_i64, offset, val):
    """Agent-scope MONOTONIC i32 store at ``addr_i64 + offset*4``.

    On gfx95x this is a ``global_store ... sc1``: the store itself is coherent
    across every XCD L2 of the agent, so publishing it needs no
    ``buffer_wbl2 sc1``.  An agent-scope release fence would write back the
    WHOLE dirty L2 of the issuing XCD, which at m128 is mostly the quant
    CTAs' FP8 output (several us per fence under quant traffic).
    Publish with ``s_waitcnt vmcnt(0)`` (store completion at agent
    coherence, the same wait the release lowering relies on) followed by the
    flag/counter atomic.
    """
    _store_i32_monotonic(addr_i64, offset, val, fx.rocdl.SyncScope.AgentOneAs)


# Dispatch-table scalar cache warm-up.  Every ``dp(slot)`` is an
# s_buffer_load that LLVM sinks to its first use, so each plan phase stalls
# on a cold scalar-cache line.  One helper warp touches every 64-byte line of
# the table while the owner is idle in its prologue barrier, so the later
# per-phase loads hit.
# Raw llvm.InlineAsmOp / llvm.inline_asm are used only here and for the
# quant-steal opaque zero: fx has no wrapper for an operand-only
# side-effecting asm, which is what keeps the scalar loads alive.


def _kcache_touch(addr_table, num_dwords):
    """Load one dword of every 64-byte line of ``addr_table[0:num_dwords]``
    and consume them in a side-effecting asm (forces the loads to land in
    the scalar cache here; the result itself is discarded)."""
    table = ptr_buf_tensor(addr_table, fx.Int32)
    offsets = list(range(0, num_dwords, 16))
    if offsets[-1] != num_dwords - 1:
        offsets.append(num_dwords - 1)
    vals = [
        arith.unwrap(fx.Int32(buf_scalar_load(table, fx.Int32(dw)))) for dw in offsets
    ]
    llvm.InlineAsmOp(None, vals, "", ",".join(["s"] * len(vals)), has_side_effects=True)


def _kcache_touch_tables(addr_tables):
    """_kcache_touch of the first 64-byte line of each table, all loads
    issued before the single consuming asm (one scalar-cache round trip)."""
    vals = []
    for addr_table in addr_tables:
        table = ptr_buf_tensor(addr_table, fx.Int32)
        for dw in (0, 15):
            vals.append(arith.unwrap(fx.Int32(buf_scalar_load(table, fx.Int32(dw)))))
    llvm.InlineAsmOp(None, vals, "", ",".join(["s"] * len(vals)), has_side_effects=True)


def _store_i32_sys_relaxed(addr_i64, offset, val):
    """System-scope MONOTONIC i32 store at ``addr_i64 + offset*4``.

    Same coherence domain as ``comm_ops.store_i32_system`` (one-as, so the
    store still bypasses the non-coherent caches), but without the per-store
    release (an L2 writeback on gfx95x).  Callers publish the data with ONE
    release fence before the flag that a consumer acquires.
    """
    _store_i32_monotonic(addr_i64, offset, val, fx.rocdl.SyncScope.OneAs)


# Write-through (sc0 sc1) cache policy for buffer stores of the owner's
# published plan.  On gfx95x a system-scope release fence lowers to
# ``buffer_wbl2 sc0 sc1``, which writes back EVERY dirty line of the issuing
# XCD's L2 (at m128 mostly the quant CTAs' FP8 output).  Every store the
# owner publishes behind COUNT/PLAN/PAIR flags is instead written through to
# the system coherence point by itself (same coherence as a system-scope
# monotonic store), so the flag only needs ``s_waitcnt vmcnt(0)`` +
# ``s_barrier`` ahead of it -- no L2 writeback on the owner chain.
_WT_CM = 17  # CPol SC0 | SC1
# ``wt_owner`` is a compile-time Python bool fixed per kernel variant by
# compile_mega_moe_prepare and passed explicitly down to every helper that
# stores part of the published plan (never module state: FlyDSL folds the
# module globals a launch references into its cache key).  Off: plain (L2
# write-back) stores + release fences.
# Large fused plans (m64 pcu4/pcu8, decode 192-452) are store-issue bound:
# thousands of write-through pair-order/metadata stores cost more than one
# buffer_wbl2, so they keep the fence.
_WT_MAX_FUSED_P = 2


def _wst(buffer, index, value, *, wt_owner):
    """``buffer[index] = value`` (i32), written through to system scope when
    the owner publishes without a release fence (``wt_owner``)."""
    if const_expr(wt_owner):
        buf_copy_store(buffer, index, value, fx.Int32, cache_modifier=_WT_CM)
    else:
        buffer[index] = value


def _wst4(buffer4, index, vec, *, wt_owner):
    """dwordx4 ``buffer4[index] = vec``; see _wst."""
    buf_copy_store(
        buffer4,
        index,
        vec,
        fx.Int32,
        unit_elems=4,
        cache_modifier=_WT_CM if wt_owner else 0,
    )


@flyc.jit
def _dpp_max_i32(value, ctrl):
    remote = fx.rocdl.update_dpp(
        T.i32, value.ir_value(), value.ir_value(), ctrl, 0xF, 0xF, False
    )
    remote = fx.Int32(remote)
    return (remote > value).select(remote, value)


@flyc.jit
def _row8_max_i32(value):
    """Max over each aligned group of 8 lanes (all lanes get the result)."""
    value = _dpp_max_i32(value, 0xB1)  # quad_perm [1,0,3,2]
    value = _dpp_max_i32(value, 0x4E)  # quad_perm [2,3,0,1]
    return _dpp_max_i32(value, 0x141)  # row_half_mirror


@flyc.jit
def _wave_max_uniform_i32(value, *, active_lanes=64):
    """Full-wave max as a uniform value (DPP + readlane, no LDS permutes)."""
    value = _row8_max_i32(value)
    value = _dpp_max_i32(value, 0x140)  # row_mirror -> 16-lane rows
    result = fx.Int32(fx.rocdl.readlane(T.i32, value, 0))
    for row in range_constexpr(1, (active_lanes + 15) // 16):
        other = fx.Int32(fx.rocdl.readlane(T.i32, value, row * 16))
        result = (other > result).select(other, result)
    return result


@flyc.jit
def _configure_payload_geometry_fast(
    addr_local_hist,
    addr_chunk_counts,
    addr_block_counts,
    lane,
    *,
    fz_npes,
    fz_epr,
    fz_total_experts,
    payload_chunk_rows,
    dispatch_blocks,
    lds_hist=None,
    wt_owner,
):
    """dispatch._configure_payload_geometry, all destinations in one pass.

    Lane ``8*d + s`` scans experts ``s, s+8, ...`` of destination ``d``; one
    8-lane DPP max replaces eight serial full-wave LDS-permute reductions and
    every histogram load is issued before the first use.
    """
    assert fz_npes <= 8
    local_hist = ptr_buf_tensor(addr_local_hist, fx.Int32)
    chunk_counts = ptr_buf_tensor(addr_chunk_counts, fx.Int32)
    block_counts = ptr_buf_tensor(addr_block_counts, fx.Int32)
    destination = lane >> fx.Int32(3)
    sub = lane & fx.Int32(7)
    active = destination < fx.Int32(fz_npes)
    safe_destination = active.select(destination, fx.Int32(0))
    values = []
    for k in range_constexpr((fz_epr + 7) // 8):
        expert = sub + fx.Int32(8 * k)
        valid = active & (expert < fx.Int32(fz_epr))
        safe_expert = valid.select(expert, fx.Int32(0))
        hist_index = safe_destination * fx.Int32(fz_epr) + safe_expert
        if const_expr(lds_hist is not None):
            loaded = fx.Int32(fx.ptr_load(lds_hist + fx.Int64(hist_index)))
        else:
            loaded = local_hist[hist_index]
        values.append((valid, loaded))
    if const_expr(lds_hist is not None):
        group_count = fx.Int32(
            fx.ptr_load(
                lds_hist + fx.Int64(fx.Int32(fz_total_experts) + safe_destination)
            )
        )
    else:
        group_count = local_hist[fx.Int32(fz_total_experts) + safe_destination]
    max_source_count = fx.Int32(0)
    for valid, loaded in values:
        loaded = valid.select(loaded, fx.Int32(0))
        max_source_count = (loaded > max_source_count).select(loaded, max_source_count)
    max_source_count = _row8_max_i32(max_source_count)
    max_source_count = (group_count > max_source_count).select(
        group_count, max_source_count
    )
    if active & (sub == fx.Int32(0)):
        chunks = (max_source_count + fx.Int32(payload_chunk_rows - 1)) // fx.Int32(
            payload_chunk_rows
        )
        chunks = (chunks > fx.Int32(0)).select(chunks, fx.Int32(1))
        _wst(chunk_counts, destination, chunks, wt_owner=wt_owner)
        _wst(
            block_counts,
            destination,
            fx.Int32(dispatch_blocks // fz_npes),
            wt_owner=wt_owner,
        )


# Flattened tile metadata (lane = tile) + dwordx4 SRCMAP padding.
# Tagged (LL-style) histogram exchange: every pushed count carries the launch
# tag, so each consumer polls exactly the words it needs instead of waiting
# on a COUNT_DONE counter plus a system release/acquire pair.


_GATE_EPOCH_MASK = 0x3FFFFFFF
_GATE_PARITY_SHIFT = 30


@flyc.jit
def _spin_epoch_gate(addr_i64, epoch_masked):
    """Spin until the gate's epoch field equals ``epoch_masked``.

    The owner publishes ``epoch | parity << 30`` with one system release
    store, so the group CTAs get the new parity from the very load that
    observes the gate (no extra acquire + dependent parity load).
    """
    cur = fx.Int32(comm_ops.load_i32_acquire(addr_i64))
    while (cur & fx.Int32(_GATE_EPOCH_MASK)) != epoch_masked:
        cur = fx.Int32(comm_ops.load_i32_acquire(addr_i64))
    return cur


def _stepped_hits(source_counts, *, fz_npes, payload_chunk_rows, fz_tile_m):
    step = payload_chunk_rows // fz_tile_m
    never = fx.Int32(0x3FFFFFFF)
    next_hits = []
    last_hits = []
    running = fx.Int32(0)
    for source in range_constexpr(fz_npes):
        count = source_counts[source]
        start = running
        running = running + count
        hits = (count > fx.Int32(0)) & (
            (start & fx.Int32(fz_tile_m - 1)) != fx.Int32(0)
        )
        first_tile = start // fx.Int32(fz_tile_m)
        last_k = (count - fx.Int32(1)) // fx.Int32(payload_chunk_rows)
        next_hits.append(hits.select(first_tile, never))
        last_hits.append(first_tile + last_k * fx.Int32(step))
    return next_hits, last_hits


@flyc.jit
def _init_sections_ready_merged(
    addr_tile_ready,
    addr_tile_expected,
    group_source_counts,
    normal_source_counts,
    row_base,
    group_num_tiles,
    normal_num_tiles,
    *,
    fz_npes,
    payload_chunk_rows,
    fz_tile_m,
    wt_owner,
):
    """TILE_READY / TILE_EXPECTED init of the group section followed by the
    normal section in ONE tile loop (the sections are contiguous: the
    normal section starts group_num_tiles tiles after row_base).  Exactly
    the same TILE_READY / TILE_EXPECTED values; the wave runs
    max(group + normal) iterations instead of max(group) + max(normal)."""
    assert payload_chunk_rows % fz_tile_m == 0
    assert fz_tile_m & (fz_tile_m - 1) == 0
    step = payload_chunk_rows // fz_tile_m
    tile_expected = ptr_buf_tensor(addr_tile_expected, fx.Int32)
    base_tile = row_base // fx.Int32(fz_tile_m)
    never = fx.Int32(0x3FFFFFFF)
    g_next, g_last = _stepped_hits(
        group_source_counts,
        fz_npes=fz_npes,
        payload_chunk_rows=payload_chunk_rows,
        fz_tile_m=fz_tile_m,
    )
    n_next, n_last = _stepped_hits(
        normal_source_counts,
        fz_npes=fz_npes,
        payload_chunk_rows=payload_chunk_rows,
        fz_tile_m=fz_tile_m,
    )
    total_tiles = group_num_tiles + normal_num_tiles
    for tile in range(fx.Int32(0), total_tiles, 1):
        tile_index = base_tile + tile
        _store_i32_sys_relaxed(addr_tile_ready, tile_index, fx.Int32(0))
        in_group = tile < group_num_tiles
        in_normal = tile >= group_num_tiles
        rel = in_group.select(tile, tile - group_num_tiles)
        expected_arrivals = fx.Int32(1)
        for source in range_constexpr(fz_npes):
            hit = in_group & (rel == g_next[source])
            expected_arrivals = expected_arrivals + hit.select(fx.Int32(1), fx.Int32(0))
            advanced = g_next[source] + fx.Int32(step)
            advanced = (advanced > g_last[source]).select(never, advanced)
            g_next[source] = hit.select(advanced, g_next[source])
        for source in range_constexpr(fz_npes):
            hit = in_normal & (rel == n_next[source])
            expected_arrivals = expected_arrivals + hit.select(fx.Int32(1), fx.Int32(0))
            advanced = n_next[source] + fx.Int32(step)
            advanced = (advanced > n_last[source]).select(never, advanced)
            n_next[source] = hit.select(advanced, n_next[source])
        _wst(tile_expected, tile_index, expected_arrivals, wt_owner=wt_owner)


@flyc.jit
def _ll_dest_issue(
    addr_count_matrix,
    addr_count_done,
    addr_pair_config,
    parity,
    destination,
    lane,
    *,
    rank,
    npes,
    epr,
    total_experts,
    total_segments,
):
    """Issue the tagged loads of one destination in few wave instructions.

    meta word: lanes 0..npes-1 load source=lane's group count (tagged),
    lane npes loads the destination's tile_m (tagged), every other lane the
    current fanout pair (untagged).  Normal counts: with epr <= 32 one
    instruction covers two sources (lane half h -> source 2k + h).
    """
    src_lane = lane & fx.Int32(npes - 1)
    own = destination == fx.Int32(rank)
    group_addr = own.select(
        addr_count_done + fx.Int64(src_lane) * fx.Int64(4),
        addr_count_matrix
        + fx.Int64(
            src_lane * fx.Int32(total_segments) + fx.Int32(total_experts) + destination
        )
        * fx.Int64(4),
    )
    tile_addr = addr_count_done + fx.Int64(fx.Int32(npes) + destination) * fx.Int64(4)
    pair_addr = addr_pair_config + fx.Int64(
        parity * fx.Int32(npes) + destination
    ) * fx.Int64(4)
    meta_addr = (lane < fx.Int32(npes)).select(
        group_addr, (lane == fx.Int32(npes)).select(tile_addr, pair_addr)
    )
    meta_val = fx.Int32(comm_ops.load_i32_acquire(meta_addr))
    split = 2 if epr <= 32 else 1
    if const_expr(split == 2):
        expert = lane & fx.Int32(31)
        half = lane >> fx.Int32(5)
    else:
        expert = lane
        half = fx.Int32(0)
    safe_expert = (expert < fx.Int32(epr)).select(expert, fx.Int32(0))
    ge = destination * fx.Int32(epr) + safe_expert
    count_addrs = []
    count_vals = []
    for k in range_constexpr(npes // split):
        source = fx.Int32(split * k) + half
        addr = addr_count_matrix + fx.Int64(
            source * fx.Int32(total_segments) + ge
        ) * fx.Int64(4)
        count_addrs.append(addr)
        count_vals.append(fx.Int32(comm_ops.load_i32_acquire(addr)))
    return meta_addr, meta_val, count_addrs, count_vals


@flyc.jit
def _ll_dest_finish(
    meta_addr,
    meta_val,
    count_addrs,
    count_vals,
    lane,
    cur_tag,
    *,
    rank,
    npes,
    epr,
    tag_shift,
):
    """Poll the words issued by _ll_dest_issue and reduce them.

    Returns per-lane (lane = expert) normal count and source<rank prefix
    (zero on lanes >= epr), the uniform group count / group source prefix,
    tile_m and the raw current fanout-pair word.
    """
    value_mask = fx.Int32((1 << tag_shift) - 1)
    tagged = lane <= fx.Int32(npes)
    v = meta_val
    while tagged & ((v >> fx.Int32(tag_shift)) != cur_tag):
        v = fx.Int32(comm_ops.load_i32_acquire(meta_addr))
    meta_masked = v & value_mask
    group_count = fx.Int32(0)
    group_prefix = fx.Int32(0)
    for source in range_constexpr(npes):
        gc = fx.Int32(fx.rocdl.readlane(T.i32, meta_masked, source))
        if const_expr(source < rank):
            group_prefix = group_prefix + gc
        group_count = group_count + gc
    tile_m = fx.Int32(fx.rocdl.readlane(T.i32, meta_masked, npes))
    packed = fx.Int32(fx.rocdl.readlane(T.i32, v, npes + 1))
    split = 2 if epr <= 32 else 1
    if const_expr(split == 2):
        half = lane >> fx.Int32(5)
    else:
        half = fx.Int32(0)
    part_count = fx.Int32(0)
    part_prefix = fx.Int32(0)
    for k in range_constexpr(npes // split):
        c = count_vals[k]
        while (c >> fx.Int32(tag_shift)) != cur_tag:
            c = fx.Int32(comm_ops.load_i32_acquire(count_addrs[k]))
        c = c & value_mask
        source = fx.Int32(split * k) + half
        part_count = part_count + c
        part_prefix = part_prefix + (source < fx.Int32(rank)).select(c, fx.Int32(0))
    if const_expr(split == 2):
        peer = (lane ^ fx.Int32(32)) * fx.Int32(4)
        part_count = part_count + fx.Int32(
            fx.rocdl.ds_bpermute(T.i32, peer, part_count)
        )
        part_prefix = part_prefix + fx.Int32(
            fx.rocdl.ds_bpermute(T.i32, peer, part_prefix)
        )
    valid_expert = lane < fx.Int32(epr)
    normal_count = valid_expert.select(part_count, fx.Int32(0))
    normal_prefix = valid_expert.select(part_prefix, fx.Int32(0))
    return normal_count, normal_prefix, group_count, group_prefix, tile_m, packed


# Decode (tile_m == 32) destination warps:
#  T: one-pass top-2 DPP reduction for the next-launch fanout pair, and
#     power-of-two tile rounding (sort_block_m is 32/64/128) instead of a
#     runtime integer division by the destination's tile_m.
#  F: a warp with two destinations issues the second destination's words
#     only after the first one's words all carried the current tag, so they
#     are read fresh while the first destination is computed/stored (the
#     up-front copies predate the peers' push and every stale word cost one
#     serial re-poll queued behind the first destination's stores).


# Warp 0 (planner) issues its own-destination words with one meta load
# (tagged group counts + current pair word) + npes per-lane count loads.


@flyc.jit
def _ll_own_issue(
    addr_count_matrix,
    addr_count_done,
    addr_pair_config,
    parity,
    lane,
    *,
    rank,
    npes,
    epr,
    total_experts,
    total_segments,
):
    """_ll_dest_issue for the planner's own rank, lane = expert layout.

    meta word: lanes 0..npes-1 load COUNT_DONE[source=lane] (the tagged copy
    of this rank's group column), lane npes the own tile_m word (tagged),
    every other lane the current fanout pair word.  Normal counts: one
    per-lane (lane = local expert) load per source.
    """
    src_lane = lane & fx.Int32(npes - 1)
    group_addr = addr_count_done + fx.Int64(src_lane) * fx.Int64(4)
    tile_addr = addr_count_done + fx.Int64(npes + rank) * fx.Int64(4)
    pair_addr = addr_pair_config + fx.Int64(
        parity * fx.Int32(npes) + fx.Int32(rank)
    ) * fx.Int64(4)
    meta_addr = (lane < fx.Int32(npes)).select(
        group_addr, (lane == fx.Int32(npes)).select(tile_addr, pair_addr)
    )
    meta_val = fx.Int32(comm_ops.load_i32_acquire(meta_addr))
    safe_expert = (lane < fx.Int32(epr)).select(lane, fx.Int32(0))
    ge = fx.Int32(rank * epr) + safe_expert
    count_addrs = []
    count_vals = []
    for source in range_constexpr(npes):
        addr = addr_count_matrix + fx.Int64(
            fx.Int32(source * total_segments) + ge
        ) * fx.Int64(4)
        count_addrs.append(addr)
        count_vals.append(fx.Int32(comm_ops.load_i32_acquire(addr)))
    return meta_addr, meta_val, count_addrs, count_vals


@flyc.jit
def _ll_own_finish(
    meta_addr, meta_val, count_addrs, count_vals, lane, cur_tag, *, npes, tag_shift
):
    """Poll _ll_own_issue's words; returns (pair word, per-source uniform
    group counts, per-source per-lane normal counts)."""
    value_mask = fx.Int32((1 << tag_shift) - 1)
    tagged = lane <= fx.Int32(npes)
    v = meta_val
    while tagged & ((v >> fx.Int32(tag_shift)) != cur_tag):
        v = fx.Int32(comm_ops.load_i32_acquire(meta_addr))
    meta_masked = v & value_mask
    group_counts = []
    for source in range_constexpr(npes):
        group_counts.append(fx.Int32(fx.rocdl.readlane(T.i32, meta_masked, source)))
    packed = fx.Int32(fx.rocdl.readlane(T.i32, v, npes + 1))
    normal_counts = []
    for k in range_constexpr(npes):
        c = count_vals[k]
        while (c >> fx.Int32(tag_shift)) != cur_tag:
            c = fx.Int32(comm_ops.load_i32_acquire(count_addrs[k]))
        normal_counts.append(c & value_mask)
    return packed, group_counts, normal_counts


@flyc.jit
def _top2_step(m1, m2, ctrl):
    """Merge this lane's (max, second max) with its DPP partner's (disjoint
    lane sets for the xor-1/xor-2/half-mirror/mirror steps)."""
    p1 = fx.Int32(
        fx.rocdl.update_dpp(T.i32, m1.ir_value(), m1.ir_value(), ctrl, 0xF, 0xF, False)
    )
    p2 = fx.Int32(
        fx.rocdl.update_dpp(T.i32, m2.ir_value(), m2.ir_value(), ctrl, 0xF, 0xF, False)
    )
    hi = (p1 > m1).select(p1, m1)
    lo = (p1 > m1).select(m1, p1)
    o2 = (p2 > m2).select(p2, m2)
    return hi, (lo > o2).select(lo, o2)


@flyc.jit
def _wave_top2_uniform_i32(value, *, active_lanes=64):
    """(max, second max) over the wave's lanes as uniform values; lane values
    must be distinct except for the -1 filler."""
    m1 = value
    m2 = fx.Int32(-1)
    for ctrl in (0xB1, 0x4E, 0x141, 0x140):
        m1, m2 = _top2_step(m1, m2, ctrl)
    r1 = fx.Int32(fx.rocdl.readlane(T.i32, m1, 0))
    r2 = fx.Int32(fx.rocdl.readlane(T.i32, m2, 0))
    for row in range_constexpr(1, (active_lanes + 15) // 16):
        o1 = fx.Int32(fx.rocdl.readlane(T.i32, m1, row * 16))
        o2 = fx.Int32(fx.rocdl.readlane(T.i32, m2, row * 16))
        lo = (o1 > r1).select(r1, o1)
        r1 = (o1 > r1).select(o1, r1)
        r2 = (o2 > r2).select(o2, r2)
        r2 = (lo > r2).select(lo, r2)
    return r1, r2


@flyc.jit
def _next_pair_packed_top2(packed, normal_count, group_count, lane, *, epr):
    """_next_pair_packed with both maxima from one DPP pass (same result:
    every valid score is distinct, invalid lanes are -1)."""
    score_stride = fx.Int32(epr + 1)
    valid_expert = lane < fx.Int32(epr)
    safe_expert = valid_expert.select(lane, fx.Int32(0))
    enabled_now = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
    cur_a = packed & fx.Int32(0xFF)
    cur_b = (packed >> fx.Int32(8)) & fx.Int32(0xFF)
    cur_mask = (fx.Int64(1) << fx.Int64(cur_a)) | (fx.Int64(1) << fx.Int64(cur_b))
    current_mask = enabled_now.select(cur_mask, fx.Int64(0))
    current_member = (
        (current_mask >> fx.Int64(safe_expert)) & fx.Int64(1)
    ) != fx.Int64(0)
    total_count = normal_count + (valid_expert & current_member).select(
        group_count, fx.Int32(0)
    )
    score = valid_expert.select(
        total_count * score_stride + fx.Int32(epr) - safe_expert,
        fx.Int32(-1),
    )
    best_score, second_score = _wave_top2_uniform_i32(score, active_lanes=epr)
    best_expert = fx.Int32(epr) - (best_score % score_stride)
    second_expert = fx.Int32(epr) - (second_score % score_stride)
    second_count = second_score // score_stride
    pair_a = (best_expert < second_expert).select(best_expert, second_expert)
    pair_b = (best_expert < second_expert).select(second_expert, best_expert)
    enabled = (
        (second_count > fx.Int32(0))
        & (pair_a >= fx.Int32(0))
        & (pair_b < fx.Int32(epr))
        & (pair_a != pair_b)
    )
    return (
        pair_a
        | (pair_b << fx.Int32(8))
        | enabled.select(fx.Int32(1 << 16), fx.Int32(0))
    )


@flyc.jit
def _next_pair_packed(packed, normal_count, group_count, lane, *, epr):
    """dispatch._derive_next_fanout_pairs selection math for one destination
    from already-reduced counts (lane = expert, zero on lanes >= epr)."""
    score_stride = fx.Int32(epr + 1)
    valid_expert = lane < fx.Int32(epr)
    safe_expert = valid_expert.select(lane, fx.Int32(0))
    enabled_now = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
    cur_a = packed & fx.Int32(0xFF)
    cur_b = (packed >> fx.Int32(8)) & fx.Int32(0xFF)
    cur_mask = (fx.Int64(1) << fx.Int64(cur_a)) | (fx.Int64(1) << fx.Int64(cur_b))
    current_mask = enabled_now.select(cur_mask, fx.Int64(0))
    current_member = (
        (current_mask >> fx.Int64(safe_expert)) & fx.Int64(1)
    ) != fx.Int64(0)
    total_count = normal_count + (valid_expert & current_member).select(
        group_count, fx.Int32(0)
    )
    score = valid_expert.select(
        total_count * score_stride + fx.Int32(epr) - safe_expert,
        fx.Int32(-1),
    )
    best_score = _wave_max_uniform_i32(score, active_lanes=epr)
    best_expert = fx.Int32(epr) - (best_score % score_stride)
    second_score = (safe_expert != best_expert).select(score, fx.Int32(-1))
    second_score = _wave_max_uniform_i32(second_score, active_lanes=epr)
    second_expert = fx.Int32(epr) - (second_score % score_stride)
    second_count = second_score // score_stride
    pair_a = (best_expert < second_expert).select(best_expert, second_expert)
    pair_b = (best_expert < second_expert).select(second_expert, best_expert)
    enabled = (
        (second_count > fx.Int32(0))
        & (pair_a >= fx.Int32(0))
        & (pair_b < fx.Int32(epr))
        & (pair_a != pair_b)
    )
    return (
        pair_a
        | (pair_b << fx.Int32(8))
        | enabled.select(fx.Int32(1 << 16), fx.Int32(0))
    )


@flyc.jit
def _derive_offsets_pairs_ll(
    addr_disp,
    parity,
    destinations,
    lane,
    poll_tag,
    *,
    rank,
    npes,
    epr,
    total_experts,
    total_segments,
    tag_shift,
    decode=False,
    wt_owner,
):
    """dispatch._derive_allgather_offsets (epr <= 64) fused with the next-launch
    fanout-pair selection of the same destinations (pair word stored
    relaxed; it is consumed only by the NEXT launch).

    Both consume exactly the same gathered words (per-expert counts, group
    counts, current pair), so each destination is loaded once, in
    2 + npes/2 wave load instructions instead of 2 + 2*npes (offsets) plus
    1 + 2*npes again (pairs).
    """
    assert epr <= 64

    def dp(slot):
        return _sload_table_i64(addr_disp, int(slot))

    task_base = ptr_buf_tensor(dp(DispatchSlot.TASK_ROW_BASE), fx.Int32)
    group_base = ptr_buf_tensor(dp(DispatchSlot.GROUP_TASK_BASE), fx.Int32)
    addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)
    addr_count_matrix = dp(DispatchSlot.COUNT_MATRIX)
    addr_count_done = dp(DispatchSlot.COUNT_DONE)
    fresh2 = decode
    top2 = decode

    def _issue(destination):
        destination = fx.Int32(fx.rocdl.readfirstlane(T.i32, destination))
        meta_addr, meta_val, count_addrs, count_vals = _ll_dest_issue(
            addr_count_matrix,
            addr_count_done,
            addr_pair_config,
            parity,
            destination,
            lane,
            rank=rank,
            npes=npes,
            epr=epr,
            total_experts=total_experts,
            total_segments=total_segments,
        )
        return (destination, meta_addr, meta_val, count_addrs, count_vals)

    issued = []
    for di, destination in enumerate(destinations):
        if const_expr(di == 0 or not fresh2):
            issued.append(_issue(destination))
    for di in range_constexpr(len(destinations)):
        destination, meta_addr, meta_val, count_addrs, count_vals = issued[di]
        (
            normal_count,
            normal_source_prefix,
            destination_group_count,
            destination_group_source_prefix,
            destination_tile_m,
            packed,
        ) = _ll_dest_finish(
            meta_addr,
            meta_val,
            count_addrs,
            count_vals,
            lane,
            poll_tag,
            rank=rank,
            npes=npes,
            epr=epr,
            tag_shift=tag_shift,
        )
        if const_expr(fresh2 and di + 1 < len(destinations)):
            # Every source's push for this destination has landed, so the
            # next destination's words (same push) are read fresh now and
            # their round trip overlaps this destination's math and stores.
            issued.append(_issue(destinations[di + 1]))
        pair_enabled = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
        pair_a = packed & fx.Int32(0xFF)
        pair_b = (packed >> fx.Int32(8)) & fx.Int32(0xFF)
        pair_mask = (fx.Int64(1) << fx.Int64(pair_a)) | (
            fx.Int64(1) << fx.Int64(pair_b)
        )
        selected_mask = pair_enabled.select(pair_mask, fx.Int64(0))
        valid_expert = lane < fx.Int32(epr)
        safe_expert = valid_expert.select(lane, fx.Int32(0))
        ge = destination * fx.Int32(epr) + safe_expert
        group_member = valid_expert & (
            ((selected_mask >> fx.Int64(safe_expert)) & fx.Int64(1)) != fx.Int64(0)
        )
        group_count = group_member.select(destination_group_count, fx.Int32(0))
        group_source_prefix = group_member.select(
            destination_group_source_prefix, fx.Int32(0)
        )
        if const_expr(top2):
            # sort_block_m is a power of two (32/64/128): round up by mask.
            tile_round = fx.Int32(0) - destination_tile_m
            group_rows = (group_count + destination_tile_m - fx.Int32(1)) & tile_round
            normal_rows = (normal_count + destination_tile_m - fx.Int32(1)) & tile_round
        else:
            group_rows = (
                (group_count + destination_tile_m - fx.Int32(1)) // destination_tile_m
            ) * destination_tile_m
            normal_rows = (
                (normal_count + destination_tile_m - fx.Int32(1)) // destination_tile_m
            ) * destination_tile_m
        padded_rows = group_rows + normal_rows
        inclusive_rows = _wave_inclusive_scan_i32(padded_rows, lane)
        local_row_base = inclusive_rows - padded_rows
        next_packed = (_next_pair_packed_top2 if top2 else _next_pair_packed)(
            packed, normal_count, destination_group_count, lane, epr=epr
        )
        if valid_expert:
            _wst(
                task_base,
                ge,
                local_row_base + group_rows + normal_source_prefix,
                wt_owner=wt_owner,
            )
            if group_member:
                _wst(
                    group_base,
                    ge,
                    local_row_base + group_source_prefix,
                    wt_owner=wt_owner,
                )
        if lane == fx.Int32(0):
            # Consumed only by the NEXT launch (relaxed, like the batch form).
            _store_i32_sys_relaxed(
                addr_pair_config,
                (parity ^ fx.Int32(1)) * fx.Int32(npes) + destination,
                next_packed,
            )


@flyc.jit
def _fill_ramp_i32(addr, start, end, base_val, *, step, wt_owner):
    """``buf[i] = base_val + (i - start) * step`` for ``i`` in ``[start, end)``.

    dwordx4 stores at dword-aligned element offsets (unit_stride=1) for the
    body, then at most three dword stores.  Under the m128 quant store stream
    a wave's outstanding-store budget (vmcnt) is the limiter, so the
    per-element loops of _store_expert_metadata are store-issue bound; quads
    cut the instruction count ~4x.  Values are identical to the per-element loop.
    """
    buf1 = ptr_buf_tensor(addr, fx.Int32)
    buf4 = ptr_buf_tensor(addr, fx.Int32, unit_elems=4, unit_stride=1)
    count = end - start
    count = (count > fx.Int32(0)).select(count, fx.Int32(0))
    body_end = start + ((count >> fx.Int32(2)) << fx.Int32(2))
    for i in range(start, body_end, 4):
        v = base_val + (i - start) * fx.Int32(step)
        vals = [v + fx.Int32(k * step) for k in range_constexpr(4)]
        _wst4(buf4, i, fx.Vector.from_elements(vals, fx.Int32), wt_owner=wt_owner)
    for i in range(body_end, start + count, 1):
        _wst(buf1, i, base_val + (i - start) * fx.Int32(step), wt_owner=wt_owner)


@flyc.jit
def _store_expert_metadata_vec(
    addr_sorted_expert,
    addr_tile_row_base,
    addr_tile_input_base,
    addr_srcmap,
    ge,
    local_row_base,
    input_row_base,
    total_count,
    num_tiles,
    padded_rows,
    *,
    fz_tile_m,
    invalid_source,
    wt_owner,
):
    """Same writes as dispatch._store_expert_metadata, quad-vectorised."""
    base_tile = local_row_base // fx.Int32(fz_tile_m)
    tile_end = base_tile + num_tiles
    _fill_ramp_i32(
        addr_sorted_expert, base_tile, tile_end, ge, step=0, wt_owner=wt_owner
    )
    _fill_ramp_i32(
        addr_tile_row_base,
        base_tile,
        tile_end,
        local_row_base,
        step=fz_tile_m,
        wt_owner=wt_owner,
    )
    _fill_ramp_i32(
        addr_tile_input_base,
        base_tile,
        tile_end,
        input_row_base,
        step=fz_tile_m,
        wt_owner=wt_owner,
    )
    _fill_ramp_i32(
        addr_srcmap,
        local_row_base + total_count,
        local_row_base + padded_rows,
        fx.Int32(invalid_source),
        step=0,
        wt_owner=wt_owner,
    )


@flyc.jit
def _plan_meta_warp(
    count_matrix,
    addr_pair_config,
    addr_sorted_expert,
    addr_tile_row_base,
    addr_tile_input_base,
    addr_srcmap,
    parity,
    lane,
    addr_count_matrix=None,
    poll_tag=None,
    addr_count_done=None,
    *,
    fz_npes,
    fz_epr,
    fz_rank,
    fz_tile_m,
    fz_total_experts,
    fz_mtpr,
    count_stride,
    tag_shift=0,
    wt_owner,
):
    """The planner's per-expert tile metadata, owned by a helper warp.

    Recomputes exactly the planner warp's row layout (same loads, same tile
    rounding, same inclusive scan) and writes SORTED_EXPERT / TILE_ROW_BASE /
    TILE_INPUT_BASE / SRCMAP padding, so the planner warp only resets the
    tile-arrival state.  PLAN_READY is published after the CTA barrier that
    follows both warps.
    """
    assert fz_epr <= 64
    local_expert = lane
    valid_expert = local_expert < fx.Int32(fz_epr)
    safe_expert = valid_expert.select(local_expert, fx.Int32(0))
    ge = fx.Int32(fz_rank * fz_epr) + local_expert
    # Same words as below in 1 + npes/2 wave loads (no separate pair
    # load + full wait): see _ll_dest_issue.
    meta_addr, meta_val, count_addrs, count_vals = _ll_dest_issue(
        addr_count_matrix,
        addr_count_done,
        addr_pair_config,
        parity,
        fx.Int32(fz_rank),
        lane,
        rank=fz_rank,
        npes=fz_npes,
        epr=fz_epr,
        total_experts=fz_total_experts,
        total_segments=count_stride,
    )
    (
        normal_count,
        _,
        own_group_count,
        _,
        _,
        packed,
    ) = _ll_dest_finish(
        meta_addr,
        meta_val,
        count_addrs,
        count_vals,
        lane,
        poll_tag,
        rank=fz_rank,
        npes=fz_npes,
        epr=fz_epr,
        tag_shift=tag_shift,
    )
    pair_enabled = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
    pair_a = packed & fx.Int32(0xFF)
    pair_b = (packed >> fx.Int32(8)) & fx.Int32(0xFF)
    pair_mask = (fx.Int64(1) << fx.Int64(pair_a)) | (fx.Int64(1) << fx.Int64(pair_b))
    local_fanout_mask = pair_enabled.select(pair_mask, fx.Int64(0))
    canonical_expert = pair_enabled.select(pair_a, fx.Int32(0))
    group_member = valid_expert & (
        ((local_fanout_mask >> fx.Int64(safe_expert)) & fx.Int64(1)) != fx.Int64(0)
    )
    group_count = group_member.select(own_group_count, fx.Int32(0))
    group_num_tiles = (group_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
    normal_num_tiles = (normal_count + fx.Int32(fz_tile_m - 1)) // fx.Int32(fz_tile_m)
    group_padded_rows = group_num_tiles * fx.Int32(fz_tile_m)
    normal_padded_rows = normal_num_tiles * fx.Int32(fz_tile_m)
    padded_rows = group_padded_rows + normal_padded_rows
    inclusive_rows = _wave_inclusive_scan_i32(padded_rows, lane)
    local_row_base = inclusive_rows - padded_rows
    group_row_base = local_row_base
    normal_row_base = local_row_base + group_padded_rows
    group_input_base = fx.Int32(
        fx.rocdl.readlane(T.i32, group_row_base, canonical_expert)
    )
    store_meta = _store_expert_metadata_vec
    if valid_expert:
        if group_member:
            store_meta(
                addr_sorted_expert,
                addr_tile_row_base,
                addr_tile_input_base,
                addr_srcmap,
                ge,
                group_row_base,
                group_input_base,
                group_count,
                group_num_tiles,
                group_padded_rows,
                fz_tile_m=fz_tile_m,
                invalid_source=fz_npes * fz_mtpr,
                wt_owner=wt_owner,
            )
        store_meta(
            addr_sorted_expert,
            addr_tile_row_base,
            addr_tile_input_base,
            addr_srcmap,
            ge,
            normal_row_base,
            normal_row_base,
            normal_count,
            normal_num_tiles,
            normal_padded_rows,
            fz_tile_m=fz_tile_m,
            invalid_source=fz_npes * fz_mtpr,
            wt_owner=wt_owner,
        )


@flyc.jit
def _fused_prefix_warp(
    fused_scratch,
    prefix_scratch,
    pair_base,
    lane,
    *,
    total_segments,
    wt_owner,
):
    """Fused-path global-segment exclusive prefix (the warp-1 prefix of
    _emit_plan) from the LDS histogram into a separate LDS table + PAIR_BASE."""
    seg_rounds = (total_segments + 63) // 64
    round_counts = []
    for rnd in range_constexpr(seg_rounds):
        seg = fx.Int32(rnd * 64) + lane
        valid_seg = seg < fx.Int32(total_segments)
        safe_seg = valid_seg.select(seg, fx.Int32(0))
        seg_count = fx.Int32(fx.ptr_load(fused_scratch + fx.Int64(safe_seg)))
        round_counts.append(valid_seg.select(seg_count, fx.Int32(0)))
    seg_carry = fx.Int32(0)
    for rnd in range_constexpr(seg_rounds):
        seg = fx.Int32(rnd * 64) + lane
        valid_seg = seg < fx.Int32(total_segments)
        seg_count = round_counts[rnd]
        seg_incl = _wave_inclusive_scan_i32(seg_count, lane)
        source_prefix = seg_carry + seg_incl - seg_count
        seg_carry = seg_carry + fx.Int32(fx.rocdl.readlane(T.i32, seg_incl, 63))
        if valid_seg:
            _wst(pair_base, seg, source_prefix, wt_owner=wt_owner)
            fx.ptr_store(source_prefix, prefix_scratch + fx.Int64(seg))


@flyc.jit
def _block_prefix_round(
    count_lds,
    block_hist,
    pair_base,
    lane,
    rnd,
    *,
    total_segments,
    group_blocks,
    write_block=True,
    wt_owner,
):
    """One 64-segment round of the non-fused BLOCK_HIST exclusive prefix
    (segments rnd*64 + lane): PAIR_BASE[seg] = sum of all earlier segment
    totals, BLOCK_HIST[g][seg] = PAIR_BASE[seg] + sum_{g' < g} count[g'][seg].
    The earlier rounds' totals come from the reduced histogram in LDS."""
    seg = fx.Int32(rnd * 64) + lane
    valid_seg = seg < fx.Int32(total_segments)
    safe_seg = valid_seg.select(seg, fx.Int32(0))
    current = []
    if const_expr(write_block):
        current = [
            block_hist[fx.Int32(group_block * total_segments) + safe_seg]
            for group_block in range_constexpr(group_blocks)
        ]
    seg_carry = fx.Int32(0)
    seg_count = fx.Int32(0)
    for prev in range_constexpr(rnd + 1):
        pseg = fx.Int32(prev * 64) + lane
        pvalid = pseg < fx.Int32(total_segments)
        psafe = pvalid.select(pseg, fx.Int32(0))
        pcount = pvalid.select(
            fx.Int32(fx.ptr_load(count_lds + fx.Int64(psafe))), fx.Int32(0)
        )
        if const_expr(prev < rnd):
            seg_carry = seg_carry + _wave_sum_uniform_i32(pcount, lane)
        else:
            seg_count = pcount
    seg_incl = _wave_inclusive_scan_i32(seg_count, lane)
    source_prefix = seg_carry + seg_incl - seg_count
    if valid_seg:
        _wst(pair_base, seg, source_prefix, wt_owner=wt_owner)
        if const_expr(write_block):
            block_prefix = source_prefix
            for group_block in range_constexpr(group_blocks):
                _wst(
                    block_hist,
                    fx.Int32(group_block * total_segments) + seg,
                    block_prefix,
                    wt_owner=wt_owner,
                )
                block_prefix = block_prefix + current[group_block]


@flyc.jit
def _wave_sum_uniform_i32(value, lane):
    incl = _wave_inclusive_scan_i32(value, lane)
    return fx.Int32(fx.rocdl.readlane(T.i32, incl, 63))


# fmt: off
@flyc.jit
def _emit_plan(
    *, num_waves, fz_npes, fz_epr, fz_k, fz_mtpr, fz_rank, fz_tile_m, fz_total_experts, addr_disp,
    i32_cur_tok, addr_in_idx, parity, expected,
    dispatch_blocks, group_blocks, group_done_slot, group_phase_base, payload_chunk_rows, tile_state_stride,
    fused_scratch=None,
    defer_plan_ready=False,
    early_fill=None,
    prefix_scratch=None,
    count_lds=None,
    groups_self_prefix=False,
    wt_owner,
):
# fmt: on
    """Build a destination-owned compact plan in one producer-only CTA."""
    def dp(i):
        return _sload_table_i64(addr_disp, int(i))

    a_pair_base = dp(DispatchSlot.PAIR_BASE)
    a_se = dp(DispatchSlot.SORTED_EXPERT)
    a_trb = dp(DispatchSlot.TILE_ROW_BASE)
    a_tib = dp(DispatchSlot.TILE_INPUT_BASE)
    a_nv = dp(DispatchSlot.NUM_VALID)
    a_sm = dp(DispatchSlot.SRCMAP)
    a_lh = dp(DispatchSlot.LOCAL_HIST)
    a_block_hist = dp(DispatchSlot.BLOCK_HIST)
    a_bc = dp(DispatchSlot.COUNT_MATRIX)
    p_bc = dp(DispatchSlot.P2P_COUNT_MATRIX)
    a_cd = dp(DispatchSlot.COUNT_DONE)
    p_cd = dp(DispatchSlot.P2P_COUNT_DONE)
    p_plan_ready = dp(DispatchSlot.P2P_PLAN_READY)
    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_expert_tile_end = dp(DispatchSlot.EXPERT_TILE_END)
    a_group_done = dp(DispatchSlot.GROUP_DONE) + fx.Int64(group_done_slot * 4)
    a_max_expert_tiles = dp(DispatchSlot.MAX_EXPERT_TILES)
    a_tile_ready = dp(DispatchSlot.TILE_READY)
    a_tile_expected = dp(DispatchSlot.TILE_EXPECTED)
    assert payload_chunk_rows > 0 and tile_state_stride > 0
    tile_state_byte_offset = (
        fx.Int64(parity) * fx.Int64(tile_state_stride) * fx.Int64(4)
    )
    a_tile_ready = a_tile_ready + tile_state_byte_offset
    a_tile_expected = a_tile_expected + tile_state_byte_offset
    a_payload_blocks_per_destination = dp(DispatchSlot.PAYLOAD_BLOCKS_PER_DESTINATION)
    a_payload_chunks_per_destination = dp(DispatchSlot.PAYLOAD_CHUNKS_PER_DESTINATION)

    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    block_threads = num_waves * 64

    total_segments = fz_total_experts + fz_npes
    addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)
    local_hist = ptr_buf_tensor(a_lh, fx.Int32)
    block_hist = ptr_buf_tensor(a_block_hist, fx.Int32)
    count_matrix = ptr_buf_tensor(a_bc, fx.Int32)
    pair_base = ptr_buf_tensor(a_pair_base, fx.Int32)
    count_done_table = ptr_buf_tensor(p_cd, fx.Int64)
    plan_ready_table = ptr_buf_tensor(p_plan_ready, fx.Int64)
    expert_tile_end = ptr_buf_tensor(a_expert_tile_end, fx.Int32)
    max_expert_tiles_buffer = ptr_buf_tensor(a_max_expert_tiles, fx.Int32)
    fused = fused_scratch is not None
    # The fast path is EP8 with at most 64 experts per rank (see
    # _fast_prepare_supported): the expert tile metadata goes to a helper
    # warp and PLAN_READY is published after the CTA barrier.
    assert fz_epr <= 64 and fz_npes == 8
    tag_shift = max(1, (int(fz_mtpr) * int(fz_k)).bit_length())
    assert tag_shift <= 24
    tag_mask = (1 << (31 - tag_shift)) - 1
    # Launch sequence: EXPECTED[parity] advances by npes per use and the two
    # parities alternate, so (expected / npes) * 2 - parity is 1, 2, 3, ...
    launch_seq = (expected // fx.Int32(fz_npes)) * fx.Int32(2) - parity
    cur_tag_bits = (launch_seq & fx.Int32(tag_mask)) << fx.Int32(tag_shift)
    poll_tag = launch_seq & fx.Int32(tag_mask)
    if const_expr(not fused):
        if tid == fx.Int32(0):
            comm_ops.wait_i32_until_equals(
                a_group_done,
                group_phase_base + fx.Int32(group_blocks),
            )
            comm_ops.fence_agent_acquire()
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()
        comm_ops.fence_agent_acquire()

    # Every prepare worker publishes a complete per-segment histogram.  The
    # owner reduces those rows with ordinary loads/stores, so counting and the
    # later pair-order fill need no global per-route atomics.
    # Reduce the group histograms and push each reduced segment to every
    # peer straight from registers: the XGMI histogram exchange starts as
    # soon as a segment is reduced (no CTA barrier / reload in between).
    count_stride = total_segments
    # Non-fused: the BLOCK_HIST prefix rounds are spread over warps 1..7
    # (one 64-segment round each) instead of all on warp 1.
    split_prefix = (
        (not fused) and count_lds is not None
        and (total_segments + 63) // 64 <= num_waves - 1
    )
    remote_count_matrices = []
    remote_count_bases = []
    for destination in range_constexpr(fz_npes):
        remote_base = _sload_table_i64(p_bc, destination)
        remote_count_bases.append(remote_base)
        remote_count_matrices.append(ptr_buf_tensor(remote_base, fx.Int32))
    push_no_drain = (
        fused and fz_tile_m == 32
        and prefix_scratch is not None and defer_plan_ready
    )
    if tid < fx.Int32(fz_npes):
        _store_i32_sys_relaxed(
            count_done_table[tid],
            fx.Int32(8 + fz_rank),
            fx.Int32(fz_tile_m) | cur_tag_bits,
        )
    for segment in range(tid, total_segments, block_threads):
        segment_count = fx.Int32(0)
        if const_expr(fused):
            # Single-CTA fused path: the owner counted the routes itself,
            # so its LDS histogram IS the rank histogram.
            segment_count = fx.Int32(fx.ptr_load(fused_scratch + fx.Int64(segment)))
        else:
            for group_block in range_constexpr(group_blocks):
                segment_count = segment_count + block_hist[
                    fx.Int32(group_block * total_segments) + segment
                ]
        _wst(local_hist, segment, segment_count, wt_owner=wt_owner)
        if const_expr(split_prefix):
            fx.ptr_store(segment_count, count_lds + fx.Int64(segment))
        # Stage2 reads the receiver's own group column untagged; that
        # word is pushed plain and its tagged copy goes to the
        # receiver's COUNT_DONE[rank].
        if segment >= fx.Int32(fz_total_experts):
            _store_i32_sys_relaxed(
                count_done_table[segment - fx.Int32(fz_total_experts)],
                fx.Int32(fz_rank),
                segment_count | cur_tag_bits,
            )
        for destination in range_constexpr(fz_npes):
            own_group_column = segment == fx.Int32(
                fz_total_experts + destination
            )
            _store_i32_sys_relaxed(
                remote_count_bases[destination],
                fx.Int32(fz_rank * total_segments) + segment,
                own_group_column.select(
                    segment_count, segment_count | cur_tag_bits
                ),
            )
    early_prefix = prefix_scratch is not None
    if const_expr(early_prefix):
        # Fused: the segment prefix needs only the LDS histogram, so the last
        # warp (no segments in the push loop above when total_segments <=
        # 7*64) builds it into a separate LDS table before the barrier; every
        # warp can then fill its pair-order share right after the barrier,
        # in the shadow of the XGMI count exchange.
        assert fused
        if warp == fx.Int32(num_waves - 1):
            _fused_prefix_warp(
                fused_scratch, prefix_scratch, pair_base, lane,
                total_segments=total_segments,
                wt_owner=wt_owner,
            )
    # Decode: the CTA barrier after the push only publishes LDS (histogram,
    # prefix); waves do not drain their remote count stores here, so the
    # XGMI write acks overlap the pair-order fill (payload geometry reads
    # the LDS histogram, not LOCAL_HIST).  Every store is still drained by
    # the s_waitcnt vmcnt(0) + s_barrier ahead of PLAN_READY.
    if const_expr(push_no_drain):
        fx.rocdl.s_waitcnt(lgkmcnt=0)
    else:
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()

    # Warp 0 plans local experts after all source matrices arrive.
    if warp == fx.Int32(0):
      if const_expr(True):

        num_valid_buffer = ptr_buf_tensor(a_nv, fx.Int32)
        row_carry = fx.Int32(0)
        max_expert_tiles = fx.Int32(0)
        own_issued = None
        # The current pair word rides in the same wave load as the
        # tagged group counts (lane npes+1), so warp 0 issues ALL of its
        # count words at once instead of a dependent pair load + full
        # wait, then 2*npes per-lane count loads.
        own_issued = _ll_own_issue(
            a_bc,
            a_cd,
            addr_pair_config,
            parity,
            lane,
            rank=fz_rank,
            npes=fz_npes,
            epr=fz_epr,
            total_experts=fz_total_experts,
            total_segments=count_stride,
        )
        if const_expr(early_fill is not None):
            early_fill()
        (
            own_packed,
            own_group_counts,
            own_normal_counts,
        ) = _ll_own_finish(
            *own_issued,
            lane,
            poll_tag,
            npes=fz_npes,
            tag_shift=tag_shift,
        )
        pair_enabled = (own_packed & fx.Int32(1 << 16)) != fx.Int32(0)
        pair_a = own_packed & fx.Int32(0xFF)
        pair_b = (own_packed >> fx.Int32(8)) & fx.Int32(0xFF)
        local_fanout_mask = pair_enabled.select(
            (fx.Int64(1) << fx.Int64(pair_a)) | (fx.Int64(1) << fx.Int64(pair_b)),
            fx.Int64(0),
        )
        for expert_chunk in range_constexpr((fz_epr + 63) // 64):
            local_expert = fx.Int32(expert_chunk * 64) + lane
            valid_expert = local_expert < fx.Int32(fz_epr)
            safe_expert = valid_expert.select(local_expert, fx.Int32(0))
            normal_source_counts = []
            normal_count = fx.Int32(0)
            group_source_counts = []
            group_count = fx.Int32(0)
            group_member = valid_expert & (
                (
                    (local_fanout_mask >> fx.Int64(safe_expert))
                    & fx.Int64(1)
                )
                != fx.Int64(0)
            )
            plan_normal_loaded = own_normal_counts
            plan_group_loaded = own_group_counts
            for source in range_constexpr(fz_npes):
                source_count = plan_normal_loaded[source]
                source_count = valid_expert.select(source_count, fx.Int32(0))
                normal_source_counts.append(source_count)
                normal_count = normal_count + source_count
                source_group_count = plan_group_loaded[source]
                source_group_count = group_member.select(
                    source_group_count, fx.Int32(0)
                )
                group_source_counts.append(source_group_count)
                group_count = group_count + source_group_count

            group_num_tiles = (
                group_count + fx.Int32(fz_tile_m - 1)
            ) // fx.Int32(fz_tile_m)
            normal_num_tiles = (
                normal_count + fx.Int32(fz_tile_m - 1)
            ) // fx.Int32(fz_tile_m)
            num_tiles = group_num_tiles + normal_num_tiles
            chunk_max = _wave_max_uniform_i32(
                valid_expert.select(num_tiles, fx.Int32(0)),
                active_lanes=min(64, fz_epr - expert_chunk * 64),
            )
            max_expert_tiles = (chunk_max > max_expert_tiles).select(
                chunk_max, max_expert_tiles
            )
            group_padded_rows = group_num_tiles * fx.Int32(fz_tile_m)
            normal_padded_rows = normal_num_tiles * fx.Int32(fz_tile_m)
            padded_rows = group_padded_rows + normal_padded_rows
            inclusive_rows = _wave_inclusive_scan_i32(padded_rows, lane)
            local_row_base = row_carry + inclusive_rows - padded_rows
            group_row_base = local_row_base

            # Stage1 requires payload_chunk_rows % sort_block_m == 0 for compact
            # dispatch, so every section's tiles share one ready-word init.
            assert payload_chunk_rows % fz_tile_m == 0
            if valid_expert:
              # group_source_counts are zero off the pair, so
              # group_num_tiles == 0 there (same as skipping the section).
              _init_sections_ready_merged(
                  a_tile_ready,
                  a_tile_expected,
                  group_source_counts,
                  normal_source_counts,
                  group_row_base,
                  group_num_tiles,
                  normal_num_tiles,
                  fz_npes=fz_npes,
                  payload_chunk_rows=payload_chunk_rows,
                  fz_tile_m=fz_tile_m,
                  wt_owner=wt_owner,
              )
              _wst(expert_tile_end, local_expert, (
                  local_row_base + padded_rows
              ) // fx.Int32(fz_tile_m), wt_owner=wt_owner)

            last_lane = min(63, fz_epr - expert_chunk * 64 - 1)
            row_carry = row_carry + fx.Int32(fx.rocdl.readlane(T.i32, inclusive_rows, last_lane))


        if lane == fx.Int32(0):
            _wst(num_valid_buffer, fx.Int32(0), row_carry, wt_owner=wt_owner)
            _wst(
                max_expert_tiles_buffer, fx.Int32(0), max_expert_tiles,
                wt_owner=wt_owner,
            )
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    else:
      if warp == fx.Int32(2):
        _configure_payload_geometry_fast(
            a_lh,
            a_payload_chunks_per_destination,
            a_payload_blocks_per_destination,
            lane,
            fz_npes=fz_npes,
            fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            payload_chunk_rows=payload_chunk_rows,
            dispatch_blocks=dispatch_blocks,
            lds_hist=fused_scratch if push_no_drain else None,
            wt_owner=wt_owner,
        )
      if const_expr(early_fill is not None):
        early_fill()
      if const_expr(split_prefix):
        for rnd in range_constexpr((total_segments + 63) // 64):
          if warp == fx.Int32(1 + rnd):
            _block_prefix_round(
                count_lds, block_hist, pair_base, lane, rnd,
                total_segments=total_segments, group_blocks=group_blocks,
                write_block=not groups_self_prefix,
                wt_owner=wt_owner,
            )
      if const_expr((not early_prefix) and (not split_prefix)):  # noqa: SIM102 - preserve DSL staging
        if warp == fx.Int32(1):
          # Global-segment exclusive prefix, segment-strided: lane l owns
          # segments r*64 + l, so every BLOCK_HIST row access is one coalesced
          # wave load/store and the group_blocks loads of a round are all in
          # flight together (instead of group_blocks dependent load->store
          # round trips per lane-owned segment).  The next round's loads are
          # issued before this round's stores (same buffer: the compiler
          # cannot hoist them past the stores itself).
          seg_rounds = (total_segments + 63) // 64
          round_counts = []
          for rnd in range_constexpr(seg_rounds):
              seg = fx.Int32(rnd * 64) + lane
              valid_seg = seg < fx.Int32(total_segments)
              safe_seg = valid_seg.select(seg, fx.Int32(0))
              if const_expr(fused):
                  seg_count = fx.Int32(fx.ptr_load(fused_scratch + fx.Int64(safe_seg)))
              else:
                  seg_count = local_hist[safe_seg]
              round_counts.append(valid_seg.select(seg_count, fx.Int32(0)))

          def _load_block_counts(rnd):
              seg = fx.Int32(rnd * 64) + lane
              safe_seg = (seg < fx.Int32(total_segments)).select(seg, fx.Int32(0))
              return [
                  block_hist[fx.Int32(group_block * total_segments) + safe_seg]
                  for group_block in range_constexpr(group_blocks)
              ]

          # When the groups self-prefix they may still be reading the raw
          # BLOCK_HIST counts after GROUP_DONE, so only PAIR_BASE is written.
          pending = None
          if const_expr((not fused) and (not groups_self_prefix)):
              pending = _load_block_counts(0)
          seg_carry = fx.Int32(0)
          for rnd in range_constexpr(seg_rounds):
              seg = fx.Int32(rnd * 64) + lane
              valid_seg = seg < fx.Int32(total_segments)
              seg_count = round_counts[rnd]
              seg_incl = _wave_inclusive_scan_i32(seg_count, lane)
              source_prefix = seg_carry + seg_incl - seg_count
              seg_carry = seg_carry + fx.Int32(fx.rocdl.readlane(T.i32, seg_incl, 63))
              current = pending
              if const_expr(
                  (not fused) and (not groups_self_prefix) and rnd + 1 < seg_rounds
              ):
                  pending = _load_block_counts(rnd + 1)
              if valid_seg:
                  _wst(pair_base, seg, source_prefix, wt_owner=wt_owner)
                  if const_expr(fused):
                      # The fused fill reads its per-segment base from LDS.
                      fx.ptr_store(source_prefix, fused_scratch + fx.Int64(seg))
                  elif const_expr(not groups_self_prefix):
                      block_prefix = source_prefix
                      for group_block in range_constexpr(group_blocks):
                          _wst(
                              block_hist,
                              fx.Int32(group_block * total_segments) + seg,
                              block_prefix,
                              wt_owner=wt_owner,
                          )
                          block_prefix = block_prefix + current[group_block]
      # Every warp observes COUNT_DONE itself and performs its own system
      # acquire, so the per-destination offset derivation below overlaps
      # warp 0's local plan instead of waiting for it at a CTA barrier.
      if warp == fx.Int32(2):
          _plan_meta_warp(
              count_matrix,
              addr_pair_config,
              a_se,
              a_trb,
              a_tib,
              a_sm,
              parity,
              lane,
              a_bc,
              poll_tag,
              a_cd,
              fz_npes=fz_npes,
              fz_epr=fz_epr,
              fz_rank=fz_rank,
              fz_tile_m=fz_tile_m,
              fz_total_experts=fz_total_experts,
              fz_mtpr=fz_mtpr,
              count_stride=count_stride,
              tag_shift=tag_shift,
              wt_owner=wt_owner,
          )
    # Destination w is derived by warp w (w >= 1) concurrently with warp 0's
    # local plan; destination 0 goes to warp 3 (it has no other local work).
    assert 4 < fz_npes <= num_waves == 8
    if warp != fx.Int32(0):
        # Offsets + next-launch pair of a destination share one load
        # set.  w1: dests 1, 2; w2: tile metadata only (its loads would
        # queue behind the metadata stores); w3: dests 3, 0;
        # w4..w7: dest w.
        if warp >= fx.Int32(4):
            _derive_offsets_pairs_ll(
                addr_disp,
                parity,
                [warp],
                lane,
                poll_tag,
                rank=fz_rank,
                npes=fz_npes,
                epr=fz_epr,
                total_experts=fz_total_experts,
                total_segments=total_segments,
                tag_shift=tag_shift,
                decode=fz_tile_m == 32,
                wt_owner=wt_owner,
            )
        if warp == fx.Int32(1):
            _derive_offsets_pairs_ll(
                addr_disp,
                parity,
                [fx.Int32(1), fx.Int32(2)],
                lane,
                poll_tag,
                rank=fz_rank,
                npes=fz_npes,
                epr=fz_epr,
                total_experts=fz_total_experts,
                total_segments=total_segments,
                tag_shift=tag_shift,
                decode=fz_tile_m == 32,
                wt_owner=wt_owner,
            )
        if warp == fx.Int32(3):
            _derive_offsets_pairs_ll(
                addr_disp,
                parity,
                [fx.Int32(3), fx.Int32(0)],
                lane,
                poll_tag,
                rank=fz_rank,
                npes=fz_npes,
                epr=fz_epr,
                total_experts=fz_total_experts,
                total_segments=total_segments,
                tag_shift=tag_shift,
                decode=fz_tile_m == 32,
                wt_owner=wt_owner,
            )
    peer_plan_ready = fx.Int64(0)
    if const_expr(not defer_plan_ready):  # noqa: SIM102 - preserve DSL staging
        if tid < fx.Int32(fz_npes):
            peer_plan_ready = plan_ready_table[tid]
    if const_expr(early_fill is not None):
        # Every warp already filled its pair-order share; the caller's
        # s_waitcnt vmcnt(0) + s_barrier before the deferred publish is the
        # only CTA rendezvous left.
        assert defer_plan_ready
    elif const_expr(defer_plan_ready):
        # The deferred publish drains every store (s_waitcnt vmcnt(0) +
        # s_barrier) after the fused fill, and the fill reads only LDS and
        # count-phase data, so the plan's stores keep draining under the
        # fill instead of stalling every wave here.
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.barrier()
    else:
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()
    if const_expr(not defer_plan_ready):  # noqa: SIM102 - preserve DSL staging
        # One system release publishes the whole plan (tile state by warp 0,
        # tile metadata by warp 2, offsets/prefixes by the other warps) to
        # the peers (PLAN_READY) and to the local group CTAs (PAIR_READY).
        # Every published plan store was written through (sc0 sc1, see
        # _wst) and drained by the s_waitcnt vmcnt(0) + s_barrier above, so
        # PLAN_READY / PAIR_READY need no release fence (buffer_wbl2).
        if warp == fx.Int32(0):
            if const_expr(not wt_owner):
                comm_ops.fence_system_release()
            for source in range(lane, fz_npes, 64):
                ready_index = parity * fx.Int32(fz_npes) + fx.Int32(fz_rank)
                _store_i32_sys_relaxed(peer_plan_ready, ready_index, expected)
            if const_expr(not fused):  # noqa: SIM102 - preserve DSL staging
                if lane == fx.Int32(0):
                    _store_i32_sys_relaxed(a_pair_ready, parity, expected)

    # Next-launch fanout pairs are consumed only after this kernel retires,
    # so derive them off the critical path, one warp per destination.


@flyc.jit
def _load_pair_table_lane(addr_pair_config, parity, lane, *, fz_npes):
    """Lane d (mod npes) holds FANOUT_PAIR_CONFIG[parity][d].

    One system-scope load per lane, issued alongside the route-index loads
    (it does not depend on them), replaces one dependent system load per
    route batch.
    """
    assert fz_npes <= 64
    destination = lane % fx.Int32(fz_npes)
    return fx.Int32(
        comm_ops.load_i32_system(
            addr_pair_config, parity * fx.Int32(fz_npes) + destination
        )
    )


def _classify_regs_batch(
    token_batch, peers, lane, topk_slot, active_slot, pair_vals, count_scratch,
    i32_cur_tok,
    *, fz_npes, fz_k, fz_epr, fz_total_experts, route_group_shift, packed_topk6_metadata,
):
    """Branch-free _fused_count_batch for EP layouts with epr <= 64, topk <= 8.

    ``peers[s]`` already holds this lane's token's top-k slot ``s`` (loaded
    directly, so the 8-lane ds_bpermute exchange is gone) and ``pair_vals``
    holds the uniform per-destination fanout pairs.  Classification is the
    exact _classify_fanout_wave_route (epr <= 64, topk <= 8 branch).  The LDS
    count is issued unconditionally (adding 0 for non-emitting lanes), so
    consecutive batches' atomics need no branch and no per-batch wait.
    """
    token = token_batch + (lane >> fx.Int32(route_group_shift))
    active_route = active_slot & (token < i32_cur_tok)
    expert = peers[0]
    for slot in range_constexpr(1, fz_k):
        expert = (topk_slot == fx.Int32(slot)).select(peers[slot], expert)
    valid = active_route & (expert >= fx.Int32(0))
    valid = valid & (expert < fx.Int32(fz_total_experts))
    safe_expert = valid.select(expert, fx.Int32(0))
    destination = safe_expert // fx.Int32(fz_epr)
    packed = pair_vals[0]
    for d in range_constexpr(1, fz_npes):
        packed = (destination == fx.Int32(d)).select(pair_vals[d], packed)
    pair_enabled = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
    pair_a = packed & fx.Int32(0xFF)
    pair_b = (packed >> fx.Int32(8)) & fx.Int32(0xFF)
    # 32-bit form of the 64-bit token/pair mask test: a peer slot is a member
    # of the selected pair iff it names global expert dest_base + pair_a (or
    # + pair_b); that equality already implies "valid" and "same
    # destination".  matched == (token_mask & selected_mask) == selected_mask.
    dest_base = destination * fx.Int32(fz_epr)
    global_a = dest_base + pair_a
    global_b = dest_base + pair_b
    member_slots = fx.Int32(0)
    has_a = pair_enabled & (peers[0] == global_a)
    has_b = pair_enabled & (peers[0] == global_b)
    member_slots = (has_a | has_b).select(fx.Int32(1), fx.Int32(0))
    for slot in range_constexpr(1, fz_k):
        member_a = pair_enabled & (peers[slot] == global_a)
        member_b = pair_enabled & (peers[slot] == global_b)
        has_a = has_a | member_a
        has_b = has_b | member_b
        member_slots = (member_a | member_b).select(
            member_slots | fx.Int32(1 << slot), member_slots
        )
    matched = has_a & has_b
    shared_member = matched & ((safe_expert == global_a) | (safe_expert == global_b))
    # canonical == pair_a whenever shared_member holds (pair enabled).
    emit = (~shared_member) | (safe_expert == global_a)
    segment = shared_member.select(
        fx.Int32(fz_total_experts) + destination, safe_expert
    )

    count_it = valid & emit
    scratch_addr = fx.Int64(fx.ptrtoint(count_scratch))
    intra_rank = fx.Int32(
        comm_ops.atomic_add_workgroup(
            scratch_addr + fx.Int64(segment) * fx.Int64(4),
            count_it.select(fx.Int32(1), fx.Int32(0)),
        )
    )
    if const_expr(packed_topk6_metadata):
        packed_segment = (
            segment | (member_slots << fx.Int32(9)) | (intra_rank << fx.Int32(15))
        )
    else:
        packed_segment = (
            segment | (member_slots << fx.Int32(10)) | (intra_rank << fx.Int32(18))
        )
    return count_it.select(packed_segment, fx.Int32(-1))



@flyc.jit
def _pred_store_i32(buffer, index, value, pred):
    if pred:
        buffer[index] = value


@flyc.jit
def _pred_store_wt(buffer, index, value, pred, *, wt_owner):
    if pred:
        _wst(buffer, index, value, wt_owner=wt_owner)


# ---------------------------------------------------------------------------
# Token-per-lane fused count/fill (fused buckets).
#
# A route-per-lane form spends one wave-wide
# classification (~150 VALU/SALU incl. SGPR-mask spills) per 8 tokens, so the
# m64 pcu4/pcu8 fused count is VALU-bound.  Here
# lane = token: each lane loads its own top-k row with wide loads and
# classifies its routes from registers, so one wave covers 64 tokens.  When
# the compile bucket (64 * P tokens) leaves waves spare, the k routes of a
# token are split over `split` waves (wave = (slot subset, token group)).
# The per-route predicate is exactly _classify_regs_batch's and the packed
# segment encoding is unchanged (ROUTE_SEGMENT keeps it for tail passes).
# ---------------------------------------------------------------------------
_FUSED_TOK_SPLIT = 8
_TOK_DUMMY_SEGS = 64


def _tok_geometry(num_waves, prepare_blocks, fz_k):
    """(split, groups, slots_per_wave) for the token-per-lane count."""
    split = 1
    while (
        split * 2 <= _FUSED_TOK_SPLIT
        and split < fz_k
        and prepare_blocks * split * 2 <= num_waves
    ):
        split *= 2
    groups = num_waves // split
    return split, groups, -(-fz_k // split)


def _tok_dest_div(fz_epr, fz_total_experts):
    """(mul, shift) with (e * mul) >> shift == e // epr for 0 <= e < total."""
    for shift in range(10, 24):
        mul = -(-(1 << shift) // fz_epr)
        if mul >= (1 << 15):
            break
        if all((e * mul) >> shift == e // fz_epr for e in range(fz_total_experts)):
            return mul, shift
    return None


def _tok_load_row(idx_vec, idx_buffer, safe_token, fz_k):
    row_base = safe_token * fx.Int32(fz_k)
    vals = []
    off = 0
    while off < fz_k:
        rem = fz_k - off
        width = 4 if rem >= 4 else (2 if rem >= 2 else 1)
        if width == 1:
            vals.append(idx_buffer[row_base + fx.Int32(off)])
        else:
            v = buf_copy_load(
                idx_vec[width], row_base + fx.Int32(off), fx.Int32, unit_elems=width
            )
            for j in range(width):
                vals.append(fx.Int32(v[j]))
        off += width
    return vals


def _tok_wave_slots(peers, sub, *, fz_k, split, slots_per_wave):
    """This wave's (slot index, expert) list: slots sub*S .. sub*S + S - 1."""
    out = []
    for j in range(slots_per_wave):
        if split == 1:
            out.append((fx.Int32(j), peers[j], j < fz_k))
            continue
        slot = sub * fx.Int32(slots_per_wave) + fx.Int32(j)
        expert = peers[j] if j < fz_k else fx.Int32(-1)
        for q in range(1, split):
            idx = q * slots_per_wave + j
            cand = peers[idx] if idx < fz_k else fx.Int32(-1)
            expert = (sub == fx.Int32(q)).select(cand, expert)
        out.append((slot, expert, None))
    return out


def _tok_classify_routes(
    token_ok, peers, wave_slots, pair_lane_packed, count_scratch, lane,
    *, fz_npes, fz_k, fz_epr, fz_total_experts, packed_topk6_metadata,
):
    """Per-lane (token) classification of this wave's routes.

    Token-level form of _classify_regs_batch: a peer t is a fanout-pair member
    of route s's selected pair iff t is a pair member (a or b) of its OWN
    destination and that destination equals s's.  So one pass over the k
    peers builds per-token destination bitmasks (A: has pair_a, B: has
    pair_b) and a (destination, slot) member bitmap; each route then reads
    its matched bit and member slots with shifts.  Exact (duplicates too).
    """
    # The slot-split waves (split > 1) own only k/split routes each; the
    # per-route compare form is shorter there.
    if const_expr(
        fz_npes > 32 or fz_npes * fz_k > 64 or wave_slots[0][2] is None
    ):
        return _tok_classify_routes_cmp(
            token_ok, peers, wave_slots, pair_lane_packed, count_scratch, lane,
            fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            packed_topk6_metadata=packed_topk6_metadata,
        )
    count_segments = fz_total_experts + fz_npes
    div = _tok_dest_div(fz_epr, fz_total_experts)
    scratch_addr = fx.Int64(fx.ptrtoint(count_scratch))
    dummy_off = fx.Int64(fx.Int32(count_segments) + lane) * fx.Int64(4)
    info = []
    for t in range_constexpr(fz_k):
        expert = peers[t]
        valid = token_ok & (expert >= fx.Int32(0))
        valid = valid & (expert < fx.Int32(fz_total_experts))
        safe_expert = valid.select(expert, fx.Int32(0))
        if const_expr(div is not None):
            destination = (safe_expert * fx.Int32(div[0])) >> fx.Int32(div[1])
        else:
            destination = safe_expert // fx.Int32(fz_epr)
        packed = fx.Int32(
            fx.rocdl.ds_bpermute(T.i32, destination * fx.Int32(4), pair_lane_packed)
        )
        info.append([valid, safe_expert, destination, packed])
    mask_a = fx.Int32(0)
    mask_b = fx.Int32(0)
    member_map = fx.Int64(0)
    for t in range_constexpr(fz_k):
        valid, safe_expert, destination, packed = info[t]
        local_expert = safe_expert - destination * fx.Int32(fz_epr)
        live = valid & ((packed & fx.Int32(1 << 16)) != fx.Int32(0))
        is_a = live & (local_expert == (packed & fx.Int32(0xFF)))
        is_b = live & (
            local_expert == ((packed >> fx.Int32(8)) & fx.Int32(0xFF))
        )
        hit = is_a | is_b
        dest_bit = fx.Int32(1) << destination
        mask_a = mask_a | is_a.select(dest_bit, fx.Int32(0))
        mask_b = mask_b | is_b.select(dest_bit, fx.Int32(0))
        member_map = member_map | hit.select(
            fx.Int64(1) << fx.Int64(destination * fx.Int32(fz_k) + fx.Int32(t)),
            fx.Int64(0),
        )
        info[t] = info[t] + [is_a, hit]
    matched_dests = mask_a & mask_b
    out = []
    for slot, _expert, static_ok in wave_slots:
        if static_ok is not None:
            # split == 1: slot is the compile-time index j.
            j = len(out)
            valid, safe_expert, destination, _packed, is_a, hit = info[j]
        else:
            valid, safe_expert, destination, _, is_a, hit = None, None, None, None, None, None
            for idx in range_constexpr(fz_k):
                cur = info[idx]
                if valid is None:
                    valid, safe_expert, destination, is_a, hit = (
                        cur[0], cur[1], cur[2], cur[4], cur[5]
                    )
                else:
                    pick = slot == fx.Int32(idx)
                    valid = pick.select(cur[0], valid)
                    safe_expert = pick.select(cur[1], safe_expert)
                    destination = pick.select(cur[2], destination)
                    is_a = pick.select(cur[4], is_a)
                    hit = pick.select(cur[5], hit)
            valid = valid & (slot < fx.Int32(fz_k))
        matched = ((matched_dests >> destination) & fx.Int32(1)) != fx.Int32(0)
        shared_member = matched & hit
        emit = (~shared_member) | is_a
        member_slots = fx.Int32(
            (member_map >> fx.Int64(destination * fx.Int32(fz_k)))
            & fx.Int64((1 << fz_k) - 1)
        )
        segment = shared_member.select(
            fx.Int32(fz_total_experts) + destination, safe_expert
        )
        count_it = valid & emit
        atomic_off = count_it.select(fx.Int64(segment) * fx.Int64(4), dummy_off)
        intra_rank = fx.Int32(
            comm_ops.atomic_add_workgroup(scratch_addr + atomic_off, fx.Int32(1))
        )
        if const_expr(packed_topk6_metadata):
            packed_segment = (
                segment | (member_slots << fx.Int32(9)) | (intra_rank << fx.Int32(15))
            )
        else:
            packed_segment = (
                segment | (member_slots << fx.Int32(10)) | (intra_rank << fx.Int32(18))
            )
        out.append(count_it.select(packed_segment, fx.Int32(-1)))
    return out


def _tok_classify_routes_cmp(
    token_ok, peers, wave_slots, pair_lane_packed, count_scratch, lane,
    *, fz_npes, fz_k, fz_epr, fz_total_experts, packed_topk6_metadata,
):
    """Per-lane (token) classification of this wave's routes."""
    count_segments = fz_total_experts + fz_npes
    div = _tok_dest_div(fz_epr, fz_total_experts)
    scratch_addr = fx.Int64(fx.ptrtoint(count_scratch))
    dummy_off = fx.Int64(fx.Int32(count_segments) + lane) * fx.Int64(4)
    # Pass 1: validity, destination and the (independent) pair lookups.
    pre = []
    for slot, expert, static_ok in wave_slots:
        valid = token_ok & (expert >= fx.Int32(0))
        valid = valid & (expert < fx.Int32(fz_total_experts))
        if static_ok is None:
            valid = valid & (slot < fx.Int32(fz_k))
        safe_expert = valid.select(expert, fx.Int32(0))
        if const_expr(div is not None):
            destination = (safe_expert * fx.Int32(div[0])) >> fx.Int32(div[1])
        else:
            destination = safe_expert // fx.Int32(fz_epr)
        packed = fx.Int32(
            fx.rocdl.ds_bpermute(T.i32, destination * fx.Int32(4), pair_lane_packed)
        )
        pre.append((slot, valid, safe_expert, destination, packed))
    out = []
    for slot, valid, safe_expert, destination, packed in pre:
        pair_enabled = (packed & fx.Int32(1 << 16)) != fx.Int32(0)
        dest_base = destination * fx.Int32(fz_epr)
        # Disabled pair: compare against an id no valid expert equals;
        # shared_member still requires safe_expert (< total) to match.
        global_a = pair_enabled.select(
            dest_base + (packed & fx.Int32(0xFF)), fx.Int32(fz_total_experts)
        )
        global_b = pair_enabled.select(
            dest_base + ((packed >> fx.Int32(8)) & fx.Int32(0xFF)),
            fx.Int32(fz_total_experts),
        )
        has_a = peers[0] == global_a
        has_b = peers[0] == global_b
        member_slots = (has_a | has_b).select(fx.Int32(1), fx.Int32(0))
        for t in range_constexpr(1, fz_k):
            member_a = peers[t] == global_a
            member_b = peers[t] == global_b
            has_a = has_a | member_a
            has_b = has_b | member_b
            member_slots = member_slots | (member_a | member_b).select(
                fx.Int32(1 << t), fx.Int32(0)
            )
        matched = has_a & has_b
        is_a = safe_expert == global_a
        shared_member = matched & (is_a | (safe_expert == global_b))
        emit = (~shared_member) | is_a
        segment = shared_member.select(
            fx.Int32(fz_total_experts) + destination, safe_expert
        )
        count_it = valid & emit
        # Non-counting lanes hit a private dummy word (no same-address
        # serialisation on segment 0, no branch).
        atomic_off = count_it.select(fx.Int64(segment) * fx.Int64(4), dummy_off)
        intra_rank = fx.Int32(
            comm_ops.atomic_add_workgroup(scratch_addr + atomic_off, fx.Int32(1))
        )
        if const_expr(packed_topk6_metadata):
            packed_segment = (
                segment | (member_slots << fx.Int32(9)) | (intra_rank << fx.Int32(15))
            )
        else:
            packed_segment = (
                segment | (member_slots << fx.Int32(10)) | (intra_rank << fx.Int32(18))
            )
        out.append(count_it.select(packed_segment, fx.Int32(-1)))
    return out


def _tok_pad8(vals):
    return list(vals) + [fx.Int32(-1)] * (8 - len(vals))


@flyc.jit
def _tok_count_batch(
    wave_active, token_ok, sub, p0, p1, p2, p3, p4, p5, p6, p7, pair_lane_packed,
    count_scratch, lane,
    *, fz_npes, fz_k, fz_epr, fz_total_experts, packed_topk6_metadata, split,
    slots_per_wave,
):
    q0 = fx.Int32(-1)
    q1 = fx.Int32(-1)
    q2 = fx.Int32(-1)
    q3 = fx.Int32(-1)
    q4 = fx.Int32(-1)
    q5 = fx.Int32(-1)
    q6 = fx.Int32(-1)
    q7 = fx.Int32(-1)
    if wave_active:
        peers = [p0, p1, p2, p3, p4, p5, p6, p7][:fz_k]
        q0, q1, q2, q3, q4, q5, q6, q7 = _tok_pad8(
            _tok_classify_routes(
                token_ok, peers,
                _tok_wave_slots(
                    peers, sub, fz_k=fz_k, split=split,
                    slots_per_wave=slots_per_wave,
                ),
                pair_lane_packed, count_scratch, lane,
                fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
                fz_total_experts=fz_total_experts,
                packed_topk6_metadata=packed_topk6_metadata,
            )
        )
    return q0, q1, q2, q3, q4, q5, q6, q7


@flyc.jit
def _tok_count_rest(
    start, pass_tokens, token_off, sub, lane, idx_vec, idx_buffer, route_segment,
    pair_lane_packed, count_scratch, i32_cur_tok,
    *, fz_npes, fz_k, fz_epr, fz_total_experts, packed_topk6_metadata, split,
    slots_per_wave,
):
    for token_base in range(start, i32_cur_tok, pass_tokens):
        token = token_base + token_off
        token_ok = token < i32_cur_tok
        safe_token = token_ok.select(token, fx.Int32(0))
        peers = _tok_load_row(idx_vec, idx_buffer, safe_token, fz_k)
        wave_slots = _tok_wave_slots(
            peers, sub, fz_k=fz_k, split=split, slots_per_wave=slots_per_wave
        )
        segs = _tok_classify_routes(
            token_ok, peers, wave_slots, pair_lane_packed, count_scratch, lane,
            fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            packed_topk6_metadata=packed_topk6_metadata,
        )
        for j in range_constexpr(slots_per_wave):
            slot = wave_slots[j][0]
            _pred_store_i32(
                route_segment, safe_token * fx.Int32(fz_k) + slot, segs[j],
                token_ok & (slot < fx.Int32(fz_k)),
            )


def _tok_lane_geometry(num_waves, prepare_blocks, fz_k):
    split, groups, slots_per_wave = _tok_geometry(num_waves, prepare_blocks, fz_k)
    tid = fx.thread_idx.x
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    if split == 1:
        group = warp
        sub = fx.Int32(0)
    else:
        group = warp & fx.Int32(groups - 1)
        sub = warp >> fx.Int32(groups.bit_length() - 1)
    token_off = group * fx.Int32(64) + lane
    return split, groups, slots_per_wave, lane, group, sub, token_off


def _tok_prefetch(
    *, num_waves, fz_npes, fz_k, fz_total_experts, addr_disp, i32_cur_tok,
    addr_in_idx, count_scratch, prepare_blocks,
):
    """Issue the token-per-lane count's global loads (top-k rows and BOTH
    parities' fanout-pair table rows) and zero the LDS histogram before the
    owner's parity/epoch block, so their latency overlaps it.  The CTA
    barrier that publishes parity also orders the LDS zeroing."""
    addr_pair_config = _sload_table_i64(
        addr_disp, int(DispatchSlot.FANOUT_PAIR_CONFIG)
    )
    idx_buffer = ptr_buf_tensor(
        addr_in_idx, fx.Int32, num_records_bytes=_idx_bytes(i32_cur_tok, fz_k)
    )
    idx_vec = {
        2: ptr_buf_tensor(
            addr_in_idx, fx.Int32, unit_elems=2, unit_stride=1,
            num_records_bytes=_idx_bytes(i32_cur_tok, fz_k),
        ),
        4: ptr_buf_tensor(
            addr_in_idx, fx.Int32, unit_elems=4, unit_stride=1,
            num_records_bytes=_idx_bytes(i32_cur_tok, fz_k),
        ),
    }
    tid = fx.thread_idx.x
    _, _, _, lane, _, _, token_off = _tok_lane_geometry(
        num_waves, prepare_blocks, fz_k
    )
    pair0 = _load_pair_table_lane(addr_pair_config, fx.Int32(0), lane, fz_npes=fz_npes)
    pair1 = _load_pair_table_lane(addr_pair_config, fx.Int32(1), lane, fz_npes=fz_npes)
    token_ok = token_off < i32_cur_tok
    safe_token = token_ok.select(token_off, fx.Int32(0))
    peers = _tok_load_row(idx_vec, idx_buffer, safe_token, fz_k)
    _zero_lds_i32(
        count_scratch, tid, fx.Int32(num_waves * 64),
        count_segments=fz_total_experts + fz_npes,
    )
    return peers, pair0, pair1


def _emit_fused_count_tok(
    *, num_waves, fz_npes, fz_k, fz_epr, fz_total_experts, addr_disp, i32_cur_tok,
    addr_in_idx, parity, count_scratch, prefetch_batches=1,
    prepare_blocks=1, prefetched=None,
):
    """Token-per-lane fused count; returns the first-pass packed segments."""
    a_route_segment = _sload_table_i64(addr_disp, int(DispatchSlot.ROUTE_SEGMENT))
    addr_pair_config = _sload_table_i64(
        addr_disp, int(DispatchSlot.FANOUT_PAIR_CONFIG)
    )
    idx_buffer = ptr_buf_tensor(
        addr_in_idx, fx.Int32, num_records_bytes=_idx_bytes(i32_cur_tok, fz_k)
    )
    idx_vec = {
        2: ptr_buf_tensor(
            addr_in_idx, fx.Int32, unit_elems=2, unit_stride=1,
            num_records_bytes=_idx_bytes(i32_cur_tok, fz_k),
        ),
        4: ptr_buf_tensor(
            addr_in_idx, fx.Int32, unit_elems=4, unit_stride=1,
            num_records_bytes=_idx_bytes(i32_cur_tok, fz_k),
        ),
    }
    route_segment = ptr_buf_tensor(a_route_segment, fx.Int32)
    tid = fx.thread_idx.x
    block_threads = num_waves * 64
    count_segments = fz_total_experts + fz_npes
    assert count_segments <= 1024, "route metadata supports at most 1024 segments"
    packed_topk6_metadata = fz_k <= 6 and count_segments <= 512
    assert fz_epr <= 64 and 0 < fz_k <= 8 and fz_npes <= 64
    split, groups, slots_per_wave, lane, group, sub, token_off = _tok_lane_geometry(
        num_waves, prepare_blocks, fz_k
    )
    pass_tokens = groups * 64

    token_ok = token_off < i32_cur_tok
    if prefetched is None:
        pair_lane_packed = _load_pair_table_lane(
            addr_pair_config, parity, lane, fz_npes=fz_npes
        )
        safe_token = token_ok.select(token_off, fx.Int32(0))
        peers = _tok_load_row(idx_vec, idx_buffer, safe_token, fz_k)
        _zero_lds_i32(
            count_scratch, tid, fx.Int32(block_threads), count_segments=count_segments
        )
        fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
        fx.barrier()
    else:
        peers, pair0, pair1 = prefetched
        pair_lane_packed = (parity == fx.Int32(0)).select(pair0, pair1)
    wave_active = (group * fx.Int32(64)) < i32_cur_tok
    if const_expr(split * slots_per_wave > fz_k):
        wave_active = wave_active & (sub * fx.Int32(slots_per_wave) < fx.Int32(fz_k))
    segs = list(
        _tok_count_batch(
            wave_active, token_ok, sub, *_tok_pad8(peers), pair_lane_packed,
            count_scratch, lane,
            fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            packed_topk6_metadata=packed_topk6_metadata,
            split=split, slots_per_wave=slots_per_wave,
        )
    )[:slots_per_wave]
    _tok_count_rest(
        fx.Int32(pass_tokens), fx.Int32(pass_tokens), token_off, sub, lane, idx_vec,
        idx_buffer, route_segment, pair_lane_packed, count_scratch, i32_cur_tok,
        fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr, fz_total_experts=fz_total_experts,
        packed_topk6_metadata=packed_topk6_metadata,
        split=split, slots_per_wave=slots_per_wave,
    )
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()
    return segs


def _tok_fill_routes(token, token_ok, slots, segs, pair_order, count_scratch, *,
                     fz_k, fz_total_experts, packed_topk6, wt_owner):
    fills = []
    for j in range_constexpr(len(slots)):
        slot = slots[j]
        packed_segment = segs[j]
        if const_expr(packed_topk6):
            emit = token_ok & (packed_segment >= fx.Int32(0))
            segment = packed_segment & fx.Int32(0x1FF)
            member_slots = (packed_segment >> fx.Int32(9)) & fx.Int32(0x3F)
            intra_rank = (packed_segment >> fx.Int32(15)) & fx.Int32(0xFFFF)
        else:
            emit = token_ok & (packed_segment != fx.Int32(-1))
            segment = packed_segment & fx.Int32(0x3FF)
            member_slots = (packed_segment >> fx.Int32(10)) & fx.Int32(0xFF)
            intra_rank = (packed_segment >> fx.Int32(18)) & fx.Int32(0x3FFF)
        safe_segment = emit.select(segment, fx.Int32(0))
        block_base = fx.Int32(fx.ptr_load(count_scratch + fx.Int64(safe_segment)))
        shared_group = segment >= fx.Int32(fz_total_experts)
        group_entry = token | (member_slots << fx.Int32(24))
        route = token * fx.Int32(fz_k) + slot
        pair_entry = shared_group.select(group_entry, route)
        fills.append((emit, block_base + intra_rank, pair_entry))
    for emit, position, pair_entry in fills:
        _pred_store_wt(pair_order, position, pair_entry, emit, wt_owner=wt_owner)


def _tok_slot_ids(sub, *, split, slots_per_wave):
    if split == 1:
        return [fx.Int32(j) for j in range(slots_per_wave)]
    return [sub * fx.Int32(slots_per_wave) + fx.Int32(j) for j in range(slots_per_wave)]


@flyc.jit
def _tok_fill_rest(
    start, pass_tokens, token_off, sub, pair_order, route_segment, count_scratch,
    i32_cur_tok,
    *, fz_k, fz_total_experts, packed_topk6, split, slots_per_wave,
    wt_owner,
):
    for token_base in range(start, i32_cur_tok, pass_tokens):
        token = token_base + token_off
        token_ok = token < i32_cur_tok
        safe_token = token_ok.select(token, fx.Int32(0))
        slots = _tok_slot_ids(sub, split=split, slots_per_wave=slots_per_wave)
        segs = []
        for j in range_constexpr(slots_per_wave):
            slot_ok = slots[j] < fx.Int32(fz_k)
            safe_slot = slot_ok.select(slots[j], fx.Int32(0))
            seg = route_segment[safe_token * fx.Int32(fz_k) + safe_slot]
            segs.append(slot_ok.select(seg, fx.Int32(-1)))
        _tok_fill_routes(
            token, token_ok, slots, segs, pair_order, count_scratch, fz_k=fz_k,
            fz_total_experts=fz_total_experts, packed_topk6=packed_topk6,
            wt_owner=wt_owner,
        )


def _emit_fused_fill_tok(
    *, num_waves, fz_npes, fz_k, fz_total_experts, addr_disp, i32_cur_tok, count_scratch,
    segments, prepare_blocks=1,
    wt_owner,
):
    pair_order = ptr_buf_tensor(
        _sload_table_i64(addr_disp, int(DispatchSlot.PAIR_ORDER)), fx.Int32
    )
    route_segment = ptr_buf_tensor(
        _sload_table_i64(addr_disp, int(DispatchSlot.ROUTE_SEGMENT)), fx.Int32
    )
    count_segments = fz_total_experts + fz_npes
    packed_topk6 = fz_k <= 6 and count_segments <= 512
    split, groups, slots_per_wave, lane, group, sub, token_off = _tok_lane_geometry(  # noqa: RUF059
        num_waves, prepare_blocks, fz_k
    )
    pass_tokens = groups * 64
    token_ok = token_off < i32_cur_tok
    _tok_fill_routes(
        token_off, token_ok,
        _tok_slot_ids(sub, split=split, slots_per_wave=slots_per_wave),
        segments, pair_order, count_scratch, fz_k=fz_k,
        fz_total_experts=fz_total_experts, packed_topk6=packed_topk6,
        wt_owner=wt_owner,
    )
    _tok_fill_rest(
        fx.Int32(pass_tokens), fx.Int32(pass_tokens), token_off, sub, pair_order,
        route_segment, count_scratch, i32_cur_tok,
        fz_k=fz_k, fz_total_experts=fz_total_experts, packed_topk6=packed_topk6,
        split=split, slots_per_wave=slots_per_wave,
        wt_owner=wt_owner,
    )


@flyc.jit
def _zero_lds_i32(scratch, tid, block_threads, *, count_segments):
    for segment in range(tid, count_segments, block_threads):
        fx.ptr_store(fx.Int32(0), scratch + fx.Int64(segment))


# Producer (group CTA) count without lane exchanges or per-route pair loads:
# the pair table is loaded once per lane, each lane loads its token's full
# top-k row, and _GROUP_COUNT_PF route batches are in flight together.
_GROUP_COUNT_PF = 4
# The count-phase BLOCK_HIST row is the only data the owner consumes after the
# first GROUP_DONE increment.  Writing it with system-coherent (write-through)
# relaxed stores and draining them (s_waitcnt vmcnt(0) + CTA barrier) before
# the GROUP_DONE atomic makes it visible to the owner's agent acquire without
# the agent release fence (an L2 writeback of every dirty line, including the
# concurrent quant output).
# The group count keeps its packed segments in registers across the
# PAIR_READY wait when one prefetch chunk covers fz_mtpr tokens (no
# ROUTE_SEGMENT store/reload round trip, no per-iteration vmcnt(0) in the
# fill loop).


def _group_load_rows(chunk_base, token_stride, lane, idx_buffer, i32_cur_tok, *, fz_k):
    rows = []
    for b in range_constexpr(_GROUP_COUNT_PF):
        token = chunk_base + fx.Int32(b) * token_stride + (lane >> fx.Int32(3))
        safe_token = (token < i32_cur_tok).select(token, fx.Int32(0))
        row_base = safe_token * fx.Int32(fz_k)
        rows.append(
            [idx_buffer[row_base + fx.Int32(slot)] for slot in range_constexpr(fz_k)]
        )
    return rows


def _group_classify_rows(
    chunk_base, token_stride, rows, lane, topk_slot, active_slot, pair_vals,
    route_segment, count_scratch, i32_cur_tok,
    *, fz_npes, fz_k, fz_epr, fz_total_experts, packed_topk6_metadata,
):
    for b in range_constexpr(_GROUP_COUNT_PF):
        _group_classify_one(
            chunk_base + fx.Int32(b) * token_stride, rows[b], lane, topk_slot,
            active_slot, pair_vals, route_segment, count_scratch, i32_cur_tok,
            fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            packed_topk6_metadata=packed_topk6_metadata,
        )


@flyc.jit
def _group_classify_one(
    token_batch, row, lane, topk_slot, active_slot, pair_vals,
    route_segment, count_scratch, i32_cur_tok,
    *, fz_npes, fz_k, fz_epr, fz_total_experts, packed_topk6_metadata,
):
    # Wave-uniform skip of batches past cur_tok (small buckets): the
    # classification code of empty prefetched batches is never executed.
    if token_batch < i32_cur_tok:
        safe_slot = active_slot.select(topk_slot, fx.Int32(0))
        cached_segment = _classify_regs_batch(
            token_batch, row, lane, topk_slot, active_slot, pair_vals,
            count_scratch, i32_cur_tok,
            fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            route_group_shift=3,
            packed_topk6_metadata=packed_topk6_metadata,
        )
        token = token_batch + (lane >> fx.Int32(3))
        active_route = active_slot & (token < i32_cur_tok)
        route = token * fx.Int32(fz_k) + safe_slot
        if active_route:
            route_segment[route] = cached_segment


@flyc.jit
def _group_classify_held(
    token_batch, row, lane, topk_slot, active_slot, pair_vals,
    count_scratch, i32_cur_tok,
    *, fz_npes, fz_k, fz_epr, fz_total_experts, packed_topk6_metadata,
):
    held = fx.Int32(-1)
    if token_batch < i32_cur_tok:
        held = _classify_regs_batch(
            token_batch, row, lane, topk_slot, active_slot, pair_vals,
            count_scratch, i32_cur_tok,
            fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            route_group_shift=3,
            packed_topk6_metadata=packed_topk6_metadata,
        )
    return held


def _group_count_held(
    token_batch0, token_stride, lane, topk_slot, active_slot,
    idx_buffer, addr_pair_config, parity, count_scratch,
    i32_cur_tok, tid, block_threads,
    *, fz_npes, fz_k, fz_epr, fz_total_experts, packed_topk6_metadata, count_segments,
):
    """_group_count_regs when ONE prefetch chunk covers every token (cur_tok
    <= fz_mtpr <= _GROUP_COUNT_PF * token_stride): the packed segments stay in
    registers for the fill, so ROUTE_SEGMENT is neither stored nor reloaded
    (it is private to this CTA)."""
    pair_lane_packed = _load_pair_table_lane(
        addr_pair_config, parity, lane, fz_npes=fz_npes
    )
    rows0 = _group_load_rows(
        token_batch0, token_stride, lane, idx_buffer, i32_cur_tok, fz_k=fz_k
    )
    _zero_lds_i32(count_scratch, tid, block_threads, count_segments=count_segments)
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.barrier()
    pair_vals = [
        fx.Int32(fx.rocdl.readlane(T.i32, pair_lane_packed, d))
        for d in range_constexpr(fz_npes)
    ]
    held = []
    for b in range_constexpr(_GROUP_COUNT_PF):
        held.append(fx.Int32(_group_classify_held(
            token_batch0 + fx.Int32(b) * token_stride, rows0[b], lane, topk_slot,
            active_slot, pair_vals, count_scratch, i32_cur_tok,
            fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            packed_topk6_metadata=packed_topk6_metadata,
        )))
    return held


@flyc.jit
def _group_fill_held_one(
    token, packed_segment, safe_slot, active_slot, a_pair_order, count_scratch,
    *, fz_k, fz_total_experts, packed_topk6_metadata,
):
    if const_expr(packed_topk6_metadata):
        emit = active_slot & (packed_segment >= fx.Int32(0))
        segment = packed_segment & fx.Int32(0x1FF)
        member_slots = (packed_segment >> fx.Int32(9)) & fx.Int32(0x3F)
        intra_rank = (packed_segment >> fx.Int32(15)) & fx.Int32(0xFFFF)
    else:
        emit = active_slot & (packed_segment != fx.Int32(-1))
        segment = packed_segment & fx.Int32(0x3FF)
        member_slots = (packed_segment >> fx.Int32(10)) & fx.Int32(0xFF)
        intra_rank = (packed_segment >> fx.Int32(18)) & fx.Int32(0x3FFF)
    if emit:
        block_base = fx.ptr_load(count_scratch + fx.Int64(segment))
        position = block_base + intra_rank
        shared_group = segment >= fx.Int32(fz_total_experts)
        group_entry = token | (member_slots << fx.Int32(24))
        route = token * fx.Int32(fz_k) + safe_slot
        pair_entry = shared_group.select(group_entry, route)
        _store_i32_agent_relaxed(a_pair_order, position, pair_entry)


@flyc.jit
def _group_count_regs(
    token_batch0, token_stride, lane, topk_slot, active_slot,
    idx_buffer, route_segment, addr_pair_config, parity, count_scratch,
    i32_cur_tok, tid, block_threads,
    *, fz_npes, fz_k, fz_epr, fz_total_experts, packed_topk6_metadata, count_segments,
):
    pair_lane_packed = _load_pair_table_lane(
        addr_pair_config, parity, lane, fz_npes=fz_npes
    )
    rows0 = _group_load_rows(
        token_batch0, token_stride, lane, idx_buffer, i32_cur_tok, fz_k=fz_k
    )
    _zero_lds_i32(count_scratch, tid, block_threads, count_segments=count_segments)
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.barrier()
    pair_vals = [
        fx.Int32(fx.rocdl.readlane(T.i32, pair_lane_packed, d))
        for d in range_constexpr(fz_npes)
    ]
    _group_classify_rows(
        token_batch0, token_stride, rows0, lane, topk_slot, active_slot, pair_vals,
        route_segment, count_scratch, i32_cur_tok,
        fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
        fz_total_experts=fz_total_experts,
        packed_topk6_metadata=packed_topk6_metadata,
    )
    chunk_stride = token_stride * fx.Int32(_GROUP_COUNT_PF)
    for chunk_base in range(token_batch0 + chunk_stride, i32_cur_tok, chunk_stride):
        rows = _group_load_rows(
            chunk_base, token_stride, lane, idx_buffer, i32_cur_tok, fz_k=fz_k
        )
        _group_classify_rows(
            chunk_base, token_stride, rows, lane, topk_slot, active_slot, pair_vals,
            route_segment, count_scratch, i32_cur_tok,
            fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            packed_topk6_metadata=packed_topk6_metadata,
        )



def _gsp_enabled(prefix_lds, count_segments, group_blocks, num_waves):
    return (
        prefix_lds is not None
        and count_segments % 4 == 0
        and (count_segments + 63) // 64 <= num_waves
        and group_blocks % num_waves == 0
    )


@flyc.jit
def _spin_counter_reached(addr_i64, target):
    """Spin until the monotonically increasing counter is >= target (wrap
    safe): later phases may already have advanced it past target."""
    cur = fx.Int32(comm_ops.load_i32_acquire(addr_i64))
    while (cur - target) < fx.Int32(0):
        cur = fx.Int32(comm_ops.load_i32_acquire(addr_i64))


@flyc.jit
def _group_self_bases(
    a_block_hist, before_lds, total_lds, producer_slot, lane, warp,
    *, num_waves, group_blocks, count_segments,
):
    """Wave w reduces BLOCK_HIST rows g = w, w + num_waves, ... with dwordx4
    lane = 4-segment loads: LDS before[s] += count[g][s] for g < own slot,
    total[s] += count[g][s]."""
    block_hist4 = ptr_buf_tensor(a_block_hist, fx.Int32, unit_elems=4)
    row_units = count_segments // 4
    chunks = (row_units + 63) // 64
    loaded = []
    for j in range_constexpr(group_blocks // num_waves):
        g = warp + fx.Int32(j * num_waves)
        for c in range_constexpr(chunks):
            q = fx.Int32(c * 64) + lane
            q_ok = q < fx.Int32(row_units)
            safe_q = q_ok.select(q, fx.Int32(0))
            v = buf_copy_load(
                block_hist4, g * fx.Int32(row_units) + safe_q, fx.Int32, unit_elems=4
            )
            loaded.append((g, c, q_ok, safe_q, v))
    before_parts = {}
    total_parts = {}
    for g, c, q_ok, safe_q, v in loaded:
        is_before = g < producer_slot
        for e in range_constexpr(4):
            val = fx.Int32(v[e])
            key = (c, e)
            bval = is_before.select(val, fx.Int32(0))
            if key in total_parts:
                total_parts[key] = total_parts[key] + val
                before_parts[key] = before_parts[key] + bval
            else:
                total_parts[key] = val
                before_parts[key] = bval
    for c in range_constexpr(chunks):
        q = fx.Int32(c * 64) + lane
        if q < fx.Int32(row_units):
            for e in range_constexpr(4):
                seg = q * fx.Int32(4) + fx.Int32(e)
                b_addr = fx.Int64(fx.ptrtoint(before_lds)) + fx.Int64(seg) * fx.Int64(4)
                t_addr = fx.Int64(fx.ptrtoint(total_lds)) + fx.Int64(seg) * fx.Int64(4)
                comm_ops.atomic_add_workgroup(b_addr, before_parts[(c, e)])
                comm_ops.atomic_add_workgroup(t_addr, total_parts[(c, e)])


@flyc.jit
def _group_base_round(before_lds, total_lds, lane, rnd, *, count_segments):
    """Segments rnd*64 + lane: base[s] = sum_{s' < s} total[s'] + before[s],
    written over before[s] (the fill reads its bases from there)."""
    seg_carry = fx.Int32(0)
    seg_count = fx.Int32(0)
    for prev in range_constexpr(rnd + 1):
        pseg = fx.Int32(prev * 64) + lane
        pvalid = pseg < fx.Int32(count_segments)
        psafe = pvalid.select(pseg, fx.Int32(0))
        pcount = pvalid.select(
            fx.Int32(fx.ptr_load(total_lds + fx.Int64(psafe))), fx.Int32(0)
        )
        if const_expr(prev < rnd):
            seg_carry = seg_carry + _wave_sum_uniform_i32(pcount, lane)
        else:
            seg_count = pcount
    seg = fx.Int32(rnd * 64) + lane
    valid_seg = seg < fx.Int32(count_segments)
    safe_seg = valid_seg.select(seg, fx.Int32(0))
    seg_incl = _wave_inclusive_scan_i32(seg_count, lane)
    before = fx.Int32(fx.ptr_load(before_lds + fx.Int64(safe_seg)))
    base = seg_carry + seg_incl - seg_count + before
    if valid_seg:
        fx.ptr_store(base, before_lds + fx.Int64(seg))


# fmt: off
@flyc.jit
def _emit_group(
    *, num_waves, fz_npes, fz_k, fz_epr, fz_total_experts, addr_disp, i32_cur_tok, addr_in_idx,
    dispatch_blocks, group_done_slot, producer_slot, parity, expected,
    count_scratch=None,
    group_phase_base=None,
    addr_expected=None,
    large_bucket=False,
    fz_mtpr=1 << 30,
    prefix_lds=None,
):
# fmt: on
    """Count and group disjoint route spans across payload producer CTAs."""
    def dp(i):
        return _sload_table_i64(addr_disp, int(i))

    a_pair_ready = dp(DispatchSlot.PAIR_READY)
    a_block_hist = dp(DispatchSlot.BLOCK_HIST)
    a_pair_order = dp(DispatchSlot.PAIR_ORDER)
    a_group_done = dp(DispatchSlot.GROUP_DONE) + fx.Int64(group_done_slot * 4)
    a_route_segment = dp(DispatchSlot.ROUTE_SEGMENT)
    addr_pair_config = dp(DispatchSlot.FANOUT_PAIR_CONFIG)
    idx_buffer = ptr_buf_tensor(
        addr_in_idx, fx.Int32, num_records_bytes=_idx_bytes(i32_cur_tok, fz_k)
    )
    block_hist = ptr_buf_tensor(a_block_hist, fx.Int32)
    route_segment = ptr_buf_tensor(a_route_segment, fx.Int32)
    tid = fx.thread_idx.x
    block_threads = fx.Int32(num_waves * 64)
    count_segments = fz_total_experts + fz_npes
    assert count_segments <= 1024, "route metadata supports at most 1024 segments"
    packed_topk6_metadata = fz_k <= 6 and count_segments <= 512
    assert count_scratch is not None
    gsp = _gsp_enabled(prefix_lds, count_segments, dispatch_blocks, num_waves)
    def record_count(segment):
        scratch_addr = fx.Int64(fx.ptrtoint(count_scratch))
        return fx.Int32(
            comm_ops.atomic_add_workgroup(
                scratch_addr + fx.Int64(segment) * fx.Int64(4),
                fx.Int32(1),
            )
        )

    tid = fx.thread_idx.x
    if const_expr(addr_expected is not None):
        # Issued here, consumed only by the PAIR_READY wait: the load's
        # latency hides behind the counting phase.
        expected = fx.Int32(comm_ops.load_i32_system(addr_expected, parity))

    assert 0 < fz_k <= 16, "wave-grouped fanout requires topk <= 16"
    route_group_width = 8 if fz_k <= 8 else 16
    route_group_shift = 3 if route_group_width == 8 else 4
    tokens_per_wave = 64 // route_group_width
    lane = tid & fx.Int32(63)
    warp = tid >> fx.Int32(6)
    wave_id = producer_slot * fx.Int32(num_waves) + warp
    token_batch0 = wave_id * fx.Int32(tokens_per_wave)
    token_stride = fx.Int32(dispatch_blocks * num_waves * tokens_per_wave)
    topk_slot = lane & fx.Int32(route_group_width - 1)
    active_slot = topk_slot < fx.Int32(fz_k)
    group_regs = (
        large_bucket
        and fz_epr <= 64
        and 0 < fz_k <= 8
        and fz_npes <= 64
    )
    held_static_stride = dispatch_blocks * num_waves * tokens_per_wave
    group_held = (
        group_regs
        and route_group_width == 8
        and fz_mtpr <= _GROUP_COUNT_PF * held_static_stride
    )
    held = None
    if const_expr(group_held):
        held = _group_count_held(
            token_batch0, token_stride, lane, topk_slot, active_slot,
            idx_buffer, addr_pair_config, parity, count_scratch,
            i32_cur_tok, tid, block_threads,
            fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            packed_topk6_metadata=packed_topk6_metadata,
            count_segments=count_segments,
        )
    elif const_expr(group_regs):
        _group_count_regs(
            token_batch0, token_stride, lane, topk_slot, active_slot,
            idx_buffer, route_segment, addr_pair_config, parity, count_scratch,
            i32_cur_tok, tid, block_threads,
            fz_npes=fz_npes, fz_k=fz_k, fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            packed_topk6_metadata=packed_topk6_metadata,
            count_segments=count_segments,
        )
    else:
      for segment in range(tid, count_segments, block_threads):
        fx.ptr_store(fx.Int32(0), count_scratch + fx.Int64(segment))
      fx.barrier()
      for token_batch in range(token_batch0, i32_cur_tok, token_stride):
        token = token_batch + (lane >> fx.Int32(route_group_shift))
        active_route = active_slot & (token < i32_cur_tok)
        safe_token = (token < i32_cur_tok).select(token, fx.Int32(0))
        safe_slot = active_slot.select(topk_slot, fx.Int32(0))
        route = safe_token * fx.Int32(fz_k) + safe_slot
        expert = idx_buffer[route]
        valid = active_route & (expert >= fx.Int32(0))
        valid = valid & (expert < fx.Int32(fz_total_experts))
        safe_expert = valid.select(expert, fx.Int32(0))
        segment, emit, member_slots = _classify_fanout_wave_route(
            safe_expert,
            expert,
            lane,
            addr_pair_config,
            parity,
            fz_k=fz_k,
            fz_epr=fz_epr,
            fz_total_experts=fz_total_experts,
            fz_npes=fz_npes,
        )
        cached_segment = fx.Int32(-1)
        if valid & emit:
            intra_rank = record_count(segment)
            if const_expr(packed_topk6_metadata):
                cached_segment = (
                    segment
                    | (member_slots << fx.Int32(9))
                    | (intra_rank << fx.Int32(15))
                )
            else:
                cached_segment = (
                    segment
                    | (member_slots << fx.Int32(10))
                    | (intra_rank << fx.Int32(18))
                )
        if active_route:
            route_segment[route] = cached_segment

    # Only the LDS counts must be complete at the barrier; the ROUTE_SEGMENT
    # stores drain together with the BLOCK_HIST stores below.
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    fx.barrier()
    for segment in range(tid, count_segments, block_threads):
        block_count = fx.ptr_load(count_scratch + fx.Int64(segment))
        block_index = producer_slot * fx.Int32(count_segments) + segment
        _store_i32_sys_relaxed(a_block_hist, block_index, block_count)
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()
    if const_expr(gsp):
        # Self prefix: every group derives its own BLOCK_HIST bases from all
        # groups' counts as soon as the last group has counted, so the fill
        # runs under the owner's XGMI exchange/plan instead of after
        # PAIR_READY.  PAIR_READY is still awaited before the fill-done
        # arrival (PAIR_ORDER_READY implies the plan).
        if tid == fx.Int32(0):
            comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
            _spin_counter_reached(
                a_group_done, group_phase_base + fx.Int32(dispatch_blocks)
            )
            comm_ops.fence_agent_acquire()
        for segment in range(tid, count_segments, block_threads):
            fx.ptr_store(fx.Int32(0), count_scratch + fx.Int64(segment))
            fx.ptr_store(fx.Int32(0), prefix_lds + fx.Int64(segment))
        fx.barrier()
        _group_self_bases(
            a_block_hist, count_scratch, prefix_lds, producer_slot, lane, warp,
            num_waves=num_waves, group_blocks=dispatch_blocks,
            count_segments=count_segments,
        )
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.barrier()
        seg_rounds_g = (count_segments + 63) // 64
        for rnd in range_constexpr(seg_rounds_g):
            if warp == fx.Int32(rnd % num_waves):
                _group_base_round(
                    count_scratch, prefix_lds, lane, rnd,
                    count_segments=count_segments,
                )
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.barrier()
    else:
      if tid == fx.Int32(0):
        comm_ops.atomic_add_agent(a_group_done, fx.Int32(1))
        comm_ops.wait_i32_until_equals(
            a_pair_ready + fx.Int64(parity) * fx.Int64(4), expected
        )
        comm_ops.fence_agent_acquire()
      fx.barrier()

      for segment in range(tid, count_segments, block_threads):
        block_index = producer_slot * fx.Int32(count_segments) + segment
        block_base = block_hist[block_index]
        fx.ptr_store(block_base, count_scratch + fx.Int64(segment))
      fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
      fx.barrier()
    if const_expr(group_held):
        safe_slot_h = active_slot.select(topk_slot, fx.Int32(0))
        for b in range_constexpr(_GROUP_COUNT_PF):
            _group_fill_held_one(
                token_batch0 + fx.Int32(b) * token_stride
                + (lane >> fx.Int32(route_group_shift)),
                held[b], safe_slot_h, active_slot, a_pair_order, count_scratch,
                fz_k=fz_k, fz_total_experts=fz_total_experts,
                packed_topk6_metadata=packed_topk6_metadata,
            )
    else:
      for token_batch in range(
          token_batch0, i32_cur_tok, token_stride
      ):
          token = token_batch + (lane >> fx.Int32(route_group_shift))
          active_route = active_slot & (token < i32_cur_tok)
          safe_token = (token < i32_cur_tok).select(token, fx.Int32(0))
          safe_slot = active_slot.select(topk_slot, fx.Int32(0))
          route = safe_token * fx.Int32(fz_k) + safe_slot
          packed_segment = route_segment[route]
          if const_expr(packed_topk6_metadata):
              emit = active_route & (packed_segment >= fx.Int32(0))
              segment = packed_segment & fx.Int32(0x1FF)
              member_slots = (packed_segment >> fx.Int32(9)) & fx.Int32(0x3F)
              intra_rank = (packed_segment >> fx.Int32(15)) & fx.Int32(0xFFFF)
          else:
              emit = active_route & (packed_segment != fx.Int32(-1))
              segment = packed_segment & fx.Int32(0x3FF)
              member_slots = (packed_segment >> fx.Int32(10)) & fx.Int32(0xFF)
              intra_rank = (packed_segment >> fx.Int32(18)) & fx.Int32(0x3FFF)
          if emit:
              block_base = fx.ptr_load(count_scratch + fx.Int64(segment))
              position = block_base + intra_rank
              shared_group = segment >= fx.Int32(fz_total_experts)
              group_entry = token | (member_slots << fx.Int32(24))
              pair_entry = shared_group.select(group_entry, route)
              # Agent-coherent (sc1) store; see _store_i32_agent_relaxed.
              _store_i32_agent_relaxed(a_pair_order, position, pair_entry)

    # Every lane's sc1 stores have completed (agent-visible) before the
    # barrier, so the GROUP_DONE arrival below needs no agent release fence
    # (its buffer_wbl2 sc1 would flush the quant output out of this XCD's
    # L2).  The last group CTA still acquires
    # and publishes PAIR_ORDER_READY with a system release store.
    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
    fx.barrier()
    if tid == fx.Int32(0):
        if const_expr(gsp):
            comm_ops.wait_i32_until_equals(
                a_pair_ready + fx.Int64(parity) * fx.Int64(4), expected
            )
            comm_ops.fence_agent_acquire()
        done_before = fx.Int32(comm_ops.atomic_add_agent(a_group_done, fx.Int32(1)))
        # The last group CTA to finish the fill publishes PAIR_ORDER_READY
        # (the owner does not poll GROUP_DONE).  Acquire the other
        # groups' agent-released fills before the system release-store.
        if done_before == group_phase_base + fx.Int32(dispatch_blocks * 2 - 1):
            comm_ops.fence_agent_acquire()
            comm_ops.store_i32_system(
                dp(DispatchSlot.PAIR_ORDER_READY), parity, expected
            )



# Fused owner: the epoch-state resets go out as one multi-lane store.
_EPOCH_STORE_LANES = 14




def _fast_prepare_supported(npes, epr, topk):
    """Layouts the fast prepare path is specialised for.

    Its fused-owner and register-held count paths assume EP8, at most 64
    experts per rank and top-k <= 8 (DeepSeek-V4-Pro, R1, GLM-5.2).  Every
    other layout (EP2/EP4, Kimi-K3's 112 experts per rank with top-k 16)
    compiles the stock kernel.
    """
    return int(npes) == 8 and int(epr) <= 64 and int(topk) <= 8


def _m64_qdpp_bucket(tile_m, prepare_blocks):
    return int(tile_m) == 64 and int(prepare_blocks) == _QDPP_M64_P


def _m64_qdpp_variant(kwargs):
    """True when this launch configuration has a second, qdpp variant."""
    return (
        kwargs.get("quant_mode", "fp8") == "fp8"
        and int(kwargs.get("num_quant_cu", 0)) > 0
        and _m64_qdpp_bucket(kwargs["sort_block_m"], kwargs["num_prepare_cu"])
        and _fast_prepare_supported(
            kwargs["fuse_npes"], kwargs["experts_per_rank"], kwargs["fuse_topk"]
        )
    )


def _m64_qdpp_for(i32_cur_tok, kwargs):
    """Host-side shape gate: the m64 P32 bucket takes the qdpp variant only
    at _QDPP_M64_MIN_TOK live tokens or more."""
    if not _m64_qdpp_variant(kwargs):
        return False
    cur = int(getattr(i32_cur_tok, "value", i32_cur_tok))
    return cur >= _QDPP_M64_MIN_TOK


@functools.cache
def compile_mega_moe_prepare(
    *,
    rank: int,
    experts_per_rank: int,
    fuse_npes: int,
    fuse_topk: int,
    fuse_mtpr: int,
    sort_block_m: int,
    num_dispatch_cu: int,
    num_prepare_cu: int,
    num_quant_cu: int,
    quant_cu_capacity: int,
    quant_mode: str = "fp8",
    model_dim: int,
    payload_chunk_rows: int,
    tile_state_stride: int,
    m64_qdpp: bool = False,
):
    """Compile compact count/group/plan without the GEMM1 shared footprint."""
    arch = str(get_rocm_arch() or "")
    if not arch.startswith("gfx95"):
        raise RuntimeError(
            f"MegaMoE prepare requires CDNA4 (gfx95x), got {arch or 'unknown'}"
        )
    npes = int(fuse_npes)
    epr = int(experts_per_rank)
    topk = int(fuse_topk)
    mtpr = int(fuse_mtpr)
    rank = int(rank)
    tile_m = int(sort_block_m)
    dispatch_blocks = int(num_dispatch_cu)
    prepare_blocks = int(num_prepare_cu)
    quant_blocks = int(num_quant_cu)
    quant_cu_capacity = int(quant_cu_capacity)
    if quant_mode not in ("fp4", "fp8"):
        raise ValueError(f"quant_mode must be fp4|fp8, got {quant_mode!r}")
    if _fast_prepare_supported(npes, epr, topk):
        return _compile_fast_mega_moe_prepare(
            rank=rank,
            experts_per_rank=epr,
            fuse_npes=npes,
            fuse_topk=topk,
            fuse_mtpr=mtpr,
            sort_block_m=tile_m,
            num_dispatch_cu=dispatch_blocks,
            num_prepare_cu=prepare_blocks,
            num_quant_cu=quant_blocks,
            quant_cu_capacity=quant_cu_capacity,
            quant_mode=quant_mode,
            model_dim=int(model_dim),
            payload_chunk_rows=int(payload_chunk_rows),
            tile_state_stride=int(tile_state_stride),
            m64_qdpp=m64_qdpp,
        )
    model_dim = int(model_dim)
    num_waves = 8
    chunk_rows = int(payload_chunk_rows)
    tile_state_stride = int(tile_state_stride)
    total_experts = npes * epr
    total_segments = total_experts + npes
    block_threads = num_waves * 64
    assert prepare_blocks >= 1
    launch_grid = prepare_blocks + quant_blocks + 1
    assert dispatch_blocks % npes == 0
    assert 0 <= quant_blocks <= quant_cu_capacity
    assert model_dim % 32 == 0
    assert chunk_rows > 0
    assert tile_state_stride > 0

    @fx.struct
    class SharedStorage:
        ticket: fx.Array[fx.Int64, 1, 8]
        count_scratch: fx.Array[fx.Int32, total_segments, 16]

    quant_suffix = "_qfp4" if quant_mode == "fp4" else ""
    kernel_name = (
        f"megamoe_prepare_compact_m{tile_m}_dcu{dispatch_blocks}_pcu{prepare_blocks}_pc{chunk_rows}"
        f"_qcu{quant_blocks}qcap{quant_cu_capacity}{quant_suffix}"
        "_fov_runtime_dyn"
        f"_tss{tile_state_stride}_v13"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def kernel(
        addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_in_idx: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        addr_quant_in: fx.Int64,
        addr_quant_out: fx.Int64,
        addr_quant_scale: fx.Int64,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ticket_scratch = fx.recast_iter(fx.Int64, lds.ticket.ptr)
        count_scratch = lds.count_scratch.ptr
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        disp_rsrc = ptr_buf_tensor(addr_disp, fx.Int64)

        def disp_ptr(slot):
            return disp_rsrc[fx.Int32(int(slot))]

        entry_slot = prepare_blocks * (quant_cu_capacity + 1) + quant_blocks
        entry_count = disp_ptr(DispatchSlot.PREP_ENTRY_COUNT) + fx.Int64(entry_slot * 8)
        epoch_gate = disp_ptr(DispatchSlot.PREP_EPOCH_GATE) + fx.Int64(entry_slot * 4)
        work_head = disp_ptr(DispatchSlot.WORK_HEAD)
        work_tail = disp_ptr(DispatchSlot.WORK_TAIL)
        ready_tile_tail = disp_ptr(DispatchSlot.READY_TILE_TAIL)
        payload_ready_rows = disp_ptr(DispatchSlot.PAYLOAD_READY_ROWS)
        local_hist = disp_ptr(DispatchSlot.LOCAL_HIST)

        if tid == fx.Int32(0):
            ticket64 = fx.Int64(comm_ops.atomic_add_agent(entry_count, fx.Int64(1)))
            fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
        fx.barrier()
        ticket64 = Vec(ticket_view.load())[0]
        generation = ticket64 // fx.Int64(launch_grid)
        ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid))
        gate_epoch = fx.Int32(generation + fx.Int64(1))
        owner = ticket == 0
        producer = (ticket > 0) & (ticket <= fx.Int32(prepare_blocks))
        producer_slot = ticket - 1
        quant_producer = ticket > fx.Int32(prepare_blocks)
        quant_slot = ticket - fx.Int32(prepare_blocks + 1)

        if const_expr(quant_blocks > 0):  # noqa: SIM102 - preserve DSL staging
            if quant_producer:
                quant_in = ptr_buf_tensor(addr_quant_in, fx.Int32, unit_elems=4)
                quant_out = ptr_buf_tensor(addr_quant_out, fx.Int32, unit_elems=4)
                quant_scale = ptr_buf_tensor(addr_quant_scale, fx.Uint8)
                scale_dim = model_dim // 32
                total_groups = i32_cur_tok * fx.Int32(scale_dim)
                group_stride = fx.Int32(quant_blocks * block_threads)
                group0 = quant_slot * fx.Int32(block_threads) + tid
                for group_id in range(group0, total_groups, group_stride):
                    if const_expr(quant_mode == "fp4"):
                        emit_per_1x32_mx_fp4_group(
                            quant_in, quant_out, quant_scale, group_id
                        )
                    else:
                        emit_per_1x32_mx_fp8_group(
                            quant_in, quant_out, quant_scale, group_id
                        )

        # Quant CTAs are independent specialists.  They must retire as soon as
        # their quant groups drain instead of entering prepare's epoch/barrier
        # protocol.  The queued quant CTAs can then immediately occupy CUs
        # released by owner/group CTAs.
        if not quant_producer:
            if owner:
                next_parity_lane = fx.Int32(0)
                if tid == fx.Int32(0):
                    old_parity = comm_ops.load_i32_system(addr_parity, fx.Int32(0))
                    next_parity_lane = old_parity ^ fx.Int32(1)
                    previous_expected = comm_ops.load_i32_system(
                        addr_expected,
                        next_parity_lane,
                    )
                    next_expected = previous_expected + fx.Int32(npes)
                    comm_ops.store_i32_system(
                        addr_expected,
                        next_parity_lane,
                        next_expected,
                    )
                next_parity = fx.Int32(fx.rocdl.readfirstlane(T.i32, next_parity_lane))
                if tid == fx.Int32(0):
                    comm_ops.store_i32_system(
                        payload_ready_rows, fx.Int32(0), fx.Int32(tile_m)
                    )
                    comm_ops.fence_system_release()
                fx.barrier()
                if tid == fx.Int32(0):
                    work_head_rsrc = ptr_buf_tensor(work_head, fx.Int32)
                    for work_shard in range(8):
                        work_head_rsrc[fx.Int32(work_shard * 16)] = fx.Int32(0)
                    comm_ops.store_i32_system(work_tail, fx.Int32(0), fx.Int32(0))
                    # Payload publishers reserve queue slots with a system-scope
                    # atomic.  Reset the tail in that same coherence domain so a
                    # layout transition cannot append after the previous epoch.
                    comm_ops.store_i32_system(ready_tile_tail, next_parity, fx.Int32(0))
                hist_rsrc = ptr_buf_tensor(local_hist, fx.Int32)
                for segment in range(tid, total_segments, block_threads):
                    hist_rsrc[segment] = fx.Int32(0)
                fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                fx.barrier()
                if tid == fx.Int32(0):
                    comm_ops.store_i32_system(
                        addr_parity,
                        fx.Int32(0),
                        next_parity,
                    )
                    comm_ops.store_i32_system(epoch_gate, fx.Int32(0), gate_epoch)
            else:
                if tid == fx.Int32(0):
                    comm_ops.wait_i32_until_equals(epoch_gate, gate_epoch)
                    comm_ops.fence_agent_acquire()
                fx.barrier()

            # The owner publishes parity from lane 0.  Converge the complete
            # CTA before any lane consumes it; otherwise the owner lanes can
            # classify and derive offsets with different epochs.
            fx.barrier()
            comm_ops.fence_system_acquire()
            parity = comm_ops.load_i32_system(addr_parity, fx.Int32(0))
            expected = comm_ops.load_i32_system(addr_expected, parity)
            group_phase_base = fx.Int32(generation) * fx.Int32(prepare_blocks * 2)
            if owner:
                emit_dispatch_plan(
                    num_waves=num_waves,
                    fz_npes=npes,
                    fz_epr=epr,
                    fz_k=topk,
                    fz_mtpr=mtpr,
                    fz_rank=rank,
                    fz_tile_m=tile_m,
                    fz_total_experts=total_experts,
                    addr_disp=addr_disp,
                    i32_cur_tok=i32_cur_tok,
                    addr_in_idx=addr_in_idx,
                    parity=parity,
                    expected=expected,
                    dispatch_blocks=dispatch_blocks,
                    group_blocks=prepare_blocks,
                    group_done_slot=entry_slot,
                    group_phase_base=group_phase_base,
                    payload_chunk_rows=chunk_rows,
                    tile_state_stride=tile_state_stride,
                )
            if producer:
                emit_dispatch_group(
                    num_waves=num_waves,
                    fz_npes=npes,
                    fz_k=topk,
                    fz_epr=epr,
                    fz_total_experts=total_experts,
                    addr_disp=addr_disp,
                    i32_cur_tok=i32_cur_tok,
                    addr_in_idx=addr_in_idx,
                    dispatch_blocks=prepare_blocks,
                    group_done_slot=entry_slot,
                    producer_slot=producer_slot,
                    parity=parity,
                    expected=expected,
                    count_scratch=count_scratch,
                )

    @flyc.jit
    def launch(
        addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_in_idx: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        addr_quant_in: fx.Int64,
        addr_quant_out: fx.Int64,
        addr_quant_scale: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            addr_disp,
            i32_cur_tok,
            addr_in_idx,
            addr_parity,
            addr_expected,
            addr_quant_in,
            addr_quant_out,
            addr_quant_scale,
        ).launch(
            grid=(launch_grid, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def _compile_fast_mega_moe_prepare(
    *,
    rank: int,
    experts_per_rank: int,
    fuse_npes: int,
    fuse_topk: int,
    fuse_mtpr: int,
    sort_block_m: int,
    num_dispatch_cu: int,
    num_prepare_cu: int,
    num_quant_cu: int,
    quant_cu_capacity: int,
    quant_mode: str = "fp8",
    model_dim: int,
    payload_chunk_rows: int,
    tile_state_stride: int,
    m64_qdpp: bool = False,
):
    """Fast EP8 prepare (see _fast_prepare_supported); called by
    compile_mega_moe_prepare, which validates the arguments."""
    npes = int(fuse_npes)
    epr = int(experts_per_rank)
    topk = int(fuse_topk)
    mtpr = int(fuse_mtpr)
    rank = int(rank)
    tile_m = int(sort_block_m)
    dispatch_blocks = int(num_dispatch_cu)
    prepare_blocks = int(num_prepare_cu)
    quant_blocks = int(num_quant_cu)
    quant_cu_capacity = int(quant_cu_capacity)
    model_dim = int(model_dim)
    num_waves = 8
    chunk_rows = int(payload_chunk_rows)
    tile_state_stride = int(tile_state_stride)
    total_experts = npes * epr
    total_segments = total_experts + npes
    block_threads = num_waves * 64
    assert prepare_blocks >= 1
    launch_grid = prepare_blocks + quant_blocks + 1
    assert dispatch_blocks % npes == 0
    assert 0 <= quant_blocks <= quant_cu_capacity
    assert model_dim % 32 == 0
    assert chunk_rows > 0
    assert tile_state_stride > 0

    m64_qdpp = (
        bool(m64_qdpp)
        and quant_mode == "fp8"
        and _m64_qdpp_bucket(tile_m, prepare_blocks)
    )
    quant_dpp = (
        quant_blocks > 0
        and quant_mode == "fp8"
        and (tile_m >= _QUANT_DPP_MIN_TILE_M or m64_qdpp)
        and model_dim % 128 == 0
    )

    if quant_dpp:
        # Deterministic per-compile-key quant width for the large-token
        # buckets: the m128 owner chain ends well before quant, so quant is
        # the kernel tail.  Grid stays co-resident (1 + P + Q <= CUs) and
        # the entry/epoch slot stays unique (Q <= capacity).
        # NOTE: this supersedes MegaMoEConfig.prepare_quant_cu (passed in as
        # num_quant_cu) for the quant_dpp buckets; the constants assume a
        # 256-CU part and are clamped to quant_cu_capacity - 1 - P.
        q_cap = _QUANT_CU_M128 if tile_m >= 128 else _QUANT_CU_M64
        q_target = min(q_cap, quant_cu_capacity - 1 - prepare_blocks)
        if q_target > quant_blocks:
            quant_blocks = q_target
            launch_grid = prepare_blocks + quant_blocks + 1

    quant_steal = (
        quant_dpp
        and _QUANT_STEAL_ROUNDS > 0
        and prepare_blocks <= quant_cu_capacity
    )

    # Small-P buckets (decode / small prefill) run the single-CTA fused
    # count+plan+fill on the owner: the P-group count -> GROUP_DONE ->
    # PAIR_READY -> fill round trip costs more than the owner counting a few
    # extra 64-token batches itself.  The launch geometry (grid, entry slot,
    # generation math) is unchanged; the P group CTAs retire immediately.
    fused_p1 = (
        prepare_blocks <= _FUSED_MAX_P
    )
    fused_defer = npes == 8 and epr <= 64
    # Route batches (num_waves * 64 / 8 tokens each) whose index loads are
    # issued up front; the compile bucket covers about 64 tokens per P.
    fused_prefetch = max(1, prepare_blocks)
    # Token-per-lane fused count/fill for every fused bucket.
    fused_tok = fused_p1
    tok_extra = _TOK_DUMMY_SEGS if fused_tok else 0
    # Decode (tile_m 32): epoch loads + count prefetch issued before the
    # ticket atomic (see kernel()).
    pre_ticket = (
        fused_tok and tile_m == 32
    )
    no_ticket = pre_ticket
    # Owner publishes with write-through stores and fence-free flags.
    wt_owner = (prepare_blocks <= _WT_MAX_FUSED_P) if fused_p1 else True

    @fx.struct
    class SharedStorage:
        ticket: fx.Array[fx.Int64, 1, 8]
        count_scratch: fx.Array[fx.Int32, total_segments + tok_extra, 16]
        prefix_scratch: fx.Array[fx.Int32, total_segments, 16]
        bcast: fx.Array[fx.Int32, 4, 16]

    quant_suffix = "_qfp4" if quant_mode == "fp4" else ""
    kernel_name = (
        f"megamoe_prepare_compact_m{tile_m}_dcu{dispatch_blocks}_pcu{prepare_blocks}_pc{chunk_rows}"
        f"_qcu{quant_blocks}qcap{quant_cu_capacity}{quant_suffix}"
        "_fov_runtime_dyn"
        f"_tss{tile_state_stride}_v14"
        + (f"_qdpp{_QUANT_CHUNKS_IN_FLIGHT}" if quant_dpp else "")
        + (f"_qs{_QUANT_STEAL_ROUNDS}" if quant_steal else "")
        + (f"_qp{_QUANT_GRAB_AT}" if quant_dpp else "")
        + ("_fp1" if 1 < prepare_blocks <= _FUSED_MAX_P else "")
    )



    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def kernel(
        addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_in_idx: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        addr_quant_in: fx.Int64,
        addr_quant_out: fx.Int64,
        addr_quant_scale: fx.Int64,
    ):
        tid = fx.thread_idx.x
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        ticket_scratch = fx.recast_iter(fx.Int64, lds.ticket.ptr)
        count_scratch = lds.count_scratch.ptr
        prefix_scratch = lds.prefix_scratch.ptr
        bcast_scratch = lds.bcast.ptr
        ticket_view = fx.make_view(ticket_scratch, fx.make_layout(1, 1))
        def disp_ptr(slot):
            return _sload_table_i64(addr_disp, int(slot))

        entry_slot = prepare_blocks * (quant_cu_capacity + 1) + quant_blocks
        entry_count = disp_ptr(DispatchSlot.PREP_ENTRY_COUNT) + fx.Int64(entry_slot * 8)
        epoch_gate = disp_ptr(DispatchSlot.PREP_EPOCH_GATE) + fx.Int64(entry_slot * 4)
        work_head = disp_ptr(DispatchSlot.WORK_HEAD)
        work_tail = disp_ptr(DispatchSlot.WORK_TAIL)
        ready_tile_tail = disp_ptr(DispatchSlot.READY_TILE_TAIL)
        payload_ready_rows = disp_ptr(DispatchSlot.PAYLOAD_READY_ROWS)
        local_hist = disp_ptr(DispatchSlot.LOCAL_HIST)

        pre_old_parity = i32_cur_tok
        pre_expected0 = i32_cur_tok
        pre_expected1 = i32_cur_tok
        pre_tok = None
        if const_expr(pre_ticket):
            # Decode: every CTA issues the owner's epoch-state loads and the
            # count's top-k/pair prefetch BEFORE the ticket atomic, so their
            # latency overlaps the ticket round trip instead of following it
            # (only the owner consumes them; the previous launch's writes are
            # stream-ordered before this kernel).
            if tid < fx.Int32(_EPOCH_STORE_LANES):
                pre_old_parity = comm_ops.load_i32_system(addr_parity, fx.Int32(0))
                pre_expected0 = comm_ops.load_i32_system(addr_expected, fx.Int32(0))
                pre_expected1 = comm_ops.load_i32_system(addr_expected, fx.Int32(1))
            pre_tok = _tok_prefetch(
                num_waves=num_waves,
                fz_npes=npes,
                fz_k=topk,
                fz_total_experts=total_experts,
                addr_disp=addr_disp,
                i32_cur_tok=i32_cur_tok,
                addr_in_idx=addr_in_idx,
                count_scratch=count_scratch,
                prepare_blocks=prepare_blocks,
            )
            # Scalar-cache warm-up of the dispatch table and the peer
            # tables (count push / count done / plan ready) in the
            # shadow of the ticket round trip.
            if (tid >> fx.Int32(6)) == fx.Int32(num_waves - 1):
                _kcache_touch(addr_disp, DISPATCH_TABLE_SIZE * 2)
            if (tid >> fx.Int32(6)) == fx.Int32(num_waves - 2):
                _kcache_touch_tables([
                    disp_ptr(p2p_slot) for p2p_slot in (
                        DispatchSlot.P2P_COUNT_MATRIX,
                        DispatchSlot.P2P_COUNT_DONE,
                        DispatchSlot.P2P_PLAN_READY,
                    )
                ])
        if const_expr(no_ticket):
            # Decode fused path: every role is self-contained (the owner
            # counts, plans and publishes alone; quant CTAs only stride their
            # own slots; nothing waits on another CTA of this launch, and the
            # grid is co-resident), so the role is the block index and the
            # ticket atomic round trip leaves the chain.  This entry slot's
            # epoch gate is never polled on the fused path.
            ticket = fx.Int32(fx.block_idx.x)
            generation = fx.Int64(0)
            gate_epoch = fx.Int32(1)
        else:
            if tid == fx.Int32(0):
                ticket64 = fx.Int64(comm_ops.atomic_add_agent(entry_count, fx.Int64(1)))
                fx.ptr_store(Vec.from_elements([ticket64], fx.Int64), ticket_scratch)
            fx.barrier()
            ticket64 = Vec(ticket_view.load())[0]
            generation = ticket64 // fx.Int64(launch_grid)
            ticket = fx.Int32(ticket64 - generation * fx.Int64(launch_grid))
            gate_epoch = fx.Int32(generation + fx.Int64(1))
        owner = ticket == fx.Int32(0)
        producer = (ticket > fx.Int32(0)) & (ticket <= fx.Int32(prepare_blocks))
        producer_slot = ticket - fx.Int32(1)
        quant_producer = ticket > fx.Int32(prepare_blocks)
        quant_slot = ticket - fx.Int32(prepare_blocks + 1)

        if const_expr(quant_blocks > 0):  # noqa: SIM102 - preserve DSL staging
            if quant_producer:
                if const_expr(quant_dpp):
                    # FP8 DPP pipeline only; fp4 and the non-DPP fp8 buckets
                    # take the per-group loop below.
                    assert _QUANT_CHUNKS_IN_FLIGHT == 16 and block_threads % 16 == 0
                    total_chunks = i32_cur_tok * fx.Int32(model_dim // 8)
                    # Exact V# extent: the past-the-end prefetch of the last
                    # block is an OOB load (zero-fill, no memory traffic).
                    quant_in = ptr_buf_tensor(
                        addr_quant_in,
                        fx.Int32,
                        unit_elems=4,
                        num_records_bytes=total_chunks * fx.Int32(16),
                    )
                    quant_out8 = ptr_buf_tensor(
                        addr_quant_out,
                        fx.Int32,
                        unit_elems=2,
                        num_records_bytes=total_chunks * fx.Int32(8),
                    )
                    quant_scale32 = ptr_buf_tensor(
                        addr_quant_scale,
                        fx.Int32,
                        num_records_bytes=total_chunks >> fx.Int32(2),
                    )
                    blk_chunks = block_threads * _QUANT_CHUNKS_IN_FLIGHT
                    num_full = total_chunks // fx.Int32(blk_chunks)
                    lane16 = tid & fx.Int32(15)
                    scale_lane = lane16 == fx.Int32(12)
                    scale_off = lane16 * fx.Int32(block_threads // 16) + (
                        tid >> fx.Int32(4)
                    )
                    static_end = num_full
                    if const_expr(quant_steal):
                        static_rounds = num_full // fx.Int32(quant_blocks) - fx.Int32(
                            _QUANT_STEAL_ROUNDS
                        )
                        static_rounds = (static_rounds > fx.Int32(0)).select(
                            static_rounds, fx.Int32(0)
                        )
                        static_end = static_rounds * fx.Int32(quant_blocks)
                        steal_addr = disp_ptr(DispatchSlot.PREP_EPOCH_GATE) + fx.Int64(
                            prepare_blocks * 4
                        )
                        steal_count = num_full - static_end
                        last_grab = steal_count + fx.Int32(quant_blocks - 1)
                        rewind = fx.Int32(0) - (steal_count + fx.Int32(quant_blocks))
                        # Opaque (divergent-looking) zero offset: keeps LLVM's
                        # atomic optimizer off the in-loop grab, whose
                        # mbcnt/readfirstlane expansion forces an immediate
                        # vmcnt(0) drain of wave 0.  The result is consumed
                        # _QUANT_GRAB_AT chunks later instead.
                        opaque0 = fx.Int32(
                            llvm.inline_asm(
                                T.i32, [], "v_mov_b32 $0, 0", "=v",
                                has_side_effects=False,
                            )
                        )
                        steal_addr_v = steal_addr + fx.Int64(opaque0)
                    n_mine = (
                        static_end - quant_slot + fx.Int32(quant_blocks - 1)
                    ) // fx.Int32(quant_blocks)
                    n_mine = (quant_slot < static_end).select(n_mine, fx.Int32(0))
                    # Block sequence of this CTA: n_mine static blocks
                    # (quant_slot + j*Q), then stolen blocks; every CTA stops
                    # grabbing after its first failing grab, so a launch takes
                    # exactly steal_count + Q grabs and the last one rewinds
                    # the counter.
                    # Look-ahead-1 steal: the grab for block j+1 is issued
                    # at the start of block j and published by a barrier
                    # after chunk _QUANT_GRAB_AT; next-block loads for
                    # chunks < _QUANT_GRAB_AT are deferred to that point.
                    K = _QUANT_GRAB_AT
                    assert 0 < K < _QUANT_CHUNKS_IN_FLIGHT
                    cur = quant_slot
                    cur_ok = n_mine > fx.Int32(0)
                    if const_expr(quant_steal):
                        if (tid == fx.Int32(0)) & (n_mine == fx.Int32(0)):
                            g0 = fx.Int32(
                                comm_ops.atomic_add_agent(steal_addr, fx.Int32(1))
                            )
                            if g0 == last_grab:
                                comm_ops.atomic_add_agent(steal_addr, rewind)
                            fx.ptr_store(g0, bcast_scratch)
                        fx.barrier()
                        s0 = fx.Int32(fx.ptr_load(bcast_scratch))
                        use_s0 = n_mine == fx.Int32(0)
                        cur = use_s0.select(static_end + s0, cur)
                        cur_ok = cur_ok | (use_s0 & (s0 < steal_count))
                    lbase = cur_ok.select(cur * fx.Int32(blk_chunks), total_chunks) + tid
                    oob_pair = fx.Vector.from_elements(
                        [fx.Int32(0), fx.Int32(0)], fx.Int32
                    )
                    # Prologue mirrors the body's VMEM order (K stores, K
                    # loads, then store/load pairs, then the scale store)
                    # with dropped out-of-range stores: exact vmcnt merge.
                    for u in range_constexpr(K):
                        buf_copy_store(
                            quant_out8, total_chunks, oob_pair, fx.Int32, unit_elems=2
                        )
                        rocdl.sched_barrier(0)
                    raws = []
                    for u in range_constexpr(_QUANT_CHUNKS_IN_FLIGHT):
                        if const_expr(u >= K):
                            buf_copy_store(
                                quant_out8, total_chunks, oob_pair, fx.Int32, unit_elems=2
                            )
                        raws.append(
                            buf_copy_load(
                                quant_in,
                                lbase + fx.Int32(u * block_threads),
                                fx.Int32,
                                unit_elems=4,
                            )
                        )
                        rocdl.sched_barrier(0)
                    quant_scale32[total_chunks >> fx.Int32(4)] = fx.Int32(0)
                    rocdl.sched_barrier(0)
                    j = fx.Int32(0)
                    g = fx.Int32(0)
                    while cur_ok:
                        j1 = j + fx.Int32(1)
                        need_grab = j1 >= n_mine
                        cbase = cur * fx.Int32(blk_chunks) + tid
                        sel = fx.Int32(0)
                        if const_expr(quant_steal):  # noqa: SIM102 - preserve DSL staging
                            if (tid == fx.Int32(0)) & need_grab:
                                g = fx.Int32(
                                    comm_ops.atomic_add_agent(steal_addr_v, fx.Int32(1))
                                )
                        for u in range_constexpr(K):
                            e8 = _mx_fp8_quad_convert(
                                raws[u], cbase + fx.Int32(u * block_threads), quant_out8
                            )
                            sw = _row_scale_word(e8, tid)
                            sel = (lane16 == fx.Int32(u)).select(sw, sel)
                            rocdl.sched_barrier(0)
                        nxt = quant_slot + j1 * fx.Int32(quant_blocks)
                        nxt_ok = j1 < n_mine
                        if const_expr(quant_steal):
                            if (tid == fx.Int32(0)) & need_grab:
                                fx.ptr_store(g, bcast_scratch + fx.Int64(j1 & fx.Int32(1)))
                                if g == last_grab:
                                    comm_ops.atomic_add_agent(steal_addr, rewind)
                            if need_grab:
                                fx.barrier()
                            s1 = fx.Int32(
                                fx.ptr_load(bcast_scratch + fx.Int64(j1 & fx.Int32(1)))
                            )
                            nxt = nxt_ok.select(nxt, static_end + s1)
                            nxt_ok = nxt_ok | (s1 < steal_count)
                        nbase = nxt_ok.select(nxt * fx.Int32(blk_chunks), total_chunks) + tid
                        new_raws = []
                        for u in range_constexpr(K):
                            new_raws.append(
                                buf_copy_load(
                                    quant_in,
                                    nbase + fx.Int32(u * block_threads),
                                    fx.Int32,
                                    unit_elems=4,
                                )
                            )
                            rocdl.sched_barrier(0)
                        for u in range_constexpr(K, _QUANT_CHUNKS_IN_FLIGHT):
                            e8 = _mx_fp8_quad_convert(
                                raws[u], cbase + fx.Int32(u * block_threads), quant_out8
                            )
                            sw = _row_scale_word(e8, tid)
                            sel = (lane16 == fx.Int32(u)).select(sw, sel)
                            new_raws.append(
                                buf_copy_load(
                                    quant_in,
                                    nbase + fx.Int32(u * block_threads),
                                    fx.Int32,
                                    unit_elems=4,
                                )
                            )
                            rocdl.sched_barrier(0)
                        quant_scale32[cur * fx.Int32(blk_chunks // 16) + scale_off] = sel
                        raws = new_raws
                        cur = nxt
                        cur_ok = nxt_ok
                        j = j1
                    tail0 = (
                        num_full * fx.Int32(blk_chunks)
                        + quant_slot * fx.Int32(block_threads)
                        + tid
                    )
                    for chunk in range(
                        tail0, total_chunks, fx.Int32(quant_blocks * block_threads)
                    ):
                        raw = buf_copy_load(quant_in, chunk, fx.Int32, unit_elems=4)
                        sword = _emit_mx_fp8_quad_chunk(raw, chunk, tid, quant_out8)
                        if scale_lane:
                            quant_scale32[chunk >> fx.Int32(4)] = sword
                if const_expr(not quant_dpp):
                    quant_in = ptr_buf_tensor(addr_quant_in, fx.Int32, unit_elems=4)
                    quant_out = ptr_buf_tensor(addr_quant_out, fx.Int32, unit_elems=4)
                    quant_scale = ptr_buf_tensor(addr_quant_scale, fx.Uint8)
                    scale_dim = model_dim // 32
                    total_groups = i32_cur_tok * fx.Int32(scale_dim)
                    group_stride = fx.Int32(quant_blocks * block_threads)
                    group0 = quant_slot * fx.Int32(block_threads) + tid
                    for group_id in range(group0, total_groups, group_stride):
                        if const_expr(quant_mode == "fp4"):
                            emit_per_1x32_mx_fp4_group(
                                quant_in, quant_out, quant_scale, group_id
                            )
                        else:
                            emit_per_1x32_mx_fp8_group(
                                quant_in, quant_out, quant_scale, group_id
                            )

        # Quant CTAs are independent specialists.  They must retire as soon as
        # their quant groups drain instead of entering prepare's epoch/barrier
        # protocol.  The queued quant CTAs can then immediately occupy CUs
        # released by owner/group CTAs.
        if not quant_producer:
            if const_expr(fused_p1):
                # Single-group bucket (P == 1): the owner counts and fills
                # the routes itself.  This removes the epoch-gate, the two
                # GROUP_DONE and the PAIR_READY intra-rank handshakes (each a
                # release + poll + acquire round trip) from the decode chain.
                # The idle ticket-1 CTA retires immediately.  Every
                # cross-rank edge (COUNT_MATRIX push, COUNT_DONE, PLAN_READY)
                # and its release/acquire ordering is unchanged.
                if owner:
                    # tid 0's epoch-state loads are issued FIRST, then the
                    # count's row/pair prefetch, so tid 0's wait for its own
                    # loads does not also wait for the prefetch.  (The
                    # non-constant placeholders keep the compiler from folding
                    # the parity xor into the load branch, which would force
                    # a wait there.)
                    old_parity = i32_cur_tok
                    expected0 = i32_cur_tok
                    expected1 = i32_cur_tok
                    epoch_pred = tid < fx.Int32(_EPOCH_STORE_LANES)
                    if const_expr(pre_ticket):
                        old_parity = pre_old_parity
                        expected0 = pre_expected0
                        expected1 = pre_expected1
                    else:
                      if epoch_pred:
                        old_parity = comm_ops.load_i32_system(addr_parity, fx.Int32(0))
                        expected0 = comm_ops.load_i32_system(addr_expected, fx.Int32(0))
                        expected1 = comm_ops.load_i32_system(addr_expected, fx.Int32(1))
                    tok_pre = pre_tok
                    if const_expr(fused_tok and not pre_ticket):
                        tok_pre = _tok_prefetch(
                            num_waves=num_waves,
                            fz_npes=npes,
                            fz_k=topk,
                            fz_total_experts=total_experts,
                            addr_disp=addr_disp,
                            i32_cur_tok=i32_cur_tok,
                            addr_in_idx=addr_in_idx,
                            count_scratch=count_scratch,
                            prepare_blocks=prepare_blocks,
                        )
                    if epoch_pred:
                      # The 14 epoch-state resets are independent relaxed
                      # write-through stores: one lane each, ONE store
                      # instruction instead of 14 on lane 0.
                      parity_lane = old_parity ^ fx.Int32(1)
                      previous_expected = (parity_lane == fx.Int32(0)).select(
                          expected0, expected1
                      )
                      expected_lane = previous_expected + fx.Int32(npes)
                      st_addr = work_head + fx.Int64(
                          (tid - fx.Int32(2)) * fx.Int32(16 * 4)
                      )
                      st_val = fx.Int32(0)
                      for lane_id, lane_addr, lane_val in (
                          (0, addr_expected + fx.Int64(parity_lane) * fx.Int64(4),
                           expected_lane),
                          (1, payload_ready_rows, fx.Int32(tile_m)),
                          (10, work_tail, fx.Int32(0)),
                          (11, ready_tile_tail + fx.Int64(parity_lane) * fx.Int64(4),
                           fx.Int32(0)),
                          (12, addr_parity, parity_lane),
                          (13, epoch_gate, gate_epoch),
                      ):
                          is_lane = tid == fx.Int32(lane_id)
                          st_addr = is_lane.select(lane_addr, st_addr)
                          st_val = is_lane.select(lane_val, st_val)
                      _store_i32_sys_relaxed(st_addr, fx.Int32(0), st_val)
                      if tid == fx.Int32(0):
                          fx.ptr_store(parity_lane, bcast_scratch)
                          fx.ptr_store(expected_lane, bcast_scratch + fx.Int64(1))
                    if const_expr(not pre_ticket):  # noqa: SIM102 - preserve DSL staging
                        if (tid >> fx.Int32(6)) == fx.Int32(num_waves - 1):
                            _kcache_touch(addr_disp, DISPATCH_TABLE_SIZE * 2)
                    fx.barrier()
                    parity = fx.Int32(fx.ptr_load(bcast_scratch))
                    expected = fx.Int32(fx.ptr_load(bcast_scratch + fx.Int64(1)))
                    first_segment = _emit_fused_count_tok(
                        num_waves=num_waves,
                        fz_npes=npes,
                        fz_k=topk,
                        fz_epr=epr,
                        fz_total_experts=total_experts,
                        addr_disp=addr_disp,
                        i32_cur_tok=i32_cur_tok,
                        addr_in_idx=addr_in_idx,
                        parity=parity,
                        count_scratch=count_scratch,
                        prefetch_batches=fused_prefetch,
                        prepare_blocks=prepare_blocks,
                        prefetched=tok_pre,
                    )
                    peer_plan_ready = fx.Int64(0)
                    pair_order_ready_addr = fx.Int64(0)
                    # Flag addresses are host constants: load them before
                    # the plan (no fill separates the plan from the
                    # publish).
                    if tid < fx.Int32(npes):
                        peer_plan_ready = ptr_buf_tensor(
                            disp_ptr(DispatchSlot.P2P_PLAN_READY), fx.Int64
                        )[tid]
                    pair_order_ready_addr = disp_ptr(DispatchSlot.PAIR_ORDER_READY)

                    def _early_fill():
                        _emit_fused_fill_tok(
                            num_waves=num_waves,
                            fz_npes=npes,
                            fz_k=topk,
                            fz_total_experts=total_experts,
                            addr_disp=addr_disp,
                            i32_cur_tok=i32_cur_tok,
                            count_scratch=prefix_scratch,
                            segments=first_segment,
                            prepare_blocks=prepare_blocks,
                            wt_owner=wt_owner,
                        )

                    _emit_plan(
                        num_waves=num_waves,
                        fz_npes=npes,
                        fz_epr=epr,
                        fz_k=topk,
                        fz_mtpr=mtpr,
                        fz_rank=rank,
                        fz_tile_m=tile_m,
                        fz_total_experts=total_experts,
                        addr_disp=addr_disp,
                        i32_cur_tok=i32_cur_tok,
                        addr_in_idx=addr_in_idx,
                        parity=parity,
                        expected=expected,
                        dispatch_blocks=dispatch_blocks,
                        group_blocks=prepare_blocks,
                        group_done_slot=entry_slot,
                        group_phase_base=fx.Int32(0),
                        payload_chunk_rows=chunk_rows,
                        tile_state_stride=tile_state_stride,
                        fused_scratch=count_scratch,
                        defer_plan_ready=fused_defer,
                        early_fill=_early_fill,
                        prefix_scratch=prefix_scratch,
                        wt_owner=wt_owner,
                    )
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    fx.barrier()
                    # PLAN_READY (to every peer) and PAIR_ORDER_READY.
                    # Every published plan / pair-order store was written
                    # through (sc0 sc1, see _wst) and drained by the
                    # s_waitcnt vmcnt(0) + s_barrier above, so the flags
                    # need no release fence (no buffer_wbl2 of quant's
                    # dirty L2 lines).
                    if tid < fx.Int32(64):
                        if const_expr(not wt_owner):
                            comm_ops.fence_system_release()
                        if tid < fx.Int32(npes):
                            _store_i32_sys_relaxed(
                                peer_plan_ready,
                                parity * fx.Int32(npes) + fx.Int32(rank),
                                expected,
                            )
                        if tid == fx.Int32(0):
                            _store_i32_sys_relaxed(
                                pair_order_ready_addr,
                                parity,
                                expected,
                            )
            else:
                if owner:
                    next_parity_lane = fx.Int32(0)
                    if tid == fx.Int32(0):
                        old_parity = comm_ops.load_i32_system(addr_parity, fx.Int32(0))
                        expected0 = comm_ops.load_i32_system(addr_expected, fx.Int32(0))
                        expected1 = comm_ops.load_i32_system(addr_expected, fx.Int32(1))
                        next_parity_lane = old_parity ^ fx.Int32(1)
                        previous_expected = (next_parity_lane == fx.Int32(0)).select(
                            expected0, expected1
                        )
                        next_expected = previous_expected + fx.Int32(npes)
                        # Every epoch-state reset below stays in the system
                        # coherence domain (one-as) but is relaxed: ONE system
                        # release right before the epoch-gate publish orders all
                        # of them (the gate is what the group CTAs acquire; the
                        # later COUNT_DONE / PLAN_READY releases cover peers).
                        _store_i32_sys_relaxed(
                            addr_expected,
                            next_parity_lane,
                            next_expected,
                        )
                        _store_i32_sys_relaxed(
                            payload_ready_rows, fx.Int32(0), fx.Int32(tile_m)
                        )
                        work_head_rsrc = ptr_buf_tensor(work_head, fx.Int32)
                        for work_shard in range(8):
                            _wst(
                                work_head_rsrc,
                                fx.Int32(work_shard * 16),
                                fx.Int32(0),
                                wt_owner=wt_owner,
                            )
                        _store_i32_sys_relaxed(work_tail, fx.Int32(0), fx.Int32(0))
                        # Payload publishers reserve queue slots with a system-scope
                        # atomic.  Reset the tail in that same coherence domain so a
                        # layout transition cannot append after the previous epoch.
                        _store_i32_sys_relaxed(
                            ready_tile_tail, next_parity_lane, fx.Int32(0)
                        )
                        _store_i32_sys_relaxed(
                            addr_parity,
                            fx.Int32(0),
                            next_parity_lane,
                        )
                        fx.ptr_store(next_parity_lane, bcast_scratch)
                        fx.ptr_store(next_expected, bcast_scratch + fx.Int64(1))
                    hist_rsrc = ptr_buf_tensor(local_hist, fx.Int32)
                    for segment in range(tid, total_segments, block_threads):
                        _wst(hist_rsrc, segment, fx.Int32(0), wt_owner=wt_owner)
                    fx.rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0, expcnt=0)
                    fx.barrier()
                    if tid == fx.Int32(0):
                        # Every reset above is a write-through store drained by
                        # the s_waitcnt + barrier: no release fence needed.
                        if const_expr(not wt_owner):
                            comm_ops.fence_system_release()
                        _store_i32_sys_relaxed(
                            epoch_gate,
                            fx.Int32(0),
                            (gate_epoch & fx.Int32(_GATE_EPOCH_MASK))
                            | (next_parity_lane << fx.Int32(_GATE_PARITY_SHIFT)),
                        )
                else:
                    if tid == fx.Int32(0):
                        gate_value = _spin_epoch_gate(
                            epoch_gate, gate_epoch & fx.Int32(_GATE_EPOCH_MASK)
                        )
                        comm_ops.fence_agent_acquire()
                        fx.ptr_store(gate_value, bcast_scratch + fx.Int64(2))
                    fx.barrier()

                # The owner publishes parity from lane 0.  Converge the complete
                # CTA before any lane consumes it; otherwise the owner lanes can
                # classify and derive offsets with different epochs.
                fx.barrier()
                # The owner already holds the new parity/expected (it wrote
                # them); broadcast through LDS instead of a system acquire
                # plus two dependent system loads.  Group CTAs still acquire
                # and load them after observing the epoch gate.
                parity = fx.Int32(0)
                expected = fx.Int32(0)
                if owner:
                    parity = fx.Int32(fx.ptr_load(bcast_scratch))
                    expected = fx.Int32(fx.ptr_load(bcast_scratch + fx.Int64(1)))
                else:
                    # Parity arrives with the gate value; ``expected`` is
                    # only needed for the PAIR_READY wait, so _emit_group
                    # loads it after counting (off the critical path).
                    gate_value = fx.Int32(fx.ptr_load(bcast_scratch + fx.Int64(2)))
                    parity = (gate_value >> fx.Int32(_GATE_PARITY_SHIFT)) & fx.Int32(1)
                group_phase_base = fx.Int32(generation) * fx.Int32(prepare_blocks * 2)
                if owner:
                    _emit_plan(
                        num_waves=num_waves,
                        fz_npes=npes,
                        fz_epr=epr,
                        fz_k=topk,
                        fz_mtpr=mtpr,
                        fz_rank=rank,
                        fz_tile_m=tile_m,
                        fz_total_experts=total_experts,
                        addr_disp=addr_disp,
                        i32_cur_tok=i32_cur_tok,
                        addr_in_idx=addr_in_idx,
                        parity=parity,
                        expected=expected,
                        dispatch_blocks=dispatch_blocks,
                        group_blocks=prepare_blocks,
                        group_done_slot=entry_slot,
                        group_phase_base=group_phase_base,
                        payload_chunk_rows=chunk_rows,
                        tile_state_stride=tile_state_stride,
                        count_lds=count_scratch,
                        groups_self_prefix=_gsp_enabled(
                            prefix_scratch, total_segments, prepare_blocks, num_waves
                        ),
                        wt_owner=wt_owner,
                    )
                if producer:
                    _emit_group(
                        num_waves=num_waves,
                        fz_npes=npes,
                        fz_k=topk,
                        fz_epr=epr,
                        fz_total_experts=total_experts,
                        addr_disp=addr_disp,
                        i32_cur_tok=i32_cur_tok,
                        addr_in_idx=addr_in_idx,
                        dispatch_blocks=prepare_blocks,
                        group_done_slot=entry_slot,
                        producer_slot=producer_slot,
                        parity=parity,
                        expected=expected,
                        count_scratch=count_scratch,
                        group_phase_base=group_phase_base,
                        addr_expected=addr_expected,
                        large_bucket=tile_m >= 128,
                        fz_mtpr=mtpr,
                        prefix_lds=prefix_scratch,
                    )

    @flyc.jit
    def launch(
        addr_disp: fx.Int64,
        i32_cur_tok: fx.Int32,
        addr_in_idx: fx.Int64,
        addr_parity: fx.Int64,
        addr_expected: fx.Int64,
        addr_quant_in: fx.Int64,
        addr_quant_out: fx.Int64,
        addr_quant_scale: fx.Int64,
        stream: fx.Stream,
    ):
        kernel(
            addr_disp,
            i32_cur_tok,
            addr_in_idx,
            addr_parity,
            addr_expected,
            addr_quant_in,
            addr_quant_out,
            addr_quant_scale,
        ).launch(
            grid=(launch_grid, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


def run_mega_moe_prepare(
    addr_disp,
    i32_cur_tok,
    addr_in_idx,
    addr_parity,
    addr_expected,
    addr_quant_in,
    addr_quant_out,
    addr_quant_scale,
    stream,
    **kwargs,
):
    if _m64_qdpp_for(i32_cur_tok, kwargs):
        kwargs = {**kwargs, "m64_qdpp": True}
    launch = compile_mega_moe_prepare(**kwargs)
    _run_compiled(
        launch,
        addr_disp,
        i32_cur_tok,
        addr_in_idx,
        addr_parity,
        addr_expected,
        addr_quant_in,
        addr_quant_out,
        addr_quant_scale,
        stream,
    )


def preload_mega_moe_prepare(
    addr_disp,
    i32_cur_tok,
    addr_in_idx,
    addr_parity,
    addr_expected,
    addr_quant_in,
    addr_quant_out,
    addr_quant_scale,
    stream,
    **kwargs,
):
    """Compile and load one prepare variant without dispatching it."""
    if _m64_qdpp_variant(kwargs):
        # The m64 P32 bucket dispatches one of two variants by live token
        # count; load the qdpp one too so neither JITs on the serving path.
        _preload_compiled(
            compile_mega_moe_prepare(**kwargs, m64_qdpp=True),
            addr_disp,
            i32_cur_tok,
            addr_in_idx,
            addr_parity,
            addr_expected,
            addr_quant_in,
            addr_quant_out,
            addr_quant_scale,
            stream,
        )
    launch = compile_mega_moe_prepare(**kwargs)
    return _preload_compiled(
        launch,
        addr_disp,
        i32_cur_tok,
        addr_in_idx,
        addr_parity,
        addr_expected,
        addr_quant_in,
        addr_quant_out,
        addr_quant_scale,
        stream,
    )

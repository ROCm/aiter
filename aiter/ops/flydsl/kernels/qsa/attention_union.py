# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 D256 multi-query/multi-head QSA sparse attention with exact membership.

The LDS, QK/PV MFMA and staggered pipeline come from a dense BM128/BN64 D256
attention kernel (see attention_common). Q/O rows and sparse masks differ.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir import ir
from flydsl.expr import rocdl

from .attention_common import (
    _buffer,
    _buffer_words,
    _center,
    _copy_coordinates,
    _cross,
    _exps,
    _join,
    _k_lds_address,
    _k_operands,
    _k_phase,
    _load_lds,
    _max,
    _maximum,
    _min,
    _pack_bf16,
    _pin,
    _pin_i32,
    _pv,
    _q_fragment,
    _qk,
    _read_k,
    _read_v,
    _rescale,
    _schedule,
    _stage_end,
    _sum,
    _uniform,
    _v_lds_address,
    _v_operands,
    _write_address,
)

__all__ = ["run"]

BM, BN, D, THREADS, LDS_BYTES = 128, 64, 256, 512, 65536


def _source_blocks(blocks, tile, count, lane_bytes):
    # SALU-only address: the descriptor starts at the tile; slots past count read 0 (masked).
    view = fx.make_view(blocks + tile * 16, fx.make_layout(16, 1))
    return _buffer_words(_buffer(view, (count - tile * 16) * 4), lane_bytes, 1)[0]


def _dma_rows(blocks, wave, hk):
    """SOFFSET rows of the wave's four slots: lanes 0..3 hold their block IDs."""
    return fx.Vector.from_elements(
        [
            fx.Int32(
                fx.Uint32(rocdl.readlane(fx.Int32.ir_type, blocks, slot))
                * (4 * hk * D * 2)
                + (wave >> 2) * 256
            )
            for slot in range(4)
        ],
        fx.Int32,
    )


def _union_dma(
    resource,
    storage,
    wave,
    lane,
    rows,
    tile,
    compact_len,
    limits,
    hk,
    packet,
    is_v,
    extent,
    tail=False,
    read_storage=None,
    read_address=None,
    read_immediate=0,
):
    # Token k of a block reads SOFFSET min(row, limits[k]) + k rows: a partial final block re-reads
    # the request's last token, whose K is masked and V weighted 0; with fewer than k + 1 tokens
    # the limit is 0 and the range-checked VOFFSET + offset returns 0, so reads stay in the request.
    token, _ = _copy_coordinates(wave, packet, 4)
    voffset = lane * 4 if is_v else (lane ^ (_k_phase(token) * 4)) * 4
    step = (packet & 3) * hk * D * 2
    # The instruction offset also moves the LDS destination, which M0 takes back; M0 must stay
    # non-negative, otherwise (HK > 1) the step goes into VOFFSET.
    destination_offset = packet * 512
    offset = step if step < 4096 and step <= destination_offset else 0
    if step != offset:
        voffset = voffset + step
    if tail:
        voffset = (tile * BN + token < compact_len).select(voffset, fx.Int32(extent))
    base_row, base_channel = _copy_coordinates(wave, 0, 4)
    base_destination = fx.Int32(base_row * 512 + base_channel * 2)
    immediate = destination_offset - offset
    row, limit = fx.Uint32(rows[packet // 4]), limits[packet & 3]
    data = (
        None
        if read_address is None
        else _load_lds(read_storage, read_address, read_immediate)
    )
    # Raw LDS DMA: BufferCopyLDS adds its own vmcnt wait; the pipeline waits itself.
    rocdl.raw_ptr_buffer_load_lds(
        resource,
        fx.to_llvm_ptr(fx.get_iter(storage) + (base_destination + immediate)),
        4,
        fx.Int32(voffset),
        _min(row, limit),
        offset,
    )
    return data


def _prefetch_mask(resource, tile, offset, bq):
    # S0 issues this request; the existing S2 vmcnt(0) retires it before S3.
    bits = rocdl.raw_ptr_buffer_load(
        fx.Int16.ir_type, resource, fx.Int32(offset), fx.Int32(tile * bq * 8), 0
    )
    # Only bits 0..15 are read, so the sign extension is harmless.
    return fx.Int16(bits).to(fx.Int32)


def _apply_mask(scores, bits):
    # Signed extraction produces 0/-1; bit-select preserves score bits or -inf.
    # LLVM can use bfe/bfi without a compare and VCC-dependent cndmask per value.
    values = fx.Vector(scores).bitcast(fx.Int32)
    result = []
    for i in range(16):
        keep = (bits << (31 - i)) >> 31
        result.append((values[i] & keep) | ((~keep) & fx.Int32(-8388608)))
    return fx.Vector.from_elements(result, fx.Int32).bitcast(fx.Float32)


def _output(
    o0, o1, inv, buffer, storages, tid, heads, g, gp, bq, qvalid, extent, row_offset
):
    # O rows 0-63 go to the K buffer, rows 64-127 to the V buffer.
    row = (tid >> 6) * 16 + (tid & 15)
    shared = [fx.Int32(fx.ptrtoint(fx.get_iter(storage))) for storage in storages]
    base = (tid < THREADS // 2).select(shared[0], shared[1])
    for half in range(2):
        output = o1 if half else o0
        for i in range(4):
            values = fx.Vector.from_elements(
                [output[n * 4 + i] * inv for n in range(8)], fx.Float32
            )
            words = _pack_bf16(values)
            column = (((tid >> 4) & 3) * 4 + i) * 8 + half * 128
            address = fx.Int32(
                base + (((row & 63) * 256 + column) ^ ((row & 7) * 8)) * 2
            )
            _write_address(address, words)
    rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    parts = []
    for part in range(8):
        element = tid * 8 + (part % 4) * THREADS * 8
        read_row = element // 256
        parts.append(
            _load_lds(storages[part // 4], (element ^ ((read_row & 7) * 8)) * 2)
        )
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.sched_barrier(0)
    for part in range(8):
        element = tid * 8 + part * THREADS * 8
        read_row, column = element // 256, element % 256
        query, head = (read_row + row_offset) // gp, (read_row + row_offset) % gp
        offset = fx.Uint32((query * heads * D + head * D + column) * 2)
        valid = (query < qvalid) & (query < bq) & (head < g)
        offset = valid.select(offset, extent)
        fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        fragment.store(parts[part].bitcast(fx.BFloat16))
        atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16)
        fx.copy(
            atom,
            fragment,
            fx.make_view(fx.get_iter(buffer) + offset // 2, fx.make_layout(8, 1)),
        )
    _stage_end()


@flyc.jit
def _body(
    Q,
    K,
    V,
    OUT,
    META,
    BLOCKS,
    MEMBERS,
    COUNTS,
    k_storage,
    v_storage,
    work,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    SCALE: fx.Constexpr[float],
    STAGGER: fx.Constexpr[bool],
):
    read_k, read_v, pv = _read_k, _read_v, _pv
    dma_rows, v_operands = _dma_rows, _v_operands
    source_blocks, output_fn = _source_blocks, _output
    qk, local_sum, local_max, cross = _qk, _sum, _max, _cross
    exps, pack, center = _exps, _pack_bf16, _center
    rescale = _rescale
    slices = (BQ * GP + 127) // 128
    tile_id, hkv = work // (HK * slices), work % HK
    row_offset = ((work // HK) % slices) * 128
    q0, qvalid = _uniform(META[tile_id * 5]), _uniform(META[tile_id * 5 + 1])
    k0, kv_len = _uniform(META[tile_id * 5 + 2]), _uniform(META[tile_id * 5 + 3])
    count = _uniform(COUNTS[tile_id])
    blocks = fx.get_iter(BLOCKS) + tile_id * CAP
    masks = fx.make_view(
        fx.get_iter(MEMBERS) + tile_id * (CAP // 16) * BQ * 4,
        fx.make_layout((CAP // 16) * BQ * 4, 1),
    )
    mask_resource = rocdl.get_buffer_rsrc(_buffer(masks, (CAP // 16) * BQ * 8))
    tid = fx.thread_idx.x
    wave, lane = _uniform(tid >> 6), tid & 63
    lane_bytes = _pin_i32(((wave & 3) * 4 + (lane & 3)) * 4)
    row = row_offset + wave * 16 + (lane & 15)
    query, head = row // GP, row % GP
    mask_offset = _pin_i32(_min(query, fx.Int32(BQ - 1)) * 8 + (lane >> 4) * 2)
    valid_row = (query < qvalid) & (query < BQ) & (head < (H // HK))
    # Byte offsets are unsigned 32-bit: Q and each request's K/V may span up to 4 GiB.
    qextent = fx.Uint32(NQ - q0) * H * D * 2 - hkv * (H // HK) * D * 2
    qptr = fx.get_iter(Q) + fx.Int64(q0) * (H * D) + hkv * (H // HK) * D
    gq = _buffer(fx.make_view(qptr, fx.make_layout(NQ * H * D, 1)), qextent)
    qoffset = valid_row.select(fx.Uint32((query * H * D + head * D) * 2), qextent)
    q = _q_fragment(gq, qoffset, lane)
    extent = (fx.Uint32(kv_len) * HK - hkv) * D * 2
    limit = fx.Uint32(kv_len - 1) * (HK * D * 2) + (wave >> 2) * 256
    limits = tuple(
        (limit > k * HK * D * 2).select(limit - k * HK * D * 2, fx.Uint32(0))
        for k in range(4)
    )
    gk = rocdl.get_buffer_rsrc(
        _buffer(
            fx.make_view(
                fx.get_iter(K) + (fx.Int64(k0) * HK + hkv) * D,
                fx.make_layout(extent // 2, 1),
            ),
            extent,
        )
    )
    gv = rocdl.get_buffer_rsrc(
        _buffer(
            fx.make_view(
                fx.get_iter(V) + (fx.Int64(k0) * HK + hkv) * D,
                fx.make_layout(extent // 2, 1),
            ),
            extent,
        )
    )
    scale = fx.Float32(SCALE * math.log2(math.e))
    compact_len = count * 4
    tiles = (compact_len + BN - 1) // BN
    last = tiles - 1
    key_row = (lane & 3) + ((lane & 12) << 1)
    kr = tuple(
        _pin_i32(_k_lds_address(key_row + n * 4, (lane >> 4) * 8 + parity * 32))
        for n in range(2)
        for parity in range(2)
    )
    vr = _pin_i32(_v_lds_address((lane >> 4) * 8, (lane & 15) * 8))
    cross_addresses = tuple(_pin_i32((lane ^ offset) * 4) for offset in (16, 32, 48))
    blocks0 = source_blocks(blocks, fx.Int32(0), count, lane_bytes)
    blocks1 = source_blocks(blocks, _min(fx.Int32(1), last), count, lane_bytes)
    blocks2 = source_blocks(blocks, _min(fx.Int32(2), last), count, lane_bytes)
    bits = _prefetch_mask(mask_resource, fx.Int32(0), mask_offset, BQ)
    offsets = dma_rows(blocks0, wave, HK)
    v_offsets = offsets
    for packet in fx.range_constexpr(16):
        _union_dma(
            gk,
            k_storage,
            wave,
            lane,
            offsets,
            fx.Int32(0),
            compact_len,
            limits,
            HK,
            packet,
            False,
            extent,
        )
    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
    _stage_end()
    if fx.const_expr(STAGGER):
        _stage_end()
    k = read_k(k_storage, kr, 0)
    rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    lo = qk(q, _k_operands(k))
    o0, o1 = _pin(fx.Vector.filled(32, 0.0, fx.Float32)), _pin(
        fx.Vector.filled(32, 0.0, fx.Float32)
    )
    _schedule(32, 3, 5)
    _stage_end()
    k = read_k(k_storage, kr, 1)
    rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    hi = qk(q, _k_operands(k))
    scores = _apply_mask(_join(lo, hi), bits)
    maximum = local_max(scores)
    maximum = _maximum(maximum, maximum.shuffle_xor(16, 64))
    maximum = _maximum(maximum, maximum.shuffle_xor(32, 64))
    maximum = _maximum(maximum * scale, fx.Float32(-1.0e30)) + 1.0
    scores = center(scores, scale, maximum)
    row_sum = fx.Float32(0.0)
    _stage_end()
    offsets = dma_rows(blocks1, wave, HK)
    current_offsets = offsets
    next_offsets = dma_rows(blocks2, wave, HK)
    for packet in fx.range_constexpr(16):
        _union_dma(
            gk,
            k_storage,
            wave,
            lane,
            offsets,
            _min(fx.Int32(1), last),
            compact_len,
            limits,
            HK,
            packet,
            False,
            extent,
        )
    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
    _stage_end()
    _stage_end()

    @flyc.jit
    def phase(
        previous,
        maximum,
        row_sum,
        o0,
        o1,
        previous_offsets,
        current_offsets,
        next_offsets,
        t,
    ):
        previous, maximum, row_sum = (
            fx.Vector(previous),
            fx.Float32(maximum),
            fx.Float32(row_sum),
        )
        o0, o1, t = fx.Vector(o0), fx.Vector(o1), fx.Int32(t)
        previous_offsets = fx.Vector(previous_offsets)
        current_offsets, next_offsets = fx.Vector(current_offsets), fx.Vector(
            next_offsets
        )
        bits = _prefetch_mask(mask_resource, t, mask_offset, BQ)
        future_blocks = source_blocks(blocks, _min(t + 2, last), count, lane_bytes)
        parts = []
        for n in fx.range_constexpr(2):
            for step in fx.range_constexpr(8):
                parts.append(
                    _union_dma(
                        gv,
                        v_storage,
                        wave,
                        lane,
                        previous_offsets,
                        t - 1,
                        compact_len,
                        limits,
                        HK,
                        n * 8 + step,
                        True,
                        extent,
                        read_storage=k_storage,
                        read_address=kr[n * 2 + step % 2],
                        read_immediate=(step // 2) * 128,
                    )
                )
        k = fx.Vector.from_elements(
            [part[i] for part in parts for i in range(4)], fx.Int32
        )
        _stage_end()
        rocdl.s_waitcnt(lgkmcnt=0)
        lo = qk(q, _k_operands(k))
        previous = exps(previous)
        _schedule(32, 1, 1, True)
        _stage_end()
        k = read_k(k_storage, kr, 1, STAGGER)
        rocdl.s_waitcnt(vmcnt=0)
        if fx.const_expr(STAGGER):
            rocdl.s_waitcnt(lgkmcnt=8)
        _stage_end()
        rocdl.s_waitcnt(lgkmcnt=0)
        hi = qk(q, _k_operands(k))
        current = _apply_mask(_join(lo, hi), bits)
        total, probabilities = local_sum(previous), pack(previous).bitcast(fx.BFloat16)
        # The stage C vmcnt(0) retired this load; only SGPR rows cross iterations.
        future_offsets = dma_rows(future_blocks, wave, HK)
        _schedule(32, 2, 2)
        _stage_end()
        sums = cross(total, cross_addresses)
        rocdl.sched_barrier(0)  # ds_bpermute counts in the hand-written lgkmcnt below
        parts = []
        for block in fx.range_constexpr(2):
            for r in fx.range_constexpr(8):
                parts.append(
                    _union_dma(
                        gk,
                        k_storage,
                        wave,
                        lane,
                        next_offsets,
                        _min(t + 1, last),
                        compact_len,
                        limits,
                        HK,
                        block * 8 + r,
                        False,
                        extent,
                        read_storage=v_storage,
                        read_address=vr,
                        read_immediate=block * 16384 + r * 512,
                    )
                )
        v = fx.Vector.from_elements(
            [part[i] for part in parts for i in range(4)], fx.Int32
        )
        prepared = v_operands(v, 0, True)
        _stage_end()
        o0, row_sum, candidate = pv(
            probabilities,
            v,
            o0,
            True,
            prepared,
            summary_args=(current, total, sums, row_sum),
        )
        _schedule(32, 3, 3)
        _stage_end()
        maxima = cross(candidate, cross_addresses)
        rocdl.sched_barrier(0)  # ds_bpermute counts in the hand-written lgkmcnt below
        rocdl.s_waitcnt(vmcnt=0)
        v = read_v(v_storage, vr, 1)
        prepared = v_operands(v, 0, True)
        _stage_end()
        o1, current, new_max, ballot = pv(
            probabilities,
            v,
            o1,
            True,
            prepared,
            (current, candidate, maxima, scale, maximum),
        )
        _schedule(32, 3, 4)
        o0, o1, row_sum = rescale(o0, o1, row_sum, maximum, new_max, ballot)
        _stage_end()
        return (
            current,
            new_max,
            row_sum,
            o0,
            o1,
            current_offsets,
            next_offsets,
            future_offsets,
        )

    state = (scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, next_offsets)
    # Two phases per trip swap the max registers; one phase copies it after its barrier.
    for t in range(fx.Int32(1), tiles - 1, fx.Int32(2)):
        state = phase(*state, t)
        state = phase(*state, fx.Int32(t) + 1)
    if (tiles > 1) & ((tiles & 1) == 0):
        state = phase(*state, tiles - 1)
    scores, maximum, row_sum, o0, o1, v_offsets, current_offsets, next_offsets = state
    offsets = fx.Vector(v_offsets)
    for packet in fx.range_constexpr(16):
        _union_dma(
            gv,
            v_storage,
            wave,
            lane,
            offsets,
            last,
            compact_len,
            limits,
            HK,
            packet,
            True,
            extent,
            True,
        )
    scores = exps(fx.Vector(scores))
    total = local_sum(scores)
    total = total + total.shuffle_xor(16, 64)
    total = total + total.shuffle_xor(32, 64)
    row_sum = fx.Float32(row_sum) + total
    probabilities = pack(scores).bitcast(fx.BFloat16)
    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
    _stage_end()
    _stage_end()
    v = read_v(v_storage, vr, 0)
    rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    o0 = pv(probabilities, v, o0)
    _stage_end()
    v = read_v(v_storage, vr, 1)
    rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    o1 = pv(probabilities, v, o1)
    _stage_end()
    if fx.const_expr(not STAGGER):
        _stage_end()
    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0))
    output = rocdl.make_buffer_tensor(
        fx.make_view(
            fx.get_iter(OUT) + fx.Int64(q0) * (H * D) + hkv * (H // HK) * D,
            fx.make_layout(NQ * H * D, 1),
        ),
        num_records_bytes=qextent,
    )
    output_fn(
        o0,
        o1,
        inv,
        output,
        (k_storage, v_storage),
        _pin_i32(tid),
        H,
        H // HK,
        GP,
        BQ,
        qvalid,
        qextent,
        row_offset,
    )


@flyc.kernel(name="attention_union_bf16_d256", known_block_size=[THREADS, 1, 1])
def _kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SCALE: fx.Constexpr[float],
):
    body = _body
    allocator = fx.SharedAllocator()
    k_storage, v_storage = (
        allocator.allocate(fx.Array[fx.Int8, LDS_BYTES // 2, 16])
        .peek()
        .view(fx.make_layout(LDS_BYTES // 2, 1))
        for _ in range(2)
    )
    work = fx.block_idx.x
    while work < ((TASKS + GRID - 1) // GRID) * GRID:
        mapped = _uniform(ORDER[work])
        tile = mapped // (HK * ((BQ * GP + 127) // 128))
        enabled = fx.Int32(0)
        if mapped >= 0:
            enabled = _uniform(ACTIVE[tile])
        if enabled != 0:
            group = _uniform(fx.thread_idx.x >> 8)
            if group != 0:
                body(
                    Q,
                    K,
                    V,
                    OUT,
                    META,
                    BLOCKS,
                    MEMBERS,
                    COUNTS,
                    k_storage,
                    v_storage,
                    mapped,
                    H,
                    HK,
                    NQ,
                    CAP,
                    BQ,
                    GP,
                    SCALE,
                    True,
                )
            else:
                body(
                    Q,
                    K,
                    V,
                    OUT,
                    META,
                    BLOCKS,
                    MEMBERS,
                    COUNTS,
                    k_storage,
                    v_storage,
                    mapped,
                    H,
                    HK,
                    NQ,
                    CAP,
                    BQ,
                    GP,
                    SCALE,
                    False,
                )
        work = work + GRID


@flyc.jit
def _launch(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SCALE: fx.Constexpr[float],
    stream: fx.Stream,
):
    if GRID > 0:
        _kernel(
            Q,
            K,
            V,
            OUT,
            META,
            BLOCKS,
            MEMBERS,
            COUNTS,
            ACTIVE,
            ORDER,
            H,
            HK,
            NQ,
            CAP,
            BQ,
            GP,
            TASKS,
            GRID,
            SCALE,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "llvm.target_features": ir.Attribute.parse(
                    '#llvm.target_features<["-packed-fp32-ops"]>'
                ),
            },
        ).launch(grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED = {}


def _args(inputs, plan, out, stream):
    return (
        inputs.q.view(-1),
        inputs.k,
        inputs.v,
        out.view(-1),
        plan.metadata.view(-1),
        plan.blocks.view(-1),
        plan.score_masks.view(-1),
        plan.counts.view(-1),
        plan.active,
        plan.task_order,
        inputs.q.shape[1],
        inputs.k.shape[1],
        inputs.q.shape[0],
        plan.block_capacity,
        plan.query_tile,
        plan.group_padded,
        plan.num_tiles
        * inputs.k.shape[1]
        * ((plan.query_tile * plan.group_padded + 127) // 128),
        plan.grid,
        inputs.scale,
        stream,
    )


def _compiled(inputs, args, stream):
    key = (inputs.q.device, args[10], args[11], args[14], args[15], args[18])
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        with torch.cuda.device(inputs.q.device), torch.cuda.stream(stream):
            compiled = _COMPILED[key] = flyc.compile(
                _launch, *args[:17], 0, args[18], stream
            )
    return compiled


def run(*, inputs, plan, out):
    if plan.num_tiles == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, out, stream)
    with torch.cuda.device(inputs.q.device):
        _compiled(inputs, args, stream)(*args)


def launcher(*, inputs, plan, out):
    """launch(q, k, v, out) with flat Q/out and unflattened K/V; other arguments prebuilt.

    Valid while the plan's scratch bindings and the current stream stay the same.
    """
    if plan.num_tiles == 0:
        return None
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, out, stream)
    compiled = _compiled(inputs, args, stream)
    tail = args[4:]
    return lambda q, k, v, o: compiled(q, k, v, o, *tail)

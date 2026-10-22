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
    _k_reads,
    _lds_words,
    _LdsQueue,
    _load_lds,
    _max,
    _MAX_FLOOR,
    _maximum,
    _min,
    _pack_bf16,
    _pin,
    _pin_i32,
    _pv,
    _qk,
    _read_q_g2r,
    _read_s2r,
    _rescale,
    _schedule,
    _stage_end,
    _sum,
    _uniform,
    _v_lds_address,
    _v_operands,
    _v_reads,
    _write_address,
)

__all__ = ["run"]

BM, BN, D, THREADS, LDS_BYTES = 128, 64, 256, 512, 65536


def _read_ids_g2r(blocks, tile, count, offset):
    """Block ids of a key tile (16 blocks of 4 keys); each lane reads the one at byte offset."""
    # SALU-only address: the descriptor starts at the tile; slots past count read 0 (masked).
    view = fx.make_view(blocks + tile * 16, fx.make_layout(16, 1))
    return _buffer_words(_buffer(view, (count - tile * 16) * 4), offset, 1)[0]


def _kv_rows(ids, wave, hk):
    """SOFFSET rows of the wave's four blocks for the K/V DMA: lanes 0..3 hold their ids."""
    return fx.Vector.from_elements(
        [
            fx.Int32(
                fx.Uint32(rocdl.readlane(fx.Int32.ir_type, ids, slot))
                * (4 * hk * D * 2)
                + (wave >> 2) * 256
            )
            for slot in range(4)
        ],
        fx.Int32,
    )


def _read_packet_g2s(
    resource,
    storage,
    wave,
    lane,
    rows,
    compact_len,
    limits,
    hk,
    packet,
    is_v,
    extent,
    tail_tile=None,
    read=None,
):
    """LDS DMA of key (wave & 3) * 16 + packet, dims (wave >> 2) * 128 + 0..127.

    The optional LDS read (storage, address, immediate) is issued just before it.
    """
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
    if tail_tile is not None:
        voffset = (tail_tile * BN + token < compact_len).select(
            voffset, fx.Int32(extent)
        )
    base_row, base_channel = _copy_coordinates(wave, 0, 4)
    base_destination = fx.Int32(base_row * 512 + base_channel * 2)
    immediate = destination_offset - offset
    row, limit = fx.Uint32(rows[packet // 4]), limits[packet & 3]
    data = None if read is None else _load_lds(*read)
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


def _read_mask_g2r(resource, tile, offset, bq):
    """Membership bits of this lane's query over the 16 keys it scores in a key tile."""
    # The next vmcnt(0) retires it before the compute stage that applies it.
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


def _shuffle_o_r2s2r(o_lo, o_hi, inv, storages, tid):
    """O * inv as BF16 through LDS; returns the thread's eight 16-byte pieces of O rows."""
    # O rows 0-63 go to the K buffer, rows 64-127 to the V buffer.
    row = (tid >> 6) * 16 + (tid & 15)
    shared = [fx.Int32(fx.ptrtoint(fx.get_iter(storage))) for storage in storages]
    base = (tid < THREADS // 2).select(shared[0], shared[1])
    for half in range(2):
        output = o_hi if half else o_lo
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
    return parts


def _store_o_r2g(parts, buffer, tid, heads, g, gp, bq, qvalid, extent, row_offset):
    # Rows past the task's queries or heads go to extent, out of the buffer range.
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
    k_lds,
    v_lds,
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
    """One task: 128 (query, head) rows of a union tile over its compact key list.

    The keys form key tiles t = 0 .. key_tiles - 1 of BN = 64 keys; k_lds and v_lds
    hold one tile each. Phase t runs QK and the row max of tile t and, one tile behind,
    the exponentials, row sum and PV of tile t - 1. A phase has four steps, each a memory
    stage then a compute stage; waves 4-7 (STAGGER) run one stage behind waves 0-3, so
    one group's memory stage overlaps the other group's compute stage.

    Softmax in log2 units: S' = S * scale * log2(e) - m, P = exp2(S'), l = sum(P) and
    O = sum(P V); the output is O / l. m starts at the first row max + 1 and moves to a
    later row max + 1 only when that exceeds m + 7, rescaling O and l.
    """
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
    ids_offset = _pin_i32(((wave & 3) * 4 + (lane & 3)) * 4)
    row = row_offset + wave * 16 + (lane & 15)
    query, head = row // GP, row % GP
    mask_offset = _pin_i32(_min(query, fx.Int32(BQ - 1)) * 8 + (lane >> 4) * 2)
    valid_row = (query < qvalid) & (query < BQ) & (head < (H // HK))
    # Byte offsets are unsigned 32-bit: Q and each request's K/V may span up to 4 GiB.
    qextent = fx.Uint32(NQ - q0) * H * D * 2 - hkv * (H // HK) * D * 2
    qptr = fx.get_iter(Q) + fx.Int64(q0) * (H * D) + hkv * (H // HK) * D
    gq = _buffer(fx.make_view(qptr, fx.make_layout(NQ * H * D, 1)), qextent)
    qoffset = valid_row.select(fx.Uint32((query * H * D + head * D) * 2), qextent)
    # Q stays in registers for all key tiles.
    q = _read_q_g2r(gq, qoffset, lane)
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
    key_tiles = (compact_len + BN - 1) // BN
    last_tile = key_tiles - 1
    key_row = (lane & 3) + ((lane & 12) << 1)
    k_s2r_base = tuple(
        _pin_i32(_k_lds_address(key_row + n * 4, (lane >> 4) * 8 + parity * 32))
        for n in range(2)
        for parity in range(2)
    )
    v_s2r_base = _pin_i32(_v_lds_address((lane >> 4) * 8, (lane & 15) * 8))
    cross_addresses = tuple(_pin_i32((lane ^ offset) * 4) for offset in (16, 32, 48))

    # g = global, s = LDS, r = registers; reads only issue, the stages place the waits.
    def read_ids_g2r(tile):
        return _read_ids_g2r(blocks, tile, count, ids_offset)

    def read_mask_g2r(tile):
        return _read_mask_g2r(mask_resource, tile, mask_offset, BQ)

    def kv_rows(ids):
        return _kv_rows(ids, wave, HK)

    def k_s2r(half, lower_first=False):
        """Read plan of K keys 32 * half .. 32 * half + 31."""
        return k_lds, _k_reads(k_s2r_base, half, lower_first)

    def v_s2r(half):
        """Read plan of V dims 128 * half .. 128 * half + 127."""
        return v_lds, _v_reads(v_s2r_base, half)

    def read_tile_g2s(resource, storage, rows, is_v, s2r, lgkm, tail_tile):
        # 16 LDS-DMA packets; read i of the s2r plan is issued just before packet i.
        parts = [None] * 16
        for packet in fx.range_constexpr(16):
            read, slot = None, packet
            if fx.const_expr(s2r is not None):
                slot, address, immediate = s2r[1][packet]
                read = (s2r[0], address, immediate)
            parts[slot] = _read_packet_g2s(
                resource,
                storage,
                wave,
                lane,
                rows,
                compact_len,
                limits,
                HK,
                packet,
                is_v,
                extent,
                tail_tile,
                read,
            )
        if fx.const_expr(s2r is None):
            return None
        return _lds_words(parts), lgkm.issue(16)

    def read_k_g2s(rows, s2r=None, lgkm=None):
        return read_tile_g2s(gk, k_lds, rows, False, s2r, lgkm, None)

    def read_v_g2s(rows, s2r=None, lgkm=None, tail_tile=None):
        return read_tile_g2s(gv, v_lds, rows, True, s2r, lgkm, tail_tile)

    def qk(k):
        return _qk(q, _k_operands(k))

    def prologue():
        """Ids of tiles 0-2 and mask(0) g2r; K(0) g2s, QK(0), m and S'(0); K(1) g2s."""
        lgkm = _LdsQueue()
        ids_0 = read_ids_g2r(fx.Int32(0))
        ids_1 = read_ids_g2r(_min(fx.Int32(1), last_tile))
        ids_2 = read_ids_g2r(_min(fx.Int32(2), last_tile))
        mask_0 = read_mask_g2r(fx.Int32(0))
        rows_0 = kv_rows(ids_0)
        # Memory: K(0) g2s.
        read_k_g2s(rows_0)
        lgkm.wait(vmcnt=0)
        _stage_end()
        if fx.const_expr(STAGGER):
            # Waves 4-7 fall one stage behind.
            _stage_end()
        # Memory: K(0) keys 0-31 s2r.
        k_lo, _ = _read_s2r(k_s2r(0), lgkm)
        lgkm.wait()
        _stage_end()
        # Compute: QK S(0) keys 0-31; O = 0.
        s_lo = qk(k_lo)
        o_lo = _pin(fx.Vector.filled(32, 0.0, fx.Float32))
        o_hi = _pin(fx.Vector.filled(32, 0.0, fx.Float32))
        _schedule(32, 3, 5)
        _stage_end()
        # Memory: K(0) keys 32-63 s2r.
        k_hi, _ = _read_s2r(k_s2r(1), lgkm)
        lgkm.wait()
        _stage_end()
        # Compute: QK S(0) keys 32-63 and mask(0); softmax m and S'(0).
        s_hi = qk(k_hi)
        s_0 = _apply_mask(_join(s_lo, s_hi), mask_0)
        row_max = _max(s_0)
        row_max = _maximum(row_max, row_max.shuffle_xor(16, 64))
        row_max = _maximum(row_max, row_max.shuffle_xor(32, 64))
        row_max = _maximum(row_max * scale, fx.Float32(_MAX_FLOOR)) + 1.0
        s_0 = _center(s_0, scale, row_max)
        row_sum = fx.Float32(0.0)
        _stage_end()
        # Memory: K(1) g2s; no compute.
        rows_1, rows_2 = kv_rows(ids_1), kv_rows(ids_2)
        read_k_g2s(rows_1)
        lgkm.wait(vmcnt=0)
        _stage_end()
        _stage_end()
        return s_0, row_max, row_sum, o_lo, o_hi, rows_0, rows_1, rows_2

    @flyc.jit
    def phase(s_prev, row_max, row_sum, o_lo, o_hi, rows_prev, rows_cur, rows_next, t):
        """Phase t: QK and row max of key tile t; exp, row sum and PV of tile t - 1.

        rows_prev, rows_cur and rows_next are the DMA rows of tiles t - 1, t and t + 1.
        k_lds holds K(t) up to step 1 and K(t + 1) from step 2; v_lds holds V(t - 1).

        step  memory stage                              compute stage
        0     g2r mask(t), ids(t+2); g2s V(t-1) with    QK: S(t) keys 0-31
              s2r K(t) keys 0-31 between its packets    softmax: P(t-1) = exp2(S'(t-1))
        1     s2r K(t) keys 32-63; V(t-1) lands         QK: S(t) keys 32-63, mask(t)
                                                        softmax: row sum of P(t-1), P to BF16
        2     exchange row sums; g2s K(t+1) with        PV: O dims 0-127 += P(t-1) V(t-1)
              s2r V(t-1) dims 0-127 between packets     softmax: l += row sum, max of S(t)
        3     exchange row maxima; K(t+1) lands;        PV: O dims 128-255 += P(t-1) V(t-1)
              s2r V(t-1) dims 128-255                   softmax: m, S'(t); rescale O and l
        """
        s_prev, row_max, row_sum = (
            fx.Vector(s_prev),
            fx.Float32(row_max),
            fx.Float32(row_sum),
        )
        o_lo, o_hi, t = fx.Vector(o_lo), fx.Vector(o_hi), fx.Int32(t)
        rows_prev = fx.Vector(rows_prev)
        rows_cur, rows_next = fx.Vector(rows_cur), fx.Vector(rows_next)
        # Each phase ends with every LDS op retired, so the ledger starts empty.
        lgkm = _LdsQueue()

        # Step 0 memory.
        mask_t = read_mask_g2r(t)
        ids_next2 = read_ids_g2r(_min(t + 2, last_tile))
        k_lo, _ = read_v_g2s(rows_prev, s2r=k_s2r(0), lgkm=lgkm)
        _stage_end()
        # Step 0 compute.
        lgkm.wait()
        s_lo = qk(k_lo)
        p_prev = _exps(s_prev)
        _schedule(32, 1, 1, True)
        _stage_end()

        # Step 1 memory.
        k_hi, ready = _read_s2r(k_s2r(1, lower_first=STAGGER), lgkm)
        rocdl.s_waitcnt(vmcnt=0)
        if fx.const_expr(STAGGER):
            ready(8)
        _stage_end()
        # Step 1 compute.
        lgkm.wait()
        s_hi = qk(k_hi)
        s_cur = _apply_mask(_join(s_lo, s_hi), mask_t)
        sum_prev, p_bf16 = _sum(p_prev), _pack_bf16(p_prev).bitcast(fx.BFloat16)
        # ids(t+2) landed at the vmcnt(0) above; only the SGPR rows cross iterations.
        rows_next2 = kv_rows(ids_next2)
        _schedule(32, 2, 2)
        _stage_end()

        # Step 2 memory.
        sum_others = _cross(sum_prev, cross_addresses, lgkm)
        v_lo, ready = read_k_g2s(rows_next, s2r=v_s2r(0), lgkm=lgkm)
        v_ops = _v_operands(v_lo, 0, ready)
        _stage_end()
        # Step 2 compute.
        o_lo, row_sum, max_cur = _pv(
            p_bf16,
            v_lo,
            o_lo,
            ready,
            v_ops,
            sum_max=(s_cur, sum_prev, sum_others, row_sum),
        )
        _schedule(32, 3, 3)
        _stage_end()

        # Step 3 memory.
        max_others = _cross(max_cur, cross_addresses, lgkm)
        rocdl.s_waitcnt(vmcnt=0)
        v_hi, ready = _read_s2r(v_s2r(1), lgkm)
        v_ops = _v_operands(v_hi, 0, ready)
        _stage_end()
        # Step 3 compute.
        o_hi, s_cur, new_max, ballot = _pv(
            p_bf16,
            v_hi,
            o_hi,
            ready,
            v_ops,
            center=(s_cur, max_cur, max_others, scale, row_max),
        )
        _schedule(32, 3, 4)
        o_lo, o_hi, row_sum = _rescale(o_lo, o_hi, row_sum, row_max, new_max, ballot)
        _stage_end()
        return s_cur, new_max, row_sum, o_lo, o_hi, rows_cur, rows_next, rows_next2

    def epilogue(state):
        """V(last) g2s, keys past the list read 0; exp, row sum and PV of the last tile."""
        s_last, _, row_sum, o_lo, o_hi, rows_last, _, _ = state
        lgkm = _LdsQueue()
        # Memory: V(last) g2s, with P(last) and l computed while it lands.
        read_v_g2s(fx.Vector(rows_last), tail_tile=last_tile)
        p_last = _exps(fx.Vector(s_last))
        sum_last = _sum(p_last)
        sum_last = sum_last + sum_last.shuffle_xor(16, 64)
        sum_last = sum_last + sum_last.shuffle_xor(32, 64)
        row_sum = fx.Float32(row_sum) + sum_last
        p_bf16 = _pack_bf16(p_last).bitcast(fx.BFloat16)
        lgkm.wait(vmcnt=0)
        _stage_end()
        _stage_end()
        # Memory: V(last) dims 0-127 s2r.  Compute: PV O dims 0-127.
        v_lo, _ = _read_s2r(v_s2r(0), lgkm)
        lgkm.wait()
        _stage_end()
        o_lo = _pv(p_bf16, v_lo, o_lo)
        _stage_end()
        # Memory: V(last) dims 128-255 s2r.  Compute: PV O dims 128-255.
        v_hi, _ = _read_s2r(v_s2r(1), lgkm)
        lgkm.wait()
        _stage_end()
        o_hi = _pv(p_bf16, v_hi, o_hi)
        _stage_end()
        if fx.const_expr(not STAGGER):
            # Waves 0-3 wait one stage for waves 4-7.
            _stage_end()
        return o_lo, o_hi, row_sum

    # Block ids run two tiles ahead of QK and K one tile ahead; PV runs one tile behind.
    state = prologue()
    # Two phases per trip swap the max registers; one phase copies it after its barrier.
    for t in range(fx.Int32(1), key_tiles - 1, fx.Int32(2)):
        state = phase(*state, t)
        state = phase(*state, fx.Int32(t) + 1)
    if (key_tiles > 1) & ((key_tiles & 1) == 0):
        state = phase(*state, key_tiles - 1)
    o_lo, o_hi, row_sum = epilogue(state)
    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0))
    out = rocdl.make_buffer_tensor(
        fx.make_view(
            fx.get_iter(OUT) + fx.Int64(q0) * (H * D) + hkv * (H // HK) * D,
            fx.make_layout(NQ * H * D, 1),
        ),
        num_records_bytes=qextent,
    )
    pinned_tid = _pin_i32(tid)
    o_parts = _shuffle_o_r2s2r(o_lo, o_hi, inv, (k_lds, v_lds), pinned_tid)
    _store_o_r2g(
        o_parts, out, pinned_tid, H, H // HK, GP, BQ, qvalid, qextent, row_offset
    )
    # The next task's K(0) DMA overwrites k_lds.
    _stage_end()


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
    k_lds, v_lds = (
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
                    k_lds,
                    v_lds,
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
                    k_lds,
                    v_lds,
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

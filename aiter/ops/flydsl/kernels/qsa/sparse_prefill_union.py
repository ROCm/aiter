# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942/gfx950 D256 multi-query/multi-head QSA sparse attention with exact membership.

The LDS, QK/PV MFMA and staggered pipeline come from a dense BM128/BN64 D256
attention kernel (see sparse_prefill_common). Q/O rows and sparse masks differ.
On gfx950 the same pipeline runs K32 MFMAs, transposed V reads, native BF16
conversion and 64-bank K/V swizzles (sparse_prefill_common's gfx950 part), and
double-buffers K and V in LDS (STAGES950).
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import rocdl

from aiter.jit.utils.chip_info import get_gfx_runtime

from .sparse_prefill_common import (
    _MAX_FLOOR,
    _buffer,
    _buffer_at,
    _buffer_words,
    _copy_coordinates,
    _cross,
    _exps,
    _join,
    _k_bases950,
    _k_lds_address,
    _k_operands,
    _k_operands950,
    _k_phase,
    _k_reads,
    _k_reads950,
    _k_swizzle950,
    _key_row,
    _kv_extent,
    _kv_offset,
    _lds_row950,
    _lds_words,
    _LdsQueue,
    _load_lds,
    _load_lds_tr,
    _max,
    _maximum,
    _min,
    _o_piece950,
    _pack_bf16,
    _pack_bf16_950,
    _pin,
    _pin_i32,
    _pv,
    _pv950,
    _qk,
    _qk950,
    _read_q_g2r,
    _read_s2r,
    _ReadPlan,
    _rescale,
    _schedule,
    _schedule_exps,
    _stage_end,
    _sum,
    _target_features,
    _uniform,
    _v_lds_address,
    _v_operands,
    _v_operands950,
    _v_reads,
    _v_swizzle950,
    _v_tr_bases950,
    _v_tr_reads950,
    _write_address,
)

__all__ = ["run"]

BM, BN, D, THREADS, LDS_BYTES = 128, 64, 256, 512, 65536
# gfx950: k_lds and v_lds hold STAGES950 tiles each, stage t & 1 holding tile t (LDS_BYTES per
# store, 128 KiB in all). A DMA then never overwrites the tile the previous reads use, so these
# rendezvous (_body's numbering) keep only their compiler fences there:
#   prologue 4  K(1) DMA after the K(0) reads of the other group
#   phase 0     V(t - 1) DMA of waves 4-7 after their V(t - 2) reads
#   phase 3     K(t + 1) DMA of waves 0-3 after the K(t) keys 32-63 reads of waves 4-7
#   phase 4     K(t + 1) DMA of waves 4-7 after their K(t) reads
#   prologue 7, epilogue 0  phase 0 of the phase that follows (or of none)
STAGES950 = 2
DROPPED_PROLOGUE950 = (4, 7)
DROPPED_PHASE950 = (0, 3, 4)
DROPPED_EPILOGUE950 = (0,)


def _read_ids_g2r(blocks, tile, count, offset):
    """Block ids of a key tile (16 blocks of 4 keys); each lane reads the one at byte offset."""
    # SALU-only address: the descriptor starts at the tile; slots past count read 0 (masked).
    view = fx.make_view(blocks + tile * 16, fx.make_layout(16, 1))
    return _buffer_words(_buffer(view, (count - tile * 16) * 4), offset, 1)[0]


def _kv_rows(ids, wave, row_bytes, slotted):
    """Row offsets of the wave's four blocks for the K/V DMA (K/V rows row_bytes apart; the
    DMA's SOFFSET on gfx942, added to its VOFFSET on gfx950): lanes 0..3 hold their ids.

    With a slot table the ids are pool blocks, kept as is for per-block descriptors."""
    rows = []
    for slot in range(4):
        block = fx.Uint32(rocdl.readlane(fx.Int32.ir_type, ids, slot))
        if not slotted:
            block = block * (4 * row_bytes) + (wave >> 2) * 256
        rows.append(fx.Int32(block))
    return fx.Vector.from_elements(rows, fx.Int32)


def _request_buffer(tensor, k0, hkv, hk, gap, extent):
    """Buffer pointer over KV head hkv of a request's K or V rows from row k0 (hk * D
    values, then gap elements of padding, a row); loads at or past extent return zero.
    """
    pointer = fx.get_iter(tensor) + _kv_offset(fx.Int64(k0), hkv, hk, gap)
    return _buffer(fx.make_view(pointer, fx.make_layout(extent // 2, 1)), extent)


def _shift(reads, offset):
    """A read plan [(slot, address, immediate)] moved by offset bytes: a static offset goes into the immediates, a
    runtime one into the addresses."""
    if isinstance(offset, int):
        return [
            (slot, address, immediate + offset) for slot, address, immediate in reads
        ]
    return [(slot, address + offset, immediate) for slot, address, immediate in reads]


def _dma_word(lane, token, is_v):
    """Word of the token's 256-byte half row that the lane fetches (gfx942); the LDS DMA stores it at word lane. The
    XOR is the K swizzle, chunks by _k_phase."""
    return lane if is_v else lane ^ (_k_phase(token) * 4)


def _dma_chunk950(lane, token, is_v):
    """16-byte chunk of the token's 256-byte half row that the lane fetches in a gfx950 x4 LDS DMA, which stores it
    at position lane & 15 of that half row: the LDS swizzle of that position (its own inverse).
    """
    return (_v_swizzle950 if is_v else _k_swizzle950)(token, lane & 15)


def _dma_block950(wave, block):
    """(first key, LDS byte offset) of the wave's x4 DMA block `block`: keys (wave & 3) * 16 +
    4 * block + 0..3, dims half wave >> 2."""
    first = (wave & 3) * 16 + block * 4
    return first, _lds_row950(first, wave >> 2)


def _read_packet_g2s(
    resource,
    storage,
    wave,
    lane,
    rows,
    compact_len,
    limits,
    row_bytes,
    packet,
    is_v,
    extent,
    tail_tile=None,
    reads=(),
    load=_load_lds,
):
    """LDS DMA of key (wave & 3) * 16 + packet, dims (wave >> 2) * 128 + 0..127 (gfx942;
    gfx950: _read_block_g2s950).

    The LDS reads (storage, address, immediate), issued with load, go just before it.
    """
    # Token k of a block reads SOFFSET min(row, limits[k]) + k rows: a partial final block re-reads
    # the request's last token, whose K is masked and V weighted 0; with fewer than k + 1 tokens
    # the limit is 0 and the range-checked VOFFSET + offset returns 0, so reads stay in the request.
    # Slot-table launches pass limits None, a resource (block 0 address, final partial block, its
    # records, records of a full block) and rows that are pool blocks: each block gets its own
    # descriptor (64-bit base), whose records end at the request's last token.
    slotted = limits is None
    token, _ = _copy_coordinates(wave, packet, 4)
    voffset = _dma_word(lane, token, is_v) * 4
    step = (packet & 3) * row_bytes
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
    if slotted:
        base, final, partial, full = resource
        block = rows[packet // 4]
        records = (block == final).select(partial, full)
        resource = rocdl.get_buffer_rsrc(
            _buffer_at(base + fx.Int64(fx.Uint32(block)) * (4 * row_bytes), records)
        )
        soffset = fx.Uint32(0)
    else:
        row, limit = fx.Uint32(rows[packet // 4]), limits[packet & 3]
        soffset = _min(row, limit)
    data = [load(*read) for read in reads]
    # Raw LDS DMA; the pipeline places the vmcnt waits. (BufferCopyLDS32b compiles to the
    # same code: the compiler's wait before the LDS reads falls where the pipeline's does.)
    rocdl.raw_ptr_buffer_load_lds(
        resource,
        fx.to_llvm_ptr(fx.get_iter(storage) + (base_destination + immediate)),
        4,
        fx.Int32(voffset),
        soffset,
        offset,
    )
    return data


def _read_block_g2s950(
    resource,
    storage,
    wave,
    lane,
    rows,
    compact_len,
    limit,
    row_bytes,
    block,
    is_v,
    extent,
    tail_tile=None,
    reads=(),
    load=_load_lds,
    lds_offset=0,
):
    """gfx950 x4 LDS DMA of the wave's block `block`: keys (wave & 3) * 16 + 4 * block + 0..3,
    dims (wave >> 2) * 128 + 0..127, into storage at lds_offset (the tile's stage).

    buffer_load_dwordx4 ... lds writes 16 bytes per lane at M0 + 16 * lane (CDNA4 ISA 9.1.9):
    lane l moves chunk _dma_chunk950 of key l >> 4's half row to its chunk l & 15, so the four
    half rows are the 1 KiB at _dma_block950's offset. limit is limits[l >> 4] (None: slot
    table). The LDS reads go just before it, as in _read_packet_g2s.
    """
    first, destination = _dma_block950(wave, block)
    key = lane >> 4
    token = first + key
    voffset = key * row_bytes + _dma_chunk950(lane, token, is_v) * 16
    if limit is None:
        # The block's own descriptor, as in _read_packet_g2s.
        base, final, partial, full = resource
        block_id = rows[block]
        records = (block_id == final).select(partial, full)
        resource = _buffer_at(
            base + fx.Int64(fx.Uint32(block_id)) * (4 * row_bytes), records
        )
    else:
        # _read_packet_g2s' SOFFSET min(row, limits[k]), per lane; with key k's row step it is
        # at most the request's last key (limit), or k rows past the request start (limit 0):
        # the range check returns 0 past the request's end, and with k + 1 tokens that row is
        # the last key. Below 4 GiB either way.
        voffset = fx.Int32(_min(fx.Uint32(rows[block]), limit)) + voffset
    if tail_tile is not None:
        voffset = (tail_tile * BN + token < compact_len).select(
            voffset, fx.Int32(extent)
        )
    data = [load(*read) for read in reads]
    fx.copy(
        fx.make_copy_atom(rocdl.BufferCopyLDS128b(), fx.Int8),
        fx.make_view(resource + fx.Int32(voffset), fx.make_layout(16, 1)),
        fx.make_view(
            fx.get_iter(storage) + (destination + lds_offset), fx.make_layout(16, 1)
        ),
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


def _shuffle_o_r2s2r(o_lo, o_hi, inv, storages, tid, gfx950=False):
    """O * inv as BF16 through LDS; returns the thread's eight 16-byte pieces of O rows.

    Piece i of a half is the lane's dims 8i .. 8i + 7: output words n * 4 + i (gfx942), _o_piece950(i, g) (gfx950,
    lane group g).
    """
    # O rows 0-63 go to the K buffer, rows 64-127 to the V buffer.
    row = (tid >> 6) * 16 + (tid & 15)
    shared = [fx.Int32(fx.ptrtoint(fx.get_iter(storage))) for storage in storages]
    base = (tid < THREADS // 2).select(shared[0], shared[1])
    for half in range(2):
        output = o_hi if half else o_lo
        for i in range(4):
            if gfx950:
                # Both lane-group parities pack the same pairs (_o_piece950), so the
                # select only reorders words.
                words = [
                    _pack_bf16_950(
                        fx.Vector.from_elements(
                            [output[j] * inv for j in _o_piece950(i, g)], fx.Float32
                        )
                    )
                    for g in range(2)
                ]
                odd = ((tid >> 4) & 1) == 1
                words = fx.Vector.from_elements(
                    [odd.select(words[1][p], words[0][p]) for p in range(4)], fx.Uint32
                )
            else:
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
    TAILS,
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
    SLOTTED: fx.Constexpr[bool],
    GFX950: fx.Constexpr[bool],
    KV_STRIDE: fx.Constexpr[int],
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

    GFX950 swaps the per-arch pieces (sparse_prefill_common's gfx950 part): half the MFMAs
    (K32), 32 transposed V reads per half (eight before each of its four x4 LDS DMA blocks),
    native BF16, and the half-major K/V layout and swizzles of its LDS DMA and reads. Stages
    and data flow are unchanged; lgkmcnt waits are capped at 15, and the double buffer below
    drops the rendezvous it makes redundant.

    GFX950 also double-buffers K and V: tile t sits in stage t & 1 of k_lds / v_lds. Each
    stage end is a rendezvous of all eight waves; waves 4-7 reach rendezvous r one stage
    later than waves 0-3 (prologue: stage k ends at r = k, plus one past the staggered
    rendezvous 1 for waves 4-7; phase: stage s ends at r = (s + lag) % 8; epilogue: stage k
    at r = k + lag). With two stages the rendezvous in DROPPED_*950 order no LDS producer and
    consumer (module comment), and keep only their compiler fences.

    K/V rows sit KV_STRIDE elements apart: HK * D values, then a gap of padding (e.g. the
    V half of an interleaved K|V record per token); only padded rows compile its terms.
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
    ids_offset = ((wave & 3) * 4 + (lane & 3)) * 4
    row = row_offset + wave * 16 + (lane & 15)
    query, head = row // GP, row % GP
    mask_offset = _min(query, fx.Int32(BQ - 1)) * 8 + (lane >> 4) * 2
    valid_row = (query < qvalid) & (query < BQ) & (head < (H // HK))
    # Byte offsets are unsigned 32-bit: Q and each request's K/V may span up to 4 GiB.
    qextent = fx.Uint32(NQ - q0) * H * D * 2 - hkv * (H // HK) * D * 2
    qptr = fx.get_iter(Q) + fx.Int64(q0) * (H * D) + hkv * (H // HK) * D
    gq = _buffer(fx.make_view(qptr, fx.make_layout(NQ * H * D, 1)), qextent)
    qoffset = valid_row.select(fx.Uint32((query * H * D + head * D) * 2), qextent)
    # Q stays in registers for all key tiles.
    if fx.const_expr(GFX950):
        q = _read_q_g2r(gq, qoffset, lane, 32)
    else:
        q = _read_q_g2r(gq, qoffset, lane)
    row_bytes, gap = KV_STRIDE * 2, KV_STRIDE - HK * D
    if fx.const_expr(SLOTTED):
        # BLOCKS holds K3's pool blocks: each gets a descriptor from the pool's base (no 4 GiB
        # limit); TAILS holds the final partial block's first pool row (-1: none).
        half = (wave >> 2) * 256
        tail = _uniform(TAILS[tile_id])
        final = (tail >= 0).select(tail >> 2, fx.Int32(-1))
        partial = (kv_len & 3) * row_bytes - half
        full = fx.Int32(4 * row_bytes) - half
        offset = fx.Int64(hkv * D * 2 + half)
        gk = (fx.Int64(fx.ptrtoint(fx.get_iter(K))) + offset, final, partial, full)
        gv = (fx.Int64(fx.ptrtoint(fx.get_iter(V))) + offset, final, partial, full)
        extent = fx.Uint32(4 * row_bytes)
        limits = None
    else:
        extent = _kv_extent(kv_len, hkv, HK, gap)
        limit = fx.Uint32(kv_len - 1) * row_bytes + (wave >> 2) * 256
        limits = tuple(
            (limit > k * row_bytes).select(limit - k * row_bytes, fx.Uint32(0))
            for k in range(4)
        )
        gk = _request_buffer(K, k0, hkv, HK, gap, extent)
        gv = _request_buffer(V, k0, hkv, HK, gap, extent)
        if fx.const_expr(not GFX950):
            # _read_packet_g2s' raw LDS DMA takes the buffer resources.
            gk, gv = rocdl.get_buffer_rsrc(gk), rocdl.get_buffer_rsrc(gv)
    # gfx950 x4 DMA: each lane's own limits[lane >> 4] (_read_block_g2s950); slot tables none.
    dma_limit = None
    if fx.const_expr(GFX950 and not SLOTTED):
        key_bytes = fx.Uint32(lane >> 4) * row_bytes
        dma_limit = (limit > key_bytes).select(limit - key_bytes, fx.Uint32(0))
    scale = fx.Float32(SCALE * math.log2(math.e))
    compact_len = count * 4
    key_tiles = (compact_len + BN - 1) // BN
    last_tile = key_tiles - 1
    # Per-arch pieces; the stages below are shared. Schedules: (MFMAs, VALU per MFMA).
    if fx.const_expr(GFX950):
        k_s2r_base = _k_bases950(lane)
        v_s2r_base = _v_tr_bases950(lane)
        k_reads, v_reads, v_load = _k_reads950, _v_tr_reads950, _load_lds_tr
        k_operands, qk_mfma, pack_bf16 = _k_operands950, _qk950, _pack_bf16_950
        v_operands, pv = _v_operands950, _pv950
        qk_schedule, mask_schedule, pv_schedule, exp_mfmas = (
            (16, 4),
            (16, 5),
            (16, 3),
            1,
        )
    else:
        key_row = _key_row(lane)
        k_s2r_base = tuple(
            _k_lds_address(key_row + n * 4, (lane >> 4) * 8 + parity * 32)
            for n in range(2)
            for parity in range(2)
        )
        v_s2r_base = _v_lds_address((lane >> 4) * 8, (lane & 15) * 8)
        k_reads, v_reads, v_load = _k_reads, _v_reads, _load_lds
        k_operands, qk_mfma, pack_bf16 = _k_operands, _qk, _pack_bf16
        v_operands, pv = _v_operands, _pv
        qk_schedule, mask_schedule, pv_schedule, exp_mfmas = (
            (32, 3),
            (32, 2),
            (32, 3),
            2,
        )
    cross_addresses = tuple((lane ^ offset) * 4 for offset in (16, 32, 48))
    # gfx950 double buffer: byte offset of stage 1 in k_lds / v_lds, and the rendezvous it
    # makes redundant (gfx942: one stage, every rendezvous kept). lag: waves 4-7 run one behind.
    stage = (LDS_BYTES // 2) * int(GFX950)
    dropped_prologue, dropped_phase, dropped_epilogue = (
        ((), (), ()),
        (DROPPED_PROLOGUE950, DROPPED_PHASE950, DROPPED_EPILOGUE950),
    )[int(GFX950)]
    lag = int(STAGGER)

    def stage_end(rendezvous, dropped):
        """_stage_end at rendezvous; one the double buffer made redundant keeps only its fences."""
        if fx.const_expr(rendezvous in dropped):
            rocdl.sched_barrier(0)
            rocdl.sched_barrier(0)
        else:
            _stage_end()

    # g = global, s = LDS, r = registers; reads only issue, the stages place the waits.
    def read_ids_g2r(tile):
        return _read_ids_g2r(blocks, tile, count, ids_offset)

    def read_mask_g2r(tile):
        return _read_mask_g2r(mask_resource, tile, mask_offset, BQ)

    def kv_rows(ids):
        return _kv_rows(ids, wave, row_bytes, SLOTTED)

    def k_s2r(half, lower_first=False, offset=0):
        """Read plan of K keys 32 * half .. 32 * half + 31 of the K stage at byte offset."""
        return _ReadPlan(
            k_lds, _shift(k_reads(k_s2r_base, half, lower_first), offset), _load_lds
        )

    def v_s2r(half, offset=0):
        """Read plan of V dims 128 * half .. 128 * half + 127 of the V stage at byte offset."""
        return _ReadPlan(v_lds, _shift(v_reads(v_s2r_base, half), offset), v_load)

    def read_tile_g2s(resource, storage, rows, is_v, s2r, lgkm, tail_tile, offset):
        # 16 LDS-DMA packets (gfx950: 4 x4 blocks); the s2r plan's reads go in plan order, an
        # equal share (one; gfx950: four K or eight V) just before each.
        reads, load = (), _load_lds
        if fx.const_expr(s2r is not None):
            reads, load = s2r.reads, s2r.load
        packets = (16, 4)[int(GFX950)]
        share = len(reads) // packets
        parts = [None] * len(reads)
        for packet in fx.range_constexpr(packets):
            batch = reads[packet * share : (packet + 1) * share]
            issued = [
                (s2r.storage, address, immediate) for _, address, immediate in batch
            ]
            if fx.const_expr(GFX950):
                data = _read_block_g2s950(
                    resource,
                    storage,
                    wave,
                    lane,
                    rows,
                    compact_len,
                    dma_limit,
                    row_bytes,
                    packet,
                    is_v,
                    extent,
                    tail_tile,
                    issued,
                    load,
                    offset,
                )
            else:
                data = _read_packet_g2s(
                    resource,
                    storage,
                    wave,
                    lane,
                    rows,
                    compact_len,
                    limits,
                    row_bytes,
                    packet,
                    is_v,
                    extent,
                    tail_tile,
                    issued,
                    load,
                )
            for i in fx.range_constexpr(share):
                parts[batch[i][0]] = data[i]
        if fx.const_expr(s2r is None):
            return None
        return _lds_words(parts), lgkm.issue(len(reads))

    def read_k_g2s(rows, s2r=None, lgkm=None, offset=0):
        return read_tile_g2s(gk, k_lds, rows, False, s2r, lgkm, None, offset)

    def read_v_g2s(rows, s2r=None, lgkm=None, tail_tile=None, offset=0):
        return read_tile_g2s(gv, v_lds, rows, True, s2r, lgkm, tail_tile, offset)

    def qk(k):
        return qk_mfma(q, k_operands(k))

    def prologue():
        """Ids of tiles 0-2 and mask(0) g2r; K(0) g2s, QK(0), m and S'(0); K(1) g2s."""
        lgkm = _LdsQueue()
        ids_0 = read_ids_g2r(fx.Int32(0))
        ids_1 = read_ids_g2r(_min(fx.Int32(1), last_tile))
        ids_2 = read_ids_g2r(_min(fx.Int32(2), last_tile))
        mask_0 = read_mask_g2r(fx.Int32(0))
        rows_0 = kv_rows(ids_0)
        # Memory: K(0) g2s (stage 0).
        read_k_g2s(rows_0)
        lgkm.wait(vmcnt=0)
        stage_end(0, dropped_prologue)
        if fx.const_expr(STAGGER):
            # Waves 4-7 fall one stage behind.
            stage_end(1, dropped_prologue)
        # Memory: K(0) keys 0-31 s2r.
        k_lo, _ = _read_s2r(k_s2r(0), lgkm)
        lgkm.wait()
        stage_end(1 + lag, dropped_prologue)
        # Compute: QK S(0) keys 0-31; O = 0.
        s_lo = qk(k_lo)
        o_lo = _pin(fx.Vector.filled(32, 0.0, fx.Float32))
        o_hi = _pin(fx.Vector.filled(32, 0.0, fx.Float32))
        _schedule(*qk_schedule, 5)
        stage_end(2 + lag, dropped_prologue)
        # Memory: K(0) keys 32-63 s2r.
        k_hi, _ = _read_s2r(k_s2r(1), lgkm)
        lgkm.wait()
        stage_end(3 + lag, dropped_prologue)
        # Compute: QK S(0) keys 32-63 and mask(0); softmax m and S'(0).
        s_hi = qk(k_hi)
        s_0 = _apply_mask(_join(s_lo, s_hi), mask_0) * scale
        row_max = _max(s_0)
        row_max = _maximum(row_max, row_max.shuffle_xor(16, 64))
        row_max = _maximum(row_max, row_max.shuffle_xor(32, 64))
        row_max = _maximum(row_max, fx.Float32(_MAX_FLOOR)) + 1.0
        row_sum = fx.Float32(0.0)
        stage_end(4 + lag, dropped_prologue)
        # Memory: K(1) g2s (stage 1 on gfx950); no compute.
        rows_1, rows_2 = kv_rows(ids_1), kv_rows(ids_2)
        read_k_g2s(rows_1, offset=stage)
        lgkm.wait(vmcnt=0)
        stage_end(5 + lag, dropped_prologue)
        stage_end(6 + lag, dropped_prologue)
        return s_0, row_max, row_sum, o_lo, o_hi, rows_0, rows_1, rows_2

    @flyc.jit
    def phase(
        s_prev, row_max, row_sum, o_lo, o_hi, rows_prev, rows_cur, rows_next, t, odd
    ):
        """Phase t: QK and row max of key tile t; exp, row sum and PV of tile t - 1.

        rows_prev, rows_cur and rows_next are the DMA rows of tiles t - 1, t and t + 1; odd is
        t & 1, a Python int (the loop runs odd and even phases in pairs).
        k_lds holds K(t) up to step 1 and K(t + 1) from step 2; v_lds holds V(t - 1). On gfx950
        K(t) sits in stage odd, K(t + 1) and V(t - 1) in stage 1 - odd.
        S' = S * scale; m tracks the row max of S' lazily (S' - m <= 7).

        step  memory stage                              compute stage
        0     g2r mask(t), ids(t+2); g2s V(t-1) with    QK: S(t) keys 0-31
              s2r K(t) keys 0-31 between its packets    softmax: P(t-1) = exp2(S' - m)
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
        cur, other = stage * odd, stage * (1 - odd)

        # Step 0 memory.
        mask_t = read_mask_g2r(t)
        ids_next2 = read_ids_g2r(_min(t + 2, last_tile))
        k_lo, _ = read_v_g2s(
            rows_prev, s2r=k_s2r(0, offset=cur), lgkm=lgkm, offset=other
        )
        stage_end(lag, dropped_phase)
        # Step 0 compute.
        lgkm.wait()
        s_lo = qk(k_lo)
        p_prev = _exps(s_prev, row_max)
        _schedule_exps(16, 1, exp_mfmas)
        stage_end(1 + lag, dropped_phase)

        # Step 1 memory.
        k_hi, ready = _read_s2r(k_s2r(1, lower_first=STAGGER, offset=cur), lgkm)
        rocdl.s_waitcnt(vmcnt=0)
        if fx.const_expr(STAGGER):
            ready(8)
        stage_end(2 + lag, dropped_phase)
        # Step 1 compute.
        lgkm.wait()
        s_hi = qk(k_hi)
        s_cur = _apply_mask(_join(s_lo, s_hi), mask_t)
        sum_prev, p_bf16 = _sum(p_prev), pack_bf16(p_prev).bitcast(fx.BFloat16)
        # ids(t+2) landed at the vmcnt(0) above; only the SGPR rows cross iterations.
        rows_next2 = kv_rows(ids_next2)
        _schedule(*mask_schedule, 2)
        stage_end(3 + lag, dropped_phase)

        # Step 2 memory.
        sum_others = _cross(sum_prev, cross_addresses, lgkm)
        v_lo, ready = read_k_g2s(
            rows_next, s2r=v_s2r(0, other), lgkm=lgkm, offset=other
        )
        v_ops = v_operands(v_lo, 0, ready)
        stage_end(4 + lag, dropped_phase)
        # Step 2 compute.
        o_lo, row_sum, max_cur = pv(
            p_bf16,
            v_lo,
            o_lo,
            ready,
            v_ops,
            sum_max=(s_cur, sum_prev, sum_others, row_sum),
        )
        _schedule(*pv_schedule, 3)
        stage_end(5 + lag, dropped_phase)

        # Step 3 memory.
        max_others = _cross(max_cur, cross_addresses, lgkm)
        rocdl.s_waitcnt(vmcnt=0)
        v_hi, ready = _read_s2r(v_s2r(1, other), lgkm)
        v_ops = v_operands(v_hi, 0, ready)
        stage_end(6 + lag, dropped_phase)
        # Step 3 compute.
        o_hi, s_cur, new_max, ballot = pv(
            p_bf16,
            v_hi,
            o_hi,
            ready,
            v_ops,
            center=(s_cur, max_cur, max_others, scale, row_max),
        )
        _schedule(*pv_schedule, 4)
        o_lo, o_hi, row_sum = _rescale(o_lo, o_hi, row_sum, row_max, new_max, ballot)
        stage_end((7 + lag) % 8, dropped_phase)
        return s_cur, new_max, row_sum, o_lo, o_hi, rows_cur, rows_next, rows_next2

    def epilogue(state):
        """V(last) g2s, keys past the list read 0; exp, row sum and PV of the last tile."""
        s_last, row_max, row_sum, o_lo, o_hi, rows_last, _, _ = state
        lgkm = _LdsQueue()
        # V(last) goes to stage last & 1 (gfx950), not the stage V(last - 1) was read from.
        if fx.const_expr(GFX950):
            v_stage = (last_tile & 1) * stage
        else:
            v_stage = 0
        # Memory: V(last) g2s, with P(last) and l computed while it lands.
        read_v_g2s(fx.Vector(rows_last), tail_tile=last_tile, offset=v_stage)
        p_last = _exps(fx.Vector(s_last), fx.Float32(row_max))
        sum_last = _sum(p_last)
        sum_last = sum_last + sum_last.shuffle_xor(16, 64)
        sum_last = sum_last + sum_last.shuffle_xor(32, 64)
        row_sum = fx.Float32(row_sum) + sum_last
        p_bf16 = pack_bf16(p_last).bitcast(fx.BFloat16)
        lgkm.wait(vmcnt=0)
        stage_end(lag, dropped_epilogue)
        stage_end(1 + lag, dropped_epilogue)
        # Memory: V(last) dims 0-127 s2r.  Compute: PV O dims 0-127.
        v_lo, _ = _read_s2r(v_s2r(0, v_stage), lgkm)
        lgkm.wait()
        stage_end(2 + lag, dropped_epilogue)
        o_lo = pv(p_bf16, v_lo, o_lo)
        stage_end(3 + lag, dropped_epilogue)
        # Memory: V(last) dims 128-255 s2r.  Compute: PV O dims 128-255.
        v_hi, _ = _read_s2r(v_s2r(1, v_stage), lgkm)
        lgkm.wait()
        stage_end(4 + lag, dropped_epilogue)
        o_hi = pv(p_bf16, v_hi, o_hi)
        stage_end(5 + lag, dropped_epilogue)
        if fx.const_expr(not STAGGER):
            # Waves 0-3 wait one stage for waves 4-7.
            stage_end(6, dropped_epilogue)
        return o_lo, o_hi, row_sum

    # Block ids run two tiles ahead of QK and K one tile ahead; PV runs one tile behind.
    state = prologue()
    # Two phases per trip swap the max registers; one phase copies it after its barrier.
    for t in range(fx.Int32(1), key_tiles - 1, fx.Int32(2)):
        state = phase(*state, t, 1)
        state = phase(*state, fx.Int32(t) + 1, 0)
    if (key_tiles > 1) & ((key_tiles & 1) == 0):
        state = phase(*state, key_tiles - 1, 1)
    o_lo, o_hi, row_sum = epilogue(state)
    inv = (row_sum > 0.0).select(fx.Float32(1.0) / row_sum, fx.Float32(0.0))
    out = rocdl.make_buffer_tensor(
        fx.make_view(
            fx.get_iter(OUT) + fx.Int64(q0) * (H * D) + hkv * (H // HK) * D,
            fx.make_layout(NQ * H * D, 1),
        ),
        num_records_bytes=qextent,
    )
    # Pin keeps output addressing inside the task loop; the mask restores tid's range.
    pinned_tid = _pin_i32(tid) & (THREADS - 1)
    o_parts = _shuffle_o_r2s2r(o_lo, o_hi, inv, (k_lds, v_lds), pinned_tid, GFX950)
    _store_o_r2g(
        o_parts, out, pinned_tid, H, H // HK, GP, BQ, qvalid, qextent, row_offset
    )
    # The next task's K(0) DMA overwrites k_lds.
    _stage_end()


@flyc.kernel(name="sparse_prefill_union_bf16_d256", known_block_size=[THREADS, 1, 1])
def _kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    TAILS: fx.Tensor,
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
    SLOTTED: fx.Constexpr[bool],
    GFX950: fx.Constexpr[bool],
    KV_STRIDE: fx.Constexpr[int],
):
    body = _body
    allocator = fx.SharedAllocator()
    # gfx950: two stages of each (K/V double buffer); gfx942: one.
    store = (LDS_BYTES // 2) * (STAGES950 if GFX950 else 1)
    k_lds, v_lds = (
        allocator.allocate(fx.Array[fx.Int8, store, 16])
        .peek()
        .view(fx.make_layout(store, 1))
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
                    TAILS,
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
                    SLOTTED,
                    GFX950,
                    KV_STRIDE,
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
                    TAILS,
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
                    SLOTTED,
                    GFX950,
                    KV_STRIDE,
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
    TAILS: fx.Tensor,
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
    SLOTTED: fx.Constexpr[bool],
    GFX950: fx.Constexpr[bool],
    KV_STRIDE: fx.Constexpr[int],
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
            TAILS,
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
            SLOTTED,
            GFX950,
            KV_STRIDE,
            value_attrs={
                "rocdl.waves_per_eu": 2,
                "llvm.target_features": _target_features(),
            },
        ).launch(grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED = {}


def _args(inputs, plan, out, stream):
    # Slot-table plans read K/V through K3's pool block table, addressed from the pool's base.
    return (
        inputs.q.view(-1),
        inputs.k,
        inputs.v,
        out.view(-1),
        plan.metadata.view(-1),
        (plan.pool_blocks if plan.slotted else plan.blocks).view(-1),
        plan.score_masks.view(-1),
        plan.counts.view(-1),
        plan.tails if plan.slotted else plan.counts,
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
        plan.slotted,
        get_gfx_runtime() == "gfx950",
        inputs.kv_stride,
        stream,
    )


def _compiled(inputs, args, stream):
    key = (
        inputs.q.device,
        args[11],
        args[12],
        args[15],
        args[16],
        args[19],
        args[20],
        args[21],
        args[22],
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        with torch.cuda.device(inputs.q.device), torch.cuda.stream(stream):
            compiled = _COMPILED[key] = flyc.compile(
                _launch, *args[:18], 0, *args[19:23], stream
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
    """launch(q, k, v, out, slots) with flat Q/out and unflattened K/V; other arguments
    prebuilt. Union reads pool rows through K3's table, so the slot table is unused here.

    Valid while the plan's scratch bindings and the current stream stay the same.
    """
    if plan.num_tiles == 0:
        return None
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, out, stream)
    compiled = _compiled(inputs, args, stream)
    tail = args[4:]
    return lambda q, k, v, o, slots: compiled(q, k, v, o, *tail)

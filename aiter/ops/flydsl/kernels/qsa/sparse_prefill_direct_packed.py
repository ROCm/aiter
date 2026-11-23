# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-call block-packed KV for QSA direct attention, packing included in the launch.

Four-token KV blocks are stored in the native QK/PV operand layouts. One query
wave then needs no K-data bpermutes or PV byte permutations. Original public
KV tensors remain unchanged; private scratch is refreshed on every call.

_DirectLoop is the key loop both direct kernels run; _Packed gives it the packed
layout, sparse_prefill_direct._Raw the raw K/V one.

On gfx950 the same operand words feed 16x16x32 MFMAs and probabilities convert
with v_cvt_pk_bf16_f32; each key tile then takes one V round trip (gfx950 part).
Batch-invariant plans keep _DirectLoop there, the loop raw direct runs, so that
packed and raw K/V give the same bits.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl

from . import sparse_prefill_common as prim
from .sparse_prefill_common import (
    _MAX_FLOOR,
    _buffer,
    _buffer_at,
    _buffer_store_words,
    _buffer_words,
    _exp,
    _join,
    _kv_extent,
    _kv_offset,
    _load_lds,
    _maximum,
    _mfma,
    _min,
    _pack_bf16,
    _pack_bf16_950,
    _pin,
    _pv_step,
    _qk,
    _read_q_g2r,
    _rescale,
    _stage_end,
    _tiles,
    _uniform,
    _untile,
    _write_address,
)


def _enabled(row, rows, first, query_tiles, active, gated, nq):
    valid = row < rows
    if gated:
        # Padded waves read the tile's last row; valid already excludes them.
        tile = fx.Int32(query_tiles[first + _min(row, rows - 1)])
        valid = valid & (fx.Int32(active[tile]) == 0)
    return valid


def _output(
    o0,
    o1,
    inv,
    buffer,
    storage,
    tid,
    h,
    g,
    rows,
    extent,
    threads,
    first,
    query_tiles,
    active,
    gated,
    nq,
    gfx950=False,
):
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    row = (tid >> 6) * 16 + (tid & 15)
    for half in range(2):
        acc = o1 if half else o0
        for i in range(4):
            values = fx.Vector.from_elements(
                [acc[n * 4 + i] * inv for n in range(8)], fx.Float32
            )
            words = (_pack_bf16_950 if gfx950 else _pack_bf16)(values)
            col = (((tid >> 4) & 3) * 4 + i) * 8 + half * 128
            address = fx.Int32(shared + ((row * 256 + col) ^ ((row & 7) * 8)) * 2)
            _write_address(address, words)
    rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    parts = []
    for part in range(8):
        element = tid * 8 + part * threads * 8
        parts.append(_load_lds(storage, (element ^ (((element // 256) & 7) * 8)) * 2))
    rocdl.s_waitcnt(lgkmcnt=0)
    prim.sched_barrier(0)
    for part in range(8):
        element = tid * 8 + part * threads * 8
        r, col = element // 256, element % 256
        query, head = r // 16, r % 16
        enabled = _enabled(query, rows, first, query_tiles, active, gated, nq) & (
            head < g
        )
        offset = enabled.select(
            fx.Uint32(query * h * 256 + head * 256 + col), extent // 2
        )
        fragment = fx.make_rmem_tensor(8, fx.BFloat16)
        fragment.store(parts[part].bitcast(fx.BFloat16))
        fx.copy(
            fx.make_copy_atom(rocdl.BufferCopy128b(), fx.BFloat16),
            fragment,
            fx.make_view(fx.get_iter(buffer) + offset, fx.make_layout(8, 1)),
        )


def _pack_block(
    kr,
    vr,
    pk,
    pv,
    sources,
    pair,
    half,
    lane,
    packed,
    hk,
    stride,
    source_extent,
    extent,
    slots,
    slotted,
):
    block, head = pair // hk, pair % hk
    valid = pair < packed * hk
    start = fx.Int32(sources[_min(block, packed - 1)])
    token, chunk = lane >> 4, lane & 15
    dim = half * 128 + chunk * 8
    # K/V rows sit stride elements apart: hk * 256 values, then a gap of padding.
    row_bytes, gap = stride * 2, stride - hk * 256
    if slotted:
        # A wave packs one block, four consecutive pool rows from the slot table's first:
        # give it its own 64-bit base, so the pool may exceed 4 GiB. kr/vr are the pool
        # addresses here.
        row = _uniform(fx.Int32(slots[_uniform(start)]))
        base = _kv_offset(fx.Int64(row), _uniform(head), hk, gap, 2)
        kr = _buffer_at(kr + base, 4 * row_bytes)
        vr = _buffer_at(vr + base, 4 * row_bytes)
        offset = _kv_offset(token, 0, hk, gap, 2, dim)
        source = valid.select(fx.Uint32(offset), fx.Uint32(4 * row_bytes))
    else:
        # Tokens past a request's end are masked by the direct kernel.
        offset = _kv_offset(fx.Uint32(start + token), head, hk, gap, 2, dim)
        source = valid.select(offset, source_extent)
    key = _buffer_words(kr, source)
    target = (
        fx.Uint32(
            ((block * 4 + dim // 64) * hk + head) * 256
            + (dim % 64 // 32) * 128
            + (dim % 32 // 8) * 32
            + token * 8
        )
        * 2
    )
    target = valid.select(target, extent)
    _buffer_store_words(key, pk, target)
    vector = _buffer_words(vr, source)
    words = [vector[i] for i in range(4)]
    # Exchange the two token bits with the two register-index bits. Each wave
    # owns all four tokens, so the native DS operations need no CTA barrier.
    for bit in range(2):
        parity = (lane & (16 << bit)) != 0
        source_lane = fx.Int32((lane ^ (16 << bit)) * 4)
        next_words = list(words)
        for first in range(4):
            if first & (1 << bit):
                continue
            second = first | (1 << bit)
            kept = parity.select(words[second], words[first])
            sent = parity.select(words[first], words[second])
            received = prim.bpermute(source_lane, sent)
            next_words[first] = parity.select(received, kept)
            next_words[second] = parity.select(kept, received)
        words = next_words
    output = fx.Vector.from_elements(
        [
            (words[0] & 65535) | (words[1] << 16),
            (words[2] & 65535) | (words[3] << 16),
            ((words[0] >> 16) & 65535) | (words[1] & -65536),
            ((words[2] >> 16) & 65535) | (words[3] & -65536),
        ],
        fx.Int32,
    )
    rank = half * 512 + (lane >> 4) * 128 + chunk * 8
    target = valid.select(
        fx.Uint32((block * 4 + rank // 256) * hk + head) * 512 + rank % 256 * 2,
        extent,
    )
    _buffer_store_words(output, pv, target)


@flyc.jit
def _pack_body(K, V, PK, PV, SOURCES, SLOTS, N, PACKED, HK, SLOTTED, KV_STRIDE):
    tid = fx.thread_idx.x
    lane, half = tid & 63, (tid >> 6) & 1
    first = fx.block_idx.x * 8 + (tid >> 7)
    # K/V rows sit KV_STRIDE elements apart: HK * 256 values, then a gap of padding.
    source_extent = _kv_extent(N, 0, HK, KV_STRIDE - HK * 256)
    extent = fx.Uint32(PACKED) * 4 * HK * 512
    if fx.const_expr(SLOTTED):
        kr = fx.Int64(fx.ptrtoint(fx.get_iter(K)))
        vr = fx.Int64(fx.ptrtoint(fx.get_iter(V)))
    else:
        kr, vr = _buffer(K, source_extent), _buffer(V, source_extent)
    pk, pv = _buffer(PK, extent), _buffer(PV, extent)
    slots = rocdl.make_buffer_tensor(SLOTS)
    for u in fx.range_constexpr(4):
        _pack_block(
            kr,
            vr,
            pk,
            pv,
            SOURCES,
            first + u * 2,
            half,
            lane,
            PACKED,
            HK,
            KV_STRIDE,
            source_extent,
            extent,
            slots,
            SLOTTED,
        )


@flyc.kernel(name="sparse_prefill_pack_kv_bf16_d256")
def _pack(
    K: fx.Tensor,
    V: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    SOURCES: fx.Tensor,
    SLOTS: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    HK: fx.Constexpr[int],
    SLOTTED: fx.Constexpr[bool],
    KV_STRIDE: fx.Constexpr[int],
):
    _pack_body(K, V, PK, PV, SOURCES, SLOTS, N, PACKED, HK, SLOTTED, KV_STRIDE)


@flyc.kernel(name="sparse_prefill_pack_kv_gated_bf16_d256")
def _pack_gated(
    K: fx.Tensor,
    V: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    SOURCES: fx.Tensor,
    SLOTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    HK: fx.Constexpr[int],
    SLOTTED: fx.Constexpr[bool],
    KV_STRIDE: fx.Constexpr[int],
):
    if _uniform(DIRECT_FLAG[0]) != 0:
        _pack_body(K, V, PK, PV, SOURCES, SLOTS, N, PACKED, HK, SLOTTED, KV_STRIDE)


def _key_offset(cached, n, visible, extent, hk, topk):
    complete = _min(visible >> 2, fx.Int32(topk))
    width = complete * 4 + (visible & 3)
    selector = fx.Int32((_min(n >> 2, fx.Int32(topk - 1)) & 63) * 4)
    block = prim.bpermute(selector, fx.Int32(cached))
    base = (n < complete * 4).select(
        fx.Uint32(block) * (hk * 2048), fx.Uint32(visible & -4) * (hk * 512)
    )
    return (n < width).select(base + (n & 3) * 16, extent)


def _key_load(resource, offset, lane, hk):
    base = offset + (lane >> 4) * 64
    words = []
    for part in range(8):
        # Invalid offset is already descriptor extent. Nonnegative stripe
        # offsets remain OOB; the host eligibility bound prevents wraparound.
        address = base + (part // 2) * hk * 512 + (part % 2) * 256
        piece = _buffer_words(resource, address)
        words.extend(piece[i] for i in range(4))
    return fx.Vector.from_elements(words, fx.Int32)


@flyc.jit
def _mask_value_tail(value, tile, segment, lane, visible, TOPK: fx.Constexpr[int]):
    value = fx.Vector(value)
    width = _min(visible >> 2, fx.Int32(TOPK)) * 4 + (visible & 3)
    if tile * 32 + 32 > width:
        first = tile * 32 + segment * 16 + (lane >> 4) * 4
        mask0 = (first < width).select(fx.Int32(65535), fx.Int32(0)) | (
            first + 1 < width
        ).select(fx.Int32(-65536), fx.Int32(0))
        mask1 = (first + 2 < width).select(fx.Int32(65535), fx.Int32(0)) | (
            first + 3 < width
        ).select(fx.Int32(-65536), fx.Int32(0))
        value = fx.Vector.from_elements(
            [value[i] & (mask1 if i % 2 else mask0) for i in range(16)], fx.Int32
        )
    return value


def _value_words(resource, source_base, half, lane, hk):
    base = fx.Uint32(source_base) + half * 2 * hk * 512 + (lane & 15) * 16
    parts = [
        _buffer_words(resource, base + (part // 2) * hk * 512 + (part % 2) * 256)
        for part in range(4)
    ]
    return fx.Vector.from_elements(
        [part[i] for part in parts for i in range(4)], fx.Int32
    )


def _value_load(resource, source_base, tile, segment, half, lane, visible, hk, topk):
    value = _value_words(resource, source_base, half, lane, hk)
    # A packed block may contain later tokens with NaNs; zero those BF16
    # elements before MFMA rather than relying on zero probabilities.
    return _mask_value_tail(value, tile, segment, lane, visible, topk)


def _reduce(value, maximum):
    a = fx.gpu.shuffle_xor(value, 16, 64)
    b = fx.gpu.shuffle_xor(value, 32, 64)
    c = fx.gpu.shuffle_xor(value, 48, 64)
    prim.sched_barrier(0)
    if maximum:
        return _maximum(_maximum(value, a), _maximum(b, c))
    return (value + a) + (b + c)


@flyc.jit
def _scale_scores(s0, s1, count, tile, lane, scale):
    first, second = fx.Vector(s0) * scale, fx.Vector(s1) * scale
    if tile * 32 + 32 > count:
        first = fx.Vector.from_elements(
            [
                (tile * 32 + (lane >> 4) * 4 + i < count).select(
                    first[i], fx.Float32(float("-inf"))
                )
                for i in range(4)
            ],
            fx.Float32,
        )
        second = fx.Vector.from_elements(
            [
                (tile * 32 + 16 + (lane >> 4) * 4 + i < count).select(
                    second[i], fx.Float32(float("-inf"))
                )
                for i in range(4)
            ],
            fx.Float32,
        )
    return first, second


class _DirectLoop:
    """The direct kernels' key loop: one query wave over its own selected keys in BN32
    tiles (two 16-key MFMA steps), online softmax per tile. A subclass gives the K/V
    layout (_Packed here, sparse_prefill_direct._Raw): where V rows come from, how K/V
    and block ids load, how scores are masked and reduced, how O rescales, and PV.

    Each tile waits for K(t) at its QK and leaves V(t) segment 0 in flight; K(t + 1)
    loads during PV. The waits count a subclass's loads: values() issues 4 VMEM
    requests (8 per segment), load_key() 8 (16 for both K halves).
    """

    @flyc.jit
    def attend(self, q, k0, k1, offset0, offset1, ids, maximum, total, o0, o1, tiles):
        next_ids = ids
        lane, count = self.lane, self.count
        for t in range(fx.Int32(0), tiles, fx.Int32(1)):
            # The lane holding a four-key block's first K offset gives its V row:
            # no second dependent block-table load.
            row0 = prim.bpermute((lane >> 4) * self.row_lane, fx.Int32(offset0))
            row1 = prim.bpermute((lane >> 4) * self.row_lane, fx.Int32(offset1))
            rocdl.s_waitcnt(lgkmcnt=0)
            prim.sched_barrier(0)
            values0 = self.values(row0, t, 0, 0)
            values1 = self.values(row0, t, 0, 1)
            if ((t + 1) & 7) == 0:
                next_ids = self.next_ids(t)
            prim.sched_barrier(0)
            # The prior tile's sixteen K requests precede these eight V requests: wait
            # for K at its consumer, leaving V in flight for QK. A block-id prefetch
            # is younger still; vmcnt(8) then also waits for the oldest V request.
            rocdl.s_waitcnt(vmcnt=8)
            prim.sched_barrier(0)
            updated, probabilities, total, o0, o1 = self.softmax(
                _qk(q, *self.key_operands(k0, k1)), t, maximum, total, o0, o1
            )
            rocdl.s_waitcnt(lgkmcnt=0)
            prim.sched_barrier(0)
            rocdl.s_waitcnt(vmcnt=0)
            prim.sched_barrier(0)
            if ((t + 1) & 7) == 0:
                ids = next_ids
            following0 = _min((t + 1) * 32, (((count + 15) >> 4) - 1) * 16)
            following1 = _min((t + 1) * 32 + 16, (((count + 15) >> 4) - 1) * 16)
            offset0 = self.key_offset(ids, following0)
            offset1 = self.key_offset(ids, following1)
            p0 = _pack_bf16(
                fx.Vector.from_elements(
                    [probabilities[i] for i in range(4)], fx.Float32
                )
            ).bitcast(fx.Int16)
            p1 = _pack_bf16(
                fx.Vector.from_elements(
                    [probabilities[4 + i] for i in range(4)], fx.Float32
                )
            ).bitcast(fx.Int16)
            next_values0 = self.values(row1, t, 1, 0)
            next_values1 = self.values(row1, t, 1, 1)
            prim.sched_barrier(0)
            o0 = self.pv(p0, values0, o0)
            prim.sched_barrier(0)
            k0 = self.load_key(offset0)
            prim.sched_barrier(0)
            o1 = self.pv(p0, values1, o1)
            prim.sched_barrier(0)
            k1 = self.load_key(offset1)
            prim.sched_barrier(0)
            # Eight V loads precede sixteen K loads: consume V while K stays in flight.
            rocdl.s_waitcnt(vmcnt=16)
            prim.sched_barrier(0)
            o0 = self.pv(p1, next_values0, o0)
            o1 = self.pv(p1, next_values1, o1)
            maximum = updated
        return maximum, total, o0, o1

    def softmax(self, s, t, maximum, total, o0, o1):
        """Online softmax of tile t's scores s (_qk's layout): (new max, P of the lane's
        eight keys, l, O rescaled)."""
        scores = self.scores(s, t)
        candidate = fx.Float32(_MAX_FLOOR)
        for score in scores:
            candidate = _maximum(candidate, score)
        candidate = self.reduce(candidate, True)
        updated = _maximum(maximum, candidate)
        alpha = _exp(maximum - updated)
        probabilities = fx.Vector.from_elements(
            [_exp(score - updated) for score in scores], fx.Float32
        )
        current = fx.Float32(0.0)
        for i in fx.range_constexpr(8):
            current = current + probabilities[i]
        current = self.reduce(current, False)
        total = total * alpha + current
        o0, o1 = self.rescale(o0, o1, alpha)
        return updated, probabilities, total, o0, o1


# ---- gfx950 (CDNA4): 16x16x32 MFMAs, native BF16 ----
#
# A K32 MFMA is two K16 MFMAs of the gfx942 loop side by side: lane l holds A row and
# B column l % 16 and K values (K / 4) * (l >> 4) .. (FlyDSL's MFMA layouts), so the
# K32 operands are the K16 ones concatenated and S and O keep their layouts. V never
# passes through LDS here: packed V is stored in the PV operand order by the pack
# kernel, raw V is transposed in registers (v_perm), so ds_read_b64_tr_b16 does not
# apply. Per tile, with g = lane >> 4 and the two 16-key segments s of BN32:
#   QK: segment s's K32 step j merges K16 steps 2j, 2j + 1: K and Q words 4j .. 4j + 3
#       (dims 32j + 8g .. + 7).
#   PV: lane group g's eight K32 keys are segment 0's keys 4g .. 4g + 3, then segment
#       1's: B is P of both segments (4 words), and A of dims group n is segment 0's
#       K16 operand words of n, then segment 1's.
# The gfx942 loop requests segment 1's V only after segment 0's has landed, and runs
# segment 0's PV meanwhile: two V round trips a tile. A K32 PV MFMA reads both
# segments, so the gfx950 loop splits a tile's V by dims half instead: half 0 of both
# segments before QK, half 1 after it, and K of the next tile before PV, so a tile
# takes one round trip. So that both V halves and the next tile's K fit in the VGPRs
# of two waves a SIMD, Q waits between tiles in LDS that the epilogue uses later
# (_store_q950, reloaded each tile); the block-ID prefetch moves one tile ahead so
# that K's addresses do not wait for V. _Packed.attend950 is that loop.

QK950 = tuple((step, s) for step in range(8) for s in range(2))  # K32 QK, issue order


def _qk_words950(step):
    """K and Q words of K32 QK step `step`: the words of K16 steps 2 * step, + 1."""
    return [step * 4 + i for i in range(4)]


def _pv_words950(n):
    """(key segment, packed V word) of the A operand of K32 PV MFMA n."""
    return [(segment, n * 2 + i) for segment in range(2) for i in range(2)]


def _store_q950(storage, q, lane, base):
    """Q words of the lane into LDS at base: words 4j .. 4j + 3 at base + 1024j + 16 lane."""
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    q = fx.Vector(q)
    for j in range(8):
        _write_address(
            shared + base + lane * 16 + j * 1024,
            fx.Vector.from_elements([q[j * 4 + i] for i in range(4)], fx.Int32),
        )


def _load_q950(storage, lane, base):
    """_store_q950's words back (eight ds_read_b128)."""
    parts = [_load_lds(storage, base + lane * 16, j * 1024) for j in range(8)]
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


def _qk950(q, k0, k1):
    """_qk on gfx950: S of key segments 0 and 1 from K32 MFMAs, in _qk's layout (keys
    0-15's accumulator, then keys 16-31's)."""
    q, keys = fx.Vector(q), (fx.Vector(k0), fx.Vector(k1))
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for step, segment in QK950:
        words = _qk_words950(step)
        acc[segment] = _mfma(
            fx.Vector.from_elements([keys[segment][i] for i in words], fx.Int32),
            fx.Vector.from_elements([q[i] for i in words], fx.Int32),
            acc[segment],
            32,
        )
    return _join(acc[0], acc[1])


def _pv950(p, values, output):
    """_pv on gfx950: O += P V of both key segments. p holds P of both (4 words),
    values the two segments' packed V words of one dims half."""
    prim.s_setprio(2)
    output = fx.Vector(output)
    values = [fx.Vector(value) for value in values]
    acc = []
    for n in range(8):
        c = fx.Vector.from_elements([output[n * 4 + i] for i in range(4)], fx.Float32)
        a = fx.Vector.from_elements(
            [values[segment][word] for segment, word in _pv_words950(n)], fx.Int32
        )
        acc.append(_mfma(a, p, c, 32))
    prim.s_setprio(0)
    return fx.Vector.from_elements(
        [acc[n][i] for n in range(8) for i in range(4)], fx.Float32
    )


class _Packed(_DirectLoop):
    """Packed K/V: each four-token block stored in the QK/PV operand layouts."""

    # Lane 4j holds token 4j's K offset.
    row_lane = 16

    def __init__(
        self, kr, vr, blocks, row, visible, extent, count, lane, hk, topk, scale
    ):
        self.kr, self.vr, self.blocks, self.row = kr, vr, blocks, row
        self.visible, self.extent, self.count, self.lane = visible, extent, count, lane
        self.hk, self.topk, self.scale = hk, topk, scale

    def values(self, row, t, segment, half):
        return _value_load(
            self.vr, row, t, segment, half, self.lane, self.visible, self.hk, self.topk
        )

    def next_ids(self, t):
        chunk = _min((t + 1) >> 3, fx.Int32(self.topk // 64 - 1)) * 64
        return fx.Int32(self.blocks[self.row * self.topk + chunk + self.lane])

    def key_operands(self, k0, k1):
        return k0, k1

    def scores(self, s, t):
        s = fx.Vector(s)
        first, second = _scale_scores(
            fx.Vector.from_elements([s[i] for i in range(4)], fx.Float32),
            fx.Vector.from_elements([s[4 + i] for i in range(4)], fx.Float32),
            self.count,
            t,
            self.lane,
            self.scale,
        )
        return [first[i] for i in range(4)] + [second[i] for i in range(4)]

    def reduce(self, value, maximum):
        return _reduce(value, maximum)

    def rescale(self, o0, o1, alpha):
        changed = prim.ballot(alpha != fx.Float32(1.0))
        return _rescale((o0, o1), changed, lambda: alpha)

    def key_offset(self, ids, position):
        return _key_offset(
            ids,
            position + (self.lane & 15),
            self.visible,
            self.extent,
            self.hk,
            self.topk,
        )

    def load_key(self, offset):
        return _key_load(self.kr, offset, self.lane, self.hk)

    @flyc.jit
    def attend950(
        self, q, k0, k1, offset0, offset1, ids, maximum, total, o0, o1, tiles, storage
    ):
        """attend on gfx950 (module comment's gfx950 part); storage holds Q between
        tiles."""
        next_ids = ids
        lane, count = self.lane, self.count
        _store_q950(storage, q, lane, 0)
        for t in range(fx.Int32(0), tiles, fx.Int32(1)):
            rows = [
                fx.gpu.shuffle_idx(fx.Int32(offset), (lane >> 4) << 2, 64)
                for offset in (offset0, offset1)
            ]
            rocdl.s_waitcnt(lgkmcnt=0)
            prim.sched_barrier(0)
            # Switch to the block IDs prefetched a tile ago here, where they have
            # landed: a switch after this tile's prefetch would make LLVM wait for it.
            if ((t + 1) & 7) == 0:
                ids = next_ids
            values0 = [_value_words(self.vr, row, 0, lane, self.hk) for row in rows]
            q_words = _load_q950(storage, lane, 0)
            prim.sched_barrier(0)
            # Only this tile's K, requested a tile ago, is older than the eight V
            # loads of dims half 0: vmcnt(8) waits for K, leaving V in flight.
            rocdl.s_waitcnt(vmcnt=8)
            prim.sched_barrier(0)
            s = _qk950(q_words, k0, k1)
            prim.sched_barrier(0)
            values1 = [_value_words(self.vr, row, 1, lane, self.hk) for row in rows]
            if ((t + 2) & 7) == 0:
                next_ids = self.next_ids(t + 1)
            prim.sched_barrier(0)
            updated, probabilities, total, o0, o1 = self.softmax(
                s, t, maximum, total, o0, o1
            )
            p = _pack_bf16_950(probabilities)
            prim.sched_barrier(0)
            following0 = _min((t + 1) * 32, (((count + 15) >> 4) - 1) * 16)
            following1 = _min((t + 1) * 32 + 16, (((count + 15) >> 4) - 1) * 16)
            offset0 = self.key_offset(ids, following0)
            offset1 = self.key_offset(ids, following1)
            k0 = self.load_key(offset0)
            k1 = self.load_key(offset1)
            prim.sched_barrier(0)
            # V half 0 is the oldest; younger are half 1, a global block-ID prefetch
            # (every eighth tile) and the sixteen K loads: vmcnt(24) waits for half 0.
            rocdl.s_waitcnt(vmcnt=24)
            prim.sched_barrier(0)
            values0 = [
                _mask_value_tail(value, t, segment, lane, self.visible, self.topk)
                for segment, value in enumerate(values0)
            ]
            o0 = _pv950(p, values0, o0)
            prim.sched_barrier(0)
            # Only K's sixteen loads are younger than half 1 and the block IDs.
            rocdl.s_waitcnt(vmcnt=16)
            prim.sched_barrier(0)
            values1 = [
                _mask_value_tail(value, t, segment, lane, self.visible, self.topk)
                for segment, value in enumerate(values1)
            ]
            o1 = _pv950(p, values1, o1)
            maximum = updated
        return maximum, total, o0, o1

    def pv(self, p, values, output):
        prim.s_setprio(2)
        values = fx.Vector(values)
        acc = _pv_step(
            p,
            lambda n: fx.Vector.from_elements(
                [values[n * 2 + i] for i in range(2)], fx.Int32
            ).bitcast(fx.Int16),
            _tiles(output),
        )
        prim.s_setprio(0)
        return _untile(acc)


@flyc.jit
def _body(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    tile,
    hkv,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    SCALE_LOG2: fx.Float32,
    TOPK: fx.Constexpr[int],
    GFX950: fx.Constexpr[bool],
):
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, 8192, 16])
        .peek()
        .view(fx.make_layout(8192, 1))
    )
    tid = fx.thread_idx.x
    wave, lane = _uniform(tid >> 6), tid & 63
    first, rows = _uniform(META[tile * 5]), _uniform(META[tile * 5 + 1])
    k0, kv_len = _uniform(META[tile * 5 + 2]), _uniform(META[tile * 5 + 3])
    position0 = _uniform(META[tile * 5 + 4])
    valid_wave = _enabled(wave, rows, first, None, None, False, NQ)
    visible = position0 + wave + 1
    count = _min(visible >> 2, fx.Int32(TOPK)) * 4 + (visible & 3)
    tiles = valid_wave.select((count + 31) >> 5, fx.Int32(0))
    qe = fx.Uint32(NQ - first) * H * 512 - hkv * (H // HK) * 512
    qptr = fx.get_iter(Q) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    qr = _buffer(fx.make_view(qptr, fx.make_layout(NQ * H * 256, 1)), qe)
    qoffset = (valid_wave & ((lane & 15) < H // HK)).select(
        fx.Uint32((wave * H + (lane & 15)) * 512), qe
    )
    q = _read_q_g2r(qr, qoffset, lane)
    extent = (fx.Uint32(kv_len) * HK - hkv) * 512
    kr = _buffer(
        fx.make_view(
            fx.get_iter(K) + (fx.Int64(k0) * HK + hkv) * 256,
            fx.make_layout(extent // 2, 1),
        ),
        extent,
    )
    vr = _buffer(
        fx.make_view(
            fx.get_iter(V) + (fx.Int64(k0) * HK + hkv) * 256,
            fx.make_layout(extent // 2, 1),
        ),
        extent,
    )
    row = _min(first + wave, fx.Int32(NQ - 1))
    blocks = rocdl.make_buffer_tensor(BLOCKS)
    maximum, total = fx.Float32(_MAX_FLOOR), fx.Float32(0.0)
    o0, o1 = _pin(fx.Vector.filled(32, 0.0, fx.Float32)), _pin(
        fx.Vector.filled(32, 0.0, fx.Float32)
    )
    cached = fx.Int32(blocks[row * TOPK + lane])
    rocdl.s_waitcnt(vmcnt=0)
    offset0 = _key_offset(cached, lane & 15, visible, extent, HK, TOPK)
    offset0 = valid_wave.select(offset0, extent)
    kval0 = _key_load(kr, offset0, lane, HK)
    offset1 = _key_offset(
        cached,
        _min(fx.Int32(16), (((count + 15) >> 4) - 1) * 16) + (lane & 15),
        visible,
        extent,
        HK,
        TOPK,
    )
    offset1 = valid_wave.select(offset1, extent)
    kval1 = _key_load(kr, offset1, lane, HK)
    rocdl.s_waitcnt(vmcnt=0)
    prim.sched_barrier(0)
    path = _Packed(
        kr, vr, blocks, row, visible, extent, count, lane, HK, TOPK, SCALE_LOG2
    )
    if fx.const_expr(GFX950):
        maximum, total, o0, o1 = path.attend950(
            q,
            kval0,
            kval1,
            offset0,
            offset1,
            cached,
            maximum,
            total,
            o0,
            o1,
            tiles,
            storage,
        )
    else:
        maximum, total, o0, o1 = path.attend(
            q, kval0, kval1, offset0, offset1, cached, maximum, total, o0, o1, tiles
        )
    rocdl.s_waitcnt(vmcnt=0)
    _stage_end()
    inv = (total > 0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    outptr = fx.get_iter(OUT) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    output = rocdl.make_buffer_tensor(
        fx.make_view(outptr, fx.make_layout(NQ * H * 256, 1)), num_records_bytes=qe
    )
    _output(
        o0,
        o1,
        inv,
        output,
        storage,
        tid,
        H,
        H // HK,
        rows,
        qe,
        64,
        first,
        None,
        None,
        False,
        NQ,
        GFX950,
    )


@flyc.kernel(name="sparse_prefill_direct_packed_bf16_d256")
def _kernel(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    UNION_META: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    GATED: fx.Constexpr[bool],
    SCALE_LOG2: fx.Float32,
    TOPK: fx.Constexpr[int],
    GFX950: fx.Constexpr[bool],
):
    task = fx.block_idx.x
    if fx.const_expr(GATED):
        # Grid (row * HK + head, union tile): one round of independent loads gates a slot.
        tile, local = fx.block_idx.y, task // HK
        first, rows = _uniform(UNION_META[tile * 5]), _uniform(UNION_META[tile * 5 + 1])
        if (_uniform(ACTIVE[tile]) == 0) & (local < rows):
            _body(
                Q,
                K,
                V,
                OUT,
                BLOCKS,
                META,
                first + local,
                task % HK,
                H,
                HK,
                NQ,
                SCALE_LOG2,
                TOPK,
                GFX950,
            )
    else:
        _body(
            Q,
            K,
            V,
            OUT,
            BLOCKS,
            META,
            task // HK,
            task % HK,
            H,
            HK,
            NQ,
            SCALE_LOG2,
            TOPK,
            GFX950,
        )


@flyc.jit
def _launch(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    SLOTS: fx.Tensor,
    SCALE_LOG2: fx.Float32,
    PK: fx.Tensor,
    PV: fx.Tensor,
    SOURCES: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    UNION_META: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    NK: fx.Int32,
    PACKED: fx.Int32,
    BQ: fx.Int32,
    TILES: fx.Int32,
    TASKS: fx.Int32,
    GATED: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
    SLOTTED: fx.Constexpr[bool],
    GFX950: fx.Constexpr[bool],
    KV_STRIDE: fx.Constexpr[int],
    stream: fx.Stream,
):
    if TASKS > 0:
        if fx.const_expr(GATED):
            _pack_gated(
                K,
                V,
                PK,
                PV,
                SOURCES,
                SLOTS,
                DIRECT_FLAG,
                NK,
                PACKED,
                HK,
                SLOTTED,
                KV_STRIDE,
            ).launch(
                grid=((PACKED * HK + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream
            )
        else:
            _pack(
                K, V, PK, PV, SOURCES, SLOTS, NK, PACKED, HK, SLOTTED, KV_STRIDE
            ).launch(
                grid=((PACKED * HK + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream
            )
        kernel = _kernel(
            Q,
            PK,
            PV,
            OUT,
            BLOCKS,
            META,
            ACTIVE,
            UNION_META,
            H,
            HK,
            NQ,
            GATED,
            SCALE_LOG2,
            TOPK,
            GFX950,
            value_attrs=prim.kernel_attrs(),
        )
        if fx.const_expr(GATED):
            kernel.launch(grid=(BQ * HK, TILES, 1), block=(64, 1, 1), stream=stream)
        else:
            kernel.launch(grid=(TASKS, 1, 1), block=(64, 1, 1), stream=stream)


def launch_args(inputs, prepared, out, tasks, stream):
    # A raw plan compiles this path ahead of time with its public KV as stand-ins. The fifth
    # argument is the slot table (a stand-in without one), the sixth the FP32
    # softmax_scale * log2(e).
    packed_key = inputs.k if prepared.packed_key is None else prepared.packed_key
    packed_value = inputs.v if prepared.packed_value is None else prepared.packed_value
    sources = (
        prepared.metadata.view(-1)
        if prepared.pack_sources is None
        else prepared.pack_sources
    )
    slots = inputs.kv_slots if prepared.slotted else prepared.metadata.view(-1)
    return (
        inputs.q.view(-1),
        inputs.k,
        inputs.v,
        out.view(-1),
        slots,
        inputs.scale_log2,
        packed_key,
        packed_value,
        sources,
        prepared.source_blocks.view(-1),
        prepared.metadata.view(-1),
        prepared.active,
        prepared.union_meta.view(-1),
        prepared.direct_flag,
        inputs.q.shape[1],
        inputs.k.shape[1],
        inputs.q.shape[0],
        inputs.k.shape[0],
        packed_key.shape[0] // 4,
        prepared.union_tile,
        prepared.union_tiles,
        tasks,
        prepared.gated,
        inputs.topk,
        prepared.slotted,
        prepared.k32,
        inputs.kv_stride,
        stream,
    )


def compile_launch(inputs, prepared, out, stream):
    return flyc.compile(_launch, *launch_args(inputs, prepared, out, 0, stream))

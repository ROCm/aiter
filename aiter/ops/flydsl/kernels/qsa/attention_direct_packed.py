# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-call block-packed KV for QSA direct attention, packing included in the launch.

Four-token KV blocks are stored in the native QK/PV operand layouts. One query
wave then needs no K-data bpermutes or PV byte permutations. Original public
KV tensors remain unchanged; private scratch is refreshed on every call.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl

from .attention_common import (
    _MAX_FLOOR,
    _buffer,
    _buffer_store_words,
    _buffer_words,
    _exp,
    _load_lds,
    _maximum,
    _mfma,
    _min,
    _pack_bf16,
    _pin,
    _stage_end,
    _target_features,
    _uniform,
    _write_address,
)


def _load(resource, offset, lane):
    parts = [
        _buffer_words(resource, offset + (lane >> 4) * 16 + k * 64) for k in range(8)
    ]
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


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
):
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    row = (tid >> 6) * 16 + (tid & 15)
    for half in range(2):
        acc = o1 if half else o0
        for i in range(4):
            values = fx.Vector.from_elements(
                [acc[n * 4 + i] * inv for n in range(8)], fx.Float32
            )
            words = _pack_bf16(values)
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
    rocdl.sched_barrier(0)
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
    kr, vr, pk, pv, sources, pair, half, lane, packed, hk, source_extent, extent
):
    block, head = pair // hk, pair % hk
    valid = pair < packed * hk
    start = fx.Int32(sources[_min(block, packed - 1)])
    token, chunk = lane >> 4, lane & 15
    dim = half * 128 + chunk * 8
    # Tokens past a request's end are masked by the direct kernel.
    source = valid.select(
        fx.Uint32((start + token) * hk + head) * 512 + dim * 2, source_extent
    )
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
            received = fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, source_lane, sent))
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
def _pack_body(K, V, PK, PV, SOURCES, N, PACKED, HK):
    tid = fx.thread_idx.x
    lane, half = tid & 63, (tid >> 6) & 1
    first = fx.block_idx.x * 8 + (tid >> 7)
    source_extent = fx.Uint32(N) * HK * 512
    extent = fx.Uint32(PACKED) * 4 * HK * 512
    kr, vr = _buffer(K, source_extent), _buffer(V, source_extent)
    pk, pv = _buffer(PK, extent), _buffer(PV, extent)
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
            source_extent,
            extent,
        )


@flyc.kernel(name="attention_pack_kv_bf16_d256")
def _pack(
    K: fx.Tensor,
    V: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    SOURCES: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    HK: fx.Constexpr[int],
):
    _pack_body(K, V, PK, PV, SOURCES, N, PACKED, HK)


@flyc.kernel(name="attention_pack_kv_bf16_d256")
def _pack_gated(
    K: fx.Tensor,
    V: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    SOURCES: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    HK: fx.Constexpr[int],
):
    if _uniform(DIRECT_FLAG[0]) != 0:
        _pack_body(K, V, PK, PV, SOURCES, N, PACKED, HK)


def _key_offset(cached, n, visible, extent, hk, topk):
    complete = _min(visible >> 2, fx.Int32(topk))
    width = complete * 4 + (visible & 3)
    selector = fx.Int32((_min(n >> 2, fx.Int32(topk - 1)) & 63) * 4)
    block = fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, selector, fx.Int32(cached)))
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


def _value_load(resource, source_base, tile, segment, half, lane, visible, hk, topk):
    base = fx.Uint32(source_base) + half * 2 * hk * 512 + (lane & 15) * 16
    parts = [
        _buffer_words(resource, base + (part // 2) * hk * 512 + (part % 2) * 256)
        for part in range(4)
    ]
    value = fx.Vector.from_elements(
        [part[i] for part in parts for i in range(4)], fx.Int32
    )
    # A packed block may contain later tokens with NaNs; zero those BF16
    # elements before MFMA rather than relying on zero probabilities.
    return _mask_value_tail(value, tile, segment, lane, visible, topk)


def _qk(q, k0, k1):
    q = fx.Vector(q).bitcast(fx.Int16)
    keys = (fx.Vector(k0).bitcast(fx.Int16), fx.Vector(k1).bitcast(fx.Int16))
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for step in range(16):
        b = fx.Vector.from_elements([q[step * 4 + i] for i in range(4)], fx.Int16)
        for n in range(2):
            a = fx.Vector.from_elements(
                [keys[n][step * 4 + i] for i in range(4)], fx.Int16
            )
            acc[n] = _mfma(a, b, acc[n])
    return acc[0], acc[1]


def _pv(p, values, output):
    rocdl.s_setprio(2)
    values, output = fx.Vector(values), fx.Vector(output)
    acc = []
    for n in range(8):
        c = fx.Vector.from_elements([output[n * 4 + i] for i in range(4)], fx.Float32)
        a = fx.Vector.from_elements(
            [values[n * 2 + i] for i in range(2)], fx.Int32
        ).bitcast(fx.Int16)
        acc.append(_mfma(a, p, c))
    rocdl.s_setprio(0)
    return fx.Vector.from_elements(
        [acc[n][i] for n in range(8) for i in range(4)], fx.Float32
    )


def _reduce(value, maximum):
    a = value.shuffle_xor(16, 64)
    b = value.shuffle_xor(32, 64)
    c = value.shuffle_xor(48, 64)
    rocdl.sched_barrier(0)
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


@flyc.jit
def _rescale(output0, output1, alpha):
    output0, output1 = fx.Vector(output0), fx.Vector(output1)
    changed = rocdl.ballot(fx.Int64.ir_type, alpha != fx.Float32(1.0))
    if fx.Int64(changed) != 0:
        output0 = output0 * alpha
        output1 = output1 * alpha
    return output0, output1


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
    SCALE: fx.Constexpr[float],
    TOPK: fx.Constexpr[int],
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
    q = _load(qr, qoffset, lane)
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
    scale = fx.Float32(SCALE * math.log2(math.e))
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
    rocdl.sched_barrier(0)
    next_cached = cached
    for t in range(fx.Int32(0), tiles, fx.Int32(1)):
        vrow0 = fx.Int32(
            rocdl.ds_bpermute(fx.Int32.ir_type, (lane >> 4) * 16, fx.Int32(offset0))
        )
        vrow1 = fx.Int32(
            rocdl.ds_bpermute(fx.Int32.ir_type, (lane >> 4) * 16, fx.Int32(offset1))
        )
        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.sched_barrier(0)
        values0 = _value_load(vr, vrow0, t, 0, 0, lane, visible, HK, TOPK)
        values1 = _value_load(vr, vrow0, t, 0, 1, lane, visible, HK, TOPK)
        if ((t + 1) & 7) == 0:
            chunk = _min((t + 1) >> 3, fx.Int32(TOPK // 64 - 1)) * 64
            next_cached = fx.Int32(blocks[row * TOPK + chunk + lane])
        rocdl.sched_barrier(0)
        # The next index-cache load is younger than V0. vmcnt(8) is
        # conservative on those boundary iterations and still releases K.
        rocdl.s_waitcnt(vmcnt=8)
        rocdl.sched_barrier(0)
        scores0, scores1 = _qk(q, kval0, kval1)
        scaled0, scaled1 = _scale_scores(scores0, scores1, count, t, lane, scale)
        scores = [scaled0[i] for i in range(4)] + [scaled1[i] for i in range(4)]
        candidate = fx.Float32(_MAX_FLOOR)
        for score in scores:
            candidate = _maximum(candidate, score)
        candidate = _reduce(candidate, True)
        updated = _maximum(maximum, candidate)
        alpha = _exp(maximum - updated)
        probabilities = fx.Vector.from_elements(
            [_exp(value - updated) for value in scores], fx.Float32
        )
        current = fx.Float32(0.0)
        for i in fx.range_constexpr(8):
            current = current + probabilities[i]
        current = _reduce(current, False)
        total = total * alpha + current
        o0, o1 = _rescale(o0, o1, alpha)
        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(vmcnt=0)
        rocdl.sched_barrier(0)
        if ((t + 1) & 7) == 0:
            cached = next_cached
        following0 = _min((t + 1) * 32, (((count + 15) >> 4) - 1) * 16)
        following1 = _min((t + 1) * 32 + 16, (((count + 15) >> 4) - 1) * 16)
        offset0 = _key_offset(
            cached, following0 + (lane & 15), visible, extent, HK, TOPK
        )
        offset1 = _key_offset(
            cached, following1 + (lane & 15), visible, extent, HK, TOPK
        )
        p0 = _pack_bf16(
            fx.Vector.from_elements([probabilities[i] for i in range(4)], fx.Float32)
        ).bitcast(fx.Int16)
        p1 = _pack_bf16(
            fx.Vector.from_elements(
                [probabilities[4 + i] for i in range(4)], fx.Float32
            )
        ).bitcast(fx.Int16)
        next_values0 = _value_load(vr, vrow1, t, 1, 0, lane, visible, HK, TOPK)
        next_values1 = _value_load(vr, vrow1, t, 1, 1, lane, visible, HK, TOPK)
        rocdl.sched_barrier(0)
        o0 = _pv(p0, values0, o0)
        rocdl.sched_barrier(0)
        kval0 = _key_load(kr, offset0, lane, HK)
        rocdl.sched_barrier(0)
        o1 = _pv(p0, values1, o1)
        rocdl.sched_barrier(0)
        kval1 = _key_load(kr, offset1, lane, HK)
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(vmcnt=16)
        rocdl.sched_barrier(0)
        o0 = _pv(p1, next_values0, o0)
        o1 = _pv(p1, next_values1, o1)
        maximum = updated
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
    )


@flyc.kernel(name="attention_direct_bf16_d256")
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
    SCALE: fx.Constexpr[float],
    TOPK: fx.Constexpr[int],
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
                SCALE,
                TOPK,
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
            SCALE,
            TOPK,
        )


@flyc.jit
def _launch(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
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
    SCALE: fx.Constexpr[float],
    TOPK: fx.Constexpr[int],
    stream: fx.Stream,
):
    if TASKS > 0:
        if fx.const_expr(GATED):
            _pack_gated(K, V, PK, PV, SOURCES, DIRECT_FLAG, NK, PACKED, HK).launch(
                grid=((PACKED * HK + 7) // 8, 1, 1), block=(256, 1, 1), stream=stream
            )
        else:
            _pack(K, V, PK, PV, SOURCES, NK, PACKED, HK).launch(
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
            SCALE,
            TOPK,
            value_attrs={"llvm.target_features": _target_features()},
        )
        if fx.const_expr(GATED):
            kernel.launch(grid=(BQ * HK, TILES, 1), block=(64, 1, 1), stream=stream)
        else:
            kernel.launch(grid=(TASKS, 1, 1), block=(64, 1, 1), stream=stream)


def launch_args(inputs, prepared, out, tasks, stream):
    # A raw plan compiles this path ahead of time with its public KV as stand-ins.
    packed_key = inputs.k if prepared.packed_key is None else prepared.packed_key
    packed_value = inputs.v if prepared.packed_value is None else prepared.packed_value
    sources = (
        prepared.metadata.view(-1)
        if prepared.pack_sources is None
        else prepared.pack_sources
    )
    return (
        inputs.q.view(-1),
        inputs.k,
        inputs.v,
        out.view(-1),
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
        inputs.scale,
        inputs.topk,
        stream,
    )


def compile_launch(inputs, prepared, out, stream):
    return flyc.compile(_launch, *launch_args(inputs, prepared, out, 0, stream))

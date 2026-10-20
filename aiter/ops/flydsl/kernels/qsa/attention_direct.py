# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Block-native gfx942 QSA sparse attention, without cross-query union construction.

Layouts whose packed PK/PV scratch fits the platform's packed-KV budget use the
private packed implementation: every call packs each request's KV into
four-token blocks and runs one query wave per CTA. Over-budget KV keeps the
raw KV fallback below, where packing would cost more than it saves. Both paths
preserve the public inputs and selection.

In the raw fallback, four query waves each own one query and its GQA heads.
V loads directly from global memory into registers for the transpose; Q/K
remain in registers. Only the output transpose uses LDS (32 KiB). Softmax
is updated once per BN tokens. Sparse addresses are full byte VOFFSETs.
The shared token recovery already supplies sorted block indices.
Each wave caches 64 block IDs for eight BN32 steps. K loads group four
adjacent lanes per token, then transpose at QK consumption. K addresses are
reused for V; the first V segment is requested before QK. K completion is
waited at the next QK consumer so PV can overlap its prefetch. Accumulation
order and RNE probability packing are unchanged.
"""

import math
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch
from flydsl._mlir import ir
from flydsl.expr import rocdl

from . import attention_direct_packed as packed
from .attention_common import (
    _buffer,
    _buffer_words,
    _exp,
    _maximum,
    _mfma,
    _min,
    _pack_bf16,
    _pin,
    _stage_end,
    _uniform,
    platform,
)
from .attention_direct_packed import _enabled, _load, _output
from .attention_prepare import allocate_scratch, upload


def _load_key(resource, offset, lane):
    # Adjacent four lanes read one token's contiguous 64B, not four tokens.
    parts = [
        _buffer_words(resource, offset + (lane & 3) * 16 + k * 64) for k in range(8)
    ]
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


def _qk_pair(q, k0, k1, lane):
    q = fx.Vector(q).bitcast(fx.Int16)
    keys = (fx.Vector(k0), fx.Vector(k1))
    # Inverse of load lane = token * 4 + channel_quarter. Keep K in its load
    # layout during PV prefetch so the transpose cannot force an early VM wait.
    source_lane = fx.Int32(((lane & 15) * 4 + (lane >> 4)) * 4)
    keys = tuple(
        fx.Vector.from_elements(
            [
                fx.Int32(rocdl.ds_bpermute(fx.Int32.ir_type, source_lane, key[i]))
                for i in range(32)
            ],
            fx.Int32,
        ).bitcast(fx.Int16)
        for key in keys
    )
    acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for s in range(16):
        first = (s // 2) * 8 + (s % 2) * 4
        b = fx.Vector.from_elements([q[first + i] for i in range(4)], fx.Int16)
        for n in range(2):
            a = fx.Vector.from_elements(
                [keys[n][first + i] for i in range(4)], fx.Int16
            )
            acc[n] = _mfma(a, b, acc[n])
    return acc[0], acc[1]


def _pv(p, values, output, alpha):
    output = fx.Vector(output)
    acc = [
        fx.Vector.from_elements(
            [output[n * 4 + i] * alpha for i in range(4)], fx.Float32
        )
        for n in range(8)
    ]
    for n in range(8):
        selector = 0x03020706 if n % 2 else 0x01000504
        words = [
            fx.Int32(
                rocdl.perm_b32(
                    values[r * 4 + n // 2], values[(r + 1) * 4 + n // 2], selector
                )
            )
            for r in (0, 2)
        ]
        a = fx.Vector.from_elements(words, fx.Int32).bitcast(fx.Int16)
        acc[n] = _mfma(a, p, acc[n])
    result = fx.Vector.from_elements(
        [acc[n][i] for n in range(8) for i in range(4)], fx.Float32
    )
    return result


@dataclass(kw_only=True)
class DirectPlan:
    metadata: torch.Tensor
    query_tile: int
    num_tiles: int
    active: torch.Tensor
    query_tiles: torch.Tensor
    gated: bool
    source_blocks: torch.Tensor | None
    # Gated launches: K3's any-direct flag and the union tiles (stand-ins when ungated).
    direct_flag: torch.Tensor
    union_meta: torch.Tensor
    union_tile: int = 1
    union_tiles: int = 0
    packed_key: torch.Tensor | None = None
    packed_value: torch.Tensor | None = None
    pack_sources: torch.Tensor | None = None
    pack_shape: tuple = ()
    pack_dtype: torch.dtype | None = None


def scratch_specs(prepared: DirectPlan) -> list:
    if not prepared.pack_shape:
        return []
    return [
        (name, prepared.pack_shape, prepared.pack_dtype, False)
        for name in ("packed_key", "packed_value")
    ]


def prepare(
    *,
    inputs,
    union=None,
    scratch: bool = True,
):
    """Bind private scratch; attention's recovery refreshes sorted block_indices each call."""
    # Each request with queries packs its KV into whole four-token blocks at a
    # four-token-aligned scratch base. Over-budget KV keeps the raw-KV path.
    hk = inputs.k.shape[1]
    q_lens = np.asarray(inputs.query_lens, dtype=np.int64)
    lengths = q_lens + np.asarray(inputs.prefix_lens, dtype=np.int64)
    blocks = np.where(q_lens > 0, -(-lengths // 4), 0)
    # The pack kernel addresses all of K/V and PK/PV with unsigned 32-bit byte offsets.
    packed = (
        int(blocks.sum()) * 4 * hk * 1024 <= platform().max_packed_kv_bytes
        and max(int(lengths.sum()), int(blocks.sum()) * 4) * hk * 512 < 2**32
    )
    waves = 1 if packed else 4
    k_starts, p_starts = np.cumsum(lengths) - lengths, np.cumsum(blocks) - blocks
    counts = -(-q_lens // waves)
    request = np.repeat(np.arange(len(q_lens)), counts)
    local = (
        np.arange(len(request)) - np.repeat(np.cumsum(counts) - counts, counts)
    ) * waves
    base, extent = (p_starts * 4, blocks * 4) if packed else (k_starts, lengths)
    metadata = np.stack(
        (
            (np.cumsum(q_lens) - q_lens)[request] + local,
            np.minimum(waves, q_lens[request] - local),
            base[request],
            extent[request],
            (lengths - q_lens)[request] + local,
        ),
        axis=1,
    )
    device = inputs.q.device
    sources = None
    if packed:
        owner = np.repeat(np.arange(len(blocks)), blocks)
        sources = upload(
            k_starts[owner] + 4 * (np.arange(len(owner)) - p_starts[owner]), device
        )
    standin = inputs.query_positions  # tensor slots ungated launches do not read
    prepared = DirectPlan(
        metadata=upload(metadata, device),
        query_tile=waves,
        num_tiles=len(metadata),
        gated=union is not None,
        active=standin if union is None else union.active,
        query_tiles=standin if union is None else union.query_tiles,
        direct_flag=standin if union is None else union.direct_flag,
        union_meta=standin if union is None else union.metadata,
        union_tile=1 if union is None else union.query_tile,
        union_tiles=0 if union is None else union.num_tiles,
        source_blocks=inputs.block_indices,
        pack_sources=sources,
        pack_shape=(int(blocks.sum()) * 4, hk, 256) if packed else (),
        pack_dtype=inputs.k.dtype,
    )
    if scratch:
        allocate_scratch(prepared, scratch_specs(prepared), device)
    return prepared


def _source_offset(cached_blocks, n, visible, extent, hk, topk):
    # Valid positions and token slots are nonnegative; padding is masked below.
    complete = _min(visible >> 2, fx.Int32(topk))
    width = complete * 4 + (visible & 3)
    lane_bytes = fx.Int32((_min(n >> 2, fx.Int32(topk - 1)) & 63) * 4)
    block = fx.Int32(
        rocdl.ds_bpermute(fx.Int32.ir_type, lane_bytes, fx.Int32(cached_blocks))
    )
    token = (n < complete * 4).select(
        block * 4 + (n & 3), (visible & -4) + n - complete * 4
    )
    return (n < width).select(fx.Uint32(token) * (hk * 256 * 2), extent)


def _load_v_global(
    resource, source_base, tile, segment, half, lane, visible, extent, hk, bn, topk
):
    first = tile * bn + segment * 16 + (lane >> 4) * 4
    complete = _min(visible >> 2, fx.Int32(topk))
    width = complete * 4 + (visible & 3)
    base = fx.Uint32(source_base) + (lane & 15) * 16 + half * 256
    parts = []
    for r in range(4):
        offset = (first + r < width).select(base + r * (hk * 256 * 2), extent)
        parts.append(_buffer_words(resource, offset))
    return fx.Vector.from_elements([p[i] for p in parts for i in range(4)], fx.Int32)


@flyc.jit
def _body(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    BN: fx.Constexpr[int],
    THREADS: fx.Constexpr[int],
    TASKS: fx.Int32,
    GATED: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
    TOPK: fx.Constexpr[int],
):
    load, load_key, qk, pv = _load, _load_key, _qk_pair, _pv
    source_offset, load_v, output_fn, enabled = (
        _source_offset,
        _load_v_global,
        _output,
        _enabled,
    )
    storage = (
        fx.SharedAllocator()
        .allocate(fx.Array[fx.Int8, 32768, 16])
        .peek()
        .view(fx.make_layout(32768, 1))
    )
    shared = fx.Int32(fx.ptrtoint(fx.get_iter(storage)))
    tid = fx.thread_idx.x
    wave, lane = _uniform(tid >> 6), tid & 63
    task = fx.block_idx.x
    tile, hkv = task // HK, task % HK
    first, rows = _uniform(META[tile * 5]), _uniform(META[tile * 5 + 1])
    k0, kv_len = _uniform(META[tile * 5 + 2]), _uniform(META[tile * 5 + 3])
    position0 = _uniform(META[tile * 5 + 4])
    valid_wave = enabled(wave, rows, first, QUERY_TILES, ACTIVE, GATED, NQ)
    visible = position0 + wave + 1
    count = _min(visible >> 2, fx.Int32(TOPK)) * 4 + (visible & 3)
    tiles = valid_wave.select((count + 31) >> 5, fx.Int32(0))
    qe = fx.Uint32(NQ - first) * H * 256 * 2 - hkv * (H // HK) * 256 * 2
    qptr = fx.get_iter(Q) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    qr = _buffer(fx.make_view(qptr, fx.make_layout(NQ * H * 256, 1)), qe)
    qoffset = (valid_wave & ((lane & 15) < H // HK)).select(
        fx.Uint32((wave * H + (lane & 15)) * 512), qe
    )
    q = load(qr, qoffset, lane)
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
    maximum, total = fx.Float32(-1e30), fx.Float32(0.0)
    o0, o1 = _pin(fx.Vector.filled(32, 0.0, fx.Float32)), _pin(
        fx.Vector.filled(32, 0.0, fx.Float32)
    )
    scale = fx.Float32(SCALE * math.log2(math.e))
    # One block ID per lane; each BN32 step consumes eight of these IDs.
    cached_blocks = fx.Int32(blocks[row * TOPK + lane])
    rocdl.s_waitcnt(vmcnt=0)
    initial = source_offset(cached_blocks, lane >> 2, visible, extent, HK, TOPK)
    initial = valid_wave.select(initial, extent)
    kval = load_key(kr, initial, lane)
    key_offset = initial
    key_offset1 = source_offset(
        cached_blocks,
        _min(fx.Int32(16), (((count + 15) >> 4) - 1) * 16) + (lane >> 2),
        visible,
        extent,
        HK,
        TOPK,
    )
    key_offset1 = valid_wave.select(key_offset1, extent)
    kval1 = load_key(kr, key_offset1, lane)
    rocdl.s_waitcnt(vmcnt=0)
    rocdl.sched_barrier(0)
    next_blocks = cached_blocks
    for t in range(fx.Int32(0), tiles, fx.Int32(1)):
        kval = fx.Vector(kval)
        all_scores = []
        v_rows = []
        for segment in fx.range_constexpr(BN // 16):
            # In K's load layout, lanes 0/16/32/48 hold block-first addresses.
            # Reuse these for V without another dependent block-table load.
            v_rows.append(
                fx.Int32(
                    rocdl.ds_bpermute(
                        fx.Int32.ir_type,
                        (lane >> 4) * 64,
                        fx.Int32(key_offset1 if segment else key_offset),
                    )
                )
            )
        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.sched_barrier(0)
        values0 = load_v(vr, v_rows[0], t, 0, 0, lane, visible, extent, HK, BN, TOPK)
        values1 = load_v(vr, v_rows[0], t, 0, 1, lane, visible, extent, HK, BN, TOPK)
        if ((t + 1) & 7) == 0:
            chunk = _min((t + 1) >> 3, fx.Int32(TOPK // 64 - 1)) * 64
            next_blocks = fx.Int32(blocks[row * TOPK + chunk + lane])
        rocdl.sched_barrier(0)
        # The prior iteration's sixteen K requests precede these eight V0
        # requests. Wait for K at its consumer, leaving V in flight for QK.
        # A block-ID prefetch is younger still; vmcnt(8) then also waits for
        # the oldest V request on those boundary iterations.
        rocdl.s_waitcnt(vmcnt=8)
        rocdl.sched_barrier(0)
        pair = qk(q, kval, kval1, lane)
        for segment in fx.range_constexpr(BN // 16):
            scores = pair[segment]
            for i in fx.range_constexpr(4):
                col = t * BN + segment * 16 + (lane >> 4) * 4 + i
                all_scores.append(
                    (col < count).select(scores[i] * scale, fx.Float32(float("-inf")))
                )
        candidate = fx.Float32(-1e30)
        for score in all_scores:
            candidate = _maximum(candidate, score)
        candidate = _maximum(candidate, candidate.shuffle_xor(16, 64))
        candidate = _maximum(candidate, candidate.shuffle_xor(32, 64))
        updated = _maximum(maximum, candidate)
        alpha = _exp(maximum - updated)
        probabilities = fx.Vector.from_elements(
            [_exp(v - updated) for v in all_scores], fx.Float32
        )
        current = fx.Float32(0.0)
        for i in fx.range_constexpr(BN // 4):
            current = current + probabilities[i]
        current = current + current.shuffle_xor(16, 64)
        current = current + current.shuffle_xor(32, 64)
        total = total * alpha + current
        o0 = fx.Vector(o0) * alpha
        o1 = fx.Vector(o1) * alpha
        rocdl.s_waitcnt(lgkmcnt=0)
        rocdl.sched_barrier(0)
        rocdl.s_waitcnt(vmcnt=0)
        rocdl.sched_barrier(0)
        if ((t + 1) & 7) == 0:
            cached_blocks = next_blocks
        following = _min((t + 1) * BN, (((count + 15) >> 4) - 1) * 16)
        following1 = _min((t + 1) * BN + 16, (((count + 15) >> 4) - 1) * 16)
        key_offset = source_offset(
            cached_blocks, following + (lane >> 2), visible, extent, HK, TOPK
        )
        key_offset1 = source_offset(
            cached_blocks, following1 + (lane >> 2), visible, extent, HK, TOPK
        )
        p0 = _pack_bf16(
            fx.Vector.from_elements([probabilities[i] for i in range(4)], fx.Float32)
        ).bitcast(fx.Int16)
        p1 = _pack_bf16(
            fx.Vector.from_elements(
                [probabilities[4 + i] for i in range(4)], fx.Float32
            )
        ).bitcast(fx.Int16)
        next_values0 = load_v(
            vr, v_rows[1], t, 1, 0, lane, visible, extent, HK, BN, TOPK
        )
        next_values1 = load_v(
            vr, v_rows[1], t, 1, 1, lane, visible, extent, HK, BN, TOPK
        )
        rocdl.sched_barrier(0)
        o0 = pv(p0, values0, o0, fx.Float32(1.0))
        rocdl.sched_barrier(0)
        kval = load_key(kr, key_offset, lane)
        rocdl.sched_barrier(0)
        o1 = pv(p0, values1, o1, fx.Float32(1.0))
        rocdl.sched_barrier(0)
        kval1 = load_key(kr, key_offset1, lane)
        rocdl.sched_barrier(0)
        # Eight V loads precede sixteen K loads. Consume V while K stays in
        # flight; the native load-use waits may be stricter after allocation.
        rocdl.s_waitcnt(vmcnt=16)
        rocdl.sched_barrier(0)
        o0 = pv(p1, next_values0, o0, fx.Float32(1.0))
        o1 = pv(p1, next_values1, o1, fx.Float32(1.0))
        maximum = updated
    # Retire the final bounded speculative K prefetch before the epilogue.
    rocdl.s_waitcnt(vmcnt=0)
    _stage_end()
    inv = (total > 0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    outptr = fx.get_iter(OUT) + (fx.Int64(first) * H + hkv * (H // HK)) * 256
    output = rocdl.make_buffer_tensor(
        fx.make_view(outptr, fx.make_layout(NQ * H * 256, 1)), num_records_bytes=qe
    )
    output_fn(
        o0,
        o1,
        inv,
        output,
        shared,
        tid,
        H,
        H // HK,
        rows,
        qe,
        THREADS,
        first,
        QUERY_TILES,
        ACTIVE,
        GATED,
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
    QUERY_TILES: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    BN: fx.Constexpr[int],
    THREADS: fx.Constexpr[int],
    TASKS: fx.Int32,
    GATED: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
    TOPK: fx.Constexpr[int],
):
    body = _body
    if fx.const_expr(GATED):
        tile = fx.block_idx.x // HK
        first, rows = _uniform(META[tile * 5]), _uniform(META[tile * 5 + 1])
        needed = (_uniform(DIRECT_FLAG[0]) != 0).select(fx.Int32(1), fx.Int32(0))
        if needed != 0:
            needed = fx.Int32(0)
            for local in fx.range_constexpr(THREADS // 64):
                row = first + _min(fx.Int32(local), rows - 1)
                group = _uniform(QUERY_TILES[row])
                needed = needed | (_uniform(ACTIVE[group]) == 0).select(
                    fx.Int32(1), fx.Int32(0)
                )
        if needed != 0:
            body(
                Q,
                K,
                V,
                OUT,
                BLOCKS,
                META,
                ACTIVE,
                QUERY_TILES,
                H,
                HK,
                NQ,
                BN,
                THREADS,
                TASKS,
                GATED,
                SCALE,
                TOPK,
            )
    else:
        body(
            Q,
            K,
            V,
            OUT,
            BLOCKS,
            META,
            ACTIVE,
            QUERY_TILES,
            H,
            HK,
            NQ,
            BN,
            THREADS,
            TASKS,
            GATED,
            SCALE,
            TOPK,
        )


@flyc.jit
def _launch(
    Q: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    BN: fx.Constexpr[int],
    THREADS: fx.Constexpr[int],
    TASKS: fx.Int32,
    GATED: fx.Constexpr[bool],
    SCALE: fx.Constexpr[float],
    TOPK: fx.Constexpr[int],
    stream: fx.Stream,
):
    if TASKS > 0:
        _kernel(
            Q,
            K,
            V,
            OUT,
            BLOCKS,
            META,
            ACTIVE,
            QUERY_TILES,
            DIRECT_FLAG,
            H,
            HK,
            NQ,
            BN,
            THREADS,
            TASKS,
            GATED,
            SCALE,
            TOPK,
            value_attrs={
                "llvm.target_features": ir.Attribute.parse(
                    '#llvm.target_features<["-packed-fp32-ops"]>'
                ),
            },
        ).launch(grid=(TASKS, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED = {}


def _bound(inputs, prepared, out, stream):
    """(compiled launcher, its arguments) of the packed or raw variant for this plan."""
    tasks = prepared.num_tiles * inputs.k.shape[1]
    args = (
        inputs.q.view(-1),
        inputs.k,
        inputs.v,
        out.view(-1),
        prepared.source_blocks.view(-1),
        prepared.metadata.view(-1),
        prepared.active,
        prepared.query_tiles,
        prepared.direct_flag,
        inputs.q.shape[1],
        inputs.k.shape[1],
        inputs.q.shape[0],
        32,
        256,
        tasks,
        prepared.gated,
        inputs.scale,
        inputs.topk,
        stream,
    )
    key = (
        inputs.q.device,
        inputs.q.shape[1],
        inputs.k.shape[1],
        prepared.gated,
        inputs.scale,
        inputs.topk,
    )
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm direct QSA before graph capture")
        # Both variants initialize together: a later packed/raw layout switch never JITs.
        with torch.cuda.device(inputs.q.device), torch.cuda.stream(stream):
            compiled = _COMPILED[key] = (
                flyc.compile(_launch, *args[:14], 0, *args[15:]),
                packed.compile_launch(inputs, prepared, out, stream),
            )
    if prepared.packed_key is not None:
        return compiled[1], packed.launch_args(inputs, prepared, out, tasks, stream)
    return compiled[0], args


def run(*, inputs, prepared: DirectPlan, out: torch.Tensor):
    if prepared.num_tiles == 0:
        return
    with torch.cuda.device(inputs.q.device):
        compiled, args = _bound(
            inputs, prepared, out, torch.cuda.current_stream(inputs.q.device)
        )
        compiled(*args)


def launcher(*, inputs, prepared: DirectPlan, out: torch.Tensor):
    """launch(q, k, v, out) with flat Q/out and unflattened K/V; other arguments prebuilt.

    Valid while the plan's scratch bindings and the current stream stay the same.
    """
    if prepared.num_tiles == 0:
        return None
    compiled, args = _bound(
        inputs, prepared, out, torch.cuda.current_stream(inputs.q.device)
    )
    tail = args[4:]
    return lambda q, k, v, o: compiled(q, k, v, o, *tail)

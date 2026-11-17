# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 D256 QSA direct attention over FP8 (E4M3FNUZ) K/V: one query wave over its own
selected blocks, FP8 MFMAs.

While the pack fits the platform's budget (the rule of sparse_prefill_direct), each call
packs every request's K/V into four-token blocks in FP8 MFMA operand order, so the direct
kernel's loads are whole, coalesced operands (no LDS, no permutes); otherwise the direct
kernel reads the raw K/V (RAW, below). The pack launch's CTAs past the blocks quantize Q
(sparse_prefill_common.quantize_rows) for both FP8 kernels. With a union plan (the FP8 union
kernel reads the raw K/V) the pack runs only when K3 routes some tile to direct
(direct_flag[0]):
  PK[block][i * 256 + (g * 4 + token) * 16 + 0..15]
      = K[token][64i + 8g + 0..7] ++ K[token][64i + 32 + 8g + 0..7]
  PV[block][p * 256 + d * 16 + (m - 4p) * 4 + token] = V[token][16d + m]   (p = m >> 2)
Per-token scales are stored per block as one 64-byte record of k_scale, log2(v_scale) and
1 / v_scale (clamped as in the union kernel); tokens past a request's end read 0 (a slot
table's later pool rows too). A pack wave reads its block from the block's own 64-bit base,
so K/V (or a pool) may exceed 4 GiB; the pack is [blocks * HK, 1024] bytes (FlyDSL passes
tensor dims as int32) and below 4 GiB.

A wave runs one query row and KV head: its G <= 16 Q heads are the columns of 16x16x32 FP8
MFMAs. Its key list (complete blocks in block_indices order, then the causal tail block)
runs in 32-key chunks of eight blocks: QK(kg) for blocks 8t + 4kg + jj; S lane (head, g)
holds blocks 8t + g and 8t + 4 + g, which are also its PV K-dim, so P needs no exchange.
PV MFMA m writes dims 16d + m: a lane (head, g) ends with dims 64g .. 64g + 63. Softmax:
the FP8 lazy max (sparse_prefill_common._lazy_max: S'' = S c - m, a rescale only when some
S'' passes LAZY or a fresh row sees its first key, the row max then at S'' = P_BIAS),
uniform per row across its four lanes; c = q_scale * SCALE_LOG2 (times k_scale per tensor,
per key otherwise), SCALE_LOG2 the runtime FP32 softmax_scale * log2(e). Q8 and its row
scales come from the pack launch.

Loads: chunk t + 1's K/V words issue at the start of chunk t; its block ids come one chunk
earlier by scalar loads, so the K/V loads stay in flight (vector loads complete in order).
The loop runs two chunks per iteration, alternating two register sets; waves of disabled
rows exit at once. Per token the 24 scale terms of a 16-lane row's blocks (its g) take two
dword loads per lane and reach their users by DPP row broadcasts.

RAW: a wave first maps its key list's slots to rows in LDS (the slot table's pool row of a
block's first token, else the request row), which each chunk then reads per lane. A lane
loads 16-byte K quarters of its token's rows (lanes g and g ^ 1 swap the halves the other
one needs, ds_swizzle) and its V rows (transposed as the pack does), so the operands and
per-token scale terms (computed as the pack does) are the pack's, bit for bit. K/V of a
request without a slot table read zero past its end; through a slot table masked keys read
zero, or for a pool past 4 GiB (WIDE, 64-bit loads) the request's first row.

Core/residual union plans (prepare.CORE_RATIOS): the pack launch also runs one K3b CTA per
union tile (prepare.residual_tile), which moves the residual blocks of each row of an active
core tile to the front of its block_indices row. Those rows then run here over their residual
blocks only (no causal tail) and merge into the union output by log-sum-exp (the union kernel
writes each row's lse).
"""

import math
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
import torch

from . import sparse_prefill_common as prim
from .sparse_prefill_common import (
    _MAX_FLOOR,
    LAZY,
    MAX_ROW_PACKED_KV_BYTES,
    QROWS,
    _as_f32,
    _as_i32,
    _buffer,
    _buffer_at,
    _buffer_store_words,
    _buffer_words,
    _exp,
    _fp8_word,
    _lazy_max,
    _lds_load,
    _lds_store,
    _maximum,
    _mfma_fp8,
    _min,
    _scaled_bf16,
    _uniform,
    _v_scale_terms,
    platform,
    quantize_rows,
)
from .sparse_prefill_direct import _global_words
from .sparse_prefill_direct_packed import _enabled
from .sparse_prefill_prepare import (
    RESIDUAL_WORDS,
    allocate_scratch,
    packed_block_starts,
    residual_tile,
    upload,
)

__all__ = ["launchers", "prepare", "run", "scratch_specs"]

D = 256
BLOCK_BYTES = 1024  # one packed four-token block of one KV head
CHUNK = 8  # blocks per 32-key chunk


def _transpose4(w):
    """w[t] holds bytes (d0..d3) of token t; returns o[d] holding bytes (t0..t3)."""

    def perm(hi, lo, selector):
        return fx.Int32(prim.perm_b32(hi, lo, selector))

    x_lo = perm(w[1], w[0], 0x05010400)
    x_hi = perm(w[1], w[0], 0x07030602)
    y_lo = perm(w[3], w[2], 0x05010400)
    y_hi = perm(w[3], w[2], 0x07030602)
    return [
        perm(y_lo, x_lo, 0x05040100),
        perm(y_lo, x_lo, 0x07060302),
        perm(y_hi, x_hi, 0x05040100),
        perm(y_hi, x_hi, 0x07060302),
    ]


# ---- Pack ----


def _pack_v(vr, pv, target, start, valid, head, lane, hk, oob):
    """V^T piece of lane p * 16 + d: dims 16d + 4p .. + 3 of the four tokens, transposed."""
    p, d = lane >> 4, lane & 15
    words = []
    for t in range(4):
        source = ((start + t) * hk + head) * D + 16 * d + 4 * p
        source = (fx.Int32(t) < valid).select(source, oob)
        words.append(_buffer_words(vr, source, 1)[0])
    _buffer_store_words(
        fx.Vector.from_elements(_transpose4(words), fx.Int32), pv, target
    )


def _pack_scales(KS, VS, PSC, start, valid, block, head, lane, inside, n, packed, hk):
    """Scale record of the block: k_scale, log2(v_scale), 1 / v_scale of its four tokens
    (lanes 0..3), 64 bytes."""
    scale_extent = fx.Uint32(n) * hk * 4
    ks, vs = _buffer(KS, scale_extent), _buffer(VS, scale_extent)
    t = lane & 3
    source = (t < valid).select(((start + t) * hk + head) * 4, fx.Int32(scale_extent))
    sk = _as_f32(_buffer_words(ks, source, 1)[0])
    lsv, rsv = _v_scale_terms(_as_f32(_buffer_words(vs, source, 1)[0]))
    record_extent = fx.Uint32(packed) * hk * 64
    record = (inside & (lane < 4)).select(
        (block * hk + head) * 64 + t * 4, fx.Int32(record_extent)
    )
    records = _buffer(PSC, record_extent)
    for part, value in enumerate((sk, lsv, rsv)):
        _buffer_store_words(
            fx.Vector.from_elements([_as_i32(value)], fx.Int32),
            records,
            record + 16 * part,
        )


@flyc.jit
def _pack_body(
    K,
    V,
    KS,
    VS,
    PK,
    PV,
    PSC,
    SOURCES,
    VALID,
    SLOTS,
    pair,
    N: fx.Int32,
    PACKED: fx.Int32,
    HK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
):
    lane = fx.Int32(fx.thread_idx.x) & 63
    block, head = pair // HK, pair % HK
    inside = pair < PACKED * HK
    start = _uniform(fx.Int32(SOURCES[_min(block, PACKED - 1)]))
    valid = inside.select(
        _uniform(fx.Int32(VALID[_min(block, PACKED - 1)])), fx.Int32(0)
    )
    if fx.const_expr(SLOTTED):
        # A block occupies four consecutive pool rows from its first token's slot.
        start = _uniform(fx.Int32(SLOTS[start]))
    # Each wave packs one block: its own 64-bit base, so K/V (or a pool) may exceed 4 GiB.
    base = (fx.Int64(start) * HK + head) * D
    kr = _buffer_at(fx.Int64(fx.ptrtoint(fx.get_iter(K))) + base, 4 * HK * D)
    vr = _buffer_at(fx.Int64(fx.ptrtoint(fx.get_iter(V))) + base, 4 * HK * D)
    oob = fx.Int32(4 * HK * D)
    out_extent = fx.Uint32(PACKED) * HK * BLOCK_BYTES
    pk, pv = _buffer(PK, out_extent), _buffer(PV, out_extent)
    target = inside.select(
        (block * HK + head) * BLOCK_BYTES + lane * 16, fx.Int32(out_extent)
    )
    # K: piece (i, token, g) of lane i * 16 + token * 4 + g (reads stay contiguous), stored
    # token-minor so each 4-lane quad of a direct K load reads 64 contiguous bytes.
    i, token, g = lane >> 4, (lane >> 2) & 3, lane & 3
    row = (token < valid).select(token * HK * D, oob)
    lo = _buffer_words(kr, row + 64 * i + 8 * g, 2)
    hi = _buffer_words(kr, row + 64 * i + 32 + 8 * g, 2)
    k_target = inside.select(
        (block * HK + head) * BLOCK_BYTES + i * 256 + (g * 4 + token) * 16,
        fx.Int32(out_extent),
    )
    _buffer_store_words(
        fx.Vector.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int32),
        pk,
        k_target,
    )
    _pack_v(vr, pv, target, 0, valid, 0, lane, HK, oob)
    if fx.const_expr(PER_TOKEN):
        _pack_scales(
            KS,
            VS,
            PSC,
            start,
            valid,
            block,
            head,
            lane,
            inside,
            N,
            PACKED,
            HK,
        )


@flyc.kernel(name="sparse_prefill_pack_kv_fp8_d256", known_block_size=[256, 1, 1])
def _pack_kernel(
    K: fx.Tensor,
    V: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    SOURCES: fx.Tensor,
    VALID: fx.Tensor,
    SLOTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    Q: fx.Tensor,
    Q8: fx.Tensor,
    QS: fx.Tensor,
    BLOCK_ROWS: fx.Tensor,
    TILE_COUNTS: fx.Tensor,
    TILE_BLOCKS: fx.Tensor,
    TILE_META: fx.Tensor,
    TILE_ACTIVE: fx.Tensor,
    TILE_RESIDUAL: fx.Tensor,
    RESIDUAL_ROWS: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    ROWS: fx.Int32,
    CAPACITY: fx.Int32,
    MAX_BLOCKS: fx.Int32,
    TILES: fx.Int32,
    HK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    GATED: fx.Constexpr[bool],
    RESIDUAL: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
):
    cta = fx.Int32(fx.block_idx.x)
    pack_ctas = (PACKED * HK + 3) // 4
    quant_end = pack_ctas + (ROWS + QROWS - 1) // QROWS
    if fx.const_expr(RESIDUAL):
        storage = (
            fx.SharedAllocator().allocate(fx.Array[fx.Int32, RESIDUAL_WORDS, 16]).peek()
        )
        bits = fx.Int32(
            fx.ptrtoint(fx.get_iter(storage.view(fx.make_layout(RESIDUAL_WORDS, 1))))
        )
    if cta >= quant_end:
        # Then one CTA per union tile lists its rows' residual blocks (K3b).
        if fx.const_expr(RESIDUAL):
            residual_tile(
                BLOCK_ROWS,
                TILE_COUNTS,
                TILE_BLOCKS,
                TILE_META,
                TILE_ACTIVE,
                TILE_RESIDUAL,
                RESIDUAL_ROWS,
                CAPACITY,
                MAX_BLOCKS,
                cta - quant_end,
                bits,
                TOPK,
            )
    elif cta >= pack_ctas:
        # CTAs past the blocks quantize Q for both FP8 kernels (one launch fewer).
        quantize_rows(Q, Q8, QS, ROWS, cta - pack_ctas)
    else:
        pair = cta * 4 + _uniform(fx.Int32(fx.thread_idx.x) >> 6)
        args = (K, V, KS, VS, PK, PV, PSC, SOURCES, VALID, SLOTS, pair)
        if fx.const_expr(GATED):
            # With a union plan only the direct kernel reads the pack: K3 sets
            # direct_flag[0] when some tile goes direct (or keeps residual blocks).
            if _uniform(fx.Int32(DIRECT_FLAG[0])) != 0:
                _pack_body(*args, N, PACKED, HK, PER_TOKEN, SLOTTED)
        else:
            _pack_body(*args, N, PACKED, HK, PER_TOKEN, SLOTTED)


@flyc.jit
def _pack_launch(
    K: fx.Tensor,
    V: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    SOURCES: fx.Tensor,
    VALID: fx.Tensor,
    SLOTS: fx.Tensor,
    DIRECT_FLAG: fx.Tensor,
    Q: fx.Tensor,
    Q8: fx.Tensor,
    QS: fx.Tensor,
    BLOCK_ROWS: fx.Tensor,
    TILE_COUNTS: fx.Tensor,
    TILE_BLOCKS: fx.Tensor,
    TILE_META: fx.Tensor,
    TILE_ACTIVE: fx.Tensor,
    TILE_RESIDUAL: fx.Tensor,
    RESIDUAL_ROWS: fx.Tensor,
    N: fx.Int32,
    PACKED: fx.Int32,
    ROWS: fx.Int32,
    CAPACITY: fx.Int32,
    MAX_BLOCKS: fx.Int32,
    TILES: fx.Int32,
    HK: fx.Constexpr[int],
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    GATED: fx.Constexpr[bool],
    RESIDUAL: fx.Constexpr[bool],
    TOPK: fx.Constexpr[int],
    stream: fx.Stream,
):
    if ROWS > 0:
        _pack_kernel(
            K,
            V,
            KS,
            VS,
            PK,
            PV,
            PSC,
            SOURCES,
            VALID,
            SLOTS,
            DIRECT_FLAG,
            Q,
            Q8,
            QS,
            BLOCK_ROWS,
            TILE_COUNTS,
            TILE_BLOCKS,
            TILE_META,
            TILE_ACTIVE,
            TILE_RESIDUAL,
            RESIDUAL_ROWS,
            N,
            PACKED,
            ROWS,
            CAPACITY,
            MAX_BLOCKS,
            TILES,
            HK,
            PER_TOKEN,
            SLOTTED,
            GATED,
            RESIDUAL,
            TOPK,
            value_attrs=prim.kernel_attrs(),
        ).launch(
            grid=(
                (PACKED * HK + 3) // 4
                + (ROWS + QROWS - 1) // QROWS
                + (TILES if RESIDUAL else 0),
                1,
                1,
            ),
            block=(256, 1, 1),
            stream=stream,
        )


# ---- Direct ----


def _raw_env(env, K, V, KS, VS, SLOTS, length, nk, topk, slotted, wide):
    """RAW: env of the raw K/V (and per token their scales) from META's start, the request's
    first K row or slot-table entry: descriptors over the request's rows, or through the slot
    table over the pool's (WIDE: 64-bit K/V addresses) with the masked-key limits of
    _raw_sources (slot s, token r is visible while s < complete + (r < tail))."""
    lane, hk, hkv = env.lane, env.hk, env.hkv
    n, g = lane & 15, lane >> 4
    env.slotted, env.wide = slotted, wide
    env.table_words = topk // 64 + 1
    env.token = lane & 3
    # Row-heads before (row 0 of the pool, or the request's first row, KV head hkv) and from it.
    first = fx.Int64(hkv) if slotted else fx.Int64(env.p_start) * hk + hkv
    count = fx.Uint32(nk if slotted else length) * hk - hkv
    k_base = fx.Int64(fx.ptrtoint(fx.get_iter(K))) + first * D
    v_base = fx.Int64(fx.ptrtoint(fx.get_iter(V))) + first * D
    # K: lane g reads quarter g // 2 (g even) or 2 + g // 2 (g odd) of each 64 bytes of its
    # token's row, the 16 bytes of the pack's operand coming from lanes g and g ^ 1.
    quarter = (g >> 1) * 16 + (g & 1) * 32
    env.k_even = (g & 1) == 0
    if wide:
        env.kbase, env.vbase = k_base + fx.Int64(quarter), v_base + fx.Int64(16 * n)
        env.gk = env.gv = None
    else:
        env.gk, env.gv = _buffer_at(k_base, count * D), _buffer_at(v_base, count * D)
        env.oob = fx.Int32(count * D)
        env.k_lane, env.v_lane = (lane & 3) * (hk * D) + quarter, n * 16
    if env.per_token:
        layout = fx.make_layout(count, 1)
        env.gks = _buffer(fx.make_view(fx.get_iter(KS) + first, layout), count * 4)
        env.gvs = _buffer(fx.make_view(fx.get_iter(VS) + first, layout), count * 4)
        env.sc_oob = fx.Int32(count * 4)
        env.sc_r = n & 3
        env.sc_k, env.sc_log = (n % 12) < 4, ((n % 12) >> 2) == 1
        env.sc_up_log = (n & 7) < 4
    if slotted:
        env.gslots = _buffer(
            fx.make_view(fx.get_iter(SLOTS) + env.p_start, fx.make_layout(length, 1)),
            fx.Uint32(length) * 4,
        )

        def one(predicate):
            return predicate.select(fx.Int32(1), fx.Int32(0))

        env.k_limit = env.complete + one((lane & 3) < env.tail)
        env.v_limits = [env.complete + one(fx.Int32(r) < env.tail) for r in range(4)]
        if env.per_token:
            limit = env.complete + one(env.sc_r < env.tail)
            env.sc_limits = [
                limit - env.sc_hi.select(fx.Int32(4), fx.Int32(0)),
                limit - 4,
            ]
    if wide:
        env.fallback = _uniform(fx.Int32(_buffer_words(env.gslots, fx.Int32(0), 1)[0]))


def _row_table(env):
    """RAW: the rows of the key list's slots into the wave's LDS table (slot s at word s): the
    complete blocks in block_indices order, the causal tail block, then padding (block 0), each
    as its first token's row (the slot table's pool row, else the request row). Lane l maps
    slots l + 64j, all its loads in flight."""
    slots = [env.lane + 64 * j for j in range(env.table_words)]
    ids = [_buffer_words(env.gblocks, (env.ids_row + s) * 4, 1)[0] for s in slots]
    rows = []
    for s, listed in zip(slots, ids):
        block = (s < env.complete).select(
            fx.Int32(listed), (s == env.complete).select(env.tail_block, fx.Int32(0))
        )
        rows.append(block * 4)
    if env.slotted:
        # Request-wide: past the request's end (a masked tail block) reads row 0.
        rows = [_buffer_words(env.gslots, row * 4, 1)[0] for row in rows]
    for s, row in zip(slots, rows):
        _lds_store(env.table + s * 4, row)


def _chunk_ids(env, t):
    """Listed block ids of key-list slots 8t .. 8t + 7 by two wave-uniform scalar loads (SMEM,
    lgkmcnt): they do not wait behind the K/V loads in the in-order vector memory counter.
    Slots past the row's list read another row's ids or zero (unused). RAW: the rows of the
    lane's blocks 8t + jj, 8t + 4 + jj (K) and 8t + g, 8t + 4 + g (V) from LDS."""
    if env.raw:
        k_at = env.table + (t * CHUNK + env.jj) * 4
        v_at = env.table + (t * CHUNK + env.g) * 4
        return [fx.Int32(_lds_load(at)) for at in (k_at, k_at + 16, v_at, v_at + 16)]
    offset = env.ids_row + t * CHUNK
    words = [
        fx.rocdl.s_buffer_load(env.gblocks, offset + 4 * h, count=4) for h in range(2)
    ]
    return [_uniform(fx.Vector(w)[i]) for w in words for i in range(4)]


def _chunk_bases(env, ids, t):
    """(K bases of blocks 8t + jj, 8t + 4 + jj; V/scale bases of blocks 8t + g, 8t + 4 + g):
    complete blocks in block_indices order, the causal tail block, then padding (block 0,
    masked). RAW: _raw_sources."""
    if env.raw:
        return _raw_sources(env, ids, t)
    first = t * CHUNK
    bases = []
    for s in range(CHUNK):
        slot = first + s
        block = (slot < env.complete).select(
            fx.Int32(ids[s]), (slot == env.complete).select(env.tail_block, fx.Int32(0))
        )
        bases.append(((env.p_start + block) * env.hk + env.hkv) * BLOCK_BYTES)

    def pick(sel, lo):
        low = (sel & 1) == 1
        a = low.select(bases[lo + 1], bases[lo])
        b = low.select(bases[lo + 3], bases[lo + 2])
        return ((sel & 2) == 2).select(b, a)

    return [pick(env.jj, 0), pick(env.jj, 4), pick(env.g, 0), pick(env.g, 4)]


def _raw_sources(env, rows, t):
    """RAW: (K sources of the lane's token in blocks 8t + jj, 8t + 4 + jj; V sources of the
    four tokens of blocks 8t + g, 8t + 4 + g; scale offsets of _raw_loads) from the rows of
    _chunk_ids: byte offsets, WIDE 64-bit K/V addresses. Through a slot table masked keys read
    zero (WIDE: the request's first row): pool rows past a request's end may hold anything.
    The request's own descriptors read zero past its end."""
    k_slot, v_slot = t * CHUNK + env.jj, t * CHUNK + env.g
    stride = env.hk * D
    if env.slotted:
        k_valid = [k_slot + 4 * kg < env.k_limit for kg in range(2)]
        v_valid = [
            v_slot + 4 * b < env.v_limits[tt] for b in range(2) for tt in range(4)
        ]
    if env.wide:
        k = [
            env.kbase
            + fx.Int64(k_valid[kg].select(rows[kg] + env.token, env.fallback)) * stride
            for kg in range(2)
        ]
        v = [
            env.vbase
            + fx.Int64(v_valid[4 * b + tt].select(rows[2 + b] + tt, env.fallback))
            * stride
            for b in range(2)
            for tt in range(4)
        ]
    else:
        k = [rows[kg] * stride + env.k_lane for kg in range(2)]
        v = [rows[2 + b] * stride + env.v_lane for b in range(2) for _ in range(4)]
        if env.slotted:
            k = [k_valid[kg].select(k[kg], env.oob) for kg in range(2)]
            v = [v_valid[i].select(v[i], env.oob) for i in range(8)]
        v = [v[i] + (i % 4) * stride for i in range(8)]
    scales = []
    if env.per_token:
        # Rows of values n (block n >= 12) and 16 + n (block 1), token n % 4.
        scales = [
            (env.sc_hi.select(rows[3], rows[2]) + env.sc_r) * (env.hk * 4),
            (rows[3] + env.sc_r) * (env.hk * 4),
        ]
        if env.slotted:
            scales = [
                (v_slot < env.sc_limits[w]).select(scales[w], env.sc_oob)
                for w in range(2)
            ]
    return k, v, scales


def _raw_loads(env, sources):
    """RAW: 8 + 8 b128 loads, K quarters of the token's row (_raw_env) and V rows (dims
    16n .. 16n + 15 of each token), which _operands turns into the pack's operands, and per
    token k_scale, v_scale of value n and v_scale of value 16 + n (_scale_words)."""
    k_src, v_src, sc_src = sources

    def load(source, buffer, offset=0):
        if env.wide:
            return _global_words(source + offset)
        return _buffer_words(buffer, source + offset)

    k = [load(k_src[kg], env.gk, 64 * i) for kg in range(2) for i in range(4)]
    v = [load(source, env.gv) for source in v_src]
    scales = []
    if env.per_token:
        scales = [
            _buffer_words(env.gks, sc_src[0], 1)[0],
            _buffer_words(env.gvs, sc_src[0], 1)[0],
            _buffer_words(env.gvs, sc_src[1], 1)[0],
        ]
    return k, v, scales


def _chunk_loads(env, bases):
    """K and V words of a chunk (and per-token scale terms): 8 + 8 b128 loads."""
    if env.raw:
        return _raw_loads(env, bases)
    k = [
        _buffer_words(env.gk, bases[kg] + 256 * i + env.k_lane, 4)
        for kg in range(2)
        for i in range(4)
    ]
    v = [
        _buffer_words(env.gv, bases[2 + b] + 256 * p + env.v_lane, 4)
        for b in range(2)
        for p in range(4)
    ]
    scales = []
    if env.per_token:
        # The 24 scale terms of the row's blocks 8t + g, 8t + 4 + g (records: k_scale,
        # log2(v_scale), 1 / v_scale of four tokens) spread over its 16 lanes: two dword
        # loads instead of six 16-byte loads per lane. Bases are unsigned: past 2 GiB of
        # packed K/V an arithmetic shift would read zeros past the records.
        rec = [fx.Int32(fx.Uint32(bases[2 + b]) >> 10) * 64 for b in range(2)]
        scales = [
            _buffer_words(env.gsc, env.sc_hi.select(rec[1], rec[0]) + env.sc_lo, 1)[0],
            _buffer_words(env.gsc, rec[1] + env.sc_up, 1)[0],
        ]
    return k, v, scales


def _chunk_words(env, bases, zero4):
    k, v, sc = _chunk_loads(env, bases)
    if not env.per_token:
        sc = [zero4[0], zero4[0]]
    return _flatten(k), _flatten(v), fx.Vector.from_elements(sc, fx.Int32)


def _scale_term(sc, b, field, r):
    """Per-token scale term (field 0 k_scale, 1 log2(v_scale), 2 1 / v_scale) of token r of
    the lane's block b: value 12b + 4 field + r of its 16-lane row (g), word 0 holding values
    0..15 and word 1 values 16..23 in lanes 0..7, read by a DPP row broadcast."""
    v = 12 * b + 4 * field + r
    return prim.row_broadcast(_as_f32(sc[v >> 4]), v & 15)


def _probabilities(vals, sc, per_token):
    """FP8 P' words of the lane's 8 keys and its row-sum term (P' / v_scale per token)."""
    p = [_exp(vals[i]) for i in range(8)]
    terms = (
        [p[i] * _scale_term(sc, i // 4, 2, i % 4) for i in range(8)] if per_token else p
    )
    total = terms[0]
    for x in terms[1:]:
        total = total + x
    return (_fp8_word(p[0:4]), _fp8_word(p[4:8])), total


def _pv16(o, v, pw):
    """O^T += V^T P for the 16 PV MFMAs (MFMA m writes dims 16d + m)."""
    o = fx.Vector(o)
    out = []
    for m in range(16):
        a = (v[m >> 2][m & 3], v[4 + (m >> 2)][m & 3])
        acc = fx.Vector.from_elements([o[4 * m + r] for r in range(4)], fx.Float32)
        acc = _mfma_fp8(a, pw, acc)
        out += [acc[r] for r in range(4)]
    return fx.Vector.from_elements(out, fx.Float32)


def _store_o(env, o, inv):
    """O / l as BF16: lane (head, g) holds dims 64g .. 64g + 63 (dim 64g + 16r + m in
    o[4m + r])."""
    words = []
    for r in range(4):
        values = [o[4 * m + r] for m in range(16)]
        for k in range(4):
            words += _scaled_bf16(values[4 * k : 4 * k + 4], inv)
    for s in range(8):
        _buffer_store_words(
            fx.Vector.from_elements(words[4 * s : 4 * s + 4], fx.Int32),
            env.gout,
            env.out_lane + 16 * s,
        )


def _merge_o(env, o, inv, m_run, total):
    """Residual row: O_d = o * inv merged with the row's union output O_u (BF16, already
    normalized) by their log-sum-exps (log2 units), then stored like _store_o. The LSE offset
    comes from out_lane (qrow * 512 + g * 128) and O_u is merged a half at a time, so the merge
    keeps no value of its own live across the chunk loop."""
    # out_lane is unsigned (Q may pass 2 GiB).
    lse_lane = fx.Int32(fx.Uint32(env.out_lane) >> 9) << 2
    q_rows = fx.Uint32(env.nq) * env.heads
    lse_u = _as_f32(_buffer_words(_buffer(env.lse, q_rows * 4), lse_lane, 1)[0])
    lse_d = fx.Float32(m_run) + prim.log2(total)
    top = _maximum(lse_u, lse_d)
    w_u, w_d = _exp(lse_u - top), _exp(lse_d - top)
    norm = prim.rcp(w_u + w_d)
    w_u, w_d = w_u * norm, w_d * norm * fx.Float32(inv)
    for half in range(2):
        union = [
            _buffer_words(env.gout, env.out_lane + 16 * s, 4)
            for s in range(4 * half, 4 * half + 4)
        ]
        words = []
        for r in range(2 * half, 2 * half + 2):
            for k in range(4):
                values = []
                for j in range(4):
                    m = 4 * k + j
                    # Dim 16r + m: BF16 half m & 1 of word 8r + m // 2.
                    w = 8 * r + m // 2 - 16 * half
                    bits = fx.Int32(fx.Vector(union[w // 4])[w % 4])
                    bits = (bits & fx.Int32(-65536)) if m & 1 else (bits << 16)
                    values.append(
                        fx.Float32(fx.fma(o[4 * m + r], w_d, _as_f32(bits) * w_u))
                    )
                words += _scaled_bf16(values, fx.Float32(1.0))
        for s in range(4):
            _buffer_store_words(
                fx.Vector.from_elements(words[4 * s : 4 * s + 4], fx.Int32),
                env.gout,
                env.out_lane + 16 * (4 * half + s),
            )


def _flatten(vectors, dtype=fx.Int32):
    return fx.Vector.from_elements([v[i] for v in vectors for i in range(4)], dtype)


def _split(vector, count):
    vector = fx.Vector(vector)
    return [
        fx.Vector.from_elements([vector[4 * j + i] for i in range(4)], vector.dtype)
        for j in range(count)
    ]


def _operands(env, kw, vw):
    """K and V^T operands of a chunk from its words. RAW: lanes g and g ^ 1 (16 apart) swap the
    half of their K quarters the other one needs (ds_swizzle), giving the pack's
    K[64i + 8g ..] ++ K[64i + 32 + 8g ..]; each block's V rows transpose as in _pack_v, under
    the swizzles' latency (the scheduling barrier)."""
    k, v = _split(kw, 8), _split(vw, 8)
    if not env.raw:
        return k, v
    even = env.k_even
    k = [fx.Vector(x) for x in k]
    sent = [[even.select(x[2 + j], x[j]) for j in range(2)] for x in k]
    got = [[fx.Int32(fx.gpu.shuffle_xor(w, 16, 64)) for w in pair] for pair in sent]
    v = [
        fx.Vector.from_elements(
            _transpose4([fx.Vector(v[4 * b + tt])[p] for tt in range(4)]), fx.Int32
        )
        for b in range(2)
        for p in range(4)
    ]
    prim.sched_barrier(0)
    k = [
        fx.Vector.from_elements(
            [even.select(x[j], r[j]) for j in range(2)]
            + [even.select(r[j], x[2 + j]) for j in range(2)],
            fx.Int32,
        )
        for x, r in zip(k, got)
    ]
    return k, v


def _scale_words(env, scw):
    """The two words _scale_term reads: RAW per token, the record's k_scale, log2(v_scale) or
    1 / v_scale from the loaded scales, as _pack_scales computes them."""
    scw = fx.Vector(scw)
    if not (env.raw and env.per_token):
        return [scw[0], scw[1]]
    lsv, rsv = _v_scale_terms(_as_f32(scw[1]))
    lsv_up, rsv_up = _v_scale_terms(_as_f32(scw[2]))
    log = env.sc_log.select(_as_i32(lsv), _as_i32(rsv))
    return [
        env.sc_k.select(fx.Int32(scw[0]), log),
        env.sc_up_log.select(_as_i32(lsv_up), _as_i32(rsv_up)),
    ]


def _scores(env, k, sc, t, m_run):
    """S''(t) of the lane's 8 keys (blocks 8t + g, 8t + 4 + g; tokens 0..3), masked past the
    key list, with the lane-local max and the lazy-max ballot."""
    s = []
    for kg in range(2):
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for c in range(8):
            w = k[4 * kg + (c >> 1)]
            q = env.q[c]
            acc = _mfma_fp8((w[2 * (c & 1)], w[2 * (c & 1) + 1]), (q[0], q[1]), acc)
        s.append(acc)
    fresh = m_run == fx.Float32(_MAX_FLOOR)
    neg = fresh.select(fx.Float32(0.0), fx.Float32(0.0) - m_run)
    vals = []
    for b in range(2):
        slot = t * CHUNK + 4 * b + env.g
        for r in range(4):
            x = s[b][r]
            if env.per_token:
                x = x * _scale_term(sc, b, 0, r)
            x = fx.Float32(fx.fma(x, env.c_scale, neg))
            if env.per_token:
                x = x + _scale_term(sc, b, 1, r)
            keep = (slot < env.complete) | (
                (slot == env.complete) & (fx.Int32(r) < env.tail)
            )
            vals.append(keep.select(x, fx.Float32(-math.inf)))
    local = vals[0]
    for x in vals[1:]:
        local = _maximum(local, x)
    trigger = (local > LAZY) | (fresh & (local > fx.Float32(-math.inf)))
    return fx.Vector.from_elements(vals, fx.Float32), local, prim.ballot(trigger)


def _row_max(local):
    """The row max of the lane-local maxima over the row's four lanes (g)."""

    def reduce():
        value = fx.Float32(local)
        value = _maximum(value, fx.gpu.shuffle_xor(value, 16, 64))
        return _maximum(value, fx.gpu.shuffle_xor(value, 32, 64))

    return reduce


@flyc.kernel(name="sparse_prefill_direct_fp8_d256", known_block_size=[64, 1, 1])
def _direct_kernel(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    RES_ROWS: fx.Tensor,
    LSE: fx.Tensor,
    SLOTS: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    ROWS: fx.Int32,
    PACKED: fx.Int32,
    NK: fx.Int32,
    SCALE_LOG2: fx.Float32,
    TOPK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    RESIDUAL: fx.Constexpr[bool],
    RAW: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    WIDE: fx.Constexpr[bool],
):
    group = H // HK
    # Rows run in reverse: later rows list the most blocks (causal-prefix rows the fewest), so
    # the heaviest waves start first and the light or empty ones fill the tail.
    work = ROWS * HK - 1 - fx.Int32(fx.block_idx.x)
    tile, hkv = work // HK, work % HK
    row = _uniform(META[tile * 5])
    p_start, position = _uniform(META[tile * 5 + 2]), _uniform(META[tile * 5 + 4])
    on = _enabled(fx.Int32(0), fx.Int32(1), row, QUERY_TILES, ACTIVE, GATED, NQ)
    lane = fx.Int32(fx.thread_idx.x) & 63
    n, g = lane & 15, lane >> 4
    visible = position + 1
    residual = fx.Boolean(False)
    if fx.const_expr(RESIDUAL):
        # A row of an active union tile with residual blocks: its block_indices row lists them
        # (complete, wholly visible blocks), as if `count` blocks were visible (no tail; the
        # masked tail slot reads block count of the request); the union output gets merged in.
        count = _uniform(RES_ROWS[row])
        residual = ~on & (count > 0)
        on = on | residual
        visible = residual.select(count * 4, visible)
    complete = _min(visible >> 2, fx.Int32(TOPK))
    tail = visible & 3
    tail_block = visible >> 2
    blocks = complete + (tail != 0).select(fx.Int32(1), fx.Int32(0))
    chunks = on.select((blocks + CHUNK - 1) // CHUNK, fx.Int32(0))
    q_rows = fx.Uint32(NQ) * H
    head_ok = on & (n < group)
    qrow = row * H + hkv * group + n
    extent = fx.Uint32(PACKED) * HK * BLOCK_BYTES
    scale_extent = fx.Uint32(PACKED) * HK * 64
    gq8 = _buffer(Q8, q_rows * D)
    q_base = head_ok.select(qrow * D + 8 * g, fx.Int32(q_rows * D))
    qs_offset = head_ok.select(qrow * 4, fx.Int32(q_rows * 4))
    qs = _as_f32(_buffer_words(_buffer(QS, q_rows * 4), qs_offset, 1)[0])
    c_scale = qs * SCALE_LOG2
    if fx.const_expr(not PER_TOKEN):
        c_scale = c_scale * fx.Float32(KS[0])
    env = SimpleNamespace(
        raw=RAW,
        lane=lane,
        hkv=hkv,
        hk=HK,
        p_start=p_start,
        complete=complete,
        tail=tail,
        tail_block=tail_block,
        residual=residual,
        jj=(lane >> 2) & 3,
        g=g,
        per_token=PER_TOKEN,
        c_scale=c_scale,
        gblocks=_buffer(BLOCKS, fx.Uint32(ROWS) * TOPK * 4),
        # Int32 offset of the row's key list (row 0 for a disabled tile, which runs no chunk).
        ids_row=on.select(row * TOPK, fx.Int32(0)),
        gk=_buffer(PK, extent),
        gv=_buffer(PV, extent),
        gsc=_buffer(PSC, scale_extent),
        k_lane=(g * 4 + (lane & 3)) * 16,
        v_lane=n * 16,
        # Scale value n (word 0: block n >= 12, field (n % 12) // 4) and 16 + n (word 1,
        # lanes 0..7: block 1, field 1 + n // 4), token n % 4.
        sc_hi=n >= 12,
        sc_lo=((n % 12) >> 2) * 16 + (n & 3) * 4,
        sc_up=(1 + ((n & 7) >> 2)) * 16 + (n & 3) * 4,
        gout=_buffer(OUT, q_rows * D * 2),
        out_lane=head_ok.select(qrow * (D * 2) + g * 128, fx.Int32(q_rows * D * 2)),
        lse=LSE,
        nq=NQ,
        heads=H,
    )
    if fx.const_expr(RAW):
        table = fx.SharedAllocator().allocate(fx.Array[fx.Int32, TOPK + 64, 16]).peek()
        env.table = fx.Int32(
            fx.ptrtoint(fx.get_iter(table.view(fx.make_layout(TOPK + 64, 1))))
        )
        length = _uniform(META[tile * 5 + 3])
        _raw_env(env, PK, PV, KS, VS, SLOTS, length, NK, TOPK, SLOTTED, WIDE)

    @flyc.jit
    def run():
        # Q (B operand): head n, dims 32c + 8g .. + 7 for MFMA c.
        env.q = [_buffer_words(gq8, q_base + 32 * c, 2) for c in range(8)]
        zero4 = fx.Vector.filled(4, 0, fx.Int32)
        if fx.const_expr(RAW):
            _row_table(env)

        def loads(bases):
            return _chunk_words(env, bases, zero4)

        ids0 = _chunk_ids(env, fx.Int32(0))
        first_k, first_v, first_sc = loads(_chunk_bases(env, ids0, fx.Int32(0)))
        state = (
            fx.Vector.filled(64, 0.0, fx.Float32),
            fx.Float32(_MAX_FLOOR),
            fx.Float32(0.0),
            first_k,
            first_v,
            first_sc,
            *_chunk_ids(env, fx.Int32(1)),
        )

        @flyc.jit
        def step(o, m_run, l_run, kw, vw, scw, *rest):
            """Chunk t from its prefetched words; issues chunk t + 1's loads (block ids from
            the previous step) and chunk t + 2's id loads first."""
            *ids, t = rest
            t = fx.Int32(t)
            bases = _chunk_bases(env, ids, t + 1)
            k_next, v_next, sc_next = loads(bases)
            ids_next = _chunk_ids(env, t + 2)
            # MachineSink cannot move the loads above the fence past the rescale branch into
            # the block that first uses their words (the next step's); the scheduling barrier
            # keeps QK(t), which waits for chunk t, after their issue.
            prim.compiler_fence()
            prim.sched_barrier(0)
            k, v = _operands(env, kw, vw)
            sc = _scale_words(env, scw)
            vals, local, ballot = _scores(env, k, sc, t, fx.Float32(m_run))
            (o,), l_run, vals, m_run = _lazy_max(
                (o,), l_run, vals, m_run, ballot, _row_max(local)
            )
            pw, total = _probabilities(fx.Vector(vals), sc, PER_TOKEN)
            l_run = fx.Float32(l_run) + total
            return (
                _pv16(o, v, pw),
                m_run,
                l_run,
                k_next,
                v_next,
                sc_next,
                *ids_next,
            )

        # Two chunks per iteration: chunk t + 1's words land in the registers chunk t - 1
        # used, so the loop carries them without copies (a copy would wait for the loads at
        # the loop end).
        for t in range(fx.Int32(0), chunks - 1, fx.Int32(2)):
            state = step(*state, t)
            state = step(*state, fx.Int32(t) + 1)
        if (chunks & 1) == 1:
            state = step(*state, chunks - 1)
        o, l_run = fx.Vector(state[0]), fx.Float32(state[2])
        total = l_run + fx.gpu.shuffle_xor(l_run, 16, 64)
        total = total + fx.gpu.shuffle_xor(total, 32, 64)
        inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
        if fx.const_expr(not PER_TOKEN):
            inv = inv * fx.Float32(VS[0])
        if fx.const_expr(RESIDUAL):
            if env.residual:
                _merge_o(env, [o[i] for i in range(64)], inv, state[1], total)
            else:
                _store_o(env, [o[i] for i in range(64)], inv)
        else:
            _store_o(env, [o[i] for i in range(64)], inv)

    # Disabled waves (rows another kernel serves) exit without loads.
    if chunks > 0:
        run()


@flyc.jit
def _direct_launch(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    PK: fx.Tensor,
    PV: fx.Tensor,
    PSC: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    SLOTS: fx.Tensor,
    SCALE_LOG2: fx.Float32,
    BLOCKS: fx.Tensor,
    META: fx.Tensor,
    ACTIVE: fx.Tensor,
    QUERY_TILES: fx.Tensor,
    RES_ROWS: fx.Tensor,
    LSE: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    ROWS: fx.Int32,
    PACKED: fx.Int32,
    NK: fx.Int32,
    TOPK: fx.Constexpr[int],
    GATED: fx.Constexpr[bool],
    PER_TOKEN: fx.Constexpr[bool],
    RESIDUAL: fx.Constexpr[bool],
    RAW: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    WIDE: fx.Constexpr[bool],
    GRID: fx.Int32,
    stream: fx.Stream,
):
    if GRID > 0:
        _direct_kernel(
            Q8,
            QS,
            PK,
            PV,
            PSC,
            KS,
            VS,
            OUT,
            BLOCKS,
            META,
            ACTIVE,
            QUERY_TILES,
            RES_ROWS,
            LSE,
            SLOTS,
            H,
            HK,
            NQ,
            ROWS,
            PACKED,
            NK,
            SCALE_LOG2,
            TOPK,
            GATED,
            PER_TOKEN,
            RESIDUAL,
            RAW,
            SLOTTED,
            WIDE,
            value_attrs=prim.kernel_attrs(),
        ).launch(grid=(GRID, 1, 1), block=(64, 1, 1), stream=stream)


# ---- Host ----


class DirectPlan:
    """FP8 direct plan: one wave per (query row, KV head) over packed or raw K/V."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


def scratch_specs(prepared):
    # [blocks * HK, 1024]: FlyDSL passes int32 dims and a packed context may exceed 2**31 bytes.
    rows = prepared.packed_blocks * prepared.hk
    specs = []
    if not prepared.raw:
        specs.append(("packed_key", (rows, BLOCK_BYTES), torch.uint8))
    if rows:
        specs.append(("packed_value", (rows, BLOCK_BYTES), torch.uint8))
    if prepared.per_token and not prepared.raw:
        specs.append(("scale_records", (rows * 16,), torch.float32))
    return specs


def prepare(*, inputs, union=None, per_token, scratch=True):
    """Plan the FP8 direct path, one wave per row. As BF16 direct (sparse_prefill_direct), the
    call packs every request's context while the pack fits the platform's budget; otherwise
    the direct kernel reads the raw K/V (raw)."""
    hk = inputs.k.shape[1]
    q_lens = np.asarray(inputs.query_lens, dtype=np.int64)
    lengths = q_lens + np.asarray(inputs.prefix_lens, dtype=np.int64)
    blocks = np.where(q_lens > 0, -(-lengths // 4), 0)
    k_starts = np.cumsum(lengths) - lengths
    p_starts = packed_block_starts(inputs.query_lens, inputs.prefix_lens)
    config = platform("token" if per_token else "tensor")
    total = int(blocks.sum())
    # PK + PV, per token with a 64-byte scale record per block and KV head. The direct kernel
    # reads PK/PV at unsigned 32-bit byte offsets (the pack reads each block at its own base).
    pack_bytes = total * hk * (2 * BLOCK_BYTES + (64 if per_token else 0))
    raw = not (
        (
            pack_bytes <= config.max_packed_kv_bytes
            or (
                total * 4 <= config.pack_tokens_per_row * int(q_lens.sum())
                and pack_bytes <= MAX_ROW_PACKED_KV_BYTES
            )
        )
        and total * hk * BLOCK_BYTES < 2**32
    )
    packed_blocks = 0 if raw else total
    request = np.repeat(np.arange(len(q_lens)), q_lens)
    local = np.arange(len(request)) - np.repeat(np.cumsum(q_lens) - q_lens, q_lens)
    rows = len(request)
    metadata = np.stack(
        (
            np.arange(rows),
            np.ones(rows, dtype=np.int64),
            # The request's first packed block, or (raw) K row / slot-table entry.
            (k_starts if raw else p_starts)[request],
            lengths[request],
            (lengths - q_lens)[request] + local,
        ),
        axis=1,
    )
    device = inputs.q.device
    metadata = upload(metadata, device)
    sources = valid = metadata.view(-1)  # stand-ins when nothing is packed
    if packed_blocks:
        owner = np.repeat(np.arange(len(blocks)), blocks)
        index = np.arange(len(owner)) - p_starts[owner]
        sources = upload(k_starts[owner] + 4 * index, device)
        valid = upload(np.minimum(lengths[owner] - 4 * index, 4), device)
    standin = inputs.query_positions
    prepared = DirectPlan(
        metadata=metadata,
        rows=rows,
        hk=hk,
        raw=raw,
        packed_blocks=packed_blocks,
        pack_sources=sources,
        pack_valid=valid,
        per_token=per_token,
        slotted=inputs.kv_slots is not None,
        # Raw through a slot table: the pool spans more than 4 GiB (64-bit loads).
        wide=raw and inputs.wide,
        gated=union is not None,
        direct_flag=standin if union is None else union.direct_flag,
        active=standin if union is None else union.active,
        query_tiles=standin if union is None else union.query_tiles,
        source_blocks=inputs.block_indices,
        # Core/residual union plans: residual rows (union.residual_rows, union.lse) merge.
        union=union,
        residual=union is not None and union.core_ratio > 0,
    )
    if scratch:
        allocate_scratch(prepared, scratch_specs(prepared), device)
    return prepared


_PACK, _DIRECT = {}, {}


def _standin(prepared):
    """A small FP32 tensor for the arguments a specialization does not read."""
    return prepared.metadata.view(-1)[:16].view(torch.float32)


def _pack_args(inputs, prepared, stream):
    empty = _standin(prepared)
    union, residual, raw = prepared.union, prepared.residual, prepared.raw
    standin = prepared.pack_valid
    # A raw call packs nothing: byte stand-ins of PK/PV's rank, so it shares the variant.
    blank = empty.view(torch.uint8).view(1, -1)
    return (
        inputs.k,
        inputs.v,
        inputs.k_scale.view(-1),
        inputs.v_scale.view(-1),
        blank if raw else prepared.packed_key,
        blank if raw else prepared.packed_value,
        prepared.scale_records if prepared.per_token and not raw else empty,
        prepared.pack_sources,
        prepared.pack_valid,
        inputs.kv_slots if prepared.slotted else prepared.pack_sources,
        prepared.direct_flag,
        inputs.q.view(-1),
        inputs.q8.view(-1),
        inputs.qs.view(-1),
        # K3b CTAs of a core/residual union plan (stand-ins otherwise).
        inputs.block_indices.view(-1) if residual else standin,
        union.counts if residual else standin,
        union.blocks.view(-1) if residual else standin,
        union.metadata.view(-1) if residual else standin,
        union.active if residual else standin,
        union.residual if residual else standin,
        union.residual_rows if residual else standin,
        inputs.k.shape[0],
        prepared.packed_blocks,
        inputs.q.shape[0] * inputs.q.shape[1],
        union.block_capacity if residual else 0,
        union.max_blocks if residual else 0,
        union.union_tiles if residual else 0,
        prepared.hk,
        prepared.per_token,
        prepared.slotted,
        prepared.gated,
        residual,
        inputs.topk if residual else 0,
        stream,
    )


def _direct_args(inputs, prepared, out, stream):
    empty = _standin(prepared)
    raw = prepared.raw
    return (
        inputs.q8.view(-1),
        inputs.qs.view(-1),
        inputs.k if raw else prepared.packed_key,
        inputs.v if raw else prepared.packed_value,
        prepared.scale_records if prepared.per_token and not raw else empty,
        inputs.k_scale.view(-1),
        inputs.v_scale.view(-1),
        out.view(-1),
        inputs.kv_slots if raw and prepared.slotted else prepared.metadata.view(-1),
        inputs.scale_log2,
        prepared.source_blocks.view(-1),
        prepared.metadata.view(-1),
        prepared.active,
        prepared.query_tiles,
        prepared.union.residual_rows if prepared.residual else prepared.query_tiles,
        prepared.union.lse.view(-1) if prepared.residual else empty,
        inputs.q.shape[1],
        prepared.hk,
        inputs.q.shape[0],
        prepared.rows,
        prepared.packed_blocks,
        inputs.k.shape[0],
        inputs.topk,
        prepared.gated,
        prepared.per_token,
        prepared.residual,
        raw,
        raw and prepared.slotted,
        prepared.wide,
        prepared.rows * prepared.hk,
        stream,
    )


def _compile(cache, key, launch, args, empty, device, stream, hints=None):
    """Compile once per specialization. Compiling runs the launcher, so args[empty] (the pack's
    Q rows, the direct grid) is 0 and it launches nothing: neither kernel is idempotent
    (the pack's K3b CTAs compact block_indices in place, residual rows merge into out).
    """
    compiled = cache.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        args = list(args)
        args[empty] = 0
        compile_ = flyc.compile if hints is None else flyc.compile[hints]
        with torch.cuda.device(device), torch.cuda.stream(stream):
            compiled = cache[key] = compile_(launch, *args)
    return compiled


def _compiled(inputs, prepared, out, stream):
    device = inputs.q.device
    pack = _pack_args(inputs, prepared, stream)
    direct = _direct_args(inputs, prepared, out, stream)
    # Specializations: pack (HK, PER_TOKEN, SLOTTED, GATED, RESIDUAL, TOPK); direct (H,
    # HK, TOPK, GATED, PER_TOKEN, RESIDUAL, RAW, SLOTTED, WIDE); the softmax scale is a kernel
    # argument.
    pack_fn = _compile(
        _PACK, (device, *pack[27:33]), _pack_launch, pack, 23, device, stream
    )
    key = (device, *direct[16:18], *direct[22:29])
    # Two direct waves per SIMD.
    direct_fn = _compile(
        _DIRECT,
        key,
        _direct_launch,
        direct,
        29,
        device,
        stream,
        {"waves_per_eu": 2},
    )
    return pack, direct, pack_fn, direct_fn


def run(*, inputs, prepared, out, stage="both"):
    """Pack the context and quantize Q into inputs.q8/qs (stage "pack"; the FP8 union kernel
    reads its V^T and Q8), then the direct waves ("direct"). inputs carries q8/qs,
    k_scale/v_scale and scale_log2 like the union kernel."""
    if prepared.rows == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    with torch.cuda.device(inputs.q.device):
        pack, direct, pack_fn, direct_fn = _compiled(inputs, prepared, out, stream)
        if stage in ("pack", "both"):
            pack_fn(*pack)
        if stage in ("direct", "both"):
            direct_fn(*direct)


def launchers(*, inputs, prepared, out):
    """(pack(q, k, v, k_scale, v_scale, slots), direct(q8, qs, k, v, out, k_scale, v_scale,
    slots, scale_log2)) with flat Q/Q8/QS/out/scales and unflattened K/V, other arguments
    prebuilt (the pack writes the bound inputs.q8/qs); None when empty.
    """
    if prepared.rows == 0:
        return None
    stream = torch.cuda.current_stream(inputs.q.device)
    pack, direct, pack_fn, direct_fn = _compiled(inputs, prepared, out, stream)
    slotted, raw = prepared.slotted, prepared.raw

    def pack_launch(q, k, v, ks, vs, slots):
        pack_fn(
            k,
            v,
            ks,
            vs,
            *pack[4:9],
            slots if slotted else pack[9],
            pack[10],
            q,
            *pack[12:],
        )

    def direct_launch(q8, qs, k, v, o, ks, vs, slots, scale_log2):
        kv = (k, v) if raw else direct[2:4]
        table = slots if raw and slotted else direct[8]
        direct_fn(q8, qs, *kv, direct[4], ks, vs, o, table, scale_log2, *direct[10:])

    return pack_launch, direct_launch

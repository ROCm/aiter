# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 D256 QSA union attention over FP8 (E4M3FNUZ) K/V with FP8 MFMAs.

Same plans and tasks as sparse_prefill_union: a task is 128 (query, head) rows of a union tile
over its compact block list, with exact per-query membership. Four waves of 32 rows (one wave
per SIMD) run 32x32x16 FP8 MFMAs, S^T = K Q^T and O^T += V^T P; the 128 O values of a lane
stay in registers (where LLVM allocates them).

Numerics. sparse_prefill_common.quantize_rows (in the FP8 pack launch) stores Q
per (query, head) row as E4M3FNUZ with scale = row amax / 240. K/V scales are per token and
KV head (FP32 [N, HK]; with a slot table [P, HK]) or one per tensor ([1]). Softmax runs in
log2 units with a lazy max m (sparse_prefill_common._lazy_max): S'' = S * c - m with
c = q_scale * SCALE_LOG2 (times k_scale per tensor, per key otherwise), SCALE_LOG2 the
runtime FP32 softmax_scale * log2(e). Per token, log2(v_scale) joins S'', so the FP8
P' = exp2(S'') carries the V scale and stays below 2**LAZY while l accumulates P' / v_scale;
per tensor the output takes v_scale. m moves only when some row's S'' exceeds LAZY or a row
sees its first key; then the row max lands at S'' = P_BIAS (P' = 2**P_BIAS, so small weights
stay above the FP8 underflow) and O, l and S'' follow.

Phase t (64-key tile t; K and V^T double-buffered in 64 KiB LDS):
  Part 1  QK(t) [K slot t&1]; after QK MFMA FIX_AT tile t-1 is finished under the issued
          MFMAs (lazy max, rescale), then P'(t-1) = exp2(S''(t-1)) and its row sums;
          K(t+1) -> K slot (t+1)&1; V(t), ids(t+2) and mask(t) loads; scale ids by
          ds_bpermute. After QK MFMA BAR_AT: lgkmcnt(0) + barrier, then the first V reads
          and the FP8 P' of PV(t-1) under the last QK MFMAs.
  Part 2  PV(t-1) [V slot (t-1)&1] || P' packs, S''(t) (scales, centering); V(t)^T -> V
          slot t&1 interleaved with S''(t)'s membership masks (every tile, branch free:
          whole tiles keep every bit) and its masked row max; K(t+2) and scale(t+1) loads;
          the first K reads of QK(t+1).

K and V come from the raw K/V (packed, or a pool through K3's pool blocks), one buffer per
four-token block from its own 64-bit address, so K/V (a pool) may exceed 4 GiB: its records
end the final partial block at the request's last token and are 0 for slots past the compact
list (phases before a task's last three tiles read whole blocks: FULL). A wave reads the same
four blocks of a tile for K and V (slots vslot + 0, 2, 4, 6). A K load reads one block whole
(lane: token lane >> 4, 16-byte chunk lane & 15), its block id by v_readlane, which the V
loads of the tile reuse two phases later. A V load reads a dword (four dims) of one token per
lane; the four tokens of a block are transposed in registers
(sparse_prefill_direct_fp8._transpose4) into one 16-byte V^T chunk per dim.

Each LDS operand wait covers the K (or V) reads of two MFMA steps: the trace counts the wave's
LDS operations (env.lgkm), which retire in issue order.

The grid is persistent: a CTA prefetches its next task's ids, masks, Q, K(0..2), V(0) and
scales under the current task's epilogue and output, and reads the task order with scalar
loads, which do not wait behind the output stores.

Split plans (fewer tasks than CUs, sparse_prefill_prepare.SPLIT_PARTS): K3 gives a long task two
order entries, its first ceil(steps / 2) and remaining N64 steps. Each part stores its O^T
fragments, m and l (system-scope stores, any XCD may merge) and counts on the task's slot; the
part counting last merges both records, part 0 first (O = sum O_p * 2^(m_p - m), l likewise),
and writes the rows like an unsplit task.

Core/residual plans (core_ratio > 0, RESIDUAL): every row's log-sum-exp m + log2(l) of S * c
(log2 units) goes to LSE, which the FP8 direct kernel merges its residual blocks with.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from . import sparse_prefill_common as prim
from .sparse_prefill_common import (
    _MAX_FLOOR,
    LAZY,
    P_BIAS,
    SC_SYSTEM,
    _as_f32,
    _as_i32,
    _buffer,
    _buffer_at,
    _buffer_store_words,
    _buffer_words,
    _exp,
    _fma,
    _lazy_max,
    _load_lds,
    _maximum,
    _mfma_fp8,
    _min,
    _scaled_bf16,
    _stage_end,
    _uniform,
    _v_scale_terms,
)
from .sparse_prefill_direct_fp8 import _transpose4
from .sparse_prefill_prepare import PARTIAL_FLOATS, SPLIT_PARTS, SPLIT_SLOTS

__all__ = ["launcher", "run"]

BN, D, THREADS = 64, 256, 256
SLOT = BN * D  # one FP8 K or V^T tile
LDS_BYTES = 4 * SLOT  # K slots 0/1, V^T slots 0/1
V0 = 2 * SLOT
NEG_INF = -8388608  # FP32 -inf bits
# Pipeline placement (MI308X): K/V LDS reads issued KPRE/VPRE MFMAs ahead, the Part 1 barrier
# after QK MFMA BAR_AT, VALU steps per Part 1 / Part 2 MFMA (P1 / P2, TAIL after the barrier),
# MFMAs per K(t+1) write and steps per V(t) write (per tensor, per token: more Part 2 VALU).
KPRE, VPRE, BAR_AT, P1, TAIL, P2, KW_GAP, SPREAD = 4, 4, 27, 1, 3, 4, 2, (4, 2)
FIX_AT = 1  # Part 1 finishes tile t-1 after this QK MFMA
# A split part's record (sparse_prefill_prepare.PARTIAL_FLOATS): O^T fragments in MFMA order
# (wave, block mb, register quad k: 16 bytes per lane), then m and the lane's l per wave.
PARTIAL_BYTES = PARTIAL_FLOATS * 4
PARTIAL_ML = 128 * D * 4
K_ORDER = tuple((b, c) for c in range(8) for b in range(2))


class _Env:
    """Trace-time bundle of one task's per-lane values; the helpers below are plain Python."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


def _drain(gen):
    for _ in gen:
        pass


def _pin():
    prim.sched_barrier(0)


def _step(gen):
    next(gen, None)
    _pin()


def _bperm(value, lane_index):
    return prim.bpermute(fx.Int32(lane_index * 4), fx.Int32(value))


def _write_lds(storage, offset, words):
    """ds_write_b64 / b128 of 2 or 4 words at a byte offset of storage."""
    count = len(words)
    ptr = fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 4 * count),
        fx.get_iter(storage) + offset,
    )
    data = fx.make_rmem_tensor(count, fx.Int32)
    data.store(fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32))
    atom = fx.UniversalCopy64b() if count == 2 else fx.UniversalCopy128b()
    fx.copy(
        fx.make_copy_atom(atom, fx.Int32),
        data,
        fx.make_view(ptr, fx.make_layout(count, 1)),
    )


# ---- LDS layouts ----
# K tile: key-major rows of 256 bytes, 16-byte chunks XOR-swizzled by the key.
# V^T tile: dim-major rows of 64 key bytes; rows pair-swapped and chunks swizzled so the b128
# writes (lanes over dims) and the b128 operand reads (lanes over dims) are conflict free.
# Chunk 2h + q of a dim holds the tokens of blocks 8q + h + 0, 2, 4, 6 (PV K-steps 2q, 2q + 1).


def _swz_k(key):
    return (key & 3) | ((key >> 2) & 4)


def _k_offset(key, chunk):
    return key * 256 + ((chunk ^ _swz_k(key)) << 4)


def _vt_chunk(d, chunk):
    row = d ^ ((d >> 3) & 1)
    sig = (((d >> 1) ^ (d >> 4)) & 1) | ((((d >> 2) ^ (d >> 6)) & 1) << 1)
    return row * 64 + ((chunk ^ sig) << 4)


def _k_read(env, slot, b, c):
    """K operand words of keys 32b + lane, dims 32c + 16h .. + 15."""
    env.lgkm += 1
    return _load_lds(env.lds, env.k_addr[c], slot * SLOT + 8192 * b)


def _v_read(env, slot, mb, q):
    """V^T operand of dims 32mb + lane: words of PV K-steps 2q, 2q + 1 in order."""
    env.lgkm += 1
    if isinstance(slot, int):
        return _load_lds(
            env.lds, env.v_addr[(mb >> 1) & 1][q], V0 + slot * SLOT + 2048 * mb
        )
    return _load_lds(
        env.lds, env.v_addr[(mb >> 1) & 1][q] + slot * SLOT, V0 + 2048 * mb
    )


# ---- Global loads ----


def _load_idl(env, t):
    """Block of slot (lane & 15) of key tile t (clamped to the last); slots past count read 0."""
    return _buffer_words(
        env.gblocks, (_min(t, env.last) * 16 + (env.lane & 15)) * 4, 1
    )[0]


def _fix_ids(env, idl, t):
    """Slots past the compact list take a block past the scales' extent, so their scale loads
    read 0 without per-load checks."""
    slot = _min(t, env.last) * 16 + (env.lane & 15)
    return (slot < env.count).select(fx.Int32(idl), env.id_oob)


def _kv_block(env, base, block, slot):
    """A K or V block's own 64-bit address (K/V may exceed 4 GiB) and buffer records: the
    final partial block ends at the request's end; past the compact list they are 0. A tile
    before a task's last (env.full) has whole blocks only."""
    address = base + fx.Int64(fx.Uint32(block)) * (4 * env.tok_bytes)
    if env.full:
        return address, fx.Int32(4 * env.tok_bytes)
    records = (block == env.final).select(env.partial, fx.Int32(4 * env.tok_bytes))
    return address, (slot < env.count).select(records, fx.Int32(0))


def _load_mask(env, t):
    """Membership words of the lane's query over key tile t (quarters 0-1, 2-3)."""
    w = _buffer_words(env.gmask, env.mask_lane + _min(t, env.last) * env.mask_step, 2)
    return w[0], w[1]


def _mask_words(env, words):
    """The lane's key-half bits: value (b, i) is bit 16 * ((i >> 2) & 1) + 8b + (i & 3) of
    word i >> 3. Rows outside the task keep every key."""
    shift = env.h * 4
    return [env.valid.select(fx.Int32(w) >> shift, fx.Int32(-1)) for w in words]


def _keep(words, b, i):
    """0 / -1 membership of value (b, i) (signed bit extraction)."""
    position = 16 * ((i >> 2) & 1) + 8 * b + (i & 3)
    return (words[i >> 3] << (31 - position)) >> 31


def _masked(value, keep):
    # Bit select: the score bits or -inf, without a compare and VCC per value.
    bits = _as_i32(value)
    return _as_f32((bits & keep) | ((~keep) & fx.Int32(NEG_INF)))


def _gen_v_loads(env, vids, t, out):
    """The wave's four V blocks of key tile t (block ids vids, from its K loads): per token a
    dword of dims 4 * lane .. + 3 (_gen_v_writes transposes them; tokens past a request's end
    and slots past the list read 0). Per token a block's loads take two steps."""
    for j, vid in enumerate(vids):
        address, records = _kv_block(env, env.v_base, vid, t * 16 + env.vslot + 2 * j)
        yield
        buffer = _buffer_at(address, records)
        words = []
        for token in range(4):
            words.append(
                _buffer_words(buffer, env.v_lane + token * env.tok_bytes, 1)[0]
            )
            if token == 1 and env.per_token:
                yield
        out.append(fx.Vector.from_elements(words, fx.Int32))
        yield


def _gen_k_loads(env, idl, t, out, ids):
    """K of key tile t: load j of a wave reads block slot vslot + 2j whole (lane: token
    lane >> 4, 16-byte chunk lane & 15), its block id read from idl by v_readlane (also to
    ids, for the V loads of the tile)."""
    slots = [env.vslot + 2 * j for j in range(4)]
    blocks = [_uniform(prim.readlane(fx.Int32(idl), s)) for s in slots]
    ids.extend(blocks)
    yield
    for slot, block in zip(slots, blocks):
        address, records = _kv_block(env, env.k_base, block, t * 16 + slot)
        yield
        part = _buffer_words(_buffer_at(address, records), env.k_lane, 4)
        out.extend(part[j] for j in range(4))
        yield


def _gen_scales(env, sids, out):
    """(sk0, sk1, sv0, sv1) of the lane's scale keys; per tensor: zeros."""
    if not env.per_token:
        out.extend([fx.Float32(0.0)] * 4)
        return
    offsets = []
    for b in range(2):
        offset = (sids[b] * 4 + env.stok) * env.scale_step
        if env.slotted:
            partial = (sids[b] == env.final) & (env.stok >= env.tail_len)
            offset = partial.select(env.scale_oob, offset)
        offsets.append(offset)
    scales = []
    for buffer in (env.gks, env.gvs):
        for b in range(2):
            scales.append(_as_f32(_buffer_words(buffer, offsets[b], 1)[0]))
            yield
    out.extend(scales)


# ---- LDS writes ----


def _gen_k_writes(env, kdata, slot, gap=1):
    for i in range(4):
        env.lgkm += 1
        _write_lds(
            env.lds, env.k_wbase[i] + slot * SLOT, [kdata[4 * i + j] for j in range(4)]
        )
        for _ in range(gap):
            yield


def _gen_v_writes(env, vdata, slot, spread):
    """V^T rows of the lane's four dims: per dim one 16-byte chunk, the four tokens of each of
    the wave's blocks (word j of a block's transpose holds dim 4 * lane + j)."""
    dims = []
    for words in vdata:
        dims.append(_transpose4([fx.Vector(words)[j] for j in range(4)]))
        yield
    for j in range(4):
        words = [dims[b][j] for b in range(4)]
        env.lgkm += 1
        _write_lds(env.lds, env.v_wbase[j] + V0 + slot * SLOT, words)
        for _ in range(spread):
            yield


# ---- Softmax pieces ----


def _gen_part1(env, s_prev, rsv, res):
    """P'(t-1) = exp2(S''(t-1)) and the row sum of P' (per token P' / v_scale) in the issue
    slack of the exps, a value's term trailing its exp by two steps (a DPP instruction
    reading the previous VALU's result needs two wait states)."""
    p, sums = [None] * 32, [None, None]
    res["p"] = p
    for i in range(34):
        if i < 32:
            p[i] = _exp(s_prev[i])
        if i > 1:
            k = i - 2
            term = (
                p[k] * prim.row_broadcast(rsv[k // 16], k % 16)
                if env.per_token
                else p[k]
            )
            sums[k & 1] = term if sums[k & 1] is None else sums[k & 1] + term
        yield
    prim.hold(*sums)
    res["sum"] = sums[0] + sums[1]


def _gen_pack(p, s, pw):
    """FP8 words of PV K-step s: values 8s .. 8s + 7 of P'."""
    words = []
    for i in (0, 4):
        values = p[8 * s + i : 8 * s + i + 4]
        lo = prim.cvt_pk_fp8(values[0], values[1])
        yield
        words.append(prim.cvt_pk_fp8(values[2], values[3], lo, True))
        yield
    pw[s] = words


def _interleave(*gens):
    """One step of each generator in turn until all are done."""
    live = list(gens)
    while live:
        for gen in list(live):
            try:
                next(gen)
            except StopIteration:
                live.remove(gen)
                continue
            yield


def _gen_part2(env, ctx):
    """P' K-steps 1-3; S''(t) = S * c - m (speculative, m of t-1) with the scales, its
    membership masks and masked row max; V(t) -> LDS."""
    p, pw = ctx["p"], ctx["pw"]
    c_scale, neg = env.c_scale, ctx["neg"]
    terms = [None, None, None, None]

    def gen_terms():
        if env.per_token:
            for b in range(2):
                terms[b], terms[2 + b] = _v_scale_terms(ctx["sv"][b])
                yield

    def gen_center(lo, hi, vals, part):
        if env.per_token:
            # A DPP instruction reading a VGPR the previous VALU wrote needs two wait states
            # (any VGPR operand): the log2(v_scale) add of a value trails its FMA by one value.
            # S * (k_scale * c): S * k_scale overflows FP32 for K scales past 2**104.
            xs = {}
            for k in range(lo, hi + 1):
                if k < hi:
                    b, i = divmod(k, 16)
                    c = prim.row_broadcast(ctx["sk"][b], i) * c_scale
                    xs[k] = _fma(ctx["s_acc"][b][i], c, neg)
                    yield
                if k > lo:
                    kk = k - 1
                    b, i = divmod(kk, 16)
                    vals[kk] = xs.pop(kk) + prim.row_broadcast(terms[b], i)
                    yield
            return
        for k in range(lo, hi):
            b, i = divmod(k, 16)
            vals[k] = _fma(ctx["s_acc"][b][i], c_scale, neg)
            yield

    def gen_mask_max(part):
        # Membership masks of every tile (whole tiles keep all bits), then the masked row
        # max: no masked branch at the next phase's fix-up.
        ws = _mask_words(env, ctx["mask"])
        yield
        for k in range(32):
            b, i = divmod(k, 16)
            vals[k] = _masked(vals[k], _keep(ws, b, i))
            yield
            if k % 2 == 1:
                a, c = vals[k - 1], vals[k]
                part.append(
                    _maximum(a, c) if not part else _maximum(_maximum(part.pop(), a), c)
                )
                yield

    vals, part = [None] * 32, []
    yield from gen_terms()
    if env.per_token:
        prim.hold(*terms[2:])
    yield from _gen_pack(p, 1, pw)
    yield from gen_center(0, 16, vals, part)
    yield from _gen_pack(p, 2, pw)
    yield from gen_center(16, 32, vals, part)
    yield from _gen_pack(p, 3, pw)
    yield from _interleave(
        _gen_v_writes(env, ctx["vdata"], ctx["cur"], SPREAD[env.per_token]),
        gen_mask_max(part),
    )
    local = part[0]
    env.lgkm += 1
    other = fx.gpu.shuffle_xor(local, 32, 64)
    yield
    ctx.update(s_cur=vals, local=local, other=other, rsv=terms[2:])


def _lazy_ballot(local, other, fresh):
    """(row max of S'', the lazy-max ballot): some row's S'' above LAZY, or a fresh row
    (no key yet, m = FLOOR, centered at 0) seeing its first key."""
    tile_max = _maximum(fx.Float32(local), fx.Float32(other))
    first = fresh & (tile_max > fx.Float32(-math.inf))
    trigger = (tile_max > LAZY) | first
    return tile_max, prim.ballot(trigger)


def _qk_simple(env, slot):
    s_acc = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(2)]
    for b in range(2):
        for c in range(8):
            kw, q = _k_read(env, slot, b, c), env.q_words[c]
            for e in range(2):
                s_acc[b] = _mfma_fp8(
                    (kw[2 * e], kw[2 * e + 1]), (q[2 * e], q[2 * e + 1]), s_acc[b]
                )
    return s_acc


def _first_max(env, s_acc, scales, words):
    """Tile 0: m = row max - P_BIAS (FLOOR without a visible key) and S''(0); (log2 sv, 1/sv)."""
    terms = [None] * 4
    if env.per_token:
        for b in range(2):
            terms[b], terms[2 + b] = _v_scale_terms(scales[2 + b])
    ws = _mask_words(env, words)
    vals = []
    for k in range(32):
        b, i = divmod(k, 16)
        v = s_acc[b][i]
        if env.per_token:
            x = v * (
                prim.row_broadcast(scales[b], i) * env.c_scale
            ) + prim.row_broadcast(terms[b], i)
        else:
            x = v * env.c_scale
        vals.append(_masked(x, _keep(ws, b, i)))
    local = vals[0]
    for v in vals[1:]:
        local = _maximum(local, v)
    row_max = _maximum(
        _maximum(local, fx.gpu.shuffle_xor(local, 32, 64)), fx.Float32(_MAX_FLOOR)
    )
    m_run = (row_max == fx.Float32(_MAX_FLOOR)).select(row_max, row_max - P_BIAS)
    neg = fx.Float32(0.0) - m_run
    return (
        fx.Vector.from_elements([v + neg for v in vals], fx.Float32),
        m_run,
        terms[2:],
    )


# ---- Phase parts ----


def _part1(env, st, t, cur, prev, fixup):
    """QK(t) and P'(t-1); the barrier; the first V reads and P' K-step 0 of PV(t-1).

    After QK MFMA FIX_AT, fixup() finishes tile t-1 (masks, lazy max, rescale) under the
    issued MFMAs; the exponentials of P'(t-1) follow it."""
    mem = {"vdata": []}

    def gen_mem():
        vgen = _gen_v_loads(env, st["vids"], t, mem["vdata"])
        kgen = _gen_k_writes(env, st["kdata"], 1 - cur, KW_GAP)
        # Interleave the K(t+1) writes (one per KW_GAP MFMAs) with the V(t) loads.
        for _ in zip(kgen, vgen):
            yield
        yield from kgen
        yield from vgen
        mem["idl2"] = _load_idl(env, t + 2)
        yield
        mem["mask"] = _load_mask(env, t)
        yield
        idl1 = _fix_ids(env, st["idl1"], t + 1)
        yield
        # Two ds_bpermute (dead per tensor: no scale loads).
        env.lgkm += env.per_token
        mem["sids"] = (_bperm(idl1, env.sj),)
        yield
        env.lgkm += env.per_token
        mem["sids"] += (_bperm(idl1, env.sj + 8),)
        yield

    res, pw, vreads = {}, [None] * 4, {}
    valu, fixed = iter(()), None
    memg = gen_mem()
    tail = None
    reads = list(st["kpre"]) + [None] * (16 - len(st["kpre"]))
    marks = {}
    s_acc = [fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(2)]
    for i in range(16):
        b, c = K_ORDER[i]
        if i + KPRE < 16:
            reads[i + KPRE] = _k_read(env, cur, *K_ORDER[i + KPRE])
            marks[i + KPRE] = env.lgkm
            _pin()
        if i % 2 == 0 and i + 1 in marks and 2 * i + 1 < BAR_AT:
            # One wait for the K reads of this pair and the next (LDS ops retire in order).
            fx.rocdl.s_waitcnt(lgkmcnt=env.lgkm - marks[i + 1])
        kw, q = reads[i], env.q_words[c]
        for e in range(2):
            j = 2 * i + e
            _pin()
            s_acc[b] = _mfma_fp8(
                (kw[2 * e], kw[2 * e + 1]), (q[2 * e], q[2 * e + 1]), s_acc[b]
            )
            _pin()
            for _ in range(P1):
                _step(valu)
            if tail is not None:
                for _ in range(TAIL):
                    _step(tail)
            else:
                _step(memg)
            if j == FIX_AT:
                fixed = fixup()
                _pin()
                valu = _gen_part1(env, fixed["s_prev"], st["rsv_prev"], res)
            if j == BAR_AT:
                # Publishes K(t+1) and V(t-1); retires every wave's QK(t-1) and PV(t-2) reads.
                _drain(memg)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                _stage_end()
                for r in range(VPRE):
                    q_, mb_ = divmod(r, 8)
                    # The V^T operand words and the LDS operation count after them.
                    vreads[(q_, mb_)] = (_v_read(env, prev, mb_, q_), env.lgkm)
                _pin()
                tail = _gen_pack(res["p"], 0, pw)
    _drain(valu)
    _drain(memg)
    if tail is not None:
        _drain(tail)
    return s_acc, res, mem, pw, vreads, fixed


def _part2(env, st, res, s_acc, mem, pw, vreads, t, cur, prev, first=False):
    """PV(t-1) and S''(t); K(t+2) and scale(t+1) loads; the first K reads of QK(t+1). In a
    task's first phase (first) the K-step 0 MFMAs start O."""
    o = list(st["o"]) if not first else [None] * 8
    m_run = st["m_run"]
    fresh = m_run == fx.Float32(_MAX_FLOOR)
    ctx = {
        "p": res["p"],
        "pw": pw,
        "s_acc": s_acc,
        "sk": st["sk"],
        "sv": st["sv"],
        "vdata": mem["vdata"],
        "mask": mem["mask"],
        "fresh": fresh,
        "neg": fresh.select(fx.Float32(0.0), fx.Float32(0.0) - m_run),
        "cur": cur,
    }
    valu = _gen_part2(env, ctx)
    out = {}

    def gen_mem():
        kdata, ids = [], []
        yield from _gen_k_loads(env, mem["idl2"], t + 2, kdata, ids)
        out["kdata"], out["ids"] = kdata, ids
        scales = []
        yield from _gen_scales(env, mem["sids"], scales)
        out["scales"] = scales
        yield

    memg = gen_mem()
    order = [(q, e, mb) for q in range(2) for e in range(2) for mb in range(8)]
    issue_at = {}
    for r in range(VPRE, 16):
        q, mb = divmod(r, 8)
        issue_at.setdefault(max(0, 16 * q + mb - VPRE), []).append((q, mb))
    kpre = []
    for j, (q, e, mb) in enumerate(order):
        for rq, rmb in issue_at.get(j, []):
            vreads[(rq, rmb)] = (_v_read(env, prev, rmb, rq), env.lgkm)
            _pin()
        if j >= 32 - KPRE:
            kpre.append(_k_read(env, 1 - cur, *K_ORDER[len(kpre)]))
            _pin()
        if pw[2 * q + e] is None:
            while pw[2 * q + e] is None:
                next(valu)
        vw = vreads[(q, mb)][0]
        _pin()
        if e == 0 and mb % 2 == 0:
            # One wait for the V reads of this MFMA and the next.
            fx.rocdl.s_waitcnt(lgkmcnt=env.lgkm - vreads[(q, mb + 1)][1])
        # A task's first K-step starts O (no accumulator to load).
        acc = fx.Vector.filled(16, 0.0, fx.Float32) if o[mb] is None else o[mb]
        o[mb] = _mfma_fp8((vw[2 * e], vw[2 * e + 1]), pw[2 * q + e], acc)
        _pin()
        for _ in range(P2):
            _step(valu)
        # Memory steps after MFMAs 1, 3, 4, 6, 7, ...: a block's buffer (SALU) one step ahead
        # of its load.
        if j % 3 != 2 and j > 0:
            _step(memg)
    _drain(valu)
    _drain(memg)
    return o, ctx, out, kpre


def _store_v_now(env, vdata, slot):
    _drain(_gen_v_writes(env, vdata, slot, 0))


def _pv_last(env, pw, o):
    o = list(o)
    for q in range(2):
        for mb in range(8):
            vw = _v_read(env, env.last & 1, mb, q)
            for e in range(2):
                o[mb] = _mfma_fp8((vw[2 * e], vw[2 * e + 1]), pw[2 * q + e], o[mb])
    return o


def _epilogue(env, s_last, l_run, o, rsv):
    """P'(last), its row sums and PV(last); returns O and the lane row sum."""
    l_run, o = fx.Float32(l_run), [fx.Vector(x) for x in o]
    res = {}
    _drain(_gen_part1(env, fx.Vector(s_last), rsv, res))
    l_run = l_run + res["sum"]
    pw = [None] * 4
    for s in range(4):
        _drain(_gen_pack(res["p"], s, pw))
    # The V(last) writes landed; the next task's id and Q loads stay in flight.
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    o = _pv_last(env, pw, o)
    # Every wave finished its V reads before the output shuffle reuses LDS.
    _stage_end()
    return o, l_run


def _store_out(env, o, inv):
    """O * inv -> BF16 rows through the wave's 16 KiB of LDS, then 16-byte row stores."""
    base = env.wave * 16384
    n, h = env.lane & 31, env.lane >> 5
    for mb in range(8):
        for k in range(4):
            words = _scaled_bf16([o[mb][4 * k + r] for r in range(4)], inv)
            chunk = 4 * mb + k
            offset = (
                base + n * 512 + ((chunk ^ (n & 7)) << 4) + ((h ^ ((n >> 3) & 1)) << 3)
            )
            _write_lds(env.lds, offset, (words[0], words[1]))
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    for i in range(16):
        r, c = 2 * i + (env.lane >> 5), env.lane & 31
        w = _load_lds(env.lds, base + r * 512 + ((c ^ (r & 7)) << 4))
        # Bit 3 of r = 2i + (lane >> 5) is bit 2 of i: the half swap is static.
        swap = (i >> 2) & 1
        words = fx.Vector.from_elements([w[j ^ (2 * swap)] for j in range(4)], fx.Int32)
        row = env.row_offset + env.wave * 32 + r
        query, head = row // env.gp, row % env.gp
        valid = (query < env.qvalid) & (query < env.bq) & (head < env.group)
        offset = (query * env.heads + head) * (D * 2) + c * 16
        _buffer_store_words(words, env.gout, valid.select(offset, env.q_oob))


def _store_whole(env, o, l_run, v_scale, m_run=None):
    """O / l (times the V scale) as BF16; with m_run also the rows' log-sum-exp
    m + log2(l) of S * c (log2 units), which the residual direct merges with."""
    total = l_run + fx.gpu.shuffle_xor(l_run, 32, 64)
    inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    if v_scale is not None:
        inv = inv * v_scale
    if m_run is not None:
        lse = fx.Float32(m_run) + prim.log2(total)
        _buffer_store_words(
            fx.Vector.from_elements([_as_i32(lse)], fx.Int32), env.glse, env.lse_lane
        )
    _store_out(env, o, inv)


def _partial_offset(env, part):
    return (env.slot * SPLIT_PARTS + part) * PARTIAL_BYTES


def _store_partial(env, o, m_run, l_run):
    """This part's O^T fragments, m and the lane's l (device-scope stores: the merging CTA may
    run on another XCD)."""
    base = _partial_offset(env, env.part)
    lane_o = base + env.wave * (64 * 128 * 4) + env.lane * 16
    for mb in range(8):
        for k in range(4):
            words = [_as_i32(o[mb][4 * k + r]) for r in range(4)]
            _buffer_store_words(
                fx.Vector.from_elements(words, fx.Int32),
                env.gpart,
                lane_o + (4 * mb + k) * 1024,
                SC_SYSTEM,
            )
    lane_ml = base + PARTIAL_ML + env.wave * 512 + env.lane * 4
    for i, value in enumerate((m_run, l_run)):
        _buffer_store_words(
            fx.Vector.from_elements([_as_i32(value)], fx.Int32),
            env.gpart,
            lane_ml + 256 * i,
            SC_SYSTEM,
        )


def _merge_out(env, o, m_run, l_run, v_scale):
    """The part counting last: its O, m and l in registers, the other part's record loaded
    (all loads in flight at once). O = O0 * 2^(m0 - m) + O1 * 2^(m1 - m) with m the larger
    (part 0 first, so either part computes the same output), l likewise; then the output of
    an unsplit task."""
    other = SPLIT_PARTS - 1 - env.part
    base = _partial_offset(env, other)
    lane_ml = base + PARTIAL_ML + env.wave * 512 + env.lane * 4
    lane_o = base + env.wave * (64 * 128 * 4) + env.lane * 16
    loaded = [
        _buffer_words(env.gpart, lane_o + (4 * mb + k) * 1024, 4, SC_SYSTEM)
        for mb in range(8)
        for k in range(4)
    ]
    m_other = _as_f32(_buffer_words(env.gpart, lane_ml, 1, SC_SYSTEM)[0])
    l_other = _as_f32(_buffer_words(env.gpart, lane_ml + 256, 1, SC_SYSTEM)[0])
    first = env.part == 0
    m0, m1 = first.select(m_run, m_other), first.select(m_other, m_run)
    top = _maximum(m0, m1)
    # A part without a key of the row has m = FLOOR: weight 0 (both: 1, l = 0, output 0).
    w0, w1 = _exp(m0 - top), _exp(m1 - top)
    l0, l1 = first.select(l_run, l_other), first.select(l_other, l_run)
    l_sum = _fma(l1, w1, l0 * w0)
    total = l_sum + fx.gpu.shuffle_xor(l_sum, 32, 64)
    inv = (total > 0.0).select(fx.Float32(1.0) / total, fx.Float32(0.0))
    if v_scale is not None:
        inv = inv * v_scale
    merged = []
    for mb in range(8):
        values = []
        for i in range(16):
            mine = fx.Float32(fx.Vector(o[mb])[i])
            theirs = _as_f32(loaded[4 * mb + i // 4][i % 4])
            a, b = first.select(mine, theirs), first.select(theirs, mine)
            values.append(_fma(b, w1, a * w0))
        merged.append(fx.Vector.from_elements(values, fx.Float32))
    _store_out(env, merged, inv)


def _lane_env(lds, per_token, slotted, hk):
    """Per-lane constants every task shares."""
    tid = fx.Int32(fx.thread_idx.x)
    wave, lane = _uniform(tid >> 6), tid & 63
    n, h = lane & 31, lane >> 5
    # V^T writes: wave (q, h) = (wave >> 1, wave & 1) fills chunk 2h + q of every dim, the
    # tokens of block slots vslot + 0, 2, 4, 6; K load j of the wave holds the keys of slot
    # vslot + 2j: key 4 * vslot + 8j + (lane >> 4), chunk lane & 15.
    vq, vh = wave >> 1, wave & 1
    vslot = 8 * vq + vh
    return {
        "lds": lds,
        "lane": lane,
        "wave": wave,
        "n": n,
        "h": h,
        "per_token": per_token,
        "slotted": slotted,
        # Constant buffer records (whole tiles); the trace's count of LDS operations.
        "full": False,
        "lgkm": 0,
        "tok_bytes": hk * D,
        "scale_step": hk * 4,
        "stok": lane & 3,
        # Membership bits of the other key half (h): value (b, i) uses bits 8b + 4h + 0..3.
        "foreign_bits": (h != 0).select(fx.Int32(0x0F0F0F0F), fx.Int32(-0x0F0F0F10)),
        "k_addr": [_k_offset(n, 2 * c + h) for c in range(8)],
        "v_addr": [
            [_vt_chunk(n + 64 * hi, 2 * h + q) - 4096 * hi for q in range(2)]
            for hi in range(2)
        ],
        "v_wbase": [_vt_chunk(4 * lane + j, 2 * vh + vq) for j in range(4)],
        "vslot": vslot,
        "v_lane": lane * 4,
        "k_wbase": [
            _k_offset(4 * vslot + 8 * j + (lane >> 4), lane & 15) for j in range(4)
        ],
        "k_lane": (lane >> 4) * (hk * D) + (lane & 15) * 16,
        "sj": 2 * ((lane & 15) >> 2) + h,
    }


def _sload(tensor, index):
    """One int32 of a tensor at a wave-uniform index by a scalar load (SMEM, lgkmcnt): it does
    not wait behind the vector memory counter, which also counts the output stores."""
    # readfirstlane marks the result uniform for LLVM's divergence analysis (no-op on an SGPR):
    # descriptors and the persistent loop built from it stay scalar.
    return _uniform(fx.rocdl.s_buffer_load(_buffer(tensor, 0x7FFFFFFF), index))


def _decode(ORDER, ACTIVE, work, limit, per_tile, split=1):
    """Order entry of a persistent slot and whether it runs; past the order or routed direct:
    idle. K3 orders each CTA's slots by decreasing cost, so its runnable tasks come first.
    With split > 1 an entry is (task << 8) | (slot << 2) | (part << 1) | split."""
    inside = work < limit
    mapped = inside.select(_sload(ORDER, _min(work, limit - 1)), fx.Int32(-1))
    task = mapped >> 8 if split > 1 else mapped
    tile = (mapped >= 0).select(task // per_tile, fx.Int32(0))
    return mapped, (mapped >= 0).select(_sload(ACTIVE, tile), fx.Int32(0))


def _task_scalars(T, mapped, enabled, HK, BQ, GP, slotted, split=1):
    """A task's tile, KV head, row slice and plan entries (scalar loads); an idle slot (enabled
    0) gets no rows and no keys, so its loads read nothing. A part of a split task keeps its
    key range: N64 steps [first, end) of the tile's list, part 0 the first ceil(steps / 2).
    """
    slices = (BQ * GP + 127) // 128
    raw = (mapped >= 0).select(mapped, fx.Int32(0))
    task = raw >> 8 if split > 1 else raw
    on = enabled != 0
    tile_id = task // (HK * slices)
    sc = {
        "tile_id": tile_id,
        "hkv": task % HK,
        "row_offset": ((task // HK) % slices) * 128,
        "q0": _sload(T.META, tile_id * 5),
        "qvalid": on.select(_sload(T.META, tile_id * 5 + 1), fx.Int32(0)),
        "k0": _sload(T.META, tile_id * 5 + 2),
        "kv_len": on.select(_sload(T.META, tile_id * 5 + 3), fx.Int32(0)),
        "count": on.select(_sload(T.COUNTS, tile_id), fx.Int32(0)),
        "tail": _sload(T.TAILS, tile_id) if slotted else fx.Int32(-1),
        "first": fx.Int32(0),
    }
    if split > 1:
        parted = (raw & 1) != 0
        part = (raw >> 1) & 1
        steps = (sc["count"] + 15) >> 4
        half = (steps + 1) >> 1
        first = (parted & (part == 1)).select(half, fx.Int32(0))
        end = (parted & (part == 0)).select(half, steps)
        sc.update(
            count=_min(sc["count"], end * 16) - first * 16,
            first=first,
            split=parted.select(fx.Int32(1), fx.Int32(0)),
            slot=(raw >> 2) & (SPLIT_SLOTS - 1),
            part=part,
        )
    return sc


def _task_env(lane, T, sc, H, HK, NQ, CAP, POOL, BQ, GP):
    """One task's rows, key list and buffers from its scalars."""
    group = H // HK
    tile_id, hkv, row_offset = sc["tile_id"], sc["hkv"], sc["row_offset"]
    q0, qvalid, k0, kv_len, count = (
        sc["q0"],
        sc["qvalid"],
        sc["k0"],
        sc["kv_len"],
        sc["count"],
    )
    n, h = lane["n"], lane["h"]
    row = row_offset + lane["wave"] * 32 + n
    query, head = row // GP, row % GP
    valid = (query < qvalid) & (query < BQ) & (head < group)
    # Byte offsets are unsigned 32-bit: Q and the scales (each request's or the pool's) stay below
    # 4 GiB; K/V are read per block (_kv_block).
    q_rows = fx.Uint32(NQ - q0) * H - hkv * group
    q_base = fx.Int64(q0) * H + hkv * group
    q8_extent = q_rows * D
    q8_oob = fx.Int32(q8_extent)
    tok_bytes = lane["tok_bytes"]
    if fx.const_expr(lane["slotted"]):
        kv_rows = POOL
        kv_offset = fx.Int64(hkv * D)
        scale_extent = (fx.Uint32(POOL) * HK - hkv) * 4
        scale_offset = fx.Int64(hkv)
        final = (sc["tail"] >= 0).select(sc["tail"] >> 2, fx.Int32(-1))
    else:
        kv_rows = kv_len
        kv_offset = fx.Int64(k0) * (HK * D) + hkv * D
        scale_extent = (fx.Uint32(kv_len) * HK - hkv) * 4
        scale_offset = fx.Int64(k0) * HK + hkv
        # Request-local blocks: a final partial block is kv_len >> 2.
        final = kv_len >> 2
    env = _Env(**lane)
    env.__dict__.update(
        valid=valid,
        count=count,
        last=(count + 15) // 16 - 1,
        final=final,
        tail_len=kv_len & 3,
        gblocks=_buffer(
            fx.make_view(
                fx.get_iter(T.BLOCKS) + tile_id * CAP + sc["first"] * 16,
                fx.make_layout(CAP, 1),
            ),
            count * 4,
        ),
        gmask=_buffer(
            fx.make_view(
                fx.get_iter(T.MEMBERS) + (tile_id * (CAP // 16) + sc["first"]) * BQ * 4,
                fx.make_layout((CAP // 16) * BQ * 4, 1),
            ),
            (CAP // 16 - sc["first"]) * BQ * 8,
        ),
        mask_lane=_min(query, fx.Int32(BQ - 1)) * 8,
        mask_step=BQ * 8,
        gq8=_buffer(
            fx.make_view(fx.get_iter(T.Q8) + q_base * D, fx.make_layout(q8_extent, 1)),
            q8_extent,
        ),
        q8_lane=valid.select((query * H + head) * D + h * 16, q8_oob),
        gqs=_buffer(
            fx.make_view(fx.get_iter(T.QS) + q_base, fx.make_layout(q_rows, 1)),
            q_rows * 4,
        ),
        qs_lane=valid.select((query * H + head) * 4, fx.Int32(q_rows * 4)),
        # Row log-sum-exp for the residual direct merge (QS layout), from the first half.
        glse=_buffer(
            fx.make_view(fx.get_iter(T.LSE) + q_base, fx.make_layout(q_rows, 1)),
            q_rows * 4,
        ),
        lse_lane=(valid & (h == 0)).select(
            (query * H + head) * 4, fx.Int32(q_rows * 4)
        ),
        gout=_buffer(
            fx.make_view(fx.get_iter(T.OUT) + q_base * D, fx.make_layout(q8_extent, 1)),
            q8_extent * 2,
        ),
        q_oob=fx.Int32(q8_extent * 2),
        k_base=fx.Int64(fx.ptrtoint(fx.get_iter(T.K))) + kv_offset,
        v_base=fx.Int64(fx.ptrtoint(fx.get_iter(T.V))) + kv_offset,
        partial=(kv_len & 3) * tok_bytes,
        # First block wholly past the scales' extent.
        id_oob=(kv_rows + 3) >> 2,
        gks=_buffer(
            fx.make_view(
                fx.get_iter(T.KS) + scale_offset, fx.make_layout(scale_extent, 1)
            ),
            scale_extent,
        ),
        gvs=_buffer(
            fx.make_view(
                fx.get_iter(T.VS) + scale_offset, fx.make_layout(scale_extent, 1)
            ),
            scale_extent,
        ),
        scale_oob=fx.Int32(scale_extent),
        row_offset=row_offset,
        qvalid=qvalid,
        bq=BQ,
        gp=GP,
        group=group,
        heads=H,
    )
    if "split" in sc:
        env.split, env.slot, env.part = sc["split"], sc["slot"], sc["part"]
        env.gpart = _buffer(T.PARTIALS, SPLIT_SLOTS * SPLIT_PARTS * PARTIAL_BYTES)
    return env


def _prefetch_ids(env):
    """Ids and masks of tiles 0-2, FP8 Q words and the q scale of a task."""
    idl = [_load_idl(env, fx.Int32(i)) for i in range(3)]
    return {
        "idl": idl,
        "w0": _load_mask(env, fx.Int32(0)),
        "q8": [_buffer_words(env.gq8, env.q8_lane + 32 * c, 4) for c in range(8)],
        "qs": _as_f32(_buffer_words(env.gqs, env.qs_lane, 1)[0]),
    }


def _prefetch_kv(env, a):
    """K(0..2), V(0) and scales(0..1) of a task once its ids landed; the V block ids of
    tiles 1 and 2 (from their K loads)."""
    fixed = [_fix_ids(env, a["idl"][i], fx.Int32(i)) for i in range(2)]
    sids = [(_bperm(fixed[i], env.sj), _bperm(fixed[i], env.sj + 8)) for i in range(2)]
    vdata, k0w, k1w, k2w, sc0, sc1 = [], [], [], [], [], []
    ids = [[], [], []]
    _drain(_gen_k_loads(env, a["idl"][0], fx.Int32(0), k0w, ids[0]))
    _drain(_gen_v_loads(env, ids[0], fx.Int32(0), vdata))
    _drain(_gen_k_loads(env, a["idl"][1], fx.Int32(1), k1w, ids[1]))
    _drain(_gen_scales(env, sids[0], sc0))
    _drain(_gen_scales(env, sids[1], sc1))
    _drain(_gen_k_loads(env, a["idl"][2], fx.Int32(2), k2w, ids[2]))
    return {
        "vdata": vdata,
        "k0": k0w,
        "k1": k1w,
        "k2": k2w,
        "sc0": sc0,
        "sc1": sc1,
        "ids": ids[1] + ids[2],
    }


def _pack_pre(a, b):
    """The loop-carried prefetch of the next task as six vectors."""
    misc = [_as_i32(a["qs"]), *a["w0"], a["idl"][2]]
    misc += [_as_i32(x) for x in b["sc0"] + b["sc1"]]
    misc += b["ids"]
    words = [w[j] for w in a["q8"] for j in range(4)]
    return (
        fx.Vector.from_elements([fx.Int32(w) for w in words], fx.Int32),
        fx.Vector.from_elements([fx.Int32(w) for w in misc], fx.Int32),
        fx.Vector.from_elements(b["k0"], fx.Int32),
        fx.Vector.from_elements(b["k1"], fx.Int32),
        fx.Vector.from_elements(b["k2"], fx.Int32),
        fx.Vector.from_elements([w[j] for w in b["vdata"] for j in range(4)], fx.Int32),
    )


def _unpack_pre(q8, misc, k0, k1, k2, v0):
    q8, misc, v0 = fx.Vector(q8), fx.Vector(misc), fx.Vector(v0)
    return {
        "q_words": [
            fx.Vector.from_elements([q8[4 * c + j] for j in range(4)], fx.Int32)
            for c in range(8)
        ],
        "qs": _as_f32(misc[0]),
        "w0": (misc[1], misc[2]),
        "idl2": misc[3],
        "sc0": [_as_f32(misc[4 + i]) for i in range(4)],
        "sc1": [_as_f32(misc[8 + i]) for i in range(4)],
        "ids": tuple(_uniform(misc[12 + j]) for j in range(8)),
        "k0": [fx.Vector(k0)[i] for i in range(16)],
        "k1": [fx.Vector(k1)[i] for i in range(16)],
        "k2": fx.Vector(k2),
        "vdata": [
            fx.Vector.from_elements([v0[4 * i + j] for j in range(4)], fx.Int32)
            for i in range(4)
        ],
    }


@flyc.jit
def _task(
    Q8,
    QS,
    K,
    V,
    KS,
    VS,
    OUT,
    META,
    BLOCKS,
    MEMBERS,
    COUNTS,
    TAILS,
    ORDER,
    ACTIVE,
    PARTIALS,
    COUNTERS,
    LSE,
    lds,
    mapped,
    work,
    limit,
    grid,
    pre_q8,
    pre_misc,
    pre_k0,
    pre_k1,
    pre_k2,
    pre_v0,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    POOL: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    SCALE_LOG2: fx.Float32,
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[int],
    RESIDUAL: fx.Constexpr[bool],
):
    """One task (128 (query, head) rows of a union tile over its compact key list, or one key
    range of it) from its prefetched first tiles; prefetches the next slot's under its
    epilogue and output."""
    T = _Env(
        Q8=Q8,
        QS=QS,
        K=K,
        V=V,
        KS=KS,
        VS=VS,
        OUT=OUT,
        META=META,
        BLOCKS=BLOCKS,
        MEMBERS=MEMBERS,
        COUNTS=COUNTS,
        TAILS=TAILS,
        PARTIALS=PARTIALS,
        LSE=LSE,
    )
    lane = _lane_env(lds, PER_TOKEN, SLOTTED, HK)
    env = _task_env(
        lane,
        T,
        _task_scalars(T, mapped, fx.Int32(1), HK, BQ, GP, SLOTTED, SPLIT),
        H,
        HK,
        NQ,
        CAP,
        POOL,
        BQ,
        GP,
    )
    pre = _unpack_pre(pre_q8, pre_misc, pre_k0, pre_k1, pre_k2, pre_v0)

    # ---- Prologue: K(0), K(1), V(0) to LDS; QK(0) and the exact max ----
    c_scale = pre["qs"] * SCALE_LOG2
    if fx.const_expr(not PER_TOKEN):
        c_scale = c_scale * fx.Float32(KS[0])
    env.q_words, env.c_scale = pre["q_words"], c_scale
    _store_v_now(env, pre["vdata"], 0)
    _drain(_gen_k_writes(env, pre["k0"], 0))
    _drain(_gen_k_writes(env, pre["k1"], 1))
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    s_acc0 = _qk_simple(env, 0)
    # The next slot's task (scalar loads) resolves under QK(0) and the first max.
    per_tile = HK * ((BQ * GP + 127) // 128)
    next_mapped, next_enabled = _decode(
        ORDER, ACTIVE, work + grid, limit, per_tile, SPLIT
    )
    s_cur, row_max, rsv0 = _first_max(env, s_acc0, pre["sc0"], pre["w0"])
    # Every wave read K(0) before phase 1 writes K(2) over it.
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    kpre = [_k_read(env, 1, *K_ORDER[i]) for i in range(KPRE)]
    zeros = tuple(fx.Vector.filled(16, 0.0, fx.Float32) for _ in range(8))
    if fx.const_expr(not PER_TOKEN):
        rsv0 = [fx.Float32(0.0), fx.Float32(0.0)]
    sc1 = pre["sc1"]
    # Tile 0 is final (exact max, masked): phase 1's fix-up passes it through.
    ninf = fx.Float32(-math.inf)
    state = (
        (
            s_cur,
            ninf,
            ninf,
            row_max,
            fx.Float32(0.0),
        )
        + zeros
        + (rsv0[0], rsv0[1], sc1[0], sc1[1], sc1[2], sc1[3])
        + (pre["k2"], *pre["ids"], pre["idl2"])
        + tuple(kpre)
        + tuple(fx.Vector.filled(4, 0, fx.Int32) for _ in range(4 - KPRE))
    )

    def finish(s_prev, local, other, m_run, l_run, o):
        """Tile t-1's lazy max and rescale of O, l and its S'' (Part 2 masked it)."""
        m_run = fx.Float32(m_run)
        tile_max, ballot = _lazy_ballot(local, other, m_run == fx.Float32(_MAX_FLOOR))
        o, l_run, s_prev, m_run = _lazy_max(
            o, l_run, fx.Vector(s_prev), m_run, ballot, lambda: tile_max
        )
        return {"o": o, "l_run": l_run, "s_prev": s_prev, "m_run": m_run}

    @flyc.jit
    def phase(
        s_prev,
        local_prev,
        other_prev,
        m_run,
        l_run,
        o0,
        o1,
        o2,
        o3,
        o4,
        o5,
        o6,
        o7,
        rsvp0,
        rsvp1,
        skc0,
        skc1,
        svc0,
        svc1,
        kflat,
        va0,
        va1,
        va2,
        va3,
        vb0,
        vb1,
        vb2,
        vb3,
        idl1,
        kp0,
        kp1,
        kp2,
        kp3,
        t,
        CUR: fx.Constexpr[int],
        PREV: fx.Constexpr[int],
        FIRST: fx.Constexpr[bool],
        FULL: fx.Constexpr[bool],
    ):
        env.full = FULL
        kflat = fx.Vector(kflat)
        o_in = [fx.Vector(x) for x in (o0, o1, o2, o3, o4, o5, o6, o7)]
        st = {
            "rsv_prev": [fx.Float32(rsvp0), fx.Float32(rsvp1)],
            "sk": [fx.Float32(skc0), fx.Float32(skc1)],
            "sv": [fx.Float32(svc0), fx.Float32(svc1)],
            "kdata": [kflat[i] for i in range(16)],
            # V block ids of tiles t and t + 1 (from their K loads).
            "vids": tuple(_uniform(fx.Int32(v)) for v in (va0, va1, va2, va3)),
            "vnext": tuple(_uniform(fx.Int32(v)) for v in (vb0, vb1, vb2, vb3)),
            "idl1": fx.Int32(idl1),
            "kpre": [fx.Vector(x) for x in (kp0, kp1, kp2, kp3)[:KPRE]],
        }
        t = fx.Int32(t)

        def fixup():
            if fx.const_expr(FIRST):
                # Tile 0 is final (exact max, masked): no ballot, no rescale, no O yet.
                return {
                    "o": None,
                    "l_run": fx.Float32(l_run),
                    "s_prev": fx.Vector(s_prev),
                    "m_run": fx.Float32(m_run),
                }
            return finish(
                s_prev,
                local_prev,
                other_prev,
                m_run,
                l_run,
                o_in,
            )

        s_acc, res, mem, pw, vreads, fixed = _part1(env, st, t, CUR, PREV, fixup)
        st.update(o=fixed["o"], m_run=fixed["m_run"])
        o, ctx, out, kpre = _part2(
            env, st, res, s_acc, mem, pw, vreads, t, CUR, PREV, FIRST
        )
        row_sum = fixed["l_run"] + res["sum"]
        env.full = False
        rsv = ctx["rsv"] if PER_TOKEN else [fx.Float32(0.0), fx.Float32(0.0)]
        sc = out["scales"]
        pads = [fx.Vector.filled(4, 0, fx.Int32)] * (4 - KPRE)
        return (
            fx.Vector.from_elements(ctx["s_cur"], fx.Float32),
            ctx["local"],
            ctx["other"],
            fixed["m_run"],
            row_sum,
            *o,
            rsv[0],
            rsv[1],
            sc[0],
            sc[1],
            sc[2],
            sc[3],
            fx.Vector.from_elements(out["kdata"], fx.Int32),
            *st["vnext"],
            *out["ids"],
            mem["idl2"],
            *kpre,
            *pads,
        )

    key_tiles = env.last + 1
    if key_tiles > 1:
        state = phase(*state, fx.Int32(1), 1, 0, True, False)
    # Phases t <= last - 3 load K(t + 2) and V(t) of whole tiles only.
    for t in range(fx.Int32(2), key_tiles - 4, fx.Int32(2)):
        state = phase(*state, t, 0, 1, False, True)
        state = phase(*state, fx.Int32(t) + 1, 1, 0, False, True)
    exact = 2 + 2 * (env.last >= 4).select((env.last - 4) >> 1, fx.Int32(0))
    for t in range(exact, key_tiles - 1, fx.Int32(2)):
        state = phase(*state, t, 0, 1, False, False)
        state = phase(*state, fx.Int32(t) + 1, 1, 0, False, False)
    if (key_tiles > 2) & ((key_tiles & 1) == 1):
        state = phase(*state, env.last, 0, 1, False, False)

    # ---- Epilogue and output; the next slot's loads issue under them ----
    next_scalars = _task_scalars(
        T, next_mapped, next_enabled, HK, BQ, GP, SLOTTED, SPLIT
    )
    nenv = _task_env(lane, T, next_scalars, H, HK, NQ, CAP, POOL, BQ, GP)
    a = _prefetch_ids(nenv)
    last = finish(*state[:5], state[5:13])
    o, l_run = _epilogue(
        env,
        last["s_prev"],
        last["l_run"],
        last["o"],
        [fx.Float32(state[13]), fx.Float32(state[14])],
    )
    b = _prefetch_kv(nenv, a)
    v_scale = None if PER_TOKEN else fx.Float32(VS[0])
    if fx.const_expr(SPLIT > 1):
        if env.split != 0:
            # Both parts store their records; the one counting last merges them.
            _store_partial(env, o, last["m_run"], l_run)
            fx.rocdl.s_waitcnt(vmcnt=0)
            _stage_end()
            if fx.Int32(fx.thread_idx.x) == 0:
                done = fx.atomic_add(
                    fx.get_iter(COUNTERS) + env.slot, fx.Int32(1), syncscope="agent"
                )
                _write_lds(env.lds, 0, (done, done))
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            _stage_end()
            done = _uniform(fx.Int32(_load_lds(env.lds, 0)[0]))
            # Every wave read the count before the output shuffle reuses LDS.
            _stage_end()
            if done == SPLIT_PARTS - 1:
                _merge_out(env, o, last["m_run"], l_run, v_scale)
        else:
            _store_whole(env, o, l_run, v_scale)
    else:
        _store_whole(env, o, l_run, v_scale, last["m_run"] if RESIDUAL else None)
    # The next task's K/V writes reuse the LDS of every wave's output shuffle.
    fx.rocdl.s_waitcnt(lgkmcnt=0)
    _stage_end()
    return (next_mapped, next_enabled) + _pack_pre(a, b)


@flyc.kernel(name="sparse_prefill_union_fp8_d256", known_block_size=[THREADS, 1, 1])
def _kernel(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    TAILS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    PARTIALS: fx.Tensor,
    COUNTERS: fx.Tensor,
    LSE: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    POOL: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Int32,
    GRID: fx.Int32,
    SCALE_LOG2: fx.Float32,
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[int],
    RESIDUAL: fx.Constexpr[bool],
):
    allocator = fx.SharedAllocator()
    lds = (
        allocator.allocate(fx.Array[fx.Int8, LDS_BYTES, 16])
        .peek()
        .view(fx.make_layout(LDS_BYTES, 1))
    )
    T = _Env(
        Q8=Q8,
        QS=QS,
        K=K,
        V=V,
        KS=KS,
        VS=VS,
        OUT=OUT,
        META=META,
        BLOCKS=BLOCKS,
        MEMBERS=MEMBERS,
        COUNTS=COUNTS,
        TAILS=TAILS,
        PARTIALS=PARTIALS,
        LSE=LSE,
    )
    per_tile = HK * ((BQ * GP + 127) // 128)
    limit = ((TASKS + GRID - 1) // GRID) * GRID
    work = fx.Int32(fx.block_idx.x)
    mapped, enabled = _decode(ORDER, ACTIVE, work, limit, per_tile, SPLIT)
    first = _task_env(
        _lane_env(lds, PER_TOKEN, SLOTTED, HK),
        T,
        _task_scalars(T, mapped, enabled, HK, BQ, GP, SLOTTED, SPLIT),
        H,
        HK,
        NQ,
        CAP,
        POOL,
        BQ,
        GP,
    )
    ids = _prefetch_ids(first)
    pre = _pack_pre(ids, _prefetch_kv(first, ids))
    while enabled != 0:
        result = _task(
            Q8,
            QS,
            K,
            V,
            KS,
            VS,
            OUT,
            META,
            BLOCKS,
            MEMBERS,
            COUNTS,
            TAILS,
            ORDER,
            ACTIVE,
            PARTIALS,
            COUNTERS,
            LSE,
            lds,
            mapped,
            work,
            limit,
            GRID,
            *pre,
            H,
            HK,
            NQ,
            CAP,
            POOL,
            BQ,
            GP,
            SCALE_LOG2,
            PER_TOKEN,
            SLOTTED,
            SPLIT,
            RESIDUAL,
        )
        work = work + GRID
        mapped, enabled, pre = result[0], result[1], result[2:]


@flyc.jit
def _launch(
    Q8: fx.Tensor,
    QS: fx.Tensor,
    K: fx.Tensor,
    V: fx.Tensor,
    KS: fx.Tensor,
    VS: fx.Tensor,
    OUT: fx.Tensor,
    SCALE_LOG2: fx.Float32,
    META: fx.Tensor,
    BLOCKS: fx.Tensor,
    MEMBERS: fx.Tensor,
    COUNTS: fx.Tensor,
    TAILS: fx.Tensor,
    ACTIVE: fx.Tensor,
    ORDER: fx.Tensor,
    PARTIALS: fx.Tensor,
    COUNTERS: fx.Tensor,
    LSE: fx.Tensor,
    H: fx.Constexpr[int],
    HK: fx.Constexpr[int],
    NQ: fx.Int32,
    CAP: fx.Int32,
    POOL: fx.Int32,
    BQ: fx.Constexpr[int],
    GP: fx.Constexpr[int],
    TASKS: fx.Int32,
    GRID: fx.Int32,
    PER_TOKEN: fx.Constexpr[bool],
    SLOTTED: fx.Constexpr[bool],
    SPLIT: fx.Constexpr[int],
    RESIDUAL: fx.Constexpr[bool],
    stream: fx.Stream,
):
    if GRID > 0:
        _kernel(
            Q8,
            QS,
            K,
            V,
            KS,
            VS,
            OUT,
            META,
            BLOCKS,
            MEMBERS,
            COUNTS,
            TAILS,
            ACTIVE,
            ORDER,
            PARTIALS,
            COUNTERS,
            LSE,
            H,
            HK,
            NQ,
            CAP,
            POOL,
            BQ,
            GP,
            TASKS,
            GRID,
            SCALE_LOG2,
            PER_TOKEN,
            SLOTTED,
            SPLIT,
            RESIDUAL,
            value_attrs=prim.kernel_attrs(),
        ).launch(grid=(GRID, 1, 1), block=(THREADS, 1, 1), stream=stream)


_COMPILED = {}


def _args(inputs, plan, out, stream):
    # Slot-table plans read K/V through K3's pool block table, per block from the pool's base.
    k_scale, v_scale = inputs.k_scale, inputs.v_scale
    return (
        inputs.q8.view(-1),
        inputs.qs.view(-1),
        inputs.k,
        inputs.v,
        k_scale.view(-1),
        v_scale.view(-1),
        out.view(-1),
        inputs.scale_log2,
        plan.metadata.view(-1),
        (plan.pool_blocks if plan.slotted else plan.blocks).view(-1),
        plan.score_masks.view(-1),
        plan.counts.view(-1),
        plan.tails if plan.slotted else plan.counts,
        plan.active,
        plan.task_order,
        plan.partials.view(-1) if plan.split > 1 else plan.counts,
        plan.split_counters if plan.split > 1 else plan.counts,
        plan.lse.view(-1) if plan.core_ratio > 0 else plan.counts,
        inputs.q.shape[1],
        inputs.k.shape[1],
        inputs.q.shape[0],
        plan.block_capacity,
        inputs.k.shape[0],
        plan.query_tile,
        plan.group_padded,
        # Order entries (tasks times key-range parts).
        plan.num_tiles
        * inputs.k.shape[1]
        * ((plan.query_tile * plan.group_padded + 127) // 128)
        * plan.split,
        plan.grid,
        k_scale.numel() > 1,
        plan.slotted,
        plan.split,
        plan.core_ratio > 0,
        stream,
    )


def _compiled(inputs, args, stream):
    # H, HK, BQ, GP, PER_TOKEN, SLOTTED, SPLIT, RESIDUAL; the softmax scale is a kernel
    # argument.
    key = (inputs.q.device, *args[18:20], *args[23:25], *args[27:31])
    compiled = _COMPILED.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm the QSA specialization before graph capture")
        with torch.cuda.device(inputs.q.device), torch.cuda.stream(stream):
            # Four waves of 512 registers (VGPRs and AGPRs) and 64 KiB LDS per CTA: one wave
            # per SIMD. Compiled with an empty grid, so it launches nothing.
            compiled = _COMPILED[key] = flyc.compile[{"waves_per_eu": 1}](
                _launch, *args[:26], 0, *args[27:]
            )
    return compiled


def run(*, inputs, plan, out):
    """FP8 union attention after the FP8 pack launch (which quantizes Q). inputs adds
    k_scale / v_scale (FP32 [N, HK] per token or [1]), the FP8 Q of the pack launch
    (inputs.q8 / inputs.qs) and the FP32 softmax_scale * log2(e) (inputs.scale_log2)."""
    if plan.num_tiles == 0:
        return
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, out, stream)
    with torch.cuda.device(inputs.q.device):
        _compiled(inputs, args, stream)(*args)


def launcher(*, inputs, plan, out):
    """launch(q8, qs, k, v, out, k_scale, v_scale, scale_log2) with flat Q8/QS/out/scales and
    unflattened K/V; other arguments prebuilt. Valid while the plan's scratch bindings and
    stream stay.
    """
    if plan.num_tiles == 0:
        return None
    stream = torch.cuda.current_stream(inputs.q.device)
    args = _args(inputs, plan, out, stream)
    compiled = _compiled(inputs, args, stream)
    tail = args[8:]
    return lambda q8, qs, k, v, o, ks, vs, scale_log2: compiled(
        q8, qs, k, v, ks, vs, o, scale_log2, *tail
    )

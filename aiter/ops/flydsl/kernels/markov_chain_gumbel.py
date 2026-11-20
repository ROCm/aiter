# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# ruff: noqa: B023  # traced helpers read the row of the loop iteration that calls them

"""DSpark's sequential Markov draft sampling: 5 steps, every row in one launch.

For vocabulary 129280 and Markov rank 256 (bf16 logits, fp32 Gumbel draws).
Request r, step s, previous token p (step 0: ``input_ids[anchor_idx[r]]``, then
the token of step s - 1): ``bias = bf16(markov_w1[p] . markov_w2^T)`` (fp32 sum
over the rank), ``y = bf16(base[r, s] + bias)`` is the draft-logits cache row
(pre-temperature) and the token is the lowest column of the maximum of ``y`` at
temperature 0, else of ``y / T + noise`` with vLLM's murmur3 Gumbel draw of
(``seed ^ (s << 32)``, ``sample_pos[r, s] - 1 + 2^30``, vocab id).

Tile math: a workgroup is a 512-column vocabulary tile (253 tiles: one
workgroup per CU of the 256) and all rows; 8 waves, wave w owns the 64 columns
64 w .. 64 w + 63 and keeps their ``markov_w2`` rows (64 x 256 bf16) in
registers across the five steps (K order permuted identically on both MFMA
operands: lane group g owns K 64 g .. 64 g + 63). The bf16 ``E`` rows (one per
request: the previous token's ``markov_w1`` row) sit in LDS. The MFMA is
``w2 x E^T`` per 16 rows: a lane holds one row and four consecutive columns, so
the base loads and cache stores are 8 bytes and the row's noise state is per
lane. A lane computes ``bias`` (fp32 sum, one bf16 rounding),
``y = bf16(base + bias)`` (the draft cache row), ``y / T + noise`` (T 0: ``y``)
and the lowest column of its maximum; the wave, the workgroup and then the 253
tiles reduce it.

Step exchange: a workgroup stores its tile winner of every row (value bits,
column | tag << 17) with system-scope stores, no wait; the tag is
``(epoch * 8 + step + 1) & 0x7FFF``. A lane then loads the winners of a slice of
the 253 tiles for one row with system-scope loads and repeats the loads until
every entry carries the step's tag (the loads are the poll; their order is the
tile order, so a tie keeps the lowest column). The wave slices combine in LDS,
the row's token goes to ``tokens[:, step]`` and LDS, and the next step's ``E``
rows are gathered from ``markov_w1``. While waiting, a workgroup loads the next
step's base logits. Every entry of the buffer, rows past the batch included, is
rewritten each launch and each row-count variant has its own buffer and epoch,
so a tag never matches an entry of an earlier launch. The epoch is word 0 of
the epoch buffer: every workgroup reads it before its first store, workgroup 0
increments it after the last step's winners are in (so after every workgroup
has read it): no resets, no atomics, graph replays reuse the buffers.

The workgroups spin-wait on each other's winners, so all 253 must be resident
at once: the launch needs >= 253 CUs not held by kernels that wait on this one.
A NaN never wins; neither does a +inf logit (its residual ``y - T (y / T)`` is
NaN). A row without a winner (every value NaN or +/-inf) gets token 0, so a
token is always in [0, V).

Memory access uses FlyDSL buffer tensors and ``fx.copy`` (``BufferCopy`` atoms
with the cache-scope operand and soffset as atom state); a masked access moves
its offset past the buffer's records, so the hardware bounds check reads 0 /
drops the store. The few raw ops left say at the call why no 0.5.0 API serves.
"""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

V, R, NSPEC = 129280, 256, 5  # vocabulary, Markov rank, draft steps
TILE = 512  # vocabulary columns of a workgroup

F, I = fx.Float32, fx.Int32
NT = (V + TILE - 1) // TILE  # 253
NWAVE = TILE // 64  # 8
ESTRIDE = R * 2 + 16  # LDS bytes of an E row (padded against bank conflicts)
BIG = 1 << 30
NEG_INF = float("-inf")
LARGE = 0x7FFFFFF0  # records of a buffer whose size the kernel does not bound
# i32-word offset at or past every buffer's records: a load reads 0, a store is dropped
OOB = LARGE // 4
SYS = 0x11  # cache policy sc0 sc1: system scope (the cross-workgroup exchange)
_C = [float(struct.unpack("<f", struct.pack("<f", 1.0 / k))[0]) for k in range(1, 9)]


def _fbits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


_LN2_HI, _LN2_LO = _fbits(0x3F317217), _fbits(0x3377D1CF)


def _i32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


def _log_normal(x):
    """Triton's ``tl.log`` (LLVM's llvm.log.f32 expansion on gfx950) for finite x >= 2^-126."""
    y = fx.math.log2(x, fastmath=fx.FastMathFlags.afn)
    r0 = y * F(_LN2_HI)
    f0 = fx.math.fma(y, F(_LN2_HI), r0 * F(-1.0))
    f1 = fx.math.fma(y, F(_LN2_LO), f0)
    return fx.math.fma(y, F(_LN2_HI), f1)


def _lshr(x, s):
    return (x >> s) & ((1 << (32 - s)) - 1)


def _rotl(x, s):
    return (x << s) | _lshr(x, 32 - s)


def _mix(h, key, key_c1=None):
    key = key * _i32(0xCC9E2D51) if key_c1 is None else key_c1
    key = _rotl(key, 15)
    key = key * _i32(0x1B873593)
    h = h ^ key
    h = _rotl(h, 13)
    return (h << 2) + h + _i32(0xE6546B64)


def _fmix(h):
    h = h ^ _lshr(h, 16)
    h = h * _i32(0x85EBCA6B)
    h = h ^ _lshr(h, 13)
    h = h * _i32(0xC2B2AE35)
    return h ^ _lshr(h, 16)


def _noise(h0, col_c1):
    """fp32 Gumbel noise of the vocab id whose product with 0xCC9E2D51 is ``col_c1`` (vLLM's fp32 draw)."""
    r = _fmix(_mix(h0, None, col_c1) ^ 16)
    u = _lshr(r, 16).to(F) * F(1.52587890625e-05) + ((r & 0xFFFF).to(F) + F(0.5)) * F(
        2.3283064365386963e-10
    )
    p = F(_C[7])
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = fx.math.fma(u, p, F(_C[k - 1]))
    series = (u * F(-1.0)) * p
    one_m = F(1.0) - u
    direct = _log_normal(F(fx.arith.maxnumf(one_m, F(5.960464477539063e-08))))
    l1 = (u < F(0.25)).select(series, direct)
    return _log_normal(l1 * F(-1.0)) * F(-1.0)


def _xor_lane(v, m):
    """Lane ``l ^ m`` (m = 16, 32) of the wave as an i32 (a ``v_permlane{16,32}_swap``)."""
    return I(fx.gpu.shuffle_xor(v, m, 64))


def _comb(v1, i1, v2, i2):
    """Keep (v1, i1) iff v1 > v2 or (v1 == v2 and i1 < i2): the maximum, lowest id first."""
    keep = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return keep.select(v1, v2), keep.select(i1, i2)


def _words(ptr, nbytes, vec):
    """Buffer tensor over the i32 words of global pointer ``ptr`` with ``nbytes``
    records (hardware bounds check); a copy moves ``vec`` words from the flat
    word offset it is sliced at."""
    base = fx.inttoptr(
        fx.PointerType.get(I.ir_type, fx.AddressSpace.Global, 4 * vec),
        fx.Int64(fx.ptrtoint(ptr)),
    )
    return fx.rocdl.make_buffer_tensor(
        fx.make_view(base, fx.make_layout((vec, 1), (1, 1))),
        num_records_bytes=fx.Int64(nbytes),
    )


def _load(view, off, vec, mask=None, cache=0, soffset=0):
    """``vec`` i32 words at word ``off`` (+ the uniform ``soffset`` words); 0 where ``mask`` is False."""
    if mask is not None:
        off = mask.select(off, I(OOB))
    frag = fx.make_rmem_tensor(vec, I)
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(32 * vec, cache), I)
    if soffset:
        fx.copy(atom, fx.slice(view, (None, off)), frag, soffset=I(soffset))
    else:
        fx.copy(atom, fx.slice(view, (None, off)), frag)
    v = Vec(frag.load())
    return I(v[0]) if vec == 1 else v


def _store(view, off, words, mask=None, cache=0):
    """Store the i32 words ``words`` (a list) at word ``off``; nothing where ``mask`` is False."""
    if mask is not None:
        off = mask.select(off, I(OOB))
    frag = fx.make_rmem_tensor(len(words), I)
    frag.store(Vec.from_elements(words, I))
    atom = fx.make_copy_atom(fx.rocdl.BufferCopy(32 * len(words), cache), I)
    fx.copy(atom, frag, fx.slice(view, (None, off)))


def _unpack(w, k):
    """Element k (0..1) of a packed bf16 pair as fp32."""
    return ((w << 16) if k == 0 else (w & -65536)).bitcast(F)


def _row_params(
    idr, tr, sdr, pr, row, rok, i32_idxs, i32_idxe, i32_poss, step, i32_pos_add
):
    """(slot or -1, temperature, murmur prefix h0) of request row ``row`` at draft ``step``: the noise state of its
    draws (``h0`` is the hash after the seed, the position and the salt; only the vocab id is left to mix in).
    """
    slot = _load(idr, row * i32_idxs + step * i32_idxe, 1, mask=rok)
    valid = rok & (slot >= 0)
    slot0 = valid.select(slot, I(0))
    temp = valid.select(_load(tr, slot0, 1).bitcast(F), F(0.0))
    sd = _load(sdr, slot0 * 2, 2)
    seed_hi = I(sd[1]) ^ step  # spec_step_seed
    pos_lo = _load(pr, (row * i32_poss + step) * 2, 1, mask=rok)
    h0 = _mix(_mix(_mix(I(0), I(sd[0])), seed_hi), pos_lo + i32_pos_add)
    return valid.select(slot, I(-1)), temp, h0


HALF = 16  # winners of the tiles a lane loads at once
GB = 4  # E chunks a thread gathers at once


def _sb():
    fx.rocdl.sched_barrier(0)


def _poll_sleep():
    """``s_sleep 1`` that is also a compiler memory barrier, at the top of the poll loop.

    Raw inline asm on purpose: ``fx.rocdl.s_sleep`` (FlyDSL 0.5.0's only sleep)
    has no memory effect, so nothing would stop LLVM from hoisting the poll's
    loop-invariant loads out of the loop, which turns the poll into an endless
    spin on stale values. It gives the same ISA today (the variant was compiled
    and diffed); the clobber keeps that from depending on the optimizer.
    """
    _llvm.inline_asm(None, [], "s_sleep 1", "~{memory}", has_side_effects=True)


def compile_markov_chain(nrf: int):
    NROW = 16 * nrf  # rows of the buffers; the row loop runs over the rows of the call
    SUB = max(1, 64 // NROW)  # lanes per row in the winner reduction
    NSL = NWAVE * SUB  # tile slices of the winner reduction
    TPS = (NT + NSL - 1) // NSL  # tiles a lane takes
    QF = (NT - NSL) // NSL + 1  # slice entries q < QF are valid tiles for every lane
    RG = (NROW + 63) // 64  # row groups of the winner reduction (lane = row)
    nch = NROW * 32 // (64 * NWAVE)  # 16-byte chunks of E a thread gathers
    name = f"markov_chain_v129280_r256_s5_nrf{nrf}"

    @fx.struct
    class Shared:
        e: fx.Array[fx.Int32, NROW * ESTRIDE // 4, 16]
        mv: fx.Array[fx.Int32, NROW * NWAVE]
        mi: fx.Array[fx.Int32, NROW * NWAVE]
        rv: fx.Array[fx.Int32, NROW * NSL + 1]
        ri: fx.Array[fx.Int32, NROW * NSL + 1]
        tok: fx.Array[fx.Int32, NROW + 1]
        prm: fx.Array[fx.Int32, (NSPEC + 1) * NROW * 4 + 4]

    @flyc.kernel(name=name, known_block_size=[64 * NWAVE, 1, 1])
    def kernel(
        arg_w2: fx.Pointer,  # bf16 [V, R]
        arg_w1: fx.Pointer,  # bf16 [V, R]
        arg_ids: fx.Pointer,  # input_ids: token of the anchor index at dword i32_ide * anchor
        i32_ide: fx.Int32,
        arg_anchor: fx.Pointer,  # int64 [n]
        arg_tokens: fx.Pointer,  # int64 [n, i32_ts]
        i32_ts: fx.Int32,
        arg_base: fx.Pointer,  # bf16, row r * NSPEC + step at row stride i32_bs
        i32_bs: fx.Int32,
        arg_cache: fx.Pointer,  # bf16 [slots, NSPEC, V]
        arg_idx: fx.Pointer,  # request slot of (r, step) at dword r * i32_idxs + step * i32_idxe; < 0: no row
        i32_idxs: fx.Int32,
        i32_idxe: fx.Int32,
        arg_pos: fx.Pointer,  # int64 sample position of (r, step) at r * i32_poss + step
        i32_poss: fx.Int32,
        arg_temp: fx.Pointer,  # fp32 [slots]
        arg_seed: fx.Pointer,  # int64 [slots]
        arg_pvi: fx.Pointer,  # int32 [NSPEC, NT, NROW, 2]: tile winners (value bits, column | tag << 17)
        arg_epoch: fx.Pointer,  # int32 [1]
        i32_n: fx.Int32,
        i32_pos_add: fx.Int32,
    ):
        tx, tile = fx.thread_idx.x, fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = I(fx.rocdl.readfirstlane(T.i32, tx // 64))
        nl = lane % 16
        gl = lane // 16
        # E rows: 16-byte chunks, ESTRIDE // 16 per row
        e_lds = smem.e.view(fx.make_layout((4, NROW * ESTRIDE // 16), (1, 4)))
        v0 = tile * TILE + wave * 64
        epr = _words(arg_epoch, 4, 1)
        epoch = _load(epr, 0, 1, cache=SYS)

        # loads in flight together: the tile's markov_w2 rows (registers), the rows' noise state, the anchor tokens
        w2r = _words(arg_w2, V * R * 2, 4)
        wv = [
            [
                _load(w2r, (v0 + vf * 16 + nl) * (R // 2) + gl * 32 + kk * 4, 4)
                for kk in range_constexpr(8)
            ]
            for vf in range_constexpr(4)
        ]
        idr = _words(arg_idx, LARGE, 1)
        tr = _words(arg_temp, LARGE, 1)
        sdr = _words(arg_seed, LARGE, 2)
        pr = _words(arg_pos, LARGE, 1)
        for i in range_constexpr((NSPEC * NROW + 64 * NWAVE - 1) // (64 * NWAVE)):
            pidx = tx + 64 * NWAVE * i
            prow = pidx % NROW
            pstep = pidx // NROW
            sl, tp, h0 = _row_params(
                idr,
                tr,
                sdr,
                pr,
                prow,
                (prow < i32_n) & (pstep < NSPEC),
                i32_idxs,
                i32_idxe,
                i32_poss,
                pstep,
                i32_pos_add,
            )
            po = (pidx < NSPEC * NROW).select(pidx * 4, I((NSPEC + 1) * NROW * 4))
            smem.prm.ptr[po] = sl
            smem.prm.ptr[po + 1] = tp.bitcast(I)
            smem.prm.ptr[po + 2] = h0

        idsr = _words(arg_ids, LARGE, 1)
        anr = _words(arg_anchor, LARGE, 1)
        w1r = _words(arg_w1, V * R * 2, 4)
        toks = []
        for i in range_constexpr(nch):
            row = (tx + (64 * NWAVE) * i) // 32
            a = _load(anr, row * 2, 1, mask=row < i32_n)
            toks.append(_load(idsr, a * i32_ide, 1, mask=row < i32_n))

        def gather_e(toks_, live):
            """E rows from markov_w1 into LDS, GB 16-byte chunks per thread at a time (registers)."""
            for b0 in range_constexpr(0, nch, GB):
                evs = []
                for i in range_constexpr(b0, min(b0 + GB, nch)):
                    row = (tx + (64 * NWAVE) * i) // 32
                    evs.append(
                        _load(
                            w1r,
                            toks_[i] * (R // 2) + (tx % 32) * 4,
                            4,
                            mask=(row < i32_n) & live,
                        )
                    )
                for i in range_constexpr(b0, min(b0 + GB, nch)):
                    row = (tx + (64 * NWAVE) * i) // 32
                    fx.slice(e_lds, (None, row * (ESTRIDE // 16) + tx % 32)).store(
                        evs[i - b0]
                    )
                _sb()

        always = fx.Int32(0) == 0
        gather_e(toks, always)
        br = _words(arg_base, LARGE, 2)
        cr = _words(arg_cache, LARGE, 2)
        pvir = _words(arg_pvi, LARGE, 2)
        tkr = _words(arg_tokens, LARGE, 2)
        c1_lane = (v0 + gl * 4) * _i32(0xCC9E2D51)

        def load_base(s, rf, live):
            brow = (rf * 16 + nl) * NSPEC + s
            return [
                _load(
                    br,
                    (brow * i32_bs + v0 + vf * 16 + gl * 4) // 2,
                    2,
                    mask=(rf * 16 + nl < i32_n) & (v0 + vf * 16 + gl * 4 < V) & live,
                )
                for vf in range_constexpr(4)
            ]

        def row_state(s, rf):
            """(slot or -1, temperature, h0) of this lane's row of fragment rf at step s, from the LDS table."""
            o = (s * NROW + rf * 16 + nl) * 4
            return (
                I(smem.prm.ptr[o]),
                I(smem.prm.ptr[o + 1]).bitcast(F),
                I(smem.prm.ptr[o + 2]),
            )

        def flat_xv(xv):
            return [x for v in xv for x in (v[0], v[1])]

        def unflat_xv(st):
            return [
                Vec.from_elements([I(st[vf * 2]), I(st[vf * 2 + 1])], I)
                for vf in range_constexpr(4)
            ]

        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()  # the table
        xv0 = load_base(0, 0, always)
        fx.rocdl.s_waitcnt(lgkmcnt=0)
        fx.gpu.barrier()  # E rows of step 0

        for iv, state in range(0, NSPEC, 1, init=flat_xv(xv0)):
            s = I(iv)
            carried = unflat_xv(state)
            for rv in range(0, (i32_n + 15) // 16, 1):
                rf = I(rv)
                row = rf * 16 + nl
                ld = load_base(s, rf, rf > 0)
                xr = [
                    Vec.from_elements(
                        [
                            (rf == 0).select(I(carried[vf][0]), I(ld[vf][0])),
                            (rf == 0).select(I(carried[vf][1]), I(ld[vf][1])),
                        ],
                        I,
                    )
                    for vf in range_constexpr(4)
                ]

                def read_e(kk):
                    return Vec(
                        fx.slice(
                            e_lds, (None, row * (ESTRIDE // 16) + gl * 8 + kk)
                        ).load()
                    ).bitcast(fx.BFloat16)

                acc = [Vec.filled(4, 0.0, F) for _ in range_constexpr(4)]
                e_cur = read_e(0)
                for kk in range_constexpr(8):
                    e_next = read_e(kk + 1) if kk < 7 else e_cur
                    for vf in range_constexpr(4):
                        # Raw MFMA: the A operands are markov_w2 rows held in registers
                        # for all five steps, with K permuted per lane group on both
                        # operands; there is no tensor / partition form of that for
                        # fx.make_mma_atom + fx.gemm to take.
                        acc[vf] = Vec(
                            fx.rocdl.mfma_f32_16x16x32_bf16(
                                Vec.make_type(4, F),
                                [
                                    wv[vf][kk].bitcast(fx.BFloat16),
                                    e_cur,
                                    acc[vf],
                                    0,
                                    0,
                                    0,
                                ],
                            )
                        )
                    _sb()
                    e_cur = e_next
                pbs = [
                    Vec(acc[vf].to(fx.BFloat16).bitcast(I)) for vf in range_constexpr(4)
                ]  # typed cast: MFMA hazard
                _sb()
                slot, temp, h0 = row_state(s, rf)
                tz = temp == F(0.0)
                ts = tz.select(F(1.0), temp)
                nmask = tz.select(F(0.0), F(1.0))  # a temperature-0 row takes no noise
                rts = (
                    F(1.0) / ts
                )  # correctly rounded: y / ts below is then correctly rounded too (Markstein)
                bv, bi = F(NEG_INF), I(BIG)
                for vf in range_constexpr(4):
                    col0 = v0 + vf * 16 + gl * 4
                    inv = col0 < V
                    nzr = [
                        _noise(h0, c1_lane + _i32((vf * 16 + k) * 0xCC9E2D51))
                        for k in range_constexpr(4)
                    ]
                    pb = [I(pbs[vf][0]), I(pbs[vf][1])]
                    sums = [
                        _unpack(I(xr[vf][k // 2]), k % 2) + _unpack(pb[k // 2], k % 2)
                        for k in range_constexpr(4)
                    ]
                    ybf = Vec.from_elements(sums, F).to(fx.BFloat16).bitcast(I)
                    py = [I(ybf[0]), I(ybf[1])]
                    _store(
                        cr,
                        ((slot * NSPEC + s) * V + col0) // 2,
                        py,
                        mask=(slot >= 0) & inv,
                    )
                    for k in range_constexpr(4):
                        y = _unpack(py[k // 2], k % 2)
                        q0 = y * rts
                        q1 = fx.math.fma(F(-1.0) * ts, q0, y)
                        val = fx.math.fma(nzr[k], nmask, fx.math.fma(q1, rts, q0))
                        better = val > bv
                        bv = better.select(val, bv)
                        bi = better.select(col0 + k, bi)
                    _sb()
                bv = (v0 < V).select(
                    bv, F(NEG_INF)
                )  # a wave past the vocabulary: all columns invalid
                for m in (16, 32):
                    ov = _xor_lane(bv.bitcast(I), m).bitcast(F)
                    oi = _xor_lane(bi, m)
                    bv, bi = _comb(bv, bi, ov, oi)
                smem.mv.ptr[(rf * 16 + nl) * NWAVE + wave] = bv.bitcast(I)
                smem.mi.ptr[(rf * 16 + nl) * NWAVE + wave] = bi
                _sb()
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            rr_ok = tx < NROW
            rr_s = rr_ok.select(tx, I(0))
            bv = I(smem.mv.ptr[rr_s * NWAVE]).bitcast(F)
            bi = I(smem.mi.ptr[rr_s * NWAVE])
            for w in range_constexpr(1, NWAVE):
                bv, bi = _comb(
                    bv,
                    bi,
                    I(smem.mv.ptr[rr_s * NWAVE + w]).bitcast(F),
                    I(smem.mi.ptr[rr_s * NWAVE + w]),
                )
            tag = (epoch * 8 + (s + 1)) & 0x7FFF
            _store(
                pvir,
                ((s * NT + tile) * NROW + tx) * 2,
                [bv.bitcast(I), (bi & 0x1FFFF) | (tag << 17)],
                mask=rr_ok,
                cache=SYS,
            )
            _sb()
            live = s + 1 < NSPEC
            nxv0 = load_base(s + 1, 0, live)
            _sb()
            for j in range_constexpr(RG):
                rowl = (lane % NROW if SUB > 1 else lane) + 64 * j
                sub = lane // NROW if SUB > 1 else I(0)
                lr = (lane < NROW * SUB) & (rowl < i32_n)
                # slice entry q is tile q * NSL + wave * SUB + sub: a constant offset from the lane's first
                t_first = wave * SUB + sub
                base_dw = lr.select(
                    ((s * NT + t_first) * NROW + rowl) * 2, I(0x1FFFFFFF)
                )
                bv, bi = F(NEG_INF), I(BIG)
                bad = fx.Int64(1)
                while bad != 0:
                    _poll_sleep()
                    bv, bi = F(NEG_INF), I(BIG)
                    miss = I(0)
                    for h in range_constexpr((TPS + HALF - 1) // HALF):
                        qs = list(range(h * HALF, min((h + 1) * HALF, TPS)))
                        ents = [
                            _load(
                                pvir,
                                base_dw,
                                2,
                                mask=(t_first + q * NSL < NT) if q >= QF else None,
                                cache=SYS,
                                soffset=q * NSL * NROW * 2,
                            )
                            for q in qs
                        ]
                        for q, e in zip(qs, ents):
                            ev_, ew_ = I(e[0]).bitcast(F), I(e[1])
                            okq = (
                                t_first + q * NSL < NT
                            )  # only used (and kept) for the entries past QF
                            tdiff = (ew_ & _i32(0xFFFE0000)) ^ (tag << 17)
                            miss = miss | (tdiff if q < QF else okq.select(tdiff, I(0)))
                            better = (ev_ > bv) if q < QF else (okq & (ev_ > bv))
                            bv = better.select(ev_, bv)
                            bi = better.select(ew_ & 0x1FFFF, bi)
                    bad = fx.Int64(fx.rocdl.ballot(T.i64, lr & (miss != 0)))
                rslot = lr.select((wave * SUB + sub) * NROW + rowl, I(NROW * NSL))
                smem.rv.ptr[rslot] = bv.bitcast(I)
                smem.ri.ptr[rslot] = bi
                _sb()
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            rr_n = rr_ok & (tx < i32_n)
            bv = I(smem.rv.ptr[rr_s]).bitcast(F)
            bi = I(smem.ri.ptr[rr_s])
            for w in range_constexpr(1, NSL):
                bv, bi = _comb(
                    bv,
                    bi,
                    I(smem.rv.ptr[w * NROW + rr_s]).bitcast(F),
                    I(smem.ri.ptr[w * NROW + rr_s]),
                )
            bi = (bi < V).select(bi, I(0))  # no winner (NaN or all -inf row): token 0
            _store(tkr, (rr_s * i32_ts + s) * 2, [bi, I(0)], mask=rr_n)
            smem.tok.ptr[rr_ok.select(tx, I(NROW))] = bi
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            gather_e(
                [
                    I(smem.tok.ptr[(tx + 64 * NWAVE * i) // 32])
                    for i in range_constexpr(nch)
                ],
                live,
            )
            fx.rocdl.s_waitcnt(lgkmcnt=0)
            fx.gpu.barrier()
            yield flat_xv(nxv0)
        if (tile == 0) & (tx == 0):
            _store(epr, 0, [epoch + 1], cache=SYS)

    @flyc.jit
    def launch_markov_chain(
        arg_w2: fx.Pointer,
        arg_w1: fx.Pointer,
        arg_ids: fx.Pointer,
        i32_ide: fx.Int32,
        arg_anchor: fx.Pointer,
        arg_tokens: fx.Pointer,
        i32_ts: fx.Int32,
        arg_base: fx.Pointer,
        i32_bs: fx.Int32,
        arg_cache: fx.Pointer,
        arg_idx: fx.Pointer,
        i32_idxs: fx.Int32,
        i32_idxe: fx.Int32,
        arg_pos: fx.Pointer,
        i32_poss: fx.Int32,
        arg_temp: fx.Pointer,
        arg_seed: fx.Pointer,
        arg_pvi: fx.Pointer,
        arg_epoch: fx.Pointer,
        i32_n: fx.Int32,
        i32_pos_add: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_w2,
            arg_w1,
            arg_ids,
            i32_ide,
            arg_anchor,
            arg_tokens,
            i32_ts,
            arg_base,
            i32_bs,
            arg_cache,
            arg_idx,
            i32_idxs,
            i32_idxe,
            arg_pos,
            i32_poss,
            arg_temp,
            arg_seed,
            arg_pvi,
            arg_epoch,
            i32_n,
            i32_pos_add,
        ).launch(grid=(NT, 1, 1), block=(64 * NWAVE, 1, 1), stream=stream)

    return launch_markov_chain

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Single-kernel Gated DeltaNet chunked prefill (gfx1151).

One workgroup owns a (sequence, value head, 64-channel V slice) and walks its
16-token chunks in order. With normalized q/k and cumulative log-decay gc:

    A    = strict_tril(beta_t * (k_t . k_j) * exp(gc_t - gc_j))  [16, 16]
    vnew = solve(I + A, beta * (v - exp(gc) * (k @ S.T)))       [16, BV]
    P    = scale * tril((q @ k.T) * exp(gc_t - gc_j))           [16, 16]
    o    = scale * exp(gc) * (q @ S.T) + P @ vnew
    S    = exp(gc_last) * S + vnew.T @ (k * exp(gc_last - gc))

The state slice S [BV, K] stays in fp32 registers; a bf16 LDS copy feeds
the next chunk's WMMA products, so per-chunk states never reach HBM. The solve
is fp32, and P uses bf16 high/residual products for output precision. Tiny
prompts and weak-decay sequence/heads use fp32 recurrence. Snapshots retain
64-token boundaries.

Lane conventions (wave32, l = lane), for 16x16x16 bf16 WMMA:
  A/B: 16-element row [l % 16, j], replicated in both half-waves
  accumulator C: eight elements [2*i + l//16, l%16]
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from ..kernels_common import LOG2E
from .gdn_fused_prefill import _g_ld, _g_st, _lds_ld, _lds_st

BT = 16
BV = 64
KD = VD = 128
# Padded row pitches in elements: eight BF16 or four FP32 elements shift
# consecutive rows by four LDS banks.
ROW = 136
SROW = 136
PROW = 20
NUM_WAVES = 8
BLOCK = NUM_WAVES * 32
# Near-unit decay amplifies BF16 operand rounding across successive chunks.
# Use FP32 recurrence if any chunk's mean log-decay is >= -0.02.
MIN_MATRIX_DECAY = 0.02


def _exp(x):
    return fx.exp2(x * LOG2E, fastmath="afn")


def _read(ptr, off, dtype=fx.Float32):
    return _lds_ld(ptr, off, 1, dtype)[0]


def _write(ptr, off, value, dtype=fx.Float32):
    _lds_st(ptr, off, fx.Vector.from_elements([value], dtype=dtype), 1, dtype)


def _wmma(a, b, acc):
    """gfx11 wave32 16x16x16 BF16 WMMA with FP32 accumulation."""
    ra = fx.make_rmem_tensor(16, fx.BFloat16)
    rb = fx.make_rmem_tensor(16, fx.BFloat16)
    rc = fx.make_rmem_tensor(8, fx.Float32)
    ra.store(a)
    rb.store(b)
    rc.store(acc)
    atom = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fx.BFloat16, fx.Float32))
    fx.gemm(atom, rc, ra, rb, rc)
    return rc.load()


def _row_fragment(ptr, base, dtype):
    width = 128 // dtype.width
    parts = [_lds_ld(ptr, base + j * width, width, dtype) for j in range(16 // width)]
    return fx.Vector.from_elements(
        [parts[j // width][j % width] for j in range(16)], dtype=dtype
    )


@flyc.jit
def _mm(
    a,
    b,
    ar,
    br,
    ast,
    bst,
    reduction,
    lane,
    adt,
    bdt,
    bfactor=None,
    bfactor_is_vector=False,
    split_a=False,
):
    """16x16 A @ B.T from LDS, optionally scaling B along its K dimension.

    Matrix operands round to BF16 and accumulators remain FP32. With
    split_a, A contributes separate BF16 high and residual products.
    """
    acc = fx.Vector.filled(8, 0.0, fx.Float32)
    for kk, carry in range(0, reduction // 16, init=[acc]):
        ki = fx.Int32(kk)
        acc = fx.Vector(carry[0])
        av = (
            _row_fragment(a, (ar + lane % 16) * ast[0] + ki * 16, adt)
            if ast[1] == 1
            else fx.Vector.from_elements(
                [
                    _read(a, (ar + lane % 16) * ast[0] + (ki * 16 + j) * ast[1], adt)
                    for j in range(16)
                ],
                dtype=adt,
            )
        )
        bv = (
            _row_fragment(b, (br + lane % 16) * bst[0] + ki * 16, bdt)
            if bst[1] == 1
            else fx.Vector.from_elements(
                [
                    _read(b, (br + lane % 16) * bst[0] + (ki * 16 + j) * bst[1], bdt)
                    for j in range(16)
                ],
                dtype=bdt,
            )
        )
        if const_expr(bfactor is not None):
            if const_expr(bfactor_is_vector):
                factors = bfactor
            else:
                factors = fx.Vector.from_elements(
                    [_read(bfactor, ki * 16 + j) for j in range(16)], dtype=fx.Float32
                )
            bv = bv.to(fx.Float32) * factors
        ah, bh = av.to(fx.BFloat16), bv.to(fx.BFloat16)
        if const_expr(split_a):
            al = (av.to(fx.Float32) - ah.to(fx.Float32)).to(fx.BFloat16)
            acc = _wmma(al, bh, acc)
        acc = _wmma(ah, bh, acc)
        result = yield [acc]
    return fx.Vector(result)


def _store_tile(ptr, row, col, pitch, lane, acc):
    for i in range_constexpr(8):
        _write(ptr, (row + 2 * i + lane // 16) * pitch + col + lane % 16, acc[i])


@flyc.jit
def _has_weak_decay(g, bos, eos, sg, head, lane):
    """Screen mean log-decay per chunk, counting only live tokens in a tail.

    A single weak chunk selects FP32 recurrence for the whole sequence/head,
    so rounded matrix updates cannot feed a later weak-decay segment.
    """
    weak = fx.Int32(0)
    for base, carry in range(bos, eos, 32, init=[weak]):
        base = fx.Int32(base)
        weak = fx.Int32(carry[0])
        value = fx.Float32(0.0)
        if base + lane < eos:
            value = _g_ld(g, (base + lane) * sg + head, 1, fx.Float32)[0]
        for shift in (1, 2, 4, 8):
            value = value + value.shuffle_xor(shift, 16)
        count = fx.max(fx.min(eos - (base + (lane // 16) * 16), 16), 0)
        active = (count > 0) & (value >= -MIN_MATRIX_DECAY * count.to(fx.Float32))
        weak = fx.max(weak, active.select(fx.Int32(1), fx.Int32(0)))
        result = yield [weak]
    weak = fx.Int32(result)
    for shift in (1, 2, 4, 8, 16):
        weak = fx.max(weak, weak.shuffle_xor(shift, 32))
    return weak > 0


@flyc.jit
def _recurrent_prefill(
    q,
    k,
    v,
    g,
    beta,
    states,
    out,
    track,
    track_idx,
    has_init,
    seq,
    head,
    kh,
    vs,
    slot,
    bos,
    eos,
    sq,
    sk,
    sv,
    sg,
    sb,
    lane,
    wave,
    H,
    scale,
    HAS_TRACK,
    HAS_INIT_FLAG,
):
    """FP32 rank-one recurrence for tiny or nearly non-decaying sequences.

    A wave owns eight V rows and all 128 K columns. FP32 state remains in
    registers; only the public output/pool writeback rounds to BF16. This
    avoids padded WMMA work for one token and accumulated operand rounding
    for near-unit decay, without intermediate global states.

    Per token: Sd = exp(g) * S, delta = beta * (v - Sd @ k),
    S = Sd + delta[:, None] * k, and o = scale * (S @ q).
    """
    load_init = fx.Int32(1)
    if const_expr(HAS_INIT_FLAG):
        load_init = _g_ld(has_init, seq, 1, fx.Int32)[0]
    values = []
    for j in range_constexpr(BV // NUM_WAVES):
        row = vs + wave * (BV // NUM_WAVES) + j
        value = fx.Vector.filled(4, 0.0, fx.Float32)
        if load_init != 0:
            value = _g_ld(
                states,
                (slot * H + head) * VD * KD + row * KD + lane * 4,
                4,
                fx.BFloat16,
            ).to(fx.Float32)
        values.extend([value[i] for i in range(4)])
    state = fx.Vector.from_elements(values, dtype=fx.Float32)
    requested = fx.Int32(-1)
    if const_expr(HAS_TRACK):
        requested = _g_ld(track_idx, seq, 1, fx.Int32)[0]
    for token, carry in range(bos, eos, init=[state]):
        token = fx.Int32(token)
        state = fx.Vector(carry[0])
        if const_expr(HAS_TRACK):  # noqa: SIM102
            if ((token - bos) % 64 == 0) & ((token - bos) // 64 == requested):
                for j in range_constexpr(BV // NUM_WAVES):
                    row = vs + wave * (BV // NUM_WAVES) + j
                    value = fx.Vector.from_elements(
                        [state[j * 4 + i] for i in range(4)], dtype=fx.Float32
                    )
                    _g_st(
                        track,
                        (seq * H + head) * VD * KD + row * KD + lane * 4,
                        value,
                        4,
                        fx.Float32,
                    )
        qf = _g_ld(q, token * sq + kh * KD + lane * 4, 4, fx.BFloat16).to(fx.Float32)
        kf = _g_ld(k, token * sk + kh * KD + lane * 4, 4, fx.BFloat16).to(fx.Float32)
        qs = (qf * qf).reduce(fx.ReductionOp.ADD)
        ks = (kf * kf).reduce(fx.ReductionOp.ADD)
        for shift in (1, 2, 4, 8, 16):
            qs = qs + qs.shuffle_xor(shift, 32)
            ks = ks + ks.shuffle_xor(shift, 32)
        qf = qf * fx.math.rsqrt(qs + 1.0e-6)
        kf = kf * fx.math.rsqrt(ks + 1.0e-6)
        qk = (qf * kf).reduce(fx.ReductionOp.ADD)
        for shift in (1, 2, 4, 8, 16):
            qk = qk + qk.shuffle_xor(shift, 32)
        decay = _exp(_g_ld(g, token * sg + head, 1, fx.Float32)[0])
        bet = _g_ld(beta, token * sb + head, 1, fx.Float32)[0]
        updated = []
        for j in range_constexpr(BV // NUM_WAVES):
            row = vs + wave * (BV // NUM_WAVES) + j
            value = (
                fx.Vector.from_elements(
                    [state[j * 4 + i] for i in range(4)], dtype=fx.Float32
                )
                * decay
            )
            kstate = (value * kf).reduce(fx.ReductionOp.ADD)
            qstate = (value * qf).reduce(fx.ReductionOp.ADD)
            for shift in (1, 2, 4, 8, 16):
                kstate = kstate + kstate.shuffle_xor(shift, 32)
                qstate = qstate + qstate.shuffle_xor(shift, 32)
            vv = _g_ld(v, token * sv + head * VD + row, 1, fx.BFloat16)[0].to(
                fx.Float32
            )
            delta = (vv - kstate) * bet
            output = fx.fma(delta, qk, qstate) * scale
            if lane == 0:
                _g_st(
                    out,
                    (token * H + head) * VD + row,
                    fx.Vector.from_elements(
                        [output.to(fx.BFloat16)], dtype=fx.BFloat16
                    ),
                    1,
                    fx.BFloat16,
                )
            value = fx.fma(kf, fx.Vector.filled(4, delta, fx.Float32), value)
            updated.extend([value[i] for i in range(4)])
        state = fx.Vector.from_elements(updated, dtype=fx.Float32)
        result = yield [state]
    state = fx.Vector(result)
    for j in range_constexpr(BV // NUM_WAVES):
        row = vs + wave * (BV // NUM_WAVES) + j
        value = fx.Vector.from_elements(
            [state[j * 4 + i] for i in range(4)], dtype=fx.Float32
        )
        _g_st(
            states,
            (slot * H + head) * VD * KD + row * KD + lane * 4,
            value.to(fx.BFloat16),
            4,
            fx.BFloat16,
        )


@functools.cache
def compile_recurrent_gdn_prefill_rdna(
    *,
    device_index,
    H,
    Hg,
    SLOTS,
    BV=BV,
    HAS_TRACK=False,
    HAS_INIT_FLAG=False,
    scale=KD**-0.5,
):
    """Token recurrence; grid (N * H, VD // BV), eight wave32s per workgroup.

    Device, head dimensions, pool size and optional-state modes specialize
    the cached launcher; token counts and token strides remain runtime values.
    """
    assert BV == 64

    @flyc.kernel
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        cu: fx.Tensor,
        indices: fx.Tensor,
        states: fx.Tensor,
        out: fx.Tensor,
        track: fx.Tensor,
        track_idx: fx.Tensor,
        has_init: fx.Tensor,
        sq: fx.Int32,
        sk: fx.Int32,
        sv: fx.Int32,
        sg: fx.Int32,
        sb: fx.Int32,
        total: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane, wave = tid % 32, tid // 32
        seq, head = fx.block_idx.x // H, fx.block_idx.x % H
        vs = fx.block_idx.y * BV
        kh = head // (H // Hg)
        gb = [
            fx.get_iter(fx.rocdl.make_buffer_tensor(t))
            for t in (
                q,
                k,
                v,
                g,
                beta,
                cu,
                indices,
                states,
                out,
                track,
                track_idx,
                has_init,
            )
        ]
        slot = _g_ld(gb[6], seq, 1, fx.Int32)[0]
        bos = _g_ld(gb[5], seq, 1, fx.Int32)[0]
        eos = _g_ld(gb[5], seq + 1, 1, fx.Int32)[0]
        if (slot >= 0) & (slot < SLOTS) & (bos >= 0) & (eos <= total) & (eos >= bos):
            _recurrent_prefill(
                gb[0],
                gb[1],
                gb[2],
                gb[3],
                gb[4],
                gb[7],
                gb[8],
                gb[9],
                gb[10],
                gb[11],
                seq,
                head,
                kh,
                vs,
                slot,
                bos,
                eos,
                sq,
                sk,
                sv,
                sg,
                sb,
                lane,
                wave,
                H,
                scale,
                HAS_TRACK,
                HAS_INIT_FLAG,
            )

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        cu: fx.Tensor,
        indices: fx.Tensor,
        states: fx.Tensor,
        out: fx.Tensor,
        track: fx.Tensor,
        track_idx: fx.Tensor,
        has_init: fx.Tensor,
        sq: fx.Int32,
        sk: fx.Int32,
        sv: fx.Int32,
        sg: fx.Int32,
        sb: fx.Int32,
        total: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            q,
            k,
            v,
            g,
            beta,
            cu,
            indices,
            states,
            out,
            track,
            track_idx,
            has_init,
            sq,
            sk,
            sv,
            sg,
            sb,
            total,
        ).launch(grid=(grid, VD // BV), block=(NUM_WAVES * 32,), stream=stream)

    return launch


@functools.cache
def compile_gdn_fused_prefill_rdna(
    *,
    device_index,
    H,
    Hg,
    SLOTS,
    BV=BV,
    HAS_TRACK=False,
    HAS_INIT_FLAG=False,
    OVERLAP_OUTPUT=False,
    scale=KD**-0.5,
):
    """Persistent chunk solve; grid (N * H, VD // BV), eight wave32s.

    Raw BF16 q/k stay in LDS; reciprocal norms fold into FP32 factors.
    OVERLAP_OUTPUT keeps weighted-key factors in registers while emitting o.
    """
    # Device is part of the handle cache key. All Tensor arguments are flattened
    # unit-stride views with fixed BF16/FP32/int32 dtypes; lengths are runtime.
    assert BV == 64

    @fx.struct
    class SharedStorage:
        q: fx.Array[fx.BFloat16, BT * ROW]
        k: fx.Array[fx.BFloat16, BT * ROW]
        state: fx.Array[fx.BFloat16, BV * SROW]
        kk: fx.Array[fx.Float32, BT * BT]
        qk: fx.Array[fx.Float32, BT * PROW]
        vn: fx.Array[fx.Float32, BT * BV]
        # Four rows: cumulative log-decay, q norm reciprocal, k norm
        # reciprocal and beta. The k row later holds weighted-key factors.
        fac: fx.Array[fx.Float32, 4 * BT]

    @flyc.kernel
    def kernel(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        cu: fx.Tensor,
        indices: fx.Tensor,
        states: fx.Tensor,
        out: fx.Tensor,
        track: fx.Tensor,
        track_idx: fx.Tensor,
        has_init: fx.Tensor,
        sq: fx.Int32,
        sk: fx.Int32,
        sv: fx.Int32,
        sg: fx.Int32,
        sb: fx.Int32,
        total: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane, wave = tid % 32, tid // 32
        seq, head = fx.block_idx.x // H, fx.block_idx.x % H
        vs = fx.block_idx.y * BV
        kh = head // (H // Hg)
        gb = [
            fx.get_iter(fx.rocdl.make_buffer_tensor(t))
            for t in (
                q,
                k,
                v,
                g,
                beta,
                cu,
                indices,
                states,
                out,
                track,
                track_idx,
                has_init,
            )
        ]
        slot = _g_ld(gb[6], seq, 1, fx.Int32)[0]
        bos = _g_ld(gb[5], seq, 1, fx.Int32)[0]
        eos = _g_ld(gb[5], seq + 1, 1, fx.Int32)[0]
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        pq, pk, ps, pkk, pqk, pvn, pf = [
            member.ptr
            for member in (
                lds.q,
                lds.k,
                lds.state,
                lds.kk,
                lds.qk,
                lds.vn,
                lds.fac,
            )
        ]
        # The solve replaces each KS row with vnew after consuming it.
        pks = pvn
        if (slot >= 0) & (slot < SLOTS) & (bos >= 0) & (eos <= total) & (eos >= bos):
            weak = _has_weak_decay(gb[3], bos, eos, sg, head, lane)
            if (eos - bos == 1) | weak:
                _recurrent_prefill(
                    gb[0],
                    gb[1],
                    gb[2],
                    gb[3],
                    gb[4],
                    gb[7],
                    gb[8],
                    gb[9],
                    gb[10],
                    gb[11],
                    seq,
                    head,
                    kh,
                    vs,
                    slot,
                    bos,
                    eos,
                    sq,
                    sk,
                    sv,
                    sg,
                    sb,
                    lane,
                    wave,
                    H,
                    scale,
                    HAS_TRACK,
                    HAS_INIT_FLAG,
                )
            else:
                load_init = fx.Int32(1)
                if const_expr(HAS_INIT_FLAG):
                    load_init = _g_ld(gb[11], seq, 1, fx.Int32)[0]
                initial_values = []
                # Wave w owns four 16x16 state tiles: rows 16*j..16*j+15,
                # columns 16*w..16*w+15, in the WMMA accumulator layout.
                for j in range_constexpr(BV * KD // (NUM_WAVES * 256)):
                    tile = wave + j * NUM_WAVES
                    r, c = tile // 8 * 16, tile % 8 * 16
                    for i in range_constexpr(8):
                        row, col = r + 2 * i + lane // 16, c + lane % 16
                        value = fx.Float32(0.0)
                        if load_init != 0:
                            value = _g_ld(
                                gb[7],
                                (slot * H + head) * VD * KD + (vs + row) * KD + col,
                                1,
                                fx.BFloat16,
                            )[0].to(fx.Float32)
                        initial_values.append(value)
                        _write(ps, row * SROW + col, value.to(fx.BFloat16), fx.BFloat16)
                state = fx.Vector.from_elements(initial_values, dtype=fx.Float32)
                gpu.barrier()
                requested = fx.Int32(-1)
                if const_expr(HAS_TRACK):
                    requested = _g_ld(gb[10], seq, 1, fx.Int32)[0]
                for start, carry in range(bos, eos, BT, init=[state]):
                    start = fx.Int32(start)
                    state = fx.Vector(carry[0])
                    if const_expr(HAS_TRACK):  # noqa: SIM102
                        if ((start - bos) % 64 == 0) & (
                            (start - bos) // 64 == requested
                        ):
                            for j in range_constexpr(BV * KD // (NUM_WAVES * 256)):
                                tile = wave + j * NUM_WAVES
                                r, c = tile // 8 * 16, tile % 8 * 16
                                for i in range_constexpr(8):
                                    _g_st(
                                        gb[9],
                                        (seq * H + head) * VD * KD
                                        + (vs + r + 2 * i + lane // 16) * KD
                                        + c
                                        + lane % 16,
                                        fx.Vector.from_elements(
                                            [state[j * 8 + i]], dtype=fx.Float32
                                        ),
                                        1,
                                        fx.Float32,
                                    )
                    # One wave normalizes a token at a time; each lane owns four K.
                    for j in range_constexpr(BT // NUM_WAVES):
                        row = wave + NUM_WAVES * j
                        token = start + row
                        qv = fx.Vector.filled(4, 0.0, fx.BFloat16)
                        kv = fx.Vector.filled(4, 0.0, fx.BFloat16)
                        if token < eos:
                            qv = _g_ld(
                                gb[0], token * sq + kh * KD + lane * 4, 4, fx.BFloat16
                            )
                            kv = _g_ld(
                                gb[1], token * sk + kh * KD + lane * 4, 4, fx.BFloat16
                            )
                        _lds_st(pq, row * ROW + lane * 4, qv, 4, fx.BFloat16)
                        _lds_st(pk, row * ROW + lane * 4, kv, 4, fx.BFloat16)
                        qf, kf = qv.to(fx.Float32), kv.to(fx.Float32)
                        qsum = (qf * qf).reduce(fx.ReductionOp.ADD)
                        ksum = (kf * kf).reduce(fx.ReductionOp.ADD)
                        for shift in (1, 2, 4, 8, 16):
                            qsum = qsum + qsum.shuffle_xor(shift, 32)
                            ksum = ksum + ksum.shuffle_xor(shift, 32)
                        if lane == 0:
                            _write(pf, BT + row, fx.math.rsqrt(qsum + 1.0e-6))
                            _write(pf, 2 * BT + row, fx.math.rsqrt(ksum + 1.0e-6))
                    if wave == 0:
                        logg = fx.Float32(0.0)
                        bet = fx.Float32(0.0)
                        if (lane < BT) & (start + lane < eos):
                            logg = _g_ld(
                                gb[3], (start + lane) * sg + head, 1, fx.Float32
                            )[0]
                            bet = _g_ld(
                                gb[4], (start + lane) * sb + head, 1, fx.Float32
                            )[0]
                        # All lanes participate in the inclusive scan. The first
                        # sixteen own this chunk's gates; the others carry zero.
                        for shift in (1, 2, 4, 8):
                            previous = gpu.shuffle_up(logg, shift, 32)
                            logg = logg + (lane >= shift).select(
                                previous, fx.Float32(0.0)
                            )
                        if lane < BT:
                            _write(pf, lane, logg)
                            _write(pf, 3 * BT + lane, bet)
                    gpu.barrier()
                    last = fx.min(eos - start, BT) - 1
                    glast = _read(pf, last)
                    # K products precede the solve. Q waves compute QK and QS
                    # while the two solve waves perform forward substitution.
                    mr, nc = fx.Int32(0), (wave % 4) * 16
                    if wave == 0:
                        token_product = _mm(
                            pk,
                            pk,
                            mr,
                            nc,
                            (ROW, 1),
                            (ROW, 1),
                            KD,
                            lane,
                            fx.BFloat16,
                            fx.BFloat16,
                        )
                        for i in range_constexpr(8):
                            r, c = 2 * i + lane // 16, lane % 16
                            kval = (
                                token_product[i]
                                * _exp(_read(pf, r) - _read(pf, c))
                                * _read(pf, 2 * BT + c)
                                * _read(pf, 2 * BT + r)
                                * _read(pf, 3 * BT + r)
                            )
                            _write(
                                pkk,
                                r * BT + c,
                                (r > c).select(kval, fx.Float32(0.0)),
                            )
                    q_state = fx.Vector.filled(8, 0.0, fx.Float32)
                    if wave < 4:
                        ks = _mm(
                            pk,
                            ps,
                            mr,
                            nc,
                            (ROW, 1),
                            (SROW, 1),
                            KD,
                            lane,
                            fx.BFloat16,
                            fx.BFloat16,
                        )
                        _store_tile(pks, mr, nc, BV, lane, ks)
                    gpu.barrier()
                    # Forward substitution, independent for each V channel. Two
                    # waves own the 64 columns until the following barrier.
                    if wave < BV // 32:
                        column = wave * 32 + lane
                        # Issue V loads together before the dependent solve.
                        prefetched = []
                        for rr in range_constexpr(BT):
                            value = fx.BFloat16(0.0)
                            if start + rr < eos:
                                value = _g_ld(
                                    gb[2],
                                    (start + rr) * sv + head * VD + vs + column,
                                    1,
                                    fx.BFloat16,
                                )[0]
                            prefetched.append(value)
                        vvalues = fx.Vector.from_elements(prefetched, dtype=fx.BFloat16)
                        # Previous rows belong to this lane, avoiding LDS rereads.
                        solved = []
                        for rr in range_constexpr(BT):
                            rhs = vvalues[rr].to(fx.Float32)
                            rhs = rhs - _read(pks, rr * BV + column) * _read(
                                pf, 2 * BT + rr
                            ) * _exp(_read(pf, rr))
                            rhs = rhs * _read(pf, 3 * BT + rr)
                            for j in range_constexpr(rr):
                                rhs = fx.fma(-_read(pkk, rr * BT + j), solved[j], rhs)
                            solved.append(rhs)
                            _write(pvn, rr * BV + column, rhs)
                    if wave == 4:
                        token_product = _mm(
                            pq,
                            pk,
                            mr,
                            nc,
                            (ROW, 1),
                            (ROW, 1),
                            KD,
                            lane,
                            fx.BFloat16,
                            fx.BFloat16,
                        )
                        for i in range_constexpr(8):
                            r, c = 2 * i + lane // 16, lane % 16
                            qval = (
                                token_product[i]
                                * _exp(_read(pf, r) - _read(pf, c))
                                * _read(pf, 2 * BT + c)
                                * _read(pf, BT + r)
                                * scale
                            )
                            _write(
                                pqk,
                                r * PROW + c,
                                (r >= c).select(qval, fx.Float32(0.0)),
                            )
                    if wave >= 4:
                        q_state = _mm(
                            pq,
                            ps,
                            mr,
                            nc,
                            (ROW, 1),
                            (SROW, 1),
                            KD,
                            lane,
                            fx.BFloat16,
                            fx.BFloat16,
                        )
                    gpu.barrier()
                    key_factors = None
                    if const_expr(OVERLAP_OUTPUT):
                        # Batched prompts overlap the Q output with state updates.
                        # Register factors preserve the read-only shared operands.
                        key_factors = fx.Vector.from_elements(
                            [
                                _read(pf, 2 * BT + j) * _exp(glast - _read(pf, j))
                                for j in range(BT)
                            ],
                            dtype=fx.Float32,
                        )
                    else:
                        # K normalization is dead after the solve. Reuse its row for
                        # the weighted-key factors while the Q waves emit output.
                        if (wave == 0) & (lane < BT):
                            _write(
                                pf,
                                2 * BT + lane,
                                _read(pf, 2 * BT + lane)
                                * _exp(glast - _read(pf, lane)),
                            )
                    if wave >= 4:
                        out_acc = _mm(
                            pqk,
                            pvn,
                            mr,
                            nc,
                            (PROW, 1),
                            (1, BV),
                            BT,
                            lane,
                            fx.Float32,
                            fx.Float32,
                            split_a=True,
                        )
                        for i in range_constexpr(8):
                            r, c = mr + 2 * i + lane // 16, nc + lane % 16
                            value = (
                                out_acc[i]
                                + q_state[i]
                                * _read(pf, BT + r)
                                * _exp(_read(pf, r))
                                * scale
                            )
                            if start + r < eos:
                                _g_st(
                                    gb[8],
                                    ((start + r) * H + head) * VD + vs + c,
                                    fx.Vector.from_elements(
                                        [value.to(fx.BFloat16)], dtype=fx.BFloat16
                                    ),
                                    1,
                                    fx.BFloat16,
                                )
                    # Shared weighted-key factors must be visible before updates.
                    # The overlap variant uses registers and read-only operands.
                    if const_expr(not OVERLAP_OUTPUT):
                        gpu.barrier()
                    updated_values = []
                    for j in range_constexpr(BV * KD // (NUM_WAVES * 256)):
                        tile = wave + j * NUM_WAVES
                        r, c = tile // 8 * 16, tile % 8 * 16
                        update = _mm(
                            pvn,
                            pk,
                            r,
                            c,
                            (1, BV),
                            (1, ROW),
                            BT,
                            lane,
                            fx.Float32,
                            fx.BFloat16,
                            key_factors if OVERLAP_OUTPUT else pf + 2 * BT,
                            OVERLAP_OUTPUT,
                        )
                        for i in range_constexpr(8):
                            pos = (r + 2 * i + lane // 16) * SROW + c + lane % 16
                            value = state[j * 8 + i] * _exp(glast) + update[i]
                            updated_values.append(value)
                            _write(ps, pos, value.to(fx.BFloat16), fx.BFloat16)
                    gpu.barrier()
                    state = fx.Vector.from_elements(updated_values, dtype=fx.Float32)
                    result = yield [state]
                state = fx.Vector(result)
                for j in range_constexpr(BV * KD // (NUM_WAVES * 256)):
                    tile = wave + j * NUM_WAVES
                    r, c = tile // 8 * 16, tile % 8 * 16
                    for i in range_constexpr(8):
                        _g_st(
                            gb[7],
                            (slot * H + head) * VD * KD
                            + (vs + r + 2 * i + lane // 16) * KD
                            + c
                            + lane % 16,
                            fx.Vector.from_elements(
                                [state[j * 8 + i].to(fx.BFloat16)], dtype=fx.BFloat16
                            ),
                            1,
                            fx.BFloat16,
                        )

    @flyc.jit
    def launch(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        cu: fx.Tensor,
        indices: fx.Tensor,
        states: fx.Tensor,
        out: fx.Tensor,
        track: fx.Tensor,
        track_idx: fx.Tensor,
        has_init: fx.Tensor,
        sq: fx.Int32,
        sk: fx.Int32,
        sv: fx.Int32,
        sg: fx.Int32,
        sb: fx.Int32,
        total: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            q,
            k,
            v,
            g,
            beta,
            cu,
            indices,
            states,
            out,
            track,
            track_idx,
            has_init,
            sq,
            sk,
            sv,
            sg,
            sb,
            total,
        ).launch(grid=(grid, VD // BV), block=(BLOCK,), stream=stream)

    return launch

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Parallel GDN chunk preparation and persistent state scan (gfx1151).

Preparation solves state-independent U/W across all 16-token chunks; the scan
walks each sequence in order. With normalized q/k and cumulative log-decay gc:

    L    = I + strict_tril(beta_t * (k_t . k_j) * exp(gc_t - gc_j))
    U    = solve(L, beta * v)                            [16, V]
    W    = solve(L, beta * exp(gc) * k)                   [16, K]
    P    = scale * tril((q @ k.T) * exp(gc_t - gc_j))      [16, 16]
    vnew = U - W @ S.T

The scan uses the single-kernel path's output and state updates. The solve
and register state are fp32; U/W, vnew and the LDS state copy are bf16 WMMA
operands. P retains its fp32 residual through two bf16 output products.
Per-chunk states never reach HBM; snapshots retain 64-token boundaries.

Preparation uses four wave32s, each lane solving two RHS columns. The scan
uses eight wave32s per (sequence, value head, 64-channel V slice).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from .gdn_fused_prefill import _g_ld, _g_st, _lds_ld, _lds_st
from .gdn_fused_prefill_rdna import (
    BT,
    BV,
    KD,
    MIN_MATRIX_DECAY,
    PROW,
    ROW,
    SROW,
    VD,
    _exp,
    _mm,
    _read,
    _recurrent_prefill,
    _write,
)


@functools.cache
def compile_prepare_gdn_prefill_rdna(*, device_index, H, Hg, N, SLOTS, scale):
    """Parallel chunk solves; grid (chunk_slots * H,), four wave32s.

    The grid includes spare workspace slots. Each block locates its sequence
    from device cu_seqlens and skips slots beyond the live chunk count.
    """

    # Tensor arguments are flattened unit-stride BF16/FP32/int32 views.
    # Device and compile dimensions key compiled handles.
    @fx.struct
    class Storage:
        q: fx.Array[fx.BFloat16, BT * ROW]
        k: fx.Array[fx.BFloat16, BT * ROW]
        v: fx.Array[fx.BFloat16, BT * ROW]
        kk: fx.Array[fx.Float32, BT * BT]
        factors: fx.Array[fx.Float32, 5 * BT]

    @flyc.kernel
    def prepare(
        q: fx.Tensor,
        k: fx.Tensor,
        v: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        cu: fx.Tensor,
        indices: fx.Tensor,
        u: fx.Tensor,
        w: fx.Tensor,
        p: fx.Tensor,
        factors: fx.Tensor,
        sq: fx.Int32,
        sk: fx.Int32,
        sv: fx.Int32,
        sg: fx.Int32,
        sb: fx.Int32,
        total: fx.Int32,
    ):
        tid = fx.thread_idx.x
        lane, wave = tid % 32, tid // 32
        chunk, head = fx.block_idx.x // H, fx.block_idx.x % H
        kh = head // (H // Hg)
        gb = [
            fx.get_iter(fx.rocdl.make_buffer_tensor(t))
            for t in (q, k, v, g, beta, cu, indices, u, w, p, factors)
        ]
        seq, start, end, offset = fx.Int32(-1), fx.Int32(0), fx.Int32(0), fx.Int32(0)
        if const_expr(N <= 8):
            for n in range_constexpr(N):
                bos = _g_ld(gb[5], n, 1, fx.Int32)[0]
                eos = _g_ld(gb[5], n + 1, 1, fx.Int32)[0]
                count = (fx.max(eos - bos, 0) + BT - 1) // BT
                if (chunk >= offset) & (chunk < offset + count):
                    seq = fx.Int32(n)
                    start = bos + (chunk - offset) * BT
                    end = eos
                offset = offset + count
        else:
            # Each lane owns one sequence; the prefix sum locates this chunk.
            for group in range_constexpr((N + 31) // 32):
                n = group * 32 + lane
                bos, eos, count = fx.Int32(0), fx.Int32(0), fx.Int32(0)
                if n < N:
                    bos = _g_ld(gb[5], n, 1, fx.Int32)[0]
                    eos = _g_ld(gb[5], n + 1, 1, fx.Int32)[0]
                    count = (fx.max(eos - bos, 0) + BT - 1) // BT
                prefix = count
                for shift in (1, 2, 4, 8, 16):
                    prior = fx.Int32(gpu.shuffle_up(prefix, shift, 32))
                    prefix = prefix + (lane >= shift).select(prior, fx.Int32(0))
                begin = offset + prefix - count
                matches = (chunk >= begin) & (chunk < offset + prefix) & (n < N)
                found = matches.select(n, fx.Int32(-1))
                for shift in (1, 2, 4, 8, 16):
                    found = fx.max(found, found.shuffle_xor(shift, 32))
                candidate_start = bos + (chunk - begin) * BT
                selected_lane = fx.max(found, 0) % 32
                selected_start = fx.Int32(
                    gpu.shuffle_idx(candidate_start, selected_lane, 32)
                )
                selected_end = fx.Int32(gpu.shuffle_idx(eos, selected_lane, 32))
                if found >= 0:
                    seq, start, end = found, selected_start, selected_end
                offset = offset + fx.Int32(gpu.shuffle_idx(prefix, 31, 32))
        if (seq >= 0) & (start >= 0) & (end <= total):
            slot = _g_ld(gb[6], seq, 1, fx.Int32)[0]
            if (slot >= 0) & (slot < SLOTS):
                seq_bos = _g_ld(gb[5], seq, 1, fx.Int32)[0]
                if (end - seq_bos > 1) & (seq_bos >= 0):
                    storage = fx.SharedAllocator().allocate(Storage).peek()
                    pq, pk, pv = storage.q.ptr, storage.k.ptr, storage.v.ptr
                    pkk, pf = storage.kk.ptr, storage.factors.ptr
                    base = (chunk * H + head) * BT
                    for j in range_constexpr(BT // 4):
                        row, col = wave + 4 * j, lane * 4
                        token = start + row
                        qv, kv, vv = [
                            fx.Vector.filled(4, 0.0, fx.BFloat16) for _ in range(3)
                        ]
                        if token < end:
                            qv = _g_ld(
                                gb[0], token * sq + kh * KD + col, 4, fx.BFloat16
                            )
                            kv = _g_ld(
                                gb[1], token * sk + kh * KD + col, 4, fx.BFloat16
                            )
                            vv = _g_ld(
                                gb[2], token * sv + head * VD + col, 4, fx.BFloat16
                            )
                        _lds_st(pq, row * ROW + col, qv, 4, fx.BFloat16)
                        _lds_st(pk, row * ROW + col, kv, 4, fx.BFloat16)
                        _lds_st(pv, row * ROW + col, vv, 4, fx.BFloat16)
                        qf, kf = qv.to(fx.Float32), kv.to(fx.Float32)
                        qs = (qf * qf).reduce(fx.ReductionOp.ADD)
                        ks = (kf * kf).reduce(fx.ReductionOp.ADD)
                        for shift in (1, 2, 4, 8, 16):
                            qs, ks = qs + qs.shuffle_xor(
                                shift, 32
                            ), ks + ks.shuffle_xor(shift, 32)
                        if lane == 0:
                            _write(pf, BT + row, fx.math.rsqrt(qs + 1.0e-6))
                            _write(pf, 2 * BT + row, fx.math.rsqrt(ks + 1.0e-6))
                    if wave == 0:
                        gv, bv = fx.Float32(0.0), fx.Float32(0.0)
                        if (lane < BT) & (start + lane < end):
                            gv = _g_ld(
                                gb[3], (start + lane) * sg + head, 1, fx.Float32
                            )[0]
                            bv = _g_ld(
                                gb[4], (start + lane) * sb + head, 1, fx.Float32
                            )[0]
                        for shift in (1, 2, 4, 8):
                            prior = gpu.shuffle_up(gv, shift, 32)
                            gv = gv + (lane >= shift).select(prior, fx.Float32(0.0))
                        if lane < BT:
                            _write(pf, lane, gv)
                            _write(pf, 3 * BT + lane, bv)
                    gpu.barrier()
                    glast = _read(pf, fx.min(end - start, BT) - 1)
                    matrix_decay = glast < -MIN_MATRIX_DECAY * fx.min(
                        end - start, BT
                    ).to(fx.Float32)
                    if (wave == 0) & (lane == 0):
                        decay = matrix_decay.select(_exp(glast), fx.Float32(-1.0))
                        _g_st(
                            gb[10],
                            (chunk * H + head) * (2 * BT + 1) + BT,
                            fx.Vector.from_elements([decay], dtype=fx.Float32),
                            1,
                            fx.Float32,
                        )
                    if matrix_decay:
                        if wave == 0:
                            if lane < BT:
                                _write(
                                    pf,
                                    4 * BT + lane,
                                    _read(pf, 3 * BT + lane)
                                    * _exp(_read(pf, lane))
                                    * _read(pf, 2 * BT + lane),
                                )
                            kk = _mm(
                                pk,
                                pk,
                                0,
                                0,
                                (ROW, 1),
                                (ROW, 1),
                                KD,
                                lane,
                                fx.BFloat16,
                                fx.BFloat16,
                            )
                            for i in range_constexpr(8):
                                row, col = 2 * i + lane // 16, lane % 16
                                value = (
                                    kk[i]
                                    * _read(pf, 2 * BT + row)
                                    * _read(pf, 2 * BT + col)
                                    * _exp(_read(pf, row) - _read(pf, col))
                                    * _read(pf, 3 * BT + row)
                                )
                                _write(
                                    pkk,
                                    row * BT + col,
                                    (row > col).select(value, fx.Float32(0.0)),
                                )
                        if wave == 1:
                            qk = _mm(
                                pq,
                                pk,
                                0,
                                0,
                                (ROW, 1),
                                (ROW, 1),
                                KD,
                                lane,
                                fx.BFloat16,
                                fx.BFloat16,
                            )
                            for i in range_constexpr(8):
                                row, col = 2 * i + lane // 16, lane % 16
                                value = (
                                    qk[i]
                                    * _exp(_read(pf, row) - _read(pf, col))
                                    * _read(pf, BT + row)
                                    * _read(pf, 2 * BT + col)
                                    * scale
                                )
                                value = (row >= col).select(value, fx.Float32(0.0))
                                _g_st(
                                    gb[9],
                                    (base + row) * BT + col,
                                    fx.Vector.from_elements([value], dtype=fx.Float32),
                                    1,
                                    fx.Float32,
                                )
                            if lane < BT:
                                value = (
                                    _read(pf, BT + lane) * _exp(_read(pf, lane)) * scale
                                )
                                _g_st(
                                    gb[10],
                                    (chunk * H + head) * (2 * BT + 1) + lane,
                                    fx.Vector.from_elements([value], dtype=fx.Float32),
                                    1,
                                    fx.Float32,
                                )
                        gpu.barrier()
                        # Each lane solves two RHS columns: waves 0..1 own U,
                        # waves 2..3 own W. Solved rows remain in registers.
                        channel = (wave % 2) * 64 + lane * 2
                        values = []
                        for row in range_constexpr(BT):
                            value = _lds_ld(pv, row * ROW + channel, 2, fx.BFloat16)
                            if wave >= 2:
                                value = _lds_ld(pk, row * ROW + channel, 2, fx.BFloat16)
                            values.append(value)
                        solved = []
                        for row in range_constexpr(BT):
                            factor = _read(pf, 3 * BT + row)
                            if wave >= 2:
                                factor = _read(pf, 4 * BT + row)
                            value = values[row].to(fx.Float32) * factor
                            for col in range_constexpr(row):
                                value = fx.fma(
                                    fx.Vector.filled(
                                        2, -_read(pkk, row * BT + col), fx.Float32
                                    ),
                                    solved[col],
                                    value,
                                )
                            solved.append(value)
                            if wave < 2:
                                _g_st(
                                    gb[7],
                                    (base + row) * KD + channel,
                                    value.to(fx.BFloat16),
                                    2,
                                    fx.BFloat16,
                                )
                            else:
                                _g_st(
                                    gb[8],
                                    (base + row) * KD + channel,
                                    value.to(fx.BFloat16),
                                    2,
                                    fx.BFloat16,
                                )
                        if (wave == 0) & (lane < BT):
                            value = _read(pf, 2 * BT + lane) * _exp(
                                glast - _read(pf, lane)
                            )
                            _g_st(
                                gb[10],
                                (chunk * H + head) * (2 * BT + 1) + BT + 1 + lane,
                                fx.Vector.from_elements([value], dtype=fx.Float32),
                                1,
                                fx.Float32,
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
        u: fx.Tensor,
        w: fx.Tensor,
        p: fx.Tensor,
        factors: fx.Tensor,
        sq: fx.Int32,
        sk: fx.Int32,
        sv: fx.Int32,
        sg: fx.Int32,
        sb: fx.Int32,
        total: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        prepare(
            q, k, v, g, beta, cu, indices, u, w, p, factors, sq, sk, sv, sg, sb, total
        ).launch(grid=(grid,), block=(128,), stream=stream)

    return launch


@functools.cache
def compile_scan_gdn_prefill_rdna(
    *, device_index, H, Hg, N, SLOTS, HAS_TRACK, HAS_INIT_FLAG, scale=KD**-0.5
):
    """Persistent state/output scan; grid (N * H, VD // BV), eight wave32s.

    Workspace chunks are packed by sequence, then chunk, with heads contiguous:
    U/W [chunk, H, 16, 128] BF16 and P [chunk, H, 16, 16] FP32. Each FP32
    factor row [chunk, H, 33] holds scale * rq * exp(gc) in 0..15,
    exp(gc_last) at 16, and rk * exp(gc_last - gc) in 17..32; rq/rk are
    reciprocal norms of raw q/k.

    A -1 decay sentinel selects FP32 recurrence for the entire sequence/head.
    All sentinels are checked before reading U/W: other workspace entries of
    a marked chunk are uninitialized. State tiles share the single-kernel
    wave/lane layout; Q staging becomes wave-private K after QS reads retire.
    """

    # Tensor arguments are flattened unit-stride BF16/FP32/int32 views.
    # Device, dimensions, metadata length and mode flags key compiled handles.
    @fx.struct
    class Storage:
        q_kw: fx.Array[fx.BFloat16, BT * ROW]
        w: fx.Array[fx.BFloat16, BT * ROW]
        state: fx.Array[fx.BFloat16, BV * SROW]
        vn: fx.Array[fx.BFloat16, BT * BV]
        p: fx.Array[fx.Float32, BT * PROW]
        factors: fx.Array[fx.Float32, 2 * BT]

    @flyc.kernel
    def scan(
        q: fx.Tensor,
        cu: fx.Tensor,
        indices: fx.Tensor,
        states: fx.Tensor,
        out: fx.Tensor,
        track: fx.Tensor,
        track_idx: fx.Tensor,
        has_init: fx.Tensor,
        u: fx.Tensor,
        w: fx.Tensor,
        k: fx.Tensor,
        p: fx.Tensor,
        factors: fx.Tensor,
        v: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
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
                cu,
                indices,
                states,
                out,
                track,
                track_idx,
                has_init,
                u,
                w,
                k,
                p,
                factors,
                v,
                g,
                beta,
            )
        ]
        slot = _g_ld(gb[2], seq, 1, fx.Int32)[0]
        bos = _g_ld(gb[1], seq, 1, fx.Int32)[0]
        eos = _g_ld(gb[1], seq + 1, 1, fx.Int32)[0]
        if (slot >= 0) & (slot < SLOTS) & (bos >= 0) & (eos <= total) & (eos >= bos):
            first = fx.Int32(0)
            if const_expr(N <= 8):
                for n in range_constexpr(N):
                    if seq > n:
                        nb = _g_ld(gb[1], n, 1, fx.Int32)[0]
                        ne = _g_ld(gb[1], n + 1, 1, fx.Int32)[0]
                        first = first + (fx.max(ne - nb, 0) + BT - 1) // BT
            else:
                for group in range_constexpr((N + 31) // 32):
                    n = group * 32 + lane
                    if n < seq:
                        nb = _g_ld(gb[1], n, 1, fx.Int32)[0]
                        ne = _g_ld(gb[1], n + 1, 1, fx.Int32)[0]
                        first = first + (fx.max(ne - nb, 0) + BT - 1) // BT
                for shift in (1, 2, 4, 8, 16):
                    first = first + first.shuffle_xor(shift, 32)
            weak = fx.Int32(0)
            if eos - bos > 1:
                chunks = (eos - bos + BT - 1) // BT
                for j, carry in range(lane, chunks, 32, init=[weak]):
                    j = fx.Int32(j)
                    weak = fx.Int32(carry[0])
                    flag = _g_ld(
                        gb[12],
                        ((first + j) * H + head) * (2 * BT + 1) + BT,
                        1,
                        fx.Float32,
                    )[0]
                    weak = fx.max(weak, (flag < 0).select(fx.Int32(1), fx.Int32(0)))
                    result = yield [weak]
                weak = fx.Int32(result)
            for shift in (1, 2, 4, 8, 16):
                weak = fx.max(weak, weak.shuffle_xor(shift, 32))
            if (eos - bos == 1) | (weak > 0):
                _recurrent_prefill(
                    gb[0],
                    gb[10],
                    gb[13],
                    gb[14],
                    gb[15],
                    gb[3],
                    gb[4],
                    gb[5],
                    gb[6],
                    gb[7],
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
                storage = fx.SharedAllocator().allocate(Storage).peek()
                pq, pw, ps = storage.q_kw.ptr, storage.w.ptr, storage.state.ptr
                pvn, pp, pf = storage.vn.ptr, storage.p.ptr, storage.factors.ptr
                load_init = fx.Int32(1)
                if const_expr(HAS_INIT_FLAG):
                    load_init = _g_ld(gb[7], seq, 1, fx.Int32)[0]
                initial = []
                for j in range_constexpr(4):
                    tile = wave + j * 8
                    row, col = tile // 8 * 16, tile % 8 * 16
                    for i in range_constexpr(8):
                        rr, cc = row + 2 * i + lane // 16, col + lane % 16
                        value = fx.Float32(0.0)
                        if load_init != 0:
                            value = _g_ld(
                                gb[3],
                                (slot * H + head) * VD * KD + (vs + rr) * KD + cc,
                                1,
                                fx.BFloat16,
                            )[0].to(fx.Float32)
                        initial.append(value)
                        _write(ps, rr * SROW + cc, value.to(fx.BFloat16), fx.BFloat16)
                state = fx.Vector.from_elements(initial, dtype=fx.Float32)
                requested = fx.Int32(-1)
                if const_expr(HAS_TRACK):
                    requested = _g_ld(gb[6], seq, 1, fx.Int32)[0]
                for start, carry in range(bos, eos, BT, init=[state]):
                    start = fx.Int32(start)
                    state = fx.Vector(carry[0])
                    if const_expr(HAS_TRACK):  # noqa: SIM102
                        if ((start - bos) % 64 == 0) & (
                            (start - bos) // 64 == requested
                        ):
                            for j in range_constexpr(4):
                                tile = wave + j * 8
                                row, col = tile // 8 * 16, tile % 8 * 16
                                for i in range_constexpr(8):
                                    _g_st(
                                        gb[5],
                                        (seq * H + head) * VD * KD
                                        + (vs + row + 2 * i + lane // 16) * KD
                                        + col
                                        + lane % 16,
                                        fx.Vector.from_elements(
                                            [state[j * 8 + i]], dtype=fx.Float32
                                        ),
                                        1,
                                        fx.Float32,
                                    )
                    chunk = first + (start - bos) // BT
                    base = (chunk * H + head) * BT
                    decay = _g_ld(
                        gb[12], (chunk * H + head) * (2 * BT + 1) + BT, 1, fx.Float32
                    )[0]
                    for j in range_constexpr(2):
                        pos = tid * 4 + j * 1024
                        row, col = pos // KD, pos % KD
                        qv = fx.Vector.filled(4, 0.0, fx.BFloat16)
                        if start + row < eos:
                            qv = _g_ld(
                                gb[0],
                                (start + row) * sq + kh * KD + col,
                                4,
                                fx.BFloat16,
                            )
                        wv = _g_ld(gb[9], (base + row) * KD + col, 4, fx.BFloat16)
                        _lds_st(pq, row * ROW + col, qv, 4, fx.BFloat16)
                        _lds_st(pw, row * ROW + col, wv, 4, fx.BFloat16)
                    row, col = tid // 16, tid % 16
                    pv = _g_ld(gb[11], (base + row) * BT + col, 1, fx.Float32)[0]
                    _write(pp, row * PROW + col, pv)
                    if tid < BT:
                        fq = _g_ld(
                            gb[12],
                            (chunk * H + head) * (2 * BT + 1) + tid,
                            1,
                            fx.Float32,
                        )[0]
                        _write(pf, tid, fq)
                        fk = _g_ld(
                            gb[12],
                            (chunk * H + head) * (2 * BT + 1) + BT + 1 + tid,
                            1,
                            fx.Float32,
                        )[0]
                        _write(pf, BT + tid, fk)
                    uv = _g_ld(gb[8], (base + row) * VD + vs + col * 4, 4, fx.BFloat16)
                    _lds_st(pvn, row * BV + col * 4, uv, 4, fx.BFloat16)
                    gpu.barrier()
                    nc = (wave % 4) * 16
                    q_state = fx.Vector.filled(8, 0.0, fx.Float32)
                    if wave < 4:
                        ws = _mm(
                            pw,
                            ps,
                            0,
                            nc,
                            (ROW, 1),
                            (SROW, 1),
                            KD,
                            lane,
                            fx.BFloat16,
                            fx.BFloat16,
                        )
                        for i in range_constexpr(8):
                            rr, cc = 2 * i + lane // 16, nc + lane % 16
                            _write(
                                pvn,
                                rr * BV + cc,
                                (
                                    _read(pvn, rr * BV + cc, fx.BFloat16).to(fx.Float32)
                                    - ws[i]
                                ).to(fx.BFloat16),
                                fx.BFloat16,
                            )
                    else:
                        q_state = _mm(
                            pq,
                            ps,
                            0,
                            nc,
                            (ROW, 1),
                            (SROW, 1),
                            KD,
                            lane,
                            fx.BFloat16,
                            fx.BFloat16,
                        )
                    gpu.barrier()
                    # Each wave stages only the K columns its four state
                    # tiles consume. The preceding barrier retires all Q reads.
                    for j in range_constexpr(2):
                        rr = lane // 4 + j * 8
                        cc = wave * 16 + (lane % 4) * 4
                        value = fx.Vector.filled(4, 0.0, fx.BFloat16)
                        if start + rr < eos:
                            value = _g_ld(
                                gb[10],
                                (start + rr) * sk + kh * KD + cc,
                                4,
                                fx.BFloat16,
                            )
                        _lds_st(pq, rr * ROW + cc, value, 4, fx.BFloat16)
                    if wave >= 4:
                        output = _mm(
                            pp,
                            pvn,
                            0,
                            nc,
                            (PROW, 1),
                            (1, BV),
                            BT,
                            lane,
                            fx.Float32,
                            fx.BFloat16,
                            split_a=True,
                        )
                        for i in range_constexpr(8):
                            rr, cc = 2 * i + lane // 16, nc + lane % 16
                            value = output[i] + q_state[i] * _read(pf, rr)
                            if start + rr < eos:
                                _g_st(
                                    gb[4],
                                    ((start + rr) * H + head) * VD + vs + cc,
                                    fx.Vector.from_elements(
                                        [value.to(fx.BFloat16)], dtype=fx.BFloat16
                                    ),
                                    1,
                                    fx.BFloat16,
                                )
                    updated = []
                    for j in range_constexpr(4):
                        tile = wave + j * 8
                        rr, cc = tile // 8 * 16, tile % 8 * 16
                        change = _mm(
                            pvn,
                            pq,
                            rr,
                            cc,
                            (1, BV),
                            (1, ROW),
                            BT,
                            lane,
                            fx.BFloat16,
                            fx.BFloat16,
                            pf + BT,
                        )
                        for i in range_constexpr(8):
                            value = state[j * 8 + i] * decay + change[i]
                            updated.append(value)
                            _write(
                                ps,
                                (rr + 2 * i + lane // 16) * SROW + cc + lane % 16,
                                value.to(fx.BFloat16),
                                fx.BFloat16,
                            )
                    gpu.barrier()
                    state = fx.Vector.from_elements(updated, dtype=fx.Float32)
                    result = yield [state]
                state = fx.Vector(result)
                for j in range_constexpr(4):
                    tile = wave + j * 8
                    row, col = tile // 8 * 16, tile % 8 * 16
                    for i in range_constexpr(8):
                        _g_st(
                            gb[3],
                            (slot * H + head) * VD * KD
                            + (vs + row + 2 * i + lane // 16) * KD
                            + col
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
        cu: fx.Tensor,
        indices: fx.Tensor,
        states: fx.Tensor,
        out: fx.Tensor,
        track: fx.Tensor,
        track_idx: fx.Tensor,
        has_init: fx.Tensor,
        u: fx.Tensor,
        w: fx.Tensor,
        k: fx.Tensor,
        p: fx.Tensor,
        factors: fx.Tensor,
        v: fx.Tensor,
        g: fx.Tensor,
        beta: fx.Tensor,
        sq: fx.Int32,
        sk: fx.Int32,
        sv: fx.Int32,
        sg: fx.Int32,
        sb: fx.Int32,
        total: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        scan(
            q,
            cu,
            indices,
            states,
            out,
            track,
            track_idx,
            has_init,
            u,
            w,
            k,
            p,
            factors,
            v,
            g,
            beta,
            sq,
            sk,
            sv,
            sg,
            sb,
            total,
        ).launch(grid=(grid, VD // BV), block=(256,), stream=stream)

    return launch

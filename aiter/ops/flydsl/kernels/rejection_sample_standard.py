# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Standard rejection sampling of 5 draft tokens per request over a 129280-token
vocabulary, in two launches.

Stats launch: one 256-thread workgroup per (row, 8192-column block) of the 5
verified rows of each request reads the target row (fp32, or bf16 divided by the
request's temperature) and the bf16 draft row once and writes the block's
(max, lowest argmax, sum exp) of both to a workspace; every reduction has one
fixed order.

Resample launch: one workgroup per (request, block). Every workgroup redoes the
request's small decision from the stats (lane j of each wave is step j:
log-sum-exps, the probability-ratio test, the Philox draw), takes the accepted
count from a ballot, and scans its block of the one row that gets a resampled
token (residual or target row plus Gumbel noise, or the plain argmax at
temperature 0). A request's blocks run on one XCD (workgroup w on XCD w % 8) and
meet in its L2: they store their winners, the last to take an atomic ticket (one
128-byte line per request, against false sharing) reduces the blocks and writes
the request's tokens, and the ticket is zero again at the end of every launch.
No other atomics; each output has one writer.

exp and log are the hardware v_exp_f32 / v_log_f32, so values differ from
Triton's in the last bits. A NaN never wins a scan.
"""

import struct

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from . import buffer_ops
from .wave64_xor_gfx950 import xor_lane, xor_lane_i32

V = 129280  # vocabulary
NSPEC = 5  # draft tokens per request
ROWS = NSPEC + 1  # logits rows per request (the drafts and the bonus row)
U_MIN = 4.6566127342e-10  # smallest positive value of Triton's fp32 tl.rand

F, I, I64 = fx.Float32, fx.Int32, fx.Int64
NT = 256
BLK = 8192  # columns of a block
NB = (V + BLK - 1) // BLK  # 16
NQ = BLK // (NT * 4)  # 8 four-column loads per thread
STATS_WORDS = 8  # per (row, block): t max, t sum, t argmax, d max, d sum
MAX_REQS = 1024
INF = float("inf")
NEG_INF_BITS = -8388608
BIG = 1 << 30
LARGE = 0x7FFFFFF0
STATS_BYTES = MAX_REQS * NSPEC * NB * STATS_WORDS * 4
PART_BYTES = MAX_REQS * NB * 2 * 4
CNT_BYTES = MAX_REQS * 128  # one line per request
WORKSPACE_BYTES = STATS_BYTES + PART_BYTES + CNT_BYTES


def _fbits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


LOG2E = _fbits(0x3FB8AA3B)
LN2 = _fbits(0x3F317218)
_C = [float(struct.unpack("<f", struct.pack("<f", 1.0 / k))[0]) for k in range(1, 9)]


def _i32(x):
    x &= 0xFFFFFFFF
    return x - (1 << 32) if x >= (1 << 31) else x


def _lshr(x, s):
    return (x >> s) & ((1 << (32 - s)) - 1)


def _rotl(x, s):
    return (x << s) | _lshr(x, 32 - s)


def _mix(h, key):
    key = key * _i32(0xCC9E2D51)
    key = _rotl(key, 15)
    key = key * _i32(0x1B873593)
    h = h ^ key
    h = _rotl(h, 13)
    return (h << 2) + h + _i32(0xE6546B64)


def _fmax(a, b):
    return (a > b).select(a, b)


def _exp2(y):
    """v_exp_f32: 2^y, 0 below the fp32 range."""
    return F(fx.rocdl.exp2(T.f32, fx.as_ir_value(y)))


def _exp(d):
    return _exp2(d * F(LOG2E))


def _log(x):
    """v_log_f32 scaled to the natural log, for x >= 2^-126."""
    return fx.math.log2(x, fastmath=fx.FastMathFlags.afn) * F(LN2)


def _log1p_neg(v):
    """vLLM's ``_log1p_neg_stable``: log(1 - v) for v in [0, 1]."""
    p = F(_C[7])
    for k in (7, 6, 5, 4, 3, 2, 1):
        p = fx.math.fma(v, p, F(_C[k - 1]))
    series = (v * F(-1.0)) * p
    one_m = F(1.0) - v
    direct = _log(
        (one_m > F(5.960464477539063e-08)).select(one_m, F(5.960464477539063e-08))
    )
    return (v < F(0.25)).select(series, direct)


def _noise(h0, col_c1):
    """fp32 Gumbel noise of the vocab id whose product with 0xCC9E2D51 is ``col_c1``."""
    k = _rotl(col_c1, 15) * _i32(0x1B873593)
    h = _rotl(h0 ^ k, 13)
    r = (h << 2) + h + _i32(0xE6546B64)
    r = r ^ 16
    r = r ^ _lshr(r, 16)
    r = r * _i32(0x85EBCA6B)
    r = r ^ _lshr(r, 13)
    r = r * _i32(0xC2B2AE35)
    r = r ^ _lshr(r, 16)
    u = _lshr(r, 16).to(F) * F(1.52587890625e-05) + ((r & 0xFFFF).to(F) + F(0.5)) * F(
        2.3283064365386963e-10
    )
    return _log(_log1p_neg(u) * F(-1.0)) * F(-1.0)


def _umulhi(a, c):
    """High 32 bits of the unsigned product of ``a`` (i32) and the constant ``c``."""
    return (((a.to(I64) & I64(0xFFFFFFFF)) * I64(c)) >> I64(32)).to(I)


def _philox_uniform(k0, k1, c0, c1):
    """Triton's ``tl.rand(seed, pos)`` clamped to its smallest positive value; seed (k0, k1), pos (c0, c1)."""
    c2 = c3 = I(0)
    for _ in range(10):
        a, b = c0, c2
        c0 = _umulhi(b, 0xCD9E8D57) ^ c1 ^ k0
        c2 = _umulhi(a, 0xD2511F53) ^ c3 ^ k1
        c1 = b * _i32(0xCD9E8D57)
        c3 = a * _i32(0xD2511F53)
        k0 = k0 + _i32(0x9E3779B9)
        k1 = k1 + _i32(0xBB67AE85)
    x = (c0 < 0).select(c0 * -1 - 1, c0)
    u = x.to(F) * F(U_MIN)
    return (u > F(U_MIN)).select(u, F(U_MIN))


def _comb(v1, i1, v2, i2):
    """Keep (v1, i1) iff v1 > v2 or (v1 == v2 and i1 < i2): the maximum, lowest id first."""
    keep = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return keep.select(v1, v2), keep.select(i1, i2)


def _unpack(w, k):
    """Element k (0..1) of a packed bf16 pair as fp32."""
    return ((w << 16) if k == 0 else (w & -65536)).bitcast(F)


def _readlane(v, k):
    if v.dtype == F:
        return I(
            fx.rocdl.readlane(T.i32, fx.as_ir_value(v.bitcast(I)), fx.as_ir_value(k))
        ).bitcast(F)
    return I(fx.rocdl.readlane(T.i32, fx.as_ir_value(v), fx.as_ir_value(k)))


def _uni(v):
    """Wave-uniform copy of an i32 (readfirstlane)."""
    return I(fx.rocdl.readfirstlane(T.i32, fx.as_ir_value(v)))


def _ticket_ptr(ptr_i64):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4),
        ptr_i64,
    )


def build_rejection_stats(target_bf16: bool):
    @fx.struct
    class Shared:
        red: fx.Array[fx.Int32, 20]

    @flyc.kernel(
        name="rejection_stats_v129280_s5" + ("_tbf16" if target_bf16 else ""),
        known_block_size=[NT, 1, 1],
    )
    def kernel(
        arg_t: fx.Int64,  # target logits [B * 6, V]: fp32 (processed) or bf16 (raw, divided by the temperature)
        arg_d: fx.Int64,  # bf16 draft logits [R, 5, V]
        arg_idx: fx.Int64,  # int32 request state of each request [B]
        arg_temp: fx.Int64,  # fp32 [R]
        arg_stats: fx.Int64,  # workspace stats [B * 5, NB, 8]
    ):
        tx, blk, row = fx.thread_idx.x, fx.block_idx.x, fx.block_idx.y
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = I(fx.rocdl.readfirstlane(T.i32, tx // 64))
        req = row // NSPEC
        step = row - req * NSPEC
        idr = buffer_ops.create_buffer_resource_from_addr(
            arg_idx, num_records_bytes=LARGE
        )
        tr = buffer_ops.create_buffer_resource_from_addr(
            arg_temp, num_records_bytes=LARGE
        )
        slot = _uni(I(buffer_ops.buffer_load(idr, req, vec_width=1, dtype=I)))
        temp = _uni(I(buffer_ops.buffer_load(tr, slot, vec_width=1, dtype=I))).bitcast(
            F
        )
        greedy = temp == F(0.0)
        tbytes = 2 if target_bf16 else 4
        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_t + I64(req * ROWS + step) * (V * tbytes), num_records_bytes=V * tbytes
        )
        dr = buffer_ops.create_buffer_resource_from_addr(
            arg_d + I64(slot * NSPEC + step) * (V * 2), num_records_bytes=V * 2
        )
        x, d = [], []
        if const_expr(target_bf16):
            rtemp = F(1.0) / greedy.select(
                F(1.0), temp
            )  # the sums tolerate the reciprocal's last bit
            for q in range_constexpr(NQ // 2):
                col = blk * BLK + q * (NT * 8) + tx * 8
                ok = col < V
                w = Vec(
                    buffer_ops.buffer_load(xr, col // 2, vec_width=4, dtype=I, mask=ok)
                )
                for k in range_constexpr(8):
                    x.append(ok.select(_unpack(I(w[k // 2]), k % 2), F(-INF)))
        else:
            for q in range_constexpr(NQ):
                col = blk * BLK + q * (NT * 4) + tx * 4
                ok = col < V
                v = Vec(buffer_ops.buffer_load(xr, col, vec_width=4, dtype=I, mask=ok))
                for k in range_constexpr(4):
                    x.append(ok.select(I(v[k]).bitcast(F), F(-INF)))
        rdraft = F(1.0) / greedy.select(F(1.0), temp)
        for q in range_constexpr(NQ // 2):
            col = blk * BLK + q * (NT * 8) + tx * 8
            ok = (col < V) & ~greedy
            w = Vec(buffer_ops.buffer_load(dr, col // 2, vec_width=4, dtype=I, mask=ok))
            for k in range_constexpr(8):
                d.append(ok.select(_unpack(I(w[k // 2]), k % 2), F(-INF)))

        per = 8 if target_bf16 else 4  # columns per thread and load

        def col_of(n):
            """Column of this thread's n-th target value."""
            return blk * BLK + (n // per) * (NT * per) + tx * per + n % per

        def local_max_sum(vals, indexed, scale):
            """(max, lowest index of the max, sum exp(x - max)) of this thread's raw values times ``scale``; the
            index only when ``indexed`` (the greedy argmax)."""
            m = vals[0]
            for e in vals[1:]:
                m = _fmax(m, e)
            i = I(BIG)
            if const_expr(indexed):  # noqa: SIM102 (trace-time vs runtime branch)
                if greedy:
                    for n in range_constexpr(len(vals) - 1, -1, -1):
                        i = (vals[n] == m).select(col_of(n), i)
            c = scale * F(LOG2E)
            ml = (m > F(-INF)).select(m, F(0.0)) * c
            s = F(0.0)
            for e in vals:
                s = s + _exp2(fx.math.fma(e, c, ml * F(-1.0)))
            return m * scale, i, s

        def merge(a, b):
            """(max, index, sum) of two partial results: the higher max, the lower index on a tie."""
            m = _fmax(a[0], b[0])
            sa = (a[0] > F(-INF)).select(a[2] * _exp2((a[0] - m) * F(LOG2E)), F(0.0))
            sb = (b[0] > F(-INF)).select(b[2] * _exp2((b[0] - m) * F(LOG2E)), F(0.0))
            _, i = _comb(a[0], a[1], b[0], b[1])
            return m, i, sa + sb

        def fold(r):
            for m_ in (1, 2, 4, 8, 16, 32):
                o = (
                    xor_lane(r[0], m_, lane),
                    xor_lane_i32(r[1], m_, lane),
                    xor_lane(r[2], m_, lane),
                )
                r = merge(r, o)
            return r

        t = fold(local_max_sum(x, True, rtemp if target_bf16 else F(1.0)))
        dd = fold(local_max_sum(d, False, rdraft))
        for k_, v_ in enumerate(
            (t[0].bitcast(I), t[1], t[2].bitcast(I), dd[0].bitcast(I), dd[2].bitcast(I))
        ):
            smem.red.ptr[k_ * 4 + wave] = v_  # every lane holds the same value
        fx.gpu.barrier()

        def lds(k_, w_):
            v_ = I(smem.red.ptr[k_ * 4 + w_])
            return v_.bitcast(F) if k_ in (0, 2, 3, 4) else v_

        r = (lds(0, 0), lds(1, 0), lds(2, 0))
        rd = (lds(3, 0), I(0), lds(4, 0))
        for w_ in range_constexpr(1, 4):
            r = merge(r, (lds(0, w_), lds(1, w_), lds(2, w_)))
            rd = merge(rd, (lds(3, w_), I(0), lds(4, w_)))
        sr = buffer_ops.create_buffer_resource_from_addr(
            arg_stats, num_records_bytes=LARGE
        )
        base = (row * NB + blk) * STATS_WORDS
        buffer_ops.buffer_store(
            Vec.from_elements(
                [r[0].bitcast(I), r[2].bitcast(I), r[1], rd[0].bitcast(I)], I
            ),
            sr,
            base,
            mask=tx == 0,
        )
        buffer_ops.buffer_store(rd[2].bitcast(I), sr, base + 4, mask=tx == 0)

    @flyc.jit
    def launch_rejection_stats(
        arg_t: fx.Int64,
        arg_d: fx.Int64,
        arg_idx: fx.Int64,
        arg_temp: fx.Int64,
        arg_stats: fx.Int64,
        i32_rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_t, arg_d, arg_idx, arg_temp, arg_stats).launch(
            grid=(NB, fx.Int64(i32_rows), 1), block=(NT, 1, 1), stream=stream
        )

    return launch_rejection_stats


def build_rejection_resample(synthetic: bool, target_bf16: bool):
    @fx.struct
    class Shared:
        red: fx.Array[fx.Int32, 16]

    @flyc.kernel(
        name="rejection_resample_v129280_s5"
        + ("_synthetic" if synthetic else "")
        + ("_tbf16" if target_bf16 else ""),
        known_block_size=[NT, 1, 1],
    )
    def kernel(
        arg_t: fx.Int64,  # target logits [B * 6, V]: fp32 (processed) or bf16 (raw)
        arg_d: fx.Int64,  # bf16 draft logits [R, 5, V]
        arg_ds: fx.Int64,  # int64 draft tokens [B * 6]
        arg_pos: fx.Int64,  # int64 [B * 6]
        arg_idx: fx.Int64,  # int32 [B]
        arg_temp: fx.Int64,  # fp32 [R]
        arg_seed: fx.Int64,  # int64 [R]
        arg_rates: fx.Int64,  # fp32 [5] (synthetic)
        arg_ws: fx.Int64,  # workspace: stats, block partials, ticket counters
        arg_out: fx.Int64,  # int64 [B, 6]
        arg_ns: fx.Int64,  # int32 [B]
        i32_reqs: fx.Int32,
    ):
        tx = fx.thread_idx.x
        wg = fx.block_idx.x
        # a request's NB blocks sit on one XCD (workgroup w runs on XCD w % 8) and meet in its L2
        req = (wg // (8 * NB)) * 8 + wg % 8
        blk = wg // 8 % NB
        smem = fx.SharedAllocator().allocate(Shared).peek()
        lane = tx % 64
        wave = I(fx.rocdl.readfirstlane(T.i32, tx // 64))
        req = (req < i32_reqs).select(req, I(0))
        live = (wg // (8 * NB)) * 8 + wg % 8 < i32_reqs
        lane6 = (lane < ROWS).select(lane, I(ROWS - 1))
        step = (lane < NSPEC).select(lane, I(NSPEC - 1))
        row = req * ROWS + lane6
        idr = buffer_ops.create_buffer_resource_from_addr(
            arg_idx, num_records_bytes=LARGE
        )
        dsr = buffer_ops.create_buffer_resource_from_addr(
            arg_ds, num_records_bytes=LARGE
        )
        psr = buffer_ops.create_buffer_resource_from_addr(
            arg_pos, num_records_bytes=LARGE
        )
        wsr = buffer_ops.create_buffer_resource_from_addr(
            arg_ws, num_records_bytes=STATS_BYTES + PART_BYTES
        )
        slot_raw = I(buffer_ops.buffer_load(idr, req, vec_width=1, dtype=I))
        tk = Vec(
            buffer_ops.buffer_load(
                dsr, (req * ROWS + step + 1) * 2, vec_width=2, dtype=I
            )
        )
        pv = Vec(buffer_ops.buffer_load(psr, row * 2, vec_width=2, dtype=I))
        tmax, tsum, targ, dmax, dsum = [], [], [], [], []
        for b in range_constexpr(NB):
            off = ((req * NSPEC + step) * NB + b) * STATS_WORDS
            s0 = Vec(buffer_ops.buffer_load(wsr, off, vec_width=4, dtype=I))
            s1 = I(buffer_ops.buffer_load(wsr, off + 4, vec_width=1, dtype=I))
            tmax.append(I(s0[0]).bitcast(F))
            tsum.append(I(s0[1]).bitcast(F))
            targ.append(I(s0[2]))
            dmax.append(I(s0[3]).bitcast(F))
            dsum.append(I(s1).bitcast(F))
        tok = I(tk[0])
        valid = I(tk[1]) >= 0
        tokc = valid.select(tok, I(0))
        slot = _uni(slot_raw)
        tr = buffer_ops.create_buffer_resource_from_addr(
            arg_temp, num_records_bytes=LARGE
        )
        sdr = buffer_ops.create_buffer_resource_from_addr(
            arg_seed, num_records_bytes=LARGE
        )
        temp = _uni(I(buffer_ops.buffer_load(tr, slot, vec_width=1, dtype=I))).bitcast(
            F
        )
        sd = Vec(buffer_ops.buffer_load(sdr, slot * 2, vec_width=2, dtype=I))
        greedy = temp == F(0.0)
        temp_eff = greedy.select(F(1.0), temp)
        tbytes = 2 if target_bf16 else 4
        tgt_base = arg_t + I64(req * ROWS) * (V * tbytes)
        tgr = buffer_ops.create_buffer_resource_from_addr(
            tgt_base, num_records_bytes=ROWS * V * tbytes
        )
        dfr = buffer_ops.create_buffer_resource_from_addr(
            arg_d + I64(slot) * (NSPEC * V * 2), num_records_bytes=NSPEC * V * 2
        )
        if const_expr(target_bf16):
            tw = I(
                buffer_ops.buffer_load(
                    tgr, (step * V + tokc) // 2, vec_width=1, dtype=I
                )
            )
            tgt_logit = ((tokc & 1) == 0).select(
                _unpack(tw, 0), _unpack(tw, 1)
            ) / temp_eff
        else:
            tgt_logit = I(
                buffer_ops.buffer_load(tgr, step * V + tokc, vec_width=1, dtype=I)
            ).bitcast(F)
        dw = I(
            buffer_ops.buffer_load(dfr, (step * V + tokc) // 2, vec_width=1, dtype=I)
        )
        seed_lo, seed_hi = _uni(I(sd[0])), _uni(I(sd[1]))
        u = _philox_uniform(seed_lo, seed_hi, I(pv[0]), I(pv[1]))

        def lse(maxes, sums):
            g = maxes[0]
            for e in maxes[1:]:
                g = _fmax(g, e)
            t = F(0.0)
            for m_, s_ in zip(maxes, sums):
                t = t + (m_ > F(-INF)).select(s_ * _exp(m_ - g), F(0.0))
            return g + _log(t)

        lse_t, lse_d = lse(tmax, tsum), lse(dmax, dsum)
        gmax = tmax[0]
        for e in tmax[1:]:
            gmax = _fmax(gmax, e)
        garg = I(0)
        for b in range_constexpr(NB - 1, -1, -1):
            garg = (tmax[b] == gmax).select(targ[b], garg)
        drf_logit = ((tokc & 1) == 0).select(_unpack(dw, 0), _unpack(dw, 1))
        if const_expr(synthetic):
            rr = buffer_ops.create_buffer_resource_from_addr(
                arg_rates, num_records_bytes=NSPEC * 4
            )
            rate = I(buffer_ops.buffer_load(rr, step, vec_width=1, dtype=I)).bitcast(F)
            accept = u < rate
        else:
            samp = (tgt_logit - lse_t) > _log(u) + (drf_logit / temp - lse_d)
            accept = greedy.select(tokc == garg, samp)
        accept = accept & valid & (lane < NSPEC)
        bal = I64(fx.rocdl.ballot(T.i64, accept))
        run = I(1)
        acc = I(0)
        for k in range_constexpr(NSPEC):
            run = run & ((bal >> I64(k)) & I64(1)).to(I)
            acc = acc + run
        bonus = acc == NSPEC
        lse_t_a, lse_d_a = _readlane(lse_t, acc), _readlane(lse_d, acc)
        valid_a = _readlane(valid.select(I(1), I(0)), acc)
        residual = ~greedy & ~bonus & (valid_a != 0)
        pos_lo_a = _readlane(I(pv[0]), acc)

        # ---- this block of the resample row
        h0 = _mix(_mix(_mix(I(0), seed_lo), seed_hi), pos_lo_a)
        vec = 8 if target_bf16 else 4
        rt = F(1.0) / temp_eff
        nlse_t = lse_t_a * F(-1.0)
        nlse_d = lse_d_a * F(-1.0)
        c0 = blk * BLK + tx * vec
        xr = buffer_ops.create_buffer_resource_from_addr(
            tgt_base + I64(acc) * (V * tbytes), num_records_bytes=V * tbytes
        )
        dr = buffer_ops.create_buffer_resource_from_addr(
            arg_d + I64(slot * NSPEC + bonus.select(I(0), acc)) * (V * 2),
            num_records_bytes=V * 2,
        )
        bv, bi = F(-INF), I(BIG)
        for q in range_constexpr(BLK // (NT * vec)):
            col = c0 + q * (NT * vec)
            ok = col < V
            if const_expr(target_bf16):
                xw = Vec(
                    buffer_ops.buffer_load(xr, col // 2, vec_width=4, dtype=I, mask=ok)
                )
                dv = Vec(
                    buffer_ops.buffer_load(
                        dr, col // 2, vec_width=4, dtype=I, mask=ok & residual
                    )
                )
            else:
                xv = Vec(buffer_ops.buffer_load(xr, col, vec_width=4, dtype=I, mask=ok))
                dv = Vec(
                    buffer_ops.buffer_load(
                        dr, col // 2, vec_width=2, dtype=I, mask=ok & residual
                    )
                )
            for k in range_constexpr(vec):
                if const_expr(target_bf16):
                    xk = ok.select(_unpack(I(xw[k // 2]), k % 2), F(-INF))
                else:
                    xk = ok.select(I(xv[k]).bitcast(F), F(-INF))
                if const_expr(target_bf16):
                    tlp = fx.math.fma(xk, rt, nlse_t)
                else:
                    tlp = xk - lse_t_a
                dlp = fx.math.fma(_unpack(I(dv[k // 2]), k % 2), rt, nlse_d)
                ratio = _exp(dlp - tlp)
                res = (ratio < F(1.0)).select(tlp + _log1p_neg(ratio), F(-INF))
                plain = xk * rt if const_expr(target_bf16) else xk
                base = residual.select(res, plain)
                noise = _noise(h0, (col + k) * _i32(0xCC9E2D51))
                val = ok.select(greedy.select(base, base + noise), F(-INF))
                better = val > bv
                bv = better.select(val, bv)
                bi = better.select(col + k, bi)
        for m in (1, 2, 4, 8, 16, 32):
            ov = xor_lane(bv, m, lane)
            oi = xor_lane_i32(bi, m, lane)
            bv, bi = _comb(bv, bi, ov, oi)
        smem.red.ptr[wave] = bv.bitcast(I)
        smem.red.ptr[4 + wave] = bi
        fx.gpu.barrier()
        bv = I(smem.red.ptr[0]).bitcast(F)
        bi = I(smem.red.ptr[4])
        for w_ in range_constexpr(1, 4):
            bv, bi = _comb(
                bv, bi, I(smem.red.ptr[w_]).bitcast(F), I(smem.red.ptr[4 + w_])
            )
        part_at = STATS_BYTES // 4 + (req * NB + blk) * 2
        buffer_ops.buffer_store(
            Vec.from_elements([bv.bitcast(I), bi], I),
            wsr,
            part_at,
            mask=(tx == 0) & live,
        )
        fx.rocdl.s_waitcnt(vmcnt=0)
        fx.gpu.barrier()
        cnt_addr = arg_ws + I64(STATS_BYTES + PART_BYTES) + I64(req) * 128
        if (tx == 0) & live:
            smem.red.ptr[8] = fx.atomic_add(
                _ticket_ptr(cnt_addr), I(1), syncscope=fx.rocdl.SyncScope.Agent
            )
        fx.gpu.barrier()
        if (I(smem.red.ptr[8]) == NB - 1) & live & (wave == 0):
            pb = lane < NB
            pw = Vec(
                buffer_ops.buffer_load(
                    wsr,
                    STATS_BYTES // 4 + (req * NB + lane) * 2,
                    vec_width=2,
                    dtype=I,
                    mask=pb,
                )
            )
            fv = pb.select(I(pw[0]), I(NEG_INF_BITS)).bitcast(F)
            fi = pb.select(I(pw[1]), I(BIG))
            for m in (1, 2, 4, 8):
                ov = xor_lane(fv, m, lane)
                oi = xor_lane_i32(fi, m, lane)
                fv, fi = _comb(fv, fi, ov, oi)
            token = _uni(fi)
            token = (token >= V).select(I(0), token)
            lo = (lane < acc).select(tok, (lane == acc).select(token, I(-1)))
            outr = buffer_ops.create_buffer_resource_from_addr(
                arg_out, num_records_bytes=LARGE
            )
            buffer_ops.buffer_store(
                Vec.from_elements([lo, (lo < 0).select(I(-1), I(0))], I),
                outr,
                (req * ROWS + lane) * 2,
                mask=lane < ROWS,
            )
            nsr = buffer_ops.create_buffer_resource_from_addr(
                arg_ns, num_records_bytes=LARGE
            )
            buffer_ops.buffer_store(acc + 1, nsr, req, mask=lane == 0)
            fx.atomic_xchg(
                _ticket_ptr(cnt_addr), I(0), syncscope=fx.rocdl.SyncScope.Agent
            )

    @flyc.jit
    def launch_rejection_resample(
        arg_t: fx.Int64,
        arg_d: fx.Int64,
        arg_ds: fx.Int64,
        arg_pos: fx.Int64,
        arg_idx: fx.Int64,
        arg_temp: fx.Int64,
        arg_seed: fx.Int64,
        arg_rates: fx.Int64,
        arg_ws: fx.Int64,
        arg_out: fx.Int64,
        arg_ns: fx.Int64,
        i32_reqs: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_t,
            arg_d,
            arg_ds,
            arg_pos,
            arg_idx,
            arg_temp,
            arg_seed,
            arg_rates,
            arg_ws,
            arg_out,
            arg_ns,
            i32_reqs,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(NT, 1, 1), stream=stream)

    return launch_rejection_resample

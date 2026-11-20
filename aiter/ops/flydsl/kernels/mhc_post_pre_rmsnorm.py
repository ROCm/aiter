# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""mHC seam for hidden 5120, hc_mult 4, gfx950, decode-sized row counts.

One launch does the post-mix of the previous sub-layer, the gate projection
``flatten(R') @ fn^T``, the pre / post / Sinkhorn comb gates, the collapse with the
carried pre gate and the RMSNorm of the collapse (optionally also its MXFP8
quantization and the mean of the four new streams)::

    R'_j   = bf16(p_j * y + sum_h c[h][j] * R_h)               HAS_POST (else R' = R)
    mix    = flatten(R') @ fn^T,  rstd = rsqrt(mean(R'^2) + rms_eps)
    pre'   = sigmoid(mix[0:4]  * rstd * s0 + b[0:4]) + hc_pre_eps
    post'  = sigmoid(mix[4:8]  * rstd * s1 + b[4:8]) * hc_post_mult
    comb'  = Sinkhorn(mix[8:24] * rstd * s2 + b[8:24])
    x1     = bf16(sum_j q_j * R'_j)                             (IDENT_PRE: x1 = R'_0)
    out    = bf16(x1 * rsqrt(mean(x1^2) + norm_eps) * w)
    QUANT  : e4m3 codes [T, H] + row-major E8M0 bytes [T, H / 32] of ``out``
    MEAN   : bf16((R'_0 + R'_1 + R'_2 + R'_3) / 4)

Work split. A workgroup owns 16 rows and ``160 / KS`` 32-column chunks of all four
streams; its ``WAVES`` waves split the chunks. A lane owns row ``l % 16`` and 8
columns ``8 * (l / 16)`` of a chunk, which is the A-operand layout of
``v_mfma_f32_16x16x32_bf16``: the post-mix, the collapse and the square sums are
lane-local, and the bf16 new residual feeds the MFMA straight from registers (no LDS
transpose). ``fn`` is pre-packed by the host as bf16 hi + lo B operands, three 1 KiB
operands per (stream, chunk): hi of mixes 0-15, lo of mixes 0-15, and hi | lo of
mixes 16-23 in lane halves.

Hand-off (``KS > 1``). Every split writes a partial row per row of its block
(24 mixes and the two square sums) and bumps the block's counter; the last arrival
sums the partials in split order (deterministic), publishes the x1 rstd of the
block's rows and computes the gates. The other splits spin on the counter's
generation and rescale their own x1 columns, which stayed in registers. The counter
word is ``arrivals | generation << 8`` and re-arms itself (the publish adds
``256 - KS``), so replays need no memset. All splits of a block run on one XCD
(workgroup ``b`` runs on XCD ``b % 8``), so they share an L2 and workgroup-scope
fences order the partials. A waiting split depends on the last arrival being
scheduled, so the host keeps ``blocks * KS`` within the CU count and runs no
split-K variant under a CU mask.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import AITER_FLYDSL_KERNARG_PRELOAD

H = 5120
CHUNKS = H // 32
ROW_DW = H // 2  # dwords of one bf16 stream row
N_STREAMS = 4
N_MIX = 24
BLOCK_M = 16
PSLOT = 32  # partial row: [0:24] mixes, [24] sum R'^2, [25] sum x1^2
CTRL_INTS = 16 + BLOCK_M  # per block: word 0 = counter, [16:32] = rstd of its rows
LDS_BYTES = 160 * 1024
KERNARG_PRELOAD_COUNT = 16
DPP_QUAD_SWAP1, DPP_QUAD_SWAP2 = 0xB1, 0x4E  # quad_perm [1,0,3,2] / [2,3,0,1]
DPP_ROW_ROR4, DPP_ROW_ROR8 = 0x124, 0x128
CM_L1_BYPASS = 1  # sc0: miss the CU's L1, hit the XCD's L2


def _put(t, idx, val):
    """``t[idx] = val`` outside the kernel AST, so a store inside a runtime branch is
    not read as a rebinding of ``t``."""
    t[idx] = val


def _f32(bits):
    return fx.Int32(bits).bitcast(fx.Float32)


def _bits(x):
    return fx.Float32(x).bitcast(fx.Int32)


def _dpp_f32(v, ctrl):
    # No fx spelling of an undef operand. bound_ctrl with full row / bank masks
    # writes every lane, so the old value is dead and need not be materialized.
    old = llvm.mlir_undef(T.i32)
    r = rocdl.update_dpp(T.i32, old, _bits(v).ir_value(), ctrl, 0xF, 0xF, True)
    return fx.Int32(r).bitcast(fx.Float32)


def _rcp(x):
    return fx.Float32(rocdl.rcp(T.f32, fx.Float32(x).ir_value()))


def _exp2(x):
    return fx.Float32(rocdl.exp2(T.f32, fx.Float32(x).ir_value()))


def _rsq(x):
    # v_rsq_f32: the precision the reference contract was validated with
    return fx.Float32(rocdl.rsq(T.f32, fx.Float32(x).ir_value()))


def _unpack8(v):
    """Four dwords of packed bf16 -> eight fp32 values."""
    out = []
    for i in range(4):
        w = fx.Int32(v[i])
        out += [(w << 16).bitcast(fx.Float32), (w & -65536).bitcast(fx.Float32)]
    return out


def _pk(a, b):
    """Two fp32 -> two bf16 in a dword (round to nearest even), ``a`` low."""
    return fx.Int32(
        rocdl.cvt_pk_bf16_f32(fx.Float32(a).ir_value(), fx.Float32(b).ir_value())
    )


def _pack8(vals):
    return [_pk(vals[2 * i], vals[2 * i + 1]) for i in range(4)]


def _permlane_swap(op, v):
    # The rocdl permlane swaps return an LLVM struct of both halves; fx has no
    # typed wrapper for it, so the struct is built and unpacked through llvm.
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    src = _bits(v).ir_value()
    res = op(pair_ty, src, src, False, False)
    return (
        _f32(llvm.extractvalue(T.i32, res, [0])),
        _f32(llvm.extractvalue(T.i32, res, [1])),
    )


def _xlane4_max(x):
    """Max over lanes L, L ^ 16, L ^ 32, L ^ 48."""
    a, b = _permlane_swap(rocdl.permlane32_swap, x)
    m = fx.maxnumf(a, b)
    a, b = _permlane_swap(rocdl.permlane16_swap, m)
    return fx.maxnumf(a, b)


def _mfma(a4, b4, acc):
    a = Vec(a4).bitcast(fx.BFloat16)
    b = Vec(b4).bitcast(fx.BFloat16)
    return Vec(
        rocdl.mfma_f32_16x16x32_bf16(Vec.make_type(4, fx.Float32), [a, b, acc, 0, 0, 0])
    )


def _codes(yb, rescale):
    """e4m3 codes of 8 values times ``rescale`` as two dwords."""
    v2i16 = ir.VectorType.get([2], ir.IntegerType.get_signless(16))

    def cvt(old, a, b, hi):
        # v_cvt_scalef32_pk_fp8_f32 with scale 1.0 on prescaled values; called
        # through the intrinsic because the packed-dword old operand is an i32 here.
        res = llvm.call_intrinsic(
            v2i16,
            "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
            [
                llvm.bitcast(v2i16, old.ir_value()),
                fx.Float32(a).ir_value(),
                fx.Float32(b).ir_value(),
                fx.Float32(1.0).ir_value(),
                fx.Boolean(hi).ir_value(),
            ],
            [],
            [],
        )
        return fx.Int32(llvm.bitcast(T.i32, res))

    q = [v * rescale for v in yb]
    return [
        cvt(
            cvt(fx.Int32(0), q[2 * i], q[2 * i + 1], False),
            q[2 * i + 2],
            q[2 * i + 3],
            True,
        )
        for i in (0, 2)
    ]


def _e8m0(yb):
    """(byte as i32, 2^(127 - byte)) of the 32-column group whose 4 lanes hold ``yb``.

    byte = clamp(ceil(log2(amax / 448)) + 127, 0, 254), computed on the fp32 bits.
    """
    a = fx.math.absf(yb[0])
    for v in yb[1:]:
        a = fx.maxnumf(a, fx.math.absf(v))
    a = fx.maxnumf(_xlane4_max(a), _f32(0x00800000))
    bits = (a / fx.Float32(448.0)).bitcast(fx.Int32)
    up = (bits + 0x7FFFFF) >> 23
    ex = (bits <= 0x00400000).select(
        fx.Int32(0),
        (up < 1).select(fx.Int32(1), (up > 254).select(fx.Int32(254), up)),
    )
    return ex, ((254 - ex) << 23).bitcast(fx.Float32)


def lds_bytes(waves: int) -> int:
    return 4 * (waves * BLOCK_M * PSLOT + waves * 128 + BLOCK_M * PSLOT + BLOCK_M + 4)


@functools.cache
def compile_mhc_post_pre_rmsnorm(
    ks: int,
    waves: int,
    has_post: bool,
    ident_pre: bool,
    quant: bool,
    mean: bool,
    sinkhorn_repeat: int,
):
    """``@flyc.jit`` launcher of one (split count, waves, variant) configuration."""
    bm = BLOCK_M
    assert CHUNKS % ks == 0 and (CHUNKS // ks) % waves == 0 and ks <= 80
    assert lds_bytes(waves) <= LDS_BYTES, "LDS"
    assert sinkhorn_repeat >= 1
    repeat = sinkhorn_repeat
    chs = CHUNKS // ks
    cpw = chs // waves
    nthr = 64 * waves
    ci = CTRL_INTS
    name = f"mhc_post_pre_rmsnorm_h5120_ks{ks}_w{waves}_r{repeat}" + "".join(
        t
        for t, c in (
            ("_post", has_post),
            ("_ident", ident_pre),
            ("_mx", quant),
            ("_mean", mean),
        )
        if c
    )

    @fx.struct
    class Shared:
        red: fx.Array[fx.Float32, waves * bm * PSLOT]
        sq: fx.Array[fx.Float32, waves * 2 * 64]
        fin: fx.Array[fx.Float32, bm * PSLOT]
        rst: fx.Array[fx.Float32, bm]
        flag: fx.Array[fx.Int32, 4]

    @flyc.kernel(name=name, known_block_size=[nthr, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_rms_eps: fx.Int32,
        i32_pre_eps: fx.Int32,
        i32_sk_eps: fx.Int32,
        i32_post_mult: fx.Int32,
        i32_norm_eps: fx.Int32,
        arg_res: fx.Int64,  # [m, 4, H] bf16
        arg_fn: fx.Int64,  # packed B operands
        arg_y: fx.Int64,  # [m, H] bf16
        arg_pm: fx.Int64,  # [m, 4] f32
        arg_cm: fx.Int64,  # [m, 4, 4] f32
        arg_pre: fx.Int64,  # [m, 4] f32
        arg_scale: fx.Int64,  # [3] f32
        arg_base: fx.Int64,  # [24] f32
        arg_nw: fx.Int64,  # [H] bf16
        arg_rout: fx.Int64,  # [m, 4, H] bf16
        arg_post: fx.Int64,  # [m, 4] f32
        arg_comb: fx.Int64,  # [m, 16] f32
        arg_npre: fx.Int64,  # [m, 4] f32
        arg_out: fx.Int64,  # [m, H] bf16
        arg_codes: fx.Int64,  # [m, H] e4m3
        arg_qs: fx.Int64,  # [m, H / 32] u8
        arg_mean: fx.Int64,  # [m, H] bf16
        arg_part: fx.Int64,  # [blocks, bm, ks, 32] f32
        arg_ctrl: fx.Int64,  # [blocks, ci] i32
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        q8 = bid // 8
        blk = (q8 // ks) * 8 + bid % 8
        split = q8 % ks
        nblk = (i32_m + bm - 1) // bm
        if blk < nblk:
            lane = tx % 64
            wave = fx.Int32(rocdl.readfirstlane(T.i32, tx // 64))
            row = lane % 16
            g = lane // 16
            rowt = blk * bm + row
            smem = fx.SharedAllocator().allocate(Shared).peek()
            red, sq, fin = smem.red.ptr, smem.sq.ptr, smem.fin.ptr
            rst, flag = smem.rst.ptr, smem.flag.ptr
            m64 = fx.Int64(i32_m)

            def rsrc(addr, nbytes):
                return buffer_ops.create_buffer_resource_from_addr(
                    addr, num_records_bytes=nbytes
                )

            def ldf(r, off, n):
                v = buffer_ops.buffer_load(r, off, vec_width=n, dtype=fx.Float32)
                return Vec(v) if n > 1 else fx.Float32(v)

            def ld4(r, off):
                return Vec(buffer_ops.buffer_load(r, off, vec_width=4, dtype=fx.Int32))

            rr = rsrc(arg_res, m64 * (N_STREAMS * H * 2))
            fr = rsrc(arg_fn, N_STREAMS * CHUNKS * 3 * 1024)
            nwr = rsrc(arg_nw, H * 2)
            outr = rsrc(arg_out, m64 * (H * 2))
            if fx.const_expr(has_post):
                yr = rsrc(arg_y, m64 * (H * 2))
                ror = rsrc(arg_rout, m64 * (N_STREAMS * H * 2))
                pmr = rsrc(arg_pm, m64 * 16)
                cmr = rsrc(arg_cm, m64 * 64)
            prer = rsrc(arg_pre, m64 * 16)
            if fx.const_expr(quant):
                cdr = rsrc(arg_codes, m64 * H)
                qsr = rsrc(arg_qs, m64 * (H // 32))
            if fx.const_expr(mean):
                mnr = rsrc(arg_mean, m64 * (H * 2))

            if fx.const_expr(has_post):
                pv = ldf(pmr, rowt * 4, 4)
                cv = [ldf(cmr, rowt * 16 + 4 * h, 4) for h in range_constexpr(4)]
            if fx.const_expr(not ident_pre):
                qv = ldf(prer, rowt * 4, 4)

            cbase = split * chs + wave * cpw
            wts = [ld4(nwr, (cbase + i) * 16 + g * 4) for i in range_constexpr(cpw)]

            acc12 = Vec.filled(4, 0.0, fx.Float32)
            acc3 = Vec.filled(4, 0.0, fx.Float32)
            sqr = fx.Float32(0.0)
            sqx = fx.Float32(0.0)
            x1s = []
            for i in range_constexpr(cpw):
                cg = cbase + i
                cd = cg * 16 + g * 4
                bops = [
                    [
                        ld4(fr, ((s * CHUNKS + cg) * 3 + o) * 256 + lane * 4)
                        for o in range_constexpr(3)
                    ]
                    for s in range_constexpr(N_STREAMS)
                ]
                rs = [
                    ld4(rr, rowt * (N_STREAMS * ROW_DW) + s * ROW_DW + cd)
                    for s in range_constexpr(N_STREAMS)
                ]
                if fx.const_expr(has_post):
                    yv = _unpack8(ld4(yr, rowt * ROW_DW + cd))
                    f = [_unpack8(r) for r in rs]
                    pk = []
                    for j in range_constexpr(N_STREAMS):
                        vals = []
                        for e in range_constexpr(8):
                            a = pv[j] * yv[e]
                            for h in range_constexpr(N_STREAMS):
                                a = fx.math.fma(cv[h][j], f[h][e], a)
                            vals.append(a)
                        pk.append(_pack8(vals))
                        buffer_ops.buffer_store(
                            Vec.from_elements(pk[j], fx.Int32),
                            ror,
                            rowt * (N_STREAMS * ROW_DW) + j * ROW_DW + cd,
                        )
                else:
                    pk = [[fx.Int32(r[k]) for k in range(4)] for r in rs]
                gf = [_unpack8(p) for p in pk]
                for s in range_constexpr(N_STREAMS):
                    for e in range_constexpr(8):
                        sqr = fx.math.fma(gf[s][e], gf[s][e], sqr)
                if fx.const_expr(mean):
                    mv = [
                        ((gf[0][e] + gf[1][e]) + gf[2][e] + gf[3][e]) * fx.Float32(0.25)
                        for e in range_constexpr(8)
                    ]
                    buffer_ops.buffer_store(
                        Vec.from_elements(_pack8(mv), fx.Int32),
                        mnr,
                        rowt * ROW_DW + cd,
                    )
                if fx.const_expr(ident_pre):
                    x1v = gf[0]
                else:
                    x1v = []
                    for e in range_constexpr(8):
                        a = qv[0] * gf[0][e]
                        for s in range_constexpr(1, N_STREAMS):
                            a = fx.math.fma(qv[s], gf[s][e], a)
                        x1v.append(a)
                x1p = _pack8(x1v)
                x1s.append(x1p)
                x1f = _unpack8(x1p)
                for e in range_constexpr(8):
                    sqx = fx.math.fma(x1f[e], x1f[e], sqx)
                for s in range_constexpr(N_STREAMS):
                    a4 = Vec.from_elements(pk[s], fx.Int32)
                    acc12 = _mfma(a4, bops[s][0], acc12)
                    acc12 = _mfma(a4, bops[s][1], acc12)
                    acc3 = _mfma(a4, bops[s][2], acc3)

            for e in range_constexpr(4):
                red[(wave * bm + 4 * g + e) * PSLOT + row] = acc12[e]
                red[(wave * bm + 4 * g + e) * PSLOT + 16 + row] = acc3[e]
            sq[(wave * 2) * 64 + lane] = sqr
            sq[(wave * 2 + 1) * 64 + lane] = sqx
            gpu.barrier()

            def mix_value(o):
                """Output o = row * 32 + slot of the split's partial row
                (slots 0-23 mixes, 24-25 square sums)."""
                ro, slot = o // PSLOT, o % PSLOT
                comb = (slot >= 16) & (slot < 24)
                lo = (slot < 24).select(slot, fx.Int32(0))
                hi = comb.select(slot + 8, fx.Int32(0))
                v = fx.Float32(0.0)
                s2 = fx.Float32(0.0)
                s3 = fx.Float32(0.0)
                for w in range_constexpr(waves):
                    a = red[(w * bm + ro) * PSLOT + lo]
                    b = red[(w * bm + ro) * PSLOT + hi]
                    v = v + a + comb.select(b, fx.Float32(0.0))
                    for gg in range_constexpr(4):
                        s2 = s2 + sq[(w * 2) * 64 + gg * 16 + ro]
                        s3 = s3 + sq[(w * 2 + 1) * 64 + gg * 16 + ro]
                return (slot == 24).select(
                    s2,
                    (slot == 25).select(s3, (slot < 24).select(v, fx.Float32(0.0))),
                )

            part_r = rsrc(arg_part, fx.Int64(nblk) * (bm * ks * PSLOT * 4))
            ctrl_r = rsrc(arg_ctrl, fx.Int64(nblk) * (ci * 4))
            ctrl_ptr_ty = fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4)

            def word_ptr(slot):
                return fx.inttoptr(
                    ctrl_ptr_ty, arg_ctrl + fx.Int64(blk) * (ci * 4) + slot * 4
                )

            def fence(ordering):
                fx.llvm.memory_fence(
                    syncscope=rocdl.SyncScope.Workgroup, ordering=ordering
                )

            def finish_gates():
                """Gates of the block's rows from ``fin``; wave rg handles rows
                4 * rg .. 4 * rg + 3, 16 lanes (one per comb entry) per row."""
                scr = rsrc(arg_scale, 12)
                s0 = ldf(scr, 0, 1)
                s1 = ldf(scr, 1, 1)
                s2 = ldf(scr, 2, 1)
                br = rsrc(arg_base, 96)
                rms_eps = i32_rms_eps.bitcast(fx.Float32)
                sk_eps = i32_sk_eps.bitcast(fx.Float32)
                pre_eps = i32_pre_eps.bitcast(fx.Float32)
                post_mult = i32_post_mult.bitcast(fx.Float32)
                for p in range_constexpr(-(-(bm // 4) // waves)):
                    rg = wave + waves * p
                    if rg < bm // 4:
                        lr = rg * 4 + lane // 16
                        e = lane % 16
                        rgl = blk * bm + lr
                        rs = _rsq(
                            fin[lr * PSLOT + 24] * fx.Float32(1.0 / (N_STREAMS * H))
                            + rms_eps
                        )
                        a = fin[lr * PSLOT + 8 + e] * rs * s2 + ldf(br, 8 + e, 1)
                        mx = fx.maxnumf(a, _dpp_f32(a, DPP_QUAD_SWAP1))
                        mx = fx.maxnumf(mx, _dpp_f32(mx, DPP_QUAD_SWAP2))
                        ex = _exp2((a - mx) * fx.Float32(LOG2E))
                        sm = ex + _dpp_f32(ex, DPP_QUAD_SWAP1)
                        sm = sm + _dpp_f32(sm, DPP_QUAD_SWAP2)
                        pm = ex * _rcp(sm) + sk_eps
                        cs = pm + _dpp_f32(pm, DPP_ROW_ROR4)
                        cs = cs + _dpp_f32(cs, DPP_ROW_ROR8)
                        pm = pm * _rcp(cs + sk_eps)
                        for _ in range_constexpr(repeat - 1):
                            sm = pm + _dpp_f32(pm, DPP_QUAD_SWAP1)
                            sm = sm + _dpp_f32(sm, DPP_QUAD_SWAP2)
                            pm = pm * _rcp(sm + sk_eps)
                            cs = pm + _dpp_f32(pm, DPP_ROW_ROR4)
                            cs = cs + _dpp_f32(cs, DPP_ROW_ROR8)
                            pm = pm * _rcp(cs + sk_eps)
                        buffer_ops.buffer_store(
                            pm, rsrc(arg_comb, m64 * 64), rgl * 16 + e
                        )
                        ep = e % 8
                        v = fin[lr * PSLOT + ep] * rs * (e < 4).select(s0, s1) + ldf(
                            br, ep, 1
                        )
                        sg = _rcp(fx.Float32(1.0) + _exp2(v * fx.Float32(-LOG2E)))
                        buffer_ops.buffer_store(
                            sg + pre_eps,
                            rsrc(arg_npre, m64 * 16),
                            rgl * 4 + e,
                            mask=e < 4,
                        )
                        buffer_ops.buffer_store(
                            sg * post_mult,
                            rsrc(arg_post, m64 * 16),
                            rgl * 4 + e - 4,
                            mask=(e >= 4) & (e < 8),
                        )

            def normalize(rstd):
                for i in range_constexpr(cpw):
                    cg = cbase + i
                    x1f = _unpack8(x1s[i])
                    wf = _unpack8(wts[i])
                    yp = _pack8([(x1f[e] * rstd) * wf[e] for e in range_constexpr(8)])
                    buffer_ops.buffer_store(
                        Vec.from_elements(yp, fx.Int32),
                        outr,
                        rowt * ROW_DW + cg * 16 + g * 4,
                    )
                    if fx.const_expr(quant):
                        yb = _unpack8(yp)
                        ex, rescale = _e8m0(yb)
                        cd = _codes(yb, rescale)
                        buffer_ops.buffer_store(
                            Vec.from_elements(cd, fx.Int32),
                            cdr,
                            rowt * (H // 4) + cg * 8 + g * 2,
                        )
                        buffer_ops.buffer_store(
                            ex.to(fx.Int8),
                            qsr,
                            rowt * (H // 32) + cg,
                            mask=g == 0,
                            offset_is_bytes=True,
                        )

            def sum_partials():
                for it in range_constexpr(-(-bm * PSLOT // nthr)):
                    o = it * nthr + tx
                    if o < bm * PSLOT:
                        base_idx = ((blk * bm + o // PSLOT) * ks) * PSLOT + o % PSLOT
                        v = fx.Float32(0.0)
                        for k in range_constexpr(ks):
                            v = v + fx.Float32(
                                buffer_ops.buffer_load(
                                    part_r,
                                    base_idx + k * PSLOT,
                                    vec_width=1,
                                    dtype=fx.Float32,
                                    cache_modifier=CM_L1_BYPASS,
                                )
                            )
                        _put(fin, o, v)

            nrm_eps_v = i32_norm_eps.bitcast(fx.Float32)

            def rstd_of(slot_row):
                return _rsq(
                    fin[slot_row * PSLOT + 25] * fx.Float32(1.0 / H) + nrm_eps_v
                )

            if fx.const_expr(ks == 1):
                for it in range_constexpr(-(-bm * PSLOT // nthr)):
                    o = it * nthr + tx
                    if o < bm * PSLOT:
                        _put(fin, o, mix_value(o))
                gpu.barrier()
                normalize(rstd_of(row))
                finish_gates()
            else:
                for it in range_constexpr(-(-bm * PSLOT // nthr)):
                    o = it * nthr + tx
                    if o < bm * PSLOT:
                        buffer_ops.buffer_store(
                            mix_value(o),
                            part_r,
                            ((blk * bm + o // PSLOT) * ks + split) * PSLOT + o % PSLOT,
                        )
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tx == 0:
                    fence(fx.AtomicOrdering.Release)
                    old = fx.llvm.atomic_add(
                        word_ptr(0),
                        fx.Int32(1),
                        syncscope=rocdl.SyncScope.Agent,
                        ordering=fx.AtomicOrdering.Monotonic,
                    )
                    flag[0] = old
                gpu.barrier()
                old_w = flag[0]
                if (old_w & 255) == ks - 1:
                    # last arrival: sum the partials, publish the rstd, finish
                    fence(fx.AtomicOrdering.Acquire)
                    sum_partials()
                    gpu.barrier()
                    for it in range_constexpr(-(-bm // nthr)):
                        r_i = it * nthr + tx
                        if r_i < bm:
                            rv = rstd_of(r_i)
                            _put(rst, r_i, rv)
                            buffer_ops.buffer_store(rv, ctrl_r, blk * ci + 16 + r_i)
                    rocdl.s_waitcnt(vmcnt=0)
                    gpu.barrier()
                    if tx == 0:
                        fence(fx.AtomicOrdering.Release)
                        fx.llvm.atomic_add(
                            word_ptr(0),
                            fx.Int32(256 - ks),
                            syncscope=rocdl.SyncScope.Agent,
                            ordering=fx.AtomicOrdering.Monotonic,
                        )
                    normalize(rst[row])
                    finish_gates()
                else:
                    # wait for the publish (generation change), then rescale own x1
                    if tx == 0:
                        sp = word_ptr(0)
                        cur = fx.generic_load(
                            sp,
                            dtype=fx.Int32,
                            memory_order=fx.AtomicOrdering.Monotonic,
                            syncscope=rocdl.SyncScope.Agent,
                        )
                        while (cur >> 8) == (old_w >> 8):
                            rocdl.s_sleep(1)
                            cur = fx.generic_load(
                                sp,
                                dtype=fx.Int32,
                                memory_order=fx.AtomicOrdering.Monotonic,
                                syncscope=rocdl.SyncScope.Agent,
                            )
                    gpu.barrier()
                    fence(fx.AtomicOrdering.Acquire)
                    normalize(
                        fx.Float32(
                            buffer_ops.buffer_load(
                                ctrl_r,
                                blk * ci + 16 + row,
                                vec_width=1,
                                dtype=fx.Float32,
                                cache_modifier=CM_L1_BYPASS,
                            )
                        )
                    )

    @flyc.jit
    def launch(
        arg_res: fx.Int64,
        arg_fn: fx.Int64,
        arg_y: fx.Int64,
        arg_pm: fx.Int64,
        arg_cm: fx.Int64,
        arg_pre: fx.Int64,
        arg_scale: fx.Int64,
        arg_base: fx.Int64,
        arg_nw: fx.Int64,
        arg_rout: fx.Int64,
        arg_post: fx.Int64,
        arg_comb: fx.Int64,
        arg_npre: fx.Int64,
        arg_out: fx.Int64,
        arg_codes: fx.Int64,
        arg_qs: fx.Int64,
        arg_mean: fx.Int64,
        arg_part: fx.Int64,
        arg_ctrl: fx.Int64,
        i32_m: fx.Int32,
        i32_rms_eps: fx.Int32,
        i32_pre_eps: fx.Int32,
        i32_sk_eps: fx.Int32,
        i32_post_mult: fx.Int32,
        i32_norm_eps: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            i32_m,
            i32_rms_eps,
            i32_pre_eps,
            i32_sk_eps,
            i32_post_mult,
            i32_norm_eps,
            arg_res,
            arg_fn,
            arg_y,
            arg_pm,
            arg_cm,
            arg_pre,
            arg_scale,
            arg_base,
            arg_nw,
            arg_rout,
            arg_post,
            arg_comb,
            arg_npre,
            arg_out,
            arg_codes,
            arg_qs,
            arg_mean,
            arg_part,
            arg_ctrl,
        ).launch(
            grid=(fx.Int64(((i32_m + 8 * bm - 1) // (8 * bm)) * 8 * ks), 1, 1),
            block=(nthr, 1, 1),
            stream=stream,
        )

    launch.kernel_name = name
    launch.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": KERNARG_PRELOAD_COUNT,
        }
    }
    return launch

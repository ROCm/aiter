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

x1 until the norm. The x1 rstd needs the whole row, so x1 waits for it either in
registers (4 VGPRs per chunk a wave owns) or, where a wave owns more than
``STAGE_CHUNKS`` chunks and registers would spill, unscaled in the bf16 output,
re-read once the rstd is known (``stage_x1``).

Hand-off (``KS > 1``). Every split writes a partial row per row of its block
(24 mixes and the two square sums) and bumps the block's counter word; the last
arrival sums the partials in split order (deterministic), publishes the x1 rstd of the
block's rows and computes the gates. The other splits wait for the word's generation
to change and rescale their own x1 columns. The word is ``arrivals | hand-off bits << 8
| generation << 28``; the publish adds ``2^28 - KS`` (arrivals back to 0, generation
+ 1), so it re-arms itself and replays need no memset. All splits of a block run on one
XCD (workgroup ``b`` runs on XCD ``b % 8``), so they share an L2 and workgroup-scope
fences order the partials.

Bounded wait. A split that has polled ``SPIN`` times without a publish stages its
unscaled x1 in the output, sets its hand-off bit with a compare-exchange that
expects the old generation, and exits, freeing its CU. If the exchange sees a new
generation the publish came first and the split finishes its columns itself (the
rstd is out before the publish). The publish returns the word as it was, so the last
arrival learns the hand-off bits from it without another round trip; it reads them
after its own columns and gates, rescales the staged columns of each set bit and
clears the bits. A split therefore never waits longer than ``SPIN`` polls on a
workgroup that cannot be scheduled: seams on several streams, or next to kernels
holding CUs, finish (slower) instead of hanging. The host still gives every real
workgroup a CU of its XCD (``ceil(blocks / 8) * KS <= CUs / 8``) so the hand-off stays
the rare path.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    buf_base_i64,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

F32 = fx.Float32
I32 = fx.Int32
H = 5120
CHUNKS = H // 32
U_ROW = H // 8  # 16 B units of one bf16 stream row
N_STREAMS = 4
N_MIX = 24
BLOCK_M = 16
PSLOT = 32  # partial row: [0:24] mixes, [24] sum R'^2, [25] sum x1^2
# per block 32 ints: [0] counter word (arrivals | hand-off bits << 8 | generation
# << 28), [2] hand-off count (diagnostic), [16:32] rstd of the block's rows
CTRL_INTS = 16 + BLOCK_M
CTRL_TMO = 2
CTRL_RSTD = 16
MAX_KS = 20  # hand-off bits 8 ... 27 of the counter word
GEN_SHIFT = 28
STAGE_CHUNKS = 8  # more chunks per wave than this: x1 waits in the output
DIST_SPIN = 1024  # polls (s_sleep 1 + a load each) before a split hands off
LDS_BYTES = 160 * 1024
DPP_QUAD_SWAP1, DPP_QUAD_SWAP2 = 0xB1, 0x4E  # quad_perm [1,0,3,2] / [2,3,0,1]
DPP_ROW_ROR4, DPP_ROW_ROR8 = 0x124, 0x128
CM_L1_BYPASS = 1  # sc0: miss the CU's L1, hit the XCD's L2


def _put(t, idx, val):
    """``t[idx] = val`` outside the kernel AST, so a store inside a runtime branch is
    not read as a rebinding of ``t``."""
    t[idx] = val


def _f32(bits):
    return I32(bits).bitcast(F32)


def _bits(x):
    return F32(x).bitcast(I32)


def _dpp_f32(v, ctrl):
    # rocdl.update_dpp with an undef old value and bound_ctrl: every lane is written,
    # and LLVM folds the move into the consuming op (v_add_f32_dpp). dpp_utils'
    # helpers pass the source as the old value with bound_ctrl off, which blocks
    # that fold: the ISA then has a v_mov_b32_dpp and an s_nop per use on the
    # finisher's serial Sinkhorn chain (0.2-0.5 us slower at 160 rows in an A/B).
    old = llvm.mlir_undef(T.i32)
    r = rocdl.update_dpp(T.i32, old, _bits(v).ir_value(), ctrl, 0xF, 0xF, True)
    return I32(r).bitcast(F32)


# v_rcp_f32 / v_exp_f32 / v_rsq_f32 directly: fx division, fx.math.exp2 and
# fx.math.rsqrt lower to the IEEE-accurate sequences (11, 6 and ~25 VALU ops on gfx950,
# checked in the ISA); the reference contract was validated with the bare instructions.
def _rcp(x):
    return F32(rocdl.rcp(T.f32, F32(x).ir_value()))


def _exp2(x):
    return F32(rocdl.exp2(T.f32, F32(x).ir_value()))


def _rsq(x):
    return F32(rocdl.rsq(T.f32, F32(x).ir_value()))


def _unpack8(v):
    """Four dwords of packed bf16 -> eight fp32 values."""
    out = []
    for i in range(4):
        w = I32(v[i])
        out += [(w << 16).bitcast(F32), (w & -65536).bitcast(F32)]
    return out


def _pack8(vals):
    """Eight fp32 -> four dwords of bf16 (round to nearest even, v_cvt_pk_bf16_f32)."""
    return Vec(Vec.from_elements(vals, F32).to(fx.BFloat16)).bitcast(I32)


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
    # One v_mfma_f32_16x16x32_bf16 on operands that already sit in registers in the
    # A / B lane layout; the fx MMA atoms take partitioned tensor fragments instead.
    a = Vec(a4).bitcast(fx.BFloat16)
    b = Vec(b4).bitcast(fx.BFloat16)
    return Vec(
        rocdl.mfma_f32_16x16x32_bf16(Vec.make_type(4, F32), [a, b, acc, 0, 0, 0])
    )


def _codes(yb, rescale):
    """e4m3 codes of 8 values times ``rescale`` as two dwords."""
    v2i16 = ir.VectorType.get([2], ir.IntegerType.get_signless(16))

    def cvt(old, a, b, hi):
        # v_cvt_scalef32_pk_fp8_f32 with scale 1.0 on prescaled values; no fx
        # spelling, and the rocdl op wants the packed-dword old operand as v2i16.
        res = llvm.call_intrinsic(
            v2i16,
            "llvm.amdgcn.cvt.scalef32.pk.fp8.f32",
            [
                llvm.bitcast(v2i16, old.ir_value()),
                F32(a).ir_value(),
                F32(b).ir_value(),
                F32(1.0).ir_value(),
                fx.Boolean(hi).ir_value(),
            ],
            [],
            [],
        )
        return I32(llvm.bitcast(T.i32, res))

    q = [v * rescale for v in yb]
    return [
        cvt(
            cvt(I32(0), q[2 * i], q[2 * i + 1], False),
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
    bits = (a / F32(448.0)).bitcast(I32)
    up = (bits + 0x7FFFFF) >> 23
    ex = (bits <= 0x00400000).select(
        I32(0),
        (up < 1).select(I32(1), (up > 254).select(I32(254), up)),
    )
    return ex, ((254 - ex) << 23).bitcast(F32)


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
    stage_x1: bool | None = None,
    spin: int = DIST_SPIN,
):
    """``@flyc.jit`` launcher of one (split count, waves, variant) configuration.

    ``stage_x1`` defaults to on where a wave owns more than ``STAGE_CHUNKS`` chunks;
    ``spin`` is the bounded wait of a split in polls.
    """
    bm = BLOCK_M
    stage = CHUNKS // ks // waves > STAGE_CHUNKS if stage_x1 is None else stage_x1
    assert CHUNKS % ks == 0 and (CHUNKS // ks) % waves == 0 and ks <= MAX_KS
    assert lds_bytes(waves) <= LDS_BYTES, "LDS"
    assert sinkhorn_repeat >= 1 and spin >= 0
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
            (f"_stage{int(stage)}", stage_x1 is not None),
            (f"_s{spin}", spin != DIST_SPIN),
        )
        if c
    )

    @fx.struct
    class Shared:
        red: fx.Array[F32, waves * bm * PSLOT]
        sq: fx.Array[F32, waves * 2 * 64]
        fin: fx.Array[F32, bm * PSLOT]
        rst: fx.Array[F32, bm]
        flag: fx.Array[I32, 4]

    @flyc.kernel(name=name, known_block_size=[nthr, 1, 1])
    def kernel(
        i32_m: fx.Int32,
        i32_rms_eps: fx.Int32,
        i32_pre_eps: fx.Int32,
        i32_sk_eps: fx.Int32,
        i32_post_mult: fx.Int32,
        i32_norm_eps: fx.Int32,
        arg_res: fx.Pointer,  # [m, 4, H] bf16
        arg_fn: fx.Pointer,  # packed B operands
        arg_y: fx.Pointer,  # [m, H] bf16
        arg_pm: fx.Pointer,  # [m, 4] f32
        arg_cm: fx.Pointer,  # [m, 4, 4] f32
        arg_pre: fx.Pointer,  # [m, 4] f32
        arg_scale: fx.Pointer,  # [3] f32
        arg_base: fx.Pointer,  # [24] f32
        arg_nw: fx.Pointer,  # [H] bf16
        arg_rout: fx.Pointer,  # [m, 4, H] bf16
        arg_post: fx.Pointer,  # [m, 4] f32
        arg_comb: fx.Pointer,  # [m, 16] f32
        arg_npre: fx.Pointer,  # [m, 4] f32
        arg_out: fx.Pointer,  # [m, H] bf16
        arg_codes: fx.Pointer,  # [m, H] e4m3
        arg_qs: fx.Pointer,  # [m, H / 32] u8
        arg_mean: fx.Pointer,  # [m, H] bf16
        arg_part: fx.Pointer,  # [blocks, bm, ks, 32] f32
        arg_ctrl: fx.Pointer,  # [blocks, ci] i32
    ):
        tx = fx.thread_idx.x
        bid = fx.block_idx.x
        q8 = bid // 8
        blk = (q8 // ks) * 8 + bid % 8
        split = q8 % ks
        nblk = (i32_m + bm - 1) // bm
        if blk < nblk:
            lane = tx % 64
            # wave index in an SGPR: per-wave offsets stay scalar (no fx spelling)
            wave = I32(rocdl.readfirstlane(T.i32, tx // 64))
            row = lane % 16
            g = lane // 16
            rowt = blk * bm + row
            smem = fx.SharedAllocator().allocate(Shared).peek()
            red, sq, fin = smem.red.ptr, smem.sq.ptr, smem.fin.ptr
            rst, flag = smem.rst.ptr, smem.flag.ptr
            m64 = fx.Int64(i32_m)

            # Buffer views with exact sizes: rows >= m read 0 and drop their stores,
            # and an index at the end of a view is a masked-off store.
            def v16(ptr, nbytes):  # 16 B units
                return ptr_buf_tensor(ptr, I32, unit_elems=4, num_records_bytes=nbytes)

            def vf(ptr, nbytes):  # fp32 elements
                return ptr_buf_tensor(ptr, F32, num_records_bytes=nbytes)

            def ld16(t, u, cm=0):
                return buf_copy_load(t, u, I32, 4, cache_modifier=cm)

            def st16(t, u, v):
                buf_copy_store(t, u, v, I32, 4)

            res_t = v16(arg_res, m64 * (N_STREAMS * H * 2))
            fn_t = v16(arg_fn, N_STREAMS * CHUNKS * 3 * 1024)
            nw_t = v16(arg_nw, H * 2)
            out_t = v16(arg_out, m64 * (H * 2))
            if fx.const_expr(has_post):
                y_t = v16(arg_y, m64 * (H * 2))
                rout_t = v16(arg_rout, m64 * (N_STREAMS * H * 2))
                pm_t = ptr_buf_tensor(
                    arg_pm, F32, unit_elems=4, num_records_bytes=m64 * 16
                )
                cm_t = ptr_buf_tensor(
                    arg_cm, F32, unit_elems=4, num_records_bytes=m64 * 64
                )
            if fx.const_expr(quant):
                # one 8 B unit of codes per 16 B norm unit
                cd_t = ptr_buf_tensor(
                    arg_codes, I32, unit_elems=2, num_records_bytes=m64 * H
                )
                qs_t = ptr_buf_tensor(
                    arg_qs, fx.Int8, num_records_bytes=m64 * (H // 32)
                )
            if fx.const_expr(mean):
                mn_t = v16(arg_mean, m64 * (H * 2))

            if fx.const_expr(has_post):
                pv = buf_copy_load(pm_t, rowt, F32, 4)
                cv = [
                    buf_copy_load(cm_t, rowt * 4 + h, F32, 4)
                    for h in range_constexpr(4)
                ]
            if fx.const_expr(not ident_pre):
                pre_t = ptr_buf_tensor(
                    arg_pre, F32, unit_elems=4, num_records_bytes=m64 * 16
                )
                qv = buf_copy_load(pre_t, rowt, F32, 4)

            cbase = split * chs + wave * cpw
            # Norm weights of the own columns: loaded up front, off the critical path
            # after the hand-off, unless x1 is staged (registers are short there).
            wts = []
            if fx.const_expr(not stage):
                wts = [ld16(nw_t, (cbase + i) * 4 + g) for i in range_constexpr(cpw)]

            def out_unit(cg):
                return rowt * U_ROW + cg * 4 + g

            acc12 = Vec.filled(4, 0.0, F32)
            acc3 = Vec.filled(4, 0.0, F32)
            sqr = F32(0.0)
            sqx = F32(0.0)
            x1s = []
            for i in range_constexpr(cpw):
                cg = cbase + i
                cu = cg * 4 + g
                bops = [
                    [
                        ld16(fn_t, ((s * CHUNKS + cg) * 3 + o) * 64 + lane)
                        for o in range_constexpr(3)
                    ]
                    for s in range_constexpr(N_STREAMS)
                ]
                rs = [
                    ld16(res_t, rowt * (N_STREAMS * U_ROW) + s * U_ROW + cu)
                    for s in range_constexpr(N_STREAMS)
                ]
                if fx.const_expr(has_post):
                    yv = _unpack8(ld16(y_t, rowt * U_ROW + cu))
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
                        st16(rout_t, rowt * (N_STREAMS * U_ROW) + j * U_ROW + cu, pk[j])
                else:
                    pk = rs
                gf = [_unpack8(p) for p in pk]
                for s in range_constexpr(N_STREAMS):
                    for e in range_constexpr(8):
                        sqr = fx.math.fma(gf[s][e], gf[s][e], sqr)
                if fx.const_expr(mean):
                    mv = [
                        ((gf[0][e] + gf[1][e]) + gf[2][e] + gf[3][e]) * F32(0.25)
                        for e in range_constexpr(8)
                    ]
                    st16(mn_t, rowt * U_ROW + cu, _pack8(mv))
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
                if fx.const_expr(stage):
                    st16(out_t, out_unit(cg), x1p)
                else:
                    x1s.append(x1p)
                x1f = _unpack8(x1p)
                for e in range_constexpr(8):
                    sqx = fx.math.fma(x1f[e], x1f[e], sqx)
                for s in range_constexpr(N_STREAMS):
                    acc12 = _mfma(pk[s], bops[s][0], acc12)
                    acc12 = _mfma(pk[s], bops[s][1], acc12)
                    acc3 = _mfma(pk[s], bops[s][2], acc3)

            for e in range_constexpr(4):
                red[(wave * bm + 4 * g + e) * PSLOT + row] = acc12[e]
                red[(wave * bm + 4 * g + e) * PSLOT + 16 + row] = acc3[e]
            sq[(wave * 2) * 64 + lane] = sqr
            sq[(wave * 2 + 1) * 64 + lane] = sqx
            if fx.const_expr(stage):
                # the staged x1 is re-read by this lane after the norm is known
                # (rocdl.s_waitcnt: fx has no vmcnt wait)
                rocdl.s_waitcnt(vmcnt=0)
            gpu.barrier()

            def mix_value(o):
                """Output o = row * 32 + slot of the split's partial row
                (slots 0-23 mixes, 24-25 square sums)."""
                ro, slot = o // PSLOT, o % PSLOT
                comb = (slot >= 16) & (slot < 24)
                lo = (slot < 24).select(slot, I32(0))
                hi = comb.select(slot + 8, I32(0))
                v = F32(0.0)
                s2 = F32(0.0)
                s3 = F32(0.0)
                for w in range_constexpr(waves):
                    a = red[(w * bm + ro) * PSLOT + lo]
                    b = red[(w * bm + ro) * PSLOT + hi]
                    v = v + a + comb.select(b, F32(0.0))
                    for gg in range_constexpr(4):
                        s2 = s2 + sq[(w * 2) * 64 + gg * 16 + ro]
                        s3 = s3 + sq[(w * 2 + 1) * 64 + gg * 16 + ro]
                return (slot == 24).select(
                    s2,
                    (slot == 25).select(s3, (slot < 24).select(v, F32(0.0))),
                )

            part_t = vf(arg_part, fx.Int64(nblk) * (bm * ks * PSLOT * 4))
            ctrl_t = vf(arg_ctrl, fx.Int64(nblk) * (ci * 4))
            ctrl_ptr_ty = fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4)

            def word_ptr(slot):
                # the hand-off atomics need a flat global pointer, not a buffer view
                return fx.inttoptr(
                    ctrl_ptr_ty,
                    buf_base_i64(arg_ctrl) + fx.Int64(blk) * (ci * 4) + slot * 4,
                )

            def load_word(p):
                return fx.generic_load(
                    p,
                    dtype=I32,
                    memory_order=fx.AtomicOrdering.Monotonic,
                    syncscope=rocdl.SyncScope.Agent,
                )

            def fence(ordering):
                fx.llvm.memory_fence(
                    syncscope=rocdl.SyncScope.Workgroup, ordering=ordering
                )

            def finish_gates():
                """Gates of the block's rows from ``fin``; wave rg handles rows
                4 * rg .. 4 * rg + 3, 16 lanes (one per comb entry) per row."""
                scale_t = vf(arg_scale, 12)
                s0 = buf_copy_load(scale_t, 0, F32, 1)
                s1 = buf_copy_load(scale_t, 1, F32, 1)
                s2 = buf_copy_load(scale_t, 2, F32, 1)
                base_t = vf(arg_base, N_MIX * 4)
                comb_t = vf(arg_comb, m64 * 64)
                npre_t = vf(arg_npre, m64 * 16)
                post_t = vf(arg_post, m64 * 16)
                rms_eps = i32_rms_eps.bitcast(F32)
                sk_eps = i32_sk_eps.bitcast(F32)
                pre_eps = i32_pre_eps.bitcast(F32)
                post_mult = i32_post_mult.bitcast(F32)
                for p in range_constexpr(-(-(bm // 4) // waves)):
                    rg = wave + waves * p
                    if rg < bm // 4:
                        lr = rg * 4 + lane // 16
                        e = lane % 16
                        rgl = blk * bm + lr
                        rs = _rsq(
                            fin[lr * PSLOT + 24] * F32(1.0 / (N_STREAMS * H)) + rms_eps
                        )
                        a = fin[lr * PSLOT + 8 + e] * rs * s2 + buf_copy_load(
                            base_t, 8 + e, F32, 1
                        )
                        mx = fx.maxnumf(a, _dpp_f32(a, DPP_QUAD_SWAP1))
                        mx = fx.maxnumf(mx, _dpp_f32(mx, DPP_QUAD_SWAP2))
                        ex = _exp2((a - mx) * F32(LOG2E))
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
                        buf_copy_store(comb_t, rgl * 16 + e, pm, F32, 1)
                        ep = e % 8
                        v = fin[lr * PSLOT + ep] * rs * (e < 4).select(
                            s0, s1
                        ) + buf_copy_load(base_t, ep, F32, 1)
                        sg = _rcp(F32(1.0) + _exp2(v * F32(-LOG2E)))
                        # lanes e >= 8 (and e >= 4 for pre) store past the view's end
                        oob = i32_m * 4
                        buf_copy_store(
                            npre_t,
                            (e < 4).select(rgl * 4 + e, oob),
                            sg + pre_eps,
                            F32,
                            1,
                        )
                        buf_copy_store(
                            post_t,
                            ((e >= 4) & (e < 8)).select(rgl * 4 + e - 4, oob),
                            sg * post_mult,
                            F32,
                            1,
                        )

            def write_out(cg, x1p, rstd, w4=None):
                """Normalized output (and MXFP8) of chunk cg from its packed x1."""
                x1f = _unpack8(x1p)
                wf = _unpack8(ld16(nw_t, cg * 4 + g) if w4 is None else w4)
                yp = _pack8([(x1f[e] * rstd) * wf[e] for e in range_constexpr(8)])
                st16(out_t, out_unit(cg), yp)
                if fx.const_expr(quant):
                    yb = _unpack8(yp)
                    ex, rescale = _e8m0(yb)
                    cd = _codes(yb, rescale)
                    buf_copy_store(
                        cd_t, out_unit(cg), Vec.from_elements(cd, I32), I32, 2
                    )
                    buf_copy_store(
                        qs_t,
                        (g == 0).select(rowt * CHUNKS + cg, i32_m * CHUNKS),
                        fx.Int8(ex),
                        fx.Int8,
                        1,
                    )

            def normalize(rstd):
                """Own columns, x1 from registers."""
                for i in range_constexpr(cpw):
                    write_out(cbase + i, x1s[i], rstd, wts[i])

            def normalize_staged(cb, rstd):
                """Columns cb .. cb + cpw of this wave's slot, x1 staged in out."""
                for i in range_constexpr(cpw):
                    x1p = ld16(out_t, out_unit(cb + i), CM_L1_BYPASS)
                    write_out(cb + i, x1p, rstd)

            def stage_own():
                for i in range_constexpr(cpw):
                    st16(out_t, out_unit(cbase + i), x1s[i])

            def sum_partials():
                for it in range_constexpr(-(-bm * PSLOT // nthr)):
                    o = it * nthr + tx
                    if o < bm * PSLOT:
                        base_idx = ((blk * bm + o // PSLOT) * ks) * PSLOT + o % PSLOT
                        v = F32(0.0)
                        for k in range_constexpr(ks):
                            v = v + buf_copy_load(
                                part_t,
                                base_idx + k * PSLOT,
                                F32,
                                1,
                                cache_modifier=CM_L1_BYPASS,
                            )
                        _put(fin, o, v)

            nrm_eps_v = i32_norm_eps.bitcast(F32)

            def rstd_of(slot_row):
                return _rsq(fin[slot_row * PSLOT + 25] * F32(1.0 / H) + nrm_eps_v)

            if fx.const_expr(ks == 1):
                for it in range_constexpr(-(-bm * PSLOT // nthr)):
                    o = it * nthr + tx
                    if o < bm * PSLOT:
                        _put(fin, o, mix_value(o))
                gpu.barrier()
                if fx.const_expr(stage):
                    normalize_staged(cbase, rstd_of(row))
                else:
                    normalize(rstd_of(row))
                finish_gates()
            else:
                for it in range_constexpr(-(-bm * PSLOT // nthr)):
                    o = it * nthr + tx
                    if o < bm * PSLOT:
                        buf_copy_store(
                            part_t,
                            ((blk * bm + o // PSLOT) * ks + split) * PSLOT + o % PSLOT,
                            mix_value(o),
                            F32,
                            1,
                        )
                # partials (and a staged x1) complete before the arrival is published
                rocdl.s_waitcnt(vmcnt=0)
                gpu.barrier()
                if tx == 0:
                    fence(fx.AtomicOrdering.Release)
                    old = fx.llvm.atomic_add(
                        word_ptr(0),
                        I32(1),
                        syncscope=rocdl.SyncScope.Agent,
                        ordering=fx.AtomicOrdering.Monotonic,
                    )
                    _put(flag, 0, old)
                gpu.barrier()
                old_w = flag[0]
                gen = (old_w >> GEN_SHIFT) & 15

                def gen_of(word):
                    return (word >> GEN_SHIFT) & 15

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
                            buf_copy_store(
                                ctrl_t, blk * ci + CTRL_RSTD + r_i, rv, F32, 1
                            )
                    rocdl.s_waitcnt(vmcnt=0)
                    gpu.barrier()
                    # publish from the last wave (no gate rows there when waves > 4):
                    # its return value, the hand-off bits, is needed only at the end
                    pub = nthr - 64
                    snap = I32(0)
                    if tx == pub:
                        fence(fx.AtomicOrdering.Release)
                        snap = fx.llvm.atomic_add(
                            word_ptr(0),
                            I32((1 << GEN_SHIFT) - ks),
                            syncscope=rocdl.SyncScope.Agent,
                            ordering=fx.AtomicOrdering.Monotonic,
                        )
                    if fx.const_expr(stage):
                        normalize_staged(cbase, rst[row])
                    else:
                        normalize(rst[row])
                    finish_gates()
                    if tx == pub:
                        _put(flag, 2, (snap >> 8) & ((1 << ks) - 1))
                    gpu.barrier()
                    handed = flag[2]
                    if handed != 0:  # rare: splits that gave up waiting
                        if tx == 0:
                            # clear the bits for the next generation
                            fx.llvm.atomic_add(
                                word_ptr(0),
                                I32(0) - (handed << 8),
                                syncscope=rocdl.SyncScope.Agent,
                                ordering=fx.AtomicOrdering.Monotonic,
                            )
                        fence(fx.AtomicOrdering.Acquire)
                        for k_h in range(I32(0), I32(ks), I32(1)):
                            kk = I32(k_h)
                            if ((handed >> kk) & 1) != 0:
                                normalize_staged(kk * chs + wave * cpw, rst[row])
                else:
                    # wait (bounded) for the publish, then rescale own x1
                    if tx == 0:
                        sp = word_ptr(0)
                        cur = load_word(sp)
                        n_spin = I32(0)
                        while (gen_of(cur) == gen) & (n_spin < spin):
                            # s_sleep: back off between L2 polls (no fx spelling)
                            rocdl.s_sleep(1)
                            cur = load_word(sp)
                            n_spin = n_spin + 1
                        _put(flag, 1, (gen_of(cur) == gen).select(I32(1), I32(0)))
                    gpu.barrier()
                    if flag[1] != 0:
                        # gave up waiting: stage the unscaled x1, then set the hand-off
                        # bit unless the publish came first
                        if fx.const_expr(not stage):
                            stage_own()
                        rocdl.s_waitcnt(vmcnt=0)
                        gpu.barrier()
                        if tx == 0:
                            fence(fx.AtomicOrdering.Release)
                            sp = word_ptr(0)
                            m_cur = load_word(sp)
                            state = I32(0)
                            while state == 0:
                                if gen_of(m_cur) != gen:
                                    state = I32(2)  # published: not handed off
                                else:
                                    m_old, m_ok = fx.llvm.atomic_cas(
                                        sp,
                                        m_cur,
                                        m_cur | (I32(1) << (split + 8)),
                                        syncscope=rocdl.SyncScope.Agent,
                                    )
                                    if m_ok:
                                        state = I32(3)  # handed off
                                    m_cur = m_old
                            if state == 3:
                                fx.llvm.atomic_add(
                                    word_ptr(CTRL_TMO),
                                    I32(1),
                                    syncscope=rocdl.SyncScope.Agent,
                                    ordering=fx.AtomicOrdering.Monotonic,
                                )
                            _put(flag, 1, state)
                        gpu.barrier()
                    if flag[1] != 3:
                        fence(fx.AtomicOrdering.Acquire)
                        rv = buf_copy_load(
                            ctrl_t,
                            blk * ci + CTRL_RSTD + row,
                            F32,
                            1,
                            cache_modifier=CM_L1_BYPASS,
                        )
                        if fx.const_expr(stage):
                            normalize_staged(cbase, rv)
                        else:
                            normalize(rv)

    @flyc.jit
    def launch(
        arg_res: fx.Pointer,
        arg_fn: fx.Pointer,
        arg_y: fx.Pointer,
        arg_pm: fx.Pointer,
        arg_cm: fx.Pointer,
        arg_pre: fx.Pointer,
        arg_scale: fx.Pointer,
        arg_base: fx.Pointer,
        arg_nw: fx.Pointer,
        arg_rout: fx.Pointer,
        arg_post: fx.Pointer,
        arg_comb: fx.Pointer,
        arg_npre: fx.Pointer,
        arg_out: fx.Pointer,
        arg_codes: fx.Pointer,
        arg_qs: fx.Pointer,
        arg_mean: fx.Pointer,
        arg_part: fx.Pointer,
        arg_ctrl: fx.Pointer,
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
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        }
    }
    return launch

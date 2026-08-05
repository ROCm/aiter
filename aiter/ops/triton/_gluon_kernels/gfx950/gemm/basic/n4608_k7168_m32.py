# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 OpenAI

"""Gluon specialization for the 32x4608x7168 block scaled decode GEMM.

Most of the helper is shared with the small-M decode kernels, but this entry
point deliberately specializes the serving shape.  Having the strides and
the split count as constexprs was important both for the CDNA buffer address
instructions and for making the FP32 association exactly match SGLang.
"""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon as Gluon
from triton.experimental.gluon import language as gl

M = 32
N = 4608
K = 7168
BLOCK_M = 32
BLOCK_N = 64
K_SPLITS = 7


@Gluon.jit
def _g_reduce(
    part_ptr,
    out_ptr,
    stride_pk,
    stride_pm,
    stride_pn,
    BLOCKM: gl.constexpr,
    BLOCKN: gl.constexpr,
    M: gl.constexpr,
    N: gl.constexpr,
    ACT: gl.constexpr,
    MAXK: gl.constexpr,
    layout: gl.constexpr,
) -> None:
    pidm = gl.program_id(0)
    pidn = gl.program_id(1)
    im = gl.arange(0, BLOCKM, layout=gl.SliceLayout(1, layout))
    jn = gl.arange(0, BLOCKN, layout=gl.SliceLayout(0, layout))
    rm = pidm * BLOCKM + im
    rn = pidn * BLOCKN + jn
    # Offsets and values kept in the two dimensional reduction layout.  Doing
    # the reduction as a short straight line loop also avoids allocating a
    # third (ksplit) dimension in a program.
    # Output remains the public row-major tensor.  The scratch used by the
    # 4608 layer is private, and writing it in [ntile, split, 32, 64] order
    # makes a sizeable difference on CDNA4.  In the planar [split, m, n]
    # view a reducer wave has seven in-flight requests going to seven remote
    # L2 slices.  Keeping the seven pages beside their producing 64-column
    # tile instead lets the small reducer stay on one slice.  There is an
    # eighth padding page per tile; besides making the tile stride a shift it
    # prevents two neighbours aliasing the same L2 set.  It is never read or
    # written.  Other specializations still use the ordinary planar layout.
    offs = rm[:, None] * N + rn[None, :]
    if ACT == 7:
        poffs = (
            ((rn[None, :] // 64) * 8) * (M * 64) + rm[:, None] * 64 + (rn[None, :] % 64)
        )
        PSTEP: gl.constexpr = M * 64
    else:
        poffs = offs
        PSTEP: gl.constexpr = M * N
    # Triton reduction is a power-of-two tree.  Write it explicitly (the
    # split counts are a small fixed set) so the fadd association is bit
    # identical to the serving kernel.  Loading adjacent pages with a joined
    # layout gives buffer_load a dwordx opportunity; split + convert is free
    # on gfx95 and keeps all the adds in the small two-dimensional layout.
    v01 = gl.amd.cdna4.buffer_load(part_ptr, gl.join(poffs, poffs + PSTEP), cache=".cg")
    v0, v1 = gl.split(v01)
    v0 = gl.convert_layout(v0, layout)
    v1 = gl.convert_layout(v1, layout)
    a01 = v0 + v1
    if ACT == 4:
        v23 = gl.amd.cdna4.buffer_load(
            part_ptr, gl.join(poffs + PSTEP * 2, poffs + PSTEP * 3), cache=".cg"
        )
        v2, v3 = gl.split(v23)
        v2 = gl.convert_layout(v2, layout)
        v3 = gl.convert_layout(v3, layout)
        a23 = v2 + v3
        val = a01 + a23
    elif ACT == 7:
        v23 = gl.amd.cdna4.buffer_load(
            part_ptr, gl.join(poffs + PSTEP * 2, poffs + PSTEP * 3), cache=".cg"
        )
        v2, v3 = gl.split(v23)
        v2 = gl.convert_layout(v2, layout)
        v3 = gl.convert_layout(v3, layout)
        a23 = v2 + v3
        first = a01 + a23
        v45 = gl.amd.cdna4.buffer_load(
            part_ptr, gl.join(poffs + PSTEP * 4, poffs + PSTEP * 5), cache=".cg"
        )
        v4, v5 = gl.split(v45)
        v4 = gl.convert_layout(v4, layout)
        v5 = gl.convert_layout(v5, layout)
        a45 = v4 + v5
        v6 = gl.amd.cdna4.buffer_load(part_ptr, poffs + PSTEP * 6, cache=".cg")
        tail = a45 + v6
        val = first + tail
    else:
        v23 = gl.amd.cdna4.buffer_load(
            part_ptr, gl.join(poffs + PSTEP * 2, poffs + PSTEP * 3), cache=".cg"
        )
        v2, v3 = gl.split(v23)
        v2 = gl.convert_layout(v2, layout)
        v3 = gl.convert_layout(v3, layout)
        a23 = v2 + v3
        first4 = a01 + a23
        v45 = gl.amd.cdna4.buffer_load(
            part_ptr, gl.join(poffs + PSTEP * 4, poffs + PSTEP * 5), cache=".cg"
        )
        v4, v5 = gl.split(v45)
        v4 = gl.convert_layout(v4, layout)
        v5 = gl.convert_layout(v5, layout)
        a45 = v4 + v5
        v67 = gl.amd.cdna4.buffer_load(
            part_ptr, gl.join(poffs + PSTEP * 6, poffs + PSTEP * 7), cache=".cg"
        )
        v6, v7 = gl.split(v67)
        v6 = gl.convert_layout(v6, layout)
        v7 = gl.convert_layout(v7, layout)
        a67 = v6 + v7
        first8 = first4 + (a45 + a67)
        v89 = gl.amd.cdna4.buffer_load(
            part_ptr, gl.join(poffs + PSTEP * 8, poffs + PSTEP * 9), cache=".cg"
        )
        v8, v9 = gl.split(v89)
        v8 = gl.convert_layout(v8, layout)
        v9 = gl.convert_layout(v9, layout)
        a89 = v8 + v9
        v1011 = gl.amd.cdna4.buffer_load(
            part_ptr, gl.join(poffs + PSTEP * 10, poffs + PSTEP * 11), cache=".cg"
        )
        v10, v11 = gl.split(v1011)
        v10 = gl.convert_layout(v10, layout)
        v11 = gl.convert_layout(v11, layout)
        a1011 = v10 + v11
        second4 = a89 + a1011
        v1213 = gl.amd.cdna4.buffer_load(
            part_ptr, gl.join(poffs + PSTEP * 12, poffs + PSTEP * 13), cache=".cg"
        )
        v12, v13 = gl.split(v1213)
        v12 = gl.convert_layout(v12, layout)
        v13 = gl.convert_layout(v13, layout)
        a1213 = v12 + v13
        val = first8 + (second4 + a1213)
    val = val.to(gl.bfloat16)
    # The BF16 result is consumed by the caller after this graph; keeping its
    # one-shot stores out of the vector cache leaves room for the following
    # graph's first partial stripe.
    gl.amd.cdna4.buffer_store(val, out_ptr, offs, cache=".wt")


@Gluon.jit
def _g_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    as_ptr,
    bs_ptr,
    # The dimensions are constexpr.  The input tensors used by the
    # server are tightly packed, specializing these here saves quite a
    # few address multiplies in the inner loop.
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    KBLKS: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    KS: gl.constexpr,
    ITERS: gl.constexpr,
    PARTIAL: gl.constexpr,
    mfma: gl.constexpr,
    alayout: gl.constexpr,
    blayout: gl.constexpr,
) -> None:
    pidm = gl.program_id(0)
    pidn0 = gl.program_id(1)
    pidk = gl.program_id(2)
    # CDNA exposes eight XCDs.  Remapping consecutive broadcast tiles to a
    # column-major strip avoids the small underfull wave at the end of the N
    # row for unsplit projections.
    if KS == 1:
        nt = N // BN
        tall = nt % 8
        # the production rows are multiples of the tile width, but not all
        # divide the eight schedulers evenly.
        pp = (nt + 7) // 8
        xcd = pidn0 % 8
        lp = pidn0 // 8
        if tall == 0:
            pidn = xcd * pp + lp
        else:
            if xcd < tall:
                pidn = xcd * pp + lp
            else:
                pidn = tall * pp + (xcd - tall) * (pp - 1) + lp
    else:
        pidn = (pidn0 % 12) * (N // BN // 12) + pidn0 // 12

    # Operand layouts used by the native fp8 mfma.
    ad: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=mfma, k_width=16)
    bd: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=mfma, k_width=16)
    if BM == 16:
        sha: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, order=[1, 0])
    else:
        sha: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, order=[1, 0])
    if BN == 16:
        shb: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, order=[0, 1])
    else:
        shb: gl.constexpr = gl.SwizzledSharedLayout(16, 2, 8, order=[0, 1])
    sma = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    smb = gl.allocate_shared_memory(gl.float8e4nv, [128, BN], shb)
    # ping-pong buffer: the second half can be populated while the first mfma is pending
    sma2 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    smb2 = gl.allocate_shared_memory(gl.float8e4nv, [128, BN], shb)
    sma3 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    smb3 = gl.allocate_shared_memory(gl.float8e4nv, [128, BN], shb)
    sma4 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    smb4 = gl.allocate_shared_memory(gl.float8e4nv, [128, BN], shb)
    sma5 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    smb5 = gl.allocate_shared_memory(gl.float8e4nv, [128, BN], shb)
    sma6 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    smb6 = gl.allocate_shared_memory(gl.float8e4nv, [128, BN], shb)
    sma7 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    smb7 = gl.allocate_shared_memory(gl.float8e4nv, [128, BN], shb)
    sma8 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    smb8 = gl.allocate_shared_memory(gl.float8e4nv, [128, BN], shb)
    # scratch pages used to materialize split accumulators for the short
    # unsplit tree
    shout: gl.constexpr = gl.SwizzledSharedLayout(4, 2, 8, order=[1, 0])
    st0 = gl.allocate_shared_memory(gl.float32, [BM, BN], shout)

    # Global-memory (blocked) views. Keeping A and B in different blocked
    # layouts is important on CDNA: A is row-major while the view of the
    # weight is column-major.
    lak = gl.arange(0, 128, layout=gl.SliceLayout(0, alayout))
    lbk = gl.arange(0, 128, layout=gl.SliceLayout(1, blayout))

    # Scale and accumulator tensors naturally live in the MFMA accumulator
    # layout; the slice layouts let a broadcast become free.
    sm = gl.arange(0, BM, layout=gl.SliceLayout(1, mfma)) + pidm * BM
    sn = gl.arange(0, BN, layout=gl.SliceLayout(0, mfma)) + pidn * BN
    wm = gl.arange(0, BM, layout=gl.SliceLayout(1, alayout)) + pidm * BM
    wn = gl.arange(0, BN, layout=gl.SliceLayout(0, blayout)) + pidn * BN

    accum = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
    zacc = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
    base_blk = pidk * ITERS
    if KS == 1:
        if ITERS == 16:
            for big in range(ITERS // 8):
                blk = base_blk + big * 8
                oa0 = wm[:, None] * K + lak[None, :] + blk * 128
                ob0 = wn[None, :] * K + lbk[:, None] + blk * 128
                sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
                av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
                bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
                bk1 = blk + 1
                oa1 = wm[:, None] * K + lak[None, :] + bk1 * 128
                ob1 = wn[None, :] * K + lbk[:, None] + bk1 * 128
                sb1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk1)
                sk1 = sm * KBLKS + blk
                sv1 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk1, sk1 + 1))
                sa0, sa1 = gl.split(sv1)
                sa0 = gl.convert_layout(sa0, gl.SliceLayout(1, mfma))
                sa1 = gl.convert_layout(sa1, gl.SliceLayout(1, mfma))
                av1 = gl.amd.cdna4.buffer_load(a_ptr, oa1)
                bv1 = gl.amd.cdna4.buffer_load(b_ptr, ob1)
                bk2 = blk + 2
                oa2 = wm[:, None] * K + lak[None, :] + bk2 * 128
                ob2 = wn[None, :] * K + lbk[:, None] + bk2 * 128
                sb2 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk2)
                av2 = gl.amd.cdna4.buffer_load(a_ptr, oa2)
                bv2 = gl.amd.cdna4.buffer_load(b_ptr, ob2)
                bk3 = blk + 3
                oa3 = wm[:, None] * K + lak[None, :] + bk3 * 128
                ob3 = wn[None, :] * K + lbk[:, None] + bk3 * 128
                sb3 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk3)
                sk3 = sm * KBLKS + bk2
                sv3 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk3, sk3 + 1))
                sa2, sa3 = gl.split(sv3)
                sa2 = gl.convert_layout(sa2, gl.SliceLayout(1, mfma))
                sa3 = gl.convert_layout(sa3, gl.SliceLayout(1, mfma))
                av3 = gl.amd.cdna4.buffer_load(a_ptr, oa3)
                bv3 = gl.amd.cdna4.buffer_load(b_ptr, ob3)
                bk4 = blk + 4
                oa4 = wm[:, None] * K + lak[None, :] + bk4 * 128
                ob4 = wn[None, :] * K + lbk[:, None] + bk4 * 128
                sb4 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk4)
                av4 = gl.amd.cdna4.buffer_load(a_ptr, oa4)
                bv4 = gl.amd.cdna4.buffer_load(b_ptr, ob4)
                bk5 = blk + 5
                oa5 = wm[:, None] * K + lak[None, :] + bk5 * 128
                ob5 = wn[None, :] * K + lbk[:, None] + bk5 * 128
                sb5 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk5)
                sk5 = sm * KBLKS + bk4
                sv5 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk5, sk5 + 1))
                sa4, sa5 = gl.split(sv5)
                sa4 = gl.convert_layout(sa4, gl.SliceLayout(1, mfma))
                sa5 = gl.convert_layout(sa5, gl.SliceLayout(1, mfma))
                av5 = gl.amd.cdna4.buffer_load(a_ptr, oa5)
                bv5 = gl.amd.cdna4.buffer_load(b_ptr, ob5)
                bk6 = blk + 6
                oa6 = wm[:, None] * K + lak[None, :] + bk6 * 128
                ob6 = wn[None, :] * K + lbk[:, None] + bk6 * 128
                sb6 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk6)
                av6 = gl.amd.cdna4.buffer_load(a_ptr, oa6)
                bv6 = gl.amd.cdna4.buffer_load(b_ptr, ob6)
                bk7 = blk + 7
                oa7 = wm[:, None] * K + lak[None, :] + bk7 * 128
                ob7 = wn[None, :] * K + lbk[:, None] + bk7 * 128
                sb7 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk7)
                sk7 = sm * KBLKS + bk6
                sv7 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk7, sk7 + 1))
                sa6, sa7 = gl.split(sv7)
                sa6 = gl.convert_layout(sa6, gl.SliceLayout(1, mfma))
                sa7 = gl.convert_layout(sa7, gl.SliceLayout(1, mfma))
                av7 = gl.amd.cdna4.buffer_load(a_ptr, oa7)
                bv7 = gl.amd.cdna4.buffer_load(b_ptr, ob7)
                sma.store(av0)
                smb.store(bv0)
                sma2.store(av1)
                smb2.store(bv1)
                sma3.store(av2)
                smb3.store(bv2)
                sma4.store(av3)
                smb4.store(bv3)
                sma5.store(av4)
                smb5.store(bv4)
                sma6.store(av5)
                smb6.store(bv5)
                sma7.store(av6)
                smb7.store(bv6)
                sma8.store(av7)
                smb8.store(bv7)
                gl.barrier()
                x0 = sma.load(ad)
                y0 = smb.load(bd)
                q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
                q0 = q0 * sa0[:, None] * sb0
                accum = accum + q0
                x1 = sma2.load(ad)
                y1 = smb2.load(bd)
                q1 = gl.amd.cdna4.mfma_scaled(x1, None, "e4m3", y1, None, "e4m3", zacc)
                q1 = q1 * sa1[:, None] * sb1
                accum = accum + q1
                x2 = sma3.load(ad)
                y2 = smb3.load(bd)
                q2 = gl.amd.cdna4.mfma_scaled(x2, None, "e4m3", y2, None, "e4m3", zacc)
                q2 = q2 * sa2[:, None] * sb2
                accum = accum + q2
                x3 = sma4.load(ad)
                y3 = smb4.load(bd)
                q3 = gl.amd.cdna4.mfma_scaled(x3, None, "e4m3", y3, None, "e4m3", zacc)
                q3 = q3 * sa3[:, None] * sb3
                accum = accum + q3
                x4 = sma5.load(ad)
                y4 = smb5.load(bd)
                q4 = gl.amd.cdna4.mfma_scaled(x4, None, "e4m3", y4, None, "e4m3", zacc)
                q4 = q4 * sa4[:, None] * sb4
                accum = accum + q4
                x5 = sma6.load(ad)
                y5 = smb6.load(bd)
                q5 = gl.amd.cdna4.mfma_scaled(x5, None, "e4m3", y5, None, "e4m3", zacc)
                q5 = q5 * sa5[:, None] * sb5
                accum = accum + q5
                x6 = sma7.load(ad)
                y6 = smb7.load(bd)
                q6 = gl.amd.cdna4.mfma_scaled(x6, None, "e4m3", y6, None, "e4m3", zacc)
                q6 = q6 * sa6[:, None] * sb6
                accum = accum + q6
                x7 = sma8.load(ad)
                y7 = smb8.load(bd)
                q7 = gl.amd.cdna4.mfma_scaled(x7, None, "e4m3", y7, None, "e4m3", zacc)
                q7 = q7 * sa7[:, None] * sb7
                accum = accum + q7
        # For the 1536 projection a split-K launch spends almost as much time
        # writing the four 3-block partials as it spends doing the MFMA.  A
        # single CTA can still keep enough N tiles in flight (BN=16 below).
        # These twelve blocks are staged six at a time; g0..g3 intentionally
        # mirror the four reference split accumulators.  st0 is a materialize
        # page -- without that LDS round trip LLVM reassociates the last add
        # with the MFMA scale FMA and a few bf16 ties round the other way.
        elif ITERS == 12:
            g0 = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
            g1 = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
            oa_0 = wm[:, None] * K + lak[None, :] + 0
            ob_0 = wn[None, :] * K + lbk[:, None] + 0
            sb_0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 0)
            av_0 = gl.amd.cdna4.buffer_load(a_ptr, oa_0)
            bv_0 = gl.amd.cdna4.buffer_load(b_ptr, ob_0)
            oa_1 = wm[:, None] * K + lak[None, :] + 128
            ob_1 = wn[None, :] * K + lbk[:, None] + 128
            sb_1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 1)
            sk_0 = sm * KBLKS + 0
            sv_0 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk_0, sk_0 + 1))
            sa_0, sa_1 = gl.split(sv_0)
            sa_0 = gl.convert_layout(sa_0, gl.SliceLayout(1, mfma))
            sa_1 = gl.convert_layout(sa_1, gl.SliceLayout(1, mfma))
            av_1 = gl.amd.cdna4.buffer_load(a_ptr, oa_1)
            bv_1 = gl.amd.cdna4.buffer_load(b_ptr, ob_1)
            oa_2 = wm[:, None] * K + lak[None, :] + 256
            ob_2 = wn[None, :] * K + lbk[:, None] + 256
            sb_2 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 2)
            av_2 = gl.amd.cdna4.buffer_load(a_ptr, oa_2)
            bv_2 = gl.amd.cdna4.buffer_load(b_ptr, ob_2)
            oa_3 = wm[:, None] * K + lak[None, :] + 384
            ob_3 = wn[None, :] * K + lbk[:, None] + 384
            sb_3 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 3)
            sk_2 = sm * KBLKS + 2
            sv_2 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk_2, sk_2 + 1))
            sa_2, sa_3 = gl.split(sv_2)
            sa_2 = gl.convert_layout(sa_2, gl.SliceLayout(1, mfma))
            sa_3 = gl.convert_layout(sa_3, gl.SliceLayout(1, mfma))
            av_3 = gl.amd.cdna4.buffer_load(a_ptr, oa_3)
            bv_3 = gl.amd.cdna4.buffer_load(b_ptr, ob_3)
            oa_4 = wm[:, None] * K + lak[None, :] + 512
            ob_4 = wn[None, :] * K + lbk[:, None] + 512
            sb_4 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 4)
            av_4 = gl.amd.cdna4.buffer_load(a_ptr, oa_4)
            bv_4 = gl.amd.cdna4.buffer_load(b_ptr, ob_4)
            oa_5 = wm[:, None] * K + lak[None, :] + 640
            ob_5 = wn[None, :] * K + lbk[:, None] + 640
            sb_5 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 5)
            sk_4 = sm * KBLKS + 4
            sv_4 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk_4, sk_4 + 1))
            sa_4, sa_5 = gl.split(sv_4)
            sa_4 = gl.convert_layout(sa_4, gl.SliceLayout(1, mfma))
            sa_5 = gl.convert_layout(sa_5, gl.SliceLayout(1, mfma))
            av_5 = gl.amd.cdna4.buffer_load(a_ptr, oa_5)
            bv_5 = gl.amd.cdna4.buffer_load(b_ptr, ob_5)
            sma.store(av_0)
            smb.store(bv_0)
            sma2.store(av_1)
            smb2.store(bv_1)
            sma3.store(av_2)
            smb3.store(bv_2)
            sma4.store(av_3)
            smb4.store(bv_3)
            sma5.store(av_4)
            smb5.store(bv_4)
            sma6.store(av_5)
            smb6.store(bv_5)
            gl.barrier()
            x_0 = sma.load(ad)
            y_0 = smb.load(bd)
            q_0 = gl.amd.cdna4.mfma_scaled(x_0, None, "e4m3", y_0, None, "e4m3", zacc)
            q_0 = q_0 * sa_0[:, None] * sb_0
            g0 = g0 + q_0
            x_1 = sma2.load(ad)
            y_1 = smb2.load(bd)
            q_1 = gl.amd.cdna4.mfma_scaled(x_1, None, "e4m3", y_1, None, "e4m3", zacc)
            q_1 = q_1 * sa_1[:, None] * sb_1
            g0 = g0 + q_1
            x_2 = sma3.load(ad)
            y_2 = smb3.load(bd)
            q_2 = gl.amd.cdna4.mfma_scaled(x_2, None, "e4m3", y_2, None, "e4m3", zacc)
            q_2 = q_2 * sa_2[:, None] * sb_2
            g0 = g0 + q_2
            x_3 = sma4.load(ad)
            y_3 = smb4.load(bd)
            q_3 = gl.amd.cdna4.mfma_scaled(x_3, None, "e4m3", y_3, None, "e4m3", zacc)
            q_3 = q_3 * sa_3[:, None] * sb_3
            g1 = g1 + q_3
            x_4 = sma5.load(ad)
            y_4 = smb5.load(bd)
            q_4 = gl.amd.cdna4.mfma_scaled(x_4, None, "e4m3", y_4, None, "e4m3", zacc)
            q_4 = q_4 * sa_4[:, None] * sb_4
            g1 = g1 + q_4
            x_5 = sma6.load(ad)
            y_5 = smb6.load(bd)
            q_5 = gl.amd.cdna4.mfma_scaled(x_5, None, "e4m3", y_5, None, "e4m3", zacc)
            q_5 = q_5 * sa_5[:, None] * sb_5
            g1 = g1 + q_5
            ltmp = g0 + g1
            st0.store(ltmp)
            lpair = st0.load(mfma)
            g2 = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
            g3 = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
            oa_6 = wm[:, None] * K + lak[None, :] + 768
            ob_6 = wn[None, :] * K + lbk[:, None] + 768
            sb_6 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 6)
            av_6 = gl.amd.cdna4.buffer_load(a_ptr, oa_6)
            bv_6 = gl.amd.cdna4.buffer_load(b_ptr, ob_6)
            oa_7 = wm[:, None] * K + lak[None, :] + 896
            ob_7 = wn[None, :] * K + lbk[:, None] + 896
            sb_7 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 7)
            sk_6 = sm * KBLKS + 6
            sv_6 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk_6, sk_6 + 1))
            sa_6, sa_7 = gl.split(sv_6)
            sa_6 = gl.convert_layout(sa_6, gl.SliceLayout(1, mfma))
            sa_7 = gl.convert_layout(sa_7, gl.SliceLayout(1, mfma))
            av_7 = gl.amd.cdna4.buffer_load(a_ptr, oa_7)
            bv_7 = gl.amd.cdna4.buffer_load(b_ptr, ob_7)
            oa_8 = wm[:, None] * K + lak[None, :] + 1024
            ob_8 = wn[None, :] * K + lbk[:, None] + 1024
            sb_8 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 8)
            av_8 = gl.amd.cdna4.buffer_load(a_ptr, oa_8)
            bv_8 = gl.amd.cdna4.buffer_load(b_ptr, ob_8)
            oa_9 = wm[:, None] * K + lak[None, :] + 1152
            ob_9 = wn[None, :] * K + lbk[:, None] + 1152
            sb_9 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 9)
            sk_8 = sm * KBLKS + 8
            sv_8 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk_8, sk_8 + 1))
            sa_8, sa_9 = gl.split(sv_8)
            sa_8 = gl.convert_layout(sa_8, gl.SliceLayout(1, mfma))
            sa_9 = gl.convert_layout(sa_9, gl.SliceLayout(1, mfma))
            av_9 = gl.amd.cdna4.buffer_load(a_ptr, oa_9)
            bv_9 = gl.amd.cdna4.buffer_load(b_ptr, ob_9)
            oa_10 = wm[:, None] * K + lak[None, :] + 1280
            ob_10 = wn[None, :] * K + lbk[:, None] + 1280
            sb_10 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 10)
            av_10 = gl.amd.cdna4.buffer_load(a_ptr, oa_10)
            bv_10 = gl.amd.cdna4.buffer_load(b_ptr, ob_10)
            oa_11 = wm[:, None] * K + lak[None, :] + 1408
            ob_11 = wn[None, :] * K + lbk[:, None] + 1408
            sb_11 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 11)
            sk_10 = sm * KBLKS + 10
            sv_10 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk_10, sk_10 + 1))
            sa_10, sa_11 = gl.split(sv_10)
            sa_10 = gl.convert_layout(sa_10, gl.SliceLayout(1, mfma))
            sa_11 = gl.convert_layout(sa_11, gl.SliceLayout(1, mfma))
            av_11 = gl.amd.cdna4.buffer_load(a_ptr, oa_11)
            bv_11 = gl.amd.cdna4.buffer_load(b_ptr, ob_11)
            sma.store(av_6)
            smb.store(bv_6)
            sma2.store(av_7)
            smb2.store(bv_7)
            sma3.store(av_8)
            smb3.store(bv_8)
            sma4.store(av_9)
            smb4.store(bv_9)
            sma5.store(av_10)
            smb5.store(bv_10)
            sma6.store(av_11)
            smb6.store(bv_11)
            gl.barrier()
            x_6 = sma.load(ad)
            y_6 = smb.load(bd)
            q_6 = gl.amd.cdna4.mfma_scaled(x_6, None, "e4m3", y_6, None, "e4m3", zacc)
            q_6 = q_6 * sa_6[:, None] * sb_6
            g2 = g2 + q_6
            x_7 = sma2.load(ad)
            y_7 = smb2.load(bd)
            q_7 = gl.amd.cdna4.mfma_scaled(x_7, None, "e4m3", y_7, None, "e4m3", zacc)
            q_7 = q_7 * sa_7[:, None] * sb_7
            g2 = g2 + q_7
            x_8 = sma3.load(ad)
            y_8 = smb3.load(bd)
            q_8 = gl.amd.cdna4.mfma_scaled(x_8, None, "e4m3", y_8, None, "e4m3", zacc)
            q_8 = q_8 * sa_8[:, None] * sb_8
            g2 = g2 + q_8
            x_9 = sma4.load(ad)
            y_9 = smb4.load(bd)
            q_9 = gl.amd.cdna4.mfma_scaled(x_9, None, "e4m3", y_9, None, "e4m3", zacc)
            q_9 = q_9 * sa_9[:, None] * sb_9
            g3 = g3 + q_9
            x_10 = sma5.load(ad)
            y_10 = smb5.load(bd)
            q_10 = gl.amd.cdna4.mfma_scaled(
                x_10, None, "e4m3", y_10, None, "e4m3", zacc
            )
            q_10 = q_10 * sa_10[:, None] * sb_10
            g3 = g3 + q_10
            x_11 = sma6.load(ad)
            y_11 = smb6.load(bd)
            q_11 = gl.amd.cdna4.mfma_scaled(
                x_11, None, "e4m3", y_11, None, "e4m3", zacc
            )
            q_11 = q_11 * sa_11[:, None] * sb_11
            g3 = g3 + q_11
            rtmp = g2 + g3
            st0.store(rtmp)
            rpair = st0.load(mfma)
            accum = lpair + rpair
        else:
            for un in range(ITERS // 6):
                blk = base_blk + un * 6
                oa0 = wm[:, None] * K + lak[None, :] + blk * 128
                ob0 = wn[None, :] * K + lbk[:, None] + blk * 128
                sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
                av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
                bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
                bk1 = blk + 1
                oa1 = wm[:, None] * K + lak[None, :] + bk1 * 128
                ob1 = wn[None, :] * K + lbk[:, None] + bk1 * 128
                sb1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk1)
                sou0 = sm * KBLKS + blk
                svu0 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sou0, sou0 + 1))
                sa0, sa1 = gl.split(svu0)
                sa0 = gl.convert_layout(sa0, gl.SliceLayout(1, mfma))
                sa1 = gl.convert_layout(sa1, gl.SliceLayout(1, mfma))
                av1 = gl.amd.cdna4.buffer_load(a_ptr, oa1)
                bv1 = gl.amd.cdna4.buffer_load(b_ptr, ob1)
                bk2 = blk + 2
                oa2 = wm[:, None] * K + lak[None, :] + bk2 * 128
                ob2 = wn[None, :] * K + lbk[:, None] + bk2 * 128
                sb2 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk2)
                av2 = gl.amd.cdna4.buffer_load(a_ptr, oa2)
                bv2 = gl.amd.cdna4.buffer_load(b_ptr, ob2)
                bk3 = blk + 3
                oa3 = wm[:, None] * K + lak[None, :] + bk3 * 128
                ob3 = wn[None, :] * K + lbk[:, None] + bk3 * 128
                sb3 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk3)
                sou1 = sm * KBLKS + bk2
                svu1 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sou1, sou1 + 1))
                sa2, sa3 = gl.split(svu1)
                sa2 = gl.convert_layout(sa2, gl.SliceLayout(1, mfma))
                sa3 = gl.convert_layout(sa3, gl.SliceLayout(1, mfma))
                av3 = gl.amd.cdna4.buffer_load(a_ptr, oa3)
                bv3 = gl.amd.cdna4.buffer_load(b_ptr, ob3)
                bk4 = blk + 4
                oa4 = wm[:, None] * K + lak[None, :] + bk4 * 128
                ob4 = wn[None, :] * K + lbk[:, None] + bk4 * 128
                sb4 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk4)
                av4 = gl.amd.cdna4.buffer_load(a_ptr, oa4)
                bv4 = gl.amd.cdna4.buffer_load(b_ptr, ob4)
                bk5 = blk + 5
                oa5 = wm[:, None] * K + lak[None, :] + bk5 * 128
                ob5 = wn[None, :] * K + lbk[:, None] + bk5 * 128
                sb5 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk5)
                sou2 = sm * KBLKS + bk4
                svu2 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sou2, sou2 + 1))
                sa4, sa5 = gl.split(svu2)
                sa4 = gl.convert_layout(sa4, gl.SliceLayout(1, mfma))
                sa5 = gl.convert_layout(sa5, gl.SliceLayout(1, mfma))
                av5 = gl.amd.cdna4.buffer_load(a_ptr, oa5)
                bv5 = gl.amd.cdna4.buffer_load(b_ptr, ob5)
                sma.store(av0)
                smb.store(bv0)
                sma2.store(av1)
                smb2.store(bv1)
                sma3.store(av2)
                smb3.store(bv2)
                sma4.store(av3)
                smb4.store(bv3)
                sma5.store(av4)
                smb5.store(bv4)
                sma6.store(av5)
                smb6.store(bv5)
                gl.barrier()
                x0 = sma.load(ad)
                y0 = smb.load(bd)
                q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
                q0 = q0 * sa0[:, None] * sb0
                accum = accum + q0
                x1 = sma2.load(ad)
                y1 = smb2.load(bd)
                q1 = gl.amd.cdna4.mfma_scaled(x1, None, "e4m3", y1, None, "e4m3", zacc)
                q1 = q1 * sa1[:, None] * sb1
                accum = accum + q1
                x2 = sma3.load(ad)
                y2 = smb3.load(bd)
                q2 = gl.amd.cdna4.mfma_scaled(x2, None, "e4m3", y2, None, "e4m3", zacc)
                q2 = q2 * sa2[:, None] * sb2
                accum = accum + q2
                x3 = sma4.load(ad)
                y3 = smb4.load(bd)
                q3 = gl.amd.cdna4.mfma_scaled(x3, None, "e4m3", y3, None, "e4m3", zacc)
                q3 = q3 * sa3[:, None] * sb3
                accum = accum + q3
                x4 = sma5.load(ad)
                y4 = smb5.load(bd)
                q4 = gl.amd.cdna4.mfma_scaled(x4, None, "e4m3", y4, None, "e4m3", zacc)
                q4 = q4 * sa4[:, None] * sb4
                accum = accum + q4
                x5 = sma6.load(ad)
                y5 = smb6.load(bd)
                q5 = gl.amd.cdna4.mfma_scaled(x5, None, "e4m3", y5, None, "e4m3", zacc)
                q5 = q5 * sa5[:, None] * sb5
                accum = accum + q5
            if ITERS % 6 >= 1:
                blk = base_blk + (ITERS // 6) * 6 + 0
                oa0 = wm[:, None] * K + lak[None, :] + blk * 128
                ob0 = wn[None, :] * K + lbk[:, None] + blk * 128
                sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
                sa0 = gl.amd.cdna4.buffer_load(as_ptr, sm * KBLKS + blk)
                av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
                bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
                sma.store(av0)
                smb.store(bv0)
                gl.barrier()
                x0 = sma.load(ad)
                y0 = smb.load(bd)
                q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
                q0 = q0 * sa0[:, None] * sb0
                accum = accum + q0
            if ITERS % 6 >= 2:
                blk = base_blk + (ITERS // 6) * 6 + 1
                oa0 = wm[:, None] * K + lak[None, :] + blk * 128
                ob0 = wn[None, :] * K + lbk[:, None] + blk * 128
                sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
                sa0 = gl.amd.cdna4.buffer_load(as_ptr, sm * KBLKS + blk)
                av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
                bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
                sma2.store(av0)
                smb2.store(bv0)
                gl.barrier()
                x0 = sma2.load(ad)
                y0 = smb2.load(bd)
                q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
                q0 = q0 * sa0[:, None] * sb0
                accum = accum + q0
            if ITERS % 6 >= 3:
                blk = base_blk + (ITERS // 6) * 6 + 2
                oa0 = wm[:, None] * K + lak[None, :] + blk * 128
                ob0 = wn[None, :] * K + lbk[:, None] + blk * 128
                sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
                sa0 = gl.amd.cdna4.buffer_load(as_ptr, sm * KBLKS + blk)
                av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
                bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
                sma2.store(av0)
                smb2.store(bv0)
                gl.barrier()
                x0 = sma2.load(ad)
                y0 = smb2.load(bd)
                q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
                q0 = q0 * sa0[:, None] * sb0
                accum = accum + q0
            if ITERS % 6 >= 4:
                blk = base_blk + (ITERS // 6) * 6 + 3
                oa0 = wm[:, None] * K + lak[None, :] + blk * 128
                ob0 = wn[None, :] * K + lbk[:, None] + blk * 128
                sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
                sa0 = gl.amd.cdna4.buffer_load(as_ptr, sm * KBLKS + blk)
                av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
                bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
                sma2.store(av0)
                smb2.store(bv0)
                gl.barrier()
                x0 = sma2.load(ad)
                y0 = smb2.load(bd)
                q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
                q0 = q0 * sa0[:, None] * sb0
                accum = accum + q0
            if ITERS % 6 >= 5:
                blk = base_blk + (ITERS // 6) * 6 + 4
                oa0 = wm[:, None] * K + lak[None, :] + blk * 128
                ob0 = wn[None, :] * K + lbk[:, None] + blk * 128
                sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
                sa0 = gl.amd.cdna4.buffer_load(as_ptr, sm * KBLKS + blk)
                av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
                bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
                sma2.store(av0)
                smb2.store(bv0)
                gl.barrier()
                x0 = sma2.load(ad)
                y0 = smb2.load(bd)
                q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
                q0 = q0 * sa0[:, None] * sb0
                accum = accum + q0
    else:
        # Stage an entire short split at once on the 2112 projection.  Its
        # chunks are eight consecutive 128-wide blocks, just small enough
        # to fit in the ping-pong LDS pages.  Having a single barrier for the
        # group instead of the four pair barriers keeps the issue slots busy
        # while preserving the exact running accumulator for the chunk.
        if ITERS == 8 and N == 2112:
            for big in range(ITERS // 8):
                blk = base_blk + big * 8
                oa0 = wm[:, None] * K + lak[None, :] + blk * 128
                ob0 = wn[None, :] * K + lbk[:, None] + blk * 128
                sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
                av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
                bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
                bk1 = blk + 1
                oa1 = wm[:, None] * K + lak[None, :] + bk1 * 128
                ob1 = wn[None, :] * K + lbk[:, None] + bk1 * 128
                sb1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk1)
                sk1 = sm * KBLKS + blk
                spair1 = gl.join(sk1, sk1 + 1)
                svx1 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(spair1, spair1 + 2))
                sv1, sv3 = gl.split(svx1)
                sa0, sa1 = gl.split(sv1)
                sa2, sa3 = gl.split(sv3)
                sa0 = gl.convert_layout(sa0, gl.SliceLayout(1, mfma))
                sa1 = gl.convert_layout(sa1, gl.SliceLayout(1, mfma))
                sa2 = gl.convert_layout(sa2, gl.SliceLayout(1, mfma))
                sa3 = gl.convert_layout(sa3, gl.SliceLayout(1, mfma))
                av1 = gl.amd.cdna4.buffer_load(a_ptr, oa1)
                bv1 = gl.amd.cdna4.buffer_load(b_ptr, ob1)
                bk2 = blk + 2
                oa2 = wm[:, None] * K + lak[None, :] + bk2 * 128
                ob2 = wn[None, :] * K + lbk[:, None] + bk2 * 128
                sb2 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk2)
                av2 = gl.amd.cdna4.buffer_load(a_ptr, oa2)
                bv2 = gl.amd.cdna4.buffer_load(b_ptr, ob2)
                bk3 = blk + 3
                oa3 = wm[:, None] * K + lak[None, :] + bk3 * 128
                ob3 = wn[None, :] * K + lbk[:, None] + bk3 * 128
                sb3 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk3)
                # scales 2 and 3 were fetched with the preceding pair
                av3 = gl.amd.cdna4.buffer_load(a_ptr, oa3)
                bv3 = gl.amd.cdna4.buffer_load(b_ptr, ob3)
                bk4 = blk + 4
                oa4 = wm[:, None] * K + lak[None, :] + bk4 * 128
                ob4 = wn[None, :] * K + lbk[:, None] + bk4 * 128
                sb4 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk4)
                av4 = gl.amd.cdna4.buffer_load(a_ptr, oa4)
                bv4 = gl.amd.cdna4.buffer_load(b_ptr, ob4)
                bk5 = blk + 5
                oa5 = wm[:, None] * K + lak[None, :] + bk5 * 128
                ob5 = wn[None, :] * K + lbk[:, None] + bk5 * 128
                sb5 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk5)
                sk5 = sm * KBLKS + bk4
                spair5 = gl.join(sk5, sk5 + 1)
                svx5 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(spair5, spair5 + 2))
                sv5, sv7 = gl.split(svx5)
                sa4, sa5 = gl.split(sv5)
                sa6, sa7 = gl.split(sv7)
                sa4 = gl.convert_layout(sa4, gl.SliceLayout(1, mfma))
                sa5 = gl.convert_layout(sa5, gl.SliceLayout(1, mfma))
                sa6 = gl.convert_layout(sa6, gl.SliceLayout(1, mfma))
                sa7 = gl.convert_layout(sa7, gl.SliceLayout(1, mfma))
                av5 = gl.amd.cdna4.buffer_load(a_ptr, oa5)
                bv5 = gl.amd.cdna4.buffer_load(b_ptr, ob5)
                bk6 = blk + 6
                oa6 = wm[:, None] * K + lak[None, :] + bk6 * 128
                ob6 = wn[None, :] * K + lbk[:, None] + bk6 * 128
                sb6 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk6)
                av6 = gl.amd.cdna4.buffer_load(a_ptr, oa6)
                bv6 = gl.amd.cdna4.buffer_load(b_ptr, ob6)
                bk7 = blk + 7
                oa7 = wm[:, None] * K + lak[None, :] + bk7 * 128
                ob7 = wn[None, :] * K + lbk[:, None] + bk7 * 128
                sb7 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk7)
                # scales 6 and 7 fetched above
                av7 = gl.amd.cdna4.buffer_load(a_ptr, oa7)
                bv7 = gl.amd.cdna4.buffer_load(b_ptr, ob7)
                sma.store(av0)
                smb.store(bv0)
                sma2.store(av1)
                smb2.store(bv1)
                sma3.store(av2)
                smb3.store(bv2)
                sma4.store(av3)
                smb4.store(bv3)
                sma5.store(av4)
                smb5.store(bv4)
                sma6.store(av5)
                smb6.store(bv5)
                sma7.store(av6)
                smb7.store(bv6)
                sma8.store(av7)
                smb8.store(bv7)
                gl.barrier()
                x0 = sma.load(ad)
                y0 = smb.load(bd)
                q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
                q0 = q0 * sa0[:, None] * sb0
                accum = accum + q0
                x1 = sma2.load(ad)
                y1 = smb2.load(bd)
                q1 = gl.amd.cdna4.mfma_scaled(x1, None, "e4m3", y1, None, "e4m3", zacc)
                q1 = q1 * sa1[:, None] * sb1
                accum = accum + q1
                x2 = sma3.load(ad)
                y2 = smb3.load(bd)
                q2 = gl.amd.cdna4.mfma_scaled(x2, None, "e4m3", y2, None, "e4m3", zacc)
                q2 = q2 * sa2[:, None] * sb2
                accum = accum + q2
                x3 = sma4.load(ad)
                y3 = smb4.load(bd)
                q3 = gl.amd.cdna4.mfma_scaled(x3, None, "e4m3", y3, None, "e4m3", zacc)
                q3 = q3 * sa3[:, None] * sb3
                accum = accum + q3
                x4 = sma5.load(ad)
                y4 = smb5.load(bd)
                q4 = gl.amd.cdna4.mfma_scaled(x4, None, "e4m3", y4, None, "e4m3", zacc)
                q4 = q4 * sa4[:, None] * sb4
                accum = accum + q4
                x5 = sma6.load(ad)
                y5 = smb6.load(bd)
                q5 = gl.amd.cdna4.mfma_scaled(x5, None, "e4m3", y5, None, "e4m3", zacc)
                q5 = q5 * sa5[:, None] * sb5
                accum = accum + q5
                x6 = sma7.load(ad)
                y6 = smb7.load(bd)
                q6 = gl.amd.cdna4.mfma_scaled(x6, None, "e4m3", y6, None, "e4m3", zacc)
                q6 = q6 * sa6[:, None] * sb6
                accum = accum + q6
                x7 = sma8.load(ad)
                y7 = smb8.load(bd)
                q7 = gl.amd.cdna4.mfma_scaled(x7, None, "e4m3", y7, None, "e4m3", zacc)
                q7 = q7 * sa7[:, None] * sb7
                accum = accum + q7
        elif ITERS == 3:
            # the single short projection chunk can fit all three LDS pages
            # under one barrier.
            blk = base_blk
            oa0 = wm[:, None] * K + lak[None, :] + blk * 128
            ob0 = wn[None, :] * K + lbk[:, None] + blk * 128
            sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
            bk1 = blk + 1
            oa1 = wm[:, None] * K + lak[None, :] + bk1 * 128
            ob1 = wn[None, :] * K + lbk[:, None] + bk1 * 128
            sb1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk1)
            bk2 = blk + 2
            oa2 = wm[:, None] * K + lak[None, :] + bk2 * 128
            ob2 = wn[None, :] * K + lbk[:, None] + bk2 * 128
            sb2 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk2)
            so0 = sm * KBLKS + blk
            so01 = gl.join(so0, so0 + 1)
            svbig2 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(so01, so01 + 2))
            slo, shi = gl.split(svbig2)
            sa0, sa1 = gl.split(slo)
            sa2, _sa_unused = gl.split(shi)
            sa0 = gl.convert_layout(sa0, gl.SliceLayout(1, mfma))
            sa1 = gl.convert_layout(sa1, gl.SliceLayout(1, mfma))
            sa2 = gl.convert_layout(sa2, gl.SliceLayout(1, mfma))
            av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
            bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
            av1 = gl.amd.cdna4.buffer_load(a_ptr, oa1)
            bv1 = gl.amd.cdna4.buffer_load(b_ptr, ob1)
            av2 = gl.amd.cdna4.buffer_load(a_ptr, oa2)
            bv2 = gl.amd.cdna4.buffer_load(b_ptr, ob2)
            sma.store(av0)
            smb.store(bv0)
            sma2.store(av1)
            smb2.store(bv1)
            sma3.store(av2)
            smb3.store(bv2)
            gl.barrier()
            x0 = sma.load(ad)
            y0 = smb.load(bd)
            q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
            q0 = q0 * sa0[:, None] * sb0
            accum = accum + q0
            x1 = sma2.load(ad)
            y1 = smb2.load(bd)
            q1 = gl.amd.cdna4.mfma_scaled(x1, None, "e4m3", y1, None, "e4m3", zacc)
            q1 = q1 * sa1[:, None] * sb1
            accum = accum + q1
            x2 = sma3.load(ad)
            y2 = smb3.load(bd)
            q2 = gl.amd.cdna4.mfma_scaled(x2, None, "e4m3", y2, None, "e4m3", zacc)
            q2 = q2 * sa2[:, None] * sb2
            accum = accum + q2
        # 14 way split leaves many tiny CTAs on this wide layer.
        # Adjacent chunks are folded in one wave; ACT==7 in the reducer is the
        # same binary tree as ACT==14 with pairs pre-added.
        elif ITERS == 8 and N == 4608:
            ga = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
            bp0 = pidk * 8 + 0
            oa0 = wm[:, None] * K + lak[None, :] + bp0 * 128
            ob0 = wn[None, :] * K + lbk[:, None] + bp0 * 128
            sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bp0)
            bp1 = bp0 + 1
            oa1 = wm[:, None] * K + lak[None, :] + bp1 * 128
            ob1 = wn[None, :] * K + lbk[:, None] + bp1 * 128
            sb1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bp1)
            sk0 = sm * KBLKS + bp0
            sv0 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk0, sk0 + 1))
            sa0, sa1 = gl.split(sv0)
            sa0 = gl.convert_layout(sa0, gl.SliceLayout(1, mfma))
            sa1 = gl.convert_layout(sa1, gl.SliceLayout(1, mfma))
            av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
            bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
            av1 = gl.amd.cdna4.buffer_load(a_ptr, oa1)
            bv1 = gl.amd.cdna4.buffer_load(b_ptr, ob1)
            sma.store(av0)
            smb.store(bv0)
            sma2.store(av1)
            smb2.store(bv1)
            gl.barrier()
            x0 = sma.load(ad)
            y0 = smb.load(bd)
            q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
            q0 = q0 * sa0[:, None] * sb0
            ga = ga + q0
            x1 = sma2.load(ad)
            y1 = smb2.load(bd)
            q1 = gl.amd.cdna4.mfma_scaled(x1, None, "e4m3", y1, None, "e4m3", zacc)
            q1 = q1 * sa1[:, None] * sb1
            ga = ga + q1
            bp0 = pidk * 8 + 2
            oa0 = wm[:, None] * K + lak[None, :] + bp0 * 128
            ob0 = wn[None, :] * K + lbk[:, None] + bp0 * 128
            sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bp0)
            bp1 = bp0 + 1
            oa1 = wm[:, None] * K + lak[None, :] + bp1 * 128
            ob1 = wn[None, :] * K + lbk[:, None] + bp1 * 128
            sb1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bp1)
            sk0 = sm * KBLKS + bp0
            sv0 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk0, sk0 + 1))
            sa0, sa1 = gl.split(sv0)
            sa0 = gl.convert_layout(sa0, gl.SliceLayout(1, mfma))
            sa1 = gl.convert_layout(sa1, gl.SliceLayout(1, mfma))
            av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
            bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
            av1 = gl.amd.cdna4.buffer_load(a_ptr, oa1)
            bv1 = gl.amd.cdna4.buffer_load(b_ptr, ob1)
            sma.store(av0)
            smb.store(bv0)
            sma2.store(av1)
            smb2.store(bv1)
            gl.barrier()
            x0 = sma.load(ad)
            y0 = smb.load(bd)
            q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
            q0 = q0 * sa0[:, None] * sb0
            ga = ga + q0
            x1 = sma2.load(ad)
            y1 = smb2.load(bd)
            q1 = gl.amd.cdna4.mfma_scaled(x1, None, "e4m3", y1, None, "e4m3", zacc)
            q1 = q1 * sa1[:, None] * sb1
            ga = ga + q1
            gb = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
            bp0 = pidk * 8 + 4
            oa0 = wm[:, None] * K + lak[None, :] + bp0 * 128
            ob0 = wn[None, :] * K + lbk[:, None] + bp0 * 128
            sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bp0)
            bp1 = bp0 + 1
            oa1 = wm[:, None] * K + lak[None, :] + bp1 * 128
            ob1 = wn[None, :] * K + lbk[:, None] + bp1 * 128
            sb1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bp1)
            sk0 = sm * KBLKS + bp0
            sv0 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk0, sk0 + 1))
            sa0, sa1 = gl.split(sv0)
            sa0 = gl.convert_layout(sa0, gl.SliceLayout(1, mfma))
            sa1 = gl.convert_layout(sa1, gl.SliceLayout(1, mfma))
            av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
            bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
            av1 = gl.amd.cdna4.buffer_load(a_ptr, oa1)
            bv1 = gl.amd.cdna4.buffer_load(b_ptr, ob1)
            sma.store(av0)
            smb.store(bv0)
            sma2.store(av1)
            smb2.store(bv1)
            gl.barrier()
            x0 = sma.load(ad)
            y0 = smb.load(bd)
            q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
            q0 = q0 * sa0[:, None] * sb0
            gb = gb + q0
            x1 = sma2.load(ad)
            y1 = smb2.load(bd)
            q1 = gl.amd.cdna4.mfma_scaled(x1, None, "e4m3", y1, None, "e4m3", zacc)
            q1 = q1 * sa1[:, None] * sb1
            gb = gb + q1
            bp0 = pidk * 8 + 6
            oa0 = wm[:, None] * K + lak[None, :] + bp0 * 128
            ob0 = wn[None, :] * K + lbk[:, None] + bp0 * 128
            sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bp0)
            bp1 = bp0 + 1
            oa1 = wm[:, None] * K + lak[None, :] + bp1 * 128
            ob1 = wn[None, :] * K + lbk[:, None] + bp1 * 128
            sb1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bp1)
            sk0 = sm * KBLKS + bp0
            sv0 = gl.amd.cdna4.buffer_load(as_ptr, gl.join(sk0, sk0 + 1))
            sa0, sa1 = gl.split(sv0)
            sa0 = gl.convert_layout(sa0, gl.SliceLayout(1, mfma))
            sa1 = gl.convert_layout(sa1, gl.SliceLayout(1, mfma))
            av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
            bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
            av1 = gl.amd.cdna4.buffer_load(a_ptr, oa1)
            bv1 = gl.amd.cdna4.buffer_load(b_ptr, ob1)
            sma.store(av0)
            smb.store(bv0)
            sma2.store(av1)
            smb2.store(bv1)
            gl.barrier()
            x0 = sma.load(ad)
            y0 = smb.load(bd)
            q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
            q0 = q0 * sa0[:, None] * sb0
            gb = gb + q0
            x1 = sma2.load(ad)
            y1 = smb2.load(bd)
            q1 = gl.amd.cdna4.mfma_scaled(x1, None, "e4m3", y1, None, "e4m3", zacc)
            q1 = q1 * sa1[:, None] * sb1
            gb = gb + q1
            accum = ga + gb
        else:
            # split-K chunks still have several adjacent blocks; pairing their LDS
            # pages makes the row scale a naturally vector load as well.
            for pairk in range(ITERS // 2):
                blk = base_blk + pairk * 2
                oa0 = wm[:, None] * K + lak[None, :] + blk * 128
                ob0 = wn[None, :] * K + lbk[:, None] + blk * 128
                sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
                bk1 = blk + 1
                oa1 = wm[:, None] * K + lak[None, :] + bk1 * 128
                ob1 = wn[None, :] * K + lbk[:, None] + bk1 * 128
                sb1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + bk1)
                so0 = sm * KBLKS + blk
                svp = gl.amd.cdna4.buffer_load(as_ptr, gl.join(so0, so0 + 1))
                sa0, sa1 = gl.split(svp)
                sa0 = gl.convert_layout(sa0, gl.SliceLayout(1, mfma))
                sa1 = gl.convert_layout(sa1, gl.SliceLayout(1, mfma))
                av0 = gl.amd.cdna4.buffer_load(a_ptr, oa0)
                bv0 = gl.amd.cdna4.buffer_load(b_ptr, ob0)
                av1 = gl.amd.cdna4.buffer_load(a_ptr, oa1)
                bv1 = gl.amd.cdna4.buffer_load(b_ptr, ob1)
                sma.store(av0)
                smb.store(bv0)
                sma2.store(av1)
                smb2.store(bv1)
                gl.barrier()
                x0 = sma.load(ad)
                y0 = smb.load(bd)
                q0 = gl.amd.cdna4.mfma_scaled(x0, None, "e4m3", y0, None, "e4m3", zacc)
                q0 = q0 * sa0[:, None] * sb0
                accum = accum + q0
                x1 = sma2.load(ad)
                y1 = smb2.load(bd)
                q1 = gl.amd.cdna4.mfma_scaled(x1, None, "e4m3", y1, None, "e4m3", zacc)
                q1 = q1 * sa1[:, None] * sb1
                accum = accum + q1
            if ITERS % 2:
                blk = base_blk + ITERS - 1
                oa = wm[:, None] * K + lak[None, :] + blk * 128
                ob = wn[None, :] * K + lbk[:, None] + blk * 128
                sb = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + blk)
                sa = gl.amd.cdna4.buffer_load(as_ptr, sm * KBLKS + blk)
                av = gl.amd.cdna4.buffer_load(a_ptr, oa)
                bv = gl.amd.cdna4.buffer_load(b_ptr, ob)
                sma.store(av)
                smb.store(bv)
                gl.barrier()
                av = sma.load(ad)
                bv = smb.load(bd)
                dot = gl.amd.cdna4.mfma_scaled(av, None, "e4m3", bv, None, "e4m3", zacc)
                dot = dot * sa[:, None] * sb
                accum = accum + dot
    offc = sm[:, None] * N + sn[None, :]
    if PARTIAL:
        if KS == 7 and N == 4608 and BN == 64:
            offc = (
                ((sn[None, :] // 64) * 8 + pidk) * (M * 64)
                + sm[:, None] * 64
                + (sn[None, :] % 64)
            )
        else:
            offc += pidk * (M * N)
    # when c points at the partial tensor its element type is float; otherwise
    # this is the final bf16 conversion, exactly like the reference triton.
    r = accum.to(c_ptr.dtype.element_ty)
    # Partials are consumed immediately and never read by this producer.  A
    # write-through stream leaves the seven pages clean in L2 for the small
    # reduction CTAs instead of filling each CU's vector cache with values it
    # cannot reuse.
    gl.amd.cdna4.buffer_store(r, c_ptr, offc, cache=".wt")


def block_scaled_mm_n4608_k7168_m32(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run the exact `(M, N, K) = (32, 4608, 7168)` specialization."""
    assert a.shape == (32, 7168)
    assert b.shape == (4608, 7168)
    assert a_scale.shape == (32, 56)
    assert b_scale.shape == (36, 56)
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert a_scale.dtype == b_scale.dtype == torch.float32
    assert all(tensor.is_contiguous() for tensor in (a, b, a_scale, b_scale))
    assert out.shape == (32, 4608)
    assert out.dtype == torch.bfloat16
    assert out.device == a.device and out.is_contiguous()
    # Shapes/strides are fixed by SGLang; using constexprs in the kernels is a
    # sizeable win on the very small-m decode GEMMs.
    k_blocks = K // 128
    iterations = k_blocks // K_SPLITS

    # A tile contains one 16x16 mfma per (m,n) subtile.  Start with one wave per
    # subtile; on gfx950 the launcher will still round very small blocks to the
    # native wave allocation.
    # Four neighboring MFMA tiles remain resident across the split.
    warps_m = 2
    warps_n = 2
    num_warps = warps_m * warps_n
    mfma = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[warps_m, warps_n],
    )
    al = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    bl = gl.BlockedLayout(
        # The K-major vector here matches buffer_load_dwordx4.  The block is
        # tiled twice over the 64 output columns; that was a little faster
        # than carrying two scalar columns per lane because the LDS swizzle
        # already folds the two column halves.
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )

    part = torch.empty((K_SPLITS + 1, M, N), dtype=torch.float32, device=a.device)

    # Everything is constexpr here: offsets fit in the buffer addressing
    # range, so the CDNA buffer ops below save both pointer registers and
    # scalar multiplies in each small workgroup.
    _g_gemm[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), K_SPLITS)](
        a,
        b,
        part,
        a_scale,
        b_scale,
        M=M,
        N=N,
        K=K,
        KBLKS=k_blocks,
        BM=BLOCK_M,
        BN=BLOCK_N,
        KS=K_SPLITS,
        ITERS=iterations,
        PARTIAL=True,
        mfma=mfma,
        alayout=al,
        blayout=bl,
        num_warps=num_warps,
        waves_per_eu=2,
    )
    # The single-wave consumer stays on one 8x32 stripe while the seven
    # sibling partial pages remain hot in one L2 slice.
    reducer_layout = gl.BlockedLayout(
        size_per_thread=[1, 1],
        threads_per_warp=[2, 32],
        warps_per_cta=[1, 1],
        order=[1, 0],
    )
    _g_reduce[(triton.cdiv(M, 8), triton.cdiv(N, 32))](
        part,
        out,
        0,
        0,
        0,
        BLOCKM=8,
        BLOCKN=32,
        M=M,
        N=N,
        ACT=K_SPLITS,
        MAXK=triton.next_power_of_2(K_SPLITS),
        layout=reducer_layout,
        num_warps=1,
        waves_per_eu=1,
    )
    return out


__all__ = ["block_scaled_mm_n4608_k7168_m32"]

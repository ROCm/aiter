# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 OpenAI

"""Exact M=32, N=3072, K=1536 FP8 GEMM specialization for MI355X."""

from __future__ import annotations

import torch
import triton
from triton.experimental import gluon as Gluon
from triton.experimental.gluon import language as gl

M = 32
N = 3072
K = 1536
BLOCK_M = 16
BLOCK_N = 32
K_SPLITS = 1


@Gluon.jit
def _g_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    as_ptr,
    bs_ptr,
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
    pidm = gl.program_id(1)
    pidn0 = gl.program_id(0)
    pidk = gl.program_id(2)
    pidn = pidn0

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
    # Both N waves read A, so tail blocks use private A pages.
    sma9 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    sma10 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    sma11 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    sma12 = gl.allocate_shared_memory(gl.float8e4nv, [BM, 128], sha)
    shout: gl.constexpr = gl.SwizzledSharedLayout(4, 1, 16, order=[1, 0])
    st0 = gl.allocate_shared_memory(gl.float32, [BM, BN], shout)  # noqa: F841

    lak = gl.arange(0, 128, layout=gl.SliceLayout(0, alayout))
    lbk = gl.arange(0, 128, layout=gl.SliceLayout(1, blayout))

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
        # Keep the 12-block reduction and reference accumulation order in one CTA.
        elif ITERS == 12:
            abaseoff = wm[:, None] * K + lak[None, :]
            bbaseoff = wn[None, :] * K + lbk[:, None]
            g0 = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
            g1 = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
            g2 = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
            g3 = gl.zeros([BM, BN], dtype=gl.float32, layout=mfma)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma, a_ptr + 0, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb, b_ptr + 0, bbaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma2, a_ptr + 128, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb2, b_ptr + 128, bbaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma3, a_ptr + 256, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb3, b_ptr + 256, bbaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma4, a_ptr + 384, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb4, b_ptr + 384, bbaseoff)
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma5, a_ptr + 512, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb5, b_ptr + 512, bbaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma6, a_ptr + 640, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb6, b_ptr + 640, bbaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma7, a_ptr + 768, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb7, b_ptr + 768, bbaseoff)
            gl.amd.cdna4.async_copy.commit_group()
            sox0 = sm * KBLKS
            qoff0 = gl.join(gl.join(sox0, sox0 + 1), gl.join(sox0 + 2, sox0 + 3))
            qval0 = gl.amd.cdna4.buffer_load(as_ptr, qoff0)
            qlo0, qhi0 = gl.split(qval0)
            p_sa0, p_sa1 = gl.split(qlo0)
            p_sa2, p_sa3 = gl.split(qhi0)
            p_sa0 = gl.convert_layout(p_sa0, gl.SliceLayout(1, mfma))
            p_sa1 = gl.convert_layout(p_sa1, gl.SliceLayout(1, mfma))
            p_sa2 = gl.convert_layout(p_sa2, gl.SliceLayout(1, mfma))
            p_sa3 = gl.convert_layout(p_sa3, gl.SliceLayout(1, mfma))
            qoff4 = qoff0 + 4
            qval4 = gl.amd.cdna4.buffer_load(as_ptr, qoff4)
            qlo4, qhi4 = gl.split(qval4)
            p_sa4, p_sa5 = gl.split(qlo4)
            p_sa6, p_sa7 = gl.split(qhi4)
            p_sa4 = gl.convert_layout(p_sa4, gl.SliceLayout(1, mfma))
            p_sa5 = gl.convert_layout(p_sa5, gl.SliceLayout(1, mfma))
            p_sa6 = gl.convert_layout(p_sa6, gl.SliceLayout(1, mfma))
            p_sa7 = gl.convert_layout(p_sa7, gl.SliceLayout(1, mfma))
            p_sb0 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 0)
            p_sb1 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 1)
            p_sb2 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 2)
            p_sb3 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 3)
            p_sb4 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 4)
            p_sb5 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 5)
            p_sb6 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 6)
            p_sb7 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 7)
            gl.amd.cdna4.async_copy.wait_group(1)
            # wait_group is wave-local; publish the cooperative LDS tile CTA-wide.
            gl.barrier()
            p_ax0 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma, layout=ad)
            p_bx0 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb, layout=bd)
            p_q0 = gl.amd.cdna4.mfma_scaled(
                p_ax0, None, "e4m3", p_bx0, None, "e4m3", zacc
            )
            p_q0 = gl.fma(p_q0 * p_sa0[:, None], p_sb0, g0)
            g0 = p_q0
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma8, a_ptr + 896, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb, b_ptr + 896, bbaseoff)
            p_ax1 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma2, layout=ad)
            p_bx1 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb2, layout=bd)
            p_q1 = gl.amd.cdna4.mfma_scaled(
                p_ax1, None, "e4m3", p_bx1, None, "e4m3", zacc
            )
            p_q1 = gl.fma(p_q1 * p_sa1[:, None], p_sb1, g0)
            g0 = p_q1
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma9, a_ptr + 1024, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb2, b_ptr + 1024, bbaseoff)
            p_ax2 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma3, layout=ad)
            p_bx2 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb3, layout=bd)
            p_q2 = gl.amd.cdna4.mfma_scaled(
                p_ax2, None, "e4m3", p_bx2, None, "e4m3", zacc
            )
            p_q2 = gl.fma(p_q2 * p_sa2[:, None], p_sb2, g0)
            g0 = p_q2
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma10, a_ptr + 1152, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb3, b_ptr + 1152, bbaseoff)
            p_ax3 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma4, layout=ad)
            p_bx3 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb4, layout=bd)
            p_q3 = gl.amd.cdna4.mfma_scaled(
                p_ax3, None, "e4m3", p_bx3, None, "e4m3", zacc
            )
            p_q3 = gl.fma(p_q3 * p_sa3[:, None], p_sb3, g1)
            g1 = p_q3
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma11, a_ptr + 1280, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb4, b_ptr + 1280, bbaseoff)
            gl.amd.cdna4.async_copy.wait_group(0)
            gl.barrier()
            p_ax4 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma5, layout=ad)
            p_bx4 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb5, layout=bd)
            p_q4 = gl.amd.cdna4.mfma_scaled(
                p_ax4, None, "e4m3", p_bx4, None, "e4m3", zacc
            )
            p_q4 = gl.fma(p_q4 * p_sa4[:, None], p_sb4, g1)
            g1 = p_q4
            gl.amd.cdna4.async_copy.buffer_load_to_shared(sma12, a_ptr + 1408, abaseoff)
            gl.amd.cdna4.async_copy.buffer_load_to_shared(smb5, b_ptr + 1408, bbaseoff)
            gl.amd.cdna4.async_copy.commit_group()
            qoff8 = sox0 + 8
            qval8 = gl.amd.cdna4.buffer_load(
                as_ptr,
                gl.join(gl.join(qoff8, qoff8 + 1), gl.join(qoff8 + 2, qoff8 + 3)),
            )
            qlo8, qhi8 = gl.split(qval8)
            p_sa8, p_sa9 = gl.split(qlo8)
            p_sa10, p_sa11 = gl.split(qhi8)
            p_sa8 = gl.convert_layout(p_sa8, gl.SliceLayout(1, mfma))
            p_sa9 = gl.convert_layout(p_sa9, gl.SliceLayout(1, mfma))
            p_sa10 = gl.convert_layout(p_sa10, gl.SliceLayout(1, mfma))
            p_sa11 = gl.convert_layout(p_sa11, gl.SliceLayout(1, mfma))
            p_sb8 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 8)
            p_sb9 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 9)
            p_sb10 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 10)
            p_sb11 = gl.load(bs_ptr + (pidn * BN // 128) * KBLKS + 11)
            p_ax5 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma6, layout=ad)
            p_bx5 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb6, layout=bd)
            p_q5 = gl.amd.cdna4.mfma_scaled(
                p_ax5, None, "e4m3", p_bx5, None, "e4m3", zacc
            )
            p_q5 = gl.fma(p_q5 * p_sa5[:, None], p_sb5, g1)
            g1 = p_q5
            p_ax6 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma7, layout=ad)
            p_bx6 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb7, layout=bd)
            p_q6 = gl.amd.cdna4.mfma_scaled(
                p_ax6, None, "e4m3", p_bx6, None, "e4m3", zacc
            )
            p_q6 = gl.fma(p_q6 * p_sa6[:, None], p_sb6, g2)
            g2 = p_q6
            ltmp = g0 + g1
            lpair = gl.inline_asm_elementwise(
                "", "=v,0", [ltmp], dtype=gl.float32, is_pure=False, pack=1
            )
            gl.amd.cdna4.async_copy.wait_group(0)
            gl.barrier()
            p_ax7 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma8, layout=ad)
            p_bx7 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb, layout=bd)
            p_q7 = gl.amd.cdna4.mfma_scaled(
                p_ax7, None, "e4m3", p_bx7, None, "e4m3", zacc
            )
            p_q7 = gl.fma(p_q7 * p_sa7[:, None], p_sb7, g2)
            g2 = p_q7
            p_ax8 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma9, layout=ad)
            p_bx8 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb2, layout=bd)
            p_q8 = gl.amd.cdna4.mfma_scaled(
                p_ax8, None, "e4m3", p_bx8, None, "e4m3", zacc
            )
            p_q8 = gl.fma(p_q8 * p_sa8[:, None], p_sb8, g2)
            g2 = p_q8
            p_ax9 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma10, layout=ad)
            p_bx9 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb3, layout=bd)
            p_q9 = gl.amd.cdna4.mfma_scaled(
                p_ax9, None, "e4m3", p_bx9, None, "e4m3", zacc
            )
            p_q9 = gl.fma(p_q9 * p_sa9[:, None], p_sb9, g3)
            g3 = p_q9
            p_ax10 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma11, layout=ad)
            p_bx10 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb4, layout=bd)
            p_q10 = gl.amd.cdna4.mfma_scaled(
                p_ax10, None, "e4m3", p_bx10, None, "e4m3", zacc
            )
            p_q10 = gl.fma(p_q10 * p_sa10[:, None], p_sb10, g3)
            g3 = p_q10
            p_ax11 = gl.amd.cdna4.async_copy.load_shared_relaxed(sma12, layout=ad)
            p_bx11 = gl.amd.cdna4.async_copy.load_shared_relaxed(smb5, layout=bd)
            p_q11 = gl.amd.cdna4.mfma_scaled(
                p_ax11, None, "e4m3", p_bx11, None, "e4m3", zacc
            )
            p_q11 = gl.fma(p_q11 * p_sa11[:, None], p_sb11, g3)
            g3 = p_q11
            rtmp = g2 + g3
            rpair = rtmp
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
        offc += pidk * (M * N)
    r = accum.to(c_ptr.dtype.element_ty)
    gl.amd.cdna4.buffer_store(r, c_ptr, offc)


def block_scaled_mm_n3072_k1536_m32(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run the exact `(M, N, K) = (32, 3072, 1536)` specialization."""
    assert a.shape == (32, 1536)
    assert b.shape == (3072, 1536)
    assert a_scale.shape == (32, 12)
    assert b_scale.shape == (24, 12)
    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert a_scale.dtype == b_scale.dtype == torch.float32
    assert all(tensor.is_contiguous() for tensor in (a, b, a_scale, b_scale))
    assert out.shape == (32, 3072)
    assert out.dtype == torch.bfloat16
    assert out.device == a.device and out.is_contiguous()
    k_blocks = K // 128
    iterations = k_blocks // K_SPLITS

    warps_m = BLOCK_M // 16
    warps_n = BLOCK_N // 16
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
        size_per_thread=[16, 1],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )

    _g_gemm[(triton.cdiv(N, BLOCK_N), triton.cdiv(M, BLOCK_M), K_SPLITS)](
        a,
        b,
        out,
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
        PARTIAL=False,
        mfma=mfma,
        alayout=al,
        blayout=bl,
        num_warps=num_warps,
        waves_per_eu=1,
    )
    return out


__all__ = ["block_scaled_mm_n3072_k1536_m32"]

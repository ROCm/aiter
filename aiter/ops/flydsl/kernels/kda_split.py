# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Sequence-parallel KDA forward for gfx950 / CDNA4: a two-kernel split.

The fused kernel in :mod:`kda_kernel` gives one workgroup a whole (head, v-split)
and walks its chunks serially, because the state carries a true serial dependency.
That costs nothing when ``BH * DV_SPLIT`` covers the GPU, but at small batch it
leaves most of the machine idle -- at B=1, H=8 only 32 of 256 CUs have work, and
latency is exactly linear in the chunk count.

Almost none of the per-chunk work actually needs the state.  Writing the chunk body
with the residual grouped,

    V~ = A (V - (Gamma . K) S),   A = (I + T')^-1 Diag(beta)

every factor except ``S`` itself depends only on that chunk's q/k/v/g/beta.  So this
module splits the work in two:

``build_kda_prep_module``
    One workgroup per *chunk* (grid ``BH * NC``), hence fully parallel over the
    sequence.  Emits, per chunk, the six state-independent tiles the body needs:
    ``A`` (C x C), ``Gamma.K``, ``Gamma.Q`` (C x DK), ``Aqk`` (C x C),
    ``Kt = (K . gamma^C/Gamma)^T`` (DK x C) and ``dec = gamma^C`` (DK).

``build_kda_scan_module``
    Keeps the serial chunk walk and the state in registers exactly as the fused
    kernel does, but its chunk body is reduced to five matmuls plus one subtract:
    no cumulative scan, no exponentials, no triangular solve.

Emitting the explicit inverse ``A`` rather than the more common ``W = A(Gamma.K)``,
``U = AV`` pair is deliberate.  The ``W``/``U`` form would have the scan compute
``V~ = U - W S``, a difference of two quantities that are individually larger than
the result, both rounded to bf16 on the way through memory.  Keeping ``A`` instead
lets the scan form ``V - (Gamma.K) S`` in fp32 from the same inputs the fused kernel
uses, so the only new rounding is on ``A`` itself, whose entries are well scaled.
It also shrinks the solve workspace from C x (DK+DV) to C x C.

Measured on gfx950 at C=32, DK=DV=128 (kernel time, from rocprofv3)::

    B= 1 H= 8 T=2048   fused 0.561 ms   prep 0.024 + scan 0.158 = 0.182   3.08x
    B=32 H=16 T=2048   fused 2.774 ms   prep 0.903 + scan 0.790 = 1.693   1.64x

The gain is larger than pure occupancy accounting predicts, and it does not vanish
at large batch, because the split also makes the serial step *cheaper*: the scan's
chunk body is five matmuls over pre-built tiles, ~2.5 us against the fused kernel's
~8.8 us.  What decides the dispatch is therefore not occupancy but chunk count --
below ~32 chunks per head the fused kernel finishes inside the split's fixed
overhead.  The real cost of the split is memory: it materializes six tiles per
chunk, O(T) where the fused kernel needs none.

One implementation note dominates the scan's performance.  Its tiles are staged
with 128-bit loads that each feed eight LDS stores; doing it element-wise instead
costs ~56 dependent 16-bit global loads per thread per chunk and made the scan
6.2x slower (15.2 vs 2.5 us per chunk), which is enough on its own to turn the
whole split from a 3x win into a 1.8x loss.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import BFloat16, Float32

LOG2E = 1.4426950408889634
EXP2_CLAMP = 126.0


def build_kda_prep_module(
    *,
    BH: int,
    T: int,
    DK: int = 128,
    DV: int = 128,
    C: int = 32,
    BLOCK: int = 256,
    solve_bs: int = 8,
):
    """State-independent per-chunk tiles, one workgroup per chunk.

    Arguments (all packed by chunk, ``tile = bh * NC + n``)::

        q, k, g   [BH*NC, C*DK]      bf16 / bf16 / fp32
        beta      [BH*NC, C]         fp32

    Outputs::

        a_out     [BH*NC, C, C]      bf16   A = (I + T')^-1 Diag(beta)
        gk_out    [BH*NC, C, DK]     bf16   Gamma . K
        gq_out    [BH*NC, C, DK]     bf16   Gamma . Q
        aqk_out   [BH*NC, C, C]      bf16   Tril((Gamma.Q)(K/Gamma)^T)
        kt_out    [BH*NC, DK, C]     bf16   (K . gamma^C/Gamma)^T
        dec_out   [BH*NC, DK]        fp32   gamma^C
    """
    if T % C != 0:
        raise ValueError(f"T={T} must be a multiple of C={C}")
    if BLOCK != 2 * DK:
        raise ValueError(f"BLOCK must be 2*DK for the in-place cumulative sum; got {BLOCK}")
    if C % 32 != 0:
        raise ValueError(f"C must be a multiple of 32 (the MFMA K step); got {C}")

    NC = T // C
    CREF = C // 2
    PDK = DK + 8
    PC = C + 8
    PM = C + 4
    BS = min(solve_bs, C)
    NB = C // BS
    for _b in range(NB - 1):
        if ((C - _b * BS - BS) * C) % BLOCK != 0:
            raise ValueError(f"blocked solve needs BLOCK | (C-r0-BS)*C; got C={C} BS={BS}")

    N_G = C * DK  # fp32 cumulative log decay
    N_M = C * PM  # fp32 T', later the raw Tril(QK)
    N_A = C * PM  # fp32 solve workspace -> A
    N_K = C * DK  # bf16
    N_Q = C * DK  # bf16
    N_X = C * PDK  # bf16 MFMA A-operand scratch
    N_Y = max(C * PDK, DK * PC)  # bf16 MFMA B-operand scratch, two views
    N_AQ = C * PC  # bf16 masked Tril tile

    lds_bytes = 4 * (N_G + N_M + N_A) + 2 * (N_K + N_Q + N_X + N_Y + N_AQ) + 4 * (C + DK)
    if lds_bytes > 160 * 1024:
        raise ValueError(f"LDS request {lds_bytes} B exceeds the 160 KB gfx950 budget")

    @fx.struct
    class SharedStorage:
        g_cum: fx.Array[Float32, N_G, 16]
        m_mat: fx.Array[Float32, N_M, 16]
        a_mat: fx.Array[Float32, N_A, 16]
        k_s: fx.Array[BFloat16, N_K, 16]
        q_s: fx.Array[BFloat16, N_Q, 16]
        x_s: fx.Array[BFloat16, N_X, 16]
        y_s: fx.Array[BFloat16, N_Y, 16]
        aq_s: fx.Array[BFloat16, N_AQ, 16]
        beta_s: fx.Array[Float32, C, 16]
        gl_s: fx.Array[Float32, DK, 16]

    @flyc.kernel
    def kda_prep_kernel(
        arg_q: fx.Tensor,  # [BH*NC, C*DK] bf16
        arg_k: fx.Tensor,  # [BH*NC, C*DK] bf16
        arg_g: fx.Tensor,  # [BH*NC, C*DK] f32
        arg_beta: fx.Tensor,  # [BH*NC, C]  f32
        arg_a: fx.Tensor,  # [BH*NC, C, C]   bf16
        arg_gk: fx.Tensor,  # [BH*NC, C, DK] bf16
        arg_gq: fx.Tensor,  # [BH*NC, C, DK] bf16
        arg_aqk: fx.Tensor,  # [BH*NC, C, C]  bf16
        arg_kt: fx.Tensor,  # [BH*NC, DK, C] bf16
        arg_dec: fx.Tensor,  # [BH*NC, DK]    f32
        q_scale: fx.Float32,
    ):
        tid = fx.thread_idx.x
        tile = fx.block_idx.x

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        vG = lds.g_cum.view(fx.make_layout((C, DK), (DK, 1)))
        vM = lds.m_mat.view(fx.make_layout((C, C), (PM, 1)))
        vA = lds.a_mat.view(fx.make_layout((C, C), (PM, 1)))
        vK = lds.k_s.view(fx.make_layout((C, DK), (DK, 1)))
        vQ = lds.q_s.view(fx.make_layout((C, DK), (DK, 1)))
        vX = lds.x_s.view(fx.make_layout((C, DK), (PDK, 1)))
        vY = lds.y_s.view(fx.make_layout((C, DK), (PDK, 1)))
        vYt = lds.y_s.view(fx.make_layout((DK, C), (PC, 1)))
        vAq = lds.aq_s.view(fx.make_layout((C, C), (PC, 1)))
        vBeta = lds.beta_s.view(fx.make_layout(C, 1))
        vGl = lds.gl_s.view(fx.make_layout(DK, 1))

        fG = fx.logical_divide(lds.g_cum.view(fx.make_layout(N_G, 1)), fx.make_layout(4, 1))
        fK = fx.logical_divide(lds.k_s.view(fx.make_layout(N_K, 1)), fx.make_layout(8, 1))
        fQ = fx.logical_divide(lds.q_s.view(fx.make_layout(N_Q, 1)), fx.make_layout(8, 1))

        gQ = fx.rocdl.make_buffer_tensor(arg_q)
        gK = fx.rocdl.make_buffer_tensor(arg_k)
        gG = fx.rocdl.make_buffer_tensor(arg_g)
        gBeta = fx.rocdl.make_buffer_tensor(arg_beta)
        gA = fx.rocdl.make_buffer_tensor(arg_a)
        gGk = fx.rocdl.make_buffer_tensor(arg_gk)
        gGq = fx.rocdl.make_buffer_tensor(arg_gq)
        gAqk = fx.rocdl.make_buffer_tensor(arg_aqk)
        gKt = fx.rocdl.make_buffer_tensor(arg_kt)
        gDec = fx.rocdl.make_buffer_tensor(arg_dec)

        cp128_bf = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), BFloat16)
        cp128_f = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), Float32)
        cp16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), BFloat16)
        cp32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Float32)
        uni128 = fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
        uni32 = fx.make_copy_atom(fx.UniversalCopy(32), Float32)

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, BFloat16))
        tmma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr_mma = tmma.thr_slice(tid)
        thr_a = fx.make_tiled_copy_A(uni128, tmma).get_slice(tid)
        thr_b = fx.make_tiled_copy_B(uni128, tmma).get_slice(tid)
        thr_c32 = fx.make_tiled_copy_C(uni32, tmma).get_slice(tid)

        def gemm_lds(sA, sB, frag_C):
            fA = thr_mma.make_fragment_A(sA)
            fB = thr_mma.make_fragment_B(sB)
            fx.copy(uni128, thr_a.partition_S(sA), thr_a.retile(fA))
            fx.copy(uni128, thr_b.partition_S(sB), thr_b.retile(fB))
            fx.gemm(tmma, frag_C, fA, fB, frag_C)

        def store_acc_f32(frag, dst_view):
            fx.copy(uni32, thr_c32.retile(frag), thr_c32.partition_S(dst_view))

        def tile_iter(R, COLS):
            return tid % COLS, tid // COLS, BLOCK // COLS, R // (BLOCK // COLS)

        c_nclamp = fx.Float32(-EXP2_CLAMP)

        def ex2(x):
            return (-((-x).maximumf(c_nclamp))).maximumf(c_nclamp).exp2()

        frag_M = thr_mma.make_fragment_C(vM)
        e_ch = tid % C
        r_ch = tid // C

        # ---- load this chunk's g, k, q, beta ------------------------------
        gg = fx.logical_divide(fx.slice(gG, (tile, None)), fx.make_layout(4, 1))
        gk = fx.logical_divide(fx.slice(gK, (tile, None)), fx.make_layout(8, 1))
        gq = fx.logical_divide(fx.slice(gQ, (tile, None)), fx.make_layout(8, 1))
        for i in range_constexpr(N_G // (4 * BLOCK)):
            idx = tid + i * BLOCK
            r = fx.make_rmem_tensor(4, Float32)
            fx.copy_atom_call(cp128_f, fx.slice(gg, (None, idx)), r)
            fx.copy_atom_call(uni128, r, fx.slice(fG, (None, idx)))
        for i in range_constexpr(N_K // (8 * BLOCK)):
            idx = tid + i * BLOCK
            r = fx.make_rmem_tensor(8, BFloat16)
            fx.copy_atom_call(cp128_bf, fx.slice(gk, (None, idx)), r)
            fx.copy_atom_call(uni128, r, fx.slice(fK, (None, idx)))
            r2 = fx.make_rmem_tensor(8, BFloat16)
            fx.copy_atom_call(cp128_bf, fx.slice(gq, (None, idx)), r2)
            fx.copy_atom_call(uni128, r2, fx.slice(fQ, (None, idx)))
        gb = fx.logical_divide(fx.slice(gBeta, (tile, None)), fx.make_layout(1, 1))
        rb = fx.make_rmem_tensor(1, Float32)
        fx.copy_atom_call(cp32, fx.slice(gb, (None, tid % C)), rb)
        fx.memref_store(fx.memref_load_vec(rb)[0], vBeta, tid % C)
        gpu.barrier()

        # ---- in-place cumulative log-decay, scaled to log2 ----------------
        d_c = tid % DK
        half = tid // DK
        r0 = half * (C // 2)
        acc = fx.Float32(0.0)
        for i in range_constexpr(C // 2):
            acc = acc + fx.Float32(fx.memref_load(vG, (r0 + i, d_c)))
            fx.memref_store(acc * LOG2E, vG, (r0 + i, d_c))
        gpu.barrier()
        col, row0, rstep, nrow = tile_iter(C // 2, DK)
        for i in range_constexpr(nrow):
            rr = C // 2 + row0 + i * rstep
            base = fx.Float32(fx.memref_load(vG, (C // 2 - 1, col)))
            fx.memref_store(fx.Float32(fx.memref_load(vG, (rr, col))) + base, vG, (rr, col))
        gpu.barrier()

        # per-channel total chunk decay, straight out to global
        gl = fx.Float32(fx.memref_load(vG, (C - 1, d_c)))
        fx.memref_store(gl, vGl, d_c)
        if tid < DK:
            rd = fx.make_rmem_tensor(1, Float32)
            fx.memref_store(ex2(gl), rd, 0)
            fx.copy_atom_call(cp32, rd, fx.slice(fx.zipped_divide(
                fx.slice(gDec, (tile, None)), fx.make_tile(1)), (None, tid)))
        gpu.barrier()

        # ---- factored C x C operands, and the decay-factor cache ---------
        gcol, grow0, grstep, gnrow = tile_iter(C, DK)
        gref = fx.Float32(fx.memref_load(vG, (CREF, gcol)))
        gcr = fx.make_rmem_tensor(gnrow, Float32)
        egr = fx.make_rmem_tensor(gnrow, Float32)
        eir = fx.make_rmem_tensor(gnrow, Float32)
        for i in range_constexpr(gnrow):
            rr = grow0 + i * grstep
            gc = fx.Float32(fx.memref_load(vG, (rr, gcol)))
            e_i = ex2(gc - gref)
            fx.memref_store(gc, gcr, i)
            fx.memref_store(ex2(gc), egr, i)
            fx.memref_store(e_i, eir, i)
            kv = fx.memref_load(vK, (rr, gcol)).to(Float32)
            fx.memref_store((kv * ex2(gref - gc)).to(BFloat16), vY, (rr, gcol))
            fx.memref_store((kv * e_i).to(BFloat16), vX, (rr, gcol))
        gpu.barrier()

        frag_M.fill(0)
        gemm_lds(vX, vY, frag_M)
        store_acc_f32(frag_M, vM)
        gpu.barrier()

        # ---- T' = StrictTril(Diag(beta) Akk), and RHS = Diag(beta) ------
        col, row0, rstep, nrow = tile_iter(C, C)
        for i in range_constexpr(nrow):
            rr = row0 + i * rstep
            a = fx.Float32(fx.memref_load(vM, (rr, col)))
            b = fx.Float32(fx.memref_load(vBeta, rr))
            keep = fx.Int32(rr) > fx.Int32(col)
            fx.memref_store(keep.select(b * a, fx.Float32(0.0)), vM, (rr, col))
            diag = fx.Int32(rr) == fx.Int32(col)
            fx.memref_store(diag.select(b, fx.Float32(0.0)), vA, (rr, col))
        gpu.barrier()

        # ---- A = (I + T')^-1 Diag(beta), blocked substitution -----------
        # Same structure as the fused kernel's solve, but over C right-hand sides
        # instead of EV, and it costs nothing on the critical path here because
        # every chunk is solving concurrently.
        for b in range_constexpr(NB):
            rb0 = b * BS
            if tid < C:
                zb = fx.make_rmem_tensor(BS, Float32)
                for c in range_constexpr(BS):
                    fx.memref_store(fx.Float32(fx.memref_load(vA, (rb0 + c, e_ch))), zb, c)
                for c in range_constexpr(BS):
                    a = fx.Float32(fx.memref_load(zb, c))
                    for j in range_constexpr(c):
                        a = a - fx.Float32(fx.memref_load(vM, (rb0 + c, rb0 + j))) * fx.Float32(
                            fx.memref_load(zb, j)
                        )
                    fx.memref_store(a, zb, c)
                    fx.memref_store(a, vA, (rb0 + c, e_ch))
            gpu.barrier()
            if const_expr(b < NB - 1):
                for i in range_constexpr((C - rb0 - BS) * C // BLOCK):
                    n = tid + i * BLOCK
                    rr = rb0 + BS + n // C
                    ee = n % C
                    acc2 = fx.Float32(0.0)
                    for j in range_constexpr(BS):
                        acc2 = acc2 + fx.Float32(
                            fx.memref_load(vM, (rr, rb0 + j))
                        ) * fx.Float32(fx.memref_load(vA, (rb0 + j, ee)))
                    fx.memref_store(
                        fx.Float32(fx.memref_load(vA, (rr, ee))) - acc2, vA, (rr, ee)
                    )
                gpu.barrier()

        # A out
        col, row0, rstep, nrow = tile_iter(C, C)
        ga = fx.zipped_divide(fx.slice(gA, (tile, None, None)), fx.make_tile(1, 1))
        for i in range_constexpr(nrow):
            rr = row0 + i * rstep
            ra = fx.make_rmem_tensor(1, BFloat16)
            fx.memref_store(fx.Float32(fx.memref_load(vA, (rr, col))).to(BFloat16), ra, 0)
            fx.copy_atom_call(cp16, ra, fx.slice(ga, (None, (rr, col))))

        # ---- Gamma . K -> global ----------------------------------------
        ggk = fx.zipped_divide(fx.slice(gGk, (tile, None, None)), fx.make_tile(1, 1))
        for i in range_constexpr(gnrow):
            rr = grow0 + i * grstep
            kv = fx.memref_load(vK, (rr, gcol)).to(Float32)
            rv = fx.make_rmem_tensor(1, BFloat16)
            fx.memref_store((kv * fx.Float32(fx.memref_load(egr, i))).to(BFloat16), rv, 0)
            fx.copy_atom_call(cp16, rv, fx.slice(ggk, (None, (rr, gcol))))

        # ---- Gamma . Q -> global ----------------------------------------
        # Both q-dependent tiles feed only the output, O = (Gamma.Q) S + Aqk V~,
        # so folding the softmax scale in here keeps the scan kernel scale-free.
        ggq = fx.zipped_divide(fx.slice(gGq, (tile, None, None)), fx.make_tile(1, 1))
        for i in range_constexpr(gnrow):
            rr = grow0 + i * grstep
            qv = fx.memref_load(vQ, (rr, gcol)).to(Float32) * q_scale
            rv = fx.make_rmem_tensor(1, BFloat16)
            fx.memref_store((qv * fx.Float32(fx.memref_load(egr, i))).to(BFloat16), rv, 0)
            fx.copy_atom_call(cp16, rv, fx.slice(ggq, (None, (rr, gcol))))

        # ---- Aqk = Tril((Gamma.Q)(K/Gamma)^T) -> global -----------------
        for i in range_constexpr(gnrow):
            rr = grow0 + i * grstep
            qv = fx.memref_load(vQ, (rr, gcol)).to(Float32) * q_scale
            fx.memref_store(
                (qv * fx.Float32(fx.memref_load(eir, i))).to(BFloat16), vX, (rr, gcol)
            )
        gpu.barrier()
        frag_M.fill(0)
        gemm_lds(vX, vY, frag_M)
        store_acc_f32(frag_M, vM)
        gpu.barrier()
        col, row0, rstep, nrow = tile_iter(C, C)
        gaq = fx.zipped_divide(fx.slice(gAqk, (tile, None, None)), fx.make_tile(1, 1))
        for i in range_constexpr(nrow):
            rr = row0 + i * rstep
            a = fx.Float32(fx.memref_load(vM, (rr, col)))
            keep = fx.Int32(rr) >= fx.Int32(col)
            rv = fx.make_rmem_tensor(1, BFloat16)
            fx.memref_store(keep.select(a, fx.Float32(0.0)).to(BFloat16), rv, 0)
            fx.copy_atom_call(cp16, rv, fx.slice(gaq, (None, (rr, col))))
        gpu.barrier()

        # ---- Kt = (K . gamma^C / Gamma)^T -> global ---------------------
        glc = fx.Float32(fx.memref_load(vGl, gcol))
        gkt = fx.zipped_divide(fx.slice(gKt, (tile, None, None)), fx.make_tile(1, 1))
        for i in range_constexpr(gnrow):
            rr = grow0 + i * grstep
            gc = fx.Float32(fx.memref_load(gcr, i))
            kv = fx.memref_load(vK, (rr, gcol)).to(Float32)
            rv = fx.make_rmem_tensor(1, BFloat16)
            fx.memref_store((kv * ex2(glc - gc)).to(BFloat16), rv, 0)
            fx.copy_atom_call(cp16, rv, fx.slice(gkt, (None, (gcol, rr))))

    @flyc.jit
    def launch_kda_prep(
        arg_q: fx.Tensor,
        arg_k: fx.Tensor,
        arg_g: fx.Tensor,
        arg_beta: fx.Tensor,
        arg_a: fx.Tensor,
        arg_gk: fx.Tensor,
        arg_gq: fx.Tensor,
        arg_aqk: fx.Tensor,
        arg_kt: fx.Tensor,
        arg_dec: fx.Tensor,
        q_scale: fx.Float32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kda_prep_kernel(
            arg_q, arg_k, arg_g, arg_beta, arg_a, arg_gk, arg_gq, arg_aqk, arg_kt,
            arg_dec, q_scale,
        ).launch(grid=(BH * NC, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch_kda_prep.lds_bytes = lds_bytes
    launch_kda_prep.config = dict(BH=BH, T=T, DK=DK, DV=DV, C=C, BLOCK=BLOCK, NC=NC)
    return launch_kda_prep


def build_kda_scan_module(
    *,
    BH: int,
    T: int,
    DK: int = 128,
    DV: int = 128,
    C: int = 32,
    DV_SPLIT: int = 4,
    BLOCK: int = 256,
    has_initial_state: bool = False,
    store_final_state: bool = True,
    out_dtype=BFloat16,
):
    """Serial state scan over the prep kernel's tiles.

    One workgroup owns (batch*head, v-channel-split) and walks its chunks in order,
    exactly as the fused kernel does, but the body is only

        Z^T  = S^T (Gamma.K)^T                       EV x C
        R^T  = V^T - Z^T                             EV x C, fp32 residual
        V~^T = R^T A^T                               EV x C
        O    = (Gamma.Q) S + Aqk V~                  C x EV
        S^T <- Diag(gamma^C) S^T + V~^T Kt           EV x DK

    Computing the residual *transposed* is what makes this cheap: ``Z^T`` comes
    straight out of ``gemm_lds(S^T, Gamma.K)``, and reading ``V`` with its two
    indices swapped costs nothing, so ``V~^T`` lands already in the orientation
    both consumers want and no LDS transpose is ever needed.
    """
    if T % C != 0:
        raise ValueError(f"T={T} must be a multiple of C={C}")
    if DV % DV_SPLIT != 0:
        raise ValueError(f"DV={DV} must be a multiple of DV_SPLIT={DV_SPLIT}")

    NC = T // C
    EV = DV // DV_SPLIT
    if EV % 32 != 0:
        raise ValueError(f"EV=DV/DV_SPLIT must be a multiple of 32; got {EV}")
    PDK = DK + 8
    PC = C + 8
    PZ = C + 4
    PO = EV + 4

    N_GK = C * PDK  # bf16 Gamma.K
    N_GQ = C * PDK  # bf16 Gamma.Q
    N_A = C * PC  # bf16 A
    N_AQ = C * PC  # bf16 Aqk
    N_KT = DK * PC  # bf16 Kt
    N_STB = EV * PDK  # bf16 mirror of S^T
    N_VN = EV * PC  # bf16 V~^T
    # fp32 residual^T (EV x C); also viewed C x EV to shape the output accumulator
    N_RT = max(EV * PZ, C * PO)
    N_SF = EV * PDK  # fp32 staging for the initial state

    lds_bytes = (
        2 * (N_GK + N_GQ + N_A + N_AQ + N_KT + N_STB + N_VN)
        + 4 * (N_RT + DK)
        + (4 * N_SF if has_initial_state else 0)
    )
    if lds_bytes > 160 * 1024:
        raise ValueError(f"LDS request {lds_bytes} B exceeds the 160 KB gfx950 budget")

    @fx.struct
    class SharedStorage:
        gk_s: fx.Array[BFloat16, N_GK, 16]
        gq_s: fx.Array[BFloat16, N_GQ, 16]
        a_s: fx.Array[BFloat16, N_A, 16]
        aq_s: fx.Array[BFloat16, N_AQ, 16]
        kt_s: fx.Array[BFloat16, N_KT, 16]
        st_b: fx.Array[BFloat16, N_STB, 16]
        vn_s: fx.Array[BFloat16, N_VN, 16]
        rt_s: fx.Array[Float32, N_RT, 16]
        dec_s: fx.Array[Float32, DK, 16]
        sf_s: fx.Array[Float32, N_SF if has_initial_state else 1, 16]

    @flyc.kernel
    def kda_scan_kernel(
        arg_a: fx.Tensor,  # [BH*NC, C*C]  bf16, flat for vectorized staging
        arg_gk: fx.Tensor,  # [BH*NC, C*DK] bf16
        arg_gq: fx.Tensor,  # [BH*NC, C*DK] bf16
        arg_aqk: fx.Tensor,  # [BH*NC, C*C]  bf16
        arg_kt: fx.Tensor,  # [BH*NC, DK*C] bf16
        arg_dec: fx.Tensor,  # [BH*NC, DK]    f32
        arg_v: fx.Tensor,  # [BH*NC, C, DV_SPLIT, EV] bf16
        arg_o: fx.Tensor,  # [BH*NC, C, DV_SPLIT, EV] out
        arg_h0: fx.Tensor,  # [BH, DV_SPLIT, EV, DK]   f32
        arg_ht: fx.Tensor,  # [BH, DV_SPLIT, EV, DK]   f32
        n_chunks: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        bh = bid // DV_SPLIT
        sp = bid % DV_SPLIT

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        vGK = lds.gk_s.view(fx.make_layout((C, DK), (PDK, 1)))
        vGQ = lds.gq_s.view(fx.make_layout((C, DK), (PDK, 1)))
        vA = lds.a_s.view(fx.make_layout((C, C), (PC, 1)))
        vAq = lds.aq_s.view(fx.make_layout((C, C), (PC, 1)))
        vKt = lds.kt_s.view(fx.make_layout((DK, C), (PC, 1)))
        vStb = lds.st_b.view(fx.make_layout((EV, DK), (PDK, 1)))
        vVn = lds.vn_s.view(fx.make_layout((EV, C), (PC, 1)))
        vRt = lds.rt_s.view(fx.make_layout((EV, C), (PZ, 1)))
        vOsh = lds.rt_s.view(fx.make_layout((C, EV), (PO, 1)))
        vDec = lds.dec_s.view(fx.make_layout(DK, 1))
        # stride-0 broadcast of the per-channel chunk decay over the S^T tile, so it
        # can be partitioned exactly like the state accumulator.
        vDecC = lds.dec_s.view(fx.make_layout((EV, DK), (0, 1)))
        vSf = lds.sf_s.view(
            fx.make_layout((EV, DK), (PDK, 1)) if has_initial_state else fx.make_layout(1, 1)
        )

        gA = fx.rocdl.make_buffer_tensor(arg_a)
        gGk = fx.rocdl.make_buffer_tensor(arg_gk)
        gGq = fx.rocdl.make_buffer_tensor(arg_gq)
        gAqk = fx.rocdl.make_buffer_tensor(arg_aqk)
        gKt = fx.rocdl.make_buffer_tensor(arg_kt)
        gDec = fx.rocdl.make_buffer_tensor(arg_dec)
        gV = fx.rocdl.make_buffer_tensor(arg_v)
        gO = fx.rocdl.make_buffer_tensor(arg_o)

        cp16 = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), BFloat16)
        cp32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Float32)
        cp128_bf = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), BFloat16)
        uni128 = fx.make_copy_atom(fx.UniversalCopy128b(), BFloat16)
        uni32 = fx.make_copy_atom(fx.UniversalCopy(32), Float32)
        uni16 = fx.make_copy_atom(fx.UniversalCopy(16), BFloat16)

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, BFloat16))
        tmma = fx.make_tiled_mma(mma_atom, fx.make_layout((2, 2, 1), (1, 2, 0)))
        thr_mma = tmma.thr_slice(tid)
        thr_a = fx.make_tiled_copy_A(uni128, tmma).get_slice(tid)
        thr_b = fx.make_tiled_copy_B(uni128, tmma).get_slice(tid)
        thr_c32 = fx.make_tiled_copy_C(uni32, tmma).get_slice(tid)
        thr_cb = fx.make_tiled_copy_C(uni16, tmma).get_slice(tid)
        thr_cbn = fx.make_tiled_copy_C(uni16, tmma).get_slice(tid)

        # NOTE: as in the fused kernel, every ``thr_*.method()`` call must stay
        # inside one of these closures -- the AST rewriter turns any object whose
        # method is invoked directly in a dynamic loop body into an scf.for
        # iter_arg, and ThrCopy cannot be rebuilt from IR values.
        def gemm_lds(sA, sB, frag_C):
            fA = thr_mma.make_fragment_A(sA)
            fB = thr_mma.make_fragment_B(sB)
            fx.copy(uni128, thr_a.partition_S(sA), thr_a.retile(fA))
            fx.copy(uni128, thr_b.partition_S(sB), thr_b.retile(fB))
            fx.gemm(tmma, frag_C, fA, fB, frag_C)

        def store_acc_f32(frag, dst_view):
            fx.copy(uni32, thr_c32.retile(frag), thr_c32.partition_S(dst_view))

        def load_acc_f32(frag, src_view):
            fx.copy(uni32, thr_c32.partition_S(src_view), thr_c32.retile(frag))

        def tile_iter(R, COLS):
            return tid % COLS, tid // COLS, BLOCK // COLS, R // (BLOCK // COLS)

        # ── staging: 8-wide global reads into padded LDS ─────────────────────
        # A row width that is a multiple of 8 means a vector never straddles rows,
        # so one buffer_load_dwordx4 feeds 8 scalar LDS stores.  Doing this
        # element-at-a-time instead costs ~56 dependent global loads per thread
        # per chunk, which dominated everything else in the scan.
        def _stage_one(gvec, dst, WV, vidx):
            r = fx.make_rmem_tensor(8, BFloat16)
            fx.copy_atom_call(cp128_bf, fx.slice(gvec, (None, vidx)), r)
            vals = fx.memref_load_vec(r)
            rr = vidx // WV
            c8 = (vidx % WV) * 8
            for j in range_constexpr(8):
                fx.memref_store(vals[j], dst, (rr, c8 + j))

        def stage(gflat, dst, W, N):
            gvec = fx.logical_divide(gflat, fx.make_layout(8, 1))
            WV = W // 8
            if const_expr(N // 8 >= BLOCK):
                for i in range_constexpr(N // 8 // BLOCK):
                    _stage_one(gvec, dst, WV, tid + i * BLOCK)
            else:
                if tid < N // 8:
                    _stage_one(gvec, dst, WV, tid)

        # ── state accumulator: S^T, fp32, never rounded across chunks ───────
        frag_S = thr_mma.make_fragment_C(vStb)
        frag_S.fill(0)
        if const_expr(has_initial_state):
            gH0 = fx.rocdl.make_buffer_tensor(arg_h0)
            gh = fx.zipped_divide(fx.slice(gH0, (bh, sp, None, None)), fx.make_tile(1, 1))
            col, row0, rstep, nrow = tile_iter(EV, DK)
            for i in range_constexpr(nrow):
                rr = row0 + i * rstep
                rh = fx.make_rmem_tensor(1, Float32)
                fx.copy_atom_call(cp32, fx.slice(gh, (None, (rr, col))), rh)
                fx.memref_store(fx.memref_load_vec(rh)[0], vSf, (rr, col))
            gpu.barrier()
            load_acc_f32(frag_S, vSf)
            gpu.barrier()

        frag_Sb = fx.make_fragment_like(frag_S, BFloat16.ir_type)

        def publish_state():
            fx.memref_store_vec(fx.memref_load_vec(frag_S).to(BFloat16), frag_Sb)
            fx.copy(uni16, thr_cb.retile(frag_Sb), thr_cb.partition_S(vStb))

        # EV x C accumulators (Z^T and V~^T) and the C x EV output accumulator
        frag_Zt = thr_mma.make_fragment_C(vRt)
        frag_Vt = thr_mma.make_fragment_C(vVn)
        frag_Vtb = fx.make_fragment_like(frag_Vt, BFloat16.ir_type)
        frag_O = thr_mma.make_fragment_C(vOsh)
        frag_dec = fx.make_fragment_like(frag_S, Float32.ir_type)
        cp_out = fx.make_copy_atom(
            fx.rocdl.BufferCopy32b()
            if const_expr(out_dtype is Float32)
            else fx.rocdl.BufferCopy16b(),
            out_dtype,
        )
        thr_co = fx.make_tiled_copy_C(cp_out, tmma).get_slice(tid)
        frag_Oo = fx.make_fragment_like(frag_O, out_dtype.ir_type)

        def publish_vn():
            """V~^T -> LDS bf16; both remaining matmuls read it as an operand."""
            fx.memref_store_vec(fx.memref_load_vec(frag_Vt).to(BFloat16), frag_Vtb)
            fx.copy(uni16, thr_cbn.retile(frag_Vtb), thr_cbn.partition_S(vVn))

        def apply_chunk_decay():
            """S^T <- Diag(gamma^C) S^T, broadcasting the per-channel decay."""
            fx.copy(uni32, thr_c32.partition_S(vDecC), thr_c32.retile(frag_dec))
            fx.memref_store_vec(
                fx.memref_load_vec(frag_S) * fx.memref_load_vec(frag_dec), frag_S
            )

        def store_output(gview):
            fx.memref_store_vec(fx.memref_load_vec(frag_O).to(out_dtype), frag_Oo)
            fx.copy(cp_out, thr_co.retile(frag_Oo), thr_co.partition_S(gview))

        # seed the bf16 mirror: chunk 0 reads it before any publish_state()
        publish_state()
        gpu.barrier()

        for n in range(n_chunks):
            tile = bh * NC + n

            # ---- stage this chunk's prep tiles into LDS ------------------
            gpu.barrier()
            stage(fx.slice(gGk, (tile, None)), vGK, DK, C * DK)
            stage(fx.slice(gGq, (tile, None)), vGQ, DK, C * DK)
            stage(fx.slice(gKt, (tile, None)), vKt, C, DK * C)
            stage(fx.slice(gA, (tile, None)), vA, C, C * C)
            stage(fx.slice(gAqk, (tile, None)), vAq, C, C * C)
            if tid < DK:
                rd = fx.make_rmem_tensor(1, Float32)
                fx.copy_atom_call(
                    cp32,
                    fx.slice(
                        fx.zipped_divide(fx.slice(gDec, (tile, None)), fx.make_tile(1)),
                        (None, tid),
                    ),
                    rd,
                )
                fx.memref_store(fx.memref_load_vec(rd)[0], vDec, tid)
            gpu.barrier()

            # ---- Z^T = S^T (Gamma.K)^T ----------------------------------
            frag_Zt.fill(0)
            gemm_lds(vStb, vGK, frag_Zt)
            store_acc_f32(frag_Zt, vRt)
            gpu.barrier()

            # ---- R^T = V^T - Z^T, read V with its indices swapped -------
            vcol, vrow0, vrstep, vnrow = tile_iter(EV, C)
            gv = fx.zipped_divide(
                fx.slice(gV, (tile, None, sp, None)), fx.make_tile(1, 1)
            )
            for i in range_constexpr(vnrow):
                rr = vrow0 + i * vrstep
                rv = fx.make_rmem_tensor(1, BFloat16)
                fx.copy_atom_call(cp16, fx.slice(gv, (None, (vcol, rr))), rv)
                z = fx.Float32(fx.memref_load(vRt, (rr, vcol)))
                fx.memref_store(
                    (fx.memref_load_vec(rv)[0].to(Float32) - z).to(BFloat16), vVn, (rr, vcol)
                )
            gpu.barrier()

            # ---- V~^T = R^T A^T ------------------------------------------
            frag_Vt.fill(0)
            gemm_lds(vVn, vA, frag_Vt)
            gpu.barrier()
            publish_vn()
            gpu.barrier()

            # ---- O = (Gamma.Q) S + Aqk V~ --------------------------------
            frag_O.fill(0)
            gemm_lds(vGQ, vStb, frag_O)
            gemm_lds(vAq, vVn, frag_O)
            store_output(fx.slice(gO, (tile, None, sp, None)))

            # ---- S^T <- Diag(gamma^C) S^T + V~^T Kt ----------------------
            apply_chunk_decay()
            gemm_lds(vVn, vKt, frag_S)
            gpu.barrier()
            publish_state()

        if const_expr(store_final_state):
            gHt = fx.rocdl.make_buffer_tensor(arg_ht)
            thr_c32g = fx.make_tiled_copy_C(cp32, tmma).get_slice(tid)
            fx.copy(
                cp32,
                thr_c32g.retile(frag_S),
                thr_c32g.partition_S(fx.slice(gHt, (bh, sp, None, None))),
            )

    @flyc.jit
    def launch_kda_scan(
        arg_a: fx.Tensor,
        arg_gk: fx.Tensor,
        arg_gq: fx.Tensor,
        arg_aqk: fx.Tensor,
        arg_kt: fx.Tensor,
        arg_dec: fx.Tensor,
        arg_v: fx.Tensor,
        arg_o: fx.Tensor,
        arg_h0: fx.Tensor,
        arg_ht: fx.Tensor,
        n_chunks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        kda_scan_kernel(
            arg_a, arg_gk, arg_gq, arg_aqk, arg_kt, arg_dec,
            arg_v, arg_o, arg_h0, arg_ht, n_chunks,
        ).launch(grid=(BH * DV_SPLIT, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch_kda_scan.lds_bytes = lds_bytes
    launch_kda_scan.config = dict(
        BH=BH, T=T, DK=DK, DV=DV, C=C, DV_SPLIT=DV_SPLIT, BLOCK=BLOCK, EV=EV, NC=NC
    )
    return launch_kda_scan

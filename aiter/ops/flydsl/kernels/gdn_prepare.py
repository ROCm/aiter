# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL GDN prefill chunk-prepare kernel for the BT=64 path.

For each chunk, form the strictly lower-triangular gated KKT matrix ``A`` and
``C = (I + A)^-1``, then produce:

    g_cumsum : [B, H, T]    in-chunk prefix sum of g
    w_bar    : [B, H, T, K] C @ (k * beta * exp(g_cumsum))
    u_bar    : [B, H, T, V] C @ (v * beta)

Inputs are token-major and outputs are head-major. ``g_scale`` controls whether
``g_cumsum`` is published in natural-log or log2 space.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    const_expr,
    gpu,
    range_constexpr,
)


def _exp2_f32(x):
    """Evaluate exp2 directly; decay exponents are always non-positive."""
    return fx.Float32(fx.rocdl.exp2(fx.Float32.ir_type, x.ir_value()))


WARP_SIZE = 64
BLOCK_THREADS = 256


def _mfma16(a_bf16x4, b_bf16x4, c_f32x4):
    frag_a = fx.make_rmem_tensor(4, fx.BFloat16)
    frag_b = fx.make_rmem_tensor(4, fx.BFloat16)
    frag_c = fx.make_rmem_tensor(4, fx.Float32)
    frag_a.store(fx.BFloat16x4(a_bf16x4))
    frag_b.store(fx.BFloat16x4(b_bf16x4))
    frag_c.store(fx.Float32x4(c_f32x4))
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    fx.gemm(mma, frag_c, frag_a, frag_b, frag_c)
    return fx.Float32x4(frag_c.load())


def _acc16_n4():
    """Return a zeroed four-tile accumulator."""
    frag = fx.make_rmem_tensor((4, 1, 4), fx.Float32)
    frag.store(fx.Vector.filled(16, 0.0, fx.Float32))
    return frag


def _gemm16_n4(frag_c, a_bf16x4, bs):
    frag_a = fx.make_rmem_tensor((4, 1), fx.BFloat16)
    frag_b = fx.make_rmem_tensor((4, 4), fx.BFloat16)
    a_v = fx.BFloat16x4(a_bf16x4)
    for p in range(4):
        frag_a[p, 0] = a_v[p]
        for en in range(4):
            frag_b[p, en] = fx.BFloat16x4(bs[en])[p]
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    fx.gemm(mma, frag_c, frag_a, frag_b, frag_c)


def _accum_to_bf16x4(d_f32x4):
    d = fx.Float32x4(d_f32x4)
    return fx.BFloat16x4([d[p].to(fx.BFloat16) for p in range(4)])


# Keep the staging layout consistent between writes and reads.
SWZ_SR = 16


def _swizzled_vt_view(s_t):
    """Return the logical ``[feature, token]`` staging view."""
    coord_swizzle = fx.static(
        fx.CoordSwizzleType.get(2, SWZ_SR.bit_length() - 1, [0], 2, [1])
    )
    logical_coords = fx.make_composed_layout(
        coord_swizzle, fx.make_identity_layout((WARP_SIZE, WARP_SIZE))
    )
    row_stride = fx.get(fx.get_stride(fx.get_layout(s_t)), 0)
    physical = fx.make_layout((WARP_SIZE, row_stride), (row_stride, 1))
    return fx.make_view(
        fx.get_iter(s_t),
        fx.make_composed_layout(physical, logical_coords),
    )


def _load_mfma_tile_vt_tiled(s_t, n_tile, k_tile, lane, tiled_copy, copy_atom):
    tile = _tile16_view(s_t, n_tile * 16, k_tile * 16)
    # Match the copy coordinate to the staging layout.
    thr_copy = tiled_copy.get_slice(lane ^ (n_tile * 16))
    part_src = thr_copy.partition_S(tile)
    frag = fx.make_fragment_like(part_src)
    fx.copy(copy_atom, part_src, frag)
    return fx.BFloat16x4(frag.load())


def _tile16_view(s_t, row_base, col_base, transpose=False):
    layout = fx.get_layout(s_t)
    row_stride = fx.get(fx.get_stride(layout), 0)
    strides = (1, row_stride) if transpose else (row_stride, 1)
    offset = fx.crd2idx((row_base, col_base), layout)
    return fx.make_view(
        fx.get_iter(s_t) + offset,
        fx.make_layout((16, 16), strides),
    )


def _load_fp32_tile(
    s_t, row_base, col_base, thr_copy, copy_atom, transpose=False, negate=False
):
    """Load one fp32 tile and cast it to bf16."""
    src = _tile16_view(s_t, row_base, col_base, transpose)
    part_src = thr_copy.partition_S(src)
    frag = fx.make_fragment_like(part_src)
    fx.copy(copy_atom, part_src, frag)
    vals = fx.Float32x4(frag.load())
    if negate:
        vals = -vals
    return fx.BFloat16x4([vals[p].to(fx.BFloat16) for p in range(4)])


def _store_fp32_tile(s_t, row_base, col_base, d_f32x4, thr_copy, copy_atom):
    """Store one fp32 tile."""
    dst = _tile16_view(s_t, row_base, col_base)
    part_dst = thr_copy.partition_D(dst)
    frag = fx.make_fragment_like(part_dst)
    frag.store(fx.Float32x4(d_f32x4))
    fx.copy(copy_atom, frag, part_dst)


def _identity_frag(lane):
    """Return the identity fragment for one diagonal block."""
    n = lane % 16
    mb4 = (lane // 16) * 4
    elems = [((mb4 + p) == n).select(1.0, 0.0) for p in range(4)]
    return fx.Float32x4(elems)


def _wy_prefetch(src, copy_atom, thr_copy):
    part_src = thr_copy.partition_S(src)
    regs = fx.make_fragment_like(part_src)
    fx.copy(copy_atom, part_src, regs)
    return regs


def _wy_epilogue_to_lds(wy, dst, copy_atom):
    for en in range_constexpr(4):
        for p in range_constexpr(4):
            dst_p = dst[(None, p), 0, en]
            frag_p = fx.make_fragment_like(dst_p)
            frag_p.store(fx.BFloat16x1(wy[p, 0, en].to(fx.BFloat16)))
            fx.copy(copy_atom, frag_p, dst_p)


def _wy_scatter(regs, s_vT, s_beta, gc, is_k, tid, stage_iters, svec_per_row, svec):
    """Scale and transpose the RHS while preserving its staging layout."""
    for it in range(stage_iters):
        vals = fx.Vector(regs[None, it, 0].load())
        p = tid + it * BLOCK_THREADS
        j = p // svec_per_row
        row0 = (p % svec_per_row) * svec
        scale_j = (gc if is_k else s_beta)[j]
        for vv in range(svec):
            val = vals[vv].to(fx.Float32)
            s_vT[row0 + vv, j] = (val * scale_j).to(fx.BFloat16)


def _wave_inclusive_scan(val, tid, width, zero):
    csum = val
    s = 1
    while s < width:
        prev = gpu.shuffle(csum, s, width, mode="up")
        csum = csum + (tid >= s).select(prev, zero)
        s <<= 1
    return csum


@functools.lru_cache(maxsize=64)
def compile_gdn_prepare(
    *,
    BT: int = 64,
    K: int = 128,
    V: int = 128,
    is_varlen: bool = False,
    g_scale: float = 1.0,
):
    """Compile the KKT, triangular inverse, and WY preparation kernel.

    ``g_scale`` affects only the published ``g_cumsum``; decay calculations
    remain in the natural-log domain.
    """
    assert BT == 64 and K == 128 and V == 128, "gdn_prepare targets the BT=64 main path"
    LOG2E = 1.4426950408889634

    KS = K + 4
    VTS = BT + 4
    ASA = BT + 1
    BK_SUB = 64
    OUT_S = BK_SUB
    N_K_ITERS = K // BK_SUB
    N_V_ITERS = V // BK_SUB
    assert BT * VTS * 2 + BT * OUT_S * 2 <= max(BT * KS * 2, BT * ASA * 4)

    @fx.struct
    class _WYStaging:
        s_vT: fx.Array[fx.BFloat16, BT * VTS, 16]
        s_out: fx.Array[fx.BFloat16, BT * OUT_S, 16]

    # These fields alias because their lifetimes do not overlap.
    @fx.union
    class _P0:
        s_k: fx.Array[fx.BFloat16, BT * KS, 16]
        s_A: fx.Array[fx.Float32, BT * ASA, 16]
        wy: _WYStaging

    @fx.struct
    class _SharedStorage:
        s_g: fx.Array[fx.Float32, BT, 16]
        s_beta: fx.Array[fx.Float32, BT, 16]
        p0: _P0

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1], name="gdn_prepare_kernel")
    def gdn_prepare_kernel(
        k_t: fx.Tensor,
        v_t: fx.Tensor,
        g_t: fx.Tensor,
        beta_t: fx.Tensor,
        cu_t: fx.Tensor,
        wbar_t: fx.Tensor,
        ubar_t: fx.Tensor,
        gcs_t: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        Hg: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        rep = H // Hg
        buf_copy_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        buf_copy_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Float32)

        lane = tid % WARP_SIZE
        warp = tid // WARP_SIZE
        warp16 = warp * 16

        i_t = fx.Int32(gpu.block_id("x"))
        i_bh = fx.Int32(gpu.block_id("y"))
        i_b = i_bh // H
        i_h = i_bh % H
        if const_expr(is_varlen):
            cu_g = fx.rocdl.make_buffer_tensor(cu_t)
            bos_src = fx.make_view(fx.get_iter(cu_g) + i_b, fx.make_layout(1, 1))
            nxt_src = fx.make_view(fx.get_iter(cu_g) + i_b + 1, fx.make_layout(1, 1))
            bos_reg = fx.make_fragment_like(bos_src)
            nxt_reg = fx.make_fragment_like(nxt_src)
            fx.copy(buf_copy_i32, bos_src, bos_reg)
            fx.copy(buf_copy_i32, nxt_src, nxt_reg)
            bos = fx.Int32(bos_reg[0])
            nxt = fx.Int32(nxt_reg[0])
            seqlen = nxt - bos
        else:
            bos = i_b * T
            seqlen = T

        i_hg = i_h // rep
        chunk_start = i_t * BT
        n_chunks = (seqlen + BT - 1) // BT
        active = i_t < n_chunks

        if active:
            # Bounded views handle ragged final chunks.
            seq_end = bos + seqlen
            lim_gb = (seq_end * H * 4).ir_value()
            lim_k = (seq_end * Hg * K * 2).ir_value()
            lim_v = (seq_end * H * V * 2).ir_value()
            # Head-major output range for this chunk.
            if const_expr(is_varlen):
                hm_row = i_h * T + bos  # B == 1, T == T_flat
                hm_end = i_h * T + seq_end
            else:
                hm_row = (i_b * H + i_h) * T
                hm_end = hm_row + seqlen
            hm_row = hm_row + chunk_start
            lim_gcs = (hm_end * 4).ir_value()
            lim_ub = (hm_end * V * 2).ir_value()
            lim_wb = (hm_end * K * 2).ir_value()
            k_g = fx.rocdl.make_buffer_tensor(k_t, num_records_bytes=lim_k)
            v_g = fx.rocdl.make_buffer_tensor(v_t, num_records_bytes=lim_v)
            g_g = fx.rocdl.make_buffer_tensor(g_t, num_records_bytes=lim_gb)
            beta_g = fx.rocdl.make_buffer_tensor(beta_t, num_records_bytes=lim_gb)
            gcs_g = fx.rocdl.make_buffer_tensor(gcs_t, num_records_bytes=lim_gcs)
            wbar_g = fx.rocdl.make_buffer_tensor(wbar_t, num_records_bytes=lim_wb)
            ubar_g = fx.rocdl.make_buffer_tensor(ubar_t, num_records_bytes=lim_ub)

            lds = fx.SharedAllocator().allocate(_SharedStorage)
            s_g = lds.s_g.peek().view(fx.make_layout(BT, 1))
            s_beta = lds.s_beta.peek().view(fx.make_layout(BT, 1))
            s_k = lds.p0.s_k.peek().view(fx.make_layout((BT, K), (KS, 1)))
            s_A = lds.p0.s_A.peek().view(fx.make_layout((BT, BT), (ASA, 1)))
            s_vT = lds.p0.wy.s_vT.peek().view(fx.make_layout((BT, BT), (VTS, 1)))
            s_vT_swz = _swizzled_vt_view(s_vT)
            s_out = lds.p0.wy.s_out.peek().view(fx.make_layout((BT, OUT_S), (OUT_S, 1)))

            base_gb = (bos + chunk_start) * H + i_h
            base_k = ((bos + chunk_start) * Hg + i_hg) * K
            base_v = ((bos + chunk_start) * H + i_h) * V
            base_gcs = hm_row
            base_ub = hm_row * V
            base_wb = hm_row * K
            zero_f = fx.Float32(0.0)
            is_lane = tid < BT

            buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
            lds_copy16 = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
            lds_copy32 = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
            lds_copy128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
            # The padded row stride requires narrower aligned copies.
            lds_copy64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)

            mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
            block_mma = fx.make_tiled_mma(
                mma_atom, fx.make_layout((4, 1, 1), (1, 0, 0))
            )
            block_thr_mma = block_mma.thr_slice(tid)
            block_copy_A = fx.make_tiled_copy_A(lds_copy64, block_mma).get_slice(tid)
            block_copy_B = fx.make_tiled_copy_B(lds_copy64, block_mma).get_slice(tid)
            block_copy_C32 = fx.make_tiled_copy_C(lds_copy32, block_mma).get_slice(tid)
            block_copy_C16 = fx.make_tiled_copy_C(lds_copy16, block_mma).get_slice(tid)

            wave_mma = fx.make_tiled_mma(mma_atom, fx.make_layout((1, 1, 1), (0, 0, 0)))
            wave_copy_A32 = fx.make_tiled_copy_A(lds_copy32, wave_mma).get_slice(lane)
            wave_copy_B32 = fx.make_tiled_copy_B(lds_copy32, wave_mma).get_slice(lane)
            wave_tiled_copy_B16 = fx.make_tiled_copy_B(lds_copy64, wave_mma)
            wave_copy_C32 = fx.make_tiled_copy_C(lds_copy32, wave_mma).get_slice(lane)

            k_load_copy = fx.make_tiled_copy_tv(
                buf_copy,
                fx.make_layout((16, 16), (16, 1)),
                fx.make_layout((1, 8), (1, 1)),
            ).get_slice(tid)
            k_store_copy = fx.make_tiled_copy_tv(
                lds_copy64,
                fx.make_layout((16, 16), (16, 1)),
                fx.make_layout((1, 8), (1, 1)),
            ).get_slice(tid)
            k_src = k_load_copy.partition_S(
                fx.make_view(
                    fx.get_iter(k_g) + base_k,
                    fx.make_layout((BT, K), (Hg * K, 1)),
                )
            )
            k_dst = k_store_copy.partition_D(s_k)

            if is_lane:
                gi = base_gb + tid * H
                g_src = fx.make_view(fx.get_iter(g_g) + gi, fx.make_layout(1, 1))
                beta_src = fx.make_view(fx.get_iter(beta_g) + gi, fx.make_layout(1, 1))
                g_reg = fx.make_fragment_like(g_src)
                beta_reg = fx.make_fragment_like(beta_src)
                fx.copy(buf_copy_f32, g_src, g_reg)
                fx.copy(buf_copy_f32, beta_src, beta_reg)
                gv = g_reg[0]
                bv = beta_reg[0]
                csum = _wave_inclusive_scan(gv, tid, BT, zero_f)
                s_g[tid] = csum
                s_beta[tid] = bv
                out = csum if g_scale == 1.0 else csum * g_scale
                gcs_dst = fx.make_view(
                    fx.get_iter(gcs_g) + base_gcs + tid, fx.make_layout(1, 1)
                )
                gcs_reg = fx.make_fragment_like(gcs_dst)
                gcs_reg[0] = out
                fx.copy(buf_copy_f32, gcs_reg, gcs_dst)

            kkt = block_thr_mma.make_fragment_C(s_A)
            kkt.fill(0.0)
            kregs = fx.make_fragment_like(k_src)
            fx.copy(buf_copy, k_src, kregs)
            gc = s_g
            # Publish k, g, and beta before forming KKT.
            fx.copy(lds_copy64, k_store_copy.retile(kregs), k_dst)
            gpu.barrier()

            # Form the gated strictly lower-triangular KKT matrix.
            for ek in range_constexpr(K // 16):
                s_k_tile = fx.make_view(
                    fx.get_iter(s_k) + ek * 16,
                    fx.make_layout((BT, 16), (KS, 1)),
                )
                frag_a = block_thr_mma.make_fragment_A(s_k_tile)
                frag_b = block_thr_mma.make_fragment_B(s_k_tile)
                fx.copy(
                    lds_copy64,
                    block_copy_A.partition_S(s_k_tile),
                    block_copy_A.retile(frag_a),
                )
                fx.copy(
                    lds_copy64,
                    block_copy_B.partition_S(s_k_tile),
                    block_copy_B.retile(frag_b),
                )
                fx.gemm(mma_atom, kkt, frag_a, frag_b, kkt)
            # s_A aliases s_k, so all s_k reads must complete before storing s_A.
            gpu.barrier()
            n16 = lane % 16
            mb4 = (lane // 16) * 4
            for en in range_constexpr(4):
                for p in range_constexpr(4):
                    s = warp16 + mb4 + p
                    r = en * 16 + n16
                    cval = kkt[(p, 0), 0, en]
                    beta_s = s_beta[s]
                    gc_s = gc[s]
                    gc_r = gc[r]
                    decay = _exp2_f32((gc_s - gc_r) * LOG2E)
                    aval = (s > r).select(cval * beta_s * decay, zero_f)
                    kkt[(p, 0), 0, en] = aval
            fx.copy(
                lds_copy32,
                block_copy_C32.retile(kkt),
                block_copy_C32.partition_D(s_A),
            )
            gpu.barrier()

            # Replace g_cumsum with the scale for the w_bar right-hand side.
            if is_lane:
                gc_j = gc[tid]
                beta_j = s_beta[tid]
                gc[tid] = beta_j * _exp2_f32(gc_j * LOG2E)

            # Invert each diagonal block with (I+B)(I+B^2)(I+B^4)(I+B^8).
            br = warp16
            neg_A = _load_fp32_tile(s_A, br, br, wave_copy_A32, lds_copy32, negate=True)
            I_acc = _identity_frag(lane)
            z4 = fx.Float32x4(0.0)
            neg_A_T = _load_fp32_tile(
                s_A,
                br,
                br,
                wave_copy_B32,
                lds_copy32,
                transpose=True,
                negate=True,
            )
            b2 = _mfma16(neg_A, neg_A_T, z4)
            b2t = _mfma16(neg_A_T, neg_A, z4)
            b2_o = _accum_to_bf16x4(b2)
            b2t_o = _accum_to_bf16x4(b2t)
            b4 = _mfma16(b2t_o, b2_o, z4)
            b4t = _mfma16(b2_o, b2t_o, z4)
            b4_o = _accum_to_bf16x4(b4)
            b4t_o = _accum_to_bf16x4(b4t)
            C_acc = _mfma16(b4t_o, b4_o, I_acc)
            C_acc = _mfma16(b4t_o, _accum_to_bf16x4(C_acc), C_acc)
            C_acc = _mfma16(b2t_o, _accum_to_bf16x4(C_acc), C_acc)
            C_acc = _mfma16(neg_A, _accum_to_bf16x4(C_acc), C_acc)
            _store_fp32_tile(s_A, br, br, C_acc, wave_copy_C32, lds_copy32)
            for it in range_constexpr((16 * 16 + WARP_SIZE - 1) // WARP_SIZE):
                idx = lane + it * WARP_SIZE
                rr = idx // 16
                cc = idx % 16
                if rr < cc:
                    s_A[br + rr, br + cc] = zero_f
            gpu.barrier()

            # Merge the diagonal inverses in place.
            sav_L32 = fx.BFloat16x4(0.0)
            sav_L43 = fx.BFloat16x4(0.0)
            sav_L42 = fx.BFloat16x4(0.0)
            if warp == 0:
                sav_L32 = _load_fp32_tile(s_A, 32, 16, wave_copy_A32, lds_copy32)
                sav_L42 = _load_fp32_tile(s_A, 48, 16, wave_copy_A32, lds_copy32)
            if warp < 2:
                sav_L43 = _load_fp32_tile(s_A, 48, 32, wave_copy_A32, lds_copy32)
            # Complete aliased block preloads before sibling stores.
            gpu.barrier()

            kept_c21 = z4
            kept_c32 = z4
            kept_c31 = z4
            if warp == 0:
                t = _mfma16(
                    _load_fp32_tile(s_A, 16, 0, wave_copy_A32, lds_copy32),
                    _load_fp32_tile(s_A, 0, 0, wave_copy_B32, lds_copy32, True),
                    z4,
                )
                kept_c21 = -_mfma16(
                    _load_fp32_tile(s_A, 16, 16, wave_copy_A32, lds_copy32),
                    _accum_to_bf16x4(t),
                    z4,
                )
                _store_fp32_tile(s_A, 16, 0, kept_c21, wave_copy_C32, lds_copy32)
            if warp == 1:
                t = _mfma16(
                    _load_fp32_tile(s_A, 32, 16, wave_copy_A32, lds_copy32),
                    _load_fp32_tile(s_A, 16, 16, wave_copy_B32, lds_copy32, True),
                    z4,
                )
                kept_c32 = -_mfma16(
                    _load_fp32_tile(s_A, 32, 32, wave_copy_A32, lds_copy32),
                    _accum_to_bf16x4(t),
                    z4,
                )
                _store_fp32_tile(s_A, 32, 16, kept_c32, wave_copy_C32, lds_copy32)
            if warp == 2:
                t = _mfma16(
                    _load_fp32_tile(s_A, 48, 32, wave_copy_A32, lds_copy32),
                    _load_fp32_tile(s_A, 32, 32, wave_copy_B32, lds_copy32, True),
                    z4,
                )
                c43 = -_mfma16(
                    _load_fp32_tile(s_A, 48, 48, wave_copy_A32, lds_copy32),
                    _accum_to_bf16x4(t),
                    z4,
                )
                _store_fp32_tile(s_A, 48, 32, c43, wave_copy_C32, lds_copy32)
            gpu.barrier()

            if warp == 0:
                t = _mfma16(
                    _load_fp32_tile(s_A, 32, 0, wave_copy_A32, lds_copy32),
                    _load_fp32_tile(s_A, 0, 0, wave_copy_B32, lds_copy32, True),
                    z4,
                )
                t = _mfma16(sav_L32, _accum_to_bf16x4(kept_c21), t)
                kept_c31 = -_mfma16(
                    _load_fp32_tile(s_A, 32, 32, wave_copy_A32, lds_copy32),
                    _accum_to_bf16x4(t),
                    z4,
                )
                _store_fp32_tile(s_A, 32, 0, kept_c31, wave_copy_C32, lds_copy32)
            if warp == 1:
                t = _mfma16(
                    _load_fp32_tile(s_A, 48, 16, wave_copy_A32, lds_copy32),
                    _load_fp32_tile(s_A, 16, 16, wave_copy_B32, lds_copy32, True),
                    z4,
                )
                t = _mfma16(sav_L43, _accum_to_bf16x4(kept_c32), t)
                c42 = -_mfma16(
                    _load_fp32_tile(s_A, 48, 48, wave_copy_A32, lds_copy32),
                    _accum_to_bf16x4(t),
                    z4,
                )
                _store_fp32_tile(s_A, 48, 16, c42, wave_copy_C32, lds_copy32)
            gpu.barrier()

            if warp == 0:
                t = _mfma16(
                    _load_fp32_tile(s_A, 48, 0, wave_copy_A32, lds_copy32),
                    _load_fp32_tile(s_A, 0, 0, wave_copy_B32, lds_copy32, True),
                    z4,
                )
                t = _mfma16(sav_L42, _accum_to_bf16x4(kept_c21), t)
                t = _mfma16(sav_L43, _accum_to_bf16x4(kept_c31), t)
                c41 = -_mfma16(
                    _load_fp32_tile(s_A, 48, 48, wave_copy_A32, lds_copy32),
                    _accum_to_bf16x4(t),
                    z4,
                )
                _store_fp32_tile(s_A, 48, 0, c41, wave_copy_C32, lds_copy32)
            gpu.barrier()

            SVEC = 8
            SVEC_PER_ROW = BK_SUB // SVEC
            STAGE_ITERS = (BT * SVEC_PER_ROW + BLOCK_THREADS - 1) // BLOCK_THREADS
            assert STAGE_ITERS * BLOCK_THREADS == BT * SVEC_PER_ROW
            wy_copy = fx.make_tiled_copy_tv(
                buf_copy,
                fx.make_layout((32, 8), (8, 1)),
                fx.make_layout((1, SVEC), (1, 1)),
            ).get_slice(tid)
            v_row_stride = H * V
            k_row_stride = Hg * K
            ub_row_stride = V
            wb_row_stride = K
            GVEC = 8
            out_copy = fx.make_tiled_copy_tv(
                buf_copy,
                fx.make_layout((8, 8), (8, 1)),
                fx.make_layout((1, GVEC), (1, 1)),
            ).get_slice(lane)
            out_src = out_copy.partition_S(
                fx.make_view(
                    fx.get_iter(s_out) + warp16 * OUT_S,
                    fx.make_layout((16, BK_SUB), (OUT_S, 1)),
                )
            )
            wy_lds_dst = block_copy_C16.partition_D(s_out)
            reg = _wy_prefetch(
                fx.make_view(
                    fx.get_iter(v_g) + base_v,
                    fx.make_layout((BT, BK_SUB), (v_row_stride, 1)),
                ),
                buf_copy,
                wy_copy,
            )

            # Cache C before reusing its aliased shared storage.
            cached_C = [
                _load_fp32_tile(
                    s_A,
                    warp16,
                    ek * 16,
                    wave_copy_A32,
                    lds_copy32,
                )
                for ek in range(4)
            ]
            gpu.barrier()

            # u_bar = C @ (v*beta)
            for v_it in range_constexpr(N_V_ITERS):
                voff = v_it * BK_SUB
                # Fence the previous s_vT reads before overwriting the tile.
                if const_expr(v_it > 0):
                    gpu.barrier()
                _wy_scatter(
                    reg,
                    s_vT_swz,
                    s_beta,
                    gc,
                    False,
                    tid,
                    STAGE_ITERS,
                    SVEC_PER_ROW,
                    SVEC,
                )
                if v_it + 1 < N_V_ITERS:
                    reg = _wy_prefetch(
                        fx.make_view(
                            fx.get_iter(v_g) + base_v + (v_it + 1) * BK_SUB,
                            fx.make_layout((BT, BK_SUB), (v_row_stride, 1)),
                        ),
                        buf_copy,
                        wy_copy,
                    )
                else:
                    reg = _wy_prefetch(
                        fx.make_view(
                            fx.get_iter(k_g) + base_k,
                            fx.make_layout((BT, BK_SUB), (k_row_stride, 1)),
                        ),
                        buf_copy,
                        wy_copy,
                    )
                gpu.barrier()
                wy = _acc16_n4()
                for ek in range_constexpr(BT // 16):
                    a = cached_C[ek]
                    bs = [
                        _load_mfma_tile_vt_tiled(
                            s_vT,
                            en,
                            ek,
                            lane,
                            wave_tiled_copy_B16,
                            lds_copy64,
                        )
                        for en in range(4)
                    ]
                    _gemm16_n4(wy, a, bs)
                _wy_epilogue_to_lds(wy, wy_lds_dst, lds_copy16)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                out_regs = fx.make_fragment_like(out_src)
                fx.copy(lds_copy128, out_src, out_regs)
                out_dst = out_copy.partition_D(
                    fx.make_view(
                        fx.get_iter(ubar_g) + base_ub + warp16 * ub_row_stride + voff,
                        fx.make_layout((16, BK_SUB), (ub_row_stride, 1)),
                    )
                )
                fx.copy(buf_copy, out_regs, out_dst)

            # w_bar = C @ (k * beta * exp(g))
            for k_it in range_constexpr(N_K_ITERS):
                koff = k_it * BK_SUB
                gpu.barrier()
                _wy_scatter(
                    reg,
                    s_vT_swz,
                    s_beta,
                    gc,
                    True,
                    tid,
                    STAGE_ITERS,
                    SVEC_PER_ROW,
                    SVEC,
                )
                if k_it + 1 < N_K_ITERS:
                    reg = _wy_prefetch(
                        fx.make_view(
                            fx.get_iter(k_g) + base_k + (k_it + 1) * BK_SUB,
                            fx.make_layout((BT, BK_SUB), (k_row_stride, 1)),
                        ),
                        buf_copy,
                        wy_copy,
                    )
                gpu.barrier()
                wy = _acc16_n4()
                for ek in range_constexpr(BT // 16):
                    a = cached_C[ek]
                    bs = [
                        _load_mfma_tile_vt_tiled(
                            s_vT,
                            en,
                            ek,
                            lane,
                            wave_tiled_copy_B16,
                            lds_copy64,
                        )
                        for en in range(4)
                    ]
                    _gemm16_n4(wy, a, bs)
                _wy_epilogue_to_lds(wy, wy_lds_dst, lds_copy16)
                fx.rocdl.s_waitcnt(lgkmcnt=0)
                out_regs = fx.make_fragment_like(out_src)
                fx.copy(lds_copy128, out_src, out_regs)
                out_dst = out_copy.partition_D(
                    fx.make_view(
                        fx.get_iter(wbar_g) + base_wb + warp16 * wb_row_stride + koff,
                        fx.make_layout((16, BK_SUB), (wb_row_stride, 1)),
                    )
                )
                fx.copy(buf_copy, out_regs, out_dst)

    @flyc.jit
    def launch_gdn_prepare(
        k_t: fx.Tensor,
        v_t: fx.Tensor,
        g_t: fx.Tensor,
        beta_t: fx.Tensor,
        cu_t: fx.Tensor,
        wbar_t: fx.Tensor,
        ubar_t: fx.Tensor,
        gcs_t: fx.Tensor,
        T: fx.Int32,
        H: fx.Int32,
        Hg: fx.Int32,
        grid_x: fx.Int32,
        grid_y: fx.Int32,
        stream: fx.Stream,
    ):
        gdn_prepare_kernel(
            k_t, v_t, g_t, beta_t, cu_t, wbar_t, ubar_t, gcs_t, T, H, Hg
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_gdn_prepare

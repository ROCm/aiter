# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from . import dpp_utils
from .mxfp4_gemm_common import (
    kStages,
    kBS_stride_k0_dw,
    _raw,
    _lds_ptr3,
    _lds_base_ptr3,
    _gep3,
    _global_base_ptr1,
    _gep1,
    _global_ptr1,
    _buffer_rsrc,
    _lds_swizzle_mask,
    _fabs_f32,
    _e8m0_roundup,
    _e8m0_from_amax,
    _umax_i32,
    _inline_dpp_quad_amax,
    kmchunks_for,
    lds_acc_bytes_for,
    k_half_for,
    k_tiles_total_for,
    kunroll_for,
    kas_c_k1_for,
    kbs_stride_n0_dw_for,
    kas_per_chunk_dw_for,
    num_n_blocks_for,
    kbs_per_expert_dw_for,
    bq_bytes_for,
    bscale_bytes_for,
)


def n_out_for(inter):
    return 2 * inter


def out_as_per_chunk_dw_for(inter):
    return ((inter // 32) // 4 // 2) * 64


def k_g2_half_for(inter):
    return inter // 2


LOG2E = 1.4426950408889634


def _silu_mul(g, u):
    e = fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E))))
    sig = fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + e)))
    return g * sig * u


def _silu_mul_batch(gs, us):
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _pkmax_u16(a_i32, b_i32):
    _v2i16 = ir.Type.parse("vector<2xi16>")
    va = llvm.BitcastOp(_v2i16, _raw(a_i32)).result
    vb = llvm.BitcastOp(_v2i16, _raw(b_i32)).result
    vm = arith.MaxUIOp(va, vb).result
    out = llvm.BitcastOp(T.i32, vm).result
    return fx.Int32(out)


def _inline_e8m0(amax_u16_i32):
    f32 = fx.Float32(
        _raw((fx.Int32(_raw(amax_u16_i32)) & fx.Int32(0xFFFF)) << fx.Int32(16)).bitcast(
            T.f32
        )
    )
    return _e8m0_roundup(f32)


def gemm1_grid(n_tokens, BM, *, NE, TOPK, INTER, BN=256):
    num_n_blocks = num_n_blocks_for(n_out_for(INTER), BN)
    if BM == 128:
        max_m_blocks = (n_tokens * TOPK + NE * (BM - 1) + BM - 1) // BM
    else:
        active = min(n_tokens * TOPK, NE)
        max_m_blocks = (n_tokens * TOPK + active * (BM - 1) + BM - 1) // BM
    return max_m_blocks * num_n_blocks


@flyc.jit
def _gemm1_body(
    allocator,
    lds_off,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_mind,
    arg_aqout,
    arg_ascaleout,
    arg_hidden,
    bx_i32,
    lane,
    wave,
    use_nt,
    i32_ntok,
    i32_total_m_blocks,
    *,
    BM,
    BN,
    BK,
    KH_TILE,
    kAStages,
    kSubBlocks,
    kMChunks,
    inline_quant=False,
    K,
    K_HALF,
    K_TILES_TOTAL,
    kUnroll,
    kAS_per_chunk_dw,
    kBS_stride_n0_dw,
    kBS_per_expert_dw,
    BQ_BYTES,
    BSCALE_BYTES,
    N_OUT,
    NUM_N_BLOCKS,
    OUT_AS_PER_CHUNK_DW,
    K_G2_HALF,
):
    BN_INT = BN // 2
    b_aux = 2 if use_nt else 0
    M_REPS = BM // 16

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(
        T.i32, llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4)))
    )
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    aq_num_records = arith.index_cast(T.index, _raw(i32_ntok * fx.Int32(K_HALF)))
    aq_rsrc = _buffer_rsrc(arg_aq, aq_num_records)
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = arith.index_cast(T.index, _raw(i32_total_m_blocks)) * fx.Index(
        _asc_per_mb
    )
    ascale_rsrc = _buffer_rsrc(arg_ascale, ascale_num)
    bq_rsrc = _buffer_rsrc(arg_bq, BQ_BYTES)
    bscale_rsrc = _buffer_rsrc(arg_bscale, BSCALE_BYTES)
    hidden_rsrc = None
    if const_expr(inline_quant):
        hidden_num = arith.index_cast(T.index, _raw(i32_ntok * fx.Int32(K * 2)))
        hidden_rsrc = _buffer_rsrc(arg_hidden, hidden_num)

    lds_base = allocator.get_base()
    s_aq = SmemPtr(lds_base, lds_off, T.i8, shape=(kAStages * BM * KH_TILE,))
    s_asc = SmemPtr(
        lds_base,
        lds_off + kAStages * BM * KH_TILE,
        T.i8,
        shape=(kSubBlocks * K_TILES_TOTAL * 256,),
    )
    lds_acc = SmemPtr(
        lds_base, lds_off, T.f32, shape=(BM * BN,)
    )

    cached_actual_row = []
    cached_row_inline = []
    if const_expr(inline_quant):
        rcls = wave * fx.Int32(4) + lane_div_16
        cached_row_inline = [
            llvm.load(T.i32, _global_ptr1(arg_mind, (m_row + rcls) * fx.Int32(4)))
        ]
    else:
        for sub in range_constexpr(kSubBlocks):
            idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
            cached_actual_row.append(
                llvm.load(T.i32, _global_ptr1(arg_mind, idx * fx.Int32(4)))
            )

    b_load_s_base = []
    for j in range_constexpr(4):
        v = (
            e * fx.Int32(N_OUT)
            + n_block_idx * fx.Int32(BN)
            + wave * fx.Int32(BN // 4)
            + fx.Int32(j * 16)
        ) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    mni_base = n_block_idx * fx.Int32(BN // 16 // 2) + wave * fx.Int32(BN // 64 // 2)
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(2):
        base = (
            e * fx.Int32(kBS_per_expert_dw)
            + (mni_base + fx.Int32(mw)) * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    accm = [[None] * 4 for _ in range(kMChunks)]
    b = [[[None, None] for _ in range(4)] for _ in range(kStages)]
    b_scale_v = [[None, None] for _ in range(kStages)]

    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(kSubBlocks):
            lds_row = wave * fx.Int32(BM // 4) + fx.Int32(sub * 8)
            mask = _lds_swizzle_mask(lds_row + lane_div_8)
            voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + cached_actual_row[
                sub
            ] * fx.Int32(K_HALF)
            base_i32 = fx.Int32(
                memref_dialect.extract_aligned_pointer_as_index(s_aq.get())
            )
            off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
            rocdl.raw_ptr_buffer_load_lds(
                aq_rsrc,
                _lds_ptr3(base_i32, off),
                fx.Int32(16),
                voffset,
                fx.Int32(kt * KH_TILE),
                fx.Int32(0),
                fx.Int32(0),
            )

    def issue_a_ds_read(slot):
        mask = _lds_swizzle_mask(lane_mod_16)
        base_ptr = _lds_base_ptr3(s_aq.get())
        a = [[None, None] for _ in range(kMChunks)]
        for k in range_constexpr(2):
            lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + fx.Int32(i * 16)
                off = (
                    fx.Int32(slot * (BM * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                    + lds_col
                )
                a[i][k] = llvm.load(T.vec(4, T.i32), _gep3(base_ptr, off))
        return a

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave * fx.Int32(64) + lane) * fx.Int32(4)
        asc_base = fx.Int32(
            memref_dialect.extract_aligned_pointer_as_index(s_asc.get())
        )
        for sub in range_constexpr(kSubBlocks):
            s_chunk = rocdl.readfirstlane(
                T.i32, (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4)
            )
            lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
            rocdl.raw_ptr_buffer_load_lds(
                ascale_rsrc,
                _lds_ptr3(asc_base, lds_sub + wave * fx.Int32(1024)),
                fx.Int32(16),
                v16,
                s_chunk,
                fx.Int32(0),
                fx.Int32(0),
            )
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                rocdl.raw_ptr_buffer_load_lds(
                    ascale_rsrc,
                    _lds_ptr3(
                        asc_base, lds_sub + fx.Int32(byte_off) + wave * fx.Int32(256)
                    ),
                    fx.Int32(4),
                    v4,
                    s_off,
                    fx.Int32(0),
                    fx.Int32(0),
                )

    def issue_a_scale_ds_read(kt):
        base_ptr = _lds_base_ptr3(s_asc.get())
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = (
                fx.Int32(sub * kAS_per_chunk_dw)
                + fx.Int32(kt * 64)
                + lane_div_16 * fx.Int32(16)
                + lane_mod_16
            )
            out.append(llvm.load(T.i32, _gep3(base_ptr, lds_dw * fx.Int32(4))))
        return out

    lib = lane & fx.Int32(3)
    lane_shr2_and3 = (lane >> fx.Int32(2)) & fx.Int32(3)
    r_in_chunk = wave * fx.Int32(4) + lane_div_16

    def inline_quant_load_kt(B128_IDX, kt, row_token):
        v_voff = (
            row_token * fx.Int32(K * 2)
            + lane_shr2_and3 * fx.Int32(64)
            + lib * fx.Int32(16)
        )
        s_soff = rocdl.readfirstlane(T.i32, fx.Int32(kt * (BK * 2) + B128_IDX * 256))
        frag = buffer_ops.buffer_load(
            hidden_rsrc,
            v_voff // fx.Int32(4),
            vec_width=4,
            dtype=T.i32,
            soffset_bytes=s_soff,
        )
        return Vec(frag)

    def _inline_quant_core(B128_IDX, SUB, slot, kt, h_v, scale_accum):
        h_dw = [fx.Int32(_raw(h_v[j])) for j in range_constexpr(4)]
        hm = [h_dw[j] & fx.Int32(0x7FFF7FFF) for j in range_constexpr(4)]
        m01 = _pkmax_u16(hm[0], hm[1])
        m23 = _pkmax_u16(hm[2], hm[3])
        m0123 = _pkmax_u16(m01, m23)
        lo = m0123 & fx.Int32(0xFFFF)
        hi = m0123.shrui(fx.Int32(16)) & fx.Int32(0xFFFF)
        local_amax = _umax_i32(lo, hi)
        amax_u32 = _inline_dpp_quad_amax(local_amax)
        e8m0 = _inline_e8m0(amax_u32)
        qs = fx.Float32(
            _raw(e8m0 << fx.Int32(23)).bitcast(T.f32)
        )
        pk = _raw(fx.Int32(0))
        qs_raw = _raw(qs)
        for j in range_constexpr(4):
            src_bf16x2 = _raw(
                Vec.from_elements([h_dw[j]], fx.Int32).bitcast(fx.BFloat16)
            )
            pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, src_bf16x2, qs_raw, j)
        pk = fx.Int32(pk)
        r = fx.Int32(SUB * 16) + r_in_chunk
        kb_in_kt = fx.Int32(B128_IDX * 4) + lane_shr2_and3
        mask_r = _lds_swizzle_mask(r)
        b_off = lib * fx.Int32(4)
        aq_base = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(s_aq.get()))
        off = (
            fx.Int32(slot * (BM * KH_TILE))
            + r * fx.Int32(KH_TILE)
            + ((kb_in_kt * fx.Int32(16)) ^ mask_r)
            + b_off
        )
        llvm.StoreOp(_raw(pk), _lds_ptr3(aq_base, off))
        pack_byte = B128_IDX * 2 + SUB
        scale_accum[0] = scale_accum[0] | (e8m0 << fx.Int32(pack_byte * 8))

    def _inline_quant_core_pair(specs, slot, kt, scale_accum):
        n = len(specs)
        h_dw = [
            [fx.Int32(_raw(h_v[j])) for j in range_constexpr(4)]
            for (_b, _s, h_v) in specs
        ]
        la = [None] * n
        for i in range_constexpr(n):
            hm = [h_dw[i][j] & fx.Int32(0x7FFF7FFF) for j in range_constexpr(4)]
            m01 = _pkmax_u16(hm[0], hm[1])
            m23 = _pkmax_u16(hm[2], hm[3])
            m0123 = _pkmax_u16(m01, m23)
            lo = m0123 & fx.Int32(0xFFFF)
            hi = m0123.shrui(fx.Int32(16)) & fx.Int32(0xFFFF)
            la[i] = _umax_i32(lo, hi)
        a = [fx.Int32(_raw(la[i])) for i in range_constexpr(n)]
        s1 = [
            fx.Int32(
                dpp_utils.update_dpp_i32(_raw(a[i]), _raw(a[i]), 0xB1, 0xF, 0xF, True)
            )
            for i in range_constexpr(n)
        ]
        a = [_umax_i32(a[i], s1[i]) for i in range_constexpr(n)]
        s2 = [
            fx.Int32(
                dpp_utils.update_dpp_i32(_raw(a[i]), _raw(a[i]), 0x4E, 0xF, 0xF, True)
            )
            for i in range_constexpr(n)
        ]
        a = [_umax_i32(a[i], s2[i]) for i in range_constexpr(n)]
        e8 = [_inline_e8m0(a[i]) for i in range_constexpr(n)]
        for i in range_constexpr(n):
            B128_IDX, SUB, _hv = specs[i]
            qs_raw = _raw(fx.Float32(_raw(e8[i] << fx.Int32(23)).bitcast(T.f32)))
            pk = _raw(fx.Int32(0))
            for j in range_constexpr(4):
                src_bf16x2 = _raw(
                    Vec.from_elements([h_dw[i][j]], fx.Int32).bitcast(fx.BFloat16)
                )
                pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, src_bf16x2, qs_raw, j)
            pk = fx.Int32(pk)
            r = fx.Int32(SUB * 16) + r_in_chunk
            kb_in_kt = fx.Int32(B128_IDX * 4) + lane_shr2_and3
            mask_r = _lds_swizzle_mask(r)
            b_off = lib * fx.Int32(4)
            aq_base = fx.Int32(
                memref_dialect.extract_aligned_pointer_as_index(s_aq.get())
            )
            off = (
                fx.Int32(slot * (BM * KH_TILE))
                + r * fx.Int32(KH_TILE)
                + ((kb_in_kt * fx.Int32(16)) ^ mask_r)
                + b_off
            )
            llvm.StoreOp(_raw(pk), _lds_ptr3(aq_base, off))
            pack_byte = B128_IDX * 2 + SUB
            scale_accum[0] = scale_accum[0] | (e8[i] << fx.Int32(pack_byte * 8))

    def inline_quant_kt(B128_IDX, SUB, slot, kt, row_token, scale_accum):
        h_v = inline_quant_load_kt(B128_IDX, kt, row_token)
        _inline_quant_core(B128_IDX, SUB, slot, kt, h_v, scale_accum)

    def inline_quant_finish_kt(B128_IDX, SUB, slot, kt, h_v, scale_accum):
        _inline_quant_core(B128_IDX, SUB, slot, kt, h_v, scale_accum)

    def inline_quant_pack_write(kt, scale_accum):
        lane_tgt = lane_shr2_and3 * fx.Int32(16) + r_in_chunk
        asc_base = fx.Int32(
            memref_dialect.extract_aligned_pointer_as_index(s_asc.get())
        )
        off = fx.Int32(kt * 256) + lane_tgt * fx.Int32(4)
        llvm.StoreOp(_raw(scale_accum[0]), _lds_ptr3(asc_base, off))

    def issue_b_load_j(b_slot, K_C, j):
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048)
        )
        for half in range_constexpr(2):
            frag = buffer_ops.buffer_load(
                bq_rsrc,
                (v + fx.Int32(half * 1024)) // fx.Int32(4),
                vec_width=4,
                dtype=T.i32,
                cache_modifier=b_aux,
                soffset_bytes=b_load_s_base[j],
            )
            b_slot[j][half] = Vec(frag)

    def issue_b_scale_load(bs_slot, K_C):
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = (K_C - K_C_HI * 16) * (
            kBS_stride_k0_dw * 4
        )
        for mw in range_constexpr(2):
            s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
            bs_slot[mw] = buffer_ops.buffer_load(
                bscale_rsrc,
                (v + fx.Int32(imm)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=s_off,
            )

    mfma_ty = T.f32x4
    zero4 = Vec.filled(4, 0.0, fx.Float32)

    def mfma_cluster(b_slot, a, a_scale, bs_slot, J, init):
        mni = J // 2
        in_b = J % 2
        sb = bs_slot[mni]
        bJ0, bJ1 = b_slot[J][0], b_slot[J][1]
        if const_expr(kMChunks == 1):
            sa = a_scale[0]
            if const_expr(init):
                accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[0][0], bJ0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                )
            else:
                accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[0][0], bJ0, accm[0][J], 4, 4, 0, sa, 0 + in_b, sb]
                )
            accm[0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                mfma_ty, [a[0][1], bJ1, accm[0][J], 4, 4, 2, sa, 2 + in_b, sb]
            )
        else:
            for sub in range_constexpr(kSubBlocks):
                i0 = sub * 2 + 0
                i1 = sub * 2 + 1
                sa = a_scale[sub]
                if const_expr(init):
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i0][0], bJ0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                    )
                    accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i1][0], bJ0, zero4, 4, 4, 1, sa, 0 + in_b, sb]
                    )
                else:
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i0][0], bJ0, accm[i0][J], 4, 4, 0, sa, 0 + in_b, sb]
                    )
                    accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_ty, [a[i1][0], bJ0, accm[i1][J], 4, 4, 1, sa, 0 + in_b, sb]
                    )
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i0][1], bJ1, accm[i0][J], 4, 4, 2, sa, 2 + in_b, sb]
                )
                accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    mfma_ty, [a[i1][1], bJ1, accm[i1][J], 4, 4, 3, sa, 2 + in_b, sb]
                )

    _relax_prologue = (BM == 128) and not inline_quant
    if const_expr(not inline_quant):
        issue_a_scale_load()
    for K_C in range_constexpr(kStages):
        if const_expr(inline_quant):
            scale_accum = [fx.Int32(0)]
            inline_quant_kt(0, 0, K_C, K_C, cached_row_inline[0], scale_accum)
            issue_b_load_j(b[K_C], K_C, 0)
            issue_b_load_j(b[K_C], K_C, 1)
            inline_quant_kt(1, 0, K_C, K_C, cached_row_inline[0], scale_accum)
            issue_b_load_j(b[K_C], K_C, 2)
            issue_b_load_j(b[K_C], K_C, 3)
            inline_quant_pack_write(K_C, scale_accum)
        else:
            issue_a_load_lds(K_C, K_C)
            if const_expr(not _relax_prologue):
                for j in range_constexpr(4):
                    issue_b_load_j(b[K_C], K_C, j)
        if const_expr(not _relax_prologue):
            issue_b_scale_load(b_scale_v[K_C], K_C)
    if const_expr(_relax_prologue):
        rocdl.sched_barrier(0)
        for K_C in range_constexpr(kStages):
            for j in range_constexpr(4):
                issue_b_load_j(b[K_C], K_C, j)
            issue_b_scale_load(b_scale_v[K_C], K_C)

    for OFFSET in range_constexpr(kUnroll):
        K_C = kStages + OFFSET
        read_slot = OFFSET % kAStages
        write_slot = K_C % kAStages
        slot_b = OFFSET % kStages
        gpu.barrier()
        if const_expr(BM == 128):
            asc_cur = issue_a_scale_ds_read(K_C - kStages)
            a_cur = issue_a_ds_read(read_slot)
        else:
            a_cur = issue_a_ds_read(read_slot)
            asc_cur = issue_a_scale_ds_read(K_C - kStages)
        if const_expr(not inline_quant):
            issue_a_load_lds(write_slot, K_C)
        if const_expr(inline_quant):
            h_v0 = inline_quant_load_kt(0, K_C, cached_row_inline[0])
            h_v1 = inline_quant_load_kt(1, K_C, cached_row_inline[0])
            rocdl.sched_barrier(0)
        for J in range_constexpr(4):
            if const_expr(BM != 128):
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
            mfma_cluster(
                b[slot_b], a_cur, asc_cur, b_scale_v[slot_b], J, init=(OFFSET == 0)
            )
            if const_expr(BM != 128):
                rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            issue_b_load_j(b[slot_b], K_C, J)
            rocdl.sched_barrier(0)
        issue_b_scale_load(b_scale_v[slot_b], K_C)
        if const_expr(inline_quant):
            scale_accum = [fx.Int32(0)]
            _inline_quant_core_pair(
                [(0, 0, h_v0), (1, 0, h_v1)], write_slot, K_C, scale_accum
            )
            inline_quant_pack_write(K_C, scale_accum)

    for S in range_constexpr(kStages):
        kt = K_TILES_TOTAL - kStages + S
        gpu.barrier()
        if const_expr(BM == 128):
            asc_cur = issue_a_scale_ds_read(kt)
            a_cur = issue_a_ds_read(kt % kAStages)
        else:
            a_cur = issue_a_ds_read(kt % kAStages)
            asc_cur = issue_a_scale_ds_read(kt)
        for J in range_constexpr(4):
            mfma_cluster(
                b[kt % kStages], a_cur, asc_cur, b_scale_v[kt % kStages], J, init=False
            )

    gpu.barrier()
    s_aq._view_cache = None
    s_asc._view_cache = None
    lds_acc._view_cache = None

    wave_n = (
        wave
    )
    lds_acc_base = _lds_base_ptr3(lds_acc.get())

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = wave_n * fx.Int32(32) + fx.Int32(J_local * 16) + lane_mod_16
            lds_col = (fx.Int32(128) + col_local) if is_up else col_local
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + lds_col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_acc_base, idx * fx.Int32(4)))

    gpu.barrier()

    tx_i32 = arith.index_cast(T.i32, gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(16)
    n_lane = tx_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    aqout_base = _global_base_ptr1(arg_aqout)
    scales_per_mr = [None] * M_REPS

    for mr in range_constexpr(M_REPS):
        row_local = fx.Int32(mr * 16) + m_lane

        gate_vs = [None] * 8
        up_vs = [None] * 8
        for ee in range_constexpr(8):
            col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
            gate_col = wave_grp * fx.Int32(32) + col_in_grp
            up_col = fx.Int32(128) + gate_col
            gate_off = (row_local * fx.Int32(BN) + gate_col) * fx.Int32(4)
            up_off = (row_local * fx.Int32(BN) + up_col) * fx.Int32(4)
            gate_vs[ee] = fx.Float32(llvm.load(T.f32, _gep3(lds_acc_base, gate_off)))
            up_vs[ee] = fx.Float32(llvm.load(T.f32, _gep3(lds_acc_base, up_off)))
        result = _silu_mul_batch(gate_vs, up_vs)

        local_max = _fabs_f32(result[0])
        for ee in range_constexpr(1, 8):
            local_max = local_max.maximumf(_fabs_f32(result[ee]))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(1), fx.Int32(64)))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(2), fx.Int32(64)))

        e8m0, qscale = _e8m0_from_amax(local_max)
        scales_per_mr[mr] = e8m0

        packed_i32 = _raw(fx.Int32(0))
        qscale_raw = _raw(qscale)
        for w in range_constexpr(4):
            packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32,
                packed_i32,
                _raw(result[2 * w]),
                _raw(result[2 * w + 1]),
                qscale_raw,
                w,
            )
        packed = fx.Int32(packed_i32)

        byte_pos = (
            n_block_idx * fx.Int32(BN_INT // 2)
            + wave_grp * fx.Int32(16)
            + kk * fx.Int32(4)
        )
        out_row = m_row + row_local
        store_off = out_row * fx.Int32(K_G2_HALF) + byte_pos
        llvm.StoreOp(
            _raw(packed),
            _gep1(aqout_base, store_off),
            alignment=4,
            nontemporal=True,
        )

    ascaleout_base = _global_base_ptr1(arg_ascaleout)
    if kk == fx.Int32(0):
        ku = n_block_idx >> fx.Int32(1)
        ikxdl = n_block_idx & fx.Int32(1)
        if const_expr(BM == 16):
            chunk = m_block_idx
            dword_off = (
                chunk * fx.Int32(OUT_AS_PER_CHUNK_DW)
                + ku * fx.Int32(64)
                + wave_grp * fx.Int32(16)
                + m_lane
            )
            addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
            byte_i8 = arith.TruncIOp(T.i8, _raw(scales_per_mr[0])).result
            llvm.StoreOp(byte_i8, _gep1(ascaleout_base, addr), alignment=1)
        else:
            for sub in range_constexpr(kSubBlocks):
                chunk = m_block_idx * fx.Int32(kSubBlocks) + fx.Int32(sub)
                dword_off = (
                    chunk * fx.Int32(OUT_AS_PER_CHUNK_DW)
                    + ku * fx.Int32(64)
                    + wave_grp * fx.Int32(16)
                    + m_lane
                )
                pair_i32 = scales_per_mr[sub * 2 + 0] | (
                    scales_per_mr[sub * 2 + 1] << fx.Int32(8)
                )
                pair_i16 = arith.TruncIOp(T.i16, _raw(pair_i32)).result
                addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
                llvm.StoreOp(
                    pair_i16,
                    _gep1(ascaleout_base, addr),
                    alignment=2,
                )


def _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL):
    kAStages = 2 if BM == 128 else 3
    kSubBlocks = 1 if BM < 32 else BM // 32
    kMChunks = kmchunks_for(BM)
    s_aq_bytes = kAStages * BM * KH_TILE
    s_asc_bytes = kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = lds_acc_bytes_for(BM, BN)
    lds_bytes = max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)
    return kAStages, kSubBlocks, kMChunks, lds_bytes


def compile_gemm1_a4w4_port(
    BM=32,
    use_nt=True,
    inline_quant=False,
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TOPK,
    BN=256,
    BK=256,
):
    print(
        f"[PORT-FLYDSL-GEMM1] compile_gemm1_a4w4_port ENTERED "
        f"BM={BM} use_nt={use_nt} inline_quant={inline_quant} "
        f"D_HIDDEN={D_HIDDEN} D_INTER={D_INTER} NE={NE}",
        flush=True,
    )
    if (BM, use_nt, inline_quant) not in {
        (32, True, False),
        (32, False, False),
        (64, False, False),
        (128, False, False),
        (16, True, True),
    }:
        raise AssertionError(
            f"unsupported gemm1 variant (BM={BM}, use_nt={use_nt}, inline_quant={inline_quant})"
        )

    assert BN == 256 and BK == 256, f"only BN==BK==256 supported, got BN={BN} BK={BK}"
    KH_TILE = BK // 2
    _K = D_HIDDEN
    assert _K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {_K}"
    _INTER = D_INTER
    _N_OUT = n_out_for(_INTER)
    assert (
        _N_OUT % BN == 0
    ), f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {_N_OUT}"
    _NE = NE
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _kUnroll = kunroll_for(_K, BK)
    _kAS_per_chunk_dw = kas_per_chunk_dw_for(_K)
    _kBS_stride_n0_dw = kbs_stride_n0_dw_for(_K)
    _kBS_per_expert_dw = kbs_per_expert_dw_for(_N_OUT, _K)
    _BQ_BYTES = bq_bytes_for(_NE, _N_OUT, _K)
    _BSCALE_BYTES = bscale_bytes_for(_NE, _N_OUT, _K)
    _NUM_N_BLOCKS = num_n_blocks_for(_N_OUT, BN)
    _OUT_AS_PER_CHUNK_DW = out_as_per_chunk_dw_for(_INTER)
    _K_G2_HALF = k_g2_half_for(_INTER)

    kAStages, kSubBlocks, kMChunks, lds_bytes = _bm_constants(
        BM, BN, KH_TILE, _K_TILES_TOTAL
    )

    variant_tag = "iq" if inline_quant else ("nt" if use_nt else "cached")
    name_suffix = f"h{_K}_i{_INTER}_ne{_NE}_bm{BM}_{variant_tag}"

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name=f"gemm1port_smem_{name_suffix}"
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = (
        lds_off + lds_bytes
    )

    @flyc.kernel(name=f"gemm1_a4w4_port_{name_suffix}", known_block_size=[256, 1, 1])
    def gemm1_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = arith.index_cast(T.i32, tx)
        bx_i32 = arith.index_cast(T.i32, bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_NUM_N_BLOCKS)
        if fx.Int32(bx_i32) < bound:
            _gemm1_body(
                allocator,
                lds_off,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_mind,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
                bx_i32,
                lane,
                wave,
                use_nt,
                i32_ntok,
                total_m_blocks,
                BM=BM,
                BN=BN,
                BK=BK,
                KH_TILE=KH_TILE,
                kAStages=kAStages,
                kSubBlocks=kSubBlocks,
                kMChunks=kMChunks,
                inline_quant=inline_quant,
                K=_K,
                K_HALF=_K_HALF,
                K_TILES_TOTAL=_K_TILES_TOTAL,
                kUnroll=_kUnroll,
                kAS_per_chunk_dw=_kAS_per_chunk_dw,
                kBS_stride_n0_dw=_kBS_stride_n0_dw,
                kBS_per_expert_dw=_kBS_per_expert_dw,
                BQ_BYTES=_BQ_BYTES,
                BSCALE_BYTES=_BSCALE_BYTES,
                N_OUT=_N_OUT,
                NUM_N_BLOCKS=_NUM_N_BLOCKS,
                OUT_AS_PER_CHUNK_DW=_OUT_AS_PER_CHUNK_DW,
                K_G2_HALF=_K_G2_HALF,
            )

    @flyc.jit
    def launch_gemm1(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        arg_aqout: fx.Int64,
        arg_ascaleout: fx.Int64,
        arg_hidden: fx.Int64,
        stream: fx.Stream,
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        allocator.finalized = False
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        grid_x = arith.index_cast(T.index, i32_grid)
        gemm1_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            arg_aqout,
            arg_ascaleout,
            arg_hidden,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1

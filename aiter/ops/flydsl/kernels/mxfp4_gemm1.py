# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from . import dpp_utils
from .layout_utils import crd2idx
from .mxfp4_gemm_common import (
    kStages,
    kBS_stride_k0_dw,
    _raw,
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
    kbs_stride_n0_dw_for,
    kas_per_chunk_dw_for,
    num_n_blocks_for,
    kbs_per_expert_dw_for,
    bq_bytes_for,
    bscale_bytes_for,
)


def _udiv(a, c):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(c))


def _umod(a, c):
    return fx.Int32(fx.Uint32(a) % fx.Uint32(c))


def _global_i32_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _global_i32_at(addr_i64, idx):
    # fx.ptr_load/add_offset for a plain scalar read -- no tiling needed, so
    # skip the Tensor/tile/register-fragment machinery fx.copy requires.
    return _global_i32_ptr(addr_i64)[idx]


def _global_i32_load(tiles, idx):
    # Must build atom/types here, not as module globals -- they need an active
    # MLIR trace context.
    atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Int32)
    reg_lay = fx.make_layout(1, 1)
    r = fx.make_rmem_tensor(reg_lay, fx.Int32)
    fx.copy_atom_call(atom, fx.slice(tiles, (None, idx)), r)
    return r.load()[0]


def _global_scalar_tiles(addr_i64, numeric_cls, num_elems):
    ptr_ty = fx.PointerType.get(
        numeric_cls.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=numeric_cls.width // 8,
    )
    ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
    flat = fx.make_view(ptr, fx.make_layout(num_elems, 1))
    return fx.logical_divide(flat, fx.make_layout(1, 1))


def _scalar_store(tiles, idx, value, numeric_cls):
    atom = fx.make_copy_atom(fx.UniversalCopy(numeric_cls.width), numeric_cls)
    reg_lay = fx.make_layout(1, 1)
    r = fx.make_rmem_tensor(reg_lay, numeric_cls)
    r.store(fx.Vector.from_elements([numeric_cls(value)], numeric_cls))
    fx.copy_atom_call(atom, r, fx.slice(tiles, (None, idx)))


def _layout_idx(layout, *coords):
    idx_coords = [fx.Int64(c) for c in coords]
    flat = crd2idx(idx_coords, layout)
    return fx.Int32(flat)


def n_out_for(inter):
    return 2 * inter


LOG2E = 1.4426950408889634


def _silu_mul_batch(gs, us):
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def _pkmax_u16(a_i32, b_i32):
    _v2i16 = T.vec(2, T.i16)
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
    lds_raw_ptr,
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
    inline_quant=False,
    K,
    N_OUT,
    NE,
    interleave=False,
):
    KH_TILE = BK // 2
    K_HALF = k_half_for(K)
    K_TILES_TOTAL = k_tiles_total_for(K, BK)
    kUnroll = kunroll_for(K, BK)
    kAS_per_chunk_dw = kas_per_chunk_dw_for(K)
    kBS_stride_n0_dw = kbs_stride_n0_dw_for(K)
    kBS_per_expert_dw = kbs_per_expert_dw_for(N_OUT, K)
    BQ_BYTES = bq_bytes_for(NE, N_OUT, K)
    BSCALE_BYTES = bscale_bytes_for(NE, N_OUT, K)
    NUM_N_BLOCKS = num_n_blocks_for(N_OUT, BN)
    inter = N_OUT // 2
    OUT_AS_PER_CHUNK_DW = kas_per_chunk_dw_for(inter)
    K_G2_HALF = k_half_for(inter)
    kAStages, kSubBlocks, kMChunks, _ = _bm_constants(BM, BN, KH_TILE, K_TILES_TOTAL)

    BN_INT = BN // 2
    b_aux = 2 if use_nt else 0
    M_REPS = BM // 16

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lane_div_8 = lane // fx.Int32(8)
    lane_mod_8 = lane % fx.Int32(8)

    aq_num_records = fx.Int64(i32_ntok * fx.Int32(K_HALF))
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = fx.Int64(i32_total_m_blocks) * fx.Int64(_asc_per_mb)

    # fx.copy's BufferCopy/BufferCopyLDS atoms take soffset as an element count,
    # not the bytes buffer_ops.buffer_load's soffset_bytes expects.
    def _global_i32_buffer_view(addr_i64, num_bytes):
        # make_layout's dynamic-shape leaf must be i32/i64, not fx.Index.
        num_bytes_i64 = fx.Int64(num_bytes)
        ptr_ty = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        ptr = fx.inttoptr(ptr_ty, fx.Int64(addr_i64))
        view = fx.Tensor(
            fx.make_view(ptr, fx.make_layout(num_bytes_i64 // fx.Int64(4), 1))
        )
        return fx.rocdl.make_buffer_tensor(
            view, max_size=False, num_records_bytes=num_bytes_i64
        )

    def _global_i32_buffer_tiles(addr_i64, num_bytes, tile_elems):
        return fx.logical_divide(
            _global_i32_buffer_view(addr_i64, num_bytes), fx.make_layout(tile_elems, 1)
        )

    bq_tiles = _global_i32_buffer_tiles(arg_bq, BQ_BYTES, 4)
    bq_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(b_aux), fx.Int32)
    bq_reg_lay = fx.make_layout(4, 1)

    bscale_tiles = _global_i32_buffer_tiles(arg_bscale, BSCALE_BYTES, 1)
    bscale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
    bscale_reg_lay = fx.make_layout(1, 1)

    # aq/ascale: global->LDS async DMA (no register fragment), via BufferCopyLDS.
    aq_buf = _global_i32_buffer_view(arg_aq, aq_num_records)
    aq_dma_tiles4 = fx.logical_divide(aq_buf, fx.make_layout(4, 1))
    aq_dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)

    ascale_buf = _global_i32_buffer_view(arg_ascale, ascale_num)
    ascale_dma_tiles4 = fx.logical_divide(ascale_buf, fx.make_layout(4, 1))
    ascale_dma_tiles1 = fx.logical_divide(ascale_buf, fx.make_layout(1, 1))
    ascale_dma_atom16 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), fx.Int32)
    ascale_dma_atom4 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), fx.Int32)

    if const_expr(inline_quant):
        hidden_num = fx.Int64(i32_ntok * fx.Int32(K * 2))
        hidden_tiles = _global_i32_buffer_tiles(arg_hidden, hidden_num, 4)
        hidden_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Int32)
        hidden_reg_lay = fx.make_layout(4, 1)

    # Union LDS region [s_aq | s_asc], reused as lds_acc (f32 accumulator) in
    # the epilogue. s_aq/lds_acc start at lds_raw_ptr; s_asc follows at
    # +kAStages*BM*KH_TILE.

    cached_actual_row = []
    cached_row_inline = None
    if const_expr(inline_quant):
        rcls = wave * fx.Int32(4) + lane_div_16
        cached_row_inline = _global_i32_at(arg_mind, m_row + rcls)
    else:
        for sub in range_constexpr(kSubBlocks):
            idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
            cached_actual_row.append(_global_i32_at(arg_mind, idx))

    # -- b_load_s_base[j] (HIP 412-416), readfirstlane'd uniform per wave ------
    N0_HALF = N_OUT // 32
    b_load_s_base = []
    for j in range_constexpr(4):
        if const_expr(interleave):
            col = (
                n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
            )
        else:
            tile_il = n_block_idx * fx.Int32(16) + wave * fx.Int32(4) + fx.Int32(j)
            g = tile_il & fx.Int32(1)
            n0 = tile_il >> fx.Int32(1)
            col = (g * fx.Int32(N0_HALF) + n0) * fx.Int32(16)
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    # -- b_scale_s_base / _hi (HIP 418-429) -----------------------------------
    if const_expr(interleave):
        mni_base = n_block_idx * fx.Int32(BN // 32) + wave * fx.Int32(BN // 128)
        np_list = [mni_base, mni_base + fx.Int32(1)]
    else:
        np_gate = n_block_idx * fx.Int32(BN // 64) + wave
        np_list = [np_gate, np_gate + fx.Int32(N_OUT // 64)]
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(2):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
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
            off = fx.Int32(slot * (BM * KH_TILE)) + lds_row * fx.Int32(KH_TILE)
            fx.copy(
                aq_dma_atom,
                fx.slice(aq_dma_tiles4, (None, voffset // fx.Int32(16))),
                fx.slice(s_aq_i32x4_tiles, (None, off // fx.Int32(16))),
                soffset=fx.Int32(kt * KH_TILE) // fx.Int32(4),
            )

    # s_aq as flat i32, divided into 4-element (128-bit) and 1-element tiles.
    s_aq_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_raw_ptr),
        fx.make_layout(kAStages * BM * KH_TILE // 4, 1),
    )
    s_aq_i32x4_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(4, 1))
    s_aq_i32x1_tiles = fx.logical_divide(s_aq_i32_flat, fx.make_layout(1, 1))
    i32x4_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    i32x4_reg_lay = fx.make_layout(4, 1)

    def _lds_i32x4_load(tile_idx):
        r = fx.make_rmem_tensor(i32x4_reg_lay, fx.Int32)
        fx.copy_atom_call(
            i32x4_copy_atom, fx.slice(s_aq_i32x4_tiles, (None, tile_idx)), r
        )
        return r.load()

    def issue_a_ds_read(slot):
        mask = _lds_swizzle_mask(lane_mod_16)
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
                a[i][k] = _lds_i32x4_load(off // fx.Int32(16))
        return a

    def issue_a_scale_load():
        chunk_base = m_row // fx.Int32(32)
        v16 = (wave * fx.Int32(64) + lane) * fx.Int32(16)
        v4 = (wave * fx.Int32(64) + lane) * fx.Int32(4)
        for sub in range_constexpr(kSubBlocks):
            s_chunk = rocdl.readfirstlane(
                T.i32, (chunk_base + fx.Int32(sub)) * fx.Int32(kAS_per_chunk_dw * 4)
            )
            lds_sub = fx.Int32(sub * kAS_per_chunk_dw * 4)
            fx.copy(
                ascale_dma_atom16,
                fx.slice(ascale_dma_tiles4, (None, v16 // fx.Int32(16))),
                fx.slice(
                    asc_i32x4_tiles,
                    (None, (lds_sub + wave * fx.Int32(1024)) // fx.Int32(16)),
                ),
                soffset=s_chunk // fx.Int32(4),
            )
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                s_off = rocdl.readfirstlane(T.i32, s_chunk + fx.Int32(byte_off))
                fx.copy(
                    ascale_dma_atom4,
                    fx.slice(ascale_dma_tiles1, (None, v4 // fx.Int32(4))),
                    fx.slice(
                        asc_i32_tiles,
                        (
                            None,
                            (lds_sub + fx.Int32(byte_off) + wave * fx.Int32(256))
                            // fx.Int32(4),
                        ),
                    ),
                    soffset=s_off // fx.Int32(4),
                )

    # s_asc as flat i32.
    s_asc_i32_flat = fx.make_view(
        fx.recast_iter(fx.Int32, fx.add_offset(lds_raw_ptr, kAStages * BM * KH_TILE)),
        fx.make_layout(kSubBlocks * K_TILES_TOTAL * 64, 1),
    )
    asc_i32_tiles = fx.logical_divide(s_asc_i32_flat, fx.make_layout(1, 1))
    asc_i32x4_tiles = fx.logical_divide(s_asc_i32_flat, fx.make_layout(4, 1))

    def issue_a_scale_ds_read(kt):
        out = []
        for sub in range_constexpr(kSubBlocks):
            lds_dw = (
                fx.Int32(sub * kAS_per_chunk_dw)
                + fx.Int32(kt * 64)
                + lane_div_16 * fx.Int32(16)
                + lane_mod_16
            )
            out.append(_raw(_global_i32_load(asc_i32_tiles, lds_dw)))
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
        r = fx.make_rmem_tensor(hidden_reg_lay, fx.Int32)
        fx.copy(
            hidden_copy_atom,
            fx.slice(hidden_tiles, (None, v_voff // fx.Int32(16))),
            r,
            soffset=s_soff // fx.Int32(4),
        )
        return r.load()

    def _inline_quant_core_batch(specs, slot, scale_accum):
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
                    fx.Vector.from_elements([h_dw[i][j]], fx.Int32).bitcast(fx.BFloat16)
                )
                pk = rocdl.cvt_scalef32_pk_fp4_bf16(T.i32, pk, src_bf16x2, qs_raw, j)
            pk = fx.Int32(pk)
            r = fx.Int32(SUB * 16) + r_in_chunk
            kb_in_kt = fx.Int32(B128_IDX * 4) + lane_shr2_and3
            mask_r = _lds_swizzle_mask(r)
            b_off = lib * fx.Int32(4)
            off = (
                fx.Int32(slot * (BM * KH_TILE))
                + r * fx.Int32(KH_TILE)
                + ((kb_in_kt * fx.Int32(16)) ^ mask_r)
                + b_off
            )
            _scalar_store(s_aq_i32x1_tiles, off // fx.Int32(4), pk, fx.Int32)
            pack_byte = B128_IDX * 2 + SUB
            scale_accum = scale_accum | (e8[i] << fx.Int32(pack_byte * 8))
        return scale_accum

    def inline_quant_kt(B128_IDX, SUB, slot, kt, row_token, scale_accum):
        h_v = inline_quant_load_kt(B128_IDX, kt, row_token)
        return _inline_quant_core_batch([(B128_IDX, SUB, h_v)], slot, scale_accum)

    def inline_quant_pack_write(kt, scale_accum):
        lane_tgt = lane_shr2_and3 * fx.Int32(16) + r_in_chunk
        off = fx.Int32(kt * 256) + lane_tgt * fx.Int32(4)
        _scalar_store(asc_i32_tiles, off // fx.Int32(4), scale_accum, fx.Int32)

    def issue_b_load_j(b_slot, K_C, j):
        v = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(K_C * 2048)
        )
        for half in range_constexpr(2):
            tile_idx = (v + fx.Int32(half * 1024)) // fx.Int32(16)
            r = fx.make_rmem_tensor(bq_reg_lay, fx.Int32)
            fx.copy(
                bq_copy_atom,
                fx.slice(bq_tiles, (None, tile_idx)),
                r,
                soffset=b_load_s_base[j] // fx.Int32(4),
            )
            b_slot[j][half] = r.load()

    def issue_b_scale_load(bs_slot, K_C):
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = (K_C - K_C_HI * 16) * (kBS_stride_k0_dw * 4)
        for mw in range_constexpr(2):
            s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
            idx = (v + fx.Int32(imm)) // fx.Int32(4)
            r = fx.make_rmem_tensor(bscale_reg_lay, fx.Int32)
            fx.copy(
                bscale_copy_atom,
                fx.slice(bscale_tiles, (None, idx)),
                r,
                soffset=s_off // fx.Int32(4),
            )
            bs_slot[mw] = r.load()[0]

    mfma_ty = T.f32x4
    zero4 = fx.Vector.filled(4, 0.0, fx.Float32)

    def mfma_cluster(b_slot, a, a_scale, bs_slot, J, init):
        if const_expr(interleave):
            mni = J // 2
            in_b = J % 2
        else:
            mni = J % 2
            in_b = J // 2
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
            scale_accum = fx.Int32(0)
            scale_accum = inline_quant_kt(
                0, 0, K_C, K_C, cached_row_inline, scale_accum
            )
            issue_b_load_j(b[K_C], K_C, 0)
            issue_b_load_j(b[K_C], K_C, 1)
            scale_accum = inline_quant_kt(
                1, 0, K_C, K_C, cached_row_inline, scale_accum
            )
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
            h_v0 = inline_quant_load_kt(0, K_C, cached_row_inline)
            h_v1 = inline_quant_load_kt(1, K_C, cached_row_inline)
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
            scale_accum = _inline_quant_core_batch(
                [(0, 0, h_v0), (1, 0, h_v1)], write_slot, fx.Int32(0)
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

    # lds_acc reuses the s_aq region (offset 0) as an f32 accumulator.
    acc_layout = fx.make_layout((BM, BN), (BN, 1))

    def acc_idx(row, col):
        return _layout_idx(acc_layout, row, col)

    acc_copy_atom = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)
    acc_reg_lay = fx.make_layout(1, 1)
    acc_flat_view = fx.make_view(
        fx.recast_iter(fx.Float32, lds_raw_ptr), fx.make_layout(BM * BN, 1)
    )
    acc_flat_tiles = fx.logical_divide(acc_flat_view, fx.make_layout(1, 1))

    def acc_store(idx, value):
        r = fx.make_rmem_tensor(acc_reg_lay, fx.Float32)
        r.store(fx.Vector.from_elements([fx.Float32(value)], fx.Float32))
        fx.copy_atom_call(acc_copy_atom, r, fx.slice(acc_flat_tiles, (None, idx)))

    def acc_load(idx):
        r = fx.make_rmem_tensor(acc_reg_lay, fx.Float32)
        fx.copy_atom_call(acc_copy_atom, fx.slice(acc_flat_tiles, (None, idx)), r)
        return r.load()[0]

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = wave * fx.Int32(32) + fx.Int32(J_local * 16) + lane_mod_16
            lds_col = (fx.Int32(128) + col_local) if is_up else col_local
            vec = fx.Vector(accm[i][J])
            for v in range_constexpr(4):
                idx = acc_idx(row_base + fx.Int32(v), lds_col)
                acc_store(idx, vec[v])

    gpu.barrier()

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(16)
    n_lane = tx_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)

    aqout_layout = fx.make_layout((BM, K_G2_HALF), (K_G2_HALF, 1))
    # UniversalCopy has no nontemporal/cache-hint knob; dropped (perf-neutral).
    aqout_tiles = _global_scalar_tiles(arg_aqout, fx.Int32, 1 << 24)
    scales_per_mr = [None] * M_REPS

    for mr in range_constexpr(M_REPS):
        row_local = fx.Int32(mr * 16) + m_lane

        gate_vs = [None] * 8
        up_vs = [None] * 8
        for ee in range_constexpr(8):
            col_in_grp = fx.Int32(8) * kk + fx.Int32(ee)
            gate_col = wave_grp * fx.Int32(32) + col_in_grp
            up_col = fx.Int32(128) + gate_col
            gate_vs[ee] = acc_load(acc_idx(row_local, gate_col))
            up_vs[ee] = acc_load(acc_idx(row_local, up_col))
        result = _silu_mul_batch(gate_vs, up_vs)

        local_max = _fabs_f32(result[0])
        for ee in range_constexpr(1, 8):
            local_max = local_max.maximumf(_fabs_f32(result[ee]))
        lm_i = _inline_dpp_quad_amax(fx.Int32(_raw(local_max).bitcast(T.i32)))
        local_max = fx.Float32(_raw(lm_i).bitcast(T.f32))

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
        store_off = _layout_idx(aqout_layout, out_row, byte_pos)
        _scalar_store(aqout_tiles, store_off // fx.Int32(4), packed, fx.Int32)

    # (chunk, ku, wave_grp, m_lane) -> dword index; shape is a placeholder.
    ascaleout_layout = fx.make_layout(
        (1 << 20, 2, 4, 16), (OUT_AS_PER_CHUNK_DW, 64, 16, 1)
    )
    ascaleout_i8_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int8, 1 << 26)
    ascaleout_i16_tiles = _global_scalar_tiles(arg_ascaleout, fx.Int16, 1 << 25)
    if kk == fx.Int32(0):
        ku = n_block_idx >> fx.Int32(1)
        ikxdl = n_block_idx & fx.Int32(1)
        if const_expr(BM == 16):
            chunk = m_block_idx
            dword_off = _layout_idx(ascaleout_layout, chunk, ku, wave_grp, m_lane)
            addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
            _scalar_store(ascaleout_i8_tiles, addr, scales_per_mr[0], fx.Int8)
        else:
            for sub in range_constexpr(kSubBlocks):
                chunk = m_block_idx * fx.Int32(kSubBlocks) + fx.Int32(sub)
                dword_off = _layout_idx(ascaleout_layout, chunk, ku, wave_grp, m_lane)
                pair_i32 = scales_per_mr[sub * 2 + 0] | (
                    scales_per_mr[sub * 2 + 1] << fx.Int32(8)
                )
                addr = dword_off * fx.Int32(4) + ikxdl * fx.Int32(2)
                _scalar_store(
                    ascaleout_i16_tiles, addr // fx.Int32(2), pair_i32, fx.Int16
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
    interleave=False,
    xcd_swizzle=0,
):
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
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _NUM_N_BLOCKS = num_n_blocks_for(_N_OUT, BN)

    _, _, _, lds_bytes = _bm_constants(BM, BN, KH_TILE, _K_TILES_TOTAL)

    variant_tag = "iq" if inline_quant else ("nt" if use_nt else "cached")
    # Tag with H/INTER/NE so different shape specializations get distinct
    # kernel/smem symbols (so KIMI and non-KIMI instances never collide).
    gu_tag = "il" if interleave else "sep"
    name_suffix = f"h{_K}_i{_INTER}_ne{_NE}_bm{BM}_{variant_tag}_{gu_tag}"
    if xcd_swizzle > 0:
        name_suffix += f"_xcd{xcd_swizzle}"

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

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
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(_NUM_N_BLOCKS)

        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd(pid):
            xc = _umod(pid, _NXCD)
            wgid = (
                xc * _xq
                + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                + _udiv(pid, _NXCD)
            )
            _ng = fx.Int32(_SW * _NUM_N_BLOCKS)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.Int32(arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW))))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(_NUM_N_BLOCKS) + n_block

        if fx.Int32(bx_i32) < bound:
            if const_expr(_SW > 0):
                _tile = _xcd(bx_i32)
            else:
                _tile = bx_i32
            _gemm1_body(
                lds_raw_ptr,
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_mind,
                arg_aqout,
                arg_ascaleout,
                arg_hidden,
                _tile,
                lane,
                wave,
                use_nt,
                i32_ntok,
                total_m_blocks,
                BM=BM,
                BN=BN,
                BK=BK,
                inline_quant=inline_quant,
                K=_K,
                N_OUT=_N_OUT,
                NE=_NE,
                interleave=interleave,
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
        grid_x = fx.Int64(i32_grid)
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

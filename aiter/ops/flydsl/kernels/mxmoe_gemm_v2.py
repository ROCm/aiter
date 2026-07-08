# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Layout-API MXFP4 MoE GEMM device bodies (BM32): gemm1 up/gate + gemm2 down."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import _to_raw as _raw
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, T
from flydsl.expr.typing import Vector as Vec

# Compat shim: FlyDSL#753 adds ArithValue.minimumf (swiglu path). Installed
# builds that predate it (have maximumf but not minimumf) get it grafted here so
# the vendored kernel is self-contained and we never patch the installed package.
from flydsl.expr.arith import ArithValue as _ArithValue

if not hasattr(_ArithValue, "minimumf"):

    def _minimumf(self, other):
        """Float minimum (NaN-propagating)."""
        return arith.minimumf(self, _raw(other))

    _ArithValue.minimumf = _minimumf

# shape constants (KIMI defaults; per-shape values come from the compile args)
NE = 385  # #experts
TOPK_DEFAULT = 9
INTER_DEFAULT = 512  # inter_dim: gemm1 D_INTER (output) / gemm2 D_INTER (contraction)
INTER_MAX_DEFAULT = 8192  # compile-time cap for runtime inter_dim (gemm2 B-view / LDS bounds)
HIDDEN_MAX_DEFAULT = 8192  # compile-time cap for runtime model_dim/hidden (gemm1 B-view / A-scale LDS)
MAX_M = 655360
BN = BK = 256
KH_TILE = BK // 2  # 128 packed-fp4 bytes per K-tile
kStages = 2
kBS_stride_k0_dw = 64  # e8m0 scale-layout K-independent stride
LOG2E = 1.4426950408889634


# -- pointer / LDS helpers ----------------------------------------------------
def lds_dma_dst(base_i32, byte_off_i32, elem_ty=None, align=16):
    """LDS dst view for buffer_load_lds DMA (AddressSpace.Shared = LDS enum 2, not addrspace 3)."""
    if elem_ty is None:
        elem_ty = T.i32
    lds_ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    lds_ptr = fx.inttoptr(lds_ptr_ty, fx.Int32(base_i32 + byte_off_i32))
    return fx.make_view(lds_ptr, fx.make_layout(1, 1))


def global_base_ptr1(addr_i64):
    """One ptr<1> base from a raw i64 device address (bare data_ptr() kernarg)."""
    return llvm.inttoptr(ir.Type.parse("!llvm.ptr<1>"), _raw(fx.Int64(addr_i64)))


def gep1(base_ptr, byte_off_i32):
    """getelementptr i8, base_ptr, byte_off_i32  (ptr<1>)."""
    return buffer_ops.get_element_ptr(base_ptr, byte_offset=_raw(byte_off_i32), elem_type=T.i8)


def global_typed_ptr(arg, elem_ty, align=4):
    """Typed global fx.Pointer over a raw i64 device address; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    return fx.inttoptr(ptr_ty, _raw(fx.Int64(arg)))


def lds_typed_ptr(base_i32, elem_ty, align=4):
    """Typed LDS (Shared) fx.Pointer over an i32 LDS base; index in ELEMENTS (ptr[i]), not bytes."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Shared, align)
    return fx.inttoptr(ptr_ty, fx.Int32(base_i32))


def flat_buffer_view(arg, base_elems, elem_ty, *, align, elem_bytes, fold=True, num_records_bytes=None):
    """Flat buffer-tensor view over a RAW i64 addr; fold=True folds wave-uniform base to a VGPR voffset, fold=False keeps per-lane offset + num_records_bytes for OOB-zero."""
    ptr_ty = fx.PointerType.get(elem_ty, fx.AddressSpace.Global, align)
    if fold:
        base = fx.rocdl.readfirstlane(T.i32, _raw(base_elems))
        off_i64 = fx.Int64(arith.ExtUIOp(T.i64, _raw(base)).result)
        base_iter = fx.inttoptr(ptr_ty, fx.Int64(arg) + off_i64 * fx.Int64(elem_bytes))
    else:
        base_iter = fx.inttoptr(ptr_ty, fx.Int64(arg))
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout((1, 1), (1, 1))))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=True)


def lds_dma_atom_128():
    """BufferCopyLDS128b copy-atom (16B global->LDS DMA chunk)."""
    return fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)


def lds_vec_load(base_i32, byte_off_i32, result_type, elem_ty, align=4):
    """Typed LDS ds-read at a BYTE offset from the i32 LDS base; mirrors raw llvm.load (vector or scalar)."""
    elem_ir_ty = elem_ty.ir_type if hasattr(elem_ty, "ir_type") else elem_ty
    ptr = lds_typed_ptr(fx.Int32(base_i32) + byte_off_i32, elem_ir_ty, align=align)
    return fx.ptr_load(ptr, result_type=result_type)


def lds_swizzle_mask(row):
    """lds_swizzle_mask<ROW_BYTES=BK/2=128>(row) = (row & 14) << 3 (fp4 A tile)."""
    return (row & 14) << 3


def lds_swizzle_mask_f8(row):
    """lds_swizzle_mask<ROW_BYTES=256>(row) = (row & 15) << 4 (fp8 A tile)."""
    return (row & 15) << 4


# -- e8m0 / SwiGLU quant math -------------------------------------------------
SWIGLU_ALPHA = 1.702


def silu_mul_batch(gs, us):
    """silu(g)*u via exp2/rcp (matches HIP silu_mul_fast)."""
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(-LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * us[i] for i in range(len(gs))]


def swiglu_mul_batch(gs, us, swiglu_limit=0.0):
    """swiglu(g,u)=g*sigmoid(alpha*g)*(u+1); clamp g<=limit, -limit<=u<=limit (limit 7.0 when swiglu_limit==0)."""
    limit = float(swiglu_limit) if swiglu_limit != 0 else 7.0
    lim = fx.Float32(limit)
    neg_lim = fx.Float32(-limit)
    gs = [g.minimumf(lim) for g in gs]
    us = [u.minimumf(lim).maximumf(neg_lim) for u in us]
    e = [fx.Float32(rocdl.exp2(T.f32, _raw(g * fx.Float32(SWIGLU_ALPHA * -LOG2E)))) for g in gs]
    sig = [fx.Float32(rocdl.rcp(T.f32, _raw(fx.Float32(1.0) + ei))) for ei in e]
    return [gs[i] * sig[i] * (us[i] + fx.Float32(1.0)) for i in range(len(gs))]


def fabs_f32(x):
    """fabsf via sign-bit clear (FlyDSL has no arith.absf)."""
    abs_bits = _raw(x).bitcast(T.i32) & _raw(fx.Int32(0x7FFFFFFF))
    return fx.Float32(abs_bits.bitcast(T.f32))


def e8m0_from_amax(amax_f32, dtype_max=6.0):
    """(e8m0_i32, quant_scale_f32) = ceil_pow2(amax/dtype_max) clamped to 254 (dtype_max: fp4=6, fp8=448)."""
    wi = fx.Int32(_raw(amax_f32 * fx.Float32(1.0 / dtype_max)).bitcast(T.i32))
    bexp = (wi + 0x7FFFFF).shrui(fx.Int32(23)) & 0xFF
    e8m0 = (bexp < 254).select(bexp, fx.Int32(254))
    qscale = fx.Float32(_raw(e8m0 << 23).bitcast(T.f32))
    return e8m0, qscale


# BM per-launch (default 32): bodies derive kMChunks=BM//16 (MFMA row-groups), kSubBlocks=BM//32.
BM = 32
kAStages = 3


# ---- Shared layout-API primitives (B / B-scale data movement + scaled MFMA) ----
def b_copy_atom(nontemporal):
    """BufferCopy128b (4x i32 = one 128b weight chunk). nt rides cache_modifier."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if nontemporal else 0), 32)


def bscale_copy_atom():
    """BufferCopy32b (1x i32 e8m0 scale word); always cached (scales reuse heavily)."""
    return fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), 32)


def bq_view(arg_bq, row_elems, KH4, K_TILES_TOTAL, num_records_bytes=None):
    """Layout view over preshuffled B for one N-row tile; slice -> i32<4:1> (16B=32 fp4). num_records_bytes (has_pad pad-skip) sizes to REAL K; None -> max_size=False byte-identical default."""
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=16)
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * fx.Int64(4))
    # i32 strides: klane[0,4)->64, nlane[0,16)->4, K_tile->512, half[0,2)->256, kpack4->1
    shape = (4, 16, K_TILES_TOTAL, 2, 4)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, (64, 4, 512, 256, 1))))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bscale_view(arg_bscale, base_dw, K_TILES_TOTAL, k0_stride_dw=64, num_records_bytes=None):
    """Layout view over e8m0 B-scale for one n-pack word; slice -> i32<1:1> scale word. num_records_bytes (has_pad pad-skip) sizes to real extent; None -> max_size=False byte-identical default."""
    base_dw = rocdl.readfirstlane(T.i32, _raw(base_dw))
    i32_ptr_ty = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
    off_i64 = fx.Int64(base_dw)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bscale) + off_i64 * fx.Int64(4))
    shape = (4, 16, K_TILES_TOTAL, 1)
    stride = (16, 1, k0_stride_dw, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, stride)))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bq_frag_tmpl(view):
    """i32<4:1> fragment template sliced from a bq_view (16B = 32 fp4)."""
    return view[0, 0, 0, 0, None]


def bscale_frag_tmpl(view):
    """i32<1:1> fragment template sliced from a bscale_view (one e8m0 word)."""
    return view[0, 0, 0, None]


def scale_mma_atoms():
    """16 (opselA,opselB) scaled-MFMA atoms; cbsz/blgp=4 for fp4 from Float4E2M1FN."""
    return {
        (osa, osb): fx.make_mma_atom(fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, Float4E2M1FN, opsel_a=osa, opsel_b=osb))
        for osa in range(4)
        for osb in range(4)
    }


def gemm_mma(atoms, a_frag, b_frag, c_frag, opsel_a, opsel_b, sa, sb):
    """One scaled MFMA via fx.gemm over rank-1 fragments; C accumulates in place."""
    fx.gemm(
        atoms[(opsel_a, opsel_b)],
        c_frag,
        a_frag,
        b_frag,
        c_frag,
        scale_a=sa,
        scale_b=sb,
    )


# ---- Shared A ds-read + per-J MMA cluster (used by both gemm bodies) ----
def issue_a_ds_read_dt(
    s_aq_base, slot, slot_bytes, KH_TILE_A, lane_div_16, lane_mod_16, is_f8, a_vals, a_frags, kMChunks
):
    """A ds-read for one slot: fp4 -> Vec4 i32 into a_frags; fp8 -> Vec8 i32 into a_vals."""
    for k in range_constexpr(2):
        for i in range_constexpr(kMChunks):
            lds_row = lane_mod_16 + i * 16
            row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
            if const_expr(is_f8):
                mask = lds_swizzle_mask_f8(lane_mod_16)
                col0 = lane_div_16 * 16 + k * 128
                col_lo = col0 ^ mask
                col_hi = (col0 + 64) ^ mask
                lo = Vec(lds_vec_load(s_aq_base, row_off + col_lo, Vec.make_type(2, fx.Int64), fx.Int64, align=16))
                hi = Vec(lds_vec_load(s_aq_base, row_off + col_hi, Vec.make_type(2, fx.Int64), fx.Int64, align=16))
                a64 = Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64)
                a_vals[i][k] = _raw(a64.bitcast(fx.Int32))
            else:
                mask = lds_swizzle_mask(lane_mod_16)
                lds_col = (lane_div_16 * 16 + k * 64) ^ mask
                vec = lds_vec_load(s_aq_base, row_off + lds_col, Vec.make_type(4, fx.Int32), fx.Int32, align=16)
                a_frags[i][k].store(Vec(vec))


def mma_one_j(
    J, in_b, sa, sb, bq_frags_kt, is_f8, cbsz_a, a_vals, a_frags, accm, c_frags, atoms, i0=0, single_rg=False, rg_off=0
):
    """One J-cluster (4 scaled MFMAs) for a 32-row A-scale group (row-groups i0, i0+1); fp4 gemm_mma / fp8 raw mfma_scale. sa: 32-row A-scale reg. single_rg (BM16): single 16-row group, rg_off picks its byte, 2 MFMAs."""
    if const_expr(single_rg):
        if const_expr(is_f8):
            bJ0 = Vec(bq_frags_kt[J][0].load())
            bJ1 = Vec(bq_frags_kt[J][1].load())
            for osa_base, k in ((0, 0), (2, 1)):
                bJ = bJ0 if k == 0 else bJ1
                osb = (0 if k == 0 else 2) + in_b
                accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                    T.f32x4, [a_vals[i0][k], bJ, accm[i0][J], cbsz_a, 4, osa_base + rg_off, sa, osb, sb]
                )
        else:
            bJ0, bJ1 = bq_frags_kt[J][0], bq_frags_kt[J][1]
            gemm_mma(atoms, a_frags[i0][0], bJ0, c_frags[i0][J], 0 + rg_off, 0 + in_b, sa, sb)
            gemm_mma(atoms, a_frags[i0][1], bJ1, c_frags[i0][J], 2 + rg_off, 2 + in_b, sa, sb)
        return
    if const_expr(is_f8):
        bJ0 = Vec(bq_frags_kt[J][0].load())
        bJ1 = Vec(bq_frags_kt[J][1].load())
        for osa, k, di in ((0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)):
            i = i0 + di
            bJ = bJ0 if k == 0 else bJ1
            osb = (0 if k == 0 else 2) + in_b
            accm[i][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                T.f32x4, [a_vals[i][k], bJ, accm[i][J], cbsz_a, 4, osa, sa, osb, sb]
            )
    else:
        bJ0, bJ1 = bq_frags_kt[J][0], bq_frags_kt[J][1]
        gemm_mma(atoms, a_frags[i0 + 0][0], bJ0, c_frags[i0 + 0][J], 0, 0 + in_b, sa, sb)
        gemm_mma(atoms, a_frags[i0 + 1][0], bJ0, c_frags[i0 + 1][J], 1, 0 + in_b, sa, sb)
        gemm_mma(atoms, a_frags[i0 + 0][1], bJ1, c_frags[i0 + 0][J], 2, 2 + in_b, sa, sb)
        gemm_mma(atoms, a_frags[i0 + 1][1], bJ1, c_frags[i0 + 1][J], 3, 2 + in_b, sa, sb)


# ---- gemm1 (up/gate-proj) ----
@flyc.jit
def gemm1_body_v2(
    lds_base_i32,
    arg_aq,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_sti,
    arg_aqout,
    arg_ascaleout,
    bx_i32,
    lane,
    wave,
    i32_ntok,
    i32_total_m_blocks,
    i32_inter,
    i32_hidden,
    i32_kpad,
    i32_npad,
    *,
    BM,
    HIDDEN_MAX,
    interleave,
    b_nontemporal,
    a_dtype,
    out_dtype,
    act="silu",
    swiglu_limit=0.0,
    has_pad=False,
    SBM=None,
    k_wave=1,
    BN=BN,
):
    # BN (fused gate|up N-tile) in {64,128,256}: 256 fused SwiGLU tile (split 128); 128/64 = 2x/4x N-blocks (mid/tiny-M).
    if BN not in (64, 128, 256):
        raise AssertionError(f"BN must be in {{64, 128, 256}}, got {BN}")
    # SBM (sort padding unit) >= BM; SBM==BM default byte-identical, SBM>BM packs SBM//BM blocks/sort-block.
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups (BM32: 2, BM64: 4)
    kSubBlocks = BM // 32  # 32-row A-scale chunks / scale-register groups (BM32: 1, BM64: 2)
    # BM16: single 16-row block owns a 32-row scale chunk (chunk==m_block_idx, rg0-only, 2nd half pad).
    is_bm16 = BM < 32
    rg_off = 0  # BM16 always maps its single 16-row group to scale row-group 0
    kScaleSubBlocks = 1 if is_bm16 else kSubBlocks  # 32-row scale chunks to gather/read
    # k_wave (intra-block K-slice): 4 waves as num_n_waves x k_wave (kw=1 byte-identical; kw>1 LDS-reduced).
    NWAVES = 4  # 256-thread block = 4 waves
    num_n_waves = NWAVES // k_wave
    NJ = (BN // num_n_waves) // 16  # J-tiles per N-wave (kw1:4, kw2:8, kw4:16)
    is_f8_a = a_dtype == "fp8"  # fp8 uses raw mfma_scale (cbsz=0); fp4 the fx.gemm path
    is_f8_out = out_dtype == "fp8"  # only the epilogue requant/pack/store differs
    out_max = 448.0 if is_f8_out else 6.0  # e4m3 / e2m1 max
    out_pack = 1 if is_f8_out else 2
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack  # A bytes/K-tile row in LDS (fp8=256, fp4=128)
    cbsz_a = 0 if is_f8_a else 4  # mfma A-format (fp8=0, fp4=4)
    # Contraction K = model_dim/hidden is runtime (i32_hidden); HIDDEN_MAX caps the B-view / A-scale-LDS bounds. kc == K_TILES (BK=256): (K//32)//4//2 == K//256.
    K_rt = fx.Int32(i32_hidden)
    K_BYTES = K_rt // fx.Int32(a_pack)  # a_quant row stride bytes (= K_HALF for fp4, runtime)
    kc_rt = K_rt // fx.Int32(256)  # (K//32)//4//2 == K_TILES
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    KT_PER_KW_RT = K_TILES_RT // fx.Int32(k_wave)  # tiles per K-wave group (kw=1: all tiles)
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)  # hidden-K-derived (runtime)
    K_TILES_MAX = HIDDEN_MAX // BK  # compile-time B-view / A-scale-LDS bound
    INTER_rt = fx.Int32(i32_inter)  # gemm1 N-output dim (runtime)
    N_OUT = INTER_rt * fx.Int32(2)
    kBS_per_expert_dw = (N_OUT // fx.Int32(32)) * kBS_stride_n0_dw  # (N_OUT//16//2)*stride
    NUM_N_BLOCKS = N_OUT // fx.Int32(BN)
    OUT_AS_PER_CHUNK_DW = (INTER_rt // fx.Int32(256)) * fx.Int32(64)  # ((INTER//32)//4//2)*64
    K_G2_BYTES = INTER_rt // fx.Int32(out_pack)  # output row stride (fp4 INTER/2, fp8 INTER)

    # has_pad OOB pad-skip (const_expr-gated): K-skip sizes each 16N B-weight buffer to REAL K (fully-pad halves load 0); N-skip zeros fully-pad-inter tiles. B-scale NOT shrunk.
    bq_num_records = None
    INTER_real = None
    if const_expr(has_pad):
        K_real = K_rt - fx.Int32(i32_kpad)
        halves_real = (K_real + fx.Int32(127)) // fx.Int32(128)
        bq_num_records = halves_real * fx.Int32(1024)
        INTER_real = INTER_rt - fx.Int32(i32_npad)

    # block -> (m_block_idx, n_block_idx); e = sorted_expert_ids[SBM-padded sort block] (SBM==BM: sort_block==m_block_idx).
    n_block_idx = bx_i32 % NUM_N_BLOCKS
    m_block_idx = bx_i32 // NUM_N_BLOCKS
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    if const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_block_idx]))
        m_row = m_block_idx * BM
    else:
        m_row = m_block_idx * BM
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_row // fx.Int32(SBM)]))

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    # k_wave partition of the wave index (kw=1: wave_n=wave, everything below compile-time skipped).
    if const_expr(k_wave > 1):
        wave_n = wave % fx.Int32(num_n_waves)
        wave_k = rocdl.readfirstlane(T.i32, wave // fx.Int32(num_n_waves))
        kw_kt_base = rocdl.readfirstlane(T.i32, wave_k * KT_PER_KW_RT)  # first ABSOLUTE K-tile
    else:
        wave_n = wave
        wave_k = None
        kw_kt_base = None

    def kt_abs_of(kt):
        # ABSOLUTE K-tile for A/B indexing; identity (no emitted op) at kw=1.
        if const_expr(k_wave > 1):
            return fx.Int32(kt) + kw_kt_base
        return kt

    # LDS base offsets (i8): s_aq | s_asc contiguous; lds_acc (f32) unions the region (kw>1: s_asc past all A regions).
    s_aq_base = lds_base_i32
    s_asc_base = lds_base_i32 + fx.Int32(k_wave * kAStages * BM * KH_TILE_A)
    lds_acc_base = lds_base_i32

    # A-gather rows: row = sorted_token_ids & 0xFFFFFF; pad rows are OOB so buffer_load_lds returns 0.
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    # k_wave: each K-wave group's num_n_waves waves (=4/wave_n=wave at kw=1) cooperate on its A-slice.
    rows_per_wave = BM // num_n_waves  # rows each wave gathers (kw1 BM32: 8; kw4 BM16: 16)
    # rows_per_wave < rows_per_call -> round-robin partial-wave scheme (BM16 fp4 kw1); else byte-identical per-wave blocks.
    partial_wave_gather = rows_per_wave < rows_per_call
    if const_expr(partial_wave_gather):
        # BM16 fp4: BM//rows_per_call (=2) calls wrapped round-robin (waves 2,3 re-load, harmless).
        n_gather_calls = BM // rows_per_call  # 2 for BM16 fp4
        gather_base_row = (wave_n % fx.Int32(n_gather_calls)) * rows_per_call
        n_row_groups = 1
    else:
        gather_base_row = wave_n * rows_per_wave
        n_row_groups = rows_per_wave // rows_per_call  # DMA calls/wave
    sti_ptr = global_typed_ptr(arg_sti, T.i32)
    cached_actual_row = []
    for g in range_constexpr(n_row_groups):
        idx = m_row + gather_base_row + g * rows_per_call + a_lane_row
        cached_actual_row.append(sti_ptr[idx] & 0xFFFFFF)

    # B-scale n-pack words (gate/up split by gate mode); NJ//2 words per wave (mni in [0,NJ//2)).
    if const_expr(interleave):
        np_per_wave = (BN // num_n_waves) // 32
        mni_base = n_block_idx * (BN // 32) + wave_n * np_per_wave
        np_list = [mni_base + p for p in range_constexpr(NJ // 2)]
    else:
        if const_expr(k_wave > 1):
            raise AssertionError("k_wave>1 is only supported in interleave gate mode")
        np_gate = n_block_idx * (BN // 64) + wave
        np_list = [np_gate, np_gate + N_OUT // fx.Int32(64)]

    # A-gather global->LDS DMA: per-lane src (no fold); aq_rsrc bounds make OOB padded rows load 0.
    a_gather_atom = lds_dma_atom_128()
    a_gather_src = flat_buffer_view(
        arg_aq,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=i32_ntok * K_BYTES,
    )

    # Per-K-wave A-LDS region (kw=1: s_aq_base_kw == s_aq_base, byte-identical).
    a_stage_bytes = kAStages * BM * KH_TILE_A
    if const_expr(k_wave > 1):
        s_aq_base_kw = s_aq_base + wave_k * fx.Int32(a_stage_bytes)
    else:
        s_aq_base_kw = s_aq_base

    def issue_a_load_lds(slot, kt):
        # lane L -> LDS[base+L*16]; kt is K-wave-LOCAL (A global adds kw_kt_base).
        lane_col = (lane % lanes_per_row) * 16
        base_i32 = s_aq_base_kw
        kt_abs = kt_abs_of(kt)
        for g in range_constexpr(n_row_groups):
            lds_row = gather_base_row + g * rows_per_call
            mask = (
                lds_swizzle_mask_f8(lds_row + a_lane_row)
                if const_expr(is_f8_a)
                else lds_swizzle_mask(lds_row + a_lane_row)
            )
            voffset = (lane_col ^ mask) + cached_actual_row[g] * K_BYTES
            off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
            # split divide: hoist loop-invariant voffset//4 out of K loop (KH_TILE_A//4 const), only kt_abs*32 varies.
            v_e = voffset // 4 + kt_abs * fx.Int32(KH_TILE_A // 4)  # per-lane i32-elem index
            fx.copy(
                a_gather_atom,
                a_gather_src[v_e, None],
                lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16),
            )

    def issue_a_ds_read(slot):
        issue_a_ds_read_dt(
            s_aq_base_kw, slot, BM * KH_TILE_A, KH_TILE_A, lane_div_16, lane_mod_16, is_f8_a, a_vals, a_frags, kMChunks
        )

    asc_dma128 = lds_dma_atom_128()
    asc_dma32 = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)  # 4B A-scale chunk

    def issue_a_scale_load():
        # global->LDS DMA: 16B + 3x4B chunks; per-chunk dword base folded to a VGPR voffset.
        chunk_base = m_block_idx if const_expr(is_bm16) else m_row // 32  # BM16: chunk==m_block_idx
        v16_e = (wave * 64 + lane) * 4  # 16B chunk: per-lane i32-elem
        v4_e = wave * 64 + lane  # 4B chunk: per-lane i32-elem
        asc_base = s_asc_base
        for sub in range_constexpr(kScaleSubBlocks):
            base_dw = (chunk_base + sub) * kAS_per_chunk_dw  # s_chunk/4
            lds_sub = sub * kAS_per_chunk_dw * 4
            src16 = flat_buffer_view(arg_ascale, base_dw, T.i32, align=16, elem_bytes=4)
            fx.copy(
                asc_dma128,
                src16[v16_e, None],
                lds_dma_dst(asc_base, lds_sub + wave * 1024, elem_ty=T.i32, align=16),
            )
            for d in range_constexpr(3):
                byte_off = 4096 + d * 1024
                src4 = flat_buffer_view(arg_ascale, base_dw + byte_off // 4, T.i32, align=16, elem_bytes=4)
                fx.copy(
                    asc_dma32,
                    src4[v4_e, None],
                    lds_dma_dst(asc_base, lds_sub + byte_off + wave * 256, elem_ty=T.i32, align=4),
                )

    def issue_a_scale_ds_read(kt):
        # kt is K-wave-LOCAL; the shared scale chunk holds all K-tiles so read the ABSOLUTE tile.
        asc_ptr = lds_typed_ptr(s_asc_base, T.i32)
        kt_abs = kt_abs_of(kt)
        out = []
        for sub in range_constexpr(kScaleSubBlocks):
            lds_dw = fx.Int32(sub * kAS_per_chunk_dw) + kt_abs * 64 + lane_div_16 * 16 + lane_mod_16
            out.append(asc_ptr[lds_dw])
        return out

    # B load: CK-preshuffle view over bq; base MUST stay wave-uniform (per-lane fold -> WATERFALL ~14x).
    KH4 = K_rt // fx.Int32(8)  # i32 stride for the col axis (= K_HALF//4)
    b_catom = b_copy_atom(b_nontemporal)
    bs_copy_atom = bscale_copy_atom()

    N0_HALF = N_OUT // fx.Int32(32)  # separate-mode gate/up column split

    # B-load view per j-tile (gate mode picks which N-row col maps to; kw=1: wave_n*(BN//4), NJ=4 byte-identical).
    def make_bq_view_for_jtile(j):
        if const_expr(interleave):
            col = n_block_idx * BN + wave_n * (BN // num_n_waves) + j * 16
        else:
            tile_il = n_block_idx * 16 + wave * 4 + j
            col = ((tile_il & 1) * N0_HALF + (tile_il >> 1)) * 16
        nrec = bq_num_records
        if const_expr(has_pad):
            # N-skip: fully-pad LOGICAL-inter tile (>= 16-aligned INTER_real; gate|up pair shares it) -> 0 records.
            gate_span_p = (BN // 2) // num_n_waves
            if const_expr(interleave):
                logical_inter = (
                    n_block_idx * fx.Int32(BN // 2) + wave_n * fx.Int32(gate_span_p) + fx.Int32((j // 2) * 16)
                )
            else:
                logical_inter = col % INTER_rt
            nrec = (logical_inter < INTER_real).select(bq_num_records, fx.Int32(0))
        return bq_view(arg_bq, e * N_OUT + col, KH4, K_TILES_MAX, num_records_bytes=nrec)

    bq_views = [make_bq_view_for_jtile(j) for j in range_constexpr(NJ)]

    # B-scale view per n-pack word; NJ//2 words per wave (2 at kw=1).
    bscale_views = [
        bscale_view(
            arg_bscale,
            e * kBS_per_expert_dw + np_list[mw] * kBS_stride_n0_dw,
            K_TILES_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(NJ // 2)
    ]

    # B fragments: i32<4:1> (16B = 32 fp4). Two fixed sets (CUR consumed, NXT prefetched) double-buffered via scf.for loop-carry (indices stay Python-const 0/1 under the runtime trip).
    frag_tmpl = bq_frag_tmpl(bq_views[0])  # i32<4:1>
    bq_frags = [
        [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(NJ)]
        for _ in range_constexpr(2)
    ]
    # fp4: A in fx.gemm fragments (C in place). fp8: A per-iter Vec8 i32, C raw f32x4.
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    a_vals = a_frags = c_frags = accm = None
    if const_expr(is_f8_a):
        a_vals = [[None, None] for _ in range(kMChunks)]
        accm = [[zero4 for _ in range(NJ)] for _ in range(kMChunks)]
    else:
        a_frags = [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(kMChunks)]
        c_frags = [
            [fx.make_fragment_like(frag_tmpl, dtype=fx.Float32.ir_type) for _ in range_constexpr(NJ)]
            for _ in range_constexpr(kMChunks)
        ]
    # B-scale fragments: i32<1:1>, two fixed sets (CUR/NXT) like bq_frags.
    bs_frag_tmpl = bscale_frag_tmpl(bscale_views[0])  # i32<1:1>
    bs_frags = [[fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(NJ // 2)] for _ in range_constexpr(2)]
    CUR, NXT = 0, 1  # fixed fragment-set indices (compile-time constant)

    def issue_b_load_j(stage, K_C, j):
        # ``K_C`` is the K-wave-LOCAL K-tile; B indexes the ABSOLUTE tile (identity at kw=1).
        view = bq_views[j]
        kc_abs = kt_abs_of(K_C)
        for half in range_constexpr(2):
            fx.copy(
                b_catom,
                view[lane_div_16, lane_mod_16, kc_abs, half, None],
                bq_frags[stage][j][half],
            )

    def issue_b_scale_load(stage, K_C):
        kc_abs = kt_abs_of(K_C)
        for mw in range_constexpr(NJ // 2):
            fx.copy(
                bs_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, kc_abs, None],
                bs_frags[stage][mw],
            )

    # MMA: fp4 via fx.gemm (one per mfma); fp8 via the raw scaled-MFMA intrinsic.
    mma_atoms = scale_mma_atoms() if const_expr(not is_f8_a) else None

    def mfma_cluster(stage, a_scale, J):
        # interleave: mni=J//2 (n0), in_b=J%2 (gate/up); separate: swapped.
        if const_expr(interleave):
            mni, in_b = J // 2, J % 2
        else:
            mni, in_b = J % 2, J // 2
        sb = _raw(Vec(bs_frags[stage][mni].load())[0])
        if const_expr(is_bm16):
            # BM16: single 16-row group; scale byte is row-group (m_block_idx&1) of shared reg a_scale[0].
            mma_one_j(
                J,
                in_b,
                a_scale[0],
                sb,
                bq_frags[stage],
                is_f8_a,
                cbsz_a,
                a_vals,
                a_frags,
                accm,
                c_frags,
                mma_atoms,
                i0=0,
                single_rg=True,
                rg_off=rg_off,
            )
            return
        # One 32-row A-scale group per kSubBlock (reg holds row-groups 2*sub, 2*sub+1).
        for sub in range_constexpr(kSubBlocks):
            mma_one_j(
                J,
                in_b,
                a_scale[sub],
                sb,
                bq_frags[stage],
                is_f8_a,
                cbsz_a,
                a_vals,
                a_frags,
                accm,
                c_frags,
                mma_atoms,
                i0=2 * sub,
            )

    # zero C (fp4 fragments accumulate in place thereafter; fp8 accm pre-init above).
    if const_expr(not is_f8_a):
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(NJ):
                c_frags[i][J].store(zero4)

    # B carry helpers: snapshot set `stage`'s B / B-scale fragment values to SSA (survive scf.for) and restore them into the fragments.
    def b_snapshot(stage):
        vals = [bq_frags[stage][j][half].load() for j in range(NJ) for half in range(2)]
        vals += [bs_frags[stage][mw].load() for mw in range(NJ // 2)]
        return vals

    def b_restore(stage, vals):
        n = 0
        for j in range_constexpr(NJ):
            for half in range_constexpr(2):
                bq_frags[stage][j][half].store(vals[n])
                n += 1
        for mw in range_constexpr(NJ // 2):
            bs_frags[stage][mw].store(vals[n])
            n += 1

    def load_carry():
        if const_expr(is_f8_a):
            return [accm[i][J] for i in range(kMChunks) for J in range(NJ)]
        return [c_frags[i][J].load() for i in range(kMChunks) for J in range(NJ)]

    def store_carry(state):
        n = 0
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(NJ):
                if const_expr(is_f8_a):
                    accm[i][J] = state[n]
                else:
                    c_frags[i][J].store(state[n])
                n += 1

    # prologue: stage the first kStages A-tiles into triple-buffered A LDS, and prefetch tile 0's B / B-scale into the CUR set (one-stage-ahead).
    issue_a_scale_load()
    for K_C in range_constexpr(kStages):
        issue_a_load_lds(K_C, K_C)
    for j in range_constexpr(NJ):
        issue_b_load_j(CUR, fx.Int32(0), j)
    issue_b_scale_load(CUR, fx.Int32(0))

    # One K-tile body; const_slots -> Python-const A-slots/B-buffers (no runtime mod) restore the compile-time schedule.
    def process_tile(kt_rt, read_slot, write_slot, cur_buf, nxt_buf, *, const_slots, guard_stream):
        gpu.barrier()
        issue_a_ds_read(read_slot if const_expr(const_slots) else kt_rt % fx.Int32(kAStages))
        asc_cur = issue_a_scale_ds_read(kt_rt)
        nxt = kt_rt + fx.Int32(kStages)
        # guard_stream: drop the A-stream of tile kt+kStages once it runs past the last tile (tail).
        if const_expr(guard_stream):
            if nxt < KT_PER_KW_RT:
                issue_a_load_lds(write_slot if const_expr(const_slots) else nxt % fx.Int32(kAStages), nxt)
        else:
            issue_a_load_lds(write_slot, nxt)
        kt_p1 = kt_rt + fx.Int32(1)
        nxt_b = fx.Int32((kt_p1 < KT_PER_KW_RT).select(kt_p1, KT_PER_KW_RT - fx.Int32(1)))
        for J in range_constexpr(NJ):
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(cur_buf, asc_cur, J)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            issue_b_load_j(nxt_buf, nxt_b, J)
            rocdl.sched_barrier(0)
        issue_b_scale_load(nxt_buf, nxt_b)

    # Fixed-factor unroll UNROLL=LCM(kAStages,2): A-slots const + B parity back to CUR per group; trip stays runtime.
    UNROLL = kAStages * 2 if (kAStages % 2) else kAStages  # 6 for kAStages=3
    nC = kMChunks * NJ
    n_full = rocdl.readfirstlane(T.i32, (KT_PER_KW_RT // fx.Int32(UNROLL)))
    full_tiles = n_full * fx.Int32(UNROLL)

    # Bulk: unrolled groups of UNROLL tiles (all slots/buffers Python-const). Carry C/accm + CUR B.
    for grp_iv, state in range(fx.Index(0), fx.Index(n_full), fx.Index(1), init=load_carry() + b_snapshot(CUR)):
        store_carry(state[:nC])
        b_restore(CUR, state[nC:])
        grp_base = fx.Int32(grp_iv) * fx.Int32(UNROLL)
        for u in range_constexpr(UNROLL):
            kt_rt = grp_base + fx.Int32(u)
            read_slot = u % kAStages
            write_slot = (u + kStages) % kAStages
            cur_buf = u % 2
            nxt_buf = (u + 1) % 2
            # bulk tiles are always in-range (remainder owns the last UNROLL tiles) -> no stream guard.
            process_tile(kt_rt, read_slot, write_slot, cur_buf, nxt_buf, const_slots=True, guard_stream=False)
        results = yield load_carry() + b_snapshot(CUR)  # UNROLL even -> next tile's B back in CUR

    # Remainder: the last (KT % UNROLL) tiles.
    rem = KT_PER_KW_RT - full_tiles
    if const_expr(not is_f8_a):
        # fp4: C accumulates in place (survives scf.if) -> unroll the tail, per-tile guard (i < rem), slots const.
        store_carry(results[:nC])
        b_restore(CUR, results[nC:])
        for i in range_constexpr(UNROLL - 1):
            if fx.Int32(i) < rem:
                process_tile(
                    full_tiles + fx.Int32(i),
                    i % kAStages,
                    (i + kStages) % kAStages,
                    i % 2,
                    (i + 1) % 2,
                    const_slots=True,
                    guard_stream=True,
                )
    else:
        # fp8: accm raw f32x4 SSA (cannot escape scf.if) so tail stays a rolled scf.for; carry slots as +1-with-wrap loop values to avoid the per-iter %kAStages divide.
        rs0 = fx.Int32(0)
        ws0 = fx.Int32(kStages % kAStages)
        nSlots = len(results)
        for kt_iv, state in range(fx.Index(0), fx.Index(rem), fx.Index(1), init=results + [rs0, ws0]):
            store_carry(state[:nC])
            b_restore(CUR, state[nC:nSlots])
            read_slot = state[nSlots]
            write_slot = state[nSlots + 1]
            process_tile(
                full_tiles + fx.Int32(kt_iv), read_slot, write_slot, CUR, NXT, const_slots=True, guard_stream=True
            )
            rs1 = read_slot + fx.Int32(1)
            ws1 = write_slot + fx.Int32(1)
            rs_next = (rs1 == fx.Int32(kAStages)).select(fx.Int32(0), rs1)
            ws_next = (ws1 == fx.Int32(kAStages)).select(fx.Int32(0), ws1)
            results = yield load_carry() + b_snapshot(NXT) + [rs_next, ws_next]
        store_carry(results[:nC])

    gpu.barrier()

    # epilog: cshuffle -> (kw>1 LDS reduce into slab 0) -> SwiGLU -> fp4 + e8m0 requant (kw=1 byte-identical).
    slab_elems = BM * BN  # f32 elems per cshuffle slab
    lds_acc_fptr = lds_typed_ptr(lds_acc_base, T.f32)

    # accumulators: fp4 from C fragments, fp8 from accm. (NJ tiles per N-wave.)
    if const_expr(is_f8_a):
        acc_vecs = [[Vec(accm[i][J]) for J in range(NJ)] for i in range(kMChunks)]
    else:
        acc_vecs = [[Vec(c_frags[i][J].load()) for J in range(NJ)] for i in range(kMChunks)]

    def acc(i, J, v):
        return acc_vecs[i][J][v]

    # cshuffle: J//2 = 16-col n0 tile, J%2 = gate(0)/up(1); gate [0,BN//2), up [BN//2,BN) (kw=1: wave_n*32).
    gate_span = (BN // 2) // num_n_waves  # gate cols this N-wave covers
    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * 4
        for J in range_constexpr(NJ):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = wave_n * gate_span + J_local * 16 + lane_mod_16
            lds_col = ((BN // 2) + col_local) if is_up else col_local
            for v in range_constexpr(4):
                idx = (row_base + v) * BN + lds_col
                if const_expr(k_wave > 1):
                    idx = idx + wave_k * fx.Int32(slab_elems)  # this K-wave's partial slab
                lds_acc_fptr[idx] = fx.Float32(acc(i, J, v))

    gpu.barrier()

    # k_wave reduce: 256 threads cooperatively sum the k_wave partial slabs into slab 0.
    if const_expr(k_wave > 1):
        tid_red = lane + wave * fx.Int32(64)  # 0..255
        per_thread = slab_elems // 256
        for e in range_constexpr(per_thread):
            eidx = tid_red + fx.Int32(e * 256)
            s = fx.Float32(lds_acc_fptr[eidx])
            for g in range_constexpr(1, k_wave):
                s = s + fx.Float32(lds_acc_fptr[fx.Int32(g * slab_elems) + eidx])
            lds_acc_fptr[eidx] = s
        gpu.barrier()

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // 16
    n_lane = tx_i32 % 16
    wave_grp = n_lane // 4
    kk = n_lane % 4

    # Requant partition: thread covers gate col-group n_lane (< BN//16 valid); BN<256 predicates the stores (BN=256 byte-identical).
    N_COL_GROUPS = BN // 16  # gate col-groups (BN=256:16, BN=64:4)

    # Output store via fx.copy (BufferCopy32b nt) over an i32 view; wave-uniform row base in view base.
    out_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(2), fx.Int32)  # nt i32 store
    out_reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32)
    aqout_view = flat_buffer_view(arg_aqout, m_row * (K_G2_BYTES // 4), T.i32, align=4, elem_bytes=4)
    scales_per_mr = [None] * kMChunks

    for mr in range_constexpr(kMChunks):
        row_local = fx.Int32(mr * 16) + m_lane
        gate_vs = [None] * 8
        up_vs = [None] * 8
        for ee in range_constexpr(8):
            col_in_grp = 8 * kk + ee
            if const_expr(BN == 256):
                gate_col = wave_grp * 32 + col_in_grp  # == n_lane*8 + ee (byte-identical literal)
                up_col = 128 + gate_col
            else:
                gate_col = n_lane * 8 + ee
                up_col = fx.Int32(BN // 2) + gate_col
            gate_idx = row_local * BN + gate_col
            up_idx = row_local * BN + up_col
            gate_vs[ee] = fx.Float32(lds_acc_fptr[gate_idx])
            up_vs[ee] = fx.Float32(lds_acc_fptr[up_idx])
        if const_expr(act == "swiglu"):
            result = swiglu_mul_batch(gate_vs, up_vs, swiglu_limit)
        else:
            result = silu_mul_batch(gate_vs, up_vs)

        local_max = fabs_f32(result[0])
        for ee in range_constexpr(1, 8):
            local_max = local_max.maximumf(fabs_f32(result[ee]))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(1), fx.Int32(64)))
        local_max = local_max.maximumf(local_max.shuffle_xor(fx.Int32(2), fx.Int32(64)))

        e8m0, qscale = e8m0_from_amax(local_max, out_max)
        scales_per_mr[mr] = e8m0

        qscale_raw = _raw(qscale)
        # byte position of this lane's 8 elems (linear INTER tiling; BN=256 keeps the literal form).
        if const_expr(BN == 256):
            byte_pos_fp4 = n_block_idx * (BN // 4) + wave_grp * 16 + kk * 4
        else:
            byte_pos_fp4 = n_block_idx * (BN // 4) + n_lane * 4
        if const_expr(is_f8_out):
            # 8 f32 -> 8 fp8: lo = elems 0..3, hi = 4..7 (2 fp8 per cvt half).
            v2i16 = T.vec(2, T.i16)
            lo = _raw(Vec.filled(2, 0, fx.Int16))
            lo = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, lo, _raw(result[0]), _raw(result[1]), qscale_raw, 0)
            lo = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, lo, _raw(result[2]), _raw(result[3]), qscale_raw, 1)
            hi = _raw(Vec.filled(2, 0, fx.Int16))
            hi = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, hi, _raw(result[4]), _raw(result[5]), qscale_raw, 0)
            hi = rocdl.cvt_scalef32_pk_fp8_f32(v2i16, hi, _raw(result[6]), _raw(result[7]), qscale_raw, 1)
            # lo at off, hi at off+1 (each vec2xi16 = one i32).
            elem_off = row_local * (K_G2_BYTES // 4) + (byte_pos_fp4 // 2)
            lo_i32 = Vec(lo).bitcast(fx.Int32)
            hi_i32 = Vec(hi).bitcast(fx.Int32)
            if const_expr(BN == 256):
                fx.memref_store_vec(Vec.filled(1, lo_i32[0], fx.Int32), out_reg)
                fx.copy(out_copy_atom, out_reg, aqout_view[elem_off, None])
                fx.memref_store_vec(Vec.filled(1, hi_i32[0], fx.Int32), out_reg)
                fx.copy(out_copy_atom, out_reg, aqout_view[elem_off + 1, None])
            else:
                # BN<256: only n_lane<N_COL_GROUPS threads hold valid gate cols (the rest read
                # out-of-slab); predicate so shrunk-tile threads don't scatter into neighbours.
                if n_lane < fx.Int32(N_COL_GROUPS):
                    fx.memref_store_vec(Vec.filled(1, lo_i32[0], fx.Int32), out_reg)
                    fx.copy(out_copy_atom, out_reg, aqout_view[elem_off, None])
                    fx.memref_store_vec(Vec.filled(1, hi_i32[0], fx.Int32), out_reg)
                    fx.copy(out_copy_atom, out_reg, aqout_view[elem_off + 1, None])
        else:
            packed_i32 = _raw(fx.Int32(0))
            for w in range_constexpr(4):
                packed_i32 = rocdl.cvt_scalef32_pk_fp4_f32(
                    T.i32,
                    packed_i32,
                    _raw(result[2 * w]),
                    _raw(result[2 * w + 1]),
                    qscale_raw,
                    w,
                )
            elem_off = row_local * (K_G2_BYTES // 4) + (byte_pos_fp4 // 4)
            fx.memref_store_vec(Vec.filled(1, fx.Int32(packed_i32), fx.Int32), out_reg)
            if const_expr(BN == 256):
                fx.copy(out_copy_atom, out_reg, aqout_view[elem_off, None])
            else:
                # BN<256: predicate the store so shrunk-tile threads don't write a neighbouring n_block.
                if n_lane < fx.Int32(N_COL_GROUPS):
                    fx.copy(out_copy_atom, out_reg, aqout_view[elem_off, None])

    # ascaleout store via fx.copy (BufferCopy16b) over an i16 view; wave-uniform byte base in view base.
    asc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), fx.Int16)
    asc_reg = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int16)
    # ascaleout layout keys on ABSOLUTE scale group g = n_block_idx*(BN//64)+wave_grp -> ku=g//8, ikxdl=(g>>2)&1, lane-grp=g&3 (BN-independent; BN=64: only wave_grp==0 stores).
    if const_expr(BN == 256):
        store_scale = kk == 0
    else:
        # BN<256: each 64-col scale group is one wave_grp; store from every valid wave_grp
        # (BN//64 of them: BN64->1 {0}, BN128->2 {0,1}).
        if const_expr(BN == 64):
            store_scale = (kk == 0) and (wave_grp == fx.Int32(0))
        else:
            store_scale = (kk == 0) and (wave_grp < fx.Int32(BN // 64))
    if store_scale:
        if const_expr(BN == 256):
            ku = n_block_idx >> 1
            ikxdl = n_block_idx & 1
            lane_grp = wave_grp
        elif const_expr(BN == 64):
            g = n_block_idx  # wave_grp==0 here; (BN//64)==1 so g == n_block_idx
            ku = g >> 3
            ikxdl = (g >> 2) & 1
            lane_grp = g & 3
        else:
            g = n_block_idx * fx.Int32(BN // 64) + wave_grp  # BN128: g spans 2 scale groups
            ku = g >> 3
            ikxdl = (g >> 2) & 1
            lane_grp = g & 3
        if const_expr(is_bm16):
            # BM16: block owns chunk==m_block_idx, fills only rg0 (16-row scale -> byte0, byte1 pad 0).
            chunk = m_block_idx
            base_i16 = (chunk * OUT_AS_PER_CHUNK_DW + ku * 64) * 2 + ikxdl
            asc_view = flat_buffer_view(arg_ascaleout, base_i16, T.i16, align=2, elem_bytes=2)
            pair_i16 = fx.Int16(scales_per_mr[0])
            asc_off = (lane_grp * 16 + m_lane) * 2
            fx.memref_store_vec(Vec.filled(1, pair_i16, fx.Int16), asc_reg)
            fx.copy(asc_copy_atom, asc_reg, asc_view[asc_off, None])
        else:
            for sub in range_constexpr(kSubBlocks):
                chunk = m_block_idx * kSubBlocks + sub
                # uniform i16 base = (chunk*OUT_AS_PER_CHUNK_DW + ku*64)*2 + ikxdl
                base_i16 = (chunk * OUT_AS_PER_CHUNK_DW + ku * 64) * 2 + ikxdl
                asc_view = flat_buffer_view(arg_ascaleout, base_i16, T.i16, align=2, elem_bytes=2)
                pair_i32 = scales_per_mr[sub * 2 + 0] | (scales_per_mr[sub * 2 + 1] << 8)
                pair_i16 = fx.Int16(pair_i32)
                # per-lane i16 offset = (lane_grp*16 + m_lane)*2
                asc_off = (lane_grp * 16 + m_lane) * 2
                fx.memref_store_vec(Vec.filled(1, pair_i16, fx.Int16), asc_reg)
                fx.copy(asc_copy_atom, asc_reg, asc_view[asc_off, None])


def lds_bytes_for(K_TILES_TOTAL, KH_TILE_A=KH_TILE, BM=BM, k_wave=1, BN=BN):
    # A staging (k_wave per-K-wave regions) + shared scale chunk, unioned with the k_wave cshuffle slabs.
    kScaleSubBlocks = 1 if BM < 32 else BM // 32
    s_aq_bytes = k_wave * kAStages * BM * KH_TILE_A
    s_asc_bytes = kScaleSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = k_wave * BM * BN * 4
    return max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)


# ---- gemm2 (down-proj) ----
def issue_a_load_lds_dt(
    arg_aq, aq_num_records, s_aq_base, slot, kt, m_row, wave, lane, is_f8, KH_TILE_A, K_BYTES, BM=BM
):
    """A->LDS DMA for one K-tile; gemm2 A is the already-sorted row, OOB-zero via aq_rsrc bounds."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // 4  # rows each wave loads (BM32: 8, BM64: 16)
    # BM16 fp4: partial-wave round-robin (waves 2,3 re-load, harmless); BM>=32 byte-identical per-wave blocks.
    partial_wave_gather = rows_per_wave < rows_per_call
    if const_expr(partial_wave_gather):
        n_gather_calls = BM // rows_per_call
        gather_base_row = (wave % fx.Int32(n_gather_calls)) * rows_per_call
        n_row_groups = 1
    else:
        gather_base_row = wave * rows_per_wave
        n_row_groups = rows_per_wave // rows_per_call
    lane_col = (lane % lanes_per_row) * 16
    base_i32 = s_aq_base
    atom = lds_dma_atom_128()
    src = flat_buffer_view(arg_aq, None, T.i32, align=16, elem_bytes=4, fold=False, num_records_bytes=aq_num_records)
    for g in range_constexpr(n_row_groups):
        lds_row = gather_base_row + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row) if const_expr(is_f8) else lds_swizzle_mask(lds_row + a_lane_row)
        )
        car = m_row + lds_row + a_lane_row  # direct sorted row
        voffset = (lane_col ^ mask) + car * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        v_e = (voffset + kt * KH_TILE_A) // 4  # per-lane i32-elem index
        fx.copy(atom, src[v_e, None], lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16))


@flyc.jit
def gemm2_body_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    i32_M,
    i32_max_m_blocks,
    arg_out,
    bx_i32,
    lane,
    wave,
    aq_rsrc,
    arg_aq,
    i32_inter,
    i32_hidden,
    i32_kpad,
    i32_npad,
    *,
    BM,
    use_nt,
    INTER_MAX,
    aStages,
    a_dtype,
    use_reduce=False,
    topk=1,
    has_pad=False,
    SBM=None,
    g2_kstages=2,
    g2_bhoist=True,
    g2_ascale_pf=True,
    g2_bf16_lds=False,
):
    # gemm2 K-loop perf knobs (default ON, no-op unless g2_kstages==2): kstages=2 double-buffers B weight+scale one tile ahead; bhoist issues that prefetch above the LDS barrier; ascale_pf prefetches A-scale one tile ahead.
    if g2_kstages not in (1, 2):
        raise AssertionError(f"g2_kstages must be 1 or 2, got {g2_kstages}")
    # SBM (sort padding unit) >= BM (compute tile); SBM==BM default byte-identical (see gemm1_body_v2).
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups (BM32: 2, BM64: 4)
    kSubBlocks = BM // 32  # 32-row A-scale chunks / scale-register groups (BM32: 1, BM64: 2)
    # BM16: single 16-row block owning a 32-row scale chunk (chunk==m_block_idx, rg0-only); mirrors gemm1.
    is_bm16 = BM < 32
    rg_off = 0
    kScaleSubBlocks = 1 if is_bm16 else kSubBlocks
    is_f8_a = a_dtype == "fp8"  # only the A path differs
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    cbsz_a = 0 if is_f8_a else 4
    # Contraction K = inter_dim runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes (runtime)
    kc_rt = K_rt // fx.Int32(256)  # (K//32)//4//2
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = (N_OUT_rt // fx.Int32(32)) * kBS_stride_n0_dw  # (N_OUT//16//2)*stride
    num_n_blocks = N_OUT_rt // fx.Int32(256)
    KH4 = K_rt // fx.Int32(8)  # i32 col stride (= K_HALF//4)
    K_TILES_MAX = INTER_MAX // BK

    # has_pad OOB pad-skip (const_expr-gated, as gemm1): K-skip sizes 16N B-weight buffer to REAL K; N-skip zeros fully-pad-N w2 tiles (col >= N_real=N_OUT-npad; PERF-ONLY). B-scale NOT shrunk.
    bq_num_records = None
    N_real = None
    if const_expr(has_pad):
        K_real = K_rt - fx.Int32(i32_kpad)
        halves_real = (K_real + fx.Int32(127)) // fx.Int32(128)
        bq_num_records = halves_real * fx.Int32(1024)
        N_real = N_OUT_rt - fx.Int32(i32_npad)

    # block -> (m_block_idx, n_block_idx); e = sorted_expert_ids[SBM-padded sort block] (SBM==BM: sort_block==m_block_idx).
    m_block_idx = bx_i32 // num_n_blocks
    n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    if const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_block_idx]))
        m_row = m_block_idx * BM
    else:
        m_row = m_block_idx * BM
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_row // fx.Int32(SBM)]))

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    # A-scale buffer resource + uniform base (raw load); chunk = m_block_idx (BM16) else m_row//32.
    asc_per_mb = fx.Int32(kScaleSubBlocks) * kAS_per_chunk_dw * fx.Int32(4)
    asc_num = fx.Index(i32_max_m_blocks) * fx.Index(asc_per_mb)
    ascale_rsrc = buffer_ops.create_buffer_resource_from_addr(_raw(fx.Int64(arg_ascale)), num_records_bytes=asc_num)
    scale_chunk0 = m_block_idx if const_expr(is_bm16) else m_row // 32
    a_scale_s_base = rocdl.readfirstlane(T.i32, scale_chunk0 * kAS_per_chunk_dw * fx.Int32(4))
    v_voff_scale = ((lane_div_16 * 16) + lane_mod_16) * 4

    def load_a_scale_tile(kt):
        # One i32 A-scale register per 32-row chunk (kScaleSubBlocks); chunk sub at sub*kAS_per_chunk_dw dwords.
        out = []
        for sub in range_constexpr(kScaleSubBlocks):
            out.append(
                buffer_ops.buffer_load(
                    ascale_rsrc,
                    (v_voff_scale + kt * 256) // 4 + sub * kAS_per_chunk_dw,
                    vec_width=1,
                    dtype=T.i32,
                    soffset_bytes=a_scale_s_base,
                )
            )
        return out

    s_aq_base = lds_base_i32
    lds_acc_base = lds_base_i32  # f32 acc unions the A-tile region

    # -- B / B-scale layout-API views (shared primitives) ---------------------
    b_catom = b_copy_atom(use_nt)
    bs_copy_atom = bscale_copy_atom()

    def make_bq_view(j):
        col = n_block_idx * BN + wave * (BN // 4) + j * 16
        nrec = bq_num_records
        if const_expr(has_pad):
            # N-skip: fully-pad-N tile (col >= 16-aligned N_real) -> 0 records so weight loads OOB -> 0.
            nrec = (col < N_real).select(bq_num_records, fx.Int32(0))
        return bq_view(arg_bq, e * N_OUT_rt + col, KH4, K_TILES_MAX, num_records_bytes=nrec)

    bq_views = [make_bq_view(j) for j in range_constexpr(4)]

    mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // 64 // 2)
    bscale_views = [
        bscale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_TILES_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(2)
    ]

    # B / B-scale fragments are streamed PER-ITER (one K-tile worth); A refilled per K via LDS.
    frag_tmpl = bq_frag_tmpl(bq_views[0])  # i32<4:1>
    bs_frag_tmpl = bscale_frag_tmpl(bscale_views[0])  # i32<1:1>
    # fp4: A in fx.gemm fragments. fp8: A a per-iter Vec8 i32, C a raw f32x4 (accm).
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    a_vals = a_frags = c_frags = accm = None
    if const_expr(is_f8_a):
        a_vals = [[None, None] for _ in range(kMChunks)]
        accm = [[zero4 for _ in range(4)] for _ in range(kMChunks)]
    else:
        a_frags = [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(kMChunks)]
        c_frags = [
            [fx.make_fragment_like(frag_tmpl, dtype=fx.Float32.ir_type) for _ in range_constexpr(4)]
            for _ in range_constexpr(kMChunks)
        ]

    def issue_b_load_into(bqf, bsf, kt_rt):
        # Issue B-weight + B-scale vmem loads for K-tile kt_rt into the given (per-stage) fragments.
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                fx.copy(b_catom, bq_views[j][lane_div_16, lane_mod_16, kt_rt, half, None], bqf[j][half])
        for mw in range_constexpr(2):
            fx.copy(bs_copy_atom, bscale_views[mw][lane_div_16, lane_mod_16, kt_rt, None], bsf[mw])

    def stream_b_tile(kt_rt):
        # Fresh per-iter fragments (B streamed, not register-resident) then issue_b_load_into.
        bqf = [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(4)]
        bsf = [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(2)]
        issue_b_load_into(bqf, bsf, kt_rt)
        return bqf, bsf

    def issue_a_ds_read(slot):
        issue_a_ds_read_dt(
            s_aq_base, slot, slot_bytes, KH_TILE_A, lane_div_16, lane_mod_16, is_f8_a, a_vals, a_frags, kMChunks
        )

    aq_num_records = fx.Index(i32_max_m_blocks) * fx.Index(fx.Int32(BM) * K_BYTES)

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(
            arg_aq, aq_num_records, s_aq_base, slot, kt, m_row, wave, lane, is_f8_a, KH_TILE_A, K_BYTES, BM=BM
        )

    mma_atoms = scale_mma_atoms() if const_expr(not is_f8_a) else None

    def mfma_cluster(bqf, bsf, sa):
        # opsel (no gate/up split): mni=J//2, in_b=J%2; sa is a per-32-row-chunk list.
        for J in range_constexpr(4):
            mni, in_b = J // 2, J % 2
            sb = _raw(Vec(bsf[mni].load())[0])
            if const_expr(is_bm16):
                mma_one_j(
                    J,
                    in_b,
                    sa[0],
                    sb,
                    bqf,
                    is_f8_a,
                    cbsz_a,
                    a_vals,
                    a_frags,
                    accm,
                    c_frags,
                    mma_atoms,
                    i0=0,
                    single_rg=True,
                    rg_off=rg_off,
                )
                continue
            for sub in range_constexpr(kSubBlocks):
                mma_one_j(
                    J,
                    in_b,
                    sa[sub],
                    sb,
                    bqf,
                    is_f8_a,
                    cbsz_a,
                    a_vals,
                    a_frags,
                    accm,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                )

    # zero C (fp4 fragments accumulate in place; fp8 accm pre-init above).
    if const_expr(not is_f8_a):
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(4):
                c_frags[i][J].store(zero4)

    # Runtime-trip scf.for K-loop: stream A->LDS (triple-buffered) + B per tile; carry C / accm.
    def load_c_carry():
        if const_expr(is_f8_a):
            return [accm[i][J] for i in range(kMChunks) for J in range(4)]
        return [c_frags[i][J].load() for i in range(kMChunks) for J in range(4)]

    def store_c_carry(state):
        n = 0
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(4):
                if const_expr(is_f8_a):
                    accm[i][J] = state[n]
                else:
                    c_frags[i][J].store(state[n])
                n += 1
        return n

    # Step 1: shape-gated, fully-unrolled K_TILES=2 straight-line mainloop (a8w4 reduce / bf16 LDS, BM64).
    # When the compile cap INTER_MAX//BK == 2 the runtime trip is 1 or 2, so we drop the scf.for entirely
    # and emit two straight-line tile bodies. This removes the C/B loop-carry (load_c_carry/store_c_carry,
    # cur/nxt B double-buffer, A-scale carry): each tile's B fragments die before the next stream_b_tile,
    # so cur+nxt B are never simultaneously live -> lower VGPR. A is fully prologue-staged into both slots
    # (kStages==2, aStages==2, no in-loop A prefetch). When runtime inter_dim spans only 1 tile
    # (K_TILES_RT==1) tile-1's A/B/A-scale buffer loads clamp OOB->0, a zero MFMA contribution -> correct.
    straight_line_k2 = is_f8_a and use_reduce and g2_bf16_lds and (BM == 64) and (K_TILES_MAX == 2)
    if const_expr(straight_line_k2):
        gpu.barrier()  # A prologue vmem->LDS (both slots) visible before ds-reads
        for kt in range_constexpr(2):
            issue_a_ds_read(fx.Int32(kt))  # const slot 0/1 (no runtime mod)
            bqf, bsf = stream_b_tile(fx.Int32(kt))  # dies after this tile's mfma_cluster
            sa = load_a_scale_tile(fx.Int32(kt))
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(bqf, bsf, sa)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
    elif const_expr(g2_kstages == 1):
        # 1-deep pipe: synchronous B load per K-tile.
        for kt_iv, state in range(fx.Index(0), fx.Index(K_TILES_RT), fx.Index(1), init=load_c_carry()):
            store_c_carry(state)
            kt_rt = fx.Int32(kt_iv)
            gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt = kt_rt + fx.Int32(kStages)
            if nxt < K_TILES_RT:
                issue_a_load_lds(nxt % fx.Int32(aStages), nxt)
            bqf, bsf = stream_b_tile(kt_rt)
            sa = load_a_scale_tile(kt_rt)
            mfma_cluster(bqf, bsf, sa)
            results = yield load_c_carry()
        store_c_carry(results)
    else:
        # 2-stage B pipeline: consume carried "current" B, prefetch next tile into the same fragments via scf.for state.
        cur_bqf = [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(4)]
        cur_bsf = [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(2)]
        nxt_bqf = [[fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(4)]
        nxt_bsf = [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(2)]
        # g2_ascale_pf: carry the A-scale through scf.for state, same rotating-buffer model as B.
        cur_saf = nxt_saf = None
        if const_expr(g2_ascale_pf):
            cur_saf = [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(kScaleSubBlocks)]
            nxt_saf = [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(kScaleSubBlocks)]

        def load_b_carry():
            # Flat CURRENT (to-consume) B-weight, B-scale, then (opt) A-scale values.
            out = []
            for j in range_constexpr(4):
                for half in range_constexpr(2):
                    out.append(cur_bqf[j][half].load())
            for mw in range_constexpr(2):
                out.append(cur_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(cur_saf[sub].load())
            return out

        def store_b_carry(state, base):
            n = base
            for j in range_constexpr(4):
                for half in range_constexpr(2):
                    cur_bqf[j][half].store(state[n])
                    n += 1
            for mw in range_constexpr(2):
                cur_bsf[mw].store(state[n])
                n += 1
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    cur_saf[sub].store(state[n])
                    n += 1
            return n

        def rotate_b_carry():
            # Yield the PREFETCHED (next-tile) values -> become "current" next iteration.
            out = []
            for j in range_constexpr(4):
                for half in range_constexpr(2):
                    out.append(nxt_bqf[j][half].load())
            for mw in range_constexpr(2):
                out.append(nxt_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(nxt_saf[sub].load())
            return out

        def issue_a_scale_load_into(saf, kt_rt):
            # A-scale vmem load(s) for K-tile kt_rt into the given (per-stage) fragment(s).
            sa = load_a_scale_tile(kt_rt)
            for sub in range_constexpr(kScaleSubBlocks):
                saf[sub].store(sa[sub])

        def load_carry():
            return load_c_carry() + load_b_carry()

        def store_carry(state):
            base = store_c_carry(state)
            store_b_carry(state, base)

        def yield_carry():
            return load_c_carry() + rotate_b_carry()

        # Prologue: prefetch tile 0's B/B-scale into "current" (VALUES enter via init=load_carry()).
        issue_b_load_into(cur_bqf, cur_bsf, fx.Int32(0))
        if const_expr(g2_ascale_pf):
            issue_a_scale_load_into(cur_saf, fx.Int32(0))
        rocdl.sched_barrier(0)

        def prefetch_next_b(kt_rt):
            # Prefetch NEXT tile's B; if none, copy current through (rotate_b_carry state, unused after loop).
            nxt_b = kt_rt + fx.Int32(1)
            if nxt_b < K_TILES_RT:
                issue_b_load_into(nxt_bqf, nxt_bsf, nxt_b)
                if const_expr(g2_ascale_pf):
                    issue_a_scale_load_into(nxt_saf, nxt_b)
            else:
                for j in range_constexpr(4):
                    for half in range_constexpr(2):
                        nxt_bqf[j][half].store(cur_bqf[j][half].load())
                for mw in range_constexpr(2):
                    nxt_bsf[mw].store(cur_bsf[mw].load())
                if const_expr(g2_ascale_pf):
                    for sub in range_constexpr(kScaleSubBlocks):
                        nxt_saf[sub].store(cur_saf[sub].load())

        for kt_iv, state in range(fx.Index(0), fx.Index(K_TILES_RT), fx.Index(1), init=load_carry()):
            store_carry(state)
            kt_rt = fx.Int32(kt_iv)
            if const_expr(g2_bhoist):
                prefetch_next_b(kt_rt)
            gpu.barrier()
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            nxt_a = kt_rt + fx.Int32(kStages)
            if nxt_a < K_TILES_RT:
                issue_a_load_lds(nxt_a % fx.Int32(aStages), nxt_a)
            # A-scale from the prefetch carry (g2_ascale_pf) or loaded synchronously here.
            if const_expr(g2_ascale_pf):
                sa = [_raw(Vec(cur_saf[sub].load())[0]) for sub in range_constexpr(kScaleSubBlocks)]
            else:
                sa = load_a_scale_tile(kt_rt)
            if const_expr(not g2_bhoist):
                prefetch_next_b(kt_rt)
            # Fence the MFMA chain from the B vmem loads (next-tile loads ride ahead of compute).
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(cur_bqf, cur_bsf, sa)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            results = yield yield_carry()
        store_carry(results)

    # epilog: atomic bf16. fp8 reads accm; fp4 loads the C fragments.
    if const_expr(is_f8_a):
        accm_vecs = accm
    else:
        accm_vecs = [[c_frags[i][J].load() for J in range(4)] for i in range(kMChunks)]
    atomic_bf16_epilog(
        lds_acc_base,
        accm_vecs,
        arg_out,
        arg_stids,
        arg_sweights,
        m_row,
        n_block_idx,
        wave,
        lane,
        i32_M,
        BM,
        N_OUT_rt,
        use_reduce=use_reduce,
        topk=topk,
        SBM=SBM,
        g2_bf16_lds=g2_bf16_lds,
    )


# ---- Atomic bf16 epilogue (shared store path; gemm2 down-proj) ----
def atomic_bf16_epilog(
    lds_acc_base,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    *,
    use_reduce=False,
    topk=1,
    SBM=None,
    g2_bf16_lds=False,
):
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16
    M_REPS = BM // 8  # BM32: 4, BM16: 2
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    # bf16 cshuffle LDS: half the C tile footprint (BM*BN*2). Weight is folded in f32 during the
    # write phase and the tile is truncated to bf16 before the LDS store (single rounding, as old
    # FlyDSL); readback then loads the already-weighted bf16 pair and stores it straight to out.
    lds_base_bf16 = lds_typed_ptr(lds_acc_base, T.bf16, align=2) if const_expr(g2_bf16_lds) else None

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // 32
    n_lane = tx_i32 % 32
    col_start = n_lane * 2
    stids_base = global_base_ptr1(arg_stids)
    sweights_base = global_base_ptr1(arg_sweights)
    out_base = global_base_ptr1(arg_out)

    # Prefetch sorted_token_ids / sorted_weights (invariant); latency overlaps stores+barriers.
    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + mr * 8 + m_lane
        packed.append(llvm.load(T.i32, gep1(stids_base, sorted_pos * 4), invariant=True))
        weight.append(llvm.load(T.f32, gep1(sweights_base, sorted_pos * 4), invariant=True))

    # pre-store fence+barrier (HIP run_one __syncthreads() before the epilog).
    gpu.barrier()

    # write accm -> lds_acc cshuffle. f32 path: scalar f32 stores (weight applied on readback).
    # bf16 path: fold the route weight in f32 (per block-row from sorted_weights) then truncate to
    # bf16 so LDS holds the final weighted output tile.
    if const_expr(g2_bf16_lds):
        for i in range_constexpr(kMChunks):
            row_base = fx.Int32(i * 16) + lane_div_16 * 4
            # one f32 route-weight per write-phase block row (row_base+v); sorted_weights is padded.
            w_row = [
                llvm.load(T.f32, gep1(sweights_base, (m_row + row_base + v) * 4), invariant=True)
                for v in range_constexpr(4)
            ]
            for J in range_constexpr(4):
                col = wave * 64 + J * 16 + lane_mod_16
                vec = Vec(accm[i][J])
                for v in range_constexpr(4):
                    idx = (row_base + v) * BN + col
                    lds_base_bf16[idx] = fx.BFloat16(fx.Float32(vec[v]) * fx.Float32(w_row[v]))
    else:
        for i in range_constexpr(kMChunks):
            row_base = fx.Int32(i * 16) + lane_div_16 * 4
            for J in range_constexpr(4):
                col = wave * 64 + J * 16 + lane_mod_16
                vec = Vec(accm[i][J])
                for v in range_constexpr(4):
                    idx = (row_base + v) * BN + col
                    lds_base_fptr[idx] = fx.Float32(vec[v])

    gpu.barrier()

    # read back + weighted store (atomic: fadd out[token_id]; reduce: store out[token_id*topk+slot]);
    # token_id<i32_M gates padding via a DEVICE scf.if (a plain Python `if` on the dynamic token_id
    # does NOT gate at runtime -> its body always ran). At small/mid M the block-sparse sort floor is
    # ~97% padding rows (M64: 12288 pad vs 384 real); ungated, every padding row (token_id==M) issues
    # a weighted atomic-fadd into an OOB out-row -> 77x wasted atomics + L2 RMW serialization
    # (rocprof M64: TCC_ATOMIC 7.3M->95K, 932us->107us). Always gate; reduce already gated via
    # use_reduce. (fp4-atomic families are gated too -> strictly correct OOB-skip, kernel IR changes.)
    guard_padding = True

    def store_one_mr(mr):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if const_expr(use_reduce):
            # reduce out_row can reach tokens*topk (large-M) so compute the element base in i64 (atomic i32 path byte-identical).
            out_row = fx.Int64(token_id * fx.Int32(topk) + (packed[mr] >> fx.Int32(24)))
            row_base_addr = out_row * fx.Int64(N_OUT) + fx.Int64(n_block_idx * BN + col_start)
        else:
            out_row = token_id
            row_base_addr = out_row * N_OUT + n_block_idx * BN + col_start
        for s in range_constexpr(4):
            # adjacent ee=0,1 contiguous -> one 2-wide load.
            idx0 = row_in_block * BN + col_start + s * 64
            if const_expr(g2_bf16_lds):
                # LDS already holds the weighted bf16 pair (4B = 2xbf16): load + store straight through.
                pk = Vec(lds_vec_load(lds_acc_base, idx0 * 2, Vec.make_type(2, fx.BFloat16), fx.BFloat16, align=4))
            else:
                v2 = Vec(lds_vec_load(lds_acc_base, idx0 * 4, Vec.make_type(2, fx.Float32), fx.Float32, align=8))
                pk = Vec.from_elements([v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32).to(fx.BFloat16)
            if const_expr(use_reduce):
                off = (row_base_addr + fx.Int64(s * 64)) * fx.Int64(2)  # bf16 byte off (i64)
            else:
                off = (row_base_addr + s * 64) * 2  # bf16 byte off
            out_ptr = gep1(out_base, off)
            if const_expr(use_reduce):
                llvm.StoreOp(_raw(pk), out_ptr, alignment=4, nontemporal=True)
            else:
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )

    for mr in range_constexpr(M_REPS):
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if const_expr(guard_padding):
            _if_valid = scf.IfOp(_raw(token_id < i32_M))
            with ir.InsertionPoint(_if_valid.then_block):
                store_one_mr(mr)
                scf.YieldOp([])
        else:
            if token_id < i32_M:
                store_one_mr(mr)

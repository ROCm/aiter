# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Layout-API MXFP4 MoE GEMM device body (BM32): gemm2 down."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import _to_raw as _raw
from flydsl.expr import buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN, Float8E4M3FN, Float32, Int32, T
from flydsl.expr.typing import Vector as Vec

from .mxfp4_gemm_common import (
    kStages,
    kBS_stride_k0_dw,
    flat_buffer_view,
    global_typed_ptr,
    lds_dma_atom_128,
    lds_dma_dst,
    lds_typed_ptr,
    lds_vec_load,
    lds_swizzle_mask_f8,
)
from .mxfp4_gemm_common import _global_base_ptr1 as global_base_ptr1
from .mxfp4_gemm_common import _gep1 as gep1
from .mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask
from .mxfp4_gemm_common import _buffer_rsrc

# shape constants (per-shape values come from the compile args)
INTER_MAX_DEFAULT = (
    8192  # compile-time cap for runtime inter_dim (gemm2 B-view / LDS bounds)
)
HIDDEN_MAX_DEFAULT = 8192  # compile-time cap for runtime model_dim/hidden
BN = BK = 256


# BM per-launch (default 32): bodies derive kMChunks=BM//16 (MFMA row-groups), kSubBlocks=BM//32.
BM = 32
kAStages = 3
_I32_COPY_BITS = 32
_CACHE_MODIFIER_DEFAULT = 0
_CACHE_MODIFIER_NONTEMPORAL = 2


# ---- Shared layout-API primitives (B / B-scale data movement + scaled MFMA) ----
def b_copy_atom(nontemporal):
    """Copy one 128-bit B-weight chunk as four i32 lanes."""
    cache_modifier = (
        _CACHE_MODIFIER_NONTEMPORAL if nontemporal else _CACHE_MODIFIER_DEFAULT
    )
    return fx.make_copy_atom(fx.rocdl.BufferCopy128b(cache_modifier), _I32_COPY_BITS)


def bscale_copy_atom():
    """Copy one cached i32 e8m0 B-scale word."""
    return fx.make_copy_atom(
        fx.rocdl.BufferCopy32b(_CACHE_MODIFIER_DEFAULT), _I32_COPY_BITS
    )


def bq_view(arg_bq, row_elems, KH4, K_TILES_TOTAL, num_records_bytes=None):
    """Layout view over preshuffled B for one N-row tile; slice -> i32<4:1> (16B=32 fp4). num_records_bytes (has_pad pad-skip) sizes to REAL K; None -> max_size=False byte-identical default."""
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=16
    )
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * fx.Int64(4))
    # i32 strides: klane[0,4)->64, nlane[0,16)->4, K_tile->512, half[0,2)->256, kpack4->1
    shape = (4, 16, K_TILES_TOTAL, 2, 4)
    view = fx.Tensor(
        fx.make_view(base_iter, fx.make_layout(shape, (64, 4, 512, 256, 1)))
    )
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bscale_view(
    arg_bscale, base_dw, K_TILES_TOTAL, k0_stride_dw=64, num_records_bytes=None
):
    """Layout view over e8m0 B-scale for one n-pack word; slice -> i32<1:1> scale word. num_records_bytes (has_pad pad-skip) sizes to real extent; None -> max_size=False byte-identical default."""
    base_dw = rocdl.readfirstlane(T.i32, _raw(base_dw))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
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


def scale_mma_atoms(a_dtype):
    """16 (opselA,opselB) scaled-MFMA atoms; A elem is fp8/fp4, B is fp4."""
    elem_a = Float8E4M3FN if a_dtype == "fp8" else Float4E2M1FN
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, elem_a, Float4E2M1FN, opsel_a=osa, opsel_b=osb
            )
        )
        for osa in range(4)
        for osb in range(4)
    }


# ---- Shared A ds-read + per-J MMA cluster (used by both gemm bodies) ----
def issue_a_ds_read_dt(
    s_aq_base,
    slot,
    slot_bytes,
    KH_TILE_A,
    lane_div_16,
    lane_mod_16,
    is_f8,
    a_frags,
    kMChunks,
):
    """A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>."""
    for k in range_constexpr(2):
        for i in range_constexpr(kMChunks):
            lds_row = lane_mod_16 + i * 16
            row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
            if const_expr(is_f8):
                mask = lds_swizzle_mask_f8(lane_mod_16)
                col0 = lane_div_16 * 16 + k * 128
                col_lo = col0 ^ mask
                col_hi = (col0 + 64) ^ mask
                lo = Vec(
                    lds_vec_load(
                        s_aq_base,
                        row_off + col_lo,
                        Vec.make_type(2, fx.Int64),
                        fx.Int64,
                        align=16,
                    )
                )
                hi = Vec(
                    lds_vec_load(
                        s_aq_base,
                        row_off + col_hi,
                        Vec.make_type(2, fx.Int64),
                        fx.Int64,
                        align=16,
                    )
                )
                a64 = Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64)
                a_frags[i][k].store(a64.bitcast(fx.Int32))
            else:
                mask = lds_swizzle_mask(lane_mod_16)
                lds_col = (lane_div_16 * 16 + k * 64) ^ mask
                vec = lds_vec_load(
                    s_aq_base,
                    row_off + lds_col,
                    Vec.make_type(4, fx.Int32),
                    fx.Int32,
                    align=16,
                )
                a_frags[i][k].store(Vec(vec))


def mma_one_j(
    J,
    in_b,
    sa,
    sb,
    bq_frags_kt,
    a_frags,
    c_frags,
    atoms,
    i0=0,
    single_rg=False,
    rg_off=0,
):
    """One J-cluster of scaled MFMAs over a 32-row A-scale group (row-groups i0, i0+1); each is
    an fx.gemm on i32 A/B frags (fp8 A = i32<8:1>, fp4 A = i32<4:1>), e8m0 words on scale_a/scale_b.
    sa: 32-row A-scale reg. single_rg (BM16): one 16-row group, rg_off picks its byte, 2 MFMAs."""
    bJ0, bJ1 = bq_frags_kt[J][0], bq_frags_kt[J][1]
    if const_expr(single_rg):
        steps = ((0 + rg_off, 0, i0, bJ0), (2 + rg_off, 1, i0, bJ1))
    else:
        steps = ((0, 0, i0, bJ0), (1, 0, i0 + 1, bJ0), (2, 1, i0, bJ1), (3, 1, i0 + 1, bJ1))
    for osa, k, i, bJ in steps:
        osb = (0 if k == 0 else 2) + in_b
        fx.gemm(
            atoms[(osa, osb)],
            c_frags[i][J],
            a_frags[i][k],
            bJ,
            c_frags[i][J],
            scale_a=sa,
            scale_b=sb,
        )


# ---- gemm2 (down-proj) ----
def issue_a_load_lds_dt(
    arg_aq,
    aq_num_records,
    s_aq_base,
    slot,
    kt,
    m_row,
    wave,
    lane,
    is_f8,
    KH_TILE_A,
    K_BYTES,
    BM=BM,
):
    """A->LDS DMA for one K-tile; gemm2 A is the already-sorted row, OOB-zero via the flat buffer view bounds."""
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
    src = flat_buffer_view(
        arg_aq,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=aq_num_records,
    )
    for g in range_constexpr(n_row_groups):
        lds_row = gather_base_row + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row)
            if const_expr(is_f8)
            else lds_swizzle_mask(lds_row + a_lane_row)
        )
        car = m_row + lds_row + a_lane_row  # direct sorted row
        voffset = (lane_col ^ mask) + car * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        v_e = (voffset + kt * KH_TILE_A) // 4  # per-lane i32-elem index
        fx.copy(
            atom, src[v_e, None], lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16)
        )


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
    # SBM (sort padding unit) >= BM (compute tile); SBM==BM default byte-identical.
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups (BM32: 2, BM64: 4)
    kSubBlocks = (
        BM // 32
    )  # 32-row A-scale chunks / scale-register groups (BM32: 1, BM64: 2)
    # BM16: single 16-row block owning a 32-row scale chunk (chunk==m_block_idx, rg0-only).
    is_bm16 = BM < 32
    rg_off = 0
    kScaleSubBlocks = 1 if is_bm16 else kSubBlocks
    is_f8_a = a_dtype == "fp8"  # only the A path differs
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    # Contraction K = inter_dim runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes (runtime)
    kc_rt = K_rt // fx.Int32(256)  # (K//32)//4//2
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = (
        N_OUT_rt // fx.Int32(32)
    ) * kBS_stride_n0_dw  # (N_OUT//16//2)*stride
    num_n_blocks = N_OUT_rt // fx.Int32(256)
    KH4 = K_rt // fx.Int32(8)  # i32 col stride (= K_HALF//4)
    K_TILES_MAX = INTER_MAX // BK

    # has_pad OOB pad-skip (const_expr-gated): K-skip sizes 16N B-weight buffer to REAL K; N-skip zeros fully-pad-N w2 tiles (col >= N_real=N_OUT-npad; PERF-ONLY). B-scale NOT shrunk.
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
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    ascale_rsrc = _buffer_rsrc(arg_ascale, asc_num)
    scale_chunk0 = m_block_idx if const_expr(is_bm16) else m_row // 32
    a_scale_s_base = rocdl.readfirstlane(
        T.i32, scale_chunk0 * kAS_per_chunk_dw * fx.Int32(4)
    )
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
        return bq_view(
            arg_bq, e * N_OUT_rt + col, KH4, K_TILES_MAX, num_records_bytes=nrec
        )

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
    # A/C both live in fx.gemm fragments (dtype-generic path). fp8 A packs two 128-K halves -> i32<8:1>.
    zero4 = Vec.filled(4, 0.0, Float32)
    A_NDW = 8 if is_f8_a else 4
    a_frags = [
        [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(2)]
        for _ in range_constexpr(kMChunks)
    ]
    c_frags = [
        [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(4)]
        for _ in range_constexpr(kMChunks)
    ]

    def issue_b_load_into(bqf, bsf, kt_rt):
        # Issue B-weight + B-scale vmem loads for K-tile kt_rt into the given (per-stage) fragments.
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                fx.copy(
                    b_catom,
                    bq_views[j][lane_div_16, lane_mod_16, kt_rt, half, None],
                    bqf[j][half],
                )
        for mw in range_constexpr(2):
            fx.copy(
                bs_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, kt_rt, None],
                bsf[mw],
            )

    def stream_b_tile(kt_rt):
        # Fresh per-iter fragments (B streamed, not register-resident) then issue_b_load_into.
        bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)]
            for _ in range_constexpr(4)
        ]
        bsf = [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(2)]
        issue_b_load_into(bqf, bsf, kt_rt)
        return bqf, bsf

    def issue_a_ds_read(slot):
        issue_a_ds_read_dt(
            s_aq_base,
            slot,
            slot_bytes,
            KH_TILE_A,
            lane_div_16,
            lane_mod_16,
            is_f8_a,
            a_frags,
            kMChunks,
        )

    aq_num_records = fx.Int64(i32_max_m_blocks) * fx.Int64(fx.Int32(BM) * K_BYTES)

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(
            arg_aq,
            aq_num_records,
            s_aq_base,
            slot,
            kt,
            m_row,
            wave,
            lane,
            is_f8_a,
            KH_TILE_A,
            K_BYTES,
            BM=BM,
        )

    mma_atoms = scale_mma_atoms(a_dtype)

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
                    a_frags,
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
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                )

    def stream_b_half(kt_rt, half_n):
        bqf = [None, None, None, None]
        for j in (2 * half_n, 2 * half_n + 1):
            bqf[j] = [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)]
            for half in range_constexpr(2):
                fx.copy(
                    b_catom,
                    bq_views[j][lane_div_16, lane_mod_16, kt_rt, half, None],
                    bqf[j][half],
                )
        bsf = fx.make_fragment_like(bs_frag_tmpl)
        fx.copy(
            bs_copy_atom,
            bscale_views[half_n][lane_div_16, lane_mod_16, kt_rt, None],
            bsf,
        )
        return bqf, bsf

    def mfma_cluster_half(bqf, bsf, sa, half_n):
        sb = _raw(Vec(bsf.load())[0])
        for j in (2 * half_n, 2 * half_n + 1):
            in_b = j % 2
            for sub in range_constexpr(kSubBlocks):
                mma_one_j(
                    j,
                    in_b,
                    sa[sub],
                    sb,
                    bqf,
                    a_frags,
                    c_frags,
                    mma_atoms,
                    i0=2 * sub,
                )

    # zero C (fragments accumulate in place).
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(4):
            c_frags[i][J].store(zero4)

    # Runtime-trip scf.for K-loop: stream A->LDS (triple-buffered) + B per tile; carry C fragments.
    def load_c_carry():
        return [c_frags[i][J].load() for i in range(kMChunks) for J in range(4)]

    def store_c_carry(state):
        n = 0
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(4):
                c_frags[i][J].store(state[n])
                n += 1
        return n

    # Straight-line K=2 unroll (compile-time INTER_MAX==512 -> K_TILES_MAX==2): range_constexpr instead of
    # scf.for drops the loop-carry (C/B regs freed -> lower VGPR) and collapses the per-tile barrier to one.
    # fp8 needs the bf16-lds epilog; fp4 uses the c_frags epilog (mma_one_j is dtype-generic), no bf16-lds req.
    straight_line_k2 = (
        use_reduce
        and (BM == 64)
        and (K_TILES_MAX == 2)
        and (g2_bf16_lds if is_f8_a else True)
    )
    if const_expr(straight_line_k2):
        gpu.barrier()
        for kt in range_constexpr(2):
            issue_a_ds_read(fx.Int32(kt))
            sa = load_a_scale_tile(fx.Int32(kt))
            for half_n in range_constexpr(2):
                bqf, bsf = stream_b_half(fx.Int32(kt), half_n)
                rocdl.sched_barrier(0)
                rocdl.s_setprio(1)
                mfma_cluster_half(bqf, bsf, sa, half_n)
                rocdl.s_setprio(0)
                rocdl.sched_barrier(0)
    elif const_expr(g2_kstages == 1):
        # 1-deep pipe: synchronous B load per K-tile.
        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_c_carry(),
        ):
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
        cur_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)]
            for _ in range_constexpr(4)
        ]
        cur_bsf = [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(2)]
        nxt_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(2)]
            for _ in range_constexpr(4)
        ]
        nxt_bsf = [fx.make_fragment_like(bs_frag_tmpl) for _ in range_constexpr(2)]
        # g2_ascale_pf: carry the A-scale through scf.for state, same rotating-buffer model as B.
        cur_saf = nxt_saf = None
        if const_expr(g2_ascale_pf):
            cur_saf = [
                fx.make_fragment_like(bs_frag_tmpl)
                for _ in range_constexpr(kScaleSubBlocks)
            ]
            nxt_saf = [
                fx.make_fragment_like(bs_frag_tmpl)
                for _ in range_constexpr(kScaleSubBlocks)
            ]

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

        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_carry(),
        ):
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
                sa = [
                    _raw(Vec(cur_saf[sub].load())[0])
                    for sub in range_constexpr(kScaleSubBlocks)
                ]
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

    # epilog: atomic bf16. Load the C fragments (fp8/fp4 unified onto the same fx.gemm path).
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
    lds_base_bf16 = (
        lds_typed_ptr(lds_acc_base, T.bf16, align=2)
        if const_expr(g2_bf16_lds)
        else None
    )

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
        packed.append(
            llvm.load(T.i32, gep1(stids_base, sorted_pos * 4), invariant=True)
        )
        weight.append(
            llvm.load(T.f32, gep1(sweights_base, sorted_pos * 4), invariant=True)
        )

    # pre-store fence+barrier (HIP run_one __syncthreads() before the epilog).
    gpu.barrier()

    # write accm -> lds_acc cshuffle. f32 path: scalar f32 stores (weight applied on readback).
    if const_expr(g2_bf16_lds):
        for i in range_constexpr(kMChunks):
            row_base = fx.Int32(i * 16) + lane_div_16 * 4
            w_row = [
                llvm.load(
                    T.f32,
                    gep1(sweights_base, (m_row + row_base + v) * 4),
                    invariant=True,
                )
                for v in range_constexpr(4)
            ]
            for J in range_constexpr(4):
                col = wave * 64 + J * 16 + lane_mod_16
                vec = Vec(accm[i][J])
                for v in range_constexpr(4):
                    idx = (row_base + v) * BN + col
                    lds_base_bf16[idx] = fx.BFloat16(
                        fx.Float32(vec[v]) * fx.Float32(w_row[v])
                    )
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
            row_base_addr = out_row * fx.Int64(N_OUT) + fx.Int64(
                n_block_idx * BN + col_start
            )
        else:
            out_row = token_id
            row_base_addr = out_row * N_OUT + n_block_idx * BN + col_start
        for s in range_constexpr(4):
            # adjacent ee=0,1 contiguous -> one 2-wide load.
            idx0 = row_in_block * BN + col_start + s * 64
            if const_expr(g2_bf16_lds):
                pk = Vec(
                    lds_vec_load(
                        lds_acc_base,
                        idx0 * 2,
                        Vec.make_type(2, fx.BFloat16),
                        fx.BFloat16,
                        align=4,
                    )
                )
            else:
                v2 = Vec(
                    lds_vec_load(
                        lds_acc_base,
                        idx0 * 4,
                        Vec.make_type(2, fx.Float32),
                        fx.Float32,
                        align=8,
                    )
                )
                pk = Vec.from_elements(
                    [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
                ).to(fx.BFloat16)
            if const_expr(use_reduce):
                off = (row_base_addr + fx.Int64(s * 64)) * fx.Int64(
                    2
                )  # bf16 byte off (i64)
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

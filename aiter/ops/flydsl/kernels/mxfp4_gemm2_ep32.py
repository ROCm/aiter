# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""Shared EP32 A8W4 GEMM2 device code and fixed-width launcher."""

import flydsl.compiler as flyc
from flydsl._mlir.dialects import llvm
import flydsl.expr as fx
from flydsl.expr import (
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr.typing import (
    BFloat16,
    Float4E2M1FN,
    Float8E4M3FN,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value as _raw

from .mxfp4_gemm_common import (
    _buffer_rsrc,
    _gep3,
    _lds_ptr3,
    _udiv,
    flat_buffer_view,
    global_typed_ptr,
    kBS_stride_k0_dw,
    kStages,
    lds_dma_atom_128,
    lds_dma_dst,
    lds_swizzle_mask_f8,
    lds_typed_ptr,
    lds_vec_load,
)
from .mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask

_FP8_E8M0_SHIFT = 7

_G2_EPI_LANES = 32

_SCOPE_CACHE = {}


def lds_slot_scopes(n, tag):
    """n LLVM alias scopes, one per LDS A slot: a read of slot s does not wait
    for LDS DMAs into the other slots."""
    from flydsl._mlir import ir

    ctx = ir.Context.current
    key = (id(ctx), tag, n)
    if key not in _SCOPE_CACHE:
        dom = f'#llvm.alias_scope_domain<id = distinct[0]<>, description = "{tag}">'
        txt = "[" + ", ".join(
            f"#llvm.alias_scope<id = distinct[{i + 1}]<>, domain = {dom}>"
            for i in range(n)
        ) + "]"
        arr = ir.ArrayAttr(ir.Attribute.parse(txt))
        scopes = [arr[i] for i in range(n)]
        _SCOPE_CACHE[key] = [
            (
                ir.ArrayAttr.get([scopes[i]]),
                ir.ArrayAttr.get([scopes[j] for j in range(n) if j != i]),
            )
            for i in range(n)
        ]
    return _SCOPE_CACHE[key]


def bq_view(
    arg_bq,
    row_elems,
    KH4,
    K_TILES_TOTAL,
    K_HALVES,
):
    """Layout view over preshuffled B for one N-row tile."""
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=16
    )
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq) + off_i64 * 4)
    # i32 strides: klane[0,4)->64, nlane[0,16)->4,
    # K_tile->K_HALVES*256, half->256, kpack4->1.
    shape = (4, 16, K_TILES_TOTAL, K_HALVES, 4)
    view = fx.Tensor(
        fx.make_view(
            base_iter,
            fx.make_layout(shape, (64, 4, K_HALVES * 256, 256, 1)),
        )
    )
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def bq_view_fp8(
    arg_bq,
    row_elems,
    KH4,
    K_TILES_TOTAL,
    K_HALVES,
):
    """Layout view over preshuffled FP8 B; pair selects two 16B cells per MFMA."""
    base = bq_view(
        arg_bq,
        row_elems,
        KH4,
        K_TILES_TOTAL * 2,
        K_HALVES,
    )
    shape = (4, 16, K_TILES_TOTAL, K_HALVES, 2, 4)
    stride = (64, 4, K_HALVES * 2 * 256, 2 * 256, 256, 1)
    return fx.Tensor(fx.make_view(fx.get_iter(base), fx.make_layout(shape, stride)))


def scale_view(
    arg_scale, base_dw, K_TILES_TOTAL, k0_stride_dw=64, num_records_bytes=None
):
    """Layout view over an e8m0 scale buffer with an optional byte bound."""
    base_dw = rocdl.readfirstlane(T.i32, _raw(base_dw))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=4
    )
    off_i64 = fx.Int64(base_dw)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_scale) + off_i64 * 4)
    shape = (4, 16, K_TILES_TOTAL, 1)
    stride = (16, 1, k0_stride_dw, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, stride)))
    if num_records_bytes is not None:
        return fx.rocdl.make_buffer_tensor(view, num_records_bytes=num_records_bytes)
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


def scale_mma_atoms(a_dtype, b_dtype):
    """16 (opselA,opselB) scaled-MFMA atoms for FP4/FP8 operands."""
    elem_a = Float8E4M3FN if a_dtype == "fp8" else Float4E2M1FN
    elem_b = Float8E4M3FN if b_dtype == "fp8" else Float4E2M1FN
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, elem_a, elem_b, opsel_a=osa, opsel_b=osb
            )
        )
        for osa in range(4)
        for osb in range(4)
    }


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
    k_halves=2,
):
    """One J-cluster of scaled MFMAs over a 32-row A-scale group (row-groups i0, i0+1); each is
    an fx.gemm on i32 A/B frags (fp8 A = i32<8:1>, fp4 A = i32<4:1>), e8m0 words on scale_a/scale_b.
    sa: 32-row A-scale reg. single_rg (BM16): one 16-row group, rg_off picks its byte.
    """
    row_groups = (rg_off,) if const_expr(single_rg) else range(2)
    for k in range(k_halves):
        for im in row_groups:
            i = i0 if const_expr(single_rg) else i0 + im
            fx.gemm(
                atoms[(2 * k + im, 2 * k + in_b)],
                c_frags[i][J],
                a_frags[i][k],
                bq_frags_kt[J][k],
                c_frags[i][J],
                scale_a=sa,
                scale_b=sb,
            )


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
    BM=32,
    resolved_rows=(),
    scopes=None,
    dup_tail=False,
    n_waves=4,
    active_waves=None,
):
    """A->LDS DMA for one K-tile; gemm2 A is the already-sorted row, OOB-zero via the flat buffer view bounds."""
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // 4
    partial_wave_gather = rows_per_wave < rows_per_call
    if const_expr(partial_wave_gather):
        n_gather_calls = BM // rows_per_call
        gather_base_row = (wave % fx.Int32(n_gather_calls)) * rows_per_call
        n_row_groups = 1
    else:
        gather_base_row = wave * rows_per_wave
        n_row_groups = rows_per_wave // rows_per_call
    # Rows past the largest multiple of 4*rows_per_call (the 16-row tail of the BM64 fold) are
    # loaded by the last waves, one rows_per_call group each. (Issuing them from
    # waves 0-1 instead produced corrupted rows 0-15 in testing; not root-caused.)
    tail_rows = 0
    if const_expr(not partial_wave_gather and rows_per_wave % rows_per_call != 0):
        main_rows = (BM // (4 * rows_per_call)) * 4 * rows_per_call
        tail_rows = BM - main_rows
        assert tail_rows % rows_per_call == 0 and tail_rows // rows_per_call <= 4
        rows_per_wave = main_rows // 4
        gather_base_row = wave * rows_per_wave
        n_row_groups = rows_per_wave // rows_per_call
    lane_col = (lane % lanes_per_row) * 16
    if const_expr(n_waves != 4):
        n_groups = BM // rows_per_call
        per_wave = (n_groups + n_waves - 1) // n_waves
        n_row_groups = 0
        tail_rows = 0
    atom = lds_dma_atom_128()
    rsrc = _buffer_rsrc(arg_aq, aq_num_records) if const_expr(scopes is not None) else None

    def _dma(voffset_bytes, off):
        if const_expr(scopes is not None):
            rocdl.raw_ptr_buffer_load_lds(
                rsrc, _lds_ptr3(s_aq_base, off), fx.Int32(16), voffset_bytes,
                fx.Int32(0), fx.Int32(0), fx.Int32(0),
                alias_scopes=scopes[0], noalias_scopes=scopes[1],
            )
            return
        v_e = voffset_bytes.shrui(fx.Int32(2))
        fx.copy(
            atom, src[v_e, None], lds_dma_dst(s_aq_base, off, elem_ty=T.i32, align=16)
        )
    src = flat_buffer_view(
        arg_aq,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=aq_num_records,
    )
    if const_expr(n_waves != 4):
        for i in range_constexpr(per_wave):
            step = fx.Int32(n_waves) if const_expr(active_waves is None) else active_waves
            gi = wave + fx.Int32(i) * step
            gi = (gi < fx.Int32(n_groups)).select(gi, gi - fx.Int32(n_groups))
            lds_row = gi * fx.Int32(rows_per_call)
            mask = lds_swizzle_mask_f8(lds_row + a_lane_row, KH_TILE_A)
            sorted_row = m_row + lds_row + a_lane_row
            voffset = (lane_col ^ mask) + sorted_row * K_BYTES
            off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
            if wave < step:
                _dma(voffset + kt * KH_TILE_A, off)
    for g in range_constexpr(n_row_groups):
        lds_row = gather_base_row + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row, KH_TILE_A)
            if const_expr(is_f8)
            else lds_swizzle_mask(lds_row + a_lane_row, KH_TILE_A)
        )
        sorted_row = m_row + lds_row + a_lane_row
        car = resolved_rows[g] if const_expr(len(resolved_rows) > 0) else sorted_row
        voffset = (lane_col ^ mask) + car * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        # The byte offset is non-negative and 4-byte aligned; avoid signed-division fixup VGPRs.
        _dma(voffset + kt * KH_TILE_A, off)
    if const_expr(tail_rows > 0 and dup_tail):
        n_tail = tail_rows // rows_per_call
        lds_row = fx.Int32(BM - tail_rows) + (wave % fx.Int32(n_tail)) * rows_per_call
        mask = lds_swizzle_mask_f8(lds_row + a_lane_row, KH_TILE_A)
        sorted_row = m_row + lds_row + a_lane_row
        voffset = (lane_col ^ mask) + sorted_row * K_BYTES
        off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
        _dma(voffset + kt * KH_TILE_A, off)
    elif const_expr(tail_rows > 0):
        tail_w0 = 4 - tail_rows // rows_per_call
        if wave >= fx.Int32(tail_w0):
            lds_row = fx.Int32(BM - tail_rows) + (wave - fx.Int32(tail_w0)) * rows_per_call
            mask = lds_swizzle_mask_f8(lds_row + a_lane_row, KH_TILE_A)
            sorted_row = m_row + lds_row + a_lane_row
            voffset = (lane_col ^ mask) + sorted_row * K_BYTES
            off = fx.Int32(slot * (BM * KH_TILE_A)) + lds_row * KH_TILE_A
            _dma(voffset + kt * KH_TILE_A, off)


@flyc.jit
def gemm2_body_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_stids,
    arg_sweights,
    arg_bias,
    i32_M,
    i32_max_m_blocks,
    arg_out,
    bx_i32,
    lane,
    wave,
    arg_aq,
    i32_inter,
    i32_hidden,
    *,
    BM,
    BN=256,
    BK=256,
    use_nt,
    INTER_MAX,
    aStages,
    a_slot_alias=False,
    a_dtype,
    b_dtype,
    use_reduce=False,
    topk=1,
    SBM=None,
    mn_idx=None,
    g2_bhoist=True,
    g2_ascale_pf=True,
    g2_bf16_lds=False,
    route_out_fp8=False,
    g2_defer_weight=0,
    g2_out_pitch_align=0,
    g2_scale_blk=8,
    g2_epi_lanes=None,
    g2_apre=False,
    enable_bias=False,
    nonatomic=False,
    scatter_out_fp4=False,
    arg_out_scale=None,
    g2_prefetch_ids=False,
    reduce_store_cache_modifier=None,
    resolved_input_rows=(),
    output_n_base=0,
    output_width=None,
    write_limit=None,
    m_row_in=None,
    BLK=None,
    row_limit=None,
    e_in=None,
    prune=False,
    b_lds=False,
    b_nt=False,
    sorted_rows=False,
    native_c=False,
    native_bf16=False,
    n_waves=4,
    b_stages=3,
    n_col_in=None,
    n_live=None,
    native_variant=True,
):
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups
    kHalves = BK // 128  # 16x16x128 MFMA K-steps per K-tile
    tilesPerScaleChunk = 256 // BK  # K-tiles sharing one 256-K E8M0 word
    CPW = BN // n_waves  # columns per wave
    assert CPW % 32 == 0, CPW  # B-scale words cover 32 columns
    numAccN = CPW // 16  # 16-column MFMA subblocks per wave
    nPairs = max(1, numAccN // 2)  # one B-scale per two 16-column subblocks
    # BM16: single 16-row block owning a 32-row scale chunk (chunk==m_block_idx, rg0-only).
    is_bm16 = BM < 32
    kScaleSubBlocks = max(1, (kMChunks + 1) // 2)
    if BLK is None:
        BLK = BM
    is_f8_a = a_dtype == "fp8"  # only the A path differs
    is_f8_b = b_dtype == "fp8"
    B_NDW = 8 if is_f8_b else 4
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_bytes = BM * KH_TILE_A
    # Contraction K = inter_dim (i32_inter); the K-loop is unrolled over INTER_MAX // BK tiles, so i32_inter must equal INTER_MAX.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = _udiv(K_rt, fx.Int32(a_pack))
    kc_rt = _udiv(K_rt + 255, fx.Int32(256))
    kAS_per_chunk_dw = kc_rt * fx.Int32(64)
    kBS_stride_n0_dw = kc_rt * fx.Int32(64)
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = _udiv(N_OUT_rt, fx.Int32(32)) * kBS_stride_n0_dw
    num_n_blocks = _udiv(N_OUT_rt, fx.Int32(BN))
    KH4 = _udiv(K_rt, fx.Int32(4 if is_f8_b else 8))
    K_TILES_MAX = INTER_MAX // BK
    K_SCALE_CHUNKS_MAX = (INTER_MAX + 255) // 256

    # block -> (m_block_idx, n_block_idx); e = sorted_expert_ids[SBM-padded sort block] (SBM==BM: sort_block==m_block_idx).
    if const_expr(mn_idx is not None):
        m_block_idx, n_block_idx = mn_idx
    else:
        m_block_idx = _udiv(bx_i32, num_n_blocks)
        n_block_idx = bx_i32 - m_block_idx * num_n_blocks
    eids_ptr = global_typed_ptr(arg_eids, T.i32)
    m_row = m_block_idx * BM if const_expr(m_row_in is None) else m_row_in
    n_col = (n_block_idx * fx.Int32(BN) if const_expr(n_col_in is None) else n_col_in) if const_expr(native_variant) else None
    if const_expr(e_in is not None):
        e = e_in
    elif const_expr(SBM == BM):
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[m_block_idx]))
    else:
        e = rocdl.readfirstlane(T.i32, _raw(eids_ptr[_udiv(m_row, fx.Int32(SBM))]))

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    # ep32 (row_limit given): a 32-row scale group whose first sorted row is
    # padding (or past row_limit) is padding throughout -- padding follows each
    # expert's valid routes -- so its MFMAs are skipped.
    sub_live = [None] * max(1, kMChunks // 2 + kMChunks % 2)
    if const_expr(prune and row_limit is not None):
        stids_ptr = global_typed_ptr(arg_stids, T.i32)
        tok_limit = i32_M if const_expr(write_limit is None) else write_limit
        for sub in range_constexpr(1, len(sub_live)):
            packed = rocdl.readfirstlane(T.i32, _raw(stids_ptr[m_row + sub * 32]))
            sub_live[sub] = ((packed & fx.Int32(0x00FFFFFF)) < tok_limit) & (
                fx.Int32(sub * 32) < row_limit
            )

    s_aq_base = lds_base_i32
    lds_acc_base = lds_base_i32
    mma_atoms = scale_mma_atoms(a_dtype, b_dtype)

    if const_expr(len(resolved_input_rows) > 0):
        aq_num_records = fx.Int64(i32_M) * fx.Int64(topk) * fx.Int64(K_BYTES)
    else:
        aq_num_records = fx.Int64(i32_max_m_blocks) * fx.Int64(BLK) * fx.Int64(K_BYTES)
    A_NDW = 8 if is_f8_a else 4
    a_frags = [
        [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(kMChunks)
    ]

    ab_scopes = lds_slot_scopes(7, "ep32_abf") if const_expr(b_lds) else None
    a_scopes = ab_scopes[:3] if const_expr(b_lds) else None
    b_scopes = ab_scopes[3:] if const_expr(b_lds) else None

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
            resolved_rows=resolved_input_rows,
            scopes=a_scopes[slot] if const_expr(b_lds) else None,
            dup_tail=b_lds,
            n_waves=n_waves,
            active_waves=None if const_expr(n_live is None) else n_live // fx.Int32(CPW),
        )

    def a_slot_probe(slot):
        # Every wave fences its DMA into this slot before sharing A. The
        # slot scopes keep future A/B DMAs outside this wait.
        llvm.load(
            T.i32, _lds_ptr3(s_aq_base, fx.Int32(slot * slot_bytes)),
            alignment=4, volatile_=True,
            alias_scopes=a_scopes[slot][0], noalias_scopes=a_scopes[slot][1],
        )

    a_rd_bases = {}

    def a_rd_base(k, h):
        if const_expr((k, h) not in a_rd_bases):
            mask = lds_swizzle_mask_f8(lane_mod_16, KH_TILE_A)
            col = (lane_div_16 * 16 + k * 128 + h * 64) ^ mask
            a_rd_bases[(k, h)] = _lds_ptr3(s_aq_base, lane_mod_16 * KH_TILE_A + col)
        return a_rd_bases[(k, h)]

    def issue_a_ds_read(slot):
        # A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>.
        for k in range_constexpr(kHalves):
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + i * 16
                row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
                mask = lds_swizzle_mask_f8(lane_mod_16, KH_TILE_A)
                col0 = lane_div_16 * 16 + k * 128
                col_lo = col0 ^ mask
                col_hi = (col0 + 64) ^ mask
                if const_expr(b_lds):
                    # Constant GEP offsets fold into DS instructions;
                    # avoid keeping one i64 address per slot/row alive.
                    lo, hi = [Vec(llvm.load(
                        Vec.make_type(2, fx.Int64),
                        _gep3(a_rd_base(k, h), slot * slot_bytes + i * 16 * KH_TILE_A),
                        alignment=16,
                        alias_scopes=a_scopes[slot][0], noalias_scopes=a_scopes[slot][1],
                    )) for h in (0, 1)]
                else:
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

    # Scale words (e8m0): shared scale_view / copy atom for both A and B. A-scale is one
    # word per 32-row chunk, each view bounded to bytes remaining after its baked base.
    sc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), 32)

    asc_per_mb = fx.Int32(max(1, BLK // 32)) * kAS_per_chunk_dw * fx.Int32(4)
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    scale_chunk0 = (
        m_block_idx if const_expr(is_bm16 and SBM == BM) else m_row // fx.Int32(32)
    )

    def make_ascale_view(sub):
        base_dw = (scale_chunk0 + fx.Int32(sub)) * kAS_per_chunk_dw
        nrec = asc_num - fx.Int64(base_dw) * 4
        return scale_view(
            arg_ascale,
            base_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=64,
            num_records_bytes=nrec,
        )

    ascale_views = [make_ascale_view(sub) for sub in range_constexpr(kScaleSubBlocks)]
    sc_frag_tmpl = ascale_views[0][0, 0, 0, None]  # i32<1:1> (one e8m0 word)

    def scale_chunk_tile(kt):
        return (
            kt
            if const_expr(tilesPerScaleChunk == 1)
            else _udiv(kt, fx.Int32(tilesPerScaleChunk))
        )


    def load_a_scale_tile(kt, realign=True):
        chunk_kt = scale_chunk_tile(kt)
        out = []
        for sub in range_constexpr(kScaleSubBlocks):
            saf = fx.make_fragment_like(sc_frag_tmpl)
            fx.copy(
                sc_copy_atom,
                ascale_views[sub][lane_div_16, lane_mod_16, chunk_kt, None],
                saf,
            )
            scale = Vec(saf.load())[0]
            out.append(scale)
        return out

    # B-weight + B-scale: global->register, streamed per K-tile (not LDS-staged).
    # b128 weight copy atom; cache modifier 2=nontemporal, 0=default.
    b_catom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(2 if use_nt else 0), 32)

    def make_bq_view(j):
        col = (n_col + wave * CPW + j * 16) if const_expr(native_variant) else (n_block_idx * BN + wave * (BN // 4) + j * 16)
        return bq_view(
            arg_bq,
            e * N_OUT_rt + col,
            KH4,
            K_TILES_MAX,
            kHalves,
        )

    bq_views = [make_bq_view(j) for j in range_constexpr(numAccN)]

    mni_base = (n_col // fx.Int32(32) + wave * (CPW // 32)) if const_expr(native_variant) else (n_block_idx * (BN // 16 // 2) + wave * (BN // 64 // 2))
    bscale_views = [
        scale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(nPairs)
    ]

    frag_tmpl = (
        None
        if const_expr(is_f8_b)
        else bq_views[0][0, 0, 0, 0, None]  # i32<4:1> (16B = 32 fp4)
    )
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_value_load(dst, j, half, kt_rt):
        fx.copy(
            b_catom,
            bq_views[j][lane_div_16, lane_mod_16, kt_rt, half, None],
            dst,
        )

    def issue_bscale_into(bsf, chunk_kt):
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
                bsf[mw],
            )

    def issue_b_load_into(bqf, bsf, kt_rt):
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                issue_b_value_load(bqf[j][half], j, half, kt_rt)
        if const_expr(bsf is not None):
            issue_bscale_into(bsf, scale_chunk_tile(kt_rt))

    def make_bq_fragments():
        return [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]

    def make_scale_fragments(count):
        return [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(count)]

    def shift_scale_word(scale, kt_rt):
        if const_expr(tilesPerScaleChunk == 1):
            return scale
        scale_shift = (kt_rt % fx.Int32(tilesPerScaleChunk)) * fx.Int32(16)
        return scale.shrui(scale_shift)

    def mfma_cluster(bqf, bsf, sa, kt_rt, interleave=None):
        # opsel (no gate/up split): mni=J//2, in_b=J%2; sa is a per-32-row-chunk list.
        sa = [
            shift_scale_word(sa[sub], kt_rt) for sub in range_constexpr(kScaleSubBlocks)
        ]
        sb_words = [
            shift_scale_word(Vec(bsf[mni].load())[0], kt_rt)
            for mni in range_constexpr(nPairs)
        ]
        for J in range_constexpr(numAccN):
            mni, in_b = J // 2, J % 2
            sb = sb_words[mni]
            for sub in range_constexpr(kScaleSubBlocks):
                # Odd chunk count: the last 32-row scale group has one 16-row chunk.
                def _mma_sub(sub=sub):
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
                        single_rg=(2 * sub + 1 >= kMChunks),
                        rg_off=0,
                        k_halves=kHalves,
                    )

                if const_expr(sub_live[sub] is None):
                    _mma_sub()
                else:
                    if sub_live[sub]:
                        _mma_sub()
            if const_expr(interleave is not None and J > 0):
                interleave[J - 1]()
        if const_expr(interleave is not None):
            interleave[numAccN - 1]()

    # C accumulator: register fragments, zeroed then accumulated in place.
    zero4 = fx.full(4, 0.0, Float32)
    c_frags = [
        [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(numAccN)]
        for _ in range_constexpr(kMChunks)
    ]
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(numAccN):
            c_frags[i][J].store(zero4)

    def _epilog(accm, **kw):
        if const_expr(native_c):
            if const_expr(native_variant):
                # Native BF16 stores the same weighted, rounded route contribution
                # that the FP32 reducer computes, so the following sum is unchanged.
                out_elem = BFloat16 if const_expr(native_bf16) else Float32
                out_ptr = global_typed_ptr(arg_out, T.bf16 if const_expr(native_bf16) else T.f32, align=16)
                out_view = fx.Tensor(fx.make_view(out_ptr, fx.make_layout((1, 1), (1, 1))))
                out_buf = fx.rocdl.make_buffer_tensor(out_view, max_size=True)
                store = fx.make_copy_atom(
                    fx.rocdl.BufferCopy64b() if const_expr(native_bf16) else fx.rocdl.BufferCopy128b(), out_elem
                )
                if const_expr(native_bf16):
                    weight_ptr = global_typed_ptr(arg_sweights, T.f32, align=16)
                    weight_view = fx.Tensor(fx.make_view(weight_ptr, fx.make_layout((1, 1), (1, 1))))
                    weight_buf = fx.rocdl.make_buffer_tensor(weight_view, max_size=True)
                    weight_load = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), Float32)
                n16 = _udiv(N_OUT_rt, fx.Int32(16))
                live_cols = fx.Int32(BN) if const_expr(n_live is None) else n_live
                if wave * fx.Int32(CPW) < live_cols:
                    for i in range_constexpr(kMChunks):
                        mc = _udiv(m_row, fx.Int32(16)) + fx.Int32(i)
                        if const_expr(native_bf16):
                            wf = fx.make_rmem_tensor(4, Float32)
                            weight_row = m_row + fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
                            fx.copy(weight_load, weight_buf[None, weight_row], wf)
                            weights4 = Vec(wf.load())
                        for J in range_constexpr(numAccN):
                            nc = n_col // fx.Int32(16) + wave * fx.Int32(CPW // 16) + fx.Int32(J)
                            offset = (mc * n16 + nc) * fx.Int32(256) + lane * fx.Int32(4)
                            frag = fx.make_rmem_tensor(4, out_elem)
                            values = Vec(accm[i][J])
                            if const_expr(native_bf16):
                                values = (values * weights4).to(BFloat16)
                            frag.store(values)
                            fx.copy(store, frag, out_buf[None, offset])
            else:
                # Store each native MFMA fragment contiguously: no C LDS transpose,
                # token gather, route-weight multiply or BF16 conversion in GEMM2.
                out_ptr = global_typed_ptr(arg_out, T.f32, align=16)
                out_view = fx.Tensor(fx.make_view(out_ptr, fx.make_layout((1, 1), (1, 1))))
                out_buf = fx.rocdl.make_buffer_tensor(out_view, max_size=True)
                store = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), Float32)
                n16 = _udiv(N_OUT_rt, fx.Int32(16))
                for i in range_constexpr(kMChunks):
                    mc = _udiv(m_row, fx.Int32(16)) + fx.Int32(i)
                    for J in range_constexpr(numAccN):
                        nc = n_block_idx * fx.Int32(BN // 16) + wave * fx.Int32(BN // 64) + fx.Int32(J)
                        offset = (mc * n16 + nc) * fx.Int32(256) + lane * fx.Int32(4)
                        frag = fx.make_rmem_tensor(4, Float32)
                        frag.store(Vec(accm[i][J]))
                        fx.copy(store, frag, out_buf[None, offset])
        else:
            atomic_bf16_epilog(
                lds_acc_base,
                accm,
                arg_out,
                arg_stids,
                arg_sweights,
                arg_bias,
                e,
                m_row,
                n_block_idx,
                wave,
                lane,
                i32_M,
                BM,
                N_OUT_rt,
                BN=BN,
                use_reduce=use_reduce,
                topk=topk,
                SBM=SBM,
                g2_bf16_lds=g2_bf16_lds,
                route_out_fp8=route_out_fp8,
                g2_defer_weight=g2_defer_weight,
                g2_out_pitch_align=g2_out_pitch_align,
                g2_scale_blk=g2_scale_blk,
                g2_epi_lanes=g2_epi_lanes,
                enable_bias=enable_bias,
                output_n_base=output_n_base,
                output_width=output_width,
                reduce_store_cache_modifier=reduce_store_cache_modifier,
                write_limit=write_limit,
                row_limit=row_limit,
                opaque_tid=b_lds,
                sorted_rows=sorted_rows,
                n_waves=n_waves,
                **kw,
            )

    g2_interleave = const_expr(g2_bf16_lds and not nonatomic and output_width is None and not native_c)
    g2_prefetch_ids = const_expr(
        g2_prefetch_ids
        and g2_interleave
        and use_reduce
        and route_out_fp8
        and bool(g2_defer_weight)
    )
    prefetched_ids = None
    epi_thunks = [] if const_expr(g2_interleave) else None
    if const_expr(g2_interleave):
        _epilog(c_frags, emit_thunks=epi_thunks)

    if const_expr(b_lds):
        # Three A slots and a per-specialization B ring. BM96 uses three B
        # slots to fit 160 KiB LDS; both streams are then two steps ahead. Scopes
        # let current-slot DS reads wait without draining future DMAs.
        assert BM in (64 + 16, 96) and BK == 128 and aStages == 3
        KT = K_TILES_MAX
        B_SLOT = BN * KH_TILE_A // 2
        b_base = aStages * slot_bytes
        KB_B = _udiv(K_rt, fx.Int32(2))
        b_row0 = (e * N_OUT_rt + n_col + wave * CPW) if const_expr(native_variant) else (e * N_OUT_rt + n_block_idx * BN + wave * (BN // 4))
        b_rsrc = _buffer_rsrc(
            fx.Int64(arg_bq) + fx.Int64(rocdl.readfirstlane(T.i32, _raw(b_row0))) * fx.Int64(KB_B),
            fx.Int64(fx.Int32(CPW) * KB_B if const_expr(n_live is None) else (wave * fx.Int32(CPW) < n_live).select(fx.Int32(CPW), fx.Int32(0)) * KB_B),
        )
        lane16 = lane * fx.Int32(16)
        bsf_slots = [make_scale_fragments(nPairs) for _ in range_constexpr(2)]
        saf_slots = [make_scale_fragments(kScaleSubBlocks) for _ in range_constexpr(2)]
        bqf = make_bq_fragments()
        b_rd_base = _lds_ptr3(s_aq_base, wave * fx.Int32(CPW * 64) + lane16)
        for k in range_constexpr(kHalves):
            for h in range_constexpr(2):
                a_rd_base(k, h)

        def issue_b_dma(t):
            slot = t % b_stages
            for J in range_constexpr(numAccN):
                off = fx.Int32(b_base + slot * B_SLOT + J * 1024) + wave * fx.Int32(CPW * 64)
                voff = lane16 + fx.Int32(J * 16) * KB_B + fx.Int32(t * 1024)
                rocdl.raw_ptr_buffer_load_lds(
                    b_rsrc, _lds_ptr3(s_aq_base, off), fx.Int32(16), voff,
                    fx.Int32(0), fx.Int32(0), 2 if b_nt else 0,
                    alias_scopes=b_scopes[slot][0], noalias_scopes=b_scopes[slot][1],
                )

        def read_b(t):
            slot = t % b_stages
            for J in range_constexpr(numAccN):
                v = llvm.load(
                    T.vec(4, T.i32), _gep3(b_rd_base, b_base + slot * B_SLOT + J * 1024),
                    alignment=16,
                    alias_scopes=b_scopes[slot][0], noalias_scopes=b_scopes[slot][1],
                )
                bqf[J][0].store(Vec(v))

        def issue_tile(t, with_b=True):
            if const_expr(t < KT):
                issue_a_load_lds(t % 3, fx.Int32(t))
                if const_expr(with_b):
                    issue_b_dma(t)
                if const_expr(t % 2 == 0):
                    c = t // 2
                    issue_bscale_into(bsf_slots[c % 2], scale_chunk_tile(fx.Int32(t)))
                    # Raw words: realigning here would wait for this step's DMAs.
                    sa = load_a_scale_tile(fx.Int32(t), realign=False)
                    for sub in range_constexpr(kScaleSubBlocks):
                        saf_slots[c % 2][sub].store(Vec.from_elements([sa[sub]], Int32))

        for t in range_constexpr(2):
            issue_tile(t)
        for t in range_constexpr(2, b_stages - 1):
            issue_b_dma(t)
        rocdl.sched_barrier(0)
        for kt in range_constexpr(KT):
            kt_rt = fx.Int32(kt)
            slot = kt % 3
            a_slot_probe(slot)
            rocdl.sched_barrier(0)
            rocdl.s_barrier()
            rocdl.sched_barrier(0)
            issue_a_ds_read(slot)
            read_b(kt)
            rocdl.sched_barrier(0)
            if const_expr(native_variant):
                issue_tile(kt + 2, with_b=False)
                if const_expr(kt + b_stages - 1 < KT):
                    issue_b_dma(kt + b_stages - 1)
            else:
                issue_tile(kt + 2)
            rocdl.sched_barrier(0)
            c = kt // 2
            sa = [Vec(saf_slots[c % 2][sub].load())[0] for sub in range_constexpr(kScaleSubBlocks)]
            interleave = None
            if const_expr(kt == KT - 1):
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                gpu.barrier()
                interleave = epi_thunks
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            if const_expr(n_live is None):
                mfma_cluster(bqf, bsf_slots[c % 2], sa, kt_rt, interleave=interleave)
            else:
                if wave * fx.Int32(CPW) < n_live:
                    mfma_cluster(bqf, bsf_slots[c % 2], sa, kt_rt, interleave=interleave)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
    else:
        KT = K_TILES_MAX
        for i in range_constexpr(kMChunks):
            for J in range_constexpr(numAccN):
                c_frags[i][J].store(zero4)
        cur_bqf = make_bq_fragments()
        nxt_bqf = make_bq_fragments()
        chunk_of = [kt // tilesPerScaleChunk for kt in range(KT)]
        n_slots = min(2, chunk_of[-1] + 1)
        bsf_slots = [make_scale_fragments(nPairs) for _ in range_constexpr(n_slots)]
        saf_slots = None
        saf_slots = [
            make_scale_fragments(kScaleSubBlocks) for _ in range_constexpr(n_slots)
        ]

        def _ks_issue_ascale(saf, kt_rt):
            sa_t = load_a_scale_tile(kt_rt)
            for sub in range_constexpr(kScaleSubBlocks):
                saf[sub].store(Vec.from_elements([sa_t[sub]], Int32))

        def _ks_issue_scales(kt):
            slot = chunk_of[kt] % n_slots
            issue_bscale_into(bsf_slots[slot], scale_chunk_tile(fx.Int32(kt)))
            _ks_issue_ascale(saf_slots[slot], fx.Int32(kt))

        def _ks_prefetch(kt):
            issue_b_load_into(nxt_bqf, None, fx.Int32(kt))
            if const_expr(kt == 0 or chunk_of[kt] != chunk_of[kt - 1]):
                _ks_issue_scales(kt)

        issue_b_load_into(cur_bqf, None, fx.Int32(0))
        _ks_issue_scales(0)
        rocdl.sched_barrier(0)

        a_all_resident = const_expr((aStages if g2_apre else kStages) >= KT)
        if const_expr(a_all_resident):
            gpu.barrier()

        for kt in range_constexpr(KT):
            kt_rt = fx.Int32(kt)
            cur_bsf = bsf_slots[chunk_of[kt] % n_slots]
            if const_expr(g2_bhoist) and const_expr(kt + 1 < KT):
                _ks_prefetch(kt + 1)
            if const_expr(not a_all_resident):
                gpu.barrier()
            issue_a_ds_read(fx.Int32(kt % aStages))
            if const_expr(not a_all_resident and kt + kStages < KT):
                issue_a_load_lds(
                    fx.Int32((kt + kStages) % aStages), fx.Int32(kt + kStages)
                )
            cur_saf = saf_slots[chunk_of[kt] % n_slots]
            sa = [
                Vec(cur_saf[sub].load())[0]
                for sub in range_constexpr(kScaleSubBlocks)
            ]
            _il = epi_thunks if const_expr(kt == KT - 1) else None
            if const_expr(g2_interleave and kt == KT - 1):
                rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                gpu.barrier()
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt, interleave=_il)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            cur_bqf, nxt_bqf = nxt_bqf, cur_bqf
    if const_expr(g2_interleave):
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        _epilog(None, lds_ready=True, prefetched_ids=prefetched_ids)
    else:
        _epilog(
            [[c_frags[i][J].load() for J in range(numAccN)] for i in range(kMChunks)]
        )


# ---- Atomic bf16 epilogue (shared store path; gemm2 down-proj) ----
def atomic_bf16_epilog(
    lds_acc_base,
    accm,
    arg_out,
    arg_stids,
    arg_sweights,
    arg_bias,
    expert_id,
    m_row,
    n_block_idx,
    wave,
    lane,
    i32_M,
    BM,
    N_OUT,
    *,
    BN=256,
    use_reduce=False,
    topk=1,
    SBM=None,
    g2_bf16_lds=False,
    route_out_fp8=False,
    g2_defer_weight=0,
    g2_out_pitch_align=0,
    g2_scale_blk=8,
    g2_epi_lanes=None,
    emit_thunks=None,
    lds_ready=False,
    enable_bias=False,
    prefetched_ids=None,
    output_n_base=0,
    output_width=None,
    reduce_store_cache_modifier=None,
    write_limit=None,
    row_limit=None,
    opaque_tid=False,
    sorted_rows=False,
    n_waves=4,
):
    if SBM is None:
        SBM = BM

    def opaque_thread_id():
        # Recompute epilogue lane arithmetic after the K loop rather than
        # keeping the prologue's address expressions live throughout it.
        return fx.Int32(llvm.inline_asm(
            T.i32, [_raw(fx.Int32(gpu.thread_id("x")))],
            "v_mov_b32 $0, $1", "=v,v", has_side_effects=True,
        ))

    EPI_LANES = _G2_EPI_LANES if g2_epi_lanes is None else int(g2_epi_lanes)
    EPI_ROWS = 64 * n_waves // EPI_LANES
    assert EPI_ROWS * EPI_LANES == 64 * n_waves and BM % EPI_ROWS == 0, (EPI_LANES, n_waves)
    M_REPS = BM // EPI_ROWS
    ROUTE_VEC = BN // EPI_LANES
    bf16_src = const_expr(bool(g2_bf16_lds) and bool(route_out_fp8))
    numAccN = (BN // n_waves) // 16  # 16-column MFMA subblocks per wave
    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16
    lds_base_fptr = lds_typed_ptr(lds_acc_base, T.f32)
    lds_base_bf16 = (
        lds_typed_ptr(lds_acc_base, T.bf16, align=2)
        if const_expr(g2_bf16_lds)
        else None
    )

    tx_i32 = opaque_thread_id() if const_expr(opaque_tid and lds_ready) else fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // EPI_LANES
    n_lane = tx_i32 % EPI_LANES
    if reduce_store_cache_modifier is not None and (
        not use_reduce or route_out_fp8 or enable_bias or g2_defer_weight
    ):
        raise ValueError("custom BF16 store policies require unweighted reduce output")
    output_width = N_OUT if const_expr(output_width is None) else fx.Int32(output_width)
    output_n_base = fx.Int32(output_n_base)
    store_vec = 8 if reduce_store_cache_modifier is not None else 2
    store_group_n = EPI_LANES * store_vec
    col_start = n_lane * store_vec
    wave_n = BN // n_waves
    assert BN % (EPI_LANES * 2) == 0, (BN, EPI_LANES)

    def flat_buffer(arg, elem_ty, align):
        ptr = global_typed_ptr(arg, elem_ty, align=align)
        view = fx.Tensor(fx.make_view(ptr, fx.make_layout((1, 1), (1, 1))))
        return fx.rocdl.make_buffer_tensor(view, max_size=True)

    stids = flat_buffer(arg_stids, T.i32, 4)
    sweights = flat_buffer(arg_sweights, T.f32, 4)
    bias_f32 = None
    out_bf16 = flat_buffer(arg_out, T.bf16, 4)
    out_bf16_ptr = global_typed_ptr(arg_out, T.bf16, align=2)
    out_i8_ptr = global_typed_ptr(arg_out, T.i8, align=1)

    load_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Int32)
    load_f32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), Float32)
    atomic_bf16x2 = fx.make_copy_atom(fx.rocdl.BufferAtomicPkAdd(BFloat16), BFloat16)
    reduce_bf16x8 = (
        fx.make_copy_atom(
            fx.rocdl.BufferCopy128b(reduce_store_cache_modifier), BFloat16
        )
        if reduce_store_cache_modifier is not None
        else None
    )

    def load_scalar(atom, src, index, elem_ty):
        frag = fx.make_rmem_tensor(1, elem_ty)
        fx.copy(atom, src[None, index], frag)
        return frag.load()[0]


    defer_w = bool(g2_defer_weight)

    # Prefetch sorted_token_ids / sorted_weights (invariant); latency overlaps stores+barriers.
    packed = [] if const_expr(prefetched_ids is None) else prefetched_ids
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + mr * EPI_ROWS + m_lane
        if const_expr(prefetched_ids is None):
            packed.append(load_scalar(load_i32, stids, sorted_pos, Int32))
        if const_expr(not defer_w):
            weight.append(load_scalar(load_f32, sweights, sorted_pos, Float32))

    if const_expr(not lds_ready):
        if const_expr(emit_thunks is None):
            gpu.barrier()
        fresh_lane = {}

        def _write_i(i, only_j=None):
            if const_expr(opaque_tid):
                if const_expr("lane" not in fresh_lane):
                    tx = opaque_thread_id()
                    fresh_lane["lane"] = ((tx % 64) // 16, tx % 16)
                ld16, lm16 = fresh_lane["lane"]
            else:
                ld16, lm16 = lane_div_16, lane_mod_16
            row_base = fx.Int32(i * 16) + ld16 * 4
            w_row = (
                None
                if const_expr(defer_w)
                else [
                    load_scalar(load_f32, sweights, m_row + row_base + v, Float32)
                    for v in range_constexpr(4)
                ]
            )
            for J in range_constexpr(numAccN):
                if const_expr(only_j is not None and J != only_j):
                    continue
                col = wave * wave_n + J * 16 + lm16
                vec = (
                    Vec(accm[i][J].load())
                    if const_expr(emit_thunks is not None)
                    else Vec(accm[i][J])
                )

                def _scaled(v, vec=vec):
                    f = fx.Float32(vec[v])
                    return f if const_expr(defer_w) else f * fx.Float32(w_row[v])

                for v0 in range_constexpr(0, 4, 2):
                    pk = Vec.from_elements(
                        [_scaled(v0), _scaled(v0 + 1)], Float32
                    ).to(BFloat16)
                    for h in range_constexpr(2):
                        lds_base_bf16[(row_base + v0 + h) * BN + col] = pk[h]

        if const_expr(emit_thunks is not None):
            for J in range_constexpr(numAccN):
                emit_thunks.append(
                    lambda J=J: [_write_i(i, J) for i in range_constexpr(BM // 16)]
                )
        else:
            for i in range_constexpr(BM // 16):
                _write_i(i)
        if const_expr(emit_thunks is not None):
            return
        gpu.barrier()

    def store_one_mr(mr):
        row_in_block = fx.Int32(mr * EPI_ROWS) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if const_expr(sorted_rows):
            out_row = fx.Int64(m_row + fx.Int32(mr * EPI_ROWS) + m_lane)
            row_base_addr = out_row * fx.Int64(N_OUT) + fx.Int64(n_block_idx * BN + col_start)
        elif const_expr(use_reduce):
            # reduce out_row can reach tokens*topk (large-M) so compute the element base in i64 (atomic i32 path byte-identical).
            out_row = fx.Int64(token_id * fx.Int32(topk) + (packed[mr] >> fx.Int32(24)))
            row_base_addr = out_row * fx.Int64(output_width) + fx.Int64(
                n_block_idx * BN + col_start - output_n_base
            )
        else:
            out_row = token_id
            row_base_addr = out_row * N_OUT + n_block_idx * BN + col_start
        if const_expr(reduce_store_cache_modifier is not None):
            for s in range_constexpr(BN // store_group_n):
                idx0 = row_in_block * BN + col_start + s * store_group_n
                pk = Vec(
                    lds_vec_load(
                        lds_acc_base,
                        idx0 * 2,
                        Vec.make_type(store_vec, BFloat16),
                        BFloat16,
                        align=16,
                    )
                )
                out_off = row_base_addr + fx.Int64(s * store_group_n)
                out_frag = fx.make_rmem_tensor(store_vec, BFloat16)
                out_frag.store(pk)
                fx.copy(reduce_bf16x8, out_frag, out_bf16[None, out_off])
        else:
            for s in range_constexpr(BN // store_group_n):
                pk = lds_pre[mr][s]
                out_frag = fx.make_rmem_tensor(store_vec, BFloat16)
                out_frag.store(pk)
                out_off = row_base_addr + fx.Int64(s * store_group_n)
                if const_expr(use_reduce):
                    fx.ptr_store(pk, out_bf16_ptr + out_off)
                else:
                    fx.copy(atomic_bf16x2, out_frag, out_bf16[None, out_off])

    def lds_pk_load(mr, s):
        # adjacent ee=0,1 contiguous -> one 2-wide load.
        idx0 = (fx.Int32(mr * EPI_ROWS) + m_lane) * BN + col_start + s * store_group_n
        return Vec(
            lds_vec_load(
                lds_acc_base,
                idx0 * 2,
                Vec.make_type(store_vec, BFloat16),
                BFloat16,
                align=4,
            )
        )

    # Read every row group's C from LDS before the per-row validity branches, so
    # the LDS latency overlaps instead of serializing read -> wait -> store per row.
    lds_pre = None
    if const_expr(
        not (use_reduce and route_out_fp8) and reduce_store_cache_modifier is None
    ):
        lds_pre = [
            [lds_pk_load(mr, s) for s in range_constexpr(BN // store_group_n)]
            for mr in range_constexpr(M_REPS)
        ]

    if const_expr(prefetched_ids is not None or (use_reduce and route_out_fp8)):
        rocdl.s_waitcnt(vmcnt=0)

    for mr in range_constexpr(M_REPS):
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        route_slot = packed[mr] >> fx.Int32(24)

        @flyc.jit
        def store_if_valid(token_id, route_slot, mr):
            token_limit = i32_M if const_expr(write_limit is None) else write_limit
            valid = token_id < token_limit
            if const_expr(row_limit is not None):
                # Rows past row_limit belong to the next sort block.
                valid = valid & (fx.Int32(mr * EPI_ROWS) + m_lane < row_limit)
            if const_expr(use_reduce):
                valid = valid & (route_slot < fx.Int32(topk))
            if valid:
                store_one_mr(mr)

        store_if_valid(token_id, route_slot, mr)


# ---------------------------------------------------------------------------
# Dedicated EP32 A8W4 GEMM2 launcher (Kimi-K3: H=3584, I=3072, sort block 64,
# atomic bf16 output). Forked from mxmoe_dispatcher.compile_gemm2_a4w4_port.
#
# Tiles, per 64-row M block (padding sits at the end of each expert run): a block
# whose rows 16..63 are padding and whose previous block has the same expert is
# folded into that block's tile (80 rows, one weight stream); its own tiles are
# skipped. Every tile runs the same 80-row body; tiles that fold nothing store
# only their own 64 rows.
# Scheduling: the host upper-bound grid is ~20x the real tile count and the
# hardware dispatcher spreads the real tiles unevenly over CUs. Instead the grid
# is fixed at GRID_WGS (2 per CU at the 2-waves/EU register budget) and every
# workgroup derives the compact list of real tiles from a wave ballot over the
# sort blocks, then strides over it, so all tiles start together.
# ---------------------------------------------------------------------------
def _compile_gemm2_ep32_fixed(
    *,
    use_nt=False,
    HIDDEN_MAX=8192,
    INTER_MAX=3072,
    topk=1,
    use_valid_token_count=True,
    GRID_WGS=512,
    SPART=404,
    PRUNE=False,
    BM=64,
    B_NT=False,
    REDUCE=False,
    NATIVE_C=False,
):
    from .mxmoe_dispatcher import _spart_output_tile_index

    if BM != 64:
        raise ValueError(f"EP32 GEMM2 supports BM64, got {BM}")
    BLK, BN, BK = BM, 256, 128
    a_dtype, b_dtype = "fp8", "fp4"
    KH_TILE_A = BK
    ROWS = BM + 16  # Retain the existing short-tail fold for BM64.
    aStages = 3
    use_b_lds = not use_nt and INTER_MAX == 3072
    lds_bytes = max(ROWS * BN * 2, aStages * ROWS * KH_TILE_A + (3 * BN * BK // 2 if use_b_lds else 0))
    WAVES_PER_EU = 2
    name = (
        f"gemm2_ep32_a8w4_hmax{HIDDEN_MAX}_imax{INTER_MAX}_g{GRID_WGS}{f'_sp{SPART}' if SPART else ''}{'_pr' if PRUNE else ''}"
        f"{'_nt' if use_nt else ''}{'_valid_tokens' if use_valid_token_count else ''}{'_blds2' if use_b_lds else ''}{'_bnt' if B_NT else ''}{'_nativef32' if NATIVE_C else ('_store128' if REDUCE else '')}"
    )

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    @flyc.jit
    def _run_rows(
        arg_aq,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_eids,
        arg_stids,
        arg_sweights,
        arg_bias,
        arg_out,
        m_row,
        n_block_idx,
        m_block_idx,
        lane,
        wave,
        i32_M,
        i32_max_m_blocks,
        i32_inter,
        i32_hidden,
        lds_base_i32,
        write_limit,
        e_cur,
        ROW_LIMIT,
    ):
        k_bytes = i32_inter
        aq_num = fx.Int64(i32_max_m_blocks) * fx.Int64(BLK) * fx.Int64(k_bytes)
        for slot in range_constexpr(0 if use_b_lds else kStages):
            issue_a_load_lds_dt(
                arg_aq,
                aq_num,
                lds_base_i32,
                slot,
                slot,
                m_row,
                wave,
                lane,
                True,
                KH_TILE_A,
                k_bytes,
                BM=ROWS,
            )
        rocdl.sched_barrier(0)
        gemm2_body_v2(
            lds_base_i32,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_stids,
            arg_sweights,
            arg_bias,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            fx.Int32(0),
            lane,
            wave,
            arg_aq,
            i32_inter,
            i32_hidden,
            BM=ROWS,
            BN=BN,
            BK=BK,
            use_nt=use_nt,
            INTER_MAX=INTER_MAX,
            aStages=aStages,
            a_slot_alias=False,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            use_reduce=REDUCE,
            topk=topk,
            SBM=BLK,
            g2_bhoist=True,
            g2_ascale_pf=True,
            g2_bf16_lds=True,
            g2_defer_weight=False,
            g2_out_pitch_align=0,
            g2_scale_blk=8,
            route_out_fp8=False,
            g2_epi_lanes=None,
            g2_apre=False,
            enable_bias=False,
            nonatomic=False,
            scatter_out_fp4=False,
            arg_out_scale=arg_out,
            g2_prefetch_ids=False,
            mn_idx=(m_block_idx, n_block_idx),
            write_limit=write_limit,
            m_row_in=m_row,
            BLK=BLK,
            row_limit=ROW_LIMIT,
            e_in=e_cur,
            prune=PRUNE,
            b_lds=use_b_lds,
            b_nt=B_NT,
            sorted_rows=REDUCE,
            native_c=NATIVE_C,
            native_variant=False,
            reduce_store_cache_modifier=0 if REDUCE else None,
        )

    @flyc.kernel(name=name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        arg_bias: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32,
        i32_hidden: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        i32_grid_blocks: fx.Int32,
    ):
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % 64
        wave = rocdl.readfirstlane(T.i32, tx_i32 // 64)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))
        num_n_blocks = _udiv(i32_hidden, BN)
        cumsum = global_typed_ptr(arg_cumsum, T.i32)
        total_m_blocks = _udiv(cumsum[0], BLK)
        valid_tokens = cumsum[1]
        write_limit = i32_M
        if const_expr(use_valid_token_count):
            write_limit = (valid_tokens < i32_M).select(valid_tokens, i32_M)
        eids = global_typed_ptr(arg_eids, T.i32)
        stids = global_typed_ptr(arg_stids, T.i32)
        one, zero = fx.Int32(1), fx.Int32(0)
        mask24 = fx.Int32(0x00FFFFFF)
        # Units = M blocks not folded into the previous block's tile, found
        # with a wave ballot over the first 128 blocks (later blocks are
        # appended as units unchanged; the per-tile mode check skips any that
        # turn out to be folded). Tiles are indexed compactly so the fixed
        # grid covers exactly the real work.
        u_lane = lane
        u_masks = []
        for c in range_constexpr(2):
            b = u_lane + fx.Int32(c * 64)
            b_ok = b < total_m_blocks
            bc = b_ok.select(b, zero)
            bp = (bc > zero).select(bc - one, bc)
            t16 = stids[bc * fx.Int32(BLK) + fx.Int32(16)] & mask24
            folded = (bc > zero).select(
                (eids[bc] == eids[bp]).select(
                    (t16 >= valid_tokens).select(one, zero), zero
                ),
                zero,
            )
            is_unit = b_ok.select((folded == zero).select(one, zero), zero)
            u_masks.append(
                fx.Int64(rocdl.ballot(T.i64, _raw(is_unit == one)))
            )
        u_cnt0 = fx.Int32(fx.math.ctpop(u_masks[0]))
        u_cnt1 = fx.Int32(fx.math.ctpop(u_masks[1]))
        extra = (total_m_blocks > fx.Int32(128)).select(
            total_m_blocks - fx.Int32(128), zero
        )
        n_units = u_cnt0 + u_cnt1 + extra
        bound = n_units * num_n_blocks
        for iv in range(bx_i32, bound, gpu.grid_dim.x):
            gpu.barrier()  # LDS of the previous tile's epilogue
            pid = fx.Int32(iv)
            if const_expr(SPART > 0):
                # Spatially local (unit, N block) order over the compact units.
                u, n_blk = _spart_output_tile_index(
                    pid, n_units, num_n_blocks, SPART // 100, SPART % 100
                )
            else:
                u = pid // num_n_blocks
                n_blk = pid - u * num_n_blocks
            in0 = u < u_cnt0
            k = in0.select(u, u - u_cnt0)
            msk = in0.select(u_masks[0], u_masks[1])
            lane64 = fx.Int64(u_lane)
            below = msk & ((fx.Int64(1) << lane64) - fx.Int64(1))
            bit = (msk >> lane64) & fx.Int64(1)
            rank = fx.Int32(fx.math.ctpop(below))
            hit = (bit == fx.Int64(1)).select((rank == k).select(one, zero), zero)
            sel = fx.Int64(rocdl.ballot(T.i64, _raw(hit == one)))
            pos = fx.Int32(fx.math.cttz(sel))
            m_from_mask = in0.select(pos, pos + fx.Int32(64))
            m_blk = (u < u_cnt0 + u_cnt1).select(
                m_from_mask, fx.Int32(128) + u - u_cnt0 - u_cnt1
            )
            m_blk = fx.Int32(rocdl.readfirstlane(T.i32, _raw(m_blk)))
            m_prev = (m_blk > zero).select(m_blk - one, m_blk)
            has_next = m_blk + one < total_m_blocks
            m_next = has_next.select(m_blk + one, m_blk)

            e_cur = eids[m_blk]
            e_prev = eids[m_prev]
            e_next = eids[m_next]
            # Row 16 of a block is padding iff the block holds <= 16 real rows.
            tok16 = stids[m_blk * fx.Int32(BLK) + fx.Int32(16)] & mask24
            tokn16 = stids[m_next * fx.Int32(BLK) + fx.Int32(16)] & mask24
            e_cur = fx.Int32(rocdl.readfirstlane(T.i32, _raw(e_cur)))
            light = (tok16 >= valid_tokens).select(one, zero)
            prev_same = (m_blk > zero).select((e_prev == e_cur).select(one, zero), zero)
            next_light = has_next.select(
                (e_next == e_cur).select((tokn16 >= valid_tokens).select(one, zero), zero),
                zero,
            )
            mode = (light * prev_same == one).select(
                fx.Int32(3), (next_light * (one - light) == one).select(fx.Int32(2), light)
            )
            mode = fx.Int32(rocdl.readfirstlane(T.i32, _raw(mode)))
            m_row = m_blk * fx.Int32(BLK)
            args = (
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_bias,
                arg_out,
                m_row,
                n_blk,
                m_blk,
                lane,
                wave,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                i32_hidden,
                lds_base_i32,
                write_limit,
                e_cur,
            )
            # One extended body for every tile (one body copy keeps register use
            # at the 2-waves/EU budget); tiles that do not fold a tail block keep
            # 64 rows: the epilogue skips rows >= ROW_LIMIT.
            if mode < fx.Int32(3):
                _run_rows(
                    *args,
                    (mode == fx.Int32(2)).select(fx.Int32(64 + 16), fx.Int32(64)),
                )


    @flyc.jit
    def launch_gemm2(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        arg_bias: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid_blocks: fx.Int32,
        i32_inter: fx.Int32,
        i32_hidden: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        num_n_blocks = fx.Int32(fx.Uint32(i32_hidden) // fx.Uint32(BN))
        grid_x = fx.Int32(GRID_WGS)
        gemm2_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            arg_bias,
            i32_M,
            i32_max_m_blocks,
            i32_inter,
            i32_hidden,
            arg_out,
            arg_out_scale,
            i32_grid_blocks,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    launch_gemm2.compile_hints = {"waves_per_eu": WAVES_PER_EU}
    return launch_gemm2


def _compile_gemm2_ep32_native(
    *,
    use_nt=False,
    HIDDEN_MAX=8192,
    INTER_MAX=3072,
    topk=1,
    use_valid_token_count=True,
    GRID_WGS=512,
    SPART=404,
    PRUNE=False,
    BM=64,
    B_NT=False,
    REDUCE=False,
    NATIVE_C=False,
    NATIVE_BF16=False,
    NW=8,
    CPW=64,
    EPI_L=32,
    WPE=2,
    BSTAGES=None,
    VARWIDTH=True,
):
    from .mxmoe_dispatcher import _spart_output_tile_index

    if BM != 96:
        raise ValueError(f"EP32 GEMM2 native output supports BM96, got {BM}")
    if NATIVE_BF16 and not (NATIVE_C and BM == 96):
        raise ValueError("weighted native BF16 staging requires BM96 native output")
    if NATIVE_BF16:
        NW, CPW, VARWIDTH = 7, 64, False
        GRID_WGS = min(GRID_WGS, 256)
    assert not VARWIDTH or (NATIVE_C and NW == 8 and CPW == 64)
    if BSTAGES is None:
        BSTAGES = 3
    BLK, BN, BK = BM, NW * CPW, 128
    a_dtype, b_dtype = "fp8", "fp4"
    KH_TILE_A = BK
    ROWS = BM
    aStages = 3
    use_b_lds = not use_nt and INTER_MAX == 3072
    lds_bytes = max(ROWS * BN * 2, aStages * ROWS * KH_TILE_A + (BSTAGES * BN * BK // 2 if use_b_lds else 0))
    assert lds_bytes <= 160 * 1024, f"EP32 GEMM2 LDS exceeds gfx950 capacity: {lds_bytes}"
    WAVES_PER_EU = WPE
    name = (
        f"gemm2_ep32_a8w4_hmax{HIDDEN_MAX}_imax{INTER_MAX}_g{GRID_WGS}{f'_sp{SPART}' if SPART else ''}{'_pr' if PRUNE else ''}"
        f"{'_nt' if use_nt else ''}{'_valid_tokens' if use_valid_token_count else ''}{'_blds2' if use_b_lds else ''}{f'_bm{BM}' if BM == 96 else ''}{'_bnt' if B_NT else ''}{'_nativef32' if NATIVE_C else ('_store128' if REDUCE else '')}"
    )

    name += f'_bn{BN}w{NW}_bs{BSTAGES}' + ('_var384x8_512' if VARWIDTH else '')
    if NATIVE_BF16:
        name += '_nativebf16w'

    @fx.struct
    class SharedStorage:
        buf: fx.Array[Int8, lds_bytes, 16]

    @flyc.jit
    def _run_rows(
        arg_aq,
        arg_ascale,
        arg_bq,
        arg_bscale,
        arg_eids,
        arg_stids,
        arg_sweights,
        arg_bias,
        arg_out,
        m_row,
        n_block_idx,
        m_block_idx,
        lane,
        wave,
        i32_M,
        i32_max_m_blocks,
        i32_inter,
        i32_hidden,
        lds_base_i32,
        write_limit,
        e_cur,
        ROW_LIMIT,
    ):
        col_start = fx.Int32(rocdl.readfirstlane(T.i32, _raw(n_block_idx * fx.Int32(384))))
        live_cols = (n_block_idx == fx.Int32(8)).select(fx.Int32(512), fx.Int32(384))
        k_bytes = i32_inter
        aq_num = fx.Int64(i32_max_m_blocks) * fx.Int64(BLK) * fx.Int64(k_bytes)
        for slot in range_constexpr(0 if use_b_lds else kStages):
            issue_a_load_lds_dt(
                arg_aq,
                aq_num,
                lds_base_i32,
                slot,
                slot,
                m_row,
                wave,
                lane,
                True,
                KH_TILE_A,
                k_bytes,
                BM=ROWS,
            )
        rocdl.sched_barrier(0)
        gemm2_body_v2(
            lds_base_i32,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_stids,
            arg_sweights,
            arg_bias,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            fx.Int32(0),
            lane,
            wave,
            arg_aq,
            i32_inter,
            i32_hidden,
            BM=ROWS,
            BN=BN,
            BK=BK,
            use_nt=use_nt,
            INTER_MAX=INTER_MAX,
            aStages=aStages,
            a_slot_alias=False,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            use_reduce=REDUCE,
            topk=topk,
            SBM=BLK,
            g2_bhoist=True,
            g2_ascale_pf=True,
            g2_bf16_lds=True,
            g2_defer_weight=False,
            g2_out_pitch_align=0,
            g2_scale_blk=8,
            route_out_fp8=False,
            g2_epi_lanes=EPI_L,
            g2_apre=False,
            enable_bias=False,
            nonatomic=False,
            scatter_out_fp4=False,
            arg_out_scale=arg_out,
            g2_prefetch_ids=False,
            mn_idx=(m_block_idx, n_block_idx),
            write_limit=write_limit,
            m_row_in=m_row,
            BLK=BLK,
            row_limit=ROW_LIMIT,
            e_in=e_cur,
            prune=PRUNE,
            b_lds=use_b_lds,
            b_nt=B_NT,
            sorted_rows=REDUCE,
            native_c=NATIVE_C,
            native_bf16=NATIVE_BF16,
            n_waves=NW,
            b_stages=BSTAGES,
            n_col_in=col_start if VARWIDTH else None,
            n_live=live_cols if VARWIDTH else None,
            reduce_store_cache_modifier=0 if REDUCE else None,
        )


    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def gemm2_kernel_block(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        arg_bias: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_inter: fx.Int32,
        i32_hidden: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        i32_grid_blocks: fx.Int32,
    ):
        # BM96 sort blocks: every block is one full tile.
        tx_i32 = fx.thread_idx.x
        bx_i32 = fx.block_idx.x
        lane = tx_i32 % 64
        wave = rocdl.readfirstlane(T.i32, tx_i32 // 64)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base_i32 = fx.Int32(fx.ptrtoint(lds.buf.ptr))
        num_n_blocks = fx.Int32(9) if const_expr(VARWIDTH) else _udiv(i32_hidden, BN)
        cumsum = global_typed_ptr(arg_cumsum, T.i32)
        n_units = _udiv(cumsum[0], BLK)
        valid_tokens = cumsum[1]
        write_limit = i32_M
        if const_expr(use_valid_token_count):
            write_limit = (valid_tokens < i32_M).select(valid_tokens, i32_M)
        eids = global_typed_ptr(arg_eids, T.i32)
        bound = n_units * num_n_blocks
        for iv in range(bx_i32, bound, gpu.grid_dim.x):
            gpu.barrier()  # LDS of the previous tile's epilogue
            pid = fx.Int32(iv)
            if const_expr(SPART > 0):
                m_blk, n_blk = _spart_output_tile_index(
                    pid, n_units, num_n_blocks, SPART // 100, SPART % 100
                )
            else:
                m_blk = pid // num_n_blocks
                n_blk = pid - m_blk * num_n_blocks
            m_blk = fx.Int32(rocdl.readfirstlane(T.i32, _raw(m_blk)))
            e_cur = fx.Int32(rocdl.readfirstlane(T.i32, _raw(eids[m_blk])))
            _run_rows(
                arg_aq,
                arg_ascale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_stids,
                arg_sweights,
                arg_bias,
                arg_out,
                m_blk * fx.Int32(BLK),
                n_blk,
                m_blk,
                lane,
                wave,
                i32_M,
                i32_max_m_blocks,
                i32_inter,
                i32_hidden,
                lds_base_i32,
                write_limit,
                e_cur,
                fx.Int32(BLK),
            )

    @flyc.jit
    def launch_gemm2(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        arg_bias: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        i32_grid_blocks: fx.Int32,
        i32_inter: fx.Int32,
        i32_hidden: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        num_n_blocks = fx.Int32(fx.Uint32(i32_hidden) // fx.Uint32(BN))
        grid_x = fx.Int32(GRID_WGS)
        gemm2_kernel_block(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            arg_bias,
            i32_M,
            i32_max_m_blocks,
            i32_inter,
            i32_hidden,
            arg_out,
            arg_out_scale,
            i32_grid_blocks,
        ).launch(grid=(grid_x, 1, 1), block=(64 * NW, 1, 1), stream=stream)

    launch_gemm2.compile_hints = {"waves_per_eu": WAVES_PER_EU}
    return launch_gemm2


def compile_gemm2_ep32(*, BM=64, variant="base", **kwargs):
    """Compile the fixed BM64 or native BM96 EP32 GEMM2 launcher."""
    if variant in ("native", "var384"):
        return _compile_gemm2_ep32_native(BM=BM, **kwargs)
    return _compile_gemm2_ep32_fixed(BM=BM, **kwargs)

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""GEMM2 compute for fused MegaMoE v2 stage2."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    Float4E2M1FN,
    Float8E4M3FN,
    Float32,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec
from flydsl.expr.typing import as_ir_value as _raw

from ..mxfp4_gemm_common import _lds_swizzle_mask as lds_swizzle_mask
from ..mxfp4_gemm_common import (
    flat_buffer_view,
    global_typed_ptr,
    kBS_stride_k0_dw,
    kStages,
    lds_dma_atom_128,
    lds_dma_dst,
    lds_swizzle_mask_f8,
    lds_vec_load,
)
from ..tensor_shim import buf_copy_load, ptr_buf_tensor
from .gemm_util import wait_lds_barrier


def scale_view(
    arg_scale, base_dw, K_TILES_TOTAL, k0_stride_dw=64, num_records_bytes=None
):
    """View one e8m0 scale word, optionally bounded to the real buffer extent."""
    base_dw = rocdl.readfirstlane(T.i32, base_dw)
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
    """Run one scaled-MFMA cluster over a 32-row A-scale group."""
    if const_expr(single_rg):
        steps = tuple(
            (2 * k + rg_off, k, i0, bq_frags_kt[J][k]) for k in range(k_halves)
        )
    else:
        steps = tuple(
            (2 * k + im, k, i0 + im, bq_frags_kt[J][k])
            for k in range(k_halves)
            for im in range(2)
        )
    for osa, k, i, bJ in steps:
        osb = 2 * k + in_b
        fx.gemm(
            atoms[(osa, osb)],
            c_frags[i][J],
            a_frags[i][k],
            bJ,
            c_frags[i][J],
            scale_a=sa,
            scale_b=sb,
        )


def issue_a_load_lds_dt(
    arg_aq,
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
    valid_rows=None,
    NW=4,
    slot_rows=None,
):
    """Load one A tile through a tile-local descriptor so allocations may span the 4 GiB buffer ABI.

    ``valid_rows`` bounds the descriptor to the tile's real rows (the rest read as zero);
    ``slot_rows`` (default ``BM``) is the row count an LDS slot holds.
    """
    lanes_per_row = KH_TILE_A // 16  # 8 (fp4) / 16 (fp8)
    rows_per_call = 64 // lanes_per_row  # 8 (fp4) / 4 (fp8)
    a_lane_row = lane // lanes_per_row
    rows_per_wave = BM // NW  # rows each wave loads (NW4: BM32 8, BM64 16; NW8: BM64 8)
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
    tile_base = fx.Int64(arg_aq) + fx.Int64(m_row) * fx.Int64(K_BYTES)
    src = flat_buffer_view(
        tile_base,
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=fx.Int64(BM if valid_rows is None else valid_rows)
        * fx.Int64(K_BYTES),
    )
    for g in range_constexpr(n_row_groups):
        lds_row = gather_base_row + g * rows_per_call
        mask = (
            lds_swizzle_mask_f8(lds_row + a_lane_row, KH_TILE_A)
            if const_expr(is_f8)
            else lds_swizzle_mask(lds_row + a_lane_row)
        )
        tile_row = lds_row + a_lane_row
        voffset = (lane_col ^ mask) + tile_row * K_BYTES
        off = fx.Int32(slot * ((BM if slot_rows is None else slot_rows) * KH_TILE_A)) + lds_row * KH_TILE_A
        v_e = (voffset + kt * KH_TILE_A) // 4  # per-lane i32-elem index
        fx.copy(
            atom, src[v_e, None], lds_dma_dst(base_i32, off, elem_ty=T.i32, align=16)
        )


# fmt: off
def issue_a_fold_rows_dt(arg_aq, s_aq_base, slot, kt, m_row, wave, lane, KH_TILE_A, K_BYTES, valid_rows, *, BM,
    slot_rows):
# fmt: on
    """fp8 A rows BM..BM+15 of a folded GEMM2 unit: each wave copies 4 rows (waves 4-7 repeat waves 0-3); rows past
    valid_rows are out of bounds (no memory traffic), so every unit issues the same copies."""
    lanes_per_row = KH_TILE_A // 16
    lds_row = BM + (wave % fx.Int32(4)) * fx.Int32(64 // lanes_per_row)
    tile_row = lds_row + lane // lanes_per_row
    src = flat_buffer_view(fx.Int64(arg_aq) + fx.Int64(m_row) * fx.Int64(K_BYTES), None, T.i32, align=16,
        elem_bytes=4, fold=False, num_records_bytes=fx.Int64(valid_rows) * fx.Int64(K_BYTES))
    voffset = (((lane % lanes_per_row) * 16) ^ lds_swizzle_mask_f8(tile_row, KH_TILE_A)) + tile_row * K_BYTES
    off = fx.Int32(slot * (slot_rows * KH_TILE_A)) + lds_row * fx.Int32(KH_TILE_A)
    fx.copy(lds_dma_atom_128(), src[(voffset + kt * KH_TILE_A) // 4, None],
        lds_dma_dst(s_aq_base, off, elem_ty=T.i32, align=16))


def issue_a_scale_lds_dt(arg_ascale, s_aq_base, lds_off, m_row, i32_inter, wave, lane, *, INTER_MAX, NW, chunks=2):
    """Copy a 64-row m-block's A scales (two 32-row chunks x every K chunk, contiguous) to LDS at lds_off."""
    kc_dw = fx.Int32(i32_inter) // fx.Int32(256) * fx.Int32(64)
    src = flat_buffer_view(
        fx.Int64(arg_ascale) + fx.Int64((m_row // fx.Int32(32)) * kc_dw) * fx.Int64(4),
        None,
        T.i32,
        align=16,
        elem_bytes=4,
        fold=False,
        num_records_bytes=fx.Int64(kc_dw * fx.Int32(4) * chunks),
    )
    for r in range_constexpr(-(-2 * (INTER_MAX // 256) * 256 // (NW * 1024))):
        blk = fx.Int32(r * NW) + wave
        fx.copy(
            lds_dma_atom_128(),
            src[(blk * fx.Int32(64) + lane) * fx.Int32(4), None],
            lds_dma_dst(s_aq_base, fx.Int32(lds_off) + blk * fx.Int32(1024), elem_ty=T.i32, align=16),
        )


@flyc.jit
def gemm2_compute_v2(
    lds_base_i32,
    arg_ascale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_aq,
    i32_max_m_blocks,
    bx_i32,
    lane,
    wave,
    i32_inter,
    i32_hidden,
    i32_kpad,
    i32_npad,
    *,
    BM,
    BN=256,
    BK=256,
    use_nt,
    INTER_MAX,
    aStages,
    a_dtype,
    has_pad=False,
    SBM=None,
    g2_bhoist=True,
    g2_ascale_pf=True,
    g2_b2stage=False,
    g2_deep_a_pipeline=False,
    expert_offset=0,
    explicit_m_row=None,
    explicit_n_block=None,
    explicit_expert=None,
    valid_rows=None,
    NW=4,
    early_b_load=False,
    a_dma_uniform=False,
    a_scale_lds=False,
    fold_rows=0,
    fold=None,
):
    """Run GEMM2, optionally using an explicitly selected expert row/tile.

    ``NW`` is the CTA wave count, ``valid_rows`` bounds the A tile loads to the
    tile's real rows, and ``early_b_load`` (1-stage B path) issues each K-tile's
    B and A-scale loads before the barrier and the next A-tile DMA.

    The fused large-input kernel's BM64 1-stage knobs: ``a_dma_uniform`` issues
    the A copy on every K-step, ``a_scale_lds`` reads the A scales from LDS
    (copied there by issue_a_scale_lds_dt), and ``fold_rows`` (16, with
    a_scale_lds) makes the A slots hold BM + fold_rows rows: when the runtime
    flag ``fold`` is set, the unit also computes rows BM..BM+15 (its expert's
    next m-block, a tail of at most 16 rows), whose accumulators come back as
    one more row group.
    """
    assert aStages == (4 if g2_deep_a_pipeline else kStages + 1)
    assert not g2_deep_a_pipeline or g2_b2stage
    # SBM is the sort padding unit; BM is the compute tile and must divide SBM.
    if SBM is None:
        SBM = BM
    kMChunks = BM // 16  # 16-row MFMA row-groups
    kHalves = BK // 128  # 16x16x128 MFMA K-steps per K-tile
    tilesPerScaleChunk = 256 // BK  # K-tiles sharing one 256-K E8M0 word
    numAccN = (BN // NW) // 16  # 16-column MFMA subblocks per wave
    nPairs = max(1, numAccN // 2)  # one B-scale per two 16-column subblocks
    # BM16: single 16-row block owning a 32-row scale chunk (chunk==m_block_idx, rg0-only).
    is_bm16 = BM < 32
    rg_off = 0
    kScaleSubBlocks = max(1, kMChunks // 2)
    is_f8_a = a_dtype == "fp8"  # only the A path differs
    a_pack = 1 if is_f8_a else 2
    KH_TILE_A = BK // a_pack
    slot_rows = BM + fold_rows
    slot_bytes = slot_rows * KH_TILE_A
    # Contraction K = inter_dim runtime (i32_inter); INTER_MAX caps compile-time view/fragment bounds.
    K_rt = fx.Int32(i32_inter)
    K_BYTES = K_rt // fx.Int32(a_pack)  # A row stride bytes (runtime)
    kc_rt = K_rt // 256  # (K//32)//4//2
    K_TILES_RT = K_rt // fx.Int32(BK)  # runtime K-tile trip count
    kAS_per_chunk_dw = kc_rt * 64
    kBS_stride_n0_dw = kc_rt * 64
    # N_OUT = model_dim/hidden is the gemm2 output N dim; runtime via i32_hidden (no K-loop dependency).
    N_OUT_rt = fx.Int32(i32_hidden)
    kbs_per_expert_dw = (N_OUT_rt // 32) * kBS_stride_n0_dw  # (N_OUT//16//2)*stride
    num_n_blocks = N_OUT_rt // fx.Int32(BN)
    KH4 = K_rt // 8  # i32 col stride (= K_HALF//4)
    K_SCALE_CHUNKS_MAX = INTER_MAX // 256
    kFenceEvery = (aStages - kStages) if g2_deep_a_pipeline else 1
    assert kFenceEvery >= 1

    # Padded shapes mask weight tiles beyond the real K/N extents.
    N_real = None
    if const_expr(has_pad):
        K_real = K_rt - fx.Int32(i32_kpad)
        halves_real = (K_real + 127) // 128
        N_real = N_OUT_rt - fx.Int32(i32_npad)

    # Map each compute block to its SBM-padded expert metadata row.
    if const_expr(explicit_m_row is not None):
        m_row = fx.Int32(explicit_m_row)
        m_block_idx = m_row // fx.Int32(BM)
        n_block_idx = fx.Int32(explicit_n_block)
        e = fx.Int32(explicit_expert)
    else:
        m_block_idx = bx_i32 // num_n_blocks
        n_block_idx = bx_i32 - m_block_idx * num_n_blocks
        eids_ptr = global_typed_ptr(arg_eids, T.i32)
        if const_expr(SBM == BM):
            e = rocdl.readfirstlane(T.i32, eids_ptr[m_block_idx])
            m_row = m_block_idx * BM
        else:
            m_row = m_block_idx * BM
            e = rocdl.readfirstlane(T.i32, eids_ptr[m_row // fx.Int32(SBM)])
    if const_expr(expert_offset != 0):
        e = e - fx.Int32(expert_offset)

    lane_div_16 = lane // 16
    lane_mod_16 = lane % 16

    s_aq_base = lds_base_i32
    mma_atoms = scale_mma_atoms(a_dtype)

    # A activation: global->LDS DMA (issue_a_load_lds), then LDS->reg ds-read (issue_a_ds_read).
    A_NDW = (
        8 if is_f8_a else 4
    )  # fp8 packs two 128-K halves -> i32<8:1>; fp4 -> i32<4:1>
    a_frags = [
        [fx.make_rmem_tensor(A_NDW, Int32) for _ in range_constexpr(kHalves)]
        for _ in range_constexpr(kMChunks + fold_rows // 16)
    ]

    def issue_a_load_lds(slot, kt):
        issue_a_load_lds_dt(
            arg_aq,
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
            valid_rows=valid_rows,
            NW=NW,
            slot_rows=slot_rows,
        )
        if const_expr(fold_rows):
            issue_a_fold_rows_dt(arg_aq, s_aq_base, slot, kt, m_row, wave, lane, KH_TILE_A, K_BYTES,
                fold.select(fx.Int32(BM + fold_rows), fx.Int32(BM)), BM=BM, slot_rows=slot_rows)

    def issue_a_ds_read(slot, chunks=None):
        # A ds-read for one slot into a_frags: fp8 -> i32<8:1> (two 128-K halves), fp4 -> i32<4:1>.
        for k in range_constexpr(kHalves):
            for i in range_constexpr(kMChunks) if chunks is None else chunks:
                lds_row = lane_mod_16 + i * 16
                row_off = fx.Int32(slot * slot_bytes) + lds_row * KH_TILE_A
                if const_expr(is_f8_a):
                    mask = lds_swizzle_mask_f8(lane_mod_16, KH_TILE_A)
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

    # The shared e8m0 scale layout bounds each A-scale view to its remaining bytes.
    sc_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

    asc_per_mb = fx.Int32(kScaleSubBlocks) * kAS_per_chunk_dw * 4
    asc_num = fx.Int64(i32_max_m_blocks) * fx.Int64(asc_per_mb)
    scale_chunk0 = m_block_idx if const_expr(is_bm16) else m_row // 32

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

    if const_expr(a_scale_lds):
        # The m-block's A scales sit in LDS past the A slots (issue_a_scale_lds_dt). The reads are inline asm so
        # the compiler sees no LDS read that could alias the in-flight A copy (it would wait for that copy).
        assert BM == 64 and is_f8_a and kScaleSubBlocks == 2
        as_lds_off = aStages * slot_bytes
        # fold_rows: a third scale word (rows BM..BM+31, the folded tail) is read on every step (zeros unless folded).
        n_sc = 3 if fold_rows else 2
        v2i32 = ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * n_sc) + ")>")
        n_a_reads = kHalves * kMChunks * (2 if is_f8_a else 1)

        def issue_a_scale_tile_lds(kt):
            chunk_kt = (
                kt
                if const_expr(tilesPerScaleChunk == 1)
                else kt // fx.Int32(tilesPerScaleChunk)
            )
            addr0 = s_aq_base + fx.Int32(as_lds_off) + (chunk_kt * fx.Int32(64) + lane) * fx.Int32(4)
            return llvm.inline_asm(
                v2i32,
                [_raw(addr0 + kc_rt * fx.Int32(256 * k)) for k in range_constexpr(n_sc)],
                "\n".join(f"ds_read_b32 ${k}, ${k + n_sc}" for k in range(n_sc)),
                ",".join(["=&v"] * n_sc + ["v"] * n_sc),
                has_side_effects=True,
            )

        def wait_a_scale_tile_lds(pair):
            # LDS returns in order: once at most n_a_reads (<= 15, the A operand reads issued after) are left,
            # both are back. The outputs are tied, so no use of the scales floats above the wait.
            i32 = ir.IntegerType.get_signless(32)
            w = llvm.inline_asm(
                v2i32,
                [llvm.extractvalue(i32, pair, [k]) for k in range(n_sc)],
                f"s_waitcnt lgkmcnt({min(n_a_reads, 15)})",
                ",".join(["=v"] * n_sc + [str(k) for k in range(n_sc)]),
                has_side_effects=True,
            )
            return [fx.Int32(llvm.extractvalue(i32, w, [k])) for k in range_constexpr(n_sc)]

    def load_a_scale_tile(kt):
        # One i32 A-scale register per 32-row chunk (kScaleSubBlocks).
        chunk_kt = (
            kt
            if const_expr(tilesPerScaleChunk == 1)
            else kt // fx.Int32(tilesPerScaleChunk)
        )
        out = []
        for sub in range_constexpr(kScaleSubBlocks):
            saf = fx.make_fragment_like(sc_frag_tmpl)
            fx.copy(
                sc_copy_atom,
                ascale_views[sub][lane_div_16, lane_mod_16, chunk_kt, None],
                saf,
            )
            out.append(Vec(saf.load())[0])
        return out

    # Stream B weights and scales through registers so use_nt reaches the ISA cache policy.
    bq_buffer = ptr_buf_tensor(arg_bq, fx.Int32, unit_elems=4)

    bq_base_dw = [
        rocdl.readfirstlane(
            T.i32,
            (e * N_OUT_rt + n_block_idx * BN + wave * (BN // NW) + j * 16) * KH4,
        )
        for j in range_constexpr(numAccN)
    ]

    mni_base = n_block_idx * (BN // 16 // 2) + wave * (BN // (16 * NW) // 2)
    bscale_views = [
        scale_view(
            arg_bscale,
            e * kbs_per_expert_dw + (mni_base + mw) * kBS_stride_n0_dw,
            K_SCALE_CHUNKS_MAX,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(nPairs)
    ]

    frag_tmpl = fx.make_rmem_tensor(4, Int32)
    # B-scale word template shares the A-scale layout (sc_frag_tmpl).

    def issue_b_load_into(bqf, bsf, kt_rt):
        # Issue B-weight + B-scale vmem loads for K-tile kt_rt into the given (per-stage) fragments.
        for j in range_constexpr(numAccN):
            for half in range_constexpr(kHalves):
                bq_off_dw = (
                    bq_base_dw[j]
                    + lane_div_16 * fx.Int32(64)
                    + lane_mod_16 * fx.Int32(4)
                    + kt_rt * fx.Int32(kHalves * 256)
                    + fx.Int32(half * 256)
                )
                load_mask = None
                if const_expr(has_pad):
                    col = n_block_idx * BN + wave * (BN // NW) + j * 16
                    load_mask = (col < N_real) & (
                        kt_rt * fx.Int32(kHalves) + fx.Int32(half) < halves_real
                    )
                safe_bq_off_dw = (
                    load_mask.select(bq_off_dw, fx.Int32(0))
                    if load_mask is not None
                    else bq_off_dw
                )
                loaded_bq = buf_copy_load(
                    bq_buffer,
                    safe_bq_off_dw // fx.Int32(4),
                    fx.Int32,
                    unit_elems=4,
                    cache_modifier=2 if use_nt else 0,
                )
                bq_vec = (
                    fx.Vector.from_elements(
                        [
                            load_mask.select(loaded_bq[i], fx.Int32(0))
                            for i in range_constexpr(4)
                        ],
                        fx.Int32,
                    )
                    if load_mask is not None
                    else loaded_bq
                )
                bqf[j][half].store(Vec(bq_vec))
        chunk_kt = (
            kt_rt
            if const_expr(tilesPerScaleChunk == 1)
            else kt_rt // fx.Int32(tilesPerScaleChunk)
        )
        for mw in range_constexpr(nPairs):
            fx.copy(
                sc_copy_atom,
                bscale_views[mw][lane_div_16, lane_mod_16, chunk_kt, None],
                bsf[mw],
            )

    def stream_b_tile(kt_rt):
        # Fresh per-iter fragments (B streamed, not register-resident) then issue_b_load_into.
        bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
        bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        issue_b_load_into(bqf, bsf, kt_rt)
        return bqf, bsf

    # Scaled-MFMA clusters over the loaded A / B / scale fragments.
    def shift_scale_word(scale, kt_rt):
        if const_expr(tilesPerScaleChunk == 1):
            return scale
        scale_shift = (kt_rt % fx.Int32(tilesPerScaleChunk)) * fx.Int32(16)
        return fx.Int32(scale).shrui(scale_shift)

    def mfma_cluster(bqf, bsf, sa, kt_rt):
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
                    k_halves=kHalves,
                )
                continue
            for sub in range_constexpr(kScaleSubBlocks):
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
                    k_halves=kHalves,
                )

    # C accumulator: register fragments, zeroed then accumulated in place; (un)packed to K-loop carry.
    zero4 = fx.full(4, 0.0, Float32)
    kCChunks = kMChunks + fold_rows // 16
    c_frags = [
        [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(numAccN)]
        for _ in range_constexpr(kCChunks)
    ]
    for i in range_constexpr(kCChunks):
        for J in range_constexpr(numAccN):
            c_frags[i][J].store(zero4)

    def load_c_carry():
        return [c_frags[i][J].load() for i in range(kCChunks) for J in range(numAccN)]

    def store_c_carry(state):
        n = 0
        for i in range_constexpr(kCChunks):
            for J in range_constexpr(numAccN):
                c_frags[i][J].store(state[n])
                n += 1
        return n

    one_stage = BM == 64 and BN // NW == 64 and not g2_b2stage
    assert one_stage or not (a_dma_uniform or a_scale_lds or fold_rows)
    assert not fold_rows or (fold_rows == 16 and a_scale_lds)
    # The A-scale wait below assumes this K-step's B loads were issued before the barrier.
    assert not a_scale_lds or early_b_load
    if const_expr(one_stage):
        # BM64 with a 64-column per-wave slab (NW4 BN256 / NW8 BN512) uses the 1-stage B path.
        for kt_iv, state in range(
            fx.Int32(0),
            K_TILES_RT,
            fx.Int32(1),
            init=load_c_carry(),
        ):
            store_c_carry(state)
            kt_rt = fx.Int32(kt_iv)
            if const_expr(early_b_load):
                # Issue this K-tile's B and A-scale loads before the barrier and the next A-tile DMA:
                # vmcnt retires in issue order, so waiting for them leaves that DMA in flight.
                bqf, bsf = stream_b_tile(kt_rt)
                if const_expr(not a_scale_lds):
                    sa = load_a_scale_tile(kt_rt)
            if const_expr(a_scale_lds):
                # Step 0: the A-scale copy (older than the A preload) is done once only this step's B loads and
                # the newest A copy are in flight; later steps have nothing older in flight (no wait).
                fx.rocdl.s_waitcnt(vmcnt=numAccN * kHalves + nPairs + BM // NW // (64 // (KH_TILE_A // 16))
                    + (1 if fold_rows else 0))
            gpu.barrier()
            if const_expr(a_scale_lds):
                sa_pair = issue_a_scale_tile_lds(kt_rt)
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            if const_expr(fold_rows):
                # Before the next A copy is issued, so these LDS reads do not wait for it.
                if fold:
                    issue_a_ds_read(kt_rt % fx.Int32(aStages), chunks=[kMChunks])
            nxt = kt_rt + fx.Int32(kStages)
            if const_expr(a_dma_uniform):
                # Every K-step issues its A copy (the last two re-read the last tile into a dead slot): with the
                # copy conditional, the waitcnt merge of the copy-less path made the MFMAs wait for this copy.
                last_kt = K_TILES_RT - fx.Int32(1)
                issue_a_load_lds(nxt % fx.Int32(aStages), fx.Boolean(nxt < K_TILES_RT).select(nxt, last_kt))
            elif nxt < K_TILES_RT:
                issue_a_load_lds(nxt % fx.Int32(aStages), nxt)
            if const_expr(not early_b_load):
                bqf, bsf = stream_b_tile(kt_rt)
                sa = load_a_scale_tile(kt_rt)
            if const_expr(a_scale_lds):
                sa = wait_a_scale_tile_lds(sa_pair)
            mfma_cluster(bqf, bsf, sa[:kScaleSubBlocks], kt_rt)
            if const_expr(fold_rows):
                if fold:
                    for J in range_constexpr(numAccN):
                        mma_one_j(J, J % 2, shift_scale_word(sa[kScaleSubBlocks], kt_rt),
                            shift_scale_word(Vec(bsf[J // 2].load())[0], kt_rt), bqf, a_frags, c_frags, mma_atoms,
                            i0=kMChunks, single_rg=True, rg_off=0, k_halves=kHalves)
            results = yield load_c_carry()
        store_c_carry(results)
        if const_expr(a_dma_uniform):
            # The epilogue reuses the LDS the two trailing copies write.
            wait_lds_barrier(0)
    else:
        # 2-stage B pipeline: consume carried "current" B, prefetch next tile into the same fragments via scf.for state.
        cur_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
        cur_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        nxt_bqf = [
            [fx.make_fragment_like(frag_tmpl) for _ in range_constexpr(kHalves)]
            for _ in range_constexpr(numAccN)
        ]
        nxt_bsf = [fx.make_fragment_like(sc_frag_tmpl) for _ in range_constexpr(nPairs)]
        # g2_ascale_pf: carry the A-scale through scf.for state, same rotating-buffer model as B.
        cur_saf = nxt_saf = None
        if const_expr(g2_ascale_pf):
            cur_saf = [
                fx.make_fragment_like(sc_frag_tmpl)
                for _ in range_constexpr(kScaleSubBlocks)
            ]
            nxt_saf = [
                fx.make_fragment_like(sc_frag_tmpl)
                for _ in range_constexpr(kScaleSubBlocks)
            ]

        def load_b_carry():
            # Flat CURRENT (to-consume) B-weight, B-scale, then (opt) A-scale values.
            out = []
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(cur_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(cur_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(cur_saf[sub].load())
            return out

        def store_b_carry(state, base):
            n = base
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    cur_bqf[j][half].store(state[n])
                    n += 1
            for mw in range_constexpr(nPairs):
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
            for j in range_constexpr(numAccN):
                for half in range_constexpr(kHalves):
                    out.append(nxt_bqf[j][half].load())
            for mw in range_constexpr(nPairs):
                out.append(nxt_bsf[mw].load())
            if const_expr(g2_ascale_pf):
                for sub in range_constexpr(kScaleSubBlocks):
                    out.append(nxt_saf[sub].load())
            return out

        def issue_a_scale_load_into(saf, kt_rt):
            # A-scale vmem load(s) for K-tile kt_rt into the given (per-stage) fragment(s).
            sa = load_a_scale_tile(kt_rt)
            for sub in range_constexpr(kScaleSubBlocks):
                saf[sub].store(fx.full(1, sa[sub], Int32))

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
                for j in range_constexpr(numAccN):
                    for half in range_constexpr(kHalves):
                        nxt_bqf[j][half].store(cur_bqf[j][half].load())
                for mw in range_constexpr(nPairs):
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
            if const_expr(g2_deep_a_pipeline):
                if (kt_rt % fx.Int32(kFenceEvery)) == 0:
                    gpu.barrier()
            else:
                gpu.barrier()
            nxt_a = kt_rt + fx.Int32(kStages)
            issue_a_ds_read(kt_rt % fx.Int32(aStages))
            if nxt_a < K_TILES_RT:
                issue_a_load_lds(nxt_a % fx.Int32(aStages), nxt_a)
            # A-scale from the prefetch carry (g2_ascale_pf) or loaded synchronously here.
            if const_expr(g2_ascale_pf):
                sa = [
                    Vec(cur_saf[sub].load())[0]
                    for sub in range_constexpr(kScaleSubBlocks)
                ]
            else:
                sa = load_a_scale_tile(kt_rt)
            if const_expr(not g2_bhoist):
                prefetch_next_b(kt_rt)
            # Fence the MFMA chain from the B vmem loads (next-tile loads ride ahead of compute).
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(cur_bqf, cur_bsf, sa, kt_rt)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            results = yield yield_carry()
        store_carry(results)

    # Load the C fragments (fp8/fp4 unified onto the same fx.gemm path) and hand them to the epilog.
    accm_vecs = [
        [c_frags[i][J].load() for J in range(numAccN)] for i in range(kCChunks)
    ]
    return accm_vecs, m_row, n_block_idx, N_OUT_rt


def _spart_output_tile_index(block_1d_id, M0, N0, group_num, m01):
    """Map a 1D block ID to a spatially local output tile."""
    gn = fx.Int32(group_num)
    n0 = fx.Int32(N0)
    m01c = fx.Int32(m01)

    # group_size = ceil(M0*N0 / GroupNum); big_group_num = GroupNum - (group_size*GroupNum - M0*N0)
    mn = M0 * n0
    group_size = (mn + gn - fx.Int32(1)) // gn
    big_group_num = gn - (group_size * gn - mn)

    group_id_y = block_1d_id // gn
    group_id_x = block_1d_id - group_id_y * gn

    # remap = group_id_x < big_group_num ? gx*gs + gy : gx*gs + big - gx + gy
    remap_a = group_id_x * group_size + group_id_y
    remap_b = group_id_x * group_size + big_group_num - group_id_x + group_id_y
    remap = (group_id_x < big_group_num).select(remap_a, remap_b)

    idx_M0 = remap // n0
    idx_N0 = remap - idx_M0 * n0

    # M0_tmp = M0 / M01 ; M0_mod_M01 = M0 - M0_tmp*M01 ; M01_adapt = (idx_M0 < M0 - M0_mod) ? M01 : M0_mod
    M0_tmp = M0 // m01c
    M0_mod = M0 - M0_tmp * m01c
    M01_adapt = (idx_M0 < (M0 - M0_mod)).select(m01c, M0_mod)

    idx_M00 = idx_M0 // m01c
    idx_M01 = idx_M0 - idx_M00 * m01c
    idx_local = idx_N0 + idx_M01 * n0

    N_out = idx_local // M01_adapt
    loc_mod = idx_local - N_out * M01_adapt

    m_block_idx = loc_mod + idx_M00 * m01c
    n_block_idx = N_out
    return m_block_idx, n_block_idx


def _resolve_g2_knobs(g2_bhoist, g2_ascale_pf, g2_spart, g2_bf16_lds, use_reduce):
    """Resolve explicit GEMM2 knobs against deterministic defaults."""
    g2_bhoist = True if g2_bhoist is None else bool(g2_bhoist)
    g2_ascale_pf = True if g2_ascale_pf is None else bool(g2_ascale_pf)
    g2_spart = 402 if g2_spart is None else int(g2_spart)
    g2_group_num = g2_spart // 100 if g2_spart > 0 else 0
    g2_m01 = g2_spart % 100 if g2_spart > 0 else 0
    if g2_spart > 0 and (g2_group_num < 1 or g2_m01 < 1):
        raise AssertionError(
            f"g2_spart={g2_spart} must encode GroupNum>=1,M01>=1 as GroupNum*100+M01 (e.g. 402)"
        )
    g2_bf16_lds = (True if g2_bf16_lds is None else bool(g2_bf16_lds)) and use_reduce
    return g2_bhoist, g2_ascale_pf, g2_spart, g2_group_num, g2_m01, g2_bf16_lds

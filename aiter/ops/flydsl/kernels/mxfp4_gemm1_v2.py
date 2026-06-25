# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""FlyDSL **layout-API** port of aiter ``gemm1_a4w4`` (MXFP4 MoE up/gate-proj).

Variants: ``(BM=32, inline_quant=False, interleave=True)`` for BOTH ``use_nt``
values (kSubBlocks=1, kMChunks=2, kAStages=3, kStages=2, kUnroll=26). ``use_nt``
selects the B-load cache policy, matching the tuned config's BM32_NT vs
BM32_CACHED kernelName1 -- BOTH go through the layout API now, differing only in
the copy atom's cache modifier:
  * ``use_nt=True``  (BM32_NT)     -> fx.copy w/ BufferCopy128b(cache_modifier=2)  (decode)
  * ``use_nt=False`` (BM32_CACHED) -> fx.copy w/ BufferCopy128b(cache_modifier=0)  (prefill)

CONVERTED to the layout API (the deliverable):
  * **B-weight load** -> a hierarchical ``fx.make_layout`` preshuffle view over the
    bq buffer (adapted to this kernel's 5-D CK layout + per-expert base offset),
    loaded via a layout-API ``fx.copy`` into register fragments. The non-temporal
    (nt) decode policy rides on the copy atom's ``cache_modifier`` (FlyDSL
    ``BufferCopy128b(cache_modifier=2)``), so there is NO raw buffer_load fallback.
  * **Scaled MMA** -> ``fx.gemm`` over rank-1 register fragments (A/B/C), one
    fx.gemm per mfma (each lowers to a single fly MmaAtomCall ->
    rocdl.mfma.scale.f32.16x16x128.f8f6f4). Per-K-block e8m0 scales ride on the
    atom STATE via fx.gemm ``scale_a=/scale_b=`` kwargs; the runtime per-K opsel
    selects one of the pre-built opsel-specialized atoms (opsel is a type param).
    Bit-identical ISA to the prior raw intrinsic (mem2reg folds the fragment
    alloca/store/load; 448 mfma_scale, no scratch/spill).
  * **Accumulators** -> C register fragments updated IN PLACE (fx.gemm d == c),
    one per (i,J); the epilogue loads them. A and B are likewise register
    fragments (A refilled per K iter from ds_read; B per-stage for the pipeline).

KEPT RAW (reused by importing from the v1 module ``mxfp4_gemm1`` -- not duplicated):
  * all math / quant helpers (``_silu_mul_batch``, ``_e8m0_*``, ``_fabs_f32``,
    ``_lds_swizzle_mask``, ``_raw``, ``_global_base_ptr1``, ``_gep1``, ``_gep3``,
    ``_lds_base_ptr3``, ``gemm1_grid``, the ``*_for(...)`` size helpers + consts).
  * the A-side LDS stage + ds-read addressing + A-scale machinery (the e8m0 byte
    layout that feeds the mfma scale operands) -- the swizzle/byte layout is
    delicate and the raw path is the bit-exact spec; the ds-read result is stored
    into layout-API A fragments. Converting the addressing itself is later polish.
  * the A-side gather (rows via m_indices) base-offset computation.
  * the SwiGLU + fp4/e8m0 requant epilogue MATH (reuses helpers).

Acceptance is the KIMI BM32 interleave numeric gate (mean_row_cos > 0.85; v1
scores ~0.874 on this case).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Float4E2M1FN
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

# --- reuse ALL the raw math / size / pointer helpers from the v1 module ---
from .mxfp4_gemm1 import (
    BK,
    BN,
    KH_TILE,
    _e8m0_from_amax,
    _fabs_f32,
    _gep1,
    _gep3,
    _global_base_ptr1,
    _global_ptr1,
    _lds_base_ptr3,
    _lds_ptr3,
    _lds_swizzle_mask,
    _raw,
    _silu_mul_batch,
    bq_bytes_for,
    ascale_bytes_for,
    bscale_bytes_for,
    gemm1_grid,  # noqa: F401  (re-exported for parity with v1's public surface)
    k_g2_half_for,
    k_half_for,
    k_tiles_total_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kbs_stride_n0_dw_for,
    kBS_stride_k0_dw,
    kunroll_for,
    n_out_for,
    num_n_blocks_for,
    out_as_per_chunk_dw_for,
)
from .mxfp4_gemm1 import INTER as INTER_DEFAULT
from .mxfp4_gemm1 import K as K_DEFAULT
from .mxfp4_gemm1 import NE as NE_DEFAULT
from .mxfp4_gemm1 import TOPK as TOPK_DEFAULT
from .mxfp4_gemm1 import kStages as kStages

# BM32_NT path: these are fixed for the single supported variant.
_BM = 32
_kAStages = 3
_kSubBlocks = 1
_kMChunks = 2  # = BM // 16
_BN_INT = BN // 2  # 128


@flyc.jit
def _gemm1_body_v2(
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
    bx_i32,
    lane,
    wave,
    i32_ntok,
    i32_total_m_blocks,
    *,
    K,
    K_HALF,
    K_TILES_TOTAL,
    kUnroll,
    kAS_per_chunk_dw,
    kBS_stride_n0_dw,
    kBS_per_expert_dw,
    BQ_BYTES,
    ASCALE_BYTES,
    BSCALE_BYTES,
    N_OUT,
    NUM_N_BLOCKS,
    OUT_AS_PER_CHUNK_DW,
    K_G2_HALF,
    interleave,
    b_nontemporal,
):
    BM = _BM
    kAStages = _kAStages
    kSubBlocks = _kSubBlocks
    kMChunks = _kMChunks
    M_REPS = BM // 16

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[m_block_idx]
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

    # -- buffer resources (exact num_bytes) -- (KEPT RAW: A-gather + scales) --
    aq_num_records = arith.index_cast(T.index, _raw(i32_ntok * fx.Int32(K_HALF)))
    aq_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_aq)), num_records_bytes=aq_num_records
    )
    _asc_per_mb = max(BM // 32, 1) * kAS_per_chunk_dw * 4
    ascale_num = arith.index_cast(T.index, _raw(i32_total_m_blocks)) * fx.Index(
        _asc_per_mb
    )
    ascale_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_ascale)), num_records_bytes=ascale_num
    )
    bscale_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_bscale)), num_records_bytes=BSCALE_BYTES
    )

    # ============================================================
    # B-weight load -- LAYOUT API (the non-negotiable core conversion).
    #
    # v1 raw addressing (issue_b_load_j, interleave=True): the global BYTE address
    # of one B element for (expert e, n-tile col, lane, K-tile K_C, half) is
    #   (e*N_OUT + col)*K_HALF
    #     + lane_div_16*256 + lane_mod_16*16 + K_C*2048 + half*1024
    # with col = n_block_idx*BN + wave*(BN/4) + j*16,  j in [0,4).
    # The vec4-i32 load consumes 16 contiguous bytes from that base.
    #
    # This is exactly the CK 5-D preshuffle layout. We encode it as a hierarchical
    # fx.make_layout over the bq buffer (mirroring preshuffle_gemm_v2 lines 470-485)
    # and read it via make_tiled_copy_B partitions + fx.copy into B fragments that
    # feed the raw mfma. The per-expert base offset e*N_OUT*K_HALF is folded into
    # the buffer-tensor base pointer (a uniform per-block address, readfirstlane'd).
    #
    # CK preshuffle byte layout (bytes, fp4x2 => 1 byte per 2 fp4 along K):
    #   col index    -> n0 = col // 16, nlane = col % 16   (16-row n-lanes)
    #   K byte index -> within a (klane=4, kpack=16) tile at K_C granularity:
    #     base K byte for K_C = K_C * 2048  (= 4*16*32 ? see below) ... we mirror
    #     v1's exact integer offsets rather than re-deriving, so it is bit-exact.
    #
    # Concretely, per (lane, K_C, half) the 16-byte fragment base is:
    #   off = lane_div_16*256 + lane_mod_16*16 + K_C*2048 + half*1024
    # which decomposes as a layout with axes
    #   (klane=lane_div_16 in [0,4) -> stride 256),
    #   (nlane=lane_mod_16 in [0,16) -> stride 16),
    #   (K_C in [0,K_TILES_TOTAL) -> stride 2048),
    #   (half in [0,2) -> stride 1024),
    #   (kpack byte in [0,16) -> stride 1),
    # plus the per-n-tile column stride (col -> *K_HALF) folded via the tile base.
    # ============================================================

    # The B preshuffle layout view + tiled-copy load is constructed below
    # (issue_b_load_j_layout). A plain buffer resource over bq is also kept for the
    # B-scale path parity (scales stay raw).

    # -- LDS views (s_aq / s_asc, union-overlapping lds_acc) -- (KEPT RAW) -----
    lds_base = allocator.get_base()
    s_aq = SmemPtr(lds_base, lds_off, T.i8, shape=(kAStages * BM * KH_TILE,))
    s_asc = SmemPtr(
        lds_base,
        lds_off + kAStages * BM * KH_TILE,
        T.i8,
        shape=(kSubBlocks * K_TILES_TOTAL * 256,),
    )
    lds_acc = SmemPtr(lds_base, lds_off, T.f32, shape=(BM * BN,))

    # -- cached A rows (KEPT RAW: A-gather base offset) -----------------------
    cached_actual_row = []
    for sub in range_constexpr(kSubBlocks):
        idx = m_row + wave * fx.Int32(BM // 4) + fx.Int32(sub * 8) + lane_div_8
        cached_actual_row.append(
            llvm.load(T.i32, _global_ptr1(arg_mind, idx * fx.Int32(4)))
        )

    # -- b_load_s_base[j] (per-jtile column base offset) ----------------------
    b_load_s_base = []
    for j in range_constexpr(4):
        col = (
            n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
        )
        v = (e * fx.Int32(N_OUT) + col) * fx.Int32(K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    # -- b_scale_s_base / _hi (KEPT RAW) --------------------------------------
    mni_base = n_block_idx * fx.Int32(BN // 32) + wave * fx.Int32(BN // 128)
    np_list = [mni_base, mni_base + fx.Int32(1)]
    b_scale_s_base, b_scale_s_base_hi = [], []
    for mw in range_constexpr(2):
        base = (
            e * fx.Int32(kBS_per_expert_dw) + np_list[mw] * fx.Int32(kBS_stride_n0_dw)
        ) * fx.Int32(4)
        base = rocdl.readfirstlane(T.i32, base)
        b_scale_s_base.append(base)
        b_scale_s_base_hi.append(base + fx.Int32(16 * kBS_stride_k0_dw * 4))

    # Per-stage B-scale (Vec, cheap to double-buffer as SSA). A/B/C register
    # fragments for fx.gemm are built below (after the preshuffle views).
    b_scale_v = [[None, None] for _ in range(kStages)]

    # raw A LDS helpers (KEPT RAW) -------------------------------------------
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
        # ds_read the A tile from LDS and STORE into the A register fragments that
        # feed fx.gemm (raw llvm.load addressing kept; the value lands in a fragment).
        mask = _lds_swizzle_mask(lane_mod_16)
        base_ptr = _lds_base_ptr3(s_aq.get())
        for k in range_constexpr(2):
            lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(kMChunks):
                lds_row = lane_mod_16 + fx.Int32(i * 16)
                off = (
                    fx.Int32(slot * (BM * KH_TILE))
                    + lds_row * fx.Int32(KH_TILE)
                    + lds_col
                )
                vec = llvm.load(T.vec(4, T.i32), _gep3(base_ptr, off))
                _a_frags[i][k].store(Vec(vec))

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

    # ============================================================
    # B-weight load -- LAYOUT API (the non-negotiable core conversion).
    #
    # The B preshuffle is encoded as a hierarchical fx.make_layout VIEW over the bq
    # buffer (mirroring preshuffle_gemm_v2 lines 470-485), adapted to this kernel's
    # CK 5-D layout. v1's raw 16-byte (vec4-i32) fragment byte address for
    # (expert e, n-tile col, lane, K-tile K_C, half) is:
    #   (e*N_OUT + col)*K_HALF
    #     + lane_div_16*256 + lane_mod_16*16 + K_C*2048 + half*1024
    # In i32 (4-byte) units, with klane=lane_div_16, nlane=lane_mod_16:
    #   base_i32 = (e*N_OUT + col)*(K_HALF/4)        # per-jtile, uniform per wave
    #            + klane*64 + nlane*4                # per-lane, folded into base
    #   view axes:  (K_C -> stride 512, half -> stride 256, kpack4 -> stride 1)
    # We build one buffer-tensor VIEW per jtile carrying those axes, and load each
    # 16-byte fragment with a layout-API fx.copy (BufferCopy128b) into a register
    # fragment; the fragment's vec4 feeds the raw mfma cluster.
    #
    # cache_modifier carries the B-load policy THROUGH the layout API:
    #   use_nt=True  -> cache_modifier=2 (nt, non-temporal -> decode)
    #   use_nt=False -> cache_modifier=0 (cached            -> prefill)
    # Both paths are now layout-API fx.copy; the nt aux hint rides on the copy atom
    # type (FlyDSL BufferCopy128b(cache_modifier=...)), no raw buffer_load fallback.
    KH4 = K_HALF // 4  # i32 stride for the col axis
    _b_cache_mod = 2 if b_nontemporal else 0
    _b_copy_atom = fx.make_copy_atom(
        fx.rocdl.BufferCopy128b(_b_cache_mod), 32
    )  # 4x i32 = 128b

    def _make_bq_view_for_jtile(j):
        # The buffer-descriptor base MUST be UNIFORM across the wave -- otherwise
        # make_buffer_tensor emits a per-lane WATERFALL loop (the real 14x killer:
        # ~900 readfirstlane + s_and_saveexec + s_cbranch per B load). v1 keeps the
        # descriptor base uniform (bq_rsrc) and carries the per-lane part as the
        # buffer_load voffset. We mirror that: anchor the view at the UNIFORM
        # per-(wave,jtile) col base, and express the per-lane (klane, nlane) offset
        # as LAYOUT AXES so indexing turns them into a VGPR voffset, not a divergent
        # descriptor base.
        col = (
            n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
        )
        col_base = rocdl.readfirstlane(
            T.i32, (e * fx.Int32(N_OUT) + col) * fx.Int32(KH4)
        )  # uniform i32 element offset from bq base
        # global i32 pointer anchored at the UNIFORM per-wave fragment-region start.
        i32_ptr_ty = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=16
        )
        # absolute byte address = bq_base + col_base*4. Extend to i64 BEFORE *4:
        # the byte offset reaches ~BQ_BYTES (~1.4e9) which overflows a signed i32.
        off_i64 = fx.Int64(arith.ExtUIOp(T.i64, _raw(col_base)).result)
        addr_i64 = fx.Int64(arg_bq) + off_i64 * fx.Int64(4)
        base_iter = fx.inttoptr(i32_ptr_ty, addr_i64)
        # hierarchical preshuffle view, i32 element strides:
        #   klane  = lane_div_16 in [0,4)  -> 64   (= 256 bytes)
        #   nlane  = lane_mod_16 in [0,16) -> 4    (= 16 bytes)
        #   K_C    in [0,K_TILES_TOTAL)    -> 512  (= 2048 bytes)
        #   half   in [0,2)                -> 256  (= 1024 bytes)
        #   kpack4 in [0,4)                -> 1
        # The K-structure AND the per-lane structure are both expressed AS A LAYOUT.
        view = fx.Tensor(
            fx.make_view(
                base_iter,
                fx.make_layout(
                    (4, 16, K_TILES_TOTAL, 2, 4),
                    (64, 4, 512, 256, 1),
                ),
            )
        )
        return fx.rocdl.make_buffer_tensor(view, max_size=False)

    # Build the preshuffle VIEWS once, hoisted out of the K-loop (both nt + cached
    # paths use the layout API now -- they differ only in the copy atom's cache mod).
    _bq_views = [_make_bq_view_for_jtile(j) for j in range_constexpr(4)]

    # All MMA operands are register FRAGMENTS feeding fx.gemm (rank-1 -> one
    # MmaAtomCall each). Element type i32<4:1> (=16B=32 fp4) for A/B; f32<4:1> (vec4
    # accumulator) for C, via make_fragment_like(dtype=f32).
    _frag_tmpl = _bq_views[0][0, 0, 0, 0, None]  # i32<4:1>

    # B fragments are PER-STAGE (kStages): the K-loop prefetches B one+ stage ahead
    # (slot_b = OFFSET % kStages), so each stage needs its own fragment to hold data
    # from load (iter N) until consumed (iter N+kStages) -- mirrors the old per-stage
    # Vec double-buffering.
    _bq_frags = [
        [
            [fx.make_fragment_like(_frag_tmpl) for _ in range_constexpr(2)]
            for _ in range_constexpr(4)
        ]
        for _ in range_constexpr(kStages)
    ]
    # A fragments: one set, refilled from ds_read each K iter (used by all 4 J in
    # that iter before the next ds_read). C accumulators: in-place across the K loop.
    _a_frags = [
        [fx.make_fragment_like(_frag_tmpl) for _ in range_constexpr(2)]
        for _ in range_constexpr(kMChunks)
    ]
    _c_frags = [
        [
            fx.make_fragment_like(_frag_tmpl, dtype=fx.Float32.ir_type)
            for _ in range_constexpr(4)
        ]
        for _ in range_constexpr(kMChunks)
    ]

    def issue_b_load_j(stage, K_C, j):
        # LAYOUT-API B load: preshuffle view -> fx.copy into the stage's B fragment
        # (cache modifier on _b_copy_atom selects nt vs cached). No Vec extraction;
        # fx.gemm reads the fragment directly at MMA time.
        view = _bq_views[j]
        for half in range_constexpr(2):
            fx.copy(
                _b_copy_atom,
                view[lane_div_16, lane_mod_16, K_C, half, None],
                _bq_frags[stage][j][half],
            )

    def issue_b_scale_load(bs_slot, K_C):
        v = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)
        K_C_HI = K_C // 16
        imm = (K_C - K_C_HI * 16) * (kBS_stride_k0_dw * 4)
        for mw in range_constexpr(2):
            s_off = b_scale_s_base[mw] if K_C_HI == 0 else b_scale_s_base_hi[mw]
            bs_slot[mw] = buffer_ops.buffer_load(
                bscale_rsrc,
                (v + fx.Int32(imm)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=s_off,
            )

    # -- MFMA cluster -- LAYOUT API via fx.gemm -------------------------------
    # Each microscaled mfma is one fx.gemm over rank-1 register fragments (A/B/C),
    # which lowers to a single fly MmaAtomCall -> rocdl.mfma.scale.f32.16x16x128.
    # The per-K-block e8m0 scales ride on the atom STATE (fx.gemm **kwargs ->
    # set_value scale_a/scale_b); the runtime per-K opsel is a type param, so we
    # select a pre-built opsel-specialized atom per (opselA,opselB) pair. cbsz/blgp
    # (=4 for fp4) are inferred from the Float4E2M1FN operand types. The accumulator
    # is the C fragment updated IN PLACE (d == c), so no SSA acc chain is threaded.
    zero4 = Vec.filled(4, 0.0, fx.Float32)

    # Build the opsel-specialized scaled-MMA atoms ONCE (opsel is a type param, so
    # one atom per (opselA,opselB) pair). All 16 combos -- cheap, built at trace time.
    _scale_mma_atoms = {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, Float4E2M1FN, opsel_a=osa, opsel_b=osb
            )
        )
        for osa in range_constexpr(4)
        for osb in range_constexpr(4)
    }

    def _gemm_mma(a_frag, b_frag, c_frag, opsel_a, opsel_b, sa, sb):
        # one fx.gemm = one MmaAtomCall (rank-1 fragments). d == c -> in-place accum.
        fx.gemm(
            _scale_mma_atoms[(opsel_a, opsel_b)],
            c_frag,
            a_frag,
            b_frag,
            c_frag,
            scale_a=sa,
            scale_b=sb,
        )

    def mfma_cluster(stage, a_scale, bs_slot, J):
        in_b = J % 2
        sb = bs_slot[J // 2]
        bJ0, bJ1 = _bq_frags[stage][J][0], _bq_frags[stage][J][1]
        for sub in range_constexpr(kSubBlocks):
            i0 = sub * 2 + 0
            i1 = sub * 2 + 1
            sa = a_scale[sub]
            _gemm_mma(_a_frags[i0][0], bJ0, _c_frags[i0][J], 0, 0 + in_b, sa, sb)
            _gemm_mma(_a_frags[i1][0], bJ0, _c_frags[i1][J], 1, 0 + in_b, sa, sb)
            _gemm_mma(_a_frags[i0][1], bJ1, _c_frags[i0][J], 2, 2 + in_b, sa, sb)
            _gemm_mma(_a_frags[i1][1], bJ1, _c_frags[i1][J], 3, 2 + in_b, sa, sb)

    # ---- zero the C accumulator fragments (in-place accumulate thereafter) ----
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(4):
            _c_frags[i][J].store(zero4)

    # ---- prologue: stages 0,1 ----
    issue_a_scale_load()
    for K_C in range_constexpr(kStages):
        issue_a_load_lds(K_C, K_C)
        for j in range_constexpr(4):
            issue_b_load_j(K_C, K_C, j)
        issue_b_scale_load(b_scale_v[K_C], K_C)

    # ---- main loop: OFFSET in [0,kUnroll) ----
    for OFFSET in range_constexpr(kUnroll):
        K_C = kStages + OFFSET
        read_slot = OFFSET % kAStages
        write_slot = K_C % kAStages
        slot_b = OFFSET % kStages
        gpu.barrier()
        issue_a_ds_read(read_slot)
        asc_cur = issue_a_scale_ds_read(K_C - kStages)
        issue_a_load_lds(write_slot, K_C)
        for J in range_constexpr(4):
            # Scheduler hints mirror v1's BM!=128 path: fence the mfma cluster from
            # the surrounding loads (sched_barrier) and raise its priority
            # (s_setprio) so the backend keeps the mfma chain dense instead of
            # interleaving the B buffer_loads into it. Closes the small-M gap.
            rocdl.sched_barrier(0)
            rocdl.s_setprio(1)
            mfma_cluster(slot_b, asc_cur, b_scale_v[slot_b], J)
            rocdl.s_setprio(0)
            rocdl.sched_barrier(0)
            issue_b_load_j(slot_b, K_C, J)
            rocdl.sched_barrier(0)
        issue_b_scale_load(b_scale_v[slot_b], K_C)

    # ---- drain: S in [0,kStages) ----
    for S in range_constexpr(kStages):
        kt = K_TILES_TOTAL - kStages + S
        gpu.barrier()
        issue_a_ds_read(kt % kAStages)
        asc_cur = issue_a_scale_ds_read(kt)
        for J in range_constexpr(4):
            mfma_cluster(kt % kStages, asc_cur, b_scale_v[kt % kStages], J)

    gpu.barrier()
    s_aq._view_cache = None
    s_asc._view_cache = None
    lds_acc._view_cache = None

    # -- epilog: cshuffle -> SwiGLU -> fp4 + e8m0 requant (KEPT RAW math) -----
    wave_n = wave
    lds_acc_base = _lds_base_ptr3(lds_acc.get())

    # Read accumulators back from the C register fragments (flat slot [i,J,v]).
    _acc_vecs = [[Vec(_c_frags[i][J].load()) for J in range(4)] for i in range(kMChunks)]

    def _acc(i, J, v):
        return _acc_vecs[i][J][v]

    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            is_up = (J % 2) == 1
            J_local = J // 2
            col_local = wave_n * fx.Int32(32) + fx.Int32(J_local * 16) + lane_mod_16
            lds_col = (fx.Int32(128) + col_local) if is_up else col_local
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + lds_col
                llvm.StoreOp(
                    _raw(fx.Float32(_acc(i, J, v))),
                    _gep3(lds_acc_base, idx * fx.Int32(4)),
                )

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
            n_block_idx * fx.Int32(_BN_INT // 2)
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


def _lds_bytes_for(K_TILES_TOTAL):
    s_aq_bytes = _kAStages * _BM * KH_TILE
    s_asc_bytes = _kSubBlocks * K_TILES_TOTAL * 256
    lds_acc_bytes = _BM * BN * 4
    return max(s_aq_bytes + s_asc_bytes, lds_acc_bytes)


def compile_gemm1_a4w4_port(
    BM=32,
    use_nt=True,
    inline_quant=False,
    D_HIDDEN=K_DEFAULT,
    D_INTER=INTER_DEFAULT,
    NE=NE_DEFAULT,
    TOPK=TOPK_DEFAULT,
    interleave=True,
):
    # use_nt IS the B-load cache policy (matches v1's `b_aux = 2 if use_nt else 0`
    # and the tuned config's BM32_NT vs BM32_CACHED kernelName1):
    #   use_nt=True  -> BM32_NT     : non-temporal raw buffer_load (decode/small M)
    #   use_nt=False -> BM32_CACHED : cached layout-API fx.copy   (prefill/large M)
    b_nontemporal = use_nt
    print(
        f"[PORT-FLYDSL-GEMM1-V2] compile_gemm1_a4w4_port ENTERED "
        f"BM={BM} use_nt={use_nt} inline_quant={inline_quant} "
        f"D_HIDDEN={D_HIDDEN} D_INTER={D_INTER} NE={NE} interleave={interleave} "
        f"b_nontemporal={b_nontemporal}",
        flush=True,
    )
    if (BM, inline_quant) != (32, False):
        raise AssertionError(
            f"mxfp4_gemm1_v2 supports only (BM=32, inline_quant=False) for either "
            f"use_nt; got (BM={BM}, use_nt={use_nt}, inline_quant={inline_quant})"
        )
    if not interleave:
        raise AssertionError(
            "mxfp4_gemm1_v2 Phase 1 supports interleave=True only "
            "(separated gate/up is a later phase; v1 handles it)."
        )

    _K = D_HIDDEN
    assert _K % BK == 0, f"D_HIDDEN (K) must be a multiple of {BK}, got {_K}"
    _INTER = D_INTER
    _N_OUT = n_out_for(_INTER)
    assert _N_OUT % BN == 0, f"2*D_INTER (N_OUT) must be a multiple of {BN}, got {_N_OUT}"
    _NE = NE

    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K)
    _kUnroll = kunroll_for(_K)
    _kAS_per_chunk_dw = kas_per_chunk_dw_for(_K)
    _kBS_stride_n0_dw = kbs_stride_n0_dw_for(_K)
    _kBS_per_expert_dw = kbs_per_expert_dw_for(_K, _INTER)
    _BQ_BYTES = bq_bytes_for(_K, _INTER, _NE)
    _ASCALE_BYTES = ascale_bytes_for(_K)
    _BSCALE_BYTES = bscale_bytes_for(_K, _INTER, _NE)
    _NUM_N_BLOCKS = num_n_blocks_for(_INTER)
    _OUT_AS_PER_CHUNK_DW = out_as_per_chunk_dw_for(_INTER)
    _K_G2_HALF = k_g2_half_for(_INTER)

    lds_bytes = _lds_bytes_for(_K_TILES_TOTAL)

    gu_tag = "il" if interleave else "sep"
    bnt_tag = "nt" if b_nontemporal else "cached"
    name_suffix = f"h{_K}_i{_INTER}_ne{_NE}_bm{BM}_{bnt_tag}_{gu_tag}_v2"

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name=f"gemm1port_v2_smem_{name_suffix}"
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + lds_bytes

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
            _gemm1_body_v2(
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
                bx_i32,
                lane,
                wave,
                i32_ntok,
                total_m_blocks,
                K=_K,
                K_HALF=_K_HALF,
                K_TILES_TOTAL=_K_TILES_TOTAL,
                kUnroll=_kUnroll,
                kAS_per_chunk_dw=_kAS_per_chunk_dw,
                kBS_stride_n0_dw=_kBS_stride_n0_dw,
                kBS_per_expert_dw=_kBS_per_expert_dw,
                BQ_BYTES=_BQ_BYTES,
                ASCALE_BYTES=_ASCALE_BYTES,
                BSCALE_BYTES=_BSCALE_BYTES,
                N_OUT=_N_OUT,
                NUM_N_BLOCKS=_NUM_N_BLOCKS,
                OUT_AS_PER_CHUNK_DW=_OUT_AS_PER_CHUNK_DW,
                K_G2_HALF=_K_G2_HALF,
                interleave=interleave,
                b_nontemporal=b_nontemporal,
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

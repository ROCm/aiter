# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors
"""FlyDSL **layout-API** port of aiter ``gemm2_a4w4`` (MXFP4 MoE down-proj).

Variant: ``(BM=32, epilog="atomic")`` for both ``use_nt`` (B-load nt/cached policy),
covering the KIMI/DSR fast path (D_INTER<=512, K_TILES<=2, fully unrolled, all tiles
preloaded) and the streaming K-loop (D_INTER>512, K_TILES>2). The BM32 GEMM core is
identical to ``mxfp4_gemm1_v2`` (same preshuffled-B layout + scaled-MFMA opsel
pattern), so the B-load / B-scale / MMA pieces come straight from
``mxfp4_moe_layout``:

  * B load -> ``ml.bq_view`` + ``fx.copy`` into per-tile register fragments;
    nt/cached rides the copy atom's cache_modifier.
  * B-scale load -> ``ml.bscale_view`` + ``fx.copy`` into per-tile i32 fragments.
  * MMA -> one ``ml.gemm_mma`` (fx.gemm) per mfma over rank-1 fragments (A/B/C),
    per-K-block e8m0 scales via scale_a=/scale_b=, C accumulating in place.

Kept raw (imported from the v1 ``mxfp4_gemm2`` module): math/pointer helpers, the
A-side LDS stage + ds-read + A-scale machinery, and the atomic-bf16 epilogue. Unlike
gemm1 (which streams B per-K), gemm2 loads ALL B tiles up front into registers (B is
not LDS-bound), so the B/B-scale fragments are PER-TILE (K_TILES_TOTAL), not
per-stage. Only the A->LDS stage streams (triple-buffered for K_TILES>2).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from . import mxfp4_moe_layout as ml

# Raw helpers / size formulas / constants reused verbatim from the v1 kernel; the
# A-side LDS stage + ds-read + A-scale and the atomic epilogue are unchanged.
from .mxfp4_gemm2 import (
    BK,
    BN,
    KH_TILE,
    K,
    MAX_M,
    NE,
    N_OUT,
    kBS_stride_k0_dw,
    kStages,
    _atomic_bf16_epilog,
    _gep3,
    _global_ptr1,
    _issue_a_load_lds,
    _lds_base_ptr3,
    _lds_swizzle_mask,
    _raw,
    _udiv,
    bscale_bytes_for,
    bq_bytes_for,
    k_half_for,
    k_tiles_total_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kbs_stride_n0_dw_for,
    kmchunks,
    kunroll_for,
    lds_bytes,
    num_n_blocks_for,
    saq_slot_bytes,
    tiling,
)


def compile_gemm2_a4w4_port(
    BM=32,
    use_nt=False,
    NE=NE,
    N_OUT=N_OUT,
    MAX_M=MAX_M,
    epilog="atomic",
    D_INTER=K,
    D_INTER_REAL=None,
):
    """Compile the gemm2 a4w4 layout-API port (v2). Only (BM=32, epilog="atomic")
    is supported; D_INTER (= contraction K = inter_dim) must be a multiple of BK(256)
    (KIMI/DSR 512 keeps the fully-unrolled fast path; >512 uses the streaming K-loop).
    D_INTER_REAL zero-pad-tail skipping is supported (dsv4 TP8)."""
    print(
        f"[PORT-FLYDSL-GEMM2-V2] compile_gemm2_a4w4_port ENTERED "
        f"BM={BM} use_nt={use_nt} NE={NE} N_OUT={N_OUT} epilog={epilog} D_INTER={D_INTER}",
        flush=True,
    )
    if (BM, epilog) != (32, "atomic"):
        raise AssertionError(
            f"mxfp4_gemm2_v2 supports only (BM=32, epilog='atomic'); "
            f"got (BM={BM}, epilog={epilog})"
        )
    _K = D_INTER
    _K_REAL = D_INTER if D_INTER_REAL is None else D_INTER_REAL
    assert _K % BK == 0, (
        f"D_INTER (gemm2 contraction K = inter_dim) must be a multiple of {BK}, got {_K}"
    )
    assert (
        _K_REAL % 128 == 0 and 0 < _K_REAL <= _K
    ), f"D_INTER_REAL={_K_REAL} must be a multiple of 128 and in (0, {_K}]"
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K)
    _aStages = kStages if _K_TILES_TOTAL <= kStages else 3
    _slot_bytes = saq_slot_bytes(BM)
    _lds_bytes = max(lds_bytes(BM), _aStages * _slot_bytes)
    _num_n_blocks = num_n_blocks_for(N_OUT)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(BM)

    _rtag = "" if _K_REAL == _K else f"r{_K_REAL}"
    _tag = f"ne{NE}_h{N_OUT}_i{_K}{_rtag}_bm{BM}{'_nt' if use_nt else ''}_atomic_v2"
    _name = f"gemm2_a4w4_port_{_tag}"

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name=f"gemm2port_v2_smem_{_tag}"
    )
    lds_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = lds_off + _lds_bytes

    @flyc.kernel(name=_name, known_block_size=[256, 1, 1])
    def gemm2_kernel(
        arg_aq: fx.Int64,
        arg_ascale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_stids: fx.Int64,
        arg_sweights: fx.Int64,
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,  # unused (atomic epilog); kept for signature parity
    ):
        tx = gpu.thread_id("x")
        bx = gpu.block_id("x")
        tx_i32 = fx.Int32(tx)
        bx_i32 = fx.Int32(bx)
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))

        _aq_num = arith.index_cast(T.index, _raw(i32_max_m_blocks)) * fx.Index(
            BM * _K_HALF
        )
        aq_rsrc = buffer_ops.create_buffer_resource_from_addr(
            _raw(fx.Int64(arg_aq)), num_records_bytes=_aq_num
        )
        saq = SmemPtr(
            allocator.get_base(), lds_off, T.i8, shape=(_aStages * _slot_bytes,)
        )

        # Preload the first kStages K-tiles (== ALL tiles for the K_TILES<=2 fast
        # path; == prologue for the streaming path). slot == kt for the preload.
        def _issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):
                for sub in range_constexpr(_kSubBlocks):
                    lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
                    car = m_row0 + lds_row + (lane // fx.Int32(8))
                    _issue_a_load_lds(
                        aq_rsrc, saq, slot, slot, car, lane, _slot_bytes, lds_row,
                        k_half=_K_HALF,
                    )

        # One-shot grid (atomic). Issue A->LDS BEFORE the cumsum load so the HBM
        # latency overlaps the cumsum + bound check (A->LDS depends only on bx/lane).
        m_row0 = _udiv(bx_i32, _num_n_blocks) * fx.Int32(BM)
        if const_expr(_n_load_waves < 4):
            if wave < fx.Int32(_n_load_waves):
                _issue_all_a_loads(m_row0)
        else:
            _issue_all_a_loads(m_row0)
        rocdl.sched_barrier(0)

        cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
        total_m_blocks = _udiv(cumsum0, BM)
        bound = total_m_blocks * fx.Int32(_num_n_blocks)

        if fx.Int32(bx_i32) < bound:
            _gemm2_body_v2(
                allocator,
                lds_off,
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
                BM=BM,
                use_nt=use_nt,
                NE=NE,
                N_OUT=N_OUT,
                D_INTER=_K,
                D_INTER_REAL=_K_REAL,
                aStages=_aStages,
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
        i32_M: fx.Int32,
        i32_max_m_blocks: fx.Int32,
        arg_out: fx.Int64,
        arg_out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        ctx = CompilationContext.get_current()
        allocator.finalized = False
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        grid_x = arith.index_cast(T.index, i32_max_m_blocks) * fx.Index(_num_n_blocks)
        gemm2_kernel(
            arg_aq,
            arg_ascale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_stids,
            arg_sweights,
            i32_M,
            i32_max_m_blocks,
            arg_out,
            arg_out_scale,
        ).launch(grid=(grid_x, 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm2


@flyc.jit
def _gemm2_body_v2(
    allocator,
    lds_off,
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
    *,
    BM,
    use_nt,
    NE,
    N_OUT,
    D_INTER,
    D_INTER_REAL,
    aStages,
):
    _aStages = aStages
    _kMChunks = kmchunks(BM)  # BM32: 2
    _slot_bytes = saq_slot_bytes(BM)
    # K-derived sizes (parametrized over contraction K = inter_dim = D_INTER).
    _K = D_INTER
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K)
    _K_REAL = D_INTER if D_INTER_REAL is None else D_INTER_REAL
    _n_real_half = (_K_REAL + 127) // 128  # valid 128-K MFMA half-steps
    _kUnroll = kunroll_for(_K)
    _kAS_per_chunk_dw = kas_per_chunk_dw_for(_K)
    _kBS_stride_n0_dw = kbs_stride_n0_dw_for(_K)
    _asc_chunk_div = 16 if const_expr(BM == 16) else 32
    _asc_per_mb = (BM // _asc_chunk_div) * _kAS_per_chunk_dw * 4
    _bq_bytes = bq_bytes_for(NE, N_OUT, _K)
    _bscale_bytes = bscale_bytes_for(NE, N_OUT, _K)
    _kbs_per_expert_dw = kbs_per_expert_dw_for(N_OUT, _K)
    _num_n_blocks = num_n_blocks_for(N_OUT)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(BM)
    KH4 = _K_HALF // 4

    # block -> (m_block_idx, n_block_idx) ; e = sorted_expert_ids[m_block_idx]
    m_block_idx = _udiv(bx_i32, _num_n_blocks)
    n_block_idx = bx_i32 - m_block_idx * fx.Int32(_num_n_blocks)
    e = rocdl.readfirstlane(
        T.i32, llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4)))
    )
    m_row = m_block_idx * fx.Int32(BM)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    # A-scale buffer resource (A-scale load stays raw).
    _asc_num = arith.index_cast(T.index, _raw(i32_max_m_blocks)) * fx.Index(_asc_per_mb)
    ascale_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_ascale)), num_records_bytes=_asc_num
    )

    lds_base = allocator.get_base()
    saq = SmemPtr(lds_base, lds_off, T.i8, shape=(_aStages * _slot_bytes,))
    lds_acc = SmemPtr(lds_base, lds_off, T.f32, shape=(BM * BN,))

    # A-scale uniform base (raw buffer_load addressing, as v1).
    chunk_base = m_row // fx.Int32(16 if const_expr(BM == 16) else 32)
    a_scale_s_base = [
        rocdl.readfirstlane(
            T.i32,
            (chunk_base + fx.Int32(sub)) * fx.Int32(_kAS_per_chunk_dw) * fx.Int32(4),
        )
        for sub in range_constexpr(_kSubBlocks)
    ]
    v_voff_scale = ((lane_div_16 * fx.Int32(16)) + lane_mod_16) * fx.Int32(4)

    def load_a_scale_tile(kt):
        out = [None] * _kSubBlocks
        for sub in range_constexpr(_kSubBlocks):
            out[sub] = buffer_ops.buffer_load(
                ascale_rsrc,
                (v_voff_scale + fx.Int32(kt * 256)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=a_scale_s_base[sub],
            )
        return out

    # -- B / B-scale layout-API views (shared primitives) ---------------------
    _b_copy_atom = ml.b_copy_atom(use_nt)
    _bs_copy_atom = ml.bscale_copy_atom()

    def _make_bq_view(j):
        col = n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + fx.Int32(j * 16)
        return ml.bq_view(arg_bq, e * fx.Int32(N_OUT) + col, KH4, _K_TILES_TOTAL)

    _bq_views = [_make_bq_view(j) for j in range_constexpr(4)]

    mni_base = n_block_idx * fx.Int32(BN // 16 // 2) + wave * fx.Int32(BN // 64 // 2)
    _bscale_views = [
        ml.bscale_view(
            arg_bscale,
            e * fx.Int32(_kbs_per_expert_dw)
            + (mni_base + fx.Int32(mw)) * fx.Int32(_kBS_stride_n0_dw),
            _K_TILES_TOTAL,
            k0_stride_dw=kBS_stride_k0_dw,
        )
        for mw in range_constexpr(2)
    ]

    # Fragments. B / B-scale are PER-TILE (all tiles loaded up front, as v1); A is
    # refilled per K iter; C (f32) accumulates in place (fx.gemm d == c).
    _frag_tmpl = ml.bq_frag_tmpl(_bq_views[0])  # i32<4:1>
    _bs_frag_tmpl = ml.bscale_frag_tmpl(_bscale_views[0])  # i32<1:1>
    _bq_frags = [
        [[fx.make_fragment_like(_frag_tmpl) for _ in range_constexpr(2)] for _ in range_constexpr(4)]
        for _ in range_constexpr(_K_TILES_TOTAL)
    ]
    _bs_frags = [
        [fx.make_fragment_like(_bs_frag_tmpl) for _ in range_constexpr(2)]
        for _ in range_constexpr(_K_TILES_TOTAL)
    ]
    _a_frags = [
        [fx.make_fragment_like(_frag_tmpl) for _ in range_constexpr(2)]
        for _ in range_constexpr(_kMChunks)
    ]
    _c_frags = [
        [fx.make_fragment_like(_frag_tmpl, dtype=fx.Float32.ir_type) for _ in range_constexpr(4)]
        for _ in range_constexpr(_kMChunks)
    ]

    def issue_b_load_tile(kt):
        # Skip the weight load for half-steps in the zero-pad tail (k >= K_REAL).
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                if const_expr(kt * 2 + half >= _n_real_half):
                    continue
                fx.copy(
                    _b_copy_atom,
                    _bq_views[j][lane_div_16, lane_mod_16, kt, half, None],
                    _bq_frags[kt][j][half],
                )

    def issue_b_scale_tile(kt):
        for mw in range_constexpr(2):
            fx.copy(
                _bs_copy_atom,
                _bscale_views[mw][lane_div_16, lane_mod_16, kt, None],
                _bs_frags[kt][mw],
            )

    # A ds-read (raw) -> A register fragments for fx.gemm.
    def issue_a_ds_read(slot):
        mask = _lds_swizzle_mask(lane_mod_16)
        base_ptr = _lds_base_ptr3(saq.get())
        for k in range_constexpr(2):
            lds_col = (lane_div_16 * fx.Int32(16) + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(_kMChunks):
                lds_row = lane_mod_16 + fx.Int32(i * 16)
                off = fx.Int32(slot * _slot_bytes) + lds_row * fx.Int32(KH_TILE) + lds_col
                vec = llvm.load(T.vec(4, T.i32), _gep3(base_ptr, off))
                _a_frags[i][k].store(Vec(vec))

    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(_kSubBlocks):
            lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
            car = m_row + lds_row + (lane // fx.Int32(8))
            _issue_a_load_lds(
                aq_rsrc, saq, slot, kt, car, lane, _slot_bytes, lds_row, k_half=_K_HALF
            )

    _scale_mma_atoms = ml.scale_mma_atoms()

    def mfma_cluster(kt, a_scale_sub):
        # interleave-equivalent opsel (gemm2 has no gate/up split): mni=J//2, in_b=J%2.
        # Skip half1 MFMAs in the zero-pad tail (b_tile[J][1] not loaded; adds 0).
        _skip_h1 = (kt * 2 + 1) >= _n_real_half
        for J in range_constexpr(4):
            mni, in_b = J // 2, J % 2
            sb = _raw(Vec(_bs_frags[kt][mni].load())[0])
            bJ0 = _bq_frags[kt][J][0]
            for sub in range_constexpr(_kSubBlocks):
                sa = a_scale_sub[sub]
                i0 = sub * 2
                i1 = sub * 2 + 1
                ml.gemm_mma(_scale_mma_atoms, _a_frags[i0][0], bJ0, _c_frags[i0][J], 0, 0 + in_b, sa, sb)
                if const_expr(_kMChunks > 1):
                    ml.gemm_mma(_scale_mma_atoms, _a_frags[i1][0], bJ0, _c_frags[i1][J], 1, 0 + in_b, sa, sb)
                if const_expr(not _skip_h1):
                    bJ1 = _bq_frags[kt][J][1]
                    ml.gemm_mma(_scale_mma_atoms, _a_frags[i0][1], bJ1, _c_frags[i0][J], 2, 2 + in_b, sa, sb)
                    if const_expr(_kMChunks > 1):
                        ml.gemm_mma(_scale_mma_atoms, _a_frags[i1][1], bJ1, _c_frags[i1][J], 3, 2 + in_b, sa, sb)

    # zero C (accumulate in place thereafter)
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(_kMChunks):
        for J in range_constexpr(4):
            _c_frags[i][J].store(zero4)

    # Load ALL B-q + B-scale + A-scale tiles up front (B is not LDS-bound), as v1.
    a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
    for kt in range_constexpr(_K_TILES_TOTAL):
        issue_b_load_tile(kt)
        issue_b_scale_tile(kt)

    if const_expr(_K_TILES_TOTAL <= kStages):
        # Fast path: all tiles preloaded in LDS by the kernel.
        for S in range_constexpr(_K_TILES_TOTAL):
            kt = S
            slot = kt % kStages
            gpu.barrier()
            issue_a_ds_read(slot)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(kt, a_scale_sub)
    else:
        # Streaming double-buffered K-loop (triple-buffered LDS). Main loop processes
        # tile kt=OFFSET (read slot kt%aStages) and streams next tile into its slot.
        for OFFSET in range_constexpr(_kUnroll):
            kt = OFFSET
            slot = kt % _aStages
            next_kt = kStages + OFFSET
            write_slot = next_kt % _aStages
            gpu.barrier()
            issue_a_ds_read(slot)
            issue_a_load_lds(write_slot, next_kt)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(kt, a_scale_sub)
        for S in range_constexpr(kStages):
            kt = _K_TILES_TOTAL - kStages + S
            slot = kt % _aStages
            gpu.barrier()
            issue_a_ds_read(slot)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(kt, a_scale_sub)

    # -- epilog: atomic bf16 (raw, reused from v1). Reads accm = loaded C frags. ---
    saq._view_cache = None
    lds_acc._view_cache = None
    accm = [
        [_c_frags[i][J].load() for J in range(4)] for i in range(_kMChunks)
    ]
    _atomic_bf16_epilog(
        lds_acc,
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
    )

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl._mlir.dialects import memref as memref_dialect
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
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
    _e8m0_from_amax,
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

NUM_CU = 256
PORT_XCD_SWIZZLE = int(os.environ.get("PORT_XCD_SWIZZLE", "-1"))


def aq_bytes_for(max_m, k):
    return max_m * k_half_for(k)


def saq_slot_bytes(BM, KH_TILE):
    return BM * KH_TILE


def tiling(BM):
    n_load_waves = min(4, BM // 8)
    rows_per_wave = BM // n_load_waves
    return n_load_waves, rows_per_wave, rows_per_wave // 8


def _udiv(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.divui(_raw(a), _raw(cc)))


def _umod(a, c):
    cc = fx.Int32(c) if isinstance(c, int) else c
    return fx.Int32(arith.remui(_raw(a), _raw(cc)))


def _issue_a_load_lds(
    aq_rsrc, saq, slot, kt, car, lane, slot_bytes, lds_row, KH_TILE, k_half
):
    lane_mod_8 = lane % fx.Int32(8)
    mask = _lds_swizzle_mask(lds_row + (lane // fx.Int32(8)))
    voffset = ((lane_mod_8 * fx.Int32(16)) ^ mask) + car * fx.Int32(k_half)
    base_i32 = fx.Int32(memref_dialect.extract_aligned_pointer_as_index(saq.get()))
    off_i32 = fx.Int32(slot * slot_bytes) + lds_row * fx.Int32(KH_TILE)
    lds_ptr = _lds_ptr3(base_i32, off_i32)
    rocdl.raw_ptr_buffer_load_lds(
        aq_rsrc,
        lds_ptr,
        fx.Int32(16),
        voffset,
        fx.Int32(kt * KH_TILE),
        fx.Int32(0),
        fx.Int32(0),
    )


def compile_gemm2_a4w4_port(
    BM=32,
    use_nt=False,
    *,
    NE,
    N_OUT,
    epilog="atomic",
    D_INTER,
    D_INTER_REAL=None,
    BN=256,
    BK=256,
):
    print(
        f"[PORT-FLYDSL-GEMM2] compile_gemm2_a4w4_port ENTERED "
        f"BM={BM} use_nt={use_nt} NE={NE} N_OUT={N_OUT} epilog={epilog} D_INTER={D_INTER}",
        flush=True,
    )
    assert BN == 256 and BK == 256, f"only BN==BK==256 supported, got BN={BN} BK={BK}"
    KH_TILE = BK // 2
    _K = D_INTER
    _K_REAL = D_INTER if D_INTER_REAL is None else D_INTER_REAL
    assert _K % BK == 0, (
        f"D_INTER (gemm2 contraction K = inter_dim) must be a multiple of {BK}, "
        f"got {_K}; inter_dim not divisible by {BK} (e.g. 384/192) is not "
        f"supported by this BK={BK} kernel"
    )
    assert (
        _K_REAL % 128 == 0 and 0 < _K_REAL <= _K
    ), f"D_INTER_REAL={_K_REAL} must be a multiple of 128 and in (0, {_K}]"
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _persistent = epilog in ("nonatomic", "nonatomic_mxfp4")
    _slot_bytes = saq_slot_bytes(BM, KH_TILE)
    _aStages = kStages if _K_TILES_TOTAL <= kStages else 3
    _acc_rows = min(BM, 64) if epilog == "nonatomic_cshuffle" else BM
    _lds_bytes = (
        max(lds_acc_bytes_for(_acc_rows, BN), _aStages * _slot_bytes)
        if epilog != "nonatomic"
        else _aStages * _slot_bytes
    )
    _num_n_blocks = num_n_blocks_for(N_OUT, BN)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(
        BM
    )
    _epi_tag = {
        "atomic": "atomic",
        "nonatomic": "nonatomic",
        "nonatomic_mxfp4": "nonatomic_mxfp4",
        "nonatomic_cshuffle": "nonatomic_cshuffle",
    }[epilog]
    _rtag = "" if _K_REAL == _K else f"r{_K_REAL}"
    _tag = f"ne{NE}_h{N_OUT}_i{_K}{_rtag}_bm{BM}{'_nt' if use_nt else ''}_{_epi_tag}"
    _name = f"gemm2_a4w4_port_{_tag}"

    allocator = SmemAllocator(
        None, arch="gfx950", global_sym_name=f"gemm2port_smem_{_tag}"
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
        arg_out_scale: fx.Int64,
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
        aq_rsrc = _buffer_rsrc(arg_aq, _aq_num)
        saq = SmemPtr(
            allocator.get_base(), lds_off, T.i8, shape=(_aStages * _slot_bytes,)
        )

        def _issue_all_a_loads(m_row0):
            for slot in range_constexpr(kStages):
                for sub in range_constexpr(_kSubBlocks):
                    lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
                    car = m_row0 + lds_row + (lane // fx.Int32(8))
                    _issue_a_load_lds(
                        aq_rsrc,
                        saq,
                        slot,
                        slot,
                        car,
                        lane,
                        _slot_bytes,
                        lds_row,
                        KH_TILE=KH_TILE,
                        k_half=_K_HALF,
                    )

        def _run_tile(tile_i32):
            _gemm2_body(
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
                arg_out_scale,
                tile_i32,
                lane,
                wave,
                BM,
                use_nt,
                NE,
                N_OUT,
                epilog,
                aq_rsrc=aq_rsrc,
                D_INTER=_K,
                D_INTER_REAL=_K_REAL,
                aStages=_aStages,
                BN=BN,
                BK=BK,
                KH_TILE=KH_TILE,
            )

        if const_expr(_persistent):
            cumsum0 = llvm.load(T.i32, _global_ptr1(arg_cumsum, fx.Int32(0)))
            total_m_blocks = _udiv(cumsum0, BM)
            bound = total_m_blocks * fx.Int32(_num_n_blocks)
            grid_nb = fx.Int32(gpu.grid_dim.x)

            _NXCD = 8
            _xq = _udiv(bound, _NXCD)
            _xr = _umod(bound, _NXCD)
            _SW = PORT_XCD_SWIZZLE

            def _xcd(pid):
                xc = _umod(pid, _NXCD)
                wgid = (
                    xc * _xq
                    + fx.Int32(arith.minsi(_raw(xc), _raw(_xr)))
                    + _udiv(pid, _NXCD)
                )
                if const_expr(_SW <= 0):
                    return wgid
                _ng = fx.Int32(_SW * _num_n_blocks)
                group_id = wgid // _ng
                first_pid_m = group_id * fx.Int32(_SW)
                remaining_m = total_m_blocks - first_pid_m
                group_size_m = fx.Int32(
                    arith.minsi(_raw(remaining_m), _raw(fx.Int32(_SW)))
                )
                wig = wgid % _ng
                m_block = first_pid_m + (wig % group_size_m)
                n_block = wig // group_size_m
                return m_block * fx.Int32(_num_n_blocks) + n_block

            if bx_i32 < bound:
                tile = _xcd(bx_i32)
                _issue_all_a_loads(_udiv(tile, _num_n_blocks) * fx.Int32(BM))
                rocdl.sched_barrier(0)
                _run_tile(tile)

            saq._view_cache = None
            for iv in range(bx_i32 + grid_nb, bound, gpu.grid_dim.x):
                wu = fx.Int32(iv)
                gpu.barrier()
                setattr(saq, "_view_cache", None)
                tile = _xcd(wu)
                _issue_all_a_loads(_udiv(tile, _num_n_blocks) * fx.Int32(BM))
                _run_tile(tile)
        else:
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

            if bx_i32 < bound:
                _run_tile(bx_i32)

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
        if const_expr(_persistent):
            tw = i32_max_m_blocks * fx.Int32(_num_n_blocks)
            persist = _raw(tw > fx.Int32(NUM_CU * 4))
            grid_i32 = arith.select(persist, _raw(fx.Int32(NUM_CU)), _raw(tw))
            grid_x = arith.index_cast(T.index, grid_i32)
        else:
            grid_x = arith.index_cast(T.index, i32_max_m_blocks) * fx.Index(
                _num_n_blocks
            )
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

    if BM == 16:
        launch_gemm2.compile_hints["llvm_options"] = {"enable-post-misched": False}

    return launch_gemm2


@flyc.jit
def _gemm2_body(
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
    arg_out_scale,
    bx_i32,
    lane,
    wave,
    BM,
    use_nt,
    NE,
    N_OUT,
    epilog,
    *,
    aq_rsrc=None,
    D_INTER,
    D_INTER_REAL=None,
    aStages=kStages,
    BN,
    BK,
    KH_TILE,
):
    _aStages = aStages
    _kMChunks = kmchunks_for(BM)
    _slot_bytes = saq_slot_bytes(BM, KH_TILE)
    _lds_acc_floats = (min(BM, 64) if epilog == "nonatomic_cshuffle" else BM) * BN
    _K = D_INTER
    _K_HALF = k_half_for(_K)
    _K_TILES_TOTAL = k_tiles_total_for(_K, BK)
    _K_REAL = D_INTER if D_INTER_REAL is None else D_INTER_REAL
    _n_real_half = (
        _K_REAL + 127
    ) // 128
    _kUnroll = kunroll_for(_K, BK)
    _kAS_per_chunk_dw = kas_per_chunk_dw_for(_K)
    _kBS_stride_n0_dw = kbs_stride_n0_dw_for(_K)
    _asc_chunk_div = 16 if const_expr(BM == 16) else 32
    _asc_per_mb = (BM // _asc_chunk_div) * _kAS_per_chunk_dw * 4
    _bq_bytes = bq_bytes_for(NE, N_OUT, _K)
    _bscale_bytes = bscale_bytes_for(NE, N_OUT, _K)
    _kbs_per_expert_dw = kbs_per_expert_dw_for(N_OUT, _K)
    _num_n_blocks = num_n_blocks_for(N_OUT, BN)
    _n_load_waves, _rows_per_wave, _kSubBlocks = tiling(
        BM
    )
    b_aux = 2 if use_nt else 0

    m_block_idx = _udiv(bx_i32, _num_n_blocks)
    n_block_idx = bx_i32 - m_block_idx * fx.Int32(_num_n_blocks)
    e = llvm.load(T.i32, _global_ptr1(arg_eids, m_block_idx * fx.Int32(4)))
    e = rocdl.readfirstlane(T.i32, e)
    m_row = m_block_idx * fx.Int32(BM)

    _asc_num = arith.index_cast(T.index, _raw(i32_max_m_blocks)) * fx.Index(_asc_per_mb)
    ascale_rsrc = _buffer_rsrc(arg_ascale, _asc_num)
    bq_rsrc = _buffer_rsrc(arg_bq, fx.Index(_bq_bytes))
    bscale_rsrc = _buffer_rsrc(arg_bscale, fx.Index(_bscale_bytes))

    lds_base = allocator.get_base()
    saq = SmemPtr(lds_base, lds_off, T.i8, shape=(_aStages * _slot_bytes,))
    lds_acc = (
        SmemPtr(lds_base, lds_off, T.f32, shape=(_lds_acc_floats,))
        if epilog != "nonatomic"
        else None
    )

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    b_load_s_base = []
    for j in range_constexpr(4):
        v = (
            e * fx.Int32(N_OUT)
            + n_block_idx * fx.Int32(BN)
            + wave * fx.Int32(BN // 4)
            + fx.Int32(j * 16)
        ) * fx.Int32(_K_HALF)
        b_load_s_base.append(rocdl.readfirstlane(T.i32, v))

    mni_base = n_block_idx * fx.Int32(BN // 16 // 2) + wave * fx.Int32(BN // 64 // 2)
    b_scale_s_base = []
    for mw in range_constexpr(2):
        v = (
            e * fx.Int32(_kbs_per_expert_dw)
            + (mni_base + fx.Int32(mw)) * fx.Int32(_kBS_stride_n0_dw)
        ) * fx.Int32(4)
        b_scale_s_base.append(rocdl.readfirstlane(T.i32, v))

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

    def load_b_scale_tile(kt):
        imm = kt * (kBS_stride_k0_dw * 4)
        out = [None, None]
        for mw in range_constexpr(2):
            out[mw] = buffer_ops.buffer_load(
                bscale_rsrc,
                (v_voff_scale + fx.Int32(imm)) // fx.Int32(4),
                vec_width=1,
                dtype=T.i32,
                soffset_bytes=b_scale_s_base[mw],
            )
        return out

    def load_b_tile(kt):
        v_voff_b = (
            (lane_div_16 * fx.Int32(256))
            + (lane_mod_16 * fx.Int32(16))
            + fx.Int32(kt * 2048)
        )
        out = [[None, None] for _ in range(4)]
        for j in range_constexpr(4):
            for half in range_constexpr(2):
                if const_expr(kt * 2 + half >= _n_real_half):
                    continue
                frag = buffer_ops.buffer_load(
                    bq_rsrc,
                    (v_voff_b + fx.Int32(half * 1024)) // fx.Int32(4),
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=b_aux,
                    soffset_bytes=b_load_s_base[j],
                )
                out[j][half] = Vec(frag)
        return out

    def issue_a_load_lds(slot, kt):
        for sub in range_constexpr(_kSubBlocks):
            lds_row = wave * fx.Int32(_rows_per_wave) + fx.Int32(sub * 8)
            car = m_row + lds_row + (lane // fx.Int32(8))
            _issue_a_load_lds(
                aq_rsrc,
                saq,
                slot,
                kt,
                car,
                lane,
                _slot_bytes,
                lds_row,
                KH_TILE=KH_TILE,
                k_half=_K_HALF,
            )

    def issue_a_ds_read(slot):
        lane_row = lane_mod_16
        lane_col = lane_div_16 * fx.Int32(16)
        mask = _lds_swizzle_mask(lane_row)
        base_ptr = _lds_base_ptr3(saq.get())
        a = [[None, None] for _ in range(_kMChunks)]
        for k in range_constexpr(2):
            lds_col = (lane_col + fx.Int32(k * 64)) ^ mask
            for i in range_constexpr(_kMChunks):
                lds_row = lane_row + fx.Int32(i * 16)
                byte_off = (
                    fx.Int32(slot * _slot_bytes) + lds_row * fx.Int32(KH_TILE) + lds_col
                )
                a[i][k] = llvm.load(
                    T.vec(4, T.i32), _gep3(base_ptr, byte_off)
                )
        return a

    mfma_res_ty = T.f32x4
    zero4 = Vec.filled(4, 0.0, fx.Float32)
    accm = [[None, None, None, None] for _ in range(_kMChunks)]

    def mfma_cluster(b_tile, a, a_scale_sub, b_scale_slot, init, kt=0):
        _skip_h1 = (kt * 2 + 1) >= _n_real_half
        for J in range_constexpr(4):
            mni = J // 2
            in_b = J % 2
            sb = b_scale_slot[mni]
            b_J0 = b_tile[J][0]
            b_J1 = None if const_expr(_skip_h1) else b_tile[J][1]
            for sub in range_constexpr(_kSubBlocks):
                sa = a_scale_sub[sub]
                i0 = sub * 2
                i1 = sub * 2 + 1
                if const_expr(init):
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty, [a[i0][0], b_J0, zero4, 4, 4, 0, sa, 0 + in_b, sb]
                    )
                    if const_expr(_kMChunks > 1):
                        accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [a[i1][0], b_J0, zero4, 4, 4, 1, sa, 0 + in_b, sb],
                        )
                else:
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [a[i0][0], b_J0, accm[i0][J], 4, 4, 0, sa, 0 + in_b, sb],
                    )
                    if const_expr(_kMChunks > 1):
                        accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [a[i1][0], b_J0, accm[i1][J], 4, 4, 1, sa, 0 + in_b, sb],
                        )
                if const_expr(not _skip_h1):
                    accm[i0][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        mfma_res_ty,
                        [a[i0][1], b_J1, accm[i0][J], 4, 4, 2, sa, 2 + in_b, sb],
                    )
                    if const_expr(_kMChunks > 1):
                        accm[i1][J] = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            mfma_res_ty,
                            [a[i1][1], b_J1, accm[i1][J], 4, 4, 3, sa, 2 + in_b, sb],
                        )

    def _kloop_fence():
        gpu.barrier()

    if const_expr(_K_TILES_TOTAL <= kStages):
        a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b_scale_v = [load_b_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b = [load_b_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        for S in range_constexpr(_K_TILES_TOTAL):
            kt = S
            slot = kt % kStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b[slot], a, a_scale_sub, b_scale_v[slot], init=(S == 0), kt=kt)
    else:
        a_scale_v = [load_a_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b_scale_v = [load_b_scale_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]
        b = [load_b_tile(kt) for kt in range_constexpr(_K_TILES_TOTAL)]

        for OFFSET in range_constexpr(_kUnroll):
            kt = OFFSET
            slot = kt % _aStages
            next_kt = kStages + OFFSET
            write_slot = next_kt % _aStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            issue_a_load_lds(write_slot, next_kt)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b[kt], a, a_scale_sub, b_scale_v[kt], init=(OFFSET == 0))

        for S in range_constexpr(kStages):
            kt = _K_TILES_TOTAL - kStages + S
            slot = kt % _aStages
            _kloop_fence()
            a = issue_a_ds_read(slot)
            a_scale_sub = [a_scale_v[kt][sub] for sub in range_constexpr(_kSubBlocks)]
            mfma_cluster(b[kt], a, a_scale_sub, b_scale_v[kt], init=False)

    saq._view_cache = None
    if epilog == "nonatomic":
        out_base = _global_base_ptr1(arg_out)
        _flat_bf16_epilog(
            accm, out_base, m_row, n_block_idx, wave, lane, N_OUT, BN, _kMChunks
        )
    elif epilog == "nonatomic_cshuffle":
        lds_acc._view_cache = None
        _cshuffle_flat_bf16_epilog(
            lds_acc, accm, arg_out, m_row, n_block_idx, wave, lane, BM, N_OUT, BN
        )
    elif epilog == "nonatomic_mxfp4":
        out_q_base = _global_base_ptr1(arg_out)
        out_scale_base = _global_base_ptr1(arg_out_scale)
        tid_i32 = fx.Int32(gpu.thread_id("x"))
        _flat_mxfp4_epilog(
            accm,
            out_q_base,
            out_scale_base,
            m_row,
            n_block_idx,
            wave,
            lane,
            tid_i32,
            N_OUT,
            BN,
            lds_acc,
            _kMChunks,
        )
    else:
        lds_acc._view_cache = None
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
            BN,
        )


def _flat_bf16_epilog(
    accm, out_base, m_row, n_block_idx, wave, lane, N_OUT, BN, kMChunks
):
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    row_base = m_row + lane_div_16 * fx.Int32(4)
    gn_base = n_block_idx * fx.Int32(BN) + wave * fx.Int32(BN // 4) + lane_mod_16
    byte_base = (fx.Int64(row_base) * fx.Int64(N_OUT) + fx.Int64(gn_base)) * fx.Int64(2)
    for i in range_constexpr(kMChunks):
        for J in range_constexpr(4):
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                const_off = ((i * 16 + v) * N_OUT + J * 16) * 2
                bf = Vec.from_elements([vec[v]], fx.Float32).to(fx.BFloat16)
                llvm.StoreOp(_raw(bf), _gep1(out_base, byte_base + fx.Int64(const_off)))


def _cshuffle_flat_bf16_epilog(
    lds_acc, accm, arg_out, m_row, n_block_idx, wave, lane, BM, N_OUT, BN
):
    _iC = BM // 16
    _REPS = BM // 8
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_base_ptr3(lds_acc.get())
    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    out_base = _global_base_ptr1(arg_out)

    gpu.barrier()
    for i in range_constexpr(_iC):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            bf4 = Vec(accm[i][J]).to(fx.BFloat16)
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(bf4[v]), _gep3(lds_base, idx * fx.Int32(2)))
    gpu.barrier()
    for mr in range_constexpr(_REPS):
        row_local = fx.Int32(mr * 8) + m_lane
        sorted_row = m_row + row_local
        for s in range_constexpr(4):
            idx0 = row_local * fx.Int32(BN) + col_start + fx.Int32(s * 64)
            pk = Vec(llvm.load(T.vec(2, T.bf16), _gep3(lds_base, idx0 * fx.Int32(2))))
            n_col = n_block_idx * fx.Int32(BN) + col_start + fx.Int32(s * 64)
            elem = fx.Int64(sorted_row) * fx.Int64(N_OUT) + fx.Int64(n_col)
            llvm.StoreOp(_raw(pk), _gep1(out_base, elem * fx.Int64(2)))


@flyc.jit
def _flat_mxfp4_epilog(
    accm,
    out_q_base,
    out_scale_base,
    m_row,
    n_block_idx,
    wave,
    lane,
    tid_i32,
    N_OUT,
    BN,
    lds_acc,
    kMChunks,
):
    lds_base = _lds_base_ptr3(lds_acc.get())
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    for i in range_constexpr(kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(BN // 4) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))
    gpu.barrier()

    NBLK = BN // 32
    m_lane = tid_i32 // fx.Int32(16)
    n_lane = tid_i32 % fx.Int32(16)
    wave_grp = n_lane // fx.Int32(4)
    kk = n_lane % fx.Int32(4)
    _m_base = m_row + m_lane
    _q_row0 = fx.Int64(_m_base) * fx.Int64(N_OUT // 2)
    _s_row0 = fx.Int64(_m_base) * fx.Int64(N_OUT // 32)
    _blocks = [(mr, half) for mr in range(kMChunks) for half in range(NBLK // 4)]

    def _issue_load(mr, half):
        row_local = fx.Int32(mr * 16) + m_lane
        group = wave_grp + fx.Int32(half * 4)
        col0 = group * fx.Int32(32) + kk * fx.Int32(8)
        base_idx = row_local * fx.Int32(BN) + col0
        v0 = Vec(llvm.load(T.vec(4, T.f32), _gep3(lds_base, base_idx * fx.Int32(4))))
        v1 = Vec(
            llvm.load(
                T.vec(4, T.f32),
                _gep3(lds_base, (base_idx + fx.Int32(4)) * fx.Int32(4)),
            )
        )
        return [v0[0], v0[1], v0[2], v0[3], v1[0], v1[1], v1[2], v1[3]], group, col0

    _r_next, _grp_next, _col0_next = _issue_load(*_blocks[0])
    for _bi in range_constexpr(len(_blocks)):
        mr, half = _blocks[_bi]
        r, group, col0 = _r_next, _grp_next, _col0_next
        if _bi + 1 < len(_blocks):
            _r_next, _grp_next, _col0_next = _issue_load(*_blocks[_bi + 1])
        if True:
            amax_f = _raw(_fabs_f32(r[0]))
            for e in range_constexpr(1, 8):
                abs_e = _raw(_fabs_f32(r[e]))
                amax_f = arith.maxnumf(amax_f, abs_e)
            amax = arith.shrui(arith.bitcast(T.i32, amax_f), _raw(fx.Int32(16)))
            amax_dpp = _raw(_inline_dpp_quad_amax(amax))
            f32b = arith.shli(amax_dpp, _raw(fx.Int32(16)))
            e8m0, qscale_f = _e8m0_from_amax(fx.Float32(arith.bitcast(T.f32, f32b)))
            e8 = _raw(e8m0)
            qscale = _raw(qscale_f)
            packed = _raw(fx.Int32(0))
            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, _raw(r[0]), _raw(r[1]), qscale, 0
            )
            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, _raw(r[2]), _raw(r[3]), qscale, 1
            )
            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, _raw(r[4]), _raw(r[5]), qscale, 2
            )
            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                T.i32, packed, _raw(r[6]), _raw(r[7]), qscale, 3
            )
            global_col = n_block_idx * fx.Int32(BN) + col0
            blk = n_block_idx * fx.Int32(NBLK) + group
            q_byte = (
                _q_row0
                + fx.Int64(mr * 16 * (N_OUT // 2))
                + fx.Int64(global_col // fx.Int32(2))
            )
            s_byte = _s_row0 + fx.Int64(mr * 16 * (N_OUT // 32)) + fx.Int64(blk)
            llvm.StoreOp(packed, _gep1(out_q_base, q_byte), nontemporal=True)
            if kk == fx.Int32(0):
                llvm.StoreOp(arith.trunci(T.i8, e8), _gep1(out_scale_base, s_byte))


@flyc.jit
def _atomic_bf16_epilog(
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
    BN,
):
    _kMChunks = kmchunks_for(BM)
    M_REPS = BM // 8
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)
    lds_base = _lds_base_ptr3(lds_acc.get())

    tx_i32 = fx.Int32(gpu.thread_id("x"))
    m_lane = tx_i32 // fx.Int32(32)
    n_lane = tx_i32 % fx.Int32(32)
    col_start = n_lane * fx.Int32(2)
    stids_base = _global_base_ptr1(arg_stids)
    sweights_base = _global_base_ptr1(arg_sweights)
    out_base = _global_base_ptr1(arg_out)

    packed = []
    weight = []
    for mr in range_constexpr(M_REPS):
        sorted_pos = m_row + fx.Int32(mr * 8) + m_lane
        packed.append(
            llvm.load(
                T.i32, _gep1(stids_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )
        weight.append(
            llvm.load(
                T.f32, _gep1(sweights_base, sorted_pos * fx.Int32(4)), invariant=True
            )
        )

    gpu.barrier()

    for i in range_constexpr(_kMChunks):
        row_base = fx.Int32(i * 16) + lane_div_16 * fx.Int32(4)
        for J in range_constexpr(4):
            col = wave * fx.Int32(64) + fx.Int32(J * 16) + lane_mod_16
            vec = Vec(accm[i][J])
            for v in range_constexpr(4):
                idx = (row_base + fx.Int32(v)) * fx.Int32(BN) + col
                llvm.StoreOp(_raw(vec[v]), _gep3(lds_base, idx * fx.Int32(4)))

    gpu.barrier()

    for mr in range_constexpr(M_REPS):
        row_in_block = fx.Int32(mr * 8) + m_lane
        token_id = packed[mr] & fx.Int32(0x00FFFFFF)
        if token_id < i32_M:
            row_base_addr = (
                token_id * fx.Int32(N_OUT) + n_block_idx * fx.Int32(BN) + col_start
            )
            for s in range_constexpr(4):
                idx0 = row_in_block * fx.Int32(BN) + col_start + fx.Int32(s * 64)
                v2 = Vec(
                    llvm.load(T.vec(2, T.f32), _gep3(lds_base, idx0 * fx.Int32(4)))
                )
                pk = Vec.from_elements(
                    [v2[0] * weight[mr], v2[1] * weight[mr]], fx.Float32
                ).to(fx.BFloat16)
                off = (row_base_addr + fx.Int32(s * 64)) * fx.Int32(
                    2
                )
                out_ptr = _gep1(out_base, off)
                llvm.AtomicRMWOp(
                    llvm.AtomicBinOp.fadd,
                    out_ptr,
                    _raw(pk),
                    llvm.AtomicOrdering.monotonic,
                    syncscope="agent",
                    alignment=4,
                )

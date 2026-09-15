# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 FlyDSL Project Contributors

"""MXFP4/MXFP6/MXFP8 A x MXFP4/MXFP8 B preshuffle GEMM (gfx950): per-32 E8M0 scales folded
into a scaled 16x16x128 fx.gemm; A streams global->LDS via double-buffered async DMA. Layout
matches the host preshuffle (shuffle_weight_w4(.,16) + shuffle_scale_w4).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
    Constexpr,
    Float4E2M1FN,
    Float6E2M3FN,
    Float8E4M3FN,
    Float16,
    Float32,
    Int8,
    Int32,
    T,
)
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxscale_preshuffle_common import (
    _DTYPE_SHORT,
    make_kernel_name,
)
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    atomic_add_agent,
    load_i32_nt,
)

_A_ELEM = {"fp4": Float4E2M1FN, "fp6": Float6E2M3FN, "fp8": Float8E4M3FN}
_B_ELEM = {"fp4": Float4E2M1FN, "fp8": Float8E4M3FN}


def _make_dma_layouts(tile_m, tile_k_bytes, row_stride, num_threads, *, swizzle):
    """Map cooperative 16-byte A copies onto global rows/K blocks and LDS."""
    rounds = tile_m * tile_k_bytes // (num_threads * 16)
    waves = num_threads // 64
    blocks_per_row = tile_k_bytes // 16
    rows_per_wave = 64 // blocks_per_row
    shape = (16, (blocks_per_row, rows_per_wave), waves, rounds)

    dma_to_coord = fx.make_layout(
        shape,
        (
            fx.E(1),
            (16 * fx.E(1), fx.E(0)),
            rows_per_wave * fx.E(0),
            rows_per_wave * waves * fx.E(0),
        ),
    )
    if swizzle:
        bits = blocks_per_row.bit_length() - 1
        coord_swizzle = fx.static(fx.CoordSwizzleType.get(bits, 0, [0], 4, [1]))
        dma_to_coord = fx.make_composed_layout(coord_swizzle, dma_to_coord)
    source = fx.make_composed_layout(
        fx.make_layout((tile_m, tile_k_bytes), (row_stride, 1)), dma_to_coord
    )
    destination = fx.make_layout((16, waves, rounds), (1, 1024, num_threads * 16))
    return source, destination


def _scale_mma_atoms(a_dtype, b_dtype):
    """16 (opsel_a, opsel_b) scaled-MFMA atoms; A elem is fp4/fp6/fp8, B is fp4/fp8."""
    elem_a = _A_ELEM[a_dtype]
    elem_b = _B_ELEM[b_dtype]
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, elem_a, elem_b, opsel_a=osa, opsel_b=osb
            )
        )
        for osa in range(4)
        for osb in range(4)
    }


def _bq_view(arg_bq_addr, row_elems, KH4, k_tiles, k_halves, pair):
    """Preshuffled B view for one N-row tile; index [l//16, l%16, kt, half, p, None] -> i32[4].

    `pair` = K0 blocks per 128-K MFMA: 1 for fp4 B (one i32[4]), 2 for fp8 B (lo/hi halves
    packed into i32[8] by load_b). K0 blocks (256 i32 each) run contiguously along K as
    ((kt*k_halves + kh)*pair + p); the fp4 case keeps a size-1 `p` dim (byte-identical view).
    """
    col_base = rocdl.readfirstlane(T.i32, row_elems * KH4)
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=16
    )
    off_i64 = fx.Int64(col_base)
    base_iter = fx.inttoptr(i32_ptr_ty, arg_bq_addr + off_i64 * fx.Int64(4))
    shape = (4, 16, k_tiles, k_halves, pair, 4)
    strides = (64, 4, k_halves * pair * 256, pair * 256, 256, 1)
    view = fx.Tensor(fx.make_view(base_iter, fx.make_layout(shape, strides)))
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


@flyc.jit
def _launch_gemm_impl(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    arg_out: fx.Pointer,
    arg_semaphore: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    a_dtype: Constexpr[str],
    out_dtype: Constexpr[str],
    b_dtype: Constexpr[str],
    batch: Constexpr[int],
    a_row_stride: Constexpr[int],
    a_batch_stride: Constexpr[int],
    sca_row_stride: Constexpr[int],
    sca_batch_stride: Constexpr[int],
    c_row_stride: Constexpr[int],
    c_batch_stride: Constexpr[int],
    waves_per_eu: Constexpr[int],
    xcd_swizzle: Constexpr[int],
    k_batch: Constexpr[int] = 1,
    blockscale: Constexpr[str] = "none",
    multi_row_tile: Constexpr[bool] = False,
):
    """Direct @flyc.jit launcher. Operands are fx.Pointer (pass ptr_arg(t): raw data_ptr, no
    per-launch DLPack). Compile once with flyc.compile, then cf(*runtime). a_dtype fp4/fp6/fp8
    A x preshuffled b_dtype (fp4/fp8) B, e8m0 scales (a8w8 = a_dtype=fp8, b_dtype=fp8).
    batch>1 = strided-batched over grid.z. The a_/sca_/c_ row/batch strides make A/scale_a/C
    addressing caller-controlled; each <0 keeps the contiguous [B,M,*] bmn default, all set =
    the [M,B,*] mbn layout. waves_per_eu<=0 = unset.
    """
    BM, BN, BK = tile_m, tile_n, tile_k
    small_m_bf16 = (
        BM == 16
        and blockscale != "none"
        and out_dtype == "bf16"
        and batch == 1
        and c_row_stride < 0
        and c_batch_stride < 0
    )
    splitk_fused = small_m_bf16 and k_batch > 1 and BN <= 128
    if const_expr(out_dtype == "bf16"):
        out_elem = BFloat16
    else:
        out_elem = Float16

    # Row sizes + read_a fragment layout (i32 units): fp6/fp8 read two b128 halves -> i32[A_NDW], fp4 one -> i32[4].
    if const_expr(a_dtype == "fp4"):  # 2 codes/byte
        a_row_bytes, A_ROW_B = K // 2, BK // 2
        A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 16, 0, 4
    else:
        a_row_bytes, A_ROW_B = K, BK
        if const_expr(a_dtype == "fp8"):
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 32, 16, 8
        else:  # fp6
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 8, 32, 4, 6

    A_LDS_B = (
        BM * A_ROW_B
    )  # LDS A buffer bytes (row-major [m][col], shared by 4 N-waves)
    A_ROW_I32 = A_ROW_B // 4
    swz_lds = a_dtype in ("fp4", "fp8")
    k_blk16 = A_ROW_B // 16
    # B fragment layout: fp8 B needs i32[8] (two K0 blocks / 128-K MFMA); fp4 B i32[4] (one).
    if const_expr(b_dtype == "fp8"):
        b_row_bytes, B_NDW, B_BLK_PER_MMA = K, 8, 2
    else:  # fp4
        b_row_bytes, B_NDW, B_BLK_PER_MMA = K // 2, 4, 1
    KH4 = b_row_bytes // 4  # i32 per N-row in preshuffled B (== (K//2)//4 for fp4)
    K_TILES = K // BK
    # split-K (k_batch>1): each grid.z split reduces k_tiles_local = K_TILES//k_batch
    # K-tiles. The M<=16 BF16 specialization writes a compact BF16 workspace and
    # reduces it in the last-arriving GEMM block; other variants write fp32 partial
    # slabs followed by a separate reduce kernel. K/k_batch is 256-K aligned (see
    # fits_shape), so a split boundary lands on whole tiles + scale chunks.
    assert K_TILES % k_batch == 0, "K_TILES must be divisible by k_batch"
    k_tiles_local = K_TILES // k_batch
    k_halves = BK // 128  # 16x16x128 MFMA k-steps per K-tile
    # e8m0 scales are 256-K granular, B 128-K: tiles_per_chunk K-tiles share a word (hi/lo 16b = 128-K half).
    tiles_per_chunk = 256 // BK  # 1 for tile_k=256, 2 for tile_k=128
    m_chunks = BM // 16
    num_waves = min(4, BN // 16)
    num_threads = num_waves * 64
    num_acc_n = (BN // num_waves) // 16  # 16-col n-subblocks per wave
    _scale_chunk_dw = ((K + 255) // 256) * 64  # e8m0 stride in dwords
    _scale_k0_dw = 64
    # blockscale (A 1x128 / B 128x128): feed the coarse scale to the 1x32 MFMA by
    # broadcasting in the load ADDRESS. A drops K_Lane(4); B drops K_Lane*N_Lane(64)
    # and reads one word per 128-N-block (nsb//4).
    _bs_a = blockscale in ("a", "ab")
    _bs_b = blockscale in ("b", "ab")
    _sc_k0_a = 16 if _bs_a else 64  # per-256K-chunk dword stride (A: drop K_Lane=4)
    _sc_k0_b = 1 if _bs_b else 64  # (B: drop K_Lane*N_Lane=64)
    _scale_chunk_dw_a = ((K + 255) // 256) * _sc_k0_a
    _scale_chunk_dw_b = ((K + 255) // 256) * _sc_k0_b
    _b_sc_rows = (N // 128) if _bs_b else (N // 32)  # B scale super-rows
    a_copy_granularity = num_threads * 16
    assert A_LDS_B % a_copy_granularity == 0, (
        f"A_LDS_B ({A_LDS_B}B) must be divisible by num_threads*16 "
        f"({a_copy_granularity}B)"
    )
    n_coop = A_LDS_B // a_copy_granularity  # 16B cooperative loads per thread
    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)

    # Scheduler counts per loop iter: MFMAs, A LDS reads/thread (fp6/fp8 2 per (mi,kh)), gmem loads.
    sched_mfma_total = k_halves * m_chunks * num_acc_n
    if const_expr(a_dtype == "fp4"):
        a_ds_per = 1
    else:
        a_ds_per = 2
    sched_num_ds_load = m_chunks * k_halves * a_ds_per
    sched_num_gmem = n_coop + num_acc_n * k_halves * B_BLK_PER_MMA + m_pairs + n_pairs

    @fx.struct
    class SharedA:
        a0: fx.Array[Int8, A_LDS_B, 16]
        a1: fx.Array[Int8, A_LDS_B, 16]

    # Name the GPU symbol after the tuned-CSV kernelName instead of letting FlyDSL
    # fall back to "kernel_gemm_<id>" -- otherwise every config profiles under the
    # same symbol and rocprof/att traces can't tell them apart.
    _kname = make_kernel_name(
        BM,
        BN,
        BK,
        a_dtype,
        b_dtype,
        out_dtype,
        waves_per_eu,
        xcd_swizzle,
        k_batch,
        blockscale,
    )
    if splitk_fused:
        _kname += "_fused_reduce"
    if multi_row_tile:
        _kname += "_multirow"

    def _kernel_body(
        arg_c: fx.Int64,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        arg_out: fx.Int64,
        arg_semaphore: fx.Int64,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        scale_atoms = _scale_mma_atoms(a_dtype, b_dtype)

        tid = fx.Int32(fx.thread_idx.x)
        bid_x, bid_y, bid_z = fx.block_idx
        # split-K: grid.z = batch*k_batch, bid_z = bz_batch*k_batch + ks. ks (=bid_z%k_batch)
        # selects this split's K range [kt0, kt0+k_tiles_local); bz_batch is the real batch
        # index. When k_batch==1, kt0 folds to 0 and bz_batch==bid_z (no-split codegen).
        if const_expr(k_batch > 1):
            bz_batch = bid_z // k_batch
            kt0 = fx.Int32(bid_z % k_batch) * fx.Int32(k_tiles_local)
        else:
            bz_batch = bid_z
            kt0 = fx.Int32(0)
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        lane = tid % 64
        lane_div_16 = lane // 16
        lane_mod_16 = lane % 16
        # XCD swizzle: remap (bid_x, bid_y) for L2-cache reuse (no-op when xcd_swizzle<=0).
        if const_expr(xcd_swizzle > 0):
            from ..mfma_preshuffle_pipeline import xcd_remap_bx_by

            _bx, _by = xcd_remap_bx_by(
                fx.Index(bid_x),
                fx.Index(bid_y),
                fx.Index(i32_m),
                tile_m=BM,
                tile_n=BN,
                N=N,
                xcd_swizzle=xcd_swizzle,
            )
            bx_m = fx.Int32(_bx) * BM
            by_n = fx.Int32(_by) * BN
        else:
            bx_m = bid_x * BM
            by_n = bid_y * BN

        # Strided-batched: shift each base to batch bid_z (A/scale_a via explicit strides or
        # the contiguous default; B/scale_b stay batch-contiguous). batch==1 emits no batch math.
        if const_expr(batch > 1):
            a_rstride = fx.Int32(a_row_bytes if a_row_stride < 0 else a_row_stride)
            sca_rstride = fx.Int32(
                _scale_chunk_dw_a if sca_row_stride < 0 else sca_row_stride
            )
            bz = fx.Int64(bz_batch)
            if const_expr(a_batch_stride < 0):
                arg_a = arg_a + bz * (fx.Int64(i32_m) * fx.Int64(a_row_bytes))
            else:
                arg_a = arg_a + bz * fx.Int64(a_batch_stride)
            arg_b = arg_b + bz * fx.Int64(N * b_row_bytes)
            if const_expr(sca_batch_stride < 0):
                sc_bstride = (
                    fx.Int64((i32_m + 31) // 32)
                    * fx.Int64(_scale_chunk_dw_a)
                    * fx.Int64(4)
                )
                arg_scale_a = arg_scale_a + bz * sc_bstride
            else:
                arg_scale_a = arg_scale_a + bz * fx.Int64(sca_batch_stride)
            arg_scale_b = arg_scale_b + bz * fx.Int64(
                _b_sc_rows * _scale_chunk_dw_b * 4
            )
        else:
            a_rstride = fx.Int32(a_row_bytes)
            sca_rstride = fx.Int32(_scale_chunk_dw_a)

        # A source view, bound to the last valid M row (ragged M OOB -> 0).
        _i8g = fx.PointerType.get(
            T.i8, address_space=fx.AddressSpace.Global, alignment=16
        )
        if const_expr(batch > 1 and a_row_stride >= 0):
            a_nrec = fx.Int64(i32_m - fx.Int32(1)) * fx.Int64(a_rstride) + fx.Int64(
                a_row_bytes
            )
        else:
            a_nrec = fx.Int64(i32_m) * fx.Int64(a_row_bytes)
        a_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(
                    fx.inttoptr(_i8g, arg_a),
                    fx.make_layout(65536 * a_row_bytes, 1),
                )
            ),
            max_size=False,
            num_records_bytes=a_nrec,
        )
        lds = fx.SharedAllocator().allocate(SharedA).peek()
        # A-LDS modeled as i32 (16B = 4 i32): fx.copy is dtype-agnostic, only the MMA cares.
        sA0_i32 = fx.recast_iter(Int32, lds.a0.ptr)
        lds_db = fx.Int32(fx.ptrtoint(lds.a1.ptr)) - fx.Int32(
            fx.ptrtoint(lds.a0.ptr)
        )  # ping/pong byte stride
        lds_db_i32 = lds_db // 4
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        _i8s = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
        sA0_i8 = fx.recast_iter(_i8s, lds.a0.ptr)

        def _iter_of(parity):  # parity in {0,1} (runtime) -> i32 LDS iterator
            return fx.add_offset(sA0_i32, parity * lds_db_i32)

        if const_expr(BM == 16):
            # The small-M tile uses layout algebra for both the global->LDS
            # DMA and LDS reads. Keep the established arithmetic path below for the
            # wider tiles: materializing these layouts increases VGPR pressure on a
            # few existing CSV kernels and can reduce their occupancy.
            dma_src_layout, dma_dst_layout = _make_dma_layouts(
                BM, A_ROW_B, a_rstride, num_threads, swizzle=swz_lds
            )
            dma_blocks_per_row = A_ROW_B // 16
            dma_lane = lane % dma_blocks_per_row, lane // dma_blocks_per_row
            dma_a_base = fx.add_offset(fx.get_iter(a_flat), bx_m * a_rstride)

            # (dword, row lane, K lane, M tile, K half, register half).
            a_read_layout = fx.make_layout(
                (4, 16, 4, m_chunks, k_halves, 1 if a_dtype == "fp4" else 2),
                (1, A_ROW_I32, A_GK_I32, 16 * A_ROW_I32, A_KH_I32, A_HI_OFF),
            )
            if const_expr(swz_lds):
                swz_bits = k_blk16.bit_length() - 1
                a_read_layout = fx.make_composed_layout(
                    fx.static(fx.SwizzleType.get(swz_bits, 2, swz_bits)),
                    a_read_layout,
                )

            def dma_a_to_lds(kt, parity):
                src = fx.Tensor(
                    fx.make_view(
                        fx.add_offset(dma_a_base, kt * A_ROW_B), dma_src_layout
                    )
                )
                base_off = rocdl.readfirstlane(T.i32, parity * lds_db)
                dst = fx.Tensor(
                    fx.make_view(fx.add_offset(sA0_i8, base_off), dma_dst_layout)
                )
                for i in range_constexpr(n_coop):
                    fx.copy(
                        dma_atom,
                        src[None, dma_lane, wave, i],
                        dst[None, wave, i],
                    )

            def _read16(src):
                t = fx.make_rmem_tensor(4, Int32)
                fx.copy(lds_copy, src, t)
                return t

            def read_a(parity):
                src = fx.Tensor(fx.make_view(_iter_of(parity), a_read_layout))
                av = []
                for mi in range_constexpr(m_chunks):
                    for kh in range_constexpr(k_halves):
                        lo = _read16(src[None, lane_mod_16, lane_div_16, mi, kh, 0])
                        if const_expr(a_dtype == "fp4"):
                            av.append(lo)
                        else:
                            hi = _read16(src[None, lane_mod_16, lane_div_16, mi, kh, 1])
                            t = fx.make_rmem_tensor(A_NDW, Int32)
                            t.store(lo.load().shuffle(hi.load(), list(range(A_NDW))))
                            av.append(t)
                return av

        else:
            a_flat_div = fx.logical_divide(a_flat, fx.make_layout(1, 1))

            def _lds_view(base_iter, off_i32):
                return fx.make_view(
                    fx.add_offset(base_iter, off_i32), fx.make_layout(4, 1)
                )

            def dma_a_to_lds(kt, parity):
                base_off = rocdl.readfirstlane(
                    T.i32, parity * lds_db + wave * (64 * 16)
                )
                lds_ptr = fx.add_offset(sA0_i8, base_off)
                base_k_byte = kt * A_ROW_B
                for i in range_constexpr(n_coop):
                    if const_expr(i > 0):
                        lds_ptr = fx.add_offset(lds_ptr, fx.Int32(num_threads * 16))
                    lin = (i * num_threads + tid) * 16
                    row = lin // A_ROW_B
                    col = lin % A_ROW_B
                    if const_expr(swz_lds):
                        col = col ^ ((row % k_blk16) * 16)
                    gmem_byte = (bx_m + row) * a_rstride + base_k_byte + col
                    dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                    src = fx.slice(a_flat_div, (None, gmem_byte))
                    fx.copy(dma_atom, src, dst)

            def _read16(base_iter, off_i32):
                t = fx.make_rmem_tensor(4, Int32)
                fx.copy(lds_copy, _lds_view(base_iter, off_i32), t)
                return t

            def read_a(parity):
                base_iter = _iter_of(parity)
                av = []
                for mi in range_constexpr(m_chunks):
                    for kh in range_constexpr(k_halves):
                        row = mi * 16 + lane_mod_16
                        row_base = row * A_ROW_I32
                        lo_blk = kh * (A_KH_I32 // 4) + lane_div_16 * (A_GK_I32 // 4)
                        if const_expr(swz_lds):
                            off = row_base + (lo_blk ^ (row % k_blk16)) * 4
                        else:
                            off = row_base + kh * A_KH_I32 + lane_div_16 * A_GK_I32
                        if const_expr(a_dtype == "fp4"):
                            av.append(_read16(base_iter, off))
                        else:
                            if const_expr(swz_lds):
                                hi_off = (
                                    row_base
                                    + ((lo_blk + A_HI_OFF // 4) ^ (row % k_blk16)) * 4
                                )
                            else:
                                hi_off = off + A_HI_OFF
                            lo = Vec(fx.memref_load_vec(_read16(base_iter, off)))
                            hi = Vec(fx.memref_load_vec(_read16(base_iter, hi_off)))
                            t = fx.make_rmem_tensor(A_NDW, Int32)
                            t.store(lo.shuffle(hi, list(range(A_NDW))))
                            av.append(t)
                return av

        n_col_base = by_n + wave * (BN // num_waves)
        bq_views = [
            _bq_view(arg_b, n_col_base + ni * 16, KH4, K_TILES, k_halves, B_BLK_PER_MMA)
            for ni in range_constexpr(num_acc_n)
        ]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        bs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        # e8m0 scale buffers bounded to real size (OOB rows read 0); scale_a to the last 32-row chunk.
        _i32g = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        _sc_layout = fx.make_layout(1 << 28, 1)
        _a_sc_chunks = (i32_m + 31) // 32
        if const_expr(batch > 1 and sca_row_stride >= 0):
            a_sc_nrec = (
                fx.Int64(_a_sc_chunks - 1) * fx.Int64(sca_rstride)
                + fx.Int64(_scale_chunk_dw_a)
            ) * fx.Int64(4)
        else:
            a_sc_nrec = (
                fx.Int64(_a_sc_chunks) * fx.Int64(_scale_chunk_dw_a) * fx.Int64(4)
            )
        b_sc_nrec = fx.Int64(_b_sc_rows * _scale_chunk_dw_b * 4)
        sa_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, arg_scale_a), _sc_layout)),
                max_size=False,
                num_records_bytes=a_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        sb_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(fx.make_view(fx.inttoptr(_i32g, arg_scale_b), _sc_layout)),
                max_size=False,
                num_records_bytes=b_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        a_sc_base = [(bx_m // 32 + mp) * sca_rstride for mp in range_constexpr(m_pairs)]
        nsb = (by_n + wave * (BN // num_waves)) // 32
        # B blockscale: 4 consecutive 32-N super-rows share one 128-N block scale.
        b_sc_base = [
            ((nsb + np) // 4 if _bs_b else (nsb + np)) * _scale_chunk_dw_b
            for np in range_constexpr(n_pairs)
        ]
        sc_lane = lane_div_16 * 16 + lane_mod_16
        # blockscale drops broadcast dims from the per-lane offset: A drops K_Lane
        # (lane_div_16); B drops K_Lane + N_Lane (all lanes read one word).
        sc_lane_a = lane_mod_16 if _bs_a else sc_lane
        sc_lane_b = fx.Int32(0) if _bs_b else sc_lane

        n_acc = m_chunks * num_acc_n

        def load_b(kt):
            # buffer_load_dwordx4 into i32[4] frags; fp8 B packs two K0 blocks (lo/hi, 64 K
            # apart, f8f6f4 ABI) into i32[8] — same shuffle as read_a's fp6/fp8 A path.
            ops = []
            for ni in range_constexpr(num_acc_n):
                for kh in range_constexpr(k_halves):
                    lo = fx.make_rmem_tensor(4, Int32)
                    fx.copy_atom_call(
                        b_copy,
                        bq_views[ni][lane_div_16, lane_mod_16, kt, kh, 0, None],
                        lo,
                    )
                    if const_expr(b_dtype == "fp4"):
                        ops.append(lo)
                    else:  # fp8: lo ++ hi -> i32[B_NDW]
                        hi = fx.make_rmem_tensor(4, Int32)
                        fx.copy_atom_call(
                            b_copy,
                            bq_views[ni][lane_div_16, lane_mod_16, kt, kh, 1, None],
                            hi,
                        )
                        t = fx.make_rmem_tensor(B_NDW, Int32)
                        t.store(
                            Vec(fx.memref_load_vec(lo)).shuffle(
                                Vec(fx.memref_load_vec(hi)), list(range(B_NDW))
                            )
                        )
                        ops.append(t)
            return ops

        def load_sc(chunk_kt):
            # (sa, sb) e8m0 words per m-/n-pair for one 256-K chunk (uniform base -> SGPR soffset).
            # blockscale uses smaller per-chunk strides (compact buffer) + broadcast lanes.
            koff_a = chunk_kt * _sc_k0_a
            koff_b = chunk_kt * _sc_k0_b
            sa = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sa_flat[
                            None,
                            rocdl.readfirstlane(T.i32, a_sc_base[mp] + koff_a)
                            + sc_lane_a,
                        ],
                    )
                )[0]
                for mp in range_constexpr(m_pairs)
            ]
            # blockscale B: a wave's N-span lies in one 128-N block (per-wave N =
            # BN//num_waves <= 128, block-aligned), so every n_pair reads the SAME
            # word -> load once and reuse instead of n_pairs redundant loads.
            _bs_b_dedup = (
                _bs_b and (BN // num_waves) <= 128 and (128 % (BN // num_waves) == 0)
            )
            if const_expr(_bs_b_dedup):
                _sb0 = Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sb_flat[
                            None,
                            rocdl.readfirstlane(T.i32, b_sc_base[0] + koff_b)
                            + sc_lane_b,
                        ],
                    )
                )[0]
                sb = [_sb0 for _ in range_constexpr(n_pairs)]
            else:
                sb = [
                    Vec(
                        fly.copy_atom_call_ssa(
                            [T.vec(1, T.i32)],
                            bs_copy,
                            sb_flat[
                                None,
                                rocdl.readfirstlane(T.i32, b_sc_base[np] + koff_b)
                                + sc_lane_b,
                            ],
                        )
                    )[0]
                    for np in range_constexpr(n_pairs)
                ]
            return sa, sb

        def compute(accs, av, bv, sa_v, sb_v, scale_shift=None):
            # tile_k=128: shift the active 128-K half of the shared 256-K word into the opsel's low bytes.
            if const_expr(scale_shift is not None):
                sa_v = [v.shrui(scale_shift) for v in sa_v]
                sb_v = [v.shrui(scale_shift) for v in sb_v]
            if const_expr(BN < 128):
                _bnsh = ((by_n + wave * (BN // num_waves)) % 32) // 16 * 8
                sb_v = [v.shrui(_bnsh) for v in sb_v]
            # kh OUTERMOST: consecutive MFMAs hit distinct accumulators (dense issue). Each
            # scaled MFMA = fx.gemm over rank-1 i32[4] A/B frags, e8m0 word on scale_a=/scale_b=.
            c_frags = [fx.make_rmem_tensor(4, Float32) for _ in range_constexpr(n_acc)]
            for idx in range_constexpr(n_acc):
                c_frags[idx].store(Vec(accs[idx]))
            for kh in range_constexpr(k_halves):
                for ni in range_constexpr(num_acc_n):
                    np_i, in_b = ni // 2, ni % 2
                    for mi in range_constexpr(m_chunks):
                        mp_i, im = mi // 2, mi % 2
                        cf = c_frags[mi * num_acc_n + ni]
                        fx.gemm(
                            scale_atoms[(kh * 2 + im, kh * 2 + in_b)],
                            cf,
                            av[mi * k_halves + kh],
                            bv[ni * k_halves + kh],
                            cf,
                            scale_a=sa_v[mp_i],
                            scale_b=sb_v[np_i],
                        )
            for idx in range_constexpr(n_acc):
                accs[idx] = c_frags[idx].load().ir_value()
            return accs

        def hot_loop_scheduler():
            # Interleave the MFMAs with the tile's vmem + A-LDS loads: preload all hints, then issue MFMAs 1-by-1.
            rocdl.sched_vmem(sched_num_gmem)
            rocdl.sched_dsrd(sched_num_ds_load)
            for _ in range_constexpr(sched_mfma_total):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        accs_init = [
            Vec.filled(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)
        ]

        # Double-buffered LDS-A: prefetch tile iv+1 into the other buffer while MFMAs compute
        # tile iv. K-tile indices are ABSOLUTE (kt0-based) so split-K addresses its own K
        # range; the loop trips k_tiles_local times and the ping-pong parity tracks the local
        # iteration (iv), not the absolute tile.
        dma_a_to_lds(kt0, fx.Int32(0))
        rocdl.s_waitcnt(0)
        gpu.barrier()
        for iv, state in range(
            fx.Index(0), fx.Index(k_tiles_local), fx.Index(1), init=accs_init
        ):
            accs = list(state)
            ivi = fx.Int32(iv)
            cur = ivi % 2
            nxt = (ivi + 1) % 2
            kt = kt0 + ivi  # absolute K-tile for A/B/scale addressing
            nkt = ivi + 1
            # clamp last-iter prefetch to the local last tile, then rebase to absolute
            pf_kt = kt0 + (nkt - nkt // k_tiles_local)
            chunk_kt = kt if tiles_per_chunk == 1 else kt // tiles_per_chunk
            scale_shift = None if tiles_per_chunk == 1 else (kt % tiles_per_chunk) * 16
            av = read_a(cur)
            bv = load_b(kt)
            sa_v, sb_v = load_sc(chunk_kt)
            dma_a_to_lds(pf_kt, nxt)  # A DMA after B/scale loads -> overlaps the MFMAs
            accs = compute(accs, av, bv, sa_v, sb_v, scale_shift)
            hot_loop_scheduler()
            rocdl.s_waitcnt(0)  # drain the A DMA before the barrier
            gpu.barrier()
            results = yield accs
        # A single loop-carried accumulator is returned directly rather than as
        # a one-element tuple by the SCF builder.
        accs = [results] if n_acc == 1 else results

        # Epilogue via fx.copy: a lane owns 4 rows per (mi,ni) accm (row
        # m*16+(l//16)*4+ii, col base+l%16), c_stride apart. The single-row
        # specialization packs adjacent lanes; multi-row tiles use masked scalar
        # stores. The dynamically last block for each output tile reduces all slabs.
        c_stride = N if c_row_stride < 0 else c_row_stride
        # Regular split-K writes fp32 partial slabs. The small-M fused path writes
        # BF16 partials; no-split writes directly to the requested output type.
        if const_expr(splitk_fused):
            store_elem = BFloat16
            _ebytes = 2
            c_addr = arg_c + fx.Int64(bid_z) * fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(
                _ebytes
            )
        elif const_expr(k_batch > 1):
            store_elem = Float32
            _ebytes = 4
            # arg_c is tmp[batch*k_batch, M, N] fp32; this WG's slab index == bid_z.
            c_addr = arg_c + fx.Int64(bid_z) * fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(
                _ebytes
            )
        else:
            store_elem = out_elem
            _ebytes = 2
            c_addr = arg_c
            if const_expr(batch > 1):
                c_bstride = (
                    fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
                    if c_batch_stride < 0
                    else fx.Int64(c_batch_stride)
                )
                c_addr = c_addr + fx.Int64(bz_batch) * c_bstride
        # >4GB output: buffer voffset is i32 and num_records is a 32-bit field, so a
        # [M,N] output exceeding 4GB overflows both. Fold THIS WG's row base (bx_m
        # rows) into the C buffer resource's i64 BASE, keep every store offset
        # relative to the WG row-tile (row_local*c_stride+col, always i32), and bound
        # num_records to this WG's own rows (also masks ragged-M OOB rows).
        c_tile_addr = c_addr + fx.Int64(bx_m) * fx.Int64(c_stride) * fx.Int64(_ebytes)
        _rows_rem = fx.Index(i32_m) - fx.Index(bx_m)
        _rows_wg = (_rows_rem < fx.Index(BM)).select(_rows_rem, fx.Index(BM))
        c_nrec = fx.Int64(_rows_wg) * fx.Int64(c_stride) * fx.Int64(_ebytes)
        c_ptr_ty = fx.PointerType.get(
            store_elem.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=4 if small_m_bf16 else _ebytes,
        )
        c_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(c_ptr_ty, c_tile_addr), fx.make_layout(1 << 28, 1)
                    )
                ),
                max_size=False,
                num_records_bytes=c_nrec,
            ),
            fx.make_layout(1, 1),
        )
        if const_expr(
            (k_batch > 1 and not splitk_fused)
            or (small_m_bf16 and not multi_row_tile)
        ):
            c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), store_elem)
        else:
            c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), store_elem)
        c_rstride = fx.Int32(c_stride)
        col_w = by_n + wave * (BN // num_waves) + lane_mod_16
        if const_expr(small_m_bf16 and not multi_row_tile):
            # Avoid issuing the other 15 rows' masked stores (and their lane
            # exchanges) for the latency-critical single-row specialization.
            for ni in range_constexpr(num_acc_n):
                col = col_w + ni * 16
                acc_f32 = Vec(accs[ni])
                if lane_div_16 == fx.Int32(0):
                    peer_bits = rocdl.ds_bpermute(
                        T.i32,
                        (lane ^ fx.Int32(1)) * fx.Int32(4),
                        acc_f32[0].bitcast(fx.Int32),
                    )
                    peer = fx.Int32(peer_bits).bitcast(Float32)
                    if lane_mod_16 % fx.Int32(2) == fx.Int32(0):
                        cf = fx.make_rmem_tensor(2, store_elem)
                        cf.store(
                            Vec.from_elements([acc_f32[0], peer], Float32).to(store_elem)
                        )
                        fx.copy(c_copy, cf, c_flat[None, col])
        else:
            for mi in range_constexpr(m_chunks):
                row_local = (
                    mi * 16 + lane_div_16 * 4
                )  # relative to this WG (base folded into c_tile_addr)
                for ni in range_constexpr(num_acc_n):
                    col = col_w + ni * 16
                    acc_f32 = Vec(accs[mi * num_acc_n + ni])
                    acc = acc_f32.to(store_elem)
                    for ii in range_constexpr(4):
                        off = (row_local + ii) * c_rstride + col
                        cf = fx.make_rmem_tensor(1, store_elem)
                        cf.store(Vec.from_elements([acc[ii]], store_elem))
                        fx.copy(c_copy, cf, c_flat[None, off])

        if const_expr(splitk_fused):
            # Publish this block's complete BF16 partial before joining the
            # per-output-tile arrival counter. The dynamically last block owns
            # reduction, so no block spins and oversized grids cannot deadlock.
            # Every lane drains its workspace stores before the block barrier;
            # the last block then reads partials non-temporally from the shared
            # agent L2.  A release fence would unnecessarily write back the
            # whole L2, while an acquire fence would invalidate unrelated cache.
            rocdl.s_waitcnt(0)
            gpu.barrier()
            arrival = fx.Int32(-1)
            if tid == fx.Int32(0):
                tile_idx = (bx_m // BM) * (i32_n // BN) + by_n // BN
                arrival = fx.Int32(
                    atomic_add_agent(
                        arg_semaphore + fx.Int64(tile_idx) * fx.Int64(4), fx.Int32(1)
                    )
                )
            if const_expr(multi_row_tile):
                # Reuse A LDS after the GEMM to broadcast the winning arrival
                # to every wave.  All waves then share the M*BN reduction; a
                # single wave becomes the bottleneck by M=8/16.
                if tid == fx.Int32(0):
                    fx.ptr_store(arrival, sA0_i32)
                gpu.barrier()
                arrival = fx.Int32(fx.ptr_load(sA0_i32))
                owns_reduce = arrival == fx.Int32(k_batch - 1)
            else:
                # M=1 has only BN/2 packed outputs, which fit in wave 0. Avoid
                # the LDS round trip and extra block barrier on that fast path.
                arrival = fx.Int32(rocdl.readfirstlane(T.i32, arrival))
                owns_reduce = (wave == fx.Int32(0)) & (
                    arrival == fx.Int32(k_batch - 1)
                )
            if owns_reduce:
                bf16_ptr_ty = fx.PointerType.get(
                    BFloat16.ir_type,
                    address_space=fx.AddressSpace.Global,
                    alignment=4,
                )
                out_flat = fx.logical_divide(
                    fx.rocdl.make_buffer_tensor(
                        fx.Tensor(
                            fx.make_view(
                                fx.inttoptr(bf16_ptr_ty, arg_out),
                                fx.make_layout(1 << 28, 1),
                            )
                        ),
                        max_size=False,
                        num_records_bytes=fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2),
                    ),
                    fx.make_layout(1, 1),
                )
                # Keep the fused epilogue's live range small: unlike the
                # standalone gfx1250 reducer, these registers coexist with the
                # GEMM body and can otherwise lower GEMM occupancy.
                reduce_vec = 2
                vec_bf16 = T.vec(reduce_vec, BFloat16.ir_type)
                reduce_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), BFloat16)
                slab_stride = fx.Int32(i32_m * N)

                # Follow the existing AITER split-K reduce's numeric path:
                # extend BF16 partials to FP32, accumulate there, then truncate
                # only the final result. Sequential loads minimize live VGPRs.
                if const_expr(multi_row_tile):
                    pairs_per_row = BN // reduce_vec
                    reduce_pairs = i32_m * fx.Int32(pairs_per_row)
                    for pair_idx in range(
                        tid, reduce_pairs, fx.Int32(num_threads)
                    ):
                        row = pair_idx // fx.Int32(pairs_per_row)
                        pair_in_row = pair_idx % fx.Int32(pairs_per_row)
                        out_off = (
                            row * fx.Int32(N)
                            + by_n
                            + pair_in_row * fx.Int32(reduce_vec)
                        )
                        reduced = fx.Vector.filled(reduce_vec, 0.0, Float32)
                        for sk in range_constexpr(k_batch):
                            partial_dw = fx.Int32(
                                load_i32_nt(
                                    arg_c,
                                    (
                                        out_off + fx.Int32(sk) * slab_stride
                                    )
                                    // fx.Int32(2),
                                )
                            )
                            partial = Vec.from_elements([partial_dw], Int32).bitcast(
                                BFloat16
                            )
                            reduced = reduced + partial.to(Float32)
                        result = fx.make_rmem_tensor(reduce_vec, BFloat16)
                        fx.memref_store_vec(reduced.truncf(vec_bf16), result)
                        fx.copy(reduce_copy, result, out_flat[None, out_off])
                elif tid < fx.Int32(BN // reduce_vec):
                    out_off = by_n + tid * fx.Int32(reduce_vec)
                    reduced = fx.Vector.filled(reduce_vec, 0.0, Float32)
                    for sk in range_constexpr(k_batch):
                        partial_dw = fx.Int32(
                            load_i32_nt(
                                arg_c,
                                (
                                    out_off + fx.Int32(sk) * slab_stride
                                )
                                // fx.Int32(2),
                            )
                        )
                        partial = Vec.from_elements([partial_dw], Int32).bitcast(
                            BFloat16
                        )
                        reduced = reduced + partial.to(Float32)
                    result = fx.make_rmem_tensor(reduce_vec, BFloat16)
                    fx.memref_store_vec(reduced.truncf(vec_bf16), result)
                    fx.copy(reduce_copy, result, out_flat[None, out_off])

                if tid == fx.Int32(0):
                    tile_idx = (bx_m // BM) * (i32_n // BN) + by_n // BN
                    sem_ptr_ty = fx.PointerType.get(
                        Int32.ir_type,
                        address_space=fx.AddressSpace.Global,
                        alignment=4,
                    )
                    sem_ptr = fx.inttoptr(
                        sem_ptr_ty,
                        arg_semaphore + fx.Int64(tile_idx) * fx.Int64(4),
                    )
                    fx.ptr_store(fx.Int32(0), sem_ptr)

    # Keep the established kernel ABI for every non-fused configuration.  The
    # fused small-M specialization needs two extra pointers, but carrying those
    # unused kernargs on all prefill/decode kernels can perturb kernarg preload
    # and regress a few latency-sensitive existing configurations.
    if const_expr(splitk_fused):

        @flyc.kernel(name=_kname)
        def kernel_gemm(
            arg_c: fx.Int64,
            arg_a: fx.Int64,
            arg_b: fx.Int64,
            arg_scale_a: fx.Int64,
            arg_scale_b: fx.Int64,
            arg_out: fx.Int64,
            arg_semaphore: fx.Int64,
            i32_m: fx.Int32,
            i32_n: fx.Int32,
        ):
            _kernel_body(
                arg_c,
                arg_a,
                arg_b,
                arg_scale_a,
                arg_scale_b,
                arg_out,
                arg_semaphore,
                i32_m,
                i32_n,
            )

    else:

        @flyc.kernel(name=_kname)
        def kernel_gemm(
            arg_c: fx.Int64,
            arg_a: fx.Int64,
            arg_b: fx.Int64,
            arg_scale_a: fx.Int64,
            arg_scale_b: fx.Int64,
            i32_m: fx.Int32,
            i32_n: fx.Int32,
        ):
            _kernel_body(
                arg_c,
                arg_a,
                arg_b,
                arg_scale_a,
                arg_scale_b,
                arg_c,
                arg_c,
                i32_m,
                i32_n,
            )

    c_addr = fx.Int64(fx.ptrtoint(arg_c))
    a_addr = fx.Int64(fx.ptrtoint(arg_a))
    b_addr = fx.Int64(fx.ptrtoint(arg_b))
    sa_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
    sb_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
    out_addr = fx.Int64(fx.ptrtoint(arg_out))
    semaphore_addr = fx.Int64(fx.ptrtoint(arg_semaphore))
    if const_expr(waves_per_eu > 0):
        wpe = waves_per_eu
    else:
        wpe = None
    gx = (i32_m + (BM - 1)) // BM
    gy = i32_n // BN
    gz = batch * k_batch  # split-K: k_batch splits per (real) batch on grid.z
    if const_expr(splitk_fused):
        kernel_gemm(
            c_addr,
            a_addr,
            b_addr,
            sa_addr,
            sb_addr,
            out_addr,
            semaphore_addr,
            i32_m,
            i32_n,
            value_attrs={"rocdl.waves_per_eu": wpe},
        ).launch(grid=(gx, gy, gz), block=(num_threads, 1, 1), stream=stream)
    else:
        kernel_gemm(
            c_addr,
            a_addr,
            b_addr,
            sa_addr,
            sb_addr,
            i32_m,
            i32_n,
            value_attrs={"rocdl.waves_per_eu": wpe},
        ).launch(grid=(gx, gy, gz), block=(num_threads, 1, 1), stream=stream)


@flyc.jit
def launch_gemm(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    a_dtype: Constexpr[str],
    out_dtype: Constexpr[str],
    b_dtype: Constexpr[str],
    batch: Constexpr[int],
    a_row_stride: Constexpr[int],
    a_batch_stride: Constexpr[int],
    sca_row_stride: Constexpr[int],
    sca_batch_stride: Constexpr[int],
    c_row_stride: Constexpr[int],
    c_batch_stride: Constexpr[int],
    waves_per_eu: Constexpr[int],
    xcd_swizzle: Constexpr[int],
    k_batch: Constexpr[int] = 1,
    blockscale: Constexpr[str] = "none",
    multi_row_tile: Constexpr[bool] = False,
):
    """Launch the established non-fused GEMM ABI used by existing configs."""
    _launch_gemm_impl(
        arg_c,
        arg_a,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        arg_c,
        arg_c,
        i32_m,
        i32_n,
        stream,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        a_dtype,
        out_dtype,
        b_dtype,
        batch,
        a_row_stride,
        a_batch_stride,
        sca_row_stride,
        sca_batch_stride,
        c_row_stride,
        c_batch_stride,
        waves_per_eu,
        xcd_swizzle,
        k_batch,
        blockscale,
        multi_row_tile,
    )


@flyc.jit
def launch_gemm_fused(
    arg_c: fx.Pointer,
    arg_a: fx.Pointer,
    arg_b: fx.Pointer,
    arg_scale_a: fx.Pointer,
    arg_scale_b: fx.Pointer,
    arg_out: fx.Pointer,
    arg_semaphore: fx.Pointer,
    i32_m: fx.Int32,
    i32_n: fx.Int32,
    stream: fx.Stream,
    N: Constexpr[int],
    K: Constexpr[int],
    tile_m: Constexpr[int],
    tile_n: Constexpr[int],
    tile_k: Constexpr[int],
    a_dtype: Constexpr[str],
    out_dtype: Constexpr[str],
    b_dtype: Constexpr[str],
    batch: Constexpr[int],
    a_row_stride: Constexpr[int],
    a_batch_stride: Constexpr[int],
    sca_row_stride: Constexpr[int],
    sca_batch_stride: Constexpr[int],
    c_row_stride: Constexpr[int],
    c_batch_stride: Constexpr[int],
    waves_per_eu: Constexpr[int],
    xcd_swizzle: Constexpr[int],
    k_batch: Constexpr[int] = 1,
    blockscale: Constexpr[str] = "none",
    multi_row_tile: Constexpr[bool] = False,
):
    """Launch the M<=16 split-K GEMM with its fused reduction arguments."""
    _launch_gemm_impl(
        arg_c,
        arg_a,
        arg_b,
        arg_scale_a,
        arg_scale_b,
        arg_out,
        arg_semaphore,
        i32_m,
        i32_n,
        stream,
        N,
        K,
        tile_m,
        tile_n,
        tile_k,
        a_dtype,
        out_dtype,
        b_dtype,
        batch,
        a_row_stride,
        a_batch_stride,
        sca_row_stride,
        sca_batch_stride,
        c_row_stride,
        c_batch_stride,
        waves_per_eu,
        xcd_swizzle,
        k_batch,
        blockscale,
        multi_row_tile,
    )


# ── split-K reduce ────────────────────────────────────────────────────────────
# The split-K path above writes fp32 partial slabs tmp[k_batch, M, N]; this kernel
# sums the k_batch slabs elementwise in fp32 and casts to bf16/fp16 out in a single
# fused pass (load slabs -> fp32 add -> cast -> store).

_REDUCE_BLOCK = 256
_REDUCE_VEC = 2  # out elems per thread == one dword (2 fp32 in per slab)


@flyc.jit
def launch_splitk_reduce(
    arg_tmp: fx.Pointer,
    arg_out: fx.Pointer,
    n_out_dw: fx.Int32,  # output dwords = M*N // 2 (2 out elems per dword)
    slab_stride_dw: fx.Int32,  # dwords per split slab = M*N (fp32: 1 dword/elem)
    stream: fx.Stream,
    split_k: Constexpr[int],
    out_dtype: Constexpr[str],
):
    """Sum ``split_k`` fp32 slabs of ``arg_tmp`` -> bf16/fp16 ``arg_out``.

    arg_tmp: (split_k, M, N) fp32 contiguous. arg_out: (M, N) out_dtype. One output
    dword (= 2 out elems = 2 fp32 inputs per slab) per thread; grid.x covers all.
    """
    if const_expr(out_dtype == "bf16"):
        out_elem = BFloat16
    else:
        out_elem = Float16

    # Same reason as the GEMM kernel: keep the split-K reduce distinguishable per
    # config in a profile instead of a bare "reduce_kernel_<id>".
    _kname = f"flydsl_mxpsh_skreduce_sk{split_k}_{_DTYPE_SHORT[out_dtype]}"

    @flyc.kernel(name=_kname)
    def reduce_kernel(
        tmp: fx.Pointer,
        out: fx.Pointer,
        n_out_dw_i: fx.Int32,
        slab_dw_i: fx.Int32,
    ):
        frag_layout = fx.make_layout(_REDUCE_VEC, 1)
        vt = fx.Int32(fx.block_idx.x) * fx.Int32(_REDUCE_BLOCK) + fx.Int32(
            fx.thread_idx.x
        )
        slab_frags = slab_dw_i // fx.Int32(_REDUCE_VEC)

        def _tiled(ptr, elem, num_records_bytes):
            typed = fx.PointerType.get(
                elem.ir_type,
                address_space=fx.AddressSpace.Global,
                alignment=_REDUCE_VEC * elem.width // 8,
            )
            buf = fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(typed, fx.Int64(fx.ptrtoint(ptr))),
                        fx.make_layout(1 << 30, 1),
                    )
                ),
                max_size=False,
                num_records_bytes=num_records_bytes,
            )
            return fx.logical_divide(buf, frag_layout)

        in_t = _tiled(
            tmp, Float32, fx.Int64(slab_dw_i) * fx.Int64(split_k) * fx.Int64(4)
        )
        out_t = _tiled(out, out_elem, fx.Int64(n_out_dw_i) * fx.Int64(4))
        in_copy = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), Float32)  # 2 x f32
        out_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), out_elem)  # 2 x 16b

        if vt < n_out_dw_i:
            acc = Vec.filled(_REDUCE_VEC, 0.0, Float32)
            for sk in range_constexpr(split_k):
                frag = fx.make_rmem_tensor(frag_layout, Float32)
                fx.copy_atom_call(
                    in_copy,
                    fx.slice(in_t, (None, vt + fx.Int32(sk) * slab_frags)),
                    frag,
                )
                acc = acc + frag.load()
            out_frag = fx.make_rmem_tensor(frag_layout, out_elem)
            out_frag.store(acc.to(out_elem))
            fx.copy_atom_call(out_copy, out_frag, fx.slice(out_t, (None, vt)))

    gx = (n_out_dw + (_REDUCE_BLOCK - 1)) // _REDUCE_BLOCK
    reduce_kernel(arg_tmp, arg_out, n_out_dw, slab_stride_dw).launch(
        grid=(gx, 1, 1), block=(_REDUCE_BLOCK, 1, 1), stream=stream
    )

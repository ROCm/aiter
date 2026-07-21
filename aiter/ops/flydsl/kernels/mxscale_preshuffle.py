# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4/MXFP6/MXFP8 A x MXFP4/MXFP8 B preshuffle GEMM (gfx950): per-32 E8M0 scales folded
into a scaled 16x16x128 fx.gemm; A streams global->LDS via double-buffered async DMA. Layout
matches the host preshuffle (shuffle_weight_w4(.,16) + shuffle_scale_w4).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly, scf
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
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

from aiter.ops.flydsl.kernels.tensor_shim import ptr_rsrc

_A_ELEM = {"fp4": Float4E2M1FN, "fp6": Float6E2M3FN, "fp8": Float8E4M3FN}
_B_ELEM = {"fp4": Float4E2M1FN, "fp8": Float8E4M3FN}


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
):
    """Direct @flyc.jit launcher. Operands are fx.Pointer (pass ptr_arg(t): raw data_ptr, no
    per-launch DLPack). Compile once with flyc.compile, then cf(*runtime). a_dtype fp4/fp6/fp8
    A x preshuffled b_dtype (fp4/fp8) B, e8m0 scales (a8w8 = a_dtype=fp8, b_dtype=fp8).
    batch>1 = strided-batched over grid.z. The a_/sca_/c_ row/batch strides make A/scale_a/C
    addressing caller-controlled; each <0 keeps the contiguous [B,M,*] bmn default, all set =
    the [M,B,*] mbn layout. waves_per_eu<=0 = unset.
    """
    BM, BN, BK = tile_m, tile_n, tile_k
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
    # K-tiles and writes an fp32 partial slab to arg_c (viewed as [batch*k_batch, M, N]);
    # a follow-up reduce kernel sums the k_batch slabs -> bf16 out. K/k_batch is 256-K
    # aligned (see fits_shape), so a split boundary lands on whole tiles + scale chunks.
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
    n_coop = A_LDS_B // num_threads // 16  # 16B cooperative loads per thread
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

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Int64,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
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
            from .mfma_preshuffle_pipeline import xcd_remap_bx_by

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
                _scale_chunk_dw if sca_row_stride < 0 else sca_row_stride
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
                    * fx.Int64(_scale_chunk_dw)
                    * fx.Int64(4)
                )
                arg_scale_a = arg_scale_a + bz * sc_bstride
            else:
                arg_scale_a = arg_scale_a + bz * fx.Int64(sca_batch_stride)
            arg_scale_b = arg_scale_b + bz * fx.Int64((N // 32) * _scale_chunk_dw * 4)
        else:
            a_rstride = fx.Int32(a_row_bytes)
            sca_rstride = fx.Int32(_scale_chunk_dw)

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
        a_flat_div = fx.logical_divide(a_flat, fx.make_layout(1, 1))
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

        def _lds_view(base_iter, off_i32):
            return fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(4, 1))

        # Async A: gmem->LDS DMA (buffer_load_lds); issued after B/scale loads to overlap the MFMAs.
        def dma_a_to_lds(kt, parity):
            base_off = rocdl.readfirstlane(T.i32, parity * lds_db + wave * (64 * 16))
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
            # ds_read_b128 straight into an i32[4] register fragment.
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
                        # fp6/fp8: pack two halves (64 K apart, f8f6f4 ABI) into i32[A_NDW].
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
                + fx.Int64(_scale_chunk_dw)
            ) * fx.Int64(4)
        else:
            a_sc_nrec = fx.Int64(_a_sc_chunks) * fx.Int64(_scale_chunk_dw) * fx.Int64(4)
        b_sc_nrec = fx.Int64((N // 32) * _scale_chunk_dw * 4)
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
        b_sc_base = [(nsb + np) * _scale_chunk_dw for np in range_constexpr(n_pairs)]
        sc_lane = lane_div_16 * 16 + lane_mod_16

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
            koff = chunk_kt * _scale_k0_dw
            sa = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sa_flat[
                            None,
                            rocdl.readfirstlane(T.i32, a_sc_base[mp] + koff) + sc_lane,
                        ],
                    )
                )[0]
                for mp in range_constexpr(m_pairs)
            ]
            sb = [
                Vec(
                    fly.copy_atom_call_ssa(
                        [T.vec(1, T.i32)],
                        bs_copy,
                        sb_flat[
                            None,
                            rocdl.readfirstlane(T.i32, b_sc_base[np] + koff) + sc_lane,
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
        accs = results

        # Epilogue via fx.copy: a lane owns 4 rows per (mi,ni) accm (row m*16+(l//16)*4+ii, col
        # base+l%16), c_stride apart.
        c_stride = N if c_row_stride < 0 else c_row_stride
        # split-K writes an fp32 partial slab (no cast); no-split writes bf16/fp16 out.
        if const_expr(k_batch > 1):
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
            store_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=_ebytes
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
        if const_expr(k_batch > 1):
            c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), store_elem)
        else:
            c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), store_elem)
        c_rstride = fx.Int32(c_stride)
        col_w = by_n + wave * (BN // num_waves) + lane_mod_16
        for mi in range_constexpr(m_chunks):
            row_local = (
                mi * 16 + lane_div_16 * 4
            )  # relative to this WG (base folded into c_tile_addr)
            for ni in range_constexpr(num_acc_n):
                col = col_w + ni * 16
                acc = Vec(accs[mi * num_acc_n + ni]).to(store_elem)
                for ii in range_constexpr(4):
                    cf = fx.make_rmem_tensor(1, store_elem)
                    cf.store(Vec.from_elements([acc[ii]], store_elem))
                    off = (row_local + ii) * c_rstride + col
                    fx.copy(c_copy, cf, c_flat[None, off])

    c_addr = fx.Int64(fx.ptrtoint(arg_c))
    a_addr = fx.Int64(fx.ptrtoint(arg_a))
    b_addr = fx.Int64(fx.ptrtoint(arg_b))
    sa_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
    sb_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
    if const_expr(waves_per_eu > 0):
        wpe = waves_per_eu
    else:
        wpe = None
    gx = (i32_m + (BM - 1)) // BM
    gy = i32_n // BN
    gz = batch * k_batch  # split-K: k_batch splits per (real) batch on grid.z
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


# ── split-K reduce ────────────────────────────────────────────────────────────
# The split-K path above writes fp32 partial slabs tmp[k_batch, M, N]; this kernel
# sums the k_batch slabs elementwise in fp32 and casts to bf16/fp16 out in a single
# fused pass (load slabs -> fp32 add -> cast -> store).

_REDUCE_BLOCK = 256


def _pack_pair_from_f32(acc_lo, acc_hi, out_dtype, *, i32):
    """Truncate two f32 accumulators to bf16/f16 and pack into one dword."""
    odt = T.bf16 if out_dtype == "bf16" else T.f16
    lo_i16 = arith.bitcast(T.i16, arith.trunc_f(odt, acc_lo))
    hi_i16 = arith.bitcast(T.i16, arith.trunc_f(odt, acc_hi))
    lo_i32 = arith.extui(i32, lo_i16)
    hi_i32 = arith.extui(i32, hi_i16)
    return lo_i32 | (hi_i32 << arith.constant(16, type=i32))


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

    @flyc.kernel
    def reduce_kernel(
        tmp: fx.Pointer,
        out: fx.Pointer,
        n_out_dw_i: fx.Int32,
        slab_dw_i: fx.Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        bid = fx.block_idx.x
        tid = fx.thread_idx.x
        in_rsrc = ptr_rsrc(tmp)
        out_rsrc = ptr_rsrc(out)
        n_out_dw_v = ArithValue(n_out_dw_i)
        slab_dw_v = ArithValue(slab_dw_i)
        dw = ArithValue(bid) * arith.constant(_REDUCE_BLOCK, type=i32) + ArithValue(tid)
        dw_valid = arith.cmpi(CmpIPredicate.ult, dw, n_out_dw_v)
        _if = scf.IfOp(dw_valid)
        with ir.InsertionPoint(_if.then_block):
            e0 = dw * arith.constant(2, type=i32)  # first input element (fp32) index
            acc_lo = ArithValue(arith.constant(0.0, type=f32))
            acc_hi = ArithValue(arith.constant(0.0, type=f32))
            for sk in range_constexpr(split_k):
                sk_off = arith.constant(sk, type=i32) * slab_dw_v
                raw = buffer_ops.buffer_load(
                    in_rsrc, e0 + sk_off, vec_width=2, dtype=f32
                )
                lo = ArithValue(
                    vector.extract(raw, static_position=[0], dynamic_position=[])
                )
                hi = ArithValue(
                    vector.extract(raw, static_position=[1], dynamic_position=[])
                )
                acc_lo = acc_lo + lo
                acc_hi = acc_hi + hi
            packed = _pack_pair_from_f32(acc_lo, acc_hi, out_dtype, i32=i32)
            buffer_ops.buffer_store(packed, out_rsrc, dw)
            scf.YieldOp([])

    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        pass

    gx = (n_out_dw + (_REDUCE_BLOCK - 1)) // _REDUCE_BLOCK
    reduce_kernel(arg_tmp, arg_out, n_out_dw, slab_stride_dw).launch(
        grid=(gx, 1, 1), block=(_REDUCE_BLOCK, 1, 1), stream=stream
    )

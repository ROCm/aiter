# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4/MXFP6/MXFP8 A x MXFP4 B preshuffle GEMM (gfx950): per-32 E8M0 scales folded into
a scaled 16x16x128 fx.gemm; A streams global->LDS via double-buffered async DMA. Layout
matches CK ``shuffle_weight_w4(.,16)`` + ``shuffle_scale_w4``."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
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

_raw = arith._to_raw  # ArithValue/Numeric -> raw ir.Value

_A_ELEM = {"fp4": Float4E2M1FN, "fp6": Float6E2M3FN, "fp8": Float8E4M3FN}


def _scale_mma_atoms(a_dtype):
    """16 (opsel_a, opsel_b) scaled-MFMA atoms; A elem is fp4/fp6/fp8, B always fp4."""
    elem_a = _A_ELEM[a_dtype]
    return {
        (osa, osb): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16, 16, 128, elem_a, Float4E2M1FN, opsel_a=osa, opsel_b=osb
            )
        )
        for osa in range(4)
        for osb in range(4)
    }


def _bq_view(arg_bq_addr, row_elems, KH4, k_tiles, k_halves):
    """CK-preshuffled B view for one N-row tile; index [l//16, l%16, kt, half, None] -> i32[4]."""
    col_base = rocdl.readfirstlane(T.i32, _raw(row_elems) * fx.Int32(KH4))
    i32_ptr_ty = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=16
    )
    off_i64 = fx.Int64(arith.ExtUIOp(T.i64, _raw(col_base)).result)
    base_iter = fx.inttoptr(i32_ptr_ty, fx.Int64(arg_bq_addr) + off_i64 * fx.Int64(4))
    shape = (4, 16, k_tiles, k_halves, 4)
    view = fx.Tensor(
        fx.make_view(base_iter, fx.make_layout(shape, (64, 4, k_halves * 256, 256, 1)))
    )
    return fx.rocdl.make_buffer_tensor(view, max_size=False)


@flyc.jit
def launch_gemm(
    arg_c: fx.Tensor,
    arg_a: fx.Tensor,
    arg_b: fx.Tensor,
    arg_scale_a: fx.Tensor,
    arg_scale_b: fx.Tensor,
    arg_bias: fx.Tensor,
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
    batch: Constexpr[int],
    a_row_stride: Constexpr[int],
    a_batch_stride: Constexpr[int],
    sca_row_stride: Constexpr[int],
    sca_batch_stride: Constexpr[int],
    c_row_stride: Constexpr[int],
    c_batch_stride: Constexpr[int],
    waves_per_eu: Constexpr[int],
):
    """Direct @flyc.jit launcher (flyc caches per Constexpr config; call it directly, no
    builder). a_dtype fp4/fp6/fp8 A x CK-preshuffled MXFP4 B, e8m0 scales, bias unused.
    batch>1 = strided-batched over grid.z. The a_/sca_/c_ row/batch strides make A/scale_a/C
    addressing caller-controlled; each <0 keeps the contiguous [B,M,*] bmn default, all set =
    the [M,B,*] mbn (deepseek grouped) layout. waves_per_eu<=0 = unset."""
    from flydsl.compiler.kernel_function import CompilationContext

    CompilationContext.get_current()

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

    A_LDS_B = BM * A_ROW_B  # LDS A buffer bytes (row-major [m][col], shared by 4 N-waves)
    A_ROW_I32 = A_ROW_B // 4
    K_HALF = K // 2
    KH4 = K_HALF // 4
    K_TILES = K // BK
    k_halves = BK // 128  # 16x16x128 MFMA k-steps per K-tile
    # e8m0 scales are 256-K granular, B 128-K: tiles_per_chunk K-tiles share a word (hi/lo 16b = 128-K half).
    tiles_per_chunk = 256 // BK  # 1 for tile_k=256, 2 for tile_k=128
    m_chunks = BM // 16
    num_acc_n = (BN // 4) // 16  # 16-col n-subblocks per wave
    _scale_chunk_dw = (K // 32 // 4 // 2) * 64  # e8m0 stride (dwords), per shuffle_scale_w4
    _scale_k0_dw = 64
    n_coop = A_LDS_B // 256 // 16  # 16B cooperative loads per thread
    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)

    # Scheduler counts per loop iter: MFMAs, A LDS reads/thread (fp6/fp8 2 per (mi,kh)), gmem loads.
    sched_mfma_total = k_halves * m_chunks * num_acc_n
    if const_expr(a_dtype == "fp4"):
        a_ds_per = 1
    else:
        a_ds_per = 2
    sched_num_ds_load = m_chunks * k_halves * a_ds_per
    sched_num_gmem = n_coop + num_acc_n * k_halves + m_pairs + n_pairs

    @fx.struct
    class SharedA:
        a0: fx.Array[Int8, A_LDS_B, 16]
        a1: fx.Array[Int8, A_LDS_B, 16]

    @flyc.kernel
    def kernel_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_scale_a: fx.Int64,
        arg_scale_b: fx.Int64,
        arg_bias: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        scale_atoms = _scale_mma_atoms(a_dtype)

        tid = fx.thread_idx.x
        bid_x, bid_y, bid_z = fx.block_idx
        wave = rocdl.readfirstlane(T.i32, fx.Int32(tid) // fx.Int32(64))
        lane = fx.Int32(tid) % fx.Int32(64)
        lane_div_16 = lane // fx.Int32(16)
        lane_mod_16 = lane % fx.Int32(16)
        bx_m = bid_x * fx.Int32(BM)
        by_n = bid_y * fx.Int32(BN)

        # Strided-batched: shift each base to batch bid_z (A/scale_a via explicit strides or
        # the contiguous default; B/scale_b stay batch-contiguous). batch==1 emits no batch math.
        if const_expr(batch > 1):
            a_rstride = fx.Int32(a_row_bytes if a_row_stride < 0 else a_row_stride)
            sca_rstride = fx.Int32(
                _scale_chunk_dw if sca_row_stride < 0 else sca_row_stride
            )
            bz = fx.Int64(bid_z)
            if const_expr(a_batch_stride < 0):
                arg_a = fx.Int64(arg_a) + bz * (fx.Int64(i32_m) * fx.Int64(a_row_bytes))
            else:
                arg_a = fx.Int64(arg_a) + bz * fx.Int64(a_batch_stride)
            arg_b = fx.Int64(arg_b) + bz * fx.Int64(N * (K // 2))
            if const_expr(sca_batch_stride < 0):
                sc_bstride = (
                    fx.Int64((i32_m + fx.Int32(31)) // fx.Int32(32))
                    * fx.Int64(_scale_chunk_dw)
                    * fx.Int64(4)
                )
                arg_scale_a = fx.Int64(arg_scale_a) + bz * sc_bstride
            else:
                arg_scale_a = fx.Int64(arg_scale_a) + bz * fx.Int64(sca_batch_stride)
            arg_scale_b = fx.Int64(arg_scale_b) + bz * fx.Int64(
                (N // 32) * _scale_chunk_dw * 4
            )
        else:
            a_rstride = fx.Int32(a_row_bytes)
            sca_rstride = fx.Int32(_scale_chunk_dw)

        # A source view, bound to the last valid M row (ragged M OOB -> 0).
        _i8g = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=16)
        if const_expr(batch > 1 and a_row_stride >= 0):
            a_nrec = fx.Int64(i32_m - fx.Int32(1)) * fx.Int64(a_rstride) + fx.Int64(
                a_row_bytes
            )
        else:
            a_nrec = fx.Int64(i32_m) * fx.Int64(a_row_bytes)
        a_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(
                    fx.inttoptr(_i8g, fx.Int64(arg_a)),
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
        lds_db_i32 = lds_db // fx.Int32(4)
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
            base_off = rocdl.readfirstlane(
                T.i32, parity * lds_db + wave * fx.Int32(64 * 16)
            )
            lds_ptr = fx.add_offset(sA0_i8, base_off)
            base_k_byte = kt * fx.Int32(A_ROW_B)
            for i in range_constexpr(n_coop):
                if const_expr(i > 0):
                    lds_ptr = fx.add_offset(lds_ptr, fx.Int32(256 * 16))
                lin = (fx.Int32(i * 256) + fx.Int32(tid)) * fx.Int32(16)
                row = lin // fx.Int32(A_ROW_B)
                col = lin % fx.Int32(A_ROW_B)
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
                    off = (
                        (fx.Int32(mi * 16) + lane_mod_16) * fx.Int32(A_ROW_I32)
                        + fx.Int32(kh * A_KH_I32)
                        + lane_div_16 * fx.Int32(A_GK_I32)
                    )
                    if const_expr(a_dtype == "fp4"):
                        av.append(_read16(base_iter, off))
                    else:
                        # fp6/fp8: pack two halves (64 K apart, f8f6f4 ABI) into i32[A_NDW].
                        lo = Vec(fx.memref_load_vec(_read16(base_iter, off)))
                        hi = Vec(
                            fx.memref_load_vec(
                                _read16(base_iter, off + fx.Int32(A_HI_OFF))
                            )
                        )
                        t = fx.make_rmem_tensor(A_NDW, Int32)
                        t.store(lo.shuffle(hi, list(range(A_NDW))))
                        av.append(t)
            return av

        n_col_base = by_n + wave * fx.Int32(BN // 4)
        bq_views = [
            _bq_view(arg_b, n_col_base + fx.Int32(ni * 16), KH4, K_TILES, k_halves)
            for ni in range_constexpr(num_acc_n)
        ]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        bs_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        # e8m0 scale buffers bounded to real size (OOB rows read 0); scale_a to the last 32-row chunk.
        _i32g = fx.PointerType.get(T.i32, address_space=fx.AddressSpace.Global, alignment=4)
        _sc_layout = fx.make_layout(1 << 28, 1)
        _a_sc_chunks = (i32_m + fx.Int32(31)) // fx.Int32(32)
        if const_expr(batch > 1 and sca_row_stride >= 0):
            a_sc_nrec = (
                fx.Int64(_a_sc_chunks - fx.Int32(1)) * fx.Int64(sca_rstride)
                + fx.Int64(_scale_chunk_dw)
            ) * fx.Int64(4)
        else:
            a_sc_nrec = fx.Int64(_a_sc_chunks) * fx.Int64(_scale_chunk_dw) * fx.Int64(4)
        b_sc_nrec = fx.Int64((N // 32) * _scale_chunk_dw * 4)
        sa_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(fx.inttoptr(_i32g, fx.Int64(arg_scale_a)), _sc_layout)
                ),
                max_size=False,
                num_records_bytes=a_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        sb_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(fx.inttoptr(_i32g, fx.Int64(arg_scale_b)), _sc_layout)
                ),
                max_size=False,
                num_records_bytes=b_sc_nrec,
            ),
            fx.make_layout(1, 1),
        )
        a_sc_base = [
            (bx_m // fx.Int32(32) + fx.Int32(mp)) * sca_rstride
            for mp in range_constexpr(m_pairs)
        ]
        nsb = by_n // fx.Int32(32) + wave * fx.Int32(BN // 128)
        b_sc_base = [
            (nsb + fx.Int32(np)) * fx.Int32(_scale_chunk_dw)
            for np in range_constexpr(n_pairs)
        ]
        sc_lane = lane_div_16 * fx.Int32(16) + lane_mod_16

        n_acc = m_chunks * num_acc_n

        def load_b(kt):
            # buffer_load_dwordx4 straight into i32[4] register fragments.
            ops = []
            for ni in range_constexpr(num_acc_n):
                for kh in range_constexpr(k_halves):
                    bf = fx.make_rmem_tensor(4, Int32)
                    fx.copy_atom_call(
                        b_copy, bq_views[ni][lane_div_16, lane_mod_16, kt, kh, None], bf
                    )
                    ops.append(bf)
            return ops

        def load_sc(chunk_kt):
            # (sa, sb) e8m0 words per m-/n-pair for one 256-K chunk (uniform base -> SGPR soffset).
            koff = chunk_kt * fx.Int32(_scale_k0_dw)
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
                sh = _raw(scale_shift)
                sa_v = [arith.shrui(_raw(v), sh) for v in sa_v]
                sb_v = [arith.shrui(_raw(v), sh) for v in sb_v]
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

        # Double-buffered LDS-A: prefetch tile iv+1 into the other buffer while MFMAs compute tile iv.
        dma_a_to_lds(fx.Int32(0), fx.Int32(0))
        rocdl.s_waitcnt(0)
        gpu.barrier()
        for iv, state in range(
            fx.Index(0), fx.Index(K_TILES), fx.Index(1), init=accs_init
        ):
            accs = list(state)
            kt = fx.Int32(iv)
            cur = kt % fx.Int32(2)
            nxt = (kt + fx.Int32(1)) % fx.Int32(2)
            nkt = kt + fx.Int32(1)
            pf_kt = nkt - nkt // fx.Int32(K_TILES)  # clamp last-iter prefetch to K_TILES-1
            chunk_kt = kt if tiles_per_chunk == 1 else kt // fx.Int32(tiles_per_chunk)
            scale_shift = (
                None
                if tiles_per_chunk == 1
                else (kt % fx.Int32(tiles_per_chunk)) * fx.Int32(16)
            )
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
        # base+l%16), c_stride apart; c_flat bounds ragged-M OOB and honors c_row/batch_stride.
        c_stride = N if c_row_stride < 0 else c_row_stride
        if const_expr(c_row_stride < 0):
            c_nrec = fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
        else:
            c_nrec = (
                fx.Int64(i32_m - fx.Int32(1)) * fx.Int64(c_stride) + fx.Int64(N)
            ) * fx.Int64(2)
        c_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_c)))
        if const_expr(batch > 1):
            c_bstride = (
                fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
                if c_batch_stride < 0
                else fx.Int64(c_batch_stride)
            )
            c_addr = c_addr + fx.Int64(bid_z) * c_bstride
        c_ptr_ty = fx.PointerType.get(
            out_elem.ir_type, address_space=fx.AddressSpace.Global, alignment=2
        )
        c_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(c_ptr_ty, c_addr), fx.make_layout(1 << 28, 1)
                    )
                ),
                max_size=False,
                num_records_bytes=c_nrec,
            ),
            fx.make_layout(1, 1),
        )
        c_copy = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), out_elem)
        c_rstride = fx.Int32(c_stride)
        col_w = by_n + wave * fx.Int32(BN // 4) + lane_mod_16
        for mi in range_constexpr(m_chunks):
            row_m = bx_m + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
            for ni in range_constexpr(num_acc_n):
                col = col_w + fx.Int32(ni * 16)
                acc = Vec(accs[mi * num_acc_n + ni]).to(out_elem)
                for ii in range_constexpr(4):
                    cf = fx.make_rmem_tensor(1, out_elem)
                    cf.store(Vec.from_elements([acc[ii]], out_elem))
                    off = (row_m + fx.Int32(ii)) * c_rstride + col
                    fx.copy(c_copy, cf, c_flat[None, off])

    a_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_a)))
    b_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_b)))
    sa_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_scale_a)))
    sb_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_scale_b)))
    if const_expr(waves_per_eu > 0):
        wpe = waves_per_eu
    else:
        wpe = None
    arg_c_2d = fx.Tensor(
        fx.make_view(fx.get_iter(arg_c), fx.make_layout((65536, N), (N, 1)))
    )
    gx = (i32_m + (BM - 1)) // BM
    gy = i32_n // BN
    kernel_gemm(
        arg_c_2d,
        a_addr,
        b_addr,
        sa_addr,
        sb_addr,
        arg_bias,
        i32_m,
        i32_n,
        value_attrs={"rocdl.waves_per_eu": wpe},
    ).launch(grid=(gx, gy, batch), block=(256, 1, 1), stream=stream)

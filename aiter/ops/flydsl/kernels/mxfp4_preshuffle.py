# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4 (E2M1) and MXFP6 (E2M3) preshuffle GEMM, per-32 E8M0 scales consumed
inside a scaled 16x16x128 MFMA.  Data layout matches
``tests/kernels/utils/fp4_utils`` (CK weight preshuffle ``shuffle_weight_w4(.,16)``
+ ``shuffle_scale_w4``).

The MMA runs via ``fx.gemm`` over rank-1 register fragments: the per-32 E8M0 word
rides ``scale_a=/scale_b=`` and the ``(opsel_a, opsel_b)`` atom selects the packed
byte.
"""

from typing import Optional

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import fly
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import (
    BFloat16,
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


def _raw(v):
    if not isinstance(v, ir.Value) and hasattr(v, "ir_value"):
        return v.ir_value()
    return v


_A_ELEM = {"fp4": Float4E2M1FN, "fp6": Float6E2M3FN, "fp8": Float8E4M3FN}


def _scale_mma_atoms(a_dtype: str = "fp4"):
    """16 (opsel_a, opsel_b) scaled-MFMA atoms (opsel is a type param).

    a_dtype='fp4': fp4×fp4 (Float4E2M1FN for both A and B).
    a_dtype='fp6': fp6×fp4 (Float6E2M3FN for A, Float4E2M1FN for B).
    a_dtype='fp8': fp8×fp4 (Float8E4M3FN for A, Float4E2M1FN for B).
    """
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
    """Layout view over the CK-preshuffled B weight for one N-row tile (i32 units).
    Index [lane//16, lane%16, kt, half, None] -> i32<4:1> (16B = 32 fp4)."""
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


def compile_mxfp4_gemm(
    *,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    a_dtype: str = "fp4",
    out_dtype: str = "bf16",
    waves_per_eu: Optional[int] = None,
    enable_scheduler: Optional[bool] = None,
    use_async_copy: bool = True,
    dsrd_preload: int = -1,
    dvmem_preload: int = -1,
    batch: int = 1,
):
    """Compile MXFP4/MXFP6/MXFP8 preshuffle GEMM -> fn(C, A, B, scale_a, scale_b, bias, M, N, stream).

    a_dtype 'fp4'/'fp6'/'fp8' = MXFP4 (2 codes/byte) / MXFP6 (FP8-padded, gfx950) /
    MXFP8 E4M3 (1 byte/code, gfx950). B is CK-preshuffled MXFP4; scales are e8m0; bias
    unused. batch>1 = strided-batched over grid.z (batch=1 emits no batch math).
    """
    if a_dtype not in _A_ELEM:
        raise ValueError(f"a_dtype must be one of {sorted(_A_ELEM)}; got {a_dtype!r}")
    BM, BN, BK = tile_m, tile_n, tile_k
    if BK not in (128, 256) or K % BK != 0:
        raise ValueError(
            f"tile_k must be 128 or 256 dividing K; got tile_k={BK}, K={K}"
        )
    if K % 256 != 0:
        raise ValueError(f"K must be a multiple of 256 (e8m0 scale chunk); got K={K}")
    out_elem = BFloat16 if out_dtype == "bf16" else Float16

    # Row sizes + read_a fragment layout (i32 units). fp6/fp8: two ds_read_b128 halves
    # A_HI_OFF apart -> i32[A_NDW]; fp4: one -> i32[4]. A_GK_I32/A_KH_I32 = per-group
    # (gk=lane//16) / per-k-half strides.
    if a_dtype == "fp8":
        a_row_bytes, A_ROW_B = K, BK
        A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 32, 16, 8
    elif a_dtype == "fp6":
        a_row_bytes, A_ROW_B = K, BK
        A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 8, 32, 4, 6
    else:  # fp4: 2 codes/byte
        a_row_bytes, A_ROW_B = K // 2, BK // 2
        A_GK_I32, A_KH_I32 = 4, 16

    # Cooperative LDS A tile (row-major [m][col]) shared by the 4 N-waves -> no 4x
    # redundant A gmem reads. fp4/fp6 = 2/1 codes/byte.
    A_LDS_B = BM * A_ROW_B  # bytes per LDS A buffer
    A_ROW_I32 = A_ROW_B // 4

    K_HALF = K // 2
    KH4 = K_HALF // 4
    K_TILES = K // BK
    k_halves = BK // 128  # 16x16x128 MFMA k-steps per K-tile
    # e8m0 scale chunks are 256-K granular, B is 128-K granular: tiles_per_chunk
    # consecutive K-tiles share one scale word (upper/lower 16b select the 128-K half).
    tiles_per_chunk = 256 // BK  # 1 for tile_k=256, 2 for tile_k=128
    m_chunks = BM // 16
    num_acc_n = (BN // 4) // 16  # 16-col n-subblocks per wave
    _scale_chunk_dw = (
        K // 32 // 4 // 2
    ) * 64  # e8m0 strides (dwords), per shuffle_scale_w4
    _scale_k0_dw = 64

    n_coop = A_LDS_B // 256 // 16  # 16B cooperative loads per thread

    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)

    # Scheduler counts (sched_group_barrier interleave), per loop iter.
    sched_mfma_total = k_halves * m_chunks * num_acc_n
    # fp6/fp8: two 128-bit LDS reads per (mi, kh); fp4: one
    if a_dtype in ("fp6", "fp8"):
        sched_num_ds_load = m_chunks * k_halves * 2  # A LDS reads/thread (read_a)
    else:
        sched_num_ds_load = m_chunks * k_halves  # A LDS reads/thread (read_a)
    sched_num_gmem = (
        n_coop + num_acc_n * k_halves + m_pairs + n_pairs
    )  # A coop + B + scales
    sched_num_a_dswr = (
        0 if use_async_copy else n_coop
    )  # A LDS writes/thread (none for DMA)

    # The interleave helps lean (num_acc_n<=2) tiles always, and fat tiles when the A
    # fill is an async gmem->LDS DMA (the explicit drain otherwise exposes its latency);
    # on fat *sync* tiles it serializes the ds_write/ds_read stream.
    if enable_scheduler is None:
        enable_scheduler = num_acc_n <= 2 or use_async_copy
    if dsrd_preload < 0:
        dsrd_preload = sched_num_ds_load
    if dvmem_preload < 0:
        dvmem_preload = sched_num_gmem

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

        # Strided-batched: shift each operand base to batch bid_z (per-batch num_records
        # below keep ragged-M OOB correct within each batch).
        if const_expr(batch > 1):
            bz = fx.Int64(bid_z)
            arg_a = fx.Int64(arg_a) + bz * (fx.Int64(i32_m) * fx.Int64(a_row_bytes))
            arg_b = fx.Int64(arg_b) + bz * fx.Int64(N * (K // 2))
            a_sc_stride = (
                fx.Int64((i32_m + fx.Int32(31)) // fx.Int32(32))
                * fx.Int64(_scale_chunk_dw)
                * fx.Int64(4)
            )
            arg_scale_a = fx.Int64(arg_scale_a) + bz * a_sc_stride
            arg_scale_b = fx.Int64(arg_scale_b) + bz * fx.Int64(
                (N // 32) * _scale_chunk_dw * 4
            )

        # A: cooperative gmem->LDS then ds_read, bound to real M rows (ragged M OOB -> 0).
        a_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        _i8g = fx.PointerType.get(
            T.i8, address_space=fx.AddressSpace.Global, alignment=16
        )
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
        # A-LDS modeled as i32 (16B = 4 i32): fx.copy is dtype-agnostic, only the MMA
        # cares about sub-byte semantics. Store + fp4/fp6 read go through fx.copy.
        sA0_i32 = fx.recast_iter(Int32, lds.a0.ptr)
        lds_db = fx.Int32(fx.ptrtoint(lds.a1.ptr)) - fx.Int32(
            fx.ptrtoint(lds.a0.ptr)
        )  # ping/pong byte stride
        lds_db_i32 = lds_db // fx.Int32(4)
        lds_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)

        if const_expr(use_async_copy):
            dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
            _i8s = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
            sA0_i8 = fx.recast_iter(_i8s, lds.a0.ptr)

        def _iter_of(parity):  # parity in {0,1} (runtime) -> i32 LDS iterator
            return fx.add_offset(sA0_i32, parity * lds_db_i32)

        def _lds_view(base_iter, off_i32):
            return fx.make_view(fx.add_offset(base_iter, off_i32), fx.make_layout(4, 1))

        def coop_load_a(kt, base_iter):
            base_k_byte = kt * fx.Int32(A_ROW_B)
            for i in range_constexpr(n_coop):
                lin = (fx.Int32(i * 256) + fx.Int32(tid)) * fx.Int32(16)
                row = lin // fx.Int32(A_ROW_B)
                col = lin % fx.Int32(A_ROW_B)
                gmem_byte = (bx_m + row) * fx.Int32(a_row_bytes) + base_k_byte + col
                reg = fx.make_rmem_tensor(4, Int32)
                fx.copy_atom_call(a_copy, a_flat_div[None, gmem_byte], reg)
                fx.copy(
                    lds_copy,
                    reg,
                    _lds_view(
                        base_iter, row * fx.Int32(A_ROW_I32) + col // fx.Int32(4)
                    ),
                )

        # Async A: direct gmem->LDS DMA (buffer_load_lds), same row-major LDS layout as
        # coop_load_a. Issued after the B/scale loads so it overlaps the MFMAs.
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
                gmem_byte = (bx_m + row) * fx.Int32(a_row_bytes) + base_k_byte + col
                dst = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                src = fx.slice(a_flat_div, (None, gmem_byte))
                fx.copy(dma_atom, src, dst)

        def _read16(base_iter, off_i32):
            # ds_read_b128 straight into an i32[4] register fragment (no vec round-trip).
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
                        # fp6/fp8: pack two halves into i32[A_NDW]. fp8's two 16-code halves
                        # sit 64 K apart (f8f6f4 ABI); fp6 drops the 2 FP8-pad zeros.
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

        # e8m0 scales: flat buffers based at the allocation start, bounded to the real
        # size (so rows past M read OOB -> 0); m/n-pair + lane + chunk offset folded in.
        _i32g = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        _sc_layout = fx.make_layout(1 << 28, 1)
        a_sc_nrec = (
            fx.Int64((i32_m + fx.Int32(31)) // fx.Int32(32))
            * fx.Int64(_scale_chunk_dw)
            * fx.Int64(4)
        )
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
            (bx_m // fx.Int32(32) + fx.Int32(mp)) * fx.Int32(_scale_chunk_dw)
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
            # (sa, sb) e8m0 words per m-pair / n-pair for one 256-K chunk. readfirstlane
            # the uniform base+chunk into an SGPR soffset; only sc_lane is per-lane voffset.
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
            # tile_k=128: two 128-K tiles share one 256-K word -> shift the active half
            # into the low bytes the opsel reads. tile_k=256 (scale_shift=None) keeps both.
            if const_expr(scale_shift is not None):
                sh = _raw(scale_shift)
                sa_v = [arith.shrui(_raw(v), sh) for v in sa_v]
                sb_v = [arith.shrui(_raw(v), sh) for v in sb_v]
            # kh OUTERMOST: consecutive MFMAs write distinct accumulators (dense issue),
            # spacing the per-acc accumulation dependency across the (mi,ni) grid. Each
            # scaled MFMA = fx.gemm over the rank-1 i32[4] A/B fragments (one MmaAtomCall);
            # the atom bitcasts to fp4/fp6 and the e8m0 word rides scale_a=/scale_b=.
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

        # Scheduler hints: interleave the MFMAs with the vmem loads + A LDS read/writes.
        def build_scheduler(numer, denom):
            if const_expr(denom <= 0):
                return []
            if const_expr(numer <= 0):
                return [0] * denom
            out = []
            prev = 0
            for i in range_constexpr(denom):
                cur = ((i + 1) * numer + (denom - 1)) // denom
                out.append(cur - prev)
                prev = cur
            return out

        def hot_loop_scheduler():
            mfma_total = sched_mfma_total
            dswr_tail = min(sched_num_a_dswr, mfma_total)
            dsrd_preload_eff = min(int(dsrd_preload), sched_num_ds_load)
            dvmem_preload_eff = min(int(dvmem_preload), sched_num_gmem)
            vmem_remaining = sched_num_gmem - dvmem_preload_eff
            dsrd_remaining = sched_num_ds_load - dsrd_preload_eff
            if const_expr(0 < vmem_remaining < mfma_total):
                vmem_schedule = build_scheduler(vmem_remaining, vmem_remaining) + [
                    0
                ] * (mfma_total - vmem_remaining)
            else:
                vmem_schedule = build_scheduler(vmem_remaining, mfma_total)
            dsrd_schedule = build_scheduler(dsrd_remaining, mfma_total)
            dswr_start = max(mfma_total - dswr_tail - 2, 0)
            last_dsrd_mfma_idx = -1
            for sched_idx in range_constexpr(mfma_total):
                if const_expr(dsrd_schedule[sched_idx]):
                    last_dsrd_mfma_idx = sched_idx
            dswr_start = max(dswr_start, last_dsrd_mfma_idx + 1)
            idx_ds_read = dsrd_preload_eff
            idx_gmem_load = dvmem_preload_eff
            idx_ds_write = 0
            if const_expr(dvmem_preload_eff):
                rocdl.sched_vmem(dvmem_preload_eff)
            if const_expr(dsrd_preload_eff):
                rocdl.sched_dsrd(dsrd_preload_eff)
            for mfma_idx in range_constexpr(mfma_total):
                rocdl.sched_mfma(1)
                n_dsrd = dsrd_schedule[mfma_idx]
                if const_expr(n_dsrd and (idx_ds_read < sched_num_ds_load)):
                    if const_expr(idx_ds_read + n_dsrd > sched_num_ds_load):
                        n_dsrd = sched_num_ds_load - idx_ds_read
                    if const_expr(n_dsrd):
                        rocdl.sched_dsrd(n_dsrd)
                        idx_ds_read += n_dsrd
                n_vmem = vmem_schedule[mfma_idx]
                if const_expr(n_vmem and (idx_gmem_load < sched_num_gmem)):
                    if const_expr(idx_gmem_load + n_vmem > sched_num_gmem):
                        n_vmem = sched_num_gmem - idx_gmem_load
                    if const_expr(n_vmem):
                        rocdl.sched_vmem(n_vmem)
                        idx_gmem_load += n_vmem
                if const_expr((idx_ds_write < dswr_tail) and (mfma_idx >= dswr_start)):
                    rocdl.sched_dswr(1)
                    idx_ds_write += 1
            if const_expr(idx_ds_write < sched_num_a_dswr):
                rocdl.sched_dswr(sched_num_a_dswr - idx_ds_write)
            rocdl.sched_barrier(0)

        accs_init = [
            Vec.filled(4, 0.0, Float32).ir_value() for _ in range_constexpr(n_acc)
        ]

        # Double-buffered LDS-A: prefetch tile iv+1's A into the other buffer while the
        # MFMAs compute tile iv. B/scales are loaded per-tile (latency hidden at 3 waves).
        if const_expr(use_async_copy):
            dma_a_to_lds(fx.Int32(0), fx.Int32(0))
            rocdl.s_waitcnt(0)
        else:
            coop_load_a(fx.Int32(0), _iter_of(fx.Int32(0)))
        gpu.barrier()
        # 1 tile/iter ping-pong; tile_k=128 shares a 256-K scale chunk across 2 K-tiles.
        for iv, state in range(
            fx.Index(0), fx.Index(K_TILES), fx.Index(1), init=accs_init
        ):
            accs = list(state)
            kt = fx.Int32(iv)
            cur = kt % fx.Int32(2)
            nxt = (kt + fx.Int32(1)) % fx.Int32(2)
            nkt = kt + fx.Int32(1)
            pf_kt = nkt - nkt // fx.Int32(
                K_TILES
            )  # clamp last-iter prefetch to K_TILES-1
            chunk_kt = kt if tiles_per_chunk == 1 else kt // fx.Int32(tiles_per_chunk)
            scale_shift = (
                None
                if tiles_per_chunk == 1
                else (kt % fx.Int32(tiles_per_chunk)) * fx.Int32(16)
            )
            if const_expr(not use_async_copy):
                coop_load_a(pf_kt, _iter_of(nxt))  # prefetch A tile iv+1 -> LDS
            av = read_a(cur)
            bv = load_b(kt)
            sa_v, sb_v = load_sc(chunk_kt)
            if const_expr(use_async_copy):
                dma_a_to_lds(
                    pf_kt, nxt
                )  # A DMA AFTER B/scale loads -> overlaps the MFMAs
            accs = compute(
                accs, av, bv, sa_v, sb_v, scale_shift
            )  # overlaps the A prefetch
            if const_expr(enable_scheduler):
                hot_loop_scheduler()
            if const_expr(use_async_copy):
                rocdl.s_waitcnt(0)  # drain the A DMA before the barrier
            gpu.barrier()
            results = yield accs
        accs = results

        # Epilogue: manual C store (MFMA 16x16 C: lane l -> col base+l%16, row m*16+
        # (l//16)*4+ii). MX scale already folded in. Bound to actual M rows (ragged M).
        c_nrec = fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
        if const_expr(batch > 1):
            c_addr = (
                fx.Int64(fx.ptrtoint(fx.get_iter(arg_c))) + fx.Int64(bid_z) * c_nrec
            )
            c_rsrc = buffer_ops.create_buffer_resource_from_addr(
                c_addr, num_records_bytes=c_nrec
            )
        else:
            c_rsrc = buffer_ops.create_buffer_resource(
                arg_c, max_size=False, num_records_bytes=c_nrec
            )
        col_w = by_n + wave * fx.Int32(BN // 4) + lane_mod_16
        for mi in range_constexpr(m_chunks):
            row_m = bx_m + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4)
            for ni in range_constexpr(num_acc_n):
                col = col_w + fx.Int32(ni * 16)
                acc = Vec(accs[mi * num_acc_n + ni])
                for ii in range_constexpr(4):
                    val = acc[ii].to(out_elem)
                    off = (row_m + fx.Int32(ii)) * fx.Int32(N) + col
                    buffer_ops.buffer_store(val.ir_value(), c_rsrc, off)

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
    ):
        from flydsl.compiler.kernel_function import CompilationContext

        CompilationContext.get_current()
        a_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_a)))
        b_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_b)))
        sa_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_scale_a)))
        sb_addr = fx.Int64(fx.ptrtoint(fx.get_iter(arg_scale_b)))
        M_max = 65536
        arg_c_2d = fx.Tensor(
            fx.make_view(fx.get_iter(arg_c), fx.make_layout((M_max, N), (N, 1)))
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
            value_attrs={"rocdl.waves_per_eu": waves_per_eu},
        ).launch(grid=(gx, gy, batch), block=(256, 1, 1), stream=stream)

    return launch_gemm


def compile_mxfp6_gemm(**kw):
    """MXFP6 (E2M3) A × MXFP4 B preshuffle GEMM (gfx950); wraps a_dtype='fp6'."""
    return compile_mxfp4_gemm(a_dtype="fp6", **kw)


def compile_mxfp8_gemm(**kw):
    """MXFP8 (E4M3) A × MXFP4 B preshuffle GEMM (gfx950); wraps a_dtype='fp8'."""
    return compile_mxfp4_gemm(a_dtype="fp8", **kw)

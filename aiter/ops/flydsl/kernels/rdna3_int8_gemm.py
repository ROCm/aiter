#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors
# Adapted from ROCm/FlyDSL kernels/gemm/rdna3_int8_gemm.py at
# 1941889400621f1fc3d8f03bada1a4e7380cdf7e.
"""INT8 WMMA GEMM kernel for RDNA3 / RDNA3.5 (gfx11*, wave32).

Computes ``C[M, N] = A[M, K] @ B_T[N, K].T`` for INT8 or UINT8 inputs.
The epilogue can either store the exact INT32 accumulator or apply per-row A
and per-column B scales before converting to a floating-point output type.

``split_k`` cuts K into slices that each get their own workgroup, for shapes
whose output tiles alone leave most of the device idle. The slices accumulate
into the INT32 output atomically, which is exact.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import fly
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.runtime.device import get_rocm_arch

from .kernels_common import atomic_add_i32

WMMA_M = 16
WMMA_N = 16
WMMA_K = 16
WAVE_SIZE = 32

LOAD_VEC = 16

LDS_BYTES = 64 * 1024


IN_DTYPES = {"int8": (fx.Int8, True), "uint8": (fx.Uint8, False)}
OUT_DTYPES = {"i32": fx.Int32, "f32": fx.Float32, "bf16": fx.BFloat16, "f16": fx.Float16}


def _sched_plan(reg_m, reg_n, reg_k, g2s_chunks):
    per_rk_wmma = reg_m * reg_n
    per_rk_dsrd = reg_m + reg_n
    chunk = max(1, reg_n)

    plan = [("vmem", g2s_chunks), ("dsrd", per_rk_dsrd)]
    dsrd_left = per_rk_dsrd * (reg_k - 1)
    dsrd_step = chunk
    dswr_left = g2s_chunks
    dswr_chunk = max(1, g2s_chunks // 2)
    for _ in range(reg_k):
        issued = 0
        while issued < per_rk_wmma:
            n = min(chunk, per_rk_wmma - issued)
            plan.append(("mfma", n))
            issued += n
            if dsrd_left:
                take = min(dsrd_step, dsrd_left)
                plan.append(("dsrd", take))
                dsrd_left -= take
            elif dswr_left:
                take = min(dswr_chunk, dswr_left)
                plan.append(("dswr", take))
                dswr_left -= take
    if dswr_left:
        plan.append(("dswr", dswr_left))
    return tuple(plan)


def _k_pad(block_k):
    return 16 if (block_k // 16) % 2 == 0 else 32


def _group_width(grid_m, group_m):
    return max(d for d in range(1, min(group_m, grid_m) + 1) if grid_m % d == 0)


def _swizzle_tile_id(pid, grid_n, group_width):
    num_pid_in_group = group_width * grid_n
    group_id = pid // num_pid_in_group
    pid_in_group = pid % num_pid_in_group
    return group_id * group_width + (pid_in_group % group_width), pid_in_group // group_width


def _zero_output(tensor, stream):
    """Clear the split-K output on the stream the kernel will run on."""
    import torch

    if isinstance(stream, torch.cuda.Stream):
        with torch.cuda.stream(stream):
            tensor.zero_()
    else:
        tensor.zero_()


def _default_reg_m(M, N):
    try:
        import torch

        slots = torch.cuda.get_device_properties(0).multi_processor_count
    except Exception:
        slots = 48

    wgs = -(-M // 128) * -(-N // 128)
    return 4 if wgs >= 40 * slots else 2


def create_wmma_int8_gemm_module(
    M: int,
    N: int,
    K: int,
    in_dtype="int8",
    out_dtype="i32",
    *,
    scale_mode="none",
    reg_m=None,
    reg_n=4,
    reg_k=4,
    waves_m=2,
    waves_n=2,
    group_m=8,
    lds_layout="pad",
    sched_hint=True,
    stagger=1,
    persistent_wgs=0,
    split_k=1,
    lda=None,
    ldb=None,
    ldc=None,
):
    if in_dtype not in IN_DTYPES:
        raise ValueError(f"in_dtype must be one of {sorted(IN_DTYPES)}, got {in_dtype!r}")
    if out_dtype not in OUT_DTYPES:
        raise ValueError(f"out_dtype must be one of {sorted(OUT_DTYPES)}, got {out_dtype!r}")
    if scale_mode not in ("none", "row_col"):
        raise ValueError(f"scale_mode must be 'none' or 'row_col', got {scale_mode!r}")
    if scale_mode == "row_col" and out_dtype == "i32":
        raise ValueError("scale_mode='row_col' dequantises to a float; out_dtype cannot be 'i32'")

    elem_dtype, elem_signed = IN_DTYPES[in_dtype]
    out_elem_cls = OUT_DTYPES[out_dtype]

    ld_a = K if lda is None else int(lda)
    ld_b = K if ldb is None else int(ldb)
    ld_c = N if ldc is None else int(ldc)
    if ld_a < K or ld_b < K or ld_c < N:
        raise ValueError(
            f"leading dimensions must cover the operands: lda={ld_a}, ldb={ld_b} " f"need K={K}, ldc={ld_c} needs N={N}"
        )

    if ld_a % LOAD_VEC or ld_b % LOAD_VEC:
        raise ValueError(
            f"lda={ld_a} and ldb={ld_b} must be multiples of {LOAD_VEC} to keep "
            f"each row 16-byte aligned for the 128-bit copy"
        )

    gpu_arch = str(get_rocm_arch() or "")
    if not gpu_arch.startswith("gfx11"):
        raise RuntimeError(
            f"rdna3_int8_gemm requires gfx11* (RDNA3 / RDNA3.5); current arch is {gpu_arch!r}. "
            "gfx120* (RDNA4) has no integer WMMA."
        )

    if reg_m is None:
        reg_m = _default_reg_m(M, N)

    BLOCK_M = WMMA_M * reg_m * waves_m
    BLOCK_N = WMMA_N * reg_n * waves_n
    BLOCK_K = WMMA_K * reg_k
    NUM_WAVES = waves_m * waves_n
    THREADS_PER_BLOCK = NUM_WAVES * WAVE_SIZE

    assert reg_k >= 1

    THRS_K = BLOCK_K // LOAD_VEC
    THRS_M = THREADS_PER_BLOCK // THRS_K

    G2S_CHUNKS = (BLOCK_M + BLOCK_N) * BLOCK_K // THREADS_PER_BLOCK // LOAD_VEC
    SCHED_PLAN = _sched_plan(reg_m, reg_n, reg_k, G2S_CHUNKS) if sched_hint else ()
    assert THRS_K * THRS_M == THREADS_PER_BLOCK
    assert BLOCK_M % THRS_M == 0 and BLOCK_N % THRS_M == 0

    assert lds_layout in ("pad", "kblock")
    if lds_layout == "kblock":
        assert BLOCK_K % LOAD_VEC == 0
    k_pad = 0 if lds_layout == "kblock" else _k_pad(BLOCK_K)
    ROW_STRIDE_A = BLOCK_K + k_pad
    ROW_STRIDE_B = BLOCK_K + k_pad
    LDS_A_SIZE = BLOCK_M * ROW_STRIDE_A
    LDS_B_SIZE = BLOCK_N * ROW_STRIDE_B
    LDS_ONE_BUF = LDS_A_SIZE + LDS_B_SIZE
    LDS_TOTAL = 2 * LDS_ONE_BUF
    if LDS_TOTAL > LDS_BYTES:
        raise ValueError(
            f"{BLOCK_M}x{BLOCK_N}x{BLOCK_K} with lds_layout={lds_layout!r} needs "
            f"{LDS_TOTAL} bytes of LDS for the double buffer, over the {LDS_BYTES}-byte "
            f"per-workgroup budget; lower reg_k or use lds_layout='kblock'"
        )

    assert N % BLOCK_N == 0
    assert K % BLOCK_K == 0
    partial_m = M % BLOCK_M != 0

    num_k_tiles = K // BLOCK_K
    if num_k_tiles < 2:
        raise ValueError(f"Need at least 2 K-tiles for prefetch pipeline; got K={K}, BLOCK_K={BLOCK_K}")

    grid_m = -(-M // BLOCK_M)
    grid_n = N // BLOCK_N

    # Buffer descriptor offsets are 32-bit (BufferFatPtr.h: kOffsetBitWidth = 32) and the
    # element-to-byte scaling happens in 32 bits too. At or above 4 GiB the offset of a
    # tail-block row wraps to a small value that compares as in-range, so a checked
    # descriptor (max_size=False, OOB_SELECT=3) would silently redirect the access
    # near the start of the buffer instead of dropping it. Reject those shapes on the
    # host rather than ship a checked-but-unsafe descriptor.
    padded_m = grid_m * BLOCK_M
    in_bytes = elem_dtype.width // 8
    out_bytes = out_elem_cls.width // 8
    for operand, span_bytes in (
        ("A", ((padded_m - 1) * ld_a + K) * in_bytes),
        ("B", ((N - 1) * ld_b + K) * in_bytes),
        ("C", ((padded_m - 1) * ld_c + N) * out_bytes),
    ):
        if span_bytes >= 2**32:
            raise ValueError(
                f"{operand} spans {span_bytes} bytes, which overflows the 32-bit buffer "
                f"offset; keep each operand under 4 GiB (M={M}, N={N}, K={K})"
            )

    group_width = _group_width(grid_m, group_m)

    assert stagger >= 0
    stagger_step = int(stagger) if num_k_tiles & (num_k_tiles - 1) == 0 else 0

    num_tiles = grid_m * grid_n
    persist_wgs = int(persistent_wgs)
    if persist_wgs < 0 or persist_wgs > num_tiles:
        raise ValueError(f"persistent_wgs must be between 0 (plain grid) and num_tiles={num_tiles}, got {persist_wgs}")

    persist_rot_step = int(stagger) if persist_wgs else 0
    if persist_wgs:
        stagger_step = 0

    splits = int(split_k)
    if splits < 1:
        raise ValueError(f"split_k must be at least 1, got {splits}")
    if splits > 1:
        # Slices accumulate atomically into one output tile. Integer adds stay
        # exact under any order; a scaled epilogue would need the full sum first.
        if out_dtype != "i32" or scale_mode != "none":
            raise ValueError("split_k > 1 accumulates atomically and needs out_dtype='i32' with scale_mode='none'")
        if persist_wgs:
            raise ValueError("split_k and a persistent grid both remap the grid; use one or the other")
        if num_k_tiles % splits:
            raise ValueError(f"split_k={splits} must divide the {num_k_tiles} K-tiles of K={K}, BLOCK_K={BLOCK_K}")
        if num_k_tiles // splits < 2:
            raise ValueError(f"split_k={splits} leaves under 2 K-tiles per slice for the prefetch pipeline")

    K_STEPS = num_k_tiles // splits
    if splits > 1:
        stagger_step = int(stagger) if K_STEPS & (K_STEPS - 1) == 0 else 0

    scaled = scale_mode == "row_col"

    def _make_mma_atom():
        return fx.make_mma_atom(
            fx.rocdl.WMMA(
                WMMA_M,
                WMMA_N,
                WMMA_K,
                elem_dtype,
                fx.Int32,
                sign_a=elem_signed,
                sign_b=elem_signed,
            )
        )

    @fx.struct
    class _SharedStorage:
        lds: fx.Array[elem_dtype, LDS_TOTAL, 16]

    @flyc.kernel
    def wmma_gemm_kernel(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        arg_sa: fx.Tensor,
        arg_sb: fx.Tensor,
        tiled_mma: fx.TiledMma,
        tiled_copy_g2s: fx.TiledCopy,
    ):

        lds_storage = fx.SharedAllocator().allocate(_SharedStorage).peek()
        lds_ptr = lds_storage.lds.ptr

        mma_atom = _make_mma_atom()
        acc_vec_ty = fx.Vector.make_type(8, fx.Int32)

        def _wmma_op(a_vec, b_vec, acc):
            return fx.Vector(
                fly.mma_atom_call_ssa(
                    [acc_vec_ty],
                    mma_atom,
                    a_vec.ir_value(),
                    b_vec.ir_value(),
                    acc.ir_value(),
                )
            )

        def _v16_load(elem_off):
            ptr_off = fx.add_offset(lds_ptr, fx.make_int_tuple(fx.Int32(elem_off)))
            typed_ptr = fx.recast_iter(elem_dtype, ptr_off)
            return fx.make_view(typed_ptr, fx.make_layout(LOAD_VEC, 1)).load()

        tid = gpu.thread_id("x")
        pid = gpu.block_id("x")

        wave_id = tid // 32
        lane = tid % 32

        lane16 = lane % 16
        lane_half = lane // 16

        if const_expr(splits > 1):
            # Neighbouring workgroups stay inside one K slice so they share B in L2.
            tile_pid = fx.Int32(pid) % num_tiles
            k_base = fx.Int32(pid) // num_tiles * K_STEPS
        else:
            tile_pid = pid
            k_base = None

        if const_expr(stagger_step):
            k_first = fx.Int32(pid) * stagger_step % K_STEPS
        else:
            k_first = 0

        def _k_tile(step, rot=None, n_iter=None):
            if const_expr(stagger_step):
                local = (k_first + fx.Int32(step)) % K_STEPS
            elif const_expr(persist_rot_step):
                kk = rot + fx.Int32(step)
                return (kk >= n_iter).select(kk - n_iter, kk)
            else:
                local = step
            return local if const_expr(k_base is None) else k_base + fx.Int32(local)

        wave_m = wave_id // waves_n
        wave_n = wave_id % waves_n

        thr_g2s = tiled_copy_g2s.get_slice(tid)
        thr_mma = tiled_mma.thr_slice(tid)
        copy_out = fx.make_copy_atom(fx.rocdl.BufferCopy(out_elem_cls.width), out_elem_cls)
        thr_r2g_C = fx.make_tiled_copy_C(copy_out, tiled_mma).get_slice(tid)

        def _tile_operands(bid_m, bid_n):
            tA = fx.flat_divide(
                fx.rocdl.make_buffer_tensor(arg_a, max_size=not partial_m),
                fx.make_tile(BLOCK_M, BLOCK_K),
            )[None, None, bid_m, None]
            tB = fx.flat_divide(fx.rocdl.make_buffer_tensor(arg_bt), fx.make_tile(BLOCK_N, BLOCK_K))[
                None, None, bid_n, None
            ]
            return thr_g2s.partition_S(tA), thr_g2s.partition_S(tB)

        buf_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
        uni_copy = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)

        def _lds_dst(buf_offset, base, rows, row_stride):
            ptr = fx.add_offset(lds_ptr, fx.make_int_tuple(buf_offset + base))
            if const_expr(lds_layout == "kblock"):
                layout = fx.make_layout(
                    (rows, (LOAD_VEC, BLOCK_K // LOAD_VEC)),
                    (LOAD_VEC, (1, rows * LOAD_VEC)),
                )
            else:
                layout = fx.make_layout((rows, BLOCK_K), (row_stride, 1))
            view = fx.make_view(fx.recast_iter(elem_dtype, ptr), layout)
            return thr_g2s.partition_D(view)[None, None, None]

        def _pA_s(buf_offset):
            return _lds_dst(buf_offset, 0, BLOCK_M, ROW_STRIDE_A)

        def _pB_s(buf_offset):
            return _lds_dst(buf_offset, LDS_A_SIZE, BLOCK_N, ROW_STRIDE_B)

        def _lds_elem(rows, row_stride, row, col):
            if const_expr(lds_layout == "kblock"):
                return (col // LOAD_VEC * rows + row) * LOAD_VEC + col % LOAD_VEC
            return row * row_stride + col

        frag_copy_A = fx.make_fragment_like(_pA_s(0))
        frag_copy_B = fx.make_fragment_like(_pB_s(0))

        def _gmem_load(pA_g, pB_g, k_tile):
            fx.copy(buf_copy, pA_g[None, None, None, k_tile], frag_copy_A)
            fx.copy(buf_copy, pB_g[None, None, None, k_tile], frag_copy_B)

        def _lds_store(buf_offset):
            fx.copy(uni_copy, frag_copy_A, _pA_s(buf_offset))
            fx.copy(uni_copy, frag_copy_B, _pB_s(buf_offset))

        def _load_b_from_lds(rk, buf_offset):
            vecs = []
            col = LOAD_VEC * rk
            for rn in range_constexpr(reg_n):
                row = wave_n * (reg_n * WMMA_N) + WMMA_N * rn + lane16
                vecs.append(_v16_load(buf_offset + LDS_A_SIZE + _lds_elem(BLOCK_N, ROW_STRIDE_B, row, col)))
            return vecs

        def _load_a_single_from_lds(rk, rm_val, buf_offset):
            col = LOAD_VEC * rk
            row = wave_m * (reg_m * WMMA_M) + WMMA_M * rm_val + lane16
            return _v16_load(buf_offset + _lds_elem(BLOCK_M, ROW_STRIDE_A, row, col))

        def _barrier():
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

        def _do_compute_rk(accs_in, rk, buf_offset, b_vecs):
            new_accs = list(accs_in)
            a_next = _load_a_single_from_lds(rk, 0, buf_offset)
            for rm in range_constexpr(reg_m):
                a_vec = a_next
                if const_expr(rm + 1 < reg_m):
                    a_next = _load_a_single_from_lds(rk, rm + 1, buf_offset)
                for rn in range_constexpr(reg_n):
                    idx = rm * reg_n + rn
                    new_accs[idx] = _wmma_op(a_vec, b_vecs[rn], new_accs[idx])
            return new_accs

        def _compute_k_tile(accs_in, buf_offset):
            new_accs = list(accs_in)
            for rk in range_constexpr(reg_k):
                new_accs = _do_compute_rk(new_accs, rk, buf_offset, _load_b_from_lds(rk, buf_offset))
            return new_accs

        def _sched_k_tile():
            emit = {
                "vmem": rocdl.sched_vmem,
                "mfma": rocdl.sched_mfma,
                "dsrd": rocdl.sched_dsrd,
                "dswr": rocdl.sched_dswr,
            }
            for group, count in SCHED_PLAN:
                emit[group](count)

        zero_acc = fx.full(8, 0, fx.Int32)
        n_acc = reg_m * reg_n
        c_lds_buf_stride = LDS_ONE_BUF

        def _one_k_tile(pA_g, pB_g, s_accs, read_off, write_off, load_tile):
            _gmem_load(pA_g, pB_g, load_tile)
            s_accs = _compute_k_tile(s_accs, read_off)
            _lds_store(write_off)
            if const_expr(sched_hint):
                _sched_k_tile()
            _barrier()
            return s_accs

        def _accumulate(pA_g, pB_g, rot=None):
            if const_expr(persist_wgs):
                _barrier()

            _gmem_load(pA_g, pB_g, _k_tile(fx.Int32(0), rot, K_STEPS))
            _lds_store(0)
            _barrier()

            init_state = [zero_acc for _ in range_constexpr(n_acc)]

            for iv, state in range(0, K_STEPS - 1, 1, init=init_state):
                s_accs = list(state[:n_acc])
                s_accs = _one_k_tile(
                    pA_g,
                    pB_g,
                    s_accs,
                    iv % 2 * c_lds_buf_stride,
                    (1 - iv % 2) * c_lds_buf_stride,
                    _k_tile(iv + 1, rot, K_STEPS),
                )
                results = yield list(s_accs)

            return _compute_k_tile(list(results[:n_acc]), ((K_STEPS - 1) % 2) * c_lds_buf_stride)

        if const_expr(scaled):
            sa_view = fx.make_view(fx.get_iter(arg_sa), fx.make_layout((M, 1), (1, 1)))
            sb_view = fx.make_view(fx.get_iter(arg_sb), fx.make_layout((N, 1), (1, 1)))

        def _load_scale_a(row):
            if const_expr(partial_m):
                row = (row < fx.Int32(M)).select(row, fx.Int32(M - 1))
            return sa_view[row, 0]

        def _atomic_add_row(accs, row, col_base, rm, si):
            offset = row * ld_c + col_base
            for rn in range_constexpr(reg_n):
                atomic_add_i32(
                    arg_c, accs[rm * reg_n + rn][si], offset + WMMA_N * rn, "agent"
                )

        def _atomic_add_C(accs, bid_m, bid_n):
            """Add this K slice into C.

            Element ``si`` of the accumulator for ``(rm, rn)`` holds row
            ``2 * si + lane // 16`` and column ``lane % 16`` of its 16x16 WMMA
            tile, the same mapping the scaled epilogue reads its scales with.
            """
            row_base = fx.Int32(bid_m) * BLOCK_M + fx.Int32(wave_m) * (reg_m * WMMA_M) + fx.Int32(lane_half)
            col_base = fx.Int32(bid_n) * BLOCK_N + fx.Int32(wave_n) * (reg_n * WMMA_N) + fx.Int32(lane16)
            for rm in range_constexpr(reg_m):
                for si in range_constexpr(8):
                    row = row_base + fx.Int32(WMMA_M * rm + 2 * si)
                    if const_expr(partial_m):
                        if row < fx.Int32(M):
                            _atomic_add_row(accs, row, col_base, rm, si)
                    else:
                        _atomic_add_row(accs, row, col_base, rm, si)

        def _store_C(accs, bid_m, bid_n):
            if const_expr(splits > 1):
                _atomic_add_C(accs, bid_m, bid_n)
                return

            tC = fx.flat_divide(
                fx.rocdl.make_buffer_tensor(arg_c, max_size=not partial_m),
                fx.make_tile(BLOCK_M, BLOCK_N),
            )[None, None, bid_m, bid_n]
            frag_C = thr_mma.make_fragment_C(tC)
            pC_g = thr_r2g_C.partition_S(tC)
            if const_expr(out_elem_cls is fx.Int32):
                frag_C_out = frag_C
            else:
                frag_C_out = fx.make_fragment_like(frag_C, out_elem_cls.ir_type)
            frag_C_retile = thr_r2g_C.retile(frag_C_out)

            if const_expr(scaled):
                row_base = fx.Int32(bid_m) * BLOCK_M + fx.Int32(wave_m) * (reg_m * WMMA_M) + fx.Int32(lane_half)
                col_base = fx.Int32(bid_n) * BLOCK_N + fx.Int32(wave_n) * (reg_n * WMMA_N) + fx.Int32(lane16)
                sb = [sb_view[col_base + WMMA_N * rn, 0] for rn in range_constexpr(reg_n)]
                sa = [
                    [_load_scale_a(row_base + WMMA_M * rm + 2 * si) for si in range_constexpr(8)]
                    for rm in range_constexpr(reg_m)
                ]
                out_elems = [
                    (accs[rm * reg_n + rn][si].to(fx.Float32) * sa[rm][si] * sb[rn]).to(out_elem_cls)
                    for rn in range_constexpr(reg_n)
                    for rm in range_constexpr(reg_m)
                    for si in range_constexpr(8)
                ]
            else:
                ordered_accs = [accs[rm * reg_n + rn] for rn in range_constexpr(reg_n) for rm in range_constexpr(reg_m)]
                if const_expr(out_elem_cls is fx.Int32):
                    out_elems = [acc[si] for acc in ordered_accs for si in range_constexpr(8)]
                else:
                    out_elems = [acc[si].to(out_elem_cls) for acc in ordered_accs for si in range_constexpr(8)]

            frag_C_out.store(fx.Vector.from_elements(out_elems, out_elem_cls))
            fx.copy(copy_out, frag_C_retile, pC_g)

        if const_expr(not persist_wgs):
            bid_m, bid_n = _swizzle_tile_id(tile_pid, grid_n, group_width)
            pA_g, pB_g = _tile_operands(bid_m, bid_n)
            accs = _accumulate(pA_g, pB_g)
            _store_C(accs, bid_m, bid_n)
        else:
            pid32 = fx.Int32(pid)
            t_first = pid32 * num_tiles // persist_wgs
            t_last = (pid32 + 1) * num_tiles // persist_wgs - 1

            for t, _carry in range(t_first, t_last + 1, 1, init=[fx.Int32(0)]):
                t32 = fx.Int32(t)
                bid_m, bid_n = _swizzle_tile_id(t32, grid_n, group_width)
                pA_g, pB_g = _tile_operands(bid_m, bid_n)
                if const_expr(persist_rot_step):
                    rot = pid32 * persist_rot_step % fx.Int32(num_k_tiles)
                else:
                    rot = None
                accs = _accumulate(pA_g, pB_g, rot)
                _store_C(accs, bid_m, bid_n)
                _ = yield [fx.Int32(0)]

    @flyc.jit
    def launch_gemm(
        arg_c: fx.Tensor,
        arg_a: fx.Tensor,
        arg_bt: fx.Tensor,
        arg_sa: fx.Tensor,
        arg_sb: fx.Tensor,
        stream: fx.Stream,
    ):
        tiled_mma = fx.make_tiled_mma(
            _make_mma_atom(),
            fx.make_layout((waves_m, waves_n, 1), (waves_n, 1, 0)),
            permutation=(
                fx.make_layout((WMMA_M, waves_m, reg_m), (1, WMMA_M * reg_m, WMMA_M)),
                fx.make_layout((WMMA_N, waves_n, reg_n), (1, WMMA_N * reg_n, WMMA_N)),
                WMMA_K,
            ),
        )
        tiled_copy_g2s = fx.make_tiled_copy(
            fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype),
            fx.make_layout(
                ((THRS_K, THRS_M), (1, LOAD_VEC)),
                ((THRS_M * LOAD_VEC, 1), (1, THRS_M)),
            ),
            fx.make_tile(THRS_M, BLOCK_K),
        )

        arg_a_2d = fx.make_view(fx.get_iter(arg_a), fx.make_layout((M, K), (ld_a, 1)))
        arg_bt_2d = fx.make_view(fx.get_iter(arg_bt), fx.make_layout((N, K), (ld_b, 1)))
        arg_c_2d = fx.make_view(fx.get_iter(arg_c), fx.make_layout((M, N), (ld_c, 1)))

        total_blocks = persist_wgs if persist_wgs else grid_m * grid_n * splits
        launcher = wmma_gemm_kernel(arg_c_2d, arg_a_2d, arg_bt_2d, arg_sa, arg_sb, tiled_mma, tiled_copy_g2s)
        launcher.launch(
            grid=(total_blocks, 1, 1),
            block=(THREADS_PER_BLOCK, 1, 1),
            stream=stream,
        )

    def launch(arg_c, arg_a, arg_bt, stream, scale_a=None, scale_b=None):
        if scaled and (scale_a is None or scale_b is None):
            raise ValueError("scale_mode='row_col' needs scale_a and scale_b")
        if splits > 1:
            # The epilogue accumulates, so the output has to start at zero.
            _zero_output(arg_c, stream)
        # The unscaled kernel ignores the scale arguments but the fixed JIT
        # signature still demands valid device pointers.
        return launch_gemm(
            arg_c,
            arg_a,
            arg_bt,
            arg_c if scale_a is None else scale_a,
            arg_c if scale_b is None else scale_b,
            stream,
        )

    return launch, BLOCK_M, BLOCK_N, BLOCK_K

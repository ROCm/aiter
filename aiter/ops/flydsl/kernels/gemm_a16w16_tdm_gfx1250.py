# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Wave32 BF16/FP16 TDM GEMM with optional interleave and cluster split-K."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, llvm, range_constexpr, rocdl, tdm_ops
from flydsl.expr.rocdl.cluster import cluster_barrier
from flydsl.expr.typing import T

from .gemm_a16w16_gfx950_utils import BlockSwizzle

WAVE_SIZE = 32
LDS_PAD = 8
NUM_XCDS = 8
NUM_PIDS_THRESHOLD = 256
GEMM_DTYPE_BF16 = 2
GEMM_DTYPE_FP16 = 3
GEMM_DTYPE_NAMES = {GEMM_DTYPE_BF16: "bf16", GEMM_DTYPE_FP16: "fp16"}
SUPPORTED_MMA = {GEMM_DTYPE_BF16: ((16, 16, 32),), GEMM_DTYPE_FP16: ((16, 16, 32),)}
GEMM_DTYPE_BITS = {GEMM_DTYPE_BF16: 16, GEMM_DTYPE_FP16: 16}
FRAGMENT_LOAD_BITS = 128
STORE_VEC_BITS = 128


def _validate_interleaved_tile(
    block_m,
    block_n,
    m_waves,
    n_waves,
    mma_m,
    mma_n,
    mma_k,
    elem_bits,
    b_k_contiguous,
    interleave,
):
    """Each quadrant must hide the independent loads of its next half-fragment."""
    if type(interleave) is not bool:
        raise ValueError("interleave must be bool")
    if not interleave:
        return
    m_steps = block_m // (m_waves * mma_m)
    n_steps = block_n // (n_waves * mma_n)
    half_m_steps, half_n_steps = (m_steps // 2, n_steps // 2)
    fragment_loads = mma_m * mma_k // WAVE_SIZE // (FRAGMENT_LOAD_BITS // elem_bits)
    if (
        not b_k_contiguous
        or m_steps % 2
        or n_steps % 2
        or (
            half_m_steps * half_n_steps
            < max(half_m_steps, half_n_steps) * fragment_loads
        )
    ):
        raise ValueError(
            "interleave requires NT and even quadrants with room for LDS loads"
        )


def validate_gemm_gfx1250_config(
    m,
    n,
    k,
    block_m,
    block_n,
    block_k,
    stages,
    m_waves,
    n_waves,
    group_m,
    dtype_id,
    mma_m,
    mma_n,
    mma_k,
    b_k_contiguous=True,
    split_k=1,
    interleave=False,
    split_k_atomic=False,
):
    if type(split_k_atomic) is not bool:
        raise ValueError("split_k_atomic must be bool")
    if split_k_atomic and (dtype_id != GEMM_DTYPE_BF16 or split_k == 1 or n % 2):
        raise ValueError("split_k_atomic requires BF16, split_k > 1 and even N")
    values = (
        m,
        n,
        k,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        mma_m,
        mma_n,
        mma_k,
    )
    if any((type(v) is not int or v <= 0 for v in values)):
        raise ValueError(
            "dimensions, tiles, stages, wave counts and WMMA sizes must be positive integers"
        )
    supported_mma = SUPPORTED_MMA.get(dtype_id)
    mma = (mma_m, mma_n, mma_k)
    if supported_mma is None or mma not in supported_mma:
        dtype_name = GEMM_DTYPE_NAMES.get(dtype_id, "unknown")
        raise ValueError(f"unsupported WMMA {mma_m}x{mma_n}x{mma_k} for {dtype_name}")
    if type(group_m) is not int or group_m < 0:
        raise ValueError("group_m must be a nonnegative integer")
    if not 2 <= stages <= 33:
        raise ValueError("stages must be in [2, 33] (two TDM loads per stage)")
    if block_k not in (32, 64, 128, 256, 512):
        raise ValueError("block_k must be a power of two in [32, 512]")
    waves = m_waves * n_waves
    if waves > 32:
        raise ValueError("a workgroup cannot exceed 1024 threads")
    if block_m % (m_waves * mma_m) or block_n % (n_waves * mma_n) or block_k % mma_k:
        raise ValueError("tile dimensions must be divisible by the WMMA atom")
    if not (
        1 <= block_m // (m_waves * mma_m) <= 8
        and 1 <= block_n // (n_waves * mma_n) <= 8
    ):
        raise ValueError("each wave must compute 1..8 WMMA tiles along M and N")
    if block_m % waves or block_n % waves:
        raise ValueError("each wave must own a whole number of A and B rows")
    elem_bits = GEMM_DTYPE_BITS.get(dtype_id)
    if elem_bits is None or STORE_VEC_BITS % elem_bits:
        dtype_name = GEMM_DTYPE_NAMES.get(dtype_id, "unknown")
        raise ValueError(f"unsupported element size for {dtype_name}")
    _validate_interleaved_tile(
        block_m,
        block_n,
        m_waves,
        n_waves,
        mma_m,
        mma_n,
        mma_k,
        elem_bits,
        b_k_contiguous,
        interleave,
    )
    out_bits = elem_bits
    if STORE_VEC_BITS % out_bits:
        raise ValueError("the output tile must support whole-workgroup 16-byte stores")
    out_vec_elems = STORE_VEC_BITS // out_bits
    threads = waves * WAVE_SIZE
    if (
        block_n % out_vec_elems
        or threads % (block_n // out_vec_elems)
        or block_m * block_n % (threads * out_vec_elems)
    ):
        raise ValueError("the output tile must support whole-workgroup 16-byte stores")
    lds_pad = LDS_PAD * 16 // elem_bits
    a_stage_size = block_m * (block_k + lds_pad)
    b_stage_size = (
        block_n * (block_k + lds_pad)
        if b_k_contiguous
        else block_k * (block_n + lds_pad)
    )
    ab_bits = stages * (a_stage_size + b_stage_size) * elem_bits
    c_pitch = block_n + (STORE_VEC_BITS // out_bits if interleave else 0)
    c_bits = block_m * c_pitch * out_bits
    shared_bytes = (max(ab_bits, c_bits) + 7) // 8
    if shared_bytes > 320 * 1024:
        raise ValueError("pipeline exceeds gfx1250's 320 KiB LDS capacity")
    if split_k not in range(1, 10):
        raise ValueError("cluster split_k must be in 1..9")
    if k % (split_k * block_k):
        raise ValueError("K must be divisible by split_k * block_k")
    if k // split_k < (stages - 1) * block_k:
        raise ValueError("prefetch needs at least stages - 1 tiles in each K slice")
    n_blocks = (n + block_n - 1) // block_n
    if group_m * n_blocks > (2**32 - 1) // 2:
        raise ValueError("block swizzle offsets exceed the supported 32-bit range")


def make_gemm_a16w16_gfx1250_kernel_name(
    dtype_id,
    block_m,
    block_n,
    block_k,
    stages,
    m_waves,
    n_waves,
    group_m,
    has_bias,
    b_k_contiguous,
    mma_m,
    mma_n,
    mma_k,
    split_k,
    interleave=False,
    split_k_atomic=False,
):
    dtype_name = GEMM_DTYPE_NAMES[dtype_id]
    layout = "nt" if b_k_contiguous else "nn"
    return f"hgemm_{dtype_name}_t{block_m}x{block_n}x{block_k}x{stages}_w{m_waves}x{n_waves}_gm{group_m}_bias{int(has_bias)}_l{layout}_sk{split_k}_mma{mma_m}x{mma_n}x{mma_k}{('_interleave' if interleave else '')}{('_atomic' if split_k_atomic else '')}"


@functools.lru_cache(maxsize=128)
def build_gemm_gfx1250(
    dtype_id: int,
    block_m: int,
    block_n: int,
    block_k: int,
    stages: int,
    m_waves: int,
    n_waves: int,
    group_m: int,
    has_bias: bool,
    b_k_contiguous: bool,
    mma_m: int,
    mma_n: int,
    mma_k: int,
    split_k: int = 1,
    interleave: bool = False,
    split_k_atomic: bool = False,
    device_id: int = 0,
):
    # device_id participates in this cached factory's identity: a compiled
    # HIP module cannot be reused by a different device in the same process.
    elem_dtype = {GEMM_DTYPE_FP16: fx.Float16, GEMM_DTYPE_BF16: fx.BFloat16}.get(
        dtype_id
    )
    out_dtype = elem_dtype
    if elem_dtype is None:
        name = GEMM_DTYPE_NAMES.get(dtype_id, "unknown")
        raise ValueError(f"unsupported dtype_id={dtype_id} ({name})")
    elem_bits = GEMM_DTYPE_BITS[dtype_id]
    lds_pad = LDS_PAD * 16 // elem_bits
    out_bits = elem_bits
    out_vec_elems = STORE_VEC_BITS // out_bits
    frag_elems = mma_m * mma_k // WAVE_SIZE
    frag_load_elems = FRAGMENT_LOAD_BITS // elem_bits
    frag_loads = frag_elems // frag_load_elems
    frag_k_stride = 2 * frag_load_elems
    acc_elems = mma_m * mma_n // WAVE_SIZE
    waves = m_waves * n_waves
    threads = waves * WAVE_SIZE
    a_pitch = block_k + lds_pad
    a_stage_size = block_m * a_pitch
    a_tile_stride = (a_pitch, 1)
    if b_k_contiguous:
        b_pitch = block_k + lds_pad
        b_stage_size = block_n * b_pitch
        b_tile_stride = (1, b_pitch)
    else:
        b_pitch = block_n + lds_pad
        b_stage_size = block_k * b_pitch
        b_tile_stride = (b_pitch, 1)
    c_pitch = block_n + (STORE_VEC_BITS // out_bits if interleave else 0)
    c_slots = block_m * c_pitch * out_bits // elem_bits
    shared_elements = max(stages * (a_stage_size + b_stage_size), c_slots)
    m_steps, n_steps = (block_m // (m_waves * mma_m), block_n // (n_waves * mma_n))
    half_m_steps, half_n_steps = (m_steps // 2, n_steps // 2)
    _validate_interleaved_tile(
        block_m,
        block_n,
        m_waves,
        n_waves,
        mma_m,
        mma_n,
        mma_k,
        elem_bits,
        b_k_contiguous,
        interleave,
    )

    @fx.struct
    class SharedStorage:
        data: fx.Array[elem_dtype, shared_elements, 16]

    @flyc.kernel
    def kernel(
        a: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        bias: fx.Pointer,
        workspace: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        a_leading_stride: fx.Int32,
        b_leading_stride: fx.Int32,
        c_leading_stride: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()
        tid = fx.thread_idx.x
        wave = rocdl.wave_id()
        lane = tid % WAVE_SIZE
        block_id = fx.block_idx.x
        m_blocks = (m + block_m - 1) // block_m
        n_blocks = (n + block_n - 1) // block_n
        block_swizzle = BlockSwizzle(
            NUM_XCDS=NUM_XCDS, NUM_PIDS_THRESHOLD=NUM_PIDS_THRESHOLD, GROUP_M=group_m
        )
        block_row, block_col = block_swizzle.swizzle(m_blocks, n_blocks, block_id)
        row_start, col_start = (block_row * block_m, block_col * block_n)
        wave_row = wave // n_waves * (block_m // m_waves)
        wave_col = wave % n_waves * (block_n // n_waves)
        storage = fx.SharedAllocator(static=True).allocate(SharedStorage).peek()
        shared_a = storage.data.ptr
        shared_b = shared_a + stages * a_stage_size
        shared_out = shared_a

        def tdm_operand(
            global_pointer,
            global_shape,
            global_stride,
            tile_shape,
            tile_stride,
            store=False,
        ):
            global_view = fx.make_view(
                global_pointer, fx.make_layout(global_shape, global_stride)
            )
            atom, coords = rocdl.cdna5.make_tiled_tdm_atom(
                rocdl.TensorStore() if store else rocdl.TensorLoad(),
                global_view,
                fx.make_layout(tile_shape, tile_stride),
                tile_shape,
                num_warps=waves,
            )
            return (atom, fx.zipped_divide(coords, tile_shape))

        a_global_stride = (a_leading_stride, 1)
        if const_expr(b_k_contiguous):
            b_global_stride = (1, b_leading_stride)
        else:
            b_global_stride = (b_leading_stride, 1)
        atom_a, coords_a = tdm_operand(
            a, (m, k), a_global_stride, (block_m, block_k), a_tile_stride
        )
        atom_b, coords_b = tdm_operand(
            b, (k, n), b_global_stride, (block_k, block_n), b_tile_stride
        )

        def copy_operand(
            atom,
            shared_pointer,
            tile_shape,
            tile_stride,
            stage_size,
            dst_stage,
            src_stage,
            coord,
        ):
            shared_view = fx.make_view(
                shared_pointer,
                fx.make_layout((tile_shape, stages), (tile_stride, stage_size)),
            )
            dst, src = rocdl.cdna5.tdm_partition(
                atom, wave, fx.make_layout(waves, 1), shared_view, coord
            )
            fx.copy(atom, src[None, src_stage], dst[None, dst_stage])

        def copy_tile(dst_stage, src_stage):
            copy_operand(
                atom_b,
                shared_b,
                (block_k, block_n),
                b_tile_stride,
                b_stage_size,
                dst_stage,
                src_stage,
                coords_b[None, (None, block_col)],
            )
            copy_operand(
                atom_a,
                shared_a,
                (block_m, block_k),
                a_tile_stride,
                a_stage_size,
                dst_stage,
                src_stage,
                coords_a[None, (block_row, None)],
            )
            rocdl.sched_barrier(0)

        def wait_tile(pending_stages):
            tdm_ops.tensor_wait(pending_stages * 2)
            fx.gpu.barrier()

        mma = fx.make_mma_atom(rocdl.WMMA(mma_m, mma_n, mma_k, elem_dtype, fx.Float32))
        accumulators = [
            fx.make_rmem_tensor(acc_elems, fx.Float32)
            for _ in range_constexpr(m_steps * n_steps)
        ]
        for acc in accumulators:
            acc.store(fx.full(acc_elems, 0.0, fx.Float32))

        def load_fragment(
            shared_pointer, stage_size, stage, row, column, k_contiguous=True
        ):
            lane_k = lane // 16 * frag_load_elems

            def fragment_k_offset(i):
                return (
                    lane_k + i // frag_load_elems * frag_k_stride + i % frag_load_elems
                )

            if const_expr(k_contiguous):
                base = (
                    stage * stage_size
                    + (row + lane % 16) * (block_k + lds_pad)
                    + column
                )
                chunks = [
                    fx.ptr_load(
                        shared_pointer
                        + base
                        + fragment_k_offset(group * frag_load_elems),
                        result_type=T.vec(frag_load_elems, elem_dtype.ir_type),
                    )
                    for group in range_constexpr(frag_loads)
                ]
                values = fx.Vector.from_elements(
                    [
                        chunks[i // frag_load_elems][i % frag_load_elems]
                        for i in range_constexpr(frag_elems)
                    ]
                )
            else:
                base = stage * stage_size + row + lane % 16
                values = fx.Vector.from_elements(
                    [
                        fx.ptr_load(
                            shared_pointer
                            + base
                            + (column + fragment_k_offset(i)) * b_pitch
                        )
                        for i in range_constexpr(frag_elems)
                    ]
                )
            fragment = fx.make_rmem_tensor(frag_elems, elem_dtype)
            fragment.store(values)
            return fragment

        def compute_tile(stage):
            for ki in range_constexpr(block_k // mma_k):
                b_fragments = [
                    load_fragment(
                        shared_b,
                        b_stage_size,
                        stage,
                        wave_col + ni * mma_n,
                        ki * mma_k,
                        k_contiguous=b_k_contiguous,
                    )
                    for ni in range_constexpr(n_steps)
                ]
                rocdl.sched_barrier(0)
                for mi in range_constexpr(m_steps):
                    a_fragment = load_fragment(
                        shared_a, a_stage_size, stage, wave_row + mi * mma_m, ki * mma_k
                    )
                    rocdl.sched_barrier(0)
                    for ni in range_constexpr(n_steps):
                        acc = accumulators[mi * n_steps + ni]
                        fx.gemm(mma, acc, a_fragment, b_fragments[ni], acc)

        k_rank = fx.block_idx.y
        k_tiles = k // split_k // block_k
        k_tile_base = k_rank * k_tiles
        if const_expr(split_k_atomic):
            if k_rank == 0:
                for i in range_constexpr(
                    block_m * block_n // (threads * out_vec_elems)
                ):
                    offset = (i * threads + tid) * out_vec_elems
                    row, col = (offset // block_n, offset % block_n)
                    destination = (
                        out
                        + fx.Int64(row_start + row) * fx.Int64(c_leading_stride)
                        + col_start
                        + col
                    )
                    if row_start + block_m <= m and col_start + block_n <= n:
                        fx.make_view(
                            destination, fx.make_layout(out_vec_elems, 1)
                        ).store(fx.full(out_vec_elems, 0.0, out_dtype))
                    else:
                        for pair in range_constexpr(out_vec_elems // 2):
                            if (
                                row_start + row < m
                                and col_start + col + pair * 2 + 1 < n
                            ):
                                fx.make_view(
                                    destination + pair * 2, fx.make_layout(2, 1)
                                ).store(fx.full(2, 0.0, out_dtype))
            llvm.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
            cluster_barrier()
            llvm.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
        if const_expr(interleave):
            fragment_words = frag_elems * elem_bits // 32
            quadrant_steps = half_m_steps * half_n_steps
            wmma_k_steps = block_k // mma_k
            load_atom = fx.make_copy_atom(
                fx.UniversalCopy(FRAGMENT_LOAD_BITS), fx.Int32
            )
            load_layout = fx.make_layout(FRAGMENT_LOAD_BITS // 32, 1)
            aligned_lds_type = fx.PointerType.get(
                elem_ty=fx.Int32.ir_type,
                address_space=fx.AddressSpace.Shared,
                alignment=16,
            )

            def fragment_pointer(operand, stage, k_step):
                if const_expr(operand == "a"):
                    pointer, stage_size, wave_start, pitch = (
                        shared_a,
                        a_stage_size,
                        wave_row,
                        a_pitch,
                    )
                else:
                    pointer, stage_size, wave_start, pitch = (
                        shared_b,
                        b_stage_size,
                        wave_col,
                        b_pitch,
                    )
                pointer = (
                    pointer
                    + stage * stage_size
                    + (wave_start + lane % 16) * pitch
                    + k_step * mma_k
                    + lane // 16 * frag_load_elems
                )
                return (pointer, mma_m * pitch)

            def load_fragment_part(pointer, element_offset):
                address = fx.Int32(fx.ptrtoint(pointer)) + fx.Int32(element_offset) * (
                    elem_bits // 8
                )
                source = fx.inttoptr(aligned_lds_type, address)
                values = fx.make_rmem_tensor(load_layout, fx.Int32)
                fx.copy(load_atom, fx.make_view(source, load_layout), values)
                return values.load()

            def join_fragment_parts(parts):
                values = parts[0].shuffle(parts[1], list(range(8)))
                if const_expr(frag_loads == 4):
                    values = values.shuffle(
                        parts[2].shuffle(parts[3], list(range(8))), list(range(16))
                    )
                return values

            def load_packed_fragment(pointer, element_offset):
                return join_fragment_parts(
                    [
                        load_fragment_part(
                            pointer, element_offset + part * frag_k_stride
                        )
                        for part in range_constexpr(frag_loads)
                    ]
                )

            a_first_half = [
                fx.make_rmem_tensor(fragment_words, fx.Int32)
                for _ in range_constexpr(half_m_steps)
            ]
            b_first_half = [
                fx.make_rmem_tensor(fragment_words, fx.Int32)
                for _ in range_constexpr(half_n_steps)
            ]

            def first_half_loads(stage, k_step, parity):
                a_pointer, a_row_stride = fragment_pointer("a", stage, k_step)
                b_pointer, b_row_stride = fragment_pointer("b", stage, k_step)
                a_loads = [
                    lambda index=index: a_first_half[index].store(
                        load_packed_fragment(a_pointer, index * a_row_stride)
                    )
                    for index in range_constexpr(half_m_steps)
                ]
                b_loads = [
                    lambda index=index: b_first_half[index].store(
                        load_packed_fragment(b_pointer, index * b_row_stride)
                    )
                    for index in range_constexpr(half_n_steps)
                ]
                return (a_loads, b_loads) if parity == 0 else (b_loads, a_loads)

            def second_half_loads(operand, stage, k_step):
                count = half_m_steps if operand == "a" else half_n_steps
                pointer, row_stride = fragment_pointer(operand, stage, k_step)
                fragments = [None] * count
                parts = [[] for _ in range_constexpr(count)]

                def load_part(index, part):
                    parts[index].append(
                        load_fragment_part(
                            pointer, (count + index) * row_stride + part * frag_k_stride
                        )
                    )
                    if const_expr(part == frag_loads - 1):
                        fragment = fx.make_rmem_tensor(fragment_words, fx.Int32)
                        fragment.store(join_fragment_parts(parts[index]))
                        fragments[index] = fragment

                loads = {
                    index * frag_loads + part: lambda index=index, part=part: load_part(
                        index, part
                    )
                    for index in range(count)
                    for part in range(frag_loads)
                }
                return (loads, fragments)

            def compute_quadrant(
                m_step_start,
                n_step_start,
                a_fragments,
                b_fragments,
                loads,
                n_fast,
                before=None,
            ):
                if const_expr(before is not None):
                    before()
                for step in range_constexpr(quadrant_steps):
                    if const_expr(n_fast):
                        m_index, n_index = (step // half_n_steps, step % half_n_steps)
                    else:
                        m_index, n_index = (step % half_m_steps, step // half_m_steps)
                    acc = accumulators[
                        (m_step_start + m_index) * n_steps + n_step_start + n_index
                    ]
                    fx.gemm(mma, acc, b_fragments[n_index], a_fragments[m_index], acc)
                    if const_expr(step in loads):
                        loads[step]()
                    if const_expr(step == max(loads, default=-1)):
                        for _ in range_constexpr(len(loads)):
                            rocdl.sched_mfma(1)
                            rocdl.sched_dsrd(frag_loads // 2)
                        rocdl.sched_barrier(0)

            def compute_interleaved_tile(
                k_tile, has_next_tile, parity, read_stage=None
            ):
                read_stage = k_tile % stages if read_stage is None else read_stage
                for k_step in range_constexpr(wmma_k_steps):
                    at_boundary = k_step == wmma_k_steps - 1
                    next_stage = (
                        (1 - read_stage if stages == 2 else (k_tile + 1) % stages)
                        if at_boundary
                        else read_stage
                    )
                    next_k_step = 0 if at_boundary else k_step + 1

                    def signal_next_stage(at_boundary=at_boundary):
                        rocdl.sched_barrier(0)
                        rocdl.s_wait_dscnt(0)
                        if const_expr(at_boundary and has_next_tile):
                            tdm_ops.tensor_wait(2 * (stages - 2))
                            rocdl.s_barrier_signal(-1)
                        rocdl.sched_barrier(0)

                    def refill_stage(at_boundary=at_boundary):
                        if const_expr(at_boundary and has_next_tile):
                            rocdl.s_barrier_wait(-1)
                            copy_tile(
                                read_stage,
                                fx.min(k_tile_base + k_tile + stages, last_tile),
                            )
                        rocdl.sched_barrier(0)

                    has_next_step = not at_boundary or has_next_tile
                    early_loads, late_loads = (
                        first_half_loads(next_stage, next_k_step, parity)
                        if has_next_step
                        else ([], [])
                    )
                    refill_step = quadrant_steps // 4
                    early_load_slots = quadrant_steps - refill_step
                    third_quadrant_loads = {refill_step - 1: refill_stage}
                    third_quadrant_loads.update(
                        {
                            refill_step + index: load
                            for index, load in enumerate(early_loads[:early_load_slots])
                        }
                    )
                    last_quadrant_loads = dict(
                        enumerate(early_loads[early_load_slots:])
                    )
                    last_quadrant_loads.update(
                        {
                            quadrant_steps - len(late_loads) + index: load
                            for index, load in enumerate(late_loads)
                        }
                    )
                    if parity == 0:
                        loads, b_second_half = second_half_loads(
                            "b", read_stage, k_step
                        )
                        compute_quadrant(0, 0, a_first_half, b_first_half, loads, False)
                        loads, a_second_half = second_half_loads(
                            "a", read_stage, k_step
                        )
                        compute_quadrant(
                            0, half_n_steps, a_first_half, b_second_half, loads, False
                        )
                        compute_quadrant(
                            half_m_steps,
                            half_n_steps,
                            a_second_half,
                            b_second_half,
                            third_quadrant_loads,
                            True,
                            signal_next_stage,
                        )
                        compute_quadrant(
                            half_m_steps,
                            0,
                            a_second_half,
                            b_first_half,
                            last_quadrant_loads,
                            True,
                        )
                    else:
                        loads, a_second_half = second_half_loads(
                            "a", read_stage, k_step
                        )
                        compute_quadrant(0, 0, a_first_half, b_first_half, loads, True)
                        loads, b_second_half = second_half_loads(
                            "b", read_stage, k_step
                        )
                        compute_quadrant(
                            half_m_steps, 0, a_second_half, b_first_half, loads, True
                        )
                        compute_quadrant(
                            half_m_steps,
                            half_n_steps,
                            a_second_half,
                            b_second_half,
                            third_quadrant_loads,
                            False,
                            signal_next_stage,
                        )
                        compute_quadrant(
                            0,
                            half_n_steps,
                            a_first_half,
                            b_second_half,
                            last_quadrant_loads,
                            False,
                        )

            last_tile = k_tile_base + k_tiles - 1
            for stage in range_constexpr(stages):
                copy_tile(fx.Int32(stage), fx.min(k_tile_base + stage, last_tile))
            tdm_ops.tensor_wait(2 * (stages - 1))
            fx.gpu.barrier()
            for loads in first_half_loads(fx.Int32(0), 0, 0):
                for load in loads:
                    load()

            def run_interleaved(parity):
                for pair in range((k_tiles - 1) // 2):
                    for step in range_constexpr(2):
                        compute_interleaved_tile(
                            pair * 2 + step, True, parity, step if stages == 2 else None
                        )
                for remaining in range((k_tiles - 1) % 2):
                    compute_interleaved_tile(k_tiles - 2, True, parity)
                compute_interleaved_tile(k_tiles - 1, False, parity)

            # A uniform scalar branch is required by the two wave schedules;
            # the pinned typed numeric surface has no readfirstlane method.
            wave_parity = fx.Int32(rocdl.readfirstlane(T.i32, wave % 2))
            if wave_parity == 0:
                run_interleaved(0)
            else:
                run_interleaved(1)
            tdm_ops.tensor_wait(0)
        else:
            for stage in range_constexpr(stages - 1):
                copy_tile(stage, k_tile_base + fx.Int32(stage))
            main_tiles = k_tiles - (stages - 1)
            current_stage = fx.Int32(0)
            for pair in range(main_tiles // 2):
                for step in range_constexpr(2):
                    read_stage = step if stages == 2 else current_stage
                    wait_tile(stages - 2)
                    copy_tile(
                        (read_stage + stages - 1) % stages,
                        k_tile_base + pair * 2 + step + stages - 1,
                    )
                    compute_tile(read_stage)
                    current_stage = (current_stage + 1) % stages
            for remaining in range(main_tiles % 2):
                wait_tile(stages - 2)
                copy_tile(
                    (current_stage + stages - 1) % stages,
                    k_tile_base + main_tiles + stages - 2,
                )
                compute_tile(current_stage)
                current_stage = (current_stage + 1) % stages
            for tail in range_constexpr(stages - 1):
                wait_tile(stages - 2 - tail)
                compute_tile(current_stage)
                current_stage = (current_stage + 1) % stages
        fx.gpu.barrier()
        if const_expr(interleave):
            if const_expr(has_bias and split_k == 1):
                bias_fragments = [
                    fx.Vector.from_elements(
                        [
                            fx.ptr_load(
                                bias
                                + fx.min(
                                    col_start
                                    + wave_col
                                    + ni * mma_n
                                    + lane // 16 * acc_elems
                                    + i,
                                    n - 1,
                                ),
                                result_type=out_dtype.ir_type,
                            )
                            for i in range_constexpr(acc_elems)
                        ]
                    )
                    for ni in range_constexpr(n_steps)
                ]
            for mi in range_constexpr(m_steps):
                for ni in range_constexpr(n_steps):
                    row = wave_row + mi * mma_m + lane % 16
                    col = wave_col + ni * mma_n + lane // 16 * acc_elems
                    values = accumulators[mi * n_steps + ni].load()
                    values = values.to(out_dtype)
                    if const_expr(has_bias and split_k == 1):
                        values = (
                            values.to(fx.Float32) + bias_fragments[ni].to(fx.Float32)
                        ).to(out_dtype)
                    fx.make_view(
                        shared_out + row * c_pitch + col, fx.make_layout(acc_elems, 1)
                    ).store(values)
        else:
            for mi in range_constexpr(m_steps):
                for ni in range_constexpr(n_steps):
                    row = wave_row + mi * mma_m + lane // 16 * acc_elems
                    col = wave_col + ni * mma_n + lane % 16
                    values = accumulators[mi * n_steps + ni].load()
                    values = values.to(out_dtype)
                    for i in range_constexpr(acc_elems):
                        fx.ptr_store(values[i], shared_out + (row + i) * block_n + col)
        fx.gpu.barrier()

        def commit_output(values, row, col):
            destination = (
                out
                + fx.Int64(row_start + row) * fx.Int64(c_leading_stride)
                + col_start
                + col
            )
            if row_start + block_m <= m and col_start + block_n <= n:
                output_values = values.to(out_dtype)
                if const_expr(has_bias):
                    bias_values = fx.ptr_load(
                        bias + col_start + col,
                        result_type=T.vec(out_vec_elems, out_dtype.ir_type),
                    )
                    output_values = (
                        output_values.to(fx.Float32) + bias_values.to(fx.Float32)
                    ).to(out_dtype)
                fx.make_view(destination, fx.make_layout(out_vec_elems, 1)).store(
                    output_values
                )
            else:
                for element in range_constexpr(out_vec_elems):
                    if row_start + row < m and col_start + col + element < n:
                        value = values[element]
                        value = value.to(out_dtype)
                        if const_expr(has_bias):
                            bias_value = fx.ptr_load(
                                bias + col_start + col + element,
                                result_type=out_dtype.ir_type,
                            )
                            value = (
                                value.to(fx.Float32) + bias_value.to(fx.Float32)
                            ).to(out_dtype)
                        fx.ptr_store(value, destination + element)

        tile_elems = block_m * block_n
        if const_expr(split_k_atomic):
            output_view = fx.make_view(
                out, fx.make_layout((m, n), (c_leading_stride, 1))
            )
            # Each pair is explicitly M/N-masked below; the host bounds the
            # complete strided byte footprint to the buffer offset range.
            output_buffer = rocdl.make_buffer_tensor(output_view)
            atomic_atom = fx.make_copy_atom(
                rocdl.BufferAtomicPkAdd(out_dtype), out_dtype
            )
            pair_layout = fx.make_layout(2, 1)
            pair_fragment = fx.make_rmem_tensor(pair_layout, out_dtype)
            for i in range_constexpr(tile_elems // (threads * out_vec_elems)):
                offset = (i * threads + tid) * out_vec_elems
                row, col = (offset // block_n, offset % block_n)
                values = fx.ptr_load(
                    shared_out + row * c_pitch + col,
                    result_type=T.vec(out_vec_elems, out_dtype.ir_type),
                )
                for pair in range_constexpr(out_vec_elems // 2):
                    pair_col = col_start + col + pair * 2
                    if row_start + row < m and pair_col + 1 < n:
                        pair_fragment.store(
                            fx.Vector.from_elements(
                                [values[pair * 2], values[pair * 2 + 1]], out_dtype
                            )
                        )
                        destination = fx.make_view(
                            fx.get_iter(output_buffer)
                            + (row_start + row) * c_leading_stride
                            + pair_col,
                            pair_layout,
                        )
                        fx.copy(atomic_atom, pair_fragment, destination)
            llvm.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
            if const_expr(has_bias):
                cluster_barrier()
                llvm.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
                if k_rank == 0:
                    for i in range_constexpr(tile_elems // (threads * out_vec_elems)):
                        offset = (i * threads + tid) * out_vec_elems
                        row, col = (offset // block_n, offset % block_n)
                        destination = (
                            out
                            + fx.Int64(row_start + row) * fx.Int64(c_leading_stride)
                            + col_start
                            + col
                        )
                        if row_start + block_m <= m and col_start + block_n <= n:
                            values = fx.ptr_load(
                                destination,
                                result_type=T.vec(out_vec_elems, out_dtype.ir_type),
                            )
                            bias_values = fx.ptr_load(
                                bias + col_start + col,
                                result_type=T.vec(out_vec_elems, out_dtype.ir_type),
                            )
                            values = (
                                values.to(fx.Float32) + bias_values.to(fx.Float32)
                            ).to(out_dtype)
                            fx.make_view(
                                destination, fx.make_layout(out_vec_elems, 1)
                            ).store(values)
                        else:
                            for pair in range_constexpr(out_vec_elems // 2):
                                if (
                                    row_start + row < m
                                    and col_start + col + pair * 2 + 1 < n
                                ):
                                    values = fx.ptr_load(
                                        destination + pair * 2,
                                        result_type=T.vec(2, out_dtype.ir_type),
                                    )
                                    bias_values = fx.ptr_load(
                                        bias + col_start + col + pair * 2,
                                        result_type=T.vec(2, out_dtype.ir_type),
                                    )
                                    values = (
                                        values.to(fx.Float32)
                                        + bias_values.to(fx.Float32)
                                    ).to(out_dtype)
                                    fx.make_view(
                                        destination + pair * 2, pair_layout
                                    ).store(values)
        elif const_expr(interleave and split_k == 1):
            c_tile = (block_m, c_pitch)
            atom_c, coords_c = tdm_operand(
                out + fx.Int64(row_start) * fx.Int64(c_leading_stride) + col_start,
                (
                    fx.max(0, fx.min(block_m, m - row_start)),
                    fx.max(0, fx.min(block_n, n - col_start)),
                ),
                (c_leading_stride, 1),
                c_tile,
                (c_pitch, 1),
                store=True,
            )
            shared_c = fx.make_view(
                shared_out,
                fx.make_layout((c_tile, 1), ((c_pitch, 1), block_m * c_pitch)),
            )
            src, dst = rocdl.cdna5.tdm_partition(
                atom_c,
                wave,
                fx.make_layout(waves, 1),
                shared_c,
                coords_c[None, (0, None)],
            )
            fx.copy(atom_c, src[None, 0], dst[None, 0])
            tdm_ops.tensor_wait(0)
        elif const_expr(split_k == 1):
            for i in range_constexpr(tile_elems // (threads * out_vec_elems)):
                offset = (i * threads + tid) * out_vec_elems
                row, col = (offset // block_n, offset % block_n)
                values = fx.ptr_load(
                    shared_out + row * c_pitch + col,
                    result_type=T.vec(out_vec_elems, out_dtype.ir_type),
                )
                commit_output(values, row, col)
        else:
            for i in range_constexpr(tile_elems // (threads * out_vec_elems)):
                offset = (i * threads + tid) * out_vec_elems
                row, col = (offset // block_n, offset % block_n)
                values = fx.ptr_load(
                    shared_out + row * c_pitch + col,
                    result_type=T.vec(out_vec_elems, out_dtype.ir_type),
                )
                partial = (
                    workspace
                    + fx.Int64(block_id) * (split_k * tile_elems)
                    + fx.Int64(k_rank) * tile_elems
                    + row * block_n
                    + col
                )
                fx.make_view(partial, fx.make_layout(out_vec_elems, 1)).store(values)
            # Publish same-dtype partials before another workgroup reads them.
            llvm.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)
            cluster_barrier()
            llvm.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Acquire)
            if k_rank == 0:
                for i in range_constexpr(tile_elems // (threads * out_vec_elems)):
                    offset = (i * threads + tid) * out_vec_elems
                    row, col = (offset // block_n, offset % block_n)
                    total = fx.ptr_load(
                        workspace
                        + fx.Int64(block_id) * (split_k * tile_elems)
                        + offset,
                        result_type=T.vec(out_vec_elems, out_dtype.ir_type),
                    ).to(fx.Float32)
                    for part in range_constexpr(split_k - 1):
                        partial = fx.ptr_load(
                            workspace
                            + fx.Int64(block_id) * (split_k * tile_elems)
                            + (part + 1) * tile_elems
                            + offset,
                            result_type=T.vec(out_vec_elems, out_dtype.ir_type),
                        )
                        total = total + partial.to(fx.Float32)
                    commit_output(total, row, col)

    @flyc.jit
    def launch(
        a: fx.Pointer,
        b: fx.Pointer,
        out: fx.Pointer,
        bias: fx.Pointer,
        workspace: fx.Pointer,
        m: fx.Int32,
        n: fx.Int32,
        k: fx.Int32,
        a_leading_stride: fx.Int32,
        b_leading_stride: fx.Int32,
        c_leading_stride: fx.Int32,
        stream: fx.Stream,
    ):
        mn_tiles = (m + block_m - 1) // block_m * ((n + block_n - 1) // block_n)
        if split_k == 1:
            kernel(
                a,
                b,
                out,
                bias,
                workspace,
                m,
                n,
                k,
                a_leading_stride,
                b_leading_stride,
                c_leading_stride,
            ).launch(grid=(mn_tiles, 1, 1), block=(threads, 1, 1), stream=stream)
        else:
            kernel(
                a,
                b,
                out,
                bias,
                workspace,
                m,
                n,
                k,
                a_leading_stride,
                b_leading_stride,
                c_leading_stride,
                value_attrs={"rocdl.cluster_dims": f"1,{split_k},1"},
            ).launch(
                grid=(mn_tiles, split_k, 1),
                block=(threads, 1, 1),
                stream=stream,
                cluster=(1, split_k, 1),
            )

    kernel._func.__name__ = make_gemm_a16w16_gfx1250_kernel_name(
        dtype_id,
        block_m,
        block_n,
        block_k,
        stages,
        m_waves,
        n_waves,
        group_m,
        has_bias,
        b_k_contiguous,
        mma_m,
        mma_n,
        mma_k,
        split_k,
        interleave,
        split_k_atomic,
    )
    if interleave:
        launch.compile_hints["llvm_options"] = {
            "amdgpu-expert-scheduling-mode": True,
            "amdgpu-sched-strategy": "coexec",
            "enable-post-misched": True,
            "amdgpu-barrier-signal-wait-latency": 32,
        }
    return launch

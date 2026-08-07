# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""MXFP4/6/8 by preshuffled MXFP4/8 GEMM for gfx950 scaled MFMA."""

import flydsl.compiler as flyc
import flydsl.expr as fx
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

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.tensor_shim import ptr_rsrc

_A_ELEM = {"fp4": Float4E2M1FN, "fp6": Float6E2M3FN, "fp8": Float8E4M3FN}
_B_ELEM = {"fp4": Float4E2M1FN, "fp8": Float8E4M3FN}


def _scale_mma_atoms(a_dtype, b_dtype):
    elem_a = _A_ELEM[a_dtype]
    elem_b = _B_ELEM[b_dtype]
    return {
        (opsel_a, opsel_b): fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(
                16,
                16,
                128,
                elem_a,
                elem_b,
                opsel_a=opsel_a,
                opsel_b=opsel_b,
            )
        )
        for opsel_a in range(4)
        for opsel_b in range(4)
    }


def _bq_view(arg_bq_addr, row_elems, KH4, k_tiles, k_halves, pair):
    """Build one preshuffled-B N-row view in i32 units."""
    col_base = rocdl.readfirstlane(T.i32, row_elems * KH4)
    i32_ptr_type = fx.PointerType.get(
        T.i32, address_space=fx.AddressSpace.Global, alignment=16
    )
    base_iter = fx.inttoptr(
        i32_ptr_type,
        arg_bq_addr + fx.Int64(col_base) * fx.Int64(4),
    )
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
    """Launch one statically configured MX-scale GEMM."""
    BM, BN, BK = tile_m, tile_n, tile_k
    out_elem = BFloat16 if const_expr(out_dtype == "bf16") else Float16

    if const_expr(a_dtype == "fp4"):
        a_row_bytes, A_ROW_B = K // 2, BK // 2
        A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 16, 0, 4
    else:
        a_row_bytes, A_ROW_B = K, BK
        if const_expr(a_dtype == "fp8"):
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 4, 32, 16, 8
        else:
            A_GK_I32, A_KH_I32, A_HI_OFF, A_NDW = 8, 32, 4, 6

    A_LDS_B = BM * A_ROW_B
    A_ROW_I32 = A_ROW_B // 4
    swizzle_lds = a_dtype in ("fp4", "fp8")
    k_blocks16 = A_ROW_B // 16
    if const_expr(b_dtype == "fp8"):
        b_row_bytes, B_NDW, B_BLK_PER_MMA = K, 8, 2
    else:
        b_row_bytes, B_NDW, B_BLK_PER_MMA = K // 2, 4, 1
    KH4 = b_row_bytes // 4
    K_TILES = K // BK
    if K_TILES % k_batch != 0:
        raise ValueError("K tiles must divide evenly across split-K")
    k_tiles_local = K_TILES // k_batch
    k_halves = BK // 128
    tiles_per_chunk = 256 // BK
    m_chunks = BM // 16
    num_waves = min(4, BN // 16)
    num_threads = num_waves * 64
    num_acc_n = (BN // num_waves) // 16
    scale_chunk_dwords = ((K + 255) // 256) * 64
    scale_k0_dwords = 64
    cooperative_loads = A_LDS_B // num_threads // 16
    n_pairs = max(1, num_acc_n // 2)
    m_pairs = max(1, m_chunks // 2)

    scheduled_mfmas = k_halves * m_chunks * num_acc_n
    a_ds_per = 1 if const_expr(a_dtype == "fp4") else 2
    scheduled_ds_loads = m_chunks * k_halves * a_ds_per
    scheduled_gmem = (
        cooperative_loads
        + num_acc_n * k_halves * B_BLK_PER_MMA
        + m_pairs
        + n_pairs
    )

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
        if const_expr(k_batch > 1):
            batch_index = bid_z // k_batch
            k_tile_start = fx.Int32(bid_z % k_batch) * fx.Int32(k_tiles_local)
        else:
            batch_index = bid_z
            k_tile_start = fx.Int32(0)
        wave = rocdl.readfirstlane(T.i32, tid // 64)
        lane = tid % 64
        lane_div_16 = lane // 16
        lane_mod_16 = lane % 16

        if const_expr(xcd_swizzle > 0):
            from .mfma_preshuffle_pipeline import xcd_remap_bx_by

            block_m, block_n = xcd_remap_bx_by(
                fx.Index(bid_x),
                fx.Index(bid_y),
                fx.Index(i32_m),
                tile_m=BM,
                tile_n=BN,
                N=N,
                xcd_swizzle=xcd_swizzle,
            )
            block_m = fx.Int32(block_m) * BM
            block_n = fx.Int32(block_n) * BN
        else:
            block_m = bid_x * BM
            block_n = bid_y * BN

        if const_expr(batch > 1):
            a_rstride = fx.Int32(
                a_row_bytes if a_row_stride < 0 else a_row_stride
            )
            scale_a_rstride = fx.Int32(
                scale_chunk_dwords if sca_row_stride < 0 else sca_row_stride
            )
            batch_i64 = fx.Int64(batch_index)
            if const_expr(a_batch_stride < 0):
                arg_a = arg_a + batch_i64 * (
                    fx.Int64(i32_m) * fx.Int64(a_row_bytes)
                )
            else:
                arg_a = arg_a + batch_i64 * fx.Int64(a_batch_stride)
            arg_b = arg_b + batch_i64 * fx.Int64(N * b_row_bytes)
            if const_expr(sca_batch_stride < 0):
                scale_batch_stride = (
                    fx.Int64((i32_m + 31) // 32)
                    * fx.Int64(scale_chunk_dwords)
                    * fx.Int64(4)
                )
                arg_scale_a = arg_scale_a + batch_i64 * scale_batch_stride
            else:
                arg_scale_a = arg_scale_a + batch_i64 * fx.Int64(
                    sca_batch_stride
                )
            arg_scale_b = arg_scale_b + batch_i64 * fx.Int64(
                (N // 32) * scale_chunk_dwords * 4
            )
        else:
            a_rstride = fx.Int32(a_row_bytes)
            scale_a_rstride = fx.Int32(scale_chunk_dwords)

        i8_global = fx.PointerType.get(
            T.i8, address_space=fx.AddressSpace.Global, alignment=16
        )
        if const_expr(batch > 1 and a_row_stride >= 0):
            a_num_records = (
                fx.Int64(i32_m - fx.Int32(1)) * fx.Int64(a_rstride)
                + fx.Int64(a_row_bytes)
            )
        else:
            a_num_records = fx.Int64(i32_m) * fx.Int64(a_row_bytes)
        a_flat = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(
                    fx.inttoptr(i8_global, arg_a),
                    fx.make_layout(65536 * a_row_bytes, 1),
                )
            ),
            max_size=False,
            num_records_bytes=a_num_records,
        )
        a_flat_divided = fx.logical_divide(a_flat, fx.make_layout(1, 1))
        shared = fx.SharedAllocator().allocate(SharedA).peek()
        shared_a_i32 = fx.recast_iter(Int32, shared.a0.ptr)
        shared_buffer_bytes = fx.Int32(fx.ptrtoint(shared.a1.ptr)) - fx.Int32(
            fx.ptrtoint(shared.a0.ptr)
        )
        shared_buffer_i32 = shared_buffer_bytes // 4
        shared_copy = fx.make_copy_atom(fx.UniversalCopy128b(), Int32)
        dma_atom = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        i8_shared = fx.PointerType.get(Int8.ir_type, fx.AddressSpace.Shared, 512)
        shared_a_i8 = fx.recast_iter(i8_shared, shared.a0.ptr)

        def shared_iter(parity):
            return fx.add_offset(shared_a_i32, parity * shared_buffer_i32)

        def shared_view(base_iter, offset_i32):
            return fx.make_view(
                fx.add_offset(base_iter, offset_i32), fx.make_layout(4, 1)
            )

        def dma_a_to_lds(k_tile, parity):
            base_offset = rocdl.readfirstlane(
                T.i32, parity * shared_buffer_bytes + wave * (64 * 16)
            )
            lds_ptr = fx.add_offset(shared_a_i8, base_offset)
            base_k_byte = k_tile * A_ROW_B
            for index in range_constexpr(cooperative_loads):
                if const_expr(index > 0):
                    lds_ptr = fx.add_offset(lds_ptr, fx.Int32(num_threads * 16))
                linear = (index * num_threads + tid) * 16
                row = linear // A_ROW_B
                col = linear % A_ROW_B
                if const_expr(swizzle_lds):
                    col = col ^ ((row % k_blocks16) * 16)
                global_byte = (
                    (block_m + row) * a_rstride + base_k_byte + col
                )
                destination = fx.make_view(lds_ptr, fx.make_layout(1, 1))
                source = fx.slice(a_flat_divided, (None, global_byte))
                fx.copy(dma_atom, source, destination)

        def read_16_bytes(base_iter, offset_i32):
            fragment = fx.make_rmem_tensor(4, Int32)
            fx.copy(shared_copy, shared_view(base_iter, offset_i32), fragment)
            return fragment

        def read_a(parity):
            base_iter = shared_iter(parity)
            fragments = []
            for m_index in range_constexpr(m_chunks):
                for k_half in range_constexpr(k_halves):
                    row = m_index * 16 + lane_mod_16
                    row_base = row * A_ROW_I32
                    low_block = (
                        k_half * (A_KH_I32 // 4)
                        + lane_div_16 * (A_GK_I32 // 4)
                    )
                    if const_expr(swizzle_lds):
                        low_offset = (
                            row_base + (low_block ^ (row % k_blocks16)) * 4
                        )
                    else:
                        low_offset = (
                            row_base
                            + k_half * A_KH_I32
                            + lane_div_16 * A_GK_I32
                        )
                    if const_expr(a_dtype == "fp4"):
                        fragments.append(read_16_bytes(base_iter, low_offset))
                    else:
                        if const_expr(swizzle_lds):
                            high_offset = (
                                row_base
                                + (
                                    (low_block + A_HI_OFF // 4)
                                    ^ (row % k_blocks16)
                                )
                                * 4
                            )
                        else:
                            high_offset = low_offset + A_HI_OFF
                        low = Vec(read_16_bytes(base_iter, low_offset).load())
                        high = Vec(read_16_bytes(base_iter, high_offset).load())
                        fragment = fx.make_rmem_tensor(A_NDW, Int32)
                        fragment.store(low.shuffle(high, list(range(A_NDW))))
                        fragments.append(fragment)
            return fragments

        n_col_base = block_n + wave * (BN // num_waves)
        b_views = [
            _bq_view(
                arg_b,
                n_col_base + n_index * 16,
                KH4,
                K_TILES,
                k_halves,
                B_BLK_PER_MMA,
            )
            for n_index in range_constexpr(num_acc_n)
        ]
        b_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 32)
        scale_copy = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        i32_global = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        scale_layout = fx.make_layout(1 << 28, 1)
        a_scale_chunks = (i32_m + 31) // 32
        if const_expr(batch > 1 and sca_row_stride >= 0):
            a_scale_num_records = (
                fx.Int64(a_scale_chunks - 1) * fx.Int64(scale_a_rstride)
                + fx.Int64(scale_chunk_dwords)
            ) * fx.Int64(4)
        else:
            a_scale_num_records = (
                fx.Int64(a_scale_chunks)
                * fx.Int64(scale_chunk_dwords)
                * fx.Int64(4)
            )
        b_scale_num_records = fx.Int64(
            (N // 32) * scale_chunk_dwords * 4
        )
        a_scale_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(i32_global, arg_scale_a), scale_layout
                    )
                ),
                max_size=False,
                num_records_bytes=a_scale_num_records,
            ),
            fx.make_layout(1, 1),
        )
        b_scale_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(i32_global, arg_scale_b), scale_layout
                    )
                ),
                max_size=False,
                num_records_bytes=b_scale_num_records,
            ),
            fx.make_layout(1, 1),
        )
        a_scale_base = [
            (block_m // 32 + pair_index) * scale_a_rstride
            for pair_index in range_constexpr(m_pairs)
        ]
        n_scale_block = (block_n + wave * (BN // num_waves)) // 32
        b_scale_base = [
            (n_scale_block + pair_index) * scale_chunk_dwords
            for pair_index in range_constexpr(n_pairs)
        ]
        scale_lane = lane_div_16 * 16 + lane_mod_16
        num_accumulators = m_chunks * num_acc_n

        def load_b(k_tile):
            fragments = []
            for n_index in range_constexpr(num_acc_n):
                for k_half in range_constexpr(k_halves):
                    low = fx.make_rmem_tensor(4, Int32)
                    fx.copy(
                        b_copy,
                        b_views[n_index][
                            lane_div_16,
                            lane_mod_16,
                            k_tile,
                            k_half,
                            0,
                            None,
                        ],
                        low,
                    )
                    if const_expr(b_dtype == "fp4"):
                        fragments.append(low)
                    else:
                        high = fx.make_rmem_tensor(4, Int32)
                        fx.copy(
                            b_copy,
                            b_views[n_index][
                                lane_div_16,
                                lane_mod_16,
                                k_tile,
                                k_half,
                                1,
                                None,
                            ],
                            high,
                        )
                        fragment = fx.make_rmem_tensor(B_NDW, Int32)
                        fragment.store(
                            Vec(low.load()).shuffle(
                                Vec(high.load()), list(range(B_NDW))
                            )
                        )
                        fragments.append(fragment)
            return fragments

        def load_scale(source):
            fragment = fx.make_rmem_tensor(1, Int32)
            fx.copy(scale_copy, source, fragment)
            return Vec(fragment.load())[0]

        def load_scales(chunk_k_tile):
            k_offset = chunk_k_tile * scale_k0_dwords
            scales_a = [
                load_scale(
                    a_scale_flat[
                        None,
                        rocdl.readfirstlane(
                            T.i32, a_scale_base[pair_index] + k_offset
                        )
                        + scale_lane,
                    ]
                )
                for pair_index in range_constexpr(m_pairs)
            ]
            scales_b = [
                load_scale(
                    b_scale_flat[
                        None,
                        rocdl.readfirstlane(
                            T.i32, b_scale_base[pair_index] + k_offset
                        )
                        + scale_lane,
                    ]
                )
                for pair_index in range_constexpr(n_pairs)
            ]
            return scales_a, scales_b

        def compute(accumulators, a_fragments, b_fragments, scales_a, scales_b, shift):
            if const_expr(shift is not None):
                scales_a = [value.shrui(shift) for value in scales_a]
                scales_b = [value.shrui(shift) for value in scales_b]
            if const_expr(BN < 128):
                n_shift = (
                    ((block_n + wave * (BN // num_waves)) % 32) // 16 * 8
                )
                scales_b = [value.shrui(n_shift) for value in scales_b]
            c_fragments = [
                fx.make_rmem_tensor(4, Float32)
                for _ in range_constexpr(num_accumulators)
            ]
            for index in range_constexpr(num_accumulators):
                c_fragments[index].store(Vec(accumulators[index]))
            for k_half in range_constexpr(k_halves):
                for n_index in range_constexpr(num_acc_n):
                    n_pair, n_in_pair = n_index // 2, n_index % 2
                    for m_index in range_constexpr(m_chunks):
                        m_pair, m_in_pair = m_index // 2, m_index % 2
                        c_fragment = c_fragments[m_index * num_acc_n + n_index]
                        fx.gemm(
                            scale_atoms[
                                (k_half * 2 + m_in_pair, k_half * 2 + n_in_pair)
                            ],
                            c_fragment,
                            a_fragments[m_index * k_halves + k_half],
                            b_fragments[n_index * k_halves + k_half],
                            c_fragment,
                            scale_a=scales_a[m_pair],
                            scale_b=scales_b[n_pair],
                        )
            for index in range_constexpr(num_accumulators):
                accumulators[index] = c_fragments[index].load().ir_value()
            return accumulators

        def schedule_hot_loop():
            rocdl.sched_vmem(scheduled_gmem)
            rocdl.sched_dsrd(scheduled_ds_loads)
            for _ in range_constexpr(scheduled_mfmas):
                rocdl.sched_mfma(1)
            rocdl.sched_barrier(0)

        initial_accumulators = [
            Vec.filled(4, 0.0, Float32).ir_value()
            for _ in range_constexpr(num_accumulators)
        ]
        dma_a_to_lds(k_tile_start, fx.Int32(0))
        rocdl.s_waitcnt(0)
        gpu.barrier()
        for iteration, state in range(
            fx.Index(0),
            fx.Index(k_tiles_local),
            fx.Index(1),
            init=initial_accumulators,
        ):
            accumulators = list(state)
            iteration_i32 = fx.Int32(iteration)
            current = iteration_i32 % 2
            next_buffer = (iteration_i32 + 1) % 2
            k_tile = k_tile_start + iteration_i32
            next_k_tile = iteration_i32 + 1
            prefetch_k_tile = k_tile_start + (
                next_k_tile - next_k_tile // k_tiles_local
            )
            chunk_k_tile = (
                k_tile
                if tiles_per_chunk == 1
                else k_tile // tiles_per_chunk
            )
            scale_shift = (
                None
                if tiles_per_chunk == 1
                else (k_tile % tiles_per_chunk) * 16
            )
            a_fragments = read_a(current)
            b_fragments = load_b(k_tile)
            scales_a, scales_b = load_scales(chunk_k_tile)
            dma_a_to_lds(prefetch_k_tile, next_buffer)
            accumulators = compute(
                accumulators,
                a_fragments,
                b_fragments,
                scales_a,
                scales_b,
                scale_shift,
            )
            schedule_hot_loop()
            rocdl.s_waitcnt(0)
            gpu.barrier()
            results = yield accumulators
        accumulators = results

        c_stride = N if c_row_stride < 0 else c_row_stride
        if const_expr(k_batch > 1):
            store_elem = Float32
            element_bytes = 4
            c_addr = arg_c + fx.Int64(bid_z) * fx.Int64(i32_m) * fx.Int64(
                N
            ) * fx.Int64(element_bytes)
        else:
            store_elem = out_elem
            element_bytes = 2
            c_addr = arg_c
            if const_expr(batch > 1):
                c_batch_bytes = (
                    fx.Int64(i32_m) * fx.Int64(N) * fx.Int64(2)
                    if c_batch_stride < 0
                    else fx.Int64(c_batch_stride)
                )
                c_addr = c_addr + fx.Int64(batch_index) * c_batch_bytes

        c_tile_addr = c_addr + fx.Int64(block_m) * fx.Int64(
            c_stride
        ) * fx.Int64(element_bytes)
        rows_remaining = fx.Index(i32_m) - fx.Index(block_m)
        rows_workgroup = (rows_remaining < fx.Index(BM)).select(
            rows_remaining, fx.Index(BM)
        )
        c_num_records = (
            fx.Int64(rows_workgroup)
            * fx.Int64(c_stride)
            * fx.Int64(element_bytes)
        )
        c_pointer_type = fx.PointerType.get(
            store_elem.ir_type,
            address_space=fx.AddressSpace.Global,
            alignment=element_bytes,
        )
        c_flat = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(c_pointer_type, c_tile_addr),
                        fx.make_layout(1 << 28, 1),
                    )
                ),
                max_size=False,
                num_records_bytes=c_num_records,
            ),
            fx.make_layout(1, 1),
        )
        if const_expr(k_batch > 1):
            c_copy = fx.make_copy_atom(
                fx.rocdl.BufferCopy32b(), store_elem
            )
        else:
            c_copy = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b(), store_elem
            )
        c_rstride = fx.Int32(c_stride)
        column_wave = block_n + wave * (BN // num_waves) + lane_mod_16
        for m_index in range_constexpr(m_chunks):
            row_local = m_index * 16 + lane_div_16 * 4
            for n_index in range_constexpr(num_acc_n):
                column = column_wave + n_index * 16
                accumulator = Vec(
                    accumulators[m_index * num_acc_n + n_index]
                ).to(store_elem)
                for item in range_constexpr(4):
                    fragment = fx.make_rmem_tensor(1, store_elem)
                    fragment.store(
                        Vec.from_elements([accumulator[item]], store_elem)
                    )
                    offset = (
                        (row_local + item) * c_rstride + column
                    )
                    fx.copy(c_copy, fragment, c_flat[None, offset])

    c_addr = fx.Int64(fx.ptrtoint(arg_c))
    a_addr = fx.Int64(fx.ptrtoint(arg_a))
    b_addr = fx.Int64(fx.ptrtoint(arg_b))
    scale_a_addr = fx.Int64(fx.ptrtoint(arg_scale_a))
    scale_b_addr = fx.Int64(fx.ptrtoint(arg_scale_b))
    waves = waves_per_eu if const_expr(waves_per_eu > 0) else None
    grid_x = (i32_m + (BM - 1)) // BM
    grid_y = i32_n // BN
    grid_z = batch * k_batch
    kernel_gemm(
        c_addr,
        a_addr,
        b_addr,
        scale_a_addr,
        scale_b_addr,
        i32_m,
        i32_n,
        value_attrs={"rocdl.waves_per_eu": waves},
    ).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(num_threads, 1, 1),
        stream=stream,
    )


_REDUCE_BLOCK = 256


def _pack_pair_from_f32(accumulator_low, accumulator_high, out_dtype, *, i32):
    out_type = T.bf16 if out_dtype == "bf16" else T.f16
    low_i16 = arith.bitcast(T.i16, arith.trunc_f(out_type, accumulator_low))
    high_i16 = arith.bitcast(T.i16, arith.trunc_f(out_type, accumulator_high))
    low_i32 = fx.Int32(arith.extui(i32, low_i16))
    high_i32 = fx.Int32(arith.extui(i32, high_i16))
    return low_i32 | (high_i32 << arith.constant(16, type=i32))


@flyc.jit
def launch_splitk_reduce(
    arg_tmp: fx.Pointer,
    arg_out: fx.Pointer,
    n_out_dwords: fx.Int32,
    slab_stride_dwords: fx.Int32,
    stream: fx.Stream,
    split_k: Constexpr[int],
    out_dtype: Constexpr[str],
):
    """Reduce split-K fp32 slabs into bf16/fp16 output."""

    @flyc.kernel
    def reduce_kernel(
        tmp: fx.Pointer,
        out: fx.Pointer,
        n_out_dwords_i32: fx.Int32,
        slab_stride_i32: fx.Int32,
    ):
        f32 = T.f32
        i32 = T.i32
        block = fx.block_idx.x
        thread = fx.thread_idx.x
        input_resource = ptr_rsrc(tmp)
        output_resource = ptr_rsrc(out)
        dword = fx.Int32(block) * _REDUCE_BLOCK + fx.Int32(thread)
        if dword < n_out_dwords_i32:
            first_element = dword * arith.constant(2, type=i32)
            accumulator_low = fx.Float32(0.0)
            accumulator_high = fx.Float32(0.0)
            for split_index in range_constexpr(split_k):
                split_offset = arith.constant(
                    split_index, type=i32
                ) * slab_stride_i32
                raw = buffer_ops.buffer_load(
                    input_resource,
                    first_element + split_offset,
                    vec_width=2,
                    dtype=f32,
                )
                raw_vector = Vec(raw)
                accumulator_low = accumulator_low + raw_vector[0]
                accumulator_high = accumulator_high + raw_vector[1]
            packed = _pack_pair_from_f32(
                accumulator_low,
                accumulator_high,
                out_dtype,
                i32=i32,
            )
            buffer_ops.buffer_store(packed, output_resource, dword)

    grid_x = (n_out_dwords + (_REDUCE_BLOCK - 1)) // _REDUCE_BLOCK
    reduce_kernel(
        arg_tmp,
        arg_out,
        n_out_dwords,
        slab_stride_dwords,
    ).launch(
        grid=(grid_x, 1, 1),
        block=(_REDUCE_BLOCK, 1, 1),
        stream=stream,
    )

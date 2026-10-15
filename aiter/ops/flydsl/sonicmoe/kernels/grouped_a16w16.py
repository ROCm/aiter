# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device-scheduled row-major A16 grouped GEMM for SonicMoE backward.

This kernel covers the ``dX = dZ @ W1`` contraction, where ``dZ`` is
route-sorted row-major A16 and each expert's public weight is row-major
``[K, N]``.  Its epilogue can retain sorted row order or scatter fixed-K rows
directly to their unique route slots.  It intentionally keeps the public
SonicMoE weight layout and performs the B transpose through LDS for gfx950 MFMA
consumption.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

from .gemm_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    __barrier,
    buffer_load_lds_inline,
    get_wave_lds_offset,
    make_lds_layout,
    make_transposed_lds_layout,
    make_wave_lds_ptr,
    swizzled_col_idx,
    transposed_contiguous_idx,
)
from .stage_utils import (
    _gep1,
    _global_base_ptr1,
    _global_i32_at,
    _raw,
)

_MAX_BUFFER_BYTES = (1 << 32) - 1


@functools.lru_cache(maxsize=128)
def compile_sonic_grouped_a16w16_nn(
    *,
    contraction_size: int,
    output_size: int,
    num_experts: int,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    stages: int = 2,
    n_waves: int = 2,
    sorted_block_m: int = 64,
    compact_grid: bool = True,
    device_index: int = 0,
    min_active_experts: int = 0,
    max_active_experts: int | None = None,
    store_route_slots: bool = False,
    store_route_ids: bool = False,
    top_k: int = 1,
    expert_m_reuse: bool = False,
    expert_m_reuse_threshold: int | None = None,
    exact_tile_queue: bool = False,
    gather_a_rows: bool = False,
):
    """Compile a BF16 grouped NN GEMM with a device-derived M schedule.

    ``compact_grid=True`` consumes ``schedule=[count, m_block...]``.  The
    non-compact mode consumes sorter metadata blocks directly and is intended
    for fixed-K T1, where each active expert has exactly one real M tile.
    ``device_index`` participates in the cache key because loaded code objects
    are tied to a ROCm device.  The optional inclusive active-expert interval
    is read from ``arg_cumsum[0]`` in compact mode and permits mutually
    exclusive gfx950 launch profiles without copying routing statistics to the
    host.  ``store_route_slots=True`` folds the fixed-K sorted-row permutation
    into the BF16 epilogue and writes ``[token, slot, output]`` directly.  The
    ``store_route_ids=True`` variant instead consumes one original route id per
    sorted row and writes route-order ``[route, output]`` storage.  The two
    direct-output modes are mutually exclusive; default sorted-output mode
    retains the original launcher ABI.

    ``expert_m_reuse=True`` changes only the device schedule: ``arg_schedule``
    uses the counter-first ``[count, (expert, first_row) * capacity]`` active
    expert ABI and ``arg_eids`` supplies exact expert frequencies.  One logical
    work item owns an expert/N slab and either the even or odd M tiles, allowing
    the same CTA to revisit the same weight slab.  The launcher ABI and the
    GEMM/load/store body are otherwise identical to compact descriptor mode.

    ``expert_m_reuse_threshold`` combines the compact descriptor and M-reuse
    schedules in one kernel launch.  ``arg_schedule`` and ``arg_eids`` retain
    their compact descriptor meanings, ``arg_cumsum`` supplies the active
    expert queue, and the extra launcher argument supplies exact expert
    frequencies.  The device-side active count selects M-reuse at or above the
    threshold; lower counts retain descriptor-major scheduling.

    ``exact_tile_queue=True`` consumes the self-contained counter-first
    ``[count, (expert, first_row, valid_rows) * capacity]`` ABI.  It allows an
    M tile wider than ``sorted_block_m`` and assigns every M/N tile pair its own
    logical work item instead of serializing an expert's M tiles in one CTA.
    ``gather_a_rows=True`` keeps that schedule but sources each A row through
    its packed sorter token ID.  This lets expert-major identity backward read
    token-major ``grad_output`` directly without materializing a padded copy.
    """

    del device_index
    if (
        min(
            contraction_size,
            output_size,
            num_experts,
            block_m,
            block_n,
            block_k,
            stages,
            n_waves,
        )
        <= 0
    ):
        raise ValueError(
            "grouped A16 GEMM dimensions and tuning values must be positive"
        )
    if not isinstance(min_active_experts, int) or min_active_experts < 0:
        raise ValueError("min_active_experts must be a non-negative int")
    if max_active_experts is not None and (
        not isinstance(max_active_experts, int)
        or max_active_experts < min_active_experts
    ):
        raise ValueError(
            "max_active_experts must be None or at least min_active_experts"
        )
    if not isinstance(store_route_slots, bool):
        raise TypeError("store_route_slots must be a bool")
    if not isinstance(store_route_ids, bool):
        raise TypeError("store_route_ids must be a bool")
    if store_route_slots and store_route_ids:
        raise ValueError("store_route_slots and store_route_ids are mutually exclusive")
    if not isinstance(expert_m_reuse, bool):
        raise TypeError("expert_m_reuse must be a bool")
    if not isinstance(exact_tile_queue, bool):
        raise TypeError("exact_tile_queue must be a bool")
    if not isinstance(gather_a_rows, bool):
        raise TypeError("gather_a_rows must be a bool")
    if expert_m_reuse_threshold is not None and (
        not isinstance(expert_m_reuse_threshold, int)
        or isinstance(expert_m_reuse_threshold, bool)
        or expert_m_reuse_threshold <= 0
    ):
        raise ValueError("expert_m_reuse_threshold must be None or a positive int")
    if not isinstance(top_k, int) or top_k <= 0 or top_k > 256:
        raise ValueError("top_k must be an int in [1, 256]")
    if (min_active_experts > 0 or max_active_experts is not None) and not compact_grid:
        raise ValueError("active-expert guards require compact_grid=True")
    if expert_m_reuse and not compact_grid:
        raise ValueError("expert_m_reuse requires compact_grid=True")
    if expert_m_reuse and not store_route_slots:
        raise ValueError("expert_m_reuse requires store_route_slots=True")
    if expert_m_reuse and expert_m_reuse_threshold is not None:
        raise ValueError(
            "expert_m_reuse and expert_m_reuse_threshold are mutually exclusive"
        )
    if exact_tile_queue and (expert_m_reuse or expert_m_reuse_threshold is not None):
        raise ValueError(
            "exact_tile_queue and other queue schedules are mutually exclusive"
        )
    if exact_tile_queue and not compact_grid:
        raise ValueError("exact_tile_queue requires compact_grid=True")
    if gather_a_rows and not exact_tile_queue:
        raise ValueError("gather_a_rows requires exact_tile_queue=True")
    if expert_m_reuse_threshold is not None and not compact_grid:
        raise ValueError("expert_m_reuse_threshold requires compact_grid=True")
    if expert_m_reuse_threshold is not None and not store_route_slots:
        raise ValueError("expert_m_reuse_threshold requires store_route_slots=True")
    if block_m % 16 or (not exact_tile_queue and sorted_block_m % block_m):
        raise ValueError(
            "block_m must be a multiple of 16 and descriptor schedules require it "
            "to divide sorted_block_m"
        )
    if output_size % block_n:
        raise ValueError("output_size must be divisible by block_n")
    if contraction_size % block_k:
        raise ValueError("contraction_size must be divisible by block_k")
    if block_n % (n_waves * 16):
        raise ValueError("block_n must be divisible by n_waves * 16")
    if stages < 2:
        raise ValueError("stages must be at least two")
    if contraction_size // block_k < stages - 1:
        raise ValueError("the contraction must contain at least stages - 1 K tiles")
    if n_waves not in (2, 4):
        raise ValueError("n_waves must be two or four")

    elem_dtype = fx.BFloat16
    elem_bytes = 2
    async_load_bytes = GFX950_DMA_BYTES
    async_load_vec_size = async_load_bytes // elem_bytes
    block_threads = n_waves * GFX950_WAVE_SIZE
    a_load_threads = min(block_threads, block_m * block_k // async_load_vec_size)
    b_load_threads = min(block_threads, block_n * block_k // async_load_vec_size)
    if a_load_threads % GFX950_WAVE_SIZE or b_load_threads % GFX950_WAVE_SIZE:
        raise ValueError("direct-to-LDS participants must contain whole waves")
    if block_m * block_k % (a_load_threads * async_load_vec_size):
        raise ValueError("A tile is not exactly covered by vector loads")
    if block_n * block_k % (b_load_threads * async_load_vec_size):
        raise ValueError("B tile is not exactly covered by vector loads")
    a_load_iters = block_m * block_k // (a_load_threads * async_load_vec_size)
    b_load_iters = block_n * block_k // (b_load_threads * async_load_vec_size)
    ldg_x_threads = block_k // async_load_vec_size
    if ldg_x_threads * async_load_vec_size != block_k:
        raise ValueError("block_k must be divisible by the 16-byte BF16 vector width")
    if a_load_threads % ldg_x_threads or b_load_threads % ldg_x_threads:
        raise ValueError("load participants must be divisible by the K-thread count")

    mma_m = 16
    mma_n = 16
    mma_k = 32
    mma_m_iters = block_m // mma_m
    mma_n_iters = block_n // (n_waves * mma_n)
    k_mma_iters = block_k // mma_k
    if mma_m_iters * mma_m != block_m or mma_n_iters * n_waves * mma_n != block_n:
        raise ValueError("tile shape is incompatible with MFMA wave decomposition")
    if k_mma_iters * mma_k != block_k:
        raise ValueError("block_k is incompatible with MFMA K=32")

    ab_bytes = stages * (block_m + block_n) * block_k * elem_bytes
    c_bytes = block_m * block_n * elem_bytes
    if max(ab_bytes, c_bytes) > 163840:
        raise ValueError("grouped A16 GEMM exceeds gfx950 LDS capacity")
    if contraction_size * output_size * elem_bytes > _MAX_BUFFER_BYTES:
        raise ValueError(
            "one expert weight slab exceeds the gfx950 buffer offset range"
        )

    num_n_blocks = output_size // block_n
    cshuffle_vec_size = async_load_vec_size
    cshuffle_x_threads = block_n // cshuffle_vec_size
    if (store_route_slots or store_route_ids) and (
        cshuffle_x_threads > GFX950_WAVE_SIZE or GFX950_WAVE_SIZE % cshuffle_x_threads
    ):
        raise ValueError(
            "route-slot output vectors for one row must form whole groups within a wave"
        )
    cshuffle_vectors = block_m * block_n // cshuffle_vec_size
    cshuffle_iters = (cshuffle_vectors + block_threads - 1) // block_threads
    if exact_tile_queue:
        schedule_suffix = "_exact_queue"
    elif expert_m_reuse:
        schedule_suffix = "_mreuse2_xcd1"
    elif expert_m_reuse_threshold is not None:
        schedule_suffix = f"_mreuse2_ge{expert_m_reuse_threshold}_xcd1"
    else:
        schedule_suffix = ""
    name = (
        f"sonic_grouped_nn_bf16_k{contraction_size}_n{output_size}_e{num_experts}"
        f"_bm{block_m}_bn{block_n}_bk{block_k}_s{stages}_nw{n_waves}"
        f"_{'compact' if compact_grid else 'metadata'}"
        f"_amin{min_active_experts}_amax{max_active_experts}"
        f"_{'routeid' if store_route_ids else (f'routek{top_k}' if store_route_slots else 'sorted')}"
        f"{schedule_suffix}"
        f"_gathera{int(gather_a_rows)}"
    )

    @fx.struct
    class SharedABStorage:
        a: fx.Array[elem_dtype, stages * block_m * block_k, 16]
        b: fx.Array[elem_dtype, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[elem_dtype, block_m * block_n, 16]

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def grouped_nn_kernel(
        arg_a: fx.Int64,
        arg_b: fx.Int64,
        arg_schedule: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_frequency: fx.Int64,
        arg_out: fx.Int64,
        arg_sorted_token_ids: fx.Int64,
        i32_tokens: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        block_id = fx.Int32(gpu.block_id("x"))
        grid_size = fx.Int32(gpu.grid_dim.x)

        storage = fx.SharedAllocator().allocate(SharedStorage)
        smem_a = storage.ab.a.peek().ptr
        smem_b = storage.ab.b.peek().ptr
        smem_c = storage.c.peek().ptr
        a_lds_layout = make_lds_layout(block_m, block_k)
        b_lds_layout = make_transposed_lds_layout(block_n, block_k)
        c_lds_layout = fx.make_layout((block_m, block_n), (block_n, 1))
        s_a = fx.make_view(smem_a, a_lds_layout)
        s_b = fx.make_view(smem_b, b_lds_layout)
        s_c = fx.make_view(smem_c, c_lds_layout)

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(mma_m, mma_n, mma_k, elem_dtype))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((1, n_waves, 1), (n_waves, 1, 0)),
            fx.make_tile(
                None,
                None,
                fx.make_layout((mma_k // 4, 4), (1, mma_k // 4)),
            ),
        )
        thr_mma = tiled_mma.thr_slice(tid)
        universal_copy = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
        buffer_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
        b_transpose_copy = fx.make_copy_atom(
            fx.rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype
        )
        thr_copy_a = fx.make_tiled_copy_A(buffer_copy, tiled_mma).get_slice(tid)
        thr_copy_b = fx.make_tiled_copy_B(b_transpose_copy, tiled_mma).get_slice(tid)
        frag_a = thr_mma.make_fragment_A(s_a)
        frag_b = thr_mma.make_fragment_B(s_b)
        frag_c = thr_mma.make_fragment_C(s_c)
        frag_a_retile = thr_copy_a.retile(frag_a)
        frag_b_retile = thr_copy_b.retile(frag_b)
        row_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (1, 0)))
        col_coords = fx.make_view(0, fx.make_layout((block_m, block_n), (0, 1)))
        thr_c_row = thr_mma.partition_C(row_coords)
        thr_c_col = thr_mma.partition_C(col_coords)
        out_base = _global_base_ptr1(arg_out)

        def _async_load(
            lds_base,
            rsrc,
            lds_layout,
            outer_tile_size,
            outer_bound,
            global_outer_offset,
            leading_stride,
            load_threads,
            load_iters,
            is_k_major,
            k_tile,
        ):
            if tid < fx.Int32(load_threads):
                wave_offset = get_wave_lds_offset(tid, async_load_bytes)
                lds_ptr = make_wave_lds_ptr(lds_base, wave_offset)
                for load_iter in range_constexpr(load_iters):
                    global_tid = fx.Int32(load_threads * load_iter) + tid
                    if const_expr(is_k_major):
                        outer_x_threads = outer_tile_size // async_load_vec_size
                        outer_lds_idx = (
                            global_tid
                            % fx.Int32(outer_x_threads)
                            * fx.Int32(async_load_vec_size)
                        )
                        k_local_idx = global_tid // fx.Int32(outer_x_threads)
                        outer_local_idx = transposed_contiguous_idx(
                            outer_lds_idx,
                            k_local_idx,
                            lds_layout,
                            outer_tile_size,
                        )
                        global_k_idx = k_tile * fx.Int32(block_k) + k_local_idx
                    else:
                        outer_local_idx = global_tid // fx.Int32(ldg_x_threads)
                        k_local_idx = (
                            global_tid
                            % fx.Int32(ldg_x_threads)
                            * fx.Int32(async_load_vec_size)
                        )
                        global_k_idx = k_tile * fx.Int32(block_k) + swizzled_col_idx(
                            outer_local_idx,
                            k_local_idx,
                            lds_layout,
                            block_k,
                        )
                    global_outer_idx = global_outer_offset + outer_local_idx
                    if const_expr(is_k_major):
                        global_byte = (
                            global_k_idx * fx.Int32(leading_stride) + global_outer_idx
                        ) * fx.Int32(elem_bytes)
                    else:
                        global_byte = (
                            global_outer_idx * fx.Int32(leading_stride) + global_k_idx
                        ) * fx.Int32(elem_bytes)
                    buffer_load_lds_inline(rsrc, lds_ptr, global_byte, async_load_bytes)
                    if load_iter < load_iters - 1:
                        lds_ptr = lds_ptr + fx.Int32(load_threads * async_load_bytes)

        def _run_tile(m_row, n_block, expert, valid_rows):
            n_col = n_block * fx.Int32(block_n)
            a_addr = fx.Int64(arg_a)
            a_records_bytes = fx.Int64(i32_tokens) * fx.Int64(
                contraction_size * elem_bytes
            )
            if const_expr(not gather_a_rows):
                a_addr = a_addr + fx.Int64(m_row) * fx.Int64(
                    contraction_size * elem_bytes
                )
                a_records_bytes = fx.Int64(valid_rows) * fx.Int64(
                    contraction_size * elem_bytes
                )
            b_addr = fx.Int64(arg_b) + fx.Int64(expert) * fx.Int64(
                contraction_size * output_size * elem_bytes
            )
            a_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(a_addr),
                num_records_bytes=_raw(a_records_bytes),
            )
            b_rsrc = buffer_ops.create_buffer_resource_from_addr(
                _raw(b_addr),
                num_records_bytes=contraction_size * output_size * elem_bytes,
            )
            frag_c.fill(0.0)

            gathered_a_tokens = []
            gathered_a_source_lanes = []
            if const_expr(gather_a_rows):
                gather_lane = tid % fx.Int32(GFX950_WAVE_SIZE)
                for load_iter in range_constexpr(a_load_iters):
                    global_tid = fx.Int32(a_load_threads * load_iter) + tid
                    local_row = global_tid // fx.Int32(ldg_x_threads)
                    row_valid = local_row < valid_rows
                    safe_sorted_row = row_valid.select(
                        m_row + local_row,
                        m_row,
                    )
                    token_lane = fx.Int32(0)
                    if gather_lane % fx.Int32(ldg_x_threads) == fx.Int32(0):
                        packed = fx.Int32(
                            _global_i32_at(arg_sorted_token_ids, safe_sorted_row)
                        )
                        token = packed & fx.Int32(0x00FFFFFF)
                        token_lane = (row_valid & (token < i32_tokens)).select(
                            token,
                            i32_tokens,
                        )
                    gathered_a_tokens.append(token_lane)
                    gathered_a_source_lanes.append(
                        gather_lane - gather_lane % fx.Int32(ldg_x_threads)
                    )

            def _load_a(k_tile, stage):
                if const_expr(gather_a_rows):
                    lds_ptr = make_wave_lds_ptr(
                        smem_a + stage * block_m * block_k,
                        get_wave_lds_offset(tid, async_load_bytes),
                    )
                    for load_iter in range_constexpr(a_load_iters):
                        global_tid = fx.Int32(a_load_threads * load_iter) + tid
                        local_row = global_tid // fx.Int32(ldg_x_threads)
                        k_local_idx = (global_tid % fx.Int32(ldg_x_threads)) * fx.Int32(
                            async_load_vec_size
                        )
                        global_k_idx = k_tile * fx.Int32(block_k) + swizzled_col_idx(
                            local_row,
                            k_local_idx,
                            a_lds_layout,
                            block_k,
                        )
                        token = fx.Int32(
                            rocdl.ds_bpermute(
                                T.i32,
                                gathered_a_source_lanes[load_iter] * fx.Int32(4),
                                gathered_a_tokens[load_iter],
                            )
                        )
                        global_byte = (
                            token * fx.Int32(contraction_size) + global_k_idx
                        ) * fx.Int32(elem_bytes)
                        buffer_load_lds_inline(
                            a_rsrc,
                            lds_ptr,
                            global_byte,
                            async_load_bytes,
                        )
                        if load_iter < a_load_iters - 1:
                            lds_ptr = lds_ptr + fx.Int32(
                                a_load_threads * async_load_bytes
                            )
                else:
                    _async_load(
                        smem_a + stage * block_m * block_k,
                        a_rsrc,
                        a_lds_layout,
                        valid_rows,
                        block_m,
                        fx.Int32(0),
                        contraction_size,
                        a_load_threads,
                        a_load_iters,
                        False,
                        k_tile,
                    )

            def _load_b(k_tile, stage):
                _async_load(
                    smem_b + stage * block_n * block_k,
                    b_rsrc,
                    b_lds_layout,
                    block_n,
                    output_size,
                    n_col,
                    output_size,
                    b_load_threads,
                    b_load_iters,
                    True,
                    k_tile,
                )

            def _compute_stage(read_stage):
                tiled_a = thr_copy_a.partition_S(
                    fx.make_view(smem_a + read_stage * block_m * block_k, a_lds_layout)
                )
                tiled_b = thr_copy_b.partition_S(
                    fx.make_view(smem_b + read_stage * block_n * block_k, b_lds_layout)
                )
                for k_iter in range_constexpr(k_mma_iters):
                    fx.copy(
                        b_transpose_copy,
                        tiled_b[None, None, k_iter],
                        frag_b_retile[None, None, k_iter],
                    )
                    fx.copy(
                        universal_copy,
                        tiled_a[None, None, k_iter],
                        frag_a_retile[None, None, k_iter],
                    )
                    fx.gemm(
                        tiled_mma,
                        frag_c,
                        frag_a[None, None, k_iter],
                        frag_b[None, None, k_iter],
                        frag_c,
                        traversal_order=fx.GemmTraversalOrder.KNM,
                    )

            for stage in range_constexpr(stages - 1):
                _load_b(fx.Int32(stage), stage)
                _load_a(fx.Int32(stage), stage)
            rocdl.sched_barrier(0)

            k_tiles = fx.Int32(contraction_size // block_k)
            main_loop_end = k_tiles - fx.Int32(stages - 1)
            for k_tile in range(0, main_loop_end, 1):
                read_stage = k_tile % fx.Int32(stages)
                write_stage = (read_stage + fx.Int32(stages - 1)) % fx.Int32(stages)
                __barrier((stages - 2) * (a_load_iters + b_load_iters))
                _load_b(k_tile + fx.Int32(stages - 1), write_stage)
                _load_a(k_tile + fx.Int32(stages - 1), write_stage)
                _compute_stage(read_stage)
                rocdl.sched_vmem(a_load_iters + b_load_iters)
                for _ in range_constexpr(k_mma_iters):
                    rocdl.sched_dsrd(mma_n_iters)
                    rocdl.sched_dsrd(mma_m_iters)
                    for _ in range_constexpr(mma_m_iters):
                        rocdl.sched_mfma(mma_n_iters)
                rocdl.sched_barrier(0)

            read_stage = main_loop_end % fx.Int32(stages)
            for drain in range_constexpr(stages - 1):
                __barrier((stages - 2 - drain) * (a_load_iters + b_load_iters))
                _compute_stage(read_stage)
                read_stage = (read_stage + fx.Int32(1)) % fx.Int32(stages)

            frag_c_out = fx.make_fragment_like(frag_c, elem_dtype)
            frag_c_out.store(frag_c.load().to(elem_dtype))
            gpu.barrier()
            for frag_idx in range_constexpr(fx.size(frag_c_out.shape).unpack()):
                row = fx.get_scalar(thr_c_row[frag_idx])
                col = fx.get_scalar(thr_c_col[frag_idx])
                s_c[row, col] = frag_c_out[frag_idx]
            gpu.barrier()

            for store_iter in range_constexpr(cshuffle_iters):
                vector_idx = fx.Int32(block_threads * store_iter) + tid
                if vector_idx < fx.Int32(cshuffle_vectors):
                    local_row = vector_idx // fx.Int32(cshuffle_x_threads)
                    local_col = (
                        vector_idx
                        % fx.Int32(cshuffle_x_threads)
                        * fx.Int32(cshuffle_vec_size)
                    )
                    c_vec = fx.ptr_load(
                        smem_c + local_row * fx.Int32(block_n) + local_col,
                        result_type=fx.Vector.make_type(cshuffle_vec_size, elem_dtype),
                    )
                    sorted_row = m_row + local_row
                    valid_output_row = local_row < valid_rows
                    if const_expr(store_route_slots or store_route_ids):
                        # Every row is written by ``block_n / 8`` adjacent
                        # lanes.  Fetch its packed route once per lane group
                        # and broadcast within the wave instead of issuing the
                        # same VMEM load for every 16-byte output vector.
                        lane = tid % fx.Int32(GFX950_WAVE_SIZE)
                        packed_lane = fx.Int32(0)
                        if lane % fx.Int32(cshuffle_x_threads) == fx.Int32(0):
                            safe_sorted_row = valid_output_row.select(sorted_row, m_row)
                            packed_lane = fx.Int32(
                                _global_i32_at(arg_sorted_token_ids, safe_sorted_row)
                            )
                        source_lane = lane - lane % fx.Int32(cshuffle_x_threads)
                        packed = fx.Int32(
                            rocdl.ds_bpermute(
                                T.i32,
                                source_lane * fx.Int32(4),
                                packed_lane,
                            )
                        )
                        if const_expr(store_route_ids):
                            route_row = packed
                            valid = (
                                valid_output_row
                                & (route_row >= fx.Int32(0))
                                & (route_row < i32_tokens)
                            )
                        else:
                            token = packed & fx.Int32(0x00FFFFFF)
                            slot = (packed >> fx.Int32(24)) & fx.Int32(0xFF)
                            route_row = token * fx.Int32(top_k) + slot
                            valid = (
                                valid_output_row
                                & (token < i32_tokens)
                                & (slot < fx.Int32(top_k))
                            )
                        if valid:
                            global_element = fx.Int64(route_row) * fx.Int64(
                                output_size
                            ) + fx.Int64(n_col + local_col)
                            llvm.StoreOp(
                                _raw(c_vec),
                                _gep1(out_base, global_element * fx.Int64(elem_bytes)),
                                alignment=16,
                            )
                    elif valid_output_row:
                        global_element = fx.Int64(sorted_row) * fx.Int64(
                            output_size
                        ) + fx.Int64(n_col + local_col)
                        llvm.StoreOp(
                            _raw(c_vec),
                            _gep1(out_base, global_element * fx.Int64(elem_bytes)),
                            alignment=16,
                        )
            gpu.barrier()

        cumsum0 = rocdl.readfirstlane(
            T.i32, _raw(_global_i32_at(arg_cumsum, fx.Int32(0)))
        )

        def _xcd_m_reuse_work(work, work_count):
            """Bijectively transpose dynamic M-reuse work across eight XCDs."""

            nxcd = fx.Int32(8)
            xq = work_count // nxcd
            xr = work_count % nxcd
            xc = work % nxcd
            return xc * xq + fx.Int32(arith.minsi(_raw(xc), _raw(xr))) + work // nxcd

        if const_expr(exact_tile_queue):
            tile_count = rocdl.readfirstlane(
                T.i32,
                _raw(_global_i32_at(arg_schedule, fx.Int32(0))),
            )
            work_count = tile_count * fx.Int32(num_n_blocks)

            def _run_work(work):
                tile_index = work // fx.Int32(num_n_blocks)
                n_block = work % fx.Int32(num_n_blocks)
                record_offset = fx.Int32(1) + tile_index * fx.Int32(3)
                expert = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_schedule, record_offset)),
                )
                first_sorted_row = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_schedule, record_offset + fx.Int32(1))),
                )
                valid_rows = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_schedule, record_offset + fx.Int32(2))),
                )
                _run_tile(
                    first_sorted_row,
                    n_block,
                    expert,
                    valid_rows,
                )

        elif const_expr(expert_m_reuse):
            active_count = rocdl.readfirstlane(
                T.i32,
                _raw(_global_i32_at(arg_schedule, fx.Int32(0))),
            )
            m_partitions = 2
            work_per_expert = num_n_blocks * m_partitions
            work_count = active_count * fx.Int32(work_per_expert)

            def _run_work(work):
                work = _xcd_m_reuse_work(work, work_count)
                active_index = work // fx.Int32(work_per_expert)
                expert_work = work % fx.Int32(work_per_expert)
                n_block = expert_work // fx.Int32(m_partitions)
                m_partition = expert_work % fx.Int32(m_partitions)
                active_offset = fx.Int32(1) + active_index * fx.Int32(2)
                expert = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_schedule, active_offset)),
                )
                first_sorted_row = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_schedule, active_offset + fx.Int32(1))),
                )
                frequency = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_eids, expert)),
                )
                num_m_blocks = (frequency + fx.Int32(block_m - 1)) // fx.Int32(block_m)
                partition_tiles = (
                    num_m_blocks + fx.Int32(m_partitions - 1) - m_partition
                ) // fx.Int32(m_partitions)
                first_m_block = first_sorted_row // fx.Int32(block_m)
                for partition_tile in range(0, partition_tiles, 1):
                    local_m_block = m_partition + fx.Int32(partition_tile) * fx.Int32(
                        m_partitions
                    )
                    _run_tile(
                        (first_m_block + local_m_block) * fx.Int32(block_m),
                        n_block,
                        expert,
                        fx.Int32(block_m),
                    )

        elif const_expr(expert_m_reuse_threshold is not None):
            active_count = cumsum0
            use_m_reuse = active_count >= fx.Int32(expert_m_reuse_threshold)
            descriptor_count = rocdl.readfirstlane(
                T.i32,
                _raw(_global_i32_at(arg_schedule, fx.Int32(0))),
            )
            m_partitions = 2
            reuse_work_per_expert = num_n_blocks * m_partitions
            descriptor_work_count = descriptor_count * fx.Int32(num_n_blocks)
            reuse_work_count = active_count * fx.Int32(reuse_work_per_expert)
            work_count = use_m_reuse.select(reuse_work_count, descriptor_work_count)

            def _run_work(work):
                first_m_block = fx.Int32(0)
                n_block = fx.Int32(0)
                expert = fx.Int32(0)
                tile_count = fx.Int32(0)
                tile_stride = fx.Int32(1)
                if use_m_reuse:
                    work = _xcd_m_reuse_work(work, reuse_work_count)
                    active_index = work // fx.Int32(reuse_work_per_expert)
                    expert_work = work % fx.Int32(reuse_work_per_expert)
                    n_block = expert_work // fx.Int32(m_partitions)
                    m_partition = expert_work % fx.Int32(m_partitions)
                    active_offset = fx.Int32(1) + active_index * fx.Int32(2)
                    expert = rocdl.readfirstlane(
                        T.i32,
                        _raw(_global_i32_at(arg_cumsum, active_offset)),
                    )
                    first_sorted_row = rocdl.readfirstlane(
                        T.i32,
                        _raw(_global_i32_at(arg_cumsum, active_offset + fx.Int32(1))),
                    )
                    frequency = rocdl.readfirstlane(
                        T.i32,
                        _raw(_global_i32_at(arg_frequency, expert)),
                    )
                    num_m_blocks = (frequency + fx.Int32(block_m - 1)) // fx.Int32(
                        block_m
                    )
                    tile_count = (
                        num_m_blocks + fx.Int32(m_partitions - 1) - m_partition
                    ) // fx.Int32(m_partitions)
                    first_m_block = first_sorted_row // fx.Int32(block_m) + m_partition
                    tile_stride = fx.Int32(m_partitions)
                else:
                    descriptor_index = work // fx.Int32(num_n_blocks)
                    n_block = work % fx.Int32(num_n_blocks)
                    first_m_block = rocdl.readfirstlane(
                        T.i32,
                        _raw(
                            _global_i32_at(arg_schedule, descriptor_index + fx.Int32(1))
                        ),
                    )
                    metadata_block = first_m_block // fx.Int32(
                        sorted_block_m // block_m
                    )
                    expert = rocdl.readfirstlane(
                        T.i32,
                        _raw(_global_i32_at(arg_eids, metadata_block)),
                    )
                    tile_count = fx.Int32(1)
                for local_tile in range(0, tile_count, 1):
                    _run_tile(
                        (first_m_block + fx.Int32(local_tile) * tile_stride)
                        * fx.Int32(block_m),
                        n_block,
                        expert,
                        fx.Int32(block_m),
                    )

        elif const_expr(compact_grid):
            descriptor_count = rocdl.readfirstlane(
                T.i32,
                _raw(_global_i32_at(arg_schedule, fx.Int32(0))),
            )
            work_count = descriptor_count * fx.Int32(num_n_blocks)

            def _run_work(work):
                descriptor_index = work // fx.Int32(num_n_blocks)
                n_block = work % fx.Int32(num_n_blocks)
                m_block = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_schedule, descriptor_index + fx.Int32(1))),
                )
                metadata_block = m_block // fx.Int32(sorted_block_m // block_m)
                expert = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_eids, metadata_block)),
                )
                _run_tile(
                    m_block * fx.Int32(block_m),
                    n_block,
                    expert,
                    fx.Int32(block_m),
                )

        else:
            metadata_count = cumsum0 // fx.Int32(sorted_block_m)
            work_count = metadata_count * fx.Int32(num_n_blocks)

            def _run_work(work):
                metadata_block = work // fx.Int32(num_n_blocks)
                n_block = work % fx.Int32(num_n_blocks)
                expert = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_eids, metadata_block)),
                )
                m_block = metadata_block * fx.Int32(sorted_block_m // block_m)
                _run_tile(
                    m_block * fx.Int32(block_m),
                    n_block,
                    expert,
                    fx.Int32(block_m),
                )

        if const_expr(min_active_experts > 0 or max_active_experts is not None):
            active_count = cumsum0
            if const_expr(min_active_experts > 0):
                work_count = (active_count >= fx.Int32(min_active_experts)).select(
                    work_count,
                    fx.Int32(0),
                )
            if const_expr(max_active_experts is not None):
                work_count = (active_count <= fx.Int32(max_active_experts)).select(
                    work_count,
                    fx.Int32(0),
                )

        if block_id < work_count:
            _run_work(block_id)
        for work in range(block_id + grid_size, work_count, gpu.grid_dim.x):
            _run_work(fx.Int32(work))

    if expert_m_reuse_threshold is not None:

        @flyc.jit
        def launch(
            arg_a: fx.Int64,
            arg_b: fx.Int64,
            arg_schedule: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_frequency: fx.Int64,
            arg_out: fx.Int64,
            arg_sorted_token_ids: fx.Int64,
            i32_tokens: fx.Int32,
            i32_grid: fx.Int32,
            stream: fx.Stream,
        ):
            grouped_nn_kernel(
                arg_a,
                arg_b,
                arg_schedule,
                arg_eids,
                arg_cumsum,
                arg_frequency,
                arg_out,
                arg_sorted_token_ids,
                i32_tokens,
            ).launch(
                grid=(fx.Int64(i32_grid), 1, 1),
                block=(block_threads, 1, 1),
                stream=stream,
            )

    elif store_route_slots or store_route_ids or gather_a_rows:

        @flyc.jit
        def launch(
            arg_a: fx.Int64,
            arg_b: fx.Int64,
            arg_schedule: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_out: fx.Int64,
            arg_sorted_token_ids: fx.Int64,
            i32_tokens: fx.Int32,
            i32_grid: fx.Int32,
            stream: fx.Stream,
        ):
            grouped_nn_kernel(
                arg_a,
                arg_b,
                arg_schedule,
                arg_eids,
                arg_cumsum,
                fx.Int64(0),
                arg_out,
                arg_sorted_token_ids,
                i32_tokens,
            ).launch(
                grid=(fx.Int64(i32_grid), 1, 1),
                block=(block_threads, 1, 1),
                stream=stream,
            )

    else:

        @flyc.jit
        def launch(
            arg_a: fx.Int64,
            arg_b: fx.Int64,
            arg_schedule: fx.Int64,
            arg_eids: fx.Int64,
            arg_cumsum: fx.Int64,
            arg_out: fx.Int64,
            i32_grid: fx.Int32,
            stream: fx.Stream,
        ):
            grouped_nn_kernel(
                arg_a,
                arg_b,
                arg_schedule,
                arg_eids,
                arg_cumsum,
                fx.Int64(0),
                arg_out,
                fx.Int64(0),
                fx.Int32(0),
            ).launch(
                grid=(fx.Int64(i32_grid), 1, 1),
                block=(block_threads, 1, 1),
                stream=stream,
            )

    return launch


__all__ = ["compile_sonic_grouped_a16w16_nn"]

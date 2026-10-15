# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""Device-driven grouped BF16 dA GEMM for gfx950 SonicMoE backward.

Computes ``dA[sorted, I] = dY[sorted, H] @ W2[expert, H, I]`` directly from
the public row-major W2 layout.  The implementation specializes the proven
NN path from :mod:`kernels.gemm.gemm_a16w16_gfx950`: both operands use async
global-to-LDS loads, while row-major B is transposed in LDS and consumed by
CDNA4 ``LDSReadTrans16_64b`` operations.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import buffer_ops

from .gemm_utils import (
    GFX950_DMA_BYTES,
    GFX950_WAVE_SIZE,
    __barrier,
    async_load_to_lds,
    get_wave_lds_offset,
    make_lds_layout,
    make_transposed_lds_layout,
)
from .stage_utils import _global_i32_at, _raw


def _global_bf16_ptr(addr_i64):
    ptr_ty = fx.PointerType.get(
        fx.BFloat16.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=GFX950_DMA_BYTES,
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _grouped_da_body(
    smem_a,
    smem_b,
    smem_c,
    tiled_mma,
    arg_dy,
    arg_w2,
    arg_da,
    expert,
    first_sorted_row,
    m_block_idx,
    n_block_idx,
    frequency,
    *,
    BM,
    BN,
    BK,
    HIDDEN,
    INTER,
    STAGES,
    M_WAVES,
    N_WAVES,
):
    """Run one expert-local ``BM x BN`` output tile."""

    elem_dtype = fx.BFloat16
    elem_bytes = 2
    block_threads = M_WAVES * N_WAVES * GFX950_WAVE_SIZE
    async_load_vec_size = GFX950_DMA_BYTES // elem_bytes
    ldg_x_threads = BK // async_load_vec_size
    ldg_a_iters = (BM * BK) // (block_threads * async_load_vec_size)
    ldg_b_iters = (BN * BK) // (block_threads * async_load_vec_size)
    ldg_wait_count = ldg_a_iters + ldg_b_iters
    mma_m_iters = BM // (M_WAVES * 16)
    mma_n_iters = BN // (N_WAVES * 16)
    k_mma_iters = BK // 32
    k_tiles = HIDDEN // BK
    cshuffle_vec_size = 8

    tid = fx.Int32(gpu.thread_id("x"))
    local_m_offset = m_block_idx * fx.Int32(BM)
    local_n_offset = n_block_idx * fx.Int32(BN)
    padded_rows = (frequency + fx.Int32(BM - 1)) // fx.Int32(BM) * fx.Int32(BM)

    sorted_byte_offset = fx.Int64(first_sorted_row) * fx.Int64(HIDDEN * elem_bytes)
    output_byte_offset = fx.Int64(first_sorted_row) * fx.Int64(INTER * elem_bytes)
    weight_byte_offset = fx.Int64(expert) * fx.Int64(HIDDEN * INTER * elem_bytes)
    dy_addr = fx.Int64(arg_dy) + sorted_byte_offset
    w2_addr = fx.Int64(arg_w2) + weight_byte_offset
    da_addr = fx.Int64(arg_da) + output_byte_offset

    dy_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(dy_addr),
        num_records_bytes=_raw(fx.Int64(padded_rows) * fx.Int64(HIDDEN * elem_bytes)),
    )
    w2_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(w2_addr),
        num_records_bytes=HIDDEN * INTER * elem_bytes,
    )
    da_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(da_addr),
        num_records_bytes=_raw(fx.Int64(padded_rows) * fx.Int64(INTER * elem_bytes)),
    )

    a_lds_layout = make_lds_layout(BM, BK)
    # Logical B is row-major [HIDDEN, INTER].  The K-major global load is
    # transposed in LDS for the MFMA B operand.
    b_lds_layout = make_transposed_lds_layout(BN, BK)
    c_lds_layout = fx.make_layout((BM, BN), (BN, 1))

    uni_copy_atom = fx.make_copy_atom(fx.UniversalCopy128b(), elem_dtype)
    buffer_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_dtype)
    a_s2r_copy_atom = uni_copy_atom
    a_tiled_copy_atom = buffer_copy_atom
    b_s2r_copy_atom = fx.make_copy_atom(fx.rocdl.cdna4.LDSReadTrans16_64b(), elem_dtype)
    b_tiled_copy_atom = b_s2r_copy_atom

    sA = fx.make_view(smem_a, a_lds_layout)
    sB = fx.make_view(smem_b, b_lds_layout)
    sC = fx.make_view(smem_c, c_lds_layout)
    thr_mma = tiled_mma.thr_slice(tid)
    thr_copy_A = fx.make_tiled_copy_A(a_tiled_copy_atom, tiled_mma).get_slice(tid)
    thr_copy_B = fx.make_tiled_copy_B(b_tiled_copy_atom, tiled_mma).get_slice(tid)
    frag_A = thr_mma.make_fragment_A(sA)
    frag_B = thr_mma.make_fragment_B(sB)

    # C's register ownership depends on the tile shape, not its address.  A
    # compact fake view avoids constructing a dynamic global tensor descriptor.
    c_fake_ptr = _global_bf16_ptr(da_addr)
    c_fake_buf = fx.rocdl.make_buffer_tensor(
        fx.make_view(c_fake_ptr, fx.make_layout((BM, BN), (BN, 1))),
        max_size=False,
    )
    frag_C = thr_mma.make_fragment_C(c_fake_buf)
    frag_A_retile = thr_copy_A.retile(frag_A)
    frag_B_retile = thr_copy_B.retile(frag_B)
    frag_C.fill(0.0)

    row_coords = fx.make_view(0, fx.make_layout((BM, BN), (1, 0)))
    col_coords = fx.make_view(0, fx.make_layout((BM, BN), (0, 1)))
    thr_mma_crow = thr_mma.partition_C(row_coords)
    thr_mma_ccol = thr_mma.partition_C(col_coords)

    wave_offset = get_wave_lds_offset(tid, GFX950_DMA_BYTES)
    async_context = (
        wave_offset,
        tid,
        block_threads,
        async_load_vec_size,
        ldg_x_threads,
        0,
        BK,
        elem_bytes,
        GFX950_DMA_BYTES,
    )

    def async_load_a_to_lds(k_tile, stage):
        async_load_to_lds(
            smem_a + stage * BM * BK,
            dy_rsrc,
            a_lds_layout,
            BM,
            frequency,
            local_m_offset,
            HIDDEN,
            ldg_a_iters,
            False,
            k_tile,
            async_context,
        )

    def async_load_b_to_lds(k_tile, stage):
        async_load_to_lds(
            smem_b + stage * BN * BK,
            w2_rsrc,
            b_lds_layout,
            BN,
            INTER,
            local_n_offset,
            INTER,
            ldg_b_iters,
            True,
            k_tile,
            async_context,
        )

    def compute_stage(read_stage):
        thr_sA_s2r = thr_copy_A.partition_S(
            fx.make_view(smem_a + read_stage * BM * BK, a_lds_layout)
        )
        thr_sB_s2r = thr_copy_B.partition_S(
            fx.make_view(smem_b + read_stage * BN * BK, b_lds_layout)
        )
        for k_iter in range_constexpr(k_mma_iters):
            fx.copy(
                b_s2r_copy_atom,
                thr_sB_s2r[None, None, k_iter],
                frag_B_retile[None, None, k_iter],
            )
            fx.copy(
                a_s2r_copy_atom,
                thr_sA_s2r[None, None, k_iter],
                frag_A_retile[None, None, k_iter],
            )
            fx.gemm(
                tiled_mma,
                frag_C,
                frag_A[None, None, k_iter],
                frag_B[None, None, k_iter],
                frag_C,
                traversal_order=fx.GemmTraversalOrder.KNM,
            )

    for stage in range_constexpr(STAGES - 1):
        async_load_b_to_lds(stage, stage)
        async_load_a_to_lds(stage, stage)
    rocdl.sched_barrier(0)

    main_loop_end = k_tiles - (STAGES - 1)
    for k_tile in range_constexpr(main_loop_end):
        current_stage = k_tile % STAGES
        write_stage = (current_stage + STAGES - 1) % STAGES
        __barrier((STAGES - 2) * ldg_wait_count)
        async_load_b_to_lds(k_tile + (STAGES - 1), write_stage)
        async_load_a_to_lds(k_tile + (STAGES - 1), write_stage)
        compute_stage(current_stage)
        rocdl.sched_vmem(ldg_wait_count)
        for _ in range_constexpr(k_mma_iters):
            rocdl.sched_dsrd(mma_n_iters)
            rocdl.sched_dsrd(mma_m_iters)
            for _ in range_constexpr(mma_m_iters):
                rocdl.sched_mfma(mma_n_iters)
        rocdl.sched_barrier(0)

    current_stage = main_loop_end % STAGES
    for tail_stage in range_constexpr(STAGES - 1):
        __barrier((STAGES - 2 - tail_stage) * ldg_wait_count)
        compute_stage(current_stage)
        current_stage = (current_stage + 1) % STAGES

    frag_C_out = fx.make_fragment_like(frag_C, elem_dtype)
    frag_C_out.store(frag_C.load().to(elem_dtype))
    gpu.barrier()
    for i in range_constexpr(fx.size(frag_C_out.shape).unpack()):
        row = fx.get_scalar(thr_mma_crow[i])
        col = fx.get_scalar(thr_mma_ccol[i])
        sC[row, col] = frag_C_out[i]
    gpu.barrier()

    cshuffle_x_threads = BN // cshuffle_vec_size
    cshuffle_vectors = BM * BN // cshuffle_vec_size
    cshuffle_iters = (cshuffle_vectors + block_threads - 1) // block_threads
    for i in range_constexpr(cshuffle_iters):
        vector_idx = block_threads * i + tid
        valid_vector = vector_idx < fx.Int32(cshuffle_vectors)
        # Keep this helper traceable when the tile contains fewer store vectors
        # than workgroup threads.  Invalid lanes safely reload sC[0, 0] and are
        # suppressed by the buffer-store predicate.
        safe_vector_idx = valid_vector.select(vector_idx, fx.Int32(0))
        local_row = safe_vector_idx // fx.Int32(cshuffle_x_threads)
        local_col = (
            safe_vector_idx % fx.Int32(cshuffle_x_threads) * fx.Int32(cshuffle_vec_size)
        )
        global_row = local_m_offset + local_row
        global_col = local_n_offset + local_col
        c_vec = fx.ptr_load(
            smem_c + local_row * fx.Int32(BN) + local_col,
            result_type=fx.Vector.make_type(cshuffle_vec_size, elem_dtype),
        )
        output_offset = global_row * fx.Int32(INTER) + global_col
        buffer_ops.buffer_store(
            c_vec,
            da_rsrc,
            output_offset,
            mask=valid_vector & (global_row < frequency),
        )

    # The expert-grid loop may immediately reuse the unioned A/B/C storage.
    gpu.barrier()


def compile_grouped_da_gfx950(
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    sorted_block_size: int = 64,
    block_m: int = 16,
    block_n: int = 128,
    block_k: int = 128,
    stages: int = 2,
    m_waves: int = 1,
    n_waves: int = 4,
    waves_per_eu: int | None = None,
    queue_direct: bool = False,
    persistent: bool = False,
    min_active_experts: int = 0,
    max_active_experts: int | None = None,
):
    """Compile a grouped BF16 ``dY @ W2`` launcher for gfx950.

    ``queue_direct=True`` consumes the counter-first
    ``[count, (expert, first_row)...]`` active queue through
    ``arg_active_count`` and skips the expert-grid scan and metadata binary
    search.  The optional inclusive active-expert interval is evaluated from
    the same device pointer.  Disjoint specializations can consequently select
    BM32 for balanced routing and BM64 for hot routing without a host readback.
    """

    if min(hidden_size, intermediate_size, num_experts) <= 0:
        raise ValueError(
            "hidden_size, intermediate_size, and num_experts must be positive"
        )
    if not isinstance(min_active_experts, int) or min_active_experts < 0:
        raise ValueError("min_active_experts must be a non-negative int")
    if not isinstance(persistent, bool):
        raise TypeError(f"persistent must be bool, got {type(persistent).__name__}")
    if queue_direct and persistent:
        raise ValueError("persistent is implicit for queue_direct profiles")
    if max_active_experts is not None and (
        not isinstance(max_active_experts, int)
        or max_active_experts < min_active_experts
    ):
        raise ValueError(
            "max_active_experts must be None or at least min_active_experts"
        )
    if block_m % (m_waves * 16) != 0:
        raise ValueError("block_m must be divisible by m_waves * 16")
    if block_n % (n_waves * 16) != 0:
        raise ValueError("block_n must be divisible by n_waves * 16")
    if block_k % 32 != 0 or hidden_size % block_k != 0:
        raise ValueError("block_k must be a multiple of 32 and divide hidden_size")
    if intermediate_size % block_n != 0:
        raise ValueError("block_n must divide intermediate_size")
    if sorted_block_size <= 0:
        raise ValueError("sorted_block_size must be positive")
    if stages < 2 or hidden_size // block_k < stages - 1:
        raise ValueError("the staged pipeline requires at least stages - 1 K tiles")

    block_threads = m_waves * n_waves * GFX950_WAVE_SIZE
    if block_threads > 1024:
        raise ValueError("the workgroup cannot contain more than 1024 threads")
    async_load_vec_size = GFX950_DMA_BYTES // 2
    ldg_x_threads = block_k // async_load_vec_size
    if (
        ldg_x_threads * async_load_vec_size != block_k
        or block_threads % ldg_x_threads != 0
    ):
        raise ValueError("block_k does not produce an exact async-load thread layout")
    for name, tile_elements in (("A", block_m * block_k), ("B", block_n * block_k)):
        if tile_elements % (block_threads * async_load_vec_size) != 0:
            raise ValueError(
                f"{name} tile is not exactly covered by 16-byte workgroup loads"
            )

    ab_elements = stages * (block_m + block_n) * block_k
    c_elements = block_m * block_n
    lds_bytes = max(ab_elements, c_elements) * 2
    if lds_bytes > 163840:
        raise ValueError(
            f"grouped dA requires {lds_bytes} LDS bytes, exceeding gfx950 capacity"
        )

    num_n_blocks = intermediate_size // block_n
    persistent_suffix = "_p1" if persistent else ""
    name = (
        f"grouped_da_bf16_gfx950_h{hidden_size}_i{intermediate_size}_e{num_experts}"
        f"_bm{block_m}_bn{block_n}_bk{block_k}_s{stages}_w{m_waves}x{n_waves}"
        f"_q{int(queue_direct)}"
        f"{persistent_suffix}"
        f"_amin{min_active_experts}_amax{max_active_experts}"
    )

    @fx.struct
    class SharedABStorage:
        a: fx.Array[fx.BFloat16, stages * block_m * block_k, 16]
        b: fx.Array[fx.BFloat16, stages * block_n * block_k, 16]

    @fx.union
    class SharedStorage:
        ab: SharedABStorage
        c: fx.Array[fx.BFloat16, block_m * block_n, 16]

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def grouped_da_kernel(
        arg_dy: fx.Int64,
        arg_w2: fx.Int64,
        arg_frequency: fx.Int64,
        arg_expert_ids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_active_count: fx.Int64,
        arg_da: fx.Int64,
    ):
        storage = fx.SharedAllocator().allocate(SharedStorage)
        smem_a = storage.ab.a.peek().ptr
        smem_b = storage.ab.b.peek().ptr
        smem_c = storage.c.peek().ptr
        block_id = fx.Int32(gpu.block_id("x"))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))

        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
        tiled_mma = fx.make_tiled_mma(
            mma_atom,
            fx.make_layout((m_waves, n_waves, 1), (n_waves, 1, 0)),
            fx.make_tile(None, None, fx.make_layout((8, 4), (1, 8))),
        )

        active_count = fx.Int32(0)
        if const_expr(
            queue_direct or min_active_experts > 0 or max_active_experts is not None
        ):
            active_count = rocdl.readfirstlane(
                T.i32,
                _raw(_global_i32_at(arg_active_count, fx.Int32(0))),
            )

        def run_expert_tile(expert, n_block, first_sorted_row, frequency):
            num_m_blocks = (frequency + fx.Int32(block_m - 1)) // fx.Int32(block_m)
            for m_block in range(0, num_m_blocks, 1):
                _grouped_da_body(
                    smem_a,
                    smem_b,
                    smem_c,
                    tiled_mma,
                    arg_dy,
                    arg_w2,
                    arg_da,
                    expert,
                    first_sorted_row,
                    fx.Int32(m_block),
                    n_block,
                    frequency,
                    BM=block_m,
                    BN=block_n,
                    BK=block_k,
                    HIDDEN=hidden_size,
                    INTER=intermediate_size,
                    STAGES=stages,
                    M_WAVES=m_waves,
                    N_WAVES=n_waves,
                )

        if const_expr(queue_direct):
            work_bound = active_count * fx.Int32(num_n_blocks)
            if const_expr(min_active_experts > 0):
                work_bound = (active_count >= fx.Int32(min_active_experts)).select(
                    work_bound,
                    fx.Int32(0),
                )
            if const_expr(max_active_experts is not None):
                work_bound = (active_count <= fx.Int32(max_active_experts)).select(
                    work_bound,
                    fx.Int32(0),
                )

            def run_queue_work(work):
                descriptor = work // fx.Int32(num_n_blocks)
                n_block = work % fx.Int32(num_n_blocks)
                descriptor_offset = fx.Int32(1) + descriptor * fx.Int32(2)
                expert = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_active_count, descriptor_offset)),
                )
                first_sorted_row = rocdl.readfirstlane(
                    T.i32,
                    _raw(
                        _global_i32_at(
                            arg_active_count, descriptor_offset + fx.Int32(1)
                        )
                    ),
                )
                frequency = rocdl.readfirstlane(
                    T.i32,
                    _raw(_global_i32_at(arg_frequency, expert)),
                )
                run_expert_tile(expert, n_block, first_sorted_row, frequency)

            grid_size = fx.Int32(gpu.grid_dim.x)
            if block_id < work_bound:
                run_queue_work(block_id)
            for work in range(block_id + grid_size, work_bound, grid_size):
                run_queue_work(fx.Int32(work))
        else:
            expert_bound = fx.Int32(num_experts * num_n_blocks)
            if const_expr(min_active_experts > 0):
                expert_bound = (active_count >= fx.Int32(min_active_experts)).select(
                    expert_bound,
                    fx.Int32(0),
                )
            if const_expr(max_active_experts is not None):
                expert_bound = (active_count <= fx.Int32(max_active_experts)).select(
                    expert_bound,
                    fx.Int32(0),
                )

            def run_expert_work(work):
                expert = work // fx.Int32(num_n_blocks)
                n_block = work % fx.Int32(num_n_blocks)
                frequency = rocdl.readfirstlane(
                    T.i32, _raw(_global_i32_at(arg_frequency, expert))
                )
                if frequency > fx.Int32(0):
                    lo = fx.Int32(0)
                    hi = cumsum0 // fx.Int32(sorted_block_size)
                    for _ in range_constexpr(25):
                        searching = lo < hi
                        mid = (lo + hi) // fx.Int32(2)
                        safe_mid = searching.select(mid, fx.Int32(0))
                        mid_expert = fx.Int32(_global_i32_at(arg_expert_ids, safe_mid))
                        move_right = searching & (mid_expert < expert)
                        lo = move_right.select(mid + fx.Int32(1), lo)
                        move_left = searching & (mid_expert >= expert)
                        hi = move_left.select(mid, hi)
                    first_sorted_row = lo * fx.Int32(sorted_block_size)
                    run_expert_tile(expert, n_block, first_sorted_row, frequency)

            if const_expr(persistent):
                # A capped persistent launch avoids paying the full expert
                # grid for a device-guarded profile that is inactive.  The
                # body ends in a workgroup barrier, so shared A/B/C storage is
                # safe to reuse on each grid-stride iteration.
                grid_size = fx.Int32(gpu.grid_dim.x)
                if block_id < expert_bound:
                    run_expert_work(block_id)
                for work in range(block_id + grid_size, expert_bound, grid_size):
                    run_expert_work(fx.Int32(work))
            else:
                if block_id < expert_bound:
                    run_expert_work(block_id)

    @flyc.jit
    def launch_grouped_da(
        arg_dy: fx.Int64,
        arg_w2: fx.Int64,
        arg_frequency: fx.Int64,
        arg_expert_ids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_active_count: fx.Int64,
        arg_da: fx.Int64,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        grouped_da_kernel(
            arg_dy,
            arg_w2,
            arg_frequency,
            arg_expert_ids,
            arg_cumsum,
            arg_active_count,
            arg_da,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(
            grid=(fx.Int64(i32_grid), 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_grouped_da

# SPDX-License-Identifier: MIT

"""Multi-wave BlockMFMA BF16 decode GEMM policy."""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import arith as arith_dialect
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, T
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr

from .gemm_decode_config import (
    ActivationSource,
    BlockMfmaDecodeConfig,
    MFMA_K,
    WAVE_SIZE,
    block_mfma_lds_bytes,
    block_mfma_staged_k,
    gemm_decode_kernel_name,
    validate_block_mfma_grid_i32,
)
from .gemm_decode_numeric import (
    bf16x4_slice,
    convert_bf16,
    masked_bf16_vector,
    mfma_4x4x4_bf16,
    raw,
    reduce_mfma_scalar,
)
from .tensor_shim import GTensor, STensor


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _accumulate_vectors(
    accumulators,
    a_vectors,
    b_vectors,
    *,
    m: int,
    columns: int,
    width: int,
):
    for fragment in range_constexpr(width // MFMA_K):
        for column in range_constexpr(columns):
            b_fragment = bf16x4_slice(b_vectors[column], fragment)
            for row in range_constexpr(m):
                index = row * columns + column
                accumulators[index] = mfma_4x4x4_bf16(
                    bf16x4_slice(a_vectors[row], fragment),
                    b_fragment,
                    accumulators[index],
                )
    return accumulators


def _load_full_a_vectors(
    *,
    use_lds: bool,
    a_global,
    a_smem,
    k_base,
    m: int,
    k: int,
    staged_k: int,
    width: int,
):
    if use_lds:
        return [
            a_smem.vec_load(
                (fx.Index(fx.Int32(row * staged_k) + k_base),),
                width,
            )
            for row in range_constexpr(m)
        ]
    return [
        a_global.vec_load((row, k_base), width)
        for row in range_constexpr(m)
    ]


def _load_tail_a_vectors(
    *,
    use_lds: bool,
    a_global,
    a_smem,
    k_base,
    m: int,
    k: int,
    staged_k: int,
):
    if use_lds:
        values = []
        for row in range_constexpr(m):
            valid = k_base < fx.Int32(k)
            safe_k = ArithValue(raw(valid)).select(k_base, fx.Int32(k))
            values.append(
                a_smem.vec_load(
                    (fx.Index(fx.Int32(row * staged_k) + safe_k),),
                    MFMA_K,
                )
            )
        return values
    return [
        masked_bf16_vector(
            a_global.rsrc,
            fx.Int32(row * k),
            k_base,
            MFMA_K,
            k,
        )
        for row in range_constexpr(m)
    ]


def _compiler_memory_barrier() -> None:
    """Keep staged-A reads inside a runtime persistent-N loop."""
    llvm_dialect.inline_asm(
        None,
        [],
        "",
        "~{memory}",
        has_side_effects=True,
    )


def _compute_column_tile(
    *,
    config: BlockMfmaDecodeConfig,
    use_lds: bool,
    a_global,
    a_smem,
    b_global,
    c_global,
    column_base,
    lane,
    m: int,
    n: int,
    k: int,
    staged_k: int,
    width: int,
    full_chunks: int,
    tail_tiles: int,
    schedule_depth: int,
    schedule_batches: int,
) -> None:
    """Compute one logical N tile with the existing BlockMFMA math."""
    columns = config.columns_per_wave
    safe_columns = []
    column_valid = []
    for column_i in range_constexpr(columns):
        column = column_base + fx.Int32(column_i)
        valid = column < fx.Int32(n)
        safe_columns.append(ArithValue(raw(valid)).select(column, fx.Int32(0)))
        column_valid.append(valid)

    accumulator_zero = arith.constant_vector(0.0, T.vec(4, T.f32))
    accumulators = [accumulator_zero] * (m * columns)
    for batch in range_constexpr(schedule_batches):
        chunks = min(
            schedule_depth,
            full_chunks - batch * schedule_depth,
        )
        prefetched_a = []
        prefetched_b = []
        for stage in range_constexpr(chunks):
            chunk = batch * schedule_depth + stage
            k_base = (
                fx.Int32(chunk * WAVE_SIZE * width)
                + lane * fx.Int32(width)
            )
            prefetched_a.append(
                _load_full_a_vectors(
                    use_lds=use_lds,
                    a_global=a_global,
                    a_smem=a_smem,
                    k_base=k_base,
                    m=m,
                    k=k,
                    staged_k=staged_k,
                    width=width,
                )
            )
            prefetched_b.append(
                [
                    b_global.vec_load(
                        (safe_columns[column], k_base),
                        width,
                    )
                    for column in range_constexpr(columns)
                ]
            )
        for stage in range_constexpr(chunks):
            accumulators = _accumulate_vectors(
                accumulators,
                prefetched_a[stage],
                prefetched_b[stage],
                m=m,
                columns=columns,
                width=width,
            )

    for tail in range_constexpr(tail_tiles):
        k_base = (
            fx.Int32(
                full_chunks * WAVE_SIZE * width
                + tail * WAVE_SIZE * MFMA_K
            )
            + lane * fx.Int32(MFMA_K)
        )
        a_vectors = _load_tail_a_vectors(
            use_lds=use_lds,
            a_global=a_global,
            a_smem=a_smem,
            k_base=k_base,
            m=m,
            k=k,
            staged_k=staged_k,
        )
        b_vectors = [
            masked_bf16_vector(
                b_global.rsrc,
                safe_columns[column] * fx.Int32(k),
                k_base,
                MFMA_K,
                k,
                config.b_cache_modifier,
            )
            for column in range_constexpr(columns)
        ]
        accumulators = _accumulate_vectors(
            accumulators,
            a_vectors,
            b_vectors,
            m=m,
            columns=columns,
            width=MFMA_K,
        )

    reduced = [
        reduce_mfma_scalar(accumulator)
        for accumulator in accumulators
    ]
    for row in range_constexpr(m):
        for column in range_constexpr(columns):
            store_valid = (
                lane == fx.Int32(WAVE_SIZE - 1)
            ) & column_valid[column]
            store_if = scf.IfOp(
                raw(store_valid),
                results_=[],
                has_else=False,
            )
            with ir.InsertionPoint(store_if.then_block):
                output = convert_bf16(
                    reduced[row * columns + column],
                    fx.Int32(row * n)
                    + column_base
                    + fx.Int32(column),
                    config.output_rounding,
                )
                c_global[
                    row,
                    column_base + fx.Int32(column),
                ] = output
                scf.YieldOp([])


def _make_block_kernel(
    *,
    arch: str,
    m: int,
    n: int,
    k: int,
    config: BlockMfmaDecodeConfig,
    kernel_name: str,
    allocator,
    activation_offset: int,
    persistent_turns: int,
):
    """Build the already-selected architecture body without ambient detection."""
    waves = config.waves_per_workgroup
    columns = config.columns_per_wave
    width = config.b_load_width
    block_threads = waves * WAVE_SIZE
    staged_k = block_mfma_staged_k(k)
    full_chunks = k // (WAVE_SIZE * width)
    tail_tiles = _ceil_div(k % (WAVE_SIZE * width), WAVE_SIZE * MFMA_K)
    schedule_depth = config.k_unroll * config.prefetch_stages
    schedule_batches = _ceil_div(full_chunks, schedule_depth)
    use_lds = config.activation_source == ActivationSource.FULL_LDS
    activation_vectors_per_row = k // 8
    activation_vectors = m * activation_vectors_per_row
    activation_tail_per_row = staged_k - activation_vectors_per_row * 8
    activation_tail_elements = m * activation_tail_per_row
    stage_iterations = _ceil_div(activation_vectors, block_threads)

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def block_mfma_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        runtime_grid_workgroups: Int32,
        runtime_persistent_turns: Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        wave = tid // fx.Int32(WAVE_SIZE)
        lane = tid % fx.Int32(WAVE_SIZE)
        first_column = (
            gpu.block_idx.x * fx.Int32(waves * columns)
            + wave * fx.Int32(columns)
        )
        a_global = GTensor(A, dtype=T.bf16, shape=(m, k))
        b_global = GTensor(
            B,
            dtype=T.bf16,
            shape=(n, k),
            cache_modifier=config.b_cache_modifier,
        )
        c_global = GTensor(C, dtype=T.bf16, shape=(m, n))
        a_smem = None
        if use_lds:
            a_smem_ptr = SmemPtr(
                allocator.get_base(),
                activation_offset,
                T.bf16,
                shape=(m * staged_k,),
            )
            a_smem = STensor(
                a_smem_ptr,
                T.bf16,
                shape=(m * staged_k,),
            )
            for iteration in range_constexpr(stage_iterations):
                slot = tid + fx.Int32(iteration * block_threads)
                slot_valid = slot < fx.Int32(activation_vectors)
                safe_slot = ArithValue(raw(slot_valid)).select(slot, fx.Int32(0))
                row = safe_slot // fx.Int32(activation_vectors_per_row)
                row_vector = safe_slot % fx.Int32(activation_vectors_per_row)
                column = row_vector * fx.Int32(8)
                stage_if = scf.IfOp(
                    raw(slot_valid),
                    results_=[],
                    has_else=False,
                )
                with ir.InsertionPoint(stage_if.then_block):
                    staged = a_global.vec_load((row, column), 8)
                    a_smem.vec_store(
                        (fx.Index(row * fx.Int32(staged_k) + column),),
                        staged,
                        8,
                    )
                    scf.YieldOp([])
            if activation_tail_elements:
                tail_if = scf.IfOp(
                    raw(tid < fx.Int32(activation_tail_elements)),
                    results_=[],
                    has_else=False,
                )
                with ir.InsertionPoint(tail_if.then_block):
                    row = tid // fx.Int32(activation_tail_per_row)
                    row_tail = tid % fx.Int32(activation_tail_per_row)
                    column = (
                        fx.Int32(activation_vectors_per_row * 8) + row_tail
                    )
                    valid = column < fx.Int32(k)
                    safe_column = ArithValue(raw(valid)).select(
                        column,
                        fx.Int32(0),
                    )
                    loaded = a_global[row, safe_column]
                    zero = arith.constant(0.0, type=T.bf16)
                    staged = ArithValue(raw(valid)).select(loaded, zero)
                    a_smem_ptr.store(
                        staged,
                        [fx.Index(row * fx.Int32(staged_k) + column)],
                    )
                    scf.YieldOp([])
            gpu.barrier()

        tile_kwargs = dict(
            config=config,
            use_lds=use_lds,
            a_global=a_global,
            a_smem=a_smem,
            b_global=b_global,
            c_global=c_global,
            lane=lane,
            m=m,
            n=n,
            k=k,
            staged_k=staged_k,
            width=width,
            full_chunks=full_chunks,
            tail_tiles=tail_tiles,
            schedule_depth=schedule_depth,
            schedule_batches=schedule_batches,
        )
        if persistent_turns == 1:
            _compute_column_tile(column_base=first_column, **tile_kwargs)
        else:
            column_stride = (
                runtime_grid_workgroups * fx.Int32(waves * columns)
            )
            turn = fx.Int32(0)
            while turn < runtime_persistent_turns:
                column_base = first_column + turn * column_stride
                _compiler_memory_barrier()
                if column_base < fx.Int32(n):
                    _compute_column_tile(column_base=column_base, **tile_kwargs)
                turn = turn + fx.Int32(1)

    block_mfma_kernel.decode_arch = arch
    return block_mfma_kernel


def _make_gfx942_block_kernel(**kwargs):
    return _make_block_kernel(arch="gfx942", **kwargs)


def _make_gfx950_block_kernel(**kwargs):
    return _make_block_kernel(arch="gfx950", **kwargs)


@functools.lru_cache(maxsize=2048)
def compile_gemm_decode_block_mfma_bf16(
    m: int,
    n: int,
    k: int,
    config: BlockMfmaDecodeConfig,
    arch: str,
    num_cus: int | None = None,
):
    """Compile one work-sized or grid-capped persistent BlockMFMA kernel."""
    config.validate(m=m, n=n, k=k, arch=arch)
    kernel_name = gemm_decode_kernel_name(arch, m, n, k, config)
    waves = config.waves_per_workgroup
    columns = config.columns_per_wave
    block_threads = waves * WAVE_SIZE
    logical_workgroups = _ceil_div(n, waves * columns)
    if config.persistent_n:
        if num_cus is None or num_cus <= 0:
            raise ValueError("N-persistent BlockMFMA requires a positive num_cus")
        grid_workgroups, persistent_turns, _ = validate_block_mfma_grid_i32(
            n,
            config,
            num_cus=num_cus,
        )
    else:
        grid_workgroups = logical_workgroups
        persistent_turns = 1
    use_lds = config.activation_source == ActivationSource.FULL_LDS
    allocator = None
    activation_offset = 0
    if use_lds:
        allocator = SmemAllocator(
            None,
            arch=arch,
            global_sym_name=f"{kernel_name}_smem",
        )
        activation_offset = allocator._align(allocator.ptr, 16)
        allocator.ptr = activation_offset + block_mfma_lds_bytes(m, k)

    factory = (
        _make_gfx942_block_kernel
        if arch == "gfx942"
        else _make_gfx950_block_kernel
    )
    kernel = factory(
        m=m,
        n=n,
        k=k,
        config=config,
        kernel_name=kernel_name,
        allocator=allocator,
        activation_offset=activation_offset,
        persistent_turns=persistent_turns,
    )
    cache_tag = (
        kernel_name,
        grid_workgroups,
        persistent_turns,
    )

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = cache_tag
        if use_lds:
            allocator.finalized = False
            context = CompilationContext.get_current()
            with ir.InsertionPoint(context.gpu_module_body):
                allocator.finalize()
        attributes = {}
        if config.waves_per_eu:
            attributes["rocdl.waves_per_eu"] = config.waves_per_eu
        kernel(
            A,
            B,
            C,
            fx.Int32(grid_workgroups),
            fx.Int32(persistent_turns),
            value_attrs=attributes,
        ).launch(
            grid=(grid_workgroups, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    launch.lds_bytes = block_mfma_lds_bytes(m, k) if use_lds else 0
    launch.grid_workgroups = grid_workgroups
    launch.policy = "block_mfma"
    launch.persistent_turns = persistent_turns
    return launch

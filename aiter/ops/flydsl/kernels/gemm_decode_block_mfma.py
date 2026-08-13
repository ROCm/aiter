# SPDX-License-Identifier: MIT

"""Multi-wave BlockMFMA BF16 decode GEMM policy."""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import Int32, T

from .gemm_decode_common import (
    ActivationSource,
    BlockMfmaDecodeConfig,
    MFMA_K,
    WAVE_SIZE,
    add_bias_f32,
    bf16x4_slice,
    block_mfma_staged_k,
    convert_bf16,
    gemm_decode_kernel_name,
    k_element,
    load_vector,
    make_buffer_matrix,
    make_buffer_vector,
    make_decode_cache_tag,
    make_vector_view,
    masked_bf16_vector,
    mfma_4x4x4_bf16,
    padded_row_coordinates,
    raw,
    reduce_mfma_scalar,
    validate_block_mfma_grid_i32,
    wave_lane_coordinates,
)
from .tensor_shim import _run_compiled


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class BlockMfmaGeometry:
    """Compile-time geometry and schedule for one BlockMFMA specialization."""

    m: int
    n: int
    k: int
    staged_k: int
    width: int
    full_chunks: int
    tail_tiles: int
    schedule_depth: int
    schedule_batches: int
    use_lds: bool


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
                accumulators[row][column] = mfma_4x4x4_bf16(
                    bf16x4_slice(a_vectors[row], fragment),
                    b_fragment,
                    accumulators[row][column],
                )
    return accumulators


def _load_full_a_vectors(
    *,
    use_lds: bool,
    a_global,
    a_smem,
    k_base,
    m: int,
    row_stride: int,
    width: int,
):
    return [
        load_vector(
            a_smem if use_lds else a_global,
            fx.Int32(row),
            k_base,
            row_stride,
            width,
        )
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
                load_vector(
                    a_smem,
                    fx.Int32(row),
                    safe_k,
                    staged_k,
                    MFMA_K,
                )
            )
        return values
    return [
        masked_bf16_vector(
            a_global,
            fx.Int32(row),
            k_base,
            MFMA_K,
            k,
        )
        for row in range_constexpr(m)
    ]


def _compute_column_tile(
    *,
    config: BlockMfmaDecodeConfig,
    geometry: BlockMfmaGeometry,
    a_global,
    a_smem,
    b_global,
    column_base,
    lane,
):
    """Compute one logical N tile and return values for guarded stores."""
    columns = config.columns_per_wave
    logical_columns = [
        column_base + fx.Int32(column)
        for column in range_constexpr(columns)
    ]
    safe_columns = []
    for column in range_constexpr(columns):
        column_coord = logical_columns[column]
        valid = column_coord < fx.Int32(geometry.n)
        safe_columns.append(
            ArithValue(raw(valid)).select(column_coord, fx.Int32(0))
        )
    accumulator_zero = arith.constant_vector(0.0, T.vec(4, T.f32))
    accumulators = [
        [accumulator_zero for _ in range_constexpr(columns)]
        for _ in range_constexpr(geometry.m)
    ]
    for batch in range_constexpr(geometry.schedule_batches):
        chunks = min(
            geometry.schedule_depth,
            geometry.full_chunks - batch * geometry.schedule_depth,
        )
        prefetched_a = []
        prefetched_b = []
        for stage in range_constexpr(chunks):
            chunk = batch * geometry.schedule_depth + stage
            k_base = k_element(
                fx.Int32(chunk),
                lane,
                fx.Int32(0),
                geometry.full_chunks,
                geometry.width,
            )
            prefetched_a.append(
                _load_full_a_vectors(
                    use_lds=geometry.use_lds,
                    a_global=a_global,
                    a_smem=a_smem,
                    k_base=k_base,
                    m=geometry.m,
                    row_stride=(
                        geometry.staged_k if geometry.use_lds else geometry.k
                    ),
                    width=geometry.width,
                )
            )
            prefetched_b.append(
                [
                    load_vector(
                        b_global,
                        safe_columns[column],
                        k_base,
                        geometry.k,
                        geometry.width,
                        config.b_cache_modifier,
                    )
                    for column in range_constexpr(columns)
                ]
            )
        for stage in range_constexpr(chunks):
            accumulators = _accumulate_vectors(
                accumulators,
                prefetched_a[stage],
                prefetched_b[stage],
                m=geometry.m,
                columns=columns,
                width=geometry.width,
            )

    for tail in range_constexpr(geometry.tail_tiles):
        k_base = fx.Int32(
            geometry.full_chunks * WAVE_SIZE * geometry.width
        ) + k_element(
            fx.Int32(tail),
            lane,
            fx.Int32(0),
            geometry.tail_tiles,
            MFMA_K,
        )
        a_vectors = _load_tail_a_vectors(
            use_lds=geometry.use_lds,
            a_global=a_global,
            a_smem=a_smem,
            k_base=k_base,
            m=geometry.m,
            k=geometry.k,
            staged_k=geometry.staged_k,
        )
        b_vectors = [
            masked_bf16_vector(
                b_global,
                safe_columns[column],
                k_base,
                MFMA_K,
                geometry.k,
                config.b_cache_modifier,
            )
            for column in range_constexpr(columns)
        ]
        accumulators = _accumulate_vectors(
            accumulators,
            a_vectors,
            b_vectors,
            m=geometry.m,
            columns=columns,
            width=MFMA_K,
        )

    reduced = [
        [
            reduce_mfma_scalar(accumulators[row][column])
            for column in range_constexpr(columns)
        ]
        for row in range_constexpr(geometry.m)
    ]
    return reduced, logical_columns


def _make_block_kernel(
    *,
    m: int,
    n: int,
    k: int,
    config: BlockMfmaDecodeConfig,
    kernel_name: str,
    persistent_turns: int,
    has_bias: bool,
):
    """Build one selected BlockMFMA specialization."""
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
    vector_prefix = activation_vectors_per_row * 8
    global_tail_per_row = k - vector_prefix
    shared_padding_per_row = staged_k - k
    global_tail_elements = m * global_tail_per_row
    shared_padding_elements = m * shared_padding_per_row
    stage_iterations = _ceil_div(activation_vectors, block_threads)
    compute_geometry = BlockMfmaGeometry(
        m=m,
        n=n,
        k=k,
        staged_k=staged_k,
        width=width,
        full_chunks=full_chunks,
        tail_tiles=tail_tiles,
        schedule_depth=schedule_depth,
        schedule_batches=schedule_batches,
        use_lds=use_lds,
    )

    @fx.struct
    class SharedStorage:
        activation: fx.Array[fx.BFloat16, m * staged_k, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def block_mfma_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        BIAS: fx.Tensor,
        runtime_grid_workgroups: Int32,
        runtime_persistent_turns: Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        wave, lane = wave_lane_coordinates(tid, waves)
        first_column = (
            gpu.block_idx.x * fx.Int32(waves * columns)
            + wave * fx.Int32(columns)
        )
        a_global = make_buffer_matrix(A, m, k)
        b_global = make_buffer_matrix(B, n, k)
        c_global = make_buffer_matrix(C, m, n)
        if const_expr(has_bias):
            bias_global = make_buffer_vector(BIAS, n)
        a_smem = None
        if use_lds:
            shared = fx.SharedAllocator().allocate(SharedStorage).peek()
            a_smem = shared.activation.view(
                fx.make_layout((m, staged_k), (staged_k, 1))
            )
            for iteration in range_constexpr(stage_iterations):
                slot = tid + fx.Int32(iteration * block_threads)
                if slot < fx.Int32(activation_vectors):
                    row, row_vector = padded_row_coordinates(
                        slot,
                        m,
                        activation_vectors_per_row,
                    )
                    column = row_vector * fx.Int32(8)
                    staged = load_vector(a_global, row, column, k, 8)
                    make_vector_view(
                        a_smem,
                        row,
                        column,
                        staged_k,
                        8,
                    ).store(staged)
            if global_tail_elements:
                a_global_tail = fx.make_view(
                    fx.add_offset(
                        fx.get_iter(a_global),
                        fx.make_int_tuple(vector_prefix),
                    ),
                    fx.make_layout(
                        (m, global_tail_per_row),
                        (k, 1),
                    ),
                )
                a_smem_tail = fx.make_view(
                    fx.add_offset(
                        fx.get_iter(a_smem),
                        fx.make_int_tuple(vector_prefix),
                    ),
                    fx.make_layout(
                        (m, global_tail_per_row),
                        (staged_k, 1),
                    ),
                )
                if tid < fx.Int32(global_tail_elements):
                    row, row_tail = padded_row_coordinates(
                        tid,
                        m,
                        global_tail_per_row,
                    )
                    a_smem_tail[row, row_tail] = a_global_tail[row, row_tail]
            if shared_padding_elements:
                a_smem_padding = fx.make_view(
                    fx.add_offset(
                        fx.get_iter(a_smem),
                        fx.make_int_tuple(k),
                    ),
                    fx.make_layout(
                        (m, shared_padding_per_row),
                        (staged_k, 1),
                    ),
                )
                if tid < fx.Int32(shared_padding_elements):
                    row, row_padding = padded_row_coordinates(
                        tid,
                        m,
                        shared_padding_per_row,
                    )
                    a_smem_padding[row, row_padding] = arith.constant(
                        0.0,
                        type=T.bf16,
                    )
            gpu.barrier()

        if persistent_turns == 1:
            reduced, logical_columns = _compute_column_tile(
                config=config,
                geometry=compute_geometry,
                a_global=a_global,
                a_smem=a_smem,
                b_global=b_global,
                column_base=first_column,
                lane=lane,
            )
            for row in range_constexpr(m):
                for column in range_constexpr(columns):
                    column_coord = logical_columns[column]
                    if (
                        lane == fx.Int32(WAVE_SIZE - 1)
                    ) & (column_coord < fx.Int32(n)):
                        element = (
                            fx.Int32(row) * fx.Int32(n)
                            + fx.Int32(column_coord)
                        )
                        value = reduced[row][column]
                        if const_expr(has_bias):
                            value = add_bias_f32(
                                value,
                                bias_global,
                                column_coord,
                            )
                        output = convert_bf16(
                            value,
                            element,
                            config.output_rounding,
                        )
                        c_global[row, column_coord] = output
        else:
            column_stride = (
                runtime_grid_workgroups * fx.Int32(waves * columns)
            )
            turn = fx.Int32(0)
            while turn < runtime_persistent_turns:
                column_base = first_column + turn * column_stride
                # Keep staged-A reads inside the runtime persistent-N loop.
                llvm_dialect.inline_asm(
                    None,
                    [],
                    "",
                    "~{memory}",
                    has_side_effects=True,
                )
                if column_base < fx.Int32(n):
                    reduced, logical_columns = _compute_column_tile(
                        config=config,
                        geometry=compute_geometry,
                        a_global=a_global,
                        a_smem=a_smem,
                        b_global=b_global,
                        column_base=column_base,
                        lane=lane,
                    )
                    for row in range_constexpr(m):
                        for column in range_constexpr(columns):
                            column_coord = logical_columns[column]
                            if (
                                lane == fx.Int32(WAVE_SIZE - 1)
                            ) & (column_coord < fx.Int32(n)):
                                element = (
                                    fx.Int32(row) * fx.Int32(n)
                                    + fx.Int32(column_coord)
                                )
                                value = reduced[row][column]
                                if const_expr(has_bias):
                                    value = add_bias_f32(
                                        value,
                                        bias_global,
                                        column_coord,
                                    )
                                output = convert_bf16(
                                    value,
                                    element,
                                    config.output_rounding,
                                )
                                c_global[row, column_coord] = output
                turn = turn + fx.Int32(1)

    return block_mfma_kernel


@functools.lru_cache(maxsize=2048)
def compile_gemm_decode_block_mfma_bf16(
    m: int,
    n: int,
    k: int,
    config: BlockMfmaDecodeConfig,
    arch: str,
    num_cus: int | None = None,
    *,
    has_bias: bool = False,
):
    """Compile one work-sized or grid-capped persistent BlockMFMA kernel."""
    config.validate(m=m, n=n, k=k, arch=arch)
    kernel_name = gemm_decode_kernel_name(
        arch,
        m,
        n,
        k,
        config,
        has_bias=has_bias,
    )
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
    cache_tag = make_decode_cache_tag(
        policy="block_mfma",
        kernel_name=kernel_name,
        arch=arch,
        num_cus=num_cus or 0,
        compile_scalars={
            "m": m,
            "n": n,
            "k": k,
            "waves_per_workgroup": config.waves_per_workgroup,
            "columns_per_wave": config.columns_per_wave,
            "activation_source": config.activation_source.value,
            "b_load_width": config.b_load_width,
            "k_unroll": config.k_unroll,
            "prefetch_stages": config.prefetch_stages,
            "persistent_n": config.persistent_n,
            "workgroups_per_cu": config.workgroups_per_cu,
            "waves_per_eu": config.waves_per_eu,
            "b_cache_modifier": config.b_cache_modifier,
            "output_rounding": config.output_rounding.value,
            "block_threads": block_threads,
            "logical_workgroups": logical_workgroups,
            "grid_workgroups": grid_workgroups,
            "persistent_turns": persistent_turns,
            "use_lds": use_lds,
            "has_bias": has_bias,
        },
    )

    kernel = _make_block_kernel(
        m=m,
        n=n,
        k=k,
        config=config,
        kernel_name=kernel_name,
        persistent_turns=persistent_turns,
        has_bias=has_bias,
    )
    kernel_attributes = (
        {"rocdl.waves_per_eu": config.waves_per_eu}
        if config.waves_per_eu
        else {}
    )

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        BIAS: fx.Tensor,
        cache_identity: fx.Constexpr[str],
        stream: fx.Stream = fx.Stream(None),
    ):
        kernel(
            A,
            B,
            C,
            BIAS,
            fx.Int32(grid_workgroups),
            fx.Int32(persistent_turns),
            value_attrs=kernel_attributes,
        ).launch(
            grid=(grid_workgroups, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    def launcher(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        bias: fx.Tensor | None = None,
        stream: fx.Stream = fx.Stream(None),
    ):
        if has_bias and bias is None:
            raise ValueError("This decode launcher requires `bias`.")
        if not has_bias and bias is not None:
            raise ValueError(
                "This decode launcher was compiled without bias support."
            )
        launch_bias = B if bias is None else bias
        return _run_compiled(
            launch,
            A,
            B,
            C,
            launch_bias,
            cache_tag,
            stream,
        )

    return launcher

# SPDX-License-Identifier: MIT

"""Exact-M one-wave/no-LDS BF16 and per-tensor FP8 decode GEMM policies."""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr

from .gemm_decode_common import (
    WORD_BYTES,
    ContractionMode,
    ReductionMode,
    WaveDecodeConfig,
    contract_pair,
    decode_fp8_words_bf16_packed,
    decode_fp8_words_f32,
    gemm_decode_kernel_name,
    k_element,
    load_fp8_words,
    load_vector,
    make_buffer_matrix,
    make_buffer_vector,
    masked_bf16_vector,
    masked_fp8_words,
    pack_bf16x2,
    packed_f32_pair,
    prepare_pair,
    reduce_wave_accumulator,
    store_bf16,
    validate_fp8_wave_config,
    wave_lane_coordinates,
    zero_wave_accumulator,
)
from .tensor_shim import _run_compiled, unused_tensor_arg


def _accumulate_vectors(
    accumulators,
    a_vectors,
    b_vectors,
    config,
):
    pairs = config.kvec // 2
    for pair in range_constexpr(pairs):
        a_pairs = [
            prepare_pair(
                pack_bf16x2(vector[2 * pair], vector[2 * pair + 1]),
                config.contraction,
            )
            for vector in a_vectors
        ]
        b_pairs = [
            prepare_pair(
                pack_bf16x2(vector[2 * pair], vector[2 * pair + 1]),
                config.contraction,
            )
            for vector in b_vectors
        ]
        for row in range_constexpr(config.m_per_wave):
            for column in range_constexpr(config.n_per_wave):
                accumulators[row][column] = contract_pair(
                    accumulators[row][column],
                    a_pairs[row],
                    b_pairs[column],
                    config.contraction,
                )
    return accumulators


def _accumulate_f32_vectors(
    accumulators,
    a_values,
    b_values,
    config,
):
    """Contract already-converted f32 K slices with ``v_pk_fma_f32``."""
    for pair in range_constexpr(config.kvec // 2):
        a_pairs = [
            packed_f32_pair(values[2 * pair], values[2 * pair + 1])
            for values in a_values
        ]
        b_pairs = [
            packed_f32_pair(values[2 * pair], values[2 * pair + 1])
            for values in b_values
        ]
        for row in range_constexpr(config.m_per_wave):
            for column in range_constexpr(config.n_per_wave):
                accumulators[row][column] = contract_pair(
                    accumulators[row][column],
                    a_pairs[row],
                    b_pairs[column],
                    config.contraction,
                )
    return accumulators


def _accumulate_packed_bf16_vectors(
    accumulators,
    a_values,
    b_values,
    config,
):
    """Contract already-packed BF16 pairs with ``v_dot2_f32_bf16``."""
    for pair in range_constexpr(config.kvec // 2):
        for row in range_constexpr(config.m_per_wave):
            for column in range_constexpr(config.n_per_wave):
                accumulators[row][column] = contract_pair(
                    accumulators[row][column],
                    a_values[row][pair],
                    b_values[column][pair],
                    config.contraction,
                )
    return accumulators


@functools.lru_cache(maxsize=2048)
def compile_gemm_decode_wave_fp8(
    m: int,
    n: int,
    k: int,
    config: WaveDecodeConfig,
    arch: str,
):
    """Compile one exact shape/config for per-tensor E4M3 operands.

    Both scales are loop invariant, so the product is folded once in the
    lane-0 epilogue rather than per K slice.
    """
    validate_fp8_wave_config(config, m=m, n=n, k=k, arch=arch)
    # gfx950 converts FP8 straight to BF16 and contracts two MACs per issue;
    # gfx942 has no such convert and stays on the portable f32 path.
    use_dot2 = config.contraction == ContractionMode.DOT2_BF16
    decode_fp8 = decode_fp8_words_bf16_packed if use_dot2 else decode_fp8_words_f32
    accumulate = (
        _accumulate_packed_bf16_vectors if use_dot2 else _accumulate_f32_vectors
    )
    kernel_name = gemm_decode_kernel_name(arch, m, n, k, config) + "_FP8"
    np = config.n_per_wave
    mp = config.m_per_wave
    kvec = config.kvec
    words = kvec // 4
    words_per_row = k // 4
    k_tile = 64 * kvec
    full_tiles = k // k_tile
    has_tail = k % k_tile != 0
    use_column_grid_y = mp == 1
    store_lane = 63 if config.reduction == ReductionMode.DPP else 0
    cache_tag = kernel_name

    @flyc.kernel(name=kernel_name, known_block_size=[64, 1, 1])
    def wave_decode_fp8_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        XSCALE: fx.Tensor,
        WSCALE: fx.Tensor,
    ):
        _, lane = wave_lane_coordinates(gpu.thread_idx.x, 1)
        column_block = gpu.block_idx.y if use_column_grid_y else gpu.block_idx.x
        row_block = gpu.block_idx.x if use_column_grid_y else gpu.block_idx.y
        column_owner_layout = fx.make_layout((n // np, np), (np, 1))
        row_owner_layout = fx.make_layout((m // mp, mp), (mp, 1))
        row_base = fx.Int32(
            fx.get_scalar(fx.crd2idx((row_block, fx.Int32(0)), row_owner_layout))
        )
        a_global = make_buffer_matrix(A, m, words_per_row, WORD_BYTES)
        b_global = make_buffer_matrix(B, n, words_per_row, WORD_BYTES)
        c_global = make_buffer_matrix(C, m, n)
        columns = [
            fx.Int32(
                fx.get_scalar(
                    fx.crd2idx((column_block, fx.Int32(column)), column_owner_layout)
                )
            )
            for column in range_constexpr(np)
        ]
        accumulators = [
            [zero_wave_accumulator(config.contraction) for _ in range_constexpr(np)]
            for _ in range_constexpr(mp)
        ]

        for tile in range_constexpr(full_tiles):
            word_base = k_element(fx.Int32(tile), lane, fx.Int32(0), full_tiles, words)
            a_values = [
                decode_fp8(
                    load_fp8_words(
                        a_global,
                        row_base + fx.Int32(row),
                        word_base,
                        words_per_row,
                        words,
                    )
                )
                for row in range_constexpr(mp)
            ]
            b_values = [
                decode_fp8(
                    load_fp8_words(
                        b_global,
                        columns[column],
                        word_base,
                        words_per_row,
                        words,
                        config.b_cache_modifier,
                    )
                )
                for column in range_constexpr(np)
            ]
            accumulators = accumulate(
                accumulators,
                a_values,
                b_values,
                config,
            )

        if has_tail:
            word_base = fx.Int32(full_tiles * k_tile // 4) + k_element(
                fx.Int32(0), lane, fx.Int32(0), 1, words
            )
            a_values = [
                decode_fp8(
                    masked_fp8_words(
                        a_global,
                        row_base + fx.Int32(row),
                        word_base,
                        words_per_row,
                        words,
                    )
                )
                for row in range_constexpr(mp)
            ]
            b_values = [
                decode_fp8(
                    masked_fp8_words(
                        b_global,
                        columns[column],
                        word_base,
                        words_per_row,
                        words,
                        config.b_cache_modifier,
                    )
                )
                for column in range_constexpr(np)
            ]
            accumulators = accumulate(
                accumulators,
                a_values,
                b_values,
                config,
            )

        scale = fx.Float32(make_buffer_vector(XSCALE, 1, 4)[0]) * fx.Float32(
            make_buffer_vector(WSCALE, 1, 4)[0]
        )
        reduced = [
            [
                reduce_wave_accumulator(
                    accumulators[row][column],
                    lane,
                    config.contraction,
                    config.reduction,
                )
                for column in range_constexpr(np)
            ]
            for row in range_constexpr(mp)
        ]
        if lane == fx.Int32(store_lane):
            for row in range_constexpr(mp):
                for column in range_constexpr(np):
                    store_bf16(
                        fx.Float32(reduced[row][column]) * scale,
                        c_global,
                        row_base + fx.Int32(row),
                        columns[column],
                        n,
                        config.output_rounding,
                    )

    default_stream = fx.Stream(None)

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        XSCALE: fx.Tensor,
        WSCALE: fx.Tensor,
        cache_identity: fx.Constexpr[str],
        stream: fx.Stream = default_stream,
    ):
        wave_decode_fp8_kernel(
            A,
            B,
            C,
            XSCALE,
            WSCALE,
            value_attrs={"rocdl.waves_per_eu": config.waves_per_eu},
        ).launch(
            grid=(
                (m // mp, n // np, 1) if use_column_grid_y else (n // np, m // mp, 1)
            ),
            block=(64, 1, 1),
            stream=stream,
        )

    def launcher(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        x_scale: fx.Tensor,
        w_scale: fx.Tensor,
        stream: fx.Stream = default_stream,
    ):
        # MLIR has no LLVM-compatible vector of f8E4M3 and a v8i8 buffer load is
        # not selectable, so the operands enter as packed 32-bit words.
        return _run_compiled(
            launch,
            A.view(torch.int32),
            B.view(torch.int32),
            C,
            x_scale,
            w_scale,
            cache_tag,
            stream,
        )

    return launcher


@functools.lru_cache(maxsize=2048)
def compile_gemm_decode_wave_bf16(
    m: int,
    n: int,
    k: int,
    config: WaveDecodeConfig,
    arch: str,
    *,
    has_bias: bool = False,
):
    """Compile one exact shape/config with the architecture in the cache key."""
    config.validate(m=m, n=n, k=k, arch=arch)
    kernel_name = gemm_decode_kernel_name(
        arch,
        m,
        n,
        k,
        config,
        has_bias=has_bias,
    )
    np = config.n_per_wave
    mp = config.m_per_wave
    kvec = config.kvec
    k_tile = 64 * kvec
    full_tiles = k // k_tile
    has_tail = k % k_tile != 0
    # Preserve the proven one-row-wave launch geometry: row groups in X and
    # output columns in Y. Multi-row waves keep columns in X.
    use_column_grid_y = mp == 1
    store_lane = 63 if config.reduction == ReductionMode.DPP else 0
    cache_tag = kernel_name

    @flyc.kernel(name=kernel_name, known_block_size=[64, 1, 1])
    def wave_decode_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        BIAS: fx.Tensor,
    ):
        _, lane = wave_lane_coordinates(gpu.thread_idx.x, 1)
        column_block = gpu.block_idx.y if use_column_grid_y else gpu.block_idx.x
        row_block = gpu.block_idx.x if use_column_grid_y else gpu.block_idx.y
        column_owner_layout = fx.make_layout(
            (n // np, np),
            (np, 1),
        )
        row_owner_layout = fx.make_layout(
            (m // mp, mp),
            (mp, 1),
        )
        row_base = fx.Int32(
            fx.get_scalar(fx.crd2idx((row_block, fx.Int32(0)), row_owner_layout))
        )
        a_global = make_buffer_matrix(A, m, k)
        b_global = make_buffer_matrix(B, n, k)
        c_global = make_buffer_matrix(C, m, n)
        if const_expr(has_bias):
            bias_global = make_buffer_vector(BIAS, n)
        columns = [
            fx.Int32(
                fx.get_scalar(
                    fx.crd2idx(
                        (column_block, fx.Int32(column)),
                        column_owner_layout,
                    )
                )
            )
            for column in range_constexpr(np)
        ]
        accumulators = [
            [zero_wave_accumulator(config.contraction) for _ in range_constexpr(np)]
            for _ in range_constexpr(mp)
        ]

        if config.prefetch_depth == 0:
            for tile in range_constexpr(full_tiles):
                k_base = k_element(
                    fx.Int32(tile),
                    lane,
                    fx.Int32(0),
                    full_tiles,
                    kvec,
                )
                a_vectors = [
                    load_vector(
                        a_global,
                        row_base + fx.Int32(row),
                        k_base,
                        k,
                        kvec,
                    )
                    for row in range_constexpr(mp)
                ]
                b_vectors = [
                    load_vector(
                        b_global,
                        columns[column],
                        k_base,
                        k,
                        kvec,
                        config.b_cache_modifier,
                    )
                    for column in range_constexpr(np)
                ]
                accumulators = _accumulate_vectors(
                    accumulators,
                    a_vectors,
                    b_vectors,
                    config,
                )
        else:
            for batch in range_constexpr(
                (full_tiles + config.prefetch_depth - 1) // config.prefetch_depth
            ):
                batch_tiles = min(
                    config.prefetch_depth,
                    full_tiles - batch * config.prefetch_depth,
                )
                prefetched_a = []
                prefetched_b = []
                for stage in range_constexpr(batch_tiles):
                    tile = batch * config.prefetch_depth + stage
                    k_base = k_element(
                        fx.Int32(tile),
                        lane,
                        fx.Int32(0),
                        full_tiles,
                        kvec,
                    )
                    prefetched_a.append(
                        [
                            load_vector(
                                a_global,
                                row_base + fx.Int32(row),
                                k_base,
                                k,
                                kvec,
                            )
                            for row in range_constexpr(mp)
                        ]
                    )
                    prefetched_b.append(
                        [
                            load_vector(
                                b_global,
                                columns[column],
                                k_base,
                                k,
                                kvec,
                                config.b_cache_modifier,
                            )
                            for column in range_constexpr(np)
                        ]
                    )
                for stage in range_constexpr(batch_tiles):
                    accumulators = _accumulate_vectors(
                        accumulators,
                        prefetched_a[stage],
                        prefetched_b[stage],
                        config,
                    )

        if has_tail:
            tail_start = full_tiles * k_tile
            k_base = fx.Int32(tail_start) + k_element(
                fx.Int32(0),
                lane,
                fx.Int32(0),
                1,
                kvec,
            )
            a_vectors = [
                masked_bf16_vector(
                    a_global,
                    row_base + fx.Int32(row),
                    k_base,
                    kvec,
                    k,
                )
                for row in range_constexpr(mp)
            ]
            b_vectors = [
                masked_bf16_vector(
                    b_global,
                    columns[column],
                    k_base,
                    kvec,
                    k,
                    config.b_cache_modifier,
                )
                for column in range_constexpr(np)
            ]
            accumulators = _accumulate_vectors(
                accumulators,
                a_vectors,
                b_vectors,
                config,
            )

        reduced = [
            [
                reduce_wave_accumulator(
                    accumulators[row][column],
                    lane,
                    config.contraction,
                    config.reduction,
                )
                for column in range_constexpr(np)
            ]
            for row in range_constexpr(mp)
        ]
        if lane == fx.Int32(store_lane):
            for row in range_constexpr(mp):
                for column in range_constexpr(np):
                    value = reduced[row][column]
                    if const_expr(has_bias):
                        value = fx.Float32(value) + bias_global[columns[column]].to(
                            fx.Float32
                        )
                    store_bf16(
                        value,
                        c_global,
                        row_base + fx.Int32(row),
                        columns[column],
                        n,
                        config.output_rounding,
                    )

    default_stream = fx.Stream(None)

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        BIAS: fx.Tensor,
        cache_identity: fx.Constexpr[str],
        stream: fx.Stream = default_stream,
    ):
        wave_decode_kernel(
            A,
            B,
            C,
            BIAS,
            value_attrs={"rocdl.waves_per_eu": config.waves_per_eu},
        ).launch(
            grid=(
                (m // mp, n // np, 1) if use_column_grid_y else (n // np, m // mp, 1)
            ),
            block=(64, 1, 1),
            stream=stream,
        )

    def launcher(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        bias: fx.Tensor | None = None,
        stream: fx.Stream = default_stream,
    ):
        if has_bias and bias is None:
            raise ValueError("This decode launcher requires `bias`.")
        if not has_bias and bias is not None:
            raise ValueError("This decode launcher was compiled without bias support.")
        return _run_compiled(
            launch,
            A,
            B,
            C,
            unused_tensor_arg(bias, B),
            cache_tag,
            stream,
        )

    return launcher

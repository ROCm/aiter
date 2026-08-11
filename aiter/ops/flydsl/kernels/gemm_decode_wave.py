# SPDX-License-Identifier: MIT

"""Exact-M one-wave/no-LDS BF16 decode GEMM policy."""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import buffer_ops, gpu, range_constexpr
from flydsl.expr.typing import T

from .gemm_decode_config import (
    ReductionMode,
    WaveDecodeConfig,
    gemm_decode_kernel_name,
)
from .gemm_decode_numeric import (
    contract_pair,
    masked_bf16_vector,
    pack_bf16x2,
    prepare_pair,
    reduce_wave_accumulator,
    store_bf16,
    zero_wave_accumulator,
)


def _load_full_vector(resource, row, k_element, k: int, width: int, cache=0):
    return buffer_ops.buffer_load(
        resource,
        row * fx.Int32(k) + k_element,
        vec_width=width,
        dtype=T.bf16,
        cache_modifier=cache,
    )


def _accumulate_vectors(accumulators, a_vectors, b_vectors, config):
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
                index = row * config.n_per_wave + column
                accumulators[index] = contract_pair(
                    accumulators[index],
                    a_pairs[row],
                    b_pairs[column],
                    config.contraction,
                )
    return accumulators


@functools.lru_cache(maxsize=2048)
def compile_gemm_decode_wave_bf16(
    m: int,
    n: int,
    k: int,
    config: WaveDecodeConfig,
    arch: str,
):
    """Compile one exact shape/config with the architecture in the cache key."""
    config.validate(m=m, n=n, k=k, arch=arch)
    kernel_name = gemm_decode_kernel_name(arch, m, n, k, config)
    np = config.n_per_wave
    mp = config.m_per_wave
    kvec = config.kvec
    k_tile = 64 * kvec
    full_tiles = k // k_tile
    has_tail = k % k_tile != 0
    # Preserve the proven one-row-wave launch geometry: row groups in X and
    # output columns in Y. Multi-row waves keep columns in X.
    use_column_grid_y = mp == 1

    @flyc.kernel(name=kernel_name, known_block_size=[64, 1, 1])
    def wave_decode_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        K: fx.Constexpr[int],
        N: fx.Constexpr[int],
    ):
        lane = gpu.thread_idx.x
        column_block = (
            gpu.block_idx.y if use_column_grid_y else gpu.block_idx.x
        )
        row_block = (
            gpu.block_idx.x if use_column_grid_y else gpu.block_idx.y
        )
        column_base = column_block * fx.Int32(np)
        row_base = row_block * fx.Int32(mp)
        a_resource = buffer_ops.create_buffer_resource(A)
        b_resource = buffer_ops.create_buffer_resource(B)
        c_resource = buffer_ops.create_buffer_resource(C)
        columns = [
            column_base + fx.Int32(column)
            for column in range_constexpr(np)
        ]
        accumulators = [
            zero_wave_accumulator(config.contraction)
            for _ in range_constexpr(mp * np)
        ]

        if config.prefetch_depth == 0:
            for tile in range_constexpr(full_tiles):
                k_element = fx.Int32(tile * k_tile) + lane * fx.Int32(kvec)
                a_vectors = [
                    _load_full_vector(
                        a_resource, row_base + fx.Int32(row), k_element, k, kvec
                    )
                    for row in range_constexpr(mp)
                ]
                b_vectors = [
                    _load_full_vector(
                        b_resource,
                        columns[column],
                        k_element,
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
                (full_tiles + config.prefetch_depth - 1)
                // config.prefetch_depth
            ):
                batch_tiles = min(
                    config.prefetch_depth,
                    full_tiles - batch * config.prefetch_depth,
                )
                prefetched_a = []
                prefetched_b = []
                for stage in range_constexpr(batch_tiles):
                    tile = batch * config.prefetch_depth + stage
                    k_element = fx.Int32(tile * k_tile) + lane * fx.Int32(kvec)
                    prefetched_a.append(
                        [
                            _load_full_vector(
                                a_resource,
                                row_base + fx.Int32(row),
                                k_element,
                                k,
                                kvec,
                            )
                            for row in range_constexpr(mp)
                        ]
                    )
                    prefetched_b.append(
                        [
                            _load_full_vector(
                                b_resource,
                                columns[column],
                                k_element,
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
            k_element = fx.Int32(tail_start) + lane * fx.Int32(kvec)
            a_vectors = [
                masked_bf16_vector(
                    a_resource,
                (row_base + fx.Int32(row)) * fx.Int32(k),
                    k_element,
                    kvec,
                    k,
                )
                for row in range_constexpr(mp)
            ]
            b_vectors = [
                masked_bf16_vector(
                    b_resource,
                    columns[column] * fx.Int32(k),
                    k_element,
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
            reduce_wave_accumulator(
                accumulator,
                lane,
                config.contraction,
                config.reduction,
            )
            for accumulator in accumulators
        ]
        store_lane = (
            63 if config.reduction == ReductionMode.DPP else 0
        )
        if lane == fx.Int32(store_lane):
            for row in range_constexpr(mp):
                for column in range_constexpr(np):
                    output_element = (
                        (row_base + fx.Int32(row)) * fx.Int32(n)
                        + columns[column]
                    )
                    store_bf16(
                        reduced[row * np + column],
                        c_resource,
                        output_element,
                        config.output_rounding,
                    )

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        wave_decode_kernel(
            A,
            B,
            C,
            k,
            n,
            value_attrs={"rocdl.waves_per_eu": config.waves_per_eu},
        ).launch(
            grid=(
                (m // mp, n // np, 1)
                if use_column_grid_y
                else (n // np, m // mp, 1)
            ),
            block=(64, 1, 1),
            stream=stream,
        )

    launch.kernel_name = kernel_name
    launch.lds_bytes = 0
    launch.policy = "wave"
    return launch

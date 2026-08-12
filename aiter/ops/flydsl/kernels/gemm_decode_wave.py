# SPDX-License-Identifier: MIT

"""Exact-M one-wave/no-LDS BF16 decode GEMM policy."""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from .gemm_decode_config import (
    ReductionMode,
    WaveDecodeConfig,
    gemm_decode_kernel_name,
    make_decode_cache_tag,
)
from .gemm_decode_layouts import (
    k_element,
    load_vector,
    make_buffer_matrix,
    wave_lane_coordinates,
)
from .tensor_shim import _run_compiled
from .gemm_decode_numeric import (
    contract_pair,
    masked_bf16_vector,
    pack_bf16x2,
    prepare_pair,
    reduce_wave_accumulator,
    store_bf16,
    zero_wave_accumulator,
)


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
    store_lane = 63 if config.reduction == ReductionMode.DPP else 0
    cache_tag = make_decode_cache_tag(
        policy="wave",
        kernel_name=kernel_name,
        arch=arch,
        num_cus=0,
        compile_scalars={
            "m": m,
            "n": n,
            "k": k,
            "m_per_wave": config.m_per_wave,
            "n_per_wave": config.n_per_wave,
            "kvec": config.kvec,
            "prefetch_depth": config.prefetch_depth,
            "waves_per_eu": config.waves_per_eu,
            "b_cache_modifier": config.b_cache_modifier,
            "reduction": config.reduction.value,
            "contraction": config.contraction.value,
            "output_rounding": config.output_rounding.value,
            "use_column_grid_y": use_column_grid_y,
        },
    )

    @flyc.kernel(name=kernel_name, known_block_size=[64, 1, 1])
    def wave_decode_kernel(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
    ):
        _, lane = wave_lane_coordinates(gpu.thread_idx.x, 1)
        column_block = (
            gpu.block_idx.y if use_column_grid_y else gpu.block_idx.x
        )
        row_block = (
            gpu.block_idx.x if use_column_grid_y else gpu.block_idx.y
        )
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
            [
                zero_wave_accumulator(config.contraction)
                for _ in range_constexpr(np)
            ]
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
                    store_bf16(
                        reduced[row][column],
                        c_global,
                        row_base + fx.Int32(row),
                        columns[column],
                        n,
                        config.output_rounding,
                    )

    @flyc.jit
    def launch(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        cache_identity: fx.Constexpr[str],
        stream: fx.Stream = fx.Stream(None),
    ):
        wave_decode_kernel(
            A,
            B,
            C,
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

    def launcher(
        A: fx.Tensor,
        B: fx.Tensor,
        C: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        return _run_compiled(launch, A, B, C, cache_tag, stream)

    return launcher

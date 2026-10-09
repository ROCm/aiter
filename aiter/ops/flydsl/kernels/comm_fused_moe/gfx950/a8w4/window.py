# SPDX-License-Identifier: Apache-2.0
"""Windowed GEMM2 and TP communication family."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from ....mxfp4_gemm_common import global_typed_ptr
from .collectives import (
    buffer_tensor_from_addr,
    decode_scaled_fp8_f32,
    e8m0_scale,
    emit_tp_all_gather,
    emit_tp_reduce_scatter,
    load_bf16,
    load_e8m0_scale,
    load_fp8_words,
    pack_fp8_words,
    store_buffer,
    store_fp8_words,
)
from .config import BLOCK, WindowConfig
from .producer import compile_window_producer


def _compose_compute(config: WindowConfig, window_index: int):
    tiles_per_window = config.tiles_per_window

    def compose(*, module_name, emit_gemm2_tile, shared_storage):
        @flyc.kernel(
            name=f"{module_name}_window_{window_index}",
            known_block_size=[BLOCK, 1, 1],
        )
        def kernel(
            route_out: fx.Pointer,
            x: fx.Pointer,
            w: fx.Pointer,
            scale_x: fx.Pointer,
            scale_w: fx.Pointer,
            sorted_token_ids: fx.Pointer,
            expert_ids: fx.Pointer,
            sorted_weights: fx.Pointer,
            num_valid_ids: fx.Pointer,
            bias: fx.Pointer,
            tokens: fx.Int32,
            model_dim: fx.Int32,
            inter_dim: fx.Int32,
            size_expert_ids: fx.Int32,
        ):
            worker = fx.Int32(gpu.block_id("x"))
            m_block = worker // fx.Int32(tiles_per_window)
            n_block = fx.Int32(window_index * tiles_per_window) + worker % fx.Int32(
                tiles_per_window
            )
            tid = fx.Int32(gpu.thread_id("x"))
            lane = tid % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tid // fx.Int32(64))
            lds = fx.SharedAllocator().allocate(shared_storage).peek()
            valid_rows = global_typed_ptr(fx.Int64(ptrtoint(num_valid_ids)), T.i32)[0]
            if m_block * fx.Int32(config.tile_m) < valid_rows:
                emit_gemm2_tile(
                    fx.Int64(ptrtoint(x)),
                    fx.Int64(ptrtoint(scale_x)),
                    fx.Int64(ptrtoint(w)),
                    fx.Int64(ptrtoint(scale_w)),
                    fx.Int64(ptrtoint(expert_ids)),
                    fx.Int64(ptrtoint(sorted_token_ids)),
                    fx.Int64(ptrtoint(sorted_weights)),
                    fx.Int64(ptrtoint(bias)),
                    fx.Int64(ptrtoint(route_out)),
                    m_block,
                    n_block,
                    lane,
                    wave,
                    tokens,
                    size_expert_ids,
                    inter_dim,
                    model_dim,
                    lds,
                )

        @flyc.jit
        def launch(
            route_out,
            x,
            w,
            scale_x,
            scale_w,
            sorted_token_ids,
            expert_ids,
            sorted_weights,
            num_valid_ids,
            bias,
            tokens,
            model_dim,
            inter_dim,
            size_expert_ids,
            stream,
        ):
            grid = fx.Int32(size_expert_ids) * fx.Int32(tiles_per_window)
            kernel(
                route_out,
                x,
                w,
                scale_x,
                scale_w,
                sorted_token_ids,
                expert_ids,
                sorted_weights,
                num_valid_ids,
                bias,
                tokens,
                model_dim,
                inter_dim,
                size_expert_ids,
            ).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

        return launch

    return compose


def _compile_compute(config: WindowConfig, window: int, compose=None):
    return compile_window_producer(
        config,
        window,
        compose or _compose_compute(config, window),
    )


@functools.cache
def compile_compute(config: WindowConfig, window: int):
    """Compile one compact Stage2 window."""
    return _compile_compute(config, window)


@flyc.jit
def _emit_local(config: WindowConfig, route, partial, shared, worker):
    shape = config.shape
    m = config.m
    window = config.window
    groups_per_row = config.groups_per_row
    for token in range(worker, fx.Int32(m), fx.Int32(config.local_workers)):
        tid = fx.Int32(gpu.thread_id("x"))
        columns_per_pass = BLOCK * 8
        column_passes = (window + columns_per_pass - 1) // columns_per_pass
        for column_pass in range_constexpr(column_passes):
            column = tid * fx.Int32(8) + fx.Int32(column_pass * columns_per_pass)
            if column < fx.Int32(window):
                route_row_bytes = window + window // 8
                route_row = buffer_tensor_from_addr(
                    fx.Int64(ptrtoint(route))
                    + fx.Int64(token) * fx.Int64(shape.topk * route_row_bytes),
                    fx.Int32,
                    shape.topk * route_row_bytes,
                )
                route_scale_row = buffer_tensor_from_addr(
                    fx.Int64(ptrtoint(route))
                    + fx.Int64(token) * fx.Int64(shape.topk * route_row_bytes),
                    fx.Int8,
                    shape.topk * route_row_bytes,
                )
                acc = fx.Vector.filled(8, 0.0, fx.Float32)
                for slot in range_constexpr(shape.topk):
                    words = load_fp8_words(
                        route_row,
                        fx.Int32(slot * (route_row_bytes // 4)) + column // fx.Int32(4),
                        word_count=2,
                        load_width=2,
                        cache_modifier=2,
                    )
                    scale = load_e8m0_scale(
                        route_scale_row,
                        fx.Int32(slot * route_row_bytes + window)
                        + column // fx.Int32(8),
                        2,
                    )
                    values = decode_scaled_fp8_f32(words, scale)
                    acc = acc + fx.Vector.from_elements(values, fx.Float32)

                shared_row = buffer_tensor_from_addr(
                    fx.Int64(ptrtoint(shared))
                    + fx.Int64(token) * fx.Int64(shape.model_dim * 2),
                    fx.BFloat16,
                    shape.model_dim * 2,
                )
                shared_values = load_bf16(shared_row, column, 8, 2).to(fx.Float32)
                acc = acc + shared_values

                lane = tid & fx.Int32(63)
                local_max = fx.Float32(1e-10).maximumf(
                    fmath.absf(acc).reduce(ReductionOp.MAX)
                )
                max_bits = local_max.bitcast(fx.Int32)
                for xor_lane in (1, 2):
                    remote_bits = fx.rocdl.ds_bpermute(
                        T.i32,
                        (lane ^ fx.Int32(xor_lane)) * fx.Int32(4),
                        max_bits,
                    )
                    local_max = local_max.maximumf(
                        fx.Int32(remote_bits).bitcast(fx.Float32)
                    )
                    max_bits = local_max.bitcast(fx.Int32)
                e8m0, quant_scale = e8m0_scale(local_max)
                packed = pack_fp8_words(acc, quant_scale, 2)
                payload_row = buffer_tensor_from_addr(
                    fx.Int64(ptrtoint(partial)) + fx.Int64(token) * fx.Int64(window),
                    fx.Int32,
                    window,
                )
                store_fp8_words(payload_row, column, packed, 2)
                if lane & fx.Int32(3) == fx.Int32(0):
                    scale_row = buffer_tensor_from_addr(
                        fx.Int64(ptrtoint(partial))
                        + fx.Int64(m * window)
                        + fx.Int64(token) * fx.Int64(groups_per_row),
                        fx.Int8,
                        groups_per_row,
                    )
                    store_buffer(
                        scale_row,
                        column // fx.Int32(32),
                        e8m0.to(fx.Int8),
                        fx.Int8,
                    )


def _compose_cycle(
    config: WindowConfig,
    window_index: int,
):
    shape = config.shape
    m = config.m
    shard_rows = config.shard_rows
    window = config.window
    local_workers = config.local_workers
    reduce_scatter_grid = config.reduce_scatter_grid
    all_gather_grid = config.all_gather_grid
    has_reduce_scatter = window_index >= 2
    has_all_gather = window_index >= 3
    service_grid = max(
        reduce_scatter_grid if has_reduce_scatter else 0,
        all_gather_grid if has_all_gather else 0,
    )
    tiles_per_window = config.tiles_per_window

    def compose(
        *,
        module_name,
        emit_gemm2_tile,
        shared_storage,
    ):
        @flyc.kernel(
            name=(
                f"gemm2_tp_window_pipeline_{shape.tag}"
                f"_cycle_p{window_index}_sr{shard_rows}"
                f"_t{config.tile_m}x{config.tile_n}x{config.tile_k}"
                f"_sbm{config.sort_block_m}_w{window}_lw{local_workers}"
                f"_rsg{reduce_scatter_grid}_agg{all_gather_grid}"
                f"_rs{int(has_reduce_scatter)}ag{int(has_all_gather)}"
            ),
            known_block_size=[BLOCK, 1, 1],
        )
        def kernel(
            route_out: fx.Pointer,
            x: fx.Pointer,
            w: fx.Pointer,
            scale_x: fx.Pointer,
            scale_w: fx.Pointer,
            sorted_token_ids: fx.Pointer,
            expert_ids: fx.Pointer,
            sorted_weights: fx.Pointer,
            num_valid_ids: fx.Pointer,
            bias: fx.Pointer,
            tokens: fx.Int32,
            model_dim: fx.Int32,
            inter_dim: fx.Int32,
            size_expert_ids: fx.Int32,
            local_route: fx.Pointer,
            local_partial: fx.Pointer,
            local_shared: fx.Pointer,
            partial_flat_base: fx.Int64,
            reduced_shard: fx.Pointer,
            reduced_payload: fx.Pointer,
            reduced_scale: fx.Pointer,
            gather_payload_base: fx.Int64,
            gather_scale_base: fx.Int64,
            gathered_output: fx.Pointer,
            rank: fx.Int32,
        ):
            linear = fx.Int32(gpu.block_id("x"))
            tid = fx.Int32(gpu.thread_id("x"))
            lane = tid % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tid // fx.Int32(64))
            lds = fx.SharedAllocator().allocate(shared_storage).peek()
            valid_rows = global_typed_ptr(fx.Int64(ptrtoint(num_valid_ids)), T.i32)[0]

            def emit_compute(worker):
                m_block = worker // fx.Int32(tiles_per_window)
                n_block = fx.Int32(window_index * tiles_per_window) + worker % fx.Int32(
                    tiles_per_window
                )
                if m_block * fx.Int32(config.tile_m) < valid_rows:
                    emit_gemm2_tile(
                        fx.Int64(ptrtoint(x)),
                        fx.Int64(ptrtoint(scale_x)),
                        fx.Int64(ptrtoint(w)),
                        fx.Int64(ptrtoint(scale_w)),
                        fx.Int64(ptrtoint(expert_ids)),
                        fx.Int64(ptrtoint(sorted_token_ids)),
                        fx.Int64(ptrtoint(sorted_weights)),
                        fx.Int64(ptrtoint(bias)),
                        fx.Int64(ptrtoint(route_out)),
                        m_block,
                        n_block,
                        lane,
                        wave,
                        tokens,
                        size_expert_ids,
                        inter_dim,
                        model_dim,
                        lds,
                    )

            if const_expr(service_grid > 0):
                paired = linear < fx.Int32(service_grid * 3)
                slot = linear % fx.Int32(3)
                is_service = paired & (slot == fx.Int32(0))
                is_compute = (linear >= fx.Int32(service_grid * 3)) | (
                    paired & (slot != fx.Int32(0))
                )
                raw_compute = paired.select(
                    (linear // fx.Int32(3)) * fx.Int32(2) + slot - fx.Int32(1),
                    linear - fx.Int32(service_grid),
                )
                compute_worker = is_compute.select(raw_compute, fx.Int32(0))
                service_worker = linear // fx.Int32(3)
            else:
                compute_worker = linear

            if const_expr(service_grid > 0):
                if is_compute:
                    emit_compute(compute_worker)
            else:
                emit_compute(compute_worker)

            local_active = compute_worker < fx.Int32(local_workers)
            if const_expr(service_grid > 0):
                local_active = is_compute & local_active
            if local_active:
                _emit_local(
                    config,
                    local_route,
                    local_partial,
                    local_shared,
                    compute_worker,
                )

            if const_expr(service_grid > 0):  # noqa: SIM102
                if is_service:
                    if const_expr(has_reduce_scatter):  # noqa: SIM102
                        if service_worker < fx.Int32(reduce_scatter_grid):
                            emit_tp_reduce_scatter(
                                partial_flat_base,
                                reduced_shard,
                                reduced_payload,
                                reduced_scale,
                                rank,
                                service_worker,
                                tokens=m,
                                output_width=shape.model_dim,
                                payload_width=window,
                                shard_rows=shard_rows,
                                tp=shape.tp_size,
                                block=BLOCK,
                                reduce_scatter_grid=reduce_scatter_grid,
                            )
                    if const_expr(has_all_gather):  # noqa: SIM102
                        if service_worker < fx.Int32(all_gather_grid):
                            emit_tp_all_gather(
                                gather_payload_base,
                                gather_scale_base,
                                gathered_output,
                                rank,
                                service_worker,
                                output_width=shape.model_dim,
                                payload_width=window,
                                shard_rows=shard_rows,
                                tp=shape.tp_size,
                                block=BLOCK,
                                all_gather_grid=all_gather_grid,
                            )

        @flyc.jit
        def launch(
            route_out,
            x,
            w,
            scale_x,
            scale_w,
            sorted_token_ids,
            expert_ids,
            sorted_weights,
            num_valid_ids,
            bias,
            tokens,
            model_dim,
            inter_dim,
            size_expert_ids,
            local_route,
            local_partial,
            local_shared,
            partial_flat_base,
            reduced_shard,
            reduced_payload,
            reduced_scale,
            gather_payload_base,
            gather_scale_base,
            gathered_output,
            rank,
            stream,
        ):
            compute_workers = fx.Int32(size_expert_ids) * fx.Int32(tiles_per_window)
            grid = compute_workers + fx.Int32(service_grid)
            kernel(
                route_out,
                x,
                w,
                scale_x,
                scale_w,
                sorted_token_ids,
                expert_ids,
                sorted_weights,
                num_valid_ids,
                bias,
                tokens,
                model_dim,
                inter_dim,
                size_expert_ids,
                local_route,
                local_partial,
                local_shared,
                partial_flat_base,
                reduced_shard,
                reduced_payload,
                reduced_scale,
                gather_payload_base,
                gather_scale_base,
                gathered_output,
                rank,
            ).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

        return launch

    return compose


@functools.cache
def compile_cycle(
    config: WindowConfig,
    window: int,
):
    """Compile G/L with optional TP reduce-scatter/all-gather service CTAs."""
    return _compile_compute(
        config,
        window,
        _compose_cycle(config, window),
    )


@functools.cache
def compile_drain(
    config: WindowConfig,
    has_local: bool,
    has_reduce_scatter: bool,
    has_all_gather: bool,
):
    """Compile the fixed pipeline tail without reserving GEMM LDS."""
    shape = config.shape
    m = config.m
    shard_rows = config.shard_rows
    window = config.window
    local_workers = config.local_workers
    reduce_scatter_grid = config.reduce_scatter_grid
    all_gather_grid = config.all_gather_grid
    service_grid = max(
        reduce_scatter_grid if has_reduce_scatter else 0,
        all_gather_grid if has_all_gather else 0,
    )

    @flyc.kernel(
        name=(
            f"gemm2_tp_window_pipeline_{shape.tag}_drain_sr{shard_rows}"
            f"_w{window}_lw{local_workers}"
            f"_rsg{reduce_scatter_grid}_agg{all_gather_grid}"
            f"_l{int(has_local)}"
            f"rs{int(has_reduce_scatter)}ag{int(has_all_gather)}"
        ),
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        route: fx.Pointer,
        partial: fx.Pointer,
        shared: fx.Pointer,
        partial_flat_base: fx.Int64,
        reduced_shard: fx.Pointer,
        reduced_payload: fx.Pointer,
        reduced_scale: fx.Pointer,
        gather_payload_base: fx.Int64,
        gather_scale_base: fx.Int64,
        gathered_output: fx.Pointer,
        rank: fx.Int32,
    ):
        worker = fx.Int32(gpu.block_id("x"))
        if const_expr(has_reduce_scatter):  # noqa: SIM102
            if worker < fx.Int32(reduce_scatter_grid):
                emit_tp_reduce_scatter(
                    partial_flat_base,
                    reduced_shard,
                    reduced_payload,
                    reduced_scale,
                    rank,
                    worker,
                    tokens=m,
                    output_width=shape.model_dim,
                    payload_width=window,
                    shard_rows=shard_rows,
                    tp=shape.tp_size,
                    block=BLOCK,
                    reduce_scatter_grid=reduce_scatter_grid,
                )
        if const_expr(has_all_gather):  # noqa: SIM102
            if worker < fx.Int32(all_gather_grid):
                emit_tp_all_gather(
                    gather_payload_base,
                    gather_scale_base,
                    gathered_output,
                    rank,
                    worker,
                    output_width=shape.model_dim,
                    payload_width=window,
                    shard_rows=shard_rows,
                    tp=shape.tp_size,
                    block=BLOCK,
                    all_gather_grid=all_gather_grid,
                )

        if const_expr(has_local):
            local_worker = worker - fx.Int32(service_grid)
            if worker >= fx.Int32(service_grid):
                _emit_local(config, route, partial, shared, local_worker)

    @flyc.jit
    def launch(
        route,
        partial,
        shared,
        partial_flat_base,
        reduced_shard,
        reduced_payload,
        reduced_scale,
        gather_payload_base,
        gather_scale_base,
        gathered_output,
        rank,
        stream,
    ):
        kernel(
            route,
            partial,
            shared,
            partial_flat_base,
            reduced_shard,
            reduced_payload,
            reduced_scale,
            gather_payload_base,
            gather_scale_base,
            gathered_output,
            rank,
        ).launch(
            grid=(service_grid + (local_workers if has_local else 0), 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch

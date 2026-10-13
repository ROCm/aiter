# SPDX-License-Identifier: Apache-2.0
"""Windowed GEMM2 and TP communication family."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from ....mxfp4_gemm_common import global_typed_ptr
from ....mxmoe_dispatcher import _spart_output_tile_index
from .collectives import (
    CPOL_COHERENT,
    buffer_tensor_from_addr,
    decode_scaled_fp8_f32,
    e8m0_scale,
    emit_tp_all_gather,
    emit_tp_reduce_scatter,
    load_bf16,
    load_e8m0_scale,
    load_fp8_words,
    mxfp4_e8m0_scale,
    pack_fp8_words,
    pack_mxfp4_8,
    store_buffer,
    store_fp8_words,
)
from .config import BLOCK, WindowConfig
from .producer import compile_window_producer


def _producer_spatial_tag(config: WindowConfig) -> str:
    partition = config.producer_spatial_partition
    if not partition:
        return ""
    tag = f"_spart{partition // 100}x{partition % 100}"
    alternate = config.producer_spatial_alt_partition
    if alternate:
        tag += (
            f"to{alternate // 100}x{alternate % 100}"
            f"gt{config.producer_spatial_switch_rows}"
        )
    return tag


def _payload_tags(config: WindowConfig) -> str:
    tags = ""
    if config.partial_payload_bits != 8:
        tags += "_ppb4_native_e2m1v1"
    if config.reduced_payload_bits != 8:
        tags += "_rpb4_native_e2m1v1"
    return tags


@flyc.jit
def _producer_tile_index(config: WindowConfig, worker, m_blocks, valid_rows):
    tiles_per_window = config.tiles_per_window
    partition = config.producer_spatial_partition
    if const_expr(partition):
        group_num = fx.Int32(partition // 100)
        if const_expr(config.producer_spatial_alt_partition):
            group_num = (
                valid_rows > fx.Int32(config.producer_spatial_switch_rows)
            ).select(
                fx.Int32(config.producer_spatial_alt_partition // 100),
                group_num,
            )
        return _spart_output_tile_index(
            worker,
            fx.Int32(m_blocks),
            tiles_per_window,
            group_num,
            partition % 100,
        )
    return (
        worker // fx.Int32(tiles_per_window),
        worker % fx.Int32(tiles_per_window),
    )


def _compose_compute(config: WindowConfig, window_index: int):
    tiles_per_window = config.tiles_per_window

    def compose(*, module_name, emit_gemm2_tile, shared_storage):
        kernel_name = (
            f"{module_name}_window_{window_index}{_producer_spatial_tag(config)}"
        )

        @flyc.kernel(
            name=kernel_name,
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
            tid = fx.Int32(gpu.thread_id("x"))
            lane = tid % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tid // fx.Int32(64))
            lds = fx.SharedAllocator().allocate(shared_storage).peek()
            valid_rows = global_typed_ptr(fx.Int64(ptrtoint(num_valid_ids)), T.i32)[0]
            m_block, n_local = _producer_tile_index(
                config, worker, size_expert_ids, valid_rows
            )
            n_block = fx.Int32(window_index * tiles_per_window) + n_local
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

        launch.__name__ = f"launch_{kernel_name}"
        return flyc.jit(launch)

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
def _emit_local(
    config: WindowConfig,
    route,
    partial,
    shared,
    worker,
    local_stride,
    shared_column_offset: fx.Constexpr[int],
):
    shape = config.shape
    m = config.m
    window = config.window
    rows_per_cta = config.local_rows_per_cta
    groups_per_row = config.groups_per_row
    tid = fx.Int32(gpu.thread_id("x"))
    threads_per_row = BLOCK // rows_per_cta
    row_in_cta = tid // fx.Int32(threads_per_row)
    row_tid = tid % fx.Int32(threads_per_row)
    token_stride = local_stride * fx.Int32(rows_per_cta)
    token_start = worker * fx.Int32(rows_per_cta)
    # Keep loop control CTA-uniform.  Adding row_in_cta to the range start makes
    # different waves take distinct loop induction paths and serializes this
    # otherwise independent two-row schedule in the generated kernel.
    for token_base in range(token_start, fx.Int32(m), token_stride):
        token = token_base + row_in_cta
        columns_per_pass = threads_per_row * 8
        column_passes = (window + columns_per_pass - 1) // columns_per_pass
        for column_pass in range_constexpr(column_passes):
            column = row_tid * fx.Int32(8) + fx.Int32(column_pass * columns_per_pass)
            if (token < fx.Int32(m)) & (column < fx.Int32(window)):
                route_row_bytes = config.route_row_bytes
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
                        cache_modifier=config.local_load_cache_modifier,
                    )
                    scale = load_e8m0_scale(
                        route_scale_row,
                        fx.Int32(slot * route_row_bytes + window)
                        + column // fx.Int32(8),
                        config.local_load_cache_modifier,
                    )
                    values = decode_scaled_fp8_f32(words, scale)
                    acc = acc + fx.Vector.from_elements(values, fx.Float32)

                if const_expr(shape.add_shared and config.gather_output):
                    shared_row = buffer_tensor_from_addr(
                        fx.Int64(ptrtoint(shared))
                        + fx.Int64(token) * fx.Int64(shape.model_dim * 2),
                        fx.BFloat16,
                        shape.model_dim * 2,
                    )
                    acc = acc + load_bf16(
                        shared_row, shared_column_offset + column, 8, 0
                    ).to(fx.Float32)

                lane = row_tid & fx.Int32(63)
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
                if const_expr(config.partial_payload_bits == 4):
                    e8m0, quant_scale = mxfp4_e8m0_scale(local_max)
                    packed = pack_mxfp4_8(acc, quant_scale)
                else:
                    e8m0, quant_scale = e8m0_scale(local_max)
                    packed = pack_fp8_words(acc, quant_scale, 2)
                payload_row = buffer_tensor_from_addr(
                    fx.Int64(ptrtoint(partial))
                    + fx.Int64(token) * fx.Int64(config.partial_row_bytes),
                    fx.Int32,
                    config.partial_row_bytes,
                )
                if const_expr(config.partial_payload_bits == 4):
                    store_buffer(
                        payload_row,
                        column // fx.Int32(8),
                        packed,
                        fx.Int32,
                        cache_modifier=CPOL_COHERENT,
                    )
                else:
                    store_fp8_words(
                        payload_row,
                        column,
                        packed,
                        2,
                        cache_modifier=CPOL_COHERENT,
                    )
                if lane & fx.Int32(3) == fx.Int32(0):
                    scale_row = buffer_tensor_from_addr(
                        fx.Int64(ptrtoint(partial))
                        + fx.Int64(config.partial_payload_bytes)
                        + fx.Int64(token) * fx.Int64(groups_per_row),
                        fx.Int8,
                        groups_per_row,
                    )
                    store_buffer(
                        scale_row,
                        column // fx.Int32(32),
                        e8m0.to(fx.Int8),
                        fx.Int8,
                        cache_modifier=CPOL_COHERENT,
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
    has_all_gather = config.gather_output and window_index >= 3
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
        local_rows_tag = ""
        if config.local_rows_per_cta != 1:
            local_rows_tag = f"_lr{config.local_rows_per_cta}"
        kernel_name = (
            f"gemm2_tp_window_pipeline_v9_{shape.tag}"
            f"_{'ar' if config.gather_output else 'rs'}"
            f"_cycle_p{window_index}_sr{shard_rows}"
            f"_t{config.tile_m}x{config.tile_n}x{config.tile_k}"
            f"_sbm{config.sort_block_m}_w{window}_lw{local_workers}"
            f"{local_rows_tag}"
            f"{_producer_spatial_tag(config)}"
            f"{_payload_tags(config)}"
            f"_rsg{reduce_scatter_grid}_agg{all_gather_grid}"
            f"_rs{int(has_reduce_scatter)}ag{int(has_all_gather)}"
            f"_co{config.collective_order}"
            f"_rsc{config.reduce_scatter_load_cache_modifier}"
            f"_agc{config.all_gather_load_cache_modifier}"
        )

        @flyc.kernel(
            name=kernel_name,
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
            local_rows: fx.Int32,
            rank_offset: fx.Int32,
        ):
            linear = fx.Int32(gpu.block_id("x"))
            tid = fx.Int32(gpu.thread_id("x"))
            lane = tid % fx.Int32(64)
            wave = rocdl.readfirstlane(T.i32, tid // fx.Int32(64))
            lds = fx.SharedAllocator().allocate(shared_storage).peek()
            valid_rows = global_typed_ptr(fx.Int64(ptrtoint(num_valid_ids)), T.i32)[0]
            producer_workers = fx.Int32(size_expert_ids) * fx.Int32(tiles_per_window)
            active_local_workers = (producer_workers < fx.Int32(local_workers)).select(
                producer_workers, fx.Int32(local_workers)
            )

            def emit_compute(worker):
                m_block, n_local = _producer_tile_index(
                    config, worker, size_expert_ids, valid_rows
                )
                n_block = fx.Int32(window_index * tiles_per_window) + n_local
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

            def emit_reduce(worker):
                if worker < fx.Int32(reduce_scatter_grid):
                    emit_tp_reduce_scatter(
                        partial_flat_base,
                        reduced_shard,
                        local_shared,
                        reduced_payload,
                        reduced_scale,
                        rank,
                        local_rows,
                        rank_offset,
                        worker,
                        tokens=m,
                        output_width=shape.model_dim,
                        payload_width=window,
                        shard_rows=shard_rows,
                        tp=shape.tp_size,
                        block=BLOCK,
                        reduce_scatter_grid=reduce_scatter_grid,
                        load_cache_modifier=config.reduce_scatter_load_cache_modifier,
                        partial_payload_bits=config.partial_payload_bits,
                        reduced_payload_bits=config.reduced_payload_bits,
                        output_column_offset=(window_index - 2) * window,
                        add_shared=shape.add_shared and not config.gather_output,
                        publish_gather=config.gather_output,
                    )

            def emit_gather(worker):
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
                        load_cache_modifier=config.all_gather_load_cache_modifier,
                        reduced_payload_bits=config.reduced_payload_bits,
                        output_column_offset=(window_index - 3) * window,
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

            local_active = compute_worker < active_local_workers
            if const_expr(service_grid > 0):
                local_active = is_compute & local_active
            if local_active:
                _emit_local(
                    config,
                    local_route,
                    local_partial,
                    local_shared,
                    compute_worker,
                    active_local_workers,
                    (window_index - 1) * window,
                )

            if const_expr(service_grid > 0):  # noqa: SIM102
                if is_service:
                    if const_expr(
                        config.collective_order == "gather_first" and has_all_gather
                    ):
                        emit_gather(service_worker)
                        emit_reduce(service_worker)
                    elif const_expr(
                        config.collective_order == "alternating" and has_all_gather
                    ):
                        # Keep the service population fixed while allowing both
                        # independent collective windows to make early progress.
                        if service_worker & fx.Int32(1) == fx.Int32(0):
                            emit_gather(service_worker)
                            emit_reduce(service_worker)
                        else:
                            emit_reduce(service_worker)
                            emit_gather(service_worker)
                    else:
                        if const_expr(has_reduce_scatter):
                            emit_reduce(service_worker)
                        if const_expr(has_all_gather):
                            emit_gather(service_worker)

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
            local_rows,
            rank_offset,
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
                local_rows,
                rank_offset,
            ).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

        launch.__name__ = f"launch_{kernel_name}"
        return flyc.jit(launch)

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
    local_window: int | None,
    reduce_window: int | None,
    gather_window: int | None,
):
    """Compile the fixed pipeline tail without reserving GEMM LDS."""
    shape = config.shape
    m = config.m
    shard_rows = config.shard_rows
    window = config.window
    local_workers = config.local_workers
    reduce_scatter_grid = config.drain_reduce_scatter_grid or config.reduce_scatter_grid
    all_gather_grid = config.drain_all_gather_grid or config.all_gather_grid
    has_local = local_window is not None
    has_reduce_scatter = reduce_window is not None
    has_all_gather = config.gather_output and gather_window is not None
    service_grid = max(
        reduce_scatter_grid if has_reduce_scatter else 0,
        all_gather_grid if has_all_gather else 0,
    )
    local_rows_tag = ""
    if config.local_rows_per_cta != 1:
        local_rows_tag = f"_lr{config.local_rows_per_cta}"
    kernel_name = (
        f"gemm2_tp_window_pipeline_v9_{shape.tag}_drain_sr{shard_rows}"
        f"_{'ar' if config.gather_output else 'rs'}"
        f"_w{window}_lw{local_workers}"
        f"{local_rows_tag}"
        f"{_payload_tags(config)}"
        f"_rsg{reduce_scatter_grid}_agg{all_gather_grid}"
        f"_l{int(has_local)}"
        f"rs{int(has_reduce_scatter)}ag{int(has_all_gather)}"
        f"_co{config.collective_order}"
        f"_rsc{config.reduce_scatter_load_cache_modifier}"
        f"_agc{config.all_gather_load_cache_modifier}"
    )

    @flyc.kernel(
        name=kernel_name,
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
        local_rows: fx.Int32,
        rank_offset: fx.Int32,
    ):
        worker = fx.Int32(gpu.block_id("x"))

        def emit_reduce(worker):
            if worker < fx.Int32(reduce_scatter_grid):
                emit_tp_reduce_scatter(
                    partial_flat_base,
                    reduced_shard,
                    shared,
                    reduced_payload,
                    reduced_scale,
                    rank,
                    local_rows,
                    rank_offset,
                    worker,
                    tokens=m,
                    output_width=shape.model_dim,
                    payload_width=window,
                    shard_rows=shard_rows,
                    tp=shape.tp_size,
                    block=BLOCK,
                    reduce_scatter_grid=reduce_scatter_grid,
                    load_cache_modifier=config.reduce_scatter_load_cache_modifier,
                    partial_payload_bits=config.partial_payload_bits,
                    reduced_payload_bits=config.reduced_payload_bits,
                    output_column_offset=(reduce_window or 0) * window,
                    add_shared=shape.add_shared and not config.gather_output,
                    publish_gather=config.gather_output,
                )

        def emit_gather(worker):
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
                    load_cache_modifier=config.all_gather_load_cache_modifier,
                    reduced_payload_bits=config.reduced_payload_bits,
                    output_column_offset=(gather_window or 0) * window,
                )

        if const_expr(
            config.collective_order == "gather_first"
            and has_reduce_scatter
            and has_all_gather
        ):
            emit_gather(worker)
            emit_reduce(worker)
        elif const_expr(
            config.collective_order == "alternating"
            and has_reduce_scatter
            and has_all_gather
        ):
            # Alternate operation order without increasing the service grid.
            if worker & fx.Int32(1) == fx.Int32(0):
                emit_gather(worker)
                emit_reduce(worker)
            else:
                emit_reduce(worker)
                emit_gather(worker)
        else:
            if const_expr(has_reduce_scatter):
                emit_reduce(worker)
            if const_expr(has_all_gather):
                emit_gather(worker)

        if const_expr(has_local):
            local_worker = worker - fx.Int32(service_grid)
            if worker >= fx.Int32(service_grid):
                _emit_local(
                    config,
                    route,
                    partial,
                    shared,
                    local_worker,
                    fx.Int32(local_workers),
                    (local_window or 0) * window,
                )

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
        local_rows,
        rank_offset,
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
            local_rows,
            rank_offset,
        ).launch(
            grid=(service_grid + (local_workers if has_local else 0), 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    launch.__name__ = f"launch_{kernel_name}"
    return flyc.jit(launch)

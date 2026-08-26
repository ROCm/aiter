# SPDX-License-Identifier: Apache-2.0
"""Persistent-window communication-fused Stage2 kernels."""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.expr.typing import Vector as Vec

from .. import buffer_ops
from .. import communication_ops_utils as comm_ops
from ..mixed_moe_gemm_2stage_common import compile_mixed_moe_gemm2_common
from .collectives import (
    decode_scaled_fp8_f32,
    e8m0_scale,
    emit_tp_all_gather,
    emit_tp_reduce_scatter,
    load_e8m0_scale,
    load_fp8_words,
    pack_fp8_words,
    store_fp8_words,
)
from .sync import peer_base


H = 7168
I = 384
E = 384
TOPK = 6
TP = 8
SLOTS = 2
BLOCK = 256
WORKER_EPOCH_OFFSET = 8


@dataclass(frozen=True)
class Config:
    m: int
    tile_m: int
    tile_n: int
    tile_k: int
    sort_block_m: int
    window: int
    local_workers: int
    service_grid: int

    @property
    def shard_rows(self) -> int:
        return self.m // TP

    @property
    def tiles_per_window(self) -> int:
        return self.window // self.tile_n

    @property
    def groups_per_row(self) -> int:
        return self.window // 32

    @property
    def phases(self) -> int:
        return H // self.window

    @property
    def phase_done_offset(self) -> int:
        return WORKER_EPOCH_OFFSET + self.service_grid * 8

    @property
    def partial_ready_offset(self) -> int:
        return self.phase_done_offset + self.phases * 8

    @property
    def reduced_ready_offset(self) -> int:
        return self.partial_ready_offset + self.phases * 8

    @property
    def phase_gate_offset(self) -> int:
        return self.reduced_ready_offset + self.phases * 8

    @property
    def state_bytes(self) -> int:
        return (self.phase_gate_offset + self.phases * 8 + 255) // 256 * 256

    @property
    def partial_stride(self) -> int:
        return self.m * (self.window + self.window // 32)

    @property
    def reduced_payload_stride(self) -> int:
        return self.shard_rows * self.window

    @property
    def reduced_scale_stride(self) -> int:
        return self.shard_rows * self.groups_per_row


@dataclass(frozen=True)
class Schedule:
    producer_workers_per_n_tile: int | None = None
    producer_ctas_per_cu_limit: int = 1


_DEFAULT_SCHEDULE = Schedule()
_PRODUCTION_SCHEDULES = {
    # Four N tiles per 1024-wide window: 112 * 4 = 448 producer CTAs.
    2048: Schedule(
        producer_workers_per_n_tile=112,
        producer_ctas_per_cu_limit=2,
    ),
    # Four N tiles per 1024-wide window: 104 * 4 = 416 producer CTAs.
    16384: Schedule(
        producer_workers_per_n_tile=104,
        producer_ctas_per_cu_limit=2,
    ),
}


def production_schedule(config: Config) -> Schedule:
    return _PRODUCTION_SCHEDULES.get(config.m, _DEFAULT_SCHEDULE)


def _config_cache_tag(kind: str, config: Config, *extra: int):
    """Return only stable scalar values used by a FlyDSL specialization.

    FlyDSL's runtime cache does not reliably distinguish JIT launchers by a
    captured dataclass instance.  Keep the complete persistent-window config
    in an explicitly referenced tuple so differently sized runners cannot
    reuse one another's compiled launcher.
    """
    return (
        kind,
        config.m,
        config.tile_m,
        config.tile_n,
        config.tile_k,
        config.sort_block_m,
        config.window,
        config.local_workers,
        config.service_grid,
        *extra,
    )


def _compile_compute(
    config: Config,
    window: int,
    compose=None,
    *,
    persistent_m_workers_per_n_tile: int | None = None,
):
    return compile_mixed_moe_gemm2_common(
        model_dim=H,
        inter_dim=I,
        experts=E,
        topk=TOPK,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        doweight_stage2=True,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="fp8",
        accumulate=False,
        persist_m=(
            -1 if persistent_m_workers_per_n_tile is not None else 1
        ),
        sort_block_m=config.sort_block_m,
        _n_tile_range=(
            window * config.tiles_per_window,
            (window + 1) * config.tiles_per_window,
        ),
        _persistent_cu_num=persistent_m_workers_per_n_tile,
        _compose_entry=compose,
    )


@functools.cache
def compile_stage2_compute(config: Config, window: int):
    """Compile one compact Stage2 window."""
    return _compile_compute(config, window)


@functools.cache
def compile_stage2_compute_bounded(
    config: Config,
    window: int,
    workers_per_n_tile: int,
    ctas_per_cu_limit: int = 1,
):
    """Compile one compact window with an explicit producer CTA budget.

    ``ctas_per_cu_limit=1`` requires a fully resident grid.  A larger explicit
    budget allows a service-first kernel to reserve CUs while producer CTAs
    drain through the remaining slots.
    """
    from aiter.jit.utils.chip_info import get_cu_num

    if not isinstance(workers_per_n_tile, int) or workers_per_n_tile < 1:
        raise ValueError(
            f"workers_per_n_tile must be int >= 1, got {workers_per_n_tile}"
        )
    if not isinstance(ctas_per_cu_limit, int) or ctas_per_cu_limit < 1:
        raise ValueError(
            f"ctas_per_cu_limit must be int >= 1, got {ctas_per_cu_limit}"
        )
    producer_ctas = config.tiles_per_window * workers_per_n_tile
    resident_limit = get_cu_num() * ctas_per_cu_limit
    if producer_ctas > resident_limit:
        raise ValueError(
            "bounded Stage2 grid exceeds its explicit CTA budget: "
            f"producer_ctas={producer_ctas}, resident_limit={resident_limit}"
        )
    return _compile_compute(
        config,
        window,
        persistent_m_workers_per_n_tile=workers_per_n_tile,
    )


def _emit_local(config: Config, route, partial, shared, worker):
    m = config.m
    window = config.window
    groups_per_row = config.groups_per_row
    work = scf.ForOp(
        arith.index_cast(T.index, worker),
        arith.constant(m, index=True),
        arith.constant(config.local_workers, index=True),
    )
    with ir.InsertionPoint(work.body):
        token = arith.index_cast(T.i32, work.induction_variable)
        tid = fx.Int32(gpu.thread_idx.x)
        column = tid * fx.Int32(8)
        active = scf.IfOp(arith.cmpi(CmpIPredicate.ult, column, fx.Int32(window)))
        with ir.InsertionPoint(active.then_block):
            route_row_bytes = window + window // 8
            route_row = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(route))
                + fx.Int64(token) * fx.Int64(TOPK * route_row_bytes),
                num_records_bytes=TOPK * route_row_bytes,
            )
            acc = fx.Vector.filled(8, 0.0, fx.Float32)
            for slot in range_constexpr(TOPK):
                words = load_fp8_words(
                    route_row,
                    fx.Int32(slot * (route_row_bytes // 4))
                    + column // fx.Int32(4),
                    word_count=2,
                    load_width=2,
                    cache_modifier=2,
                )
                scale = load_e8m0_scale(
                    route_row,
                    fx.Int32(slot * route_row_bytes + window)
                    + column // fx.Int32(8),
                    2,
                )
                values = decode_scaled_fp8_f32(words, scale)
                acc = acc + fx.Vector.from_elements(values, fx.Float32)

            shared_row = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(shared)) + fx.Int64(token) * fx.Int64(H * 2),
                num_records_bytes=H * 2,
            )
            shared_values = fx.Vector(
                buffer_ops.buffer_load(
                    shared_row,
                    column,
                    vec_width=8,
                    dtype=T.bf16,
                    cache_modifier=2,
                )
            ).extf(T.vec(8, T.f32))
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
            payload_row = buffer_ops.create_buffer_resource_from_addr(
                fx.Int64(ptrtoint(partial)) + fx.Int64(token) * fx.Int64(window),
                num_records_bytes=window,
            )
            store_fp8_words(payload_row, column, packed, 2)
            scale_leader = scf.IfOp(
                arith.cmpi(CmpIPredicate.eq, lane & fx.Int32(3), fx.Int32(0))
            )
            with ir.InsertionPoint(scale_leader.then_block):
                scale_row = buffer_ops.create_buffer_resource_from_addr(
                    fx.Int64(ptrtoint(partial))
                    + fx.Int64(m * window)
                    + fx.Int64(token) * fx.Int64(groups_per_row),
                    num_records_bytes=groups_per_row,
                )
                buffer_ops.buffer_store(
                    e8m0.to(fx.Int8),
                    scale_row,
                    column // fx.Int32(32),
                    offset_is_bytes=True,
                )
                scf.YieldOp([])
            scf.YieldOp([])
        scf.YieldOp([])


def _publish_partial(config: Config, state, phase: int):
    leader = scf.IfOp(
        arith.andi(
            arith.cmpi(
                CmpIPredicate.eq,
                fx.Int32(gpu.block_idx.x),
                fx.Int32(0),
            ),
            arith.cmpi(
                CmpIPredicate.eq,
                fx.Int32(gpu.thread_idx.x),
                fx.Int32(0),
            ),
        )
    )
    with ir.InsertionPoint(leader.then_block):
        ready = fx.Int64(ptrtoint(state)) + fx.Int64(
            config.partial_ready_offset + phase * 8
        )
        epoch = fx.Int64(comm_ops.load_i64_global(ready)) + fx.Int64(1)
        comm_ops.fence_system_release()
        comm_ops.store_i64_global_system(ready, epoch)
        scf.YieldOp([])


def _compose_persistent_producer(config: Config, phase: int):
    cache_tag = _config_cache_tag("producer", config, phase)

    def compose(*, module_name, emit_gemm2, allocator):
        @flyc.kernel(
            name=(
                f"flydsl_fused_moe_pwin_prod_p{phase}_sr{config.shard_rows}"
                f"_t{config.tile_m}x{config.tile_n}x{config.tile_k}"
                f"_sbm{config.sort_block_m}_w{config.window}"
                f"_lw{config.local_workers}_sg{config.service_grid}"
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
            state: fx.Pointer,
        ):
            worker = fx.Int32(gpu.block_idx.x)
            if phase > 0:
                _publish_partial(config, state, phase - 1)
            emit_gemm2(
                route_out,
                x,
                w,
                scale_x,
                scale_w,
                w,
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
                block_id=arith.index_cast(T.index, worker),
            )
            local_if = scf.IfOp(
                arith.cmpi(
                    CmpIPredicate.ult,
                    worker,
                    fx.Int32(config.local_workers),
                )
            )
            with ir.InsertionPoint(local_if.then_block):
                _emit_local(config, local_route, local_partial, local_shared, worker)
                scf.YieldOp([])

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
            state,
            stream,
        ):
            _ = cache_tag
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()
            grid = arith.index_cast(T.index, size_expert_ids) * arith.constant(
                config.tiles_per_window, index=True
            )
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
                state,
            ).launch(grid=(grid, 1, 1), block=(BLOCK, 1, 1), stream=stream)

        return launch

    return compose


@functools.cache
def compile_persistent_cycle(config: Config, phase: int):
    return _compile_compute(
        config,
        phase + 1,
        _compose_persistent_producer(config, phase),
    )


@functools.cache
def compile_persistent_drain(config: Config):
    cache_tag = _config_cache_tag("drain", config)

    @flyc.kernel(
        name=(
            f"flydsl_fused_moe_pwin_drain_sr{config.shard_rows}"
            f"_w{config.window}"
            f"_lw{config.local_workers}_sg{config.service_grid}"
        ),
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        route: fx.Pointer,
        partial: fx.Pointer,
        shared: fx.Pointer,
        state: fx.Pointer,
    ):
        _publish_partial(config, state, config.phases - 2)
        _emit_local(config, route, partial, shared, fx.Int32(gpu.block_idx.x))

    @flyc.jit
    def launch(route, partial, shared, state, stream):
        _ = cache_tag
        kernel(route, partial, shared, state).launch(
            grid=(config.local_workers, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch


@functools.cache
def compile_persistent_final_publish(config: Config):
    cache_tag = _config_cache_tag("final_publish", config)

    @flyc.kernel(
        name=(
            f"flydsl_fused_moe_pwin_publish_w{config.window}"
            f"_sg{config.service_grid}"
        ),
        known_block_size=[64, 1, 1],
    )
    def kernel(state: fx.Pointer):
        if fx.Int32(gpu.thread_idx.x) == fx.Int32(0):
            ready = fx.Int64(ptrtoint(state)) + fx.Int64(
                config.partial_ready_offset + (config.phases - 1) * 8
            )
            epoch = fx.Int64(comm_ops.load_i64_global(ready)) + fx.Int64(1)
            comm_ops.fence_system_release()
            comm_ops.store_i64_global_system(ready, epoch)

    @flyc.jit
    def launch(state, stream):
        _ = cache_tag
        kernel(state).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


def _byte_ptr(addr):
    pointer = fx.PointerType.get(
        fx.Uint8.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=1,
    )
    return fx.inttoptr(pointer, fx.Int64(addr))


def _wait_agent(addr, expected):
    def load():
        return llvm_d.LoadOp(
            ir.IntegerType.get_signless(64),
            comm_ops._to_ptr_global(addr),
            alignment=8,
            volatile_=True,
            ordering=llvm_d.AtomicOrdering.monotonic,
            syncscope=fx.rocdl.SyncScope.AgentOneAs,
        ).result

    loop = scf.WhileOp([T.i64], [load()])
    before = ir.Block.create_at_start(loop.before, [T.i64])
    after = ir.Block.create_at_start(loop.after, [T.i64])
    with ir.InsertionPoint(before):
        current = before.arguments[0]
        waiting = arith.CmpIOp(
            arith.CmpIPredicate.slt, current, arith.unwrap(expected)
        ).result
        scf.ConditionOp(waiting, [current])
    with ir.InsertionPoint(after):
        llvm_d.InlineAsmOp(
            None,
            [],
            "s_sleep 1",
            "",
            has_side_effects=True,
        )
        scf.YieldOp([load()])
    return loop.results[0]


def _store_agent(addr, value):
    llvm_d.StoreOp(
        arith.unwrap(value),
        comm_ops._to_ptr_global(addr),
        alignment=8,
        ordering=llvm_d.AtomicOrdering.release,
        syncscope=fx.rocdl.SyncScope.AgentOneAs,
    )


@functools.cache
def compile_stage2_service(config: Config):
    cache_tag = _config_cache_tag("service", config)
    m = config.m
    window = config.window
    shard_rows = config.shard_rows
    service_grid = config.service_grid
    partial_stride = config.partial_stride
    reduced_payload_stride = config.reduced_payload_stride
    reduced_scale_stride = config.reduced_scale_stride

    @fx.struct
    class SharedStorage:
        epoch: fx.Array[fx.Int64, 1, 16]

    @flyc.kernel(
        name=(
            f"flydsl_fused_moe_pwin_service_sr{shard_rows}_w{window}"
            f"_sg{service_grid}"
        ),
        known_block_size=[BLOCK, 1, 1],
    )
    def kernel(
        state: fx.Pointer,
        state_flat_base: fx.Int64,
        partial_flat_base: fx.Int64,
        reduced_payload_flat_base: fx.Int64,
        reduced_scale_flat_base: fx.Int64,
        output: fx.Pointer,
        reduced_payloads: fx.Pointer,
        reduced_scales: fx.Pointer,
        rank: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        worker = fx.Int32(gpu.block_idx.x)
        local_state = fx.Int64(ptrtoint(state))
        epoch_scratch = fx.recast_iter(
            fx.Int64,
            fx.SharedAllocator().allocate(SharedStorage).peek().epoch.ptr,
        )
        epoch_view = fx.make_view(epoch_scratch, fx.make_layout(1, 1))

        if tid == fx.Int32(0):
            worker_epoch = (
                local_state
                + fx.Int64(WORKER_EPOCH_OFFSET)
                + fx.Int64(worker) * fx.Int64(8)
            )
            expected = fx.Int64(comm_ops.load_i64_global(worker_epoch)) + fx.Int64(1)
            if worker == fx.Int32(0):
                _store_agent(local_state, expected)
            epoch = fx.Int64(_wait_agent(local_state, expected))
            _store_agent(worker_epoch, epoch)
            fx.ptr_store(Vec.from_elements([epoch], fx.Int64), epoch_scratch)
        gpu.barrier()
        epoch = Vec(epoch_view.load())[0]

        for phase in range(config.phases):
            partial_ready = config.partial_ready_offset + phase * 8
            reduced_ready = config.reduced_ready_offset + phase * 8
            gate = local_state + fx.Int64(config.phase_gate_offset + phase * 8)
            partial_gate = epoch * fx.Int64(2) - fx.Int64(1)
            reduced_gate = epoch * fx.Int64(2)

            if worker == fx.Int32(0):
                if tid < fx.Int32(TP):
                    comm_ops.wait_i64_system_until_at_least(
                        peer_base(state_flat_base, tid) + fx.Int64(partial_ready),
                        epoch,
                    )
                    comm_ops.fence_system_acquire()
                gpu.barrier()
                if tid == fx.Int32(0):
                    _store_agent(gate, partial_gate)
            if tid == fx.Int32(0):
                _wait_agent(gate, partial_gate)
                comm_ops.fence_agent_acquire()
            gpu.barrier()

            emit_tp_reduce_scatter(
                partial_flat_base + fx.Int64(phase * partial_stride),
                _byte_ptr(
                    fx.Int64(ptrtoint(output))
                    + fx.Int64(shard_rows * H * 2) * fx.Int64(rank)
                    + fx.Int64(phase * window * 2)
                ),
                _byte_ptr(
                    fx.Int64(ptrtoint(reduced_payloads))
                    + fx.Int64(phase * reduced_payload_stride)
                ),
                _byte_ptr(
                    fx.Int64(ptrtoint(reduced_scales))
                    + fx.Int64(phase * reduced_scale_stride)
                ),
                rank,
                worker,
                tokens=m,
                output_width=H,
                payload_width=window,
                shard_rows=shard_rows,
                tp=TP,
                block=BLOCK,
                reduce_scatter_grid=service_grid,
            )

            gpu.barrier()
            if tid == fx.Int32(0):
                comm_ops.fence_agent_release()
                done = fx.Int64(
                    comm_ops.atomic_add_agent(
                        local_state
                        + fx.Int64(config.phase_done_offset + phase * 8),
                        fx.Int64(1),
                    )
                )
                if done == epoch * fx.Int64(service_grid) - fx.Int64(1):
                    comm_ops.fence_agent_acquire()
                    comm_ops.fence_system_release()
                    comm_ops.store_i64_global_system(
                        local_state + fx.Int64(reduced_ready), epoch
                    )

            if worker == fx.Int32(0):
                if tid == fx.Int32(0):
                    _wait_agent(local_state + fx.Int64(reduced_ready), epoch)
                    comm_ops.fence_agent_acquire()
                gpu.barrier()
                if tid < fx.Int32(TP):
                    comm_ops.wait_i64_system_until_at_least(
                        peer_base(state_flat_base, tid) + fx.Int64(reduced_ready),
                        epoch,
                    )
                    comm_ops.fence_system_acquire()
                gpu.barrier()
                if tid == fx.Int32(0):
                    _store_agent(gate, reduced_gate)
            if tid == fx.Int32(0):
                _wait_agent(gate, reduced_gate)
                comm_ops.fence_agent_acquire()
            gpu.barrier()

            emit_tp_all_gather(
                reduced_payload_flat_base
                + fx.Int64(phase * reduced_payload_stride),
                reduced_scale_flat_base + fx.Int64(phase * reduced_scale_stride),
                _byte_ptr(
                    fx.Int64(ptrtoint(output)) + fx.Int64(phase * window * 2)
                ),
                rank,
                worker,
                output_width=H,
                payload_width=window,
                shard_rows=shard_rows,
                tp=TP,
                block=BLOCK,
                all_gather_grid=service_grid,
            )

    @flyc.jit
    def launch(
        state,
        state_flat_base,
        partial_flat_base,
        reduced_payload_flat_base,
        reduced_scale_flat_base,
        output,
        reduced_payloads,
        reduced_scales,
        rank,
        stream,
    ):
        _ = cache_tag
        kernel(
            state,
            state_flat_base,
            partial_flat_base,
            reduced_payload_flat_base,
            reduced_scale_flat_base,
            output,
            reduced_payloads,
            reduced_scales,
            rank,
        ).launch(
            grid=(service_grid, 1, 1),
            block=(BLOCK, 1, 1),
            stream=stream,
        )

    return launch

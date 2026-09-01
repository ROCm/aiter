# SPDX-License-Identifier: Apache-2.0
"""Single-launch Stage2 producer/consumer kernel.

Producer workgroups run the natural-grid route GEMM. Completion is tracked per
N tile; the final ``service_groups`` producers become communication consumers
and execute the selected direct, reduce-broadcast, or reduce-scatter/all-gather
path. This keeps the native GEMM tiling and avoids an artificial window.
"""

import functools
import hashlib
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, ptrtoint
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemPtr

from .... import communication_ops_utils as comm_ops
from ....mixed_moe_gemm_2stage_common import compile_mixed_moe_gemm2_common
from .gemm2_tp_producer import Gemm2TPComposition, RouteOutputEpilogue
from .gemm2_tp_service import (
    _atomic_add_i32_agent,
    _wait_i32_agent_until_at_least,
    emit_tile,
)
from .shape import Gemm2TPShape

BLOCK = 256
SLOTS = 2
PRODUCER_COUNTER_STRIDE = 64


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _byte_ptr(addr):
    pointer = fx.PointerType.get(
        fx.Uint8.ir_type,
        address_space=fx.AddressSpace.Global,
        alignment=1,
    )
    return fx.inttoptr(pointer, fx.Int64(addr))


@dataclass(frozen=True)
class Gemm2TPMegakernelConfig:
    """Configuration for the single-launch GEMM2 + TP collective kernel."""

    shape: Gemm2TPShape
    m: int
    tile_m: int = 16
    tile_n: int = 256
    tile_k: int = 128
    sort_block_m: int = 32
    compute_groups: int = 96
    block_threads: int = BLOCK
    vector_width: int = 16
    waves_per_eu: int = 0
    b_cache_modifier: int = 0
    route_store_scope: str = "device"
    local_load_cache_modifier: int = 1
    remote_load_cache_modifier: int = 1
    gather_load_cache_modifier: int = -1
    remote_store_cache_modifier: int = 0
    fp8_scale_exponent: int = 127
    n_tile_cohort: int = 0
    collective: str = "direct"
    service_groups: int = 1
    service_tile_group: int = 1
    producer_mode: str = "routes"
    flat_producer_grid: bool = False

    def __post_init__(self):
        if self.m <= 0:
            raise ValueError(f"m must be positive, got {self.m}")
        if self.tile_m <= 0 or self.tile_n <= 0 or self.tile_k <= 0:
            raise ValueError(
                "tile sizes must be positive, got "
                f"{(self.tile_m, self.tile_n, self.tile_k)}"
            )
        if self.shape.model_dim % self.tile_n:
            raise ValueError(
                f"model_dim={self.shape.model_dim} must be divisible by "
                f"tile_n={self.tile_n}"
            )
        if self.shape.inter_dim % self.tile_k:
            raise ValueError(
                f"inter_dim={self.shape.inter_dim} must be divisible by "
                f"tile_k={self.tile_k}"
            )
        if self.sort_block_m % self.tile_m:
            raise ValueError(
                "sort_block_m must be divisible by tile_m, got "
                f"sort_block_m={self.sort_block_m}, tile_m={self.tile_m}"
            )
        if self.tile_n % self.vector_width:
            raise ValueError(
                "tile_n must be divisible by vector_width, got "
                f"tile_n={self.tile_n}, vector_width={self.vector_width}"
            )
        if self.b_cache_modifier not in (0, 1, 2, 3):
            raise ValueError(
                "b_cache_modifier must be one of (0, 1, 2, 3), got "
                f"{self.b_cache_modifier}"
            )
        if not 0 <= self.waves_per_eu <= 10:
            raise ValueError(
                f"waves_per_eu must be in [0, 10], got {self.waves_per_eu}"
            )
        if self.compute_groups <= 0:
            raise ValueError(
                f"compute_groups must be positive, got {self.compute_groups}"
            )
        if self.block_threads not in (128, 256):
            raise ValueError(
                f"block_threads must be 128 or 256, got {self.block_threads}"
            )
        num_waves = self.block_threads // 64
        if self.tile_n % (num_waves * 16):
            raise ValueError(
                "tile_n must provide an integral number of 16-column MFMA "
                "tiles per wave, got "
                f"tile_n={self.tile_n}, block_threads={self.block_threads}"
            )
        if self.route_store_scope not in ("default", "device"):
            raise ValueError(
                "route_store_scope must be 'default' or 'device', got "
                f"{self.route_store_scope!r}"
            )
        if self.local_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("local_load_cache_modifier must be in [0, 3]")
        if self.remote_load_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("remote_load_cache_modifier must be in [0, 3]")
        if self.gather_load_cache_modifier not in (-1, 0, 1, 2, 3):
            raise ValueError("gather_load_cache_modifier must be -1 or in [0, 3]")
        if self.remote_store_cache_modifier not in (0, 1, 2, 3):
            raise ValueError("remote_store_cache_modifier must be in [0, 3]")
        if self.vector_width not in (8, 16):
            raise ValueError(
                "production GEMM2 TP megakernel requires vector_width 8 or 16"
            )
        if not 0 <= self.fp8_scale_exponent <= 254:
            raise ValueError(
                "fp8_scale_exponent must be in [0, 254], got "
                f"{self.fp8_scale_exponent}"
            )
        if self.n_tile_cohort < 0:
            raise ValueError(
                f"n_tile_cohort must be non-negative, got {self.n_tile_cohort}"
            )
        if self.n_tile_cohort and self.n_tiles % self.n_tile_cohort:
            raise ValueError(
                "n_tile_cohort must divide n_tiles, got "
                f"n_tile_cohort={self.n_tile_cohort}, n_tiles={self.n_tiles}"
            )
        if self.collective not in (
            "direct",
            "rsag",
            "rs_broadcast",
        ):
            raise ValueError(
                "collective must be 'direct', 'rsag', or 'rs_broadcast', got "
                f"{self.collective!r}"
            )
        if not 1 <= self.service_groups <= self.compute_groups:
            raise ValueError(
                "service_groups must be in [1, compute_groups], got "
                f"service_groups={self.service_groups}, "
                f"compute_groups={self.compute_groups}"
            )
        if self.collective != "rsag" and self.service_groups != 1:
            raise ValueError(
                "direct and rs_broadcast collectives require service_groups=1"
            )
        if self.collective == "rsag" and (
            self.service_groups not in (1, 2, 4, 8)
            or self.service_groups > self.shape.tp_size
            or self.shape.tp_size % self.service_groups
        ):
            raise ValueError(
                "rsag service_groups must be a supported divisor of TP, got "
                f"service_groups={self.service_groups}, "
                f"TP={self.shape.tp_size}"
            )
        if self.service_tile_group <= 0 or self.n_tiles % self.service_tile_group:
            raise ValueError(
                "service_tile_group must be a positive divisor of n_tiles, got "
                f"service_tile_group={self.service_tile_group}, "
                f"n_tiles={self.n_tiles}"
            )
        if self.service_tile_group > 1 and (
            self.collective != "rsag" or self.service_groups == 1
        ):
            raise ValueError(
                "grouped service synchronization requires collective='rsag' "
                "and service_groups > 1"
            )
        if self.producer_mode not in (
            "routes",
            "routes_fp8_fixed",
            "atomic_shared",
        ):
            raise ValueError(
                "producer_mode must be 'routes', 'routes_fp8_fixed', or "
                "'atomic_shared', got "
                f"{self.producer_mode!r}"
            )
        if self.flat_producer_grid and self.collective == "direct":
            raise ValueError("flat_producer_grid requires a dynamic collective path")
        if self.flat_producer_grid and self.n_tile_cohort:
            raise ValueError(
                "flat_producer_grid and n_tile_cohort are mutually exclusive"
            )
        if self.collective == "direct" and self.producer_mode != "routes":
            raise ValueError("direct production path requires producer_mode='routes'")
        if self.collective == "rs_broadcast" and self.producer_mode != "atomic_shared":
            raise ValueError(
                "rs_broadcast production path requires " "producer_mode='atomic_shared'"
            )
        if self.uses_rsag and self.m % self.shape.tp_size:
            raise ValueError(
                f"m={self.m} must be divisible by TP={self.shape.tp_size} "
                f"for collective={self.collective!r}"
            )
        if (
            self.uses_rsag
            and (self.m * self.tile_n // self.vector_width) % self.shape.tp_size
        ):
            raise ValueError("rsag vector items must divide evenly across TP ranks")

    @property
    def n_tiles(self) -> int:
        return self.shape.model_dim // self.tile_n

    @property
    def uses_rsag(self) -> bool:
        return self.collective in ("rsag", "rs_broadcast")

    @property
    def shared_bf16_partials(self) -> bool:
        return self.collective == "rs_broadcast"

    @property
    def single_pass_direct(self) -> bool:
        return bool(
            self.collective == "direct"
            and self.m * self.tile_n // self.vector_width <= self.block_threads
        )

    @property
    def producer_rows(self) -> int:
        route_rows = self.m * self.shape.topk
        # Sorting pads each non-empty expert independently to sort_block_m.
        max_sort_blocks = (
            route_rows
            if route_rows <= self.shape.experts
            else self.shape.experts
            + (route_rows - self.shape.experts) // self.sort_block_m
        )
        return max_sort_blocks * self.sort_block_m // self.tile_m

    @property
    def payload_bytes(self) -> int:
        return self.m * self.shape.model_dim * 2

    @property
    def partial_bytes(self) -> int:
        return self.m * self.shape.model_dim

    @property
    def reduced_shard_bytes(self) -> int:
        if not self.uses_rsag:
            return 0
        element_bytes = 2 if self.collective == "rs_broadcast" else 1
        return self.m * self.shape.model_dim * element_bytes // self.shape.tp_size

    @property
    def reduced_offset(self) -> int:
        return SLOTS * self.partial_bytes

    @property
    def route_offset(self) -> int:
        return self.reduced_offset + SLOTS * self.reduced_shard_bytes

    @property
    def route_bytes(self) -> int:
        if self.producer_mode == "routes":
            return self.m * self.shape.topk * self.shape.model_dim * 2
        if self.producer_mode == "routes_fp8_fixed":
            return self.m * self.shape.topk * self.shape.model_dim
        return self.payload_bytes

    @property
    def output_offset(self) -> int:
        if self.producer_mode in ("routes", "routes_fp8_fixed"):
            return self.route_offset + self.route_bytes
        return self.route_offset

    @property
    def producer_done_offset(self) -> int:
        return self.output_offset + self.payload_bytes

    @property
    def epoch_offset(self) -> int:
        return self.gather_service_done_offset + self.n_tiles * 8

    @property
    def service_done_offset(self) -> int:
        return self.producer_done_offset + self.n_tiles * PRODUCER_COUNTER_STRIDE

    @property
    def reduce_done_offset(self) -> int:
        return self.service_done_offset + self.n_tiles * 8

    @property
    def gather_service_done_offset(self) -> int:
        return self.reduce_done_offset + self.n_tiles * 8

    @property
    def rank_ready_offset(self) -> int:
        return self.epoch_offset + self.n_tiles * 8

    @property
    def flat_base_offset(self) -> int:
        return _align_up(
            self.gather_done_offset
            + (self.n_tiles * self.shape.tp_size * 4 if self.uses_rsag else 0),
            8,
        )

    @property
    def gather_done_offset(self) -> int:
        return self.owner_ready_offset + (
            self.n_tiles * self.shape.tp_size * 4
            if self.uses_rsag
            else self.n_tiles * 4
        )

    @property
    def owner_ready_offset(self) -> int:
        return self.reduced_collective_ready_offset + self.n_tiles * 4

    @property
    def reduced_collective_ready_offset(self) -> int:
        return self.collective_ready_offset + self.n_tiles * 4

    @property
    def collective_ready_offset(self) -> int:
        return self.rank_ready_offset + self.n_tiles * self.shape.tp_size * 4

    @property
    def workspace_bytes(self) -> int:
        return _align_up(self.flat_base_offset + 8, 256)


@functools.cache
def _compile_gemm2_tp_megakernel(
    config: Gemm2TPMegakernelConfig,
    specialized_rank: int,
):
    """Compile the production single-launch GEMM2 + TP pipeline."""

    shape = config.shape
    if not 0 <= specialized_rank < shape.tp_size:
        raise ValueError(f"invalid TP rank {specialized_rank}")

    n_tiles = config.n_tiles
    dynamic_producer = config.collective != "direct"
    gather_cache_tag = (
        "inherit"
        if config.gather_load_cache_modifier < 0
        else str(config.gather_load_cache_modifier)
    )
    rows_per_group = (
        config.producer_rows + config.compute_groups - 1
    ) // config.compute_groups
    launch_grid = (
        (config.compute_groups * n_tiles, 1, 1)
        if config.n_tile_cohort or config.flat_producer_grid
        else (n_tiles, config.compute_groups, 1)
    )
    cache_abi = "gemm2_tp_mega_v2"
    cache_config = hashlib.sha256(repr(config).encode()).hexdigest()[:16]

    def compose(*, module_name, emit_gemm2, allocator):
        @flyc.kernel(
            name=(
                f"{module_name}_gemm2_tp_mega_r{specialized_rank}_m{config.m}"
                f"_n{config.tile_n}_cg{config.compute_groups}"
                f"_bt{config.block_threads}_v{config.vector_width}"
                f"_bc{config.b_cache_modifier}"
                f"_rts{config.route_store_scope}"
                f"_glc{gather_cache_tag}"
                f"_rsc{config.remote_store_cache_modifier}"
                f"_fp8e{config.fp8_scale_exponent}"
                f"_ntc{config.n_tile_cohort}_{config.collective}"
                f"_sg{config.service_groups}_stg{config.service_tile_group}"
                f"_p{config.producer_mode}"
                f"_fpg{int(config.flat_producer_grid)}"
                f"_sbp{int(config.shared_bf16_partials)}"
            ),
            known_block_size=[config.block_threads, 1, 1],
        )
        def kernel(
            workspace: fx.Pointer,
            x: fx.Pointer,
            w: fx.Pointer,
            scale_x: fx.Pointer,
            scale_w: fx.Pointer,
            sorted_token_ids: fx.Pointer,
            expert_ids: fx.Pointer,
            sorted_weights: fx.Pointer,
            num_valid_ids: fx.Pointer,
            shared_partial: fx.Pointer,
            shared_partial_flat_base: fx.Int64,
            tokens: fx.Int32,
            model_dim: fx.Int32,
            inter_dim: fx.Int32,
            size_expert_ids: fx.Int32,
        ):
            local_workspace_base = fx.Int64(ptrtoint(workspace))
            route_output = _byte_ptr(
                local_workspace_base + fx.Int64(config.route_offset)
            )
            producer_output = (
                shared_partial
                if config.producer_mode == "atomic_shared"
                else route_output
            )
            tid = fx.Int32(gpu.thread_idx.x)
            base = allocator.get_base()

            def emit_gemm(block_id=None):
                emit_gemm2(
                    producer_output,
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
                    shared_partial,
                    tokens,
                    model_dim,
                    inter_dim,
                    size_expert_ids,
                    block_id=block_id,
                )

            def emit_service(n_tile, service_group):
                workspace_flat_base = fx.Int64(
                    comm_ops.load_i64_global(
                        local_workspace_base + fx.Int64(config.flat_base_offset)
                    )
                )
                emit_tile(
                    config,
                    workspace,
                    workspace_flat_base,
                    shared_partial,
                    shared_partial_flat_base,
                    specialized_rank,
                    n_tile,
                    tid,
                    service_group,
                    base,
                    hidden_dim=shape.model_dim,
                    topk=shape.topk,
                    tp_size=shape.tp_size,
                    slots=SLOTS,
                    producer_counter_stride=PRODUCER_COUNTER_STRIDE,
                )

            def emit_and_service(n_tile, block_id=None):
                def publish_completion():
                    completion_publisher = scf.IfOp(
                        arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                    )
                    with ir.InsertionPoint(completion_publisher.then_block):
                        ticket = fx.Int32(
                            _atomic_add_i32_agent(
                                local_workspace_base
                                + fx.Int64(config.producer_done_offset)
                                + fx.Int64(n_tile) * fx.Int64(PRODUCER_COUNTER_STRIDE),
                                fx.Int32(1),
                            )
                        )
                        service_marker_ptr = SmemPtr(base, 0, T.i32, shape=(1,))
                        service_begin = fx.Int32(
                            config.compute_groups - config.service_groups
                        )
                        is_service = arith.cmpi(
                            CmpIPredicate.uge,
                            ticket,
                            service_begin,
                        )
                        service_marker_ptr.store(
                            arith.select(
                                is_service,
                                ticket - service_begin + fx.Int32(1),
                                fx.Int32(0),
                            )
                        )
                        scf.YieldOp([])

                emit_gemm(block_id)
                # Fixed producers drain before publishing their completion ticket.
                if not dynamic_producer:
                    fx.rocdl.s_waitcnt(0)
                    gpu.barrier()
                publish_completion()
                gpu.barrier()

                service_marker = fx.Int32(SmemPtr(base, 0, T.i32, shape=(1,)).load())
                last_producer = scf.IfOp(
                    arith.cmpi(
                        CmpIPredicate.ugt,
                        service_marker,
                        fx.Int32(0),
                    )
                )
                with ir.InsertionPoint(last_producer.then_block):
                    service_group = service_marker - fx.Int32(1)
                    if config.service_groups > 1:
                        producer_waiter = scf.IfOp(
                            arith.cmpi(CmpIPredicate.eq, tid, fx.Int32(0))
                        )
                        with ir.InsertionPoint(producer_waiter.then_block):
                            _wait_i32_agent_until_at_least(
                                local_workspace_base
                                + fx.Int64(config.producer_done_offset)
                                + fx.Int64(n_tile) * fx.Int64(PRODUCER_COUNTER_STRIDE),
                                fx.Int32(config.compute_groups),
                            )
                            comm_ops.fence_agent_acquire()
                            scf.YieldOp([])
                    else:
                        if tid == fx.Int32(0):
                            comm_ops.fence_agent_acquire()
                    gpu.barrier()
                    emit_service(n_tile, service_group)
                    scf.YieldOp([])

            if config.flat_producer_grid:
                physical_block = fx.Int32(gpu.block_idx.x)
                n_tiles_i32 = fx.Int32(config.n_tiles)
                n_tile_index = physical_block % n_tiles_i32
                compute_group = physical_block // n_tiles_i32
                emit_and_service(
                    n_tile_index,
                    arith.index_cast(
                        T.index,
                        compute_group * n_tiles_i32 + n_tile_index,
                    ),
                )
            elif config.n_tile_cohort:
                physical_block = fx.Int32(gpu.block_idx.x)
                cohort_size = fx.Int32(config.n_tile_cohort)
                cohort_span = fx.Int32(config.n_tile_cohort * config.compute_groups)
                cohort_base = physical_block // cohort_span * cohort_size
                within_cohort = physical_block % cohort_span
                n_tile_index = cohort_base + within_cohort % cohort_size
                compute_group = within_cohort // cohort_size
                emit_and_service(
                    n_tile_index,
                    arith.index_cast(
                        T.index,
                        compute_group * fx.Int32(config.n_tiles) + n_tile_index,
                    ),
                )
            else:
                emit_and_service(fx.Int32(gpu.block_idx.x))

        def launch(
            workspace,
            shared_partial,
            shared_partial_flat_base,
            x,
            w,
            scale_x,
            scale_w,
            sorted_token_ids,
            expert_ids,
            sorted_weights,
            num_valid_ids,
            tokens,
            model_dim,
            inter_dim,
            size_expert_ids,
            stream,
        ):
            allocator.finalized = False
            context = CompilationContext.get_current()
            with ir.InsertionPoint(context.gpu_module_body):
                allocator.finalize()
            if const_expr(config.waves_per_eu > 0):
                for op in context.gpu_module_body.operations:
                    if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                        op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                            T.i32, config.waves_per_eu
                        )
            kernel(
                workspace,
                x,
                w,
                scale_x,
                scale_w,
                sorted_token_ids,
                expert_ids,
                sorted_weights,
                num_valid_ids,
                shared_partial,
                shared_partial_flat_base,
                tokens,
                model_dim,
                inter_dim,
                size_expert_ids,
            ).launch(
                grid=launch_grid,
                block=(config.block_threads, 1, 1),
                stream=stream,
            )

        # Encode closure-captured configuration in the JIT name for cache isolation.
        launch.__name__ = f"launch_{cache_abi}_r{specialized_rank}_{cache_config}"
        return flyc.jit(launch)

    route_producer = config.producer_mode in ("routes", "routes_fp8_fixed")
    output_epilogue = (
        RouteOutputEpilogue(
            row_width=shape.model_dim,
            fp8_fixed=config.producer_mode == "routes_fp8_fixed",
            device_coherent=config.route_store_scope == "device",
        )
        if route_producer
        else None
    )
    composition = Gemm2TPComposition(
        compose_entry=compose,
        block_threads=config.block_threads,
        persistent_groups=(config.compute_groups if dynamic_producer else None),
        output_epilogue=output_epilogue,
        b_cache_modifier=config.b_cache_modifier,
    )
    return compile_mixed_moe_gemm2_common(
        model_dim=shape.model_dim,
        inter_dim=shape.inter_dim,
        experts=shape.experts,
        topk=shape.topk,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        doweight_stage2=True,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        accumulate=not route_producer,
        persist_m=0 if dynamic_producer else rows_per_group,
        sort_block_m=config.sort_block_m,
        waves_per_eu=config.waves_per_eu or None,
        _composition=composition,
    )


def compile_gemm2_tp_megakernel(
    config: Gemm2TPMegakernelConfig,
    specialized_rank: int,
):
    return _compile_gemm2_tp_megakernel(config, specialized_rank)

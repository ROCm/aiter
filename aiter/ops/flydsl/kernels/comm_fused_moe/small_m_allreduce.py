# SPDX-License-Identifier: MIT
"""Small-message TP8 BF16 AllReduce kernels.

This is a standalone M=1/2 specialization used to qualify a FlyDSL
implementation before embedding the same service body into GEMM2.  It mirrors
the useful shape of ``cross_device_reduce_1stage``:

* one 64-thread wave loads one rank's 16-byte BF16 pack;
* eight waves stage the eight ranks in LDS;
* wave 0 accumulates the packs in fixed rank order using FP32;
* every workgroup has independent cross-rank start/end epochs.

With the default 512-thread grid, H=7168 and M=1/2 need exactly one pack per
lane, so the generic double-buffered loop specializes to a single LDS stage.
The 256-thread megakernel-compatible shape may cap the grid and execute two
iterations instead.

The module also owns the production gfx950 M=1 single-launch Stage2 + TP8
AllReduce specialization.  Its per-N-tile last producer performs local route
reduction and immediately starts the rank collective, overlapping early tiles
with the remaining GEMM work.
"""

import functools
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T
from flydsl.utils.smem_allocator import SmemPtr

from .. import buffer_ops
from .. import communication_ops_utils as comm_ops
from ..mixed_moe_gemm_2stage_common import compile_mixed_moe_gemm2_common
from .sync import peer_base


TP = 8
BLOCK = 512
WAVE = 64
PACK = 8  # Eight BF16 values = one 16-byte transaction.
DEFAULT_HIDDEN = 7168
M1_INTER_DIM = 384
M1_EXPERTS = 384
M1_TOPK = 6
M1_BLOCK = 256
M1_PARTIAL_BUFFERS = 2
M1_LOCAL_LOAD_CACHE_MODIFIER = 0
M1_REMOTE_LOAD_CACHE_MODIFIER = 1


@dataclass(frozen=True, slots=True)
class Config:
    """Exact gfx950 TP8 small-M GEMM2 + AllReduce specialization."""

    m: int
    tile_m: int
    tile_n: int
    tile_k: int
    sort_block_m: int

    def validate(self) -> None:
        supported_m = (1, 2, 4, 8, 16)
        expected_tiles = (16, 512, 128, 32)
        actual_tiles = (
            self.tile_m,
            self.tile_n,
            self.tile_k,
            self.sort_block_m,
        )
        if self.m not in supported_m or actual_tiles != expected_tiles:
            raise ValueError(
                "small-M megakernel currently supports only "
                f"m in {supported_m} and "
                f"(tile_m,tile_n,tile_k,sort_block_m)={expected_tiles}; "
                f"got m={self.m}, tiles={actual_tiles}"
            )


def _global_ptr(addr):
    return llvm_d.IntToPtrOp(
        llvm_d.PointerType.get(address_space=1), arith.unwrap(addr)
    ).result


def _load_i32(addr):
    return llvm_d.LoadOp(
        T.i32, _global_ptr(addr), alignment=4
    ).result


def _store_i32(addr, value, *, system: bool):
    kwargs = {}
    if system:
        kwargs = {
            "ordering": llvm_d.AtomicOrdering.monotonic,
            "syncscope": fx.rocdl.SyncScope.OneAs,
        }
    llvm_d.StoreOp(
        arith.unwrap(value), _global_ptr(addr), alignment=4, **kwargs
    )


def _wait_i32_agent(addr, expected):
    def load():
        return llvm_d.LoadOp(
            T.i32,
            _global_ptr(addr),
            alignment=4,
            volatile_=True,
            ordering=llvm_d.AtomicOrdering.monotonic,
            syncscope=fx.rocdl.SyncScope.AgentOneAs,
        ).result

    loop = scf.WhileOp([T.i32], [load()])
    before = ir.Block.create_at_start(loop.before, [T.i32])
    after = ir.Block.create_at_start(loop.after, [T.i32])
    with ir.InsertionPoint(before):
        current = before.arguments[0]
        keep_waiting = arith.CmpIOp(
            arith.CmpIPredicate.ult, current, arith.unwrap(expected)
        ).result
        scf.ConditionOp(keep_waiting, [current])
    with ir.InsertionPoint(after):
        scf.YieldOp([load()])
    return loop.results[0]


def grid_blocks(
    m: int,
    hidden: int = DEFAULT_HIDDEN,
    block_threads: int = BLOCK,
    block_limit: int | None = None,
) -> int:
    elements = m * hidden
    if m not in (1, 2):
        raise ValueError(f"small-M AllReduce only supports M=1/2, got {m}")
    if block_threads not in (256, 512) or block_threads % TP != 0:
        raise ValueError(f"unsupported block size {block_threads}")
    lanes_per_rank = block_threads // TP
    natural = (elements + lanes_per_rank * PACK - 1) // (
        lanes_per_rank * PACK
    )
    blocks = natural if block_limit is None else min(natural, block_limit)
    if elements % (blocks * lanes_per_rank * PACK) != 0:
        raise ValueError(
            "small-M specialization requires an exact static iteration count: "
            f"elements={elements}, blocks={blocks}, lanes={lanes_per_rank}"
        )
    return blocks


def state_bytes(
    m: int,
    hidden: int = DEFAULT_HIDDEN,
    block_threads: int = BLOCK,
    block_limit: int | None = None,
) -> int:
    blocks = grid_blocks(m, hidden, block_threads, block_limit)
    # epoch[block], start[block][rank], end[block][rank], all i32.
    raw = blocks * (1 + TP + TP) * 4
    return (raw + 255) // 256 * 256


@functools.cache
def compile_bf16_one_stage(
    m: int,
    hidden: int = DEFAULT_HIDDEN,
    block_threads: int = BLOCK,
    block_limit: int | None = None,
):
    """Compile the exact-M TP8 BF16 one-stage AllReduce launcher."""

    blocks = grid_blocks(m, hidden, block_threads, block_limit)
    elements = m * hidden
    lanes_per_rank = block_threads // TP
    packs = elements // PACK
    step = blocks * lanes_per_rank
    iterations = packs // step
    epoch_offset = 0
    start_offset = blocks * 4
    end_offset = start_offset + blocks * TP * 4

    @fx.struct
    class SharedStorage:
        packs: fx.Array[fx.BFloat16, 2 * block_threads * PACK, 16]

    @flyc.kernel(
        name=(
            f"comm_fused_moe_small_m_bf16_ar_m{m}_h{hidden}"
            f"_t{block_threads}_b{blocks}"
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        input_flat_base: fx.Int64,
        state: fx.Pointer,
        state_flat_base: fx.Int64,
        output: fx.Pointer,
        rank: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_idx.x)
        block = fx.Int32(gpu.block_idx.x)
        source = tid // fx.Int32(lanes_per_rank)
        lane = tid - source * fx.Int32(lanes_per_rank)
        local_state = fx.Int64(ptrtoint(state))

        epoch_addr = (
            local_state + fx.Int64(epoch_offset) + fx.Int64(block) * fx.Int64(4)
        )
        expected = fx.Int32(_load_i32(epoch_addr)) + fx.Int32(1)

        # Match the custom AR's per-workgroup start handshake.  Each of the
        # first eight threads publishes this rank's epoch to one peer, then
        # waits for the corresponding peer epoch in local memory.
        if tid < fx.Int32(TP):
            slot = block * fx.Int32(TP) + rank
            remote_start = (
                peer_base(state_flat_base, tid)
                + fx.Int64(start_offset)
                + fx.Int64(slot) * fx.Int64(4)
            )
            _store_i32(remote_start, expected, system=True)
            local_slot = block * fx.Int32(TP) + tid
            _wait_i32_agent(
                local_state
                + fx.Int64(start_offset)
                + fx.Int64(local_slot) * fx.Int64(4),
                expected,
            )
        gpu.barrier()
        if tid == fx.Int32(0):
            _store_i32(epoch_addr, expected, system=False)

        # Each logical subgroup fetches the same 16-byte pack from one rank.
        # The capped 256-thread form uses the same double-buffered iteration
        # pattern as the C++ kernel; the default 512-thread form specializes to
        # one iteration for M=1/2.
        first_pack = block * fx.Int32(lanes_per_rank) + lane
        source_rsrc = buffer_ops.create_buffer_resource_from_addr(
            peer_base(input_flat_base, source),
            num_records_bytes=elements * 2,
        )
        first_values = fx.Vector(
            buffer_ops.buffer_load(
                source_rsrc,
                first_pack * fx.Int32(PACK),
                vec_width=PACK,
                dtype=T.bf16,
                cache_modifier=2,
            )
        )
        lds = fx.SharedAllocator().allocate(SharedStorage).peek().packs.ptr
        lds_base = tid * fx.Int32(PACK)
        for element in range_constexpr(PACK):
            fx.ptr_store(
                first_values[element], lds + lds_base + fx.Int32(element)
            )
        gpu.barrier()

        output_rsrc = buffer_ops.create_buffer_resource_from_addr(
            fx.Int64(ptrtoint(output)), num_records_bytes=elements * 2
        )
        for iteration in range_constexpr(iterations):
            current_buffer = iteration & 1
            next_buffer = current_buffer ^ 1
            pack_index = first_pack + fx.Int32(iteration * step)
            if source == fx.Int32(0):
                acc = fx.Vector.filled(PACK, 0.0, fx.Float32)
                for peer in range_constexpr(TP):
                    peer_base_index = (
                        fx.Int32(
                            current_buffer * block_threads * PACK
                            + peer * lanes_per_rank * PACK
                        )
                        + lane * fx.Int32(PACK)
                    )
                    peer_values = fx.Vector.from_elements(
                        [
                            fx.ptr_load(
                                lds
                                + peer_base_index
                                + fx.Int32(element)
                            )
                            for element in range_constexpr(PACK)
                        ],
                        fx.BFloat16,
                    ).extf(T.vec(PACK, T.f32))
                    acc = acc + peer_values
                buffer_ops.buffer_store(
                    acc.truncf(T.vec(PACK, T.bf16)),
                    output_rsrc,
                    pack_index * fx.Int32(PACK),
                )

            if iteration + 1 < iterations:
                next_pack = pack_index + fx.Int32(step)
                next_values = fx.Vector(
                    buffer_ops.buffer_load(
                        source_rsrc,
                        next_pack * fx.Int32(PACK),
                        vec_width=PACK,
                        dtype=T.bf16,
                        cache_modifier=2,
                    )
                )
                next_lds_base = fx.Int32(
                    next_buffer * block_threads * PACK
                ) + tid * fx.Int32(PACK)
                for element in range_constexpr(PACK):
                    fx.ptr_store(
                        next_values[element],
                        lds + next_lds_base + fx.Int32(element),
                    )
            gpu.barrier()

        # Keep graph replays in lockstep, just as the custom AR's final sync
        # does.  The final signal need not publish output data to other ranks.
        finished = expected + fx.Int32(1)
        if tid < fx.Int32(TP):
            slot = block * fx.Int32(TP) + rank
            remote_end = (
                peer_base(state_flat_base, tid)
                + fx.Int64(end_offset)
                + fx.Int64(slot) * fx.Int64(4)
            )
            _store_i32(remote_end, finished, system=True)
            local_slot = block * fx.Int32(TP) + tid
            _wait_i32_agent(
                local_state
                + fx.Int64(end_offset)
                + fx.Int64(local_slot) * fx.Int64(4),
                finished,
            )
        gpu.barrier()
        if tid == fx.Int32(0):
            _store_i32(epoch_addr, finished, system=False)

    @flyc.jit
    def launch(
        input_flat_base,
        state,
        state_flat_base,
        output,
        rank,
        stream,
    ):
        kernel(input_flat_base, state, state_flat_base, output, rank).launch(
            grid=(blocks, 1, 1), block=(block_threads, 1, 1), stream=stream
        )

    launch.grid_blocks = blocks
    launch.state_bytes = state_bytes(m, hidden, block_threads, block_limit)
    return launch


def _atomic_add_i32(addr, value):
    return llvm_d.AtomicRMWOp(
        llvm_d.AtomicBinOp.add,
        _global_ptr(addr),
        arith.unwrap(value),
        llvm_d.AtomicOrdering.monotonic,
        syncscope=fx.rocdl.SyncScope.Agent,
    ).res


def m1_state_layout(config: Config) -> tuple[int, int, int, int]:
    """Return done/epoch/ready offsets and total symmetric state bytes."""

    config.validate()
    n_tiles = DEFAULT_HIDDEN // config.tile_n
    done = 0
    epoch = (done + n_tiles * 4 + 127) // 128 * 128
    ready = (epoch + n_tiles * 4 + 127) // 128 * 128
    end = ready + n_tiles * TP * 4
    return done, epoch, ready, (end + 255) // 256 * 256


@functools.cache
def compile_m1_gemm2_allreduce(config: Config, specialized_rank: int):
    """Compile the small-M route-tile GEMM2 + TP8 AllReduce."""

    config.validate()
    if not 0 <= specialized_rank < TP:
        raise ValueError(f"invalid TP rank {specialized_rank}")
    if DEFAULT_HIDDEN % config.tile_n:
        raise ValueError(
            f"hidden={DEFAULT_HIDDEN} must be divisible by tile_n={config.tile_n}"
        )
    if config.sort_block_m % config.tile_m:
        raise ValueError(
            f"sort_block_m={config.sort_block_m} must be divisible by "
            f"tile_m={config.tile_m}"
        )

    n_tiles = DEFAULT_HIDDEN // config.tile_n
    packs_per_tile = config.tile_n // PACK
    total_packs_per_tile = config.m * packs_per_tile
    service_iterations = (total_packs_per_tile + M1_BLOCK - 1) // M1_BLOCK
    m_subtiles = config.sort_block_m // config.tile_m
    done_offset, epoch_offset, ready_offset, _ = m1_state_layout(config)

    def compose(*, module_name, emit_gemm2, allocator):
        # The last producer reuses the GEMM workgroup's LDS to retain this
        # rank's partial while it waits for peer readiness.  Grow the allocation
        # for M=8/16 instead of re-reading the local partial from global memory.
        local_cache_bytes = 16 + config.m * config.tile_n * 2
        allocator.ptr = max(allocator.ptr, allocator._align(local_cache_bytes, 128))

        def store_service(base, value):
            SmemPtr(base, 0, T.i32, shape=(1,)).store(value)

        def load_service(base):
            return SmemPtr(base, 0, T.i32, shape=(1,)).load()

        def store_local_partial(base, index, value):
            SmemPtr(
                base, 16, T.bf16, shape=(config.m * config.tile_n,)
            ).store(
                value, [index]
            )

        def load_local_partial(base, index):
            return SmemPtr(
                base, 16, T.bf16, shape=(config.m * config.tile_n,)
            ).load([index])

        @flyc.kernel(
            name=f"{module_name}_small_m_route_tile_local_ar_lds_v2",
            known_block_size=[M1_BLOCK, 1, 1],
        )
        def kernel(
            routes: fx.Pointer,
            x: fx.Pointer,
            w: fx.Pointer,
            scale_x: fx.Pointer,
            scale_w: fx.Pointer,
            sorted_token_ids: fx.Pointer,
            expert_ids: fx.Pointer,
            sorted_weights: fx.Pointer,
            num_valid_ids: fx.Pointer,
            shared: fx.Pointer,
            tokens: fx.Int32,
            model_dim: fx.Int32,
            inter_dim: fx.Int32,
            size_expert_ids: fx.Int32,
            partial: fx.Pointer,
            partial_flat_base: fx.Int64,
            state: fx.Pointer,
            state_flat_base: fx.Int64,
            output: fx.Pointer,
            rank: fx.Int32,
        ):
            n_tile_idx = gpu.block_id("x")
            expert_idx = gpu.block_id("y")
            logical = (
                expert_idx
                * arith.constant(m_subtiles * n_tiles, index=True)
                + n_tile_idx
            )
            emit_gemm2(
                routes,
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
                shared,
                tokens,
                model_dim,
                inter_dim,
                size_expert_ids,
                block_id=logical,
            )

            fx.rocdl.s_waitcnt(0)
            gpu.barrier()
            tid = fx.Int32(gpu.thread_idx.x)
            n_tile = fx.Int32(n_tile_idx)
            local_state = fx.Int64(ptrtoint(state))
            base = allocator.get_base()
            epoch_addr = (
                local_state
                + fx.Int64(epoch_offset)
                + fx.Int64(n_tile) * fx.Int64(4)
            )
            expected = fx.Int32(_load_i32(epoch_addr)) + fx.Int32(1)
            expert_blocks = size_expert_ids // fx.Int32(m_subtiles)
            partial_epoch_offset = (
                (expected & fx.Int32(1)) * fx.Int32(config.m * DEFAULT_HIDDEN * 2)
            )

            if tid == fx.Int32(0):
                store_service(base, fx.Int32(0))
                comm_ops.fence_agent_release()
                ticket = fx.Int32(
                    _atomic_add_i32(
                        local_state
                        + fx.Int64(done_offset)
                        + fx.Int64(n_tile) * fx.Int64(4),
                        fx.Int32(1),
                    )
                )
                is_last = arith.cmpi(
                    CmpIPredicate.eq, ticket, expert_blocks - fx.Int32(1)
                )
                store_service(
                    base,
                    arith.select(is_last, fx.Int32(1), fx.Int32(0)),
                )
            gpu.barrier()

            service = scf.IfOp(
                arith.cmpi(
                    CmpIPredicate.eq,
                    fx.Int32(load_service(base)),
                    fx.Int32(1),
                )
            )
            with ir.InsertionPoint(service.then_block):
                if tid == fx.Int32(0):
                    comm_ops.fence_agent_acquire()
                gpu.barrier()

                for service_iteration in range_constexpr(service_iterations):
                    linear_pack = tid + fx.Int32(service_iteration * M1_BLOCK)
                    active = scf.IfOp(
                        arith.cmpi(
                            CmpIPredicate.ult,
                            linear_pack,
                            fx.Int32(total_packs_per_tile),
                        )
                    )
                    with ir.InsertionPoint(active.then_block):
                        row = linear_pack // fx.Int32(packs_per_tile)
                        tile_pack = linear_pack - row * fx.Int32(packs_per_tile)
                        pack = (
                            row * fx.Int32(DEFAULT_HIDDEN // PACK)
                            + n_tile * fx.Int32(packs_per_tile)
                            + tile_pack
                        )
                        route_rsrc = buffer_ops.create_buffer_resource_from_addr(
                            fx.Int64(ptrtoint(routes)),
                            num_records_bytes=(
                                config.m * M1_TOPK * DEFAULT_HIDDEN * 2
                            ),
                        )
                        shared_rsrc = buffer_ops.create_buffer_resource_from_addr(
                            fx.Int64(ptrtoint(shared)),
                            num_records_bytes=config.m * DEFAULT_HIDDEN * 2,
                        )
                        shared_values = fx.Vector(
                            buffer_ops.buffer_load(
                                shared_rsrc,
                                pack * fx.Int32(PACK),
                                vec_width=PACK,
                                dtype=T.bf16,
                                cache_modifier=M1_LOCAL_LOAD_CACHE_MODIFIER,
                            )
                        ).extf(T.vec(PACK, T.f32))
                        route_values = []
                        for route in range_constexpr(M1_TOPK):
                            route_row = row * fx.Int32(M1_TOPK) + fx.Int32(route)
                            route_pack = (
                                route_row * fx.Int32(DEFAULT_HIDDEN // PACK)
                                + n_tile * fx.Int32(packs_per_tile)
                                + tile_pack
                            )
                            route_values.append(
                                fx.Vector(
                                    buffer_ops.buffer_load(
                                        route_rsrc,
                                        route_pack * fx.Int32(PACK),
                                        vec_width=PACK,
                                        dtype=T.bf16,
                                        cache_modifier=M1_LOCAL_LOAD_CACHE_MODIFIER,
                                    )
                                ).extf(T.vec(PACK, T.f32))
                            )

                        local01 = shared_values + route_values[0]
                        local23 = route_values[1] + route_values[2]
                        local45 = route_values[3] + route_values[4]
                        local0123 = local01 + local23
                        local456 = local45 + route_values[5]
                        partial_values = (local0123 + local456).truncf(
                            T.vec(PACK, T.bf16)
                        )
                        partial_rsrc = buffer_ops.create_buffer_resource_from_addr(
                            fx.Int64(ptrtoint(partial))
                            + fx.Int64(partial_epoch_offset),
                            num_records_bytes=config.m * DEFAULT_HIDDEN * 2,
                        )
                        buffer_ops.buffer_store(
                            partial_values,
                            partial_rsrc,
                            pack * fx.Int32(PACK),
                            cache_modifier=0,
                        )
                        for element in range_constexpr(PACK):
                            store_local_partial(
                                base,
                                linear_pack * fx.Int32(PACK)
                                + fx.Int32(element),
                                partial_values[element],
                            )
                        scf.YieldOp([])

                fx.rocdl.s_waitcnt(0)
                gpu.barrier()
                if tid == fx.Int32(0):
                    comm_ops.fence_system_release()
                gpu.barrier()

                if tid < fx.Int32(TP):
                    slot = n_tile * fx.Int32(TP) + rank
                    _store_i32(
                        peer_base(state_flat_base, tid)
                        + fx.Int64(ready_offset)
                        + fx.Int64(slot) * fx.Int64(4),
                        expected,
                        system=True,
                    )
                    local_slot = n_tile * fx.Int32(TP) + tid
                    _wait_i32_agent(
                        local_state
                        + fx.Int64(ready_offset)
                        + fx.Int64(local_slot) * fx.Int64(4),
                        expected,
                    )
                gpu.barrier()

                for service_iteration in range_constexpr(service_iterations):
                    linear_pack = tid + fx.Int32(service_iteration * M1_BLOCK)
                    reduce_active = scf.IfOp(
                        arith.cmpi(
                            CmpIPredicate.ult,
                            linear_pack,
                            fx.Int32(total_packs_per_tile),
                        )
                    )
                    with ir.InsertionPoint(reduce_active.then_block):
                        row = linear_pack // fx.Int32(packs_per_tile)
                        tile_pack = linear_pack - row * fx.Int32(packs_per_tile)
                        pack = (
                            row * fx.Int32(DEFAULT_HIDDEN // PACK)
                            + n_tile * fx.Int32(packs_per_tile)
                            + tile_pack
                        )
                        peer_values = []
                        for peer in range_constexpr(TP):
                            if const_expr(peer == specialized_rank):
                                peer_values.append(
                                    fx.Vector.from_elements(
                                        [
                                            load_local_partial(
                                                base,
                                                linear_pack * fx.Int32(PACK)
                                                + fx.Int32(element),
                                            )
                                            for element in range_constexpr(PACK)
                                        ],
                                        fx.BFloat16,
                                    ).extf(T.vec(PACK, T.f32))
                                )
                            else:
                                peer_rsrc = (
                                    buffer_ops.create_buffer_resource_from_addr(
                                        peer_base(partial_flat_base, peer)
                                        + fx.Int64(partial_epoch_offset),
                                        num_records_bytes=(
                                            config.m * DEFAULT_HIDDEN * 2
                                        ),
                                    )
                                )
                                peer_values.append(
                                    fx.Vector(
                                        buffer_ops.buffer_load(
                                            peer_rsrc,
                                            pack * fx.Int32(PACK),
                                            vec_width=PACK,
                                            dtype=T.bf16,
                                            cache_modifier=(
                                                M1_REMOTE_LOAD_CACHE_MODIFIER
                                            ),
                                        )
                                    ).extf(T.vec(PACK, T.f32))
                                )

                        peer01 = peer_values[0] + peer_values[1]
                        peer23 = peer_values[2] + peer_values[3]
                        peer45 = peer_values[4] + peer_values[5]
                        peer67 = peer_values[6] + peer_values[7]
                        peer0123 = peer01 + peer23
                        peer4567 = peer45 + peer67
                        reduced = peer0123 + peer4567
                        output_rsrc = (
                            buffer_ops.create_buffer_resource_from_addr(
                                fx.Int64(ptrtoint(output)),
                                num_records_bytes=config.m * DEFAULT_HIDDEN * 2,
                            )
                        )
                        buffer_ops.buffer_store(
                            reduced.truncf(T.vec(PACK, T.bf16)),
                            output_rsrc,
                            pack * fx.Int32(PACK),
                        )
                        scf.YieldOp([])

                fx.rocdl.s_waitcnt(0)
                gpu.barrier()
                if tid == fx.Int32(0):
                    _store_i32(
                        local_state
                        + fx.Int64(done_offset)
                        + fx.Int64(n_tile) * fx.Int64(4),
                        fx.Int32(0),
                        system=False,
                    )
                    _store_i32(epoch_addr, expected, system=False)
                scf.YieldOp([])

        @flyc.jit
        def launch(
            routes,
            x,
            w,
            scale_x,
            scale_w,
            sorted_token_ids,
            expert_ids,
            sorted_weights,
            num_valid_ids,
            shared,
            tokens,
            model_dim,
            inter_dim,
            size_expert_ids,
            partial,
            partial_flat_base,
            state,
            state_flat_base,
            output,
            rank,
            stream,
        ):
            allocator.finalized = False
            context = CompilationContext.get_current()
            with ir.InsertionPoint(context.gpu_module_body):
                allocator.finalize()
            kernel(
                routes,
                x,
                w,
                scale_x,
                scale_w,
                sorted_token_ids,
                expert_ids,
                sorted_weights,
                num_valid_ids,
                shared,
                tokens,
                model_dim,
                inter_dim,
                size_expert_ids,
                partial,
                partial_flat_base,
                state,
                state_flat_base,
                output,
                rank,
            ).launch(
                grid=(n_tiles, size_expert_ids // m_subtiles, 1),
                block=(M1_BLOCK, 1, 1),
                stream=stream,
            )

        return launch

    return compile_mixed_moe_gemm2_common(
        model_dim=DEFAULT_HIDDEN,
        inter_dim=M1_INTER_DIM,
        experts=M1_EXPERTS,
        topk=M1_TOPK,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        doweight_stage2=True,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        accumulate=False,
        # M<=16 occupies one tile_m=16 subtile.  persist_m is the number of
        # M subtiles emitted per expert block, not the number of logical rows.
        persist_m=1,
        sort_block_m=config.sort_block_m,
        _compose_entry=compose,
        _direct_small_m_rows=config.m,
    )

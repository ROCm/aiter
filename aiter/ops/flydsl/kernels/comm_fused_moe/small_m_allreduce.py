# SPDX-License-Identifier: MIT
"""Small-message TP8 one-stage BF16 AllReduce.

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
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.dialects import scf
from flydsl.expr import arith, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T

from .. import buffer_ops
from .sync import peer_base


TP = 8
BLOCK = 512
WAVE = 64
PACK = 8  # Eight BF16 values = one 16-byte transaction.
DEFAULT_HIDDEN = 7168


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

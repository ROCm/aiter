# SPDX-License-Identifier: Apache-2.0
"""Persistent TP reduce-scatter/all-gather service for windowed Stage2."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_d
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects.arith import CmpIPredicate
from flydsl.expr import arith, gpu, ptrtoint
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from .. import communication_ops_utils as comm_ops
from . import windowed as k
from .collectives import emit_tp_all_gather, emit_tp_reduce_scatter
from .sync import peer_base


PARTIAL_STRIDE = k.M * (k.WINDOW + k.WINDOW // 32)
REDUCED_PAYLOAD_STRIDE = k.SHARD_ROWS * k.WINDOW
REDUCED_SCALE_STRIDE = k.SHARD_ROWS * (k.WINDOW // 32)


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
def compile_stage2_service():
    @fx.struct
    class SharedStorage:
        epoch: fx.Array[fx.Int64, 1, 16]

    @flyc.kernel(
        name="comm_fused_moe_persistent_service",
        known_block_size=[k.BLOCK, 1, 1],
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
                + fx.Int64(k.WORKER_EPOCH)
                + fx.Int64(worker) * fx.Int64(8)
            )
            expected = fx.Int64(comm_ops.load_i64_global(worker_epoch)) + fx.Int64(1)
            if worker == fx.Int32(0):
                _store_agent(local_state + fx.Int64(k.SERVICE_EPOCH), expected)
            epoch = fx.Int64(
                _wait_agent(local_state + fx.Int64(k.SERVICE_EPOCH), expected)
            )
            _store_agent(worker_epoch, epoch)
            fx.ptr_store(Vec.from_elements([epoch], fx.Int64), epoch_scratch)
        gpu.barrier()
        epoch = Vec(epoch_view.load())[0]

        for phase in range(k.PHASES):
            partial_ready = k.PARTIAL_READY + phase * 8
            reduced_ready = k.REDUCED_READY + phase * 8
            gate = local_state + fx.Int64(k.PHASE_GATE + phase * 8)
            partial_gate = epoch * fx.Int64(2) - fx.Int64(1)
            reduced_gate = epoch * fx.Int64(2)

            if worker == fx.Int32(0):
                if tid < fx.Int32(k.TP):
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
                partial_flat_base + fx.Int64(phase * PARTIAL_STRIDE),
                _byte_ptr(
                    fx.Int64(ptrtoint(output))
                    + fx.Int64(k.SHARD_ROWS * k.H * 2) * fx.Int64(rank)
                    + fx.Int64(phase * k.WINDOW * 2)
                ),
                _byte_ptr(
                    fx.Int64(ptrtoint(reduced_payloads))
                    + fx.Int64(phase * REDUCED_PAYLOAD_STRIDE)
                ),
                _byte_ptr(
                    fx.Int64(ptrtoint(reduced_scales))
                    + fx.Int64(phase * REDUCED_SCALE_STRIDE)
                ),
                rank,
                worker,
                tokens=k.M,
                output_width=k.H,
                payload_width=k.WINDOW,
                shard_rows=k.SHARD_ROWS,
                tp=k.TP,
                block=k.BLOCK,
                reduce_scatter_grid=k.SERVICE_GRID,
            )

            gpu.barrier()
            if tid == fx.Int32(0):
                comm_ops.fence_agent_release()
                done = fx.Int64(
                    comm_ops.atomic_add_agent(
                        local_state + fx.Int64(k.PHASE_DONE + phase * 8),
                        fx.Int64(1),
                    )
                )
                if done == epoch * fx.Int64(k.SERVICE_GRID) - fx.Int64(1):
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
                if tid < fx.Int32(k.TP):
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
                + fx.Int64(phase * REDUCED_PAYLOAD_STRIDE),
                reduced_scale_flat_base + fx.Int64(phase * REDUCED_SCALE_STRIDE),
                _byte_ptr(
                    fx.Int64(ptrtoint(output)) + fx.Int64(phase * k.WINDOW * 2)
                ),
                rank,
                worker,
                output_width=k.H,
                payload_width=k.WINDOW,
                shard_rows=k.SHARD_ROWS,
                tp=k.TP,
                block=k.BLOCK,
                all_gather_grid=k.SERVICE_GRID,
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
            grid=(k.SERVICE_GRID, 1, 1),
            block=(k.BLOCK, 1, 1),
            stream=stream,
        )

    return launch

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""ReduceScatter for the fused TP MoE layer."""

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr

from .. import communication_ops_utils as comm
from .p2p import (
    DESC_ARRIVE,
    DESC_FLAGS,
    DESC_PEER_BASE,
    desc_peer_base,
    desc_slot,
    flat_buffer,
    load_words,
    store_words,
)

__all__ = [
    "ARRIVE_STRIDE_BYTES",
    "PHASE_CTR_SLOT",
    "PHASE_GATE_SLOT",
    "WRITEBACK_BLOCKS",
    "RS_DESC_FANIN",
    "MAX_SERVICE_BLOCKS",
    "RS_ARRIVE_SLOTS",
    "RS_DESC_EPOCH",
    "RS_DESC_OUTPUT",
    "RS_DESC_PARTIAL",
    "RS_PULL_UNROLL",
    "RS_UNIT_ELEMS",
    "compile_reduce_scatter_publish",
    "compile_reduce_scatter_pull",
    "rs_desc_size",
    "rs_desc_slot",
]

_PULL_BLOCK = int(os.environ.get("AITER_TP_RS_BLOCK", "1024"))
_PUBLISH_BLOCKS = 64
_PUBLISH_BLOCK = 64
RS_UNIT_ELEMS = 8
_VEC_WORDS = RS_UNIT_ELEMS // 2
RS_PULL_UNROLL = 1

RS_DESC_PARTIAL = 0
RS_DESC_OUTPUT = 1
RS_DESC_EPOCH = 2
RS_DESC_FANIN = 3
_RS_EXTRA = 4

MAX_SERVICE_BLOCKS = 256
ARRIVE_STRIDE_DW = 32
ARRIVE_STRIDE_BYTES = ARRIVE_STRIDE_DW * 4
RS_ARRIVE_SLOTS = (MAX_SERVICE_BLOCKS + 6) * ARRIVE_STRIDE_DW
PHASE_CTR_SLOT = MAX_SERVICE_BLOCKS + 4
PHASE_GATE_SLOT = MAX_SERVICE_BLOCKS + 5
WRITEBACK_BLOCKS = 8


def rs_desc_size(tp_size: int) -> int:
    return DESC_PEER_BASE + tp_size + _RS_EXTRA


def rs_desc_slot(tp_size: int, which: int) -> int:
    return DESC_PEER_BASE + tp_size + which


def _validate(tp_size: int, model_dim: int) -> int:
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")
    if model_dim <= 0 or model_dim % RS_UNIT_ELEMS:
        raise ValueError(
            f"model_dim must be a positive multiple of {RS_UNIT_ELEMS}, "
            f"got {model_dim}"
        )
    return model_dim // RS_UNIT_ELEMS


@functools.cache
def compile_reduce_scatter_publish(tp_size: int, *, block: int = _PUBLISH_BLOCK):
    """Announce that this rank's GEMM2 partial is visible to its peers."""
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")
    name = f"mega_moe_tp_rs_publish_tp{tp_size}"

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def rs_publish_kernel(
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        if tid == fx.Int32(0):
            comm.fence_system_release()
            arrive_addr = desc_slot(arg_desc, DESC_ARRIVE)
            previous = fx.Int32(comm.atomic_add_agent(arrive_addr, fx.Int32(1)))
            if previous == i32_grid - fx.Int32(1):
                comm.store_i32_global_agent_release(arrive_addr, fx.Int32(0))
                epoch_addr = desc_slot(arg_desc, rs_desc_slot(tp_size, RS_DESC_EPOCH))
                epoch = fx.Int32(comm.load_i32_global_agent(epoch_addr)) + fx.Int32(1)
                comm.store_i32_global_agent_release(epoch_addr, epoch)
                flags_offset = desc_slot(arg_desc, DESC_FLAGS)
                for slot in range_constexpr(tp_size):
                    peer = (i32_rank + fx.Int32(slot)) % fx.Int32(tp_size)
                    comm.store_i32_global_system_release(
                        desc_peer_base(arg_desc, peer)
                        + flags_offset
                        + fx.Int64(i32_rank) * fx.Int64(4),
                        epoch,
                    )

    launcher = _wrap_launch(rs_publish_kernel, block, 0, name)
    launcher.blocks = _PUBLISH_BLOCKS
    return launcher


@functools.cache
def compile_reduce_scatter_pull(
    tp_size: int,
    model_dim: int,
    *,
    block: int = _PULL_BLOCK,
):
    """Sum this rank's row range across every peer's partial."""
    units_per_row = _validate(tp_size, model_dim)
    row_bytes = model_dim * 2
    name = f"mega_moe_tp_rs_pull_tp{tp_size}_h{model_dim}_b{block}"

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def rs_pull_kernel(
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        gid = bid * fx.Int32(block) + tid
        stride = i32_grid * fx.Int32(block)
        units = i32_rows * fx.Int32(units_per_row)

        local_base = desc_peer_base(arg_desc, i32_rank)
        if tid == fx.Int32(0):
            epoch_addr = desc_slot(arg_desc, rs_desc_slot(tp_size, RS_DESC_EPOCH))
            epoch = fx.Int32(comm.load_i32_global_agent(epoch_addr))
            flags_offset = desc_slot(arg_desc, DESC_FLAGS)
            for source in range_constexpr(tp_size):
                comm.spin_until_ge_i32_system(
                    local_base + flags_offset + fx.Int64(source * 4),
                    epoch,
                    acquire=True,
                    sleep=False,
                )
            comm.fence_system_acquire()
        gpu.barrier()

        partial_offset = desc_slot(arg_desc, rs_desc_slot(tp_size, RS_DESC_PARTIAL))
        total_bytes = fx.Int64(i32_rows) * fx.Int64(tp_size) * fx.Int64(row_bytes)
        sources = []
        for slot in range_constexpr(tp_size):
            peer = (i32_rank + bid + fx.Int32(slot)) % fx.Int32(tp_size)
            sources.append(
                flat_buffer(
                    desc_peer_base(arg_desc, peer) + partial_offset,
                    fx.Int32,
                    total_bytes,
                )
            )
        output = flat_buffer(
            desc_slot(arg_desc, rs_desc_slot(tp_size, RS_DESC_OUTPUT)),
            fx.Int32,
            fx.Int64(i32_rows) * fx.Int64(row_bytes),
        )
        shard_base = i32_rank * units
        for unit in range(gid, units, stride * fx.Int32(RS_PULL_UNROLL)):
            for step in range_constexpr(RS_PULL_UNROLL):
                index = fx.Int32(unit) + fx.Int32(step) * stride
                acc = fx.Vector.filled(RS_UNIT_ELEMS, 0.0, fx.Float32)
                for slot in range_constexpr(tp_size):
                    words = load_words(
                        sources[slot],
                        shard_base + index,
                        width=_VEC_WORDS,
                        cache_modifier=2,
                    )
                    acc = acc + words.bitcast(fx.BFloat16).to(fx.Float32)
                store_words(
                    output,
                    index,
                    acc.to(fx.BFloat16).bitcast(fx.Int32),
                    width=_VEC_WORDS,
                )

    return _wrap_launch(rs_pull_kernel, block, units_per_row, name)


def _wrap_launch(kernel, block, units_per_row, name):
    @flyc.jit
    def launch(
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_desc, i32_rank, i32_rows, i32_grid).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(block, 1, 1), stream=stream
        )

    launch.block = block
    launch.units_per_row = units_per_row
    launch.__name__ = f"launch_{name}"
    return launch

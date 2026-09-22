# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AllGather that quantizes on the wire: MXFP4 quant fused into the push."""

from __future__ import annotations

import functools
import os


import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .. import communication_ops_utils as comm
from .allgather_push import AG_DESC_DONE, AG_DESC_EPOCH, PUSH_VEC_BYTES
from .p2p import (
    DESC_ARRIVE,
    DESC_FLAGS,
    desc_peer_base,
    desc_region_offset,
    desc_region_src,
    desc_size,
    desc_slot,
    flat_buffer,
    load_words,
    store_words,
)

__all__ = [
    "GROUPS_PER_THREAD",
    "QUANT_ELEMS_PER_THREAD",
    "compile_allgather_quant_push",
    "quant_push_supported",
    "quant_push_units",
]

_BLOCK = int(os.environ.get("AITER_TP_AGQ_BLOCK", "256"))
_FP4_INV_MAX_POS_BITS = 0x3E2AAAAB
_MX_GROUP = 32
GROUPS_PER_THREAD = 4
QUANT_ELEMS_PER_THREAD = _MX_GROUP * GROUPS_PER_THREAD
_R_PAYLOAD, _R_SCALE, _R_IDS, _R_WEIGHTS = 0, 1, 2, 3
_REGIONS = 4


def quant_push_supported(model_dim: int, topk: int) -> bool:
    """Whether this shape's rows split into whole per-thread quant units."""
    return model_dim % QUANT_ELEMS_PER_THREAD == 0 and (topk * 4) % 4 == 0


def quant_push_units(rows: int, model_dim: int, topk: int, regions: str = "all") -> int:
    """Grid-stride work items one fused push covers, for the selected regions."""
    route_row_bytes = topk * 4
    route_unit = PUSH_VEC_BYTES if route_row_bytes % PUSH_VEC_BYTES == 0 else 4
    quant = rows * (model_dim // QUANT_ELEMS_PER_THREAD)
    route = 2 * rows * (route_row_bytes // route_unit)
    if regions == "payload":
        return quant
    if regions == "route":
        return route
    return quant + route


@functools.cache
def compile_allgather_quant_push(
    tp_size: int,
    model_dim: int,
    topk: int,
    *,
    block: int = _BLOCK,
    regions: str = "all",
    _composition=None,
):
    """Build the quantize-and-push AllGather launcher for one shape."""
    if regions not in ("all", "route", "payload"):
        raise ValueError(f"regions must be all/route/payload, got {regions!r}")
    if not 1 <= tp_size <= 8:
        raise ValueError(f"tp_size must be in [1, 8], got {tp_size}")
    if not quant_push_supported(model_dim, topk):
        raise ValueError(
            f"model_dim={model_dim} must be a multiple of "
            f"{QUANT_ELEMS_PER_THREAD} to fuse quant into the push"
        )

    chunks_per_row = model_dim // QUANT_ELEMS_PER_THREAD
    payload_row_bytes = model_dim // 2
    scale_row_bytes = model_dim // 32
    route_row_bytes = topk * 4
    route_vec_words = 4 if route_row_bytes % PUSH_VEC_BYTES == 0 else 1
    route_units_per_row = route_row_bytes // (route_vec_words * 4)
    epoch_index = desc_size(tp_size, _REGIONS) + AG_DESC_EPOCH
    done_index = desc_size(tp_size, _REGIONS) + AG_DESC_DONE
    suffix = "" if regions == "all" else f"_{regions}"
    name = (
        f"mega_moe_tp_agq_push_tp{tp_size}_h{model_dim}_k{topk}_b{block}{suffix}"
    )

    @flyc.jit
    def emit_quant_payload_push(arg_desc, i32_rank, i32_rows, bid, gid, stride):
        """Quantize this rank's rows to MXFP4 and push them to every peer."""

        src_bytes = fx.Int64(i32_rows) * fx.Int64(model_dim * 2)
        src = flat_buffer(
            desc_slot(arg_desc, desc_region_src(tp_size, _R_PAYLOAD)),
            fx.Int32,
            src_bytes,
        )
        payload_bytes = fx.Int64(i32_rows) * fx.Int64(payload_row_bytes)
        scale_bytes = fx.Int64(i32_rows) * fx.Int64(scale_row_bytes)
        payload_offset = desc_slot(
            arg_desc, desc_region_offset(tp_size, _R_PAYLOAD)
        )
        scale_offset = desc_slot(arg_desc, desc_region_offset(tp_size, _R_SCALE))

        payload_dst = []
        scale_dst = []
        for slot in range_constexpr(tp_size):
            peer = (i32_rank + bid + fx.Int32(slot)) % fx.Int32(tp_size)
            base = desc_peer_base(arg_desc, peer)
            payload_dst.append(
                flat_buffer(
                    base + payload_offset + fx.Int64(i32_rank) * payload_bytes,
                    fx.Int32,
                    payload_bytes,
                )
            )
            scale_dst.append(
                flat_buffer(
                    base + scale_offset + fx.Int64(i32_rank) * scale_bytes,
                    fx.Int32,
                    scale_bytes,
                )
            )

        quant_units = i32_rows * fx.Int32(chunks_per_row)
        for unit in range(gid, quant_units, stride):
            u = fx.Int32(unit)
            row = u // fx.Int32(chunks_per_row)
            chunk = u - row * fx.Int32(chunks_per_row)
            src_unit = (
                row * fx.Int32(model_dim * 2 // PUSH_VEC_BYTES)
                + chunk * fx.Int32(QUANT_ELEMS_PER_THREAD * 2 // PUSH_VEC_BYTES)
            )
            words = []
            scale_byte = []
            for g in range_constexpr(GROUPS_PER_THREAD):
                act = []
                local_max = fx.Float32(1e-10)
                for piece in range_constexpr(_MX_GROUP * 2 // PUSH_VEC_BYTES):
                    raw = load_words(
                        src,
                        src_unit
                        + fx.Int32(g * (_MX_GROUP * 2 // PUSH_VEC_BYTES) + piece),
                        width=4,
                        cache_modifier=1,
                    )
                    values = raw.bitcast(fx.BFloat16).to(fx.Float32)
                    local_max = local_max.maximumf(
                        fmath.absf(values).reduce(ReductionOp.MAX)
                    )
                    for elem in range_constexpr(8):
                        act.append(values[elem])

                working = (
                    local_max * fx.Int32(_FP4_INV_MAX_POS_BITS).bitcast(fx.Float32)
                ).bitcast(fx.Int32)
                mantissa = working & fx.Int32(0x7FFFFF)
                biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
                e8m0 = (mantissa != fx.Int32(0)).select(
                    biased_exp + fx.Int32(1), biased_exp
                )
                e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
                scale_byte.append(e8m0)

                dequant_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
                for word in range_constexpr(_MX_GROUP // 8):
                    packed = fx.Int32(0)
                    for pair in range_constexpr(4):
                        idx = word * 8 + pair * 2
                        packed = rocdl.cvt_scalef32_pk_fp4_f32(
                            T.i32,
                            packed,
                            act[idx],
                            act[idx + 1],
                            dequant_scale,
                            pair,
                        )
                    words.append(packed)

            scale_word = scale_byte[0]
            for g in range_constexpr(GROUPS_PER_THREAD - 1):
                scale_word = scale_word | (scale_byte[g + 1] << fx.Int32(8 * (g + 1)))

            payload_unit = row * fx.Int32(payload_row_bytes // PUSH_VEC_BYTES) + (
                chunk * fx.Int32(GROUPS_PER_THREAD)
            )
            scale_unit = row * fx.Int32(scale_row_bytes // 4) + chunk
            for slot in range_constexpr(tp_size):
                for quad in range_constexpr(GROUPS_PER_THREAD):
                    store_words(
                        payload_dst[slot],
                        payload_unit + fx.Int32(quad),
                        fx.Vector.from_elements(
                            words[quad * 4 : quad * 4 + 4], fx.Int32
                        ),
                        width=4,
                    )
                store_words(
                    scale_dst[slot],
                    scale_unit,
                    fx.Vector.from_elements([scale_word], fx.Int32),
                    width=1,
                )


    @flyc.jit
    def emit_route_push(arg_desc, i32_rank, i32_rows, bid, gid, stride):
        """Push topk ids and weights to every peer (regions 2 and 3)."""
        route_bytes = fx.Int64(i32_rows) * fx.Int64(route_row_bytes)
        route_units = i32_rows * fx.Int32(route_units_per_row)
        for region in range_constexpr(2):
            which = _R_IDS + region
            route_src = flat_buffer(
                desc_slot(arg_desc, desc_region_src(tp_size, which)),
                fx.Int32,
                route_bytes,
            )
            region_offset = desc_slot(arg_desc, desc_region_offset(tp_size, which))
            for slot in range_constexpr(tp_size):
                peer = (i32_rank + bid + fx.Int32(slot)) % fx.Int32(tp_size)
                dst = flat_buffer(
                    desc_peer_base(arg_desc, peer)
                    + region_offset
                    + fx.Int64(i32_rank) * route_bytes,
                    fx.Int32,
                    route_bytes,
                )
                for unit in range(gid, route_units, stride):
                    store_words(
                        dst,
                        fx.Int32(unit),
                        load_words(
                            route_src,
                            fx.Int32(unit),
                            width=route_vec_words,
                            cache_modifier=1,
                        ),
                        width=route_vec_words,
                    )


    @flyc.jit
    def emit_ag_gate(arg_desc, entry_epoch, tid):
        """Wait until this rank's AllGather has completed, and nothing else."""
        if tid == fx.Int32(0):
            comm.spin_until_ge_i32_agent(
                desc_slot(arg_desc, done_index), entry_epoch
            )
            comm.fence_agent_acquire()
        gpu.barrier()

    @flyc.jit
    def emit_ag_barrier(arg_desc, i32_rank, i32_grid, tid, entry_epoch=None):
        """Publish this rank's arrival and wait until every peer has landed."""
        comm.fence_agent_release()
        gpu.barrier()
        if tid == fx.Int32(0):
            comm.fence_system_release()
            arrive_addr = desc_slot(arg_desc, DESC_ARRIVE)
            previous = fx.Int32(comm.atomic_add_agent(arrive_addr, fx.Int32(1)))
            if previous == i32_grid - fx.Int32(1):
                comm.store_i32_global_agent_release(arrive_addr, fx.Int32(0))
                epoch_addr = desc_slot(arg_desc, epoch_index)
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
                local_base = desc_peer_base(arg_desc, i32_rank)
                for source in range_constexpr(tp_size):
                    comm.spin_until_ge_i32_system(
                        local_base + flags_offset + fx.Int64(source * 4),
                        epoch,
                        acquire=True,
                    )
                comm.fence_system_acquire()
                if entry_epoch is not None:
                    comm.store_i32_global_agent_release(
                        desc_slot(arg_desc, done_index), entry_epoch
                    )
        if entry_epoch is not None:
            if tid == fx.Int32(0):
                comm.spin_until_ge_i32_agent(
                    desc_slot(arg_desc, done_index), entry_epoch
                )
                comm.fence_agent_acquire()
            gpu.barrier()


    if _composition is not None:
        return _composition(
            module_name=name,
            emit_quant_payload_push=emit_quant_payload_push,
            emit_route_push=emit_route_push,
            emit_ag_barrier=emit_ag_barrier,
            emit_ag_gate=emit_ag_gate,
            block=block,
        )

    @flyc.kernel(name=name, known_block_size=[block, 1, 1])
    def ag_quant_push_kernel(
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        gid = bid * fx.Int32(block) + tid
        stride = i32_grid * fx.Int32(block)
        if regions in ("all", "payload"):
            emit_quant_payload_push(arg_desc, i32_rank, i32_rows, bid, gid, stride)
        if regions in ("all", "route"):
            emit_route_push(arg_desc, i32_rank, i32_rows, bid, gid, stride)
        emit_ag_barrier(arg_desc, i32_rank, i32_grid, tid)
    @flyc.jit
    def launch(
        arg_desc: fx.Int64,
        i32_rank: fx.Int32,
        i32_rows: fx.Int32,
        i32_grid: fx.Int32,
        stream: fx.Stream,
    ):
        ag_quant_push_kernel(arg_desc, i32_rank, i32_rows, i32_grid).launch(
            grid=(fx.Int64(i32_grid), 1, 1), block=(block, 1, 1), stream=stream
        )

    launch.block = block
    return launch

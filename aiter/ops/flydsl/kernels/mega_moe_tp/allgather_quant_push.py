# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""AllGather that quantizes on the wire: MXFP4 quant fused into the push.

On the MXFP4 wire the unfused chain is two launches over three passes of the
activation::

    per_1x32_mx_quant   x_local[m, H] bf16  ->  xq[m, H/2] + scale[m, H/32]
    ag_push             xq, scale, route    ->  every peer's arena

The quantization is pure per-row-group arithmetic with no cross-thread
dependency, so there is no reason for it to be its own kernel and its own
round trip through HBM.  This module reads ``x_local`` once, quantizes in
registers, and stores the packed result straight into all ``TP`` peers --
removing a launch, the staging buffer, and a full read-modify-write of it.

Work split
----------
One thread owns ``GROUPS_PER_THREAD`` consecutive 32-element MX groups, i.e.
128 activations: 256 B in, 64 B of packed FP4 out, and 4 B of E8M0 out.  Four
groups rather than one is what makes the scale store a whole dword, so the
narrow side of the payload never drops to byte stores.  ``model_dim`` must
therefore be a multiple of 128; :func:`quant_push_supported` reports that, and
the caller keeps the split quant + push for shapes that fail it.

The routing regions are plain copies, unchanged from
:mod:`.allgather_push` -- they carry no arithmetic to fuse.
"""

from __future__ import annotations

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .. import communication_ops_utils as comm
from .allgather_push import AG_DESC_EPOCH, PUSH_VEC_BYTES
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
#: fp32 bits of 1/max_pos for the RoundUp ceil_pow2(amax/max_pos) scale; fp4's
#: max_pos is 6. Same constant the standalone quant kernel uses, so the two
#: produce bit-identical output.
_FP4_INV_MAX_POS_BITS = 0x3E2AAAAB
_MX_GROUP = 32
#: Four groups per thread makes the E8M0 side exactly one dword.
GROUPS_PER_THREAD = 4
QUANT_ELEMS_PER_THREAD = _MX_GROUP * GROUPS_PER_THREAD  # 128
#: Region indices in the descriptor, matching the fp4-wire region order.
_R_PAYLOAD, _R_SCALE, _R_IDS, _R_WEIGHTS = 0, 1, 2, 3
_REGIONS = 4


def quant_push_supported(model_dim: int, topk: int) -> bool:
    """Whether this shape's rows split into whole per-thread quant units."""
    return model_dim % QUANT_ELEMS_PER_THREAD == 0 and (topk * 4) % 4 == 0


def quant_push_units(rows: int, model_dim: int, topk: int) -> int:
    """Grid-stride work items one fused push covers, across every region.

    Must mirror the kernel's own split exactly: a route row that is not a whole
    number of :data:`..allgather_push.PUSH_VEC_BYTES` (topk 6 or 9, say) drops
    to dword copies and therefore has *more* units per row, not fewer.
    """
    route_row_bytes = topk * 4
    route_unit = PUSH_VEC_BYTES if route_row_bytes % PUSH_VEC_BYTES == 0 else 4
    quant = rows * (model_dim // QUANT_ELEMS_PER_THREAD)
    route = 2 * rows * (route_row_bytes // route_unit)
    return quant + route


@functools.cache
def compile_allgather_quant_push(
    tp_size: int,
    model_dim: int,
    topk: int,
    *,
    block: int = _BLOCK,
):
    """Build the quantize-and-push AllGather launcher for one shape.

    The descriptor is the ordinary four-region fp4-wire descriptor, except that
    region 0's source address is the **BF16** ``x_local`` rather than a
    pre-quantized buffer, and region 1 has no source at all -- the E8M0 scales
    are produced here.
    """
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
    name = (
        f"mega_moe_tp_agq_push_tp{tp_size}_h{model_dim}_k{topk}_b{block}"
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

        # Peer destinations, resolved once. Stagger by rank *and* block so the
        # grid spreads over all xGMI links instead of pounding one.
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

        # -- quantize and push the activation ------------------------------
        quant_units = i32_rows * fx.Int32(chunks_per_row)
        for unit in range(gid, quant_units, stride):
            u = fx.Int32(unit)
            row = u // fx.Int32(chunks_per_row)
            chunk = u - row * fx.Int32(chunks_per_row)
            # 128 bf16 = 256 B = 16 dwordx4 loads, starting at this chunk.
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

            # Four E8M0 bytes, little-endian, as one dword.
            scale_word = scale_byte[0]
            for g in range_constexpr(GROUPS_PER_THREAD - 1):
                scale_word = scale_word | (scale_byte[g + 1] << fx.Int32(8 * (g + 1)))

            # 64 B of payload as four dwordx4, 4 B of scale as one dword.
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

        # -- push the route ------------------------------------------------
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

        # -- barrier, identical to the copy-only push -----------------------
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

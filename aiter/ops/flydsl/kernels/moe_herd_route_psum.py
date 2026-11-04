# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""HERD finalize fused with the grouped-MoE route and tile-aligned psum (FlyDSL).

One workgroup does what the HERD finalize, the route-counter fill, the route
kernel and the psum remap do in four launches. Phase C is
``herd_topk.build_herd_finalize_module``'s per-row code, so the routes and
weights match it bit for bit. Rows at or past ``num_valid_rows`` get ids -1,
weights 0 and ``DROPPED_ROUTE_ROW``, and count toward neither popularity nor
any expert. The scan takes one lane per expert and phase C one lane per row,
so both are bounded by the block.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import Float32, Int32, const_expr, gpu, range_constexpr

from aiter.ops.flydsl.kernels.kernels_common import kernel_signature
from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_buf_tensor,
)

HERD_ROUTE_BLOCK = 1024


@cache
def build_herd_route_psum_module(topk: int, num_experts: int, has_num_valid_rows: bool):
    """JIT launcher: HERD finalize + route + tile-aligned psum + row remap."""
    if not 0 < topk < num_experts <= HERD_ROUTE_BLOCK:
        raise ValueError(f"unsupported HERD route: topk={topk}, experts={num_experts}")
    kp1 = topk + 1
    block = HERD_ROUTE_BLOCK

    @fx.struct
    class SharedStorage:
        popularity: fx.Array[Int32, num_experts, 16]
        counts: fx.Array[Int32, num_experts, 16]
        lds0: fx.Array[Int32, block, 16]
        lds1: fx.Array[Int32, block, 16]

    @flyc.kernel(
        name="moe_herd_route_psum_"
        + kernel_signature(k=topk, e=num_experts, nv=has_num_valid_rows),
        known_block_size=[block, 1, 1],
    )
    def herd_route_psum_kernel(
        candidate_ids: fx.Pointer,  # (rows, K+1) i32
        candidate_values: fx.Pointer,  # (rows, K+1) f32
        topk_weights: fx.Pointer,  # (rows, K) f32 out
        topk_ids: fx.Pointer,  # (rows, K) i32 out
        topids_to_rows: fx.Pointer,  # (rows, K) i32 out: contiguous grouped row
        masked_m: fx.Pointer,  # (E,) i32 out
        psum: fx.Pointer,  # (E,) i32 out
        num_valid_rows: fx.Pointer,  # (1,) i32
        rows: Int32,
        tile_m: Int32,
        routed_scaling_factor: Float32,
    ):
        tid = fx.thread_idx.x
        utid = fx.Uint32(fx.thread_idx.x)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        popularity = lds.popularity.ptr
        counts = lds.counts.ptr
        ids_in = ptr_buf_tensor(candidate_ids)
        values_in = ptr_buf_tensor(candidate_values, elem=Float32)
        weights_out = ptr_buf_tensor(topk_weights, elem=Float32)
        ids_out = ptr_buf_tensor(topk_ids)
        rows_out = ptr_buf_tensor(topids_to_rows)

        valid_rows = rows
        if const_expr(has_num_valid_rows):
            valid_rows = fx.min(rows, ptr_buf_tensor(num_valid_rows)[0])

        # A: zero the popularity and route counters.
        in_expert = utid < fx.Uint32(num_experts)
        if in_expert:
            popularity[utid] = Int32(0)
            counts[utid] = Int32(0)
        gpu.barrier()

        # B: popularity over the valid rows' candidates.
        for idx in range(tid, valid_rows * Int32(kp1), Int32(block)):
            fx.atomic_add(popularity + ids_in[idx], Int32(1), syncscope="workgroup")
        gpu.barrier()

        # C: per row, drop one candidate, write the kept K in ascending expert
        # order, and take each kept route's slot in its expert.
        row = tid
        out_base = row * Int32(topk)
        if const_expr(has_num_valid_rows):
            if (row >= valid_rows) & (row < rows):
                for j in range_constexpr(topk):
                    ids_out[out_base + j] = Int32(-1)
                    weights_out[out_base + j] = Float32(0.0)
                    rows_out[out_base + j] = Int32(DROPPED_ROUTE_ROW)
        if row < valid_rows:
            in_base = row * Int32(kp1)
            ids = [ids_in[in_base + j] for j in range_constexpr(kp1)]
            values = [values_in[in_base + j] for j in range_constexpr(kp1)]
            pops = [popularity[ids[j]] for j in range_constexpr(kp1)]

            drop = Int32(0)
            drop_id = ids[0]
            drop_value = values[0]
            drop_pop = pops[0]
            for j in range_constexpr(1, kp1):
                take = (pops[j] < drop_pop) | (
                    (pops[j] == drop_pop)
                    & (
                        (values[j] < drop_value)
                        | ((values[j] == drop_value) & (ids[j] < drop_id))
                    )
                )
                drop = take.select(Int32(j), drop)
                drop_id = take.select(ids[j], drop_id)
                drop_value = take.select(values[j], drop_value)
                drop_pop = take.select(pops[j], drop_pop)

            kept_sum = Float32(0.0)
            for j in range_constexpr(kp1):
                kept_sum = kept_sum + (drop != Int32(j)).select(values[j], Float32(0.0))
            scale = routed_scaling_factor / fx.max(kept_sum, Float32(1e-20))

            for j in range_constexpr(kp1):
                rank = Int32(0)
                for other in range_constexpr(kp1):
                    before = (drop != Int32(other)) & (ids[other] < ids[j])
                    rank = rank + before.select(Int32(1), Int32(0))
                if drop != Int32(j):
                    slot = fx.atomic_add(counts + ids[j], Int32(1), syncscope="workgroup")
                    ids_out[out_base + rank] = ids[j]
                    weights_out[out_base + rank] = values[j] * scale
                    rows_out[out_base + rank] = slot
        gpu.barrier()

        # D: tile-aligned exclusive scan of the counts.
        tile_v = fx.Uint32(tile_m)
        if in_expert:
            m = fx.Uint32(counts[utid])
            lds.lds0.ptr[utid] = fx.Int32((m + tile_v - 1) // tile_v * tile_v)
            ptr_buf_tensor(masked_m)[utid] = fx.Int32(m)
        gpu.barrier()
        src = lds.lds0.ptr
        dst = lds.lds1.ptr
        for offset in range_constexpr(1, block):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            if in_expert:
                prev = fx.Int32(0)
                if utid >= offset:
                    prev = src[utid - offset]
                dst[utid] = src[utid] + prev
            gpu.barrier()
            src, dst = dst, src
        starts = dst  # the buffer the last step read from is free
        if in_expert:
            start = fx.Int32(0)
            if utid != 0:
                start = src[utid - 1]
            starts[utid] = start
            ptr_buf_tensor(psum)[utid] = start + counts[utid]
        gpu.barrier()

        # E: each row's lane turns the slots it took into contiguous rows.
        if row < valid_rows:
            for j in range_constexpr(topk):
                slot = rows_out[out_base + j]
                rows_out[out_base + j] = starts[ids_out[out_base + j]] + slot

    @flyc.jit
    def launch_herd_route_psum(
        candidate_ids: fx.Pointer,
        candidate_values: fx.Pointer,
        topk_weights: fx.Pointer,
        topk_ids: fx.Pointer,
        topids_to_rows: fx.Pointer,
        masked_m: fx.Pointer,
        psum: fx.Pointer,
        num_valid_rows: fx.Pointer,
        rows: fx.Int32,
        tile_m: fx.Int32,
        routed_scaling_factor: Float32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        herd_route_psum_kernel(
            candidate_ids,
            candidate_values,
            topk_weights,
            topk_ids,
            topids_to_rows,
            masked_m,
            psum,
            num_valid_rows,
            rows,
            tile_m,
            routed_scaling_factor,
        ).launch(grid=(1, 1, 1), block=(block, 1, 1), stream=stream)

    launch_herd_route_psum.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_herd_route_psum

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""HERD finalize fused with the grouped-MoE route and tile-aligned psum (FlyDSL).

One workgroup does what the HERD finalize, the route-counter fill, the route
kernel and the psum remap do in four launches. Rows at or past
``num_valid_rows`` get ids -1, weights 0 and ``DROPPED_ROUTE_ROW``, and count
toward neither popularity nor any expert. The scan takes one lane per expert
and the row phase one lane per row, so both are bounded by the block.

Every global access is coalesced: the candidates are copied into LDS, each
row's lane works out of LDS (rows are ``kp1`` words apart, an odd stride, so
the lanes of a wave hit distinct banks), and the outputs are staged in LDS and
copied out. The row phase picks the drop exactly as
``herd_topk.build_herd_finalize_module`` does, so routes and weights match it
bit for bit; it orders the kept routes by expert id with a sorting network over
``id * 32 + candidate`` keys rather than by pairwise ranks.
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
# Candidate index bits of a sort key: id * 32 + candidate, for K+1 <= 32.
_KEY_SHIFT = 5
# Popularity and route counters are split into this many copies by wave: under
# a skewed router one expert takes most rows, and a single counter would
# serialize all of their atomics.
_COUNTER_COPIES = 4


def _sort_network(n: int) -> list[tuple[int, int]]:
    """Batcher's odd-even merge sort on the next power of two, keeping only the
    comparators within the first ``n`` inputs (the rest would compare +inf)."""
    size = 1 << (n - 1).bit_length()
    pairs = []
    p = 1
    while p < size:
        k = p
        while k >= 1:
            for j in range(k % p, size - k, 2 * k):
                for i in range(min(k, size - j - k)):
                    if (i + j) // (2 * p) == (i + j + k) // (2 * p):
                        pairs.append((i + j, i + j + k))
            k //= 2
        p *= 2
    return [(a, b) for a, b in pairs if b < n]


@cache
def build_herd_route_psum_module(
    topk: int, num_experts: int, has_num_valid_rows: bool, wave_size: int = 32
):
    """JIT launcher: HERD finalize + route + tile-aligned psum + row remap."""
    if not 0 < topk < num_experts <= HERD_ROUTE_BLOCK or topk + 1 > 1 << _KEY_SHIFT:
        raise ValueError(f"unsupported HERD route: topk={topk}, experts={num_experts}")
    kp1 = topk + 1
    block = HERD_ROUTE_BLOCK
    waves = block // wave_size
    network = _sort_network(kp1)
    copies = _COUNTER_COPIES

    @fx.struct
    class SharedStorage:
        popularity: fx.Array[Int32, copies * num_experts, 16]
        counts: fx.Array[Int32, copies * num_experts, 16]
        starts: fx.Array[Int32, num_experts, 16]
        wave_sums: fx.Array[Int32, waves, 16]
        # Row r's candidates at [r * kp1, r * kp1 + kp1); the row phase then
        # stages its K kept ids / weights / expert slots in the first K words.
        ids: fx.Array[Int32, block * kp1, 16]
        values: fx.Array[Float32, block * kp1, 16]
        slots: fx.Array[Int32, block * kp1, 16]

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
        lane = utid % fx.Uint32(wave_size)
        wave = utid // fx.Uint32(wave_size)
        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        popularity = lds.popularity.ptr
        counts = lds.counts.ptr
        starts = lds.starts.ptr
        wave_sums = lds.wave_sums.ptr
        s_ids = lds.ids.ptr
        s_values = lds.values.ptr
        s_slots = lds.slots.ptr
        ids_in = ptr_buf_tensor(candidate_ids)
        values_in = ptr_buf_tensor(candidate_values, elem=Float32)
        weights_out = ptr_buf_tensor(topk_weights, elem=Float32)
        ids_out = ptr_buf_tensor(topk_ids)
        rows_out = ptr_buf_tensor(topids_to_rows)

        valid_rows = rows
        if const_expr(has_num_valid_rows):
            valid_rows = fx.min(rows, ptr_buf_tensor(num_valid_rows)[0])

        # A: zero the counters while the candidate loads are in flight.
        in_expert = utid < fx.Uint32(num_experts)
        if in_expert:
            for c in range_constexpr(copies):
                popularity[utid + fx.Uint32(c * num_experts)] = Int32(0)
                counts[utid + fx.Uint32(c * num_experts)] = Int32(0)
        copy_base = fx.Int32(wave % fx.Uint32(copies) * fx.Uint32(num_experts))
        n_in = rows * Int32(kp1)
        n_valid_in = valid_rows * Int32(kp1)
        cand_ids = []
        cand_values = []
        for k in range_constexpr(kp1):
            idx = tid + Int32(k * block)
            cid = Int32(0)
            cval = Float32(0.0)
            if idx < n_in:
                cid = ids_in[idx]
                cval = values_in[idx]
            cand_ids.append(cid)
            cand_values.append(cval)
        gpu.barrier()

        # B: stage the candidates; popularity over the valid rows' candidates.
        for k in range_constexpr(kp1):
            idx = tid + Int32(k * block)
            s_ids[idx] = cand_ids[k]
            s_values[idx] = cand_values[k]
            if idx < n_valid_in:
                fx.atomic_add(
                    popularity + (copy_base + cand_ids[k]),
                    Int32(1),
                    syncscope="workgroup",
                )
        gpu.barrier()
        if const_expr(copies > 1):
            if in_expert:
                total = popularity[utid]
                for c in range_constexpr(1, copies):
                    total = total + popularity[utid + fx.Uint32(c * num_experts)]
                popularity[utid] = total
            gpu.barrier()

        # C: per row, drop one candidate, order the kept K by expert id, take
        # each one's slot in its expert, and stage ids / weights / slots.
        row = tid
        base = row * Int32(kp1)
        if const_expr(has_num_valid_rows):
            if (row >= valid_rows) & (row < rows):
                for p in range_constexpr(topk):
                    s_ids[base + p] = Int32(-1)
                    s_values[base + p] = Float32(0.0)
                    s_slots[base + p] = Int32(DROPPED_ROUTE_ROW)
        if row < valid_rows:
            ids = [s_ids[base + j] for j in range_constexpr(kp1)]
            values = [s_values[base + j] for j in range_constexpr(kp1)]
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

            # Row ids are distinct, so the keys are too.
            keys = [(ids[j] << _KEY_SHIFT) | Int32(j) for j in range_constexpr(kp1)]
            for c in range_constexpr(len(network)):
                a, b = network[c]
                keys[a], keys[b] = fx.min(keys[a], keys[b]), fx.max(keys[a], keys[b])
            drop_key = (drop_id << _KEY_SHIFT) | drop
            kept = [
                (keys[p] < drop_key).select(keys[p], keys[p + 1])
                for p in range_constexpr(topk)
            ]
            # Read every kept weight before the staging below overwrites the row.
            weights = [
                s_values[base + (kept[p] & Int32((1 << _KEY_SHIFT) - 1))] * scale
                for p in range_constexpr(topk)
            ]
            for p in range_constexpr(topk):
                s_ids[base + p] = kept[p] >> _KEY_SHIFT
                s_values[base + p] = weights[p]
            # A hot expert sits at about the same sorted position in every row:
            # each lane starts at its own position so a wave's atomics spread.
            for p in range_constexpr(topk):
                staged = base + fx.Int32((lane + fx.Uint32(p)) % fx.Uint32(topk))
                s_slots[staged] = fx.atomic_add(
                    counts + (copy_base + s_ids[staged]),
                    Int32(1),
                    syncscope="workgroup",
                )
        gpu.barrier()

        # D: copy the kept ids and weights out; tile-aligned exclusive scan of
        # the counts, within each wave by shuffles, then across the waves.
        n_out = rows * Int32(topk)
        for k in range_constexpr(topk):
            out = tid + Int32(k * block)
            if out < n_out:
                staged = out // Int32(topk) * Int32(kp1) + out % Int32(topk)
                ids_out[out] = s_ids[staged]
                weights_out[out] = s_values[staged]

        tile_v = fx.Uint32(tile_m)
        m = fx.Uint32(0)
        if in_expert:
            # Each copy's routes of an expert follow the previous copies'.
            mi = Int32(0)
            for c in range_constexpr(copies):
                at = utid + fx.Uint32(c * num_experts)
                count = counts[at]
                counts[at] = mi
                mi = mi + count
            m = fx.Uint32(mi)
            ptr_buf_tensor(masked_m)[utid] = mi
        aligned = fx.Int32((m + tile_v - 1) // tile_v * tile_v)
        inclusive = aligned
        for offset in range_constexpr(1, wave_size):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            prev = fx.shuffle_up(inclusive, offset, wave_size)
            inclusive = inclusive + (lane >= fx.Uint32(offset)).select(prev, Int32(0))
        if lane == fx.Uint32(wave_size - 1):
            wave_sums[wave] = inclusive
        gpu.barrier()
        before = Int32(0)
        if lane < wave:
            before = wave_sums[lane]
        for offset in range_constexpr(1, wave_size):
            if const_expr((offset & (offset - 1)) != 0):
                continue
            before = before + fx.shuffle_xor(before, offset, wave_size)
        if in_expert:
            start = before + inclusive - aligned
            starts[utid] = start
            ptr_buf_tensor(psum)[utid] = start + fx.Int32(m)
        gpu.barrier()

        # E: each route's contiguous grouped row, copied out.
        for k in range_constexpr(topk):
            out = tid + Int32(k * block)
            if out < n_out:
                staged = out // Int32(topk) * Int32(kp1) + out % Int32(topk)
                expert = s_ids[staged]
                slot = s_slots[staged]
                if expert >= Int32(0):
                    copy = out // Int32(topk * wave_size) % Int32(copies)
                    slot = starts[expert] + counts[copy * Int32(num_experts) + expert] + slot
                rows_out[out] = slot

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

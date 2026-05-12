# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# Triton kernels that convert FusedMoE topk outputs (topk_weights, topk_ids)
# into the (gather_indx, scatter_indx, gate_scal, hist) routing data consumed
# by triton_kernels.matmul_ogs. Three single-CTA kernels implement a counting
# sort over (token, slot) pairs by their expert id; replaces ~12 small torch
# ops (per-row sort, gather, two stable argsorts, advanced indexing, fp32
# histc, plus dtype casts) with three kernel launches.
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_routing_from_topk_hist_kernel_repr = make_kernel_repr(
    "_fused_routing_from_topk_hist_kernel",
    [
        "E",
        "BLOCK_NK",
        "BLOCK_E",
    ],
)

_fused_routing_from_topk_offset_kernel_repr = make_kernel_repr(
    "_fused_routing_from_topk_offset_kernel",
    [
        "E",
        "BLOCK_E",
    ],
)

_fused_routing_from_topk_place_kernel_repr = make_kernel_repr(
    "_fused_routing_from_topk_place_kernel",
    [
        "BLOCK_NK",
    ],
)


@triton.jit(repr=_fused_routing_from_topk_hist_kernel_repr)
def _fused_routing_from_topk_hist_kernel(
    # inputs
    topk_ids_ptr,  # [NK] int32 — flattened topk_ids
    # outputs
    hist_ptr,  # [E] int32 — tokens-per-expert histogram
    # shapes
    NK,  # runtime int — actual valid item count (≤ BLOCK_NK)
    E: tl.constexpr,
    BLOCK_NK: tl.constexpr,  # padded to next pow2 of NK
    BLOCK_E: tl.constexpr,  # padded to next pow2 of E (tl.histogram needs pow2)
):
    """Phase A: histogram via tl.histogram (warp-local shared-memory reduction).

    No global atomics, no debug barrier — the reduction completes within a
    single wave/CTA and the result is written with a plain tl.store.
    ``tl.histogram`` requires ``num_bins`` to be a power of two, so the
    reduction is over BLOCK_E bins; bins ``>= E`` are unreachable because
    expert ids are in ``[0, E)`` and the trailing entries are dropped via
    a masked store.
    """
    item_offs = tl.arange(0, BLOCK_NK)
    item_mask = item_offs < NK
    # Clamp the offset for masked-out lanes to 0 so the pointer arithmetic
    # below stays within the allocated buffers.
    safe_item = tl.where(item_mask, item_offs, 0)
    expt = tl.load(topk_ids_ptr + safe_item, mask=item_mask, other=0).to(tl.int32)

    hist = tl.histogram(expt, BLOCK_E, mask=item_mask)

    e_offs = tl.arange(0, BLOCK_E)
    tl.store(hist_ptr + e_offs, hist, mask=e_offs < E)


@triton.jit(repr=_fused_routing_from_topk_offset_kernel_repr)
def _fused_routing_from_topk_offset_kernel(
    # inputs
    hist_ptr,  # [E] int32 — published by the hist kernel
    # outputs
    offset_ptr,  # [E] int32 — exclusive prefix sum of hist
    # shapes
    E: tl.constexpr,
    BLOCK_E: tl.constexpr,  # padded to next pow2 of E
):
    """Phase B: exclusive prefix-sum hist → offset.

    The previous kernel's exit publishes hist, so this kernel observes
    them on entry without an explicit fence.
    """
    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < E
    safe_e = tl.where(e_mask, e_offs, 0)
    h = tl.load(hist_ptr + safe_e, mask=e_mask, other=0)
    incl = tl.cumsum(h, axis=0)
    excl = incl - h
    tl.store(offset_ptr + safe_e, excl, mask=e_mask)


@triton.jit(repr=_fused_routing_from_topk_place_kernel_repr)
def _fused_routing_from_topk_place_kernel(
    # inputs
    topk_ids_ptr,  # [NK] int32 — flattened topk_ids
    topk_weights_ptr,  # [NK] (any float dtype) — flattened topk_weights
    offset_ptr,  # [E] int32 — exclusive prefix sums from the offset kernel
    # outputs
    topk_indx_ptr,  # [NK] int32 — output gather_indx.src_indx
    gate_indx_ptr,  # [NK] int32 — output gather_indx.dst_indx
    gate_scal_ptr,  # [NK] same dtype as topk_weights
    # shapes
    NK,  # runtime int — actual valid item count (≤ BLOCK_NK)
    BLOCK_NK: tl.constexpr,  # padded to next pow2 of NK
):
    """Phase C: place items.

    For each valid item, atomic_add on offset[expert] returns its
    expert-sorted position; write topk_indx, gate_indx, gate_scal.

    The kernel does NOT pre-sort each token's K experts. The resulting
    topk_indx / gate_indx differ from a stable-argsort reference at
    intra-expert ordering, but they form a valid inverse permutation pair
    and matmul_ogs produces the same per-token aggregation (gather +
    weighted scatter sum are both commutative over a per-expert slice).
    """
    item_offs = tl.arange(0, BLOCK_NK)
    item_mask = item_offs < NK
    safe_item = tl.where(item_mask, item_offs, 0)
    expt = tl.load(topk_ids_ptr + safe_item, mask=item_mask, other=0).to(tl.int32)
    weights = tl.load(topk_weights_ptr + safe_item, mask=item_mask, other=0.0)

    pos = tl.atomic_add(offset_ptr + expt, 1, mask=item_mask)

    # Clamp pos for masked-out lanes — `pos` is undefined there, and
    # `topk_indx_ptr + pos` / `gate_scal_ptr + pos` would otherwise be
    # arbitrary addresses. The mask=False store doesn't write, but the
    # address calc is still evaluated and may fault on OOB pages.
    safe_pos = tl.where(item_mask, pos, 0)

    # gate_indx[i]   = pos       (original_flat → expert_sorted_pos)
    tl.store(gate_indx_ptr + safe_item, pos, mask=item_mask)
    # topk_indx[pos] = i         (expert_sorted_pos → original_flat)
    tl.store(topk_indx_ptr + safe_pos, item_offs.to(tl.int32), mask=item_mask)
    # gate_scal[pos] = weight at the original flat item
    tl.store(gate_scal_ptr + safe_pos, weights, mask=item_mask)

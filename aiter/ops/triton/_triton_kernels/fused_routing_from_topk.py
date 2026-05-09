# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# Triton kernel that converts FusedMoE topk outputs (topk_weights, topk_ids)
# into the (gather_indx, scatter_indx, gate_scal, hist) routing data consumed
# by triton_kernels.matmul_ogs. Single-CTA counting sort over (token, slot)
# pairs by their expert id; replaces ~12 small torch ops (per-row sort,
# gather, two stable argsorts, advanced indexing, fp32 histc, plus dtype
# casts) with one kernel launch.
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_routing_from_topk_kernel_repr = make_kernel_repr(
    "_fused_routing_from_topk_kernel",
    [
        "E",
        "BLOCK_NK",
        "BLOCK_E",
    ],
)


@triton.jit(repr=_fused_routing_from_topk_kernel_repr)
def _fused_routing_from_topk_kernel(
    # inputs
    topk_ids_ptr,  # [NK] int32 — flattened topk_ids
    topk_weights_ptr,  # [NK] (any float dtype) — flattened topk_weights
    # scratch / outputs
    hist_ptr,  # [E] int32 — caller MUST zero before launch
    offset_ptr,  # [E] int32 — scratch (overwritten)
    topk_indx_ptr,  # [NK] int32 — output gather_indx.src_indx
    gate_indx_ptr,  # [NK] int32 — output gather_indx.dst_indx
    gate_scal_ptr,  # [NK] same dtype as topk_weights
    # shapes
    NK,  # runtime int — actual valid item count (≤ BLOCK_NK)
    E: tl.constexpr,
    BLOCK_NK: tl.constexpr,  # padded to next pow2 of NK
    BLOCK_E: tl.constexpr,  # padded to next pow2 of E
):
    """Single-CTA counting sort over (token, slot) → expert_id.

    Three phases:
      A. Build histogram via global atomic_add on hist[E].
      B. Exclusive prefix-sum hist → offset[E] (in-CTA cumsum).
      C. Place each item via atomic_add on offset[expert] → its expert-sorted
         position; write topk_indx, gate_indx, gate_scal.

    Designed for decode (NK ≤ a few thousand, single CTA fits in SMEM/VGPR).
    The Python wrapper falls back to a reference impl when NK is too large.

    The kernel does NOT pre-sort each token's K experts. The resulting
    topk_indx / gate_indx differ from a stable-argsort reference at
    intra-expert ordering, but they form a valid inverse permutation pair
    and matmul_ogs produces the same per-token aggregation (gather +
    weighted scatter sum are both commutative over a per-expert slice).

    ``num_warps=1`` is required at launch site: a single hardware wave per
    CTA serialises all atomics through one wavefront, removing inter-wave
    memory-ordering races on the global hist/offset atomics that
    ``tl.debug_barrier`` alone does not fully fence on AMD CDNA.
    """
    # ===== Phase A: build histogram =====
    item_offs = tl.arange(0, BLOCK_NK)
    item_mask = item_offs < NK
    # Clamp the offset for masked-out lanes to 0 so the pointer arithmetic
    # below stays within the allocated buffers, even on backends where a
    # masked store/atomic still touches the cache line of the computed
    # address. Backing buffers are sized exactly to NK / E.
    safe_item = tl.where(item_mask, item_offs, 0)
    expt = tl.load(topk_ids_ptr + safe_item, mask=item_mask, other=0).to(tl.int32)
    weights = tl.load(topk_weights_ptr + safe_item, mask=item_mask, other=0.0)

    # Per-item atomic increment of hist[expt]. expt is 0 for masked lanes
    # (loaded with other=0), so the address is in-bounds either way.
    tl.atomic_add(hist_ptr + expt, 1, mask=item_mask, sem="release")
    tl.debug_barrier()

    # ===== Phase B: exclusive prefix-sum hist → offset =====
    # cache_modifier=".cv" bypasses L1 so Phase B observes the values
    # atomically-incremented in Phase A.
    e_offs = tl.arange(0, BLOCK_E)
    e_mask = e_offs < E
    safe_e = tl.where(e_mask, e_offs, 0)
    h = tl.load(hist_ptr + safe_e, mask=e_mask, other=0, cache_modifier=".cv")
    incl = tl.cumsum(h, axis=0)
    excl = incl - h
    tl.atomic_xchg(offset_ptr + safe_e, excl, mask=e_mask, sem="release")
    tl.debug_barrier()

    # ===== Phase C: place items =====
    # atomic_add on offset[expt] with acquire — observe the prefix sums
    # written by Phase B with the proper memory ordering.
    pos = tl.atomic_add(offset_ptr + expt, 1, mask=item_mask, sem="acquire")

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

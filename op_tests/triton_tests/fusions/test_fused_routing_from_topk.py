# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# Correctness test for ``aiter.ops.triton.fused_routing_from_topk``.
#
# The fused kernel skips the per-row sort that a stable-argsort reference
# performs, so its ``topk_indx``/``gate_indx`` may differ from the reference
# at *intra-expert* ordering. Equivalence is therefore checked at multiple
# levels:
#
#   1. Inverse-permutation invariant on the fused output.
#   2. Bucket invariant: items at expert-sorted positions for expert ``e``
#      reference (token, slot) pairs whose expert id equals ``e``.
#   3. ``hist`` matches the reference exactly.
#   4. Per-expert ``(token, weight)`` multisets match the reference (and a
#      ground-truth bucket built directly from inputs).
import pytest
import torch

from aiter.ops.triton.fusions.fused_routing_from_topk import fused_routing_from_topk

DEVICE = "cuda"


# ---------------------------------------------------------------------------
# Reference implementation — mirrors the multi-kernel torch chain originally
# used to bridge FusedMoE.select_experts to triton_kernels.matmul_ogs.
# Returns ``(hist, topk_indx, gate_indx, gate_scal)`` for direct comparison
# against the fused kernel.
# ---------------------------------------------------------------------------
def routing_from_topk_reference(topk_weights, topk_ids, n_expts_tot):
    """Multi-kernel torch reference for fused_routing_from_topk.

    Per-row sort of ``topk_ids`` followed by a stable global argsort by
    expert id, an inverse-permutation argsort, and an integer histogram.
    The output indices are bit-exact stable across runs (modulo torch
    version), unlike the fused kernel which is non-deterministic at
    intra-expert ordering.
    """
    expt_indx_sorted, sort_indices = torch.sort(topk_ids.int(), dim=1)
    expt_scal_sorted = torch.gather(topk_weights, 1, sort_indices.long())

    expt_scal = expt_scal_sorted.reshape(-1).to(topk_weights.dtype)
    expt_indx = expt_indx_sorted.reshape(-1).to(torch.int32)

    topk_indx = torch.argsort(expt_indx, stable=True).int()
    gate_indx = torch.argsort(topk_indx, stable=True).int()
    gate_scal = expt_scal[topk_indx.long()]

    hist = torch.histc(expt_indx.float(), bins=n_expts_tot, max=n_expts_tot - 1).int()
    return hist, topk_indx, gate_indx, gate_scal


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------
def _make_inputs(n_tokens, n_expts_act, n_expts_tot, dtype, device, seed):
    """Random topk-style inputs: distinct expert ids per row + L1-normalized
    positive weights (matches FusedMoE.select_experts post-renormalize)."""
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.empty(n_tokens, n_expts_act, dtype=torch.int32, device=device)
    for n in range(n_tokens):
        ids[n] = torch.randperm(n_expts_tot, generator=g, device=device)[
            :n_expts_act
        ].to(torch.int32)
    weights = torch.rand(n_tokens, n_expts_act, generator=g, device=device, dtype=dtype)
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return ids, weights


def _check_routing_invariants(
    hist,
    topk_indx,
    gate_indx,
    gate_scal,
    topk_ids,
    n_expts_tot,
    *,
    bucket_unsorted_layout,
):
    """Sanity invariants that any valid fused-routing output must satisfy.

    ``bucket_unsorted_layout`` enables the bucket invariant against
    ``topk_ids.flatten()`` directly. That only holds for the fused kernel
    (which skips the per-row sort); a stable-argsort reference uses a
    per-token-sorted flat layout, so its ``topk_indx`` indexes a different
    array.
    """
    n_tokens, K = topk_ids.shape
    NK = n_tokens * K
    device = topk_ids.device

    # 1. Inverse permutation: gate_indx[topk_indx[j]] == j for all j.
    iota = torch.arange(NK, dtype=torch.int32, device=device)
    inv_check = gate_indx[topk_indx.long()]
    assert torch.equal(inv_check, iota), (
        "gate_indx[topk_indx[j]] != j (first mismatch at "
        f"{(inv_check != iota).nonzero()[0].item()})"
    )

    # 2. hist is non-negative int32 and sums to NK.
    assert hist.dtype == torch.int32, f"hist dtype != int32 (got {hist.dtype})"
    assert (hist >= 0).all(), "hist has negative entries"
    assert hist.sum().item() == NK, f"hist.sum()={hist.sum().item()} != NK={NK}"

    # 3. gate_scal is finite and same length as topk_indx.
    assert gate_scal.numel() == NK
    assert torch.isfinite(gate_scal).all(), "gate_scal has non-finite values"

    # 4. Bucket invariant (fused-only): items at expert-sorted positions
    #    [prefix[e], prefix[e+1]) reference original (token, slot) pairs
    #    whose expert id equals e in the *unsorted* topk_ids flat layout.
    if bucket_unsorted_layout:
        prefix_end = torch.cumsum(hist, dim=0).cpu().tolist()
        flat_ids = topk_ids.reshape(-1).cpu().tolist()
        src = topk_indx.cpu().tolist()
        start = 0
        for e in range(n_expts_tot):
            end = prefix_end[e]
            for j in range(start, end):
                assert flat_ids[src[j]] == e, (
                    f"expert-sorted pos {j}: expected expert {e} "
                    f"but original_flat={src[j]} has expert "
                    f"{flat_ids[src[j]]}"
                )
            start = end


def _ground_truth_buckets(topk_ids, topk_weights):
    """Build the (token, weight) multiset per expert directly from the
    inputs — independent of any routing implementation."""
    _, K = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).cpu().tolist()
    flat_w = topk_weights.reshape(-1).float().cpu().tolist()
    buckets: dict[int, list] = {}
    for i, e in enumerate(flat_ids):
        token = i // K
        buckets.setdefault(e, []).append((token, flat_w[i]))
    for e in buckets:
        buckets[e].sort()
    return buckets


def _per_expert_triples(hist, topk_indx, gate_scal, K):
    """Walk the expert-sorted layout and bucket (token, weight) pairs by
    expert id, using ``hist`` to determine each bucket's slice."""
    NK = topk_indx.numel()
    n_expts_tot = hist.numel()
    cum = torch.cumsum(hist, dim=0).cpu().tolist()

    src = topk_indx.cpu().tolist()
    scal = gate_scal.float().cpu().tolist()

    buckets: dict[int, list] = {e: [] for e in range(n_expts_tot)}
    e = 0
    for j in range(NK):
        while e < n_expts_tot and j >= cum[e]:
            e += 1
        original_flat = src[j]
        token = original_flat // K
        buckets[e].append((token, scal[j]))
    for e in buckets:
        buckets[e].sort()
    return buckets


def _compare_buckets(ref_buckets, test_buckets, atol=1e-6):
    keys = set(ref_buckets) | set(test_buckets)
    for e in keys:
        rb = ref_buckets.get(e, [])
        tb = test_buckets.get(e, [])
        assert len(rb) == len(
            tb
        ), f"expert {e}: bucket size ref={len(rb)} test={len(tb)}"
        for (tt_r, w_r), (tt_t, w_t) in zip(rb, tb):
            assert tt_r == tt_t, f"expert {e}: token mismatch ref={tt_r} test={tt_t}"
            assert (
                abs(w_r - w_t) <= atol
            ), f"expert {e}: token {tt_r} weight ref={w_r} test={w_t}"


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n_tokens, n_expts_act, n_expts_tot",
    [
        # V4-Flash decode shapes (E=256, K=6).
        (1, 6, 256),
        (16, 6, 256),
        (64, 6, 256),
        (256, 6, 256),
        # Generic decode shapes used by other MoE configs.
        (1, 8, 384),
        (4, 8, 384),
        (64, 8, 384),
        (256, 8, 384),
        # Edge: small E.
        (32, 4, 16),
        # Boundary: NK at the kernel's MAX_NK = 4096.
        (512, 8, 384),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32])
def test_fused_routing_from_topk(n_tokens, n_expts_act, n_expts_tot, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    topk_ids, topk_weights = _make_inputs(
        n_tokens, n_expts_act, n_expts_tot, dtype, DEVICE, seed=0
    )

    ref_hist, ref_topk_indx, ref_gate_indx, ref_gate_scal = routing_from_topk_reference(
        topk_weights, topk_ids, n_expts_tot
    )
    _check_routing_invariants(
        ref_hist,
        ref_topk_indx,
        ref_gate_indx,
        ref_gate_scal,
        topk_ids,
        n_expts_tot,
        bucket_unsorted_layout=False,  # ref uses per-row-sorted layout
    )
    ground_buckets = _ground_truth_buckets(topk_ids, topk_weights)
    ref_buckets = _per_expert_triples(
        ref_hist, ref_topk_indx, ref_gate_scal, n_expts_act
    )
    _compare_buckets(ground_buckets, ref_buckets)

    test_hist, test_topk_indx, test_gate_indx, test_gate_scal = fused_routing_from_topk(
        topk_weights, topk_ids, n_expts_tot
    )
    _check_routing_invariants(
        test_hist,
        test_topk_indx,
        test_gate_indx,
        test_gate_scal,
        topk_ids,
        n_expts_tot,
        bucket_unsorted_layout=True,  # fused uses unsorted topk_ids layout
    )

    # hist must match the reference exactly.
    assert torch.equal(
        ref_hist, test_hist
    ), f"hist mismatch:\n  ref={ref_hist}\n  fused={test_hist}"

    # Per-expert (token, weight) multisets match the reference.
    test_buckets = _per_expert_triples(
        test_hist, test_topk_indx, test_gate_scal, n_expts_act
    )
    _compare_buckets(ref_buckets, test_buckets)

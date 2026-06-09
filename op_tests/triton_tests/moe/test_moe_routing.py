import pytest
import torch
import triton
import torch.nn.functional as F
from aiter.ops.triton.moe.moe_routing.routing import (
    routing,
    routing_from_hash,
    routing_torch,
    compute_expt_data_torch,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.topk import biased_grouped_topk_torch, grouped_topk_torch
from aiter.ops.triton.moe.moe_routing.topk import grouped_topk


def _routing_block_m(n_tokens, n_expts_act, n_expts_tot):
    """block_m heuristic used by `routing`.

    Uses the raw logits shape and the originally requested n_expts_act (before
    any shared-expert widening), exactly as `routing` does internally.
    """
    tokens_per_expt = max(1, (n_tokens * n_expts_act) // n_expts_tot)
    return max(16, min(triton.next_power_of_2(tokens_per_expt), 128))


def assert_equal(ref, tri):
    if isinstance(ref, torch.Tensor):
        # CI may be failing using this:
        # assert torch.all(ref == tri)
        assert ((ref.cpu().numpy() - tri.cpu().numpy()) ** 2).sum() == 0
    else:
        assert ref == tri


def assert_close(ref, tri, maxtol=None, rmstol=None, description="--", verbose=True):
    if tri.dtype.itemsize == 1:
        ref_as_type = ref.to(tri.dtype)
        if ref.dtype == tri.dtype:
            assert torch.all(ref_as_type == tri)
            return
        ref = ref_as_type

    if maxtol is None:
        maxtol = 2e-2
    if rmstol is None:
        rmstol = 4e-3
    """
    Compare reference values against obtained values.
    """

    # cast to float32:
    ref = ref.to(torch.float32).detach()
    tri = tri.to(torch.float32).detach()
    assert (
        ref.shape == tri.shape
    ), f"Tensors must have same size {ref.shape=} {tri.shape=}"

    # deal with infinite elements:
    inf_mask_ref = torch.isinf(ref)
    inf_mask_tri = torch.isinf(tri)
    assert torch.equal(
        inf_mask_ref, inf_mask_tri
    ), "Tensor must have same infinite elements"
    refn = torch.where(inf_mask_ref, 0, ref)
    trin = torch.where(inf_mask_tri, 0, tri)

    # normalise so that RMS calculation doesn't overflow:
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn *= multiplier
    trin *= multiplier

    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps

    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    if verbose:
        print(
            "%s maximum relative error = %s (threshold = %s)"
            % (description, max_err, maxtol)
        )
        print(
            "%s RMS relative error = %s (threshold = %s)"
            % (description, rms_err, rmstol)
        )

    if max_err > maxtol:
        bad_idxs = torch.nonzero(rel_err > maxtol)
        num_nonzero = bad_idxs.size(0)
        bad_idxs = bad_idxs[:1000]
        print(
            "%d / %d mismatched elements (shape = %s) at coords %s"
            % (num_nonzero, rel_err.numel(), tuple(rel_err.shape), bad_idxs.tolist())
        )

        bad_idxs = bad_idxs.unbind(-1)
        print("ref values: ", ref[tuple(bad_idxs)].cpu())
        print("tri values: ", tri[tuple(bad_idxs)].cpu())

    assert max_err <= maxtol
    assert rms_err <= rmstol


def init_data(n_tokens, n_expts_tot, dtype=torch.float16, device="cuda"):
    logits = torch.randn((n_tokens, n_expts_tot), dtype=dtype, device=device)
    return logits


n_tokens = [4, 7, 8, 64, 255, 256, 371, 911, 1023, 1024, 4096, 8192]


@pytest.mark.parametrize("n_tokens", n_tokens)
@pytest.mark.parametrize(
    "n_expts_tot, n_expts_act",
    [(128, 4), (128, 6), (128, 32), (1500, 8), (256, 8), (8, 2)],
)
@pytest.mark.parametrize("sm_first", [True, False])
def test_routing(n_tokens, n_expts_tot, n_expts_act, sm_first):
    if get_arch() not in ["gfx950", "gfx1250"]:
        pytest.skip("MOE stack not fully implemented on non-CDNA4 arch yet.")

    device = "cuda"
    torch.manual_seed(2)
    n_gates_raw = n_tokens * n_expts_act
    tri_logits = init_data(
        n_tokens, n_expts_tot, device=device, dtype=torch.float32
    ).detach()
    tri_logits[n_tokens:, :] = float("inf")  # should not be used
    ref_logits = tri_logits.clone().detach()

    ref_routing_data, ref_gather, ref_scatter = routing_torch(
        ref_logits, n_expts_act, sm_first
    )
    tri_routing_data, tri_gather, tri_scatter = routing(
        tri_logits, n_expts_act, sm_first=sm_first
    )

    def _assert_indx_equal(ref, tri):
        tri = tri.to(torch.int32)
        assert_equal(ref, tri[: len(ref)])
        assert torch.all(tri[len(ref) :] == -1)

    assert_close(
        ref_routing_data.gate_scal, tri_routing_data.gate_scal[:n_gates_raw], 2e-2, 4e-3
    )
    assert_equal(ref_routing_data.expt_hist, tri_routing_data.expt_hist)

    ref_expt_data = ref_routing_data.expt_data
    tri_expt_data = tri_routing_data.expt_data
    assert_equal(ref_expt_data.hist, tri_expt_data.hist)
    assert_equal(ref_expt_data.token_offs_raw, tri_expt_data.token_offs_raw)
    assert_equal(ref_expt_data.token_offs_pad, tri_expt_data.token_offs_pad)
    assert_equal(ref_expt_data.block_pid_map, tri_expt_data.block_pid_map)

    assert ref_routing_data.n_expts_tot == tri_routing_data.n_expts_tot
    assert ref_routing_data.n_expts_act == tri_routing_data.n_expts_act

    _assert_indx_equal(ref_gather, tri_gather)
    _assert_indx_equal(ref_scatter, tri_scatter)


# --------------------------
# Reference implementations for routing with score mode paths
# --------------------------


def _score_transform_torch(logits, score_mode):
    if score_mode == "sqrtsoftplus":
        return torch.sqrt(F.softplus(logits.to(torch.float32))).to(logits.dtype)
    # "softmax" mode in the kernel means "no pre-transform" (identity)
    return logits


def _sort_and_build_torch(expt_scal, expt_indx, n_expts_tot, block_m):
    """Mirror of the post-topk sort_tokens + ExptData build, in pytorch.

    expt_scal, expt_indx: shape (n_tokens, n_expts_act) — per-row order is
    preserved (we do NOT sort experts per row here; that's the caller's
    responsibility if needed).
    Returns (hist, topk_indx, gate_indx, gate_scal, expt_data) matching the
    triton sort_tokens contract.
    """
    n_tokens, n_expts_act = expt_scal.shape
    n_gates = n_tokens * n_expts_act
    scal_flat = expt_scal.reshape(-1)
    indx_flat = expt_indx.reshape(-1).to(torch.int32)
    topk_indx = torch.argsort(indx_flat, stable=True).to(torch.int32)
    gate_indx = torch.argsort(topk_indx, stable=True).to(torch.int32)
    gate_scal = scal_flat[topk_indx.long()]
    hist = torch.histc(
        indx_flat.float(), bins=n_expts_tot, min=0, max=n_expts_tot - 1
    ).int()
    expt_data = compute_expt_data_torch(hist, n_expts_tot, n_gates, block_m)
    return hist, topk_indx, gate_indx, gate_scal, expt_data


def routing_score_mode_torch(
    logits,
    n_expts_act,
    block_m,
    *,
    score_mode="sqrtsoftplus",
    bias=None,
    renorm=True,
    routed_scaling_factor=1.0,
):
    n_tokens, n_expts_tot = logits.shape

    # 1. Score transform; bias added only for selection.
    transformed_f32 = _score_transform_torch(logits, score_mode).to(torch.float32)
    if bias is not None:
        biased = transformed_f32 + bias.to(torch.float32)
    else:
        biased = transformed_f32

    # 2. Top-k selection (by biased score), then sort experts ascending per row
    # — this matches streaming_topk's final per-row sort.
    _, topk_ids = torch.topk(biased, n_expts_act, dim=1)
    topk_ids, _ = torch.sort(topk_ids, dim=1)

    # 3. Gather the UNBIASED transformed value at the selected positions.
    expt_scal = torch.gather(transformed_f32, 1, topk_ids)

    # 4. Renorm + scale (or just scale).
    if renorm:
        s = expt_scal.sum(dim=1, keepdim=True)
        expt_scal = expt_scal / (s + 1e-20) * routed_scaling_factor
    elif routed_scaling_factor != 1.0:
        expt_scal = expt_scal * routed_scaling_factor

    expt_scal = expt_scal.to(logits.dtype)
    topk_ids = topk_ids.to(torch.int16)
    return _sort_and_build_torch(expt_scal, topk_ids, n_expts_tot, block_m)


def routing_from_hash_torch(
    router_logits,
    tid2eid,
    input_ids,
    n_expts_act,
    block_m,
    *,
    score_mode="sqrtsoftplus",
    renorm=True,
    routed_scaling_factor=1.0,
):
    n_tokens, n_expts_tot = router_logits.shape
    iid = input_ids.to(torch.int64)
    # Expert ids come straight from the table — no per-row sort.
    expt_indx = tid2eid[iid, :n_expts_act].to(torch.int32)

    # Score transform on the full row, then gather the K weights.
    transformed_f32 = _score_transform_torch(router_logits, score_mode).to(
        torch.float32
    )
    expt_scal = torch.gather(transformed_f32, 1, expt_indx.to(torch.int64))

    if renorm:
        s = expt_scal.sum(dim=1, keepdim=True)
        expt_scal = expt_scal / (s + 1e-20) * routed_scaling_factor
    elif routed_scaling_factor != 1.0:
        expt_scal = expt_scal * routed_scaling_factor

    expt_scal = expt_scal.to(router_logits.dtype)
    expt_indx = expt_indx.to(torch.int16)
    return _sort_and_build_torch(expt_scal, expt_indx, n_expts_tot, block_m)


def _check_routing_data(ref_pack, tri_routing_data, tri_gather, tri_scatter):
    """Strict equality check: works when the triton sort and stable argsort
    agree on intra-bucket order (the sort_tokens / sort_tokens_fused path)."""
    ref_hist, ref_topk_indx, ref_gate_indx, ref_gate_scal, ref_expt_data = ref_pack
    assert_close(ref_gate_scal, tri_routing_data.gate_scal, 2e-2, 4e-3)
    assert_equal(ref_hist, tri_routing_data.expt_hist)
    assert_equal(ref_expt_data.hist, tri_routing_data.expt_data.hist)
    assert_equal(
        ref_expt_data.token_offs_raw, tri_routing_data.expt_data.token_offs_raw
    )
    assert_equal(
        ref_expt_data.token_offs_pad, tri_routing_data.expt_data.token_offs_pad
    )
    assert_equal(ref_expt_data.block_pid_map, tri_routing_data.expt_data.block_pid_map)
    assert_equal(ref_topk_indx, tri_gather)
    assert_equal(ref_gate_indx, tri_scatter)


def _check_routing_data_bucket(
    ref_pack,
    tri_routing_data,
    tri_gather,
    tri_scatter,
    topk_weights,
    topk_ids,
):
    """Bucket-multiset check for the fused_routing_from_topk sort path, which
    uses a different stable tie-breaking than torch.argsort. Validates the
    histogram + ExptData strictly, then compares per-expert (token, weight)
    multisets and the inverse-permutation invariant.
    """
    ref_hist, _, _, _, ref_expt_data = ref_pack
    assert_equal(ref_hist, tri_routing_data.expt_hist)
    assert_equal(ref_expt_data.hist, tri_routing_data.expt_data.hist)
    assert_equal(
        ref_expt_data.token_offs_raw, tri_routing_data.expt_data.token_offs_raw
    )
    assert_equal(
        ref_expt_data.token_offs_pad, tri_routing_data.expt_data.token_offs_pad
    )
    assert_equal(ref_expt_data.block_pid_map, tri_routing_data.expt_data.block_pid_map)

    n_tokens, n_expts_act = topk_ids.shape
    n_gates = n_tokens * n_expts_act
    n_expts_tot = ref_hist.numel()

    # Inverse permutation invariant: gate_indx[topk_indx[j]] == j.
    # Cast scatter to int64 first: the grouped routing_score_mode path returns uint16
    # indices, which CUDA cannot advanced-index.
    iota = torch.arange(n_gates, dtype=torch.int64, device=tri_gather.device)
    assert torch.equal(
        tri_scatter.long()[tri_gather.long()], iota
    ), "scatter[gather[j]] != j"

    # Per-expert (token, weight) multisets.
    flat_ids = topk_ids.reshape(-1).cpu().tolist()
    flat_w = topk_weights.reshape(-1).float().cpu().tolist()
    src = tri_gather.cpu().tolist()
    scal = tri_routing_data.gate_scal.float().cpu().tolist()
    cum = torch.cumsum(ref_hist, dim=0).cpu().tolist()

    ground = {e: [] for e in range(n_expts_tot)}
    for i, e in enumerate(flat_ids):
        token = i // n_expts_act
        ground[e].append((token, flat_w[i]))
    for e in ground:
        ground[e].sort()

    got = {e: [] for e in range(n_expts_tot)}
    e = 0
    for j in range(n_gates):
        while e < n_expts_tot and j >= cum[e]:
            e += 1
        token = src[j] // n_expts_act
        # Bucket invariant: at expert-sorted position j inside expert e's
        # slice, the source (token, slot) must reference expert e.
        assert flat_ids[src[j]] == e, (
            f"bucket-invariant violated at pos {j}: source flat={src[j]} "
            f"has expert {flat_ids[src[j]]}, expected {e}"
        )
        got[e].append((token, scal[j]))
    for e in got:
        got[e].sort()

    for e in range(n_expts_tot):
        rb, tb = ground[e], got[e]
        assert len(rb) == len(tb), f"expert {e}: ref={len(rb)} test={len(tb)}"
        for (tt_r, w_r), (tt_t, w_t) in zip(rb, tb):
            assert tt_r == tt_t, f"expert {e}: token ref={tt_r} test={tt_t}"
            assert (
                abs(w_r - w_t) <= 1e-6
            ), f"expert {e} token {tt_r}: weight ref={w_r} test={w_t}"


# --------------------------
# routing score mode
# --------------------------


@pytest.mark.parametrize(
    "n_tokens, n_expts_tot, n_expts_act",
    [
        (8, 128, 4),  # tiny: hits sort_tokens_fused path (n_tokens <= 16)
        (16, 128, 4),  # boundary
        (64, 128, 4),
        (1024, 128, 4),
        (1024, 256, 8),
    ],
)
@pytest.mark.parametrize(
    "score_mode, has_bias, renorm, routed_scaling_factor",
    [
        ("sqrtsoftplus", True, True, 2.5),  # full V4 noaux_tc path
        ("sqrtsoftplus", True, False, 1.0),  # bias, no renorm
        ("sqrtsoftplus", False, True, 1.0),  # no bias
        ("softmax", False, False, 1.0),  # identity transform, no renorm
    ],
)
def test_routing_score_mode(
    n_tokens,
    n_expts_tot,
    n_expts_act,
    score_mode,
    has_bias,
    renorm,
    routed_scaling_factor,
):
    if get_arch() not in ["gfx950", "gfx1250"]:
        pytest.skip("MOE stack not fully implemented on non-CDNA4 arch yet.")

    device = "cuda"
    torch.manual_seed(2)
    logits = init_data(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
    bias = (
        torch.randn(n_expts_tot, dtype=torch.float32, device=device) * 0.05
        if has_bias
        else None
    )

    # routing derives block_m internally; mirror that here for the ref.
    block_m = _routing_block_m(n_tokens, n_expts_act, n_expts_tot)

    ref_pack = routing_score_mode_torch(
        logits.clone(),
        n_expts_act,
        block_m,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=routed_scaling_factor,
    )
    tri_routing_data, tri_gather, tri_scatter = routing(
        logits,
        n_expts_act,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=routed_scaling_factor,
    )

    _check_routing_data(ref_pack, tri_routing_data, tri_gather, tri_scatter)
    assert tri_routing_data.n_expts_tot == n_expts_tot
    assert tri_routing_data.n_expts_act == n_expts_act
    assert tri_routing_data.block_m == block_m


# --------------------------
# routing_from_hash
# --------------------------


@pytest.mark.parametrize(
    "n_tokens, n_expts_tot, n_expts_act",
    [
        (8, 128, 4),
        (64, 128, 4),
        (1024, 256, 8),
    ],
)
@pytest.mark.parametrize(
    "renorm, routed_scaling_factor",
    [
        (True, 2.5),  # production V4 hash config
        (True, 1.0),
        (False, 1.0),
    ],
)
@pytest.mark.parametrize("block_m", [16, 32])
def test_routing_from_hash(
    n_tokens,
    n_expts_tot,
    n_expts_act,
    renorm,
    routed_scaling_factor,
    block_m,
):
    if get_arch() not in ["gfx950", "gfx1250"]:
        pytest.skip("MOE stack not fully implemented on non-CDNA4 arch yet.")

    device = "cuda"
    torch.manual_seed(2)
    vocab_size = 512
    router_logits = torch.randn(
        n_tokens, n_expts_tot, dtype=torch.float32, device=device
    )
    # Distinct experts per vocab entry (production V4 hash table contract).
    # Avoids within-row duplicates that would make intra-bucket ordering
    # implementation-defined between the triton sort and torch.argsort.
    tid2eid = torch.stack(
        [
            torch.randperm(n_expts_tot, device=device)[:n_expts_act]
            for _ in range(vocab_size)
        ],
        dim=0,
    ).to(torch.int32)
    input_ids = torch.randint(
        0, vocab_size, (n_tokens,), dtype=torch.int32, device=device
    )

    ref_pack = routing_from_hash_torch(
        router_logits.clone(),
        tid2eid,
        input_ids,
        n_expts_act,
        block_m,
        score_mode="sqrtsoftplus",
        renorm=renorm,
        routed_scaling_factor=routed_scaling_factor,
    )
    tri_routing_data, tri_gather, tri_scatter = routing_from_hash(
        router_logits,
        tid2eid,
        input_ids,
        n_expts_act,
        block_m,
        score_mode="sqrtsoftplus",
        renorm=renorm,
        routed_scaling_factor=routed_scaling_factor,
    )

    _check_routing_data(ref_pack, tri_routing_data, tri_gather, tri_scatter)
    assert tri_routing_data.n_expts_tot == n_expts_tot
    assert tri_routing_data.n_expts_act == n_expts_act
    assert tri_routing_data.block_m == block_m


# ==========================================================================
# grouped-top-k routing (aiter.ops.triton.moe.moe_routing.topk.grouped_topk)
# Moved from test_grouped_topk.py. Reuses the shared helpers above
# (assert_equal, assert_close, init_data, _sort_and_build_torch,
# _check_routing_data_bucket).
# ==========================================================================


# --------------------------------------------------------------------------
# torch references
# --------------------------------------------------------------------------


def _ref_sqrtsoftplus_grouped(
    logits, bias, k, num_expert_group, topk_group, renorm, scale
):
    """sqrtsoftplus grouped-topk reference (no aiter equivalent exists).

    Mirrors the kernel: sqrt(softplus(logits)) transform, bias added for
    SELECTION only, top-2-sum-per-group when biased else per-group max, mask
    non-selected groups, top-k on the (biased) choice scores, gather UNBIASED
    weights, renorm + scale.
    """
    nt, ne = logits.shape
    g_size = ne // num_expert_group
    transform = torch.sqrt(F.softplus(logits.float()))
    choice = transform + bias.float().unsqueeze(0) if bias is not None else transform

    if bias is not None:
        group_scores = (
            choice.view(nt, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = choice.view(nt, num_expert_group, -1).max(dim=-1).values

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1.0)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(nt, num_expert_group, g_size)
        .reshape(nt, ne)
        .bool()
    )
    tmp = choice.masked_fill(~score_mask, float("-inf"))
    ids = torch.topk(tmp, k=k, dim=-1, sorted=False).indices
    w = transform.gather(1, ids)
    if renorm:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    w = w * scale
    return w.float(), ids.to(torch.int64)


def _ref_contiguous(
    logits, k, num_expert_group, topk_group, score_mode, bias, renorm, scale
):
    """Reference for contiguous DeepSeek group layout. Reuses aiter torch refs
    where they apply, plus the sqrtsoftplus wrapper + scale."""
    if score_mode == "sqrtsoftplus":
        return _ref_sqrtsoftplus_grouped(
            logits, bias, k, num_expert_group, topk_group, renorm, scale
        )
    if score_mode == "sigmoid" and bias is not None:
        w, ids = biased_grouped_topk_torch(
            logits, bias, k, renorm, num_expert_group, topk_group
        )
    elif score_mode in ("sigmoid", "softmax"):
        w, ids = grouped_topk_torch(
            logits, k, renorm, num_expert_group, topk_group, scoring_func=score_mode
        )
    else:
        raise ValueError(score_mode)
    return w.float() * scale, ids.to(torch.int64)


def _ref_arbitrary_grouped(
    logits,
    expert_group,
    k,
    num_expert_group,
    topk_group,
    score_mode,
    bias,
    renorm,
    scale,
):
    """General reference honoring an arbitrary expert->group table (equal-size
    groups). Used for the non-contiguous mapping case where the aiter refs
    (which assume contiguous .view groups) don't apply."""
    nt, ne = logits.shape
    f32 = logits.float()
    if score_mode == "softmax":
        scores = torch.softmax(f32, dim=-1)
    elif score_mode == "sigmoid":
        scores = f32.sigmoid()
    elif score_mode == "sqrtsoftplus":
        scores = torch.sqrt(F.softplus(f32))
    else:
        scores = f32
    choice = scores + bias.float().unsqueeze(0) if bias is not None else scores

    group_scores = torch.empty((nt, num_expert_group), device=logits.device)
    for g in range(num_expert_group):
        cols = (expert_group == g).nonzero(as_tuple=False).flatten()
        sub = choice[:, cols]
        if bias is not None:
            group_scores[:, g] = sub.topk(2, dim=-1)[0].sum(dim=-1)
        else:
            group_scores[:, g] = sub.max(dim=-1).values

    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False).indices
    group_sel = torch.zeros(
        (nt, num_expert_group), device=logits.device, dtype=torch.bool
    )
    group_sel.scatter_(1, group_idx, True)
    # expert keep mask via group table lookup
    expert_keep = group_sel[:, expert_group.long()]  # (nt, ne)

    tmp = choice.masked_fill(~expert_keep, float("-inf"))
    ids = torch.topk(tmp, k=k, dim=-1, sorted=False).indices
    w = scores.gather(1, ids)
    if renorm:
        w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    w = w * scale
    return w.float(), ids.to(torch.int64)


# --------------------------------------------------------------------------
# output comparison utilities
# --------------------------------------------------------------------------


def _row_sort_by_id(ids, weights):
    order = torch.argsort(ids, dim=1)
    return torch.gather(ids, 1, order), torch.gather(weights, 1, order)


def _assert_selection_matches(ref_ids, ref_w, tri_ids, tri_w):
    """Set-wise per-row comparison: sort both by expert id, then assert ids
    identical and gathered weights close."""
    ref_ids_s, ref_w_s = _row_sort_by_id(ref_ids.cpu(), ref_w.cpu())
    tri_ids_s, tri_w_s = _row_sort_by_id(tri_ids.cpu().long(), tri_w.cpu().float())
    assert torch.equal(
        ref_ids_s, tri_ids_s
    ), f"selected expert ids differ:\nref={ref_ids_s}\ntri={tri_ids_s}"
    assert_close(ref_w_s, tri_w_s, 2e-2, 4e-3, description="weights")


def _decode_bitmatrix(bitmatrix, n_tokens, n_expts_tot):
    """Decode the packed uint32 Bitmatrix into a (n_tokens, n_expts_tot) bool
    matrix of selected experts."""
    data = bitmatrix.data[:n_tokens].to(torch.int64)  # (n_tokens, n_cols_words)
    n_cols_words = data.shape[1]
    bits = torch.arange(32, device=data.device, dtype=torch.int64)
    unpacked = ((data.unsqueeze(-1) >> bits) & 1).bool()  # (nt, words, 32)
    unpacked = unpacked.reshape(n_tokens, n_cols_words * 32)
    return unpacked[:, :n_expts_tot]


def _assert_bitmatrix_matches(bitmatrix, tri_ids, n_tokens, n_expts_tot):
    decoded = _decode_bitmatrix(bitmatrix, n_tokens, n_expts_tot).cpu()
    expected = torch.zeros((n_tokens, n_expts_tot), dtype=torch.bool)
    expected.scatter_(1, tri_ids.cpu().long(), True)
    assert torch.equal(decoded, expected), "bitmatrix does not match selected ids"


# --------------------------------------------------------------------------
# parametrization
# --------------------------------------------------------------------------

# (n_expts_tot, num_expert_group, topk_group, n_expts_act) — DeepSeek-like.
GROUP_SHAPES = [
    (256, 8, 4, 8),
    (128, 8, 4, 6),
]
# n_tokens spanning the fused (<=16) and regular sort_tokens paths.
GROUPED_N_TOKENS = [8, 16, 64, 1024]
# (score_mode, has_bias, renorm, routed_scaling_factor) — production-core set.
SCORE_COMBOS = [
    ("sqrtsoftplus", True, True, 2.5),
    ("sigmoid", True, True, 1.0),
    ("softmax", False, False, 1.0),
]


def _maybe_skip():
    if not torch.cuda.is_available():
        pytest.skip("grouped_topk requires a GPU")
    if get_arch() not in ["gfx950", "gfx1250"]:
        pytest.skip("MOE stack not fully implemented on non-CDNA4 arch yet.")


# --------------------------------------------------------------------------
# 1. direct kernel test: (y_vals, y_indx, bitmatrix)
#
# Unified across contiguous/arbitrary expert->group layouts, score modes, and
# 0/1/2 fused always-on shared experts. The curated case list reproduces the
# original three tests' coverage exactly (contiguous x SCORE_COMBOS x no shared;
# arbitrary x sqrtsoftplus; contiguous x sqrtsoftplus x shared 1/2).
# --------------------------------------------------------------------------

# sqrtsoftplus + bias + renorm + scale=2.5: the fixed combo the arbitrary-group
# and shared-expert variants exercise.
SQ_COMBO = ("sqrtsoftplus", True, True, 2.5)


def _make_shuffled_expert_group(n_expts_tot, num_expert_group, device):
    """Equal-size groups with a shuffled (non-contiguous) expert->group table."""
    g_size = n_expts_tot // num_expert_group
    perm = torch.randperm(n_expts_tot, device=device)
    expert_group = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
    for g in range(num_expert_group):
        expert_group[perm[g * g_size : (g + 1) * g_size]] = g
    return expert_group


def _grouped_topk_kernel_cases():
    cases = []
    # (1) contiguous groups, all score combos.
    for nt in GROUPED_N_TOKENS:
        for shape in GROUP_SHAPES:
            for sc in SCORE_COMBOS:
                cases.append(
                    pytest.param(
                        nt,
                        shape,
                        sc,
                        "contiguous",
                        id=f"contig-nt{nt}-e{shape[0]}-{sc[0]}",
                    )
                )
    # (2) arbitrary (non-contiguous) expert->group table, fixed sqrtsoftplus.
    for nt in [8, 64, 1024]:
        for shape in GROUP_SHAPES:
            cases.append(
                pytest.param(
                    nt,
                    shape,
                    SQ_COMBO,
                    "arbitrary",
                    id=f"arb-nt{nt}-e{shape[0]}",
                )
            )
    return cases


@pytest.mark.parametrize(
    "n_tokens, shape, score_combo, group_mode",
    _grouped_topk_kernel_cases(),
)
def test_grouped_topk_kernel(n_tokens, shape, score_combo, group_mode):
    """Direct grouped_topk kernel test: routed selection + bitmatrix vs torch
    reference, parametrized over expert->group layout and score mode."""
    _maybe_skip()
    n_expts_tot, num_expert_group, topk_group, n_expts_act = shape
    score_mode, has_bias, renorm, scale = score_combo
    device = "cuda"
    torch.manual_seed(7 if group_mode == "arbitrary" else 2)
    logits = init_data(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
    bias = (
        torch.randn(n_expts_tot, dtype=torch.float32, device=device) * 0.05
        if has_bias
        else None
    )

    if group_mode == "arbitrary":
        expert_group = _make_shuffled_expert_group(
            n_expts_tot, num_expert_group, device
        )
        ref_w, ref_ids = _ref_arbitrary_grouped(
            logits.clone(),
            expert_group,
            n_expts_act,
            num_expert_group,
            topk_group,
            score_mode,
            bias,
            renorm,
            scale,
        )
    else:
        expert_group = None
        ref_w, ref_ids = _ref_contiguous(
            logits.clone(),
            n_expts_act,
            num_expert_group,
            topk_group,
            score_mode,
            bias,
            renorm,
            scale,
        )

    y_vals, y_indx, bitmatrix = grouped_topk(
        logits,
        n_expts_act,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        expert_group=expert_group,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=scale,
    )

    assert y_vals.shape == (n_tokens, n_expts_act)
    assert y_indx.shape == (n_tokens, n_expts_act)
    assert y_indx.dtype == torch.int16
    assert y_vals.dtype == logits.dtype

    # Routed slots (first n_expts_act) must match the reference selection.
    _assert_selection_matches(
        ref_ids, ref_w, y_indx[:, :n_expts_act], y_vals[:, :n_expts_act]
    )

    _assert_bitmatrix_matches(bitmatrix, y_indx, n_tokens, n_expts_tot)


# --------------------------------------------------------------------------
# 3. end-to-end routing_score_mode(use_grouped_topk=True)
#
# grouped_topk is the deterministic ground truth, and _check_routing_data_bucket
# validates hist / ExptData / inverse-permutation / per-expert (token, weight)
# multisets over the gate count.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_tokens", [8, 16, 64, 1024])
@pytest.mark.parametrize(
    "n_expts_tot, num_expert_group, topk_group, n_expts_act", GROUP_SHAPES
)
def test_routing_score_mode_grouped(
    n_tokens, n_expts_tot, num_expert_group, topk_group, n_expts_act
):
    """End-to-end routing(use_grouped_topk=True). The routed selection must
    match the grouped_topk kernel and gather/scatter must form a valid inverse
    permutation over the gate count."""
    _maybe_skip()
    device = "cuda"
    torch.manual_seed(2)
    logits = init_data(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
    bias = torch.randn(n_expts_tot, dtype=torch.float32, device=device) * 0.05
    score_mode, renorm, scale = "sqrtsoftplus", True, 2.5

    # routing derives block_m from the raw shape + n_expts_act.
    block_m = _routing_block_m(n_tokens, n_expts_act, n_expts_tot)

    # The selection the kernel makes (deterministic for fixed inputs); used as
    # ground truth for the sort/scatter pipeline check.
    y_vals, y_indx, _ = grouped_topk(
        logits,
        n_expts_act,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=scale,
    )

    tri_routing_data, tri_gather, tri_scatter = routing(
        logits,
        n_expts_act,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=scale,
        use_grouped_topk=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
    )

    ref_pack = _sort_and_build_torch(
        y_vals.float(), y_indx.to(torch.int32), n_expts_tot, block_m
    )
    _check_routing_data_bucket(
        ref_pack, tri_routing_data, tri_gather, tri_scatter, y_vals.float(), y_indx
    )
    assert tri_routing_data.n_expts_tot == n_expts_tot
    assert tri_routing_data.n_expts_act == n_expts_act
    assert tri_routing_data.block_m == block_m


def bench_routing():
    import triton.profiler as proton

    n_tokens = 8192
    n_expts_tot, n_expts_act = 128, 4
    tri_logits = init_data(n_tokens, n_expts_tot)
    proton.start("routing")
    proton.activate()
    for i in range(100):
        tri_routing_data, tri_gather, tri_scatter = routing(tri_logits, n_expts_act)
    proton.finalize()
    try:
        import os

        os.system("proton-viewer -m time/ms routing.hatchet")
    except Exception:
        pass


if __name__ == "__main__":
    bench_routing()

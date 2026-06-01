"""Unit tests for ATOM's single-fused Triton grouped-top-k routing kernel
(``atom.model_ops.grouped_topk_triton.grouped_topk``).

Structured after ``test_moe_routing.py``:
  * Reference uses aiter's torch grouped-topk
    (``biased_grouped_topk_torch`` / ``grouped_topk_torch``) for the standard
    contiguous DeepSeek group layout, plus a thin wrapper for the
    ``sqrtsoftplus`` score mode and the ``routed_scaling_factor`` scale that
    the aiter refs don't apply.
  * ``(y_vals, y_indx)`` are compared per-row set-wise (sorted by expert id),
    robust to the kernel returning experts in descending-score order.
  * The emitted ``Bitmatrix`` is decoded and checked against the selected
    expert set.
  * End-to-end ``routing_a8w4(use_grouped_topk=True)`` is validated through the
    sort_tokens / ExptData pipeline with a bucket-multiset check.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.topk import biased_grouped_topk_torch, grouped_topk_torch
from aiter.ops.triton.moe.moe_routing.routing import (
    routing_a8w4,
    compute_expt_data_torch,
)

# grouped_topk lives in ATOM; skip the whole module if ATOM isn't importable
# in this environment (e.g. aiter-only CI).
atom_grouped_topk = pytest.importorskip(
    "atom.model_ops.grouped_topk_triton"
).grouped_topk


# --------------------------------------------------------------------------
# comparison helpers (copied from test_moe_routing.py for self-containment)
# --------------------------------------------------------------------------


def assert_equal(ref, tri):
    if isinstance(ref, torch.Tensor):
        assert ((ref.cpu().numpy() - tri.cpu().numpy()) ** 2).sum() == 0
    else:
        assert ref == tri


def assert_close(ref, tri, maxtol=None, rmstol=None, description="--", verbose=True):
    if maxtol is None:
        maxtol = 2e-2
    if rmstol is None:
        rmstol = 4e-3
    ref = ref.to(torch.float32).detach()
    tri = tri.to(torch.float32).detach()
    assert (
        ref.shape == tri.shape
    ), f"Tensors must have same size {ref.shape=} {tri.shape=}"

    inf_mask_ref = torch.isinf(ref)
    inf_mask_tri = torch.isinf(tri)
    assert torch.equal(
        inf_mask_ref, inf_mask_tri
    ), "Tensor must have same infinite elements"
    refn = torch.where(inf_mask_ref, 0, ref)
    trin = torch.where(inf_mask_tri, 0, tri)

    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(refn)) + eps)
    refn *= multiplier
    trin *= multiplier

    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps
    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    if verbose:
        print(f"{description} max rel err = {max_err} (thr {maxtol})")
        print(f"{description} rms rel err = {rms_err} (thr {rmstol})")

    assert max_err <= maxtol
    assert rms_err <= rmstol


def init_data(n_tokens, n_expts_tot, dtype=torch.float32, device="cuda"):
    return torch.randn((n_tokens, n_expts_tot), dtype=dtype, device=device)


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
    logits, expert_group, k, num_expert_group, topk_group, score_mode, bias, renorm, scale
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
    group_sel = torch.zeros((nt, num_expert_group), device=logits.device, dtype=torch.bool)
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
    assert torch.equal(ref_ids_s, tri_ids_s), (
        f"selected expert ids differ:\nref={ref_ids_s}\ntri={tri_ids_s}"
    )
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
# end-to-end routing helpers (mirror of test_moe_routing.py, compacted)
# --------------------------------------------------------------------------


def _sort_and_build_torch(expt_scal, expt_indx, n_expts_tot, block_m):
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


def _check_routing_data_bucket(
    ref_pack, tri_routing_data, tri_gather, tri_scatter, topk_weights, topk_ids
):
    ref_hist, _, _, _, ref_expt_data = ref_pack
    assert_equal(ref_hist, tri_routing_data.expt_hist)
    assert_equal(ref_expt_data.hist, tri_routing_data.expt_data.hist)
    assert_equal(ref_expt_data.token_offs_raw, tri_routing_data.expt_data.token_offs_raw)
    assert_equal(ref_expt_data.token_offs_pad, tri_routing_data.expt_data.token_offs_pad)
    assert_equal(ref_expt_data.block_pid_map, tri_routing_data.expt_data.block_pid_map)

    n_tokens, n_expts_act = topk_ids.shape
    n_gates = n_tokens * n_expts_act
    n_expts_tot = ref_hist.numel()

    iota = torch.arange(n_gates, dtype=torch.int32, device=tri_gather.device)
    assert torch.equal(tri_scatter[tri_gather.long()], iota), "scatter[gather[j]] != j"

    flat_ids = topk_ids.reshape(-1).cpu().tolist()
    flat_w = topk_weights.reshape(-1).float().cpu().tolist()
    src = tri_gather.cpu().tolist()
    scal = tri_routing_data.gate_scal.float().cpu().tolist()
    cum = torch.cumsum(ref_hist, dim=0).cpu().tolist()

    ground = {e: [] for e in range(n_expts_tot)}
    for i, e in enumerate(flat_ids):
        ground[e].append((i // n_expts_act, flat_w[i]))
    for e in ground:
        ground[e].sort()

    got = {e: [] for e in range(n_expts_tot)}
    e = 0
    for j in range(n_gates):
        while e < n_expts_tot and j >= cum[e]:
            e += 1
        assert flat_ids[src[j]] == e, f"bucket-invariant violated at pos {j}"
        got[e].append((src[j] // n_expts_act, scal[j]))
    for e in got:
        got[e].sort()

    for e in range(n_expts_tot):
        rb, tb = ground[e], got[e]
        assert len(rb) == len(tb), f"expert {e}: ref={len(rb)} test={len(tb)}"
        for (tt_r, w_r), (tt_t, w_t) in zip(rb, tb):
            assert tt_r == tt_t, f"expert {e}: token ref={tt_r} test={tt_t}"
            assert abs(w_r - w_t) <= 1e-6, f"expert {e} token {tt_r}: w {w_r} vs {w_t}"


# --------------------------------------------------------------------------
# parametrization
# --------------------------------------------------------------------------

# (n_expts_tot, num_expert_group, topk_group, n_expts_act) — DeepSeek-like.
GROUP_SHAPES = [
    (256, 8, 4, 8),
    (128, 8, 4, 6),
]
# n_tokens spanning the fused (<=16) and regular sort_tokens paths.
N_TOKENS = [8, 16, 64, 1024]
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
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_tokens", N_TOKENS)
@pytest.mark.parametrize(
    "n_expts_tot, num_expert_group, topk_group, n_expts_act", GROUP_SHAPES
)
@pytest.mark.parametrize("score_mode, has_bias, renorm, scale", SCORE_COMBOS)
def test_grouped_topk_kernel(
    n_tokens,
    n_expts_tot,
    num_expert_group,
    topk_group,
    n_expts_act,
    score_mode,
    has_bias,
    renorm,
    scale,
):
    _maybe_skip()
    device = "cuda"
    torch.manual_seed(2)
    logits = init_data(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
    bias = (
        torch.randn(n_expts_tot, dtype=torch.float32, device=device) * 0.05
        if has_bias
        else None
    )

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
    y_vals, y_indx, bitmatrix = atom_grouped_topk(
        logits,
        n_expts_act,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=scale,
    )

    assert y_vals.shape == (n_tokens, n_expts_act)
    assert y_indx.shape == (n_tokens, n_expts_act)
    assert y_indx.dtype == torch.int16
    assert y_vals.dtype == logits.dtype

    _assert_selection_matches(ref_ids, ref_w, y_indx, y_vals)
    _assert_bitmatrix_matches(bitmatrix, y_indx, n_tokens, n_expts_tot)


# --------------------------------------------------------------------------
# 2. arbitrary (non-contiguous) expert->group mapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_tokens", [8, 64, 1024])
@pytest.mark.parametrize(
    "n_expts_tot, num_expert_group, topk_group, n_expts_act", GROUP_SHAPES
)
def test_grouped_topk_arbitrary_group(
    n_tokens, n_expts_tot, num_expert_group, topk_group, n_expts_act
):
    _maybe_skip()
    device = "cuda"
    torch.manual_seed(7)
    logits = init_data(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
    bias = torch.randn(n_expts_tot, dtype=torch.float32, device=device) * 0.05

    # Equal-size groups but a shuffled (non-contiguous) expert->group table.
    g_size = n_expts_tot // num_expert_group
    perm = torch.randperm(n_expts_tot, device=device)
    expert_group = torch.empty(n_expts_tot, dtype=torch.int32, device=device)
    for g in range(num_expert_group):
        expert_group[perm[g * g_size : (g + 1) * g_size]] = g

    score_mode, renorm, scale = "sqrtsoftplus", True, 2.5
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
    y_vals, y_indx, bitmatrix = atom_grouped_topk(
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

    _assert_selection_matches(ref_ids, ref_w, y_indx, y_vals)
    _assert_bitmatrix_matches(bitmatrix, y_indx, n_tokens, n_expts_tot)


# --------------------------------------------------------------------------
# 3. end-to-end routing_a8w4(use_grouped_topk=True)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_tokens", [8, 16, 64, 1024])
@pytest.mark.parametrize(
    "n_expts_tot, num_expert_group, topk_group, n_expts_act", GROUP_SHAPES
)
@pytest.mark.parametrize("block_m", [16, 32])
def test_routing_a8w4_grouped(
    n_tokens, n_expts_tot, num_expert_group, topk_group, n_expts_act, block_m
):
    _maybe_skip()
    device = "cuda"
    torch.manual_seed(2)
    logits = init_data(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
    bias = torch.randn(n_expts_tot, dtype=torch.float32, device=device) * 0.05
    score_mode, renorm, scale = "sqrtsoftplus", True, 2.5

    # The selection the kernel makes (deterministic for fixed inputs); used as
    # ground truth for the sort/scatter pipeline check.
    y_vals, y_indx, _ = atom_grouped_topk(
        logits,
        n_expts_act,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=scale,
    )

    tri_routing_data, tri_gather, tri_scatter = routing_a8w4(
        logits,
        n_expts_act,
        block_m,
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


# --------------------------------------------------------------------------
# 4. fused shared experts (DeepSeek-R1/V3 always-on shared expert)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n_tokens", [8, 64, 1024])
@pytest.mark.parametrize(
    "n_expts_tot, num_expert_group, topk_group, n_expts_act", GROUP_SHAPES
)
@pytest.mark.parametrize("n_shared", [1, 2])
def test_grouped_topk_shared_expert(
    n_tokens, n_expts_tot, num_expert_group, topk_group, n_expts_act, n_shared
):
    """The kernel appends `n_shared` always-on shared experts (id n_expts_tot+i,
    weight 1.0) AFTER the routed renorm. The routed portion must still match the
    reference, and the shared columns + bitmatrix must reflect the append."""
    _maybe_skip()
    device = "cuda"
    torch.manual_seed(2)
    logits = init_data(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
    bias = torch.randn(n_expts_tot, dtype=torch.float32, device=device) * 0.05
    score_mode, renorm, scale = "sqrtsoftplus", True, 2.5

    ref_w, ref_ids = _ref_contiguous(
        logits.clone(), n_expts_act, num_expert_group, topk_group,
        score_mode, bias, renorm, scale,
    )
    y_vals, y_indx, bitmatrix = atom_grouped_topk(
        logits,
        n_expts_act,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=scale,
        num_fused_shared_experts=n_shared,
        shared_experts_score=1.0,
    )

    assert y_vals.shape == (n_tokens, n_expts_act + n_shared)
    assert y_indx.shape == (n_tokens, n_expts_act + n_shared)

    # Routed slots (first n_expts_act) must match the reference selection.
    _assert_selection_matches(
        ref_ids, ref_w, y_indx[:, :n_expts_act], y_vals[:, :n_expts_act]
    )

    # Shared slots: fixed id n_expts_tot+i, weight 1.0, for every token.
    for i in range(n_shared):
        ids_i = y_indx[:, n_expts_act + i].cpu().long()
        w_i = y_vals[:, n_expts_act + i].float().cpu()
        assert torch.all(ids_i == n_expts_tot + i), f"shared id col {i}: {ids_i}"
        assert torch.allclose(w_i, torch.ones(n_tokens)), f"shared weight col {i}"

    # Bitmatrix must contain routed + shared selections over the widened width.
    _assert_bitmatrix_matches(bitmatrix, y_indx, n_tokens, n_expts_tot + n_shared)


@pytest.mark.parametrize("n_tokens", [8, 16, 64, 1024])
@pytest.mark.parametrize(
    "n_expts_tot, num_expert_group, topk_group, n_expts_act", GROUP_SHAPES
)
@pytest.mark.parametrize("block_m", [16, 32])
@pytest.mark.parametrize("n_shared", [1, 2])
def test_routing_a8w4_grouped_shared(
    n_tokens, n_expts_tot, num_expert_group, topk_group, n_expts_act, block_m, n_shared
):
    """End-to-end routing_a8w4 with fused shared experts: histogram must include
    a full shared bucket (n_tokens) per shared expert and the gather/scatter must
    form a valid inverse permutation over the widened gate count."""
    _maybe_skip()
    device = "cuda"
    torch.manual_seed(2)
    logits = init_data(n_tokens, n_expts_tot, device=device, dtype=torch.float32)
    bias = torch.randn(n_expts_tot, dtype=torch.float32, device=device) * 0.05
    score_mode, renorm, scale = "sqrtsoftplus", True, 2.5

    rd, gather, scatter = routing_a8w4(
        logits,
        n_expts_act,
        block_m,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=scale,
        use_grouped_topk=True,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        num_fused_shared_experts=n_shared,
    )

    assert rd.n_expts_tot == n_expts_tot + n_shared
    assert rd.n_expts_act == n_expts_act + n_shared

    # Every token is routed to each shared expert exactly once.
    for i in range(n_shared):
        assert rd.expt_hist[n_expts_tot + i].item() == n_tokens
    assert rd.expt_hist.sum().item() == n_tokens * (n_expts_act + n_shared)

    n_gates = n_tokens * (n_expts_act + n_shared)
    iota = torch.arange(n_gates, dtype=torch.int32, device=gather.device)
    assert torch.equal(scatter[gather.long()], iota), "scatter[gather[j]] != j"

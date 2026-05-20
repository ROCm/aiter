import pytest
import torch
import torch.nn.functional as F
from aiter.ops.triton.moe.moe_routing.routing import (
    routing,
    routing_a8w4,
    routing_a8w4_from_hash,
    routing_a8w4_from_topk,
    routing_torch,
    compute_expt_data_torch,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch


def assert_equal(ref, tri):
    if isinstance(ref, torch.Tensor):
        assert torch.all(ref == tri)
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
        tri_logits, n_expts_act, sm_first
    )

    def _assert_indx_equal(ref, tri):
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
# Reference implementations for routing_a8w4* paths
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


def routing_a8w4_torch(
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


def routing_a8w4_from_hash_torch(
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


def routing_a8w4_from_topk_torch(topk_weights, topk_ids, n_expts_tot, block_m):
    return _sort_and_build_torch(
        topk_weights,
        topk_ids.to(torch.int16),
        n_expts_tot,
        block_m,
    )


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
    iota = torch.arange(n_gates, dtype=torch.int32, device=tri_gather.device)
    assert torch.equal(tri_scatter[tri_gather.long()], iota), "scatter[gather[j]] != j"

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
# routing_a8w4
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
@pytest.mark.parametrize("block_m", [16, 32])
def test_routing_a8w4(
    n_tokens,
    n_expts_tot,
    n_expts_act,
    score_mode,
    has_bias,
    renorm,
    routed_scaling_factor,
    block_m,
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

    ref_pack = routing_a8w4_torch(
        logits.clone(),
        n_expts_act,
        block_m,
        score_mode=score_mode,
        bias=bias,
        renorm=renorm,
        routed_scaling_factor=routed_scaling_factor,
    )
    tri_routing_data, tri_gather, tri_scatter = routing_a8w4(
        logits,
        n_expts_act,
        block_m,
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
# routing_a8w4_from_hash
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
def test_routing_a8w4_from_hash(
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

    ref_pack = routing_a8w4_from_hash_torch(
        router_logits.clone(),
        tid2eid,
        input_ids,
        n_expts_act,
        block_m,
        score_mode="sqrtsoftplus",
        renorm=renorm,
        routed_scaling_factor=routed_scaling_factor,
    )
    tri_routing_data, tri_gather, tri_scatter = routing_a8w4_from_hash(
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


# --------------------------
# routing_a8w4_from_topk
# --------------------------


# fused_routing_from_topk requires n_tokens * n_expts_act <= 4096.
@pytest.mark.parametrize(
    "n_tokens, n_expts_tot, n_expts_act",
    [
        (8, 128, 4),
        (64, 128, 4),
        (256, 128, 4),
        (256, 256, 8),
        (512, 128, 8),
    ],
)
@pytest.mark.parametrize("block_m", [16, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_routing_a8w4_from_topk(
    n_tokens,
    n_expts_tot,
    n_expts_act,
    block_m,
    dtype,
):
    if get_arch() not in ["gfx950", "gfx1250"]:
        pytest.skip("MOE stack not fully implemented on non-CDNA4 arch yet.")

    device = "cuda"
    torch.manual_seed(2)
    topk_weights = torch.randn(n_tokens, n_expts_act, dtype=dtype, device=device).abs()
    # Per-row unique expert ids (the natural V4 case).
    topk_ids = torch.stack(
        [
            torch.randperm(n_expts_tot, device=device)[:n_expts_act]
            for _ in range(n_tokens)
        ],
        dim=0,
    ).to(torch.int32)

    ref_pack = routing_a8w4_from_topk_torch(
        topk_weights.clone(),
        topk_ids.clone(),
        n_expts_tot,
        block_m,
    )
    tri_routing_data, tri_gather, tri_scatter = routing_a8w4_from_topk(
        topk_weights,
        topk_ids,
        n_expts_tot,
        block_m,
    )

    _check_routing_data_bucket(
        ref_pack,
        tri_routing_data,
        tri_gather,
        tri_scatter,
        topk_weights,
        topk_ids,
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

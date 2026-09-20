"""herd_fused_topk vs fused_topk: same [M, k] contract, different selection.

herd_fused_topk is the HERD min-unique selector extracted from routing_minunique
without _combined_routing, so fused_moe can consume it like fused_topk.
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.fused_moe import fused_topk, herd_fused_topk
from aiter.ops.triton.moe.moe_routing.minunique import (
    herd_fused_topk as herd_fused_topk_impl,
)
from op_tests.triton_tests.moe.test_moe_routing_herd import (
    HERD_N_TOKENS,
    HERD_SHAPES,
    _decode_per_token,
    _enable_herd,
    _init_logits,
    _minunique_select_torch,
    _ref_per_token,
    _skip_if_unsupported,
)


def _hidden(gating):
    # fused_topk only reads M / device from hidden_states.
    return torch.empty((gating.shape[0], 1), dtype=gating.dtype, device=gating.device)


def _maps_close(ref, got, atol=2e-3, rtol=2e-2):
    assert set(got) == set(ref), "tokens covered differ"
    for t in ref:
        assert set(got[t]) == set(
            ref[t]
        ), f"token {t} expert set: ref={sorted(ref[t])} got={sorted(got[t])}"
        for e, w_ref in ref[t].items():
            w_got = got[t][e]
            assert abs(w_got - w_ref) <= atol + rtol * abs(
                w_ref
            ), f"token {t} expert {e}: weight ref={w_ref} got={w_got}"


def _unique_experts(ids):
    return int(torch.unique(ids.reshape(-1).long()).numel())


def _fused_topk_torch(logits, k, renormalize):
    scores = F.softmax(logits.float(), dim=-1)
    w, ids = scores.topk(k=k, dim=-1, largest=True, sorted=True)
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return w.float(), ids.to(torch.int32)


# ==========================================================================
# 1. herd_fused_topk == torch min-unique (the extracted selector)
# ==========================================================================
@pytest.mark.parametrize("n_tokens", HERD_N_TOKENS)
@pytest.mark.parametrize("n_expts_tot, n_expts_act", HERD_SHAPES)
@pytest.mark.parametrize("sm_first", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
def test_herd_fused_topk_matches_minunique_torch(
    n_tokens, n_expts_tot, n_expts_act, sm_first, renormalize
):
    _skip_if_unsupported()
    logits = _init_logits(n_tokens, n_expts_tot)
    ref_w, ref_ids = _minunique_select_torch(logits.clone(), n_expts_act, sm_first)
    if renormalize and sm_first:
        ref_w = ref_w / ref_w.sum(dim=-1, keepdim=True)

    w, ids = herd_fused_topk(
        _hidden(logits),
        logits,
        n_expts_act,
        renormalize,
        sm_first=sm_first,
    )

    assert w.shape == (n_tokens, n_expts_act) and w.dtype == torch.float32
    assert ids.shape == (n_tokens, n_expts_act) and ids.dtype == torch.int32
    assert int((ids >= 0).all() and (ids < n_expts_tot).all())
    _maps_close(
        _ref_per_token(ref_ids.cpu(), ref_w.cpu()), _ref_per_token(ids.cpu(), w.cpu())
    )


# ==========================================================================
# 2. fused_topk == torch softmax+topk (the baseline this drop-in replaces)
# ==========================================================================
@pytest.mark.parametrize("n_tokens", HERD_N_TOKENS)
@pytest.mark.parametrize("n_expts_tot, n_expts_act", HERD_SHAPES)
@pytest.mark.parametrize("renormalize", [False, True])
def test_fused_topk_matches_softmax_topk(
    n_tokens, n_expts_tot, n_expts_act, renormalize
):
    _skip_if_unsupported()
    logits = _init_logits(n_tokens, n_expts_tot)
    ref_w, ref_ids = _fused_topk_torch(logits, n_expts_act, renormalize)
    w, ids = fused_topk(_hidden(logits), logits, n_expts_act, renormalize)

    assert w.shape[:2] == (n_tokens, n_expts_act) and w.dtype == torch.float32
    assert ids.shape[:2] == (n_tokens, n_expts_act) and ids.dtype == torch.int32
    torch.testing.assert_close(w.float(), ref_w, atol=2e-3, rtol=2e-3)
    for row in range(n_tokens):
        assert set(ids[row].tolist()) == set(
            ref_ids[row].tolist()
        ), f"row {row}: fused_topk set {sorted(ids[row].tolist())} vs torch {sorted(ref_ids[row].tolist())}"


# ==========================================================================
# 3. 对照 fused_topk: same [M, k] contract, HERD still k-per-token, union shrinks
# ==========================================================================
@pytest.mark.parametrize("n_tokens", [32, 64, 128])
@pytest.mark.parametrize("n_expts_tot, n_expts_act", [(128, 4), (256, 8)])
@pytest.mark.parametrize("renormalize", [False, True])
def test_herd_fused_topk_vs_fused_topk(n_tokens, n_expts_tot, n_expts_act, renormalize):
    _skip_if_unsupported()
    logits = _init_logits(n_tokens, n_expts_tot)
    hidden = _hidden(logits)

    stock_w, stock_ids = fused_topk(hidden, logits, n_expts_act, renormalize)
    herd_w, herd_ids = herd_fused_topk(
        hidden, logits, n_expts_act, renormalize, sm_first=True
    )

    assert herd_w.shape == stock_w[:n_tokens].shape
    assert herd_ids.shape == stock_ids[:n_tokens].shape
    assert herd_w.dtype == stock_w.dtype and herd_ids.dtype == stock_ids.dtype
    assert herd_w.shape == (n_tokens, n_expts_act)

    # still exactly k distinct experts per token
    for row in range(n_tokens):
        assert len(set(herd_ids[row].tolist())) == n_expts_act
        assert len(set(stock_ids[row].tolist())) == n_expts_act

    uniq_stock = _unique_experts(stock_ids[:n_tokens])
    uniq_herd = _unique_experts(herd_ids)
    assert (
        uniq_herd <= uniq_stock
    ), f"HERD did not shrink the union: {uniq_herd} > {uniq_stock}"
    # decode-sized 128/256 experts: sharing is guaranteed -> selection must change
    stock_hist = torch.bincount(
        stock_ids[:n_tokens].reshape(-1).long(), minlength=n_expts_tot
    )
    herd_hist = torch.bincount(herd_ids.reshape(-1).long(), minlength=n_expts_tot)
    assert not torch.equal(stock_hist, herd_hist), "HERD did not engage vs fused_topk"


# ==========================================================================
# 4. extracted selector == routing() selection (no sort in the [M, k] view)
# ==========================================================================
@pytest.mark.parametrize("n_tokens", HERD_N_TOKENS)
@pytest.mark.parametrize("n_expts_tot, n_expts_act", HERD_SHAPES)
@pytest.mark.parametrize("sm_first", [False, True])
def test_herd_fused_topk_matches_routing_selection(
    monkeypatch, n_tokens, n_expts_tot, n_expts_act, sm_first
):
    _skip_if_unsupported()
    from aiter.ops.triton.moe.moe_routing.routing import routing

    logits = _init_logits(n_tokens, n_expts_tot)
    w, ids = herd_fused_topk(
        _hidden(logits), logits, n_expts_act, False, sm_first=sm_first
    )

    _enable_herd(monkeypatch)
    rd, gather, _scatter = routing(logits, n_expts_act, sm_first=sm_first)
    got = _decode_per_token(gather, rd.gate_scal, rd.expt_hist, n_expts_act)
    _maps_close(_ref_per_token(ids.cpu(), w.cpu()), got)


def test_herd_fused_topk_fused_moe_reexport_is_impl():
    _skip_if_unsupported()
    assert herd_fused_topk is not herd_fused_topk_impl
    logits = _init_logits(16, 128)
    hidden = _hidden(logits)
    a = herd_fused_topk(hidden, logits, 4, False, sm_first=True)
    b = herd_fused_topk_impl(hidden, logits, 4, False, sm_first=True)
    _maps_close(
        _ref_per_token(a[1].cpu(), a[0].cpu()), _ref_per_token(b[1].cpu(), b[0].cpu())
    )

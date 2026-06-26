"""End-to-end unit tests for HERD min-unique routing (AITER_TRITON_USE_HERD).

HERD is the env-gated flat-top-k path: take top-(k+1) per token, drop the
least-batch-popular candidate (tiebreak min value, then min expert id), keep k.
These tests drive routing() from logits to all MoE inputs (covering topk(k+1)+pop,
_keepk_sort0, and _combined_routing) in both env states; the flags are read at
import, so we toggle them by monkeypatching the module globals.
"""

import pytest
import torch
import triton

import aiter.ops.triton.moe.moe_routing.routing as routing_mod
from aiter.ops.triton.moe.moe_routing.routing import (
    routing,
    routing_torch,
    compute_expt_data_torch,
)
from aiter.ops.triton.moe.moe_routing.topk import topk
from aiter.ops.triton.utils._triton.arch_info import get_arch


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _skip_if_unsupported():
    if not torch.cuda.is_available():
        pytest.skip("HERD routing requires a GPU")
    if get_arch() not in ["gfx950", "gfx1250"]:
        pytest.skip("MOE stack not fully implemented on non-CDNA4 arch yet.")


def _init_logits(n_tokens, n_expts_tot, device="cuda", seed=2):
    torch.manual_seed(seed)
    return torch.randn((n_tokens, n_expts_tot), dtype=torch.float32, device=device)


def _block_m(n_tokens, n_expts_act, n_expts_tot):
    """block_m heuristic used by routing() internally."""
    tokens_per_expt = max(1, (n_tokens * n_expts_act) // n_expts_tot)
    return max(16, min(triton.next_power_of_2(tokens_per_expt), 128))


def _enable_herd(monkeypatch, min_m=0, max_m=1 << 30):
    monkeypatch.setattr(routing_mod, "_USE_HERD", True)
    monkeypatch.setattr(routing_mod, "_HERD_MIN_M", min_m)
    monkeypatch.setattr(routing_mod, "_HERD_MAX_M", max_m)


def _disable_herd(monkeypatch):
    monkeypatch.setattr(routing_mod, "_USE_HERD", False)


def _assert_int_equal(ref, tri, what):
    ref = ref.cpu()
    tri = tri.cpu().to(ref.dtype)
    assert torch.equal(ref, tri), f"{what} mismatch:\n ref={ref}\n tri={tri}"


# --------------------------------------------------------------------------
# torch reference: the HERD min-unique selection
# --------------------------------------------------------------------------
def _minunique_select_torch(logits, k, sm_first):
    """top-(k+1) -> drop least-batch-popular (tiebreak min value, then min expert
    id) -> keep k. Returns (weights[M,k] float32, ids[M,k] int64), experts
    ascending per row to match streaming_topk's emission order."""
    M, E = logits.shape
    f32 = logits.float()
    kp1 = k + 1
    scores = torch.softmax(f32, dim=1) if sm_first else f32

    cand_val, cand_idx = torch.topk(scores, kp1, dim=1)  # value-descending
    # sort candidates by expert id so the col tiebreak matches the kernel order.
    cand_idx, order = torch.sort(cand_idx, dim=1)
    cand_val = torch.gather(cand_val, 1, order)

    # batch popularity over the (k+1) selection.
    pop = torch.zeros(E, dtype=torch.int64, device=logits.device)
    pop.scatter_add_(
        0,
        cand_idx.reshape(-1),
        torch.ones(M * kp1, dtype=torch.int64, device=logits.device),
    )
    cand_pop = pop[cand_idx]  # [M, k+1]

    # drop = lexicographic argmin over (popularity, value, column).
    BIG = float("inf")
    is_mp = cand_pop == cand_pop.min(dim=1, keepdim=True).values
    val_m = torch.where(is_mp, cand_val, torch.full_like(cand_val, BIG))
    is_mv = is_mp & (val_m == val_m.min(dim=1, keepdim=True).values)
    cols = torch.arange(kp1, device=logits.device).expand(M, kp1)
    col_m = torch.where(is_mv, cols, torch.full_like(cols, 1 << 30))
    drop_col = col_m.argmin(dim=1)  # [M]

    keep = torch.ones(M, kp1, dtype=torch.bool, device=logits.device)
    keep.scatter_(1, drop_col.unsqueeze(1), False)
    kept_val = cand_val[keep].reshape(M, k)
    kept_idx = cand_idx[keep].reshape(M, k)  # already ascending

    # sm_first: raw probs, no renorm (kernel apply_softmax=False); else softmax kept.
    w = kept_val if sm_first else torch.softmax(kept_val, dim=1)
    return w.float(), kept_idx.to(torch.int64)


def _ref_per_token(ids, weights):
    """{token: {expert: weight}} from a [M,k] selection."""
    out = {}
    for t in range(ids.shape[0]):
        out[t] = {
            int(e): float(wv) for e, wv in zip(ids[t].tolist(), weights[t].tolist())
        }
    return out


def _decode_per_token(tri_gather, gate_scal, hist, k):
    """{token: {expert: weight}} reconstructed from the triton output. gather is
    in expert-sorted order, so hist's cumulative bounds map position j to its
    expert and gather[j] // k is the source token."""
    n_gates = int(hist.sum().item())
    cum = torch.cumsum(hist, 0).tolist()
    src = tri_gather[:n_gates].long().cpu().tolist()
    scal = gate_scal[:n_gates].float().cpu().tolist()
    out = {}
    e = 0
    for j in range(n_gates):
        while e < len(cum) and j >= cum[e]:
            e += 1
        tok = src[j] // k
        out.setdefault(tok, {})[e] = scal[j]
    return out


def _check_minunique(ref_w, ref_ids, rd, gather, scatter, n_expts_tot, k, block_m):
    """Validate routing() output against the torch min-unique reference."""
    M = ref_ids.shape[0]
    n_gates = M * k

    # exactly k experts kept per token (the over-count that crashed capture breaks this).
    assert int(rd.expt_hist.sum().item()) == n_gates, "sum(hist) != M*k"

    # histogram + ExptData match the reference selection exactly.
    ref_hist = torch.histc(
        ref_ids.float().reshape(-1), bins=n_expts_tot, min=0, max=n_expts_tot - 1
    ).to(torch.int32)
    _assert_int_equal(ref_hist, rd.expt_hist, "expt_hist")
    ref_ed = compute_expt_data_torch(ref_hist, n_expts_tot, n_gates, block_m)
    _assert_int_equal(ref_ed.hist, rd.expt_data.hist, "expt_data.hist")
    _assert_int_equal(
        ref_ed.token_offs_raw, rd.expt_data.token_offs_raw, "token_offs_raw"
    )
    _assert_int_equal(
        ref_ed.token_offs_pad, rd.expt_data.token_offs_pad, "token_offs_pad"
    )
    _assert_int_equal(ref_ed.block_pid_map, rd.expt_data.block_pid_map, "block_pid_map")

    # gather/scatter form an inverse permutation over the gate count.
    g = gather[:n_gates].long()
    s = scatter.long()
    iota = torch.arange(n_gates, device=g.device, dtype=torch.int64)
    assert torch.equal(s[g], iota), "scatter[gather[j]] != j"

    # per-token (expert, weight) selection matches the reference.
    ref = _ref_per_token(ref_ids.cpu(), ref_w.cpu())
    got = _decode_per_token(gather, rd.gate_scal, rd.expt_hist, k)
    assert set(got) == set(ref), "tokens covered differ"
    for t in ref:
        assert set(got[t]) == set(
            ref[t]
        ), f"token {t} expert set: ref={sorted(ref[t])} got={sorted(got[t])}"
        for e, w_ref in ref[t].items():
            w_got = got[t][e]
            assert abs(w_got - w_ref) <= 2e-3 + 2e-2 * abs(
                w_ref
            ), f"token {t} expert {e}: weight ref={w_ref} got={w_got}"


def _routing_fields_equal(a_rd, a_g, a_s, b_rd, b_g, b_s):
    """Strict equality of two routing() results (HERD-off / out-of-window == stock)."""
    _assert_int_equal(a_rd.expt_hist, b_rd.expt_hist, "expt_hist")
    _assert_int_equal(a_rd.expt_data.hist, b_rd.expt_data.hist, "expt_data.hist")
    _assert_int_equal(
        a_rd.expt_data.block_pid_map, b_rd.expt_data.block_pid_map, "block_pid_map"
    )
    _assert_int_equal(a_g.to(torch.int32), b_g.to(torch.int32), "gather")
    _assert_int_equal(a_s.to(torch.int32), b_s.to(torch.int32), "scatter")
    assert torch.allclose(
        a_rd.gate_scal.float(), b_rd.gate_scal.float(), atol=1e-5, rtol=0
    ), "gate_scal differs"


# decode-sized shapes (HERD is a small-M, weight-bound feature); (256,8) is the
# k=8 case that pads 8->16.
HERD_SHAPES = [(128, 4), (128, 6), (256, 8)]
HERD_N_TOKENS = [16, 32, 64, 128]


# ==========================================================================
# 1. HERD enabled: end-to-end routing() == torch min-unique reference
# ==========================================================================
@pytest.mark.parametrize("n_tokens", HERD_N_TOKENS)
@pytest.mark.parametrize("n_expts_tot, n_expts_act", HERD_SHAPES)
@pytest.mark.parametrize("sm_first", [False, True])
def test_routing_herd_enabled(
    monkeypatch, n_tokens, n_expts_tot, n_expts_act, sm_first
):
    _skip_if_unsupported()
    logits = _init_logits(n_tokens, n_expts_tot)
    block_m = _block_m(n_tokens, n_expts_act, n_expts_tot)

    ref_w, ref_ids = _minunique_select_torch(logits.clone(), n_expts_act, sm_first)

    _enable_herd(monkeypatch)
    rd, gather, scatter = routing(logits, n_expts_act, sm_first=sm_first)

    _check_minunique(
        ref_w, ref_ids, rd, gather, scatter, n_expts_tot, n_expts_act, block_m
    )
    assert rd.n_expts_tot == n_expts_tot
    assert rd.n_expts_act == n_expts_act
    assert rd.block_m == block_m


# ==========================================================================
# 2. HERD disabled: end-to-end routing() == stock flat-top-k (routing_torch)
# ==========================================================================
@pytest.mark.parametrize("n_tokens", HERD_N_TOKENS)
@pytest.mark.parametrize("n_expts_tot, n_expts_act", HERD_SHAPES)
@pytest.mark.parametrize("sm_first", [False, True])
def test_routing_herd_disabled_matches_stock(
    monkeypatch, n_tokens, n_expts_tot, n_expts_act, sm_first
):
    _skip_if_unsupported()
    logits = _init_logits(n_tokens, n_expts_tot)

    ref_rd, ref_g, ref_s = routing_torch(logits.clone(), n_expts_act, sm_first)

    _disable_herd(monkeypatch)
    tri_rd, tri_g, tri_s = routing(logits, n_expts_act, sm_first=sm_first)

    n_gates = n_tokens * n_expts_act
    _assert_int_equal(ref_rd.expt_hist, tri_rd.expt_hist, "expt_hist")
    _assert_int_equal(ref_rd.expt_data.hist, tri_rd.expt_data.hist, "expt_data.hist")
    _assert_int_equal(
        ref_rd.expt_data.block_pid_map, tri_rd.expt_data.block_pid_map, "block_pid_map"
    )
    # stock ref is unpadded; triton pads the gather/scatter tail with -1.
    _assert_int_equal(ref_g, tri_g[:n_gates].to(torch.int32), "gather")
    _assert_int_equal(ref_s, tri_s[:n_gates].to(torch.int32), "scatter")
    assert torch.all(tri_g[n_gates:] == -1) and torch.all(tri_s[n_gates:] == -1)
    assert torch.allclose(
        ref_rd.gate_scal.float(),
        tri_rd.gate_scal[:n_gates].float(),
        atol=2e-2,
        rtol=2e-2,
    )


# ==========================================================================
# 3. HERD changes routing: the unique-expert union shrinks, k kept per token.
# ==========================================================================
@pytest.mark.parametrize("n_tokens", [32, 64, 128])
def test_herd_shrinks_expert_union(monkeypatch, n_tokens):
    _skip_if_unsupported()
    n_expts_tot, k = 128, 4
    logits = _init_logits(n_tokens, n_expts_tot)

    _disable_herd(monkeypatch)
    rd_off, _, _ = routing(logits, k, sm_first=False)

    _enable_herd(monkeypatch)
    rd_on, _, _ = routing(logits, k, sm_first=False)

    n_gates = n_tokens * k
    assert int(rd_off.expt_hist.sum()) == n_gates
    assert int(rd_on.expt_hist.sum()) == n_gates  # still exactly k per token

    uniq_off = int((rd_off.expt_hist > 0).sum())
    uniq_on = int((rd_on.expt_hist > 0).sum())
    assert uniq_on <= uniq_off, f"HERD did not shrink the union: {uniq_on} > {uniq_off}"
    # with 128 experts at decode sizes, sharing is guaranteed -> routing must change.
    assert not torch.equal(rd_off.expt_hist, rd_on.expt_hist), "HERD did not engage"


# ==========================================================================
# 4. Gating window: only [MIN_M, MAX_M] engages; outside it falls through to stock.
# ==========================================================================
@pytest.mark.parametrize("n_tokens, engaged", [(8, False), (64, True), (256, False)])
def test_herd_gating_window(monkeypatch, n_tokens, engaged):
    _skip_if_unsupported()
    n_expts_tot, k = 128, 4
    logits = _init_logits(n_tokens, n_expts_tot)
    block_m = _block_m(n_tokens, k, n_expts_tot)

    _enable_herd(monkeypatch, min_m=16, max_m=128)
    tri_rd, tri_g, tri_s = routing(logits, k, sm_first=False)
    assert int(tri_rd.expt_hist.sum()) == n_tokens * k

    if engaged:
        ref_w, ref_ids = _minunique_select_torch(logits.clone(), k, False)
        _check_minunique(ref_w, ref_ids, tri_rd, tri_g, tri_s, n_expts_tot, k, block_m)
    else:
        # below MIN_M / above MAX_M -> identical to fully-disabled HERD.
        _disable_herd(monkeypatch)
        st_rd, st_g, st_s = routing(logits, k, sm_first=False)
        _routing_fields_equal(tri_rd, tri_g, tri_s, st_rd, st_g, st_s)


# ==========================================================================
# 5. Fusion-1 in isolation: topk(k+1, pop_out=) candidates + popularity histogram.
# ==========================================================================
@pytest.mark.parametrize("n_tokens", [16, 64, 128])
@pytest.mark.parametrize("n_expts_tot, n_expts_act", [(128, 4), (256, 8)])
def test_topk_pop_out(n_tokens, n_expts_tot, n_expts_act):
    _skip_if_unsupported()
    logits = _init_logits(n_tokens, n_expts_tot)
    kp1 = n_expts_act + 1

    pop = torch.zeros(n_expts_tot, dtype=torch.int32, device=logits.device)
    expt_scal, expt_indx, _ = topk(
        logits, kp1, apply_softmax=False, HIST_BLOCK_M=32, pop_out=pop
    )
    assert expt_indx.shape == (n_tokens, kp1)

    # selection set matches torch top-(k+1).
    _, ref_idx = torch.topk(logits.float(), kp1, dim=1)
    assert [set(r.tolist()) for r in expt_indx.long()] == [
        set(r.tolist()) for r in ref_idx
    ]

    # popularity == bincount of the emitted selection.
    ref_pop = torch.bincount(expt_indx.reshape(-1).long(), minlength=n_expts_tot).to(
        torch.int32
    )
    _assert_int_equal(ref_pop, pop, "popularity")
    assert int(pop.sum()) == n_tokens * kp1

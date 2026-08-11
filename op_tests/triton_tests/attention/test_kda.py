# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.attention.kda import fused_recurrent_kda
from aiter.ops.triton.utils._triton.arch_info import get_arch

arch = get_arch()

pytestmark = pytest.mark.skipif(
    arch != "gfx1250", reason=f"KDA gluon decode is gfx1250 only, got {arch}"
)

DEVICE = "cuda"
RATIO = 0.005
POISON = 1e30


def err_ratio(ref, tri):
    """fla.utils.get_err_ratio: RMS(ref - tri) / RMS(ref)."""
    ref, tri = ref.detach().float(), tri.detach().float()
    err = (ref - tri).flatten().square().mean().sqrt()
    base = ref.flatten().square().mean().sqrt()
    return (err / (base + 1e-8)).item()


def assert_close(name, ref, tri, ratio=RATIO):
    assert not torch.isnan(tri).any(), f"{name}: NaN in kernel output"
    assert not torch.isinf(tri).any(), f"{name}: Inf in kernel output"
    r = err_ratio(ref, tri)
    assert r < ratio, f"{name}: err ratio {r:.6f} >= {ratio}"


def naive_kda_gate(g, A_log, dt_bias=None):
    """fla.ops.kda.gate.naive_kda_gate: -exp(A_log) * softplus(g + dt_bias)."""
    H = g.shape[-2]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    return -A_log.view(H, 1).float().exp() * F.softplus(g)


def naive_kda_lowerbound_gate(g, A_log, dt_bias=None, lower_bound=-5.0):
    """fla.ops.kda.gate.naive_kda_lowerbound_gate."""
    H = g.shape[-2]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.view(H, -1)
    return lower_bound * torch.sigmoid(A_log.view(H, 1).float().exp() * g)


def naive_recurrent_kda(q, k, v, g, beta, scale=None, initial_state=None):
    """fla.ops.kda.naive.naive_recurrent_kda. State is [B, HV, K, V].

    `g` is log-space. Headwise beta (``[B, T, HV, V]``) scales the error term
    rather than k -- equivalent for the scalar case, since the update is an
    outer product.
    """
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[-1]
    G = HV // H
    if scale is None:
        scale = K**-0.5

    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q.repeat_interleave(G, dim=2) * scale
    k = k.repeat_interleave(G, dim=2)

    S = q.new_zeros(B, HV, K, V)
    if initial_state is not None:
        S = S + initial_state.float()
    o = torch.zeros_like(v)
    for i in range(T):
        k_i, g_i, b_i = k[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()  # decay
        err = v[:, i] - (k_i[..., None] * S).sum(-2)  # v - S^T k
        err = err * (b_i if b_i.ndim == 3 else b_i.unsqueeze(-1))
        S = S + k_i[..., None] * err[..., None, :]  # k (x) err
        o[:, i] = (q[:, i][..., None] * S).sum(-2)  # S^T q
    return o, S


def make_qkv(B, T, H, HV, D, dtype, seed=42):
    torch.manual_seed(seed)
    q = torch.rand(B, T, H, D, dtype=dtype, device=DEVICE)
    k = torch.rand(B, T, H, D, dtype=dtype, device=DEVICE)
    v = torch.rand(B, T, HV, D, dtype=dtype, device=DEVICE)
    return q, k, v


def to_vk(h_kv):
    """[*, K, V] initial state -> the [*, V, K] layout our kernel requires."""
    return h_kv.transpose(-1, -2).contiguous()


def make_pool(B, T, H, D, seed=42):
    """Oversized slot pool with a random disjoint assignment; every slot the
    kernel must not touch is poisoned so a stray access blows up visibly.
    Returns (pool as [slots, H, K, V], indices [B, T], untouched mask)."""
    torch.manual_seed(seed)
    max_slots = B * T * 3
    pool_kv = torch.randn(max_slots, H, D, D, dtype=torch.float32, device=DEVICE)
    perm = torch.randperm(max_slots, device=DEVICE)[: B * T]
    untouched = torch.ones(max_slots, dtype=torch.bool, device=DEVICE)
    untouched[perm] = False
    pool_kv[untouched] = POISON
    return pool_kv, perm.int().view(B, T), untouched


def run(q, k, v, g, beta, h0_kv, **kw):
    """Call the kernel with a V-first state, return (o, final_state as [*, K, V])."""
    state = to_vk(h0_kv)
    o, state = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        **kw,
    )
    return o, state.transpose(-1, -2)


def _seq_lens(kw, n_seq):
    cu = kw.get("cu_seqlens")
    if cu is not None:
        return (cu[1:] - cu[:-1]).tolist()
    return [kw["q"].shape[1]] * n_seq


def _replay_into_snapshots(pool, indices, lens, BV=32):
    """Rebuild the per-token snapshot states a cached-update launch elided:
    replay each sequence's updates over its base state and write the results
    where snapshot mode would have put them (torch replay is ULP-close, not
    bitwise -- callers compare with assert_close). Records are replicated per
    i_v tile in chunks of 2*K + BV; a/k are read from tile 0, err assembled
    across tiles."""
    K, V = pool.shape[-1], pool.shape[-2]
    H = pool.shape[1]
    ch = 2 * K + BV
    for n, ln in enumerate(lens):
        if ln == 0:
            continue
        S = pool[int(indices[n, 0])].clone()
        for t in range(1, ln):
            rec = pool[int(indices[n, t])].reshape(H, V * K)
            a, kr = rec[:, :K], rec[:, K : 2 * K]
            er = torch.cat(
                [rec[:, i * ch + 2 * K : i * ch + 2 * K + BV] for i in range(V // BV)],
                dim=1,
            )
            S = S * a[:, None, :] + er[:, :, None] * kr[:, None, :]
            pool[int(indices[n, t])] = S


@pytest.fixture(autouse=True, params=["snapshots", "cache_state_updates"])
def kda_store_mode(request, monkeypatch):
    """Run the whole suite twice: once as-is, once routing every compatible
    kernel call through the cached-update store path. Non-paged calls are
    transparently rewritten onto a slot pool; the snapshots each test expects
    are then reconstructed from the updates, so every existing assertion also
    validates the record contents."""
    if request.param == "snapshots":
        yield request.param
        return

    real = fused_recurrent_kda

    def wrapped(**kw):
        state = kw.get("initial_state")
        idx = kw.get("ssm_state_indices")
        if (
            "cache_state_updates" in kw
            or kw.get("num_accepted_tokens") is not None
            or kw.get("use_tdm_store")
            or kw.get("state_v_first") is False
            or not kw.get("inplace_final_state", True)
            or state is None
            or (idx is not None and idx.ndim != 2)
        ):
            return real(**kw)

        if idx is not None:  # already paged: just flip the store path
            o, pool = real(**kw, cache_state_updates=True)
            _replay_into_snapshots(
                pool, idx, _seq_lens(kw, idx.shape[0]), kw.get("BV", 32)
            )
            return o, pool

        # non-paged: emulate on a pool where slot n is the caller's state row
        # (the seed and base slot), with scratch slots for the token updates
        N = state.shape[0]
        lens = _seq_lens(kw, N)
        maxlen = max(lens)
        if len(lens) != N or maxlen > 64:
            # malformed calls stay on the real path so guards still fire;
            # training-length tests stay off the B*T scratch pool
            return real(**kw)
        scratch = N * max(0, maxlen - 1)
        pool = torch.cat([state, state.new_empty(scratch, *state.shape[1:])])
        indices = torch.empty(N, max(1, maxlen), dtype=torch.int32, device=DEVICE)
        for n in range(N):
            indices[n, 0] = n
            for t in range(1, max(1, maxlen)):
                indices[n, t] = N + n * (maxlen - 1) + t - 1
        o, pool = real(
            **dict(kw, initial_state=pool),
            ssm_state_indices=indices,
            cache_state_updates=True,
        )
        _replay_into_snapshots(pool, indices, lens, kw.get("BV", 32))
        for n, ln in enumerate(lens):
            if ln:
                state[n] = pool[int(indices[n, ln - 1])]
        return o, state

    monkeypatch.setattr(sys.modules[__name__], "fused_recurrent_kda", wrapped)
    yield request.param


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "scale", "use_qk_l2norm_in_kernel"),
    [
        pytest.param(*t, id="B{}-T{}-H{}-HV{}-D{}-scale{}-qk_l2{}".format(*t))
        for t in [
            (1, 1, 1, 1, 64, 1.0, False),
            (4, 1, 4, 4, 128, 0.1, True),
            (2, 3, 3, 3, 128, 1.0, False),
            (7, 4, 8, 8, 128, 0.5, True),
            (2, 2, 2, 4, 64, 1.0, True),
        ]
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fused_recurrent(B, T, H, HV, D, scale, use_qk_l2norm_in_kernel, dtype):
    q, k, v = make_qkv(B, T, H, HV, D, dtype)
    g = F.logsigmoid(torch.randn(B, T, HV, D, dtype=torch.float32, device=DEVICE))
    beta = torch.randn(B, T, HV, dtype=torch.float32, device=DEVICE).sigmoid()
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=DEVICE)

    ref, ref_ht = naive_recurrent_kda(
        F.normalize(q.clone(), p=2, dim=-1),
        F.normalize(k.clone(), p=2, dim=-1),
        v,
        g,
        beta,
        scale=scale,
        initial_state=h0,
    )
    tri, tri_ht = run(
        q if use_qk_l2norm_in_kernel else F.normalize(q.clone(), p=2, dim=-1),
        k if use_qk_l2norm_in_kernel else F.normalize(k.clone(), p=2, dim=-1),
        v,
        g,
        beta,
        h0.clone(),
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    assert_close("o", ref, tri)
    assert_close("ht", ref_ht, tri_ht)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "scale"),
    [
        pytest.param(*t, id="B{}-T{}-H{}-HV{}-D{}-scale{}".format(*t))
        for t in [
            (1, 64, 1, 1, 64, 1.0),
            (2, 512, 3, 3, 128, 1.0),
            (3, 1000, 4, 4, 128, 0.1),
            (4, 1024, 4, 4, 128, 0.1),
            (2, 1024, 2, 8, 128, 0.1),
        ]
    ],
)
def test_fused_recurrent_long_sequence(B, T, H, HV, D, scale):
    """Training-length sequences, as fla validates its recurrent oracle at."""
    q, k, v = make_qkv(B, T, H, HV, D, torch.float32)
    q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    g = F.logsigmoid(torch.randn(B, T, HV, D, dtype=torch.float32, device=DEVICE))
    beta = torch.randn(B, T, HV, dtype=torch.float32, device=DEVICE).sigmoid()
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=DEVICE)

    ref, ref_ht = naive_recurrent_kda(q, k, v, g, beta, scale=scale, initial_state=h0)
    tri, tri_ht = run(q, k, v, g, beta, h0.clone(), scale=scale)
    assert_close("o", ref, tri)
    assert_close("ht", ref_ht, tri_ht)


@pytest.mark.parametrize("allow_neg_eigval", [False, True])
@pytest.mark.parametrize("T", [1, 4])
def test_fused_recurrent_beta_sigmoid_in_kernel(allow_neg_eigval, T):
    """Raw beta + in-kernel sigmoid matches pre-sigmoid beta."""
    B, H, D = 2, 4, 128
    q, k, v = make_qkv(B, T, H, H, D, torch.float32)
    q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE))
    beta_post = torch.randn(B, T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    beta_raw = torch.logit(beta_post.clamp(1e-4, 1 - 1e-4))
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=DEVICE)

    ref, ref_ht = run(
        q, k, v, g, beta_post * (2 if allow_neg_eigval else 1), h0.clone()
    )
    tri, tri_ht = run(
        q,
        k,
        v,
        g,
        beta_raw,
        h0.clone(),
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=allow_neg_eigval,
    )
    assert_close("o", ref, tri)
    assert_close("ht", ref_ht, tri_ht)


@pytest.mark.parametrize(
    ("B", "T", "H", "HV", "D", "has_dt_bias", "safe_gate"),
    [
        pytest.param(*t, id="B{}-T{}-H{}-HV{}-D{}-bias{}-safe{}".format(*t))
        for t in [
            (1, 1, 1, 1, 64, False, False),
            (2, 1, 2, 2, 128, True, False),
            (2, 4, 2, 4, 128, True, True),
            (4, 3, 4, 4, 128, False, True),
        ]
    ],
)
def test_fused_recurrent_gate_in_kernel(B, T, H, HV, D, has_dt_bias, safe_gate):
    """In-kernel gate chain matches the torch gate applied beforehand."""
    q, k, v = make_qkv(B, T, H, HV, D, torch.float32)
    g_raw = torch.randn(B, T, HV, D, dtype=torch.float32, device=DEVICE)
    beta = torch.rand(B, T, HV, dtype=torch.float32, device=DEVICE).sigmoid()
    A_log = torch.log(
        torch.empty(HV, dtype=torch.float32, device=DEVICE).uniform_(1, 16)
    )
    dt_bias = (
        torch.randn(HV * D, dtype=torch.float32, device=DEVICE) if has_dt_bias else None
    )
    h0 = torch.randn(B, HV, D, D, dtype=torch.float32, device=DEVICE)

    lower_bound = -5.0 if safe_gate else None
    gate_fn = naive_kda_lowerbound_gate if safe_gate else naive_kda_gate
    g_ref = gate_fn(g_raw, A_log, dt_bias)

    ref, ref_ht = run(q, k, v, g_ref, beta, h0.clone(), use_qk_l2norm_in_kernel=True)
    tri, tri_ht = run(
        q,
        k,
        v,
        g_raw,
        beta,
        h0.clone(),
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
    )
    assert_close("o", ref, tri, 0.002)
    assert_close("ht", ref_ht, tri_ht, 0.002)


@pytest.mark.parametrize("lens", [[1, 1, 1], [3, 1, 2], [2, 0, 3], [1, 4, 1, 2, 3]])
def test_fused_recurrent_varlen(lens):
    H, D = 4, 128
    N, total_T = len(lens), sum(lens)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), device=DEVICE).long()

    q, k, v = make_qkv(1, total_T, H, H, D, torch.float32)
    q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    g = F.logsigmoid(torch.randn(1, total_T, H, D, dtype=torch.float32, device=DEVICE))
    beta = torch.rand(1, total_T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=DEVICE)

    state = to_vk(h0)
    tri, state = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
    )
    tri_ht = state.transpose(-1, -2)

    for n, ln in enumerate(lens):
        b, e = int(cu[n]), int(cu[n + 1])
        if ln == 0:
            # empty sequence: the kernel returns early, state must be untouched
            assert torch.equal(tri_ht[n], h0[n]), f"seq {n}: empty sequence written"
            continue
        ref, ref_ht = naive_recurrent_kda(
            q[:, b:e],
            k[:, b:e],
            v[:, b:e],
            g[:, b:e],
            beta[:, b:e],
            initial_state=h0[n : n + 1],
        )
        assert_close(f"o[{n}]", ref, tri[:, b:e])
        assert_close(f"ht[{n}]", ref_ht[0], tri_ht[n])


@pytest.mark.parametrize("B, T", [(2, 1), (7, 1), (2, 4)])
@pytest.mark.parametrize("spec", [False, True])
def test_fused_recurrent_vllm_decode(B, T, spec):
    """Continuous batching with paged state, poisoning untouched slots."""
    H, D = 8, 128
    pool_kv, indices, untouched = make_pool(B, T, H, D)

    total_T = B * T
    q, k, v = make_qkv(1, total_T, H, H, D, torch.float32)
    q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    g = F.logsigmoid(torch.randn(1, total_T, H, D, dtype=torch.float32, device=DEVICE))
    beta = torch.rand(1, total_T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    cu = torch.arange(0, total_T + 1, step=T, device=DEVICE).long()
    num_accepted = (
        torch.arange(1, B + 1, device=DEVICE, dtype=torch.int32).clamp(max=T)
        if spec
        else None
    )

    state = to_vk(pool_kv)
    tri, state = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=num_accepted,
    )
    tri_pool = state.transpose(-1, -2)

    for n in range(B):
        seed = int(indices[n, int(num_accepted[n]) - 1]) if spec else int(indices[n, 0])
        b, e = n * T, (n + 1) * T
        ref, ref_ht = naive_recurrent_kda(
            q[:, b:e],
            k[:, b:e],
            v[:, b:e],
            g[:, b:e],
            beta[:, b:e],
            initial_state=pool_kv[seed : seed + 1],
        )
        assert_close(f"o[{n}]", ref, tri[:, b:e])
        # only the last token's snapshot is checked against the closed recurrence;
        # earlier slots hold intermediate states by construction
        assert_close(f"ht[{n}]", ref_ht[0], tri_pool[int(indices[n, T - 1])])

    assert torch.equal(
        tri_pool[untouched], pool_kv[untouched]
    ), "kernel wrote outside its slots"


def _spec_inputs(total_T, H, HV, D, seed, gate=False, beta_headwise=False, dtype=None):
    """Raw spec-decode inputs; l2norm always in-kernel, gate chain optional."""
    dtype = dtype or torch.float32
    torch.manual_seed(seed)
    q = torch.rand(1, total_T, H, D, dtype=dtype, device=DEVICE)
    k = torch.rand(1, total_T, H, D, dtype=dtype, device=DEVICE)
    v = torch.rand(1, total_T, HV, D, dtype=dtype, device=DEVICE)
    g = torch.randn(1, total_T, HV, D, dtype=torch.float32, device=DEVICE)
    bshape = (1, total_T, HV, D) if beta_headwise else (1, total_T, HV)
    beta = torch.rand(bshape, dtype=torch.float32, device=DEVICE)
    kw = {"q": q, "k": k, "v": v, "use_qk_l2norm_in_kernel": True}
    if gate:
        A_log = torch.log(
            torch.empty(HV, dtype=torch.float32, device=DEVICE).uniform_(1, 16)
        )
        kw.update(
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=torch.randn(HV * D, dtype=torch.float32, device=DEVICE),
            lower_bound=-5.0,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
        )
    else:
        kw.update(g=F.logsigmoid(g), beta=beta.sigmoid())
    return kw


def _cached_vs_snapshot(B, T, H, HV, D, accepted_rounds, **kw):
    """Run the same launch chain in snapshot and cached-update mode: every launch
    output and the token-0 base state must be bitwise identical, and poisoned
    slots must stay untouched. Round 0 has no verifier seed; each later round
    seeds from its `accepted_rounds` entry against the previous round's pool."""
    gate = kw.pop("gate", False)
    beta_headwise = kw.pop("beta_headwise", False)
    dtype = kw.pop("dtype", None)
    total_T = B * T
    pool_kv, indices, untouched = make_pool(B, T, HV, D)
    base = to_vk(pool_kv)
    cu = torch.arange(0, total_T + 1, step=T, device=DEVICE).long()

    results = {}
    for cached in (False, True):
        pool = base.clone()
        outs = []
        for r, acc in enumerate([None] + list(accepted_rounds)):
            inp = _spec_inputs(
                total_T,
                H,
                HV,
                D,
                seed=17 + r,
                gate=gate,
                beta_headwise=beta_headwise,
                dtype=dtype,
            )
            o, _ = fused_recurrent_kda(
                **inp,
                initial_state=pool,
                output_final_state=True,
                cu_seqlens=cu,
                ssm_state_indices=indices,
                num_accepted_tokens=acc,
                cache_state_updates=cached,
                **kw,
            )
            outs.append(o)
        results[cached] = (outs, pool)

    snap_outs, snap_pool = results[False]
    cached_outs, cached_pool = results[True]
    for r, (o_s, o_d) in enumerate(zip(snap_outs, cached_outs)):
        assert torch.equal(o_s, o_d), f"round {r}: cached-update output diverged"
    s0 = indices[:, 0].long()
    assert torch.equal(snap_pool[s0], cached_pool[s0]), "base state diverged"
    assert torch.equal(
        cached_pool[untouched], base[untouched]
    ), "kernel wrote outside its slots"


@pytest.mark.parametrize("T", [1, 4, 8])
@pytest.mark.parametrize("pattern", ["one", "all", "perseq"])
def test_cache_state_updates_matches_snapshots(T, pattern):
    """Spec-decode two-launch round trip: replaying the (a, k, err) updates must
    reconstruct the seed state bitwise identically to loading the full snapshot,
    for every acceptance count the verifier can produce."""
    B = 4
    acc = {
        "one": torch.ones(B, device=DEVICE, dtype=torch.int32),
        "all": torch.full((B,), T, device=DEVICE, dtype=torch.int32),
        "perseq": torch.arange(1, B + 1, device=DEVICE, dtype=torch.int32).clamp(max=T),
    }[pattern]
    _cached_vs_snapshot(B, T, 4, 4, 128, [acc])


def test_cache_state_updates_multi_round():
    """Four chained verifier rounds with random per-sequence acceptance: updates
    are overwritten in place every round and must never drift from snapshots."""
    B, T = 3, 6
    torch.manual_seed(123)
    rounds = [
        torch.randint(1, T + 1, (B,), device=DEVICE, dtype=torch.int32)
        for _ in range(4)
    ]
    _cached_vs_snapshot(B, T, 4, 4, 128, rounds)


@pytest.mark.parametrize(
    ("gate", "beta_headwise", "dtype", "H", "HV"),
    [
        pytest.param(True, False, torch.bfloat16, 4, 4, id="gate-bf16"),
        pytest.param(True, True, torch.bfloat16, 4, 4, id="gate-headwise-bf16"),
        pytest.param(False, True, torch.float32, 4, 4, id="headwise-fp32"),
        pytest.param(True, False, torch.float32, 2, 4, id="gate-grouped-HV2H"),
    ],
)
def test_cache_state_updates_variants(gate, beta_headwise, dtype, H, HV):
    """Cached state updates across the K3 math flags: in-kernel gate + dt_bias +
    lower bound, headwise beta, bf16 inputs, and grouped heads (HV != H)."""
    acc = torch.tensor([1, 2, 4, 3], device=DEVICE, dtype=torch.int32)
    _cached_vs_snapshot(
        4, 4, H, HV, 128, [acc], gate=gate, beta_headwise=beta_headwise, dtype=dtype
    )


@pytest.mark.parametrize(
    ("BV", "num_warps", "SK", "num_buffers", "use_tdm_load", "use_tdm_fused_load"),
    [
        (32, 4, 32, 1, False, False),
        (32, 2, 16, 2, False, False),
        (64, 2, 32, 2, False, False),
        (32, 4, 32, 2, True, False),
        (32, 4, 32, 2, False, True),
        (32, 2, 16, 2, False, True),
        (64, 2, 32, 2, False, True),
    ],
)
def test_cache_state_updates_tiling(
    BV, num_warps, SK, num_buffers, use_tdm_load, use_tdm_fused_load
):
    """Update loads/stores are layout-parameterized: every tiling and load path
    must reconstruct bitwise identically to snapshots under the same tiling."""
    acc = torch.tensor([2, 4, 1, 3], device=DEVICE, dtype=torch.int32)
    _cached_vs_snapshot(
        4,
        4,
        4,
        4,
        128,
        [acc],
        BV=BV,
        num_warps=num_warps,
        SK=SK,
        num_buffers=num_buffers,
        use_tdm_load=use_tdm_load,
        use_tdm_fused_load=use_tdm_fused_load,
    )


@pytest.mark.parametrize("BV, num_warps", [(32, 4), (32, 2), (64, 2), (128, 4)])
def test_cache_state_updates_k_first(BV, num_warps):
    """The [K, V] layout mirrors the replay's broadcast axes; the update offsets
    themselves are layout-neutral. Must still be bitwise-exact vs snapshots."""
    acc = torch.tensor([1, 4, 2, 3], device=DEVICE, dtype=torch.int32)
    _cached_vs_snapshot(
        4, 4, 4, 4, 128, [acc], BV=BV, num_warps=num_warps, state_v_first=False
    )


def test_cache_state_updates_vs_reference():
    """Two cached-update launches checked against the pure-torch recurrence: seq n's
    second launch must continue from the state after its accepted[n] tokens."""
    B, T, H, D = 4, 6, 4, 128
    total_T = B * T
    pool_kv, indices, untouched = make_pool(B, T, H, D)
    pool = to_vk(pool_kv)
    cu = torch.arange(0, total_T + 1, step=T, device=DEVICE).long()
    acc = torch.arange(1, B + 1, device=DEVICE, dtype=torch.int32).clamp(max=T)

    def mk(seed):
        q, k, v = make_qkv(1, total_T, H, H, D, torch.float32, seed=seed)
        q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
        g = F.logsigmoid(
            torch.randn(1, total_T, H, D, dtype=torch.float32, device=DEVICE)
        )
        beta = torch.rand(1, total_T, H, dtype=torch.float32, device=DEVICE).sigmoid()
        return {"q": q, "k": k, "v": v, "g": g, "beta": beta}

    in1, in2 = mk(3), mk(5)
    o1, _ = fused_recurrent_kda(
        **in1,
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        cache_state_updates=True,
    )
    o2, _ = fused_recurrent_kda(
        **in2,
        initial_state=pool,
        output_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_accepted_tokens=acc,
        cache_state_updates=True,
    )

    for n in range(B):
        a, (b, e) = int(acc[n]), (n * T, (n + 1) * T)
        h0 = pool_kv[int(indices[n, 0])][None]
        sl1 = {k_: v_[:, b:e] for k_, v_ in in1.items()}
        ref1, _ = naive_recurrent_kda(**sl1, initial_state=h0)
        assert_close(f"o1[{n}]", ref1, o1[:, b:e])
        sl1a = {k_: v_[:, b : b + a] for k_, v_ in in1.items()}
        _, ht_a = naive_recurrent_kda(**sl1a, initial_state=h0)
        sl2 = {k_: v_[:, b:e] for k_, v_ in in2.items()}
        ref2, _ = naive_recurrent_kda(**sl2, initial_state=ht_a)
        assert_close(f"o2[{n}]", ref2, o2[:, b:e])

    base = to_vk(pool_kv)
    assert torch.equal(pool[untouched], base[untouched]), "wrote outside slots"


def test_cache_state_updates_varlen():
    """Unequal sequence lengths: per-sequence record counts and acceptance."""
    lens, H, D = [3, 1, 4], 4, 128
    N, total_T, maxlen = len(lens), sum(lens), max(lens)
    cu = torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), device=DEVICE).long()

    torch.manual_seed(11)
    max_slots = N * maxlen * 3
    pool_kv = torch.randn(max_slots, H, D, D, dtype=torch.float32, device=DEVICE)
    perm = torch.randperm(max_slots, device=DEVICE)[: N * maxlen]
    untouched = torch.ones(max_slots, dtype=torch.bool, device=DEVICE)
    untouched[perm] = False
    pool_kv[untouched] = POISON
    indices = perm.int().view(N, maxlen)
    base = to_vk(pool_kv)
    acc = torch.tensor([2, 1, 4], device=DEVICE, dtype=torch.int32)

    results = {}
    for cached in (False, True):
        pool = base.clone()
        outs = []
        for r, a in enumerate([None, acc]):
            inp = _spec_inputs(total_T, H, H, D, seed=29 + r)
            o, _ = fused_recurrent_kda(
                **inp,
                initial_state=pool,
                output_final_state=True,
                cu_seqlens=cu,
                ssm_state_indices=indices,
                num_accepted_tokens=a,
                cache_state_updates=cached,
            )
            outs.append(o)
        results[cached] = (outs, pool)

    for r in range(2):
        assert torch.equal(
            results[False][0][r], results[True][0][r]
        ), f"round {r}: output diverged"
    s0 = indices[:, 0].long()
    assert torch.equal(results[False][1][s0], results[True][1][s0])
    assert torch.equal(results[True][1][untouched], base[untouched])


def test_cache_state_updates_guards():
    """The wrapper must reject configurations the record layout cannot support."""
    B, T, H, D = 2, 2, 2, 128
    inp = _spec_inputs(B * T, H, H, D, seed=1)
    cu = torch.arange(0, B * T + 1, step=T, device=DEVICE).long()
    state = torch.randn(B * T, H, D, D, dtype=torch.float32, device=DEVICE)
    indices = torch.arange(B * T, device=DEVICE, dtype=torch.int32).view(B, T)
    shared = dict(inp, initial_state=state, cu_seqlens=cu, cache_state_updates=True)

    with pytest.raises(AssertionError, match="ssm_state_indices"):
        fused_recurrent_kda(
            **dict(shared, initial_state=state[:B])  # non-paged: one slot per seq
        )
    with pytest.raises(AssertionError, match="inplace_final_state"):
        fused_recurrent_kda(
            **shared,
            ssm_state_indices=indices,
            inplace_final_state=False,
            state_out=torch.empty_like(state),
        )
    with pytest.raises(AssertionError, match="tdm_store"):
        fused_recurrent_kda(**shared, ssm_state_indices=indices, use_tdm_store=True)


@pytest.mark.parametrize("BV, num_warps", [(32, 1), (64, 2), (128, 4), (64, 1)])
def test_tiling_equivalence(BV, num_warps):
    """BV / num_warps must not change results."""
    B, T, H, D = 2, 4, 4, 128
    q, k, v = make_qkv(B, T, H, H, D, torch.float32)
    q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE))
    beta = torch.rand(B, T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=DEVICE)

    ref, ref_ht = naive_recurrent_kda(q, k, v, g, beta, initial_state=h0)
    tri, tri_ht = run(q, k, v, g, beta, h0.clone(), BV=BV, num_warps=num_warps)
    assert_close("o", ref, tri)
    assert_close("ht", ref_ht, tri_ht)


def test_guards():
    B, T, H, D = 1, 1, 4, 128
    q, k, v = make_qkv(B, T, H, H, D, torch.float32)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE))
    beta = torch.rand(B, T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    state = torch.zeros(B, H, D, D, dtype=torch.float32, device=DEVICE)
    args = {"q": q, "k": k, "v": v, "g": g, "beta": beta, "initial_state": state}

    with pytest.raises(ValueError):  # fla parity check
        fused_recurrent_kda(**args, allow_neg_eigval=True)

    with pytest.raises(AssertionError):  # BV must divide V
        fused_recurrent_kda(**args, BV=48)

    with pytest.raises(AssertionError):  # BV*SK must cover 32*num_warps lanes
        fused_recurrent_kda(**args, BV=32, num_warps=4, SK=1)

    with pytest.raises(AssertionError):  # gate chain needs A_log
        fused_recurrent_kda(**args, use_gate_in_kernel=True)

    with pytest.raises(AssertionError):  # spec decoding needs the slot table
        fused_recurrent_kda(
            **args,
            num_accepted_tokens=torch.ones(B, device=DEVICE, dtype=torch.int32),
        )


def _small(T=3, B=2, H=4, D=128):
    q, k, v = make_qkv(B, T, H, H, D, torch.float32)
    q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE))
    beta = torch.rand(B, T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32, device=DEVICE)
    return q, k, v, g, beta, h0


def test_not_inplace_leaves_initial_state_untouched():
    """inplace_final_state=False must write a distinct ht and never touch h0."""
    q, k, v, g, beta, h0 = _small()
    h0_vk = to_vk(h0)
    before = h0_vk.clone()

    o, ht = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0_vk,
        output_final_state=True,
        inplace_final_state=False,
    )
    assert torch.equal(h0_vk, before), "h0 was modified on the non-inplace path"
    assert ht.data_ptr() != h0_vk.data_ptr(), "ht must be a distinct buffer"

    ref, ref_ht = naive_recurrent_kda(q, k, v, g, beta, initial_state=h0)
    assert_close("o", ref, o)
    assert_close("ht", ref_ht, ht.transpose(-1, -2))


def test_inplace_matches_not_inplace():
    # pinned to snapshot mode: the bit-for-bit contract between the two paths
    # only exists when both actually store full states
    q, k, v, g, beta, h0 = _small()
    _, ht_out = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=to_vk(h0),
        output_final_state=True,
        inplace_final_state=False,
        cache_state_updates=False,
    )
    _, ht_ip = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=to_vk(h0),
        output_final_state=True,
        inplace_final_state=True,
        cache_state_updates=False,
    )
    assert torch.equal(ht_out, ht_ip), "in-place and not must agree bit-for-bit"


def test_no_initial_state():
    """initial_state=None starts the recurrence from zeros (ref: h0 is None)."""
    q, k, v, g, beta, _ = _small()
    o, ht = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=None,
        output_final_state=True,
        inplace_final_state=False,
    )
    ref, ref_ht = naive_recurrent_kda(q, k, v, g, beta, initial_state=None)
    assert_close("o", ref, o)
    assert_close("ht", ref_ht, ht.transpose(-1, -2))


def test_no_final_state():
    """output_final_state=False returns None and writes no state at all."""
    q, k, v, g, beta, h0 = _small()
    h0_vk = to_vk(h0)
    before = h0_vk.clone()
    o, ht = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0_vk,
        output_final_state=False,
        inplace_final_state=False,
    )
    assert ht is None, "expected no final state"
    assert torch.equal(h0_vk, before), "state written despite output_final_state=False"
    ref, _ = naive_recurrent_kda(q, k, v, g, beta, initial_state=h0)
    assert_close("o", ref, o)


def test_out_argument():
    q, k, v, g, beta, h0 = _small()
    dst = torch.empty_like(v)
    o, _ = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=to_vk(h0),
        output_final_state=True,
        out=dst,
    )
    assert o.data_ptr() == dst.data_ptr(), "out was not written in place"
    ref, _ = naive_recurrent_kda(q, k, v, g, beta, initial_state=h0)
    assert_close("o", ref, dst)


@pytest.mark.parametrize("T", [1, 4])
def test_beta_headwise(T):
    """IS_BETA_HEADWISE: beta is [B, T, HV, V] rather than a per-head scalar."""
    B, H, D = 2, 4, 128
    q, k, v, g, _, h0 = _small(T=T, B=B, H=H, D=D)
    beta = torch.rand(B, T, H, D, dtype=torch.float32, device=DEVICE).sigmoid()
    assert beta.ndim == v.ndim

    ref, ref_ht = naive_recurrent_kda(q, k, v, g, beta, initial_state=h0)
    tri, tri_ht = run(q, k, v, g, beta, h0.clone())
    assert_close("o", ref, tri)
    assert_close("ht", ref_ht, tri_ht)


@pytest.mark.parametrize("num_buffers", [1, 2])
@pytest.mark.parametrize("use_tdm_store", [False, True])
@pytest.mark.parametrize("use_tdm_fused_load", [False, True])
def test_num_buffers(num_buffers, use_tdm_store, use_tdm_fused_load):
    """Operand prefetch mode and load/store paths must not change results (paged)."""
    B, T, H, D = 2, 4, 8, 128
    pool_kv, indices, untouched = make_pool(B, T, H, D)
    total_T = B * T
    q, k, v = make_qkv(1, total_T, H, H, D, torch.float32)
    q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    g = F.logsigmoid(torch.randn(1, total_T, H, D, dtype=torch.float32, device=DEVICE))
    beta = torch.rand(1, total_T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    cu = torch.arange(0, total_T + 1, step=T, device=DEVICE).long()

    o, state = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=to_vk(pool_kv),
        output_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        num_buffers=num_buffers,
        use_tdm_store=use_tdm_store,
        use_tdm_fused_load=use_tdm_fused_load,
    )
    pool_out = state.transpose(-1, -2)
    for n in range(B):
        b, e = n * T, (n + 1) * T
        ref, ref_ht = naive_recurrent_kda(
            q[:, b:e],
            k[:, b:e],
            v[:, b:e],
            g[:, b:e],
            beta[:, b:e],
            initial_state=pool_kv[int(indices[n, 0]) : int(indices[n, 0]) + 1],
        )
        assert_close(f"o[{n}]", ref, o[:, b:e])
        assert_close(f"ht[{n}]", ref_ht[0], pool_out[int(indices[n, T - 1])])
    assert torch.equal(pool_out[untouched], pool_kv[untouched]), "wrote outside slots"


def test_paged_state_out():
    """Paged + inplace_final_state=False: snapshots go to state_out by token,
    and the slot pool stays read-only."""
    B, T, H, D = 2, 3, 8, 128
    pool_kv, indices, _ = make_pool(B, T, H, D)
    total_T = B * T
    q, k, v = make_qkv(1, total_T, H, H, D, torch.float32)
    q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    g = F.logsigmoid(torch.randn(1, total_T, H, D, dtype=torch.float32, device=DEVICE))
    beta = torch.rand(1, total_T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    cu = torch.arange(0, total_T + 1, step=T, device=DEVICE).long()

    pool_vk = to_vk(pool_kv)
    before = pool_vk.clone()
    state_out = torch.zeros(total_T, H, D, D, dtype=torch.float32, device=DEVICE)

    o, ht = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=pool_vk,
        output_final_state=True,
        inplace_final_state=False,
        state_out=state_out,
        cu_seqlens=cu,
        ssm_state_indices=indices,
    )
    assert torch.equal(pool_vk, before), "slot pool must be read-only in this mode"
    assert ht.data_ptr() == state_out.data_ptr()

    snaps = state_out.transpose(-1, -2)
    for n in range(B):
        b, e = n * T, (n + 1) * T
        ref, ref_ht = naive_recurrent_kda(
            q[:, b:e],
            k[:, b:e],
            v[:, b:e],
            g[:, b:e],
            beta[:, b:e],
            initial_state=pool_kv[int(indices[n, 0]) : int(indices[n, 0]) + 1],
        )
        assert_close(f"o[{n}]", ref, o[:, b:e])
        assert_close(f"ht[{n}]", ref_ht[0], snaps[e - 1])


@pytest.mark.parametrize("T", [1, 4])
@pytest.mark.parametrize("BV", [32, 128])
def test_state_v_first_matches_k_first(T, BV):
    """The [V, K] and [K, V] state layouts must agree, given transposed states."""
    B, H, D = 2, 4, 128
    q, k, v, g, beta, h0_kv = _small(T=T, B=B, H=H, D=D)

    o_kv, ht_kv = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0_kv.clone(),
        output_final_state=True,
        state_v_first=False,
        BV=BV,
    )
    o_vk, ht_vk = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=to_vk(h0_kv),
        output_final_state=True,
        state_v_first=True,
        BV=BV,
    )
    assert_close("o", o_kv, o_vk, 1e-4)
    assert_close("ht", ht_kv, ht_vk.transpose(-1, -2), 1e-4)


@pytest.mark.parametrize("T", [1, 3])
def test_k_first_against_oracle(T):
    """[K, V] is the reference's own default layout, so no transpose is needed."""
    B, H, D = 2, 4, 128
    q, k, v, g, beta, h0 = _small(T=T, B=B, H=H, D=D)

    ref, ref_ht = naive_recurrent_kda(q, k, v, g, beta, initial_state=h0)
    o, ht = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=h0.clone(),
        output_final_state=True,
        state_v_first=False,
    )
    assert_close("o", ref, o)
    assert_close("ht", ref_ht, ht)


def test_k_first_paged():
    B, T, H, D = 2, 3, 8, 128
    pool_kv, indices, untouched = make_pool(B, T, H, D)
    total_T = B * T
    q, k, v = make_qkv(1, total_T, H, H, D, torch.float32)
    q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    g = F.logsigmoid(torch.randn(1, total_T, H, D, dtype=torch.float32, device=DEVICE))
    beta = torch.rand(1, total_T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    cu = torch.arange(0, total_T + 1, step=T, device=DEVICE).long()

    o, pool_out = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=pool_kv.clone(),
        output_final_state=True,
        state_v_first=False,
        cu_seqlens=cu,
        ssm_state_indices=indices,
    )
    for n in range(B):
        b, e = n * T, (n + 1) * T
        ref, ref_ht = naive_recurrent_kda(
            q[:, b:e],
            k[:, b:e],
            v[:, b:e],
            g[:, b:e],
            beta[:, b:e],
            initial_state=pool_kv[int(indices[n, 0]) : int(indices[n, 0]) + 1],
        )
        assert_close(f"o[{n}]", ref, o[:, b:e])
        assert_close(f"ht[{n}]", ref_ht[0], pool_out[int(indices[n, T - 1])])
    assert torch.equal(pool_out[untouched], pool_kv[untouched]), "wrote outside slots"


def test_reference_guards():
    B, T, H, D = 1, 2, 4, 128
    q, k, v, g, beta, h0 = _small(T=T, B=B, H=H, D=D)
    args = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": to_vk(h0),
    }

    with pytest.warns(DeprecationWarning):
        fused_recurrent_kda(**args, transpose_state_layout=True)
    with pytest.raises(ValueError):
        fused_recurrent_kda(**args, state_v_first=True, transpose_state_layout=True)
    with pytest.raises(TypeError):
        fused_recurrent_kda(**args, bogus_kwarg=1)
    with pytest.raises(ValueError):
        fused_recurrent_kda(**args, allow_neg_eigval=True)

    qb, kb, vb, gb, betab, h0b = _small(T=T, B=2, H=H, D=D)
    with pytest.raises(ValueError):
        fused_recurrent_kda(
            q=qb,
            k=kb,
            v=vb,
            g=gb,
            beta=betab,
            initial_state=to_vk(h0b),
            cu_seqlens=torch.tensor([0, T], device=DEVICE).long(),
        )

    with pytest.raises(ValueError):
        fused_recurrent_kda(
            **args, cu_seqlens=torch.tensor([0, 1, T], device=DEVICE).long()
        )


def test_non_contiguous_inputs():
    B, T, H, D = 2, 3, 4, 128
    q, k, v, g, beta, h0 = _small(T=T, B=B, H=H, D=D)
    ref, ref_ht = naive_recurrent_kda(q, k, v, g, beta, initial_state=h0)

    wide = torch.empty(B, T, H, D * 2, dtype=torch.float32, device=DEVICE)
    wide[..., ::2] = q
    q_nc = wide[..., ::2]
    assert not q_nc.is_contiguous()

    o, ht = run(q_nc, k, v, g, beta, h0.clone())
    assert_close("o", ref, o)
    assert_close("ht", ref_ht, ht)


# (BV, SK, num_warps) -> ROWS = BV*SK // (32*num_warps); SK lanes split the K axis
SK_CONFIGS = [
    (32, 1, 1),
    (32, 2, 2),
    (32, 4, 4),
    (32, 8, 8),
    (32, 32, 4),
    (64, 2, 4),
    (128, 4, 8),
]


@pytest.mark.parametrize("BV, SK, num_warps", SK_CONFIGS)
@pytest.mark.parametrize("state_v_first", [True, False])
def test_sk_equivalence(BV, SK, num_warps, state_v_first):
    """Splitting K across SK lanes must not change results, in either layout."""
    B, T, H, D = 2, 3, 4, 128
    q, k, v, g, beta, h0 = _small(T=T, B=B, H=H, D=D)
    ref, ref_ht = naive_recurrent_kda(q, k, v, g, beta, initial_state=h0)

    state = to_vk(h0) if state_v_first else h0.clone()
    o, ht = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        output_final_state=True,
        state_v_first=state_v_first,
        BV=BV,
        SK=SK,
        num_warps=num_warps,
    )
    assert_close("o", ref, o)
    assert_close("ht", ref_ht, ht.transpose(-1, -2) if state_v_first else ht)


@pytest.mark.parametrize("SK", [2, 4, 8])
def test_sk_paged(SK):
    """SK on the paged snapshot path, where the LDS staging layout also changes."""
    B, T, H, D = 2, 3, 8, 128
    pool_kv, indices, untouched = make_pool(B, T, H, D)
    total_T = B * T
    q, k, v = make_qkv(1, total_T, H, H, D, torch.float32)
    q, k = F.normalize(q, p=2, dim=-1), F.normalize(k, p=2, dim=-1)
    g = F.logsigmoid(torch.randn(1, total_T, H, D, dtype=torch.float32, device=DEVICE))
    beta = torch.rand(1, total_T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    cu = torch.arange(0, total_T + 1, step=T, device=DEVICE).long()

    o, state = fused_recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=to_vk(pool_kv),
        output_final_state=True,
        cu_seqlens=cu,
        ssm_state_indices=indices,
        BV=32,
        SK=SK,
        num_warps=SK,
    )
    pool_out = state.transpose(-1, -2)
    for n in range(B):
        b, e = n * T, (n + 1) * T
        ref, ref_ht = naive_recurrent_kda(
            q[:, b:e],
            k[:, b:e],
            v[:, b:e],
            g[:, b:e],
            beta[:, b:e],
            initial_state=pool_kv[int(indices[n, 0]) : int(indices[n, 0]) + 1],
        )
        assert_close(f"o[{n}]", ref, o[:, b:e])
        assert_close(f"ht[{n}]", ref_ht[0], pool_out[int(indices[n, T - 1])])
    assert torch.equal(pool_out[untouched], pool_kv[untouched]), "wrote outside slots"


def test_sk_guards():
    B, T, H, D = 1, 1, 4, 128
    q, k, v, g, beta, h0 = _small(T=T, B=B, H=H, D=D)
    args = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": to_vk(h0),
    }
    with pytest.raises(AssertionError):  # SK must divide the wave
        fused_recurrent_kda(**args, SK=3)
    with pytest.raises(AssertionError):  # BV*SK must cover 32*num_warps
        fused_recurrent_kda(**args, BV=32, SK=2, num_warps=4)

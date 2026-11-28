# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import hashlib
import json
import os

import pytest
import torch

import aiter.ops.triton.kimi_delta_attn.batch_invariant.kda_bi as kda_bi_mod
from aiter.ops.triton.kimi_delta_attn.batch_invariant.kda_bi import (
    BI_CONTRACT_VERSION,
    kda_bi_fwd,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx950", reason="kda_bi has pinned configs for gfx950 only"
)

H, D, LOWER = 12, 128, -5.0  # Kimi-K3 KDA heads per GPU at TP8, head dim, gate bound
GOLDEN = os.path.join(os.path.dirname(__file__), "kda_bi_golden.json")


def make_inputs(T, seed, h=H, n_seq=1, with_state=False):
    """q, k, v, g, beta, A_log, dt_bias (and an initial state), drawn on the CPU: the CUDA
    RNG differs between torch builds."""
    gen = torch.Generator().manual_seed(seed)
    q, k, v, g = (torch.randn(1, T, h, D, generator=gen).bfloat16() for _ in range(4))
    beta = torch.randn(1, T, h, generator=gen).sigmoid()
    a_log = torch.randn(h, generator=gen) * 0.5
    dt_bias = torch.randn(h * D, generator=gen) * 0.5
    x = tuple(t.cuda() for t in (q, k, v, g, beta, a_log, dt_bias))
    if with_state:
        return x, (torch.randn(n_seq, h, D, D, generator=gen) * 0.1).cuda()
    return x


def _cu(lens):
    return torch.tensor([0] + torch.tensor(lens).cumsum(0).tolist(), dtype=torch.int32)


def _run(x, lens, h0=None, final=False, boundary=False):
    cu = _cu(lens)
    return kda_bi_fwd(
        *x[:7],
        LOWER,
        cu.cuda(),
        cu_seqlens_cpu=cu,
        initial_state=h0,
        output_final_state=final,
        output_boundary_state=boundary,
    )


def _pack(*xs):
    return tuple(torch.cat(ts, 1) if ts[0].dim() >= 3 else ts[0] for ts in zip(*xs))


def _cut(x, a, b):
    return tuple(t[:, a:b] if t.dim() >= 3 else t for t in x)


def _reference(x, lens, h0=None):
    """FP64 token-by-token KDA recurrence: (o [1, T, H, D], final states [N, H, D, D])."""
    q, k, v, g, beta, a_log, dt_bias = (t.double() for t in x)
    h = q.shape[2]
    q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt() * D**-0.5
    k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    gate = LOWER * torch.sigmoid(a_log.exp().view(h, 1) * (g + dt_bias.view(h, D)))
    o = torch.empty_like(v)
    states = []
    off = 0
    for i, n in enumerate(lens):
        S = torch.zeros(h, D, D, dtype=torch.float64, device=q.device)
        if h0 is not None:
            S = h0[i].double().clone()
        for t in range(off, off + n):
            S = S * gate[0, t].exp()[..., None]
            kt = k[0, t]
            S = S + torch.einsum(
                "hk,hv->hkv",
                beta[0, t][:, None] * kt,
                v[0, t] - torch.einsum("hk,hkv->hv", kt, S),
            )
            o[0, t] = torch.einsum("hk,hkv->hv", q[0, t], S)
        states.append(S)
        off += n
    return o, torch.stack(states)


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _digest(t):
    t = t.contiguous()
    view = torch.int16 if t.dtype == torch.bfloat16 else torch.int32
    return hashlib.sha256(t.view(view).cpu().numpy().tobytes()).hexdigest()


GOLDEN_CASES = {
    "T256": ([256], 11),
    "packed": ([150, 90, 1, 64], 12),
    "init_state": ([200, 56], 13),
}


def golden_digests(name):
    lens, seed = GOLDEN_CASES[name]
    x, h0 = make_inputs(sum(lens), seed, n_seq=len(lens), with_state=True)
    o, ht = _run(x, lens, h0=h0 if name == "init_state" else None, final=True)
    return {"o": _digest(o), "final_state": _digest(ht)}


@pytest.mark.parametrize("name", list(GOLDEN_CASES))
def test_golden_bits(name):
    """The output bits are a contract: a change must bump BI_CONTRACT_VERSION and the
    golden file."""
    with open(GOLDEN) as f:
        golden = json.load(f)
    assert golden["contract_version"] == BI_CONTRACT_VERSION
    assert golden_digests(name) == golden["gfx950"][name]


@pytest.mark.parametrize("T,seed", [(160, 0), (256, 2)])
def test_every_prefix_same_bits(T, seed):
    # every prefix length 1..T, packed in one call, gives the full sequence's rows: row p of
    # a chunk depends only on rows 0..p (what a decode step recomputes)
    x = make_inputs(T, seed)
    full, _ = _run(x, [T])
    lens = list(range(1, T + 1))
    out, _ = _run(_pack(*(_cut(x, 0, n) for n in lens)), lens)
    offs = _cu(lens).tolist()
    bad = [
        n for n, a in zip(lens, offs) if not torch.equal(out[0, a : a + n], full[0, :n])
    ]
    assert not bad, f"prefix lengths with other bits: {bad[:20]}"


def test_packing_and_order_same_bits():
    lens = [150, 90, 1, 33, 64]
    xs = [make_inputs(n, 20 + i) for i, n in enumerate(lens)]
    xs = [x[:5] + xs[0][5:] for x in xs]  # one layer: same A_log, dt_bias
    alone = [_run(x, [n], final=True) for x, n in zip(xs, lens)]
    for order in (range(len(lens)), reversed(range(len(lens)))):
        order = list(order)
        o, ht = _run(
            _pack(*(xs[i] for i in order)), [lens[i] for i in order], final=True
        )
        off = 0
        for j, i in enumerate(order):
            assert torch.equal(o[0, off : off + lens[i]], alone[i][0][0])
            assert torch.equal(ht[j], alone[i][1][0])
            off += lens[i]


@pytest.mark.parametrize("split", [64, 128])
def test_chunk_aligned_continuation_same_bits(split):
    T = 200
    x = make_inputs(T, 3)
    full, h_full = _run(x, [T], final=True)
    o1, h1 = _run(_cut(x, 0, split), [split], final=True)
    o2, h2 = _run(_cut(x, split, T), [T - split], h0=h1, final=True)
    assert torch.equal(torch.cat([o1, o2], 1), full)
    assert torch.equal(h2, h_full)


@pytest.mark.parametrize("bv", [16, 32, 64, 128])
def test_state_block_size_same_bits(bv, monkeypatch):
    # the wrapper picks the state pass's V block from the grid size; the bits must not
    # depend on it
    lens = [150, 90, 1, 64]
    x, h0 = make_inputs(sum(lens), 14, n_seq=len(lens), with_state=True)
    ref = _run(x, lens, h0=h0, final=True, boundary=True)
    monkeypatch.setattr(kda_bi_mod, "_state_bv", lambda n: bv)
    out = _run(x, lens, h0=h0, final=True, boundary=True)
    for a, b in zip(out, ref):
        assert torch.equal(a, b)


@pytest.mark.parametrize("fuse", [False, True])
def test_fused_output_same_bits(fuse, monkeypatch):
    # the output is computed in the state pass for wide grids; the bits must not depend on it
    lens = [150, 90, 1, 64, 200]
    x, h0 = make_inputs(sum(lens), 15, n_seq=len(lens), with_state=True)
    monkeypatch.setattr(kda_bi_mod, "_fuse_out", lambda n: not fuse)
    ref = _run(x, lens, h0=h0, final=True, boundary=True)
    monkeypatch.setattr(kda_bi_mod, "_fuse_out", lambda n: fuse)
    out = _run(x, lens, h0=h0, final=True, boundary=True)
    for a, b in zip(out, ref):
        assert torch.equal(a, b)


def test_many_sequences_with_equal_lengths():
    # the state pass runs sequences longest first; ties and empty sequences included
    lens = [64, 1, 0, 128, 64, 65, 1, 128, 0, 63, 64, 200, 1, 65] * 3
    x = make_inputs(sum(lens), 16)
    o, ht = _run(x, lens, final=True)
    offs = _cu(lens).tolist()
    for i, n in enumerate(lens):
        if n == 0:
            assert torch.equal(ht[i], torch.zeros_like(ht[i]))
            continue
        oi, hi = _run(_cut(x, offs[i], offs[i] + n), [n], final=True)
        assert torch.equal(o[:, offs[i] : offs[i] + n], oi)
        assert torch.equal(ht[i], hi[0])


def test_boundary_state():
    # the state after each sequence's last complete chunk; the initial state if there is none
    lens = [200, 64, 30, 0, 128]
    x, h0 = make_inputs(sum(lens), 4, n_seq=len(lens), with_state=True)
    _, ht, hb = _run(x, lens, h0=h0, final=True, boundary=True)
    offs = _cu(lens).tolist()
    for i, n in enumerate(lens):
        a = n // 64 * 64
        if a == 0:
            assert torch.equal(hb[i], h0[i])
        else:
            _, ref = _run(
                _cut(x, offs[i], offs[i] + a), [a], h0=h0[i : i + 1], final=True
            )
            assert torch.equal(hb[i], ref[0])
    assert torch.equal(ht[3], h0[3])


@pytest.mark.parametrize("lens", [[300], [100, 37, 64]])
@pytest.mark.parametrize("init", [False, True])
def test_accuracy_vs_fp64_recurrence(lens, init):
    x, h0 = make_inputs(sum(lens), 5, n_seq=len(lens), with_state=True)
    h0 = h0 if init else None
    o, ht = _run(x, lens, h0=h0, final=True)
    o_ref, s_ref = _reference(x, lens, h0)
    assert _rel(o, o_ref) < 1e-2
    assert _rel(ht, s_ref) < 1e-2


def test_accuracy_vs_fla():
    fla_kda = pytest.importorskip("fla.ops.kda")
    lens = [300, 77]
    x = make_inputs(sum(lens), 6)
    o, ht = _run(x, lens, final=True)
    q, k, v, g, beta, a_log, dt_bias = x
    o_fla, ht_fla = fla_kda.chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=a_log,
        dt_bias=dt_bias,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER,
        cu_seqlens=_cu(lens).long().cuda(),
    )
    o_ref, _ = _reference(x, lens)
    # same order of error as FLA's own, against the FP64 recurrence
    assert _rel(o, o_fla) < 1e-2 and _rel(ht, ht_fla) < 1e-2
    assert _rel(o, o_ref) < 1.5 * _rel(o_fla, o_ref) + 1e-4


def test_small_head_count():
    lens = [100, 64]
    x = make_inputs(sum(lens), 7, h=2)
    o, ht = _run(x, lens, final=True)
    o_ref, s_ref = _reference(x, lens)
    assert _rel(o, o_ref) < 1e-2 and _rel(ht, s_ref) < 1e-2
    o1, _ = _run(_cut(x, 0, 100), [100])
    assert torch.equal(o[:, :100], o1)


def test_strided_inputs_same_bits():
    T = 130
    x = make_inputs(T, 8)
    o, _ = _run(x, [T])

    def strided(t):
        if t.dim() < 3:
            return t
        wide = torch.empty(
            *t.shape[:-1], 2 * t.shape[-1], dtype=t.dtype, device=t.device
        )
        wide.fill_(float("nan"))
        wide[..., ::2] = t
        return wide[..., ::2]

    o2, _ = _run(tuple(strided(t) for t in x), [T])
    assert torch.equal(o, o2)


def test_cuda_graph_capture():
    lens = [150, 70]
    x = make_inputs(sum(lens), 9)
    y = make_inputs(sum(lens), 10)
    cu = _cu(lens)
    cu_dev = cu.cuda()
    bufs = tuple(t.clone() for t in x)

    def call():
        return kda_bi_fwd(
            *bufs, LOWER, cu_dev, cu_seqlens_cpu=cu, output_final_state=True
        )

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o_g, h_g = call()
    for b, t in zip(bufs, y):
        b.copy_(t)
    graph.replay()
    o, ht = _run(y, lens, final=True)
    assert torch.equal(o_g, o) and torch.equal(h_g, ht)


def test_rejects_bad_arguments():
    x = make_inputs(64, 1)
    q, k, v, g, beta, a_log, dt_bias = x
    lens = [64]
    with pytest.raises(ValueError, match="bf16"):
        _run((q.float(), k, v, g, beta, a_log, dt_bias), lens)
    with pytest.raises(ValueError, match="lower_bound"):
        kda_bi_fwd(*x, 0.0, _cu(lens).cuda())
    with pytest.raises(ValueError, match="lower_bound"):
        kda_bi_fwd(*x, -6.0, _cu(lens).cuda())
    with pytest.raises(ValueError, match="head dim"):
        _run(tuple(t[..., :64] if t.dim() == 4 else t for t in x), lens)
    with pytest.raises(ValueError, match="initial_state"):
        _run(x, lens, h0=torch.zeros(1, H, D, D, dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="initial_state"):
        _run(x, lens, h0=torch.zeros(2, H, D, D, device="cuda"))
    with pytest.raises(ValueError, match=r"\[1, T, H, 128\]"):
        kda_bi_fwd(
            q.expand(2, -1, -1, -1),
            k,
            v,
            g,
            beta,
            a_log,
            dt_bias,
            LOWER,
            _cu(lens).cuda(),
        )

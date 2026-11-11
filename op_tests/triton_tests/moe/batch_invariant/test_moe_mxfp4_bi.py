# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import hashlib
import json
import os

import pytest
import torch

from aiter.ops.triton.moe.batch_invariant.moe_mxfp4_bi import (
    BI_CONTRACT_VERSION,
    moe_mxfp4_bi_forward,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx950", reason="moe_mxfp4_bi has pinned configs for gfx950 only"
)

H, N, TOP_K = 512, 512, 16
BETA, LBETA = 4.0, 7.0
GOLDEN = os.path.join(os.path.dirname(__file__), "moe_mxfp4_bi_golden.json")
_E2M1 = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def _weights(E, n_out, K, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randint(
        0, 256, (E, n_out, K // 2), dtype=torch.uint8, device="cuda", generator=g
    )
    s = torch.randint(
        120, 128, (E, n_out, K // 32), dtype=torch.uint8, device="cuda", generator=g
    )
    return w, s


def generate_moe_mxfp4_bi_inputs(T, E, seed=0, h=H, n=N):
    """(x, w13, w13_scale, w2, w2_scale, topk_weights, topk_ids) for T tokens and E
    experts, all on the GPU."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w13, s13 = _weights(E, 2 * n, h, seed + 1)
    w2, s2 = _weights(E, h, n, seed + 2)
    x = torch.randn((T, h), dtype=torch.bfloat16, device="cuda", generator=g)
    ids = torch.argsort(
        torch.rand((T, E), dtype=torch.float32, device="cuda", generator=g), dim=1
    )[:, :TOP_K]
    tw = torch.rand((T, TOP_K), dtype=torch.float32, device="cuda", generator=g)
    return x, w13, s13, w2, s2, tw, ids


def _run(p, **kw):
    x, w13, s13, w2, s2, tw, ids = p
    return moe_mxfp4_bi_forward(
        x, w13, s13, w2, s2, tw, ids, situ_beta=BETA, situ_linear_beta=LBETA, **kw
    )


def _dequant(w, s):
    lut = torch.tensor(_E2M1, dtype=torch.float32, device="cuda")
    v = torch.stack((lut[(w & 0xF).long()], lut[(w >> 4).long()]), dim=-1).flatten(-2)
    return v * torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=-1)


def _bf16(t):
    return t.to(torch.bfloat16).float()


def _reference(p):
    """PyTorch reference with the bf16 roundings of the op (gate / up, activation, each
    W2 output). Only the FP32 summation order differs from the op."""
    x, w13, s13, w2, s2, tw, ids = p
    W13, W2 = _dequant(w13, s13), _dequant(w2, s2)
    out = torch.zeros((x.shape[0], H), dtype=torch.float32, device="cuda")
    for t in range(x.shape[0]):
        for j in range(TOP_K):
            e = ids[t, j]
            gu = _bf16(W13[e] @ x[t].float())
            g, u = gu[:N], gu[N:]
            act = _bf16(
                BETA
                * torch.tanh(g / BETA)
                * torch.sigmoid(g)
                * (LBETA * torch.tanh(u / LBETA))
            )
            out[t] += tw[t, j] * _bf16(W2[e] @ act)
    return out


def test_accuracy():
    p = generate_moe_mxfp4_bi_inputs(8, 32)
    ref = _reference(p)
    # tolerance relative to the output scale (rows sum terms of both signs)
    scale = ref.abs().max().item()
    torch.testing.assert_close(_run(p).float(), ref, rtol=1e-2, atol=1e-2 * scale)


@pytest.mark.parametrize("E", [32, 256])
def test_batch_invariance(E):
    """A token's output bits do not depend on the batch size or on the order of the
    tokens."""
    full = generate_moe_mxfp4_bi_inputs(4096, E)
    ref = _run(full)
    for M in (1, 2, 7, 16, 64, 255, 256, 512, 1024):
        sub = tuple(t[:M] if t.dim() and t.shape[0] == 4096 else t for t in full)
        assert torch.equal(_run(sub), ref[:M]), f"M={M}"
    perm = torch.randperm(4096, device="cuda")
    shuf = tuple(t[perm] if t.dim() and t.shape[0] == 4096 else t for t in full)
    assert torch.equal(_run(shuf), ref[perm])


def _two_pass(run_rank, ranks, reduce):
    """Simulate a collective on one GPU: pass 1 records each rank's inputs (one per
    call), pass 2 hands every rank the full result of each call in the same order."""
    seen = [[] for _ in range(ranks)]
    for r in range(ranks):

        def rec(t, r=r):
            seen[r].append(t.clone())
            return reduce([t] * ranks)

        run_rank(r, rec)
    fulls = [reduce([seen[r][k] for r in range(ranks)]) for k in range(len(seen[0]))]
    outs = []
    for r in range(ranks):
        it = iter(fulls)
        outs.append(run_rank(r, lambda t, it=it: next(it).clone()))
    return outs


@pytest.mark.parametrize("tp", [2, 8])
def test_tensor_parallel_layout(tp):
    p = generate_moe_mxfp4_bi_inputs(64, 32)
    ref = _run(p)
    x, w13, s13, w2, s2, tw, ids = p
    n = N // tp

    def run_rank(r, gather):
        rows = torch.cat(
            (
                torch.arange(r * n, (r + 1) * n, device="cuda"),
                torch.arange(N + r * n, N + (r + 1) * n, device="cuda"),
            )
        )
        q = (
            x,
            w13[:, rows].contiguous(),
            s13[:, rows].contiguous(),
            w2[:, :, r * n // 2 : (r + 1) * n // 2].contiguous(),
            s2[:, :, r * n // 32 : (r + 1) * n // 32].contiguous(),
            tw,
            ids,
        )
        return _run(q, n_global=N, chunk_gather=gather)

    for out in _two_pass(run_rank, tp, lambda planes: torch.cat(planes)):
        assert torch.equal(out, ref)


def test_chunk_planes_past_int32():
    """One TP rank, one large token block: 7 chunk-plane strides pass 2**31."""
    h, n, T = 3584, 3072, 5600
    assert 7 * T * TOP_K * h >= 2**31
    p = generate_moe_mxfp4_bi_inputs(T, 32, h=h, n=n)
    x, w13, s13, w2, s2, tw, ids = p
    ref = moe_mxfp4_bi_forward(
        x, w13, s13, w2, s2, tw, ids, situ_beta=BETA, situ_linear_beta=LBETA
    )
    out = moe_mxfp4_bi_forward(
        x,
        w13,
        s13,
        w2,
        s2,
        tw,
        ids,
        situ_beta=BETA,
        situ_linear_beta=LBETA,
        chunk_gather=lambda planes: planes,
        token_block=T,
    )
    assert torch.equal(out, ref)


@pytest.mark.parametrize("ep", [2, 4])
def test_expert_parallel_layout(ep):
    E = 32
    p = generate_moe_mxfp4_bi_inputs(64, E)
    ref = _run(p)
    x, w13, s13, w2, s2, tw, ids = p
    el = E // ep

    def run_rank(r, reduce):
        emap = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        emap[r * el : (r + 1) * el] = torch.arange(el, dtype=torch.int32, device="cuda")
        sl = slice(r * el, (r + 1) * el)
        q = (x, w13[sl], s13[sl], w2[sl], s2[sl], tw, ids)
        return _run(q, expert_map=emap, leaf_reduce=reduce)

    def leaf_sum(parts):
        acc = parts[0].clone()
        for t in parts[1:]:
            acc += t
        return acc

    for out in _two_pass(run_rank, ep, leaf_sum):
        assert torch.equal(out, ref)


def test_cuda_graph_replay():
    p = generate_moe_mxfp4_bi_inputs(64, 32)
    ref = _run(p)
    _run(p)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _run(p)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


@pytest.mark.parametrize(
    "ta,tb,el,eg,hn,blk",
    [
        (64, 256, 8, 32, (H, N), 64),
        (128, 512, 8, 32, (H, N), 128),
        (1024, 4096, 224, 896, (3584, 3072), None),
    ],
)
def test_ep_eager_after_cuda_graph(ta, tb, el, eg, hn, blk):
    """Expert parallel: an eager forward after a graph at another token count is bitwise
    equal to eager only."""
    emap = torch.full((eg,), -1, dtype=torch.int32, device="cuda")
    emap[:el] = torch.arange(el, dtype=torch.int32, device="cuda")
    g = torch.Generator(device="cuda").manual_seed(5)

    def inputs(T):
        p = generate_moe_mxfp4_bi_inputs(T, el, h=hn[0], n=hn[1])
        ids = torch.argsort(
            torch.rand((T, eg), dtype=torch.float32, device="cuda", generator=g), dim=1
        )[:, :TOP_K].contiguous()
        return p[:-1] + (ids,)

    def run(p):
        return _run(p, expert_map=emap, leaf_reduce=lambda t: t, token_block=blk)

    pa, pb = inputs(ta), inputs(tb)
    ref_a, ref_b = run(pa), run(pb)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run(pa)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_a = run(pa)
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_a, ref_a)
    del graph, out_a
    assert torch.equal(run(pb), ref_b)
    assert torch.equal(run(pa), ref_a)


@pytest.mark.parametrize("ep", [False, True])
def test_zero_tokens_non_contiguous(ep):
    """T = 0 with a strided x gives an empty output. With EP, leaf_reduce runs once."""
    E = 32
    _, w13, s13, w2, s2, _, _ = generate_moe_mxfp4_bi_inputs(1, E)
    x = torch.empty((0, 2 * H), dtype=torch.bfloat16, device="cuda")[:, ::2]
    tw = torch.empty((0, TOP_K), dtype=torch.float32, device="cuda")
    ids = torch.empty((0, TOP_K), dtype=torch.int64, device="cuda")
    calls = []
    kw = {}
    if ep:
        emap = torch.full((2 * E,), -1, dtype=torch.int32, device="cuda")
        emap[:E] = torch.arange(E, dtype=torch.int32, device="cuda")
        kw = {
            "expert_map": emap,
            "leaf_reduce": lambda t: calls.append(t.shape) or t,
        }
    out = _run((x, w13, s13, w2, s2, tw, ids), **kw)
    assert out.shape == (0, H) and out.dtype == torch.bfloat16
    assert calls == ([torch.Size((0, H))] if ep else [])


@pytest.mark.parametrize("T,E", [(1, 32), (64, 32), (1024, 256)])
def test_golden_bits(T, E):
    """The output bits are a contract: a change must bump BI_CONTRACT_VERSION and the
    golden file."""
    with open(GOLDEN) as f:
        golden = json.load(f)
    assert golden["contract_version"] == BI_CONTRACT_VERSION
    digest = hashlib.sha256(
        _run(generate_moe_mxfp4_bi_inputs(T, E, seed=7))
        .view(torch.int16)
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    assert digest == golden["gfx950"][f"T{T}_E{E}"]


@pytest.mark.parametrize("h,n", [(512 + 32, 512), (512, 8 * 96)])
def test_rejects_unsupported_sizes(h, n):
    """H and the W2 chunk must be multiples of the GEMM's 64-wide K step; other sizes
    are refused, not miscomputed."""
    x, w13, s13, w2, s2, tw, ids = generate_moe_mxfp4_bi_inputs(4, 32, h=h, n=n)
    with pytest.raises(ValueError):
        moe_mxfp4_bi_forward(x, w13, s13, w2, s2, tw, ids, situ_beta=BETA)


def _digest(out):
    return hashlib.sha256(out.view(torch.int16).cpu().numpy().tobytes()).hexdigest()


def _rows(p, sl):
    return tuple(t[sl] if t.dim() and t.shape[0] == p[0].shape[0] else t for t in p)


@pytest.mark.parametrize(
    "T,E,h",
    [
        (64, 112, H),
        (1500, 32, H),
        (1500, 112, H),
        (64, 32, 576),
        (64, 32, 640),
        (64, 32, 1024),
    ],
)
def test_split_invariance(T, E, h):
    """A batch gives the bits of its halves and of its first token: other tile
    heights, more than one routing block, H not a multiple of 512."""
    p = generate_moe_mxfp4_bi_inputs(T, E, seed=T, h=h)
    full = _run(p)
    k = T // 2
    for sl in (slice(0, k), slice(k, T), slice(0, 1)):
        assert torch.equal(_run(_rows(p, sl)), full[sl])


@pytest.mark.parametrize("n", [512, 1024])
@pytest.mark.parametrize("T", [0, 1, 64, 1024, 1500])
def test_expert_parallel_leaves(T, n):
    """Two expert-parallel ranks equal one GPU with all experts, with one leaf_reduce
    call per 1024 tokens (one for T = 0); N / 8 = 128 takes the async W2 leaves."""
    E = 32
    p = generate_moe_mxfp4_bi_inputs(T, E, seed=T, n=n)
    ref = _run(p)
    x, w13, s13, w2, s2, tw, ids = p
    el = E // 2
    calls = []

    def run_rank(r, reduce):
        emap = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        emap[r * el : (r + 1) * el] = torch.arange(el, dtype=torch.int32, device="cuda")
        sl = slice(r * el, (r + 1) * el)
        q = (x, w13[sl], s13[sl], w2[sl], s2[sl], tw, ids)

        def counted(t):
            calls.append(r)
            return reduce(t)

        return _run(q, expert_map=emap, leaf_reduce=counted)

    def leaf_sum(parts):
        acc = parts[0].clone()
        for t in parts[1:]:
            acc += t
        return acc

    for out in _two_pass(run_rank, 2, leaf_sum):
        assert torch.equal(out, ref)
    assert calls.count(0) == 2 * max(1, -(-T // 1024))


def test_token_block_over_buffer_range_raises():
    """A collective block whose buffers pass 2 GiB (32-bit buffer offsets) is refused,
    not silently wrapped."""
    x, w13, s13, w2, s2, tw, ids = generate_moe_mxfp4_bi_inputs(1, 32, seed=0)
    with pytest.raises(ValueError, match="token_block"):
        moe_mxfp4_bi_forward(
            x,
            w13,
            s13,
            w2,
            s2,
            tw,
            ids,
            situ_beta=BETA,
            situ_linear_beta=LBETA,
            chunk_gather=lambda t: t,
            token_block=1 << 22,
        )


@pytest.mark.parametrize("T", [0, 1, 64, 1500])
def test_leaf_planes_same_bits(T):
    """leaf_planes (reduce-scatter by column planes), 4 simulated ranks: each rank's
    result is the all-reduce result on its H / 4 columns and +0 elsewhere."""
    ws, e_loc = 4, 32
    p = list(generate_moe_mxfp4_bi_inputs(T, e_loc, seed=T, n=1024))
    g = torch.Generator(device="cuda").manual_seed(T)
    p[6] = torch.argsort(
        torch.rand((T, ws * e_loc), dtype=torch.float32, device="cuda", generator=g),
        dim=1,
    )[:, :16]
    perm = torch.randperm(ws * e_loc, device="cuda", generator=g)
    emaps = []
    for r in range(ws):
        m = torch.full((ws * e_loc,), -1, dtype=torch.int32, device="cuda")
        m[perm[r * e_loc : (r + 1) * e_loc]] = torch.arange(
            e_loc, dtype=torch.int32, device="cuda"
        )
        emaps.append(m)
    kw = {"situ_beta": BETA, "situ_linear_beta": LBETA}

    def ranks(planes, red):
        """Two passes: record every rank's leaves, then hand each rank the group's sum
        (or its plane of it)."""
        seen = [[] for _ in range(ws)]
        for r in range(ws):
            moe_mxfp4_bi_forward(
                *p,
                **kw,
                expert_map=emaps[r],
                leaf_planes=(ws, r) if planes else None,
                leaf_reduce=lambda z, r=r: seen[r].append(z.clone()) or red(z, r),
            )
        sums = [sum(c) for c in zip(*seen)]
        outs = []
        for r in range(ws):
            it = iter(sums)
            outs.append(
                moe_mxfp4_bi_forward(
                    *p,
                    **kw,
                    expert_map=emaps[r],
                    leaf_planes=(ws, r) if planes else None,
                    leaf_reduce=lambda z, r=r, it=it: red(next(it), r),
                )
            )
        return outs

    full = ranks(False, lambda s, r: s)
    plane = ranks(True, lambda s, r: s[r])
    # one GPU with all experts: expert perm[r * e_loc + k] has the weights of local k
    loc = torch.empty(ws * e_loc, dtype=torch.int64, device="cuda")
    loc[perm] = torch.arange(ws * e_loc, device="cuda") % e_loc
    one = list(p)
    for i in (1, 2, 3, 4):
        one[i] = p[i][loc]
    assert torch.equal(full[0], moe_mxfp4_bi_forward(*one, **kw))
    hp = p[0].shape[1] // ws
    for r in range(ws):
        cols = slice(r * hp, (r + 1) * hp)
        assert torch.equal(plane[r][:, cols], full[r][:, cols])
        rest = torch.cat((plane[r][:, : r * hp], plane[r][:, (r + 1) * hp :]), 1)
        assert torch.equal(
            rest.view(torch.int16), torch.zeros_like(rest.view(torch.int16))
        )
        assert torch.equal(full[r], full[0])


def _pad_view(t, extra):
    """The values of t in a buffer with longer rows: same logical tensor, rows no longer
    16-byte aligned."""
    buf = torch.zeros(
        t.shape[:-1] + (t.shape[-1] + extra,), dtype=t.dtype, device=t.device
    )
    buf[..., : t.shape[-1]] = t
    return buf[..., : t.shape[-1]]


def test_layout_fallbacks_same_bits():
    """Layout-selected kernels give the bits of the default ones for the same logical
    weights: w13 or w2 rows that are not 16-byte aligned (kernels without async
    copies)."""
    T, E, n = 64, 32, 1024
    p = generate_moe_mxfp4_bi_inputs(T, E, seed=3, n=n)
    ref = _run(p)
    x, w13, s13, w2, s2, tw, ids = p
    q = (x, _pad_view(w13, 8), _pad_view(s13, 1), w2, s2, tw, ids)
    assert torch.equal(_run(q), ref)
    q = (x, w13, s13, _pad_view(w2, 8), _pad_view(s2, 1), tw, ids)
    assert torch.equal(_run(q), ref)


def test_unaligned_hidden_same_bits():
    """H = 576 (K steps of 64) gives, on its columns, the bits of the same weights
    zero-padded to H = 1024 (default kernels)."""
    T, E, h, hp = 64, 32, 576, 1024
    p = generate_moe_mxfp4_bi_inputs(T, E, seed=4, h=h)
    x, w13, s13, w2, s2, tw, ids = p

    def pad_k(t, k):
        return torch.nn.functional.pad(t, (0, k - t.shape[-1]))

    def pad_rows(t, k):
        return torch.nn.functional.pad(t, (0, 0, 0, k - t.shape[-2]))

    q = (
        pad_k(x, hp),
        pad_k(w13, hp // 2),
        pad_k(s13, hp // 32),
        pad_rows(w2, hp),
        pad_rows(s2, hp),
        tw,
        ids,
    )
    assert torch.equal(_run(q)[:, :h], _run(p))


@pytest.mark.parametrize("T", [1024, 1025])
def test_expert_parallel_full_shape(T):
    """H 3584, N 3072: two expert-parallel ranks equal one GPU on both sides of 1024
    tokens (W2 epilogue leaves up to 1024, leaf scatter above)."""
    E, h, n = 16, 3584, 3072
    p = generate_moe_mxfp4_bi_inputs(T, E, seed=T, h=h, n=n)
    ref = _run(p)
    x, w13, s13, w2, s2, tw, ids = p
    el = E // 2

    def run_rank(r, reduce):
        emap = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        emap[r * el : (r + 1) * el] = torch.arange(el, dtype=torch.int32, device="cuda")
        sl = slice(r * el, (r + 1) * el)
        q = (x, w13[sl], s13[sl], w2[sl], s2[sl], tw, ids)
        return _run(q, expert_map=emap, leaf_reduce=reduce)

    def leaf_sum(parts):
        acc = parts[0].clone()
        for t in parts[1:]:
            acc += t
        return acc

    for out in _two_pass(run_rank, 2, leaf_sum):
        assert torch.equal(out, ref)


def _offset_view(t, off):
    """The values of t at a base pointer off bytes past a 16-byte boundary (rows stay
    16-byte aligned)."""
    buf = torch.zeros(
        t.shape[:-1] + (t.shape[-1] + 16,), dtype=t.dtype, device=t.device
    )
    v = buf[..., off : off + t.shape[-1]]
    v.copy_(t)
    return v


@pytest.mark.parametrize("n", [1024, 3072])
@pytest.mark.parametrize("T", [1, 64])
def test_w2_layout_fallback_same_bits(T, n):
    """W2 without async copies (rows or base not 16-byte aligned) at 16- and 64-row
    tiles gives the default bits."""
    E = 32
    p = generate_moe_mxfp4_bi_inputs(T, E, seed=7, n=n)
    ref = _run(p)
    x, w13, s13, w2, s2, tw, ids = p
    for q in (
        (x, w13, s13, _pad_view(w2, 8), _pad_view(s2, 1), tw, ids),
        (x, w13, s13, _offset_view(w2, 8), s2, tw, ids),
        (x, _offset_view(w13, 8), s13, w2, s2, tw, ids),
    ):
        assert torch.equal(_run(q), ref)


@pytest.mark.parametrize("ep", [False, True])
def test_out_of_range_ids_are_dropped(ep):
    """Ids outside [0, E_global) and non-local expert_map values count as another rank:
    the bits of in-range ids, shifted by one to keep the order, mapped to no expert."""
    T, E = 64, 32
    x, w13, s13, w2, s2, tw, ids = generate_moe_mxfp4_bi_inputs(T, E, seed=11)
    emap = torch.arange(E, dtype=torch.int32, device="cuda")
    if ep:
        emap[5] = E + 3  # not a local expert
    bad = ids.clone()
    bad[:, 3], bad[:, 7] = -1, E + 9
    kw = {"expert_map": emap, "leaf_reduce": lambda t: t} if ep else {}
    out = _run((x, w13, s13, w2, s2, tw, bad), **kw)
    emap_ref = torch.full((E + 16,), -1, dtype=torch.int32, device="cuda")
    emap_ref[1 : E + 1] = (
        emap if ep else torch.arange(E, dtype=torch.int32, device="cuda")
    )
    ref = _run(
        (x, w13, s13, w2, s2, tw, bad + 1), expert_map=emap_ref, leaf_reduce=lambda t: t
    )
    assert torch.equal(out, ref)


def test_bf16_topk_weights():
    p = list(generate_moe_mxfp4_bi_inputs(64, 32, seed=12))
    p[5] = p[5].to(torch.bfloat16)
    ref = _run(tuple(p[:5] + [p[5].to(torch.float32)] + p[6:]))
    assert torch.equal(_run(tuple(p)), ref)


def test_more_than_one_gemm_block():
    """No collectives: a batch above the 16384-token GEMM block gives the bits of its
    parts."""
    T = 16384 + 300
    p = generate_moe_mxfp4_bi_inputs(T, 32, seed=13)
    full = _run(p)
    for sl in (slice(0, 16384), slice(16384, T)):
        assert torch.equal(_run(_rows(p, sl)), full[sl])


def test_tensor_and_expert_parallel_together_refused():
    p = generate_moe_mxfp4_bi_inputs(4, 32)
    emap = torch.arange(32, dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="not both"):
        _run(p, expert_map=emap, leaf_reduce=lambda t: t, chunk_gather=lambda t: t)

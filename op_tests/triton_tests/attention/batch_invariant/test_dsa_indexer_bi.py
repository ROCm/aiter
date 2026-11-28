# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests of the batch-invariant DSA indexer.

The random inputs are drawn on the CPU and then copied to the GPU: the CUDA RNG
differs between torch builds, and the golden digests need the same inputs everywhere.
"""

import hashlib
import json
import os

import pytest
import torch

from aiter.ops.triton.attention.batch_invariant.dsa_indexer_bi import (
    BI_CONTRACT_VERSION,
    dsa_indexer_bi,
    dsa_mqa_logits_bi,
    dsa_topk_bi,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx950", reason="dsa_indexer_bi is for gfx950 only"
)

D = 128
TOPK = 2048  # DeepSeek V3.2 index_topk
FP8 = torch.float8_e4m3fn
GOLDEN = os.path.join(os.path.dirname(__file__), "dsa_indexer_bi_golden.json")


class Case:
    """Requests with keys in a paged NORMAL index cache, and query rows on them."""

    def __init__(self, req_lens, seed, H=32, BS=64, spare_blocks=3):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.H, self.BS, self.req_lens = H, BS, list(req_lens)
        self.keys = []  # per request: (fp8 [L, D], fp32 [L]) on the GPU
        for L in req_lens:
            k = (torch.randn((L, D), dtype=torch.float32, generator=g) * 2).to(FP8)
            s = torch.rand((L,), dtype=torch.float32, generator=g) * 0.1 + 0.01
            self.keys.append((k.cuda(), s.cuda()))
        self.g = g
        self.place(seed + 1, BS, spare_blocks)

    def place(self, seed, BS, spare_blocks=3):
        """Write the keys to a fresh cache with block size BS, blocks in random order."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        nblk = [-(-L // BS) for L in self.req_lens]
        total = sum(nblk) + spare_blocks
        perm = torch.randperm(total, generator=g)
        width = max(max(nblk), 1) + 2
        bt = torch.full((len(nblk), width), -1, dtype=torch.int32)
        cache = torch.randint(
            0, 256, (total, BS, D + 4), dtype=torch.uint8, generator=g
        )
        cache = cache.cuda()
        flat = cache.view(total, -1)
        i = 0
        for r, (L, n) in enumerate(zip(self.req_lens, nblk)):
            k, s = self.keys[r]
            kp = torch.zeros((n * BS, D), dtype=FP8, device="cuda")
            sp = torch.ones((n * BS,), dtype=torch.float32, device="cuda")
            kp[:L], sp[:L] = k, s
            for b in range(n):
                pb = int(perm[i])
                i += 1
                bt[r, b] = pb
                flat[pb, : BS * D] = (
                    kp[b * BS : (b + 1) * BS].view(torch.uint8).reshape(-1)
                )
                flat[pb, BS * D :] = (
                    sp[b * BS : (b + 1) * BS].view(torch.uint8).reshape(-1)
                )
        self.BS, self.cache, self.bt = BS, cache, bt.cuda()

    def rows(self, spec, seed):
        """Query rows for spec = [(request, row_len), ...]: (q, w, row_req, row_len)."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        T = len(spec)
        q = (torch.randn((T, self.H, D), dtype=torch.float32, generator=g) * 2).to(FP8)
        w = torch.randn((T, self.H), dtype=torch.float32, generator=g) * 0.1
        req = torch.tensor([r for r, _ in spec], dtype=torch.int32)
        ln = torch.tensor([n for _, n in spec], dtype=torch.int32)
        return q.cuda(), w.cuda(), req.cuda(), ln.cuda()

    def max_len(self):
        return max(self.req_lens)

    def logits(self, q, w, req, ln, max_len=None, **kw):
        return dsa_mqa_logits_bi(
            q, w, self.cache, self.bt, req, ln, max_len or self.max_len(), **kw
        )

    def indexer(self, q, w, req, ln, max_len=None, **kw):
        return dsa_indexer_bi(
            q, w, self.cache, self.bt, req, ln, TOPK, max_len or self.max_len(), **kw
        )

    def ref_logits(self, q, w, req, ln):
        """FP64 logits from the same fp8 values, and an fp32 error bound for each."""
        out = []
        for t in range(q.shape[0]):
            k, s = self.keys[int(req[t])]
            n = int(ln[t])
            qd, kd, wd, sd = (
                q[t].double(),
                k[:n].double(),
                w[t].double(),
                s[:n].double(),
            )
            sc = torch.einsum("hd,nd->hn", qd, kd).relu()
            mag = torch.einsum("hd,nd->hn", qd.abs(), kd.abs())
            ref = (sc * wd[:, None]).sum(0) * sd
            # D + H fp32 roundings at most, each relative to the absolute terms
            bound = (D + self.H) * 2.0**-24 * (mag * wd.abs()[:, None]).sum(0) * sd
            out.append((ref, bound))
        return out


def assert_logits_close(got, ref):
    for t, (a, (b, bound)) in enumerate(zip(got, ref)):
        err = (a.double() - b).abs()
        assert bool((err <= bound + 1e-30).all()), (t, float((err - bound).max()))


def generate_dsa_indexer_bi_cache(num_reqs, req_len, BS=64, device="cuda"):
    """A paged cache of ``num_reqs`` requests of ``req_len`` random keys, blocks in
    random order (GPU RNG: not for goldens). Returns (index_cache, block_table)."""
    nblk = -(-req_len // BS)
    total = num_reqs * nblk
    cache = torch.empty((total, BS, D + 4), dtype=torch.uint8, device=device)
    flat = cache.view(total, -1)
    keys = torch.randn((total, BS * D), dtype=torch.float32, device=device) * 2
    flat[:, : BS * D] = keys.to(FP8).view(torch.uint8)
    scales = torch.rand((total, BS), dtype=torch.float32, device=device) * 0.1 + 0.01
    flat[:, BS * D :] = scales.view(torch.uint8)
    bt = torch.randperm(total, device=device).to(torch.int32).view(num_reqs, nblk)
    return cache, bt


def generate_dsa_indexer_bi_rows(num_reqs, req_len, rows_per_req, H=32, device="cuda"):
    """The last ``rows_per_req`` causal rows of each request (1: decode), GPU RNG.
    Returns (q, weights, row_req, row_len)."""
    T = num_reqs * rows_per_req
    q = (torch.randn((T, H, D), dtype=torch.float32, device=device) * 2).to(FP8)
    w = torch.randn((T, H), dtype=torch.float32, device=device) * 0.1
    req = torch.arange(num_reqs, device=device).repeat_interleave(rows_per_req)
    pos = torch.arange(req_len - rows_per_req + 1, req_len + 1, device=device)
    return q, w, req.to(torch.int32), pos.repeat(num_reqs).to(torch.int32)


def valid(logits, ln):
    return [logits[t, : int(ln[t])] for t in range(logits.shape[0])]


def ordered_key(x):
    """fp32 -> int64 key in the total order of the bit patterns (-0.0 < +0.0)."""
    b = x.contiguous().view(torch.int32).long()
    return torch.where(b < 0, -(b & 0x7FFFFFFF) - 1, b)


def ref_topk(logits, ln, k):
    out = torch.full((logits.shape[0], k), -1, dtype=torch.int32)
    for t in range(logits.shape[0]):
        n = int(ln[t])
        if n == 0:
            continue
        # by key, ties to the larger column: (key, col) is unique and exact in int64
        col = torch.arange(n)
        comp = ordered_key(logits[t, :n].cpu()) * (1 << 31) + col
        sel = torch.sort(col[torch.topk(comp, min(k, n)).indices]).values
        out[t, : sel.numel()] = sel.to(torch.int32)
    return out


def digest(*tensors):
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


# Rows of three requests: causal prefill rows, decode rows, short and empty rows.
REQ_LENS = [3000, 700, 9000]
SPEC = [
    (0, 1),
    (0, 1500),
    (0, 3000),
    (1, 64),
    (1, 700),
    (2, 2048),
    (2, 4097),
    (2, 9000),
    (0, 0),
    (2, 65),
]


@pytest.mark.parametrize("H", [32, 64])
@pytest.mark.parametrize("BS", [64, 128])
def test_logits_match_fp64(H, BS):
    c = Case(REQ_LENS, seed=1, H=H, BS=BS)
    q, w, req, ln = c.rows(SPEC, seed=2)
    assert_logits_close(valid(c.logits(q, w, req, ln), ln), c.ref_logits(q, w, req, ln))


def topk_case(seed, n=9000):
    g = torch.Generator(device="cpu").manual_seed(seed)
    # Few distinct values: many ties at the cutoff. Some -0.0 next to +0.0.
    x = torch.randint(0, 600, (9, n), generator=g).float() / 8 - 30
    x[x == 0] = torch.where(
        torch.rand(int((x == 0).sum()), generator=g) < 0.5, -0.0, 0.0
    )
    x[3, :3000] = 0.0
    x[3, 1000:2000] = -0.0
    lens = [9000, 4000, 1500, 3000, 0, 1, 4096, 4097, TOPK]
    return x.cuda(), torch.tensor(lens, dtype=torch.int32).cuda()


@pytest.mark.parametrize("seed", [0, 1])
def test_topk_matches_reference(seed):
    x, ln = topk_case(seed)
    got = dsa_topk_bi(x, ln, TOPK).cpu()
    assert torch.equal(got, ref_topk(x, ln, TOPK))


def test_topk_row_depends_only_on_its_row():
    x, ln = topk_case(0)
    full = dsa_topk_bi(x, ln, TOPK)
    for t in range(x.shape[0]):
        alone = dsa_topk_bi(x[t : t + 1].clone(), ln[t : t + 1].clone(), TOPK)
        assert torch.equal(alone[0], full[t]), t
    perm = torch.tensor([6, 2, 0, 5, 1, 4, 3, 8, 7], device="cuda")
    assert torch.equal(dsa_topk_bi(x[perm], ln[perm], TOPK), full[perm])
    # a wider buffer with other values past each row
    wide = torch.full((9, 12000), 1e30, dtype=torch.float32, device="cuda")
    wide[:, :9000] = x
    assert torch.equal(dsa_topk_bi(wide, ln, TOPK), full)


def test_topk_long_rows_any_chunking():
    # Long tie-heavy rows; the launch splits them in other chunks for other T and W.
    g = torch.Generator(device="cpu").manual_seed(5)
    W = 1 << 22
    x = (torch.randint(0, 3000, (6, W), generator=g).float() / 16 - 90).cuda()
    x[1, 100:200000] = 0.0
    x[1, 300000:400000] = -0.0
    lens = [W, 3000000, 8192, 8193, 200001, 1000003]
    ln = torch.tensor(lens, dtype=torch.int32).cuda()
    full = dsa_topk_bi(x, ln, TOPK)
    assert torch.equal(full.cpu(), ref_topk(x, ln, TOPK))
    for t in range(x.shape[0]):
        n = lens[t]
        alone = dsa_topk_bi(x[t : t + 1, :n].clone(), ln[t : t + 1].clone(), TOPK)
        assert torch.equal(alone[0], full[t]), t


@pytest.mark.parametrize("H", [32, 64])
def test_rows_are_batch_invariant(H):
    """Each row's logits and indices are the same alone, reordered, with another
    max_len, block size, block placement or workspace group size."""
    c = Case(REQ_LENS, seed=3, H=H)
    q, w, req, ln = c.rows(SPEC, seed=4)
    ref_l = valid(c.logits(q, w, req, ln), ln)
    ref_i = c.indexer(q, w, req, ln)
    T = q.shape[0]
    for t in range(T):
        sl = slice(t, t + 1)
        args = (q[sl].clone(), w[sl].clone(), req[sl].clone(), ln[sl].clone())
        assert torch.equal(valid(c.logits(*args), args[3])[0], ref_l[t]), t
        assert torch.equal(c.indexer(*args)[0], ref_i[t]), t
    rev = torch.arange(T - 1, -1, -1, device="cuda")
    args = (q[rev], w[rev], req[rev], ln[rev])
    out_l = valid(c.logits(*args, max_len=12000), args[3])
    out_i = c.indexer(*args, max_len=12000)
    for i, t in enumerate(rev.tolist()):
        assert torch.equal(out_l[i], ref_l[t]), t
        assert torch.equal(out_i[i], ref_i[t]), t
    for G in (1, 3):
        ws = torch.empty((G, c.max_len()), dtype=torch.float32, device="cuda")
        assert torch.equal(c.indexer(q, w, req, ln, logits_ws=ws), ref_i)
    c.place(seed=11, BS=128, spare_blocks=7)
    assert all(
        torch.equal(a, b) for a, b in zip(valid(c.logits(q, w, req, ln), ln), ref_l)
    )
    assert torch.equal(c.indexer(q, w, req, ln), ref_i)


@pytest.mark.parametrize("H", [32, 64])
def test_decode_row_equals_prefill_row(H):
    """A decode row (one token on the cached keys) equals that token's prefill row,
    also when the batch starts one row later (other row groups)."""
    c = Case([5000], seed=5, H=H)
    L = 5000
    prefill = [(0, t + 1) for t in range(0, L, 7)] + [(0, L)]
    q, w, req, ln = c.rows(prefill, seed=6)
    pre = c.indexer(q, w, req, ln)
    pre_l = valid(c.logits(q, w, req, ln), ln)
    for t in (0, len(prefill) // 2, len(prefill) - 1):
        sl = slice(t, t + 1)
        args = (q[sl].clone(), w[sl].clone(), req[sl].clone(), ln[sl].clone())
        assert torch.equal(c.indexer(*args, max_len=131072)[0], pre[t])
        assert torch.equal(valid(c.logits(*args, max_len=131072), args[3])[0], pre_l[t])
    sh = (q[1:], w[1:], req[1:], ln[1:])
    assert torch.equal(c.indexer(*sh), pre[1:])
    assert all(
        torch.equal(a, b) for a, b in zip(valid(c.logits(*sh), sh[3]), pre_l[1:])
    )


def test_long_decode_rows():
    """Rows past the in-register top-k and over many KV splits."""
    c = Case([70000, 20000], seed=7)
    q, w, req, ln = c.rows([(0, 70000), (1, 20000), (0, 33333)], seed=8)
    logits = c.logits(q, w, req, ln)
    assert_logits_close(valid(logits, ln), c.ref_logits(q, w, req, ln))
    idx = c.indexer(q, w, req, ln)
    assert torch.equal(idx.cpu(), ref_topk(logits, ln, TOPK))
    assert torch.equal(c.indexer(q[2:], w[2:], req[2:], ln[2:])[0], idx[2])


def test_strided_inputs_and_outputs():
    c = Case(REQ_LENS, seed=9)
    q, w, req, ln = c.rows(SPEC, seed=10)
    ref_l = valid(c.logits(q, w, req, ln), ln)
    ref_i = c.indexer(q, w, req, ln)
    T, H = w.shape
    q2 = torch.empty((T, 2 * H, D), dtype=FP8, device="cuda")[:, ::2]
    q2.copy_(q)
    w2 = w.t().contiguous().t()
    bt2 = torch.full(
        (c.bt.shape[0], c.bt.shape[1] + 5), -1, dtype=torch.int32, device="cuda"
    )
    bt2[:, 5:] = c.bt
    out = torch.full((T, TOPK + 9), 7, dtype=torch.int32, device="cuda")
    logits = torch.empty((T, c.max_len() + 33), dtype=torch.float32, device="cuda")
    got_l = dsa_mqa_logits_bi(
        q2, w2, c.cache, bt2[:, 5:], req, ln, c.max_len(), out=logits
    )
    got_i = dsa_indexer_bi(
        q2, w2, c.cache, bt2[:, 5:], req, ln, TOPK, c.max_len(), out=out[:, :TOPK]
    )
    assert all(torch.equal(a, b) for a, b in zip(valid(got_l, ln), ref_l))
    assert torch.equal(got_i, ref_i)
    assert bool((out[:, TOPK:] == 7).all())


def test_empty_batch_and_rows():
    c = Case([100], seed=12)
    q, w, req, ln = c.rows([(0, 0), (0, 0)], seed=13)
    assert bool((c.indexer(q, w, req, ln) == -1).all())
    out = c.indexer(q[:0], w[:0], req[:0], ln[:0])
    assert out.shape == (0, TOPK)


def test_cuda_graph_replay():
    c = Case(REQ_LENS, seed=14)
    q, w, req, ln = c.rows(SPEC, seed=15)
    max_len = 16384
    c.indexer(q, w, req, ln, max_len=max_len)  # compile outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = c.indexer(q, w, req, ln, max_len=max_len)
    q2, w2, _, _ = c.rows(SPEC, seed=16)
    ln2 = torch.flip(ln, [0]).clamp(max=700)
    q.copy_(q2)
    w.copy_(w2)
    ln.copy_(ln2)
    req.copy_(torch.tensor([1] * len(SPEC), dtype=torch.int32))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, c.indexer(q2, w2, req, ln2, max_len=max_len))


def _bad(c, q, w, req, ln, **change):
    a = {
        "q": q,
        "weights": w,
        "index_cache": c.cache,
        "block_table": c.bt,
        "row_req": req,
        "row_len": ln,
        "topk": TOPK,
        "max_len": c.max_len(),
    }
    a.update(change)
    return a


@pytest.mark.parametrize(
    "change",
    [
        "cpu",
        "q_bf16",
        "head_size",
        "heads",
        "w_dtype",
        "cache_bs",
        "bt_int64",
        "row_len_int64",
        "max_len",
        "topk",
    ],
)
def test_refusals(change):
    c = Case([300], seed=17)
    q, w, req, ln = c.rows([(0, 300)], seed=18)
    bad = {
        "cpu": {"q": q.cpu()},
        "q_bf16": {"q": q.to(torch.bfloat16)},
        "head_size": {"q": q[:, :, :64]},
        "heads": {"q": q[:, :16], "weights": w[:, :16]},
        "w_dtype": {"weights": w.to(torch.bfloat16)},
        "cache_bs": {"index_cache": c.cache[:, :32]},
        "bt_int64": {"block_table": c.bt.long()},
        "row_len_int64": {"row_len": ln.long()},
        "max_len": {"max_len": 1 << 30},
        "topk": {"topk": 1000},
    }[change]
    with pytest.raises((ValueError, TypeError)):
        dsa_indexer_bi(**_bad(c, q, w, req, ln, **bad))


@pytest.mark.parametrize("H", [32, 64])
def test_golden_bits(H):
    """The output bits are a contract: a change must bump BI_CONTRACT_VERSION and
    the golden file."""
    with open(GOLDEN) as f:
        golden = json.load(f)
    assert golden["contract_version"] == BI_CONTRACT_VERSION
    c = Case(REQ_LENS, seed=21, H=H)
    q, w, req, ln = c.rows(SPEC, seed=22)
    got = (
        digest(*valid(c.logits(q, w, req, ln), ln)),
        digest(c.indexer(q, w, req, ln)),
    )
    want = golden["gfx950"][f"H{H}"]
    assert list(got) == [want["logits"], want["indices"]], f"H{H}: {got}"

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""OpusMlaFlatDecode: the host flat-row planner, the opus fp8 32mx4 stage 1 and the bf16-partial
merge.

The plans are checked for exact KV coverage per (request, 128-row chunk), partial slots and merge
units, and keys for every row. The decode is compared with an fp32 torch reference at 96 heads with
round-robin context parallelism (8 ranks, one of them) and at 12 and 16 heads without it, at qlen
2-8, for KV mixes from a few tiles per request to long lognormal contexts and contexts ending 1-5
tokens past a KV tile, with non-unit fp8 descales and a scattered kv_indices. A CUDA graph captured
on one plan is replayed on another.
"""

import math
import sys

import numpy as np
import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

KV_LORA_RANK = 512
DIM = KV_LORA_RANK + 64
SLOT = 128
KV_TILE = 32
CP_WORLD = 8
RANK = 3
Q_SCALE, KV_SCALE = 0.75, 1.25


def _skip_unless_gfx950():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    if get_gfx() != "gfx950":
        pytest.skip("the flat-row opus decode is gfx950 only")


def _decoder(max_batch=64):
    from aiter.ops.opus.mla_flat import OpusMlaFlatDecode

    return OpusMlaFlatDecode(max_batch=max_batch)


def _ctxs(kind, seed, bs=24):
    """Global context lengths."""
    rng = np.random.default_rng(seed)
    if kind == "short":
        # a few KV tiles per request
        return [CP_WORLD * int(x) for x in rng.integers(40, 600, size=bs)]
    if kind == "tail":
        # local lengths of 1-5 past a whole KV tile: a last tile shorter than the query
        return [
            CP_WORLD * (KV_TILE * int(rng.integers(4, 64)) + 1 + i % 5)
            for i in range(bs)
        ]
    # lognormal around 66k global tokens
    return [
        int(min(max(66000 * math.exp(0.67 * rng.standard_normal()), 2000), 200000))
        for _ in range(bs)
    ]


def _local_lens(ctxs, cp_rr):
    if cp_rr:
        return [(c - RANK + CP_WORLD - 1) // CP_WORLD for c in ctxs]
    return [c // CP_WORLD for c in ctxs]


def _indptr(lens):
    out = np.zeros(len(lens) + 1, dtype=np.int32)
    out[1:] = np.cumsum(lens)
    return out


def _check_plan(dec, kvp, q_len, nhead, balance):
    """Plan invariants on the host copy of the last plan."""
    plan = dec._host[(dec._step - 1) & 1].numpy()
    nwg, cap = dec.num_workers, dec.item_cap
    wip = plan[: nwg + 1]
    items = plan[nwg + 1 : nwg + 1 + cap * 8].reshape(cap, 8)
    units = plan[nwg + 1 + cap * 8 :].reshape(-1, 4)
    n_items, n_units = dec.last["items"], dec.last["units"]
    bs = len(kvp) - 1
    nrows = q_len * nhead
    nch = -(-nrows // SLOT)
    assert wip[0] == 0 and (np.diff(wip) >= 0).all() and wip[-1] == n_items
    it = items[:n_items]
    assert not items[n_items:].any() and not units[n_units:].any()
    pieces = {}
    for b, slot, q0, q1, k0, k1, _, _ in it.tolist():
        chunk = (q0 - b * nrows) // SLOT
        assert q0 == b * nrows + chunk * SLOT and q1 == q0 + min(
            SLOT, nrows - chunk * SLOT
        )
        assert kvp[b] <= k0 < k1 <= kvp[b + 1]
        if balance and k1 == kvp[b + 1]:
            # the query's first token sees kv_end - q_len + 1 keys of the request
            assert k0 <= kvp[b + 1] - q_len, "a piece has rows that see no key"
        pieces.setdefault((b, chunk), []).append((k0, k1, slot))
    assert len(pieces) == bs * nch
    merged = {}
    for (b, chunk), ps in pieces.items():
        ps.sort()
        assert ps[0][0] == kvp[b] and ps[-1][1] == kvp[b + 1]
        assert all(
            ps[i][1] == ps[i + 1][0] for i in range(len(ps) - 1)
        ), "KV gap or overlap"
        slots = sorted(p[2] for p in ps)
        if len(ps) == 1:
            assert slots == [-1], "an unsplit unit writes its output directly"
        else:
            assert slots == list(range(slots[0], slots[0] + len(ps)))
            merged[(b, chunk)] = (slots[0], slots[-1] + 1)
    assert n_units == len(merged)
    got = {}
    for s0, s1, row0, rows in units[:n_units].tolist():
        b, chunk = row0 // nrows, (row0 % nrows) // SLOT
        assert rows == min(SLOT, nrows - chunk * SLOT)
        got[(b, chunk)] = (s0, s1)
    assert got == merged
    used = [s for ps in pieces.values() for _, _, s in ps if s >= 0]
    assert len(set(used)) == len(used), "two pieces share a partial slot"
    assert all(s < n_items for s in used), "a partial slot is out of range"


def _reference(q, kv, q_len, lens, ctxs, cp_rr):
    """fp32 causal attention over each request's local KV, on global positions with CP."""
    outs, lses = [], []
    kv_at = 0
    for b, n in enumerate(lens):
        qb = q[b * q_len : (b + 1) * q_len].float() * Q_SCALE
        k = kv[kv_at : kv_at + n].float() * KV_SCALE
        kv_at += n
        s = torch.einsum("thd,ld->thl", qb, k) * DIM**-0.5
        j = torch.arange(n, device="cuda")
        t = torch.arange(q_len, device="cuda")
        if cp_rr:
            vis = (j[None, :] * CP_WORLD + RANK) <= (ctxs[b] - q_len + t[:, None])
        else:
            vis = j[None, :] <= (n - q_len + t[:, None])
        s = s.masked_fill(~vis[:, None, :], float("-inf"))
        lses.append(s.logsumexp(-1))
        outs.append(torch.einsum("thl,ld->thd", s.softmax(-1), k[:, :KV_LORA_RANK]))
    return torch.cat(outs), torch.cat(lses)


class _Case:
    def __init__(self, nhead, q_len, ctxs, cp_rr, seed=7):
        self.nhead, self.q_len, self.ctxs, self.cp_rr = nhead, q_len, ctxs, cp_rr
        self.lens = _local_lens(ctxs, cp_rr)
        bs, total = len(ctxs), sum(self.lens)
        g = torch.Generator(device="cuda").manual_seed(seed)
        self.q = torch.randn((bs * q_len, nhead, DIM), generator=g, device="cuda").to(
            dtypes.fp8
        )
        self.kv = torch.randn((total, DIM), generator=g, device="cuda").to(dtypes.fp8)
        # the requests' cache rows, scattered over a larger cache
        self.slots = torch.randperm(total + 4096, generator=g, device="cuda")[
            :total
        ].int()
        self.cache = torch.zeros((total + 4096, DIM), device="cuda").to(dtypes.fp8)
        self.cache[self.slots.long()] = self.kv
        self.kvp = _indptr(self.lens)
        self.out = torch.full(
            (bs * q_len, nhead, KV_LORA_RANK), float("nan"), device="cuda"
        ).to(torch.bfloat16)
        self.qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device="cuda") * q_len
        self.kv_indptr = torch.from_numpy(self.kvp).cuda()
        self.g_kv_indptr = torch.from_numpy(_indptr(ctxs)).cuda() if cp_rr else None
        self.cp = CP_WORLD if cp_rr else 1
        self.q_scale = torch.tensor([Q_SCALE], device="cuda")
        self.kv_scale = torch.tensor([KV_SCALE], device="cuda")

    def plan(self, dec):
        return dec.plan(self.kvp, self.q_len, self.nhead, self.cp)

    def run(self, dec, lse=None):
        return dec(
            self.q,
            self.cache.view(-1, 1, 1, DIM),
            self.out,
            self.qo_indptr,
            self.kv_indptr,
            self.slots,
            DIM**-0.5,
            self.q_scale,
            self.kv_scale,
            self.g_kv_indptr,
            self.cp,
            RANK if self.cp_rr else 0,
            lse=lse,
        )

    def check(self, lse):
        torch.cuda.synchronize()
        o_ref, lse_ref = _reference(
            self.q, self.kv, self.q_len, self.lens, self.ctxs, self.cp_rr
        )
        o32 = self.out.float()
        assert not o32.isnan().any(), "some output row is NaN or was never written"
        err = ((o32 - o_ref).abs().max() / o_ref.abs().max()).item()
        # P is rounded to fp8 for P*V and the partials to bf16
        assert err < 5e-2, f"O max error {err:.3e} of the reference's max"
        lerr = (lse.float() - lse_ref).abs().max().item()
        assert lerr < 1e-2, f"LSE max abs error {lerr:.3e}"


SHAPES = [
    # (heads, cp round-robin, qlen): 96 heads gathered under 8-way CP, 12 / 16 per rank without
    *[(96, True, q) for q in (2, 4, 5, 6, 8)],
    *[(96, False, q) for q in (4, 6)],
    *[(h, False, q) for h in (12, 16) for q in (2, 4, 6, 8)],
]


@pytest.mark.parametrize("kind", ["mixed", "short", "tail"])
@pytest.mark.parametrize(
    "nhead,cp_rr,q_len",
    SHAPES,
    ids=[f"h{h}-{'cprr' if c else 'local'}-q{q}" for h, c, q in SHAPES],
)
def test_flat_decode(nhead, cp_rr, q_len, kind):
    _skip_unless_gfx950()
    dec = _decoder()
    case = _Case(nhead, q_len, _ctxs(kind, seed=5), cp_rr)
    case.plan(dec)
    _check_plan(dec, case.kvp, q_len, nhead, balance=not cp_rr)
    case.check(case.run(dec))


@pytest.mark.parametrize("cp_rr", [True, False], ids=["cprr", "local"])
@pytest.mark.parametrize("bs", [1, 2, 3, 4, 11, 32, 64])
def test_flat_decode_batch(bs, cp_rr):
    _skip_unless_gfx950()
    nhead, q_len = (96, 4) if cp_rr else (12, 6)
    dec = _decoder()
    case = _Case(nhead, q_len, _ctxs("mixed", seed=bs, bs=bs), cp_rr)
    case.plan(dec)
    _check_plan(dec, case.kvp, q_len, nhead, balance=not cp_rr)
    case.check(case.run(dec))


@pytest.mark.parametrize("cp_rr", [True, False], ids=["cprr", "local"])
def test_flat_decode_graph_replay(cp_rr):
    """A graph captured on an empty plan replays the plan published before each replay."""
    _skip_unless_gfx950()
    nhead, q_len, bs = (96, 5, 16) if cp_rr else (16, 6, 16)
    dec = _decoder()
    cases = [
        _Case(nhead, q_len, _ctxs(kind, seed=s, bs=bs), cp_rr, seed=s)
        for s, kind in ((1, "mixed"), (2, "tail"), (3, "short"))
    ]
    # the graph's inputs, sized for the largest case; each case is copied in before its replay
    rows = max(c.cache.shape[0] for c in cases)
    g = _Case(nhead, q_len, cases[0].ctxs, cp_rr, seed=1)
    g.cache = torch.zeros((rows, DIM), device="cuda").to(dtypes.fp8)
    g.slots = torch.zeros(
        max(c.slots.numel() for c in cases), dtype=torch.int32, device="cuda"
    )
    g.g_kv_indptr = (
        torch.zeros(bs + 1, dtype=torch.int32, device="cuda") if cp_rr else None
    )
    lse = torch.empty(bs * q_len, nhead, dtype=torch.float32, device="cuda")
    dec.clear()
    g.run(dec, lse)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g.run(dec, lse)
    for case in cases:
        case.plan(dec)
        g.q.copy_(case.q)
        g.cache.zero_()
        g.cache[: case.cache.shape[0]] = case.cache
        g.slots.zero_()
        g.slots[: case.slots.numel()] = case.slots
        g.kv_indptr.copy_(case.kv_indptr)
        if cp_rr:
            g.g_kv_indptr.copy_(case.g_kv_indptr)
        g.out.fill_(float("nan"))
        graph.replay()
        case.out.copy_(g.out)
        case.check(lse.clone())


def test_flat_decode_rejects():
    _skip_unless_gfx950()
    dec = _decoder()
    with pytest.raises(ValueError, match="rows"):
        dec.plan(_indptr([100, 200]), 9, 128)
    with pytest.raises(ValueError, match="max_batch"):
        dec.plan(_indptr([100] * 65), 4, 96, CP_WORLD)
    case = _Case(16, 4, _ctxs("short", seed=3, bs=2), False)
    case.plan(dec)
    case.q = case.q.to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="fp8"):
        case.run(dec)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rs", *sys.argv[1:]]))

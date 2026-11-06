# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""topk_topp_sample against the sampler it replaces: softmax(logits / t), then
top_k_top_p_sampling_from_probs on its top-k-first path with the small-k
top-p draw, then the argmax of greedy rows. The radix top-k of that path
returns the k columns in no fixed order, and on a tie at the k-th value no
fixed set. The reference here takes the k largest logits by a stable sort
(the lowest column first on a tie), the order and set the kernel uses. The
draws then agree bit for bit."""

import pytest
import torch

from aiter.ops import sampling
from aiter.ops.triton.fusions.topk_topp_sample import topk_topp_sample

VOCAB = 248320
EPS = 1e-10


def _logits(kind, rows, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    if kind == "zipf":
        a = torch.rand((rows, 1), dtype=torch.float32, device="cuda", generator=g)
        r = torch.rand((rows, VOCAB), dtype=torch.float32, device="cuda", generator=g)
        rank = torch.argsort(r, dim=1).float() + 1
        x = -(a * 2.5 + 0.8) * rank.log() + 20
        x += 0.3 * torch.randn(
            (rows, VOCAB), dtype=torch.float32, device="cuda", generator=g
        )
    elif kind == "peaky":
        x = torch.randn((rows, VOCAB), dtype=torch.float32, device="cuda", generator=g)
        x = x * 2
        hot = torch.randint(0, VOCAB, (rows,), device="cuda", generator=g)
        x[torch.arange(rows, device="cuda"), hot] += 12
    else:  # coarse values: ties at the k-th value
        x = torch.randn((rows, VOCAB), dtype=torch.float32, device="cuda", generator=g)
        x = (x * 2).round() + 10
    return x.to(torch.bfloat16)


def _reference(logits, temperatures, k, top_p, seed, monkeypatch):
    """The sampler's eager path, with a stable top-k."""

    def stable_topk(probs, kk):
        ids = torch.sort(logits.float(), dim=1, descending=True, stable=True)
        ids = ids.indices[:, :kk]
        return probs.gather(1, ids), ids

    monkeypatch.setattr(sampling, "_SMALL_K_TOPP", True)
    monkeypatch.setattr(sampling, "_select_topk", stable_topk)
    probs = (logits / temperatures.unsqueeze(-1)).softmax(dim=-1, dtype=torch.float32)
    k_arr = None
    p_arr, p_val = (top_p, 0.0) if isinstance(top_p, torch.Tensor) else (None, top_p)
    torch.manual_seed(seed)
    tokens = sampling.top_k_top_p_sampling_from_probs(
        probs, None, k_arr, k, p_arr, p_val, deterministic=True
    )
    greedy = torch.argmax(probs, dim=-1).to(torch.int32)
    return torch.where(temperatures <= EPS, greedy, tokens.view(-1))


@pytest.mark.parametrize("rows", [3, 47, 512])
@pytest.mark.parametrize("kind", ["zipf", "peaky", "ties"])
@pytest.mark.parametrize("temps", ["one", "mixed"])
def test_matches_top_k_first_sampler(rows, kind, temps, monkeypatch):
    logits = _logits(kind, rows, rows)
    if temps == "one":
        t = torch.ones((rows,), dtype=torch.float32, device="cuda")
    else:
        t = torch.tensor([1.0, EPS, 0.7, 1.3], dtype=torch.float32, device="cuda")
        t = t.repeat(rows // 4 + 1)[:rows].contiguous()
    for seed in range(3):
        torch.manual_seed(seed)
        out = topk_topp_sample(logits, t, 20, 0.95, eps=EPS)
        ref = _reference(logits, t, 20, 0.95, seed, monkeypatch)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("k", [1, 8, 20, 64])
def test_k_and_per_request_top_p(k, monkeypatch):
    rows = 64
    logits = _logits("zipf", rows, k)
    t = torch.ones((rows,), dtype=torch.float32, device="cuda")
    top_p = torch.rand((rows,), dtype=torch.float32, device="cuda") * 0.6 + 0.4
    torch.manual_seed(k)
    out = topk_topp_sample(logits, t, k, top_p, eps=EPS)
    ref = _reference(logits, t, k, top_p, k, monkeypatch)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_verify_and_bonus_rows(monkeypatch):
    """Rows read in place: verify rows, then bonus rows, each with its request."""
    bs, steps = 37, 3
    logits = _logits("zipf", bs * (steps + 1) + 5, 11)
    i = torch.arange(bs, device="cuda")
    verify = (i[:, None] * (steps + 1) + torch.arange(steps, device="cuda")).view(-1)
    bonus = i * (steps + 1) + steps
    cu = ((i + 1) * steps).to(torch.int32)
    t = torch.rand((bs,), dtype=torch.float32, device="cuda") + 0.5
    t[::5] = EPS
    torch.manual_seed(3)
    out = topk_topp_sample(
        logits,
        t,
        20,
        0.95,
        verify_rows=verify,
        bonus_rows=bonus,
        cu_num_draft_tokens=cu,
        eps=EPS,
    )
    req = torch.cat(
        [torch.searchsorted(cu, torch.arange(bs * steps, device="cuda"), right=True), i]
    )
    gathered = logits.index_select(0, torch.cat([verify, bonus]))
    ref = _reference(gathered, t[req], 20, 0.95, 3, monkeypatch)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_degenerate_rows(monkeypatch):
    """Plateaus (more tied groups than one batch), masked columns, +-0."""
    x = torch.randn((8, VOCAB), dtype=torch.float32, device="cuda") * 2
    x[0] = 1.0
    x[1] = -5.0
    x[1, 1000:6000] = 3.0
    x[2] = float("-inf")
    x[3, ::7] = float("-inf")
    x[4, :21] = 50.0
    x[5, VOCAB - 21 :] = 50.0
    x[6, ::1000] = 40.0
    x[7] = 0.0
    x[7, 3] = -0.0
    logits = x.to(torch.bfloat16)
    for temp in (1.0, 0.7, EPS):
        t = torch.full((8,), temp, dtype=torch.float32, device="cuda")
        for seed in range(3):
            torch.manual_seed(seed)
            out = topk_topp_sample(logits, t, 20, 0.95, eps=EPS)
            ref = _reference(logits, t, 20, 0.95, seed, monkeypatch)
            torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_nan_rows_stay_in_range():
    x = torch.randn((3, VOCAB), dtype=torch.float32, device="cuda")
    x[0, 5] = float("nan")
    x[1, 7] = float("inf")
    logits = x.to(torch.bfloat16)
    t = torch.tensor([1.0, 1.0, EPS], dtype=torch.float32, device="cuda")
    out = topk_topp_sample(logits, t, 20, 0.95, eps=EPS)
    assert bool(((out >= 0) & (out < VOCAB)).all())


@pytest.mark.parametrize("max_extra", [3, 8])
def test_presence_penalty(max_extra, monkeypatch):
    """The penalty applied while the logits are read equals an eager
    ``bf16(x - p)`` on the penalized columns, then the sampler."""
    bs, q = 21, 4
    rows = bs * q
    logits = _logits("zipf", rows, 5)
    g = torch.Generator(device="cuda")
    g.manual_seed(5)
    nslots = 2 * bs
    mask = torch.zeros((nslots, VOCAB), dtype=torch.uint8, device="cuda")
    slots = torch.randperm(nslots, device="cuda", generator=g)[:bs].to(torch.int32)
    for i in range(bs):
        top = torch.topk(logits[i * q].float(), 40).indices
        hist = torch.randint(0, VOCAB, (2000,), device="cuda", generator=g)
        mask[slots[i].long(), torch.cat([hist, top[::3]])] = 1
    pens = torch.full((bs,), 1.5, dtype=torch.float32, device="cuda")
    pens[::4] = 0.0
    seq_start = (torch.arange(bs, device="cuda") * q).to(torch.int32)
    row_seq = torch.arange(bs, device="cuda", dtype=torch.int32).repeat_interleave(q)
    row_pos = torch.arange(q, device="cuda", dtype=torch.int32).repeat(bs)
    input_ids = torch.randint(
        0, VOCAB, (rows + 1,), device="cuda", generator=g, dtype=torch.int32
    )
    for i in range(bs):
        input_ids[q * i + 1] = torch.topk(logits[q * i + 1].float(), 2).indices[1]
    presence = {
        "input_ids": input_ids,
        "row_seq": row_seq,
        "row_pos": row_pos,
        "seq_start": seq_start,
        "slots": slots,
        "penalties": pens,
        "mask": mask,
        "max_extra": max_extra,
    }
    # eager reference penalty
    present = mask[slots.long()[row_seq.long()]] != 0
    for r in range(rows):
        s, pos = int(row_seq[r]), int(row_pos[r])
        drafts = input_ids[int(seq_start[s]) + 1 : int(seq_start[s]) + 1 + pos]
        present[r, drafts[:max_extra].long()] = True
    pen_rows = pens[row_seq.long()][:, None]
    penalized = (logits.float() - torch.where(present, pen_rows, 0.0)).to(
        torch.bfloat16
    )
    t = torch.ones((rows,), dtype=torch.float32, device="cuda")
    t[1::5] = EPS
    for seed in range(3):
        torch.manual_seed(seed)
        out = topk_topp_sample(logits, t, 20, 0.95, eps=EPS, presence=presence)
        ref = _reference(penalized, t, 20, 0.95, seed, monkeypatch)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


def _gluon_ok():
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    return get_arch() == "gfx1250"


@pytest.mark.skipif(not _gluon_ok(), reason="Gluon select: gfx1250")
@pytest.mark.parametrize("rows", [4, 16, 36, 100])
@pytest.mark.parametrize("kind", ["zipf", "peaky", "ties"])
@pytest.mark.parametrize("k", [1, 20, 64])
def test_gluon_select_same_bits(rows, kind, k):
    """gluon_select: the same tokens, columns, probabilities and Z bits as the
    Triton kernels for the same uniform."""
    logits = _logits(kind, rows, rows + k)
    t = torch.rand((rows,), dtype=torch.float32, device="cuda") + 0.5
    t[::3] = EPS
    u = torch.rand((rows,), dtype=torch.float32, device="cuda")
    kb = max(16, 1 << (k - 1).bit_length())
    d0 = torch.zeros((rows, 3 * kb + 2), dtype=torch.int32, device="cuda")
    d1 = torch.zeros_like(d0)
    o0 = topk_topp_sample(logits, t, k, 0.95, uniform=u, eps=EPS, debug=d0)
    o1 = topk_topp_sample(
        logits, t, k, 0.95, uniform=u, eps=EPS, debug=d1, gluon_select=True
    )
    torch.testing.assert_close(o1, o0, rtol=0, atol=0)
    torch.testing.assert_close(d1, d0, rtol=0, atol=0)


@pytest.mark.skipif(not _gluon_ok(), reason="Gluon select: gfx1250")
def test_gluon_select_presence_anchor():
    """presence with anchor: the same draw as with each query's first token
    marked in the mask beforehand, and the mask ends with it marked."""
    bs, q = 9, 4
    rows = bs * q
    logits = _logits("zipf", rows, 7)
    g = torch.Generator(device="cuda")
    g.manual_seed(7)
    mask = torch.zeros((2 * bs, VOCAB), dtype=torch.uint8, device="cuda")
    slots = torch.randperm(2 * bs, device="cuda", generator=g)[:bs].to(torch.int32)
    input_ids = torch.randint(
        0, VOCAB, (rows,), device="cuda", generator=g, dtype=torch.int32
    )
    for i in range(bs):
        top = torch.topk(logits[i * q].float(), 40).indices
        mask[slots[i].long(), top[::3]] = 1
        input_ids[i * q] = top[1]
        input_ids[i * q + 1] = top[2]
    seq_start = (torch.arange(bs, device="cuda") * q).to(torch.int32)
    row_seq = torch.arange(bs, device="cuda", dtype=torch.int32).repeat_interleave(q)
    row_pos = torch.arange(q, device="cuda", dtype=torch.int32).repeat(bs)
    pens = torch.full((bs,), 1.5, dtype=torch.float32, device="cuda")
    presence = {
        "input_ids": input_ids,
        "row_seq": row_seq,
        "row_pos": row_pos,
        "seq_start": seq_start,
        "slots": slots,
        "penalties": pens,
        "mask": mask,
        "max_extra": 8,
    }
    marked = mask.clone()
    marked[slots.long(), input_ids[seq_start.long()].long()] = 1
    t = torch.ones((rows,), dtype=torch.float32, device="cuda")
    u = torch.rand((rows,), dtype=torch.float32, device="cuda")
    d0 = torch.zeros((rows, 98), dtype=torch.int32, device="cuda")
    d1 = torch.zeros_like(d0)
    ref = topk_topp_sample(
        logits,
        t,
        20,
        0.95,
        uniform=u,
        eps=EPS,
        debug=d0,
        presence={**presence, "mask": marked.clone()},
        gluon_select=True,
    )
    out = topk_topp_sample(
        logits,
        t,
        20,
        0.95,
        uniform=u,
        eps=EPS,
        debug=d1,
        presence={**presence, "anchor": True},
        gluon_select=True,
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(d1, d0, rtol=0, atol=0)
    assert torch.equal(mask, marked)


@pytest.mark.skipif(not _gluon_ok(), reason="Gluon select: gfx1250")
def test_gluon_select_philox_distribution():
    """philox: each row's draw follows the kept top-p probabilities."""
    rows = 8192
    one = _logits("zipf", 1, 3)
    logits = one.expand(rows, -1).contiguous()
    t = torch.ones((rows,), dtype=torch.float32, device="cuda")
    d = torch.zeros((rows, 98), dtype=torch.int32, device="cuda")
    out = topk_topp_sample(
        logits, t, 20, 0.95, eps=EPS, debug=d, gluon_select=True, philox=(1234, 5)
    )
    cols = d[0, :20].long()
    p = d[0, 32:52].view(torch.float32).double()
    top_p = float(d[0, 96].view(torch.float32))
    keep = (torch.cumsum(p, 0) - p) < top_p
    q = torch.where(keep, p, 0.0)
    q = q / q.sum()
    counts = (out.long()[:, None] == cols[None, :]).sum(0).double()
    sigma = (rows * q * (1 - q)).sqrt().clamp_min(1.0)
    assert int(counts.sum()) == rows
    assert bool(((counts - rows * q).abs() <= 5 * sigma).all()), (counts, rows * q)
    again = topk_topp_sample(
        logits, t, 20, 0.95, eps=EPS, gluon_select=True, philox=(1234, 6)
    )
    assert not torch.equal(out, again)

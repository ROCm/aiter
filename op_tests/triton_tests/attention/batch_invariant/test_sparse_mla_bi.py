# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import hashlib
import json
import os

import pytest
import torch

from aiter.ops.triton.attention.batch_invariant.sparse_mla_bi import (
    BI_CONTRACT_VERSION,
    sparse_mla_bi_fwd,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx950", reason="sparse_mla_bi has pinned configs for gfx950 only"
)

D, R = 512, 64
ROW = 656
SCALE = (128 + 64) ** -0.5
GOLDEN = os.path.join(os.path.dirname(__file__), "sparse_mla_bi_golden.json")


def make_inputs(lens, H, n_slots, fp8=True, seed=0, block=1, pitch_pad=0):
    """Kimi-shaped inputs drawn on a CPU generator (the CUDA RNG differs between
    torch builds), moved once. Returns (q, kv, indptr, indices, deq), deq is the
    dequantized [n_slots, 576] cache in f64 for the reference."""
    g = torch.Generator().manual_seed(seed)
    C = len(lens)
    q = (torch.randn((C, H, D + R), generator=g, dtype=torch.float32) * 0.5).to(
        torch.bfloat16
    )
    lat = torch.randn((n_slots, D), generator=g, dtype=torch.float32)
    rope = torch.randn((n_slots, R), generator=g, dtype=torch.float32).to(
        torch.bfloat16
    )
    if fp8:
        g4 = lat.reshape(n_slots, 4, 128)
        sc = g4.abs().amax(-1) / 448.0
        f8 = (g4 / sc[..., None]).to(torch.float8_e4m3fn)
        deq_lat = (f8.to(torch.float32) * sc[..., None]).reshape(n_slots, D)
        deq_lat = deq_lat.to(torch.bfloat16)
        rec = torch.cat(
            [
                f8.reshape(n_slots, D).view(torch.uint8),
                sc.contiguous().view(torch.uint8),
                rope.view(torch.uint8),
            ],
            1,
        )
        width = ROW
    else:
        deq_lat = lat.to(torch.bfloat16)
        rec = torch.cat([deq_lat, rope], 1)
        width = D + R
    deq = torch.cat([deq_lat, rope], 1).to(torch.float64)
    nb = n_slots // block
    pitch = block * width + pitch_pad
    pool = torch.zeros((nb, pitch), dtype=rec.dtype)
    pool[:, : block * width] = rec.reshape(nb, block * width)
    kv = pool.cuda().as_strided((nb, block, width), (pitch, width, 1))
    indptr = torch.zeros(C + 1, dtype=torch.int64)
    indptr[1:] = torch.tensor(lens, dtype=torch.int64).cumsum(0)
    idx = [torch.randperm(n_slots, generator=g)[:n] for n in lens]
    idx = torch.cat(idx) if idx else torch.zeros(0, dtype=torch.int64)
    return (
        q.cuda(),
        kv,
        indptr.to(torch.int32).cuda(),
        idx.to(torch.int32).cuda(),
        deq,
    )


def generate_sparse_mla_bi_inputs(C, H, topk=2048, n_slots=65536, fp8=True):
    """GPU-drawn Kimi-shaped inputs for the benchmark: every row attends topk keys."""
    q = torch.randn((C, H, D + R), dtype=torch.bfloat16, device="cuda")
    if fp8:
        kv = torch.randint(0, 120, (n_slots, 1, ROW), dtype=torch.uint8, device="cuda")
        kv[:, 0, D : D + 16] = (
            torch.full((n_slots, 4), 0.01, dtype=torch.float32, device="cuda")
            .view(torch.uint8)
        )
        kv[:, 0, D + 16 :] = (
            torch.randn((n_slots, R), dtype=torch.bfloat16, device="cuda")
            .view(torch.uint8)
        )
    else:
        kv = torch.randn((n_slots, D + R), dtype=torch.bfloat16, device="cuda")
    indptr = torch.arange(C + 1, dtype=torch.int32, device="cuda") * topk
    idx = torch.randint(0, n_slots, (C * topk,), dtype=torch.int32, device="cuda")
    return {"q": q, "kv_buffer": kv, "kv_indptr": indptr, "kv_indices": idx}


def reference(q, deq, indptr, idx, sink=None):
    """FP64 softmax attention over each row's keys."""
    q = q.double().cpu()
    indptr, idx = indptr.cpu().long(), idx.cpu().long()
    C, H, _ = q.shape
    out = torch.zeros((C, H, D), dtype=torch.float64)
    lse = torch.full((C, H), float("-inf"), dtype=torch.float64)
    for r in range(C):
        ids = idx[indptr[r] : indptr[r + 1]]
        ids = ids[(ids >= 0) & (ids < deq.shape[0])]
        s = q[r] @ deq[ids].T * SCALE
        if sink is not None:
            s = torch.cat([s, sink.double().cpu()[:, None]], 1)
        if s.shape[1] == 0:
            continue
        p = torch.softmax(s, -1)
        out[r] = p[:, : len(ids)] @ deq[ids][:, :D]
        lse[r] = torch.logsumexp(s, -1)
    return out, lse


def run(inp, **kw):
    q, kv, indptr, idx, _ = inp
    return sparse_mla_bi_fwd(q, kv, indptr, idx, SCALE, **kw)


LENS = [1, 63, 64, 65, 500, 2048, 0, 1000, 129, 7]


@pytest.mark.parametrize("fp8", [True, False])
@pytest.mark.parametrize("H", [12, 96])
def test_accuracy(fp8, H):
    inp = make_inputs(LENS, H, 4096, fp8=fp8)
    out, lse = run(inp, return_lse=True)
    ref, ref_lse = reference(inp[0], inp[4], inp[2], inp[3])
    torch.testing.assert_close(out.double().cpu(), ref, atol=1e-2, rtol=2e-2)
    torch.testing.assert_close(lse.double().cpu(), ref_lse, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("H", [12, 96])
def test_against_sparse_mla_fwd(H):
    """Same tolerance as the FP64 test against AITER's one-split kernel."""
    sm = pytest.importorskip("aiter.ops.triton.attention.sparse_mla")
    inp = make_inputs(LENS, H, 4096)
    q, kv, indptr, idx, _ = inp
    out, _ = run(inp)
    ref, _ = sm.sparse_mla_fwd(q, kv, indptr, idx, SCALE, kv_splits=1)
    # sparse_mla_fwd leaves NaN (0 / 0) in a row without keys; this op writes zeros.
    rows = torch.tensor(LENS, device="cuda") > 0
    torch.testing.assert_close(
        out[rows].float(), ref[rows].float(), atol=1e-2, rtol=2e-2
    )
    assert torch.count_nonzero(out[~rows]) == 0


@pytest.mark.parametrize("fp8", [True, False])
@pytest.mark.parametrize("H", [12, 16, 96, 128])
def test_modes_same_bits(fp8, H):
    """Split + combine and fused, with buffer loads or 64-bit pointers, give the
    same bits."""
    inp = make_inputs(LENS, H, 4096, fp8=fp8)
    a, la = run(inp, _mode="split", return_lse=True)
    for mode, addr64 in (("fused", False), ("split", True), ("fused", True)):
        b, lb = run(inp, _mode=mode, _addr64=addr64, return_lse=True)
        assert torch.equal(a, b), (mode, addr64)
        assert torch.equal(la, lb), (mode, addr64)


@pytest.mark.parametrize("fp8", [True, False])
def test_row_invariance(fp8):
    """Each row alone (decode), in a big batch (prefill) and in a shuffled, repacked
    batch gives the same bits."""
    H = 96
    g = torch.Generator().manual_seed(3)
    lens = torch.randint(0, 2049, (300,), generator=g).tolist()
    lens[:4] = [2048, 1, 0, 64]
    inp = make_inputs(lens, H, 8192, fp8=fp8, seed=1)
    q, kv, indptr, idx, _ = inp
    full, _ = sparse_mla_bi_fwd(q, kv, indptr, idx, SCALE)
    perm = torch.randperm(len(lens), generator=g)
    starts = indptr[:-1].cpu()
    q_p = q[perm.cuda()]
    lens_p = torch.tensor(lens)[perm]
    ip = torch.zeros(len(lens) + 1, dtype=torch.int32)
    ip[1:] = lens_p.cumsum(0)
    idx_p = torch.cat(
        [idx[starts[i] : starts[i] + lens[i]] for i in perm.tolist()]
    ).cuda()
    out_p, _ = sparse_mla_bi_fwd(q_p, kv, ip.cuda(), idx_p, SCALE)
    assert torch.equal(out_p, full[perm.cuda()])
    for b in (1, 2, 5, 33):
        rows = list(range(7, 7 + b))
        ipb = torch.zeros(b + 1, dtype=torch.int32)
        ipb[1:] = torch.tensor([lens[i] for i in rows]).cumsum(0)
        idxb = torch.cat([idx[starts[i] : starts[i] + lens[i]] for i in rows])
        ob, _ = sparse_mla_bi_fwd(q[7 : 7 + b], kv, ipb.cuda(), idxb, SCALE)
        assert torch.equal(ob, full[7 : 7 + b]), b
    for i in range(8):
        o1, _ = sparse_mla_bi_fwd(
            q[i : i + 1],
            kv,
            torch.tensor([0, lens[i]], dtype=torch.int32, device="cuda"),
            idx[starts[i] : starts[i] + lens[i]],
            SCALE,
        )
        assert torch.equal(o1[0], full[i]), i


def test_head_count_invariance():
    """A head's bits do not depend on how many heads share the call (TP layouts)."""
    inp = make_inputs(LENS, 96, 4096)
    q, kv, indptr, idx, _ = inp
    full, _ = run(inp)
    for h0, h1 in ((0, 12), (12, 24), (84, 96), (40, 41)):
        part, _ = sparse_mla_bi_fwd(q[:, h0:h1], kv, indptr, idx, SCALE)
        assert torch.equal(part, full[:, h0:h1])


@pytest.mark.parametrize("block,pad", [(1, 0), (64, 0), (16, 64)])
def test_paged_layouts_same_bits(block, pad):
    """Paged caches, padded page pitch and strided q / out give the bits of the
    flat cache."""
    flat = make_inputs(LENS, 96, 4096, seed=5)
    paged = make_inputs(LENS, 96, 4096, seed=5, block=block, pitch_pad=pad)
    a, _ = run(flat)
    q, kv, indptr, idx, _ = paged
    qs = torch.zeros((q.shape[0], 100, 640), dtype=torch.bfloat16, device="cuda")
    qs[:, :96, :576] = q
    o = torch.zeros((q.shape[0], 96, 768), dtype=torch.bfloat16, device="cuda")
    sparse_mla_bi_fwd(qs[:, :96, :576], kv, indptr, idx, SCALE, out=o)
    assert torch.equal(o[..., :512], a)
    assert torch.count_nonzero(o[..., 512:]) == 0


def test_invalid_slots_and_sink():
    """-1 sentinels and ids past the cache are masked out (never read)."""
    inp = make_inputs(LENS, 12, 4096)
    q, kv, indptr, idx, deq = inp
    idx = idx.clone()
    idx[::3] = -1
    idx[1::7] = 4096 + 5
    idx[2::11] = 2**31 - 1
    sink = torch.linspace(-2.0, 2.0, 12, dtype=torch.float32, device="cuda")
    out, lse = sparse_mla_bi_fwd(
        q, kv, indptr, idx, SCALE, has_invalid=True, attn_sink=sink, return_lse=True
    )
    ref, ref_lse = reference(q, deq, indptr, idx, sink=sink)
    torch.testing.assert_close(out.double().cpu(), ref, atol=1e-2, rtol=2e-2)
    torch.testing.assert_close(lse.double().cpu(), ref_lse, atol=1e-4, rtol=1e-4)
    a, _ = sparse_mla_bi_fwd(q, kv, indptr, idx, SCALE, has_invalid=True, _mode="split")
    b, _ = sparse_mla_bi_fwd(q, kv, indptr, idx, SCALE, has_invalid=True, _mode="fused")
    assert torch.equal(a, b)


def test_int64_index_stream_same_bits():
    inp = make_inputs(LENS, 12, 4096)
    q, kv, indptr, idx, _ = inp
    a, _ = run(inp)
    b, _ = sparse_mla_bi_fwd(q, kv, indptr.long(), idx.long(), SCALE)
    assert torch.equal(a, b)


@pytest.mark.parametrize("C", [4, 256])
def test_cuda_graph_replay(C):
    lens = [(i * 37) % 2049 for i in range(C)]
    inp = make_inputs(lens, 96, 8192, seed=2)
    q, kv, indptr, idx, _ = inp
    eager, _ = run(inp)
    out = torch.empty_like(eager)
    run(inp, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sparse_mla_bi_fwd(q, kv, indptr, idx, SCALE, out=out)
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, eager)


def test_zero_rows():
    q = torch.zeros((0, 12, 576), dtype=torch.bfloat16, device="cuda")
    kv = torch.zeros((16, 1, ROW), dtype=torch.uint8, device="cuda")
    ip = torch.zeros(1, dtype=torch.int32, device="cuda")
    idx = torch.zeros(0, dtype=torch.int32, device="cuda")
    out, lse = sparse_mla_bi_fwd(q, kv, ip, idx, SCALE, return_lse=True)
    assert out.shape == (0, 12, 512) and lse.shape == (0, 12)


GOLDEN_CASES = {
    "fp8_H96_C10": dict(lens=LENS, H=96, fp8=True),
    "fp8_H12_C64": dict(lens=[(i * 131) % 2049 for i in range(64)], H=12, fp8=True),
    "bf16_H96_C10": dict(lens=LENS, H=96, fp8=False),
}


def golden_digest(case):
    c = GOLDEN_CASES[case]
    inp = make_inputs(c["lens"], c["H"], 4096, fp8=c["fp8"], seed=11)
    out, lse = run(inp, return_lse=True)
    h = hashlib.sha256(out.view(torch.int16).cpu().numpy().tobytes())
    h.update(lse.view(torch.int32).cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize("case", list(GOLDEN_CASES))
def test_golden_bits(case):
    """The output bits are a contract: a change must bump BI_CONTRACT_VERSION and the
    golden file. The inputs are drawn on the CPU."""
    with open(GOLDEN) as f:
        golden = json.load(f)
    assert golden["contract_version"] == BI_CONTRACT_VERSION
    assert golden_digest(case) == golden["gfx950"][case]


def _refuse(match, **kw):
    inp = make_inputs([3, 5], 12, 64)
    q, kv, indptr, idx, _ = inp
    args = dict(q=q, kv_buffer=kv, kv_indptr=indptr, kv_indices=idx)
    args.update({k: v for k, v in kw.items() if k in args})
    extra = {k: v for k, v in kw.items() if k not in args}
    with pytest.raises(ValueError, match=match):
        sparse_mla_bi_fwd(
            args["q"], args["kv_buffer"], args["kv_indptr"], args["kv_indices"],
            SCALE, **extra
        )


def test_refusals():
    dev = "cuda"
    _refuse("kv_scale", kv_scale=torch.ones(1, dtype=torch.float32, device=dev))
    _refuse("kv_lora_rank", kv_lora_rank=256)
    _refuse("kv_lora_rank", qk_rope_head_dim=0)
    _refuse("dot_precision", dot_precision="fp8")
    _refuse("q must", q=torch.zeros((2, 12, 576), dtype=torch.float16, device=dev))
    _refuse("q must", q=torch.zeros((2, 12, 512), dtype=torch.bfloat16, device=dev))
    _refuse("H must", q=torch.zeros((2, 129, 576), dtype=torch.bfloat16, device=dev))
    _refuse(
        "unsupported kv_buffer",
        kv_buffer=torch.zeros((8, 1, 600), dtype=torch.uint8, device=dev),
    )
    _refuse(
        "bf16 cache",
        kv_buffer=torch.zeros((8, 512), dtype=torch.bfloat16, device=dev),
    )
    odd_pitch = torch.zeros((8, 658), dtype=torch.uint8, device=dev)
    _refuse("4 B", kv_buffer=odd_pitch.as_strided((8, 1, 656), (658, 656, 1)))
    _refuse("C \\+ 1", kv_indptr=torch.zeros(5, dtype=torch.int32, device=dev))
    _refuse("int32", kv_indices=torch.zeros(8, dtype=torch.float32, device=dev))
    _refuse("is on", kv_indices=torch.zeros(8, dtype=torch.int32, device="cpu"))
    _refuse("out must", out=torch.zeros((2, 12, 256), dtype=torch.bfloat16, device=dev))
    _refuse("attn_sink", attn_sink=torch.zeros(5, dtype=torch.float32, device=dev))
    _refuse("_mode", _mode="auto")

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx11 paged attention (the vLLM unified_attention form) on the FlyDSL kernels:
``flydsl_unified_attention_gfx11`` against an fp32 reference that gathers the
keys through the block table, over prefills, chunks over a longer prefix, mixed
prefill + decode batches, speculative decode and pure decode batches (the
decode rows take the segmented path and the FlyDSL merge)."""

import pytest
import torch

try:
    from aiter.ops.flydsl import paged_attention_gfx11 as pa
except Exception:  # noqa: BLE001  FlyDSL / aiter import problems surface as skips
    pa = None


def _arch() -> str:
    if not torch.cuda.is_available():
        return ""
    return torch.cuda.get_device_properties(0).gcnArchName.lower().split(":")[0]


_gfx11_only = pytest.mark.skipif(
    pa is None or not _arch().startswith(("gfx110", "gfx115")),
    reason="FlyDSL gfx11 paged attention runs on gfx110x / gfx115x",
)


def _ref_paged(query, k_cache, v_cache, cu, sk, bt, scale):
    """fp32 reference: per sequence, gather the cache rows through the block
    table, bottom-right causal mask, softmax, PV; a row without keys gives 0."""
    total, H, D = query.shape
    HKV = k_cache.shape[2]
    page = k_cache.shape[1]
    G = H // HKV
    out = torch.zeros(total, H, D, dtype=torch.float32, device=query.device)
    for b in range(sk.numel()):
        q0, q1 = cu[b].item(), cu[b + 1].item()
        kv = sk[b].item()
        if q1 == q0 or kv == 0:
            continue
        t = torch.arange(kv, device=query.device)
        rows = bt[b, t // page].long() * page + (t % page)
        k = k_cache.view(-1, HKV, D)[rows].float().repeat_interleave(G, dim=1)
        v = v_cache.view(-1, HKV, D)[rows].float().repeat_interleave(G, dim=1)
        q = query[q0:q1].float()
        lq = q1 - q0
        s = torch.einsum("qhd,khd->hqk", q, k) * scale
        qi = torch.arange(lq, device=query.device)[:, None]
        ki = torch.arange(kv, device=query.device)[None, :]
        s = s.masked_fill((ki > qi + (kv - lq))[None], float("-inf"))
        p = torch.softmax(s, dim=-1).nan_to_num(0.0)
        out[q0:q1] = torch.einsum("hqk,khd->qhd", p, v)
    return out


def _make(seq_lens, H, HKV, D, page, dtype):
    """Random q and a paged cache whose block table maps every sequence's pages
    to distinct blocks in a shuffled order."""
    g = torch.Generator(device="cuda").manual_seed(0)
    total_q = sum(a for a, _ in seq_lens)
    pages = [-(-kv // page) for _, kv in seq_lens]
    max_pages = max(max(pages), 1)
    nblk = sum(pages) + 4
    q = torch.randn(total_q, H, D, device="cuda", dtype=dtype, generator=g)
    k_cache = torch.randn(nblk, page, HKV, D, device="cuda", dtype=dtype, generator=g)
    v_cache = torch.randn(nblk, page, HKV, D, device="cuda", dtype=dtype, generator=g)
    perm = torch.randperm(nblk, generator=torch.Generator().manual_seed(1)).tolist()
    bt = torch.zeros(len(seq_lens), max_pages, dtype=torch.int32)
    nxt = 0
    for b, n in enumerate(pages):
        for p in range(n):
            bt[b, p] = perm[nxt]
            nxt += 1
    bt = bt.cuda()
    cu = torch.tensor(
        [0] + [a for a, _ in seq_lens], dtype=torch.int32, device="cuda"
    ).cumsum(0, dtype=torch.int32)
    sk = torch.tensor([kv for _, kv in seq_lens], dtype=torch.int32, device="cuda")
    return q, k_cache, v_cache, cu, sk, bt


def _check(seq_lens, H, HKV, D, page, dtype):
    q, k_cache, v_cache, cu, sk, bt = _make(seq_lens, H, HKV, D, page, dtype)
    out = torch.zeros_like(q)
    scale = D**-0.5
    assert pa.gfx11_paged_supported(
        q,
        k_cache,
        v_cache,
        out,
        cu,
        sk,
        bt,
        True,
        (-1, -1),
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    max_q = max(a for a, _ in seq_lens)
    max_k = max(kv for _, kv in seq_lens)
    res = pa.flydsl_unified_attention_gfx11(
        q, k_cache, v_cache, out, cu, max_q, sk, max_k, scale, bt
    )
    assert res is out
    torch.cuda.synchronize()
    ref = _ref_paged(q, k_cache, v_cache, cu, sk, bt, scale)
    got = out.float()
    assert not torch.isnan(got).any()
    # bf16 output of a bf16 kernel against fp32 math: 1.5e-2 absolute, as the
    # Triton unified_attention test's tolerance, plus 1e-2 relative.
    err = (got - ref).abs()
    bad = err > 1.5e-2 + 1e-2 * ref.abs()
    assert (
        not bad.any()
    ), f"{bad.float().mean().item():.3%} of the outputs off; max abs {err.max().item():.3g}"


@_gfx11_only
@pytest.mark.parametrize(
    "seq_lens, H, HKV, D, page, dtype_str",
    [
        # prefills and chunks over a longer prefix (prefill kernel)
        ([(512, 512)], 8, 2, 256, 16, "bf16"),
        ([(1280, 1280)], 8, 2, 256, 32, "bf16"),
        ([(1024, 2048)], 8, 2, 256, 32, "bf16"),
        ([(129, 2049), (37, 1000), (200, 200), (5, 5)], 8, 2, 256, 16, "bf16"),
        ([(300, 300), (700, 1500)], 32, 8, 128, 16, "bf16"),
        ([(200, 200), (16, 16)], 16, 16, 64, 16, "f16"),
        ([(100, 100), (33, 77)], 4, 4, 40, 16, "bf16"),  # head_dim % 16 == 8
        # mixed: the prefill kernel takes the long sequences, the decode kernel
        # the short ones (segments + merge, the prefill rows left alone)
        ([(1280, 1280)] + [(1, 1536)] * 31, 8, 2, 256, 32, "bf16"),
        ([(2048, 2048)] + [(1, 2048)] * 15, 8, 2, 256, 16, "bf16"),
        (
            [(37, 1000), (3, 17), (200, 200), (1, 1), (129, 2049), (5, 5)],
            8,
            2,
            256,
            16,
            "bf16",
        ),
        ([(600, 600), (2, 900), (1, 1), (4, 4)], 32, 8, 128, 16, "bf16"),
        # speculative decode and pure decode (segments + merge)
        ([(4, 2048)] * 32, 8, 2, 256, 16, "bf16"),
        ([(4, 512)] * 8, 8, 2, 256, 32, "bf16"),
        ([(3, 100), (4, 100), (2, 2), (4, 4), (3, 17)], 8, 2, 256, 16, "bf16"),
        ([(1, 1536)], 8, 2, 256, 16, "bf16"),
        ([(1, 2048)] * 8, 8, 2, 256, 32, "bf16"),
        ([(1, 1536)] * 32, 8, 2, 256, 16, "bf16"),
        ([(1, 6400)] * 4, 8, 2, 256, 32, "bf16"),
        (
            [
                (1, 1),
                (1, 7),
                (1, 16),
                (1, 17),
                (1, 33),
                (1, 1000),
                (1, 1023),
                (1, 1025),
            ],
            8,
            2,
            256,
            16,
            "bf16",
        ),
        ([(1, 4096)] * 2, 32, 8, 128, 16, "bf16"),
        ([(1, 1500)] * 16, 16, 16, 64, 32, "f16"),
        ([(1, 300)] * 4, 16, 1, 128, 16, "bf16"),  # MQA: 16 heads on one KV head
    ],
)
def test_flydsl_paged_attention_gfx11(seq_lens, H, HKV, D, page, dtype_str):
    dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float16
    _check(seq_lens, H, HKV, D, page, dtype)


@_gfx11_only
def test_flydsl_paged_attention_gfx11_gate():
    """The gate takes dense bf16 / f16 caches with int32 tables and refuses
    what the kernels have no code for."""
    q, k_cache, v_cache, cu, sk, bt = _make([(16, 64)], 8, 2, 128, 16, torch.bfloat16)
    out = torch.empty_like(q)
    ok = lambda **kw: pa.gfx11_paged_supported(  # noqa: E731
        kw.get("q", q),
        kw.get("k", k_cache),
        kw.get("v", v_cache),
        kw.get("out", out),
        kw.get("cu", cu),
        kw.get("sk", sk),
        kw.get("bt", bt),
        kw.get("causal", True),
        kw.get("window", (-1, -1)),
        kw.get("softcap", 0),
        kw.get("qd", None),
        kw.get("kd", None),
        kw.get("vd", None),
        kw.get("alibi", None),
        kw.get("sinks", None),
        kw.get("qq", None),
        kw.get("oscale", None),
    )
    assert ok()
    assert not ok(causal=False)
    assert not ok(window=(64, -1))
    assert not ok(softcap=30.0)
    assert not ok(sinks=torch.ones(8, device="cuda"))
    assert not ok(kd=torch.ones(1, device="cuda"))
    assert not ok(
        k=(
            k_cache.to(torch.float8_e4m3fnuz)
            if hasattr(torch, "float8_e4m3fnuz")
            else k_cache.float()
        )
    )
    assert not ok(cu=cu.to(torch.int64))
    assert not ok(bt=bt.t().contiguous())  # wrong shape
    assert not ok(q=q.transpose(0, 1))  # not packed
    k24 = torch.randn(k_cache.shape[0], 24, 2, 128, device="cuda", dtype=torch.bfloat16)
    assert ok(k=k24, v=k24)  # any page size that is a multiple of 8
    k20 = torch.randn(k_cache.shape[0], 20, 2, 128, device="cuda", dtype=torch.bfloat16)
    assert not ok(k=k20, v=k20)

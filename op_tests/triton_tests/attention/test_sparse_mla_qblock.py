# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FlyDSL sparse MLA correctness for the GLM-5.3-Flash contract."""

from __future__ import annotations

import pytest
import torch

from aiter import dtypes
from aiter.ops.flydsl.sparse_mla_qblock_kernels import (
    sparse_mla_one_query_fwd_flydsl,
    sparse_mla_qblock_fwd_flydsl,
)
from aiter.ops.triton.attention.pa_decode_sparse import (
    _qblock_reuse_wins,
    pa_decode_sparse,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import assertAllclose

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx942",
    reason="FlyDSL sparse MLA is gfx942-only (64 KiB LDS)",
)

H, D = 16, 512


def _indices(pattern: str, n_tok: int, pool: int, topk: int):
    if pattern == "shared":
        return torch.arange(topk, dtype=torch.int32, device="cuda").repeat(n_tok)
    seed = torch.randint(0, pool, (n_tok, 1), dtype=torch.int32, device="cuda")
    k = torch.arange(topk, dtype=torch.int32, device="cuda")
    return ((k * 17 + seed) % pool).reshape(-1).contiguous()


def _indptr(n_tok, topk):
    return torch.arange(0, (n_tok + 1) * topk, topk, device="cuda", dtype=torch.int32)


def _torch_flat(q, kv, indices, topk, sink=None, scale=None):
    n_tok = q.shape[0]
    if scale is None:
        scale = q.shape[-1] ** -0.5
    gathered = kv[indices.view(n_tok, topk).long()].float()
    scores = torch.einsum("thd,tkd->thk", q.float(), gathered) * scale
    if sink is None:
        probs = torch.softmax(scores, dim=-1)
    else:
        sink_b = sink.float().view(1, -1, 1).expand(n_tok, q.shape[1], 1)
        probs = torch.softmax(torch.cat([scores, sink_b], dim=-1), dim=-1)[..., :topk]
    return torch.einsum("thk,tkd->thd", probs, gathered).to(q.dtype)


def _torch_csr(q, kv, indices, indptr, scale, sink):
    rows = []
    for t in range(q.shape[0]):
        gathered = kv[indices[indptr[t] : indptr[t + 1]].long()].float()
        scores = torch.einsum("hd,kd->hk", q[t].float(), gathered) * scale
        sink_b = sink.float().view(-1, 1)
        probs = torch.softmax(torch.cat([scores, sink_b], dim=-1), dim=-1)[:, :-1]
        rows.append(torch.einsum("hk,kd->hd", probs, gathered))
    return torch.stack(rows).to(q.dtype)


@pytest.mark.parametrize("n_tok", [7, 16])
@pytest.mark.parametrize("pattern", ["shared", "random"])
@pytest.mark.parametrize("kernel", ["one_q", "qblock"])
def test_flydsl_matches_torch(n_tok, pattern, kernel):
    torch.manual_seed(0)
    topk, pool = 2048, 8192
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices(pattern, n_tok, pool, topk)
    out = torch.empty_like(q)
    indptr = _indptr(n_tok, topk)
    if kernel == "one_q":
        sparse_mla_one_query_fwd_flydsl(q, kv, indices, out, kv_indptr=indptr)
    else:
        sparse_mla_qblock_fwd_flydsl(q, kv, indices, out, kv_indptr=indptr)
    ref = _torch_flat(q, kv, indices, topk)
    assertAllclose(ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("kernel", ["one_q", "qblock"])
def test_flydsl_attn_sink(kernel):
    torch.manual_seed(3)
    n_tok, topk, pool = 8, 64, 8192
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices("shared", n_tok, pool, topk)
    sink = torch.randn(H, dtype=torch.float32, device="cuda")
    scale = 0.037
    out = torch.empty_like(q)
    common = {
        "kv_indptr": _indptr(n_tok, topk),
        "attn_sink": sink,
        "softmax_scale": scale,
    }
    if kernel == "one_q":
        sparse_mla_one_query_fwd_flydsl(q, kv, indices, out, **common)
    else:
        sparse_mla_qblock_fwd_flydsl(q, kv, indices, out, block_q=4, **common)
    ref = _torch_flat(q, kv, indices, topk, sink, scale)
    assertAllclose(ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("pattern", ["shared"])
def test_pa_decode_sparse_dispatch(pattern):
    torch.manual_seed(2)
    n_tok, topk, pool = 16, 256, 8192
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices(pattern, n_tok, pool, topk)
    sink = torch.zeros(H, dtype=torch.float32, device="cuda")
    scale = D**-0.5
    out = pa_decode_sparse(
        q,
        kv,
        indices,
        _indptr(n_tok, topk),
        sink,
        scale,
        has_invalid=False,
        kv_splits=1,
    )
    ref = _torch_flat(q, kv, indices, topk, sink, scale)
    assertAllclose(ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=1e-2, atol=1e-2)


def test_qblock_probe_requires_every_row():
    topk = 16
    shared = _indices("shared", 4, 64, topk)
    assert _qblock_reuse_wins(shared, 4, topk)
    rows = torch.stack(
        [(torch.arange(topk, device="cuda") + t * 8) % 64 for t in range(4)]
    ).to(torch.int32)
    rows[3] = rows[0]
    assert not _qblock_reuse_wins(rows.reshape(-1), 4, topk)


def test_unpacked_csr_falls_back():
    torch.manual_seed(4)
    n_tok, topk, pool = 2, 16, 64
    q = torch.randn(n_tok, H, D, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    rows = _indices("random", n_tok, pool, topk).view(n_tok, topk)
    indices = torch.cat(
        [rows.reshape(-1), torch.zeros(32, dtype=torch.int32, device="cuda")]
    )
    indptr = _indptr(n_tok, topk)
    sink = torch.zeros(H, dtype=torch.float32, device="cuda")
    scale = D**-0.5
    out = pa_decode_sparse(
        q, kv, indices, indptr, sink, scale, has_invalid=False, kv_splits=1
    )
    lying = torch.tensor([0, 8, 32], dtype=torch.int32, device="cuda")
    hinted = torch.arange(32, dtype=torch.int32, device="cuda")
    out_hint = pa_decode_sparse(
        q,
        kv,
        hinted,
        lying,
        sink,
        scale,
        has_invalid=False,
        kv_splits=1,
        uniform_topk=16,
    )
    assertAllclose(
        _torch_flat(q, kv, rows.reshape(-1), topk, sink, scale).to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
    )
    assertAllclose(
        _torch_csr(q, kv, hinted, lying, scale, sink).to(dtypes.fp32),
        out_hint.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
    )


def test_noncontiguous_q_falls_back():
    torch.manual_seed(5)
    n_tok, topk, pool = 4, 16, 64
    buf = torch.randn(n_tok, H, D + 8, dtype=torch.bfloat16, device="cuda")
    q = buf[:, :, :D]
    assert not q.is_contiguous()
    kv = torch.randn(pool, D, dtype=torch.bfloat16, device="cuda")
    indices = _indices("shared", n_tok, pool, topk)
    sink = torch.zeros(H, dtype=torch.float32, device="cuda")
    scale = D**-0.5
    out = pa_decode_sparse(
        q,
        kv,
        indices,
        _indptr(n_tok, topk),
        sink,
        scale,
        has_invalid=False,
        kv_splits=1,
    )
    ref = _torch_flat(q.contiguous(), kv, indices, topk, sink, scale)
    assertAllclose(ref.to(dtypes.fp32), out.to(dtypes.fp32), rtol=1e-2, atol=1e-2)

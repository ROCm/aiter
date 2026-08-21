# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness for the DSv4 sparse-MLA training backward against torch autograd.

The reference is a differentiable fp32 re-implementation of the V4 forward, differentiated by
autograd -- an independent path from the kernels, not a re-expression of them. It is a per-token
python loop, so the shapes here are deliberately small; the kernels are exercised at production
shapes (T=4096, H=128, topk=512) out of tree.
"""

import pytest
import torch

from aiter.ops.triton.attention.sparse_attention_dsv4_bwd import sparse_mla_bwd_dsv4
from aiter.ops.triton.utils._triton import arch_info

D = 512
COS_TOL = 0.999

pytestmark = pytest.mark.skipif(
    arch_info.get_arch() != "gfx950",
    reason="DSv4 sparse-MLA backward is gfx950 (CDNA4) only",
)


def _ref_fwd_diff(q, kv, attn_sink, topk, scale):
    """Differentiable fp32 V4 forward. q/kv/attn_sink require grad. Returns O [T, H, D]."""
    outs = []
    for t in range(q.shape[0]):
        idx = topk[t].long()
        valid = idx != -1
        k = kv[idx.clamp(min=0)]
        k = torch.where(valid[:, None], k, torch.zeros_like(k))
        s = (q[t] @ k.t()) * scale
        s = torch.where(valid[None, :], s, torch.full_like(s, float("-inf")))
        if attn_sink is None:
            m = s.max(dim=1).values
            p = torch.where(
                valid[None, :], torch.exp(s - m[:, None]), torch.zeros_like(s)
            )
            denom = p.sum(dim=1)
        else:
            m = torch.maximum(s.max(dim=1).values, attn_sink)
            p = torch.where(
                valid[None, :], torch.exp(s - m[:, None]), torch.zeros_like(s)
            )
            denom = p.sum(dim=1) + torch.exp(attn_sink - m)
        outs.append((p @ k) / denom[:, None])
    return torch.stack(outs, dim=0)


def _ref_fwd(q, kv, attn_sink, topk, scale):
    """Non-differentiable forward giving the kernel inputs O (bf16) and sink-inclusive lse."""
    with torch.no_grad():
        o = _ref_fwd_diff(q.float(), kv.float(), attn_sink, topk, scale)
        lse = torch.empty(q.shape[0], q.shape[1], device=q.device, dtype=torch.float32)
        for t in range(q.shape[0]):
            idx = topk[t].long()
            valid = idx != -1
            k = kv.float()[idx.clamp(min=0)]
            k = torch.where(valid[:, None], k, torch.zeros_like(k))
            s = (q.float()[t] @ k.t()) * scale
            s = torch.where(valid[None, :], s, torch.full_like(s, float("-inf")))
            m = s.max(dim=1).values
            p = torch.where(
                valid[None, :], torch.exp(s - m[:, None]), torch.zeros_like(s)
            )
            denom = p.sum(dim=1)
            if attn_sink is not None:
                denom = denom + torch.exp(attn_sink - m)
            lse[t] = m + torch.log(denom)
    return o.to(torch.bfloat16).contiguous(), lse


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()


@pytest.mark.parametrize(
    "T, H, topk, npool, has_sink, r_chunk",
    [
        (128, 64, 128, 0, True, None),  # no pool, unchunked
        (128, 128, 128, 0, True, None),  # H=128
        (128, 128, 128, 64, True, None),  # compressed pool (num_kv > T)
        (128, 64, 128, 0, False, None),  # no attn_sink
        (128, 128, 128, 64, True, 64),  # chunked: dQ RMW + per-chunk CSR build
    ],
)
def test_sparse_mla_bwd_dsv4(T, H, topk, npool, has_sink, r_chunk):
    torch.manual_seed(0)
    dev = "cuda"
    num_kv = T + npool
    scale = 1.0 / (D**0.5)

    q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(num_kv, D, device=dev, dtype=torch.bfloat16)
    do = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
    sink = (torch.randn(H, device=dev, dtype=torch.float32) * 0.1) if has_sink else None

    indices = torch.randint(0, num_kv, (T, topk), dtype=torch.int32, device=dev)
    invalid = torch.rand(T, topk, device=dev) < 0.1
    indices = torch.where(invalid, torch.full_like(indices, -1), indices).contiguous()

    o, lse = _ref_fwd(q, kv, sink, indices, scale)

    dq, dkv, d_sink = sparse_mla_bwd_dsv4(
        q, kv, do, o, lse, indices, attn_sink=sink, scale=scale, R_CHUNK=r_chunk
    )

    qg = q.float().clone().requires_grad_(True)
    kvg = kv.float().clone().requires_grad_(True)
    sg = sink.clone().requires_grad_(True) if has_sink else None
    _ref_fwd_diff(qg, kvg, sg, indices, scale).backward(do.float())

    assert _cos(dq, qg.grad) > COS_TOL, f"dq cos {_cos(dq, qg.grad)}"
    assert _cos(dkv, kvg.grad) > COS_TOL, f"dkv cos {_cos(dkv, kvg.grad)}"
    if has_sink:
        assert _cos(d_sink, sg.grad) > COS_TOL, f"d_sink cos {_cos(d_sink, sg.grad)}"
    else:
        assert d_sink is None


def _dummy_inputs(T=64, H=64, topk=64, dev="cuda"):
    return {
        "q": torch.randn(T, H, D, device=dev, dtype=torch.bfloat16),
        "kv": torch.randn(T, D, device=dev, dtype=torch.bfloat16),
        "do": torch.randn(T, H, D, device=dev, dtype=torch.bfloat16),
        "o": torch.randn(T, H, D, device=dev, dtype=torch.bfloat16),
        "lse": torch.randn(T, H, device=dev, dtype=torch.float32),
        "idx": torch.randint(0, T, (T, topk), dtype=torch.int32, device=dev),
    }


def test_sparse_mla_bwd_dsv4_rejects_bad_chunk():
    """R_CHUNK must be a multiple of the mfma tile width."""
    t = _dummy_inputs(topk=64)
    with pytest.raises(AssertionError, match="multiple of 32"):
        sparse_mla_bwd_dsv4(
            t["q"], t["kv"], t["do"], t["o"], t["lse"], t["idx"], R_CHUNK=48
        )


def test_sparse_mla_bwd_dsv4_rejects_short_topk():
    """topk_indices must have one row per query token.

    The dQ grid is sized from `q`, so a shorter index tensor is read past its end rather than
    producing a shape error anywhere downstream.
    """
    t = _dummy_inputs(T=64, topk=64)
    with pytest.raises(AssertionError, match="topk_indices must be"):
        sparse_mla_bwd_dsv4(t["q"], t["kv"], t["do"], t["o"], t["lse"], t["idx"][:32])


def test_sparse_mla_bwd_dsv4_rejects_indivisible_chunk():
    """A chunk width that is a valid tile multiple but does not divide TOPK is still rejected.

    96 = 3*32 so it clears the tile-width check, but 128 % 96 == 32 would leave a 32-wide tail
    chunk. The kernels take the chunk width as a constexpr and would read past the end of each
    top-k row, silently, so this is an error rather than a handled case.
    """
    t = _dummy_inputs(topk=128)
    with pytest.raises(AssertionError, match="must divide TOPK"):
        sparse_mla_bwd_dsv4(
            t["q"], t["kv"], t["do"], t["o"], t["lse"], t["idx"], R_CHUNK=96
        )

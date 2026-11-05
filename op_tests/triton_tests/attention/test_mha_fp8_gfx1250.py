# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass, fields

import pytest
import torch

from aiter.ops.triton.attention.mha_fp8 import mha_fwd, mha_varlen_fwd
from aiter.ops.triton.utils._triton.arch_info import get_arch

arch = get_arch()


# ---------------
# reference
# ---------------


def mha_torch(q, k, v, sm_scale, causal):
    # GQA: broadcast each K/V head to the group of query heads that shares it
    group_size = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group_size, dim=1)
    v = v.repeat_interleave(group_size, dim=1)
    qk = q @ k.transpose(-2, -1)
    qk = qk * sm_scale
    if causal:
        # bottom-right aligned: query i attends to keys j <= i + (SK - SQ)
        sq, sk = q.shape[2], k.shape[2]
        M = torch.tril(torch.ones((sq, sk), device=q.device), diagonal=sk - sq)
        qk[:, :, M == 0] = float("-inf")
    p = torch.exp(qk - qk.amax(-1, keepdim=True))
    l = p.sum(-1, keepdim=True)
    p_q = p.to(torch.float8_e4m3fn).to(torch.float32)
    o = (p_q @ v) / l
    return o


# ---------------
# initialize data
# ---------------


def quant_fp8(x):
    """Per-tensor fp8 e4m3 quantization; returns ``(fp8 tensor, fp32 scale)``."""
    scale = x.abs().amax().to(torch.float32) / 448.0
    return (x / scale).to(torch.float8_e4m3fn), scale


def init_qkv(b, hq, hk, sq, sk, d, device="cuda"):
    """fp8 q/k/v in the requested layout, plus their per-tensor scales."""
    q_shape = (b, hq, sq, d)
    kv_shape = (b, hk, sk, d)
    q, q_scale = quant_fp8(
        torch.randn(q_shape, device=device, dtype=torch.float32) * 0.5
    )
    k, k_scale = quant_fp8(
        torch.randn(kv_shape, device=device, dtype=torch.float32) * 0.5
    )
    v, v_scale = quant_fp8(
        torch.randn(kv_shape, device=device, dtype=torch.float32) * 0.5
    )
    return q, k, v, q_scale, k_scale, v_scale


# ---------------
# unit tests
# ---------------


@dataclass
class Case:
    b: int
    hq: int
    hk: int
    sq: int
    sk: int
    d: int


@pytest.mark.parametrize(
    ", ".join(f.name for f in fields(Case)),
    [
        tuple(getattr(case, f.name) for f in fields(Case))
        for case in [
            # one K/V tile, one M tile: the shortest path through the pipeline
            Case(1, 2, 2, 64, 64, 128),
            Case(1, 1, 1, 128, 128, 128),
            Case(1, 8, 8, 256, 256, 128),
            Case(2, 8, 8, 512, 512, 128),
            Case(2, 8, 8, 512, 512, 64),
            Case(4, 32, 32, 1024, 1024, 128),
            Case(2, 16, 16, 2048, 2048, 128),
            Case(1, 8, 8, 4096, 4096, 128),
            Case(3, 5, 5, 1024, 1024, 128),
            # M not a multiple of BLOCK_M: the last program writes a partial tile
            Case(2, 4, 4, 600, 600, 128),
            # HEAD_DIM != HEAD_DIM_POW2: the descriptors pad the head dim to 128
            Case(2, 8, 8, 512, 512, 96),
            # GQA: several query heads share one K/V head
            Case(2, 8, 2, 512, 512, 128),
            Case(4, 32, 8, 1024, 1024, 128),
            Case(2, 16, 4, 1024, 1024, 64),
            Case(1, 48, 8, 2048, 2048, 128),
            # seqlen_q < seqlen_k
            Case(2, 8, 2, 512, 1024, 128),
            Case(2, 4, 4, 600, 700, 128),
            # seqlen_q > seqlen_k
            Case(1, 8, 8, 2048, 512, 128),
        ]
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_mha_fp8(b, hq, hk, sq, sk, d, causal, device="cuda"):
    if get_arch() != "gfx1250":
        pytest.skip("Only supported on gfx1250")
    if causal and sq > sk:
        pytest.skip("Causal requires seqlen_k >= seqlen_q")
    torch.manual_seed(0)

    q_tri, k_tri, v_tri, q_scale, k_scale, v_scale = init_qkv(
        b, hq, hk, sq, sk, d, device=device
    )
    q_ref = q_tri.to(torch.float32) * q_scale
    k_ref = k_tri.to(torch.float32) * k_scale
    v_ref = v_tri.to(torch.float32) * v_scale
    sm_scale = 1.0 / (d**0.5)

    o_ref = mha_torch(q_ref, k_ref, v_ref, sm_scale, causal=causal)

    out = torch.empty(q_tri.shape, device=device, dtype=torch.bfloat16)
    o_tri = mha_fwd(
        q_tri,
        k_tri,
        v_tri,
        q_scale,
        k_scale,
        v_scale,
        o=out,
        sm_scale=sm_scale,
        causal=causal,
    )
    o_tri = o_tri.float()

    torch.testing.assert_close(o_tri, o_ref, atol=0.02, rtol=0.02)


def mha_varlen_torch(q, k, v, sm_scale, causal):
    qk = torch.einsum("qhd,khd->hqk", q, k) * sm_scale
    if causal:
        sq, sk = q.shape[0], k.shape[0]
        mask = torch.ones((sq, sk), dtype=torch.bool, device=q.device).tril(sk - sq)
        qk = qk.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(qk, -1)
    m = qk.amax(-1, keepdim=True).nan_to_num(neginf=0.0)
    p = torch.exp(qk - m)
    l = p.sum(-1, keepdim=True)
    # The kernel rounds the max-relative P (scaled to the e4m3 range) to fp8 before P @ V.
    p = (p * 448.0).to(torch.float8_e4m3fn).to(torch.float32) / (
        448.0 * l.clamp_min(1e-30)
    )
    return torch.einsum("hqk,khd->qhd", p, v), lse


@pytest.mark.parametrize(
    "seqlens_q, seqlens_k",
    [
        ([128], [128]),
        ([600], [600]),
        ([1024, 77, 300], [1024, 77, 300]),
        ([256, 100], [2048, 900]),
        # short queries
        ([64], [8192]),
        ([16, 130], [4096, 6000]),
    ],
)
@pytest.mark.parametrize("heads", [12, 16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("pad", [0, 50])
def test_mha_varlen_fp8(seqlens_q, seqlens_k, heads, causal, pad, device="cuda"):
    if get_arch() != "gfx950":
        pytest.skip("Only supported on gfx950")
    D_NOPE, D_PE, D_V = 128, 64, 128
    torch.manual_seed(0)
    cu_q = torch.tensor([0] + seqlens_q, device=device).cumsum(0).to(torch.int32)
    cu_k = torch.tensor([0] + seqlens_k, device=device).cumsum(0).to(torch.int32)
    tq, tk = sum(seqlens_q), sum(seqlens_k)
    q, q_scale = quant_fp8(
        torch.randn(
            (tq + pad, heads, D_NOPE + D_PE), dtype=torch.float32, device=device
        )
    )
    k, k_scale = quant_fp8(
        torch.randn((tk, heads, D_NOPE + D_PE), dtype=torch.float32, device=device)
    )
    v, v_scale = quant_fp8(
        torch.randn((tk, heads, D_V), dtype=torch.float32, device=device)
    )
    q_scale, k_scale, v_scale = (s.reshape(1) for s in (q_scale, k_scale, v_scale))
    sm_scale = (D_NOPE + D_PE) ** -0.5
    out = torch.full((tq + pad, heads, D_V), 7.0, dtype=torch.bfloat16, device=device)
    lse = torch.full((heads, tq + pad), 7.0, dtype=torch.float32, device=device)

    mha_varlen_fwd(
        q, k, v, cu_q, cu_k, max(seqlens_q), max(seqlens_k),
        q_scale, k_scale, v_scale,
        softmax_scale=sm_scale, causal=causal, return_lse=True, out=out, lse=lse,
    )  # fmt: skip

    qf, kf, vf = q.float() * q_scale, k.float() * k_scale, v.float() * v_scale
    for i in range(len(seqlens_q)):
        qs, ks = slice(cu_q[i], cu_q[i + 1]), slice(cu_k[i], cu_k[i + 1])
        o_ref, lse_ref = mha_varlen_torch(qf[qs], kf[ks], vf[ks], sm_scale, causal)
        torch.testing.assert_close(out[qs].float(), o_ref, atol=3e-2, rtol=3e-2)
        torch.testing.assert_close(lse[:, qs], lse_ref, atol=1e-2, rtol=1e-3)
    assert (out[tq:] == 7.0).all(), "rows outside cu_seqlens_q were written"
    assert (lse[:, tq:] == 7.0).all(), "LSE outside cu_seqlens_q was written"

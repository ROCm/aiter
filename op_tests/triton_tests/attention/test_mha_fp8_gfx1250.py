# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass, fields

import pytest
import torch

from aiter.ops.triton.attention.mha_fp8 import mha_fwd
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

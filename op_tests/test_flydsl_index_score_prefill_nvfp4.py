# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill index block scores over the NVFP4 index cache (gfx1250).

``aiter.ops.flydsl.index_score_prefill`` against an FP32 torch reference:
score[h, r, blk] = max over the block's keys at or before r's position of
e4m3(clamp(q)) . dequant(K) times sm_scale * log2(e). The kernel's dots are the
hardware's 128-deep scaled WMMA, the reference's an FP32 matmul: they agree to
~2e-3 relative (MI455X: the WMMA's internal sum is not FP32-exact). (Bitwise equality with ATOM's Triton scorer, which issues
the same WMMA, is checked on the ATOM side: ``index_check.py``.)

Run with: python -m pytest -q op_tests/test_flydsl_index_score_prefill_nvfp4.py
"""

import importlib.util

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx_runtime() != "gfx1250":
        pytest.skip("the NVFP4 index scorer runs on gfx1250")
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    return torch.device("cuda")


def dequant(payload, scale):
    table = torch.tensor(E2M1, device=payload.device)
    codes = torch.stack([payload & 15, payload >> 4], -1).flatten(-2).long()
    values = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    values = values.reshape(*values.shape[:-1], -1, 16)
    return (values * scale.view(torch.float8_e4m3fn).float()[..., None]).flatten(-2)


def make(reqs, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    nbs = [(c + n + 127) // 128 for n, c in reqs]
    pages = sum(nbs) + 2
    payload = torch.randint(
        0, 256, (pages, 128, 64), dtype=torch.uint8, device=device, generator=g
    )
    scale = (
        (torch.rand(pages, 128, 8, device=device, generator=g) * 0.5 + 0.01)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    total = sum(n for n, _ in reqs)
    q = (torch.randn(total, 1, 128, device=device, generator=g) * 3).to(torch.bfloat16)
    perm = torch.randperm(pages, device=device).int()
    bt = torch.zeros(len(reqs), max(nbs), dtype=torch.int32, device=device)
    s = 0
    for i, nb in enumerate(nbs):
        bt[i, :nb] = perm[s : s + nb]
        s += nb
    lens = torch.tensor([n for n, _ in reqs])
    cu = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]).int().to(device)
    prefix = torch.tensor([c for _, c in reqs], dtype=torch.int32, device=device)
    return q, payload, scale, bt, cu, prefix, prefix + lens.int().to(device)


@pytest.mark.parametrize(
    "reqs", [[(64, 4096)], [(300, 0)], [(77, 1000), (259, 5000)], [(1024, 30000)]]
)
@pytest.mark.parametrize("variant", ["256:8:4", "256:4:8"])
def test_index_score_prefill_nvfp4(reqs, variant, device):
    from aiter.ops.flydsl.index_score_prefill import index_score_prefill_nvfp4
    from aiter.ops.flydsl.kernels.index_score_prefill_nvfp4_gfx1250 import (
        sm_scale_log2e,
    )

    bq, waves, ring = (int(x) for x in variant.split(":"))
    q, payload, scale, bt, cu, prefix, seq = make(reqs, device)
    max_block = (int(seq.max()) + 127) // 128
    score = torch.full((1, q.shape[0], max_block), float("nan"), device=device)
    index_score_prefill_nvfp4(
        q,
        payload,
        scale,
        score,
        bt,
        cu,
        seq,
        prefix,
        128**-0.5,
        max(n for n, _ in reqs),
        37,
        block_q=bq,
        waves=waves,
        ring=ring,
    )
    torch.cuda.synchronize()
    qe = q[:, 0].float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
    c = sm_scale_log2e(128**-0.5)
    for b, (n, cached) in enumerate(reqs):
        nb = (cached + n + 127) // 128
        k = dequant(payload[bt[b, :nb].long()], scale[bt[b, :nb].long()]).reshape(
            -1, 128
        )
        rows = qe[cu[b] : cu[b + 1]]
        dots = rows @ k.T  # [n, nb * 128]
        pos = torch.arange(cached, cached + n, device=device)
        visible = torch.arange(nb * 128, device=device)[None] <= pos[:, None]
        dots = dots.masked_fill(~visible, float("-inf"))
        ref = dots.view(n, nb, 128).amax(-1) * c
        valid = torch.arange(nb, device=device)[None] < ((pos + 128) // 128)[:, None]
        got = score[0, cu[b] : cu[b + 1], :nb]
        err = ((got - ref).abs() / ref.abs().clamp(min=1e-3))[valid]
        assert torch.isfinite(got[valid]).all()
        assert err.max().item() < 5e-3, err.max().item()

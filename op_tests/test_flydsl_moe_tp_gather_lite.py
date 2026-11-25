# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 decode GEMM1 (tp_gather): the wave-level route (``lite``) and the
m-major tile order give the derived route's bytes; split-K runs. MiniMax-M3.1
TP4 routed MoE shapes: E 128, top-4, H 6144, I 768, MXFP4 x MXFP4."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import grouped_moe_gfx1250 as gm

E, TOPK, H, I, ALIGN = 128, 4, 6144, 768, 16


@pytest.mark.parametrize(
    "tokens,topk,experts,align,tile_m,want",
    [(8, 4, 128, 16, 16, 8), (16, 4, 128, 16, 16, 16), (17, 4, 128, 16, 16, 0),
     (8, 5, 129, 16, 16, 0), (8, 4, 128, 32, 16, 0), (8, 4, 512, 16, 16, 0)],
)  # fmt: skip
def test_lite_route_only_where_it_derives_the_same_rows(
    tokens, topk, experts, align, tile_m, want
):
    assert gm._tp_route_lite(tokens, topk, experts, align, tile_m) == want


def _gemm1(t, ids, xq, xs, w1, w1s, lite, **kw):
    from aiter.ops.flydsl.grouped_gemm_mxfp4 import flydsl_grouped_gemm_a8w4_masked

    cm = (t * TOPK + min(E, t * TOPK) * (ALIGN - 1) + ALIGN - 1) // ALIGN * ALIGN
    a2 = torch.full((1, cm, I // 2), 0xA5, dtype=torch.uint8, device="cuda")
    a2s = torch.full((1, cm, I // 32), 0x5A, dtype=torch.uint8, device="cuda")
    psum, mm = (torch.full((E,), -7, dtype=torch.int32, device="cuda") for _ in "ab")
    rows = torch.full((t, TOPK), -7, dtype=torch.int32, device="cuda")
    tpg = {"topk_ids": ids, "topids_to_rows": rows, "masked_m": mm, "align": ALIGN}
    if lite:
        tpg["lite"] = lite
    flydsl_grouped_gemm_a8w4_masked(
        a2, xq, w1, xs, w1s, psum, n_experts=E, contiguous_m=cm, N=2 * I, K=H,
        a_is_fp4=1, stage1_act=2, swiglu_limit=7.0, stage1_quant_out=1,
        quant_scale=a2s, quant_wmma_rep=1, cluster_m=-1, cluster_n=-1,
        tp_gather=tpg, tile_m=16, m_warp=1, tile_n=64, n_warp=1, num_buffers=2,
        waves_per_tensor_tdm=1, **kw,
    )  # fmt: skip
    return a2, a2s, psum, mm, rows


@pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250",
    reason="gfx1250 only",
)
@pytest.mark.parametrize("t,n", [(8, 7), (8, 29), (16, 11), (16, 29)])
def test_lite_and_tile_order_match_the_derived_route(t, n):
    g = torch.Generator().manual_seed(t * 100 + n)
    pool = torch.randperm(E, generator=g)[:n]
    ids = torch.stack([pool[(torch.arange(TOPK) + i * TOPK) % n] for i in range(t)]).to(
        "cuda", torch.int32
    )

    def rnd(lo, hi, *shape):
        return torch.randint(lo, hi, shape, dtype=torch.uint8, generator=g).cuda()

    xq, xs = rnd(0, 256, t, H // 2), rnd(120, 130, t, H // 32)
    w1 = rnd(0, 256, E, 2 * I, H // 2)
    w1s = rnd(118, 125, E * 2 * I * H // 32).view(torch.int32)
    lite = 8 if t <= 8 else 16
    for k_waves, tile_k in ((1, 1024), (4, 256)):
        base = _gemm1(t, ids, xq, xs, w1, w1s, 0, k_waves=k_waves, tile_k=tile_k)
        for lite_, order in ((lite, 0), (0, 1), (lite, 1)):
            got = _gemm1(
                t,
                ids,
                xq,
                xs,
                w1,
                w1s,
                lite_,
                k_waves=k_waves,
                tile_k=tile_k,
                tile_order=order,
            )
            for name, x, y in zip(("a2", "a2s", "psum", "masked_m", "rows"), base, got):
                assert torch.equal(x, y), (k_waves, lite_, order, name)

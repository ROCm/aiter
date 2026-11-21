# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 decode GEMM (mxfp8_decode_gemm), MiniMax-M3 TP4 o_proj N 6144 x K 2048.

Against gemm_a8w8_blockscale(isBpreshuffled=True) on the same operands and an fp64
dequantized reference: within the tuned GEMM's error of the reference, bf16 rows
equal to the tuned GEMM's but for rare one-ulp flips, deterministic, rows past M
untouched, CUDA-graph replay equal to eager.

Run with: python -m pytest -q op_tests/test_mxfp8_decode_gemm.py
"""

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight

N, K = 6144, 2048


@pytest.fixture(scope="module", autouse=True)
def _gpu():
    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 only")


def _case(m, seed):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)

    def e8m0(shape, lo, hi):
        return (torch.randint(lo, hi, shape, generator=g, device=dev, dtype=torch.int32)
                .to(torch.uint8).view(torch.float8_e8m0fnu))  # fmt: skip

    x = torch.randn(m, K, generator=g, device=dev).clamp(-8, 8).to(torch.float8_e4m3fn)
    xs = e8m0((m, K // 32), 120, 127)
    w = (
        (torch.randn(N, K, generator=g, device=dev) * 0.5)
        .clamp(-8, 8)
        .to(torch.float8_e4m3fn)
    )
    ws = e8m0((N, K // 32), 118, 126)
    return (
        x,
        xs,
        w,
        ws,
        shuffle_weight(w, layout=(16, 16)),
        shuffle_mxfp8_bpreshuffle_scale(ws),
    )


def _deq(codes, scales):
    s = torch.exp2(scales.view(torch.uint8).double() - 127.0)
    return (codes.double().view(codes.shape[0], -1, 32) * s[..., None]).view(
        codes.shape[0], -1
    )


@pytest.mark.parametrize("m", [1, 16, 33, 64, 100, 128, 150, 192, 256])
def test_mxfp8_decode_gemm(m):
    from aiter.ops.flydsl.mxfp8_decode_gemm_gfx1250 import (
        mxfp8_decode_gemm,
        mxfp8_decode_gemm_supported,
    )

    if not mxfp8_decode_gemm_supported(m, N, K):
        pytest.skip(f"M={m} keeps the tuned GEMM")
    x, xs, w, ws, w_sh, ws_sh = _case(m, seed=m)
    ref = _deq(x, xs) @ _deq(w, ws).T
    prod = aiter.gemm_a8w8_blockscale(
        x, w_sh, xs, ws_sh, dtype=torch.bfloat16, isBpreshuffled=True
    )
    out = torch.full((m + 7, N), -3.0, dtype=torch.bfloat16, device="cuda")
    y = mxfp8_decode_gemm(x, xs, w_sh, ws_sh, out=out[:m])
    torch.cuda.synchronize()
    assert torch.all(out[m:] == -3.0), "rows past M written"
    err = (y.double() - ref).abs().max().item()
    err_prod = (prod.double() - ref).abs().max().item()
    assert err <= 2 * err_prod + 1e-6, (err, err_prod)
    same = (y.view(torch.int16) == prod.view(torch.int16)).double().mean().item()
    assert same > 0.995, same
    y2 = mxfp8_decode_gemm(x, xs, w_sh, ws_sh)
    assert torch.equal(y, y2), "not deterministic"
    yg = torch.empty_like(y)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        mxfp8_decode_gemm(x, xs, w_sh, ws_sh, out=yg)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(y, yg), "graph replay differs from eager"

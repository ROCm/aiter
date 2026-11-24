# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""4-wave MXFP8 backend (aiter.ops.flydsl.gemm_a8w8_mxfp8_4wave): kernelName round trip,
launch plan, scale shuffle, and the dispatcher entry ``run_gemm_a8w8_mxfp8_gfx950`` against
the 8-wave backend (bit-identical: same MFMA, same per-accumulator K order)."""

import pytest
import torch

from aiter import dtypes
from aiter.ops.flydsl.gemm_a8w8_mxfp8_4wave import (
    launch_plan,
    make_mxfp8_4wave_kernel_name,
    parse_mxfp8_4wave_kernel_name,
)


@pytest.mark.parametrize("xcd", [True, False])
@pytest.mark.parametrize("nt", [True, False])
def test_kernel_name_round_trip(xcd, nt):
    name = make_mxfp8_4wave_kernel_name(xcd, nt)
    cfg = parse_mxfp8_4wave_kernel_name(name)
    assert cfg["xcd_remap"] is xcd and cfg["store_nt"] is nt
    assert (cfg["block_m"], cfg["block_n"], cfg["block_k"]) == (256, 256, 128)
    assert parse_mxfp8_4wave_kernel_name(name + "_x") is None
    assert (
        parse_mxfp8_4wave_kernel_name("flydsl_mxfp8_8w_256x256x128_F8_F8_B16_1x8")
        is None
    )


def test_launch_plan():
    # no remainder round / a full round: one launch over the parent grid
    assert launch_plan(2048, 6144, 256, True) == [((4, 4), 0, 192)]
    assert launch_plan(5120, 6144, 256, True) == [
        ((4, 4), 0, 480)
    ]  # 480 % 256 = 224 > 128
    # 288 parents: 32 left over -> 128x128 sub-tiles (4 per parent, parents dealt 8 at a time)
    assert launch_plan(3072, 6144, 256, True) == [((4, 4), 0, 256), ((2, 2), 256, 128)]
    # 384 parents: 128 left over -> 128x256 sub-tiles (2 per parent)
    assert launch_plan(4096, 6144, 256, True) == [((4, 4), 0, 256), ((2, 4), 256, 256)]
    assert launch_plan(3072, 6144, 256, False) == [((4, 4), 0, 256), ((2, 2), 256, 128)]


def _scale_case(m, k, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        100, 140, (m, k // 32), dtype=torch.uint8, device="cuda", generator=g
    )


@pytest.mark.parametrize("m", [1, 31, 32, 1000, 4100, 8192])
@pytest.mark.parametrize("k", [512, 8192])
def test_scale_shuffle_matches_a16w4(m, k):
    from aiter.ops.flydsl.gemm_a8w8_mxfp8_4wave import _shuffled_scale
    from aiter.ops.shuffle import shuffle_scale_a16w4

    s = _scale_case(m, k)
    pad = -(-m // 32) * 32
    p = torch.full((pad, k // 32), 127, dtype=torch.uint8, device="cuda")
    p[:m] = s
    ref = (
        shuffle_scale_a16w4(p.view(dtypes.fp8_e8m0), 1, False)
        .view(torch.uint8)
        .view(-1)
    )
    assert torch.equal(_shuffled_scale(s.view(dtypes.fp8_e8m0)), ref)


def _operands(m, n, k, seed=0):
    from aiter.ops.quant import per_group_quant_hip
    from aiter.ops.shuffle import shuffle_weight

    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.05
    kw = dict(quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0)
    xq, xs = per_group_quant_hip(x, **kw)
    wq, ws = per_group_quant_hip(w, **kw)
    return xq, xs, shuffle_weight(wq, layout=(16, 16)), ws


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize(
    "m,n,k",
    [
        (300, 512, 1024),
        (3072, 6144, 8192),  # 288 parents: a 128x128 sub-tile remainder round
        (4096, 6144, 8192),  # 384 parents: a 128x256 sub-tile remainder round
        (4100, 6144, 8192),
        (5120, 2560, 6144),
    ],
)
@pytest.mark.parametrize("nt", [False, True])
def test_dispatch_bit_identical_to_8wave(m, n, k, nt):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("gfx950 only")
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950

    xq, xs, wsh, ws = _operands(m, n, k)
    ref = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    run_gemm_a8w8_mxfp8_gfx950(
        xq, wsh, xs, ws, ref, "flydsl_mxfp8_8w_256x256x128_F8_F8_B16_2x0"
    )
    got = torch.empty_like(ref)
    run_gemm_a8w8_mxfp8_gfx950(
        xq, wsh, xs, ws, got, make_mxfp8_4wave_kernel_name(True, nt)
    )
    assert torch.equal(got, ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_w_scale_cache_follows_in_place_update():
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("gfx950 only")
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950

    m, n, k = 512, 512, 1024
    xq, xs, wsh, ws = _operands(m, n, k)
    name = make_mxfp8_4wave_kernel_name(True)
    a = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws, a, name)
    ws.view(torch.uint8).add_(1)  # a new weight scale in the same storage
    b = torch.empty_like(a)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws, b, name)
    ref = torch.empty_like(a)
    run_gemm_a8w8_mxfp8_gfx950(
        xq, wsh, xs, ws, ref, "flydsl_mxfp8_8w_256x256x128_F8_F8_B16_2x0"
    )
    assert torch.equal(b, ref) and not torch.equal(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_w_scale_cache_accepts_inference_tensor():
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() != "gfx950":
        pytest.skip("gfx950 only")
    from aiter.ops.flydsl.mxscale_preshuffle_kernels import run_gemm_a8w8_mxfp8_gfx950

    m, n, k = 512, 512, 1024
    xq, xs, wsh, ws = _operands(m, n, k)
    with torch.inference_mode():
        ws_inf = ws.clone()  # an inference tensor: no version counter
    assert ws_inf.is_inference()
    name = make_mxfp8_4wave_kernel_name(True)
    a = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
    b = torch.empty_like(a)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws_inf, a, name)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws_inf, b, name)  # cache hit
    ref = torch.empty_like(a)
    run_gemm_a8w8_mxfp8_gfx950(xq, wsh, xs, ws, ref, name)
    assert torch.equal(a, ref) and torch.equal(b, ref)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q", "-x"]))

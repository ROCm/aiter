# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import (
    get_cdna4_num_xcds,
    get_cu_num,
    get_gfx_runtime,
)
from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_4wave import (
    default_4wave_kernel_name,
    make_4wave_kernel_name,
    parse_4wave_kernel_name,
)
from aiter.ops.shuffle import shuffle_weight


def _is_gfx950() -> bool:
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


requires_gfx950 = pytest.mark.skipif(not _is_gfx950(), reason="gfx950 FlyDSL kernel")


def _run_4wave(x, weight_shuffled, x_scale, w_scale, out, num_cu):
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.gemm_a8w8_int8_4wave import compile_int8_gemm_4w
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

    m, k = x.shape
    n = weight_shuffled.shape[0]
    exe = compile_int8_gemm_4w(K=k, xcd_swizzle=8, num_cu=num_cu)
    _run_compiled(
        exe,
        x.view(-1),
        weight_shuffled.view(-1),
        out.view(-1),
        x_scale.view(-1),
        w_scale.view(-1),
        m,
        n,
        fx.Stream(torch.cuda.current_stream()),
    )


@pytest.mark.parametrize("cu_num,num_xcds", [(32, 1), (64, 2), (128, 4), (256, 8)])
def test_cdna4_cu_to_xcd_mapping(cu_num, num_xcds):
    assert get_cdna4_num_xcds(cu_num) == num_xcds


@pytest.mark.parametrize("cu_num", [0, 96, 160, 288])
def test_cdna4_cu_to_xcd_mapping_rejects_invalid_counts(cu_num):
    with pytest.raises(ValueError, match="CDNA4"):
        get_cdna4_num_xcds(cu_num)


@pytest.mark.parametrize(
    "config",
    [
        {"K": 5120, "xcd_swizzle": 8, "num_cu": 256, "num_xcds": 8},
        {"K": 5120, "xcd_swizzle": 8, "num_cu": 128, "num_xcds": 4},
        {"K": 13824, "xcd_swizzle": 8, "num_cu": 0, "num_xcds": 8},
        {"K": 13824, "xcd_swizzle": 0, "num_cu": 0, "num_xcds": 4},
    ],
)
def test_4wave_kernel_name_roundtrip(config):
    name = make_4wave_kernel_name(**config)
    parsed = parse_4wave_kernel_name(name)
    assert parsed is not None
    assert parsed["k"] == config["K"]
    for key in ("xcd_swizzle", "num_cu", "num_xcds"):
        assert parsed[key] == config[key]
    assert (parsed["q_dtype_a"], parsed["q_dtype_w"], parsed["dtype"]) == (
        "I8",
        "I8",
        "B16",
    )


def test_4wave_default_persistent_policy():
    persistent = default_4wave_kernel_name(9450, 5120, 5120, cu_num=256, num_xcds=8)
    assert "_p256" in persistent
    assert "_xcd" not in persistent

    mi350p = default_4wave_kernel_name(9450, 5120, 5120, cu_num=128, num_xcds=4)
    assert "_p128_xcd4" in mi350p

    small_grid = default_4wave_kernel_name(512, 5120, 5120, cu_num=128, num_xcds=4)
    assert "_p" not in small_grid
    assert small_grid.endswith("_xcd4")

    long_k = default_4wave_kernel_name(9450, 5120, 13824, cu_num=256, num_xcds=8)
    assert "_p" not in long_k


@requires_gfx950
@pytest.mark.parametrize(
    "m,n,k",
    [
        # 2..9 K tiles: no loop trip, the loop with both tail parities.
        (256, 256, 256),
        (256, 256, 384),
        (256, 256, 512),
        (256, 256, 640),
        (256, 256, 1024),
        (256, 256, 1152),
        # M/N tails of the 256x256 workgroup tile.
        (1, 16, 256),
        (131, 272, 384),
        (513, 784, 1152),
        (1000, 1008, 5120),
        # More blocks than CUs.
        (4352, 4608, 1024),
    ],
)
# 0: one block per workgroup; 8: persistent workgroups take several blocks
# each on the small shapes; None: persistent on all CUs.
@pytest.mark.parametrize("num_cu", [0, 8, None])
@torch.no_grad()
def test_flydsl_int8_4wave_bit_exact(m, n, k, num_cu):
    if num_cu is None:
        num_cu = get_cu_num()
    generator = torch.Generator(device="cuda").manual_seed(m * 7 + n * 3 + k)
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = torch.rand(
        (m, 1), dtype=torch.float32, device="cuda", generator=generator
    )
    w_scale = torch.rand(
        (1, n), dtype=torch.float32, device="cuda", generator=generator
    )
    weight_shuffled = shuffle_weight(weight, layout=(16, 16)).contiguous()
    actual = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")

    _run_4wave(x, weight_shuffled, x_scale, w_scale, actual, num_cu)

    # Small values keep every FP32 product and sum exact, so the kernel's
    # single post-reduction scale product must match bit for bit.
    accum = torch.nn.functional.linear(x.float(), weight.float())
    expected = (accum * (x_scale * w_scale)).to(torch.bfloat16)
    assert torch.equal(actual, expected)


def test_4wave_aot_csv_parse(tmp_path):
    from aiter.aot.flydsl.gemm import parse_csv

    kernel_name = make_4wave_kernel_name(K=5120, xcd_swizzle=8, num_cu=128, num_xcds=4)
    csv_path = tmp_path / "int8.csv"
    csv_path.write_text(
        "gfx,cu_num,M,N,K,q_dtype_w,libtype,kernelId,splitK,us,"
        "kernelName,tflops,bw,errRatio\n"
        f"gfx950,128,9450,5120,5120,torch.int8,flydsl,1000040,0,0,"
        f"{kernel_name},0,0,0\n"
    )
    jobs = parse_csv(str(csv_path))
    assert len(jobs) == 1
    assert jobs[0]["kind"] == "4wave"
    assert jobs[0]["kernel_name"] == kernel_name
    assert jobs[0]["num_cu"] == 128
    assert jobs[0]["num_xcds"] == 4


@requires_gfx950
@torch.no_grad()
def test_public_int8_fallback_and_post_gemm_bias():
    m, n, k = 257, 272, 384
    generator = torch.Generator(device="cuda").manual_seed(20261022)
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = torch.rand(
        (m, 1), dtype=torch.float32, device="cuda", generator=generator
    )
    w_scale = torch.rand(
        (1, n), dtype=torch.float32, device="cuda", generator=generator
    )
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda", generator=generator)
    weight_shuffled = shuffle_weight(weight, layout=(16, 16)).contiguous()

    actual = aiter.gemm_a8w8_bpreshuffle(
        x, weight_shuffled, x_scale, w_scale, bias=bias
    )
    accum = torch.nn.functional.linear(x.float(), weight.float())
    expected = (accum * (x_scale * w_scale)).to(torch.bfloat16) + bias
    assert torch.equal(actual, expected)

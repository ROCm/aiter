# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from aiter.ops.triton.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale as triton_gemm_a8w8_blockscale,
)
# from aiter.ops.triton.gluon.gemm_a8w8_blockscale import (
from aiter.ops.triton.gluon import gemm_a8w8_blockscale 
from aiter.ops.triton.gluon import gemm_a8w8_bpreshuffle_blockscale 
from aiter.ops.triton.gluon import gemm_a8w8_bpreshuffle 
# from aiter.ops.triton.gluon.gemm_a8w8_bpreshuffle import (
from aiter.ops.triton.utils.types import str_to_torch_dtype, get_fp8_dtypes
import torch.nn.functional as F

from aiter.test_common import checkAllclose, benchmark, perftest, run_perftest
import aiter.ops.triton.utils._triton.arch_info as arch_info


block_shape = (128, 128)
DEVICE_ARCH = arch_info.get_device()


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_shape_n - 1) // block_shape_n
    scale_k = (k + block_shape_k - 1) // block_shape_k
    x_scale = x_scale.repeat_interleave(block_shape_k, dim=1)
    x = x.to(x_scale.dtype) * x_scale[:m, :k]
    x = x.view(m, k)
    w_scale = w_scale.repeat_interleave(block_shape_n, dim=0)
    w_scale = w_scale.repeat_interleave(block_shape_k, dim=1)
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(torch.float32), weight.to(torch.float32))

    return out.to(dtype)

@perftest()
def run_triton(x, weight, x_scale, w_scale, dtype=torch.bfloat16, y=None, impl=None):
    return impl(x, weight, x_scale, w_scale, dtype, y)


e5m2_type, e4m3_type = get_fp8_dtypes()


def get_x_vals():

    # x_vals = [(8192, 8192, 8192),(16, 8192, 8192),]
    x_vals = [(16,128,128),]
    # x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    # x_vals += [(4864, 4096, 8192), (9728, 8192, 65536)]
    # x_vals += [
    #     (1, 1280, 8192),
    #     (32, 1280, 8192),
    #     (64, 1280, 8192),
    #     (128, 1280, 8192),
    #     (192, 1280, 8192),
    #     (256, 1280, 8192),
    #     (320, 1280, 8192),
    #     (512, 1280, 8192),
    #     (1024, 1280, 8192),
    #     (2048, 1280, 8192),
    #     (4096, 1280, 8192),
    #     (8192, 1280, 8192),
    #     (16384, 1280, 8192),
    #     (1, 8192, 1024),
    #     (32, 8192, 1024),
    #     (64, 8192, 1024),
    #     (128, 8192, 1024),
    #     (192, 8192, 1024),
    #     (256, 8192, 1024),
    #     (320, 8192, 1024),
    #     (512, 8192, 1024),
    #     (1024, 8192, 1024),
    #     (2048, 8192, 1024),
    #     (4096, 8192, 1024),
    #     (8192, 8192, 1024),
    #     (16384, 8192, 1024),
    #     (16, 16, 128),
    #     (2048, 2048, 2049),
    #     (159, 17389, 597),
    #     (16, 576, 7168),
    # ]
    # x_vals += [
    #     (256, 8192, 1024),
    #     (256, 1024, 8192),
    #     (256, 32768, 8192),
    #     (256, 8192, 32768),
    # ]
    # x_vals += [
    #     (16, 2112, 7168),
    #     (32, 2112, 7168),
    #     (64, 2112, 7168),
    #     (128, 2112, 7168),
    #     (16, 3072, 1536),
    #     (32, 3072, 1536),
    #     (64, 3072, 1536),
    #     (128, 3072, 1536),
    #     (16, 7168, 2048),
    #     (32, 7168, 2048),
    #     (64, 7168, 2048),
    #     (128, 7168, 2048),
    #     (16, 4096, 7168),
    #     (32, 4096, 7168),
    #     (64, 4096, 7168),
    #     (128, 4096, 7168),
    #     (16, 7168, 256),
    #     (32, 7168, 256),
    #     (64, 7168, 256),
    #     (128, 7168, 256),
    # ]
    # x_vals += [(1, 1, 1)]  # minimal case
    return x_vals


def generate_gemm_a8w8_blockscale_inputs(
    M: int,
    N: int,
    K: int,
    block_shape_n: int,
    block_shape_k: int,
    dtype=torch.bfloat16,
    layout: str = "TN",
    output=False,
    block_scale=False,
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    """
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    if layout[0] == "T":
        x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    else:
        x = (
            (torch.rand((K, M), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    if layout[1] == "N":
        # weight = (torch.ones((N, K), dtype=torch.float16, device="cuda") / 10).to(
        # weight = torch.arange(0, K, dtype=torch.float16, device="cuda").view(1,K).repeat(N, 1).view(N,K).contiguous()
        # weight = weight.to(
        #     e4m3_type
        # )
        weight = (
            (torch.randn((N, K), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
        )
    else:
        raise ValueError(f"Unknown layout: {layout}")
        # weight = (
        #     (torch.randn((K, N), dtype=torch.float16, device="cuda") / 10)
        #     .to(e4m3_type)
        #     .T
        # )
    if not block_scale:
        x_scale = torch.ones([M, scale_k], dtype=torch.float32, device="cuda")
        w_scale = torch.ones([scale_n, scale_k], dtype=torch.float32, device="cuda")
    else:
        x_scale = torch.randn([M, scale_k], dtype=torch.float32, device="cuda")
        w_scale = torch.randn([scale_n, scale_k], dtype=torch.float32, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda").cuda()

    return x, weight, x_scale, w_scale, y


@pytest.mark.parametrize(
    "dtype, M, N, K, layout, output",
    [
        (dtype, *shape, layout, output)
        for output in [True]
        for dtype in ["bf16"]
        # for layout in ["TN", "TT", "NN", "NT"]
        for layout in ["TN"]
        for shape in get_x_vals()
    ],
)
@pytest.mark.parametrize(
    "impl",
    [
        "gluon",
        # "triton",
    ],
)
@pytest.mark.parametrize(
    "block_scale",
    [
        False,
        True,
    ],
)
def test_gemm(dtype, M, N, K, layout, output, impl: str, block_scale=False):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.cuda.synchronize()

    block_shape_n, block_shape_k = block_shape
    if K % block_shape_k != 0:
        pytest.skip(
            "Latest upstream compiler as of Aug 22 (necessary for Gluon) causes"
            " infinite hang when EVEN_K is false. Try seeing if it's fixed if it's been a while."
        )

    # if impl == "gluon" and int(DEVICE_ARCH.split("MI")[1].replace("X", "")) < 350:
    #     pytest.skip(
    #         "Gluon implementation is not supported on this device (requires CDNA4)."
    #     )

    dtype = str_to_torch_dtype[dtype]
    x, weight, x_scale, w_scale, y = generate_gemm_a8w8_blockscale_inputs(
        M,
        N,
        K,
        block_shape_n,
        block_shape_k,
        dtype=dtype,
        layout=layout,
        output=output,
        block_scale=block_scale,
    )

    a = run_torch(x, weight, x_scale, w_scale, dtype)
    if impl == "gluon":
        impl = gemm_a8w8_bpreshuffle_blockscale.gemm_a8w8_blockscale if block_scale else gemm_a8w8_bpreshuffle.gemm_a8w8_ptpc

    elif impl == "triton":
        impl = triton_gemm_a8w8_blockscale
    else:
        raise ValueError(f"Unknown implementation: {impl}")
    # print(weight)
    b,us = run_triton(x, weight, x_scale, w_scale, dtype, y, impl)
    flops = 2 * M * N * K / 1e6 / us
    bw = (M * N * 2 +  M * K + N * K) / 1e6 / us
    print(f"{us=}, {flops=}, {bw=}")
    # torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
    checkAllclose(a, b, atol=0.01, rtol=1e-2)

# def test_gemm(dtype, M, N, K, layout, output, impl: str):
if __name__ == "__main__":
    #blockscale
    # test_gemm("bf16", 16, 7168, 2048, "TN", True, "gluon", True)
    #pure fp8
    test_gemm("bf16", 8192, 7168, 2048, "TN", True, "gluon", False)
    # test_gemm("bf16", 16, 8192, 8192, "TN", True, "gluon", False)
    # test_gemm("bf16", 8192, 8192, 8320, "TN", True, "gluon", False)
    # test_gemm("bf16", 8192, 8192, 8320, "TN", True, "gluon", False)
    # test_gemm("bf16", 8192, 4608, 7168, "TN", True, "gluon", False)
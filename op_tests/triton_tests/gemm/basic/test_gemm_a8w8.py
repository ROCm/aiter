# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import torch.nn.functional as F
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8 as triton_gemm_a8w8
from aiter.ops.triton.gluon.gemm_a8w8 import (
    gemm_a8w8 as gluon_gemm_a8w8,
    gemm_a8w8_preshuffle as gluon_gemm_a8w8_preshuffle,
)
from aiter.ops.triton.utils.types import get_fp8_dtypes
from aiter.ops.triton.utils.types import str_to_torch_dtype
from typing import Union

from aiter.ops.shuffle import shuffle_weight
import aiter.ops.triton.utils._triton.arch_info as arch_info

DEVICE_ARCH = arch_info.get_arch()


def run_torch(x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16):
    x = F.linear(x.to(torch.float32), weight.to(torch.float32))
    scale = torch.matmul(x_scale, w_scale)
    out = torch.mul(x, scale)
    if bias is not None:
        out = out.to(bias) + bias
    return out.to(dtype)


def run_triton(
    x, weight, x_scale, w_scale, bias=None, dtype=torch.bfloat16, y=None, impl=None
):
    return impl(x, weight, x_scale, w_scale, bias, dtype, y)


e5m2_type, e4m3_type = get_fp8_dtypes()


dtype_max = {
    dtype: (torch.finfo(dtype) if dtype.is_floating_point else torch.iinfo(dtype)).max
    for dtype in [
        e5m2_type,
        e4m3_type,
        torch.int8,
    ]
}


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536)]
    # This shape (4864, 8192, 4160) failing due to compiler change https://github.com/triton-lang/triton/commit/8a34c216a9d7fa3a7e456186ddaab23428320766
    x_vals += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
        (1, 8192, 1024),
        (32, 8192, 1024),
        (64, 8192, 1024),
        (128, 8192, 1024),
        (192, 8192, 1024),
        (256, 8192, 1024),
        (320, 8192, 1024),
        (512, 8192, 1024),
        (1024, 8192, 1024),
        (2048, 8192, 1024),
        (4096, 8192, 1024),
        (8192, 8192, 1024),
        (16384, 8192, 1024),
    ]
    x_vals += [(1, 1, 1)]
    return x_vals


def generate_gemm_a8w8_inputs(
    M: int,
    N: int,
    K: int,
    in_dtype: Union[torch.dtype, str],
    out_dtype: Union[torch.dtype, str],
    layout: str = "TN",
    output: bool = False,
    shuffle: bool = False,
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    """
    if layout[0] == "T":
        # T (transposed) in Fortran notation equals row-major
        x = torch.randn((M, K), dtype=torch.float32, device="cuda")
    else:
        x = torch.randn((K, M), dtype=torch.float32, device="cuda").T

    if layout[1] == "N":
        weight = torch.randn((N, K), dtype=torch.float32, device="cuda")
    else:
        weight = torch.randn((K, N), dtype=torch.float32, device="cuda").T

    max_x = x.abs().float().amax(dim=1, keepdim=True)
    x_scale = max_x / dtype_max[in_dtype]
    x = x / x_scale
    x = x.to(in_dtype)

    max_weight = weight.abs().float().amax(dim=1, keepdim=True).T.contiguous()
    w_scale = max_weight / dtype_max[in_dtype]
    weight = weight / w_scale.T
    weight = weight.to(in_dtype)

    bias = torch.rand([1, N], dtype=torch.float32, device="cuda") * 10

    if shuffle:
        weight_shuffle_layout = (16, 16)
        weight_shuffled = shuffle_weight(weight, weight_shuffle_layout).reshape(
            weight.shape[0] // weight_shuffle_layout[0],
            weight.shape[1] * weight_shuffle_layout[0],
        )
    else:
        weight_shuffled = weight

    y = None
    if output:
        y = torch.empty((M, N), dtype=out_dtype, device="cuda")

    return x, weight, weight_shuffled, x_scale, w_scale, bias, y


@pytest.mark.parametrize(
    "in_dtype, out_dtype, m, n, k, layout, output",
    [
        (in_dtype, out_dtype, *shape, layout, output)
        for in_dtype in ["fp8e4m3", "fp8e5m2", "int8"]
        for out_dtype in ["bf16", "fp16", "fp32", "int32"]
        for shape in get_x_vals()
        for layout in ["TN", "TT", "NN", "NT"]
        for output in [True, False]
    ],
)
@pytest.mark.parametrize(
    "impl",
    [
        "triton",
        "gluon",
        "gluon_shuffle",
    ],
)
def test_gemm(in_dtype, out_dtype, m, n, k, layout, output, impl: str):

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    if out_dtype == "int32" and in_dtype in ["fp8e4m3", "fp8e5m2"]:
        pytest.skip(
            "This kernel is not supported for in_dtype of float and out_dtype of int."
        )

    if impl in ["gluon", "gluon_shuffle"] and DEVICE_ARCH != "gfx950":
        pytest.skip(
            "Gluon implementation is not supported on this device (requires CDNA4)."
        )

    if impl == "gluon_shuffle" and (n % 16 != 0 or k % 32 != 0):
        pytest.skip(
            "For preshuffle, N must be multiple of 16 and K must be multiple of 32."
        )

    in_dtype = str_to_torch_dtype[in_dtype]
    out_dtype = str_to_torch_dtype[out_dtype]
    x, weight, weight_triton, x_scale, w_scale, bias, y = generate_gemm_a8w8_inputs(
        M=m,
        N=n,
        K=k,
        in_dtype=in_dtype,
        out_dtype=out_dtype,
        layout=layout,
        output=output,
        shuffle=("_shuffle" in impl),
    )

    a = run_torch(x, weight, x_scale, w_scale, bias, out_dtype)
    if impl == "triton":
        impl = triton_gemm_a8w8
    elif impl == "gluon":
        impl = gluon_gemm_a8w8
    elif impl == "gluon_shuffle":
        impl = gluon_gemm_a8w8_preshuffle
    else:
        raise ValueError(f"Unknown implementation: {impl}")
    b = run_triton(x, weight_triton, x_scale, w_scale, bias, out_dtype, y, impl)

    if out_dtype in [torch.int8, torch.int32]:
        torch.testing.assert_close(a, b, atol=1, rtol=1e-2)
    else:
        torch.testing.assert_close(a, b, atol=0.02, rtol=1e-2)

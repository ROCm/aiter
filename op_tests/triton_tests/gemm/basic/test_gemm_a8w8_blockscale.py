# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import importlib.util
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale as triton_gemm_a8w8_blockscale,
    gemm_a8w8_blockscale_preshuffle as triton_gemm_a8w8_blockscale_preshuffle,
)
from aiter.ops.triton.gluon.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale as gluon_gemm_a8w8_blockscale,
)
from aiter.ops.triton.utils.types import str_to_torch_dtype, get_fp8_dtypes
import torch.nn.functional as F

from aiter.ops.shuffle import shuffle_weight
import aiter.ops.triton.utils._triton.arch_info as arch_info

block_shape = (128, 128)
DEVICE_ARCH = arch_info.get_arch()


def get_gpu_arch():
    """Get the GPU architecture name (e.g., 'gfx1250', 'gfx942')."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return getattr(props, "gcnArchName", None)


def is_flydsl_supported():
    """Check if flydsl kernels are supported on the current GPU."""
    if importlib.util.find_spec("flydsl") is None:
        return False
    arch = get_gpu_arch()
    return arch is not None and arch.startswith("gfx1250")


def _flydsl_gemm_a8w8_blockscale(x, weight, x_scale, w_scale, dtype, y):
    """Lazy import and call the flydsl gemm_a8w8_blockscale kernel."""
    from aiter.ops.flydsl.kernels.gemm_a8w8_blockscale_gfx1250 import (
        gemm_a8w8_blockscale as flydsl_gemm,
    )

    return flydsl_gemm(x, weight, x_scale, w_scale, y=y, dtype=dtype)


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    x_scale = x_scale.repeat_interleave(block_shape_k, dim=1)
    x = x.to(x_scale.dtype) * x_scale[:m, :k]
    x = x.view(m, k)
    w_scale = w_scale.repeat_interleave(block_shape_n, dim=0)
    w_scale = w_scale.repeat_interleave(block_shape_k, dim=1)
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(torch.float32), weight.to(torch.float32))

    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, dtype=torch.bfloat16, y=None, impl=None):
    return impl(x, weight, x_scale, w_scale, dtype, y)


e5m2_type, e4m3_type = get_fp8_dtypes()


def get_x_vals():
    # K must be a multiple of 128 (kernel constraint); M and N are free.
    x_vals = [
        # Aligned baselines
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (128, 256, 512),
        (256, 512, 256),
        # Tiny M / N (masking at tile edges)
        (1, 1, 128),
        (1, 16, 128),
        (16, 1, 128),
        (1, 3, 128),
        # Unaligned M and N, aligned K
        (3, 5, 128),
        (7, 9, 128),
        (17, 33, 128),
        (63, 127, 128),
        (65, 129, 128),
        # Unaligned M/N with multi-tile K
        (17, 33, 512),
        (63, 127, 256),
        (129, 31, 512),
        # Asymmetric rectangular
        (32, 256, 128),
        (256, 32, 128),
        (128, 128, 1024),
        (1024, 128, 128),
        (1535, 512, 768),
        (1536, 511, 768),
        # User-requested shapes
        (32, 5120, 2944),
        (2048, 5120, 2944),
    ]
    return x_vals


def generate_gemm_a8w8_blockscale_inputs(
    M: int,
    N: int,
    K: int,
    block_shape_n: int,
    block_shape_k: int,
    dtype=torch.bfloat16,
    layout: str = "TN",
    output: bool = False,
    shuffle: bool = False,
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
        weight = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        weight = (
            (torch.rand((K, N), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    if shuffle:
        weight_shuffle_layout = (16, 16)
        weight_shuffled = shuffle_weight(weight, weight_shuffle_layout).reshape(
            weight.shape[0] // weight_shuffle_layout[0],
            weight.shape[1] * weight_shuffle_layout[0],
        )
        x_scale_shuffled = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    else:
        weight_shuffled = weight
        x_scale_shuffled = x_scale

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda").cuda()

    return x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y
    return x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y


@pytest.mark.parametrize(
    "dtype, M, N, K, layout, output",
    [
        (dtype, *shape, layout, output)
        for output in [True]
        for dtype in ["bf16"]
        for layout in ["TN"]
        for shape in get_x_vals()
    ],
)
@pytest.mark.parametrize(
    "impl",
    [
        # "gluon",
        "triton",
        "triton_shuffle",
        "flydsl",
    ],
)
def test_gemm(dtype, M, N, K, layout, output, impl: str):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.cuda.synchronize()

    block_shape_n, block_shape_k = block_shape

    if impl == "gluon" and DEVICE_ARCH not in ("gfx950",):
        pytest.skip(
            "Gluon implementation is not supported on this device (requires CDNA4/gfx950)."
        )

    if impl == "flydsl" and not is_flydsl_supported():
        pytest.skip("FlyDSL not supported (requires gfx1250 and flydsl package)")

    if impl == "triton_shuffle":
        if N % 16 > 0 or K % 32 > 0:
            pytest.skip(
                "N has to be multiple of 16 and K has to be multiple of 32 for preshuffle cases"
            )

    dtype = str_to_torch_dtype[dtype]
    x, weight, weight_triton, x_scale, x_scale_shuffled, w_scale, y = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            N,
            K,
            block_shape_n,
            block_shape_k,
            dtype=dtype,
            layout=layout,
            output=output,
            shuffle=("_shuffle" in impl),
        )
    )
    x, weight, weight_triton, x_scale, x_scale_shuffled, w_scale, y = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            N,
            K,
            block_shape_n,
            block_shape_k,
            dtype=dtype,
            layout=layout,
            output=output,
            shuffle=("_shuffle" in impl),
        )
    )

    a = run_torch(x, weight, x_scale, w_scale, dtype)

    if impl == "gluon":
        impl = gluon_gemm_a8w8_blockscale
    elif impl == "triton":
        impl = triton_gemm_a8w8_blockscale
    elif impl == "triton_shuffle":
        impl = triton_gemm_a8w8_blockscale_preshuffle
    elif impl == "flydsl":
        impl = _flydsl_gemm_a8w8_blockscale
    else:
        raise ValueError(f"Unknown implementation: {impl}")

    b = run_triton(x, weight_triton, x_scale_shuffled, w_scale, dtype, y, impl)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)

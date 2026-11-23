# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
import sys

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale,
    gemm_a8w8_blockscale_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype
from op_tests.triton_tests.gemm.gemm_shapes import get_shapes

block_shape = (128, 128)
DEVICE_ARCH = arch_info.get_arch()


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
    torch.manual_seed(0)
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
        y = torch.empty((M, N), dtype=dtype, device="cuda")

    return x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y


@pytest.mark.parametrize(
    "dtype, M, N, K, layout, output",
    [
        (dtype, *shape, layout, output)
        for output in [True]
        for dtype in ["bf16"]
        for layout in ["TN"]
        for shape in get_shapes(min_k=128)
    ],
)
@pytest.mark.parametrize("backend", ["gluon", "triton"])
@pytest.mark.parametrize("shuffle", [True, False])
def test_gemm(dtype, M, N, K, layout, output, backend, shuffle):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.cuda.synchronize()

    block_shape_n, block_shape_k = block_shape

    if backend == "gluon":
        if shuffle:
            if DEVICE_ARCH not in ("gfx1250"):
                pytest.skip("Gluon + shuffle implementation requires gfx1250.")
        elif DEVICE_ARCH not in ("gfx950", "gfx1250"):
            pytest.skip("Gluon implementation requires gfx950 or gfx1250.")

    if shuffle and (N % 16 > 0 or K % 32 > 0):
        pytest.skip(
            "N has to be multiple of 16 and K has to be multiple of 32 for preshuffle cases"
        )
    if backend == "triton" and shuffle and K % 128 > 0:
        pytest.skip("Triton preshuffle needs K to be a multiple of 128")
    if backend == "gluon" and K % 64 > 0:
        pytest.skip("Gluon needs K to be a multiple of 64")

    if backend not in ("gluon",) and K < 512:
        pytest.skip("Small-K shapes exercise gluon-only paths.")

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
            shuffle=shuffle,
        )
    )

    a = run_torch(x, weight, x_scale, w_scale, dtype)

    if shuffle:

        def impl(x, w, xs, ws, dt, y):
            return gemm_a8w8_blockscale_preshuffle(x, w, xs, ws, dt, y, backend=backend)

    else:

        def impl(x, w, xs, ws, dt, y):
            return gemm_a8w8_blockscale(x, w, xs, ws, dt, y, backend=backend)

    b = run_triton(x, weight_triton, x_scale_shuffled, w_scale, dtype, y, impl)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)


def test_legacy_gluon_import_path_warns():
    """The pre-move path still resolves here, but tells callers to move on."""
    legacy = "aiter.ops.triton.gluon.gemm_a8w8_blockscale"
    sys.modules.pop(legacy, None)

    with pytest.warns(DeprecationWarning, match="has moved to"):
        mod = importlib.import_module(legacy)

    assert (
        mod.gemm_a8w8_blockscale.__module__
        == "aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale"
    )

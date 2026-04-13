# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import pytest
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
import importlib.util
from op_tests.triton_tests.utils.types import str_to_torch_dtype


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


def generate_gemm_a16w16_inputs(M, N, K, dtype, layout="TN", output=True, bias=False):
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        weight = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        weight = torch.randn((N, K), dtype=dtype, device="cuda")

    bias_tensor = None
    if bias:
        bias_tensor = torch.empty((N), dtype=dtype, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return x, weight, bias_tensor, out_dtype, y


def _flydsl_gemm_a16w16(x, w, bias=None, dtype=torch.bfloat16, y=None, activation=None):
    """Lazy import and call the flydsl gemm_a16w16 kernel."""
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx1250 import (
        gemm_a16w16 as flydsl_gemm,
    )
    return flydsl_gemm(x, w, bias=bias, dtype=dtype, y=y, activation=activation)


def get_x_vals():
    x_vals = [
        (1, 1, 1),
        (1, 16, 16),
        (16, 1, 16),
        (16, 16, 1),
        # Irregular shapes (masking & OOB)
        (3, 5, 7),
        (17, 33, 65),
        (63, 127, 255),
        (65, 129, 257),
        #
        (64, 64, 64),
        (128, 128, 128),
        # Multiple blocks
        (128, 256, 512),
        (256, 512, 256),
        # Asymmetric shapes
        (32, 256, 128),
        (256, 32, 128),
        (128, 128, 1024),
        (1024, 128, 128),
        (1536, 512, 768),
    ]
    return x_vals


# Test plain BF16 GEMMs - the most common types.
@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("backend", ["triton", "flydsl"])
def test_gemm_a16_w16(M: int, N: int, K: int, backend):
    if backend == "flydsl" and not is_flydsl_supported():
        pytest.skip("FlyDSL not supported (requires gfx1250 and flydsl package)")

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M,
        N,
        K,
        dtype=torch.bfloat16,
        output=False,
    )

    torch_out = F.linear(x, w, bias=None)

    if backend == "flydsl":
        kernel_out = _flydsl_gemm_a16w16(x, w, dtype=torch.bfloat16)
    else:
        kernel_out = gemm_a16w16(x, w)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


# Smaller set for testing activations, setting the output tensor and dtype
def get_fewer_x_vals():
    x_vals = [
        (64, 64, 64),
        (128, 256, 512),
        (256, 512, 256),
        (128, 128, 1024),
        (1024, 128, 128),
        (1536, 512, 768),
    ]
    return x_vals


# A smaller set of shapes that tests fused activations, different dtypes
# and output tensor arg. We don't want the larger set above to test
# all these combinations.
@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu", "silu_exp2"])
@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("backend", ["triton", "flydsl"])
def test_gemm_a16_w16_activation(
    M: int, N: int, K: int, dtype, output, activation, backend
):
    if backend == "flydsl" and not is_flydsl_supported():
        pytest.skip("FlyDSL not supported (requires gfx1250 and flydsl package)")

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M,
        N,
        K,
        dtype,
        output=output,
    )

    torch_out = F.linear(x, w, bias=None)
    if activation == "gelu":
        torch_out = F.gelu(torch_out)
    elif activation == "gelu_tanh":
        torch_out = F.gelu(torch_out, approximate="tanh")
    elif activation in ("silu", "silu_exp2"):
        torch_out = F.silu(torch_out)

    if backend == "flydsl":
        kernel_out = _flydsl_gemm_a16w16(
            x, w, dtype=out_dtype if not isinstance(out_dtype, tuple) else dtype,
            y=y, activation=activation,
        )
    else:
        kernel_out = gemm_a16w16(x, w, None, out_dtype, y, activation=activation)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
@pytest.mark.parametrize("backend", ["triton", "flydsl"])
def test_gemm_a16_w16_layout(M: int, N: int, K: int, layout, backend):
    if backend == "flydsl" and not is_flydsl_supported():
        pytest.skip("FlyDSL not supported (requires gfx1250 and flydsl package)")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=False
    )

    torch_out = F.linear(x, w, bias=None)

    if backend == "flydsl":
        kernel_out = _flydsl_gemm_a16w16(x, w, dtype=torch.bfloat16)
    else:
        kernel_out = gemm_a16w16(x, w, None, out_dtype, y)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_atomic(M: int, N: int, K: int, output):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=output
    )

    torch_out = F.linear(x, w, bias=None)

    # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
    if output:
        y = y.to(torch.float32).zero_()
        triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(torch.bfloat16)
    else:
        triton_out = gemm_a16w16_atomic(x, w, dtype=torch.float32).to(torch.bfloat16)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
def test_gemm_a16_w16_atomic_layout(M: int, N: int, K: int, layout):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=True
    )

    torch_out = F.linear(x, w, bias=None)

    y = y.to(torch.float32).zero_()
    triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(torch.bfloat16)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)

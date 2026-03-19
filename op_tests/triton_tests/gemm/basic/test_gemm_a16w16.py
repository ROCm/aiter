# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for A16W16 GEMM kernels.

Backend filtering (via conftest.py):
    pytest ... --gluon          # only gluon tests
    pytest ... --triton         # only triton tests

pytest -k filter:
    pytest ... -k "gluon"       # only gluon tests
    pytest ... -k "triton"      # only triton tests (note: also matches "not gluon")
    pytest ... -k "gluon and not atomic"   # gluon tests that are not atomic

Default (no flags / no -k): runs both backends where available.
Gluon tests are automatically skipped if gluon is not available or not supported
on the current GPU architecture.
"""

import torch
import torch.nn.functional as F
import pytest
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16, _is_gluon_available
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.utils.types import str_to_torch_dtype


def get_gpu_arch():
    """Get the GPU architecture name (e.g., 'gfx1250', 'gfx942')."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return getattr(props, 'gcnArchName', None)


def is_gluon_supported():
    """Check if gluon kernels are supported on the current GPU."""
    return _is_gluon_available()

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


def run_gemm(x, w, bias, out_dtype, y, backend, activation=None):
    """Unified GEMM runner dispatching via the backend parameter."""
    if isinstance(out_dtype, tuple):
        dtype_arg = out_dtype
    else:
        dtype_arg = out_dtype

    return gemm_a16w16(
        x, w,
        bias=bias,
        dtype=dtype_arg,
        y=y,
        activation=activation,
        backend=backend,
    )


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu", "silu_exp2"])
@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_gemm_a16_w16_activation(M: int, N: int, K: int, dtype, output, activation, backend):
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")

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
    elif activation == "silu":
        torch_out = F.silu(torch_out)
    elif activation == "silu_exp2":
        torch_out = F.silu(torch_out)

    kernel_out = run_gemm(x, w, None, out_dtype, y, backend, activation=activation)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_gemm_a16_w16(M: int, N: int, K: int, dtype, output, backend):
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")

    torch.cuda.empty_cache()

    x, w, bias, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, dtype, output=output, bias=True
    )

    torch_out = F.linear(x, w, bias=bias)

    kernel_out = run_gemm(x, w, bias, out_dtype, y, backend)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_gemm_a16_w16_layout(M: int, N: int, K: int, dtype, layout, output, backend):
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")

    torch.cuda.empty_cache()

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, dtype, layout=layout, output=output
    )

    torch_out = F.linear(x, w, bias=None)

    kernel_out = run_gemm(x, w, None, out_dtype, y, backend)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_atomic(M: int, N: int, K: int, dtype, output):
    torch.cuda.empty_cache()

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(M, N, K, dtype, output=output)

    torch_out = F.linear(x, w, bias=None)

    # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
    if output:
        y = y.to(torch.float32).zero_()
        triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(dtype)
    else:
        triton_out = gemm_a16w16_atomic(x, w, dtype=torch.float32).to(dtype)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_atomic_layout(M: int, N: int, K: int, dtype, layout, output):
    torch.cuda.empty_cache()

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, dtype, layout=layout, output=output
    )

    torch_out = F.linear(x, w, bias=None)

    # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
    if output:
        y = y.to(torch.float32).zero_()
        triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(dtype)
    else:
        triton_out = gemm_a16w16_atomic(x, w, dtype=torch.float32).to(dtype)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)

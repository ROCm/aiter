# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL-only unit test for gemm_a16w16 on gfx1250.

Intentionally isolated from the triton test harness: does NOT import
`aiter.ops.flydsl` or `aiter.ops.triton` at top level, because those
trigger a build of `module_aiter_core` (via `aiter.ops.enum`) which is
orthogonal to the flydsl kernel under test. The kernel file is loaded
directly by file path.
"""

import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

_KERNEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "aiter",
        "ops",
        "flydsl",
        "kernels",
        "gemm_a16w16_gfx1250.py",
    )
)


def _get_gpu_arch():
    if not torch.cuda.is_available():
        return None
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)


def _flydsl_available():
    if importlib.util.find_spec("flydsl") is None:
        return False
    arch = _get_gpu_arch()
    return arch is not None and arch.startswith("gfx1250")


if not _flydsl_available():
    pytest.skip(
        "FlyDSL a16w16 tests require gfx1250 and the flydsl package.",
        allow_module_level=True,
    )


def _load_kernel():
    """Load the kernel module by file path to skip aiter.ops.flydsl.__init__."""
    spec = importlib.util.spec_from_file_location("_flydsl_a16w16_kernel", _KERNEL_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gemm_a16w16


gemm_a16w16 = _load_kernel()


def _generate_inputs(M, N, K, dtype, layout="TN", output=False, bias=False):
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        w = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        w = torch.randn((N, K), dtype=dtype, device="cuda")

    bias_tensor = None
    if bias:
        bias_tensor = torch.randn((N,), dtype=dtype, device="cuda")

    y = torch.empty((M, N), dtype=dtype, device="cuda") if output else None
    return x, w, bias_tensor, y


def get_x_vals():
    return [
        (1, 1, 1),
        (1, 16, 16),
        (16, 1, 16),
        (16, 16, 1),
        # Irregular
        (3, 5, 7),
        (17, 33, 65),
        (63, 127, 255),
        (65, 129, 257),
        # Aligned
        (64, 64, 64),
        (128, 128, 128),
        # Multi-block
        (128, 256, 512),
        (256, 512, 256),
        # Asymmetric
        (32, 256, 128),
        (256, 32, 128),
        (128, 128, 1024),
        (1024, 128, 128),
        (1536, 512, 768),
    ]


def get_fewer_x_vals():
    return [
        (64, 64, 64),
        (128, 256, 512),
        (256, 512, 256),
        (128, 128, 1024),
        (1024, 128, 128),
        (1536, 512, 768),
    ]


@pytest.mark.parametrize("M, N, K", get_x_vals())
def test_gemm_a16_w16(M, N, K):
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)

    torch_out = F.linear(x, w, bias=None)
    kernel_out = gemm_a16w16(x, w, dtype=torch.bfloat16)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu", "silu_exp2"])
@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_activation(M, N, K, dtype, output, activation):
    torch.cuda.empty_cache()
    x, w, _, y = _generate_inputs(M, N, K, dtype, output=output)

    torch_out = F.linear(x, w, bias=None)
    if activation == "gelu":
        torch_out = F.gelu(torch_out)
    elif activation == "gelu_tanh":
        torch_out = F.gelu(torch_out, approximate="tanh")
    elif activation in ("silu", "silu_exp2"):
        torch_out = F.silu(torch_out)

    kernel_out = gemm_a16w16(x, w, dtype=dtype, y=y, activation=activation)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
def test_gemm_a16_w16_layout(M, N, K, layout):
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16, layout=layout)

    torch_out = F.linear(x, w, bias=None)
    kernel_out = gemm_a16w16(x, w, dtype=torch.bfloat16)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-1)

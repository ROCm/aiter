# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import pytest
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16, _is_gluon_available
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from op_tests.triton_tests.utils.types import str_to_torch_dtype


def is_gluon_supported():
    """gluon a16w16 kernels are only available on supported archs (gfx1250)."""
    return _is_gluon_available()


def _skip_if_gluon_unaligned_k(backend, M, N, K):
    """The gluon a16w16 kernels require K to be a multiple of BLOCK_K (no host
    K-padding); triton handles any K, so only skip the gluon backend."""
    if backend != "gluon":
        return
    cfg, _ = get_gemm_config("GEMM-A16W16", M, N, K)
    if K % cfg["BLOCK_K"] != 0:
        pytest.skip(
            f"gluon a16w16 requires K % BLOCK_K == 0 (K={K}, BLOCK_K={cfg['BLOCK_K']})"
        )


def _skip_if_triton_on_gfx1250(backend):
    """gfx1250 only ships gluon-format a16w16 configs, so the triton backend has
    no usable config there; skip the triton backend on gfx1250."""
    if backend != "triton":
        return
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if "gfx1250" in (get_arch() or ""):
        pytest.skip("triton backend has no gfx1250 a16w16 config (gluon-only arch)")


def generate_gemm_a16w16_inputs(M, N, K, dtype, layout="TN", output=True, bias=False):
    torch.manual_seed(0)
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
    x_vals = [(1, 1, 1)]  # minimal case
    x_vals += [(3, 5, 2)]  # irregular shape
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(2**i, 256, 7168) for i in range(5, 9)]  # DSR1 router GEMM
    # GPT-OSS-120B attention projections
    x_vals += [(2**i, 5120, 2880) for i in range(5, 9)]  # GPTOSS QKV input projection
    x_vals += [(2**i, 2880, 4096) for i in range(5, 9)]  # output projection
    x_vals += [(2**i, 128, 2880) for i in range(5, 9)]  # Router GEMM
    x_vals += [(v, 106496, 16384) for v in (256, 4096)]  # LL3 405B FC1
    return x_vals


# Test plain BF16 GEMMs - the most common types.
@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16(M: int, N: int, K: int, backend, kernel_type):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    _skip_if_triton_on_gfx1250(backend)
    _skip_if_gluon_unaligned_k(backend, M, N, K)

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M,
        N,
        K,
        dtype=torch.bfloat16,
        output=False,
    )

    torch_out = F.linear(x, w, bias=None)

    triton_out = gemm_a16w16(x, w, backend=backend, kernel_type=kernel_type)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


# Smaller set for testing activations, setting the output tensor and dtype
def get_fewer_x_vals():
    x_vals = [(16, 1024, 1024)]
    x_vals += [(128, 8192, 512)]
    x_vals += [(256, 512, 8192)]
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 5, 8)]
    return x_vals


# A smaller set of shapes that tests fused activations, different dtypes
# and output tensor arg. We don't want the larger set above to test
# all these combinations.
@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu"])
@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16_activation(
    M: int, N: int, K: int, dtype, output, activation, backend, kernel_type
):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    _skip_if_triton_on_gfx1250(backend)
    _skip_if_gluon_unaligned_k(backend, M, N, K)

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

    triton_out = gemm_a16w16(
        x,
        w,
        None,
        out_dtype,
        y,
        activation=activation,
        backend=backend,
        kernel_type=kernel_type,
    )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16_layout(M: int, N: int, K: int, layout, backend, kernel_type):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    _skip_if_triton_on_gfx1250(backend)
    _skip_if_gluon_unaligned_k(backend, M, N, K)

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=False
    )

    torch_out = F.linear(x, w, bias=None)

    triton_out = gemm_a16w16(
        x, w, None, out_dtype, y, backend=backend, kernel_type=kernel_type
    )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


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

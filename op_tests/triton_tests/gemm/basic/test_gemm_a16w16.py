# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Unit tests for A16W16 GEMM kernels.

-k filter:
    pytest ... -k "gluon"       # only gluon tests
    pytest ... -k "triton"      # only triton tests (note: also matches "not gluon")
    pytest ... -k "gluon and not atomic"   # Do gluon tests that are not atomic
    
Default (no -k): runs both tests where available.
Gluon tests are automatically skipped if gluon is not available or not supported on the architecture.
"""

import torch
import torch.nn.functional as F
import pytest
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.utils.types import str_to_torch_dtype
from aiter.ops.gluon.gemm.basic.gemm_a16w16_gfx1250 import gemm_a16w16_gfx1250


def get_gpu_arch():
    """Get the GPU architecture name (e.g., 'gfx1250', 'gfx942')."""
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    # gcnArchName is available on ROCm
    return getattr(props, 'gcnArchName', None)


def is_gluon_supported():
    """Check if gluon kernels are supported on the current GPU."""
    arch = get_gpu_arch()
    # Gluon GFX1250 kernel only supports gfx1250 (RDNA4)
    return arch is not None and 'gfx1250' in arch

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

    # # Original performance test cases (commented out for functional testing)
    # x_vals = [(1, 1, 1)]  # minimal case
    # x_vals += [(3, 5, 2)]  # irregular shape
    # x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    # x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]
    # x_vals += [(2**i, 256, 7168) for i in range(5, 9)]
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
    # ]
    # return x_vals

def run_gemm_triton(x, w, bias, out_dtype, y, activation=None):
    """Triton GEMM kernel."""
    if y is not None:
        return gemm_a16w16(x, w, bias, out_dtype, y, activation=activation)
    else:
        return gemm_a16w16(x, w, bias, out_dtype, activation=activation)


def run_gemm_gluon(x, w, bias, dtype, y, activation=None):
    """
    Run the Gluon GEMM kernel.
    
    Note: Gluon kernel expects weights in (N, K) format with preshuffled=True,
    which matches how the test generates weights.
    """
    if isinstance(dtype, tuple):
        out_dtype = x.dtype
    else:
        out_dtype = dtype
    
    return gemm_a16w16_gfx1250(
        x, w,
        bias=bias,
        dtype=out_dtype,
        y=y,
        preshuffled=True, 
        activation=activation,
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

    if backend == "triton":
        kernel_out = run_gemm_triton(x, w, None, out_dtype, y, activation=activation)
    else:
        kernel_out = run_gemm_gluon(x, w, None, out_dtype, y, activation=activation)

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

    if backend == "triton":
        kernel_out = run_gemm_triton(x, w, bias, out_dtype, y)
    else:
        kernel_out = run_gemm_gluon(x, w, bias, out_dtype, y)

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

    if backend == "triton":
        kernel_out = run_gemm_triton(x, w, None, out_dtype, y)
    else:
        kernel_out = run_gemm_gluon(x, w, None, out_dtype, y)

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
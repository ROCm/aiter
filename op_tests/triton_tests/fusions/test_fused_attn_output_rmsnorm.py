# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Test for Fused Attention Output + RMSNorm kernel
"""

import pytest
import torch
from aiter.ops.triton.fusions.fused_attn_output_rmsnorm import fused_attn_output_rmsnorm


def rmsnorm_reference(x, weight, epsilon=1e-5):
    """Reference RMSNorm implementation"""
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    return weight * x


@pytest.mark.parametrize("M", [1, 16, 128, 1024])
@pytest.mark.parametrize("N", [512, 2048, 4096, 8192])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_residual", [False, True])
def test_fused_attn_output_rmsnorm(M, N, dtype, has_residual):
    """Test correctness of fused kernel"""
    torch.manual_seed(42)
    
    # Create inputs
    attn_output = torch.randn(M, N, dtype=dtype, device="cuda")
    weight = torch.randn(N, dtype=dtype, device="cuda")
    residual = torch.randn(M, N, dtype=dtype, device="cuda") if has_residual else None
    
    # Fused kernel
    if has_residual:
        output_fused, residual_out = fused_attn_output_rmsnorm(
            attn_output, weight, residual=residual
        )
    else:
        output_fused = fused_attn_output_rmsnorm(attn_output, weight)
    
    # Reference implementation
    if has_residual:
        x = attn_output + residual
    else:
        x = attn_output
    output_ref = rmsnorm_reference(x.float(), weight.float()).to(dtype)
    
    # Check correctness
    torch.testing.assert_close(output_fused, output_ref, rtol=1e-2, atol=1e-2)
    
    # Check residual output if present
    if has_residual:
        torch.testing.assert_close(residual_out, x, rtol=1e-3, atol=1e-3)
    
    print(f"✓ Test passed: M={M}, N={N}, dtype={dtype}, has_residual={has_residual}")


@pytest.mark.parametrize("M", [1, 128, 1024])
@pytest.mark.parametrize("N", [4096, 8192])
@pytest.mark.parametrize("x_pad_to_multiple", [0, 256])
def test_fused_attn_output_rmsnorm_with_padding(M, N, x_pad_to_multiple):
    """Test correctness with output padding (for MoE compatibility)"""
    torch.manual_seed(42)
    dtype = torch.bfloat16
    
    # Create inputs
    attn_output = torch.randn(M, N, dtype=dtype, device="cuda")
    weight = torch.randn(N, dtype=dtype, device="cuda")
    residual = torch.randn(M, N, dtype=dtype, device="cuda")
    
    # Fused kernel with padding
    output_fused, residual_out = fused_attn_output_rmsnorm(
        attn_output, weight, residual=residual, x_pad_to_multiple=x_pad_to_multiple
    )
    
    # Calculate expected output shape
    if x_pad_to_multiple > 0:
        N_out = ((N + x_pad_to_multiple - 1) // x_pad_to_multiple) * x_pad_to_multiple
    else:
        N_out = N
    
    # Check output shape
    assert output_fused.shape == (M, N_out), f"Expected shape {(M, N_out)}, got {output_fused.shape}"
    assert residual_out.shape == (M, N), f"Residual shape should be {(M, N)}, got {residual_out.shape}"
    
    # Reference implementation
    x = attn_output + residual
    output_ref = rmsnorm_reference(x.float(), weight.float()).to(dtype)
    
    # Check correctness (only valid portion)
    torch.testing.assert_close(output_fused[:, :N], output_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(residual_out, x, rtol=1e-3, atol=1e-3)
    
    print(f"✓ Padding test passed: M={M}, N={N}, x_pad_to_multiple={x_pad_to_multiple}, N_out={N_out}")


@pytest.mark.parametrize("M", [128, 1024])
@pytest.mark.parametrize("N", [2048, 4096, 8192])
def test_fused_attn_output_rmsnorm_benchmark(M, N, benchmark=None):
    """Benchmark fused vs unfused implementation"""
    torch.manual_seed(42)
    
    attn_output = torch.randn(M, N, dtype=torch.float16, device="cuda")
    weight = torch.randn(N, dtype=torch.float16, device="cuda")
    residual = torch.randn(M, N, dtype=torch.float16, device="cuda")
    
    # Warmup
    for _ in range(10):
        _ = fused_attn_output_rmsnorm(attn_output, weight, residual=residual)
    
    # Benchmark fused
    torch.cuda.synchronize()
    import time
    start = time.time()
    for _ in range(100):
        output_fused, _ = fused_attn_output_rmsnorm(attn_output, weight, residual=residual)
    torch.cuda.synchronize()
    fused_time = (time.time() - start) / 100
    
    # Benchmark unfused
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        x = attn_output + residual
        output_unfused = rmsnorm_reference(x, weight)
    torch.cuda.synchronize()
    unfused_time = (time.time() - start) / 100
    
    speedup = unfused_time / fused_time
    print(f"M={M}, N={N}: Fused={fused_time*1000:.3f}ms, Unfused={unfused_time*1000:.3f}ms, Speedup={speedup:.2f}x")


if __name__ == "__main__":
    # Quick test
    print("Running quick correctness tests...")
    test_fused_attn_output_rmsnorm(128, 4096, torch.float16, True)
    test_fused_attn_output_rmsnorm(128, 4096, torch.float16, False)
    
    print("\nRunning padding tests...")
    test_fused_attn_output_rmsnorm_with_padding(128, 4096, 256)
    
    print("\nRunning benchmark...")
    test_fused_attn_output_rmsnorm_benchmark(128, 4096, None)
    
    print("\n✅ All tests passed!")

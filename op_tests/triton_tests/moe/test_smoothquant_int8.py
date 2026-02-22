# Unit tests for SmoothQuant Int8 Quantization Kernel
# Tests the smoothquant_int8.py triton kernels in isolation

import pytest
import torch
import triton

from aiter.ops.triton._triton_kernels.moe.smoothquant_int8 import (
    _smoothquant_fuse_quant_kernel,
    _smoothquant_fuse_quant_kernel_single_pass,
)


# ---------------
# Reference Implementation
# ---------------


def smoothquant_quantize_torch(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of smoothquant quantization.
    
    Args:
        x: Input tensor in bf16/fp16 [M, K]
        smooth_scale: Per-column smooth scale in fp32 [K]
        
    Returns:
        x_int8: Quantized int8 tensor [M, K]
        x_scale: Per-row quantization scale in fp32 [M]
    """
    # Apply smooth scale (per column)
    x_smooth = x.to(torch.float32) * smooth_scale[None, :]
    
    # Compute per-row max absolute value
    row_max = x_smooth.abs().max(dim=1).values
    
    # Compute per-row scale
    INT8_MAX = 127.0
    row_scale = row_max / INT8_MAX + 1e-12
    
    # Quantize: x_int8 = round(x_smooth / row_scale)
    x_scaled = x_smooth / row_scale[:, None]
    x_int8 = x_scaled.round().clamp(-127, 127).to(torch.int8)
    
    return x_int8, row_scale


# ---------------
# Triton Wrapper Functions
# ---------------


def smoothquant_quantize_triton_single_pass(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for single-pass triton kernel."""
    M, K = x.shape
    device = x.device
    
    x_int8 = torch.empty((M, K), dtype=torch.int8, device=device)
    x_scale = torch.empty((M,), dtype=torch.float32, device=device)
    
    smooth_scale = smooth_scale.to(torch.float32).contiguous()
    
    BLOCK_M = min(triton.next_power_of_2(M), 32)
    BLOCK_K = triton.next_power_of_2(K)
    grid = (triton.cdiv(M, BLOCK_M),)
    
    _smoothquant_fuse_quant_kernel_single_pass[grid](
        x, x.stride(0), x.stride(1),
        smooth_scale,
        x_int8, x_int8.stride(0), x_int8.stride(1),
        x_scale, 1,
        M, K,
        BLOCK_M, BLOCK_K,
        num_warps=4,
    )
    
    return x_int8, x_scale


def smoothquant_quantize_triton_two_pass(
    x: torch.Tensor,
    smooth_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Wrapper for two-pass triton kernel."""
    M, K = x.shape
    device = x.device
    
    x_int8 = torch.empty((M, K), dtype=torch.int8, device=device)
    x_scale = torch.empty((M,), dtype=torch.float32, device=device)
    
    smooth_scale = smooth_scale.to(torch.float32).contiguous()
    
    BLOCK_M = min(triton.next_power_of_2(M), 32)
    BLOCK_K = 256  # Fixed block size for K iteration
    grid = (triton.cdiv(M, BLOCK_M),)
    
    _smoothquant_fuse_quant_kernel[grid](
        x, x.stride(0), x.stride(1),
        smooth_scale,
        x_int8, x_int8.stride(0), x_int8.stride(1),
        x_scale, 1,
        M, K,
        BLOCK_M, BLOCK_K,
        num_warps=4,
    )
    
    return x_int8, x_scale


# ---------------
# Helper Functions
# ---------------


def assert_scales_close(ref_scale, tri_scale, rtol=1e-4, atol=1e-6, verbose=True):
    """Compare scales with tolerance."""
    diff = (ref_scale - tri_scale).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    if verbose:
        print(f"Scale comparison - max_diff: {max_diff:.6e}, mean_diff: {mean_diff:.6e}")
    
    assert torch.allclose(ref_scale, tri_scale, rtol=rtol, atol=atol), \
        f"Scales differ: max_diff={max_diff}, mean_diff={mean_diff}"


def assert_int8_close(ref_int8, tri_int8, max_diff_allowed=1, verbose=True):
    """Compare int8 values allowing small rounding differences."""
    diff = (ref_int8.to(torch.int32) - tri_int8.to(torch.int32)).abs()
    max_diff = diff.max().item()
    num_different = (diff > 0).sum().item()
    num_large_diff = (diff > max_diff_allowed).sum().item()
    
    if verbose:
        print(f"Int8 comparison - max_diff: {max_diff}, num_different: {num_different}/{ref_int8.numel()}, "
              f"num_large_diff (>{max_diff_allowed}): {num_large_diff}")
    
    assert max_diff <= max_diff_allowed, \
        f"Int8 values differ by more than {max_diff_allowed}: max_diff={max_diff}"


def dequantize_and_compare(x_int8, x_scale, x_original, smooth_scale, rtol=0.15, verbose=True):
    """Dequantize and compare to original scaled input.
    
    Note: Int8 quantization has inherent quantization error because we're mapping
    continuous values to only 255 discrete levels. A mean relative error of 10-15%
    is expected and acceptable for int8 quantization.
    """
    # Dequantize
    x_dequant = x_int8.to(torch.float32) * x_scale[:, None]
    
    # Original scaled value
    x_scaled_original = x_original.to(torch.float32) * smooth_scale[None, :]
    
    # Compute relative error
    rel_error = (x_dequant - x_scaled_original).abs() / (x_scaled_original.abs() + 1e-6)
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()
    
    if verbose:
        print(f"Dequant comparison - max_rel_error: {max_rel_error:.4f}, mean_rel_error: {mean_rel_error:.4f}")
    
    # Allow reasonable quantization error for int8 (typically 10-15% mean error)
    assert mean_rel_error < rtol, f"Mean relative error {mean_rel_error} exceeds tolerance {rtol}"


# ---------------
# Unit Tests
# ---------------


@pytest.mark.parametrize("m", [1, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("k", [64, 128, 256, 512])
def test_smoothquant_single_pass_vs_reference(m, k, device="cuda"):
    """Test single-pass kernel against PyTorch reference."""
    torch.manual_seed(42)
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Reference
    ref_int8, ref_scale = smoothquant_quantize_torch(x, smooth_scale)
    
    # Triton
    tri_int8, tri_scale = smoothquant_quantize_triton_single_pass(x, smooth_scale)
    
    print(f"\n=== Testing single-pass M={m}, K={k} ===")
    assert_scales_close(ref_scale, tri_scale)
    assert_int8_close(ref_int8, tri_int8)
    dequantize_and_compare(tri_int8, tri_scale, x, smooth_scale)


@pytest.mark.parametrize("m", [1, 16, 32, 64, 128, 256])
@pytest.mark.parametrize("k", [512, 1024, 2048, 4096])
def test_smoothquant_two_pass_vs_reference(m, k, device="cuda"):
    """Test two-pass kernel against PyTorch reference."""
    torch.manual_seed(42)
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Reference
    ref_int8, ref_scale = smoothquant_quantize_torch(x, smooth_scale)
    
    # Triton
    tri_int8, tri_scale = smoothquant_quantize_triton_two_pass(x, smooth_scale)
    
    print(f"\n=== Testing two-pass M={m}, K={k} ===")
    assert_scales_close(ref_scale, tri_scale)
    assert_int8_close(ref_int8, tri_int8)
    dequantize_and_compare(tri_int8, tri_scale, x, smooth_scale)


@pytest.mark.parametrize("m", [17, 33, 65, 100, 127])
@pytest.mark.parametrize("k", [65, 129, 257, 500])
def test_smoothquant_non_power_of_2(m, k, device="cuda"):
    """Test with non-power-of-2 dimensions."""
    torch.manual_seed(42)
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Reference
    ref_int8, ref_scale = smoothquant_quantize_torch(x, smooth_scale)
    
    # Triton (use appropriate kernel based on K size)
    if k <= 512:
        tri_int8, tri_scale = smoothquant_quantize_triton_single_pass(x, smooth_scale)
    else:
        tri_int8, tri_scale = smoothquant_quantize_triton_two_pass(x, smooth_scale)
    
    print(f"\n=== Testing non-power-of-2 M={m}, K={k} ===")
    assert_scales_close(ref_scale, tri_scale)
    assert_int8_close(ref_int8, tri_int8)


@pytest.mark.parametrize("scale_magnitude", [1e-3, 1.0, 1e3])
def test_smoothquant_scale_magnitudes(scale_magnitude, device="cuda"):
    """Test with different input scale magnitudes."""
    torch.manual_seed(42)
    m, k = 64, 256
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16) * scale_magnitude
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Triton
    tri_int8, tri_scale = smoothquant_quantize_triton_single_pass(x, smooth_scale)
    
    print(f"\n=== Testing scale_magnitude={scale_magnitude} ===")
    
    # Verify no NaN or Inf
    assert not torch.isnan(tri_int8.to(torch.float32)).any(), "NaN in quantized values"
    assert not torch.isinf(tri_scale).any(), "Inf in scales"
    assert not torch.isnan(tri_scale).any(), "NaN in scales"
    
    # Verify int8 range
    assert tri_int8.abs().max() <= 127, f"Int8 out of range: {tri_int8.abs().max()}"
    
    print(f"Scale range: [{tri_scale.min():.6e}, {tri_scale.max():.6e}]")
    print(f"Int8 range: [{tri_int8.min()}, {tri_int8.max()}]")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_smoothquant_input_dtypes(dtype, device="cuda"):
    """Test with different input dtypes."""
    torch.manual_seed(42)
    m, k = 64, 256
    
    x = torch.randn((m, k), device=device, dtype=dtype)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Reference
    ref_int8, ref_scale = smoothquant_quantize_torch(x, smooth_scale)
    
    # Triton
    tri_int8, tri_scale = smoothquant_quantize_triton_single_pass(x, smooth_scale)
    
    print(f"\n=== Testing dtype={dtype} ===")
    assert_scales_close(ref_scale, tri_scale)
    assert_int8_close(ref_int8, tri_int8)


def test_smoothquant_zero_rows(device="cuda"):
    """Test handling of zero rows."""
    torch.manual_seed(42)
    m, k = 64, 256
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    # Set some rows to zero
    x[10, :] = 0
    x[20, :] = 0
    x[30, :] = 0
    
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Triton
    tri_int8, tri_scale = smoothquant_quantize_triton_single_pass(x, smooth_scale)
    
    print("\n=== Testing zero rows ===")
    
    # Zero rows should produce zero int8 values
    assert tri_int8[10, :].abs().max() == 0, "Row 10 should be all zeros"
    assert tri_int8[20, :].abs().max() == 0, "Row 20 should be all zeros"
    assert tri_int8[30, :].abs().max() == 0, "Row 30 should be all zeros"
    
    # Scales should be small (just epsilon)
    assert tri_scale[10] < 1e-10, f"Scale for zero row should be tiny: {tri_scale[10]}"
    
    print("Zero row handling: PASSED")


def test_smoothquant_deterministic(device="cuda"):
    """Test that results are deterministic."""
    torch.manual_seed(42)
    m, k = 64, 512
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Run multiple times
    results = []
    for i in range(3):
        tri_int8, tri_scale = smoothquant_quantize_triton_single_pass(x, smooth_scale)
        results.append((tri_int8.clone(), tri_scale.clone()))
    
    print("\n=== Testing determinism ===")
    
    # All results should be identical
    for i in range(1, len(results)):
        assert torch.equal(results[0][0], results[i][0]), f"Int8 results differ at run {i}"
        assert torch.equal(results[0][1], results[i][1]), f"Scale results differ at run {i}"
    
    print("Determinism: PASSED")


@pytest.mark.parametrize("m", [1, 8, 32])
def test_smoothquant_batch_size_1(m, device="cuda"):
    """Test with very small batch sizes."""
    torch.manual_seed(42)
    k = 256
    
    x = torch.randn((m, k), device=device, dtype=torch.bfloat16)
    smooth_scale = torch.randn((k,), device=device, dtype=torch.float32).abs() + 0.1
    
    # Reference
    ref_int8, ref_scale = smoothquant_quantize_torch(x, smooth_scale)
    
    # Triton
    tri_int8, tri_scale = smoothquant_quantize_triton_single_pass(x, smooth_scale)
    
    print(f"\n=== Testing small batch M={m} ===")
    assert_scales_close(ref_scale, tri_scale)
    assert_int8_close(ref_int8, tri_int8)


if __name__ == "__main__":
    # Run a quick sanity check
    print("Running smoothquant int8 kernel tests...")
    
    # Basic tests
    test_smoothquant_single_pass_vs_reference(64, 256)
    test_smoothquant_two_pass_vs_reference(64, 1024)
    test_smoothquant_non_power_of_2(33, 129)
    test_smoothquant_scale_magnitudes(1.0)
    test_smoothquant_zero_rows()
    test_smoothquant_deterministic()
    
    print("\n" + "="*50)
    print("All sanity checks passed!")
    print("="*50)

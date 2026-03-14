import torch
import pytest
from aiter.ops.triton.gemm.batched.batched_gemm_a16w8 import batched_gemm_a16w8
from aiter.ops.triton.utils.types import get_fp8_dtypes
import torch.nn.functional as F

# Block shape for blockscale quantization
block_shape = (128, 128)


def generate_batched_gemm_a16w8_inputs(
    B: int,
    M: int,
    N: int,
    K: int,
    block_shape_n: int,
    block_shape_k: int,
    dtype=torch.bfloat16,
    with_bias: bool = False,
    output: bool = False,
):
    """
    Generate inputs for batched GEMM A16W8.
    
    Returns:
        x: (B, M, K) in FP16/BF16
        w: (B, N, K) in FP8
        w_scale: (B, scale_n, scale_k) in FP32
        bias: (B, N) in dtype or None
        y: (B, M, N) in dtype or None
    """
    e5m2_type, e4m3_type = get_fp8_dtypes()
    
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    torch.manual_seed(42)
    
    # Generate input tensor A
    x = torch.randn((B, M, K), dtype=dtype, device="cuda") / 10

    # Generate weight tensor B in FP8
    w = (torch.rand((B, N, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)

    # Generate weight scales
    w_scale = torch.rand([B, scale_n, scale_k], dtype=torch.float32, device="cuda")

    # Generate bias if requested
    bias = None
    if with_bias:
        bias = torch.randn([B, N], dtype=dtype, device="cuda") / 10

    # Allocate output if requested
    y = None
    if output:
        y = torch.empty((B, M, N), dtype=dtype, device="cuda")

    return x, w, w_scale, bias, y


def run_torch(x, weight, w_scale, bias=None, dtype=torch.bfloat16):
    """
    Reference implementation using PyTorch.
    
    Args:
        x: (B, M, K) tensor
        weight: (B, N, K) tensor in FP8
        w_scale: (B, scale_n, scale_k) tensor
        bias: Optional (B, N) tensor
    """
    block_shape_n, block_shape_k = block_shape
    B, M, K = x.shape
    _, N, _ = weight.shape
    
    outputs = []
    for b in range(B):
        x_b = x[b]  # (M, K)
        w_b = weight[b]  # (N, K)
        w_scale_b = w_scale[b]  # (scale_n, scale_k)
        
        # Dequantize weights using blockscale
        w_scale_expanded = w_scale_b.repeat_interleave(block_shape_n, dim=0)
        w_scale_expanded = w_scale_expanded.repeat_interleave(block_shape_k, dim=1)
        w_dequant = w_b.to(w_scale_expanded.dtype) * w_scale_expanded[:N, :K]
        
        # Compute matmul: (M, K) x (N, K)^T = (M, N)
        out = F.linear(x_b.to(torch.float32), w_dequant.to(torch.float32))
        
        # Add bias if provided
        if bias is not None:
            out = out + bias[b].to(torch.float32)
        
        outputs.append(out.to(dtype))
    
    return torch.stack(outputs, dim=0)


def get_x_vals():
    """Generate test shapes (B, M, N, K)"""
    x_vals = []
    
    # Small batch sizes with various M, N, K
    x_vals += [(2, 256, 256, 256), (4, 512, 512, 512)]
    x_vals += [(2, 1024, 1024, 1024), (4, 2048, 2048, 2048)]
    
    # Larger batches
    x_vals += [(8, 256, 256, 256), (16, 512, 512, 512)]
    
    # Non-square matrices
    x_vals += [(2, 128, 256, 512), (4, 256, 512, 1024)]
    x_vals += [(2, 1280, 8192, 1024), (4, 8192, 1024, 1280)]
    
    # Edge cases
    x_vals += [(1, 128, 128, 128)]  # Single batch
    x_vals += [(2, 64, 64, 64)]     # Small dimensions
    
    # Realistic LLM shapes
    x_vals += [(2, 2048, 8192, 1024), (4, 1024, 4096, 8192)]
    
    # Varying batch sizes with fixed shapes
    x_vals += [(b, 512, 512, 512) for b in [1, 2, 3, 5, 7, 8]]
    
    # Minimal case
    x_vals += [(1, 1, 128, 128)]
    
    return x_vals


@pytest.mark.parametrize("B, M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("prequant", [False, True])
def test_batched_gemm_a16w8(B: int, M: int, N: int, K: int, dtype, with_bias, prequant):
    """Test batched GEMM A16W8 kernel for correctness."""
    block_shape_n, block_shape_k = block_shape

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, w_scale, bias, y = generate_batched_gemm_a16w8_inputs(
        B, M, N, K, block_shape_n, block_shape_k, dtype=dtype, with_bias=with_bias, output=True
    )

    # Run reference implementation
    torch_out = run_torch(x, w, w_scale, bias, dtype)
    
    # Run Triton implementation
    batched_gemm_a16w8(x, w, w_scale, bias, dtype, y, prequant=prequant)

    # Compare results
    # Use relaxed tolerances for FP8 operations
    atol = 0.15 if prequant else 0.1
    rtol = 0.15 if prequant else 0.1
    
    torch.testing.assert_close(torch_out, y, atol=atol, rtol=rtol)


@pytest.mark.parametrize("B, M, N, K", [(2, 256, 256, 256), (4, 512, 512, 512)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_batched_gemm_a16w8_no_output_alloc(B: int, M: int, N: int, K: int, dtype):
    """Test that the kernel can allocate output internally."""
    block_shape_n, block_shape_k = block_shape
    
    x, w, w_scale, bias, _ = generate_batched_gemm_a16w8_inputs(
        B, M, N, K, block_shape_n, block_shape_k, dtype=dtype, with_bias=False, output=False
    )
    
    torch_out = run_torch(x, w, w_scale, None, dtype)
    triton_out = batched_gemm_a16w8(x, w, w_scale, None, dtype, None, prequant=False)
    
    torch.testing.assert_close(torch_out, triton_out, atol=0.1, rtol=0.1)


def test_batched_gemm_a16w8_shape_mismatch():
    """Test that shape mismatches raise appropriate errors."""
    B, M, N, K = 2, 256, 256, 256
    block_shape_n, block_shape_k = block_shape
    
    x, w, w_scale, _, _ = generate_batched_gemm_a16w8_inputs(
        B, M, N, K, block_shape_n, block_shape_k
    )
    
    # Test batch size mismatch
    w_wrong_batch = w[:1]  # Different batch size
    with pytest.raises(AssertionError, match="Batch size mismatch"):
        batched_gemm_a16w8(x, w_wrong_batch, w_scale)
    
    # Test K dimension mismatch
    x_wrong_k = x[:, :, :K//2]
    with pytest.raises(AssertionError, match="K dimension mismatch"):
        batched_gemm_a16w8(x_wrong_k, w, w_scale)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

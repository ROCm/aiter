#!/usr/bin/env python3
"""
Optimized test for gated RMSNorm with FP8 group quantization - HIP kernel validation.

Tests the fused HIP kernel that performs:
1. Per-head RMSNorm(x, weight, eps)
2. Gating with SiLU: out = norm(x) * silu(z)
3. FP8 group quantization with group_size=128
4. Flatten: [num_tokens, num_heads, head_dim] -> [num_tokens, num_heads*head_dim]

Constraint: ONLY supports head_dim=128 and group_size=128
"""

import pandas as pd
import torch
import aiter
from aiter.test_common import checkAllclose, perftest
import argparse


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def rms_norm_forward(x: torch.Tensor, weight: torch.Tensor, eps: float):
    """Reference RMSNorm implementation."""
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    x_normed = x_normed.to(input_dtype)
    return weight * x_normed


def normalize_scales_layout(
    scales: torch.Tensor,
    num_tokens: int,
    num_heads: int,
    transpose_scale: bool,
) -> torch.Tensor:
    """Convert the physical scale buffer layout into logical [token, head]."""
    if not transpose_scale:
        return scales
    return scales.reshape(-1).view(num_heads, num_tokens).transpose(0, 1).contiguous()


def gated_rmsnorm_fp8_group_quant_reference_impl(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    quant_dtype,
):
    """Reference that matches the fused HIP kernel math and quantization path."""
    if quant_dtype == torch.float8_e4m3fnuz:
        fp8_max = 240.0
    elif quant_dtype == torch.float8_e4m3fn:
        fp8_max = 448.0
    else:
        raise ValueError(f"Unsupported FP8 dtype for this test: {quant_dtype}")

    variance = x.float().pow(2).mean(-1, keepdim=True)
    inv_std = torch.rsqrt(variance + eps)
    normed = x.float() * inv_std
    normed = normed * weight.float().view(1, 1, -1)

    gated = normed * silu(z.float())
    scales = gated.abs().amax(dim=-1) / fp8_max
    scales = torch.maximum(scales, torch.full_like(scales, 1e-10))

    out_quant = torch.clamp(gated / scales.unsqueeze(-1), -fp8_max, fp8_max).to(
        quant_dtype
    )
    return out_quant.reshape(x.shape[0], -1), scales


@perftest()
def test_gated_rmsnorm_fp8_group_quant_reference(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    quant_dtype,
    transpose_scale: bool = False,
):
    """Reference implementation that matches the fused kernel numerics."""
    del group_size, transpose_scale
    return gated_rmsnorm_fp8_group_quant_reference_impl(x, z, weight, eps, quant_dtype)


@perftest()
def test_gated_rmsnorm_fp8_group_quant_hip(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    quant_dtype,
    transpose_scale: bool = False,
):
    """HIP kernel implementation."""
    from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_fp8_group_quant

    num_tokens, num_heads, head_dim = x.shape

    # Allocate output tensors
    out_quant = torch.empty(
        num_tokens, num_heads * head_dim, dtype=quant_dtype, device=x.device
    )

    # Scale tensor is always [num_tokens, num_heads] regardless of transpose_scale
    # The transpose_scale flag only affects the internal data layout/indexing
    scales = torch.empty((num_tokens, num_heads), dtype=torch.float32, device=x.device)

    # Call HIP kernel
    gated_rmsnorm_fp8_group_quant(
        out_quant, scales, x, z, weight, eps, group_size, transpose_scale
    )

    return out_quant, normalize_scales_layout(
        scales, num_tokens, num_heads, transpose_scale
    )


def calculate_bandwidth(num_tokens, num_heads, head_dim, time_us):
    """
    Calculate memory bandwidth in GB/s.

    Memory operations:
    - Read x: num_tokens * num_heads * head_dim * 2 bytes (bf16)
    - Read z: num_tokens * num_heads * head_dim * 2 bytes (bf16)
    - Read weight: head_dim * 2 bytes (bf16, broadcast)
    - Write out: num_tokens * num_heads * head_dim * 1 byte (fp8)
    - Write scales: num_tokens * num_heads * 4 bytes (fp32)

    Total bytes = num_tokens * num_heads * head_dim * (2 + 2 + 1) + head_dim * 2 + num_tokens * num_heads * 4
    """
    read_x = num_tokens * num_heads * head_dim * 2  # bf16
    read_z = num_tokens * num_heads * head_dim * 2  # bf16
    read_weight = head_dim * 2  # bf16 (broadcast)
    write_out = num_tokens * num_heads * head_dim * 1  # fp8
    write_scales = num_tokens * num_heads * 4  # fp32

    total_bytes = read_x + read_z + read_weight + write_out + write_scales
    time_s = time_us * 1e-6
    bandwidth_gbs = (total_bytes / time_s) / 1e9

    return bandwidth_gbs


def test_gated_rmsnorm_fp8_group_quant(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    eps: float = 1e-6,
    group_size: int = 128,
    quant_dtype=torch.float8_e4m3fnuz,
    transpose_scale: bool = False,
):
    """
    Test gated RMSNorm with FP8 group quantization.
    """
    torch.manual_seed(42)
    device = "cuda"

    # Validate constraints
    assert head_dim == 128, f"ONLY head_dim=128 is supported, got {head_dim}"
    assert group_size == 128, f"ONLY group_size=128 is supported, got {group_size}"

    # Generate test data
    x = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    z = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    weight = torch.randn(head_dim, dtype=dtype, device=device)

    print(f"\n{'='*80}")
    print("Test Configuration:")
    print(f"  Shape: [{num_tokens}, {num_heads}, {head_dim}]")
    print(f"  dtype: {dtype}, quant_dtype: {quant_dtype}")
    print(f"  group_size: {group_size}, transpose_scale: {transpose_scale}")
    print(f"  eps: {eps}")
    print(f"{'='*80}")

    # Run reference
    (ref_quant, ref_scales), ref_time = test_gated_rmsnorm_fp8_group_quant_reference(
        x.clone(), z.clone(), weight, eps, group_size, quant_dtype, transpose_scale
    )

    # Run HIP kernel
    (hip_quant, hip_scales), hip_time = test_gated_rmsnorm_fp8_group_quant_hip(
        x.clone(), z.clone(), weight, eps, group_size, quant_dtype, transpose_scale
    )

    # Calculate bandwidth
    ref_bw = calculate_bandwidth(num_tokens, num_heads, head_dim, ref_time)
    hip_bw = calculate_bandwidth(num_tokens, num_heads, head_dim, hip_time)

    # Verify results
    print("\nPerformance:")
    print(f"  Reference time: {ref_time:.2f} us  ({ref_bw:.2f} GB/s)")
    print(f"  HIP kernel time: {hip_time:.2f} us  ({hip_bw:.2f} GB/s)")
    print(f"  Speedup: {ref_time / hip_time:.2f}x")

    print("\nShape verification:")
    print(f"  Reference: quant={ref_quant.shape}, scales={ref_scales.shape}")
    print(f"  HIP kernel: quant={hip_quant.shape}, scales={hip_scales.shape}")

    # Verify shapes match
    assert (
        ref_quant.shape == hip_quant.shape
    ), f"Shape mismatch: ref={ref_quant.shape} vs hip={hip_quant.shape}"
    assert (
        ref_scales.shape == hip_scales.shape
    ), f"Scale shape mismatch: ref={ref_scales.shape} vs hip={hip_scales.shape}"

    print("\nVerifying quantized output...")
    print(f"  Ref quant sample (first token, first 5): {ref_quant[0, :5]}")
    print(f"  HIP quant sample (first token, first 5): {hip_quant[0, :5]}")

    # Dequantized comparison
    print("\nDequantized comparison:")

    ref_dequant = (
        ref_quant.float().view(ref_quant.shape[0], -1, 128) * ref_scales[:, :, None]
    )
    hip_dequant = (
        hip_quant.float().view(ref_quant.shape[0], -1, 128) * hip_scales[:, :, None]
    )

    checkAllclose(
        ref_dequant, hip_dequant, rtol=1e-2, atol=1e-2, msg="Dequantized values"
    )

    print("\nScale comparison:")
    checkAllclose(
        ref_scales.float(), hip_scales.float(), rtol=1e-3, atol=1e-3, msg="Scales"
    )

    print(f"\n{'='*80}")
    print("? Test PASSED!")
    print(f"{'='*80}\n")

    return {
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "ref_time_us": ref_time,
        "hip_time_us": hip_time,
        "ref_bw_gbs": ref_bw,
        "hip_bw_gbs": hip_bw,
        "speedup": ref_time / hip_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test HIP kernel for gated RMSNorm + FP8 group quant"
    )
    parser.add_argument("--num_tokens", type=int, default=None, help="Number of tokens")
    parser.add_argument("--num_heads", type=int, default=None, help="Number of heads")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])

    args = parser.parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    if args.num_tokens is not None and args.num_heads is not None:
        # Single test with command line args
        test_gated_rmsnorm_fp8_group_quant(
            num_tokens=args.num_tokens,
            num_heads=args.num_heads,
            head_dim=128,  # Fixed constraint
            dtype=dtype,
            transpose_scale=False,
        )
        test_gated_rmsnorm_fp8_group_quant(
            num_tokens=args.num_tokens,
            num_heads=args.num_heads,
            head_dim=128,  # Fixed constraint
            dtype=dtype,
            transpose_scale=True,
        )
    else:

        # Comprehensive benchmark configurations
        # head_dim=128, group_size=128 are fixed constraints
        test_configs = [
            # (num_tokens, num_heads, head_dim)
            (128, 32, 128),
            (256, 32, 128),
            (512, 32, 128),
            (1024, 32, 128),
            (2048, 32, 128),
            (4096, 32, 128),
            (8192, 32, 128),
            # Different head counts
            (1024, 16, 128),
            (1024, 64, 128),
            (2048, 16, 128),
            (2048, 64, 128),
        ]

        print("\n" + "=" * 80)
        print(
            "COMPREHENSIVE BENCHMARK - Gated RMSNorm + FP8 Group Quantization HIP Kernel"
        )
        print("=" * 80)

        results = []
        for num_tokens, num_heads, head_dim in test_configs:
            for transpose in [False, True]:
                result = test_gated_rmsnorm_fp8_group_quant(
                    num_tokens=num_tokens,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                    transpose_scale=transpose,
                )
                result["transpose_scale"] = transpose
                results.append(result)

        df = pd.DataFrame(results)
        df_md = df.to_markdown(index=False)
        aiter.logger.info(
            "gated_rmsnorm_fp8_group_quant summary (markdown):\n%s", df_md
        )

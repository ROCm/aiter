#!/usr/bin/env python3
"""
Test for gated RMSNorm with FP8 PER-TOKEN quantization - HIP kernel validation.

Tests the fused HIP kernel that performs:
1. Per-head RMSNorm(x, weight, eps)
2. Gating with SiLU: out = norm(x) * silu(z)
3. FP8 per-token quantization: one scale per token over the full flattened row
4. Flatten: [num_tokens, num_heads, head_dim] -> [num_tokens, num_heads*head_dim]

Constraint: ONLY supports head_dim=128 and num_heads <= 128.
"""

import argparse

import pandas as pd
import torch

import aiter
from aiter.test_common import checkAllclose, perftest


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def gated_rmsnorm_fp8_per_token_quant_reference_impl(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    quant_dtype,
):
    """Reference that matches the fused HIP kernel math and per-token quant path."""
    if quant_dtype == torch.float8_e4m3fnuz:
        fp8_max = 240.0
    elif quant_dtype == torch.float8_e4m3fn:
        fp8_max = 448.0
    else:
        raise ValueError(f"Unsupported FP8 dtype for this test: {quant_dtype}")

    num_tokens = x.shape[0]

    variance = x.float().pow(2).mean(-1, keepdim=True)
    inv_std = torch.rsqrt(variance + eps)
    normed = x.float() * inv_std
    normed = normed * weight.float().view(1, 1, -1)

    gated = normed * silu(z.float())  # [num_tokens, num_heads, head_dim]
    flat = gated.reshape(num_tokens, -1)  # [num_tokens, num_heads*head_dim]

    # One scale per token across the whole row.
    scales = flat.abs().amax(dim=-1) / fp8_max  # [num_tokens]
    scales = torch.maximum(scales, torch.full_like(scales, 1e-10))

    out_quant = torch.clamp(
        flat / scales.unsqueeze(-1), -fp8_max, fp8_max
    ).to(quant_dtype)
    return out_quant, scales


@perftest()
def run_reference(x, z, weight, eps, quant_dtype):
    return gated_rmsnorm_fp8_per_token_quant_reference_impl(
        x, z, weight, eps, quant_dtype
    )


@perftest()
def run_hip(x, z, weight, eps, quant_dtype):
    from aiter.ops.gated_rmsnorm_fp8_per_token_quant import (
        gated_rmsnorm_fp8_per_token_quant,
    )

    num_tokens, num_heads, head_dim = x.shape
    out_quant = torch.empty(
        num_tokens, num_heads * head_dim, dtype=quant_dtype, device=x.device
    )
    scales = torch.empty((num_tokens,), dtype=torch.float32, device=x.device)

    gated_rmsnorm_fp8_per_token_quant(out_quant, scales, x, z, weight, eps)
    return out_quant, scales


def calculate_bandwidth(num_tokens, num_heads, head_dim, time_us):
    read_x = num_tokens * num_heads * head_dim * 2  # bf16
    read_z = num_tokens * num_heads * head_dim * 2  # bf16
    read_weight = head_dim * 2  # bf16 (broadcast)
    write_out = num_tokens * num_heads * head_dim * 1  # fp8
    write_scales = num_tokens * 4  # fp32, one per token
    total_bytes = read_x + read_z + read_weight + write_out + write_scales
    return (total_bytes / (time_us * 1e-6)) / 1e9


def test_gated_rmsnorm_fp8_per_token_quant(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    eps: float = 1e-6,
    quant_dtype=torch.float8_e4m3fnuz,
):
    torch.manual_seed(42)
    device = "cuda"

    assert head_dim == 128, f"ONLY head_dim=128 is supported, got {head_dim}"
    assert num_heads <= 128, f"ONLY num_heads <= 128 is supported, got {num_heads}"

    x = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    z = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    weight = torch.randn(head_dim, dtype=dtype, device=device)

    print(f"\n{'='*80}")
    print("Test Configuration:")
    print(f"  Shape: [{num_tokens}, {num_heads}, {head_dim}]")
    print(f"  dtype: {dtype}, quant_dtype: {quant_dtype}, eps: {eps}")
    print(f"{'='*80}")

    (ref_quant, ref_scales), ref_time = run_reference(
        x.clone(), z.clone(), weight, eps, quant_dtype
    )
    (hip_quant, hip_scales), hip_time = run_hip(
        x.clone(), z.clone(), weight, eps, quant_dtype
    )

    ref_bw = calculate_bandwidth(num_tokens, num_heads, head_dim, ref_time)
    hip_bw = calculate_bandwidth(num_tokens, num_heads, head_dim, hip_time)

    print("\nPerformance:")
    print(f"  Reference time: {ref_time:.2f} us  ({ref_bw:.2f} GB/s)")
    print(f"  HIP kernel time: {hip_time:.2f} us  ({hip_bw:.2f} GB/s)")
    print(f"  Speedup: {ref_time / hip_time:.2f}x")

    assert (
        ref_quant.shape == hip_quant.shape
    ), f"Shape mismatch: ref={ref_quant.shape} vs hip={hip_quant.shape}"
    assert (
        ref_scales.shape == hip_scales.shape
    ), f"Scale shape mismatch: ref={ref_scales.shape} vs hip={hip_scales.shape}"

    # Dequantized comparison (scale broadcast per token across the whole row).
    ref_dequant = ref_quant.float() * ref_scales[:, None]
    hip_dequant = hip_quant.float() * hip_scales[:, None]
    checkAllclose(
        ref_dequant, hip_dequant, rtol=1e-2, atol=1e-2, msg="Dequantized values"
    )

    print("\nScale comparison:")
    checkAllclose(
        ref_scales.float(), hip_scales.float(), rtol=1e-3, atol=1e-3, msg="Scales"
    )

    print(f"\n{'='*80}\nTest PASSED!\n{'='*80}\n")

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
        description="Test HIP kernel for gated RMSNorm + FP8 per-token quant"
    )
    parser.add_argument("--num_tokens", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]

    if args.num_tokens is not None and args.num_heads is not None:
        test_gated_rmsnorm_fp8_per_token_quant(
            num_tokens=args.num_tokens,
            num_heads=args.num_heads,
            head_dim=128,
            dtype=dtype,
        )
    else:
        test_configs = [
            # (num_tokens, num_heads, head_dim) -- num_heads spans TP/DP variants
            (128, 32, 128),
            (256, 32, 128),
            (512, 32, 128),
            (1024, 32, 128),
            (2048, 32, 128),
            (4096, 32, 128),
            (8192, 32, 128),
            (1024, 16, 128),
            (1024, 64, 128),
            (2048, 16, 128),
            (2048, 64, 128),
        ]

        print("\n" + "=" * 80)
        print("BENCHMARK - Gated RMSNorm + FP8 Per-Token Quantization HIP Kernel")
        print("=" * 80)

        results = []
        for num_tokens, num_heads, head_dim in test_configs:
            results.append(
                test_gated_rmsnorm_fp8_per_token_quant(
                    num_tokens=num_tokens,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dtype=dtype,
                )
            )

        df = pd.DataFrame(results)
        aiter.logger.info(
            "gated_rmsnorm_fp8_per_token_quant summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

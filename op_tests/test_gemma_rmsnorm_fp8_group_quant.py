#!/usr/bin/env python3
"""
Test for Gemma RMSNorm + FP8 group quantization - HIP kernel validation.

Tests the fused HIP kernel that performs:
1. Optional residual add: x = x + residual (inplace write-back to residual)
2. Gemma RMSNorm: out = x * rsqrt(mean(x^2) + eps) * (1 + weight)
   - Variance over full hidden_size, Gemma-style (1+weight)
3. FP8 group quantization with group_size=128
4. Optional: also write unquantized normed output (out_normed)

Constraint: hidden_size must be a multiple of 128
"""

import pandas as pd
import torch
import aiter
from aiter.test_common import checkAllclose, perftest
import argparse


def gemma_rmsnorm_fp8_group_quant_reference_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    quant_dtype,
    residual: torch.Tensor | None = None,
):
    """
    Reference implementation matching the fused HIP kernel.

    Returns:
        (out_quant, scales, normed, residual_out)
        - out_quant: [num_tokens, hidden_size] in quant_dtype
        - scales: [num_tokens, num_groups] in float32
        - normed: [num_tokens, hidden_size] in input dtype (unquantized)
        - residual_out: [num_tokens, hidden_size] or None
    """
    if quant_dtype == torch.float8_e4m3fnuz:
        fp8_max = 240.0
    elif quant_dtype == torch.float8_e4m3fn:
        fp8_max = 448.0
    else:
        raise ValueError(f"Unsupported FP8 dtype: {quant_dtype}")

    num_tokens, hidden_size = x.shape
    group_size = 128
    num_groups = hidden_size // group_size

    # Step 1: Optional residual add
    x_f = x.float()
    residual_out = None
    if residual is not None:
        x_f = x_f + residual.float()
        residual_out = x_f.to(x.dtype)

    # Step 2: Gemma RMSNorm - variance over full hidden_size
    variance = x_f.pow(2).mean(dim=-1, keepdim=True)
    inv_std = torch.rsqrt(variance + eps)
    normed = x_f * inv_std * (1.0 + weight.float().unsqueeze(0))

    # Unquantized normed output in input dtype
    normed_out = normed.to(x.dtype)

    # Step 3: Per-group FP8 quantization
    normed_grouped = normed.view(num_tokens, num_groups, group_size)
    scales = normed_grouped.abs().amax(dim=-1) / fp8_max
    scales = torch.maximum(scales, torch.full_like(scales, 1e-10))

    out_quant = torch.clamp(
        normed_grouped / scales.unsqueeze(-1), -fp8_max, fp8_max
    ).to(quant_dtype)

    return out_quant.reshape(num_tokens, hidden_size), scales, normed_out, residual_out


@perftest()
def test_gemma_rmsnorm_fp8_group_quant_reference(
    x, weight, eps, group_size, quant_dtype, transpose_scale=False, residual=None
):
    """Reference implementation wrapped for perf timing."""
    del group_size, transpose_scale
    # Clone residual since reference modifies it inplace across warmup iterations
    res = residual.clone() if residual is not None else None
    out_q, scales, normed, res_out = gemma_rmsnorm_fp8_group_quant_reference_impl(
        x, weight, eps, quant_dtype, res
    )
    return out_q, scales, normed, res_out


@perftest()
def test_gemma_rmsnorm_fp8_group_quant_hip(
    x,
    weight,
    eps,
    group_size,
    quant_dtype,
    transpose_scale=False,
    residual=None,
    with_out_normed=False,
):
    """HIP kernel implementation wrapped for perf timing."""
    from aiter.ops.gemma_rmsnorm_fp8_group_quant import gemma_rmsnorm_fp8_group_quant

    num_tokens, hidden_size = x.shape
    num_groups = hidden_size // group_size

    out_quant = torch.empty(num_tokens, hidden_size, dtype=quant_dtype, device=x.device)
    scales = torch.empty(num_tokens, num_groups, dtype=torch.float32, device=x.device)

    # Clone residual since kernel modifies it inplace across warmup iterations
    res = residual.clone() if residual is not None else None

    out_normed = None
    if with_out_normed:
        out_normed = torch.empty(
            num_tokens, hidden_size, dtype=x.dtype, device=x.device
        )

    gemma_rmsnorm_fp8_group_quant(
        out_quant, scales, x, weight, eps, group_size, transpose_scale, res, out_normed
    )

    # Normalize scale layout for comparison
    if transpose_scale:
        scales = (
            scales.reshape(-1).view(num_groups, num_tokens).transpose(0, 1).contiguous()
        )

    return out_quant, scales, out_normed, res


def calculate_bandwidth(
    num_tokens, hidden_size, time_us, has_residual=False, has_out_normed=False
):
    """
    Calculate memory bandwidth in GB/s.

    Base:
    - Read x: num_tokens * hidden_size * 2 bytes (bf16)
    - Read weight: hidden_size * 2 bytes (bf16, broadcast)
    - Write out: num_tokens * hidden_size * 1 byte (fp8)
    - Write scales: num_tokens * (hidden_size/128) * 4 bytes (fp32)

    With residual (additional):
    - Read residual: num_tokens * hidden_size * 2 bytes (bf16)
    - Write residual: num_tokens * hidden_size * 2 bytes (bf16)

    With out_normed (additional):
    - Write out_normed: num_tokens * hidden_size * 2 bytes (bf16)
    """
    read_x = num_tokens * hidden_size * 2
    read_weight = hidden_size * 2
    write_out = num_tokens * hidden_size * 1
    num_groups = hidden_size // 128
    write_scales = num_tokens * num_groups * 4

    total_bytes = read_x + read_weight + write_out + write_scales

    if has_residual:
        total_bytes += num_tokens * hidden_size * 2  # read residual
        total_bytes += num_tokens * hidden_size * 2  # write residual

    if has_out_normed:
        total_bytes += num_tokens * hidden_size * 2  # write out_normed

    time_s = time_us * 1e-6
    bandwidth_gbs = (total_bytes / time_s) / 1e9

    return bandwidth_gbs


def test_gemma_rmsnorm_fp8_group_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    eps: float = 1e-6,
    group_size: int = 128,
    quant_dtype=torch.float8_e4m3fnuz,
    transpose_scale: bool = False,
    has_residual: bool = False,
    with_out_normed: bool = False,
):
    """Test Gemma RMSNorm + FP8 group quantization, optionally with out_normed."""
    torch.manual_seed(42)
    device = "cuda"

    assert (
        hidden_size % 128 == 0
    ), f"hidden_size must be multiple of 128, got {hidden_size}"

    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device) * 0.02
    residual = (
        torch.randn(num_tokens, hidden_size, dtype=dtype, device=device)
        if has_residual
        else None
    )

    print(f"\n{'='*80}")
    print("Test Configuration:")
    print(f"  Shape: [{num_tokens}, {hidden_size}]")
    print(f"  dtype: {dtype}, quant_dtype: {quant_dtype}")
    print(f"  group_size: {group_size}, transpose_scale: {transpose_scale}")
    print(f"  has_residual: {has_residual}, with_out_normed: {with_out_normed}")
    print(f"  eps: {eps}")
    print(f"{'='*80}")

    # Clone inputs for reference (residual is modified inplace)
    ref_residual = residual.clone() if residual is not None else None
    hip_residual = residual.clone() if residual is not None else None

    # Run reference
    (ref_quant, ref_scales, ref_normed, ref_res_out), ref_time = (
        test_gemma_rmsnorm_fp8_group_quant_reference(
            x.clone(),
            weight,
            eps,
            group_size,
            quant_dtype,
            transpose_scale,
            ref_residual,
        )
    )

    # Run HIP kernel
    (hip_quant, hip_scales, hip_normed, hip_res_out), hip_time = (
        test_gemma_rmsnorm_fp8_group_quant_hip(
            x.clone(),
            weight,
            eps,
            group_size,
            quant_dtype,
            transpose_scale,
            hip_residual,
            with_out_normed,
        )
    )

    # Calculate bandwidth
    ref_bw = calculate_bandwidth(
        num_tokens, hidden_size, ref_time, has_residual, with_out_normed
    )
    hip_bw = calculate_bandwidth(
        num_tokens, hidden_size, hip_time, has_residual, with_out_normed
    )

    print("\nPerformance:")
    print(f"  Reference time: {ref_time:.2f} us  ({ref_bw:.2f} GB/s)")
    print(f"  HIP kernel time: {hip_time:.2f} us  ({hip_bw:.2f} GB/s)")
    print(f"  Speedup: {ref_time / hip_time:.2f}x")

    print("\nShape verification:")
    print(f"  Reference: quant={ref_quant.shape}, scales={ref_scales.shape}")
    print(f"  HIP kernel: quant={hip_quant.shape}, scales={hip_scales.shape}")

    assert (
        ref_quant.shape == hip_quant.shape
    ), f"Shape mismatch: ref={ref_quant.shape} vs hip={hip_quant.shape}"
    assert (
        ref_scales.shape == hip_scales.shape
    ), f"Scale shape mismatch: ref={ref_scales.shape} vs hip={hip_scales.shape}"

    # Dequantized comparison
    num_groups = hidden_size // group_size
    print("\nDequantized comparison:")
    ref_dequant = (
        ref_quant.float().view(num_tokens, num_groups, group_size)
        * ref_scales[:, :, None]
    )
    hip_dequant = (
        hip_quant.float().view(num_tokens, num_groups, group_size)
        * hip_scales[:, :, None]
    )
    checkAllclose(
        ref_dequant, hip_dequant, rtol=1e-2, atol=1e-2, msg="Dequantized values"
    )

    print("\nScale comparison:")
    checkAllclose(
        ref_scales.float(), hip_scales.float(), rtol=1e-3, atol=1e-3, msg="Scales"
    )

    # Check out_normed if requested
    if with_out_normed:
        assert hip_normed is not None, "out_normed should not be None"
        print("\nout_normed comparison:")
        checkAllclose(
            ref_normed.float(),
            hip_normed.float(),
            rtol=1e-2,
            atol=0.05,
            msg="Unquantized normed output",
        )

    # Check residual if applicable
    if has_residual:
        print("\nResidual comparison:")
        checkAllclose(
            ref_res_out.float(),
            hip_res_out.float(),
            rtol=1e-2,
            atol=0.05,
            msg="Residual output",
        )

    print(f"\n{'='*80}")
    print("PASSED!")
    print(f"{'='*80}\n")

    return {
        "num_tokens": num_tokens,
        "hidden_size": hidden_size,
        "has_residual": has_residual,
        "transpose_scale": transpose_scale,
        "with_out_normed": with_out_normed,
        "ref_time_us": ref_time,
        "hip_time_us": hip_time,
        "ref_bw_gbs": ref_bw,
        "hip_bw_gbs": hip_bw,
        "speedup": ref_time / hip_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test HIP kernel for Gemma RMSNorm + FP8 group quant"
    )
    parser.add_argument("--num_tokens", type=int, default=None)
    parser.add_argument("--hidden_size", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    parser.add_argument("--residual", action="store_true", help="Test with residual")
    parser.add_argument(
        "--out-normed",
        action="store_true",
        help="Also output unquantized normed values",
    )

    args = parser.parse_args()
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.num_tokens is not None and args.hidden_size is not None:
        for transpose in [False, True]:
            for has_res in [False, True] if args.residual else [False]:
                test_gemma_rmsnorm_fp8_group_quant(
                    num_tokens=args.num_tokens,
                    hidden_size=args.hidden_size,
                    dtype=dtype,
                    transpose_scale=transpose,
                    has_residual=has_res,
                    with_out_normed=args.out_normed,
                )
    else:
        # Comprehensive benchmark: Gemma model hidden sizes
        test_configs = [
            # (num_tokens, hidden_size) - typical Gemma model dimensions
            # Gemma 2B
            (128, 2048),
            (256, 2048),
            (1024, 2048),
            (4096, 2048),
            (8192, 2048),
            # Gemma 7B
            (128, 3072),
            (256, 3072),
            (1024, 3072),
            (4096, 3072),
            (8192, 3072),
            # Gemma 2 27B
            (128, 4608),
            (256, 4608),
            (1024, 4608),
            (4096, 4608),
            (8192, 4608),
            # Gemma 3 27B
            (128, 3584),
            (1024, 3584),
            (4096, 3584),
        ]

        print("\n" + "=" * 80)
        print(
            "COMPREHENSIVE BENCHMARK - Gemma RMSNorm + FP8 Group Quantization HIP Kernel"
        )
        print("=" * 80)

        results = []
        for num_tokens, hidden_size in test_configs:
            for transpose in [False, True]:
                for has_res in [False, True]:
                    for out_normed in [False, True]:
                        result = test_gemma_rmsnorm_fp8_group_quant(
                            num_tokens=num_tokens,
                            hidden_size=hidden_size,
                            dtype=dtype,
                            transpose_scale=transpose,
                            has_residual=has_res,
                            with_out_normed=out_normed,
                        )
                        results.append(result)

        df = pd.DataFrame(results)
        df_md = df.to_markdown(index=False)
        aiter.logger.info(
            "gemma_rmsnorm_fp8_group_quant summary (markdown):\n%s", df_md
        )

#!/usr/bin/env python3
"""
Test for gated RMSNorm with BF16 output (no quantization) - HIP kernel validation.

Tests the fused HIP kernel that performs:
1. Per-head RMSNorm(x, weight, eps)
2. Gating with SiLU: out = norm(x) * silu(z)
3. Flatten: [num_tokens, num_heads, head_dim] -> [num_tokens, num_heads*head_dim]

Constraint: ONLY supports head_dim=128
"""

import argparse

import pandas as pd
import torch
import triton
import triton.language as tl

import aiter
from aiter.test_common import checkAllclose, perftest


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def gated_rmsnorm_bf16_reference(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Reference: norm(x) * silu(z), output flattened to [num_tokens, num_heads*head_dim]."""
    input_dtype = x.dtype
    variance = x.float().pow(2).mean(-1, keepdim=True)
    inv_std = torch.rsqrt(variance + eps)
    normed = x.float() * inv_std * weight.float().view(1, 1, -1)
    gated = normed * silu(z.float())
    return gated.to(input_dtype).reshape(x.shape[0], -1)


# ---------------------------------------------------------------------------
# Triton kernels (from ATOM PR#697 for comparison)
# ---------------------------------------------------------------------------
@triton.jit
def _rmsnorm_gated_triton_decode_kernel(
    x_ptr, z_ptr, weight_ptr, out_ptr,
    num_heads: tl.constexpr, eps: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)
    offsets = tl.arange(0, 128)
    row_offset = (token_id * num_heads + head_id) * 128
    x = tl.load(x_ptr + row_offset + offsets).to(tl.float32)
    z = tl.load(z_ptr + row_offset + offsets).to(tl.float32)
    w = tl.load(weight_ptr + offsets).to(tl.float32)
    variance = tl.sum(x * x, axis=0) * 0.0078125
    inv_rms = tl.rsqrt(variance + eps)
    gate = z * tl.sigmoid(z)
    out = x * inv_rms * w * gate
    tl.store(out_ptr + row_offset + offsets, out)


@triton.jit
def _rmsnorm_gated_triton_prefill_kernel(
    x_ptr, z_ptr, weight_ptr, out_ptr,
    num_rows: tl.constexpr, eps: tl.constexpr, block_rows: tl.constexpr,
):
    row_offsets = tl.program_id(0) * block_rows + tl.arange(0, block_rows)
    dim_offsets = tl.arange(0, 128)
    mask_rows = row_offsets < num_rows
    offsets = row_offsets[:, None] * 128 + dim_offsets[None, :]
    x = tl.load(x_ptr + offsets, mask=mask_rows[:, None], other=0.0).to(tl.float32)
    z = tl.load(z_ptr + offsets, mask=mask_rows[:, None], other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + dim_offsets).to(tl.float32)
    variance = tl.sum(x * x, axis=1) * 0.0078125
    inv_rms = tl.rsqrt(variance + eps)
    gate = z * tl.sigmoid(z)
    out = x * inv_rms[:, None] * w[None, :] * gate
    tl.store(out_ptr + offsets, out, mask=mask_rows[:, None])


def triton_gated_rmsnorm(x, z, weight, eps):
    """Run Triton gated rmsnorm kernel, return flattened bf16 output."""
    num_tokens, num_heads, head_dim = x.shape
    out = torch.empty(num_tokens, num_heads * head_dim, dtype=x.dtype, device=x.device)
    num_rows = num_tokens * num_heads
    if num_rows >= 65536:
        block_rows = 32
        _rmsnorm_gated_triton_prefill_kernel[(triton.cdiv(num_rows, block_rows),)](
            x, z, weight, out, num_rows, eps, block_rows, num_warps=4, num_stages=1,
        )
    else:
        _rmsnorm_gated_triton_decode_kernel[(num_tokens, num_heads)](
            x, z, weight, out, num_heads, eps, num_warps=1, num_stages=1,
        )
    return out


# ---------------------------------------------------------------------------
# Perf-wrapped callables
# ---------------------------------------------------------------------------
@perftest()
def bench_reference(x, z, weight, eps):
    return gated_rmsnorm_bf16_reference(x, z, weight, eps)


@perftest()
def bench_hip(x, z, weight, eps):
    from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_bf16

    num_tokens, num_heads, head_dim = x.shape
    out = torch.empty(num_tokens, num_heads * head_dim, dtype=x.dtype, device=x.device)
    gated_rmsnorm_bf16(out, x, z, weight, eps)
    return out


@perftest()
def bench_triton(x, z, weight, eps):
    return triton_gated_rmsnorm(x, z, weight, eps)


def calculate_bandwidth(num_tokens, num_heads, head_dim, time_us):
    """Calculate memory bandwidth in GB/s (bf16 in, bf16 out)."""
    read_x = num_tokens * num_heads * head_dim * 2
    read_z = num_tokens * num_heads * head_dim * 2
    read_weight = head_dim * 2
    write_out = num_tokens * num_heads * head_dim * 2  # bf16 output
    total_bytes = read_x + read_z + read_weight + write_out
    return (total_bytes / (time_us * 1e-6)) / 1e9


def run_test(num_tokens, num_heads, head_dim=128, dtype=torch.bfloat16, eps=1e-6):
    torch.manual_seed(42)
    device = "cuda"

    x = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    z = torch.randn(num_tokens, num_heads, head_dim, dtype=dtype, device=device)
    weight = torch.randn(head_dim, dtype=dtype, device=device)

    print(f"\n{'='*80}")
    print(f"Shape: [{num_tokens}, {num_heads}, {head_dim}], dtype: {dtype}, eps: {eps}")
    print(f"{'='*80}")

    # --- Correctness ---
    ref_out = gated_rmsnorm_bf16_reference(x, z, weight, eps)

    from aiter.ops.gated_rmsnorm_fp8_group_quant import gated_rmsnorm_bf16

    hip_out = torch.empty(num_tokens, num_heads * head_dim, dtype=dtype, device=device)
    gated_rmsnorm_bf16(hip_out, x, z, weight, eps)

    triton_out = triton_gated_rmsnorm(x, z, weight, eps)

    print("\nCorrectness (vs reference):")
    checkAllclose(ref_out, hip_out, rtol=5e-3, atol=5e-3, msg="HIP bf16")
    checkAllclose(ref_out, triton_out, rtol=5e-3, atol=5e-3, msg="Triton")

    # --- Performance ---
    _, ref_time = bench_reference(x.clone(), z.clone(), weight, eps)
    _, hip_time = bench_hip(x.clone(), z.clone(), weight, eps)
    _, triton_time = bench_triton(x.clone(), z.clone(), weight, eps)

    ref_bw = calculate_bandwidth(num_tokens, num_heads, head_dim, ref_time)
    hip_bw = calculate_bandwidth(num_tokens, num_heads, head_dim, hip_time)
    triton_bw = calculate_bandwidth(num_tokens, num_heads, head_dim, triton_time)

    print(f"\nPerformance:")
    print(f"  Reference:  {ref_time:8.2f} us  ({ref_bw:7.2f} GB/s)")
    print(f"  HIP bf16:   {hip_time:8.2f} us  ({hip_bw:7.2f} GB/s)")
    print(f"  Triton:     {triton_time:8.2f} us  ({triton_bw:7.2f} GB/s)")
    print(f"  HIP vs Triton speedup: {triton_time / hip_time:.2f}x")

    return {
        "num_tokens": num_tokens,
        "num_heads": num_heads,
        "ref_us": ref_time,
        "hip_us": hip_time,
        "triton_us": triton_time,
        "ref_bw": ref_bw,
        "hip_bw": hip_bw,
        "triton_bw": triton_bw,
        "hip_vs_triton": triton_time / hip_time,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test gated RMSNorm BF16 HIP kernel")
    parser.add_argument("--num_tokens", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    if args.num_tokens is not None and args.num_heads is not None:
        run_test(args.num_tokens, args.num_heads, dtype=dtype)
    else:
        configs = [
            # Qwen3.5 GDN layer shapes (24 heads, head_dim=128)
            # Benchmark aligned with profile_test_1/rmsnorm_gated_fusion
            (1, 24, 128),       # single-token decode
            (4, 24, 128),
            (16, 24, 128),
            (64, 24, 128),
            (128, 24, 128),
            (256, 24, 128),
            (512, 24, 128),
            (1024, 24, 128),
            (2048, 24, 128),
            (4096, 24, 128),
            (8192, 24, 128),
            (16384, 24, 128),   # large prefill (benchmark default)
        ]

        results = []
        for nt, nh, hd in configs:
            results.append(run_test(nt, nh, hd, dtype=dtype))

        df = pd.DataFrame(results)
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(df.to_markdown(index=False))

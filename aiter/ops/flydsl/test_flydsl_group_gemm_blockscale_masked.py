#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Tests for Masked Grouped FP8 GEMM kernel.

Tests the masked grouped FP8 GEMM with block scaling, matching DeepGEMM's
m_grouped_fp8_gemm_nt_masked API.

Usage:
    python aiter/ops/flydsl/test_flydsl_group_gemm_blockscale_masked.py
    pytest -q aiter/ops/flydsl/test_flydsl_group_gemm_blockscale_masked.py

Example external usage (e.g. from vLLM)::

    from aiter.ops.shuffle import shuffle_weight
    from aiter.ops.flydsl import flydsl_grouped_gemm_masked

    # At model load time — preshuffle weights ONCE:
    b_shuffled = shuffle_weight(b_fp8, layout=(16, 16))

    # At inference time — call with pre-shuffled weights (no re-shuffle):
    out = flydsl_grouped_gemm_masked(
        a_fp8, b_shuffled, scale_a, scale_b, masked_m,
    )
"""

from __future__ import annotations

import logging

import torch
import pytest

from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_weight

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL masked grouped GEMM tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.grouped_gemm_kernels import flydsl_grouped_gemm_masked
    from flydsl.runtime.device import get_rocm_arch
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL masked grouped GEMM kernels: {exc}",
        allow_module_level=True,
    )

logging.basicConfig(level=logging.INFO)

ARCH = get_rocm_arch()
# Use appropriate FP8 dtype for the architecture
DTYPE_FP8 = torch.float8_e4m3fn if "gfx95" in ARCH else torch.float8_e4m3fnuz


def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


def verify_output(out, ref, rtol=1e-2, atol=1e-2, msg=""):
    """Simple correctness check: element-wise allclose."""
    passed = torch.allclose(out, ref, rtol=rtol, atol=atol)
    if not passed:
        diff = (out - ref).abs()
        max_diff = diff.max().item()
        logging.warning(f"FAILED {msg}: max_diff={max_diff:.6f}")
    else:
        logging.info(f"PASSED {msg}")
    return passed


def quantize_a_masked_to_fp8(x: torch.Tensor, scale_block_k: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize padded 3D A tensor to FP8 with per-row, per-block scaling.

    Args:
        x: Input tensor [G, max_m, K]
        scale_block_k: K-dimension block size for scaling

    Returns:
        (x_fp8, scale): FP8 tensor and scale factors [G, scale_k, max_m]
    """
    G, max_m, K = x.shape
    nblk_k = K // scale_block_k

    # Reshape to [G, max_m, nblk_k, scale_block_k]
    x_blocks = x.view(G, max_m, nblk_k, scale_block_k)

    # Compute per-block max (for scale)
    x_amax = x_blocks.abs().amax(dim=-1).clamp(min=1e-12)

    fp8_max = torch.finfo(DTYPE_FP8).max
    scale = x_amax / fp8_max

    # Quantize
    x_scaled = x_blocks / scale.unsqueeze(-1)
    x_fp8 = x_scaled.to(DTYPE_FP8).view(G, max_m, K)

    # Transpose scale to [G, scale_k, max_m] to match kernel layout
    scale = scale.transpose(1, 2).contiguous()

    return x_fp8, scale


def quantize_b_to_fp8(
    b: torch.Tensor, scale_block_n: int = 128, scale_block_k: int = 128
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize B tensor to FP8 with per-block scaling.

    Args:
        b: Input tensor [num_groups, N, K]
        scale_block_n: N-dimension block size
        scale_block_k: K-dimension block size

    Returns:
        (b_fp8, scale_b): FP8 tensor and scale factors [num_groups, scale_n, scale_k]
    """
    num_groups, N, K = b.shape
    nblk_n = N // scale_block_n
    nblk_k = K // scale_block_k

    # Reshape to [num_groups, nblk_n, scale_block_n, nblk_k, scale_block_k]
    b_blocks = b.view(num_groups, nblk_n, scale_block_n, nblk_k, scale_block_k)

    # Compute per-block max
    b_amax = b_blocks.abs().amax(dim=(2, 4)).clamp(min=1e-12)

    fp8_max = torch.finfo(DTYPE_FP8).max
    scale = b_amax / fp8_max

    # Quantize
    b_scaled = b_blocks / scale.view(num_groups, nblk_n, 1, nblk_k, 1)
    b_fp8 = b_scaled.to(DTYPE_FP8).view(num_groups, N, K)

    return b_fp8, scale


def torch_masked_grouped_gemm_ref(
    a: torch.Tensor,
    scale_a: torch.Tensor,
    b: torch.Tensor,
    scale_b: torch.Tensor,
    masked_m: torch.Tensor,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
) -> torch.Tensor:
    """PyTorch reference implementation for masked grouped FP8 GEMM."""
    G, max_m, K = a.shape
    _, N, _ = b.shape
    nblk_k = K // scale_block_k
    nblk_n = N // scale_block_n

    # Dequantize A
    a_f32 = a.to(torch.float32)
    scale_a_t = scale_a.transpose(1, 2)
    a_scaled = a_f32.view(G, max_m, nblk_k, scale_block_k) * scale_a_t.unsqueeze(-1)
    a_scaled = a_scaled.view(G, max_m, K)

    # Dequantize B
    b_f32 = b.to(torch.float32)
    b_scaled = b_f32.view(G, nblk_n, scale_block_n, nblk_k, scale_block_k)
    b_scaled = b_scaled * scale_b.view(G, nblk_n, 1, nblk_k, 1)
    b_scaled = b_scaled.view(G, N, K)

    # Compute masked grouped GEMM on CPU
    a_scaled_cpu = a_scaled.cpu()
    b_scaled_cpu = b_scaled.cpu()
    m_cpu = masked_m.cpu()

    d = torch.zeros(G, max_m, N, dtype=torch.float32, device="cpu")
    for g in range(G):
        m_actual = m_cpu[g].item()
        if m_actual > 0:
            d[g, :m_actual, :] = a_scaled_cpu[g, :m_actual, :] @ b_scaled_cpu[g].T

    return d.to(torch.bfloat16).to(a.device)


def generate_masked_grouped_gemm_inputs(
    num_groups: int,
    max_m: int,
    expected_m_per_group: int,
    n: int,
    k: int,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    device: str = "cuda",
):
    """Generate test inputs for masked grouped GEMM.

    Returns:
        (a_fp8, scale_a, b_shuffled, scale_b, masked_m, ref_d)

    Note: ``b_shuffled`` is pre-shuffled via ``shuffle_weight`` — this
    simulates what a serving framework (vLLM) would do at model load time.
    The reference output ``ref_d`` is computed from the *unshuffled* B.
    """
    # Generate valid length array
    masked_m = torch.empty(num_groups, dtype=torch.int32, device=device)
    for g in range(num_groups):
        m_val = int(expected_m_per_group * (0.8 + 0.4 * torch.rand(1).item()))
        m_val = min(m_val, max_m)  # cap at max_m
        masked_m[g] = m_val

    # Generate random padded data
    a_f32 = torch.randn(num_groups, max_m, k, device=device, dtype=torch.float32)
    b_f32 = torch.randn(num_groups, n, k, device=device, dtype=torch.float32)

    # Quantize to FP8
    a_fp8, scale_a = quantize_a_masked_to_fp8(a_f32, scale_block_k)
    b_fp8, scale_b = quantize_b_to_fp8(b_f32, scale_block_n, scale_block_k)

    # Reference output (uses original unshuffled B)
    ref_d = torch_masked_grouped_gemm_ref(
        a_fp8, scale_a, b_fp8, scale_b, masked_m, scale_block_k, scale_block_n
    )

    # ── Preshuffle B ONCE (simulates model-load time) ──
    b_shuffled = shuffle_weight(b_fp8, layout=(16, 16))

    return a_fp8, scale_a, b_shuffled, scale_b, masked_m, ref_d


@pytest.mark.parametrize(
    "num_groups,max_m,expected_m,n,k",
    [
        pytest.param(1, 256, 100, 128, 128, id="single-group-small"),
        pytest.param(4, 256, 150, 128, 128, id="four-groups-small"),
        pytest.param(8, 512, 300, 256, 256, id="eight-groups-medium"),
        pytest.param(8, 1024, 800, 512, 512, id="eight-groups-larger"),
    ],
)
def test_masked_grouped_fp8_gemm_correctness(num_groups, max_m, expected_m, n, k,
                                              tile_m=128, tile_n=128, tile_k=128,
                                              out_dtype="bf16"):
    """Test masked grouped FP8 GEMM correctness against PyTorch reference."""
    scale_block_k = 128
    scale_block_n = 128

    # Generate inputs (b is pre-shuffled once here)
    a_fp8, scale_a, b_shuffled, scale_b, masked_m, ref_d = (
        generate_masked_grouped_gemm_inputs(
            num_groups, max_m, expected_m, n, k, scale_block_k, scale_block_n
        )
    )

    # ── Call public API (exactly as vLLM would) ──
    out = flydsl_grouped_gemm_masked(
        a_fp8,
        b_shuffled,       # pre-shuffled weight — NOT re-shuffled each call
        scale_a,
        scale_b,
        masked_m,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        scale_block_k=scale_block_k,
        scale_block_n=scale_block_n,
        out_dtype=out_dtype,
    )
    torch.cuda.synchronize()

    # Verify correctness — zero out padded rows for fair comparison
    c_out_f32 = out.to(torch.float32)
    c_ref = ref_d.to(torch.float32)
    for g in range(num_groups):
        m_val = masked_m[g].item()
        c_out_f32[g, m_val:, :] = 0.0
        c_ref[g, m_val:, :] = 0.0

    msg = f"num_groups={num_groups}, max_m={max_m}, N={n}, K={k}"
    passed = verify_output(c_out_f32, c_ref, rtol=1e-2, atol=1e-2, msg=msg)
    assert passed, f"Correctness check failed for {msg}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Masked Grouped FP8 GEMM benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--max_m", type=int, default=512)
    parser.add_argument("--expected_m", type=int, default=0,
                        help="Approx valid M rows per group (0 = sweep [128, 256, 384])")
    parser.add_argument("-N", type=int, default=512)
    parser.add_argument("-K", type=int, default=512)
    parser.add_argument("--tile_m", type=int, default=128)
    parser.add_argument("--tile_n", type=int, default=128)
    parser.add_argument("--tile_k", type=int, default=128)
    parser.add_argument("--out_dtype", type=str, default="bf16", choices=["bf16", "f16"])
    args = parser.parse_args()

    torch.set_default_device("cuda")

    m_list = [args.expected_m] if args.expected_m > 0 else [128, 256, 384]

    for expected_m in m_list:
        test_masked_grouped_fp8_gemm_correctness(args.num_groups, args.max_m, expected_m, args.N, args.K,
                                                  tile_m=args.tile_m, tile_n=args.tile_n,
                                                  tile_k=args.tile_k, out_dtype=args.out_dtype)

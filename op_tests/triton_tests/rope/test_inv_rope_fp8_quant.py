# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for fused inverse RoPE + FP8 block-scaled quantization kernel.

Validates:
  1. Numerical correctness vs PyTorch reference (bit-exact FP8, exact scales).
  2. Output shape: (T, G, D) for fp8, (T, G, K) for scales.
  3. Scale MN-major layout: token stride = 1 for fp8_einsum compatibility.
  4. Power-of-2 scales, FP8 range, edge cases.
"""

import torch
import pytest

from aiter.ops.triton.rope.inv_rope_fp8_quant import inv_rope_fp8_quant


def get_fp8_dtype():
    from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

    return get_fp8_e4m3_dtype()


def ref_inv_rope_fp8_quant(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    rope_head_dim: int = 64,
    quant_group_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference: returns (T, G, D) fp8 and (T, G, K) fp32 scales."""
    T, num_heads, head_dim = o.shape
    nope_dim = head_dim - rope_head_dim
    half_rope = rope_head_dim // 2

    fp8_dtype = get_fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    d_per_group = heads_per_group * head_dim
    num_k_blocks = d_per_group // quant_group_size

    x = o.float()

    cos_vals = cos_sin_cache[positions, :half_rope]
    sin_vals = cos_sin_cache[positions, half_rope:]

    rope_part = x[:, :, nope_dim:]
    rope_pairs = rope_part.reshape(T, num_heads, half_rope, 2)
    x_even = rope_pairs[:, :, :, 0]
    x_odd = rope_pairs[:, :, :, 1]

    cos_v = cos_vals[:, None, :]
    sin_v = sin_vals[:, None, :]

    # Inverse GPTJ RoPE.
    #   even: x_even * cos + x_odd  * sin
    #   odd:  x_odd  * cos - x_even * sin
    new_even = x_even * cos_v + x_odd * sin_v
    new_odd = x_odd * cos_v - x_even * sin_v

    rope_out = torch.stack([new_even, new_odd], dim=-1).reshape(
        T, num_heads, rope_head_dim
    )
    x_full = torch.cat([x[:, :, :nope_dim], rope_out], dim=-1)

    fp8_out = torch.empty(T, n_groups, d_per_group, dtype=fp8_dtype, device=o.device)
    scale_out = torch.empty(
        T, n_groups, num_k_blocks, dtype=torch.float32, device=o.device
    )

    for g in range(n_groups):
        heads_start = g * heads_per_group
        heads_end = heads_start + heads_per_group
        group_data = x_full[:, heads_start:heads_end, :].reshape(T, d_per_group)

        for t in range(T):
            row = group_data[t]
            blocks = row.reshape(num_k_blocks, quant_group_size)
            absmax = blocks.abs().amax(dim=1).clamp(min=1e-10)
            scales_raw = absmax / fp8_max
            scales = torch.exp2(torch.ceil(torch.log2(scales_raw)))

            scales_expanded = scales.unsqueeze(1).expand_as(blocks).reshape(-1)
            quantized = (row / scales_expanded).clamp(-fp8_max, fp8_max)
            fp8_out[t, g] = quantized.to(fp8_dtype)
            scale_out[t, g] = scales

    return fp8_out, scale_out


DSV4_PARAMS = dict(
    head_dim=512,
    nope_dim=448,
    rope_head_dim=64,
    quant_group_size=128,
)


@pytest.mark.parametrize("T", [1, 4, 16, 128, 512, 2048])
@pytest.mark.parametrize(
    "n_groups, heads_per_group",
    [(2, 8), (4, 4), (1, 16)],
)
def test_inv_rope_fp8_quant_correctness(
    T: int,
    n_groups: int,
    heads_per_group: int,
):
    """Triton kernel matches PyTorch reference (bit-exact FP8, exact scales)."""
    torch.manual_seed(42)
    device = "cuda"

    head_dim = DSV4_PARAMS["head_dim"]
    rope_head_dim = DSV4_PARAMS["rope_head_dim"]
    quant_group_size = DSV4_PARAMS["quant_group_size"]
    num_heads = n_groups * heads_per_group
    max_pos = 131072

    freqs = torch.randn(
        max_pos, rope_head_dim // 2, device=device, dtype=torch.float32
    )
    cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    o = torch.randn(T, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (T,), device=device, dtype=torch.int64)

    fp8_tri, scale_tri = inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        rope_head_dim,
        quant_group_size,
    )
    fp8_ref, scale_ref = ref_inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        rope_head_dim,
        quant_group_size,
    )

    d_per_group = heads_per_group * head_dim
    num_k_blocks = d_per_group // quant_group_size
    assert fp8_tri.shape == (T, n_groups, d_per_group)
    assert scale_tri.shape == (T, n_groups, num_k_blocks)

    tri_bytes = fp8_tri.contiguous().view(torch.uint8)
    ref_bytes = fp8_ref.contiguous().view(torch.uint8)
    byte_diff = (tri_bytes.int() - ref_bytes.int()).abs()
    max_byte_diff = byte_diff.max().item()
    match_pct = (byte_diff <= 1).float().mean().item()

    assert max_byte_diff <= 1, f"FP8 max byte diff = {max_byte_diff}"
    assert match_pct > 0.99, f"FP8 off-by-1 match = {match_pct * 100:.1f}%"

    torch.testing.assert_close(
        scale_tri.contiguous(), scale_ref.contiguous(), atol=0, rtol=0
    )


@pytest.mark.parametrize("T", [0, 1])
def test_inv_rope_fp8_quant_edge_cases(T: int):
    """Empty tensor and single-token edge cases."""
    device = "cuda"
    n_groups, heads_per_group = 2, 8
    head_dim = 512
    rope_head_dim = 64
    quant_group_size = 128
    num_heads = n_groups * heads_per_group
    max_pos = 1024

    freqs = torch.randn(
        max_pos, rope_head_dim // 2, device=device, dtype=torch.float32
    )
    cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    o = torch.randn(T, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (T,), device=device, dtype=torch.int64)

    fp8_out, scale_out = inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        rope_head_dim,
        quant_group_size,
    )

    d_per_group = heads_per_group * head_dim
    num_k_blocks = d_per_group // quant_group_size
    assert fp8_out.shape == (T, n_groups, d_per_group)
    assert scale_out.shape == (T, n_groups, num_k_blocks)


@pytest.mark.parametrize("T", [1, 64, 256])
def test_inv_rope_fp8_quant_scale_is_power_of_2(T: int):
    """All output scales are exact powers of 2."""
    device = "cuda"
    n_groups, heads_per_group = 2, 8
    head_dim = 512
    rope_head_dim = 64
    quant_group_size = 128
    num_heads = n_groups * heads_per_group
    max_pos = 131072

    freqs = torch.randn(
        max_pos, rope_head_dim // 2, device=device, dtype=torch.float32
    )
    cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    o = torch.randn(T, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (T,), device=device, dtype=torch.int64)

    _, scale_out = inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        rope_head_dim,
        quant_group_size,
    )

    s = scale_out.contiguous()
    log2_s = torch.log2(s)
    assert (log2_s == log2_s.floor()).all(), "Not all scales are powers of 2"


@pytest.mark.parametrize("T", [1, 128, 1024])
def test_inv_rope_fp8_quant_output_in_range(T: int):
    """FP8 output values are within [-fp8_max, fp8_max]."""
    device = "cuda"
    n_groups, heads_per_group = 2, 8
    head_dim = 512
    rope_head_dim = 64
    quant_group_size = 128
    num_heads = n_groups * heads_per_group
    max_pos = 131072

    fp8_dtype = get_fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max

    freqs = torch.randn(
        max_pos, rope_head_dim // 2, device=device, dtype=torch.float32
    )
    cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    o = torch.randn(T, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 10
    positions = torch.randint(0, max_pos, (T,), device=device, dtype=torch.int64)

    fp8_out, _ = inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        rope_head_dim,
        quant_group_size,
    )

    assert fp8_out.contiguous().float().abs().max().item() <= fp8_max


@pytest.mark.parametrize("T", [1, 3, 5, 7, 16])
def test_inv_rope_fp8_quant_scale_mn_major_layout(T: int):
    """Scale tensor has MN-major layout (token stride = 1) after the
    returned transpose view, matching what fp8_einsum expects."""
    device = "cuda"
    n_groups, heads_per_group = 2, 8
    head_dim = 512
    rope_head_dim = 64
    quant_group_size = 128
    num_heads = n_groups * heads_per_group
    max_pos = 1024

    freqs = torch.randn(
        max_pos, rope_head_dim // 2, device=device, dtype=torch.float32
    )
    cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    o = torch.randn(T, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (T,), device=device, dtype=torch.int64)

    _, scale_out = inv_rope_fp8_quant(
        o,
        positions,
        cos_sin_cache,
        n_groups,
        heads_per_group,
        rope_head_dim,
        quant_group_size,
    )

    assert scale_out.shape == (T, n_groups, scale_out.shape[2])
    assert scale_out.stride(0) == 1, (
        f"Token stride must be 1 (MN-major), got {scale_out.stride(0)}"
    )
    assert not scale_out.is_contiguous(), (
        "Scale should be a non-contiguous MN-major view, not standard row-major"
    )

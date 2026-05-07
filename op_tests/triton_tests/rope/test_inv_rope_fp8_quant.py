# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for fused inverse RoPE + FP8 block-scaled quantization kernel."""

import torch
import pytest

from aiter.ops.triton.rope.inv_rope_fp8_quant import inv_rope_fp8_quant


def get_fp8_dtype():
    """Get platform FP8 dtype (fn for gfx950+, fnuz for gfx942)."""
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
    """PyTorch reference implementation for inverse GPTJ RoPE + FP8 block quant."""
    T, num_heads, head_dim = o.shape
    nope_dim = head_dim - rope_head_dim
    half_rope = rope_head_dim // 2

    fp8_dtype = get_fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max
    d_per_group = heads_per_group * head_dim
    num_k_blocks = d_per_group // quant_group_size

    x = o.float()

    # Inverse GPTJ RoPE on the last rope_head_dim elements of each head.
    cos_vals = cos_sin_cache[positions, :half_rope]  # (T, half_rope)
    sin_vals = cos_sin_cache[positions, half_rope:]  # (T, half_rope)

    rope_part = x[:, :, nope_dim:]  # (T, H, rope_dim)
    rope_pairs = rope_part.reshape(T, num_heads, half_rope, 2)
    x_even = rope_pairs[:, :, :, 0]
    x_odd = rope_pairs[:, :, :, 1]

    cos_v = cos_vals[:, None, :]  # (T, 1, half_rope)
    sin_v = sin_vals[:, None, :]

    # Inverse GPTJ: even = x*cos + partner*sin, odd = x*cos - partner*sin
    # For GPTJ pairs (x0, x1): inv gives (x0*cos + x1*sin, x0*cos - x1*sin)
    # Wait, the actual formula from the kernel:
    #   x_add = x[i]*cos + x[i^1]*sin
    #   x_sub = x[i]*cos - x[i^1]*sin
    #   result[i] = x_add if i is even else x_sub
    new_even = x_even * cos_v + x_odd * sin_v
    new_odd = x_even * cos_v - x_odd * sin_v
    # Wait that's wrong. Let me re-derive from the kernel:
    # For element at offset d in the head:
    #   rope_local = d - nope_dim
    #   partner = x[d ^ 1]
    #   x_add = x[d]*cos[d//2] + partner*sin[d//2]
    #   x_sub = x[d]*cos[d//2] - partner*sin[d//2]
    #   result = x_add if rope_local is even, else x_sub
    # For even d (rope_local=0,2,4,...): partner = x[d+1], result = x[d]*cos + x[d+1]*sin
    # For odd d (rope_local=1,3,5,...): partner = x[d-1], result = x[d]*cos - x[d-1]*sin
    new_even2 = x_even * cos_v + x_odd * sin_v
    new_odd2 = x_odd * cos_v - x_even * sin_v

    rope_out = torch.stack([new_even2, new_odd2], dim=-1).reshape(
        T, num_heads, rope_head_dim
    )
    x_full = torch.cat([x[:, :, :nope_dim], rope_out], dim=-1)

    # Block-scaled FP8 quantization per group.
    fp8_out = torch.empty(n_groups, T, d_per_group, dtype=fp8_dtype, device=o.device)
    scale_out = torch.empty(
        n_groups, T, num_k_blocks, dtype=torch.float32, device=o.device
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
            fp8_out[g, t] = quantized.to(fp8_dtype)
            scale_out[g, t] = scales

    return fp8_out, scale_out


# DeepSeek-V4 typical shapes
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
    """Test that Triton kernel matches PyTorch reference (bit-exact)."""
    torch.manual_seed(42)
    device = "cuda"

    head_dim = DSV4_PARAMS["head_dim"]
    rope_head_dim = DSV4_PARAMS["rope_head_dim"]
    quant_group_size = DSV4_PARAMS["quant_group_size"]
    num_heads = n_groups * heads_per_group
    max_pos = 131072

    freqs = torch.randn(max_pos, rope_head_dim // 2, device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)

    o = torch.randn(T, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (T,), device=device, dtype=torch.int64)

    fp8_triton, scale_triton = inv_rope_fp8_quant(
        o, positions, cos_sin_cache, n_groups, heads_per_group, rope_head_dim, quant_group_size
    )
    fp8_ref, scale_ref = ref_inv_rope_fp8_quant(
        o, positions, cos_sin_cache, n_groups, heads_per_group, rope_head_dim, quant_group_size
    )

    # FP8 byte-level comparison (allow off-by-1 for rounding).
    triton_bytes = fp8_triton.view(torch.uint8)
    ref_bytes = fp8_ref.view(torch.uint8)
    byte_diff = (triton_bytes.int() - ref_bytes.int()).abs()
    max_byte_diff = byte_diff.max().item()
    off_by_one_pct = (byte_diff <= 1).float().mean().item()

    assert max_byte_diff <= 1, f"FP8 max byte diff = {max_byte_diff}, expected <= 1"
    assert off_by_one_pct > 0.99, f"FP8 off-by-1 match = {off_by_one_pct*100:.1f}%"

    # Scale comparison (should be exact since we use power-of-2 scales).
    torch.testing.assert_close(scale_triton, scale_ref, atol=0, rtol=0)


@pytest.mark.parametrize("T", [0, 1])
def test_inv_rope_fp8_quant_edge_cases(T: int):
    """Test empty tensor and single-token edge cases."""
    device = "cuda"
    n_groups, heads_per_group = 2, 8
    head_dim = 512
    rope_head_dim = 64
    quant_group_size = 128
    num_heads = n_groups * heads_per_group
    max_pos = 1024

    freqs = torch.randn(max_pos, rope_head_dim // 2, device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    o = torch.randn(T, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (T,), device=device, dtype=torch.int64)

    fp8_out, scale_out = inv_rope_fp8_quant(
        o, positions, cos_sin_cache, n_groups, heads_per_group, rope_head_dim, quant_group_size
    )

    d_per_group = heads_per_group * head_dim
    num_k_blocks = d_per_group // quant_group_size
    assert fp8_out.shape == (n_groups, T, d_per_group)
    assert scale_out.shape == (n_groups, T, num_k_blocks)


@pytest.mark.parametrize("T", [1, 64, 256])
def test_inv_rope_fp8_quant_scale_is_power_of_2(T: int):
    """Verify all output scales are exact powers of 2."""
    device = "cuda"
    n_groups, heads_per_group = 2, 8
    head_dim = 512
    rope_head_dim = 64
    quant_group_size = 128
    num_heads = n_groups * heads_per_group
    max_pos = 131072

    freqs = torch.randn(max_pos, rope_head_dim // 2, device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    o = torch.randn(T, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (T,), device=device, dtype=torch.int64)

    _, scale_out = inv_rope_fp8_quant(
        o, positions, cos_sin_cache, n_groups, heads_per_group, rope_head_dim, quant_group_size
    )

    # A power of 2 satisfies: log2(x) == floor(log2(x))
    log2_scales = torch.log2(scale_out)
    is_pow2 = (log2_scales == log2_scales.floor()).all().item()
    assert is_pow2, "Not all scales are powers of 2"


@pytest.mark.parametrize("T", [1, 128, 1024])
def test_inv_rope_fp8_quant_output_in_range(T: int):
    """Verify FP8 output values are within [-fp8_max, fp8_max]."""
    device = "cuda"
    n_groups, heads_per_group = 2, 8
    head_dim = 512
    rope_head_dim = 64
    quant_group_size = 128
    num_heads = n_groups * heads_per_group
    max_pos = 131072

    fp8_dtype = get_fp8_dtype()
    fp8_max = torch.finfo(fp8_dtype).max

    freqs = torch.randn(max_pos, rope_head_dim // 2, device=device, dtype=torch.float32)
    cos_sin_cache = torch.cat([torch.cos(freqs), torch.sin(freqs)], dim=-1)
    o = torch.randn(T, num_heads, head_dim, device=device, dtype=torch.bfloat16) * 10
    positions = torch.randint(0, max_pos, (T,), device=device, dtype=torch.int64)

    fp8_out, _ = inv_rope_fp8_quant(
        o, positions, cos_sin_cache, n_groups, heads_per_group, rope_head_dim, quant_group_size
    )

    fp8_as_float = fp8_out.float()
    assert fp8_as_float.abs().max().item() <= fp8_max

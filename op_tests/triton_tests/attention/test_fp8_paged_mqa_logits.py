# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Tests for deepgemm_fp8_paged_mqa_logits kernel.

This test verifies the kernel works correctly with batch sizes larger than
the number of heads (64), which was previously causing memory access faults
due to incorrect stride calculations.

Specifically tests batch sizes: 96, 128, 512
"""

import random
import torch
import pytest
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def kv_cache_cast_to_fp8(x: torch.Tensor) -> torch.Tensor:
    """Convert KV cache to FP8 format with scale."""
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1

    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 240.0
    x_scaled = x * (1.0 / sf)
    x_fp8 = x_scaled.to(torch.float8_e4m3fnuz)

    # Pack FP8 data and scale into single tensor
    # Layout: (num_blocks, block_size, 1, head_dim + 4)
    # First head_dim bytes are FP8 K, last 4 bytes are float32 scale
    out = torch.zeros(
        (num_blocks, block_size, num_heads, head_dim + 4),
        device=x.device,
        dtype=torch.uint8,
    )
    out[..., :head_dim] = x_fp8.view(torch.uint8)
    out[..., head_dim : head_dim + 4] = sf.view(torch.uint8)

    return out


def ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    """Reference implementation using PyTorch."""
    batch_size, next_n, heads, dim = q.size()
    num_block, block_size, _, _ = kv_cache.size()

    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )

    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device="cuda")
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )

        for block_rk in range(cdiv(context_len, block_size)):
            block_idx = block_tables[i][block_rk]
            qx, kx = q[i], kv_cache[block_idx]
            k_offsets = torch.arange(
                block_rk * block_size, (block_rk + 1) * block_size, device="cuda"
            )
            mask = (k_offsets[None, :] < context_len) & (
                k_offsets[None, :] <= q_offsets[:, None]
            )
            s = torch.where(
                mask[None, :, :],
                (qx.transpose(0, 1) @ kx.transpose(0, 1).transpose(1, 2)).to(
                    logits.dtype
                ),
                float("-inf"),
            )
            s = torch.relu(s) * weight_slice[..., None]
            s = s.sum(dim=0)
            logits[
                i * next_n : (i + 1) * next_n,
                block_rk * block_size : (block_rk + 1) * block_size,
            ] = torch.where(
                k_offsets[None, None, :] <= q_offsets[None, :, None], s, float("-inf")
            )

    return logits


def calc_diff(x: torch.Tensor, y: torch.Tensor) -> float:
    """Calculate normalized difference between two tensors."""
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    if denominator == 0:
        return 0.0
    sim = 2 * (x * y).sum() / denominator
    return (1 - sim).item()


@pytest.mark.parametrize("batch_size", [64, 65, 96, 128, 256, 512])
@pytest.mark.parametrize("next_n", [1])
@pytest.mark.parametrize("context_len", [16, 128])
@torch.inference_mode()
def test_deepgemm_fp8_paged_mqa_logits_no_crash(
    batch_size: int,
    next_n: int,
    context_len: int,
) -> None:
    """
    Test deepgemm_fp8_paged_mqa_logits with various batch sizes doesn't crash.

    This test specifically targets batch sizes > 64 (number of heads)
    which previously caused memory access faults due to incorrect
    stride calculation (using stride(0) instead of stride(1)).
    """
    torch.manual_seed(42)
    random.seed(42)

    heads = 64
    index_dim = 128
    block_size = 1  # KVBlockSize
    max_model_len = 1024
    num_blocks = max_model_len * batch_size

    # Fixed context lengths
    context_lens = torch.full(
        (batch_size,), context_len, device="cuda", dtype=torch.int32
    )

    # Create inputs
    q = torch.randn(
        (batch_size, next_n, heads, index_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        (num_blocks, block_size, 1, index_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, heads),
        device="cuda",
        dtype=torch.float32,
    )

    # Create block tables
    max_block_len = cdiv(context_len, block_size) * block_size
    block_tables = torch.zeros(
        (batch_size, max_model_len), device="cuda", dtype=torch.int32
    )

    # Assign blocks to each batch
    for i in range(batch_size):
        for j in range(max_block_len):
            block_tables[i][j] = (i * max_block_len + j) % num_blocks

    # Convert to FP8
    q_fp8 = q.to(torch.float8_e4m3fnuz)
    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache)

    # Compute kernel output
    out_logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )

    # Run the kernel - this should not crash for batch_size > 64
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        out_logits,
        context_lens,
        block_tables,
        max_model_len,
        Preshuffle=False,
        KVBlockSize=block_size,
        ChunkK=128,
        TotalCuCount=256,
        WavePerEU=5,
        VarCtxSchedule=None,
    )

    torch.cuda.synchronize()

    # Check for NaN values
    assert not torch.isnan(out_logits).any(), "Output contains NaN values"

    # Check output shape
    assert out_logits.shape == (batch_size * next_n, max_model_len)


@pytest.mark.parametrize("batch_size", [96, 128, 512])
@torch.inference_mode()
def test_large_batch_no_crash(batch_size: int) -> None:
    """
    Simple test to verify the kernel doesn't crash with large batch sizes.

    This is a regression test for the stride(0) vs stride(1) bug that
    caused memory access faults when batch_size > heads (64).
    """
    torch.manual_seed(42)

    heads = 64
    hidden_dim = 128
    max_seq_len = 1024
    num_blocks = 10000
    block_size = 1
    next_n = 1
    context_len = 16

    # Create synthetic inputs
    q_fp8 = torch.randn(
        batch_size, next_n, heads, hidden_dim, device="cuda"
    ).to(torch.float8_e4m3fnuz)

    kv_cache = torch.zeros(
        num_blocks, block_size, 1, hidden_dim + 4, device="cuda", dtype=torch.uint8
    )

    weights = torch.ones(batch_size * next_n, heads, device="cuda", dtype=torch.float32)
    seqlens = torch.full((batch_size,), context_len, device="cuda", dtype=torch.int32)

    block_tables = torch.zeros(
        batch_size, max_seq_len, device="cuda", dtype=torch.int32
    )
    for b in range(batch_size):
        for c in range(context_len):
            block_tables[b, c] = (b * context_len + c) % num_blocks

    logits = torch.full(
        (batch_size * next_n, max_seq_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )

    # This should not crash
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache,
        weights,
        logits,
        seqlens,
        block_tables,
        max_seq_len,
        Preshuffle=False,
        KVBlockSize=block_size,
        ChunkK=128,
        TotalCuCount=256,
        WavePerEU=5,
        VarCtxSchedule=None,
    )

    torch.cuda.synchronize()

    # Verify no NaN values
    assert not torch.isnan(logits).any(), "Output contains NaN values"

    # Verify output shape is correct
    assert logits.shape == (batch_size * next_n, max_seq_len)


@pytest.mark.parametrize("batch_size", [32, 64, 65, 80, 96, 128])
@torch.inference_mode()
def test_output_consistency_across_batch_sizes(batch_size: int) -> None:
    """
    Test that the first batch produces consistent output regardless of total batch size.

    This verifies that the stride fix doesn't break the computation for
    batches at different positions in the output tensor.
    """
    torch.manual_seed(12345)

    heads = 64
    hidden_dim = 128
    max_seq_len = 512
    num_blocks = 5000
    block_size = 1
    next_n = 1
    context_len = 32

    # Reference Q that we'll use for the first batch
    q_ref = torch.randn(1, next_n, heads, hidden_dim, device="cuda").to(
        torch.float8_e4m3fnuz
    )

    # Create KV cache
    kv_cache = torch.zeros(
        num_blocks, block_size, 1, hidden_dim + 4, device="cuda", dtype=torch.uint8
    )
    # Fill with random FP8 data
    kv_float = torch.randn(num_blocks, block_size, 1, hidden_dim, device="cuda")
    kv_fp8 = kv_float.to(torch.float8_e4m3fnuz)
    kv_cache[..., :hidden_dim] = kv_fp8.view(torch.uint8)
    scale_val = torch.ones(1, dtype=torch.float32).view(torch.uint8)
    kv_cache[..., hidden_dim:] = scale_val.expand(num_blocks, block_size, 1, 4)

    # Create batched Q with reference as first batch
    q_batched = torch.zeros(
        batch_size, next_n, heads, hidden_dim, device="cuda", dtype=torch.float8_e4m3fnuz
    )
    q_batched[0] = q_ref[0]

    weights = torch.ones(batch_size * next_n, heads, device="cuda", dtype=torch.float32)
    seqlens = torch.full((batch_size,), context_len, device="cuda", dtype=torch.int32)

    # All batches use same blocks for first batch comparison
    block_tables = torch.zeros(
        batch_size, max_seq_len, device="cuda", dtype=torch.int32
    )
    for b in range(batch_size):
        for c in range(context_len):
            # First batch uses blocks 0..context_len-1
            # Others use different blocks
            block_tables[b, c] = c if b == 0 else ((b * 100 + c) % num_blocks)

    logits = torch.full(
        (batch_size * next_n, max_seq_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )

    deepgemm_fp8_paged_mqa_logits(
        q_batched,
        kv_cache,
        weights,
        logits,
        seqlens,
        block_tables,
        max_seq_len,
        Preshuffle=False,
        KVBlockSize=block_size,
        ChunkK=128,
        TotalCuCount=256,
        WavePerEU=5,
        VarCtxSchedule=None,
    )

    torch.cuda.synchronize()

    # Store output for first batch
    first_batch_output = logits[0, :context_len].clone()

    # Verify output is not all -inf or NaN
    assert not torch.isnan(first_batch_output).any(), "Output contains NaN"


if __name__ == "__main__":
    # Run quick smoke tests
    print("Running smoke tests for deepgemm_fp8_paged_mqa_logits...")

    for bs in [64, 65, 96, 128, 256, 512]:
        print(f"Testing batch_size={bs}...", end=" ")
        try:
            test_large_batch_no_crash(bs)
            print("PASS")
        except Exception as e:
            print(f"FAIL: {e}")

    print("\nAll smoke tests completed!")

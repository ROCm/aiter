# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modifications Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Import-safe KV-cache shuffle shared by the unified-attention tests."""

import torch

from aiter.ops.triton.utils.types import e4m3_dtype


def shuffle_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
):
    """
    Shuffle key and value cache layout for optimized memory access.

        layout: (num_lanes, num_elements_per_thread)
            gfx1250: (16, 8) for BF16 and FP8.
            gfx950: (16, 8) for BF16 and (16, 16) for FP8.

        WMMA/MFMA instruction shape:
            BF16: 16x16x32
            FP8: 16x16x64
    """
    dtype = key_cache.dtype
    assert value_cache.dtype == dtype
    assert dtype in (torch.bfloat16, e4m3_dtype)

    num_blocks, block_size, num_kv_heads, head_size = key_cache.shape
    num_blocks_v, block_size_v, num_kv_heads_v, head_size_v = value_cache.shape
    assert block_size >= 16
    assert num_blocks == num_blocks_v
    assert num_kv_heads == num_kv_heads_v
    assert head_size == head_size_v
    assert block_size == block_size_v

    k_width = 16 // key_cache.element_size()
    key_cache_shuffled = key_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 3, 1)
    key_cache_shuffled = key_cache_shuffled.view(
        -1,
        num_kv_heads,
        head_size // k_width,
        k_width,
        block_size,
    )
    key_cache_shuffled = key_cache_shuffled.permute(0, 1, 2, 4, 3).contiguous()

    value_cache_shuffled = value_cache.view(
        -1, block_size, num_kv_heads, head_size
    ).permute(0, 2, 1, 3)
    value_cache_shuffled = value_cache_shuffled.view(
        -1,
        num_kv_heads,
        block_size // k_width,
        k_width,
        head_size,
    )
    value_cache_shuffled = value_cache_shuffled.permute(0, 1, 2, 4, 3).contiguous()

    return key_cache_shuffled, value_cache_shuffled

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the FlyDSL paged-attention Tile kernel."""

from __future__ import annotations

import pytest
import torch

from aiter import per_tensor_quant
from aiter.ops.flydsl import is_flydsl_available

pytest.importorskip("flydsl")

if not torch.cuda.is_available():
    pytest.skip("ROCm is not available", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip("FlyDSL is not available", allow_module_level=True)

from aiter.ops.flydsl.pa_decode_tile import pa_decode_tile  # noqa: E402


def _get_arch() -> str:
    return torch.cuda.get_device_properties(0).gcnArchName.lower().split(":")[0]


pytestmark = pytest.mark.skipif(
    not (_get_arch().startswith("gfx942") or _get_arch().startswith("gfx95")),
    reason="PA Tile requires gfx942 or gfx95x",
)


def _quant_dtype() -> torch.dtype:
    if _get_arch().startswith("gfx95"):
        return torch.float8_e4m3fn
    return torch.float8_e4m3fnuz


def _reference_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_query_heads, head_dim = query.shape
    block_size = key_cache.shape[2]
    num_kv_heads = key_cache.shape[1]
    query_group_size = num_query_heads // num_kv_heads
    softmax_scale = head_dim**-0.5
    output = torch.empty_like(query)

    for seq_idx in range(batch_size):
        context_length = int(context_lengths[seq_idx].item())
        token_ids = torch.arange(context_length, device=query.device)
        logical_pages = token_ids // block_size
        token_offsets = token_ids % block_size
        physical_pages = block_tables[seq_idx, logical_pages].long()

        keys = (
            key_cache[physical_pages, :, token_offsets, :].float() * key_scale.float()
        )
        values = (
            value_cache[physical_pages, :, :, token_offsets].float()
            * value_scale.float()
        )
        keys = keys.repeat_interleave(query_group_size, dim=1)
        values = values.repeat_interleave(query_group_size, dim=1)
        scores = (
            torch.einsum("hd,khd->hk", query[seq_idx].float(), keys) * softmax_scale
        )
        probs = torch.softmax(scores, dim=-1)
        output[seq_idx] = torch.einsum("hk,khd->hd", probs, values).to(query.dtype)
    return output


@pytest.mark.parametrize("block_size", [16, 64])
def test_pa_decode_tile_fp8(block_size: int) -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size = 3
    num_query_heads = 8
    num_kv_heads = 1
    head_dim = 128
    context_length = 257
    blocks_per_sequence = (context_length + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_sequence

    query = torch.empty(
        batch_size,
        num_query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.5, 0.5)
    key = torch.empty(
        num_blocks,
        num_kv_heads,
        block_size,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.5, 0.5)
    value = torch.empty(
        num_blocks,
        num_kv_heads,
        head_dim,
        block_size,
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.5, 0.5)

    quant_dtype = _quant_dtype()
    key_quant, key_scale = per_tensor_quant(key, quant_dtype=quant_dtype)
    value_quant, value_scale = per_tensor_quant(value, quant_dtype=quant_dtype)
    key_cache = (
        key_quant.view(
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim // 16,
            16,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    value_cache = value_quant.contiguous()
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(
        batch_size, blocks_per_sequence
    )
    context_lengths = torch.full(
        (batch_size,), context_length, dtype=torch.int32, device=device
    )
    output = torch.empty_like(query)

    pa_decode_tile(
        output,
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        key_scale,
        value_scale,
    )
    torch.cuda.synchronize()

    reference = _reference_attention(
        query,
        key_quant,
        value_quant,
        block_tables,
        context_lengths,
        key_scale,
        value_scale,
    )
    torch.testing.assert_close(output, reference, atol=3e-2, rtol=3e-2)

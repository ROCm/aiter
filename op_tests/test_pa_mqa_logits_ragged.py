#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.ops.triton.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits_ragged_k,
    deepgemm_fp8_paged_mqa_logits_stage1_ragged_k,
)


NEXT_N = 4
HEADS = 32
HEAD_DIM = 128
CONTEXT_LEN = 8
CHUNK_K = 64
QUERY_POSITIONS = (4, 5, 6, 7)


def _make_inputs():
    q_bits = torch.full(
        (1, NEXT_N, HEADS, HEAD_DIM), 56, dtype=torch.uint8, device="cuda"
    )
    q_fp8 = q_bits.view(dtypes.fp8)

    kv_bits = torch.full(
        (CONTEXT_LEN, 1, 1, HEAD_DIM + 4),
        56,
        dtype=torch.uint8,
        device="cuda",
    )
    kv_bits[..., HEAD_DIM:] = torch.tensor(
        [0, 0, 128, 63], dtype=torch.uint8, device="cuda"
    )
    kv_cache_fp8 = kv_bits.view(dtypes.fp8)

    weights = torch.ones((NEXT_N, HEADS), dtype=torch.float32, device="cuda")
    prefix_sum_context_lens = torch.tensor(
        [0, CONTEXT_LEN], dtype=torch.int32, device="cuda"
    )
    kv_indices = torch.arange(CONTEXT_LEN, dtype=torch.int32, device="cuda")
    return q_fp8, kv_cache_fp8, weights, prefix_sum_context_lens, kv_indices


def _expected_causal_mask(width):
    query_positions = torch.tensor(QUERY_POSITIONS, device="cuda")
    kv_positions = torch.arange(width, device="cuda")
    return kv_positions[None, :] <= query_positions[:, None]


def _assert_exact_causal_mask(out, expected):
    assert torch.equal(torch.isfinite(out), expected)
    assert bool(torch.isneginf(out[~expected]).all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_paged_mqa_logits_ragged_k_uses_per_query_causal_boundary():
    q, kv, weights, prefix_lens, kv_indices = _make_inputs()
    out = torch.full(
        (NEXT_N, CHUNK_K), float("nan"), dtype=torch.float32, device="cuda"
    )

    deepgemm_fp8_paged_mqa_logits_ragged_k(
        q,
        kv,
        weights,
        out,
        prefix_lens,
        kv_indices,
        CHUNK_K,
        ChunkK=CHUNK_K,
        SplitKV=1,
    )
    torch.cuda.synchronize()

    _assert_exact_causal_mask(out, _expected_causal_mask(CHUNK_K))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_paged_mqa_logits_stage1_ragged_k_uses_per_query_causal_boundary():
    q, kv, weights, prefix_lens, kv_indices = _make_inputs()
    out = torch.full(
        (HEADS, NEXT_N, CHUNK_K),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )

    deepgemm_fp8_paged_mqa_logits_stage1_ragged_k(
        q,
        kv,
        weights,
        out,
        prefix_lens,
        kv_indices,
        CHUNK_K,
    )
    torch.cuda.synchronize()

    expected = _expected_causal_mask(CHUNK_K).unsqueeze(0).expand(HEADS, -1, -1)
    _assert_exact_causal_mask(out, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

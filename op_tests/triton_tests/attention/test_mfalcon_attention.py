# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch

from aiter.ops.triton.attention.mfalcon_attention import (
    build_mfalcon_plan,
    mfalcon_attention,
)


def make_inputs(
    history_lengths,
    num_candidates,
    head_dim,
    dtype,
    num_heads=2,
    layout="contiguous",
):
    if isinstance(history_lengths, int):
        history_lengths = [history_lengths]
    if isinstance(num_candidates, int):
        num_candidates = [num_candidates] * len(history_lengths)
    if len(history_lengths) != len(num_candidates):
        raise ValueError("history_lengths and num_candidates must have equal length")

    seq_lengths = [
        history + candidates
        for history, candidates in zip(history_lengths, num_candidates)
    ]
    total_tokens = sum(seq_lengths)
    if layout == "contiguous":
        q = torch.randn((total_tokens, num_heads, head_dim), device="cuda", dtype=dtype)
        k = torch.randn((total_tokens, num_heads, head_dim), device="cuda", dtype=dtype)
        v = torch.randn((total_tokens, num_heads, head_dim), device="cuda", dtype=dtype)
    elif layout == "transposed":
        q = torch.randn(
            (total_tokens, head_dim, num_heads), device="cuda", dtype=dtype
        ).transpose(1, 2)
        k = torch.randn(
            (total_tokens, head_dim, num_heads), device="cuda", dtype=dtype
        ).transpose(1, 2)
        v = torch.randn(
            (total_tokens, head_dim, num_heads), device="cuda", dtype=dtype
        ).transpose(1, 2)
    elif layout == "sliced":
        q = torch.randn(
            (total_tokens, num_heads, head_dim + 8), device="cuda", dtype=dtype
        )[..., :head_dim]
        k = torch.randn(
            (total_tokens, num_heads, head_dim + 8), device="cuda", dtype=dtype
        )[..., :head_dim]
        v = torch.randn(
            (total_tokens, num_heads, head_dim + 8), device="cuda", dtype=dtype
        )[..., :head_dim]
    else:
        raise ValueError(f"unsupported layout: {layout}")

    seq_offsets = torch.tensor(
        [0] + [sum(seq_lengths[: i + 1]) for i in range(len(seq_lengths))],
        dtype=torch.int64,
        device="cuda",
    )
    num_candidates_tensor = torch.tensor(
        num_candidates, dtype=torch.int64, device="cuda"
    )
    return q, k, v, seq_offsets, num_candidates_tensor


def reference_attention(q, k, v, seq_offsets, num_candidates, return_lse=False):
    batch_size = num_candidates.numel()
    num_heads = q.shape[1]
    head_dim = q.shape[2]
    total_candidates = int(num_candidates.sum().item())
    out = torch.empty(
        (total_candidates, num_heads, head_dim), dtype=q.dtype, device=q.device
    )
    lse = (
        torch.empty((total_candidates, num_heads), dtype=torch.float32, device=q.device)
        if return_lse
        else None
    )
    out_offset = 0
    scale = 1.0 / math.sqrt(head_dim)

    for batch_index in range(batch_size):
        start = int(seq_offsets[batch_index].item())
        end = int(seq_offsets[batch_index + 1].item())
        seq_len = end - start
        candidate_count = int(num_candidates[batch_index].item())
        history_len = seq_len - candidate_count
        q_seq = q[start:end]
        k_seq = k[start:end]
        v_seq = v[start:end]

        scores = torch.einsum("ihd,jhd->hij", q_seq, k_seq).float() * scale
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device)
        if history_len > 0:
            mask[:history_len, :history_len] = torch.triu(
                torch.ones(
                    (history_len, history_len), dtype=torch.bool, device=q.device
                ),
                diagonal=1,
            )
        if candidate_count > 0:
            candidate_indices = torch.arange(candidate_count, device=q.device)
            mask[history_len:, :history_len] = False
            mask[history_len:, history_len:] = True
            mask[history_len + candidate_indices, history_len + candidate_indices] = (
                False
            )
        scores.masked_fill_(mask.unsqueeze(0), float("-inf"))

        probabilities = torch.softmax(scores, dim=-1)
        out_seq = torch.einsum("hij,jhd->ihd", probabilities, v_seq.float()).to(q.dtype)
        out[out_offset : out_offset + candidate_count] = out_seq[history_len:]
        if return_lse:
            lse_seq = torch.logsumexp(scores, dim=-1)
            lse[out_offset : out_offset + candidate_count] = lse_seq[
                :, history_len:
            ].transpose(0, 1)
        out_offset += candidate_count

    return (out, lse) if return_lse else out


@pytest.mark.parametrize("head_dim", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("num_candidates", [1, 8, 65, 128, 129])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_mfalcon_attention(head_dim, num_candidates, dtype):
    q, k, v, seq_offsets, num_candidates_tensor = make_inputs(
        [67], num_candidates, head_dim, dtype
    )
    out = mfalcon_attention(q, k, v, seq_offsets, num_candidates_tensor)
    ref = reference_attention(q, k, v, seq_offsets, num_candidates_tensor)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("head_dim", [16, 64, 128, 256])
@pytest.mark.parametrize("num_candidates", [1, 8, 33])
def test_mfalcon_attention_lse(head_dim, num_candidates):
    q, k, v, seq_offsets, num_candidates_tensor = make_inputs(
        [67], num_candidates, head_dim, torch.float16
    )
    out, lse = mfalcon_attention(
        q, k, v, seq_offsets, num_candidates_tensor, return_lse=True
    )
    ref_out, ref_lse = reference_attention(
        q, k, v, seq_offsets, num_candidates_tensor, return_lse=True
    )
    torch.testing.assert_close(out, ref_out, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-3)


def test_mfalcon_attention_explicit_tile():
    q, k, v, seq_offsets, num_candidates = make_inputs([67], 33, 64, torch.float16)
    out = mfalcon_attention(
        q, k, v, seq_offsets, num_candidates, block_m=16, block_n=16
    )
    ref = reference_attention(q, k, v, seq_offsets, num_candidates)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("head_dim", [64, 256])
@pytest.mark.parametrize("layout", ["transposed", "sliced"])
def test_mfalcon_attention_non_contiguous(head_dim, layout):
    q, k, v, seq_offsets, num_candidates = make_inputs(
        [67], 33, head_dim, torch.float16, layout=layout
    )
    out = mfalcon_attention(q, k, v, seq_offsets, num_candidates)
    ref = reference_attention(q, k, v, seq_offsets, num_candidates)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize(
    "history_lengths",
    [[63], [65], [300], [0, 300]],
)
def test_mfalcon_attention_histories(history_lengths):
    q, k, v, seq_offsets, num_candidates = make_inputs(
        history_lengths, 8, 64, torch.float16
    )
    out = mfalcon_attention(q, k, v, seq_offsets, num_candidates)
    ref = reference_attention(q, k, v, seq_offsets, num_candidates)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_mfalcon_attention_rejects_invalid_inputs():
    q, k, v, seq_offsets, num_candidates = make_inputs([67], 8, 64, torch.float16)
    with pytest.raises(ValueError):
        bad_q = q[..., :48]
        bad_k = k[..., :48]
        bad_v = v[..., :48]
        mfalcon_attention(bad_q, bad_k, bad_v, seq_offsets, num_candidates)
    with pytest.raises(ValueError):
        bad_offsets = seq_offsets.clone()
        bad_offsets[-1] += 1
        mfalcon_attention(q, k, v, bad_offsets, num_candidates)
    with pytest.raises(ValueError):
        bad_num_candidates = num_candidates.clone()
        bad_num_candidates[0] = 100
        mfalcon_attention(q, k, v, seq_offsets, bad_num_candidates)
    with pytest.raises(ValueError):
        mfalcon_attention(q, k, v, seq_offsets, num_candidates, block_m=8)


def test_mfalcon_attention_plan_conflict():
    q, k, v, seq_offsets, num_candidates = make_inputs([67], 8, 64, torch.float16)
    plan = build_mfalcon_plan(
        q, k, v, seq_offsets, num_candidates, block_m=16, block_n=16
    )
    with pytest.raises(ValueError):
        mfalcon_attention(q, k, v, seq_offsets, num_candidates, plan=plan, block_m=32)


def test_mfalcon_attention_plan_reuse():
    q, k, v, seq_offsets, num_candidates = make_inputs([67], 8, 64, torch.float16)
    plan = build_mfalcon_plan(q, k, v, seq_offsets, num_candidates)
    out_one = mfalcon_attention(q, k, v, seq_offsets, num_candidates, plan=plan)
    out_two = mfalcon_attention(q, k, v, seq_offsets, num_candidates, plan=plan)
    torch.testing.assert_close(out_one, out_two, atol=0.0, rtol=0.0)

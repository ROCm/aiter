# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import math

import pytest
import torch

import aiter

BLOCK_SIZE = 128


def make_delta_lut(batch, heads, seqlen_q, seqlen_k, active_blocks, device):
    q_blocks = math.ceil(seqlen_q / BLOCK_SIZE)
    kv_blocks = math.ceil(seqlen_k / BLOCK_SIZE)
    assert 1 <= active_blocks < kv_blocks

    lut = torch.zeros(
        batch, heads, q_blocks, kv_blocks, dtype=torch.int32, device=device
    )
    counts = torch.full(
        (batch, heads, q_blocks),
        active_blocks,
        dtype=torch.int32,
        device=device,
    )
    selected = torch.empty(
        batch, heads, q_blocks, active_blocks, dtype=torch.int64, device=device
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(7)
    for b in range(batch):
        for h in range(heads):
            for qb in range(q_blocks):
                indices = (
                    torch.randperm(kv_blocks, generator=generator, device=device)[
                        :active_blocks
                    ]
                    .sort()
                    .values
                )
                selected[b, h, qb] = indices
                lut[b, h, qb, 0] = indices[0]
                if active_blocks > 1:
                    lut[b, h, qb, 1:active_blocks] = (indices[1:] - indices[:-1]).to(
                        torch.int32
                    )
    return lut, counts, selected


def masked_dense_reference(q, k, v, selected):
    batch, heads, seqlen_q, _ = q.shape
    seqlen_k = k.size(2)
    q_blocks = math.ceil(seqlen_q / BLOCK_SIZE)

    block_mask = torch.zeros(
        batch, heads, seqlen_q, seqlen_k, dtype=torch.bool, device=q.device
    )
    for qb in range(q_blocks):
        q_start = qb * BLOCK_SIZE
        q_end = min(q_start + BLOCK_SIZE, seqlen_q)
        for b in range(batch):
            for h in range(heads):
                for block in selected[b, h, qb].tolist():
                    k_start = block * BLOCK_SIZE
                    k_end = min(k_start + BLOCK_SIZE, seqlen_k)
                    block_mask[b, h, q_start:q_end, k_start:k_end] = True

    scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) / math.sqrt(
        q.size(-1)
    )
    probs = torch.softmax(scores.masked_fill(~block_mask, -torch.inf), dim=-1)
    return torch.matmul(probs, v.float()).to(q.dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("active_blocks", [1, 3])
def test_vsa_sparse_attention_parity(dtype, active_blocks):
    device = torch.device("cuda")
    batch, heads, seqlen_q, seqlen_k, dim = 1, 2, 257, 481, 128
    torch.manual_seed(11)
    q = torch.randn(batch, heads, seqlen_q, dim, device=device, dtype=dtype)
    k = torch.randn(batch, heads, seqlen_k, dim, device=device, dtype=dtype)
    v = torch.randn_like(k)
    lut, counts, selected = make_delta_lut(
        batch, heads, seqlen_q, seqlen_k, active_blocks, device
    )

    actual = aiter.vsa_sparse_attention(q, k, v, lut, counts)
    expected = masked_dense_reference(q, k, v, selected)

    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


def test_vsa_sparse_attention_uses_current_stream():
    device = torch.device("cuda")
    q = torch.randn(1, 1, 129, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 1, 384, 128, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    lut, counts, selected = make_delta_lut(1, 1, 129, 384, 2, device)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = aiter.vsa_sparse_attention(q, k, v, lut, counts)
    torch.cuda.current_stream().wait_stream(stream)

    expected = masked_dense_reference(q, k, v, selected)
    torch.testing.assert_close(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda q, k, v, lut, counts: (q.float(), k, v, lut, counts),
            "float16 or bfloat16",
        ),
        (
            lambda q, k, v, lut, counts: (
                q.transpose(2, 3),
                k,
                v,
                lut,
                counts,
            ),
            "contiguous BHSD",
        ),
        (
            lambda q, k, v, lut, counts: (q, k, v, lut.long(), counts),
            "dtype int32",
        ),
        (
            lambda q, k, v, lut, counts: (q, k, v, lut[..., :-1], counts),
            "block_lut must have shape",
        ),
        (
            lambda q, k, v, lut, counts: (
                q,
                k,
                v,
                lut,
                counts.fill_(lut.size(-1)),
            ),
            "final slot is reserved",
        ),
    ],
)
def test_vsa_sparse_attention_validation(mutate, match):
    device = torch.device("cuda")
    q = torch.randn(1, 1, 128, 128, device=device, dtype=torch.float16)
    k = torch.randn(1, 1, 384, 128, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    lut, counts, _ = make_delta_lut(1, 1, 128, 384, 1, device)

    args = mutate(q, k, v, lut, counts)
    with pytest.raises(RuntimeError, match=match):
        aiter.vsa_sparse_attention(*args)

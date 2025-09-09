# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from aiter.ops.triton.prefill_attention import context_attention_fwd as prefill_triton


def input_helper(BATCH, SEQLEN, H, HEAD_DIM, dtype, absorb=False):
    if absorb:
        q = torch.randn((BATCH * SEQLEN, H, 512), device="cuda", dtype=dtype)
        k = torch.randn((BATCH * SEQLEN, 1, 512), device="cuda", dtype=dtype)
        v = torch.randn((BATCH * SEQLEN, 1, 512), device="cuda", dtype=dtype)
    else:
        q = torch.randn((BATCH * SEQLEN, H, HEAD_DIM), device="cuda", dtype=dtype)
        k = torch.randn((BATCH * SEQLEN, H, HEAD_DIM), device="cuda", dtype=dtype)
        v = torch.randn((BATCH * SEQLEN, H, HEAD_DIM), device="cuda", dtype=dtype)

    b_seq_len = torch.ones((BATCH,), dtype=torch.int32) * SEQLEN
    b_start_loc = torch.zeros((BATCH,), dtype=torch.int32)

    b_start_loc = torch.cat(
        [
            torch.tensor([0], dtype=torch.int32),
            b_seq_len.cumsum(dim=0, dtype=torch.int32),
        ]
    )
    b_start_loc = b_start_loc.to(device="cuda")
    b_seq_len = b_seq_len.to(device="cuda")

    return q, k, v, b_seq_len, b_start_loc[:BATCH]


def varlen_input_helper(
    BATCH, SEQLEN, H, HEAD_DIM, dtype, absorb=False, equal_seqlens=False
):
    if not equal_seqlens:
        max_seqlens = SEQLEN // BATCH
        seqlens = torch.randint(1, max_seqlens + 1, (BATCH,), dtype=torch.int32)
    else:
        seqlens = torch.full((BATCH,), SEQLEN // BATCH)

    # Calculate cumulative sequence lengths
    cu_seqlens = torch.cat(
        [torch.tensor([0], dtype=torch.int32), seqlens.cumsum(dim=0, dtype=torch.int32)]
    )
    cu_seqlens = cu_seqlens.to(device="cuda")
    seqlens = seqlens.to(device="cuda")
    # Initialize q, k, v with variable lengths
    total_seq = cu_seqlens[-1].item()

    if absorb:
        q = torch.randn((total_seq, H, 512), device="cuda", dtype=dtype)
        k = torch.randn((total_seq, 1, 512), device="cuda", dtype=dtype)
        v = torch.randn((total_seq, 1, 512), device="cuda", dtype=dtype)
    else:
        q = torch.randn((total_seq, H, HEAD_DIM), device="cuda", dtype=dtype)
        k = torch.randn((total_seq, H, HEAD_DIM), device="cuda", dtype=dtype)
        v = torch.randn((total_seq, H, HEAD_DIM), device="cuda", dtype=dtype)

    return q, k, v, seqlens, cu_seqlens[:BATCH]


@pytest.mark.parametrize(
    "Z, H, SEQLEN, HEAD_DIM",
    [
        (4, 48, 1024, 64),
        (1, 24, 8192, 64),
        (1, 4, 16384, 128),
        (2, 16, 1020, 128),
        (2, 16, 15498, 128),
        (4, 48, 1001, 64),
        (1, 8, 8081, 64),
        (1, 4, 16330, 128),
    ],
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("absorb", [False])
@pytest.mark.parametrize("varlen", [True, False])
def test_op_fwd(Z, H, SEQLEN, HEAD_DIM, causal, absorb, varlen, dtype=torch.float16):
    torch.manual_seed(20)
    if varlen:
        q, k, v, b_seq_len, b_start_loc = varlen_input_helper(
            Z, SEQLEN, H, HEAD_DIM, dtype, absorb
        )
    else:
        q, k, v, b_seq_len, b_start_loc = input_helper(
            Z, SEQLEN, H, HEAD_DIM, dtype, absorb
        )

    tri_out = torch.empty_like(q)
    ref_out = torch.empty_like(q)

    # triton implementation
    prefill_triton(q, k, v, tri_out, b_start_loc, b_seq_len, SEQLEN, causal)
    sm_scale = 1.0 / (HEAD_DIM**0.5)

    for i in range(0, Z):
        start_q, start_k = b_start_loc[i], b_start_loc[i]
        seqlen_q, seqlen_k = b_seq_len[i], b_seq_len[i]
        end_q, end_k = start_q + seqlen_q, start_k + seqlen_k
        scores = torch.einsum(
            "qhd,khd->qhk", q[start_q:end_q], k[start_k:end_k]
        ).float()
        if causal:
            # Apply causal mask to scores (set future positions to -inf)
            # Create a causal mask
            causal_mask = torch.triu(
                torch.ones(seqlen_q, seqlen_k, device=scores.device), diagonal=1
            ).bool()
            scores.masked_fill_(causal_mask.unsqueeze(1), float("-inf"))
        scores = scores.float()

        p = torch.softmax(scores * sm_scale, dim=-1).half()
        ref_out[start_q:end_q] = torch.einsum("qhk,khd->qhd", p, v[start_k:end_k])

    # compare
    torch.testing.assert_close(ref_out, tri_out, atol=2e-2, rtol=2e-2)

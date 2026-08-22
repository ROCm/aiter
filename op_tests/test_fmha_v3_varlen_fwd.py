# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Pytest for aiter.ops.mha.fmha_v3_varlen_fwd.

Running this test triggers the JIT build of module_fmha_v3_varlen_fwd
(compiled with -DFAV3_ON=1) the first time it executes on a ROCm/CUDA device.
"""

import math

import pytest
import torch

import aiter
from aiter.ops.mha import fmha_v3_varlen_fwd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_varlen_inputs(b, seqlens_q, seqlens_k, hq, hk, d, dtype, device):
    """Build packed (varlen) q/k/v tensors and cu_seqlens."""
    assert len(seqlens_q) == b and len(seqlens_k) == b

    total_q = sum(seqlens_q)
    total_k = sum(seqlens_k)

    q = torch.randn(total_q, hq, d, dtype=dtype, device=device)
    k = torch.randn(total_k, hk, d, dtype=dtype, device=device)
    v = torch.randn(total_k, hk, d, dtype=dtype, device=device)

    cu_seqlens_q = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(torch.tensor(seqlens_q, dtype=torch.int32), dim=0),
        ]
    ).to(device)
    cu_seqlens_k = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32),
            torch.cumsum(torch.tensor(seqlens_k, dtype=torch.int32), dim=0),
        ]
    ).to(device)

    max_seqlen_q = max(seqlens_q)
    max_seqlen_k = max(seqlens_k)
    min_seqlen_q = min(seqlens_q)

    return q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, min_seqlen_q


# ---------------------------------------------------------------------------
# Parametrise over a small set of dtypes / head configs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "hq, hk, d",
    [
        (4, 4, 64),   # MHA, head-dim 64
        (4, 2, 128),  # GQA, head-dim 128
    ],
)
def test_fmha_v3_varlen_fwd_basic(dtype, hq, hk, d):
    """
    Verify that fmha_v3_varlen_fwd:
      - succeeds and returns exactly 4 tensors,
      - output tensor has the expected shape / dtype / device,
      - softmax_lse has the expected shape / dtype / device.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA/ROCm device not available")

    device = torch.device("cuda")

    # b=2, two sequences with different lengths
    b = 2
    seqlens_q = [8, 12]
    seqlens_k = [8, 12]

    (
        q, k, v,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, min_seqlen_q,
    ) = _make_varlen_inputs(b, seqlens_q, seqlens_k, hq, hk, d, dtype, device)

    softmax_scale = 1.0 / math.sqrt(d)

    result = fmha_v3_varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        logits_soft_cap=0.0,
        zero_tensors=False,
        is_causal=False,
        window_size_left=-1,
        window_size_right=-1,
        return_softmax_lse=True,
        return_dropout_randval=False,
        how_v3_bf16_cvt=0,
    )

    # Must return 4 tensors: out, softmax_lse, p (dropout randval), rng_state
    assert len(result) == 4, f"Expected 4 outputs, got {len(result)}"

    out, softmax_lse, _p, _rng = result

    total_q = sum(seqlens_q)

    # out: [total_q, hq, d]
    assert out.shape == (total_q, hq, d), f"out shape mismatch: {out.shape}"
    assert out.dtype == dtype, f"out dtype mismatch: {out.dtype}"
    assert out.device.type == device.type, f"out device mismatch: {out.device}"

    # softmax_lse: [hq, total_q]
    assert softmax_lse.shape == (hq, total_q), (
        f"softmax_lse shape mismatch: {softmax_lse.shape}"
    )
    assert softmax_lse.dtype == torch.float32, (
        f"softmax_lse dtype mismatch: {softmax_lse.dtype}"
    )
    assert softmax_lse.device.type == device.type, (
        f"softmax_lse device mismatch: {softmax_lse.device}"
    )

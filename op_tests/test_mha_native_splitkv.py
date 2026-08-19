# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Targeted test for the native HIP D64 bf16 split-K FMHA forward
(`torch.ops.aiter.mha_fwd_native_splitkv`).

This path is only reached from the `mha_fwd` dispatcher when
`can_impl_fmha_native()` is true and the split heuristic returns num_splits>1,
so it has no direct op_test otherwise. Here we force num_splits>1 explicitly and
compare against an fp32 attention reference.
"""

import pytest
import torch

import aiter.ops.mha  # noqa: F401  (registers torch.ops.aiter.mha_fwd_native_splitkv)
from aiter import dtypes
from aiter.test_common import checkAllclose

D = 64  # native splitkv is D64 only


def ref_attn(q, k, v, scale, causal):
    # q: (B, Sq, Hq, D); k/v: (B, Sk, Hk, D). Reference in fp32.
    B, Sq, Hq, _ = q.shape
    Sk, Hk = k.shape[1], k.shape[2]
    g = Hq // Hk
    qf = q.float().permute(0, 2, 1, 3)  # B,Hq,Sq,D
    kf = k.float().permute(0, 2, 1, 3).repeat_interleave(g, dim=1)  # B,Hq,Sk,D
    vf = v.float().permute(0, 2, 1, 3).repeat_interleave(g, dim=1)
    attn = torch.matmul(qf, kf.transpose(-1, -2)) * scale  # B,Hq,Sq,Sk
    if causal:
        # bottom-right alignment (query i attends to keys <= i + (Sk - Sq))
        mask = torch.ones(Sq, Sk, device=q.device, dtype=torch.bool).tril(
            diagonal=Sk - Sq
        )
        attn = attn.masked_fill(~mask, float("-inf"))
    attn = attn.softmax(dim=-1)
    o = torch.matmul(attn, vf)  # B,Hq,Sq,D
    return o.permute(0, 2, 1, 3).contiguous()  # B,Sq,Hq,D


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "B,Sq,Sk,Hq,Hk",
    [
        (1, 128, 512, 8, 8),  # MHA
        (2, 96, 1024, 16, 4),  # GQA, non-square
        (1, 256, 2048, 8, 8),
    ],
)
def test_mha_native_splitkv(B, Sq, Sk, Hq, Hk, causal):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    torch.manual_seed(0)
    dev = "cuda"
    q = torch.randn(B, Sq, Hq, D, dtype=dtypes.bf16, device=dev)
    k = torch.randn(B, Sk, Hk, D, dtype=dtypes.bf16, device=dev)
    v = torch.randn(B, Sk, Hk, D, dtype=dtypes.bf16, device=dev)
    scale = 1.0 / (D**0.5)

    ns = min((Sk + 63) // 64, 8)
    assert ns > 1, "test must exercise the multi-split path"

    o, lse = torch.ops.aiter.mha_fwd_native_splitkv(
        q, k, v, None, scale, causal, False, ns
    )
    ref = ref_attn(q, k, v, scale, causal)
    checkAllclose(
        ref,
        o.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"B{B} Sq{Sq} Sk{Sk} Hq{Hq} Hk{Hk} causal={causal} ns={ns}",
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "-s"]))

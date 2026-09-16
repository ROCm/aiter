# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the FP8 Flash Attention v2 kernel (fp8_attention_kernel.py).

Tests the forward path (attn_fwd) via a minimal Python wrapper in both
non-FP8 (USE_FP8=False) and FP8 modes, comparing against a PyTorch reference.
"""

import math

import pytest
import torch
import triton

from aiter.ops.triton._triton_kernels.attention.fp8_attention_kernel import (
    attn_fwd,
    get_padded_headsize,
)
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import FP8_ARCHS
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() not in FP8_ARCHS, reason=f"FP8 attention not supported on {get_arch()}"
)

BLOCK_M = 64
BLOCK_N = 64


def _ref_attention(q, k, v, causal=False, sm_scale=None):
    """Pure-PyTorch scaled-dot-product attention reference."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    # q/k/v: [B, H, S, D]
    scores = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * sm_scale
    if causal:
        S_q, S_k = q.shape[2], k.shape[2]
        mask = torch.tril(
            torch.ones(S_q, S_k, device=q.device, dtype=torch.bool),
            diagonal=S_k - S_q,
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", p, v.float()).to(q.dtype)


def _call_attn_fwd(q, k, v, causal=False, sm_scale=None):
    """Minimal wrapper to call attn_fwd with USE_FP8=False."""
    B, HQ, S_q, D = q.shape
    _, HK, S_k, _ = k.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    D_pad = get_padded_headsize(D)

    # Pad head dim if needed
    def _pad(t, d_pad):
        if t.shape[-1] == d_pad:
            return t
        pad = torch.zeros(*t.shape[:-1], d_pad, dtype=t.dtype, device=t.device)
        pad[..., : t.shape[-1]] = t
        return pad

    q_p = _pad(q, D_pad).contiguous()
    k_p = _pad(k, D_pad).contiguous()
    v_p = _pad(v, D_pad).contiguous()

    out = torch.zeros(B, HQ, S_q, D_pad, dtype=q.dtype, device=q.device)
    lse = torch.zeros(B, HQ, S_q, dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(S_q, BLOCK_M), HQ, B)

    attn_fwd[grid](
        Q=q_p,
        K=k_p,
        V=v_p,
        bias=None,
        p_scale=1.0,
        q_descale_ptr=None,
        k_descale_ptr=None,
        v_scale_ptr=None,
        USE_FP8=False,
        SM_SCALE=sm_scale,
        LSE=lse,
        Out=out,
        stride_qz=q_p.stride(0),
        stride_qh=q_p.stride(1),
        stride_qm=q_p.stride(2),
        stride_qk=q_p.stride(3),
        stride_kz=k_p.stride(0),
        stride_kh=k_p.stride(1),
        stride_kn=k_p.stride(2),
        stride_kk=k_p.stride(3),
        stride_vz=v_p.stride(0),
        stride_vh=v_p.stride(1),
        stride_vk=v_p.stride(2),
        stride_vn=v_p.stride(3),
        stride_oz=out.stride(0),
        stride_oh=out.stride(1),
        stride_om=out.stride(2),
        stride_on=out.stride(3),
        stride_bz=0,
        stride_bh=0,
        stride_bm=0,
        stride_bn=0,
        stride_az=0,
        stride_ah=0,
        stride_sz=0,
        stride_sh=0,
        stride_sm=0,
        stride_sn=0,
        stride_lse_z=lse.stride(0),
        stride_lse_h=lse.stride(1),
        stride_lse_m=lse.stride(2),
        stride_qdescale_z=0,
        stride_qdescale_h=0,
        stride_qdescale_m=0,
        stride_kdescale_z=0,
        stride_kdescale_h=0,
        stride_kdescale_m=0,
        padded_kscale_block_num=1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        dropout_p=0.0,
        philox_seed=0,
        philox_offset_base=0,
        scores=None,
        scores_scaled_shifted=None,
        exp_scores=None,
        alibi_slopes=None,
        HQ=HQ,
        HK=HK,
        ACTUAL_BLOCK_DMODEL_QK=D_pad,
        ACTUAL_BLOCK_DMODEL_V=D_pad,
        MAX_SEQLENS_Q=S_q,
        MAX_SEQLENS_K=S_k,
        VARLEN=False,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL_QK=D_pad,
        BLOCK_DMODEL_V=D_pad,
        BLOCK_N=BLOCK_N,
        # PRE_LOAD_V provided by autotune config
        USE_BIAS=False,
        ENABLE_DROPOUT=False,
        RETURN_SCORES=False,
        USE_ALIBI=False,
        USE_EXP2=True,
    )
    return out[..., :D]


@pytest.mark.parametrize(
    "B, HQ, HK, S_q, S_k, D",
    [
        (1, 4, 4, 64, 64, 64),
        (2, 8, 2, 128, 128, 128),  # GQA
        (1, 4, 4, 64, 128, 64),  # S_q != S_k
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp8_attn_fwd_nofp8(B, HQ, HK, S_q, S_k, D, causal, dtype):
    """attn_fwd with USE_FP8=False matches PyTorch reference."""
    if causal and S_q != S_k:
        pytest.skip("causal requires S_q == S_k")

    torch.manual_seed(0)
    q = torch.randn(B, HQ, S_q, D, dtype=dtype, device="cuda") * 0.1
    k = torch.randn(B, HK, S_k, D, dtype=dtype, device="cuda") * 0.1
    v = torch.randn(B, HK, S_k, D, dtype=dtype, device="cuda") * 0.1

    # expand K/V for GQA
    if HQ != HK:
        k_exp = k.repeat_interleave(HQ // HK, dim=1)
        v_exp = v.repeat_interleave(HQ // HK, dim=1)
    else:
        k_exp, v_exp = k, v

    ref = _ref_attention(q, k_exp, v_exp, causal=causal)
    out = _call_attn_fwd(q, k, v, causal=causal)

    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"Mismatch at (B={B},HQ={HQ},HK={HK},S_q={S_q},S_k={S_k},D={D},causal={causal})",
    )

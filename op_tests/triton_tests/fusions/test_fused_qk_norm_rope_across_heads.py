# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.fusions.fused_qk_norm_rope_across_heads import (
    fused_qk_norm_rope_across_heads,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

# Archs with a fused_qk_norm_rope_across_heads config.
SUPPORTED_ARCHS = ("gfx1150", "gfx1151", "gfx1201")


def torch_apply_rotary_emb(x, cos, sin):
    """apply_rotary_emb of the diffusers Wan attention; x is (B, S, H, D)."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    c = cos[..., 0::2]
    s = sin[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out.type_as(x)


def torch_qk_norm_rope_ref(q, k, q_weight, k_weight, eps, head_dim, cos=None, sin=None):
    """norm_q / norm_k (RMSNorm over all heads), then RoPE on each head."""
    N = q.shape[-1]
    qn = F.rms_norm(q, (N,), q_weight, eps)
    kn = F.rms_norm(k, (N,), k_weight, eps)
    if cos is None:
        return qn, kn
    qh = qn.unflatten(-1, (-1, head_dim))
    kh = kn.unflatten(-1, (-1, head_dim))
    tab = (1, -1, 1, head_dim)
    qo = torch_apply_rotary_emb(qh, cos.view(tab), sin.view(tab))
    ko = torch_apply_rotary_emb(kh, cos.view(tab), sin.view(tab))
    return qo.flatten(-2), ko.flatten(-2)


def generate_qk_norm_rope_inputs(
    B, S_q, S_k, heads, head_dim, dtype, rope, fused_qkv, interleaved=False
):
    """q, k, weights and fp32 RoPE tables; shared with the benchmark.

    By default the tables are random, with different even and odd columns, so
    a wrong column fails the test. With interleaved, each pair of columns has
    one value, as in the Wan tables. With fused_qkv, q and k are chunks of one
    packed row.
    """
    torch.manual_seed(0)
    N = heads * head_dim
    if fused_qkv:
        qkv = torch.randn((B, S_q, 3 * N), dtype=dtype, device="cuda")
        q, k, _ = qkv.chunk(3, dim=-1)
    else:
        q = torch.randn((B, S_q, N), dtype=dtype, device="cuda")
        k = torch.randn((B, S_k, N), dtype=dtype, device="cuda")
    q_weight = 1.0 + 0.5 * torch.randn(N, dtype=dtype, device="cuda")
    k_weight = 1.0 + 0.5 * torch.randn(N, dtype=dtype, device="cuda")
    cos = sin = None
    if rope:
        if interleaved:
            angle = 6.28 * torch.rand(
                (1, S_q, 1, head_dim // 2), dtype=torch.float32, device="cuda"
            )
            angle = angle.repeat_interleave(2, dim=-1)
            cos, sin = torch.cos(angle), torch.sin(angle)
        else:
            angle = 6.28 * torch.rand(
                (1, S_q, 1, head_dim), dtype=torch.float32, device="cuda"
            )
            cos = torch.cos(angle)
            sin = torch.sin(angle + 0.3)
    return q, k, q_weight, k_weight, cos, sin


_SHAPES = [
    (1, 1560, 1560, 12, 128),
    (1, 1560, 512, 12, 128),
    (2, 333, 333, 24, 128),
    (1, 777, 512, 40, 128),
    (1, 64, 64, 3, 64),
    (1, 5, 0, 12, 128),
]
# S_k != S_q is cross-attention (Wan attn2): no RoPE, and q and k come from
# different inputs, so no fused QKV.
_CASES = [
    (*shape, rope, fused_qkv)
    for shape in _SHAPES
    for rope in (True, False)
    for fused_qkv in (False, True)
    if shape[1] == shape[2] or not (rope or fused_qkv)
]


@pytest.mark.parametrize("B, S_q, S_k, heads, head_dim, rope, fused_qkv", _CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fused_qk_norm_rope_across_heads(
    B, S_q, S_k, heads, head_dim, rope, fused_qkv, dtype
):
    if get_arch() not in SUPPORTED_ARCHS:
        pytest.skip(f"no fused_qk_norm_rope_across_heads config for {get_arch()}")
    q, k, wq, wk, cos, sin = generate_qk_norm_rope_inputs(
        B, S_q, S_k, heads, head_dim, dtype, rope, fused_qkv
    )
    eps = 1e-6
    ref_q, ref_k = torch_qk_norm_rope_ref(q, k, wq, wk, eps, head_dim, cos, sin)
    out_q, out_k = fused_qk_norm_rope_across_heads(
        q, k, wq, wk, eps, head_dim, cos, sin
    )
    for out, ref in ((out_q, ref_q), (out_k, ref_k)):
        assert out.shape == ref.shape and out.dtype == ref.dtype and out.is_contiguous()
        if dtype == torch.float32:
            torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
        else:
            # The fp32 sum order differs from PyTorch, so a normalized value can
            # round one step apart before the RoPE.
            torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)

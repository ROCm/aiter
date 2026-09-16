# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the MXFP8 Flash Attention v2 kernel.

Covers:
  * non-quantized forward (use_mxfp8=False) vs PyTorch
  * quantized forward (use_mxfp8=True, e4m3 + e8m0 2D block scales)
  * backward (preprocess / dq / dkdv) with and without MXFP8, including GQA
    and causal masks.

Only runs on gfx950 (CDNA4).
"""

import math

import pytest
import torch

from aiter.ops.triton.attention.mxfp8_attention import (
    mxfp8_attention_backward,
    mxfp8_attention_forward,
)
from aiter.ops.triton.quant.quant_mxfp8 import convert_from_mxfp8, convert_to_mxfp8
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() != "gfx950",
    reason=f"MXFP8 Flash Attention requires gfx950 (CDNA4), got {get_arch()}",
)

_QUANT_BLOCK = 32
_FP8 = torch.float8_e4m3fn


def _ref_attention(q, k, v, causal=False, sm_scale=None):
    """Pure-PyTorch scaled-dot-product attention. q: [B, HQ, S, D], k/v: [B, HK, S, D]."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])
    hq, hk = q.shape[1], k.shape[1]
    if hq != hk:
        k = k.repeat_interleave(hq // hk, dim=1)
        v = v.repeat_interleave(hq // hk, dim=1)
    scores = torch.einsum("bhmd,bhnd->bhmn", q.float(), k.float()) * sm_scale
    if causal:
        s_q, s_k = q.shape[2], k.shape[2]
        mask = torch.tril(
            torch.ones(s_q, s_k, device=q.device, dtype=torch.bool),
            diagonal=s_k - s_q,
        )
        scores = scores.masked_fill(~mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhmn,bhnd->bhmd", p, v.float()).to(q.dtype)


def _dummy_scales(b, h, s, d, quant_block_size, device):
    """Dummy e8m0 scales (all 127 = 1.0) for the use_mxfp8=False path."""
    scale_blocks = (d + quant_block_size - 1) // quant_block_size
    return torch.full((b, h, s, scale_blocks), 127, dtype=torch.uint8, device=device)


def _quantize_bshd(x, quant_block_size=_QUANT_BLOCK):
    """MXFP8 2D-block quantize a bshd tensor.

    x: (B, S, H, D) -> fp8 (B, S, H, D), e8m0 (B, S/qbs, H, D/qbs).

    The kernel indexes scales as seq//qbs and dim//qbs, so scales are 2D
    blocked along both sequence and head dim. Flattening (B, H, S, D) to
    (B*H*S, D) keeps each head's tokens contiguous, so 32-row blocks stay
    inside a single head when S % qbs == 0.
    """
    b, s, h, d = x.shape
    assert s % quant_block_size == 0 and d % quant_block_size == 0
    x2 = x.permute(0, 2, 1, 3).contiguous().reshape(b * h * s, d)
    y, scale = convert_to_mxfp8(
        x2,
        _FP8,
        quant_block_size=quant_block_size,
        is_2d_block=True,
    )
    y = y.reshape(b, h, s, d).permute(0, 2, 1, 3).contiguous()
    scale = (
        scale.reshape(b, h, s // quant_block_size, d // quant_block_size)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return y, scale


def _dequantize_bshd(y, scale, dtype, quant_block_size=_QUANT_BLOCK):
    b, s, h, d = y.shape
    y2 = y.permute(0, 2, 1, 3).contiguous().reshape(b * h * s, d)
    s2 = (
        scale.permute(0, 2, 1, 3)
        .contiguous()
        .reshape(b * h * (s // quant_block_size), d // quant_block_size)
    )
    x = convert_from_mxfp8(
        y2,
        s2,
        dtype,
        quant_block_size=quant_block_size,
        is_2d_block=True,
    )
    return x.reshape(b, h, s, d).permute(0, 2, 1, 3).contiguous()


def _bshd_to_bhsd(t):
    return t.permute(0, 2, 1, 3).contiguous()


def _bhsd_to_bshd(t):
    return t.permute(0, 2, 1, 3).contiguous()


def _fwd(
    q,
    k,
    v,
    q_scale,
    k_scale,
    v_scale,
    sm_scale,
    causal,
    use_mxfp8,
    quant_block_size=_QUANT_BLOCK,
):
    return mxfp8_attention_forward(
        q,
        k,
        v,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=sm_scale,
        causal=causal,
        use_mxfp8=use_mxfp8,
        block_m=64,
        block_n=64,
        quant_block_size=quant_block_size,
        layout="bshd",
    )


@pytest.mark.parametrize(
    "B, H, S, D",
    [
        (1, 4, 64, 64),
        (2, 8, 128, 128),
        (1, 4, 64, 128),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mxfp8_attn_fwd_nofp8(B, H, S, D, causal, dtype):
    """mxfp8_attention_forward with use_mxfp8=False matches PyTorch reference."""
    torch.manual_seed(0)
    sm_scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, S, H, D, dtype=dtype, device="cuda") * 0.1
    k = torch.randn(B, S, H, D, dtype=dtype, device="cuda") * 0.1
    v = torch.randn(B, S, H, D, dtype=dtype, device="cuda") * 0.1

    q_scale = _dummy_scales(B, H, S, D, _QUANT_BLOCK, "cuda")
    k_scale = _dummy_scales(B, H, S, D, _QUANT_BLOCK, "cuda")
    v_scale = _dummy_scales(B, H, S, D, _QUANT_BLOCK, "cuda")

    out, _lse, _ = _fwd(
        q, k, v, q_scale, k_scale, v_scale, sm_scale, causal, use_mxfp8=False
    )

    ref = _ref_attention(
        _bshd_to_bhsd(q),
        _bshd_to_bhsd(k),
        _bshd_to_bhsd(v),
        causal=causal,
        sm_scale=sm_scale,
    )
    ref = _bhsd_to_bshd(ref)

    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"Mismatch at (B={B},H={H},S={S},D={D},causal={causal})",
    )


@pytest.mark.parametrize(
    "B, HQ, HK, S, D",
    [
        (1, 4, 4, 64, 64),
        (1, 8, 2, 64, 128),  # GQA
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_mxfp8_attn_fwd_quantized(B, HQ, HK, S, D, causal):
    """use_mxfp8=True forward matches PyTorch on dequantized Q/K/V."""
    torch.manual_seed(0)
    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)

    q_hp = torch.randn(B, S, HQ, D, dtype=dtype, device="cuda") * 0.1
    k_hp = torch.randn(B, S, HK, D, dtype=dtype, device="cuda") * 0.1
    v_hp = torch.randn(B, S, HK, D, dtype=dtype, device="cuda") * 0.1

    q, q_scale = _quantize_bshd(q_hp)
    k, k_scale = _quantize_bshd(k_hp)
    v, v_scale = _quantize_bshd(v_hp)

    out, _lse, _ = _fwd(
        q, k, v, q_scale, k_scale, v_scale, sm_scale, causal, use_mxfp8=True
    )

    q_dq = _dequantize_bshd(q, q_scale, dtype)
    k_dq = _dequantize_bshd(k, k_scale, dtype)
    v_dq = _dequantize_bshd(v, v_scale, dtype)
    ref = _ref_attention(
        _bshd_to_bhsd(q_dq),
        _bshd_to_bhsd(k_dq),
        _bshd_to_bhsd(v_dq),
        causal=causal,
        sm_scale=sm_scale,
    )
    ref = _bhsd_to_bshd(ref)

    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=2e-1,
        rtol=2e-1,
        msg=f"Quantized fwd mismatch (B={B},HQ={HQ},HK={HK},S={S},D={D},causal={causal})",
    )


@pytest.mark.parametrize(
    "B, HQ, HK, S, D",
    [
        (1, 4, 4, 64, 64),
        (1, 8, 2, 64, 64),  # GQA
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("use_mxfp8", [False, True])
def test_mxfp8_attn_bwd(B, HQ, HK, S, D, causal, use_mxfp8):
    """mxfp8_attention_backward matches autograd on (de)quantized Q/K/V.

    use_mxfp8=True exercises preprocess (do MXFP8), dq, and dkdv kernels.
    use_mxfp8=False still launches those three kernels on the non-quant path.
    """
    torch.manual_seed(1)
    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)

    q_hp = torch.randn(B, S, HQ, D, dtype=dtype, device="cuda") * 0.1
    k_hp = torch.randn(B, S, HK, D, dtype=dtype, device="cuda") * 0.1
    v_hp = torch.randn(B, S, HK, D, dtype=dtype, device="cuda") * 0.1

    if use_mxfp8:
        q, q_scale = _quantize_bshd(q_hp)
        k, k_scale = _quantize_bshd(k_hp)
        v, v_scale = _quantize_bshd(v_hp)
        q_ref = _dequantize_bshd(q, q_scale, torch.float32)
        k_ref = _dequantize_bshd(k, k_scale, torch.float32)
        v_ref = _dequantize_bshd(v, v_scale, torch.float32)
        atol, rtol = 3.5e-1, 3.5e-1
    else:
        q, k, v = q_hp, k_hp, v_hp
        q_scale = _dummy_scales(B, HQ, S, D, _QUANT_BLOCK, "cuda")
        k_scale = _dummy_scales(B, HK, S, D, _QUANT_BLOCK, "cuda")
        v_scale = _dummy_scales(B, HK, S, D, _QUANT_BLOCK, "cuda")
        q_ref, k_ref, v_ref = q.float(), k.float(), v.float()
        atol, rtol = 8e-2, 8e-2

    o, lse, _ = _fwd(
        q, k, v, q_scale, k_scale, v_scale, sm_scale, causal, use_mxfp8=use_mxfp8
    )
    do = torch.randn_like(o, dtype=dtype) * 0.1

    dq, dk, dv = mxfp8_attention_backward(
        do,
        q,
        k,
        v,
        o,
        lse,
        dq=None,
        dk=None,
        dv=None,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        sm_scale=sm_scale,
        causal=causal,
        use_mxfp8=use_mxfp8,
        quant_block_size=_QUANT_BLOCK,
        layout="bshd",
    )

    q_t = _bshd_to_bhsd(q_ref).detach().requires_grad_(True)
    k_t = _bshd_to_bhsd(k_ref).detach().requires_grad_(True)
    v_t = _bshd_to_bhsd(v_ref).detach().requires_grad_(True)
    ref = _ref_attention(q_t, k_t, v_t, causal=causal, sm_scale=sm_scale)
    ref.backward(_bshd_to_bhsd(do.float()))

    dq_ref = _bhsd_to_bshd(q_t.grad)
    dk_ref = _bhsd_to_bshd(k_t.grad)
    dv_ref = _bhsd_to_bshd(v_t.grad)

    tag = f"(B={B},HQ={HQ},HK={HK},S={S},D={D},causal={causal},mxfp8={use_mxfp8})"
    torch.testing.assert_close(
        dq.float(), dq_ref.float(), atol=atol, rtol=rtol, msg=f"dq mismatch {tag}"
    )
    torch.testing.assert_close(
        dk.float(), dk_ref.float(), atol=atol, rtol=rtol, msg=f"dk mismatch {tag}"
    )
    torch.testing.assert_close(
        dv.float(), dv_ref.float(), atol=atol, rtol=rtol, msg=f"dv mismatch {tag}"
    )

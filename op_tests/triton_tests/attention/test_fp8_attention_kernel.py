# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness tests for the FP8 Flash Attention v2 kernel.

Covers:
  * non-FP8 forward (USE_FP8=False) vs PyTorch
  * FP8 forward (USE_FP8=True) with gfx942 e4m3fnuz / gfx950 e4m3fn
  * backward preprocess / dq / dkdv with and without FP8, including GQA
    and causal masks
"""

import math

import pytest
import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.utils import FP8_ARCHS
from aiter.ops.triton.attention.fp8_attention import (
    _bwd_kernel_dkdv,
    _bwd_kernel_dq,
    _bwd_preprocess_use_o,
    attn_fwd,
    get_padded_headsize,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(
    get_arch() not in FP8_ARCHS, reason=f"FP8 attention not supported on {get_arch()}"
)

BLOCK_M = 64
BLOCK_N = 64


def _fp8_dtype():
    """Arch-native FP8 dtype: e4m3fnuz on gfx942, e4m3fn on gfx950+."""
    if get_arch() == "gfx942":
        return torch.float8_e4m3fnuz
    return torch.float8_e4m3fn


def _fp8_tl_dtype(fp8_dtype):
    if fp8_dtype == torch.float8_e4m3fnuz:
        return getattr(tl, "float8e4b8", tl.float8e4nv)
    return tl.float8e4nv


def _ref_attention(q, k, v, causal=False, sm_scale=None):
    """Pure-PyTorch SDPA. q: [B, HQ, S, D], k/v: [B, HK, S, D]."""
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


def _pad_head(t, d_pad):
    if t.shape[-1] == d_pad:
        return t.contiguous()
    pad = torch.zeros(*t.shape[:-1], d_pad, dtype=t.dtype, device=t.device)
    pad[..., : t.shape[-1]] = t
    return pad


def _quantize_per_block(x, block, fp8_dtype):
    """Per-(B, H, block) amax quantize along sequence. Returns fp8 x and descale (B, H, n_blocks)."""
    b, h, s, d = x.shape
    n_blocks = triton.cdiv(s, block)
    pad_s = n_blocks * block
    if pad_s != s:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_s - s))
    blocks = x.float().reshape(b, h, n_blocks, block, d)
    fp8_max = torch.finfo(fp8_dtype).max
    amax = blocks.abs().amax(dim=(3, 4)).clamp(min=1e-7)
    descale = amax / fp8_max
    q = (blocks / descale[..., None, None]).to(fp8_dtype)
    q = q.reshape(b, h, pad_s, d)[:, :, :s].contiguous()
    return q, descale.contiguous()


def _quantize_v_tensor(v, fp8_dtype):
    """Per-tensor FP8 quantize. Returns fp8 v and scale (fp8_max/amax)."""
    fp8_max = torch.finfo(fp8_dtype).max
    amax = v.float().abs().amax().clamp(min=1e-7)
    scale = fp8_max / amax
    v_fp8 = (v.float() * scale).to(fp8_dtype)
    return v_fp8.contiguous(), scale.to(dtype=torch.float32).reshape(1)


def _dequant_per_block(x_fp8, descale, block):
    b, h, s, d = x_fp8.shape
    n_blocks = descale.shape[-1]
    pad_s = n_blocks * block
    x = x_fp8.float()
    if pad_s != s:
        x = torch.nn.functional.pad(x, (0, 0, 0, pad_s - s))
    blocks = x.reshape(b, h, n_blocks, block, d)
    dq = blocks * descale[..., None, None]
    return dq.reshape(b, h, pad_s, d)[:, :, :s]


def _attn_fwd(
    q,
    k,
    v,
    causal=False,
    sm_scale=None,
    use_fp8=False,
    q_descale=None,
    k_descale=None,
    v_scale=None,
):
    """Launch attn_fwd. q/k/v are bhsd. Returns (out[..., :D], lse)."""
    b, hq, s_q, d = q.shape
    _, hk, s_k, _ = k.shape
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(d)
    d_pad = get_padded_headsize(d)
    q_p, k_p, v_p = _pad_head(q, d_pad), _pad_head(k, d_pad), _pad_head(v, d_pad)
    out = torch.zeros(
        b,
        hq,
        s_q,
        d_pad,
        dtype=torch.bfloat16 if use_fp8 else q.dtype,
        device=q.device,
    )
    lse = torch.zeros(b, hq, 2 * s_q, dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(s_q, BLOCK_M), hq, b)

    if use_fp8:
        n_k_blocks = k_descale.shape[-1]
        stride_qdz, stride_qdh, stride_qdm = q_descale.stride()
        stride_kdz, _, stride_kdm = k_descale.stride()
        padded_k = n_k_blocks
    else:
        n_k_blocks = 1
        stride_qdz = stride_qdh = stride_qdm = 0
        stride_kdz = stride_kdm = 0
        padded_k = 1

    attn_fwd[grid](
        Q=q_p,
        K=k_p,
        V=v_p,
        bias=None,
        p_scale=1.0,
        q_descale_ptr=q_descale,
        k_descale_ptr=k_descale,
        v_scale_ptr=v_scale,
        USE_FP8=use_fp8,
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
        stride_qdescale_z=stride_qdz,
        stride_qdescale_h=stride_qdh,
        stride_qdescale_m=stride_qdm,
        stride_kdescale_z=stride_kdz,
        stride_kdescale_h=n_k_blocks if use_fp8 else 0,
        stride_kdescale_m=stride_kdm,
        padded_kscale_block_num=padded_k,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        dropout_p=0.0,
        philox_seed=0,
        philox_offset_base=0,
        scores=None,
        scores_scaled_shifted=None,
        exp_scores=None,
        alibi_slopes=None,
        HQ=hq,
        HK=hk,
        ACTUAL_BLOCK_DMODEL_QK=d_pad,
        ACTUAL_BLOCK_DMODEL_V=d_pad,
        MAX_SEQLENS_Q=s_q,
        MAX_SEQLENS_K=s_k,
        VARLEN=False,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL_QK=d_pad,
        BLOCK_DMODEL_V=d_pad,
        BLOCK_N=BLOCK_N,
        USE_BIAS=False,
        ENABLE_DROPOUT=False,
        RETURN_SCORES=False,
        USE_ALIBI=False,
        USE_EXP2=True,
    )
    return out[..., :d], lse


def _attn_bwd(
    do,
    q,
    k,
    v,
    o,
    lse,
    sm_scale,
    causal,
    use_fp8=False,
    q_descale=None,
    k_descale=None,
    v_scale=None,
):
    """Launch preprocess + dq + dkdv. Returns (dq, dk, dv) in bf16."""
    b, hq, s_q, d = q.shape
    _, hk, s_k, _ = k.shape
    d_pad = get_padded_headsize(d)
    q_p, k_p, v_p = _pad_head(q, d_pad), _pad_head(k, d_pad), _pad_head(v, d_pad)
    o_p = _pad_head(o, d_pad)
    do_p = _pad_head(do, d_pad)
    dq = torch.zeros_like(q_p, dtype=torch.bfloat16)
    dk = torch.zeros_like(k_p, dtype=torch.bfloat16)
    dv = torch.zeros_like(v_p, dtype=torch.bfloat16)

    fp8_dtype = _fp8_dtype()
    fp8_max = float(torch.finfo(fp8_dtype).max)
    n_q_blocks = triton.cdiv(s_q, BLOCK_M)
    n_k_blocks = triton.cdiv(s_k, BLOCK_N)

    if use_fp8:
        do_fp8 = torch.empty_like(do_p, dtype=fp8_dtype)
        do_scale = torch.empty(b, hq, n_q_blocks, dtype=torch.float32, device=q.device)
        # Backward fuses sm_scale into q_descale (see _attn_bwd_dkdv / _attn_bwd_dq).
        q_descale_bwd = (q_descale * sm_scale).contiguous()
        stride_qsz, stride_qsh, stride_qsm = q_descale_bwd.stride()
        stride_ksz, _, stride_ksm = k_descale.stride()
        stride_dosz, stride_dosh, stride_dosm = do_scale.stride()
        padded_q = n_q_blocks
        padded_k = n_k_blocks
        padded_do = n_q_blocks
    else:
        do_fp8 = torch.empty(0, device=q.device)
        do_scale = torch.empty(0, device=q.device)
        q_descale_bwd = None
        stride_qsz = stride_qsh = stride_qsm = 0
        stride_ksz = stride_ksm = 0
        stride_dosz = stride_dosh = stride_dosm = 0
        padded_q = padded_k = padded_do = 1

    grid_pre = (n_q_blocks, b * hq)
    _bwd_preprocess_use_o[grid_pre](
        o_p,
        do_p,
        do_fp8,
        do_scale,
        lse,
        use_fp8,
        o_p.stride(0),
        o_p.stride(1),
        o_p.stride(2),
        o_p.stride(3),
        do_p.stride(0),
        do_p.stride(1),
        do_p.stride(2),
        do_p.stride(3),
        lse.stride(0),
        lse.stride(1),
        lse.stride(2),
        stride_dosz,
        stride_dosh,
        None,
        None,
        s_q,
        s_k,
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL_V=d_pad,
        ACTUAL_BLOCK_DMODEL_V=d_pad,
        N_CTX_Q=s_q,
        Z=b,
        HQ=hq,
        IS_VARLEN=False,
        F8_BWD_DTYPE=_fp8_tl_dtype(fp8_dtype),
        F8_BWD_MAX=fp8_max,
    )

    do_bwd = do_fp8 if use_fp8 else do_p
    log_p_scale = 0.0
    common = dict(  # noqa: C408
        Q=q_p,
        K=k_p,
        V=v_p,
        sm_scale=sm_scale,
        q_scale_ptr=q_descale_bwd,
        k_descale_ptr=k_descale,
        v_scale_ptr=v_scale,
        p_scale=1.0,
        do_descale_ptr=do_scale,
        Out=o_p,
        DO=do_bwd,
        DQ=dq,
        DK=dk,
        DV=dv,
        LD=lse,
        stride_dq_all=dq.numel(),
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
        stride_vn=v_p.stride(2),
        stride_vk=v_p.stride(3),
        stride_doz=do_bwd.stride(0) if do_bwd.ndim == 4 else 0,
        stride_doh=do_bwd.stride(1) if do_bwd.ndim == 4 else 0,
        stride_dom=do_bwd.stride(2) if do_bwd.ndim == 4 else 0,
        stride_dok=do_bwd.stride(3) if do_bwd.ndim == 4 else 0,
        stride_ldz=lse.stride(0),
        stride_ldh=lse.stride(1),
        stride_ldm=lse.stride(2),
        stride_doscalez=stride_dosz,
        stride_doscaleh=stride_dosh,
        stride_doscalem=stride_dosm,
        stride_qscalez=stride_qsz,
        stride_qscaleh=stride_qsh,
        stride_qscalem=stride_qsm,
        stride_kscalez=stride_ksz,
        stride_kscaleh=n_k_blocks if use_fp8 else 0,
        stride_kscalem=stride_ksm,
        padded_doscale_block_num=padded_do,
        padded_qscale_block_num=padded_q,
        padded_kscale_block_num=padded_k,
        Z=b,
        HQ=hq,
        HK=hk,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q=s_q,
        max_seqlen_k=s_k,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_DMODEL_QK=d_pad,
        BLOCK_DMODEL_V=d_pad,
        ACTUAL_BLOCK_DMODEL_QK=d_pad,
        ACTUAL_BLOCK_DMODEL_V=d_pad,
        SEQUENCE_PARALLEL=False,
        CAUSAL=causal,
        USE_EXP2=True,
        IS_VARLEN=False,
        USE_FP8=use_fp8,
        log_p_scale=log_p_scale,
        F8_FWD_MAX=fp8_max,
    )

    grid_dq = (b * hq, n_q_blocks)
    _bwd_kernel_dq[grid_dq](**common, num_block_m=n_q_blocks)

    grid_dkdv = (b * hk, n_k_blocks)
    _bwd_kernel_dkdv[grid_dkdv](**common, num_block_m=n_q_blocks)

    return dq[..., :d], dk[..., :d], dv[..., :d]


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

    ref = _ref_attention(q, k, v, causal=causal)
    out, _lse = _attn_fwd(q, k, v, causal=causal, use_fp8=False)

    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=1e-2,
        rtol=1e-2,
        msg=f"Mismatch at (B={B},HQ={HQ},HK={HK},S_q={S_q},S_k={S_k},D={D},causal={causal})",
    )


@pytest.mark.parametrize(
    "B, HQ, HK, S, D",
    [
        (1, 4, 4, 64, 64),
        (1, 8, 2, 64, 128),  # GQA
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_fp8_attn_fwd_fp8(B, HQ, HK, S, D, causal):
    """attn_fwd with USE_FP8=True matches PyTorch on dequantized Q/K/V.

    Uses the arch-native FP8 dtype (e4m3fnuz on gfx942, e4m3fn on gfx950).
    """
    torch.manual_seed(0)
    fp8_dtype = _fp8_dtype()
    sm_scale = 1.0 / math.sqrt(D)
    q_hp = torch.randn(B, HQ, S, D, dtype=torch.bfloat16, device="cuda") * 0.1
    k_hp = torch.randn(B, HK, S, D, dtype=torch.bfloat16, device="cuda") * 0.1
    v_hp = torch.randn(B, HK, S, D, dtype=torch.bfloat16, device="cuda") * 0.1

    q, q_descale = _quantize_per_block(q_hp, BLOCK_M, fp8_dtype)
    k, k_descale = _quantize_per_block(k_hp, BLOCK_N, fp8_dtype)
    v, v_scale = _quantize_v_tensor(v_hp, fp8_dtype)

    out, _lse = _attn_fwd(
        q,
        k,
        v,
        causal=causal,
        sm_scale=sm_scale,
        use_fp8=True,
        q_descale=q_descale,
        k_descale=k_descale,
        v_scale=v_scale,
    )

    q_dq = _dequant_per_block(q, q_descale, BLOCK_M)
    k_dq = _dequant_per_block(k, k_descale, BLOCK_N)
    v_dq = v.float() / v_scale
    ref = _ref_attention(q_dq, k_dq, v_dq, causal=causal, sm_scale=sm_scale)

    torch.testing.assert_close(
        out.float(),
        ref.float(),
        atol=2e-1,
        rtol=2e-1,
        msg=f"FP8 fwd mismatch (B={B},HQ={HQ},HK={HK},S={S},D={D},causal={causal},dtype={fp8_dtype})",
    )


@pytest.mark.parametrize(
    "B, HQ, HK, S, D",
    [
        (1, 4, 4, 64, 64),
        (1, 8, 2, 64, 64),  # GQA
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("use_fp8", [False, True])
def test_fp8_attn_bwd(B, HQ, HK, S, D, causal, use_fp8):
    """Backward preprocess / dq / dkdv match autograd.

    use_fp8=True exercises the arch-native FP8 dtype path.
    use_fp8=False still launches all three backward kernels.
    """
    torch.manual_seed(1)
    dtype = torch.bfloat16
    sm_scale = 1.0 / math.sqrt(D)
    fp8_dtype = _fp8_dtype()

    q_hp = torch.randn(B, HQ, S, D, dtype=dtype, device="cuda") * 0.1
    k_hp = torch.randn(B, HK, S, D, dtype=dtype, device="cuda") * 0.1
    v_hp = torch.randn(B, HK, S, D, dtype=dtype, device="cuda") * 0.1

    if use_fp8:
        q, q_descale = _quantize_per_block(q_hp, BLOCK_M, fp8_dtype)
        k, k_descale = _quantize_per_block(k_hp, BLOCK_N, fp8_dtype)
        v, v_scale = _quantize_v_tensor(v_hp, fp8_dtype)
        q_ref = _dequant_per_block(q, q_descale, BLOCK_M)
        k_ref = _dequant_per_block(k, k_descale, BLOCK_N)
        v_ref = v.float() / v_scale
        atol, rtol = 3.5e-1, 3.5e-1
    else:
        q, k, v = q_hp, k_hp, v_hp
        q_descale = k_descale = v_scale = None
        q_ref, k_ref, v_ref = q.float(), k.float(), v.float()
        atol, rtol = 8e-2, 8e-2

    o, lse = _attn_fwd(
        q,
        k,
        v,
        causal=causal,
        sm_scale=sm_scale,
        use_fp8=use_fp8,
        q_descale=q_descale,
        k_descale=k_descale,
        v_scale=v_scale,
    )
    do = torch.randn_like(o) * 0.1

    dq, dk, dv = _attn_bwd(
        do,
        q,
        k,
        v,
        o,
        lse,
        sm_scale,
        causal,
        use_fp8=use_fp8,
        q_descale=q_descale,
        k_descale=k_descale,
        v_scale=v_scale,
    )

    q_t = q_ref.detach().requires_grad_(True)
    k_t = k_ref.detach().requires_grad_(True)
    v_t = v_ref.detach().requires_grad_(True)
    ref = _ref_attention(q_t, k_t, v_t, causal=causal, sm_scale=sm_scale)
    ref.backward(do.float())

    tag = (
        f"(B={B},HQ={HQ},HK={HK},S={S},D={D},causal={causal},fp8={use_fp8},{fp8_dtype})"
    )
    torch.testing.assert_close(
        dq.float(), q_t.grad.float(), atol=atol, rtol=rtol, msg=f"dq mismatch {tag}"
    )
    torch.testing.assert_close(
        dk.float(), k_t.grad.float(), atol=atol, rtol=rtol, msg=f"dk mismatch {tag}"
    )
    torch.testing.assert_close(
        dv.float(), v_t.grad.float(), atol=atol, rtol=rtol, msg=f"dv mismatch {tag}"
    )

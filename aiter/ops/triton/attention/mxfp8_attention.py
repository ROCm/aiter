# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MXFP8 Flash Attention v2 - high-level Python wrappers around the Triton
# kernels for forward and backward passes.

import math

import torch
import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.attention.mxfp8_attention_kernel import (
    _bwd_kernel_dkdv_mxfp8,
    _bwd_kernel_dq_mxfp8,
    _bwd_preprocess_use_o_mxfp8,
    attn_fwd_mxfp8,
    get_padded_head_dim,
    get_shape_from_layout,
    get_strides_from_layout,
    philox_offset,
    philox_seed,
)
from aiter.ops.triton.utils._triton.arch_info import is_cdna4
from aiter.ops.triton.utils.logger import AiterTritonLogger

__all__ = [
    "_bwd_kernel_dkdv_mxfp8",
    "_bwd_kernel_dq_mxfp8",
    "_bwd_preprocess_use_o_mxfp8",
    "attn_fwd_mxfp8",
    "get_padded_head_dim",
    "get_shape_from_layout",
    "get_strides_from_layout",
    "is_cdna4",
    "mxfp8_attention_backward",
    "mxfp8_attention_forward",
]

_LOGGER = AiterTritonLogger()


def _get_f8_bwd_dtype():
    return torch.float8_e4m3fn


def _get_tl_f8_bwd_dtype():
    return tl.float8e4nv


def mxfp8_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    p_scale: int = 127,
    sm_scale: float = 1.0,
    alibi_slopes: torch.Tensor | None = None,
    causal: bool = False,
    bias: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    return_softmax: bool = False,
    use_mxfp8: bool = True,
    block_m: int = 64,
    block_n: int = 64,
    quant_block_size: int = 32,
    layout: str = "bshd",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass of MXFP8 Flash Attention v2.

    Args:
        q, k, v: Query, Key, Value tensors in FP8 format.
        q_scale, k_scale, v_scale: Scale tensors for Q, K, V (uint8, e8m0).
        p_scale: Scale for attention probability quantization (default 127 = no quantization).
        sm_scale: Softmax scale factor (typically 1/sqrt(d)).
        alibi_slopes: Optional ALiBi slopes tensor.
        causal: Whether to apply causal masking.
        bias: Optional attention bias tensor.
        dropout_p: Dropout probability.
        return_softmax: Whether to return softmax scores.
        use_mxfp8: Whether to use MXFP8 quantization.
        block_m: Block size for query sequence dimension.
        block_n: Block size for key/value sequence dimension.
        quant_block_size: Quantization block size.
        layout: Tensor layout ("bshd", "bhsd", or "thd").

    Returns:
        Tuple of (output, softmax_lse, exp_scores).
    """
    _LOGGER.info(f"MXFP8_ATTENTION_FWD: q={tuple(q.shape)}, k={tuple(k.shape)}")
    assert is_cdna4(), "mxfp8 attention requires gfx950 or newer"
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous()
    assert q_scale.is_contiguous() and k_scale.is_contiguous()

    cu_seqlens_q = 0
    cu_seqlens_k = 0
    max_seqlens_q = q.shape[1] if layout == "bshd" else q.shape[2]
    max_seqlens_k = k.shape[1] if layout == "bshd" else k.shape[2]
    use_exp2 = True
    quant_size = 32

    o_shape = list(q.shape)
    o_shape[-1] = v.shape[-1]
    o = torch.empty(
        o_shape,
        device=q.device,
        dtype=torch.bfloat16 if use_mxfp8 else q.dtype,
        requires_grad=True,
    )

    is_varlen = layout == "thd"
    if bias is not None:
        assert bias.numel() < 2**31

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, _seqlen_q, _seqlen_k = (
        get_shape_from_layout(
            q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlens_q, max_seqlens_k
        )
    )

    assert quant_block_size % quant_size == 0
    assert block_m % quant_block_size == 0
    assert block_n % quant_block_size == 0

    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)

    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)
    assert padded_d_model_qk % quant_block_size == 0
    assert padded_d_model_v % quant_block_size == 0

    grid = (triton.cdiv(max_seqlens_q, block_m), nheads_q, batch)

    if return_softmax:
        scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32,
        )
        scores_scaled_shifted = torch.zeros_like(scores)
        scores_strides = tuple(scores.stride())
    else:
        scores = torch.empty([], device=q.device, dtype=torch.float32)
        scores_scaled_shifted = None
        scores_strides = (0, 0, 0, 0)

    if return_softmax:
        exp_scores = torch.zeros(
            (batch, nheads_q, max_seqlens_q, max_seqlens_k),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        exp_scores = torch.empty([], device=q.device, dtype=torch.float32)

    if is_varlen:
        softmax_lse = torch.empty(
            (q.shape[0], nheads_q), device=q.device, dtype=torch.float32
        )
        stride_lse_m, stride_lse_h = softmax_lse.stride()
        stride_lse_z = 0
    else:
        softmax_lse = torch.empty(
            (batch, nheads_q, max_seqlens_q), device=q.device, dtype=torch.float32
        )
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

    bias_strides = tuple(bias.stride()) if bias is not None else (0, 0, 0, 0)
    alibi_strides = tuple(alibi_slopes.stride()) if alibi_slopes is not None else (0, 0)

    if use_mxfp8:
        stride_qdescale_z, stride_qdescale_h, stride_qdescale_m, stride_qdescale_d = (
            get_strides_from_layout(q_scale, layout)
        )
        stride_kdescale_z, stride_kdescale_h, stride_kdescale_m, stride_kdescale_d = (
            get_strides_from_layout(k_scale, layout)
        )
        stride_vdescale_z, stride_vdescale_h, stride_vdescale_m, stride_vdescale_d = (
            get_strides_from_layout(v_scale, layout)
        )
    else:
        stride_qdescale_z = stride_qdescale_h = stride_qdescale_m = (
            stride_qdescale_d
        ) = None
        stride_kdescale_z = stride_kdescale_h = stride_kdescale_m = (
            stride_kdescale_d
        ) = None
        stride_vdescale_z = stride_vdescale_h = stride_vdescale_m = (
            stride_vdescale_d
        ) = None

    kernel_kwargs = {}
    if padded_d_model_qk % 128 == 0 and block_n % 128 == 0:
        kernel_kwargs["matrix_instr_nonkdim"] = 16

    attn_fwd_mxfp8[grid](
        q,
        k,
        v,
        bias,
        p_scale,
        q_scale,
        k_scale,
        v_scale,
        use_mxfp8,
        sm_scale,
        softmax_lse,
        o,
        *q_strides,
        *k_strides,
        *v_strides,
        *o_strides,
        *bias_strides,
        *alibi_strides,
        *scores_strides,
        stride_lse_z,
        stride_lse_h,
        stride_lse_m,
        stride_qdescale_z,
        stride_qdescale_h,
        stride_qdescale_m,
        stride_qdescale_d,
        stride_kdescale_z,
        stride_kdescale_h,
        stride_kdescale_m,
        stride_kdescale_d,
        stride_vdescale_z,
        stride_vdescale_h,
        stride_vdescale_m,
        stride_vdescale_d,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p=dropout_p,
        philox_seed=philox_seed,
        philox_offset_base=philox_offset,
        scores=scores,
        scores_scaled_shifted=scores_scaled_shifted,
        exp_scores=exp_scores,
        alibi_slopes=alibi_slopes,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=max_seqlens_q,
        MAX_SEQLENS_K=max_seqlens_k,
        IS_CAUSAL=causal,
        VARLEN=is_varlen,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        USE_BIAS=bias is not None,
        USE_ALIBI=alibi_slopes is not None,
        ENABLE_DROPOUT=dropout_p > 0.0,
        USE_EXP2=use_exp2,
        RETURN_SCORES=return_softmax,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        QUANT_BLOCK_SIZE=quant_block_size,
        QUANT_SIZE=quant_size,
        **kernel_kwargs,
    )

    return o, softmax_lse, exp_scores


def mxfp8_attention_backward(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: torch.Tensor | None,
    dk: torch.Tensor | None,
    dv: torch.Tensor | None,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    sm_scale: float,
    p_scale: int = 127,
    alibi_slopes: torch.Tensor | None = None,
    causal: bool = False,
    use_mxfp8: bool = True,
    block_m_dq_bwd: int = 64,
    block_n_dq_bwd: int = 64,
    block_m_dkv_bwd: int = 64,
    block_n_dkv_bwd: int = 64,
    quant_block_size: int = 32,
    layout: str = "bshd",
    cu_seqlens_q: int | None = None,
    cu_seqlens_k: int | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Backward pass of MXFP8 Flash Attention v2.
    """
    _LOGGER.info(f"MXFP8_ATTENTION_BWD: q={tuple(q.shape)}, k={tuple(k.shape)}")
    assert is_cdna4(), "mxfp8 attention requires gfx950 or newer"

    use_exp2 = True
    quant_size = 32
    do = do.contiguous()

    if cu_seqlens_q is None:
        cu_seqlens_q = 0
    if cu_seqlens_k is None:
        cu_seqlens_k = 0
    if max_seqlen_q is None:
        max_seqlen_q = q.shape[1] if layout == "bshd" else q.shape[2]
    if max_seqlen_k is None:
        max_seqlen_k = k.shape[1] if layout == "bshd" else k.shape[2]

    batch, nheads_q, nheads_k, head_size_qk, head_size_v, max_seqlen_q, max_seqlen_k = (
        get_shape_from_layout(
            q, k, v, layout, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
        )
    )

    q_strides = get_strides_from_layout(q, layout)
    k_strides = get_strides_from_layout(k, layout)
    v_strides = get_strides_from_layout(v, layout)
    o_strides = get_strides_from_layout(o, layout)
    do_strides = get_strides_from_layout(do, layout)
    stride_qz, stride_qh, stride_qm, stride_qk = q_strides
    stride_kz, stride_kh, stride_kn, stride_kk = k_strides
    stride_vz, stride_vh, stride_vn, stride_vk = v_strides
    stride_oz, stride_oh, stride_om, stride_ok = o_strides
    stride_doz, stride_doh, stride_dom, stride_dok = do_strides
    batch_headsize_q = batch * nheads_q
    batch_headsize_k = batch * nheads_k
    is_varlen = layout == "thd"

    padded_d_model_qk = get_padded_head_dim(head_size_qk)
    padded_d_model_v = get_padded_head_dim(head_size_v)

    copy_back = {"dq": False, "dk": False, "dv": False}
    bwd_dtype = torch.bfloat16

    if dq is None:
        dq = torch.zeros_like(q, dtype=bwd_dtype)
    else:
        dq_og = dq
        if not dq.is_contiguous():
            dq = dq.contiguous()
            copy_back["dq"] = True
        dq.zero_()

    if dk is None or dv is None:
        dk = torch.zeros_like(k, dtype=bwd_dtype)
        dv = torch.zeros_like(v, dtype=bwd_dtype)
    else:
        if not dk.is_contiguous():
            dk_og = dk
            dk = dk.contiguous()
            copy_back["dk"] = True
        if not dv.is_contiguous():
            dv_og = dv
            dv = dv.contiguous()
            copy_back["dv"] = True

    delta = torch.empty_like(softmax_lse)
    if is_varlen:
        stride_lse_delta_m, stride_lse_delta_h = softmax_lse.stride()
        stride_lse_delta_z = 0
    else:
        stride_lse_delta_z, stride_lse_delta_h, stride_lse_delta_m = (
            softmax_lse.stride()
        )

    f8_bwd_dtype = _get_f8_bwd_dtype()
    tl_f8_bwd_dtype = _get_tl_f8_bwd_dtype()

    if use_mxfp8:
        m_blocks_q = triton.cdiv(max_seqlen_q, quant_block_size)
        dv_blocks = triton.cdiv(head_size_v, quant_block_size)
        if layout == "bhsd":
            _shape = (batch, nheads_q, m_blocks_q, dv_blocks)
        elif layout == "bshd":
            _shape = (batch, m_blocks_q, nheads_q, dv_blocks)
        elif layout == "thd":
            _shape = (q_scale.shape[0], q_scale.shape[1], dv_blocks)
        else:
            raise AssertionError(f"Unsupported layout: {layout}")
        do_fp8 = torch.empty_like(do, dtype=f8_bwd_dtype)
        do_scale = torch.empty(_shape, dtype=torch.uint8, device=q.device)
        stride_dodescalez, stride_dodescaleh, stride_dodescalem, stride_dodescaled = (
            get_strides_from_layout(do_scale, layout)
        )
        stride_qdescalez, stride_qdescaleh, stride_qdescalem, stride_qdescaled = (
            get_strides_from_layout(q_scale, layout)
        )
        stride_kdescalez, stride_kdescaleh, stride_kdescalem, stride_kdescaled = (
            get_strides_from_layout(k_scale, layout)
        )
        stride_vdescalez, stride_vdescaleh, stride_vdescalem, stride_vdescaled = (
            get_strides_from_layout(v_scale, layout)
        )
    else:
        do_fp8 = None
        do_scale = None
        stride_dodescalez = stride_dodescaleh = stride_dodescalem = (
            stride_dodescaled
        ) = None
        stride_qdescalez = stride_qdescaleh = stride_qdescalem = stride_qdescaled = None
        stride_kdescalez = stride_kdescaleh = stride_kdescalem = stride_kdescaled = None
        stride_vdescalez = stride_vdescaleh = stride_vdescalem = stride_vdescaled = None

    preprocess_o_block = min(max_seqlen_q, 64)
    preprocess_o_block = max(preprocess_o_block, quant_block_size)
    grid_prebwd = (triton.cdiv(max_seqlen_q, preprocess_o_block), batch_headsize_q)
    _bwd_preprocess_use_o_mxfp8[grid_prebwd](
        o,
        do,
        do_fp8,
        do_scale,
        delta,
        use_mxfp8,
        stride_oz,
        stride_oh,
        stride_om,
        stride_ok,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_dodescalez,
        stride_dodescaleh,
        stride_dodescalem,
        stride_dodescaled,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        BLOCK_M=preprocess_o_block,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        N_CTX_Q=max_seqlen_q,
        Z=batch,
        HQ=nheads_q,
        IS_VARLEN=is_varlen,
        F8_BWD_DTYPE=tl_f8_bwd_dtype,
        QUANT_BLOCK_SIZE=quant_block_size,
    )

    p_scale_t = math.pow(2.0, int(p_scale - 127))
    log_p_scale = math.log(p_scale_t)
    num_block_m = triton.cdiv(max_seqlen_q, block_m_dq_bwd)

    kernel_kwargs = {}
    if (
        block_n_dq_bwd % 128 == 0
        and padded_d_model_qk % 128 == 0
        and padded_d_model_v % 128 == 0
    ):
        kernel_kwargs["matrix_instr_nonkdim"] = 16

    grid_bwd = (batch_headsize_q, num_block_m)
    _bwd_kernel_dq_mxfp8[grid_bwd](
        q,
        k,
        v,
        sm_scale,
        p_scale,
        log_p_scale,
        q_scale,
        k_scale,
        v_scale,
        do_scale,
        o,
        do_fp8 if use_mxfp8 else do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_dodescalez,
        stride_dodescaleh,
        stride_dodescalem,
        stride_dodescaled,
        stride_qdescalez,
        stride_qdescaleh,
        stride_qdescalem,
        stride_qdescaled,
        stride_kdescalez,
        stride_kdescaleh,
        stride_kdescalem,
        stride_kdescaled,
        stride_vdescalez,
        stride_vdescaleh,
        stride_vdescalem,
        stride_vdescaled,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=block_m_dq_bwd,
        BLOCK_N=block_n_dq_bwd,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        use_mxfp8=use_mxfp8,
        F8_BWD_DTYPE=tl_f8_bwd_dtype,
        QUANT_BLOCK_SIZE=quant_block_size,
        QUANT_SIZE=quant_size,
        **kernel_kwargs,
    )

    if (
        block_m_dkv_bwd % 128 == 0
        and padded_d_model_qk % 128 == 0
        and padded_d_model_v % 128 == 0
    ):
        kernel_kwargs["matrix_instr_nonkdim"] = 16
    else:
        kernel_kwargs = {}

    grid_bwd_dkdv = (batch_headsize_k, triton.cdiv(max_seqlen_k, block_n_dkv_bwd))
    _bwd_kernel_dkdv_mxfp8[grid_bwd_dkdv](
        q,
        k,
        v,
        p_scale,
        log_p_scale,
        sm_scale,
        q_scale,
        k_scale,
        v_scale,
        do_scale,
        o,
        do_fp8 if use_mxfp8 else do,
        dq,
        dk,
        dv,
        softmax_lse,
        delta,
        stride_qz,
        stride_qh,
        stride_qm,
        stride_qk,
        stride_kz,
        stride_kh,
        stride_kn,
        stride_kk,
        stride_vz,
        stride_vh,
        stride_vn,
        stride_vk,
        stride_doz,
        stride_doh,
        stride_dom,
        stride_dok,
        stride_lse_delta_z,
        stride_lse_delta_h,
        stride_lse_delta_m,
        stride_dodescalez,
        stride_dodescaleh,
        stride_dodescalem,
        stride_dodescaled,
        stride_qdescalez,
        stride_qdescaleh,
        stride_qdescalem,
        stride_qdescaled,
        stride_kdescalez,
        stride_kdescaleh,
        stride_kdescalem,
        stride_kdescaled,
        stride_vdescalez,
        stride_vdescaleh,
        stride_vdescalem,
        stride_vdescaled,
        batch,
        nheads_q,
        nheads_k,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        num_block_m=num_block_m,
        BLOCK_M=block_m_dkv_bwd,
        BLOCK_N=block_n_dkv_bwd,
        BLOCK_DMODEL_QK=padded_d_model_qk,
        BLOCK_DMODEL_V=padded_d_model_v,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        CAUSAL=causal,
        USE_EXP2=use_exp2,
        IS_VARLEN=is_varlen,
        use_mxfp8=use_mxfp8,
        F8_BWD_DTYPE=tl_f8_bwd_dtype,
        QUANT_BLOCK_SIZE=quant_block_size,
        QUANT_SIZE=quant_size,
        **kernel_kwargs,
    )

    if copy_back["dq"]:
        dq_og.copy_(dq)
        dq = dq_og
    if copy_back["dk"]:
        dk_og.copy_(dk)
        dk = dk_og
    if copy_back["dv"]:
        dv_og.copy_(dv)
        dv = dv_og

    return dq, dk, dv

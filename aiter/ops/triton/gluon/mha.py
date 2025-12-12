# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl

import aiter.ops.triton.utils.types as types
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.gluon._gluon_kernels.mha import _attn_fwd, _get_config
from aiter.ops.triton.mha import (
    _cast_to_fp8,
    _cast_varlen_to_fp8,
    _USE_FUSED_BWD_KERNEL,
    _USE_INT64_STRIDES,
)

_LOGGER = AiterTritonLogger()


def _get_mfma_instr_shape(dtype: torch.dtype) -> Tuple[int, int, int]:
    # V_MFMA_F32_32x32x16_F16
    if dtype == torch.float16:
        return (32, 32, 16)
    # V_MFMA_F32_16x16x32_BF16
    elif dtype == torch.bfloat16:
        return (16, 16, 32)
    # V_MFMA_F32_16x16x4_F32
    elif dtype == torch.float32:
        return (16, 16, 4)
    else:
        raise ValueError(f"Unsupported data type for MFMA: {dtype}")


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
    max_seqlen_q: int,
    max_seqlen_k: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    descale_q: Optional[torch.Tensor] = None,
    descale_k: Optional[torch.Tensor] = None,
    descale_v: Optional[torch.Tensor] = None,
    config: Optional[dict[str, any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if bias is not None:
        raise ValueError("Bias is not supported yet in the Triton Backend")
    if window_size_left != -1 or window_size_right != -1:
        raise ValueError("Sliding Window is not supported yet in the Triton Backend")

    # FP8
    IS_FP8 = types._is_fp8(q)
    FP8_MAX: gl.constexpr = torch.finfo(q.dtype).max
    is_varlen = True if cu_seqlens_q is not None else False

    # FP8 input types ==> float32 output type
    if IS_FP8:
        o = torch.zeros(
            (q.shape[:-1] + v.shape[-1:]), dtype=torch.float32, device=q.device
        )
    else:
        o = torch.zeros((q.shape[:-1] + v.shape[-1:]), dtype=q.dtype, device=q.device)
    if is_varlen:
        # Layout is thd.
        # q and k are [total_tokens, num_head, head_dim_qk].
        # v is [total_tokens, num_head, head_dim_v].
        batch, seqlen_q, num_q_heads = (
            len(cu_seqlens_q) - 1,
            max_seqlen_q,
            q.shape[1],
        )
        num_k_heads = k.shape[1]

        assert (
            num_q_heads <= num_k_heads
        ), "Grouped query and multi-query attention not supported yet in Gluon"
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    else:
        # Layout is bshd.
        # q and k are [batch, seq_len, num_head, head_dim_qk].
        # v is [batch, seq_len, num_head, head_dim_v].
        raise NotImplementedError(
            "Flash Attention with bshd/bhsd layout is not implemented yet for Gluon"
        )

    qk_head_dim = q.shape[-1]
    v_head_dim = v.shape[-1]
    pe_head_dim = qk_head_dim - v_head_dim
    # padding for head_dim. Power of 2 or 16
    BLOCK_DMODEL_POW2 = max(triton.next_power_of_2(v_head_dim), 16)
    BLOCK_DMODEL_PE_POW2 = (
        0 if pe_head_dim == 0 else max(triton.next_power_of_2(pe_head_dim), 16)
    )
    assert (pe_head_dim == 0 and BLOCK_DMODEL_PE_POW2 == 0) or (
        v_head_dim == BLOCK_DMODEL_POW2 and pe_head_dim == BLOCK_DMODEL_PE_POW2
    ), "Positional encoding support requires NOPE and PE head sizes to be unpadded powers of 2."
    assert (not IS_FP8) or (
        IS_FP8 and pe_head_dim == 0
    ), "Positional encoding doesn't support FP8."

    if is_varlen:
        # softmax_lse [total_tokens, num_q_heads]
        softmax_lse = torch.zeros(
            (q.shape[0], num_q_heads), device=q.device, dtype=torch.float32
        )
        stride_lse_z, stride_lse_h, stride_lse_m = (
            0,
            softmax_lse.stride(1),
            softmax_lse.stride(0),
        )
    else:
        raise NotImplementedError(
            "Flash Attention with bshd/bhsd layout is not implemented yet for Gluon"
        )

    # exp_scores [batch, num_q_heads, seqlen_q, seqlen_k]
    enable_dropout = dropout_p > 0.0
    if enable_dropout:
        philox_seed = torch.randint(0, 0xFFFFFF, (1,))[
            0
        ].item()  # No specific reason to restrict range to 0xffffff
        philox_offset = torch.randint(0, 0xFFFFFF, (1,))[
            0
        ].item()  # Pass in an int, not Tensor
    else:
        philox_seed = 0
        philox_offset = 0
    if return_softmax or enable_dropout:
        s_dmask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
        dropout_mask = torch.zeros(
            (batch, num_q_heads, max_seqlen_q, max_seqlen_k),
            device=q.device,
            dtype=torch.float32,
        )
    else:
        s_dmask = None
        dropout_mask = None

    if config is None:
        config = _get_config(enable_dropout, q.dtype, has_pe=pe_head_dim > 0)

    # Flash Attention 2: we also launch blocks over sequence length
    grid = lambda META: (  # noqa: E731
        batch * num_q_heads * triton.cdiv(seqlen_q, META["BLOCK_M"]),
    )

    buffer_size = 16  # we want to load contiguous 128 bits at a time
    size_per_thread_d = buffer_size // q.element_size()
    size_per_thread_m = triton.cdiv(
        config["BLOCK_M"] // (config["num_warps"] * 64), size_per_thread_d
    )
    size_per_thread_m = max(1, size_per_thread_m)
    size_per_thread_n = triton.cdiv(
        config["BLOCK_N"] // (config["num_warps"] * 64), size_per_thread_d
    )
    size_per_thread_n = max(1, size_per_thread_n)

    mfma_instr_shape_m, mfma_instr_shape_n, mfma_instr_shape_k = _get_mfma_instr_shape(
        q.dtype
    )

    _attn_fwd[grid](
        q,
        k,
        v,
        descale_q,
        descale_k,
        descale_v,
        o,
        alibi_slopes,
        s_dmask,
        dropout_mask,
        softmax_lse,
        *q_strides,
        *k_strides,
        *v_strides,
        descale_q.stride(0) if descale_q is not None else 0,
        descale_k.stride(0) if descale_k is not None else 0,
        descale_v.stride(0) if descale_v is not None else 0,
        *o_strides,
        alibi_slopes.stride(0) if alibi_slopes is not None else 0,
        alibi_slopes.stride(1) if alibi_slopes is not None else 0,
        s_dmask.stride(0) if s_dmask is not None else 0,
        s_dmask.stride(1) if s_dmask is not None else 0,
        s_dmask.stride(2) if s_dmask is not None else 0,
        s_dmask.stride(3) if s_dmask is not None else 0,
        stride_lse_z if softmax_lse is not None else 0,
        stride_lse_h if softmax_lse is not None else 0,
        stride_lse_m if softmax_lse is not None else 0,
        softmax_scale,
        cu_seqlens_q,
        cu_seqlens_k,
        dropout_p,
        philox_seed,
        philox_offset,
        SEQLEN_Q=max_seqlen_q,
        SEQLEN_K=max_seqlen_k,
        IS_CAUSAL=causal,
        NUM_Q_HEADS=num_q_heads,
        NUM_K_HEADS=num_k_heads,
        BLOCK_DMODEL=v_head_dim,
        BLOCK_DMODEL_POW2=BLOCK_DMODEL_POW2,
        BLOCK_DMODEL_PE=pe_head_dim,
        RETURN_SCORES=return_softmax,
        ENABLE_DROPOUT=enable_dropout,
        IS_FP8=IS_FP8,
        FP8_MAX=FP8_MAX,
        VARLEN=is_varlen,
        BATCH=batch,
        NUM_XCD=get_num_xcds(),
        USE_INT64_STRIDES=_USE_INT64_STRIDES,
        MFMA_INSTR_SHAPE_M=mfma_instr_shape_m,
        MFMA_INSTR_SHAPE_N=mfma_instr_shape_n,
        MFMA_INSTR_SHAPE_K=mfma_instr_shape_k,
        SIZE_PER_THREAD_M=size_per_thread_m,
        SIZE_PER_THREAD_N=size_per_thread_n,
        SIZE_PER_THREAD_D=size_per_thread_d,
        **config,
    )

    return o, softmax_lse, s_dmask, philox_seed, philox_offset


class _FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        config=None,
    ):
        raise NotImplementedError(
            "Flash Attention with bshd/bhsd layout is not implemented yet for Gluon"
        )

    @staticmethod
    def backward(ctx, do, *args):
        raise NotImplementedError("Backward pass not implemented for Gluon")


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    config: Optional[dict[str, any]] = None,
):
    raise NotImplementedError(
        "Flash Attention with bshd/bhsd layout is not implemented yet for Gluon"
    )


class _FlashAttnFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        config=None,
    ):
        raise NotImplementedError(
            "Flash Attention with bshd/bhsd layout is not implemented yet for Gluon"
        )

    @staticmethod
    def backward(ctx, do, *args):
        raise NotImplementedError("Backward pass not implemented for Gluon")


def flash_attn_fp8_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    config: Optional[dict[str, any]] = None,
):
    raise NotImplementedError(
        "Flash Attention with bshd/bhsd layout is not implemented yet for Gluon"
    )


class _FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        block_table,
        out,
        is_grad_enabled,
        config=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=bias,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                return_softmax=return_softmax and dropout_p > 0.0,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                config=config,
            )
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k
            )
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.bias = bias
            ctx.alibi_slopes = alibi_slopes
        out = out_padded[..., :head_size_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        raise NotImplementedError("Backward pass not implemented for Gluon")


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    block_table=None,
    out=None,
    config: Optional[dict[str, any]] = None,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        bias: (seqlen_q, seqlen_k)
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """

    _LOGGER.info(
        f"FLASH_ATTN_VARLEN:  q={tuple(q.shape)}  k={tuple(k.shape)}  v={tuple(v.shape)}"
    )
    return _FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        block_table,
        out,
        torch.is_grad_enabled(),
        config,
    )


class _FlashAttnVarlenFP8Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        block_table,
        is_grad_enabled,
        config=None,
    ):
        raise NotImplementedError("FP8 not implemented for Gluon")

    @staticmethod
    def backward(ctx, do, *args):
        raise NotImplementedError("Backward pass not implemented for Gluon")


def flash_attn_varlen_fp8_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    block_table=None,
    config: Optional[dict[str, any]] = None,
):
    raise NotImplementedError("FP8 not implemented for Gluon")

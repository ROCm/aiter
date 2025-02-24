# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor, Generator
from typing import Optional, Tuple
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR, AITER_ROOT_DIR
import torch

@compile_ops("module_mha_fwd_fp16_nobias", fc_name="mha_fwd_fp16_nobias")
def mha_fwd_fp16_nobias(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_fwd_fp16_alibi", fc_name="mha_fwd_fp16_alibi")
def mha_fwd_fp16_alibi(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_fwd_bf16_nobias", fc_name="mha_fwd_bf16_nobias")
def mha_fwd_bf16_nobias(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_fwd_bf16_alibi", fc_name="mha_fwd_bf16_alibi")
def mha_fwd_bf16_alibi(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_varlen_fwd_fp16_nobias", fc_name="mha_varlen_fwd_fp16_nobias")
def mha_varlen_fwd_fp16_nobias(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_varlen_fwd_fp16_alibi", fc_name="mha_varlen_fwd_fp16_alibi")
def mha_varlen_fwd_fp16_alibi(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_varlen_fwd_bf16_nobias", fc_name="mha_varlen_fwd_bf16_nobias")
def mha_varlen_fwd_bf16_nobias(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_varlen_fwd_bf16_alibi", fc_name="mha_varlen_fwd_bf16_alibi")
def mha_varlen_fwd_bf16_alibi(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_bwd_fp16_nobias", fc_name="mha_bwd_fp16_nobias")
def mha_bwd_fp16_nobias(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_bwd_fp16_alibi", fc_name="mha_bwd_fp16_alibi")
def mha_bwd_fp16_alibi(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_bwd_bf16_nobias", fc_name="mha_bwd_bf16_nobias")
def mha_bwd_bf16_nobias(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_bwd_bf16_alibi", fc_name="mha_bwd_bf16_alibi")
def mha_bwd_bf16_alibi(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_varlen_bwd_fp16_nobias", fc_name="mha_varlen_bwd_fp16_nobias")
def mha_varlen_bwd_fp16_nobias(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_varlen_bwd_fp16_alibi", fc_name="mha_varlen_bwd_fp16_alibi")
def mha_varlen_bwd_fp16_alibi(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_varlen_bwd_bf16_nobias", fc_name="mha_varlen_bwd_bf16_nobias")
def mha_varlen_bwd_bf16_nobias(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


@compile_ops("module_mha_varlen_bwd_bf16_alibi", fc_name="mha_varlen_bwd_bf16_alibi")
def mha_varlen_bwd_bf16_alibi(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
): ...


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    if q.dtype == torch.float16:
        if alibi_slopes is None:
            out, softmax_lse, S_dmask, rng_state = mha_fwd_fp16_nobias(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                return_lse,
                return_softmax,
                None,
                alibi_slopes,
                None,
            )
        elif alibi_slopes is not None:
            out, softmax_lse, S_dmask, rng_state = mha_fwd_fp16_alibi(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                return_lse,
                return_softmax,
                None,
                alibi_slopes,
                None,
            )
    elif q.dtype == torch.bfloat16:
        if alibi_slopes is None:
            out, softmax_lse, S_dmask, rng_state = mha_fwd_bf16_nobias(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                return_lse,
                return_softmax,
                None,
                alibi_slopes,
                None,
            )
        elif alibi_slopes is not None:
            out, softmax_lse, S_dmask, rng_state = mha_fwd_bf16_alibi(
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                return_lse,
                return_softmax,
                None,
                alibi_slopes,
                None,
            )
    return out, softmax_lse, S_dmask, rng_state


def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    if q.dtype == torch.float16:
        if alibi_slopes is None:
            dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
            (
                dq,
                dk,
                dv,
                softmax_d,
            ) = mha_bwd_fp16_nobias(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                deterministic,
                dq,
                dk,
                dv,
                alibi_slopes,
                rng_state,
                None,
            )
        elif alibi_slopes is not None:
            dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
            (
                dq,
                dk,
                dv,
                softmax_d,
            ) = mha_bwd_fp16_alibi(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                deterministic,
                dq,
                dk,
                dv,
                alibi_slopes,
                rng_state,
                None,
            )
    elif q.dtype == torch.bfloat16:
        if alibi_slopes is None:
            dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
            (
                dq,
                dk,
                dv,
                softmax_d,
            ) = mha_bwd_bf16_nobias(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                deterministic,
                dq,
                dk,
                dv,
                alibi_slopes,
                rng_state,
                None,
            )
        elif alibi_slopes is not None:
            dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
            (
                dq,
                dk,
                dv,
                softmax_d,
            ) = mha_bwd_bf16_alibi(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                dropout_p,
                softmax_scale,
                causal,
                window_size_left,
                window_size_right,
                deterministic,
                dq,
                dk,
                dv,
                alibi_slopes,
                rng_state,
                None,
            )
    return softmax_d


class FlashAttnFunc(torch.autograd.Function):
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
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
        )
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
        out = out_padded[..., :head_size_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return tuple(result)


    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_og = dout.size(3)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _flash_attn_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
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
        q: (batch_size, seqlen, nheads, headdim)
        k: (batch_size, seqlen, nheads_k, headdim)
        v: (batch_size, seqlen, nheads_k, headdim)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (batch_size, seqlen, nheads, headdim).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    return FlashAttnFunc.apply(
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
        return_attn_probs,
        torch.is_grad_enabled(),
    )


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int = -1,
    window_size_right: int = -1,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    return_softmax: bool = False,
    block_table: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if q.dtype == torch.float16:
        if alibi_slopes is None:
            out, softmax_lse, S_dmask, rng_state = mha_varlen_fwd_fp16_nobias(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                return_lse,
                return_softmax,
                None,
                block_table,
                alibi_slopes,
                None
            )
        elif alibi_slopes is not None:
            out, softmax_lse, S_dmask, rng_state = mha_varlen_fwd_fp16_alibi(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                return_lse,
                return_softmax,
                None,
                block_table,
                alibi_slopes,
                None
            )
    elif q.dtype == torch.bfloat16:
        if alibi_slopes is None:
            out, softmax_lse, S_dmask, rng_state = mha_varlen_fwd_bf16_nobias(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                return_lse,
                return_softmax,
                None,
                block_table,
                alibi_slopes,
                None
            )
        elif alibi_slopes is not None:
            out, softmax_lse, S_dmask, rng_state = mha_varlen_fwd_bf16_alibi(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                return_lse,
                return_softmax,
                None,
                block_table,
                alibi_slopes,
                None
            )
    return out, softmax_lse, S_dmask, rng_state


def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
) -> torch.Tensor:
    # dq, dk, dv are allocated by us so they should already be contiguous
    if q.dtype == torch.float16:
        if alibi_slopes is None:
            dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
            (
                dq,
                dk,
                dv,
                softmax_d,
            ) = mha_varlen_bwd_fp16_nobias(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                deterministic,
                dq,
                dk,
                dv,
                alibi_slopes,
                rng_state,
                None,
            )
        elif alibi_slopes is not None:
            dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
            (
                dq,
                dk,
                dv,
                softmax_d,
            ) = mha_varlen_bwd_fp16_alibi(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                deterministic,
                dq,
                dk,
                dv,
                alibi_slopes,
                rng_state,
                None,
            )
    elif q.dtype == torch.bfloat16:
        if alibi_slopes is None:
            dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
            (
                dq,
                dk,
                dv,
                softmax_d,
            ) = mha_varlen_bwd_bf16_nobias(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                deterministic,
                dq,
                dk,
                dv,
                alibi_slopes,
                rng_state,
                None,
            )
        elif alibi_slopes is not None:
            dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
            (
                dq,
                dk,
                dv,
                softmax_d,
            ) = mha_varlen_bwd_bf16_alibi(
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p,
                softmax_scale,
                zero_tensors,
                causal,
                window_size_left,
                window_size_right,
                deterministic,
                dq,
                dk,
                dv,
                alibi_slopes,
                rng_state,
                None,
            )
    return softmax_d


class FlashAttnVarlenFunc(torch.autograd.Function):
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
    ):
        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            alibi_slopes=alibi_slopes,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
            block_table=block_table,
        )
        if is_grad:
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_og = dout.size(2)
        dout_padded = dout
        if head_size_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_og % 8])
        _flash_attn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
        )
        dq = dq[..., : dout.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : dout.shape[-1]]
        dv = dv[..., : dout.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None


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
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    block_table=None,
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
    return FlashAttnVarlenFunc.apply(
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
        return_attn_probs,
        block_table,
        torch.is_grad_enabled(),
    )

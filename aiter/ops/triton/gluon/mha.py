# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional, Tuple
import torch
import triton
import triton.language as tl
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl

import aiter.ops.triton.utils.types as types
from aiter.ops.triton.mha_onekernel_bwd import flash_attn_onekernel_backward
from aiter.ops.triton.mha_fused_bwd import flash_attn_fused_backward
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.gluon._gluon_kernels.mha import _attn_fwd, _get_config


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
        q_strides = (0, q.stride(1), q.stride(0), q.stride(2))
        k_strides = (0, k.stride(1), k.stride(0), k.stride(2))
        v_strides = (0, v.stride(1), v.stride(0), v.stride(2))
        o_strides = (0, o.stride(1), o.stride(0), o.stride(2))
    else:
        # Layout is bshd.
        # q and k are [batch, seq_len, num_head, head_dim_qk].
        # v is [batch, seq_len, num_head, head_dim_v].
        batch, seqlen_q, num_q_heads = q.shape[:-1]
        num_k_heads = k.shape[2]
        q_strides = (q.stride(0), q.stride(2), q.stride(1), q.stride(3))
        k_strides = (k.stride(0), k.stride(2), k.stride(1), k.stride(3))
        v_strides = (v.stride(0), v.stride(2), v.stride(1), v.stride(3))
        o_strides = (o.stride(0), o.stride(2), o.stride(1), o.stride(3))

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

    # softmax_lse [batch, num_q_heads, seqlen_q]
    if is_varlen:
        softmax_lse = torch.zeros(
            (q.shape[0], num_q_heads), device=q.device, dtype=torch.float32
        )
        stride_lse_z, stride_lse_h, stride_lse_m = (
            0,
            softmax_lse.stride(1),
            softmax_lse.stride(0),
        )
    else:
        softmax_lse = torch.zeros(
            (batch, num_q_heads, max_seqlen_q), device=q.device, dtype=torch.float32
        )
        stride_lse_z, stride_lse_h, stride_lse_m = softmax_lse.stride()

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

    """
    # Tuned for MI300x
    config = {
        "BLOCK_M": 128,
        "BLOCK_N": 64,
        "waves_per_eu": 2,
        "num_warps": 4,
        "num_ctas": 1,
        "num_stages": 1,
    }
    # Dropout significantly increases VGPR usage so use small tiles
    if enable_dropout or q.dtype == torch.float32:
        config = {
            "BLOCK_M": 32,
            "BLOCK_N": 32,
            "waves_per_eu": 1,
            "num_warps": 2,
            "num_ctas": 1,
            "num_stages": 1,
        }
    """

    grid = lambda META: (  # noqa: E731
        batch * num_q_heads * triton.cdiv(seqlen_q, META["BLOCK_M"]),
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
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
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
                return_softmax=return_softmax and dropout_p > 0,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                config=config,
            )
        )

        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse)
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.bias = bias
            ctx.window_size = window_size
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic

        out = out_padded[..., :head_size_og]
        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, do, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])

        if _USE_FUSED_BWD_KERNEL:
            flash_attn_fused_backward(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,
                None,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        else:
            flash_attn_onekernel_backward(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,
                None,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )

        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
    pass


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
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        # cast input to fp8
        fp8_dtype = types.get_fp8_e4m3_dtype()
        q_fp8, descale_q = _cast_to_fp8(q, fp8_dtype, "bshd")
        k_fp8, descale_k = _cast_to_fp8(k, fp8_dtype, "bshd")
        v_fp8, descale_v = _cast_to_fp8(v, fp8_dtype, "bshd")

        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(
                q_fp8,
                k_fp8,
                v_fp8,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=None,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                return_softmax=return_softmax and dropout_p > 0,
                max_seqlen_q=q.shape[1],
                max_seqlen_k=k.shape[1],
                cu_seqlens_q=None,
                cu_seqlens_k=None,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                config=config,
            )
        )

        if is_grad:
            ctx.save_for_backward(
                q_fp8,
                k_fp8,
                v_fp8,
                out_padded,
                softmax_lse,
                descale_q,
                descale_k,
                descale_v,
            )
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
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
        q_fp8, k_fp8, v_fp8, out, softmax_lse, descale_q, descale_k, descale_v = (
            ctx.saved_tensors
        )
        dq, dk, dv = (
            torch.zeros_like(q_fp8, dtype=torch.float32),
            torch.zeros_like(k_fp8, dtype=torch.float32),
            torch.zeros_like(v_fp8, dtype=torch.float32),
        )
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])

        fp8_dtype = types.get_fp8_e4m3_dtype()
        do_padded_fp8, descale_do = _cast_to_fp8(do_padded, fp8_dtype, "bshd")
        if _USE_FUSED_BWD_KERNEL:
            flash_attn_fused_backward(
                do_padded_fp8,
                q_fp8,
                k_fp8,
                v_fp8,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                None,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,
                None,
                max_seqlen_q=q_fp8.shape[1],
                max_seqlen_k=k_fp8.shape[1],
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        else:
            flash_attn_onekernel_backward(
                do_padded_fp8,
                q_fp8,
                k_fp8,
                v_fp8,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                None,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                None,
                None,
                max_seqlen_q=q_fp8.shape[1],
                max_seqlen_k=k_fp8.shape[1],
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )

        # dq = dq[..., : q_fp8.shape[-1]]  # We could have padded the head dimension
        # dk = dk[..., : k_fp8.shape[-1]]
        # dv = dv[..., : v_fp8.shape[-1]]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
    pass


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
        q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = torch.zeros_like(q), torch.empty_like(k), torch.empty_like(v)
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None
        head_size_og = do.size(2)
        do_padded = do
        if head_size_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_og % 8])

        if _USE_FUSED_BWD_KERNEL:
            flash_attn_fused_backward(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        else:
            flash_attn_onekernel_backward(
                do_padded,
                q,
                k,
                v,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                dbias,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )

        dq = dq[..., : q.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k.shape[-1]]
        dv = dv[..., : v.shape[-1]]
        return (
            dq,
            dk,
            dv,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dbias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
    pass


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
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_og = q.size(2)
        if head_size_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
            v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])

        # cast input to fp8
        fp8_dtype = types.get_fp8_e4m3_dtype()
        q_fp8, descale_q = _cast_varlen_to_fp8(q, fp8_dtype, cu_seqlens=cu_seqlens_q)
        k_fp8, descale_k = _cast_varlen_to_fp8(k, fp8_dtype, cu_seqlens=cu_seqlens_k)
        v_fp8, descale_v = _cast_varlen_to_fp8(v, fp8_dtype, cu_seqlens=cu_seqlens_k)

        out_padded, softmax_lse, S_dmask, philox_seed, philox_offset = (
            _flash_attn_forward(
                q_fp8,
                k_fp8,
                v_fp8,
                dropout_p,
                softmax_scale,
                causal=causal,
                window_size_left=int(window_size[0]),
                window_size_right=int(window_size[1]),
                bias=None,
                alibi_slopes=alibi_slopes,
                return_lse=return_lse,
                return_softmax=return_softmax and dropout_p > 0,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                config=config,
            )
        )
        if is_grad:
            ctx.save_for_backward(
                q_fp8,
                k_fp8,
                v_fp8,
                out_padded,
                softmax_lse,
                cu_seqlens_q,
                cu_seqlens_k,
                descale_q,
                descale_k,
                descale_v,
            )
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.philox_seed = philox_seed
            ctx.philox_offset = philox_offset
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
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
        (
            q_fp8,
            k_fp8,
            v_fp8,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            descale_q,
            descale_k,
            descale_v,
        ) = ctx.saved_tensors
        dq, dk, dv = (
            torch.zeros_like(q_fp8, dtype=torch.float32),
            torch.zeros_like(k_fp8, dtype=torch.float32),
            torch.zeros_like(v_fp8, dtype=torch.float32),
        )
        head_size_v_og = do.size(3)
        do_padded = do
        if head_size_v_og % 8 != 0:
            do_padded = torch.nn.functional.pad(do, [0, 8 - head_size_v_og % 8])

        fp8_dtype = types.get_fp8_e4m3_dtype()
        do_padded_fp8, descale_do = _cast_varlen_to_fp8(
            do_padded, fp8_dtype, "thd", cu_seqlens_q
        )
        if _USE_FUSED_BWD_KERNEL:
            flash_attn_fused_backward(
                do_padded_fp8,
                q_fp8,
                k_fp8,
                v_fp8,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                None,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        else:
            flash_attn_onekernel_backward(
                do_padded_fp8,
                q_fp8,
                k_fp8,
                v_fp8,
                out,
                softmax_lse,
                dq,
                dk,
                dv,
                None,
                ctx.softmax_scale,
                ctx.alibi_slopes,
                ctx.causal,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=ctx.max_seqlen_k,
                dropout_p=ctx.dropout_p,
                philox_seed=ctx.philox_seed,
                philox_offset=ctx.philox_offset,
                descale_q=descale_q,
                descale_k=descale_k,
                descale_v=descale_v,
                descale_do=descale_do,
                USE_INT64_STRIDES=_USE_INT64_STRIDES,
            )
        dq = dq[..., : q_fp8.shape[-1]]  # We could have padded the head dimension
        dk = dk[..., : k_fp8.shape[-1]]
        dv = dv[..., : v_fp8.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


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
    pass

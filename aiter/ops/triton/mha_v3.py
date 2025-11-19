# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional, Tuple, Union
import torch

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3


class _FlashAttnV3Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        softmax_scale: float | None,
        causal: bool,
        qv: Optional[torch.Tensor],
        q_descale: Optional[torch.Tensor],
        k_descale: Optional[torch.Tensor],
        v_descale: Optional[torch.Tensor],
        window_size: Tuple[int, int],
        attention_chunk: int,
        softcap: float,
        num_splits: int,
        pack_gqa: Optional[bool],
        deterministic: bool,
        sm_margin: int,
    ):
        # Derive softmax scale if not provided (include qv width like Hopper v3)
        if softmax_scale is None:
            q_extra = qv.shape[-1] if qv is not None else 0
            softmax_scale = (q.shape[-1] + q_extra) ** (-0.5)

        # Fast validation of unsupported features
        if qv is not None:
            raise NotImplementedError("qv is not supported in AMD Triton v3 yet")
        if attention_chunk not in (0, 1):
            raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
        if softcap != 0.0:
            raise NotImplementedError("softcap not implemented in AMD Triton v3")
        if num_splits != 1:
            raise NotImplementedError("num_splits != 1 not supported in AMD Triton v3")
        if pack_gqa is not None:
            raise NotImplementedError("pack_gqa not implemented in AMD Triton v3")
        if sm_margin != 0:
            raise NotImplementedError("sm_margin != 0 not supported in AMD Triton v3")

        out, softmax_lse = flash_attn_3.fwd(
            q,
            k,
            v,
            None,  # k_new
            None,  # v_new
            None,  # qv
            None,  # out tensor (allocate inside)
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # cu_seqlens_k_new
            None,  # seqused_q
            None,  # seqused_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # page_table
            None,  # kv_batch_idx
            None,  # leftpad_k
            None,  # rotary_cos
            None,  # rotary_sin
            None,  # seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            int(window_size[0]),
            int(window_size[1]),
            attention_chunk,
            softcap,
            False,  # rotary_interleaved
            None,  # scheduler_metadata
            num_splits,
            pack_gqa,
            sm_margin,
        )

        ctx.save_for_backward(
            q, k, v, out, softmax_lse, q_descale, k_descale, v_descale
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, softmax_lse, q_descale, k_descale, v_descale = ctx.saved_tensors

        dq, dk, dv, _delta = flash_attn_3.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,  # dq
            None,  # dk
            None,  # dv
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # seqused_q
            None,  # seqused_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            ctx.softmax_scale,
            ctx.causal,
            int(ctx.window_size[0]),
            int(ctx.window_size[1]),
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return (
            dq,  # q
            dk,  # k
            dv,  # v
            None,  # softmax_scale
            None,  # causal
            None,  # qv
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
            None,  # window_size
            None,  # attention_chunk
            None,  # softcap
            None,  # num_splits
            None,  # pack_gqa
            None,  # deterministic
            None,  # sm_margin
        )


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
):
    """FlashAttention v3 entry point."""
    return _FlashAttnV3Func.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        qv,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        num_splits,
        pack_gqa,
        deterministic,
        sm_margin,
    )


class _FlashAttnVarlenV3Func(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None,
        causal: bool,
        q_descale: torch.Tensor | None,
        k_descale: torch.Tensor | None,
        v_descale: torch.Tensor | None,
        window_size: tuple[int, int],
        attention_chunk: int,
        softcap: float,
        deterministic: bool,
        sm_margin: int,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if attention_chunk != 0:
            raise NotImplementedError(
                "attention_chunk != 0 not supported in varlen v3 yet"
            )
        if softcap != 0.0:
            raise NotImplementedError("softcap not implemented in varlen v3 yet")
        if sm_margin != 0:
            raise NotImplementedError("sm_margin != 0 not supported in varlen v3 yet")

        out, softmax_lse = flash_attn_3.fwd(
            q,
            k,
            v,
            None,  # k_new
            None,  # v_new
            None,  # qv
            None,  # out tensor
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # cu_seqlens_k_new
            None,  # seqused_q
            None,  # seqused_k
            max_seqlen_q,
            max_seqlen_k,
            None,  # page_table
            None,  # kv_batch_idx
            None,  # leftpad_k
            None,  # rotary_cos
            None,  # rotary_sin
            None,  # seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            int(window_size[0]),
            int(window_size[1]),
            attention_chunk,
            softcap,
            False,  # rotary_interleaved
            None,  # scheduler_metadata
            1,  # num_splits
            None,  # pack_gqa
            sm_margin,
        )

        ctx.save_for_backward(
            q, k, v, out, softmax_lse, q_descale, k_descale, v_descale
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, softmax_lse, q_descale, k_descale, v_descale = ctx.saved_tensors

        dq, dk, dv, _delta = flash_attn_3.bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            None,  # dq
            None,  # dk
            None,  # dv
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            None,  # seqused_q
            None,  # seqused_k
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
            int(ctx.window_size[0]),
            int(ctx.window_size[1]),
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return (
            dq,
            dk,
            dv,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # softmax_scale
            None,  # causal
            None,  # q_descale
            None,  # k_descale
            None,  # v_descale
            None,  # window_size
            None,  # attention_chunk
            None,  # softcap
            None,  # deterministic
            None,  # sm_margin
        )


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
):
    """FlashAttention v3 varlen path."""
    return _FlashAttnVarlenV3Func.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        q_descale,
        k_descale,
        v_descale,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        sm_margin,
    )


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    qv: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[Union[torch.Tensor, int]] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 0,
    pack_gqa: Optional[bool] = None,
    sm_margin: int = 0,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    return_softmax_lse: bool = False,
    page_table: Optional[torch.Tensor] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    rotary_cos: Optional[torch.Tensor] = None,
    rotary_sin: Optional[torch.Tensor] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
):
    """
    Arguments mirror Hopper's `flash_attn_with_kvcache` with current backend limitations.
    Unsupported: backward, qv, softcap!=0, pack_gqa, sm_margin!=0, attention_chunk>1, num_splits>1,
    simultaneous varlen (cu_seqlens_q) + cache_seqlens tensor, and partial rotary inputs.
    """
    # Scale
    if softmax_scale is None:
        q_extra = qv.shape[-1] if qv is not None else 0
        softmax_scale = (q.shape[-1] + q_extra) ** (-0.5)

    # Feature guards
    if qv is not None:
        raise NotImplementedError("qv not supported in KV cache path yet")
    if softcap != 0.0:
        raise NotImplementedError("softcap not implemented in KV cache path")
    if pack_gqa is not None:
        raise NotImplementedError("pack_gqa not implemented in KV cache path")
    if sm_margin != 0:
        raise NotImplementedError("sm_margin != 0 not supported in KV cache path")
    if attention_chunk not in (0, 1):
        raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
    if num_splits not in (0, 1):
        raise NotImplementedError("num_splits > 1 not supported in KV cache path")

    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )

    if cu_seqlens_q is not None and cache_seqlens is not None:
        raise NotImplementedError(
            "Varlen decode with cache_seqlens tensor not supported yet"
        )
    if (rotary_cos is None) ^ (rotary_sin is None):
        raise ValueError(
            "Both rotary_cos and rotary_sin must be provided together or neither"
        )
    if (
        (rotary_cos is not None)
        and rotary_seqlens is not None
        and cu_seqlens_q is None
        and cache_seqlens is None
    ):
        raise ValueError(
            "rotary_seqlens provided without cu_seqlens_q or cache_seqlens context"
        )

    kv_batch_idx = cache_batch_idx
    leftpad_k = cache_leftpad
    seqlens_rotary = rotary_seqlens

    out, softmax_lse = flash_attn_3.fwd(
        q,
        k_cache,
        v_cache,
        k,
        v,
        None,  # qv
        None,  # out allocate
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens if isinstance(cache_seqlens, torch.Tensor) else None,  # seqused_k
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        kv_batch_idx,
        leftpad_k,
        rotary_cos,
        rotary_sin,
        seqlens_rotary,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        int(window_size[0]),
        int(window_size[1]),
        attention_chunk,
        softcap,
        False,  # rotary_interleaved
        None,  # scheduler_metadata
        num_splits if num_splits != 0 else 1,
        pack_gqa,
        sm_margin,
    )
    return (out, softmax_lse) if return_softmax_lse else out


# -------------------------------
# FP8 Wrappers
# -------------------------------
# do the quantization to fp8 internally and maintain high-precision inputs/outputs


def _quantize_bshd(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    clamp_val=1e-9,
    group_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a tensor to FP8 format, returning an FP8 tensor and a descale factor.

    Args:
        x (torch.Tensor): shape [batch, seq_len, heads, dim]
        fp8_dtype (torch.dtype): FP8 data type (e.g., torch.float8_e4m3fnuz)
        clamp_val (float): minimum value for scaling to avoid division by zero
        group_size (int, optional): For GQA/MQA on query tensors, specify the group size (num_heads // num_kv_heads)
                                     to group query heads appropriately. If None, computes scaling per head.
    Returns:
        x_fp8 (torch.Tensor): FP8 tensor with the same shape as x (leaf tensor if requires_grad=True)
        descale_factor (torch.Tensor): tensor of shape [batch, num_heads // group_size] if group_size is specified,
                                        otherwise [batch, heads]
    """
    if len(x.shape) != 4:
        raise ValueError(
            f"'bshd' tensor should have shape [batch, seqlen, heads, dim], got {x.shape}"
        )

    batch, seqlen, num_heads, head_dim = x.shape

    # For GQA/MQA: if group_size is specified and > 1,
    # we need to group query heads and compute scaling per group
    if group_size is not None and group_size > 1:
        assert (
            num_heads % group_size == 0
        ), f"num_heads ({num_heads}) must be divisible by group_size ({group_size})"

        num_groups = num_heads // group_size

        # Reshape to group query heads: [batch, seqlen, num_groups, group_size, head_dim]
        x_grouped = x.view(batch, seqlen, num_groups, group_size, head_dim)

        # Compute max over seqlen, group_size (query heads in group), and head_dim
        # Result shape: [batch, num_groups]
        x_abs_max = x_grouped.abs().amax(dim=(1, 3, 4))
        x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))

        # Unsqueeze to [batch, 1, num_groups, 1, 1] for broadcasting
        x_abs_max_broadcast = x_abs_max.unsqueeze(1).unsqueeze(3).unsqueeze(4)

        # Compute scale and descale
        fp8_max = torch.finfo(fp8_dtype).max
        scale = fp8_max / x_abs_max_broadcast
        descale_factor = (x_abs_max / fp8_max).to(torch.float32)

        # Quantize to FP8 and reshape back to original shape
        x_fp8 = (
            (x_grouped * scale).view(batch, seqlen, num_heads, head_dim).to(fp8_dtype)
        )
    else:
        # Standard case: compute scaling per head
        reduce_dims = (1, 3)  # seq_len and dim dimensions

        # Compute the absolute max along reduce_dims, clamped to avoid 0-scale
        # Result shape: [batch, heads]
        x_abs_max = x.abs().amax(dim=reduce_dims)
        x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))

        # Unsqueeze to [batch, 1, heads, 1] for broadcasting during scaling
        x_abs_max_broadcast = x_abs_max.unsqueeze(1).unsqueeze(3)

        # compute scale and descale
        fp8_max = torch.finfo(fp8_dtype).max
        scale = fp8_max / x_abs_max_broadcast
        descale_factor = (x_abs_max / fp8_max).to(torch.float32)

        # Quantize to FP8
        x_fp8 = (x * scale).to(fp8_dtype)

    # Detach to make a leaf tensor, This is required because PyTorch only populates .grad on leaf tensors
    # x_fp8_leaf = x_fp8.detach().requires_grad_(True)

    return x_fp8, descale_factor


def _quantize_thd(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    cu_seqlens: torch.Tensor,
    clamp_val=1e-9,
    group_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a tensor to FP8 format for varlen inputs, returning an FP8 tensor and a descale factor.

    This function computes descale factors per sequence in the batch, analogous to how _quantize_bshd
    computes per-batch descale factors.

    Args:
        x (torch.Tensor): shape [total_tokens, heads, dim]
        fp8_dtype (torch.dtype): FP8 data type (e.g., torch.float8_e4m3fnuz)
        cu_seqlens (torch.Tensor): Cumulative sequence lengths [batch_size + 1]
        clamp_val (float): minimum value for scaling to avoid division by zero
        group_size (int, optional): For GQA/MQA on query tensors, specify the group size (num_heads // num_kv_heads)
                                     to group query heads appropriately. If None, computes scaling per head.
    Returns:
        x_fp8 (torch.Tensor): FP8 tensor with the same shape as x
        descale_factor (torch.Tensor): tensor of shape [batch_size, num_heads // group_size] if group_size is specified,
                                        otherwise [batch_size, heads]
    """
    if len(x.shape) != 3:
        raise ValueError(
            f"'thd' tensor should have shape [total_tokens, heads, dim], got {x.shape}"
        )

    total_tokens, num_heads, head_dim = x.shape
    batch_size = len(cu_seqlens) - 1

    fp8_max = torch.finfo(fp8_dtype).max

    # For GQA/MQA: if group_size is specified and > 1,
    # we need to group query heads and compute scaling per group
    if group_size is not None and group_size > 1:
        assert (
            num_heads % group_size == 0
        ), f"num_heads ({num_heads}) must be divisible by group_size ({group_size})"

        num_groups = num_heads // group_size

        # Reshape to group query heads: [total_tokens, num_groups, group_size, head_dim]
        x_grouped = x.view(total_tokens, num_groups, group_size, head_dim)

        # Compute descale factors per sequence (analogous to per-batch in bshd)
        descale_list = []
        x_fp8_list = []

        for b in range(batch_size):
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()

            # Get tokens for this sequence: [seq_len, num_groups, group_size, head_dim]
            x_seq = x_grouped[start:end]

            # Compute max over seq_len, group_size, and head_dim
            # Result shape: [num_groups]
            x_abs_max = x_seq.abs().amax(dim=(0, 2, 3))
            x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))

            # Compute descale for this sequence: [num_groups]
            descale_seq = (x_abs_max / fp8_max).to(torch.float32)
            descale_list.append(descale_seq)

            # Quantize this sequence
            # Unsqueeze to [1, num_groups, 1, 1] for broadcasting
            x_abs_max_broadcast = x_abs_max.unsqueeze(0).unsqueeze(2).unsqueeze(3)
            scale = fp8_max / x_abs_max_broadcast
            x_seq_fp8 = (x_seq * scale).to(fp8_dtype)
            x_fp8_list.append(x_seq_fp8)

        # Stack descale factors: [batch_size, num_groups]
        descale_factor = torch.stack(descale_list, dim=0)

        # Concatenate quantized sequences and reshape back to original shape
        x_fp8 = torch.cat(x_fp8_list, dim=0).view(total_tokens, num_heads, head_dim)
    else:
        # Standard case: compute scaling per head for each sequence
        descale_list = []
        x_fp8_list = []

        for b in range(batch_size):
            start = cu_seqlens[b].item()
            end = cu_seqlens[b + 1].item()

            # Get tokens for this sequence: [seq_len, num_heads, head_dim]
            x_seq = x[start:end]

            # Compute max over seq_len and head_dim
            # Result shape: [num_heads]
            x_abs_max = x_seq.abs().amax(dim=(0, 2))
            x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))

            # Compute descale for this sequence: [num_heads]
            descale_seq = (x_abs_max / fp8_max).to(torch.float32)
            descale_list.append(descale_seq)

            # Quantize this sequence
            # Unsqueeze to [1, num_heads, 1] for broadcasting
            x_abs_max_broadcast = x_abs_max.unsqueeze(0).unsqueeze(2)
            scale = fp8_max / x_abs_max_broadcast
            x_seq_fp8 = (x_seq * scale).to(fp8_dtype)
            x_fp8_list.append(x_seq_fp8)

        # Stack descale factors: [batch_size, num_heads]
        descale_factor = torch.stack(descale_list, dim=0)

        # Concatenate quantized sequences
        x_fp8 = torch.cat(x_fp8_list, dim=0)

    return x_fp8, descale_factor


class _FlashAttnFP8Wrapper(torch.autograd.Function):
    """
    FP8 Flash Attention wrapper that maintains high-precision inputs/outputs.

    This wrapper allows users to pass BF16/FP32 tensors and automatically handles
    the FP8 quantization internally, maintaining backward compatibility with
    high-precision training workflows.

    Forward: BF16/FP32 -> FP8 -> flash_attn -> FP32 output
    Backward: FP32 grad_out -> flash_attn_bwd -> FP32 grads -> input dtype grads
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # High precision (BF16/FP32)
        k: torch.Tensor,  # High precision (BF16/FP32)
        v: torch.Tensor,  # High precision (BF16/FP32)
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        attention_chunk: int,
        softcap: float,
        deterministic: bool,
        sm_margin: int,
    ):
        batch, seqlen, num_q_heads, head_dim = q.shape
        _, _, num_kv_heads, _ = k.shape

        # Quantize inputs to FP8
        fp8_dtype = torch.float8_e4m3fnuz

        # For GQA/MQA: quantize query with grouped scaling
        group_size = (
            num_q_heads // num_kv_heads if num_q_heads != num_kv_heads else None
        )
        q_fp8, q_descale = _quantize_bshd(q, fp8_dtype, group_size=group_size)
        k_fp8, k_descale = _quantize_bshd(k, fp8_dtype)
        v_fp8, v_descale = _quantize_bshd(v, fp8_dtype)

        # Verify descale shapes for GQA/MQA
        assert q_descale.shape == (
            batch,
            num_kv_heads,
        ), f"q_descale shape {q_descale.shape} != expected {(batch, num_kv_heads)}"
        assert k_descale.shape == (
            batch,
            num_kv_heads,
        ), f"k_descale shape {k_descale.shape} != expected {(batch, num_kv_heads)}"
        assert v_descale.shape == (
            batch,
            num_kv_heads,
        ), f"v_descale shape {v_descale.shape} != expected {(batch, num_kv_heads)}"

        # Derive softmax scale if not provided
        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)

        # Validate unsupported features
        if attention_chunk not in (0, 1):
            raise NotImplementedError("attention_chunk > 1 not supported (0 or 1 only)")
        if softcap != 0.0:
            raise NotImplementedError(
                "softcap not implemented in FP8 high-precision API"
            )
        if sm_margin != 0:
            raise NotImplementedError(
                "sm_margin != 0 not supported in FP8 high-precision API"
            )

        # Call flash attention forward
        out, softmax_lse = flash_attn_3.fwd(
            q_fp8,
            k_fp8,
            v_fp8,
            None,
            None,
            None,
            None,  # k_new, v_new, qv, out
            None,
            None,
            None,  # cu_seqlens_q, cu_seqlens_k, cu_seqlens_k_new
            None,
            None,
            None,
            None,  # seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
            None,
            None,
            None,  # page_table, kv_batch_idx, leftpad_k
            None,
            None,
            None,  # rotary_cos, rotary_sin, seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            int(window_size[0]),
            int(window_size[1]),
            attention_chunk,
            softcap,
            False,  # rotary_interleaved
            None,
            1,
            None,
            sm_margin,  # scheduler_metadata, num_splits, pack_gqa, sm_margin
        )

        # Save tensors needed for backward
        ctx.save_for_backward(
            q_fp8, k_fp8, v_fp8, out, softmax_lse, q_descale, k_descale, v_descale
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.input_dtype = q.dtype

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Compute gradients w.r.t. inputs.
        The backward pass returns FP32 gradients, which we convert to the input dtype.
        """
        # Retrieve saved tensors
        q_fp8, k_fp8, v_fp8, out, softmax_lse, q_descale, k_descale, v_descale = (
            ctx.saved_tensors
        )

        # Call flash attention backward - returns FP32 gradients
        dq, dk, dv, _delta = flash_attn_3.bwd(
            grad_output,
            q_fp8,
            k_fp8,
            v_fp8,
            out,
            softmax_lse,
            None,
            None,
            None,  # dq, dk, dv (will be allocated)
            None,
            None,  # cu_seqlens_q, cu_seqlens_k
            None,
            None,
            None,
            None,  # seqused_q, seqused_k, max_seqlen_q, max_seqlen_k
            ctx.softmax_scale,
            ctx.causal,
            int(ctx.window_size[0]),
            int(ctx.window_size[1]),
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

        # Convert gradients to input dtype (FP32 -> BF16 if needed)
        dq = dq.to(ctx.input_dtype)
        dk = dk.to(ctx.input_dtype)
        dv = dv.to(ctx.input_dtype)

        # Return gradients for all forward inputs (None for non-tensor inputs)
        return (
            dq,  # q
            dk,  # k
            dv,  # v
            None,  # softmax_scale
            None,  # causal
            None,  # window_size
            None,  # attention_chunk
            None,  # softcap
            None,  # deterministic
            None,  # sm_margin
        )


def flash_attn_fp8_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    qv: Optional[torch.Tensor] = None,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    deterministic: bool = False,
    sm_margin: int = 0,
):
    """
    FlashAttention v3 FP8 high-precision entry point.

    This function accepts high-precision (BF16/FP32) tensors and internally
    quantizes them to FP8 for computation. The output and gradients remain
    in high precision (FP32 for output, input dtype for gradients).

    This API is designed for seamless integration with existing training code
    that uses BF16/FP32 tensors, providing FP8 acceleration without requiring
    manual quantization.

    Args:
        q: Query tensor [batch, seqlen, num_q_heads, head_dim] (BF16/FP32)
        k: Key tensor [batch, seqlen, num_kv_heads, head_dim] (BF16/FP32)
        v: Value tensor [batch, seqlen, num_kv_heads, head_dim] (BF16/FP32)
        softmax_scale: Scaling factor for softmax (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        qv: Extra query-value tensor (not yet supported in FP8 mode)
        window_size: Sliding window attention size (left, right)
        attention_chunk: Chunking parameter (0 or 1 only)
        softcap: Softcapping value (not yet supported in FP8 mode)
        num_splits: Number of splits for parallel processing (not yet supported in FP8 mode)
        pack_gqa: GQA packing flag (not yet supported in FP8 mode)
        deterministic: Whether to use deterministic backward
        sm_margin: SM margin parameter (not yet supported in FP8 mode)

    Returns:
        out: Output tensor [batch, seqlen, num_q_heads, head_dim] (FP32)

    Note:
        - Supports GQA/MQA (num_q_heads != num_kv_heads)
        - Automatically handles grouped quantization for GQA/MQA queries
        - Gradients are computed in FP32 and converted to input dtype
        - qv, softcap, num_splits, pack_gqa, and sm_margin are not yet supported in FP8 mode
    """
    # Check that inputs are high precision (not already FP8)
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"flash_attn_fp8_func expects high-precision inputs (fp16/bf16/fp32), got q.dtype={q.dtype}. "
        f"If you already have FP8 tensors, use flash_attn_func() with q_descale/k_descale/v_descale parameters instead."
    )
    assert k.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"flash_attn_fp8_func expects high-precision inputs (fp16/bf16/fp32), got k.dtype={k.dtype}. "
        f"If you already have FP8 tensors, use flash_attn_func() with q_descale/k_descale/v_descale parameters instead."
    )
    assert v.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"flash_attn_fp8_func expects high-precision inputs (fp16/bf16/fp32), got v.dtype={v.dtype}. "
        f"If you already have FP8 tensors, use flash_attn_func() with q_descale/k_descale/v_descale parameters instead."
    )

    if qv is not None:
        raise NotImplementedError("qv not supported in FP8 high-precision API")
    if softcap != 0.0:
        raise NotImplementedError("softcap not supported in FP8 high-precision API")
    if num_splits != 1:
        raise NotImplementedError(
            "num_splits != 1 not supported in FP8 high-precision API"
        )
    if pack_gqa is not None:
        raise NotImplementedError("pack_gqa not supported in FP8 high-precision API")
    if sm_margin != 0:
        raise NotImplementedError(
            "sm_margin != 0 not supported in FP8 high-precision API"
        )

    return _FlashAttnFP8Wrapper.apply(
        q,
        k,
        v,
        softmax_scale,
        causal,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        sm_margin,
    )


class _FlashAttnVarlenFP8Wrapper(torch.autograd.Function):
    """
    FP8 Flash Attention varlen wrapper that maintains high-precision inputs/outputs.

    This wrapper allows users to pass BF16/FP32 tensors and automatically handles
    the FP8 quantization internally for variable-length sequences, maintaining
    backward compatibility with high-precision training workflows.

    Forward: BF16/FP32 -> FP8 -> flash_attn_varlen -> FP32 output
    Backward: FP32 grad_out -> flash_attn_varlen_bwd -> FP32 grads -> input dtype grads
    """

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,  # High precision (BF16/FP32)
        k: torch.Tensor,  # High precision (BF16/FP32)
        v: torch.Tensor,  # High precision (BF16/FP32)
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: Optional[float],
        causal: bool,
        window_size: Tuple[int, int],
        attention_chunk: int,
        softcap: float,
        deterministic: bool,
        sm_margin: int,
    ):
        # Determine heads and head_dim from input shapes
        total_q = q.shape[0]
        num_q_heads = q.shape[1]
        head_dim = q.shape[2]

        total_k = k.shape[0]
        num_kv_heads = k.shape[1]

        # Quantize inputs to FP8 using _quantize_thd for varlen tensors
        fp8_dtype = torch.float8_e4m3fnuz

        # For GQA/MQA: quantize query with grouped scaling
        group_size = (
            num_q_heads // num_kv_heads if num_q_heads != num_kv_heads else None
        )
        q_fp8, q_descale = _quantize_thd(
            q, fp8_dtype, cu_seqlens_q, group_size=group_size
        )
        k_fp8, k_descale = _quantize_thd(k, fp8_dtype, cu_seqlens_k)
        v_fp8, v_descale = _quantize_thd(v, fp8_dtype, cu_seqlens_k)

        # Verify descale shapes - _quantize_thd now returns shape [batch_size, num_heads] or [batch_size, num_groups]
        batch_size = len(cu_seqlens_q) - 1
        assert q_descale.shape == (
            batch_size,
            num_kv_heads,
        ), f"q_descale shape {q_descale.shape} != expected {(batch_size, num_kv_heads)}"
        assert k_descale.shape == (
            batch_size,
            num_kv_heads,
        ), f"k_descale shape {k_descale.shape} != expected {(batch_size, num_kv_heads)}"
        assert v_descale.shape == (
            batch_size,
            num_kv_heads,
        ), f"v_descale shape {v_descale.shape} != expected {(batch_size, num_kv_heads)}"

        # Derive softmax scale if not provided
        if softmax_scale is None:
            softmax_scale = head_dim ** (-0.5)

        # Validate unsupported features
        if attention_chunk != 0:
            raise NotImplementedError(
                "attention_chunk != 0 not supported in FP8 varlen high-precision API"
            )
        if softcap != 0.0:
            raise NotImplementedError(
                "softcap not implemented in FP8 varlen high-precision API"
            )
        if sm_margin != 0:
            raise NotImplementedError(
                "sm_margin != 0 not supported in FP8 varlen high-precision API"
            )

        # Call flash attention varlen forward
        out, softmax_lse = flash_attn_3.fwd(
            q_fp8,
            k_fp8,
            v_fp8,
            None,  # k_new
            None,  # v_new
            None,  # qv
            None,  # out tensor
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # cu_seqlens_k_new
            None,  # seqused_q
            None,  # seqused_k
            max_seqlen_q,
            max_seqlen_k,
            None,  # page_table
            None,  # kv_batch_idx
            None,  # leftpad_k
            None,  # rotary_cos
            None,  # rotary_sin
            None,  # seqlens_rotary
            q_descale,
            k_descale,
            v_descale,
            softmax_scale,
            causal,
            int(window_size[0]),
            int(window_size[1]),
            attention_chunk,
            softcap,
            False,  # rotary_interleaved
            None,  # scheduler_metadata
            1,  # num_splits
            None,  # pack_gqa
            sm_margin,
        )

        # Save tensors needed for backward
        ctx.save_for_backward(
            q_fp8, k_fp8, v_fp8, out, softmax_lse, q_descale, k_descale, v_descale
        )
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.input_dtype = q.dtype
        ctx.cu_seqlens_q = cu_seqlens_q
        ctx.cu_seqlens_k = cu_seqlens_k
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Compute gradients w.r.t. inputs.
        The backward pass returns FP32 gradients, which we convert to the input dtype.
        """
        # Retrieve saved tensors
        q_fp8, k_fp8, v_fp8, out, softmax_lse, q_descale, k_descale, v_descale = (
            ctx.saved_tensors
        )

        # Call flash attention varlen backward - returns FP32 gradients
        dq, dk, dv, _delta = flash_attn_3.bwd(
            grad_output,
            q_fp8,
            k_fp8,
            v_fp8,
            out,
            softmax_lse,
            None,  # dq
            None,  # dk
            None,  # dv
            ctx.cu_seqlens_q,
            ctx.cu_seqlens_k,
            None,  # seqused_q
            None,  # seqused_k
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.softmax_scale,
            ctx.causal,
            int(ctx.window_size[0]),
            int(ctx.window_size[1]),
            ctx.softcap,
            ctx.deterministic,
            ctx.sm_margin,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

        # Convert gradients to input dtype (FP32 -> BF16 if needed)
        dq = dq.to(ctx.input_dtype)
        dk = dk.to(ctx.input_dtype)
        dv = dv.to(ctx.input_dtype)

        # Return gradients for all forward inputs (None for non-tensor inputs)
        return (
            dq,  # q
            dk,  # k
            dv,  # v
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # softmax_scale
            None,  # causal
            None,  # window_size
            None,  # attention_chunk
            None,  # softcap
            None,  # deterministic
            None,  # sm_margin
        )


def flash_attn_varlen_fp8_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    deterministic: bool = False,
    sm_margin: int = 0,
):
    """
    FlashAttention v3 FP8 varlen high-precision entry point.

    This function accepts high-precision (BF16/FP32) tensors and internally
    quantizes them to FP8 for computation. The output and gradients remain
    in high precision (FP32 for output, input dtype for gradients).

    This API is designed for seamless integration with existing training code
    that uses BF16/FP32 tensors with variable-length sequences, providing
    FP8 acceleration without requiring manual quantization.

    Args:
        q: Query tensor [total_q, num_q_heads, head_dim] (BF16/FP32)
        k: Key tensor [total_k, num_kv_heads, head_dim] (BF16/FP32)
        v: Value tensor [total_k, num_kv_heads, head_dim] (BF16/FP32)
        cu_seqlens_q: Cumulative sequence lengths for queries [batch_size + 1]
        cu_seqlens_k: Cumulative sequence lengths for keys [batch_size + 1]
        max_seqlen_q: Maximum query sequence length
        max_seqlen_k: Maximum key sequence length
        softmax_scale: Scaling factor for softmax (default: 1/sqrt(head_dim))
        causal: Whether to apply causal masking
        window_size: Sliding window attention size (left, right)
        attention_chunk: Chunking parameter (must be 0 in varlen FP8 mode)
        softcap: Softcapping value (not yet supported in FP8 mode)
        deterministic: Whether to use deterministic backward
        sm_margin: SM margin parameter (not yet supported in FP8 mode)

    Returns:
        out: Output tensor [total_q, num_q_heads, head_dim] (FP32)

    Note:
        - Supports GQA/MQA (num_q_heads != num_kv_heads)
        - Automatically handles grouped quantization for GQA/MQA queries
        - Gradients are computed in FP32 and converted to input dtype
        - attention_chunk, softcap, and sm_margin are not yet supported in varlen FP8 mode
    """
    # Check that inputs are high precision (not already FP8)
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"flash_attn_varlen_fp8_func expects high-precision inputs (fp16/bf16/fp32), got q.dtype={q.dtype}. "
        f"If you already have FP8 tensors, use flash_attn_varlen_func() with q_descale/k_descale/v_descale parameters instead."
    )
    assert k.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"flash_attn_varlen_fp8_func expects high-precision inputs (fp16/bf16/fp32), got k.dtype={k.dtype}. "
        f"If you already have FP8 tensors, use flash_attn_varlen_func() with q_descale/k_descale/v_descale parameters instead."
    )
    assert v.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        f"flash_attn_varlen_fp8_func expects high-precision inputs (fp16/bf16/fp32), got v.dtype={v.dtype}. "
        f"If you already have FP8 tensors, use flash_attn_varlen_func() with q_descale/k_descale/v_descale parameters instead."
    )

    if attention_chunk != 0:
        raise NotImplementedError(
            "attention_chunk != 0 not supported in FP8 varlen high-precision API"
        )
    if softcap != 0.0:
        raise NotImplementedError(
            "softcap not supported in FP8 varlen high-precision API"
        )
    if sm_margin != 0:
        raise NotImplementedError(
            "sm_margin != 0 not supported in FP8 varlen high-precision API"
        )

    return _FlashAttnVarlenFP8Wrapper.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        attention_chunk,
        softcap,
        deterministic,
        sm_margin,
    )

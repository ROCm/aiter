# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
from typing import Optional, Tuple
import torch

from aiter.ops.triton._triton_kernels.flash_attn_triton_amd import flash_attn_3

FP8_DEQUANTIZED_BACKWARD = False


def set_fp8_dequantized_backward(value: bool):
    """Enable storing dequantized FP8 tensors (q/k/v * descale) in ctx instead of raw FP8.

    This is a debug / research aid; no gradients are produced for FP8 yet.
    """
    global FP8_DEQUANTIZED_BACKWARD
    FP8_DEQUANTIZED_BACKWARD = bool(value)


def _cast_to_fp8(
    x: torch.Tensor,
    fp8_dtype,
    layout,
    clamp_val=1e-9,
    group_size: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a tensor to FP8 format, returning an FP8 tensor and a descale factor.
    Args:
        - x (torch.Tensor): shape [batch, seq_len, heads, dim]
    Returns:
        - x_fp8 (torch.Tensor): FP8 tensor with the same shape as x
        - descale_factor (torch.Tensor): tensor of shape [batch, heads]
    """
    if len(x.shape) != 4:
        raise ValueError(
            f"'bshd' tensor should have shape [batch, seqlen, heads, dim], got {x.shape}"
        )
    batch, seqlen, nheads, dim = x.shape
    if group_size is None:
        # Standard per-head
        x_abs_max = x.abs().amax(dim=(1, 3))  # (batch, heads)
        x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))
        fp8_max = torch.finfo(fp8_dtype).max
        scale = (fp8_max / x_abs_max).view(batch, 1, nheads, 1)
        x_scaled = x * scale
        x_fp8 = x_scaled.to(fp8_dtype)
        # Preserve intent of requires_grad for downstream API expectations.
        # This does NOT create a differentiable path back to x; gradients stop at the cast.
        if x.requires_grad:
            x_fp8.requires_grad_(True)
        descale_factor = x_abs_max / fp8_max
        return x_fp8, descale_factor
    # Grouped path
    if nheads % group_size != 0:
        raise ValueError(
            f"group_size {group_size} must divide number of heads {nheads} in _cast_to_fp8"
        )
    ngroups = nheads // group_size
    # reshape to (B,S,ngroups,group_size,D)
    xg = x.view(batch, seqlen, ngroups, group_size, dim)
    x_abs_max_group = xg.abs().amax(dim=(1, 3, 4))  # (B, ngroups)
    x_abs_max_group = torch.maximum(x_abs_max_group, x.new_tensor(clamp_val))
    fp8_max = torch.finfo(fp8_dtype).max
    scale_group = (fp8_max / x_abs_max_group).view(batch, 1, ngroups, 1, 1)
    x_scaled = xg * scale_group
    x_fp8 = x_scaled.to(fp8_dtype).view(batch, seqlen, nheads, dim)
    if x.requires_grad:
        x_fp8.requires_grad_(True)
    descale_factor = x_abs_max_group / fp8_max  # (B, ngroups)
    return x_fp8, descale_factor


def _cast_varlen_to_fp8(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    cu_seqlens,
    clamp_val: float = 1e-9,
    group_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a tensor of sequences with variable seq_len into fp8.
    Args:
        - x (torch.Tensor): shape [total_seq_len, heads, dim]
    Returns:
        - x_fp8 (torch.Tensor): shape [total_seq_len, heads, dim]
        - descale_factors (torch.Tensor): shape [batch, heads]
    """
    # validate tensor shape
    if len(x.shape) != 3:
        raise ValueError(
            f"tensor should have shape [total_seqlen, heads, dim], got {x.shape}"
        )
    num_heads = x.shape[1]

    # Get batch size from cu_seqlens
    batch = cu_seqlens.shape[0] - 1
    fp8_max = torch.finfo(fp8_dtype).max

    # Compute scale and descale factors per sequence
    x_fp8 = torch.zeros_like(x, dtype=fp8_dtype)
    if group_size is not None and num_heads % group_size != 0:
        raise ValueError(
            f"group_size {group_size} must divide number of heads {num_heads} in _cast_varlen_to_fp8"
        )
    out_heads = num_heads if group_size is None else num_heads // group_size
    descale_factors = torch.zeros(
        (batch, out_heads), device=x.device, dtype=torch.float32
    )

    for i in range(batch):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        x_slice = x[start:end]  # Slice for current sequence

        if group_size is None:
            x_abs_max = x_slice.abs().amax(dim=(0, 2))  # (heads)
            x_abs_max = torch.maximum(x_abs_max, x.new_tensor(clamp_val))
            scale_i = fp8_max / x_abs_max
            descale_i = x_abs_max / fp8_max
            descale_factors[i, :] = descale_i
            scale_reshape = scale_i.view(1, num_heads, 1)
            x_fp8[start:end] = (x_slice * scale_reshape).to(fp8_dtype)
        else:
            ngroups = num_heads // group_size
            xg = x_slice.view(end - start, ngroups, group_size, x_slice.shape[2])
            x_abs_max_group = xg.abs().amax(dim=(0, 2, 3))  # (ngroups)
            x_abs_max_group = torch.maximum(x_abs_max_group, x.new_tensor(clamp_val))
            scale_group = fp8_max / x_abs_max_group
            descale_group = x_abs_max_group / fp8_max
            descale_factors[i, :] = descale_group
            scale_group_reshape = scale_group.view(1, ngroups, 1, 1)
            x_fp8[start:end] = (
                (xg * scale_group_reshape)
                .to(fp8_dtype)
                .view(end - start, num_heads, x_slice.shape[2])
            )

    if x.requires_grad:
        x_fp8.requires_grad_(True)
    return x_fp8, descale_factors


def _dequantize_bshd(x: torch.Tensor, descale: torch.Tensor | None) -> torch.Tensor:
    """Return a float32 dequantized (or widened) version of a BSHD activation.

    Steps:
      1. If `x` is FP8, cast to float32 first (ALWAYS) to avoid unsupported FP8 * fp16/fp32 promotion.
         If `x` is already fp16/bf16/fp32 we keep its current dtype (unless scaling forces fp32).
      2. If `descale` is None, return the widened tensor (no scaling).
      3. If `descale` provided, support per-head or grouped scaling shapes:
            (B, H)  -> per-head scaling
            (B, G) with H % G == 0 -> grouped scaling expanded over heads
         Any mismatch now RAISES a ValueError (previous behavior silently continued).

    Error conditions raised when descale is provided:
      - x.dim() != 4 or descale.dim() != 2
      - descale.shape[0] != B
      - head/group dimension neither equals H nor divides H

    Returns: float32 tensor (widened and optionally scaled) when successful.
    """
    is_fp8 = x.dtype in (
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    )
    if not is_fp8:
        raise TypeError(
            f"_dequantize_bshd expects an FP8 tensor; got {x.dtype}. "
            "This helper should only be invoked in the FP8 saving path."
        )
    # Always widen FP8 to float32 up front.
    x_fp = x.float()
    if descale is None:
        return x_fp
    # Ensure descale is float32 for stable math.
    descale_fp = descale.float()
    if x_fp.dim() != 4:
        raise ValueError(
            f"_dequantize_bshd expected x to have 4 dims (B,S,H,D); got shape {tuple(x_fp.shape)}"
        )
    if descale_fp.dim() != 2:
        raise ValueError(
            f"_dequantize_bshd expected descale to have 2 dims (B,H or B,G); got shape {tuple(descale_fp.shape)}"
        )
    B, S, H, D = x_fp.shape
    if descale_fp.shape[0] != B:
        raise ValueError(
            f"Batch size mismatch: x has B={B} but descale first dim={descale_fp.shape[0]}"
        )
    head_or_groups = descale_fp.shape[1]
    if head_or_groups == H:
        return x_fp * descale_fp.view(B, 1, H, 1)
    if H % head_or_groups == 0:  # grouped scaling
        group_size = H // head_or_groups
        expanded = descale_fp.unsqueeze(-1).repeat(1, 1, group_size).view(B, H)
        return x_fp * expanded.view(B, 1, H, 1)
    raise ValueError(
        "Incompatible descale shape: second dim neither equals number of heads nor divides it "
        f"(H={H}, descale second dim={head_or_groups})."
    )


def _dequantize_varlen_thd(
    x: torch.Tensor,
    descale: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> torch.Tensor:
    """Dequantize (or just widen) a concatenated varlen tensor of shape [T, H, D].

    This mirrors `_dequantize_bshd` semantics but for variable-length packed sequences.

    Arguments:
      x: FP8 (required) tensor with shape [total_tokens, heads, dim]. Always widened to fp32.
      descale: Optional (B,H) or (B,G) scaling factors (per-head or grouped). If None, we
               simply return widened fp32 tensor.
      cu_seqlens: Cumulative sequence lengths (int32/int64) of shape [B+1]; required when
                  descale is provided so we can map tokens -> batch rows of descale.

    Behavior:
      * Always widens FP8 to float32 first.
      * If `descale` is None returns widened tensor.
      * Validates tensor ranks & basic shape consistency; raises ValueError on mismatch.
      * Supports grouped scaling when H % G == 0.
    """
    is_fp8 = x.dtype in (
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    )
    if not is_fp8:
        raise TypeError(
            f"_dequantize_varlen_thd expects an FP8 tensor; got {x.dtype}. "
            "This helper should only be used for FP8 debug save path."
        )
    x_fp = x.float()  # widen first
    if descale is None:
        return x_fp
    if x_fp.dim() != 3:
        raise ValueError(
            f"_dequantize_varlen_thd expected x to have 3 dims (T,H,D); got shape {tuple(x_fp.shape)}"
        )
    if cu_seqlens is None:
        raise ValueError(
            "cu_seqlens must be provided when descale is not None for varlen dequantization"
        )
    if descale.dim() != 2:
        raise ValueError(
            f"_dequantize_varlen_thd expected descale to have 2 dims (B,H or B,G); got shape {tuple(descale.shape)}"
        )
    T, H, D = x_fp.shape
    B = cu_seqlens.shape[0] - 1
    if descale.shape[0] != B:
        raise ValueError(
            f"Batch mismatch: descale batch {descale.shape[0]} vs cu_seqlens implies B={B}"
        )
    head_or_groups = descale.shape[1]
    if head_or_groups != H and (H % head_or_groups) != 0:
        raise ValueError(
            "Incompatible descale second dim: neither equals number of heads nor divides it "
            f"(H={H}, descale second dim={head_or_groups})."
        )
    out = x_fp
    grouped = head_or_groups != H
    group_size = H // head_or_groups if grouped else 1
    # Iterate sequences to broadcast correct descales (cost acceptable for debug path)
    for b in range(B):
        start = int(cu_seqlens[b].item())
        end = int(cu_seqlens[b + 1].item())
        if start == end:
            continue  # empty sequence
        if not grouped:
            out[start:end] *= descale[b].view(1, H, 1)
        else:
            expanded = (
                descale[b]
                .unsqueeze(-1)
                .repeat(1, group_size)  # (G, group_size)
                .view(H)
            )
            out[start:end] *= expanded.view(1, H, 1)
    return out


class _FlashAttnV3Func(torch.autograd.Function):  # Backend (native) path
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

        if FP8_DEQUANTIZED_BACKWARD and q.dtype in (
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            q_save = _dequantize_bshd(q, q_descale)
            k_save = _dequantize_bshd(k, k_descale)
            v_save = _dequantize_bshd(v, v_descale)
        else:
            q_save, k_save, v_save = q, k, v

        ctx.save_for_backward(q_save, k_save, v_save, out, softmax_lse)
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        return out

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        q, k, v, out, softmax_lse = ctx.saved_tensors
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
    softmax_scale: float | None = None,
    causal: bool = False,
    qv: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
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

        if FP8_DEQUANTIZED_BACKWARD and q.dtype in (
            torch.float8_e5m2,
            torch.float8_e5m2fnuz,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            q_save = _dequantize_varlen_thd(q, q_descale, cu_seqlens_q)
            k_save = _dequantize_varlen_thd(k, k_descale, cu_seqlens_k)
            v_save = _dequantize_varlen_thd(v, v_descale, cu_seqlens_k)
        else:
            q_save, k_save, v_save = q, k, v
        ctx.save_for_backward(q_save, k_save, v_save, out, softmax_lse)
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
        q, k, v, out, softmax_lse = ctx.saved_tensors
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
    softmax_scale: float | None = None,
    causal: bool = False,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    window_size: tuple[int, int] = (-1, -1),
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
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    qv: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | int | None = None,
    softmax_scale: float | None = None,
    causal: bool = True,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int = 0,
    softcap: float = 0.0,
    num_splits: int = 0,
    pack_gqa: bool | None = None,
    sm_margin: int = 0,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    return_softmax_lse: bool = False,
    page_table: torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    rotary_seqlens: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k_new: torch.Tensor | None = None,
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

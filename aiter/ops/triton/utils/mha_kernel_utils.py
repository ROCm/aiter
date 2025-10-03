# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from typing import Optional, Tuple
import torch


@triton.jit
def _compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x))  # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x


def _quantize_bshd(
    x: torch.Tensor,
    fp8_dtype,
    clamp_val=1e-9,
    group_size: Optional[int] = None,
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


def _quantize_thd(
    x: torch.Tensor,
    fp8_dtype: torch.dtype,
    cu_seqlens,
    clamp_val: float = 1e-9,
    group_size: Optional[int] = None,
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


def _dequantize_bshd(x: torch.Tensor, descale: Optional[torch.Tensor]) -> torch.Tensor:
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

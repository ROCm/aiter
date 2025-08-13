from __future__ import annotations

from typing import Optional, Union, Tuple

import torch
import triton
import triton.language as tl

__all__ = ["apply_rotary"]


@triton.jit
def _rotary_kernel(
    OUT,
    X,
    COS,
    SIN,
    CU_SEQLENS,
    SEQLEN_OFFSETS,
    seqlen,
    nheads,
    seqlen_ro,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_nheads,
    stride_out_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    ROTARY_DIM: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF = ROTARY_DIM // 2
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    if not IS_VARLEN:
        X = X + pid_batch * stride_x_batch
        OUT = OUT + pid_batch * stride_out_batch
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        X = X + start_idx * stride_x_seqlen
        OUT = OUT + start_idx * stride_out_seqlen

    if pid_m * BLOCK_M >= seqlen:
        return

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    rk_half = tl.arange(0, BLOCK_K // 2)
    COS = COS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    cos = tl.load(COS, mask=mask_cs, other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=mask_cs, other=0.0).to(tl.float32)
    if CONJUGATE:
        sin = -sin

    if not INTERLEAVED:
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk_half[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk_half[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk_half[None, None, :] < ROTARY_DIM_HALF)
        )
        x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(X + ROTARY_DIM_HALF * stride_x_headdim, mask=mask, other=0.0).to(
            tl.float32
        )
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT, o0, mask=mask)
        tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        X = X + (
            rh[:, None, None] * stride_x_nheads
            + rm[None, :, None] * stride_x_seqlen
            + rk[None, None, :] * stride_x_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk[None, None, :] < ROTARY_DIM)
        )
        x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT, o, mask=mask)


def _apply_rotary_kernel(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert (
            max_seqlen is not None
        ), "If cu_seqlens is passed, max_seqlen must also be provided"
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim_half = cos.shape
    assert sin.shape == cos.shape
    rotary_dim = 2 * rotary_dim_half
    assert rotary_dim <= headdim
    assert headdim <= 256
    assert seqlen_ro >= seqlen

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in (torch.int32, torch.int64)
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    out = torch.empty_like(x) if not inplace else x
    if rotary_dim < headdim and not inplace:
        out[..., rotary_dim:].copy_(x[..., rotary_dim:])

    # Block heuristics
    BLOCK_M = 8 if rotary_dim <= 128 else 4
    grid = (
        triton.cdiv(nheads, 2),
        triton.cdiv(seqlen, BLOCK_M),
        batch,
    )

    # NOTE: We assume CUDA device indexing compatibility in upstream; adapt for ROCm by using device context.
    # For ROCm, torch.cuda.device works if HIP_VISIBLE_DEVICES mapping is set.
    with torch.cuda.device(x.device.index):  # Works for ROCm as alias
        torch.library.wrap_triton(_rotary_kernel)[grid](
            out,
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            seqlen,
            nheads,
            seqlen_ro,
            out.stride(0) if not is_varlen else 0,
            out.stride(-3),
            out.stride(-2),
            out.stride(-1),
            x.stride(0) if not is_varlen else 0,
            x.stride(-3),
            x.stride(-2),
            x.stride(-1),
            rotary_dim,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M=BLOCK_M,
            BLOCK_H=2,
        )
    return out


class _ApplyRotary(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        interleaved: bool,
        inplace: bool,
        seqlen_offsets: Union[int, torch.Tensor],
        cu_seqlens: Optional[torch.Tensor],
        max_seqlen: Optional[int],
    ):
        out = _apply_rotary_kernel(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
            conjugate=False,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do: torch.Tensor):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors
        dx = _apply_rotary_kernel(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """Public API: apply rotary embeddings to tensor x.

    Args:
        x: (B, S, H, D) if `cu_seqlens` is None else (total_S, H, D).
        cos, sin: (S_rotary, rotary_dim/2)
        interleaved: GPT-J style if True.
        inplace: modify x in place (saves memory if rotary_dim == D).
        seqlen_offsets: int or (B,) tensor of starting offsets per sequence (KV cache decode).
        cu_seqlens: (B+1,) tensor enabling varlen mode.
        max_seqlen: required when `cu_seqlens` is provided.
    """
    # FP8 path: upcast to bfloat16 (preferred) or float16 for rotary math to avoid excessive error
    original_dtype = x.dtype
    is_fp8_input = original_dtype == getattr(torch, "float8_e4m3fn", None)
    if is_fp8_input:
        # Choose bf16 if available in cos.dtype path; otherwise fallback to float16
        target_dtype = (
            torch.bfloat16
            if cos.dtype == torch.bfloat16 or torch.cuda.is_bf16_supported()
            else torch.float16
        )
        # Upcast x, cos, sin for computation (without modifying originals in-place)
        x_up = x.to(target_dtype)
        cos_up = cos.to(target_dtype) if cos.dtype != target_dtype else cos
        sin_up = sin.to(target_dtype) if sin.dtype != target_dtype else sin
        out_up = _ApplyRotary.apply(
            x_up,
            cos_up,
            sin_up,
            interleaved,
            False,
            seqlen_offsets,
            cu_seqlens,
            max_seqlen,
        )
        # Cast result back to original fp8 dtype
        if inplace:
            x.copy_(out_up.to(original_dtype))
            return x
        return out_up.to(original_dtype)
    else:
        return _ApplyRotary.apply(
            x, cos, sin, interleaved, inplace, seqlen_offsets, cu_seqlens, max_seqlen
        )


def apply_rotary(
    q: torch.Tensor,
    k_new: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    causal: bool,
    local: bool,
    interleaved: bool = False,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """High-level rotary application used by AMD prefill & decode paths.

    Policy (matches test reference & legacy semantics):
      - If causal OR local attention ⇒ apply rotary directly on (B, S, H, D).
      - Else (non-causal global) ⇒ flatten heads into sequence: (B, 1, S*H, D),
        apply rotary once, then unflatten back.
      - k_new (incremental KV slice) is always rotated directly when provided.

    Args:
        q: (B, S, H, D)
        k_new: Optional (B, S_k, H_k, D)
        cos, sin: rotary caches (S_rotary, rotary_dim/2)
        causal: causal attention flag
        local: sliding-window / local attention flag (pre-computed outside)
        interleaved: GPT-J style rotary layout
        seqlen_offsets: int or (B,) tensor of per-sequence start offsets
    Returns:
        (q_rot, k_new_rot)
    """
    assert q.ndim == 4, f"Expected q shape (B,S,H,D), got {q.shape}"
    B, S, H, D = q.shape
    use_flatten = (not causal) and (not local)

    if use_flatten:
        # Flatten (S,H) -> (S*H) with an added singleton dim to preserve expected 4D shape.
        q_flat = q.reshape(B, S * H, D).unsqueeze(1)  # (B, 1, S*H, D)
        q_flat = apply_rotary_emb(
            q_flat,
            cos,
            sin,
            interleaved=interleaved,
            seqlen_offsets=seqlen_offsets,
        )
        # Restore shape back to (B, S, H, D)
        q = q_flat.view(B, 1, S * H, D).reshape(B, S, H, D)
    else:
        q = apply_rotary_emb(
            q,
            cos,
            sin,
            interleaved=interleaved,
            seqlen_offsets=seqlen_offsets,
        )

    if k_new is not None:
        k_new = apply_rotary_emb(
            k_new,
            cos,
            sin,
            interleaved=interleaved,
            seqlen_offsets=seqlen_offsets,
        )
    return q, k_new

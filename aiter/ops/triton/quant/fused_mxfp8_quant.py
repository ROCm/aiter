# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.quant.fused_mxfp8_quant import (
    _fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_aligned_kernel,
    _fused_deepseek_v4_mxfp8_quant_q_pack_kernel,
    _fused_dual_rmsnorm_mxfp8_quant_kernel,
    _fused_flatten_mxfp8_quant_kernel,
    _fused_rms_mxfp8_kernel,
)

__all__ = [
    "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_aligned",
    "fused_deepseek_v4_mxfp8_quant_q_pack",
    "fused_dual_rmsnorm_mxfp8_quant",
    "fused_flatten_mxfp8_quant",
    "fused_rms_mxfp8_quant",
]

_QUANT_BLOCK_SIZE = 32

# DSv4 a8w8 Q operand: 448 NoPE (UE8M0 fp8, 64-element groups) + 64 RoPE
# (bf16, never quantized). The decode kernel reads the scale bytes inline,
# each duplicated, because its scaled-MMA blocks are half the quant group.
_V4_DIM_NOPE = 448
_V4_DIM_ROPE = 64
_V4_DIM_QK = _V4_DIM_NOPE + _V4_DIM_ROPE
_FP8_GROUP_SIZE = 64
_V4_NUM_TILES = _V4_DIM_NOPE // _FP8_GROUP_SIZE




def fused_rms_mxfp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    y: torch.Tensor | None = None,
    scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused RMSNorm + MXFP8 (1x32 e8m0) quant in a single Triton launch.

    Args:
        x: (M, K) bf16 or fp16.
        weight: (K,) bf16 or fp16 RMSNorm weight.
        eps: RMSNorm epsilon.
        y: optional preallocated FP8 e4m3fn output (M, K).
        scale: optional preallocated uint8 e8m0 output (M, K // 32).

    Returns:
        y (M, K) fp8 e4m3fn, scale (M, K // 32) uint8.
    """
    assert x.dim() == 2, f"x must be 2D, got {x.dim()}"
    M, K = x.shape
    assert weight.shape == (K,), f"weight shape {weight.shape} != ({K},)"
    assert K % _QUANT_BLOCK_SIZE == 0
    Ns = K // _QUANT_BLOCK_SIZE
    BLOCK_SIZE_K = triton.next_power_of_2(K)

    if y is None:
        y = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=x.device)
    if scale is None:
        scale = torch.empty((M, Ns), dtype=torch.uint8, device=x.device)

    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _fused_rms_mxfp8_kernel[grid](
        x,
        weight,
        y,
        scale,
        M,
        K,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        scale.stride(0),
        scale.stride(1),
        eps,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )
    return y, scale


def fused_dual_rmsnorm_mxfp8_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps_q: float,
    eps_k: float | None = None,
    yq: torch.Tensor | None = None,
    sq: torch.Tensor | None = None,
    yk: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused dual RMSNorm in a single Triton launch.

    - Q side: RMSNorm(q, q_weight, eps_q) -> MXFP8 (FP8 e4m3fn + uint8 e8m0 1x32).
    - K side: RMSNorm(k, k_weight, eps_k) -> bf16.

    Replaces the CK `fused_qk_rmsnorm_group_quant` kernel for the MXFP8 GEMM
    path on V4 (Task #77): one launch instead of two (fused_rms_mxfp8_quant +
    rmsnorm2d_fwd_), eliminating the ~6us/layer launch-overhead regression.

    Args:
        q: (M, KQ) bf16 or fp16 — Q-side input (e.g. q_lora).
        k: (M, KK) bf16 or fp16 — K-side input (e.g. kv_pre).
        q_weight: (KQ,) bf16 or fp16 — Q RMSNorm weight.
        k_weight: (KK,) bf16 or fp16 — K RMSNorm weight.
        eps_q: Q RMSNorm epsilon.
        eps_k: K RMSNorm epsilon; defaults to eps_q.
        yq, sq, yk: optional pre-allocated outputs.

    Returns:
        yq (M, KQ) fp8 e4m3fn, sq (M, KQ // 32) uint8 e8m0, yk (M, KK) bf16.
    """
    assert q.dim() == 2, f"q must be 2D, got {q.dim()}"
    assert k.dim() == 2, f"k must be 2D, got {k.dim()}"
    M, KQ = q.shape
    Mk, KK = k.shape
    assert M == Mk, f"q rows {M} != k rows {Mk}"
    assert q_weight.shape == (KQ,), f"q_weight shape {q_weight.shape} != ({KQ},)"
    assert k_weight.shape == (KK,), f"k_weight shape {k_weight.shape} != ({KK},)"
    assert (
        KQ % _QUANT_BLOCK_SIZE == 0
    ), f"KQ={KQ} must be a multiple of {_QUANT_BLOCK_SIZE}"
    if eps_k is None:
        eps_k = eps_q

    Ns = KQ // _QUANT_BLOCK_SIZE
    BLOCK_SIZE_KQ = triton.next_power_of_2(KQ)
    BLOCK_SIZE_KK = triton.next_power_of_2(KK)

    if yq is None:
        yq = torch.empty((M, KQ), dtype=torch.float8_e4m3fn, device=q.device)
    if sq is None:
        sq = torch.empty((M, Ns), dtype=torch.uint8, device=q.device)
    if yk is None:
        yk = torch.empty((M, KK), dtype=k.dtype, device=k.device)

    NUM_PRGMS = M
    grid = (NUM_PRGMS,)

    _fused_dual_rmsnorm_mxfp8_quant_kernel[grid](
        q,
        k,
        q_weight,
        k_weight,
        yq,
        sq,
        yk,
        M,
        KQ,
        KK,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        yq.stride(0),
        yq.stride(1),
        sq.stride(0),
        sq.stride(1),
        yk.stride(0),
        yk.stride(1),
        eps_q,
        eps_k,
        BLOCK_SIZE_KQ=BLOCK_SIZE_KQ,
        BLOCK_SIZE_KK=BLOCK_SIZE_KK,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
        NUM_PRGMS=NUM_PRGMS,
    )
    return yq, sq, yk


def fused_flatten_mxfp8_quant(
    x: torch.Tensor,
    quant_dtype: torch.dtype = torch.float8_e4m3fn,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Flatten the last two dimensions of x and apply per-1x32 MXFP8 quant along
    the flattened axis (FP8 e4m3 values + uint8 e8m0 scales).

    Equivalent in shape to `fused_flatten_fp8_group_quant` but emits MXFP8
    1x32 (e8m0) scales using the same recipe as `dynamic_mxfp8_quant`.

    Args:
        x: Input tensor of shape (M, N1, N2). N2 must be a multiple of 32.
        quant_dtype: FP8 dtype to cast quantized values to (defaults to
            torch.float8_e4m3fn).

    Returns:
        Tuple of:
            out: FP8 tensor of shape (M, N1 * N2).
            out_scales: e8m0 (uint8) scale tensor of shape
                (M, (N1 * N2) // 32).
    """
    assert x.dim() == 3, f"x must be 3D, got {x.dim()}"
    M, N1, N2 = x.shape
    assert (
        N2 % _QUANT_BLOCK_SIZE == 0
    ), f"N2={N2} must be a multiple of {_QUANT_BLOCK_SIZE}"

    BLOCK_SIZE_N2 = max(triton.next_power_of_2(N2), _QUANT_BLOCK_SIZE)
    N = N1 * N2

    out = torch.empty((M, N), dtype=quant_dtype, device=x.device)
    out_scales = torch.empty(
        (M, N // _QUANT_BLOCK_SIZE), dtype=torch.uint8, device=x.device
    )

    grid = (M, N1)
    _fused_flatten_mxfp8_quant_kernel[grid](
        x,
        out,
        out_scales,
        *x.stride(),
        *out.stride(),
        *out_scales.stride(),
        N2,
        BLOCK_SIZE_N2=BLOCK_SIZE_N2,
        QUANT_BLOCK_SIZE=_QUANT_BLOCK_SIZE,
    )

    return out, out_scales


def fused_deepseek_v4_mxfp8_quant_q_pack(q: torch.Tensor):
    """``[..., 512]`` bf16 Q -> ``(packed [..., 512] fp8, rope [..., 64] bf16)``.

    The Q form ``_pa_decode_sparse_v4`` reads on gfx1250 (a8w8). RoPE is never
    quantized, which is why this returns a pair.

    A zeroed (padding) head packs to zero data with the 1e-4 floor's exponent in
    its scale bytes -- harmless, since a zero mantissa dequantizes to zero
    whatever the exponent says. What must never appear is an UNWRITTEN scale
    byte: 0xFF is E8M0 NaN and a NaN scale poisons a whole score row through
    ``0 * NaN``.
    """
    if q.shape[-1] != _V4_DIM_QK:
        raise RuntimeError(
            f"q last dim must be {_V4_DIM_QK} (448 NoPE + 64 RoPE), got "
            f"{q.shape[-1]}"
        )
    q = q.contiguous()
    lead = q.shape[:-1]
    rows = 1
    for d in lead:
        rows *= d
    packed = torch.empty((*lead, _V4_DIM_QK), dtype=torch.uint8, device=q.device)
    rope = torch.empty((*lead, _V4_DIM_ROPE), dtype=q.dtype, device=q.device)
    _fused_deepseek_v4_mxfp8_quant_q_pack_kernel[(rows,)](
        q, packed, rope, float(torch.finfo(torch.float8_e4m3fn).max),
        _V4_DIM_NOPE, _V4_DIM_ROPE, _V4_DIM_QK, _FP8_GROUP_SIZE, _V4_NUM_TILES,
        num_warps=4,
    )
    return packed.view(torch.float8_e4m3fn), rope


# The ALIGNED paged KV record. 640 = 5 * 128, so every token starts on a
# 128-byte boundary and TDM reads it on the direct global->L2->LDS path.
_V4_REC_ALIGNED = 640
_V4_SC_IN_REC = _V4_DIM_NOPE  # 448
_V4_ROPE_IN_REC = 512  # the 2buff row is 512 B; RoPE follows it


def fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_aligned(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache_block_size: int,
    eps: float,
    padded_heads: int,
    apply_q_norm: bool = True,
    pack_q: bool = True,
    use_fnuz: bool = False,
):
    """The whole DSv4 decode producer in one launch.

    Q gets RMSNorm (no weight) then GPT-J RoPE, and -- when ``pack_q`` -- the
    UE8M0 fp8 pack off the same registers, so packing costs no second pass.
    KV gets RoPE, UE8M0 quant and an insert into the aligned paged cache.

    Args:
        q: ``[T, H, 512]`` bf16.
        kv: ``[T, 512]`` bf16, the compressed KV for these tokens.
        kv_cache: ``[nb, block, 640]`` uint8, or any view whose dim-0 stride is
            the block stride -- a per-layer view of a pooled allocation is one.
        kv_slot_mapping: ``[T]``, ``-1`` for a token with no cache slot.
        positions: ``[T]`` RoPE positions, shared by Q and KV.
        cos_sin_cache: ``[max_pos, 64]`` fp32, laid out cos || sin.
        padded_heads: the head count the decode kernel expects; slots at or
            past ``H`` are zero-filled.

    Returns:
        ``q_bf16`` when ``pack_q`` is False, else
        ``(q_bf16, q_packed, q_rope)``.
    """
    t, h, dim = q.shape
    if dim != _V4_DIM_QK:
        raise RuntimeError(f"q must be [T, H, {_V4_DIM_QK}], got {tuple(q.shape)}")
    if padded_heads < h:
        raise RuntimeError(f"padded_heads {padded_heads} < H {h}")
    if kv.dim() != 2 or kv.shape != (t, _V4_DIM_QK):
        raise RuntimeError(
            f"kv must be [{t}, {_V4_DIM_QK}], got {tuple(kv.shape)}"
        )
    if kv_cache.shape[-1] != _V4_REC_ALIGNED:
        raise RuntimeError(
            f"kv_cache records must be {_V4_REC_ALIGNED} B, got "
            f"{kv_cache.shape[-1]}"
        )
    q = q.contiguous()
    kv = kv.contiguous()

    q_out = torch.empty(t, padded_heads, dim, dtype=q.dtype, device=q.device)
    if pack_q:
        q_packed = torch.empty(
            t, padded_heads, dim, dtype=torch.uint8, device=q.device
        )
        q_rope = torch.empty(
            t, padded_heads, _V4_DIM_ROPE, dtype=q.dtype, device=q.device
        )
    else:
        # unused under PACK_Q=False, but the launch still has to typecheck
        q_packed = q_rope = q_out

    fp8_max = 224.0 if use_fnuz else float(torch.finfo(torch.float8_e4m3fn).max)
    # One extra slot along dim 1 carries the KV row for the token.
    _fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_aligned_kernel[
        (t, padded_heads + 1)
    ](
        q,
        q_out,
        q_packed,
        q_rope,
        kv,
        kv_cache,
        kv_slot_mapping,
        positions,
        cos_sin_cache,
        kv_cache.stride(0),
        kv_cache_block_size,
        float(eps),
        fp8_max,
        h,
        padded_heads,
        _V4_DIM_QK,
        _V4_DIM_NOPE,
        _V4_DIM_ROPE,
        _FP8_GROUP_SIZE,
        _V4_NUM_TILES,
        _V4_REC_ALIGNED,
        _V4_SC_IN_REC,
        _V4_ROPE_IN_REC,
        bool(apply_q_norm),
        bool(pack_q),
        bool(use_fnuz),
        num_warps=4,
    )
    if pack_q:
        return q_out, q_packed.view(torch.float8_e4m3fn), q_rope
    return q_out

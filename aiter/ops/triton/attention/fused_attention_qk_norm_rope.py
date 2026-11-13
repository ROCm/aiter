# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused Qwen3-Next attention prologue (gfx950): Q/K norm + partial RoPE.

A single launch replaces the Qwen3-Next full-attention prologue: the per-head
``[Q||gate]`` split, Q/K GemmaRMSNorm (``x*(1+w)``), and partial NeoX RoPE
(rotary_dim 64). The op reads the packed ``(rows, 2560)`` projection (4 Q heads
+ 1 KV head, head_dim 256) and returns ``(query, gate, key, value)``. The
KV-cache write is left to the attention backend (RadixAttention), so this op
does not touch the cache.

The norm+rope+gate math is not reimplemented here. This wrapper launches the
same ``persistent_qk_norm_rope_gate_token_amax_kernel`` that
:func:`aiter.ops.triton.rope.fused_qk_norm_rope_gate_fp8_quant` uses, and drops
only the FP8 quantization tail that Qwen3-Next does not need (it consumes BF16
Q/K). That kernel is token-persistent and covers every decode and prefill batch
size in one launch, so there is no per-size schedule to pick. The shared kernel
also writes a per-token amax for its FP8 path; this BF16 prologue sends it to
throwaway scratch.

Support is reported by :func:`fused_attention_qk_norm_rope_supported` rather
than asserted, so callers fall back to the native (unfused) Q/K-norm + RoPE
path instead of crashing.
"""

import functools

import torch

_PACKED = 2560
_HEAD_DIM = 256
_KV_HEADS = 1
_ROTARY_DIM = 64


@functools.lru_cache(maxsize=1)
def _arch_supported() -> tuple[bool, str]:
    """Probe the GPU arch lazily and fail closed if it cannot be read."""
    try:
        from aiter.ops.triton.utils._triton.arch_info import get_arch

        arch = get_arch()
    except Exception as exc:  # noqa: BLE001 - any probe failure => unsupported
        return False, f"could not probe GPU arch ({exc})"
    if arch != "gfx950":
        return False, f"gfx950 only, got {arch}"
    return True, ""


def fused_attention_qk_norm_rope_supported(
    projected_qkv_gate: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[bool, str]:
    """Report whether this prologue call is covered, and if not, why.

    Returns ``(True, "")`` or ``(False, reason)``; the reason is meant to be
    logged once by the caller on its native fallback path.
    """
    arch_ok, arch_reason = _arch_supported()
    if not arch_ok:
        return False, arch_reason

    if (
        not isinstance(projected_qkv_gate, torch.Tensor)
        or not projected_qkv_gate.is_cuda
        or projected_qkv_gate.ndim != 2
    ):
        return False, "projected_qkv_gate must be a 2-D CUDA matrix"
    rows, packed = projected_qkv_gate.shape
    if packed != _PACKED:
        return (
            False,
            f"packed width must be {_PACKED} (4Q+1KV, head_dim 256), got {packed}",
        )
    if rows < 1:
        return False, f"rows={rows} must be positive"
    device = projected_qkv_gate.device
    if projected_qkv_gate.dtype != torch.bfloat16:
        return False, "projected_qkv_gate must be bf16"
    # The shared kernel reads Q/K by row stride, so the packed matrix must be
    # row-major contiguous.
    if not projected_qkv_gate.is_contiguous():
        return False, "projected_qkv_gate must be contiguous"

    for name, t, shape in (
        ("q_norm_weight", q_norm_weight, (_HEAD_DIM,)),
        ("k_norm_weight", k_norm_weight, (_HEAD_DIM,)),
    ):
        if not isinstance(t, torch.Tensor) or t.device != device:
            return False, f"{name} must be a CUDA tensor on {device}"
        if tuple(t.shape) != shape:
            return False, f"{name} must be {shape}, got {tuple(t.shape)}"
        if not t.is_contiguous():
            return False, f"{name} must be contiguous"

    # The kernel reads the table at a fixed stride of 64 (32 cos then 32 sin
    # per position), so it must be contiguous with exactly rotary_dim columns.
    if (
        not isinstance(cos_sin_cache, torch.Tensor)
        or cos_sin_cache.device != device
        or cos_sin_cache.shape[-1] != _ROTARY_DIM
        or not cos_sin_cache.is_contiguous()
    ):
        return (
            False,
            (
                f"cos_sin_cache must be a contiguous table on {device} with "
                f"exactly {_ROTARY_DIM} columns"
            ),
        )
    if (
        not isinstance(positions, torch.Tensor)
        or positions.device != device
        or positions.ndim != 1
        or positions.shape[0] < rows
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        return (
            False,
            (
                f"positions must be a 1-D int32/int64 CUDA tensor on {device} "
                f"covering {rows} rows"
            ),
        )
    return True, ""


def fused_attention_qk_norm_rope(
    projected_qkv_gate: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    eps: float = 1.0e-6,
    rotary_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused Qwen3-Next attention prologue in a single launch.

    Args:
        projected_qkv_gate: ``(rows, 2560)`` bf16 packed projection. Per token,
            four Q slabs ``[Q(256)||gate(256)]`` then the KV slab
            ``[K(256)||V(256)]``.
        q_norm_weight, k_norm_weight: ``(256,)`` GemmaRMSNorm gains.
        positions: ``(rows,)`` int RoPE positions.
        cos_sin_cache: ``(max_pos, 64)`` table (32 cos then 32 sin per position).
        eps: RMSNorm epsilon. rotary_dim: must be 64.

    Returns ``(query, gate, key, value)``: ``query`` ``(rows,4,256)`` and
    ``key`` ``(rows,1,256)`` are normed+roped bf16; ``gate`` ``(rows,4*256)``
    and ``value`` ``(rows,1,256)`` are raw bf16 views of the projection. The
    caller writes K/V to the attention cache.

    Narrow by design; guard with
    :func:`fused_attention_qk_norm_rope_supported` and fall back to the native
    path when it returns ``False``.
    """
    ok, reason = fused_attention_qk_norm_rope_supported(
        projected_qkv_gate,
        q_norm_weight,
        k_norm_weight,
        positions,
        cos_sin_cache,
    )
    if not ok:
        raise ValueError(
            f"fused_attention_qk_norm_rope does not support this call: {reason}"
        )
    if rotary_dim != _ROTARY_DIM:
        raise ValueError(f"rotary_dim must be {_ROTARY_DIM}, got {rotary_dim}")

    import triton
    import triton.language as tl

    from aiter.ops.triton._triton_kernels.rope.fused_qk_norm_rope_gate_fp8_quant import (
        persistent_qk_norm_rope_gate_token_amax_kernel,
    )
    from aiter.ops.triton.rope.fused_qk_norm_rope_gate_fp8_quant import (
        QK_TOKENS_PER_PROGRAM,
    )

    rows = projected_qkv_gate.shape[0]
    heads = (projected_qkv_gate.shape[1] - 512) // 512  # 4 Q heads
    device = projected_qkv_gate.device

    # Views into the packed projection: four [Q||gate] slabs then [K||V]. The
    # shared kernel reads Q/K by stride, so these keep the 2560 row stride.
    q_gate = projected_qkv_gate[:, : heads * 512]
    key_in = projected_qkv_gate[:, heads * 512 : heads * 512 + _HEAD_DIM]
    value = projected_qkv_gate[:, heads * 512 + _HEAD_DIM :].view(rows, 1, _HEAD_DIM)

    query = torch.empty((rows, heads * _HEAD_DIM), dtype=torch.bfloat16, device=device)
    key = torch.empty((rows, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    gate = torch.empty((rows, heads * _HEAD_DIM), dtype=torch.bfloat16, device=device)
    # The shared kernel also emits a per-token amax for its FP8 path; this BF16
    # prologue does not quantize, so route it to throwaway scratch.
    q_token_amax = torch.empty((rows, heads), dtype=torch.float32, device=device)
    k_token_amax = torch.empty((rows, _KV_HEADS), dtype=torch.float32, device=device)

    head_block = triton.next_power_of_2(_HEAD_DIM)
    persistent_qk_norm_rope_gate_token_amax_kernel[
        (triton.cdiv(rows, QK_TOKENS_PER_PROGRAM), heads + _KV_HEADS)
    ](
        q_gate,
        key_in,
        query,
        key,
        gate,
        q_token_amax,
        k_token_amax,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        rows,
        q_gate.stride(0),
        key_in.stride(0),
        query.stride(0),
        key.stride(0),
        gate.stride(0),
        cos_sin_cache.stride(0),
        num_q_heads=heads,
        num_kv_heads=_KV_HEADS,
        head_dim=_HEAD_DIM,
        rotary_dim=_ROTARY_DIM,
        half_rotary=_ROTARY_DIM // 2,
        eps=eps,
        INPUT_DTYPE=tl.bfloat16,
        HEAD_BLOCK=head_block,
        TOKENS_PER_PROGRAM=QK_TOKENS_PER_PROGRAM,
        ADD_GEMMA_OFFSET=True,
        num_warps=max(1, head_block // 64),
        num_stages=2,
    )
    return query.view(rows, heads, _HEAD_DIM), gate, key.view(rows, 1, _HEAD_DIM), value

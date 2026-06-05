# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_reduce_qk_norm_rope_swa_write import (
    _fused_reduce_qk_norm_rope_swa_write_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def _pick_block_size_m(M: int, num_local_heads: int, num_splitk: int) -> int:
    """Pick BLOCK_SIZE_M for the fused reduce/norm/rope kernel.

    Each program loads ``[NUM_SPLITK, BLOCK_SIZE_M, HEAD_DIM]`` of fp32 q-tile,
    so register pressure scales with ``num_splitk * BLOCK_SIZE_M``. Wide head
    grids (large ``num_local_heads``) amortize per-program constant work
    (cos/sin offset prep, etc.) better with larger BM; small head grids need
    more M-tiles for occupancy, which favors smaller BM.
    """
    # Register-pressure cap from the splitk fp32 q-tile.
    if num_splitk >= 4:
        cap = 4
    elif num_splitk >= 2:
        cap = 8
    else:
        cap = 16

    if num_local_heads >= 64:
        target = 16
    elif num_local_heads >= 16:
        target = 8
    else:
        target = 4

    bm = min(target, cap)

    # Shrink to a power-of-two not exceeding M so tiny inputs don't pay tail
    # masking overhead for the whole tile.
    while bm > 1 and bm > M:
        bm //= 2

    num_warps = 4
    waves_per_eu = 1

    return bm, num_warps, waves_per_eu


def fused_reduce_qk_norm_rope_swa_write(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_norm_weight: Optional[torch.Tensor],
    kv_norm_weight: Optional[torch.Tensor],
    q_rms_eps: float,
    kv_rms_eps: float,
    rope_head_dim: int,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_out: Optional[torch.Tensor] = None,
    is_neox: bool = False,
    write_indices: Optional[torch.Tensor] = None,
    batch_id_per_token: Optional[torch.Tensor] = None,
    state_slot_mapping: Optional[torch.Tensor] = None,
    swa_kv: Optional[torch.Tensor] = None,
    win: int = 128,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused split-K reduce + per-head weighted RMSNorm + RoPE tail on Q,
    weighted RMSNorm + RoPE tail on KV (+ optional SWA).

    Replaces the (split-K reduce → q-norm → q-rope → kv-norm → kv-rope → swa-write)
    post-GEMM chain.

    Shapes:
        ``q``: ``[M, N]`` or ``[num_splitk, M, N]`` where ``N = num_local_heads * head_dim``.
            The 3D form lets the caller pass an unreduced split-K GEMM output; the kernel
            sums across ``num_splitk`` while loading.
        ``kv``: ``[M, head_dim]`` updated in-place (RMSNorm over head_dim + RoPE on tail).
        ``q_norm_weight``: ``[head_dim]`` (or None for weight-free RMSNorm).
        ``kv_norm_weight``: ``[head_dim]`` (or None for weight-free RMSNorm).
        ``cos_cache`` / ``sin_cache``: ``[max_position, 1, 1, rope_head_dim // 2]``.
        ``positions``: ``[M]`` int32/int64 indices into cos/sin rows.

    Optional SWA: pass ``swa_kv``, ``win``, ``write_indices`` (``[M]`` int32), ``batch_id_per_token``,
    ``state_slot_mapping`` — same semantics as ``state_writes.swa_write``.
    """
    assert q.is_cuda and kv.is_cuda
    head_dim = kv.shape[1]

    if q.dim() == 3:
        num_splitk, M, N = q.shape
        q_in_splitk_stride = q.stride(0)
        q_in_m_stride = q.stride(1)
        q_in_d_stride = q.stride(2)
    elif q.dim() == 2:
        M, N = q.shape
        num_splitk = 1
        q_in_splitk_stride = 0
        q_in_m_stride = q.stride(0)
        q_in_d_stride = q.stride(1)
    else:
        raise ValueError(f"q must be 2D or 3D, got shape {q.shape}")

    assert N % head_dim == 0, f"N={N} must be divisible by head_dim={head_dim}"
    num_local_heads = N // head_dim
    assert kv.shape[0] == M, f"kv {tuple(kv.shape)} vs M={M}"
    assert rope_head_dim <= head_dim
    assert cos_cache.shape[-1] == rope_head_dim // 2
    assert sin_cache.shape == cos_cache.shape

    if q_out is None:
        q_out = torch.empty(
            (M, num_local_heads, head_dim), dtype=dtype, device=q.device
        )
    else:
        assert q_out.shape == (M, num_local_heads, head_dim)

    cos_stride_t = cos_cache.stride(0)
    cos_stride_d = cos_cache.stride(-1)

    HAS_SWA = (
        swa_kv is not None
        and write_indices is not None
        and batch_id_per_token is not None
        and state_slot_mapping is not None
    )
    if HAS_SWA:
        assert (
            swa_kv.dim() == 3 and swa_kv.shape[1] == win and swa_kv.shape[2] == head_dim
        )
        assert write_indices.shape == (M,)
        assert batch_id_per_token.shape[0] >= M
        assert state_slot_mapping.dim() == 1

    _LOGGER.info(
        "FUSED_REDUCE_QK_NORM_ROPE_SWA_WRITE "
        f"M={M} num_splitk={num_splitk} heads={num_local_heads} "
        f"D={head_dim} rd={rope_head_dim} HAS_SWA={HAS_SWA}"
    )

    BLOCK_SIZE_M, num_warps, waves_per_eu = _pick_block_size_m(
        M, num_local_heads, num_splitk
    )
    grid = (triton.cdiv(M, BLOCK_SIZE_M), num_local_heads + 1)
    _fused_reduce_qk_norm_rope_swa_write_kernel[grid](
        q,
        q_out,
        kv,
        q_norm_weight,
        kv_norm_weight,
        positions,
        cos_cache,
        sin_cache,
        write_indices,
        batch_id_per_token,
        state_slot_mapping,
        swa_kv,
        M,
        q_in_splitk_stride,
        q_in_m_stride,
        q_in_d_stride,
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        kv.stride(0),
        kv.stride(1),
        cos_stride_t,
        cos_stride_d,
        swa_kv.stride(0) if HAS_SWA else 0,
        swa_kv.stride(1) if HAS_SWA else 0,
        win,
        q_rms_eps,
        kv_rms_eps,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_head_dim,
        NUM_LOCAL_HEADS=num_local_heads,
        NUM_SPLITK=num_splitk,
        HAS_SWA=HAS_SWA,
        IS_NEOX=is_neox,
        REUSE_FREQS_FRONT_PART=True,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        num_warps=num_warps,
        waves_per_eu=waves_per_eu,
    )
    return q_out

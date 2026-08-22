# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.rope.fused_qk_norm_rope_cached import (
    _fused_qk_norm_rope_cached_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_qk_norm_rope_cached(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-head RMSNorm then partial NeoX RoPE on q and k, in place.

        q[t, h] = rope(rmsnorm(q[t, h], q_weight), cos_sin_cache[t])

    Written for diffusion transformers, where the rotated subspace is a
    fraction of the head that is neither the whole head nor half of it -- 96 of
    128 for MiniMax-H3 -- which the existing rope ops do not cover, and where
    there is no KV cache to write through.

    Key parameters:
    - q, k: [T, H, D] contiguous, modified in place
    - q_weight, k_weight: [D] RMSNorm gains
    - cos_sin_cache: [T, R] with cos in the first R/2 columns and sin in the
      rest; R is the rotated width and must be even and <= D
    - eps: RMSNorm epsilon

    One program owns a token. A token's heads are contiguous, so the [H, D]
    tile is one coalesced run and the token's cos/sin row is read once for all
    heads -- unfused, that row is broadcast into a [T, H, D] temporary, and the
    rotation costs a negate and two concatenates on top.

    Returns:
    - (q, k), the same tensors
    """
    _LOGGER.info(
        f"FUSED_QK_NORM_ROPE_CACHED: q={tuple(q.shape)} "
        f"cache={tuple(cos_sin_cache.shape)}"
    )

    assert q.ndim == 3 and k.ndim == 3, "q and k must be [T, H, D]"
    assert q.shape == k.shape, f"q {tuple(q.shape)} != k {tuple(k.shape)}"
    T, H, D = q.shape
    # Only each token's [H, D] block has to be contiguous, not the whole
    # tensor: q and k are usually strided views into a packed qkv projection,
    # and operating on them in place avoids materialising either.
    for name, t in (("q", q), ("k", k)):
        assert (
            t.stride(-1) == 1 and t.stride(-2) == D
        ), f"{name} must have contiguous [H, D] blocks, got strides {t.stride()}"
    assert q_weight.shape == (D,) and k_weight.shape == (
        D,
    ), f"norm weights must be [{D}]"
    assert (
        cos_sin_cache.ndim == 2 and cos_sin_cache.shape[0] == T
    ), f"cache must be [{T}, R], got {tuple(cos_sin_cache.shape)}"
    rot = cos_sin_cache.shape[1]
    assert (
        rot % 2 == 0 and rot <= D
    ), f"rotated width {rot} must be even and at most head_dim {D}"
    if T == 0:
        return q, k

    _fused_qk_norm_rope_cached_kernel[(T,)](
        q,
        k,
        q_weight,
        k_weight,
        cos_sin_cache,
        H,
        D,
        rot,
        q.stride(0),
        k.stride(0),
        cos_sin_cache.stride(0),
        eps,
        BLOCK_H=triton.next_power_of_2(H),
        BLOCK_D=triton.next_power_of_2(D),
        num_warps=4,
    )
    return q, k

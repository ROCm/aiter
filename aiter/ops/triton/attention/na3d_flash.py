# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""3D neighborhood flash attention launcher.

Each query token at grid position (t, h, w) attends to the KT x KH x KW neighborhood
centered on it, with inward border-shift so every window has exactly KT x KH x KW keys.

Usage::

    from aiter.ops.triton.attention.na3d_flash import na3d_flash_attn

    out = na3d_flash_attn(q, k, v, kernel_size=(11, 11, 11))
    # q, k, v: (B, T, H, W, NH, HD) bfloat16, Q already scaled by head_dim**-0.5
    # out    : (B, T, H, W, NH, HD) bfloat16
"""

from __future__ import annotations

import torch
import triton

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton._triton_kernels.attention.na3d_flash import _na3d_flash_fwd
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_NA3D_FLASH_ARCHS = ("gfx942", "gfx950")


def na3d_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
) -> torch.Tensor:
    """Flash-style 3D neighborhood attention.

    Args:
        q, k, v     : ``(B, T, H, W, NH, HD)`` bfloat16, channels-last.
                      Q is expected to be pre-scaled by ``head_dim ** -0.5``.
        kernel_size : ``(KT, KH, KW)`` neighborhood window.

    Returns:
        Output tensor ``(B, T, H, W, NH, HD)`` bfloat16.

    Notes:
        Input tensors are consumed in their native ``(B, T, H, W, NH, HD)`` layout
        without staging copies. Only a ``contiguous()`` call is made if a tensor is
        not already contiguous in that order.  ``W >= 16`` is required so that all
        queries in a BLOCK_Q=16 program share the same (t, h) grid row.
    """
    B, T, H, W, NH, HD = q.shape
    KT, KH, KW = kernel_size
    SEQ = T * H * W

    gfx = get_gfx()
    assert (
        gfx in _NA3D_FLASH_ARCHS
    ), f"na3d_flash_attn is only supported on {_NA3D_FLASH_ARCHS}; got {gfx}."

    _LOGGER.info(
        "NA3D_FLASH_FWD: q=%s kernel=(%d,%d,%d) SEQ=%d HD=%d",
        tuple(q.shape),
        KT,
        KH,
        KW,
        SEQ,
        HD,
    )

    assert q.dtype == torch.bfloat16, "na3d_flash_attn: inputs must be bfloat16"
    assert (
        k.dtype == q.dtype and v.dtype == q.dtype
    ), "na3d_flash_attn: q/k/v must share dtype"
    assert (
        k.shape == q.shape and v.shape == q.shape
    ), "na3d_flash_attn: q/k/v must have shape (B, T, H, W, NH, HD)"
    assert (
        KT > 0 and KH > 0 and KW > 0
    ), f"na3d_flash_attn: kernel_size dimensions must be positive; got ({KT},{KH},{KW})"
    assert (
        KT <= T and KH <= H and KW <= W
    ), f"na3d_flash_attn: kernel_size=({KT},{KH},{KW}) must be <= (T,H,W)=({T},{H},{W})"
    assert KW <= 33, (
        f"na3d_flash_attn: KW={KW} is too large for the current autotune configs "
        f"(max supported KW is 33 with BLOCK_Q=32/BLOCK_KV=64)."
    )
    # Config coverage: BLOCK_Q=16/BLOCK_KV=32 supports KW <= 17; larger KW requires
    # BLOCK_Q=32/BLOCK_KV=64 which the pruner keeps only when W >= 32.
    assert KW <= 17 or W >= 32, (
        f"na3d_flash_attn: KW={KW} > 17 requires W >= 32 "
        f"(only the BLOCK_Q=32/BLOCK_KV=64 config covers KW > 17); got W={W}."
    )
    assert HD >= 16 and HD & (HD - 1) == 0, (
        f"head_dim {HD} must be a power of 2 and >= 16 "
        f"(tl.dot requires matrix dimensions of at least 16)."
    )
    assert W >= 16, f"W={W} is too small; kernel requires W >= BLOCK_Q (default 16)."

    # Ensure contiguous layout
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    stride_b = SEQ * NH * HD  # elements between batches
    stride_nh = HD  # elements between heads
    stride_seq = NH * HD  # elements between tokens

    # Grid: one program per (t, h) row per W-block.  This guarantees each
    # program covers queries from exactly one (t, h) row regardless of W % BLOCK_Q.
    grid = lambda meta: (T * H * triton.cdiv(W, meta["BLOCK_Q"]), B * NH)

    _na3d_flash_fwd[grid](
        q,
        k,
        v,
        out,
        stride_b,
        stride_nh,
        stride_seq,
        NH,
        T,
        H,
        W,
        HD=HD,
        KT=KT,
        KH=KH,
        KW=KW,
    )

    return out

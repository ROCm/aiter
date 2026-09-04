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
        ``W >= 16`` is required so that all queries in a BLOCK_Q=16 program share
        the same (t, h) grid row.  The autotune pruner enforces this.
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
        KT <= T and KH <= H and KW <= W
    ), f"na3d_flash_attn: kernel_size=({KT},{KH},{KW}) must be <= (T,H,W)=({T},{H},{W})"
    assert KW <= 33, (
        f"na3d_flash_attn: KW={KW} is too large for the current autotune configs "
        f"(max supported KW is 33 with BLOCK_Q=32/BLOCK_KV=64)."
    )
    assert HD & (HD - 1) == 0, f"head_dim {HD} must be a power of 2"
    assert W >= 16, f"W={W} is too small; kernel requires W >= BLOCK_Q (default 16)."

    def _flat(t: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, NH, HD) -> (B*NH, SEQ, HD) contiguous."""
        return t.permute(0, 4, 1, 2, 3, 5).reshape(B * NH, SEQ, HD).contiguous()

    q_f, k_f, v_f = _flat(q), _flat(k), _flat(v)
    out_f = torch.empty_like(q_f)

    # Grid: one program per (t, h) row per W-block.  This guarantees each
    # program covers queries from exactly one (t, h) row regardless of W % BLOCK_Q.
    grid = lambda meta: (T * H * triton.cdiv(W, meta["BLOCK_Q"]), B * NH)

    _na3d_flash_fwd[grid](
        q_f,
        k_f,
        v_f,
        out_f,
        SEQ * HD,  # stride_bnh
        HD,  # stride_seq
        T,
        H,
        W,
        HD=HD,
        KT=KT,
        KH=KH,
        KW=KW,
    )

    return out_f.reshape(B, NH, T, H, W, HD).permute(0, 2, 3, 4, 1, 5).contiguous()

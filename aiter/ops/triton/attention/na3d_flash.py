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

from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.triton._triton_kernels.attention.na3d_flash import (
    _get_config,
    _get_config_3d,
    _na3d_flash_fwd,
    _na3d_flash_fwd_3d,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()
_NA3D_FLASH_ARCHS = ("gfx942", "gfx950", "gfx1151")


def _ensure_seq_strided_6d(t: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
    """Return ``t`` as-is if addressable via ``(stride_b, stride_nh, stride_seq)``.

    Avoids full-volume staging copies for strided QKV views such as
    ``qkv.chunk(3, dim=-1)[2].view(B, T, H, W, NH, HD)`` while falling back to
    ``.contiguous()`` for arbitrary non-linear strides.
    """
    s0, s1, s2, s3, s4, s5 = t.stride()
    if (
        s5 == 1
        and s0 >= 0
        and s3 >= 0
        and s4 >= 0
        and (H == 1 or s2 == W * s3)
        and (T == 1 or s1 == H * W * s3)
    ):
        return t
    return t.contiguous()


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

    Supported domain (all asserted):
        - Device : running arch in ``{gfx942, gfx950, gfx1151}``; q/k/v on the
          current CUDA/HIP device.
        - dtype  : ``bfloat16`` for q, k, and v (shared).
        - Shape  : k and v match q's ``(B, T, H, W, NH, HD)``.
        - head_dim ``HD`` : power of two and ``>= 16`` (``tl.dot`` needs matrix
          dims of at least 16).
        - Window : ``0 < KT <= T``, ``0 < KH <= H``, ``0 < KW <= W``.
        - Width  : ``W >= 16`` (the smallest configured ``BLOCK_Q``).  ``KW <= 33``,
          and ``KW > 17`` additionally requires ``W >= 32`` -- large KW is served
          only by the ``BLOCK_Q=32/BLOCK_KV=64`` config, whose ``BLOCK_KV`` must
          cover ``BLOCK_Q + KW - 1``.

    Notes:
        **Inference / forward-pass only.** No autograd backward is implemented.
        Inputs with ``requires_grad=True`` will produce an output that is detached
        from the autograd graph, silently dropping gradients.

        Tensors are consumed in their native ``(B, T, H, W, NH, HD)`` layout; a
        ``contiguous()`` copy is made only for an input not already contiguous in
        that order (on ``gfx1151``, sequence-strided views such as
        ``qkv.chunk(3, dim=-1)[2].view(...)`` are consumed directly without a
        staging copy).  The tile configuration is read from the per-arch config
        file (no runtime autotune).
    """
    B, T, H, W, NH, HD = q.shape
    KT, KH, KW = kernel_size
    SEQ = T * H * W

    assert (
        q.is_cuda and k.is_cuda and v.is_cuda
    ), "na3d_flash_attn requires CUDA/HIP tensors"
    assert q.device == k.device == v.device, (
        f"na3d_flash_attn: q/k/v must be on the same device; got "
        f"q={q.device}, k={k.device}, v={v.device}"
    )
    current_device = torch.cuda.current_device()
    assert q.device.index == current_device, (
        f"na3d_flash_attn: kernel will launch on cuda:{current_device}, "
        f"but inputs are on {q.device}"
    )

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
        f"na3d_flash_attn: KW={KW} is too large for the published configs "
        f"(max supported KW is 33 with BLOCK_Q=32/BLOCK_KV=64)."
    )
    # Config coverage: BLOCK_Q=16/BLOCK_KV=32 (na3d_flash_small_kw) supports KW <= 17;
    # larger KW uses BLOCK_Q=32/BLOCK_KV=64 (na3d_flash_large_kw), which needs W >= 32.
    assert KW <= 17 or W >= 32, (
        f"na3d_flash_attn: KW={KW} > 17 requires W >= 32 "
        f"(only the BLOCK_Q=32/BLOCK_KV=64 config covers KW > 17); got W={W}."
    )
    assert HD >= 16 and HD & (HD - 1) == 0, (
        f"head_dim {HD} must be a power of 2 and >= 16 "
        f"(tl.dot requires matrix dimensions of at least 16)."
    )
    assert W >= 16, f"W={W} is too small; W must be >= the smallest BLOCK_Q (16)."

    if gfx == "gfx1151":
        q = _ensure_seq_strided_6d(q, T, H, W)
        k = _ensure_seq_strided_6d(k, T, H, W)
        v = _ensure_seq_strided_6d(v, T, H, W)
        out = torch.empty((B, T, H, W, NH, HD), dtype=q.dtype, device=q.device)

        cfg = _get_config_3d(KT, KH, KW, HD)
        bt, bh, bw = cfg["BT"], cfg["BH"], cfg["BW"]
        block_kv = cfg["BLOCK_KV"]
        wu = bw + KW - 1
        hu = bh + KH - 1
        no_divmod_2d = (wu & (wu - 1)) != 0
        # Bound includes masked tail lanes on the last HW iteration so int32
        # relative indexing never overflows even before predicate masking.
        max_rel_elems = (hu + block_kv) * W * max(k.stride(3), v.stride(3)) + HD
        use_i32_rel = max_rel_elems < (1 << 31)

        grid = (
            B * NH,
            triton.cdiv(T, bt) * triton.cdiv(H, bh) * triton.cdiv(W, bw),
        )
        launch_kwargs = {
            "num_warps": cfg["num_warps"],
            "num_stages": cfg["num_stages"],
        }
        waves_per_eu = cfg.get("waves_per_eu", 0)
        if waves_per_eu > 0:
            launch_kwargs["waves_per_eu"] = waves_per_eu

        _na3d_flash_fwd_3d[grid](
            q,
            k,
            v,
            out,
            q.stride(0),
            q.stride(4),
            q.stride(3),
            k.stride(0),
            k.stride(4),
            k.stride(3),
            v.stride(0),
            v.stride(4),
            v.stride(3),
            out.stride(0),
            out.stride(4),
            out.stride(3),
            NH,
            T,
            H,
            W,
            HD=HD,
            KT=KT,
            KH=KH,
            KW=KW,
            BT=bt,
            BH=bh,
            BW=bw,
            BLOCK_KV=block_kv,
            SWIZZLE_T=cfg["SWIZZLE_T"],
            SWIZZLE_H=cfg["SWIZZLE_H"],
            USE_I32_REL=use_i32_rel,
            NO_DIVMOD_2D=no_divmod_2d,
            **launch_kwargs,
        )
        return out

    # Ensure contiguous layout
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = torch.empty_like(q)

    stride_b = SEQ * NH * HD  # elements between batches
    stride_nh = HD  # elements between heads
    stride_seq = NH * HD  # elements between tokens

    # Per-arch tile from the config file (no runtime autotune).  The KW-based
    # selection guarantees BLOCK_Q <= W and BLOCK_KV >= BLOCK_Q + KW - 1 given the
    # asserts above (W >= 16; KW <= 17 or W >= 32; KW <= 33).
    cfg = _get_config(KW)
    block_q = cfg["BLOCK_Q"]

    # Grid: one program per (t, h) row per W-block.  This guarantees each
    # program covers queries from exactly one (t, h) row regardless of W % BLOCK_Q.
    grid = (T * H * triton.cdiv(W, block_q), B * NH)

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
        BLOCK_Q=block_q,
        BLOCK_KV=cfg["BLOCK_KV"],
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )

    return out

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..utility import dtypes

_BN = 256
_BK = 64


def _require_gfx1201(tensor: Tensor) -> None:
    # The kernel is a stub on other architectures in a multi-arch build.
    if tensor.device.type != "cuda":
        raise ValueError("gemm_a8w8_bpreshuffle_gfx1201 expects GPU tensors")
    arch = torch.cuda.get_device_properties(tensor.device).gcnArchName.split(":")[0]
    if arch != "gfx1201":
        raise RuntimeError(
            f"gemm_a8w8_bpreshuffle_gfx1201 requires gfx1201, got {arch}"
        )


@compile_ops("module_gfx1201_gemm_a8w8", develop=True)
def gfx1201_gemm_a8w8_bpreshuffle_hip(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Y: Tensor,
) -> None: ...


def gemm_a8w8_bpreshuffle_gfx1201_supported(
    n: int, k: int, dtype: torch.dtype = dtypes.bf16
) -> bool:
    """True if ``gemm_a8w8_bpreshuffle_gfx1201`` accepts an ``[N, K]`` weight and
    ``dtype`` output."""
    return n % _BN == 0 and k % (2 * _BK) == 0 and dtype == dtypes.bf16


def shuffle_weight_gfx1201(w: Tensor) -> Tensor:
    """Preshuffle an fp8 ``[N, K]`` weight for ``gemm_a8w8_bpreshuffle_gfx1201``.

    This layout is specific to the gfx1201 kernel and is NOT the
    ``shuffle_weight(w, layout=(16, 16))`` layout of ``gemm_a8w8_bpreshuffle``.

    Layout ``[N / 32][K / 64][lane 32][t 2][s 4][8 bytes]``: lane
    ``16 * h + 8 * hb + e`` of a 32-column group holds column ``16 * hb + 8 * t + e``
    at K bytes ``32 * h + 8 * s + (0..7)``, i.e. the WMMA fp8 A-operand order.
    Requires ``N % 32 == 0`` and ``K % 64 == 0``. The result has the shape, dtype
    and strides of ``w``.
    """
    n, k = w.shape
    if n % 32 or k % _BK:
        raise ValueError(
            f"shuffle_weight_gfx1201 needs N % 32 == 0 and K % 64 == 0, got {n}x{k}"
        )
    u = w.contiguous().view(torch.uint8).view(n // 32, 2, 2, 8, k // _BK, 2, 4, 8)
    return u.permute(0, 4, 5, 1, 3, 2, 6, 7).contiguous().view(n, k).view(w.dtype)


def gemm_a8w8_bpreshuffle_gfx1201(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    dtype: torch.dtype = dtypes.bf16,
) -> Tensor:
    """Per-token/per-channel FP8 GEMM on gfx1201 with a preshuffled weight.

    Returns a new ``[M, N]`` bf16 tensor ``(XQ @ W.T) * x_scale * w_scale.T``,
    where ``W`` is the row-major weight that ``shuffle_weight_gfx1201`` turned
    into ``WQ``.

    Args:
        XQ: ``[M, K]`` fp8 (e4m3fn) activation.
        WQ: ``[N, K]`` fp8 weight from ``shuffle_weight_gfx1201``.
        x_scale: fp32 per-token ``[M, 1]`` scale, or a per-tensor ``[1]``.
        w_scale: fp32 per-channel ``[N, 1]`` scale, or a per-tensor ``[1]``.
        dtype: output dtype; only ``bf16``.

    Requires ``N % 256 == 0``, ``K % 128 == 0`` and ``M * K``, ``N * K`` below
    ``2**31``. The kernel accumulates in fp32 and rounds once to bf16.
    """
    _require_gfx1201(XQ)
    if XQ.dtype != dtypes.fp8 or WQ.dtype != dtypes.fp8:
        raise TypeError(
            f"gemm_a8w8_bpreshuffle_gfx1201 expects fp8 XQ and WQ, got {XQ.dtype}, {WQ.dtype}"
        )
    m, k = XQ.shape
    n, w_k = WQ.shape
    if w_k != k or not gemm_a8w8_bpreshuffle_gfx1201_supported(n, k, dtype):
        raise ValueError(
            "gemm_a8w8_bpreshuffle_gfx1201 needs bf16 output, WQ K == XQ K, "
            f"N % 256 == 0 and K % 128 == 0, got M={m}, N={n}, K={k}, WQ K={w_k}, {dtype=}"
        )
    Y = torch.empty(m, n, dtype=dtype, device=XQ.device)
    gfx1201_gemm_a8w8_bpreshuffle_hip(
        XQ.contiguous(),
        WQ.contiguous(),
        x_scale.reshape(-1).to(dtypes.fp32).contiguous(),
        w_scale.reshape(-1).to(dtypes.fp32).contiguous(),
        Y,
    )
    return Y

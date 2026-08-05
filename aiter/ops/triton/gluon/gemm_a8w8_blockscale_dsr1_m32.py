# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-R1 M=32 block-scaled FP8 GEMM dispatch for gfx950."""

from __future__ import annotations

import functools
from collections.abc import Callable

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

Shape = tuple[int, int, int]
Kernel = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    torch.Tensor,
]

DSR1_M32_SHAPES: tuple[Shape, ...] = (
    (32, 2112, 7168),
    (32, 3072, 1536),
    (32, 4608, 7168),
    (32, 7168, 2048),
    (32, 7168, 2304),
)
DSR1_M32_KERNEL_SHAPES: dict[str, Shape] = {
    "gluon_dsr1_m32_n2112_k7168": (32, 2112, 7168),
    "gluon_dsr1_m32_n3072_k1536": (32, 3072, 1536),
    "gluon_dsr1_m32_n4608_k7168": (32, 4608, 7168),
    "gluon_dsr1_m32_n7168_k2048": (32, 7168, 2048),
    "gluon_dsr1_m32_n7168_k2304": (32, 7168, 2304),
}
_CK_FALLBACK_KERNEL_NAME = (
    "a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_1x1_"
    "16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1"
)
# Preserve the CK winners that occupied the five DSv3 CSV rows before Gluon.
# Padded-M and unsupported dtype/layout calls can still select an M=32 row;
# when the exact-shape contract rejects them, these configs avoid a regression
# to CK's untuned default heuristic.
DSR1_M32_CK_FALLBACK_CONFIGS: dict[str, tuple[int, str]] = {
    "gluon_dsr1_m32_n2112_k7168": (0, _CK_FALLBACK_KERNEL_NAME),
    "gluon_dsr1_m32_n3072_k1536": (0, _CK_FALLBACK_KERNEL_NAME),
    "gluon_dsr1_m32_n4608_k7168": (2, _CK_FALLBACK_KERNEL_NAME),
    "gluon_dsr1_m32_n7168_k2048": (0, _CK_FALLBACK_KERNEL_NAME),
    "gluon_dsr1_m32_n7168_k2304": (0, _CK_FALLBACK_KERNEL_NAME),
}


@functools.cache
def _load_kernel(shape: Shape) -> Kernel:
    if shape == (32, 2112, 7168):
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.n2112_k7168_m32 import (
            block_scaled_mm_n2112_k7168_m32,
        )

        return block_scaled_mm_n2112_k7168_m32
    if shape == (32, 3072, 1536):
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.n3072_k1536_m32 import (
            block_scaled_mm_n3072_k1536_m32,
        )

        return block_scaled_mm_n3072_k1536_m32
    if shape == (32, 4608, 7168):
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.n4608_k7168_m32 import (
            block_scaled_mm_n4608_k7168_m32,
        )

        return block_scaled_mm_n4608_k7168_m32
    if shape == (32, 7168, 2048):
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.n7168_k2048_m32 import (
            block_scaled_mm_n7168_k2048_m32,
        )

        return block_scaled_mm_n7168_k2048_m32
    if shape == (32, 7168, 2304):
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.n7168_k2304_m32 import (
            block_scaled_mm_n7168_k2304_m32,
        )

        return block_scaled_mm_n7168_k2304_m32
    raise KeyError(f"No DeepSeek-R1 M=32 kernel registered for {shape}")


def _has_exact_contract(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    gfx: str,
    expected_shape: Shape,
) -> bool:
    if gfx != "gfx950" or x.ndim != 2 or weight.ndim != 2:
        return False

    m, k = x.shape
    n, weight_k = weight.shape
    shape = (m, n, k)
    if shape != expected_shape or weight_k != k:
        return False

    if x.dtype != torch.float8_e4m3fn or weight.dtype != torch.float8_e4m3fn:
        return False
    if x_scale.dtype != torch.float32 or weight_scale.dtype != torch.float32:
        return False
    if out.dtype != torch.bfloat16:
        return False

    k_blocks = (k + 127) // 128
    n_blocks = (n + 127) // 128
    if x_scale.shape != (m, k_blocks):
        return False
    if weight_scale.shape != (n_blocks, k_blocks):
        return False
    if out.shape != (m, n):
        return False

    tensors = (x, weight, x_scale, weight_scale, out)
    if any(not tensor.is_cuda or tensor.device != x.device for tensor in tensors):
        return False
    if any(not tensor.is_contiguous() for tensor in tensors):
        return False
    return weight.stride(0) == k


def try_gemm_a8w8_blockscale_dsr1_m32(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    *,
    kernel_name: str,
    gfx: str | None = None,
) -> torch.Tensor | None:
    """Run a registered exact-shape kernel, or return ``None`` to fall back."""

    try:
        expected_shape = DSR1_M32_KERNEL_SHAPES[kernel_name]
    except KeyError as error:
        raise RuntimeError(
            f"Unknown gfx950 DeepSeek-R1 M=32 Gluon kernel {kernel_name!r}"
        ) from error

    runtime_gfx = get_gfx_runtime() if gfx is None else gfx
    if not _has_exact_contract(
        x,
        weight,
        x_scale,
        weight_scale,
        out,
        runtime_gfx,
        expected_shape,
    ):
        return None

    return _load_kernel(expected_shape)(x, weight, x_scale, weight_scale, out)


__all__ = [
    "DSR1_M32_CK_FALLBACK_CONFIGS",
    "DSR1_M32_KERNEL_SHAPES",
    "DSR1_M32_SHAPES",
    "try_gemm_a8w8_blockscale_dsr1_m32",
]

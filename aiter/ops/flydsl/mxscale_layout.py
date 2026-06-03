# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side layout helpers for the FlyDSL gfx1250 MXScale dense GEMM kernel.

The B weight is prepared with the ordinary ``aiter.ops.shuffle.shuffle_weight``
16x16 byte preshuffle (shape-preserving): the kernel's B layout is byte-identical
to that generic shuffle, so MXScale needs no bespoke weight preprocessing. K is
*not* padded here — the selected kernel must divide K (callers tune kernels whose
``tile_k * split_k`` divides K); only M is padded at runtime to ``tile_m``.

These helpers therefore cover only the per-call work the generic shuffle cannot
do: padding M, and swizzling the E8M0 scales into the WMMA scale-fragment layout.

Kept separate from ``aiter.utility.fp4_utils`` because:
  * ``utility/fp4_utils.py`` must not depend on FlyDSL.
  * The scale preshuffle here is a FlyDSL-WMMA private convention
    (distinct from ``e8m0_shuffle`` which targets the ASM 256x8 layout).
"""

from __future__ import annotations

import torch
from torch import Tensor

# Compile-time constants of the gfx1250 mxscale kernel; see
# kernels/gemm_fp8fp4_gfx1250.py: SCALE_BLOCK / WMMA_M / WMMA_K.
SCALE_BLOCK: int = 32
WMMA_DIM: int = 16
WMMA_K: int = 128
SCALES_PER_WMMA: int = 4

# The kernel pads scale rows with E8M0(127) which decodes to 2^0 = 1.0
# so padded contributions accumulate to zero (data is padded with 0).
E8M0_ONE: int = 127
# Keep in sync with the num_buffers validation in
# kernels/gemm_fp8fp4_gfx1250.py:compile_mxscale_gemm.
SUPPORTED_NUM_BUFFERS: tuple[int, ...] = (2, 3, 4)


def mxscale_pack_factors(data_format: str) -> tuple[int, int]:
    """Return (PACK_FACTOR_A, PACK_FACTOR_B) for the given data_format.

    fp8:  A FP8 (1 value/byte), B FP8 (1 value/byte) -> (1, 1)
    a8w4: A FP8 (1 value/byte), B FP4 (2 values/byte) -> (1, 2)
    fp4:  A FP4 (2 values/byte), B FP4 (2 values/byte) -> (2, 2)
    """
    if data_format == "fp8":
        return 1, 1
    if data_format == "a8w4":
        return 1, 2
    if data_format == "fp4":
        return 2, 2
    raise ValueError(f"unsupported data_format={data_format!r}")


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def align_up(value: int, alignment: int) -> int:
    """Public wrapper for the fixed MXScale host-side padding helpers."""
    return _align_up(value, alignment)


def validate_mxscale_kernel_shape(
    *,
    N: int,
    K: int,
    tile_n: int,
    tile_k: int,
    num_buffers: int,
    split_k: int = 1,
) -> None:
    """Validate a selected kernel against the (unpadded) logical B shape.

    K is never padded: a kernel only runs when ``tile_k * split_k`` divides K and
    N is divisible by ``tile_n``. Otherwise this raises (no silent fallback).
    """
    if num_buffers not in SUPPORTED_NUM_BUFFERS:
        raise ValueError(
            f"num_buffers must be one of {SUPPORTED_NUM_BUFFERS}, got {num_buffers}"
        )
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    if N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_n={tile_n}")
    # gfx1250 WMMA contracts K in chunks of WMMA_K=128; K padding is not yet
    # supported, so a non-multiple K is rejected up front.
    if K % WMMA_K != 0:
        raise ValueError(
            f"K={K} must be divisible by WMMA_K={WMMA_K} (K padding not yet supported)"
        )
    if K % tile_k != 0:
        raise ValueError(f"K={K} must be divisible by tile_k={tile_k}")
    if K % split_k != 0:
        raise ValueError(f"K={K} must be divisible by split_k={split_k}")
    split_k_chunk = K // split_k
    if split_k_chunk % tile_k != 0:
        raise ValueError(
            f"K/split_k={split_k_chunk} must be divisible by tile_k={tile_k}"
        )
    num_k_tiles = split_k_chunk // tile_k
    if num_k_tiles < num_buffers:
        raise ValueError(
            f"num_buffers={num_buffers} requires at least {num_buffers} K tiles per "
            f"split, but K={K}, tile_k={tile_k}, split_k={split_k} gives "
            f"{num_k_tiles}"
        )


def _pad_2d(t: Tensor, rows: int, cols: int, fill_value: int) -> Tensor:
    if t.shape == (rows, cols):
        # Force contiguous: the downstream preshuffle uses .view(), which would
        # silently reorder a non-contiguous tensor by its underlying storage.
        return t.contiguous()
    # All MX operands are 1-byte (fp8 / fp4x2 / e8m0). Fill through a uint8 buffer
    # so fill_value lands as a *raw byte* — e.g. E8M0 127 == 2^0 == 1.0. Filling
    # with torch.full(..., dtype=float8_*) would treat 127 as the float 127.0 and
    # re-encode it (E8M0 would store 0x86 = 2^7), corrupting the pad value.
    if t.element_size() == 1:
        padded_u8 = torch.full(
            (rows, cols), fill_value, dtype=torch.uint8, device=t.device
        )
        padded_u8[: t.shape[0], : t.shape[1]] = t.contiguous().view(torch.uint8)
        return padded_u8.view(t.dtype)
    padded = torch.full((rows, cols), fill_value, dtype=t.dtype, device=t.device)
    padded[: t.shape[0], : t.shape[1]] = t
    return padded


def preshuffle_e8m0_scale_wmma(
    scale: Tensor,
    warp_tile: int,
    scale_k_per_tile: int = 4,
    wmma_dim: int = WMMA_DIM,
) -> Tensor:
    """Preshuffle E8M0 scale into the layout consumed by WMMA scale fragments.

    Distinct from ``aiter.utility.fp4_utils.e8m0_shuffle`` which targets a
    different (ASM 256x8) layout. Vendored from
    FlyDSL/tests/kernels/test_gemm_fp8fp4_gfx1250.py:preshuffle_e8m0_scale.
    """
    _, k_scale = scale.shape
    if scale_k_per_tile <= 0:
        raise ValueError(f"scale_k_per_tile must be positive, got {scale_k_per_tile}")
    if scale_k_per_tile % SCALES_PER_WMMA != 0:
        raise ValueError(
            f"scale_k_per_tile must be divisible by {SCALES_PER_WMMA}, "
            f"got {scale_k_per_tile}"
        )
    if k_scale % scale_k_per_tile != 0:
        raise ValueError(
            f"K_scale={k_scale} must be divisible by scale_k_per_tile="
            f"{scale_k_per_tile}"
        )
    wmma_rep = warp_tile // wmma_dim
    k_groups = k_scale // scale_k_per_tile
    k_wmma_steps = scale_k_per_tile // SCALES_PER_WMMA
    scale = scale.contiguous()  # .view() below requires contiguous storage
    g = scale.view(-1, wmma_rep, wmma_dim, k_groups, k_wmma_steps, SCALES_PER_WMMA)
    g = g.permute(0, 2, 3, 4, 1, 5).contiguous()
    return g.reshape(-1, k_groups * k_wmma_steps * wmma_rep * SCALES_PER_WMMA)


def preshuffle_mxscale_scale_for_kernel(
    b_scale: Tensor,
    *,
    tile_n: int,
    tile_k: int,
    n_warp: int,
) -> Tensor:
    """Swizzle a raw row-major B scale (N, K // 32) for a selected MXScale kernel.

    No padding: N must be divisible by ``tile_n`` and ``K_scale`` by the kernel's
    scale-per-tile (guaranteed when ``tile_k`` divides K).
    """
    prepared_n, k_scale = b_scale.shape
    if prepared_n % tile_n != 0:
        raise ValueError(f"N={prepared_n} must be divisible by tile_n={tile_n}")
    if n_warp <= 0:
        raise ValueError(f"n_warp must be positive, got {n_warp}")
    if tile_n % n_warp != 0:
        raise ValueError(f"tile_n={tile_n} must be divisible by n_warp={n_warp}")
    if tile_k % SCALE_BLOCK != 0:
        raise ValueError(
            f"tile_k={tile_k} must be divisible by SCALE_BLOCK={SCALE_BLOCK}"
        )
    scale_k_per_tile = tile_k // SCALE_BLOCK
    warp_tile_n = tile_n // n_warp
    if warp_tile_n % WMMA_DIM != 0:
        raise ValueError(
            f"tile_n/n_warp={warp_tile_n} must be divisible by WMMA_DIM={WMMA_DIM}"
        )
    if k_scale % scale_k_per_tile != 0:
        raise ValueError(
            f"K_scale={k_scale} must be divisible by scale_k_per_tile="
            f"{scale_k_per_tile}"
        )
    return preshuffle_e8m0_scale_wmma(
        b_scale, warp_tile_n, scale_k_per_tile=scale_k_per_tile
    )


def preshuffle_mxscale_activation(
    a: Tensor,
    a_scale: Tensor,
    data_format: str,
    tile_m: int,
    tile_k: int,
    m_warp: int,
) -> tuple[Tensor, Tensor]:
    """Pad M + swizzle the A activation and its E8M0 scale for the MXScale kernel.

    Per-call activation preparation: only M is padded (to ``tile_m``); K is not
    padded (the kernel divides K). A_scale is padded in M with E8M0(1.0) and then
    swizzled into the WMMA scale-fragment layout.
    """
    pack_a, _ = mxscale_pack_factors(data_format)
    m = a.shape[0]
    k = a.shape[1] * pack_a
    padded_m = _align_up(m, tile_m)
    a_p = _pad_2d(a, padded_m, k // pack_a, fill_value=0)
    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // m_warp
    a_s_p = _pad_2d(a_scale, padded_m, k // SCALE_BLOCK, fill_value=E8M0_ONE)
    a_s_p = preshuffle_e8m0_scale_wmma(a_s_p, warp_tile_m, scale_k_per_tile=skt)
    return a_p, a_s_p


def to_kernel_uint8(t: Tensor) -> Tensor:
    """Flatten an FP8 / E8M0 / packed-FP4 tensor to a 1-D uint8 view.

    The FlyDSL launcher takes raw byte buffers; viewing as uint8 sidesteps
    DLPack dtype quirks for the sub-byte / FP8 dtypes.
    """
    return t.contiguous().view(torch.uint8).view(-1)

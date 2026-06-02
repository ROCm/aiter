# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side layout helpers for the FlyDSL gfx1250 MXScale dense GEMM kernel.

These helpers prepare A / B / scale tensors into the exact layout the
``compile_mxscale_gemm(...)`` kernel expects: padded to tile boundaries,
B preshuffled into 16x16 byte tiles, E8M0 scales preshuffled for the
WMMA scale-fragment access pattern.

Kept separate from ``aiter.utility.fp4_utils`` because:
  * ``utility/fp4_utils.py`` must not depend on FlyDSL.
  * The preshuffle layouts here are FlyDSL-WMMA private conventions
    (distinct from ``e8m0_shuffle`` which targets the ASM 256x8 layout).
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

# Compile-time constants of the gfx1250 mxscale kernel; see
# kernels/gemm_fp8fp4_gfx1250.py: SCALE_BLOCK / WMMA_M.
SCALE_BLOCK: int = 32
WMMA_DIM: int = 16
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


def mxscale_k_tiles_per_split(K: int, tile_k: int, split_k: int = 1) -> tuple[int, int]:
    """Return (num_k_tiles_per_split, padded_k) for the host padding rule."""
    if tile_k <= 0:
        raise ValueError(f"tile_k must be positive, got {tile_k}")
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    padded_k = _align_up(K, tile_k * split_k)
    return padded_k // (tile_k * split_k), padded_k


def recommended_num_buffers(K: int, tile_k: int, split_k: int = 1) -> Optional[int]:
    """Return the largest supported num_buffers value that fits this K shape."""
    num_k_tiles, _ = mxscale_k_tiles_per_split(K, tile_k, split_k)
    choices = [value for value in SUPPORTED_NUM_BUFFERS if value <= num_k_tiles]
    return max(choices) if choices else None


def validate_mxscale_num_buffers(
    K: int,
    tile_k: int,
    num_buffers: int,
    split_k: int = 1,
) -> None:
    """Reject K shapes that cannot satisfy the requested pipeline depth."""
    if num_buffers not in SUPPORTED_NUM_BUFFERS:
        raise ValueError(
            f"num_buffers must be one of {SUPPORTED_NUM_BUFFERS}, got {num_buffers}"
        )

    num_k_tiles, padded_k = mxscale_k_tiles_per_split(K, tile_k, split_k)
    if num_k_tiles >= num_buffers:
        return

    min_padded_k = tile_k * split_k * num_buffers
    suggestion = recommended_num_buffers(K, tile_k, split_k)
    if suggestion is None:
        recommendation = (
            "No supported num_buffers value fits this shape; increase K so "
            f"padded K >= {min_padded_k} or reduce split_k."
        )
    else:
        recommendation = (
            f"Recommended num_buffers={suggestion} for this shape, or increase "
            f"K so padded K >= {min_padded_k}."
        )
    raise ValueError(
        f"num_buffers={num_buffers} requires at least {num_buffers} K tiles per "
        f"split, but K={K} pads to {padded_k}, giving {num_k_tiles}. "
        f"{recommendation}"
    )


def get_padded_problem_shape(
    data_format: str,
    M: int,
    N: int,
    K: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int = 1,
) -> dict:
    """Pad runtime (M, N, K) so the kernel always runs full tiles.

    K is padded to ``tile_k * split_k`` boundary because each split-K chunk
    must be a multiple of tile_k. M and N are padded to tile_m / tile_n.
    """
    if K % SCALE_BLOCK != 0:
        raise ValueError(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")
    pack_a, pack_b = mxscale_pack_factors(data_format)
    padded_k = _align_up(K, tile_k * split_k)
    return {
        "M": _align_up(M, tile_m),
        "N": _align_up(N, tile_n),
        "K": padded_k,
        "K_scale": padded_k // SCALE_BLOCK,
        "pack_a": pack_a,
        "pack_b": pack_b,
    }


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


def pad_mxscale_inputs(
    a: Tensor,
    b: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    padded_shape: dict,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Pad A / B / A_scale / B_scale to padded_shape boundaries.

    Data is padded with 0 (decodes to 0); scales are padded with
    E8M0(127) = 2^0 = 1.0 so zero data contributes zero to the
    accumulator regardless of scale.
    """
    a = _pad_2d(
        a, padded_shape["M"], padded_shape["K"] // padded_shape["pack_a"], fill_value=0
    )
    b = _pad_2d(
        b, padded_shape["N"], padded_shape["K"] // padded_shape["pack_b"], fill_value=0
    )
    a_scale = _pad_2d(
        a_scale, padded_shape["M"], padded_shape["K_scale"], fill_value=E8M0_ONE
    )
    b_scale = _pad_2d(
        b_scale, padded_shape["N"], padded_shape["K_scale"], fill_value=E8M0_ONE
    )
    return a, b, a_scale, b_scale


def preshuffle_b_16x16(b: Tensor, rows: int, cols: int) -> Tensor:
    """Preshuffle B into 16x16 byte tiles for WMMA-friendly LDS loads.

    Works for both FP8 (cols = K) and FP4 (cols = K // 2).
    Vendored from FlyDSL/tests/kernels/utils/fp4_utils.py:preshuffle_b_16x16.
    """
    if rows % 16 != 0:
        raise ValueError(f"rows must be a multiple of 16, got {rows}")
    if cols % 16 != 0:
        raise ValueError(f"cols must be a multiple of 16, got {cols}")
    # .contiguous() first: .view() on a non-contiguous tensor reorders by storage.
    b = b.contiguous().view(rows, cols)
    b = b.view(rows // 16, 16, cols // 16, 16)
    b = b.permute(0, 2, 1, 3).contiguous()
    return b.view(rows, cols)


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


def to_kernel_uint8(t: Tensor) -> Tensor:
    """Flatten an FP8 / E8M0 / packed-FP4 tensor to a 1-D uint8 view.

    The FlyDSL launcher takes raw byte buffers; viewing as uint8 sidesteps
    DLPack dtype quirks for the sub-byte / FP8 dtypes.
    """
    return t.contiguous().view(torch.uint8).view(-1)

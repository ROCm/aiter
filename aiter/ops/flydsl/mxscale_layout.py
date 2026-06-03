# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host-side layout helpers for the FlyDSL gfx1250 MXScale dense GEMM kernel.

These helpers prepare A / B / scale tensors for the gfx1250 MXScale GEMM.
B-side weight preprocessing is intentionally stable: it keeps logical N
unchanged, pads only K to the kernel family contract, then applies a 16x16 byte
preshuffle. Kernel-specific E8M0 scale swizzling is handled privately by the
runtime wrapper.

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

# Stable B-side preprocessing contract. Keep N unchanged, matching the ordinary
# bpreshuffle convention: callers/tuners choose kernels whose tile_n divides N.
# K is padded once so B can be prepared before runtime kernel selection.
MXSCALE_B_PAD_K: int = 128
MXSCALE_B_PAD_K_MIN: int = 256

MXSCALE_B_LAYOUT: str = "b16x16_kpad_v1"
MXSCALE_B_SCALE_LAYOUT: str = "e8m0_rowmajor_kpad_v1"


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


def mxscale_k_tiles_per_split(K: int, tile_k: int, split_k: int = 1) -> tuple[int, int]:
    """Return (num_k_tiles_per_split, padded_k) for the host padding rule."""
    if tile_k <= 0:
        raise ValueError(f"tile_k must be positive, got {tile_k}")
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    padded_k = _align_up(K, tile_k * split_k)
    return padded_k // (tile_k * split_k), padded_k


def get_padded_weight_shape(data_format: str, N: int, K: int) -> dict:
    """Return the stable B-side padded shape for an MXScale weight tensor."""
    if N % WMMA_DIM != 0:
        raise ValueError(f"N={N} must be divisible by {WMMA_DIM} for 16x16 B shuffle")
    if K % SCALE_BLOCK != 0:
        raise ValueError(f"K={K} must be divisible by SCALE_BLOCK={SCALE_BLOCK}")
    _, pack_b = mxscale_pack_factors(data_format)
    padded_k = max(_align_up(K, MXSCALE_B_PAD_K), MXSCALE_B_PAD_K_MIN)
    return {
        "N": N,
        "K": padded_k,
        "K_scale": padded_k // SCALE_BLOCK,
        "pack_b": pack_b,
    }


def validate_mxscale_kernel_shape(
    *,
    N: int,
    K: int,
    tile_n: int,
    tile_k: int,
    num_buffers: int,
    split_k: int = 1,
) -> None:
    """Validate a selected kernel against the prepared B shape."""
    if num_buffers not in SUPPORTED_NUM_BUFFERS:
        raise ValueError(
            f"num_buffers must be one of {SUPPORTED_NUM_BUFFERS}, got {num_buffers}"
        )
    if split_k < 1:
        raise ValueError(f"split_k must be >= 1, got {split_k}")
    if N % tile_n != 0:
        raise ValueError(f"N={N} must be divisible by tile_n={tile_n}")
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
    num_buffers: int = 2,
) -> dict:
    """Pad runtime (M, N, K) using the stable B-side K-padding contract.

    ``N`` and ``K`` are logical B dimensions; this helper first applies the
    fixed B-side K padding policy, then validates the selected kernel can
    consume that prepared shape. ``M`` remains activation/runtime local and is
    padded to ``tile_m``.
    """
    pack_a, pack_b = mxscale_pack_factors(data_format)
    weight_shape = get_padded_weight_shape(data_format, N, K)
    prepared_n = weight_shape["N"]
    padded_k = weight_shape["K"]
    validate_mxscale_kernel_shape(
        N=prepared_n,
        K=padded_k,
        tile_n=tile_n,
        tile_k=tile_k,
        num_buffers=num_buffers,
        split_k=split_k,
    )
    return {
        "M": _align_up(M, tile_m),
        "N": prepared_n,
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


def _set_mxscale_b_metadata(
    b: Tensor,
    *,
    data_format: str,
    logical_n: int,
    logical_k: int,
    prepared_n: int,
    padded_k: int,
    b_layout: str,
) -> Tensor:
    b._mxscale_layout = b_layout
    b._mxscale_data_format = data_format
    b._mxscale_logical_shape = (logical_n, logical_k)
    b._mxscale_padded_shape = (prepared_n, padded_k)
    return b


def _set_mxscale_scale_metadata(
    b_scale: Tensor,
    *,
    data_format: str,
    logical_n: int,
    logical_k: int,
    prepared_n: int,
    padded_k: int,
) -> Tensor:
    b_scale._mxscale_layout = MXSCALE_B_SCALE_LAYOUT
    b_scale._mxscale_data_format = data_format
    b_scale._mxscale_logical_shape = (logical_n, logical_k // SCALE_BLOCK)
    b_scale._mxscale_padded_shape = (prepared_n, padded_k // SCALE_BLOCK)
    return b_scale


def pad_weight_data_mxscale(
    b: Tensor,
    data_format: str,
) -> Tensor:
    """Apply the MXScale B data K-padding contract, without touching scale."""
    _, pack_b = mxscale_pack_factors(data_format)
    n = b.shape[0]
    k = b.shape[1] * pack_b
    weight_shape = get_padded_weight_shape(data_format, n, k)
    prepared_n, padded_k = weight_shape["N"], weight_shape["K"]
    b_p = _pad_2d(b, prepared_n, padded_k // pack_b, fill_value=0)
    return _set_mxscale_b_metadata(
        b_p,
        data_format=data_format,
        logical_n=n,
        logical_k=k,
        prepared_n=prepared_n,
        padded_k=padded_k,
        b_layout="rowmajor_kpad_v1",
    )


def pad_scale_mxscale(
    b_scale: Tensor,
    *,
    data_format: str,
    logical_n: int,
    logical_k: int,
    prepared_n: int,
    padded_k: int,
) -> Tensor:
    """Pad raw row-major MXScale B scale to match a prepared B tensor."""
    if b_scale.shape != (logical_n, logical_k // SCALE_BLOCK):
        raise ValueError(
            f"b_scale shape must be {(logical_n, logical_k // SCALE_BLOCK)}, "
            f"got {tuple(b_scale.shape)}"
        )
    b_s_p = _pad_2d(b_scale, prepared_n, padded_k // SCALE_BLOCK, fill_value=E8M0_ONE)
    return _set_mxscale_scale_metadata(
        b_s_p,
        data_format=data_format,
        logical_n=logical_n,
        logical_k=logical_k,
        prepared_n=prepared_n,
        padded_k=padded_k,
    )


def preshuffle_mxscale_weight_data(
    b: Tensor,
    data_format: str,
) -> Tensor:
    """Pad + stable 16x16-preshuffle B data; B scale stays runtime-private."""
    b_p = pad_weight_data_mxscale(b, data_format=data_format)
    logical_n, logical_k = b_p._mxscale_logical_shape
    prepared_n, padded_k = b_p._mxscale_padded_shape
    _, pack_b = mxscale_pack_factors(data_format)
    b_p = preshuffle_b_16x16(b_p, prepared_n, padded_k // pack_b)
    b_p.is_shuffled = True
    return _set_mxscale_b_metadata(
        b_p,
        data_format=data_format,
        logical_n=logical_n,
        logical_k=logical_k,
        prepared_n=prepared_n,
        padded_k=padded_k,
        b_layout=MXSCALE_B_LAYOUT,
    )


def preshuffle_mxscale_scale_for_kernel(
    b_scale_pad: Tensor,
    *,
    tile_n: int,
    tile_k: int,
    n_warp: int,
) -> Tensor:
    """Swizzle a padded row-major B scale for a selected MXScale kernel."""
    padded_shape = getattr(b_scale_pad, "_mxscale_padded_shape", None)
    if padded_shape is None:
        raise ValueError(
            "B_scale is missing MXScale padded-shape metadata; build it with "
            "the internal MXScale B-scale padding helper"
        )
    prepared_n, k_scale = padded_shape
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
        b_scale_pad, warp_tile_n, scale_k_per_tile=scale_k_per_tile
    )


def preshuffle_mxscale_activation(
    a: Tensor,
    a_scale: Tensor,
    data_format: str,
    tile_m: int,
    tile_k: int,
    m_warp: int,
    split_k: int = 1,
    padded_k: Optional[int] = None,
) -> tuple[Tensor, Tensor]:
    """Pad + preshuffle the A activation and its E8M0 scale for the MXScale kernel.

    Per-call activation preparation. B data uses a stable weight-load-time
    preshuffle; A and A_scale stay runtime-local because M and the selected
    kernel tile are per-call concerns.
    """
    pack_a, _ = mxscale_pack_factors(data_format)
    m = a.shape[0]
    k = a.shape[1] * pack_a
    padded_m = _align_up(m, tile_m)
    if padded_k is None:
        padded_k = _align_up(k, tile_k * split_k)
    elif padded_k < k:
        raise ValueError(f"padded_k={padded_k} must be >= logical K={k}")
    a_p = _pad_2d(a, padded_m, padded_k // pack_a, fill_value=0)
    skt = tile_k // SCALE_BLOCK
    warp_tile_m = tile_m // m_warp
    a_s_p = _pad_2d(a_scale, padded_m, padded_k // SCALE_BLOCK, fill_value=E8M0_ONE)
    a_s_p = preshuffle_e8m0_scale_wmma(a_s_p, warp_tile_m, scale_k_per_tile=skt)
    return a_p, a_s_p


def to_kernel_uint8(t: Tensor) -> Tensor:
    """Flatten an FP8 / E8M0 / packed-FP4 tensor to a 1-D uint8 view.

    The FlyDSL launcher takes raw byte buffers; viewing as uint8 sidesteps
    DLPack dtype quirks for the sub-byte / FP8 dtypes.
    """
    return t.contiguous().view(torch.uint8).view(-1)

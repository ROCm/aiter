# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 index-cache layout constants and view validation, for the prefill scorers.

The same constants and `_validate_cache` as `minimax_m3_index_score_fp4` (the
decode scorer); duplicated so the prefill scorers do not depend on the decode PR.
Keep the two in step, or have the decode file import from here once both are in.
"""

import torch

from aiter.ops.flydsl.kernels.minimax_m3_index_score import (
    HEAD_DIM,
    PAGE,
    _validate_tensor,
)

DATA_BYTES = HEAD_DIM // 2  # packed e2m1 bytes per token
SCALE_BYTES = HEAD_DIM // 32  # e8m0 bytes per token


def _validate_cache(k_data, k_scale, device):
    """Shapes, dtypes and strides of the two views. Contents (block ids inside
    the pool, e8m0 not 0/255) are a caller contract."""
    _validate_tensor(k_data, "k_data", (torch.uint8,), 3, device, 16, False)
    _validate_tensor(k_scale, "k_scale", (torch.uint8,), 3, device, 4, False)
    if k_data.shape[1:] != (PAGE, DATA_BYTES) or k_data.shape[0] > 0x7FFFFFFF:
        raise ValueError("k_data: expected [pages, 128, 64] uint8 with int32 page IDs")
    if k_scale.shape != (k_data.shape[0], PAGE, SCALE_BYTES):
        raise ValueError(
            "k_scale: expected [pages, 128, 4] uint8, same pages as k_data"
        )
    if k_data.stride(1) != DATA_BYTES or k_data.stride(2) != 1:
        raise ValueError(
            f"k_data: tokens must be packed (stride(1) {DATA_BYTES}, stride(2) 1), "
            f"got {k_data.stride()}"
        )
    if k_scale.stride(1) != SCALE_BYTES or k_scale.stride(2) != 1:
        raise ValueError(
            f"k_scale: tokens must be packed (stride(1) {SCALE_BYTES}, stride(2) 1), "
            f"got {k_scale.stride()}"
        )
    # A page stride below the view's own extent makes successive pages overlap.
    if k_data.stride(0) < PAGE * DATA_BYTES or k_data.stride(0) % 16:
        raise ValueError(
            f"k_data: page stride must be a multiple of 16 B and >= {PAGE * DATA_BYTES}"
        )
    if k_scale.stride(0) < PAGE * SCALE_BYTES or k_scale.stride(0) % 4:
        raise ValueError(
            f"k_scale: page stride must be a multiple of 4 B and >= {PAGE * SCALE_BYTES}"
        )

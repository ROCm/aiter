# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton scale shuffle of the 4-wave mxfp8 GEMM (``gemm_a8w8_mxfp8_4wave``); a separate module so that
Triton is imported only when a scale is really shuffled."""

import triton
import triton.language as tl


@triton.jit
def _shuffle_scale_kernel(src_ptr, dst_ptr, rows, cols, k1, tiles, TILES: tl.constexpr):
    """dst[N1, K1, 4 (k lane), 16 (n lane), 2 (k pack), 2 (n pack)] = src[N1, 2, 16, K1, 2, 4]
    over 32-row groups (``shuffle_scale_a16w4``'s order for one expert, not gate/up); rows
    >= ``rows`` read as 127 (2^0), the padding of the last 32-row group. One program moves
    TILES tiles of 32 rows x 8 columns (256 contiguous output bytes each)."""
    e = tl.arange(0, 256)
    # the element's place in the tile: (k_l, n_l, k_p, n_p) are the output's fastest 4 digits
    n_p = e % 2
    k_p = (e // 2) % 2
    n_l = (e // 4) % 16
    k_l = e // 64
    r = n_p * 16 + n_l
    c = k_p * 4 + k_l
    for t in range(TILES):
        tile = (tl.program_id(0) * TILES + t).to(tl.int64)
        n1 = tile // k1
        kk = tile % k1
        row = n1 * 32 + r
        live = tile < tiles
        v = tl.load(
            src_ptr + row * cols + kk * 8 + c, mask=live & (row < rows), other=127
        )
        tl.store(dst_ptr + tile * 256 + e, v, mask=live)

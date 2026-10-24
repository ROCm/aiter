# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""B operand layouts. No padding at any width.

B operand of v_mfma_scale_f32_16x16x128_f8f6f4 (fp4): lane l supplies column
(l % 16) and K elements [(l // 16) * 32, +32) of a 128-wide K step, i.e. 16
contiguous bytes. One (16-column, 128-K) atom is therefore 1024 contiguous
bytes and a wave loads it with a single 16 B/lane load.

    B       : [E][N/16][K/128][64 lanes][16 B]   (natural rows)
    B scale : [E][K/128][N/64][64 lanes][4 B] (stage 1) or [E][N/64][K/128][64][4 B] (stage 2)
              byte j -> n16 tile j of 64-column block n64, lane l -> (col l%16, group l//16)

B is exactly AITER's ``shuffle_weight(w, layout=(16, 16))``, so the kernels read
the preshuffled fp4 weights in place; only the e8m0 scales are repacked.

Tile j of 64-column block g covers natural n16 tile ``b_tile(stage, N, g, j)``:
stage 2 is the identity; stage 1 (W13 = [gate; up], I rows each) maps block g to
gate tiles 2g, 2g+1 and up tiles I/16 + 2g, I/16 + 2g + 1, so a wave holds the
gate and up values of the same 32 intermediate columns (SwiGLU in registers).

Requires N % 64 == 0 and K % 128 == 0, which holds for every MiniMax-M3 width
(I = 384 / 768 / 1536, H = 6144): nothing is padded.
"""

import torch

from .gemm import b_tile


def _tile_rows(stage: int, n: int, device=None) -> torch.Tensor:
    """Natural row of each (block, tile, column) slot, i.e. the row order of the scales."""
    g = torch.arange(n // 64, device=device).view(-1, 1, 1)
    j = torch.arange(4, device=device).view(1, -1, 1)
    c = torch.arange(16, device=device).view(1, 1, -1)
    return (b_tile(stage, n, g, j) * 16 + c).reshape(-1)


def pack_scales(stage: int, s: torch.Tensor) -> torch.Tensor:
    """Natural e8m0 scales s [E, N, K/32] uint8 -> the stage's packed B scales."""
    if stage not in (1, 2):
        raise ValueError(f"stage must be 1 or 2, got {stage!r}")
    E, N, KG = s.shape
    K = KG * 32
    if N % 64 or K % 128:
        raise ValueError(f"scales need N % 64 == 0 and K % 128 == 0, got N={N} K={K}")
    s = s[:, _tile_rows(stage, N, s.device)].view(E, N // 64, 4, 16, K // 128, 4)
    if stage == 1:
        # K-step-major: a CTA's 64-column blocks for one step form one 1 KB DMA.
        return s.permute(0, 4, 1, 5, 3, 2).contiguous().view(E, K // 128, N // 64, 256)
    # Column-block-major: for pipes that load B scales per lane every step.
    return s.permute(0, 1, 4, 5, 3, 2).contiguous().view(E, N // 64, K // 128, 256)

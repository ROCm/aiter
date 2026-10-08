# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Scale layout references for optional Opus BMM tuning and validation.

Public BMM callers, including per-row K32 kernels, use ordinary scale tensors.
"""

import torch
import torch.nn.functional as F


def shuffle_scale_mxsk_mpack(
    a_scale: torch.Tensor, b_m: int, sfa_mb: int
) -> torch.Tensor:
    """Pack A-scale M subtiles into adjacent bytes; pad M to b_m."""
    s = a_scale.view(torch.uint8)
    *lead, rows, ksc = s.shape
    com_rep_m = b_m // sfa_mb
    if b_m != sfa_mb * com_rep_m:
        raise ValueError(f"b_m={b_m} must be a multiple of sfa_mb={sfa_mb}")
    if rows % b_m:
        # Pad partial M tiles with E8M0 1.0; all lanes read the scale panel.
        s = F.pad(s, (0, 0, 0, (-rows) % b_m), value=0x7F)
        rows = s.shape[-2]
    s = s.view(*lead, rows // b_m, com_rep_m, sfa_mb, ksc)
    nd = len(lead)
    perm = (*range(nd), nd, nd + 2, nd + 3, nd + 1)
    out = s.permute(*perm).contiguous().view(*lead, rows * ksc)
    return out.view(a_scale.dtype)


def _shuf_kblock_pairs(s: torch.Tensor, K: int) -> tuple[torch.Tensor, int]:
    """Pad K128 blocks to pairs with E8M0 1.0 (0x7F)."""
    K1 = (K + 255) // 256
    if s.shape[-1] < 2 * K1:
        s = F.pad(s, (0, 2 * K1 - s.shape[-1]), value=0x7F)
    return s, K1


def shuffle_scale_a(a_scale: torch.Tensor, K: int, sub: int) -> torch.Tensor:
    """Pack two M subtiles crossed with two K128 blocks into each scale word.

    sub is the M-subtile distance read from _opus_sf_shuf_sub()."""
    s = a_scale.view(torch.uint8)
    *lead, rows, ksc = s.shape
    if ksc != K // 128:
        raise ValueError(
            f"a_scale must be (..., rows, K//128)=(..., {K // 128}); got {tuple(a_scale.shape)}"
        )
    if rows % (2 * sub):
        # Pad partial row blocks because every lane reads whole scale words.
        s = F.pad(s, (0, 0, 0, (-rows) % (2 * sub)), value=0x7F)
        rows = s.shape[-2]
    s, K1 = _shuf_kblock_pairs(s, K)
    nd = len(lead)
    # (..., rows, 2*K1) -> [n1, np, nl, k1, kp] -> [n1, k1, nl, kp, np]
    s = s.view(*lead, rows // (2 * sub), 2, sub, K1, 2)
    perm = (*range(nd), nd, nd + 3, nd + 2, nd + 4, nd + 1)
    return s.permute(*perm).contiguous().view(*lead, -1).view(a_scale.dtype)


def shuffle_scale_b(b_scale: torch.Tensor, N: int, K: int) -> torch.Tensor:
    """Pack K128 scale pairs as duplicated bytes (k, k, k+1, k+1)."""
    s = b_scale.view(torch.uint8)
    *lead, nblk, ksc = s.shape
    if (nblk, ksc) != (N // 128, K // 128):
        raise ValueError(
            f"b_scale must be (..., N//128, K//128)=(..., {N // 128}, {K // 128}); "
            f"got {tuple(b_scale.shape)}"
        )
    s, K1 = _shuf_kblock_pairs(s, K)
    out = s.view(*lead, nblk, K1, 2, 1).expand(*lead, nblk, K1, 2, 2)
    return out.contiguous().view(*lead, -1).view(b_scale.dtype)

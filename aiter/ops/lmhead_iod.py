# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""IOD-local placement of a bf16 [N, K] GEMM weight (the LM head) on gfx1250 (SPX + NPS1), for the a16w16 compute_bound
kernel at BLOCK_N = BLOCK_K = 128 (B_LOC).

A 16 KiB granule is homed on IOD label a14 ^ a17 ^ a20 ^ phase(2 MiB page) (aiter.ops.kv_iod). N-tile t (128 rows)
sits in page t // 4, slot parity s = (t // 2) & 1, units 32 (t & 1) + 2 k + h (K step k, half h = 64 rows), each
unit a dense [64, 128] bf16 block in granule gran(page, s, u). Tile t is then homed on label s ^ phase[page]: every
page holds two tiles of each label, so the labels split the tiles evenly. The kernel gives a workgroup on AID a
(label a ^ label_xor) the rank-th tile of its label's list (mode 2), or of the other label's (mode 3, a control).
"""

import os

import numpy as np
import torch

from aiter.ops import kv_iod

PAGE, GRAN = kv_iod.PAGE, kv_iod.GRAN
_PLACED = {}


def mode() -> int:
    return int(os.environ.get("AITER_LMHEAD_LOC", "0"))


class PlacedWeight:
    def __init__(self, w: torch.Tensor):
        assert w.dim() == 2 and w.dtype == torch.bfloat16 and w.is_contiguous()
        N, K = w.shape
        assert (
            N % 128 == 0 and K % 128 == 0 and K // 128 <= 16
        ), "<= 16 K steps of 128 a tile (32 units of 64)"
        self.N, self.K = N, K
        nt, kt = N // 128, K // 128
        assert nt % 4 == 0, "four N-tiles a page"
        npg = nt // 4
        raw = torch.empty((npg + 1) * PAGE, dtype=torch.uint8, device=w.device)
        off = (-raw.data_ptr()) % PAGE
        self._raw = raw
        self.buf = raw[off : off + npg * PAGE]
        torch.cuda.synchronize()
        r = kv_iod.probe(self.buf)
        if r is None:  # the caller keeps the stock weight
            raise RuntimeError("IOD probe failed")
        self.phase, self.label_xor = r
        t = np.arange(nt)
        p, s, j = t // 4, (t // 2) & 1, t & 1
        k = np.arange(kt)
        h = np.arange(2)
        u = (
            32 * j[:, None, None] + 2 * k[None, :, None] + h[None, None, :]
        )  # [nt, kt, 2]
        g = kv_iod.gran(p[:, None, None], s[:, None, None], u)
        src = (
            w.view(nt, 2, 64, kt, 128).permute(0, 3, 1, 2, 4).reshape(-1, 64 * 128)
        )  # [nt, kt, h] units
        dst = self.buf.view(torch.bfloat16).view(-1, 64 * 128)
        dst[torch.from_numpy(g.reshape(-1)).to(w.device)] = src
        lab = s ^ self.phase[p]
        lists = [t[lab == d] for d in (0, 1)]
        assert len(lists[0]) == len(lists[1]) == nt // 2
        self.half = nt // 2
        self.lists = torch.from_numpy(np.concatenate(lists).astype(np.int32)).to(
            w.device
        )
        self.weight = self.buf.view(torch.bfloat16)
        torch.cuda.synchronize()


def place(w: torch.Tensor) -> PlacedWeight:
    pw = PlacedWeight(w)
    _PLACED[(w.data_ptr(), tuple(w.shape))] = pw
    return pw


def lookup(w: torch.Tensor):
    return _PLACED.get((w.data_ptr(), tuple(w.shape)))

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""IOD-local placement of GEMM1 weights on MI455X (gfx1250, SPX + NPS1).

A 16 KiB granule at ``addr`` is homed on IOD domain
``a14 ^ a17 ^ a20 ^ phase[addr >> 21]``; the per-2 MiB-page phase is
physical, so it is probed once per allocation (``kernels/iod_loc_probe``).
Workgroup b runs on domain ``((b + 3) % 8) >= 4`` when the whole grid runs.

``place_w13`` (``aiter.ops.shuffle.moe_shuffle_weight_iod``, at weight-load
time) re-lays the preshuffled W1 ([E, N/16, K/2*16] bytes) and its
n32k4 scales ([E, N/32, K] bytes) into one buffer:
page 0 holds the int32 phase of every granule page (plus one pad entry),
then for unit (expert e, n-tile nt) on domain nt % 2, local index
lu = e * NT/2 + nt/2, its local granules lu * GPU + j hold
j < 2 KT: B rows [8 (j % 2), 8 (j % 2) + 8) x 2048 B of k-tile j / 2,
j >= 2 KT: the 2 KiB B-scale stages of k-tiles 8 (j - 2 KT) ... + 7.
"""

import logging
import os

import numpy as np
import torch

logger = logging.getLogger("aiter")

GRAN = 16384
PAGE = 2 << 20
LOCAL_PER_PAGE = PAGE // GRAN // 2

_PLACED = {}


def _probe(buf: torch.Tensor, npass: int = 4):
    from .kernels.iod_loc_probe import probe_phase

    torch.cuda.synchronize()
    return probe_phase(buf, npass)


def gran_per_unit(K: int, tile_k: int) -> int:
    kt = (K + tile_k - 1) // tile_k
    return 2 * kt + (kt + 7) // 8


def supported(N, K, tile_n, tile_k) -> bool:
    """Whether ``placed_w1`` / ``b_placed`` cover this GEMM1 tiling."""
    return (
        tile_n * tile_k == 4 * GRAN
        and tile_n % 64 == 0
        and N % (2 * tile_n) == 0
        and K % tile_k == 0
        and gran_per_unit(K, tile_k) <= LOCAL_PER_PAGE
    )


def _place(w, w_scales, E, N, K, tile_n, tile_k):
    assert supported(N, K, tile_n, tile_k)
    nt, kt = N // tile_n, K // tile_k
    bstage, sstage = tile_n * tile_k // 2, tile_n * tile_k // 32
    gpu = gran_per_unit(K, tile_k)
    nsg = gpu - 2 * kt
    nu = E * nt // 2
    npg = (nu * gpu + LOCAL_PER_PAGE - 1) // LOCAL_PER_PAGE

    raw = torch.empty((npg + 2) * PAGE, dtype=torch.uint8, device=w.device)
    off = (-raw.data_ptr()) % PAGE
    buf = raw[off : off + (npg + 1) * PAGE]
    phase, weak, agree = _probe(buf[PAGE:])
    if phase is None:
        logger.warning(
            "[iod_locality] workgroup classes off the IOD domain formula "
            "(agreement %s); W1 %s stays unplaced",
            np.round(agree, 2).tolist(),
            tuple(w.shape),
        )
        return None
    ph = torch.from_numpy(np.append(phase, phase[-1]).astype(np.int32))
    buf[: ph.numel() * 4].copy_(ph.view(torch.uint8))

    # [E, N/16, KT, 2048] -> [E, NT, KT, 16 x 2048] -> two granules per k-tile.
    b = w.reshape(E, nt, tile_n // 16, kt, bstage // (tile_n // 16))
    b = b.permute(0, 1, 3, 2, 4).reshape(E, nt, 2 * kt, GRAN)
    s = w_scales.reshape(-1).view(torch.uint8).reshape(E, nt, tile_n // 32, kt, -1)
    s = s.permute(0, 1, 3, 2, 4).reshape(E, nt, kt, sstage)
    sp = torch.zeros(E, nt, nsg * 8, sstage, dtype=torch.uint8, device=w.device)
    sp[:, :, :kt] = s
    sp = sp.reshape(E, nt, nsg, GRAN)

    bg = buf[PAGE:].view(-1, GRAN)
    k = np.arange(nu * gpu)
    pg, kk = k // LOCAL_PER_PAGE, k % LOCAL_PER_PAGE
    for d in range(2):
        bit = (d ^ phase[pg] ^ (kk >> 2) ^ (kk >> 5)) & 1
        ids = torch.from_numpy(pg * 2 * LOCAL_PER_PAGE + ((kk << 1) | bit))
        ids = ids.to(w.device).view(nu, gpu)
        bg[ids[:, : 2 * kt].reshape(-1)] = b[:, d::2].reshape(-1, GRAN)
        bg[ids[:, 2 * kt :].reshape(-1)] = sp[:, d::2].reshape(-1, GRAN)
    del b, s, sp
    torch.cuda.synchronize()
    buf._iod_raw = raw
    logger.info(
        "[iod_locality] placed W1 %s: %.3f GB, %d pages, weak %d, class agreement >= %.3f",
        tuple(w.shape),
        buf.numel() / 1e9,
        npg,
        weak,
        agree.min(),
    )
    return buf


# The one tiling the placed layout serves (``b_placed``): every 16 KiB
# granule is one wave's B half of a k-tile.
PLACED_TILE_N = 256
PLACED_TILE_K = 256


def enabled() -> bool:
    """``AITER_MOE_LOC=1`` on gfx1250."""
    if os.environ.get("AITER_MOE_LOC", "0").strip() != "1":
        return False
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx1250"


def place_w13(w13: torch.Tensor, w13_scale: torch.Tensor, experts_cnt=None):
    """Build and register the IOD-placed copy of a shuffled GEMM1 weight.

    ``w13`` / ``w13_scale`` are the outputs of ``moe_shuffle_weight`` /
    ``moe_shuffle_scale`` (GUGU, gate_up), which stay valid and in use: tilings
    other than 256 x 256 keep reading them. Call once per layer at weight-load
    time, outside CUDA graph capture. Returns the placed buffer, or None when
    the shape has no placed layout or the probe finds pages whose IOD home
    does not follow the address formula.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("IOD placement must run before CUDA graph capture")
    E = int(experts_cnt or w13.shape[0])
    N = int(w13.shape[-2])
    K = int(w13.shape[-1]) * 2
    if not supported(N, K, PLACED_TILE_N, PLACED_TILE_K):
        return None
    w = w13.detach().contiguous().view(torch.uint8)
    sc = w13_scale.detach().contiguous().reshape(-1).view(torch.int32)
    key = (w.data_ptr(), sc.data_ptr())
    hit = _PLACED.get(key)
    if hit is not None:
        return hit[0]
    buf = _place(w, sc, E, N, K, PLACED_TILE_N, PLACED_TILE_K)
    if buf is None:
        return None
    # Holding the sources keeps their addresses, the key, from being reused.
    _PLACED[key] = (buf, w, sc, N, K)
    return buf


def lookup(w, w_scales, N, K, tile_n, tile_k):
    """The placed buffer registered for (w, w_scales), or None."""
    if not _PLACED or tile_n != PLACED_TILE_N or tile_k != PLACED_TILE_K:
        return None
    hit = _PLACED.get((w.data_ptr(), w_scales.data_ptr()))
    if hit is None or hit[3] != N or hit[4] != K:
        return None
    return hit[0]

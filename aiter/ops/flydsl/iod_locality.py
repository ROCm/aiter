# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""IOD-local placement of MoE GEMM weights on MI455X (gfx1250, SPX + NPS1).

A 16 KiB granule at ``addr`` is homed on IOD domain
``a14 ^ a17 ^ a20 ^ phase[addr >> 21]``; the per-2 MiB-page phase is
physical, so it is probed once per allocation (``kernels/iod_loc_probe``).
Workgroup b runs on domain ``((b + c) % 8) >= 4``, c the rotation of its
launch, which the kernels read at run time (``launch_rotation``).

``place_w13`` / ``place_w2`` (``aiter.ops.shuffle.moe_shuffle_weight_iod``,
at weight-load time) re-lay a preshuffled weight ([E, N/16, K/2*16] bytes)
and its n32k4 scales ([E, N/32, K] bytes) for one tile_n x tile_k layout
tiling (tile_n * tile_k = 64 Ki) into one buffer:
page 0 holds the int32 phase of every granule page (plus one pad entry),
then for unit (expert e, n-tile nt) on domain nt % 2, local index
lu = e * NT/2 + nt/2, its local granules lu * GPU + j (GPU even) hold
j < 2 KT, j = 4 (m / 2) + 2 h + m % 2: B rows [tile_n/32 h, ... + tile_n/32)
x tile_k*8 B of k-tile m (half h, one wave's share),
j >= 2 KT: the 2 KiB B-scale stages of k-tiles 8 (j - 2 KT) ... + 7.
The k-tile pairs 2i, 2i + 1 of a half sit 32 KiB apart, so a GEMM tiled
tile_n/2 x 2 tile_k reads the same buffer (``b_placed=2``) with one TDM load
per wave and k-tile.
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
    g = 2 * kt + (kt + 7) // 8
    return g + g % 2


def supported(N, K, tile_n, tile_k) -> bool:
    """Whether a placed layout exists for this layout tiling."""
    return (
        tile_n * tile_k == 4 * GRAN
        and tile_n % 128 == 0
        and N % (2 * tile_n) == 0
        and K % (2 * tile_k) == 0
        and gran_per_unit(K, tile_k) <= LOCAL_PER_PAGE
    )


def placed_ratio(layout, tile_n, tile_k):
    """``b_placed`` for a GEMM tiled tile_n x tile_k on ``layout``, or 0."""
    for r in (1, 2):
        if (tile_n * r, tile_k) == (layout[0], layout[1] * r):
            return r
    return 0


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
    phase, weak, agree, rotations = _probe(buf[PAGE:])
    if phase is None:
        logger.warning(
            "[iod_locality] workgroup classes off the IOD domain formula "
            "(agreement %s, launch rotations %s); weight %s stays unplaced",
            np.round(agree, 2).tolist(),
            rotations,
            tuple(w.shape),
        )
        return None
    ph = torch.from_numpy(np.append(phase, phase[-1]).astype(np.int32))
    buf[: ph.numel() * 4].copy_(ph.view(torch.uint8))

    # [E, N/16, KT, row] -> [E, NT, KT/2, k-tile 2i + q, half h, 16 KiB]
    # -> granule j = 4 i + 2 h + q.
    b = w.reshape(E, nt, tile_n // 16, kt, bstage // (tile_n // 16))
    b = b.permute(0, 1, 3, 2, 4).reshape(E, nt, kt // 2, 2, 2, GRAN)
    b = b.transpose(3, 4).reshape(E, nt, 2 * kt, GRAN)
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
        "[iod_locality] placed weight %s for %dx%d tiles: %.3f GB, %d pages, weak %d, "
        "class agreement >= %.3f, launch rotations %s",
        tuple(w.shape),
        tile_n,
        tile_k,
        buf.numel() / 1e9,
        npg,
        weak,
        agree.min(),
        rotations,
    )
    return buf


# The layout tiling of each placed weight: every 16 KiB granule is one wave's
# B half of a k-tile. It serves that tiling and the one with tile_n halved and
# tile_k doubled (``placed_ratio``): GEMM1 runs 256 x 256 at every tuned token
# count, GEMM2 512 x 128 or 256 x 256.
W13_TILE = (256, 256)
W2_TILE = (512, 128)


def enabled() -> bool:
    """``AITER_MOE_LOC=1`` on gfx1250."""
    if os.environ.get("AITER_MOE_LOC", "0").strip() != "1":
        return False
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx() == "gfx1250"


def _register(weight, scale, experts_cnt, tile):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("IOD placement must run before CUDA graph capture")
    E = int(experts_cnt or weight.shape[0])
    N = int(weight.shape[-2])
    K = int(weight.shape[-1]) * 2
    if not supported(N, K, *tile):
        return None
    w = weight.detach().contiguous().view(torch.uint8)
    sc = scale.detach().contiguous().reshape(-1).view(torch.int32)
    key = (w.data_ptr(), sc.data_ptr())
    hit = _PLACED.get(key)
    if hit is not None:
        return hit[0]
    buf = _place(w, sc, E, N, K, *tile)
    if buf is None:
        return None
    # Holding the sources keeps their addresses, the key, from being reused.
    _PLACED[key] = (buf, w, sc, N, K, tile)
    return buf


def place_w13(w13: torch.Tensor, w13_scale: torch.Tensor, experts_cnt=None):
    """Build and register the IOD-placed copy of a shuffled GEMM1 weight.

    ``w13`` / ``w13_scale`` are the outputs of ``moe_shuffle_weight`` /
    ``moe_shuffle_scale`` (GUGU, gate_up), which stay valid and in use: tilings
    other than 256 x 256 keep reading them. Call once per layer at weight-load
    time, outside CUDA graph capture. Returns the placed buffer, or None when
    the shape has no placed layout or the probe finds pages whose IOD home
    does not follow the address formula.
    """
    return _register(w13, w13_scale, experts_cnt, W13_TILE)


def place_w2(w2: torch.Tensor, w2_scale: torch.Tensor, experts_cnt=None):
    """``place_w13`` for the shuffled GEMM2 weight, laid out for 512 x 128 tiles
    (served to 256 x 256 as well)."""
    return _register(w2, w2_scale, experts_cnt, W2_TILE)


def placed_tile(w, w_scales):
    """Layout (tile_n, tile_k) of the placed copy registered for (w, w_scales), or None."""
    if not _PLACED:
        return None
    hit = _PLACED.get((w.data_ptr(), w_scales.data_ptr()))
    return None if hit is None else hit[5]


def lookup(w, w_scales, N, K, tile_n, tile_k):
    """(placed buffer, ``b_placed``) registered for (w, w_scales) that serves a
    tile_n x tile_k GEMM, or (None, 0)."""
    if not _PLACED:
        return None, 0
    hit = _PLACED.get((w.data_ptr(), w_scales.data_ptr()))
    if hit is None or hit[3:5] != (N, K):
        return None, 0
    r = placed_ratio(hit[5], tile_n, tile_k)
    return (hit[0], r) if r else (None, 0)

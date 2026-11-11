# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""IOD-local placement of MoE GEMM weights on gfx1250 (SPX + NPS1).

A 16 KiB granule at ``addr`` is homed on IOD domain
``a14 ^ a17 ^ a20 ^ phase[addr >> 21]``; the per-2 MiB-page phase is
physical, so it is probed once per allocation (``kernels/iod_loc_probe``).
Workgroup b runs on domain ``((b + c) % 8) >= 4``, c the rotation of its
launch, which the kernels read at run time (``launch_rotation``).

``place_w13`` / ``place_w2`` (``aiter.ops.shuffle.moe_shuffle_weight_iod``,
at weight-load time) re-lay a preshuffled weight ([E, N/16, K/2*16] bytes)
and its n32k4 scales ([E, N/32, K] bytes) in their own storage, for one
tile_n x tile_k layout tiling (tile_n * tile_k = 64 Ki). Each storage is
whole 32 KiB granule pairs, 64 KiB aligned, q0 pairs past a page start; a
pair has one granule on each domain, so local granule g of domain d is the
storage's granule 2 g + bit, bit = d ^ phase[q / 64] ^ (k >> 2) ^ (k >> 5)
for q = g + q0, k = q % 64.
Unit (expert e, n-tile nt) is on domain nt % 2 with local index
lu = e * NT/2 + nt/2. In the weight, its local granules lu * 2 KT + j,
j = 4 (m / 2) + 2 h + m % 2, hold B rows [tile_n/32 h, ... + tile_n/32)
x tile_k*8 B of k-tile m (half h, one wave's share). In the scale, its
2 KiB B-scale stages of k-tiles m sit at byte (lu * KT + m) * 2 KiB of the
domain's local granules, back to back. A small int32 table, which the
kernels take as ``arg_loc_phase``, holds the weight's q0, the scale's q0,
the index of the scale's phases, then the phase of every page the weight
touches and of every page the scale touches (each list padded by one entry).
The k-tile pairs 2i, 2i + 1 of a half sit 32 KiB apart (q0 even keeps
them in pairs 2c, 2c + 1, which share a17 and a20), so a GEMM tiled
tile_n/2 x 2 tile_k reads the same layout (``b_placed=2``) with one TDM load
per wave and k-tile. Those two tilings are the only readers of a placed
weight: ``lookup`` raises for any other.
"""

import logging
import os

import numpy as np
import torch

logger = logging.getLogger("aiter")

GRAN = 16384
PAGE = 2 << 20
LOCAL_PER_PAGE = PAGE // GRAN // 2
SC_STAGE = 2048

# Placed weight address -> (phase table, weight, scale, N, K, layout tile).
# Holding the tensors keeps their addresses, the key, from being reused.
_PLACED = {}


def supported(N, K, tile_n, tile_k) -> bool:
    """Whether a placed layout exists for this layout tiling."""
    kt = K // tile_k
    return (
        tile_n * tile_k == 4 * GRAN
        and tile_n % 128 == 0
        and N % (2 * tile_n) == 0
        and K % (2 * tile_k) == 0
        and 2 * kt <= LOCAL_PER_PAGE
    )


def placed_ratio(layout, tile_n, tile_k):
    """``b_placed`` for a GEMM tiled tile_n x tile_k on ``layout``, or 0."""
    for r in (1, 2):
        if (tile_n * r, tile_k) == (layout[0], layout[1] * r):
            return r
    return 0


def _span(t: torch.Tensor):
    """Bytes (lo, hi) of the device allocation holding ``t``: the probe may
    read them. ``t``'s own bytes when the runtime does not say."""
    import ctypes

    nbytes = t.numel() * t.element_size()
    own = (t.data_ptr(), t.data_ptr() + nbytes)
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        base, size = ctypes.c_void_p(), ctypes.c_size_t()
        err = hip.hipMemGetAddressRange(
            ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(t.data_ptr())
        )
    except (OSError, AttributeError):
        return own
    if err or not base.value or base.value + size.value < own[1]:
        return own
    return base.value, base.value + size.value


def _storage_unfit(t: torch.Tensor):
    """Why ``t``'s bytes cannot be placed in place, or None."""
    from .kernels.iod_loc_probe import probe_windows

    nbytes = t.numel() * t.element_size()
    if t.data_ptr() % (4 * GRAN):
        return f"starts {t.data_ptr() % (4 * GRAN)} B past a 64 KiB boundary"
    if nbytes % (2 * GRAN):
        return f"{nbytes} B is not whole 32 KiB granule pairs"
    if probe_windows(t.data_ptr(), nbytes, *_span(t)) is None:
        return "a page it touches has no 256 KiB of its allocation to probe"
    return None


def _probe(t: torch.Tensor, npass: int = 4):
    from .kernels.iod_loc_probe import probe_phase

    torch.cuda.synchronize()
    return probe_phase(t.view(torch.uint8).view(-1), npass, _span(t))


def _scatter(dst: torch.Tensor, units: torch.Tensor, phase, q0) -> None:
    """Write ``units`` [2, n, 16 KiB] (domain, local granule) into ``dst``'s
    granules; ``q0`` granule pairs precede ``dst`` in its first page."""
    n = units.shape[1]
    g = np.arange(n)
    q = g + q0
    pg, k = q // LOCAL_PER_PAGE, q % LOCAL_PER_PAGE
    dg = dst.view(-1, GRAN)
    for d in range(2):
        bit = (d ^ phase[pg] ^ (k >> 2) ^ (k >> 5)) & 1
        ids = torch.from_numpy((g << 1) | bit)
        dg[ids.to(dst.device)] = units[d]


def _place(w, w_scales, E, N, K, tile_n, tile_k):
    """Re-lay ``w`` / ``w_scales`` (contiguous uint8) in place; their phase table, or None."""
    nt, kt = N // tile_n, K // tile_k
    bstage = tile_n * tile_k // 2
    probes = []
    for t in (w, w_scales):
        phase, weak, agree, rotations = _probe(t)
        if phase is None:
            logger.warning(
                "[iod_locality] workgroup classes off the IOD domain formula "
                "(agreement %s, launch rotations %s); weight %s stays unplaced",
                np.round(agree, 2).tolist(),
                rotations,
                tuple(w.shape),
            )
            return None
        probes.append((phase, weak, agree, rotations))

    # [E, NT/2, d, half h, rows, KT/2, q, row] -> [d, E, NT/2, KT/2, h, q, 16 KiB]:
    # local granule j = 4 i + 2 h + q of unit (e, nt/2). The one reshape copies,
    # so w can take the bytes back.
    rows = tile_n // 32
    qb, qs = ((t.data_ptr() % PAGE) // (2 * GRAN) for t in (w, w_scales))
    b = w.reshape(E, nt // 2, 2, 2, rows, kt // 2, 2, bstage // (2 * rows))
    b = b.permute(2, 0, 1, 5, 3, 6, 4, 7).reshape(2, -1, GRAN)
    _scatter(w, b, probes[0][0], qb)
    del b
    # [E, NT/2, d, rows, KT, row] -> [d, E, NT/2, KT, 2 KiB stage].
    s = w_scales.reshape(E, nt // 2, 2, rows, kt, -1)
    s = s.permute(2, 0, 1, 4, 3, 5).reshape(2, -1, GRAN)
    _scatter(w_scales, s, probes[1][0], qs)
    del s
    pb, ps = (np.append(p[0], p[0][-1]) for p in probes)
    table = np.concatenate([[qb, qs, 3 + pb.size], pb, ps]).astype(np.int32)
    torch.cuda.synchronize()
    logger.info(
        "[iod_locality] placed weight %s in place for %dx%d tiles: %d + %d pages, "
        "weak %d + %d, class agreement >= %.3f, launch rotations %s",
        tuple(w.shape),
        tile_n,
        tile_k,
        probes[0][0].size,
        probes[1][0].size,
        probes[0][1],
        probes[1][1],
        min(p[2].min() for p in probes),
        probes[0][3],
    )
    return torch.from_numpy(table).to(w.device)


# The layout tiling of each placed weight: every 16 KiB granule is one wave's
# B half of a k-tile. It serves that tiling and the one with tile_n halved and
# tile_k doubled (``placed_ratio``): GEMM1 runs 256 x 256 at every tuned token
# count, GEMM2 512 x 128 or 256 x 256.
W13_TILE = (256, 256)
W2_TILE = (512, 128)


def mode() -> int:
    """``AITER_MOE_LOC`` on gfx1250, 0 on every other arch.

    Every nonzero mode places the weights at load time; the grouped GEMM then
    runs 1 the domain-parity persistent schedule (IOD-local B), 2 its
    remote-only control, 3 the stock schedule.
    """
    m = os.environ.get("AITER_MOE_LOC", "0").strip() or "0"
    if m not in ("0", "1", "2", "3"):
        raise ValueError(f"AITER_MOE_LOC must be 0, 1, 2 or 3, got {m!r}")
    if m == "0":
        return 0
    from aiter.jit.utils.chip_info import get_gfx

    return int(m) if get_gfx() == "gfx1250" else 0


def _register(weight, scale, experts_cnt, tile):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("IOD placement must run before CUDA graph capture")
    E = int(experts_cnt or weight.shape[0])
    N = int(weight.shape[-2])
    K = int(weight.shape[-1]) * 2
    if weight.data_ptr() in _PLACED:
        return True
    why = None
    if not supported(N, K, *tile):
        why = f"no placed layout for {tile[0]}x{tile[1]} tiles"
    elif not (weight.is_contiguous() and scale.is_contiguous()):
        why = "weight or scale is not contiguous"
    else:
        for name, t in (("weight", weight), ("scale", scale)):
            unfit = _storage_unfit(t)
            if unfit:
                why = f"its {name} {unfit}"
                break
    if why:
        logger.warning(
            "[iod_locality] weight %s stays unplaced: %s", tuple(weight.shape), why
        )
        return False
    table = _place(
        weight.detach().view(torch.uint8),
        scale.detach().view(torch.uint8),
        E,
        N,
        K,
        *tile,
    )
    if table is None:
        return False
    _PLACED[weight.data_ptr()] = (table, weight, scale, N, K, tile)
    return True


def place_w13(w13: torch.Tensor, w13_scale: torch.Tensor, experts_cnt=None) -> bool:
    """Re-lay a shuffled GEMM1 weight and its scale IOD-placed, in place.

    ``w13`` / ``w13_scale`` are the outputs of ``moe_shuffle_weight`` /
    ``moe_shuffle_scale`` (GUGU, gate_up). On success both keep their storage,
    shape and dtype but hold the placed layout, which only the grouped GEMM /
    S1F tilings that ``lookup`` serves can read. Call once per layer at
    weight-load time, outside CUDA graph capture. Returns False, leaving both
    untouched, when the shape has no placed layout, a storage is not 64 KiB
    aligned whole granule pairs, or the probe finds pages whose IOD home does
    not follow the address formula.
    """
    return _register(w13, w13_scale, experts_cnt, W13_TILE)


def place_w2(w2: torch.Tensor, w2_scale: torch.Tensor, experts_cnt=None) -> bool:
    """``place_w13`` for the shuffled GEMM2 weight, laid out for 512 x 128 tiles
    (served to 256 x 256 as well)."""
    return _register(w2, w2_scale, experts_cnt, W2_TILE)


def placed_tile(w):
    """Layout (tile_n, tile_k) of ``w`` if it was placed, else None."""
    if not _PLACED:
        return None
    hit = _PLACED.get(w.data_ptr())
    return None if hit is None else hit[5]


def pin_tile(w, tile_n=None):
    """(tile_n, tile_k) a GEMM reading a placed ``w`` runs: the halved layout
    tiling when the tuned tile_n already matches it, else the layout tiling.
    None if ``w`` is not placed."""
    layout = placed_tile(w)
    if layout is None:
        return None
    ln, lk = layout
    return (ln // 2, lk * 2) if tile_n == ln // 2 else (ln, lk)


def lookup(w, w_scales, N, K, tile_n, tile_k):
    """(phase table, ``b_placed``) of ``w`` for a tile_n x tile_k GEMM, or
    (None, 0) when ``w`` was not placed. Raises when ``w`` was placed but this
    GEMM cannot read it."""
    if not _PLACED:
        return None, 0
    hit = _PLACED.get(w.data_ptr())
    if hit is None:
        return None, 0
    table, _, scale, pn, pk, layout = hit
    r = placed_ratio(layout, tile_n, tile_k)
    if scale.data_ptr() != w_scales.data_ptr() or (pn, pk) != (N, K) or not r:
        raise ValueError(
            f"[iod_locality] weight {tuple(w.shape)} is IOD-placed for N={pn} "
            f"K={pk} {layout[0]}x{layout[1]} tiles; a N={N} K={K} "
            f"{tile_n}x{tile_k} GEMM (or a different scale tensor) cannot read it"
        )
    return table, r

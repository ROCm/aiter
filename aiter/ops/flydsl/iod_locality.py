# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""IOD-local placement of GEMM1 weights on MI455X (gfx1250, SPX + NPS1).

A 16 KiB granule at ``addr`` is homed on IOD domain
``a14 ^ a17 ^ a20 ^ phase[addr >> 21]``; the per-2 MiB-page phase is
physical, so it is probed once per allocation (``iod_loc_probe.cpp``, built
with hipcc on first use). Workgroup b
runs on domain ``((b + 3) % 8) >= 4``.

``placed_w1`` re-lays the preshuffled W1 ([E, N/16, K/2*16] bytes) and its
n32k4 scales ([E, N/32, K] bytes) into one buffer:
page 0 holds the int32 phase of every granule page (plus one pad entry),
then for unit (expert e, n-tile nt) on domain nt % 2, local index
lu = e * NT/2 + nt/2, its local granules lu * GPU + j hold
j < 2 KT: B rows [8 (j % 2), 8 (j % 2) + 8) x 2048 B of k-tile j / 2,
j >= 2 KT: the 2 KiB B-scale stages of k-tiles 8 (j - 2 KT) ... + 7.
"""

import ctypes
import os
import subprocess

import numpy as np
import torch

GRAN = 16384
PAGE = 2 << 20
LOCAL_PER_PAGE = PAGE // GRAN // 2

_lib = None
_PLACED = {}


def _lib_path() -> str:
    path = os.environ.get("AITER_IOD_LOC_LIB")
    if path:
        return path
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iod_loc_probe.cpp")
    out_dir = os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "aiter"
    )
    path = os.path.join(out_dir, "libiod_loc_probe.so")
    if not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(src):
        os.makedirs(out_dir, exist_ok=True)
        tmp = f"{path}.{os.getpid()}"
        subprocess.run(
            ["hipcc", "-O3", "--offload-arch=gfx1250", "-shared", "-fPIC", src, "-o", tmp],
            check=True,
        )
        os.replace(tmp, path)
    return path


def _probe(buf: torch.Tensor, npass: int = 4) -> np.ndarray:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(_lib_path())
        _lib.loc_probe.restype = ctypes.c_int
        _lib.loc_probe.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_int,
        ]
    assert buf.data_ptr() % PAGE == 0
    npg = buf.numel() // PAGE
    phase = np.zeros(npg, dtype=np.uint8)
    torch.cuda.synchronize()
    weak = _lib.loc_probe(ctypes.c_void_p(buf.data_ptr()), npg, phase.ctypes.data, npass)
    if weak < 0:
        raise RuntimeError(f"IOD phase probe failed ({weak})")
    return phase, weak


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


def placed_w1(w, w_scales, n_experts, N, K, tile_n, tile_k):
    """Placed copy of (w, w_scales) for ``b_placed`` GEMM1, built once per weight."""
    key = (w.data_ptr(), w_scales.data_ptr(), n_experts, N, K, tile_n, tile_k)
    hit = _PLACED.get(key)
    if hit is not None:
        return hit
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("IOD placement must run before CUDA graph capture")
    assert supported(N, K, tile_n, tile_k)
    E, nt, kt = n_experts, N // tile_n, K // tile_k
    bstage, sstage = tile_n * tile_k // 2, tile_n * tile_k // 32
    gpu = gran_per_unit(K, tile_k)
    nsg = gpu - 2 * kt
    nu = E * nt // 2
    npg = (nu * gpu + LOCAL_PER_PAGE - 1) // LOCAL_PER_PAGE

    raw = torch.empty((npg + 2) * PAGE, dtype=torch.uint8, device=w.device)
    off = (-raw.data_ptr()) % PAGE
    buf = raw[off : off + (npg + 1) * PAGE]
    phase, weak = _probe(buf[PAGE:])
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
    print(
        f"[iod_locality] placed W1 {tuple(w.shape)}: {buf.numel() / 1e9:.3f} GB, "
        f"{npg} pages, weak {weak}",
        flush=True,
    )
    _PLACED[key] = buf
    return buf

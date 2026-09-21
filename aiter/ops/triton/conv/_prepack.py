# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Weight/input repacking for convolution kernels.

Several conv kernels don't consume the raw OIHW weight (or NCHW input) layout
directly — they need it reshaped into a kernel-local format for coalesced loads:
K-major padded tiles for the 1x1/general GEMM, [K_out, 9, C_pad] for the 3x3
kernels, channel-blocked NCHWc for the cblocked path, and the G·g·Gᵀ filter
transform for Winograd F(4x4,3x3). These packs are pure functions of the weight
tensor, so the results are LRU-cached keyed on (storage ptr, shape, dtype,
block, version): a weight repacks once and every later call with the same
weight is a cache hit, making the steady-state repack cost negligible.
"""

import os
from collections import OrderedDict

import torch

from aiter.ops.triton.conv._launch import (
    _launch_ncdhw_to_cblocked,
    _launch_nchw_to_cblocked,
)
from aiter.ops.triton.conv._utils import (
    BLOCK_K,
    _winograd_transform_storage_dtype,
)

_DEFAULT_PACK_CACHE_MAXSIZE = 256


def _read_pack_cache_maxsize(default: int = _DEFAULT_PACK_CACHE_MAXSIZE) -> int:
    """Read AITER_TRITON_CONV_PACK_CACHE_SIZE, falling back to `default` for
    missing, non-integer, or non-positive values."""
    raw = os.environ.get("AITER_TRITON_CONV_PACK_CACHE_SIZE")
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except ValueError:
        return default
    return value if value > 0 else default


_PACK_CACHE_MAXSIZE = _read_pack_cache_maxsize()


class _LRUPackCache:
    """Bounded LRU for weight prepacks. Stores (src_tensor, item) — the
    strong ref to src keeps storage alive so the storage_ptr in the key
    cannot be reused by a different tensor while this entry lives."""

    def __init__(self, maxsize: int = _PACK_CACHE_MAXSIZE):
        self._d: OrderedDict[tuple, tuple] = OrderedDict()
        self._max = max(1, maxsize)

    def get(self, key):
        entry = self._d.get(key)
        if entry is None:
            return None
        self._d.move_to_end(key)
        return entry

    def put(self, key, src, item):
        self._d[key] = (src, item)
        self._d.move_to_end(key)
        while len(self._d) > self._max:
            self._d.popitem(last=False)

    def clear(self):
        self._d.clear()


_PACK_CACHE = _LRUPackCache()
_PACK_CACHE_3x3 = _LRUPackCache()
_PACK_CACHE_WINOGRAD_F4X3 = _LRUPackCache()

# The 3-D packed layouts use independent cache instances to prevent collisions
# when source tensor metadata matches a cache entry for another packed format.
_PACK_CACHE_3D_GENERAL = _LRUPackCache()
_PACK_CACHE_3D_3X3X3 = _LRUPackCache()
_PACK_CACHE_3D_WINOGRAD_HW = _LRUPackCache()


def clear_conv2d_weight_pack_caches() -> None:
    """Release cached Conv2D weight packs, primarily for synthetic sweeps."""
    _PACK_CACHE.clear()
    _PACK_CACHE_3x3.clear()
    _PACK_CACHE_WINOGRAD_F4X3.clear()


def clear_conv3d_weight_pack_caches() -> None:
    """Release cached Conv3D weight packs, primarily for synthetic sweeps."""
    _PACK_CACHE_3D_GENERAL.clear()
    _PACK_CACHE_3D_3X3X3.clear()
    _PACK_CACHE_3D_WINOGRAD_HW.clear()


def _pack_cache_key(w: torch.Tensor, block: int) -> tuple:
    """Identity for a packed weight, including aliases and in-place updates."""
    return (
        w.data_ptr(),
        w.device.type,
        w.device.index,
        tuple(w.shape),
        tuple(w.stride()),
        w.dtype,
        w._version,
        block,
    )


def prepack_oihw_to_kmajor(w_oihw: torch.Tensor, block_k: int = BLOCK_K):
    K_out, C, R, S = w_oihw.shape
    K_red = C * R * S
    K_pad = ((K_red + block_k - 1) // block_k) * block_k
    w_rs = w_oihw.reshape(K_out, K_red)
    if K_pad != K_red:
        pad = torch.zeros(
            (K_out, K_pad - K_red), device=w_oihw.device, dtype=w_oihw.dtype
        )
        w_rs = torch.cat([w_rs, pad], dim=1)
    return w_rs.contiguous(), K_pad


def get_or_make_weight_pack(w_oihw: torch.Tensor, block_k: int = BLOCK_K):
    key = _pack_cache_key(w_oihw, block_k)
    entry = _PACK_CACHE.get(key)
    if entry is not None:
        return entry[1]
    item = prepack_oihw_to_kmajor(w_oihw, block_k)
    _PACK_CACHE.put(key, w_oihw, item)
    return item


def prepack_oihw_to_3x3(w_oihw: torch.Tensor, block_c: int = BLOCK_K):
    """Pack weights as [K_out, 9, C_pad] for 3x3 specialized kernel."""
    K_out, C, R, S = w_oihw.shape
    assert R == 3 and S == 3
    C_pad = ((C + block_c - 1) // block_c) * block_c
    w_rs = w_oihw.reshape(K_out, C, 9).permute(0, 2, 1).contiguous()  # [K_out, 9, C]
    if C_pad != C:
        pad = torch.zeros(
            (K_out, 9, C_pad - C), device=w_oihw.device, dtype=w_oihw.dtype
        )
        w_rs = torch.cat([w_rs, pad], dim=2)
    return w_rs.contiguous(), C_pad


def get_or_make_weight_pack_3x3(w_oihw: torch.Tensor, block_c: int = BLOCK_K):
    key = _pack_cache_key(w_oihw, block_c)
    cached = _PACK_CACHE_3x3.get(key)
    if cached is not None:
        return cached[1]
    item = prepack_oihw_to_3x3(w_oihw, block_c)
    _PACK_CACHE_3x3.put(key, w_oihw, item)
    return item


def prepack_nchw_to_cblocked(x: torch.Tensor, block_c: int = BLOCK_K):
    """Materialize NCHW as [N, C_blocks, H, W, Cb] in one Triton pass.

    Within each block of Cb channels, data is contiguous (stride=1).
    """
    if not x.is_contiguous():
        x = x.contiguous()
    N, C, H, W = x.shape
    Cb = block_c
    C_blocks = (C + Cb - 1) // Cb
    C_pad = C_blocks * Cb

    x_blocked = torch.empty((N, C_blocks, H, W, Cb), device=x.device, dtype=x.dtype)
    _launch_nchw_to_cblocked(x, x_blocked, N, C, H, W, C_pad, Cb)
    return x_blocked, C_pad


def prepack_winograd_filter_f4x3(w_oihw: torch.Tensor, block_c: int = BLOCK_K):
    """Transform 3x3 filters for Winograd F(4x4,3x3). G @ g @ G^T for each (k,c).
    Input: [K_out, C, 3, 3] fp16.  Output: [36, K_out, C_pad] fp16."""
    K_out, C, R, S = w_oihw.shape
    assert R == 3 and S == 3
    C_pad = ((C + block_c - 1) // block_c) * block_c
    # G matrix (6x3)
    G = torch.tensor(
        [
            [1.0 / 4, 0.0, 0.0],
            [-1.0 / 6, -1.0 / 6, -1.0 / 6],
            [-1.0 / 6, 1.0 / 6, -1.0 / 6],
            [1.0 / 24, 1.0 / 12, 1.0 / 6],
            [1.0 / 24, -1.0 / 12, 1.0 / 6],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=w_oihw.device,
    )

    g = w_oihw.float()  # [K_out, C, 3, 3]
    u = torch.einsum("ij,kcjl,lm->kcim", G, g, G.t())
    u = u.reshape(K_out, C, 36).permute(2, 0, 1).contiguous()
    if C_pad != C:
        pad = torch.zeros(
            (36, K_out, C_pad - C), device=w_oihw.device, dtype=torch.float32
        )
        u = torch.cat([u, pad], dim=2)
    return u.to(w_oihw.dtype).contiguous(), C_pad


def get_or_make_winograd_filter_f4x3(w_oihw: torch.Tensor, block_c: int = BLOCK_K):
    key = _pack_cache_key(w_oihw, block_c)
    cached = _PACK_CACHE_WINOGRAD_F4X3.get(key)
    if cached is not None:
        return cached[1]
    item = prepack_winograd_filter_f4x3(w_oihw, block_c)
    _PACK_CACHE_WINOGRAD_F4X3.put(key, w_oihw, item)
    return item


# ---------------------------------------------------------------------------
# Conv3D packing
# ---------------------------------------------------------------------------


def prepack_oidhw_to_kmajor(w_oidhw: torch.Tensor, block_k: int = BLOCK_K):
    """Pack ``[K_out,C,T,R,S]`` into zero-padded ``[K_out,K_pad]``."""
    K_out, C, T, R, S = w_oidhw.shape
    K_red = C * T * R * S
    K_pad = ((K_red + block_k - 1) // block_k) * block_k
    packed = w_oidhw.reshape(K_out, K_red)
    if K_pad != K_red:
        packed = torch.cat(
            (
                packed,
                torch.zeros(
                    (K_out, K_pad - K_red),
                    device=w_oidhw.device,
                    dtype=w_oidhw.dtype,
                ),
            ),
            dim=1,
        )
    return packed.contiguous(), K_pad


def get_or_make_weight_pack_3d(w_oidhw: torch.Tensor, block_k: int = BLOCK_K):
    key = _pack_cache_key(w_oidhw, block_k)
    cached = _PACK_CACHE_3D_GENERAL.get(key)
    if cached is not None:
        return cached[1]
    packed = prepack_oidhw_to_kmajor(w_oidhw, block_k)
    _PACK_CACHE_3D_GENERAL.put(key, w_oidhw, packed)
    return packed


def prepack_oidhw_to_3x3x3(w_oidhw: torch.Tensor, block_c: int = BLOCK_K):
    """Pack OIDHW weights as ``[K_out,27,C_pad]`` in depth-major tap order."""
    K_out, C, T, R, S = w_oidhw.shape
    if (T, R, S) != (3, 3, 3):
        raise ValueError(f"3x3x3 prepack requires a 3x3x3 weight, got {(T, R, S)}")
    C_pad = ((C + block_c - 1) // block_c) * block_c
    packed = w_oidhw.reshape(K_out, C, 27).permute(0, 2, 1).contiguous()
    if C_pad != C:
        packed = torch.cat(
            (
                packed,
                torch.zeros(
                    (K_out, 27, C_pad - C),
                    device=w_oidhw.device,
                    dtype=w_oidhw.dtype,
                ),
            ),
            dim=2,
        )
    return packed.contiguous(), C_pad


def get_or_make_weight_pack_3x3x3(w_oidhw: torch.Tensor, block_c: int = BLOCK_K):
    key = _pack_cache_key(w_oidhw, block_c)
    cached = _PACK_CACHE_3D_3X3X3.get(key)
    if cached is not None:
        return cached[1]
    packed = prepack_oidhw_to_3x3x3(w_oidhw, block_c)
    _PACK_CACHE_3D_3X3X3.put(key, w_oidhw, packed)
    return packed


def prepack_ncdhw_to_cblocked(x: torch.Tensor, block_c: int = BLOCK_K):
    """Materialize NCDHW as contiguous ``[N,C_blocks,D,H,W,Cb]``."""
    if not x.is_contiguous():
        x = x.contiguous()
    N, C, D, H, W = x.shape
    C_blocks = (C + block_c - 1) // block_c
    C_pad = C_blocks * block_c
    packed = torch.empty(
        (N, C_blocks, D, H, W, block_c), device=x.device, dtype=x.dtype
    )
    _launch_ncdhw_to_cblocked(x, packed, N, C, D, H, W, C_pad, block_c)
    return packed, C_pad


def prepack_winograd_hw_filter_f4x3(w_oidhw: torch.Tensor, block_c: int = BLOCK_K):
    """Apply F(4,3)'s 2-D filter transform to each of three depth taps.

    The result is ``[3,36,K_out,C_pad]``. BF16 weights use FP16 transformed
    storage, reducing the extra rounding introduced by the Winograd domain.
    """
    K_out, C, T, R, S = w_oidhw.shape
    if (T, R, S) != (3, 3, 3):
        raise ValueError(f"Winograd prepack requires 3x3x3, got {(T, R, S)}")
    C_pad = ((C + block_c - 1) // block_c) * block_c
    G = torch.tensor(
        [
            [1.0 / 4, 0.0, 0.0],
            [-1.0 / 6, -1.0 / 6, -1.0 / 6],
            [-1.0 / 6, 1.0 / 6, -1.0 / 6],
            [1.0 / 24, 1.0 / 12, 1.0 / 6],
            [1.0 / 24, -1.0 / 12, 1.0 / 6],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
        device=w_oidhw.device,
    )
    transformed = torch.einsum("ij,kctjl,lm->kctim", G, w_oidhw.float(), G.t())
    transformed = transformed.reshape(K_out, C, 3, 36).permute(2, 3, 0, 1).contiguous()
    if C_pad != C:
        transformed = torch.cat(
            (
                transformed,
                torch.zeros(
                    (3, 36, K_out, C_pad - C),
                    device=w_oidhw.device,
                    dtype=torch.float32,
                ),
            ),
            dim=3,
        )
    return transformed.to(_winograd_transform_storage_dtype(w_oidhw.dtype)), C_pad


def get_or_make_winograd_hw_filter_f4x3(w_oidhw: torch.Tensor, block_c: int = BLOCK_K):
    key = _pack_cache_key(w_oidhw, block_c)
    cached = _PACK_CACHE_3D_WINOGRAD_HW.get(key)
    if cached is not None:
        return cached[1]
    packed = prepack_winograd_hw_filter_f4x3(w_oidhw, block_c)
    _PACK_CACHE_3D_WINOGRAD_HW.put(key, w_oidhw, packed)
    return packed

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations
import os
from collections import OrderedDict
from typing import Dict
import torch

from aiter.ops.triton.conv._utils import BLOCK_K, _storage_ptr

_PACK_CACHE_MAXSIZE = int(os.environ.get("AITER_TRITON_CONV_PACK_CACHE_SIZE", "256"))


class _LRUPackCache:
    """Bounded LRU for weight prepacks. Stores (src_tensor, item) — the
    strong ref to src keeps storage alive so the storage_ptr in the key
    cannot be reused by a different tensor while this entry lives."""

    def __init__(self, maxsize: int = _PACK_CACHE_MAXSIZE):
        self._d: "OrderedDict[tuple, tuple]" = OrderedDict()
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

    def __len__(self):
        return len(self._d)


_PACK_CACHE = _LRUPackCache()
_PACK_CACHE_3x3 = _LRUPackCache()
# Input pack — kept as single-entry dict by design: in real inference each
# layer's input is a unique intermediate activation that won't be reused,
# and the bench clears this per-call to model per-batch repack cost.
_PACK_CACHE_CBLOCKED: Dict = {}
_PACK_CACHE_WINOGRAD_F4X3 = _LRUPackCache()


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
    return w_rs.contiguous(), (K_out, K_pad)


def get_or_make_weight_pack(w_oihw: torch.Tensor, block_k: int = BLOCK_K):
    key = (
        _storage_ptr(w_oihw),
        tuple(w_oihw.shape),
        w_oihw.dtype,
        block_k,
        int(getattr(w_oihw, "_version", 0)),
    )
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
    return w_rs.contiguous(), (K_out, C_pad)


def get_or_make_weight_pack_3x3(w_oihw: torch.Tensor, block_c: int = BLOCK_K):
    key = (
        _storage_ptr(w_oihw),
        tuple(w_oihw.shape),
        w_oihw.dtype,
        block_c,
        int(getattr(w_oihw, "_version", 0)),
    )
    cached = _PACK_CACHE_3x3.get(key)
    if cached is not None:
        return cached[1]
    item = prepack_oihw_to_3x3(w_oihw, block_c)
    _PACK_CACHE_3x3.put(key, w_oihw, item)
    return item


def prepack_nchw_to_cblocked(x: torch.Tensor, block_c: int = BLOCK_K):
    """Pack NCHW input into channel-blocked layout [N, C_blocks, H, W, Cb].

    Within each block of Cb channels, data is contiguous (stride=1).
    """
    N, C, H, W = x.shape
    Cb = block_c
    C_blocks = (C + Cb - 1) // Cb
    C_pad = C_blocks * Cb

    if C_pad != C:
        x_padded = torch.zeros((N, C_pad, H, W), device=x.device, dtype=x.dtype)
        x_padded[:, :C, :, :] = x
    else:
        x_padded = x

    x_blocked = (
        x_padded.reshape(N, C_blocks, Cb, H, W).permute(0, 1, 3, 4, 2).contiguous()
    )
    return x_blocked, C_pad


def get_or_make_input_pack_cblocked(x: torch.Tensor, block_c: int = BLOCK_K):
    key = (
        _storage_ptr(x),
        tuple(x.shape),
        x.dtype,
        block_c,
        int(getattr(x, "_version", 0)),
    )
    cached = _PACK_CACHE_CBLOCKED.get(key)
    if cached is not None:
        src_ref, item = cached
        if src_ref is not None and _storage_ptr(src_ref) == key[0]:
            return item
    item = prepack_nchw_to_cblocked(x, block_c)
    _PACK_CACHE_CBLOCKED.clear()
    _PACK_CACHE_CBLOCKED[key] = (x, item)
    return item


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
    return u.to(w_oihw.dtype).contiguous(), (K_out, C_pad)


def get_or_make_winograd_filter_f4x3(w_oihw: torch.Tensor, block_c: int = BLOCK_K):
    key = (
        _storage_ptr(w_oihw),
        tuple(w_oihw.shape),
        w_oihw.dtype,
        block_c,
        int(getattr(w_oihw, "_version", 0)),
    )
    cached = _PACK_CACHE_WINOGRAD_F4X3.get(key)
    if cached is not None:
        return cached[1]
    item = prepack_winograd_filter_f4x3(w_oihw, block_c)
    _PACK_CACHE_WINOGRAD_F4X3.put(key, w_oihw, item)
    return item

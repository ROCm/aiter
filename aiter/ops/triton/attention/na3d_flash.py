# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""3D neighborhood flash attention launcher.

Each query token at grid position (t, h, w) attends to the KT x KH x KW neighborhood
centered on it, with inward border-shift so every window has exactly KT x KH x KW keys.

Usage::

    from aiter.ops.triton.attention.na3d_flash import na3d_flash_attn

    out = na3d_flash_attn(q, k, v, kernel_size=(11, 11, 11))
    # q, k, v: (B, T, H, W, NH, HD) bfloat16, Q already scaled by head_dim**-0.5
    # out    : (B, T, H, W, NH, HD) bfloat16
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F
import triton

from aiter.ops.triton._triton_kernels.attention.na3d_flash import _na3d_flash_fwd
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


# ---------------------------------------------------------------------------
# Exact per-query neighborhood SDPA (generic; used for rightmost-block correction)
# ---------------------------------------------------------------------------


def _window_bounds_1d(length: int, kernel: int) -> tuple[list[int], list[int]]:
    """Inward-shifted window (start, end) for each position along one axis."""
    lo = length - kernel
    half = kernel // 2
    starts: list[int] = []
    ends: list[int] = []
    for i in range(length):
        s = min(max(i - half, 0), lo)
        starts.append(s)
        ends.append(s + kernel)
    return starts, ends


@functools.lru_cache(maxsize=256)
def _na3d_sdpa_mask(
    rel_bounds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Cached additive ``[1, 1, Nq, Nk]`` mask for one tile-geometry group.

    Masks are reused across repeated calls with the same geometry (e.g. all
    8 diff_block layers at W=192 share the same rel_bounds).
    """
    bools = []
    for starts, ends in rel_bounds:
        st = torch.tensor(starts, device=device)
        en = torch.tensor(ends, device=device)
        kj = torch.arange(max(ends), device=device)
        bools.append((kj[None, :] >= st[:, None]) & (kj[None, :] < en[:, None]))
    visible = (
        bools[0][:, None, None, :, None, None]
        & bools[1][None, :, None, None, :, None]
        & bools[2][None, None, :, None, None, :]
    )
    nq = math.prod(visible.shape[:3])
    nk = math.prod(visible.shape[3:])
    mask = torch.zeros((nq, nk), dtype=dtype, device=device)
    mask.masked_fill_(~visible.reshape(nq, nk), torch.finfo(dtype).min)
    return mask.reshape(1, 1, nq, nk)


def _na3d_sdpa_exact(
    q_right: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
    w_offset: int = 0,
) -> torch.Tensor:
    """Exact per-query 3D neighborhood attention via grouped tiled SDPA.

    When called with the full q tensor and w_offset=0, computes exact neighborhood
    attention for all W positions. When called with a W sub-slice and non-zero
    w_offset, K/V are drawn from the full k/v tensors using global W coordinates -
    used by correct_rightmost_block.

    Args:
        q_right    : ``(B, T, H, W_sub, NH, HD)`` bfloat16, Q pre-scaled.
        k, v       : ``(B, T, H, W_full, NH, HD)`` bfloat16.
        kernel_size: ``(KT, KH, KW)`` neighborhood window.
        w_offset   : Global W start of q_right; 0 for a full-tensor call.

    Returns:
        ``(B, T, H, W_sub, NH, HD)`` bfloat16.
    """
    B, T, H, W_sub, NH, HD = q_right.shape
    W_full = k.shape[3]
    KT, KH, KW = kernel_size
    device = q_right.device

    bt = _window_bounds_1d(T, min(KT, T))
    bh = _window_bounds_1d(H, min(KH, H))
    bw = _window_bounds_1d(W_full, min(KW, W_full))  # global W bounds

    # Pick tile sizes based on q_right,
    # but estimate K/V union sizes using the full W.
    eff_kt, eff_kh = min(KT, T), min(KH, H)
    eff_kw = min(KW, W_full)

    def _cost(ts: list[int]) -> int:
        tile_t, tile_h, tile_w = ts
        nq = tile_t * tile_h * tile_w
        nk_t = min(T, tile_t + eff_kt - 1)
        nk_h = min(H, tile_h + eff_kh - 1)
        nk_w = min(W_full, tile_w + eff_kw - 1)
        return nq * nk_t * nk_h * nk_w

    tiles = [T, H, W_sub]
    while _cost(tiles) > 2**25 and max(tiles) > 1:
        i = max(range(3), key=lambda a: tiles[a] / [eff_kt, eff_kh, eff_kw][a])
        if tiles[i] <= 1:
            break
        tiles[i] = max(1, (tiles[i] + 1) // 2)
    tile_t, tile_h, tile_w = tiles

    groups: dict = {}
    for t0 in range(0, T, tile_t):
        t1 = min(t0 + tile_t, T)
        rt0, rt1 = bt[0][t0], bt[1][t1 - 1]
        rel_t = (
            tuple(s - rt0 for s in bt[0][t0:t1]),
            tuple(e - rt0 for e in bt[1][t0:t1]),
        )
        for h0 in range(0, H, tile_h):
            h1 = min(h0 + tile_h, H)
            rh0, rh1 = bh[0][h0], bh[1][h1 - 1]
            rel_h = (
                tuple(s - rh0 for s in bh[0][h0:h1]),
                tuple(e - rh0 for e in bh[1][h0:h1]),
            )
            for w0 in range(0, W_sub, tile_w):
                w1 = min(w0 + tile_w, W_sub)
                w0g, w1g = w_offset + w0, w_offset + w1 - 1
                rw0, rw1 = bw[0][w0g], bw[1][w1g]
                rel_w = (
                    tuple(s - rw0 for s in bw[0][w0g : w1g + 1]),
                    tuple(e - rw0 for e in bw[1][w0g : w1g + 1]),
                )
                groups.setdefault((rel_t, rel_h, rel_w), []).append(
                    (
                        (slice(t0, t1), slice(h0, h1), slice(w0, w1)),
                        (slice(rt0, rt1), slice(rh0, rh1), slice(rw0, rw1)),
                    )
                )

    out = torch.empty((B, T, H, W_sub, NH, HD), device=device, dtype=v.dtype)
    kv_budget = 2**28
    for rel, tiles_list in groups.items():
        mask = _na3d_sdpa_mask(rel, q_right.dtype, device)
        nq, nk = mask.shape[2], mask.shape[3]
        g_max = max(1, kv_budget // max(1, B * NH * nk * HD * 2))
        qs0, _ = tiles_list[0]
        tq = qs0[0].stop - qs0[0].start
        th_sz = qs0[1].stop - qs0[1].start
        tw_sz = qs0[2].stop - qs0[2].start
        for c0 in range(0, len(tiles_list), g_max):
            chunk = tiles_list[c0 : c0 + g_max]
            g = len(chunk)
            q_s = torch.stack([q_right[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
            k_s = torch.stack([k[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            v_s = torch.stack([v[:, rs[0], rs[1], rs[2]] for _, rs in chunk])
            q_s = q_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * B, NH, nq, HD)
            k_s = k_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * B, NH, nk, HD)
            v_s = v_s.permute(0, 1, 5, 2, 3, 4, 6).reshape(g * B, NH, nk, HD)
            o = F.scaled_dot_product_attention(q_s, k_s, v_s, attn_mask=mask, scale=1.0)
            o = o.view(g, B, NH, tq, th_sz, tw_sz, HD).permute(0, 1, 3, 4, 5, 2, 6)
            for i, (qs, _) in enumerate(chunk):
                out[:, qs[0], qs[1], qs[2]] = o[i]
    return out


# ---------------------------------------------------------------------------
# Public launcher
# ---------------------------------------------------------------------------


def na3d_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
    correct_rightmost_block: bool = False,
) -> torch.Tensor:
    """Flash-style 3D neighborhood attention.

    Args:
        q, k, v     : ``(B, T, H, W, NH, HD)`` bfloat16, channels-last.
                      Q is expected to be pre-scaled by ``head_dim ** -0.5``.
        kernel_size : ``(KT, KH, KW)`` neighborhood window.
        correct_rightmost_block : When True, overwrite the last W-block
            (``W_last:``) with per-query exact SDPA (via _na3d_sdpa_exact),
            where ``W_last = floor((W - 1) / BLOCK_Q) * BLOCK_Q`` for
            the autotuned BLOCK_Q. The flash kernel loads ``BF16(0)`` for
            out-of-bounds K/V positions in the last W-block (``kv_ok=False``),
            which causes a ~0.002 rounding divergence from AOTriton
            at those positions.  Off by default.

    Returns:
        Output tensor ``(B, T, H, W, NH, HD)`` bfloat16.

    Notes:
        ``W >= 16`` is required so that all queries in a BLOCK_Q=16 program share
        the same (t, h) grid row.  The autotune pruner enforces this.
    """
    B, T, H, W, NH, HD = q.shape
    KT, KH, KW = kernel_size
    SEQ = T * H * W

    _LOGGER.info(
        f"NA3D_FLASH_FWD: q={tuple(q.shape)} kernel=({KT},{KH},{KW}) SEQ={SEQ} HD={HD}"
    )

    assert q.dtype == torch.bfloat16, "na3d_flash_attn: inputs must be bfloat16"
    assert (
        k.dtype == q.dtype and v.dtype == q.dtype
    ), "na3d_flash_attn: q/k/v must share dtype"
    assert (
        k.shape == q.shape and v.shape == q.shape
    ), "na3d_flash_attn: q/k/v must have shape (B, T, H, W, NH, HD)"
    assert (
        KT <= T and KH <= H and KW <= W
    ), f"na3d_flash_attn: kernel_size=({KT},{KH},{KW}) must be <= (T,H,W)=({T},{H},{W})"
    assert HD & (HD - 1) == 0, f"head_dim {HD} must be a power of 2"
    assert W >= 16, f"W={W} is too small; kernel requires W >= BLOCK_Q (default 16)."

    def _flat(t: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, NH, HD) -> (B*NH, SEQ, HD) contiguous."""
        return t.permute(0, 4, 1, 2, 3, 5).reshape(B * NH, SEQ, HD).contiguous()

    q_f, k_f, v_f = _flat(q), _flat(k), _flat(v)
    out_f = torch.empty_like(q_f)

    # Lambda grid: Triton passes the autotuned BLOCK_Q via meta.
    grid = lambda meta: (triton.cdiv(SEQ, meta["BLOCK_Q"]), B * NH)

    _na3d_flash_fwd[grid](
        q_f,
        k_f,
        v_f,
        out_f,
        SEQ * HD,  # stride_bnh
        HD,  # stride_seq
        T,
        H,
        W,
        SEQ,
        HD=HD,
        KT=KT,
        KH=KH,
        KW=KW,
    )

    out = out_f.reshape(B, NH, T, H, W, HD).permute(0, 2, 3, 4, 1, 5).contiguous()

    if correct_rightmost_block:
        # Use the BLOCK_Q actually selected by autotune for this (KT,KH,KW,W) key.
        actual_bq = _na3d_flash_fwd.best_config.kwargs["BLOCK_Q"]
        W_last = ((W - 1) // actual_bq) * actual_bq
        if W_last > 0:
            q_right = q[:, :, :, W_last:, :, :]
            out_right = _na3d_sdpa_exact(q_right, k, v, kernel_size, w_offset=W_last)
            out[:, :, :, W_last:, :, :] = out_right

    return out

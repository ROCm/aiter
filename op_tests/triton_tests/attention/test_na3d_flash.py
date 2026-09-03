# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test and benchmark for 3D neighborhood flash attention (na3d_flash).

Covers representative LTX-2.5 VAE decode shapes plus edge cases that exercise
boundary masking and non-aligned sequence lengths.

Each shape is compared against a pure-PyTorch tiled SDPA reference that implements
the same inward-shifted centered window.  TFLOPS and TB/s are reported per shape.
"""

import argparse
import functools
import math

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.na3d_flash import na3d_flash_attn
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]


# ---------------------------------------------------------------------------
# Pure-PyTorch reference (not timed, not in the table)
# ---------------------------------------------------------------------------
def _window_bounds(length: int, kernel: int) -> tuple[list[int], list[int]]:
    """Per-index (start, end) of the inward-shifted centered neighborhood window."""
    lo = length - kernel
    half = kernel // 2
    starts, ends = [], []
    for i in range(length):
        start = min(max(i - half, 0), lo)
        starts.append(start)
        ends.append(start + kernel)
    return starts, ends


def na3d_sdpa_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
) -> torch.Tensor:
    """FP32 per-query neighborhood attention - ground-truth cross-check only.

    Simple per-query loop used exclusively in ``test_na3d_sdpa_exact_vs_ref``
    to verify that ``_na3d_sdpa_exact`` agrees with this FP32 baseline.
    For routine correctness tests use ``_na3d_sdpa_exact`` (GPU-grouped, fast).

    Args:
        q, k, v     : ``(B, T, H, W, NH, HD)`` bfloat16 - Q pre-scaled.
        kernel_size : ``(KT, KH, KW)``.
    Returns:
        Output ``(B, T, H, W, NH, HD)`` bfloat16.
    """
    B, T, H, W, NH, HD = q.shape
    KT, KH, KW = kernel_size

    bounds_t = _window_bounds(T, KT)
    bounds_h = _window_bounds(H, KH)
    bounds_w = _window_bounds(W, KW)

    out = torch.empty_like(q)
    for t in range(T):
        for h in range(H):
            for w in range(W):
                ts, te = bounds_t[0][t], bounds_t[1][t]
                hs, he = bounds_h[0][h], bounds_h[1][h]
                ws, we = bounds_w[0][w], bounds_w[1][w]
                # q_tok: (B, NH, 1, HD); kv_k/kv_v: (B, NH, K, HD)
                q_tok = q[:, t, h, w, :, :].unsqueeze(2).float()
                kv_k = (
                    k[:, ts:te, hs:he, ws:we, :, :]
                    .reshape(B, -1, NH, HD)
                    .permute(0, 2, 1, 3)
                    .float()
                )
                kv_v = (
                    v[:, ts:te, hs:he, ws:we, :, :]
                    .reshape(B, -1, NH, HD)
                    .permute(0, 2, 1, 3)
                    .float()
                )
                o = F.scaled_dot_product_attention(q_tok, kv_k, kv_v, scale=1.0)
                out[:, t, h, w, :, :] = o.squeeze(2).to(q.dtype)
    return out


# ---------------------------------------------------------------------------
# GPU-grouped exact reference (fast alternative to na3d_sdpa_ref)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _na3d_sdpa_mask(
    rel_bounds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Cached additive ``[1, 1, Nq, Nk]`` mask for one tile-geometry group."""
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
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kernel_size: tuple[int, int, int],
    w_offset: int = 0,
) -> torch.Tensor:
    """Exact per-query 3D neighborhood attention via grouped tiled SDPA.

    Generic GPU-grouped reference function.  With ``w_offset=0`` and full
    ``q/k/v`` this computes exact neighborhood attention for all W positions
    and is equivalent to ``na3d_sdpa_ref`` but GPU-accelerated with cached masks.

    Args:
        q          : ``(B, T, H, W_sub, NH, HD)`` bfloat16, Q pre-scaled.
        k, v       : ``(B, T, H, W_full, NH, HD)`` bfloat16.
        kernel_size: ``(KT, KH, KW)`` neighborhood window.
        w_offset   : Global W start of q; 0 for a full-tensor call.

    Returns:
        ``(B, T, H, W_sub, NH, HD)`` bfloat16.
    """
    B, T, H, W_sub, NH, HD = q.shape
    W_full = k.shape[3]
    KT, KH, KW = kernel_size
    device = q.device

    bt = _window_bounds(T, min(KT, T))
    bh = _window_bounds(H, min(KH, H))
    bw = _window_bounds(W_full, min(KW, W_full))

    eff_kt, eff_kh, eff_kw = min(KT, T), min(KH, H), min(KW, W_full)

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
        mask = _na3d_sdpa_mask(rel, q.dtype, device)
        nq, nk = mask.shape[2], mask.shape[3]
        g_max = max(1, kv_budget // max(1, B * NH * nk * HD * 2))
        qs0, _ = tiles_list[0]
        tq = qs0[0].stop - qs0[0].start
        th_sz = qs0[1].stop - qs0[1].start
        tw_sz = qs0[2].stop - qs0[2].start
        for c0 in range(0, len(tiles_list), g_max):
            chunk = tiles_list[c0 : c0 + g_max]
            g = len(chunk)
            q_s = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk])
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
# Default LTX-2.5 and edge-case shapes
# ---------------------------------------------------------------------------
# Each tuple: (B, T, H, W, NH, HD, KT, KH, KW)
_DEFAULT_SHAPES = [
    # --- LTX-2.5 default shapes (B=1) ---
    (1, 18, 32, 48, 32, 64, 3, 7, 7),  # det-0 full vol
    (1, 36, 64, 96, 8, 64, 3, 5, 5),  # det-2 full vol
    (1, 40, 96, 96, 8, 64, 3, 5, 5),  # det-3 tile (T=40, H=W=96)
    (1, 16, 80, 32, 4, 64, 11, 11, 11),  # diff-5 small tile
    (1, 79, 192, 192, 4, 64, 11, 11, 11),  # diff-5 dominant tile
    # --- Edge cases for correctness ---
    # SEQ % 16 != 0: exercises q_mask for the partial last program block.
    # SEQ = 3*5*17 = 255, 255 % 16 = 15.
    (1, 3, 5, 17, 4, 64, 3, 5, 5),
    # T == KT: every temporal token is at a border, so the T-clamping
    # (tl.minimum(..., T - KT)) is the active constraint for all queries.
    (1, 11, 16, 16, 4, 64, 11, 11, 11),
    # W == 16 == BLOCK_Q minimum: upper half of every BLOCK_KV=32 chunk is
    # masked by kv_ok, exercising W-boundary masking on every inner iteration.
    (1, 16, 16, 16, 4, 64, 11, 11, 11),
    # B > 1: tests pid_bnh addressing across batch elements.
    (2, 16, 16, 32, 4, 64, 3, 5, 5),
    # Cross-(t,h)-row: W=33 is not a multiple of BLOCK_Q=16.
    (1, 4, 8, 33, 4, 64, 3, 5, 5),
]


# ---------------------------------------------------------------------------
# Fast subset for cross-validation against na3d_sdpa_ref (FP32 per-query loop).
# Only shapes with SEQ <= 50K to keep na3d_sdpa_ref within a few seconds each.
# Covers all 4 edge-case categories plus 2 LTX-2.5 default shapes.
# Large shapes (36,64,96), (40,96,96), (79,192,192) are excluded here; they
# are already exercised in test_na3d_flash via the faster _na3d_sdpa_exact ref.
# ---------------------------------------------------------------------------
_SDPA_FAST_SHAPES = [
    (1, 18, 32, 48, 32, 64, 3, 7, 7),  # det-0 (SEQ = 27 648)
    (1, 16, 80, 32, 4, 64, 11, 11, 11),  # diff-5 small tile (SEQ = 40 960)
    (1, 3, 5, 17, 4, 64, 3, 5, 5),  # edge: SEQ % 16 != 0  (SEQ = 255)
    (1, 11, 16, 16, 4, 64, 11, 11, 11),  # edge: T == KT         (SEQ = 2 816)
    (1, 16, 16, 16, 4, 64, 11, 11, 11),  # edge: W == 16         (SEQ = 4 096)
    (2, 16, 16, 32, 4, 64, 3, 5, 5),  # edge: B > 1           (SEQ = 8 192)
]


# ---------------------------------------------------------------------------
# Pytest correctness tests
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,T,H,W,NH,HD,KT,KH,KW", _DEFAULT_SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_na3d_flash(B, T, H, W, NH, HD, KT, KH, KW, dtype):
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"na3d_flash unsupported on {get_gfx()}")

    scale = HD**-0.5
    q = torch.randn(B, T, H, W, NH, HD, dtype=dtype) * scale
    k = torch.randn(B, T, H, W, NH, HD, dtype=dtype)
    v = torch.randn(B, T, H, W, NH, HD, dtype=dtype)

    ref = _na3d_sdpa_exact(q, k, v, kernel_size=(KT, KH, KW))
    out = na3d_flash_attn(q, k, v, kernel_size=(KT, KH, KW))

    checkAllclose(
        ref.float(),
        out.float(),
        rtol=1e-2,
        atol=5e-2,
        msg=f"na3d_flash (T={T},H={H},W={W},k=({KT},{KH},{KW}))",
    )


# ---------------------------------------------------------------------------
# Benchmark function (one row per shape)
# ---------------------------------------------------------------------------
@benchmark()
def bench_na3d_flash(B, T, H, W, NH, HD, KT, KH, KW, dtype):
    SEQ = T * H * W
    C = NH * HD
    K = KT * KH * KW  # neighborhood size

    # Build inputs matching the real decoder call: pre-scaled Q, BF16, channels-last.
    scale = HD**-0.5
    q = torch.randn(B, T, H, W, NH, HD, dtype=dtype) * scale
    k = torch.randn(B, T, H, W, NH, HD, dtype=dtype)
    v = torch.randn(B, T, H, W, NH, HD, dtype=dtype)

    # Reference (not timed): GPU-grouped exact SDPA, faster than na3d_sdpa_ref loop.
    ref = _na3d_sdpa_exact(q, k, v, kernel_size=(KT, KH, KW))

    # FLOPs: QK and AV dot products over K neighbors per query.
    #   QK: 2 * B * SEQ * K * C
    #   AV: 2 * B * SEQ * K * C
    flops = 4 * B * SEQ * K * C
    # Bytes: Q loaded once, K and V reloaded per (t_kv, h_kv) row.
    elem = q.element_size()
    nbytes = (B * SEQ * C + 2 * B * SEQ * K * C) * elem

    candidates = {
        "triton": lambda: na3d_flash_attn(q, k, v, kernel_size=(KT, KH, KW)),
    }

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.float(),
            out.float(),
            rtol=1e-2,
            atol=5e-2,
            msg=f"{name}: na3d_flash (T={T},H={H},W={W},k=({KT},{KH},{KW}))",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# ---------------------------------------------------------------------------
# _na3d_sdpa_exact tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("B,T,H,W,NH,HD,KT,KH,KW", _SDPA_FAST_SHAPES)
def test_na3d_sdpa_exact_vs_ref(B, T, H, W, NH, HD, KT, KH, KW):
    """``_na3d_sdpa_exact`` agrees with ``na3d_sdpa_ref`` (FP32 per-query loop).

    Both compute the same inward-shifted neighborhood attention; the difference is
    numerical precision (BF16 vs FP32) and implementation (grouped tiled SDPA vs
    per-query loop).  The tolerance is the same as ``test_na3d_flash``.
    """
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"na3d_flash unsupported on {get_gfx()}")

    scale = HD**-0.5
    q = torch.randn(B, T, H, W, NH, HD, dtype=torch.bfloat16) * scale
    k = torch.randn(B, T, H, W, NH, HD, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, W, NH, HD, dtype=torch.bfloat16)

    ref = na3d_sdpa_ref(q, k, v, kernel_size=(KT, KH, KW))
    exact = _na3d_sdpa_exact(q, k, v, kernel_size=(KT, KH, KW))  # w_offset=0 default

    checkAllclose(
        ref.float(),
        exact.float(),
        rtol=1e-2,
        atol=5e-2,
        msg=f"_na3d_sdpa_exact vs na3d_sdpa_ref (T={T},H={H},W={W},k=({KT},{KH},{KW}))",
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("na3d_flash unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark 3D neighborhood flash attention",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["bf16"]],
        help="Input dtype (default: bf16)",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=_DEFAULT_SHAPES,
        help=(
            "Shapes as (B,T,H,W,NH,HD,KT,KH,KW) tuples.\n"
            "e.g.: -s '(1,79,192,192,4,64,11,11,11)'"
        ),
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        rows = []
        for shape in args.shapes:
            B, T, H, W, NH, HD, KT, KH, KW = shape
            rows.append(bench_na3d_flash(B, T, H, W, NH, HD, KT, KH, KW, dtype))
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "na3d_flash summary (%s):\n%s",
            str(dtype).replace("torch.", ""),
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()

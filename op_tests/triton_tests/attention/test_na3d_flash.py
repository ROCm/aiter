# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Test and benchmark for 3D neighborhood flash attention (na3d_flash).

Covers representative LTX-2.5 VAE decode shapes plus edge cases that exercise
boundary masking and non-aligned sequence lengths.

Each shape is compared against a pure-PyTorch tiled SDPA reference that implements
the same inward-shifted centered window.  TFLOPS and TB/s are reported per shape.
"""

import argparse

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.triton.attention.na3d_flash import _na3d_sdpa_exact, na3d_flash_attn
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
# Default LTX-2.5 and edge-case shapes
# ---------------------------------------------------------------------------
# Each tuple: (B, T, H, W, NH, HD, KT, KH, KW)
_DEFAULT_SHAPES = [
    # --- LTX-2.5 default shapes (B=1) ---
    (1, 18, 32, 48, 32, 64, 3, 7, 7),      # det-0 full vol
    (1, 36, 64, 96, 8, 64, 3, 5, 5),       # det-2 full vol
    (1, 40, 96, 96, 8, 64, 3, 5, 5),       # det-3 tile (T=40, H=W=96)
    (1, 16, 80, 32, 4, 64, 11, 11, 11),    # diff-5 small tile
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
]


# ---------------------------------------------------------------------------
# Fast subset for cross-validation against na3d_sdpa_ref (FP32 per-query loop).
# Only shapes with SEQ <= 50K to keep na3d_sdpa_ref within a few seconds each.
# Covers all 4 edge-case categories plus 2 LTX-2.5 default shapes.
# Large shapes (36,64,96), (40,96,96), (79,192,192) are excluded here; they
# are already exercised in test_na3d_flash via the faster _na3d_sdpa_exact ref.
# ---------------------------------------------------------------------------
_SDPA_FAST_SHAPES = [
    (1, 18, 32, 48, 32, 64, 3, 7, 7),      # det-0 (SEQ = 27 648)
    (1, 16, 80, 32, 4, 64, 11, 11, 11),    # diff-5 small tile (SEQ = 40 960)
    (1, 3, 5, 17, 4, 64, 3, 5, 5),         # edge: SEQ % 16 != 0  (SEQ = 255)
    (1, 11, 16, 16, 4, 64, 11, 11, 11),    # edge: T == KT         (SEQ = 2 816)
    (1, 16, 16, 16, 4, 64, 11, 11, 11),    # edge: W == 16         (SEQ = 4 096)
    (2, 16, 16, 32, 4, 64, 3, 5, 5),       # edge: B > 1           (SEQ = 8 192)
]


# ---------------------------------------------------------------------------
# Shapes to test correct_rightmost_block
# ---------------------------------------------------------------------------
_CORRECTION_SHAPES = [
    (1, 16, 80,  32, 4, 64, 11, 11, 11),   # diff-5 small tile
    (1, 16, 80, 192, 4, 64, 11, 11, 11),   # diff-5 wide tile
    (1, 18, 32,  48, 32, 64, 3,  7,  7),   # det-0
    (1, 36, 64,  96,  8, 64, 3,  5,  5),   # det-2
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


@pytest.mark.parametrize("B,T,H,W,NH,HD,KT,KH,KW", _CORRECTION_SHAPES)
def test_correct_rightmost_block(B, T, H, W, NH, HD, KT, KH, KW):
    """Verify the ``correct_rightmost_block`` docstring claim.

    Asserts:
    1. Interior W positions are bit-identical before and after correction.
    2. Rightmost W positions differ (correction fires and changes output).
    3. Corrected rightmost positions match ``_na3d_sdpa_exact`` applied to
       just those queries, confirming the kv_ok=False BF16(0) MFMA divergence
       is eliminated.
    """
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"na3d_flash unsupported on {get_gfx()}")

    scale = HD**-0.5
    q = torch.randn(B, T, H, W, NH, HD, dtype=torch.bfloat16) * scale
    k = torch.randn(B, T, H, W, NH, HD, dtype=torch.bfloat16)
    v = torch.randn(B, T, H, W, NH, HD, dtype=torch.bfloat16)

    out_base = na3d_flash_attn(q, k, v, (KT, KH, KW), correct_rightmost_block=False)
    out_corr = na3d_flash_attn(q, k, v, (KT, KH, KW), correct_rightmost_block=True)

    from aiter.ops.triton._triton_kernels.attention.na3d_flash import _na3d_flash_fwd
    actual_bq = _na3d_flash_fwd.best_config.kwargs["BLOCK_Q"]
    W_last = ((W - 1) // actual_bq) * actual_bq

    diff = (out_corr.float() - out_base.float()).abs()

    if W_last > 0:
        # Claim 1: interior positions unchanged.
        interior_max = diff[:, :, :, :W_last, :, :].max().item()
        assert interior_max == 0.0, (
            f"Interior W positions changed (max_diff={interior_max:.6f}) - "
            f"correction must not touch positions 0..{W_last - 1}"
        )

        # Claim 2: rightmost positions differ (correction fired).
        rightmost_max = diff[:, :, :, W_last:, :, :].max().item()
        assert rightmost_max > 0.0, (
            f"Rightmost W positions unchanged - "
            f"T*H*(W-W_last)={T*H*(W-W_last)} should be <= 100_000 so correction fires"
        )

        # Claim 3: corrected positions match _na3d_sdpa_exact.
        q_right = q[:, :, :, W_last:, :, :]
        exact_right = _na3d_sdpa_exact(q_right, k, v, (KT, KH, KW), w_offset=W_last)
        checkAllclose(
            exact_right.float(),
            out_corr[:, :, :, W_last:, :, :].float(),
            rtol=1e-2,
            atol=5e-2,
            msg=(
                f"corrected rightmost vs _na3d_sdpa_exact "
                f"(T={T},H={H},W={W},W_last={W_last},k=({KT},{KH},{KW}))"
            ),
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

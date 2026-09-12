# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the gfx950 MXFP A x MXFP4 B preshuffle GEMM via
``flydsl_batched_gemm_mxfp4``.

Covers ``a_dtype="fp4"`` (the pre-existing path) and ``a_dtype="fp6"``, which
the vendored ``kernels/mxfp4_preshuffle.py`` enables.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("flydsl")
from aiter.utility.fp4_utils import e8m0_to_f32, mxfp4_to_f32


def _is_gfx950() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        arch = torch.cuda.get_device_properties(0).gcnArchName
    except Exception:  # noqa: BLE001
        return False
    return arch.lower().split(":")[0] == "gfx950"


pytestmark = pytest.mark.skipif(
    not _is_gfx950(),
    reason="the MXFP preshuffle GEMM path is gfx950/CDNA4 only",
)


# B > 1 is deliberately not covered. The batched A-scale layout is caller-supplied
# via sca_row_stride / sca_batch_stride, and aiter has no helper that produces it --
# flydsl_batched_gemm_mxfp4's docstring points at a `preshuffle_operands` that is not
# defined anywhere in the tree. Asserting a guessed layout would be worse than not
# testing it; single-batch is enough to cover the vendored kernel and the call site.
@pytest.mark.parametrize("a_dtype", ["fp4", "fp6"])
@pytest.mark.parametrize("B", [1])
def test_flydsl_batched_gemm_mxfp4_a_dtypes(a_dtype, B):
    from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_batched_gemm_mxfp4
    from aiter.ops.flydsl.mxfp6_utils import (
        fp6_e2m3_to_f32,
        per_1x32_f6_quant,
        shuffle_scale_w4,
    )
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.ops.shuffle import shuffle_weight_NK

    device = torch.device("cuda")
    torch.manual_seed(0)
    M, N, K = 128, 1024, 8192

    a_bf = torch.randn(B, M, K, device=device, dtype=torch.bfloat16)
    w_bf = torch.randn(N, K, device=device, dtype=torch.bfloat16)

    # A operand.
    if a_dtype == "fp6":
        a_codes, a_scale, a_unpacked = per_1x32_f6_quant(a_bf.view(-1, K))
        a_deq = fp6_e2m3_to_f32(a_unpacked)
    else:
        a_codes, a_scale = per_1x32_f4_quant(a_bf.view(-1, K).float())[:2]
        a_deq = mxfp4_to_f32(a_codes)
    a_deq = a_deq * e8m0_to_f32(a_scale.repeat_interleave(32, dim=1))

    # B operand: MXFP4, CK-preshuffled. shuffle_weight_NK(w, 16, 64) is the same
    # layout the kernel expects; only the scale needs the CDNA4-specific shuffle.
    w_q, w_scale = per_1x32_f4_quant(w_bf.float())[:2]
    w_deq = mxfp4_to_f32(w_q) * e8m0_to_f32(w_scale.repeat_interleave(32, dim=1))

    out = flydsl_batched_gemm_mxfp4(
        a_codes.view(B, M, -1),
        shuffle_weight_NK(w_q, 16, 64),
        shuffle_scale_w4(a_scale, 1, False),
        shuffle_scale_w4(w_scale, 1, False),
        N,
        torch.bfloat16,
        a_dtype=a_dtype,
        tile_m=128,
        tile_n=128,
        tile_k=256,
    )
    assert tuple(out.shape) == (B, M, N)

    # Reference is the dequantised operands, so this isolates GEMM accumulation
    # error from the (much larger) quantization error.
    ref = a_deq @ w_deq.T
    rel = torch.linalg.vector_norm(
        out.reshape(-1, N).float() - ref
    ) / torch.linalg.vector_norm(ref)
    assert rel < 1e-2, f"relative error {rel:.4e} too high (a_dtype={a_dtype}, B={B})"


@pytest.mark.parametrize("a_dtype", ["fp4", "fp6"])
def test_pick_mx_tiles_only_returns_valid_combos(a_dtype):
    """pick_mx_tiles must never hand the kernel a combination it asserts on.

    The kernel's constraints are not all obvious -- notably the A tile must be a
    whole number of ``num_threads*16``-byte cooperative DMA rounds, which couples
    tile_m/tile_k/tile_n *and* a_dtype -- so this sweeps the shape space rather
    than spot-checking.
    """
    from aiter.ops.flydsl.batched_gemm_mxfp4 import pick_mx_tiles, tiles_are_valid

    for N in (512, 1024, 4096, 8192, 5120):
        for K in (256, 1024, 4096, 8192):
            for M in (1, 7, 32, 33, 128, 1024, 4096, 32768):
                tm, tn, tk = pick_mx_tiles(M, N, K, a_dtype)
                assert tiles_are_valid(tm, tn, tk, N, K, a_dtype), (
                    f"pick_mx_tiles({M},{N},{K},{a_dtype}) -> {(tm, tn, tk)} "
                    "violates the kernel constraints"
                )


@pytest.mark.parametrize("a_dtype", ["fp4", "fp6"])
@pytest.mark.parametrize("M", [1, 33, 128])
def test_flydsl_gemm_mxfp4_2d(a_dtype, M):
    """The un-batched 2D entry point, driven through the public prep helpers."""
    from aiter.ops.flydsl.batched_gemm_mxfp4 import (
        flydsl_gemm_mxfp4,
        preshuffle_mx_weight,
        quant_mx_act,
    )
    from aiter.ops.quant import per_1x32_f4_quant

    device = torch.device("cuda")
    torch.manual_seed(0)
    N, K = 1024, 4096

    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.5
    w_codes, w_scales = preshuffle_mx_weight(w)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
    a_codes, a_scales = quant_mx_act(x, a_dtype)

    out = flydsl_gemm_mxfp4(
        a_codes, w_codes, a_scales, w_scales, N, torch.bfloat16, a_dtype=a_dtype
    )
    assert tuple(out.shape) == (M, N)

    w_q, w_scale = per_1x32_f4_quant(w.float())[:2]
    w_deq = mxfp4_to_f32(w_q) * e8m0_to_f32(w_scale.repeat_interleave(32, dim=1))
    ref = x.float() @ w_deq.T
    rel = torch.linalg.vector_norm(out.float() - ref) / torch.linalg.vector_norm(ref)
    # Tolerances are the activation quantization error of each format: MXFP6-E2M3
    # lands ~3e-2 (about 31 dB SQNR), MXFP4-E2M1 ~1.2e-1 (about 19 dB).
    assert rel < (0.06 if a_dtype == "fp6" else 0.25), f"rel={rel:.4e}"


@pytest.mark.parametrize("M", [1, 8, 32])
def test_split_k_matches_single_k(M):
    """Split-K must be numerically identical, not merely close.

    launch_gemm accumulates each split in fp32 and launch_splitk_reduce sums
    them, so the only difference from k_batch=1 is summation order over fp32
    partials -- which for these shapes reproduces exactly.
    """
    from aiter.ops.flydsl.batched_gemm_mxfp4 import (
        flydsl_gemm_mxfp4,
        pick_mx_split_k,
        pick_mx_tiles,
        preshuffle_mx_weight,
        quant_mx_act,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    N, K = 4096, 7168

    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.5
    w_codes, w_scales = preshuffle_mx_weight(w)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
    a_codes, a_scales = quant_mx_act(x, "fp6")

    tiles = pick_mx_tiles(M, N, K, "fp6")
    k_batch = pick_mx_split_k(M, N, K, *tiles)
    assert k_batch > 1, (
        f"expected split-K to engage at M={M} (grid would be "
        f"{-(-M // tiles[0]) * (N // tiles[1])} workgroups)"
    )

    def run(kb):
        return flydsl_gemm_mxfp4(a_codes, w_codes, a_scales, w_scales, N,
                                 torch.bfloat16, a_dtype="fp6",
                                 tile_m=tiles[0], tile_n=tiles[1],
                                 tile_k=tiles[2], k_batch=kb)

    torch.testing.assert_close(run(k_batch), run(1), rtol=0, atol=0)

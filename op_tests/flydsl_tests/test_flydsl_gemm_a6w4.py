# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for a6w4 (MXFP6-E2M3 A x MXFP4 B) on the gfx950 preshuffle GEMM.

Also covers a4w4 through the same kernel, since the call-site fix this lands
with affects both.
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


@pytest.mark.parametrize("a_dtype", ["fp4", "fp6"])
@pytest.mark.parametrize("M", [1, 33, 32, 128])
def test_flydsl_gemm_mxfp_a_dtypes(a_dtype, M):
    from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_batched_gemm_mxfp4
    from aiter.ops.flydsl.mxfp6_utils import (
        fp6_e2m3_to_f32,
        per_1x32_f6_quant,
        shuffle_scale_w4,
        shuffle_weight_w4,
    )
    from aiter.ops.quant import per_1x32_f4_quant

    device = torch.device("cuda")
    torch.manual_seed(0)
    N, K = 1024, 4096

    x = torch.randn(M, K, device=device, dtype=torch.bfloat16) * 0.5
    w = torch.randn(N, K, device=device, dtype=torch.bfloat16) * 0.5

    # A operand. The kernel reads the A scale in whole 32-row supers, so the
    # scale must be 32-row aligned even when M is not. per_1x32_f6_quant does
    # that internally; aiter's per_1x32_f4_quant returns an M-row scale, so the
    # fp4 arm pads the activation first, exactly as a real caller must.
    m_pad = max(32, -(-M // 32) * 32)
    if a_dtype == "fp6":
        a_codes, a_scale, a_unpacked = per_1x32_f6_quant(x)
        a_deq = fp6_e2m3_to_f32(a_unpacked)
    else:
        x_pad = torch.nn.functional.pad(x, (0, 0, 0, m_pad - M))
        a_codes, a_scale = per_1x32_f4_quant(x_pad.float())[:2]
        a_codes = a_codes[:M].contiguous()
        a_deq = mxfp4_to_f32(a_codes)
    assert a_scale.shape[0] == m_pad, a_scale.shape
    a_deq = a_deq * e8m0_to_f32(a_scale[:M].repeat_interleave(32, dim=1))

    # B operand: MXFP4, CK-preshuffled.
    w_q, w_scale = per_1x32_f4_quant(w.float())[:2]
    w_deq = mxfp4_to_f32(w_q) * e8m0_to_f32(w_scale.repeat_interleave(32, dim=1))

    out = flydsl_batched_gemm_mxfp4(
        a_codes.view(1, M, -1),
        shuffle_weight_w4(w_q, 16),
        shuffle_scale_w4(a_scale),
        shuffle_scale_w4(w_scale),
        N,
        torch.bfloat16,
        a_dtype=a_dtype,
        tile_m=32,
        tile_n=128,
        tile_k=256,
    )
    assert tuple(out.shape) == (1, M, N)

    # Reference is the dequantised operands, so this isolates GEMM accumulation
    # error from the (much larger) quantization error.
    ref = a_deq @ w_deq.T
    rel = torch.linalg.vector_norm(
        out.reshape(M, N).float() - ref
    ) / torch.linalg.vector_norm(ref)
    assert rel < 1e-2, f"relative error {rel:.4e} (a_dtype={a_dtype}, M={M})"


def test_tile_m_guard_matches_kernel():
    """tile_m=16 must raise here, not inside the kernel."""
    from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_batched_gemm_mxfp4

    a = torch.zeros(1, 32, 2048, dtype=torch.uint8, device="cuda")
    z = torch.zeros(1, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="multiple of 32"):
        flydsl_batched_gemm_mxfp4(a, z, z, z, 1024, torch.bfloat16,
                                  a_dtype="fp6", tile_m=16, tile_n=128,
                                  tile_k=256)


def test_operand_prep_handles_ragged_m():
    """The documented quant -> shuffle sequence must survive a non-multiple-of-32 M.

    The kernel indexes the A scale in whole 32-row supers, so per_1x32_f6_quant
    returns a 32-row-aligned scale; an M-row scale would make shuffle_scale_w4
    throw for exactly the decode-shaped calls that matter.
    """
    from aiter.ops.flydsl.mxfp6_utils import per_1x32_f6_quant, shuffle_scale_w4

    K = 4096
    for M in (1, 7, 33, 64):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        codes, scale, unpacked = per_1x32_f6_quant(x)
        assert codes.shape == (M, K), codes.shape
        assert unpacked.shape == (M, K), unpacked.shape
        assert scale.shape == (max(32, -(-M // 32) * 32), K // 32), scale.shape
        shuffle_scale_w4(scale)  # must not raise


@pytest.mark.parametrize(
    "a_dtype, tiles",
    [
        ("fp4", (32, 128, 128)),  # A tile 2048B vs a 4096B copy round
        ("fp4", (32, 256, 128)),
    ],
)
def test_rejects_tiles_the_kernel_cannot_stage(a_dtype, tiles):
    """Tile combos the kernel asserts on must be rejected by the wrapper.

    The A tile is staged in whole num_threads*16-byte rounds, which couples
    tile_m/tile_n/tile_k *and* a_dtype -- the per-dimension checks cannot see it.
    """
    from aiter.ops.flydsl.batched_gemm_mxfp4 import flydsl_batched_gemm_mxfp4

    a = torch.zeros(1, 32, 2048, dtype=torch.uint8, device="cuda")
    z = torch.zeros(1024 * 2048, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError, match="num_threads"):
        flydsl_batched_gemm_mxfp4(a, z, z, z, 1024, torch.bfloat16,
                                  a_dtype=a_dtype, tile_m=tiles[0],
                                  tile_n=tiles[1], tile_k=tiles[2])


def test_shuffle_weight_w4_rejects_unsupported_variants():
    """Silently returning the dense permutation for a MoE/gate-up caller would
    produce wrong numbers, so those must raise."""
    from aiter.ops.flydsl.mxfp6_utils import shuffle_weight_w4

    w = torch.zeros(1024, 2048, dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="16-lane"):
        shuffle_weight_w4(w, 32)
    with pytest.raises(NotImplementedError):
        shuffle_weight_w4(w, 16, gate_up=True)
    with pytest.raises(NotImplementedError):
        shuffle_weight_w4(w, 16, moe_gemm=True)

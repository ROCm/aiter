# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MX scale preshuffle layout contract for the gfx1250 MXFP4 preshuffle GEMM.

``gemm_afp4wfp4_preshuffle`` dispatches gfx1250 to the gluon kernel
``gemm_mxfp4_preshuffle_gfx1250``. That kernel's TDM descriptors index the scale
buffers in the N16K4 tiling -- ``ceil(rows/16)`` rows of ``K/2`` bytes -- but
take the row stride straight from the tensor it is handed.

Every pre-gfx1250 producer emits the CDNA N32K8 tiling instead, viewed as
``(rows/32, K)``. That view spans the *same buffer*, so nothing about its size
gives the mistake away -- but it has half the rows at twice the pitch, so the
descriptors stride 2x and run ~2x the buffer length off the end. That was a
``Memory access fault`` in the DeepSeek-R1 MXFP4 dense layers, and silently
wrong results (~97% relative error) wherever the overrun landed on mapped
memory.

These tests pin down both halves of the contract:
  * the device swizzle behind ``per_1x32_mx_quant_hip(scale_shuffle_layout=...)``
    matches its host mirror ``shuffle_scale_gemm`` for both tilings;
  * feeding the GEMM the wrong tiling raises instead of reading out of bounds.
"""

from __future__ import annotations

import pytest
import torch

from aiter import QuantType, dtypes
from aiter.ops.quant import (
    MX_SCALE_SHUFFLE_N16K4,
    MX_SCALE_SHUFFLE_N32K8,
    get_hip_quant,
    mx_scale_shuffle_layout_for_gfx,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4_preshuffle
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm
from aiter.utility import fp4_utils

SCALE_GROUP = 32

requires_mxfp4 = pytest.mark.skipif(
    not torch.cuda.is_available() or not arch_info.is_fp4_avail(),
    reason="MXFP4 GPU required",
)
requires_gfx1250 = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_info.get_arch() != "gfx1250",
    reason="gluon MXFP4 preshuffle GEMM is gfx1250-only",
)


def _e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x == 255, float("nan"), 2.0 ** (x.to(torch.float32) - 127))


def _mxfp4_to_f32(x: torch.Tensor) -> torch.Tensor:
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    lut = torch.tensor(
        # fmt: off
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        # fmt: on
        dtype=torch.float32,
        device=x.device,
    )
    return lut[x.long()]


@requires_mxfp4
@pytest.mark.parametrize(
    "layout, arch, pf, kw",
    [
        (MX_SCALE_SHUFFLE_N32K8, "gfx950", 32, 8),
        (MX_SCALE_SHUFFLE_N16K4, "gfx1250", 16, 4),
    ],
)
@pytest.mark.parametrize("M, K", [(512, 7168), (64, 256), (37, 512)])
def test_device_scale_swizzle_matches_host(layout, arch, pf, kw, M, K):
    """The HIP quantizer's swizzle must equal the host shuffle it mirrors."""
    quant = get_hip_quant(QuantType.per_1x32)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)

    _, plain = quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    _, got = quant(
        x, quant_dtype=dtypes.fp4x2, shuffle=True, scale_shuffle_layout=layout
    )
    plain, got = plain.view(torch.uint8), got.view(torch.uint8)

    # The quantizer pads to (pad256(M), pad8(scaleN)); build the host reference
    # over that same padded extent, and only compare bytes the kernel writes
    # (x < M and y < scaleN) -- the padding is left untouched.
    rows, cols = got.shape
    padded = torch.zeros(rows, cols, dtype=torch.uint8, device="cuda")
    padded[:M, : plain.shape[1]] = plain
    written = torch.zeros(rows, cols, dtype=torch.uint8, device="cuda")
    written[:M, : plain.shape[1]] = 1

    ref = shuffle_scale_gemm(
        padded, arch=arch, preshuffle_factor=pf, scale_kwidth=kw
    ).reshape(-1)
    mask = (
        shuffle_scale_gemm(written, arch=arch, preshuffle_factor=pf, scale_kwidth=kw)
        .reshape(-1)
        .bool()
    )
    torch.testing.assert_close(got.reshape(-1)[mask], ref[mask], rtol=0, atol=0)


@requires_gfx1250
def test_n32k8_scale_view_is_rejected_not_read_out_of_bounds():
    """The CDNA view of a right-sized buffer must raise, not fault."""
    M, N, K = 512, 2048, 7168
    quant = get_hip_quant(QuantType.per_1x32)

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    xq, x_scale = quant(
        x,
        quant_dtype=dtypes.fp4x2,
        shuffle=True,
        scale_shuffle_layout=MX_SCALE_SHUFFLE_N32K8,
    )
    wq, w_scale = quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)
    w_scale = fp4_utils.e8m0_shuffle(w_scale, layout=MX_SCALE_SHUFFLE_N32K8)

    with pytest.raises(ValueError, match="N16K4"):
        gemm_afp4wfp4_preshuffle(
            xq.view(torch.uint8),
            shuffle_weight(wq, layout=(16, 16)).view(torch.uint8).view(N // 16, -1),
            x_scale.view(torch.uint8).view(x_scale.shape[0] // 32, -1),
            w_scale.view(torch.uint8).view(w_scale.shape[0] // 32, -1),
            y=torch.empty((M, N), dtype=torch.bfloat16, device="cuda"),
        )


@requires_gfx1250
@pytest.mark.parametrize("M", [16384, 512, 64, 33, 32])
@pytest.mark.parametrize(
    "N, K",
    [
        (36864, 7168),  # DeepSeek-R1 dense gate_up -- the shape that faulted
        (7168, 18432),  # dense down
        (2048, 512),
    ],
)
def test_n16k4_end_to_end_matches_reference(M, N, K):
    """quantize -> preshuffle -> GEMM in N16K4 must be numerically exact."""
    layout = mx_scale_shuffle_layout_for_gfx()
    assert layout == MX_SCALE_SHUFFLE_N16K4
    rows_per = 16
    quant = get_hip_quant(QuantType.per_1x32)

    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) / 4
    wq, w_scale = quant(w, quant_dtype=dtypes.fp4x2, shuffle=False)
    w_ref = _mxfp4_to_f32(wq.view(torch.uint8)) * _e8m0_to_f32(
        w_scale.view(torch.uint8)[:N, : K // SCALE_GROUP]
    ).repeat_interleave(SCALE_GROUP, dim=1)

    weight = shuffle_weight(wq, layout=(16, 16))
    weight_scale = fp4_utils.e8m0_shuffle(w_scale, layout=layout)

    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) / 4
    xq, x_scale = quant(
        x, quant_dtype=dtypes.fp4x2, shuffle=True, scale_shuffle_layout=layout
    )
    _, x_scale_plain = quant(x, quant_dtype=dtypes.fp4x2, shuffle=False)
    x_ref = _mxfp4_to_f32(xq.view(torch.uint8)) * _e8m0_to_f32(
        x_scale_plain.view(torch.uint8)[:M, : K // SCALE_GROUP]
    ).repeat_interleave(SCALE_GROUP, dim=1)

    out = gemm_afp4wfp4_preshuffle(
        xq.view(torch.uint8),
        weight.view(torch.uint8).view(N // 16, -1),
        x_scale.view(torch.uint8).view(x_scale.shape[0] // rows_per, -1),
        weight_scale.view(torch.uint8).view(weight_scale.shape[0] // rows_per, -1),
        y=torch.empty(((M + 31) // 32 * 32, N), dtype=torch.bfloat16, device="cuda"),
    )[:M]

    torch.testing.assert_close(
        out, (x_ref @ w_ref.T).to(torch.bfloat16), rtol=0, atol=0
    )

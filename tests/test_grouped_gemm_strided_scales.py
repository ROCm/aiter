# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The grouped MXFP4 GEMM takes a scale tensor whose rows sit inside wider rows.

MegaMoE's expert-grouped dispatch lands each row's e8m0 scales right after its
payload, inside one wire row, and hands the grouped GEMM that strided view; the
kernel reads it from its base address, by ``a_scale_row_stride_bytes``. No GPU
needed: the compiled launch is replaced by one that records its arguments.
"""

import pytest
import torch

pytest.importorskip("flydsl", reason="the grouped GEMM is built with FlyDSL")

from aiter.ops.flydsl import grouped_gemm_mxfp4 as GG
from aiter.ops.flydsl.kernels.tensor_shim import flat_arg

ROWS, PAYLOAD, SCALE_ROW, WIRE_ROW = 8, 256, 8, 384


def _wire_scales():
    """ROWS wire rows of distinct bytes, and the view of each row's scales."""
    wire = (torch.arange(ROWS * WIRE_ROW) % 251).to(torch.uint8)
    return torch.as_strided(wire[PAYLOAD:], (ROWS, SCALE_ROW), (WIRE_ROW, 1))


def test_flat_arg_contiguous_is_view_minus_one():
    t = torch.arange(24, dtype=torch.int32).view(4, 6)
    flat = flat_arg(t)
    assert flat.shape == (24,) and flat.data_ptr() == t.data_ptr()


def test_flat_arg_strided_keeps_the_base_and_every_row():
    scales = _wire_scales().view(torch.int32)
    flat = flat_arg(scales)
    assert flat.dim() == 1 and flat.data_ptr() == scales.data_ptr()
    row_stride, width = scales.stride(0), scales.shape[1]
    for r in range(ROWS):
        assert torch.equal(flat[r * row_stride : r * row_stride + width], scales[r])
    # Nothing past the last row's scales.
    assert flat.numel() == (ROWS - 1) * row_stride + width


def test_launcher_passes_strided_scales_at_their_base(monkeypatch):
    launched = []
    monkeypatch.setattr(
        GG, "_run_compiled", lambda fn, *args, **kw: launched.append(args)
    )
    a_scales = _wire_scales()
    w_scales = torch.zeros(4, 16, dtype=torch.uint8)
    GG.flydsl_grouped_gemm_a8w4_masked(
        torch.zeros(ROWS, 256, dtype=torch.bfloat16),
        torch.zeros(ROWS, PAYLOAD, dtype=torch.uint8),
        torch.zeros(1, 256, 128, dtype=torch.uint8),
        a_scales,
        w_scales,
        torch.zeros(4, dtype=torch.int32),
        n_experts=1,
        contiguous_m=ROWS,
        N=256,
        K=256,
        stream=object(),
        row_major_ascale=1,
        a_row_stride_bytes=WIRE_ROW,
        a_scale_row_stride_bytes=WIRE_ROW,
    )
    (args,) = launched
    sent_a, sent_w = args[3], args[4]
    assert sent_a.dim() == 1 and sent_a.data_ptr() == a_scales.data_ptr()
    assert sent_w.shape == (16,) and sent_w.data_ptr() == w_scales.data_ptr()

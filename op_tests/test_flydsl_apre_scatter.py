# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Bitwise payload/scale checks for all supported full-row A-preshuffle sizes.

Run on an idle gfx1250 GPU:
    python -m pytest op_tests/test_flydsl_apre_scatter.py -q
"""

import pytest
import torch

from aiter import get_gfx
from aiter.ops.flydsl.kernels.moe_apreshuffled_quant_scatter import (
    build_moe_apre_scatter_module,
)
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg


@pytest.mark.parametrize("k", range(1024, 16385, 1024))
@pytest.mark.parametrize("wmma_rep", [1, 8])
@pytest.mark.parametrize("scale_padding", [0, 32])
def test_apre_scatter(k, wmma_rep, scale_padding):
    if not torch.cuda.is_available() or get_gfx() != "gfx1250":
        pytest.skip("requires gfx1250")

    torch.manual_seed(42)
    tokens = 17
    rows = 3 * 16 * wmma_rep
    payload_bytes, scale_bytes = k // 2, k // 32
    payload = torch.randint(
        0, 256, (tokens, payload_bytes), dtype=torch.uint8, device="cuda"
    )
    scales = torch.randint(
        0, 256, (tokens, scale_bytes + scale_padding), dtype=torch.uint8, device="cuda"
    )
    row_map = torch.arange(rows, dtype=torch.int32, device="cuda") % tokens
    row_map[5::7] = -1
    row_map[16:32] = -1  # A whole empty tile must take the uniform early exit.

    # Guard both ends so any output overrun is visible, including at K=16384.
    guard, sentinel = 64, 0xA5
    payload_storage = torch.full(
        (rows * payload_bytes + 2 * guard,), sentinel, dtype=torch.uint8, device="cuda"
    )
    scale_storage = torch.full(
        (rows * scale_bytes + 2 * guard,), sentinel, dtype=torch.uint8, device="cuda"
    )
    out_payload = payload_storage[guard:-guard]
    out_scale = scale_storage[guard:-guard]
    launch = build_moe_apre_scatter_module(
        k, wmma_rep, src_scale_stride_bytes=scales.shape[-1]
    )
    launch(
        ptr_arg(payload),
        ptr_arg(scales),
        ptr_arg(out_payload),
        ptr_arg(out_scale),
        ptr_arg(row_map),
        tokens,
        rows,
        rows // 16,
        stream=torch.cuda.current_stream(),
    )

    # Partially padded tiles load token 0 for invalid rows; entirely empty
    # tiles leave their destinations untouched. Check both behaviors exactly.
    source_rows = row_map.clamp_min(0).long()
    empty_rows = (row_map.view(-1, 16) < 0).all(dim=1).repeat_interleave(16)
    gathered_payload = payload[source_rows]
    gathered_scale = scales[source_rows, :scale_bytes]
    gathered_payload[empty_rows] = sentinel
    gathered_scale[empty_rows] = sentinel
    expected_payload = (
        gathered_payload.view(rows // 16, 16, k // 32, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(-1)
    )
    expected_scale = (
        gathered_scale.reshape(rows // (16 * wmma_rep), wmma_rep, 16, k // 128, 4)
        .permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(-1)
    )
    torch.testing.assert_close(out_payload, expected_payload, rtol=0, atol=0)
    torch.testing.assert_close(out_scale, expected_scale, rtol=0, atol=0)
    for storage in (payload_storage, scale_storage):
        assert bool((storage[:guard] == sentinel).all())
        assert bool((storage[-guard:] == sentinel).all())

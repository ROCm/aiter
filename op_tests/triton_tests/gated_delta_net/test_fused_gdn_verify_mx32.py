# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fused_gdn_verify_qkvz(..., out_scale=...): the 1x32 MXFP8 output for the out projection
must be bit-identical -- fp8 codes and e8m0 scale bytes, in the GEMM's x-scale layout -- to the
bf16 output followed by the standalone 1x32 quant: per_group_quant_hip(group_size=32,
fp8_e8m0, scale_layout_m32k4=True) for >= 32 GEMM rows (n32k4, pad rows 0x7F), and
per_1x32_mx_quant_hip(shuffle=False) row-major below. The state updates must not change.
"""

import pytest
import torch

from aiter import dtypes
from aiter.ops.quant import per_1x32_mx_quant_hip, per_group_quant_hip
from aiter.ops.triton.gated_delta_net.fused_gdn_verify_qkvz import (
    fused_gdn_verify_mx32_supported,
    fused_gdn_verify_qkvz,
)
from op_tests.triton_tests.gated_delta_net.test_fused_gdn_verify_qkvz import (
    EPS,
    K,
    V,
    _requires_gfx1250,
    make_layer,
    make_pools,
    make_step,
    op_args,
)

MS = [
    1,
    2,
    3,
    4,
    8,
    16,
    31,
    32,
    33,
    61,
    64,
    100,
    128,
    192,
    244,
    256,
    300,
    512,
    1024,
    4096,
]


def _scale_bytes(rows, cols):
    return (rows + 31) // 32 * 32 * (cols // 32)


def _reference(y):
    """Standalone quant of bf16 [rows, cols] -> (fp8 bytes, the scale bytes the GEMM reads)."""
    rows, cols = y.shape
    if rows >= 32:
        q, s = per_group_quant_hip(
            y,
            quant_dtype=dtypes.fp8,
            group_size=32,
            scale_type=dtypes.fp8_e8m0,
            scale_layout_m32k4=True,
        )
        return q.view(torch.uint8), s.view(torch.uint8).reshape(-1)
    q, s = per_1x32_mx_quant_hip(
        y, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0, shuffle=False
    )
    return q.view(torch.uint8).view(rows, cols), (
        s.view(torch.uint8)[:rows, : cols // 32].reshape(-1)
    )


@pytest.mark.parametrize("extra_rows", [0, 5])
@pytest.mark.parametrize("m", MS)
def test_fused_gdn_verify_mx32_out(m, extra_rows):
    _requires_gfx1250()
    kh, vh, window, cap = 16, 32, 4, 16  # Qwen3.5-35B-A3B with DP attention, MTP-3
    layer = make_layer(kh, vh, 0)
    seqs = (m + window - 1) // window
    lengths = [window] * (seqs - 1) + [m - window * (seqs - 1)]
    slots = max(96, seqs)
    pools = make_pools(layer, window, cap, 1, num_slots=slots)
    step = make_step(layer, seqs, window, 2, num_slots=slots, lengths=lengths)
    assert fused_gdn_verify_mx32_supported(
        pools["ckpt"], pools["conv"].transpose(-1, -2), pools["rk"], window, seqs
    )
    rows, cols = m + extra_rows, vh * V

    def run(out, **kw):
        p = {k: v.clone() for k, v in pools.items()}
        fused_gdn_verify_qkvz(
            max_query_len=window,
            scale=K**-0.5,
            norm_eps=EPS,
            **op_args(layer, step, p, out),
            **kw,
        )
        return p

    out_bf16 = torch.zeros((m, vh, V), dtype=torch.bfloat16, device="cuda")
    p_bf16 = run(out_bf16)
    y = torch.zeros((rows, cols), dtype=torch.bfloat16, device="cuda")
    y[:m] = out_bf16.view(m, cols)
    q_ref, s_ref = _reference(y)

    q = torch.zeros((rows, cols), dtype=torch.float8_e4m3fn, device="cuda")
    s = torch.empty(_scale_bytes(rows, cols), dtype=torch.uint8, device="cuda")
    p_mx = run(q[:m].view(m, vh, V), out_scale=s, out_scale_rows=rows)
    # Rows past the written ones: what the standalone quant makes of zero rows (the caller's job)
    nb = s_ref.numel()
    s_view = s[:nb]
    if rows >= 32:
        # n32k4 bytes as [row block, 128-col group, row % 32, group % 4]
        tail = s.view((rows + 31) // 32, cols // 128, 32, 4)
        r_of = (
            torch.arange(tail.shape[0], device="cuda")[:, None] * 32
            + torch.arange(32, device="cuda")[None, :]
        )[:, None, :, None].expand_as(tail)
        tail[(r_of >= m) & (r_of < rows)] = 85
        tail[r_of >= rows] = 0x7F
    else:
        s_view.view(rows, cols // 32)[m:] = 85
    torch.cuda.synchronize()
    assert torch.equal(q.view(torch.uint8), q_ref), "fp8 codes differ"
    assert torch.equal(s_view, s_ref), "e8m0 scale bytes differ"
    for k in p_bf16:
        assert torch.equal(p_bf16[k], p_mx[k]), f"state {k} differs"

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-shape (1-128 row, E128 top-4, H6144) MoE routing front end:
``mega_moe_tp.decode_router_chain.route_quant_small`` against a torch reference (ids, weights, MXFP4 / MXFP8 input quant).
"""

import pytest
import torch

from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_chain import route_quant_small
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_gemm import router_line_layout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a gfx950 GPU"
)
H, E, TOPK, SCALE = 6144, 128, 4, 2.0
# e2m1 values by nibble
E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _weights(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(E, H, device="cuda", generator=g) * 0.05).float().contiguous()
    bias = (torch.randn(E, device="cuda", generator=g) * 0.1).float()
    return w, bias


def _dequant_fp4(q, s):
    lut = torch.tensor(E2M1, device=q.device, dtype=torch.float32)
    lo, hi = lut[(q & 15).long()], lut[(q >> 4).long()]
    v = torch.stack([lo, hi], -1).reshape(q.shape[0], -1)
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, 1)


def _dequant_fp8(q, s):
    v = q.view(torch.float8_e4m3fn).float()
    return v * torch.exp2(s.float() - 127).repeat_interleave(32, 1)


@pytest.mark.parametrize("n", [1, 8, 16, 17, 24, 32, 33, 57, 58, 64, 65, 96, 128])
def test_route_quant_small(n):
    w, bias = _weights()
    wl = router_line_layout(w)
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16)
    (q4, s4), (q8, s8), wt, ids = route_quant_small(x, w, wl, bias, SCALE)

    logits = x.double() @ w.double().T
    sig = torch.sigmoid(logits)
    ref_ids = torch.topk(sig + bias.double(), TOPK, dim=1).indices
    assert torch.equal(ids.long().sort(1).values, ref_ids.sort(1).values)
    sel = sig.gather(1, ids.long())
    ref_wt = sel / sel.sum(1, keepdim=True) * SCALE
    torch.testing.assert_close(wt.double(), ref_wt, rtol=1e-4, atol=1e-6)

    xf = x.float()
    rel8 = (_dequant_fp8(q8, s8) - xf).norm() / xf.norm()
    rel4 = (_dequant_fp4(q4, s4) - xf).norm() / xf.norm()
    assert rel8 < 0.05, rel8
    assert rel4 < 0.25, rel4

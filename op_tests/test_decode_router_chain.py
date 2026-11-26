# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-shape (1-128 row, E128 top-4, H6144) MoE routing front end:
``mega_moe_tp.decode_router_chain.route_quant_small`` against a torch reference (ids, weights, MXFP4 / MXFP8 input quant).
"""

import pytest
import torch

from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_chain import route_quant_small
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_gemm import router_line_layout


def _gfx950():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950"


pytestmark = pytest.mark.skipif(not _gfx950(), reason="needs a gfx950 GPU")
H, E, TOPK, SCALE, G = 6144, 128, 4, 2.0, 32
# e2m1 values by nibble
E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _weights(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(E, H, device="cuda", generator=g) * 0.05).float().contiguous()
    bias = (torch.randn(E, device="cuda", generator=g) * 0.1).float()
    return w, bias


def _e8m0_fp8_ref(amax):
    """exponent byte of ceil_pow2(amax / 448), at least 1."""
    u = (amax * (1.0 / 448.0)).view(torch.int32)
    e = (u >> 23) & 255
    return torch.where(((u & 0x7FFFFF) != 0) & (e < 255), e + 1, e).clamp(min=1)


def _e8m0_fp4_ref(amax):
    """Quark even scale: exponent of amax plus half an e2m1 mantissa ulp (0x200000), minus 2, at least 1."""
    return ((amax.view(torch.int32) - 0xE00000) >> 23).clamp(min=1)


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

    assert ids.dtype == torch.int32 and ((ids >= 0) & (ids < E)).all()

    xf = x.float()
    # every row's scales equal the torch rule exactly, and every row reconstructs (a lost row would score 1.0)
    amax = xf.view(n, H // G, G).abs().amax(-1)
    assert torch.equal(s8, _e8m0_fp8_ref(amax).to(torch.uint8))
    assert torch.equal(s4, _e8m0_fp4_ref(amax).to(torch.uint8))
    rel8 = (_dequant_fp8(q8, s8) - xf).norm(dim=1) / xf.norm(dim=1)
    rel4 = (_dequant_fp4(q4, s4) - xf).norm(dim=1) / xf.norm(dim=1)
    assert rel8.max() < 0.05, rel8.max()
    assert rel4.max() < 0.25, rel4.max()


def test_unsupported_inputs_return_none():
    w, bias = _weights()
    wl = router_line_layout(w)
    x = torch.randn(4, H, device="cuda", dtype=torch.bfloat16)
    assert route_quant_small(x[:0], w, wl, bias) is None  # no rows
    assert route_quant_small(x.float(), w, wl, bias) is None  # not bf16
    assert route_quant_small(x[:1].expand(58, H), w, wl, bias) is None  # stride 0
    padded = torch.randn(4, H + 32, device="cuda", dtype=torch.bfloat16)[:, :H]
    assert route_quant_small(padded, w, wl, bias) is None  # rows not contiguous
    assert route_quant_small(x, w, wl, bias[:1]) is None  # short bias
    assert route_quant_small(x, w, wl, bias.cpu()) is None  # bias on the host
    assert route_quant_small(x, w.cpu(), wl, bias) is None
    big = torch.randn(129, H, device="cuda", dtype=torch.bfloat16)
    assert route_quant_small(big, w, wl, bias) is None  # > 128 rows
    x0 = torch.randn((), device="cuda", dtype=torch.bfloat16)
    assert (
        route_quant_small(x0, w, wl, bias) is None
    )  # 0-d: rank checked before x.shape[0]


def test_topk_quant_rejects_bad_outputs():
    from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_topk_quant import (
        splitk_biased_topk_mxfp4_mxfp8_quant,
    )

    _w, bias = _weights()
    x = torch.randn(4, H, device="cuda", dtype=torch.bfloat16)
    parts = torch.zeros(4, 48, E, device="cuda")
    ids = torch.empty(4, TOPK, device="cuda", dtype=torch.int32)
    wt = torch.empty(4, TOPK, device="cuda")
    with pytest.raises(ValueError):  # fp16 weights are half the bytes the kernel writes
        splitk_biased_topk_mxfp4_mxfp8_quant(parts, bias, wt.half(), ids, SCALE, x)
    with pytest.raises(ValueError):  # a one-element bias, the kernel reads 128
        splitk_biased_topk_mxfp4_mxfp8_quant(parts, bias[:1], wt, ids, SCALE, x)
    with pytest.raises(ValueError):  # wrong top-k width
        splitk_biased_topk_mxfp4_mxfp8_quant(
            parts, bias, wt[:, :2], ids[:, :2], SCALE, x
        )
    with pytest.raises(ValueError):  # topk_ids aliases topk_weights' storage
        splitk_biased_topk_mxfp4_mxfp8_quant(
            parts, bias, wt, wt.view(torch.int32), SCALE, x
        )

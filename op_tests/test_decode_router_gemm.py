# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-shape router GEMM (E128, H6144): the split-K FP32 partials of ``decode_router_gemm.router_gemm_partials``
sum to the logits ``x @ W.T``."""

import pytest
import torch

from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_gemm import (
    router_gemm_partials,
    router_line_layout,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a gfx950 GPU"
)
H, E = 6144, 128


@pytest.mark.parametrize(
    "n", [1, 4, 8, 16, 17, 24, 32, 33, 48, 57, 58, 64, 65, 96, 128]
)
@pytest.mark.parametrize("lines", [True, False])
def test_partials_sum_is_the_logits(n, lines):
    g = torch.Generator(device="cuda").manual_seed(n)
    w = (torch.randn(E, H, device="cuda", generator=g) * 0.05).float().contiguous()
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=g)
    p = router_gemm_partials(x, w, router_line_layout(w) if lines else None)
    ref = x.double() @ w.double().T
    torch.testing.assert_close(p.double().sum(1), ref, rtol=1e-4, atol=1e-4)

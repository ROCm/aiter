# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-shape router GEMM (E128, H6144): the split-K FP32 partials of ``decode_router_gemm.router_gemm_partials``
sum to the logits ``x @ W.T``, each split's partial is its own K chunk's product, and unsupported inputs return None.

CI runs this file through pytest (``.github/scripts/aiter_test.sh``): it has no script entry point.
"""

import pytest
import torch

from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_gemm import (
    router_gemm_partials,
    router_line_layout,
    split_config,
)

H, E = 6144, 128


def _gfx950():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950"


pytestmark = pytest.mark.skipif(not _gfx950(), reason="needs a gfx950 GPU")


def _inputs(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(E, H, device="cuda", generator=g) * 0.05).float().contiguous()
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, w


@pytest.mark.parametrize("n", [1, 4, 8, 16, 17, 24, 32])
def test_partials_sum_is_the_logits(n):
    x, w = _inputs(n, n)
    p = router_gemm_partials(x, w, router_line_layout(w))
    ref = x.double() @ w.double().T
    torch.testing.assert_close(p.double().sum(1), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [1, 16, 17, 32])
def test_each_split_is_its_k_chunk(n):
    """The fixed split-K order is the contract (it fixes the downstream FP32 sum and so the top-k ids): slot ``s``
    must hold the product over K chunk ``s``, so permuting the slots fails here although their sum is unchanged.
    """
    x, w = _inputs(n, 100 + n)
    p = router_gemm_partials(x, w, router_line_layout(w))
    split_k, _ = split_config(n)
    kc = H // split_k
    xd, wd = x.double(), w.double()
    for s in range(split_k):
        ref = xd[:, s * kc : (s + 1) * kc] @ wd[:, s * kc : (s + 1) * kc].T
        torch.testing.assert_close(p[:, s].double(), ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("n", [3, 17])
def test_padded_row_stride(n):
    """Rows of a [n, H + 32] buffer's first H columns: the stride must reach the kernel."""
    x, w = _inputs(n, 7)
    xp = torch.randn(n, H + 32, device="cuda", dtype=torch.bfloat16)
    xp[:, :H] = x
    p = router_gemm_partials(xp[:, :H], w, router_line_layout(w))
    ref = x.double() @ w.double().T
    torch.testing.assert_close(p.double().sum(1), ref, rtol=1e-4, atol=1e-4)


def test_unsupported_inputs_return_none():
    x, w = _inputs(4, 1)
    wl = router_line_layout(w)
    assert router_gemm_partials(x[:0], w, wl) is None  # no rows
    assert (
        router_gemm_partials(torch.randn(33, H, device="cuda").bfloat16(), w, wl)
        is None
    )  # > 32 rows
    assert router_gemm_partials(x.float(), w, wl) is None  # not bf16
    assert (
        router_gemm_partials(x[:1].expand(4, H), w, wl) is None
    )  # overlapping rows (stride 0)
    assert router_gemm_partials(x, w, None) is None  # the line-layout gate is required
    assert router_gemm_partials(x, w, wl.cpu()) is None  # gate copy on the host
    assert router_gemm_partials(x.cpu(), w.cpu(), wl.cpu()) is None

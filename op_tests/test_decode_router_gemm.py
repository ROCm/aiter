# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode-shape router GEMM (E128, H6144): the split-K FP32 partials of ``decode_router_gemm.router_gemm_partials``
sum to the logits ``x @ W.T``, each split's partial is its own K chunk's product, the n-blocks kernel's optional
MXFP4 / MXFP8 input-quant workgroups match a torch reference, and unsupported inputs return None.

CI runs this file through pytest (``.github/scripts/aiter_test.sh``): it has no script entry point.
"""

import pytest
import torch

from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_common import run_on_device
from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_gemm import (
    nblocks_config,
    router_gemm_partials,
    router_line_layout,
    split_config,
)

H, E, G = 6144, 128, 32  # G: values per quant group


def _gfx950():
    if not torch.cuda.is_available():
        return False
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950"


pytestmark = pytest.mark.skipif(not _gfx950(), reason="needs a gfx950 GPU")

ROWS = [1, 4, 8, 16, 17, 24, 32, 33, 48, 57, 58, 64, 65, 96, 128]


def _inputs(n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(E, H, device="cuda", generator=g) * 0.05).float().contiguous()
    x = torch.randn(n, H, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, w


@pytest.mark.parametrize("n", ROWS)
@pytest.mark.parametrize("lines", [True, False])
def test_partials_sum_is_the_logits(n, lines):
    x, w = _inputs(n, n)
    p = router_gemm_partials(x, w, router_line_layout(w) if lines else None)
    ref = x.double() @ w.double().T
    torch.testing.assert_close(p.double().sum(1), ref, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("n", [1, 16, 17, 32, 33, 64, 65, 128])
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


@pytest.mark.parametrize("n", [3, 17, 40, 100])
def test_padded_row_stride(n):
    """The first H columns of a [n, H + 32] buffer: the row stride must reach every tier."""
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
    assert router_gemm_partials(_inputs(129, 1)[0], w, wl) is None  # > 128 rows
    assert router_gemm_partials(x.float(), w, wl) is None  # not bf16
    assert router_gemm_partials(x[:1].expand(4, H), w, wl) is None  # stride 0
    assert router_gemm_partials(x, w.cpu(), wl) is None  # gate on the host
    # a gate copy on the host is not used: the n-blocks tier reads the plain gate and the result is still right
    p = router_gemm_partials(x, w, wl.cpu())
    torch.testing.assert_close(
        p.double().sum(1), x.double() @ w.double().T, rtol=1e-4, atol=1e-4
    )
    assert router_gemm_partials(x.cpu(), w.cpu(), wl.cpu()) is None


def _e8m0_fp8_ref(amax):
    """exponent byte of ceil_pow2(amax / 448), at least 1."""
    u = (amax * (1.0 / 448.0)).view(torch.int32)
    e = (u >> 23) & 255
    return torch.where(((u & 0x7FFFFF) != 0) & (e < 255), e + 1, e).clamp(min=1)


def _e8m0_fp4_ref(amax):
    """Quark even scale: exponent of amax plus half an e2m1 mantissa ulp (0x200000), minus 2, at least 1."""
    return ((amax.view(torch.int32) - 0xE00000) >> 23).clamp(min=1)


_E2M1 = (0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6)


def _run_quant(x, w, quant):
    """The n-blocks launch with the quant workgroups (``quant`` 1 appended, 2 prepended), token-major partials."""
    from aiter.ops.flydsl.kernels.mega_moe_tp.decode_router_gemm_nblocks import (
        _get_kernel,
    )

    m, dev = x.shape[0], x.device
    split_k, block_k, waves, nbw = nblocks_config(m)
    p = torch.empty((m, split_k, E), dtype=torch.float32, device=dev)
    q = torch.empty((m, H // 2), dtype=torch.uint8, device=dev)
    s = torch.empty((m, H // G), dtype=torch.uint8, device=dev)
    q8 = torch.empty((m, H), dtype=torch.uint8, device=dev)
    s8 = torch.empty((m, H // G), dtype=torch.uint8, device=dev)
    run_on_device(
        _get_kernel(split_k, block_k, waves, nbw, quant, False, True),
        dev,
        x.data_ptr(),
        w.data_ptr(),
        p.data_ptr(),
        m,
        x.stride(0),
        q.data_ptr(),
        s.data_ptr(),
        q8.data_ptr(),
        s8.data_ptr(),
    )
    return p, (q, s), (q8, s8)


@pytest.mark.parametrize("pad", [0, 32])
@pytest.mark.parametrize("quant", [1, 2])
@pytest.mark.parametrize("n", [33, 58, 64, 65, 128])
def test_input_quant_workgroups(n, quant, pad):
    """Both quants against torch: scales exactly, codes by their dequantization error bound per element, and the
    router partials still sum to the logits (the extra workgroups share the launch)."""
    x, w = _inputs(n, 5)
    xp = torch.randn(n, H + pad, device="cuda", dtype=torch.bfloat16)
    xp[:, :H] = x
    p, (q4, s4), (q8, s8) = _run_quant(xp[:, :H], w, quant)
    torch.testing.assert_close(
        p.double().sum(1), x.double() @ w.double().T, rtol=1e-4, atol=1e-4
    )

    xf = x.float()
    amax = xf.view(n, H // G, G).abs().amax(-1)
    assert torch.equal(s8, _e8m0_fp8_ref(amax).to(torch.uint8))
    assert torch.equal(s4, _e8m0_fp4_ref(amax).to(torch.uint8))
    scale8 = torch.exp2(s8.float() - 127).repeat_interleave(G, 1)
    scale4 = torch.exp2(s4.float() - 127).repeat_interleave(G, 1)
    # e4m3: relative error <= 2^-4 (3 mantissa bits), plus the subnormal step 2^-9 * scale
    deq8 = q8.view(torch.float8_e4m3fn).float() * scale8
    assert ((deq8 - xf).abs() <= xf.abs() / 16 + scale8 * 2**-9).all()
    # e2m1: neighbouring codes are at most 2 apart (4 -> 6), and amax / scale < 8 saturates at 6
    lut = torch.tensor(_E2M1, device="cuda", dtype=torch.float32)
    deq4 = torch.stack([lut[(q4 & 15).long()], lut[(q4 >> 4).long()]], -1).reshape(n, H)
    assert ((deq4 * scale4 - xf).abs() <= 2 * scale4).all()
    # no row is lost: every row reconstructs
    assert ((deq8 - xf).norm(dim=1) / xf.norm(dim=1)).max() < 0.05
    assert ((deq4 * scale4 - xf).norm(dim=1) / xf.norm(dim=1)).max() < 0.25

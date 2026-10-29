# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16_streamk import (
    _SUPPORTED_ARCHS,
    gemm_a16w16_streamk,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch


def is_supported():
    return any(a in (get_arch() or "") for a in _SUPPORTED_ARCHS)


pytestmark = pytest.mark.skipif(
    not is_supported(), reason=f"gemm_a16w16_streamk needs one of {_SUPPORTED_ARCHS}"
)


def generate_inputs(M, N, K, dtype, layout="TN"):
    """layout[1] == 'T' gives an N-major weight, which takes the native-B path."""
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    if layout[1] == "T":
        w = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        w = torch.randn((N, K), dtype=dtype, device="cuda")
    return x, w


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 8)]
    x_vals += [(64, 4096, 4096), (128, 4096, 4096), (512, 4096, 4096)]
    x_vals += [(8192, 5120, 2880)]  # K % BLOCK_K == BLOCK_K / 2, takes the KTAIL drain
    x_vals += [(4096, 2880, 4096)]  # N not a multiple of the cluster tile
    x_vals += [(3000, 2048, 2048)]  # M not a multiple of the cluster tile
    x_vals += [(3000, 2880, 2048)]  # M and N both ragged
    return x_vals


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TN", "TT"])
def test_gemm_a16w16_streamk(M: int, N: int, K: int, layout):
    x, w = generate_inputs(M, N, K, torch.bfloat16, layout=layout)

    out = gemm_a16w16_streamk(x, w)
    torch_out = F.linear(x, w, bias=None)

    torch.testing.assert_close(out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", [(2048, 2048, 2048), (3000, 2880, 2048)])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("out_dtype", [None, torch.float32])
@pytest.mark.parametrize("layout", ["TN", "TT"])
def test_gemm_a16w16_streamk_dtypes(M, N, K, in_dtype, out_dtype, layout):
    """A 2-byte C must accumulate in fp32. An fp16 accumulator does not compile.

    out_dtype None keeps the input dtype, which also turns on the aliased C store. fp32 is
    4 bytes, so it takes the masked store instead.
    """
    x, w = generate_inputs(M, N, K, in_dtype, layout=layout)

    out = gemm_a16w16_streamk(x, w, dtype=out_dtype)

    assert out.dtype == (out_dtype or in_dtype)
    torch_out = F.linear(x.float(), w.float(), bias=None).to(out.dtype)
    torch.testing.assert_close(out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", [(2048, 2048, 2048)])
def test_gemm_a16w16_streamk_out_tensor(M: int, N: int, K: int):
    """A caller-supplied y must be written in place and returned."""
    x, w = generate_inputs(M, N, K, torch.bfloat16)
    y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

    out = gemm_a16w16_streamk(x, w, y=y)

    assert out.data_ptr() == y.data_ptr()
    torch.testing.assert_close(y, F.linear(x, w, bias=None), atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "per_cta, ctas, expect_a_interval, expect_b_interval",
    [
        ((256, 256, 128), (4, 4), 128, 256),
        ((128, 128, 128), (2, 2), 128, 128),
        ((64, 256, 128), (1, 1), 128, 256),
    ],
)
def test_layout_builders(per_cta, ctas, expect_a_interval, expect_b_interval):
    """The shared layouts must pad on the per-CTA innermost extent, under the TDM cap of 256.

    An over-cap interval makes the TDM descriptor fail to encode, and a CGA-wide interval breaks
    the C-stage reinterpret, so pin both here rather than wait for a compile error.
    """
    from aiter.ops.triton._gluon_kernels.gfx1250.layouts import (
        TDM_PAD_INTERVAL_MAX,
        operand_bundle_cga,
        wmma_layout,
    )

    BM, BN, BK = per_cta
    wmma = wmma_layout((BM, BN), ctas, 4)
    _, shared_a = operand_bundle_cga(wmma, 0, (BM, BK), ctas)
    _, shared_b = operand_bundle_cga(wmma, 1, (BK, BN), ctas)

    a_interval = shared_a.interval_padding_pairs[0][0]
    b_interval = shared_b.interval_padding_pairs[0][0]
    assert a_interval == expect_a_interval
    assert b_interval == expect_b_interval
    assert b_interval <= TDM_PAD_INTERVAL_MAX

    # A splits over M only and B over N only, so each gets zero bases on the other axis.
    assert all(row[1] == 0 for row in shared_a.cga_layout)
    assert all(row[0] == 0 for row in shared_b.cga_layout)

    # with_identity_for needs the cluster-wide shape, not the per-CTA tile.
    assert shared_a.shape == [BM * ctas[0], BK]
    assert shared_b.shape == [BK, BN * ctas[1]]

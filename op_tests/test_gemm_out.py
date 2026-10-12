# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Contract of the caller-owned ``out=`` buffer of gemm_a8w8_bpreshuffle and
gemm_a4w4: the result lands in (and aliases) ``out``, matches the call without
``out``, and a buffer the kernels cannot write densely is rejected up front."""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops import gemm_op_a4w4
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GEMM out= tests require a CUDA/HIP device"
)


def _gfx():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime()


def _a8w8_inputs(m, n, k):
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=dtypes.bf16, device="cuda")
    w = torch.randn((n, k), dtype=dtypes.bf16, device="cuda")
    xq, x_scale = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    wq, w_scale = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
    return xq, shuffle_weight(wq, layout=(16, 16)), x_scale, w_scale


@pytest.mark.parametrize("m,n,k", [(1, 256, 512), (37, 1280, 512), (256, 7168, 3584)])
def test_a8w8_bpreshuffle_out_matches_and_aliases(m, n, k):
    xq, wq, x_scale, w_scale = _a8w8_inputs(m, n, k)
    ref = aiter.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, None, dtypes.bf16)
    out = torch.full((m, n), float("nan"), dtype=dtypes.bf16, device="cuda")

    res = aiter.gemm_a8w8_bpreshuffle(
        xq, wq, x_scale, w_scale, None, dtypes.bf16, out=out
    )

    assert res.data_ptr() == out.data_ptr()
    assert res.shape == (m, n)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


def test_a8w8_bpreshuffle_out_under_torch_compile():
    m, n, k = 64, 1280, 512
    xq, wq, x_scale, w_scale = _a8w8_inputs(m, n, k)
    ref = aiter.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, None, dtypes.bf16)

    @torch.compile(backend="aot_eager", fullgraph=True)
    def fn(xq, wq, x_scale, w_scale, out):
        aiter.gemm_a8w8_bpreshuffle(
            xq, wq, x_scale, w_scale, None, dtypes.bf16, out=out
        )
        return out * 1

    out = torch.empty((m, n), dtype=dtypes.bf16, device="cuda")
    res = fn(xq, wq, x_scale, w_scale, out)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(res, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "make_out",
    [
        pytest.param(
            lambda m, n: torch.empty((m + 1, n), dtype=dtypes.bf16), id="shape"
        ),
        pytest.param(lambda m, n: torch.empty((m, n), dtype=dtypes.fp16), id="dtype"),
        pytest.param(
            lambda m, n: torch.empty((m, 2 * n), dtype=dtypes.bf16)[:, :n],
            id="column-slice",
        ),
        pytest.param(
            lambda m, n: torch.empty((n, m), dtype=dtypes.bf16).t(), id="transposed"
        ),
    ],
)
def test_a8w8_bpreshuffle_out_rejected(make_out):
    m, n, k = 32, 256, 512
    xq, wq, x_scale, w_scale = _a8w8_inputs(m, n, k)
    with torch.device("cuda"):
        out = make_out(m, n)
    base = out._base if out._base is not None else out
    before = base.clone()

    with pytest.raises(ValueError, match="out must be a contiguous tensor"):
        aiter.gemm_a8w8_bpreshuffle(
            xq, wq, x_scale, w_scale, None, dtypes.bf16, out=out
        )
    assert torch.equal(base.view(torch.int16), before.view(torch.int16))


def _a4w4_inputs(m, n, k):
    torch.manual_seed(0)
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((m, k), dtype=dtypes.bf16, device="cuda")
    w = torch.randn((n, k), dtype=dtypes.bf16, device="cuda")
    xq, x_scale = quant(x, shuffle=True)
    wq, w_scale = quant(w, shuffle=True)
    return xq, shuffle_weight(wq, layout=(16, 16)), x_scale, w_scale


a4w4_only = pytest.mark.skipif(
    torch.cuda.is_available() and _gfx() != "gfx950",
    reason="gemm_a4w4 out= runs the gfx950 kernels",
)


@a4w4_only
@pytest.mark.parametrize("m", [1, 32, 45])
def test_a4w4_padded_out_matches_and_aliases(m):
    n, k = 512, 512
    xq, wq, x_scale, w_scale = _a4w4_inputs(m, n, k)
    ref = aiter.gemm_a4w4(xq, wq, x_scale, w_scale, bpreshuffle=True)
    m_padded = (m + 31) // 32 * 32
    out = torch.full((m_padded, n), float("nan"), dtype=dtypes.bf16, device="cuda")

    res = aiter.gemm_a4w4(xq, wq, x_scale, w_scale, bpreshuffle=True, out=out)

    assert res.shape == (m, n)
    assert res.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out[:m], ref, rtol=0, atol=0)


@a4w4_only
@pytest.mark.parametrize(
    "make_out",
    [
        pytest.param(
            lambda m, n: torch.empty((m, n), dtype=dtypes.bf16), id="unpadded"
        ),
        pytest.param(lambda m, n: torch.empty((64, n), dtype=dtypes.fp16), id="dtype"),
        pytest.param(
            lambda m, n: torch.empty((64, 2 * n), dtype=dtypes.bf16)[:, :n],
            id="column-slice",
        ),
    ],
)
def test_a4w4_out_rejected(make_out):
    m, n, k = 45, 512, 512
    xq, wq, x_scale, w_scale = _a4w4_inputs(m, n, k)
    with torch.device("cuda"):
        out = make_out(m, n)

    with pytest.raises(ValueError, match=r"\(64, 512\)"):
        aiter.gemm_a4w4(xq, wq, x_scale, w_scale, bpreshuffle=True, out=out)


def test_a4w4_out_not_supported_on_gfx1250(monkeypatch):
    monkeypatch.setattr(gemm_op_a4w4, "get_gfx", lambda: "gfx1250")
    a = torch.zeros((32, 256), dtype=torch.uint8, device="cuda")
    b = torch.zeros((256, 256), dtype=torch.uint8, device="cuda")
    scale = torch.zeros((32, 16), dtype=torch.uint8, device="cuda")
    out = torch.empty((32, 256), dtype=dtypes.bf16, device="cuda")

    with pytest.raises(NotImplementedError, match="gfx1250"):
        gemm_op_a4w4.gemm_a4w4(a, b, scale, scale, out=out)

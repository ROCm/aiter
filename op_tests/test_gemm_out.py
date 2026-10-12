# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Contract of gemm_a8w8_bpreshuffle_out and gemm_a4w4_out: the result lands in
the caller-owned ``out``, matches the allocating op bit for bit, only ``out`` is
declared mutable, and a buffer the kernels cannot write densely is rejected."""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops import gemm_op_a4w4
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GEMM out tests require a CUDA/HIP device"
)


def _gfx():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime()


def _mutated_args(op):
    schema = op.default._schema
    return [
        a.name
        for a in schema.arguments
        if a.alias_info is not None and a.alias_info.is_write
    ]


def _a8w8_inputs(m, n, k):
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=dtypes.bf16, device="cuda")
    w = torch.randn((n, k), dtype=dtypes.bf16, device="cuda")
    xq, x_scale = aiter.pertoken_quant(x, quant_dtype=dtypes.fp8)
    wq, w_scale = aiter.pertoken_quant(w, quant_dtype=dtypes.fp8)
    return xq, shuffle_weight(wq, layout=(16, 16)), x_scale, w_scale


def test_a8w8_bpreshuffle_out_declares_only_out_mutable():
    assert _mutated_args(torch.ops.aiter.gemm_a8w8_bpreshuffle_out) == ["out"]


@pytest.mark.parametrize(
    "m,n,k,dtype",
    [
        (1, 256, 512, dtypes.bf16),
        (37, 1280, 512, dtypes.bf16),
        (256, 7168, 3584, dtypes.bf16),
        (37, 1280, 512, dtypes.fp16),
    ],
)
def test_a8w8_bpreshuffle_out_matches(m, n, k, dtype):
    xq, wq, x_scale, w_scale = _a8w8_inputs(m, n, k)
    ref = aiter.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, None, dtype)
    out = torch.full((m, n), float("nan"), dtype=dtype, device="cuda")

    res = aiter.gemm_a8w8_bpreshuffle_out(out, xq, wq, x_scale, w_scale)

    assert res is None
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
def test_a8w8_bpreshuffle_out_under_torch_compile(backend):
    m, n, k = 64, 1280, 512
    xq, wq, x_scale, w_scale = _a8w8_inputs(m, n, k)
    ref = aiter.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, None, dtypes.bf16)

    @torch.compile(backend=backend, fullgraph=True)
    def fn(out, xq, wq, x_scale, w_scale):
        aiter.gemm_a8w8_bpreshuffle_out(out, xq, wq, x_scale, w_scale)
        return out * 1

    torch._dynamo.reset()
    out = torch.full((m, n), float("nan"), dtype=dtypes.bf16, device="cuda")
    res = fn(out, xq, wq, x_scale, w_scale)
    torch.testing.assert_close(out, ref, rtol=0, atol=0)
    torch.testing.assert_close(res, ref, rtol=0, atol=0)


@pytest.mark.parametrize(
    "make_out",
    [
        pytest.param(
            lambda m, n: torch.empty((m + 1, n), dtype=dtypes.bf16), id="shape"
        ),
        pytest.param(lambda m, n: torch.empty((m, n), dtype=dtypes.fp32), id="dtype"),
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

    with pytest.raises(ValueError, match="out "):
        aiter.gemm_a8w8_bpreshuffle_out(out, xq, wq, x_scale, w_scale)
    assert torch.equal(base.view(torch.uint8), before.view(torch.uint8))


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
    reason="gemm_a4w4_out runs the gfx950 kernels",
)


def test_a4w4_out_declares_only_out_mutable():
    assert _mutated_args(torch.ops.aiter.gemm_a4w4_out) == ["out"]


@a4w4_only
@pytest.mark.parametrize("m", [1, 32, 45])
def test_a4w4_padded_out_matches(m):
    n, k = 512, 512
    xq, wq, x_scale, w_scale = _a4w4_inputs(m, n, k)
    ref = aiter.gemm_a4w4(xq, wq, x_scale, w_scale, bpreshuffle=True)
    m_padded = (m + 31) // 32 * 32
    out = torch.full((m_padded, n), float("nan"), dtype=dtypes.bf16, device="cuda")

    res = aiter.gemm_a4w4_out(out, xq, wq, x_scale, w_scale)

    assert res is None
    torch.testing.assert_close(out[:m], ref, rtol=0, atol=0)


@a4w4_only
def test_a4w4_padded_out_under_torch_compile():
    # aot_eager only: gemm_a4w4 already fails to lower under inductor on main
    # ("auto_functionalized_v2 was not removed"), with or without out.
    m, n, k = 45, 512, 512
    xq, wq, x_scale, w_scale = _a4w4_inputs(m, n, k)
    ref = aiter.gemm_a4w4(xq, wq, x_scale, w_scale, bpreshuffle=True)

    @torch.compile(backend="aot_eager", fullgraph=True)
    def fn(out, xq, wq, x_scale, w_scale):
        aiter.gemm_a4w4_out(out, xq, wq, x_scale, w_scale)
        return out[:m] * 1

    torch._dynamo.reset()
    out = torch.full((64, n), float("nan"), dtype=dtypes.bf16, device="cuda")
    res = fn(out, xq, wq, x_scale, w_scale)
    torch.testing.assert_close(out[:m], ref, rtol=0, atol=0)
    torch.testing.assert_close(res, ref, rtol=0, atol=0)


@a4w4_only
@pytest.mark.parametrize(
    "make_out",
    [
        pytest.param(
            lambda m, n: torch.empty((m, n), dtype=dtypes.bf16), id="unpadded"
        ),
        pytest.param(lambda m, n: torch.empty((64, n), dtype=dtypes.fp32), id="dtype"),
        # gemm_a4w4_asm only writes bf16, and it serves every shape without a
        # CK tuned row, so fp16 is rejected up front instead of failing at launch.
        pytest.param(lambda m, n: torch.empty((64, n), dtype=dtypes.fp16), id="fp16"),
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

    with pytest.raises(ValueError, match="out "):
        aiter.gemm_a4w4_out(out, xq, wq, x_scale, w_scale)


def test_a4w4_out_not_supported_on_gfx1250(monkeypatch):
    monkeypatch.setattr(gemm_op_a4w4, "get_gfx", lambda: "gfx1250")
    a = torch.zeros((32, 256), dtype=torch.uint8, device="cuda")
    b = torch.zeros((256, 256), dtype=torch.uint8, device="cuda")
    scale = torch.zeros((32, 16), dtype=torch.uint8, device="cuda")
    out = torch.empty((32, 256), dtype=dtypes.bf16, device="cuda")

    with pytest.raises(NotImplementedError, match="gfx1250"):
        gemm_op_a4w4.gemm_a4w4_out(out, a, b, scale, scale)

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""torch.compile contracts for the allocating A4W4 GEMM entry points."""

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a4w4 import gemm_a4w4
from aiter.ops.shuffle import shuffle_weight


def _written_arguments(op):
    return [
        argument.name
        for argument in op.default._schema.arguments
        if argument.alias_info is not None and argument.alias_info.is_write
    ]


@pytest.mark.parametrize("op_name", ["gemm_a4w4", "gemm_a4w4o8"])
def test_allocating_a4w4_ops_have_read_only_schemas(op_name):
    assert _written_arguments(getattr(torch.ops.aiter, op_name)) == []


_GFX = get_gfx() if torch.cuda.is_available() else None


@pytest.mark.skipif(_GFX != "gfx950", reason="requires an MI350-series GPU")
def test_gemm_a4w4_compiles_with_inductor(monkeypatch):
    """False input mutations used to leave auto_functionalized_v2 in this graph."""
    import torch._inductor.config as inductor_config

    monkeypatch.setattr(inductor_config, "force_disable_caches", True)

    m, n, k = 64, 256, 512
    quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x = torch.randn((m, k), dtype=dtypes.bf16, device="cuda")
    w = torch.randn((n, k), dtype=dtypes.bf16, device="cuda")
    xq, x_scale = quant(x, shuffle=True)
    wq, w_scale = quant(w, shuffle=True)
    wq = shuffle_weight(wq)

    def fn(xq, wq, x_scale, w_scale):
        return gemm_a4w4(xq, wq, x_scale, w_scale, dtype=dtypes.bf16)

    expected = fn(xq, wq, x_scale, w_scale)
    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="inductor", fullgraph=True)
    actual = compiled(xq, wq, x_scale, w_scale)

    torch.testing.assert_close(actual, expected)

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.shuffle import shuffle_weight
from op_tests.triton_tests.gemm.basic.test_gemm_a8w8 import (
    generate_gemm_a8w8_inputs,
    run_torch,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or arch_info.get_arch() != "gfx1201",
    reason="gfx1201 FP8 bpreshuffle Triton path",
)


def make_inputs(m, n, k, row_strided=False):
    """Shared a8w8 generator; row_strided gives x a row stride wider than K."""
    x, w, _, sx, sw, _, _ = generate_gemm_a8w8_inputs(
        m, n, k * (2 if row_strided else 1), torch.float8_e4m3fn, torch.bfloat16
    )
    if row_strided:
        x, w = x[:, :k], w[:, :k].contiguous()
    return x, w, sx, sw.T  # the bpreshuffle entry takes w_scale as (N, 1)


def reference(x, w, sx, sw, dtype):
    return run_torch(x, w, sx, sw.T, dtype=dtype)


@pytest.mark.parametrize("m", [1, 4, 8, 16, 32, 129])
@pytest.mark.parametrize(
    "n,k", [(16, 32), (32, 96), (64, 65), (1536, 1024), (8192, 2048)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_public_bpreshuffle(m, n, k, dtype):
    x, w, sx, sw = make_inputs(m, n, k)
    shuffled = shuffle_weight(w, pad_k_to=32)
    output = gemm_a8w8_bpreshuffle(x, shuffled, sx, sw, dtype=dtype)
    assert output.shape == (m, n)
    assert output.dtype == dtype
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output, reference(x, w, sx, sw, dtype), rtol=0.02, atol=0.01
    )


@pytest.mark.parametrize("m,k", [(1, 65), (3, 513), (32, 2048)])
def test_resolved_config_and_output(m, k):
    x, w, sx, sw = make_inputs(m, 80, k, row_strided=True)
    out = torch.empty((m, 80), device="cuda", dtype=torch.bfloat16)
    result = gemm_a8w8(
        x,
        shuffle_weight(w, pad_k_to=32),
        sx,
        sw.T,
        y=out,
        b_preshuffled=True,
    )
    assert result is out
    torch.testing.assert_close(
        result, reference(x, w, sx, sw, out.dtype), rtol=0.02, atol=0.01
    )


def test_zero_input_and_graph_replay():
    x, w, sx, sw = make_inputs(4, 64, 65)
    shuffled = shuffle_weight(w, pad_k_to=32)
    # Warm all kernels before capture.
    for _ in range(3):
        gemm_a8w8_bpreshuffle(x, shuffled, sx, sw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gemm_a8w8_bpreshuffle(x, shuffled, sx, sw)
    graph.replay()
    torch.testing.assert_close(
        output, reference(x, w, sx, sw, output.dtype), rtol=0.02, atol=0.01
    )
    x.copy_(torch.zeros_like(x))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(output) == 0


def test_torch_compile_public_entry():
    x, w, sx, sw = make_inputs(4, 64, 96)
    compiled = torch.compile(gemm_a8w8_bpreshuffle, backend="eager", fullgraph=True)
    output = compiled(x, shuffle_weight(w), sx, sw)
    torch.testing.assert_close(
        output, reference(x, w, sx, sw, output.dtype), rtol=0.02, atol=0.01
    )

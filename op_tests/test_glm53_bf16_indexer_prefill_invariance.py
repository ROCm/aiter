# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared indexer query rows must agree in full and cache-reuse prefills."""

import pytest
import torch

from aiter.execution_context import execution_phase, execution_role
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.tuned_gemm import gemm_a16w16, tgemm

M_VALUES = (
    1,
    2,
    8,
    16,
    32,
    64,
    96,
    128,
    160,
    512,
    1023,
    1024,
    1025,
    1535,
    1536,
    1537,
    2048,
    4096,
    5952,
    6048,
    8192,
)


@pytest.mark.parametrize("seed", (0, 898, 42))
@pytest.mark.parametrize("wide_values", (False, True))
def test_glm53_bf16_indexer_prefill_invariance(seed, wide_values):
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    if get_gfx_runtime() != "gfx950" or get_cu_num() != 256:
        pytest.skip("Requires gfx950 with 256 CUs")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    n, k = 4096, 1536
    x = torch.randn(
        max(M_VALUES), k, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    w = torch.randn(n, k, generator=generator, device="cuda", dtype=torch.bfloat16)
    if wide_values:
        for tensor in (x, w):
            exponents = torch.randint(
                -12,
                13,
                tensor.shape,
                generator=generator,
                device="cuda",
                dtype=torch.int32,
            )
            tensor.mul_(2.0**exponents)
    with execution_role("target"), execution_phase("prefill"):
        full = gemm_a16w16(x, w)
        for m in M_VALUES:
            actual = gemm_a16w16(x[:m].clone(), w)
            torch.testing.assert_close(actual, gemm_a16w16(x[:m], w), rtol=0, atol=0)
            torch.testing.assert_close(
                actual, full[:m], rtol=0, atol=0, msg=f"Cross-M drift at M={m}"
            )

        indices = torch.tensor([6047, 32, 0, 512, 7], device="cuda")
        selected = x[indices]
        expected = full[indices]
        torch.testing.assert_close(gemm_a16w16(selected, w), expected, rtol=0, atol=0)
        padded = torch.zeros(71, k, device="cuda", dtype=x.dtype)
        padded[33:38] = selected
        torch.testing.assert_close(
            gemm_a16w16(padded, w)[33:38], expected, rtol=0, atol=0
        )
        reference = (selected.double() @ w.double().T).to(x.dtype)
        torch.testing.assert_close(expected, reference, rtol=1e-2, atol=1e-2)

        out = torch.empty_like(expected)
        actual = tgemm.mm(selected, w, out=out)
        assert actual.data_ptr() == out.data_ptr()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        captures = []
        for m in (1, 1025, 1537):
            graph_input = x[:m].clone()
            graph_out = torch.empty_like(full[:m])
            tgemm.mm(graph_input, w, out=graph_out)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                tgemm.mm(graph_input, w, out=graph_out)
            captures.append((graph, graph_input, graph_out, full[:m]))
    # Replay after context exit still uses the captured prefill kernel.
    for graph, _, graph_out, graph_expected in captures:
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(graph_out, graph_expected, rtol=0, atol=0)

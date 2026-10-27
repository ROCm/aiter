# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare the same BF16 rows across decode, verification and prefill sizes."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.tuned_gemm import gemm_a16w16, tgemm

M_VALUES = (1, 2, 4, 8, 16, 32, 57, 64, 65, 71, 95, 96, 97, 128, 193, 320, 513, 8192)
FAMILIES = ((4096, 8192), (8192, 128), (32768, 512), (288, 4096), (128, 4096))


@pytest.mark.parametrize("n,k", FAMILIES)
@pytest.mark.parametrize("seed", (0, 898))
def test_glm53_bf16_batch_invariance(n, k, seed):
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    if get_gfx_runtime() != "gfx950" or get_cu_num() != 256:
        pytest.skip("Requires gfx950 with 256 CUs")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        max(M_VALUES), k, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    w = torch.randn(n, k, generator=generator, device="cuda", dtype=torch.bfloat16)
    full = gemm_a16w16(x, w)
    for m in M_VALUES:
        actual = gemm_a16w16(x[:m], w)
        torch.testing.assert_close(actual, gemm_a16w16(x[:m], w), rtol=0, atol=0)
        torch.testing.assert_close(
            actual, full[:m], rtol=0, atol=0, msg=f"Cross-M drift at M={m}"
        )

    # Move shared rows to different positions and change their neighbours.
    indices = torch.tensor([8191, 32, 0, 512, 7], device="cuda")
    selected = x[indices]
    expected = full[indices]
    torch.testing.assert_close(gemm_a16w16(selected, w), expected, rtol=0, atol=0)
    padded = torch.zeros(71, k, device="cuda", dtype=x.dtype)
    padded[33:38] = selected
    torch.testing.assert_close(gemm_a16w16(padded, w)[33:38], expected, rtol=0, atol=0)
    reference = (selected.float() @ w.float().T).to(x.dtype)
    torch.testing.assert_close(expected, reference, rtol=1e-2, atol=1e-2)

    out = torch.empty_like(expected)
    result = tgemm.mm(selected, w, out=out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # Warm up the exact call before capture; replay must keep the same results.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tgemm.mm(selected, w, out=out)
    for _ in range(2):
        graph.replay()
        torch.testing.assert_close(out, expected, rtol=0, atol=0)

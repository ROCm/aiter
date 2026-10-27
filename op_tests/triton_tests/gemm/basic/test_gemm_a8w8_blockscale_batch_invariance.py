# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GLM-5.3 q_b projection: shared rows must agree across M buckets."""

import pytest
import torch

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.types import get_fp8_dtypes

M_VALUES = (
    1,
    2,
    4,
    8,
    9,
    15,
    16,
    17,
    31,
    32,
    33,
    51,
    63,
    64,
    65,
    71,
    128,
    129,
    256,
    257,
    512,
    513,
    1024,
    4096,
    8192,
)


@pytest.mark.parametrize("seed", (0, 898, 42))
@pytest.mark.parametrize("wide_scales", (False, True))
def test_glm53_qb_batch_invariance(seed, wide_scales):
    if not torch.cuda.is_available() or get_arch() != "gfx950":
        pytest.skip("Requires gfx950")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    n, k = 16384, 1536
    fp8 = get_fp8_dtypes()[1]
    x = (
        (
            torch.randn(
                max(M_VALUES),
                k,
                generator=generator,
                device="cuda",
                dtype=torch.float16,
            )
            * 143
        )
        .clamp(-448, 448)
        .to(fp8)
    )
    w = (
        (
            torch.randn(n, k, generator=generator, device="cuda", dtype=torch.float16)
            * 87
        )
        .clamp(-448, 448)
        .to(fp8)
    )
    xs = (
        torch.rand(
            max(M_VALUES),
            k // 128,
            generator=generator,
            device="cuda",
            dtype=torch.float32,
        )
        * 0.0058
        + 0.0034
    )
    ws = (
        torch.rand(
            n // 128, k // 128, generator=generator, device="cuda", dtype=torch.float32
        )
        * 0.00036
        + 0.0002
    )
    if wide_scales:
        for scale in (xs, ws):
            exponents = torch.randint(
                -6,
                7,
                scale.shape,
                generator=generator,
                device="cuda",
                dtype=torch.int32,
            )
            scale.mul_(2.0**exponents)
    full = gemm_a8w8_blockscale(x, w, xs, ws, dtype=torch.bfloat16)
    for m in M_VALUES:
        actual = gemm_a8w8_blockscale(
            x[:m].clone(), w, xs[:m].clone(), ws, dtype=torch.bfloat16
        )
        repeated = gemm_a8w8_blockscale(x[:m], w, xs[:m], ws, dtype=torch.bfloat16)
        torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
        torch.testing.assert_close(
            actual, full[:m], rtol=0, atol=0, msg=f"Cross-M drift at M={m}"
        )

    # Use different positions and neighbours, with scales moved with their rows.
    indices = torch.tensor([8191, 32, 0, 512, 7], device="cuda")
    selected = x.float()[indices].to(fp8)
    selected_scale = xs[indices]
    expected = full[indices]
    out = torch.empty_like(expected)
    actual = gemm_a8w8_blockscale(
        selected, w, selected_scale, ws, dtype=torch.bfloat16, y=out
    )
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    padded = torch.zeros(71, k, device="cuda", dtype=torch.float16)
    padded[33:38] = selected.to(torch.float16)
    padded_scale = torch.ones(71, k // 128, device="cuda", dtype=torch.float32)
    padded_scale[33:38] = selected_scale
    padded_out = gemm_a8w8_blockscale(
        padded.to(fp8), w, padded_scale, ws, dtype=torch.bfloat16
    )
    torch.testing.assert_close(padded_out[33:38], expected, rtol=0, atol=0)
    if wide_scales:
        a = selected.double() * selected_scale.double().repeat_interleave(128, dim=1)
        b = w.double() * ws.double().repeat_interleave(128, dim=0).repeat_interleave(
            128, dim=1
        )
        reference = a @ b.T
        # Cancellation needs an absolute FP32 dot-product error bound,
        # including two scale multiplications and final BF16 rounding.
        u = torch.finfo(torch.float32).eps / 2
        gamma = (k + 2) * u / (1 - (k + 2) * u)
        output_u = torch.finfo(expected.dtype).eps / 2
        bound = (1 + output_u) * gamma * (
            a.abs() @ b.abs().T
        ) + output_u * reference.abs()
        assert torch.all((expected.double() - reference).abs() <= bound)
    else:
        reference = (
            selected.float() * selected_scale.repeat_interleave(128, dim=1)
        ) @ (
            w.float() * ws.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
        ).T
        torch.testing.assert_close(
            expected, reference.to(expected.dtype), rtol=1e-2, atol=1e-2
        )

    for m in (1, 8, 33):
        graph_out = torch.empty_like(full[:m])
        gemm_a8w8_blockscale(x[:m], w, xs[:m], ws, dtype=torch.bfloat16, y=graph_out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gemm_a8w8_blockscale(
                x[:m], w, xs[:m], ws, dtype=torch.bfloat16, y=graph_out
            )
        for _ in range(2):
            graph.replay()
            torch.testing.assert_close(graph_out, full[:m], rtol=0, atol=0)

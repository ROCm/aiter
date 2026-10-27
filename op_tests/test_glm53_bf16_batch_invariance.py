# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare the same BF16 rows across decode, verification and prefill sizes."""

import pytest
import torch

import aiter.tuned_gemm as tuned
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.tuned_gemm import gemm_a16w16, tgemm

# CSV and Triton M buckets are upper bounds. Exercise both sides of each one.
BOUNDARIES = (8, 16, 32, 64, 96, 128, 256, 512, 1024, 2048, 4096)
M_VALUES = sorted(
    {1, 2, 4, 57, 71, 193, 320, 8192}
    | {m + delta for m in BOUNDARIES for delta in (-1, 0, 1)}
)
FAMILIES = ((4096, 8192), (8192, 128), (32768, 512), (288, 4096), (128, 4096))


@pytest.mark.parametrize("n,k", FAMILIES)
@pytest.mark.parametrize("seed", (0, 898, 42))
@pytest.mark.parametrize("wide_exponents", (False, True))
def test_glm53_bf16_batch_invariance(n, k, seed, wide_exponents, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    if get_gfx_runtime() != "gfx950" or get_cu_num() != 256:
        pytest.skip("Requires gfx950 with 256 CUs")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        max(M_VALUES), k, generator=generator, device="cuda", dtype=torch.bfloat16
    )
    w = torch.randn(n, k, generator=generator, device="cuda", dtype=torch.bfloat16)
    if wide_exponents:
        for tensor in (x, w):
            exponents = torch.randint(
                -12, 13, tensor.shape, generator=generator, device="cuda"
            )
            tensor.copy_(torch.ldexp(tensor.float(), exponents))

    # The exact family must also work without optional FlyDSL or OPUS kernels.
    def no_flydsl():
        raise ImportError("FlyDSL is not installed")

    monkeypatch.setattr(tuned, "_get_flydsl_gemm_kernels", no_flydsl)
    monkeypatch.setattr(tuned, "_opus_launch", None)
    tuned.get_GEMM_A16W16_config.cache_clear()
    full = gemm_a16w16(x, w)
    permutation = torch.randperm(x.shape[0], generator=generator, device="cuda")
    permuted = x[permutation].clone()
    for m in M_VALUES:
        actual = gemm_a16w16(x[:m], w)
        assert torch.equal(actual, gemm_a16w16(x[:m].clone(), w))
        assert torch.equal(actual, full[:m]), f"Cross-M drift at M={m}"
        assert torch.equal(
            gemm_a16w16(permuted[:m], w), full[permutation[:m]]
        ), f"Row permutation drift at M={m}"

    # Move shared rows to different positions and change their neighbours.
    indices = torch.tensor(
        [8191, 32, 0, 512, 7, 15, 63, 64, 127, 128, 255, 511, 1023, 1024, 2047, 4095],
        device="cuda",
    )
    selected = x[indices]
    expected = full[indices]
    assert torch.equal(gemm_a16w16(selected, w), expected)
    for m in (71, 129, 513, 1025, 8193):
        padded = torch.zeros(m, k, device="cuda", dtype=x.dtype)
        padded[33:49] = selected
        assert torch.equal(gemm_a16w16(padded, w)[33:49], expected)
    if wide_exponents:
        # Cancellation needs an absolute dot-product bound, including BF16 rounding.
        # BF16 products are exact in FP32; gamma_K bounds the accumulation error.
        reference = selected.double() @ w.double().T
        magnitude = selected.double().abs() @ w.double().abs().T
        u32 = torch.finfo(torch.float32).eps / 2
        ubf16 = torch.finfo(torch.bfloat16).eps / 2
        gamma = k * u32 / (1 - k * u32)
        bound = (1 + ubf16) * gamma * magnitude + ubf16 * reference.abs()
        assert torch.all((expected.double() - reference).abs() <= bound)
    else:
        reference = (selected.float() @ w.float().T).to(x.dtype)
        torch.testing.assert_close(expected, reference, rtol=1e-2, atol=1e-2)

    guarded = torch.full((expected.shape[0] + 2, n), -123, device="cuda", dtype=x.dtype)
    out = guarded[1:-1]
    result = tgemm.mm(selected, w, out=out)
    assert result.data_ptr() == out.data_ptr()
    assert torch.equal(out, expected)
    assert (guarded[0] == -123).all() and (guarded[-1] == -123).all()
    # Warm up the exact call before capture; replay must keep the same results.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        tgemm.mm(selected, w, out=out)
    for _ in range(2):
        graph.replay()
        assert torch.equal(out, expected)
        assert (guarded[0] == -123).all() and (guarded[-1] == -123).all()
    tuned.get_GEMM_A16W16_config.cache_clear()

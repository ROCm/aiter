# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16 import _is_gluon_available, gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.utils.types import str_to_torch_dtype


def is_gluon_supported():
    """gluon a16w16 kernels are only available on supported archs (gfx1250)."""
    return _is_gluon_available()


def _skip_if_triton_on_gfx1250(backend):
    """gfx1250 only ships gluon-format a16w16 configs, so the triton backend has
    no usable config there; skip the triton backend on gfx1250."""
    if backend != "triton":
        return
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if "gfx1250" in (get_arch() or ""):
        pytest.skip("triton backend has no gfx1250 a16w16 config (gluon-only arch)")


def generate_gemm_a16w16_inputs(M, N, K, dtype, layout="TN", output=True, bias=False):
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        weight = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        weight = torch.randn((N, K), dtype=dtype, device="cuda")

    bias_tensor = None
    if bias:
        bias_tensor = torch.empty((N), dtype=dtype, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return x, weight, bias_tensor, out_dtype, y


def get_x_vals():
    x_vals = [(1, 1, 1)]  # minimal case
    x_vals += [(3, 5, 2)]  # irregular shape
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(2**i, 256, 7168) for i in range(5, 9)]  # DSR1 router GEMM
    # GPT-OSS-120B attention projections
    x_vals += [(2**i, 5120, 2880) for i in range(5, 9)]  # GPTOSS QKV input projection
    x_vals += [(2048, 5120, 2880)]  # GPTOSS QKV @ M=2048
    x_vals += [(8192, 5120, 2880)]  # GPTOSS QKV @ M=8192
    x_vals += [(2**i, 2880, 4096) for i in range(5, 9)]  # output projection
    x_vals += [(2**i, 128, 2880) for i in range(5, 9)]  # Router GEMM
    x_vals += [(v, 106496, 16384) for v in (256, 4096)]  # LL3 405B FC1
    return x_vals


# Test plain BF16 GEMMs - the most common types.
@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16(M: int, N: int, K: int, backend, kernel_type):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    _skip_if_triton_on_gfx1250(backend)

    x, w, _, _out_dtype, _y = generate_gemm_a16w16_inputs(
        M,
        N,
        K,
        dtype=torch.bfloat16,
        output=False,
    )

    torch_out = F.linear(x, w, bias=None)

    triton_out = gemm_a16w16(x, w, backend=backend, kernel_type=kernel_type)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


# Output larger than 2**31 bytes (Wan2.2 FFN up-projection at 720p x 193 frames).
@pytest.mark.parametrize("M, N, K", [(90000, 13824, 5120)])
def test_gemm_a16_w16_large_output(M: int, N: int, K: int):
    if not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    assert M * N * 2 > 2**31

    x, w, _, _, _ = generate_gemm_a16w16_inputs(
        M, N, K, dtype=torch.bfloat16, output=False
    )
    # NaN-fill so any row the kernel never writes fails the comparison.
    y = torch.full((M, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    gemm_a16w16(x, w, y=y, backend="gluon")

    boundary_row = 2**31 // (N * 2)
    for rows in (
        slice(0, 1024),
        slice(boundary_row - 512, boundary_row + 512),
        slice(M - 1024, M),
    ):
        torch.testing.assert_close(y[rows], F.linear(x[rows], w), atol=1e-1, rtol=1e-2)


# Smaller set for testing activations, setting the output tensor and dtype
def get_fewer_x_vals():
    x_vals = [(16, 1024, 1024)]
    x_vals += [(128, 8192, 512)]
    x_vals += [(256, 512, 8192)]
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 5, 8)]
    return x_vals


# A smaller set of shapes that tests fused activations, different dtypes
# and output tensor arg. We don't want the larger set above to test
# all these combinations.
@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu"])
@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16_activation(
    M: int, N: int, K: int, dtype, output, activation, backend, kernel_type
):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    _skip_if_triton_on_gfx1250(backend)

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M,
        N,
        K,
        dtype,
        output=output,
    )

    torch_out = F.linear(x, w, bias=None)
    if activation == "gelu":
        torch_out = F.gelu(torch_out)
    elif activation == "gelu_tanh":
        torch_out = F.gelu(torch_out, approximate="tanh")
    elif activation == "silu":
        torch_out = F.silu(torch_out)

    triton_out = gemm_a16w16(
        x,
        w,
        None,
        out_dtype,
        y,
        activation=activation,
        backend=backend,
        kernel_type=kernel_type,
    )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16_layout(M: int, N: int, K: int, layout, backend, kernel_type):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    _skip_if_triton_on_gfx1250(backend)

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=False
    )

    torch_out = F.linear(x, w, bias=None)

    triton_out = gemm_a16w16(
        x, w, None, out_dtype, y, backend=backend, kernel_type=kernel_type
    )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_atomic(M: int, N: int, K: int, output):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=output
    )

    torch_out = F.linear(x, w, bias=None)

    # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
    if output:
        y = y.to(torch.float32).zero_()
        triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(torch.bfloat16)
    else:
        triton_out = gemm_a16w16_atomic(x, w, dtype=torch.float32).to(torch.bfloat16)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
def test_gemm_a16_w16_atomic_layout(M: int, N: int, K: int, layout):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=True
    )

    torch_out = F.linear(x, w, bias=None)

    y = y.to(torch.float32).zero_()
    triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(torch.bfloat16)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("strided", [False, True])
def test_gemm_a16_w16_atomic_accumulate(M: int, N: int, K: int, strided):
    """accumulate=True adds X @ W^T into y in place, like torch.addmm_."""
    torch.cuda.empty_cache()

    x, w, _, _out_dtype, _y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    base = torch.randn((M, 2 * N if strided else N), dtype=torch.float32, device="cuda")
    y = base[:, :N]
    torch_out = torch.addmm(y, x.float(), w.float().T)

    triton_out = gemm_a16w16_atomic(x, w, torch.float32, y, accumulate=True)

    assert triton_out.data_ptr() == y.data_ptr()
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("layout", ["TN", "TT"])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_gemm_a16w16_persistent_output(M: int, N: int, K: int, layout, output, backend):
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    torch.cuda.empty_cache()

    x, w, _, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=output
    )

    torch_out = F.linear(x, w, bias=None)

    if output:
        triton_out = gemm_a16w16(
            x, w, dtype=torch.bfloat16, y=y, backend=backend, persistent=True
        )
    else:
        triton_out = gemm_a16w16(
            x, w, dtype=torch.bfloat16, backend=backend, persistent=True
        )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


def test_gluon_layout_cache_key_and_eviction(monkeypatch):
    """The private cache keeps no tensors and separates every layout input."""
    import importlib

    mod = importlib.import_module("aiter.ops.triton.gemm.basic.gemm_a16w16")
    impl = importlib.import_module(
        "aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16"
    )
    cache = mod._get_gluon_layouts
    calls = {"wmma": 0, "shared": 0}

    def wmma(warps):
        calls["wmma"] += 1
        return (("wmma", warps), ("a", warps), ("b", warps))

    def shared(bm, bn, bk, layout, bits):
        calls["shared"] += 1
        return (("sa", bm, bn, bk, layout, bits), ("sb", bm, bn, bk, layout, bits))

    monkeypatch.setattr(impl, "create_wmma_layouts", wmma)
    monkeypatch.setattr(impl, "create_shared_layouts", shared)
    cache.cache_clear()
    try:
        base = (128, 128, 64, "TN", 16, 4)
        variants = [
            base,
            (64, 128, 64, "TN", 16, 4),
            (128, 64, 64, "TN", 16, 4),
            (128, 128, 128, "TN", 16, 4),
            (128, 128, 64, "NN", 16, 4),
            (128, 128, 64, "TN", 8, 4),
            (128, 128, 64, "TN", 16, 8),
        ]
        for args in variants:
            first = cache(*args)
            assert cache(*args) is first
            assert len(first) == 5
        assert calls == {"wmma": len(variants), "shared": len(variants)}
        assert cache.cache_info().maxsize == 128
        for i in range(129):
            cache(i + 1, 128, 64, "TN", 16, 4)
        assert cache.cache_info().currsize == 128
        misses = cache.cache_info().misses
        cache(1, 128, 64, "TN", 16, 4)
        assert cache.cache_info().misses == misses + 1
    finally:
        cache.cache_clear()


@pytest.mark.parametrize("layout", ["TN", "NN", "TT", "NT"])
def test_gluon_cached_layout_values(layout):
    import importlib

    mod = importlib.import_module("aiter.ops.triton.gemm.basic.gemm_a16w16")
    impl = importlib.import_module(
        "aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_a16w16"
    )
    cache = mod._get_gluon_layouts
    cache.cache_clear()
    try:
        for bits, warps in ((8, 4), (16, 4), (16, 8)):
            expected = (
                *impl.create_wmma_layouts(warps),
                *impl.create_shared_layouts(128, 128, 64, layout, bits),
            )
            got = cache(128, 128, 64, layout, bits, warps)
            assert got == expected
            assert cache(128, 128, 64, layout, bits, warps) is got
    finally:
        cache.cache_clear()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["TN", "NN", "TT", "NT"])
def test_gluon_layout_cache_replay(dtype, layout, monkeypatch):
    import dataclasses
    import gc
    import importlib
    import weakref

    if not torch.cuda.is_available() or not is_gluon_supported():
        pytest.skip("requires gfx1250")
    mod = importlib.import_module("aiter.ops.triton.gemm.basic.gemm_a16w16")
    cache = mod._get_gluon_layouts
    cache.cache_clear()
    try:
        # Both kernel variants and output-buffer modes. Smaller irregular M
        # avoids spending GPU memory on an unrelated large-matrix test.
        for kernel_type, rows in (("bandwidth_bound", 37), ("compute_bound", 129)):
            x, w, _, _, _ = generate_gemm_a16w16_inputs(
                rows, 128, 512, dtype=dtype, layout=layout, output=False
            )
            x *= 0.1
            w *= 0.1
            bias = torch.randn(128, dtype=dtype, device="cuda") * 0.1
            with monkeypatch.context() as patch:
                patch.setattr(mod, "_get_gluon_layouts", cache.__wrapped__)
                expected = gemm_a16w16(
                    x, w, bias, dtype=dtype, backend="gluon", kernel_type=kernel_type
                )
            y = torch.empty_like(expected)
            actual = gemm_a16w16(
                x, w, bias, dtype=dtype, y=y, backend="gluon", kernel_type=kernel_type
            )
            assert actual.data_ptr() == y.data_ptr()
            assert torch.equal(actual, expected)
            torch.testing.assert_close(
                actual.float(),
                x.float() @ w.float().T + bias.float(),
                atol=3e-2,
                rtol=3e-2,
            )
            # Dataclasses are frozen but contain nested lists. Verify
            # compiler/launch consumers do not mutate the shared metadata.
            tracked = []
            original_get = mod._get_gluon_layouts

            def observe(*args, original_get=original_get, tracked=tracked):
                layouts = original_get(*args)
                tracked.append(layouts)
                return layouts

            with monkeypatch.context() as patch:
                patch.setattr(mod, "_get_gluon_layouts", observe)
                gemm_a16w16(
                    x,
                    w,
                    bias,
                    dtype=dtype,
                    y=y,
                    backend="gluon",
                    kernel_type=kernel_type,
                )
            layouts = tracked[-1]
            snapshot = [dataclasses.asdict(v) for v in layouts]
            hashes = [hash(v) for v in layouts]
            before = cache.cache_info()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                gemm_a16w16(
                    x,
                    w,
                    bias,
                    dtype=dtype,
                    y=y,
                    backend="gluon",
                    kernel_type=kernel_type,
                )
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, expected)
            assert cache.cache_info().hits > before.hits
            # Reused metadata must not capture previous tensor values/pointers.
            x.neg_()
            changed = gemm_a16w16(
                x, w, bias, dtype=dtype, backend="gluon", kernel_type=kernel_type
            )
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, changed)
            assert [dataclasses.asdict(v) for v in layouts] == snapshot
            assert [hash(v) for v in layouts] == hashes
            # Only scalar configuration/layout metadata is cached.
            refs = [weakref.ref(v) for v in (x, w, bias, y)]
            del graph, actual, expected, changed, x, w, bias, y
            gc.collect()
            assert all(ref() is None for ref in refs)
    finally:
        cache.cache_clear()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gluon_layout_cache_compile(dtype):
    import importlib

    if not torch.cuda.is_available() or not is_gluon_supported():
        pytest.skip("requires gfx1250")
    mod = importlib.import_module("aiter.ops.triton.gemm.basic.gemm_a16w16")
    mod._get_gluon_layouts.cache_clear()
    torch._dynamo.reset()

    def call(x, w):
        return gemm_a16w16(x, w, dtype=dtype, backend="gluon")

    compiled = torch.compile(call, fullgraph=True, dynamic=True)
    try:
        for rows in (12, 257):
            torch.manual_seed(73)
            x = torch.randn(rows, 6144, device="cuda", dtype=dtype) * 0.1
            w = torch.randn(128, 6144, device="cuda", dtype=dtype) * 0.1
            expected = call(x, w)
            assert torch.equal(compiled(x, w), expected)
    finally:
        mod._get_gluon_layouts.cache_clear()


def test_gluon_layout_cache_concurrent():
    import importlib
    from concurrent.futures import ThreadPoolExecutor

    mod = importlib.import_module("aiter.ops.triton.gemm.basic.gemm_a16w16")
    cache = mod._get_gluon_layouts
    cache.cache_clear()
    try:
        keys = [
            (128, 128, 64, layout, 16, warps)
            for layout in ("TN", "NN", "TT", "NT")
            for warps in (4, 8)
        ]
        expected = {key: cache.__wrapped__(*key) for key in keys}
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda key: cache(*key), keys * 8))
        for key, value in zip(keys * 8, results):
            assert value == expected[key]
            assert tuple(map(hash, value)) == tuple(map(hash, expected[key]))
        assert cache.cache_info().currsize == len(keys)
    finally:
        cache.cache_clear()

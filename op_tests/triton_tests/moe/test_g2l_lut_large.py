# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""G2L integration: original FlyDSL <=512, Triton <=16384, then torch.

Standard correctness/perf sweep: op_tests/test_moe_g2l_lut.py
This file retains focused pytest regression checks.

Run: python -m pytest -q op_tests/triton_tests/moe/test_g2l_lut_large.py
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
from aiter.ops.triton.moe import g2l_lut as wrapper
from op_tests.test_moe_g2l_lut import SUPPORTED_GFX

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def reference(host, E):
    prefix = 0
    out = []
    for v in host.tolist():
        if v != 0:
            out.append(prefix)
            prefix += 1
        else:
            out.append(E)
    return torch.tensor(out, dtype=torch.int32, device="cpu")


def exact(out, ref):
    assert out.dtype == torch.int32 and out.is_contiguous()
    assert torch.equal(out.cpu(), ref)


@pytest.fixture(autouse=True)
def supported_arch():
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"G2L auxiliary kernels are unsupported on {get_gfx()}")


@pytest.fixture(autouse=True)
def context(monkeypatch):
    monkeypatch.delenv("AITER_G2L_TORCH", raising=False)
    monkeypatch.delenv("AITER_G2L_TRITON", raising=False)
    monkeypatch.setenv("AITER_TDM_DIRECT_EP_MASK", "1")
    monkeypatch.setattr(grouped, "_flydsl_dispatch_context", lambda: None)
    monkeypatch.setattr(grouped, "_G2L_COUNTER_CACHE", {})


@pytest.fixture
def no_fallback(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("unexpected FlyDSL/torch fallback")

    monkeypatch.setattr(grouped, "_get_compiled_g2l_lut", fail)
    monkeypatch.setattr(torch, "cumsum", fail)


def mask_data(n, dtype, pattern, stride):
    host = torch.zeros(n, device="cpu", dtype=dtype)
    if pattern == "one":
        host.fill_(1)
    elif pattern == "sentinel":
        host[:-1:4] = 1
    elif pattern == "last":
        host[-1] = 1
    elif pattern == "random":
        host = torch.randint(
            -3,
            4,
            (n,),
            generator=torch.Generator(device="cpu").manual_seed(n),
            device="cpu",
        ).to(dtype)
        if dtype == torch.int64:
            host[0], host[1] = 2**40, -(2**40)
        if dtype == torch.float32:
            host[0], host[1], host[2] = 0.5, -0.25, float("nan")
    backing = torch.empty(n * stride, dtype=dtype, device="cuda")
    mask = backing[::stride]
    mask.copy_(host)
    return host, mask, int((host != 0).sum())


@pytest.mark.parametrize("n", [8191, 8192, 8193, 16384])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool, torch.float32])
@pytest.mark.parametrize("pattern", ["zero", "one", "sentinel", "last", "random"])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("clear", [False, True])
def test_raw(n, dtype, pattern, stride, clear):
    host, mask, E = mask_data(n, dtype, pattern, stride)
    storage = torch.full((E + 2,), 777, dtype=torch.int32, device="cuda")
    counter = storage[1:-1]
    nvt = torch.tensor([4096], dtype=torch.int64, device="cuda")
    lut, count, nvr = wrapper.build_g2l_lut(
        mask, E, nvt, 8, counter=counter, clear_counter=clear
    )
    assert count is counter
    exact(lut, reference(host, E))
    exact(count, torch.full((E,), 0 if clear else 777, dtype=torch.int32, device="cpu"))
    exact(nvr, torch.tensor([32768], dtype=torch.int32, device="cpu"))
    assert storage[0].item() == storage[-1].item() == 777


@pytest.mark.parametrize("n", [8192, 8193])
@pytest.mark.parametrize("E", [128, 256, 1024, 8192])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool, torch.float32])
@pytest.mark.parametrize("direct", [False, True])
def test_integrated(n, E, dtype, direct, monkeypatch, no_fallback):
    host = torch.zeros(n, dtype=dtype, device="cpu")
    host[:E] = 1
    backing = torch.empty(2 * n, dtype=dtype, device="cuda")
    mask = backing[::2]
    mask.copy_(host)
    nvt = torch.tensor([4096], dtype=torch.int32, device="cuda")
    counter = torch.full((E,), 777, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(grouped, "route_counter_buffer", lambda E, device: counter)
    if direct:
        monkeypatch.setattr(grouped, "_flydsl_dispatch_context", lambda: object())
    lut, count, built = grouped._build_g2l_lut(mask, E, mask.device, nvt, 8)
    ep_nvr = built if built is not None else (nvt * 8).contiguous()
    assert built is not None and ep_nvr is built
    assert (count is counter) == direct
    exact(lut, reference(host, E))
    exact(
        count, torch.full((E,), 777 if direct else 0, dtype=torch.int32, device="cpu")
    )
    exact(ep_nvr, torch.tensor([32768], dtype=torch.int32, device="cpu"))


@pytest.mark.parametrize("n", [8192, 8193])
@pytest.mark.parametrize("E", [128, 1024, 8192])
def test_graph(n, E, no_fallback):
    mask = torch.zeros(n, dtype=torch.int32, device="cuda")
    nvt = torch.zeros(1, dtype=torch.int32, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            grouped._build_g2l_lut(mask, E, mask.device, nvt, 8)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        lut, counter, built = grouped._build_g2l_lut(mask, E, mask.device, nvt, 8)
        nvr = built if built is not None else (nvt * 8).contiguous()
    assert nvr is built and built is not None
    generator = torch.Generator(device="cpu").manual_seed(5906)
    for v in [0, 1, 4096, (2**31 - 1) // 8]:
        host = torch.zeros(n, dtype=torch.int32, device="cpu")
        host[torch.randperm(n, generator=generator, device="cpu")[:E]] = -3
        mask.copy_(host)
        nvt.fill_(v)
        for t in [lut, counter, nvr]:
            t.fill_(777)
        graph.replay()
        exact(lut, reference(host, E))
        exact(counter, torch.zeros(E, dtype=torch.int32, device="cpu"))
        exact(nvr, torch.tensor([v * 8], dtype=torch.int32, device="cpu"))


@pytest.mark.parametrize("n", [8192, 8193])
@pytest.mark.parametrize(
    "with_nvt,topk", [(False, None), (False, 8), (True, None), (True, 0), (True, 8)]
)
def test_optional(n, with_nvt, topk, no_fallback):
    mask = torch.ones(n, dtype=torch.int32, device="cuda")
    nvt = torch.tensor([37], dtype=torch.int64, device="cuda") if with_nvt else None
    lut, counter, nvr = grouped._build_g2l_lut(mask, n, mask.device, nvt, topk)
    exact(lut, torch.arange(n, dtype=torch.int32, device="cpu"))
    exact(counter, torch.zeros(n, dtype=torch.int32, device="cpu"))
    if with_nvt and topk is not None:
        exact(nvr, torch.tensor([37 * topk], dtype=torch.int32, device="cpu"))
    else:
        assert nvr is None


@pytest.mark.parametrize(
    "n,E,expected",
    [
        (511, 128, "flydsl"),
        (512, 128, "flydsl"),
        (513, 128, "triton"),
        (1, 513, "triton"),
        (1024, 256, "triton"),
        (1025, 256, "triton"),
        (8192, 1024, "triton"),
        (8193, 8192, "triton"),
        (16384, 8192, "triton"),
        (1, 8192, "triton"),
        (16385, 8192, "torch"),
        (1, 16385, "torch"),
    ],
)
def test_dispatch_boundaries(n, E, expected, monkeypatch):
    calls = []
    orig_flydsl = grouped._get_compiled_g2l_lut
    orig_triton = wrapper.build_g2l_lut

    def flydsl(*args, **kwargs):
        calls.append("flydsl")
        return orig_flydsl(*args, **kwargs)

    def triton(*args, **kwargs):
        calls.append("triton")
        return orig_triton(*args, **kwargs)

    monkeypatch.setattr(grouped, "_get_compiled_g2l_lut", flydsl)
    monkeypatch.setattr(wrapper, "build_g2l_lut", triton)
    host = torch.zeros(n, dtype=torch.int32, device="cpu")
    host[: min(n, E)] = 1
    mask = host.cuda()
    nvt = torch.tensor([37], dtype=torch.int32, device="cuda")
    lut, counter, nvr = grouped._build_g2l_lut(mask, E, mask.device, nvt, 8)
    assert calls == ([] if expected == "torch" else [expected])
    exact(lut, reference(host, E))
    if expected == "torch":
        assert counter is None and nvr is None
    else:
        exact(counter, torch.zeros(E, dtype=torch.int32, device="cpu"))
        exact(nvr, torch.tensor([296], dtype=torch.int32, device="cpu"))


@pytest.mark.parametrize("n", [512, 513, 1024, 8193])
@pytest.mark.parametrize("force_torch", [False, True])
@pytest.mark.parametrize("enable_triton", [False, True])
def test_env_selection(n, force_torch, enable_triton, monkeypatch):
    monkeypatch.setenv("AITER_G2L_TORCH", "1" if force_torch else "0")
    monkeypatch.setenv("AITER_G2L_TRITON", "1" if enable_triton else "0")
    calls = []
    orig_flydsl = grouped._get_compiled_g2l_lut
    orig_triton = wrapper.build_g2l_lut

    def flydsl(*args, **kwargs):
        calls.append("flydsl")
        return orig_flydsl(*args, **kwargs)

    def triton(*args, **kwargs):
        calls.append("triton")
        return orig_triton(*args, **kwargs)

    monkeypatch.setattr(grouped, "_get_compiled_g2l_lut", flydsl)
    monkeypatch.setattr(wrapper, "build_g2l_lut", triton)
    mask = torch.ones(n, dtype=torch.int32, device="cuda")
    lut, counter, nvr = grouped._build_g2l_lut(mask, n, mask.device)
    expected = []
    if not force_torch:
        if n <= 512:
            expected = ["flydsl"]
        elif enable_triton:
            expected = ["triton"]
    assert calls == expected
    assert (counter is not None) == bool(expected)
    assert nvr is None
    exact(lut, torch.arange(n, dtype=torch.int32, device="cpu"))


@pytest.mark.parametrize("failure", ["compile", "import"])
def test_triton_failure_falls_back(failure, monkeypatch):
    import builtins

    calls = []
    if failure == "compile":

        def fail(*args, **kwargs):
            calls.append("compile")
            raise RuntimeError("injected Triton build failure")

        monkeypatch.setattr(wrapper, "build_g2l_lut", fail)
    else:
        original_import = builtins.__import__

        def fail_import(name, *args, **kwargs):
            if name == "aiter.ops.triton.moe.g2l_lut":
                calls.append("import")
                raise ImportError("injected unavailable Triton")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fail_import)

    mask = torch.ones(8193, dtype=torch.int32, device="cuda")
    nvt = torch.tensor([37], dtype=torch.int32, device="cuda")
    lut, counter, nvr = grouped._build_g2l_lut(mask, 8193, mask.device, nvt, 8)
    assert calls == [failure]
    assert counter is None and nvr is None
    exact(lut, torch.arange(8193, dtype=torch.int32, device="cpu"))
    ep_nvr = nvr if nvr is not None else (nvt * 8).contiguous()
    exact(ep_nvr, torch.tensor([296], dtype=torch.int32, device="cpu"))


@pytest.mark.parametrize("direct_mask", ["0", "1"])
def test_dispatch_reset_ownership(direct_mask, monkeypatch, no_fallback):
    monkeypatch.setattr(grouped, "_flydsl_dispatch_context", lambda: object())
    monkeypatch.setenv("AITER_TDM_DIRECT_EP_MASK", direct_mask)
    mask = torch.ones(8193, dtype=torch.int32, device="cuda")
    counter = torch.full((8193,), 777, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(grouped, "route_counter_buffer", lambda E, device: counter)
    lut, got_counter, _ = grouped._build_g2l_lut(mask, 8193, mask.device)
    assert (got_counter is counter) == (direct_mask == "1")
    exact(
        got_counter,
        torch.full(
            (8193,),
            777 if direct_mask == "1" else 0,
            dtype=torch.int32,
            device="cpu",
        ),
    )
    exact(lut, torch.arange(8193, dtype=torch.int32, device="cpu"))


@pytest.mark.parametrize("n,E", [(513, 128), (8192, 1024), (8193, 8192)])
def test_integrated_single_kernel(n, E, no_fallback):
    mask = torch.zeros(n, device="cuda", dtype=torch.int32)
    mask[:E] = 1
    nvt = torch.tensor([4096], device="cuda", dtype=torch.int32)
    for _ in range(3):
        grouped._build_g2l_lut(mask, E, mask.device, nvt, 8)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        lut, counter, built = grouped._build_g2l_lut(mask, E, mask.device, nvt, 8)
        nvr = built if built is not None else (nvt * 8).contiguous()
        torch.cuda.synchronize()
    assert nvr is built and built is not None
    assert "aten::mul" not in {event.key for event in prof.key_averages()}
    gpu = [
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    if not gpu:
        pytest.skip("GPU profiler activity collection unavailable")
    assert len(gpu) == 1 and "_g2l_lut_kernel" in gpu[0], gpu
    exact(lut, reference(mask.cpu(), E))
    exact(counter, torch.zeros(E, dtype=torch.int32, device="cpu"))
    exact(nvr, torch.tensor([32768], dtype=torch.int32, device="cpu"))


@pytest.mark.parametrize("n", range(513, 1025))
def test_former_flydsl_extension_uses_triton(n, no_fallback):
    host = (torch.arange(n, device="cpu") % 3 == 0).to(torch.int32)
    host[-1] = 0  # 512 experts + sentinel at the first newly routed size.
    E = int(host.sum())
    mask = host.cuda()
    nvt = torch.tensor([37], dtype=torch.int32, device="cuda")
    lut, counter, nvr = grouped._build_g2l_lut(mask, E, mask.device, nvt, 8)
    assert counter is not None and nvr is not None
    exact(lut, reference(host, E))
    exact(counter, torch.zeros(E, dtype=torch.int32, device="cpu"))
    exact(nvr, torch.tensor([296], dtype=torch.int32, device="cpu"))


@pytest.mark.parametrize("n", [513, 8192, 8193])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.bool, torch.float32])
@pytest.mark.parametrize("pattern", ["zero", "one", "sentinel", "last", "random"])
@pytest.mark.parametrize("stride", [1, 2])
def test_integrated_nonzero_semantics(n, dtype, pattern, stride, no_fallback):
    # Compare before cast, including fractional floats, NaN, and wide int64.
    host, mask, E = mask_data(n, dtype, pattern, stride)
    nvt = torch.tensor([4096], dtype=torch.int32, device="cuda")
    lut, counter, nvr = grouped._build_g2l_lut(mask, E, mask.device, nvt, 8)
    assert counter is not None and nvr is not None
    exact(lut, reference(host, E))
    exact(counter, torch.zeros(E, dtype=torch.int32, device="cpu"))
    exact(nvr, torch.tensor([32768], dtype=torch.int32, device="cpu"))


def test_standard_harness_preserves_int32_precision():
    from op_tests.test_moe_g2l_lut import check_outputs

    ref = (
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([0], dtype=torch.int32, device="cuda"),
        torch.tensor([2**31 - 8], dtype=torch.int32, device="cuda"),
    )
    bad = tuple(x.clone() for x in ref)
    bad[2].sub_(1)
    # A float-only comparison would falsely pass here.
    assert torch.equal(ref[2].float(), bad[2].float())
    with pytest.raises(AssertionError, match="nvr int32 mismatch"):
        check_outputs(ref, bad, "injected one-bit error")


@pytest.mark.parametrize(
    "n,E,backend",
    [
        (512, 128, "flydsl"),
        (513, 128, "triton"),
        (8193, 8192, "triton"),
        (16385, 1024, "torch"),
    ],
)
def test_standard_harness_candidates(n, E, backend, monkeypatch):
    import os

    from op_tests import test_moe_g2l_lut as bench

    # Exercise real candidates/output validation once. The CLI owns the actual
    # timing sweep; pytest should not profile every API regression test.
    calls = []

    def single_run(fn, **kwargs):
        calls.append((os.environ["AITER_G2L_TORCH"], kwargs))
        return fn(), 1.0

    monkeypatch.setattr(bench, "run_perftest", single_run)
    monkeypatch.setenv("AITER_G2L_TORCH", "caller-setting")
    monkeypatch.setenv("AITER_G2L_TRITON", "0")
    row = bench.test_moe_g2l_lut(n, E, torch.int32, 2, 37, 8, "graph")
    assert row["auto_backend"] == backend
    assert [flag for flag, _ in calls] == ["0", "1"]
    assert all(kwargs["testGraph"] for _, kwargs in calls)
    for name in ("auto", "torch_fallback"):
        assert row[f"{name} err"] == row[f"{name} TFLOPS"] == 0
        assert row[f"{name} us"] > 0 and row[f"{name} TB/s"] > 0
    assert os.environ["AITER_G2L_TORCH"] == "caller-setting"
    assert os.environ["AITER_G2L_TRITON"] == "0"


def test_standard_harness_rejects_silent_fallback(monkeypatch):
    from op_tests import test_moe_g2l_lut as bench

    def fail(*args, **kwargs):
        raise RuntimeError("injected Triton build failure")

    monkeypatch.setattr(wrapper, "build_g2l_lut", fail)
    with pytest.raises(AssertionError, match="unexpected G2L nvr fallback"):
        bench.test_moe_g2l_lut(8193, 128, torch.int32, 1, 37, 8, "graph")


# The original FlyDSL kernel is unchanged. Keep its small-input dispatch,
# counter and graph regressions here rather than in a separate test file.
def _check_flydsl_fused(host, *, clear_counter=True, stride=1, monkeypatch):
    E = int((host != 0).sum())
    backing = torch.empty(host.numel() * stride, dtype=host.dtype, device="cuda")
    mask = backing[::stride]
    mask.copy_(host)
    storage = torch.full((E + 2,), 777, dtype=torch.int32, device="cuda")
    counter = storage[1:-1]
    monkeypatch.setattr(grouped, "route_counter_buffer", lambda E, device: counter)
    monkeypatch.setattr(
        grouped, "_flydsl_dispatch_context", lambda: None if clear_counter else object()
    )
    nvt = torch.tensor([37], dtype=torch.int32, device="cuda")
    lut, got_counter, built_nvr = grouped._build_g2l_lut(
        mask, E, mask.device, nvt=nvt, topk=8
    )
    # Match the grouped-MoE caller: the fused nvr must bypass the torch mul.
    ep_nvr = built_nvr if built_nvr is not None else (nvt * 8).contiguous()
    assert built_nvr is not None and ep_nvr is built_nvr
    assert got_counter is counter
    exact(lut, reference(host, E))
    exact(
        counter,
        torch.full((E,), 0 if clear_counter else 777, dtype=torch.int32, device="cpu"),
    )
    exact(ep_nvr, torch.tensor([296], dtype=torch.int32, device="cpu"))
    assert storage[0].item() == storage[-1].item() == 777


class TestOriginalFlyDSL:
    """Small-input compatibility; Triton must not hide a FlyDSL regression."""

    @pytest.fixture(autouse=True)
    def isolated_context(self, monkeypatch, context):
        monkeypatch.delenv("AITER_G2L_TORCH", raising=False)
        monkeypatch.setenv("AITER_G2L_TRITON", "0")
        monkeypatch.setenv("AITER_TDM_DIRECT_EP_MASK", "1")
        monkeypatch.setattr(grouped, "_flydsl_dispatch_context", lambda: None)
        monkeypatch.setattr(grouped, "_G2L_COUNTER_CACHE", {})

    @pytest.fixture
    def no_fallback(self, monkeypatch):
        def fail(*args, **kwargs):
            pytest.fail("unexpected torch fallback: FlyDSL must compile and run")

        monkeypatch.setattr(torch, "cumsum", fail)

    @pytest.mark.parametrize(
        "n",
        [0, 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512],
    )
    @pytest.mark.parametrize(
        "pattern", ["zero", "one", "last", "alternating", "random"]
    )
    @pytest.mark.parametrize("clear_counter", [False, True])
    def test_scan_patterns(self, n, pattern, clear_counter, monkeypatch, no_fallback):
        host = torch.zeros(n, dtype=torch.int32, device="cpu")
        if pattern == "one":
            host.fill_(1)
        elif pattern == "last" and n:
            host[-1] = 1
        elif pattern == "alternating":
            host[::2] = -3
        elif pattern == "random":
            host = torch.randint(
                -3,
                4,
                (n,),
                generator=torch.Generator(device="cpu").manual_seed(n),
                dtype=torch.int32,
                device="cpu",
            )
        _check_flydsl_fused(host, clear_counter=clear_counter, monkeypatch=monkeypatch)

    @pytest.mark.parametrize("n", [257, 512])
    @pytest.mark.parametrize(
        "dtype", [torch.bool, torch.int32, torch.int64, torch.float32]
    )
    @pytest.mark.parametrize("stride", [1, 2])
    def test_existing_input_preparation(
        self, n, dtype, stride, monkeypatch, no_fallback
    ):
        host = (torch.arange(n, device="cpu") % 3 == 0).to(dtype)
        _check_flydsl_fused(host, stride=stride, monkeypatch=monkeypatch)

    @pytest.mark.parametrize(
        "with_nvt,topk", [(False, None), (False, 8), (True, None), (True, 0), (True, 8)]
    )
    def test_optional_nvr(self, with_nvt, topk, no_fallback):
        mask = torch.ones(512, device="cuda", dtype=torch.int32)
        nvt = (
            torch.tensor([37, 99], device="cuda", dtype=torch.int64)
            if with_nvt
            else None
        )
        lut, counter, nvr = grouped._build_g2l_lut(mask, 512, mask.device, nvt, topk)
        exact(lut, torch.arange(512, dtype=torch.int32, device="cpu"))
        exact(counter, torch.zeros(512, dtype=torch.int32, device="cpu"))
        if with_nvt and topk is not None:
            exact(nvr, torch.tensor([37 * topk], dtype=torch.int32, device="cpu"))
        else:
            assert nvr is None

    @pytest.mark.parametrize("n,E", [(0, 512), (1, 512), (256, 512)])
    def test_counter_coverage(self, n, E, monkeypatch, no_fallback):
        # The original block must cover all E counter entries, even when E > N.
        mask = torch.ones(n, device="cuda", dtype=torch.int32)
        storage = torch.full((E + 2,), 777, device="cuda", dtype=torch.int32)
        counter = storage[1:-1]
        monkeypatch.setattr(grouped, "route_counter_buffer", lambda E, device: counter)
        lut, got_counter, _ = grouped._build_g2l_lut(mask, E, mask.device)
        assert got_counter is counter
        exact(lut, torch.arange(n, dtype=torch.int32, device="cpu"))
        exact(counter, torch.zeros(E, dtype=torch.int32, device="cpu"))
        assert storage[0].item() == storage[-1].item() == 777

    @pytest.mark.parametrize("n", [257, 512])
    def test_dynamic_graph(self, n, no_fallback):
        E, topk = 73, 8
        mask = torch.zeros(n, device="cuda", dtype=torch.int32)
        nvt = torch.zeros(1, device="cuda", dtype=torch.int32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                grouped._build_g2l_lut(mask, E, mask.device, nvt, topk)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            lut, counter, built_nvr = grouped._build_g2l_lut(
                mask, E, mask.device, nvt, topk
            )
            ep_nvr = built_nvr if built_nvr is not None else (nvt * topk).contiguous()
        assert built_nvr is not None and ep_nvr is built_nvr
        generator = torch.Generator(device="cpu").manual_seed(5906)
        for value in [0, 1, 97, (2**31 - 1) // topk]:
            host = torch.zeros(n, dtype=torch.int32, device="cpu")
            host[torch.randperm(n, generator=generator, device="cpu")[:E]] = -3
            mask.copy_(host)
            nvt.fill_(value)
            for tensor in (lut, counter, ep_nvr):
                tensor.fill_(777)
            graph.replay()
            exact(lut, reference(host, E))
            exact(counter, torch.zeros(E, dtype=torch.int32, device="cpu"))
            exact(ep_nvr, torch.tensor([value * topk], dtype=torch.int32, device="cpu"))

    @pytest.mark.parametrize(
        "reason,n,E",
        [
            ("size", 16385, 256),
            ("size", 1, 16385),
            ("env", 512, 128),
            ("compile", 512, 128),
        ],
    )
    def test_torch_fallback(self, reason, n, E, monkeypatch):
        calls = []

        def unavailable(*args, **kwargs):
            calls.append(True)
            raise RuntimeError("injected FlyDSL failure")

        monkeypatch.setattr(grouped, "_get_compiled_g2l_lut", unavailable)
        if reason == "env":
            monkeypatch.setenv("AITER_G2L_TORCH", "1")
        host = torch.zeros(n, dtype=torch.int32, device="cpu")
        host[: min(E, n)] = 1
        mask = host.cuda()
        nvt = torch.tensor([17], dtype=torch.int32, device="cuda")
        lut, counter, built_nvr = grouped._build_g2l_lut(mask, E, mask.device, nvt, 8)
        assert counter is None and built_nvr is None
        assert len(calls) == (1 if reason == "compile" else 0)
        exact(lut, reference(host, E))
        ep_nvr = built_nvr if built_nvr is not None else (nvt * 8).contiguous()
        exact(ep_nvr, torch.tensor([136], dtype=torch.int32, device="cpu"))

    def test_matching_limits(self):
        from aiter.ops.flydsl.kernels.moe_g2l_lut import MAX_G2L_EXPERTS

        assert grouped._G2L_MAX_N == MAX_G2L_EXPERTS == 512

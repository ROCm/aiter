# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Large-expert G2L integration: FlyDSL <=512, Triton <=16384, then torch.

Run: python -m pytest -q op_tests/triton_tests/moe/test_g2l_lut_large.py
"""

import pytest
import torch

from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
from aiter.ops.triton.moe import g2l_lut as wrapper

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

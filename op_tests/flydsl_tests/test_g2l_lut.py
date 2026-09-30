# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unmodified FlyDSL G2L scan for masks/counters fitting its 512-entry limit.

Standard correctness/perf sweep: op_tests/test_moe_g2l_lut.py
This file retains focused pytest regression checks.

Run: python -m pytest -q op_tests/flydsl_tests/test_g2l_lut.py
Inputs retain the original int32-mask contract; dtype preparation is checked
only for integral values, not fractional masks or out-of-range int64 values.
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from op_tests.test_moe_g2l_lut import SUPPORTED_GFX

pytest.importorskip("flydsl")

from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
from aiter.ops.flydsl.kernels.moe_g2l_lut import MAX_G2L_EXPERTS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def reference(mask, E):
    prefix = 0
    values = []
    for value in mask.tolist():
        if value != 0:
            values.append(prefix)
            prefix += 1
        else:
            values.append(E)
    return torch.tensor(values, dtype=torch.int32, device="cpu")


def assert_exact(actual, expected):
    assert actual.dtype == torch.int32
    assert actual.is_contiguous()
    assert torch.equal(actual.cpu(), expected)


@pytest.fixture(autouse=True)
def supported_arch():
    if get_gfx() not in SUPPORTED_GFX:
        pytest.skip(f"G2L auxiliary kernels are unsupported on {get_gfx()}")


@pytest.fixture(autouse=True)
def isolated_context(monkeypatch):
    monkeypatch.delenv("AITER_G2L_TORCH", raising=False)
    monkeypatch.setenv("AITER_G2L_TRITON", "0")
    monkeypatch.setenv("AITER_TDM_DIRECT_EP_MASK", "1")
    monkeypatch.setattr(grouped, "_flydsl_dispatch_context", lambda: None)
    monkeypatch.setattr(grouped, "_G2L_COUNTER_CACHE", {})


@pytest.fixture
def no_fallback(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("unexpected torch fallback: FlyDSL must compile and run")

    monkeypatch.setattr(torch, "cumsum", fail)


def check_fused(host, *, clear_counter=True, stride=1, monkeypatch):
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
    assert_exact(lut, reference(host, E))
    assert_exact(
        counter,
        torch.full((E,), 0 if clear_counter else 777, dtype=torch.int32, device="cpu"),
    )
    assert_exact(ep_nvr, torch.tensor([296], dtype=torch.int32, device="cpu"))
    assert storage[0].item() == storage[-1].item() == 777


@pytest.mark.parametrize(
    "n",
    [0, 1, 31, 32, 33, 127, 128, 129, 255, 256, 257, 511, 512],
)
@pytest.mark.parametrize("pattern", ["zero", "one", "last", "alternating", "random"])
@pytest.mark.parametrize("clear_counter", [False, True])
def test_scan_patterns(n, pattern, clear_counter, monkeypatch, no_fallback):
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
    check_fused(host, clear_counter=clear_counter, monkeypatch=monkeypatch)


@pytest.mark.parametrize("n", [257, 512])
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("stride", [1, 2])
def test_existing_input_preparation(n, dtype, stride, monkeypatch, no_fallback):
    host = (torch.arange(n, device="cpu") % 3 == 0).to(dtype)
    check_fused(host, stride=stride, monkeypatch=monkeypatch)


@pytest.mark.parametrize(
    "with_nvt,topk", [(False, None), (False, 8), (True, None), (True, 0), (True, 8)]
)
def test_optional_nvr(with_nvt, topk, no_fallback):
    mask = torch.ones(512, device="cuda", dtype=torch.int32)
    nvt = torch.tensor([37, 99], device="cuda", dtype=torch.int64) if with_nvt else None
    lut, counter, nvr = grouped._build_g2l_lut(mask, 512, mask.device, nvt, topk)
    assert_exact(lut, torch.arange(512, dtype=torch.int32, device="cpu"))
    assert_exact(counter, torch.zeros(512, dtype=torch.int32, device="cpu"))
    if with_nvt and topk is not None:
        assert_exact(nvr, torch.tensor([37 * topk], dtype=torch.int32, device="cpu"))
    else:
        assert nvr is None


@pytest.mark.parametrize("n,E", [(0, 512), (1, 512), (256, 512)])
def test_counter_coverage(n, E, monkeypatch, no_fallback):
    # The original block must cover all E counter entries, even when E > N.
    mask = torch.ones(n, device="cuda", dtype=torch.int32)
    storage = torch.full((E + 2,), 777, device="cuda", dtype=torch.int32)
    counter = storage[1:-1]
    monkeypatch.setattr(grouped, "route_counter_buffer", lambda E, device: counter)
    lut, got_counter, _ = grouped._build_g2l_lut(mask, E, mask.device)
    assert got_counter is counter
    assert_exact(lut, torch.arange(n, dtype=torch.int32, device="cpu"))
    assert_exact(counter, torch.zeros(E, dtype=torch.int32, device="cpu"))
    assert storage[0].item() == storage[-1].item() == 777


@pytest.mark.parametrize("n", [257, 512])
def test_dynamic_graph(n, no_fallback):
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
        assert_exact(lut, reference(host, E))
        assert_exact(counter, torch.zeros(E, dtype=torch.int32, device="cpu"))
        assert_exact(
            ep_nvr, torch.tensor([value * topk], dtype=torch.int32, device="cpu")
        )


@pytest.mark.parametrize(
    "reason,n,E",
    [
        ("size", 16385, 256),
        ("size", 1, 16385),
        ("env", 512, 128),
        ("compile", 512, 128),
    ],
)
def test_torch_fallback(reason, n, E, monkeypatch):
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
    assert_exact(lut, reference(host, E))
    ep_nvr = built_nvr if built_nvr is not None else (nvt * 8).contiguous()
    assert_exact(ep_nvr, torch.tensor([136], dtype=torch.int32, device="cpu"))


def test_matching_limits():
    assert grouped._G2L_MAX_N == MAX_G2L_EXPERTS == 512

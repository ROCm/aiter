# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU exactness and grouped-MoE integration tests for G2L + nvt*topk fusion.

Run: pytest -q op_tests/triton_tests/moe/test_g2l_lut.py
Bench: python -m op_tests.triton_tests.moe.test_g2l_lut --bench
"""

import os
import statistics

import pytest
import torch

from aiter.ops.flydsl import grouped_moe_gfx1250 as grouped
from aiter.ops.triton.moe.g2l_lut import MAX_G2L_EXPERTS, build_g2l_lut

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def reference(host, E):
    prefix = 0
    values = []
    for value in host.reshape(-1).tolist():
        if value != 0:
            prefix += 1
            values.append(prefix - 1)
        else:
            values.append(E)
    return torch.tensor(values, dtype=torch.int32)


def assert_exact(tensor, expected):
    assert tensor.dtype == torch.int32
    assert tensor.is_contiguous()
    assert torch.equal(tensor.cpu(), expected)


def grouped_call(mask, E, nvt, topk):
    """The same two statements as the grouped-MoE call site."""
    lut, counter, built_nvr = grouped._build_g2l_lut(
        mask, E, mask.device, nvt=nvt, topk=int(topk)
    )
    ep_nvr = built_nvr if built_nvr is not None else (nvt * int(topk)).contiguous()
    return lut, counter, ep_nvr, built_nvr


def require_triton(monkeypatch):
    monkeypatch.delenv("AITER_G2L_TORCH", raising=False)
    monkeypatch.delenv("AITER_G2L_TRITON", raising=False)  # test the DEFAULT
    monkeypatch.setattr(grouped, "_flydsl_dispatch_context", lambda: None)

    def fail(*args, **kwargs):
        raise AssertionError("unexpected FlyDSL/torch fallback")

    monkeypatch.setattr(grouped, "_get_compiled_g2l_lut", fail)
    monkeypatch.setattr(torch, "cumsum", fail)


@pytest.mark.parametrize(
    "n",
    [
        0,
        1,
        7,
        31,
        32,
        33,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        257,
        511,
        512,
        513,
        1024,
        4096,
        16384,
    ],
)
@pytest.mark.parametrize("dtype", [torch.bool, torch.int32, torch.int64, torch.float32])
@pytest.mark.parametrize("pattern", ["zero", "one", "last", "alternating", "random"])
@pytest.mark.parametrize("clear_counter", [False, True])
def test_exact(n, dtype, pattern, clear_counter):
    generator = torch.Generator().manual_seed(n)
    host = torch.zeros(n, dtype=dtype)
    if pattern == "one":
        host.fill_(1)
    elif pattern == "last" and n:
        host[-1] = 1
    elif pattern == "alternating":
        host[::2] = 1
    elif pattern == "random":
        host = torch.randint(-3, 4, (n,), generator=generator).to(dtype)
        if n >= 3 and dtype == torch.int64:
            host[0], host[1] = 2**40, -(2**40)
        if n >= 3 and dtype == torch.float32:
            host[0], host[1], host[2] = 0.5, -0.25, float("nan")
    E = int((host != 0).sum())
    stride = 2 if pattern in ("alternating", "random") else 1
    backing = torch.zeros(n * stride, dtype=dtype, device="cuda")
    mask = backing[::stride]
    mask.copy_(host)
    # Sentinel canaries + dirty data ensure the reset really writes every slot.
    storage = torch.full((E + 2,), 1777, dtype=torch.int32, device="cuda")
    counter = storage[1:-1]
    nvt_value = [0, 1, 97, (2**31 - 1) // 8][n % 4]
    nvt = torch.tensor([nvt_value], dtype=torch.int32, device="cuda")
    lut, got_counter, nvr = build_g2l_lut(
        mask, E, nvt, 8, counter=counter, clear_counter=clear_counter
    )
    assert_exact(lut, reference(host, E))
    assert got_counter is counter
    assert_exact(
        counter, torch.full((E,), 0 if clear_counter else 1777, dtype=torch.int32)
    )
    assert_exact(nvr, torch.tensor([nvt_value * 8], dtype=torch.int32))
    assert storage[0].item() == storage[-1].item() == 1777


@pytest.mark.parametrize(
    "with_nvt,topk", [(False, None), (False, 8), (True, None), (True, 0), (True, 8)]
)
def test_optional_nvr(with_nvt, topk):
    mask = torch.tensor([0, -1, 0, 2], device="cuda")
    nvt = (
        torch.tensor([37, 123], device="cuda", dtype=torch.int64) if with_nvt else None
    )
    lut, counter, nvr = build_g2l_lut(mask, 2, nvt, topk)
    assert_exact(lut, torch.tensor([2, 0, 2, 1], dtype=torch.int32))
    assert_exact(counter, torch.zeros(2, dtype=torch.int32))
    if with_nvt and topk is not None:
        assert_exact(nvr, torch.tensor([37 * topk], dtype=torch.int32))
    else:
        assert nvr is None


@pytest.mark.parametrize("n", [7, 512, 513, 16384])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64, torch.float32])
def test_grouped_default_reuses_nvr(monkeypatch, n, dtype):
    require_triton(monkeypatch)
    host = (torch.arange(n) % 3 == 0).to(dtype)
    E = int((host != 0).sum())
    mask = host.to("cuda")
    nvt = torch.tensor([37], device="cuda", dtype=torch.int64)

    # Generic first call must not allocate a torch.zeros counter.
    def fail_counter(*args):
        raise AssertionError("generic path should let fused kernel reset empty counter")

    monkeypatch.setattr(grouped, "route_counter_buffer", fail_counter)
    lut, counter, ep_nvr, built_nvr = grouped_call(mask, E, nvt, 8)
    assert ep_nvr is built_nvr
    assert_exact(lut, reference(host, E))
    assert_exact(counter, torch.zeros(E, dtype=torch.int32))
    assert_exact(ep_nvr, torch.tensor([296], dtype=torch.int32))


@pytest.mark.parametrize("direct_mask", ["0", "1"])
def test_dispatch_owned_counter(monkeypatch, direct_mask):
    require_triton(monkeypatch)
    monkeypatch.setattr(grouped, "_flydsl_dispatch_context", lambda: object())
    monkeypatch.setenv("AITER_TDM_DIRECT_EP_MASK", direct_mask)
    counter = torch.full((2,), 41, device="cuda", dtype=torch.int32)
    monkeypatch.setattr(grouped, "route_counter_buffer", lambda E, device: counter)
    mask = torch.tensor([0, 1, 0, 1], device="cuda", dtype=torch.int32)
    nvt = torch.tensor([5], device="cuda", dtype=torch.int32)
    lut, out_counter, nvr, built_nvr = grouped_call(mask, 2, nvt, 8)
    assert nvr is built_nvr
    if direct_mask == "1":
        assert out_counter is counter
        assert_exact(out_counter, torch.full((2,), 41, dtype=torch.int32))
    else:
        assert out_counter is not counter
        assert_exact(out_counter, torch.zeros(2, dtype=torch.int32))
    assert_exact(lut, torch.tensor([2, 0, 2, 1], dtype=torch.int32))
    assert_exact(nvr, torch.tensor([40], dtype=torch.int32))


@pytest.mark.parametrize("force", ["env", "size", "error"])
def test_torch_fallback_mul(monkeypatch, force):
    monkeypatch.setattr(grouped, "_flydsl_dispatch_context", lambda: None)
    monkeypatch.setenv("AITER_G2L_TRITON", "1")
    monkeypatch.setenv("AITER_G2L_TORCH", "1" if force == "env" else "0")
    n = MAX_G2L_EXPERTS + 1 if force == "size" else 513
    if force == "error":
        import aiter.ops.triton.moe.g2l_lut as module

        def fail(*args, **kwargs):
            raise RuntimeError("injected Triton failure")

        monkeypatch.setattr(module, "build_g2l_lut", fail)
    host = (torch.arange(n) % 2).to(torch.int32)
    E = int(host.sum())
    nvt = torch.tensor([17], dtype=torch.int32, device="cuda")
    lut, counter, nvr, built_nvr = grouped_call(host.to("cuda"), E, nvt, 8)
    assert built_nvr is None
    assert counter is None
    assert_exact(lut, reference(host, E))
    assert_exact(nvr, torch.tensor([136], dtype=torch.int32))


def test_disable_triton_restores_flydsl_selection(monkeypatch):
    import aiter.ops.triton.moe.g2l_lut as module

    monkeypatch.setenv("AITER_G2L_TRITON", "0")
    monkeypatch.setenv("AITER_G2L_TORCH", "0")
    monkeypatch.setattr(grouped, "_flydsl_dispatch_context", lambda: None)
    calls = []

    def fail_triton(*args, **kwargs):
        raise AssertionError("Triton disabled")

    def fail_flydsl(*args, **kwargs):
        calls.append("flydsl")
        raise RuntimeError("force legacy torch fallback")

    monkeypatch.setattr(module, "build_g2l_lut", fail_triton)
    monkeypatch.setattr(grouped, "_get_compiled_g2l_lut", fail_flydsl)
    mask = torch.tensor([0, 1, 1, 0], device="cuda", dtype=torch.int32)
    nvt = torch.tensor([17], device="cuda", dtype=torch.int32)
    lut, counter, nvr, built_nvr = grouped_call(mask, 2, nvt, 8)
    assert calls == ["flydsl"]
    assert counter is None and built_nvr is None
    assert_exact(lut, reference(mask.cpu(), 2))
    assert_exact(nvr, torch.tensor([136], dtype=torch.int32))


def test_grouped_dynamic_graph(monkeypatch):
    require_triton(monkeypatch)
    n, E, topk = 257, 73, 8
    mask = torch.zeros(n, device="cuda", dtype=torch.int32)
    nvt = torch.zeros(1, device="cuda", dtype=torch.int32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            grouped_call(mask, E, nvt, topk)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        lut, counter, ep_nvr, built_nvr = grouped_call(mask, E, nvt, topk)
    assert ep_nvr is built_nvr
    generator = torch.Generator().manual_seed(97)
    for nvt_value in [0, 1, 97, (2**31 - 1) // topk]:
        host = torch.zeros(n, dtype=torch.int32)
        host[torch.randperm(n, generator=generator)[:E]] = -3
        mask.copy_(host)
        nvt.fill_(nvt_value)
        for tensor in (lut, counter, ep_nvr):
            tensor.fill_(777)
        graph.replay()
        assert_exact(lut, reference(host, E))
        assert_exact(counter, torch.zeros(E, dtype=torch.int32))
        assert_exact(ep_nvr, torch.tensor([nvt_value * topk], dtype=torch.int32))


def test_input_validation():
    mask = torch.ones(4, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError, match="preallocated"):
        build_g2l_lut(mask, 4, clear_counter=False)
    with pytest.raises(ValueError, match="max"):
        build_g2l_lut(mask, MAX_G2L_EXPERTS + 1)
    with pytest.raises(ValueError, match="nvt"):
        build_g2l_lut(mask, 4, torch.ones(1, device="cuda"), 8)


def test_grouped_single_gpu_kernel(monkeypatch):
    require_triton(monkeypatch)
    host = (torch.arange(512) % 4 == 0).to(torch.int32)
    mask = host.to("cuda")
    nvt = torch.tensor([4096], device="cuda", dtype=torch.int32)
    for _ in range(3):
        grouped_call(mask, 128, nvt, 8)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        lut, counter, ep_nvr, built_nvr = grouped_call(mask, 128, nvt, 8)
        torch.cuda.synchronize()
    assert ep_nvr is built_nvr
    assert "aten::mul" not in {event.key for event in prof.key_averages()}
    gpu_events = [
        e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    if not gpu_events:
        pytest.skip("GPU profiler activity collection unavailable")
    assert len(gpu_events) == 1, [e.name for e in gpu_events]
    assert "_g2l_lut_kernel" in gpu_events[0].name
    assert_exact(lut, reference(host, 128))
    assert_exact(counter, torch.zeros(128, dtype=torch.int32))
    assert_exact(ep_nvr, torch.tensor([32768], dtype=torch.int32))


def graph_us(fn, unroll=32, repeats=30):
    # Small graph avoids large-graph instability on some development runtimes.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(10):
            result = fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(unroll):
            result = fn()
    with torch.cuda.stream(stream):
        for _ in range(5):
            graph.replay()
    stream.synchronize()
    events = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in range(repeats)
    ]
    with torch.cuda.stream(stream):
        for start, end in events:
            start.record()
            graph.replay()
            end.record()
    stream.synchronize()
    del result
    return statistics.median(
        start.elapsed_time(end) * 1000 / unroll for start, end in events
    )


def benchmark(sizes):
    """Time the REAL integrated helper + caller, not a duplicate Triton kernel."""
    import triton

    saved = {k: os.environ.get(k) for k in ("AITER_G2L_TORCH", "AITER_G2L_TRITON")}
    ctx = grouped._flydsl_dispatch_context()
    assert ctx is None, "run this standalone (no active dispatch context)"
    print(torch.cuda.get_device_properties(0))
    print(f"torch={torch.__version__} triton={triton.__version__}")
    print("Graph GPU us/call: unroll=32, repeats=30, median, nvt=4096, topk=8")
    print(f"{'N':>6} {'E':>6} {'native us':>12} {'integrated us':>15} {'speedup':>9}")
    try:
        os.environ["AITER_G2L_TRITON"] = "1"
        for n in sizes:
            host = (torch.arange(n) % 4 == 0).to(torch.int32)
            E = int(host.sum())
            mask = host.to("cuda")
            nvt = torch.tensor([4096], device="cuda", dtype=torch.int32)

            def run(mask=mask, E=E, nvt=nvt):
                lut, counter, nvr, _ = grouped_call(mask, E, nvt, 8)
                # torch fallback returns None; include its caller-owned reset
                # so BOTH implementations perform LUT + mul + counter clear.
                if counter is None:
                    counter = torch.zeros(E, device=mask.device, dtype=torch.int32)
                return lut, counter, nvr

            times = {}
            for force_torch in ("1", "0"):
                os.environ["AITER_G2L_TORCH"] = force_torch
                lut, counter, nvr = run()
                assert_exact(lut, reference(host, E))
                assert_exact(counter, torch.zeros(E, dtype=torch.int32))
                assert_exact(nvr, torch.tensor([32768], dtype=torch.int32))
                if force_torch == "0":
                    assert grouped_call(mask, E, nvt, 8)[3] is not None
                times[force_torch] = graph_us(run)
            native, fused = times["1"], times["0"]
            print(
                f"{n:6d} {E:6d} {native:12.3f} {fused:15.3f} {native/fused:8.2f}x",
                flush=True,
            )
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", action="store_true")
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[32, 64, 128, 256, 512, 1024, 4096]
    )
    args = parser.parse_args()
    if args.bench:
        if any(n <= 0 or n > MAX_G2L_EXPERTS for n in args.sizes):
            parser.error(f"sizes must be in [1, {MAX_G2L_EXPERTS}]")
        benchmark(args.sizes)
    else:
        raise SystemExit(pytest.main([__file__]))

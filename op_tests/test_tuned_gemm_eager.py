# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Ordinary tuned GEMM: descriptor lifetime, dispatch, and host timing.

Run the same command on main (including #8116) and head for a controlled A/B.
The host timings below are warm observations, not a controlled performance claim.
Canonical test_gemm_a16w16.py covers the broader numerical/kernel matrix.
"""

import argparse
import contextlib
import gc
import hashlib
import itertools
import time
import weakref

import pandas as pd
import torch
from torch.profiler import ProfilerActivity, profile

import aiter
import aiter.tuned_gemm as tuned
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import tgemm

BF16, FP16, FP32 = dtypes.bf16, dtypes.fp16, dtypes.fp32
CUSTOM_OPS = ("aiter::gemm_a16w16", "aiter::gemm_a16w16_out")
SHAPES = [
    (1056, 324, 10240),
    (1056, 10240, 320),
    (2112, 513, 2560),
    (2112, 48, 2560),
    (2112, 640, 2560),
    (256, 512, 1024),
    (96, 384, 768),
]


def _rand(*shape, dtype=BF16, scale=1.0):
    return torch.randn(*shape, device="cuda").mul_(scale).to(dtype)


@contextlib.contextmanager
def _routed(libtype="hipblaslt"):
    """Use one backend for the controlled shapes, and count ordinary lookups."""
    real = tuned.get_GEMM_A16W16_config
    calls = []

    def lookup(**kwargs):
        calls.append(kwargs)
        return {"libtype": libtype, "solidx": 2 if libtype == "skinny" else -1}

    tuned.get_GEMM_A16W16_config = lookup
    try:
        yield calls
    finally:
        tuned.get_GEMM_A16W16_config = real


def _reference(x, w, bias=None, otype=None):
    result = x.float() @ w.float().t()
    if bias is not None:
        result += bias.float()
    return result.to(otype or x.dtype)


def _check(result, reference):
    assert result.shape == reference.shape and result.dtype == reference.dtype
    assert bool(torch.isfinite(result).all()), "nonfinite GEMM output"
    assert checkAllclose(reference.float(), result.float()) == 0


def _observed(call):
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        result = call()
    return result, sum(e.name in CUSTOM_OPS for e in prof.events())


def _host_us(call, iters=50):
    call()
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            call()
        times.append((time.perf_counter() - start) / iters * 1e6)
    torch.cuda.synchronize()
    return sorted(times)[2]


@benchmark()
def test_gemm(m, n, k, dtype, bias, out_kind, libtype="hipblaslt"):
    torch.manual_seed(42)
    x, w = _rand(m, k, dtype=dtype), _rand(n, k, dtype=dtype, scale=k**-0.5)
    b = _rand(n, dtype=dtype) if bias else None
    out = torch.empty(m, n, dtype=dtype, device="cuda") if out_kind else None

    def call():
        if out_kind == "out_op":
            tuned.gemm_a16w16_out(x, w, out, bias=b)
            return out
        return tgemm.mm(x, w, b, out=out)

    reference = _reference(x, w, b)
    with _routed(libtype) as lookups:
        y, ops = _observed(call)
        assert ops == 1 and len(lookups) == 1, (libtype, ops, len(lookups))
        assert out is None or y.data_ptr() == out.data_ptr()
        _check(y, reference)
        checksum = hashlib.sha256(
            y.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        ).hexdigest()
        result, device_us = run_perftest(call)
        _check(result, reference)
        host_us = _host_us(call)
    return {
        "gfx": get_gfx_runtime(),
        "device us": device_us,
        "warm host us": host_us,
        "output sha256": checksum,
    }


def check_weight_descriptors():
    """Cached descriptors cannot keep weights alive or confuse same-address views."""
    x = _rand(128, 512)
    w = _rand(256, 512, scale=512**-0.5)
    with _routed():
        _check(tgemm.mm(x, w), _reference(x, w))
        w.add_(0.125)
        for a, b in ((x, w), (x, w[:128]), (x[:, :256], w[:, :256])):
            _check(tgemm.mm(a, b), _reference(a, b))
        del a, b
        alive = weakref.ref(w)
        del w
        gc.collect()
        assert alive() is None, "descriptor cache retains the weight"
        w = _rand(256, 512, scale=512**-0.5)
        _check(tgemm.mm(x, w), _reference(x, w))


def check_compile_and_graph():
    x, w = _rand(2, 128, 512), _rand(384, 512, scale=512**-0.5)
    out = torch.empty(2, 128, 384, dtype=FP32, device="cuda")
    targets = []

    def fn(a, b, result):
        return tgemm.mm(a, b, otype=FP32, out=result)

    def backend(gm, example_inputs):
        targets.extend(str(n.target) for n in gm.graph.nodes)
        return gm.forward

    with _routed():
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        _check(compiled(x, w, out), _reference(x, w, otype=FP32))
        assert "aiter.gemm_a16w16_out" in targets, targets
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn(x, w, out)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fn(x, w, out)
        torch.cuda.current_stream().wait_stream(stream)
        for _ in range(2):
            x.copy_(_rand(2, 128, 512))
            graph.replay()
            torch.cuda.synchronize()
            _check(out, _reference(x, w, otype=FP32))
        graph.reset()


def main():
    gfx = get_gfx_runtime()
    if gfx not in ("gfx942", "gfx950"):
        aiter.logger.warning("tuned GEMM host test: %s not covered", gfx)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--mnk", type=dtypes.str2tuple, nargs="*", default=SHAPES)
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[BF16, FP16]
    )
    args = parser.parse_args()
    rows = [
        test_gemm(*shape, dtype, bias, out)
        for shape, dtype, bias, out in itertools.product(
            args.mnk, args.dtype, (False, True), (None, "mm", "out_op")
        )
    ]
    rows += [
        test_gemm(4, 64, 1024, BF16, False, None, libtype)
        for libtype in ("torch", "skinny", "triton")
    ]
    check_weight_descriptors()
    check_compile_and_graph()
    aiter.logger.info(
        "ordinary tuned GEMM (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()

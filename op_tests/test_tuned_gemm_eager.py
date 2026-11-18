# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Eager fast path of `tgemm.mm`, `gemm_a16w16` and `gemm_a16w16_out`.

Outside torch.compile, an unscaled GEMM whose tuned config is hipBLASLt calls
the hipBLASLt entry directly instead of the aiter::gemm_a16w16(_out) and
aiter::_hipb_mm custom ops. It must give the custom ops' bits, and every call
those ops may treat differently must still reach them. Which path ran is read
from the profiler: each custom-op call is a dispatcher event, and the fast path
records none.

Tensors are created with an explicit device: `torch.set_default_device` is a
TorchFunctionMode, under which every call takes the custom ops.
"""

import argparse
import contextlib
import gc
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
import weakref

import pandas as pd
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import TorchFunctionMode
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

import aiter
import aiter.tuned_gemm as tuned
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import tgemm

BF16, FP16, FP32 = dtypes.bf16, dtypes.fp16, dtypes.fp32
SUPPORTED_GFX = ["gfx942", "gfx950"]
CUSTOM_OPS = ("aiter::gemm_a16w16", "aiter::gemm_a16w16_out")
HIPBLASLT = {"libtype": "hipblaslt", "solidx": -1}
OUT_KINDS = [None, "mm", "out_op"]
# The other backends a BF16 GEMM may be tuned to, and the archs they run on.
FALLBACKS = {
    "torch": ({"libtype": "torch", "solidx": 0}, SUPPORTED_GFX),
    "skinny": ({"libtype": "skinny", "solidx": 2}, ["gfx942", "gfx950"]),
    "triton": ({"libtype": "triton", "solidx": 0}, SUPPORTED_GFX),
}


def _rand(*shape, dtype=BF16, scale=1.0):
    return torch.randn(*shape, device="cuda").mul_(scale).to(dtype)


def _key(m, n, k, bias, dtype, otype, bpreshuffle=False):
    return (m, n, k, bias, str(dtype), str(otype), bpreshuffle)


@contextlib.contextmanager
def _routed(rows):
    """Serve the shapes of ``rows`` (`_key` -> config) from ``rows``.

    Installed over `get_GEMM_A16W16_config`, which both paths call through the
    module on every call; other shapes and scaled calls keep the tuned lookup.
    """
    real = tuned.get_GEMM_A16W16_config

    def lookup(M, N, K, bias, dtype, otype, scaleAB=False, bpreshuffle=False, **kw):
        row = rows.get((M, N, K, bias, dtype, otype, bpreshuffle))
        if row is not None and not scaleAB:
            return dict(row)
        return real(
            M=M,
            N=N,
            K=K,
            bias=bias,
            dtype=dtype,
            otype=otype,
            scaleAB=scaleAB,
            bpreshuffle=bpreshuffle,
            **kw,
        )

    tuned.get_GEMM_A16W16_config = lookup
    try:
        yield
    finally:
        tuned.get_GEMM_A16W16_config = real


def _custom_op_calls(fn):
    """``fn()``, and how many custom-op calls the dispatcher saw while it ran."""
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        result = fn()
    return result, sum(e.name in CUSTOM_OPS for e in prof.events())


def _op(x, w, bias=None, otype=None, out=None):
    """The custom-op path, called through the dispatcher."""
    if out is None:
        return torch.ops.aiter.gemm_a16w16(x, w, bias, otype, None, None, None)
    torch.ops.aiter.gemm_a16w16_out(x, w, out, bias, otype, None, None, None)
    return out


def _reference(x, w, bias=None, otype=None):
    """``x @ w.T + bias`` by torch in float32."""
    out = x.float() @ w.float().t()
    if bias is not None:
        out += bias.float()
    return out.to(otype or x.dtype)


def _check_bits(y, op_call, ref, what):
    """``y`` must be the custom op's bits wherever its kernel is reproducible.

    The op is repeated: if its results vary, the kernel itself is not
    reproducible for this input, and ``y`` is held to the float32 reference
    ``ref`` only. Returns whether the bits were compared.
    """
    samples = [op_call().clone() for _ in range(2)]
    if torch.equal(samples[0], samples[1]) and not torch.equal(y, samples[0]):
        # Two agreeing samples do not make a kernel reproducible.
        samples += [op_call().clone() for _ in range(4)]
    exact = all(torch.equal(s, samples[0]) for s in samples[1:])
    if exact:
        differ = int((y != samples[0]).sum())
        assert differ == 0, f"{what}: {differ} elements differ from the custom op"
    err = checkAllclose(ref.float(), y.float(), msg=f"{what}: ")
    assert err == 0, f"{what}: {err:.2%} of elements off the float32 reference"
    return exact


def _host_us(fn, iters=50):
    """Host time of one call in us: the median, over 5 rounds, of ``iters``
    calls queued back to back."""
    fn()
    times = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        times.append((time.perf_counter() - start) / iters * 1e6)
    torch.cuda.synchronize()
    return sorted(times)[2]


@benchmark()
def test_eager(m, n, k, dtype, otype, bias, rank, out):
    """The fast path against the custom op, for a shape routed to hipBLASLt."""
    x = _rand(m, k, dtype=dtype)
    if rank == 3:
        x = x.view(2, m // 2, k)
    w = _rand(n, k, dtype=dtype, scale=k**-0.5)
    b = _rand(n, dtype=dtype) if bias else None
    buf = None
    if out is not None:
        buf = torch.full(
            (*x.shape[:-1], n), float("nan"), dtype=otype or dtype, device="cuda"
        )

    def eager():
        if out is None:
            return tgemm.mm(x, w, b, otype=otype)
        if out == "mm":
            return tgemm.mm(x, w, b, otype=otype, out=buf)
        tuned.gemm_a16w16_out(x, w, buf, bias=b, otype=otype)
        return buf

    candidates = {"op": lambda: _op(x, w, b, otype, buf), "eager": eager}
    ref = _reference(x, w, b, otype)
    flops = 2 * m * n * k
    nbytes = (m * k + n * k) * x.element_size() + m * n * ref.element_size()
    ret = {"gfx": get_gfx_runtime()}
    with _routed({_key(m, n, k, bias, dtype, otype or dtype): HIPBLASLT}):
        y, calls = _custom_op_calls(eager)
        assert calls == 0, f"{m}x{n}x{k}: the call reached the custom op"
        assert y.shape == ref.shape and y.dtype == ref.dtype
        assert buf is None or y.data_ptr() == buf.data_ptr()
        ret["bitwise"] = _check_bits(y.clone(), candidates["op"], ref, "eager")
        for name, fn in candidates.items():
            result, us = run_perftest(fn)
            err = checkAllclose(ref.float(), result.float(), msg=f"{name}: ")
            assert err == 0, f"{name}: {err:.2%} of elements off the reference"
            ret[f"{name} us"] = us
            ret[f"{name} host us"] = _host_us(fn)
            ret[f"{name} TFLOPS"] = flops / us / 1e6
            ret[f"{name} TB/s"] = nbytes / us / 1e6
            ret[f"{name} err"] = err
    return ret


@benchmark()
def test_fallback(m, n, k, libtype):
    """The public `gemm_a16w16` against the custom op wrapper it was before the
    fast path, for a shape routed to another backend: the fast path declines,
    after a config lookup of its own, and the custom op runs."""
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    ref = _reference(x, w)
    candidates = {
        "custom op": lambda: tuned._gemm_a16w16_op(x, w, None, None, None, None, None),
        "gemm_a16w16": lambda: tuned.gemm_a16w16(x, w),
    }
    ret = {"gfx": get_gfx_runtime()}
    with _routed({_key(m, n, k, False, BF16, BF16): FALLBACKS[libtype][0]}):
        y, calls = _custom_op_calls(candidates["gemm_a16w16"])
        assert calls == 1, f"{libtype}: {calls} custom-op calls"
        ret["bitwise"] = _check_bits(y.clone(), lambda: _op(x, w), ref, libtype)
        for name, fn in candidates.items():
            ret[f"{name} host us"] = _host_us(fn)
    ret["extra host us"] = ret["gemm_a16w16 host us"] - ret["custom op host us"]
    return ret


def _row(case, calls, custom_op, bitwise=None):
    """A row of the path table: the call reached the custom op iff ``custom_op``."""
    path = "custom op" if calls else "fast path"
    assert bool(calls) == custom_op, f"{case}: took the {path}"
    return {"case": case, "path": path, "custom-op calls": calls, "bitwise": bitwise}


def _fast_row(case, x, w):
    """A row for a call that must take the fast path and give the op's bits."""
    y, calls = _custom_op_calls(lambda: tgemm.mm(x, w))
    exact = _check_bits(y, lambda: _op(x, w), _reference(x, w), case)
    return _row(case, calls, False, exact)


def check_first_calls():
    """The first hipBLASLt GEMM of a process creates the extension on the
    custom-op path; the calls after it take the fast path."""
    x, w = _rand(64, 256), _rand(128, 256)
    with _routed({_key(64, 128, 256, False, BF16, BF16): HIPBLASLT}):
        calls = [_custom_op_calls(lambda: tgemm.mm(x, w))[1] for _ in range(3)]
    return [_row(f"call {i} of the process", c, i == 1) for i, c in enumerate(calls, 1)]


def _record_shapes(csv):
    """`check_shape_recording`'s child: three calls, shapes recorded to ``csv``."""
    tuned.untune_path = csv
    x, w = _rand(64, 256), _rand(128, 256)
    with _routed({_key(64, 128, 256, False, BF16, BF16): HIPBLASLT}):
        calls = [_custom_op_calls(lambda: tgemm.mm(x, w))[1] for _ in range(3)]
    print(json.dumps(calls))


def check_shape_recording():
    """With AITER_TUNE_GEMM set at start, every call takes the custom op, which
    records the shape for tuning."""
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        csv = os.path.join(tmp, "untuned.csv")
        code = (
            f"import sys; sys.path.insert(0, {here!r}); "
            f"import test_tuned_gemm_eager as t; t._record_shapes({csv!r})"
        )
        child = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "AITER_TUNE_GEMM": "1"},
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        assert child.returncode == 0, child.stderr[-4000:]
        calls = json.loads(child.stdout.strip().splitlines()[-1])
        shapes = pd.read_csv(csv)
    assert ((shapes["M"] == 64) & (shapes["N"] == 128) & (shapes["K"] == 256)).any()
    return [
        _row(f"AITER_TUNE_GEMM=1, call {i}", c, True) for i, c in enumerate(calls, 1)
    ]


def check_declines():
    """Calls the custom op may treat differently reach it; the others do not."""
    m, n, k = 1056, 324, 2560
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    ref = _reference(x, w)
    rows = []
    with _routed({_key(m, n, k, False, BF16, BF16): HIPBLASLT}):
        s = torch.ones(1, device="cuda")
        _, calls = _custom_op_calls(lambda: tgemm.mm(x, w, scale_a=s, scale_b=s))
        rows.append(_row("scale_a and scale_b", calls, True))

        wg = w.clone().requires_grad_(True)
        y, calls = _custom_op_calls(lambda: tgemm.mm(x, wg))
        assert y.requires_grad
        rows.append(_row("weight requires grad", calls, True))
        with torch.no_grad():
            y, calls = _custom_op_calls(lambda: tgemm.mm(x, wg))
        exact = _check_bits(y, lambda: _op(x, w), ref, "no_grad")
        rows.append(_row("weight requires grad, under no_grad", calls, False, exact))

        wp = torch.nn.Parameter(w.clone(), requires_grad=False)
        y, calls = _custom_op_calls(lambda: tgemm.mm(x, wp))
        exact = _check_bits(y, lambda: _op(x, w), ref, "Parameter")
        rows.append(_row("nn.Parameter weight", calls, False, exact))

        seen = []

        class Record(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                seen.append(str(func))
                return func(*args, **(kwargs or {}))

        class Passthrough(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with Record():
            tgemm.mm(x, w)
        calls = sum(name.startswith("aiter.gemm_a16w16.") for name in seen)
        rows.append(_row("TorchDispatchMode, ops it saw", calls, True))
        with Passthrough():
            _, calls = _custom_op_calls(lambda: tgemm.mm(x, w))
        rows.append(_row("TorchFunctionMode", calls, True))
        torch.set_default_device("cuda")
        try:
            _, calls = _custom_op_calls(lambda: tgemm.mm(x, w))
        finally:
            torch.set_default_device(None)
        rows.append(_row("torch.set_default_device", calls, True))

        with FakeTensorMode():
            xf = torch.empty(m, k, dtype=BF16, device="cuda")
            wf = torch.empty(n, k, dtype=BF16, device="cuda")
            y, calls = _custom_op_calls(lambda: tgemm.mm(xf, wf))
        assert isinstance(y, FakeTensor) and tuple(y.shape) == (m, n)
        rows.append(_row("FakeTensorMode", calls, True))
        # Fake tensors outside their mode still have to reach the op's fake.
        y, calls = _custom_op_calls(lambda: tgemm.mm(xf, wf))
        assert isinstance(y, FakeTensor)
        rows.append(_row("fake tensors outside their mode", calls, True))

    with _routed({_key(m, n, k, False, BF16, BF16): {"libtype": "torch", "solidx": 0}}):
        y, calls = _custom_op_calls(lambda: tgemm.mm(x, w))
    exact = _check_bits(y, lambda: F.linear(x, w), ref, "torch config")
    rows.append(_row("config is not hipBLASLt", calls, True, exact))

    xt = _rand(4, 64, k).transpose(0, 1)  # `view(-1, K)` refuses it
    with _routed({_key(256, n, k, False, BF16, BF16): HIPBLASLT}):
        y, calls = _custom_op_calls(lambda: tgemm.mm(xt, w))
    exact = _check_bits(y, lambda: F.linear(xt, w), _reference(xt, w), "3-D input")
    rows.append(_row("3-D input that view cannot flatten", calls, True, exact))
    return rows


def check_replaced_lookup():
    """A function installed over `get_GEMM_A16W16_config` whose answer flips at
    run time is asked on every call, and its answer is never cached."""
    m, n, k = 1056, 324, 2560
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    rows = []
    with _routed({_key(m, n, k, False, BF16, BF16): HIPBLASLT}):
        routed = tuned.get_GEMM_A16W16_config
        state = {"bypass": False, "asked": 0}

        def replacement(**kw):
            state["asked"] += 1
            config = routed(**kw)
            if state["bypass"] and config["libtype"] == "hipblaslt":
                return dict(config, libtype="torch", solidx=0)
            return config

        tuned.get_GEMM_A16W16_config = replacement
        try:
            for bypass in (False, True, False, True):
                state["bypass"], asked = bypass, state["asked"]
                y, calls = _custom_op_calls(lambda: tgemm.mm(x, w))
                # Bypassed: asked by the fast path, then by the op it hands over to.
                assert state["asked"] - asked == (2 if bypass else 1)
                torch_path = (lambda: F.linear(x, w)) if bypass else (lambda: _op(x, w))
                exact = _check_bits(y, torch_path, _reference(x, w), "replaced")
                rows.append(
                    _row(f"replaced lookup, bypass={bypass}", calls, bypass, exact)
                )
        finally:
            tuned.get_GEMM_A16W16_config = routed
    return rows


def check_plugin_route():
    """While a kernel plugin is enabled, every call goes through the route of
    `gemm_a16w16`: a provider that serves it runs, one that declines leaves it
    to the custom op. With none enabled, the fast path takes the call again."""
    from aiter import plugin

    m, n, k = 1056, 324, 2560
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    buf = torch.empty(m, n, dtype=BF16, device="cuda")
    ref = _reference(x, w)

    def run(A, B, bias, otype, scale_a, scale_b, scale_c, out):
        if out is None:
            return torch.full((A.shape[0], B.shape[0]), 7.0, device=A.device)
        return out.fill_(7.0)

    class Provider:
        name = "eager-test"
        api_versions = (plugin.API_VERSION,)
        op_versions = ("gemm_a16w16",)
        serve = True

        def select(self, op, args, kwargs):
            if not self.serve:
                return None
            return plugin.KernelCandidate(
                provider=self.name,
                kernel_id="sevens",
                artifact_digest="sha256:0",
                run=run,
            )

    class EntryPoint:
        name = Provider.name

        def load(self):
            return provider

    provider = Provider()
    entry_points = plugin._entry_points
    plugin._entry_points = lambda: (EntryPoint(),)
    plugin._reset_for_testing()
    rows = []
    try:
        with _routed({_key(m, n, k, False, BF16, BF16): HIPBLASLT}):
            plugin.configure(enable=(Provider.name,), source="test_tuned_gemm_eager")
            for case, call in (
                ("plugin serves the call", lambda: tgemm.mm(x, w)),
                ("plugin serves the call, out=", lambda: tgemm.mm(x, w, out=buf)),
            ):
                y, calls = _custom_op_calls(call)
                assert bool((y == 7).all()), f"{case}: the provider did not run"
                assert plugin.last_decision().route == "provider"
                rows.append(_row(case, calls, True))
            provider.serve = False
            y, calls = _custom_op_calls(lambda: tgemm.mm(x, w))
            assert plugin.last_decision().route == "native"
            exact = _check_bits(y, lambda: _op(x, w), ref, "plugin declines")
            rows.append(_row("plugin enabled, provider declines", calls, True, exact))
            plugin.configure(enable=(), source="test_tuned_gemm_eager")
            rows.append(_fast_row("plugin configured with nothing enabled", x, w))
    finally:
        plugin._entry_points = entry_points
        plugin._reset_for_testing()
    return rows


def check_weight_handles():
    """The fast path keeps converted weights per tensor descriptor. A weight in
    a freed one's memory, a view with the same pointer and another shape, and a
    strided view compute what the custom op does, and no weight is kept alive."""
    m, n, k = 128, 256, 512
    x = _rand(m, k)
    shapes = ((n, k), (n // 2, k), (n, k // 2))
    with _routed({_key(m, nn, kk, False, BF16, BF16): HIPBLASLT for nn, kk in shapes}):
        w1 = _rand(n, k, scale=k**-0.5)
        rows = [_fast_row("weight", x, w1)]
        ptr, alive = w1.data_ptr(), weakref.ref(w1)
        del w1
        gc.collect()
        assert alive() is None, "the fast path keeps a weight alive"
        w2 = _rand(n, k, scale=k**-0.5)
        reused = "in the freed one's memory" if w2.data_ptr() == ptr else "elsewhere"
        rows.append(_fast_row(f"new weight, {reused}", x, w2))
        rows.append(_fast_row("weight view, same pointer", x, w2[: n // 2]))
        rows.append(_fast_row("strided weight view", x[:, : k // 2], w2[:, : k // 2]))
    return rows


def check_compile():
    """torch.compile records the custom ops and gives the eager bits."""
    m, n, k = 256, 512, 1024
    x, w, b = _rand(m, k), _rand(n, k, scale=k**-0.5), _rand(n)
    out = torch.empty(m, n, dtype=BF16, device="cuda")
    targets = []

    def backend(gm, example_inputs):
        targets.extend(str(node.target) for node in gm.graph.nodes)
        return gm.forward

    def fn(x, w, b, out):
        y = tgemm.mm(x, w, b)
        tgemm.mm(x, w, out=out)
        return y

    torch._dynamo.reset()
    rows = {_key(m, n, k, bias, BF16, BF16): HIPBLASLT for bias in (False, True)}
    with _routed(rows):
        y = torch.compile(fn, backend=backend, fullgraph=True)(x, w, b, out)
        exact = _check_bits(y, lambda: _op(x, w, b), _reference(x, w, b), "compile")
        exact &= _check_bits(out, lambda: _op(x, w), _reference(x, w), "compile out=")
    ops = sum(t in ("aiter.gemm_a16w16", "aiter.gemm_a16w16_out") for t in targets)
    assert ops == 2, f"the compiled graph holds {ops} of the 2 custom ops"
    return [_row("torch.compile, custom ops in the graph", ops, True, exact)]


def check_cuda_graph():
    """The fast path captured in a CUDA graph replays to the custom op's bits."""
    m, n, k = 1056, 324, 2560
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    out = torch.empty(m, n, dtype=BF16, device="cuda")
    rows = []
    with _routed({_key(m, n, k, False, BF16, BF16): HIPBLASLT}):
        tgemm.mm(x, w)  # build outside the capture
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()

        def capture():
            with torch.cuda.graph(graph, stream=stream):
                y = tgemm.mm(x, w)
                tgemm.mm(x, w, out=out)
            return y

        y, calls = _custom_op_calls(capture)
        torch.cuda.current_stream().wait_stream(stream)
        for i in range(2):
            x.copy_(_rand(m, k))
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            ref = _reference(x, w)
            exact = _check_bits(y.clone(), lambda: _op(x, w), ref, "replay")
            exact &= _check_bits(out.clone(), lambda: _op(x, w), ref, "replay out=")
            rows.append(_row(f"CUDA graph, replay {i}", calls, False, exact))
        graph.reset()
    return rows


def check_preshuffled_weight():
    """A preshuffled BF16 weight, which the default config sends to hipBLASLt on
    gfx942, takes the fast path with bpreshuffle set."""
    m, n, k = 128, 256, 512
    config = tuned.get_GEMM_A16W16_config(
        M=m, N=n, K=k, bias=False, dtype=str(BF16), otype=str(BF16), bpreshuffle=True
    )
    assert config["libtype"] == "hipblaslt", f"{m}x{n}x{k} now selects {config}"
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    ws = shuffle_weight(w, layout=(16, 16))
    y, calls = _custom_op_calls(lambda: tgemm.mm(x, ws))
    exact = _check_bits(y, lambda: _op(x, ws), _reference(x, w), "preshuffled")
    return [_row("preshuffled weight, default config", calls, False, exact)]


def summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("tuned_gemm eager fast path: %s not covered", gfx)
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="eager fast path of tgemm.mm / gemm_a16w16 vs the custom ops",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[BF16, FP16],
        help="Input dtypes. e.g.: -d bf16",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        # Qwen3.8-Flash-Next prefill BF16 GEMMs at TP2 (2112 new tokens, 1056
        # per sequence-parallel shard), then two small shapes.
        default=[
            (1056, 324, 10240),
            (1056, 10240, 320),
            (2112, 513, 2560),
            (2112, 48, 2560),
            (2112, 640, 2560),
            (256, 512, 1024),
            (96, 384, 768),
        ],
        help="M,N,K shapes. e.g.: -s 1056,324,10240",
    )
    args = parser.parse_args()

    # First, while this process has not run a hipBLASLt GEMM yet.
    paths = check_first_calls()

    df = []
    for (m, n, k), dtype, bias, out in itertools.product(
        args.mnk, args.dtype, [False, True], OUT_KINDS
    ):
        df.append(test_eager(m, n, k, dtype, dtype, bias, 2, out))
    m, n, k = args.mnk[0]
    for otype, rank, out in itertools.product([None, FP32], [2, 3], OUT_KINDS):
        df.append(test_eager(m - m % 2, n, k, BF16, otype, False, rank, out))
    summarize("tuned_gemm eager fast path vs custom op", df)
    # A decode-sized GEMM, in the skinny kernels' default range on every arch.
    df = [
        test_fallback(4, 64, 1024, libtype)
        for libtype, (_, archs) in FALLBACKS.items()
        if gfx in archs
    ]
    summarize("tuned_gemm fallback to other backends vs custom op", df)

    paths += check_declines()
    paths += check_replaced_lookup()
    paths += check_plugin_route()
    paths += check_weight_handles()
    paths += check_compile()
    paths += check_cuda_graph()
    if gfx == "gfx942":
        paths += check_preshuffled_weight()
    else:
        # Only on gfx942 does the default config send it to hipBLASLt.
        aiter.logger.warning("%s: skipping the preshuffled-weight check", gfx)
    paths += check_shape_recording()
    summarize("tuned_gemm eager path checks", paths)


if __name__ == "__main__":
    main()

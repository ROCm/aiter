# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Eager fast paths of `gemm_a8w8_bpreshuffle` and `per_token_quant_hip`.

Outside torch.compile, a bpreshuffle GEMM that runs a CK kernel and a dynamic
per-token quant call their pybind entries directly instead of the
aiter::gemm_a8w8_bpreshuffle(_ck) and aiter::per_token_quant_hip /
dynamic_per_token_scaled_quant custom ops. They must give the custom ops' bits,
and every call those ops may treat differently must still reach them. Which
path ran is read from the profiler: each custom-op call is a dispatcher event,
and the fast paths record none.

Tensors are created with an explicit device: `torch.set_default_device` is a
TorchFunctionMode, under which every call takes the custom ops.
"""

import argparse
import contextlib
import itertools
import json
import os
import subprocess
import sys
import time

import pandas as pd
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import TorchFunctionMode
from torch.profiler import ProfilerActivity, profile
from torch.utils._python_dispatch import TorchDispatchMode

import aiter
import aiter.ops.gemm_op_a8w8 as gop
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import quant
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

BF16, FP16, FP8, I8 = dtypes.bf16, dtypes.fp16, dtypes.fp8, dtypes.i8
SUPPORTED_GFX = ["gfx942", "gfx950"]
CUSTOM_OPS = ("aiter::gemm_a8w8_bpreshuffle", "aiter::per_token_quant_hip")
CK = {"libtype": "ck", "splitK": 0}
CONFIGS = {"ck row": CK, "no row": None}


def _rand(*shape, dtype=BF16, scale=1.0):
    return torch.randn(*shape, device="cuda").mul_(scale).to(dtype)


@contextlib.contextmanager
def _routed(rows):
    """Serve the (M, N, K) of ``rows`` from ``rows``, None meaning no tuned row.

    Installed over `get_GEMM_config_with_quant_type`, which both paths call
    through the module on every call; other shapes keep the tuned lookup.
    """
    real = gop.get_GEMM_config_with_quant_type

    def lookup(M, N, K, q_dtype_w, tuned_file=None):
        if (M, N, K) in rows:
            row = rows[M, N, K]
            return None if row is None else dict(row)
        return real(M, N, K, q_dtype_w, tuned_file)

    gop.get_GEMM_config_with_quant_type = lookup
    try:
        yield
    finally:
        gop.get_GEMM_config_with_quant_type = real


def _custom_op_calls(fn):
    """``fn()``, and how many custom-op calls the dispatcher saw while it ran;
    an exception ``fn`` raises is returned in place of its result."""
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        try:
            result = fn()
        except (AssertionError, RuntimeError, ValueError) as e:
            result = e
    return result, sum(e.name in CUSTOM_OPS for e in prof.events())


def _gemm(xq, wq, x_scale, w_scale, **kwargs):
    return gop.gemm_a8w8_bpreshuffle(xq, wq, x_scale, w_scale, **kwargs)


def _gemm_op(xq, wq, x_scale, w_scale, dtype=BF16):
    """The custom-op path, called through the dispatcher."""
    return torch.ops.aiter.gemm_a8w8_bpreshuffle(
        xq, wq, x_scale, w_scale, None, dtype, False
    )


def _quant(x, quant_dtype=FP8, **kwargs):
    return quant.per_token_quant_hip(x, quant_dtype=quant_dtype, **kwargs)


def _quant_op(x, quant_dtype=FP8, num_rows=None):
    """The custom-op path, called through the dispatcher."""
    return torch.ops.aiter.per_token_quant_hip(x, None, quant_dtype, num_rows, 1)


def _weight(n, k):
    """A preshuffled FP8 weight, its scales, and the weight unshuffled."""
    w, w_scale = aiter.pertoken_quant(_rand(n, k, scale=k**-0.5), quant_dtype=FP8)
    return shuffle_weight(w, layout=(16, 16)), w_scale, w


def _inputs(m, n, k):
    xq, x_scale = _quant_op(_rand(m, k))
    wq, w_scale, w = _weight(n, k)
    return xq, wq, x_scale, w_scale, w


def _gemm_reference(xq, w, x_scale, w_scale, dtype):
    """The dequantized product by torch in float32."""
    return ((xq.float() * x_scale) @ (w.float() * w_scale).t()).to(dtype)


def _check_bits(y, op_call, what):
    """``y`` must be the custom op's bits wherever its kernel is reproducible;
    returns whether the bits were compared."""
    samples = [op_call() for _ in range(2)]
    if torch.equal(samples[0], samples[1]) and not torch.equal(y, samples[0]):
        # Two agreeing samples do not make a kernel reproducible.
        samples += [op_call() for _ in range(4)]
    exact = all(torch.equal(s, samples[0]) for s in samples[1:])
    if exact:
        differ = int((y != samples[0]).sum())
        assert differ == 0, f"{what}: {differ} elements differ from the custom op"
    return exact


def _same_quant(a, b):
    (qa, sa), (qb, sb) = a, b
    return torch.equal(qa.view(torch.uint8), qb.view(torch.uint8)) and torch.equal(
        sa, sb
    )


def _check_quant(x, q, scale, what):
    """``scale`` is the torch reference's, and every dequantized value is within
    one quantization step of ``x``."""
    _, ref_scale = aiter.pertoken_quant(x, quant_dtype=q.dtype)
    err = checkAllclose(ref_scale, scale, rtol=1e-3, atol=1e-3, msg=f"{what}: ")
    assert err == 0, f"{what}: {err:.2%} of the scales off the reference"
    xf = x.float()
    off = (q.float() * scale - xf).abs() > xf.abs() * 2**-3 + scale
    assert not off.any(), f"{what}: {int(off.sum())} values more than a step off"


def _host_us(fn, iters=50):
    """Host time of one call in us, over ``iters`` calls queued back to back."""
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return elapsed / iters * 1e6


def _row(case, calls, custom_op, bitwise=None):
    """A row of the path table: the call reached a custom op iff ``custom_op``."""
    path = "custom op" if calls else "fast path"
    assert bool(calls) == custom_op, f"{case}: took the {path}"
    return {"case": case, "path": path, "custom-op calls": calls, "bitwise": bitwise}


@benchmark()
def test_gemm(m, n, k, dtype, config):
    """The GEMM fast path against the custom op, for a CK row or no row."""
    xq, wq, x_scale, w_scale, w = _inputs(m, n, k)
    candidates = {
        "op": lambda: _gemm_op(xq, wq, x_scale, w_scale, dtype),
        "eager": lambda: _gemm(xq, wq, x_scale, w_scale, dtype=dtype),
    }
    ref = _gemm_reference(xq, w, x_scale, w_scale, dtype)
    flops = 2 * m * n * k
    nbytes = (m * k + n * k) * xq.element_size() + m * n * ref.element_size()
    ret = {"gfx": get_gfx_runtime()}
    with _routed({(m, n, k): CONFIGS[config]}):
        y, calls = _custom_op_calls(candidates["eager"])
        assert calls == 0, f"{m}x{n}x{k}: the call reached the custom op"
        assert y.dtype == dtype and y.shape == (m, n)
        ret["bitwise"] = _check_bits(y, candidates["op"], f"GEMM {m}x{n}x{k}")
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
def test_quant(shape, in_dtype, quant_dtype):
    """The quant fast path against the custom op."""
    x = _rand(*shape, dtype=in_dtype, scale=3.0)
    candidates = {
        "op": lambda: _quant_op(x, quant_dtype),
        "eager": lambda: _quant(x, quant_dtype=quant_dtype),
    }
    (q, scale), calls = _custom_op_calls(candidates["eager"])
    assert calls == 0, f"{shape}: the call reached the custom op"
    assert q.dtype == quant_dtype and q.shape == x.shape
    assert scale.dtype == dtypes.fp32 and scale.shape == (*shape[:-1], 1)
    assert _same_quant((q, scale), candidates["op"]()), "not the custom op's bits"
    nbytes = x.numel() * (x.element_size() + q.element_size())
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        (q, scale), us = run_perftest(fn)
        _check_quant(x, q, scale, name)
        ret[f"{name} us"] = us
        ret[f"{name} host us"] = _host_us(fn)
        ret[f"{name} TB/s"] = nbytes / us / 1e6
    return ret


def check_first_calls():
    """The first call of each op in a process runs the custom op, which loads
    its module; the calls after it take the fast path."""
    xq, wq, x_scale, w_scale, _ = _inputs(64, 256, 512)
    x = _rand(64, 512)
    rows = []
    with _routed({(64, 256, 512): CK}):
        for i in range(1, 4):
            _, calls = _custom_op_calls(lambda: _gemm(xq, wq, x_scale, w_scale))
            rows.append(_row(f"GEMM, call {i} of the process", calls, i == 1))
    for i in range(1, 4):
        _, calls = _custom_op_calls(lambda: _quant(x))
        rows.append(_row(f"quant, call {i} of the process", calls, i == 1))
    return rows


def _logged_calls():
    """`check_argument_logging`'s child: a quant and a GEMM, three times."""
    xq, wq, x_scale, w_scale, _ = _inputs(64, 256, 512)
    x = _rand(64, 512)
    calls = []
    with _routed({(64, 256, 512): CK}):
        for _ in range(3):
            calls.append(_custom_op_calls(lambda: _quant(x))[1])
            calls.append(_custom_op_calls(lambda: _gemm(xq, wq, x_scale, w_scale))[1])
    print(json.dumps(calls))


def check_argument_logging():
    """With AITER_LOG_MORE=2 set at start, which logs every op's arguments, each
    call takes the custom op."""
    here = os.path.dirname(os.path.abspath(__file__))
    code = (
        f"import sys; sys.path.insert(0, {here!r}); "
        "import test_a8w8_eager as t; t._logged_calls()"
    )
    child = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "AITER_LOG_MORE": "2"},
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert child.returncode == 0, child.stderr[-4000:]
    calls = json.loads(child.stdout.strip().splitlines()[-1])
    return [
        _row(f"AITER_LOG_MORE=2, {('quant', 'GEMM')[i % 2]} call {i // 2 + 1}", c, True)
        for i, c in enumerate(calls)
    ]


def check_gemm_declines():
    """GEMM calls the custom op treats differently reach it."""
    m, n, k = 64, 256, 512
    xq, wq, x_scale, w_scale, _ = _inputs(m, n, k)
    rows = []
    with _routed({(m, n, k): CK}):
        bias = torch.zeros(n, device="cuda")
        for case, kwargs, error in (
            ("bias", {"bias": bias}, "bias"),
            ("float32 output", {"dtype": torch.float32}, "not supported"),
        ):
            err, calls = _custom_op_calls(
                lambda kwargs=kwargs: _gemm(xq, wq, x_scale, w_scale, **kwargs)
            )
            assert error in str(err), f"{case}: not rejected by the custom op"
            rows.append(_row(f"GEMM, {case}", calls, True))
        err, calls = _custom_op_calls(lambda: _gemm(xq.to(BF16), wq, x_scale, w_scale))
        assert "must match" in str(err), "BF16 XQ: not rejected by the custom op"
        rows.append(_row("GEMM, BF16 XQ with an FP8 WQ", calls, True))

        ws = w_scale.clone().requires_grad_(True)
        _, calls = _custom_op_calls(lambda: _gemm(xq, wq, x_scale, ws))
        rows.append(_row("GEMM, w_scale requires grad", calls, True))
        with torch.no_grad():
            y, calls = _custom_op_calls(lambda: _gemm(xq, wq, x_scale, ws))
        exact = _check_bits(y, lambda: _gemm_op(xq, wq, x_scale, w_scale), "no_grad")
        rows.append(_row("GEMM, under no_grad", calls, False, exact))

    # The custom op zero-pads XQ to a K-padded weight's K.
    wq_pad, w_scale_pad, _ = _weight(n, k + 128)
    with _routed({(m, n, k): CK}):
        y, calls = _custom_op_calls(lambda: _gemm(xq, wq_pad, x_scale, w_scale_pad))
        exact = _check_bits(
            y, lambda: _gemm_op(xq, wq_pad, x_scale, w_scale_pad), "K-padded weight"
        )
    rows.append(_row("GEMM, K-padded weight", calls, True, exact))

    m, n, k = 2112, 2560, 3072
    xq, wq, x_scale, w_scale, _ = _inputs(m, n, k)
    with _routed({(m, n, k): {"libtype": "cktile", "splitK": 0}}):
        y, calls = _custom_op_calls(lambda: _gemm(xq, wq, x_scale, w_scale))
        exact = _check_bits(y, lambda: _gemm_op(xq, wq, x_scale, w_scale), "CKTile")
    rows.append(_row("GEMM, CKTile row", calls, True, exact))
    return rows


def check_quant_declines():
    """Quant calls the custom op treats differently reach it."""
    x = _rand(64, 512)
    err, calls = _custom_op_calls(lambda: _quant(x, scale=torch.ones(1, device="cuda")))
    assert "static per token quant" in str(err), "static scale not rejected"
    rows = [_row("quant, static scale", calls, True)]
    xg = x.clone().requires_grad_(True)
    _, calls = _custom_op_calls(lambda: _quant(xg))
    rows.append(_row("quant, x requires grad", calls, True))
    with torch.no_grad():
        out, calls = _custom_op_calls(lambda: _quant(xg))
    assert _same_quant(out, _quant_op(x)), "under no_grad: not the custom op's bits"
    rows.append(_row("quant, under no_grad", calls, False, True))

    num_rows = torch.tensor([100], dtype=torch.int32, device="cuda")
    x = _rand(256, 2560)
    (q, s), calls = _custom_op_calls(lambda: _quant(x, num_rows=num_rows))
    q_op, s_op = _quant_op(x, num_rows=num_rows)
    assert _same_quant((q[:100], s[:100]), (q_op[:100], s_op[:100]))
    rows.append(_row("quant, num_rows", calls, False, True))
    return rows


def check_modes():
    """Under dispatch or function modes and fake tensors both ops reach their
    custom ops."""
    m, n, k = 64, 256, 512
    xq, wq, x_scale, w_scale, _ = _inputs(m, n, k)
    x = _rand(m, k)
    rows = []
    with _routed({(m, n, k): CK}):
        seen = []

        class Record(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                seen.append(str(func))
                return func(*args, **(kwargs or {}))

        class Passthrough(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with Record():
            _quant(x)
            _gemm(xq, wq, x_scale, w_scale)
        for op in ("per_token_quant_hip", "gemm_a8w8_bpreshuffle"):
            calls = sum(name.startswith(f"aiter.{op}.") for name in seen)
            rows.append(_row(f"TorchDispatchMode, {op}", calls, True))
        with Passthrough():
            _, calls = _custom_op_calls(lambda: _quant(x))
            rows.append(_row("TorchFunctionMode, quant", calls, True))
            _, calls = _custom_op_calls(lambda: _gemm(xq, wq, x_scale, w_scale))
            rows.append(_row("TorchFunctionMode, GEMM", calls, True))

        with FakeTensorMode():
            xf = torch.empty(m, k, dtype=BF16, device="cuda")
            (qf, sf), calls = _custom_op_calls(lambda: _quant(xf))
            rows.append(_row("FakeTensorMode, quant", calls, True))
            wf = torch.empty(n, k, dtype=FP8, device="cuda")
            wsf = torch.empty(n, 1, dtype=dtypes.fp32, device="cuda")
            yf, calls = _custom_op_calls(lambda: _gemm(qf, wf, sf, wsf))
            rows.append(_row("FakeTensorMode, GEMM", calls, True))
        assert isinstance(qf, FakeTensor) and tuple(sf.shape) == (m, 1)
        assert isinstance(yf, FakeTensor) and tuple(yf.shape) == (m, n)
        # Fake tensors outside their mode still have to reach the ops' fakes.
        yf, calls = _custom_op_calls(lambda: _gemm(qf, wf, sf, wsf))
        assert isinstance(yf, FakeTensor)
        rows.append(_row("fake tensors outside their mode, GEMM", calls, True))
    return rows


def check_replaced_lookup():
    """A function installed over `get_GEMM_config_with_quant_type` is asked on
    every call, and its answer is never cached."""
    m, n, k = 128, 256, 1024
    xq, wq, x_scale, w_scale, _ = _inputs(m, n, k)
    asked = []

    def lookup(M, N, K, q_dtype_w, tuned_file=None):
        asked.append((M, N, K))
        return dict(CK)

    real = gop.get_GEMM_config_with_quant_type
    gop.get_GEMM_config_with_quant_type = lookup
    try:
        rows = []
        for i in range(1, 3):
            y, calls = _custom_op_calls(lambda: _gemm(xq, wq, x_scale, w_scale))
            assert asked == [(m, n, k)] * i
            rows.append(_row(f"replaced lookup, call {i}", calls, False))
        assert torch.equal(y, _gemm_op(xq, wq, x_scale, w_scale))
        assert len(asked) == 3, "the custom op asks the replaced lookup too"
    finally:
        gop.get_GEMM_config_with_quant_type = real
    return rows


def check_rebound_ops():
    """With an inner op replaced, the outer custom op calls the replacement, so
    the fast path, which would stand in for the original, declines."""
    m, n, k = 64, 256, 512
    xq, wq, x_scale, w_scale, _ = _inputs(m, n, k)
    x = _rand(m, k)
    ck, dq = gop.gemm_a8w8_bpreshuffle_ck, quant.dynamic_per_token_scaled_quant
    called = []

    def ck_spy(*args, **kwargs):
        called.append("ck")
        return ck(*args, **kwargs)

    def dq_spy(*args, **kwargs):
        called.append("quant")
        return dq(*args, **kwargs)

    gop.gemm_a8w8_bpreshuffle_ck = ck_spy
    quant.dynamic_per_token_scaled_quant = dq_spy
    try:
        with _routed({(m, n, k): CK}):
            _, quant_calls = _custom_op_calls(lambda: _quant(x))
            _, gemm_calls = _custom_op_calls(lambda: _gemm(xq, wq, x_scale, w_scale))
    finally:
        gop.gemm_a8w8_bpreshuffle_ck = ck
        quant.dynamic_per_token_scaled_quant = dq
    assert called == ["quant", "ck"], called
    return [
        _row("quant, inner op replaced", quant_calls, True),
        _row("GEMM, inner op replaced", gemm_calls, True),
    ]


def check_plugin_route():
    """While a kernel plugin is enabled, every GEMM goes through the route of
    `gemm_a8w8_bpreshuffle`: a provider that serves it runs, one that declines
    leaves it to the custom op. The quant is not routed and keeps its fast
    path. With none enabled, the GEMM fast path takes the call again."""
    from aiter import plugin

    m, n, k = 128, 256, 1024
    xq, wq, x_scale, w_scale, _ = _inputs(m, n, k)
    x = _rand(m, k)

    def run(XQ, WQ, x_scale, w_scale, bias, dtype, check):
        return torch.full((XQ.shape[0], WQ.shape[0]), 7.0, dtype=dtype, device="cuda")

    class Provider:
        name = "eager-test"
        api_versions = (plugin.API_VERSION,)
        op_versions = ("gemm_a8w8_bpreshuffle",)
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
        with _routed({(m, n, k): CK}):
            plugin.configure(enable=(Provider.name,), source="test_a8w8_eager")
            y, calls = _custom_op_calls(lambda: _gemm(xq, wq, x_scale, w_scale))
            assert bool((y == 7).all()), "the provider did not run"
            assert plugin.last_decision().route == "provider"
            rows.append(_row("GEMM, plugin serves the call", calls, True))
            provider.serve = False
            y, calls = _custom_op_calls(lambda: _gemm(xq, wq, x_scale, w_scale))
            assert plugin.last_decision().route == "native"
            exact = _check_bits(
                y, lambda: _gemm_op(xq, wq, x_scale, w_scale), "plugin declines"
            )
            rows.append(_row("GEMM, plugin enabled, declines", calls, True, exact))
            q, calls = _custom_op_calls(lambda: _quant(x))
            assert _same_quant(q, _quant_op(x)), "quant, plugin enabled"
            rows.append(_row("quant, plugin enabled", calls, False, True))
            plugin.configure(enable=(), source="test_a8w8_eager")
            y, calls = _custom_op_calls(lambda: _gemm(xq, wq, x_scale, w_scale))
            exact = _check_bits(
                y, lambda: _gemm_op(xq, wq, x_scale, w_scale), "nothing enabled"
            )
            rows.append(_row("GEMM, plugin with nothing enabled", calls, False, exact))
    finally:
        plugin._entry_points = entry_points
        plugin._reset_for_testing()
    return rows


def check_compile():
    """torch.compile records both custom ops and gives the eager bits."""
    m, n, k = 256, 512, 1024
    x = _rand(m, k)
    wq, w_scale, _ = _weight(n, k)
    targets = []

    def backend(gm, example_inputs):
        targets.extend(str(node.target) for node in gm.graph.nodes)
        return gm.forward

    def fn(x, wq, w_scale):
        xq, x_scale = _quant(x)
        return _gemm(xq, wq, x_scale, w_scale)

    torch._dynamo.reset()
    with _routed({(m, n, k): CK}):
        y = torch.compile(fn, backend=backend, fullgraph=True)(x, wq, w_scale)
        xq, x_scale = _quant_op(x)
        assert torch.equal(y, _gemm_op(xq, wq, x_scale, w_scale))
    ops = sum(
        t in ("aiter.per_token_quant_hip", "aiter.gemm_a8w8_bpreshuffle")
        for t in targets
    )
    assert ops == 2, f"the compiled graph holds {ops} of the 2 custom ops"
    return [_row("torch.compile, custom ops in the graph", ops, True)]


def check_cuda_graph():
    """Both fast paths captured in a CUDA graph replay to the custom ops' bits."""
    m, n, k = 2112, 8192, 2560
    x = _rand(m, k)
    wq, w_scale, _ = _weight(n, k)
    rows = []
    with _routed({(m, n, k): CK}):
        xq, x_scale = _quant(x)  # load both modules outside the capture
        _gemm(xq, wq, x_scale, w_scale)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()

        def capture():
            with torch.cuda.graph(graph, stream=stream):
                q, s = _quant(x)
                return q, s, _gemm(q, wq, s, w_scale)

        (q, s, y), calls = _custom_op_calls(capture)
        torch.cuda.current_stream().wait_stream(stream)
        for i in range(2):
            x.copy_(_rand(m, k))
            graph.replay()
            torch.cuda.synchronize()
            q_op, s_op = _quant_op(x)
            assert _same_quant((q, s), (q_op, s_op)), "replay: quant differs"
            exact = _check_bits(
                y, lambda q=q_op, s=s_op: _gemm_op(q, wq, s, w_scale), "replay"
            )
            rows.append(_row(f"CUDA graph, replay {i}", calls, False, exact))
        graph.reset()
    return rows


def summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    gfx = get_gfx_runtime()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning("a8w8 eager fast paths: %s not covered", gfx)
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="eager fast paths of gemm_a8w8_bpreshuffle and "
        "per_token_quant_hip vs the custom ops",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[BF16, FP16],
        help="GEMM output and quant input dtypes. e.g.: -d bf16",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        # FP8 linear layers of Qwen3.8-Flash-Next PTPC FP8 at TP2 (GDN in/out
        # projections, attention qkv/o) for a 2112-token prefill chunk, and
        # small batches.
        default=[
            (2112, 8192, 2560),
            (2112, 2560, 3072),
            (2112, 6656, 2560),
            (2112, 3584, 2560),
            (1, 8192, 2560),
            (33, 2560, 3072),
        ],
        help="GEMM M,N,K shapes. e.g.: -s 2112,8192,2560",
    )
    args = parser.parse_args()

    # First, while this process has run neither op yet.
    paths = check_first_calls()

    df = [
        test_gemm(m, n, k, dtype, config)
        for (m, n, k), dtype, config in itertools.product(args.mnk, args.dtype, CONFIGS)
    ]
    summarize("gemm_a8w8_bpreshuffle eager fast path vs custom op", df)
    df = [
        test_quant(shape, in_dtype, quant_dtype)
        for shape, in_dtype, quant_dtype in itertools.product(
            [(2112, 2560), (1, 3072), (4, 33, 2560)], args.dtype, [FP8, I8]
        )
    ]
    summarize("per_token_quant_hip eager fast path vs custom op", df)

    paths += check_gemm_declines()
    paths += check_quant_declines()
    paths += check_modes()
    paths += check_replaced_lookup()
    paths += check_rebound_ops()
    paths += check_plugin_route()
    paths += check_compile()
    paths += check_cuda_graph()
    paths += check_argument_logging()
    summarize("a8w8 eager path checks", paths)


if __name__ == "__main__":
    main()

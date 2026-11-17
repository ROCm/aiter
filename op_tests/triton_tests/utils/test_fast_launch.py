# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for `aiter.ops.triton.utils.fast_launch`.

A cached launch is only sound while its key holds everything Triton
specializes on. A key that is too coarse does not raise: it runs a kernel
compiled for other arguments. So most tests here compare the key with
Triton's own cache key, or check that a launch which needs Triton's side
effects (hooks, global checks, config pre-hooks) still gets them.
"""

import pytest
import torch
import triton
import triton.language as tl
from triton import knobs
from triton.compiler.compiler import make_backend
from triton.runtime import driver
from triton.runtime.jit import compute_cache_key, create_function_from_signature

from aiter.ops.triton.utils import fast_launch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fast_launch needs a GPU"
)


@triton.jit
def _probe_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    rows,
    stride,
    scale,
    flag,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    ALPHA: tl.constexpr,
):
    pass


@triton.jit
def _axpy_kernel(x_ptr, y_ptr, out_ptr, n, alpha, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x * alpha + y, mask=mask)


@triton.jit
def _scale_by_first_kernel(x_ptr, out_ptr, factors, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) * factors[0], mask=mask)


_OFFSET = tl.constexpr(2.0)


@triton.jit
def _offset_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) + _OFFSET, mask=mask)


# `JITFunction.run` puts this knob in the compile options only on the Triton
# builds that have it.
_FPSAN = "fpsan_homomorphic_casts" in knobs.compilation.knob_descriptors


def _triton_key(kernel, args, kwargs):
    """`JITFunction.run`'s cache key for a launch, from Triton's own binder."""
    backend = make_backend(driver.active.get_current_target())
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    kwargs = dict(kwargs)
    kwargs["debug"] = kwargs.get("debug", kernel.debug) or knobs.runtime.debug
    kwargs["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    if _FPSAN:
        kwargs["fpsan_homomorphic_casts"] = knobs.compilation.fpsan_homomorphic_casts
    _, specialization, options = binder(*args, **kwargs)
    return compute_cache_key({}, specialization, options)


_BIG = []


def _tensor(numel, dtype=torch.bfloat16, misaligned=False, big=False):
    """A tensor whose data pointer is 16-byte aligned or not, on a storage of
    at most or just over 2 GiB."""
    if big:
        if not _BIG:
            _BIG.append(torch.empty(2**31 + 64, dtype=torch.uint8, device="cuda"))
        flat = _BIG[0].view(dtype)
    else:
        flat = torch.empty(numel + 16, dtype=dtype, device="cuda")
    start = 1 if misaligned else 0
    return flat[start : start + numel]


def _probe_args():
    x = _tensor(64)
    return [x, x, _tensor(32), 64, 48, 0.5, True, 32, 1, 2.0]


def _variants(base):
    """`base`, then each argument changed in every way Triton's
    specialization or the key can tell apart."""
    constexpr = [p.is_constexpr for p in _probe_kernel.params]
    yield base
    for i, arg in enumerate(base):
        if constexpr[i]:
            options = [arg + 1, float(arg) + 0.5] if not isinstance(arg, bool) else []
            options += [not arg] if isinstance(arg, bool) else [True]
        elif isinstance(arg, torch.Tensor):
            numel, dtype = arg.numel(), arg.dtype
            options = [
                _tensor(numel, dtype, misaligned=True),
                _tensor(numel, dtype, big=True),
                _tensor(numel, dtype, misaligned=True, big=True),
                _tensor(numel, torch.float32),
                _tensor(numel, torch.float16),
                torch.nn.Parameter(_tensor(numel, dtype), requires_grad=False),
            ]
        elif isinstance(arg, bool):
            options = [not arg, 1, 1.0, None]
        elif isinstance(arg, int):
            options = [0, 1, 2, 15, 16, 17, 32, 324, 1056, -16, -3, True, 1.0]
            options += [2**31, -(2**31) - 1, 2**31 + 16, 2**63, 2**63 + 16]
        else:
            options = [0.25, 1.0, 1e-6, 0, 1, 16, True, None]
        for option in options:
            variant = list(base)
            variant[i] = option
            yield variant


@pytest.mark.parametrize("by_specialization", [False, True])
@pytest.mark.parametrize("buffer_ops", ["1", "0"])
@pytest.mark.parametrize("constexprs_by_name", [False, True])
def test_key_tells_apart_what_triton_tells_apart(
    monkeypatch, by_specialization, buffer_ops, constexprs_by_name
):
    """Two launches with the same key must have the same Triton cache key.
    This also fails when a Triton upgrade changes what it specializes on."""
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", buffer_ops)
    wrapped = fast_launch.fast_launch(
        _probe_kernel, by_specialization=by_specialization
    )
    seen, distinct = {}, set()
    for args in _variants(_probe_args()):
        kwargs = {"num_warps": 4}
        if constexprs_by_name:
            kwargs.update(zip(("BLOCK", "MODE", "ALPHA"), args[7:]))
            args = args[:7]
        triton_key = _triton_key(_probe_kernel, args, kwargs)
        distinct.add(triton_key)
        key = wrapped._key(0, tuple(args), kwargs)
        assert seen.setdefault(key, triton_key) == triton_key, args
    assert len(distinct) > 20


@pytest.mark.parametrize(
    "knob,env",
    [
        ("runtime.debug", None),
        ("compilation.instrumentation_mode", None),
        (None, "TRITON_FPSAN_HOMOMORPHIC_CASTS"),
        (None, "AMDGCN_USE_BUFFER_OPS"),
    ],
)
def test_key_follows_the_knobs_in_tritons_key(monkeypatch, knob, env):
    if env == "TRITON_FPSAN_HOMOMORPHIC_CASTS" and not _FPSAN:
        pytest.skip("this Triton build has no fpsan_homomorphic_casts knob")
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.delenv("TRITON_FPSAN_HOMOMORPHIC_CASTS", raising=False)
    wrapped = fast_launch.fast_launch(_probe_kernel)
    args, kwargs = _probe_args(), {"num_warps": 4}
    before = (
        wrapped._key(0, tuple(args), kwargs),
        _triton_key(_probe_kernel, args, kwargs),
    )
    if knob is not None:
        group, attr = knob.split(".")
        monkeypatch.setattr(
            getattr(knobs, group), attr, True if attr == "debug" else "fpsan"
        )
    else:
        monkeypatch.setenv(env, "0" if env == "AMDGCN_USE_BUFFER_OPS" else "1")
    after = (
        wrapped._key(0, tuple(args), kwargs),
        _triton_key(_probe_kernel, args, kwargs),
    )
    assert before[1] != after[1]
    assert before[0] != after[0]


def test_key_tells_apart_devices_and_argument_layouts():
    wrapped = fast_launch.fast_launch(_probe_kernel)
    args = _probe_args()
    named = dict(zip(("BLOCK", "MODE", "ALPHA"), args[7:]))
    keys = {
        wrapped._key(0, tuple(args), {}),
        wrapped._key(1, tuple(args), {}),
        wrapped._key(0, tuple(args[:7]), named),
        wrapped._key(0, tuple(args[:7]), dict(reversed(named.items()))),
    }
    assert len(keys) == 4


def test_true_one_and_one_point_zero_do_not_share_an_entry():
    """They are equal and hash the same, but Triton specializes them apart."""
    wrapped = fast_launch.fast_launch(_probe_kernel)
    keys = set()
    for value in (1, True, 1.0):
        args = _probe_args()
        args[8] = value
        keys.add(wrapped._key(0, tuple(args), {}))
    assert len(keys) == 3


def test_by_specialization_needs_a_plain_kernel():
    tuned = triton.heuristics({"BLOCK": lambda args: 64})(_axpy_kernel)
    with pytest.raises(ValueError, match="plain"):
        fast_launch.fast_launch(tuned, by_specialization=True)


def _axpy_inputs(n, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(n, generator=gen, device="cuda", dtype=torch.float32)
    y = torch.randn(n, generator=gen, device="cuda", dtype=torch.float32)
    return x, y, torch.empty(n, device="cuda", dtype=torch.float32)


def _axpy(wrapped, x, y, out, alpha=0.5, style="positional"):
    n = x.numel()
    grid = (triton.cdiv(n, 128),)
    if style == "positional":
        wrapped[grid](x, y, out, n, alpha, 128, num_warps=4)
    elif style == "mixed":
        wrapped[grid](x, y, out, n=n, alpha=alpha, BLOCK=128, num_warps=4)
    else:
        grid = lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),)
        wrapped[grid](x, y, out, n, alpha=alpha, BLOCK=128, num_warps=4)
    return out


@pytest.mark.parametrize("by_specialization", [False, True])
@pytest.mark.parametrize("style", ["positional", "mixed", "callable_grid"])
def test_cached_launch_matches_ordinary_launch(by_specialization, style):
    wrapped = fast_launch.fast_launch(_axpy_kernel, by_specialization=by_specialization)
    for n in (1000, 4096, 1000, 77):
        x, y, out = _axpy_inputs(n, seed=n)
        _axpy(wrapped, x, y, out, style=style)
        with fast_launch.recording() as misses:
            got = _axpy(wrapped, x, y, out.clone(), style=style)
        assert misses == [], "the second launch of a key has to hit"
        with fast_launch.bypassed():
            want = _axpy(wrapped, x, y, torch.empty_like(out), style=style)
        assert torch.equal(got, want)


def test_by_specialization_shares_an_entry_across_sizes():
    exact = fast_launch.fast_launch(_axpy_kernel)
    by_spec = fast_launch.fast_launch(_axpy_kernel, by_specialization=True)
    for n in (1024, 2048):
        x, y, out = _axpy_inputs(n)
        with fast_launch.recording() as exact_misses:
            _axpy(exact, x, y, out)
        with fast_launch.recording() as spec_misses:
            got = _axpy(by_spec, x, y, out.clone())
        assert exact_misses, "the exact key separates every size"
        assert (len(spec_misses) == 1) == (n == 1024)
        torch.testing.assert_close(got, x * 0.5 + y, rtol=0, atol=0)


def test_a_launch_hook_sees_every_launch():
    wrapped = fast_launch.fast_launch(_axpy_kernel)
    x, y, out = _axpy_inputs(256)
    _axpy(wrapped, x, y, out)
    seen = []
    chain = knobs.runtime.launch_enter_hook
    hook = lambda metadata: seen.append(1)
    chain.add(hook)
    try:
        for _ in range(3):
            _axpy(wrapped, x, y, out)
    finally:
        chain.remove(hook)
    assert seen == [1, 1, 1]


@pytest.mark.parametrize("case", ["pre_run_hook", "stages_hook"])
def test_launches_that_need_the_ordinary_path_take_it(monkeypatch, case):
    wrapped = fast_launch.fast_launch(_axpy_kernel)
    x, y, out = _axpy_inputs(256)
    _axpy(wrapped, x, y, out)
    assert wrapped._plain()
    if case == "pre_run_hook":
        calls = []
        monkeypatch.setattr(
            _axpy_kernel, "pre_run_hooks", [lambda *a, **k: calls.append(1)]
        )
    else:
        monkeypatch.setattr(
            knobs.runtime, "add_stages_inspection_hook", lambda: ("key", "hash")
        )
    assert not wrapped._plain()
    if case == "pre_run_hook":
        for _ in range(2):
            _axpy(wrapped, x, y, out)
        assert calls == [1, 1]


@pytest.mark.parametrize("by_specialization", [False, True])
def test_a_tuple_argument_takes_the_ordinary_path(by_specialization):
    """Triton compiles (2,) and (2.0,) apart, but the two tuples are equal."""
    wrapped = fast_launch.fast_launch(
        _scale_by_first_kernel, by_specialization=by_specialization
    )
    x = torch.randn(256, device="cuda", dtype=torch.float32)
    for factors in ((2,), (2.0,)):
        out = torch.empty_like(x)
        wrapped[(2,)](x, out, factors, 256, BLOCK=128)
        torch.testing.assert_close(out, x * 2, rtol=0, atol=0)
    assert not wrapped._cache


@pytest.mark.parametrize("backend", ["eager", "inductor"])
@pytest.mark.parametrize("by_specialization", [False, True])
@pytest.mark.parametrize("style", ["positional", "mixed", "callable_grid"])
def test_fullgraph_compile_takes_the_ordinary_path(backend, by_specialization, style):
    """Under torch.compile the wrapper must hand the launch to Triton before it
    reads any state Dynamo cannot branch on, even with an entry cached."""
    torch._dynamo.reset()
    wrapped = fast_launch.fast_launch(_axpy_kernel, by_specialization=by_specialization)
    x, y, out = _axpy_inputs(1000, seed=1)
    _axpy(wrapped, x, y, out, style=style)

    def step(x, y):
        out = _axpy(wrapped, x, y, torch.empty_like(x), style=style)
        return out * 2.0

    compiled = torch.compile(step, backend=backend, fullgraph=True)
    with fast_launch.recording() as misses:
        got = compiled(x, y)
    assert misses == [], "a traced launch must not reach the cache"
    with fast_launch.bypassed():
        want = step(x, y)
    assert torch.equal(got, want)
    torch._dynamo.reset()


def test_a_changed_global_raises_as_triton_does(monkeypatch):
    wrapped = fast_launch.fast_launch(_offset_kernel)
    x = torch.randn(256, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)
    for _ in range(2):
        wrapped[(2,)](x, out, 256, BLOCK=128)
    torch.testing.assert_close(out, x + 2.0, rtol=0, atol=0)
    monkeypatch.setitem(globals(), "_OFFSET", tl.constexpr(3.0))
    with pytest.raises(RuntimeError, match="_OFFSET"):
        wrapped[(2,)](x, out, 256, BLOCK=128)


def test_a_config_pre_hook_runs_on_every_launch():
    calls = []
    tuned = triton.autotune(
        configs=[triton.Config({}, num_warps=4, pre_hook=lambda args: calls.append(1))],
        key=[],
    )(_axpy_kernel)
    wrapped = fast_launch.fast_launch(tuned)
    x, y, out = _axpy_inputs(256)
    for _ in range(3):
        wrapped[(2,)](x, y, out, 256, 0.5, BLOCK=128)
    assert calls == [1, 1, 1]
    assert not wrapped._cache


def test_a_heuristic_on_a_tensor_shape_is_keyed():
    """A config chosen from a tensor's shape must not serve another shape."""
    tuned = triton.heuristics(
        {"BLOCK": lambda args: 128 if args["x_ptr"].shape[0] > 256 else 32}
    )(_axpy_kernel)
    wrapped = fast_launch.fast_launch(tuned)
    grid = lambda meta: (triton.cdiv(meta["n"], meta["BLOCK"]),)
    for numel in (200, 1024):
        x, y, out = _axpy_inputs(numel)
        with fast_launch.recording() as misses:
            wrapped[grid](x, y, out, 64, 0.5)
        assert misses, "a new shape has to miss on a wrapped kernel"
        torch.testing.assert_close(out[:64], x[:64] * 0.5 + y[:64], rtol=0, atol=0)
    blocks = sorted(frozen["BLOCK"] for _, _, frozen in wrapped._cache.values())
    assert blocks == [32, 128]


def test_cached_launch_replays_in_a_cuda_graph():
    wrapped = fast_launch.fast_launch(_axpy_kernel, by_specialization=True)
    x, y, out = _axpy_inputs(4096)
    for _ in range(2):
        _axpy(wrapped, x, y, out)  # the second one is a cached launch
    out.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _axpy(wrapped, x, y, out)
    x.copy_(torch.randn_like(x))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, x * 0.5 + y, rtol=0, atol=0)

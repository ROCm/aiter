# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device dispatch anchors: schema-compatible input reuse, not tensor caching."""

import importlib
import importlib.util
import itertools

import pytest
import torch

guard = importlib.import_module("aiter.jit.utils.torch_guard")


class _LegacyDummy:
    """Reference call spelling before inference-anchor reuse.

    The decorator still registers the same schema; only the adapter being
    compared here always allocates its dummy, as the previous wrapper did.
    """

    @staticmethod
    def torch_compile_guard(**kwargs):
        def decorate(fn):
            guard.torch_compile_guard(**kwargs)(fn)
            device = kwargs.get("device", "cpu")

            def call(*args, **kw):
                return getattr(torch.ops.aiter, fn.__name__)(
                    torch.empty(1, device=device), *args, **kw
                )

            return call

        return decorate


old = _LegacyDummy()
counter = itertools.count()


def register(func, g=guard, **kw):
    func.__name__ = f"test_inference_anchor_{next(counter)}"
    return func, g.torch_compile_guard(**kw)(func)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("callstyle", ["positional", "keyword", "mixed"])
@pytest.mark.parametrize("shape", [(), (0, 3), (3, 4)])
def test_no_allocation(device, callstyle, shape, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs GPU")

    def kernel(handle: int, x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(x + handle)

    fn, op = register(kernel, device=device)
    x = torch.ones(shape, device=device)
    o = torch.empty_like(x)
    empty = torch.empty
    alloc = []

    def counted(*a, **kw):
        alloc.append((a, kw))
        return empty(*a, **kw)

    with monkeypatch.context() as m:
        m.setattr(torch, "empty", counted)
        if callstyle == "positional":
            op(2, x, o)
        elif callstyle == "keyword":
            op(handle=2, x=x, out=o)
        else:
            op(2, x=x, out=o)
    assert not alloc
    torch.testing.assert_close(o, x + 2)
    assert "Tensor dummy" in str(getattr(torch.ops.aiter, fn.__name__).default._schema)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_raw_schema_and_existing_registration(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs GPU")

    def kernel(handle: int, x: torch.Tensor) -> torch.Tensor:
        return x + handle

    fn, _op = register(kernel, device=device)
    rebound = guard.torch_compile_guard(device=device)(fn)
    x = torch.ones(3, device=device)
    dummy = torch.empty(1, device=device)
    torch.testing.assert_close(
        getattr(torch.ops.aiter, fn.__name__)(dummy, 3, x), x + 3
    )
    empty = torch.empty
    alloc = []

    def counted(*a, **kw):
        alloc.append(1)
        return empty(*a, **kw)

    with monkeypatch.context() as m:
        m.setattr(torch, "empty", counted)
        rebound(3, x)
    assert (
        len(alloc) == 1
    ), "External/pre-existing registration must retain dummy allocation"


def test_optional_and_scalar_unchanged(monkeypatch):
    def optional(handle: int, x: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros(1) if x is None else x + handle

    _, op = register(optional)
    empty = torch.empty
    alloc = []

    def counted(*a, **kw):
        alloc.append(1)
        return empty(*a, **kw)

    with monkeypatch.context() as m:
        m.setattr(torch, "empty", counted)
        assert op(1).item() == 0
    assert len(alloc) == 1

    def scalar(handle: int, x: torch.Tensor) -> int:
        return handle

    _, op = register(scalar)
    with monkeypatch.context() as m:
        alloc.clear()
        m.setattr(torch, "empty", counted)
        assert op(7, torch.ones(3)) == 7
    assert len(alloc) == 2  # input and returned carrier still follow old scalar path


def test_kwonly_tensor(monkeypatch):
    def kernel(h: int, *, x: torch.Tensor) -> torch.Tensor:
        return x + h

    _, op = register(kernel)
    x = torch.ones(3)

    def fail(*a, **kw):
        pytest.fail("No dummy allocation needed")

    with monkeypatch.context() as m:
        m.setattr(torch, "empty", fail)
        y = op(3, x=x)
    torch.testing.assert_close(y, x + 3)


def test_cpu_tensor_cuda_anchor_falls_back(monkeypatch):
    # Inspect dispatch before trying a GPU allocation. Preserve old device argument.
    def kernel(h: int, x: torch.Tensor) -> torch.Tensor:
        return x + h

    fn, op = register(kernel, device="cuda")
    sent = []
    sentinel = torch.ones(1)

    def record(*a, **kw):
        sent.append(kw)
        return sentinel

    monkeypatch.setattr(torch, "empty", record)
    # Avoid actual mixed-device dispatch by replacing the packet with a recorder.
    seen = []
    monkeypatch.setattr(torch.ops.aiter, fn.__name__, lambda *a: seen.append(a) or a[2])
    x = torch.ones(3)
    assert op(1, x) is x
    assert sent == [{"device": "cuda"}] and seen[0][0] is sentinel


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fake_compile_functionalization_and_grad(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs GPU")

    def kernel(h: int, x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(x + h)

    def fake(h, x, out):
        return None

    fn, op = register(kernel, device=device, mutates_args=["out"], gen_fake=fake)

    def call(x):
        out = torch.empty_like(x)
        op(3, x, out)
        return out

    x = torch.ones((4, 5), device=device)
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        y = call(torch.empty((4, 5), device=device))
        assert y.shape == (4, 5)
    graphs = []

    def backend(gm, _inputs):
        graphs.append(gm)
        return gm.forward

    compiled = torch.compile(call, backend=backend, fullgraph=True)
    torch.testing.assert_close(compiled(x), x + 3)
    assert graphs and any(fn.__name__ in str(n.target) for n in graphs[0].graph.nodes)
    compiled = torch.compile(call, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), x + 3)
    # This schema's direct functionalize failure exists on the unchanged wrapper.
    try:
        actual = torch.func.functionalize(call)(x)
        candidate_error = None
    except RuntimeError as e:
        candidate_error = str(e).split(":")[0]
    _, old_op = register(
        kernel, g=old, device=device, mutates_args=["out"], gen_fake=fake
    )

    def old_call(x):
        y = torch.empty_like(x)
        old_op(3, x, y)
        return y

    try:
        expected = torch.func.functionalize(old_call)(x)
        baseline_error = None
    except RuntimeError as e:
        baseline_error = str(e).split(":")[0]
    assert candidate_error == baseline_error
    if candidate_error is None:
        torch.testing.assert_close(actual, expected)
    # Real graph: aliasing input as dummy must not change in-place output.
    if device == "cuda":
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            y = call(x)
        torch.cuda.current_stream().wait_stream(s)
        x.add_(4)
        g.replay()
        torch.testing.assert_close(y, x + 3)


def test_pure_functionalization_and_input_mutation():
    for g in (old, guard):

        def pure(h: int, x: torch.Tensor) -> torch.Tensor:
            return x + h

        _, pure_op = register(pure, g=g, device="cpu", mutates_args=[])
        x = torch.ones(4)
        torch.testing.assert_close(
            torch.func.functionalize(lambda x, op=pure_op: op(3, x))(x), x + 3
        )

        def inplace(h: int, x: torch.Tensor, out: torch.Tensor) -> None:
            x.add_(h)
            out.copy_(x * 2)

        _, mut_op = register(
            inplace,
            g=g,
            device="cpu",
            mutates_args="unknown",
            gen_fake=lambda h, x, out: None,
        )

        def f(x, op=mut_op):
            y = torch.empty_like(x)
            op(3, x, y)
            return x, y

        compiled = torch.compile(f, backend="aot_eager", fullgraph=True)
        for _ in range(3):
            x = torch.ones(4)
            a, b = compiled(x)
            torch.testing.assert_close(a, torch.full_like(x, 4))
            torch.testing.assert_close(b, torch.full_like(x, 8))
            torch.testing.assert_close(x, torch.full_like(x, 4))


def test_default_device_fallback(monkeypatch):
    # The compile-time device field need not be a simple string. Keep it unchanged.
    for device in (torch.device("cpu"), "cpu:0"):

        def kernel(h: int, x: torch.Tensor) -> torch.Tensor:
            return x + h

        _, op = register(kernel, device=device)
        x = torch.ones(4)
        calls = []
        empty = torch.empty

        def counted(*a, _calls=calls, _empty=empty, **kw):
            _calls.append(kw)
            return _empty(*a, **kw)

        with monkeypatch.context() as m:
            m.setattr(torch, "empty", counted)
            y = op(2, x)
        torch.testing.assert_close(y, x + 2)
        assert len(calls) == 1


@pytest.mark.parametrize("special", ["grad", "subclass", "parameter"])
def test_special_tensors_keep_allocation(monkeypatch, special):
    def kernel(h: int, x: torch.Tensor) -> torch.Tensor:
        return x + h

    _, op = register(kernel, device="cpu", mutates_args=[])
    x = torch.ones(3)
    if special == "grad":
        x.requires_grad_()
    elif special == "parameter":
        x = torch.nn.Parameter(x, requires_grad=False)
    else:

        class TensorSubclass(torch.Tensor):
            pass

        x = x.as_subclass(TensorSubclass)
    alloc = []
    empty = torch.empty

    def counted(*a, **kw):
        alloc.append(1)
        return empty(*a, **kw)

    with monkeypatch.context() as m:
        m.setattr(torch, "empty", counted)
        y = op(2, x)
    torch.testing.assert_close(y, x + 2)
    assert len(alloc) == 1


def test_tensor_first_path_remains_allocation_free(monkeypatch):
    def kernel(x: torch.Tensor, h: int) -> torch.Tensor:
        return x + h

    _, op = register(kernel, device="cpu", mutates_args=[])
    x = torch.ones(3)
    with monkeypatch.context() as m:
        m.setattr(
            torch, "empty", lambda *a, **kw: pytest.fail("tensor-first allocation")
        )
        y = op(x, 2)
    torch.testing.assert_close(y, x + 2)


@pytest.mark.parametrize("style", ["missing", "invalid", "duplicate"])
def test_invalid_calls_preserve_dispatcher_errors(style):
    results = []
    for g in (old, guard):

        def kernel(h: int, x: torch.Tensor) -> torch.Tensor:
            return x + h

        _, op = register(kernel, g=g, device="cpu", mutates_args=[])
        try:
            if style == "missing":
                op(2)
            elif style == "invalid":
                op(2, "not a tensor")
            else:
                op(2, torch.ones(3), x=torch.ones(3))
        except (TypeError, RuntimeError) as e:
            results.append(type(e))
        else:
            pytest.fail("Invalid call unexpectedly accepted")
    assert results[0] is results[1]


def test_existing_registration_can_inspect_dummy(monkeypatch):
    def kernel(h: int, x: torch.Tensor) -> torch.Tensor:
        return x + h

    fn, _ = register(kernel, device="cpu", mutates_args=[])
    # A foreign implementation could rely on dummy size=1 or independence.
    rebound = guard.torch_compile_guard(device="cpu", mutates_args=[])(fn)
    x = torch.ones(3)
    seen = []
    monkeypatch.setattr(
        torch.ops.aiter, fn.__name__, lambda dummy, h, x: seen.append(dummy) or x + h
    )
    torch.testing.assert_close(rebound(2, x), x + 2)
    assert seen[0].shape == (1,) and seen[0].data_ptr() != x.data_ptr()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_inference_mode(device, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs GPU")

    def kernel(h: int, x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(x + h)

    _, op = register(kernel, device=device)
    with torch.inference_mode():
        x = torch.ones(4, device=device)
        o = torch.empty_like(x)
        with monkeypatch.context() as m:
            m.setattr(
                torch,
                "empty",
                lambda *a, **kw: pytest.fail("inference dummy allocated"),
            )
            op(3, x, o)
        torch.testing.assert_close(o, x + 3)

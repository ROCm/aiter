"""`aiter.jit.utils.call_hook`: a registered callback sees every call of a
`torch_compile_guard` op with its nesting depth, including the calls a
compiled graph makes at runtime, and costs nothing once removed."""

import torch

from aiter.jit.utils.call_hook import register_call_hook, remove_call_hook
from aiter.jit.utils.torch_guard import torch_compile_guard


@torch_compile_guard(device="cpu")
def _test_call_hook_inner(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@torch_compile_guard(device="cpu")
def _test_call_hook_outer(x: torch.Tensor) -> torch.Tensor:
    return _test_call_hook_inner(x) + 1


def _recorder():
    calls = []

    def hook(name, depth, args, kwargs):
        calls.append((name, depth, tuple(args[0].shape)))

    return calls, hook


def test_both_torch_guard_copies_share_one_registry():
    import sys

    import aiter  # noqa: F401  loads jit/core.py's by-path copy too

    copies = [
        sys.modules[name]
        for name in ("torch_guard", "aiter.jit.utils.torch_guard")
        if name in sys.modules
    ]
    assert len(copies) == 2
    assert copies[0]._call_hooks is copies[1]._call_hooks


def test_no_hook_runs_the_op():
    x = torch.ones(3)
    assert torch.equal(_test_call_hook_outer(x), torch.full((3,), 3.0))


def test_hook_sees_nesting_depth():
    calls, hook = _recorder()
    register_call_hook(hook)
    try:
        out = _test_call_hook_outer(torch.ones(4, 2))
    finally:
        remove_call_hook(hook)
    assert torch.equal(out, torch.full((4, 2), 3.0))
    assert calls == [
        ("_test_call_hook_outer", 0, (4, 2)),
        ("_test_call_hook_inner", 1, (4, 2)),
    ]


def test_hook_fires_on_every_compiled_call():
    def forward(x):
        return _test_call_hook_outer(x) * 3

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    compiled(torch.ones(5))  # trace first, so only runtime calls are counted
    calls, hook = _recorder()
    register_call_hook(hook)
    try:
        for _ in range(3):
            out = compiled(torch.ones(5))
    finally:
        remove_call_hook(hook)
    assert torch.equal(out, torch.full((5,), 9.0))
    assert [c for c in calls if c[1] == 0] == [("_test_call_hook_outer", 0, (5,))] * 3


def test_removed_hook_is_not_called():
    calls, hook = _recorder()
    register_call_hook(hook)
    remove_call_hook(hook)
    _test_call_hook_outer(torch.ones(2))
    assert calls == []


def test_hook_sees_ops_registered_through_cpp_itfs():
    """`csrc.cpp_itfs.torch_utils.direct_register_custom_op` is aiter's other
    registration path (`pa_decode_gluon` takes it); its ops call the hooks
    too."""
    import aiter  # noqa: F401  puts csrc on the path
    from csrc.cpp_itfs.torch_utils import direct_register_custom_op

    def _test_call_hook_cpp_itfs(x: torch.Tensor, out: torch.Tensor) -> None:
        out.copy_(x * 5)

    direct_register_custom_op(
        "_test_call_hook_cpp_itfs",
        _test_call_hook_cpp_itfs,
        ["out"],
        dispatch_key="CPU",
    )
    calls, hook = _recorder()
    register_call_hook(hook)
    try:
        out = torch.empty(3)
        torch.ops.aiter._test_call_hook_cpp_itfs(torch.ones(3), out)
    finally:
        remove_call_hook(hook)
    assert torch.equal(out, torch.full((3,), 5.0))
    assert calls == [("_test_call_hook_cpp_itfs", 0, (3,))]


def test_observe_shows_a_plain_entry_point_with_its_depth():
    """`observe` makes a launcher no custom op registers -- a Triton entry --
    visible, nested like a registered op."""
    from aiter.jit.utils.call_hook import observe

    @observe
    def _test_observed_launcher(x):
        return _test_call_hook_inner(x)

    @observe(name="renamed_entry")
    def _test_observed_other(x):
        return x

    calls, hook = _recorder()
    register_call_hook(hook)
    try:
        _test_observed_launcher(torch.ones(2, 3))
        _test_observed_other(torch.ones(4))
    finally:
        remove_call_hook(hook)
    assert calls == [
        ("_test_observed_launcher", 0, (2, 3)),
        ("_test_call_hook_inner", 1, (2, 3)),
        ("renamed_entry", 0, (4,)),
    ]
    assert _test_observed_launcher.__name__ == "_test_observed_launcher"

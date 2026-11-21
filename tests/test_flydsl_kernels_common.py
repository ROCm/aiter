"""CPU-only tests for shared FlyDSL kernel dispatch helpers."""

from unittest import mock

import pytest

pytest.importorskip("flydsl")
kernels_common = pytest.importorskip("aiter.ops.flydsl.kernels.kernels_common")


class _Param:
    def __init__(self, value):
        self.value = value

    def __cache_signature__(self):
        return (_Param, self.value)


def test_run_cached_compiles_once_and_keeps_jit_function_caches_separate():
    jit_a, jit_b = object(), object()
    compiled_a, compiled_b = mock.Mock(), mock.Mock()
    compiler = mock.Mock(side_effect=(compiled_a, compiled_b))

    try:
        first = kernels_common.run_cached(
            jit_a,
            "compile-a",
            constexpr_param=_Param(7),
            compiler=compiler,
            dispatch_args=("dispatch-a",),
        )
        second = kernels_common.run_cached(
            jit_a,
            "unused",
            constexpr_param=_Param(7),
            compiler=compiler,
            dispatch_args=("dispatch-a",),
        )
        other_jit = kernels_common.run_cached(
            jit_b,
            "compile-b",
            constexpr_param=_Param(7),
            compiler=compiler,
            dispatch_args=("dispatch-b",),
        )
    finally:
        kernels_common._compiled_caches.pop(jit_a, None)
        kernels_common._compiled_caches.pop(jit_b, None)

    assert first is compiled_a
    assert second is compiled_a
    assert other_jit is compiled_b
    assert compiler.call_args_list == [
        mock.call(jit_a, "compile-a"),
        mock.call(jit_b, "compile-b"),
    ]
    compiled_a.assert_called_once_with("dispatch-a")
    compiled_b.assert_not_called()

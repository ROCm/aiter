# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU tests for PA specialization values in FlyDSL's closure cache key."""

from dataclasses import dataclass, replace
from pathlib import Path
from types import CellType, CodeType, FunctionType

import pytest


@dataclass(frozen=True)
class _Traits:
    head_dim: int = 128
    query_splits: int = 1

    @property
    def cache_key(self):
        return self.head_dim, self.query_splits


@pytest.fixture(scope="module")
def pa_function_codes():
    path = (
        Path(__file__).resolve().parents[1]
        / "aiter/ops/flydsl/kernels/pa_decode_kernel.py"
    )
    # Compile Python code objects without importing aiter, running decorators,
    # or executing the factory. This preserves the actual PA closure layout.
    pending = [compile(path.read_text(), str(path), "exec")]
    codes = {}
    while pending:
        code = pending.pop()
        codes[code.co_name] = code
        pending.extend(value for value in code.co_consts if isinstance(value, CodeType))
    return codes


@pytest.fixture
def flydsl_cache(monkeypatch):
    module = pytest.importorskip("flydsl.compiler.jit_function")
    # The toolchain/device fingerprint is independent of closure handling.
    # Keep the real dependency walker and cache-key hashing under test.
    monkeypatch.setattr(module, "_flydsl_key", lambda: "pa-closure-cpu-test")
    return module


def _bind_closure(code, traits, tag):
    values = {"traits": traits, "_pa_decode_cache_tag": tag}
    # Use opaque placeholders for unrelated DSL objects; no kernel is executed.
    placeholder = object()
    closure = tuple(
        CellType(values.get(name, placeholder)) for name in code.co_freevars
    )
    return FunctionType(code, {}, closure=closure)


@pytest.mark.parametrize(
    "entry_point", ["_pa_decode_tile_task", "pa_decode_tile_launch"]
)
@pytest.mark.parametrize("field,value", [("head_dim", 256), ("query_splits", 4)])
def test_pa_decode_closure_cache_key(
    pa_function_codes, flydsl_cache, entry_point, field, value
):
    code = pa_function_codes[entry_point]
    traits = _Traits()
    changed = replace(traits, **{field: value})
    tag = (traits.cache_key, "fixed-implementation")
    changed_tag = (changed.cache_key, "fixed-implementation")

    original = _bind_closure(code, traits, tag)
    traits_only = _bind_closure(code, changed, tag)
    specialized = _bind_closure(code, changed, changed_tag)
    same_traits = replace(traits)
    same_values = _bind_closure(
        code, same_traits, (same_traits.cache_key, "fixed-implementation")
    )
    collect = flydsl_cache._collect_closure_scalar_vals
    cache_key = flydsl_cache._jit_function_cache_key

    # Changing fields inside an opaque traits object is currently invisible.
    assert collect(original) == collect(traits_only)
    original_key = cache_key(original)
    assert cache_key(traits_only) == original_key

    # The explicit tuple distinguishes specializations, while equivalent
    # values remain stable across independently created traits objects.
    # Removing PA's `_ = _pa_decode_cache_tag` breaks these assertions.
    assert collect(specialized) != collect(original)
    assert cache_key(specialized) != original_key
    assert cache_key(same_values) == original_key

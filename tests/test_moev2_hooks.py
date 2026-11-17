# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 test hooks (steps doc 1.5)."""

import pytest

from aiter.moev2 import hooks
from aiter.moev2.dispatch import tuned


def test_kernel_bench_callable_collects_and_restores():
    hooks.record_kernel("stage1", lambda: None)  # no collector: ignored
    with hooks.set_kernel_bench_callable() as calls:
        hooks.record_kernel("stage1", "k1")
        with hooks.set_kernel_bench_callable() as inner:
            hooks.record_kernel("stage2", "k2")
        hooks.record_kernel("stage2", "k3")
    assert calls == [("stage1", "k1"), ("stage2", "k3")]
    assert inner == [("stage2", "k2")]
    assert hooks._kernel_calls is None


def test_sort_impl_override_is_restored_on_error():
    with pytest.raises(RuntimeError), hooks.set_sort_impl("fake"):
        assert hooks.sort_impl("default") == "fake"
        raise RuntimeError
    assert hooks.sort_impl("default") == "default"


def test_clear_cache_clears_registered_caches():
    calls = []

    @tuned.cached
    def lookup(key):
        calls.append(key)
        return key

    try:
        lookup(1)
        lookup(1)
        tuned.clear_cache()
        lookup(1)
        assert calls == [1, 1]
    finally:
        tuned._CACHES.remove(lookup)

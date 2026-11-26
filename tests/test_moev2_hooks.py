# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 test hooks.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

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


def test_clear_cache_recomputes_the_default_parameters(monkeypatch):
    from aiter.moev2.segments import defaults

    try:
        monkeypatch.setenv("AITER_USE_NT", "0")
        monkeypatch.setenv("AITER_KSPLIT", "2")
        monkeypatch.setattr(defaults, "get_cu_num", lambda: 80)
        tuned.clear_cache()
        before = (
            defaults.use_nt(1, 8, 256),
            defaults.get_ksplit(1, 8, 256, 512, 7168),
            defaults.get_block_size_M(256, 8, 8, 512),
        )
        monkeypatch.setenv("AITER_USE_NT", "1")
        monkeypatch.setenv("AITER_KSPLIT", "4")
        monkeypatch.setattr(defaults, "get_cu_num", lambda: 304)
        tuned.clear_cache()
        after = (
            defaults.use_nt(1, 8, 256),
            defaults.get_ksplit(1, 8, 256, 512, 7168),
            defaults.get_block_size_M(256, 8, 8, 512),
        )
        assert before[:2] == (False, 2) and after[:2] == (True, 4)
        assert (before[2], after[2]) == (128, 32)
    finally:
        monkeypatch.undo()
        tuned.clear_cache()

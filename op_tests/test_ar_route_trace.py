# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU checks of the fused all-reduce route trace (``AITER_AR_ROUTE_TRACE``)."""

import importlib
import json

import pytest
import torch

MOD = "aiter.dist.device_communicators.ar_route_trace"


@pytest.fixture
def trace(monkeypatch):
    def load(value):
        monkeypatch.setenv("AITER_AR_ROUTE_TRACE", value)
        mod = importlib.import_module(MOD)
        return importlib.reload(mod)

    yield load
    monkeypatch.delenv("AITER_AR_ROUTE_TRACE", raising=False)
    importlib.reload(importlib.import_module(MOD))


def test_off_by_default_records_nothing(trace):
    mod = trace("0")
    assert not mod.ENABLED
    mod.note("fused_allreduce_rmsnorm", "pull:oneshot", torch.empty(8, 6144))
    assert mod.table() == []


def test_counts_per_key_and_writes_json(trace, tmp_path):
    path = str(tmp_path / "route")
    mod = trace(path)
    x8 = torch.empty(8, 6144, dtype=torch.bfloat16)
    x64 = torch.empty(64, 6144, dtype=torch.bfloat16)
    for _ in range(3):
        mod.note("fused_allreduce_rmsnorm", "pull:oneshot", x8, gemma=True)
    mod.note("fused_allreduce_rmsnorm", "custom_fused_ar_rms_2stage", x64, gemma=True)
    rows = {(r["backend"], r["rows"]): r for r in mod.table()}
    assert rows[("pull:oneshot", 8)]["calls"] == 3
    assert rows[("pull:oneshot", 8)]["bytes"] == 8 * 6144 * 2
    assert rows[("custom_fused_ar_rms_2stage", 64)]["calls"] == 1
    assert rows[("pull:oneshot", 8)]["captured"] is False
    with open(f"{path}.rank0.json") as f:
        written = json.load(f)
    assert {(r["backend"], r["rows"]) for r in written} == set(rows)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

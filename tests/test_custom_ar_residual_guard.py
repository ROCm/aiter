# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""custom_all_reduce_residual fuses only where it was measured no slower than
custom_all_reduce followed by an add, and returns None elsewhere so the caller
runs that pair. No GPU or process group is needed: the communicator is a stub.
"""

import re
from pathlib import Path

import pytest
import torch

from aiter.dist.device_communicators import custom_all_reduce as car
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

DTYPES = [torch.bfloat16, torch.float16, torch.float32]
BOUNDED = [k for k, v in car._AR_RESIDUAL_MAX_BYTES.items() if v is not None]
UNBOUNDED = [k for k, v in car._AR_RESIDUAL_MAX_BYTES.items() if v is None]
CUH = Path(__file__).resolve().parents[1] / "csrc/include/custom_all_reduce.cuh"


def _make(monkeypatch, gfx, world_size):
    monkeypatch.setattr(car, "_runtime_gfx", lambda: gfx)
    ca = CustomAllreduce.__new__(CustomAllreduce)
    ca.disabled = False
    ca._ptr = 0  # close() disposes nothing
    ca.world_size = world_size
    ca._IS_CAPTURING = False
    ca.calls = []
    ca._ops_all_reduce_residual = lambda *args: ca.calls.append(args)
    ca.should_custom_ar = lambda inp: True
    ca.all_reduce_residual = lambda inp, residual, **kw: ca.calls.append(kw) or inp
    return ca


def _inp(nbytes, dtype):
    itemsize = torch.empty((), dtype=dtype).element_size()
    assert nbytes % itemsize == 0
    # meta: the guard reads only the shape and dtype.
    return torch.empty(nbytes // itemsize, dtype=dtype, device="meta")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("key", BOUNDED)
def test_bound_is_inclusive_and_in_bytes(monkeypatch, key, dtype):
    ca = _make(monkeypatch, *key)
    bound = car._AR_RESIDUAL_MAX_BYTES[key]
    assert ca.should_fuse_ar_residual(_inp(bound, dtype))
    assert not ca.should_fuse_ar_residual(_inp(bound + 16, dtype))


@pytest.mark.parametrize("key", UNBOUNDED)
def test_unbounded_pairs_fuse_up_to_the_custom_ar_cap(monkeypatch, key):
    ca = _make(monkeypatch, *key)
    assert ca.should_fuse_ar_residual(_inp(car._DEFAULT_CAR_MAX_SIZE, torch.bfloat16))


@pytest.mark.parametrize(
    "key",
    [("gfx942", 3), ("gfx90a", 8), ("gfx1250", 8), (None, 8)],
    ids=["gfx942-tp3", "gfx90a-tp8", "gfx1250-tp8", "unknown-tp8"],
)
def test_unmeasured_pairs_never_fuse(monkeypatch, key):
    assert key not in car._AR_RESIDUAL_MAX_BYTES
    ca = _make(monkeypatch, *key)
    assert not ca.should_fuse_ar_residual(_inp(16, torch.bfloat16))


def test_declined_call_returns_none_without_running_the_kernel(monkeypatch):
    key = BOUNDED[0]
    ca = _make(monkeypatch, *key)
    bound = car._AR_RESIDUAL_MAX_BYTES[key]
    small, large = _inp(bound, torch.bfloat16), _inp(bound + 16, torch.bfloat16)
    assert ca.custom_all_reduce_residual(large, large) is None
    assert not ca.calls
    assert ca.custom_all_reduce_residual(small, small) is small
    assert ca.calls == [{"registered_input": False}]


@pytest.mark.parametrize("fn", ["allreduce", "allreduce_residual"])
def test_all_reduce_and_residual_share_the_write_mode_switch(fn):
    """Both paths take write mode at the same size, as measured."""
    src = CUH.read_text()
    body = re.search(rf"\bvoid {fn}\(.*?\n}}\n", src, re.DOTALL)
    assert body, f"{fn} not found in {CUH}"
    assert "select_ar_write_mode(bytes)" in body.group(0)
    assert src.count("512 * 4096 * 2") == 1, "write-mode threshold duplicated"


def test_registered_capture_checks_that_out_can_be_exported(monkeypatch):
    """Write mode exports out at flush, so a caller's out is checked as in all_reduce."""
    ca = CustomAllreduce.__new__(CustomAllreduce)
    ca.disabled = True
    ca._ptr = 0
    checked = []
    ca._capture_registered = lambda t: checked.append(t) or True
    ca._ops_all_reduce_residual = lambda *args: None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    inp = torch.zeros(4, 8, dtype=torch.bfloat16)
    out = torch.empty_like(inp)
    ca.all_reduce_residual(inp, inp, out=out, registered_input=True)
    assert len(checked) == 1 and checked[0] is out

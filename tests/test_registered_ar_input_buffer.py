# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only ownership and dispatch tests for registered AR input leases."""

from types import SimpleNamespace

import pytest
import torch

from aiter.dist import communication_op
from aiter.dist.device_communicators.communicator_cuda import CudaCommunicator
from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


class _Entry:
    def __init__(self, nbytes: int):
        self.tensor = torch.empty(nbytes, dtype=torch.uint8)

    @property
    def data_ptr(self):
        return self.tensor.data_ptr()

    @property
    def max_size(self):
        return self.tensor.numel()


class _Pool:
    def __init__(self, nbytes: int):
        self.entry = _Entry(nbytes)
        self.closed = 0

    def __getitem__(self, key):
        assert key == "input"
        return self.entry

    def close(self):
        self.closed += 1


def _communicator(nbytes=4096):
    ca = CustomAllreduce.__new__(CustomAllreduce)
    ca.disabled = False
    ca._is_gfx1250 = True
    ca.device = torch.device("cpu")
    ca.max_size = nbytes
    ca._pool = _Pool(nbytes)
    ca._ptr = 0
    ca._closed = False
    ca._registered_input_lease_generation = 0
    ca._registered_input_lease_active = False
    ca._registered_input_tensor = None
    ca._registered_input_array_interface = None
    ca._deferred_pool = None
    ca._IS_CAPTURING = False
    return ca


def test_lease_is_exclusive_owner_backed_and_releasable():
    ca = _communicator()
    first = ca.lease_registered_input_buffer()
    assert first is not None
    assert ca.lease_registered_input_buffer() is None

    like = torch.empty((7, 16), dtype=torch.bfloat16)
    view = first.view_like(like)
    assert view.shape == like.shape and view.dtype == like.dtype
    assert view.data_ptr() == ca._pool["input"].data_ptr
    assert ca.is_registered_input_buffer(view)

    first.release()
    with pytest.raises(RuntimeError, match="released"):
        _ = first.buffer
    second = ca.lease_registered_input_buffer()
    assert second is not None
    second.release()


def test_nonzero_offset_and_oversize_views_are_not_registered():
    ca = _communicator(256)
    lease = ca.lease_registered_input_buffer()
    assert lease is not None
    assert not ca.is_registered_input_buffer(lease.buffer[16:])
    with pytest.raises(ValueError, match="needs"):
        lease.view_like(torch.empty(257, dtype=torch.uint8))
    lease.release()


def test_close_defers_raw_pool_lifetime_until_lease_release():
    ca = _communicator()
    lease = ca.lease_registered_input_buffer()
    pool = ca._pool
    ca.close()
    assert pool.closed == 0
    assert ca._deferred_pool is pool
    lease.release()
    assert pool.closed == 1
    assert ca._deferred_pool is None


@pytest.mark.parametrize("leased", [False, True])
def test_fused_ar_marks_only_the_arena_view_registered(leased):
    ca = _communicator()
    seen = []
    ca.should_custom_ar = lambda _inp: True
    ca.fused_ar_rms = lambda *args, **kwargs: seen.append(kwargs["registered"]) or (
        args[0],
        args[1],
    )
    inp = torch.empty((4, 8), dtype=torch.bfloat16)
    if leased:
        lease = ca.lease_registered_input_buffer()
        inp = lease.view_like(inp)

    ca.custom_fused_ar_rms(
        inp,
        torch.empty_like(inp),
        torch.empty(8, dtype=inp.dtype),
        1e-6,
        use_1stage=True,
    )
    assert seen == [leased]
    if leased:
        lease.release()


def test_public_api_delegates_without_exposing_a_pointer(monkeypatch):
    token = object()
    group = SimpleNamespace(lease_registered_input_buffer=lambda: token)
    monkeypatch.setattr(communication_op, "get_tp_group", lambda: group)
    monkeypatch.setattr(communication_op, "has_custom_group", lambda: False)
    result = communication_op.tensor_model_parallel_lease_registered_input_buffer()
    assert result is token


def test_cuda_communicator_capability_miss_and_delegation():
    comm = CudaCommunicator.__new__(CudaCommunicator)
    comm.ca_comm = None
    assert comm.lease_registered_input_buffer() is None
    token = object()
    comm.ca_comm = SimpleNamespace(
        disabled=False, lease_registered_input_buffer=lambda: token
    )
    assert comm.lease_registered_input_buffer() is token

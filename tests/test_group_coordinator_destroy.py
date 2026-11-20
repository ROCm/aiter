# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`GroupCoordinator.destroy` releases its device communicator while its
process groups are still alive, so communicators that free IPC buffers shared
with the peers can still agree with them; the FlyDSL symmetric heap warns when
it is closed after its group is gone. CPU only: the process groups and the
communicator are stand-ins."""

from types import SimpleNamespace

import pytest
import torch

from aiter.dist import parallel_state


def _group(events, owns=True):
    group = object.__new__(parallel_state.GroupCoordinator)
    group.device_group = "device group"
    group.cpu_group = "cpu group"
    group._owns_device_group = owns
    group._owns_cpu_group = owns
    group._owns_device_communicator = owns
    group.device_communicator = SimpleNamespace(
        destroy=lambda: events.append(("communicator", set(alive)))
    )
    group.mq_broadcaster = None
    alive = {"device group", "cpu group"}
    group._alive = alive
    return group


def _record_pg_destroy(monkeypatch, group, events):
    def destroy_process_group(pg):
        group._alive.discard(pg)
        events.append(("process group", pg))

    monkeypatch.setattr(
        torch.distributed, "destroy_process_group", destroy_process_group
    )


def test_device_communicator_goes_before_the_process_groups(monkeypatch):
    events = []
    group = _group(events)
    _record_pg_destroy(monkeypatch, group, events)
    group.destroy()
    assert events == [
        ("communicator", {"device group", "cpu group"}),
        ("process group", "device group"),
        ("process group", "cpu group"),
    ]
    assert group.device_communicator is None
    assert not hasattr(group, "device_group") and not hasattr(group, "cpu_group")


def test_borrowed_communicator_and_groups_are_dropped_not_destroyed(monkeypatch):
    events = []
    group = _group(events, owns=False)
    _record_pg_destroy(monkeypatch, group, events)
    group.destroy()
    assert events == []
    assert group.device_communicator is None


def test_heap_closed_after_its_group_warns(monkeypatch):
    pytest.importorskip("flydsl")
    from aiter.ops.flydsl import ipc_symm_heap

    heap = object.__new__(ipc_symm_heap.IpcSymmetricAllocator)
    heap.group = "gone"
    heap._bufs = [SimpleNamespace(bytes=torch.empty(4096, dtype=torch.uint8))] * 2
    warnings = []
    monkeypatch.setattr(ipc_symm_heap, "_group_alive", lambda g: False)
    monkeypatch.setattr(
        ipc_symm_heap,
        "logger",
        SimpleNamespace(warning=lambda msg, *args: warnings.append(msg % args)),
    )
    heap.close()
    assert heap._bufs == []
    assert len(warnings) == 1 and "2 IPC buffer(s) (8192 bytes)" in warnings[0]

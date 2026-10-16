# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
from types import ModuleType
from typing import ClassVar

import pytest

from aiter.dist.device_communicators import all2all
from aiter.dist.device_communicators.all2all import MoriV2All2AllManager
from aiter.dist.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)


class FakeCcoCommunicator:
    unique_id_calls = 0
    init_calls: ClassVar[list[tuple]] = []
    instances: ClassVar[list["FakeCcoCommunicator"]] = []
    fail_barrier = False

    def __init__(self):
        self.events = []
        self.destroyed = False

    @classmethod
    def reset(cls):
        cls.unique_id_calls = 0
        cls.init_calls = []
        cls.instances = []
        cls.fail_barrier = False

    @classmethod
    def get_unique_id(cls):
        cls.unique_id_calls += 1
        return "root-uid"

    @classmethod
    def init(cls, world_size, rank, uid, *, per_rank_vmm):
        cls.init_calls.append((world_size, rank, uid, per_rank_vmm))
        comm = cls()
        cls.instances.append(comm)
        return comm

    def barrier(self):
        self.events.append("barrier")
        if type(self).fail_barrier:
            raise RuntimeError("barrier failed")

    def destroy(self):
        self.events.append("destroy")
        self.destroyed = True


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeOp:
    events: ClassVar[list[str]] = []

    def __init__(self, config, comm):
        self.config = config
        self.comm = comm

    def close(self):
        self.events.append("handle-close")


@pytest.fixture
def manager(monkeypatch):
    FakeCcoCommunicator.reset()
    FakeOp.events = []

    def init_base(self, cpu_group):
        self.cpu_group = cpu_group
        self.rank = 0
        self.world_size = 4
        self.internode = False

    monkeypatch.setattr(All2AllManagerBase, "__init__", init_base)
    monkeypatch.setattr(all2all, "has_mori", lambda: True)

    mori_module = ModuleType("mori")
    mori_module.__path__ = []
    cco_module = ModuleType("mori.cco")
    cco_module.Communicator = FakeCcoCommunicator
    ops_module = ModuleType("mori.ops")
    ops_module.__path__ = []
    v2_module = ModuleType("mori.ops.dispatch_combine_v2")
    v2_module.EpDispatchCombineConfig = FakeConfig
    v2_module.EpDispatchCombineOp = FakeOp
    mori_module.cco = cco_module
    mori_module.ops = ops_module
    ops_module.dispatch_combine_v2 = v2_module
    monkeypatch.setitem(sys.modules, "mori", mori_module)
    monkeypatch.setitem(sys.modules, "mori.cco", cco_module)
    monkeypatch.setitem(sys.modules, "mori.ops", ops_module)
    monkeypatch.setitem(sys.modules, "mori.ops.dispatch_combine_v2", v2_module)
    monkeypatch.setattr(
        all2all.torch.distributed,
        "get_global_rank",
        lambda group, group_rank: 17,
    )
    return MoriV2All2AllManager(cpu_group="ep-cpu-group")


def test_cco_communicator_is_created_collectively_and_reused(manager, monkeypatch):
    broadcasts = []

    def broadcast(objects, *, src, group):
        broadcasts.append((list(objects), src, group))

    monkeypatch.setattr(
        all2all.torch.distributed,
        "broadcast_object_list",
        broadcast,
    )

    comm = manager.get_or_create_cco_communicator(4096)

    assert manager.get_or_create_cco_communicator(2048) is comm
    assert FakeCcoCommunicator.unique_id_calls == 1
    assert FakeCcoCommunicator.init_calls == [(4, 0, "root-uid", 4096)]
    assert broadcasts == [(["root-uid"], 17, "ep-cpu-group")]
    assert comm.events == ["barrier"]


def test_non_root_receives_the_broadcast_unique_id(manager, monkeypatch):
    manager.rank = 2

    def broadcast(objects, *, src, group):
        assert objects == [None]
        assert src == 17
        assert group == "ep-cpu-group"
        objects[0] = "broadcast-uid"

    monkeypatch.setattr(
        all2all.torch.distributed,
        "broadcast_object_list",
        broadcast,
    )

    manager.get_or_create_cco_communicator(1024)

    assert FakeCcoCommunicator.unique_id_calls == 0
    assert FakeCcoCommunicator.init_calls == [(4, 2, "broadcast-uid", 1024)]


def test_cco_communicator_cannot_grow_after_creation(manager, monkeypatch):
    monkeypatch.setattr(
        all2all.torch.distributed,
        "broadcast_object_list",
        lambda objects, **kwargs: None,
    )
    manager.get_or_create_cco_communicator(4096)

    with pytest.raises(RuntimeError, match="cannot satisfy the requested 8192"):
        manager.get_or_create_cco_communicator(8192)


@pytest.mark.parametrize("invalid", [0, -1])
def test_cco_communicator_rejects_non_positive_capacity(manager, invalid):
    with pytest.raises(ValueError, match="must be > 0"):
        manager.get_or_create_cco_communicator(invalid)


@pytest.mark.parametrize("invalid", [True, 1.5, "4096"])
def test_cco_communicator_rejects_non_integer_capacity(manager, invalid):
    with pytest.raises(TypeError, match="must be an integer"):
        manager.get_or_create_cco_communicator(invalid)


def test_failed_collective_initialization_is_not_published(manager, monkeypatch):
    monkeypatch.setattr(
        all2all.torch.distributed,
        "broadcast_object_list",
        lambda objects, **kwargs: None,
    )
    FakeCcoCommunicator.fail_barrier = True

    with pytest.raises(RuntimeError, match="barrier failed"):
        manager.get_or_create_cco_communicator(4096)

    failed = FakeCcoCommunicator.instances[0]
    assert failed.destroyed
    assert manager._cco_communicator is None
    assert manager._cco_per_rank_vmm == 0

    FakeCcoCommunicator.fail_barrier = False
    assert manager.get_or_create_cco_communicator(4096) is not failed
    assert len(FakeCcoCommunicator.init_calls) == 2


def test_get_handle_injects_group_geometry_and_caches_op(manager, monkeypatch):
    monkeypatch.setattr(
        all2all.torch.distributed,
        "broadcast_object_list",
        lambda objects, **kwargs: None,
    )
    args = {
        "per_rank_vmm": 4096,
        "hidden_dim": 7168,
        "max_num_inp_token_per_rank": 128,
        "num_experts_per_rank": 8,
        "num_experts_per_token": 2,
    }

    first = manager.get_handle(args)
    same = manager.get_handle(args)
    second_slot = manager.get_handle(args, index=1)

    assert same is first
    assert second_slot is not first
    assert first.comm is manager._cco_communicator
    assert first.config.kwargs == {
        "hidden_dim": 7168,
        "max_num_inp_token_per_rank": 128,
        "num_experts_per_rank": 8,
        "num_experts_per_token": 2,
        "rank": 0,
        "world_size": 4,
    }


def test_get_handle_rejects_missing_capacity(manager):
    with pytest.raises(ValueError, match="must include per_rank_vmm"):
        manager.get_handle({"hidden_dim": 7168})


@pytest.mark.parametrize(
    ("name", "value"),
    [("rank", 1), ("world_size", 8)],
)
def test_get_handle_rejects_group_geometry_mismatch(manager, name, value):
    args = {
        "per_rank_vmm": 4096,
        "hidden_dim": 7168,
        name: value,
    }

    with pytest.raises(ValueError, match="does not match the EP group"):
        manager.get_handle(args)

    assert manager._cco_communicator is None


def test_destroy_closes_handles_before_communicator(manager, monkeypatch):
    events = []

    class OrderedOp(FakeOp):
        def close(self):
            events.append("handle-close")

    monkeypatch.setattr(
        sys.modules["mori.ops.dispatch_combine_v2"],
        "EpDispatchCombineOp",
        OrderedOp,
    )
    monkeypatch.setattr(
        all2all.torch.distributed,
        "broadcast_object_list",
        lambda objects, **kwargs: None,
    )

    handle = manager.get_handle(
        {
            "per_rank_vmm": 4096,
            "hidden_dim": 7168,
            "max_num_inp_token_per_rank": 128,
            "num_experts_per_rank": 8,
            "num_experts_per_token": 2,
        }
    )
    comm = manager._cco_communicator
    original_destroy = comm.destroy

    def destroy():
        events.append("comm-destroy")
        original_destroy()

    comm.destroy = destroy
    manager.destroy()

    assert handle is not None  # Keep the weakly cached handle live through destroy.
    assert events == ["handle-close", "comm-destroy"]
    assert manager._cco_communicator is None
    assert manager._cco_per_rank_vmm == 0

    manager.destroy()
    assert events == ["handle-close", "comm-destroy"]

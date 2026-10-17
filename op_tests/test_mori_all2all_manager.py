# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging
import sys
from enum import Enum
from types import ModuleType, SimpleNamespace
from typing import ClassVar

from aiter.dist.device_communicators.all2all import MoriAll2AllManager
from aiter.dist.device_communicators.base_device_communicator import Cache


class FakeKernelType(Enum):
    IntraNode = 1
    AsyncLL = 2


class FakeHandle:
    instances: ClassVar[list["FakeHandle"]] = []

    def __init__(self, config):
        self.config = config
        self.instances.append(self)


def test_mori_handle_creation_logs_only_on_cache_miss(caplog, monkeypatch):
    manager = MoriAll2AllManager.__new__(MoriAll2AllManager)
    manager.handle_cache = Cache()
    manager.internode = False
    FakeHandle.instances = []

    monkeypatch.setattr(manager, "_make_all2all_kwargs", lambda **kwargs: kwargs)
    mori = ModuleType("mori")
    mori.ops = SimpleNamespace(
        EpDispatchCombineConfig=lambda **kwargs: kwargs,
        EpDispatchCombineOp=FakeHandle,
    )
    monkeypatch.setitem(sys.modules, "mori", mori)
    caplog.set_level(logging.INFO, logger="aiter")

    intranode = {"kernel_type": FakeKernelType.IntraNode, "low_latency": False}
    first = manager.get_handle(intranode)
    assert manager.get_handle(intranode) is first
    assert manager.get_handle(intranode, index=1) is not first

    async_ll = {"kernel_type": FakeKernelType.AsyncLL, "low_latency": True}
    manager.get_handle(async_ll)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("[MORI] EP all2all")
    ]
    assert messages == [
        "[MORI] EP all2all handle created: kernel=IntraNode internode=False",
        "[MORI] EP all2all handle created: kernel=IntraNode internode=False",
        "[MORI] EP all2all handle created: kernel=AsyncLL internode=False",
    ]
    assert len(FakeHandle.instances) == 3

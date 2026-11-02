"""`GroupCoordinator.all_gather` reads the custom all-reduce only when asked to
use it, and takes the custom kernel only where it has one for the group size.
CPU only: the collective and the custom op are stand-ins."""

from types import SimpleNamespace

import torch

from aiter.dist import parallel_state


def _group(world_size, device_communicator):
    group = object.__new__(parallel_state.GroupCoordinator)
    group.world_size = world_size
    group.device_group = "device group"
    group.unique_name = "group"
    group.device_communicator = device_communicator
    return group


def _record_paths(monkeypatch):
    paths = []

    def nccl(output, input_, group=None):
        paths.append("nccl")
        output.copy_(input_.expand_as(output))

    def custom(input_, group_name, dim=0):
        paths.append("custom")
        return input_

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", nccl)
    monkeypatch.setattr(parallel_state, "outplace_all_gather", custom)
    return paths


def _ca_comm(has_kernel):
    return SimpleNamespace(_has_gather_scatter_kernel=lambda: has_kernel)


def test_group_without_device_communicator_gathers_through_nccl(monkeypatch):
    """The world group is built without a device communicator."""
    paths = _record_paths(monkeypatch)
    x = torch.arange(6.0).view(3, 2)
    out = _group(2, None).all_gather(x, use_custom=False, dim=0)
    assert out.shape == (6, 2) and torch.equal(out[3:], x)
    assert paths == ["nccl"]


def test_custom_path_only_where_the_kernel_exists(monkeypatch):
    paths = _record_paths(monkeypatch)
    x = torch.arange(6.0).view(3, 2)
    for has_kernel in (True, False):
        dc = SimpleNamespace(ca_comm=_ca_comm(has_kernel))
        _group(6, dc).all_gather(x, use_custom=True, dim=0)
    _group(8, SimpleNamespace(ca_comm=_ca_comm(True))).all_gather(x, dim=0)
    assert paths == ["custom", "nccl", "nccl"]

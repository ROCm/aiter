"""`GroupCoordinator.all_gather` reads the custom all-reduce only when asked to
use it, and takes the custom kernel only where it has one for the group size.
Its NCCL path, `reduce_scatter_tensor(use_custom=False)` and send/recv run on
the device communicator's PyNccl when that is usable and carries the dtype,
and on torch.distributed otherwise and while dynamo traces.
CPU only: the collectives, PyNccl and the custom op are stand-ins."""

from types import SimpleNamespace

import pytest
import torch

from aiter.dist import parallel_state


def _group(world_size, device_communicator):
    group = object.__new__(parallel_state.GroupCoordinator)
    group.world_size = world_size
    group.device_group = "device group"
    group.unique_name = "group"
    group.device_communicator = device_communicator
    group.rank_in_group = 0
    group.ranks = [4 + r for r in range(world_size)]
    group.device = torch.device("cpu")
    return group


def _record_paths(monkeypatch):
    paths = []

    def nccl(output, input_, group=None):
        paths.append("nccl")
        output.view(-1, *input_.shape).copy_(input_)

    def nccl_reduce_scatter(output, input_, group=None):
        paths.append("nccl reduce_scatter")

    def nccl_send(tensor, dst, group=None):
        paths.append(("nccl send", dst))

    def nccl_recv(tensor, src, group=None):
        paths.append(("nccl recv", src))

    def custom(input_, group_name, dim=0):
        paths.append("custom")
        return input_

    monkeypatch.setattr(torch.distributed, "all_gather_into_tensor", nccl)
    monkeypatch.setattr(torch.distributed, "reduce_scatter_tensor", nccl_reduce_scatter)
    monkeypatch.setattr(torch.distributed, "send", nccl_send)
    monkeypatch.setattr(torch.distributed, "recv", nccl_recv)
    monkeypatch.setattr(parallel_state, "outplace_all_gather", custom)
    return paths


def _ca_comm(has_kernel):
    return SimpleNamespace(_has_gather_scatter_kernel=lambda: has_kernel)


class _PyNccl:
    """PyNccl's calls (caller-allocated contiguous buffers, peers by rank in
    the group) for this rank 0 of ranks whose inputs are its own plus the
    rank."""

    def __init__(self, world_size, disabled=False):
        self.world_size, self.disabled, self.calls = world_size, disabled, []

    def all_gather(self, output_tensor, input_tensor):
        self.calls.append("all_gather")
        assert input_tensor.is_contiguous() and output_tensor.is_contiguous()
        flat = input_tensor.reshape(-1)
        parts = [flat + r for r in range(self.world_size)]
        output_tensor.view(-1).copy_(torch.cat(parts))

    def reduce_scatter(self, output_tensor, input_tensor):
        self.calls.append("reduce_scatter")
        assert input_tensor.is_contiguous() and output_tensor.is_contiguous()
        total = sum(input_tensor + r for r in range(self.world_size))
        output_tensor.view(-1).copy_(total.view(self.world_size, -1)[0])

    def send(self, tensor, dst):
        self.calls.append(("send", dst))
        assert tensor.is_contiguous()

    def recv(self, tensor, src):
        self.calls.append(("recv", src))
        tensor.fill_(7)


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
        dc = SimpleNamespace(ca_comm=_ca_comm(has_kernel), pynccl_comm=None)
        _group(6, dc).all_gather(x, use_custom=True, dim=0)
    dc = SimpleNamespace(ca_comm=_ca_comm(True), pynccl_comm=None)
    _group(8, dc).all_gather(x, dim=0)
    assert paths == ["custom", "nccl", "nccl"]


@pytest.mark.parametrize("dim", [0, 1, -1])
def test_nccl_path_runs_on_pynccl(monkeypatch, dim):
    """Each rank's input lands in its slice along `dim`, and only this rank's
    slice of the sum comes back, for a non-contiguous input too."""
    paths = _record_paths(monkeypatch)
    pynccl = _PyNccl(2)
    group = _group(2, SimpleNamespace(ca_comm=None, pynccl_comm=pynccl))
    x = torch.arange(48.0).view(4, 6, 2).transpose(0, 1)
    gathered = group.all_gather(x, use_custom=False, dim=dim)
    assert torch.equal(gathered, torch.cat([x, x + 1], dim))
    scattered = group.reduce_scatter_tensor(x, use_custom=False, dim=dim)
    assert torch.equal(scattered, (2 * x + 1).chunk(2, dim)[0])
    assert pynccl.calls == ["all_gather", "reduce_scatter"] and paths == []


def test_send_recv_run_on_pynccl(monkeypatch):
    """PyNccl addresses the peer by its rank in the group."""
    paths = _record_paths(monkeypatch)
    pynccl = _PyNccl(2)
    group = _group(2, SimpleNamespace(ca_comm=None, pynccl_comm=pynccl))
    group.send(torch.ones(3, 2).t())
    received = group.recv(torch.Size([3]), torch.float32)
    assert torch.equal(received, torch.full((3,), 7.0, device=group.device))
    assert pynccl.calls == [("send", 1), ("recv", 1)] and paths == []


@pytest.mark.parametrize(
    "case",
    ["no device communicator", "no pynccl", "pynccl disabled", "bool", "compiling"],
)
def test_nccl_path_falls_back_to_torch_distributed(monkeypatch, case):
    """Without a usable PyNccl, for a dtype PyNccl does not carry, or while
    dynamo traces, the collectives keep torch.distributed, which takes the
    peer's global rank."""
    paths = _record_paths(monkeypatch)
    pynccl = _PyNccl(2, disabled=case == "pynccl disabled")
    dc = {
        "no device communicator": None,
        "no pynccl": SimpleNamespace(ca_comm=None, pynccl_comm=None),
    }.get(case, SimpleNamespace(ca_comm=None, pynccl_comm=pynccl))
    if case == "compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    dtype = torch.bool if case == "bool" else torch.float32
    group = _group(2, dc)
    x = torch.ones(4, 2, dtype=dtype)
    group.all_gather(x, use_custom=False, dim=0)
    group.reduce_scatter_tensor(x, use_custom=False, dim=0)
    group.send(x)
    group.recv(torch.Size([4, 2]), dtype)
    assert pynccl.calls == []
    assert paths == ["nccl", "nccl reduce_scatter", ("nccl send", 5), ("nccl recv", 5)]

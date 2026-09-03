# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for comm-group reuse (AITER_REUSE_IDENTICAL_COMM_GROUPS).

When several parallel groups span the same set of ranks, they should share one
set of process groups + allreduce communicators while staying distinct
GroupCoordinator objects with correct unique_names -- an EP group keeps its
"ep"-named device_communicator (so is_ep_communicator / use_all2all stay True and
all2all still initializes), but no second communicator set is allocated.

Two independent test surfaces:

1. **Decision logic + bookkeeping (GPU-free, gloo/CPU).** Drive the *real*
   GroupCoordinator over gloo with only the CUDA leaf (CudaCommunicator)
   stubbed, so the reuse path, ownership flags, EP private cpu_group and guarded
   destroy() all execute on any box. They also assert reuse allocates strictly
   fewer process groups and handles than the flag-off baseline. Topologies:
     - all-alias        (tp==dcp==ep over all ranks): DCP/EP reuse TP
     - dp-attention     (tp=1, dp=world): EP reuses **DP**, not TP
     - partial/no-alias (tp=2, dp=2):   DP is its own source, EP aliases nobody
     - flag off:        no group reuses
     - flag disagreement across ranks: the unanimity all_reduce asserts (turns a
       silent new_group() hang into a loud error).

2. **Handle sharing (needs 2 GPUs / NCCL).** Drives the real CudaCommunicator and
   asserts the reusing group shares pynccl/ca/qr handles and process groups, that
   EP keeps an ep-named communicator, and that a collective over the shared comm
   still produces correct values. Skipped when <2 GPUs are visible.

Run:
    pytest op_tests/multigpu_tests/test_reuse_identical_rank_groups.py
    python  op_tests/multigpu_tests/test_reuse_identical_rank_groups.py  # no pytest
"""

import os
import time
from typing import ClassVar

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter.dist import parallel_state as ps
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_dcp_group,
    get_dp_group,
    get_ep_group,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_open_port

_ENV = "AITER_REUSE_IDENTICAL_COMM_GROUPS"

_TOPO_NAMES = ("all_alias", "dp_attention", "partial")


def _dims(topo, world):
    """initialize_model_parallel kwargs for a named topology, scaled to `world`.

    - all_alias:    tp==dcp==ep span every rank        -> DCP/EP reuse TP
    - dp_attention: tp=1, dp=world                      -> EP reuses DP, not TP
    - partial:      tp=2, dp=world//2, dcp=2 (world>=4) -> DP separate, EP aliases none
    """
    if topo == "all_alias":
        return {
            "tensor_model_parallel_size": world,
            "decode_context_model_parallel_size": world,
        }
    if topo == "dp_attention":
        return {"tensor_model_parallel_size": 1, "data_parallel_size": world}
    if topo == "partial":
        return {
            "tensor_model_parallel_size": 2,
            "decode_context_model_parallel_size": 2,
            "data_parallel_size": world // 2,
        }
    raise ValueError(topo)


# --------------------------------------------------------------------------- #
# Decision-logic surface (GPU-free)
# --------------------------------------------------------------------------- #
class _StubCommunicator:
    """GPU-free stand-in for CudaCommunicator.

    Only the CUDA leaf is replaced, so the real GroupCoordinator reuse path,
    ownership flags and guarded destroy() still execute. Mirrors the two
    behaviours parallel_state depends on: the ``"ep" in unique_name`` rule and
    handle inheritance. Constructions are logged in ``instances`` for counting.
    """

    instances: ClassVar[list] = []

    def __init__(
        self,
        cpu_group,
        device=None,
        device_group=None,
        unique_name: str = "",
        reuse_from=None,
    ):
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.device = device
        self.unique_name = unique_name
        self.reuse_from = reuse_from
        self.ranks = dist.get_process_group_ranks(cpu_group)
        self.world_size = len(self.ranks)
        self.rank_in_group = self.ranks.index(dist.get_rank())
        # As DeviceCommunicatorBase derives them.
        self.is_ep_communicator = "ep" in unique_name
        self.use_all2all = self.is_ep_communicator
        if reuse_from is not None:
            self.pynccl_comm = reuse_from.pynccl_comm
            self.ca_comm = reuse_from.ca_comm
            self.qr_comm = reuse_from.qr_comm
        else:
            # Sentinels for the per-group NCCL/CA/QR buffers; distinct ids
            # counted == memory saved.
            self.pynccl_comm = object()
            self.ca_comm = object()
            self.qr_comm = object()
        self.destroyed = False
        type(self).instances.append(self)

    def destroy(self):
        self.destroyed = True


def _patch_cuda_communicator():
    """Swap in _StubCommunicator, reset the log, return the restorer.

    GroupCoordinator imports it inside __init__, so patching the attr suffices.
    """
    from aiter.dist.device_communicators import communicator_cuda as cc

    original = cc.CudaCommunicator
    _StubCommunicator.instances = []
    cc.CudaCommunicator = _StubCommunicator
    return lambda: setattr(cc, "CudaCommunicator", original)


def _count_new_group():
    """Count torch.distributed.new_group calls; return (calls, restorer)."""
    original = torch.distributed.new_group
    calls = []

    def counting(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("ranks"))
        return original(*args, **kwargs)

    torch.distributed.new_group = counting
    return calls, lambda: setattr(torch.distributed, "new_group", original)


def _rankset(g):
    return tuple(sorted(g.ranks))


def _groups():
    return {
        "tp": get_tp_group(),
        "dcp": get_dcp_group(),
        "pcp": ps._PCP,
        "pp": ps._PP,
        "dp": get_dp_group(),
        "ep": get_ep_group(),
    }


def _assert_reuse_invariants(groups):
    """Hold in every topology, independent of the specific rank layout."""
    for name, g in groups.items():
        if g.reuse_from is not None:
            src = g.reuse_from
            # ranks/rank_in_group are inherited verbatim, so the *ordered* rank
            # list must match -- not merely the same set. Asserting ordered
            # equality locks in the tuple(my_ranks) dedup key.
            assert (
                g.ranks == src.ranks
            ), f"{name} reuses a source with a different rank order"
            assert g.rank_in_group == src.rank_in_group
            # ...and single-member groups never reuse (they hold no communicator).
            assert g.world_size > 1, f"{name} is single-rank yet reuses"

            # device_group (the NCCL one, i.e. the memory saved) is always shared.
            assert g.device_group is src.device_group, f"{name} did not share NCCL PG"
            assert not g._owns_device_group, f"{name} claims to own a borrowed PG"

            if name == "ep":
                # mori assumes exclusive use of the EP cpu_group, so it stays
                # private (a gloo PG holds none of the saved buffers).
                assert (
                    g.cpu_group is not src.cpu_group
                ), "EP borrower must not share the source's cpu_group with mori"
                assert g._owns_cpu_group, "EP borrower must own its private cpu_group"
                assert dist.get_process_group_ranks(g.cpu_group) == list(g.ranks)
            else:
                assert g.cpu_group is src.cpu_group, f"{name} did not share cpu_group"
                assert not g._owns_cpu_group, f"{name} claims to own a borrowed PG"

            # The shm ring is single-owner and hard-asserts src == 0.
            assert g.mq_broadcaster is None, f"{name} built a broadcaster while reusing"

            # EP needs its own (ep-named, all2all-capable); others share.
            if name == "ep":
                assert g.device_communicator is not src.device_communicator
                assert g._owns_device_communicator
                assert g.device_communicator.is_ep_communicator
                # ...with no second set of allreduce handles.
                assert (
                    g.device_communicator.pynccl_comm
                    is src.device_communicator.pynccl_comm
                ), "EP borrower allocated a second pynccl comm"
            else:
                assert (
                    g.device_communicator is src.device_communicator
                ), f"{name} allocated a second device communicator"
                assert not g._owns_device_communicator
                assert not g.device_communicator.is_ep_communicator, (
                    f"{name} borrowed an EP communicator; its collectives would "
                    "route through all2all"
                )
        else:
            assert g._owns_cpu_group and g._owns_device_group
        # EP always keeps an ep-named unique_name, reuse or not.
        if name == "ep":
            assert "ep" in g.unique_name, f"EP unique_name lost 'ep': {g.unique_name}"
            if g.device_communicator is not None:
                assert g.device_communicator.use_all2all, "EP lost all2all"


def _decision_worker(rank, world_size, port, topo, reuse):
    restore = []
    try:
        os.environ[_ENV] = "1" if reuse else "0"
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="gloo",
        )
        restore.append(_patch_cuda_communicator())
        initialize_model_parallel(**_dims(topo, world_size))
        g = _groups()

        if not reuse:
            # Flag off: nothing reuses, regardless of topology.
            for name, grp in g.items():
                assert grp.reuse_from is None, f"{name} reused with flag off"
                assert grp._owns_cpu_group and grp._owns_device_group
        else:
            _assert_reuse_invariants(g)
            if topo == "all_alias":
                assert g["tp"].reuse_from is None, "TP must be the source"
                assert g["dcp"].reuse_from is g["tp"], "DCP must reuse TP"
                assert g["ep"].reuse_from is g["tp"], "EP must reuse TP"
                for s in ("pp", "pcp", "dp"):
                    assert g[s].world_size == 1 and g[s].reuse_from is None
            elif topo == "dp_attention":
                # TP is a singleton here; the shared comm belongs to DP, and EP
                # must reuse DP -- the case that silently broke on the ATOM side.
                assert g["dp"].reuse_from is None, "DP must be the source"
                assert g["dp"].world_size == world_size
                assert g["ep"].reuse_from is g["dp"], "EP must reuse DP, not TP"
                assert g["tp"].world_size == 1 and g["tp"].reuse_from is None
            elif topo == "partial":
                assert g["tp"].reuse_from is None and g["tp"].world_size == 2
                assert g["dcp"].reuse_from is g["tp"], "DCP must reuse TP"
                # DP spans a different rank set than TP -> its own source.
                assert g["dp"].reuse_from is None, "DP must be a separate source"
                assert _rankset(g["dp"]) != _rankset(g["tp"])
                # EP spans all ranks, matching no earlier group -> aliases nobody.
                assert (
                    g["ep"].reuse_from is None
                ), "EP matches no prior rank set; must not reuse"
                assert g["ep"].world_size == world_size

        # Real guarded teardown: a double destroy_process_group() would raise,
        # and borrowers must drop their reference to the source's communicator.
        borrowers = [grp for grp in g.values() if grp.reuse_from is not None]
        destroy_model_parallel()
        for grp in borrowers:
            assert grp.device_communicator is None, "borrower kept a dead communicator"
    finally:
        for undo in reversed(restore):
            undo()
        if dist.is_initialized():
            destroy_distributed_environment()


def _saving_worker(rank, world_size, port, topo):
    """A/B the same topology in one process and assert reuse allocates less."""
    restore = []
    try:
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="gloo",
        )
        restore.append(_patch_cuda_communicator())

        counts = {}
        for reuse in (False, True):
            os.environ[_ENV] = "1" if reuse else "0"
            _StubCommunicator.instances = []
            calls, undo_count = _count_new_group()
            try:
                initialize_model_parallel(**_dims(topo, world_size))
                comms = list(_StubCommunicator.instances)
                # Distinct handles == buffers actually allocated.
                counts[reuse] = (
                    len(calls),
                    len(comms),
                    len({id(c.pynccl_comm) for c in comms}),
                )
            finally:
                undo_count()
                destroy_model_parallel()

        off_pg, off_comm, off_handles = counts[False]
        on_pg, on_comm, on_handles = counts[True]
        assert on_pg < off_pg, f"{topo}: no process groups saved ({on_pg} vs {off_pg})"
        assert (
            on_handles < off_handles
        ), f"{topo}: no allreduce handles saved ({on_handles} vs {off_handles})"
        assert on_comm <= off_comm
    finally:
        for undo in reversed(restore):
            undo()
        if dist.is_initialized():
            destroy_distributed_environment()


def _unanimity_worker(rank, world_size, port):
    """rank 0 sets the flag off, the rest on -> the unanimity guard must fire."""
    try:
        os.environ[_ENV] = "0" if rank == 0 else "1"
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="gloo",
        )
        # RuntimeError, not assert (`python -O` strips those). Match the message
        # so an unrelated failure cannot pass this test.
        with pytest.raises(RuntimeError, match=_ENV):
            initialize_model_parallel(tensor_model_parallel_size=world_size)
    finally:
        if dist.is_initialized():
            destroy_distributed_environment()


def _spawn(worker, world_size, *args, timeout=180):
    # Fresh port per spawn -> no "address in use" across back-to-back tests.
    port = get_open_port()
    # Explicit timeout: a reuse bug here hangs (ranks disagreeing on whether to
    # call new_group), which would otherwise wedge CI until the job timeout.
    ctx = mp.spawn(
        worker, args=(world_size, port, *args), nprocs=world_size, join=False
    )
    deadline = time.monotonic() + timeout
    while not ctx.join(timeout=1):
        if time.monotonic() > deadline:
            for proc in ctx.processes:
                if proc.is_alive():
                    proc.terminate()
            raise TimeoutError(
                f"{worker.__name__} did not finish within {timeout}s "
                "(ranks likely disagree on which process groups to create)"
            )


@pytest.mark.parametrize("topo", _TOPO_NAMES)
@pytest.mark.parametrize("reuse", [True, False])
def test_reuse_decision(topo, reuse):
    """GPU-free: the real GroupCoordinator picks the right source per topology."""
    _spawn(_decision_worker, 4, topo, reuse)


@pytest.mark.parametrize("topo", _TOPO_NAMES)
def test_reuse_saves_process_groups_and_handles(topo):
    """GPU-free: reuse must actually allocate less -- the point of the PR."""
    _spawn(_saving_worker, 4, topo)


def test_reuse_flag_disagreement_asserts():
    """GPU-free: a per-rank flag mismatch is caught, not a silent hang."""
    _spawn(_unanimity_worker, 4)


# --------------------------------------------------------------------------- #
# Handle-sharing surface (needs 2 GPUs / NCCL)
# --------------------------------------------------------------------------- #
def _gpu_init(rank, world_size, port, topo, reuse):
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        local_rank=rank,
        backend="nccl",
    )
    os.environ[_ENV] = "1" if reuse else "0"
    initialize_model_parallel(**_dims(topo, world_size))
    # ca_comm is checked by identity only; the functional check below runs over
    # pynccl_comm. Driving CA would need init_dist_env's full buffer handshake.


def _gpu_teardown():
    destroy_model_parallel()
    destroy_distributed_environment()
    torch.cuda.empty_cache()


def _assert_shares(a, b):
    """b reuses a: distinct coordinators, shared underlying comm handles."""
    assert a is not b
    assert a.device_group is b.device_group
    assert a.cpu_group is b.cpu_group
    ca, cb = a.device_communicator, b.device_communicator
    assert ca.pynccl_comm is cb.pynccl_comm
    assert ca.ca_comm is cb.ca_comm
    assert ca.qr_comm is cb.qr_comm


def _gpu_worker(rank, world_size, port, topo, reuse):
    try:
        _gpu_init(rank, world_size, port, topo, reuse)
        tp, ep, dcp, dp = (
            get_tp_group(),
            get_ep_group(),
            get_dcp_group(),
            get_dp_group(),
        )

        if not reuse:
            # Baseline: every identical-rank group owns its communicators.
            assert ep.device_communicator is not tp.device_communicator
            assert (
                ep.device_communicator.pynccl_comm
                is not tp.device_communicator.pynccl_comm
            )
            assert ep.device_communicator.is_ep_communicator is True
            _gpu_teardown()
            if rank == 0:
                print(f"[gpu:{topo}:noreuse] PASSED")
            return

        # EP is always distinct with an ep-named comm so all2all can initialize,
        # but it reuses its source's allreduce handles (the whole point).
        assert ep is not tp
        assert "ep" in ep.unique_name
        assert ep.device_communicator is not tp.device_communicator
        assert ep.device_communicator.is_ep_communicator is True
        assert ep.device_communicator.use_all2all is True

        # The source EP reuses depends on topology: TP in the all-alias case,
        # DP in the dp-attention case (TP is a singleton there).
        source = dp if topo == "dp_attention" else tp
        assert (
            ep.device_communicator.pynccl_comm is source.device_communicator.pynccl_comm
        )
        assert ep.device_communicator.ca_comm is source.device_communicator.ca_comm
        assert ep.device_communicator.qr_comm is source.device_communicator.qr_comm
        assert ep.device_group is source.device_group
        # ...but cpu_group stays private for mori (see the GPU-free suite).
        assert ep.cpu_group is not source.cpu_group
        assert dist.get_process_group_ranks(ep.cpu_group) == list(source.ranks)

        if topo == "all_alias":
            # DCP (non-EP) shares TP's device_communicator wholesale.
            assert dcp.device_communicator is tp.device_communicator
            _assert_shares(tp, dcp)

        # A collective over the reused comm must still be correct. pynccl is the
        # shared handle; exercise it through EP's distinct coordinator.
        dev = f"cuda:{rank}"
        t = torch.ones(8, device=dev)
        out = ep.device_communicator.pynccl_comm.all_reduce(t)
        torch.cuda.synchronize()
        assert torch.allclose(
            out, torch.full_like(out, float(source.world_size))
        ), f"reused all_reduce gave {out[0].item()}, expected {source.world_size}"

        _gpu_teardown()
        if rank == 0:
            print(f"[gpu:{topo}:reuse] PASSED")
    except Exception:
        _gpu_teardown()
        raise


_NEED_2_GPU = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs / NCCL"
)


@_NEED_2_GPU
@pytest.mark.parametrize("topo", ["all_alias", "dp_attention"])
def test_gpu_reuse_shares_handles(topo):
    _spawn(_gpu_worker, 2, topo, True)


@_NEED_2_GPU
def test_gpu_reuse_off_baseline():
    _spawn(_gpu_worker, 2, "all_alias", False)


def main():
    # Manual entry point (no pytest): run the GPU-free decision tests always,
    # the GPU tests only when 2 GPUs are visible.
    for reuse in (True, False):
        for topo in _TOPO_NAMES:
            _spawn(_decision_worker, 4, topo, reuse)
    for topo in _TOPO_NAMES:
        _spawn(_saving_worker, 4, topo)
    _spawn(_unanimity_worker, 4)
    print("decision-logic tests: PASSED")

    if torch.cuda.device_count() >= 2:
        for topo in ("all_alias", "dp_attention"):
            _spawn(_gpu_worker, 2, topo, True)
        _spawn(_gpu_worker, 2, "all_alias", False)
        print("gpu handle-sharing tests: PASSED")
    else:
        print(f"SKIP gpu tests: need 2 GPUs, have {torch.cuda.device_count()}")


if __name__ == "__main__":
    main()

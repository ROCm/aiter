# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for comm-group reuse (AITER_REUSE_IDENTICAL_COMM_GROUPS).

When several parallel groups span the same set of ranks, they should share one
set of process groups + allreduce communicators while staying distinct
GroupCoordinator objects with correct unique_names -- an EP group keeps its
"ep"-named device_communicator (so is_ep_communicator / use_all2all stay True and
all2all still initializes), but no second communicator set is allocated.

Two independent test surfaces:

1. **Decision logic (GPU-free, gloo/CPU).** The dedup decision is pure Python over
   rank lists. These tests monkeypatch `init_model_parallel_group` to record the
   reuse graph without building any device communicator, so they run on any box
   (no GPU / NCCL). They cover the topologies that actually occur in production:
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
class _FakeGroup:
    """Stand-in returned by the patched init_model_parallel_group.

    Records only what the reuse decision produces: this rank's subgroup, the
    group_name-derived unique_name, and the source it was told to reuse. Builds
    no process group / device communicator, so the decision runs GPU-free.
    """

    def __init__(self, group_name, my_ranks, rank, reuse_from):
        # Real code appends a counter ("ep:0"); "ep" in name is all we assert on.
        self.unique_name = group_name
        self.group_name = group_name
        self.ranks = list(my_ranks)
        self.world_size = len(my_ranks)
        self.rank_in_group = my_ranks.index(rank)
        self.reuse_from = reuse_from

    def destroy(self):
        pass


def _record_groups(rank):
    """Patch ps.init_model_parallel_group with a recorder; return the restorer.

    _build_group looks the symbol up in the module namespace at call time, so
    patching the attribute is enough. Restored even if the body raises.
    """
    original = ps.init_model_parallel_group

    def recorder(
        group_ranks,
        local_rank,
        backend,
        use_device_communicator=True,
        use_message_queue_broadcaster=False,
        group_name=None,
        reuse_from=None,
    ):
        my = next((r for r in group_ranks if rank in r), group_ranks[0])
        return _FakeGroup(group_name, my, rank, reuse_from)

    ps.init_model_parallel_group = recorder
    return lambda: setattr(ps, "init_model_parallel_group", original)


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
            # A reusing group inherits its source's ranks/rank_in_group verbatim,
            # so reuse is only correct when the *ordered* rank list is identical --
            # not merely the same set. Dedup keys on tuple(my_ranks) for exactly
            # this reason; asserting ordered equality (not set equality) locks that
            # in: if keying ever reverts to sorted() and a non-ascending group
            # appears, a same-set/different-order collapse would trip this.
            assert (
                g.ranks == g.reuse_from.ranks
            ), f"{name} reuses a source with a different rank order"
            # ...and single-member groups never reuse (they hold no communicator).
            assert g.world_size > 1, f"{name} is single-rank yet reuses"
        # EP always keeps an ep-named unique_name, reuse or not.
        if name == "ep":
            assert "ep" in g.unique_name, f"EP unique_name lost 'ep': {g.unique_name}"


def _decision_worker(rank, world_size, port, topo, reuse):
    restore = None
    try:
        os.environ[_ENV] = "1" if reuse else "0"
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="gloo",
        )
        restore = _record_groups(rank)
        initialize_model_parallel(**_dims(topo, world_size))
        g = _groups()

        if not reuse:
            # Flag off: nothing reuses, regardless of topology.
            for name, grp in g.items():
                assert grp.reuse_from is None, f"{name} reused with flag off"
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

        destroy_model_parallel()
    finally:
        if restore is not None:
            restore()
        if dist.is_initialized():
            destroy_distributed_environment()


def _unanimity_worker(rank, world_size, port):
    """rank 0 sets the flag off, the rest on -> the unanimity assert must fire."""
    try:
        os.environ[_ENV] = "0" if rank == 0 else "1"
        init_distributed_environment(
            world_size=world_size,
            rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            local_rank=rank,
            backend="gloo",
        )
        raised = False
        try:
            initialize_model_parallel(tensor_model_parallel_size=world_size)
        except AssertionError:
            raised = True
        assert raised, "disagreeing reuse flag across ranks must raise AssertionError"
        destroy_model_parallel()
    finally:
        if dist.is_initialized():
            destroy_distributed_environment()


def _spawn(worker, world_size, *args):
    # Fresh port per spawn -> no "address in use" across back-to-back tests.
    port = get_open_port()
    mp.spawn(worker, args=(world_size, port, *args), nprocs=world_size, join=True)


@pytest.mark.parametrize("topo", ["all_alias", "dp_attention", "partial"])
@pytest.mark.parametrize("reuse", [True, False])
def test_reuse_decision(topo, reuse):
    """GPU-free: the dedup decision picks the right source per topology."""
    _spawn(_decision_worker, 4, topo, reuse)


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
    # Wire the custom-allreduce signal buffer exactly as init_dist_env does, so
    # the (possibly shared) ca_comm is functional. Use the multi-rank source
    # group -- TP is a singleton (device_communicator is None) under dp-attention.
    src = get_dp_group() if topo == "dp_attention" else get_tp_group()
    dc = src.device_communicator
    ca = dc.ca_comm if dc is not None else None
    if ca is not None and not getattr(ca, "_is_gfx1250", False):
        signal = torch.zeros(world_size * 64, dtype=torch.int64, device=rank)
        ca.signal = signal
        ca.register_input_buffer(signal)
        ca.buffer = ca._pool["input"].tensor


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
        assert ep.cpu_group is source.cpu_group

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

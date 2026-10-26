# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import contextlib
import functools
import logging
import os
import sys

# from ..dist.utils import get_open_port, get_distributed_init_method, get_ip
import torch
import torch.distributed as dist

from ..dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,  # noqa: F401 -- re-exported; downstreams import via this module
    init_distributed_environment,
    set_custom_all_reduce,
)

logger = logging.getLogger("aiter")


def init_dist_env(
    tensor_model_parallel_size: int,
    rankID: int,
    backend: str = "cpu:gloo,cuda:nccl",
    distributed_init_method: str | None = "env://",
    local_rank: int = -1,
    data_parallel_size: int = 1,
    data_parallel_rank: int = 0,
    decode_context_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
):
    pipeline_model_parallel_size = 1
    # world_size is TP x PP x PCP (PCP is an independent dimension that grows
    # world_size; see initialize_model_parallel in dist/parallel_state.py).
    world_size = (
        pipeline_model_parallel_size
        * tensor_model_parallel_size
        * prefill_context_model_parallel_size
    )
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
        # distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
        backend=backend,
        local_rank=local_rank,
        data_parallel_size=data_parallel_size,
        data_parallel_rank=data_parallel_rank,
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        decode_context_model_parallel_size=decode_context_parallel_size,
        data_parallel_size=data_parallel_size,
        prefill_context_model_parallel_size=prefill_context_model_parallel_size,
    )

    # No per-rank signal/input-buffer registration here. An earlier version
    # registered a torch.zeros "signal" tensor with CustomAllreduce and mirrored
    # the input pool as `ca_comm.buffer`. All of it was vestigial:
    #   * `ca_comm.signal` / `ca_comm.buffer` are never read anywhere;
    #   * register_input_buffer only inserts a pointer-translation entry keyed
    #     by the registered tensor's own address, consulted when an allreduce
    #     is invoked with that exact tensor as input -- which never happens for
    #     the signal tensor;
    #   * gfx1250 has skipped the whole block (deadlock in its vmm_exchange
    #     rendezvous) and works without it.
    # It was also what made raw IPC input pools unusable (#4921):
    #   * under PYTORCH_HIP_ALLOC_CONF=expandable_segments:True the torch.zeros
    #     signal is VMM-backed, so registering it dies in hipIpcGetMemHandle
    #     (custom_all_reduce.cu:417, "invalid argument");
    #   * the raw_cached input pool has no backing torch.Tensor, so
    #     `ca_comm.buffer = ca_comm._pool["input"].tensor` raised by design.
    # CustomAllreduce.__init__ already sets up its own meta/input pools and the
    # copy-in path for expandable segments (#4174); nothing further is needed.
    logger.debug(f"RANK: {rankID}/{tensor_model_parallel_size} init_dist_env...")


def destroy_dist_env():
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()


# ---- MORI SDMA (the big-M MegaMoE TP kernel's transfers) ----
# Intra-node copy-engine (SDMA) transfers of the big-M MegaMoE TP kernel. MORI CCO provides
# the SDMA queues and their signal slots (its devComm); the kernel posts remote-signalled puts
# (each lands on the peer, then bumps the peer's signal slot for (sender, queue)) on raw
# addresses: the data regions live in a same-layout hipIpc arena on every rank (SymmetricArena,
# torch caching-allocator memory), not in a MORI window. A MORI window is a VMM allocation
# (hipMemCreate), which ROCm can refuse after a pageable host copy pinned a low heap address
# (KFD's SVM-range overlap check, reported as out of memory); hipMalloc'd memory is not affected.
#
# Needs MORI built with ``BUILD_CCO_SDMA=ON`` whose CCO device wrapper (libmori_cco_device.bc)
# provides ``cco_sdma_put_ptr_rsig__thread`` / ``cco_sdma_signal_slot``, and ``MORI_ENABLE_SDMA=1``.
# MORI and FlyDSL are imported on first use (not when aiter is imported).


@contextlib.contextmanager
def _mori_env():
    """For MORI's import / communicator setup only (the process environment is restored):
    AITER_MORI_PATH (MORI's install directory, if not importable) on sys.path, and
    MORI_ENABLE_SDMA=1 unless set (MORI builds the SDMA queues at communicator creation).
    """
    path = os.environ.get("AITER_MORI_PATH")
    added = bool(path) and path not in sys.path
    if added:
        sys.path.insert(0, path)
    unset = "MORI_ENABLE_SDMA" not in os.environ
    if unset:
        os.environ["MORI_ENABLE_SDMA"] = "1"
    try:
        yield
    finally:
        if added and path in sys.path:
            sys.path.remove(path)
        if unset:
            os.environ.pop("MORI_ENABLE_SDMA", None)


def _mori():
    with _mori_env():
        from mori.cco import GDA_CONNECTION_NONE, CCODevCommRequirements, Communicator

    return Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE


def sdma_bitcode() -> str:
    """MORI's CCO device bitcode (code object version 6), which the big-M kernel links."""
    with _mori_env():
        from mori.cco.device.bitcode import find_cco_bitcode

        return find_cco_bitcode(6)


class SdmaComm:
    """MORI SDMA queues (``num_queues`` per peer) of ``group`` plus a same-layout data arena on
    every rank. ``reserve`` carves named regions (same order on every rank), ``commit``
    (collective) creates the MORI communicator / devComm and allocates and maps the arena.
    ``base`` is this rank's arena address, ``bases`` [world] int64 every rank's arena address
    as mapped here (the kernel's put destinations)."""

    def __init__(self, *, group=None, num_queues: int = 8, device=None):
        from .flydsl.kernels.mega_moe_tp.mega_moe_tp import SymmetricArena

        Communicator, Reqs, gda_none = _mori()
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.nq = num_queues
        self._comm_cls, self._reqs_cls, self._gda_none = Communicator, Reqs, gda_none
        self.arena = SymmetricArena(group=group, device=self.device)
        self._slices: dict = {}
        self._committed = False

    def reserve(self, name: str, nbytes: int) -> None:
        if self._committed or name in self._slices:
            raise RuntimeError(f"SdmaComm: cannot reserve {name!r}")
        self._slices[name] = self.arena.reserve(
            name, (max(int(nbytes), 1),), torch.uint8
        )

    def offset(self, name: str) -> int:
        return self._slices[name].offset

    def commit(self) -> "SdmaComm":
        uid = [self._comm_cls.get_unique_id() if self.rank == 0 else None]
        src0 = dist.get_global_rank(self.group, 0) if self.group else 0
        dist.broadcast_object_list(uid, src=src0, group=self.group)
        err = ""
        try:
            with _mori_env():
                # (per_rank_vmm: only a virtual-address reservation, rounded up to 4 GiB by
                # MORI; no MORI window is used, the data lives in the hipIpc arena)
                self.comm = self._comm_cls.init(
                    self.world, self.rank, uid[0], per_rank_vmm=1 << 30
                )
                reqs = self._reqs_cls()
                reqs.gda_connection_type = self._gda_none
                reqs.gda_signal_count = 0
                reqs.gda_counter_count = 0
                reqs.sdma_queue_count = self.nq
                self.dev_comm = self.comm.create_dev_comm(reqs)
        except Exception as exc:  # noqa: BLE001 (reported on every rank below)
            err = f"{type(exc).__name__}: {exc}"
        errs: list = [None] * self.world
        dist.all_gather_object(errs, err, group=self.group)
        bad = {r: e for r, e in enumerate(errs) if e}
        if bad:
            raise RuntimeError(f"SdmaComm: MORI SDMA setup failed on ranks {bad}")
        try:
            self.arena.commit()  # (raises on every rank if any rank fails)
        except BaseException:
            self.dev_comm.close()
            self.comm.destroy()
            raise
        self.storage = self.arena.storage
        self.base = int(self.storage.data_ptr())
        self.bases = torch.tensor(
            self.arena.base_ptrs, dtype=torch.int64, device=self.device
        )
        # [2][world * nq] int32 per-(peer, queue) signal bases of the big-M kernel: each launch
        # reads one buffer and writes the next launch's into the other
        self.consumed = torch.zeros(
            (2, self.world * self.nq), dtype=torch.int32, device=self.device
        )
        self._committed = True
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return self

    def close(self) -> None:
        """Collective: free the MORI device communicator / communicator (its SDMA queues)
        and the arena. The instance is unusable afterwards."""
        if not self._committed:
            return
        self._committed = False
        try:
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.group)
            try:
                self.dev_comm.close()
            finally:
                self.comm.destroy()
        finally:
            self.arena.close()

    def view(self, name: str, shape, dtype: torch.dtype) -> torch.Tensor:
        """A torch view of region ``name``."""
        sl = self._slices[name]
        n = torch.Size(shape).numel() * dtype.itemsize
        assert n <= sl.nbytes, (name, n, sl.nbytes)
        return self.storage[sl.offset : sl.offset + n].view(dtype).view(shape)


@functools.cache
def compile_signal_read(tp, nq):
    """launch(dc, out, stream): out [tp][nq] int32 <- the signal count received per (sender,
    queue) (MORI's signal slots, low 32 bits)"""
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.compiler.extern_link import link_extern
    from flydsl.expr import gpu
    from flydsl.expr.extern import ffi

    from .flydsl.kernels.mega_moe_tp.common import g_ld_sys, g_st_sys, i32

    sig_slot = link_extern(
        ffi("cco_sdma_signal_slot", ["uint64", "int32", "int32"], "uint64"),
        bitcode_path=sdma_bitcode(),
    )

    @flyc.kernel(name=f"sdma_signal_read_tp{tp}_q{nq}", known_block_size=[64, 1, 1])
    def kernel(dc: fx.Int64, out: fx.Int64):
        lane = fx.Int32(gpu.thread_id("x"))
        if lane < i32(tp * nq):
            slot = fx.Int64(sig_slot(dc, lane // i32(nq), lane % i32(nq)))
            g_st_sys(out + fx.Int64(lane * i32(4)), g_ld_sys(slot))

    @flyc.jit
    def launch(dc: fx.Int64, out: fx.Int64, stream: fx.Stream):
        kernel(dc, out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch

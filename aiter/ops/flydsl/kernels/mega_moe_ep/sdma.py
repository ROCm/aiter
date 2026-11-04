# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MORI device-SDMA queues for the large EP kernel's payload puts (AITER_MEGAMOE_EP_SDMA).

The data stays in the EP hipIpc arena (the puts take raw peer-mapped addresses); MORI only
provides the per-peer SDMA queues and the per-(sender, queue) remote signal counters.
"""

from __future__ import annotations

import contextlib
import functools
import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl.expr import gpu

NQ = 8  # SDMA queues per peer (queue 0 unused)


@contextlib.contextmanager
def _mori_env():
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


@functools.cache
def sdma_bitcode() -> str:
    with _mori_env():
        from mori.cco.device.bitcode import find_cco_bitcode

        return find_cco_bitcode(6)


@functools.cache
def sdma_externs():
    """(put_ptr, sig_slot): cco_sdma_put_ptr_rsig__thread(dc, peer, dst, src, nbytes, q)
    and cco_sdma_signal_slot(dc, peer, q) -> the received-signal counter address."""
    from flydsl.compiler.extern_link import link_extern
    from flydsl.expr.extern import ffi

    bc = sdma_bitcode()
    put_ptr = link_extern(
        ffi(
            "cco_sdma_put_ptr_rsig__thread",
            ["uint64", "int32", "uint64", "uint64", "uint64", "int32"],
            "int32",
        ),
        bitcode_path=bc,
    )
    sig_slot = link_extern(
        ffi("cco_sdma_signal_slot", ["uint64", "int32", "int32"], "uint64"),
        bitcode_path=bc,
    )
    return put_ptr, sig_slot


@functools.cache
def _signal_read(world):
    from .common import g_ld_sys, g_st_sys, i32

    sig_slot = sdma_externs()[1]

    @flyc.kernel(
        name=f"ep_sdma_signal_read_np{world}_q{NQ}", known_block_size=[64, 1, 1]
    )
    def kernel(dc: fx.Int64, out: fx.Int64):
        lane = fx.Int32(gpu.thread_id("x"))
        if lane < i32(world * NQ):
            slot = fx.Int64(sig_slot(dc, lane // i32(NQ), lane % i32(NQ)))
            g_st_sys(out + fx.Int64(lane * i32(4)), g_ld_sys(slot))

    @flyc.jit
    def launch(dc: fx.Int64, out: fx.Int64, stream: fx.Stream):
        kernel(dc, out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


class SdmaQueues:
    """Collective: the MORI communicator + device communicator (NQ SDMA queues per peer)."""

    def __init__(self, *, group=None, device=None):
        with _mori_env():
            from mori.cco import (
                GDA_CONNECTION_NONE,
                CCODevCommRequirements,
                Communicator,
            )
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device = device
        uid = [Communicator.get_unique_id() if self.rank == 0 else None]
        dist.broadcast_object_list(
            uid, src=dist.get_global_rank(group, 0) if group else 0, group=group
        )
        err = ""
        try:
            with _mori_env():
                self.comm = Communicator.init(
                    self.world, self.rank, uid[0], per_rank_vmm=1 << 30
                )
                reqs = CCODevCommRequirements()
                reqs.gda_connection_type = GDA_CONNECTION_NONE
                reqs.gda_signal_count = 0
                reqs.gda_counter_count = 0
                reqs.sdma_queue_count = NQ
                self.dev_comm = self.comm.create_dev_comm(reqs)
        except Exception as exc:  # noqa: BLE001 (reported on every rank below)
            err = f"{type(exc).__name__}: {exc}"
        errs: list = [None] * self.world
        dist.all_gather_object(errs, err, group=group)
        bad = {r: e for r, e in enumerate(errs) if e}
        if bad:
            raise RuntimeError(f"SdmaQueues: MORI SDMA setup failed on ranks {bad}")
        self.ptr = int(self.dev_comm.ptr)

    def signals(self):
        """[world][NQ] int32: the signal count received per (sender, queue)."""
        from ..tensor_shim import _run_compiled

        torch.cuda.synchronize(self.device)
        out = torch.zeros(self.world * NQ, dtype=torch.int32, device=self.device)
        _run_compiled(
            _signal_read(self.world),
            self.ptr,
            out.data_ptr(),
            torch.cuda.current_stream(self.device),
        )
        torch.cuda.synchronize(self.device)
        return out.view(self.world, NQ)

    def close(self):
        try:
            self.dev_comm.close()
        finally:
            self.comm.destroy()

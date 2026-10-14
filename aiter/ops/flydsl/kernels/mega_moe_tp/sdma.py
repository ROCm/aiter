# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Intra-node copy-engine (SDMA) transfers through MORI CCO, driven from tiny FlyDSL
kernels: ``post`` rings remote-signalled puts (each lands on the peer, then bumps the
peer's signal slot for (sender, queue)), ``wait`` polls this rank's slots until every
expected put arrived. Both are ordinary stream-ordered launches, so a transfer runs on
the DMA engines underneath whatever compute the stream does between post and wait,
and they capture into CUDA graphs. The data lives in one symmetric MORI window per
rank; torch views of its regions are handed to the compute ops.

Needs MORI built with ``BUILD_CCO_SDMA=ON`` (its device wrapper carrying
``cco_sdma_put_rsig__thread`` / ``cco_sdma_signal_slot``) and ``MORI_ENABLE_SDMA=1``.
"""

from __future__ import annotations

import os
import sys

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl.expr import range_constexpr
from flydsl.expr.typing import Int64

from ..tensor_shim import _run_compiled
from .common import g_ld_sys, g_st_sys, i32, poll_sys_ge

__all__ = ["SdmaComm"]

_ALIGN = 4096
DESC_POST = 5  # peer, dst offset, src offset, bytes, queue (int64 each)
DESC_WAIT = 3  # src rank, queue, puts expected
MAX_DESC = 64


def _mori():
    path = os.environ.get("AITER_MORI_PATH")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    os.environ.setdefault("MORI_ENABLE_SDMA", "1")
    from mori.cco import CCODevCommRequirements, Communicator, GDA_CONNECTION_NONE
    from mori.cco.device.bitcode import find_cco_bitcode

    return Communicator, CCODevCommRequirements, GDA_CONNECTION_NONE, find_cco_bitcode


class _Cai:
    def __init__(self, ptr, nbytes):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def _kernels(bitcode, tag=""):
    from flydsl.compiler.extern_link import link_extern
    from flydsl.expr.extern import ffi

    put_rsig = link_extern(
        ffi(
            "cco_sdma_put_rsig__thread",
            ["uint64", "int32", "uint64", "uint64", "uint64", "uint64", "uint64", "int32", "int32"],
            "int32",
        ),
        bitcode_path=bitcode,
    )
    sig_slot = link_extern(
        ffi("cco_sdma_signal_slot", ["uint64", "int32", "int32"], "uint64"),
        bitcode_path=bitcode,
    )

    def ld64(addr):
        lo = fx.Int64(g_ld_sys(addr)) & fx.Int64(0xFFFFFFFF)
        hi = fx.Int64(g_ld_sys(addr + fx.Int64(4)))
        return lo | (hi << fx.Int64(32))

    @flyc.kernel(name=f"sdma_post{tag}", known_block_size=[64, 1, 1])
    def post_kernel(dc: Int64, win: Int64, tab: Int64, n: fx.Int32):
        t = fx.Int32(fx.thread_idx.x)
        if t < n:
            d = fx.Int64(tab) + fx.Int64(t * i32(DESC_POST * 8))
            put_rsig(
                dc,
                fx.Int32(ld64(d)),
                win,
                ld64(d + fx.Int64(8)),
                win,
                ld64(d + fx.Int64(16)),
                ld64(d + fx.Int64(24)),
                fx.Int32(ld64(d + fx.Int64(32))),
                0,
            )

    # one thread submits every put in order: concurrent puts on the same queue lose
    # remote signals (MORI ring submission), so tables with repeated (peer, queue) use this
    @flyc.kernel(name=f"sdma_post_serial{tag}", known_block_size=[64, 1, 1])
    def post_serial_kernel(dc: Int64, win: Int64, tab: Int64, n: fx.Int32):
        if fx.Int32(fx.thread_idx.x) == i32(0):
            for t_ in range(i32(0), n, i32(1)):
                d = fx.Int64(tab) + fx.Int64(i32(t_) * i32(DESC_POST * 8))
                put_rsig(
                    dc,
                    fx.Int32(ld64(d)),
                    win,
                    ld64(d + fx.Int64(8)),
                    win,
                    ld64(d + fx.Int64(16)),
                    ld64(d + fx.Int64(24)),
                    fx.Int32(ld64(d + fx.Int64(32))),
                    0,
                )

    @flyc.jit
    def post_serial(dc: Int64, win: Int64, tab: Int64, n: fx.Int32, stream=fx.Stream(None)):
        post_serial_kernel(dc, win, tab, n).launch(grid=(1, 1, 1), block=[64, 1, 1], stream=stream)

    @flyc.kernel(name=f"sdma_wait{tag}", known_block_size=[64, 1, 1])
    def wait_kernel(dc: Int64, tab: Int64, n: fx.Int32, consumed: Int64, nq: fx.Int32):
        t = fx.Int32(fx.thread_idx.x)
        if t < n:
            d = fx.Int64(tab) + fx.Int64(t * i32(DESC_WAIT * 8))
            src = fx.Int32(ld64(d))
            q = fx.Int32(ld64(d + fx.Int64(8)))
            cnt = fx.Int32(ld64(d + fx.Int64(16)))
            slot = fx.Int64(sig_slot(dc, src, q))
            ca = fx.Int64(consumed) + fx.Int64((src * nq + q) * i32(4))
            tgt = g_ld_sys(ca) + cnt
            poll_sys_ge(slot, tgt)
            g_st_sys(ca, tgt)

    @flyc.jit
    def post(dc: Int64, win: Int64, tab: Int64, n: fx.Int32, stream=fx.Stream(None)):
        post_kernel(dc, win, tab, n).launch(grid=(1, 1, 1), block=[64, 1, 1], stream=stream)

    @flyc.jit
    def wait(
        dc: Int64, tab: Int64, n: fx.Int32, consumed: Int64, nq: fx.Int32, stream=fx.Stream(None)
    ):
        wait_kernel(dc, tab, n, consumed, nq).launch(
            grid=(1, 1, 1), block=[64, 1, 1], stream=stream
        )

    return post, wait, post_serial


class SdmaComm:
    """One symmetric MORI window per rank of ``group`` plus ``num_queues`` SDMA queues
    per peer. ``reserve`` carves named regions (same layout on every rank; call it in
    the same order everywhere), ``commit`` allocates and maps them. ``post`` / ``wait``
    take descriptor tables built once by ``post_table`` / ``wait_table``."""

    def __init__(self, *, group=None, num_queues: int = 8, device=None):
        Communicator, Reqs, gda_none, self._find_bc = _mori()
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.nq = num_queues
        self._regions: dict[str, tuple[int, int]] = {}
        self._cursor = 0
        self._comm_cls, self._reqs_cls, self._gda_none = Communicator, Reqs, gda_none
        self._win = None

    def reserve(self, name: str, nbytes: int) -> int:
        if self._win is not None or name in self._regions:
            raise RuntimeError(f"SdmaComm: cannot reserve {name!r}")
        off = -(-self._cursor // _ALIGN) * _ALIGN
        self._regions[name] = (off, int(nbytes))
        self._cursor = off + int(nbytes)
        return off

    def offset(self, name: str) -> int:
        return self._regions[name][0]

    def commit(self) -> "SdmaComm":
        uid = [self._comm_cls.get_unique_id() if self.rank == 0 else None]
        dist.broadcast_object_list(uid, src=dist.get_global_rank(self.group, 0) if self.group else 0, group=self.group)
        total = -(-max(self._cursor, _ALIGN) // _ALIGN) * _ALIGN
        vmm = 1 << max(30, (total + (1 << 28)).bit_length())
        self.comm = self._comm_cls.init(self.world, self.rank, uid[0], per_rank_vmm=vmm)
        self.mem = self.comm.alloc_mem(total)
        self._win = self.comm.register_window(self.mem.ptr, self.mem.size)
        reqs = self._reqs_cls()
        reqs.gda_connection_type = self._gda_none
        reqs.gda_signal_count = 0
        reqs.gda_counter_count = 0
        reqs.sdma_queue_count = self.nq
        self.dev_comm = self.comm.create_dev_comm(reqs)
        self.base = int(self._win.local_ptr)
        self.handle = int(self._win.handle)
        self.storage = torch.as_tensor(_Cai(self.base, total), device=self.device)
        self.storage.zero_()
        self.consumed = torch.zeros(self.world * self.nq, dtype=torch.int32, device=self.device)
        self._bc = self._find_bc(6)
        self._kn = {"": _kernels(self._bc)}
        self._serial: set = set()
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        return self

    def view(self, name: str, shape, dtype: torch.dtype, offset: int = 0) -> torch.Tensor:
        """A torch view of region ``name`` (plus a byte offset into it)."""
        off, size = self._regions[name]
        n = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        assert offset + n <= size, (name, offset, n, size)
        return self.storage[off + offset : off + offset + n].view(dtype).view(shape)

    def post_table(self, puts, dup: bool = False) -> torch.Tensor:
        """puts: [(peer, dst region, dst byte offset, src region, src byte offset, bytes,
        queue)], at most one per (peer, queue) unless ``dup`` (FIFO single-queue use)."""
        keys = [(p, q) for p, *_, q in puts]
        assert (dup or len(keys) == len(set(keys))) and len(puts) <= MAX_DESC, puts
        serial = len(keys) != len(set(keys))
        rows = [
            [p, self.offset(dr) + do, self.offset(sr) + so, n, q]
            for p, dr, do, sr, so, n, q in puts
        ]
        t = torch.tensor(rows or [[0] * DESC_POST], dtype=torch.int64, device=self.device)
        if serial:
            self._serial.add(t.data_ptr())
        return t

    def wait_table(self, waits) -> torch.Tensor:
        """waits: [(src rank, queue[, n])] — n (default 1) puts of src on that queue."""
        agg: dict = {}
        for w in waits:  # entries on the same (src, queue) add up (FIFO single-queue use)
            r = list(w) + [1] * (3 - len(w))
            agg[(r[0], r[1])] = agg.get((r[0], r[1]), 0) + r[2]
        rows = [[k[0], k[1], n] for k, n in agg.items()]
        assert len(rows) <= MAX_DESC, waits
        return torch.tensor(rows or [[0, 0, 0]], dtype=torch.int64, device=self.device)

    def _k(self, tag):
        k = self._kn.get(tag)
        if k is None:
            k = self._kn[tag] = _kernels(self._bc, "_" + tag)
        return k

    def post(self, tab: torch.Tensor, n: int, tag: str = "") -> None:
        """tag: suffix of the post kernel's name (profiling)."""
        if n and "nocomm" not in os.environ.get("XL_ABL", ""):
            k = self._k(tag)
            _run_compiled(
                k[2] if tab.data_ptr() in self._serial else k[0],
                self.dev_comm.ptr,
                self.handle,
                tab.data_ptr(),
                n,
                torch.cuda.current_stream(),
            )

    def wait(self, tab: torch.Tensor, n: int, tag: str = "") -> None:
        if os.environ.get("XL_DBGW"):  # [dbg] per-wait elapsed via events, printed by dbg_flush
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            self._wait(tab, n, tag)
            e1.record()
            self.__dict__.setdefault("_dbg", []).append((f"{tag}@{tab.data_ptr():x}", tab.tolist(), e0, e1))
            return
        self._wait(tab, n, tag)

    def dbg_flush(self, rank):
        torch.cuda.synchronize()
        for tag, rows, e0, e1 in self.__dict__.pop("_dbg", []):
            ms = e0.elapsed_time(e1)
            if ms > 100:
                print(f"[dbgw r{rank}] wait {tag} {rows} {ms:.1f} ms", flush=True)

    def _wait(self, tab: torch.Tensor, n: int, tag: str = "") -> None:
        if n and "nocomm" not in os.environ.get("XL_ABL", ""):
            _run_compiled(
                self._k(tag)[1],
                self.dev_comm.ptr,
                tab.data_ptr(),
                n,
                self.consumed.data_ptr(),
                self.nq,
                torch.cuda.current_stream(),
            )

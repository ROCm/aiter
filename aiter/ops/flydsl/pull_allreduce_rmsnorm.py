# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project

"""Host side of the ``pull`` schedule of ``FlyQuickAllReduceRMSNorm``: bf16 all-reduce + residual add + Gemma
RMSNorm on registered CUDA-graph inputs (TP4, gfx950, hidden 6144).

``PullAllReduceRMSNorm`` is what ``FlyQuickAllReduceRMSNorm(algorithm="pull")`` delegates to. Its wire is plain bf16
rows read in place, so it has no codec and no super-tile ladder; the ranks exchange two things instead:

* one uncached flag + reduced-row buffer per rank (``quick_allreduce_pull`` documents the layout), and
* the address of every graph-captured input. Calls recorded inside ``capture()`` each take a row of an int64[4]
  table; when the context exits the ranks exchange the IPC handles of the inputs' allocations and fill the rows, so
  the kernels read the peers' inputs through the table with no copy. ``capture()`` is the counterpart of
  ``CustomAllreduce.capture()``; calls outside a registered capture are declined (``accepts``) and the caller's next
  backend serves them.

Which payloads it serves and which kernel serves each is not decided here: the payload window is the ``pull`` window
of the fused policy row (``allreduce_policy.FUSED_FAMILY_POLICIES``, ``AITER_FLY_AR_FUSED_PULL_{MIN,MAX}_BYTES``) and
the kernel of each payload is a rung of the ``pull`` ladder of the width (``fused_allreduce_ladders.json``). The
kernels and the numerics follow amd-kernels' ``StagedArNormComm`` registered tiers
(``allreduce_add_rmsnorm/staged_host.py`` and ``host.py``): the residual output equals AITER's two-stage fp32 rank sum
bit for bit except a rare fp32 summation-order difference, and the norm output matches ``gemma_add_rmsnorm`` (Triton's
``tl.sum`` order).
"""

from __future__ import annotations

import contextlib
import ctypes
import functools
import logging
import struct

import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64, Stream

from . import allreduce_policy as policy
from .comm_collectives_host_shared import (
    _cuda_index,
    _validate_ipc_process_group,
    check_launch_outside_capture,
    launcher_loaded,
    preload_launcher,
    require_arch,
)
from .ipc_heap import IpcHeap
from .kernels.comm import quick_allreduce_pull as K
from .kernels.tensor_shim import _run_compiled

logger = logging.getLogger("aiter")

WORLD = K.WORLD
HIDDEN = K.H
ROW_BYTES = 2 * HIDDEN
#: Rows the flag tables of the kernels hold: the physical bound, whatever the policy window says.
MAX_ROWS = K.MAX_ROWS
#: Captured calls the address table holds.
MAX_ENTRIES = 1 << 15


def _f32_bits(v: float) -> int:
    return struct.unpack("<i", struct.pack("<f", v))[0]


# One launcher object per device: `_run_compiled` keeps a single device-bound handle in the object, so
# engines of two GPUs in one process must not share one.
@functools.cache
def _oneshot_kernel(device: int):
    return K.make_oneshot_ack_kernel()


@functools.cache
def _pull_kernel(device: int, wave: bool):
    return K.make_pull_kernel(wave)


@functools.cache
def _stream_kernel(device: int, rpw: int, hoist: bool):
    return K.make_stream_pull_kernel(rpw, hoist)


def rung_name(rung: policy.FusedPullRung) -> str:
    return "_".join(
        str(k).lower() if not isinstance(k, bool) else ("wave" if k else "block")
        for k in rung.cfg
    )


class _IpcRegistry:
    """The peers' views of graph-captured input allocations, opened once per handle."""

    def __init__(self):
        self.opened: dict[bytes, int] = {}
        hip = IpcHeap._load_hip()
        hip.hipMemGetAddressRange.restype = ctypes.c_int
        hip.hipMemGetAddressRange.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        self._hip = hip

    def export(self, ptr: int) -> tuple[int, bytes]:
        """(base of the allocation holding *ptr*, its IPC handle)."""
        base, size = ctypes.c_void_p(), ctypes.c_size_t()
        rc = self._hip.hipMemGetAddressRange(
            ctypes.byref(base), ctypes.byref(size), ctypes.c_void_p(ptr)
        )
        if rc != 0:
            raise RuntimeError(f"hipMemGetAddressRange failed: {rc}")
        return int(base.value), IpcHeap.get_mem_handle_bytes(int(base.value))

    def open(self, handle: bytes) -> int:
        if handle not in self.opened:
            self.opened[handle] = IpcHeap.open_mem_handle(handle)
        return self.opened[handle]

    def close(self):
        for base in self.opened.values():
            try:
                IpcHeap.close_mem_handle(base)
            except RuntimeError:
                pass
        self.opened.clear()


class PullAllReduceRMSNorm:
    """TP4 bf16 all-reduce + residual add + Gemma RMSNorm of rows of 6144, on graph-registered inputs.

    *window* (``policy.Window``, payload bytes) is where the engine serves and *ladders* maps a hidden width to its
    ``policy.FusedPullRung`` ladder, both from the fused policy (``allreduce_policy``): this class has no thresholds of
    its own, only the physical bound ``MAX_ROWS`` of the kernels' flag tables.

    Construction is a collective (IPC handle exchange of one flag buffer per rank). ``capture()`` wraps every graph
    capture on every rank. ``accepts`` says whether a call is served; ``allreduce_rmsnorm`` raises when it is not.
    """

    def __init__(
        self, *, group, device, rank: int, world_size: int, window, ladders: dict
    ):
        if window is None or window.is_empty:
            raise ValueError("the pull window is empty: nothing to serve")
        # The kernels are built for one width; a ladder for another one has nothing to run on.
        self._ladders = {
            int(h): tuple(r) for h, r in ladders.items() if int(h) == HIDDEN
        }
        if not self._ladders:
            raise ValueError(f"no pull ladder for hidden={HIDDEN}")
        self.window = window
        self.max_rows = min(MAX_ROWS, window.inclusive_range()[1] // ROW_BYTES)
        for ladder in self._ladders.values():
            self._check_ladder(ladder)
        if int(world_size) != WORLD:
            raise ValueError(f"the pull schedule is TP{WORLD} only, got TP{world_size}")
        _validate_ipc_process_group(
            group, rank=int(rank), world_size=world_size, owner="PullAllReduceRMSNorm"
        )
        arch = require_arch(owner="PullAllReduceRMSNorm")
        if arch != "gfx950":
            raise ValueError(f"the pull schedule needs gfx950, got {arch}")
        self._device_index = _cuda_index(device, owner="PullAllReduceRMSNorm")
        self.device = torch.device("cuda", self._device_index)
        self.group = group
        self.rank = int(rank)
        self.world_size = WORLD
        self._buf = None
        self._peer_bases: list[int | None] = [None] * WORLD
        self._bufs: list[int] = []
        self._registry = None
        self._inputs: list[int] = []  # this rank's input address per table row
        self._filled = 0
        self._capturing = False
        self._launched = False
        # Setup has two collectives; a rank that fails still takes part in both and reports through them, so
        # its peers cannot hang (see ``_StEngine``).
        error = None
        try:
            with torch.cuda.device(self._device_index):
                self._buf = IpcHeap.alloc_uncached(K.BUF_BYTES)
                mine = IpcHeap.get_mem_handle_bytes(self._buf)
                self._registry = _IpcRegistry()
        except Exception as e:  # noqa: BLE001
            error = e
            mine = None
        try:
            handles = IpcHeap.gather_object_list_via_broadcast(group, mine)
        except Exception:
            self.close()
            raise
        if error is None:
            try:
                if any(h is None for h in handles):
                    raise RuntimeError("a peer rank failed to allocate its pull buffer")
                for r in range(WORLD):
                    if r == self.rank:
                        self._bufs.append(self._buf)
                    else:
                        self._peer_bases[r] = IpcHeap.open_mem_handle(bytes(handles[r]))
                        self._bufs.append(self._peer_bases[r])
                with torch.cuda.device(self._device_index):
                    # the call counts (ack, pull, stream domains) and the table are written by this rank only
                    self._counts = torch.zeros(
                        3 * K.MAX_ROWS, dtype=torch.int32, device=self.device
                    )
                    self._table = torch.zeros(
                        (MAX_ENTRIES, WORLD), dtype=torch.int64, device=self.device
                    )
                    torch.cuda.synchronize(self.device)
            except Exception as e:  # noqa: BLE001
                error = e
        ok = torch.tensor([0 if error is not None else 1], dtype=torch.int32)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
        if error is not None or not ok.item():
            self.close()
            raise error or RuntimeError("a peer rank failed to set up the pull engine")

    def _check_ladder(self, ladder) -> None:
        """Refuse a ladder that runs a kernel on rows it is not known to survive:
        the one-row-per-workgroup kernels hang when their grid cannot be
        resident (``policy.PULL_KERNEL_MAX_ROWS``), and a hang inside a graph
        replay cannot be recovered."""
        first = max(1, -(-self.window.min // ROW_BYTES))
        for i, rung in enumerate(ladder):
            lo = max(first, -(-rung.min_bytes // ROW_BYTES))
            hi = self.max_rows
            if i + 1 < len(ladder):
                hi = min(hi, -(-ladder[i + 1].min_bytes // ROW_BYTES) - 1)
            cap = policy.PULL_KERNEL_MAX_ROWS[rung.kernel]
            if lo <= hi and hi > cap:
                raise ValueError(
                    f"pull ladder: the {rung.kernel} kernel serves rows {lo}..{hi} "
                    f"but is only known to run to {cap} rows"
                )

    # -- registration --------------------------------------------------------

    @contextlib.contextmanager
    def capture(self):
        """Around graph capture on every rank (same captures in the same order): the calls recorded inside take table
        rows, filled with the ranks' input addresses when the context exits (a collective).
        """
        self._capturing = True
        try:
            yield
        except BaseException:
            # a failed capture is discarded: its rows would never be filled
            del self._inputs[self._filled :]
            raise
        finally:
            self._capturing = False
        self.register_graph_buffers()

    @property
    def capturing(self) -> bool:
        return self._capturing

    def _row(self, x: torch.Tensor) -> int:
        n = len(self._inputs)
        if n >= MAX_ENTRIES:
            raise RuntimeError("PullAllReduceRMSNorm: address table full")
        self._inputs.append(x.data_ptr())
        return self._table.data_ptr() + n * WORLD * 8

    def register_graph_buffers(self) -> None:
        """Fill the rows taken since the last call with the four ranks' input addresses (a collective: the ranks
        captured the same calls in order)."""
        new = self._inputs[self._filled :]
        counts = IpcHeap.gather_object_list_via_broadcast(self.group, len(new))
        if len(set(counts)) != 1:
            raise RuntimeError(f"PullAllReduceRMSNorm: captured calls differ {counts}")
        if not new:
            return
        handles: list[bytes] = []
        index: dict[int, int] = {}
        metas = []
        for p in new:
            base, raw = self._registry.export(p)
            if base not in index:
                index[base] = len(handles)
                handles.append(raw)
            metas.append((index[base], p - base))
        every = IpcHeap.gather_object_list_via_broadcast(self.group, (handles, metas))
        rows = torch.empty((len(new), WORLD), dtype=torch.int64)
        for r, (hs, ms) in enumerate(every):
            if r == self.rank:
                rows[:, r] = torch.tensor(new, dtype=torch.int64)
                continue
            bases = [self._registry.open(bytes(h)) for h in hs]
            rows[:, r] = torch.tensor([bases[i] + off for i, off in ms])
        self._table[self._filled : self._filled + len(new)].copy_(rows)
        torch.cuda.synchronize(self.device)
        self._filled += len(new)

    # -- selection -----------------------------------------------------------

    def rung(self, rows: int) -> policy.FusedPullRung:
        """The rung of the ladder that runs *rows* rows."""
        return policy.select_rung(self._ladders[HIDDEN], rows * ROW_BYTES)

    def tier_name(self, rows: int) -> str:
        return rung_name(self.rung(rows))

    def supports_hidden(self, hidden: int) -> bool:
        return int(hidden) in self._ladders

    def accepts(self, inp, residual_in, weight, out=None, residual_out=None) -> bool:
        """Whether this call is served: registered graph capture, bf16 [rows, 6144], a payload inside the window, a contiguous
        input and residual rows of 16-byte multiples. Never raises."""
        if not (self._capturing and torch.cuda.is_current_stream_capturing()):
            return False
        if len(self._inputs) >= MAX_ENTRIES:
            # the address table is full: the caller's next backend serves it
            return False
        if (
            inp.dim() != 2
            or inp.shape[1] != HIDDEN
            or not 0 < inp.shape[0] <= self.max_rows
        ):
            return False
        if not self.window.contains(inp.shape[0] * ROW_BYTES):
            return False
        for t in (inp, residual_in, weight):
            if t.dtype is not torch.bfloat16 or t.device.index != self._device_index:
                return False
        rows = inp.shape[0]
        if not inp.is_contiguous() or residual_in.shape != inp.shape:
            return False
        # the stride is an Int32 launch argument (the kernels widen it before the byte offset)
        if (
            residual_in.stride(1) != 1
            or residual_in.stride(0) % 8
            or residual_in.stride(0) >= 2**31
        ):
            return False
        if weight.numel() != HIDDEN or not weight.is_contiguous():
            return False
        for t in (inp, residual_in, weight, out, residual_out):
            if t is not None and t.data_ptr() % 16:
                return False
        for t in (out, residual_out):
            if t is not None and (
                t.dtype is not torch.bfloat16
                or t.device.index != self._device_index  # a CPU tensor has no index
                or tuple(t.shape) != (rows, HIDDEN)
                or not t.is_contiguous()
            ):
                return False
        if out is not None and residual_out is not None:
            # each kernel stores the residual and then the normalized row: overlapping destinations
            # cannot both be returned
            span = lambda t: (t.data_ptr(), t.data_ptr() + t.numel() * t.element_size())
            (a0, a1), (b0, b1) = span(out), span(residual_out)
            if a0 < b1 and b0 < a1:
                return False

        # A destination (out, residual_out) must not share bytes with a source (inp, residual_in,
        # weight) either, unless it is the exact same tensor (identical data_ptr, shape and stride:
        # every element then reads and writes the same address, so there is no cross-row hazard).
        # A shifted or partial overlap -- e.g. residual_out advanced one row from residual_in, or an
        # output row landing on weight -- lets one workgroup's write land on bytes a different
        # workgroup (or the weight read of every row) has not read yet.
        def _extent(t):
            if t.dim() == 2 and t.shape[0] > 0:
                lo = t.data_ptr()
                hi = (
                    lo
                    + ((t.shape[0] - 1) * t.stride(0) + t.shape[1]) * t.element_size()
                )
                return lo, hi
            return t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()

        def _exact_alias(a, b):
            return (
                a.data_ptr() == b.data_ptr()
                and tuple(a.shape) == tuple(b.shape)
                and tuple(a.stride()) == tuple(b.stride())
            )

        for d in (out, residual_out):
            if d is None:
                continue
            d0, d1 = _extent(d)
            for s in (inp, residual_in, weight):
                if _exact_alias(d, s):
                    continue
                s0, s1 = _extent(s)
                if d0 < s1 and s0 < d1:
                    return False
        if not launcher_loaded(self._launcher(rows)[0]):
            return False  # not preloaded: a capture cannot JIT-compile
        if self.rung(rows).kernel == "oneshot":
            # the one-shot acks after the norm stores start: an output over the input would race the readers
            lo, hi = inp.data_ptr(), inp.data_ptr() + inp.numel() * inp.element_size()
            for t in (out, residual_out):
                if t is not None and not (
                    t.data_ptr() >= hi
                    or t.data_ptr() + t.numel() * t.element_size() <= lo
                ):
                    return False
        return True

    # -- launch --------------------------------------------------------------

    def _launcher(self, rows: int):
        key = self.rung(rows).cfg
        if key[0] == "oneshot":
            return _oneshot_kernel(self._device_index), K.ACK_OFF, 0
        if key[0] == "pull":
            return _pull_kernel(self._device_index, key[1]), K.PULL_OFF, K.MAX_ROWS
        return (
            _stream_kernel(self._device_index, key[1], key[2]),
            K.STREAM_OFF,
            2 * K.MAX_ROWS,
        )

    def _args(self, bufoff, cntoff, rows, ptrs, res_stride, eps, stream):
        x, res_in, w, out, res_out = ptrs
        return (
            Int32(rows),
            Int32((rows + WORLD - 1) // WORLD),
            Int32(self.rank),
            Int32(res_stride),
            Int64(x),
            *(Int64(b + bufoff) for b in self._bufs),
            Int64(res_in),
            Int64(w),
            Int64(out),
            Int64(res_out),
            Int64(self._counts.data_ptr() + cntoff * 4),
            Int32(_f32_bits(eps)),
            stream,
        )

    def preload(self, hidden: int = HIDDEN, *, payload_range=None) -> None:
        """JIT-compile every tier kernel and load it onto the device without launching (graph capture cannot
        compile). Local to the rank; no collective."""
        if not self.supports_hidden(hidden):
            raise ValueError(f"the pull schedule serves hidden={HIDDEN}, got {hidden}")
        lo_rows, hi_rows = 1, self.max_rows
        if payload_range is not None:
            lo_rows = max(1, -(-int(payload_range[0]) // ROW_BYTES))
            hi_rows = min(self.max_rows, int(payload_range[1]) // ROW_BYTES)
        # One row per kernel the ladder can pick inside [lo_rows, hi_rows]: the first row of each rung that the
        # window and the range leave.
        ladder = self._ladders[HIDDEN]
        seen = {}
        for i, rung in enumerate(ladder):
            first = max(lo_rows, -(-rung.min_bytes // ROW_BYTES))
            last = hi_rows
            if i + 1 < len(ladder):
                last = min(last, -(-ladder[i + 1].min_bytes // ROW_BYTES) - 1)
            if first <= last:
                seen.setdefault(rung.cfg, first)
        with torch.cuda.device(self._device_index):
            for rows in seen.values():
                launch, bufoff, cntoff = self._launcher(rows)
                args = self._args(
                    bufoff, cntoff, rows, (0,) * 5, HIDDEN, 0.0,
                    Stream(torch.cuda.current_stream(self._device_index)),
                )  # fmt: skip
                preload_launcher(launch, *args)

    def allreduce_rmsnorm(
        self, inp, residual_in, weight, eps, *, out=None, residual_out=None, stream=None
    ):
        """``(out, residual_out)`` of ``gemma_add_rmsnorm(all_reduce(inp), residual_in)``; only inside a registered
        graph capture (``accepts``)."""
        if not self.accepts(inp, residual_in, weight, out, residual_out):
            raise ValueError(
                f"PullAllReduceRMSNorm serves bf16 [1..{self.max_rows}, {HIDDEN}] calls "
                "recorded inside capture() on a graph capture"
            )
        if out is None:
            out = torch.empty_like(inp)
        if residual_out is None:
            residual_out = torch.empty_like(residual_in)
        rows = inp.shape[0]
        launch, bufoff, cntoff = self._launcher(rows)
        check_launch_outside_capture(launch, owner="PullAllReduceRMSNorm")
        if stream is None:
            stream = Stream(torch.cuda.current_stream(self._device_index))
        elif not isinstance(stream, Stream):
            stream = Stream(stream)
        args = self._args(
            bufoff,
            cntoff,
            rows,
            (
                self._row(inp),
                residual_in.data_ptr(),
                weight.data_ptr(),
                out.data_ptr(),
                residual_out.data_ptr(),
            ),
            residual_in.stride(0),
            eps,
            stream,
        )
        self._launched = True
        with torch.cuda.device(self._device_index):
            _run_compiled(launch, *args)
        return out, residual_out

    @property
    def buf_bytes(self) -> int:
        return K.BUF_BYTES

    def close(self):
        if getattr(self, "_launched", False):
            torch.cuda.synchronize(self._device_index)
            self._launched = False
        for i, base in enumerate(getattr(self, "_peer_bases", [])):
            if base is not None:
                try:
                    IpcHeap.close_mem_handle(int(base))
                except RuntimeError:
                    pass
                self._peer_bases[i] = None
        registry = getattr(self, "_registry", None)
        if registry is not None:
            registry.close()
            self._registry = None
        if getattr(self, "_buf", None):
            try:
                IpcHeap.free_device_mem(self._buf)
            except RuntimeError:
                pass
            self._buf = None

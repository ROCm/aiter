# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""IFOE cross-node custom all-reduce communicator (gfx1250).

Shares peer buffers via HIP *fabric* handles instead of IPC, so the identical
2-stage kernel runs cross-node over the IFOE fabric.  Bootstrap and the
64-byte fabric-handle exchange ride ``torch.distributed`` (works cross-node),
mirroring ``custom_all_reduce.py``'s IPC-handle exchange.

Usage::

    comm = IfoeCustomAllreduce(group, device, max_bytes=1 << 30)
    out = comm.all_reduce(x)                 # fp32, lossless
    out = comm.all_reduce(x, mode="bf16")    # bf16 on the wire (lossy)
    comm.dispose()
"""

import torch
import torch.distributed as dist

from aiter.ops.custom_all_reduce_ifoe import (
    ifoe_alloc_fabric,
    ifoe_import_fabric,
    ifoe_init,
    ifoe_all_reduce,
    ifoe_meta_size,
    ifoe_dispose,
)

_HANDLE_BYTES = 64  # sizeof(hipMemFabricHandle_t)
_MODES = {"fp32": 0, "bf16": 1}


class IfoeCustomAllreduce:
    def __init__(self, group, device, max_bytes: int = 1 << 30):
        """Register fabric buffers of capacity ``max_bytes`` and exchange handles.

        ``group`` is a torch.distributed process group spanning all ranks (any
        backend that supports ``all_gather_object``); ``max_bytes`` bounds the
        largest fp32 tensor that can be all-reduced.
        """
        self.group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device = torch.device(device)
        self.max_bytes = int(max_bytes)
        self.ctx = None
        if self.world < 2 or self.world > 8:
            raise ValueError("IFOE all-reduce supports world size in [2, 8]")

        torch.cuda.set_device(self.device)
        self.sig_bytes = ifoe_meta_size() + self.max_bytes
        self.bf_bytes = self.max_bytes // 2

        # allocate local fabric buffers; each call writes a 64-byte export handle
        self._h_in = torch.zeros(_HANDLE_BYTES, dtype=torch.uint8)
        self._h_sig = torch.zeros(_HANDLE_BYTES, dtype=torch.uint8)
        self._h_bf = torch.zeros(_HANDLE_BYTES, dtype=torch.uint8)
        self.in_ptr = ifoe_alloc_fabric(self.max_bytes, self._h_in.data_ptr())
        self.sig_ptr = ifoe_alloc_fabric(self.sig_bytes, self._h_sig.data_ptr())
        self.bf_ptr = ifoe_alloc_fabric(self.bf_bytes, self._h_bf.data_ptr())

        # all-gather the (input, signal, bf16) handles across the group
        mine = (
            bytes(self._h_in.numpy().tobytes()),
            bytes(self._h_sig.numpy().tobytes()),
            bytes(self._h_bf.numpy().tobytes()),
        )
        gathered = [None] * self.world
        dist.all_gather_object(gathered, mine, group=self.group)

        peer_in, peer_sig, peer_bf = [], [], []
        for r in range(self.world):
            if r == self.rank:
                peer_in.append(self.in_ptr)
                peer_sig.append(self.sig_ptr)
                peer_bf.append(self.bf_ptr)
                continue
            hi, hs, hb = gathered[r]
            peer_in.append(self._import(hi, self.max_bytes))
            peer_sig.append(self._import(hs, self.sig_bytes))
            peer_bf.append(self._import(hb, self.bf_bytes))

        self.ctx = ifoe_init(
            self.rank,
            self.world,
            self.in_ptr,
            self.sig_ptr,
            self.bf_ptr,
            peer_in,
            peer_sig,
            peer_bf,
        )

    @staticmethod
    def _import(handle: bytes, nbytes: int) -> int:
        # import copies the handle immediately, so the staging tensor is transient
        staging = torch.frombuffer(bytearray(handle), dtype=torch.uint8)
        return ifoe_import_fabric(staging.data_ptr(), nbytes)

    def all_reduce(
        self,
        inp: torch.Tensor,
        *,
        mode: str = "fp32",
        out: torch.Tensor | None = None,
        unroll: int = 0,
        blocks: int = 0,
    ) -> torch.Tensor:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of {list(_MODES)}")
        if inp.dtype != torch.float32:
            raise ValueError("IFOE all-reduce currently supports fp32 tensors")
        if not inp.is_contiguous():
            inp = inp.contiguous()
        nbytes = inp.numel() * inp.element_size()
        if nbytes > self.max_bytes:
            raise ValueError(
                f"tensor ({nbytes} B) exceeds max_bytes ({self.max_bytes})"
            )
        if nbytes % 32 != 0:
            raise ValueError("tensor byte size must be a multiple of 32")
        if out is None:
            out = torch.empty_like(inp)
        ifoe_all_reduce(
            self.ctx,
            inp.data_ptr(),
            out.data_ptr(),
            inp.numel(),
            inp.element_size(),
            _MODES[mode],
            unroll,
            blocks,
        )
        return out

    def dispose(self) -> None:
        if self.ctx is not None:
            ifoe_dispose(self.ctx)
            self.ctx = None

    def __del__(self):
        try:
            self.dispose()
        except Exception:
            pass

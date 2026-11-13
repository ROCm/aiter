# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The RCCL baseline: the ``RcclCall``s of an op's ``Traffic`` as one thunk.

Two backends do the same calls:

* ``torch``: ``torch.distributed`` on the script's device group.
* ``pynccl``: aiter's ``PyNcclCommunicator``. This is for graph timing where
  ``torch.distributed`` collectives do not capture. Its all-to-all is a
  grouped send/recv per peer, as RCCL's own ``ncclAllToAll`` is.

The all-reduce input is zeros, so in-place repeats never overflow to inf. RCCL
does not inspect values, so the time does not depend on them.
"""

from __future__ import annotations

import itertools

import torch
from measure import Thunk

from aiter.ops.flydsl.comm_traffic import RcclCall, Traffic

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
BACKENDS = ("torch", "pynccl")


class RcclBaseline:
    def __init__(
        self,
        *,
        device: torch.device,
        group=None,
        pynccl=None,
        rotations: int = 1,
    ):
        if (group is None) == (pynccl is None):
            raise ValueError("pass exactly one of group (torch) and pynccl")
        self.device = device
        self.group = group
        self.pynccl = pynccl
        self.rotations = max(1, int(rotations))
        self.backend = "torch" if pynccl is None else "pynccl"

    def thunk(self, traffic: Traffic) -> Thunk:
        """One thunk doing every RCCL call of *traffic*, in order. With
        ``rotations`` > 1, consecutive calls cycle through that many input
        sets, as the op's own thunk does."""
        sets = [
            [self._buffers(call) for call in traffic.rccl]
            for _ in range(self.rotations)
        ]
        calls = traffic.rccl
        counter = itertools.count()

        def run():
            bufs = sets[next(counter) % len(sets)]
            for call, (inp, out) in zip(calls, bufs):
                self._call(call, inp, out)

        return run

    def _buffers(self, call: RcclCall):
        dtype = DTYPES[call.dtype]
        numel = call.nbytes // torch.empty((), dtype=dtype).itemsize
        if call.collective == "all_reduce":
            inp = torch.zeros(numel, dtype=dtype, device=self.device)
            out = inp if self.pynccl is None else torch.zeros_like(inp)
        else:
            inp = torch.randn(numel, device=self.device).to(dtype)
            out = torch.empty_like(inp)
        return inp, out

    def _call(self, call: RcclCall, inp, out) -> None:
        if self.pynccl is None:
            import torch.distributed as dist

            if call.collective == "all_reduce":
                dist.all_reduce(out, group=self.group)
            else:
                dist.all_to_all_single(out, inp, group=self.group)
            return
        if call.collective == "all_reduce":
            self.pynccl.all_reduce(inp, out)
            return
        n = self.pynccl.world_size
        ins, outs = inp.chunk(n), out.chunk(n)
        self.pynccl.group_start()
        for peer in range(n):
            self.pynccl.send(ins[peer], peer)
            self.pynccl.recv(outs[peer], peer)
        self.pynccl.group_end()

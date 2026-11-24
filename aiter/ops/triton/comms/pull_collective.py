# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pull collectives over XGMI for sequence-parallel prefill: all-gather,
reduce-scatter, all-to-all and an MXFP8 attention-output all-to-all with fused
unpack.

Every rank owns one HIP-IPC payload buffer (``IpcHeap``). The producer of a
collective writes its data straight into that buffer (``buffer()``: a tensor
view, so a GEMM/quant epilogue can target it without a stage copy), then the
collective *pulls*: the destination rank reads the peers' buffers through the
IPC mappings and writes its own output.

Ordering is done on the device, per call, with an epoch the kernels keep
themselves, so every op is stream ordered and can be captured in a CUDA/HIP graph
(the buffer address is fixed):

    acquire()        peers finished reading my buffer from the previous call
    <producer writes the buffer>
    op(...)          publish ready=e+1, wait for the peers' ready, pull, then the
                     last block publishes done=e+1 and bumps the epoch

All ranks must issue the same ops in the same order; a rank that skips one makes
its peers spin. Pulling needs no peer participation beyond the flags, and each
flag is polled by one program per peer (polling from every block of a pull kernel
costs 5-10x in bandwidth).

Reduce-scatter accumulates in fp32 in ascending rank order and rounds once to
bf16 (RCCL's ring rounds to bf16 on every hop), so it is not bit-equal to RCCL
but is bit-equal to the fp32 rank-order reference.

The stage/publish/pull structure is the one used by AMD's internal ``amd_kernels``
bounded-pull collectives; this file is an independent implementation on aiter's
``IpcHeap`` and Triton.
"""

from __future__ import annotations

import types
from collections.abc import Sequence

import torch
import torch.distributed as dist
import triton

from aiter.ops.flydsl.ipc_heap import IpcHeap
from aiter.ops.triton._triton_kernels.comms.pull_collective import (
    _pull_a2a_mxfp8_kernel,
    _pull_copy_kernel,
    _pull_publish_wait_kernel,
    _pull_rs_kernel,
    _pull_rs_persistent_kernel,
    _pull_wait_kernel,
)

__all__ = ["PullCollective"]

_FLAG_BYTES = 4096  # int64 flags: [0] ready, [1] done
_AG_WARPS = 16
_AG_PROGRAMS = (
    192  # programs per copy kernel: more than the CU count loses to a second wave
)
_RS_BLOCK, _RS_WARPS = 4096, 16  # bf16 elements per tile
_RS_PERSISTENT_TILES = 3072  # from here (8192 tokens at H 6144) the loop wins
_A2A_WARPS = 16
_MEMORY_MODES = {"uncached": 0x3, "fine_grained": 0x1, "coarse_grained": 0x0}


def _pow2_ceil(x: int) -> int:
    return 1 << max(0, (int(x) - 1).bit_length())


def _is_pow2(x: int) -> bool:
    return x > 0 and x & (x - 1) == 0


class PullCollective:
    """Pull collectives of one process group (intra-node, one rank per GPU).

    ``max_bytes`` is the largest payload a rank ever stages. ``group`` is any
    group usable for ``all_gather_object`` (gloo is fine); it is used only during
    construction.
    """

    def __init__(
        self,
        group,
        max_bytes: int,
        device: torch.device | None = None,
        memory_mode: str = "fine_grained",
    ):
        if memory_mode not in _MEMORY_MODES:
            raise ValueError(f"memory_mode must be one of {sorted(_MEMORY_MODES)}")
        self._group = group
        self.rank = dist.get_rank(group)
        self.world = dist.get_world_size(group)
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.max_bytes = (int(max_bytes) + 4095) // 4096 * 4096
        # flags are always uncached so polls see the peers' stores
        self._flags = IpcHeap.alloc(_FLAG_BYTES)
        self._data = IpcHeap.alloc(self.max_bytes, flags=_MEMORY_MODES[memory_mode])
        handles = (
            IpcHeap.get_mem_handle_bytes(self._flags),
            IpcHeap.get_mem_handle_bytes(self._data),
        )
        all_handles = IpcHeap.gather_object_list_via_broadcast(group, handles)
        self._flag_peers, self._data_peers = [], []
        for r in range(self.world):
            mine = r == self.rank
            self._flag_peers.append(
                self._flags if mine else IpcHeap.open_mem_handle(all_handles[r][0])
            )
            self._data_peers.append(
                self._data if mine else IpcHeap.open_mem_handle(all_handles[r][1])
            )
        self._flag_ptrs = torch.tensor(
            self._flag_peers, dtype=torch.int64, device=self.device
        )
        self._data_ptrs = torch.tensor(
            self._data_peers, dtype=torch.int64, device=self.device
        )
        self._epoch = torch.zeros(1, dtype=torch.int64, device=self.device)
        self._cnt = torch.zeros(1, dtype=torch.int32, device=self.device)
        dist.barrier(group=group)

    def close(self) -> None:
        # no peer may still pull from my buffer when I free it
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self._group)
        for r in range(self.world):
            if r != self.rank:
                IpcHeap.close_mem_handle(self._flag_peers[r])
                IpcHeap.close_mem_handle(self._data_peers[r])
        IpcHeap.free_device_mem(self._flags)
        IpcHeap.free_device_mem(self._data)

    # ---- producer side -------------------------------------------------
    def buffer(
        self, shape: Sequence[int], dtype: torch.dtype, byte_offset: int = 0
    ) -> torch.Tensor:
        """A tensor over my IPC payload buffer for the producer to write."""
        n = 1
        for s in shape:
            n *= int(s)
        nbytes = n * dtype.itemsize
        if byte_offset < 0 or byte_offset % 8 or byte_offset + nbytes > self.max_bytes:
            raise ValueError("buffer view outside the payload buffer or misaligned")
        holder = types.SimpleNamespace()
        holder.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (self._data + byte_offset, False),
            "version": 3,
        }
        return torch.as_tensor(holder, device=self.device).view(dtype).view(*shape)

    def acquire(self) -> None:
        """Stream-ordered: every peer finished reading my buffer from the last call.
        Call before the producer writes the buffer."""
        _pull_wait_kernel[(self.world,)](
            self._flag_ptrs, self._epoch, FLAG=1, DELTA=0, num_warps=1
        )

    def _publish(self) -> None:
        _pull_publish_wait_kernel[(self.world,)](
            self._flag_ptrs, self._epoch, self.rank, num_warps=1
        )

    # ---- collectives -----------------------------------------------------
    def _plan(self, fields):
        plan, total = [], 0
        for off, nbytes, out in fields:
            if off % 8 or nbytes % 8 or nbytes <= 0 or off + nbytes > self.max_bytes:
                raise ValueError(
                    "field offset and size must be 8-byte aligned and inside the buffer"
                )
            if not out.is_contiguous():
                raise ValueError("outputs must be contiguous")
            if out.numel() * out.element_size() != self.world * nbytes:
                raise ValueError("output must hold world_size times the field bytes")
            n64 = nbytes // 8
            block = min(
                max(_pow2_ceil(triton.cdiv(n64 * self.world, _AG_PROGRAMS)), 8192),
                1 << 18,
            )
            nb = triton.cdiv(n64, block)
            plan.append((off // 8, n64, nb, block, out))
            total += nb * self.world
        return plan, total

    def _pull_copy(self, fields, all_to_all: bool) -> None:
        plan, total = self._plan(fields)
        self._publish()
        for off64, n64, nb, block, out in plan:
            _pull_copy_kernel[(nb, self.world)](
                out,
                self._flag_ptrs,
                self._data_ptrs,
                self._epoch,
                self._cnt,
                off64,
                n64,
                self.rank,
                total,
                W=self.world,
                BLOCK=block,
                ALL_TO_ALL=all_to_all,
                num_warps=_AG_WARPS,
            )

    def all_gather(self, fields) -> None:
        """Gather fields from all ranks.

        ``fields`` has one ``(byte_offset, nbytes, out)`` entry for each field. The
        producer staged ``nbytes`` bytes at ``byte_offset`` of its buffer. ``out``
        receives the bytes of rank 0 to W-1, in rank order. All fields share one
        publish and one wait.
        """
        self._pull_copy(fields, False)

    def all_to_all(self, fields) -> None:
        """Exchange blocks between ranks.

        Like ``all_gather``, but each staged field has W equal blocks. Block j goes
        to rank j. Block r of ``out`` comes from rank r. ``nbytes`` is the size of
        the whole field and a multiple of 8 x world_size.
        """
        blocks = []
        for off, nbytes, out in fields:
            if nbytes % (8 * self.world):
                raise ValueError("field bytes must be a multiple of 8 x world_size")
            blocks.append((off, nbytes // self.world, out))
        self._pull_copy(blocks, True)

    def reduce_scatter(self, out: torch.Tensor, numel: int) -> None:
        """bf16 reduce-scatter of the first ``numel`` elements of my buffer: ``out``
        gets rows [rank*numel/W, (rank+1)*numel/W) summed over ranks in fp32,
        ascending rank order, rounded once."""
        if out.dtype != torch.bfloat16 or not out.is_contiguous():
            raise ValueError("out must be contiguous bf16")
        if numel % self.world or out.numel() != numel // self.world:
            raise ValueError("numel must be W x out.numel()")
        if numel * 2 > self.max_bytes:
            raise ValueError("payload larger than the buffer")
        shard = numel // self.world
        ntiles = triton.cdiv(shard, _RS_BLOCK)
        self._publish()
        if ntiles < _RS_PERSISTENT_TILES:  # short payload: one tile for each program
            _pull_rs_kernel[(ntiles,)](
                out,
                self._flag_ptrs,
                self._data_ptrs,
                self._epoch,
                self._cnt,
                shard,
                self.rank,
                ntiles,
                W=self.world,
                BLOCK=_RS_BLOCK,
                num_warps=_RS_WARPS,
            )
            return
        # long payload: a persistent loop of 192 programs reaches a higher link rate
        nprog = _AG_PROGRAMS
        _pull_rs_persistent_kernel[(nprog,)](
            out,
            self._flag_ptrs,
            self._data_ptrs,
            self._epoch,
            self._cnt,
            shard,
            self.rank,
            nprog,
            ntiles,
            W=self.world,
            BLOCK=_RS_BLOCK,
            num_warps=_RS_WARPS,
        )

    def all_to_all_mxfp8(
        self, out_codes: torch.Tensor, out_scales: torch.Tensor, rows: int, d: int
    ) -> None:
        """MXFP8 attention-output all-to-all with the unpack fused in.

        My buffer holds, from byte 0, codes ``[W*rows, d]`` (1 byte each) and then
        scales ``[W*rows, d/32]`` (1 byte each), written by the producer. Row block
        j goes to rank j. ``out_codes`` ``[rows, W*d]`` and ``out_scales``
        ``[rows, W*d/32]`` (1-byte dtypes) receive peer p's block in column block p.
        ``d/8`` and ``d/256`` must be powers of two (d = 2048, 4096, ...).
        """
        w = self.world
        if not (_is_pow2(d // 8) and _is_pow2(d // 256) and d % 256 == 0):
            raise ValueError("d must be 256 * 2^k")
        if (
            out_codes.numel() != rows * w * d
            or out_scales.numel() != rows * w * d // 32
        ):
            raise ValueError("output sizes do not match rows, d, world_size")
        if not (out_codes.is_contiguous() and out_scales.is_contiguous()):
            raise ValueError("outputs must be contiguous")
        if out_codes.element_size() != 1 or out_scales.element_size() != 1:
            raise ValueError("codes and scales are 1-byte dtypes")
        if w * rows * (d + d // 32) > self.max_bytes:
            raise ValueError("payload larger than the buffer")
        # about 48 programs per peer for the codes, fewer for the small scales
        rows_c = min(max(_pow2_ceil(triton.cdiv(rows, 48)), 16), 256)
        rows_s = min(max(_pow2_ceil(triton.cdiv(rows, 8)), 64), 1024)
        nc = triton.cdiv(rows, rows_c)
        ns = triton.cdiv(rows, rows_s)
        self._publish()
        _pull_a2a_mxfp8_kernel[(nc + ns, w)](
            out_codes,
            out_scales,
            self._flag_ptrs,
            self._data_ptrs,
            self._epoch,
            self._cnt,
            self.rank,
            rows,
            w * rows * d // 8,
            nc,
            (nc + ns) * w,
            W=w,
            D64=d // 8,
            S64=d // 256,
            RC=rows_c,
            RS=rows_s,
            num_warps=_A2A_WARPS,
        )

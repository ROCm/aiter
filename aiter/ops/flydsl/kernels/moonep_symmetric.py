# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Symmetric buffers for MoonEP on either communication backend.

MoonEP's kernels address peers only through an int64 ``[world_size]`` table of
P2P base addresses, so the backend is a host-side choice:

* Mori shmem (gfx950): ``mori_shmem_create_tensor`` + ``shmem_ptr_p2p``.
* cco (gfx1250, where MegaMoE runs on a cco communicator): ``alloc_mem`` +
  ``register_window``, with the table filled on device by ``Window.lsa_ptr``.

The cco communicator is the caller's.  Its ``per_rank_vmm`` must cover every
MoonEP buffer, weight pools included: sum ``MoonEPPlanner.vmm_bytes`` and
``MoonEPWeightPool.vmm_bytes``.
"""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl.expr.typing import Stream

from .tensor_shim import _run_compiled, ptr_buf_tensor


def device_gfx(device: torch.device | None = None) -> str:
    """Arch of the GPU the kernels dispatch on, e.g. ``gfx950``."""
    return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]


def device_wave_size(device: torch.device | None = None) -> int:
    props = torch.cuda.get_device_properties(device)
    warp_size = getattr(props, "warp_size", None)
    if warp_size:
        return int(warp_size)
    return 32 if device_gfx(device).startswith(("gfx10", "gfx11", "gfx12")) else 64


# cco maps each alloc_mem in whole pages of this size (2 MiB on gfx1250).
CCO_ALLOC_GRANULARITY = 2 << 20


def cco_vmm_bytes(*nbytes: int) -> int:
    """``per_rank_vmm`` that the ``alloc_mem`` calls of these sizes occupy."""
    g = CCO_ALLOC_GRANULARITY
    return sum(-(-n // g) * g for n in nbytes)


@functools.cache
def _lsa_table_jit(world_size: int):
    import mori.cco.device.flydsl as cco

    @flyc.kernel(name=f"moonep_lsa_table_n{world_size}", known_block_size=[64, 1, 1])
    def kernel(handle: fx.Int64, addr_out: fx.Int64):
        tid = fx.Int32(fx.thread_idx.x)
        if tid < fx.Int32(world_size):
            out = ptr_buf_tensor(addr_out, fx.Int64)
            out[tid] = fx.Int64(cco.Window(handle).lsa_ptr(tid, 0))

    @flyc.jit
    def launch(
        handle: fx.Int64,
        addr_out: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        kernel(handle, addr_out).launch(grid=(1, 1, 1), block=(64, 1, 1), stream=stream)

    return launch


class MoonEPSymmetric:
    """One collective byte allocation and every peer's address of it.

    Every rank of the group constructs it in the same order with the same size.
    """

    def __init__(
        self,
        nbytes: int,
        *,
        rank: int,
        world_size: int,
        communicator=None,
        group: dist.ProcessGroup | None = None,
    ) -> None:
        """``communicator`` selects cco; without it Mori shmem must be up.

        ``group`` is the EP process group used for host barriers on the cco
        path; it may be omitted only when the EP group is the whole world.
        """
        self.rank, self.world_size = rank, world_size
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._communicator = communicator
        self._group = group
        if communicator is None:
            import mori.shmem as ms
            from mori.shmem import mori_shmem_create_tensor

            self.storage = mori_shmem_create_tensor((nbytes,), torch.uint8)
            self.peer_bases = torch.tensor(
                [
                    ms.shmem_ptr_p2p(self.storage.data_ptr(), rank, pe)
                    for pe in range(world_size)
                ],
                dtype=torch.int64,
                device=self.device,
            )
        else:
            from .mega_moe_gfx1250.types import _from_gpu_ptr

            if communicator.nranks != world_size or communicator.rank != rank:
                raise ValueError(
                    f"cco communicator is rank {communicator.rank} of "
                    f"{communicator.nranks}, expected {rank} of {world_size}"
                )
            if group is None and world_size != dist.get_world_size():
                raise ValueError(
                    f"EP{world_size} inside a world of {dist.get_world_size()} "
                    "ranks: pass the EP process group as group="
                )
            self._memory = communicator.alloc_mem(nbytes)
            self._window = communicator.register_window(self._memory.ptr, nbytes)
            self.storage = _from_gpu_ptr(self._window.local_ptr, (nbytes,), torch.uint8)
            self.peer_bases = torch.empty(
                (world_size,), dtype=torch.int64, device=self.device
            )
            _run_compiled(
                _lsa_table_jit(world_size),
                fx.Int64(self._window.handle),
                fx.Int64(self.peer_bases.data_ptr()),
                fx.Stream(torch.cuda.current_stream(self.device).cuda_stream),
            )
        self.storage.zero_()
        self.barrier()

    def barrier(self) -> None:
        """Device-synchronize, then wait for every rank of the group."""
        torch.cuda.synchronize(self.device)
        if self._communicator is None:
            import mori.shmem as ms

            ms.shmem_barrier_all()
        else:
            dist.barrier(group=self._group)

    def close(self) -> None:
        """Collective.  On cco ``storage`` does not own the memory, so views
        taken from it (e.g. weight windows bound to MegaMoE) die here too."""
        if self.storage is None:
            return
        self.barrier()
        self.storage = self.peer_bases = None
        if self._communicator is not None:
            self._window.close()
            self._memory.close()


__all__ = [
    "CCO_ALLOC_GRANULARITY",
    "MoonEPSymmetric",
    "cco_vmm_bytes",
    "device_gfx",
    "device_wave_size",
]

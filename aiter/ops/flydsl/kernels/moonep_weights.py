# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Symmetric expert-weight pool for MoonEP: home and prefetch slots in one
contiguous allocation, so a grouped GEMM can index both with one local slot id.

Layout per matrix: ``(epn + B, *shape)`` in the mori symmetric heap.
``home = view[:epn]`` (peers prefetch from here over P2P), ``prefetched =
view[epn:]``.  ATOM's ``w1``/``w2`` are ordinary local tensors and are not
reachable by peers, so they must be staged into ``home`` once after loading.

The pool is dtype-transparent: staging and prefetch are byte copies, so it
holds ATOM's expert weights in whatever layout and dtype the experts kernel
already expects -- fp8 slabs shuffled by ``moe_shuffle_weight``, and their
block scales -- without unshuffling or dequantising anything.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor, mori_shmem_free_tensor

from aiter.ops.flydsl.kernels.moonep_weight_prefetch_fast import (
    make_moonep_weight_prefetch_fast_jit,
)


class MoonEPWeightPool:
    """One symmetric ``(epn + B, *shape)`` pool with a prefetch launcher."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        experts_per_rank: int,
        prefetch_slots: int,
        weight_shape: tuple[int, ...],
        dtype: torch.dtype = torch.bfloat16,
        block_num: int = 1024,
        block_threads: int = 256,
    ) -> None:
        numel = 1
        for d in weight_shape:
            numel *= d
        elem_bytes = torch.empty(0, dtype=dtype).element_size()
        if (numel * elem_bytes) % 16 != 0:
            raise ValueError(
                f"expert weight must be 16-byte aligned, got "
                f"{numel * elem_bytes} bytes"
            )

        self.dtype = dtype
        self.elem_bytes = elem_bytes
        self.rank = rank
        self.world_size = world_size
        self.experts_per_rank = experts_per_rank
        self.prefetch_slots = prefetch_slots
        self.weight_shape = tuple(weight_shape)
        self.weight_numel = numel
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._staged = False
        self._closed = False

        # Allocated as raw bytes and viewed, so the symmetric heap never has to
        # know about fp8 or any other narrow dtype.
        slots = experts_per_rank + prefetch_slots
        self._raw = mori_shmem_create_tensor(
            (slots, numel * elem_bytes), torch.uint8
        )
        self._raw.zero_()
        self.pool = self._raw.view(dtype).reshape(slots, *weight_shape)
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()

        self.home = self.pool[:experts_per_rank]
        self.prefetched = self.pool[experts_per_rank:]

        self.peer_home_ptrs = torch.tensor(
            [
                ms.shmem_ptr_p2p(self.pool.data_ptr(), rank, p)
                for p in range(world_size)
            ],
            dtype=torch.int64,
            device=self.device,
        )
        self._jit = make_moonep_weight_prefetch_fast_jit(
            experts_per_rank=experts_per_rank,
            prefetch_slots=prefetch_slots,
            weight_numel=numel,
            elem_bytes=elem_bytes,
            block_num=block_num,
            block_threads=block_threads,
        )
        self._compiled = None

    def stage_home(self, weights: torch.Tensor) -> None:
        """Copy this rank's expert weights into the symmetric home segment."""
        if self._closed:
            raise RuntimeError("weight pool is closed")
        expected = (self.experts_per_rank, *self.weight_shape)
        if tuple(weights.shape) != expected:
            raise ValueError(
                f"expected home weights of shape {expected}, got "
                f"{tuple(weights.shape)}"
            )
        if weights.dtype != self.dtype:
            raise ValueError(
                f"pool holds {self.dtype} but got {weights.dtype}; the pool is "
                "a byte copy and must match the experts kernel's dtype exactly"
            )
        # Copy as bytes. The narrow dtypes this pool carries (fp4x2, e8m0) have
        # no complete elementwise kernel coverage in torch -- fill_ and index
        # both raise on them -- so an elementwise copy_ is not something to
        # rely on, and the pool only ever needs the bytes anyway.
        src = weights.contiguous()
        self._raw[: self.experts_per_rank].copy_(
            src.view(torch.uint8).reshape(self.experts_per_rank, -1)
        )
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        self._staged = True

    def prefetch(self, experts_to_copy_row: torch.Tensor) -> torch.Tensor:
        """Pull the selected remote experts into the prefetch segment."""
        if self._closed:
            raise RuntimeError("weight pool is closed")
        if not self._staged:
            raise RuntimeError(
                "stage_home() must run on every rank before prefetch: peers "
                "read the home segment directly, so an unstaged rank serves "
                "zeros without any error"
            )
        if experts_to_copy_row.dtype != torch.int32:
            raise ValueError("experts_to_copy must be int32")
        stream = torch.cuda.current_stream(self.device)
        sel = experts_to_copy_row.contiguous()
        raw = (
            sel.data_ptr(),
            self.peer_home_ptrs.data_ptr(),
            self.prefetched.data_ptr(),
            stream,
        )
        if self._compiled is None:
            self._compiled = flyc.compile(
                self._jit,
                fx.Int64(raw[0]),
                fx.Int64(raw[1]),
                fx.Int64(raw[2]),
                stream,
            )
        self._compiled(*raw)
        return self.prefetched

    def slot_of(self, group: int, num_experts: int, expert: int) -> int:
        """Local pool slot for a plan group; grouped GEMM indexes ``pool``."""
        if group < num_experts:
            return expert % self.experts_per_rank
        return self.experts_per_rank + (group - num_experts)

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        mori_shmem_free_tensor(self._raw)
        self._closed = True


__all__ = ["MoonEPWeightPool"]

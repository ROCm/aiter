# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Row-contiguous expert-weight pool for MoonEP: the whole world's experts plus
this rank's prefetch slots in one virtual range, indexed by global row.

Layout per matrix: ``((R + 1) * epn_padded, *shape)`` over HIP VMM
(``aiter.ops.flydsl.moonep_vmm_pool.MoonEPVmmPool``)::

    slot 0 .. R-1   rank pe's home experts   row = pe * epn_padded + k
    slot R          this rank's B prefetch slots, past row E

Only ``epn_padded + B`` rows per rank are physically resident; rows ``[0, E)``
that belong to other ranks are *their* memory, mapped here over XGMI.  A grouped
GEMM therefore reaches **every** expert by row index in a single call, and an
expert that got no prefetch slot is still addressable at its home row -- so
``B`` is a cache size, not a correctness bound.

ATOM's ``w1``/``w2`` are ordinary local tensors and are not reachable by peers,
so they must be staged into ``home`` once after loading.

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

from aiter.ops.flydsl.kernels.moonep_weight_prefetch_fast import (
    make_moonep_weight_prefetch_fast_jit,
)
from aiter.ops.flydsl.moonep_vmm_pool import MoonEPVmmPool


class MoonEPWeightPool:
    """One row-contiguous ``((R+1)*epn_padded, *shape)`` pool + prefetch launcher."""

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

        # Allocated as raw bytes and viewed, so the pool never has to know about
        # fp8 or any other narrow dtype.
        self._vmm = MoonEPVmmPool(
            row_bytes=numel * elem_bytes,
            experts_per_rank=experts_per_rank,
            prefetch_slots=prefetch_slots,
            rank=rank,
            world_size=world_size,
            device=self.device,
        )
        self.epn_padded = self._vmm.epn_padded
        self.rows = self._vmm.rows
        self._raw = self._vmm.tensor(torch.uint8, (numel * elem_bytes,))
        self.pool = self._vmm.tensor(dtype, tuple(weight_shape))

        self._home0 = self._vmm.home_row_begin
        self._pf0 = self.world_size * self.epn_padded
        self.home = self.pool[self._home0 : self._home0 + experts_per_rank]
        self.prefetched = self.pool[self._pf0 : self._pf0 + prefetch_slots]
        # Zero only what we own -- the rest of the range is peer memory and
        # clearing it would wipe their weights. Through the byte view, never the
        # typed one: the narrow dtypes this pool carries (fp4x2, e8m0) have no
        # fill_ kernel in torch and zero_() on them raises NotImplementedError.
        self._raw[self._home0 : self._home0 + experts_per_rank].zero_()
        self._raw[self._pf0 : self._pf0 + prefetch_slots].zero_()
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()

        self._jit = make_moonep_weight_prefetch_fast_jit(
            experts_per_rank=experts_per_rank,
            experts_per_rank_padded=self.epn_padded,
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
        lo = self._home0
        self._raw[lo : lo + self.experts_per_rank].copy_(
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
        # One base for the whole world now: the kernel turns a global expert id
        # into a row itself, so there is no per-owner pointer table to pass.
        raw = (
            sel.data_ptr(),
            self.pool.data_ptr(),
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
        """Pool **row** for a plan group; the grouped GEMM indexes ``pool``.

        Home groups map to the expert's global row -- identical on every rank --
        rather than to a rank-local slot, which is what lets one ``fused_moe``
        call span the whole range.  Migration groups map to the prefetch tail.
        """
        if group < num_experts:
            return self._vmm.global_row(expert)
        return self._vmm.prefetch_row(group - num_experts)

    def close(self) -> None:
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        # Drop the views before the mapping they borrow from.
        self.home = self.prefetched = self.pool = self._raw = None
        self._vmm.close()
        self._closed = True


__all__ = ["MoonEPWeightPool"]

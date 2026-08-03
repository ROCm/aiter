# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Remote expert-weight prefetch for the MoonEP gfx950 prototype."""

from __future__ import annotations

from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
from flydsl.expr import T
from flydsl.expr.typing import Stream
from mori.shmem import mori_shmem_create_tensor, mori_shmem_free_tensor

from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)


def make_moonep_weight_prefetch_jit(
    *,
    experts_per_rank: int,
    prefetch_slots: int,
    weight_numel: int,
    block_num: int = 128,
    block_threads: int = 256,
):
    """Build a BF16 peer-load kernel driven by ``experts_to_copy[rank]``."""

    if experts_per_rank <= 0 or prefetch_slots <= 0:
        raise ValueError("expert and slot counts must be positive")
    if weight_numel <= 0 or weight_numel % 8 != 0:
        raise ValueError("BF16 expert weight size must be divisible by 8 elements")
    if block_num <= 0 or block_threads <= 0:
        raise ValueError("launch geometry must be positive")

    weight_bytes = weight_numel * 2
    weight_i32 = weight_bytes // 4
    total_i32 = prefetch_slots * weight_i32
    global_threads = block_num * block_threads
    name = (
        f"moonep_weight_prefetch_epr{experts_per_rank}_b{prefetch_slots}"
        f"_n{weight_numel}_g{block_num}_t{block_threads}"
    )

    @flyc.kernel(name=name, known_block_size=[block_threads, 1, 1])
    def prefetch_kernel(
        addr_experts_to_copy: fx.Int64,  # INT32 [B], global expert ids
        addr_peer_home_weight_ptrs: fx.Int64,  # INT64 [world_size]
        addr_prefetched_weights: fx.Int64,  # BF16 [B, weight_numel]
    ):
        global_thread = fx.block_idx.x * block_threads + fx.thread_idx.x
        experts_rsrc = create_buffer_resource_from_addr(addr_experts_to_copy)
        peer_ptrs_rsrc = create_buffer_resource_from_addr(
            addr_peer_home_weight_ptrs
        )
        dst_rsrc = create_buffer_resource_from_addr(addr_prefetched_weights)

        # One work item moves 16 B (vec4 i32).  Slots containing -1 are unused
        # by cu_seqlens and are intentionally left untouched.
        for i32_off in range(global_thread * 4, total_i32, global_threads * 4):
            slot = i32_off // weight_i32
            expert = buffer_load(
                experts_rsrc, slot, vec_width=1, dtype=T.i32()
            )
            if expert >= 0:
                owner = expert // experts_per_rank
                local_expert = expert % experts_per_rank
                owner_base = buffer_load(
                    peer_ptrs_rsrc, owner, vec_width=1, dtype=T.i64()
                )
                src_addr = (
                    owner_base + fx.Int64(local_expert) * weight_bytes
                )
                src_rsrc = create_buffer_resource_from_addr(src_addr)
                expert_i32_off = i32_off - slot * weight_i32
                value = buffer_load(
                    src_rsrc,
                    expert_i32_off,
                    vec_width=4,
                    dtype=T.i32(),
                )
                buffer_store(value, dst_rsrc, i32_off)

    @flyc.jit
    def launch(
        addr_experts_to_copy: fx.Int64,
        addr_peer_home_weight_ptrs: fx.Int64,
        addr_prefetched_weights: fx.Int64,
        stream: Stream = Stream(None),  # noqa: B008
    ):
        prefetch_kernel(
            addr_experts_to_copy,
            addr_peer_home_weight_ptrs,
            addr_prefetched_weights,
        ).launch(
            grid=(block_num, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


class MoonEPWeightPrefetchOp:
    """Own symmetric home weights and local dynamic-expert weight slots.

    This is a bring-up ownership model: each rank copies its resident expert
    weights into a symmetric allocation once, then any peer can directly load
    the selected expert into a local prefetch slot.  Production integration can
    replace that allocation with model-owned VMM/peer mappings later.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        num_experts: int,
        prefetch_slots: int,
        weight_shape: tuple[int, ...],
        block_num: int = 128,
        block_threads: int = 256,
    ) -> None:
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid rank/world_size")
        if num_experts <= 0 or num_experts % world_size != 0:
            raise ValueError("num_experts must be divisible by world_size")
        if prefetch_slots <= 0 or not weight_shape:
            raise ValueError("prefetch slots and weight shape must be non-empty")
        if torch.cuda.current_device() != rank:
            raise ValueError("current ROCm device must match rank")
        if ms.shmem_mype() != rank or ms.shmem_npes() != world_size:
            raise ValueError("MORI SHMEM rank/world_size mismatch")

        weight_numel = 1
        for dim in weight_shape:
            if dim <= 0:
                raise ValueError("weight dimensions must be positive")
            weight_numel *= dim
        if weight_numel % 8 != 0:
            raise ValueError("BF16 expert weight size must be 16-byte aligned")

        self.rank = rank
        self.world_size = world_size
        self.num_experts = num_experts
        self.experts_per_rank = num_experts // world_size
        self.prefetch_slots = prefetch_slots
        self.weight_shape = weight_shape
        self.weight_numel = weight_numel
        self.block_num = block_num
        self.block_threads = block_threads
        self.device = torch.device("cuda", rank)

        self.home_weights = mori_shmem_create_tensor(
            (self.experts_per_rank, *weight_shape), torch.bfloat16
        )
        self.prefetched_weights = torch.empty(
            (prefetch_slots, *weight_shape),
            dtype=torch.bfloat16,
            device=self.device,
        )
        ms.shmem_barrier_all()

        self.peer_home_weight_ptrs = torch.empty(
            world_size, dtype=torch.int64, device=self.device
        )
        for peer in range(world_size):
            self.peer_home_weight_ptrs[peer] = ms.shmem_ptr_p2p(
                self.home_weights.data_ptr(), rank, peer
            )

        self._jit = make_moonep_weight_prefetch_jit(
            experts_per_rank=self.experts_per_rank,
            prefetch_slots=prefetch_slots,
            weight_numel=weight_numel,
            block_num=block_num,
            block_threads=block_threads,
        )
        self._compiled: Any | None = None
        self._closed = False

    def load_home_weights(self, weights: torch.Tensor) -> None:
        """Collectively publish every rank's resident expert weights."""

        if self._closed:
            raise RuntimeError("weight prefetch op is closed")
        if tuple(weights.shape) != tuple(self.home_weights.shape):
            raise ValueError("home weight shape mismatch")
        if weights.dtype != torch.bfloat16 or weights.device != self.device:
            raise ValueError("home weights must be BF16 on this ROCm device")
        self.home_weights.copy_(weights)
        ms.shmem_barrier_on_stream(torch.cuda.current_stream(self.device))

    def prefetch(
        self,
        experts_to_copy: torch.Tensor,
    ) -> torch.Tensor:
        """Prefetch selected experts on the current stream.

        Dispatch and prefetch intentionally share the caller's current stream;
        this prototype does not introduce a second PyTorch stream contract.
        """

        if self._closed:
            raise RuntimeError("weight prefetch op is closed")
        expected = (self.world_size, self.prefetch_slots)
        if tuple(experts_to_copy.shape) != expected:
            raise ValueError(f"experts_to_copy must have shape {expected}")
        if experts_to_copy.dtype != torch.int32 or not experts_to_copy.is_contiguous():
            raise ValueError("experts_to_copy must be contiguous INT32")
        if experts_to_copy.device != self.device:
            raise ValueError("experts_to_copy must be on this ROCm device")
        stream = torch.cuda.current_stream(self.device)

        local_selection = experts_to_copy[self.rank]
        args = (
            fx.Int64(local_selection.data_ptr()),
            fx.Int64(self.peer_home_weight_ptrs.data_ptr()),
            fx.Int64(self.prefetched_weights.data_ptr()),
            stream,
        )
        if self._compiled is None:
            self._compiled = flyc.compile(self._jit, *args)
        else:
            self._compiled(
                local_selection.data_ptr(),
                self.peer_home_weight_ptrs.data_ptr(),
                self.prefetched_weights.data_ptr(),
                stream,
            )
        return self.prefetched_weights

    def close(self) -> None:
        """Collectively release the symmetric resident-weight allocation."""

        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        mori_shmem_free_tensor(self.home_weights)
        self._closed = True


__all__ = ["MoonEPWeightPrefetchOp", "make_moonep_weight_prefetch_jit"]

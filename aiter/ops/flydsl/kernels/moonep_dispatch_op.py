# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Isolated MORI wrapper for MoonEP preplanned direct-P2P dispatch."""

from __future__ import annotations

from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
from mori.shmem import mori_shmem_create_tensor, mori_shmem_free_tensor

from aiter.ops.flydsl.moonep import MoonEPPlanConfig, MoonEPReferencePlan

from .moonep_dispatch import make_moonep_preplanned_dispatch_jit
from .moonep_dispatch_epilogue import make_moonep_dispatch_epilogue_jit
from .moonep_dispatch_prefetch import make_moonep_dispatch_prefetch_jit
from .moonep_combine import make_moonep_combine_jit


class MoonEPPreplannedDispatchOp:
    """Own the symmetric receive shard used by the gfx950 dispatch PoC.

    Construction is collective: every PE must construct the object in the same
    allocation order after MORI SHMEM initialization.  ``dispatch`` is also a
    collective operation because it appends ``shmem_barrier_on_stream`` after
    the direct peer writes.

    The returned buffers have already passed the local duplicate-expansion and
    padding-zero epilogue; rows outside the plan's cu_seqlens remain undefined.
    """

    def __init__(
        self,
        config: MoonEPPlanConfig,
        hidden_dim: int,
        *,
        block_num: int = 128,
        warp_num_per_block: int = 4,
    ) -> None:
        if hidden_dim <= 0 or hidden_dim % 8 != 0:
            raise ValueError("hidden_dim must be positive and divisible by 8")
        if block_num <= 0 or warp_num_per_block <= 0:
            raise ValueError("launch geometry must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("MoonEPPreplannedDispatchOp requires a ROCm device")

        device_index = torch.cuda.current_device()
        if device_index != config.rank:
            raise ValueError(
                f"current ROCm device is {device_index}, but config.rank={config.rank}"
            )
        if ms.shmem_mype() != config.rank or ms.shmem_npes() != config.world_size:
            raise ValueError("MORI SHMEM rank/world_size do not match the plan config")

        self.config = config
        self.hidden_dim = hidden_dim
        self.block_num = block_num
        self.warp_num_per_block = warp_num_per_block
        self.device = torch.device("cuda", device_index)

        rows = config.num_dispatch_rows
        # These allocations must be symmetric and appear in the same order on
        # every PE.  They are the local shard peers write into directly.
        self.recv_hidden = mori_shmem_create_tensor(
            (rows, hidden_dim), torch.bfloat16
        )
        self.recv_route_weights = mori_shmem_create_tensor(
            (rows,), torch.float32
        )
        self.recv_duplicate_src = mori_shmem_create_tensor((rows,), torch.int32)
        self.expert_output = mori_shmem_create_tensor(
            (rows, hidden_dim), torch.bfloat16
        )
        self.recv_duplicate_src.fill_(-1)
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()

        self.peer_hidden_ptrs = torch.empty(
            config.world_size, dtype=torch.int64, device=self.device
        )
        self.peer_weight_ptrs = torch.empty(
            config.world_size, dtype=torch.int64, device=self.device
        )
        self.peer_duplicate_src_ptrs = torch.empty(
            config.world_size, dtype=torch.int64, device=self.device
        )
        self.peer_expert_output_ptrs = torch.empty(
            config.world_size, dtype=torch.int64, device=self.device
        )
        for peer in range(config.world_size):
            self.peer_hidden_ptrs[peer] = ms.shmem_ptr_p2p(
                self.recv_hidden.data_ptr(), config.rank, peer
            )
            self.peer_weight_ptrs[peer] = ms.shmem_ptr_p2p(
                self.recv_route_weights.data_ptr(), config.rank, peer
            )
            self.peer_duplicate_src_ptrs[peer] = ms.shmem_ptr_p2p(
                self.recv_duplicate_src.data_ptr(), config.rank, peer
            )
            self.peer_expert_output_ptrs[peer] = ms.shmem_ptr_p2p(
                self.recv_hidden.data_ptr(), config.rank, peer
            )

        self._jit = make_moonep_preplanned_dispatch_jit(
            hidden_dim=hidden_dim,
            top_k=config.top_k,
            num_dispatch_rows=rows,
            block_num=block_num,
            warp_num_per_block=warp_num_per_block,
        )
        self._compiled: Any | None = None
        self._epilogue_jit = make_moonep_dispatch_epilogue_jit(
            hidden_dim=hidden_dim,
            num_dispatch_rows=rows,
            num_groups=config.num_experts + int(config.prefetch_slots),
            block_num=block_num,
            warp_num_per_block=warp_num_per_block,
        )
        self._epilogue_compiled: Any | None = None
        self._combined_jit: Any | None = None
        self._combined_compiled: Any | None = None
        self._combined_key: tuple[int, int, int] | None = None
        self.combine_output = torch.empty(
            (config.num_tokens, hidden_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self.gathered_route_weights = torch.empty(
            (config.num_tokens, config.top_k),
            dtype=torch.float32,
            device=self.device,
        )
        self._combine_jit = make_moonep_combine_jit(
            num_tokens=config.num_tokens,
            hidden_dim=hidden_dim,
            top_k=config.top_k,
            num_dispatch_rows=rows,
        )
        self._combine_compiled: Any | None = None
        self._closed = False

    def _check_inputs(
        self,
        hidden: torch.Tensor,
        route_weights: torch.Tensor,
        plan: MoonEPReferencePlan,
    ) -> None:
        cfg = self.config
        if plan.config != cfg:
            raise ValueError("plan config does not match this dispatch op")
        if tuple(hidden.shape) != (cfg.num_tokens, self.hidden_dim):
            raise ValueError(
                f"hidden must have shape {(cfg.num_tokens, self.hidden_dim)}"
            )
        if tuple(route_weights.shape) != (cfg.num_tokens, cfg.top_k):
            raise ValueError(
                f"route_weights must have shape {(cfg.num_tokens, cfg.top_k)}"
            )
        if tuple(plan.dst.shape) != (cfg.num_tokens, cfg.top_k):
            raise ValueError("plan.dst has the wrong shape")
        if hidden.dtype != torch.bfloat16:
            raise TypeError("hidden must be BF16")
        if route_weights.dtype != torch.float32:
            raise TypeError("route_weights must be FP32")
        if plan.dst.dtype != torch.int32:
            raise TypeError("plan.dst must be INT32")
        if not hidden.is_contiguous():
            raise ValueError("hidden must be contiguous")
        if not route_weights.is_contiguous() or not plan.dst.is_contiguous():
            raise ValueError("route_weights and plan.dst must be contiguous")
        if (
            hidden.device != self.device
            or route_weights.device != self.device
            or plan.dst.device != self.device
        ):
            raise ValueError("dispatch inputs must be on this op's ROCm device")

    def dispatch(
        self,
        hidden: torch.Tensor,
        route_weights: torch.Tensor,
        plan: MoonEPReferencePlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter according to ``plan.dst`` and enqueue a completion barrier.

        All PEs must call this method in the same collective order.  The MORI
        device barrier is enqueued on the current stream after the FlyDSL
        kernel; subsequent work on that stream can consume peer writes safely.
        """

        if self._closed:
            raise RuntimeError("dispatch op is closed")
        self._check_inputs(hidden, route_weights, plan)
        stream = torch.cuda.current_stream(self.device)
        compile_args = (
            fx.Int64(hidden.data_ptr()),
            fx.Int64(route_weights.data_ptr()),
            fx.Int64(plan.dst.data_ptr()),
            fx.Int64(self.peer_hidden_ptrs.data_ptr()),
            fx.Int64(self.peer_weight_ptrs.data_ptr()),
            fx.Int64(self.peer_duplicate_src_ptrs.data_ptr()),
            self.config.num_tokens,
            stream,
        )
        if self._compiled is None:
            self._compiled = flyc.compile(self._jit, *compile_args)
        else:
            self._compiled(
                hidden.data_ptr(),
                route_weights.data_ptr(),
                plan.dst.data_ptr(),
                self.peer_hidden_ptrs.data_ptr(),
                self.peer_weight_ptrs.data_ptr(),
                self.peer_duplicate_src_ptrs.data_ptr(),
                self.config.num_tokens,
                stream,
            )

        ms.shmem_barrier_on_stream(stream)
        self._run_epilogue(plan, stream)
        return self.recv_hidden, self.recv_route_weights

    def _run_epilogue(
        self,
        plan: MoonEPReferencePlan,
        stream: torch.cuda.Stream,
    ) -> None:
        epilogue_args = (
            fx.Int64(self.recv_hidden.data_ptr()),
            fx.Int64(self.recv_route_weights.data_ptr()),
            fx.Int64(self.recv_duplicate_src.data_ptr()),
            fx.Int64(plan.zero_fill_ranges.data_ptr()),
            stream,
        )
        if self._epilogue_compiled is None:
            self._epilogue_compiled = flyc.compile(
                self._epilogue_jit, *epilogue_args
            )
        else:
            self._epilogue_compiled(
                self.recv_hidden.data_ptr(),
                self.recv_route_weights.data_ptr(),
                self.recv_duplicate_src.data_ptr(),
                plan.zero_fill_ranges.data_ptr(),
                stream,
            )

    def dispatch_and_prefetch(
        self,
        hidden: torch.Tensor,
        route_weights: torch.Tensor,
        plan: MoonEPReferencePlan,
        weight_prefetch_op: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run token scatter and remote weight prefetch in one kernel launch.

        The launch contains two disjoint CTA ranges on the current stream.  It
        has no second-stream contract: dispatch CTAs and prefetch CTAs become
        concurrently schedulable only because they belong to the same grid.
        """

        if self._closed:
            raise RuntimeError("dispatch op is closed")
        self._check_inputs(hidden, route_weights, plan)
        cfg = self.config
        if getattr(weight_prefetch_op, "_closed", True):
            raise ValueError("weight prefetch op is closed or invalid")
        if (
            weight_prefetch_op.rank != cfg.rank
            or weight_prefetch_op.world_size != cfg.world_size
            or weight_prefetch_op.num_experts != cfg.num_experts
            or weight_prefetch_op.prefetch_slots != int(cfg.prefetch_slots)
        ):
            raise ValueError("dispatch and weight-prefetch configurations differ")
        expected_threads = self.warp_num_per_block * 64
        if weight_prefetch_op.block_threads != expected_threads:
            raise ValueError(
                "combined launch requires prefetch block_threads == "
                "dispatch warp_num_per_block * 64"
            )

        combined_key = (
            weight_prefetch_op.weight_numel,
            weight_prefetch_op.block_num,
            weight_prefetch_op.block_threads,
        )
        if self._combined_key is not None and self._combined_key != combined_key:
            raise ValueError("one dispatch op cannot change combined weight geometry")
        if self._combined_jit is None:
            self._combined_key = combined_key
            self._combined_jit = make_moonep_dispatch_prefetch_jit(
                hidden_dim=self.hidden_dim,
                top_k=cfg.top_k,
                num_dispatch_rows=cfg.num_dispatch_rows,
                dispatch_block_num=self.block_num,
                warp_num_per_block=self.warp_num_per_block,
                experts_per_rank=weight_prefetch_op.experts_per_rank,
                prefetch_slots=weight_prefetch_op.prefetch_slots,
                weight_numel=weight_prefetch_op.weight_numel,
                prefetch_block_num=weight_prefetch_op.block_num,
            )

        stream = torch.cuda.current_stream(self.device)
        local_selection = plan.experts_to_copy[cfg.rank]
        args = (
            fx.Int64(hidden.data_ptr()),
            fx.Int64(route_weights.data_ptr()),
            fx.Int64(plan.dst.data_ptr()),
            fx.Int64(self.peer_hidden_ptrs.data_ptr()),
            fx.Int64(self.peer_weight_ptrs.data_ptr()),
            fx.Int64(self.peer_duplicate_src_ptrs.data_ptr()),
            cfg.num_tokens,
            fx.Int64(local_selection.data_ptr()),
            fx.Int64(weight_prefetch_op.peer_home_weight_ptrs.data_ptr()),
            fx.Int64(weight_prefetch_op.prefetched_weights.data_ptr()),
            stream,
        )
        if self._combined_compiled is None:
            self._combined_compiled = flyc.compile(self._combined_jit, *args)
        else:
            self._combined_compiled(
                hidden.data_ptr(),
                route_weights.data_ptr(),
                plan.dst.data_ptr(),
                self.peer_hidden_ptrs.data_ptr(),
                self.peer_weight_ptrs.data_ptr(),
                self.peer_duplicate_src_ptrs.data_ptr(),
                cfg.num_tokens,
                local_selection.data_ptr(),
                weight_prefetch_op.peer_home_weight_ptrs.data_ptr(),
                weight_prefetch_op.prefetched_weights.data_ptr(),
                stream,
            )

        ms.shmem_barrier_on_stream(stream)
        self._run_epilogue(plan, stream)
        return (
            self.recv_hidden,
            self.recv_route_weights,
            weight_prefetch_op.prefetched_weights,
        )

    def combine(
        self,
        plan: MoonEPReferencePlan,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather expert outputs from their execution ranks and top-k reduce.

        Expert outputs must be staged in the symmetric ``recv_hidden`` shard.
        The device barrier publishes every rank's writes before direct peer
        loads begin on the same current stream.
        """

        if self._closed:
            raise RuntimeError("dispatch op is closed")
        if plan.config != self.config:
            raise ValueError("plan config does not match this combine op")
        if plan.dst.device != self.device:
            raise ValueError("combine inputs must be on this op's ROCm device")

        stream = torch.cuda.current_stream(self.device)
        ms.shmem_barrier_on_stream(stream)
        args = (
            fx.Int64(plan.dst.data_ptr()),
            fx.Int64(self.peer_expert_output_ptrs.data_ptr()),
            fx.Int64(self.peer_weight_ptrs.data_ptr()),
            fx.Int64(self.combine_output.data_ptr()),
            fx.Int64(self.gathered_route_weights.data_ptr()),
            stream,
        )
        if self._combine_compiled is None:
            self._combine_compiled = flyc.compile(self._combine_jit, *args)
        else:
            self._combine_compiled(
                plan.dst.data_ptr(),
                self.peer_expert_output_ptrs.data_ptr(),
                self.peer_weight_ptrs.data_ptr(),
                self.combine_output.data_ptr(),
                self.gathered_route_weights.data_ptr(),
                stream,
            )
        return self.combine_output, self.gathered_route_weights

    def close(self) -> None:
        """Collectively release the symmetric receive buffers."""

        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        mori_shmem_free_tensor(self.recv_hidden)
        mori_shmem_free_tensor(self.recv_route_weights)
        mori_shmem_free_tensor(self.recv_duplicate_src)
        mori_shmem_free_tensor(self.expert_output)
        self._closed = True


__all__ = ["MoonEPPreplannedDispatchOp"]

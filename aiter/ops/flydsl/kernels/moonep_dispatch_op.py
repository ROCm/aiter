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


class MoonEPPreplannedDispatchOp:
    """Own the symmetric receive shard used by the gfx950 dispatch PoC.

    Construction is collective: every PE must construct the object in the same
    allocation order after MORI SHMEM initialization.  ``dispatch`` is also a
    collective operation because it appends ``shmem_barrier_on_stream`` after
    the direct peer writes.

    The returned buffers contain directly scattered real route entries only.
    Segment padding and negative-encoded duplicate hidden rows are intentionally
    left for the next local epilogue milestone.
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
        return self.recv_hidden, self.recv_route_weights

    def close(self) -> None:
        """Collectively release the symmetric receive buffers."""

        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        ms.shmem_barrier_all()
        mori_shmem_free_tensor(self.recv_hidden)
        mori_shmem_free_tensor(self.recv_route_weights)
        mori_shmem_free_tensor(self.recv_duplicate_src)
        self._closed = True


__all__ = ["MoonEPPreplannedDispatchOp"]

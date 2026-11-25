# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of o_proj + TP all-reduce + residual + Gemma RMSNorm in one
persistent launch (gfx942/gfx950): mega paged attention's back half, fed by
the unfused ``pa_decode`` and per-token FP8 quant.

Per rank, for ``rows`` decode tokens:

    y_r = bf16(x * x_scale @ (W_r * w_scale_r)^T)     # this rank's o_proj partial
    acc = float(bf16(sum_r y_r)) + residual           # all-reduce, rank order
    residual_out = bf16(acc)
    out = bf16(acc * rsqrt(mean(acc^2) + eps) * (1 + norm_weight))

``x`` is either the quantizer's FP8 [rows, K] with per-row scales, or (an
``x_dtype`` of bf16) the attention output [rows, K], which the kernel quantizes
per token itself as ``dynamic_per_token_scaled_quant`` would. ``W`` is the FP8
o_proj weight [hidden, K] as ``shuffle_weight(layout=(16, 16))`` leaves it,
with per-output-channel scales.

Contract, as mega paged attention's: construction and ``__call__`` are
collectives over ``group``; the kernel's CTAs and ranks spin on each other, so
it needs the whole GPU (no kernels on other streams or processes next to it).
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist

from aiter.utility import dtypes

from .ipc_heap import IpcHeap
from .kernels.oproj_allreduce_rmsnorm import (
    SENTINEL_FILL_BYTE,
    TRACE_STAMPS,
    compile_oproj_allreduce_rmsnorm,
)
from .kernels.tensor_shim import _run_compiled, ptr_arg, torch_to_fx_dtype

__all__ = ["OprojAllreduceRmsnorm"]


class OprojAllreduceRmsnorm:
    """One TP rank's fused o_proj + all-reduce + RMSNorm for one layer shape."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device,
        row_k: int,
        max_rows: int,
        hidden: int,
        eps: float,
        x_dtype: torch.dtype,
        group=None,
        block_n: int = 48,
        prefetch: int | None = None,
        occupancy: int = 2,
        two_shot: bool = True,
        grid: int | None = None,
        trace: bool = False,
    ):
        if world_size > 1 and group is None:
            raise ValueError("world_size > 1 needs the TP process group")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.device = torch.device(device)
        self.max_rows = int(max_rows)
        self.row_k = int(row_k)
        self.hidden = int(hidden)
        self.eps = float(eps)
        self.x_dtype = x_dtype
        self.quantize = x_dtype == torch.bfloat16
        if not self.quantize and x_dtype != dtypes.fp8:
            raise ValueError(f"x_dtype must be bf16 or {dtypes.fp8}, got {x_dtype}")
        with torch.cuda.device(self.device):
            num_cu = torch.cuda.get_device_properties(self.device).multi_processor_count
            self._compiled = compile_oproj_allreduce_rmsnorm(
                row_k=row_k,
                max_rows=max_rows,
                num_compute_units=num_cu,
                hidden=hidden,
                world_size=world_size,
                rank=rank,
                block_n=block_n,
                prefetch=prefetch,
                occupancy=occupancy,
                two_shot=two_shot,
                grid=grid,
                quantize=self.quantize,
                fp8_max=float(torch.finfo(dtypes.fp8).max),
                trace=trace,
            )
        c = self._compiled
        dev = self.device
        with torch.cuda.device(dev):
            # x in MFMA-order chunks (and, quantizing, its row scales), handed
            # between CTAs past the caches.
            self._xs = IpcHeap.alloc(
                max_rows * row_k, IpcHeap._HIP_DEVICE_MALLOC_UNCACHED
            )
            self._x_scale = (
                IpcHeap.alloc(max_rows * 4, IpcHeap._HIP_DEVICE_MALLOC_UNCACHED)
                if self.quantize
                else 0
            )
        # Per-CTA phase stamps (``trace``), 100 MHz ticks.
        self.trace = torch.zeros(
            c["grid"] if trace else 1, TRACE_STAMPS, dtype=torch.int64, device=dev
        )
        self._sync = torch.zeros(c["sync_words"], dtype=torch.int32, device=dev)
        self._peer_bases = []
        with torch.cuda.device(dev):
            self._inbox = IpcHeap.alloc(
                c["inbox_bytes"],
                IpcHeap._HIP_DEVICE_MALLOC_UNCACHED,
                fill=SENTINEL_FILL_BYTE,
            )
            peer_ptrs = [self._inbox] * self.world_size
            if self.world_size > 1:
                handles = IpcHeap.gather_object_list_via_broadcast(
                    group, IpcHeap.get_mem_handle_bytes(self._inbox)
                )
                for r, handle in enumerate(handles):
                    if r != self.rank:
                        base = IpcHeap.open_mem_handle(bytes(handle))
                        self._peer_bases.append(base)
                        peer_ptrs[r] = base
                dist.barrier(group=group)
        self._peers = torch.tensor(peer_ptrs, dtype=torch.int64, device=dev)

    def __call__(
        self,
        x: torch.Tensor,  # [rows, row_k] FP8 or bf16 (quantized here), row-major
        x_scale: torch.Tensor | None,  # FP8 x: [rows] or [rows, 1] fp32
        w: torch.Tensor,  # [hidden, row_k] FP8, shuffled
        w_scale: torch.Tensor,  # [hidden] fp32
        residual: torch.Tensor,  # [rows, hidden] bf16
        norm_weight: torch.Tensor,  # [hidden] bf16 (Gemma: the gain is 1 + w)
        out: torch.Tensor | None = None,
        residual_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(x.shape[0])
        if not 0 < rows <= self.max_rows:
            raise ValueError(f"rows must be in [1, {self.max_rows}], got {rows}")
        if x.dtype != self.x_dtype or tuple(x.shape) != (rows, self.row_k):
            raise ValueError(
                f"x must be {self.x_dtype} [rows, {self.row_k}], got {x.dtype} {tuple(x.shape)}"
            )
        if self.quantize:
            if x_scale is not None:
                raise ValueError("bf16 x is quantized here: x_scale must be None")
        elif (
            x_scale is None or x_scale.numel() != rows or x_scale.dtype != torch.float32
        ):
            raise ValueError("x_scale must be fp32 with one scale a row")
        if tuple(w.shape) != (self.hidden, self.row_k):
            raise ValueError(
                f"w must be [{self.hidden}, {self.row_k}], got {tuple(w.shape)}"
            )
        if tuple(residual.shape) != (rows, self.hidden):
            raise ValueError(f"residual must be [{rows}, {self.hidden}]")
        for name, t in (
            ("x", x),
            ("x_scale", x if x_scale is None else x_scale),
            ("w", w),
            ("w_scale", w_scale),
            ("residual", residual),
            ("norm_weight", norm_weight),
        ):
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if out is None:
            out = torch.empty_like(residual)
        if residual_out is None:
            residual_out = torch.empty_like(residual)
        with torch.cuda.device(self.device):
            _run_compiled(
                self._compiled["launch"],
                ptr_arg(x, torch_to_fx_dtype(self.x_dtype)),
                (
                    flyc.from_c_void_p(fx.Float32, self._x_scale)
                    if self.quantize
                    else ptr_arg(x_scale, fx.Float32)
                ),
                flyc.from_c_void_p(fx.Int32, self._xs),
                ptr_arg(self._sync, fx.Int32),
                ptr_arg(w, fx.Int32),
                ptr_arg(w_scale, fx.Float32),
                ptr_arg(residual, fx.BFloat16),
                ptr_arg(residual_out, fx.BFloat16),
                ptr_arg(out, fx.BFloat16),
                ptr_arg(norm_weight, fx.BFloat16),
                ptr_arg(self._peers, fx.Int64),
                ptr_arg(self.trace, fx.Int64),
                rows,
                self.eps,
                torch.cuda.current_stream(self.device),
            )
        return out, residual_out

    def close(self):
        for base in self._peer_bases:
            try:
                IpcHeap.close_mem_handle(int(base))
            except RuntimeError:
                pass
        self._peer_bases = []
        for ptr in (self._inbox, self._xs, self._x_scale):
            if ptr:
                try:
                    IpcHeap.free_device_mem(ptr)
                except RuntimeError:
                    pass
        self._inbox = 0
        self._xs = 0
        self._x_scale = 0

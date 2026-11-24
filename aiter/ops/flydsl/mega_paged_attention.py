# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of mega paged attention: decode attention, o_proj, all-reduce and
Gemma RMSNorm of one TP layer in one persistent launch (gfx942/gfx950).

Per rank, for ``rows`` decode tokens with one KV head and ``query_group_size``
query heads on this rank:

    attn = paged_attention(query, KV)                 # bf16, as pa_decode
    x, x_scale = per_token_fp8(attn)                  # o_proj's input quant
    y_r = bf16(x * x_scale @ (W_r * w_scale_r)^T)     # this rank's o_proj partial
    acc = float(bf16(sum_r y_r)) + residual           # all-reduce, rank order
    residual_out = bf16(acc)
    out = bf16(acc * rsqrt(mean(acc^2) + eps) * (1 + norm_weight))

``W`` is the FP8 o_proj weight [hidden, query_group_size * head_dim] as
``shuffle_weight(layout=(16, 16))`` leaves it, with per-output-channel scales.

Contract (the kernel is persistent and its CTAs and ranks spin on each other):

* Construction is a collective over ``group`` (IPC handle exchange).
* ``__call__`` is a collective: every rank calls it the same number of times,
  in the same order, with the same row count.
* It needs the whole GPU: one CTA per CU, all resident. Do not run it next to
  kernels on other streams or other processes.
"""

from __future__ import annotations

import ctypes

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist

from .ipc_heap import IpcHeap
from .kernels.mega_paged_attention import (
    SENTINEL_FILL_BYTE,
    TRACE_STAMPS,
    compile_mega_paged_attention,
)
from .kernels.tensor_shim import _run_compiled, ptr_arg, torch_to_fx_dtype

__all__ = ["MegaPagedAttention"]


class MegaPagedAttention:
    """One TP rank's mega paged attention for one layer shape.

    Layers of one shape may share an instance: their calls form one collective
    sequence.
    """

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device,
        head_dim: int,
        query_group_size: int,
        block_size: int,
        max_rows: int,
        max_context_length: int,
        hidden: int,
        kv_dtype: torch.dtype,
        softmax_scale: float,
        eps: float,
        group=None,
        max_tiles_per_task: int = 4,
        block_n: int | None = None,
        prefetch: int | None = None,
        occupancy: int = 1,
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
        self.hidden = int(hidden)
        self.eps = float(eps)
        self.kv_dtype = kv_dtype
        with torch.cuda.device(self.device):
            num_cu = torch.cuda.get_device_properties(self.device).multi_processor_count
            self._compiled = compile_mega_paged_attention(
                head_dim=head_dim,
                query_group_size=query_group_size,
                block_size=block_size,
                max_rows=max_rows,
                max_context_length=max_context_length,
                max_tiles_per_task=max_tiles_per_task,
                num_compute_units=num_cu,
                softmax_scale=softmax_scale,
                query_dtype=torch.bfloat16,
                kv_dtype=kv_dtype,
                hidden=hidden,
                world_size=world_size,
                rank=rank,
                block_n=block_n,
                prefetch=prefetch,
                occupancy=occupancy,
                two_shot=two_shot,
                grid=grid,
                trace=trace,
            )
        c = self._compiled
        self.query_group_size = int(query_group_size)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        cap, k = c["capacity"], c["row_k"]
        dev = self.device
        self._row_k = k
        # Uncached: the kernel hands partials and rows between CTAs without
        # L2 writebacks or invalidates. pmax, psum, pout, x, x_scale.
        self._handoff = []
        with torch.cuda.device(dev):
            for nbytes in (
                cap * query_group_size * 4,
                cap * query_group_size * 4,
                cap * query_group_size * head_dim * 2,
                max_rows * k,
                max_rows * 4,
            ):
                self._handoff.append(
                    IpcHeap.alloc(nbytes, IpcHeap._HIP_DEVICE_MALLOC_UNCACHED)
                )
        self._sync = torch.zeros(c["sync_words"], dtype=torch.int32, device=dev)
        self.trace = torch.zeros(
            (c["grid"], TRACE_STAMPS), dtype=torch.int64, device=dev
        )
        self._inbox = 0
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

    def _copy_out(self, ptr, tensor):
        with torch.cuda.device(self.device):
            torch.cuda.synchronize(self.device)
            hip = IpcHeap._load_hip()
            err = hip.hipMemcpy(
                ctypes.c_void_p(tensor.data_ptr()),
                ctypes.c_void_p(int(ptr)),
                ctypes.c_size_t(tensor.numel() * tensor.element_size()),
                ctypes.c_int(3),  # hipMemcpyDeviceToDevice
            )
            IpcHeap._hip_check(err, what="hipMemcpy")
        return tensor

    def _x_ptrs(self):
        return self._handoff[3:5]

    @property
    def x(self):
        """A copy of the last call's FP8 attention rows (o_proj's input)."""
        rows, k = self.max_rows, self._row_k
        chunks = self._copy_out(
            self._x_ptrs()[0],
            torch.empty(
                (rows // 16, k // 64, 4, 16, 16), dtype=torch.uint8, device=self.device
            ),
        )
        # [row / 16][K / 64][K % 64 / 16][row % 16][K % 16] -> [row][K]
        rows_k = chunks.permute(0, 3, 1, 2, 4).reshape(rows, k).contiguous()
        return rows_k.view(self.kv_dtype)

    @property
    def x_scale(self):
        """A copy of the last call's per-row x scales."""
        return self._copy_out(
            self._x_ptrs()[1], torch.empty(self.max_rows, device=self.device)
        )

    def __call__(
        self,
        query: torch.Tensor,  # [rows, query_group_size, head_dim] bf16
        key_cache: torch.Tensor,  # [blocks, 1, head_dim // 16, block_size, 16] FP8
        value_cache: torch.Tensor,  # [blocks, 1, block_size // 16, head_dim, 16] FP8
        key_scale: torch.Tensor,  # [blocks, 1, block_size] fp32, per token
        value_scale: torch.Tensor,  # [blocks, 1, block_size] fp32, per token
        block_tables: torch.Tensor,  # [rows, max_blocks] int32
        context_lengths: torch.Tensor,  # [rows] int32
        w: torch.Tensor,  # [hidden, query_group_size * head_dim] FP8, shuffled
        w_scale: torch.Tensor,  # [hidden] fp32
        residual: torch.Tensor,  # [rows, hidden] bf16
        norm_weight: torch.Tensor,  # [hidden] bf16 (Gemma: the gain is 1 + w)
        out: torch.Tensor | None = None,
        residual_out: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows = int(query.shape[0])
        if not 0 < rows <= self.max_rows:
            raise ValueError(f"rows must be in [1, {self.max_rows}], got {rows}")
        if query.dtype != torch.bfloat16 or query.shape[1:] != (
            self.query_group_size,
            self.head_dim,
        ):
            raise ValueError(
                "query must be bf16 [rows, query_group_size, head_dim], "
                f"got {query.dtype} {tuple(query.shape)}"
            )
        if query.stride(2) != 1:
            raise ValueError("query's head_dim must be contiguous")
        if key_cache.dtype != self.kv_dtype or value_cache.dtype != self.kv_dtype:
            raise ValueError(f"KV caches must be {self.kv_dtype}")
        for name, t in (
            ("key_cache", key_cache),
            ("value_cache", value_cache),
            ("w", w),
            ("w_scale", w_scale),
            ("residual", residual),
            ("norm_weight", norm_weight),
            ("block_tables", block_tables),
            ("context_lengths", context_lengths),
        ):
            if not t.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if key_scale.dim() != 3 or key_scale.stride() != value_scale.stride():
            raise ValueError(
                "key_scale and value_scale must be [blocks, 1, block_size]"
            )
        if tuple(w.shape) != (self.hidden, self._compiled["row_k"]):
            raise ValueError(
                f"w must be [{self.hidden}, {self._compiled['row_k']}], got {tuple(w.shape)}"
            )
        if tuple(residual.shape) != (rows, self.hidden):
            raise ValueError(f"residual must be [{rows}, {self.hidden}]")
        if block_tables.dtype != torch.int32 or context_lengths.dtype != torch.int32:
            raise ValueError("block_tables and context_lengths must be int32")
        if out is None:
            out = torch.empty_like(residual)
        if residual_out is None:
            residual_out = torch.empty_like(residual)
        fp8 = torch_to_fx_dtype(self.kv_dtype)
        with torch.cuda.device(self.device):
            _run_compiled(
                self._compiled["launch"],
                ptr_arg(query, fx.BFloat16),
                ptr_arg(key_cache, fp8),
                ptr_arg(value_cache, fp8),
                ptr_arg(key_scale, fx.Float32),
                ptr_arg(value_scale, fx.Float32),
                ptr_arg(block_tables, fx.Int32),
                ptr_arg(context_lengths, fx.Int32),
                flyc.from_c_void_p(fx.Float32, self._handoff[0]),
                flyc.from_c_void_p(fx.Float32, self._handoff[1]),
                flyc.from_c_void_p(fx.BFloat16, self._handoff[2]),
                flyc.from_c_void_p(fx.Int32, self._x_ptrs()[0]),
                flyc.from_c_void_p(fx.Float32, self._x_ptrs()[1]),
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
                int(block_tables.shape[1]),
                int(key_scale.stride(0)),
                int(key_scale.stride(1)),
                int(query.stride(0)),
                int(query.stride(1)),
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
        for ptr in [self._inbox, *self._handoff]:
            if ptr:
                try:
                    IpcHeap.free_device_mem(ptr)
                except RuntimeError:
                    pass
        self._inbox = 0
        self._handoff = []

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP IPC workspace for the small-token Engram gate/residual kernel."""

import flydsl.compiler as flyc
import torch
import torch.distributed as dist

from .kernels.engram_post_wkv import GRID_CAP, SLOT_BYTES, launch
from .kernels.tensor_shim import _preload_compiled
from .quick_allreduce_int4_ipc import UncachedIpcHeap as Heap


class FusedEngramPost:
    """One workspace per serialized TP group call sequence, initialized eagerly.

    All ranks must invoke the same shapes in the same order. Warm each shape
    before graph capture. Graphs retain the workspace and its device epochs;
    callers must keep this object alive for the lifetime of their graphs.
    """

    def __init__(self, group):
        from aiter.jit.utils.chip_info import get_gfx_runtime

        if get_gfx_runtime() != "gfx950":
            raise ValueError("Fused Engram currently supports gfx950")
        self.group = group
        self.rank = dist.get_rank(group)
        self.tp = dist.get_world_size(group)
        if self.tp not in (2, 4, 8):
            raise ValueError("Fused Engram requires TP2, TP4 or TP8")
        from .quick_allreduce_int4 import _validate_ipc_process_group

        _validate_ipc_process_group(group, rank=self.rank)
        self.own, self.imported = None, []
        self.compiled = {}
        try:
            self.own = Heap.alloc_uncached(GRID_CAP * SLOT_BYTES)
            handles = Heap.gather_object_list_via_broadcast(
                group, Heap.get_mem_handle_bytes(self.own)
            )
            ptrs = []
            for rank, handle in enumerate(handles):
                ptr = (
                    self.own
                    if rank == self.rank
                    else Heap.open_mem_handle(bytes(handle))
                )
                if rank != self.rank:
                    self.imported.append(ptr)
                ptrs.append(ptr)
            self.peers = torch.tensor(ptrs, device="cuda", dtype=torch.int64)
            self.epoch = torch.zeros(GRID_CAP, device="cuda", dtype=torch.int32)
            self.error = torch.zeros(GRID_CAP * 4, device="cuda", dtype=torch.int32)
            self.dummy_mask = torch.ones(1, device="cuda", dtype=torch.int8)
            torch.cuda.synchronize()
            dist.barrier(group=group)
        except Exception:
            self.close()
            raise

    def __call__(self, h, kv, weight, mask=None, *, value_start=0, eps=1e-20):
        m = h.numel() // (4 * 5120)
        if h.shape[-2:] != (4, 5120) or tuple(kv.shape) != (
            *h.shape[:-2],
            20480 // self.tp + 5120,
        ):
            raise ValueError(
                "Fused Engram requires matching [...,4,5120] H and local KV"
            )
        if (
            weight.shape != (4, 5120)
            or h.dtype != torch.bfloat16
            or kv.dtype != torch.bfloat16
            or weight.dtype != torch.float32
        ):
            raise ValueError(
                "Fused Engram requires BF16 H/KV and FP32 [4,5120] weights"
            )
        if not all(
            t.is_cuda and t.is_contiguous() and t.device == h.device
            for t in (h, kv, weight)
        ):
            raise ValueError("Fused Engram tensors must be contiguous on the same GPU")
        if (
            h.device != self.epoch.device
            or torch.cuda.current_device() != self.epoch.device.index
        ):
            raise ValueError("Fused Engram must run on its workspace's GPU")
        if mask is not None and (
            mask.dtype != torch.bool
            or mask.shape != h.shape[:-2]
            or not mask.is_contiguous()
            or mask.device != h.device
        ):
            raise ValueError(
                "Fused Engram mask must be contiguous bool with the token shape"
            )
        if not 0 <= value_start < 5120 or value_start % 2:
            raise ValueError("Fused Engram V rotation must be aligned to two elements")
        if m == 0:
            return h.clone()
        if m > 128:
            raise ValueError(
                "Fused Engram supports at most 128 tokens; use the composed path"
            )
        out = torch.empty_like(h)
        args = (
            h.view(m, 4, 5120),
            kv.view(m, -1),
            weight,
            self.dummy_mask if mask is None else mask.view(torch.int8).view(m),
            out.view(m, 4, 5120),
            self.epoch,
            self.error,
            self.peers,
            m,
            self.tp,
            self.rank,
            min(m * (4 // min(self.tp, 4)), GRID_CAP),
            mask is not None,
            value_start,
            eps,
            torch.cuda.current_stream(),
        )
        key = (m, mask is not None, value_start, eps)
        if key not in self.compiled:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm the fused Engram shape before CUDA Graph capture"
                )
            _preload_compiled(launch, *args)
            dist.barrier(group=self.group)
            # preload materializes the artifact but returns no callable on
            # some FlyDSL versions. compile now executes the cached launcher
            # and returns its callable, after every rank finished compilation.
            self.compiled[key] = flyc.compile(launch, *args)
            return out
        self.compiled[key](*args)
        return out

    def check(self):
        """Explicit test/debug check outside graph capture; no hot-path D2H sync."""
        if any(self.error.cpu().tolist()):
            raise RuntimeError("Fused Engram peer exchange timed out")

    def close(self):
        for ptr in self.imported:
            try:
                Heap.close_mem_handle(ptr)
            except RuntimeError:
                pass
        self.imported = []
        if self.own is not None:
            try:
                Heap.free_device_mem(self.own)
            except RuntimeError:
                pass
            self.own = None

    def __del__(self):
        if hasattr(self, "imported"):
            self.close()

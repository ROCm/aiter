# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused expert-parallel MoE layer (dispatch, routed experts and combine in one launch per rank), gfx950.

Layout: hidden 5120, intermediate 2304, top 6, ``experts_per_rank`` <= 63 local experts on each of 2 / 4 / 8
ranks of one node (DeepSeek-V4.1-Flash EP8: 48), MXFP4 weights in the ``shuffle_weight_a16w4(16, gate_up)`` /
``shuffle_scale_a16w4`` layouts MegaMoEV2 takes, MXFP8 activations (bf16 rows are quantized in the dispatch),
clamp-SwiGLU (limit 10), at most 1024 tokens per rank. ``fused_ep_moe_supported`` is the eligibility check a
caller uses to fall back to another MoE path; the choice must be the same on every rank of the EP group (decide
it from the group's largest per-rank token count). On 8 x MI355X it is faster than the MegaMoEV2 path this layout
runs (the staged one) at 1 .. 768 tokens per rank under uniform, Zipf and single-rank routing (at 768 only at
parity under Zipf routing); at 1024 it is slower under Zipf routing, so callers fall back to MegaMoEV2 above 768
tokens per rank (``op_tests/multigpu_tests/bench_fused_ep_moe.py``).

Contract:

* Construction and ``prepare`` are collectives over ``group``; every rank calls ``forward`` once per layer, in the
  same order, idle ranks with 0 rows included. One instance serves every layer of a model (the weights are
  forward arguments); two calls must not run concurrently on one device.
* ``topk_ids`` are physical expert ids, rank-major (owner = ``id // experts_per_rank``); an id outside
  ``[0, experts_per_rank * world_size)`` (e.g. -1 padding) is not routed and contributes zero. The routed ids of
  one token must be distinct, as a top-k produces them (an expert's record list holds one record per source
  token). All ranks use the same ``experts_per_rank``.
* The kernel is persistent and its workgroups and ranks wait on each other: it needs every rank's launch to run
  and must not share the GPU with other work that waits on it (as MegaMoEV2's small-input single kernel).
* The output is bitwise reproducible (no atomics into outputs) and the call is CUDA-graph capturable; the kernel
  resets its own counters. ``prepare(w)`` compiles every tile tier before capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from aiter.jit.utils.chip_info import get_gfx_runtime

from .ipc_heap import IpcHeap
from .kernels.fused_ep_moe_a8w4 import (
    CTL_BYTES,
    LOC_WORDS,
    NE_MAX,
    READY_STRIDE,
    TM,
    TOPK,
    H,
    I,
    compile_fused,
    max_blocks,
    pool_layout,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "MAX_TOKENS_PER_RANK",
    "FusedEPMoE",
    "FusedEPMoEWeights",
    "fused_ep_moe_supported",
]

MAX_TOKENS_PER_RANK = TM
_ND = 128  # dispatch workgroups per rank
# token bound -> (bm, GEMM1 K-waves, GEMM1 prefetch, GEMM1 16-column pairs per wave, GEMM1 LDS ring stages,
# GEMM2 columns, GEMM2 prefetch, GEMM2 LDS ring stages, cached weights for an expert with several blocks,
# waves per workgroup, persistent workgroups); measured on 8 ranks over uniform / Zipf / single-rank routing.
_TIERS = (
    (2, (16, 4, 3, 2, 0, 256, 4, 0, False, 4, 256)),
    (8, (16, 4, 3, 4, 0, 256, 4, 0, False, 4, 256)),
    (48, (16, 4, 3, 2, 0, 256, 4, 0, False, 4, 512)),
    (160, (32, 1, 3, 4, 3, 256, 4, 3, True, 4, 256)),
    (448, (64, 1, 2, 2, 4, 256, 3, 6, True, 8, 256)),
    (MAX_TOKENS_PER_RANK, (128, 1, 2, 2, 3, 256, 3, 3, True, 8, 256)),
)


def fused_ep_moe_supported(
    *,
    world_size: int,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    quant: str = "a8w4",
    swiglu_limit: float = 10.0,
    act: str = "silu",
    max_tokens_per_rank: int = 0,
    arch: str | None = None,
    num_cu: int | None = None,
) -> bool:
    """Whether ``FusedEPMoE`` runs this layer; callers fall back to another MoE path when it does not.

    The work queues are split by XCD, so the part must have all 8 XCDs (256 CUs).
    """
    if arch is None:
        arch = get_gfx_runtime()
    if num_cu is None:
        num_cu = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    return (
        arch.split(":", 1)[0] == "gfx950"
        and num_cu == 256
        and world_size in (2, 4, 8)
        and model_dim == H
        and inter_dim == I
        and topk == TOPK
        and experts % world_size == 0
        and 1 <= experts // world_size <= NE_MAX
        and quant == "a8w4"
        and swiglu_limit == 10.0
        and act == "silu"
        and 0 <= max_tokens_per_rank <= MAX_TOKENS_PER_RANK
    )


def _sized_tokens(m: int, rows_hint: int | None = None) -> int:
    """The token count the tiles are sized for: twice ``m`` up to 448 (a rank that owns popular experts receives
    more rows than its tokens' share), at least ``rows_hint``. From 9 tokens the 32-row tier is no slower than the
    16-row one under uniform routing and 25 % faster under skew; from 49 tokens the 8-wave 64-row tier costs 3 %
    uniform and saves 6 % (Zipf) to 25 % (all routes on one rank)."""
    s = max(m, min(2 * m, _TIERS[-2][0]), rows_hint or 0, 1)
    if m > 8:
        s = max(s, _TIERS[3][0])
    if m > 48:
        s = max(s, _TIERS[3][0] + 1)
    return s


def _tier(n_tokens: int):
    """The tile configuration for ``n_tokens`` (1 .. MAX_TOKENS_PER_RANK) sized tokens."""
    return next(cfg for bound, cfg in _TIERS if n_tokens <= bound)


@dataclass
class FusedEPMoEWeights:
    """One layer's local experts in the kernel layouts: uint8 views of ``shuffle_weight_a16w4(w13, 16, True)``,
    ``shuffle_scale_a16w4(w13_scale, E, True)``, ``shuffle_weight_a16w4(w2, 16, False)`` and
    ``shuffle_scale_a16w4(w2_scale, E, False)`` (the tensors MegaMoEV2 takes)."""

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor

    def __post_init__(self):
        u8 = torch.uint8
        self.w13, self.w13_scale, self.w2, self.w2_scale = (
            t.view(u8).contiguous()
            for t in (self.w13, self.w13_scale, self.w2, self.w2_scale)
        )
        e = self.w13.numel() // (2 * I * (H // 2))
        assert 1 <= e <= NE_MAX, f"{e} local experts (1..{NE_MAX} supported)"
        assert self.w13.numel() == e * 2 * I * (H // 2)
        assert self.w13_scale.numel() == e * 2 * I * (H // 32)
        assert self.w2.numel() == e * H * (I // 2)
        assert self.w2_scale.numel() == e * H * (I // 32)
        self.experts = e

    @classmethod
    def from_row_major(cls, w13, w13_scale, w2, w2_scale) -> FusedEPMoEWeights:
        """From checkpoint layouts: ``w13 [E, 2 * 2304, 2560]`` (gate rows, then up rows; two e2m1 per byte, low
        nibble first), ``w13_scale [E, 4608, 160]``, ``w2 [E, 5120, 1152]``, ``w2_scale [E, 5120, 72]`` (e8m0).
        """
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        n = w13.shape[0]
        assert w13.shape[1:] == (2 * I, H // 2) and w2.shape[1:] == (H, I // 2)
        u8 = torch.uint8
        return cls(
            shuffle_weight_a16w4(w13.view(u8), 16, True),
            shuffle_scale_a16w4(w13_scale.view(u8).reshape(-1, H // 32), n, True),
            shuffle_weight_a16w4(w2.view(u8), 16, False),
            shuffle_scale_a16w4(w2_scale.view(u8).reshape(-1, I // 32), n, False),
        )


class FusedEPMoE:
    """One rank's symmetric pool, control words and scratch; rank ``r`` owns the experts
    ``experts_per_rank * r`` .. ``experts_per_rank * (r + 1) - 1``."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        group,
        device: torch.device,
        experts_per_rank: int = 48,
    ):
        if world_size not in (2, 4, 8) or not 1 <= experts_per_rank <= NE_MAX:
            raise ValueError(
                f"world_size {world_size} (2, 4 or 8), experts_per_rank {experts_per_rank} (1..{NE_MAX})"
            )
        if (rank, world_size) != (
            dist.get_rank(group),
            dist.get_world_size(group),
        ):
            raise ValueError(
                f"rank {rank} / world_size {world_size} must be this process's rank in `group` and its size "
                f"({dist.get_rank(group)} / {dist.get_world_size(group)})"
            )
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError(f"device {device}: expected a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.rank, self.world_size, self.group = rank, world_size, group
        self.device, self.epr = device, experts_per_rank
        self._closed = False
        layout = pool_layout(world_size, experts_per_rank)
        with torch.cuda.device(device):
            self._pool = IpcHeap.alloc_coarse_grained(layout["total"])
            self._ctl = IpcHeap.alloc_uncached(CTL_BYTES)
            handles = (
                IpcHeap.get_mem_handle_bytes(self._pool),
                IpcHeap.get_mem_handle_bytes(self._ctl),
            )
            peers = IpcHeap.gather_object_list_via_broadcast(group, handles)
            self._opened = []
            pools, ctls = [], []
            for r, (hp, hc) in enumerate(peers):
                if r == rank:
                    pools.append(self._pool)
                    ctls.append(self._ctl)
                    continue
                pools.append(IpcHeap.open_mem_handle(hp))
                ctls.append(IpcHeap.open_mem_handle(hc))
                self._opened += [pools[-1], ctls[-1]]
        self._pool_addrs = pools + [0] * (8 - world_size)
        self._ctl_addrs = ctls + [0] * (8 - world_size)
        self._loc = torch.zeros(LOC_WORDS, dtype=torch.int32, device=device)
        self._ready = torch.zeros(
            (max_blocks(world_size, 16, experts_per_rank) + 1) * READY_STRIDE // 4,
            dtype=torch.int32,
            device=device,
        )
        self._rows = max_blocks(world_size, 128, experts_per_rank) * 128
        self._q = torch.empty(self._rows * I, dtype=torch.uint8, device=device)
        self._s = torch.empty(self._rows * (I // 32), dtype=torch.uint8, device=device)
        torch.cuda.synchronize(device)
        dist.barrier(group=group)

    def prepare(self, w: FusedEPMoEWeights) -> None:
        """A collective before graph capture: runs every tile tier, with bf16 and MXFP8 rows, once on calls that
        route nothing, so no kernel compiles while serving."""
        dev = self.device
        first = {}
        for m in range(1, MAX_TOKENS_PER_RANK + 1):
            first.setdefault(_tier(_sized_tokens(m)), m)
        for m in first.values():
            ids = torch.full((m, TOPK), -1, dtype=torch.int32, device=dev)
            tw = torch.zeros((m, TOPK), dtype=torch.float32, device=dev)
            self.forward(
                torch.zeros((m, H), dtype=torch.bfloat16, device=dev), None, w, tw, ids
            )
            self.forward(
                torch.zeros((m, H), dtype=torch.float8_e4m3fn, device=dev),
                torch.zeros((m, H // 32), dtype=torch.uint8, device=dev),
                w,
                tw,
                ids,
            )
        torch.cuda.synchronize(dev)
        dist.barrier(group=self.group)

    def forward(
        self,
        x: torch.Tensor,
        x_scales: torch.Tensor | None,
        w: FusedEPMoEWeights,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        out: torch.Tensor | None = None,
        rows_hint: int | None = None,
    ) -> torch.Tensor:
        """The routed experts of this rank's ``m`` tokens over every rank's experts: bf16 ``[m, 5120]``.

        Args:
            x: bf16 ``[m, 5120]`` (quantized to MXFP8 in the dispatch), or ``float8_e4m3fn`` codes with
                ``x_scales`` uint8 / e8m0 ``[m, 160]`` (one exponent per 32 columns). ``m`` <= 1024, 0 allowed.
            topk_ids: int32 ``[m, 6]`` physical expert ids, distinct within a row; ``topk_weights`` fp32 ``[m, 6]``.
            out: optional bf16 ``[max(m, 1), 5120]`` output.
            rows_hint: the tiles are sized for at least this many tokens (default: from ``m``), at most
                ``MAX_TOKENS_PER_RANK``.

        Every tensor must be on this instance's device.
        """
        if self._closed:
            raise RuntimeError("FusedEPMoE is closed")
        m = x.shape[0]
        if not (
            0 <= m <= MAX_TOKENS_PER_RANK and x.shape[1] == H and x.is_contiguous()
        ):
            raise ValueError(
                f"x {tuple(x.shape)}: expected contiguous [<= {MAX_TOKENS_PER_RANK}, {H}]"
            )
        tensors = {
            "x": x,
            "x_scales": x_scales,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "out": out,
            "w.w13": w.w13,
            "w.w13_scale": w.w13_scale,
            "w.w2": w.w2,
            "w.w2_scale": w.w2_scale,
        }
        for name, t in tensors.items():
            if t is not None and t.device != self.device:
                raise ValueError(f"{name} on {t.device}, instance on {self.device}")
        if rows_hint is not None and not 0 <= rows_hint <= MAX_TOKENS_PER_RANK:
            raise ValueError(
                f"rows_hint {rows_hint}: expected 0..{MAX_TOKENS_PER_RANK}"
            )
        if tuple(topk_ids.shape) != (m, TOPK) or tuple(topk_weights.shape) != (m, TOPK):
            raise ValueError(f"topk_ids / topk_weights must be [{m}, {TOPK}]")
        quant_in = x.dtype == torch.bfloat16
        if quant_in:
            if x_scales is not None:
                raise ValueError("x_scales must be None for bf16 rows")
        elif (
            x.dtype != torch.float8_e4m3fn
            or x_scales is None
            or tuple(x_scales.shape) != (m, H // 32)
        ):
            raise ValueError("x must be bf16, or float8_e4m3fn with x_scales [m, 160]")
        if w.experts != self.epr:
            raise ValueError(f"weights hold {w.experts} experts, instance {self.epr}")
        bm, kw, pf1, ni, nb1, tn2, pf2, nb2, hot, nwv, nw = _tier(
            _sized_tokens(m, rows_hint)
        )
        if out is None:
            out = torch.empty((max(m, 1), H), dtype=torch.bfloat16, device=self.device)
        elif (
            out.dtype != torch.bfloat16
            or out.shape[0] < max(m, 1)
            or out.shape[1] != H
            or not out.is_contiguous()
        ):
            raise ValueError(f"out must be contiguous bf16 [>= {max(m, 1)}, {H}]")
        tid = topk_ids.to(torch.int32).contiguous().view(-1)
        tw = topk_weights.to(torch.float32).contiguous().view(-1)
        xs = (
            x_scales.contiguous().view(torch.uint8).view(-1)
            if x_scales is not None
            else tid
        )
        fn = compile_fused(
            device=self.device.index,
            world=self.world_size,
            epr=self.epr,
            nd=_ND,
            nw=nw,
            bm=bm,
            kw=kw,
            pf1=pf1,
            ni=ni,
            nb1=nb1,
            tn2=tn2,
            pf2=pf2,
            nb2=nb2,
            hot=hot,
            nwv=nwv,
            quant_in=quant_in,
        )
        _run_compiled(
            fn,
            x.view(-1) if quant_in else x.view(torch.uint8).view(-1),
            xs,
            tid,
            tw,
            w.w13.view(-1),
            w.w13_scale.view(-1),
            self._q,
            self._s,
            w.w2.view(-1),
            w.w2_scale.view(-1),
            out,
            self._loc,
            self._ready,
            self.rank,
            m,
            self._rows,
            *self._pool_addrs,
            *self._ctl_addrs,
            torch.cuda.current_stream(self.device),
        )
        return out[:m]

    __call__ = forward

    def close(self) -> None:
        """A collective: unmaps the peers' memory and frees this rank's pool (no call may be in flight)."""
        if self._closed:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        with torch.cuda.device(self.device):
            for p in self._opened:
                IpcHeap.close_mem_handle(p)
        dist.barrier(group=self.group)
        with torch.cuda.device(self.device):
            IpcHeap.free_device_mem(self._pool)
            IpcHeap.free_device_mem(self._ctl)
        self._closed = True

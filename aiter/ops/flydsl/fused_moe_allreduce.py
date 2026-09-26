# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import ctypes
import functools

import torch

from .kernels.fused_moe_allreduce.fused_moe_allreduce_w8a8 import (
    ERR_AR,
    ERR_MIDS,
    ERR_SCORES,
    GRID,
    HIDDEN,
    INTER,
    MAX_PES,
    NUM_EXPERTS,
    SLOTS,
    SUPPORTED_PROTOS,
    SUPPORTED_SAMPLES,
    SYM_BYTES,
    TOP_K,
    compile_fused_moe_allreduce,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "FusedMoeAllreduceW8A8",
    "UncachedSymmetricBuffer",
    "NUM_MOE_WEIGHTS",
    "SYM_BYTES",
    "fused_moe_allreduce_w8a8",
    "fused_moe_allreduce_w8a8_supported",
    "swizzle_256_bf16",
    "swizzle_down_k128",
    "swizzle_pair_interleaved_k128",
]

NUM_MOE_WEIGHTS = NUM_EXPERTS + 1
MAX_SAMPLES = max(SUPPORTED_SAMPLES)
SCORE_LINE_WORDS = 32


def fused_moe_allreduce_w8a8_supported(gfx: str | None = None) -> bool:
    if gfx is None:
        from aiter.jit.utils.chip_info import get_gfx

        gfx = get_gfx()
    return gfx == "gfx950"


def swizzle_256_bf16(w: torch.Tensor) -> torch.Tensor:
    rows, k = w.shape
    assert rows % 8 == 0 and k % 64 == 0
    dev = w.device
    w16 = w.to(torch.bfloat16).view(torch.int16)
    rg, kc, lane, s, i = (
        torch.arange(n, device=dev) for n in (rows // 8, k // 64, 64, 2, 4)
    )
    RG, KC, L, S, II = torch.meshgrid(rg, kc, lane, s, i, indexing="ij")
    rows_ix = RG * 8 + (L >> 5) * 4 + (L & 3)
    ks = KC * 64 + S * 32 + (L >> 2 & 7) * 4 + II
    return w16[rows_ix, ks].reshape(-1).contiguous().view(torch.uint8)


@functools.cache
def _up_gate_index(inter: int, k: int, device) -> torch.Tensor:
    rows = 2 * inter
    p = torch.arange(rows, device=device)
    perm = (p % 16 // 8) * inter + (p // 16) * 8 + p % 8
    t, c, h, lane, i = (
        torch.arange(n, device=device) for n in (rows // 16, k // 128, 2, 64, 16)
    )
    T, C, H, L, II = torch.meshgrid(t, c, h, lane, i, indexing="ij")
    rows_ix = perm[T * 16 + L % 16]
    ks = C * 128 + L // 16 * 32 + H * 16 + II
    return (rows_ix * k + ks).reshape(-1)


def swizzle_pair_interleaved_k128(w_fp8: torch.Tensor, inter: int = INTER) -> torch.Tensor:
    k = w_fp8.shape[-1]
    idx = _up_gate_index(inter, k, w_fp8.device)
    w8 = w_fp8.contiguous().view(torch.uint8).reshape(-1, 2 * inter * k)
    return w8[:, idx].reshape(-1).contiguous()


@functools.cache
def _down_index(rows: int, device) -> torch.Tensor:
    k = 2 * 128
    idx = torch.empty(rows // 24 * 6144, dtype=torch.long, device=device)
    blk = torch.arange(rows // 24, device=device)
    L, H, II = torch.meshgrid(*(torch.arange(n, device=device) for n in (64, 2, 16)), indexing="ij")
    G, M, H2, I2 = torch.meshgrid(*(torch.arange(n, device=device) for n in (4, 8, 2, 16)), indexing="ij")
    for c in range(2):
        ks = c * 128 + L // 16 * 32 + H * 16 + II
        src = (blk[:, None, None, None] * 24 + (L % 16)[None]) * k + ks[None]
        dst = blk[:, None, None, None] * 6144 + (c * 2048 + H * 1024 + L * 16 + II)[None]
        idx[dst.reshape(-1)] = src.reshape(-1)
        ks = c * 128 + G * 32 + H2 * 16 + I2
        src = (blk[:, None, None, None, None] * 24 + 16 + M[None]) * k + ks[None]
        dst = blk[:, None, None, None, None] * 6144 + (
            4096 + c * 1024 + H2 * 512 + (G * 8 + M) * 16 + I2
        )[None]
        idx[dst.reshape(-1)] = src.reshape(-1)
    return idx


def swizzle_down_k128(w_fp8: torch.Tensor) -> torch.Tensor:
    rows, k = w_fp8.shape[-2:]
    assert rows % 24 == 0 and k == 256
    idx = _down_index(rows, w_fp8.device)
    w8 = w_fp8.contiguous().view(torch.uint8).reshape(-1, rows * k)
    return w8[:, idx].reshape(-1).contiguous()


def _ptr(t: torch.Tensor | None) -> int:
    return 0 if t is None else int(t.data_ptr())


def fused_moe_allreduce_w8a8(
    hidden_in: torch.Tensor,
    gamma: torch.Tensor,
    router_w: torch.Tensor,
    ug_w: torch.Tensor,
    ug_scales: torch.Tensor,
    bias: torch.Tensor,
    down_w: torch.Tensor,
    down_scales: torch.Tensor,
    residual: torch.Tensor | None,
    sym: torch.Tensor | None,
    mype: int,
    npes: int,
    flag: int,
    norm_hidden: torch.Tensor,
    scores: torch.Tensor,
    score_lines: torch.Tensor,
    flags: torch.Tensor,
    probs_out: torch.Tensor,
    indices_out: torch.Tensor,
    hidden_mid: torch.Tensor | None,
    out: torch.Tensor,
    mid_pairs: torch.Tensor,
    sen_tag: int,
    proto: int = 0,
    timeline: torch.Tensor | None = None,
) -> None:
    s_n = int(hidden_in.shape[0])
    if s_n not in SUPPORTED_SAMPLES:
        raise ValueError(f"{s_n} tokens: supported {SUPPORTED_SAMPLES}")
    if not 1 <= npes <= MAX_PES or (npes > 1 and sym is None):
        raise ValueError(f"npes={npes} needs 1..{MAX_PES} and a symmetric table")
    if flags.numel() < 2 * GRID or score_lines.numel() < s_n * 32 * SCORE_LINE_WORDS:
        raise ValueError("flags / score_lines workspaces are too small")
    exe = compile_fused_moe_allreduce(
        s_n, int(proto), timeline is not None, hidden_in.device.index or 0
    )
    _run_compiled(
        exe,
        _ptr(hidden_in),
        _ptr(gamma),
        _ptr(router_w),
        _ptr(ug_w),
        _ptr(ug_scales),
        _ptr(bias),
        _ptr(down_w),
        _ptr(down_scales),
        _ptr(residual),
        _ptr(sym),
        int(mype),
        int(npes),
        int(flag) & 0x7FFFFFFF,
        _ptr(norm_hidden),
        _ptr(scores),
        _ptr(score_lines),
        _ptr(flags),
        _ptr(probs_out),
        _ptr(indices_out),
        _ptr(hidden_mid),
        _ptr(out),
        _ptr(mid_pairs),
        int(sen_tag) & 0x7FFFFFFF,
        _ptr(timeline),
        torch.cuda.current_stream(),
    )


class UncachedSymmetricBuffer:
    def __init__(self, nbytes: int, *, rank: int, world_size: int, group=None, device=None):
        from .quick_allreduce_int4_ipc import UncachedIpcHeap

        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self._heap = UncachedIpcHeap
        self._opened = []
        with torch.cuda.device(self.device):
            self.base = UncachedIpcHeap.alloc_uncached(nbytes)
            ptrs = [self.base]
            if world_size > 1:
                import torch.distributed as dist

                handle = UncachedIpcHeap.get_mem_handle_bytes(self.base)
                handles = [None] * world_size
                dist.all_gather_object(handles, handle, group=group)
                ptrs = []
                for r, h in enumerate(handles):
                    if r == rank:
                        ptrs.append(self.base)
                    else:
                        peer = UncachedIpcHeap.open_mem_handle(h)
                        self._opened.append(peer)
                        ptrs.append(peer)
                dist.barrier(group=group)
        self.table = torch.tensor(ptrs, dtype=torch.int64, device=self.device)

    def close(self) -> None:
        with torch.cuda.device(self.device):
            for p in self._opened:
                self._heap.close_mem_handle(p)
            self._opened = []
            if self.base:
                self._heap.free_device_mem(self.base)
                self.base = 0


class FusedMoeAllreduceW8A8:
    def __init__(
        self,
        *,
        rank: int = 0,
        world_size: int = 1,
        group=None,
        device: torch.device | None = None,
        proto: int = 0,
        sym: torch.Tensor | None = None,
    ):
        if proto not in SUPPORTED_PROTOS:
            raise ValueError(f"proto must be one of {SUPPORTED_PROTOS}")
        if not 1 <= world_size <= MAX_PES:
            raise ValueError(f"world_size must be 1..{MAX_PES}")
        self.rank, self.world_size, self.proto = int(rank), int(world_size), int(proto)
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        i32 = {"dtype": torch.int32, "device": self.device}
        self.score_lines = torch.zeros(MAX_SAMPLES, 32, SCORE_LINE_WORDS, **i32)
        self.flags = torch.zeros(2 * GRID, **i32)
        self.mid_pairs = torch.zeros(MAX_SAMPLES, SLOTS, INTER, **i32)
        self._symbuf = None
        if sym is None:
            self._symbuf = UncachedSymmetricBuffer(
                SYM_BYTES, rank=self.rank, world_size=world_size, group=group, device=self.device
            )
            sym = self._symbuf.table
        self.sym = sym
        self.router_w = self.gamma = self.bias = None
        self.ug_w = self.ug_scales = self.down_w = self.down_scales = None

    @classmethod
    def peer_group(cls, devices, *, proto: int = 0) -> list[FusedMoeAllreduceW8A8]:
        devices = [torch.device(d) if not isinstance(d, torch.device) else d for d in devices]
        world = len(devices)
        if not 1 <= world <= MAX_PES:
            raise ValueError(f"1..{MAX_PES} devices, got {world}")
        hip = ctypes.CDLL("libamdhip64.so")
        prev = torch.cuda.current_device()
        for d in devices:
            torch.cuda.set_device(d)
            for peer in devices:
                if peer != d:
                    err = hip.hipDeviceEnablePeerAccess(peer.index, 0)
                    if err not in (0, 704):
                        raise RuntimeError(f"hipDeviceEnablePeerAccess({d} -> {peer}) = {err}")
        torch.cuda.set_device(prev)
        bufs = [torch.zeros(SYM_BYTES, dtype=torch.uint8, device=d) for d in devices]
        ptrs = [b.data_ptr() for b in bufs]
        ops = []
        for r, d in enumerate(devices):
            op = cls(
                rank=r,
                world_size=world,
                device=d,
                proto=proto,
                sym=torch.tensor(ptrs, dtype=torch.int64, device=d),
            )
            op._peer_bufs = bufs
            ops.append(op)
        return ops

    def load_weights(
        self,
        *,
        router_w: torch.Tensor,
        gamma: torch.Tensor,
        bias: torch.Tensor,
        ug_w: torch.Tensor,
        ug_scales: torch.Tensor,
        down_w: torch.Tensor,
        down_scales: torch.Tensor,
    ) -> None:
        dev = self.device
        assert router_w.shape == (NUM_EXPERTS, HIDDEN)
        assert ug_w.shape[1:] == (2 * INTER, HIDDEN) and ug_w.dtype == torch.float8_e4m3fn
        assert down_w.shape[1:] == (HIDDEN, INTER) and down_w.dtype == torch.float8_e4m3fn
        self.router_w = swizzle_256_bf16(router_w.to(dev))
        self.gamma = gamma.to(dev, torch.float32).contiguous()
        self.bias = bias.to(dev, torch.float32).contiguous()
        self.ug_w = swizzle_pair_interleaved_k128(ug_w.to(dev))
        self.ug_scales = ug_scales.to(dev, torch.float32).contiguous()
        self.down_w = swizzle_down_k128(down_w.to(dev))
        self.down_scales = down_scales.to(dev, torch.float32).contiguous()

    def forward(
        self,
        hidden: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        out: torch.Tensor | None = None,
        sen_tag: int = 0,
        flag: int = 0,
        timeline: torch.Tensor | None = None,
    ):
        s_n = int(hidden.shape[0])
        norm, scores, mid, probs, indices, out_new = self._outputs(s_n, hidden.device)
        if out is None:
            out = out_new
        fused_moe_allreduce_w8a8(
            hidden,
            self.gamma,
            self.router_w,
            self.ug_w,
            self.ug_scales,
            self.bias,
            self.down_w,
            self.down_scales,
            residual,
            self.sym,
            self.rank,
            self.world_size,
            flag,
            norm,
            scores,
            self.score_lines,
            self.flags,
            probs,
            indices,
            mid,
            out,
            self.mid_pairs,
            sen_tag,
            proto=self.proto,
            timeline=timeline,
        )
        return norm, scores, mid, probs, indices, out

    __call__ = forward

    def warmup(self, samples=SUPPORTED_SAMPLES, protos=SUPPORTED_PROTOS) -> None:
        dev = self.device
        with torch.cuda.device(dev):
            for s_n in samples:
                hidden = torch.zeros(s_n, HIDDEN, dtype=torch.bfloat16, device=dev)
                for proto in protos:
                    self._warm_tag = getattr(self, "_warm_tag", 1 << 30) + 1
                    norm, scores, mid, probs, indices, out = self._outputs(s_n, dev)
                    fused_moe_allreduce_w8a8(
                        hidden, self.gamma, self.router_w, self.ug_w, self.ug_scales,
                        self.bias, self.down_w, self.down_scales, None, self.sym, 0, 1,
                        self._warm_tag, norm, scores, self.score_lines, self.flags, probs,
                        indices, mid, out, self.mid_pairs, self._warm_tag, proto=proto,
                    )
            torch.cuda.synchronize(dev)

    @staticmethod
    def _outputs(s_n, dev):
        return (
            torch.empty(s_n, HIDDEN, dtype=torch.bfloat16, device=dev),
            torch.empty(s_n, NUM_EXPERTS, dtype=torch.float32, device=dev),
            torch.empty(s_n, SLOTS, INTER, dtype=torch.bfloat16, device=dev),
            torch.empty(s_n, TOP_K, dtype=torch.float32, device=dev),
            torch.empty(s_n, TOP_K, dtype=torch.int32, device=dev),
            torch.empty(s_n, HIDDEN, dtype=torch.bfloat16, device=dev),
        )

    def poll_errors(self) -> dict[str, list[int]]:
        err = self.flags[GRID:].cpu()
        stages = (("scores", ERR_SCORES), ("mids", ERR_MIDS), ("allreduce", ERR_AR))
        return {
            name: (err & code).nonzero().flatten().tolist()
            for name, code in stages
            if bool((err & code).any())
        }

    def close(self) -> None:
        if self._symbuf is not None:
            self._symbuf.close()
            self._symbuf = None

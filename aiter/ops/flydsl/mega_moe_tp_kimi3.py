# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the Kimi-K3 fused MoE + all-reduce (A4W4) decode kernel.

Shapes per TP8 rank (https://github.com/ROCm/FlyDSL/pull/1204): router input
7168 (bf16 router [896, 7168]), routed latent 3584, 896 experts, top-16,
intermediate 384 per rank, SiTUv2 (beta 4, linear beta 25), MXFP4 experts with
one E8M0 scale per 32 values. The op covers the routed experts and the TP
all-reduce of their latent output; the shared experts and the latent
down/up projections around it are separate ops in the Kimi-K3 layer.
"""

from __future__ import annotations

import ctypes
import functools

import torch

from .mega_moe_tp_glm import UncachedSymmetricBuffer, _ptr, mega_moe_tp_w8a8_glm_supported
from .kernels.mega_moe_tp.mega_moe_tp_kimi3 import (
    DN_BLOCKS,
    ERR_AR,
    ERR_MIDS,
    ERR_SCORES,
    GRID,
    HIDDEN,
    INTER,
    MAX_PES,
    MX_BLOCK,
    NUM_EXPERTS,
    ROUTER_HIDDEN,
    RT_CH,
    RT_WAVES,
    SLOTS,
    SUPPORTED_SAMPLES,
    SYM_BYTES,
    TOP_K,
    compile_mega_moe_tp_kimi3,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "MegaMoeTpKimi3",
    "mega_moe_tp_kimi3",
    "mega_moe_tp_kimi3_supported",
    "pack_down_kimi3",
    "pack_up_gate_kimi3",
]

E8M0_ONE = 127
MAX_SAMPLES = max(SUPPORTED_SAMPLES)
mega_moe_tp_kimi3_supported = mega_moe_tp_w8a8_glm_supported


@functools.cache
def _up_gate_index(device):
    """Up/gate MXFP4 [E, 2*INTER, HIDDEN/2] -> per expert [48 tiles][28 k-blocks]
    [64 lanes][16 B]: tile t holds gate rows 8t..8t+7 then up rows; lane l the
    row l % 16 and bytes 16 * (l // 16) of the k-block. Scales -> [48][7 waves]
    [64 lanes][4 k-blocks]."""
    rows, kb_n = 2 * INTER, HIDDEN // 128
    p = torch.arange(rows, device=device)
    perm = (p % 16 // 8) * INTER + (p // 16) * 8 + p % 8
    t, kb, lane, i = (torch.arange(n, device=device) for n in (rows // 16, kb_n, 64, 16))
    Tt, KB, L, II = torch.meshgrid(t, kb, lane, i, indexing="ij")
    w_idx = (perm[Tt * 16 + L % 16] * (HIDDEN // 2) + KB * 64 + L // 16 * 16 + II).reshape(-1)
    t, w, lane, ch = (torch.arange(n, device=device) for n in (rows // 16, RT_WAVES, 64, RT_CH))
    Tt, W, L, CH = torch.meshgrid(t, w, lane, ch, indexing="ij")
    s_idx = perm[Tt * 16 + L % 16] * (HIDDEN // MX_BLOCK) + (W * RT_CH + CH) * 4 + L // 16
    return w_idx, s_idx.reshape(-1)


def pack_up_gate_kimi3(w_fp4: torch.Tensor, scales: torch.Tensor):
    w_idx, s_idx = _up_gate_index(w_fp4.device)
    w8 = w_fp4.contiguous().view(torch.uint8).reshape(-1, 2 * INTER * HIDDEN // 2)
    s8 = scales.contiguous().view(torch.uint8).reshape(-1, 2 * INTER * HIDDEN // MX_BLOCK)
    return w8[:, w_idx].reshape(-1).contiguous(), s8[:, s_idx].reshape(-1).contiguous()


@functools.cache
def _down_index(device):
    """Down MXFP4 [E, HIDDEN, INTER/2] -> per expert [224 blocks of 16 rows]
    [3 k-blocks][64 lanes][16 B]. Scales -> [224][64 lanes][3 k-blocks + pad]."""
    kh = INTER // 2
    blk, kb, lane, i = (torch.arange(n, device=device) for n in (DN_BLOCKS, INTER // 128, 64, 16))
    B, KB, L, II = torch.meshgrid(blk, kb, lane, i, indexing="ij")
    w_idx = ((B * 16 + L % 16) * kh + KB * 64 + L // 16 * 16 + II).reshape(-1)
    blk, lane, part = (torch.arange(n, device=device) for n in (DN_BLOCKS, 64, 4))
    B, L, P = torch.meshgrid(blk, lane, part, indexing="ij")
    s_idx = (B * 16 + L % 16) * (INTER // MX_BLOCK) + P.clamp(max=INTER // 128 - 1) * 4 + L // 16
    return w_idx, s_idx.reshape(-1), (P < INTER // 128).reshape(-1)


def pack_down_kimi3(w_fp4: torch.Tensor, scales: torch.Tensor):
    w_idx, s_idx, valid = _down_index(w_fp4.device)
    w8 = w_fp4.contiguous().view(torch.uint8).reshape(-1, HIDDEN * INTER // 2)
    s8 = scales.contiguous().view(torch.uint8).reshape(-1, HIDDEN * INTER // MX_BLOCK)
    ps = s8[:, s_idx]
    ps[:, ~valid] = E8M0_ONE
    return w8[:, w_idx].reshape(-1).contiguous(), ps.reshape(-1).contiguous()


def mega_moe_tp_kimi3(
    x: torch.Tensor,
    latent: torch.Tensor,
    router_w: torch.Tensor,
    bias: torch.Tensor,
    ug_w: torch.Tensor,
    ug_scales: torch.Tensor,
    down_w: torch.Tensor,
    down_scales: torch.Tensor,
    residual: torch.Tensor | None,
    sym: torch.Tensor | None,
    mype: int,
    npes: int,
    flag: int,
    scores: torch.Tensor,
    score_pairs: torch.Tensor,
    flags: torch.Tensor,
    probs_out: torch.Tensor,
    indices_out: torch.Tensor,
    hidden_mid: torch.Tensor | None,
    out: torch.Tensor,
    mid_pairs: torch.Tensor,
    sen_tag: int,
    timeline: torch.Tensor | None = None,
) -> None:
    s_n = int(x.shape[0])
    if s_n not in SUPPORTED_SAMPLES:
        raise ValueError(f"{s_n} tokens: supported {SUPPORTED_SAMPLES}")
    if not 1 <= npes <= MAX_PES or (npes > 1 and sym is None):
        raise ValueError(f"npes={npes} needs 1..{MAX_PES} and a symmetric table")
    exe = compile_mega_moe_tp_kimi3(s_n, timeline is not None, x.device.index or 0)
    _run_compiled(
        exe,
        _ptr(x),
        _ptr(latent),
        _ptr(router_w),
        _ptr(bias),
        _ptr(ug_w),
        _ptr(ug_scales),
        _ptr(down_w),
        _ptr(down_scales),
        _ptr(residual),
        _ptr(sym),
        int(mype),
        int(npes),
        int(flag) & 0x7FFFFFFF,
        _ptr(scores),
        _ptr(score_pairs),
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


class MegaMoeTpKimi3:
    """One TP rank of the Kimi-K3 routed MoE + all-reduce (1, 2 or 4 tokens
    per call). forward(x [S, 7168], latent [S, 3584], residual=None) returns
    (scores [S, 896] logits, mid [S, 16, 384], probs [S, 16], indices [S, 16],
    out [S, 3584] = all-reduced routed output (+ residual))."""

    def __init__(
        self,
        *,
        rank: int = 0,
        world_size: int = 1,
        group=None,
        device: torch.device | None = None,
        sym: torch.Tensor | None = None,
    ):
        if not 1 <= world_size <= MAX_PES:
            raise ValueError(f"world_size must be 1..{MAX_PES}")
        self.rank, self.world_size = int(rank), int(world_size)
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        i32 = {"dtype": torch.int32, "device": self.device}
        self.score_pairs = torch.zeros(MAX_SAMPLES, NUM_EXPERTS, 2, **i32)
        self.flags = torch.zeros(2 * GRID, **i32)
        self.mid_pairs = torch.zeros(MAX_SAMPLES, SLOTS, INTER, **i32)
        self._symbuf = None
        if sym is None:
            self._symbuf = UncachedSymmetricBuffer(
                SYM_BYTES, rank=self.rank, world_size=world_size, group=group, device=self.device
            )
            sym = self._symbuf.table
        self.sym = sym
        self.router_w = self.bias = None
        self.ug_w = self.ug_scales = self.down_w = self.down_scales = None

    @classmethod
    def peer_group(cls, devices) -> list[MegaMoeTpKimi3]:
        """Ranks of one process driving every device through peer access."""
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
                    if err:
                        hip.hipGetLastError()
        torch.cuda.set_device(prev)
        bufs = [torch.zeros(SYM_BYTES, dtype=torch.uint8, device=d) for d in devices]
        ptrs = [b.data_ptr() for b in bufs]
        ops = []
        for r, d in enumerate(devices):
            op = cls(rank=r, world_size=world, device=d, sym=torch.tensor(ptrs, dtype=torch.int64, device=d))
            op._peer_bufs = bufs
            ops.append(op)
        return ops

    def load_weights(
        self,
        *,
        router_w: torch.Tensor,
        bias: torch.Tensor,
        ug_w: torch.Tensor,
        ug_scales: torch.Tensor,
        down_w: torch.Tensor,
        down_scales: torch.Tensor,
    ) -> None:
        """router_w [896, 7168] bf16, bias [896], ug_w [896, 768, 1792] MXFP4
        (gate rows then up rows), ug_scales [896, 768, 112] E8M0, down_w
        [896, 3584, 192] MXFP4, down_scales [896, 3584, 12] E8M0."""
        dev = self.device
        assert router_w.shape == (NUM_EXPERTS, ROUTER_HIDDEN)
        assert ug_w.shape[1:] == (2 * INTER, HIDDEN // 2)
        assert ug_scales.shape[1:] == (2 * INTER, HIDDEN // MX_BLOCK)
        assert down_w.shape[1:] == (HIDDEN, INTER // 2)
        assert down_scales.shape[1:] == (HIDDEN, INTER // MX_BLOCK)
        self.router_w = router_w.to(dev, torch.bfloat16).contiguous()
        self.bias = bias.to(dev, torch.float32).contiguous()
        self.ug_w, self.ug_scales = pack_up_gate_kimi3(ug_w.to(dev), ug_scales.to(dev))
        self.down_w, self.down_scales = pack_down_kimi3(down_w.to(dev), down_scales.to(dev))

    def forward(
        self,
        x: torch.Tensor,
        latent: torch.Tensor,
        residual: torch.Tensor | None = None,
        *,
        out: torch.Tensor | None = None,
        sen_tag: int = 0,
        flag: int = 0,
        timeline: torch.Tensor | None = None,
    ):
        s_n = int(x.shape[0])
        scores, mid, probs, indices, out_new = self._outputs(s_n, x.device)
        if out is None:
            out = out_new
        mega_moe_tp_kimi3(
            x, latent, self.router_w, self.bias, self.ug_w, self.ug_scales, self.down_w,
            self.down_scales, residual, self.sym, self.rank, self.world_size, flag, scores,
            self.score_pairs, self.flags, probs, indices, mid, out, self.mid_pairs, sen_tag,
            timeline=timeline,
        )
        return scores, mid, probs, indices, out

    __call__ = forward

    def warmup(self, samples=SUPPORTED_SAMPLES) -> None:
        dev = self.device
        with torch.cuda.device(dev):
            for s_n in samples:
                x = torch.zeros(s_n, ROUTER_HIDDEN, dtype=torch.bfloat16, device=dev)
                lat = torch.zeros(s_n, HIDDEN, dtype=torch.bfloat16, device=dev)
                self._warm_tag = getattr(self, "_warm_tag", 1 << 30) + 1
                scores, mid, probs, indices, out = self._outputs(s_n, dev)
                mega_moe_tp_kimi3(
                    x, lat, self.router_w, self.bias, self.ug_w, self.ug_scales, self.down_w,
                    self.down_scales, None, self.sym, 0, 1, self._warm_tag, scores, self.score_pairs,
                    self.flags, probs, indices, mid, out, self.mid_pairs, self._warm_tag,
                )
            torch.cuda.synchronize(dev)

    @staticmethod
    def _outputs(s_n, dev):
        return (
            torch.empty(s_n, NUM_EXPERTS, dtype=torch.float32, device=dev),
            torch.empty(s_n, SLOTS, INTER, dtype=torch.bfloat16, device=dev),
            torch.empty(s_n, TOP_K, dtype=torch.float32, device=dev),
            torch.empty(s_n, TOP_K, dtype=torch.int32, device=dev),
            torch.empty(s_n, HIDDEN, dtype=torch.bfloat16, device=dev),
        )

    def poll_errors(self) -> dict[str, list[int]]:
        err = self.flags[GRID:].cpu()
        stages = (("scores", ERR_SCORES), ("mids", ERR_MIDS), ("allreduce", ERR_AR))
        return {name: (err & code).nonzero().flatten().tolist() for name, code in stages if bool((err & code).any())}

    def close(self) -> None:
        if self._symbuf is not None:
            self._symbuf.close()
            self._symbuf = None

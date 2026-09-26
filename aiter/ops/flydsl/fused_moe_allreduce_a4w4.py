# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import functools

import torch

from .fused_moe_allreduce import (
    SCORE_LINE_WORDS,
    FusedMoeAllreduceW8A8,
    _ptr,
    fused_moe_allreduce_w8a8_supported,
    swizzle_256_bf16,
)
from .kernels.fused_moe_allreduce.fused_moe_allreduce_a4w4 import (
    GRID,
    HIDDEN,
    INTER,
    MAX_PES,
    MX_BLOCK,
    NUM_EXPERTS,
    RT_CH,
    RT_WAVES,
    SUPPORTED_SAMPLES,
    compile_fused_moe_allreduce_a4w4,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "FusedMoeAllreduceA4W4",
    "fused_moe_allreduce_a4w4",
    "fused_moe_allreduce_a4w4_supported",
    "pack_down_a4w4",
    "pack_up_gate_a4w4",
]

E8M0_ONE = 127
fused_moe_allreduce_a4w4_supported = fused_moe_allreduce_w8a8_supported


@functools.cache
def _up_gate_a4w4_index(inter: int, k: int, device):
    rows = 2 * inter
    kb_n = k // 128
    p = torch.arange(rows, device=device)
    perm = (p % 16 // 8) * inter + (p // 16) * 8 + p % 8
    t, kb, lane, i = (torch.arange(n, device=device) for n in (rows // 16, kb_n, 64, 16))
    T, KB, L, II = torch.meshgrid(t, kb, lane, i, indexing="ij")
    w_idx = (perm[T * 16 + L % 16] * (k // 2) + KB * 64 + L // 16 * 16 + II).reshape(-1)
    t, w, lane, ch = (torch.arange(n, device=device) for n in (rows // 16, RT_WAVES, 64, 8))
    T, W, L, CH = torch.meshgrid(t, w, lane, ch, indexing="ij")
    kbs = W * RT_CH + CH
    valid = (CH < RT_CH) & (kbs < kb_n)
    s_idx = perm[T * 16 + L % 16] * (k // MX_BLOCK) + kbs.clamp(max=kb_n - 1) * 4 + L // 16
    return w_idx, s_idx.reshape(-1), valid.reshape(-1)


def pack_up_gate_a4w4(w_fp4: torch.Tensor, scales: torch.Tensor, inter: int = INTER):
    k = w_fp4.shape[-1] * 2
    w_idx, s_idx, valid = _up_gate_a4w4_index(inter, k, w_fp4.device)
    w8 = w_fp4.contiguous().view(torch.uint8).reshape(-1, 2 * inter * k // 2)
    s8 = scales.contiguous().view(torch.uint8).reshape(-1, 2 * inter * k // MX_BLOCK)
    ps = s8[:, s_idx]
    ps[:, ~valid] = E8M0_ONE
    return w8[:, w_idx].reshape(-1).contiguous(), ps.reshape(-1).contiguous()


@functools.cache
def _down_a4w4_index(rows: int, device):
    kh = INTER // 2
    blk, part, lane, i = (torch.arange(n, device=device) for n in (rows // 24, 3, 64, 16))
    B, P, L, II = torch.meshgrid(blk, part, lane, i, indexing="ij")
    tail = P == 2
    row = B * 24 + torch.where(tail, 16 + (L & 7), L % 16)
    kblk = torch.where(tail, (L >> 3) & 1, P)
    w_idx = (row * kh + kblk * 64 + L // 16 * 16 + II).reshape(-1)
    blk, lane, part = (torch.arange(n, device=device) for n in (rows // 24, 64, 4))
    B, L, P = torch.meshgrid(blk, lane, part, indexing="ij")
    tail = P == 2
    row = B * 24 + torch.where(tail, 16 + (L & 7), L % 16)
    kblk = torch.where(tail, (L >> 3) & 1, P.clamp(max=1))
    s_idx = row * (INTER // MX_BLOCK) + kblk * 4 + L // 16
    return w_idx, s_idx.reshape(-1), (P < 3).reshape(-1)


def pack_down_a4w4(w_fp4: torch.Tensor, scales: torch.Tensor):
    rows = w_fp4.shape[-2]
    assert rows % 24 == 0 and w_fp4.shape[-1] == INTER // 2
    w_idx, s_idx, valid = _down_a4w4_index(rows, w_fp4.device)
    w8 = w_fp4.contiguous().view(torch.uint8).reshape(-1, rows * INTER // 2)
    s8 = scales.contiguous().view(torch.uint8).reshape(-1, rows * INTER // MX_BLOCK)
    ps = s8[:, s_idx]
    ps[:, ~valid] = E8M0_ONE
    return w8[:, w_idx].reshape(-1).contiguous(), ps.reshape(-1).contiguous()


def fused_moe_allreduce_a4w4(
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
    exe = compile_fused_moe_allreduce_a4w4(
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


class FusedMoeAllreduceA4W4(FusedMoeAllreduceW8A8):
    _op = staticmethod(fused_moe_allreduce_a4w4)

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
        assert ug_w.shape[1:] == (2 * INTER, HIDDEN // 2)
        assert ug_scales.shape[1:] == (2 * INTER, HIDDEN // MX_BLOCK)
        assert down_w.shape[1:] == (HIDDEN, INTER // 2)
        assert down_scales.shape[1:] == (HIDDEN, INTER // MX_BLOCK)
        self.router_w = swizzle_256_bf16(router_w.to(dev))
        self.gamma = gamma.to(dev, torch.float32).contiguous()
        self.bias = bias.to(dev, torch.float32).contiguous()
        self.ug_w, self.ug_scales = pack_up_gate_a4w4(ug_w.to(dev), ug_scales.to(dev))
        self.down_w, self.down_scales = pack_down_a4w4(down_w.to(dev), down_scales.to(dev))


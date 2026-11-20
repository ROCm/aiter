# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Routed experts of one expert-parallel rank: MXFP8 activations x MXFP4 weights.

Hidden 5120, intermediate 2304, top-3 or top-6 routing, up to 63 local experts,
1-1024 tokens per call, gfx950. One sort launch (with the bf16 -> MXFP8 quant
when the input is bf16) and one fused launch: GEMM1 + clamp-SwiGLU (limit 10) +
MXFP8 requant, GEMM2 + route weight, and the deterministic top-k combine.
No output atomics: every output element has one writer, so runs are bitwise
repeatable and graph replay equals eager.

Numerics per (token, slot) route of expert ``e``: ``gate, up = x @ w13[e].T`` in
fp32, ``act = silu(min(gate, 10)) * clamp(up, -10, 10)`` in fp32, ``act``
quantized to MXFP8 per 32 columns (e8m0 = exponent of
``ceil_pow2(amax * fp32(1 / 448))``, at least 1), ``slot = bf16(w * (q @ w2[e].T))``,
``out = bf16(fp32 sum of the slots in slot order)``.

``moe_a8w4_experts_supported`` is the eligibility check; callers fall back to
``aiter.fused_moe`` when it returns False.

Ready counters: the fused launch orders its GEMM1, GEMM2 and combine workgroups
through a counter workspace that is zero between calls (the last reader of each
counter resets it). Calls that may run at the same time (e.g. on two streams)
must each pass their own workspace (``flydsl_moe_a8w4_experts_workspace``):
two such calls on one workspace add into the same counters, so a consumer can
pass its wait early (silent wrong outputs) and the counters can be left
non-zero, which corrupts or hangs every later call on that workspace, not only
the overlapping ones. Without ``ready_counters`` a call uses one workspace per
device. ``AITER_FLYDSL_MOE_A8W4_CHECK=1`` synchronizes around each eager call
and raises if the workspace is not zero before or after it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.moe_a8w4_experts import (
    COMBINE_WGS,
    NE_MAX,
    READY_STRIDE,
    TOPKS,
    H,
    I,
    compile_blocks,
    compile_combine,
    compile_experts,
    compile_gemm1,
    compile_gemm2,
    compile_sort,
    max_blocks,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "A8W4ExpertWeights",
    "flydsl_moe_a8w4_experts",
    "flydsl_moe_a8w4_experts_rows",
    "flydsl_moe_a8w4_experts_warmup",
    "flydsl_moe_a8w4_experts_workspace",
    "moe_a8w4_experts_supported",
]

MAX_TOKENS = 1024
SWIGLU_LIMIT = 10.0

# Per top-k: upper token bound -> (bm, GEMM1 K-waves, GEMM1 prefetch, GEMM1 16-column
# pairs per wave, GEMM1 ring stages, GEMM2 columns, GEMM2 prefetch, GEMM2 ring
# stages, cached weight loads for experts with several blocks (1 GEMM2, 2 GEMM1,
# 3 both), waves per workgroup, tokens per combine workgroup). Fastest per tier
# over uniform and skewed routing (48 experts top-6, 16 experts top-3), MI355X.
_DISPATCH = {
    6: (
        (48, (16, 4, 3, 2, 0, 256, 4, 0, 1, 4, 1)),
        (192, (32, 1, 3, 2, 3, 256, 4, 3, 3, 4, 1)),
        (448, (64, 1, 2, 2, 4, 256, 3, 6, 1, 8, 2)),
        (MAX_TOKENS, (128, 1, 2, 2, 3, 512, 2, 3, 1, 8, 4)),
    ),
    3: (
        (48, (16, 4, 3, 2, 0, 128, 4, 0, 1, 4, 1)),
        (100, (32, 4, 2, 2, 0, 256, 4, 3, 1, 4, 1)),
        (400, (64, 1, 2, 2, 4, 256, 3, 6, 1, 8, 2)),
        (MAX_TOKENS, (128, 1, 2, 2, 3, 256, 3, 3, 1, 8, 4)),
    ),
}
# Token counts the sort kernel is compiled for (its LDS tables are sized by the tier).
_SORT_TIERS = {
    6: (16, 32, 64, 128, 192, 256, 384, 448, 512, 768, MAX_TOKENS),
    3: (16, 32, 64, 128, 192, 256, 384, 512, 768, MAX_TOKENS),
}


def _config(n_tokens: int, topk: int):
    return next(cfg for bound, cfg in _DISPATCH[topk] if n_tokens <= bound)


def _sort_tier(n_tokens: int, topk: int) -> int:
    return next(t for t in _SORT_TIERS[topk] if n_tokens <= t)


def _warm_up_tokens(topk: int) -> tuple[int, ...]:
    bounds = {bound for bound, _ in _DISPATCH[topk]}
    return tuple(sorted(set(_SORT_TIERS[topk]) | bounds | {1}))


def _launch(exe, device, *args):
    _run_compiled(exe, *args, specialization_key=device.index)


def moe_a8w4_experts_supported(
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    topk: int,
    num_tokens: int | None = None,
    swiglu_limit: float | None = SWIGLU_LIMIT,
) -> bool:
    """Whether ``flydsl_moe_a8w4_experts`` serves this shape on the current GPU.

    Args:
        intermediate_size: per-expert intermediate size (w13 has twice as many rows).
        num_local_experts: experts held by this rank (the weights' first dimension).
        num_tokens: tokens per call; ``None`` skips the token bound (1..1024).
        swiglu_limit: the clamp of the SwiGLU inputs; the kernels implement 10.
    """
    return (
        get_gfx_runtime() == "gfx950"
        and hidden_size == H
        and intermediate_size == I
        and topk in TOPKS
        and 1 <= num_local_experts <= NE_MAX
        and swiglu_limit == SWIGLU_LIMIT
        and (num_tokens is None or 1 <= num_tokens <= MAX_TOKENS)
    )


@dataclass
class A8W4ExpertWeights:
    """One layer's local experts in the kernel layouts.

    ``w13`` / ``w2`` are the MXFP4 codes in the layout of
    ``shuffle_weight_a16w4(w, 16, gate_up)`` (gate rows then up rows before the
    shuffle), ``w13_scale`` / ``w2_scale`` the e8m0 scales in the layout of
    ``shuffle_scale_a16w4(s, E, gate_up)``: the tensors ``aiter.fused_moe``'s
    a8w4 path already holds, viewed as uint8.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor

    @property
    def num_experts(self) -> int:
        return self.w13.numel() // (2 * I * (H // 2))

    @classmethod
    def from_shuffled(cls, w13, w13_scale, w2, w2_scale) -> A8W4ExpertWeights:
        """Wrap preshuffled tensors (``fused_moe`` a8w4 layout) without a copy."""
        u8 = torch.uint8
        w = cls(
            w13.view(u8).contiguous(),
            w13_scale.view(u8).contiguous(),
            w2.view(u8).contiguous(),
            w2_scale.view(u8).contiguous(),
        )
        n = w.num_experts
        assert 1 <= n <= NE_MAX, f"{n} local experts, at most {NE_MAX}"
        assert w.w13.numel() == n * 2 * I * (H // 2)
        assert w.w13_scale.numel() == n * 2 * I * (H // 32)
        assert w.w2.numel() == n * H * (I // 2)
        assert w.w2_scale.numel() == n * H * (I // 32)
        return w

    @classmethod
    def from_row_major(cls, w13, w13_scale, w2, w2_scale) -> A8W4ExpertWeights:
        """From checkpoint layouts: ``w13 [E, 4608, 2560]`` (gate rows then up rows),
        ``w13_scale [E, 4608, 160]``, ``w2 [E, 5120, 1152]``, ``w2_scale [E, 5120, 72]``
        (packed FP4 / e8m0 bytes). Run once, before graph capture."""
        from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

        n = w13.shape[0]
        assert 1 <= n <= NE_MAX
        assert tuple(w13.shape[1:]) == (2 * I, H // 2)
        assert tuple(w2.shape[1:]) == (H, I // 2)
        u8 = torch.uint8
        return cls.from_shuffled(
            shuffle_weight_a16w4(w13.view(u8), 16, True),
            shuffle_scale_a16w4(w13_scale.view(u8).reshape(-1, H // 32), n, True),
            shuffle_weight_a16w4(w2.view(u8), 16, False),
            shuffle_scale_a16w4(w2_scale.view(u8).reshape(-1, I // 32), n, False),
        )


_X_DTYPES = (torch.bfloat16, torch.float8_e4m3fn, torch.uint8)
_SCALE_DTYPES = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))


def _activation(x, x_scales):
    """(codes, scales, quantize): MXFP8 rows pass through, bf16 rows are quantized in the
    routing launch."""
    assert x.dim() == 2 and x.shape[1] == H and x.is_contiguous()
    assert x.dtype in _X_DTYPES, f"x: {x.dtype}, expected one of {_X_DTYPES}"
    if x.dtype == torch.bfloat16:
        assert x_scales is None, "bf16 input takes no scales"
        dev = x.device
        return (
            torch.empty(x.shape, dtype=torch.uint8, device=dev),
            torch.empty((x.shape[0], H // 32), dtype=torch.uint8, device=dev),
            True,
        )
    assert x_scales is not None and x_scales.is_contiguous()
    assert x_scales.dtype in _SCALE_DTYPES, f"x_scales: {x_scales.dtype}"
    assert x_scales.numel() == x.shape[0] * (H // 32)
    assert x_scales.device == x.device
    return x.view(torch.uint8), x_scales.view(torch.uint8), False


def _sort(topk_ids, topk_weights, block_m, ne, x=None, xq=None, xs=None):
    """Rows sorted by expert in blocks of ``block_m``: (ids, weights, block experts,
    padded row count). With bf16 ``x`` the launch also quantizes it into ``xq`` / ``xs``.
    """
    n, topk = topk_ids.shape
    tier = _sort_tier(n, topk)
    ms = max_blocks(tier, block_m, ne, topk) * block_m
    dev = topk_ids.device
    ids = torch.empty(ms, dtype=torch.int32, device=dev)
    wts = torch.empty(ms, dtype=torch.float32, device=dev)
    # + 1: a tile also reads the next block's expert (it only picks the cache policy)
    eids = torch.empty(ms // block_m + 1, dtype=torch.int32, device=dev)
    nv = torch.empty(1, dtype=torch.int32, device=dev)
    quant = x is not None
    flat = topk_ids.contiguous().int().view(-1)
    exe = compile_sort(block_m=block_m, max_tokens=tier, ne=ne, topk=topk, quant=quant)
    _launch(
        exe,
        dev,
        flat,
        topk_weights.contiguous().float().view(-1),
        ids,
        wts,
        eids,
        nv,
        x.view(-1) if quant else flat,
        xq.view(-1) if quant else flat,
        xs.view(-1) if quant else flat,
        n,
        torch.cuda.current_stream(),
    )
    return ids, wts, eids, nv


def _blocks(expert_offsets, row_weights, block_m, topk, x=None, xq=None, xs=None):
    """The block table of rows already sorted by expert: as ``_sort`` with ids = input rows."""
    n = row_weights.numel()
    ne = expert_offsets.numel() - 1
    tier = topk * _sort_tier(-(-n // topk), topk)
    ms = (tier // block_m + ne) * block_m
    dev = row_weights.device
    ids = torch.empty(ms, dtype=torch.int32, device=dev)
    wts = torch.empty(ms, dtype=torch.float32, device=dev)
    eids = torch.empty(ms // block_m + 1, dtype=torch.int32, device=dev)  # as in _sort
    nv = torch.empty(1, dtype=torch.int32, device=dev)
    quant = x is not None
    off = expert_offsets.contiguous().int()
    exe = compile_blocks(block_m=block_m, max_rows=tier, ne=ne, quant=quant)
    _launch(
        exe,
        dev,
        off,
        row_weights.contiguous().float(),
        ids,
        wts,
        eids,
        nv,
        x.view(-1) if quant else off,
        xq.view(-1) if quant else off,
        xs.view(-1) if quant else off,
        n,
        torch.cuda.current_stream(),
    )
    return ids, wts, eids, nv


_READY_WORDS = (max_blocks(MAX_TOKENS, 16, NE_MAX, max(TOPKS)) + 1) * READY_STRIDE // 4
_ready: dict = {}


def flydsl_moe_a8w4_experts_workspace(device=None) -> torch.Tensor:
    """A new ready-counter workspace (int32, zero) for ``ready_counters=``.

    Give each stream that may run a call while another is in flight its own
    workspace; allocate it before graph capture. It stays zero between calls.
    """
    return torch.zeros(_READY_WORDS, dtype=torch.int32, device=device or "cuda")


def _ready_counters(device, ready_counters=None):
    """The counters of the fused launch: ``ready_counters`` or the device's shared
    workspace (see the module docstring on concurrent calls)."""
    if ready_counters is not None:
        assert ready_counters.dtype == torch.int32 and ready_counters.is_contiguous()
        assert ready_counters.device == device, "ready_counters on another device"
        assert (
            ready_counters.numel() >= _READY_WORDS
        ), "use flydsl_moe_a8w4_experts_workspace"
        return ready_counters
    arena = _ready.get(device)
    if arena is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "flydsl_moe_a8w4_experts: call flydsl_moe_a8w4_experts_warmup (or run "
                "once) before graph capture; the first call allocates the counters"
            )
        arena = flydsl_moe_a8w4_experts_workspace(device)
        _ready[device] = arena
    return arena


def _check_counters(arena, when):
    """``AITER_FLYDSL_MOE_A8W4_CHECK=1``: the workspace must be zero ``when``."""
    if os.environ.get("AITER_FLYDSL_MOE_A8W4_CHECK", "0") != "1":
        return
    if torch.cuda.is_current_stream_capturing():
        return
    torch.cuda.current_stream(arena.device).synchronize()
    if bool(arena.any()):
        raise RuntimeError(
            f"flydsl_moe_a8w4_experts: ready counters not zero {when} the call: two "
            "calls ran at the same time on one workspace (pass each stream its own "
            "ready_counters); every later call on this workspace is unreliable"
        )


def _fused(
    xq, xs, w, topk, tid, ids, wts, eids, nv, n_rows, nmb, *, cfg, rows_out, ready
):
    bm, kw, pf1, ni, nb1, tn, pf2, nb2, hot, nwv, cw = cfg
    # the combine workgroups of the token entry cover every token
    assert rows_out or cw * COMBINE_WGS >= n_rows, (cw, n_rows)
    rows = ids.numel()
    dev = xq.device
    counters = _ready_counters(dev, ready)
    _check_counters(counters, "before")
    q = torch.empty(rows * I, dtype=torch.uint8, device=dev)
    s = torch.empty(rows * (I // 32), dtype=torch.uint8, device=dev)
    slots = torch.empty(
        0 if rows_out else n_rows * topk * H, dtype=torch.bfloat16, device=dev
    )
    out = torch.empty((n_rows, H), dtype=torch.bfloat16, device=dev)
    exe = compile_experts(
        bm=bm,
        kw=kw,
        pf1=pf1,
        ni=ni,
        nb1=nb1,
        tn2=tn,
        pf2=pf2,
        nb2=nb2,
        ne=w.num_experts,
        topk=topk,
        hot=bool(hot & 1),
        nwv=nwv,
        cw=cw,
        rows_out=rows_out,
        hot1=bool(hot & 2),
    )
    _launch(
        exe,
        dev,
        xq.view(-1),
        xs.view(-1),
        w.w13.view(-1),
        w.w13_scale.view(-1),
        q,
        s,
        w.w2.view(-1),
        w.w2_scale.view(-1),
        slots,
        out.view(-1),
        tid,
        ids,
        wts,
        eids,
        nv,
        counters,
        n_rows,
        rows,
        nmb,
        torch.cuda.current_stream(),
    )
    _check_counters(counters, "after")
    return out


def _slots(xq, xs, w, topk, ids, wts, eids, nv, n, cfg):
    """Unfused GEMM1 and GEMM2: the route-weighted bf16 slots ``[n, topk, 5120]``."""
    bm, kw, pf1, ni, nb1, tn, pf2, nb2, hot, nwv, _ = cfg
    ne = w.num_experts
    dev = xq.device
    rows = ids.numel()
    nmb = max_blocks(n, bm, ne, topk)
    q = torch.empty(rows * I, dtype=torch.uint8, device=dev)
    s = torch.empty(rows * (I // 32), dtype=torch.uint8, device=dev)
    exe1 = compile_gemm1(
        ne=ne, kw=kw, pf=pf1, bm=bm, ni=ni, nbuf=nb1, nwv=nwv, hot=bool(hot & 2)
    )
    _launch(
        exe1,
        dev,
        xq.view(-1),
        xs.view(-1),
        w.w13.view(-1),
        w.w13_scale.view(-1),
        q,
        s,
        ids,
        eids,
        nv,
        n,
        rows,
        nmb,
        torch.cuda.current_stream(),
    )
    slots = torch.zeros((n, topk, H), dtype=torch.bfloat16, device=dev)
    exe2 = compile_gemm2(
        ne=ne, topk=topk, tn=tn, bm=bm, pf=pf2, nbuf=nb2, hot=bool(hot & 1), nwv=nwv
    )
    _launch(
        exe2,
        dev,
        q,
        s,
        w.w2.view(-1),
        w.w2_scale.view(-1),
        slots.view(-1),
        ids,
        wts,
        eids,
        nv,
        n,
        rows,
        nmb,
        torch.cuda.current_stream(),
    )
    return slots


def combine_slots(slots: torch.Tensor, topk_ids: torch.Tensor, num_experts: int):
    """bf16 ``[n, topk, 5120]`` slots -> ``[n, 5120]``: the fp32 sum in slot order,
    a slot whose id is outside [0, num_experts) counting as zero."""
    n, topk, _ = slots.shape
    out = torch.empty((n, H), dtype=torch.bfloat16, device=slots.device)
    _launch(
        compile_combine(ne=num_experts, topk=topk),
        slots.device,
        topk_ids.contiguous().int().view(-1),
        slots.contiguous().view(-1),
        out.view(-1),
        n,
        torch.cuda.current_stream(),
    )
    return out


def flydsl_moe_a8w4_experts(
    x: torch.Tensor,
    x_scales: torch.Tensor | None,
    w: A8W4ExpertWeights,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    return_slots: bool = False,
    ready_counters: torch.Tensor | None = None,
) -> torch.Tensor:
    """The routed experts of each token.

    Args:
        x: MXFP8 e4m3 codes ``[n, 5120]`` (float8_e4m3fn or uint8) with ``x_scales``
            e8m0 ``[n, 160]`` row-major, or bf16 ``[n, 5120]`` with ``x_scales`` None
            (quantized in the routing launch).
        topk_ids: int32 ``[n, topk]`` local expert ids, topk 3 or 6; an id outside
            [0, w.num_experts) (padding, another rank's expert) is not routed.
        topk_weights: ``[n, topk]`` route weights (read as fp32).
        return_slots: return the route-weighted bf16 slots ``[n, topk, 5120]``
            (unfused GEMM1 / GEMM2 launches) instead of their sum.
        ready_counters: the counter workspace
            (``flydsl_moe_a8w4_experts_workspace``); required for calls that may
            overlap another call on the same device (module docstring).

    Returns:
        bf16 ``[n, 5120]`` (``n`` 0 gives an empty tensor and launches nothing).
    """
    n, topk = topk_ids.shape
    assert topk in TOPKS, f"topk {topk} not in {TOPKS}"
    assert 0 <= n <= MAX_TOKENS, f"{n} tokens, 0..{MAX_TOKENS}"
    assert x.shape[0] == n and topk_weights.shape == (n, topk)
    assert topk_ids.device == x.device and topk_weights.device == x.device
    assert (
        w.w13.device == x.device and w.w2.device == x.device
    ), "weights on another device"
    if n == 0:
        shape = (0, topk, H) if return_slots else (0, H)
        return torch.empty(shape, dtype=torch.bfloat16, device=x.device)
    cfg = _config(n, topk)
    bm = cfg[0]
    xq, xs, quant = _activation(x, x_scales)
    ne = w.num_experts
    ids, wts, eids, nv = _sort(
        topk_ids, topk_weights, bm, ne, *((x, xq, xs) if quant else ())
    )
    if return_slots:
        return _slots(xq, xs, w, topk, ids, wts, eids, nv, n, cfg)
    tid = topk_ids.contiguous().int().view(-1)
    nmb = max_blocks(n, bm, ne, topk)
    return _fused(
        xq,
        xs,
        w,
        topk,
        tid,
        ids,
        wts,
        eids,
        nv,
        n,
        nmb,
        cfg=cfg,
        rows_out=False,
        ready=ready_counters,
    )


def flydsl_moe_a8w4_experts_rows(
    x: torch.Tensor,
    x_scales: torch.Tensor | None,
    w: A8W4ExpertWeights,
    row_weights: torch.Tensor,
    expert_offsets: torch.Tensor,
    topk: int = 6,
    *,
    ready_counters: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rows already sorted by expert (an expert-parallel dispatch's receive buffer).

    Args:
        x: ``[R, 5120]`` MXFP8 codes + ``x_scales`` ``[R, 160]``, or bf16 with None.
        row_weights: fp32 ``[R]``.
        expert_offsets: int32 ``[w.num_experts + 1]``: expert ``e`` owns rows
            ``expert_offsets[e]`` .. ``expert_offsets[e + 1]``; the last entry is the
            row count (at most ``R``).
        topk: the routing's top-k; picks the dispatch tier (``R / topk`` tokens).
        ready_counters: as in ``flydsl_moe_a8w4_experts``.

    Returns:
        bf16 ``[R, 5120]``: row ``r`` = ``bf16(row_weights[r] * expert(x[r]))`` for
        ``r < expert_offsets[-1]``; the rows from ``expert_offsets[-1]`` on are
        undefined (not written; the output is allocated uninitialized). ``R`` 0
        gives an empty tensor and launches nothing.
    """
    n = x.shape[0]
    assert topk in TOPKS
    assert row_weights.shape == (n,)
    assert expert_offsets.shape == (w.num_experts + 1,)
    assert 0 <= n <= topk * MAX_TOKENS
    assert row_weights.device == x.device and expert_offsets.device == x.device
    assert (
        w.w13.device == x.device and w.w2.device == x.device
    ), "weights on another device"
    if n == 0:
        return torch.empty((0, H), dtype=torch.bfloat16, device=x.device)
    cfg = _config(-(-n // topk), topk)
    bm = cfg[0]
    xq, xs, quant = _activation(x, x_scales)
    ids, wts, eids, nv = _blocks(
        expert_offsets, row_weights, bm, topk, *((x, xq, xs) if quant else ())
    )
    nmb = n // bm + min(w.num_experts, n)
    return _fused(
        xq,
        xs,
        w,
        topk,
        ids,
        ids,
        wts,
        eids,
        nv,
        n,
        nmb,
        cfg=cfg,
        rows_out=True,
        ready=ready_counters,
    )


@torch.inference_mode()
def flydsl_moe_a8w4_experts_warmup(
    w: A8W4ExpertWeights,
    topk: int,
    *,
    bf16_input: bool = False,
    rows: bool = False,
    return_slots: bool = False,
    device: torch.device | None = None,
) -> None:
    """Compile every dispatch and sort tier of one mode and allocate the counters.

    Run before graph capture: a capture must not compile or allocate.
    """
    dev = device or w.w13.device
    dtype = torch.bfloat16 if bf16_input else torch.uint8
    for t in _warm_up_tokens(topk):
        n = t * topk if rows else t
        x = torch.zeros((n, H), dtype=dtype, device=dev)
        xs = (
            None
            if bf16_input
            else torch.zeros((n, H // 32), dtype=torch.uint8, device=dev)
        )
        if rows:
            offsets = torch.zeros(w.num_experts + 1, dtype=torch.int32, device=dev)
            weights = torch.ones(n, dtype=torch.float32, device=dev)
            flydsl_moe_a8w4_experts_rows(x, xs, w, weights, offsets, topk)
        else:
            ids = torch.zeros((n, topk), dtype=torch.int32, device=dev)
            weights = torch.ones((n, topk), dtype=torch.float32, device=dev)
            flydsl_moe_a8w4_experts(x, xs, w, weights, ids, return_slots=return_slots)
    torch.cuda.synchronize(dev)

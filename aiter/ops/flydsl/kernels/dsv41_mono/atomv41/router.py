# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""V4.1's router GEMV for decode: bf16 x [T, 5120] times the gate weight
[384, 5120] -> bf16 logits, T <= 16, in one fixed summation order.

The mono decode's MoE kernel (K2b) runs the same tile (its loads and MFMAs,
``router_weight_loads`` / ``router_x_loads`` / ``router_tile_mfma``, and
``atom.mono.device.ops.row_sum``), so both paths route alike bit for bit. The gate's
``tgemm.mm`` would reach hipBLASLt instead, whose Stream-K kernel splits K over
workgroups by the CU count: an order no other kernel can follow.

Rows 16 a tile, K in ``PARTS`` contiguous parts, 16x16x32 bf16 MFMA: in part
p wave w takes the part's K steps w, w + 8, ... (32 columns each) in that
order, the eight wave partials are summed in wave order, and the parts' sums
in part order, rounded once to bf16. This kernel runs a tile's parts in one CTA
(24 CTAs, 512 threads); K2b spreads them over PARTS CTAs, and past 16 tokens
a token tile a CTA too, which the order allows: a CTA alone cannot read a
tile, or every token's x, fast enough.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int64, T

from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.ops import (
    bf16_pair,
    kernel_symbol,
    mfma_bf16,
    row_sum,
    rsrc,
    traced,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.execution import THREADS, WAVES

EXPERTS = 384
HIDDEN = 5120
ROWS = 16  # a tile's rows: the MFMA's M
MAX_TOKENS = 16  # the MFMA's N
PARTS = 4  # contiguous K parts of a tile, summed in order
_CURRENT_STREAM = fx.Stream(None)


@traced
def router_tile(x, w, task, part, s, lane, wave, red):
    """Wave ``wave``'s partial of rows 16 task .., K part ``part``, for the s
    tokens of ``x`` (raw bf16 pointers) -> red[wave][lane] (4 f32: rows
    4 (lane / 16) .., token lane % 16). All of the wave's loads issue before its
    MFMAs."""
    wvs = router_weight_loads(w, task, part, lane, wave)
    xvs = router_x_loads(x, part, s, lane, wave)
    router_tile_mfma(wvs, xvs, lane, wave, red)


def _k0s(part, lane, wave):
    """The K offsets of this lane's loads in part ``part`` (8 bf16 each)."""
    per_wave = HIDDEN // 32 // WAVES // PARTS
    assert per_wave * 32 * WAVES * PARTS == HIDDEN
    return [
        (part * per_wave * WAVES + wave + WAVES * i) * 32 + 8 * (lane // 16)
        for i in range(per_wave)
    ]


def router_weight_loads(w, task, part, lane, wave, w_cm=0):
    """``router_tile``'s weight operands (rows 16 task .., K part ``part``).
    ``w_cm``: their load cache policy (nontemporal where nothing else of the
    step reuses them)."""
    row = task * ROWS + lane % 16
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(w),
                (row * HIDDEN + k0) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=w_cm,
            )
        )
        for k0 in _k0s(part, lane, wave)
    ]


def router_x_loads(x, part, s, lane, wave, x_cm=0, t0=0):
    """``router_tile``'s x operands: the s tokens from ``t0`` (column
    lane % 16; ``t0`` traced or an int). ``x_cm``: their load cache policy
    (device scope when another CTA of the same launch wrote x)."""
    tok = fx.min(lane % 16, s - 1)
    if const_expr(not isinstance(t0, int) or t0 != 0):
        tok = tok + t0
    return [
        fx.Vector(
            bo.buffer_load(
                rsrc(x),
                (tok * HIDDEN + k0) // 2,
                vec_width=4,
                dtype=T.i32,
                cache_modifier=x_cm,
            )
        )
        for k0 in _k0s(part, lane, wave)
    ]


def router_tile_mfma(wvs, xvs, lane, wave, red):
    """``router_tile``'s MFMAs, K steps in order -> red[wave][lane]."""
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(len(wvs)):
        acc = mfma_bf16(wvs[i].bitcast(fx.BFloat16), xvs[i].bitcast(fx.BFloat16), acc)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)


@functools.cache
def _build(tokens: int):
    assert 1 <= tokens <= MAX_TOKENS

    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 64 * 4, 16]

    @flyc.kernel(
        name=kernel_symbol("v41_router", s=tokens), known_block_size=[THREADS, 1, 1]
    )
    def router(x: Int64, w: Int64, out: Int64):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        red = fx.SharedAllocator().allocate(Smem).peek().red.ptr
        # a thread a (token, row pair): its two logits' part sums, in part order
        mine = tid < ROWS // 2 * tokens
        rp = tid % (ROWS // 2)
        t = fx.min(tid // (ROWS // 2), tokens - 1)
        acc0, acc1 = fx.Float32(0.0), fx.Float32(0.0)
        for part in range_constexpr(PARTS):
            router_tile(x, w, bid, part, tokens, tid % 64, tid // 64, red)
            gpu.barrier()
            acc0 = acc0 + row_sum(red, 2 * rp, t)
            acc1 = acc1 + row_sum(red, 2 * rp + 1, t)
            gpu.barrier()
        if mine:
            bo.buffer_store(
                bf16_pair(acc0, acc1).bitcast(fx.Int32),
                rsrc(out),
                (t * EXPERTS + bid * ROWS + 2 * rp) // 2,
            )

    @flyc.jit
    def launch(x: Int64, w: Int64, out: Int64, stream: fx.Stream = _CURRENT_STREAM):
        router(x, w, out).launch(
            grid=(EXPERTS // ROWS,), block=(THREADS,), stream=stream
        )

    return launch


def router_supported(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and 1 <= x.shape[0] <= MAX_TOKENS
        and x.shape[1] == HIDDEN
        and x.dtype == torch.bfloat16
        and x.is_contiguous()
        and weight.shape == (EXPERTS, HIDDEN)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
    )


def _router_logits_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty((x.shape[0], EXPERTS), dtype=torch.bfloat16, device=x.device)


@torch_compile_guard(gen_fake=_router_logits_fake, mutates_args=[])
def router_logits(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """[T, 384] bf16 logits of ``x`` (``router_supported``)."""
    out = torch.empty((x.shape[0], EXPERTS), dtype=torch.bfloat16, device=x.device)
    _build(x.shape[0])(
        x.data_ptr(),
        weight.data_ptr(),
        out.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    return out

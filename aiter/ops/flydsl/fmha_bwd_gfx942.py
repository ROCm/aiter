# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host side of the FlyDSL varlen FMHA backward (causal, bf16, gfx942, head dims 64..192).

Grid planning plus cached ``CompiledFunction`` dispatch for the kernels in
``kernels/fmha_bwd_gfx942/fmha_bwd_core.py``: a ``k_delta`` pre-pass (D = rowsum(dO*O)) and one
fused ``k_bwd`` that carries all five backward GEMMs.  The device code stays under ``kernels/``
and everything host-side lives here.  Unlike the CK and ASM backwards this needs neither a
separate ``FmhaBwdOGradDotOKernel`` nor a ``FmhaBwdConvertQGradKernel``, and it does no work on
zeros: every head dim in ``[64, 192]`` with ``d % 8 == 0`` gets its own natively tiled kernel set,
so v/out/dout/dv are never padded up to the next supported size the way the ASM backward has to
pad them (it ships hd64 / hd128 / hd192 code objects and nothing in between).

``(d_qk, d_v)`` selects the kernel set; each pair compiles on first use and is cached.  Which
shapes are actually ROUTED here is decided by ``can_impl_fmha_bwd_flydsl`` in
``aiter/ops/mha.py``.

The launcher never synchronises with the device: the grid is sized purely from tensor shapes and
``max_seqlen_*`` (both host-side), and the per-sequence bounds are read from ``cu_seqlens`` on
the GPU inside the kernels.

Contract (asserted by the caller in ``aiter/ops/flydsl/fmha_kernels.py``):
  * causal, bf16, ``64 <= d_qk, d_v <= 192`` with ``d % 8 == 0``
  * THD varlen self-attention -- one ``cu_seqlens`` for both q and k, ``nhead_q == nhead_k``
  * q, k contiguous ``[T, H, d_qk]``; v, out, dout contiguous ``[T, H, d_v]``;
    lse contiguous ``[H, T]`` fp32; cu_seqlens ``[B+1]`` int32
  * ``P = exp(softmax_scale * Q@K^T - LSE)`` (natural log, scale folded in, causal j <= i)
"""

from __future__ import annotations

import functools
import os

import torch

from ...jit.utils.chip_info import get_cu_num
from .kernels.fmha_bwd_gfx942.fmha_bwd_core import (
    D_MAX,
    D_MIN,
    D_STEP,
    KEYS_PER_WG,
    QUERIES_PER_WG,
    QUERY_TILE,
    RED_THREADS,
    RED_VEC,
    ROWS_DELTA,
    _geom,
    build,
    fused_role_enabled,
    supported_head_dim,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "flash_attn_varlen_bwd_gfx942",
    "flash_attn_varlen_bwd_d192_gfx942",
    "flydsl_bwd_shape_supported",
]

# Split-K factor override for workloads whose whole grid is co-resident (see `_split`).  A test
# hook ONLY: `1` is the way to run the n_split == 1 (and therefore FUSED) kernel on the small
# shapes the correctness tests use, which otherwise all take the split-K path.
_NSPLIT_OVERRIDE = os.environ.get("AITER_FMHA_BWD_FLYDSL_NSPLIT")


def flydsl_bwd_shape_supported(d_qk: int, d_v: int) -> bool:
    """True when a kernel set can be BUILT for this (d_qk, d_v) -- not that it should be used."""
    if not (supported_head_dim(d_qk) and supported_head_dim(d_v)):
        return False
    # d_v > d_qk is never asked for by a caller and is not exercised by any test, so it is not
    # claimed.  d_v < d_qk (the MLA shape) and d_v == d_qk both are.
    return d_v <= d_qk


@functools.lru_cache(maxsize=1)
def _num_cu() -> int:
    try:
        return int(get_cu_num())
    except (RuntimeError, AssertionError, ValueError):
        return int(
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        )


def _est_wgs(t: int, h: int, n_seqs: int, n_dkdv_blocks: int) -> int:
    """Upper-bound estimate of the n_split == 1 grid's live workgroups (dK/dV + dQ roles)."""
    return 2 * h * min(t // KEYS_PER_WG + n_seqs, n_seqs * n_dkdv_blocks)


def _interleave(est_wgs: int, ncu: int) -> int:
    """Pick the k_bwd grid-decode mode: 1 = merged-LPT interleave of the dK/dV and dQ job lists,
    0 = the "all dK/dV blocks, then all dQ blocks" concatenation.
    """
    return 1 if est_wgs <= 2 * ncu else 0


def _split(interleave: int, est_wgs: int, d_qk: int, ncu: int) -> int:
    """Split-K factor along the STREAMED index, only on a co-resident (interleaved) grid.

    Measured on MI300X against the ASM backward (tiny / 4x1024h2 / 2x4096h4 sweeps): up to
    d = 128 two splits beat both three and the fused n_split == 1 kernel (2x4096h4 d64: 1.12x
    vs 1.06x vs 0.85x).  The d >= 136 workgroups are heavier, so once the grid already covers
    every CU the extra reduction pass costs more than the parallelism buys (2x4096h4 d192:
    1.46x unsplit vs 1.25x at three); below that, three.
    """
    if not interleave:
        return 1
    if _NSPLIT_OVERRIDE:
        return int(_NSPLIT_OVERRIDE)
    if d_qk <= 128:
        return 2
    return 1 if est_wgs > ncu else 3


# ---- FUSED SPLIT-K: a COMPUTED makespan predicate --------------------------------------------
# The fused role's grid is the dK/dV block list, and the per-workgroup trip count of key block kb
# is ceil((s - kb*KEYS_PER_WG) / QUERY_TILE): causal makes the block list a LINEAR RAMP of work.
# A kernel's makespan is at least max(L, A): L the LONGEST workgroup, A the total work spread over
# the co-resident slots.  When L > A the ramp's head is the critical path and slots idle behind
# it however many workgroups there are (1x8192h8 d64: 512 WGs on 608 slots, the kb = 0 block runs
# 256 iterations against a 111 average -- 43 % slot efficiency).  Splitting each block's streamed
# query range n ways divides L by n and adds only per-workgroup fixed cost to A, at the price of
# one bf16 dK/dV partial slab per split and a `k_red` pass.  The split factor is the SMALLEST n
# that brings L down to A (more splits only add reduction traffic and fixed cost; measured
# 1x8192h8 / 2x6000h6 d64: n = 2 / 3 / 4 -> 0.512 / 0.493 / 0.499 and 0.418 / 0.406 / 0.411 ms
# against 0.618 / 0.474 unsplit), and a split is taken only when the modelled time it saves beats
# the reduction's cost.  Everything is priced from host-side shapes alone.
_SPLIT_MAX = 4
# The unsplit grid must be critical-path-bound by at least this margin to be considered.
_SPLIT_MIN_GAIN = 0.03
# Per-workgroup fixed cost in streamed-iteration units: K / V / K^T operand loads, the pipeline
# prologue and the epilogue stores.
_WG_FIXED_ITERS = 2.0
# `k_red` pricing: the dependent dispatch's gap plus its HBM traffic.
_RED_FIXED_US = 4.0
_RED_BYTES_PER_US = 5.0e6
# One streamed iteration of one fused workgroup costs ~0.25 us per MFMA d tile (d64: 8 tiles,
# d128: 16), and co-resident workgroups share one CU's MFMA / LDS throughput almost fully.
_ITER_US_PER_TILE = 0.25
# VGPR fit of the FUSED k_bwd against ISA dumps (vgpr_count 125 / 152 / 168 / 190 / 232 at
# 8 / 10 / 12 / 14 / 16 MFMA d tiles, no spills): ~13.5 per d tile over a 17 floor.  Unlike the
# core's `est_vgpr_fused` -- a deliberate UPPER bound used only to gate feasibility -- this one is
# for OCCUPANCY, so it has to land on the right side of the 128-VGPR two-workgroup boundary.  The
# split build is compiled with that occupancy PINNED (`waves_per_eu`), so the model cannot drift
# from the kernel it prices.
_VGPR_FIT_BASE = 17
_VGPR_FIT_PER_TILE = 13.5
_VGPRS_PER_SIMD_LANE = 512
_SIMDS_PER_CU = 4
_WAVES_PER_WG = 8
_LDS_PER_CU = 64 * 1024


def _fused_wgs_per_cu(d_qk: int, d_v: int) -> int:
    """Co-resident fused k_bwd workgroups per CU, from the register and LDS footprints."""
    g = _geom(d_qk, d_v)
    vgpr = _VGPR_FIT_BASE + _VGPR_FIT_PER_TILE * (g.n_tiles_qk + g.n_tiles_v)
    vgpr = -(-int(vgpr) // 8) * 8  # allocation granule
    waves_per_simd = _VGPRS_PER_SIMD_LANE // vgpr
    by_vgpr = (waves_per_simd * _SIMDS_PER_CU) // _WAVES_PER_WG
    lds = (
        g.arena_dkdv + QUERY_TILE * (KEYS_PER_WG + 4 + g.ld_dqs)
    ) * 2 + 2 * QUERY_TILE * 4
    by_lds = _LDS_PER_CU // lds
    return max(1, min(by_vgpr, by_lds))


def _model_seq_lengths(t: int, n_seqs: int, max_seqlen: int):
    """Sequence lengths as seen from the host: as many full ``max_seqlen`` sequences as fit in
    ``t`` and the remainder in one more; the rest of ``n_seqs`` (if any) is empty."""
    n_full = min(n_seqs, t // max(1, max_seqlen))
    lens = [max_seqlen] * n_full
    rem = t - n_full * max_seqlen
    if rem > 0 and n_full < n_seqs:
        lens.append(min(rem, max_seqlen))
    return lens


def _ramp_bounds(lens, h: int, n_split: int, slots: int):
    """(L, A) of the fused grid split n ways, in streamed-iteration units: the longest
    workgroup and the total work per co-resident slot."""
    longest = 0.0
    total = 0.0
    for s in lens:
        for kb in range((s + KEYS_PER_WG - 1) // KEYS_PER_WG):
            nt = -(-(s - kb * KEYS_PER_WG) // QUERY_TILE)
            longest = max(longest, -(-nt // n_split) + _WG_FIXED_ITERS)
            total += (nt + n_split * _WG_FIXED_ITERS) * h
    return longest, total / slots


def _fused_split(
    t: int, h: int, n_seqs: int, max_seqlen: int, ncu: int, d_qk: int, d_v: int
) -> int:
    """Split-K factor for the FUSED role: 1 (today's unsplit kernel) unless the grid's causal
    critical path, not its work, sets the makespan -- see the block comment above."""
    lens = _model_seq_lengths(t, n_seqs, max_seqlen)
    if not lens:
        return 1
    wgs_cu = _fused_wgs_per_cu(d_qk, d_v)
    slots = ncu * wgs_cu
    l1, a1 = _ramp_bounds(lens, h, 1, slots)
    if l1 <= (1.0 + _SPLIT_MIN_GAIN) * a1:
        return 1  # throughput-bound: splitting can only add work
    g = _geom(d_qk, d_v)
    t_iter = _ITER_US_PER_TILE * (g.n_tiles_qk + g.n_tiles_v) * wgs_cu
    kv_bytes = t * h * (d_qk + d_v) * 2
    base = max(l1, a1) * t_iter
    fallback_n, fallback_t = 1, base
    for n in range(2, _SPLIT_MAX + 1):
        ln, an = _ramp_bounds(lens, h, n, slots)
        # k_red reads n partial slabs and writes one result (the extra slab stores of k_bwd
        # itself hide under its MFMA work).
        cost = (
            max(ln, an) * t_iter
            + _RED_FIXED_US
            + (n + 1) * kv_bytes / _RED_BYTES_PER_US
        )
        if cost >= (1.0 - _SPLIT_MIN_GAIN) * base:
            continue  # the reduction eats the gain
        if ln <= an:
            return n  # smallest split whose critical path no longer sets the makespan
        if cost < fallback_t:
            fallback_n, fallback_t = n, cost
    return fallback_n


@functools.lru_cache(maxsize=256)
def _plan(
    t: int,
    h: int,
    n_seqs: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    ncu: int,
    d_qk: int,
    d_v: int,
    deterministic: bool,
):
    n_dkdv_blocks = (max_seqlen_k + KEYS_PER_WG - 1) // KEYS_PER_WG
    n_dq_blocks = (max_seqlen_q + QUERIES_PER_WG - 1) // QUERIES_PER_WG
    est_wgs = _est_wgs(t, h, n_seqs, n_dkdv_blocks)
    interleave = _interleave(est_wgs, ncu)
    n_split = _split(interleave, est_wgs, d_qk, ncu)
    elems_per_red_blk = RED_THREADS * RED_VEC
    fuse_split = False
    if n_split == 1 and fused_role_enabled(1, d_qk, d_v, deterministic):
        # The fused grid's makespan may be its causal critical path rather than its work.
        n_split = _fused_split(t, h, n_seqs, max_seqlen_k, ncu, d_qk, d_v)
        fuse_split = n_split > 1
    fused = fused_role_enabled(n_split, d_qk, d_v, deterministic, fuse_split)
    if fused:
        # ONE role: the fused kernel carries all five GEMMs on the dK/dV block list, so the dQ
        # half of the grid is deleted outright.
        n_dq_blocks = 0
    return (
        (t * h + ROWS_DELTA - 1) // ROWS_DELTA,  # 0: k_delta grid
        # 1: dK/dV list length -- scaled by n_split so the split index rides the low digit
        n_dkdv_blocks * n_split,
        (n_dkdv_blocks + n_dq_blocks) * n_split,  # 2: total list length
        interleave,  # 3: grid-decode mode
        n_split,  # 4: split-K factor
        # 5: reduction grid.x, sized for the larger (dq/dk) tensor; grid.y picks dq/dk/dv
        (t * h * d_qk + elems_per_red_blk - 1) // elems_per_red_blk,
        fused,  # 6: FUSED single-role kernel (split or not)
    )


def flash_attn_varlen_bwd_gfx942(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    dq: torch.Tensor | None = None,
    dk: torch.Tensor | None = None,
    dv: torch.Tensor | None = None,
    deterministic: bool = False,
):
    """Run the FlyDSL varlen causal backward.  Returns ``(dq, dk, dv, softmax_d)``.

    ``dq`` / ``dk`` / ``dv`` are written in place when supplied and allocated otherwise.
    ``softmax_d`` is the ``[H, T]`` fp32 ``rowsum(dO*O)`` the CK and ASM backwards also return.

    With ``deterministic=False`` head dims up to 128 run the FUSED single-role kernel, whose
    ``dq`` is a ``global_atomic_pk_add_bf16`` sum across key blocks and therefore not
    bit-reproducible run to run (``dk`` / ``dv`` always are).  ``deterministic=True`` keeps
    every head dim on the two-role kernel, where each output element has exactly one writer.
    """
    t, h, d_qk = q.shape
    d_v = v.shape[-1]
    assert flydsl_bwd_shape_supported(d_qk, d_v), (
        f"FlyDSL gfx942 backward serves d_v <= d_qk with both in "
        f"[{D_MIN}, {D_MAX}] and a multiple of {D_STEP}; got d_qk={d_qk}, d_v={d_v}"
    )
    # The kernels size the LSE buffer resource as `nrow * 4` bytes and issue fp32 loads, so a
    # narrower dtype or a transposed layout would read out of bounds and silently corrupt P.
    assert softmax_lse.dtype == torch.float32 and tuple(softmax_lse.shape) == (h, t), (
        f"lse must be fp32 [{h}, {t}], "
        f"got {softmax_lse.dtype} {tuple(softmax_lse.shape)}"
    )
    device = q.device
    n_seqs = cu_seqlens.numel() - 1
    (
        delta_blocks,
        n_dkdv_blocks,
        n_blocks,
        interleave,
        n_split,
        red_blocks,
        fused,
    ) = _plan(
        t,
        h,
        n_seqs,
        int(max_seqlen_q),
        int(max_seqlen_k),
        _num_cu(),
        d_qk,
        d_v,
        bool(deterministic),
    )
    # Largest buffer one descriptor spans: dq itself, or the split-K [dq | dk | dv] workspace
    # ([dk | dv] on the fused split-K kernel, whose dq is accumulated in place).
    if n_split == 1:
        per_token_head = d_qk
    elif fused:
        per_token_head = (d_qk + d_v) * n_split
    else:
        per_token_head = (2 * d_qk + d_v) * n_split
    ws_bytes = t * h * per_token_head * 2
    assert ws_bytes < (1 << 31), (
        f"FlyDSL gfx942 backward addresses tensors through 32-bit buffer descriptors; "
        f"T={t} x H={h} needs a {ws_bytes / 2**30:.2f} GiB buffer (>= 2 GiB). "
        f"Split the batch so T*H < {(1 << 31) // (per_token_head * 2)}."
    )

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.empty_like(k)
    if dv is None:
        dv = torch.empty_like(v)
    if not softmax_lse.is_contiguous():
        softmax_lse = softmax_lse.contiguous()

    stream = torch.cuda.current_stream(device)
    delta = torch.empty((h, t), dtype=torch.float32, device=device)

    # Only the selected variant is compiled: `k_delta` does not depend on `n_split`, so the
    # split-K build supplies an identical `launch_delta` and the n_split == 1 kernel set is never
    # built on a shape that will not use it.
    # `deterministic` normalised to "not fused" so a head dim the fused role does not cover
    # shares ONE two-role kernel set between the two modes.
    if fused and n_split > 1:
        kernels = build(n_split, d_qk, d_v, False, True, _fused_wgs_per_cu(d_qk, d_v))
    else:
        kernels = build(n_split, d_qk, d_v, not fused)
    launch_delta, launch_bwd = kernels[0], kernels[1]

    # k_delta goes out FIRST, before k_bwd's argument marshalling: the GPU is idle until the
    # first packet lands, so every microsecond of host work moved after this launch overlaps
    # k_delta's ~10 us of device time instead of adding to the latency.
    # On the FUSED build `dq` rides along as the accumulation target: k_delta zeroes it, because
    # the fused k_bwd ADDS into it with packed bf16 atomics.  The two-role build's k_delta does
    # not take the argument at all -- one fewer tensor to marshal per launch, which the smallest
    # split-K shape (5 dispatches, ~90 us total) measures.
    if fused:
        _run_compiled(launch_delta, dout, out, delta, dq, t, h, delta_blocks, stream)
    else:
        _run_compiled(launch_delta, dout, out, delta, t, h, delta_blocks, stream)

    if fused and n_split > 1:
        # FUSED split-K: dQ accumulates straight into the (k_delta-zeroed) dq with bf16 atomics
        # as on the unsplit fused kernel; only the dK/dV partials take the ``[dk | dv]`` bf16
        # workspace, n_split slabs each, which k_red sums in fixed slab order.
        n_qk, n_v = dk.numel(), dv.numel()
        ws = torch.empty(n_split * (n_qk + n_v), dtype=dk.dtype, device=device)
        a = n_split * n_qk
        _run_compiled(
            launch_bwd,
            q,
            k,
            v,
            dout,
            softmax_lse,
            delta,
            cu_seqlens,
            dq,
            ws[:a],
            ws[a:],
            t,
            h,
            float(softmax_scale),
            n_dkdv_blocks,
            n_blocks,
            n_seqs,
            interleave,
            stream,
        )
        _run_compiled(kernels[2], ws, dq, dk, dv, n_qk, n_v, red_blocks, stream)
        return dq, dk, dv, delta

    if n_split > 1:
        # ONE bf16 partial workspace ``[dq | dk | dv]``, n_split slabs each, and ONE reduction
        # dispatch over it: this path only runs on grids small enough to be co-resident, where
        # the ~90 us total is mostly host launch cost, so every dispatch and allocation counts.
        # No zeroing: every (token, d, split) slot is written by exactly one workgroup.
        n_qk, n_v = dq.numel(), dv.numel()
        ws = torch.empty(n_split * (2 * n_qk + n_v), dtype=dq.dtype, device=device)
        a, b = n_split * n_qk, 2 * n_split * n_qk
        _run_compiled(
            launch_bwd,
            q,
            k,
            v,
            dout,
            softmax_lse,
            delta,
            cu_seqlens,
            ws[:a],
            ws[a:b],
            ws[b:],
            t,
            h,
            float(softmax_scale),
            n_dkdv_blocks,
            n_blocks,
            n_seqs,
            interleave,
            stream,
        )
        _run_compiled(kernels[2], ws, dq, dk, dv, n_qk, n_v, red_blocks, stream)
        return dq, dk, dv, delta

    _run_compiled(
        launch_bwd,
        q,
        k,
        v,
        dout,
        softmax_lse,
        delta,
        cu_seqlens,
        dq,
        dk,
        dv,
        t,
        h,
        float(softmax_scale),
        n_dkdv_blocks,
        n_blocks,
        n_seqs,
        interleave,
        stream,
    )
    return dq, dk, dv, delta


# The name this entry point shipped under while d_qk = 192 was the only shape it served.  Kept
# because the GEAK task loader and `op_tests/test_mha_flydsl_varlen_bwd.py` resolve it by name.
flash_attn_varlen_bwd_d192_gfx942 = flash_attn_varlen_bwd_gfx942

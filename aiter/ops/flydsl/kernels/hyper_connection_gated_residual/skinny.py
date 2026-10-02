# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Two-kernel skinny (GEMV) Gated Residual for decode M <= DECODE_MAX_M.

Kernel A (:func:`_build_down_fused`) is combine + per-stream RMS + down/inject
GEMV + split-K reduce + SiLU. One wave owns one merged-weight column ``n`` of one
stream ``s`` and covers the whole stream with 16-byte loads, so it re-forms its
slice of ``r2`` and the stream's sum of squares itself: the RMS needs no
prologue kernel and ``xn`` is never materialized. The ``hc`` waves of a column
then meet through a counter; the last to arrive sums their partials and writes
``packed`` (lora after SiLU, inject columns raw). That replaces the prologue and
the reduce launches.

Kernel B (:func:`_build_up_mix_grouped`) is the up GEMV + gated mean. A wave
owns one output channel and splits its lanes into ``hc`` groups, one stream per
group, so the ``hc`` low-rank dots reduce together in ``log2(WAVE / hc)`` steps
instead of one full-wave reduction per stream.

Branch-free by construction: stores that only one wave (or lane) may perform go
through a buffer view bounded to the tensor, with the other lanes' offsets
pushed past the end so the hardware drops them.
"""

import math
import os
from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.act import sigmoid_f32
from aiter.ops.flydsl.kernels.communication_ops_utils import atomic_add_agent
from aiter.ops.flydsl.kernels.tensor_shim import (
    GTensor,
    _run_compiled,
    buf_copy_load,
    buf_copy_store,
    ptr_buf_tensor,
)

WAVE = 64
# sc0|sc1: the partials are written and read across XCDs, which do not share L2.
CPOL_COHERENT = 0x1 | 0x10
DOWN_VEC = 8  # bf16 per 16-byte load
UP_VEC = 4
PART_M = 4  # partial slots per column; bounds m_rows


def _bounded(t, elem, n_elems, unit_elems=1):
    return ptr_buf_tensor(
        t,
        elem,
        unit_elems=unit_elems,
        unit_stride=1,
        num_records_bytes=n_elems * (elem.width // 8),
    )


@lru_cache(maxsize=64)
def _build_down_fused(
    hidden: int,
    n_pad: int,
    hc_count: int,
    lowrank: int,
    m_rows: int,
    eps: float,
    inj_stride: int,
    waves_per_block: int,
    cols_per_wave: int,
    m_groups: int = 1,
):
    stream_dim = hidden // hc_count
    cols_per_block = waves_per_block * cols_per_wave
    per_iter = WAVE * DOWN_VEC
    assert stream_dim % per_iter == 0, f"stream_dim={stream_dim} % {per_iter}"
    assert n_pad % cols_per_block == 0
    # A 16-byte coherent store carries PART_M f32 dots; more rows spill into
    # consecutive chunks (one extra store each). Beyond ~2 chunks the GEMV's
    # redundant r2 re-reads (no operand reuse) overtake the weight load, so the
    # tiled MMA path wins -- this stays a small-M (decode) schedule.
    n_chunks = (m_rows + PART_M - 1) // PART_M
    # The arrival counters are never reset; they only have to wrap cleanly.
    assert hc_count & (hc_count - 1) == 0, "hc_count must be a power of two"
    iters = stream_dim // per_iter
    block_threads = waves_per_block * WAVE
    log2_wave = int(math.log2(WAVE))
    inv_hc = 1.0 / hc_count
    inv_hs = 1.0 / stream_dim
    total_rows = m_groups * m_rows
    r2_elems = total_rows * hidden
    rrms_elems = total_rows * hc_count
    part_group_elems = hc_count * n_pad * PART_M * n_chunks
    part_elems = m_groups * part_group_elems
    packed_elems = total_rows * n_pad

    @flyc.kernel(
        name=f"gr_skinny_down_hc{hc_count}_h{hidden}_np{n_pad}_m{m_rows}"
        f"_g{m_groups}_is{inj_stride}_w{waves_per_block}_c{cols_per_wave}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        residual: fx.Tensor,  # [m_rows, hidden] bf16
        block_output: fx.Tensor,  # [m_rows, stream_dim] bf16
        injection: fx.Tensor,  # [m_rows, hc] bf16, row stride inj_stride
        w_dn: fx.Tensor,  # [n_pad, hidden] bf16, (1+w) folded in
        r2_out: fx.Tensor,  # [m_rows, hidden] bf16
        rrms_out: fx.Tensor,  # [m_rows, hc] f32
        partial: fx.Tensor,  # [hc, n_pad, PART_M] f32
        packed: fx.Tensor,  # [m_rows, n_pad] bf16
        counters: fx.Tensor,  # [n_pad * WAVE] i32, persistent; one slot set per wave
    ):
        tid = fx.thread_idx.x
        wid = tid // WAVE
        lane = tid % WAVE
        grp = fx.block_idx.x * waves_per_block + wid
        n0 = grp * cols_per_wave
        mg = fx.block_idx.y
        m_base = mg * m_rows
        s = fx.block_idx.z

        res_g = GTensor(residual, T.bf16, (1, r2_elems))
        y_g = GTensor(block_output, T.bf16, (1, total_rows * stream_dim))
        inj_g = GTensor(injection, T.bf16, (1, total_rows * inj_stride))
        w_g = GTensor(w_dn, T.bf16, (1, n_pad * hidden))
        r2_t = _bounded(r2_out, fx.BFloat16, r2_elems, DOWN_VEC)
        rrms_t = _bounded(rrms_out, fx.Float32, rrms_elems)
        part_t = _bounded(partial, fx.Float32, part_elems, PART_M)
        packed_t = _bounded(packed, fx.BFloat16, packed_elems)

        # Column 0 of each stream writes that stream's r2 and rrms.
        is_n0 = n0 == fx.Int32(0)

        gate = []
        for m in range_constexpr(m_rows):
            gm = m_base + m
            iv = fx.BFloat16(inj_g.load(gm * inj_stride + s, vec_size=1)).to(
                fx.Float32
            )
            gate.append(fx.Float32(2.0) * sigmoid_f32(iv * fx.Float32(inv_hc)))

        # One r2 re-form feeds cols_per_wave weight rows.
        acc = [
            [fx.Float32(0.0) for _ in range_constexpr(m_rows)]
            for _ in range_constexpr(cols_per_wave)
        ]
        sq = [fx.Float32(0.0) for _ in range_constexpr(m_rows)]
        # Occupancy here is bounded by the grid (~1 wave per SIMD), not by
        # registers, so latency is only hidden if every load is in flight before
        # the first use. The scheduler won't batch them on its own.
        ks = [i * per_iter + lane * DOWN_VEC for i in range_constexpr(iters)]
        w_raw = [
            [
                w_g.load((n0 + cc) * hidden + s * stream_dim + ks[i], vec_size=DOWN_VEC)
                for cc in range_constexpr(cols_per_wave)
            ]
            for i in range_constexpr(iters)
        ]
        r_raw = [
            [
                res_g.load(
                    (m_base + m) * hidden + s * stream_dim + ks[i],
                    vec_size=DOWN_VEC,
                )
                for m in range_constexpr(m_rows)
            ]
            for i in range_constexpr(iters)
        ]
        y_raw = [
            [
                y_g.load((m_base + m) * stream_dim + ks[i], vec_size=DOWN_VEC)
                for m in range_constexpr(m_rows)
            ]
            for i in range_constexpr(iters)
        ]
        for i in range_constexpr(iters):
            k = ks[i]
            wv = [
                fx.Vector(w_raw[i][cc]).to(fx.Float32)
                for cc in range_constexpr(cols_per_wave)
            ]
            for m in range_constexpr(m_rows):
                gm = m_base + m
                off = gm * hidden + s * stream_dim + k
                rv = fx.Vector(r_raw[i][m]).to(fx.Float32)
                yv = fx.Vector(y_raw[i][m]).to(fx.Float32)
                comb = [rv[e] + yv[e] * gate[m] for e in range_constexpr(DOWN_VEC)]
                cb = fx.Vector.from_elements(comb, dtype=fx.Float32).to(fx.BFloat16)
                buf_copy_store(
                    r2_t,
                    is_n0.select(off, fx.Int32(r2_elems)),
                    cb,
                    elem=fx.BFloat16,
                    unit_elems=DOWN_VEC,
                )
                cf = cb.to(fx.Float32)
                for e in range_constexpr(DOWN_VEC):
                    sq[m] = sq[m] + cf[e] * cf[e]
                    for cc in range_constexpr(cols_per_wave):
                        acc[cc][m] = acc[cc][m] + cf[e] * wv[cc][e]

        for sh in range_constexpr(log2_wave):
            off = WAVE // (2 << sh)
            for m in range_constexpr(m_rows):
                sq[m] = sq[m] + fx.gpu.shuffle_xor(sq[m], off, WAVE)
                for cc in range_constexpr(cols_per_wave):
                    acc[cc][m] = acc[cc][m] + fx.gpu.shuffle_xor(acc[cc][m], off, WAVE)

        rstds = []
        for m in range_constexpr(m_rows):
            gm = m_base + m
            rstd = fmath.rsqrt(sq[m] * fx.Float32(inv_hs) + fx.Float32(eps))
            buf_copy_store(
                rrms_t,
                is_n0.select(gm * hc_count + s, fx.Int32(rrms_elems)),
                rstd,
                elem=fx.Float32,
            )
            rstds.append(rstd)
        for cc in range_constexpr(cols_per_wave):
            alldots = [acc[cc][m] * rstds[m] for m in range_constexpr(m_rows)]
            for ch in range_constexpr(n_chunks):
                dots = [
                    alldots[ch * PART_M + mm]
                    for mm in range_constexpr(min(PART_M, m_rows - ch * PART_M))
                ]
                for _ in range_constexpr(PART_M - len(dots)):
                    dots.append(fx.Float32(0.0))
                buf_copy_store(
                    part_t,
                    mg * part_group_elems
                    + ((s * n_pad + n0 + cc) * n_chunks + ch) * PART_M,
                    fx.Vector.from_elements(dots, dtype=fx.Float32),
                    elem=fx.Float32,
                    unit_elems=PART_M,
                    cache_modifier=CPOL_COHERENT,
                )
        # Our partial must be visible before we announce it.
        rocdl.s_waitcnt(0)

        # One counter per lane of a column, so lanes don't contend on an address.
        # Every lane of the last-arriving wave sees hc-1 (mod hc) on its own slot.
        slot = fx.Int64(fx.ptrtoint(fx.get_iter(counters))) + fx.Int64(
            (mg * n_pad + grp) * WAVE + lane
        ) * fx.Int64(4)
        arrival = fx.Int32(atomic_add_agent(slot, fx.Int32(1)))
        rocdl.s_waitcnt(0)
        is_last = (arrival & fx.Int32(hc_count - 1)) == fx.Int32(hc_count - 1)
        writer = is_last & (lane == fx.Int32(0))

        for cc in range_constexpr(cols_per_wave):
            n = n0 + cc
            is_lora = n < fx.Int32(lowrank)
            for ch in range_constexpr(n_chunks):
                tot = fx.Vector(
                    buf_copy_load(
                        part_t,
                        mg * part_group_elems + (n * n_chunks + ch) * PART_M,
                        elem=fx.Float32,
                        unit_elems=PART_M,
                        cache_modifier=CPOL_COHERENT,
                    )
                )
                for ss in range_constexpr(1, hc_count):
                    tot = tot + fx.Vector(
                        buf_copy_load(
                            part_t,
                            mg * part_group_elems
                            + ((ss * n_pad + n) * n_chunks + ch) * PART_M,
                            elem=fx.Float32,
                            unit_elems=PART_M,
                            cache_modifier=CPOL_COHERENT,
                        )
                    )
                for mm in range_constexpr(min(PART_M, m_rows - ch * PART_M)):
                    m = ch * PART_M + mm
                    gm = m_base + m
                    v = tot[mm] * fx.Float32(inv_hc)
                    out = is_lora.select(v * sigmoid_f32(v), tot[mm])
                    buf_copy_store(
                        packed_t,
                        writer.select(gm * n_pad + n, fx.Int32(packed_elems)),
                        out.to(fx.BFloat16),
                        elem=fx.BFloat16,
                    )

    @flyc.jit
    def launch(
        residual: fx.Tensor,
        block_output: fx.Tensor,
        injection: fx.Tensor,
        w_dn: fx.Tensor,
        r2_out: fx.Tensor,
        rrms_out: fx.Tensor,
        partial: fx.Tensor,
        packed: fx.Tensor,
        counters: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(
            residual,
            block_output,
            injection,
            w_dn,
            r2_out,
            rrms_out,
            partial,
            packed,
            counters,
        ).launch(
            grid=(n_pad // cols_per_block, m_groups, hc_count),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


@lru_cache(maxsize=64)
def _build_up_mix_grouped(
    hidden: int,
    hc_count: int,
    lowrank: int,
    w_len: int,
    m_rows: int,
    lora_stride: int,
    waves_per_block: int,
    m_groups: int = 1,
    w_bf16: bool = False,
):
    stream_dim = hidden // hc_count
    group = WAVE // hc_count
    assert WAVE % hc_count == 0 and hc_count & (hc_count - 1) == 0
    assert lowrank % (group * UP_VEC) == 0, f"lowrank={lowrank} % {group * UP_VEC}"
    assert stream_dim % waves_per_block == 0
    k_iters = lowrank // (group * UP_VEC)
    log2_group = int(math.log2(group))
    log2_hc = int(math.log2(hc_count))
    inv_hc = 1.0 / hc_count
    block_threads = waves_per_block * WAVE
    shared_w = w_len != hidden
    total_rows = m_groups * m_rows

    @flyc.kernel(
        name=f"gr_skinny_up_hc{hc_count}_hs{stream_dim}_r{lowrank}_m{m_rows}"
        f"_g{m_groups}_ls{lora_stride}_w{waves_per_block}"
        f"{'_wbf16' if w_bf16 else ''}",
        known_block_size=[block_threads, 1, 1],
    )
    def kernel(
        packed: fx.Tensor,  # [m_rows, lora_stride] bf16, lora is cols [0, lowrank)
        r2: fx.Tensor,  # [m_rows, hidden] bf16
        rrms: fx.Tensor,  # [m_rows, hc] f32
        w: fx.Tensor,  # [w_len] bf16 or f32
        w_up: fx.Tensor,  # [hidden, lowrank] bf16
        block_input: fx.Tensor,  # [m_rows, stream_dim] bf16
    ):
        tid = fx.thread_idx.x
        wid = tid // WAVE
        lane = tid % WAVE
        c = fx.block_idx.x * waves_per_block + wid
        mg = fx.block_idx.y
        m_base = mg * m_rows
        g = lane // group
        j = lane % group
        n = g * stream_dim + c

        lora_g = GTensor(packed, T.bf16, (1, total_rows * lora_stride))
        r2_g = GTensor(r2, T.bf16, (1, total_rows * hidden))
        rrms_g = GTensor(rrms, T.f32, (1, total_rows * hc_count))
        w_g = GTensor(w, T.bf16 if w_bf16 else T.f32, (1, w_len))
        wup_g = GTensor(w_up, T.bf16, (1, hidden * lowrank))
        out_g = GTensor(block_input, T.bf16, (1, total_rows * stream_dim))

        acc = [fx.Float32(0.0) for _ in range_constexpr(m_rows)]
        # All loads before the first use; see _build_down_fused.
        rs = [i * (group * UP_VEC) + j * UP_VEC for i in range_constexpr(k_iters)]
        wu_raw = [
            wup_g.load(n * lowrank + rs[i], vec_size=UP_VEC)
            for i in range_constexpr(k_iters)
        ]
        lo_raw = [
            [
                lora_g.load((m_base + m) * lora_stride + rs[i], vec_size=UP_VEC)
                for m in range_constexpr(m_rows)
            ]
            for i in range_constexpr(k_iters)
        ]
        wv_raw = w_g.load(c if shared_w else n, vec_size=1)
        wv = (
            fx.BFloat16(wv_raw).to(fx.Float32)
            if w_bf16
            else fx.Float32(wv_raw)
        )
        rr_raw = [
            rrms_g.load((m_base + m) * hc_count + g, vec_size=1)
            for m in range_constexpr(m_rows)
        ]
        r2_raw = [
            r2_g.load((m_base + m) * hidden + n, vec_size=1)
            for m in range_constexpr(m_rows)
        ]
        for i in range_constexpr(k_iters):
            wu = fx.Vector(wu_raw[i]).to(fx.Float32)
            for m in range_constexpr(m_rows):
                lo = fx.Vector(lo_raw[i][m]).to(fx.Float32)
                for e in range_constexpr(UP_VEC):
                    acc[m] = acc[m] + lo[e] * wu[e]
        for sh in range_constexpr(log2_group):
            for m in range_constexpr(m_rows):
                acc[m] = acc[m] + fx.gpu.shuffle_xor(acc[m], group // (2 << sh), WAVE)

        onepw = fx.Float32(1.0) + wv
        mix = []
        for m in range_constexpr(m_rows):
            rr = rr_raw[m]
            r2v = fx.BFloat16(r2_raw[m]).to(fx.Float32)
            xn = fx.BFloat16(r2v * rr * onepw).to(fx.Float32)
            mix.append(sigmoid_f32(acc[m]) * xn)
        for sh in range_constexpr(log2_hc):
            for m in range_constexpr(m_rows):
                mix[m] = mix[m] + fx.gpu.shuffle_xor(mix[m], group << sh, WAVE)
        for m in range_constexpr(m_rows):
            gm = m_base + m
            out_g.store(
                gm * stream_dim + c,
                (mix[m] * fx.Float32(inv_hc)).to(fx.BFloat16),
                vec_size=1,
            )

    @flyc.jit
    def launch(
        packed: fx.Tensor,
        r2: fx.Tensor,
        rrms: fx.Tensor,
        w: fx.Tensor,
        w_up: fx.Tensor,
        block_input: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kernel(packed, r2, rrms, w, w_up, block_input).launch(
            grid=(stream_dim // waves_per_block, m_groups, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch


_COLS_PER_WAVE = int(os.environ.get("AITER_GR_SKINNY_COLS_PER_WAVE", "2"))
_COUNTERS: dict[tuple[int, int, int], torch.Tensor] = {}


def _counters(device: torch.device, n_pad: int, m_groups: int = 1) -> torch.Tensor:
    key = (device.index, n_pad, m_groups)
    c = _COUNTERS.get(key)
    if c is None:
        c = torch.zeros(m_groups * n_pad * WAVE, dtype=torch.int32, device=device)
        _COUNTERS[key] = c
    return c


def skinny_two_kernel(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    inj_stride: int,
    w: torch.Tensor,
    w_up: torch.Tensor,
    w_down_merged: torch.Tensor,
    lowrank: int,
    hc_count: int,
    eps: float,
    stream: torch.cuda.Stream,
    waves_per_block: int = 4,
    cols_per_wave: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns ``(r2, block_input, packed)``; see the module docstring.

    Two calls must not run concurrently on different streams: they share the
    per-column arrival counters.
    """
    tokens, hidden = residual.shape
    stream_dim = hidden // hc_count
    n_pad = w_down_merged.shape[0]
    dev = residual.device
    r2 = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=dev)
    rrms = torch.empty(tokens, hc_count, dtype=torch.float32, device=dev)
    n_chunks = (tokens + PART_M - 1) // PART_M
    partial = torch.empty(
        hc_count, n_pad, n_chunks * PART_M, dtype=torch.float32, device=dev
    )
    packed = torch.empty(tokens, n_pad, dtype=torch.bfloat16, device=dev)
    x = torch.empty(tokens, stream_dim, dtype=torch.bfloat16, device=dev)

    if cols_per_wave is None:
        cols_per_wave = _COLS_PER_WAVE
    down = _build_down_fused(
        hidden,
        n_pad,
        hc_count,
        lowrank,
        tokens,
        float(eps),
        inj_stride,
        waves_per_block,
        cols_per_wave,
    )
    up = _build_up_mix_grouped(
        hidden,
        hc_count,
        lowrank,
        w.numel(),
        tokens,
        n_pad,
        waves_per_block,
        w_bf16=w.dtype == torch.bfloat16,
    )
    fxs = fx.Stream(stream)
    _run_compiled(
        down,
        residual,
        block_output,
        injection,
        w_down_merged,
        r2,
        rrms,
        partial,
        packed,
        _counters(dev, n_pad),
        fxs,
    )
    _run_compiled(up, packed, r2, rrms, w, w_up, x, fxs)
    return r2, x, packed


def skinny_chunked_two_kernel(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection: torch.Tensor,
    inj_stride: int,
    w: torch.Tensor,
    w_up: torch.Tensor,
    w_down_merged: torch.Tensor,
    lowrank: int,
    hc_count: int,
    eps: float,
    stream: torch.cuda.Stream,
    chunk_m: int = 4,
    waves_per_block: int = 4,
    cols_per_wave: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prototype true-M schedule for low prefill.

    The original skinny kernel holds every token accumulator in one wave and
    spills badly above M=8. This variant maps independent ``chunk_m`` token
    groups onto grid.y, keeping the same register footprint while increasing
    parallelism. It intentionally trades repeated weight reads between groups
    against avoiding the MMA path's 64-row padding.
    """
    tokens, hidden = residual.shape
    assert tokens % chunk_m == 0
    m_groups = tokens // chunk_m
    stream_dim = hidden // hc_count
    n_pad = w_down_merged.shape[0]
    dev = residual.device
    r2 = torch.empty(tokens, hidden, dtype=torch.bfloat16, device=dev)
    rrms = torch.empty(tokens, hc_count, dtype=torch.float32, device=dev)
    n_chunks = (chunk_m + PART_M - 1) // PART_M
    partial = torch.empty(
        m_groups,
        hc_count,
        n_pad,
        n_chunks * PART_M,
        dtype=torch.float32,
        device=dev,
    )
    packed = torch.empty(tokens, n_pad, dtype=torch.bfloat16, device=dev)
    x = torch.empty(tokens, stream_dim, dtype=torch.bfloat16, device=dev)

    if cols_per_wave is None:
        cols_per_wave = _COLS_PER_WAVE
    down = _build_down_fused(
        hidden,
        n_pad,
        hc_count,
        lowrank,
        chunk_m,
        float(eps),
        inj_stride,
        waves_per_block,
        cols_per_wave,
        m_groups,
    )
    up = _build_up_mix_grouped(
        hidden,
        hc_count,
        lowrank,
        w.numel(),
        chunk_m,
        n_pad,
        waves_per_block,
        m_groups,
        w.dtype == torch.bfloat16,
    )
    fxs = fx.Stream(stream)
    _run_compiled(
        down,
        residual,
        block_output,
        injection,
        w_down_merged,
        r2,
        rrms,
        partial,
        packed,
        _counters(dev, n_pad, m_groups),
        fxs,
    )
    _run_compiled(up, packed, r2, rrms, w, w_up, x, fxs)
    return r2, x, packed

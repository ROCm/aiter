# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused per-unit GEMM1 + activation + GEMM2 for the EP MegaMoE kernel.

Ported from the TP kernel's MFMA path (``mega_moe_tp.common.mfma`` + the per-lane
``a_frag`` fragment reads), adapted to the EP flat-L / indexed-payload world:

* ``a_loader`` -- no-op.  The per-unit row gather already lands in ``L_RIX`` (the
  scheduler's ``gather_recv_chunk`` stores ``recv_slot = source_key*topk + route``);
  the compute waves read the A activation **directly from the global payload**
  (payload row = ``L_RIX[r] // topk``) one MFMA fragment at a time -- no LDS A tile,
  no DMA ring.  This is the correctness-first bring-up form; the TP software-pipelined
  A-ring (LDS staging + double buffering) is the tune-milestone upgrade.
* ``gemm1`` -- A x W1 (gate|up) over the planned inter piece ``[i0, i0+icnt)``, fused
  SiLU/SwiGLU/situv2 activation, E8M0-quantized into the **global** intermediate
  (``a["inter"]`` / ``a["inters"]``; a full RG x inter FP8 tile will not fit LDS).
* ``gemm2`` -- intermediate x W2 -> model_dim bf16 rows into ``gout`` for
  ``combine_send`` to scatter back to the source ranks.

The MFMA tiling mirrors TP exactly (16x16x128 scaled f8f6f4; the 2x16-col gate/up
sub-tiles per wave, MT row tiles, 4 K-sub-groups selected by ``lane//16``).  The
weight / scale **addressing** is a direct global read (no shuffle-ring); getting the
exact ``shuffle_weight_w4`` byte layout + the E8M0 group packing bit-identical to the
reference is the numeric bring-up pass (``TODO(bring-up)``), after this compiles.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import T

from ..mxfp4_gemm_common import _activation_mul_batch, _e8m0_from_amax
from .common import (
    V4I,
    amax,
    bld,
    bst,
    cat8,
    fp4_pack,
    fp8x4_pack,
    i32,
    lds_ld_i32,
    mfma,
    rsrc,
)
from .mega_moe_ep_config import NT, KernelCtx

__all__ = ["build_gemm"]


def build_gemm(kc: KernelCtx) -> dict:
    """The per-unit GEMM1+act+GEMM2 device functions of one instance (added to kc)."""
    H, I, TOPK, a8 = kc.get("H I TOPK a8")
    act, swiglu_limit = kc.get("act swiglu_limit")
    situ_beta, situ_linear_beta = kc.get("situ_beta situ_linear_beta")
    RG, L_RIX, const_expr = kc.get("RG L_RIX const_expr")

    MT = RG // 16  # MFMA row tiles (16 rows each)
    NWAVES = NT // 64  # compute waves (= NW)
    a_pack = 1 if a8 else 2  # bytes per element (FP8=1, packed FP4=2 per byte)
    a_row_bytes = H // a_pack  # payload (A) bytes per row
    scale_cols = H // 32  # E8M0 A-scale groups per row
    inter_row_bytes = I // a_pack  # quantized intermediate bytes per row
    inter_scols = I // 32  # E8M0 inter-scale groups per row
    K1 = H // 128  # GEMM1 K steps (MFMA K=128)
    K2 = I // 128  # GEMM2 K steps
    max_norm = 448.0 if a8 else 6.0

    def _act_batch(gs, us):
        return _activation_mul_batch(
            gs,
            us,
            act=act,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            swiglu_limit=swiglu_limit,
        )

    def _afrag(rbuf, row_base, k, q4):
        """One scaled-MFMA activation fragment (this lane's K=128 slice) read straight
        from a global row at byte ``row_base``.  FP8 -> two 16B chunks (cat8); FP4 ->
        one 16B chunk.  No LDS bank swizzle (contiguous global rows)."""
        if const_expr(a8):
            lo = i32(k * 8) + q4
            hi = i32(k * 8 + 4) + q4
            return cat8(
                bld(rbuf, row_base + lo * i32(16), 0, V4I, 0),
                bld(rbuf, row_base + hi * i32(16), 0, V4I, 0),
            )
        col = i32(k * 4) + q4
        return bld(rbuf, row_base + col * i32(16), 0, V4I, 0)

    def _ascale(rbuf, scale_base, k, q4):
        """E8M0 activation scale byte for this lane's K group."""
        off = scale_base + i32(k * 4) + q4
        return fx.Int32(bld(rbuf, off, 0, T.i8, 0)) & i32(0xFF)

    def _wfrag(rw, col0, k, lane, kdim):
        """One FP4 weight fragment (this lane's K=128 slice of 16 output cols).
        TODO(bring-up): match the exact shuffle_weight_w4 tile layout; this is a direct
        [out_col, kdim] read (col = col0 + lane%16, K group = lane//16)."""
        col = col0 + lane % i32(16)
        return bld(
            rw,
            col * i32(kdim // 2) + i32(k * 64) + (lane // i32(16)) * i32(16),
            0,
            V4I,
            0,
        )

    def _wscale(rws, col0, k, lane, kdim):
        col = col0 + lane % i32(16)
        off = col * i32(kdim // 32) + i32(k * 4) + lane // i32(16)
        return fx.Int32(bld(rws, off, 0, T.i8, 0)) & i32(0xFF)

    @flyc.jit
    def a_loader(L, tid, a):
        # No-op: the per-unit row gather already populated L_RIX; the compute waves read
        # A directly from global in gemm1/gemm2 (bring-up form, no LDS A tile).
        del L, tid, a

    @flyc.jit
    def gemm1(L, tid, a, expert, i0, icnt, rows):
        """GEMM1 (A x W1 gate|up) + activation -> E8M0-quantized global intermediate.

        Correctness-first: N-blocks are a runtime loop; the K loop (H/128) is unrolled;
        A fragments are read straight from the global payload per (row-tile, K)."""
        lane = tid % i32(64)
        w = tid // i32(64)
        q4 = lane // i32(16)
        rpay = rsrc(a["mine"] + a["off_payload"])  # A activation (global)
        rsc = rsrc(a["mine"] + a["off_pscale"])  # A E8M0 scales (global)
        rw = rsrc(fx.Int64(a["w1"]) + fx.Int64(expert) * fx.Int64(2 * I * (H // 2)))
        rws = rsrc(fx.Int64(a["w1s"]) + fx.Int64(expert) * fx.Int64(2 * I * (H // 32)))
        r_inter = rsrc(a["inter"])
        r_inters = rsrc(a["inters"])
        bid = i32(gpu.block_id("x"))
        ibase = bid * i32(RG * inter_row_bytes)  # this CTA's intermediate scratch
        isbase = bid * i32(RG * inter_scols)
        zf = fx.Vector.filled(4, 0.0, fx.Float32)

        for nb_ in range(i32(0), icnt // i32(128), i32(1)):
            nb = i32(nb_)
            c0 = i0 + nb * i32(128) + w * i32(32)  # this wave's 32 inter cols
            accg = [[zf, zf] for _ in range(MT)]
            accu = [[zf, zf] for _ in range(MT)]
            for k in range_constexpr(K1):
                bg = [_wfrag(rw, c0 + i32(t * 16), k, lane, H) for t in range(2)]
                bu = [
                    _wfrag(rw, i32(I) + c0 + i32(t * 16), k, lane, H) for t in range(2)
                ]
                sbg = [_wscale(rws, c0 + i32(t * 16), k, lane, H) for t in range(2)]
                sbu = [
                    _wscale(rws, i32(I) + c0 + i32(t * 16), k, lane, H)
                    for t in range(2)
                ]
                for m in range_constexpr(MT):
                    row = i32(m * 16) + lane % i32(16)
                    rr = (row < rows).select(row, i32(0))
                    src = lds_ld_i32(L, L_RIX + rr * i32(4)) // i32(TOPK)
                    af = _afrag(rpay, src * i32(a_row_bytes), k, q4)
                    sa = _ascale(rsc, src * i32(scale_cols), k, q4)
                    for t in range_constexpr(2):
                        accg[m][t] = mfma(bg[t], af, accg[m][t], sbg[t], sa, a8)
                        accu[m][t] = mfma(bu[t], af, accu[m][t], sbu[t], sa, a8)
            # epilogue: act(gate)*up -> E8M0 quant -> global intermediate
            for m in range_constexpr(MT):
                row = i32(m * 16) + lane % i32(16)
                gs = [
                    fx.Float32(fx.Vector(accg[m][t])[v])
                    for t in range(2)
                    for v in range(4)
                ]
                us = [
                    fx.Float32(fx.Vector(accu[m][t])[v])
                    for t in range(2)
                    for v in range(4)
                ]
                xs = [
                    fx.Float32(x).to(fx.BFloat16).to(fx.Float32)
                    for x in _act_batch(gs, us)
                ]
                am = amax(xs)
                am = am.maximumf(am.shuffle_xor(i32(16), i32(64)))
                am = am.maximumf(am.shuffle_xor(i32(32), i32(64)))
                e8, qs = _e8m0_from_amax(am, max_norm=max_norm)
                rbase = ibase + row * i32(inter_row_bytes)
                for t in range_constexpr(2):
                    col = c0 + i32(t * 16) + q4 * i32(4)
                    if const_expr(a8):
                        bst(
                            fp8x4_pack(xs[t * 4 : t * 4 + 4], qs),
                            r_inter,
                            rbase + col,
                            0,
                            0,
                        )
                    else:
                        pk = fp4_pack(xs[t * 4 : t * 4 + 4], qs)
                        bst(
                            fx.Int16(pk & i32(0xFFFF)),
                            r_inter,
                            rbase + col // i32(2),
                            0,
                            0,
                        )
                if q4 == i32(0):
                    bst(
                        fx.Int8(e8),
                        r_inters,
                        isbase + row * i32(inter_scols) + c0 // i32(32),
                        0,
                        0,
                    )

    @flyc.jit
    def gemm2(L, tid, a, expert, rows, src_out):
        """GEMM2 (intermediate x W2) -> model_dim bf16 rows into ``src_out`` (gout).

        A2 = this unit's quantized intermediate (global, written by gemm1); no
        activation / requant on the output (bf16 for the weighted combine)."""
        del L
        lane = tid % i32(64)
        w = tid // i32(64)
        q4 = lane // i32(16)
        rw = rsrc(fx.Int64(a["w2"]) + fx.Int64(expert) * fx.Int64(H * (I // 2)))
        rws = rsrc(fx.Int64(a["w2s"]) + fx.Int64(expert) * fx.Int64(H * (I // 32)))
        r_inter = rsrc(a["inter"])
        r_inters = rsrc(a["inters"])
        bid = i32(gpu.block_id("x"))
        ibase = bid * i32(RG * inter_row_bytes)
        isbase = bid * i32(RG * inter_scols)
        obase = bid * i32(RG * H)  # bf16 element base of this CTA's gout scratch
        zf = fx.Vector.filled(4, 0.0, fx.Float32)

        for nb_ in range(i32(0), i32(H // 128), i32(1)):
            nb = i32(nb_)
            c0 = nb * i32(128) + w * i32(32)  # this wave's 32 model-dim cols
            acc = [[zf, zf] for _ in range(MT)]
            for k in range_constexpr(K2):
                b = [_wfrag(rw, c0 + i32(t * 16), k, lane, I) for t in range(2)]
                sb = [_wscale(rws, c0 + i32(t * 16), k, lane, I) for t in range(2)]
                for m in range_constexpr(MT):
                    row = i32(m * 16) + lane % i32(16)
                    rbase = ibase + row * i32(inter_row_bytes)
                    sbase = isbase + row * i32(inter_scols)
                    af = _afrag(r_inter, rbase, k, q4)
                    sa = _ascale(r_inters, sbase, k, q4)
                    for t in range_constexpr(2):
                        acc[m][t] = mfma(b[t], af, acc[m][t], sb[t], sa, a8)
            # store bf16 (valid rows only) for combine_send
            for m in range_constexpr(MT):
                row = i32(m * 16) + lane % i32(16)
                if row < rows:
                    for t in range_constexpr(2):
                        col = c0 + i32(t * 16) + q4 * i32(4)
                        ov = fx.Vector.from_elements(
                            [
                                fx.Float32(fx.Vector(acc[m][t])[v]).to(fx.BFloat16)
                                for v in range(4)
                            ],
                            fx.BFloat16,
                        )
                        bst(ov, src_out, (obase + row * i32(H) + col) * i32(2), 0, 0)
        del rows

    return locals()

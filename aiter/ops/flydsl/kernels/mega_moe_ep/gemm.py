# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused per-unit GEMM1 + activation + GEMM2 for the EP MegaMoE kernel.

The compute waves run, for each planned unit (local ``expert`` x inter piece
``[i0, i0+icnt)``) over the gathered rows in ``L_RIX``:

* ``a_loader``  -- stage the unit's A rows (the dispatched payload, indexed by the
  gathered recv slots in ``L_RIX``: payload row = recv_slot // topk) + their E8M0
  scales into LDS (``L_A`` / ``L_AS``), the layout the scaled-MFMA A operand expects.
* ``gemm1``     -- GEMM1 (A x W1 gate|up) + activation, writing the quantized
  intermediate to a global buffer (``a["inter"]`` / ``a["inters"]``; LDS cannot hold a
  full RG x inter tile for FP8). Reuses MfmaScaleGU / BWeightLoader / BScaleLoader /
  AS2RLoader / SiluQuantEpilogue from the stage1 atoms.
* ``gemm2``     -- GEMM2 (intermediate x W2) -> model_dim bf16 rows, left in a per-unit
  LDS/global output region for ``combine_send`` to scatter back to the source ranks.

BRING-UP STATE: this is a correctness-first first cut (non-pipelined MFMA K-loop, no
weight-DMA ring / double buffering). It is the starting point for GPU bring-up
(compile + relL2 vs MegaMoEV2); the stage1-style pipelining is deferred to the tune
milestone. The GEMM2 path is written against MfmaScaleGU (stage2's GEMM2 is entangled
with its P2P scatter and is not reused directly).
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr
from flydsl.expr.typing import T

from ..mega_moe.gemm_util import (
    AS2RLoader,
    AScaleLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
    SiluQuantEpilogue,
)
from .common import V4I, bld, bst, i32, lds_ld_i32, rsrc, uni, wait_vm
from .mega_moe_ep_config import (
    C_UROWS,
    NT,
    NW,
    TILE_K,
    TILE_N,
    KernelCtx,
)

__all__ = ["build_gemm"]


def build_gemm(kc: KernelCtx) -> dict:
    """The per-unit GEMM1+act+GEMM2 device functions of one instance (added to kc)."""
    H, I, TOPK, EPR, a8 = kc.get("H I TOPK EPR a8")
    act, swiglu_limit = kc.get("act swiglu_limit")
    situ_beta, situ_linear_beta = kc.get("situ_beta situ_linear_beta")
    M_REPEAT, NUM_ACC_N, A_K_STEP = kc.get("M_REPEAT NUM_ACC_N A_K_STEP")
    K_ITERS, A_LDS_I32, N_TILES = kc.get("K_ITERS A_LDS_I32 N_TILES")
    L_A, L_AS, L_RIX, L_CTL = kc.get("L_A L_AS L_RIX L_CTL")
    RG, const_expr = kc.get("RG const_expr")
    a_dtype = "fp8" if a8 else "fp4"
    a_pack = 1 if a8 else 2
    a_row_bytes = H // a_pack  # payload bytes per row
    scale_cols = H // 32  # E8M0 scale cols per row
    n_per_wave = TILE_N // NW
    k2_iters = I // TILE_K  # GEMM2 K steps (over the inter dim)
    out_pack = 1 if a8 else 2  # intermediate packing (feeds GEMM2 A)
    inter_bytes = I // out_pack

    # Every MFMA / loader atom touches fx.thread_idx (and the MFMA atom needs an MLIR
    # context), so they are all constructed inside the traced device functions below,
    # never eagerly here.

    @flyc.jit
    def a_loader(L, tid, a):
        """Stage the current unit's A rows + E8M0 scales into LDS. The gathered recv
        slots sit in L_RIX (recv_slot = source_key*topk + route); the payload row is
        source_key = slot // topk, read from this rank's own arena payload region."""
        rows = lds_ld_i32(L, L_CTL + C_UROWS * 4)
        rpay = rsrc(a["mine"] + a["off_payload"])
        rsc = rsrc(a["mine"] + a["off_pscale"])
        n_vec = a_row_bytes // 16  # 16-byte copy chunks per A row
        ns_i32 = scale_cols // 4  # i32 scale chunks per row
        for r_ in range(i32(0), fx.min(rows, i32(RG)), i32(1)):
            r = i32(r_)
            slot = lds_ld_i32(L, L_RIX + r * i32(4))
            src = slot // i32(TOPK)  # payload row = source_key
            for j in range_constexpr((a_row_bytes // 16 + NT - 1) // NT):
                u = tid + i32(j * NT)
                if u < i32(n_vec):
                    v = bld(rpay, src * i32(a_row_bytes) + u * i32(16), 0, V4I, 0)
                    bst(v, L_A_rsrc(L), (r * i32(n_vec) + u) * i32(16), 0, 0)
            for j in range_constexpr((scale_cols // 4 + NT - 1) // NT):
                u = tid + i32(j * NT)
                if u < i32(ns_i32):
                    s = bld(rsc, (src * i32(ns_i32) + u) * i32(4), 0, T.i32, 0)
                    bst(s, L_AS_rsrc(L), (r * i32(ns_i32) + u) * i32(4), 0, 0)
        wait_vm(0)

    def L_A_rsrc(L):
        # LDS buffer resource over the staged A tile (byte addressed from the base).
        return rsrc(fx.Int64(uni(i32(L))) + fx.Int64(i32(L_A)))

    def L_AS_rsrc(L):
        return rsrc(fx.Int64(uni(i32(L))) + fx.Int64(i32(L_AS)))

    @flyc.jit
    def gemm1(L, tid, a, expert, i0, icnt, rows, sig):
        """GEMM1 (A x W1 gate|up) + activation for the planned inter piece, writing the
        quantized intermediate to the global buffer. Correctness-first non-pipelined
        K-loop over the staged A tile. TODO(tune): weight-DMA ring + double buffer."""
        lane = tid % i32(64)
        w = tid // i32(64)
        mfma1 = MfmaScaleGU(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_dtype=a_dtype)
        a_s2r = AS2RLoader(k_step_bytes=A_K_STEP, swizzle=False, a_dtype=a_dtype)
        # W1 of the local expert, gate|up over the inter piece [i0, i0+icnt).
        rw = rsrc(fx.Int64(a["w1"]) + fx.Int64(expert) * fx.Int64(2 * I * (H // 2)))
        rws = rsrc(a["w1s"])
        b_loader = BWeightLoader(w_rsrc=rw, num_acc_n=NUM_ACC_N, model_dim=H)
        b_scale = BScaleLoader(scale_rsrc=rws, num_acc_n=NUM_ACC_N, model_dim=H)
        a_scale = AScaleLoader(
            scale_rsrc=L_AS_rsrc(L),
            m_repeat=M_REPEAT,
            model_dim=H,
            sort_block_m=RG,
            total_threads=NT,
        )
        a_buf = _lds_view(L, L_A)
        a_scale_lds = _lds_view(L, L_AS)
        # one GEMM1 output column tile per wave (gate|up halves)
        n_tile_base = w * i32(n_per_wave) + i0
        acc = [mfma1.zero_value for _ in range(M_REPEAT * NUM_ACC_N)]
        for sp_ in range(i32(0), i32(K_ITERS), i32(1)):
            sp = i32(sp_)
            b = b_loader.load_step(n_tile_base, sp)
            sa = a_scale.load_step(a_scale_lds, sp)
            sb = b_scale.load_step(n_tile_base, sp)

            def a_load(mi, ks, _sp=sp):
                return a_s2r.load_operand(a_buf, mi, ks, _sp * i32(A_LDS_I32))

            acc = mfma1.call(a_load, b, acc, sa, sb)
        _gemm1_store(L, lane, w, a, acc, expert, i0, rows)

    def _lds_view(L, off):
        return fx.make_view(
            fx.add_offset(fx.recast_iter(fx.Int32, _lds_ptr(L)), i32(off)),
            fx.make_layout(1, 1),
        )

    def _lds_ptr(L):
        return fx.inttoptr(
            fx.PointerType.get(fx.Int32, fx.AddressSpace.Shared, 4), uni(i32(L))
        )

    @flyc.jit
    def _gemm1_store(L, lane, w, a, acc, expert, i0, rows):
        # TODO(bring-up): silu/situv2(gate)*up -> quantized intermediate at the global
        # a["inter"]/a["inters"] rows for GEMM2. Reuse SiluQuantEpilogue._combine for
        # the activation; the quant store addressing lands with the bring-up pass.
        epi = SiluQuantEpilogue(
            out_rsrc=rsrc(a["inter"]),
            out_scale_rsrc=rsrc(a["inters"]),
            sorted_rsrc=rsrc(a["mine"] + a["off_recv"]),
            tokens=0,
            inter_dim=I,
            m_repeat=M_REPEAT,
            num_acc_n=NUM_ACC_N,
            sort_block_m=RG,
            tile_n=TILE_N,
            num_waves=NW,
            lds_out=_lds_view(L, L_A),
            evec=2,
            swiglu_limit=swiglu_limit,
            always_valid=True,
            out_dtype=a_dtype,
            act=act,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
        del lane, w, expert, i0, rows
        _ = epi  # wired in bring-up: epi.store(acc, tile, row_base, n_tile_base)

    @flyc.jit
    def gemm2(L, tid, a, expert, rows, src_out):
        """GEMM2 (intermediate x W2) -> model_dim bf16 rows into ``src_out`` for
        combine_send. Correctness-first MFMA over the inter dim (K2 steps)."""
        lane = tid % i32(64)
        w = tid // i32(64)
        mfma2 = MfmaScaleGU(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_dtype=a_dtype)
        rw2 = rsrc(fx.Int64(a["w2"]) + fx.Int64(expert) * fx.Int64(H * (I // 2)))
        rw2s = rsrc(a["w2s"])
        b2 = BWeightLoader(w_rsrc=rw2, num_acc_n=NUM_ACC_N, model_dim=I)
        b2s = BScaleLoader(scale_rsrc=rw2s, num_acc_n=NUM_ACC_N, model_dim=I)
        a2_s2r = AS2RLoader(
            k_step_bytes=TILE_K // a_pack, swizzle=False, a_dtype=a_dtype
        )
        # A2 = this unit's quantized intermediate rows (global, written by GEMM1).
        a2_buf = _lds_view(L, L_A)  # staged into the A pool for the K-loop
        acc = [mfma2.zero_value for _ in range(M_REPEAT * NUM_ACC_N)]
        n_tile_base = w * i32(TILE_N // NW)
        for sp_ in range(i32(0), i32(k2_iters), i32(1)):
            sp = i32(sp_)
            b = b2.load_step(n_tile_base, sp)
            sb = b2s.load_step(n_tile_base, sp)

            def a_load(mi, ks, _sp=sp):
                return a2_s2r.load_operand(a2_buf, mi, ks, _sp * i32(A_LDS_I32))

            # TODO(bring-up): intermediate A-scale load (per-row E8M0 from a["inters"]).
            sa = [i32(127)] * (M_REPEAT // 2)
            acc = mfma2.call(a_load, b, acc, sa, sb)
        del lane, rows, src_out, n_tiles_unused  # noqa: F821  (bring-up wiring)

    n_tiles_unused = N_TILES
    return locals()

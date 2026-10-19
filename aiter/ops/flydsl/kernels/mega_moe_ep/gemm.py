# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused per-unit GEMM1 + activation + GEMM2 for the EP MegaMoE kernel.

GEMM1 is the **pipelined** MegaMoEV2 path (``mega_moe.gemm_util`` atoms): an A-ring
LDS double-buffer (``ATileLoader`` / ``AS2RLoader``, source-key gather via
``indexed_input`` + a per-CTA ``row_map`` written to global scratch), weight-prefetch
software pipeline (``MfmaScaleGU.call_pipe`` + ``BWeightLoader`` / ``BScaleLoader``),
E8M0 A-scale staged to LDS (``AScaleLoader``), and a cshuffle epilogue that un-shuffles
the gate|up accumulators back to **natural column order** and writes the row-major
E8M0-quantized intermediate (``inter`` / ``inters``) exactly as the previous naive GEMM1
did -- so the (unchanged, correct) GEMM2 reads the same layout.

The V2 atoms are reused verbatim, but their orchestration (``do_tile`` / the
``SiluQuantEpilogue`` store) uses whole-block ``s_barrier`` -- which would deadlock EP's
warp-specialized 512-thread block (only ``NT`` compute threads run the GEMM). So the
K-loop pipeline and the epilogue are re-expressed here with the compute-wave ``cbar``
(the same 4-wave LDS generation barrier the planner uses).

GEMM2 stays the correctness-first naive MFMA path (A2 = the quantized intermediate read
straight from global, W2 natural-layout) -> model_dim bf16 rows into ``gout`` for
``combine_send`` to scatter back to the source ranks.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..mega_moe.gemm_util import (
    AS2RLoader,
    AScaleLoader,
    ATileLoader,
    BScaleLoader,
    BWeightLoader,
    MfmaScaleGU,
)
from ..mxfp4_gemm_common import _activation_mul_batch
from ..tensor_shim import ptr_buf_tensor
from .common import (
    ACQ_REL,
    AUX_SYS,
    REL,
    V4I,
    bld,
    bst,
    cat8,
    i32,
    lds_atomic_add,
    lds_ld_acq,
    lds_ld_i32,
    lds_st,
    lds_st_rel,
    mfma,
    rsrc,
    spin0,
    spin_lds_ge,
    uni,
    wait_lgkm0,
    wait_vm,
)
from .mega_moe_ep_config import (
    B_LOADER,
    C_BARCNT,
    C_BARGEN,
    MAX_NPES,
    NLOAD,
    NT,
    NW,
    RC_AFREE,
    RC_ASEQ,
    RC_G1GO,
    TILE_K,
    TILE_N,
    KernelCtx,
)

__all__ = ["build_gemm"]


class _LdsView:
    """Minimal ``.ptr`` holder the gemm_util atoms expect for an LDS buffer."""

    def __init__(self, ptr):
        self.ptr = ptr


def _lds_byte_ptr(L, off):
    """Byte-typed LDS pointer at ``L + off`` (gemm_util atoms address A/A-scale in
    bytes; ``AS2RLoader._load_16b`` offsets the raw ``.ptr`` by a byte count, so the
    element type must be ``i8``).  Uses the MLIR type ``T.i8`` -- NOT the fx class."""
    pt = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Shared, alignment=16)
    return fx.inttoptr(pt, uni(i32(L) + i32(off)))


def _lds_f32_ptr(L, off):
    """f32-typed LDS pointer at ``L + off`` (the cshuffle epilogue addresses the C tile
    in f32 elements)."""
    pt = fx.PointerType.get(T.f32, address_space=fx.AddressSpace.Shared, alignment=16)
    return fx.inttoptr(pt, uni(i32(L) + i32(off)))


def _global_byte_tensor(addr_i64, nbytes):
    """An ``fx.Tensor`` byte-view over a raw global address (for ``ATileLoader``)."""
    pt = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=16)
    it = fx.inttoptr(pt, fx.Int64(addr_i64))
    return fx.Tensor(fx.make_view(it, fx.make_layout(nbytes, 1)))


def build_gemm(kc: KernelCtx) -> dict:
    """The per-unit GEMM1+act+GEMM2 device functions of one instance (added to kc)."""
    H, I, TOPK, a8 = kc.get("H I TOPK a8")
    act, swiglu_limit = kc.get("act swiglu_limit")
    situ_beta, situ_linear_beta = kc.get("situ_beta situ_linear_beta")
    RG, L_RIX, const_expr = kc.get("RG L_RIX const_expr")
    L_A, L_AS, L_CTL, TMAX = kc.get("L_A L_AS L_CTL TMAX")
    N_TILES, NUM_ACC_N, M_REPEAT = kc.get("N_TILES NUM_ACC_N M_REPEAT")
    A_K_STEP, K_ITERS, A_LDS_I32, NAB = kc.get("A_K_STEP K_ITERS A_LDS_I32 NAB")
    L_C, L_RCTL = kc.get("L_C L_RCTL")
    combine_arrive = kc.get("combine_arrive")[0]  # fused-combine epilogue signals source

    def _ring_off(q0, j):
        """i32-element LDS offset of A-ring bank ``(q0 + j) % NAB`` (runtime bank index),
        for ``ATileLoader.store`` / ``AS2RLoader.load_operand`` ``base_i32``."""
        return uni(((q0 + i32(j)) % i32(NAB)) * i32(A_LDS_I32))

    # ---- A-loader ring (producer=loader wave / consumer=compute waves) sync -----------
    @flyc.jit
    def _ring_wait(L, tid, q):
        """Compute: block until the A-loader staged A K-tile ``q`` into its ring bank."""
        if (tid % i32(64)) == i32(0):
            spin_lds_ge(
                L, L_RCTL + (i32(RC_ASEQ) + i32(q) % i32(NAB)) * i32(4), i32(q) + i32(1)
            )
        rocdl.sched_barrier(0)

    @flyc.jit
    def _ring_release(L, tid, q):
        """Compute: this wave finished reading A K-tile ``q``; free its ring bank (the
        NW-th release lets the loader overwrite the bank's next generation)."""
        rocdl.sched_barrier(0)
        if (tid % i32(64)) == i32(0):
            lds_atomic_add(
                L, L_RCTL + (i32(RC_AFREE) + i32(q) % i32(NAB)) * i32(4), 1, REL
            )

    @flyc.jit
    def _loader_wait_free(L, lane, q):
        """Loader: before (over)writing ring bank ``q % NAB`` for generation ``q // NAB``,
        wait until all NW compute waves released the bank's previous generation."""
        if lane == i32(0):
            spin_lds_ge(
                L,
                L_RCTL + (i32(RC_AFREE) + i32(q) % i32(NAB)) * i32(4),
                i32(NW) * (i32(q) // i32(NAB)),
            )
        rocdl.sched_barrier(0)

    @flyc.jit
    def _loader_publish(L, lane, q):
        """Loader: A K-tile ``q`` is staged into its ring bank; publish the produced seq."""
        if lane == i32(0):
            lds_st_rel(
                L, L_RCTL + (i32(RC_ASEQ) + i32(q) % i32(NAB)) * i32(4), i32(q) + i32(1)
            )

    def _part_base(a, src_rank):
        """Runtime-select source rank ``src_rank``'s ``part`` region base (peer arena +
        off_part), for the fused-combine GEMM2 epilogue's direct scatter-back."""
        v = fx.Int64(a["peer"][0])
        for j in range_constexpr(1, MAX_NPES):
            v = (src_rank == i32(j)).select(fx.Int64(a["peer"][j]), v)
        return v + fx.Int64(a["off_part"])

    MT = RG // 16  # MFMA row tiles (16 rows each) == M_REPEAT
    assert MT == M_REPEAT
    a_pack = 1 if a8 else 2  # bytes per element (FP8=1, packed FP4=2 per byte)
    a_row_bytes = H // a_pack  # payload (A) bytes per row
    scale_cols = H // 32  # E8M0 A-scale groups per row
    inter_row_bytes = I // a_pack  # quantized intermediate bytes per row
    inter_scols = I // 32  # E8M0 inter-scale groups per row
    K2 = I // 128  # GEMM2 K steps (MFMA K=128)
    max_norm = 448.0 if a8 else 6.0
    a_dtype = "fp8" if a8 else "fp4"
    n_per_wave = TILE_N // NW  # GEMM1 output cols per wave of one n-tile
    cs_tile_n = TILE_N // 2  # intermediate cols produced by one n-tile
    N_ACC = M_REPEAT * NUM_ACC_N
    # GEMM2 (down-proj) pipeline geometry: A = the quantized intermediate [RG, I]
    # (K-dim = I), B = W2 [H, I] (N-dim = H), output bf16 [RG, H]. tile_n2=TILE_N=256
    # (the L_A pool is sized RG*TILE_N*f32 for exactly this cshuffle) halves the n-tile
    # count (-> fewer cbars) vs 128. A's K tile step (A_K_STEP) and the A-ring stride
    # (A_LDS_I32) are identical to GEMM1 (same TILE_K).
    TILE_N2 = TILE_N
    n_per_wave2 = TILE_N2 // NW
    cs_tile_n2 = TILE_N2  # no gate|up pairing -> output cols == n cols
    N_TILES2 = H // TILE_N2
    NUM_ACC_N2 = (TILE_N2 // NW) // 16
    K_ITERS2 = I // TILE_K
    N_ACC2 = M_REPEAT * NUM_ACC_N2
    INTER_NBYTES = 256 * 64 * inter_row_bytes  # cosmetic x_tensor extent (NCTA*rg_max)

    def _act_batch(gs, us):
        return _activation_mul_batch(
            gs,
            us,
            act=act,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            swiglu_limit=swiglu_limit,
        )

    # ---- compute-wave (NW) LDS generation barrier (mirror schedule.cbar) ---------
    @flyc.jit
    def cbar(L, tid):
        wait_lgkm0()
        if (tid % i32(64)) == i32(0):
            g = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
            old = lds_atomic_add(L, L_CTL + C_BARCNT * 4, 1, ACQ_REL)
            if old == i32(NW - 1):
                lds_st(L, L_CTL + C_BARCNT * 4, i32(0))
                lds_st_rel(L, L_CTL + C_BARGEN * 4, g + i32(1))
            else:
                cur = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
                while cur == g:
                    rocdl.s_sleep(0)
                    cur = lds_ld_acq(L, L_CTL + C_BARGEN * 4)
        rocdl.sched_barrier(0)

    # ================= GEMM2 helpers (naive MFMA path, unchanged) =================
    def _afrag(rbuf, row_base, k, q4):
        """One scaled-MFMA activation fragment (this lane's K=128 slice) read straight
        from a global row at byte ``row_base``."""
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
        """E8M0 activation scale byte for this lane's K group (row-major)."""
        off = scale_base + i32(k * 4) + q4
        return fx.Int32(bld(rbuf, off, 0, T.i8, 0)) & i32(0xFF)

    def _wfrag(rw, col0, k, lane, kdim):
        """One FP4 weight fragment (direct natural [out_col, kdim] read)."""
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
    def a_loader_g1(L, lane, a, expert, e, s, P, uord):
        """Dedicated A-loader wave producer for one GEMM1 unit: stage the A K-tiles of
        echunk ``e``'s n-tile shard into the NAB-bank LDS ring (the identical swizzle the
        compute side reads back via ``AS2RLoader``), paced by the RC_AFREE back-pressure.
        ``lane`` is loader-local (0..NLOAD-1 = ``tid - B_LOADER``, also the ATileLoader
        ``tid_override``).  Mirrors ``gemm1``'s unit / n-tile / K walk so the running ring
        tile index ``q`` matches the consumer."""
        payload = _global_byte_tensor(
            fx.Int64(a["mine"]) + fx.Int64(a["off_payload"]), TMAX * a_row_bytes
        )
        rm_rsrc = ptr_buf_tensor(fx.Int64(a["rowmap"]), fx.Int32)
        a_gather = ATileLoader(
            row_bytes=a_row_bytes,
            sort_block_m=RG,
            k_step_bytes=A_K_STEP,
            total_threads=NLOAD,
            swizzle=True,
            x_tensor=payload,
            async_copy=False,
            indexed_input=True,
            row_map_rsrc=rm_rsrc,
            source_rows=TMAX,
            a_dtype=a_dtype,
            tid_override=lane,
        )
        a_buf = _LdsView(_lds_byte_ptr(L, L_A))
        tib = e * i32(RG)
        a_gather.for_tile(tib)
        nt_lo = s * i32(N_TILES) // P
        nt_hi = (s + i32(1)) * i32(N_TILES) // P
        ntpu = nt_hi - nt_lo

        # Plain closure (as in gemm1): the runtime n-tile loop body is only a plain call so
        # a_gather isn't treated as scf.for loop-carried state.
        def _produce_nt(nt):
            q0 = uni((uord * ntpu + (nt - nt_lo)) * i32(K_ITERS))
            for sp in range_constexpr(K_ITERS):
                q = q0 + i32(sp)
                _loader_wait_free(L, lane, q)  # back-pressure on the ring bank
                a_gather.store(
                    a_buf, a_gather.load_regs(i32(sp * A_K_STEP)), _ring_off(q, 0)
                )
                wait_lgkm0()  # the A payload ds_write landed in LDS before publishing
                _loader_publish(L, lane, q)

        for nt_ in range(nt_lo, nt_hi, i32(1)):
            _produce_nt(i32(nt_))

    # ============================ GEMM1 (pipelined) ==============================
    def _epilogue(L, tid, acc, n_tile, cptr, r_inter, r_inters, e):
        """CShuffle the gate|up accumulators -> silu(gate)*up -> E8M0-quant -> the
        row-major intermediate (natural col order) of echunk ``e`` (rows e*RG..), using
        ``cbar`` for the two LDS syncs.  Mirrors ``SiluQuantEpilogue.store`` (evec==2) but
        stores the E8M0 out-scale ROW-MAJOR (``row_g*inter_scols + col_s``) so the GEMM2
        reader is unchanged."""
        wave = tid // i32(64)
        lane = tid % i32(64)
        l16 = lane % i32(16)
        ld4 = (lane // i32(16)) * i32(4)
        nwo = cs_tile_n // NW
        cbase = wave * i32(nwo)
        gui_n = NUM_ACC_N // 2
        # combine gate/up accumulators + activation (round through bf16, as the naive
        # path did, to match the reference intermediate numerics)
        combined = []
        for mi in range_constexpr(M_REPEAT):
            for ni in range_constexpr(gui_n):
                g_idx = mi * NUM_ACC_N + ni * 2
                gv = fx.Vector(acc[g_idx])
                uv = fx.Vector(acc[g_idx + 1])
                gs = [fx.Float32(gv[v]) for v in range(4)]
                us = [fx.Float32(uv[v]) for v in range(4)]
                xs = [
                    fx.Float32(x).to(fx.BFloat16).to(fx.Float32)
                    for x in _act_batch(gs, us)
                ]
                combined.append(fx.Vector.from_elements(xs, fx.Float32))
        n_per = gui_n
        # cshuffle write (wave-local 32 cols)
        for mi in range_constexpr(M_REPEAT):
            for nj in range_constexpr(n_per):
                v4 = fx.Vector(combined[mi * n_per + nj])
                col = cbase + i32(nj * 16) + l16
                for ii in range_constexpr(4):
                    row = i32(mi * 16) + ld4 + i32(ii)
                    idx = row * i32(cs_tile_n) + col
                    ptr = fx.add_offset(cptr, fx.make_int_tuple(idx))
                    fx.ptr_store(fx.Vector.from_elements([v4[ii]], fx.Float32), ptr)
        cbar(L, tid)
        # read back + per-1x32 E8M0 quant + row-major store
        NLANE = 32
        EVEC = 2
        rows_per_iter = NW * 64 // NLANE
        mlane = tid // i32(NLANE)
        nlane = tid % i32(NLANE)
        m_reps = RG // rows_per_iter
        n_reps = cs_tile_n // (NLANE * EVEC)
        c64 = i32(64)
        for mr in range_constexpr(m_reps):
            row = i32(mr * rows_per_iter) + mlane
            row_g = e * i32(RG) + row
            out_row = row_g * i32(inter_row_bytes)
            s_row = row_g * i32(inter_scols)
            for nr in range_constexpr(n_reps):
                col0 = i32(nr * NLANE * EVEC) + nlane * i32(EVEC)
                idx = row * i32(cs_tile_n) + col0
                f32_iter = fx.recast_iter(
                    fx.Float32, fx.add_offset(cptr, fx.make_int_tuple(idx))
                )
                frag = fx.make_view(f32_iter, fx.make_layout(EVEC, 1)).load()
                v0 = frag[0]
                v1 = frag[1]
                a0 = v0.maximumf(fx.Float32(0.0) - v0)
                a1 = v1.maximumf(fx.Float32(0.0) - v1)
                m = a0.maximumf(a1)
                for off in (1, 2, 4, 8):
                    m = m.maximumf(m.shuffle_xor(i32(off), c64))
                max_rounded = (m.bitcast(fx.Int32) + i32(0x400000)) & i32(0xFF800000)
                _e = (max_rounded >> i32(23)) - i32(8 if a8 else 2)
                e8 = (_e > i32(0)).select(_e, i32(0))
                gcol = i32(n_tile) * i32(cs_tile_n) + col0
                if const_expr(a8):
                    qscale = ((i32(254) - e8) << i32(23)).bitcast(fx.Float32)
                    s0 = v0 * qscale
                    s1 = v1 * qscale
                    packed = rocdl.cvt_pk_fp8_f32(T.i32, s0, s1, i32(0), 0)
                    bst(fx.Int32(packed).to(fx.Int16), r_inter, out_row + gcol, 0, 0)
                else:
                    dequant = (e8 << i32(23)).bitcast(fx.Float32)
                    packed = rocdl.cvt_scalef32_pk_fp4_f32(
                        T.i32, i32(0), v0, v1, dequant, 0
                    )
                    bst(
                        fx.Int32(packed).to(fx.Int8),
                        r_inter,
                        out_row + gcol // i32(2),
                        0,
                        0,
                    )
                # Every lane of a 1x32 group holds the same reduced max -> same e8 and
                # same col_s (gcol>>5), so all 16 writers store the identical byte; no
                # dynamic group-writer guard needed (this fn is not in the jit scope).
                bst(e8.to(fx.Int8), r_inters, s_row + gcol // i32(32), 0, 0)
        wait_lgkm0()
        cbar(L, tid)

    @flyc.jit
    def gemm1(L, tid, a, expert, e, s, P, uord):
        """GEMM1 (A x W1 gate|up) + activation -> E8M0-quantized global intermediate of
        echunk ``e`` (inter[e*RG..]).  ``s``/``P`` select this CTA's n-tile shard
        ``[s*N_TILES//P, (s+1)*N_TILES//P)`` of the 2*I gate|up cols (P-way CU split).

        TP-style A path: the dedicated A-loader wave (``a_loader_g1``) stages each A K-tile
        into the LDS NAB-bank ring; the compute waves read them back via ``AS2RLoader`` and
        synchronise through the RC_ASEQ (produced) / RC_AFREE (released) ring counters, so
        the HBM A-load latency is off the compute critical path.  ``uord`` is this CTA's
        unit ordinal in its stride-#CU walk, so the running ring-tile index matches the
        loader.  The A-scale + W1 weights are still staged by the compute wave."""
        rm_rsrc = ptr_buf_tensor(fx.Int64(a["rowmap"]), fx.Int32)
        a_s2r = AS2RLoader(k_step_bytes=A_K_STEP, swizzle=True, a_dtype=a_dtype)
        # Fold the per-expert weight offset into the 64-bit resource BASE (not b_row): the
        # B loaders use a 32-bit voffset, which b_row=expert*2*I would overflow for large
        # local-expert ids. b_row then addresses only within one expert's W1 slice.
        w_rsrc = ptr_buf_tensor(
            fx.Int64(a["w1"]) + fx.Int64(expert) * fx.Int64(2 * I * (H // 2)),
            fx.Int32,
            unit_elems=4,
        )
        b_loader = BWeightLoader(w_rsrc=w_rsrc, num_acc_n=NUM_ACC_N, model_dim=H)
        sw_rsrc = ptr_buf_tensor(
            fx.Int64(a["w1s"]) + fx.Int64(expert) * fx.Int64(2 * I * (H // 32)),
            fx.Int32,
        )
        b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=NUM_ACC_N, model_dim=H)
        sx_rsrc = ptr_buf_tensor(
            fx.Int64(a["mine"]) + fx.Int64(a["off_pscale"]), fx.Int32, unit_elems=4
        )
        a_scale = AScaleLoader(
            scale_rsrc=sx_rsrc,
            m_repeat=M_REPEAT,
            model_dim=H,
            sort_block_m=RG,
            total_threads=NT,
            indexed_input=True,
            row_map_rsrc=rm_rsrc,
        )
        mfma_gu = MfmaScaleGU(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_dtype=a_dtype)

        a_buf = _LdsView(_lds_byte_ptr(L, L_A))
        a_scale_lds = _LdsView(_lds_byte_ptr(L, L_AS))
        cptr = _lds_f32_ptr(L, L_C)  # cshuffle C tile (own region; the A-ring runs ahead)
        r_inter = rsrc(a["inter"])
        r_inters = rsrc(a["inters"])
        wave_id = tid // i32(64)
        tib = e * i32(RG)
        nt_lo = s * i32(N_TILES) // P  # this CTA's n-tile shard (P-way CU split)
        nt_hi = (s + i32(1)) * i32(N_TILES) // P
        ntpu = nt_hi - nt_lo  # n-tiles this unit runs (matches a_loader_g1's walk)

        # The per-n-tile body is a plain closure: a runtime ``range`` loop that invokes
        # a *method* on an atom treats that atom as scf.for loop-carried state (and
        # chokes extracting IR from it), so the atom method calls live inside this
        # function and the loop body is only a plain call.
        # Stage the A-scale ONCE per unit (the same per-echunk block for every n-tile):
        # off the per-n-tile critical path, one cbar instead of N_TILES, and it lets the
        # waves keep drifting across n-tiles without a cooperative re-stage each time.
        cbar(L, tid)  # sync waves entering the unit before the cooperative stage
        a_scale.stage(a_scale_lds, tib)
        cbar(L, tid)  # A-scale staged + visible to all compute waves

        # Plain closure (see the module note): the runtime n-tile loop body is only a plain
        # call so the atoms aren't treated as scf.for loop-carried state.  No nt-boundary
        # cbar -- the previous n-tile's epilogue (or the per-unit stage above) already
        # re-synced the waves, and the A-scale is staged once above.
        def _run_ntile(nt):
            b_row = wave_id * i32(n_per_wave) + nt * i32(TILE_N)
            q0 = uni((uord * ntpu + (nt - nt_lo)) * i32(K_ITERS))  # running ring-tile base
            _ring_wait(L, tid, q0)  # wait the loader staged A K-step 0
            b_prev = b_loader.load_step(b_row, i32(0))
            acc = [mfma_gu.zero_value for _ in range(N_ACC)]
            for sp in range_constexpr(K_ITERS - 1):
                cur_off = _ring_off(q0, sp)
                spn = sp + 1
                sb = b_scale.load_step(b_row, i32(sp))
                sa = a_scale.load_step(a_scale_lds, i32(sp))

                def _al(mi, ks, _b=cur_off):
                    return a_s2r.load_operand(a_buf, mi, ks, _b)

                def _ln(ni, _k=spn, _br=b_row):
                    return b_loader.load_ni(_br, ni, i32(_k))

                acc, b_prev = mfma_gu.call_pipe(_al, b_prev, acc, sa, sb, _ln)
                _ring_release(L, tid, q0 + i32(sp))  # done reading A K=sp
                _ring_wait(L, tid, q0 + i32(spn))  # wait A K=sp+1 staged
                # No per-K-step cbar: A is loader-produced (RC_ASEQ/RC_AFREE ring sync) and
                # call_pipe has no cross-wave LDS write, so the 4 compute waves drift freely
                # (ring back-pressure bounds the drift to <=NAB K-steps) -> their load stalls
                # interleave and fill the MFMA pipe at 1 wave/SIMD.  Waves re-sync at the
                # pre-epilogue cbar below.
            final_off = _ring_off(q0, K_ITERS - 1)

            def _alf(mi, ks, _b=final_off):
                return a_s2r.load_operand(a_buf, mi, ks, _b)

            sb = b_scale.load_step(b_row, i32(K_ITERS - 1))
            sa = a_scale.load_step(a_scale_lds, i32(K_ITERS - 1))
            acc = mfma_gu.call_pipe_am_final(_alf, b_prev, acc, sa, sb)
            _ring_release(L, tid, q0 + i32(K_ITERS - 1))  # done reading the final A K-step
            cbar(L, tid)
            _epilogue(L, tid, acc, nt, cptr, r_inter, r_inters, e)

        for nt_ in range(nt_lo, nt_hi, i32(1)):
            _run_ntile(i32(nt_))

    # ============================ GEMM2 (pipelined) ==============================
    def _epilogue_g2(L, tid, acc, n_tile, cptr, r_gout, e):
        """CShuffle the GEMM2 accumulators -> bf16 -> ``gout`` (row-major
        [e*RG+row, H]) with ``cbar`` syncs.  No activation / quant (combine_final
        applies the top-k weight).  All RG rows are written (combine_send reads only the
        valid prefix), so no dynamic per-row guard is needed."""
        wave = tid // i32(64)
        lane = tid % i32(64)
        l16 = lane % i32(16)
        ld4 = (lane // i32(16)) * i32(4)
        cbase = wave * i32(cs_tile_n2 // NW)
        for mi in range_constexpr(M_REPEAT):
            for ni in range_constexpr(NUM_ACC_N2):
                v4 = fx.Vector(acc[mi * NUM_ACC_N2 + ni])
                col = cbase + i32(ni * 16) + l16
                for ii in range_constexpr(4):
                    row = i32(mi * 16) + ld4 + i32(ii)
                    idx = row * i32(cs_tile_n2) + col
                    ptr = fx.add_offset(cptr, fx.make_int_tuple(idx))
                    fx.ptr_store(fx.Vector.from_elements([v4[ii]], fx.Float32), ptr)
        cbar(L, tid)
        EVEC2 = 8
        NLANE2 = cs_tile_n2 // EVEC2
        rows_per_iter = NW * 64 // NLANE2
        mlane = tid // i32(NLANE2)
        nlane = tid % i32(NLANE2)
        m_reps = RG // rows_per_iter
        n_reps = cs_tile_n2 // (NLANE2 * EVEC2)
        for mr in range_constexpr(m_reps):
            row = i32(mr * rows_per_iter) + mlane
            obase = (e * i32(RG) + row) * i32(H)  # gout bf16 row base
            for nr in range_constexpr(n_reps):
                col0 = i32(nr * NLANE2 * EVEC2) + nlane * i32(EVEC2)
                idx = row * i32(cs_tile_n2) + col0
                f32_iter = fx.recast_iter(
                    fx.Float32, fx.add_offset(cptr, fx.make_int_tuple(idx))
                )
                frag = fx.make_view(f32_iter, fx.make_layout(EVEC2, 1)).load()
                gcol = i32(n_tile) * i32(cs_tile_n2) + col0
                ov = fx.Vector.from_elements(
                    [fx.Float32(frag[v]).to(fx.BFloat16) for v in range(EVEC2)],
                    fx.BFloat16,
                )
                bst(ov, r_gout, (obase + gcol) * i32(2), 0, 0)
        wait_lgkm0()
        cbar(L, tid)

    def _gemm2_core(L, tid, a, expert, e, epi, uord, nt_lo, ntpu):
        """Shared GEMM2 pipeline body for echunk ``e``: A = inter[e*RG..] (non-indexed,
        A-scale ``inters``), B = W2, cshuffle the accumulators to the LDS C-tile, then
        ``epi(acc, nt, cptr)`` consumes it (gout store or fused part scatter).

        TP-style A path: the dedicated A-loader wave (``a_loader_g2``) stages each inter
        K-tile into the LDS NAB-bank ring; this reads them back via ``AS2RLoader`` + the
        RC_ASEQ/RC_AFREE ring sync (same mechanism as GEMM1), so the HBM inter-load latency
        is off the compute critical path.  ``uord``/``nt_lo``/``ntpu`` give the running ring
        tile index (matches the loader).  Returns ``_run_ntile(nt)``; the @flyc.jit caller
        drives the dynamic n-tile loop.  A plain def so both wrappers share the identical
        GEMM.  The A-scale + W2 weights are still staged by the compute wave."""
        a_s2r = AS2RLoader(k_step_bytes=A_K_STEP, swizzle=True, a_dtype=a_dtype)
        w_rsrc = ptr_buf_tensor(
            fx.Int64(a["w2"]) + fx.Int64(expert) * fx.Int64(H * (I // 2)),
            fx.Int32,
            unit_elems=4,
        )
        b_loader = BWeightLoader(w_rsrc=w_rsrc, num_acc_n=NUM_ACC_N2, model_dim=I)
        sw_rsrc = ptr_buf_tensor(
            fx.Int64(a["w2s"]) + fx.Int64(expert) * fx.Int64(H * (I // 32)), fx.Int32
        )
        b_scale = BScaleLoader(scale_rsrc=sw_rsrc, num_acc_n=NUM_ACC_N2, model_dim=I)
        sx_rsrc = ptr_buf_tensor(fx.Int64(a["inters"]), fx.Int32, unit_elems=4)
        a_scale = AScaleLoader(
            scale_rsrc=sx_rsrc,
            m_repeat=M_REPEAT,
            model_dim=I,
            sort_block_m=RG,
            total_threads=NT,
            indexed_input=False,
        )
        mfma_gu = MfmaScaleGU(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N2, a_dtype=a_dtype)
        a_buf = _LdsView(_lds_byte_ptr(L, L_A))
        a_scale_lds = _LdsView(_lds_byte_ptr(L, L_AS))
        cptr = _lds_f32_ptr(L, L_C)  # cshuffle C tile (own region; the A-ring runs ahead)
        wave_id = tid // i32(64)
        tib = e * i32(RG)
        # Stage the A-scale ONCE per unit (same per-echunk block for every n-tile) -- off
        # the per-n-tile critical path (see gemm1).  Runs when the @flyc.jit wrapper calls
        # _gemm2_core, before its n-tile loop.
        cbar(L, tid)  # sync waves entering the unit before the cooperative stage
        a_scale.stage(a_scale_lds, tib)
        cbar(L, tid)  # A-scale staged + visible to all compute waves

        # No nt-boundary cbar (see gemm1): prev n-tile's epilogue / the per-unit stage
        # already re-synced the waves; A-scale is staged once above.
        def _run_ntile(nt):
            b_row = wave_id * i32(n_per_wave2) + nt * i32(TILE_N2)
            q0 = uni((uord * ntpu + (nt - nt_lo)) * i32(K_ITERS2))  # running ring-tile base
            _ring_wait(L, tid, q0)  # wait the loader staged inter K-step 0
            b_prev = b_loader.load_step(b_row, i32(0))
            acc = [mfma_gu.zero_value for _ in range(N_ACC2)]
            for sp in range_constexpr(K_ITERS2 - 1):
                cur_off = _ring_off(q0, sp)
                spn = sp + 1
                sb = b_scale.load_step(b_row, i32(sp))
                sa = a_scale.load_step(a_scale_lds, i32(sp))

                def _al(mi, ks, _b=cur_off):
                    return a_s2r.load_operand(a_buf, mi, ks, _b)

                def _ln(ni, _k=spn, _br=b_row):
                    return b_loader.load_ni(_br, ni, i32(_k))

                acc, b_prev = mfma_gu.call_pipe(_al, b_prev, acc, sa, sb, _ln)
                _ring_release(L, tid, q0 + i32(sp))  # done reading inter K=sp
                _ring_wait(L, tid, q0 + i32(spn))  # wait inter K=sp+1 staged
                # No per-K-step cbar (see gemm1): compute waves drift, the RC ring syncs A.
            final_off = _ring_off(q0, K_ITERS2 - 1)

            def _alf(mi, ks, _b=final_off):
                return a_s2r.load_operand(a_buf, mi, ks, _b)

            sb = b_scale.load_step(b_row, i32(K_ITERS2 - 1))
            sa = a_scale.load_step(a_scale_lds, i32(K_ITERS2 - 1))
            acc = mfma_gu.call_pipe_am_final(_alf, b_prev, acc, sa, sb)
            _ring_release(L, tid, q0 + i32(K_ITERS2 - 1))  # done reading the final K-step
            cbar(L, tid)
            epi(acc, nt, cptr)

        return _run_ntile  # caller (an @flyc.jit wrapper) drives the dynamic n-tile loop

    @flyc.jit
    def a_loader_g2(L, lane, a, e, nt_lo, ntpu, uord):
        """Dedicated A-loader producer for one GEMM2 unit: stage inter[e*RG..] K-tiles
        (GEMM1's output, non-indexed) into the NAB-bank LDS ring for the compute waves,
        paced by the RC_AFREE back-pressure.  ``lane`` is loader-local (= ``tid-B_LOADER``,
        the ATileLoader ``tid_override``).  Mirrors ``_gemm2_core``'s n-tile / K walk so the
        running ring tile index ``q`` matches the consumer."""
        inter_tensor = _global_byte_tensor(fx.Int64(a["inter"]), INTER_NBYTES)
        a_gather = ATileLoader(
            row_bytes=inter_row_bytes,
            sort_block_m=RG,
            k_step_bytes=A_K_STEP,
            total_threads=NLOAD,
            swizzle=True,
            x_tensor=inter_tensor,
            async_copy=False,
            indexed_input=False,
            a_dtype=a_dtype,
            tid_override=lane,
        )
        a_buf = _LdsView(_lds_byte_ptr(L, L_A))
        tib = e * i32(RG)
        a_gather.for_tile(tib)

        def _produce_nt(nt):
            q0 = uni((uord * ntpu + (nt - nt_lo)) * i32(K_ITERS2))
            for sp in range_constexpr(K_ITERS2):
                q = q0 + i32(sp)
                _loader_wait_free(L, lane, q)
                a_gather.store(
                    a_buf, a_gather.load_regs(i32(sp * A_K_STEP)), _ring_off(q, 0)
                )
                wait_lgkm0()
                _loader_publish(L, lane, q)

        for nt_ in range(nt_lo, nt_lo + ntpu, i32(1)):
            _produce_nt(i32(nt_))

    @flyc.jit
    def gemm2_pipe(L, tid, a, expert, e, s, P, src_out, uord):
        """GEMM2 -> bf16 ``gout[e*RG..]`` for echunk ``e`` (n-tile shard
        ``[s*N_TILES2//P, (s+1)*N_TILES2//P)``; P-way CU split).  Thin @flyc.jit wrapper
        over ``_gemm2_core`` with the gout epilogue.  ``uord`` = this CTA's unit ordinal."""
        nt_lo = s * i32(N_TILES2) // P  # this CTA's n-tile shard (P-way CU split)
        nt_hi = (s + i32(1)) * i32(N_TILES2) // P
        run_ntile = _gemm2_core(
            L, tid, a, expert, e,
            lambda acc, nt, cptr: _epilogue_g2(L, tid, acc, nt, cptr, src_out, e),
            uord, nt_lo, nt_hi - nt_lo,
        )
        for nt_ in range(nt_lo, nt_hi, i32(1)):
            run_ntile(i32(nt_))

    # ================= GEMM2 with FUSED combine (P==1 overlap path) ================
    @flyc.jit
    def _scatter_combine(L, tid, a, e, rows, n_tile, cptr):
        """Scatter this n-tile's cshuffled C-tile slice (the H cols
        ``[n_tile*TILE_N2, +TILE_N2)``) of each VALID row straight to its source rank's
        ``part`` (async AUX_SYS P2P -- only xGMI, no local-HBM gout round-trip, so it
        overlaps the next n-tile's MFMA).  Mirrors ``combine_send`` EXACTLY (wave-strided
        rows, lane-strided V4I cols; only valid rows iterated so padding is skipped) --
        the sole change is reading the LDS C-tile (``cptr``) instead of global ``gout``."""
        HV = H // 8
        mtpr = a["mtpr"]
        r_rowidx = rsrc(a["rowidx"])
        wave = tid // i32(64)
        lane = tid % i32(64)
        NV = cs_tile_n2 // 8  # V4I (8-bf16) cols produced by this n-tile
        hv0 = i32(n_tile) * i32(NV)  # this n-tile's base in the row's HV V4I cols
        for k_ in range(wave, rows, i32(NW)):
            k = i32(k_)
            idx = fx.Int32(bld(r_rowidx, (e * i32(RG) + k) * i32(4), 0, T.i32, 0))
            sk = idx // i32(TOPK)
            route = idx - sk * i32(TOPK)
            src_rank = sk // mtpr
            slot = (sk - src_rank * mtpr) * i32(TOPK) + route
            dpart = rsrc(_part_base(a, src_rank))
            pbase = slot * i32(HV) + hv0
            for g_ in range(lane, i32(NV), i32(64)):
                g = i32(g_)
                col0 = g * i32(8)
                cidx = k * i32(cs_tile_n2) + col0
                f32_iter = fx.recast_iter(
                    fx.Float32, fx.add_offset(cptr, fx.make_int_tuple(cidx))
                )
                frag = fx.make_view(f32_iter, fx.make_layout(8, 1)).load()
                ov = fx.Vector.from_elements(
                    [fx.Float32(frag[v]).to(fx.BFloat16) for v in range(8)],
                    fx.BFloat16,
                )
                bst(ov, dpart, (pbase + g) * i32(16), 0, AUX_SYS)
        # the remote part stores overlap the next n-tile's MFMA (gemm2_combine drains
        # them once at the end, before the arrives); the cptr (LDS) reads must finish
        # before the next n-tile's cshuffle overwrites the C-tile.
        wait_lgkm0()
        cbar(L, tid)

    def _epilogue_g2_combine(L, tid, acc, n_tile, cptr, a, e, rows):
        """Fused-combine GEMM2 epilogue: cshuffle the accumulators to the LDS C-tile (same
        as ``_epilogue_g2``), then ``_scatter_combine`` sends each row's bf16 output
        straight to its source ``part`` (no gout) -- the cross-rank P2P overlaps the next
        n-tile's MFMA."""
        wave = tid // i32(64)
        lane = tid % i32(64)
        l16 = lane % i32(16)
        ld4 = (lane // i32(16)) * i32(4)
        cbase = wave * i32(cs_tile_n2 // NW)
        for mi in range_constexpr(M_REPEAT):
            for ni in range_constexpr(NUM_ACC_N2):
                v4 = fx.Vector(acc[mi * NUM_ACC_N2 + ni])
                col = cbase + i32(ni * 16) + l16
                for ii in range_constexpr(4):
                    row = i32(mi * 16) + ld4 + i32(ii)
                    idx = row * i32(cs_tile_n2) + col
                    ptr = fx.add_offset(cptr, fx.make_int_tuple(idx))
                    fx.ptr_store(fx.Vector.from_elements([v4[ii]], fx.Float32), ptr)
        cbar(L, tid)
        _scatter_combine(L, tid, a, e, rows, n_tile, cptr)

    @flyc.jit
    def gemm2_combine(L, tid, a, expert, e, rows, uord):
        """GEMM2 (P==1) that FUSES combine into the cshuffle epilogue: the bf16 output
        scatters straight to each row's source ``part`` (no gout), the cross-rank P2P
        overlapping the next n-tile's MFMA; after the loop one wait_vm drains the scatters
        and a per-row combine_arrive signals the sources.  Thin @flyc.jit wrapper over
        ``_gemm2_core`` (the *same* GEMM as ``gemm2_pipe``) with the fused epilogue.
        ``rows`` = the echunk's valid row count (padding rows are skipped); ``uord`` = this
        CTA's unit ordinal (P==1 -> nt_lo=0, ntpu=N_TILES2)."""
        run_ntile = _gemm2_core(
            L, tid, a, expert, e,
            lambda acc, nt, cptr: _epilogue_g2_combine(L, tid, acc, nt, cptr, a, e, rows),
            uord, i32(0), i32(N_TILES2),
        )
        for nt_ in range(i32(0), i32(N_TILES2), i32(1)):
            run_ntile(i32(nt_))
        wait_vm(0)  # drain the fused part stores before signalling the sources
        parity = a["epoch"] & i32(1)
        mt = a["mtpr"]
        exp = a["m"] * i32(TOPK)
        for k_ in range(tid, rows, i32(NT)):
            k = i32(k_)
            idx = fx.Int32(bld(rsrc(a["rowidx"]), (e * i32(RG) + k) * i32(4), 0, T.i32, 0))
            combine_arrive(a, (idx // i32(TOPK)) // mt, parity, i32(0), i32(1), exp)

    @flyc.jit
    def _gemm2_naive(L, tid, a, expert, rows, src_out):
        """GEMM2 (intermediate x W2) -> model_dim bf16 rows into ``src_out`` (gout)."""
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

    # Select the pipelined GEMM2 (the naive path is kept as a fallback for debugging).
    gemm2 = gemm2_pipe

    return locals()

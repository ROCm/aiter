# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused GEMM1 + activation + GEMM2 for the EP MegaMoE kernel (all 8 waves compute).

Both GEMMs self-stage A into an LDS ring (``mega_moe.gemm_util`` ``ATileLoader``,
source-key gather via the per-echunk ``chunk_payload_row``, read back by
``_ALdsReader``; E8M0 A-scale via ``AScaleLoader``) and software-pipeline the weights
(``MfmaScaleGU`` + ``_WeightLoader``); the K-loops sync the 8 waves with the hardware
block barrier.  GEMM1's register epilogue writes the row-major E8M0-quantized
intermediate (``inter`` / ``inter_scale``); GEMM2's cshuffle epilogue scatters its rows
straight to the source ranks' ``part`` (fused combine).

K-loop: the A operand ds_reads are two per-lane pointers + immediate offsets
(``_ALdsReader``), W + B-scale go one K-step ahead with every wave-uniform address term
in the SGPR soffset + immediate offset (``_WeightLoader``), and each A operand is read
one (m-tile, ksub) ahead of its MFMAs (``_mfma_step``)."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value
from flydsl.expr.typing import Vector as Vec

from ..dpp_utils import update_dpp_i32
from ..mega_moe.gemm_util import (
    AScaleLoader,
    ATileLoader,
    MfmaScaleGU,
)
from ..mxfp4_gemm_common import _activation_mul_batch
from ..tensor_shim import ptr_buf_tensor
from .common import (
    ACQ_REL,
    AUX_SYS,
    V4I,
    _ctpop,
    bld,
    bst,
    i32,
    lds_atomic_add,
    lds_ld_acq,
    lds_st,
    lds_st_rel,
    rsrc,
    uni,
    uni64,
    wait_lgkm0,
    wait_vm,
)
from .communication import _quantize_group8  # per-8 fp8 combine codec (fused scatter)
from .mega_moe_ep_config import (
    CTL_BAR_COUNT,
    CTL_BAR_GEN,
    TILE_K,
    KernelCtx,
)

__all__ = ["build_gemm"]


class _LdsView:
    """Minimal ``.ptr`` holder the gemm_util atoms expect for an LDS buffer."""

    def __init__(self, ptr):
        self.ptr = ptr


def _lds_byte_ptr(L, off):
    """Byte-typed LDS pointer at ``L + off`` (the gemm_util atoms and ``_ALdsReader``
    offset the raw ``.ptr`` by byte counts, so the element type must be ``i8``).  Uses
    the MLIR type ``T.i8`` -- NOT the fx class."""
    pt = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Shared, alignment=16)
    return fx.inttoptr(pt, uni(i32(L) + i32(off)))


def _lds_f32_ptr(L, off):
    """f32-typed LDS pointer at ``L + off`` (the cshuffle epilogue addresses the C tile
    in f32 elements)."""
    pt = fx.PointerType.get(T.f32, address_space=fx.AddressSpace.Shared, alignment=16)
    return fx.inttoptr(pt, uni(i32(L) + i32(off)))


def _opaque_v(v):
    """``v`` through a ``v_mov``: the compiler no longer knows its high bits are zero, so
    ``v + C`` stays an add (not a disjoint ``or``) that the backend splits into the
    buffer instruction's immediate offset."""
    return fx.Int32(
        _llvm.InlineAsmOp(
            T.i32, [as_ir_value(i32(v))], "v_mov_b32 $0, $1", "=v,v"
        ).result
    )


def _dpp_quad_perm(l0, l1, l2, l3):
    """DPP ``quad_perm`` control: lane j of every 4-lane quad reads its lane ``l_j``."""
    return l0 | l1 << 2 | l2 << 4 | l3 << 6


DPP_XOR1 = _dpp_quad_perm(1, 0, 3, 2)  # lane ^ 1
DPP_XOR2 = _dpp_quad_perm(2, 3, 0, 1)  # lane ^ 2
DPP_ROW_HALF_MIRROR = 0x141  # lane i of each 8 reads lane 7 - i
DPP_ROW_MIRROR = 0x140  # lane i of the 16-lane row reads lane 15 - i

F32_MANTISSA_BITS = 23
F32_MANTISSA_HALF = 1 << (F32_MANTISSA_BITS - 1)
F32_SIGN_EXP_MASK = 0xFF800000


def _dpp_f32(x, ctrl):
    """``x`` from the lane DPP ``ctrl`` selects (within a 16-lane row)."""
    xi = fx.Float32(x).bitcast(fx.Int32)
    return fx.Int32(update_dpp_i32(xi, xi, ctrl, 0xF, 0xF, True)).bitcast(fx.Float32)


class _ALdsReader:
    """``AS2RLoader`` (same swizzle) with the lane part of the address computed once: the
    XOR mask depends on row & 14 (fp4) / row & 15 (fp8) only, so for row = mi*16 + lane%16
    it is the same for every m-tile, and operand (mi, ksub, bank) is a constant byte
    offset off its ksub's lane pointer -- the ds_read immediate."""

    def __init__(self, lds_ptr, *, k_step_bytes, a_dtype):
        self._ksb = k_step_bytes
        self._fp4 = a_dtype == "fp4"
        lane = i32(fx.thread_idx.x) % i32(64)
        l16 = lane % i32(16)
        klane4 = (lane // i32(16)) * i32(4)
        mask = ((l16 & i32(14)) << i32(1)) if self._fp4 else ((l16 & i32(15)) << i32(2))
        row = l16 * i32(k_step_bytes // 4)
        self._p = [
            [
                fx.add_offset(
                    lds_ptr,
                    fx.make_int_tuple(
                        (
                            row
                            + (
                                (klane4 + i32(ks * (16 if self._fp4 else 32) + h * 16))
                                ^ mask
                            )
                        )
                        * i32(4)
                    ),
                )
                for h in range(1 if self._fp4 else 2)
            ]
            for ks in range(2)
        ]

    def _ld(self, p, nbytes):
        it = fx.recast_iter(fx.Uint8, fx.add_offset(p, fx.make_int_tuple(nbytes)))
        return Vec(fx.make_view(it, fx.make_layout(16, 1)).load()).bitcast(fx.Int32)

    def load_operand(self, mi, ksub, bank_bytes):
        c = mi * 16 * self._ksb + bank_bytes
        lo = self._ld(self._p[ksub][0], c)
        if self._fp4:
            return lo
        lo = lo.bitcast(fx.Int64)
        hi = self._ld(self._p[ksub][1], c).bitcast(fx.Int64)
        return Vec.from_elements([lo[0], lo[1], hi[0], hi[1]], fx.Int64).bitcast(
            fx.Int32
        )


class _WeightLoader:
    """``BWeightLoader`` + ``BScaleLoader`` (same layouts) with the lane term a fixed VGPR,
    the k-step's low 12 bits in the immediate offset and the other wave-uniform terms
    (n-block, ni, group, k-step high bits) in the SGPR soffset: no per-load VALU address
    math and only one distinct soffset per 4 W k-substeps / 16 scale k-steps, so the
    hoisted soffsets do not spill SGPRs in the unrolled k-loop."""

    def __init__(self, w_addr, s_addr, b_row, *, num_acc_n, model_dim, aux=0):
        lane = i32(fx.thread_idx.x) % i32(64)
        self._wr = rsrc(w_addr)
        self._sr = rsrc(s_addr)
        self._aux = aux
        self._wv = _opaque_v((lane // i32(16)) * i32(256) + (lane % i32(16)) * i32(16))
        self._sv = _opaque_v(lane * i32(4))
        self._nacc = num_acc_n
        self._n0 = model_dim * 8  # bytes per 16-row n-block
        self._srow = (model_dim // 256) * 256  # bytes per 32-row scale group
        b = uni(b_row)
        self._ws = (b // i32(16)) * i32(self._n0)
        self._ss = (b // i32(32)) * i32(self._srow)

    def load_step(self, k):
        kw, kl = (k * 2) // 4, (
            k * 2
        ) % 4  # W k-substep: 4 x 1024 B per 4096 B soffset step
        w = [
            [
                Vec(
                    bld(
                        self._wr,
                        self._wv + i32((kl + ks) * 1024),
                        self._ws + i32(ni * self._n0 + kw * 4096),
                        V4I,
                        self._aux,
                    )
                )
                for ks in range(2)
            ]
            for ni in range(self._nacc)
        ]
        s = [
            fx.Int32(
                bld(
                    self._sr,
                    self._sv + i32((k % 16) * 256),
                    self._ss + i32(g * self._srow + (k // 16) * 4096),
                    T.i32,
                )
            )
            for g in range(self._nacc // 2)
        ]
        return w, s


def _mfma_step(mfma_gu, a_load, b, acc, sa, sb, m_repeat, num_acc_n):
    """``MfmaScaleGU.call`` with the A operand read one (m-tile, ksub) ahead: its
    ds_read is in flight under the previous operand's MFMAs instead of waited on by them.
    """
    order = [(ks, mi) for ks in range(2) for mi in range(m_repeat)]
    a_nx = a_load(order[0][1], order[0][0])
    for t, (ks, mi) in enumerate(order):
        a_op = a_nx
        if t + 1 < len(order):
            a_nx = a_load(order[t + 1][1], order[t + 1][0])
        for ni in range(num_acc_n):
            k = mi * num_acc_n + ni
            acc[k] = mfma_gu._mfma(
                a_op, b[ni][ks], acc[k], sa[mi // 2], sb[ni // 2], ks, mi % 2, ni % 2
            )
    return acc


def _global_byte_tensor(addr_i64, nbytes):
    """An ``fx.Tensor`` byte-view over a raw global address (for ``ATileLoader``)."""
    pt = fx.PointerType.get(T.i8, address_space=fx.AddressSpace.Global, alignment=16)
    it = fx.inttoptr(pt, fx.Int64(addr_i64))
    return fx.Tensor(fx.make_view(it, fx.make_layout(nbytes, 1)))


def build_gemm(kc: KernelCtx) -> dict:
    """The per-unit GEMM1+act+GEMM2 device functions of one instance (added to kc)."""
    H, I, TOPK, a8 = kc.get("H I TOPK a8")
    INTER_REBASE = kc.get("INTER_REBASE")[0]
    act, swiglu_limit = kc.get("act swiglu_limit")
    situ_beta, situ_linear_beta = kc.get("situ_beta situ_linear_beta")
    TILE_M, const_expr = kc.get("TILE_M const_expr")
    HW_BARRIER = kc.get("HW_BARRIER")[0]
    LDS_A_RING, LDS_A_SCALE, LDS_CTRL, TOKEN_SLOTS = kc.get(
        "LDS_A_RING LDS_A_SCALE LDS_CTRL TOKEN_SLOTS"
    )
    N_TILES, NUM_ACC_N, M_REPEAT = kc.get("N_TILES NUM_ACC_N M_REPEAT")
    A_K_STEP, K_ITERS, A_LDS_I32 = kc.get("A_K_STEP K_ITERS A_LDS_I32")
    # Inline-A ring: A_BANKS banks, K-step sp+A_AHEAD staged during step sp, one barrier
    # per A_AHEAD K-steps (bank sp+A_AHEAD last held K-step sp-A_AHEAD, read a barrier
    # earlier) and one after the last step (the next n-tile primes banks 0 .. A_AHEAD-1)
    A_BANKS = kc.get("A_BANKS")[0]
    A_AHEAD = A_BANKS // 2
    assert I % TILE_K == 0 and min(K_ITERS, I // TILE_K) >= A_AHEAD
    LDS_C_TILE = kc.get("LDS_C_TILE")[0]
    combine_arrive, peer_sel = kc.get("combine_arrive peer_sel")
    NW, NT, TILE_N, TILE_N2 = kc.get("NW NT TILE_N TILE_N2")
    WEIGHT_AUX = kc.get("WEIGHT_AUX")[0]  # W1 / W2 load cache policy (aux bits)
    N_RANKS = kc.get("N_RANKS")[0]
    assert TILE_M <= NT  # combine_arrive_rows: one thread per echunk row

    a_pack = 1 if a8 else 2  # bytes per element (FP8=1, packed FP4=2 per byte)
    a_row_bytes = H // a_pack  # payload (A) bytes per row
    inter_row_bytes = I // a_pack  # quantized intermediate bytes per row
    inter_scols = I // 32  # E8M0 inter-scale groups per row

    # inter / inter_scale base + first row of echunk e: past 2 GB (INTER_REBASE) the
    # 64-bit base absorbs e*TILE_M rows so the 32-bit buffer offsets stay within echunk
    # e.
    def _inter_at(a, e):
        if INTER_REBASE:
            return uni64(
                fx.Int64(a["inter"]) + fx.Int64(e) * fx.Int64(TILE_M * inter_row_bytes)
            )
        return fx.Int64(a["inter"])

    def _inter_scale_at(a, e):
        if INTER_REBASE:
            return uni64(
                fx.Int64(a["inter_scale"])
                + fx.Int64(e) * fx.Int64(TILE_M * inter_scols)
            )
        return fx.Int64(a["inter_scale"])

    def _inter_row0(e):
        return i32(0) if INTER_REBASE else e * i32(TILE_M)

    a_dtype = "fp8" if a8 else "fp4"
    # largest exponent of an element: fp8-e4m3 448 = 1.75 * 2^8, fp4-e2m1 6 = 1.5 * 2^2
    elem_emax = 8 if a8 else 2
    n_per_wave = TILE_N // NW  # GEMM1 output cols per wave of one n-tile
    inter_cols = TILE_N // 2  # intermediate cols produced by one n-tile
    N_ACC = M_REPEAT * NUM_ACC_N
    # GEMM2 (down-proj) pipeline geometry: A = the quantized intermediate [TILE_M, I]
    # (K-dim = I), B = W2 [H, I] (N-dim = H), output [TILE_M, H]; tile_n2 = TILE_N2.
    # A's K tile step (A_K_STEP) and the A-ring stride
    # (A_LDS_I32) are identical to GEMM1 (same TILE_K).
    n_per_wave2 = TILE_N2 // NW
    out_cols = TILE_N2  # no gate|up pairing -> output cols == n cols
    N_TILES2 = H // TILE_N2
    NUM_ACC_N2 = (TILE_N2 // NW) // 16
    K_ITERS2 = I // TILE_K
    N_ACC2 = M_REPEAT * NUM_ACC_N2
    # GEMM2 cshuffle C-tile is processed one MFMA m-tile (16 rows) at a time so the LDS
    # C region (LDS_C_TILE) holds only CROWS x out_cols f32 instead of TILE_M x
    # out_cols, while keeping the shuffled layout (coalesced combine P2P).
    CROWS = TILE_M // M_REPEAT  # == 16 (one MFMA m-tile of rows per cshuffle chunk)
    # nominal x_tensor extent: ATileLoader only takes the view's base iterator
    INTER_NBYTES = 256 * 64 * inter_row_bytes

    def _act_batch(gs, us):
        return _activation_mul_batch(
            gs,
            us,
            act=act,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
            swiglu_limit=swiglu_limit,
        )

    @flyc.jit
    def cta_barrier(L, tid):
        """CTA barrier of the NW waves: s_barrier (HW_BARRIER, small-batch kernel) or
        the LDS generation barrier."""
        wait_lgkm0()
        if const_expr(HW_BARRIER):
            gpu.barrier()
        else:
            if (tid % i32(64)) == i32(0):
                g = lds_ld_acq(L, LDS_CTRL + CTL_BAR_GEN * 4)
                old = lds_atomic_add(L, LDS_CTRL + CTL_BAR_COUNT * 4, 1, ACQ_REL)
                if old == i32(NW - 1):
                    lds_st(L, LDS_CTRL + CTL_BAR_COUNT * 4, i32(0))
                    lds_st_rel(L, LDS_CTRL + CTL_BAR_GEN * 4, g + i32(1))
                else:
                    cur = lds_ld_acq(L, LDS_CTRL + CTL_BAR_GEN * 4)
                    while cur == g:
                        rocdl.s_sleep(0)
                        cur = lds_ld_acq(L, LDS_CTRL + CTL_BAR_GEN * 4)
        rocdl.sched_barrier(0)

    # ============================ GEMM1 (pipelined) ==============================
    def _epilogue(tid, acc, n_tile, r_inter, r_inter_scale, e, mis):
        """gate|up accumulators -> silu(gate)*up -> E8M0-quant, in registers -> the
        row-major intermediate (natural col order) of echunk ``e`` (``r_inter`` /
        ``r_inter_scale`` are based at its first row).  The E8M0 out-scale is stored
        ROW-MAJOR (``row_g*inter_scols + col_s``), the layout the GEMM2 reader expects.
        ``mis``: the 16-row MFMA blocks to emit."""
        row0 = _inter_row0(e)
        lane = tid % i32(64)
        wave = tid // i32(64)
        l16 = lane % i32(16)
        q4 = lane // i32(16)
        # this wave's base col (of 32) within the n-tile
        cbase = wave * i32(inter_cols // NW)
        gui_n = NUM_ACC_N // 2
        # Quantize straight from the MFMA accumulators (no C-tile).  acc[mi][nj*2 | +1]
        # (gate|up) element v lands at (row = mi*16 + q4*4 + v, col = cbase + nj*16 + l16);
        # the wave's 32 cols (nj in {0,1}) are exactly one 1x32 E8M0 group of that row, so
        # the group-max = reg-max over nj + shuffle_xor over l16, and the fp4 pack of adjacent
        # cols reads the l16^1 neighbour (the even lane owns the (l16, l16+1) byte).
        for mi in mis:
            cnj = []
            for nj in range_constexpr(gui_n):
                g_idx = mi * NUM_ACC_N + nj * 2
                gv = fx.Vector(acc[g_idx])
                uv = fx.Vector(acc[g_idx + 1])
                gs = [fx.Float32(gv[v]) for v in range(4)]
                us = [fx.Float32(uv[v]) for v in range(4)]
                cnj.append(
                    [
                        fx.Float32(x).to(fx.BFloat16).to(fx.Float32)
                        for x in _act_batch(gs, us)
                    ]
                )
            for v in range_constexpr(4):
                row_g = row0 + i32(mi * 16) + q4 * i32(4) + i32(v)
                out_row = row_g * i32(inter_row_bytes)
                s_row = row_g * i32(inter_scols)
                c0, c1 = cnj[0][v], cnj[1][v]
                a0 = fx.absf(c0)
                a1 = fx.absf(c1)
                m = fx.max(a0, a1)
                # quad xor1, xor2, then 8- / 16-lane mirrors: every lane of the 16-lane
                # row ends with the row max (VALU DPP, no ds_swizzle round trip)
                for ctrl in (DPP_XOR1, DPP_XOR2, DPP_ROW_HALF_MIRROR, DPP_ROW_MIRROR):
                    m = fx.max(m, _dpp_f32(m, ctrl))
                # the max's biased f32 exponent, +1 from 1.5 * 2^e up (mantissa >= 0.5
                # carries into the exponent)
                max_rounded = (m.bitcast(fx.Int32) + i32(F32_MANTISSA_HALF)) & i32(
                    F32_SIGN_EXP_MASK
                )
                _e = (max_rounded >> i32(F32_MANTISSA_BITS)) - i32(elem_emax)
                e8 = fx.max(_e, i32(0))
                # this group's first global col
                gbase = i32(n_tile) * i32(inter_cols) + cbase
                # All lanes of the group hold the same reduced e8 -> redundant scale store
                # (no dynamic group-writer guard; this fn is not in the @flyc.jit scope).
                bst(e8.to(fx.Int8), r_inter_scale, s_row + gbase // i32(32), 0, 0)
                # fp4/fp8 packs 2 adjacent cols per store; the cols live on lanes l16 and
                # l16^1.  Both lanes compute the SAME (even_col, odd_col) pair via a parity
                # select on the shuffled neighbour and store the SAME byte -> no runtime
                # even-lane guard needed (redundant store, identical value).
                is_odd = l16 & i32(1)
                even = is_odd == i32(0)
                for nj in range_constexpr(gui_n):
                    cv = cnj[nj][v]
                    # neighbour col (l16 ^ 1)
                    cvn = _dpp_f32(cv, DPP_XOR1)
                    # the even col's value of this 2-col byte
                    first = even.select(cv, cvn)
                    second = even.select(cvn, cv)  # the odd col's value
                    gcol = gbase + i32(nj * 16) + l16 - is_odd  # even col of the pair
                    if const_expr(a8):
                        qscale = ((i32(254) - e8) << i32(23)).bitcast(fx.Float32)
                        packed = rocdl.cvt_pk_fp8_f32(
                            T.i32, first * qscale, second * qscale, i32(0), 0
                        )
                        bst(
                            fx.Int32(packed).to(fx.Int16), r_inter, out_row + gcol, 0, 0
                        )
                    else:
                        dequant = (e8 << i32(23)).bitcast(fx.Float32)
                        packed = rocdl.cvt_scalef32_pk_fp4_f32(
                            T.i32, i32(0), first, second, dequant, 0
                        )
                        bst(
                            fx.Int32(packed).to(fx.Int8),
                            r_inter,
                            out_row + gcol // i32(2),
                            0,
                            0,
                        )

    # ============================ inline-A GEMM1 ==================================
    @flyc.jit
    def gemm1(L, tid, a, expert, e, nt_lo, nt_hi, rows, pre_epi=None):
        """Inline-A GEMM1: the compute waves stage each A K-tile into an LDS bank ring
        THEMSELVES (``ATileLoader`` with ``total_threads=NT=512``) and read it back via
        ``_ALdsReader``, syncing with the hardware block barrier (all 8 waves are compute).
        Multi-banking + the 2 MFMA waves/SIMD hide the A-load + barrier latency."""
        payload_row_rsrc = ptr_buf_tensor(fx.Int64(a["chunk_payload_row"]), fx.Int32)
        payload = _global_byte_tensor(
            fx.Int64(a["own_arena"]) + fx.Int64(a["off_payload"]),
            TOKEN_SLOTS * a_row_bytes,
        )
        a_gather = ATileLoader(
            row_bytes=a_row_bytes,
            sort_block_m=TILE_M,
            k_step_bytes=A_K_STEP,
            total_threads=NT,
            swizzle=True,
            x_tensor=payload,
            async_copy=False,
            indexed_input=True,
            row_map_rsrc=payload_row_rsrc,
            source_rows=TOKEN_SLOTS,
            a_dtype=a_dtype,
        )
        w_addr = fx.Int64(a["w1"]) + fx.Int64(expert) * fx.Int64(2 * I * (H // 2))
        s_addr = fx.Int64(a["w1_scale"]) + fx.Int64(expert) * fx.Int64(
            2 * I * (H // 32)
        )
        a_scale_rsrc = ptr_buf_tensor(
            fx.Int64(a["own_arena"]) + fx.Int64(a["off_pscale"]), fx.Int32, unit_elems=4
        )
        a_scale = AScaleLoader(
            scale_rsrc=a_scale_rsrc,
            m_repeat=M_REPEAT,
            model_dim=H,
            sort_block_m=TILE_M,
            total_threads=NT,
            indexed_input=True,
            row_map_rsrc=payload_row_rsrc,
        )
        mfma_gu = MfmaScaleGU(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N, a_dtype=a_dtype)
        a_buf = _LdsView(_lds_byte_ptr(L, LDS_A_RING))
        a_scale_lds = _LdsView(_lds_byte_ptr(L, LDS_A_SCALE))
        r_inter = rsrc(_inter_at(a, e))
        r_inter_scale = rsrc(_inter_scale_at(a, e))
        wave_id = tid // i32(64)
        tile_row0 = e * i32(TILE_M)
        a_gather.for_tile(tile_row0)
        gpu.barrier()
        a_scale.stage(a_scale_lds, tile_row0)
        gpu.barrier()

        def _bank(j):
            return uni((i32(j) % i32(A_BANKS)) * i32(A_LDS_I32))

        def _run_ntile(nt):
            b_row = wave_id * i32(n_per_wave) + nt * i32(TILE_N)
            a_fold = _ALdsReader(a_buf.ptr, k_step_bytes=A_K_STEP, a_dtype=a_dtype)
            b_fold = _WeightLoader(
                w_addr, s_addr, b_row, num_acc_n=NUM_ACC_N, model_dim=H, aux=WEIGHT_AUX
            )

            # prime banks 0 .. A_AHEAD-1 (K-steps 0 .. A_AHEAD-1), prefetch the next
            # A_AHEAD K-steps into regs (HBM->reg, overlaps the barrier)
            for j in range_constexpr(A_AHEAD):
                a_gather.store(
                    a_buf, a_gather.load_regs(i32(j * A_K_STEP)), _bank(i32(j))
                )
            regs = [
                a_gather.load_regs(i32((A_AHEAD + j) % K_ITERS * A_K_STEP))
                for j in range(A_AHEAD)
            ]
            b_nx, sb_nx = b_fold.load_step(0)
            wait_lgkm0()
            gpu.barrier()
            acc = [mfma_gu.zero_value for _ in range(N_ACC)]
            for sp in range_constexpr(K_ITERS):
                b_cur, sb = b_nx, sb_nx
                sa = a_scale.load_step(a_scale_lds, i32(sp))

                def _load_a(mi, ks, _bank_off=(sp % A_BANKS) * A_LDS_I32 * 4):
                    return a_fold.load_operand(mi, ks, _bank_off)

                # store the oldest prefetched K-tile (K-step sp+A_AHEAD) into bank
                # sp+A_AHEAD (not one the MFMAs read before the next barrier), then
                # prefetch K-step sp+2*A_AHEAD (unconditional wrap so flyc doesn't treat
                # ``regs`` as conditionally-assigned scf state).
                a_gather.store(a_buf, regs[0], _bank(i32(sp + A_AHEAD)))
                regs = regs[1:] + [
                    a_gather.load_regs(i32(((sp + 2 * A_AHEAD) % K_ITERS) * A_K_STEP))
                ]
                if const_expr(sp + 1 < K_ITERS):
                    b_nx, sb_nx = b_fold.load_step(sp + 1)
                acc = _mfma_step(
                    mfma_gu, _load_a, b_cur, acc, sa, sb, M_REPEAT, NUM_ACC_N
                )
                if const_expr(sp % A_AHEAD == A_AHEAD - 1 or sp == K_ITERS - 1):
                    wait_lgkm0()  # the next banks' ds_writes landed
                    gpu.barrier()  # next banks visible; all waves done reading these
            if const_expr(pre_epi is not None):
                pre_epi()
            # 16-row blocks past the echunk's valid rows are skipped: their inter rows
            # only feed GEMM2 rows that are never scattered
            _epilogue(tid, acc, nt, r_inter, r_inter_scale, e, [0])
            for mi in range_constexpr(1, M_REPEAT):
                if rows > i32(mi * 16):
                    _epilogue(tid, acc, nt, r_inter, r_inter_scale, e, [mi])

        for nt_ in range(nt_lo, nt_hi, i32(1)):
            _run_ntile(i32(nt_))

    # ================= GEMM2 with FUSED combine ================
    @flyc.jit
    def _scatter_combine(L, tid, a, e, rows, n_tile, cptr, row_lo):
        """Scatter ONE cshuffle chunk (global rows ``[row_lo, row_lo+CROWS)``) of this
        n-tile's C-tile slice (the H cols ``[n_tile*TILE_N2, +TILE_N2)``) of each VALID row
        straight to its source rank's ``part`` (async AUX_SYS P2P -- only xGMI, no local-HBM
        round-trip, so it overlaps the next chunk's cshuffle): wave-strided rows,
        lane-strided 8-col groups, padding skipped, read from the LDS C-tile (``cptr``) at
        the chunk-LOCAL row ``lr`` (0..CROWS-1); the C-tile holds only this 16-row chunk
        (``_epilogue_g2_combine``'s per-mi cshuffle)."""
        mtpr = a["mtpr"]
        r_recv_slot = rsrc(a["chunk_recv_slot"])
        wave = tid // i32(64)
        lane = tid % i32(64)
        # part row bytes (H fp8 + H/8 per-8 E8M0); matches communication
        PART_ROW_BYTES = H + H // 8
        NV = out_cols // 8  # 8-element groups produced by this n-tile
        group0 = i32(n_tile) * i32(NV)  # this n-tile's base group in the part row
        hi = fx.min(i32(CROWS), rows - row_lo)  # valid rows within this chunk (>=0)
        for lr_ in range(wave, hi, i32(NW)):
            lr = i32(lr_)  # chunk-local row 0..CROWS-1 (== C-tile row)
            k = row_lo + lr  # global row within the echunk
            idx = fx.Int32(
                bld(r_recv_slot, (e * i32(TILE_M) + k) * i32(4), 0, T.i32, 0)
            )
            source_key = idx // i32(TOPK)
            route = idx - source_key * i32(TOPK)
            src_rank = source_key // mtpr
            slot = (source_key - src_rank * mtpr) * i32(TOPK) + route
            r_part = rsrc(peer_sel(a, src_rank) + fx.Int64(a["off_part"]))
            prow = slot * i32(PART_ROW_BYTES)
            # per-8 fp8: each lane reads 8 f32 of the C-tile and quantizes them to 8 fp8
            # (w0, w1: one 8-B store) + 1 E8M0 (coalesced across lanes).
            for g_ in range(lane, i32(NV), i32(64)):
                g = i32(g_)
                cidx = lr * i32(out_cols) + g * i32(8)  # C-tile LOCAL row
                f32_iter = fx.recast_iter(
                    fx.Float32, fx.add_offset(cptr, fx.make_int_tuple(cidx))
                )
                fv = fx.make_view(f32_iter, fx.make_layout(8, 1)).load()
                e8m0, w0, w1 = _quantize_group8(fv)
                gg = group0 + g  # global 8-group index in the part row
                bst(
                    fx.Vector.from_elements([fx.Int32(w0), fx.Int32(w1)], fx.Int32),
                    r_part,
                    prow + gg * i32(8),
                    0,
                    AUX_SYS,
                )
                bst(fx.Int32(e8m0).to(fx.Int8), r_part, prow + i32(H) + gg, 0, AUX_SYS)
        # the remote part stores overlap the next chunk's cshuffle (gemm2_scatter drains
        # them once at the end, before the arrives); the cptr (LDS) reads must finish
        # before the next chunk's cshuffle overwrites the C-tile.
        wait_lgkm0()
        cta_barrier(L, tid)

    def _epilogue_g2_combine(L, tid, acc, n_tile, cptr, a, e, rows, mis):
        """Fused-combine GEMM2 epilogue, per-mi (CROWS=16-row) chunked: for each mi,
        cshuffle that mi's accumulators into the 16-row C-tile, then
        ``_scatter_combine`` sends those rows' output (per-8 fp8 + E8M0) straight to each
        source ``part`` -- the cross-rank P2P overlaps the next mi's cshuffle.  ``mis``: the
        16-row chunks to emit."""
        wave = tid // i32(64)
        lane = tid % i32(64)
        l16 = lane % i32(16)
        ld4 = (lane // i32(16)) * i32(4)
        cbase = wave * i32(out_cols // NW)
        for mi in mis:
            for ni in range_constexpr(NUM_ACC_N2):
                v4 = fx.Vector(acc[mi * NUM_ACC_N2 + ni])
                col = cbase + i32(ni * 16) + l16
                for ii in range_constexpr(4):
                    lrow = ld4 + i32(ii)  # local row 0..15 within this mi chunk
                    idx = lrow * i32(out_cols) + col
                    ptr = fx.add_offset(cptr, fx.make_int_tuple(idx))
                    fx.ptr_store(fx.Vector.from_elements([v4[ii]], fx.Float32), ptr)
            cta_barrier(L, tid)
            _scatter_combine(L, tid, a, e, rows, n_tile, cptr, i32(mi * 16))

    @flyc.jit
    def _epilogue_g2_combine_rows(L, tid, acc, n_tile, cptr, a, e, rows):
        """``_epilogue_g2_combine`` minus the chunks past the echunk's ``rows`` valid rows
        (no cshuffle / cta_barrier pair for an all-padding chunk)."""
        _epilogue_g2_combine(L, tid, acc, n_tile, cptr, a, e, rows, [0])
        for mi in range_constexpr(1, M_REPEAT):
            if rows > i32(mi * 16):
                _epilogue_g2_combine(L, tid, acc, n_tile, cptr, a, e, rows, [mi])

    # ============================ inline-A GEMM2 ==================================
    def _gemm2_core(L, tid, a, expert, e, epi, pre_epi=None):
        """Inline-A GEMM2 core: the compute waves self-stage the A=inter K-tiles into the
        A_BANKS-bank LDS ring.  ``epi(acc, nt, cptr)`` consumes each n-tile (the
        fused-combine scatter).  Returns ``_run_ntile``; the @flyc.jit wrapper drives the
        dynamic n-tile loop (a plain def cannot host a runtime ``range`` over dynamic
        bounds).
        """
        row0 = _inter_row0(e)
        inter_tensor = _global_byte_tensor(_inter_at(a, e), INTER_NBYTES)
        a_gather = ATileLoader(
            row_bytes=inter_row_bytes,
            sort_block_m=TILE_M,
            k_step_bytes=A_K_STEP,
            total_threads=NT,
            swizzle=True,
            x_tensor=inter_tensor,
            async_copy=False,
            indexed_input=False,
            a_dtype=a_dtype,
        )
        w_addr = fx.Int64(a["w2"]) + fx.Int64(expert) * fx.Int64(H * (I // 2))
        s_addr = fx.Int64(a["w2_scale"]) + fx.Int64(expert) * fx.Int64(H * (I // 32))
        a_scale_rsrc = ptr_buf_tensor(_inter_scale_at(a, e), fx.Int32, unit_elems=4)
        a_scale = AScaleLoader(
            scale_rsrc=a_scale_rsrc,
            m_repeat=M_REPEAT,
            model_dim=I,
            sort_block_m=TILE_M,
            total_threads=NT,
            indexed_input=False,
        )
        mfma_gu = MfmaScaleGU(m_repeat=M_REPEAT, num_acc_n=NUM_ACC_N2, a_dtype=a_dtype)
        a_buf = _LdsView(_lds_byte_ptr(L, LDS_A_RING))
        a_scale_lds = _LdsView(_lds_byte_ptr(L, LDS_A_SCALE))
        cptr = _lds_f32_ptr(L, LDS_C_TILE)
        wave_id = tid // i32(64)

        a_gather.for_tile(row0)
        gpu.barrier()
        a_scale.stage(a_scale_lds, row0)
        gpu.barrier()

        def _bank(j):
            return uni((i32(j) % i32(A_BANKS)) * i32(A_LDS_I32))

        def _run_ntile(nt):
            b_row = wave_id * i32(n_per_wave2) + nt * i32(TILE_N2)
            a_fold = _ALdsReader(a_buf.ptr, k_step_bytes=A_K_STEP, a_dtype=a_dtype)
            b_fold = _WeightLoader(
                w_addr, s_addr, b_row, num_acc_n=NUM_ACC_N2, model_dim=I, aux=WEIGHT_AUX
            )

            for j in range_constexpr(A_AHEAD):
                a_gather.store(
                    a_buf, a_gather.load_regs(i32(j * A_K_STEP)), _bank(i32(j))
                )
            regs = [
                a_gather.load_regs(i32((A_AHEAD + j) % K_ITERS2 * A_K_STEP))
                for j in range(A_AHEAD)
            ]
            b_nx, sb_nx = b_fold.load_step(0)
            wait_lgkm0()
            gpu.barrier()
            acc = [mfma_gu.zero_value for _ in range(N_ACC2)]
            for sp in range_constexpr(K_ITERS2):
                b_cur, sb = b_nx, sb_nx
                sa = a_scale.load_step(a_scale_lds, i32(sp))

                def _load_a(mi, ks, _bank_off=(sp % A_BANKS) * A_LDS_I32 * 4):
                    return a_fold.load_operand(mi, ks, _bank_off)

                a_gather.store(a_buf, regs[0], _bank(i32(sp + A_AHEAD)))
                regs = regs[1:] + [
                    a_gather.load_regs(i32(((sp + 2 * A_AHEAD) % K_ITERS2) * A_K_STEP))
                ]
                if const_expr(sp + 1 < K_ITERS2):
                    b_nx, sb_nx = b_fold.load_step(sp + 1)
                acc = _mfma_step(
                    mfma_gu, _load_a, b_cur, acc, sa, sb, M_REPEAT, NUM_ACC_N2
                )
                if const_expr(sp % A_AHEAD == A_AHEAD - 1 or sp == K_ITERS2 - 1):
                    wait_lgkm0()
                    gpu.barrier()
            if const_expr(pre_epi is not None):
                pre_epi()
            epi(acc, nt, cptr)

        return _run_ntile

    @flyc.jit
    def gemm2_scatter(L, tid, a, expert, e, rows, nt_lo, nt_hi, pre_epi=None):
        """GEMM2 over n-tiles [nt_lo, nt_hi): inline-A, scatter each row's output slice
        (per-8 fp8 + E8M0) straight to its source ``part``, then drain the stores."""
        run_ntile = _gemm2_core(
            L,
            tid,
            a,
            expert,
            e,
            lambda acc, nt, cptr: _epilogue_g2_combine_rows(
                L, tid, acc, nt, cptr, a, e, rows
            ),
            pre_epi,
        )
        for nt_ in range(nt_lo, nt_hi, i32(1)):
            run_ntile(i32(nt_))
        wait_vm(0)

    @flyc.jit
    def combine_arrive_rows(tid, a, e, rows, ntiles):
        """Signal each of echunk ``e``'s ``rows`` rows' source that ``ntiles`` n-tiles of
        its part landed (the stores must already be drained): one arrive per (wave, source
        rank), the rows' sources ballot-counted (a per-row arrive puts tens of thousands of
        same-address remote atomics on each source's counter under a hot expert)."""
        parity = a["epoch"] & i32(1)
        mt = a["mtpr"]
        if tid < i32(TILE_M):
            k = fx.min(tid, fx.max(rows - i32(1), i32(0)))
            idx = fx.Int32(
                bld(
                    rsrc(a["chunk_recv_slot"]),
                    (e * i32(TILE_M) + k) * i32(4),
                    0,
                    T.i32,
                    0,
                )
            )
            src = (tid < rows).select((idx // i32(TOPK)) // mt, i32(-1))
            for p in range_constexpr(N_RANKS):
                bal = fx.Int64(rocdl.ballot(T.i64, src == i32(p)))
                cnt = _ctpop(i32(bal & fx.Int64(0xFFFFFFFF))) + _ctpop(
                    i32(bal >> fx.Int64(32))
                )
                if (tid % i32(64) == i32(0)) & (cnt > i32(0)):
                    combine_arrive(a, i32(p), parity, cnt * ntiles)

    return locals()

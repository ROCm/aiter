# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Standalone single-GPU FlyDSL AttnRes-read kernel (SILOTIGER-1277 followup:
# attnres_rewrite). This is the register-streaming + DPP rewrite of the hot
# AttnRes read path, ported from the DEPLOYED HIP reference
#   sglang kevin44988: python/sglang/kernels/jit/csrc/kimi_k3/attn_res/
#       read_hip.cuh  (AttnResRead<kNvb>::finish -- register tile + DPP + lds_barrier)
#       common_hip.cuh (wave_sum via DPP butterflies + readlane)
#       comm/ll_hip.cuh (lds_barrier = s_waitcnt lgkmcnt(0) + s_barrier)
# OSS attribution: sgl-project/sglang #44985 / #44988 (Kevin Mi).
#
# WHY this exists (from wavescope.md + kevin_ideas_mined.md): our Triton AttnRes
# read is ~71% WAIT -- P1 bank-HBM load (vmcnt) + P2 softmax-reduction barriers
# (tl.sum stages every reduction through LDS with a full s_barrier + s_waitcnt
# that *also drains vmcnt*). Triton cannot express DPP cross-lane reductions nor
# an lgkmcnt-only barrier; FlyDSL can. This kernel:
#   * STREAMS the nvb bank rows through registers (load row -> score -> online-
#     softmax accumulate -> discard), one atom per thread, no whole-bank LDS;
#   * reduces each row's dot/sumsq with a DPP wave_sum (fx.coop warp_reduce,
#     zero LDS crossbar) across the wave, then a *tiny* (n_waves floats) LDS
#     cross-wave fold scored once-per-row on wave 0 (read_hip.cuh's structure);
#   * crosses waves with lds_barrier() -- lgkmcnt(0)+s_barrier, NOT the Triton
#     full barrier that drains vmcnt.
#
# It is a drop-in math-equivalent of attn_res_hip._agg_kernel (the deployed
# Triton HIP agg) and of the FlyDSL scores-only mix, GATED <=1 ulp vs the fp32
# torch reference. Opt-in only: nothing imports it on the serving path.

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, ReductionOp, T
from flydsl.extension.coop.warp.rocdl import warp_reduce

from aiter.ops.flydsl.kernels.comm.transport import (
    _buffer_load,
    _buffer_ptr,
    _buffer_store,
    _to_sgpr_i64,
    atom_bf16_to_f32,
    atom_f32_to_bf16,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64

ATOM_ELEMS = 8        # bf16 per thread: one 16 B vector of the row
ATOM_I32 = ATOM_ELEMS // 2
WAVE = 64
MAX_NVB = 8


def _dpp_sum(v):
    """Wave-wide sum of a scalar fp32, uniform in every lane, via DPP butterflies
    + readlane (common_hip.cuh wave_sum). No LDS crossbar: this is the P2 lever."""
    return warp_reduce(v, ReductionOp.ADD, width=WAVE)


def _lds_barrier():
    """lgkmcnt(0) + s_barrier -- ll_hip.cuh lds_barrier. Drains only the LDS
    counter (the per-wave partials just stored), NOT vmcnt, so it never waits on
    the bank/cw/ow HBM loads the way Triton's full s_barrier + s_waitcnt(0) does."""
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()


def build_attn_res_read_dpp(hidden: int, nvb: int, *, out_norm: bool = True,
                            write_bank: bool = False):
    """Build the standalone FlyDSL AttnRes-read kernel for (hidden, nvb).

    One workgroup per token; block = hidden/ATOM_ELEMS threads, each owning one
    16 B atom of the row. nvb is a build constant (bank rows unrolled)."""
    hidden = int(hidden)
    nvb = int(nvb)
    assert hidden % ATOM_ELEMS == 0, "hidden must be a whole number of 16 B atoms"
    assert 1 <= nvb <= MAX_NVB
    BLOCK = hidden // ATOM_ELEMS          # threads per workgroup (= kThreads)
    assert BLOCK % WAVE == 0, "block must be whole waves"
    N_WAVES = BLOCK // WAVE
    N_CAND = nvb + 1                      # bank rows + prefix
    assert N_CAND <= WAVE, "candidates must fit one wave for the scoring lanes"
    INV_H = 1.0 / hidden

    # LDS: per-wave partials of each candidate's (dot, sumsq) + published softmax
    # weights + per-wave out-norm sumsq. This is O(n_waves * n_cand) floats, NOT
    # the whole bank -- the bank stays in registers.
    N_PART = N_WAVES * (2 * N_CAND)
    N_W = N_CAND + 1   # +1 trash slot for inactive scoring lanes
    N_SQ = N_WAVES

    @fx.struct
    class SharedStorage:
        part: fx.Array[fx.Float32, N_PART, 16]
        weight: fx.Array[fx.Float32, max(1, N_W), 16]
        sq: fx.Array[fx.Float32, max(1, N_SQ), 16]

    @flyc.kernel
    def kern(
        prefix_ptr: fx.Tensor,   # [T, H] bf16 -- prefix row, already prefix_in+addend
        bank_ptr: fx.Tensor,     # [T, NB, H] bf16
        cw_ptr: fx.Tensor,       # [H] fp32 score weight
        ow_ptr: fx.Tensor,       # [H] fp32 out-norm weight (unread if not out_norm)
        out_ptr: fx.Tensor,      # [T, H] bf16
        score_eps: fx.Float32,
        out_eps: fx.Float32,
        bank_stride_t: Int32,    # elements between tokens in bank (= NB*H)
        bank_stride_b: Int32,    # elements between bank rows (= H)
    ):
        tid = fx.thread_idx.x
        t = fx.block_idx.x
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)

        sh = fx.SharedAllocator().allocate(SharedStorage).peek()
        part = sh.part.ptr
        wbuf = sh.weight.ptr
        sqbuf = sh.sq.ptr

        prefix_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(prefix_ptr)), T.i32, 16)
        bank_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(bank_ptr)), T.i32, 16)
        cw_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(cw_ptr)), T.f32, 16)
        ow_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(ow_ptr)), T.f32, 16)
        out_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(out_ptr)), T.i32, 16)

        col_i32 = tid * fx.Int32(ATOM_I32)      # i32 offset of this thread's atom
        col_f32 = tid * fx.Int32(ATOM_ELEMS)    # f32 offset

        # cw (fp32) for this thread's 8 columns: two 16 B loads.
        cw_lo = _buffer_load(cw_buf, col_f32, 4, fx.Float32)
        cw_hi = _buffer_load(cw_buf, col_f32 + fx.Int32(4), 4, fx.Float32)
        cw = fx.Vector.from_elements(
            [cw_lo[e] for e in range_constexpr(4)]
            + [cw_hi[e] for e in range_constexpr(4)],
            fx.Float32,
        )

        # --- prefix row (score candidate NVB), loaded up front ---
        p_i32 = t * fx.Int32(hidden // 2) + col_i32
        prefix_atom = _buffer_load(prefix_buf, p_i32, ATOM_I32, fx.Int32)
        pv = atom_bf16_to_f32(prefix_atom)

        # --- bank rows: STREAM through registers. Keep each row's fp32 atom live
        # for the mix (register tile, read_hip.cuh); score it with DPP. ---
        bank_base = t * fx.Int32(bank_stride_t)
        bank_fp32 = []
        for r in range_constexpr(nvb):
            row_i32 = bank_base + fx.Int32(r) * fx.Int32(bank_stride_b)
            atom = _buffer_load(bank_buf, (row_i32 // fx.Int32(2)) + col_i32,
                                ATOM_I32, fx.Int32)
            f = atom_bf16_to_f32(atom)
            bank_fp32.append(f)
            dot = fx.Float32((f * cw).reduce(ReductionOp.ADD, fastmath="reassoc"))
            sq = fx.Float32((f * f).reduce(ReductionOp.ADD, fastmath="reassoc"))
            dot = _dpp_sum(dot)
            sq = _dpp_sum(sq)
            # After _dpp_sum every lane of the wave holds the same value, so all
            # 64 lanes write the same slot the same value (block_reduce_add's
            # trick) -- no predicate, no race on the winner.
            fx.ptr_store(dot, part + fx.Int64(fx.Int32(r * 2 * N_WAVES) + wave))
            fx.ptr_store(sq, part + fx.Int64(fx.Int32((r * 2 + 1) * N_WAVES) + wave))

        # prefix candidate's dot/sumsq, same DPP path.
        d_p = fx.Float32((pv * cw).reduce(ReductionOp.ADD, fastmath="reassoc"))
        s_p = fx.Float32((pv * pv).reduce(ReductionOp.ADD, fastmath="reassoc"))
        d_p = _dpp_sum(d_p)
        s_p = _dpp_sum(s_p)
        fx.ptr_store(d_p, part + fx.Int64(fx.Int32(nvb * 2 * N_WAVES) + wave))
        fx.ptr_store(s_p, part + fx.Int64(fx.Int32((nvb * 2 + 1) * N_WAVES) + wave))

        _lds_barrier()

        # --- KEVIN-WAY scoring: ONLY wave 0 scores the candidates & publishes
        # the softmax weights (read_hip.cuh finish: `if (wave == 0)`). The
        # transcendentals run once per ROW (on a single wave), not redundantly
        # on all N_WAVES waves as our original does. The softmax max & denom use
        # a 16-lane group reduction (candidates sit on lanes 0..N_CAND-1, one
        # 16-lane row): 4 shuffle_xor steps (offset 8,4,2,1) vs our 6-step
        # full-wave tree -- Kevin's exact reduction order. ---
        def _score_wave0():
            live = lane < fx.Int32(N_CAND)
            dot_c = fx.Float32(0.0)
            sq_c = fx.Float32(0.0)
            for w in range_constexpr(N_WAVES):
                didx = lane * fx.Int32(2 * N_WAVES) + fx.Int32(w)
                sidx = lane * fx.Int32(2 * N_WAVES) + fx.Int32(N_WAVES + w)
                safe_d = live.select(didx, fx.Int32(0))
                safe_s = live.select(sidx, fx.Int32(0))
                dot_c = dot_c + fx.Float32(part[safe_d])
                sq_c = sq_c + fx.Float32(part[safe_s])
            with fx.fastmath("afn"):
                score = dot_c * fmath.rsqrt(sq_c * INV_H + score_eps)
            neg_inf = fx.Float32(float("-inf"))
            score = live.select(score, neg_inf)
            # 16-lane max: offsets 8,4,2,1 (xor stays within the 16-lane row).
            m = score
            for off in (8, 4, 2, 1):
                moved = m.shuffle_xor(off, WAVE)
                m = (moved > m).select(moved, m)
            e = live.select(fmath.exp(score - m), fx.Float32(0.0))
            # 16-lane denom sum, same butterfly order (read_hip.cuh).
            denom = e
            for off in (8, 4, 2, 1):
                denom = denom + denom.shuffle_xor(off, WAVE)
            wgt = e * (fx.Float32(1.0) / denom)
            idx = live.select(lane, fx.Int32(N_CAND))
            fx.ptr_store(wgt, wbuf + fx.Int64(idx))

        if wave == fx.Int32(0):
            _score_wave0()
        _lds_barrier()

        # --- mix: weighted sum of the register-held bank rows + prefix, read the
        # weights back from LDS (one uniform load per candidate). ---
        wp = fx.Float32(wbuf[fx.Int32(nvb)])
        mix = pv * wp
        for r in range_constexpr(nvb):
            wr = fx.Float32(wbuf[fx.Int32(r)])
            mix = mix + bank_fp32[r] * wr

        if const_expr(out_norm):
            local = fx.Float32((mix * mix).reduce(ReductionOp.ADD, fastmath="reassoc"))
            local = _dpp_sum(local)
            fx.ptr_store(local, sqbuf + fx.Int64(wave))
            _lds_barrier()
            total = fx.Float32(0.0)
            for w in range_constexpr(N_WAVES):
                total = total + fx.Float32(sqbuf[fx.Int32(w)])
            with fx.fastmath("afn"):
                scale = fx.Float32(1.0) / fmath.sqrt(total * INV_H + out_eps)
            ow_lo = _buffer_load(ow_buf, col_f32, 4, fx.Float32)
            ow_hi = _buffer_load(ow_buf, col_f32 + fx.Int32(4), 4, fx.Float32)
            ow = fx.Vector.from_elements(
                [ow_lo[e] for e in range_constexpr(4)]
                + [ow_hi[e] for e in range_constexpr(4)],
                fx.Float32,
            )
            out_f = mix * scale * ow
        else:
            out_f = mix

        out_atom = atom_f32_to_bf16(out_f)
        _buffer_store(out_buf, p_i32, out_atom)

    @flyc.jit
    def launch(
        prefix_ptr: fx.Tensor,
        bank_ptr: fx.Tensor,
        cw_ptr: fx.Tensor,
        ow_ptr: fx.Tensor,
        out_ptr: fx.Tensor,
        score_eps: fx.Float32,
        out_eps: fx.Float32,
        bank_stride_t: Int32,
        bank_stride_b: Int32,
        num_tokens: Int32,
        stream: fx.Stream,
    ):
        kern(
            prefix_ptr, bank_ptr, cw_ptr, ow_ptr, out_ptr,
            score_eps, out_eps, bank_stride_t, bank_stride_b,
        ).launch(grid=(fx.Int64(num_tokens), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch._block = BLOCK
    launch._nvb = nvb
    launch._hidden = hidden
    launch._out_norm = bool(out_norm)
    return launch


_CACHE = {}


def attn_res_read_dpp(prefix, bank, cw, ow, nvb, score_eps, out_eps, *, out=None):
    """Standalone FlyDSL register-streaming + DPP AttnRes read.

    prefix : [T, H] bf16  (= prefix_in + addend, already materialized)
    bank   : [T, NB, H] bf16 (rows 0..nvb-1 aggregated)
    cw     : [H] fp32 score weight
    ow     : [H] fp32 out-norm weight, or None for the pre-norm mix
    Returns out [T, H] bf16."""
    T_, H = prefix.shape
    NB = bank.shape[1]
    out_norm = ow is not None
    if out is None:
        out = torch.empty_like(prefix)
    key = (H, int(nvb), out_norm)
    launch = _CACHE.get(key)
    if launch is None:
        launch = build_attn_res_read_dpp(H, int(nvb), out_norm=out_norm)
        _CACHE[key] = launch
    cwf = cw if cw.dtype == torch.float32 else cw.float()
    owf = ow.float() if (ow is not None and ow.dtype != torch.float32) else (
        ow if ow is not None else cwf)
    stream = torch.cuda.current_stream().cuda_stream
    launch(
        prefix, bank, cwf, owf, out,
        float(score_eps), float(out_eps),
        int(NB * H), int(H), int(T_),
        fx.Stream(stream),
    )
    return out

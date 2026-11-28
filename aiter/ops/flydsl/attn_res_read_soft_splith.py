# SPDX-License-Identifier: MIT
# SOFT split-H AttnRes read (read-only). Parametrized by an ARBITRARY number of
# workgroups (nwg, 1..256) and waves-per-WG (nwaves, 1..16), INDEPENDENTLY, so the
# geometry can be swept to find the fusion-favourable point (the AR geometry at
# M=1 is 7 WG x 2 waves). Unlike attn_res_read_dpp_kw_splith (K in {1,2,7,14}),
# the split is atom-granular: H/8 = 896 atoms are spread as evenly as possible over
# nwg workgroups (uneven tail allowed), and each lane walks its WG's atoms in a
# grid-stride loop with masking. Keeps the 8-wide vectorized bank load.
import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import math as fmath
from flydsl.expr import range_constexpr, rocdl
from flydsl.expr.typing import Int32, ReductionOp, T
from flydsl.extension.coop.warp.rocdl import warp_reduce

from aiter.ops.flydsl.kernels import communication_ops_utils as C
from aiter.ops.flydsl.kernels.comm.transport import (
    _buffer_load,
    _buffer_ptr,
    _buffer_store,
    _to_sgpr_i64,
    atom_bf16_to_f32,
    atom_f32_to_bf16,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_base_i64

ATOM_ELEMS = 8
ATOM_I32 = ATOM_ELEMS // 2
WAVE = 64
MAX_NVB = 8


def _dpp_sum(v):
    return warp_reduce(v, ReductionOp.ADD, width=WAVE)


def _lds_barrier():
    rocdl.s_waitcnt(lgkmcnt=0)
    rocdl.s_barrier()


def build_soft_splith(hidden, nvb, nwg, nwaves, num_tokens):
    hidden = int(hidden)
    nvb = int(nvb)
    NWG = int(nwg)
    NWAVES = int(nwaves)
    M = int(num_tokens)
    assert hidden % ATOM_ELEMS == 0
    A_TOTAL = hidden // ATOM_ELEMS  # 896 atoms
    assert 1 <= NWG <= A_TOTAL
    assert 1 <= NWAVES <= 16
    BLOCK = NWAVES * WAVE  # lanes per WG
    base = A_TOTAL // NWG  # atoms per WG (floor)
    rem = A_TOTAL % NWG  # first `rem` WGs get base+1
    MAX_ATOMS = base + (1 if rem else 0)  # ceil atoms/WG
    MAXIT = (MAX_ATOMS + BLOCK - 1) // BLOCK  # max atoms a lane walks
    assert MAX_ATOMS >= 1, "a WG would own 0 atoms; reduce nwg"
    N_CAND = nvb + 1
    assert N_CAND <= WAVE
    INV_H = 1.0 / hidden
    NPC = 2 * N_CAND
    TOTWG = M * NWG

    N_PART = NWAVES * NPC
    N_W = N_CAND + 1

    @fx.struct
    class Shared:
        part: fx.Array[fx.Float32, N_PART, 16]
        weight: fx.Array[fx.Float32, max(1, N_W), 16]

    @flyc.kernel
    def kern(
        prefix_ptr: fx.Tensor,
        bank_ptr: fx.Tensor,
        cw_ptr: fx.Tensor,
        out_ptr: fx.Tensor,
        gpart_ptr: fx.Tensor,
        counter_ptr: fx.Tensor,
        score_eps: fx.Float32,
        bank_stride_t: Int32,
        bank_stride_b: Int32,
    ):
        tid = fx.thread_idx.x
        bidx = fx.block_idx.x
        g = bidx % fx.Int32(NWG)  # which H-slice / workgroup
        t = bidx // fx.Int32(NWG)  # token
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)

        # per-WG atom range (runtime): first `rem` WGs own base+1 atoms.
        extra = (g < fx.Int32(rem)).select(fx.Int32(1), fx.Int32(0))
        cnt = fx.Int32(base) + extra  # atoms this WG owns
        gmin = (g < fx.Int32(rem)).select(g, fx.Int32(rem))  # min(g, rem)
        start = g * fx.Int32(base) + gmin  # first atom index of this WG

        sh = fx.SharedAllocator().allocate(Shared).peek()
        part = sh.part.ptr
        wbuf = sh.weight.ptr

        prefix_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(prefix_ptr)), T.i32, 16)
        bank_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(bank_ptr)), T.i32, 16)
        cw_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(cw_ptr)), T.f32, 16)
        out_buf = _buffer_ptr(_to_sgpr_i64(buf_base_i64(out_ptr)), T.i32, 16)
        gpart_addr = buf_base_i64(gpart_ptr)
        counter_addr = buf_base_i64(counter_ptr)
        gfp = C._to_ptr_global(gpart_addr, fx.Float32, 4)

        bank_base = t * fx.Int32(bank_stride_t)

        # per-lane accumulators over its atoms, per candidate (nvb bank + prefix)
        acc_dot = [fx.Float32(0.0) for _ in range_constexpr(N_CAND)]
        acc_sq = [fx.Float32(0.0) for _ in range_constexpr(N_CAND)]
        # keep each atom's values for the mix step (no HBM reread)
        pv_keep = []
        bank_keep = []  # bank_keep[it][r]
        col_keep = []  # i32 col per iter (for the store)
        valid_keep = []

        for it in range_constexpr(MAXIT):
            la = tid + fx.Int32(
                it * BLOCK
            )  # local atom index (full block, not per-wave)
            valid = la < cnt
            atom = start + la
            safe_atom = valid.select(atom, fx.Int32(0))
            col_i32 = safe_atom * fx.Int32(ATOM_I32)
            col_f32 = safe_atom * fx.Int32(ATOM_ELEMS)
            col_keep.append(col_i32)
            valid_keep.append(valid)

            cw_lo = _buffer_load(cw_buf, col_f32, 4, fx.Float32)
            cw_hi = _buffer_load(cw_buf, col_f32 + fx.Int32(4), 4, fx.Float32)
            cw = fx.Vector.from_elements(
                [cw_lo[e] for e in range_constexpr(4)]
                + [cw_hi[e] for e in range_constexpr(4)],
                fx.Float32,
            )

            p_i32 = t * fx.Int32(hidden // 2) + col_i32
            pv = atom_bf16_to_f32(_buffer_load(prefix_buf, p_i32, ATOM_I32, fx.Int32))
            pv_keep.append(pv)

            row_banks = []
            for r in range_constexpr(nvb):
                row_i32 = bank_base + fx.Int32(r) * fx.Int32(bank_stride_b)
                f = atom_bf16_to_f32(
                    _buffer_load(
                        bank_buf, (row_i32 // fx.Int32(2)) + col_i32, ATOM_I32, fx.Int32
                    )
                )
                row_banks.append(f)
                pdot = fx.Float32((f * cw).reduce(ReductionOp.ADD, fastmath="reassoc"))
                psq = fx.Float32((f * f).reduce(ReductionOp.ADD, fastmath="reassoc"))
                acc_dot[r] = acc_dot[r] + valid.select(pdot, fx.Float32(0.0))
                acc_sq[r] = acc_sq[r] + valid.select(psq, fx.Float32(0.0))
            bank_keep.append(row_banks)
            pdp = fx.Float32((pv * cw).reduce(ReductionOp.ADD, fastmath="reassoc"))
            psp = fx.Float32((pv * pv).reduce(ReductionOp.ADD, fastmath="reassoc"))
            acc_dot[nvb] = acc_dot[nvb] + valid.select(pdp, fx.Float32(0.0))
            acc_sq[nvb] = acc_sq[nvb] + valid.select(psp, fx.Float32(0.0))

        # DPP-sum each candidate across the 64 lanes of this wave -> per-wave partial
        for cand in range_constexpr(N_CAND):
            dot = _dpp_sum(acc_dot[cand])
            sq = _dpp_sum(acc_sq[cand])
            fx.ptr_store(dot, part + fx.Int64(fx.Int32(cand * 2 * NWAVES) + wave))
            fx.ptr_store(sq, part + fx.Int64(fx.Int32((cand * 2 + 1) * NWAVES) + wave))
        _lds_barrier()

        # wave0 folds this WG's NWAVES partials -> per-WG dot/sq, write own gpart slot
        base_slot = (t * fx.Int32(NWG) + g) * fx.Int32(NPC)
        if wave == fx.Int32(0):
            live = lane < fx.Int32(N_CAND)
            dot_c = fx.Float32(0.0)
            sq_c = fx.Float32(0.0)
            for w in range_constexpr(NWAVES):
                di = lane * fx.Int32(2 * NWAVES) + fx.Int32(w)
                si = lane * fx.Int32(2 * NWAVES) + fx.Int32(NWAVES + w)
                dot_c = dot_c + fx.Float32(part[live.select(di, fx.Int32(0))])
                sq_c = sq_c + fx.Float32(part[live.select(si, fx.Int32(0))])
            if live:
                fx.ptr_store(dot_c, gfp + fx.Int64(base_slot + lane * fx.Int32(2)))
                fx.ptr_store(
                    sq_c, gfp + fx.Int64(base_slot + lane * fx.Int32(2) + fx.Int32(1))
                )

        # ---- GRID BARRIER across all TOTWG workgroups ----
        if tid == fx.Int32(0):
            C.fence_agent_release()
            a = fx.Int32(C.atomic_add_agent_one_as(counter_addr, 1))
            target = ((a // fx.Int32(TOTWG)) + fx.Int32(1)) * fx.Int32(TOTWG)
            C.spin_until_ge_i32_agent(counter_addr, target)
            C.fence_agent_acquire()
        _lds_barrier()

        # wave0: sum all NWG per-WG partials -> full score, softmax
        if wave == fx.Int32(0):
            live = lane < fx.Int32(N_CAND)
            fdot = fx.Float32(0.0)
            fsq = fx.Float32(0.0)
            tok_base = t * fx.Int32(NWG) * fx.Int32(NPC)
            for s in range_constexpr(NWG):
                off = tok_base + fx.Int32(s * NPC) + lane * fx.Int32(2)
                off = live.select(off, fx.Int32(0))
                fdot = fdot + fx.Float32(fx.ptr_load(gfp + fx.Int64(off)))
                fsq = fsq + fx.Float32(fx.ptr_load(gfp + fx.Int64(off + fx.Int32(1))))
            with fx.fastmath("afn"):
                score = fdot * fmath.rsqrt(fsq * INV_H + score_eps)
            neg_inf = fx.Float32(float("-inf"))
            score = live.select(score, neg_inf)
            m = score
            for o in (8, 4, 2, 1):
                mv = m.shuffle_xor(o, WAVE)
                m = (mv > m).select(mv, m)
            e = live.select(fmath.exp(score - m), fx.Float32(0.0))
            denom = e
            for o in (8, 4, 2, 1):
                denom = denom + denom.shuffle_xor(o, WAVE)
            wgt = e * (fx.Float32(1.0) / denom)
            idx = live.select(lane, fx.Int32(N_CAND))
            fx.ptr_store(wgt, wbuf + fx.Int64(idx))
        _lds_barrier()

        # mix each of this lane's atoms (bank + prefix kept in registers), store
        wp = fx.Float32(wbuf[fx.Int32(nvb)])
        for it in range_constexpr(MAXIT):
            mix = pv_keep[it] * wp
            for r in range_constexpr(nvb):
                mix = mix + bank_keep[it][r] * fx.Float32(wbuf[fx.Int32(r)])
            p_i32 = t * fx.Int32(hidden // 2) + col_keep[it]
            if valid_keep[it]:
                _buffer_store(out_buf, p_i32, atom_f32_to_bf16(mix))

    @flyc.jit
    def launch(
        prefix_ptr,
        bank_ptr,
        cw_ptr,
        out_ptr,
        gpart_ptr,
        counter_ptr,
        score_eps: fx.Float32,
        bank_stride_t: Int32,
        bank_stride_b: Int32,
        stream: fx.Stream,
    ):
        kern(
            prefix_ptr,
            bank_ptr,
            cw_ptr,
            out_ptr,
            gpart_ptr,
            counter_ptr,
            score_eps,
            bank_stride_t,
            bank_stride_b,
        ).launch(grid=(fx.Int64(TOTWG), 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch._nwg = NWG
    launch._nwaves = NWAVES
    launch._block = BLOCK
    launch._maxit = MAXIT
    return launch


_CACHE = {}
_SCRATCH = {}


def attn_res_read_soft_splith(
    prefix, bank, cw, nvb, nwg, nwaves, score_eps, *, out=None
):
    T_, H = prefix.shape
    NB = bank.shape[1]
    if out is None:
        out = torch.empty_like(prefix)
    key = (H, int(nvb), int(nwg), int(nwaves), int(T_))
    launch = _CACHE.get(key)
    if launch is None:
        launch = build_soft_splith(H, int(nvb), int(nwg), int(nwaves), int(T_))
        _CACHE[key] = launch
    sk = (int(T_), int(nwg), int(nvb))
    sc = _SCRATCH.get(sk)
    if sc is None:
        gpart = torch.zeros(
            T_ * int(nwg) * 2 * (int(nvb) + 1),
            dtype=torch.float32,
            device=prefix.device,
        )
        counter = torch.zeros(1, dtype=torch.int32, device=prefix.device)
        _SCRATCH[sk] = sc = (gpart, counter)
    gpart, counter = sc
    cwf = cw if cw.dtype == torch.float32 else cw.float()
    stream = torch.cuda.current_stream().cuda_stream
    launch(
        prefix,
        bank,
        cwf,
        out,
        gpart,
        counter,
        float(score_eps),
        int(NB * H),
        int(H),
        fx.Stream(stream),
    )
    return out

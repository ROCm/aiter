# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""M8192 Kimi-K3 EP32 GEMM1: full BM160 with twelve waves."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw

from .mxfp4_gemm2_ep32 import lds_slot_scopes
from .mxfp4_gemm_common import (
    _buffer_rsrc,
    _dpp_umax_step,
    _gep3,
    _e8m0_from_amax,
    _fabs_f32,
    _global_i32_at,
    _inline_dpp_quad_amax,
    _lds_ptr3,
    _scale_mma_atoms,
    _udiv,
    _umod,
    global_typed_ptr,
    bq_bytes_for,
    bscale_bytes_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kbs_stride_n0_dw_for,
    lds_swizzle_mask_f8,
)


def _dpp_row16_amax(a32):
    # Quad xor-1/xor-2, then row_half_mirror and row_mirror span one 16-lane row.
    return _dpp_umax_step(_dpp_umax_step(_inline_dpp_quad_amax(a32), 0x141), 0x140)


def _situv2_mul_batch(gates, ups, beta, linear_beta):
    """SiTUv2 gate * up with c * tanh(x / c) = c - 2c / (1 + exp(2x / c)).

    Equal to the shared helper up to rounding, without the |x| and sign select;
    +-inf exponentials give the exact +-c limits.
    """
    beta, linear_beta = float(beta), float(linear_beta)

    def rcp(x):
        return fx.Float32(rocdl.rcp(T.f32, _raw(x)))

    one = fx.Float32(1.0)
    out = []
    for g, u in zip(gates, ups):
        eg = fx.exp2(g * fx.Float32(2.0 * LOG2E / beta), fastmath="afn")
        es = fx.exp2(g * fx.Float32(-LOG2E), fastmath="afn")
        eu = fx.exp2(u * fx.Float32(2.0 * LOG2E / linear_beta), fastmath="afn")
        tg = fx.Float32(beta) - rcp(eg + one) * fx.Float32(2.0 * beta)
        tu = fx.Float32(linear_beta) - rcp(eu + one) * fx.Float32(2.0 * linear_beta)
        out.append(tg * rcp(es + one) * tu)
    return out


def compile_gemm1_ep8192(*, BN=256, use_nt=True, situ_beta=4.0, situ_linear_beta=25.0):
    if BN != 256:
        raise ValueError("M8192 W12 GEMM1 requires BN256")
    BM, BK, H, INTER, NE = 160, 128, 3584, 3072, 28
    N_OUT, N_BLOCKS, M_CHUNKS, N_REPS = 2 * INTER, 2 * INTER // BN, 4, BN // 64
    A_STAGES = 2
    B_STAGES = 3
    GRID = 256
    AS_DW = kas_per_chunk_dw_for(H)
    OUT_AS_DW = kas_per_chunk_dw_for(INTER)
    BS_DW = kbs_per_expert_dw_for(N_OUT, H)
    BS_N_DW = kbs_stride_n0_dw_for(H)
    A_SLOT, B_SLOT, S_SLOT = BM * BK, BN * BK // 2, (5 + BN // 32) * 256
    B_BASE, S_BASE = A_STAGES * A_SLOT, A_STAGES * A_SLOT + B_STAGES * B_SLOT
    LDS_BYTES = S_BASE + 3 * S_SLOT
    assert LDS_BYTES <= 160 * 1024
    name = f"gemm1_ep8192_a8w4_bm160_bn{BN}_w12_m64x2_m32_as{A_STAGES}_bs{B_STAGES}_ak128_{'nt' if use_nt else 'cached'}_situv2_runtimek_regepi3_earlypro_fastsitu"

    @fx.struct
    class Storage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=name, known_block_size=[768, 1, 1])
    def gemm1(
        aq: fx.Int64,
        asc: fx.Int64,
        bq: fx.Int64,
        bsc: fx.Int64,
        eids: fx.Int64,
        counts: fx.Int64,
        token_rows: fx.Int64,
        tokens: fx.Int32,
        out: fx.Int64,
        out_scale: fx.Int64,
        hidden: fx.Int64,
        bias: fx.Int64,
    ):
        tx = fx.thread_idx.x
        lane = tx % fx.Int32(64)
        # Uniform wave IDs preserve scalar resource-address arithmetic.
        wave = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(tx // fx.Int32(64))))
        wn, wm = wave % fx.Int32(4), wave // fx.Int32(4)
        # Three M groups cover 64 + 64 + 32 complete rows.
        m_chunk_base = wm * fx.Int32(4)
        loader = wm == fx.Int32(0)
        ld16, lm16 = lane // fx.Int32(16), lane % fx.Int32(16)
        ld8, lm8 = lane // fx.Int32(8), lane % fx.Int32(8)
        lds_ptr = fx.SharedAllocator().allocate(Storage).peek().raw.ptr
        lds = fx.Int32(fx.ptrtoint(lds_ptr))
        scopes = lds_slot_scopes(A_STAGES + B_STAGES + 3, "ep8192_g1")
        ac, bc, sc = (
            scopes[:A_STAGES],
            scopes[A_STAGES : A_STAGES + B_STAGES],
            scopes[A_STAGES + B_STAGES :],
        )
        m_blocks = _global_i32_at(counts, fx.Int32(0)) // fx.Int32(BM)
        bound = m_blocks * fx.Int32(N_BLOCKS)
        ares = _buffer_rsrc(aq, fx.Int64(tokens) * H)
        asres = _buffer_rsrc(asc, fx.Int64(m_blocks) * 5 * AS_DW * 4)
        bres = _buffer_rsrc(bq, bq_bytes_for(NE, N_OUT, H))
        bsres = _buffer_rsrc(bsc, bscale_bytes_for(NE, N_OUT, H))
        mma = _scale_mma_atoms("fp8")
        mem4 = fx.make_layout(4, 1)
        # Constant GEP displacements fold into DS offsets instead of
        # keeping one VGPR address for each K-slot and M chunk.
        a_mask = lds_swizzle_mask_f8(lm16, BK)
        a_row_base = (m_chunk_base * 16 + lm16) * BK
        a_ptr_lo = _lds_ptr3(lds, a_row_base + ((ld16 * 16) ^ a_mask))
        a_ptr_hi = _lds_ptr3(lds, a_row_base + ((ld16 * 16 + 64) ^ a_mask))
        b_ptr = _lds_ptr3(lds, wn * (BN * 16) + ld16 * 256 + lm16 * 16)
        s_ptr0 = _lds_ptr3(lds, (m_chunk_base // 2) * 256 + (ld16 * 16 + lm16) * 4)
        s_ptr1 = _lds_ptr3(
            lds, ((m_chunk_base + 1) // 2) * 256 + (ld16 * 16 + lm16) * 4
        )
        # Byte buffer views keep each epilogue store to one VGPR offset.
        qbuf, sbuf = [
            fx.rocdl.make_buffer_tensor(
                fx.Tensor(
                    fx.make_view(
                        global_typed_ptr(addr, T.i8, align=1),
                        fx.make_layout((1, 1), (1, 1)),
                    )
                ),
                max_size=True,
            )
            for addr in (out, out_scale)
        ]
        byte_copy = fx.make_copy_atom(fx.rocdl.BufferCopy8b(), fx.Int8)

        def store_byte(buf, offset, value):
            r = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int8)
            r.store(fx.Vector.from_elements([value], fx.Int8))
            fx.copy(byte_copy, r, buf[None, offset])

        def tile_setup(pid):
            # Returns [ni, mrow, bbase x4, scale_base, a_voff x5] for one tile.
            xq, xr, xc = _udiv(bound, 8), _umod(bound, 8), _umod(pid, 8)
            tile = xc * xq + fx.min(xc, xr) + _udiv(pid, 8)
            mi, ni = tile // fx.Int32(N_BLOCKS), tile % fx.Int32(N_BLOCKS)
            mrow = mi * fx.Int32(BM)
            expert = fx.Int32(
                rocdl.readfirstlane(T.i32, as_ir_value(_global_i32_at(eids, mi)))
            )
            bbase = []
            for j in range_constexpr(N_REPS):
                col = ni * BN + wn * (BN // 4) + j * 16
                # fx.Int32 has no scalar/uniform annotation for buffer bases.
                bbase.append(
                    fx.Int32(
                        rocdl.readfirstlane(
                            T.i32, as_ir_value((expert * N_OUT + col) * (H // 2))
                        )
                    )
                )
            scale_base = (
                expert * BS_DW + (ni * (BN // 32) + wn * (BN // 128)) * BS_N_DW
            ) * 4
            scale_base = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(scale_base)))
            # Reuse route addresses for every K step. Loading row IDs inside
            # load_a forces vmcnt(0) waits between the future A/B DMAs.
            a_voff = []
            for sub in range_constexpr(5):
                row = wn * 40 + sub * 8
                actual = _global_i32_at(token_rows, mrow + row + ld8)
                mask = lds_swizzle_mask_f8(row + ld8, BK)
                a_voff.append(actual * H + ((lm8 * 16) ^ mask))
            return [ni, mrow] + bbase + [scale_base] + a_voff

        def load_a(t, slot, step):
            if loader:
                for sub in range_constexpr(5):
                    row = wn * 40 + sub * 8
                    # Scoped DMA has no equivalent fx.copy alias-scope API.
                    rocdl.raw_ptr_buffer_load_lds(
                        ares,
                        _lds_ptr3(lds, fx.Int32(slot * A_SLOT) + row * BK),
                        fx.Int32(16),
                        t[7 + sub],
                        step * fx.Int32(BK),
                        fx.Int32(0),
                        fx.Int32(0),
                        alias_scopes=ac[slot][0],
                        noalias_scopes=ac[slot][1],
                    )

        def load_b(t, slot, step):
            if loader:
                for j in range_constexpr(N_REPS):
                    dst = fx.Int32(B_BASE + slot * B_SLOT + j * 1024) + wn * (BN * 16)
                    rocdl.raw_ptr_buffer_load_lds(
                        bres,
                        _lds_ptr3(lds, dst),
                        fx.Int32(16),
                        ld16 * 256 + lm16 * 16 + step * fx.Int32(1024),
                        t[2 + j],
                        fx.Int32(0),
                        2 if use_nt else 0,
                        alias_scopes=bc[slot][0],
                        noalias_scopes=bc[slot][1],
                    )

        def load_s(t, slot, word):
            if loader:
                for group in range_constexpr(2):
                    sub = wn + group * 4
                    if sub < fx.Int32(5):
                        src = (t[1] // 32 + sub) * (AS_DW * 4)
                        dst = fx.Int32(S_BASE + slot * S_SLOT) + sub * 256
                        rocdl.raw_ptr_buffer_load_lds(
                            asres,
                            _lds_ptr3(lds, dst),
                            fx.Int32(4),
                            lane * 4,
                            src + word * fx.Int32(256),
                            fx.Int32(0),
                            fx.Int32(0),
                            alias_scopes=sc[slot][0],
                            noalias_scopes=sc[slot][1],
                        )

        def load_bs(t, slot, word):
            if loader:
                for pair in range_constexpr(2):
                    dst = (
                        fx.Int32(S_BASE + slot * S_SLOT + 5 * 256 + pair * 256)
                        + wn * 512
                    )
                    src = t[6] + pair * BS_N_DW * 4
                    rocdl.raw_ptr_buffer_load_lds(
                        bsres,
                        _lds_ptr3(lds, dst),
                        fx.Int32(4),
                        lane * 4,
                        src + word * fx.Int32(256),
                        fx.Int32(0),
                        fx.Int32(0),
                        alias_scopes=sc[slot][0],
                        noalias_scopes=sc[slot][1],
                    )

        def read_a(slot, ci):
            disp = slot * A_SLOT + ci * 16 * BK
            lo = fx.Vector(
                llvm.load(
                    T.vec(4, T.i32),
                    _gep3(a_ptr_lo, disp),
                    alignment=16,
                    alias_scopes=ac[slot][0],
                    noalias_scopes=ac[slot][1],
                )
            )
            hi = fx.Vector(
                llvm.load(
                    T.vec(4, T.i32),
                    _gep3(a_ptr_hi, disp),
                    alignment=16,
                    alias_scopes=ac[slot][0],
                    noalias_scopes=ac[slot][1],
                )
            )
            af = fx.make_rmem_tensor(8, fx.Int32)
            af.store(lo.shuffle(hi, list(range(8))))
            return af

        def prologue(t):
            for p in range_constexpr(A_STAGES - 1):
                load_a(t, p, fx.Int32(p))
            for p in range_constexpr(B_STAGES - 1):
                load_b(t, p, fx.Int32(p))
            load_s(t, 0, fx.Int32(0))
            load_s(t, 1, fx.Int32(1))
            load_bs(t, 0, fx.Int32(0))
            load_bs(t, 1, fx.Int32(1))

        # Each tile's setup and prologue DMAs are issued before the previous
        # tile's epilogue, whose VALU work then hides the pipeline refill.
        wg = fx.Int32(fx.block_idx.x)
        first = tile_setup(fx.min(wg, fx.max(bound - fx.Int32(1), fx.Int32(0))))
        if wg < bound:
            prologue(first)
        for iv, cur in range(wg, bound, fx.Int32(gpu.grid_dim.x), init=first):
            ni, mrow = cur[0], cur[1]
            acc = [
                [fx.make_rmem_tensor(mem4, fx.Float32) for _ in range_constexpr(N_REPS)]
                for _ in range_constexpr(M_CHUNKS)
            ]
            for i in range_constexpr(M_CHUNKS):
                for j in range_constexpr(N_REPS):
                    acc[i][j].store(fx.full(4, 0.0, fx.Float32))
            init_c = [
                acc[ci][j].load()
                for ci in range_constexpr(4)
                for j in range_constexpr(4)
            ]
            for kw, state in range(
                fx.Int32(0), fx.Int32(H // 256), fx.Int32(1), init=init_c
            ):
                word = fx.Int32(kw)
                for ci in range_constexpr(4):
                    for j in range_constexpr(4):
                        acc[ci][j].store(fx.Vector(state[ci * 4 + j]))
                for phase in range_constexpr(3):
                    if (word % fx.Int32(3)) == fx.Int32(phase):
                        for h in range_constexpr(2):
                            slot, slot_b = h, (2 * phase + h) % 3
                            step = word * fx.Int32(2) + fx.Int32(h)
                            rocdl.sched_barrier(0)
                            if const_expr(h == 0):
                                if word == fx.Int32(0):
                                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                                elif word == fx.Int32(H // 256 - 1):
                                    rocdl.s_waitcnt(vmcnt=4, lgkmcnt=0)
                                else:
                                    rocdl.s_waitcnt(vmcnt=7, lgkmcnt=0)
                            else:
                                if word == fx.Int32(H // 256 - 1):
                                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                                else:
                                    rocdl.s_waitcnt(vmcnt=4, lgkmcnt=0)
                            rocdl.s_barrier()
                            rocdl.sched_barrier(0)
                            bf = []
                            for j in range_constexpr(N_REPS):
                                disp = B_BASE + slot_b * B_SLOT + j * 1024
                                r = fx.make_rmem_tensor(mem4, fx.Int32)
                                r.store(
                                    fx.Vector(
                                        llvm.load(
                                            T.vec(4, T.i32),
                                            _gep3(b_ptr, disp),
                                            alignment=16,
                                            alias_scopes=bc[slot_b][0],
                                            noalias_scopes=bc[slot_b][1],
                                        )
                                    )
                                )
                                bf.append(r)
                            if step + fx.Int32(1) < fx.Int32(H // BK):
                                load_a(cur, 1 - h, step + fx.Int32(1))
                            if step + fx.Int32(2) < fx.Int32(H // BK):
                                load_b(cur, (slot_b + 2) % 3, step + fx.Int32(2))
                            bs_current = []
                            for pair in range_constexpr(2):
                                disp = S_BASE + phase * S_SLOT + 5 * 256 + pair * 256
                                ptr = _lds_ptr3(lds, wn * 512 + (ld16 * 16 + lm16) * 4)
                                bs_current.append(
                                    fx.Int32(
                                        llvm.load(
                                            T.i32,
                                            _gep3(ptr, disp),
                                            alignment=4,
                                            alias_scopes=sc[phase][0],
                                            noalias_scopes=sc[phase][1],
                                        )
                                    )
                                )
                            for ci in range_constexpr(4):
                                if (wm < fx.Int32(2)) | (fx.Int32(ci) < fx.Int32(2)):
                                    af = read_a(slot, ci)
                                    sptr = s_ptr0 if ci % 2 == 0 else s_ptr1
                                    disp = S_BASE + (phase) * S_SLOT + (ci // 2) * 256
                                    scale_word = fx.Int32(
                                        llvm.load(
                                            T.i32,
                                            _gep3(sptr, disp),
                                            alignment=4,
                                            alias_scopes=sc[phase][0],
                                            noalias_scopes=sc[phase][1],
                                        )
                                    )
                                    scale_a = scale_word
                                    rocdl.sched_barrier(0)
                                    for j in range_constexpr(N_REPS):
                                        fx.gemm(
                                            mma[(ci % 2 + 2 * h, j % 2 + 2 * h)],
                                            acc[ci][j],
                                            af,
                                            bf[j],
                                            acc[ci][j],
                                            scale_a=scale_a,
                                            scale_b=bs_current[j // 2],
                                        )
                                    rocdl.sched_barrier(0)
                            if const_expr(h == 1):
                                if word + fx.Int32(2) < fx.Int32(H // 256):
                                    load_s(cur, (phase + 2) % 3, word + fx.Int32(2))
                                    load_bs(cur, (phase + 2) % 3, word + fx.Int32(2))
                carried = yield [
                    acc[ci][j].load()
                    for ci in range_constexpr(4)
                    for j in range_constexpr(4)
                ]
            for ci in range_constexpr(4):
                for j in range_constexpr(4):
                    acc[ci][j].store(fx.Vector(carried[ci * 4 + j]))

            # Every wave has finished reading this tile's LDS slots.
            gpu.barrier()
            nxt_pid = fx.Int32(iv) + fx.Int32(gpu.grid_dim.x)
            nxt = tile_setup(fx.min(nxt_pid, bound - fx.Int32(1)))
            if nxt_pid < bound:
                prologue(nxt)

            # FlyDSL has no opaque identity primitive. The v1 ISA kept
            # epilogue addresses live across K and spilled them to scratch.
            epi_tx = fx.Int32(
                llvm.inline_asm(
                    T.i32,
                    [as_ir_value(tx)],
                    "v_mov_b32 $0, $1",
                    "=v,v",
                    has_side_effects=True,
                )
            )
            epi_lane = fx.Int32(fx.Uint32(epi_tx) % fx.Uint32(64))
            epi_wave = fx.Int32(
                rocdl.readfirstlane(
                    T.i32, as_ir_value(fx.Int32(fx.Uint32(epi_tx) // fx.Uint32(64)))
                )
            )
            epi_wn, epi_wm = epi_wave % fx.Int32(4), epi_wave // fx.Int32(4)
            epi_ld16, epi_lm16 = epi_lane // fx.Int32(16), epi_lane % fx.Int32(16)
            z = fx.full(2, 0, fx.Int16)
            # Row mrow + 64 * wm + 16 * ci + 4 * ld16 + v, column
            # 128 * ni + 32 * wn + 16 * half + lm16 of the FP8 output.
            q_base = (
                mrow + epi_wm * fx.Int32(64) + epi_ld16 * fx.Int32(4)
            ) * fx.Int32(INTER) + ni * fx.Int32(BN // 2) + epi_wn * fx.Int32(32) + epi_lm16
            # The shuffled E8M0 offset of the same row and 32-column group,
            # with the non-negative divisions written as shifts and masks.
            s_base = (
                (mrow.shrui(fx.Int32(5)) + epi_wm * fx.Int32(2))
                * fx.Int32(OUT_AS_DW * 4)
                + ni.shrui(fx.Int32(1)) * fx.Int32(256)
                + epi_wn * fx.Int32(64)
                + epi_ld16 * fx.Int32(16)
                + (ni & fx.Int32(1)) * fx.Int32(2)
            )
            # The interleaved weights put gate (even j) and up (odd j) of one
            # output column in the same lane, and one 16-lane row holds a full
            # 32-column E8M0 group, so activation and quantization stay in
            # registers. All 160 rows, including padding, are written.
            for ci in range_constexpr(4):
                if (epi_wm < fx.Int32(2)) | (fx.Int32(ci) < fx.Int32(2)):
                    gates, ups = [], []
                    for half in range_constexpr(2):
                        g = fx.Vector(acc[ci][2 * half].load())
                        u = fx.Vector(acc[ci][2 * half + 1].load())
                        for v in range_constexpr(4):
                            gates.append(g[v])
                            ups.append(u[v])
                    result = _situv2_mul_batch(gates, ups, situ_beta, situ_linear_beta)
                    result = [x.to(fx.BFloat16).to(fx.Float32) for x in result]
                    for v in range_constexpr(4):
                        lo, hi = result[v], result[4 + v]
                        amax = _fabs_f32(lo).maximumf(_fabs_f32(hi))
                        amax = _dpp_row16_amax(amax.bitcast(fx.Int32)).bitcast(
                            fx.Float32
                        )
                        code, scale = _e8m0_from_amax(amax, max_norm=448.0)
                        p = rocdl.cvt_scalef32_pk_fp8_f32(
                            fx.Vector.make_type(2, fx.Int16),
                            z,
                            as_ir_value(lo),
                            as_ir_value(hi),
                            as_ir_value(scale),
                            0,
                        )
                        word = fx.Vector(p).bitcast(fx.Int32)[0]
                        q_off = q_base + fx.Int32((ci * 16 + v) * INTER)
                        store_byte(qbuf, q_off, word)
                        store_byte(qbuf, q_off + fx.Int32(16), word.shrui(fx.Int32(8)))
                        if epi_lm16 == fx.Int32(0):
                            s_off = s_base + fx.Int32(
                                (ci // 2) * OUT_AS_DW * 4 + v * 4 + ci % 2
                            )
                            store_byte(sbuf, s_off, code)

            results = yield nxt

    @flyc.jit
    def launch(
        aq: fx.Int64,
        asc: fx.Int64,
        bq: fx.Int64,
        bsc: fx.Int64,
        eids: fx.Int64,
        counts: fx.Int64,
        rows: fx.Int64,
        tokens: fx.Int32,
        grid: fx.Int32,
        out: fx.Int64,
        out_scale: fx.Int64,
        hidden: fx.Int64,
        bias: fx.Int64,
        stream: fx.Stream,
    ):
        gemm1(
            aq, asc, bq, bsc, eids, counts, rows, tokens, out, out_scale, hidden, bias
        ).launch(grid=(GRID, 1, 1), block=(768, 1, 1), stream=stream)

    launch.compile_hints = {
        "waves_per_eu": 3,
        "llvm_options": {"amdgpu-sched-strategy": "iterative-minreg"},
    }
    return launch

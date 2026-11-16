# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""M8192 Kimi-K3 EP32 GEMM1: full BM160 with twelve waves."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from .mxfp4_gemm2_ep32 import lds_slot_scopes
from .mxfp4_gemm_common import (
    _activation_mul_batch,
    _buffer_rsrc,
    _gep3,
    _e8m0_from_amax,
    _fabs_f32,
    _global_i32_at,
    _global_scalar_tiles,
    _inline_dpp_quad_amax,
    _lds_ptr3,
    _scalar_store,
    _scale_mma_atoms,
    _udiv,
    _umod,
    bq_bytes_for,
    bscale_bytes_for,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kbs_stride_n0_dw_for,
    lds_swizzle_mask_f8,
)


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
    name = f"gemm1_ep8192_a8w4_bm160_bn{BN}_w12_m64x2_m32_as{A_STAGES}_bs{B_STAGES}_ak128_{'nt' if use_nt else 'cached'}_situv2_runtimek_lateepi"

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
        mem1, mem4 = fx.make_layout(1, 1), fx.make_layout(4, 1)
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
        qtiles = _global_scalar_tiles(out, fx.Int32, 1 << 28)
        stiles = _global_scalar_tiles(out_scale, fx.Int8, 1 << 28)
        ctiles = fx.logical_divide(
            fx.make_view(
                fx.recast_iter(fx.Float32, lds_ptr), fx.make_layout(32 * (BN + 4), 1)
            ),
            mem1,
        )
        ccopy = fx.make_copy_atom(fx.UniversalCopy32b(), fx.Float32)

        for iv in range(fx.block_idx.x, bound, gpu.grid_dim.x):
            gpu.barrier()
            pid = fx.Int32(iv)
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
                    rocdl.readfirstlane(
                        T.i32, as_ir_value((expert * N_OUT + col) * (H // 2))
                    )
                )
            scale_base = (
                expert * BS_DW + (ni * (BN // 32) + wn * (BN // 128)) * BS_N_DW
            ) * 4
            scale_base = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(scale_base)))
            acc = [
                [fx.make_rmem_tensor(mem4, fx.Float32) for _ in range_constexpr(N_REPS)]
                for _ in range_constexpr(M_CHUNKS)
            ]
            for i in range_constexpr(M_CHUNKS):
                for j in range_constexpr(N_REPS):
                    acc[i][j].store(fx.full(4, 0.0, fx.Float32))

            # Reuse route addresses for every K step. Loading row IDs inside
            # load_a forces vmcnt(0) waits between the future A/B DMAs.
            a_voff = []
            for sub in range_constexpr(5):
                row = wn * 40 + sub * 8
                actual = _global_i32_at(token_rows, mrow + row + ld8)
                mask = lds_swizzle_mask_f8(row + ld8, BK)
                a_voff.append(actual * H + ((lm8 * 16) ^ mask))

            def load_a(slot, step):
                if loader:
                    for sub in range_constexpr(5):
                        row = wn * 40 + sub * 8
                        # Scoped DMA has no equivalent fx.copy alias-scope API.
                        rocdl.raw_ptr_buffer_load_lds(
                            ares,
                            _lds_ptr3(lds, fx.Int32(slot * A_SLOT) + row * BK),
                            fx.Int32(16),
                            a_voff[sub],
                            step * fx.Int32(BK),
                            fx.Int32(0),
                            fx.Int32(0),
                            alias_scopes=ac[slot][0],
                            noalias_scopes=ac[slot][1],
                        )

            def load_b(slot, step):
                if loader:
                    for j in range_constexpr(N_REPS):
                        dst = fx.Int32(B_BASE + slot * B_SLOT + j * 1024) + wn * (
                            BN * 16
                        )
                        rocdl.raw_ptr_buffer_load_lds(
                            bres,
                            _lds_ptr3(lds, dst),
                            fx.Int32(16),
                            ld16 * 256 + lm16 * 16 + step * fx.Int32(1024),
                            fx.Int32(bbase[j]),
                            fx.Int32(0),
                            2 if use_nt else 0,
                            alias_scopes=bc[slot][0],
                            noalias_scopes=bc[slot][1],
                        )

            def load_s(slot, word):
                if loader:
                    for group in range_constexpr(2):
                        sub = wn + group * 4
                        if sub < fx.Int32(5):
                            src = (mrow // 32 + sub) * (AS_DW * 4)
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

            def load_bs(slot, word):
                if loader:
                    for pair in range_constexpr(2):
                        dst = (
                            fx.Int32(S_BASE + slot * S_SLOT + 5 * 256 + pair * 256)
                            + wn * 512
                        )
                        src = scale_base + pair * BS_N_DW * 4
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

            for p in range_constexpr(A_STAGES - 1):
                load_a(p, fx.Int32(p))
            for p in range_constexpr(B_STAGES - 1):
                load_b(p, fx.Int32(p))
            load_s(0, fx.Int32(0))
            load_s(1, fx.Int32(1))
            load_bs(0, fx.Int32(0))
            load_bs(1, fx.Int32(1))
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
                                load_a(1 - h, step + fx.Int32(1))
                            if step + fx.Int32(2) < fx.Int32(H // BK):
                                load_b((slot_b + 2) % 3, step + fx.Int32(2))
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
                                    load_s((phase + 2) % 3, word + fx.Int32(2))
                                    load_bs((phase + 2) % 3, word + fx.Int32(2))
                carried = yield [
                    acc[ci][j].load()
                    for ci in range_constexpr(4)
                    for j in range_constexpr(4)
                ]
            for ci in range_constexpr(4):
                for j in range_constexpr(4):
                    acc[ci][j].store(fx.Vector(carried[ci * 4 + j]))

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
            epi_m_chunk_base = epi_wm * fx.Int32(4)
            for region in range_constexpr(5):
                gpu.barrier()
                for ci in range_constexpr(4):
                    epi_global_ci = epi_m_chunk_base + ci
                    if (epi_global_ci // 2 == fx.Int32(region)) & (
                        (epi_wm < fx.Int32(2)) | (fx.Int32(ci) < fx.Int32(2))
                    ):
                        row = (epi_global_ci % 2) * 16 + epi_ld16 * 4
                        for j in range_constexpr(N_REPS):
                            col = (
                                epi_wn * (BN // 8)
                                + (j // 2) * 16
                                + epi_lm16
                                + (BN // 2 if j % 2 else 0)
                            )
                            values = fx.Vector(acc[ci][j].load())
                            for v in range_constexpr(4):
                                r = fx.make_rmem_tensor(mem1, fx.Float32)
                                r.store(
                                    fx.Vector.from_elements([values[v]], fx.Float32)
                                )
                                fx.copy(
                                    ccopy,
                                    r,
                                    fx.slice(
                                        ctiles, (None, (row + v) * (BN + 4) + col)
                                    ),
                                )
                gpu.barrier()
                if epi_tx < fx.Int32(2 * BN):
                    mr, nl = epi_tx // (BN // 16), epi_tx % (BN // 16)
                    ng, kk = nl // 4, nl % 4
                    result = []
                    for part in range_constexpr(2):
                        gates, ups = [], []
                        for e in range_constexpr(4):
                            col = ng * 32 + kk * 8 + part * 4 + e
                            gr = fx.make_rmem_tensor(mem1, fx.Float32)
                            ur = fx.make_rmem_tensor(mem1, fx.Float32)
                            fx.copy(
                                ccopy, fx.slice(ctiles, (None, mr * (BN + 4) + col)), gr
                            )
                            fx.copy(
                                ccopy,
                                fx.slice(ctiles, (None, mr * (BN + 4) + col + BN // 2)),
                                ur,
                            )
                            gates.append(gr.load()[0])
                            ups.append(ur.load()[0])
                        result.extend(
                            _activation_mul_batch(
                                gates,
                                ups,
                                act="situv2",
                                situ_beta=situ_beta,
                                situ_linear_beta=situ_linear_beta,
                            )
                        )
                        rocdl.sched_barrier(0)
                    result = [x.to(fx.BFloat16).to(fx.Float32) for x in result]
                    amax = _fabs_f32(result[0])
                    for e in range_constexpr(1, 8):
                        amax = amax.maximumf(_fabs_f32(result[e]))
                    amax = _inline_dpp_quad_amax(amax.bitcast(fx.Int32)).bitcast(
                        fx.Float32
                    )
                    code, scale = _e8m0_from_amax(amax, max_norm=448.0)
                    z = fx.full(2, 0, fx.Int16)
                    words = []
                    for e0 in range_constexpr(0, 8, 4):
                        p = rocdl.cvt_scalef32_pk_fp8_f32(
                            fx.Vector.make_type(2, fx.Int16),
                            z,
                            as_ir_value(result[e0]),
                            as_ir_value(result[e0 + 1]),
                            as_ir_value(scale),
                            0,
                        )
                        p = rocdl.cvt_scalef32_pk_fp8_f32(
                            fx.Vector.make_type(2, fx.Int16),
                            p,
                            as_ir_value(result[e0 + 2]),
                            as_ir_value(result[e0 + 3]),
                            as_ir_value(scale),
                            1,
                        )
                        words.append(fx.Vector(p).bitcast(fx.Int32)[0])
                    row = mrow + region * 32 + mr
                    byte = row * INTER + ni * (BN // 2) + ng * 32 + kk * 8
                    _scalar_store(qtiles, byte // 4, words[0], fx.Int32)
                    _scalar_store(qtiles, byte // 4 + 1, words[1], fx.Int32)
                    if kk == fx.Int32(0):
                        chunk = mrow // 32 + region
                        scale_k = ni * (BN // 64) + ng
                        dw = (
                            chunk * OUT_AS_DW
                            + (scale_k // 8) * 64
                            + (scale_k % 4) * 16
                            + (mr % 16)
                        )
                        _scalar_store(
                            stiles,
                            dw * 4 + ((scale_k // 4) % 2) * 2 + mr // 16,
                            code,
                            fx.Int8,
                        )

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

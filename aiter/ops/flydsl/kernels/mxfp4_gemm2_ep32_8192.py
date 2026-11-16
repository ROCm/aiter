# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""M8192 Kimi-K3 EP32 GEMM2: full BM160 with twelve waves."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T, as_ir_value

from .mxfp4_gemm2_ep32 import lds_slot_scopes
from .mxfp4_gemm_common import (
    _buffer_rsrc,
    _gep3,
    _global_i32_at,
    _lds_ptr3,
    _scale_mma_atoms,
    bq_bytes_for,
    bscale_bytes_for,
    global_typed_ptr,
    kas_per_chunk_dw_for,
    kbs_per_expert_dw_for,
    kbs_stride_n0_dw_for,
    lds_swizzle_mask_f8,
)


def compile_gemm2_ep8192(*, B_NT=True, GRID_WGS=256, topk=16):
    if topk not in (16, 17):
        raise ValueError("M8192 GEMM2 requires Kimi-K3 EP32 routing")
    from .mxmoe_dispatcher import _spart_output_tile_index

    BM, BN, BK, H, INTER, NE = 160, 256, 128, 3584, 3072, 28
    ADW = kas_per_chunk_dw_for(INTER)
    BSDW = kbs_per_expert_dw_for(H, INTER)
    BSNDW = kbs_stride_n0_dw_for(INTER)
    A_SLOT, B_SLOT, S_SLOT = BM * BK, BN * BK // 2, (5 + BN // 32) * 256
    B_BASE, S_BASE = 2 * A_SLOT, 2 * A_SLOT + 3 * B_SLOT
    LDS_BYTES = S_BASE + 3 * S_SLOT
    grid_wgs = min(GRID_WGS, 256)
    name = f"gemm2_ep8192_a8w4_bm160_bn256_w12_m64x2_m32_as2_bs3_nativebf16w_runtimek_lateepi{'_bnt' if B_NT else ''}"

    @fx.struct
    class Storage:
        raw: fx.Array[fx.Uint8, LDS_BYTES, 16]

    @flyc.kernel(name=name, known_block_size=[768, 1, 1])
    def gemm2(
        aq: fx.Int64,
        asc: fx.Int64,
        bq: fx.Int64,
        bsc: fx.Int64,
        eids: fx.Int64,
        counts: fx.Int64,
        stids: fx.Int64,
        sweights: fx.Int64,
        bias: fx.Int64,
        tokens: fx.Int32,
        max_m: fx.Int32,
        inter: fx.Int32,
        hidden: fx.Int32,
        out: fx.Int64,
        out_scale: fx.Int64,
        grid: fx.Int32,
    ):
        tx = fx.thread_idx.x
        lane = tx % fx.Int32(64)
        # The typed API does not annotate wave-uniform resource bases.
        wave = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(tx // fx.Int32(64))))
        wn, wm = wave % fx.Int32(4), wave // fx.Int32(4)
        # Complete M groups of 64, 64, and 32 rows share the weights.
        m_chunk_base = wm * fx.Int32(4)
        loader = wm == fx.Int32(0)
        ld16, lm16 = lane // fx.Int32(16), lane % fx.Int32(16)
        ld8, lm8 = lane // fx.Int32(8), lane % fx.Int32(8)
        lds_ptr = fx.SharedAllocator().allocate(Storage).peek().raw.ptr
        lds = fx.Int32(fx.ptrtoint(lds_ptr))
        scopes = lds_slot_scopes(8, "ep8192_g2")
        ac, bc, sc = scopes[:2], scopes[2:5], scopes[5:]
        m_blocks = _global_i32_at(counts, fx.Int32(0)) // fx.Int32(BM)
        bound = m_blocks * fx.Int32(H // BN)
        ares = _buffer_rsrc(aq, fx.Int64(max_m) * BM * INTER)
        asres = _buffer_rsrc(asc, fx.Int64(max_m) * 5 * ADW * 4)
        bres = _buffer_rsrc(bq, bq_bytes_for(NE, H, INTER))
        bsres = _buffer_rsrc(bsc, bscale_bytes_for(NE, H, INTER))
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
        weight_view = fx.Tensor(
            fx.make_view(
                global_typed_ptr(sweights, T.f32, align=16),
                fx.make_layout((1, 1), (1, 1)),
            )
        )
        weight_buf = fx.rocdl.make_buffer_tensor(weight_view, max_size=True)
        weight_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        output_view = fx.Tensor(
            fx.make_view(
                global_typed_ptr(out, T.bf16, align=16), fx.make_layout((1, 1), (1, 1))
            )
        )
        output_buf = fx.rocdl.make_buffer_tensor(output_view, max_size=True)
        output_copy = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)

        for iv in range(fx.block_idx.x, bound, gpu.grid_dim.x):
            gpu.barrier()
            mi, ni = _spart_output_tile_index(
                fx.Int32(iv), m_blocks, fx.Int32(H // BN), 4, 4
            )
            mi = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(mi)))
            ni = fx.Int32(rocdl.readfirstlane(T.i32, as_ir_value(ni)))
            mrow = mi * fx.Int32(BM)
            expert = fx.Int32(
                rocdl.readfirstlane(T.i32, as_ir_value(_global_i32_at(eids, mi)))
            )
            bbase = [
                fx.Int32(
                    rocdl.readfirstlane(
                        T.i32,
                        as_ir_value(
                            (expert * H + ni * BN + wn * 64 + j * 16) * (INTER // 2)
                        ),
                    )
                )
                for j in range_constexpr(4)
            ]
            scale_base = (expert * BSDW + (ni * 8 + wn * 2) * BSNDW) * 4
            acc = [
                [fx.make_rmem_tensor(mem4, fx.Float32) for _ in range_constexpr(4)]
                for _ in range_constexpr(4)
            ]
            for ci in range_constexpr(4):
                for j in range_constexpr(4):
                    acc[ci][j].store(fx.full(4, 0.0, fx.Float32))
            a_voff = []
            for sub in range_constexpr(5):
                row = wn * 40 + sub * 8
                mask = lds_swizzle_mask_f8(row + ld8, BK)
                a_voff.append((mrow + row + ld8) * INTER + ((lm8 * 16) ^ mask))

            def load_a(slot, step):
                if loader:
                    for sub in range_constexpr(5):
                        row = wn * 40 + sub * 8
                        # FlyDSL copy atoms do not expose LDS alias scopes.
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
                    for j in range_constexpr(4):
                        offset = fx.Int32(B_BASE + slot * B_SLOT + j * 1024) + wn * 4096
                        rocdl.raw_ptr_buffer_load_lds(
                            bres,
                            _lds_ptr3(lds, offset),
                            fx.Int32(16),
                            ld16 * 256 + lm16 * 16 + step * fx.Int32(1024),
                            bbase[j],
                            fx.Int32(0),
                            2 if B_NT else 0,
                            alias_scopes=bc[slot][0],
                            noalias_scopes=bc[slot][1],
                        )

            def load_s(slot, word):
                if loader:
                    for group in range_constexpr(2):
                        sub = wn + group * 4
                        if sub < fx.Int32(5):
                            src = (mrow // 32 + sub) * (ADW * 4)
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
                        src = scale_base + pair * BSNDW * 4
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

            load_a(0, fx.Int32(0))
            load_b(0, fx.Int32(0))
            load_b(1, fx.Int32(1))
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
                fx.Int32(0), fx.Int32(INTER // 256), fx.Int32(1), init=init_c
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
                                elif word == fx.Int32(INTER // 256 - 1):
                                    rocdl.s_waitcnt(vmcnt=4, lgkmcnt=0)
                                else:
                                    rocdl.s_waitcnt(vmcnt=7, lgkmcnt=0)
                            else:
                                if word == fx.Int32(INTER // 256 - 1):
                                    rocdl.s_waitcnt(vmcnt=0, lgkmcnt=0)
                                else:
                                    rocdl.s_waitcnt(vmcnt=4, lgkmcnt=0)
                            rocdl.s_barrier()
                            rocdl.sched_barrier(0)
                            bf = []
                            for j in range_constexpr(4):
                                disp = B_BASE + slot_b * B_SLOT + j * 1024
                                r = fx.make_rmem_tensor(mem4, fx.Int32)
                                # Scoped LDS reads retain the DMA ordering metadata.
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
                            if step + fx.Int32(1) < fx.Int32(INTER // BK):
                                load_a(1 - h, step + fx.Int32(1))
                            if step + fx.Int32(2) < fx.Int32(INTER // BK):
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
                                    for j in range_constexpr(4):
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
                                if word + fx.Int32(2) < fx.Int32(INTER // 256):
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
            epi_ld16 = epi_lane // fx.Int32(16)
            epi_m_chunk_base = epi_wm * fx.Int32(4)
            # The native layout and per-route BF16 rounding match the
            # committed reducer input. All 160 rows, including padding, write.
            for ci in range_constexpr(4):
                if (epi_wm < fx.Int32(2)) | (fx.Int32(ci) < fx.Int32(2)):
                    weight_row = mrow + (epi_m_chunk_base + ci) * 16 + epi_ld16 * 4
                    wf = fx.make_rmem_tensor(mem4, fx.Float32)
                    fx.copy(weight_copy, weight_buf[None, weight_row], wf)
                    weights4 = fx.Vector(wf.load())
                    mc = mrow // 16 + epi_m_chunk_base + ci
                    for j in range_constexpr(4):
                        nc = ni * 16 + epi_wn * 4 + j
                        offset = (mc * (H // 16) + nc) * 256 + epi_lane * 4
                        values = (fx.Vector(acc[ci][j].load()) * weights4).to(
                            fx.BFloat16
                        )
                        frag = fx.make_rmem_tensor(mem4, fx.BFloat16)
                        frag.store(values)
                        fx.copy(output_copy, frag, output_buf[None, offset])

    @flyc.jit
    def launch(
        aq: fx.Int64,
        asc: fx.Int64,
        bq: fx.Int64,
        bsc: fx.Int64,
        eids: fx.Int64,
        counts: fx.Int64,
        stids: fx.Int64,
        sweights: fx.Int64,
        bias: fx.Int64,
        tokens: fx.Int32,
        max_m: fx.Int32,
        grid: fx.Int32,
        inter: fx.Int32,
        hidden: fx.Int32,
        out: fx.Int64,
        out_scale: fx.Int64,
        stream: fx.Stream,
    ):
        gemm2(
            aq,
            asc,
            bq,
            bsc,
            eids,
            counts,
            stids,
            sweights,
            bias,
            tokens,
            max_m,
            inter,
            hidden,
            out,
            out_scale,
            grid,
        ).launch(grid=(grid_wgs, 1, 1), block=(768, 1, 1), stream=stream)

    launch.compile_hints = {
        "waves_per_eu": 3,
        "llvm_options": {"amdgpu-sched-strategy": "iterative-minreg"},
    }
    return launch

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""PA decode prologue and single-/multi-query tile pipelines.

The pipeline owns instruction ordering. Operation helpers keep descriptors and
compile-time policy; loop-carried K/V, maxima, denominators and outputs remain
explicit SSA values. Every LDS publication/retirement barrier stays here.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp

from .op_epilog import PaDecodeEpilogue
from .op_gemm import PaDecodeGemm
from .op_lds import PaDecodeKVLoader, PaDecodeLds, PaDecodeQueryLoader
from .op_softmax import PaDecodeSoftmax
from .traits import MFMA_MNK


class PaDecodePipeline:
    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.lds = PaDecodeLds(ctx.traits, ctx.shared_storage)
        self.gemm = PaDecodeGemm(ctx.traits)
        self.query = PaDecodeQueryLoader(ctx, self.lds, self.gemm)
        self.kv = PaDecodeKVLoader(ctx, self.lds)
        self.softmax = PaDecodeSoftmax(ctx, self.lds, self.kv, self.gemm)
        self.epilogue = PaDecodeEpilogue(ctx)

    @flyc.jit
    def run(self):
        """Initialize resources in load order, then compute and store one CTA."""
        self.ctx.init_thread_mapping()
        self.epilogue.init_outputs()
        self.kv.init_loaders()
        self.query.init_loaders()
        self.ctx.init_query_mapping()
        # Fused MTP starts every Q load before the first K/page/scale prefetch.
        q_units_prefetched = self.query.prefetch()
        self.ctx.init_context_length()
        self.kv.init_page_table()
        self.ctx.init_partition_bounds()
        self.lds.allocate()
        self.kv.init_scale_loaders()
        k_pf0 = self.kv.prefetch_first()
        self.ctx.init_scales(self.kv.key_scale, self.kv.value_scale)
        self.query.stage(q_units_prefetched)
        gpu.barrier()
        v_page_pf0 = self.kv.read_v_pages()
        q_ops_all = self.query.load_operands()
        self.softmax.init_mask()
        self.ctx.init_causal_offsets()
        o_final = self.mainloop(k_pf0, v_page_pf0, q_ops_all)
        self.epilogue.store(o_final)

    @flyc.jit
    def mainloop(self, k_pf0, v_page_pf0, q_ops_all):
        """Carry prefetch operands and online-softmax state between KV tiles."""
        o_zero = fx.Vector.filled(self.traits.OP_ELEMS, 0.0, fx.Float32)
        init_state = [k_pf0, v_page_pf0]
        for _m in range_constexpr(self.traits.M_TILES):
            init_state.extend(
                [o_zero] * self.traits.VHE_CHUNKS + [self.ctx.NEG_INF, self.ctx.ZERO_F]
            )
        # Reuse carried V registers only after the final query consumes a chunk.
        if const_expr(self.traits.MTP4_PREFETCH_V):
            v_pf0 = fx.Vector.filled(
                self.traits.VHE_CHUNKS * self.traits.NVOPS, 0, fx.Int64
            )
            if self.ctx.part_start < self.ctx.part_end:
                v_flat0 = []
                for vh in range_constexpr(self.traits.VHE_CHUNKS):
                    v_flat0.extend(self.kv.load_v(v_page_pf0, vh))
                v_pf0 = fx.Vector.from_elements(v_flat0, dtype=fx.Int64)
            init_state.append(v_pf0)
        # Single-tile plans eliminate loop/history but keep absolute tt for
        # addressing and masks; the outer guard excludes padded tasks.
        loop_start = (
            0 if const_expr(self.traits.single_tile_plan) else self.ctx.part_start
        )
        loop_end = 1 if const_expr(self.traits.single_tile_plan) else self.ctx.part_end
        for loop_i, ostate in range(loop_start, loop_end, 1, init=init_state):
            k_cur = ostate[
                self.traits.K_SLOT
            ]  # this tile's prefetched K, as one (NCHUNK*N_SUBCHUNKS,) i64 vector
            v_page_cur = ostate[
                self.traits.V_SLOT
            ]  # this tile's V pages, as one PAGES_PER_CHUNK-wide i32 vector
            tt = (
                fx.Int32(self.ctx.part_start)
                if const_expr(self.traits.single_tile_plan)
                else fx.Int32(loop_i)
            )
            tok0 = tt * self.traits.TILE_TOK
            # Interleave MFMA with VALU/LDS or page16 V loads.
            if const_expr(
                (not self.traits.per_token_kv and self.traits.M_TILES > 1)
                or self.traits.PAGE16_VPIPE
            ):
                fx.rocdl.iglp_opt(0)

            tt1 = tt + 1

            cur_kv_buf = self.kv.scale_buffer_offset(tt)

            # Exclude unwritten-tail scales from FP8 normalization using the
            # context bound, not a per-query causal bound, for shared MTP scales.
            tile_valid = self.ctx.context_len - tok0
            window_left = None
            if const_expr(self.traits.sliding_window > 0):
                # Clamp before subtracting MTP offsets to avoid int32 underflow;
                # negative left edges exclude no tile-relative token.
                window_left = tile_valid - self.traits.sliding_window
                window_left = (window_left > 0).select(window_left, 0)

            scale_bounds = None
            if const_expr(self.traits.per_token_kv):
                scale_bounds = self.softmax.scale_mask(tile_valid, window_left)

            # Independent V loads overlap QK/softmax and are shared across M-tiles.
            v_vh_shared = None
            if const_expr(self.traits.MTP4_PREFETCH_V):
                v_carried = ostate[self.traits.V_DATA_SLOT]
                v_vh_shared = [
                    [
                        v_carried[vh * self.traits.NVOPS + i]
                        for i in range_constexpr(self.traits.NVOPS)
                    ]
                    for vh in range_constexpr(self.traits.VHE_CHUNKS)
                ]
            elif const_expr(
                (self.traits.M_TILES > 1 and not self.traits.MTP4_FUSED)
                or (self.traits.prefetch_v and not self.traits.SCALES_BEFORE_CURRENT_V)
            ):
                v_vh_shared = [
                    self.kv.load_v(v_page_cur, vh)
                    for vh in range_constexpr(self.traits.VHE_CHUNKS)
                ]

            q_scale_vec = None
            if const_expr(self.traits.M_TILES > 1):
                q_scale_vec = self.lds.load(
                    self.traits.sQscale_off
                    + self.ctx.lane16 * (self.traits.M_TILES * self.traits.f32),
                    fx.Float32,
                    self.traits.M_TILES,
                )

            # All query tiles share one K/V tile; each path retains its publication barriers.
            if const_expr(self.traits.M_TILES > 1):
                next_state = self.multi_query_tile(
                    ostate,
                    k_cur,
                    v_page_cur,
                    tt1,
                    tile_valid,
                    window_left,
                    cur_kv_buf,
                    scale_bounds,
                    v_vh_shared,
                    q_scale_vec,
                    q_ops_all,
                )
            else:
                next_state = self.single_query_tile(
                    ostate,
                    k_cur,
                    v_page_cur,
                    tt1,
                    tile_valid,
                    window_left,
                    cur_kv_buf,
                    scale_bounds,
                    v_vh_shared,
                    q_ops_all,
                )
            results = yield next_state
        o_final = results
        return o_final

    @flyc.jit
    def multi_query_tile(
        self,
        ostate,
        k_cur,
        v_page_cur,
        tt1,
        tile_valid,
        window_left,
        cur_kv_buf,
        scale_bounds,
        v_vh_shared,
        q_scale_vec,
        q_ops_all,
    ):
        """Publish all query maxima together, then consume shared K/V per query."""
        next_state = [None, None]
        masked_chunks_saved = [None] * self.traits.M_TILES

        # Reread V scales after Phase A to reduce peak register liveness.
        k_scale_shared = None
        if const_expr(self.traits.per_token_kv):
            v_scale_A = [
                self.kv.load_scale(self.traits.sVScale_off, a, cur_kv_buf)
                for a in range_constexpr(self.traits.NCHUNK)
            ]
            self.softmax.stage_v_scale_max(v_scale_A, scale_bounds)
            k_scale_shared = [
                self.kv.load_scale(self.traits.sKScale_off, a, cur_kv_buf)
                for a in range_constexpr(self.traits.NCHUNK)
            ]

        for m in range_constexpr(self.traits.M_TILES):
            frag_Ss = self.gemm.qk(k_cur, q_ops_all, m)

            scale = self.ctx.scale_qk * fx.Float32(q_scale_vec[m])
            masked_chunks = self.softmax.multi_scores(
                frag_Ss, scale, k_scale_shared, tile_valid, window_left, m
            )

            self.softmax.stage_row_max(
                masked_chunks, self.softmax.lmax_offset(m), scale, True
            )

            masked_chunks_saved[m] = masked_chunks

        # Publish next pages/scales with the Phase A barrier.
        k_next = (
            fx.Vector.filled(self.traits.NCHUNK * self.traits.N_SUBCHUNKS, 0, fx.Int64)
            if const_expr(self.traits.MTP4_FUSED)
            else k_cur
        )
        if const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end:
            if const_expr(self.traits.MTP4_FUSED):
                # Defer K until P packing releases saved score registers.
                phys_vec1 = self.kv.fetch_v_pages(tt1)
                self.kv.stage_scales(phys_vec1, self.kv.scale_buffer_offset(tt1))
            else:
                k_next, phys_vec1 = self.kv.load_k(tt1)
                self.kv.fetch_v_pages(tt1)
                if const_expr(self.traits.per_token_kv):
                    self.kv.stage_scales(phys_vec1, self.kv.scale_buffer_offset(tt1))
        next_state[self.traits.K_SLOT] = k_next

        gpu.barrier()

        v_page_next = v_page_cur
        if const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end:
            v_page_next = self.kv.read_v_pages()
        next_state[self.traits.V_SLOT] = v_page_next

        # Large M-tile groups reread scale chunks to bound VGPR liveness.
        v_scale_shared = None
        if const_expr(self.traits.per_token_kv and self.traits.M_TILES < 4):
            v_scale_shared = [
                self.kv.load_scale(self.traits.sVScale_off, a, cur_kv_buf)
                for a in range_constexpr(self.traits.NCHUNK)
            ]

        if const_expr(self.traits.MTP4_FUSED):
            if const_expr(not self.traits.MTP4_PREFETCH_V and not self.traits.trans_v):
                # Hide plain V's strided loads behind P packing.
                v_vh_shared = [
                    self.kv.load_v(v_page_cur, vh)
                    for vh in range_constexpr(self.traits.VHE_CHUNKS)
                ]
            # V-scale normalization is shared by all query rows.
            v_max_scaled, norm_factor_b = self.softmax.v_scale_normalization()

            # Private P/Lsum slots share one barrier without overwrite races.
            m_new_saved = self.softmax.pack_fused(
                masked_chunks_saved, ostate, cur_kv_buf, norm_factor_b
            )
            if const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end:
                # Saved scores are dead; next K can overlap the P barrier/PV.
                k_next = self.kv.load_k_from_pages(self.kv.read_k_pages())
            next_state[self.traits.K_SLOT] = k_next

            gpu.barrier()

            if const_expr(not self.traits.MTP4_PREFETCH_V and self.traits.trans_v):
                # Delay transposed V until saved score registers are dead.
                v_vh_shared = [
                    self.kv.load_v(v_page_cur, vh)
                    for vh in range_constexpr(self.traits.VHE_CHUNKS)
                ]
            v_next_chunks = []
            for m in range_constexpr(self.traits.M_TILES):
                p_base = self.traits.sP_off + m * MFMA_MNK * self.traits.SP_ROW_BYTES
                lsum_base = (
                    self.traits.sLsum_off
                    + m * MFMA_MNK * self.traits.NWARP_PAD * self.traits.f32
                )
                o_acc = [
                    ostate[self.traits.o_slot(m, vh)]
                    for vh in range_constexpr(self.traits.VHE_CHUNKS)
                ]
                m_prev = ostate[self.traits.m_slot(m)]
                l_prev = ostate[self.traits.l_slot(m)]
                m_new = m_new_saved[m]
                safe_max = (m_new > self.ctx.NEG_INF).select(m_new, self.ctx.ZERO_F)
                corr_reg = (
                    self.ctx.ZERO_F
                    if const_expr(self.traits.single_tile_plan)
                    else fx.Float32(fx.exp2(m_prev - safe_max, fastmath="fast"))
                )
                gsum = self.lds.load_wave(lsum_base, self.ctx.lane16).reduce(
                    ReductionOp.ADD
                )
                l_new = (
                    gsum
                    if const_expr(self.traits.single_tile_plan)
                    else l_prev * corr_reg + gsum
                )
                p_ops = self.lds.load(
                    p_base
                    + self.ctx.lane16 * self.traits.SP_ROW_BYTES
                    + self.ctx.rgroup * 64,
                    fx.Int64,
                    self.traits.NVOPS,
                )
                corr_b = fx.Vector.from_elements(
                    [corr_reg], dtype=fx.Float32
                ).broadcast_to(self.traits.OP_ELEMS)
                for vh in range_constexpr(self.traits.VHE_CHUNKS):
                    v_vh = v_vh_shared[vh]
                    op = self.gemm.pv(v_vh, p_ops) * fx.Vector.from_elements(
                        [v_max_scaled], dtype=fx.Float32
                    ).broadcast_to(self.traits.OP_ELEMS)
                    o_acc[vh] = (
                        op
                        if const_expr(self.traits.single_tile_plan)
                        else o_acc[vh] * corr_b + op
                    )
                    if const_expr(
                        self.traits.MTP4_PREFETCH_V
                        and m == self.traits.M_TILES - 1
                        and not self.traits.single_tile_plan
                    ):
                        # Reuse the dead V chunk while the final PV finishes.
                        v_next_chunk = fx.Vector.filled(self.traits.NVOPS, 0, fx.Int64)
                        if (
                            const_expr(not self.traits.single_tile_plan)
                            and tt1 < self.ctx.part_end
                        ):
                            v_next_chunk = fx.Vector.from_elements(
                                self.kv.load_v(v_page_next, vh), dtype=fx.Int64
                            )
                        v_next_chunks.append(v_next_chunk)
                next_state.extend([*o_acc, m_new, l_new])
                if const_expr(m < self.traits.M_TILES - 1):
                    fx.rocdl.sched_barrier(0)
            if const_expr(self.traits.MTP4_PREFETCH_V):
                if const_expr(self.traits.single_tile_plan):
                    # Preserve the unused next-V slot in the loop state.
                    v_next = ostate[self.traits.V_DATA_SLOT]
                else:
                    v_next = fx.Vector.from_elements(
                        [
                            v_next_chunks[vh][i]
                            for vh in range_constexpr(self.traits.VHE_CHUNKS)
                            for i in range_constexpr(self.traits.NVOPS)
                        ],
                        dtype=fx.Int64,
                    )
        else:
            for m in range_constexpr(self.traits.M_TILES):
                p_base = (
                    self.traits.sP_off
                    + (m % self.traits.P_BUFFERS) * MFMA_MNK * self.traits.SP_ROW_BYTES
                )
                lsum_base = (
                    self.traits.sLsum_off
                    + (m % self.traits.P_BUFFERS)
                    * MFMA_MNK
                    * self.traits.NWARP_PAD
                    * self.traits.f32
                )
                o_acc = [
                    ostate[self.traits.o_slot(m, vh)]
                    for vh in range_constexpr(self.traits.VHE_CHUNKS)
                ]
                m_prev = ostate[
                    self.traits.m_slot(m)
                ]  # this thread's own running max, carried from last tile
                l_prev = ostate[
                    self.traits.l_slot(m)
                ]  # this thread's own running denom, carried from last tile

                masked_chunks = masked_chunks_saved[m]

                v_max_scaled = None
                norm_factor_b = None
                if const_expr(self.traits.per_token_kv):
                    v_max_scaled, norm_factor_b = self.softmax.v_scale_normalization()

                m_new, corr_reg = self.softmax.pack_multi(
                    masked_chunks,
                    m,
                    m_prev,
                    p_base,
                    lsum_base,
                    v_scale_shared,
                    cur_kv_buf,
                    norm_factor_b,
                )
                gpu.barrier()
                gsum = self.lds.load_wave(lsum_base, self.ctx.lane16).reduce(
                    ReductionOp.ADD
                )
                l_new = (
                    gsum
                    if const_expr(self.traits.single_tile_plan)
                    else l_prev * corr_reg + gsum
                )

                p_ops = self.lds.load(
                    p_base
                    + self.ctx.lane16 * self.traits.SP_ROW_BYTES
                    + self.ctx.rgroup * 64,
                    fx.Int64,
                    self.traits.NVOPS,
                )

                corr_b = fx.Vector.from_elements(
                    [corr_reg], dtype=fx.Float32
                ).broadcast_to(self.traits.OP_ELEMS)
                for vh in range_constexpr(self.traits.VHE_CHUNKS):
                    v_vh = v_vh_shared[vh]
                    op = self.gemm.pv(v_vh, p_ops)
                    if const_expr(self.traits.per_token_kv):
                        op = op * fx.Vector.from_elements(
                            [v_max_scaled], dtype=fx.Float32
                        ).broadcast_to(self.traits.OP_ELEMS)
                    o_acc[vh] = (
                        op
                        if const_expr(self.traits.single_tile_plan)
                        else o_acc[vh] * corr_b + op
                    )
                next_state.extend([*o_acc, m_new, l_new])
                # Single P/Lsum slots need a read-retirement barrier.
                # Alternating slots use the next write barrier; Phase A
                # protects loop boundaries. The fence bounds live registers.
                if const_expr(m < self.traits.M_TILES - 1):
                    if const_expr(self.traits.P_BUFFERS == 1):
                        gpu.barrier()
                    fx.rocdl.sched_barrier(0)
        if const_expr(self.traits.MTP4_PREFETCH_V):
            next_state.append(v_next)
        return next_state

    @flyc.jit
    def single_query_tile(
        self,
        ostate,
        k_cur,
        v_page_cur,
        tt1,
        tile_valid,
        window_left,
        cur_kv_buf,
        scale_bounds,
        v_vh_shared,
        q_ops_all,
    ):
        """Overlap QK, page/scale prefetch and PV for one padded query tile."""
        next_state = [None, None]
        o_acc = [
            ostate[self.traits.o_slot(0, vh)]
            for vh in range_constexpr(self.traits.VHE_CHUNKS)
        ]
        m_prev = ostate[self.traits.m_slot(0)]  # running max, carried from last tile
        l_prev = ostate[self.traits.l_slot(0)]  # running denom, carried from last tile
        frag_Ss = self.gemm.qk(k_cur, q_ops_all, 0)
        k_next = k_cur
        if const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end:
            if const_expr(self.traits.REUSE_KV_PAGES):
                # Stage scales before K so scale waits cannot drain K;
                # page16 defers K until the P barrier.
                phys_vec1 = self.kv.fetch_v_pages(tt1)
                if const_expr(self.traits.per_token_kv):
                    self.kv.stage_scales(phys_vec1, self.kv.scale_buffer_offset(tt1))
                if const_expr(not self.traits.PAGE16_VPIPE):
                    k_next = self.kv.load_k_from_pages(phys_vec1)
            else:
                k_next, phys_vec1 = self.kv.load_k(tt1)
                self.kv.fetch_v_pages(tt1)
                if const_expr(self.traits.per_token_kv):
                    self.kv.stage_scales(phys_vec1, self.kv.scale_buffer_offset(tt1))
        if const_expr(self.traits.SCALES_BEFORE_CURRENT_V):
            # Issue V after scale staging so scale waits do not drain V.
            v_vh_shared = [
                self.kv.load_v(v_page_cur, vh)
                for vh in range_constexpr(self.traits.VHE_CHUNKS)
            ]
        scale = (
            self.ctx.scale_qk
            if const_expr(self.traits.SCALAR_FP8_DECODE)
            else self.ctx.scale_qk
            * self.lds.load_scalar(self.traits.sQscale_off, self.ctx.lane16)
        )  # per-qhead positive score scale
        masked_chunks, v_scale_vecs = self.softmax.single_scores(
            frag_Ss, scale, tile_valid, window_left, cur_kv_buf, scale_bounds
        )
        gpu.barrier()
        v_page_next = v_page_cur
        if const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end:
            v_page_next = self.kv.read_v_pages()
        next_state[self.traits.V_SLOT] = v_page_next
        v_max_scaled = None
        norm_factor_b = None
        if const_expr(self.traits.per_token_kv):
            v_max_scaled, norm_factor_b = self.softmax.v_scale_normalization()
        m_new, corr_reg = self.softmax.pack_single(
            masked_chunks, m_prev, scale, v_scale_vecs, cur_kv_buf, norm_factor_b
        )
        gpu.barrier()
        if const_expr(self.traits.PAGE16_VPIPE):  # noqa: SIM102
            # Next K can now overlap P reads/PV.
            if const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end:
                k_next = self.kv.load_k_from_pages(self.kv.read_k_pages())
        next_state[self.traits.K_SLOT] = k_next
        gsum = self.lds.load_wave(self.traits.sLsum_off, self.ctx.lane16).reduce(
            ReductionOp.ADD
        )
        l_new = (
            gsum
            if const_expr(self.traits.single_tile_plan)
            else l_prev * corr_reg + gsum
        )
        p_ops = self.lds.load(
            self.traits.sP_off
            + self.ctx.lane16 * self.traits.SP_ROW_BYTES
            + self.ctx.rgroup * 64,
            fx.Int64,
            self.traits.NVOPS,
        )
        corr_b = fx.Vector.from_elements([corr_reg], dtype=fx.Float32).broadcast_to(
            self.traits.OP_ELEMS
        )
        if const_expr(self.traits.prefetch_v):
            v_vh_batch = v_vh_shared
        else:
            v_vh_batch = [
                self.kv.load_v(v_page_cur, vh)
                for vh in range_constexpr(self.traits.VHE_CHUNKS)
            ]
        for vh in range_constexpr(self.traits.VHE_CHUNKS):
            v_vh = v_vh_batch[vh]
            op = self.gemm.pv(v_vh, p_ops)
            if const_expr(self.traits.per_token_kv):
                op = op * fx.Vector.from_elements(
                    [v_max_scaled], dtype=fx.Float32
                ).broadcast_to(self.traits.OP_ELEMS)
            o_acc[vh] = (
                op
                if const_expr(self.traits.single_tile_plan)
                else o_acc[vh] * corr_b + op
            )
        next_state.extend([*o_acc, m_new, l_new])
        return next_state

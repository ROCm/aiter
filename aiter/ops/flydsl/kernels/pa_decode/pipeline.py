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
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import ReductionOp

from ..utils import rcp_f32
from .op_epilog import PaDecodeEpilogue
from .op_gemm import Gfx1250Gemm, PaDecodeGemm
from .op_lds import Gfx1250Memory, PaDecodeKVLoader, PaDecodeLds, PaDecodeQueryLoader
from .op_softmax import PaDecodeSoftmax
from .traits import KV_COMPUTE_BLOCK, LOG2E, MFMA_MNK


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
        if const_expr(self.traits.stage_page_capacity):
            self.kv.stage_page_ids()
            gpu.barrier()
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
        o_zero = fx.full(self.traits.OP_ELEMS, 0.0, fx.Float32)
        init_state = [k_pf0, v_page_pf0]
        for _m in range_constexpr(self.traits.M_TILES):
            init_state.extend(
                [o_zero] * self.traits.VHE_CHUNKS + [self.ctx.NEG_INF, self.ctx.ZERO_F]
            )
        # Reuse carried V registers only after the final query consumes a chunk.
        if const_expr(self.traits.MTP4_PREFETCH_V):
            v_pf0 = fx.full(self.traits.VHE_CHUNKS * self.traits.NVOPS, 0, fx.Int64)
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
        # The d256 pipe keeps its M-tiles on the single-tile schedule.
        multi_query = self.traits.M_TILES > 1 and not self.traits.D256_PIPE
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
                (not self.traits.per_token_kv and multi_query)
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
            if const_expr(self.traits.M_TILES > 1 and not self.traits.is_bf16_kv):
                q_scale_vec = self.lds.load(
                    self.traits.sQscale_off
                    + self.ctx.lane16 * (self.traits.M_TILES * self.traits.f32),
                    fx.Float32,
                    self.traits.M_TILES,
                )

            # All query tiles share one K/V tile; each path retains its publication barriers.
            if const_expr(multi_query):
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
                    q_scale_vec,
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

            scale = (
                self.ctx.scale_qk
                if const_expr(self.traits.is_bf16_kv)
                else self.ctx.scale_qk * fx.Float32(q_scale_vec[m])
            )
            masked_chunks = self.softmax.multi_scores(
                frag_Ss, scale, k_scale_shared, tile_valid, window_left, m
            )

            self.softmax.stage_row_max(
                masked_chunks, self.softmax.lmax_offset(m), scale, True
            )

            masked_chunks_saved[m] = masked_chunks

        # Publish next pages/scales with the Phase A barrier.
        k_next = (
            fx.full(self.traits.NCHUNK * self.traits.N_SUBCHUNKS, 0, fx.Int64)
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
            if const_expr(not self.traits.MTP4_PREFETCH_V and self.traits.trans_v):
                # Overlap current V with next K and the P barrier.
                v_vh_shared = [
                    self.kv.load_v(v_page_cur, vh)
                    for vh in range_constexpr(self.traits.VHE_CHUNKS)
                ]
            if const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end:
                # Saved scores are dead; next K can overlap the P barrier/PV.
                k_next = self.kv.load_k_from_pages(self.kv.read_k_pages())
            next_state[self.traits.K_SLOT] = k_next

            gpu.barrier()

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
                    + self.ctx.rgroup * 64 * self.traits.QP_ELEM_BYTES,
                    fx.Int64,
                    self.traits.NVOPS,
                )
                corr_b = fx.full(self.traits.OP_ELEMS, fx.Float32(corr_reg), fx.Float32)
                for vh in range_constexpr(self.traits.VHE_CHUNKS):
                    v_vh = v_vh_shared[vh]
                    op = self.gemm.pv(v_vh, p_ops) * fx.full(
                        self.traits.OP_ELEMS, v_max_scaled, fx.Float32
                    )
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
                        v_next_chunk = fx.full(self.traits.NVOPS, 0, fx.Int64)
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
                    + self.ctx.rgroup * 64 * self.traits.QP_ELEM_BYTES,
                    fx.Int64,
                    self.traits.NVOPS,
                )

                corr_b = fx.full(self.traits.OP_ELEMS, fx.Float32(corr_reg), fx.Float32)
                for vh in range_constexpr(self.traits.VHE_CHUNKS):
                    v_vh = v_vh_shared[vh]
                    op = self.gemm.pv(v_vh, p_ops)
                    if const_expr(self.traits.per_token_kv):
                        op = op * fx.full(
                            self.traits.OP_ELEMS, v_max_scaled, fx.Float32
                        )
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
        q_scale_vec,
    ):
        """Overlap QK, page/scale prefetch and PV for up to two padded query tiles.

        Every M-tile reuses one K tile and one V gather, owns private P and Lsum
        slots, and keeps its own accumulator chain. The tile's memory-op sequence
        is independent of M_TILES, so the vmcnt counts below hold for any M.
        """
        next_state = [None, None]
        o_acc = [
            [
                ostate[self.traits.o_slot(m, vh)]
                for vh in range_constexpr(self.traits.VHE_CHUNKS)
            ]
            for m in range_constexpr(self.traits.M_TILES)
        ]
        # Running max and denominator, carried from the last tile.
        m_prev = [
            ostate[self.traits.m_slot(m)] for m in range_constexpr(self.traits.M_TILES)
        ]
        l_prev = [
            ostate[self.traits.l_slot(m)] for m in range_constexpr(self.traits.M_TILES)
        ]
        frag_Ss = [
            self.gemm.qk(k_cur, q_ops_all, m)
            for m in range_constexpr(self.traits.M_TILES)
        ]
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
        scales = []
        masked_chunks = []
        v_scale_vecs = None
        for m in range_constexpr(self.traits.M_TILES):
            if const_expr(self.traits.is_bf16_kv):
                scale = self.ctx.scale_qk
            elif const_expr(self.traits.M_TILES > 1):
                scale = self.ctx.scale_qk * fx.Float32(q_scale_vec[m])
            else:
                scale = self.ctx.scale_qk * self.lds.load_scalar(
                    self.traits.sQscale_off, self.ctx.lane16
                )
            scales.append(scale)
            chunks, v_scale_vecs = self.softmax.single_scores(
                frag_Ss[m], scale, tile_valid, window_left, cur_kv_buf, scale_bounds, m
            )
            masked_chunks.append(chunks)
        gpu.barrier()
        v_page_next = v_page_cur
        if const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end:
            v_page_next = self.kv.read_v_pages()
        next_state[self.traits.V_SLOT] = v_page_next
        v_max_scaled = None
        norm_factor_b = None
        if const_expr(self.traits.per_token_kv):
            v_max_scaled, norm_factor_b = self.softmax.v_scale_normalization()
        m_new = []
        corr_reg = []
        for m in range_constexpr(self.traits.M_TILES):
            m_new_m, corr_m = self.softmax.pack_single(
                masked_chunks[m],
                m_prev[m],
                scales[m],
                v_scale_vecs,
                cur_kv_buf,
                norm_factor_b,
                m,
            )
            m_new.append(m_new_m)
            corr_reg.append(corr_m)
        gpu.barrier()
        if const_expr(self.traits.PAGE16_VPIPE):  # noqa: SIM102
            # Next K can now overlap P reads/PV.
            if const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end:
                k_next = self.kv.load_k_from_pages(self.kv.read_k_pages())
        next_state[self.traits.K_SLOT] = k_next
        l_new = []
        p_ops = []
        corr_b = []
        for m in range_constexpr(self.traits.M_TILES):
            gsum = self.lds.load_wave(
                self.softmax.lsum_offset(m), self.ctx.lane16
            ).reduce(ReductionOp.ADD)
            l_new.append(
                gsum
                if const_expr(self.traits.single_tile_plan)
                else l_prev[m] * corr_reg[m] + gsum
            )
            p_ops.append(
                self.lds.load(
                    self.softmax.p_offset(m)
                    + self.ctx.lane16 * self.traits.SP_ROW_BYTES
                    + self.ctx.rgroup * 64 * self.traits.QP_ELEM_BYTES,
                    fx.Int64,
                    self.traits.NVOPS,
                )
            )
            corr_b.append(
                fx.full(self.traits.OP_ELEMS, fx.Float32(corr_reg[m]), fx.Float32)
            )
        if const_expr(self.traits.prefetch_v):
            v_vh_batch = v_vh_shared
        else:
            v_vh_batch = [
                self.kv.load_v(v_page_cur, vh)
                for vh in range_constexpr(self.traits.VHE_CHUNKS)
            ]
        # vh outer, m inner: each M-tile is a separate accumulator chain on the
        # same V operand.
        for vh in range_constexpr(self.traits.VHE_CHUNKS):
            v_vh = v_vh_batch[vh]
            for m in range_constexpr(self.traits.M_TILES):
                op = self.gemm.pv(v_vh, p_ops[m])
                if const_expr(self.traits.per_token_kv):
                    op = op * fx.full(self.traits.OP_ELEMS, v_max_scaled, fx.Float32)
                o_acc[m][vh] = (
                    op
                    if const_expr(self.traits.single_tile_plan)
                    else o_acc[m][vh] * corr_b[m] + op
                )
        for m in range_constexpr(self.traits.M_TILES):
            next_state.extend([*o_acc[m], m_new[m], l_new[m]])
        return next_state


class Gfx1250Bf16Pipeline:
    """Four waves share packed BF16 pages and retain Q/PV operands."""

    def __init__(self, ctx):
        self.ctx, self.t = ctx, ctx.traits
        self.mem = Gfx1250Memory(ctx)

    @flyc.jit
    def run(self):
        ctx, t = self.ctx, self.t
        ctx.tid = fx.Int32(gpu.thread_id("x"))
        ctx.warp, ctx.lane = ctx.tid // t.WAVE, ctx.tid % t.WAVE
        ctx.lane16, ctx.half = ctx.lane % 16, ctx.lane // 16
        ctx.kv_h = fx.Int32(gpu.block_id("y")) // t.query_splits
        ctx.query_begin = (fx.Int32(gpu.block_id("y")) % t.query_splits) * (
            t.query_length // t.query_splits
        )
        ctx.n_kv = fx.Int32(gpu.grid_dim.y) // t.query_splits
        self.mem.allocate()
        query = fx.recast_iter(t.q_dtype, ctx.query_ptr)
        q_operands = []
        initial = []
        for m in range_constexpr(t.m_tiles):
            row = m * 16 + ctx.lane16
            qbase = (
                fx.Int64(ctx.planned_seq) * t.query_length
                + ctx.query_begin
                + row // t.query_group_size
            ) * ctx.stride_q_row + (
                ctx.kv_h * t.query_group_size + row % t.query_group_size
            ) * ctx.stride_q_head
            operands = []
            for d in range_constexpr(t.head_dim // 32):
                values = []
                for j in range_constexpr(2):
                    fragment = fx.Vector.filled(8, 0.0, fx.BFloat16)
                    if row < t.rows:
                        fragment = fx.Vector(
                            fx.ptr_load(
                                fx.add_offset(
                                    query, qbase + d * 32 + ctx.half * 8 + j * 16
                                ),
                                result_type=fx.Vector.make_type(8, t.q_dtype),
                            )
                        ).to(fx.BFloat16)
                    values.extend([fragment[i] for i in range_constexpr(8)])
                operands.append(fx.Vector.from_elements(values, dtype=fx.BFloat16))
            q_operands.append(operands)
            initial.extend([fx.Float32(float("-inf")), fx.Float32(0.0)])
            initial.extend(
                [
                    fx.Vector.filled(8, 0.0, fx.Float32)
                    for _ in range_constexpr(t.head_dim // 64)
                ]
            )
        state_stride = 2 + t.head_dim // 64
        begin, end = (
            ctx.planned_start * KV_COMPUTE_BLOCK,
            ctx.planned_end * KV_COMPUTE_BLOCK,
        )
        self.mem.stage_bf16_page(begin, fx.Int32(0))
        for token_i, state in range(begin, end, 128, init=initial):
            token_base = fx.Int32(token_i)
            buffer = ((token_base - begin) // 128) % 2
            tdm_ops.tensor_wait(0)
            gpu.barrier()
            kv_off = t.q_bytes + buffer * t.kv_stride
            if token_base + 128 < end:
                self.mem.stage_bf16_page(token_base + 128, 1 - buffer)
            next_state = []
            for m in range_constexpr(t.m_tiles):
                row = m * 16 + ctx.lane16
                upper = (
                    ctx.planned_context
                    - t.query_length
                    + ctx.query_begin
                    + row // t.query_group_size
                    + 1
                )
                units = []
                for n in range_constexpr(2):
                    scores = fx.Vector.filled(8, 0.0, fx.Float32)
                    for d in range_constexpr(t.head_dim // 32):
                        k = self.mem.bf16_k_operand(
                            kv_off,
                            ctx.warp * 32 + n * 16,
                            d * 32,
                            ctx.lane16,
                            ctx.half,
                        )
                        scores = Gfx1250Gemm.mma(k, q_operands[m][d], scores, 32, True)
                    for r in range_constexpr(8):
                        token = token_base + ctx.warp * 32 + n * 16 + ctx.half * 8 + r
                        visible = (token < upper) & (row < t.rows)
                        if const_expr(t.sliding_window > 0):
                            visible = visible & (token >= upper - t.sliding_window)
                        units.append(
                            visible.select(
                                scores[r] * fx.Float32(t.softmax_scale * LOG2E),
                                fx.Float32(float("-inf")),
                            )
                        )
                scores = fx.Vector.from_elements(units, dtype=fx.Float32)
                tile_max = scores.reduce(ReductionOp.MAX)
                tile_max = fx.maxnumf(tile_max, tile_max.shuffle_xor(16, t.WAVE))
                if ctx.half == 0:
                    self.mem.store(
                        t.stats_offset + (ctx.lane16 * 4 + ctx.warp) * 4,
                        tile_max,
                        fx.Float32,
                    )
                gpu.barrier()
                tile_max = self.mem.load(
                    t.stats_offset + ctx.lane16 * 16, fx.Float32, 4
                ).reduce(ReductionOp.MAX)
                old_max = state[m * state_stride]
                new_max = fx.maxnumf(old_max, tile_max)
                safe_max = (new_max > fx.Float32(float("-inf"))).select(
                    new_max, fx.Float32(0.0)
                )
                corr = fx.Float32(fx.exp2(old_max - safe_max, fastmath="afn"))
                probs = fx.Vector(
                    fx.exp2(
                        scores
                        - fx.Vector.from_elements(
                            [safe_max], dtype=fx.Float32
                        ).broadcast_to(16),
                        fastmath="afn",
                    )
                )
                tile_sum = probs.reduce(ReductionOp.ADD)
                tile_sum = tile_sum + tile_sum.shuffle_xor(16, t.WAVE)
                if ctx.half == 0:
                    self.mem.store(
                        t.stats_offset + 256 + (ctx.lane16 * 4 + ctx.warp) * 4,
                        tile_sum,
                        fx.Float32,
                    )
                for n in range_constexpr(2):
                    p = fx.Vector.from_elements(
                        [probs[n * 8 + r] for r in range_constexpr(8)], dtype=fx.Float32
                    ).to(fx.BFloat16)
                    self.mem.store(
                        t.p_offset
                        + ctx.lane16 * t.v_stride
                        + (ctx.warp * 32 + n * 16 + ctx.half * 8) * 2,
                        p,
                        fx.BFloat16,
                    )
                gpu.barrier()
                denom = state[m * state_stride + 1] * corr + self.mem.load(
                    t.stats_offset + 256 + ctx.lane16 * 16, fx.Float32, 4
                ).reduce(ReductionOp.ADD)
                next_state.extend([new_max, denom])
                p_operands = [
                    self.mem.operand(
                        t.p_offset + d * 64, t.v_stride, 32, ctx.lane16, ctx.half
                    )
                    for d in range_constexpr(4)
                ]
                for vh in range_constexpr(t.head_dim // 64):
                    # Rescale the running accumulator before PV's fused adds.
                    out = state[m * state_stride + 2 + vh] * fx.Vector.from_elements(
                        [corr],
                        dtype=fx.Float32,
                    ).broadcast_to(8)
                    for d in range_constexpr(4):
                        v = self.mem.bf16_v_operand(
                            kv_off + t.k_bytes,
                            vh * 64 + ctx.warp * 16,
                            d * 32,
                            ctx.lane16,
                            ctx.half,
                        )
                        out = Gfx1250Gemm.mma(v, p_operands[d], out, 32, True)
                    next_state.append(out)
                if const_expr(m + 1 < t.m_tiles):
                    # Retire P reads before another query tile overwrites them.
                    gpu.barrier()
            final = yield next_state
        pout = fx.recast_iter(t.q_dtype, ctx.pout_ptr)
        pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
        psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
        slot = fx.Int64(ctx.kv_h) * fx.Int64(gpu.grid_dim.x) + fx.Int64(
            gpu.block_id("x")
        )
        for m in range_constexpr(t.m_tiles):
            row = m * 16 + ctx.lane16
            output_row = ctx.query_begin * t.query_group_size + row
            partial_row = slot * (t.rows * t.query_splits) + output_row
            denom = final[m * state_stride + 1]
            inv = fx.Float32(rcp_f32((denom > 0.0).select(denom, fx.Float32(1.0))))
            if row < t.rows:
                for vh in range_constexpr(t.head_dim // 64):
                    d = vh * 64 + ctx.warp * 16 + ctx.half * 8
                    value = (
                        final[m * state_stride + 2 + vh]
                        * fx.Vector.from_elements(
                            [inv],
                            dtype=fx.Float32,
                        ).broadcast_to(8)
                    ).to(t.q_dtype)
                    offset = (
                        (
                            fx.Int64(ctx.planned_seq) * t.query_length
                            + output_row // t.query_group_size
                        )
                        * ctx.stride_q_row
                        + (
                            ctx.kv_h * t.query_group_size
                            + output_row % t.query_group_size
                        )
                        * ctx.stride_q_head
                        + d
                        if t.direct_output
                        else partial_row * t.head_dim + d
                    )
                    fx.ptr_store(value, fx.add_offset(pout, offset))
                if const_expr(not t.direct_output):  # noqa: SIM102
                    if ctx.warp == 0 and ctx.half == 0:
                        pmax[partial_row] = final[m * state_stride] * fx.Float32(
                            1.0 / LOG2E
                        )
                        psum[partial_row] = denom


class Gfx1250Pipeline:
    def __init__(self, ctx):
        self.ctx, self.t = ctx, ctx.traits
        self.mem = Gfx1250Memory(ctx)

    @flyc.jit
    def wait_oldest_tile(self, later):
        """Retire the oldest staged tile while ``later`` staged tiles stay in flight.

        TDM transfers complete in issue order, and each staged BF16 page16 tile
        holds at least two per-wave transfers (K, then V in one or two chunks).
        """
        if const_expr(self.t.buffers == 3):
            if later > 0:
                tdm_ops.tensor_wait(2)
            else:
                tdm_ops.tensor_wait(0)
        else:
            if later > 1:
                tdm_ops.tensor_wait(4)
            elif later > 0:
                tdm_ops.tensor_wait(2)
            else:
                tdm_ops.tensor_wait(0)

    @flyc.jit
    def run(self):
        ctx, t = self.ctx, self.t
        ctx.tid = fx.Int32(gpu.thread_id("x"))
        ctx.warp, ctx.lane = ctx.tid // t.WAVE, ctx.tid % t.WAVE
        ctx.lane16, ctx.half = ctx.lane % 16, ctx.lane // 16
        ctx.kv_h = fx.Int32(gpu.block_id("y")) // t.query_splits
        ctx.query_begin = (fx.Int32(gpu.block_id("y")) % t.query_splits) * (
            t.query_length // t.query_splits
        )
        ctx.n_kv = fx.Int32(gpu.grid_dim.y) // t.query_splits
        self.mem.allocate()
        if const_expr(t.q_tiles == t.m_tiles):
            self.mem.stage_query()
        begin, end = (
            ctx.planned_start * KV_COMPUTE_BLOCK,
            ctx.planned_end * KV_COMPUTE_BLOCK,
        )
        self.mem.stage_kv(begin, fx.Int32(0))
        # Rings deeper than two buffers stage buffers - 1 tiles ahead.
        for ahead in range_constexpr(1, t.buffers - 1):
            if begin + ahead * t.TOKENS < end:
                self.mem.stage_kv(begin + ahead * t.TOKENS, fx.Int32(ahead))
        # Per-M-tile state: maximum, denominator, then wave-owned D fragments.
        vh_count = t.head_dim // 64
        initial = []
        for m in range_constexpr(t.m_tiles):
            initial.extend([fx.Float32(float("-inf")), fx.Float32(0.0)])
            initial.extend(
                [
                    fx.Vector.filled(8, 0.0, fx.Float32)
                    for vh in range_constexpr(vh_count)
                ]
            )
        state_stride = 2 + vh_count
        if const_expr(t.packed_lds):
            initial.append(fx.Float32(1.0))
        resident_q = (
            not t.bf16
            and (t.head_dim == 128 or t.packed_lds)
            and (t.m_tiles == 1 or t.fp8_multi_tile)
        )
        if const_expr(resident_q):
            gpu.barrier()
            q_operands, q_scales = [], []
            for m in range_constexpr(t.m_tiles):
                q_operands.append(
                    [
                        self.mem.operand(
                            m * 16 * t.q_stride + d * 128,
                            t.q_stride,
                            128,
                            ctx.lane16,
                            ctx.half,
                        )
                        for d in range_constexpr(t.head_dim // 128)
                    ]
                )
                q_scales.append(
                    fx.Float32(
                        self.mem.load(
                            (m * 16 + ctx.lane16) * t.q_stride + t.head_dim,
                            fx.Float32,
                        )[0]
                    )
                )
        for token_i, state in range(begin, end, t.TOKENS, init=initial):
            token_base = fx.Int32(token_i)
            buffer = ((token_base - begin) // t.TOKENS) % t.buffers
            if const_expr(t.use_tdm):
                if const_expr(t.buffers > 2):
                    self.wait_oldest_tile((end - token_base) // t.TOKENS - 1)
                else:
                    tdm_ops.tensor_wait(0)
                if const_expr(t.async_page or t.page16_async_scales):
                    fx.rocdl.s_wait_asynccnt(0)
            else:
                fx.rocdl.s_wait_asynccnt(0)
            gpu.barrier()
            kv_off = t.q_bytes + buffer * t.kv_stride
            if const_expr(t.buffers == 2):  # noqa: SIM102
                if token_base + t.TOKENS < end:
                    self.mem.stage_kv(token_base + t.TOKENS, 1 - buffer)
            if const_expr(t.buffers > 2):  # noqa: SIM102
                # The barrier above retired the previous tile, whose buffer
                # this refills.
                if token_base + (t.buffers - 1) * t.TOKENS < end:
                    self.mem.stage_kv(
                        token_base + (t.buffers - 1) * t.TOKENS,
                        (buffer + t.buffers - 1) % t.buffers,
                    )
            vnorm = fx.Float32(1.0)
            vfactor = fx.Float32(1.0 if t.bf16 else 1.0 / t.FP8_MAX)
            if const_expr(t.per_token_kv):
                # Every wave reduces the tile's scales, avoiding a CTA barrier.
                scale_base = kv_off + t.k_bytes + t.v_bytes + t.TOKENS * 4
                vmax = fx.maxnumf(
                    fx.Float32(self.mem.load(scale_base + ctx.lane * 4, fx.Float32)[0]),
                    fx.Float32(
                        self.mem.load(scale_base + (ctx.lane + 32) * 4, fx.Float32)[0]
                    ),
                )
                if const_expr(t.TOKENS == 128):
                    vmax = fx.maxnumf(
                        vmax,
                        fx.Float32(
                            self.mem.load(scale_base + (ctx.lane + 64) * 4, fx.Float32)[
                                0
                            ]
                        ),
                    )
                    vmax = fx.maxnumf(
                        vmax,
                        fx.Float32(
                            self.mem.load(scale_base + (ctx.lane + 96) * 4, fx.Float32)[
                                0
                            ]
                        ),
                    )
                for shift in (16, 8, 4, 2, 1):
                    vmax = fx.maxnumf(vmax, vmax.shuffle_xor(shift, t.WAVE))
                vfactor = vmax * fx.Float32(1.0 / t.FP8_MAX)
                if const_expr(t.packed_lds):
                    # Empty planned tails carry no new mass. Keep the prior
                    # scale so their zero V contribution preserves the output.
                    vfactor = (vfactor > 0.0).select(vfactor, fx.Float32(state[-1]))
                vnorm = fx.Float32(rcp_f32(fx.maxnumf(vfactor, fx.Float32(1e-20))))
            elif const_expr(not t.bf16):
                vs = fx.recast_iter(fx.Float32, ctx.value_scale_ptr)
                vfactor = fx.Float32(vs[0]) * fx.Float32(1.0 / t.FP8_MAX)
                if const_expr(t.packed_lds):
                    # The fused accumulator carries unscaled V until epilogue.
                    # A zero scalar V scale must produce zero without 0 * inf.
                    vnorm = fx.Float32(
                        rcp_f32((vfactor != 0.0).select(vfactor, fx.Float32(1.0)))
                    )
            # All query tiles read the same chunk. Retain its WMMA
            # operands instead of reloading K/V from LDS for each tile.
            if const_expr(t.fp8_multi_tile):
                packed_k_operands = [
                    [
                        self.mem.packed_k_operand(kv_off, n, d)
                        for d in range_constexpr(t.head_dim // 128)
                    ]
                    for n in range_constexpr(t.TOKENS // 64)
                ]
                page_v_operands = self.mem.page_v_operands(kv_off)
            next_state = []
            for m in range_constexpr(t.m_tiles):
                if const_expr(t.q_tiles != t.m_tiles):
                    self.mem.stage_query(m)
                    gpu.barrier()
                q_tile = m if t.q_tiles == t.m_tiles else 0
                row = m * 16 + ctx.lane16
                qscale = fx.Float32(1.0)
                if const_expr(resident_q):
                    qscale = q_scales[m]
                elif const_expr(not t.bf16):
                    qscale = fx.Float32(
                        self.mem.load(
                            (q_tile * 16 + ctx.lane16) * t.q_stride + t.head_dim,
                            fx.Float32,
                        )[0]
                    )
                scale = qscale * fx.Float32(t.softmax_scale * LOG2E)
                if const_expr(not t.per_token_kv and not t.bf16):
                    ks = fx.recast_iter(fx.Float32, ctx.key_scale_ptr)
                    scale = scale * fx.Float32(ks[0])
                upper = (
                    ctx.planned_context
                    - t.query_length
                    + ctx.query_begin
                    + row // t.query_group_size
                    + 1
                )
                units, values = [], []
                qk_k = 32 if t.bf16 else (128 if t.head_dim % 128 == 0 else 64)
                for n in range_constexpr(t.TOKENS // 64):
                    scores = fx.Vector.filled(8, 0.0, fx.Float32)
                    for d in range_constexpr(t.head_dim // qk_k):
                        if const_expr(t.fp8_multi_tile):
                            a = packed_k_operands[n][d]
                        elif const_expr(t.packed_lds):
                            a = self.mem.packed_k_operand(kv_off, n, d)
                        elif const_expr(t.bf16_page16_packed):
                            a = self.mem.bf16_k_operand(
                                kv_off + ctx.warp * t.head_dim * 32,
                                0,
                                d * qk_k,
                                ctx.lane16,
                                ctx.half,
                            )
                        else:
                            a = self.mem.operand(
                                kv_off
                                + ctx.warp * 16 * t.q_stride
                                + d * qk_k * t.elem_bytes,
                                t.q_stride,
                                qk_k,
                                ctx.lane16,
                                ctx.half,
                            )
                        if const_expr(resident_q):
                            b = q_operands[m][d]
                        else:
                            b = self.mem.operand(
                                q_tile * 16 * t.q_stride + d * qk_k * t.elem_bytes,
                                t.q_stride,
                                qk_k,
                                ctx.lane16,
                                ctx.half,
                            )
                        scores = Gfx1250Gemm.mma(a, b, scores, qk_k, t.bf16)
                    if const_expr(t.per_token_kv):
                        scale_token = ctx.warp * (t.TOKENS // 4) + n * 16 + ctx.half * 8
                        ks = self.mem.load(
                            kv_off + t.k_bytes + t.v_bytes + scale_token * 4,
                            fx.Float32,
                            8,
                        )
                        vs = self.mem.load(
                            kv_off
                            + t.k_bytes
                            + t.v_bytes
                            + (t.TOKENS + scale_token) * 4,
                            fx.Float32,
                            8,
                        )
                        values.extend([vs[r] for r in range_constexpr(8)])
                    for r in range_constexpr(8):
                        token = ctx.warp * (t.TOKENS // 4) + n * 16 + ctx.half * 8 + r
                        value = scores[r] * scale
                        if const_expr(t.per_token_kv):
                            value = value * fx.Float32(ks[r])
                        visible = (token_base + token < upper) & (row < t.rows)
                        if const_expr(t.sliding_window > 0):
                            visible = visible & (
                                token_base + token >= upper - t.sliding_window
                            )
                        units.append(visible.select(value, fx.Float32(float("-inf"))))
                scores = fx.Vector.from_elements(units, dtype=fx.Float32)
                if const_expr(t.per_token_kv):
                    vscales = fx.Vector.from_elements(values, dtype=fx.Float32)
                local_max = scores.reduce(ReductionOp.MAX)
                local_max = fx.maxnumf(local_max, local_max.shuffle_xor(16, t.WAVE))
                if ctx.half == 0:
                    self.mem.store(
                        t.stats_offset + (ctx.lane16 * t.NWARP + ctx.warp) * 4,
                        local_max,
                        fx.Float32,
                    )
                gpu.barrier()
                tile_max = self.mem.load(
                    t.stats_offset + ctx.lane16 * t.NWARP * 4, fx.Float32, t.NWARP
                ).reduce(ReductionOp.MAX)
                old_max = state[m * state_stride]
                new_max = fx.maxnumf(old_max, tile_max)
                safe_max = (new_max > fx.Float32(float("-inf"))).select(
                    new_max, fx.Float32(0.0)
                )
                # Approximate exponentials avoid the library underflow fixup.
                # Keep infinity semantics for masked rows and empty history.
                corr = fx.Float32(fx.exp2(old_max - safe_max, fastmath="afn"))
                probs = fx.Vector(
                    fx.exp2(
                        scores
                        - fx.Vector.from_elements(
                            [fx.Float32(safe_max)], dtype=fx.Float32
                        ).broadcast_to(t.TOKENS // 8),
                        fastmath="afn",
                    )
                )
                local_sum = probs.reduce(ReductionOp.ADD)
                local_sum = local_sum + local_sum.shuffle_xor(16, t.WAVE)
                pscale = []
                for r in range_constexpr(t.TOKENS // 8):
                    if const_expr(t.per_token_kv):
                        vs = fx.Float32(vscales[r])
                        pscale.append(probs[r] * vs * vnorm)
                    else:
                        pscale.append(
                            probs[r] * fx.Float32(1.0 if t.bf16 else t.FP8_MAX)
                        )
                for n in range_constexpr(t.TOKENS // 64):
                    pvalues = fx.Vector.from_elements(
                        [pscale[n * 8 + r] for r in range_constexpr(8)],
                        dtype=fx.Float32,
                    )
                    p = pvalues.to(fx.BFloat16) if t.bf16 else self.mem.fp8(pvalues)
                    self.mem.store(
                        t.p_offset
                        + ctx.lane16 * t.v_stride
                        + (ctx.warp * (t.TOKENS // 4) + n * 16 + ctx.half * 8)
                        * t.elem_bytes,
                        p,
                        fx.BFloat16 if t.bf16 else fx.Int32,
                    )
                if ctx.half == 0:
                    self.mem.store(
                        t.stats_offset + 256 + (ctx.lane16 * t.NWARP + ctx.warp) * 4,
                        local_sum,
                        fx.Float32,
                    )
                # Prefetch V operands before waiting for cross-wave P stores.
                pv_k = 32 if t.bf16 else t.TOKENS
                if const_expr(t.fp8_multi_tile):
                    v_operands = page_v_operands
                elif const_expr(t.packed_lds):
                    v_operands = self.mem.page_v_operands(kv_off)
                elif const_expr(t.bf16_page16_packed):
                    v_operands = [
                        [
                            self.mem.bf16_v_operand(
                                kv_off + t.k_bytes,
                                vh * 64 + ctx.warp * 16,
                                d * pv_k,
                                ctx.lane16,
                                ctx.half,
                            )
                            for d in range_constexpr(t.TOKENS // pv_k)
                        ]
                        for vh in range_constexpr(vh_count)
                    ]
                else:
                    v_operands = [
                        [
                            self.mem.operand(
                                kv_off
                                + t.k_bytes
                                + (vh * 64 + ctx.warp * 16) * t.v_stride
                                + d * pv_k * t.elem_bytes,
                                t.v_stride,
                                pv_k,
                                ctx.lane16,
                                ctx.half,
                            )
                            for d in range_constexpr(t.TOKENS // pv_k)
                        ]
                        for vh in range_constexpr(vh_count)
                    ]
                gpu.barrier()
                denom = state[m * state_stride + 1] * corr + self.mem.load(
                    t.stats_offset + 256 + ctx.lane16 * t.NWARP * 4, fx.Float32, t.NWARP
                ).reduce(ReductionOp.ADD)
                next_state.extend([new_max, denom])
                p_operands = [
                    self.mem.operand(
                        t.p_offset + d * pv_k * t.elem_bytes,
                        t.v_stride,
                        pv_k,
                        ctx.lane16,
                        ctx.half,
                    )
                    for d in range_constexpr(t.TOKENS // pv_k)
                ]
                for vh in range_constexpr(vh_count):
                    # Packed outputs remain normalized by the previous V
                    # factor. Rescale before WMMA to fuse its accumulation.
                    out = (
                        state[m * state_stride + 2 + vh]
                        * fx.Vector.from_elements(
                            [corr * state[-1] * vnorm], dtype=fx.Float32
                        ).broadcast_to(8)
                        if t.packed_lds
                        else fx.Vector.filled(8, 0.0, fx.Float32)
                    )
                    for d in range_constexpr(t.TOKENS // pv_k):
                        out = Gfx1250Gemm.mma(
                            v_operands[vh][d], p_operands[d], out, pv_k, t.bf16
                        )
                    if const_expr(t.packed_lds):
                        next_state.append(out)
                    else:
                        next_state.append(
                            state[m * state_stride + 2 + vh]
                            * fx.Vector.from_elements(
                                [fx.Float32(corr)], dtype=fx.Float32
                            ).broadcast_to(8)
                            + out
                            * fx.Vector.from_elements(
                                [fx.Float32(vfactor)], dtype=fx.Float32
                            ).broadcast_to(8)
                        )
                # The next iteration's barrier retires the final M-tile reads
                # before a double buffer is reused. Intra-iteration reuse and
                # the single-buffer staging path still need an explicit barrier.
                if const_expr(m + 1 < t.m_tiles or t.buffers == 1):
                    gpu.barrier()
            if const_expr(t.buffers == 1):  # noqa: SIM102
                if token_base + t.TOKENS < end:
                    self.mem.stage_kv(token_base + t.TOKENS, fx.Int32(0))
            if const_expr(t.packed_lds):
                next_state.append(vfactor)
            final = yield next_state
        pout = fx.recast_iter(t.q_dtype, ctx.pout_ptr)
        pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
        psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
        slot = fx.Int64(ctx.kv_h) * fx.Int64(gpu.grid_dim.x) + fx.Int64(
            gpu.block_id("x")
        )
        for m in range_constexpr(t.m_tiles):
            row = m * 16 + ctx.lane16
            output_row = ctx.query_begin * t.query_group_size + row
            partial_row = slot * (t.rows * t.query_splits) + output_row
            denom = final[m * state_stride + 1]
            inv = fx.Float32(rcp_f32((denom > 0.0).select(denom, fx.Float32(1.0))))
            if row < t.rows:
                for vh in range_constexpr(vh_count):
                    d = vh * 64 + ctx.warp * 16 + ctx.half * 8
                    value = (
                        final[m * state_stride + 2 + vh]
                        * fx.Vector.from_elements(
                            [fx.Float32(inv * final[-1] if t.packed_lds else inv)],
                            dtype=fx.Float32,
                        ).broadcast_to(8)
                    ).to(t.q_dtype)
                    fx.ptr_store(
                        value,
                        fx.add_offset(
                            pout,
                            (
                                (
                                    fx.Int64(ctx.planned_seq) * t.query_length
                                    + output_row // t.query_group_size
                                )
                                * ctx.stride_q_row
                                + (
                                    ctx.kv_h * t.query_group_size
                                    + output_row % t.query_group_size
                                )
                                * ctx.stride_q_head
                                + d
                                if t.direct_output
                                else partial_row * t.head_dim + d
                            ),
                        ),
                    )
                if const_expr(not t.direct_output):  # noqa: SIM102
                    if ctx.warp == 0 and ctx.half == 0:
                        pmax[partial_row] = final[m * state_stride] * fx.Float32(
                            1.0 / LOG2E
                        )
                        psum[partial_row] = denom

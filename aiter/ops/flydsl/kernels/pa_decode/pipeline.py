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
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import ReductionOp, T

from ..tensor_shim import buf_base_i64
from ..utils import rcp_f32
from .op_epilog import PaDecodeEpilogue
from .op_gemm import Gfx1250Gemm, PaDecodeGemm
from .op_lds import (
    Gfx1250Memory,
    Gfx1250Mxfp4Loader,
    PaDecodeKVLoader,
    PaDecodeLds,
    PaDecodeQueryLoader,
    _gptr,
)
from .op_merge import Gfx1250PartitionMerge
from .op_softmax import PaDecodeSoftmax
from .traits import (
    KV_COMPUTE_BLOCK,
    LOG2E,
    MFMA_MNK,
    MXFP4_KV_DTYPE,
    MXFP4_P_EXP_SHIFT,
    WAVE,
)


class PaDecodePipeline:
    def __init__(self, ctx):
        self.ctx = ctx
        self.traits = ctx.traits
        self.lds = PaDecodeLds(ctx.traits, ctx.shared_storage)
        self.gemm = PaDecodeGemm(ctx.traits)
        self.query = PaDecodeQueryLoader(ctx, self.lds, self.gemm)
        self.kv = PaDecodeKVLoader(ctx, self.lds)
        self.softmax = PaDecodeSoftmax(ctx, self.lds, self.kv, self.gemm)
        self.epilogue = PaDecodeEpilogue(ctx, self.lds)

    @flyc.jit
    def run(self):
        """Initialize resources in load order, then compute and store one CTA."""
        if const_expr(self.traits.BF16_MTP_FUSED):
            self.run_bf16_fused()
        else:
            self.run_generic()

    @flyc.jit
    def run_bf16_fused(self):
        """Q, then tile-0 K and V, then tile-1 page ids: in-order vmcnt waits
        for an earlier group never drain a later one."""
        self.ctx.init_thread_mapping()
        self.epilogue.init_outputs()
        self.kv.init_loaders()
        # V is consumed once and last (PV): stream it, keep K's lines cacheable.
        self.kv._v_load_vector = self.bf16_nt_loader(self.ctx.value_cache_ptr)
        self.query.init_loaders()
        self.ctx.init_query_mapping()
        q_units = self.bf16_q_prefetch()
        self.ctx.init_context_length()
        self.kv.init_page_table()
        self.ctx.init_partition_bounds()
        self.lds.allocate()
        # The kernel runs a task only when part_start < part_end.
        pages0 = self.tile_pages(self.ctx.part_start)
        k_pf0 = self.kv.load_k_from_pages(
            fx.Vector.from_elements([pages0[0]], dtype=fx.Int32)
        )
        v_pf0 = []
        for vh in range_constexpr(self.traits.VHE_CHUNKS):
            v_pf0.extend(
                self.kv.load_v(fx.Vector.from_elements([pages0[1]], dtype=fx.Int32), vh)
            )
        pages1 = self.tile_pages(self.ctx.part_start + 1)
        self.ctx.init_scales(None, None)
        self.bf16_stage_q(q_units)
        gpu.barrier()
        self.softmax.init_mask()
        self.ctx.init_causal_offsets()
        o_zero = fx.full(self.traits.OP_ELEMS, 0.0, fx.Float32)
        init_state = [k_pf0, pages1]
        for _m in range_constexpr(self.traits.M_TILES):
            init_state.extend(
                [o_zero] * self.traits.VHE_CHUNKS + [self.ctx.NEG_INF, self.ctx.ZERO_F]
            )
        init_state.append(fx.Vector.from_elements(v_pf0, dtype=fx.Int64))
        # The task's last tile is peeled: it carries the V tail mask and
        # issues no next-tile loads, so the loop body has no branches.
        for loop_i, ostate in range(
            self.ctx.part_start, self.ctx.part_end - 1, 1, init=init_state
        ):
            next_state = self.bf16_fused_tile(ostate, fx.Int32(loop_i), False)
            results = yield next_state
        o_final = self.bf16_fused_tile(results, self.ctx.part_end - 1, True)
        self.epilogue.store(o_final)

    def bf16_nt_loader(self, tensor_ptr):
        """Nontemporal 16-byte loads at BF16 element ``elem_idx``."""
        base = buf_base_i64(tensor_ptr)
        ptr_ty = fx.PointerType.get(
            fx.Int32.ir_type, address_space=fx.AddressSpace.Global, alignment=16
        )

        def _load(elem_idx):
            ptr = fx.Pointer(fx.inttoptr(ptr_ty, base + fx.Int64(elem_idx) * 2))
            return fx.generic_load(ptr, dtype=fx.Int32, count=4, nontemporal=True)

        return _load

    def tile_pages(self, tt):
        """[page of this warp's K chunk, page of this rgroup's V run] of tile tt.

        Past the context the table load is pinned to page 0, which a tile
        never consumes.
        """
        base_page = tt * (KV_COMPUTE_BLOCK // self.traits.block_size)
        k_page = self.kv.load_pages(
            base_page
            + (self.ctx.warp * self.traits.TOK_PER_WARP) // self.traits.block_size
        )
        v_page = self.kv.load_pages(
            base_page
            + (self.ctx.rgroup * self.traits.TOK_PER_WARP) // self.traits.block_size
        )
        return fx.Vector.from_elements([k_page, v_page], dtype=fx.Int32)

    def bf16_q_prefetch(self):
        # Full M-tiles load Q ahead of the K/V bursts; a padded tile loads late.
        q_units = []
        for m in range_constexpr(self.traits.M_TILES):
            if const_expr((m + 1) * MFMA_MNK <= self.traits.CTA_ROWS):
                flat_idx = m * MFMA_MNK + self.ctx.qh_local
                qi = flat_idx // self.traits.query_group_size
                gs_head = flat_idx - qi * self.traits.query_group_size
                q_units.append(self.query.load_row(qi, gs_head))
            else:
                q_units.append(None)
        return q_units

    @flyc.jit
    def bf16_stage_q(self, q_units):
        """Stage Q rows at Q_ROW_BYTES, zero-filling rows past CTA_ROWS."""
        t = self.traits
        for m in range_constexpr(t.M_TILES):
            flat_idx = m * MFMA_MNK + self.ctx.qh_local
            row_off = flat_idx * t.Q_ROW_BYTES + self.ctx.lane16 * (
                t.QCHUNK * t.QP_ELEM_BYTES
            )
            if const_expr((m + 1) * MFMA_MNK <= t.CTA_ROWS):
                for u in range_constexpr(t.N_QLOADS):
                    self.lds.store(
                        row_off + u * t.QLOAD_UNIT * 2, fx.BFloat16, q_units[m][u]
                    )
            elif flat_idx < t.CTA_ROWS:
                qi = flat_idx // t.query_group_size
                gs_head = flat_idx - qi * t.query_group_size
                units = self.query.load_row(qi, gs_head)
                for u in range_constexpr(t.N_QLOADS):
                    self.lds.store(
                        row_off + u * t.QLOAD_UNIT * 2, fx.BFloat16, units[u]
                    )
            else:
                for u in range_constexpr(t.N_QLOADS):
                    self.lds.store(
                        row_off + u * t.QLOAD_UNIT * 2,
                        fx.BFloat16,
                        fx.full(t.QLOAD_UNIT, 0.0, fx.BFloat16),
                    )

    def bf16_q_ops(self, m):
        """One M-tile's QK B operands in PaDecodeKVLoader.load_k_chunk's order."""
        t = self.traits
        ops = []
        for qkhe in range_constexpr(t.QKHE_LOOP):
            he_idx = qkhe * t.RGROUP_QUARTERS + self.ctx.rgroup
            # Volatile: a reread must not merge with (and extend the live
            # range of) an earlier chunk group's Q operands.
            chunk = fx.generic_load(
                self.lds.pointer(
                    (m * MFMA_MNK + self.ctx.lane16) * t.Q_ROW_BYTES
                    + he_idx * t.QK_CHUNK_ELEMS * t.QP_ELEM_BYTES,
                    fx.Int64,
                ),
                dtype=fx.Int64,
                count=4,
                volatile=True,
            )
            for qkr in range_constexpr(4):
                ops.append(chunk[qkr])
        return ops

    def bf16_pv(self, ostate, m, v_cur, corr, last, v_masks, tile_valid, v_next):
        """O_m = O_m * corr + P_m V; with ``v_next`` (a list), each retired V
        chunk is refilled with the next tile's V right after its last MFMA."""
        t = self.traits
        l_new = ostate[t.l_slot(m)] * corr + self.softmax.fused_row_sum(m)
        p_ops = self.lds.load(
            self.softmax.p_offset(m)
            + self.ctx.lane16 * t.SP_ROW_BYTES
            + self.ctx.rgroup * t.TOK_PER_WARP * t.QP_ELEM_BYTES,
            fx.Int64,
            t.NVOPS,
        )
        corr_b = fx.full(t.OP_ELEMS, corr, fx.Float32)
        o_acc = []
        for vh in range_constexpr(t.VHE_CHUNKS):
            if const_expr(last and m == 0):
                v_cur[vh] = self.kv.mask_v_chunk(v_cur[vh], v_masks, tile_valid)
            o_acc.append(
                self.gemm.pv_bf16_k32(
                    v_cur[vh], p_ops, ostate[t.o_slot(m, vh)] * corr_b
                )
            )
            if const_expr(v_next is not None):
                v_next[0].extend(self.kv.load_v(v_next[1], vh))
        return o_acc, l_new

    @flyc.jit
    def bf16_fused_tile(self, ostate, tt, last):
        """QK -> max barrier -> P of the first half of the M-tiles -> barrier ->
        their PV overlapping the second half's P -> barrier -> its PV.

        ``last`` peels the task's final tile: it masks V rows past the context
        and issues no next-tile loads. Other tiles issue next K as scores die
        and next V while the final M-tile's PV retires the current V.
        """
        t = self.traits
        k_cur = ostate[t.K_SLOT]
        pages_next = ostate[t.V_SLOT]
        v_carried = ostate[t.V_DATA_SLOT]
        v_cur = [
            [v_carried[vh * t.NVOPS + i] for i in range_constexpr(t.NVOPS)]
            for vh in range_constexpr(t.VHE_CHUNKS)
        ]
        tile_valid = self.ctx.context_len - tt * KV_COMPUTE_BLOCK

        # K chunk groups outer: each K chunk dies after its last M-tile.
        frags = [[None] * t.NCHUNK for _m in range_constexpr(t.M_TILES)]
        for g in range_constexpr(t.NCHUNK // t.QK_AGROUP):
            for m in range_constexpr(t.M_TILES):
                q_ops = self.bf16_q_ops(m)
                for j in range_constexpr(t.QK_AGROUP):
                    a = g * t.QK_AGROUP + j
                    frags[m][a] = self.gemm.qk_bf16_k32(k_cur, q_ops, a)
        masked_saved = self.softmax.fused_scores(frags, tile_valid)
        for m in range_constexpr(t.M_TILES):
            self.softmax.fused_stage_max(masked_saved[m], m)
        gpu.barrier()

        half = t.M_TILES // 2
        m_new = [None] * t.M_TILES
        corr = [None] * t.M_TILES
        for m in range_constexpr(half):
            m_new[m], corr[m] = self.softmax.pack_fused_bf16(
                masked_saved[m], m, ostate[t.m_slot(m)]
            )
        k_flat = []
        k_page = fx.Int32(pages_next[0])
        if const_expr(not last):
            # The first half's scores are dead: start next K.
            for a in range_constexpr(t.NCHUNK // 2):
                k_flat.extend(self.kv.load_k_chunk(k_page, a))
        gpu.barrier()

        v_masks = self.kv.tail_masks(tile_valid) if const_expr(last) else None
        o_res = [None] * t.M_TILES
        l_res = [None] * t.M_TILES
        # Published PV MFMAs overlap the second half's exp/pack VALU.
        for i in range_constexpr(t.M_TILES - half):
            if const_expr(i < half):
                o_res[i], l_res[i] = self.bf16_pv(
                    ostate, i, v_cur, corr[i], last, v_masks, tile_valid, None
                )
            m_new[half + i], corr[half + i] = self.softmax.pack_fused_bf16(
                masked_saved[half + i], half + i, ostate[t.m_slot(half + i)]
            )
        next_pages = pages_next
        if const_expr(not last):
            for a in range_constexpr(t.NCHUNK // 2, t.NCHUNK):
                k_flat.extend(self.kv.load_k_chunk(k_page, a))
        gpu.barrier()

        v_next = []
        for m in range_constexpr(half, t.M_TILES):
            refill = None
            if const_expr(not last and m == t.M_TILES - 1):
                # Ahead of V on vmcnt: next K's wait never drains V.
                next_pages = self.tile_pages(tt + 2)
                refill = (
                    v_next,
                    fx.Vector.from_elements([pages_next[1]], dtype=fx.Int32),
                )
            o_res[m], l_res[m] = self.bf16_pv(
                ostate, m, v_cur, corr[m], last, v_masks, tile_valid, refill
            )

        next_state = [
            (
                k_cur
                if const_expr(last)
                else fx.Vector.from_elements(k_flat, dtype=fx.Int64)
            ),
            next_pages,
        ]
        for m in range_constexpr(t.M_TILES):
            next_state.extend([*o_res[m], m_new[m], l_res[m]])
        if const_expr(last):
            next_state.append(v_carried)
        else:
            next_state.append(fx.Vector.from_elements(v_next, dtype=fx.Int64))
        return next_state

    @flyc.jit
    def run_generic(self):
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
        self.lds.allocate(self.ctx.group)
        self.kv.init_scale_loaders()
        if const_expr(self.traits.stage_page_capacity):
            self.kv.stage_page_ids()
            gpu.barrier()
        pages1 = None
        if const_expr(self.traits.FP8_MTP4_PAGES_AHEAD):
            k_pf0, v_page_pf0, pages1, scale_stores = self.fp8_mtp4_prefetch()
            self.ctx.init_scales(self.kv.key_scale, self.kv.value_scale)
            self.query.stage(q_units_prefetched)
            self.kv.store_scales(scale_stores)
        elif const_expr(self.traits.FP8_SCALES_AFTER_Q):
            # Scale LDS stores wait on vmcnt behind K: issue them after Q
            # staging, so every Q row quantizes while K is in flight.
            k_pf0, scale_stores = self.kv.prefetch_first_scales_deferred()
            self.ctx.init_scales(self.kv.key_scale, self.kv.value_scale)
            self.query.stage(q_units_prefetched)
            self.kv.store_scales(scale_stores)
        else:
            k_pf0 = self.kv.prefetch_first()
            if const_expr(self.traits.kv_dtype == MXFP4_KV_DTYPE):
                v_page_pf0 = self.kv.mxfp4_prefetch("v")
            self.ctx.init_scales(self.kv.key_scale, self.kv.value_scale)
            self.query.stage(q_units_prefetched)
        gpu.barrier()
        if const_expr(
            self.traits.kv_dtype != MXFP4_KV_DTYPE
            and not self.traits.FP8_MTP4_PAGES_AHEAD
        ):
            v_page_pf0 = self.kv.read_v_pages()
        q_ops_all = (
            None
            if const_expr(self.traits.FP8_MTP4_Q_LDS)
            else self.query.load_operands()
        )
        self.softmax.init_mask()
        self.ctx.init_causal_offsets()
        o_final = self.mainloop(k_pf0, v_page_pf0, q_ops_all, pages1)
        if const_expr(
            (self.traits.kv_dtype == MXFP4_KV_DTYPE) and self.traits.DUAL_GROUP
        ):
            # Lane-wise (O, max, sum) merge, then group 0's four-wave merge;
            # its barrier passes once group 1's waves have ended.
            o_final = self.merge_groups(o_final)
            if self.ctx.group == 0:
                self.epilogue.store_mxfp4(o_final)
        elif const_expr(self.traits.kv_dtype == MXFP4_KV_DTYPE):
            self.epilogue.store_mxfp4(o_final)
        elif const_expr(self.traits.DUAL_GROUP):
            o_final = self.merge_groups(o_final)
            if self.ctx.group == 0:
                self.epilogue.store(o_final)
        else:
            self.epilogue.store(o_final)

    def publish_group_state(self, o_final, slot):
        """Write this lane's (O, m, l) at byte ``slot`` of the whole allocation."""
        t = self.traits
        for vh in range_constexpr(t.VHE_CHUNKS):
            fx.ptr_store(
                o_final[t.o_slot(0, vh)],
                self.lds.root_pointer(slot + vh * t.OP_ELEMS * 4, fx.Float32),
            )
        fx.ptr_store(
            fx.Vector.from_elements(
                [o_final[t.m_slot(0)], o_final[t.l_slot(0)]], dtype=fx.Float32
            ),
            self.lds.root_pointer(slot + t.VHE_CHUNKS * t.OP_ELEMS * 4, fx.Float32),
        )

    @flyc.jit
    def merge_groups(self, o_final):
        """Fold dual group 1's online-softmax state into group 0's.

        Lanes of both groups hold the same rows and head dims. After a barrier
        retires every tile's LDS use, group 1 publishes (O, m, l) per lane over
        the whole allocation; group 0 rescales both to the shared maximum.
        """
        t = self.traits
        n = t.VHE_CHUNKS * t.OP_ELEMS
        slot = self.ctx.tid * ((n + 2) * 4)
        gpu.barrier()
        if self.ctx.group == 1:
            self.publish_group_state(o_final, slot)
        gpu.barrier()
        ml = fx.ptr_load(
            self.lds.root_pointer(slot + n * 4, fx.Float32),
            result_type=fx.Vector.make_type(2, fx.Float32),
        )
        m0 = o_final[t.m_slot(0)]
        m1 = fx.Float32(ml[0])
        m = fx.maxnumf(m0, m1)
        safe = (m > self.ctx.NEG_INF).select(m, self.ctx.ZERO_F)
        a0 = fx.Float32(fx.exp2(m0 - safe, fastmath="fast"))
        a1 = fx.Float32(fx.exp2(m1 - safe, fastmath="fast"))
        merged = list(o_final)
        a0_b = fx.full(t.OP_ELEMS, a0, fx.Float32)
        a1_b = fx.full(t.OP_ELEMS, a1, fx.Float32)
        for vh in range_constexpr(t.VHE_CHUNKS):
            o1 = fx.ptr_load(
                self.lds.root_pointer(slot + vh * t.OP_ELEMS * 4, fx.Float32),
                result_type=fx.Vector.make_type(t.OP_ELEMS, fx.Float32),
            )
            merged[t.o_slot(0, vh)] = (
                o_final[t.o_slot(0, vh)] * a0_b + fx.Vector(o1) * a1_b
            )
        merged[t.m_slot(0)] = m
        merged[t.l_slot(0)] = o_final[t.l_slot(0)] * a0 + fx.Float32(ml[1]) * a1
        if const_expr(t.kv_dtype == MXFP4_KV_DTYPE):
            # The MXFP4 store merges its waves through LDS that overlaps the
            # merge slots: every lane reads its slot first.
            gpu.barrier()
        return merged

    def fp8_mtp4_prefetch(self):
        """First tile's scales and K, then the next tile's page ids.

        The scales lead on vmcnt, so their LDS stores (after Q staging) do
        not drain K. V issues after the barrier (V with K delays QK).
        """
        pages0 = self.tile_pages(self.ctx.part_start)
        k_page0 = fx.Vector.from_elements([pages0[0]], dtype=fx.Int32)
        scale_stores = self.kv.stage_scales(
            k_page0,
            self.kv.scale_buffer_offset(fx.Int32(self.ctx.part_start)),
            self.ctx.part_start,
            defer=True,
        )
        k_pf0 = self.kv.load_k_from_pages(k_page0)
        pages1 = self.tile_pages(self.ctx.part_start + 1)
        v_page0 = fx.Vector.from_elements([pages0[1]], dtype=fx.Int32)
        return k_pf0, v_page0, pages1, scale_stores

    @flyc.jit
    def mainloop(self, k_pf0, v_page_pf0, q_ops_all, pages_next=None):
        """Carry prefetch operands and online-softmax state between KV tiles.

        ``pages_next`` (FP8_MTP4_PAGES_AHEAD) replaces the carried V page ids
        with the next tile's K/V page ids.
        """
        o_zero = fx.full(self.traits.OP_ELEMS, 0.0, fx.Float32)
        init_state = [
            k_pf0,
            v_page_pf0 if const_expr(pages_next is None) else pages_next,
        ]
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
        # The d256 pipe keeps its M-tiles on the single-tile schedule.
        multi_query = self.traits.M_TILES > 1 and not self.traits.D256_PIPE
        if const_expr(self.traits.single_tile_plan):
            # Single-tile plans eliminate loop/history but keep absolute tt for
            # addressing and masks; the outer guard excludes padded tasks.
            o_final = self.tile(
                init_state, fx.Int32(self.ctx.part_start), multi_query, q_ops_all, True
            )
        elif const_expr(self.traits.DUAL_GROUP):
            # Group tiles part_start + 2j; only the peeled last one can hold
            # the task's tail or lie past it.
            for loop_j, ostate in range(0, self.ctx.dual_iters - 1, 1, init=init_state):
                next_state = self.tile(
                    ostate,
                    self.ctx.part_start + fx.Int32(loop_j) * 2,
                    multi_query,
                    q_ops_all,
                    False,
                )
                results = yield next_state
            o_final = self.tile(
                results,
                self.ctx.part_start + 2 * (self.ctx.dual_iters - 1),
                multi_query,
                q_ops_all,
                True,
            )
        else:
            # Only a sequence's last tile reads rows past its context, so the
            # loop leaves out the V tail mask and the task's last tile is
            # peeled to carry it. The outer guard ensures part_start < part_end.
            for loop_i, ostate in range(
                self.ctx.part_start, self.ctx.part_end - 1, 1, init=init_state
            ):
                next_state = self.tile(
                    ostate, fx.Int32(loop_i), multi_query, q_ops_all, False
                )
                results = yield next_state
            o_final = self.tile(
                results,
                self.ctx.part_end - 1,
                multi_query,
                q_ops_all,
                True,
                has_next=not (
                    self.traits.DECODE_LAST_V_AFTER_QK
                    or self.traits.FP8_LOOP_SCALES_LATE
                    or self.traits.FP8_MTP4_PAGES_AHEAD
                ),
            )
        return o_final

    @flyc.jit
    def tile(self, ostate, tt, multi_query, q_ops_all, mask_tail, has_next=True):
        """Run one KV tile; ``mask_tail`` emits the V mask for rows past the context.

        ``has_next=False`` marks the task's last tile: it emits no next-tile loads.
        """
        k_cur = ostate[
            self.traits.K_SLOT
        ]  # this tile's prefetched K, as one (NCHUNK*N_SUBCHUNKS,) i64 vector
        v_page_cur = ostate[
            self.traits.V_SLOT
        ]  # this tile's V pages, as one PAGES_PER_CHUNK-wide i32 vector
        tok0 = tt * KV_COMPUTE_BLOCK
        # Interleave MFMA with VALU/LDS or page16 V loads.
        if const_expr(
            (not self.traits.per_token_kv and multi_query) or self.traits.PAGE16_VPIPE
        ):
            fx.rocdl.iglp_opt(0)

        tt1 = tt + (2 if const_expr(self.traits.DUAL_GROUP) else 1)

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

        # Next page ids go ahead of this tile's V (PAGE16_PAGES_FIRST). Past
        # the task they read masked ids, which only the guarded staging sees.
        pages_ahead = None
        if const_expr(self.traits.PAGE16_PAGES_FIRST and has_next):
            pages_ahead = self.kv.load_tile_pages(tt1)
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
        if const_expr(self.traits.M_TILES > 1 and self.traits.kv_dtype != fx.BFloat16):
            q_scale_vec = self.lds.load(
                self.traits.sQscale_off
                + self.ctx.lane16 * (self.traits.M_TILES * self.traits.f32),
                fx.Float32,
                self.traits.M_TILES,
            )

        # All query tiles share one K/V tile; each path retains its publication barriers.
        if const_expr(multi_query):
            return self.multi_query_tile(
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
                mask_tail,
                has_next,
            )
        if const_expr(self.traits.kv_dtype == MXFP4_KV_DTYPE):
            # Masks tail scores/V by tile_valid at runtime, so it ignores mask_tail.
            return self.single_query_tile_mxfp4(
                ostate, k_cur, tt1, tile_valid, q_ops_all
            )
        return self.single_query_tile(
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
            mask_tail,
            has_next,
            pages_ahead,
        )

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
        mask_tail,
        has_next=True,
    ):
        """Publish all query maxima together, then consume shared K/V per query."""
        next_state = [None, None]
        masked_chunks_saved = [None] * self.traits.M_TILES
        # FP8_MTP4_PAGES_AHEAD: v_page_cur holds the next tile's page ids.
        pages_ahead = self.traits.FP8_MTP4_PAGES_AHEAD

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

        row_scales = [None] * self.traits.M_TILES
        lane_maxes = [None] * self.traits.M_TILES
        q_ops_tile = None
        if const_expr(self.traits.FP8_MTP4_Q_LDS):
            # One LDS round trip for every M-tile; each dies after its QK.
            q_ops_tile = [
                self.query.load_operands_m(m)
                for m in range_constexpr(self.traits.M_TILES)
            ]
        for m in range_constexpr(self.traits.M_TILES):
            if const_expr(self.traits.FP8_MTP4_Q_LDS):
                frag_Ss = self.gemm.qk(k_cur, q_ops_tile[m], 0)
            else:
                frag_Ss = self.gemm.qk(k_cur, q_ops_all, m)

            scale = (
                self.ctx.scale_qk
                if const_expr(self.traits.kv_dtype == fx.BFloat16)
                else self.ctx.scale_qk * fx.Float32(q_scale_vec[m])
            )
            masked_chunks = self.softmax.multi_scores(
                frag_Ss, scale, k_scale_shared, tile_valid, window_left, m
            )

            if const_expr(self.traits.FP8_MTP4_LEAN):
                # Unscaled row max. A zero-Q row has scale 0 and all-zero
                # scores: scale 1 keeps them, and -inf * 0 never occurs.
                lane_maxes[m] = self.softmax.lane_row_max(masked_chunks)
                row_scales[m] = (scale > self.ctx.ZERO_F).select(scale, fx.Float32(1.0))
            else:
                self.softmax.stage_row_max(
                    masked_chunks, self.softmax.lmax_offset(m), scale, True
                )

            masked_chunks_saved[m] = masked_chunks

        if const_expr(self.traits.FP8_MTP4_LEAN):
            self.softmax.stage_max4(lane_maxes)
        # Publish next pages/scales with the Phase A barrier.
        k_next = (
            fx.full(self.traits.NCHUNK * self.traits.N_SUBCHUNKS, 0, fx.Int64)
            if const_expr(self.traits.MTP4_FUSED)
            else k_cur
        )
        next_scale_stores = None
        pages_after = v_page_cur
        if const_expr(pages_ahead):
            if const_expr(has_next):
                # Page ids are in registers: the next scales issue without a
                # wait and land in LDS before the second barrier.
                next_scale_stores = self.kv.stage_scales(
                    fx.Vector.from_elements([v_page_cur[0]], dtype=fx.Int32),
                    self.kv.scale_buffer_offset(tt1),
                    tt1,
                    defer=True,
                )
                pages_after = self.tile_pages(tt1 + 1)
        elif const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end:
            if const_expr(self.traits.MTP4_FUSED):
                # Defer K until P packing releases saved score registers.
                phys_vec1 = self.kv.fetch_v_pages(tt1)
                self.kv.stage_scales(phys_vec1, self.kv.scale_buffer_offset(tt1), tt1)
            else:
                k_next, phys_vec1 = self.kv.load_k(tt1)
                self.kv.fetch_v_pages(tt1)
                if const_expr(self.traits.per_token_kv):
                    self.kv.stage_scales(
                        phys_vec1, self.kv.scale_buffer_offset(tt1), tt1
                    )
        next_state[self.traits.K_SLOT] = k_next

        gpu.barrier()

        v_page_next = v_page_cur
        if const_expr(pages_ahead):
            v_page_next = fx.Vector.from_elements([v_page_cur[1]], dtype=fx.Int32)
            next_state[self.traits.V_SLOT] = pages_after
        else:
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
            if const_expr(self.traits.FP8_MTP4_LEAN):
                m_new_saved = self.softmax.pack_fused_lean(
                    masked_chunks_saved, ostate, cur_kv_buf, norm_factor_b, row_scales
                )
            else:
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
            if const_expr(pages_ahead):
                if const_expr(has_next):
                    k_next = self.kv.load_k_from_pages(
                        fx.Vector.from_elements([v_page_cur[0]], dtype=fx.Int32)
                    )
                    # Behind next K on vmcnt; read after this barrier.
                    self.kv.store_scales(next_scale_stores)
            elif (
                const_expr(not self.traits.single_tile_plan) and tt1 < self.ctx.part_end
            ):
                # Saved scores are dead; next K can overlap the P barrier/PV.
                k_next = self.kv.load_k_from_pages(self.kv.read_k_pages())
            next_state[self.traits.K_SLOT] = k_next

            gpu.barrier()

            v_masks = self.kv.tail_masks(tile_valid) if const_expr(mask_tail) else None
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
                    if const_expr(mask_tail and m == 0):
                        v_vh_shared[vh] = self.kv.mask_v_chunk(
                            v_vh_shared[vh], v_masks, tile_valid
                        )
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
                        if const_expr(pages_ahead):
                            if const_expr(has_next):
                                v_next_chunk = fx.Vector.from_elements(
                                    self.kv.load_v(v_page_next, vh), dtype=fx.Int64
                                )
                        elif (
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
                if const_expr(mask_tail and m == 0):
                    v_masks = self.kv.tail_masks(tile_valid)
                for vh in range_constexpr(self.traits.VHE_CHUNKS):
                    if const_expr(mask_tail and m == 0):
                        v_vh_shared[vh] = self.kv.mask_v_chunk(
                            v_vh_shared[vh], v_masks, tile_valid
                        )
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
        mask_tail,
        has_next=True,
        pages_ahead=None,
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
        v_after_qk = self.traits.DECODE_LAST_V_AFTER_QK and not has_next
        if const_expr(v_after_qk):
            # No next K to load: issue V behind QK so its latency overlaps
            # the softmax barriers.
            v_vh_shared = [
                self.kv.load_v(v_page_cur, vh)
                for vh in range_constexpr(self.traits.VHE_CHUNKS)
            ]
        k_next = k_cur
        late_scales = self.traits.FP8_LOOP_SCALES_LATE and has_next
        scale_stores = None
        if const_expr(late_scales):
            # Loop tiles always have a next tile.
            k_next, next_pages = self.kv.load_k(tt1)
            self.kv.fetch_v_pages(tt1)
            scale_stores = self.kv.stage_scales(
                next_pages, self.kv.scale_buffer_offset(tt1), tt1, defer=True
            )
        if const_expr(
            has_next and not late_scales and not self.traits.single_tile_plan
        ) and (tt1 < self.ctx.part_end):
            if const_expr(self.traits.REUSE_KV_PAGES):
                # Stage scales before K so scale waits cannot drain K;
                # page16 defers K until the P barrier.
                phys_vec1 = self.kv.fetch_v_pages(tt1)
                if const_expr(self.traits.per_token_kv):
                    self.kv.stage_scales(
                        phys_vec1, self.kv.scale_buffer_offset(tt1), tt1
                    )
                if const_expr(not self.traits.PAGE16_VPIPE):
                    k_next = self.kv.load_k_from_pages(phys_vec1)
            elif const_expr(pages_ahead is not None):
                self.kv.stage_v_pages(pages_ahead)
                if const_expr(self.traits.PAGE16_EARLY_K):
                    k_next = self.kv.load_k_from_pages(pages_ahead)
            else:
                k_next, phys_vec1 = self.kv.load_k(tt1)
                self.kv.fetch_v_pages(tt1)
                if const_expr(self.traits.per_token_kv):
                    self.kv.stage_scales(
                        phys_vec1, self.kv.scale_buffer_offset(tt1), tt1
                    )
        if const_expr(self.traits.SCALES_BEFORE_CURRENT_V):
            # Issue V after scale staging so scale waits do not drain V.
            v_vh_shared = [
                self.kv.load_v(v_page_cur, vh)
                for vh in range_constexpr(self.traits.VHE_CHUNKS)
            ]
        scales = []
        masked_chunks = []
        v_scale_vecs = None
        # Single-query head-256 decode without a window: only the task's
        # peeled last tile can hold tokens past the context, so interior tiles
        # skip the score mask. (On the page-16 V pipe the masks are free.)
        score_mask = mask_tail or not (
            self.traits.D256_PIPE
            and self.traits.query_length == 1
            and self.traits.sliding_window == 0
        )
        for m in range_constexpr(self.traits.M_TILES):
            if const_expr(self.traits.kv_dtype == fx.BFloat16):
                scale = self.ctx.scale_qk
            elif const_expr(self.traits.M_TILES > 1):
                scale = self.ctx.scale_qk * fx.Float32(q_scale_vec[m])
            else:
                scale = self.ctx.scale_qk * self.lds.load_scalar(
                    self.traits.sQscale_off, self.ctx.lane16
                )
            scales.append(scale)
            chunks, v_scale_vecs = self.softmax.single_scores(
                frag_Ss[m],
                scale,
                tile_valid,
                window_left,
                cur_kv_buf,
                scale_bounds,
                m,
                score_mask,
            )
            masked_chunks.append(chunks)
        gpu.barrier()
        v_page_next = v_page_cur
        if const_expr(late_scales):
            v_page_next = self.kv.read_v_pages()
        if const_expr(
            has_next and not late_scales and not self.traits.single_tile_plan
        ) and (tt1 < self.ctx.part_end):
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
                score_mask,
            )
            m_new.append(m_new_m)
            corr_reg.append(corr_m)
        gpu.barrier()
        if const_expr(  # noqa: SIM102
            self.traits.PAGE16_VPIPE and not self.traits.PAGE16_EARLY_K
        ):
            # Next K can now overlap P reads/PV.
            if const_expr(has_next and not self.traits.single_tile_plan) and (
                tt1 < self.ctx.part_end
            ):
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
        if const_expr(self.traits.prefetch_v or v_after_qk):
            v_vh_batch = v_vh_shared
        else:
            v_vh_batch = [
                self.kv.load_v(v_page_cur, vh)
                for vh in range_constexpr(self.traits.VHE_CHUNKS)
            ]
        v_masks = self.kv.tail_masks(tile_valid) if const_expr(mask_tail) else None
        if const_expr(not mask_tail and self.traits.block_size >= 64):
            # Without the tail-mask branch, the scheduler sinks page64/128 V
            # loads next to their PV MFMAs and exposes their latency.
            fx.rocdl.sched_barrier(0)
        # vh outer, m inner: each M-tile is a separate accumulator chain on the
        # same V operand.
        for vh in range_constexpr(self.traits.VHE_CHUNKS):
            v_vh = v_vh_batch[vh]
            if const_expr(mask_tail):
                v_vh = self.kv.mask_v_chunk(v_vh, v_masks, tile_valid)
            for m in range_constexpr(self.traits.M_TILES):
                op = self.gemm.pv(v_vh, p_ops[m])
                if const_expr(self.traits.per_token_kv):
                    op = op * fx.full(self.traits.OP_ELEMS, v_max_scaled, fx.Float32)
                o_acc[m][vh] = (
                    op
                    if const_expr(self.traits.single_tile_plan)
                    else o_acc[m][vh] * corr_b[m] + op
                )
        if const_expr(late_scales):
            # The next tile reads only its own wave's scale slots.
            self.kv.store_scales(scale_stores)
        for m in range_constexpr(self.traits.M_TILES):
            next_state.extend([*o_acc[m], m_new[m], l_new[m]])
        return next_state

    @flyc.jit
    def single_query_tile_mxfp4(
        self,
        ostate,
        k_cur,
        tt1,
        tile_valid,
        q_ops_all,
    ):
        """MXFP4 tile: one wave, its own 64 tokens, all 128 dims: QK, online softmax,
        PV with P in registers (no barrier). Next-tile K/V loads are issued early;
        the four waves merge once after the loop."""
        t = self.traits
        o_acc = [ostate[t.o_slot(0, vh)] for vh in range_constexpr(t.VHE_CHUNKS)]
        m_prev = ostate[t.m_slot(0)]
        l_prev = ostate[t.l_slot(0)]
        v_raw = ostate[t.V_SLOT]
        # Unconditional: a branch around the loads makes the compiler's vmcnt counts
        # assume they were skipped and drain them at issue. The last tile re-reads itself.
        base = self.kv.mxfp4_base(
            self.kv.mxfp4_pages(fx.min(tt1, self.ctx.part_end - 1))
        )
        k_scales = [
            self.kv.mxfp4_e8m0_byte(k_cur[a * 5]) for a in range_constexpr(t.NCHUNK)
        ]
        # q_scale = 2**e * mantissa: 2**e is Q's e8m0 MFMA scale, the mantissa is applied
        # after, so the accumulator holds the logit, not the e4m3-normalized (up to 448x) one.
        q_bits = self.lds.load(
            self.lds.row_offset(t.sQscale_off, self.ctx.lane16, 1, fx.Float32),
            fx.Int32,
            1,
        )[0]
        q_mantissa = fx.Vector.from_elements(
            [(q_bits & fx.Int32(0x7FFFFF)) | fx.Int32(0x3F800000)], dtype=fx.Int32
        ).bitcast(fx.Float32)[0]
        frag_Ss = self.gemm.qk_mxfp4(k_cur, k_scales, q_ops_all[0], q_bits >> 23)
        k_next = self.kv.mxfp4_load("k", base)
        # Widen V while the softmax runs.
        for a in range_constexpr(t.NCHUNK):
            self.kv.mxfp4_stage_v_bf16(v_raw, a, tile_valid)
        scale_b = fx.full(4, self.ctx.scale_qk * q_mantissa, fx.Float32)
        scores = [frag_Ss[a] * scale_b for a in range_constexpr(t.NCHUNK)]
        if (
            tile_valid < KV_COMPUTE_BLOCK
        ):  # only a sequence's last tile has masked tokens
            thr, _ = self.softmax.row_thresholds(tile_valid, None, 0)
            neg4 = fx.full(4, float("-inf"), fx.Float32)
            scores = [
                self.softmax.score_mask(a, thr, None).select(scores[a], neg4)
                for a in range_constexpr(t.NCHUNK)
            ]
        nnan = self.ctx.fm_nnan
        m_tile = scores[0].reduce(ReductionOp.MAX, fastmath=nnan)
        for a in range_constexpr(1, t.NCHUNK):
            m_tile = fx.maxnumf(
                m_tile, scores[a].reduce(ReductionOp.MAX, fastmath=nnan), fastmath=nnan
            )
        for sh in (16, 32):
            m_tile = fx.maxnumf(m_tile, m_tile.shuffle_xor(sh, WAVE), fastmath=nnan)
        m_new = fx.maxnumf(m_prev, m_tile, fastmath=nnan)
        # Relative to 0 while nothing is valid yet: no -inf - -inf.
        safe = fx.Float32((m_new > self.ctx.NEG_INF).select(m_new, self.ctx.ZERO_F))
        safe_b = fx.full(4, safe, fx.Float32)
        alpha = fx.Float32(fx.exp2(m_prev - safe, fastmath="fast"))
        probs = [
            fx.Vector(fx.exp2(scores[a] - safe_b, fastmath="fast"))
            for a in range_constexpr(t.NCHUNK)
        ]
        l_new = l_prev * alpha
        for a in range_constexpr(t.NCHUNK):
            l_new = l_new + probs[a].reduce(ReductionOp.ADD)  # this lane's tokens
        # S^T[token 4g+v][head] is already the PV B operand layout. P is scaled by
        # 2**-MXFP4_P_EXP_SHIFT (l_new is not); ``store_mxfp4`` restores it.
        p_shift = fx.full(4, 2.0**-MXFP4_P_EXP_SHIFT, fx.Float32)
        p_ops = [
            (probs[a] * p_shift).to(fx.BFloat16).bitcast(fx.Int64)[0]
            for a in range_constexpr(t.NCHUNK)
        ]
        # V loads follow K's by part of a tile so the bursts do not queue together.
        v_next = self.kv.mxfp4_load("v", base)
        alpha_b = fx.full(t.OP_ELEMS, alpha, fx.Float32)
        o_acc = [o_acc[vh] * alpha_b for vh in range_constexpr(t.VHE_CHUNKS)]
        for a in range_constexpr(t.NCHUNK):
            v_ops = self.kv.mxfp4_read_staged_v_bf16(a)
            for vh in range_constexpr(t.VHE_CHUNKS):
                o_acc[vh] = fx.Vector(
                    self.gemm.mfma([v_ops[vh]], [p_ops[a]], 0, 0, 1, o_acc[vh])
                )
        return [k_next, v_next, *o_acc, m_new, l_new]


class Gfx1250Bf16Pipeline:
    """Four waves share packed BF16 pages and retain Q/PV operands."""

    def __init__(self, ctx):
        self.ctx, self.t = ctx, ctx.traits
        self.mem = Gfx1250Memory(ctx)
        self.merge = Gfx1250PartitionMerge(ctx, self.mem)

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
        # Keep the Q loads ahead of the first page's table lookup.
        fx.rocdl.sched_barrier(0)
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
                # The next query tile writes P and its sum only after its max
                # barrier, which every wave reaches after these reads.
            final = yield next_state
        slot = fx.Int64(ctx.kv_h) * fx.Int64(ctx.capacity) + fx.Int64(ctx.task_slot)

        def store(direct):
            # Direct stores write final rows; other tasks store packed partials.
            pout = fx.recast_iter(
                t.q_dtype,
                ctx.output_ptr if t.merged_output and direct else ctx.pout_ptr,
            )
            pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
            psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
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
                            self.merge.output_offset(output_row, d)
                            if direct
                            else partial_row * t.head_dim + d
                        )
                        if const_expr(t.merge_partitions and not direct):
                            self.merge.publish(fx.add_offset(pout, offset), value)
                        else:
                            fx.ptr_store(value, fx.add_offset(pout, offset))
                    if const_expr(not direct):  # noqa: SIM102
                        if ctx.warp == 0 and ctx.half == 0:
                            row_max = final[m * state_stride] * fx.Float32(1.0 / LOG2E)
                            if const_expr(t.merge_partitions):
                                self.merge.publish(
                                    fx.add_offset(pmax, partial_row), row_max
                                )
                                self.merge.publish(
                                    fx.add_offset(psum, partial_row), denom
                                )
                            else:
                                pmax[partial_row] = row_max
                                psum[partial_row] = denom

        if const_expr(t.merged_output):
            if ctx.task_count == 1:
                store(True)
            else:
                store(False)
                if const_expr(t.merge_partitions):
                    self.merge.arrive()
        else:
            store(t.direct_output)


class Gfx1250Bf16WavePipeline:
    """Wave pairs share two 16-row BF16 query tiles across 128-token pages.

    Each wave covers half of every page for both tiles, so one K or V operand
    feeds two WMMAs and a wave reads half the page from LDS. Half-wave h of QK
    token tile i holds tokens 16 i + 8 h + r, which is how PV's 32-token P
    operand splits its contraction, so P stays in registers. A pair merges its
    two online-softmax states once, after the last page.
    """

    def __init__(self, ctx):
        self.ctx, self.t = ctx, ctx.traits
        self.mem = Gfx1250Memory(ctx)
        self.merge = Gfx1250PartitionMerge(ctx, self.mem)
        # Bytes between eight-element chunks: K's eight dims span a page's
        # 128 tokens, V's eight tokens span the head. Both are 2048 at D128.
        self.k_chunk, self.v_chunk = 128 * 16, self.t.head_dim * 16

    @flyc.jit
    def _stage_next(self, token_base, begin, end, buffer):
        # The prologue stages the second page.
        if (token_base != begin) & (token_base + 128 < end):
            self.mem.stage_bf16_page(token_base + 128, 1 - buffer)

    @flyc.jit
    def _wait(self, token_base, begin, end):
        # The second page's K and V copies may stay in flight; a partial
        # page's extra V copies only lengthen this wait.
        if (token_base == begin) & (begin + 128 < end):
            tdm_ops.tensor_wait(2)
        else:
            tdm_ops.tensor_wait(0)

    def _operand(self, base, offset, chunk_bytes):
        # Half-wave h holds eight-element chunks h and h + 2 of a 32-wide
        # contraction slice. Packed K spaces chunks a 128-token page apart
        # (2048 bytes); transposed V spaces them a head row apart (16 D bytes).
        words = []
        for j in range_constexpr(2):
            pointer = fx.recast_iter(
                fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, 16),
                fx.add_offset(self.mem.base, base + offset + 2 * chunk_bytes * j),
            )
            fragment = fx.Vector(
                fx.ptr_load(pointer, result_type=fx.Vector.make_type(4, fx.Int32))
            )
            words.extend([fragment[w] for w in range_constexpr(4)])
        return fx.Vector.from_elements(words, dtype=fx.Int32).bitcast(fx.BFloat16)

    def _page(self, token_base, begin, end, state, q_operands, uppers, bases, masked):
        """Consume half a page; ``masked`` applies the causal and window bounds."""
        t = self.t
        # A positive scale commutes with the maximum and fuses into the
        # exponent; masked scores stay -inf.
        fused = t.softmax_scale > 0
        scale = fx.Float32(t.softmax_scale * LOG2E)
        buffer = ((token_base - begin) // 128) % 2
        self._wait(token_base, begin, end)
        gpu.barrier()
        self._stage_next(token_base, begin, end, buffer)
        # Selecting a precomputed base keeps every K/V load at a single base
        # register plus an immediate offset.
        kbase = (buffer == 0).select(bases[0][0], bases[0][1])
        vbase = (buffer == 0).select(bases[1][0], bases[1][1])
        tiles = [
            [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range_constexpr(4)]
            for _ in range_constexpr(2)
        ]
        for i in range_constexpr(4):
            for d in range_constexpr(t.head_dim // 32):
                k = self._operand(kbase, 4 * self.k_chunk * d + 256 * i, self.k_chunk)
                for m in range_constexpr(2):
                    tiles[m][i] = Gfx1250Gemm.mma(
                        k, q_operands[m][d], tiles[m][i], 32, True
                    )
        stride = 2 + t.head_dim // 16
        next_state, p_operands, outs = [], [], []
        for m in range_constexpr(2):
            limit = uppers[m] - token_base
            units = []
            for i in range_constexpr(4):
                for r in range_constexpr(8):
                    value = tiles[m][i][r]
                    if const_expr(not fused):
                        value = value * scale
                    if const_expr(masked):
                        # Rows never mix, and padded rows are never stored.
                        visible = fx.Int32(16 * i + r) < limit
                        if const_expr(t.sliding_window > 0):
                            visible = visible & (
                                fx.Int32(16 * i + r) >= limit - t.sliding_window
                            )
                        value = visible.select(value, fx.Float32(float("-inf")))
                    units.append(value)
            maxima = units
            while len(maxima) > 1:
                maxima = [
                    fx.maxnumf(maxima[k], maxima[k + 1])
                    for k in range_constexpr(0, len(maxima), 2)
                ]
            tile_max = fx.maxnumf(maxima[0], maxima[0].shuffle_xor(16, t.WAVE))
            if const_expr(fused):
                tile_max = tile_max * scale
            old_max = state[m * stride]
            new_max = fx.maxnumf(old_max, tile_max)
            safe_max = (new_max > fx.Float32(float("-inf"))).select(
                new_max, fx.Float32(0.0)
            )
            corr = fx.Float32(fx.exp2(old_max - safe_max, fastmath="afn"))
            scores = fx.Vector.from_elements(units, dtype=fx.Float32)
            if const_expr(fused):
                exponent = fx.fma(
                    scores,
                    fx.Vector.from_elements([scale], dtype=fx.Float32).broadcast_to(32),
                    fx.Vector.from_elements(
                        [-fx.Float32(safe_max)], dtype=fx.Float32
                    ).broadcast_to(32),
                )
            else:
                exponent = scores - fx.Vector.from_elements(
                    [fx.Float32(safe_max)], dtype=fx.Float32
                ).broadcast_to(32)
            probs = fx.Vector(fx.exp2(exponent, fastmath="afn"))
            pairs = [
                fx.Vector.from_elements([probs[k], probs[k + 1]], dtype=fx.Float32)
                for k in range_constexpr(0, 32, 2)
            ]
            while len(pairs) > 1:
                pairs = [
                    pairs[k] + pairs[k + 1] for k in range_constexpr(0, len(pairs), 2)
                ]
            tile_sum = pairs[0][0] + pairs[0][1]
            tile_sum = tile_sum + tile_sum.shuffle_xor(16, t.WAVE)
            p_operands.append(
                [
                    fx.Vector.from_elements(
                        [probs[16 * s + k] for k in range_constexpr(16)],
                        dtype=fx.Float32,
                    ).to(fx.BFloat16)
                    for s in range_constexpr(2)
                ]
            )
            next_state.append([new_max, state[m * stride + 1] * corr + tile_sum])
            # Rescale the running accumulator before PV's fused adds.
            factor = fx.Vector.from_elements([corr], dtype=fx.Float32).broadcast_to(8)
            outs.append(
                [
                    state[m * stride + 2 + d] * factor
                    for d in range_constexpr(t.head_dim // 16)
                ]
            )
        for d in range_constexpr(t.head_dim // 16):
            for s in range_constexpr(2):
                v = self._operand(
                    vbase,
                    t.k_bytes + 4 * self.v_chunk * s + 256 * d,
                    self.v_chunk,
                )
                for m in range_constexpr(2):
                    outs[m][d] = Gfx1250Gemm.mma(
                        v, p_operands[m][s], outs[m][d], 32, True
                    )
        return next_state[0] + outs[0] + next_state[1] + outs[1]

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
        self.mem.allocate()
        begin, end = (
            ctx.planned_start * KV_COMPUTE_BLOCK,
            ctx.planned_end * KV_COMPUTE_BLOCK,
        )
        self.mem.stage_bf16_page(begin, fx.Int32(0))
        # Wave 2 g + p takes tokens [64 p, 64 p + 64) of each page for tiles
        # 2 g and 2 g + 1. Its slot 0 holds tile 2 g + p, the one it stores.
        part = ctx.warp % 2
        query = fx.recast_iter(t.q_dtype, ctx.query_ptr)
        # Pages ending at or before the first query's bound mask nothing.
        unmasked_end = ctx.planned_context - t.query_length + ctx.query_begin + 1
        q_operands, uppers = [], []
        for m in range_constexpr(2):
            row = (ctx.warp - part + (part ^ m)) * 16 + ctx.lane16
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
            # This lane's tokens are page offsets 64 part + 8 half + c for
            # constant c.
            uppers.append(
                unmasked_end + row // t.query_group_size - 8 * ctx.half - 64 * part
            )
        # After the query loads, which the first page's QK waits for too.
        if begin + 128 < end:
            self.mem.stage_bf16_page(begin + 128, fx.Int32(1))
        pages = (end - begin) // 128
        split = (
            begin
            + fx.max(fx.min((unmasked_end - begin) // 128, pages), fx.Int32(0)) * 128
        )
        k_lanes = t.q_bytes + ctx.half * self.k_chunk + ctx.lane16 * 16
        v_lanes = t.q_bytes + ctx.half * self.v_chunk + ctx.lane16 * 16
        kbase = k_lanes + part * 64 * 16
        vbase = v_lanes + part * 8 * self.v_chunk
        bases = ((kbase, kbase + t.kv_stride), (vbase, vbase + t.kv_stride))
        initial = []
        for _ in range_constexpr(2):
            initial.extend([fx.Float32(float("-inf")), fx.Float32(0.0)])
            initial.extend(
                [
                    fx.Vector.filled(8, 0.0, fx.Float32)
                    for _ in range_constexpr(t.head_dim // 16)
                ]
            )
        head, head_state = begin, initial
        if const_expr(t.sliding_window > 0):
            # Pages starting before the last query's window edge are masked.
            window_begin = (
                ctx.planned_context
                - t.query_length
                + ctx.query_begin
                + t.query_length // t.query_splits
                - t.sliding_window
            )
            head = (
                begin
                + fx.max(
                    fx.min((window_begin - begin + 127) // 128, pages), fx.Int32(0)
                )
                * 128
            )
            split = fx.max(split, head)
            for token_i, state in range(begin, head, 128, init=initial):
                next_state = self._page(
                    fx.Int32(token_i),
                    begin,
                    end,
                    state,
                    q_operands,
                    uppers,
                    bases,
                    True,
                )
                head_state = yield next_state
        for token_i, state in range(head, split, 128, init=head_state):
            next_state = self._page(
                fx.Int32(token_i), begin, end, state, q_operands, uppers, bases, False
            )
            interior = yield next_state
        for token_i, state in range(split, end, 128, init=interior):
            next_state = self._page(
                fx.Int32(token_i), begin, end, state, q_operands, uppers, bases, True
            )
            final = yield next_state
        # Pass slot 1 to the partner through the buffer the last page left
        # unused; nothing reads or stages it any more.
        stride = 2 + t.head_dim // 16
        mine, sent = final[:stride], final[stride:]
        area = (1 - fx.max(pages - 1, fx.Int32(0)) % 2) * t.kv_stride
        slot_bytes = (t.head_dim // 16) * t.WAVE * 32 + t.WAVE * 8
        stats = (t.head_dim // 16) * t.WAVE * 32 + ctx.lane * 8
        outgoing = area + (ctx.warp ^ 1) * slot_bytes
        for d in range_constexpr(t.head_dim // 16):
            self.mem.store(
                outgoing + (d * t.WAVE + ctx.lane) * 32, sent[2 + d], fx.Float32
            )
        self.mem.store(
            outgoing + stats,
            fx.Vector.from_elements([sent[0], sent[1]], dtype=fx.Float32),
            fx.Float32,
        )
        gpu.barrier()
        incoming = area + ctx.warp * slot_bytes
        partner = self.mem.load(incoming + stats, fx.Float32, 2)
        new_max = fx.maxnumf(mine[0], fx.Float32(partner[0]))
        safe_max = (new_max > fx.Float32(float("-inf"))).select(
            new_max, fx.Float32(0.0)
        )
        own = fx.Float32(fx.exp2(mine[0] - safe_max, fastmath="afn"))
        other = fx.Float32(fx.exp2(fx.Float32(partner[0]) - safe_max, fastmath="afn"))
        final = [new_max, mine[1] * own + fx.Float32(partner[1]) * other]
        for d in range_constexpr(t.head_dim // 16):
            final.append(
                mine[2 + d]
                * fx.Vector.from_elements([own], dtype=fx.Float32).broadcast_to(8)
                + self.mem.load(incoming + (d * t.WAVE + ctx.lane) * 32, fx.Float32, 8)
                * fx.Vector.from_elements([other], dtype=fx.Float32).broadcast_to(8)
            )
        row = ctx.warp * 16 + ctx.lane16
        slot = fx.Int64(ctx.kv_h) * fx.Int64(ctx.capacity) + fx.Int64(ctx.task_slot)
        output_row = ctx.query_begin * t.query_group_size + row
        partial_row = slot * (t.rows * t.query_splits) + output_row
        denom = final[1]
        inv = fx.Float32(rcp_f32((denom > 0.0).select(denom, fx.Float32(1.0))))

        def store(direct):
            # Direct stores write final rows; other tasks store packed partials.
            pout = fx.recast_iter(
                t.q_dtype,
                ctx.output_ptr if t.merged_output and direct else ctx.pout_ptr,
            )
            pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
            psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
            if row < t.rows:
                for d in range_constexpr(t.head_dim // 16):
                    dim = d * 16 + ctx.half * 8
                    value = (
                        final[2 + d]
                        * fx.Vector.from_elements([inv], dtype=fx.Float32).broadcast_to(
                            8
                        )
                    ).to(t.q_dtype)
                    target = fx.add_offset(
                        pout,
                        (
                            self.merge.output_offset(output_row, dim)
                            if direct
                            else partial_row * t.head_dim + dim
                        ),
                    )
                    if const_expr(t.merge_partitions and not direct):
                        self.merge.publish(target, value)
                    else:
                        fx.ptr_store(value, target)
                if const_expr(not direct):  # noqa: SIM102
                    if ctx.half == 0:
                        row_max = final[0] * fx.Float32(1.0 / LOG2E)
                        if const_expr(t.merge_partitions):
                            self.merge.publish(
                                fx.add_offset(pmax, partial_row), row_max
                            )
                            self.merge.publish(fx.add_offset(psum, partial_row), denom)
                        else:
                            pmax[partial_row] = row_max
                            psum[partial_row] = denom

        if const_expr(t.merged_output):
            if ctx.task_count == 1:
                store(True)
            else:
                store(False)
                if const_expr(t.merge_partitions):
                    self.merge.arrive()
        else:
            store(t.direct_output)


class Gfx1250WavePipeline:
    """Each wave owns one 16-row FP8 query tile across every 128-token page.

    QK reads K rows in PV's contraction order: accumulator row 8 h + r of
    token tile i holds token 32 (i // 2) + 16 h + 8 (i % 2) + r, where half-wave
    h's P operand keeps it. P stays in registers, and the double-buffered page
    is the only LDS state the waves share.
    """

    def __init__(self, ctx):
        self.ctx, self.t = ctx, ctx.traits
        # One K128 WMMA spans each query tile's QK contraction.
        assert self.t.head_dim == 128, "the FP8 wave pipeline requires D128"
        self.mem = Gfx1250Memory(ctx)
        self.merge = Gfx1250PartitionMerge(ctx, self.mem)

    @flyc.jit
    def _stage_next(self, token_base, end, buffer):
        if token_base + self.t.TOKENS < end:
            self.mem.stage_kv(token_base + self.t.TOKENS, 1 - buffer)

    def _page(self, token_base, begin, end, state, q_operand, scale, upper, masked):
        """Consume one page; ``masked`` applies the causal and length bounds."""
        ctx, t = self.ctx, self.t
        # Unmasked scores are finite and the row scale is nonnegative, so it
        # commutes with the maximum and fuses into the exponent.
        fused = not masked and t.per_token_kv and t.softmax_scale >= 0
        buffer = ((token_base - begin) // t.TOKENS) % 2
        tdm_ops.tensor_wait(0)
        fx.rocdl.s_wait_asynccnt(0)
        gpu.barrier()
        kv_off = t.q_bytes + buffer * t.kv_stride
        self._stage_next(token_base, end, buffer)
        # K, then V per-token scales; piece s of this lane's 16-token chunks
        # starts at token 16 half + 8 (s ^ half).
        scales = kv_off + t.k_bytes + t.v_bytes
        piece = [16 * ctx.half + 8 * (s ^ ctx.half) for s in range_constexpr(2)]
        limit = [upper - token_base - 8 * (s ^ ctx.half) for s in range_constexpr(2)]
        if const_expr(t.per_token_kv):
            vscale = kv_off + t.k_bytes + t.v_bytes + t.TOKENS * 4
            vmax = fx.Float32(self.mem.load(vscale + ctx.lane * 4, fx.Float32)[0])
            for k in range_constexpr(1, t.TOKENS // t.WAVE):
                vmax = fx.maxnumf(
                    vmax,
                    fx.Float32(
                        self.mem.load(vscale + (ctx.lane + k * t.WAVE) * 4, fx.Float32)[
                            0
                        ]
                    ),
                )
            for shift in (16, 8, 4, 2, 1):
                vmax = fx.maxnumf(vmax, vmax.shuffle_xor(shift, t.WAVE))
            vfactor = vmax * fx.Float32(1.0 / t.FP8_MAX)
            # Empty planned tails carry no new mass. Keep the prior scale
            # so their zero V contribution preserves the output.
            vfactor = (vfactor > 0.0).select(vfactor, fx.Float32(state[-1]))
            vnorm = fx.Float32(rcp_f32(fx.maxnumf(vfactor, fx.Float32(1e-20))))
        else:
            value_scale = fx.recast_iter(fx.Float32, ctx.value_scale_ptr)
            vfactor = fx.Float32(value_scale[0]) * fx.Float32(1.0 / t.FP8_MAX)
            # A zero scalar V scale must produce zero without 0 * inf.
            vnorm = fx.Float32(
                rcp_f32((vfactor != 0.0).select(vfactor, fx.Float32(1.0)))
            )
        units = []
        for i in range_constexpr(t.TOKENS // 16):
            first, s = 32 * (i // 2), i % 2
            # Rows 0-7 and 8-15 read opposite chunk halves: distinct LDS banks.
            row_half = ctx.lane16 // 8
            token = first + 16 * row_half + 8 * (s ^ row_half) + ctx.lane16 % 8
            words = []
            for j in range_constexpr(4):
                frag = self.mem.load(
                    kv_off + (ctx.half + 2 * j) * t.TOKENS * 16 + token * 16,
                    fx.Int32,
                    4,
                )
                words.extend([frag[k] for k in range_constexpr(4)])
            scores = Gfx1250Gemm.mma(
                fx.Vector.from_elements(words, dtype=fx.Int32),
                q_operand,
                fx.Vector.filled(8, 0.0, fx.Float32),
                128,
            )
            if const_expr(t.per_token_kv):
                ks = self.mem.load(scales + (first + piece[s]) * 4, fx.Float32, 8)
            for r in range_constexpr(8):
                if const_expr(fused):
                    value = scores[r] * fx.Float32(ks[r])
                else:
                    value = scores[r] * scale
                    if const_expr(t.per_token_kv):
                        value = value * fx.Float32(ks[r])
                if const_expr(masked):
                    # Rows never mix, and padded rows are never stored.
                    value = (fx.Int32(first + r) < limit[s]).select(
                        value, fx.Float32(float("-inf"))
                    )
                units.append(value)
        scores = fx.Vector.from_elements(units, dtype=fx.Float32)
        tile_max = scores.reduce(ReductionOp.MAX)
        tile_max = fx.maxnumf(tile_max, tile_max.shuffle_xor(16, t.WAVE))
        if const_expr(fused):
            tile_max = tile_max * scale
        old_max = state[0]
        new_max = fx.maxnumf(old_max, tile_max)
        safe_max = (new_max > fx.Float32(float("-inf"))).select(
            new_max, fx.Float32(0.0)
        )
        corr = fx.Float32(fx.exp2(old_max - safe_max, fastmath="afn"))
        tokens = t.TOKENS // 2
        if const_expr(fused):
            # exp2(scale * u - max + log2(vnorm)) is the probability times
            # vnorm, which P needs; the denominator divides vnorm back out.
            probs = fx.Vector(
                fx.exp2(
                    fx.fma(
                        scores,
                        fx.Vector.from_elements([scale], dtype=fx.Float32).broadcast_to(
                            tokens
                        ),
                        fx.Vector.from_elements(
                            [
                                fx.Float32(fx.log2(vnorm, fastmath="afn"))
                                - fx.Float32(safe_max)
                            ],
                            dtype=fx.Float32,
                        ).broadcast_to(tokens),
                    ),
                    fastmath="afn",
                )
            )
        else:
            probs = fx.Vector(
                fx.exp2(
                    scores
                    - fx.Vector.from_elements(
                        [fx.Float32(safe_max)], dtype=fx.Float32
                    ).broadcast_to(tokens),
                    fastmath="afn",
                )
            )
        pairs = [
            fx.Vector.from_elements([probs[k], probs[k + 1]], dtype=fx.Float32)
            for k in range_constexpr(0, tokens, 2)
        ]
        while len(pairs) > 1:
            pairs = [pairs[k] + pairs[k + 1] for k in range_constexpr(0, len(pairs), 2)]
        tile_sum = pairs[0][0] + pairs[0][1]
        tile_sum = tile_sum + tile_sum.shuffle_xor(16, t.WAVE)
        if const_expr(fused):
            tile_sum = tile_sum * fx.maxnumf(vfactor, fx.Float32(1e-20))
        pieces = []
        for i in range_constexpr(t.TOKENS // 16):
            first, s = 32 * (i // 2), i % 2
            if const_expr(fused):
                vs = self.mem.load(
                    scales + (t.TOKENS + first + piece[s]) * 4, fx.Float32, 8
                )
                pvalues = [
                    probs[i * 8 + r] * fx.Float32(vs[r]) for r in range_constexpr(8)
                ]
            elif const_expr(t.per_token_kv):
                vs = self.mem.load(
                    scales + (t.TOKENS + first + piece[s]) * 4, fx.Float32, 8
                )
                pvalues = [
                    probs[i * 8 + r] * fx.Float32(vs[r]) * vnorm
                    for r in range_constexpr(8)
                ]
            else:
                pvalues = [
                    probs[i * 8 + r] * fx.Float32(t.FP8_MAX) for r in range_constexpr(8)
                ]
            pieces.append(
                self.mem.fp8(fx.Vector.from_elements(pvalues, dtype=fx.Float32))
            )
        # Half-wave 1 holds each chunk's upper piece in its even token tile.
        lower = ctx.half == 0
        words = []
        for i in range_constexpr(0, t.TOKENS // 16, 2):
            for w in range_constexpr(2):
                words.append(lower.select(pieces[i][w], pieces[i + 1][w]))
            for w in range_constexpr(2):
                words.append(lower.select(pieces[i + 1][w], pieces[i][w]))
        p_operand = fx.Vector.from_elements(words, dtype=fx.Int32)
        # Outputs remain normalized by the previous V factor. Rescale
        # before WMMA to fuse its accumulation.
        factor = fx.Vector.from_elements(
            [corr * state[-1] * vnorm], dtype=fx.Float32
        ).broadcast_to(8)
        next_state = [new_max, state[1] * corr + tile_sum]
        for d in range_constexpr(t.head_dim // 16):
            v_operand = self.mem.packed_operand(
                kv_off + t.k_bytes + d * 16 * 16,
                t.head_dim,
                t.TOKENS,
                ctx.lane16,
                ctx.half,
            )
            next_state.append(
                Gfx1250Gemm.mma(v_operand, p_operand, state[2 + d] * factor, 128)
            )
        next_state.append(vfactor)
        return next_state

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
        self.mem.allocate()
        begin, end = (
            ctx.planned_start * KV_COMPUTE_BLOCK,
            ctx.planned_end * KV_COMPUTE_BLOCK,
        )
        # The first page's DMA overlaps Q staging; the regions are disjoint.
        self.mem.stage_kv(begin, fx.Int32(0))
        self.mem.stage_query()
        row = ctx.warp * 16 + ctx.lane16
        gpu.barrier()
        q_operand = self.mem.operand(
            ctx.warp * 16 * t.q_stride, t.q_stride, 128, ctx.lane16, ctx.half
        )
        scale = fx.Float32(
            self.mem.load(row * t.q_stride + t.head_dim, fx.Float32)[0]
        ) * fx.Float32(t.softmax_scale * LOG2E)
        if const_expr(not t.per_token_kv):
            key_scale = fx.recast_iter(fx.Float32, ctx.key_scale_ptr)
            scale = scale * fx.Float32(key_scale[0])
        # Pages ending at or before the first query's bound mask nothing.
        unmasked_end = ctx.planned_context - t.query_length + ctx.query_begin + 1
        split = (
            begin
            + fx.max(
                fx.min((unmasked_end - begin) // t.TOKENS, (end - begin) // t.TOKENS),
                fx.Int32(0),
            )
            * t.TOKENS
        )
        # This lane's tokens are page offsets 16 half + c for constant c.
        upper = unmasked_end + row // t.query_group_size - 16 * ctx.half
        initial = [fx.Float32(float("-inf")), fx.Float32(0.0)]
        initial.extend(
            [
                fx.Vector.filled(8, 0.0, fx.Float32)
                for _ in range_constexpr(t.head_dim // 16)
            ]
        )
        initial.append(fx.Float32(1.0))
        for token_i, state in range(begin, split, t.TOKENS, init=initial):
            next_state = self._page(
                fx.Int32(token_i), begin, end, state, q_operand, scale, upper, False
            )
            interior = yield next_state
        for token_i, state in range(split, end, t.TOKENS, init=interior):
            next_state = self._page(
                fx.Int32(token_i), begin, end, state, q_operand, scale, upper, True
            )
            final = yield next_state
        slot = fx.Int64(ctx.kv_h) * fx.Int64(ctx.capacity) + fx.Int64(ctx.task_slot)
        output_row = ctx.query_begin * t.query_group_size + row
        partial_row = slot * (t.rows * t.query_splits) + output_row
        denom = final[1]
        inv = fx.Float32(rcp_f32((denom > 0.0).select(denom, fx.Float32(1.0))))

        def store(direct):
            # Direct stores write final rows; other tasks store packed partials.
            pout = fx.recast_iter(
                t.q_dtype,
                ctx.output_ptr if t.merged_output and direct else ctx.pout_ptr,
            )
            pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
            psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
            if row < t.rows:
                for d in range_constexpr(t.head_dim // 16):
                    dim = d * 16 + ctx.half * 8
                    value = (
                        final[2 + d]
                        * fx.Vector.from_elements(
                            [fx.Float32(inv * final[-1])], dtype=fx.Float32
                        ).broadcast_to(8)
                    ).to(t.q_dtype)
                    target = fx.add_offset(
                        pout,
                        (
                            self.merge.output_offset(output_row, dim)
                            if direct
                            else partial_row * t.head_dim + dim
                        ),
                    )
                    if const_expr(t.merge_partitions and not direct):
                        self.merge.publish(target, value)
                    else:
                        fx.ptr_store(value, target)
                if const_expr(not direct):  # noqa: SIM102
                    if ctx.half == 0:
                        row_max = final[0] * fx.Float32(1.0 / LOG2E)
                        if const_expr(t.merge_partitions):
                            self.merge.publish(
                                fx.add_offset(pmax, partial_row), row_max
                            )
                            self.merge.publish(fx.add_offset(psum, partial_row), denom)
                        else:
                            pmax[partial_row] = row_max
                            psum[partial_row] = denom

        if const_expr(t.merged_output):
            if ctx.task_count == 1:
                store(True)
            else:
                store(False)
                if const_expr(t.merge_partitions):
                    self.merge.arrive()
        else:
            store(t.direct_output)


class Gfx1250Pipeline:
    def __init__(self, ctx):
        self.ctx, self.t = ctx, ctx.traits
        self.mem = Gfx1250Memory(ctx)
        self.merge = Gfx1250PartitionMerge(ctx, self.mem)

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
        self.mem.allocate()
        begin, end = (
            ctx.planned_start * KV_COMPUTE_BLOCK,
            ctx.planned_end * KV_COMPUTE_BLOCK,
        )
        # The first page's DMA overlaps Q staging; the regions are disjoint.
        self.mem.stage_kv(begin, fx.Int32(0))
        if const_expr(t.q_tiles == t.m_tiles):
            self.mem.stage_query()
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
                # The next M-tile writes P and statistics only after its max
                # barrier, which every wave reaches after these reads. The next
                # iteration's barrier likewise retires them before a double
                # buffer is reused; single-buffer staging still needs one here.
                if const_expr(t.buffers == 1 and m + 1 == t.m_tiles):
                    gpu.barrier()
            if const_expr(t.buffers == 1):  # noqa: SIM102
                if token_base + t.TOKENS < end:
                    self.mem.stage_kv(token_base + t.TOKENS, fx.Int32(0))
            if const_expr(t.packed_lds):
                next_state.append(vfactor)
            final = yield next_state
        slot = fx.Int64(ctx.kv_h) * fx.Int64(ctx.capacity) + fx.Int64(ctx.task_slot)

        def store(direct):
            # Direct stores write final rows; other tasks store packed partials.
            pout = fx.recast_iter(
                t.q_dtype,
                ctx.output_ptr if t.merged_output and direct else ctx.pout_ptr,
            )
            pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
            psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
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
                        target = fx.add_offset(
                            pout,
                            (
                                self.merge.output_offset(output_row, d)
                                if direct
                                else partial_row * t.head_dim + d
                            ),
                        )
                        if const_expr(t.merge_partitions and not direct):
                            self.merge.publish(target, value)
                        else:
                            fx.ptr_store(value, target)
                    if const_expr(not direct):  # noqa: SIM102
                        if ctx.warp == 0 and ctx.half == 0:
                            row_max = final[m * state_stride] * fx.Float32(1.0 / LOG2E)
                            if const_expr(t.merge_partitions):
                                self.merge.publish(
                                    fx.add_offset(pmax, partial_row), row_max
                                )
                                self.merge.publish(
                                    fx.add_offset(psum, partial_row), denom
                                )
                            else:
                                pmax[partial_row] = row_max
                                psum[partial_row] = denom

        if const_expr(t.merged_output):
            if ctx.task_count == 1:
                store(True)
            else:
                store(False)
                if const_expr(t.merge_partitions):
                    self.merge.arrive()
        else:
            store(t.direct_output)


class Gfx1250Mxfp4Pipeline:
    """MXFP4 K/V on gfx1250: head_dim 128, page 128, one query token a row.

    Each wave owns 64 tokens of every 256-token tile and all 128 dims, so the
    tile loop has no barrier; the waves merge once after it. Raw K/V rows load
    one tile ahead, page ids two tiles ahead.

    QK is one scaled WMMA per 16 tokens: E2M1 K rows with their E8M0 scales
    against the E4M3 query, whose row scale's exponent rides on the WMMA's B
    scale (its mantissa scales the scores), so the accumulator holds the logit.

    V is dequantized by the same scaled WMMA against E2M1 identity blocks: the
    product is exact in f32 and lands transposed (lane = dim, registers =
    tokens), which packed to bf16 is the A operand of the bf16 PV WMMA; P stays
    in registers as its B operand. The v_cvt_scale_pk8_*_fp4 conversions would
    widen V on the VALU, but on gfx1250 (FlyDSL 0.5.0 LLVM, B0) they ignore
    their data operand: 0x22222222 and 0x76543210 both convert to the
    nibbles of 0x7F7F7F7F.
    """

    TILE_TOKENS = 64  # per wave
    DIM_TILES = 8  # 16-dim PV output tiles of head_dim 128
    DEQUANT_GROUP = 4  # dim tiles whose V dequant WMMAs issue back to back
    UNIT_SCALES = 0x7F7F7F7F
    MIN_NORMAL_F32 = 1.17549435e-38

    def __init__(self, ctx):
        self.ctx, self.t = ctx, ctx.traits
        self.mem = Gfx1250Memory(ctx)

    def query_operand(self, lane16, half):
        """This lane's E4M3 B operand of the QK WMMA (row lane16), the row's
        E8M0 scale word and the mantissa of its scale. A bf16 row is normalized
        by absmax / 448, floored at the smallest normal f32 so 1 / scale stays
        finite and a bf16 subnormal row stays inside E4M3."""
        ctx, t = self.ctx, self.t
        # Padded rows repeat the last query head; their outputs are dropped.
        row = fx.min(lane16, fx.Int32(t.query_group_size - 1))
        head = ctx.kv_h * t.query_group_size + row
        offset = ctx.planned_seq * ctx.stride_q_row + head * ctx.stride_q_head
        if const_expr(t.query_dtype == "fp8"):
            q = _gptr(ctx.query_ptr, fx.Int32, 16)
            words = []
            for j in range_constexpr(4):
                chunk = fx.ptr_load(
                    fx.add_offset(q, fx.Uint32((offset + 16 * (half + 2 * j)) // 4)),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
                words.extend([chunk[i] for i in range_constexpr(4)])
            return (
                fx.Vector.from_elements(words, dtype=fx.Int32),
                fx.Int32(self.UNIT_SCALES),
                fx.Float32(1.0),
            )
        q = _gptr(ctx.query_ptr, fx.BFloat16, 16)
        units = [
            fx.Vector(
                fx.ptr_load(
                    fx.add_offset(q, offset + 16 * (half + 2 * j) + 8 * u),
                    result_type=fx.Vector.make_type(8, fx.BFloat16),
                )
            ).to(fx.Float32)
            for j in range_constexpr(4)
            for u in range_constexpr(2)
        ]
        absmax = fmath.absf(units[0]).reduce(ReductionOp.MAX)
        for u in range_constexpr(1, len(units)):
            absmax = fx.maxnumf(absmax, fmath.absf(units[u]).reduce(ReductionOp.MAX))
        absmax = fx.maxnumf(absmax, absmax.shuffle_xor(16, t.WAVE))
        q_scale = fx.maxnumf(
            absmax * fx.Float32(1.0 / t.FP8_MAX), fx.Float32(self.MIN_NORMAL_F32)
        )
        inv = fx.full(8, fx.Float32(rcp_f32(q_scale)), fx.Float32)
        words = []
        for u in range_constexpr(len(units)):
            packed = self.mem.fp8(units[u] * inv)
            words.extend([packed[0], packed[1]])
        bits = fx.Int32(
            fx.Vector.from_elements([q_scale], dtype=fx.Float32).bitcast(fx.Int32)[0]
        )
        mantissa = fx.Vector.from_elements(
            [(bits & fx.Int32(0x7FFFFF)) | fx.Int32(0x3F800000)], dtype=fx.Int32
        ).bitcast(fx.Float32)[0]
        return (
            fx.Vector.from_elements(words, dtype=fx.Int32),
            (bits >> 23) * fx.Int32(0x01010101),
            fx.Float32(mantissa),
        )

    @staticmethod
    def identity_rows(lane16, half):
        """Per owner half, 14 dwords [0 x 6, one, one', 0 x 6] whose 8-dword
        windows are the E2M1 identity blocks (see ``identity_operand``)."""
        zero = fx.Int32(0)
        upper = lane16 >= 8
        rows = []
        for owner in range_constexpr(2):
            one = (half == owner).select(fx.Int32(2) << ((lane16 % 8) * 4), zero)
            rows.append(
                fx.Vector.from_elements(
                    [zero] * 6
                    + [upper.select(zero, one), upper.select(one, zero)]
                    + [zero] * 6,
                    dtype=fx.Int32,
                )
            )
        return rows

    @staticmethod
    def identity_operand(rows, j):
        """E2M1 B operand I_j with I_j[k][n] = 1.0 iff k == 16 j + n: column n's
        one sits in dword 4 (j // 4) + 2 (j % 2) + n // 8, nibble n % 8, of the
        lanes of half (j // 2) % 2. Windows of one 14-dword row keep the eight
        operands within 28 registers."""
        row = rows[(j // 2) % 2]
        first = 6 - (4 * (j // 4) + 2 * (j % 2))
        return row.shuffle(row, list(range(first, first + 8)))

    @flyc.jit
    def run(self):
        ctx, t = self.ctx, self.t
        tid = fx.Int32(gpu.thread_id("x"))
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // t.WAVE))
        lane = tid % t.WAVE
        lane16, half = lane % 16, lane // 16
        ctx.kv_h = fx.Int32(gpu.block_id("y"))
        self.mem.allocate()
        kv = Gfx1250Mxfp4Loader(ctx, warp, lane16, half)
        begin, end = ctx.planned_start, ctx.planned_end
        base = kv.base(kv.page_id(begin))
        k_first = kv.load("k", base)
        v_first = kv.load("v", base)
        page_second = kv.page_id(fx.min(begin + 1, end - 1))
        q_op, q_scale, q_mantissa = self.query_operand(lane16, half)
        score_scale = q_mantissa * fx.Float32(t.softmax_scale * LOG2E)
        scale8 = fx.full(8, score_scale, fx.Float32)
        identity_rows = self.identity_rows(lane16, half)
        # Lane offsets of the 8 scores of chunk c: tokens 16 c + 8 half + i.
        score_tokens = fx.Vector.from_elements(
            [float(i) for i in range_constexpr(8)], dtype=fx.Float32
        )
        neg_inf = fx.full(8, float("-inf"), fx.Float32)
        zero8 = fx.full(8, 0, fx.Int32)
        initial = [fx.Float32(float("-inf")), fx.Float32(0.0)]
        initial.extend(
            [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range_constexpr(8)]
        )
        initial.extend([k_first, v_first, page_second])
        for tt_i, state in range(begin, end, 1, init=initial):
            tt = fx.Int32(tt_i)
            m_prev, l_prev = state[0], state[1]
            out = [state[2 + j] for j in range_constexpr(self.DIM_TILES)]
            k_cur, v_cur, page_next = state[10], state[11], state[12]
            # Unconditional: the last tile re-reads itself rather than branch
            # around loads, which would make their wait counts conservative.
            page_after = kv.page_id(fx.min(tt + 2, end - 1))
            base_next = kv.base(page_next)
            scores = []
            for c in range_constexpr(kv.CHUNKS):
                k_scale, k_rows = kv.chunk(k_cur, c)
                scores.append(
                    Gfx1250Gemm.mma_mx(
                        k_rows, q_op, k_scale, q_scale, Gfx1250Gemm.FMT_E4M3
                    )
                )
            # The next tile loads while this one computes; its K reuses this
            # tile's K registers.
            k_next = kv.load("k", base_next)
            v_next = kv.load("v", base_next)
            v_scales, v_rows = [], []
            for c in range_constexpr(kv.CHUNKS):
                v_scale, v_row = kv.chunk(v_cur, c)
                v_scales.append(v_scale)
                v_rows.append(v_row)
            tile_valid = ctx.planned_context - tt * KV_COMPUTE_BLOCK
            if tile_valid < KV_COMPUTE_BLOCK:
                # Only a sequence's last tile has tokens past the context. Their
                # scores are -inf, and their V rows become exact zeros: E2M1 has
                # no NaN, but a scale byte of 255 does.
                score_limit = fx.full(
                    8,
                    (tile_valid - (warp * self.TILE_TOKENS + half * 8)).to(fx.Float32),
                    fx.Float32,
                )
                for c in range_constexpr(kv.CHUNKS):
                    scores[c] = (
                        score_tokens + fx.full(8, float(16 * c), fx.Float32)
                        < score_limit
                    ).select(scores[c], neg_inf)
                    row_valid = warp * self.TILE_TOKENS + 16 * c + lane16 < tile_valid
                    v_rows[c] = row_valid.select(v_rows[c], zero8)
                    v_scales[c] = row_valid.select(
                        v_scales[c], fx.Int32(self.UNIT_SCALES)
                    )
            # The running maximum stays in accumulator units (the scale is
            # positive and per row): the max token's difference is an exact 0
            # even when the scaled logit is near the f32 limit.
            m_tile = scores[0].reduce(ReductionOp.MAX)
            for c in range_constexpr(1, kv.CHUNKS):
                m_tile = fx.maxnumf(m_tile, scores[c].reduce(ReductionOp.MAX))
            m_tile = fx.maxnumf(m_tile, m_tile.shuffle_xor(16, t.WAVE))
            m_new = fx.maxnumf(m_prev, m_tile)
            # Relative to 0 while nothing is visible yet: no -inf - -inf.
            safe = fx.Float32(
                (m_new > fx.Float32(float("-inf"))).select(m_new, fx.Float32(0.0))
            )
            alpha = fx.Float32(fx.exp2((m_prev - safe) * score_scale, fastmath="afn"))
            # P * 2**-MXFP4_P_EXP_SHIFT = exp2((s - max) * scale - shift): a long
            # sum of large V stays finite in f32. The denominator carries the
            # same shift, so it cancels in the normalization.
            safe8 = fx.full(8, safe, fx.Float32)
            shift8 = fx.full(8, -float(MXFP4_P_EXP_SHIFT), fx.Float32)
            probs = [
                fx.Vector(
                    fx.exp2(
                        fmath.fma(scores[c] - safe8, scale8, shift8), fastmath="afn"
                    )
                )
                for c in range_constexpr(kv.CHUNKS)
            ]
            # Each lane sums its own tokens; the halves combine after the loop.
            l_new = l_prev * alpha
            for c in range_constexpr(kv.CHUNKS):
                l_new = l_new + probs[c].reduce(ReductionOp.ADD)
            p_bf16 = [probs[c].to(fx.BFloat16) for c in range_constexpr(4)]
            p_ops = [
                p_bf16[2 * step].shuffle(p_bf16[2 * step + 1], list(range(16)))
                for step in range_constexpr(2)
            ]
            # alpha is exactly 1 for every row whose maximum held: skip the
            # rescale unless some row's maximum grew. Raw ballot: FlyDSL has no
            # wave-vote API.
            grew = fx.Int32(
                fx.rocdl.ballot(T.i32, fx.as_ir_value(alpha != fx.Float32(1.0)))
            )
            if grew != 0:
                alpha8 = fx.full(8, alpha, fx.Float32)
                out = [out[j] * alpha8 for j in range_constexpr(self.DIM_TILES)]
            group = self.DEQUANT_GROUP
            for step in range_constexpr(2):
                # Chunks 2 step and 2 step + 1 are the K32 halves of the step.
                for j0 in range_constexpr(0, self.DIM_TILES, group):
                    # Dequantize a group of dim tiles before packing any: one
                    # dequant -> pack -> PV chain at a time leaves each WMMA
                    # waiting on the last (B16 ctx64k pa_decode 15.0 -> 13.9 us).
                    raw = [
                        [
                            Gfx1250Gemm.mma_mx(
                                v_rows[c],
                                self.identity_operand(identity_rows, j),
                                v_scales[c],
                                fx.Int32(self.UNIT_SCALES),
                                Gfx1250Gemm.FMT_E2M1,
                            )
                            for c in (2 * step, 2 * step + 1)
                        ]
                        for j in range_constexpr(j0, j0 + group)
                    ]
                    for j in range_constexpr(j0, j0 + group):
                        pair = raw[j - j0]
                        out[j] = Gfx1250Gemm.mma(
                            pair[0]
                            .to(fx.BFloat16)
                            .shuffle(pair[1].to(fx.BFloat16), list(range(16))),
                            p_ops[step],
                            out[j],
                            32,
                            bf16=True,
                        )
            final = yield [m_new, l_new, *out, k_next, v_next, page_after]
        self.store(final, warp, lane16, half, tid, score_scale)

    @flyc.jit
    def store(self, final, warp, lane16, half, tid, score_scale):
        """Merge the four waves' (max, sum, O) through LDS and store the rows:
        final rows without a plan or for a cluster-merged sequence's only task,
        else packed partials for the reducer or the cluster merge."""
        ctx, t = self.ctx, self.t
        stats = t.mxfp4_stats_offset
        o_row = warp * t.MXFP4_O_BYTES + lane16 * (t.MXFP4_O_STRIDE * 4)
        for j in range_constexpr(self.DIM_TILES):
            self.mem.store(o_row + (16 * j + 8 * half) * 4, final[2 + j], fx.Float32)
        row_sum = fx.Float32(final[1])
        row_sum = row_sum + row_sum.shuffle_xor(16, t.WAVE)
        if half == 0:
            # Maxima in log2 logit units: rows of the merge below may differ in
            # their score scale.
            self.mem.store(
                stats + (warp * 16 + lane16) * 4,
                fx.Float32(final[0]) * score_scale,
                fx.Float32,
            )
            self.mem.store(
                stats + (t.NWARP + warp) * 64 + lane16 * 4, row_sum, fx.Float32
            )
        gpu.barrier()
        row, dim = tid // 8, (tid % 8) * 16
        maxima = [
            fx.Float32(self.mem.load(stats + (w * 16 + row) * 4, fx.Float32)[0])
            for w in range_constexpr(t.NWARP)
        ]
        sums = [
            fx.Float32(
                self.mem.load(stats + (t.NWARP + w) * 64 + row * 4, fx.Float32)[0]
            )
            for w in range_constexpr(t.NWARP)
        ]
        row_max = maxima[0]
        for w in range_constexpr(1, t.NWARP):
            row_max = fx.maxnumf(row_max, maxima[w])
        safe = fx.Float32(
            (row_max > fx.Float32(float("-inf"))).select(row_max, fx.Float32(0.0))
        )
        denom = fx.Float32(0.0)
        acc = fx.Vector.filled(16, 0.0, fx.Float32)
        for w in range_constexpr(t.NWARP):
            weight = fx.Float32(fx.exp2(maxima[w] - safe, fastmath="afn"))
            denom = denom + sums[w] * weight
            acc = acc + self.mem.load(
                w * t.MXFP4_O_BYTES + (row * t.MXFP4_O_STRIDE + dim) * 4,
                fx.Float32,
                16,
            ) * fx.full(16, weight, fx.Float32)
        # O and the denominator both carry 2**-MXFP4_P_EXP_SHIFT; the row's max
        # token has p = 1, so the denominator is a normal f32 when nonzero.
        inv = fx.Float32(
            (denom > fx.Float32(0.0)).select(
                fx.Float32(rcp_f32(denom)), fx.Float32(0.0)
            )
        )
        values = (acc * fx.full(16, inv, fx.Float32)).to(t.q_dtype)
        if row < t.query_group_size:
            head = ctx.kv_h * t.query_group_size + row
            if const_expr(t.cluster_merge):
                # A sequence's only task stores its rows; others leave partials
                # for the cluster's merge.
                direct = ctx.task_count == 1
            else:
                direct = fx.Int32(int(t.plan_free)) == 1
            if direct:
                fx.ptr_store(
                    values,
                    fx.add_offset(
                        fx.recast_iter(t.q_dtype, ctx.output_ptr),
                        fx.Int64(ctx.planned_seq) * ctx.stride_q_row
                        + head * ctx.stride_q_head
                        + dim,
                    ),
                )
            else:
                partial = (
                    fx.Int64(ctx.kv_h) * fx.Int64(ctx.capacity)
                    + fx.Int64(ctx.task_slot)
                ) * t.query_group_size + row
                fx.ptr_store(
                    values,
                    fx.add_offset(
                        fx.recast_iter(t.q_dtype, ctx.pout_ptr),
                        partial * t.head_dim + dim,
                    ),
                )
                if tid % 8 == 0:
                    # The shared reducer expects maxima in natural-log units.
                    pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
                    psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
                    pmax[partial] = row_max * fx.Float32(1.0 / LOG2E)
                    psum[partial] = denom * fx.Float32(2.0**MXFP4_P_EXP_SHIFT)

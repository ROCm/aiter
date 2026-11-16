# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Normalize PA output and store packed partition partials."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import SyncScope

from ..pa_decode_plan import ARRIVAL_COUNTER_STRIDE
from ..tensor_shim import buf_base_i64, buf_copy_store, ptr_buf_tensor
from ..utils import rcp_f32
from .context import logical_slots
from .op_softmax import tree_max
from .traits import (
    FP8_KV_DTYPES,
    LOG2E,
    MFMA_MNK,
    MXFP4_P_EXP_SHIFT,
    WAVE,
)

# Partitions whose partials a merging thread loads together.
MERGE_CHUNK = 8


def _agent_pointer(target, dtype):
    # Atomic accesses need natural alignment, not the element alignment.
    return fx.recast_iter(
        fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, dtype.width // 8),
        target,
    )


def _publish(target, value):
    """Store at device scope, for the merge in another CTA."""
    if const_expr(isinstance(value, fx.Vector)):
        value = value.bitcast(fx.Int64)[0]
        target = _agent_pointer(target, fx.Int64)
    else:
        target = _agent_pointer(target, fx.Float32)
    fx.generic_store(
        target,
        value,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=SyncScope.Agent,
    )


def _gather(target, dtype):
    """Device-scope load of another CTA's published partial."""
    return fx.generic_load(
        _agent_pointer(target, dtype),
        dtype=dtype,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=SyncScope.Agent,
    )


class PaDecodeEpilogue:
    def __init__(self, ctx, lds):
        self.ctx = ctx
        self.lds = lds
        self.traits = ctx.traits

    def init_outputs(self):
        self.pmax = fx.recast_iter(fx.Float32, self.ctx.pmax_ptr)
        self.psum = fx.recast_iter(fx.Float32, self.ctx.psum_ptr)
        self.pout = fx.recast_iter(self.traits.OUT_DTYPE, self.ctx.pout_ptr)
        if const_expr(self.traits.buffer_plan_output):
            # Widen before multiplying: packed storage can exceed 2 GiB.
            pout_slot = fx.Int64(self.ctx.kv_h) * fx.Int64(
                logical_slots(self.ctx.traits)
            ) + fx.Int64(self.ctx.part)
            pout_slot_bytes = self.traits.TOTAL_ROWS * self.traits.head_dim * 2
            pout_base = buf_base_i64(self.ctx.pout_ptr) + pout_slot * fx.Int64(
                pout_slot_bytes
            )
            self.pout_buffer = ptr_buf_tensor(
                pout_base,
                self.traits.OUT_DTYPE,
                n=self.traits.TOTAL_ROWS * self.traits.head_dim,
                unit_elems=self.traits.OP_ELEMS,
                # BF16 views may be only 2-byte aligned, even for 8-byte stores.
                unit_stride=1,
                num_records_bytes=pout_slot_bytes,
            )

    def _emit(self, o_norm, sub, global_row):
        if const_expr(self.traits.DIRECT and self.traits.query_length == 1):
            # One task per row: the output has the query's strides.
            head = self.ctx.kv_h * self.traits.query_group_size + global_row
            offset = (
                self.ctx.seq * self.ctx.stride_q_row
                + head * self.ctx.stride_q_head
                + sub * self.traits.OP_ELEMS
            )
            fx.ptr_store(o_norm, fx.add_offset(self.pout, offset))
        elif const_expr(self.traits.DIRECT):
            # Query positions are flattened into the CTA's rows.
            query_pos = global_row // self.traits.query_group_size
            head = self.ctx.kv_h * self.traits.query_group_size + (
                global_row - query_pos * self.traits.query_group_size
            )
            offset = (
                (self.ctx.seq * self.traits.query_length + query_pos)
                * self.ctx.stride_q_row
                + head * self.ctx.stride_q_head
                + sub * self.traits.OP_ELEMS
            )
            fx.ptr_store(o_norm, fx.add_offset(self.pout, offset))
        elif const_expr(self.traits.buffer_plan_output):
            # Row guards keep the full 8-byte store within this slot.
            pout_offset = global_row * self.traits.head_dim + sub * self.traits.OP_ELEMS
            buf_copy_store(
                self.pout_buffer,
                pout_offset,
                o_norm,
                elem=self.traits.OUT_DTYPE,
                unit_elems=self.traits.OP_ELEMS,
                cache_modifier=0,
            )
        else:
            base = self.ctx.partial_slot * self.traits.TOTAL_ROWS + global_row
            pout_offset = base * self.traits.head_dim + sub * self.traits.OP_ELEMS
            fx.ptr_store(o_norm, fx.add_offset(self.pout, pout_offset))

    def _final_offset(self, global_row, dim):
        """Final output element of a CTA row (MTP position, GQA head)."""
        query_pos = global_row // self.traits.query_group_size
        head = self.ctx.kv_h * self.traits.query_group_size + (
            global_row - query_pos * self.traits.query_group_size
        )
        return (
            (self.ctx.seq * self.traits.query_length + query_pos)
            * self.ctx.stride_q_row
            + head * self.ctx.stride_q_head
            + dim
        )

    def _sink_logits(self, global_rows):
        """Base-2 sink logits of CTA rows, or None without sinks. Every row
        maps to a query head of this KV head, so the loads stay in bounds."""
        if const_expr(self.traits.SINK_DTYPE is None):
            return None
        sinks = fx.recast_iter(self.traits.SINK_DTYPE, self.ctx.sinks_ptr)
        group = self.traits.query_group_size
        logits = []
        for global_row in global_rows:
            query_pos = global_row // group
            head = self.ctx.kv_h * group + (global_row - query_pos * group)
            logits.append(fx.Float32(sinks[head]) * fx.Float32(LOG2E))
        return logits

    def _sink_norm(self, top, total, sink):
        """Scale from (O, sum) relative to the finite max ``top`` to final rows
        with the sink's unit-value mass, as the reducer folds it: a +inf sink
        takes all mass, a -inf one none, an empty row stays zero."""
        neg_inf = self.ctx.NEG_INF
        zero = self.ctx.ZERO_F
        new_top = top.maximumf(sink)
        safe = fx.Float32((new_top > neg_inf).select(new_top, zero))
        keep = fx.Float32((top > neg_inf).select(fx.exp2(top - safe), zero))
        mass = fx.Float32(
            (sink == safe).select(fx.Float32(1.0), fx.Float32(fx.exp2(sink - safe)))
        )
        total = total * keep + mass
        return fx.Float32(
            (total > zero).select(keep * fx.Float32(rcp_f32(total)), zero)
        )

    @flyc.jit
    def store(self, o_final):
        # Keep pointer writes as SSA operands of the row/warp guards.
        pmax = self.pmax
        psum = self.psum
        # Each lane stores four head-dim values for one query row.
        for m in range_constexpr(self.traits.M_TILES):
            # Flat (mtp, gqa) query-row for this lane.
            row = m * MFMA_MNK + self.ctx.lane16
            global_row = self.ctx.query_begin * self.traits.query_group_size + row
            l_row = o_final[self.traits.l_slot(m)]
            safe_l = (l_row > self.ctx.ZERO_F).select(l_row, fx.Float32(1.0))
            inv_l = fx.Float32(rcp_f32(safe_l))
            if const_expr(
                self.traits.per_token_kv or self.traits.kv_dtype not in FP8_KV_DTYPES
            ):
                o_scale = inv_l
            else:
                inv_fp8 = fx.Float32(1.0 / self.traits.FP8_MAX)
                o_scale = inv_l * (self.ctx.v_scale_f * inv_fp8)
            o_scale_b = fx.full(self.traits.OP_ELEMS, o_scale, fx.Float32)

            for vh in range_constexpr(self.traits.VHE_CHUNKS):
                o_slot = self.traits.o_slot(m, vh)
                o_norm = (o_final[o_slot] * o_scale_b).to(self.traits.OUT_DTYPE)
                head_base = (
                    vh * (self.traits.NWARP * MFMA_MNK)
                    + self.ctx.warp * MFMA_MNK
                    + self.ctx.rgroup * self.traits.OP_ELEMS
                )
                sub = head_base // self.traits.OP_ELEMS
                if row < self.traits.CTA_ROWS:
                    self._emit(o_norm, sub, global_row)

            if const_expr(not self.traits.DIRECT):  # noqa: SIM102
                if self.ctx.warp == 0 and self.ctx.rgroup == 0:
                    base = self.ctx.partial_slot * self.traits.TOTAL_ROWS + global_row
                    if row < self.traits.CTA_ROWS:
                        # The shared reducer expects maxima in natural-log units.
                        pmax[base] = o_final[self.traits.m_slot(m)] * fx.Float32(
                            1.0 / LOG2E
                        )
                        psum[base] = l_row

    @flyc.jit
    def store_mxfp4(self, o_final):
        """MXFP4: each wave holds (max, sum, O over all dims) of its own tokens. Per
        M-tile, one barrier publishes them and each of the head x 8-dim threads
        merges its slice; a second barrier frees the LDS for the next M-tile."""
        # Keep pointer writes as SSA operands of the guards (as in ``store``).
        pmax = self.pmax
        psum = self.psum
        # The wave's V staging is free: its O goes there, [head][dim] f32.
        wave_bytes = self.traits.MXFP4_WAVE_BYTES
        o_off = self.traits.sMxfp4VBf16_off + self.ctx.warp * wave_bytes
        # Rows are padded by 16 B: unpadded, the 16 lanes of a row group would
        # write the same LDS banks.
        row_bytes = self.traits.head_dim * 4 + 16
        # A thread merges 8 dims of a row: head_dim 256 takes two passes of 8
        # rows, head_dim 64 leaves half of the threads idle.
        dim_threads = self.traits.head_dim // 8
        pass_rows = min(MFMA_MNK, self.traits.BLOCK_THREADS // dim_threads)
        # head_dim 384 merges 5 rows a pass; a fourth pass takes the last row.
        passes = -(-MFMA_MNK // pass_rows)
        pass_row = self.ctx.tid // dim_threads
        dim8 = self.ctx.tid % dim_threads
        # Sink loads issue before the publication barriers.
        sink_rows = self._sink_logits(
            [
                self.ctx.query_begin * self.traits.query_group_size
                + m * MFMA_MNK
                + p * pass_rows
                + pass_row
                for m in range_constexpr(self.traits.M_TILES)
                for p in range_constexpr(passes)
            ]
        )
        for m in range_constexpr(self.traits.M_TILES):
            if const_expr(m > 0):
                gpu.barrier()
            l_w = o_final[self.traits.l_slot(m)]
            for sh in (16, 32):
                l_w = l_w + l_w.shuffle_xor(
                    sh, WAVE
                )  # this lane's tokens -> the wave's
            if self.ctx.rgroup == 0:
                self.lds.store_wave(
                    self.traits.sLmax_off,
                    self.ctx.lane16,
                    self.ctx.warp,
                    o_final[self.traits.m_slot(m)],
                )
                self.lds.store_wave(
                    self.traits.sLsum_off, self.ctx.lane16, self.ctx.warp, l_w
                )
            if m * MFMA_MNK + self.ctx.lane16 < self.traits.CTA_ROWS:
                for vh in range_constexpr(self.traits.VHE_CHUNKS):
                    self.lds.store(
                        o_off
                        + self.ctx.lane16 * row_bytes
                        + (vh * MFMA_MNK + self.ctx.rgroup * 4) * 4,
                        fx.Float32,
                        o_final[self.traits.o_slot(m, vh)],
                    )
            gpu.barrier()
            for p in range_constexpr(passes):
                row = p * pass_rows + pass_row
                owned = m * MFMA_MNK + row < self.traits.CTA_ROWS
                if const_expr(passes * pass_rows > MFMA_MNK):
                    owned = owned & (row < MFMA_MNK)
                if const_expr(self.traits.query_length % self.traits.query_splits):
                    owned = owned & (m * MFMA_MNK + row >= self.ctx.first_stored_row)
                if const_expr(pass_rows * dim_threads < self.traits.BLOCK_THREADS):
                    owned = owned & (pass_row < pass_rows)
                if owned:
                    ms = self.lds.load_wave(self.traits.sLmax_off, row)
                    ls = self.lds.load_wave(self.traits.sLsum_off, row)
                    h_max = tree_max(
                        [ms[w] for w in range(self.traits.NWARP)], self.ctx.fm_nnan
                    )
                    h_safe = (h_max > self.ctx.NEG_INF).select(h_max, self.ctx.ZERO_F)
                    wts = [
                        fx.Float32(fx.exp2(ms[w] - h_safe))
                        for w in range_constexpr(self.traits.NWARP)
                    ]
                    h_sum = self.ctx.ZERO_F
                    o = fx.full(8, 0.0, fx.Float32)
                    for w in range_constexpr(self.traits.NWARP):
                        h_sum = h_sum + ls[w] * wts[w]
                        part = self.lds.load(
                            self.traits.sMxfp4VBf16_off
                            + w * wave_bytes
                            + row * row_bytes
                            + dim8 * 8 * 4,
                            fx.Float32,
                            8,
                        )
                        o = o + part * fx.full(8, wts[w], fx.Float32)
                    inv = (h_sum > self.ctx.ZERO_F).select(
                        fx.Float32(rcp_f32(h_sum)), self.ctx.ZERO_F
                    )
                    if const_expr(sink_rows is not None):
                        # Final rows fold in the sink; published partials do not.
                        final_inv = self._sink_norm(
                            h_max,
                            h_sum,
                            sink_rows[m * passes + p],
                        )
                        if const_expr(self.traits.ONE_SLOT_PLAN):
                            inv = (self.ctx.task_count == 1).select(final_inv, inv)
                        else:
                            inv = final_inv
                    # O holds PV * 2**-MXFP4_P_EXP_SHIFT (``single_query_tile_mxfp4``):
                    # restore it here, after the normalization (the row's max token has
                    # p = 1, so h_sum >= 1).
                    inv = inv * fx.Float32(2.0**MXFP4_P_EXP_SHIFT)
                    o_norm = (o * fx.full(8, inv, fx.Float32)).to(self.traits.OUT_DTYPE)
                    global_row = (
                        self.ctx.query_begin * self.traits.query_group_size
                        + m * MFMA_MNK
                        + row
                    )
                    halves = [
                        fx.Vector.from_elements(
                            [o_norm[half * 4 + i] for i in range_constexpr(4)],
                            dtype=self.traits.OUT_DTYPE,
                        )
                        for half in range_constexpr(2)
                    ]
                    base = self.ctx.partial_slot * self.traits.TOTAL_ROWS + global_row
                    if const_expr(self.traits.ONE_SLOT_PLAN):
                        # A sequence's only task writes its final rows.
                        if self.ctx.task_count == 1:
                            out = fx.recast_iter(
                                self.traits.OUT_DTYPE, self.ctx.output_ptr
                            )
                            for half in range_constexpr(2):
                                fx.ptr_store(
                                    halves[half],
                                    fx.add_offset(
                                        out,
                                        self._final_offset(
                                            global_row, (dim8 * 2 + half) * 4
                                        ),
                                    ),
                                )
                        else:
                            for half in range_constexpr(2):
                                _publish(
                                    fx.add_offset(
                                        self.pout,
                                        base * self.traits.head_dim
                                        + (dim8 * 2 + half) * 4,
                                    ),
                                    halves[half],
                                )
                            if dim8 == 0:
                                _publish(fx.add_offset(pmax, base), h_max)
                                _publish(fx.add_offset(psum, base), h_sum)
                    else:
                        for half in range_constexpr(2):
                            self._emit(halves[half], dim8 * 2 + half, global_row)
                        if const_expr(not self.traits.DIRECT) and dim8 == 0:
                            # The shared reducer expects maxima in natural-log units.
                            pmax[base] = h_max * fx.Float32(1.0 / LOG2E)
                            psum[base] = h_sum
        if const_expr(self.traits.ONE_SLOT_PLAN):  # noqa: SIM102
            if self.ctx.task_count > 1:
                self._arrive()

    @flyc.jit
    def _arrive(self):
        """Count this CTA's published partials; the sequence's last CTA to
        arrive resets the counter (zero across launches and graph replays) and
        merges every task."""
        ctx, t = self.ctx, self.traits
        # One 128-byte line per (sequence, KV cache head); a head split's
        # halves count in separate words of their head's line.
        counter = fx.add_offset(
            fx.recast_iter(fx.Int32, ctx.counters_ptr),
            (ctx.seq * ctx.n_kv + ctx.kv_h)
            * (ARRIVAL_COUNTER_STRIDE // t.KV_HEAD_SPLITS),
        )
        # Every wave retires its device-scope partials before the arrival.
        fx.rocdl.s_waitcnt(vmcnt=0)
        gpu.barrier()
        if ctx.tid == 0:
            arrived = fx.atomic_add(counter, fx.Int32(1), syncscope=SyncScope.Agent)
            last = fx.Int32(arrived == ctx.task_count * t.query_splits - 1)
            if last != 0:
                fx.generic_store(
                    counter,
                    fx.Int32(0),
                    memory_order=fx.AtomicOrdering.Monotonic,
                    syncscope=SyncScope.Agent,
                )
            self.lds.store(
                t.sLmax_off, fx.Int32, fx.Vector.from_elements([last], dtype=fx.Int32)
            )
        gpu.barrier()
        if self.lds.load(t.sLmax_off, fx.Int32, 1)[0] != 0:
            self._merge()

    @flyc.jit
    def _merge(self):
        """Write this sequence's final rows (every query split) from its
        tasks' partials: (base-2 max, sum, O / sum) per row, folded
        ``MERGE_CHUNK`` tasks at a time with a running max."""
        ctx, t = self.ctx, self.traits
        rows = t.TOTAL_ROWS
        dim_threads = t.head_dim // 8
        pass_rows = t.BLOCK_THREADS // dim_threads
        row_in = ctx.tid // dim_threads
        dim8 = ctx.tid % dim_threads
        count = ctx.task_count
        first = ctx.kv_h * logical_slots(ctx.traits) + ctx.first_task
        neg_inf = fx.Float32(float("-inf"))
        zero = self.ctx.ZERO_F
        out = fx.recast_iter(t.OUT_DTYPE, ctx.output_ptr)
        for ps in range_constexpr(-(-rows // pass_rows)):
            # Rows past the CTA's merge the last one again, unstored.
            row = fx.min(ps * pass_rows + row_in, fx.Int32(rows - 1))
            init = [neg_inf, zero, fx.full(8, 0.0, fx.Float32)]
            for c, state in range(0, count, MERGE_CHUNK, init=init):
                top, total, acc = state
                c = fx.Int32(c)
                parts = [fx.min(c + p, count - 1) for p in range_constexpr(MERGE_CHUNK)]
                live = [c + p < count for p in range_constexpr(MERGE_CHUNK)]
                stats = [fx.Int64(first + q) * rows + fx.Int64(row) for q in parts]
                maxima = [
                    live[p].select(
                        _gather(fx.add_offset(self.pmax, stats[p]), fx.Float32),
                        neg_inf,
                    )
                    for p in range_constexpr(MERGE_CHUNK)
                ]
                sums = [
                    _gather(fx.add_offset(self.psum, stats[p]), fx.Float32)
                    for p in range_constexpr(MERGE_CHUNK)
                ]
                values = []
                for p in range_constexpr(MERGE_CHUNK):
                    elem = stats[p] * t.head_dim + fx.Int64(dim8 * 8)
                    words = [
                        _gather(fx.add_offset(self.pout, elem + 4 * i), fx.Int64)
                        for i in range_constexpr(2)
                    ]
                    values.append(
                        fx.Vector.from_elements(words, dtype=fx.Int64)
                        .bitcast(t.OUT_DTYPE)
                        .to(fx.Float32)
                    )
                new_top = tree_max([top] + maxima, self.ctx.fm_nnan)
                safe = fx.Float32((new_top > neg_inf).select(new_top, zero))
                keep = fx.Float32((top > neg_inf).select(fx.exp2(top - safe), zero))
                total = total * keep
                acc = acc * fx.full(8, keep, fx.Float32)
                for p in range_constexpr(MERGE_CHUNK):
                    weight = fx.Float32(
                        (maxima[p] > neg_inf).select(
                            fx.Float32(fx.exp2(maxima[p] - safe)) * sums[p], zero
                        )
                    )
                    total = total + weight
                    acc = acc + values[p] * fx.full(8, weight, fx.Float32)
                results = yield [new_top, total, acc]
            total = fx.Float32(results[1])
            acc = fx.Vector(results[2])
            inv = fx.Float32((total > zero).select(fx.Float32(rcp_f32(total)), zero))
            if const_expr(t.SINK_DTYPE is not None):
                inv = self._sink_norm(
                    fx.Float32(results[0]), total, self._sink_logits([row])[0]
                )
            o_norm = (acc * fx.full(8, inv, fx.Float32)).to(t.OUT_DTYPE)
            if ps * pass_rows + row_in < rows:
                for half in range_constexpr(2):
                    fx.ptr_store(
                        fx.Vector.from_elements(
                            [o_norm[half * 4 + i] for i in range_constexpr(4)],
                            dtype=t.OUT_DTYPE,
                        ),
                        fx.add_offset(
                            out, self._final_offset(row, dim8 * 8 + half * 4)
                        ),
                    )

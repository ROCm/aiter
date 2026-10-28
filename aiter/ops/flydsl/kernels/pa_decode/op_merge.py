# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 FlyDSL Project Contributors

"""Merge gfx1250 PA decode partitions in the tile kernel.

Both schemes reproduce the register reducer's math and partition order, so
their outputs match the shared reducer bit for bit.

``Gfx1250PartitionMerge`` merges in the last-arriving CTA: each task of a
multi-partition sequence publishes the reducer's partials, then increments the
(sequence, KV head) arrival counter. The last CTA to arrive resets the counter
and merges every partition, so the counters stay zero across launches and
graph replays. Partials and the counter use device-scope (SCOPE_DEV) accesses,
which reach the device coherence point without cache maintenance. Agent-scope
fences would also order them, but they lower to a device-wide L2 writeback
that serializes across the grid (sparse decode 64 rows: tile kernel 7.5 -> 54
us).

``Gfx1250ClusterMerge`` runs a sequence's tasks as one workgroup cluster,
whose CTAs share an XCC and hence an L2. Plain partial stores, a cluster-scope
release, the cluster barrier and a cluster-scope acquire (s_wait_storecnt 0,
s_barrier -3, global_inv SCOPE_SE) publish them to the cluster without an L2
writeback; each CTA then writes its share of the sequence's rows.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import SyncScope, cluster

from ..pa_decode_plan import ARRIVAL_COUNTER_STRIDE
from ..utils import rcp_f32
from .traits import LOG2E

# Partition bound of a merged sequence; each one is an unrolled load per row.
MERGE_PARTITIONS = 8
# Largest head dimension whose merge loads stay within 64 VGPRs per row group.
MERGE_HEAD_DIM = 512
# Largest gfx1250 workgroup cluster.
CLUSTER_PARTITIONS = 16


def _global_pointer(target, dtype):
    # Atomic accesses need natural alignment, not the element alignment.
    return fx.recast_iter(
        fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, dtype.width // 8),
        target,
    )


def _softmax_weight(t, part_sum, part_max):
    # Wave-wide reducer statistics: lane p owns partition p.
    neg_inf, zero = fx.Float32(float("-inf")), fx.Float32(0.0)
    offsets = (16, 8, 4, 2, 1)
    lane_max = part_max
    for offset in offsets:
        lane_max = lane_max.maximumf(
            lane_max.shuffle_xor(fx.Int32(offset), fx.Int32(t.WAVE))
        )
    safe_max = (lane_max > neg_inf).select(lane_max, zero)
    scale = (part_max > neg_inf).select(
        fx.exp2((part_max - safe_max) * fx.Float32(LOG2E), fastmath="fast"), zero
    )
    scaled = part_sum * scale
    exp_sum = scaled
    for offset in offsets:
        exp_sum = fx.Float32(
            exp_sum.addf(
                exp_sum.shuffle_xor(fx.Int32(offset), fx.Int32(t.WAVE)),
                fastmath="fast",
            )
        )
    inv = fx.Float32(rcp_f32((exp_sum > zero).select(exp_sum, fx.Float32(1.0))))
    return scaled * inv


def _combine(t, values, weight):
    # Increasing partition order with separate multiply/add rounding, as in
    # the reducer.
    acc = fx.full(t.head_dim // t.WAVE, 0.0, fx.Float32)
    for part, value in enumerate(values):
        acc = acc + value.to(fx.Float32) * fx.gpu.shuffle_idx(weight, part, t.WAVE)
    return acc


class Gfx1250PartitionMerge:
    def __init__(self, ctx, mem):
        self.ctx, self.t, self.mem = ctx, ctx.traits, mem

    def output_offset(self, row, d):
        """Final output element of query row ``row`` (MTP position, GQA head)."""
        ctx, t = self.ctx, self.t
        return (
            (fx.Int64(ctx.planned_seq) * t.query_length + row // t.query_group_size)
            * ctx.stride_q_row
            + (ctx.kv_h * t.query_group_size + row % t.query_group_size)
            * ctx.stride_q_head
            + d
        )

    def publish(self, target, value):
        """Store a partial at device scope for a merge on another CTA."""
        if const_expr(isinstance(value, fx.Vector)):
            words = value.bitcast(fx.Int64)
            target = _global_pointer(target, fx.Int64)
            for i in range_constexpr(words.numel):
                fx.generic_store(
                    fx.add_offset(target, i),
                    words[i],
                    memory_order=fx.AtomicOrdering.Monotonic,
                    syncscope=SyncScope.Agent,
                )
        else:
            fx.generic_store(
                _global_pointer(target, fx.Float32),
                fx.Float32(value),
                memory_order=fx.AtomicOrdering.Monotonic,
                syncscope=SyncScope.Agent,
            )

    def _gather(self, target, dtype, n=1):
        # Device-scope loads observe another CTA's published partials.
        if const_expr(n == 1):
            return fx.generic_load(
                _global_pointer(target, dtype),
                dtype=dtype,
                memory_order=fx.AtomicOrdering.Monotonic,
                syncscope=SyncScope.Agent,
            )
        word = fx.Int64 if n * dtype.width % 64 == 0 else fx.Int32
        target = _global_pointer(target, word)
        words = [
            fx.generic_load(
                fx.add_offset(target, i),
                dtype=word,
                memory_order=fx.AtomicOrdering.Monotonic,
                syncscope=SyncScope.Agent,
            )
            for i in range_constexpr(n * dtype.width // word.width)
        ]
        return fx.Vector.from_elements(words, dtype=word).bitcast(dtype)

    @flyc.jit
    def arrive(self):
        """Count this CTA's published partials; the last arrival merges them."""
        ctx, t = self.ctx, self.t
        # One 128-byte line per counter: arrivals of other sequences sharing
        # a line would contend with these at the device coherence point.
        counter = fx.add_offset(
            fx.recast_iter(fx.Int32, ctx.counters_ptr),
            (ctx.planned_seq * ctx.n_kv + ctx.kv_h) * ARRIVAL_COUNTER_STRIDE,
        )
        # Retired device-scope stores are visible to every CTA, so each wave
        # retires its partials before the barrier that precedes the arrival.
        fx.rocdl.s_wait_storecnt(0)
        gpu.barrier()
        if ctx.tid == 0:
            arrived = fx.atomic_add(counter, fx.Int32(1), syncscope=SyncScope.Agent)
            last = fx.Int32(arrived == ctx.task_count * t.query_splits - 1)
            if last != 0:
                # Every other CTA of this sequence has arrived in this launch.
                fx.generic_store(
                    counter,
                    fx.Int32(0),
                    memory_order=fx.AtomicOrdering.Monotonic,
                    syncscope=SyncScope.Agent,
                )
            self.mem.store(t.stats_offset, last, fx.Int32)
        gpu.barrier()
        if self.mem.load(t.stats_offset, fx.Int32)[0] != 0:
            self.merge()

    @flyc.jit
    def merge(self):
        """Write final rows from every partition of this sequence and KV head."""
        ctx, t = self.ctx, self.t
        rows = t.rows * t.query_splits
        per_wave = -(-rows // t.NWARP)
        elems = t.head_dim // t.WAVE
        # Rows whose MERGE_PARTITIONS loads are in flight together.
        group = max(1, 16 // elems)
        pmax = fx.recast_iter(fx.Float32, ctx.pmax_ptr)
        psum = fx.recast_iter(fx.Float32, ctx.psum_ptr)
        pout = fx.recast_iter(t.q_dtype, ctx.pout_ptr)
        output = fx.recast_iter(t.q_dtype, ctx.output_ptr)
        first = ctx.kv_h * fx.Int32(gpu.grid_dim.x) + ctx.first_task
        count = ctx.task_count
        live = ctx.lane < count
        stats_slot = fx.Int64(first + live.select(ctx.lane, fx.Int32(0)))
        row_ids = [ctx.warp * per_wave + i for i in range_constexpr(per_wave)]
        load_rows = row_ids
        if const_expr(rows % t.NWARP):
            load_rows = [fx.min(row, fx.Int32(rows - 1)) for row in row_ids]
        weights = []
        for row in load_rows:
            offset = stats_slot * rows + fx.Int64(row)
            weights.append(
                _softmax_weight(
                    t,
                    live.select(
                        self._gather(fx.add_offset(psum, offset), fx.Float32),
                        fx.Float32(0.0),
                    ),
                    live.select(
                        self._gather(fx.add_offset(pmax, offset), fx.Float32),
                        fx.Float32(float("-inf")),
                    ),
                )
            )
        # Partitions past the count reread the last one with zero weight.
        slots = [
            fx.Int64(first + fx.min(fx.Int32(part), count - 1))
            for part in range_constexpr(MERGE_PARTITIONS)
        ]
        lane_elem = ctx.lane * elems
        for begin in range_constexpr(0, per_wave, group):
            members = range_constexpr(begin, min(begin + group, per_wave))
            values = [
                [
                    self._gather(
                        fx.add_offset(
                            pout,
                            (slot * rows + fx.Int64(load_rows[i])) * t.head_dim
                            + fx.Int64(lane_elem),
                        ),
                        t.q_dtype,
                        elems,
                    )
                    for slot in slots
                ]
                for i in members
            ]
            for i in members:
                acc = _combine(t, values[i - begin], weights[i])
                target = fx.add_offset(
                    output, self.output_offset(row_ids[i], lane_elem)
                )
                if const_expr(rows % t.NWARP):
                    if row_ids[i] < rows:
                        fx.ptr_store(acc.to(t.q_dtype), target)
                else:
                    fx.ptr_store(acc.to(t.q_dtype), target)


class Gfx1250ClusterMerge:
    """Merge one (sequence, KV head) across the CTAs of its cluster.

    The sequence's ``count`` tasks occupy the first CTAs of its cluster of
    ``t.cluster_merge``, so ``count`` and the barrier are cluster-uniform; the
    other CTAs only merge. Partials sit at packed slots from ``first`` on.
    """

    def __init__(self, t, pmax, psum, pout, output, stride_q_row, stride_q_head):
        self.t = t
        self.pmax, self.psum, self.pout, self.output = pmax, psum, pout, output
        self.stride_q_row, self.stride_q_head = stride_q_row, stride_q_head
        # Each CTA writes ``share`` rows of its query split, ``per_wave`` per wave.
        self.share = -(-t.rows // t.cluster_merge)
        self.per_wave = -(-self.share // t.NWARP)

    @flyc.jit
    def run(self, seq, part, first, count, capacity):
        """Publish this CTA's partials, then write rows ``part`` / P of the sequence."""
        fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Release)
        cluster.cluster_barrier()
        fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Acquire)
        wave_begin = fx.Int32(fx.rocdl.wave_id()) * self.per_wave
        if wave_begin < self.share:
            self._merge_rows(seq, part * self.share, wave_begin, first, count, capacity)

    @flyc.jit
    def _merge_rows(self, seq, share_begin, wave_begin, first, count, capacity):
        t, share, per_wave = self.t, self.share, self.per_wave
        elems = t.head_dim // t.WAVE
        # Rows whose partition loads are in flight together (64 VGPRs).
        group = max(1, 128 // (t.cluster_merge * elems))
        lane = fx.Int32(gpu.thread_id("x")) % t.WAVE
        kv_query = fx.Int32(gpu.block_id("y"))
        kv_h = kv_query // t.query_splits
        rows = t.rows * t.query_splits
        pmax = fx.recast_iter(fx.Float32, self.pmax)
        psum = fx.recast_iter(fx.Float32, self.psum)
        pout = fx.recast_iter(t.q_dtype, self.pout)
        output = fx.recast_iter(t.q_dtype, self.output)
        first_slot = kv_h * capacity + first
        live = lane < count
        stats_slot = fx.Int64(first_slot + live.select(lane, fx.Int32(0)))
        # Rows of this CTA's query split; the last share may end past them.
        local = [wave_begin + i for i in range_constexpr(per_wave)]
        row_ids = [share_begin + row for row in local]
        split_row = (kv_query % t.query_splits) * t.rows
        load_rows = [
            split_row + fx.min(row_id, fx.Int32(t.rows - 1)) for row_id in row_ids
        ]
        # Partitions past the count reread the last one with zero weight.
        slots = [
            fx.Int64(first_slot + fx.min(fx.Int32(p), count - 1))
            for p in range_constexpr(t.cluster_merge)
        ]
        lane_elem = lane * elems
        vector = fx.PointerType.get(
            t.q_dtype.ir_type, fx.AddressSpace.Global, elems * t.q_dtype.width // 8
        )

        def load_values(i):
            return [
                fx.Vector(
                    fx.ptr_load(
                        fx.recast_iter(
                            vector,
                            fx.add_offset(
                                pout,
                                (slot * rows + fx.Int64(load_rows[i])) * t.head_dim
                                + fx.Int64(lane_elem),
                            ),
                        ),
                        result_type=fx.Vector.make_type(elems, t.q_dtype),
                    )
                )
                for slot in slots
            ]

        # Issue the first rows' partials with the statistics, so one L2 round
        # trip covers both.
        stats = []
        for row in load_rows:
            offset = stats_slot * rows + fx.Int64(row)
            stats.append((fx.Float32(psum[offset]), fx.Float32(pmax[offset])))
        values = [load_values(i) for i in range_constexpr(min(group, per_wave))]
        weights = [
            _softmax_weight(
                t,
                live.select(part_sum, fx.Float32(0.0)),
                live.select(part_max, fx.Float32(float("-inf"))),
            )
            for part_sum, part_max in stats
        ]
        for begin in range_constexpr(0, per_wave, group):
            members = range_constexpr(begin, min(begin + group, per_wave))
            if const_expr(begin > 0):
                values = [load_values(i) for i in members]
            for i in members:
                acc = _combine(t, values[i - begin], weights[i])
                row = load_rows[i]
                target = fx.recast_iter(
                    vector,
                    fx.add_offset(
                        output,
                        (fx.Int64(seq) * t.query_length + row // t.query_group_size)
                        * self.stride_q_row
                        + (kv_h * t.query_group_size + row % t.query_group_size)
                        * self.stride_q_head
                        + lane_elem,
                    ),
                )
                if const_expr(share % per_wave or t.rows % t.cluster_merge):
                    if (local[i] < share) & (row_ids[i] < t.rows):
                        fx.ptr_store(acc.to(t.q_dtype), target)
                else:
                    fx.ptr_store(acc.to(t.q_dtype), target)

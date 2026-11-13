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

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import SyncScope, cluster
from flydsl.expr.typing import ReductionOp, T

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
        first = ctx.kv_h * ctx.capacity + ctx.first_task
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


def _cluster_sync():
    """Publish global stores to the cluster and observe the other CTAs' stores."""
    fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Release)
    cluster.cluster_barrier()
    fx.memory_fence(syncscope="cluster", ordering=fx.AtomicOrdering.Acquire)


class Gfx1250ClusterMerge:
    """Merge one (sequence, KV head) across the CTAs of its cluster.

    The sequence's ``count`` tasks occupy the first CTAs of its cluster of
    ``t.cluster_merge``, so ``count`` and the barrier are cluster-uniform; the
    other CTAs only merge. Partials sit at packed slots from ``first`` on.
    With ``quant`` the merging waves quantize their rows after a second
    cluster barrier, which the other waves also reach.
    """

    def __init__(
        self, t, pmax, psum, pout, output, stride_q_row, stride_q_head, quant=None
    ):
        self.t = t
        self.pmax, self.psum, self.pout, self.output = pmax, psum, pout, output
        self.stride_q_row, self.stride_q_head = stride_q_row, stride_q_head
        # Each CTA writes ``share`` rows of its query split, ``per_wave`` per wave.
        self.share = -(-t.rows // t.cluster_merge)
        self.per_wave = -(-self.share // t.NWARP)
        self.quant = quant

    @flyc.jit
    def run(self, seq, part, first, count, capacity):
        """Publish this CTA's partials, then write rows ``part`` / P of the sequence."""
        _cluster_sync()
        wave_begin = fx.Int32(fx.rocdl.wave_id()) * self.per_wave
        if wave_begin < self.share:
            self._merge_rows(seq, part * self.share, wave_begin, first, count, capacity)
        else:
            if const_expr(self.quant is not None):
                # A wave without a row still reaches store_cluster's barrier.
                self.quant.sync()

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
        # Contiguous scratch can start at an offset, and final output heads
        # can have odd strides. Both buffers guarantee only element alignment.
        vector = fx.PointerType.get(
            t.q_dtype.ir_type, fx.AddressSpace.Global, t.q_dtype.width // 8
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
        ragged = share % per_wave or t.rows % t.cluster_merge
        quant_rows = []
        for begin in range_constexpr(0, per_wave, group):
            members = range_constexpr(begin, min(begin + group, per_wave))
            if const_expr(begin > 0):
                values = [load_values(i) for i in members]
            for i in members:
                acc = _combine(t, values[i - begin], weights[i])
                row = load_rows[i]
                offset = (
                    (fx.Int64(seq) * t.query_length + row // t.query_group_size)
                    * self.stride_q_row
                    + (kv_h * t.query_group_size + row % t.query_group_size)
                    * self.stride_q_head
                    + lane_elem
                )
                valid = (local[i] < share) & (row_ids[i] < t.rows) if ragged else None
                if const_expr(self.quant is not None):
                    quant_rows.append((row, valid, acc.to(t.q_dtype), offset))
                else:
                    target = fx.recast_iter(vector, fx.add_offset(output, offset))
                    if const_expr(ragged):
                        if valid:
                            fx.ptr_store(acc.to(t.q_dtype), target)
                    else:
                        fx.ptr_store(acc.to(t.q_dtype), target)
        if const_expr(self.quant is not None):
            # The first partition's maxima of the sequence's rows; only the
            # owner of a row read its entry, before merging it.
            self.quant.store_cluster(
                seq,
                lane,
                fx.add_offset(pmax, fx.Int64(first_slot) * rows),
                quant_rows,
                fx.recast_iter(vector, output),
            )


def _store_values(quant, values, offsets, output, inv_scale):
    for value, offset in zip(values, offsets):
        if output is not None:
            fx.ptr_store(value, fx.add_offset(output, offset))
        if inv_scale is not None:
            quant._store_fp8(value, offset, inv_scale)


class Gfx1250OutputQuant:
    """Per-token OCP FP8 output of gfx1250 PA from registers.

    A token is one query position of a sequence: ``query_group_size`` rows of
    ``head_dim`` values. pa_decode fuses the quant only for one KV head, so a
    CTA or a cluster owns every row of its tokens. Owners round their final
    values to the output dtype in registers, so the FP8 bytes and descale
    equal the separate per-token quant of the stored output.

    * ``store_cta``: a CTA holds its tokens in registers. Lanes, waves and
      16-row tiles split them into units of gcd(GQA, 16) rows, which never
      straddle a token; unit maxima meet in LDS behind one workgroup barrier.
    * ``store_cluster``: a cluster merge in which each wave owns whole rows.
      A wave keeps a row's maximum in the first partition's ``pmax`` entry of
      the row, which only that wave has read; one cluster barrier publishes
      them.

    The BF16/FP16 output is written only when the caller keeps it.
    """

    def __init__(self, t, quantized_output, output_scale):
        self.t = t
        self.quantized = fx.recast_iter(fx.Uint8, quantized_output)
        self.scales = fx.recast_iter(fx.Float32, output_scale)
        self.unit = math.gcd(t.query_group_size, 16)

    def _lanes_max(self, value, unit=16):
        """Max over the lanes that differ from this one in bit 4 (the other
        half-wave) or below ``unit``; 16 spans the wave."""
        for shift in (16, 8, 4, 2, 1):
            if shift == 16 or shift < unit:
                value = fx.maxnumf(
                    value, value.shuffle_xor(fx.Int32(shift), fx.Int32(self.t.WAVE))
                )
        return value

    def _lane_max(self, values):
        lane_max = fx.Float32(0.0)
        for value in values:
            lane_max = fx.maxnumf(
                lane_max, fmath.absf(value.to(fx.Float32)).reduce(ReductionOp.MAX)
            )
        return lane_max

    def _store_fp8(self, values, offset, inv_scale):
        """Store output-dtype values as FP8 at element (= byte) ``offset``."""
        t = self.t
        floats = values.to(fx.Float32)
        words = []
        for base in range_constexpr(0, floats.numel, 4):
            word = fx.Int32(0).ir_value()
            for half in range_constexpr(min(2, (floats.numel - base) // 2)):
                pair = []
                for j in range_constexpr(2):
                    # The separate kernel's multiply and +-448 saturation.
                    value = fx.Float32(floats[base + 2 * half + j]) * inv_scale
                    value = fx.maxnumf(value, fx.Float32(-t.FP8_MAX))
                    value = fx.minnumf(value, fx.Float32(t.FP8_MAX))
                    pair.append(value.ir_value())
                word = fx.rocdl.cvt_pk_fp8_f32(T.i32, pair[0], pair[1], word, half)
            words.append(word)
        if const_expr(floats.numel == 2):
            # D64 cluster merges hold two values per lane.
            value, dtype = fx.Int32(words[0]).to(fx.Int16), fx.Int16
        else:
            value, dtype = fx.Vector.from_elements(words, dtype=fx.Int32), fx.Int32
        target = fx.recast_iter(
            fx.PointerType.get(dtype.ir_type, fx.AddressSpace.Global, dtype.width // 8),
            fx.add_offset(self.quantized, offset),
        )
        fx.ptr_store(value, target)

    @flyc.jit
    def _store(self, valid, values, offsets, output, inv_scale):
        """Store the rows of ``valid`` (None: every row): BF16/FP16 to
        ``output`` unless it is None, then FP8 unless ``inv_scale`` is None."""
        # Runtime branches call module functions: objects whose methods they
        # call would become branch state.
        if const_expr(valid is None):
            _store_values(self, values, offsets, output, inv_scale)
        else:
            if valid:
                _store_values(self, values, offsets, output, inv_scale)

    @flyc.jit
    def _finish(self, token, leader, token_max, valid, values, offsets, output):
        # Local names keep the guarded stores' operands SSA values.
        t, scales = self.t, self.scales
        scale = token_max * fx.Float32(1.0 / t.FP8_MAX)
        inv_scale = fx.Float32(rcp_f32(scale))
        if leader:
            scales[token] = scale
        self._store(valid, values, offsets, output, inv_scale)

    @flyc.jit
    def store_cta(self, mem, seq, lane, part, parts, tiles, output):
        """Write the tokens whose values all live in this CTA's registers.

        ``tiles`` lists this lane's (row, values, offsets) per 16-row tile:
        lane % 16 picks the row, and ``parts`` waves (this one is ``part``)
        and the two half-waves split its values.
        """
        t, g, unit = self.t, self.t.query_group_size, self.unit
        writer = lane == 0
        if const_expr(unit < 16):
            writer = (lane < 16) & (lane % unit == 0)
        valid = [None] * len(tiles)
        for i in range_constexpr(len(tiles)):
            row, values, _ = tiles[i]
            unit_max = fx.Vector.from_elements(
                [self._lanes_max(self._lane_max(values), unit)], dtype=fx.Float32
            )
            leader = writer
            if const_expr(t.rows % 16):
                # Rows past the CTA's tokens pad the last tile.
                valid[i] = row < t.rows
                leader = leader & valid[i]
            slot = mem.pointer(
                t.quant_offset + ((row // unit) * parts + part) * 4, fx.Float32
            )
            if leader:
                fx.ptr_store(unit_max, slot)
        # A workgroup barrier does not wait for global stores, so the kept
        # output drains while the waves exchange their maxima.
        if const_expr(t.store_output):
            for i in range_constexpr(len(tiles)):
                self._store(valid[i], tiles[i][1], tiles[i][2], output, None)
        # The epilogue's only barrier.
        gpu.barrier()
        units = g // unit * parts
        for i in range_constexpr(len(tiles)):
            row, values, offsets = tiles[i]
            token = row // g
            if const_expr(t.rows % 16):
                token = fx.min(token, fx.Int32(t.rows // g - 1))
            token_max = mem.load(
                t.quant_offset + token * (units * 4), fx.Float32, units
            ).reduce(ReductionOp.MAX)
            leader = (row % g == 0) & (lane < 16) & (part == 0)
            if const_expr(valid[i] is not None):
                leader = leader & valid[i]
            self._finish(
                seq * t.query_length + token,
                leader,
                token_max,
                valid[i],
                values,
                offsets,
                None,
            )

    @flyc.jit
    def store_cluster(self, seq, lane, row_maxima, rows, output):
        """Write whole rows of tokens that the cluster's CTAs merged together.

        ``rows`` lists this wave's (row, valid, value, offset), ``valid``
        None when every row is real. ``row_maxima`` holds one float per row
        that only the row's owner has read; every wave of the cluster must
        call this or ``sync``.
        """
        t, g = self.t, self.t.query_group_size
        for row, valid, value, _ in rows:
            row_max = self._lanes_max(self._lane_max([value]))
            leader = lane == 0
            if const_expr(valid is not None):
                leader = leader & valid
            if leader:
                row_maxima[row] = row_max
        _cluster_sync()
        for row, valid, value, offset in rows:
            first = row // g * g
            token_max = fx.Float32(row_maxima[first + lane % g])
            for j in range_constexpr(1, -(-g // t.WAVE)):
                token_max = fx.maxnumf(
                    token_max, fx.Float32(row_maxima[first + (lane + j * t.WAVE) % g])
                )
            leader = (row % g == 0) & (lane == 0)
            if const_expr(valid is not None):
                leader = leader & valid
            # The cluster release waits for every store, so the kept output
            # follows the barrier.
            self._finish(
                seq * t.query_length + row // g,
                leader,
                self._lanes_max(token_max),
                valid,
                [value],
                [offset],
                output if t.store_output else None,
            )

    def sync(self):
        """The cluster barrier of ``store_cluster`` for waves without a row."""
        _cluster_sync()

    @flyc.jit
    def store_empty(self, token, tokens, scale, stride_row, stride_head):
        """FP8 zeros and descale ``scale`` for ``tokens`` tokens from
        ``token`` on: 0 for an empty sequence, as the separate quant gives,
        and 1 for a padded token past the query's, so a consumer of the whole
        padded height reads inert finite rows."""
        t, scales, token = self.t, self.scales, fx.Int32(token)
        g, tid = t.query_group_size, fx.Int32(gpu.thread_id("x"))
        size = tokens * g * t.head_dim
        inv_scale = fx.Float32(rcp_f32(scale))
        for i in range_constexpr(-(-size // (t.BLOCK_THREADS * 16))):
            # 16 values per thread and store, within one row as 16 | head_dim.
            index = (i * t.BLOCK_THREADS + tid) * 16
            row = index // t.head_dim
            offset = (
                fx.Int64(token + row // g) * stride_row
                + (row % g) * stride_head
                + index % t.head_dim
            )
            values = fx.Vector.filled(16, 0.0, t.q_dtype)
            self._store(
                index < size if size % (t.BLOCK_THREADS * 16) else None,
                [values],
                [offset],
                None,
                inv_scale,
            )
        if tid < tokens:
            scales[token + tid] = scale

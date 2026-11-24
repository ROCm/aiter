# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mega paged attention: decode attention, o_proj, all-reduce and RMSNorm in
one persistent kernel (gfx942/gfx950).

One launch per TP rank and layer, for decode batches of at most ``max_rows``
rows (one query token a row, one KV head a rank). ``grid`` persistent CTAs,
all co-resident, walk the phases in program order, so a CTA only ever waits on
work an earlier phase already handed to a resident CTA (or to a peer rank's
kernel):

1. attention: tasks of S 256-token tiles of a row's paged KV, each run by
   ``pa_decode``'s own tile pipeline and written as a partial (row max, row
   sum, normalized output). S (up to ``max_tiles_per_task``) minimizes the
   rounds of tasks over the grid times a task's cost; a CTA claims tasks
   past its first;
2. merge + per-token FP8: the CTA that lands a row's last partial merges the
   row's partials as ``pa_decode``'s reducer does, rounds the row to bf16 (the
   unfused PA output) and publishes it as FP8 with one scale, as
   ``dynamic_per_token_scaled_quant`` would;
3. o_proj: ``block_n``-column tiles of ``x @ W^T`` once every row is
   published, each bf16 partial tile pushed into every rank's inbox (one-shot)
   or only to the rank owning its columns (``two_shot``), whose sums of them
   come back to every rank: half the wire bytes for one more hop. Small
   batches (``max_rows`` 16) add the peers' partials at the owner's CTA for
   the tile, overlapping the other tiles; larger ones in phase 4, a row per
   CTA, which spreads the hop over more CTAs;
4. all-reduce + residual + Gemma RMSNorm: a row per CTA, gathering the summed
   slices (two-shot) or summing every rank's partials (one-shot). Both round
   the fp32 rank-order sum to bf16 once, so they return the same bits.

The PA tile task runs unchanged: ``_MegaPaContext`` gives it block ids from
this kernel's schedule instead of the hardware grid, and ``_MegaPaLds`` its
share of the kernel's one LDS allocation.

The partials (``pmax``, ``psum``, ``pout``) and the published rows (``x``,
``x_scale``) must be uncached memory (``IpcHeap`` uncached): a retired store is
then visible to every CTA and the reader loads past the caches -- no
agent-scope L2 writeback or invalidate, which a whole XCD serializes on. A
task publishes its partial with ``vmcnt(0)`` and a per-row count.

Cross-CTA hand-offs go through ``sync`` (int32, zero-initialized once), every
counter on a cache line of its own (``SYNC_LINE`` words apart): counters
sharing a line serialize their atomics and polls on it, which cost the merge
chain up to 6 us a launch.

* ``ROW_COUNT`` line ``row``: partials landed a row; the merging CTA zeroes it.
* ``ROWS_DONE`` line ``parity``: rows published this launch. A launch uses the parity
  of its epoch and zeroes the other one for the next launch.
* ``TASK_CLAIM`` line ``parity``: attention tasks past each CTA's first,
  claimed this launch (when there are more tasks than CTAs), the same way: a
  CU whose CTAs run alone takes more of them.
* ``TILE_CLAIM`` line ``parity``: o_proj tiles claimed this launch, the same way. CTAs
  claim tiles as they finish attention, so the last ones to finish (on the
  critical path) are the ones left without a tile, and a claimed tile's W
  loads overlap the wait for the last rows.
* ``EPOCH + block``: launches each CTA has run, so every CTA of every rank
  agrees on the launch's parity without a host-side argument (graph replay).
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from . import buffer_ops
from .comm.one_shot_allreduce import _canonicalize, _has_sentinel
from .comm.quick_allreduce_fusions import (
    residual_add,
    rms_local_sumsq,
    rstd_from_total,
    scale_by_weight,
)
from .comm.transport import (
    _CM_SC0,
    _CM_SC1,
    _ST_NT,
    _global_ptr,
    _poll_v4i32,
    _store_v4i32_peer,
    atom_bf16_to_f32,
    atom_f32_to_bf16,
)
from .communication_ops_utils import memrealtime
from .pa_decode.context import PaDecodeContext
from .pa_decode.op_epilog import PaDecodeEpilogue
from .pa_decode.op_gemm import PaDecodeGemm
from .pa_decode.op_lds import PaDecodeKVLoader, PaDecodeLds, PaDecodeQueryLoader
from .pa_decode.op_softmax import PaDecodeSoftmax
from .pa_decode.pipeline import PaDecodePipeline
from .pa_decode.traits import (
    KV_COMPUTE_BLOCK,
    LOG2E,
    WAVE,
    PaDecodeSchedule,
    PaDecodeTraits,
)
from .tensor_shim import buf_base_i64, ptr_buf_tensor
from .utils import rcp_f32

# The most rows a launch takes; the task planner gives a wave lane
# MAX_ROWS / WAVE of them.
MAX_ROWS = 2 * WAVE
# o_proj runs a tile in passes of up to PASS_ROWS rows: the W slice stays in
# VGPRs across them, the accumulators and the LDS partial tile are a pass's.
PASS_ROWS = WAVE
# ``sync`` layout, in int32 words: a 128 B cache line a counter (a CTA's epoch
# is read and written once a launch, so those share lines).
SYNC_LINE = 32
ROW_COUNT = 0
ROWS_DONE = MAX_ROWS * SYNC_LINE
TILE_CLAIM = ROWS_DONE + 2 * SYNC_LINE
TASK_CLAIM = TILE_CLAIM + 2 * SYNC_LINE
EPOCH = TASK_CLAIM + 2 * SYNC_LINE
# ``trace`` int64 stamps a CTA: start, phases 1-2 done, rows ready, phase 3
# done, phase 4 done; then the ticks its PA tasks took, the ticks landing
# their partials took (merges included) and its task count, all accumulated.
TRACE_STAMPS = 24
# The quantizer's amax floor (``kAmaxFloor``): an all-zero row gets a finite scale.
AMAX_FLOOR = 1e-10
# x, o_proj's FP8 input, is stored as 1 KB chunks of 16 rows x 64 K, each in
# the order the MFMA lanes load it ([K / 16][row][16 K]): a wave's load of
# one is one contiguous 1 KB, where row-major x would be 16 rows of 64 B.
X_CHUNK = 1024
# o_proj partials travel as 16 B atoms of 8 bf16 columns.
ATOM_COLS = 8
# Inbox buffers, alternating by launch parity. A rank re-arms the slots of a row
# as soon as it has read them: a peer writing the same buffer again is two
# launches ahead, which needs this launch's partials from us first.
INBOX_BUFFERS = 2
# Every inbox byte starts as 0xFF: bf16 NaN 0xFFFF, the one-shot all-reduce's
# sentinel (a sender rewrites its own 0xFFFF to 0x7FFF, still NaN).
SENTINEL_FILL_BYTE = 0xFF


def sync_words(grid: int) -> int:
    """int32 words of the ``sync`` workspace for a ``grid``-CTA build."""
    return EPOCH + grid


def inbox_bytes(
    *, world_size: int, max_rows: int, hidden: int, two_shot: bool = False
) -> int:
    """Bytes of one rank's inbox, bf16.

    One-shot: ``[buffer][source rank][row][hidden]``. Two-shot: the partials of
    this rank's column slice, ``[buffer][source rank][row][hidden / world]``,
    then the reduced slices of every rank, ``[buffer][row][hidden]``.
    """
    if two_shot:
        return 2 * INBOX_BUFFERS * max_rows * hidden * 2
    return INBOX_BUFFERS * world_size * max_rows * hidden * 2


class _MegaPaContext(PaDecodeContext):
    """PA tile context whose block ids come from the mega kernel's schedule.

    ``virtual_grid`` is ``(kv_query, slot, kv_queries, slots)``: what the
    hardware grid's ids and sizes would be for this task in ``pa_decode``.
    """

    def __init__(self, *args, virtual_grid, **kwargs):
        super().__init__(*args, **kwargs)
        self.virtual_grid = virtual_grid

    def init_thread_mapping(self):
        self.tid = fx.Int32(gpu.thread_id("x"))
        self.group = None
        self.warp = self.tid // WAVE
        self.lane = self.tid - self.warp * WAVE
        self.seq = self.planned_seq
        kv_query, slot, kv_queries, slots = self.virtual_grid
        self.kv_query = fx.Int32(kv_query)
        self.kv_h = self.kv_query // self.traits.query_splits
        self.query_begin = (
            self.kv_query % self.traits.query_splits
        ) * self.traits.QUERIES_PER_CTA
        self.n_kv = fx.Int32(kv_queries) // self.traits.query_splits
        self.part = fx.Int32(slot)
        self.slots = fx.Int32(slots)
        self.partial_slot = self.kv_h * self.slots + self.part


class _MegaPaEpilogue(PaDecodeEpilogue):
    """``PaDecodeEpilogue`` addressing partial slots by the virtual grid."""

    def init_outputs(self):
        if const_expr(not self.traits.buffer_plan_output):
            super().init_outputs()
            return
        self.pmax = fx.recast_iter(fx.Float32, self.ctx.pmax_ptr)
        self.psum = fx.recast_iter(fx.Float32, self.ctx.psum_ptr)
        self.pout = fx.recast_iter(self.traits.OUT_DTYPE, self.ctx.pout_ptr)
        pout_slot = fx.Int64(self.ctx.kv_h) * fx.Int64(self.ctx.slots) + fx.Int64(
            self.ctx.part
        )
        pout_slot_bytes = self.traits.TOTAL_ROWS * self.traits.head_dim * 2
        self.pout_buffer = ptr_buf_tensor(
            buf_base_i64(self.ctx.pout_ptr) + pout_slot * fx.Int64(pout_slot_bytes),
            self.traits.OUT_DTYPE,
            n=self.traits.TOTAL_ROWS * self.traits.head_dim,
            unit_elems=self.traits.OP_ELEMS,
            unit_stride=1,
            num_records_bytes=pout_slot_bytes,
        )


class _MegaPaLds(PaDecodeLds):
    """``PaDecodeLds`` over LDS the mega kernel allocated.

    A kernel has one ``SharedAllocator``; the mega kernel's holds the PA tile's
    region (field ``buf``) next to its own, and hands the PA its base.
    """

    def __init__(self, traits, storage_type, base):
        super().__init__(traits, storage_type)
        self._mega_base = base

    def allocate(self, group=None):
        assert group is None, "mega paged attention runs single-group PA tiles"
        self.base = self._mega_base
        self.root = self.base


class _MegaPaPipeline(PaDecodePipeline):
    """``PaDecodePipeline`` (its ``__init__`` spelled out) over the mega
    kernel's LDS and partial slots."""

    def __init__(self, ctx, lds_base):
        self.ctx = ctx
        self.traits = ctx.traits
        self.lds = _MegaPaLds(ctx.traits, ctx.shared_storage, lds_base)
        self.gemm = PaDecodeGemm(ctx.traits)
        self.query = PaDecodeQueryLoader(ctx, self.lds, self.gemm)
        self.kv = PaDecodeKVLoader(ctx, self.lds)
        self.softmax = PaDecodeSoftmax(ctx, self.lds, self.kv, self.gemm)
        self.epilogue = _MegaPaEpilogue(ctx, self.lds)


def select_pa_traits(
    *,
    head_dim,
    query_group_size,
    block_size,
    max_rows,
    num_compute_units,
    capacity,
    max_tasks_per_row,
    max_context_length,
    softmax_scale,
    query_dtype,
    kv_dtype,
):
    """The PA tile specialization one mega launch runs (one KV head a rank)."""
    schedule = PaDecodeSchedule.select(
        head_dim=head_dim,
        query_group_size=query_group_size,
        block_size=block_size,
        num_seqs=max_rows,
        num_kv_heads=1,
        num_compute_units=num_compute_units,
        work_capacity=capacity,
        num_partitions=max_tasks_per_row,
        softmax_scale=softmax_scale,
        query_dtype=query_dtype,
        kv_dtype=kv_dtype,
        per_token_kv=True,
        query_length=1,
        trans_v=True,
        max_context_length=max_context_length,
    )
    traits = PaDecodeTraits.create(schedule)
    if traits.DUAL_GROUP or traits.query_splits != 1 or traits.DIRECT:
        raise NotImplementedError(
            "mega paged attention runs the four-wave, unsplit, partial-output PA tile"
        )
    return traits


def _i32_ptr(base, index):
    """int32 global pointer ``base[index]``."""
    return fx.add_offset(fx.recast_iter(fx.Int32, base), index)


def _release_agent():
    fx.memory_fence(syncscope="agent", ordering=fx.AtomicOrdering.Release)


def _load16(addr):
    """16 B at a raw global byte address, as four int32."""
    return fx.Vector(
        fx.generic_load(_global_ptr(addr, T.i32, 16), dtype=fx.Int32, count=4)
    )


def _store16(addr, atom):
    fx.generic_store(_global_ptr(addr, T.i32, 16), atom)


# Buffer loads of uncached hand-off memory go past L1 and L2.
_BYPASS = _CM_SC0 | _CM_SC1


def _rsrc(ptr):
    """Buffer resource over a kernel pointer argument (wave-uniform base)."""
    return buffer_ops.create_buffer_resource_from_addr(buf_base_i64(ptr))


def _uniform(value):
    """A wave-uniform int32 in an SGPR (buffer ``soffset``)."""
    return fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(fx.Int32(value))))


def _as_f32(word):
    """An int32 word's bits as fp32."""
    return fx.Vector.from_elements([fx.Int32(word)], fx.Int32).bitcast(fx.Float32)[0]


def _bload(rsrc, byte_off, words, cache_modifier=0, soffset=None):
    """``words`` int32 at ``byte_off`` (per lane) + ``soffset`` (uniform) bytes."""
    value = buffer_ops.buffer_load(
        rsrc,
        fx.as_ir_value(fx.Int32(byte_off // 4)),
        vec_width=words,
        dtype=T.i32,
        cache_modifier=cache_modifier,
        soffset_bytes=None if soffset is None else fx.as_ir_value(fx.Int32(soffset)),
    )
    return fx.Vector(value) if words > 1 else fx.Int32(value)


def _load_agent(base, index):
    """A spin-wait read of ``base[index]`` (int32) that sees other CTAs' atomics."""
    return fx.Int32(
        fx.generic_load(
            _i32_ptr(base, index),
            dtype=fx.Int32,
            memory_order=fx.AtomicOrdering.Monotonic,
            syncscope="agent",
            volatile=True,
        )
    )


def compile_mega_paged_attention(
    *,
    head_dim: int,
    query_group_size: int,
    block_size: int,
    max_rows: int,
    max_context_length: int,
    max_tiles_per_task: int,
    num_compute_units: int,
    softmax_scale: float,
    query_dtype,
    kv_dtype,
    hidden: int,
    world_size: int = 1,
    rank: int = 0,
    block_n: int = 48,
    prefetch: int | None = None,
    occupancy: int = 1,
    two_shot: bool = True,
    grid: int | None = None,
    trace: bool = False,
):
    """Build the mega paged attention launcher for one rank of ``world_size``.

    ``hidden`` is o_proj's N (the model width), ``block_n`` its tile width and
    ``prefetch`` how many K steps of x a wave keeps in flight (its W slice is
    loaded whole, as soon as the tile is claimed); ``None`` takes every step
    when the registers allow and four otherwise.
    ``occupancy`` is the CTAs a CU holds at once; the default ``grid`` is that
    many per CU, which the persistent schedule needs resident together (the
    build caps registers to fit it).
    ``trace`` records each CTA's phase boundaries (``s_memrealtime``) in
    ``trace[block * TRACE_STAMPS + point]``.
    """
    if max_rows > MAX_ROWS or max_rows % 16:
        raise ValueError(
            f"max_rows must be a multiple of 16 up to {MAX_ROWS}, got {max_rows}"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")
    # One rank has nothing to scatter.
    two_shot = two_shot and world_size > 1
    max_tiles = -(-max_context_length // KV_COMPUTE_BLOCK)
    # Slots for the finest split: one tile a task.
    max_tasks_per_row = max_tiles
    capacity = max_rows * max_tasks_per_row
    grid = occupancy * num_compute_units if grid is None else int(grid)
    if grid > occupancy * num_compute_units:
        raise ValueError(
            f"grid {grid} exceeds the {occupancy * num_compute_units} CTAs that are "
            "resident at once; the persistent schedule would deadlock"
        )
    traits = select_pa_traits(
        head_dim=head_dim,
        query_group_size=query_group_size,
        block_size=block_size,
        max_rows=max_rows,
        num_compute_units=num_compute_units,
        capacity=capacity,
        max_tasks_per_row=max_tasks_per_row,
        max_context_length=max_context_length,
        softmax_scale=softmax_scale,
        query_dtype=query_dtype,
        kv_dtype=kv_dtype,
    )
    threads = traits.CTA_THREADS
    waves = threads // WAVE
    group = query_group_size
    # The attention output a row: o_proj's K.
    row_k = group * head_dim
    # Each thread merges and quantizes ELEMS contiguous dims of one head.
    elems = row_k // threads
    if elems * threads != row_k or head_dim % elems or elems % 4:
        raise NotImplementedError(
            f"a row of {row_k} attention outputs must split into 4k-element "
            f"head pieces over {threads} threads"
        )
    if elems != 8:
        raise NotImplementedError(
            "the merge reads one 16 B bf16 atom of a partial a thread"
        )
    pieces = head_dim // elems
    fp8_max = float(traits.FP8_MAX)
    # o_proj: [rows, row_k] x [hidden, row_k]^T in block_n-column tiles; the
    # waves split K, so no operand is loaded twice, and their partial sums
    # meet in LDS in wave order.
    m_sub = max_rows // 16
    pass_subs = min(m_sub, PASS_ROWS // 16)
    # (first M subtile, M subtiles) of each pass.
    passes = [(m0, min(pass_subs, m_sub - m0)) for m0 in range(0, m_sub, pass_subs)]
    if hidden % block_n or block_n % 16:
        raise ValueError(
            f"block_n {block_n} must be a multiple of 16 dividing {hidden}"
        )
    n_sub = block_n // 16
    n_tiles = hidden // block_n
    if row_k % (waves * 64):
        raise NotImplementedError(
            f"o_proj K {row_k} must split into 64-wide steps a wave"
        )
    k_steps = row_k // (waves * 64)
    if prefetch is None:
        # x, W and the accumulators of a wave, in VGPRs: every x step in flight
        # while that still fits next to the PA tile's budget.
        all_x = 4 * (pass_subs * k_steps + n_sub * k_steps + pass_subs * n_sub)
        prefetch = k_steps if all_x <= 240 else min(4, k_steps)
        if len(passes) > 1:
            # A later pass's x loads would spill the W slice it shares.
            prefetch = min(2, k_steps)
    # A full pass's atoms; a thread's share of them.
    push_iters = -(-(16 * pass_subs * block_n // ATOM_COLS) // threads)
    if hidden % (ATOM_COLS * threads):
        raise NotImplementedError(
            f"hidden {hidden} must split into {ATOM_COLS}-column atoms over {threads} threads"
        )
    row_iters = hidden // (ATOM_COLS * threads)
    # Two-shot: rank r owns columns [r * slice, (r + 1) * slice).
    slice_cols = hidden // world_size
    if two_shot and (n_tiles % world_size or slice_cols % ATOM_COLS):
        raise ValueError(
            f"two-shot needs {n_tiles} o_proj tiles to split evenly over {world_size} ranks"
        )
    tiles_per_rank = n_tiles // world_size
    slice_atoms = slice_cols // ATOM_COLS
    slice_iters = -(-slice_atoms // threads)
    rs_bytes = INBOX_BUFFERS * world_size * max_rows * slice_cols * 2
    # Two-shot's first hop at the owner's tile CTA (else in phase 4, by row).
    rs_in_tile = two_shot and max_rows <= 16
    # An o_proj partial tile in LDS: f32 rows padded to 4 mod 8 floats, so the
    # two quarter-waves of a half-wave (rows 4 apart, 16 banks apart) never
    # share a bank, and every row stays 16 B aligned.
    gemm_stride = block_n + (4 - block_n) % 8
    gemm_region = 16 * pass_subs * gemm_stride * 4
    if waves != 4:
        raise NotImplementedError("the o_proj reduction pairs four waves")
    # The planner's rows: a wave lane holds ``lane_rows`` consecutive ones.
    lane_rows = -(-max_rows // WAVE)
    plan_rows = lane_rows * WAVE

    @fx.struct
    class MegaStorage:
        # The PA tile's region (compile_pa_decode_tile's layout, one group);
        # phase 3 reuses it for the two o_proj partial tiles.
        buf: fx.Array[fx.Int32, max(traits.total_bytes, 2 * gemm_region) // 4, 16]
        # Inclusive prefix of tasks a row, then each row's context length.
        prefix: fx.Array[fx.Int32, plan_rows, 16]
        context: fx.Array[fx.Int32, plan_rows, 16]
        # W's scales of the o_proj tile in hand, loaded with W (in VGPRs they
        # would live through every pass).
        w_scale: fx.Array[fx.Float32, block_n, 16]
        # A wave's partial reduction; a CTA-wide broadcast word.
        red: fx.Array[fx.Float32, 16, 16]
        word: fx.Array[fx.Int32, 4, 16]

    def _lds(base, index, elem=fx.Int32):
        ptr_ty = fx.PointerType.get(elem.ir_type, fx.AddressSpace.Shared, 4)
        return fx.recast_iter(ptr_ty, fx.add_offset(base, fx.make_int_tuple(index * 4)))

    def _lds_shift(base, nbytes):
        return fx.add_offset(base, fx.make_int_tuple(nbytes))

    def _lds_i32(base, index):
        return fx.Int32(fx.ptr_load(_lds(base, index)))

    def _load_lengths(context_ptr, num_rows, tid):
        """The context lengths (0 past ``num_rows``) of lane ``tid % WAVE``'s
        planner rows, issued first so their latency overlaps the epoch's."""
        last = num_rows - 1
        lengths = []
        for i in range_constexpr(lane_rows):
            row = (tid % WAVE) * lane_rows + i
            safe = (row < last).select(row, last)
            length = fx.Int32(fx.ptr_load(_i32_ptr(context_ptr, safe)))
            lengths.append((row < num_rows).select(length, fx.Int32(0)))
        return lengths

    @flyc.jit
    def _plan_tasks(lengths, num_rows, nblk, prefix_base, context_base, word_base, tid):
        """Wave 0: each row's task count, its inclusive prefix and its length."""
        if tid < WAVE:
            lane = tid
            tiles = [(n + (KV_COMPUTE_BLOCK - 1)) // KV_COMPUTE_BLOCK for n in lengths]
            # Tiles a task: the span whose rounds of tasks over the grid cost
            # least, a task costing its tiles plus half a tile (its query load,
            # partial and merge count); ties go to the longer span.
            span = fx.Int32(1)
            best = fx.Int32(0)
            for cand in range_constexpr(1, max_tiles_per_task + 1):
                count = (tiles[0] + (cand - 1)) // cand
                for t in tiles[1:]:
                    count = count + (t + (cand - 1)) // cand
                for off in range_constexpr(5, -1, -1):
                    count = count + count.shuffle_xor(fx.Int32(1 << off), WAVE)
                cost = (count + (nblk - 1)) // nblk * (2 * cand + 1)
                if const_expr(cand == 1):
                    best = cost
                else:
                    take = cost <= best
                    span = take.select(fx.Int32(cand), span)
                    best = take.select(cost, best)
            tasks = [(t + (span - 1)) // span for t in tiles]
            mine = tasks[0]
            for t in tasks[1:]:
                mine = mine + t
            prefix = mine
            for off in range_constexpr(6):
                step = 1 << off
                other = fx.gpu.shuffle_idx(prefix, (lane - step) & (WAVE - 1), WAVE)
                prefix = prefix + (lane >= step).select(other, fx.Int32(0))
            # The lane's rows in order, from the tasks of the lanes before it.
            prefix = prefix - mine
            for i in range_constexpr(lane_rows):
                prefix = prefix + tasks[i]
                fx.ptr_store(prefix, _lds(prefix_base, lane * lane_rows + i))
                fx.ptr_store(lengths[i], _lds(context_base, lane * lane_rows + i))
            if lane == 0:
                fx.ptr_store(span, _lds(word_base, 2))

    def _row_tasks(prefix_base, row):
        """(first task, task count) of ``row``."""
        last = _lds_i32(prefix_base, row)
        has_prev = row > 0
        prev = _lds_i32(prefix_base, has_prev.select(row - 1, fx.Int32(0)))
        first = has_prev.select(prev, fx.Int32(0))
        return first, last - first

    def _find_row(prefix_base, task):
        """The row whose task range holds ``task``: prefix entries <= task."""
        row = fx.Int32(0)
        for bit in range_constexpr(plan_rows.bit_length() - 2, -1, -1):
            probe = row + (1 << bit)
            row = (_lds_i32(prefix_base, probe - 1) <= task).select(probe, row)
        return row

    @flyc.jit
    def _pa_task(
        pmax_ptr,
        psum_ptr,
        pout_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        key_scale_ptr,
        value_scale_ptr,
        max_blocks_per_seq,
        stride_ks_block,
        stride_ks_head,
        stride_q_row,
        stride_q_head,
        num_rows,
        seq,
        start,
        end,
        context,
        slot,
        lds_base,
    ):
        zero = fx.Int32(0)
        ctx = _MegaPaContext(
            traits,
            MegaStorage,
            pmax_ptr,
            psum_ptr,
            pout_ptr,
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            block_tables_ptr,
            key_scale_ptr,
            value_scale_ptr,
            max_blocks_per_seq,
            stride_ks_block,
            stride_ks_head,
            stride_q_row,
            stride_q_head,
            num_rows,
            seq,
            start,
            end,
            context,
            stride_k_block=zero,
            stride_k_token=zero,
            stride_k_head=zero,
            stride_v_block=zero,
            stride_v_token=zero,
            stride_v_head=zero,
            virtual_grid=(zero, slot, 1, capacity),
        )
        _MegaPaPipeline(ctx, lds_base).run()

    @flyc.jit
    def _block_max(value, red_base, tid):
        """max of ``value`` over the CTA (every thread gets it)."""
        for off in range_constexpr(5, -1, -1):
            value = fx.maxnumf(value, value.shuffle_xor(fx.Int32(1 << off), WAVE))
        if tid % WAVE == 0:
            fx.ptr_store(value, _lds(red_base, tid // WAVE, fx.Float32))
        gpu.barrier()
        result = fx.Float32(fx.ptr_load(_lds(red_base, 0, fx.Float32)))
        for w in range_constexpr(1, waves):
            result = fx.maxnumf(
                result, fx.Float32(fx.ptr_load(_lds(red_base, w, fx.Float32)))
            )
        return result

    def _x_offset(row, k):
        """Byte of x (row, k): o_proj's A operand order, see ``X_CHUNK``."""
        chunk = (row // 16) * (row_k // 64) + k // 64
        kk = k % 64
        return chunk * X_CHUNK + (kk // 16) * 256 + (row % 16) * 16 + kk % 16

    def _store_x(x_ptr, row, tid, words):
        """This thread's ``elems`` FP8 bytes of row ``row`` of x."""
        fx.ptr_store(words, _i32_ptr(x_ptr, _x_offset(row, tid * elems) // 4))

    @flyc.jit
    def _merge_row(
        row,
        first,
        count,
        pmax_ptr,
        psum_ptr,
        pout_ptr,
        x_ptr,
        x_scale_ptr,
        red_base,
        tid,
    ):
        """Phase 2 for one row: merge its partials, round, quantize, publish."""
        head = tid // pieces
        piece = tid - head * pieces
        pmax_rsrc, psum_rsrc, pout_rsrc = (
            _rsrc(pmax_ptr),
            _rsrc(psum_ptr),
            _rsrc(pout_ptr),
        )
        neg_inf = fx.Float32(float("-inf"))
        zero = fx.Float32(0.0)
        lives, maxes, sums, parts = [], [], [], []
        # Every load of the row's partials issues before any is used.
        for p in range_constexpr(max_tasks_per_row):
            live = fx.Int32(p) < count
            slot = first + live.select(fx.Int32(p), fx.Int32(0))
            stat = (slot * group + head) * 4
            lives.append(live)
            maxes.append(_bload(pmax_rsrc, stat, 1, _BYPASS))
            sums.append(_bload(psum_rsrc, stat, 1, _BYPASS))
            parts.append(
                _bload(
                    pout_rsrc,
                    ((slot * group + head) * head_dim + piece * elems) * 2,
                    4,
                    _BYPASS,
                )
            )
        maxes = [live.select(_as_f32(m), neg_inf) for live, m in zip(lives, maxes)]
        sums = [live.select(_as_f32(v), zero) for live, v in zip(lives, sums)]
        row_max = maxes[0]
        for m in maxes[1:]:
            row_max = row_max.maximumf(m)
        safe_max = (row_max > neg_inf).select(row_max, zero)
        weights = []
        total = zero
        for m, s in zip(maxes, sums):
            w = (m > neg_inf).select(
                s * fx.exp2((m - safe_max) * fx.Float32(LOG2E), fastmath="fast"), zero
            )
            weights.append(w)
            total = total + w
        inv_total = fx.Float32(rcp_f32((total > zero).select(total, fx.Float32(1.0))))
        acc = fx.full(elems, 0.0, fx.Float32)
        for part, w in zip(parts, weights):
            acc = acc + atom_bf16_to_f32(part) * fx.full(
                elems, w * inv_total, fx.Float32
            )
        # The unfused PA hands o_proj's quantizer a bf16 row.
        out = acc.to(fx.BFloat16).to(fx.Float32)
        amax = _block_max(fmath.absf(out).reduce(ReductionOp.MAX), red_base, tid)
        scale = fx.maxnumf(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / fp8_max)
        inv_scale = fx.full(elems, fx.Float32(rcp_f32(scale)), fx.Float32)
        scaled = out * inv_scale
        words = []
        for i in range_constexpr(elems // 4):
            lo = rocdl.cvt_pk_fp8_f32(T.i32, scaled[4 * i], scaled[4 * i + 1], 0, False)
            words.append(
                rocdl.cvt_pk_fp8_f32(
                    T.i32, scaled[4 * i + 2], scaled[4 * i + 3], lo, True
                )
            )
        _store_x(x_ptr, row, tid, fx.Vector.from_elements(words, dtype=fx.Int32))
        if tid == 0:
            fx.ptr_store(
                scale, fx.add_offset(fx.recast_iter(fx.Float32, x_scale_ptr), row)
            )

    @flyc.jit
    def _publish_row(sync_ptr, rows_done, tid):
        """Count one published row once every thread's x stores have landed."""
        rocdl.s_waitcnt(vmcnt=0)
        fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Release)
        gpu.barrier()
        if tid == 0:
            fx.atomic_add(_i32_ptr(sync_ptr, rows_done), fx.Int32(1), syncscope="agent")

    @flyc.jit
    def _land_partial(
        row,
        first,
        count,
        sync_ptr,
        rows_done,
        pmax_ptr,
        psum_ptr,
        pout_ptr,
        x_ptr,
        x_scale_ptr,
        red_base,
        word_base,
        tid,
        trace_ptr=None,
        bid=None,
    ):
        """Count this task's partial; the CTA landing a row's last one merges it.

        The partials are uncached: retired stores are visible device-wide, so
        the workgroup fences only keep the compiler from moving the stores
        below the count, or the merge's loads above it.
        """
        t0 = _now()
        rocdl.s_waitcnt(vmcnt=0)
        fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Release)
        gpu.barrier()
        t1 = _now()
        if tid == 0:
            landed = fx.atomic_add(
                _i32_ptr(sync_ptr, ROW_COUNT + row * SYNC_LINE),
                fx.Int32(1),
                syncscope="agent",
            )
            fx.ptr_store(fx.Int32(landed), _lds(word_base, 0))
        gpu.barrier()
        fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Acquire)
        t2 = _now()
        if _lds_i32(word_base, 0) == count - 1:
            _merge_row(
                row,
                first,
                count,
                pmax_ptr,
                psum_ptr,
                pout_ptr,
                x_ptr,
                x_scale_ptr,
                red_base,
                tid,
            )
            t3 = _now()
            if tid == 0:
                # No other task of this row is left in this launch.
                fx.ptr_store(
                    fx.Int32(0), _i32_ptr(sync_ptr, ROW_COUNT + row * SYNC_LINE)
                )
            _publish_row(sync_ptr, rows_done, tid)
            t4 = _now()
            if const_expr(trace):
                _stamp_value(trace_ptr, bid, 16, t1 - t0, tid)
                _stamp_value(trace_ptr, bid, 17, t2 - t1, tid)
                _stamp_value(trace_ptr, bid, 18, t3 - t2, tid)
                _stamp_value(trace_ptr, bid, 19, t4 - t3, tid)

    @flyc.jit
    def _publish_empty_rows(
        num_rows, bid, nblk, prefix_base, sync_ptr, rows_done, x_ptr, x_scale_ptr, tid
    ):
        """Rows without context (graph padding) publish a zero row."""
        for row in range(bid, num_rows, nblk):
            _first, count = _row_tasks(prefix_base, row)
            if count == 0:
                _store_x(x_ptr, row, tid, fx.full(elems // 4, 0, fx.Int32))
                if tid == 0:
                    fx.ptr_store(
                        fx.Float32(AMAX_FLOOR / fp8_max),
                        fx.add_offset(fx.recast_iter(fx.Float32, x_scale_ptr), row),
                    )
                _publish_row(sync_ptr, rows_done, tid)

    @flyc.jit
    def _stamp(trace_ptr, bid, point, tid):
        if const_expr(trace):  # noqa: SIM102
            if tid == 0:
                fx.ptr_store(
                    memrealtime(),
                    fx.add_offset(
                        fx.recast_iter(fx.Int64, trace_ptr), bid * TRACE_STAMPS + point
                    ),
                )

    def _now():
        return memrealtime() if const_expr(trace) else fx.Int64(0)

    @flyc.jit
    def _stamp_value(trace_ptr, bid, point, value, tid):
        if const_expr(trace):  # noqa: SIM102
            if tid == 0:
                fx.ptr_store(
                    value,
                    fx.add_offset(
                        fx.recast_iter(fx.Int64, trace_ptr), bid * TRACE_STAMPS + point
                    ),
                )

    @flyc.jit
    def _accumulate(trace_ptr, bid, point, ticks, tid):
        if const_expr(trace):  # noqa: SIM102
            if tid == 0:
                ptr = fx.add_offset(
                    fx.recast_iter(fx.Int64, trace_ptr), bid * TRACE_STAMPS + point
                )
                fx.ptr_store(fx.Int64(fx.ptr_load(ptr)) + ticks, ptr)

    @flyc.jit
    def _wait_rows(sync_ptr, rows_done, num_rows, tid):
        """Phase 3 entry: every row of this launch is published as x."""
        if tid == 0:
            done = _load_agent(sync_ptr, rows_done)
            while done < num_rows:
                rocdl.s_sleep(1)
                done = _load_agent(sync_ptr, rows_done)
        gpu.barrier()
        # x is uncached and read past the caches: order the loads only.
        fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Acquire)

    def _k_step(tile, s):
        """The K step a wave takes ``s``-th in ``tile``: tiles start at
        different steps, so the CTAs loading x at once spread over it."""
        return _uniform((tile + s) % k_steps)

    def _load_w_slice(tile, w_rsrc, lane, warp):
        """This wave's K slice of W for tile ``tile``: 16 B a lane per (step, N-subtile).

        W is ``shuffle_weight(layout=(16, 16))``: per 16 rows and 32 K, two
        [16 rows][16 K] halves (512 B). Lane (n, g) loads K [16g, 16g + 16) of
        a 64-K step, for A (x) and B (W) alike, so the two MFMAs of the step
        pair the same K on both operands. The per-lane offset is shared by
        every load; the (tile, subtile, step) part is a uniform ``soffset``.
        """
        n1 = lane % 16
        g = lane // 16
        lane_off = (g // 2) * 512 + (g % 2) * 256 + n1 * 16
        base = _uniform(tile * (n_sub * (row_k // 32) * 512) + warp * (k_steps * 1024))
        ws = []
        for s in range_constexpr(k_steps):
            step = _k_step(tile, s)
            ws.append(
                [
                    _bload(
                        w_rsrc,
                        lane_off,
                        4,
                        soffset=base + (j * (row_k // 32) * 512 + step * 1024),
                    )
                    for j in range_constexpr(n_sub)
                ]
            )
        # Keep every W load ahead of the wait for rows.
        rocdl.sched_barrier(0)
        return ws

    def _gemm_tile(ws, x_rsrc, lane, warp, tile, m0, m_count):
        """This wave's K slice of M subtiles ``[m0, m0 + m_count)`` of an o_proj
        tile: f32x4 per (M, N) subtile."""
        # One 1 KB chunk of x a load: lane (n1, g) reads row n1, K [16g, 16g + 16).
        lane_off = lane * 16
        base = _uniform(warp * (k_steps * X_CHUNK))

        def issue_x(s):
            return [
                _bload(
                    x_rsrc,
                    lane_off,
                    4,
                    _BYPASS,
                    soffset=base
                    + ((m0 + m) * (row_k // 64) + _k_step(tile, s)) * X_CHUNK,
                )
                for m in range_constexpr(m_count)
            ]

        acc = [
            [fx.full(4, 0.0, fx.Float32) for _j in range(n_sub)]
            for _m in range(m_count)
        ]
        # x issues in consumption order: in-order vmcnt waits keep ``prefetch``
        # steps in flight.
        pending = [issue_x(s) for s in range(min(prefetch, k_steps))]
        for s in range_constexpr(k_steps):
            if s + prefetch < k_steps:
                pending.append(issue_x(s + prefetch))
                rocdl.sched_barrier(0)
            xs = pending[s]
            for m in range_constexpr(m_count):
                a = xs[m].bitcast(fx.Int64)
                for j in range_constexpr(n_sub):
                    b = ws[s][j].bitcast(fx.Int64)
                    v = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [a[0], b[0], acc[m][j], 0, 0, 0]
                    )
                    v = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [a[1], b[1], v, 0, 0, 0]
                    )
                    acc[m][j] = fx.Vector(v)
        return [v for row in acc for v in row]

    @flyc.jit
    def _claim_tile(sync_ptr, claim, word_base, tid):
        """The next unclaimed o_proj tile (``n_tiles`` and up: none left)."""
        # The broadcast word is free once every thread has read the last claim.
        gpu.barrier()
        if tid == 0:
            tile = fx.atomic_add(
                _i32_ptr(sync_ptr, claim), fx.Int32(1), syncscope="agent"
            )
            fx.ptr_store(fx.Int32(tile), _lds(word_base, 1))
        gpu.barrier()
        return _lds_i32(word_base, 1)

    @flyc.jit
    def _next_task(sync_ptr, claim, word_base, tid, total, nblk):
        """The next attention task: one claimed past the first round, if any."""
        task = total
        if total > nblk:
            task = nblk + _claim_tile(sync_ptr, claim, word_base, tid)
        else:
            gpu.barrier()
        return task

    def _stage_partial(acc, lane, region, accumulate):
        """Store (or add into) one of the two LDS partial tiles."""
        for m in range_constexpr(len(acc) // n_sub):
            for j in range_constexpr(n_sub):
                for i in range_constexpr(4):
                    row = 16 * m + 4 * (lane // 16) + i
                    col = 16 * j + lane % 16
                    ptr = _lds(region, row * gemm_stride + col, fx.Float32)
                    v = acc[m * n_sub + j][i]
                    if const_expr(accumulate):
                        v = fx.Float32(fx.ptr_load(ptr)) + v
                    fx.ptr_store(v, ptr)

    @flyc.jit
    def _reduce_waves(acc, warp, lane, gemm_base):
        """Pair the waves' K slices in LDS: (w0 + w1) and (w2 + w3), fixed order.

        Waves 1 and 3 store, waves 0 and 2 add theirs; the push adds the two.
        """
        region = gemm_base if warp < 2 else _lds_shift(gemm_base, gemm_region)
        if (warp == 1) | (warp == 3):
            _stage_partial(acc, lane, region, False)
        gpu.barrier()
        if (warp == 0) | (warp == 2):
            _stage_partial(acc, lane, region, True)
        gpu.barrier()

    def _inbox_off(buf, src, row, col):
        return fx.Int64(
            (((buf * world_size + src) * max_rows + row) * hidden + col) * 2
        )

    def _rs_off(buf, src, row, col_local):
        """Two-shot: a source rank's partial of this rank's column slice."""
        return fx.Int64(
            (((buf * world_size + src) * max_rows + row) * slice_cols + col_local) * 2
        )

    def _ag_off(buf, row, col):
        """Two-shot: a row's reduced sums, every rank's slice."""
        return fx.Int64(rs_bytes) + fx.Int64(
            ((buf * max_rows + row) * hidden + col) * 2
        )

    def _peer(peers, rank_id):
        """``peers[rank_id]`` for a runtime rank."""
        addr = peers[0]
        for d in range_constexpr(1, world_size):
            addr = (rank_id == d).select(peers[d], addr)
        return addr

    def _pack(vecs):
        return fx.Vector.from_elements(
            [v[e] for v in vecs for e in range_constexpr(4)], fx.Int32
        )

    def _unpack(got, i):
        return fx.Vector.from_elements([got[4 * i + e] for e in range(4)], fx.Int32)

    def _push_atom(c, tid):
        """(atom, row, column atom) of this thread's ``c``-th atom of a tile."""
        atom = tid + c * threads
        row = atom // (block_n // ATOM_COLS)
        return atom, row, atom - row * (block_n // ATOM_COLS)

    def _load_col_scales(tile, tid, w_scale_ptr):
        """Four of W's per-column scales of ``tile`` a thread (the first
        ``block_n / 4`` threads' are the tile's), issued ahead of W so that
        staging them waits for them alone."""
        quad = tid % (block_n // 4)
        return fx.Vector(
            fx.ptr_load(
                fx.add_offset(
                    fx.recast_iter(fx.Float32, w_scale_ptr), tile * block_n + 4 * quad
                ),
                result_type=fx.Vector.make_type(4, fx.Float32),
            )
        )

    @flyc.jit
    def _stage_col_scales(scales, tid, w_scale_base):
        """The tile's W scales into LDS; the push reads them after the wait
        for rows' barrier."""
        if tid < block_n // 4:
            fx.ptr_store(scales, _lds(w_scale_base, 4 * tid, fx.Float32))

    def _col_scales(col, w_scale_base):
        """W's scales of column atom ``col`` of the tile in hand."""
        return fx.Vector(
            fx.ptr_load(
                _lds(w_scale_base, col * ATOM_COLS, fx.Float32),
                result_type=fx.Vector.make_type(ATOM_COLS, fx.Float32),
            )
        )

    def _row_scales(tid, x_scale_ptr, row0):
        """x's per-row scales of this thread's atoms of the pass at ``row0``
        (loaded with x)."""
        x_scale_rsrc = _rsrc(x_scale_ptr)
        scales = []
        for c in range_constexpr(push_iters):
            _atom, row, _col = _push_atom(c, tid)
            row = row + row0
            row = (row < max_rows).select(row, fx.Int32(max_rows - 1))
            scales.append(_as_f32(_bload(x_scale_rsrc, row * 4, 1, _BYPASS)))
        return scales

    @flyc.jit
    def _push_tile(
        tile,
        num_rows,
        tid,
        gemm_base,
        x_scales,
        w_scale_base,
        peers,
        buf,
        row0,
        m_count,
    ):
        """Scale the pass's tile at ``row0`` to bf16 and push its live rows into
        every rank's inbox."""
        tile_atoms = 16 * m_count * block_n // ATOM_COLS
        for c in range_constexpr(-(-tile_atoms // threads)):
            atom, local, col = _push_atom(c, tid)
            row = local + row0
            if (atom < tile_atoms) & (row < num_rows):
                pair = [
                    fx.Vector(
                        fx.ptr_load(
                            _lds(
                                gemm_base,
                                r * (gemm_region // 4)
                                + local * gemm_stride
                                + col * ATOM_COLS,
                                fx.Float32,
                            ),
                            result_type=fx.Vector.make_type(ATOM_COLS, fx.Float32),
                        )
                    )
                    for r in range_constexpr(2)
                ]
                acc = pair[0] + pair[1]
                n = tile * block_n + col * ATOM_COLS
                y = (
                    acc
                    * fx.full(ATOM_COLS, x_scales[c], fx.Float32)
                    * _col_scales(col, w_scale_base)
                )
                payload = _canonicalize(atom_f32_to_bf16(y))
                if const_expr(two_shot):
                    owner = tile // tiles_per_rank
                    col_local = n - owner * slice_cols
                    if const_expr(rs_in_tile):
                        if owner == rank:
                            _reduce_atom(payload, peers, buf, row, n, col_local)
                        else:
                            _store_v4i32_peer(
                                _peer(peers, owner)
                                + _rs_off(buf, rank, row, col_local),
                                payload,
                                _ST_NT,
                            )
                    else:
                        _store_v4i32_peer(
                            _peer(peers, owner) + _rs_off(buf, rank, row, col_local),
                            payload,
                            _ST_NT,
                        )
                else:
                    for dst in range_constexpr(world_size):
                        _store_v4i32_peer(
                            peers[dst] + _inbox_off(buf, rank, row, n), payload, _ST_NT
                        )
        # The next tile reuses the LDS tile.
        gpu.barrier()

    @flyc.jit
    def _block_sum(value, red_base, tid):
        """Sum of ``value`` over the CTA in a fixed order (every thread gets it)."""
        for off in range_constexpr(5, -1, -1):
            value = value + value.shuffle_xor(fx.Int32(1 << off), WAVE)
        if tid % WAVE == 0:
            fx.ptr_store(value, _lds(red_base, tid // WAVE, fx.Float32))
        gpu.barrier()
        result = fx.Float32(fx.ptr_load(_lds(red_base, 0, fx.Float32)))
        for w in range_constexpr(1, waves):
            result = result + fx.Float32(fx.ptr_load(_lds(red_base, w, fx.Float32)))
        return result

    @flyc.jit
    def _reduce_row(
        row,
        buf,
        inbox,
        residual_ptr,
        residual_out_ptr,
        out_ptr,
        norm_w_ptr,
        eps,
        red_base,
        tid,
    ):
        """Phase 4 for one row: rank-order sum, residual add, Gemma RMSNorm."""
        cols = [(tid + c * threads) * ATOM_COLS for c in range_constexpr(row_iters)]
        residual = buf_base_i64(residual_ptr) + fx.Int64(row * hidden * 2)
        residual_out = buf_base_i64(residual_out_ptr) + fx.Int64(row * hidden * 2)
        out = buf_base_i64(out_ptr) + fx.Int64(row * hidden * 2)
        norm_w = buf_base_i64(norm_w_ptr)
        res_atoms = [_load16(residual + fx.Int64(col * 2)) for col in cols]
        w_atoms = [_load16(norm_w + fx.Int64(col * 2)) for col in cols]
        if const_expr(two_shot):
            addrs = [
                inbox + _ag_off(buf, row, cols[c]) for c in range_constexpr(row_iters)
            ]
        else:
            addrs = [
                inbox + _inbox_off(buf, src, row, cols[c])
                for src in range_constexpr(world_size)
                for c in range_constexpr(row_iters)
            ]
        got = _pack([_poll_v4i32(addr) for addr in addrs])
        while _has_sentinel(got):
            got = _pack([_poll_v4i32(addr) for addr in addrs])
        ar = []
        for c in range_constexpr(row_iters):
            if const_expr(two_shot):
                # Already the rank-order sum, rounded once to bf16.
                ar.append(atom_bf16_to_f32(_unpack(got, c)))
            else:
                acc = None
                for src in range_constexpr(world_size):
                    v = atom_bf16_to_f32(_unpack(got, src * row_iters + c))
                    acc = v if acc is None else acc + v
                ar.append(acc)
        vals = residual_add(ar, res_atoms)
        for c in range_constexpr(row_iters):
            _store16(residual_out + fx.Int64(cols[c] * 2), atom_f32_to_bf16(vals[c]))
        total = _block_sum(rms_local_sumsq([vals])[0], red_base, tid)
        rstd = rstd_from_total(total, eps, hidden)
        for c, atom in enumerate(scale_by_weight(vals, rstd, w_atoms, w_bias=1.0)):
            _store16(out + fx.Int64(cols[c] * 2), atom)
        sentinel = fx.Vector.filled((4,), -1, fx.Int32)
        for addr in addrs:
            _store_v4i32_peer(addr, sentinel, _ST_NT)

    @flyc.jit
    def _reduce_atom(payload, peers, buf, row, col, col_local):
        """Two-shot, first hop, at the tile of a column this rank owns: our own
        partial plus the peers' (rank order, one bf16 rounding), pushed to every
        rank. Our own partial never leaves the CTA."""
        inbox = peers[rank]
        addrs = [
            inbox + _rs_off(buf, src, row, col_local)
            for src in range_constexpr(world_size)
            if src != rank
        ]
        got = _pack([_poll_v4i32(addr) for addr in addrs])
        while _has_sentinel(got):
            got = _pack([_poll_v4i32(addr) for addr in addrs])
        acc = None
        peer = 0
        for src in range_constexpr(world_size):
            if const_expr(src == rank):
                v = atom_bf16_to_f32(payload)
            else:
                v = atom_bf16_to_f32(_unpack(got, peer))
                peer += 1
            acc = v if acc is None else acc + v
        summed = _canonicalize(atom_f32_to_bf16(acc))
        for dst in range_constexpr(world_size):
            _store_v4i32_peer(peers[dst] + _ag_off(buf, row, col), summed, _ST_NT)
        sentinel = fx.Vector.filled((4,), -1, fx.Int32)
        for addr in addrs:
            _store_v4i32_peer(addr, sentinel, _ST_NT)

    @flyc.jit
    def _reduce_slice(row, buf, peers, tid):
        """Two-shot, first hop, by row: sum this rank's column slice of ``row``
        over the ranks (rank order, one bf16 rounding), push it to every rank."""
        inbox = peers[rank]
        sentinel = fx.Vector.filled((4,), -1, fx.Int32)
        for c in range_constexpr(slice_iters):
            atom = tid + c * threads
            if atom < slice_atoms:
                col = atom * ATOM_COLS
                addrs = [
                    inbox + _rs_off(buf, src, row, col)
                    for src in range_constexpr(world_size)
                ]
                got = _pack([_poll_v4i32(addr) for addr in addrs])
                while _has_sentinel(got):
                    got = _pack([_poll_v4i32(addr) for addr in addrs])
                acc = atom_bf16_to_f32(_unpack(got, 0))
                for src in range_constexpr(1, world_size):
                    acc = acc + atom_bf16_to_f32(_unpack(got, src))
                payload = _canonicalize(atom_f32_to_bf16(acc))
                for dst in range_constexpr(world_size):
                    _store_v4i32_peer(
                        peers[dst] + _ag_off(buf, row, rank * slice_cols + col),
                        payload,
                        _ST_NT,
                    )
                for addr in addrs:
                    _store_v4i32_peer(addr, sentinel, _ST_NT)

    @flyc.jit
    def _tile_pass(
        tile,
        m0,
        m_count,
        ws,
        w_scale_base,
        x_rsrc,
        x_scale_ptr,
        num_rows,
        gemm_base,
        peers,
        buf,
        lane,
        warp,
        tid,
        trace_ptr,
        bid,
    ):
        """One pass of an o_proj tile: rows ``[16 m0, 16 (m0 + m_count))``."""
        row0 = 16 * m0
        t_g = _now()
        x_scales = _row_scales(tid, x_scale_ptr, row0)
        acc = _gemm_tile(ws, x_rsrc, lane, warp, tile, m0, m_count)
        if const_expr(trace):
            rocdl.s_waitcnt(vmcnt=0)
        t_r = _now()
        _reduce_waves(acc, warp, lane, gemm_base)
        t_p = _now()
        _push_tile(
            tile,
            num_rows,
            tid,
            gemm_base,
            x_scales,
            w_scale_base,
            peers,
            buf,
            row0,
            m_count,
        )
        t_c = _now()
        # The gemm (loads waited for), the wave reduction and the push.
        _accumulate(trace_ptr, bid, 9, t_r - t_g, tid)
        _accumulate(trace_ptr, bid, 10, t_p - t_r, tid)
        _accumulate(trace_ptr, bid, 11, t_c - t_p, tid)

    kernel_name = (
        f"mega_paged_attention_d{head_dim}_g{group}_b{block_size}"
        f"_r{max_rows}_t{max_tiles_per_task}_n{hidden}_bn{block_n}"
        f"_w{world_size}r{rank}_pf{prefetch}_o{occupancy}"
        + ("_2s" if two_shot else "")
        + ("_trace" if trace else "")
    )

    @flyc.kernel(name=kernel_name, known_block_size=(threads, 1, 1))
    def mega_paged_attention_kernel(
        query_ptr: fx.Pointer,
        key_cache_ptr: fx.Pointer,
        value_cache_ptr: fx.Pointer,
        key_scale_ptr: fx.Pointer,
        value_scale_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        context_ptr: fx.Pointer,
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        x_ptr: fx.Pointer,
        x_scale_ptr: fx.Pointer,
        sync_ptr: fx.Pointer,
        w_ptr: fx.Pointer,
        w_scale_ptr: fx.Pointer,
        residual_ptr: fx.Pointer,
        residual_out_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        norm_w_ptr: fx.Pointer,
        peers_ptr: fx.Pointer,
        trace_ptr: fx.Pointer,
        num_rows: fx.Int32,
        max_blocks_per_seq: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        eps: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        nblk = fx.Int32(gpu.grid_dim.x)
        mega = fx.SharedAllocator().allocate(MegaStorage).peek()
        pa_base = fx.recast_iter(fx.Uint8, mega.buf.ptr)
        prefix_base = fx.recast_iter(fx.Uint8, mega.prefix.ptr)
        context_base = fx.recast_iter(fx.Uint8, mega.context.ptr)
        red_base = fx.recast_iter(fx.Uint8, mega.red.ptr)
        w_scale_base = fx.recast_iter(fx.Uint8, mega.w_scale.ptr)
        word_base = fx.recast_iter(fx.Uint8, mega.word.ptr)
        gemm_base = pa_base

        _stamp(trace_ptr, bid, 0, tid)
        lengths = _load_lengths(context_ptr, num_rows, tid)
        epoch = fx.Int32(fx.ptr_load(_i32_ptr(sync_ptr, EPOCH + bid)))
        parity = epoch & 1
        rows_done = ROWS_DONE + parity * SYNC_LINE

        _plan_tasks(lengths, num_rows, nblk, prefix_base, context_base, word_base, tid)
        if (bid == 0) & (tid == 0):
            # Nothing reads the other parity this launch; the next one counts on it.
            other = (parity ^ 1) * SYNC_LINE
            fx.ptr_store(fx.Int32(0), _i32_ptr(sync_ptr, ROWS_DONE + other))
            fx.ptr_store(fx.Int32(0), _i32_ptr(sync_ptr, TILE_CLAIM + other))
            fx.ptr_store(fx.Int32(0), _i32_ptr(sync_ptr, TASK_CLAIM + other))
        gpu.barrier()
        total = _lds_i32(prefix_base, plan_rows - 1)
        span = _lds_i32(word_base, 2)
        _stamp(trace_ptr, bid, 14, tid)
        _publish_empty_rows(
            num_rows,
            bid,
            nblk,
            prefix_base,
            sync_ptr,
            rows_done,
            x_ptr,
            x_scale_ptr,
            tid,
        )

        # Phases 1-2: attention tasks, the first a CTA, the rest claimed; the
        # CTA landing a row's last partial merges and publishes the row.
        task = bid
        while task < total:
            row = _find_row(prefix_base, task)
            first, count = _row_tasks(prefix_base, row)
            length = _lds_i32(context_base, row)
            tiles = (length + (KV_COMPUTE_BLOCK - 1)) // KV_COMPUTE_BLOCK
            start = (task - first) * span
            end = start + span
            end = (end < tiles).select(end, tiles)
            t_task = _now()
            _pa_task(
                pmax_ptr,
                psum_ptr,
                pout_ptr,
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                block_tables_ptr,
                key_scale_ptr,
                value_scale_ptr,
                max_blocks_per_seq,
                stride_ks_block,
                stride_ks_head,
                stride_q_row,
                stride_q_head,
                num_rows,
                row,
                start,
                end,
                length,
                task,
                pa_base,
            )
            t_land = _now()
            _land_partial(
                row,
                first,
                count,
                sync_ptr,
                rows_done,
                pmax_ptr,
                psum_ptr,
                pout_ptr,
                x_ptr,
                x_scale_ptr,
                red_base,
                word_base,
                tid,
                trace_ptr,
                bid,
            )
            t_done = _now()
            _stamp_value(trace_ptr, bid, 5, t_land - t_task, tid)
            _stamp_value(trace_ptr, bid, 6, t_done - t_land, tid)
            _stamp_value(trace_ptr, bid, 15, t_task, tid)
            _accumulate(trace_ptr, bid, 7, fx.Int64(1), tid)
            # The next task restages Q and K/V over the same LDS (the claim's
            # barrier orders that).
            task = _next_task(
                sync_ptr, TASK_CLAIM + parity * SYNC_LINE, word_base, tid, total, nblk
            )

        _stamp(trace_ptr, bid, 1, tid)
        # Phase 3: o_proj tiles once every row is published.
        peers = [
            fx.Int64(fx.ptr_load(fx.add_offset(fx.recast_iter(fx.Int64, peers_ptr), d)))
            for d in range_constexpr(world_size)
        ]
        lane = tid % WAVE
        warp = tid // WAVE
        w_rsrc = _rsrc(w_ptr)
        x_rsrc = _rsrc(x_ptr)
        claim = TILE_CLAIM + parity * SYNC_LINE
        tile = _claim_tile(sync_ptr, claim, word_base, tid)
        while tile < n_tiles:
            t_w = _now()
            col_scales = _load_col_scales(tile, tid, w_scale_ptr)
            ws = _load_w_slice(tile, w_rsrc, lane, warp)
            _stage_col_scales(col_scales, tid, w_scale_base)
            _wait_rows(sync_ptr, rows_done, num_rows, tid)
            _stamp(trace_ptr, bid, 2, tid)
            t_g = _now()
            for p in range_constexpr(len(passes)):
                m0, m_count = passes[p]
                if const_expr(p == 0):
                    _tile_pass(
                        tile,
                        m0,
                        m_count,
                        ws,
                        w_scale_base,
                        x_rsrc,
                        x_scale_ptr,
                        num_rows,
                        gemm_base,
                        peers,
                        parity,
                        lane,
                        warp,
                        tid,
                        trace_ptr,
                        bid,
                    )
                else:
                    # A pass of padding rows only is skipped.
                    if 16 * m0 < num_rows:
                        _tile_pass(
                            tile,
                            m0,
                            m_count,
                            ws,
                            w_scale_base,
                            x_rsrc,
                            x_scale_ptr,
                            num_rows,
                            gemm_base,
                            peers,
                            parity,
                            lane,
                            warp,
                            tid,
                            trace_ptr,
                            bid,
                        )
            t_c = _now()
            if const_expr(grid < n_tiles):
                tile = _claim_tile(sync_ptr, claim, word_base, tid)
            else:
                # Every tile goes to some CTA's first claim.
                tile = fx.Int32(n_tiles)
            t_e = _now()
            _accumulate(trace_ptr, bid, 8, t_g - t_w, tid)
            _accumulate(trace_ptr, bid, 12, t_e - t_c, tid)
            _accumulate(trace_ptr, bid, 13, fx.Int64(1), tid)

        _stamp(trace_ptr, bid, 3, tid)
        # Phase 4: a row's all-reduce, residual add and RMSNorm.
        for row in range(bid, num_rows, nblk):
            if const_expr(two_shot and not rs_in_tile):
                _reduce_slice(row, parity, peers, tid)
            _reduce_row(
                row,
                parity,
                peers[rank],
                residual_ptr,
                residual_out_ptr,
                out_ptr,
                norm_w_ptr,
                eps,
                red_base,
                tid,
            )
            # The next row reuses the reduction words.
            gpu.barrier()

        _stamp(trace_ptr, bid, 4, tid)
        if tid == 0:
            fx.ptr_store(epoch + 1, _i32_ptr(sync_ptr, EPOCH + bid))

    @flyc.jit
    def launch(
        query: fx.Pointer,
        key_cache: fx.Pointer,
        value_cache: fx.Pointer,
        key_scale: fx.Pointer,
        value_scale: fx.Pointer,
        block_tables: fx.Pointer,
        context: fx.Pointer,
        pmax: fx.Pointer,
        psum: fx.Pointer,
        pout: fx.Pointer,
        x: fx.Pointer,
        x_scale: fx.Pointer,
        sync: fx.Pointer,
        w: fx.Pointer,
        w_scale: fx.Pointer,
        residual: fx.Pointer,
        residual_out: fx.Pointer,
        out: fx.Pointer,
        norm_w: fx.Pointer,
        peers: fx.Pointer,
        trace: fx.Pointer,
        num_rows: fx.Int32,
        max_blocks_per_seq: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        # The PA tile is tuned with FMA contraction allowed (see pa_decode).
        with CompilationContext.compile_hints({"fastmath": "contract"}):
            mega_paged_attention_kernel(
                query,
                key_cache,
                value_cache,
                key_scale,
                value_scale,
                block_tables,
                context,
                pmax,
                psum,
                pout,
                x,
                x_scale,
                sync,
                w,
                w_scale,
                residual,
                residual_out,
                out,
                norm_w,
                peers,
                trace,
                num_rows,
                max_blocks_per_seq,
                stride_ks_block,
                stride_ks_head,
                stride_q_row,
                stride_q_head,
                eps,
            ).launch(grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream)

    # One wave a SIMD per resident CTA: the register cap that keeps
    # ``occupancy`` CTAs of four waves on a CU.
    waves_per_eu = max(occupancy * threads // (4 * WAVE), 1)
    if traits.FP8_MTP4_LEAN or traits.WAVES_PER_EU_2:
        waves_per_eu = max(waves_per_eu, 2)
    if waves_per_eu > 1:
        launch.compile_hints["waves_per_eu"] = waves_per_eu
    return {
        "launch": launch,
        "traits": traits,
        "capacity": capacity,
        "max_tasks_per_row": max_tasks_per_row,
        "grid": grid,
        "threads": threads,
        "row_k": row_k,
        "sync_words": sync_words(grid),
        "inbox_bytes": inbox_bytes(
            world_size=world_size, max_rows=max_rows, hidden=hidden, two_shot=two_shot
        ),
    }

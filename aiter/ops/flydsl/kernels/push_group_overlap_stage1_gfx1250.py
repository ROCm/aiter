# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Device-side halves of the fused push-group dispatch -> GEMM1 stage-1 kernel
(gfx1250, a8w4 with send-side MX quant).

The whole point is that dispatch and GEMM1 must live in ONE kernel launch: two
kernels queued on the same stream are serialized by stream ordering, so no amount
of tuning makes them overlap. The GEMM half stays where it is, in
``mxfp4_preshuffle_gfx1250_tdm.py``; this module contributes the dispatch half
plus the readiness protocol, and the GEMM kernel calls into it.

CTA roles come from a ticket (``atomic_add`` on ``pg_ov_entry``), not from
``blockIdx``. The grid is deliberately oversubscribed, so with blockIdx-based
roles a producer could be assigned to a CTA that never gets a slot while resident
consumers spin on payload it owes -- a deadlock. Ticket order can only hand a role
to a CTA that is already executing, hence already resident. The same atomic yields
the launch generation, which is what makes every flag here monotonic and therefore
CUDA-graph-replay-safe (a host-supplied epoch would repeat on every replay).

Phase order, per rank, per layer:

  0. producers+planner zero ``tile_arrived`` and the route histogram   [barrier 0]
  1. producers count their (token,k) routes into the histogram and     [barrier 1]
     remember each route's index within this rank's rows for that expert
  2. planner all-gathers the count matrix (one one-way P2P hop, ~1.5 KB per peer;
     NOT a round trip -- every rank can derive both its source-side base and its
     destination-side totals from the same matrix), then builds the compacted
     GEMM1 tile schedule and the per-tile expected row counts   [plan_ready]
  3. producers push fp8 payload + e8m0 scale + rowmap into the planned contiguous
     rows and bump the destination's per-tile arrival counter
  4. every CTA -- producers and planner included, once their duty is done --
     pulls GEMM1 tiles off the sharded work queue until it is empty, gating each
     M-tile on its arrival count

What this does NOT currently buy, measured at bs=256 on 4 ranks, because it keeps
being re-proposed: the fused kernel runs 311 us against 276 for the unfused trio
(85 dispatch + 188 GEMM1 + 3 finalize), and the gap is structural rather than
unfinished work. Phases 2 and 3 are serial with phase 4, not overlapped with it.
The plan costs 39 us before any producer may push, the push then runs 40 to 106,
and the first GEMM tile waits for nearly all of it (see :func:`_emit_pay_meta`),
so phase 4 spans 106 to 307 of a 321 us kernel. Deleting the plan outright would
recover about 19 us of that and no more.

There is no headroom behind it either, which is what any redesign has to clear
first. Fusion's one structural advantage was that the push saturates the fabric
from far fewer CTAs than the unfused dispatch spends on it, but that advantage is
spent: with no overlap to pay for CU efficiency the push is sized for wall time
instead (see the producer count in grouped_moe_gfx1250), which takes it from 85
us on 129 CTAs to 66 on 249 and the CU.us it costs from 11.0k up to 16.3k. Add
GEMM1's 48.1k and the total is 64.5k, or 252 us across 256 CUs; with nothing in
front of it but a local route count, and the ramp, that is 275 -- the unfused 276
to within the run-to-run spread. A perfectly overlapped fusion would tie.

Two things that look like they should help do not, and both were measured rather
than reasoned about: publishing in stages from inside a CTA (:func:`_emit_payload`)
and raising the grid multiplier so a stalled producer shares its CU with a
consumer (330 us at 1 CTA/CU, 327 at 2, 334 at 3 -- the consumers have no ready
tile to work on, so more of them resident changes nothing).

Everything below is called from inside a ``@flyc.kernel`` body. FlyDSL only
AST-rewrites the decorated function, so a plain helper module cannot use dynamic
``if``; ``ASTRewriter.transform`` at the bottom of this file opts these helpers
into the same rewrite, which is what lets them read like ordinary kernel code.
"""
import os

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import arith, const_expr, range_constexpr
from flydsl.expr.rocdl import readlane
from flydsl.expr.typing import Pointer, T
from flydsl.expr.typing import Vector as Vec

import mori.cco.device.flydsl as cco

from aiter.ops.flydsl.kernels import vector
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import (
    make_lds_copy_ops,
    workgroup_barrier,
)
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.dispatch_combine_v2 import flydsl_prims as P
from aiter.ops.flydsl.dispatch_combine_v2.dispatch_combine_op import (
    PG_OV_BARR_GROUPS as _BARR_GROUPS_MAX,
    PG_OV_BARR_SLOT as _BARR_SLOT,
    PG_OV_BARR_STRIDE as _BARR_STRIDE_SLOTS,
    PG_OV_BREL_FANOUT as _BREL_FANOUT_MAX,
    PG_OV_BREL_PHASES as _BAR_PHASES,
    PG_OV_BREL_SLOT as _BREL_SLOT,
    PG_OV_BREL_STRIDE as _BREL_STRIDE_SLOTS,
    PG_OV_PLAN_FANOUT_MAX as _PLAN_FANOUT_MAX,
    PG_OV_PLAN_SLOT as _PLAN_SLOT,
    PG_OV_PLAN_STRIDE as _PLAN_STRIDE_SLOTS,
    PG_OV_WORK_SHARDS as _WORK_SHARDS_MAX,
    PG_OV_WORK_SLOT as WORK_SLOT,
    PG_OV_WORK_STRIDE as _WORK_STRIDE_SLOTS,
)

WAVE = 32
LANE_MASK = WAVE - 1
LOG2_WAVE = 5
# Token payload copy: 4 vec4 streams per lane for memory-level parallelism, the
# same shape the standalone gfx1250 ep_dispatch settled on.
_NSTREAMS = 4
_LANE_STRIDE_I32 = WAVE * 4
_MAIN_STRIDE_I32 = _NSTREAMS * _LANE_STRIDE_I32

# Debug bisect for the payload push: bit 1 rowmap+tis, 2 scale, 4 token, 8 the
# arrival publish. Anything but 15 computes garbage; it exists to isolate which
# scatter a fault comes from.
_PARTS = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PAYLOAD", "15"))
PAYLOAD_PARTS = _PARTS
_PROBE = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PROBE", "0"))
# Debug-only cap on the per-tile arrival spin: a short tile then falls through
# with wrong data instead of hanging, so the host can dump which tiles are short.
_WAIT_MAX = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_WAIT_MAX", "0"))
_BULK_SYNC = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_BULK_SYNC", "0"))
# Scope of the acquire the consumer runs per M-tile: 2 = system/all-address-space,
# 1 = system one-as, 0 = agent. The tile gate runs once per tile per CTA, so an
# over-strong scope here costs a cache reload of the weights on every tile.
_TILE_FENCE = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_TILE_FENCE", "2"))
_NO_GEMM = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_NO_GEMM", "0"))
# Per-tile arrival gate: 1 = on, 0 = dropped entirely, 2 = probed but never
# waited on. 0 computes garbage but is the perfect-overlap bound, and 2 splits
# the difference between the two into evaluation cost and stall. Unlike WAIT_MAX
# neither adds instrumentation of its own.
TILE_GATE = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_TILE_GATE", "1"))
#: Run the whole prefix (reset, count, sort) on the planner instead of spreading
#: it over every dispatch CTA. Spreading it needs three grid-wide barriers to
#: close its three phases, and a barrier's cost here is the slowest of N CTAs,
#: not the counter it increments -- grouped arrival counters, weaker release
#: fences and every release fanout from 2 to 32 all measured neutral or worse
#: against it. The planner's own fences already cover what the barriers did: the
#: release before the count publish orders the reset ahead of any peer's
#: arrivals, and the one before plan_ready orders the sort ahead of any
#: producer's read of it.
PREFIX_OWNER = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PREFIX_OWNER", "1"))
#: Merge a payload run's arrival increments that share a tile into one atomic.
#: 0 restores one atomic per row.
PUB_COALESCE = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PUB_COALESCE", "1"))
#: CPol carried by the payload stores. gfx12 encodes scope in bits 3:4, so 24 is
#: SCOPE_SYS: the write leaves the local L2 instead of dirtying it. The payload is
#: peer memory this rank never reads back, so caching it buys nothing and costs a
#: write-back at the release. Pairs with PUB_FENCE=0 -- a store that is already
#: system-visible on retirement needs a drain, not a cache flush.
#: Default because the pair is worth 54 us of the dispatch half at bs=256: the
#: release used to lower to a device-wide L2 write-back per producer CTA.
PUB_SCOPE = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PUB_SCOPE", "24"))
#: Scope of the release that closes the payload copies: 2 = system over all
#: address spaces (lowers to global_wb), 1 = system one-as, 0 = drain only. 0 is
#: only sound when PUB_SCOPE already took the stores out to system scope.
PUB_FENCE = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PUB_FENCE", "0"))
NO_WAIT_PLAN = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_NO_WAIT_PLAN", "0"))
# Replicas of the plan_ready epoch, each on its own 128 B line. Every CTA in the
# grid takes this wait at once, and on a superscribed grid a single flag makes
# the launch quadratically slower: 2048 CTAs polling one line drown out the
# planner's own P2P traffic (measured 1.9 ms/layer with one flag vs 0.28 ms with
# the wait removed altogether).
_PLAN_FANOUT = int(
    os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PLAN_FANOUT", _PLAN_FANOUT_MAX)
)
if not 1 <= _PLAN_FANOUT <= _PLAN_FANOUT_MAX:
    raise ValueError(f"PLAN_FANOUT must be in [1, {_PLAN_FANOUT_MAX}]")
# Debug bisect inside the planner: 1 publish, 2 count_done+wait, 4 my_base,
# 8 tile_expected, 16 compaction. Only 31 is correct.
_PLAN_PARTS = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PLAN_PARTS", "31"))
BULK_SYNC = _BULK_SYNC
# Spare pg_ov_bar slots (0/1 are the protocol barriers). The count matrix is not
# usable as scratch: a slot that is dead space on one rank holds a real peer
# count on another.
_SYNC_SLOT = 2
_DBG_SLOT = 4

# GEMM1 work queue, in the tail of pg_ov_bar (the region that sizes it owns the
# layout constants). Fewer shards means a hotter counter, more means a coarser
# interleave of the schedule across CTAs.
WORK_SHARDS = int(
    os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_WORK_SHARDS", _WORK_SHARDS_MAX)
)
if WORK_SHARDS not in (1, 2, 4, 8) or WORK_SHARDS > _WORK_SHARDS_MAX:
    raise ValueError(
        f"WORK_SHARDS must be a power of two <= {_WORK_SHARDS_MAX}, got {WORK_SHARDS}"
    )
#: Work items handed out per queue atomic.
#:
#: A work item is one (M, N) tile. It used to be a whole M-tile's worth of N, so
#: that the arrival gate -- keyed on M -- was asked once per item, but that tied
#: the item count to the M-tile count: 384 items at bs=256 over a grid whose
#: resident half is 256 CTAs, so a third of the machine sat out the last round.
#: Measured GEMM-only under this launch with the gate off, 423 us/call at a whole
#: M-tile per item against 279 at one (M, N) tile.
#:
#: What the coarse item bought is bought here instead, without the imbalance: one
#: atomic hands out a contiguous run of ids, and the ids are M-major, so a run
#: nearly always lies inside one M-tile and is gated once. It also divides the
#: pull's own cost -- three workgroup barriers and an atomic, otherwise paid 9216
#: times a layer at bs=256.
#:
#: A chunk is also the granularity the queue balances at, so widening it trades
#: the pull's cost against the tail a coarse chunk leaves, and which way that
#: goes depends on ``tile_m`` -- see :func:`pull_chunk`. Zero picks by shape.
_PULL_CHUNK = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PULL_CHUNK", "0"))
if _PULL_CHUNK < 0:
    raise ValueError(f"PULL_CHUNK must be >= 0, got {_PULL_CHUNK}")


def pull_chunk(C):
    """Work items handed out per queue atomic, from the M-tile size.

    The tail a coarse chunk leaves is what dominates: at bs=256 the schedule is
    48 M-tiles by 24 N-tiles, 1152 items over a 256 CTA grid, so at four items a
    pull there are barely more chunks than CTAs and the last round runs a handful
    of CTAs against an idle machine. Measured GEMM-only with the gate off, 270 us
    at four against 236 at one.

    Pulling against that is the gate, asked once per chunk, so halving the chunk
    doubles how often it is asked. Which side wins is set by how much work an
    item is, and that is ``tile_m``: at 16 the gate's share is large enough that
    four still wins, at 32 and above the tail does. On the full path, us/call at
    four against two: bs=8 262/259, bs=128 500/513, bs=256 599/582, bs=512
    794/770 -- the split falling exactly where tile_m goes from 16 to 32.

    Four was measured back when bs=256 also ran tile_m=16 and was right for it;
    what went stale is the M-tile heuristic moving underneath it, not the tuning.
    """
    if _PULL_CHUNK:
        return _PULL_CHUNK
    return 4 if C.tile_m <= 16 else 2
# Do the dispatch CTAs join the GEMM1 work pool once their payload is out? They
# are a small slice of the grid, but they are also the CTAs that are guaranteed
# resident, so they are worth having.
PROD_JOIN = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PROD_JOIN", "1"))
# A row stays with one warp. The warp pool is narrower than the route count, so
# the rows in flight are a fixed fraction of the push and, bandwidth being
# shared, they all land together: no tile completes until that fraction is out,
# which is the bubble GEMM1 waits out at the start. Splitting a row across warps
# narrows the front, but it multiplies the per-row fixed cost -- metadata chain,
# store drain, arrival atomic -- by the split factor, and measured worse at every
# split (141 -> 315 us of payload at two warps per row). :func:`pay_rows` is that
# same cost taken from the other end, dividing it instead.
# Push the payload sorted by destination instead of in (token, k) order. In token
# order a tile's rows arrive interleaved with every other tile's, so every tile
# completes only near the end of the push and the consumer's gate never opens
# early -- payload and GEMM1 end up strictly serialized.
#   0 = off, 1 = by global expert, 2 = by local expert, peers rank-rotated.
# 1 is the wrong sort even though it is the obvious one: global-expert order is
# destination-major, so rank 0 receives everything in the first 1/npes of the
# push and rank npes-1 in the last, leaving the slowest rank -- the one that sets
# the layer time -- with no overlap at all. 2 walks local experts outer and peers
# inner, so every destination's low tiles fill first, and starting each source at
# a different peer keeps the wire from incasting.
ROUTE_ORDER = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_ROUTE_ORDER", "2"))
#: Run the sort inside the count all-gather instead of in front of it. Only the
#: histogram feeds it and only the payload pass reads it, so on the planner's
#: timeline it belongs in the one stall that is pure fabric latency -- the wait
#: for the peers' counts -- rather than ahead of the publish that starts that
#: latency. Measured 4.5 us of the planner's 43.9 at bs=256, all of it recovered.
#: Only meaningful under :data:`PREFIX_OWNER`; the spread version already ran the
#: sort on producers, which are idle across the same window.
ORDER_IN_GATHER = int(
    os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_ORDER_IN_GATHER", "1")
)
# Rows a run may hold. Each one keeps six values live across the run's copies,
# so this is a register budget shared with the GEMM the kernel also carries.
_PAY_ROWS_MAX = 8
# Override for :func:`pay_rows`; 0 leaves it derived from the shape.
_PAY_ROWS = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_PAY_ROWS", "0"))
if _PAY_ROWS and not 1 <= _PAY_ROWS <= _PAY_ROWS_MAX:
    raise ValueError(f"PAY_ROWS must be in [1, {_PAY_ROWS_MAX}], got {_PAY_ROWS}")


def pay_rows(C, block):
    """Consecutive routes one warp pushes between two arrival publishes.

    A publish is a store drain, a system release and the arrival atomic, and at
    one row per publish a warp pays it for every row it owns -- eight full fabric
    round trips deep in a warp on the 512-token shape, each one also keeping the
    next row's metadata chain from being issued. Batching makes it one publish
    per run and, the run being unrolled, lets every row's metadata chain be in
    flight at once. Measured 847 -> 562 us per layer on that shape, and it is
    also what took the 2-to-61-layer growth from +61% to +3%: a per-row publish
    injects the rank skew once per row, and the skew is what compounds.

    The routes are destination-sorted, so a run is usually one expert's
    contiguous rows on one peer, which is the shape the fabric wants anyway.

    One run per warp is the ideal and the row count per warp is the value that
    gives it: a run is indivisible, so a longer one only idles warps -- at 8 on
    the 64-token shape, where a warp owns a single row, 7 warps in 8 go empty and
    the layer costs 238 us against 175. The measured optimum is this ratio at
    both ends (8 and 2 respectively), which is what this returns.
    """
    if _PAY_ROWS:
        return _PAY_ROWS
    warps = max(C.n_producers * (block // WAVE), 1)
    return max(2, min(_PAY_ROWS_MAX, (C.max_tok * C.topk) // warps))


# Width of the two-level scan over the histogram. The inner runs unrolled in one
# thread, so this trades code size against how many threads the outer level can
# keep busy. At e_total=384 a chunk of 32 leaves only 12 threads with any work,
# but widths from 4 to 32 all measure the same, so the scan is not what the
# prefix is waiting on.
_SCAN_CHUNK = 32


#: In-kernel phase timing into the spare pg_ov_bar slots. Off by default: it costs
#: a scalar counter read and one store per instrumented CTA per layer, which is
#: nothing, but the slots it writes are the ones the debug builds share.
TSTAMP = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_TSTAMP", "0"))
#: Timing slots live in the gap between the last work-stealing head (WORK_SLOT +
#: 7*WORK_STRIDE = 128) and the plan-ready replicas (PLAN_SLOT = 256). Not the
#: low spare slots: 4..7 are the WAIT_MAX timeout report, and anything inside the
#: work-head range would false-share a line with an atomic the whole grid hits.
TS_BASE = 192
TS_PLAN_OWNER = TS_BASE + 0  # planner: entry -> plan published
TS_PROD_WAIT = TS_BASE + 1  # producer: entry -> plan seen
TS_PROD_END = TS_BASE + 2  # producer: entry -> payload pushed
TS_CONS_WAIT = TS_BASE + 3  # consumer: entry -> plan seen
TS_CONS_END = TS_BASE + 4  # consumer: entry -> work pool drained
TS_PLAN_COUNT = TS_BASE + 5  # planner: entry -> reset and route count done
TS_PLAN_ORDER = TS_BASE + 6  # planner: entry -> routes sorted by destination
TS_PLAN_PUB = TS_BASE + 7  # planner: entry -> own counts pushed to every peer
TS_PLAN_GATHER = TS_BASE + 8  # planner: entry -> every peer's counts in hand
#: Maxima over every CTA of a role, not one representative. The kernel ends when
#: the LAST CTA does, and a single sample cannot see that: raising the grid
#: multiplier moves the first consumer 32 us earlier while the kernel does not
#: budge, which only makes sense once the tail is measured too.
TS_PROD_LAST = TS_BASE + 9  # last producer to finish pushing
TS_POOL_LAST = TS_BASE + 10  # last CTA of any role to drain the work pool
#: Ticks summed over every gate poll by every CTA, so the average wait per CTA is
#: this over the grid. A sum and not a sample: whether a tile is ready depends on
#: which band it is in and which CTA drew it, and one CTA's history says nothing
#: about the grid's.
TS_GATE_SPIN = TS_BASE + 11


def _tstamp_addr(C, slot):
    return fx.Int64(C.ptr_bar) + fx.Int64(slot * 4)


def emit_now():
    """Read the constant-rate counter, or nothing when timing is compiled out."""
    if const_expr(not TSTAMP):
        return None
    return P.read_realtime()


def _emit_tstamp(C, slot, t0, tid, active):
    """Record ``now - t0`` in ticks for one representative CTA per role.

    A plain store, not an atomic max over the role's CTAs: the point is to split
    one CTA's timeline into phases that sum to its own total, and a max over CTAs
    would mix phases from different ones into a total no CTA ever saw. The last
    layer of the run wins, which is the one worth reading anyway -- the first
    still pays for cold weights.
    """
    if const_expr(not TSTAMP):
        return
    if active:
        if tid == fx.Int32(0):
            delta = fx.Int32(P.read_realtime() - t0)
            P.store_i32_system(_tstamp_addr(C, slot), fx.Int32(0), delta)


def _emit_tstamp_acc(C, slot, t0, tid):
    """Add ``now - t0`` to a running total shared by the whole grid."""
    if const_expr(not TSTAMP):
        return
    if tid == fx.Int32(0):
        P.atomic_add_agent(_tstamp_addr(C, slot), fx.Int32(P.read_realtime() - t0))


def _emit_tstamp_max(C, slot, t0, tid):
    """Fold ``now - t0`` into a max over every CTA that reaches this point.

    The planner zeroes these slots before it publishes the plan, so the value is
    this launch's tail and not the worst layer of the run -- layer 0 pays for cold
    weights and would otherwise win every max.
    """
    if const_expr(not TSTAMP):
        return
    if tid == fx.Int32(0):
        P.atomic_max_agent(_tstamp_addr(C, slot), fx.Int32(P.read_realtime() - t0))


def work_head_addr(C, shard):
    """Address of one work-stealing head, shard-major on 64 B lines."""
    return (
        fx.Int64(C.ptr_bar)
        + fx.Int64(WORK_SLOT * 4)
        + fx.Int64(shard) * fx.Int64(_WORK_STRIDE_SLOTS * 4)
    )


def scan_chunks(C):
    """Outer-level width of the histogram scan (see :func:`_emit_route_order`)."""
    return (C.e_total + _SCAN_CHUNK - 1) // _SCAN_CHUNK


def lds_scratch_bytes(C):
    """LDS the stage-1 phases borrow from the GEMM's tile arena.

    Both users (the planner's compaction scan and the producers' counting sort)
    run while this CTA is still several barriers away from its first tile, so the
    arena is theirs -- as long as it is this wide.
    """
    chunks = scan_chunks(C)
    sort = chunks * _SCAN_CHUNK + chunks + C.e_total
    return max(C.epr + 1, sort) * 4


def plan_ready_addr(C, replica):
    """Address of one plan-ready replica, one per 128 B line."""
    return (
        fx.Int64(C.ptr_bar)
        + fx.Int64(_PLAN_SLOT * 4)
        + fx.Int64(replica) * fx.Int64(_PLAN_STRIDE_SLOTS * 4)
    )


#: Scope of the release each CTA takes on its way into a producer barrier.
#: Debug: 0 drops it to agent, which is not enough to publish the tile_arrived
#: zeroing to the peers that write it.
_BAR_FENCE = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_BAR_FENCE", "1"))
#: Override for :func:`_bar_fanout`; 0 leaves it derived from the barrier width.
_BAR_FANOUT = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_BAR_FANOUT", "0"))
if _BAR_FANOUT and not 1 <= _BAR_FANOUT <= _BREL_FANOUT_MAX:
    raise ValueError(f"BAR_FANOUT must be in [1, {_BREL_FANOUT_MAX}]")


def _bar_fanout(width):
    """Release replicas for a barrier of ``width`` CTAs.

    Two costs pull against each other. The last arriver writes the replicas one
    at a time from a single thread, so the burst is linear in this; but every
    replica is shared by ``width / fanout`` pollers, and a shared line is what the
    fanout exists to avoid. Measured (dispatch side, 61 layers): at 33 CTAs, 4
    replicas cost 379 us and 64 cost 521; at 128, 16 cost 248 and either 4 or 64
    cost around 310. An eighth of the width lands on both optima.
    """
    return _BAR_FANOUT or max(4, min(_BREL_FANOUT_MAX, width // 8))


def _brel_addr(C, phase, replica):
    """Address of one barrier-release replica, one per 128 B line."""
    return (
        fx.Int64(C.ptr_bar)
        + fx.Int64((_BREL_SLOT + phase * _BREL_FANOUT_MAX * _BREL_STRIDE_SLOTS) * 4)
        + fx.Int64(replica) * fx.Int64(_BREL_STRIDE_SLOTS * 4)
    )


#: CTAs per arrival group, mirroring the release fanout onto the arrival side.
#: 0 keeps the single shared counter, which is what measured fastest.
_BAR_GROUP = int(os.environ.get("AITER_EP_PUSH_GROUP_OVERLAP_BAR_GROUP", "0"))


def _bar_groups(width):
    """Arrival groups for a barrier of ``width`` CTAs; 1 is the single counter."""
    if not _BAR_GROUP or width <= _BAR_GROUP:
        return 1
    n = (width + _BAR_GROUP - 1) // _BAR_GROUP
    if n > _BARR_GROUPS_MAX:
        raise ValueError(
            f"barrier of {width} CTAs needs {n} arrival groups, "
            f"only {_BARR_GROUPS_MAX} are allocated"
        )
    return n


def _barr_addr(C, phase, group):
    """Address of one barrier arrival counter, one group per 128 B line."""
    return (
        fx.Int64(C.ptr_bar)
        + fx.Int64((_BARR_SLOT + phase * _BARR_GROUPS_MAX * _BARR_STRIDE_SLOTS) * 4)
        + fx.Int64(group) * fx.Int64(_BARR_STRIDE_SLOTS * 4)
    )




#: Dispatch-side barriers, in the order a launch takes them. Each owns its own
#: arrival counter (slot 0..2 of ``pg_ov_bar``) and its own replica bank.
_BAR_ZERO, _BAR_COUNT, _BAR_ORDER = 0, 1, 2


def _emit_cta_barrier(C, phase, epoch, tid, me, width):
    """Block until all ``width`` dispatch CTAs have reached ``phase``.

    Arrival and release are deliberately not the same address. Spinning on the
    counter a CTA also increments makes every poll miss against every other
    CTA's atomic, and the misses go as the square of the participant count: these
    barriers together are free at 32 producers and cost 284 us/layer at 128,
    which is what pins the push to a producer count far below the one it wants
    (245 us of payload at 32 producers against 59 at 128).

    So only the last arriver writes, and it writes replicas -- one per 128 B line
    -- that nobody contends for. Both the counter and the released value stay
    monotonic across launches, so nothing here needs resetting and a CTA that
    arrives while a replica still holds an older epoch simply spins on.

    Arrival stays on one line unless :data:`_BAR_GROUP` says otherwise, which is
    the mirror of the release fanout and would be the obvious next move -- but
    the two sides are not symmetric. Polling a line every other CTA is writing
    is what was quadratic; incrementing one the L2 pipelines, so grouping the
    arrivals only buys a second dependent atomic round trip on the critical
    path.
    """
    fanout = _bar_fanout(width)
    n_groups = _bar_groups(width)
    if tid == fx.Int32(0):
        if const_expr(_BAR_FENCE):
            P.fence_system_release()
        else:
            P.fence_agent_release()
        last = fx.Int32(1) == fx.Int32(1)
        if const_expr(n_groups > 1):
            group = me // fx.Int32(_BAR_GROUP)
            # The last group is short unless the group size divides the width.
            rest = fx.Int32(width) - group * fx.Int32(_BAR_GROUP)
            gsize = arith.select(rest < fx.Int32(_BAR_GROUP), rest, fx.Int32(_BAR_GROUP))
            in_group = fx.Int32(
                P.atomic_add_agent(_barr_addr(C, phase, group), fx.Int32(1))
            )
            last = in_group == gsize * epoch - fx.Int32(1)
        # Without grouping every CTA reaches the top counter; with it, one per
        # group does.
        top = n_groups if n_groups > 1 else width
        if last:
            arrived = fx.Int32(
                P.atomic_add_agent(
                    fx.Int64(C.ptr_bar) + fx.Int64(phase * 4), fx.Int32(1)
                )
            )
            if arrived == fx.Int32(top) * epoch - fx.Int32(1):
                for r in range_constexpr(fanout):
                    P.store_i32_system(
                        _brel_addr(C, phase, fx.Int32(r)), fx.Int32(0), epoch
                    )
        # The last arriver takes this too and falls straight through, which is
        # cheaper than branching around it.
        P.spin_until_ge_i32_backoff(
            _brel_addr(C, phase, me % fx.Int32(fanout)), epoch, agent=True
        )
    workgroup_barrier()


_emit_cta_barrier = ASTRewriter.transform(_emit_cta_barrier)


#: Constructor argument order, also the wire format of the ``Constexpr[tuple]``
#: the GEMM kernel carries. Constexpr only accepts scalars and tuples of scalars,
#: so the geometry travels as a flat tuple and is rehydrated during tracing.
OVERLAP_FIELDS = (
    "rank",
    "npes",
    "experts_per_rank",
    "experts_per_token",
    "token_nbytes",
    "max_tok_per_rank",
    "cap",
    "tile_m",
    "tiles_per_expert",
    "scale_num_i32",
    "scale_wmma_rep",
    "dispatch_ctas",
    "grid_ctas",
    "arena_handle",
    "off_count",
    "off_count_done",
    "off_tile_arrived",
    "off_disp_out",
    "off_out_scales",
    "off_pg_rowmap",
    "off_tis",
    "ptr_entry",
    "ptr_bar",
    "ptr_plan_ready",
    "ptr_my_base",
    "ptr_hist",
    "ptr_route_slot",
    "ptr_route_order",
    "ptr_count",
    "ptr_count_done",
    "ptr_tile_arrived",
    "ptr_tile_expected",
)


class OverlapConsts:
    """Compile-time geometry shared by the stage-1 helpers.

    Plain attribute bag rather than a dataclass: it is read during FlyDSL tracing,
    where every field must stay a Python int so it folds into the IR.
    """

    def __init__(
        self,
        *,
        rank,
        npes,
        experts_per_rank,
        experts_per_token,
        token_nbytes,
        max_tok_per_rank,
        cap,
        tile_m,
        tiles_per_expert,
        scale_num_i32,
        scale_wmma_rep,
        dispatch_ctas,
        grid_ctas,
        arena_handle,
        off_count,
        off_count_done,
        off_tile_arrived,
        off_disp_out,
        off_out_scales,
        off_pg_rowmap,
        off_tis,
        ptr_entry,
        ptr_bar,
        ptr_plan_ready,
        ptr_my_base,
        ptr_hist,
        ptr_route_slot,
        ptr_route_order,
        ptr_count,
        ptr_count_done,
        ptr_tile_arrived,
        ptr_tile_expected,
    ):
        self.rank = rank
        self.npes = npes
        self.epr = experts_per_rank
        self.topk = experts_per_token
        self.e_total = npes * experts_per_rank
        self.token_nbytes = token_nbytes
        self.n_i32 = token_nbytes // 4
        self.max_tok = max_tok_per_rank
        self.cap = cap
        self.tile_m = tile_m
        self.tiles_per_expert = tiles_per_expert
        self.scale_num_i32 = scale_num_i32
        self.wmma_rep = scale_wmma_rep
        self.dispatch_ctas = dispatch_ctas
        self.grid_ctas = grid_ctas
        self.arena_handle = arena_handle
        self.off_count = off_count
        self.off_count_done = off_count_done
        self.off_tile_arrived = off_tile_arrived
        self.off_disp_out = off_disp_out
        self.off_out_scales = off_out_scales
        self.off_pg_rowmap = off_pg_rowmap
        self.off_tis = off_tis
        self.ptr_entry = ptr_entry
        self.ptr_bar = ptr_bar
        self.ptr_plan_ready = ptr_plan_ready
        self.ptr_my_base = ptr_my_base
        self.ptr_hist = ptr_hist
        self.ptr_route_slot = ptr_route_slot
        self.ptr_route_order = ptr_route_order
        self.ptr_count = ptr_count
        self.ptr_count_done = ptr_count_done
        self.ptr_tile_arrived = ptr_tile_arrived
        self.ptr_tile_expected = ptr_tile_expected
        # Per-lane run of scale dwords: pick the narrowest vector width that still
        # covers the row in one step (same reasoning as ep_dispatch's scale copy).
        steps = (scale_num_i32 + WAVE - 1) // WAVE
        self.scale_vec = 1
        for v in (2, 4, 8):
            if v >= steps and scale_num_i32 % v == 0:
                self.scale_vec = v
                break
        if cap % (max(scale_wmma_rep, 1) * 16):
            raise ValueError(
                f"cap={cap} must be a multiple of wmma_rep*16="
                f"{scale_wmma_rep * 16} for the send-side scale preshuffle"
            )
        if tile_m != scale_wmma_rep * 16:
            raise ValueError(
                f"tile_m={tile_m} must equal wmma_rep*16={scale_wmma_rep * 16}"
            )
        if cap % tile_m or cap // tile_m != tiles_per_expert:
            raise ValueError(f"cap={cap} / tile_m={tile_m} != {tiles_per_expert}")
        # Role layout: 0 = planner, [1, dispatch_ctas) = producers, the rest are
        # GEMM1 consumers. The planner is kept out of the producer set so it can
        # start the count exchange the instant barrier 1 opens.
        self.n_producers = dispatch_ctas - 1
        self.consumer_ctas = grid_ctas - dispatch_ctas
        if self.n_producers < 1 or self.consumer_ctas < 1:
            raise ValueError(
                f"grid_ctas={grid_ctas} / dispatch_ctas={dispatch_ctas} leaves "
                f"{self.n_producers} producers and {self.consumer_ctas} consumers; "
                "both must be >= 1"
            )

    def as_tuple(self):
        return tuple(getattr(self, _CTOR_TO_ATTR.get(f, f)) for f in OVERLAP_FIELDS)

    @classmethod
    def from_tuple(cls, values):
        return cls(**dict(zip(OVERLAP_FIELDS, values)))


#: Constructor keyword -> attribute name, for the few that are stored shortened.
_CTOR_TO_ATTR = {
    "experts_per_rank": "epr",
    "experts_per_token": "topk",
    "max_tok_per_rank": "max_tok",
    "scale_wmma_rep": "wmma_rep",
}


def _addr(x):
    """Address of a kernel argument, whether it arrived as a pointer or an i64.

    The GEMM half takes its tile-schedule buffers as ``fx.Pointer`` and the
    dispatch-side inputs as raw i64 (matching the standalone dispatch ABI), and
    both land in these helpers."""
    if isinstance(x, int) or isinstance(x, fx.Int64):
        return fx.Int64(x)
    if isinstance(x, Pointer):
        return fx.Int64(fx.ptrtoint(x))
    return fx.Int64(x)


def _rsrc(addr):
    return create_buffer_resource_from_addr(_addr(addr))


# --------------------------------------------------------------------------- #
# Phase 0/1: role tickets, counter reset, route counting
# --------------------------------------------------------------------------- #
def _emit_ticket(C, tid, lds_idx):
    """One atomic per CTA yields both the role and the launch generation.

    Launch N's CTAs all increment before any of launch N+1's do (the stream
    serializes the launches), so ``ticket // grid_ctas`` is a per-launch
    generation and the remainder is a dense role id.

    Thread 0 broadcasts the ticket over LDS, the same way the work queue hands
    out its items. Going through a per-CTA global slot instead costs 23 us/layer:
    the slots pack 32 CTAs into one 128 B line and every CTA writes its own at
    system scope and reads it straight back, so the line bounces once per CTA
    with the whole grid's launch behind it.
    """
    lds_load, lds_store = make_lds_copy_ops(32)
    if tid == fx.Int32(0):
        t = P.atomic_add_agent(fx.Int64(C.ptr_entry), fx.Int32(1))
        lds_store(lds_idx, arith.index(0), Vec.from_elements([fx.Int32(t)], fx.Int32))
    workgroup_barrier()
    ticket = lds_load(lds_idx, arith.index(0))[0]
    workgroup_barrier()
    generation = ticket // fx.Int32(C.grid_ctas)
    role = ticket - generation * fx.Int32(C.grid_ctas)
    # 1-based epoch: every flag here counts up by one per launch from a zeroed
    # arena, so launch 0 must wait for 1. A 0-based epoch would make the first
    # launch's waits pass immediately and skip the barrier entirely.
    return role, generation + fx.Int32(1)


def _emit_prefix_owner(C, addr_inp_idx, tid, nthreads, cur_tok):
    """Reset and count on the planner alone, closed by workgroup barriers.

    Both grid barriers of the spread version were only there to close these two
    phases against each other and against the peers; on one CTA the workgroup
    barrier closes the first, and the release that :func:`_emit_plan` already
    takes before publishing the counts closes the second -- a peer cannot bump
    ``tile_arrived`` until it has seen those counts.
    """
    rs_arrived = _rsrc(C.ptr_tile_arrived)
    for i in range(tid, fx.Int32(C.epr * C.tiles_per_expert), nthreads):
        buffer_store(fx.Int32(0), rs_arrived, i)
    rs_hist = _rsrc(C.ptr_hist)
    for i in range(tid, fx.Int32(C.e_total), nthreads):
        buffer_store(fx.Int32(0), rs_hist, i)
    P.waitcnt_stores()
    workgroup_barrier()
    # No fence in between: the atomics below land on the very lines the reset
    # just wrote, from the same CTA, so the L2 orders them.
    rs_idx = _rsrc(addr_inp_idx)
    rs_slot = _rsrc(C.ptr_route_slot)
    for w in range(tid, cur_tok * fx.Int32(C.topk), nthreads):
        ge = buffer_load(rs_idx, w, vec_width=1, dtype=T.i32)
        slot = P.atomic_add_agent(
            fx.Int64(C.ptr_hist) + fx.Int64(ge) * fx.Int64(4), fx.Int32(1)
        )
        buffer_store(slot, rs_slot, w)
    P.waitcnt_stores()
    workgroup_barrier()
    P.fence_agent_acquire()


def _emit_zero_and_count(C, addr_inp_idx, role, epoch, tid, nthreads, cur_tok):
    """Phase 0 + 1 on the planner and producer CTAs.

    ``tile_arrived`` can be zeroed here without racing a peer: a peer only writes
    it after observing this rank's count publication, and that publication is
    release-ordered behind barrier 1, which is behind this reset.

    Under :data:`PREFIX_OWNER` the planner does both phases alone and the two
    barriers go away with them. Moving only the reset that way measures neutral
    -- the convergence it saves is the serialization it adds back -- so the two
    have to move together for either to pay.
    """
    if const_expr(PREFIX_OWNER):
        _prefix_owner(C, addr_inp_idx, tid, nthreads, cur_tok)
        return
    slots = C.epr * C.tiles_per_expert
    n_reset_ctas = C.dispatch_ctas
    reset_stride = fx.Int32(n_reset_ctas) * nthreads
    reset_id = role * nthreads + tid

    rs_arrived = _rsrc(C.ptr_tile_arrived)
    for i in range(reset_id, slots, reset_stride):
        buffer_store(fx.Int32(0), rs_arrived, i)
    rs_hist = _rsrc(C.ptr_hist)
    for i in range(reset_id, C.e_total, reset_stride):
        buffer_store(fx.Int32(0), rs_hist, i)

    P.waitcnt_stores()
    workgroup_barrier()
    _emit_cta_barrier(C, _BAR_ZERO, epoch, tid, role, n_reset_ctas)
    # The acquire half of the barrier, and not optional: the counting below reads
    # hist through a buffer resource while the zeroing wrote it the same way, but
    # the barrier alone does not order them for the compiler.
    P.fence_system_acquire()

    # Phase 1: count. Producers only -- the planner has the count publish to do
    # the moment barrier 1 opens, so leaving it out of the count keeps it free.
    if role > fx.Int32(0):
        prod_id = role - fx.Int32(1)
        work_limit = cur_tok * fx.Int32(C.topk)
        rs_idx = _rsrc(addr_inp_idx)
        rs_slot = _rsrc(C.ptr_route_slot)
        base = prod_id * nthreads + tid
        stride = fx.Int32(C.n_producers) * nthreads
        for w in range(base, work_limit, stride):
            ge = buffer_load(rs_idx, w, vec_width=1, dtype=T.i32)
            # Agent-scope atomic on a local counter: the returned old value IS
            # this route's index among this rank's rows for that expert, which is
            # all the payload pass needs to rebuild its destination row.
            slot = P.atomic_add_agent(
                fx.Int64(C.ptr_hist) + fx.Int64(ge) * fx.Int64(4), fx.Int32(1)
            )
            buffer_store(slot, rs_slot, w)

    P.waitcnt_stores()
    workgroup_barrier()
    # Peers start bumping tile_arrived once the planner publishes counts, which is
    # gated on this barrier -- so the zeroing above has to be out of this CTA's
    # caches by now, not merely store-complete. The barrier's release fence is
    # what does that.
    _emit_cta_barrier(C, _BAR_COUNT, epoch, tid, role, n_reset_ctas)
    # Without this the planner's hist read floats above the barrier -- the routes
    # are counted with global atomics but read back through a buffer resource, and
    # nothing else tells the compiler those touch the same memory.
    P.fence_system_acquire()


# --------------------------------------------------------------------------- #
# Phase 1b: counting-sort the routes by destination expert (producer CTAs)
# --------------------------------------------------------------------------- #
def _push_order_expert(C, j):
    """Global expert at position ``j`` of the push order (see :data:`ROUTE_ORDER`)."""
    if const_expr(ROUTE_ORDER >= 2):
        le = j // fx.Int32(C.npes)
        peer = j - le * fx.Int32(C.npes) + fx.Int32(C.rank)
        peer = arith.select(
            peer >= fx.Int32(C.npes), peer - fx.Int32(C.npes), peer
        )
        return peer * fx.Int32(C.epr) + le
    return j


def _emit_route_order_body(
    C, addr_inp_idx, tid, nthreads, cur_tok, scatter_base, scatter_stride
):
    """Counting sort over the histogram; the caller owns the LDS and the barrier.

    The scan is the same work on every CTA that runs it, so only the scatter
    takes a partition.
    """
    lds_load, lds_store = make_lds_copy_ops(32)
    lds = fx.index_cast(T.index, fx.ptrtoint(fx.get_dyn_shared()))
    nch = scan_chunks(C)
    # Histogram in push order, padded to a whole number of chunks so the scan
    # needs no tail predicate; then the chunk sums; then the bases, which are
    # indexed by global expert because that is what the scatter has in hand.
    cs_off = fx.Int32(nch * _SCAN_CHUNK * 4)
    base_off = fx.Int32((nch * _SCAN_CHUNK + nch) * 4)
    rs_hist = _rsrc(C.ptr_hist)

    for c in range(tid, fx.Int32(nch), nthreads):
        acc = fx.Int32(0)
        for i in range_constexpr(_SCAN_CHUNK):
            j = c * fx.Int32(_SCAN_CHUNK) + fx.Int32(i)
            inb = j < fx.Int32(C.e_total)
            ge = _push_order_expert(C, arith.select(inb, j, fx.Int32(0)))
            v = arith.select(
                inb,
                buffer_load(rs_hist, ge, vec_width=1, dtype=T.i32),
                fx.Int32(0),
            )
            # Cached here so the third pass scans LDS instead of re-reading the
            # histogram from L2.
            lds_store(lds, j * fx.Int32(4), Vec.from_elements([v], fx.Int32))
            acc = acc + v
        lds_store(lds, cs_off + c * fx.Int32(4), Vec.from_elements([acc], fx.Int32))
    workgroup_barrier()

    # Scan the chunk totals on one thread, then have every chunk owner walk its
    # own 32 experts from that base: nch + 32 serial steps instead of e_total,
    # which matters because e_total is 384 at epr=96.
    if tid == fx.Int32(0):
        acc = fx.Int32(0)
        for c in range_constexpr(nch):
            v = lds_load(lds, arith.index(nch * _SCAN_CHUNK * 4 + c * 4))[0]
            lds_store(
                lds,
                arith.index(nch * _SCAN_CHUNK * 4 + c * 4),
                Vec.from_elements([acc], fx.Int32),
            )
            acc = acc + v
    workgroup_barrier()

    for c in range(tid, fx.Int32(nch), nthreads):
        acc = lds_load(lds, cs_off + c * fx.Int32(4))[0]
        for i in range_constexpr(_SCAN_CHUNK):
            j = c * fx.Int32(_SCAN_CHUNK) + fx.Int32(i)
            v = lds_load(lds, j * fx.Int32(4))[0]
            # The padding slots carry a zero count, so they leave acc alone; they
            # must still not scatter, their expert id is not theirs.
            if j < fx.Int32(C.e_total):
                lds_store(
                    lds,
                    base_off + _push_order_expert(C, j) * fx.Int32(4),
                    Vec.from_elements([acc], fx.Int32),
                )
            acc = acc + v
    workgroup_barrier()

    rs_idx = _rsrc(addr_inp_idx)
    rs_slot = _rsrc(C.ptr_route_slot)
    rs_order = _rsrc(C.ptr_route_order)
    for w in range(scatter_base, cur_tok * fx.Int32(C.topk), scatter_stride):
        ge = buffer_load(rs_idx, w, vec_width=1, dtype=T.i32)
        slot = buffer_load(rs_slot, w, vec_width=1, dtype=T.i32)
        pos = lds_load(lds, base_off + ge * fx.Int32(4))[0] + slot
        buffer_store(w, rs_order, pos)


def _emit_route_order(C, addr_inp_idx, role, epoch, tid, nthreads, cur_tok):
    """Build ``order[base[ge] + slot] = route``, the routes grouped by destination.

    Free, in two senses. It costs no extra pass over the routes: phase 1 already
    left the histogram and each route's ``slot`` within its expert, so this is a
    counting sort with O(1) work per route -- a prefix sum over ``e_total`` plus
    one scatter. And it costs no wall time: only producers run it, while the
    planner is off doing the count all-gather, so it hides in the plan's shadow.
    ``base`` here is this rank's own prefix over experts in push order, unrelated
    to ``my_base`` (the prefix over peer ranks, which is not known yet).

    The positions are a permutation of ``[0, cur_tok*topk)``, so every entry is
    written this launch and no reader can pick up a stale one.

    Under :data:`PREFIX_OWNER` the planner owns this too, so it runs before the
    plan rather than inside the all-gather's shadow, and the barrier that used to
    close it is subsumed by the release in front of plan_ready. Every producer
    scanned the same 384-entry histogram to build the same table anyway; only the
    scatter was ever divided.
    """
    if const_expr(PREFIX_OWNER):
        _route_order_body(C, addr_inp_idx, tid, nthreads, cur_tok, tid, nthreads)
        return
    if role > fx.Int32(0):
        # Same route partition as phase 1, so ``slot`` is read back by the thread
        # that produced it.
        prod_id = role - fx.Int32(1)
        _route_order_body(
            C, addr_inp_idx, tid, nthreads, cur_tok,
            prod_id * nthreads + tid, fx.Int32(C.n_producers) * nthreads,
        )
        # The acquire is not optional even though order is local scratch that no
        # peer reads: last launch's order sat at these very addresses and can
        # still be in a reader's L1.
        P.waitcnt_stores()
        workgroup_barrier()
        _emit_cta_barrier(C, _BAR_ORDER, epoch, tid, prod_id, C.n_producers)
        P.fence_agent_acquire()


# --------------------------------------------------------------------------- #
# Phase 2: count all-gather + plan (planner CTA only)
# --------------------------------------------------------------------------- #
def _emit_plan(
    C, addr_trb, addr_eids, addr_tvd, addr_num_valid, epoch, parity, tid, nthreads,
    t0=None, mid=None,
):
    """Publish this rank's histogram to every peer, wait for theirs, then plan.

    The all-gather (as opposed to sending each peer only the slice it owns) is
    what keeps this to a single one-way hop: with the full matrix in hand, a rank
    computes its own source-side row base AND its destination-side totals locally.
    Sending only the owned slice would force the destination to prefix-sum and
    send bases back -- a second hop, and on gfx1250 a cross-device handshake costs
    more than the payload it is guarding.

    ``mid`` runs after this rank's counts are on the wire and before the wait for
    everyone else's -- local work that depends on the histogram but not on the
    peers, hidden inside a stall that is pure fabric latency. The release in front
    of plan_ready covers whatever it writes.
    """
    window = cco.Window(fx.Int64(C.arena_handle))
    rs_hist = _rsrc(C.ptr_hist)

    # Rearm the GEMM1 work queue. This is the one safe window for it: a CTA only
    # pulls after it has seen plan_ready, which this CTA bumps at the end, and
    # the previous launch's pulls are behind the stream's kernel boundary.
    if tid < fx.Int32(WORK_SHARDS):
        P.store_i32_system(work_head_addr(C, tid), fx.Int32(0), fx.Int32(0))
    if const_expr(TSTAMP):
        # Same window, same reason: nobody folds into these until they have seen
        # the plan this CTA is about to publish, so the release in front of it
        # orders the reset and these can be plain stores.
        if tid == fx.Int32(0):
            buffer_store(fx.Int32(0), _rsrc(C.ptr_bar), fx.Int32(TS_PROD_LAST))
            buffer_store(fx.Int32(0), _rsrc(C.ptr_bar), fx.Int32(TS_POOL_LAST))
            buffer_store(fx.Int32(0), _rsrc(C.ptr_bar), fx.Int32(TS_GATE_SPIN))

    if const_expr(_PROBE):
        # Four writes that differ in one axis each (local/peer, buffer/raw), so a
        # single dump of count[0][0][1..4] says which axis is broken.
        if tid == fx.Int32(0):
            buffer_store(fx.Int32(1000 + C.rank), _rsrc(C.ptr_count), 1)
            P.store_i32_system(
                fx.Int64(C.ptr_count), fx.Int32(2), fx.Int32(2000 + C.rank)
            )
            _p0 = fx.Int64(window.lsa_ptr(fx.Int32(0), C.off_count))
            buffer_store(fx.Int32(3000 + C.rank), _rsrc(_p0), 3)
            P.store_i32_system(_p0, fx.Int32(4), fx.Int32(4000 + C.rank))

    # Publish the WHOLE histogram to EVERY peer, not just each expert's owner: a
    # rank needs the counts for the experts it sends to (to place its rows) as
    # much as for the experts it owns (to size their tiles). Work item w is
    # (peer, expert group), expert-major so neighbouring lanes write neighbouring
    # dwords. npes*e_total dwords is under 2 KB of wire traffic.
    #
    # Four experts per store, not one. The bytes are trivial but each scalar store
    # is a separate partial-line write across the fabric that the drain below then
    # waits on, and this loop measured 27 us of the plan's 49 at e_total=384. The
    # count region is 256 B aligned and e_total is a multiple of npes, so a
    # 4-expert group never straddles a peer's row.
    _PUB_VEC = 4 if C.e_total % 4 == 0 else 1
    _pub_per_peer = C.e_total // _PUB_VEC
    row_base = (
        parity * fx.Int32(C.npes * C.e_total) + fx.Int32(C.rank * C.e_total)
    )
    publish_end = fx.Int32(C.npes * _pub_per_peer) if const_expr(_PLAN_PARTS & 1) else tid
    for w in range(tid, publish_end, nthreads):
        pe = w // fx.Int32(_pub_per_peer)
        ge = (w - pe * fx.Int32(_pub_per_peer)) * fx.Int32(_PUB_VEC)
        cnt = buffer_load(rs_hist, ge, vec_width=_PUB_VEC, dtype=T.i32)
        peer_count = fx.Int64(window.lsa_ptr(pe, C.off_count))
        buffer_store(cnt, _rsrc(peer_count), row_base + ge)
        if const_expr(_PROBE):
            # count[0][0][8] = iterations executed, [9] = the hist value seen for
            # ge 0: separates "loop never ran" from "loop ran but read zeros".
            P.atomic_add_global(fx.Int64(C.ptr_count) + fx.Int64(32), fx.Int32(1))
            if ge == fx.Int32(0):
                _c0 = (
                    vector.extract(cnt, static_position=[0])
                    if const_expr(_PUB_VEC > 1)
                    else cnt
                )
                buffer_store(_c0 + fx.Int32(5000), _rsrc(C.ptr_count), 9)
    P.waitcnt_stores()
    workgroup_barrier()
    # System scope, and it costs nothing to leave it there: taking the counts out
    # at SCOPE_SYS so this could drop to an agent release measured neutral. The
    # payload's identical change was worth 54 us only because 129 CTAs were each
    # paying for a write-back; one CTA paying twice is under a microsecond.
    P.fence_system_release()
    _emit_tstamp(C, TS_PLAN_PUB, t0, tid, True)
    if const_expr(_PLAN_PARTS & 2):
      if tid < fx.Int32(C.npes):
        # Rotate the peer order by rank so all ranks do not hammer rank 0 first.
        peer = (tid + fx.Int32(C.rank)) % fx.Int32(C.npes)
        done = fx.Int64(window.lsa_ptr(peer, C.off_count_done))
        P.atomic_add_global(done + fx.Int64(C.rank) * fx.Int64(4), fx.Int32(1))
    workgroup_barrier()

    if mid is not None:
        mid()
    _emit_tstamp(C, TS_PLAN_ORDER, t0, tid, True)

    # Wait for every source's counts, then make them visible to this CTA.
    if const_expr(_PLAN_PARTS & 2):
      if tid < fx.Int32(C.npes):
        P.spin_until_ge_i32_backoff(
            fx.Int64(C.ptr_count_done) + fx.Int64(tid) * fx.Int64(4), epoch
        )
    workgroup_barrier()
    P.fence_system_acquire()
    _emit_tstamp(C, TS_PLAN_GATHER, t0, tid, True)

    rs_count = _rsrc(C.ptr_count)
    parity_base = parity * fx.Int32(C.npes * C.e_total)

    # Source side: this rank's first row within each expert == the exclusive
    # prefix over lower-numbered ranks. One thread per global expert.
    rs_my_base = _rsrc(C.ptr_my_base)
    base_end = fx.Int32(C.e_total) if const_expr(_PLAN_PARTS & 4) else tid
    for ge in range(tid, base_end, nthreads):
        acc = fx.Int32(0)
        for s in range_constexpr(C.rank):
            acc = acc + buffer_load(
                rs_count,
                parity_base + fx.Int32(s * C.e_total) + ge,
                vec_width=1,
                dtype=T.i32,
            )
        buffer_store(acc, rs_my_base, ge)

    # Destination side: total rows per local expert -> compacted tile schedule.
    # The compaction is not optional: the static grid is E_local*CAP/tile_m tiles
    # of which only a handful carry rows, and padding tiles are not free (they
    # still walk the N/K mainloop).
    lds_load, lds_store = make_lds_copy_ops(32)
    # Scratch for the per-expert tile counts, borrowed from the GEMM tile arena:
    # this CTA is the planner and does not touch a tile until it joins the work
    # pool, which is several barriers past the end of this function. epr+1 dwords
    # against an arena that is kilobytes wide.
    ntiles_lds = fx.index_cast(T.index, fx.ptrtoint(fx.get_dyn_shared()))

    rs_te = _rsrc(C.ptr_tile_expected)
    te_end = fx.Int32(C.epr) if const_expr(_PLAN_PARTS & 8) else tid
    for le in range(tid, te_end, nthreads):
        ge = fx.Int32(C.rank * C.epr) + le
        total = fx.Int32(0)
        for s in range_constexpr(C.npes):
            total = total + buffer_load(
                rs_count,
                parity_base + fx.Int32(s * C.e_total) + ge,
                vec_width=1,
                dtype=T.i32,
            )
        # A count above CAP cannot have landed; clamp so the schedule never points
        # at rows the producers were not allowed to write.
        total = arith.select(total > fx.Int32(C.cap), fx.Int32(C.cap), total)
        for t in range_constexpr(C.tiles_per_expert):
            rem = total - fx.Int32(t * C.tile_m)
            exp = arith.select(rem > fx.Int32(C.tile_m), fx.Int32(C.tile_m), rem)
            exp = arith.select(exp < fx.Int32(0), fx.Int32(0), exp)
            buffer_store(exp, rs_te, le * fx.Int32(C.tiles_per_expert) + fx.Int32(t))
        lds_store(
            ntiles_lds,
            le * fx.Int32(4),
            Vec.from_elements(
                [(total + fx.Int32(C.tile_m - 1)) // fx.Int32(C.tile_m)], fx.Int32
            ),
        )
    P.waitcnt_stores()
    workgroup_barrier()

    # Compaction. An expert's live tiles are exactly the first
    # ceil(total/tile_m) of its slots, so the compacted position of tile
    # (le, t) is the exclusive prefix of those counts plus t -- a scan over epr
    # elements, not a walk over every (expert, tile) slot. The walk was a chain
    # of dependent global loads on one thread and cost ~640 us/layer at
    # epr=96, more than the GEMM it was scheduling.
    # One thread, and it does not matter: the loop is constexpr-unrolled over
    # constant LDS addresses, so all epr loads issue together and the only serial
    # part is the add chain. Scanning it across the CTA in log2(epr) barriered
    # rounds instead measured the compact phase unchanged at 10.4 us.
    if tid == fx.Int32(0):
        acc = fx.Int32(0)
        for le in range_constexpr(C.epr if _PLAN_PARTS & 16 else 0):
            n = lds_load(ntiles_lds, arith.index(le * 4))[0]
            lds_store(
                ntiles_lds, arith.index(le * 4), Vec.from_elements([acc], fx.Int32)
            )
            acc = acc + n
        lds_store(
            ntiles_lds, arith.index(C.epr * 4), Vec.from_elements([acc], fx.Int32)
        )
        # Debug: publish an empty schedule so the whole stage-1 still runs but the
        # GEMM half has nothing to do -- separates protocol cost from GEMM cost.
        published = fx.Int32(0) if const_expr(_NO_GEMM) else acc * fx.Int32(C.tile_m)
        buffer_store(published, _rsrc(addr_num_valid), 0)
    workgroup_barrier()

    rs_trb = _rsrc(addr_trb)
    rs_eids = _rsrc(addr_eids)
    rs_tvd = _rsrc(addr_tvd)
    sched_end = (
        fx.Int32(C.epr * C.tiles_per_expert) if const_expr(_PLAN_PARTS & 16) else tid
    )
    for w in range(tid, sched_end, nthreads):
        le = w // fx.Int32(C.tiles_per_expert)
        t = w - le * fx.Int32(C.tiles_per_expert)
        base = lds_load(ntiles_lds, le * fx.Int32(4))[0]
        nxt = lds_load(ntiles_lds, (le + fx.Int32(1)) * fx.Int32(4))[0]
        if t < nxt - base:
            m = base + t
            buffer_store(le * fx.Int32(C.cap) + t * fx.Int32(C.tile_m), rs_trb, m)
            buffer_store(le, rs_eids, m)
            buffer_store(buffer_load(rs_te, w, vec_width=1, dtype=T.i32), rs_tvd, m)

    P.waitcnt_stores()
    workgroup_barrier()
    if tid == fx.Int32(0):
        P.fence_system_release()
        P.atomic_add_agent(fx.Int64(C.ptr_plan_ready), fx.Int32(1))
    # The release fence above covers the replicas too, so they can go out in
    # parallel; a CTA that sees any of them sees the whole plan.
    for r in range(tid, fx.Int32(_PLAN_FANOUT), nthreads):
        P.store_i32_system(plan_ready_addr(C, r), fx.Int32(0), epoch)


# --------------------------------------------------------------------------- #
# Phase 3: payload push + per-tile arrival publish (producer CTAs)
# --------------------------------------------------------------------------- #
def _emit_rowmap(C, addr_inp_idx, addr_inp_wts, role, tid, nthreads, cur_tok):
    """Write the GEMM2 combine map for every route this rank owns.

    A pass of its own, one thread per route, rather than a few bytes tacked onto
    each row of the payload push. Nothing downstream reads the map until GEMM2, a
    kernel away, so it has no reason to be on the payload's critical path, and
    here all 4096 producer threads issue their one store together. Inside the
    push only one lane per row would have an entry to store -- at most
    _PAY_ROWS_MAX of a warp's 32 -- and that measured 169 us against 160 for the
    dispatch half.

    Both stores go through a buffer descriptor keyed on a lane-varying
    ``dest_pe``, so the compiler cannot keep it in SGPRs and wraps each in a
    readfirstlane waterfall that reruns once per distinct peer in the wave, a
    full ``s_wait_loadcnt`` inside the serialized loop. Measured on its own that
    is the whole of this pass: 62 us of the dispatch half against 0.4 us with the
    peer forced to a constant. Two ways out both measure a wash end to end,
    because what the waterfall costs is latency and this pass only ever runs
    alongside another CTA's payload push, which absorbs it -- against the payload
    the pass is worth 11 us, not 62.

      * A constant-peer loop hoisted outside this one: 22 us alone, but it
        rescans the routes once per peer, and with the payload that is 152 us
        against 121. Latency the push hides; the rescan's bandwidth it does not.
      * Pointer-addressed stores, which take no descriptor and so no waterfall:
        51 us alone, 401 us end to end against 402.
    """
    if role > fx.Int32(0):
        window = cco.Window(fx.Int64(C.arena_handle))
        rs_idx = _rsrc(addr_inp_idx)
        rs_wts = _rsrc(addr_inp_wts)
        rs_base = _rsrc(C.ptr_my_base)
        rs_slot = _rsrc(C.ptr_route_slot)
        slot_stride = C.max_tok * C.topk
        work_limit = cur_tok * fx.Int32(C.topk)
        base = (role - fx.Int32(1)) * nthreads + tid
        stride = fx.Int32(C.n_producers) * nthreads
        # Raw route order, not the push's expert-sorted walk. Sorting would land
        # neighbouring lanes on neighbouring fixed-slot rows and coalesce the two
        # stores, but it gathers the four metadata loads that feed them; measured
        # a wash across four samples either way.
        for w in range(base, work_limit, stride):
            src_tok = w // fx.Int32(C.topk)
            k_slot = w - src_tok * fx.Int32(C.topk)
            ge = buffer_load(rs_idx, w, vec_width=1, dtype=T.i32)
            dest_pe = ge // fx.Int32(C.epr)
            row_in_expert = buffer_load(
                rs_base, ge, vec_width=1, dtype=T.i32
            ) + buffer_load(rs_slot, w, vec_width=1, dtype=T.i32)
            if row_in_expert < fx.Int32(C.cap):
                dest_row = (ge - dest_pe * fx.Int32(C.epr)) * fx.Int32(
                    C.cap
                ) + row_in_expert
                wt = buffer_load(rs_wts, w, vec_width=1, dtype=T.f32)
                # {origin slot, route weight} as one 8B-aligned dwordx2, at the
                # same fixed-slot row GEMM2 derives from tile_row_base.
                buffer_store(
                    vector.from_elements(
                        T.vec(2, T.i32),
                        [
                            fx.Int32(C.rank * slot_stride)
                            + src_tok * fx.Int32(C.topk)
                            + k_slot,
                            arith.bitcast(T.i32, wt),
                        ],
                    ),
                    _rsrc(fx.Int64(window.lsa_ptr(dest_pe, C.off_pg_rowmap))),
                    dest_row * fx.Int32(2),
                )
                buffer_store(
                    fx.Int32(C.rank * C.max_tok) + src_tok,
                    _rsrc(window.lsa_ptr(dest_pe, C.off_tis)),
                    dest_row,
                )


def _emit_pay_meta(C, rs_order, rs_idx, rs_base, rs_slot, run_base, work_limit,
                   n_pay_rows):
    """Destination metadata for one run's rows: where each route lands.

    Integer loads only -- no payload traffic -- which is what makes it cheap
    enough for :func:`_emit_payload` to recompute rather than carry across its
    two passes.
    """
    src_toks, dest_pes, local_exps, rows_in_exp, lives, dest_rows = (
        [], [], [], [], [], [])
    for i in range_constexpr(n_pay_rows):
        pos = run_base + fx.Int32(i)
        # Expert-sorted, so a run is usually one expert's contiguous rows on
        # one peer, which is the shape the fabric wants.
        #
        # It does NOT make the destination's early tiles complete early, which
        # is what the ordering looks like it should buy. Every warp owns one run
        # and they all push at once through a fabric that is already saturated,
        # so they progress at the same rate and finish together: instrumenting
        # the first and last per-expert completion put them three quarters of
        # the way into the push and at its very end, with nothing at all ready
        # before that. Any consumer gate, however fine-grained, waits that long,
        # which is why the arrival counters below buy no overlap.
        work_idx = (
            buffer_load(rs_order, pos, vec_width=1, dtype=T.i32)
            if const_expr(ROUTE_ORDER)
            else pos
        )
        src_tok = work_idx // fx.Int32(C.topk)
        ge = buffer_load(rs_idx, work_idx, vec_width=1, dtype=T.i32)
        dest_pe = ge // fx.Int32(C.epr)
        local_expert = ge - dest_pe * fx.Int32(C.epr)
        row_in_expert = buffer_load(
            rs_base, ge, vec_width=1, dtype=T.i32
        ) + buffer_load(rs_slot, work_idx, vec_width=1, dtype=T.i32)
        # A run is indivisible, so the last one of the push runs off the end
        # of the route list. Those slots load in bounds but stale, and CAP is
        # how they are retired: it is already the overflow row's answer --
        # the planner clamped the schedule to match, so such a row is simply
        # dropped on both sides -- and it kills every store below.
        if const_expr(i > 0):
            row_in_expert = arith.select(
                pos < work_limit, row_in_expert, fx.Int32(C.cap)
            )
        src_toks.append(src_tok)
        dest_pes.append(dest_pe)
        local_exps.append(local_expert)
        rows_in_exp.append(row_in_expert)
        lives.append(row_in_expert < fx.Int32(C.cap))
        dest_rows.append(local_expert * fx.Int32(C.cap) + row_in_expert)
    return src_toks, dest_pes, local_exps, rows_in_exp, lives, dest_rows


def _emit_payload(
    C,
    addr_inp_tok,
    addr_inp_idx,
    addr_inp_scales,
    role,
    tid,
    nthreads,
    cur_tok,
    n_pay_rows,
):
    """Push this CTA's routes, then publish their rows' arrival.

    Row allocation is entirely deterministic here -- ``my_base[ge] + route_slot``
    -- unlike the non-overlap path's remote ``atomic_add(pg_running)``. That is
    the whole reason the destination can know a tile's expected row count before
    the payload shows up, and hence start GEMM1 on complete tiles while later ones
    are still in flight.

    The push is two passes over the same runs with one release between them,
    rather than a copy-then-publish per run. A publish has to drain the copies it
    covers and then fence at system scope, and the fence is the expensive half:
    at one per run per warp it ran 512 times a layer at bs=256 and cost 330 us of
    the 455 the whole dispatch half took -- more than five times the 55 us of
    token payload it was ordering. Hoisting it makes it once per CTA. The second
    pass recomputes each row's destination instead of carrying it in registers or
    LDS across the first; that is a handful of integer loads against a 7 KB row
    copy, and it keeps the run width free of a live-range budget.

    Ordering still holds. A CTA only ever signals arrival for rows it pushed
    itself, so its own drain and its own release cover exactly the writes its
    atomics announce.

    The GEMM2 combine map stays in :func:`_emit_rowmap`, a pass of its own, even
    though hoisting the drain removed the round trip that originally justified
    it. Folding it in here measured 169 us against 160 for the dispatch half: a
    run holds at most _PAY_ROWS_MAX rows, so only that many lanes of a warp have
    a map entry to store, while the separate pass spreads one store over every
    producer thread. It is thread count, not the drain, that keeps it out.
    """
    window = cco.Window(fx.Int64(C.arena_handle))
    lane = tid & fx.Int32(LANE_MASK)
    warp = tid >> fx.Int32(LOG2_WAVE)
    warps_per_cta = nthreads // WAVE
    # Role 0 is the planner and owns no producer slice. Without this clamp its
    # prod_id would be -1, the work index would start negative, and the very
    # first route load would read below the buffer.
    prod_id = arith.select(role > fx.Int32(0), role - fx.Int32(1), fx.Int32(0))
    global_warp = prod_id * warps_per_cta + warp
    global_warps = fx.Int32(C.n_producers) * warps_per_cta
    # The planner gets an empty work range rather than a branch around the whole
    # body: no barrier lives in here, so an empty loop is the cheaper opt-out.
    work_limit = arith.select(
        role > fx.Int32(0), cur_tok * fx.Int32(C.topk), fx.Int32(0)
    )

    rs_idx = _rsrc(addr_inp_idx)
    rs_base = _rsrc(C.ptr_my_base)
    rs_slot = _rsrc(C.ptr_route_slot)
    rs_order = _rsrc(C.ptr_route_order)

    run_stride = global_warps * fx.Int32(n_pay_rows)
    run_start = global_warp * fx.Int32(n_pay_rows)
    _pay_copies(
        C, window, addr_inp_tok, addr_inp_scales, rs_order, rs_idx, rs_base,
        rs_slot, lane, run_start, work_limit, run_stride, work_limit, n_pay_rows,
    )
    # One drain and one release for everything this CTA pushed. They must cover
    # payload, scale, rowmap and tis, and the release is what the whole two-pass
    # split exists to amortize: it is a system-scope fence, and its cost is scope,
    # not extent, so covering a CTA's entire push costs what covering one run did.
    # The barrier below it is what makes it a CTA-wide guarantee -- every warp has
    # drained before the one thread that fences runs.
    #
    # Per CTA and not once for the whole dispatch set, even though the release
    # lowers to ``global_wb``, a device-wide L2 write-back that one CTA could take
    # on everyone's behalf once they had all drained. These write-backs do
    # serialize in the cache controller and they are not cheap -- deleting them
    # takes the dispatch half from 162 us to 125 at 129 producers and 151 to 104
    # at 224 -- but the CTA barrier needed to make one of them sound costs more
    # than they do: the same measurement with the release hoisted onto a barrier
    # runs 201 us at 129 producers and 230 at 224. Arrival there is an atomic on
    # one line, so the barrier gets worse exactly as the producer count grows,
    # which is the direction that would have made hoisting worth it.
    #
    # Splitting this into several publishes so the low tiles could open while the
    # high ones were still in flight cannot work from inside a CTA, and not just
    # because it measured flat: pay_rows is chosen to give each warp exactly one
    # run, so a CTA's push is a single round and there is no seam in it to cut.
    # The tiles all open at once for a different reason anyway -- the producers
    # are symmetric and share a saturated fabric, so they all finish together, and
    # a per-CTA schedule cannot stagger that. Only a grid-wide one could.
    P.waitcnt_stores()
    workgroup_barrier()
    if tid == fx.Int32(0):
        # A system release over all address spaces. The narrower one-as scope
        # measured identical, and drain-only is wrong for stores left at default
        # scope -- it is only sound once PUB_SCOPE has taken them to SCOPE_SYS,
        # which is what makes them visible on retirement rather than at a flush.
        if const_expr(PUB_FENCE >= 2):
            P.fence_release_all()
        elif const_expr(PUB_FENCE == 1):
            P.fence_system_release()
    workgroup_barrier()
    if const_expr(_PARTS & 8):
        _pay_arrivals(
            C, window, rs_order, rs_idx, rs_base, rs_slot, lane,
            run_start, work_limit, run_stride, work_limit, n_pay_rows,
        )

    if const_expr(_BULK_SYNC):
        # Debug: collapse the pipeline to "dispatch, then GEMM1" so a wrong
        # result can be blamed on the payload rather than on the tile gating.
        P.waitcnt_stores()
        workgroup_barrier()
        if tid == fx.Int32(0):
            P.fence_system_release()
            P.atomic_add_global(
                fx.Int64(C.ptr_bar) + fx.Int64(_SYNC_SLOT * 4), fx.Int32(1)
            )


def _emit_pay_copies(
    C, window, addr_inp_tok, addr_inp_scales, rs_order, rs_idx, rs_base, rs_slot,
    lane, run_first, run_end, run_stride, work_limit, n_pay_rows,
):
    """Pass 1 of one stage: the copies, with nothing ordering them."""
    for run_base in range(run_first, run_end, run_stride):
        src_toks, dest_pes, local_exps, rows_in_exp, lives, dest_rows = _pay_meta(
            C, rs_order, rs_idx, rs_base, rs_slot, run_base, work_limit, n_pay_rows
        )

        for i in range_constexpr(n_pay_rows):
            src_tok = src_toks[i]
            dest_pe = dest_pes[i]
            local_expert = local_exps[i]
            row_in_expert = rows_in_exp[i]
            live = lives[i]
            dest_row = dest_rows[i]

            # e8m0 scale, permuted into the GEMM's WMMA layout as it is scattered
            # so the receiver needs no repack pass. All four bytes of a source
            # dword share one mx block and land in one destination dword.
            if const_expr(_PARTS & 2):
              if live:
                rows_per_tile = fx.Int32(C.wmma_rep * 16)
                scale_tile = row_in_expert // rows_per_tile
                row_in_tile = row_in_expert - scale_tile * rows_per_tile
                wmma_row = row_in_tile // fx.Int32(16)
                out_row = scale_tile * fx.Int32(16) + (
                    row_in_tile - wmma_row * fx.Int32(16)
                )
                dst_base = (
                    local_expert * fx.Int32(C.cap * C.scale_num_i32)
                    + out_row * fx.Int32(C.scale_num_i32 * C.wmma_rep)
                    + wmma_row
                )
                rs_in_sc = _rsrc(addr_inp_scales)
                rs_peer_sc = _rsrc(
                    fx.Int64(window.lsa_ptr(dest_pe, C.off_out_scales))
                )
                for k_off in range(
                    lane * fx.Int32(C.scale_vec),
                    fx.Int32(C.scale_num_i32),
                    fx.Int32(WAVE * C.scale_vec),
                ):
                    sc = buffer_load(
                        rs_in_sc,
                        src_tok * fx.Int32(C.scale_num_i32) + k_off,
                        vec_width=C.scale_vec,
                        dtype=T.i32,
                    )
                    dst_off = dst_base + k_off * fx.Int32(C.wmma_rep)
                    for j in range_constexpr(C.scale_vec):
                        elem = (
                            vector.extract(sc, static_position=[j])
                            if const_expr(C.scale_vec > 1)
                            else sc
                        )
                        buffer_store(
                            elem,
                            rs_peer_sc,
                            dst_off + fx.Int32(j * C.wmma_rep),
                            cache_modifier=PUB_SCOPE,
                        )

            # fp8 token payload: each lane owns 16 B, _NSTREAMS vec4 streams in
            # flight.
            if const_expr(_PARTS & 4):
             peer_tok = fx.Int64(window.lsa_ptr(dest_pe, C.off_disp_out))
             rs_src = _rsrc(
                 _addr(addr_inp_tok) + fx.Int64(src_tok) * fx.Int64(C.token_nbytes)
             )
             rs_dst = _rsrc(peer_tok + fx.Int64(dest_row) * fx.Int64(C.token_nbytes))
             lane_i32_off = lane * fx.Int32(4)
             safe_end = (C.n_i32 // _MAIN_STRIDE_I32) * _MAIN_STRIDE_I32
             if const_expr(C.n_i32 >= _MAIN_STRIDE_I32 and safe_end > 0):
                 end_main = arith.select(live, fx.Int32(safe_end), lane_i32_off)
                 for chunk in range(lane_i32_off, end_main, _MAIN_STRIDE_I32):
                     vecs = [
                         buffer_load(
                             rs_src, chunk + k * _LANE_STRIDE_I32, vec_width=4, dtype=T.i32
                         )
                         for k in range_constexpr(_NSTREAMS)
                     ]
                     for k in range_constexpr(_NSTREAMS):
                         buffer_store(
                             vecs[k],
                             rs_dst,
                             chunk + k * _LANE_STRIDE_I32,
                             cache_modifier=PUB_SCOPE,
                         )
             if const_expr(safe_end < C.n_i32):
                 end_tail = arith.select(live, fx.Int32(C.n_i32), lane_i32_off)
                 for chunk in range(
                     lane_i32_off + fx.Int32(safe_end), end_tail, _LANE_STRIDE_I32
                 ):
                     v = buffer_load(rs_src, chunk, vec_width=4, dtype=T.i32)
                     buffer_store(v, rs_dst, chunk, cache_modifier=PUB_SCOPE)
             elif const_expr(C.n_i32 < _MAIN_STRIDE_I32):
                 end_small = arith.select(live, fx.Int32(C.n_i32), lane_i32_off)
                 for chunk in range(lane_i32_off, end_small, _LANE_STRIDE_I32):
                     v = buffer_load(rs_src, chunk, vec_width=4, dtype=T.i32)
                     buffer_store(v, rs_dst, chunk, cache_modifier=PUB_SCOPE)


def _emit_pay_arrivals(
    C, window, rs_order, rs_idx, rs_base, rs_slot, lane, run_first, run_end,
    run_stride, work_limit, n_pay_rows,
):
    """Pass 2 of one stage: the arrivals, now that every row they announce is
    visible. The counter is additive, so the consumer's "tile is complete" test is
    arrived == planned rather than a boolean flag that could be set while a
    sibling row is still in flight -- which is also what lets a stage publish on
    its own, with no agreement between producers about where a stage ends."""
    for run_base in range(run_first, run_end, run_stride):
        _, dest_pes, local_exps, rows_in_exp, lives, _ = _pay_meta(
            C, rs_order, rs_idx, rs_base, rs_slot, run_base, work_limit, n_pay_rows
        )
        if lane == fx.Int32(0):
            # One atomic per tile the run touches, not per row. The run is a
            # contiguous window of the destination-sorted order, so its rows carry
            # a non-decreasing key and usually a single one: counting the group and
            # flushing it at its last row turns n_pay_rows dependent fabric round
            # trips from one lane into (almost always) one. The counter stays
            # additive, so the consumer's arrived == planned test is unchanged.
            keys, dests = [], []
            for i in range_constexpr(n_pay_rows):
                tile_idx = rows_in_exp[i] // fx.Int32(C.tile_m)
                keys.append(local_exps[i] * fx.Int32(C.tiles_per_expert) + tile_idx)
                dests.append(dest_pes[i])
            def _joins(a, b):
                """Row ``a`` continues row ``b``'s group: same live tile, same peer.

                Constant-false under :data:`PUB_COALESCE` = 0, which puts every
                row back on its own atomic.
                """
                if const_expr(not PUB_COALESCE):
                    return fx.Boolean(False)
                return arith.andi(
                    lives[a],
                    arith.andi(keys[a] == keys[b], dests[a] == dests[b]),
                )

            group = fx.Int32(0)
            for i in range_constexpr(n_pay_rows):
                one = arith.select(lives[i], fx.Int32(1), fx.Int32(0))
                if const_expr(i == 0):
                    group = one
                else:
                    group = arith.select(_joins(i, i - 1), group + fx.Int32(1), one)
                if const_expr(i == n_pay_rows - 1):
                    flush = lives[i]
                else:
                    flush = arith.andi(
                        lives[i], arith.xori(_joins(i + 1, i), fx.Boolean(True))
                    )
                if flush:
                    peer_arr = fx.Int64(
                        window.lsa_ptr(dests[i], C.off_tile_arrived)
                    ) + fx.Int64(keys[i]) * fx.Int64(4)
                    P.atomic_add_global(peer_arr, group)


# --------------------------------------------------------------------------- #
# Phase 4 helpers: consumer-side waits
# --------------------------------------------------------------------------- #
def _emit_wait_plan(C, epoch, tid, bid=None):
    """Every CTA blocks here until the planner's tile schedule is visible."""
    if tid == fx.Int32(0):
        if const_expr(_PLAN_FANOUT > 1) and bid is not None:
            P.spin_until_ge_i32_backoff(
                plan_ready_addr(C, bid % fx.Int32(_PLAN_FANOUT)), epoch, agent=True
            )
        else:
            P.spin_until_ge_i32_backoff(
                fx.Int64(C.ptr_plan_ready), epoch, agent=True
            )
    workgroup_barrier()
    P.fence_acquire_all()


def _emit_wait_all_payload(C, epoch, tid):
    """Debug counterpart of :func:`_emit_wait_all_payload`'s publisher: block a
    consumer until every local producer CTA has finished pushing."""
    if tid == fx.Int32(0):
        P.spin_until_ge_i32_backoff(
            fx.Int64(C.ptr_bar) + fx.Int64(_SYNC_SLOT * 4),
            epoch * fx.Int32(C.dispatch_ctas),
        )
    workgroup_barrier()
    P.fence_acquire_all()


def _emit_take_work(C, shard, tid, lds_idx):
    """Pull this CTA's next run of GEMM1 work items and broadcast its base.

    One atomic hands out :func:`pull_chunk` *contiguous* ids, and the caller
    walks them; the id space is M-major, so a run usually lies inside one M-tile
    and the caller's cached arrival gate opens once for all of it.

    Pull, not a grid-stride: the grid is superscribed for GEMM1 occupancy, so
    binding work item w to the CTA with role w would park most of the schedule
    on CTAs that are not resident yet -- behind resident CTAs that are spinning
    on a tile those very CTAs would have filled. Pulling means only a running
    CTA ever holds work.

    Each CTA pulls from one shard, and shard s only ever hands out the chunks
    ``s, s+SHARDS, ...``; every shard must therefore have at least one CTA, which
    holds because shards are assigned by ticket over a grid far wider than
    ``WORK_SHARDS``.

    ``lds_idx`` is scratch the caller owns for the length of this call; the
    barriers on both sides are what let it be the GEMM's own tile arena. The
    leading one is not redundant with the trailing one of the previous call: in
    between sits a whole tile, whose epilogue is still reading that arena in the
    slower waves when wave 0 arrives back here.
    """
    lds_load, lds_store = make_lds_copy_ops(32)
    workgroup_barrier()
    if tid == fx.Int32(0):
        got = fx.Int32(P.atomic_add_agent(work_head_addr(C, shard), fx.Int32(1)))
        chunk = shard + got * fx.Int32(WORK_SHARDS)
        lds_store(
            lds_idx,
            arith.index(0),
            Vec.from_elements([chunk * fx.Int32(pull_chunk(C))], fx.Int32),
        )
    workgroup_barrier()
    work = lds_load(lds_idx, arith.index(0))[0]
    workgroup_barrier()
    return work


def _emit_tile_acquire(tid):
    """Close a run of :func:`_emit_spin_tile` waits with one acquire.

    Split out from the wait because the two have very different costs and very
    different natural rates. A spin is a load off a counter thread 0 already has
    the address of; the acquire is a cache invalidate that also throws away the
    B tile the GEMM is about to reuse. Taking one per work item is what forced
    the schedule to hand out coarse items, so the caller takes one per pull
    instead and spins on every tile the pull covers underneath it.
    """
    workgroup_barrier()
    if const_expr(_TILE_FENCE >= 2):
        P.fence_acquire_all()
    elif const_expr(_TILE_FENCE == 1):
        P.fence_system_acquire()
    else:
        P.fence_agent_acquire()


def _emit_spin_tile(C, row_base, tid):
    """Wait for one GEMM1 M-tile's planned rows, with no fence of its own.

    ``row_base`` is the fixed-slot row ``local_expert*CAP + tile*tile_m``, which is
    exactly what the tile schedule already carries, so the arrival key needs no
    extra per-tile metadata array.

    Thread 0 alone spins and there is no barrier here, so the caller must run
    :func:`_emit_tile_acquire` before any wave reads the rows.
    """
    le = row_base // fx.Int32(C.cap)
    tile_idx = (row_base - le * fx.Int32(C.cap)) // fx.Int32(C.tile_m)
    key = le * fx.Int32(C.tiles_per_expert) + tile_idx
    if tid == fx.Int32(0):
        # Volatile load, not buffer_load: the planner rewrites this every launch
        # and a cached read of last launch's (larger) count would wait for rows
        # that are never coming.
        want = P.load_i32_acquire(
            fx.Int64(C.ptr_tile_expected) + fx.Int64(key) * fx.Int64(4)
        )
        if const_expr(TILE_GATE >= 2):
            # Debug: probe the counter once and run on regardless. Keeps the
            # expected load, the arrival load and the fence below, so the
            # difference against the real gate is the waiting alone, and the
            # difference against TILE_GATE=0 is what the gate costs to evaluate.
            P.spin_until_ge_i32_bounded(
                fx.Int64(C.ptr_tile_arrived) + fx.Int64(key) * fx.Int64(4), want, 1
            )
        elif const_expr(_WAIT_MAX > 0):
            got = P.spin_until_ge_i32_bounded(
                fx.Int64(C.ptr_tile_arrived) + fx.Int64(key) * fx.Int64(4),
                want,
                _WAIT_MAX,
            )
            # Timed-out tiles land in the tail of the count matrix (only the
            # local experts' columns are ever published, so it is dead space).
            if got < want:
                dbg = fx.Int64(C.ptr_bar) + fx.Int64(_DBG_SLOT * 4)
                P.atomic_add_global(dbg, fx.Int32(1))
                P.store_i32_system(dbg, fx.Int32(1), key)
                P.store_i32_system(dbg, fx.Int32(2), want)
                P.store_i32_system(dbg, fx.Int32(3), got)
        else:
            P.spin_until_ge_i32_backoff(
                fx.Int64(C.ptr_tile_arrived) + fx.Int64(key) * fx.Int64(4), want
            )


emit_tstamp = ASTRewriter.transform(_emit_tstamp)
emit_tstamp_max = ASTRewriter.transform(_emit_tstamp_max)
emit_tstamp_acc = ASTRewriter.transform(_emit_tstamp_acc)
emit_ticket = ASTRewriter.transform(_emit_ticket)
emit_zero_and_count = ASTRewriter.transform(_emit_zero_and_count)
_prefix_owner = ASTRewriter.transform(_emit_prefix_owner)
_route_order_body = ASTRewriter.transform(_emit_route_order_body)
emit_route_order = ASTRewriter.transform(_emit_route_order)
emit_plan = ASTRewriter.transform(_emit_plan)
emit_rowmap = ASTRewriter.transform(_emit_rowmap)
_pay_meta = ASTRewriter.transform(_emit_pay_meta)
_pay_copies = ASTRewriter.transform(_emit_pay_copies)
_pay_arrivals = ASTRewriter.transform(_emit_pay_arrivals)
emit_payload = ASTRewriter.transform(_emit_payload)
emit_wait_plan = ASTRewriter.transform(_emit_wait_plan)
emit_wait_all_payload = ASTRewriter.transform(_emit_wait_all_payload)
emit_take_work = ASTRewriter.transform(_emit_take_work)
emit_spin_tile = ASTRewriter.transform(_emit_spin_tile)
emit_tile_acquire = ASTRewriter.transform(_emit_tile_acquire)

# SPDX-License-Identifier: MIT

"""FlyDSL decode TopK-per-row kernel (tiered persistent multi-block radix-select)

Computes an unordered Top-K index set per decode row, fusing a single-workgroup and
a multi-block radix-select into one persistent launch grid=(blocks_per_row, num_rows)
that picks a per-row strategy by valid length -- which is what a decode batch needs,
since its sequences differ in length. Each row derives how many of its blocks_per_row
workgroups cooperate (active_parts); the rest return immediately.

Inputs/outputs:
  - logits: fp32, logical shape (num_rows, L), strides (stride0, stride1) with
    stride1 == 1 (contiguous within a row).
  - seq_lens: int32 causal lengths per sequence; row r scores sequence r // next_n at
    decode slot r % next_n, valid length seq_len - next_n + slot + 1.
  - indices: flattened int32 output with shape (num_rows, top_k); each row writes its
    unordered Top-K index set. A row with fewer than top_k valid entries is
    identity-filled and padded with -1.
  - workspace: row-major int32 scratch sized by topk_workspace_slots(num_rows,
    bits_per_pass). The multi-block tiers merge per-block LDS histograms into its
    pass-private global histograms over an inter-workgroup acquire/release barrier and
    coordinate through its counters; the single-workgroup tier never touches it.

Paths (per row, by valid length row_len):
  - short (row_len <= short_max): active_parts = 1; part 0 runs the whole radix-select
    in one workgroup — LDS-only histograms, no inter-workgroup barrier, no workspace
    round-trip.
  - mid (short_max < row_len <= mid_max): active_parts = min(blocks_per_row, mid_cap).
  - long (row_len > mid_max): active_parts = min(blocks_per_row, long_cap).

Constraints:
  - logits are fp32; the order-preserving radix key twiddle is fp32-specific.
  - bits_per_pass is 10 or 11; the short tier requires 11 bits (2048-bin LDS histogram).
  - BLOCK_THREADS is fixed at 1024 (wave64); the histogram/scan layout and the
    occupancy deadlock guard rely on it.
  - workspace must be zeroed before any launch that enters a multi-block tier; its
    counters and histograms accumulate from zero (needs_workspace_zero reports when).
  - The row barrier spins (s_sleep), so a row's blocks_per_row workgroups must be
    co-resident. This is a regular launch, not hipLaunchCooperativeKernel, and is safe
    only because the grid is flattened x-fastest: a row's parts launch contiguously and
    drain in order, which is scheduler launch order rather than a cooperative
    guarantee. The wrapper's deadlock guard keeps num_rows * blocks_per_row co-resident,
    forcing larger batches onto the barrier-free short tier.
"""

import math
from functools import cache
from typing import Any, Literal

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import (
    arith,
    as_ir_value,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
)
from flydsl.expr.typing import T

# buffer_ops and vector come from aiter's own shims, not flydsl.expr: the flydsl
# cleanup in #4501 dropped those from the stable interface.
from aiter.ops.flydsl.kernels import buffer_ops, vector

# HW max block size; also assumed by the bucket scan (2 bins/thread -> 2048 bins)
# and the occupancy=2 deadlock guard. Changing it breaks both.
BLOCK_THREADS = 1024
WARP_SIZE = 64
LOAD_VEC = 4
# log2(LOAD_VEC); LOAD_VEC must be a power of two. vec_blocks = ceil(row_len/LOAD_VEC)
# is a right shift, so the shift amount must track LOAD_VEC, not a hardcoded width.
LOAD_VEC_LOG2 = LOAD_VEC.bit_length() - 1
# Default histogram-scan staging (one of 1/2/4/8)
SCAN_STAGES = 2
# Load staging width for ordered emit (matches unordered last-pass unroll).
ORDERED_STAGES = 4
# vec-loads a thread takes per compact-fill tile. The fill's block scan costs two
# barriers and drains the loads in flight, so it wants to be paid as rarely as the
# register budget allows; each step holds compact_fill_vecs * LOAD_VEC keys live.
COMPACT_FILL_VECS = 4

# 128B-spaced inter-workgroup counter groups (32 int32 == 128B each), kept in the int32 workspace.
COUNTER_STRIDE = 32
# Bound on the certificate's re-read loop, so a bug cannot hang the GPU. It is a
# liveness guard, not a schedule: the loop exits when the bins sum to the row's
# live length, which every part's atomics make inevitable. Sized far past any
# real wait -- the loop is a scan of 2048 bins, so this is minutes.
CERTIFICATE_MAX_SPINS = 4194304
# Group 0, indexed by part: how many candidates that part appended to its slice.
COUNTER_COMPACT_COUNT = 0 * COUNTER_STRIDE
# Group 1, indexed by pass: the live count the parts have added to that pass's
# histogram. The certificate's proof is the bins summing to the row's live length,
# and this is that sum published directly instead of recomputed from 2048 bins on
# every poll. Per pass rather than one slot, so no pass has to reset it.
COUNTER_CERT_TOTAL = 1 * COUNTER_STRIDE
# Diagnostic, in the same group's spare slots since it holds three of thirty-two.
# Answers one question: when the re-read loop ends at the cap, had the histogram
# already arrived? A waiter that is merely slow leaves a total short of the live
# length; one that cannot see what landed leaves the same short total with the
# bins full. Written only under certificate_trace.
CERT_TRACE_SPINS = COUNTER_CERT_TOTAL + 8  # most spins any part reached, atomic max
CERT_TRACE_CAPPED = COUNTER_CERT_TOTAL + 9  # how many parts ended at the cap
CERT_TRACE_SEEN = COUNTER_CERT_TOTAL + 10  # the total such a part last read
CERT_TRACE_WANT = COUNTER_CERT_TOTAL + 11  # the live length it was waiting for
CERT_TRACE_TRUTH = COUNTER_CERT_TOTAL + 12  # the same bins summed after it gave up
CERT_TRACE_ARRIVED = COUNTER_CERT_TOTAL + 13  # parts that reached a certified merge
CERT_TRACE_PASS = COUNTER_CERT_TOTAL + 14  # which pass gave up, 1-based so 0 is none
# Per part, for the fill pass only, so the two sides of the disagreement can be
# compared: the digit each part settled at pass 0 and the count it is waiting for.
# If the parts settled different digits they are histogramming different sets and
# no total could satisfy both; if they agree, the count is what is wrong.
CERT_TRACE_BITS = COUNTER_CERT_TOTAL + 16  # + part
CERT_TRACE_WANT_P = COUNTER_CERT_TOTAL + 24  # + part
COUNTER_SLOTS = 8 * COUNTER_STRIDE
COUNTER_ARRIVALS = 2 * COUNTER_STRIDE
COUNTER_OUT_FRONT = 3 * COUNTER_STRIDE
COUNTER_OUT_BACK = 4 * COUNTER_STRIDE
COUNTER_PASS_DONE = 5 * COUNTER_STRIDE
# Per-workgroup selected/tied counts for ordered emit (reserved for both modes).
COUNTER_ORDERED_ABOVE = 6 * COUNTER_STRIDE
COUNTER_ORDERED_EQUAL = 7 * COUNTER_STRIDE

SMEM_META_K = 0
SMEM_META_LEN = 1
SMEM_META_THRESHOLD = 2
SMEM_META_ABOVE = 3
# Whether the passes behind the fill can settle their digit without a merge. Lives
# in LDS rather than a traced value: it is decided inside one region and read from
# several nested ones, and an SSA value would not dominate those uses.
SMEM_META_COLLAPSED = 4
# The live length a certified merge has to see the histogram sum to. Written once
# per pass by the region that knows the count, read by the re-read loop, so it has
# the same domination problem as the collapse flag and the same answer.
SMEM_META_TOTAL = 5
# Where a part accumulates its own contribution to the pass's histogram while it
# flushes, before one thread publishes it. Shares a slot with the short tier's
# front count, which is safe because a row takes one tier and the short tier has
# no parts to publish to.
SMEM_META_CONTRIB = 6

# Compact candidate buffer, one region per row after the histograms. Entry i is the
# pair (column, twiddled key) at 2*i, so the later passes read a candidate without
# touching the row. Slot 0 of the header holds the count the parts append into and
# slot 1 the overflow flag; the header is a full 128B group so the data stays aligned.
COMPACT_HDR_COUNT = 0
COMPACT_HDR_OVERFLOW = 1
COMPACT_HDR_SLOTS = COUNTER_STRIDE

# Short-tier one-workgroup metadata reuses the same 8-int LDS block after zeroing.
SMEM_META_SHORT_FIRST_ABOVE = 0
SMEM_META_SHORT_FIRST_THRESHOLD = 1
SMEM_META_SHORT_SECOND_ABOVE = 2
SMEM_META_SHORT_SECOND_THRESHOLD = 3
SMEM_META_SHORT_THIRD_ABOVE = 4
SMEM_META_SHORT_THIRD_THRESHOLD = 5
SMEM_META_SHORT_FRONT_COUNT = 6
SMEM_META_SHORT_BACK_COUNT = 7


# --- the host-side rule -----------------------------------------------------
#
# Everything below decides, from the batch shape alone, how wide to launch and
# which of the kernel's paths to run. It lives here rather than in a benchmark
# because every one of these numbers was measured, and a caller that picks them
# differently gets a different kernel than the one the measurements describe.
#
# `decode_compact_config` is the single entry point; the pieces are separate only
# so each can carry the reason it is what it is.

PARTS_CAP = 32
# A row shorter than this reads its candidates from the row rather than from the
# buffer. The floor cannot be fitted on its own, and the earlier one was: a
# compacted pass reads a slice of the candidate buffer and an uncompacted one a
# slice of the row, and those two do not want the same number of parts. Swept
# both states against the full width ladder, three reps, every multi-block cell
# with its own argmin in each state -- and once each state is allowed the width
# it wants, ten cells that used to compact prefer to run one to three rungs wider
# and read the row.
#
# What moved them is the certificate. Compaction's saving is a shorter re-read;
# its price is the fill, and the fill's barrier used to be one of four. Certifying
# the other passes makes that barrier a larger share of what a compacted row
# spends, so the saving has to be larger to cover it -- which it only is at 1M,
# and at 256K once there are 32 rows to spread the fill across.
COMPACT_MIN = ((32, 262144), (1, 1048576))

COMPACT_SLICE = 40960
"""How long a part's own slice has to be before compacting it pays, past 32 rows.

`COMPACT_MIN` is a floor on the row and was fitted where a row had six or seven
workgroups. Past 32 rows a row has at most four and usually one, and the floor
does not carry across: compaction saves each part two re-reads of its own slice,
so what it is worth follows the slice, seq/parts, and not the row. Keeping the
floor on the row left four cells at or behind the HIP one-block kernel -- the only
cells on the wide grid that were, 0.867-1.006x -- because a row alone on the
cooperating path was scanning 64K or 128K elements three times with one workgroup.

Measured at 64, 128, 256 and 512 rows by 48K, 64K, 96K, 128K and 192K, three reps,
spread within a cell under 0.5%: `seq >= COMPACT_SLICE * parts` calls all twenty,
and both crossovers land on it to the rung -- at two parts between 64K and 96K, at
four parts between 128K and 192K, which put the slice at 40960 from either side.
It is worth 1.038-1.498x on the fifteen cells where it turns compaction on, and
the four cells behind HIP come out 1.096-1.265x ahead of it.

That this is `SHORT_MAX_CAP` again is one hardware fact seen a third time: about
40K elements is as long a run as one workgroup will scan before it would rather
pay to avoid scanning it twice.
"""

# parts = sqrt(c2 * seq / (c1 * rows)) -- only the ratio decides a width. Fitted
# against a forced-width ladder over all 36 cells with `poll_then_acquire` on,
# which is what makes an extra participant cheap enough to want width at all.
PARTS_C1 = {False: 187.979e-3, True: 153.875e-3}
PARTS_C2 = {False: 1.87979e-3, True: 464.695e-6}

# Widths measured to be optimal: every cell of the grid that reaches the
# multi-block path. The ladder has fifteen rungs and was walked three times over
# all thirty cells; all thirty give the identical argmin in all three, over 1164
# runs with no incorrect result. The nine cells at 16 and 32 rows repeat what four
# earlier independent processes found. Seven entries at 1, 2 and 4 rows were then
# raised by two follow-up sweeps -- one that let the grid reach 32, one that moved
# the compaction floor -- each also three reps, and stable across them.
#
# Ten more entries moved when the table was refitted against the compaction
# decision rather than after it. A cell's best width depends on what a pass reads:
# compacted, a part reads its slice of the candidate buffer, and the buffer is
# small enough that more parts mostly buy more barrier. Reading the row instead,
# the same cell wants one to three rungs more. Those ten are exactly the cells
# `COMPACT_MIN` stopped compacting, and they are listed here at the width the
# uncompacted path wants, not the width the compacted one did.
#
# A table is used because a closed form provably cannot be right here. Two reasons,
# either of which is enough. First, 16x128K and 32x256K have the same seq/rows,
# 8192, and stable optima two rungs apart in opposite directions, so no
# parts = f(seq/rows) can serve both; refitting the ratio confirms it from the
# other side, fixing 32x256K and breaking 32x64K.
#
# Second, and larger: where the row is narrow enough to reach them, the good
# widths are 16 and 32, and the ladder steps down into them rather than curving.
# 4x256K runs 14:32.0us, 16:29.0us, 20:30.3us. 1x128K runs 30:24.9us, 31:25.0us,
# 32:23.0us. No smooth cost model produces a floor one rung wide, and the square
# root, landing at 14 to 31, pays 2-10% for missing it. Both are out of reach at
# 16 rows and wider, where the grid is 15 and 7, which is why those rows answer
# with a different shape -- and there the ladder really does curve.
#
# A cost model was tried before the table and lost even with its coefficients
# tuned on argmin agreement: its residual is 1.4us against minima 0.4-2us deep.
#
# 16x32K is here because the tier threshold moved it. It was 2 for as long as the
# cell ran on the short tier, where the entry is never read; the multi-block path
# wants 8. 32x32K is still 2 and still unread -- 32 rows keep the short tier until
# 33.5K -- so that entry describes nothing and should not be trusted as if it did.
#
# Off this grid, and for any k but 2048, the square root runs. It is a fallback.
MEASURED_PARTS = {
    (1, 32768): 8,
    (1, 65536): 16,
    (1, 131072): 32,
    (1, 262144): 32,
    (1, 1048576): 32,
    (2, 32768): 8,
    (2, 65536): 16,
    (2, 131072): 32,
    (2, 262144): 32,
    (2, 1048576): 32,
    (4, 32768): 8,
    (4, 65536): 16,
    (4, 131072): 16,
    (4, 262144): 24,
    (4, 1048576): 32,
    (8, 32768): 8,
    (8, 65536): 16,
    (8, 131072): 16,
    (8, 262144): 24,
    (8, 1048576): 24,
    (16, 32768): 8,
    (16, 65536): 8,
    (16, 131072): 12,
    (16, 262144): 14,
    (16, 1048576): 14,
    (32, 32768): 2,
    (32, 65536): 6,
    (32, 131072): 7,
    (32, 262144): 6,
    (32, 1048576): 7,
}
MEASURED_PARTS_K = 2048

WIDE_SPLIT_WORK = 28416
"""How much work a split has to take off a row's critical path to pay for itself.

Fitted at 64 and 128 rows, where the batch is wider than the launch and a row's
share of the grid is fixed at four and two workgroups. Splitting a row G ways
removes seq*(G-1)/G elements from its critical path and adds one barrier, so the
crossover should sit where that product is constant, and it does: measured
one-part-against-split at 32K, 36K, 40K, 48K and 64K, the crossover is between
36.9K and 41K at four parts, which puts the product at 28.5K, and between 49K and
64K at two parts, which puts it at 27.5K. One constant, ten cells, no exceptions:
1.048-1.182x to one part below it, 1.07-1.39x to the split above it.

That the number lands on SHORT_MAX_BASE is a coincidence worth naming rather than
relying on: both say a row barrier costs about what 28K elements cost, which is
the same hardware fact measured from two directions -- the tier threshold asks
when a barrier is worth taking at all, and this asks when a second one is.
"""


def decode_compact_grid(rows: int, want: int, seq: int | None = None) -> int:
    """The launch width, from the batch size and the width the row wants.

    Narrowing inside a grid is free when each row's cooperating workgroups still
    cover all eight XCDs and the launch fits the 256 CUs in one go. An odd width
    shifts every row's starting XCD by one, which keeps a narrowed row balanced
    while its parts stay contiguous -- the property the deadlock argument rests
    on, since a row's parts have to be co-resident to clear a spin barrier.

    That shift is only worth a workgroup when there is narrowing to balance. A row
    that wants the whole grid is not narrowing, so the grid stays at its full even
    width -- and the full width matters, because 32 parts is a sharply better
    place to be than 31. Measured at 1, 2 and 4 rows across both grids: taking the
    even 32 when the row wants it is worth 1.021-1.087x, and taking it when the
    row wants to narrow loses up to 1.1%.

    Past 256 rows the batch fills the wave by itself and there is no second
    workgroup to give a row, so a row runs alone. That is not only the available
    shape, it is the fast one: measured at 256 rows against two parts, one part
    wins by 1.66x at 32K, 1.41x at 256K and 1.12x at 1M, because a second part
    there buys a barrier and no extra CU. It is also the only shape that stays
    safe as the batch grows, since a single part satisfies the row barrier by
    arriving at it -- co-residency stops binding, which is what makes 512 rows no
    more dangerous than 256.

    Between 32 rows and that point the row has a share of the grid rather than a
    choice of width, and whether to use it at all is a question the tuned grid
    never asks: below `WIDE_SPLIT_WORK` of critical path removed, the share's
    barrier costs more than the parallelism it buys, and the row is better off
    alone. `seq` is optional only so that callers from before this rule existed
    keep working; without it a short row at 64 rows takes the split and loses up
    to 1.18x.
    """
    widest = min(32, 256 // max(1, rows))
    if widest < 2:
        return 1
    grid = widest if want >= widest else max(
        3, widest - 1 if widest % 2 == 0 else widest
    )
    if (
        seq is not None
        and rows > 32
        and grid > 1
        and seq * (grid - 1) <= WIDE_SPLIT_WORK * grid
    ):
        return 1
    return grid


def decode_compact_uses_buffer(rows: int, seq: int) -> bool:
    """`COMPACT_MIN` read as a floor on the row. The width table is fitted to it."""
    floor = next(f for at, f in COMPACT_MIN if rows >= at)
    return seq >= floor


def decode_compact_compacts(
    rows: int, seq: int, parts: int, short_max: int, ordered: bool = True
) -> bool:
    """Whether the passes behind the first read candidates back from the buffer.

    Two floors, because the grid has two regimes. Up to 32 rows a row chooses its
    width, and the floor is `COMPACT_MIN` on the row, fitted jointly with the width
    table and left exactly as it was. Past 32 rows a row takes what is left of one
    wave, so the floor moves onto the part's slice; see `COMPACT_SLICE`.

    A single-part row does not compact for the unordered emit, which is the one
    place the two emits want different answers. Both floors above were fitted at
    `ordered=True`, and up to 128 rows they suit the unordered emit too: at 16 rows
    by 1M the buffer is worth 1.28x to it, at 32 by 256K and 128 by 128K a little
    over 1.05x. At one part it inverts and the buffer costs the unordered emit
    1.18x -- 66.52 us against 56.20 at 256 rows by 65536, 134.93 against 114.66 at
    512 by 65536 -- while the ordered emit still gains 1.17x from keeping it. One
    part is also where the part's slice of the buffer stops being a share and
    becomes the whole of it, so a wide row overruns the cap, spills, and reaches
    the last pass with a buffer it cannot read; the ordered emit is paid back for
    the fill by the passes behind it, and the unordered emit is not.

    Only `parts` is consulted and not the width, because the eight cells this
    turns off are every compacting cell at 256 and 512 rows and no others, and all
    eight are single-part. A width floor would have to be fitted; this does not.

    The short tier keeps no candidate buffer, so a row that never reaches the
    cooperating path never compacts. That is already implied for `COMPACT_MIN`,
    whose lowest floor is six times any tier threshold, and has to be said for the
    slice rule, which at one part crosses below one.

    `decode_compact_want` deliberately still reads `decode_compact_uses_buffer`
    rather than this. The width table was fitted against that floor, so asking the
    slice rule there would move widths that no measurement has visited -- 64 rows
    by 192K would drop from four parts to three. Past 32 rows it makes no
    difference in any case, since `parts` is what the grid can give and not what
    the row asks for.
    """
    if seq <= short_max:
        return False
    if not ordered and parts <= 1:
        return False
    if rows > 32:
        return seq >= COMPACT_SLICE * parts
    return decode_compact_uses_buffer(rows, seq)


def decode_compact_want(rows: int, seq: int, k: int = MEASURED_PARTS_K) -> int:
    """The width the row wants, before the grid gets a say."""
    if k == MEASURED_PARTS_K and (rows, seq) in MEASURED_PARTS:
        return MEASURED_PARTS[(rows, seq)]
    compact = decode_compact_uses_buffer(rows, seq)
    ratio = PARTS_C2[compact] / PARTS_C1[compact]
    # Floored at two because the caps reject one at no cost: a row short enough to
    # want a single workgroup is already on the short tier, and the spare
    # workgroup returns immediately.
    return max(2, min(PARTS_CAP, round(math.sqrt(ratio * seq / rows))))


def decode_compact_parts(rows: int, seq: int, k: int = MEASURED_PARTS_K) -> int:
    """How many of the grid's workgroups per row actually take part."""
    want = decode_compact_want(rows, seq, k)
    return min(decode_compact_grid(rows, want, seq), want)


def decode_compact_collapse(rows: int, parts: int) -> bool:
    """Whether the pass behind the fill settles its digit without a merge.

    Retired, and the histogram certificate is why. Collapsing buys one barrier with
    every part re-reading the whole candidate buffer instead of its own slice, and a
    certified pass no longer has a barrier to sell: a compacted row certifies its
    fill pass only when the collapse is off, so the collapse now gives up a
    certificate to remove a barrier the certificate would have removed for free.

    Re-measured over all twenty-three buffer-reading cells, three reps, spread
    within a cell 0.05us: off wins on twenty of them by 0.4-12.6%, and the three it
    loses -- 32 rows by 64K, 64 rows by 512K and 1M -- are 0.5-1.5%, which is
    inside the drift between sessions on this node. A rule fitted to those three
    would be fitting noise.
    """
    return False


SHORT_MAX_BASE, SHORT_MAX_SLOPE, SHORT_MAX_CAP = 28416, 160, 40960


def decode_compact_short_max(rows: int) -> int:
    """The length below which a row runs on the single-workgroup short tier.

    Measured here rather than read from the dispatcher, which is a divergence and
    is meant to be one. Forcing the tier both ways from 4K to 49K, with the width
    swept on the multi-block side, at k=128, 512 and 2048: the crossover is 28.3K
    at one row, 28.1K at two, 28.9K at four, 31.0K at eight, 31.5K at sixteen and
    33.1K at thirty-two. It is nearly flat in the row count, and the same to
    within 800 elements at all three k, so k does not enter the rule.

    The dispatcher's rule climbs six times as steeply -- 20K at one row to the 41K
    cap by sixteen -- because it was fitted against the plain tiered kernel's
    multi-block path, and this kernel's is not that path. Its crossover has not
    been re-measured and must not be moved to match this one.

    On the grid this moves exactly one cell, 16 rows by 32K, which the old
    threshold held on the short tier at 23.45us against 21.68us multi-block: 7.6%.
    Off the grid it is worth 10-12% between 20K and 28K at one to four rows, and
    7-19% between 31K and 41K at sixteen and thirty-two.
    """
    # Imported inside the call because the dispatcher imports this module, so the
    # dependency only closes once both are loaded.
    from .. import topk_per_row_decode as _dispatch

    return min(
        SHORT_MAX_CAP, SHORT_MAX_BASE + _dispatch._next_pow2(rows) * SHORT_MAX_SLOPE
    )


def decode_compact_certificate(seq: int, parts: int) -> bool:
    """Whether pass 0 should settle its digit from the histogram's own total.

    The certificate trades the arrival counter for re-reading the merged histogram
    uncached until its bins sum to the row length. What it saves scales with how
    many parts contend for the arrival counter's cache line; what it costs is a
    2048-bin uncached read, which is fixed. So it pays once the row is long enough
    that the read is small beside the scan, and at 32K it needs the wider grids to
    be worth it. Measured over the 36-cell grid: on in all 30 qualifying cells and
    off in the six where it lost, worst of those 0.967x.
    """
    if seq >= 65536:
        return True
    return seq >= 32768 and parts >= 4


def decode_compact_config(
    rows: int,
    seq: int,
    k: int = MEASURED_PARTS_K,
    *,
    compact_cap_mult: int = 16,
    tiered_short_max: int | None = None,
    ordered: bool = True,
) -> dict:
    """Everything `create_topk_per_row_decode_compact_kernel` needs for a shape.

    Returns the keyword arguments for the factory plus `parts`, `grid` and
    `compact`, which the caller needs for the workspace size and for reporting.

    The caps are floored at two even when the grid gives a row a single
    workgroup, because they are upper bounds: the tier takes
    `min(blocks_per_row, cap)`, so a floor of two still leaves one active part on
    a one-wide grid, while a cap of one is rejected outright as a cooperating tier
    that cannot cooperate. Without the floor every batch past 256 rows raised
    instead of running, which is a shape a decode batch arrives in.
    """
    parts = decode_compact_parts(rows, seq, k)
    grid = decode_compact_grid(rows, decode_compact_want(rows, seq, k), seq)
    short_max = (
        decode_compact_short_max(rows)
        if tiered_short_max is None
        else tiered_short_max
    )
    compact = decode_compact_compacts(rows, seq, parts, short_max, ordered=ordered)
    kw = {
        "blocks_per_row": grid,
        "bits_per_pass": 11,
        "tier_mode": "auto",
        "tiered_mid_cap": max(2, parts),
        "tiered_long_cap": max(2, parts),
        "tiered_short_max": short_max,
        "ordered": ordered,
        # Poll the barrier with monotonic loads and take one acquire once the token
        # lands. Two thirds of what an extra participant cost was the acquire every
        # waiting workgroup issued on every unsuccessful poll.
        "poll_then_acquire": True,
    }
    if decode_compact_certificate(seq, parts):
        kw["histogram_certificate"] = True
    if compact:
        kw["compact"] = True
        kw["compact_cap_mult"] = compact_cap_mult
        kw["compact_fill_vecs"] = COMPACT_FILL_VECS
        # Publishing the append carry from wave 0 between the fill scan's two
        # barriers lets each tile drop its closing barrier: three to two. The scan
        # itself measures as noise and is carried because the carry is built on it.
        kw["compact_fast_scan"] = True
        kw["compact_early_carry"] = True
        if decode_compact_collapse(rows, parts):
            kw["collapse"] = True
    return {"kw": kw, "grid": grid, "parts": parts, "compact": compact}


def _num_passes(bits_per_pass: int) -> int:
    return (32 + bits_per_pass - 1) // bits_per_pass


def _compact_row_slots(compact: bool, compact_cap: int) -> int:
    """Extra per-row int32 slots for the candidate buffer (header + col/key pairs)."""
    return COMPACT_HDR_SLOTS + 2 * int(compact_cap) if compact else 0


def topk_workspace_slots(
    num_rows: int,
    bits_per_pass: int = 11,
    compact: bool = False,
    compact_cap: int = 0,
) -> int:
    """Return int32 workspace slots for the tiered path (row-major, per row)."""
    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")
    row_slots = COUNTER_SLOTS + _num_passes(bits_per_pass) * (1 << bits_per_pass)
    row_slots += _compact_row_slots(compact, compact_cap)
    return int(num_rows) * row_slots


def needs_workspace_zero(
    max_row_len: int,
    top_k: int,
    short_max: int,
    tier_mode: str = "auto",
    bits_per_pass: int = 11,
) -> bool:
    """Return whether any row can enter the persistent multi-block path."""
    if tier_mode == "short":
        return False
    if tier_mode in ("mid", "long"):
        return True
    # No short tier below 11 bits, so every row is persistent regardless of length.
    if bits_per_pass != 11:
        return True
    return max_row_len > max(short_max, top_k)


@cache
def create_topk_per_row_decode_compact_kernel(
    top_k: int,
    *,
    blocks_per_row: int = 8,
    bits_per_pass: int = 11,
    scan_stages: int = SCAN_STAGES,
    tier_mode: Literal["auto", "short", "mid", "long"] = "auto",
    tiered_short_max: int = 16384,
    tiered_mid_cap: int = 16,
    tiered_mid_max: int = 65536,
    tiered_long_cap: int = 32,
    mask_non_finite: bool = False,
    row_proportional_parts: bool = False,
    early_stop: bool = False,
    ordered: bool = False,
    compact: bool = False,
    compact_cap_mult: int = 16,
    compact_fill_vecs: int = COMPACT_FILL_VECS,
    compact_fill_only: bool = False,
    compact_fast_scan: bool = False,
    compact_early_carry: bool = False,
    collapse: bool = False,
    collapse_cap: int = 16384,
    collapse_budget: int = 250,
    spin_sleep: int = 1,
    poll_then_acquire: bool = False,
    histogram_certificate: bool = False,
    certificate_publish_total: bool = False,
    certificate_trace: bool = False,
    certificate_fill_pass: bool = True,
    barrier_relaxed: bool = False,
    barrier_relax_scope: str = "all",
    merge_off: bool = False,
) -> Any:
    """Build a launcher that selects the Top-K largest values' column indices per
    decode row, matching torch.topk by value -- as an unordered set by default, or
    ascending with deterministic tie-breaking under ``ordered``. Implemented as a
    tiered persistent multi-block radix-select; the returned launcher is cached.

    top_k: number of indices selected per row (compile-time; any positive value).
    blocks_per_row: workgroups launched per row (grid width); the mid/long tiers cap
        how many actually cooperate, excess workgroups return immediately.
    bits_per_pass: radix digit width, 10 or 11; 11 = 2048-bin LDS histogram, required
        by the short tier.
    scan_stages: histogram block-scan staging, one of 1/2/4/8.
    tier_mode: "auto" picks a tier per row by valid length; "short"/"mid"/"long"
        force that tier for every row.
    tiered_short_max: row_len <= this -> short tier (single workgroup, barrier-free).
    tiered_mid_max: short_max < row_len <= this -> mid tier; longer -> long tier.
    tiered_mid_cap / tiered_long_cap: max cooperating workgroups per row in the mid /
        long tier (clamped to blocks_per_row).
    mask_non_finite: clamp inf/NaN to -inf so they never rank into the top-k. Off
        by default, which ranks them by their raw twiddled bits the way torch.topk
        and the HIP kernel do; a direct caller has to ask for the divergence.
    ordered: ascending output with smallest-index tie-break on the kth value.
    compact: pass 1 writes the elements that survive pass 0's digit into a per-row
        candidate buffer, and pass 2 and the emit read that buffer instead of the
        row. Pass 1 already tests every element against the settled prefix, so the
        survivors cost nothing extra to identify -- only to store.
    compact_cap_mult: candidate buffer capacity per row, in multiples of top_k. A
        row whose survivors exceed it falls back to rescanning the row, so this
        trades workspace for how often the fast path is taken, never correctness.
    compact_fill_only: diagnostic. Fill the buffer but keep reading the row, so the
        cost of appending can be priced apart from the saving of reading back. Still
        correct, just strictly slower than compact.
    compact_fast_scan: reuse the previous tile's closing workgroup barrier and read
        the append carry before prefix scan, removing two redundant local barriers
        per compact-fill tile while preserving exact candidate order.
    compact_early_carry: publish the next append carry from wave 0 before the
        prefix-scan closing barrier, removing the tile's trailing barrier.
    spin_sleep: s_sleep units between polls in the row barrier's wait, 0..15 at about
        64 clocks each. Every waiting workgroup polls with an agent-scope acquire
        load, so the poll rate sets how much fabric traffic the wait costs the
        workgroups that are still doing useful work.
    poll_then_acquire: poll pass_done with volatile monotonic loads, then perform
        one volatile acquire load after observing the release-published token.
        This keeps the synchronizes-with edge while avoiding acquire semantics on
        unsuccessful polls.
    histogram_certificate: settle pass 0's digit without its row barrier, by letting
        the merged histogram certify its own completion. Every part contributes each
        of its elements to exactly one bin with an agent-scope monotonic atomic, so
        the bins only grow and their sum can only reach the row's live length once
        every part has published. The bucket scan already computes that sum on its
        way to the k-th bucket, so a part re-reads and re-scans until the sum equals
        the live length instead of counting arrivals. Because the bins are monotone
        and the total is conserved, a snapshot that sums to the live length cannot
        hold a partial bin, which is what makes the payload its own certificate.
        Only pass 0 qualifies: later passes can have a part fall back to scanning its
        row slice on candidate-buffer overflow, which breaks the conservation.
    certificate_publish_total: keep the certificate's proof but stop recomputing it.
        Each part adds the count it flushed to one per-pass word with a release, and
        a waiting part polls that word instead of re-reading 2048 bins and rescanning
        them. The proof is the same conserved quantity and the same comparison
        against the live length; only where the sum comes from changes. It exists
        because the rescan is what makes the wait expensive: a poll costs about a
        microsecond, and at 128 rows by 1M the loop reached CERTIFICATE_MAX_SPINS and
        gave up after 4.28 s. The cost this reintroduces is the contention the
        certificate was built to avoid -- one hot word again -- but a part writes it
        once per pass rather than polling it, which is the half that was cheap.
    certificate_trace: diagnostic. Records what the re-read loop had seen when it
        stopped -- spins reached, the total it last read, the live length it wanted,
        and the same bins summed once more after it gave up. The last of those is
        the one that matters: if the histogram was complete by then, the loop was
        not waiting for anything and the wait was a visibility failure rather than a
        slow peer. Costs a handful of atomics per pass and is off by default.
    certificate_fill_pass: whether the compact fill pass may certify as well as
        pass 0. It does today, and `certificate_trace` says that is where every
        freeze comes from: eight caught freezes, eight at the fill pass and none at
        pass 0, each with the bins complete and stable and the live length they were
        compared against wrong by -26 to +46 in about 5100. Pass 0 compares against
        the row length, which is exact; the fill pass compares against a count the
        previous pass settled, which is not the same quantity the fill's histogram
        sums to. Setting this False leaves the certificate only where its
        conservation argument holds, which is what the docstring above already
        claims it does.
    barrier_relaxed: diagnostic, and wrong on purpose. Drops the row barrier's
        acquire/release pairs to monotonic, which prices what the ordering costs
        without changing a single loop bound -- the histogram a peer reads may then
        be stale, so results are not to be trusted, only the timing.
    barrier_relax_scope: which half of the barrier barrier_relaxed applies to, so
        the ordering cost can be split. "arrival" relaxes only the read-modify-write
        that counts arrivals; "payload" relaxes only the release publish and the
        acquire poll that carry histogram visibility; "all" relaxes both. Splitting
        it says whether a scheme that removes the arrival counter but keeps payload
        ordering, such as a cardinality certificate, can reach the cost at all.
    merge_off: diagnostic, and wrong on purpose. Keeps every row barrier but drops
        the flush and the read-back around it, so each workgroup settles its digit
        from its own slice's histogram. Prices the merge separately from the
        barrier's round trip, which the short tier cannot do because it has neither.
    """
    short_max = tiered_short_max
    mid_cap = tiered_mid_cap
    mid_max = tiered_mid_max
    long_cap = tiered_long_cap

    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")
    if compact and compact_cap_mult < 1:
        raise ValueError(
            f"compact_cap_mult must be >= 1, got {compact_cap_mult}"
        )
    if compact and _num_passes(bits_per_pass) < 3:
        raise ValueError(
            "compact needs a pass after the one that fills the buffer; "
            f"bits_per_pass={bits_per_pass} leaves too few passes"
        )
    if compact and not ordered and early_stop:
        raise ValueError(
            "compact with the unordered emit does not implement early_stop: the "
            "early write still re-reads the row, so it would miss the buffer"
        )
    if scan_stages not in (1, 2, 4, 8):
        raise ValueError(f"scan_stages must be one of (1, 2, 4, 8), got {scan_stages}")

    # blocks_per_row == 1 collapses the launch to grid=(1, num_rows): every row runs
    # the barrier-free single-workgroup short tier (no cooperative parts, so no dead
    # blocks hogging co-resident slots). Only valid when the short tier exists
    # (auto/short + bpp==11); mid/long forced modes still need >=2 cooperating parts.
    _min_blocks_per_row = (
        1 if (tier_mode in ("auto", "short") and bits_per_pass == 11) else 2
    )
    if not _min_blocks_per_row <= blocks_per_row <= 32:
        raise ValueError(
            f"blocks_per_row must be in [{_min_blocks_per_row}, 32], got {blocks_per_row}"
        )

    if mid_cap < 2 or long_cap < 2:
        raise ValueError(f"mid_cap/long_cap must be >= 2, got {mid_cap}/{long_cap}")
    if mid_max < short_max:
        raise ValueError(f"mid_max must be >= short_max, got {mid_max} < {short_max}")
    if tier_mode not in ("auto", "short", "mid", "long"):
        raise ValueError(
            f"tier_mode must be one of auto/short/mid/long, got {tier_mode!r}"
        )
    # The short tier runs the standalone one-workgroup radix-select (2048-bin LDS
    # histogram), so it needs bits_per_pass == 11. It is compiled in for "auto"
    # (short rows) and "short" (all rows); forcing "short" without bpp==11 is an
    # error rather than a silent fallback.
    if tier_mode == "short" and bits_per_pass != 11:
        raise ValueError(
            f"tier_mode='short' requires bits_per_pass == 11, got {bits_per_pass}"
        )

    # ordered=True drops the short tier and early_stop; both assume unordered emit.
    if ordered:
        short_tier = False
        early_stop = False
    else:
        short_tier = tier_mode in ("auto", "short") and bits_per_pass == 11
    if collapse and not compact:
        raise ValueError("collapse=True requires compact=True")
    if collapse and compact_fill_only:
        raise ValueError("collapse=True is incompatible with compact_fill_only")
    if compact_early_carry and not compact_fast_scan:
        raise ValueError("compact_early_carry=True requires compact_fast_scan=True")
    if collapse and collapse_cap < top_k:
        raise ValueError(f"collapse_cap must be >= top_k, got {collapse_cap}")
    if collapse and collapse_budget < 1:
        raise ValueError(f"collapse_budget must be >= 1, got {collapse_budget}")
    # s_sleep takes 0..15 and waits about 64 clocks per unit.
    if not 0 <= spin_sleep <= 15:
        raise ValueError(f"spin_sleep must be 0..15, got {spin_sleep}")
    if ordered and blocks_per_row > WARP_SIZE:
        raise ValueError(
            f"ordered=True scans slice bases within one wave, so blocks_per_row "
            f"must be <= {WARP_SIZE}, got {blocks_per_row}"
        )
    block_threads = BLOCK_THREADS
    red_slots = (block_threads + WARP_SIZE - 1) // WARP_SIZE
    num_passes = _num_passes(bits_per_pass)
    num_buckets = 1 << bits_per_pass
    # Pass 0 settles the digit that defines a candidate, so pass 1 is the earliest
    # that can fill the buffer; every pass after it reads the buffer instead of the row.
    compact_fill_pass = 1

    def certified_pass(pass_id: int) -> bool:
        """Whether this pass's merge can prove its own completion.

        A pass histograms exactly the count its predecessor settled on, so the
        conservation argument carries past pass 0. Two things end it.

        The candidate buffer: from the pass after the fill, a part whose slice
        overflowed histograms its row slice instead of its candidates, and the
        row's total is then no longer the count anyone expects.

        And collapse, which is why the fill is excluded when it is on. The fill's
        barrier does not only order the histogram; the parts publish their
        candidate counts across it, and the collapse decision reads all of them.
        Those counts are plain stores, not a conserved quantity, so certifying the
        histogram says nothing about whether they have landed. With collapse off, a
        part reads only its own slice and there is no such side payload -- an
        overflowing part still counts every candidate it found, it just cannot
        store them all.
        """
        if not histogram_certificate or merge_off:
            return False
        if not compact:
            return True
        if pass_id < compact_fill_pass:
            return True
        return (
            certificate_fill_pass and pass_id == compact_fill_pass and not collapse
        )

    def barrier_token_for(pass_id: int) -> int:
        # The token a pass waits on is a count of the barriers actually reached,
        # not of the passes behind it: a certified pass never arrives, so it must
        # not advance the counter the uncertified ones agree on.
        return 1 + sum(not certified_pass(q) for q in range(pass_id))

    # The emit waits behind every barrier the passes reached, for the same reason.
    emit_barrier_token = 1 + sum(not certified_pass(q) for q in range(num_passes))

    compact_cap = compact_cap_mult * top_k if compact else 0
    compact_base = COUNTER_SLOTS + num_passes * num_buckets
    compact_data = compact_base + COMPACT_HDR_SLOTS
    row_workspace_slots = compact_base + _compact_row_slots(compact, compact_cap)

    # Caps/thresholds only affect codegen for modes that use them; include them in
    # the name for those modes so distinct configs cache separately.
    _cap_tag = (
        ""
        if tier_mode == "short"
        else (
            f"_s{tiered_short_max}_mc{tiered_mid_cap}"
            f"_mm{tiered_mid_max}_lc{tiered_long_cap}"
        )
    )
    kernel_name = (
        f"topk_per_row_decode_compact_k{top_k}_"
        f"bpp{bits_per_pass}_g{blocks_per_row}_v2"
        f"_stage{scan_stages}"
        f"_{tier_mode}"
        f"{_cap_tag}"
        f"{'_1wg' if short_tier else ''}"
        f"{'_mf' if mask_non_finite else ''}"
        f"{'_rpp' if row_proportional_parts else ''}"
        f"{'_es' if early_stop else ''}"
        f"{'_ord' if ordered else ''}"
        f"{f'_cmp{compact_cap_mult}' if compact else ''}"
        f"{f'_fv{compact_fill_vecs}' if compact else ''}"
        f"{'_fillonly' if compact and compact_fill_only else ''}"
        f"{'_fastscan' if compact and compact_fast_scan else ''}"
        f"{'_earlycarry' if compact and compact_early_carry else ''}"
        f"{f'_col{collapse_cap}b{collapse_budget}' if collapse else ''}"
        f"{f'_sl{spin_sleep}' if spin_sleep != 1 else ''}"
        f"{'_poll1acq' if poll_then_acquire else ''}"
        f"{'_cert' if histogram_certificate else ''}"
        f"{'pub' if histogram_certificate and certificate_publish_total else ''}"
        f"{'trace' if histogram_certificate and certificate_trace else ''}"
        f"{'p0' if histogram_certificate and not certificate_fill_pass else ''}"
        f"{f'_relaxed{barrier_relax_scope}' if barrier_relaxed else ''}"
        f"{'_nomerge' if merge_off else ''}"
    )

    @fx.struct
    class SharedStorage:
        s_hist: fx.Array[fx.Int32, num_buckets, 16]
        s_scan: fx.Array[fx.Int32, red_slots * 2, 16]
        s_meta: fx.Array[fx.Int32, 8, 16]
        # 0..5 ordered emit carry / run bases; 6..7 compact append carry / overflow
        s_run: fx.Array[fx.Int32, 8 if compact else 6, 16]
        s_own_hist: fx.Array[fx.Int32, num_buckets if ordered else 1, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def topk_per_row_decode_compact_kernel(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        stride0: fx.Int32,
    ) -> None:
        block_x = gpu.block_id("x")
        block_y = gpu.block_id("y")
        thread_x = gpu.thread_id("x")
        part = fx.Int32(block_x)
        row = fx.Int32(block_y)
        tid = fx.Int32(thread_x)
        tid_idx = fx.Index(thread_x)
        lane = tid % fx.Int32(WARP_SIZE)
        wave = tid // fx.Int32(WARP_SIZE)

        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_two = fx.Int32(2)
        c_four = fx.Int32(4)
        c_red_slots = fx.Int32(red_slots)
        c_last_wave = fx.Int32(red_slots - 1)
        c_last_lane = fx.Int32(WARP_SIZE - 1)
        c_vec = fx.Int32(LOAD_VEC)
        c_top_k = fx.Int32(top_k)
        c_block_i32 = fx.Int32(block_threads)
        c_block_idx = fx.Index(block_threads)
        c_bins_i32 = fx.Int32(num_buckets)
        c_bins_idx = fx.Index(num_buckets)
        c_parts = fx.Int32(blocks_per_row)
        c_sign_bit = fx.Int32(-2147483648)
        c_exp_mask = fx.Int32(0x7F800000)  # fp32 exponent bits (all-ones => inf/NaN)
        c_neg_inf = fx.Float32(float("-inf"))
        c_neg_one = fx.Int32(-1)
        c_sixteen = fx.Int32(16)
        c_low16 = fx.Int32(0xFFFF)
        c_three = fx.Int32(3)
        c_five = fx.Int32(5)
        c_six = fx.Int32(6)
        c_seven = fx.Int32(7)
        c_row_ws = fx.Int32(row_workspace_slots)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_hist = lds.s_hist.view(fx.make_layout(num_buckets, 1))
        s_scan = lds.s_scan.view(fx.make_layout(red_slots * 2, 1))
        s_meta = lds.s_meta.view(fx.make_layout(8, 1))
        s_run = lds.s_run.view(fx.make_layout(8 if compact else 6, 1))
        s_own_hist = lds.s_own_hist.view(
            fx.make_layout(num_buckets if ordered else 1, 1)
        )

        # Bound this descriptor to the real allocation instead of 4 GiB. The vec4 tail
        # loads a whole group even when 1-3 elements remain and predicates the lanes
        # only afterwards, so the last row of an unpadded buffer fetches past the
        # tensor. The true size restores the hardware range check, which returns zero
        # for those lanes, and the tail mask discards it as before.
        logits_bytes = fx.Int64(gpu.grid_dim.y) * fx.Int64(stride0) * fx.Int64(4)
        logits_rsrc = buffer_ops.create_buffer_resource(
            logits, max_size=False, num_records_bytes=logits_bytes
        )
        seq_lens_rsrc = buffer_ops.create_buffer_resource(seq_lens, max_size=True)
        indices_bytes = fx.Int64(gpu.grid_dim.y) * fx.Int64(top_k) * fx.Int64(4)
        indices_rsrc = buffer_ops.create_buffer_resource(
            indices, max_size=False, num_records_bytes=indices_bytes
        )
        workspace_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        workspace_base_idx = buffer_ops.extract_base_index(workspace, address_space=1)

        hist_base_ptr = fx.ptrtoint(lds.s_hist.ptr)
        meta_base_ptr = fx.ptrtoint(lds.s_meta.ptr)

        # Decode row geometry.
        seq_row = row // next_n
        slot = row - seq_row * next_n
        seq_len = fx.Int32(
            buffer_ops.buffer_load(seq_lens_rsrc, seq_row, vec_width=1, dtype=T.i32)
        )
        row_len = seq_len - next_n + slot + c_one
        row_len = (row_len > c_zero).select(row_len, c_zero)
        row_base = row * stride0
        row_out = row * c_top_k
        row_ws_base = row * c_row_ws

        # Active cooperating workgroups per row over the fixed grid (excess blocks
        # return immediately). "auto" picks per row by length; short/mid/long force
        # that tier for every row. Caps are clamped to the grid.
        c_mid_cap = fx.Int32(mid_cap)
        c_long_cap = fx.Int32(long_cap)
        mid_parts = (c_parts < c_mid_cap).select(c_parts, c_mid_cap)
        long_parts = (c_parts < c_long_cap).select(c_parts, c_long_cap)
        if const_expr(row_proportional_parts):
            # gfx950: the grid is sized for the padded width, so a short row would spin
            # up more cooperating workgroups than it needs, each adding barrier/merge
            # latency. Cap parts by the row's coverage need (one per items_per_block),
            # floored at 2. Fewer parts is always correct -- the scan stride still
            # covers every vec-block.
            items_per_block = LOAD_VEC * block_threads
            cover_shift = (items_per_block).bit_length() - 1
            row_cover = (row_len + fx.Int32(items_per_block - 1)).shrui(
                fx.Int32(cover_shift)
            )
            row_cover = (row_cover < c_two).select(c_two, row_cover)
            mid_parts = (mid_parts < row_cover).select(mid_parts, row_cover)
            long_parts = (long_parts < row_cover).select(long_parts, row_cover)
        if const_expr(tier_mode == "short"):
            active_parts = c_one
        elif const_expr(tier_mode == "mid"):
            active_parts = mid_parts
        elif const_expr(tier_mode == "long"):
            active_parts = long_parts
        else:  # "auto": pick per row by valid length
            c_short = fx.Int32(short_max)
            c_mid = fx.Int32(mid_max)
            active_parts = (row_len <= c_short).select(
                c_one,
                (row_len <= c_mid).select(mid_parts, long_parts),
            )
        active_threads = active_parts * c_block_i32
        single_part_active = active_parts == c_one

        def collapsed_flag():
            """Re-read the collapse decision where it is used."""
            return fx.memref_load(s_meta, fx.Int32(SMEM_META_COLLAPSED)) == c_one

        def active_stride_idx(mult: int = 1):
            """Vec-block stride covering `mult` rounds of the active workgroups."""
            return fx.Index(active_threads * fx.Int32(mult))

        def scan_run_stride_idx(mult: int = 1):
            """Vec-block stride within one workgroup's ordered run."""
            return fx.Index(c_block_i32 * fx.Int32(mult))

        def counter_slot(slot_const: int):
            return row_ws_base + fx.Int32(slot_const)

        def histogram_slot(pass_id: int, bin_i32):
            return (
                row_ws_base + fx.Int32(COUNTER_SLOTS + pass_id * num_buckets) + bin_i32
            )

        def compact_hdr_slot(field: int):
            return row_ws_base + fx.Int32(compact_base + field)

        def compact_col_slot(entry_i32):
            return row_ws_base + fx.Int32(compact_data) + entry_i32 * c_two

        def compact_key_slot(entry_i32):
            return row_ws_base + fx.Int32(compact_data + 1) + entry_i32 * c_two

        def global_i32_ptr(elem_i32):
            elem_idx = fx.Index(elem_i32)
            addr = fx.Index(workspace_base_idx) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def lds_i32_ptr(base, elem_i32):
            elem_idx = fx.Index(elem_i32)
            addr = fx.Index(base) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=3)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def ws_load(elem_i32):
            return buffer_ops.buffer_load(
                workspace_rsrc, elem_i32, vec_width=1, dtype=T.i32
            )

        def global_atomic_add_i32(
            elem_i32, value, ordering=llvm.AtomicOrdering.monotonic
        ):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                global_i32_ptr(elem_i32),
                as_ir_value(value),
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_max_i32(elem_i32, value):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.max,
                global_i32_ptr(elem_i32),
                as_ir_value(value),
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_xchg_i32(elem_i32, value, ordering):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.xchg,
                global_i32_ptr(elem_i32),
                as_ir_value(value),
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def barrier_ordering(strong, half: str = "payload"):
            # The two halves are priced separately because they are removable
            # separately: the arrival count is what a certificate replaces, the
            # payload ordering is what it still has to provide.
            relax = barrier_relaxed and barrier_relax_scope in ("all", half)
            return (llvm.AtomicOrdering.monotonic
                    if const_expr(relax) else strong)

        def global_atomic_load_i32_acquire(elem_i32):
            # Volatile agent-scoped acquire load for the row-barrier spin. The
            # matching release publish below makes histogram updates visible to
            # peer workgroups without issuing a read-modify-write on the polled slot.
            return llvm.LoadOp(
                T.i32,
                global_i32_ptr(elem_i32),
                alignment=4,
                volatile_=True,
                ordering=barrier_ordering(llvm.AtomicOrdering.acquire),
                syncscope="agent",
            ).result

        def global_atomic_load_i32_monotonic(elem_i32):
            return llvm.LoadOp(
                T.i32,
                global_i32_ptr(elem_i32),
                alignment=4,
                volatile_=True,
                ordering=llvm.AtomicOrdering.monotonic,
                syncscope="agent",
            ).result

        def lds_atomic_add_i32(base, elem_i32, value):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                lds_i32_ptr(base, elem_i32),
                as_ir_value(value),
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result

        def spin_until_slot_ge(elem_i32, target):
            w = scf.WhileOp([T.i32], [as_ir_value(c_zero)])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.slt, cur, as_ir_value(target)
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                rocdl.s_sleep(spin_sleep)
                data = (
                    global_atomic_load_i32_monotonic(elem_i32)
                    if const_expr(poll_then_acquire)
                    else global_atomic_load_i32_acquire(elem_i32)
                )
                scf.YieldOp([data])
            if const_expr(poll_then_acquire):
                # The loop's monotonic observation only decides when to stop. This
                # acquire re-reads the same monotonically increasing token and pairs
                # with the last workgroup's release publish before any histogram
                # data is consumed.
                global_atomic_load_i32_acquire(elem_i32)

        def row_barrier(token):
            # Intentional no-drain acquire/release protocol: workgroup barriers
            # bracket local LDS work, the last workgroup release-publishes pass_done,
            # and peers spin with acquire loads. A full waitcnt drain here is
            # performance/correctness sensitive.
            # The token doubles as the arrival count, so it may be a value the
            # row settles at runtime: skip a barrier and every later token has to
            # come down by one, or the last arrival never lands and the row hangs.
            token_value = fx.Int32(token) if isinstance(token, int) else token
            target_arrivals = token_value * active_parts
            gpu.barrier()
            if tid == c_zero:
                prev = global_atomic_add_i32(
                    counter_slot(COUNTER_ARRIVALS),
                    c_one,
                    barrier_ordering(llvm.AtomicOrdering.acq_rel, "arrival"),
                )
                last = (prev + c_one) == target_arrivals
                if last:
                    global_atomic_xchg_i32(
                        counter_slot(COUNTER_PASS_DONE),
                        token_value,
                        barrier_ordering(llvm.AtomicOrdering.release),
                    )
                else:
                    spin_until_slot_ge(counter_slot(COUNTER_PASS_DONE), token_value)
            gpu.barrier()

        def mask_nonfinite(val):
            # inf/NaN (exponent all-ones) -> -inf so they sort below every finite
            # value and are never selected.
            if const_expr(not mask_non_finite):
                return val
            bits = val.bitcast(T.i32)
            is_nonfinite = (bits & c_exp_mask) == c_exp_mask
            return is_nonfinite.select(c_neg_inf, val)

        def radix_twiddle_key(val):
            # Map larger fp32 values to smaller unsigned keys so ascending
            # bucket scans select descending values. Signed zero is left alone:
            # torch.topk and the HIP kernel both rank -0.0 strictly below +0.0,
            # so collapsing the two here would change which tied index we emit.
            val = mask_nonfinite(val)
            bits = val.bitcast(T.i32)
            sign = bits.shrui(fx.Int32(31))
            positive_mask = bits ^ fx.Int32(0x7FFFFFFF)
            return (sign == c_zero).select(positive_mask, bits)

        def bucket_for_key(key, start_bit: int):
            return (key.shrui(fx.Int32(start_bit))) & fx.Int32(num_buckets - 1)

        def prefix_for_key(key, previous_start_bit: int):
            return arith.shli(
                key.shrui(fx.Int32(previous_start_bit)),
                fx.Int32(previous_start_bit),
            )

        def load_row_vec(col_base_i32):
            return buffer_ops.buffer_load(
                logits_rsrc,
                row_base + col_base_i32,
                vec_width=LOAD_VEC,
                dtype=T.f32,
            )

        def clear_local_histogram():
            for hist_idx in range(tid_idx, c_bins_idx, c_block_idx):
                fx.memref_store(c_zero, s_hist, fx.Int32(hist_idx))
            gpu.barrier()

        def wave_inclusive_scan_i32(value):
            cur = value
            for sh in range_constexpr(int.bit_length(WARP_SIZE) - 1):
                d = fx.Int32(1 << sh)
                src_lane = lane - d
                byte_addr = src_lane * c_four
                peer = rocdl.ds_bpermute(
                    T.i32, as_ir_value(byte_addr), as_ir_value(cur)
                )
                take = lane >= d
                cur = take.select(cur + peer, cur)
            return cur

        def choose_bucket_prefix(target_k):
            # Multi-block ascending block scan over the LDS histogram; each thread owns a bin pair.
            first_bin = tid * c_two
            bin0_valid = first_bin < c_bins_i32
            bin1 = first_bin + c_one
            bin1_valid = bin1 < c_bins_i32
            safe0 = bin0_valid.select(first_bin, c_zero)
            safe1 = bin1_valid.select(bin1, c_zero)
            c0 = bin0_valid.select(fx.memref_load(s_hist, safe0), c_zero)
            c1 = bin1_valid.select(fx.memref_load(s_hist, safe1), c_zero)
            local_total = c0 + c1

            wave_incl = wave_inclusive_scan_i32(local_total)
            wave_excl_thread = wave_incl - local_total

            if lane == c_last_lane:
                fx.memref_store(wave_incl, s_scan, wave)
            gpu.barrier()

            if wave == c_zero:
                in16 = lane < c_red_slots
                lane_safe = in16.select(lane, c_zero)
                wtot = in16.select(fx.memref_load(s_scan, lane_safe), c_zero)
                wincl = wave_inclusive_scan_i32(wtot)
                wexcl = wincl - wtot
                if in16:
                    fx.memref_store(wexcl, s_scan, lane + c_red_slots)
                # The scan already carries the histogram's grand total in the last
                # wave's inclusive value, so the certificate's test costs a store
                # here and a compare there, not a second pass over the bins.
                if const_expr(histogram_certificate) and lane == c_last_wave:
                    fx.memref_store(wincl, s_meta, fx.Int32(SMEM_META_TOTAL))
            gpu.barrier()

            wave_off = fx.memref_load(s_scan, wave + c_red_slots)
            excl0 = wave_off + wave_excl_thread
            incl0 = excl0 + c0
            incl1 = incl0 + c1

            def emit_find(bucket, excl, incl, count):
                crosses = (excl < target_k) & (incl >= target_k)
                if crosses:
                    fx.memref_store(
                        target_k - excl,
                        s_meta,
                        fx.Int32(SMEM_META_K),
                    )
                    fx.memref_store(count, s_meta, fx.Int32(SMEM_META_LEN))
                    fx.memref_store(bucket, s_meta, fx.Int32(SMEM_META_THRESHOLD))
                    fx.memref_store(excl, s_meta, fx.Int32(SMEM_META_ABOVE))

            emit_find(first_bin, excl0, incl0, c0)
            emit_find(bin1, incl0, incl1, c1)
            gpu.barrier()

        def certified_merge(pass_id: int, target_k, expected_total, current_bits=None):
            """Merge without a row barrier, using the histogram's own total as proof.

            Re-reads the bins and re-runs the bucket scan until the scan's total
            equals the row's live cardinality. The attempt cap only bounds a
            pathological wait: a row that hits it settles its digit from an
            incomplete histogram, which the caller's correctness check would catch,
            rather than wedging the queue.
            """
            if const_expr(certificate_publish_total):
                # Same proof, read instead of recomputed. The parts' release adds
                # form a release sequence on this word, so one acquire after the
                # total lands synchronizes with all of them and the bins can be read
                # through the caches exactly as they are after a row barrier.
                if tid == c_zero:
                    spin_until_slot_ge(
                        counter_slot(COUNTER_CERT_TOTAL + pass_id), expected_total
                    )
                gpu.barrier()
                load_global_histogram(pass_id)
                choose_bucket_prefix(target_k)
                return
            if const_expr(certificate_trace):
                if tid == c_zero:
                    global_atomic_add_i32(counter_slot(CERT_TRACE_ARRIVED), c_one)
                    if const_expr(
                        pass_id == compact_fill_pass
                        and current_bits is not None
                        and blocks_per_row <= 8
                    ):
                        # Written by every part, not only one that gives up, so the
                        # peer's digit is on record even when the peer never waits.
                        buffer_ops.buffer_store(
                            current_bits,
                            workspace_rsrc,
                            counter_slot(CERT_TRACE_BITS) + part,
                        )
                        buffer_ops.buffer_store(
                            expected_total,
                            workspace_rsrc,
                            counter_slot(CERT_TRACE_WANT_P) + part,
                        )
            load_global_histogram(pass_id, coherent=True)
            choose_bucket_prefix(target_k)
            w = scf.WhileOp([T.i32], [as_ir_value(c_zero)])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                spins = before.arguments[0]
                total = fx.memref_load(s_meta, fx.Int32(SMEM_META_TOTAL))
                short = arith.CmpIOp(
                    arith.CmpIPredicate.ne,
                    as_ir_value(total),
                    as_ir_value(expected_total),
                ).result
                under_cap = arith.CmpIOp(
                    arith.CmpIPredicate.slt,
                    spins,
                    as_ir_value(fx.Int32(CERTIFICATE_MAX_SPINS)),
                ).result
                scf.ConditionOp(arith.AndIOp(short, under_cap).result, [spins])
            with ir.InsertionPoint(after):
                rocdl.s_sleep(spin_sleep)
                load_global_histogram(pass_id, coherent=True)
                choose_bucket_prefix(target_k)
                scf.YieldOp(
                    [arith.AddIOp(after.arguments[0], as_ir_value(c_one)).result]
                )
            if const_expr(certificate_trace):
                spins_done = fx.Int32(w.results[0])
                seen = fx.memref_load(s_meta, fx.Int32(SMEM_META_TOTAL))
                # Read the bins once more now the loop has given up. 4.28 s is four
                # orders of magnitude past the whole kernel, so a peer that was
                # merely slow has finished long since and this comes back complete;
                # a peer that never ran leaves it short. The two answers point at
                # different bugs.
                load_global_histogram(pass_id, coherent=True)
                choose_bucket_prefix(target_k)
                truth = fx.memref_load(s_meta, fx.Int32(SMEM_META_TOTAL))
                if tid == c_zero:
                    global_atomic_max_i32(counter_slot(CERT_TRACE_SPINS), spins_done)
                    if spins_done >= fx.Int32(CERTIFICATE_MAX_SPINS):
                        global_atomic_add_i32(counter_slot(CERT_TRACE_CAPPED), c_one)
                        buffer_ops.buffer_store(
                            seen, workspace_rsrc, counter_slot(CERT_TRACE_SEEN)
                        )
                        buffer_ops.buffer_store(
                            expected_total,
                            workspace_rsrc,
                            counter_slot(CERT_TRACE_WANT),
                        )
                        buffer_ops.buffer_store(
                            truth, workspace_rsrc, counter_slot(CERT_TRACE_TRUTH)
                        )
                        buffer_ops.buffer_store(
                            fx.Int32(pass_id + 1),
                            workspace_rsrc,
                            counter_slot(CERT_TRACE_PASS),
                        )

        def flush_local_histogram(pass_id: int, publish_total: bool = False):
            if const_expr(publish_total):
                if tid == c_zero:
                    fx.memref_store(c_zero, s_meta, fx.Int32(SMEM_META_CONTRIB))
                gpu.barrier()
            for hist_idx in range(tid_idx, c_bins_idx, c_block_idx):
                hist_i32 = fx.Int32(hist_idx)
                count = fx.memref_load(s_hist, hist_i32)
                if count != c_zero:
                    global_atomic_add_i32(histogram_slot(pass_id, hist_i32), count)
                    # Accumulated in LDS rather than straight to the global word, so
                    # the part contends for it once instead of once per live bin.
                    if const_expr(publish_total):
                        lds_atomic_add_i32(
                            meta_base_ptr, fx.Int32(SMEM_META_CONTRIB), count
                        )
            # s_hist is read here and written by load_global_histogram, which follows
            # with no barrier of its own on the certified path -- the uncertified one
            # has the row barrier in between. Without this, a thread still walking its
            # bins reads back the merged global counts a faster thread has already
            # stored over them and adds those to the histogram instead of its own.
            gpu.barrier()
            if const_expr(publish_total):
                # The barrier above also orders every thread's bin atomics ahead of
                # this thread's release, which is the same pairing row_barrier uses
                # for the arrival count.
                if tid == c_zero:
                    global_atomic_add_i32(
                        counter_slot(COUNTER_CERT_TOTAL + pass_id),
                        fx.memref_load(s_meta, fx.Int32(SMEM_META_CONTRIB)),
                        llvm.AtomicOrdering.release,
                    )

        def load_global_histogram(pass_id: int, coherent: bool = False):
            # Vectorized reload. `coherent` bypasses the caches, which the
            # certificate's re-read needs and nothing else does: a barrier has
            # already made the histogram visible everywhere it is read after one.
            n_vec = num_buckets // LOAD_VEC
            c_nvec_idx = fx.Index(n_vec)
            for grp in range(tid_idx, c_nvec_idx, c_block_idx):
                base_bin = fx.Int32(grp) * c_vec
                vec = buffer_ops.buffer_load(
                    workspace_rsrc,
                    histogram_slot(pass_id, base_bin),
                    vec_width=LOAD_VEC,
                    dtype=T.i32,
                    cache_modifier=17 if coherent else 0,
                )
                for j in range_constexpr(LOAD_VEC):
                    total = vector.extract(
                        vec, static_position=[j], dynamic_position=[]
                    )
                    fx.memref_store(total, s_hist, base_bin + fx.Int32(j))
            gpu.barrier()

        def process_loaded_scan_vec(
            col_base,
            vec,
            pass_id: int,
            start_bit: int,
            previous_start_bit: int,
            current_bits,
        ):
            for j in range_constexpr(LOAD_VEC):
                col_i32 = col_base + fx.Int32(j)
                if col_i32 < row_len:
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    key = radix_twiddle_key(val)
                    matches_prefix = True
                    if const_expr(pass_id != 0):
                        matches_prefix = (
                            prefix_for_key(key, previous_start_bit) == current_bits
                        )
                    if matches_prefix:
                        lds_atomic_add_i32(
                            hist_base_ptr, bucket_for_key(key, start_bit), c_one
                        )

        def compact_scan_append_tile(
            col_base, vecs, start_bit: int, previous_start_bit: int, current_bits
        ):
            """Histogram this tile and append its candidates in column order.

            A candidate is any element whose settled prefix ranks at or above the
            digit pass 0 chose. The twiddle sends larger values to smaller keys, so
            that is prefix <= current_bits, and the comparison is unsigned via the
            sign-bit flip the ordered classifier already uses. Elements strictly
            better than the digit are already in the top-k and the emit still has to
            place them, so a buffer holding only the equal ones would lose them.
            Only the equal ones feed the next histogram.

            Offsets come from a block scan carried across tiles, the same shape as
            the ordered emit's placement, because the emit reads this buffer back in
            column order and that order has to survive the append.

            A thread takes compact_fill_vecs consecutive vec-loads rather than one, so
            the block scan -- two barriers, and a stop for the loads in flight -- is
            paid once per that many. The loads have to be the thread's own contiguous
            columns and not a strided share, because a single scan over per-thread
            totals orders candidates by thread, and only a contiguous share makes
            thread order the same thing as column order.
            """
            biased_current = current_bits ^ c_sign_bit
            keeps = []
            n_keep = c_zero
            for v in range_constexpr(compact_fill_vecs):
                vec = vecs[v]
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = col_base + fx.Int32(v * LOAD_VEC + j)
                    # Bound by the run, not the row: the tile count is rounded up to
                    # whole blocks, so the last tile of a part reaches past its run
                    # into the next part's columns, which would append and histogram
                    # them twice.
                    in_run = col_i32 < run_col_hi
                    val = vector.extract(
                        vec, static_position=[j], dynamic_position=[]
                    )
                    key = radix_twiddle_key(val)
                    prefix = prefix_for_key(key, previous_start_bit)
                    keep = in_run.select(
                        ((prefix ^ c_sign_bit) <= biased_current).select(
                            c_one, c_zero
                        ),
                        c_zero,
                    )
                    if (prefix == current_bits) & in_run:
                        lds_atomic_add_i32(
                            hist_base_ptr, bucket_for_key(key, start_bit), c_one
                        )
                    keeps.append((col_i32, key, keep))
                    n_keep = n_keep + keep

            carried = fx.memref_load(s_run, c_six)
            if const_expr(compact_early_carry):
                my_excl = compact_early_carry_scan_i32(n_keep, carried)
            elif const_expr(compact_fast_scan):
                my_excl, tile_total = compact_exclusive_scan_i32(n_keep)
            else:
                my_excl, tile_total = block_exclusive_scan_i32(n_keep)
                gpu.barrier()
            slot = part_slice_base + carried + my_excl
            for col_i32, key, keep in keeps:
                if (keep == c_one) & (slot < part_slice_end):
                    buffer_ops.buffer_store(
                        col_i32, workspace_rsrc, compact_col_slot(slot)
                    )
                    buffer_ops.buffer_store(
                        key, workspace_rsrc, compact_key_slot(slot)
                    )
                slot = slot + keep
            if const_expr(not compact_early_carry):
                if tid == c_zero:
                    total = carried + tile_total
                    fx.memref_store(total, s_run, c_six)
                    if total > part_slice_cap:
                        fx.memref_store(c_one, s_run, c_seven)
                gpu.barrier()

        def scan_vec_block(
            vblk, pass_id: int, start_bit: int, previous_start_bit: int, current_bits
        ):
            col_base = fx.Int32(vblk) * c_vec
            process_loaded_scan_vec(
                col_base,
                load_row_vec(col_base),
                pass_id,
                start_bit,
                previous_start_bit,
                current_bits,
            )

        def staged_scan_vec_blocks(
            vblk,
            pass_id: int,
            start_bit: int,
            previous_start_bit: int,
            current_bits,
            stride_idx=None,
        ):
            stride_idx = stride_idx or active_stride_idx
            if const_expr(scan_stages == 1):
                strides = [fx.Index(0)]
            elif const_expr(scan_stages == 2):
                strides = [fx.Index(0), stride_idx()]
            elif const_expr(scan_stages == 4):
                strides = [fx.Index(0)] + [stride_idx(m) for m in (1, 2, 3)]
            else:
                strides = [fx.Index(0)] + [stride_idx(m) for m in (1, 2, 3, 4, 5, 6, 7)]
            cols_v = [fx.Int32(vblk + s) * c_vec for s in strides]
            vecs = [load_row_vec(cb) for cb in cols_v]
            for cb, vc in zip(cols_v, vecs):
                process_loaded_scan_vec(
                    cb, vc, pass_id, start_bit, previous_start_bit, current_bits
                )

        def process_loaded_early_vec(col_base, vec, previous_start_bit: int, kth_bits):
            # Early-stop write: the boundary bucket after the previous pass is taken
            # whole (remaining_len == remaining_k), so every element whose prefix at
            # the previous resolution is <= the boundary prefix is in the top-k. No
            # tie-break needed. Mirrors HIP mb `previous_bits <= kth_value_bits`.
            for j in range_constexpr(LOAD_VEC):
                col_i32 = col_base + fx.Int32(j)
                if col_i32 < row_len:
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    key = radix_twiddle_key(val)
                    prefix = prefix_for_key(key, previous_start_bit)
                    if arith.cmpi(arith.CmpIPredicate.ule, prefix, kth_bits):
                        pos = global_atomic_add_i32(
                            counter_slot(COUNTER_OUT_FRONT), c_one
                        )
                        if pos < c_top_k:
                            buffer_ops.buffer_store(
                                col_i32, indices_rsrc, row_out + pos
                            )

        def early_write_vec_block(vblk, previous_start_bit: int, kth_bits):
            col_base = fx.Int32(vblk) * c_vec
            process_loaded_early_vec(
                col_base, load_row_vec(col_base), previous_start_bit, kth_bits
            )

        def early_write_all(previous_start_bit: int, kth_bits):
            # Same 4x-staged unroll as the normal last-pass write so the early-stop
            # row re-scan is not slower than the pass it replaces.
            unroll_limit_idx = vec_blocks_idx - active_stride_idx(3)
            for vblk, write_state in range(
                global_vec_tid_idx,
                unroll_limit_idx,
                active_stride_idx(4),
                init=[global_vec_tid_idx],
            ):
                for unroll_id in range_constexpr(4):
                    early_write_vec_block(
                        vblk + active_stride_idx(unroll_id),
                        previous_start_bit,
                        kth_bits,
                    )
                write_results = yield [vblk + active_stride_idx(4)]
            for vblk, write_state in range(
                write_results,
                vec_blocks_idx,
                active_stride_idx(),
                init=[c_zero],
            ):
                early_write_vec_block(vblk, previous_start_bit, kth_bits)
                write_results = yield [write_state[0]]

        def block_exclusive_scan_i32(value):
            """Exclusive prefix over tid and block total for one i32 per thread."""
            gpu.barrier()
            wave_incl = wave_inclusive_scan_i32(value)
            wave_excl_thread = wave_incl - value
            if lane == c_last_lane:
                fx.memref_store(wave_incl, s_scan, wave)
            gpu.barrier()
            if wave == c_zero:
                in_slots = lane < c_red_slots
                lane_safe = in_slots.select(lane, c_zero)
                wtot = in_slots.select(fx.memref_load(s_scan, lane_safe), c_zero)
                wincl = wave_inclusive_scan_i32(wtot)
                if in_slots:
                    fx.memref_store(wincl - wtot, s_scan, lane + c_red_slots)
            gpu.barrier()
            wave_off = fx.memref_load(s_scan, wave + c_red_slots)
            last_off = fx.memref_load(s_scan, c_last_wave + c_red_slots)
            last_tot = fx.memref_load(s_scan, c_last_wave)
            return wave_off + wave_excl_thread, last_off + last_tot

        def compact_exclusive_scan_i32(value):
            """Prefix scan for fill tiles already bracketed by a closing barrier."""
            wave_incl = wave_inclusive_scan_i32(value)
            wave_excl_thread = wave_incl - value
            if lane == c_last_lane:
                fx.memref_store(wave_incl, s_scan, wave)
            gpu.barrier()
            if wave == c_zero:
                in_slots = lane < c_red_slots
                lane_safe = in_slots.select(lane, c_zero)
                wtot = in_slots.select(
                    fx.memref_load(s_scan, lane_safe), c_zero
                )
                wincl = wave_inclusive_scan_i32(wtot)
                if in_slots:
                    fx.memref_store(
                        wincl - wtot, s_scan, lane + c_red_slots
                    )
            gpu.barrier()
            wave_off = fx.memref_load(s_scan, wave + c_red_slots)
            last_off = fx.memref_load(s_scan, c_last_wave + c_red_slots)
            last_tot = fx.memref_load(s_scan, c_last_wave)
            return wave_off + wave_excl_thread, last_off + last_tot

        def compact_early_carry_scan_i32(value, carried):
            """Prefix scan that publishes the next tile's carry at barrier 2."""
            wave_incl = wave_inclusive_scan_i32(value)
            wave_excl_thread = wave_incl - value
            if lane == c_last_lane:
                fx.memref_store(wave_incl, s_scan, wave)
            gpu.barrier()
            if wave == c_zero:
                in_slots = lane < c_red_slots
                lane_safe = in_slots.select(lane, c_zero)
                wtot = in_slots.select(
                    fx.memref_load(s_scan, lane_safe), c_zero
                )
                wincl = wave_inclusive_scan_i32(wtot)
                if in_slots:
                    fx.memref_store(
                        wincl - wtot, s_scan, lane + c_red_slots
                    )
                if lane == c_last_wave:
                    total = carried + wincl
                    fx.memref_store(total, s_run, c_six)
                    if total > part_slice_cap:
                        fx.memref_store(c_one, s_run, c_seven)
            gpu.barrier()
            wave_off = fx.memref_load(s_scan, wave + c_red_slots)
            return wave_off + wave_excl_thread

        # Bins each thread owns when reducing a whole histogram. num_buckets is
        # 1024 or 2048 against 1024 threads, so this divides exactly.
        bins_per_thread = num_buckets // block_threads

        def save_own_histogram():
            """Copy this workgroup's histogram before the row-wide merge."""
            for i in range_constexpr(bins_per_thread):
                bin_i32 = tid + fx.Int32(i * block_threads)
                fx.memref_store(fx.memref_load(s_hist, bin_i32), s_own_hist, bin_i32)

        def accumulate_run_counts(pass_id: int, chosen_bucket):
            """Fold s_own_hist lower bins into this run's selected/tied totals."""
            if ~single_part_active:
                mine = c_zero
                for i in range_constexpr(bins_per_thread):
                    bin_i32 = tid + fx.Int32(i * block_threads)
                    count = fx.memref_load(s_own_hist, bin_i32)
                    mine = mine + (bin_i32 < chosen_bucket).select(count, c_zero)
                run_total = block_exclusive_scan_i32(mine)[1]
                if tid == c_zero:
                    carried = (
                        c_zero
                        if const_expr(pass_id == 0)
                        else fx.memref_load(s_run, c_four)
                    )
                    fx.memref_store(carried + run_total, s_run, c_four)
                    if const_expr(pass_id == num_passes - 1):
                        fx.memref_store(
                            fx.memref_load(s_own_hist, chosen_bucket), s_run, c_five
                        )
                gpu.barrier()

        def ordered_classify(col_base, vec, col_hi, kth_bits):
            """Classify one loaded vec-block vs the settled kth key."""
            biased_kth = kth_bits ^ c_sign_bit
            cols = []
            n_selected = c_zero
            n_tied = c_zero
            for j in range_constexpr(LOAD_VEC):
                col_i32 = col_base + fx.Int32(j)
                val = vector.extract(vec, static_position=[j], dynamic_position=[])
                key = radix_twiddle_key(val)
                in_run = col_i32 < col_hi
                selected = in_run.select(
                    ((key ^ c_sign_bit) < biased_kth).select(c_one, c_zero), c_zero
                )
                tied = in_run.select((key == kth_bits).select(c_one, c_zero), c_zero)
                cols.append((col_i32, selected, tied))
                n_selected = n_selected + selected
                n_tied = n_tied + tied
            return cols, n_selected, n_tied

        def ordered_emit(need, kth_bits):
            """Write k ascending indices; ties on the kth value keep smallest column.

            Under compact this emits both a buffer-sourced and a row-sourced placement
            loop and gives the unused one zero trips, rather than branching around
            them. Both would otherwise have to run the cross-part base exchange, and
            that exchange contains a row barrier every part must reach exactly once.
            """
            c_block_log2 = fx.Int32(int.bit_length(block_threads) - 1)
            # Trip count is uniform across workgroups; tail runs predicate columns away.
            row_steps_all = fx.Index(
                (vblks_per_run + c_block_i32 - c_one).shrui(c_block_log2)
            )
            steps_idx = row_steps_all
            if const_expr(compact):
                # A part that overflowed its slice has no usable buffer, so it falls
                # back to its row and the buffer loop is skipped; a part that fits
                # skips the row.
                spilled = (
                    c_one == c_one
                    if const_expr(compact_fill_only)
                    else fx.memref_load(s_run, c_seven) == c_one
                )
                steps_idx = spilled.select(fx.Int32(row_steps_all), c_zero)
                steps_idx = fx.Index(steps_idx)
            col_hi = run_col_hi
            run_selected = fx.memref_load(s_run, c_four)
            run_tied = fx.memref_load(s_run, c_five)

            stages = ORDERED_STAGES
            c_stage_idx = fx.Index(stages)
            staged_limit_idx = steps_idx - fx.Index(stages - 1)

            def stage_col_base(step_i32, stage: int):
                return (
                    run_first_vblk + (step_i32 + fx.Int32(stage)) * c_block_i32 + tid
                ) * c_vec

            def stage_loads(step_i32):
                bases = [stage_col_base(step_i32, s) for s in range_constexpr(stages)]
                return bases, [load_row_vec(b) for b in bases]

            base_selected, base_tied = ordered_emit_bases(
                run_selected, run_tied, kth_bits
            )

            # Phase 2: place.
            def place_tile(col_base, vec):
                cols, n_selected, n_tied = ordered_classify(
                    col_base, vec, col_hi, kth_bits
                )
                ordered_place(cols, n_selected, n_tied, base_selected, base_tied, need)

            if const_expr(not compact):
                for step_b, place_state in range(
                    fx.Index(0), staged_limit_idx, c_stage_idx, init=[fx.Index(0)]
                ):
                    bases, vecs = stage_loads(fx.Int32(step_b))
                    for s in range_constexpr(stages):
                        place_tile(bases[s], vecs[s])
                    place_results = yield [step_b + c_stage_idx]
                for step_b, place_state in range(
                    place_results, steps_idx, fx.Index(1), init=[c_zero]
                ):
                    col_base = stage_col_base(fx.Int32(step_b), 0)
                    place_tile(col_base, load_row_vec(col_base))
                    place_results = yield [place_state[0]]
            else:
                # Staging is dropped here: the trip count is usually zero, and a
                # staged prologue would have to be predicated against that anyway.
                for step_b, place_state in range(
                    fx.Index(0), steps_idx, fx.Index(1), init=[c_zero]
                ):
                    col_base = stage_col_base(fx.Int32(step_b), 0)
                    place_tile(col_base, load_row_vec(col_base))
                    place_results = yield [place_state[0]]

                count = fx.memref_load(s_run, c_six)
                count_hi = part_slice_base + count
                c_tile = c_block_i32 * c_vec
                buf_steps = spilled.select(c_zero, (count + c_tile - c_one) // c_tile)
                for step_b, buf_state in range(
                    fx.Index(0), fx.Index(buf_steps), fx.Index(1), init=[c_zero]
                ):
                    entry_base = (
                        part_slice_base
                        + (fx.Int32(step_b) * c_block_i32 + tid) * c_vec
                    )
                    cols, n_selected, n_tied = compact_ordered_classify(
                        entry_base, count_hi, kth_bits
                    )
                    ordered_place(
                        cols, n_selected, n_tied, base_selected, base_tied, need
                    )
                    buf_results = yield [buf_state[0]]

        def collapsed_emit_bases(kth_bits):
            """Read the bases off the buffer instead of waiting for the parts to
            publish them.

            A collapsed row has no part that spilled, so the buffer holds every
            candidate the row has -- the fill keeps everything ranking at or above
            pass 0's digit, which is every element that can reach the top-k. A part
            can therefore classify the slices in front of its own against the
            settled kth key and count what they will place, which is exactly what
            the exchange would have told it. That trades a barrier every part has to
            arrive at for a walk over the front of the buffer.
            """
            biased_kth = kth_bits ^ c_sign_bit
            sel = c_zero
            tie = c_zero
            for p in range_constexpr(blocks_per_row):
                p_i32 = fx.Int32(p)
                ahead = p_i32 < part
                other_base = p_i32 * part_slice_cap
                other_count = ahead.select(
                    ws_load(counter_slot(COUNTER_COMPACT_COUNT) + p_i32), c_zero
                )
                for entry, base_state in range(
                    fx.Index(other_base + tid),
                    fx.Index(other_base + other_count),
                    c_block_idx,
                    init=[c_zero, c_zero],
                ):
                    key = ws_load(compact_key_slot(fx.Int32(entry)))
                    base_results = yield [
                        base_state[0]
                        + ((key ^ c_sign_bit) < biased_kth).select(c_one, c_zero),
                        base_state[1] + (key == kth_bits).select(c_one, c_zero),
                    ]
                sel = sel + base_results[0]
                tie = tie + base_results[1]
            sel_total = block_exclusive_scan_i32(sel)[1]
            tie_total = block_exclusive_scan_i32(tie)[1]
            if tid == c_zero:
                fx.memref_store(sel_total, s_run, c_two)
                fx.memref_store(tie_total, s_run, c_three)

        def ordered_emit_bases(run_selected, run_tied, kth_bits):
            """Exclusive scan of selected/tied counts over the parts before this one."""
            # Single-workgroup rows skip the cross-part exchange; bases stay zero.
            if tid == c_zero:
                fx.memref_store(c_zero, s_run, c_zero)
                fx.memref_store(c_zero, s_run, c_one)
                if single_part_active:
                    fx.memref_store(c_zero, s_run, c_two)
                    fx.memref_store(c_zero, s_run, c_three)

            exchange = ~single_part_active
            if const_expr(collapse):
                # Row-uniform, so a row either barriers here or none of it does.
                if ~single_part_active & collapsed_flag():
                    collapsed_emit_bases(kth_bits)
                exchange = ~single_part_active & ~collapsed_flag()

            if exchange:
                if tid == c_zero:
                    buffer_ops.buffer_store(
                        run_selected,
                        workspace_rsrc,
                        counter_slot(COUNTER_ORDERED_ABOVE) + part,
                    )
                    buffer_ops.buffer_store(
                        run_tied,
                        workspace_rsrc,
                        counter_slot(COUNTER_ORDERED_EQUAL) + part,
                    )
                row_barrier(fx.Int32(emit_barrier_token))
                if wave == c_zero:
                    in_runs = lane < active_parts
                    lane_safe = in_runs.select(lane, c_zero)
                    peer_selected = in_runs.select(
                        ws_load(counter_slot(COUNTER_ORDERED_ABOVE) + lane_safe),
                        c_zero,
                    )
                    peer_tied = in_runs.select(
                        ws_load(counter_slot(COUNTER_ORDERED_EQUAL) + lane_safe),
                        c_zero,
                    )
                    selected_incl = wave_inclusive_scan_i32(peer_selected)
                    tied_incl = wave_inclusive_scan_i32(peer_tied)
                    if lane == part:
                        fx.memref_store(selected_incl - peer_selected, s_run, c_two)
                        fx.memref_store(tied_incl - peer_tied, s_run, c_three)
            gpu.barrier()
            return (
                fx.memref_load(s_run, c_two),
                fx.memref_load(s_run, c_three),
            )

        def ordered_place(cols, n_selected, n_tied, base_selected, base_tied, need):
            """Place one classified tile at its column-ordered output positions."""
            packed_excl, packed_total = block_exclusive_scan_i32(
                arith.shli(n_selected, c_sixteen) + n_tied
            )
            carried_selected = fx.memref_load(s_run, c_zero)
            carried_tied = fx.memref_load(s_run, c_one)
            gpu.barrier()

            my_selected = (
                base_selected + carried_selected + packed_excl.shrui(c_sixteen)
            )
            my_tied = base_tied + carried_tied + (packed_excl & c_low16)
            for col_i32, selected, tied in cols:
                accepted = (my_tied < need).select(my_tied, need)
                out_pos = my_selected + accepted
                keep = selected + tied * (my_tied < need).select(c_one, c_zero)
                if (keep == c_one) & (out_pos < c_top_k):
                    buffer_ops.buffer_store(col_i32, indices_rsrc, row_out + out_pos)
                my_selected = my_selected + selected
                my_tied = my_tied + tied

            if tid == c_zero:
                fx.memref_store(
                    carried_selected + packed_total.shrui(c_sixteen), s_run, c_zero
                )
                fx.memref_store(
                    carried_tied + (packed_total & c_low16), s_run, c_one
                )
            gpu.barrier()

        def unordered_place(cols, n_selected, n_tied, need):
            """Place one classified tile, reserving its output range in one atomic.

            The emit this replaces took a global atomic per selected column, so a
            row admitted k elements through one counter and, wherever parts had
            fallen to 1, all k of those increments serialised inside a single
            workgroup. The counters stay global and per-row -- that is what lets
            parts emit without agreeing on who owns which slot -- but a tile now
            scans its own counts and spends one atomic on the block total, which is
            the trade `ordered_place` already makes against the run counts.

            Winners and ties are packed into one scan, high half and low half, for
            the same reason the ordered emit packs them: a tile can hold at most
            block_threads * LOAD_VEC * ORDERED_STAGES of either, which is far
            inside 16 bits.
            """
            packed_excl, packed_total = block_exclusive_scan_i32(
                arith.shli(n_selected, c_sixteen) + n_tied
            )
            if tid == c_zero:
                fx.memref_store(
                    global_atomic_add_i32(
                        counter_slot(COUNTER_OUT_FRONT),
                        packed_total.shrui(c_sixteen),
                    ),
                    s_run,
                    c_zero,
                )
                fx.memref_store(
                    global_atomic_add_i32(
                        counter_slot(COUNTER_OUT_BACK), packed_total & c_low16
                    ),
                    s_run,
                    c_one,
                )
            gpu.barrier()
            my_selected = fx.memref_load(s_run, c_zero) + packed_excl.shrui(c_sixteen)
            my_tied = fx.memref_load(s_run, c_one) + (packed_excl & c_low16)
            for col_i32, selected, tied in cols:
                if (selected == c_one) & (my_selected < c_top_k):
                    buffer_ops.buffer_store(
                        col_i32, indices_rsrc, row_out + my_selected
                    )
                if (tied == c_one) & (my_tied < need):
                    buffer_ops.buffer_store(
                        col_i32, indices_rsrc, row_out + c_top_k - c_one - my_tied
                    )
                my_selected = my_selected + selected
                my_tied = my_tied + tied

        def unordered_emit(need, kth_bits):
            """Row walk for the unordered emit, over a uniform number of tiles.

            The loop this replaces started every thread at its own vec block and
            stepped by the grid, so the trip count differed by one across the block
            and a block scan inside it would have been a barrier that some threads
            never reached. Every thread now runs the same tile count and the tail is
            predicated away by `col_hi`, which is what `ordered_emit` already does
            for the same reason.
            """
            stages = ORDERED_STAGES
            tile_stride = active_threads * fx.Int32(stages)
            tiles_i32 = (vec_blocks_i32 + tile_stride - c_one) // tile_stride
            for tile, place_state in range(
                fx.Index(0), fx.Index(tiles_i32), fx.Index(1), init=[c_zero]
            ):
                # The whole tile is classified under one scan, so the staging that
                # the old loop spent on memory parallelism now also amortises the
                # scan and the atomic over `stages` times as many columns.
                tile_first = global_vec_tid + fx.Int32(tile) * tile_stride
                spans = []
                for s in range_constexpr(stages):
                    vblk_i32 = tile_first + fx.Int32(s) * active_threads
                    in_row = vblk_i32 < vec_blocks_i32
                    spans.append(
                        (
                            in_row.select(vblk_i32 * c_vec, c_zero),
                            in_row.select(row_len, c_zero),
                        )
                    )
                vecs = [load_row_vec(col_base) for col_base, _ in spans]
                cols = []
                n_selected = c_zero
                n_tied = c_zero
                for (col_base, col_hi), vec in zip(spans, vecs):
                    stage_cols, stage_selected, stage_tied = ordered_classify(
                        col_base, vec, col_hi, kth_bits
                    )
                    cols = cols + stage_cols
                    n_selected = n_selected + stage_selected
                    n_tied = n_tied + stage_tied
                unordered_place(cols, n_selected, n_tied, need)
                place_results = yield [place_state[0]]

        global_vec_tid = part * c_block_i32 + tid
        global_vec_tid_idx = fx.Index(global_vec_tid)
        vec_blocks_i32 = (row_len + c_vec - c_one).shrui(fx.Int32(LOAD_VEC_LOG2))
        vec_blocks_idx = fx.Index(vec_blocks_i32)

        # ordered=True: one contiguous vec-block range per workgroup.
        vblks_per_run = (vec_blocks_i32 + active_parts - c_one) // active_parts
        run_first_vblk = part * vblks_per_run
        run_last_vblk = run_first_vblk + vblks_per_run
        run_end_vblk = (run_last_vblk < vec_blocks_i32).select(
            run_last_vblk, vec_blocks_i32
        )
        run_end_col = run_end_vblk * c_vec
        run_col_hi = (run_end_col < row_len).select(run_end_col, row_len)
        run_first_idx = fx.Index(run_first_vblk)
        run_end_idx = fx.Index(run_end_vblk)

        # Compact: each part appends only into its own equal slice of the candidate
        # buffer. Parts own ascending column runs, so slice order is column order and
        # the emit needs no cross-part compaction step -- each part reads back exactly
        # the slice it wrote.
        part_slice_cap = fx.Int32(compact_cap) // active_parts if compact else c_zero
        part_slice_base = part * part_slice_cap if compact else c_zero
        part_slice_end = part_slice_base + part_slice_cap if compact else c_zero

        def compact_ordered_classify(entry_base, count_hi, kth_bits):
            """Classify LOAD_VEC consecutive candidates against the settled kth key."""
            biased_kth = kth_bits ^ c_sign_bit
            cols = []
            n_selected = c_zero
            n_tied = c_zero
            for j in range_constexpr(LOAD_VEC):
                entry = entry_base + fx.Int32(j)
                in_buf = entry < count_hi
                # Clamp rather than predicate the load: out-of-slice entries would
                # otherwise read a neighbouring part's candidates.
                safe = in_buf.select(entry, part_slice_base)
                col_i32 = ws_load(compact_col_slot(safe))
                key = ws_load(compact_key_slot(safe))
                selected = in_buf.select(
                    ((key ^ c_sign_bit) < biased_kth).select(c_one, c_zero), c_zero
                )
                tied = in_buf.select((key == kth_bits).select(c_one, c_zero), c_zero)
                cols.append((col_i32, selected, tied))
                n_selected = n_selected + selected
                n_tied = n_tied + tied
            return cols, n_selected, n_tied

        def compact_steps_idx():
            """Tile count over a part's run, uniform across threads.

            The append block-scans, so every thread has to reach every barrier the
            same number of times. A tid-keyed loop bound would not; this walks a
            uniform trip count and predicates the tail columns away instead.
            """
            per_tile = c_block_i32 * fx.Int32(compact_fill_vecs)
            return fx.Index((vblks_per_run + per_tile - c_one) // per_tile)

        def compact_fill(start_bit: int, previous_start_bit: int, current_bits):
            """Scan the row once more, this time also writing the candidates out."""
            if tid == c_zero:
                fx.memref_store(c_zero, s_run, c_six)
                fx.memref_store(c_zero, s_run, c_seven)
            gpu.barrier()

            steps_idx = compact_steps_idx()
            c_fill_span = fx.Int32(compact_fill_vecs) * c_vec
            for step_b, fill_state in range(
                fx.Index(0), steps_idx, fx.Index(1), init=[c_zero]
            ):
                col_base = (
                    run_first_vblk * c_vec
                    + (fx.Int32(step_b) * c_block_i32 + tid) * c_fill_span
                )
                vecs = [
                    load_row_vec(col_base + fx.Int32(v * LOAD_VEC))
                    for v in range_constexpr(compact_fill_vecs)
                ]
                compact_scan_append_tile(
                    col_base,
                    vecs,
                    start_bit,
                    previous_start_bit,
                    current_bits,
                )
                fill_results = yield [fill_state[0]]

            # Surface overflow to the host: the buffer is per part, so a row is only
            # trustworthy if no part ran out of slice.
            if (tid == c_zero) & (fx.memref_load(s_run, c_seven) == c_one):
                global_atomic_add_i32(
                    compact_hdr_slot(COMPACT_HDR_OVERFLOW), c_one
                )
            if const_expr(collapse):
                if tid == c_zero:
                    buffer_ops.buffer_store(
                        fx.memref_load(s_run, c_six),
                        workspace_rsrc,
                        counter_slot(COUNTER_COMPACT_COUNT) + part,
                    )
            gpu.barrier()

        def collapse_decision():
            """Can one workgroup scan every candidate the row has left?

            If so, the passes behind the fill need no cross-part histogram merge:
            each part reads the whole buffer and settles the same digit alone. Read
            after the fill's barrier so the counts are final, and read from global
            memory only, so every part decides the same way -- a split decision
            would leave some parts waiting at a barrier the others skipped.
            """
            total = c_zero
            for p in range_constexpr(blocks_per_row):
                p_i32 = fx.Int32(p)
                cnt = ws_load(counter_slot(COUNTER_COMPACT_COUNT) + p_i32)
                total = total + (p_i32 < active_parts).select(cnt, c_zero)
            spilled_any = ws_load(compact_hdr_slot(COMPACT_HDR_OVERFLOW)) != c_zero
            # Collapsing trades one barrier for every part re-reading the buffer, so
            # it only pays while the extra reads stay under what the barrier cost.
            # The barrier grows with the arrivals it waits on, the reads with the
            # parts doing them, hence rows*parts against (parts-1)*candidates.
            reread = (active_parts - c_one) * total
            budget = (
                fx.Int32(collapse_budget) * fx.Int32(gpu.grid_dim.y) * active_parts
            )
            return (
                (total <= fx.Int32(collapse_cap))
                & (reread <= budget)
                & (~spilled_any)
            )

        def store_collapse_decision():
            fit = collapse_decision()
            if tid == c_zero:
                fx.memref_store(
                    fit.select(c_one, c_zero), s_meta, fx.Int32(SMEM_META_COLLAPSED)
                )
            gpu.barrier()

        def compact_rescan(start_bit: int, previous_start_bit: int, current_bits):
            """Histogram this part's candidates, or its row if the slice overflowed.

            Both loops are always emitted and the unused one is given zero trips, so
            that a part which spilled and a part which did not stay in lockstep at
            the barrier that follows.
            """
            spilled = (
                c_one == c_one
                if const_expr(compact_fill_only)
                else fx.memref_load(s_run, c_seven) == c_one
            )
            count = spilled.select(c_zero, fx.memref_load(s_run, c_six))
            for entry, rescan_state in range(
                fx.Index(part_slice_base + tid),
                fx.Index(part_slice_base + count),
                c_block_idx,
                init=[c_zero],
            ):
                key = ws_load(compact_key_slot(fx.Int32(entry)))
                if prefix_for_key(key, previous_start_bit) == current_bits:
                    lds_atomic_add_i32(
                        hist_base_ptr, bucket_for_key(key, start_bit), c_one
                    )
                rescan_results = yield [rescan_state[0]]

            if const_expr(collapse):
                # One branch around the whole thing, so a row that declined to
                # collapse pays a compare rather than a walk over empty ranges.
                # The flag is workgroup-uniform, which is what lets a barrier sit
                # inside the branch.
                if collapsed_flag():
                    # This part's own slice is counted; save it before folding in
                    # the others, because the ordered emit still places its own
                    # slice from its own counts, while the digit has to be settled
                    # on the whole row.
                    gpu.barrier()
                    save_own_histogram()
                    for p in range_constexpr(blocks_per_row):
                        p_i32 = fx.Int32(p)
                        mine_or_idle = (p_i32 == part) | (p_i32 >= active_parts)
                        other_base = p_i32 * part_slice_cap
                        other_count = mine_or_idle.select(
                            c_zero,
                            ws_load(counter_slot(COUNTER_COMPACT_COUNT) + p_i32),
                        )
                        for entry, borrow_state in range(
                            fx.Index(other_base + tid),
                            fx.Index(other_base + other_count),
                            c_block_idx,
                            init=[c_zero],
                        ):
                            key = ws_load(compact_key_slot(fx.Int32(entry)))
                            if prefix_for_key(key, previous_start_bit) == current_bits:
                                lds_atomic_add_i32(
                                    hist_base_ptr,
                                    bucket_for_key(key, start_bit),
                                    c_one,
                                )
                            borrow_results = yield [borrow_state[0]]

            row_stop = spilled.select(fx.Int32(run_end_idx), fx.Int32(run_first_idx))
            for vblk, fallback_state in range(
                run_first_idx + tid_idx,
                fx.Index(row_stop),
                scan_run_stride_idx(),
                init=[c_zero],
            ):
                scan_vec_block(
                    vblk, 2, start_bit, previous_start_bit, current_bits
                )
                fallback_results = yield [fallback_state[0]]
            gpu.barrier()

        def compact_write_entries(local_k, kth_bits):
            """Unordered last-pass emit over this part's candidate slice.

            The fill keeps every element whose settled prefix ranks at or above the
            digit pass 0 chose, so an element that beats the kth key is in the
            buffer too, not just the ties -- which is exactly what this emit needs
            and why it never has to look at the row again. Parts partition the row
            and both output counters are global, so the union of the slices is the
            row and no part has to know what another emitted.

            A part that overflowed its slice never wrote a trustworthy one, so it
            walks its run instead. Both loops are always emitted and the unused one
            is given zero trips, matching `compact_rescan`; nothing waits behind
            this emit, but keeping the two shapes identical keeps the spill path
            from being the one shape that never runs.

            Both loops now step by whole tiles rather than by one entry per thread,
            because `unordered_place` scans the block and a scan is a barrier every
            thread has to reach the same number of times. `spilled` is row-uniform
            within a part, so the two trip counts are block-uniform even though one
            of them is zero.
            """
            spilled = (
                c_one == c_one
                if const_expr(compact_fill_only)
                else fx.memref_load(s_run, c_seven) == c_one
            )
            count = spilled.select(c_zero, fx.memref_load(s_run, c_six))
            count_hi = part_slice_base + count
            c_tile = c_block_i32 * c_vec
            buf_steps = spilled.select(c_zero, (count + c_tile - c_one) // c_tile)
            for step_b, write_state in range(
                fx.Index(0), fx.Index(buf_steps), fx.Index(1), init=[c_zero]
            ):
                entry_base = (
                    part_slice_base + (fx.Int32(step_b) * c_block_i32 + tid) * c_vec
                )
                cols, n_selected, n_tied = compact_ordered_classify(
                    entry_base, count_hi, kth_bits
                )
                unordered_place(cols, n_selected, n_tied, local_k)
                write_results = yield [write_state[0]]

            c_block_log2 = fx.Int32(int.bit_length(block_threads) - 1)
            row_steps = spilled.select(
                (vblks_per_run + c_block_i32 - c_one).shrui(c_block_log2), c_zero
            )
            for step_b, fallback_state in range(
                fx.Index(0), fx.Index(row_steps), fx.Index(1), init=[c_zero]
            ):
                col_base = (
                    run_first_vblk + fx.Int32(step_b) * c_block_i32 + tid
                ) * c_vec
                cols, n_selected, n_tied = ordered_classify(
                    col_base, load_row_vec(col_base), run_col_hi, kth_bits
                )
                unordered_place(cols, n_selected, n_tied, local_k)
                fallback_results = yield [fallback_state[0]]

        def scan_pass(
            pass_id: int, current_k, current_bits, barrier_token: int, current_len
        ):
            start_bit = max(32 - (pass_id + 1) * bits_per_pass, 0)
            previous_start_bit = max(32 - pass_id * bits_per_pass, 0)

            clear_local_histogram()
            if const_expr(compact and pass_id == compact_fill_pass):
                compact_fill(start_bit, previous_start_bit, current_bits)
                return finish_pass(
                    pass_id, current_k, current_bits, barrier_token, current_len
                )
            if const_expr(compact and pass_id > compact_fill_pass):
                compact_rescan(start_bit, previous_start_bit, current_bits)
                return finish_pass(
                    pass_id, current_k, current_bits, barrier_token, current_len
                )

            scan_step_idx = scan_run_stride_idx if ordered else active_stride_idx
            scan_start_idx = run_first_idx + tid_idx if ordered else global_vec_tid_idx
            scan_stop_idx = run_end_idx if ordered else vec_blocks_idx
            if const_expr(scan_stages == 8):
                unroll_limit_idx = scan_stop_idx - scan_step_idx(7)
                staged_stride_idx = scan_step_idx(8)
            elif const_expr(scan_stages == 4):
                unroll_limit_idx = scan_stop_idx - scan_step_idx(3)
                staged_stride_idx = scan_step_idx(4)
            elif const_expr(scan_stages == 2):
                unroll_limit_idx = scan_stop_idx - scan_step_idx()
                staged_stride_idx = scan_step_idx(2)
            else:
                unroll_limit_idx = scan_stop_idx
                staged_stride_idx = scan_step_idx()
            for vblk, pass_state in range(
                scan_start_idx,
                unroll_limit_idx,
                staged_stride_idx,
                init=[scan_start_idx],
            ):
                staged_scan_vec_blocks(
                    vblk,
                    pass_id,
                    start_bit,
                    previous_start_bit,
                    current_bits,
                    scan_step_idx,
                )
                pass_results = yield [vblk + staged_stride_idx]
            for vblk, pass_state in range(
                pass_results,
                scan_stop_idx,
                scan_step_idx(),
                init=[c_zero],
            ):
                scan_vec_block(
                    vblk, pass_id, start_bit, previous_start_bit, current_bits
                )
                pass_results = yield [pass_state[0]]
            gpu.barrier()
            return finish_pass(
                pass_id, current_k, current_bits, barrier_token, current_len
            )

        def finish_pass(
            pass_id: int, current_k, current_bits, barrier_token: int, current_len
        ):
            """Merge the pass histogram, settle its digit, and emit on the last one.

            Split out of scan_pass because the compact passes reach it after reading
            the candidate buffer rather than the row; everything from the merge on is
            the same work either way.
            """
            start_bit = max(32 - (pass_id + 1) * bits_per_pass, 0)

            # A collapsed pass has read every candidate the row has, so its
            # histogram is already row-wide and the merge is redundant.
            merged_locally = single_part_active
            if const_expr(collapse):
                if const_expr(pass_id > compact_fill_pass):
                    merged_locally = single_part_active | collapsed_flag()
            certified = certified_pass(pass_id)

            if merged_locally:
                choose_bucket_prefix(current_k)
            if ~merged_locally:
                if const_expr(ordered):
                    save_own_histogram()
                if const_expr(not merge_off):
                    flush_local_histogram(
                        pass_id,
                        publish_total=certified and certificate_publish_total,
                    )
                # A certified pass proves the merge finished instead of waiting to
                # be told, so it takes neither the barrier nor the read-back after
                # it -- certified_merge does its own, uncached.
                if const_expr(certified):
                    certified_merge(pass_id, current_k, current_len, current_bits)
                if const_expr(not certified):
                    row_barrier(barrier_token)
                    if const_expr(not merge_off):
                        load_global_histogram(pass_id)
                    choose_bucket_prefix(current_k)

            chosen_bucket = fx.memref_load(s_meta, fx.Int32(SMEM_META_THRESHOLD))
            if const_expr(ordered):
                accumulate_run_counts(pass_id, chosen_bucket)
            next_k = fx.memref_load(s_meta, fx.Int32(SMEM_META_K))
            next_len = fx.memref_load(s_meta, fx.Int32(SMEM_META_LEN))
            next_bits = current_bits | fx.Int32(
                arith.shli(chosen_bucket, fx.Int32(start_bit))
            )
            if const_expr(pass_id == num_passes - 1):
                if const_expr(ordered):
                    ordered_emit(next_k, next_bits)
                elif const_expr(compact and pass_id > compact_fill_pass):
                    compact_write_entries(next_k, next_bits)
                else:
                    unordered_emit(next_k, next_bits)
            return next_k, next_len, next_bits

        def one_workgroup_short_tier():
            # Faithful copy of the standalone one-workgroup unordered radix-select,
            # running entirely in part 0: LDS-only histograms, a hierarchical block
            # scan, and an atomic-append write. Unlike the multi-block path it uses the
            # standalone ascending key and total-k threshold convention, and its three
            # 11/11/10-bit passes require a 2048-bin histogram (bits_per_pass == 11).
            # Reuses the persistent kernel's LDS with no extra shared memory.
            c_shift = fx.Int32(32 - 11)
            c_mid_shift = fx.Int32(10)
            c_bin_mask = fx.Int32((1 << 11) - 1)
            c_low_mask = fx.Int32((1 << 10) - 1)
            c_thirtyone = fx.Int32(31)

            def ordered_key(val):
                val = mask_nonfinite(val)
                bits = val.bitcast(T.i32)
                sign = bits.shrui(c_thirtyone)
                neg_key = ~bits
                pos_key = bits ^ c_sign_bit
                return (sign != c_zero).select(neg_key, pos_key)

            def ordered_bucket(val):
                return ordered_key(val).shrui(c_shift)

            def radix_bucket(val, shift, mask):
                return ordered_key(val).shrui(shift) & mask

            def clear_hist():
                for h in range(tid_idx, c_bins_idx, c_block_idx):
                    fx.memref_store(c_zero, s_hist, fx.Int32(h))
                gpu.barrier()

            def choose_threshold(target_k, above_slot, threshold_slot):
                # Hierarchical inclusive block scan over the 2048-bin histogram;
                # each thread owns the contiguous bin pair (2*tid, 2*tid+1). The
                # kth-largest boundary is the first bucket whose inclusive prefix
                # passes ``K' = total - target_k`` (excl <= K' < incl).
                two_tid = tid * c_two
                c0 = fx.memref_load(s_hist, two_tid)
                c1 = fx.memref_load(s_hist, two_tid + c_one)
                local_total = c0 + c1

                wave_incl = wave_inclusive_scan_i32(local_total)
                wave_excl_thread = wave_incl - local_total

                if lane == c_last_lane:
                    fx.memref_store(wave_incl, s_scan, wave)
                gpu.barrier()

                if wave == c_zero:
                    in16 = lane < c_red_slots
                    lane_safe = in16.select(lane, c_zero)
                    wtot = in16.select(fx.memref_load(s_scan, lane_safe), c_zero)
                    wincl = wave_inclusive_scan_i32(wtot)
                    wexcl = wincl - wtot
                    if in16:
                        fx.memref_store(wexcl, s_scan, lane + c_red_slots)
                gpu.barrier()

                wave_off = fx.memref_load(s_scan, wave + c_red_slots)
                last_off = fx.memref_load(s_scan, c_last_wave + c_red_slots)
                last_tot = fx.memref_load(s_scan, c_last_wave)
                total = last_off + last_tot
                kprime = total - target_k

                excl0 = wave_off + wave_excl_thread
                incl0 = excl0 + c0
                incl1 = incl0 + c1

                def emit_find(b, excl, incl):
                    crosses = (excl <= kprime) & (incl > kprime)
                    if crosses:
                        fx.memref_store(b, s_meta, threshold_slot)
                        fx.memref_store(total - incl, s_meta, above_slot)

                emit_find(two_tid, excl0, incl0)
                emit_find(two_tid + c_one, incl0, incl1)
                gpu.barrier()

            if tid == c_zero:
                for meta_slot in range_constexpr(8):
                    fx.memref_store(c_zero, s_meta, fx.Int32(meta_slot))
            gpu.barrier()

            # Per-chunk bodies for the three radix passes plus the final scatter.
            # Each takes the chunk's first column index and its already-loaded
            # vec4 and feeds a fresh HBM load through the exact same logic.
            def hist_pass1_chunk(col_base, vec):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = col_base + fx.Int32(j)
                    if col_i32 < row_len:
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
                        lds_atomic_add_i32(hist_base_ptr, ordered_bucket(val), c_one)

            def hist_pass2_chunk(col_base, vec, first_threshold):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = col_base + fx.Int32(j)
                    if col_i32 < row_len:
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
                        if ordered_bucket(val) == first_threshold:
                            lds_atomic_add_i32(
                                hist_base_ptr,
                                radix_bucket(val, c_mid_shift, c_bin_mask),
                                c_one,
                            )

            def hist_pass3_chunk(col_base, vec, first_threshold, second_threshold):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = col_base + fx.Int32(j)
                    if col_i32 < row_len:
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
                        high_bucket = ordered_bucket(val)
                        mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                        if (high_bucket == first_threshold) & (
                            mid_bucket == second_threshold
                        ):
                            lds_atomic_add_i32(
                                hist_base_ptr,
                                radix_bucket(val, c_zero, c_low_mask),
                                c_one,
                            )

            def final_scatter_chunk(
                col_base,
                vec,
                first_threshold,
                second_threshold,
                third_threshold,
                num_needed,
            ):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = col_base + fx.Int32(j)
                    if col_i32 < row_len:
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
                        high_bucket = ordered_bucket(val)
                        mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                        low_bucket = radix_bucket(val, c_zero, c_low_mask)
                        above_first = high_bucket > first_threshold
                        at_first = high_bucket == first_threshold
                        above_second = mid_bucket > second_threshold
                        at_second = mid_bucket == second_threshold
                        above_low = low_bucket > third_threshold
                        at_low = low_bucket == third_threshold
                        strictly_above = above_first | (
                            at_first & (above_second | (at_second & above_low))
                        )
                        at_boundary = at_first & at_second & at_low
                        if strictly_above:
                            pos = lds_atomic_add_i32(
                                meta_base_ptr,
                                fx.Int32(SMEM_META_SHORT_FRONT_COUNT),
                                c_one,
                            )
                            buffer_ops.buffer_store(
                                col_i32, indices_rsrc, row_out + pos
                            )
                        if at_boundary:
                            back = lds_atomic_add_i32(
                                meta_base_ptr,
                                fx.Int32(SMEM_META_SHORT_BACK_COUNT),
                                c_one,
                            )
                            if back < num_needed:
                                out_pos = c_top_k - c_one - back
                                buffer_ops.buffer_store(
                                    col_i32, indices_rsrc, row_out + out_pos
                                )

            # Reread driver: stream the whole valid row from HBM once per pass.
            # This keeps the short tier's VGPR footprint low while sharing the
            # same per-chunk logic across radix passes and final scatter.
            def reread_pass(chunk_fn):
                for vblk in range(
                    tid_idx,
                    vec_blocks_idx,
                    c_block_idx,
                ):
                    col_base = fx.Int32(vblk) * c_vec
                    chunk_fn(col_base, load_row_vec(col_base))

            # Pass 1: high 11 bits over the whole valid row.
            clear_hist()
            reread_pass(lambda cb, v: hist_pass1_chunk(cb, v))
            gpu.barrier()
            choose_threshold(
                c_top_k,
                fx.Int32(SMEM_META_SHORT_FIRST_ABOVE),
                fx.Int32(SMEM_META_SHORT_FIRST_THRESHOLD),
            )
            first_threshold = fx.memref_load(
                s_meta, fx.Int32(SMEM_META_SHORT_FIRST_THRESHOLD)
            )

            # Pass 2: mid 11 bits within the high boundary bucket.
            clear_hist()
            reread_pass(lambda cb, v: hist_pass2_chunk(cb, v, first_threshold))
            gpu.barrier()
            first_above = fx.memref_load(s_meta, fx.Int32(SMEM_META_SHORT_FIRST_ABOVE))
            need_after_first = c_top_k - first_above
            choose_threshold(
                need_after_first,
                fx.Int32(SMEM_META_SHORT_SECOND_ABOVE),
                fx.Int32(SMEM_META_SHORT_SECOND_THRESHOLD),
            )
            second_threshold = fx.memref_load(
                s_meta, fx.Int32(SMEM_META_SHORT_SECOND_THRESHOLD)
            )

            # Pass 3: low 10 bits within the high+mid boundary.
            clear_hist()
            reread_pass(
                lambda cb, v: hist_pass3_chunk(cb, v, first_threshold, second_threshold)
            )
            gpu.barrier()
            second_above = fx.memref_load(
                s_meta, fx.Int32(SMEM_META_SHORT_SECOND_ABOVE)
            )
            need_after_second = need_after_first - second_above
            choose_threshold(
                need_after_second,
                fx.Int32(SMEM_META_SHORT_THIRD_ABOVE),
                fx.Int32(SMEM_META_SHORT_THIRD_THRESHOLD),
            )
            third_threshold = fx.memref_load(
                s_meta, fx.Int32(SMEM_META_SHORT_THIRD_THRESHOLD)
            )
            third_above = fx.memref_load(s_meta, fx.Int32(SMEM_META_SHORT_THIRD_ABOVE))
            num_needed = need_after_second - third_above

            # Final phase: direct atomic-append write (LDS counters only).
            reread_pass(
                lambda cb, v: final_scatter_chunk(
                    cb,
                    v,
                    first_threshold,
                    second_threshold,
                    third_threshold,
                    num_needed,
                )
            )

        # Direct-fill: rows with row_len <= top_k (part 0 only) emit identity indices + -1.
        direct_fill = row_len <= c_top_k
        direct_fill_active = (part == c_zero) & direct_fill
        direct_fill_iters = direct_fill_active.select(fx.Index(top_k), fx.Index(0))
        for out_col in range(tid_idx, direct_fill_iters, c_block_idx):
            out_col_i32 = fx.Int32(out_col)
            valid = out_col_i32 < row_len
            out_val = valid.select(out_col_i32, c_neg_one)
            buffer_ops.buffer_store(out_val, indices_rsrc, row_out + out_col_i32)

        if const_expr(short_tier):
            short_active = single_part_active & (part == c_zero) & (row_len > c_top_k)
            if short_active:
                one_workgroup_short_tier()
            persistent_active = (
                (row_len > c_top_k) & (part < active_parts) & (~single_part_active)
            )
        else:
            persistent_active = (row_len > c_top_k) & (part < active_parts)

        if persistent_active:
            local_k = c_top_k
            local_len = row_len
            kth_bits = c_zero
            if early_stop:
                # Early termination: skip the final radix pass when the boundary bucket
                # is taken whole (remaining_len == remaining_k) and write those elements
                # directly. `early` comes from the globally merged histogram, so it is
                # identical across the row's blocks -- they enter or skip the last
                # row_barrier together, which is what keeps this deadlock-free.
                for pass_id in range_constexpr(num_passes - 1):
                    local_k, local_len, kth_bits = scan_pass(
                        pass_id, local_k, kth_bits, barrier_token_for(pass_id), local_len
                    )
                last_pass = num_passes - 1
                prev_start_bit = max(32 - last_pass * bits_per_pass, 0)
                early = local_len == local_k
                if early:
                    early_write_all(prev_start_bit, kth_bits)
                if ~early:
                    local_k, local_len, kth_bits = scan_pass(
                        last_pass,
                        local_k,
                        kth_bits,
                        barrier_token_for(last_pass),
                        local_len,
                    )
            else:
                for pass_id in range_constexpr(num_passes):
                    if const_expr(collapse):
                        if const_expr(pass_id == compact_fill_pass + 1):
                            store_collapse_decision()
                    local_k, local_len, kth_bits = scan_pass(
                        pass_id, local_k, kth_bits, barrier_token_for(pass_id), local_len
                    )

    @flyc.jit
    def launcher(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        grid_y = fx.Index(num_rows)
        topk_per_row_decode_compact_kernel(
            logits, next_n, seq_lens, indices, workspace, stride0
        ).launch(
            grid=(blocks_per_row, grid_y, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launcher

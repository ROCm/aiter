# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL decode TopK-per-row kernel, with a per-row candidate buffer

Computes an unordered Top-K index set per decode row from one persistent launch,
grid=(blocks_per_row, num_rows), picking a per-row strategy by valid length --
which is what a decode batch needs, since its sequences differ in length. Each row
derives how many of its blocks_per_row workgroups cooperate (active_parts); the
rest return immediately. The candidate buffer is what distinguishes this kernel
from the ancestor `topk_per_row_decode_tiered`: a pass writes the elements that
survive the settled digit, and the passes behind it read those instead of the row.

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
    only because the grid is flattened x-fastest: a row's parts launch contiguously
    and drain in order, which is scheduler launch order rather than a cooperative
    guarantee. Do not reorder the grid. The deadlock guard keeps
    num_rows * blocks_per_row co-resident, forcing larger batches onto the
    barrier-free short tier.
"""

import math
import os
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

# buffer_ops comes from aiter's own shim, not flydsl.expr: the flydsl cleanup in
# #4501 dropped it from the stable interface.
from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.kernels_common import create_llvm_ptr

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
# buffer_load cache policy: sc0 (bit 0) and sc1 (bit 4) together, so the load is
# system-scope coherent and reads past both L1 and L2. The certificate's re-read is
# the only thing that needs it -- every other read of the histogram follows a
# barrier, which has already made it visible.
CACHE_BYPASS_L1_L2 = 17
# Groups 0 and 1 are spare and stay reserved, so the numbering below keeps
# matching what a workspace dump shows.
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
# Slot 4 is spare.
# The live length a certified merge has to see the histogram sum to. In LDS rather
# than an SSA value because it is written by the region that knows the count and
# read by the re-read loop, which that value would not dominate.
SMEM_META_TOTAL = 5

# Compact candidate buffer, one region per row after the histograms. Entry i is the
# pair (column, twiddled key) at 2*i, so the later passes read a candidate without
# touching the row. The header is a full 128B group so the data stays aligned.
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


# The early stop is on for every card measured, so it has no shape term to fit.
# The environment variable exists for an A/B; the second name is the one the
# first harnesses used and is read so an old script cannot silently mean "on".
EARLY_STOP_DEFAULT = True
EARLY_STOP_ENV = "FLYDSL_TOPK_ADAPTIVE_ES"
_EARLY_STOP_ENV_LEGACY = "FLYDSL_TOPK_COMPACT_ES"


def early_stop_default() -> bool:
    """Whether a config built with `early_stop=None` asks for the early stop."""
    for name in (EARLY_STOP_ENV, _EARLY_STOP_ENV_LEGACY):
        asked = os.environ.get(name)
        if asked is not None:
            return asked not in ("0", "")
    return EARLY_STOP_DEFAULT


# --- the host-side rule -----------------------------------------------------
#
# Everything below decides, from the batch shape alone, how wide to launch and
# which path to run. It lives beside the kernel because every number here was
# measured against it: a caller that picks them differently gets a different
# kernel than the one the measurements describe. `decode_adaptive_config` is the
# single entry point, and the pieces are split only so each carries its reason.

PARTS_CAP = 32
# A row shorter than this reads its candidates from the row rather than from the
# buffer. The floor cannot be fitted on its own -- a compacted pass reads a slice
# of the buffer and an uncompacted one a slice of the row, and those two do not
# want the same number of parts -- so it was swept jointly with the width table.
COMPACT_MIN = ((32, 262144), (1, 1048576))

COMPACT_SLICE = 40960
"""How long a part's own slice has to be before compacting it pays, past 32 rows.

`COMPACT_MIN` is a floor on the row, fitted where a row had six or seven
workgroups. Past 32 rows a row has at most four and usually one, so the floor has
to move onto the slice: compaction saves each part two re-reads of its own slice,
which makes seq/parts the quantity and not seq.

Landing on `SHORT_MAX_CAP` is the same hardware fact from a third direction --
about 40K elements is as long a run as one workgroup will scan before it would
rather pay to avoid scanning it twice.
"""

# parts = sqrt(c2 * seq / (c1 * rows)) -- only the ratio decides a width. Fitted
# against a forced-width ladder over all 36 cells with `poll_then_acquire` on,
# which is what makes an extra participant cheap enough to want width at all.
PARTS_C1 = {False: 187.979e-3, True: 153.875e-3}
PARTS_C2 = {False: 1.87979e-3, True: 464.695e-6}

# The measured argmin width for every grid cell that reaches the multi-block path,
# read at every admitted k though fitted at 2048. Off the grid the square root
# below runs, and it is a poor fallback at narrow rows.
#
# A table and not a closed form because two cells with the same seq/rows have
# stable optima two rungs apart in opposite directions, and because the good
# widths at narrow rows form a floor one rung wide that no smooth model reproduces.
#
# Every row at 32 rows by 32K is on the short tier, so that entry decides only the
# launch width; leaving it at 2 keeps the fallback from raising it and recompiling.
# The ladder and every per-cell number behind it live in the PR description.
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
# The k the table above was fitted at. It no longer gates the lookup -- the table
# is read at every admitted k -- and survives only as `decode_adaptive_config`'s
# default k.
MEASURED_PARTS_K = 2048

WIDE_SPLIT_WORK = 28416
"""How much work a split has to take off a row's critical path to pay for itself.

Splitting a row G ways removes seq*(G-1)/G elements from its critical path and
adds one barrier, so the crossover sits where that product is constant, and the
measured crossovers at two and four parts agree on this value.

Landing on SHORT_MAX_BASE is a coincidence worth naming rather than relying on:
both say a row barrier costs about what 28K elements cost, measured from two
directions -- whether a barrier is worth taking at all, and whether a second is.
"""

# The CU count every table in this file was fitted at. It is the default rather
# than a device read because the caller knows which device the row will run on and
# this module must not import the one that answers that; pass `decode_cu_count`.
FITTED_CU = 256


def decode_adaptive_grid(
    rows: int, want: int, seq: int | None = None, cu_count: int = FITTED_CU
) -> int:
    """The launch width, from the batch size and the width the row wants.

    The odd width is deliberate: it shifts every row's starting XCD by one so a
    narrowed row stays balanced across all eight while its parts remain
    contiguous. Contiguity is load-bearing -- a row's parts must be co-resident to
    clear the spin barrier -- so do not reorder the grid to close the dead blocks.
    A row that wants the whole grid is not narrowing and keeps the full even width.

    Overshooting the CU count is the expensive direction, because a row deferred to
    a second wave holds its resident parts spinning. `cu_count` is the only device
    input here, so passing the real one ports the tables rather than refitting
    them.

    Passing `seq` is not optional in practice; it is defaulted only for callers
    older than the wide-split rule, and without it a short row past 32 rows takes a
    split that costs up to 1.18x. Past 256 rows a row always runs alone, which is
    also what keeps co-residency from binding as the batch grows.
    """
    widest = min(32, cu_count // max(1, rows))
    if widest < 2:
        return 1
    grid = (
        widest if want >= widest else max(3, widest - 1 if widest % 2 == 0 else widest)
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

    Two floors, because the grid has two regimes: up to 32 rows a row chooses its
    width and the floor is `COMPACT_MIN` on the row; past 32 it takes a share of one
    wave and the floor moves onto the part's slice, `COMPACT_SLICE`.

    The short tier keeps no candidate buffer, hence the `short_max` test -- it is
    redundant against `COMPACT_MIN` but not against the slice rule, which at one
    part crosses below any tier threshold.

    `decode_adaptive_want` must keep reading `decode_compact_uses_buffer` rather than
    this: the width table was fitted against that floor, and asking the slice rule
    there moves widths no measurement has visited.
    """
    if seq <= short_max:
        return False
    if rows > 32:
        return seq >= COMPACT_SLICE * parts
    return decode_compact_uses_buffer(rows, seq)


def decode_adaptive_want(rows: int, seq: int) -> int:
    """The width the row wants, before the grid gets a say."""
    if (rows, seq) in MEASURED_PARTS:
        return MEASURED_PARTS[(rows, seq)]
    compact = decode_compact_uses_buffer(rows, seq)
    ratio = PARTS_C2[compact] / PARTS_C1[compact]
    # Floored at two because the caps reject one at no cost: a row short enough to
    # want a single workgroup is already on the short tier, and the spare
    # workgroup returns immediately.
    return max(2, min(PARTS_CAP, round(math.sqrt(ratio * seq / rows))))


def decode_adaptive_parts(rows: int, seq: int, cu_count: int = FITTED_CU) -> int:
    """How many of the grid's workgroups per row actually take part."""
    want = decode_adaptive_want(rows, seq)
    return min(decode_adaptive_grid(rows, want, seq, cu_count), want)


SHORT_MAX_BASE, SHORT_MAX_SLOPE, SHORT_MAX_CAP = 28416, 160, 40960


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def decode_adaptive_short_max(rows: int) -> int:
    """The length below which a row runs on the single-workgroup short tier.

    Deliberately not the dispatcher's rule, which climbs six times as steeply
    because it was fitted against the ancestor tiered kernel's multi-block path.
    The two must not be reconciled until that one is re-measured. k does not enter
    here: the measured crossover moves under 800 elements across k.
    """
    return min(SHORT_MAX_CAP, SHORT_MAX_BASE + _next_pow2(rows) * SHORT_MAX_SLOPE)


def decode_adaptive_certificate(seq: int, parts: int) -> bool:
    """Whether pass 0 should settle its digit from the histogram's own total.

    It trades the arrival counter for an uncached re-read of the merged histogram.
    The saving grows with how many parts contend for the counter's cache line; the
    cost is a fixed 2048-bin read. Hence the two-part test: long rows always, 32K
    only once the grid is wide enough to have contention worth removing.
    """
    if seq >= 65536:
        return True
    return seq >= 32768 and parts >= 4


def decode_adaptive_config(
    rows: int,
    seq: int,
    k: int = MEASURED_PARTS_K,
    *,
    compact_cap_mult: int = 16,
    tiered_short_max: int | None = None,
    ordered: bool = True,
    early_stop: bool | None = None,
    cu_count: int = FITTED_CU,
) -> dict:
    """Everything `create_topk_per_row_decode_adaptive_kernel` needs for a shape.

    Returns the keyword arguments for the factory plus `parts`, `grid` and
    `compact`, which the caller needs for the workspace size and for reporting.

    Pass `cu_count` for the device that will run this, from `decode_cu_count`; the
    default is the count every table here was fitted at.

    The caps are floored at two even when the grid gives a row a single
    workgroup, because they are upper bounds: the tier takes
    `min(blocks_per_row, cap)`, so a floor of two still leaves one active part on
    a one-wide grid, while a cap of one is rejected outright as a cooperating tier
    that cannot cooperate. Without the floor every batch past 256 rows raised
    instead of running, which is a shape a decode batch arrives in.
    """
    parts = decode_adaptive_parts(rows, seq, cu_count)
    grid = decode_adaptive_grid(rows, decode_adaptive_want(rows, seq), seq, cu_count)
    short_max = (
        decode_adaptive_short_max(rows)
        if tiered_short_max is None
        else tiered_short_max
    )
    compact = decode_compact_compacts(rows, seq, parts, short_max, ordered=ordered)
    kw = {
        "blocks_per_row": grid,
        "bits_per_pass": 11,
        # A build whose rows all fit the short tier says so, rather than leaving
        # "auto" to work it out per row. At one part "auto" cannot: `active_parts`
        # is then a compile-time one, so the cooperating path folds away and a
        # long row has nowhere to go but the single-workgroup tier.
        "tier_mode": "short" if seq <= short_max else "auto",
        "tiered_mid_cap": max(2, parts),
        "tiered_long_cap": max(2, parts),
        "tiered_short_max": short_max,
        "ordered": ordered,
        # Poll the barrier with monotonic loads and take one acquire once the token
        # lands. Two thirds of what an extra participant cost was the acquire every
        # waiting workgroup issued on every unsuccessful poll.
        "poll_then_acquire": True,
    }
    if early_stop is None:
        early_stop = early_stop_default()
    # Three builds cannot reach it, and asking anyway only splits the JIT cache:
    # `ordered` and `compact` are dropped by the factory, and an all-short build
    # never runs the pass it replaces.
    if early_stop and not ordered and not compact and kw["tier_mode"] != "short":
        kw["early_stop"] = True
    if decode_adaptive_certificate(seq, parts):
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
def create_topk_per_row_decode_adaptive_kernel(
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
    early_stop: bool = False,
    ordered: bool = False,
    compact: bool = False,
    compact_cap_mult: int = 16,
    compact_fill_vecs: int = COMPACT_FILL_VECS,
    compact_fast_scan: bool = False,
    compact_early_carry: bool = False,
    spin_sleep: int = 1,
    poll_then_acquire: bool = False,
    histogram_certificate: bool = False,
) -> Any:
    """Build a launcher selecting the Top-K largest values' column indices per decode
    row, matching torch.topk by value -- an unordered set by default, ascending with
    deterministic tie-breaking under ``ordered``. The launcher is cached.

    Prefer `decode_adaptive_config` to choosing these by hand; it answers every knob
    from the shape. Constraints a caller can otherwise get wrong:

    - `bits_per_pass` is 10 or 11, and the short tier needs 11 (2048-bin histogram).
    - `scan_stages` is one of 1/2/4/8; `spin_sleep` is 0..15.
    - `mask_non_finite` off is the default *because* it matches torch.topk and the
      HIP kernel, which rank inf/NaN by raw twiddled bits. Asking for it is asking
      to diverge from both.
    - `compact_cap_mult` trades workspace for how often the buffer path is taken,
      never correctness: a row that overflows rescans instead.
    - `early_stop` is silently dropped under `ordered` or `compact`, which are the
      two ways the last pass stops being a row walk it can replace.
    - `tiered_mid_cap` / `tiered_long_cap` are clamped to `blocks_per_row`.

    The module docstring describes what each tier, the candidate buffer and the
    histogram certificate actually do.
    """
    short_max = tiered_short_max
    mid_cap = tiered_mid_cap
    mid_max = tiered_mid_max
    long_cap = tiered_long_cap

    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")
    if compact and compact_cap_mult < 1:
        raise ValueError(f"compact_cap_mult must be >= 1, got {compact_cap_mult}")
    if compact and _num_passes(bits_per_pass) < 3:
        raise ValueError(
            "compact needs a pass after the one that fills the buffer; "
            f"bits_per_pass={bits_per_pass} leaves too few passes"
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

    # ordered=True drops the short tier; it assumes the unordered emit.
    if ordered:
        short_tier = False
    else:
        short_tier = tier_mode in ("auto", "short") and bits_per_pass == 11
    # Early stop replaces the last pass with one row walk, so it needs that pass to
    # be reading the row: under `ordered` the last pass is what produces the order,
    # and under `compact` it reads the candidate buffer instead.
    early_stop = early_stop and not ordered and not compact
    # At one block per row "auto" cannot tell a short row from a row the grid could
    # only give one workgroup: `active_parts` folds to a compile-time one either
    # way, so a long row has nowhere to go but the single-workgroup tier. Only
    # there is the length read at runtime, which keeps every other build's
    # cooperating path exactly as dead or as live as it was.
    short_tier_by_len = short_tier and tier_mode == "auto" and blocks_per_row == 1
    if compact_early_carry and not compact_fast_scan:
        raise ValueError("compact_early_carry=True requires compact_fast_scan=True")
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
        conservation argument carries past pass 0. The candidate buffer is what
        ends it: from the pass after the fill, a part whose slice overflowed
        histograms its row slice instead of its candidates, and the row's total is
        then no longer the count anyone expects. An overflowing part at the fill
        itself still counts every candidate it found, it just cannot store them
        all, so the fill's histogram is conserved and the fill can certify.
        """
        if not histogram_certificate:
            return False
        if not compact:
            return True
        return pass_id <= compact_fill_pass

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
        f"topk_per_row_decode_adaptive_k{top_k}_"
        f"bpp{bits_per_pass}_g{blocks_per_row}_v2"
        f"_stage{scan_stages}"
        f"_{tier_mode}"
        f"{_cap_tag}"
        f"{'_1wg' if short_tier else ''}"
        f"{'_mf' if mask_non_finite else ''}"
        f"{'_es' if early_stop else ''}"
        f"{'_ord' if ordered else ''}"
        f"{f'_cmp{compact_cap_mult}' if compact else ''}"
        f"{f'_fv{compact_fill_vecs}' if compact else ''}"
        f"{'_fastscan' if compact and compact_fast_scan else ''}"
        f"{'_earlycarry' if compact and compact_early_carry else ''}"
        f"{f'_sl{spin_sleep}' if spin_sleep != 1 else ''}"
        f"{'_poll1acq' if poll_then_acquire else ''}"
        f"{'_cert' if histogram_certificate else ''}"
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
    def topk_per_row_decode_adaptive_kernel(
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
        if const_expr(short_tier_by_len):
            short_row = row_len <= fx.Int32(short_max)
        else:
            short_row = single_part_active

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
            # Same address arithmetic `kernels_common.atomic_add_i32` does; spelled
            # out because the atomics below need orderings it does not take.
            ptr = fx.to_llvm_ptr(fx.get_iter(workspace) + fx.Int32(elem_i32))
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def lds_i32_ptr(base, elem_i32):
            # `create_llvm_ptr` inttoptrs what it is given, so the address stays an
            # integer here; an index reaches `fly.inttoptr` as the wrong type.
            addr = fx.Int64(base) + fx.Int64(elem_i32) * fx.Int64(4)
            ptr = create_llvm_ptr(addr, address_space=3)
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

        def global_atomic_xchg_i32(elem_i32, value, ordering):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.xchg,
                global_i32_ptr(elem_i32),
                as_ir_value(value),
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_load_i32_acquire(elem_i32):
            # Volatile agent-scoped acquire load for the row-barrier spin. The
            # matching release publish below makes histogram updates visible to
            # peer workgroups without issuing a read-modify-write on the polled slot.
            return llvm.LoadOp(
                T.i32,
                global_i32_ptr(elem_i32),
                alignment=4,
                volatile_=True,
                ordering=llvm.AtomicOrdering.acquire,
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
                    llvm.AtomicOrdering.acq_rel,
                )
                last = (prev + c_one) == target_arrivals
                if last:
                    global_atomic_xchg_i32(
                        counter_slot(COUNTER_PASS_DONE),
                        token_value,
                        llvm.AtomicOrdering.release,
                    )
                else:
                    spin_until_slot_ge(counter_slot(COUNTER_PASS_DONE), token_value)
            gpu.barrier()

        def mask_nonfinite(val):
            # inf/NaN (exponent all-ones) -> -inf so they sort below every finite
            # value and are never selected.
            if const_expr(not mask_non_finite):
                return val
            bits = val.bitcast(fx.Int32)
            is_nonfinite = (bits & c_exp_mask) == c_exp_mask
            return is_nonfinite.select(c_neg_inf, val)

        def shrsi(value, amount):
            """Arithmetic right shift; the DSL wraps only the unsigned form."""
            return arith.shrsi(
                arith.as_ir_value(value), arith.as_ir_value(fx.Int32(amount))
            )

        def radix_twiddle_key(val):
            # Map larger fp32 values to smaller unsigned keys so ascending bucket
            # scans select descending values -- the bitwise complement of the
            # ascending twiddle, which is why the mask is a nor rather than an or.
            # Signed zero is left alone: torch.topk and the HIP kernel both rank
            # -0.0 strictly below +0.0, so collapsing the two here would change
            # which tied index we emit.
            bits = mask_nonfinite(val).bitcast(fx.Int32)
            return bits ^ ~(shrsi(bits, 31) | c_sign_bit)

        def bucket_for_key(key, start_bit: int):
            return (key.shrui(fx.Int32(start_bit))) & fx.Int32(num_buckets - 1)

        def prefix_for_key(key, previous_start_bit: int):
            return arith.shli(
                key.shrui(fx.Int32(previous_start_bit)),
                fx.Int32(previous_start_bit),
            )

        def load_row_vec(col_base_i32):
            return fx.Vector(
                buffer_ops.buffer_load(
                    logits_rsrc,
                    row_base + col_base_i32,
                    vec_width=LOAD_VEC,
                    dtype=T.f32,
                )
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

        def certified_merge(pass_id: int, target_k, expected_total):
            """Merge without a row barrier, using the histogram's own total as proof.

            Re-reads the bins and re-runs the bucket scan until the scan's total
            equals the row's live cardinality.

            The attempt cap is a liveness guard, not a recovery: a row that reaches
            it settles its digit from an incomplete histogram and returns a wrong
            answer that nothing catches, because no caller inspects the result.
            Reaching the cap has only ever meant a bug elsewhere.
            """
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

        def flush_local_histogram(pass_id: int):
            for hist_idx in range(tid_idx, c_bins_idx, c_block_idx):
                hist_i32 = fx.Int32(hist_idx)
                count = fx.memref_load(s_hist, hist_i32)
                if count != c_zero:
                    global_atomic_add_i32(histogram_slot(pass_id, hist_i32), count)
            # s_hist is read here and written by load_global_histogram, which follows
            # with no barrier of its own on the certified path -- the uncertified one
            # has the row barrier in between. Without this, a thread still walking its
            # bins reads back the merged global counts a faster thread has already
            # stored over them and adds those to the histogram instead of its own.
            gpu.barrier()

        def load_global_histogram(pass_id: int, coherent: bool = False):
            # Vectorized reload. `coherent` bypasses the caches, which the
            # certificate's re-read needs and nothing else does: a barrier has
            # already made the histogram visible everywhere it is read after one.
            n_vec = num_buckets // LOAD_VEC
            c_nvec_idx = fx.Index(n_vec)
            for grp in range(tid_idx, c_nvec_idx, c_block_idx):
                base_bin = fx.Int32(grp) * c_vec
                vec = fx.Vector(
                    buffer_ops.buffer_load(
                        workspace_rsrc,
                        histogram_slot(pass_id, base_bin),
                        vec_width=LOAD_VEC,
                        dtype=T.i32,
                        cache_modifier=CACHE_BYPASS_L1_L2 if coherent else 0,
                    )
                )
                for j in range_constexpr(LOAD_VEC):
                    total = vec[j]
                    fx.memref_store(total, s_hist, base_bin + fx.Int32(j))
            gpu.barrier()

        def over_chunk(col_base, body):
            """Run `body(j, col_base + j)` on each element of the vec inside the row.

            A chunk wholly inside the row, which is every chunk but the last,
            skips the per-element bounds test.
            """
            if col_base + c_vec <= row_len:
                for j in range_constexpr(LOAD_VEC):
                    body(j, col_base + fx.Int32(j))
            else:
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = col_base + fx.Int32(j)
                    if col_i32 < row_len:
                        body(j, col_i32)

        def process_loaded_scan_vec(
            col_base,
            vec,
            pass_id: int,
            start_bit: int,
            previous_start_bit: int,
            current_bits,
        ):
            def body(j, col_i32):
                val = vec[j]
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

            over_chunk(col_base, body)

        def compact_scan_append_tile(
            col_base, vecs, start_bit: int, previous_start_bit: int, current_bits
        ):
            """Histogram this tile and append its candidates in column order.

            A candidate is any element ranking at or above pass 0's digit, not only
            the ties: the elements strictly better are already in the top-k and the
            emit still has to place them, so a buffer of ties alone would lose them.
            Only the ties feed the next histogram.

            The emit reads this buffer back in column order, so the append must
            preserve it. That is why each thread takes `compact_fill_vecs`
            *contiguous* vec-loads rather than a strided share -- one scan over
            per-thread totals orders candidates by thread, and only a contiguous
            share makes thread order the same thing as column order.
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
                    val = vec[j]
                    key = radix_twiddle_key(val)
                    prefix = prefix_for_key(key, previous_start_bit)
                    keep = in_run.select(
                        ((prefix ^ c_sign_bit) <= biased_current).select(c_one, c_zero),
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
                    buffer_ops.buffer_store(key, workspace_rsrc, compact_key_slot(slot))
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
                wtot = in_slots.select(fx.memref_load(s_scan, lane_safe), c_zero)
                wincl = wave_inclusive_scan_i32(wtot)
                if in_slots:
                    fx.memref_store(wincl - wtot, s_scan, lane + c_red_slots)
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
                wtot = in_slots.select(fx.memref_load(s_scan, lane_safe), c_zero)
                wincl = wave_inclusive_scan_i32(wtot)
                if in_slots:
                    fx.memref_store(wincl - wtot, s_scan, lane + c_red_slots)
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
                val = vec[j]
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
                spilled = fx.memref_load(s_run, c_seven) == c_one
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

            base_selected, base_tied = ordered_emit_bases(run_selected, run_tied)

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
                        part_slice_base + (fx.Int32(step_b) * c_block_i32 + tid) * c_vec
                    )
                    cols, n_selected, n_tied = compact_ordered_classify(
                        entry_base, count_hi, kth_bits
                    )
                    ordered_place(
                        cols, n_selected, n_tied, base_selected, base_tied, need
                    )
                    yield [buf_state[0]]

        def ordered_emit_bases(run_selected, run_tied):
            """Exclusive scan of selected/tied counts over the parts before this one."""
            # Single-workgroup rows skip the cross-part exchange; bases stay zero.
            if tid == c_zero:
                fx.memref_store(c_zero, s_run, c_zero)
                fx.memref_store(c_zero, s_run, c_one)
                if single_part_active:
                    fx.memref_store(c_zero, s_run, c_two)
                    fx.memref_store(c_zero, s_run, c_three)

            exchange = ~single_part_active

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
                fx.memref_store(carried_tied + (packed_total & c_low16), s_run, c_one)
            gpu.barrier()

        def unordered_place(cols, n_selected, n_tied, need):
            """Place one classified tile, reserving its output range in one atomic.

            The counters stay global and per-row, which is what lets parts emit
            without agreeing on who owns which slot; the tile scans its own counts
            so only the block total costs an atomic.

            Winners and ties share one scan as high and low halves. That is safe
            because a tile holds at most block_threads * LOAD_VEC * ORDERED_STAGES
            of either, far inside 16 bits -- raising any of the three has to be
            checked against that bound.
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
                yield [place_state[0]]

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
                yield [fill_state[0]]

            # Surface overflow to the host: the buffer is per part, so a row is only
            # trustworthy if no part ran out of slice.
            if (tid == c_zero) & (fx.memref_load(s_run, c_seven) == c_one):
                global_atomic_add_i32(compact_hdr_slot(COMPACT_HDR_OVERFLOW), c_one)
            gpu.barrier()

        def compact_rescan(start_bit: int, previous_start_bit: int, current_bits):
            """Histogram this part's candidates, or its row if the slice overflowed.

            Both loops are always emitted and the unused one is given zero trips, so
            that a part which spilled and a part which did not stay in lockstep at
            the barrier that follows.
            """
            spilled = fx.memref_load(s_run, c_seven) == c_one
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
                yield [rescan_state[0]]

            row_stop = spilled.select(fx.Int32(run_end_idx), fx.Int32(run_first_idx))
            for vblk, fallback_state in range(
                run_first_idx + tid_idx,
                fx.Index(row_stop),
                scan_run_stride_idx(),
                init=[c_zero],
            ):
                scan_vec_block(vblk, 2, start_bit, previous_start_bit, current_bits)
                yield [fallback_state[0]]
            gpu.barrier()

        def compact_write_entries(local_k, kth_bits):
            """Unordered last-pass emit over this part's candidate slice.

            The buffer holds everything ranking at or above pass 0's digit, which is
            why this never looks at the row again. Parts partition the row and both
            output counters are global, so no part needs to know what another wrote.

            A part that overflowed its slice walks its run instead. Both loops are
            always emitted with the unused one given zero trips, and both step by
            whole tiles: `unordered_place` scans the block, and a scan is a barrier
            every thread must reach the same number of times. `spilled` is
            row-uniform within a part, which is what keeps the two trip counts
            block-uniform even when one is zero.
            """
            spilled = fx.memref_load(s_run, c_seven) == c_one
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
                yield [write_state[0]]

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
                yield [fallback_state[0]]

        def early_classify(col_base, vec, col_hi, previous_start_bit: int, kth_bits):
            """Classify one loaded vec-block against the previous pass's prefix.

            The boundary bucket is taken whole, so `prefix <= kth_bits` is exactly
            the top-k and nothing ties. The bias makes the DSL's signed compare
            answer the unsigned one, as `ordered_classify` does, and the arms are
            swapped because the test is the complement of a `<`.
            """
            biased_kth = kth_bits ^ c_sign_bit
            cols = []
            n_selected = c_zero
            for j in range_constexpr(LOAD_VEC):
                col_i32 = col_base + fx.Int32(j)
                val = vec[j]
                prefix = prefix_for_key(radix_twiddle_key(val), previous_start_bit)
                above = biased_kth < (prefix ^ c_sign_bit)
                selected = (col_i32 < col_hi).select(
                    above.select(c_zero, c_one), c_zero
                )
                cols.append((col_i32, selected, c_zero))
                n_selected = n_selected + selected
            return cols, n_selected

        def early_write_all(previous_start_bit: int, kth_bits, need):
            # Same tiled walk and block scan as `unordered_emit`, which is what
            # keeps the scan's barrier at a uniform trip count and costs one atomic
            # per tile rather than one per winner -- the winners are exactly top_k,
            # so a per-winner atomic scales with k and cost 1.41x at k=2048.
            stages = ORDERED_STAGES
            tile_stride = active_threads * fx.Int32(stages)
            tiles_i32 = (vec_blocks_i32 + tile_stride - c_one) // tile_stride
            for tile, place_state in range(
                fx.Index(0), fx.Index(tiles_i32), fx.Index(1), init=[c_zero]
            ):
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
                for (col_base, col_hi), vec in zip(spans, vecs):
                    stage_cols, stage_selected = early_classify(
                        col_base, vec, col_hi, previous_start_bit, kth_bits
                    )
                    cols = cols + stage_cols
                    n_selected = n_selected + stage_selected
                unordered_place(cols, n_selected, c_zero, need)
                yield [place_state[0]]

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

            merged_locally = single_part_active
            certified = certified_pass(pass_id)

            if merged_locally:
                choose_bucket_prefix(current_k)
            if ~merged_locally:
                if const_expr(ordered):
                    save_own_histogram()
                flush_local_histogram(pass_id)
                # A certified pass proves the merge finished instead of waiting to
                # be told, so it takes neither the barrier nor the read-back after
                # it -- certified_merge does its own, uncached.
                if const_expr(certified):
                    certified_merge(pass_id, current_k, current_len)
                if const_expr(not certified):
                    row_barrier(barrier_token)
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
            c_bin_bits = fx.Int32(11)
            c_low_bits = fx.Int32(10)

            def ordered_key(val):
                # The arithmetic shift broadcasts the sign, so the mask flips only
                # the sign bit of a positive value and every bit of a negative one.
                # An unsigned compare of two keys then orders the floats, which is
                # what makes the histogram's bins ascend.
                bits = mask_nonfinite(val).bitcast(fx.Int32)
                return bits ^ (shrsi(bits, 31) | c_sign_bit)

            def signed_key(val):
                # `ordered_key` biased by the sign bit, so a *signed* compare
                # answers the same ordering. Keep the two in step: the scatter
                # compares this against a threshold assembled from digits the
                # histogram produced under `ordered_key`.
                bits = mask_nonfinite(val).bitcast(fx.Int32)
                return bits ^ (shrsi(bits, 31) & ~c_sign_bit)

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
                def body(j, col_i32):
                    val = vec[j]
                    lds_atomic_add_i32(hist_base_ptr, ordered_bucket(val), c_one)

                over_chunk(col_base, body)

            def hist_pass2_chunk(col_base, vec, first_threshold):
                def body(j, col_i32):
                    val = vec[j]
                    if ordered_bucket(val) == first_threshold:
                        lds_atomic_add_i32(
                            hist_base_ptr,
                            radix_bucket(val, c_mid_shift, c_bin_mask),
                            c_one,
                        )

                over_chunk(col_base, body)

            def hist_pass3_chunk(col_base, vec, high_mid_prefix):
                def body(j, col_i32):
                    val = vec[j]
                    key = ordered_key(val)
                    if key.shrui(c_mid_shift) == high_mid_prefix:
                        lds_atomic_add_i32(hist_base_ptr, key & c_low_mask, c_one)

                over_chunk(col_base, body)

            def final_scatter_chunk(col_base, vec, kth_signed, num_needed):
                def body(j, col_i32):
                    val = vec[j]
                    key = signed_key(val)
                    strictly_above = key > kth_signed
                    at_boundary = key == kth_signed
                    if strictly_above:
                        pos = lds_atomic_add_i32(
                            meta_base_ptr,
                            fx.Int32(SMEM_META_SHORT_FRONT_COUNT),
                            c_one,
                        )
                        buffer_ops.buffer_store(col_i32, indices_rsrc, row_out + pos)
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

                over_chunk(col_base, body)

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
            # Bits 10..31 of the key that pass 3 must match, assembled once so the
            # scan compares one field instead of two digits.
            high_mid_prefix = arith.shli(first_threshold, c_bin_bits) | second_threshold
            clear_hist()
            reread_pass(lambda cb, v: hist_pass3_chunk(cb, v, high_mid_prefix))
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
            # The three settled digits are the three fields of one key. Biasing it
            # the way `signed_key` biases an element turns the scatter's ranking
            # test into a single compare.
            kth_signed = (
                arith.shli(high_mid_prefix, c_low_bits) | third_threshold
            ) ^ c_sign_bit
            reread_pass(
                lambda cb, v: final_scatter_chunk(cb, v, kth_signed, num_needed)
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
            short_active = short_row & (part == c_zero) & (row_len > c_top_k)
            if short_active:
                one_workgroup_short_tier()
            if const_expr(short_tier_by_len):
                persistent_active = (
                    (row_len > c_top_k) & (part < active_parts) & (~short_active)
                )
            else:
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
                # `early` is read off the row's merged histogram, so every block on
                # the row computes the same value and they skip the last pass
                # together. Nothing here may make it block-local: where the
                # certificate is off that last pass carries a row_barrier, and a row
                # whose blocks disagreed about reaching it would hang.
                for pass_id in range_constexpr(num_passes - 1):
                    local_k, local_len, kth_bits = scan_pass(
                        pass_id,
                        local_k,
                        kth_bits,
                        barrier_token_for(pass_id),
                        local_len,
                    )
                last_pass = num_passes - 1
                early = local_len == local_k
                if early:
                    early_write_all(
                        max(32 - last_pass * bits_per_pass, 0), kth_bits, local_k
                    )
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
                    local_k, local_len, kth_bits = scan_pass(
                        pass_id,
                        local_k,
                        kth_bits,
                        barrier_token_for(pass_id),
                        local_len,
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
        topk_per_row_decode_adaptive_kernel(
            logits, next_n, seq_lens, indices, workspace, stride0
        ).launch(
            grid=(blocks_per_row, grid_y, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launcher

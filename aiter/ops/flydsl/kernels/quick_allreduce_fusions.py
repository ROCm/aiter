# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Epilogues fused onto an all-reduce, shared by every schedule.

The one-shot, mesh and ring kernels all end a tile holding reduced values in
registers. Anything that can be computed from those before they are stored is
free bandwidth: a fused residual-add + RMSNorm turns six HBM passes (all-reduce
writes, norm reads it back) into four, and removes a launch.

Two layers, and the split matters because RMSNorm is not the last fusion this
will carry:

* **Fusion-agnostic** -- ``FUSIONS``, the row-to-workgroup geometry
  (:func:`row_block`, :func:`row_block_options`, :func:`quick_reduce_row_block`),
  the per-wave LDS partials and :func:`block_reduce_add`. Any row-local epilogue
  needs exactly these.
* **RMSNorm** -- :func:`residual_add`, :func:`rms_rstd`,
  :func:`scale_by_weight`, kept as three steps rather than one call so a caller
  can issue the ``residual_out`` store *before* the reduction's barrier instead
  of behind it.
* **RMSNorm across workgroups** -- the ``xchg_*`` helpers: the arithmetic of
  the one i64 word through which the one-shot's split build joins a row's sum
  of squares over K workgroups. The spin that waits on it lives in the kernel
  body, which is the only code the AST rewriter sees.

Why a whole workgroup per row: RMSNorm reduces over the row, so a row split
across two workgroups needs a cross-workgroup join -- which none of the
schedules has by default, and which a persistent collective kernel cannot
cheaply acquire. Sizing the block to the row instead makes the reduction local,
and for the ring it does something stronger: with one 16 B atom per row, every
chunk the ring receives is a whole number of rows, so the epilogue runs inside
the op that receives it rather than waiting for the tile to be reassembled. The
one-shot's ``split`` build is the exception: at decode sizes one workgroup per
row leaves the machine idle, so it splits the row and pays for one exchange.
"""

import math

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp

from .quick_allreduce_codec import BLOCK_ALIGN, MAX_BLOCK
from .quick_allreduce_shared import (
    ATOMS,
    WAVE,
    atom_bf16_to_f32,
    atom_f32_to_bf16,
)

#: Epilogues a kernel factory may be asked to build. ``"none"`` is the plain
#: all-reduce, and is not a special case anywhere except in the factories'
#: argument validation.
FUSIONS = ("none", "rmsnorm")

#: bf16 values in one 16 B atom, per thread.
ATOM_ELEMS = 8


# -- fusion-agnostic: geometry ------------------------------------------------


def row_block(
    hidden: int,
    *,
    per_thread: int,
    align: int = WAVE,
    max_block: int = MAX_BLOCK,
) -> int:
    """Threads per block when one workgroup must cover one token row.

    *per_thread* is the elements a thread owns of that row (``ATOM_ELEMS *
    atoms_per_row``); *align* is the width the block has to be a multiple of --
    ``WAVE`` where the only constraint is that the shuffles see whole waves, and
    ``BLOCK_ALIGN`` for the quantized schedules, where a rank-tile also has to
    land on the 64 B fabric sector grid.

    Raises ``ValueError`` naming the constraint that failed, because these
    messages reach a user through a host-side hidden gate.
    """
    hidden = int(hidden)
    if hidden <= 0 or hidden % per_thread:
        raise ValueError(
            f"fused hidden must be a positive multiple of {per_thread}, got {hidden}"
        )
    block = hidden // per_thread
    if block % align:
        raise ValueError(
            f"fused hidden={hidden} gives BLOCK={block} threads, not a multiple "
            f"of {align}; needs hidden % {align * per_thread} == 0"
        )
    if block > max_block:
        raise ValueError(
            f"fused hidden={hidden} needs BLOCK={block} threads, over the "
            f"{max_block} limit; widen per-thread coverage or split the row"
        )
    return block


def row_block_supported(hidden: int, **kwargs) -> bool:
    """Whether :func:`row_block` has an answer for *hidden*. For host gates."""
    try:
        row_block(hidden, **kwargs)
    except ValueError:
        return False
    return True


def row_block_options(
    hidden: int,
    *,
    atoms_choices,
    align: int = WAVE,
    max_block: int = MAX_BLOCK,
) -> tuple[tuple[int, int], ...]:
    """Every ``(block, atoms_per_row)`` a fused build can use for hidden dim.

    Widest block first, which is ascending ``atoms_per_row``: the fewer atoms a
    thread owns of the row, the more threads the row spreads over. ``[0]`` is
    therefore the pick every caller made before this enumerated.

    Empty when hidden dim admits no build at all.
    """
    opts = []
    for atoms_per_row in sorted({int(a) for a in atoms_choices}):
        try:
            block = row_block(
                hidden,
                per_thread=ATOM_ELEMS * atoms_per_row,
                align=align,
                max_block=max_block,
            )
        except ValueError:
            continue
        opts.append((block, atoms_per_row))
    return tuple(opts)


# -- fusion-agnostic: geometry with padding -----------------------------------
#: Largest payload a padded build may address, in bytes: the 31-bit byte-offset
#: range. The host gates reject a padded fused payload above it.
PAD_MASK_MAX_BYTES = 0x7FFFFFFF


def padded_row_block_options(
    hidden: int,
    *,
    atoms_choices,
    align: int = WAVE,
    max_block: int = MAX_BLOCK,
) -> tuple[tuple[int, int, int], ...]:
    """Every ``(block, atoms_per_row, h_pad)`` a padded fused build can use.

    Ordered by **ascending h_pad** -- least padding, hence least wire volume,
    first. Ties break on the widest block, which keeps ``[0]``.

    *hidden* must be a whole number of 16 B atoms (``ATOM_ELEMS`` bf16). That is
    what puts the pad boundary on an atom granule, which in turn makes the
    kernel's lane mask a compare against a trace-time constant rather than a
    byte-level predicate. Returns empty otherwise.
    """
    hidden = int(hidden)
    if hidden <= 0 or hidden % ATOM_ELEMS:
        return ()
    opts = []
    for atoms_per_row in sorted({int(a) for a in atoms_choices}):
        per_thread = ATOM_ELEMS * atoms_per_row
        granule = align * per_thread
        for h_pad in range(
            -(-hidden // granule) * granule, max_block * per_thread + 1, granule
        ):
            opts.append((h_pad // per_thread, atoms_per_row, h_pad))
    return tuple(sorted(opts, key=lambda bah: (bah[2], -bah[0])))


def padded_row_block(
    hidden: int,
    *,
    atoms_choices,
    align: int = WAVE,
    max_block: int = MAX_BLOCK,
) -> tuple[int, int, int]:
    """``(block, atoms_per_row, h_pad)`` for a padded build, least padding first.

    Raises ``ValueError`` naming the constraint that failed; these messages
    reach a user through a host-side hidden gate.
    """
    opts = padded_row_block_options(
        hidden, atoms_choices=atoms_choices, align=align, max_block=max_block
    )
    if opts:
        return opts[0]
    hidden = int(hidden)
    if hidden <= 0 or hidden % ATOM_ELEMS:
        raise ValueError(
            f"fused hidden must be a positive multiple of {ATOM_ELEMS} (one 16 B "
            f"atom) even with padding, got {hidden}"
        )
    widest = max(int(a) for a in atoms_choices)
    raise ValueError(
        f"no padded fused build for hidden={hidden}: the widest per-thread "
        f"coverage available here is {widest} atoms, which tops out at "
        f"{max_block * ATOM_ELEMS * widest} elements per row"
    )


#: i32 words in one 16 B atom (``ATOM_ELEMS`` bf16 = 8 halves = 4 i32).
ATOM_I32 = ATOM_ELEMS // 2


def make_rowbuf_atom_row(
    *,
    atoms_per_row: int,
    rows_per_tile: int,
    row_stride_i32: int,
    block: int,
    hidden: int,
    hbm_i32_ptr,
    hbm_row_layout,
    atom_i32: int = ATOM_I32,
    nbytes=None,
):
    """Build a padded build's per-row buffer-tensor view, shared by all schedules.

    Returns ``_rowbuf_atom_row(ptr_i64, tile, atom)`` -- a one-atom-wide row view
    (``hbm_row_layout``) over a buffer descriptor whose base is the start of
    *atom*'s column of *tile*'s row and whose ``num_records`` is the remaining
    live bytes of that row. Pad columns (element index >= hidden) exceed
    ``num_records`` for the descriptor, so a plain ``buffer_load_dwordx4`` returns
    0 and a plain ``buffer_store_dwordx4`` is dropped by the hardware -- no
    exec-mask split, no per-lane predicate arithmetic.

    A fused tile is ``rows_per_tile`` token rows laid end to end, so the atom
    index splits into a row ``r`` and an atom-within-row ``a`` via ``divmod``.
    The row base strides by the *true* width (``row_stride_i32``); the column
    inside it by the *padded* one -- which is the whole of the padding: the
    tensor's rows stay where they are, only the workgroup gets wider. One-shot
    passes ``atoms_per_row == atoms`` and ``rows_per_tile == 1``, so ``r == 0``
    and ``a == atom`` -- the same single-row geometry it had before.

    ``ptr_i64`` is the raw ``Int64`` base of the ``(M, hidden)`` operand;
    ``atom`` is a trace-time constant, so its column offset and the live-byte
    count fold at trace time and the descriptor base is one scalar add.

    *nbytes* is the runtime payload size, and bounds the **rows**. The column
    bound above says nothing about whether the row exists: a multi-row tile
    (mesh, ring) whose last tile is partial has rows past ``M``, and without
    this their descriptors point past the end of the tensor with a full row of
    live records -- loads read whatever follows it, and stores write it, which
    corrupts neighbouring allocations or faults. A dead row gets
    ``num_records = 0``, so its loads return 0 and its stores are dropped, the
    same thing the unpadded whole-payload descriptor does. ``None`` skips the
    check, for a schedule whose tiles never outrun ``M`` (the one-shot's tile is
    a single row).
    """

    def _rowbuf_atom_row(ptr_i64, tile, atom):
        r, a = divmod(atom, atoms_per_row)
        # Trace-time: column byte offset of this atom within one row, and the
        # bytes still live in the descriptor (clamped to 0 for a fully-OOB atom,
        # which least-padding selection never produces but which stays safe).
        atom_col_bytes = a * block * atom_i32 * 4
        live_bytes = max(0, hidden * 2 - atom_col_bytes)
        # Runtime: byte address of this atom's first element in this row. The row
        # index is computed in i32 (small), then widened to i64 for the byte
        # multiply so ``row * row_bytes`` cannot overflow at large M.
        row = tile * fx.Int32(rows_per_tile) + fx.Int32(r)
        row_start = fx.Int64(row) * fx.Int64(row_stride_i32 * 4)
        row_byte_off = row_start + fx.Int64(atom_col_bytes)
        records = fx.Int64(live_bytes)
        if const_expr(nbytes is not None):
            # ``select``, not ``if``: this module is outside the kernel body, so
            # the AST rewriter never sees a Python branch here.
            records = (row_start < nbytes).select(records, fx.Int64(0))
        buf_ptr = rocdl.make_buffer_ptr(
            fx.inttoptr(hbm_i32_ptr, ptr_i64 + row_byte_off),
            num_records_bytes=records,
        )
        return fx.make_view(buf_ptr, hbm_row_layout)

    return _rowbuf_atom_row


def _quick_reduce_atoms_choices(world_size: int) -> tuple[int, ...]:
    """``atoms_per_row`` values one reduce-scatter chunk can carry.

    A rank owns ``rank_atoms = ATOMS // world_size`` consecutive atoms of a
    tile and the epilogue runs on a whole chunk, so a row must fit inside one:
    ``atoms_per_row`` has to divide ``rank_atoms``.
    """
    world_size = int(world_size)
    if world_size <= 0 or ATOMS % world_size:
        raise ValueError(f"ATOMS={ATOMS} is not divisible by world_size={world_size}")
    rank_atoms = ATOMS // world_size
    return tuple(a for a in range(1, rank_atoms + 1) if rank_atoms % a == 0)


def quick_reduce_row_block_options(
    hidden: int, world_size: int
) -> tuple[tuple[int, int], ...]:
    """Every ``(block, atoms_per_row)`` a fused mesh or ring build can use."""
    return row_block_options(
        hidden,
        atoms_choices=_quick_reduce_atoms_choices(world_size),
        align=BLOCK_ALIGN,
    )


def quick_reduce_padded_row_block_options(
    hidden: int, world_size: int
) -> tuple[tuple[int, int, int], ...]:
    """Every ``(block, atoms_per_row, h_pad)`` a padded mesh or ring build can use."""
    return padded_row_block_options(
        hidden,
        atoms_choices=_quick_reduce_atoms_choices(world_size),
        align=BLOCK_ALIGN,
    )


def quick_reduce_row_block(hidden: int, world_size: int) -> tuple[int, int]:
    """``(block, atoms_per_row)`` for a fused mesh or ring build.

    The narrowest row -- the fewest atoms per row, hence the widest block --
    that both :func:`row_block` accepts and the reduce-scatter split can carry.

    At ``atoms_per_row == 1`` one atom is one row and this succeeds for every
    multiple of 1024 up to 8192. Wider rows need ``atoms_per_row > 1``, which
    TP8 cannot offer and hidden=16384 is TP2/TP4 only.
    """
    opts = quick_reduce_row_block_options(hidden, world_size)
    if opts:
        return opts[0]
    rank_atoms = ATOMS // int(world_size)
    narrowest = None
    try:
        row_block(hidden, per_thread=ATOM_ELEMS, align=BLOCK_ALIGN)
    except ValueError as exc:
        narrowest = exc
    raise ValueError(
        f"no fused build for hidden={hidden} at world_size={world_size}: a row "
        f"must be 1..{rank_atoms} atoms wide (it has to fit inside one "
        f"reduce-scatter chunk) and the resulting block a multiple of "
        f"{BLOCK_ALIGN} at most {MAX_BLOCK} threads -- {narrowest}"
    ) from narrowest


def quick_reduce_row_block_at(
    hidden: int, world_size: int, block: int | None = None
) -> tuple[int, int]:
    """``(block, atoms_per_row)`` for a fused mesh or ring build, block optional.

    ``None`` keeps :func:`quick_reduce_row_block`'s widest pick, which is what
    every caller got before the block was tunable.
    """
    if block is None:
        return quick_reduce_row_block(hidden, world_size)
    block = int(block)
    opts = quick_reduce_row_block_options(hidden, world_size)
    for b, a in opts:
        if b == block:
            return b, a
    legal = ", ".join(str(b) for b, _ in opts) if opts else "none"
    raise ValueError(
        f"no fused build for hidden={hidden} at world_size={world_size} with "
        f"block={block}: one workgroup covers one token row, so the legal "
        f"blocks at this width are {legal}"
    )


#: ``atoms_per_row`` for a fused mesh/ring build, per ``(link, world_size,
#: algorithm)``. The portable form of the block knob: which blocks exist
#: depends on the width, which ``atoms_per_row`` values exist does not.
FUSED_QR_ROW_ATOMS: dict[tuple[str, int, str], int] = {
    # PCIe: From measurements
    ("pcie", 2, "mesh"): 4,
    ("pcie", 2, "ring"): 2,
    ("pcie", 4, "mesh"): 2,
    ("pcie", 4, "ring"): 2,
    ("pcie", 8, "mesh"): 1,
    # xGMI: From measurements
    ("xgmi", 2, "ring"): 4,
    ("xgmi", 4, "mesh"): 2,
    ("xgmi", 8, "mesh"): 1,
}


def fused_qr_row_atoms(
    world_size: int, algorithm: str, link: str = "pcie"
) -> int | None:
    """Table entry for *(link, world_size, algorithm)*, or None for the default."""
    return FUSED_QR_ROW_ATOMS.get((str(link), int(world_size), str(algorithm)))


def quick_reduce_row_block_for(
    hidden: int,
    world_size: int,
    *,
    block: int | None = None,
    atoms_per_row: int | None = None,
) -> tuple[int, int]:
    """``(block, atoms_per_row)``, pinned by either knob or the widest default."""
    if block is not None and atoms_per_row is not None:
        raise ValueError("pin block or atoms_per_row, not both")
    if atoms_per_row is None:
        return quick_reduce_row_block_at(hidden, world_size, block)
    opts = quick_reduce_row_block_options(hidden, world_size)
    if not opts:
        return quick_reduce_row_block(hidden, world_size)  # raises, naming why
    want = int(atoms_per_row)
    pick = min(opts, key=lambda ba: (abs(ba[1] - want), ba[1]))
    return pick


def quick_reduce_hidden_supported(hidden: int, world_size: int) -> bool:
    """Whether a fused mesh/ring build exists for this width. For host gates."""
    try:
        quick_reduce_row_block(hidden, world_size)
    except ValueError:
        return False
    return True


# -- fusion-agnostic: block-wide reduction ------------------------------------


def make_wave_partials(n_floats: int):
    """LDS staging for a block-wide reduction: one f32 per wave per row.

    A factory rather than a struct because the count is
    ``rows * (block // WAVE)`` and both terms are build parameters. Allocate it
    **once** per kernel -- ``SharedAllocator`` is static, so calling this from
    inside a trace-time loop would emit one LDS symbol per iteration.
    """

    @fx.struct
    class WavePartials:
        wave: fx.Array[fx.Float32, max(1, n_floats), 16]

    return WavePartials


def wave_reduce_add(values):
    """Wave-wide sums of *values*, one per row, in every lane of the wave.

    A full-wave ``shuffle_xor`` butterfly: a fixed reduction tree, so every
    lane -- and every wave that reduces the same values -- gets identical bits.
    """
    sums = []
    for v in values:
        acc = v
        for sh in range_constexpr(int(math.log2(WAVE))):
            acc = acc + acc.shuffle_xor(WAVE // (2 << sh), WAVE)
        sums.append(acc)
    return sums


def block_reduce_add(values, *, tid, block: int, lds=None):
    """Block-wide sums of *values*, one per row, broadcast to every thread.

    Two levels: a full-wave ``shuffle_xor`` butterfly, then the cross-wave
    combine through LDS. What lands in LDS is a per-wave partial -- every thread
    reads all the partials back and finishes the sum itself. That redundancy is
    deliberate: it removes the broadcast, so the whole thing costs one barrier
    no matter how many rows are reduced, which is why this takes a *list*
    rather than being called once per row.

    *lds* may be ``None`` when the block is a single wave, where the butterfly
    already finished and LDS would only add a barrier. Each row gets its own
    slot, so rows within one call never alias; reusing the same *lds* for a
    later call is safe only with an intervening barrier, which every caller here
    has between tiles.

    The partial is stored by **every lane of the wave**, not by a leader. After
    a full butterfly all 64 lanes hold the same sum, so they all write the same
    value to the same address and the hardware's choice of winner does not
    matter. That is deliberate: this module is not inside a ``@flyc.kernel``
    body, so the AST rewriter never sees it and a Python ``if`` on a traced
    predicate would be evaluated as a host-side bool. Predication has to be
    expressed as ``select`` or avoided, and here it is cheaper to avoid it.
    """
    n_waves = block // WAVE
    locals_ = wave_reduce_add(values)

    if const_expr(n_waves == 1):
        return locals_

    wid = tid // fx.Int32(WAVE)
    for row in range_constexpr(len(locals_)):
        lds[fx.Int32(row * n_waves) + wid] = locals_[row]
    gpu.barrier()

    totals = []
    for row in range_constexpr(len(locals_)):
        total = None
        for w in range_constexpr(n_waves):
            v = fx.Float32(lds[fx.Int32(row * n_waves + w)])
            total = v if total is None else total + v
        totals.append(total)
    return totals


# -- RMSNorm ------------------------------------------------------------------


def residual_add(ar_f32, res_atoms):
    """``float(bf16(all_reduce)) + float(residual)``, per atom, in fp32.

    The round-trip through bf16 is deliberate and load-bearing: it is exactly
    what the *plain* path of each schedule stores, so a fused result stays
    bit-identical to "plain all-reduce into bf16, then fused_add_rmsnorm" on the
    residual output. Keeping the extra fp32 mantissa bits instead would be more
    accurate and would diverge from the kernel this replaces, compounding one
    ULP per layer.

    *ar_f32* is the reduced value as fp32 vectors -- an fp32 accumulator on the
    one-shot, a widened fp16 one on the quantized schedules. *res_atoms* are raw
    16 B bf16 atoms as loaded from HBM.
    """
    out = []
    for atom in range_constexpr(len(ar_f32)):
        x = ar_f32[atom].to(fx.BFloat16).to(fx.Float32)
        out.append(x + atom_bf16_to_f32(res_atoms[atom]))
    return out


def pack_bf16(values):
    """fp32 vectors back to raw 16 B bf16 atoms, one rounding."""
    return [atom_f32_to_bf16(v) for v in values]


def rms_local_sumsq(rows):
    """This thread's ``sum(x^2)`` per row, on the fp32 values.

    *rows* is a list of lists: the fp32 atoms of each row this thread holds.
    """
    sums = []
    for row in range_constexpr(len(rows)):
        local = None
        for atom in range_constexpr(len(rows[row])):
            sq = rows[row][atom] * rows[row][atom]
            part = fx.Float32(sq.reduce(ReductionOp.ADD))
            local = part if local is None else local + part
        sums.append(local)
    return sums


def rstd_from_total(total, eps, hidden: int):
    """``rsqrt(total / hidden + eps)``: *total* is the row's whole ``sum(x^2)``
    and *hidden* the row's whole width, however many workgroups computed it."""
    return fmath.rsqrt(total * (1.0 / hidden) + eps)


def rms_rstd(rows, eps, hidden: int, *, tid, block: int, lds=None):
    """``rsqrt(mean(x^2) + eps)`` per row, on the fp32 values.

    *rows* is a list of lists: the fp32 atoms of each row this thread holds. The
    reciprocal square root is taken on the fp32 accumulator rather than on the
    bf16 already stored to ``residual_out``, which is what the reference does.
    """
    totals = block_reduce_add(rms_local_sumsq(rows), tid=tid, block=block, lds=lds)
    return [rstd_from_total(t, eps, hidden) for t in totals]


# -- RMSNorm across workgroups: the split-row exchange word ---------------------
#
# A row split over K workgroups joins its sum of squares through one i64 word
# per (parity, row group) in HBM. Every contributing wave atomically adds
#
#     fixed(partial) * 2**XCHG_COUNT_BITS + 1
#
# so the word is ``2**8 * (sum of fixed partials) + (arrivals)`` as a single
# integer. Integer addition is order-independent, so every rank and every wave
# decodes bit-identical sums whatever order the waves arrive in -- the property
# a float atomic_add cannot give. Count and sum move in one atomic op, so a
# reader that sees the count complete sees the sum complete: no fence needed.
#
# The word is never reset. A reader keeps the value it last saw complete on
# this parity (``prev``); ``d = cur - prev`` is this call's contribution, done
# when ``d % 2**8 == n_writers``, and its sum is ``d >> 8``. Wrap-around of the
# 64-bit word is harmless: the deltas are exact modulo 2**64.

#: Fixed-point fraction bits of a partial sum of squares.
XCHG_FIX_BITS = 16
#: Low bits of the word counting arrivals; one call's writers must fit.
XCHG_COUNT_BITS = 8
#: One call's summed fixed-point partials stay below ``2**XCHG_SUM_BITS``, so
#: ``d`` stays a positive i64 (``2**8 * 2**55 = 2**63``).
XCHG_SUM_BITS = 55
#: Words live one per cache line, so row groups never share one.
XCHG_LINE_BYTES = 128


def xchg_clamp_units(n_writers: int) -> int:
    """Largest fixed-point partial one of *n_writers* may contribute.

    A power of two, so it is exact in f32 and the clamp cannot round past it;
    ``n_writers`` of them sum below ``2**XCHG_SUM_BITS``.
    """
    n = int(n_writers)
    if not 0 < n < (1 << XCHG_COUNT_BITS):
        raise ValueError(
            f"a split-row exchange counts arrivals in {XCHG_COUNT_BITS} bits, so "
            f"it takes 1..{(1 << XCHG_COUNT_BITS) - 1} writers, got {n}"
        )
    return 1 << (((1 << XCHG_SUM_BITS) - 1) // n).bit_length() - 1


def xchg_contribution(partial, n_writers: int):
    """The i64 a writer adds to the exchange word for its fp32 *partial*.

    ``partial`` is a sum of squares, so ``>= 0``. It is clamped to
    ``xchg_clamp_units`` first -- with a select on ``partial < clamp``, so a
    NaN clamps too instead of reaching the float-to-int conversion -- then
    rounded to the nearest fixed-point unit.
    """
    clamp = fx.Float32(xchg_clamp_units(n_writers) * 2.0**-XCHG_FIX_BITS)
    u = (partial < clamp).select(partial, clamp)
    q = fx.Int64(u * fx.Float32(2.0**XCHG_FIX_BITS) + fx.Float32(0.5))
    return q * fx.Int64(1 << XCHG_COUNT_BITS) + fx.Int64(1)


def xchg_arrivals(delta):
    """Writers that have arrived, from ``cur - prev`` (mod ``2**8``)."""
    return delta & fx.Int64((1 << XCHG_COUNT_BITS) - 1)


def xchg_total(delta):
    """The fp32 sum of squares a *complete* ``cur - prev`` carries."""
    units = delta // fx.Int64(1 << XCHG_COUNT_BITS)
    return fx.Float32(units) * fx.Float32(2.0**-XCHG_FIX_BITS)


def scale_by_weight(values, rstd, w_atoms):
    """``bf16(x * rstd * weight)`` per atom, the RMSNorm output."""
    out = []
    for atom in range_constexpr(len(values)):
        w = atom_bf16_to_f32(w_atoms[atom])
        out.append(atom_f32_to_bf16(values[atom] * rstd * w))
    return out

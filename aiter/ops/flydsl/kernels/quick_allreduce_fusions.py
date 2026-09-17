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
  (:func:`row_block`, :func:`quick_reduce_row_block`), the per-wave LDS
  partials and :func:`block_reduce_add`. Any row-local epilogue needs exactly
  these.
* **RMSNorm** -- :func:`residual_add`, :func:`rms_rstd`,
  :func:`scale_by_weight`, kept as three steps rather than one call so a caller
  can issue the ``residual_out`` store *before* the reduction's barrier instead
  of behind it.

Why a whole workgroup per row: RMSNorm reduces over the row, so a row split
across two workgroups could only be joined with a grid-wide barrier -- which
none of these schedules has, and which a persistent collective kernel cannot
cheaply acquire. Sizing the block to the row instead makes the reduction local,
and for the ring it does something stronger: with one 16 B atom per row, every
chunk the ring receives is a whole number of rows, so the epilogue runs inside
the op that receives it rather than waiting for the tile to be reassembled.
"""

import math

import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
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


def quick_reduce_row_block(hidden: int, world_size: int) -> tuple[int, int]:
    """``(block, atoms_per_row)`` for a fused mesh or ring build.

    Picks the narrowest row -- the fewest atoms per row, hence the widest
    block -- that both :func:`row_block` accepts and the reduce-scatter split
    can carry. A rank owns ``rank_atoms = ATOMS // world_size``
    consecutive atoms of a tile, and the epilogue runs on a whole chunk, so a
    row must fit inside one. ``atoms_per_row`` therefore has to divide
    ``rank_atoms``.

    At ``atoms_per_row == 1`` one atom is one row and this succeeds for every
    multiple of 1024 up to 8192. Wider rows need ``atoms_per_row > 1``, which
    TP8 cannot offer and hidden=16384 is TP2/TP4 only.
    """
    world_size = int(world_size)
    if world_size <= 0 or ATOMS % world_size:
        raise ValueError(f"ATOMS={ATOMS} is not divisible by world_size={world_size}")
    rank_atoms = ATOMS // world_size
    first_error = None
    for atoms_per_row in range(1, rank_atoms + 1):
        if rank_atoms % atoms_per_row:
            continue
        try:
            block = row_block(
                hidden,
                per_thread=ATOM_ELEMS * atoms_per_row,
                align=BLOCK_ALIGN,
            )
        except ValueError as exc:
            first_error = first_error or exc
            continue
        return block, atoms_per_row
    raise ValueError(
        f"no fused build for hidden={hidden} at world_size={world_size}: a row "
        f"must be 1..{rank_atoms} atoms wide (it has to fit inside one "
        f"reduce-scatter chunk) and the resulting block a multiple of "
        f"{BLOCK_ALIGN} at most {MAX_BLOCK} threads -- {first_error}"
    )


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
    locals_ = []
    for v in values:
        acc = v
        for sh in range_constexpr(int(math.log2(WAVE))):
            acc = acc + acc.shuffle_xor(WAVE // (2 << sh), WAVE)
        locals_.append(acc)

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


def rms_rstd(rows, eps, hidden: int, *, tid, block: int, lds=None):
    """``rsqrt(mean(x^2) + eps)`` per row, on the fp32 values.

    *rows* is a list of lists: the fp32 atoms of each row this thread holds. The
    reciprocal square root is taken on the fp32 accumulator rather than on the
    bf16 already stored to ``residual_out``, which is what the reference does.
    """
    sums = []
    for row in range_constexpr(len(rows)):
        local = None
        for atom in range_constexpr(len(rows[row])):
            sq = rows[row][atom] * rows[row][atom]
            part = fx.Float32(sq.reduce(ReductionOp.ADD))
            local = part if local is None else local + part
        sums.append(local)
    totals = block_reduce_add(sums, tid=tid, block=block, lds=lds)
    return [fmath.rsqrt(t * (1.0 / hidden) + eps) for t in totals]


def scale_by_weight(values, rstd, w_atoms):
    """``bf16(x * rstd * weight)`` per atom, the RMSNorm output."""
    out = []
    for atom in range_constexpr(len(values)):
        w = atom_bf16_to_f32(w_atoms[atom])
        out.append(atom_f32_to_bf16(values[atom] * rstd * w))
    return out

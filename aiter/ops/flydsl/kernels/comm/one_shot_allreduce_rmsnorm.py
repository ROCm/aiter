# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Residual-add + RMSNorm epilogue of the one-shot all-reduce.

The one-shot schedule (``one_shot_allreduce``) ends each tile holding the fp32
sum in registers and hands it to an epilogue object. The plain epilogue rounds
it to bf16 and stores it; :class:`OneShotRMSNormEpilogue` here adds the
residual, normalizes the row and stores both outputs.

In the fused schedule the tile is pinned to one token row -- or, with
``split``, to one 1/split slice of it -- because RMSNorm reduces over the row.
``atoms`` therefore sets the *block width* instead of the tile width --
BLOCK = hidden/(split*8*atoms) -- and the tile and the block count are both
independent of it. The fused ladder is ``FUSED_ONESHOT_LADDER`` in
``allreduce_policy``.

``split`` spreads each token row over that many workgroups, each owning a
contiguous ``hidden/split`` slice; they join the row's sum of squares through
one HBM exchange word (:func:`_xchg_join`). ``split=1`` is the
one-workgroup-per-row build and compiles none of it.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T

from ...allreduce_policy import SUPPORTED_ATOMS, SUPPORTED_SPLITS
from .quick_allreduce_fusions import (
    ATOM_ELEMS,
    XCHG_LINE_BYTES,
    make_wave_partials,
    pack_bf16,
    padded_row_block,
    padded_row_block_options,
    residual_add,
    resolve_h_pad,
    rms_local_sumsq,
    rms_rstd,
    row_block,
    row_block_options,
    row_block_supported,
    rstd_from_total,
    scale_by_weight,
    wave_reduce_add,
    xchg_arrivals,
    xchg_clamp_units,
    xchg_contribution,
    xchg_total,
)
from .transport import WAVE, _global_ptr

# -- geometry -----------------------------------------------------------------


def fused_block(hidden: int, atoms: int) -> int:
    """Threads per block for a fused build, or raise saying why hidden dim
    cannot covered.

    One block covers one whole token row -- ``BLOCK * atoms * 8 == hidden`` --
    because the RMSNorm reduction spans the row and a row split across two
    blocks could only be joined with a grid-wide barrier."""

    return row_block(hidden, per_thread=ATOM_ELEMS * int(atoms), align=WAVE)


def fused_hidden_supported(hidden: int, atoms: int = 1) -> bool:
    """Whether a fused build exists for this (hidden, atoms). For host-side gates."""
    return row_block_supported(hidden, per_thread=ATOM_ELEMS * int(atoms), align=WAVE)


def fused_block_options(hidden: int) -> tuple[tuple[int, int], ...]:
    """Every ``(block, atoms)`` a fused build can use at hidden dim, widest first."""
    return row_block_options(hidden, atoms_choices=SUPPORTED_ATOMS, align=WAVE)


def fused_padded_block_options(hidden: int) -> tuple[tuple[int, int, int], ...]:
    """Every ``(block, atoms, h_pad)`` a padded fused build can use, least pad first."""
    return padded_row_block_options(hidden, atoms_choices=SUPPORTED_ATOMS, align=WAVE)


def fused_padded_block(hidden: int) -> tuple[int, int, int]:
    """``(block, atoms, h_pad)`` for a padded fused build, least padding first."""
    return padded_row_block(hidden, atoms_choices=SUPPORTED_ATOMS, align=WAVE)


def fused_split_options(hidden: int) -> tuple[tuple[int, int, int], ...]:
    """Every ``(split, block, atoms)`` a native (unpadded) fused build can use
    at hidden dim, by ascending split, widest block first within one.

    ``split == 1`` is the unsplit build, i.e. ``fused_block_options``. A split
    build needs each ``hidden/split`` slice to be a whole-wave row geometry, and
    ``split * waves`` exchange writers to fit the word's arrival count.
    """
    hidden = int(hidden)
    opts = []
    for split in SUPPORTED_SPLITS:
        if hidden % split:
            continue
        for block, atoms in fused_block_options(hidden // split):
            if split > 1:
                try:
                    xchg_clamp_units(split * (block // WAVE))
                except ValueError:
                    continue
            opts.append((split, block, atoms))
    return tuple(opts)


def fused_atoms_for_block(hidden: int, block: int) -> int:
    """``atoms`` giving *block* threads at hidden dim, or raise naming the legal set."""
    block = int(block)
    opts = fused_block_options(hidden)
    for b, a in opts:
        if b == block:
            return a
    legal = ", ".join(str(b) for b, _ in opts) if opts else "none"
    raise ValueError(
        f"fused hidden={hidden} has no build at block={block}: one workgroup "
        f"covers one token row, so block is pinned to hidden/(8*atoms) for "
        f"atoms in {SUPPORTED_ATOMS} -- the legal blocks here are {legal}"
    )


# -- device helpers -----------------------------------------------------------


class _LdsWindow:
    """``base[offset + i]``: one colour's set of the per-wave LDS partials, in
    the indexing ``block_reduce_add`` uses."""

    def __init__(self, base, offset):
        self.base = base
        self.offset = offset

    def __getitem__(self, i):
        return self.base[self.offset + i]

    def __setitem__(self, i, value):
        self.base[self.offset + i] = value


@flyc.jit
def _xchg_join(word, wave_sum, prev_same, n_writers, tid):
    """Join this wave's sum of squares with every other wave of the row.

    ``n_writers`` waves -- every wave of every slice of the row -- each add
    their fixed-point sum of squares and an arrival to the row group's *word*
    for this colour (``quick_allreduce_fusions.xchg_contribution``; a wave
    whose partial is not finite or is out of range marks the word poisoned
    instead, and the row's total decodes as NaN), with a relaxed
    agent-scope 64-bit atomic whose result is left unused. Count and sum
    move in one atomic op, so a wave that sees the count complete sees the sum
    complete, and no fence orders anything. Integer addition is order-independent,
    so every wave on every rank decodes the same bits whatever order the writers
    arrived in.

    The word rotates with the inbox colour, so the group's workgroups, which
    run the same rows from the same colour, agree on it. It is never reset.
    *prev_same* is its value when this colour last completed; this call is
    done when ``cur - prev_same`` counts ``n_writers`` arrivals. That value
    cannot be overtaken: before any wave can add to this colour's word again it
    must pass the other two colours' exchanges, which need every wave's
    contribution there, which each wave makes only after it has finished
    reading this one.
    """
    if (tid & fx.Int32(WAVE - 1)) == fx.Int32(0):
        fx.atomic_add(
            word,
            xchg_contribution(wave_sum, n_writers),
            syncscope=rocdl.SyncScope.AgentOneAs,
        )
    # Every lane polls the same address: one request per wave.
    cur = fx.generic_load(
        word,
        dtype=fx.Int64,
        memory_order=fx.AtomicOrdering.Monotonic,
        syncscope=rocdl.SyncScope.AgentOneAs,
    )
    while xchg_arrivals(cur - prev_same) != fx.Int64(n_writers):
        cur = fx.generic_load(
            word,
            dtype=fx.Int64,
            memory_order=fx.AtomicOrdering.Monotonic,
            syncscope=rocdl.SyncScope.AgentOneAs,
        )
    return xchg_total(cur - prev_same), cur


# -- the epilogue -------------------------------------------------------------


class OneShotRMSNormEpilogue:
    """Residual add + RMSNorm on the one-shot's reduced tile.

    Built by the kernel factory; ``bind`` is called once inside the kernel body
    and returns the object the schedule drives (see the plain epilogue in
    ``one_shot_allreduce`` for the hook set).
    """

    fusion = "rmsnorm"

    def __init__(
        self,
        *,
        hidden: int,
        h_pad: int | None,
        split: int,
        atoms: int,
        block: int | None,
        grid: int,
        n_colors: int,
    ):
        hidden = int(hidden)
        h_pad = resolve_h_pad(hidden, h_pad)
        if split > 1:
            # The padded path bounds a per-row descriptor at the true row
            # width; splitting it is not needed by any shipped width.
            if h_pad != hidden:
                raise ValueError(
                    f"split={split} needs a native geometry, but hidden={hidden} "
                    f"runs padded to {h_pad}"
                )
            if hidden % split:
                raise ValueError(f"split={split} does not divide hidden={hidden}")
            if grid % split:
                # A row group is split consecutive workgroups, and every one of
                # them has to be launched or its siblings wait forever.
                raise ValueError(f"grid={grid} is not a multiple of split={split}")
        row_width = fused_block(h_pad // split, atoms)
        if block is not None and int(block) != row_width:
            legal = (
                fused_block_options(h_pad)
                if split == 1
                else [(b, a) for k, b, a in fused_split_options(hidden) if k == split]
            )
            raise ValueError(
                f"fused hidden={hidden} (h_pad={h_pad}, split={split}) at "
                f"atoms={atoms} needs block={row_width}, got block={block}; the "
                f"legal (block, atoms) pairs for this width are {legal}"
            )
        self.hidden = hidden
        self.h_pad = h_pad
        self.split = split
        self.block = row_width
        # Whether any lane of the workgroup sits past the end of the real row.
        self.padded = h_pad != hidden
        # i32 between one token row and the next *in HBM*. The tile the wire
        # and the workgroup see is h_pad wide, but the tensor's rows are still
        # packed at the true width, so these part company exactly when a build
        # is padded. A split build views an (M, hidden) operand as
        # (M*split, hidden/split) -- the same memory -- so its "row" here is
        # one slice.
        self.row_stride_i32 = hidden // split // 2
        self.n_colors = n_colors
        self.n_waves = self.block // WAVE
        # Split build: every wave of every slice of a row contributes one
        # partial to the row group's exchange word. Raises if the arrival count
        # overflows.
        self.n_writers = split * self.n_waves
        if split > 1:
            xchg_clamp_units(self.n_writers)
        # Row groups: ``split`` consecutive workgroups share one row.
        self.n_groups = grid // split
        # Exchange state, split builds only: one word per (colour, group), each
        # on its own line, then each workgroup's last-complete word per colour.
        self.xchg_prev_off = n_colors * self.n_groups * XCHG_LINE_BYTES
        self.xchg_bytes = self.xchg_prev_off + grid * n_colors * 8 if split > 1 else 0
        # Per-wave partials for the block-wide sum of squares, one set per
        # colour. A split build has no LDS at all: its waves reduce through HBM.
        #
        # One set per colour, because nothing else separates one tile's reads
        # of the partials from the next tile's writes: there is no barrier
        # between tiles. A wave writes a set again three tiles later, after it
        # has passed the reduction barrier of the two tiles in between, which
        # every wave reaches only once it has finished reading this one.
        self.lds_bytes = n_colors * self.n_waves * 4 if split == 1 else 0
        self._shared = (
            make_wave_partials(n_colors * self.n_waves) if split == 1 else None
        )

    @property
    def tag(self) -> str:
        # hidden sets BLOCK and the tile, so it belongs in the key just as much
        # as atoms does. h_pad changes both the geometry and which loads carry
        # a mask, so a padded build must not share a key with the plain one.
        tag = f"_rms_h{self.hidden}"
        if self.padded:
            tag += f"p{self.h_pad}"
        if self.split > 1:
            # split sets the tile and adds the exchange; block alone does not
            # name it (7168 at split=7 and 896 at split=1 differ in both).
            tag += f"_k{self.split}"
        return tag

    def spec(self) -> dict:
        return {
            "fusion": self.fusion,
            "hidden": self.hidden,
            "h_pad": self.h_pad,
            "padded": self.padded,
            "split": self.split,
            # Local (never IPC-shared) exchange state for a split build, zeroed
            # at engine build: ``_StEngine`` appends it to its meta allocation.
            "xchg_bytes": self.xchg_bytes,
        }

    def bind(
        self,
        *,
        tid,
        bid,
        io,
        out_buf,
        res_in_ptr,
        res_out_ptr,
        w_ptr,
        eps,
        xchg_ptr,
        wrap,
    ):
        """*wrap* is the schedule's colour rotation, ``color mod n_colors``."""
        return _BoundOneShotRMSNorm(
            self,
            wrap=wrap,
            tid=tid,
            bid=bid,
            io=io,
            out_buf=out_buf,
            res_in_ptr=res_in_ptr,
            res_out_ptr=res_out_ptr,
            w_ptr=w_ptr,
            eps=eps,
            xchg_ptr=xchg_ptr,
        )


class _BoundOneShotRMSNorm:
    def __init__(
        self,
        epi,
        *,
        wrap,
        tid,
        bid,
        io,
        out_buf,
        res_in_ptr,
        res_out_ptr,
        w_ptr,
        eps,
        xchg_ptr,
    ):
        self.epi = epi
        self.wrap = wrap
        self.tid = tid
        self.bid = bid
        self.io = io
        self.eps = eps
        self.xchg_ptr = xchg_ptr
        self.out_buf = out_buf
        # residual in/out are (M, hidden) bf16 exactly like the payload, so
        # they ride the same addressing.
        self.res_in_buf = io.operand(res_in_ptr)
        self.res_out_buf = io.operand(res_out_ptr)
        # The gain is a single (hidden,) row shared by every token. A padded
        # build reads it through the same per-row descriptor at row 0; an
        # unpadded build bounds a one-row buffer tensor at the true width,
        # which is the whole mask for this one operand (nothing lies past it).
        # A split build sees it as (split, hidden/split) and reads row
        # ``part``: its own slice of the gain.
        w_buf = io.operand(w_ptr, records=fx.Int64(epi.hidden * 2))

        self.sq_lds = None
        if epi.split > 1:
            # ``split`` consecutive workgroups share one token row: ``group``
            # names the row group, ``part`` this workgroup's slice. The host
            # launches a multiple of ``split`` workgroups, so tile
            # ``bid + i*n_blocks`` is row ``group + i*(n_blocks/split)``, slice
            # ``part`` -- the same row for the whole group at every i -- and
            # the schedule's tile count is equal across the group:
            # ceil((split*M - split*g - part) / (split*G)) = ceil((M - g) / G)
            # for every 0 <= part < split.
            self.group = bid // fx.Int32(epi.split)
            part = bid % fx.Int32(epi.split)
            # The gain is one row shared by every token, so it is read once
            # here rather than once per token -- a split build reads its slice.
            self.w_atoms = io.load_tile(w_buf, part)
        else:
            self.w_atoms = io.load_tile(w_buf, fx.Int32(0))
            if epi.n_waves > 1:
                # Allocated once, at kernel scope: SharedAllocator is static, so
                # an allocation reached from inside the tile loop would emit a
                # fresh LDS symbol per trace-time visit.
                self.sq_lds = fx.SharedAllocator().allocate(epi._shared).peek().wave.ptr

    def _prev_word(self, slot):
        """This workgroup's persisted exchange word for colour *slot*."""
        return _global_ptr(self.prev_base + fx.Int64(slot) * fx.Int64(8), T.i64, 8)

    def prefetch(self, tile):
        """The residual atoms of *tile*.

        Issued before the push on purpose: the residual is plain HBM with no
        dependence on any peer, so this load retires during the poll instead of
        after it.
        """
        return self.io.load_tile(self.res_in_buf, tile)

    def init_state(self, color):
        """Loop-carried state the tiles thread through: a split build's
        exchange word as each colour last completed it -- ``prev_0`` for the
        coming row's colour, ``prev_1`` and ``prev_2`` for the two after it --
        persisted per workgroup across launches. Empty otherwise."""
        if self.epi.split == 1:
            return ()
        self.prev_base = (
            self.xchg_ptr
            + fx.Int64(self.epi.xchg_prev_off)
            + fx.Int64(self.bid) * fx.Int64(self.epi.n_colors * 8)
        )
        return (
            fx.generic_load(self._prev_word(color), dtype=fx.Int64),
            fx.generic_load(
                self._prev_word(self.wrap(color + fx.Int32(1))), dtype=fx.Int64
            ),
            fx.generic_load(
                self._prev_word(self.wrap(color + fx.Int32(2))), dtype=fx.Int64
            ),
        )

    def finish(self, tile, color, accs, x_atoms, state):
        """bf16 round-trip, residual add, RMSNorm, on the reduced *accs*.

        ``residual_out`` is written *before* the row reduction, so it is in
        flight across the barrier (or, split, the exchange) rather than issued
        behind it -- it has no dependence on the norm.

        Returns the state for the next tile, which uses the next colour.
        """
        epi = self.epi
        accs = residual_add(accs, x_atoms)
        self.io.store_tile(self.res_out_buf, tile, pack_bf16(accs))
        if epi.split > 1:
            # The row spans ``split`` workgroups: reduce this wave's share,
            # then join the rest through HBM. 1/hidden is the whole row's.
            wave_sum = wave_reduce_add(rms_local_sumsq([accs]))[0]
            word = _global_ptr(
                self.xchg_ptr
                + fx.Int64(color * fx.Int32(epi.n_groups) + self.group)
                * fx.Int64(XCHG_LINE_BYTES),
                T.i64,
                8,
            )
            total, done = _xchg_join(word, wave_sum, state[0], epi.n_writers, self.tid)
            rstd = rstd_from_total(total, self.eps, epi.hidden)
            state = (state[1], state[2], done)
        else:
            # One block covers one row, so there is a single row to reduce.
            lds = None
            if self.sq_lds is not None:
                lds = _LdsWindow(self.sq_lds, color * fx.Int32(epi.n_waves))
            rstds = rms_rstd(
                [accs], self.eps, epi.hidden, tid=self.tid, block=epi.block, lds=lds
            )
            rstd = rstds[0]
        self.io.store_tile(
            self.out_buf, tile, scale_by_weight(accs, rstd, self.w_atoms)
        )
        return state

    def persist(self, color, state):
        """Called by one thread after the last tile: save the exchange words.

        Every wave decoded the same completed words, so any one thread's copy
        is the workgroup's.
        """
        if self.epi.split == 1:
            return
        fx.generic_store(self._prev_word(color), state[0])
        fx.generic_store(self._prev_word(self.wrap(color + fx.Int32(1))), state[1])
        fx.generic_store(self._prev_word(self.wrap(color + fx.Int32(2))), state[2])


__all__ = [
    "OneShotRMSNormEpilogue",
    "fused_atoms_for_block",
    "fused_block",
    "fused_block_options",
    "fused_hidden_supported",
    "fused_padded_block",
    "fused_padded_block_options",
    "fused_split_options",
]

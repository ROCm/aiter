# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The plain end of the quantized (mesh, ring) all-reduce schedules.

The mesh and ring kernels touch HBM only through an epilogue object, so a fused
variant is a different object rather than a branch in the schedule. This module
holds the plain one -- bf16 in, bf16 out through ``_payload_io`` -- and
documents the hooks; ``quick_allreduce_rmsnorm`` holds the residual-add +
RMSNorm ones.

At build time an epilogue fixes ``block`` and the HBM geometry (``spec()``,
``tag``, ``hbm_tile_bytes``, ``partial_floats`` -- per-wave LDS partials the
kernel allocates for it). ``takes_operands`` picks the launcher: the plain one
passes zeros for the residual, weight, eps, Gemma bias and MXFP4 output
arguments.

In the kernel body ``bind`` returns an object with:

* ``load_atom(tile, atom)`` -- this rank's input atom, as packed fp16.
* Mesh: ``stash_own(tile, atom, value)`` / ``unstash_own(tile, atom)`` park
  this rank's reduced chunk between the super-tile loops (``unstash_own``
  returns None where the stash was already the final store), and
  ``finish_tile(tile, gathered, pre)`` disposes of a whole gathered tile
  (``None`` marks atoms already stored). ``prefetch(tile)`` runs once the
  tile's input is loaded and before its reduce-scatter, and returns what
  ``finish_tile`` takes as *pre* (``None`` when there is nothing to hoist); only
  the single-super-tile loop calls it, so an epilogue that prefetches has to
  be built for ``super_tile=1``.
* Ring: ``begin_chunk(tile, chunk, recv)`` runs before a completing op's
  receive, ``complete(tile, atom, value, recv)`` takes each finished atom, and
  ``end_chunk(tile, chunk, recv)`` runs after the op's fanout. *recv* is a list
  the kernel owns per op, for the epilogue's own use.

Hooks run inline in the kernel trace, so they hold no runtime ``if`` or loop of
their own -- those belong to the kernel body or a ``@flyc.jit`` helper, the only
code the AST rewriter sees.
"""

from flydsl.expr import range_constexpr

from ...allreduce_policy import SUPPORTED_BLOCKS
from .quick_allreduce_codec import _atom_bf16_to_f16, _atom_f16_to_bf16
from .transport import ATOMS, _payload_io


class QuickReducePlainEpilogue:
    """The plain all-reduce: every finished atom goes straight to ``out``."""

    fusion = "none"
    takes_operands = False
    padded = False
    atoms_per_row = 1
    partial_floats = 0
    tag = ""

    def __init__(self, *, block: int | None, world_size: int):
        if block is None:
            raise ValueError("a plain build requires block")
        if block not in SUPPORTED_BLOCKS:
            raise ValueError(f"block must be one of {SUPPORTED_BLOCKS}, got {block!r}")
        self.block = int(block)
        self.rows_per_tile = ATOMS
        self.rows_per_chunk = ATOMS // world_size
        self.hbm_tile_bytes = self.block * ATOMS * 16

    def spec(self) -> dict:
        return {
            "fusion": self.fusion,
            "hidden": None,
            "h_pad": None,
            "padded": False,
            "atoms_per_row": self.atoms_per_row,
            "rows_per_tile": self.rows_per_tile,
        }

    def bind(self, *, tid, nbytes, num_tiles, inp_ptr, out_ptr, **_unused):
        return _BoundPlain(
            *_payload_io(
                inp_ptr,
                out_ptr,
                nbytes,
                num_tiles,
                ATOMS,
                self.block,
                tid,
                decode=_atom_bf16_to_f16,
                encode=_atom_f16_to_bf16,
            )
        )


class _BoundPlain:
    def __init__(self, load, store):
        self.load_atom = load
        self._store = store

    def stash_own(self, tile, atom, value):
        """A plain build's store here is already its final one."""
        self._store(tile, atom, value)

    def unstash_own(self, tile, atom):
        return None

    def prefetch(self, tile):
        return None

    def finish_tile(self, tile, gathered, pre=None):
        for atom in range_constexpr(ATOMS):
            if gathered[atom] is not None:
                self._store(tile, atom, gathered[atom])

    def begin_chunk(self, tile, chunk, recv):
        pass

    def complete(self, tile, atom, value, recv):
        self._store(tile, atom, value)

    def end_chunk(self, tile, chunk, recv):
        pass

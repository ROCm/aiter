# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 TDM (Tensor Data Mover) descriptors over RAW 64-bit addresses.

``tdm_ops.make_tensor_descriptor_2d`` derives the global base from a fly
memref/tensor, which the EP dispatch cannot supply: a payload's destination is a
*peer's* copy of a symmetric region, i.e. the integer ``lsa_ptr(pe, off)``, a
runtime value rather than a kernel tensor argument. So GROUP0 is packed by hand
from an i64 address plus an LDS byte address, and GROUP1 from the compile-time
tile geometry. The bit layout is copied from upstream and must track it:

  GROUP0 (vector<4xi32>)   lane0 pred | lane1 lds byte addr | lane2/3 global lo/hi
                           (lane3 bit31 = descriptor type field)
  GROUP1 (vector<8xi32>)   s0 data_size[17:16], pad/mcast bits left 0
                           s1 tensor_dim0[15:0] << 16
                           s2 tensor_dim0[31:16] | tensor_dim1[15:0] << 16
                           s3 tensor_dim1[31:16] | tile_dim0 << 16
                           s4 tile_dim1[15:0]
                           s5 tensor_dim0_stride[31:0]
                           s6 tensor_dim0_stride[47:32] | tensor_dim1_stride[15:0] << 16
                           s7 tensor_dim1_stride[31:16]

dim0 is the innermost extent, dim1 the row count. ``s5`` is named
``tensor_dim0_stride`` by the ISA but is the stride *between* rows (flydsl's
``outer_stride``). ``tensor_dim1_stride`` must be set to ``dim1``: upstream
leaves s6/s7 zero, fine for its own dim1 >= 2 tiles, but a ``dim1 == 1``
descriptor with a zero dim1 stride hangs the engine -- and the payload move is
always 1 x hidden_dim. ``tensor_dim == tile_dim`` is the "OOB checking off"
encoding, so the caller guarantees the run is in bounds. Padding, the atomic
barrier and multicast are never enabled: each changes the transfer protocol and
none is wanted for a point-to-point copy.

The engine commits global memory in whole 128-byte blocks. A store that covers
only part of its first or last block still rewrites the rest of it out of the
tile, so a run must begin and end on a 128B boundary unless nothing else is
writing the blocks it straddles. :func:`tdm_plan_xfer_4b` is how a caller with
arbitrary endpoints gets there: it peels the unaligned ends off for the caller
to move by hand and leaves a body the engine can own outright.
"""

from __future__ import annotations

import math

from flydsl._mlir import ir
from flydsl.expr import arith
from flydsl.expr.rocdl import readfirstlane, tdm_ops
from flydsl.expr.typing import T

from . import vector

#: Widest ISA dim/tile field. Both extents and the row stride are 16-bit in the
#: parts of GROUP1 a 2D descriptor actually uses.
TDM_MAX_DIM = 0xFFFF

#: A TDM row narrower than this moves at a fraction of peak: the engine issues
#: one descriptor-driven burst per row, so short rows pay the per-row overhead
#: on every one of them. It is a bandwidth floor, not a legality bound -- a
#: 1-row / sub-128B transfer is well-formed and lands correctly.
TDM_ROW_FLOOR_BYTES = 128

#: 128B / 4B: one row in elements for the 4-byte metadata runs.
TDM_ROW_ELEMS_4B = TDM_ROW_FLOOR_BYTES // 4

_TDM_ROW_LOG2_4B = TDM_ROW_ELEMS_4B.bit_length() - 1


def _i32(v):
    """i32 constant from a bit pattern that may set bit 31 (arith wants signed)."""
    v &= 0xFFFFFFFF
    if v >= 1 << 31:
        v -= 1 << 32
    return arith.constant(v, type=T.i32)


def tdm_group1(dim0, dim1, elem_bytes, stride=None):
    """GROUP1 for a ``dim1`` x ``dim0`` tile of ``elem_bytes`` elements.

    ``dim0`` is the contiguous extent of one row and ``stride`` (default
    ``dim0``, i.e. a dense run) the element distance between rows. All three are
    compile-time, so the whole descriptor folds to eight SGPR constants.
    """
    if elem_bytes not in (1, 2, 4, 8):
        raise ValueError(
            f"TDM data_size encodes 1/2/4/8-byte elements, got {elem_bytes}"
        )
    if not (1 <= dim0 <= TDM_MAX_DIM and 1 <= dim1 <= TDM_MAX_DIM):
        raise ValueError(
            f"TDM tile dims must be in [1, {TDM_MAX_DIM}], got {dim1}x{dim0}"
        )
    if stride is None:
        stride = dim0
    ds = int(math.log2(elem_bytes))
    return vector.from_elements(
        T.vec(8, T.i32),
        [
            _i32(ds << 16),
            _i32((dim0 & 0xFFFF) << 16),
            _i32(((dim0 >> 16) & 0xFFFF) | ((dim1 & 0xFFFF) << 16)),
            _i32(((dim1 >> 16) & 0xFFFF) | ((dim0 & 0xFFFF) << 16)),
            _i32(dim1 & 0xFFFF),
            _i32(stride & 0xFFFFFFFF),
            _i32(((stride >> 32) & 0xFFFF) | ((dim1 & 0xFFFF) << 16)),
            _i32((dim1 >> 16) & 0xFFFFFFFF),
        ],
    )


def tdm_group1_rows_4b(rows):
    """GROUP1 for a ``rows`` x 128B dense tile of 4-byte elements.

    The runtime twin of :func:`tdm_group1`. A metadata body planned by
    :func:`tdm_plan_xfer_4b` is always exactly one row wide, so only the row
    count is dynamic and just three of the eight words stop being constants.
    ``rows`` must be wave-uniform and in ``[1, TDM_MAX_DIM]``; it is put through
    ``readfirstlane`` for the same reason :func:`tdm_group0` does it.
    """
    dim0 = TDM_ROW_ELEMS_4B
    r = readfirstlane(T.i32, arith.unwrap(rows))
    r_hi = arith.shli(r, _i32(16))
    return vector.from_elements(
        T.vec(8, T.i32),
        [
            _i32(2 << 16),  # data_size = log2(4B)
            _i32(dim0 << 16),
            r_hi,
            _i32(dim0 << 16),
            r,
            _i32(dim0),  # row stride == row width: the body is dense
            r_hi,
            _i32(0),
        ],
    )


def tdm_plan_xfer_4b(src_addr, dst_addr, n_elems):
    """Split a dword run into a scalar head/tail and a row-aligned TDM body.

    Returns ``(head, rows)``: ``head`` leading elements the caller moves itself,
    then ``rows`` x 32 elements for the engine, then whatever is left over --
    also the caller's. ``rows == 0`` asks for the whole run by hand.

    The engine commits global memory in whole 128-byte blocks, so a transfer
    that only partly covers its first or last block rewrites the rest of that
    block from the tile, and two runs sharing an end block clobber each other.
    Deciding the split from the element count alone -- long enough, therefore
    row-aligned -- is precisely that bug: the phase has to come from the
    addresses, which is why this takes them rather than a length.

    ``src_addr``/``dst_addr`` are the run's real start addresses (i64), so a
    misaligned arena base is caught here and not by a peer's corrupted
    metadata. Runs of fewer than two rows go entirely scalar: a single row is
    not worth a descriptor once the head and tail are peeled off it.
    """
    row_mask = TDM_ROW_FLOOR_BYTES - 1
    s_lo = arith.trunci(T.i32, arith.unwrap(src_addr))
    d_lo = arith.trunci(T.i32, arith.unwrap(dst_addr))
    skew = arith.andi(arith.xori(s_lo, d_lo), _i32(row_mask))
    sub_dword = arith.andi(arith.ori(s_lo, d_lo), _i32(3))
    phase = arith.shrui(arith.andi(s_lo, _i32(row_mask)), _i32(2))
    head = arith.andi(
        arith.subi(_i32(TDM_ROW_ELEMS_4B), phase), _i32(TDM_ROW_ELEMS_4B - 1)
    )
    head = arith.select(head > n_elems, n_elems, head)
    rows = arith.shrui(arith.subi(n_elems, head), _i32(_TDM_ROW_LOG2_4B))
    usable = (skew == 0) & (sub_dword == 0) & (rows >= 2)
    return (
        arith.select(usable, head, n_elems),
        arith.select(usable, rows, arith.constant(0, type=T.i32)),
    )


def tdm_group0(lds_addr_i32, global_addr_i64):
    """GROUP0 binding a descriptor to one (LDS byte address, global address) pair.

    Both operands must be wave-uniform -- the intrinsic reads them out of SGPRs.
    They are put through ``readfirstlane`` here rather than left to the compiler's
    uniformity analysis: the addresses this kernel feeds in are uniform by
    construction (a warp-strided token id, a ``readlane``-broadcast peer slot) but
    are computed from ``thread_idx``, and a descriptor lane that silently lands in
    a VGPR moves the wrong bytes instead of failing to compile. Every caller is in
    a wave-uniform branch, so lane 0 is always active.
    """
    i32 = ir.IntegerType.get_signless(32)
    addr = arith.unwrap(global_addr_i64)
    lo = arith.trunci(i32, addr)
    hi = arith.ori(
        arith.trunci(i32, arith.shrui(addr, arith.constant(32, type=T.i64))),
        _i32(1 << 31),  # descriptor type field
    )
    return vector.from_elements(
        T.vec(4, T.i32),
        [
            _i32(1),
            readfirstlane(T.i32, arith.unwrap(lds_addr_i32)),
            readfirstlane(T.i32, lo),
            readfirstlane(T.i32, hi),
        ],
    )


def tdm_load(group0, group1, cache_policy=0):
    """Async global -> LDS. Completion is observed with :func:`tdm_wait`."""
    tdm_ops.tensor_load_2d(tdm_ops.TDMDescriptor2D(group0, group1), cache_policy)


def tdm_store(group0, group1, cache_policy=0):
    """Async LDS -> global. Completion is observed with :func:`tdm_wait`."""
    tdm_ops.tensor_store_2d(tdm_ops.TDMDescriptor2D(group0, group1), cache_policy)


def tdm_wait(count=0):
    """Wait until at most ``count`` TDM transfers are still in flight.

    Loads and stores share the one tensor counter and retire in issue order, so
    a partial wait releases the oldest transfers -- there is no way to wait on
    just the loads.
    """
    tdm_ops.tensor_wait(count)

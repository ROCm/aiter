# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Transport parts that the one-shot, mesh and ring all-reduce kernels share.

This module contains the tile geometry, the inbox cache-policy table, the peer
store and load primitives, and the LDS staging factory. The limits and the
policy records are in ``aiter.ops.flydsl.allreduce_policy``.
"""

import logging

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

logger = logging.getLogger("aiter")

# Tile geometry
# =============

WORLD = 8  # Default value for world size
BLOCK = 256
ATOMS = 8
TILE_BYTES = BLOCK * ATOMS * 16
WAVE = 64
WAVES = BLOCK // WAVE
# 4 lanes × 16 B = one 64 B NT sector. Not world_size.
QUAD_LANES = 4
QUADS_PER_WAVE = WAVE // QUAD_LANES
# Wire/inbox addresses are byte pointers; tile math is in i32 slots.
I32_BYTES = 4
# The 64 B handshake sector is written as 8 lanes × 8 B: one colour-bearing
# store per destination. 8 B per lane because it is the widest store the
# backend performs atomically.
FLAG_LANES = 8
FLAG_I32_PER_LANE = 2

# Buffer aux bits on gfx942/gfx950. This is LLVM's CPol encoding, which the
# backend renames for CDNA: bit 0 (GLC) prints as `sc0`, bit 1 (SLC) as `nt`,
# bit 4 (SCC) as `sc1`. Bit 2 exists in the encoding but CDNA has no use for
# it, so it is dropped and emits nothing.
_CM_SC0 = 1
_CM_NT = 2
_CM_SC1 = 16

# Cache policy for the loads of a rank-tile from our own inbox, in the one-shot
# and the ring, on every inbox type. These loads need a cache bypass: with
# ``sc0 sc1`` it goes to memory. The ``recv`` entry of ``_INBOX_POLICY`` is
# ``nt`` on a fine-grained inbox, which is only a reuse hint.
_RECV_POLICY = _CM_SC0 | _CM_SC1

# Cache policy for each inbox memory type
# =======================================
#
# Each entry of ``_INBOX_POLICY`` has these keys:
#
#   payload  Keyword arguments for ``fx.generic_store`` on the payload stores
#            to a peer.
#   flag     Keyword arguments for ``fx.generic_store`` on the flag stores to
#            a peer.
#   release  The sync scope of the release fence before the flag store, or
#            None for no fence.
#   acquire  The sync scope of the acquire fence after the flag spin and
#            before the payload loads.
#   fanout   The axis of the (peer, sector) fanout that changes fastest
#            between consecutive quads.
#   recv     The ``_CM_*`` cache-modifier bits on the payload loads of the
#            reader.
#
# The table is the policy of the flag-based mesh. The ring uses
# ``_RECV_POLICY`` instead of ``recv``. The one-shot has no flag and no
# acquire: it polls the payload itself for a sentinel, and so it loads every
# payload with ``_CM_SC0 | _CM_SC1`` on both memory types, and it does not use
# the ``acquire`` or ``recv`` entries.
#
# On gfx942/gfx950, the stores lower as follows:
#
#   _ST_NT      global_store_* ... nt
#   _ST_SYSTEM  global_store_* ... sc0 sc1   (a relaxed system-scope atomic)
#
# Uncached inbox
# --------------
# A store to this memory type cannot stay in a cache. When ``vmcnt(0)`` retires
# a store, all peers can see the data.
#
# * payload, flag: ``nt``.
# * release: none. The ``vmcnt(0)`` before the flag store is enough.
# * acquire: workgroup scope. This fence does not invalidate a cache. It is a
#   compiler barrier only: it stops the compiler from moving the payload loads
#   above the flag spin.
# * recv: ``sc0 sc1``. The payload loads go past L1 and L2 to memory.
# * fanout: "sector".
#
# Fine-grained inbox
# ------------------
# This memory type is cacheable. A retired store can stay in the L2 of the
# writer, where the peers cannot see it.
#
# * payload: ``nt``.
# * release: system scope. The fence writes the payload back from L2 before the
#   flag store. The peers are other GPUs. Only a system-scope release puts the
#   payload before the flag for them.
# * flag: ``sc0 sc1``, which writes through to memory.
# * acquire: system scope. It lowers to ``buffer_inv sc0 sc1``, which
#   invalidates L1 and L2 of the reader.
# * recv: ``nt``.
# * fanout: "peer".
#
_ST_NT = (("nontemporal", True),)
_ST_SYSTEM = (("memory_order", fx.AtomicOrdering.Monotonic),)

_INBOX_POLICY = {
    "uncached": {
        "payload": _ST_NT,
        "flag": _ST_NT,
        "release": None,
        "acquire": rocdl.SyncScope.WorkgroupOneAs,
        "fanout": "sector",
        "recv": _CM_SC0 | _CM_SC1,
    },
    "finegrained": {
        "payload": _ST_NT,
        "flag": _ST_SYSTEM,
        "release": rocdl.SyncScope.OneAs,
        "acquire": rocdl.SyncScope.OneAs,
        "fanout": "peer",
        "recv": _CM_NT,
    },
}


def has_release_fence(inbox_memory: str) -> bool:
    """Whether this inbox type needs an L2 writeback at every publish."""
    return _INBOX_POLICY[inbox_memory]["release"] is not None


def _i32_to_bytes(i32_off):
    return fx.Int64(i32_off) * fx.Int64(I32_BYTES)


def _to_sgpr_i64(addr):
    """Copy a wave-uniform i64 from vector to scalar registers."""
    return fx.Int64(rocdl.readfirstlane(T.i64, as_ir_value(addr)))


def _global_ptr(addr_i64, elem_ty, alignment):
    """A global-address-space pointer at a raw byte address."""
    ptr_ty = fx.PointerType.get(
        elem_ty, address_space=fx.AddressSpace.Global, alignment=alignment
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _store_v4i32_peer(addr_i64, data, policy):
    """Store 16 B to a peer through a per-lane global address.

    *policy* is the inbox's ``payload`` entry in ``_INBOX_POLICY``. Every wire
    offset is a multiple of 16 B, hence the alignment.
    """
    fx.generic_store(_global_ptr(addr_i64, T.i32, 16), data, **dict(policy))


def _flag_word(color):
    """One flag lane's 8 B: *color* twice, so the sector's first i32 is it."""
    return fx.Vector.from_elements([color, color], fx.Int32).bitcast(fx.Int64)[0]


def _store_flag_peer(addr_i64, color, policy):
    """Write one lane's share of a peer's 64 B handshake sector.

    *policy* is the inbox's ``flag`` entry in ``_INBOX_POLICY``. On a cacheable
    inbox it is a relaxed system-scope atomic store, which the backend emits
    ``sc0 sc1`` (write-through) without the ``vmcnt(0)`` a volatile store would
    add after every flag. Atomic stores stop at 64 bits, which is why a sector
    takes ``FLAG_LANES`` lanes of 8 B rather than four of 16 B.
    """
    fx.generic_store(_global_ptr(addr_i64, T.i64, 8), _flag_word(color), **dict(policy))


def _release_inbox(scope):
    """Release fence over global memory: publish everything stored before it."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Release, syncscope=scope)


def _load_flag(addr_i64):
    """Poll one handshake flag: a relaxed system-scope atomic i32 load.

    The backend emits ``global_load_dword ... sc0 sc1``, fetched past L1 and L2
    every time, so a spin loop needs no fence in its body and no retry can see
    a stale line.

    A global load takes a per-lane address, so lanes spinning on different
    sources issue together.
    """
    return fx.generic_load(
        _global_ptr(addr_i64, T.i32, 4),
        dtype=fx.Int32,
        memory_order=fx.AtomicOrdering.Monotonic,
    )


def _poll_v4i32(addr_i64):
    """Re-read 16 B of an inbox that a peer writes without a flag.

    Two relaxed system-scope atomic i64 loads, ``global_load_dwordx2 ... sc0
    sc1``, so each retry is fetched past L1 and L2. The atomic is what keeps a
    spin on it alive: a side-effect-free loop over a plain or buffer load may be
    assumed to terminate, and the compiler is then free to delete it. A 128-bit
    atomic would be lowered to a library call, and ``volatile`` would add a
    ``vmcnt(0)`` after every load.
    """
    halves = [
        fx.generic_load(
            _global_ptr(addr_i64 + fx.Int64(8 * h), T.i64, 8),
            dtype=fx.Int64,
            memory_order=fx.AtomicOrdering.Monotonic,
        )
        for h in range(2)
    ]
    return fx.Vector.from_elements(halves, fx.Int64).bitcast(fx.Int32)


def _buffer_ptr(addr_i64, elem_ty, alignment, num_records_bytes=None):
    """A buffer-descriptor pointer at a raw global byte address.

    Without *num_records_bytes* the descriptor spans 4 GiB unchecked; with
    it, a load past the end returns 0 and a store is dropped. *addr_i64* must
    sit in scalar registers, see :func:`_to_sgpr_i64`.
    """
    return rocdl.make_buffer_ptr(
        _global_ptr(addr_i64, elem_ty, alignment), num_records_bytes=num_records_bytes
    )


def _buffer_load(ptr, elem_off, n, dtype, cache_modifier=0):
    """*n* consecutive elements of a buffer pointer, at an element offset.

    *cache_modifier* is ``_CM_*`` bits. An inbox read passes the policy's
    ``recv`` entry. ``nt`` is a reuse *hint* and does not stop the load being
    answered from the reader's own L1/L2, so it is only used where the
    policy's ``acquire`` invalidates both first (fine-grained). The uncached
    inbox reads ``_CM_SC0 | _CM_SC1``, which does bypass.
    """
    atom = fx.make_copy_atom(rocdl.BufferCopy(n * dtype.width, cache_modifier), dtype)
    reg = fx.make_rmem_tensor(n, dtype)
    fx.copy(atom, fx.make_view(ptr + elem_off, fx.make_layout(n, 1)), reg)
    return fx.Vector(fx.memref_load_vec(reg))


def _buffer_store(ptr, elem_off, vec, cache_modifier=0):
    """Store the vector *vec* to a buffer pointer, at an element offset.

    *cache_modifier* is ``_CM_*`` bits; ``_CM_SC0 | _CM_SC1`` writes through
    to memory.
    """
    atom = fx.make_copy_atom(
        rocdl.BufferCopy(vec.numel * vec.dtype.width, cache_modifier), vec.dtype
    )
    reg = fx.make_rmem_tensor(vec.numel, vec.dtype)
    fx.memref_store_vec(vec, reg)
    fx.copy(atom, reg, fx.make_view(ptr + elem_off, fx.make_layout(vec.numel, 1)))


def _load_peers(peer_ptrs, world_size):
    """The ``world_size`` inbox base addresses from the device-side peer table."""
    table = _buffer_ptr(peer_ptrs, T.i64, 8)
    return [_buffer_load(table, i, 1, fx.Int64)[0] for i in range(world_size)]


def _color_io(colors_ptr, bid):
    """``(load, store)`` for this workgroup's colour, one i32 per block."""
    colors = _buffer_ptr(colors_ptr, T.i32, 4)

    def load():
        return _buffer_load(colors, bid, 1, fx.Int32)[0]

    def store(color):
        _buffer_store(colors, bid, fx.Vector.from_elements([color], fx.Int32))

    return load, store


def _payload_io(
    inp_ptr, out_ptr, nbytes, num_tiles, atoms, block, tid, decode=None, encode=None
):
    """``(load, store)`` for this thread's 16 B of one payload atom.

    The payload is ``(num_tiles, atoms, block * 4)`` i32 and thread *tid*
    owns i32 ``[4 * tid, 4 * tid + 4)`` of every atom row. ``load(tile,
    atom)`` reads it from *inp_ptr* as i32x4, passed through *decode* if
    given; ``store(tile, atom, vec)`` writes ``encode(vec)`` to *out_ptr*.
    Both tensors are bounded by *nbytes*, the live payload, so a partial last
    tile reads 0 and its stores are dropped rather than faulting.
    """
    row_i32 = block * 4
    layout = fx.make_layout((num_tiles, atoms, row_i32), (atoms * row_i32, row_i32, 1))
    row_layout = fx.make_layout((1, row_i32), (row_i32, 1))
    copy_atom = fx.make_copy_atom(rocdl.BufferCopy128b(), fx.Int32)
    thr_copy = fx.make_tiled_copy_tv(
        copy_atom,
        fx.make_layout((1, block), (1, 1)),
        fx.make_layout((1, 4), (1, 1)),
    ).get_slice(tid)

    def _tensor(addr):
        view = fx.make_view(_global_ptr(addr, T.i32, 16), layout)
        return rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=nbytes)

    inp, out = _tensor(inp_ptr), _tensor(out_ptr)

    def _row(buf, tile, atom):
        return fx.make_view(fx.get_iter(fx.slice(buf, (tile, atom, None))), row_layout)

    def load(tile, atom):
        src = thr_copy.partition_S(_row(inp, tile, atom))
        frag = fx.make_fragment_like(src)
        fx.copy(copy_atom, src, frag)
        vec = fx.Vector(frag.load())
        return vec if decode is None else decode(vec)

    def store(tile, atom, vec):
        dst = thr_copy.partition_D(_row(out, tile, atom))
        frag = fx.make_fragment_like(dst)
        frag.store(vec if encode is None else encode(vec))
        fx.copy(copy_atom, frag, dst)

    return load, store


def _acquire_inbox(scope):
    """Acquire fence over global memory at *scope*."""
    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope=scope)


def atom_bf16_to_f32(atom_i32):
    """16 B of bf16 (8 values) -> 8 f32.

    bf16 is the high half of f32, so this is a widening move, not a conversion
    -- exact, no rounding.
    """
    return fx.Vector(atom_i32).bitcast(fx.BFloat16).to(fx.Float32)


def atom_f32_to_bf16(acc_f32):
    """8 f32 -> 16 B of bf16.

    One rounding, at the end of whatever computed *acc_f32*, which is what makes
    the exact schedules bit-comparable with ``cross_device_reduce``'s fp32
    accumulate + single downcast.
    """
    return acc_f32.to(fx.BFloat16).bitcast(fx.Int32)


def make_pack_storage(n_i32: int):
    """LDS staging for *n_i32* packed words, 16 B aligned."""

    @fx.struct
    class PackStorage:
        pack: fx.Array[fx.Int32, n_i32, 16]

    return PackStorage


def make_hbm_operand(*, padded, nbytes, hbm_i32_ptr, hbm_layout):
    """``operand(ptr, records=None)``: the handle the atom load/store helpers
    address an HBM operand through, given its raw ``Int64`` base pointer.

    A padded build hands the pointer back untouched: its pad lanes need a
    *per-lane* bound, which a whole-tensor descriptor cannot express, so
    ``make_rowbuf_atom_row`` builds a bounded descriptor per row from it. An
    unpadded build hands back a tiled-copy tensor (layout + descriptor) whose
    ``num_records_bytes`` is *records*, the live payload by default, so a
    partial last tile reads 0 and its stores are dropped rather than faulting.

    ``hbm_i32_ptr``/``hbm_layout`` are only read on the unpadded branch, so a
    padded build may pass ``None`` for ``hbm_layout`` (it is never built there).
    The one-shot, mesh and ring kernels all share this verbatim.
    """

    def _operand(ptr, records=None):
        if padded:
            return ptr
        n = nbytes if records is None else records
        view = fx.make_view(fx.inttoptr(hbm_i32_ptr, ptr), hbm_layout)
        return rocdl.make_buffer_tensor(view, max_size=False, num_records_bytes=n)

    return _operand

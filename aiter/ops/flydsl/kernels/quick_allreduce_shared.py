# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Transport assets shared by every quick-allreduce schedule.

Tile geometry, the inbox cache-policy table, the peer store/load primitives and
the LDS staging factory.

The mesh, ring and one-shot kernels all build on this module, and they must
agree byte for byte on what it defines. ``one_shot_allreduce`` uses it
*without* the codec, which is why the two are separate modules.
"""

import logging

import flydsl.expr as fx
from flydsl.expr import rocdl
from flydsl.expr.typing import T, as_ir_value

from . import buffer_ops

logger = logging.getLogger("aiter")

WORLD = 8  # Default value for world size
SUPPORTED_WORLDS = (2, 4, 8)
BLOCK = 256
ATOMS = 8
TILE_BYTES = BLOCK * ATOMS * 16
TILE_I32 = TILE_BYTES // 4
TILE_FP16 = TILE_BYTES // 2
DEFAULT_GRID_CAP = 304 * 4
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

# Cache policy for the peer stores in _fanout_nt / _publish, per inbox memory
# type.
#
# On an uncached inbox the memory type does all the work: a store cannot sit in
# any cache, so every peer sees the payload as soon as `vmcnt(0)` retires and
# the release needs nothing beyond that.
#
#
# ``payload`` and ``flag`` are keyword arguments for ``fx.generic_store``, as
# ``(name, value)`` tuples rather than dicts: the kernel factories capture them,
# and FlyDSL folds a captured tuple into its compile-cache key but silently
# leaves a dict out. On gfx942/gfx950 they lower to:
#
#   _ST_PLAIN   global_store_*
#   _ST_NT      global_store_* ... nt
#   _ST_SYSTEM  global_store_* ... sc0 sc1   (a relaxed system-scope atomic)
#
# ``release`` is the sync scope of the release fence that precedes the flag, or
# None for none. System scope: the peers are other GPUs, and
# only a system-scope release orders the payload before the flag for them.
#
# ``fanout`` picks which axis of the (peer, sector) fanout runs fastest across
# consecutive quads; see the layouts in the kernel body.
#
# ``recv`` is the cache modifier the *reader* uses on its payload loads. It is
# part of the same policy because it is the other half of the same decision: the
# more the writer is allowed to cache, the harder the reader has to work to
# avoid a stale line.
_ST_PLAIN = ()
_ST_NT = (("nontemporal", True),)
_ST_SYSTEM = (("memory_order", fx.AtomicOrdering.Monotonic),)
_RELEASE_SCOPE = rocdl.SyncScope.OneAs

_INBOX_POLICY = {
    "uncached": {
        "payload": _ST_NT,
        "flag": _ST_NT,
        "release": None,
        "fanout": "sector",
        "recv": _CM_NT,
    },
    "finegrained": {
        "payload": _ST_NT,
        "flag": _ST_SYSTEM,
        "release": _RELEASE_SCOPE,
        "fanout": "peer",
        "recv": _CM_NT,
    },
    # Coarse-grained: the only mode whose pages are marked cacheable, so the
    # only one where a peer write can actually sit in the writer's L2 and be
    # combined with its neighbours before going out on the wire.
    #
    # The payload stores are plain (no `nt`) so lines stay dirty in L2;
    # the release fence at the publish point is what puts them on the wire;
    # the flag goes out write-through so the peer's spin sees it after the
    # payload; and the reader must bypass both its caches (`sc0 sc1`) rather
    # than trust `nt`, which is only a hint and can be answered from a stale
    # line.
    "default": {
        "payload": _ST_PLAIN,
        "flag": _ST_SYSTEM,
        "release": _RELEASE_SCOPE,
        "fanout": "peer",
        "recv": _CM_SC0 | _CM_SC1,
    },
}
FANOUT_ORDERS = ("sector", "peer")


def has_release_fence(inbox_memory: str) -> bool:
    """Whether this inbox type needs an L2 writeback at every publish.

    Callers use it to decide how hard to work at batching publishes: with a
    fence they are expensive, without one they are nearly free.
    """
    return _INBOX_POLICY[inbox_memory]["release"] is not None


def _i32_to_bytes(i32_off):
    return fx.Int64(i32_off) * fx.Int64(I32_BYTES)


def _to_sgpr_i64(addr):
    """Copy a wave-uniform i64 from vector to scalar registers.

    Buffer loads/stores need the descriptor in scalar registers. After
    ``peers[rank]``, every lane holds the same pointer, but it sits in a
    vector register, so LLVM cannot prove that. It then serializes the
    wave: one lane at a time, copy that lane's pointer to a scalar
    register, mask to that lane, issue the load, repeat. ``readfirstlane``
    copies lane 0's value into a scalar register once so the whole wave
    issues a single buffer op.

    Do this at the inbox descriptor, not on the peer list: fanout stores
    use a different peer per lane, so those addresses must stay in vector
    registers. ``T.i64`` is the result type ``readfirstlane`` requires.
    """
    return fx.Int64(rocdl.readfirstlane(T.i64, as_ir_value(addr)))


def _global_ptr(addr_i64, elem_ty, alignment):
    """A global-address-space pointer at a raw byte address."""
    ptr_ty = fx.PointerType.get(
        elem_ty, address_space=fx.AddressSpace.Global, alignment=alignment
    )
    return fx.inttoptr(ptr_ty, fx.Int64(addr_i64))


def _store_v4i32_peer(addr_i64, data, policy):
    """Store 16 B to a peer through a per-lane global address.

    One instruction here sends 16 B to a different GPU in each lane of a
    4-wide group. A buffer-descriptor store wants the descriptor in scalar
    registers, so LLVM would serialize those lanes (one destination at a
    time). A flat global store takes the address from a vector register,
    so all destinations issue together.

    *policy* is the inbox's ``payload`` entry in ``_INBOX_POLICY``: plain or
    ``nt``. Every wire offset is a multiple of 16 B, hence the alignment.

    A native store, so the backend sees it. That matters beyond the cache
    bits: a VMEM store of more than 64 bits needs two wait states (gfx940 and
    later) before a VALU may overwrite its data VGPRs, and LLVM's hazard
    recognizer only inserts them after a store it can see.
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
    """Release fence over global memory: publish everything stored before it.

    At system scope (``one-as``) the backend emits ``buffer_wbl2 sc0 sc1``
    followed by ``s_waitcnt vmcnt(0)``: the L2 writeback that a cacheable inbox
    needs before its flag goes out. ``one-as`` scopes it to global memory, so it
    does not wait on ``lgkmcnt``.
    """
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


def _load_i32_nt(rsrc, elem_off, cache_modifier=_CM_NT):
    """Load one i32 from the inbox under the policy's receive modifier.

    Defaults to ``nt`` -- correct when the inbox memory cannot hold a stale
    line in the first place. A cacheable (coarse-grained) inbox must pass
    ``_CM_SC0 | _CM_SC1`` instead, because ``nt`` is a reuse *hint* and does
    not stop this load being answered from the reader's own L1/L2.
    """
    return fx.Int32(
        buffer_ops.buffer_load(
            rsrc, elem_off, vec_width=1, dtype=T.i32, cache_modifier=cache_modifier
        )
    )


def _acquire_inbox():
    """Acquire fence over global memory, system scope.

    Replaces a raw ``buffer_inv sc1`` asm. On gfx950 the memory legalizer
    lowers this to ``s_waitcnt vmcnt(0)`` + ``buffer_inv sc0 sc1``, so the
    encoding comes from the target rather than from a string in this file.
    ``one-as`` scopes it to global memory, which is why it does not also
    wait on ``lgkmcnt``.

    System scope, not agent. The data being acquired was written by another
    GPU, and another GPU is a different agent: agent scope lowers to
    ``buffer_inv sc1``, which clears L2 but leaves the vector L1 holding
    whatever it had. That is what the old asm did.

    Call this **once, after the workgroup joins** -- not inside a spin loop.
    A flag load that carries ``sc0 sc1`` can never be answered from a stale
    line, so the retry loop needs no fence of its own; what needs one is the
    payload read that follows, and that needs it exactly once. See
    :func:`quick_allreduce_int4.make_quick_allreduce_int4_kernel._wait_release`.
    """
    fx.memory_fence(ordering=fx.AtomicOrdering.Acquire, syncscope=rocdl.SyncScope.OneAs)


def make_pack_storage(n_i32: int):
    """LDS staging for *n_i32* packed words, 16 B aligned.

    A factory rather than one struct because the two schedules need different
    sizes: the mesh stages every destination's packet (``ATOMS`` rows), while
    the ring stages one destination's (``rank_atoms`` rows) and its row stride
    depends on which codec that hop carries.
    """

    @fx.struct
    class PackStorage:
        pack: fx.Array[fx.Int32, n_i32, 16]

    return PackStorage

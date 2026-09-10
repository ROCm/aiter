# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Transport assets shared by every QRInt4 schedule.

Tile geometry, the inbox cache-policy table, the peer store/load primitives and
the LDS staging factory.

The mesh, ring and one-shot kernels all build on this module, and they must
agree byte for byte on what it defines. ``qr_1stage_kernel`` uses it *without*
the codec, which is why the two are separate modules.
"""

import logging
import os

import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
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
WAVES = 4
# 4 lanes × 16 B = one 64 B NT sector. Not world_size.
QUAD_LANES = 4
QUADS_PER_WAVE = WAVE // QUAD_LANES
# Wire/inbox addresses are byte pointers; tile math is in i32 slots.
I32_BYTES = 4

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
# A fine-grained inbox is cacheable, which cuts both ways. Letting the payload
# land in the writer's L2 is exactly what makes it fast on PCIe: the L2 coalesces
# this kernel's 64 B destination-interleaved stores into large bursts, worth 30x
# at prefill sizes (227 us against 6708 us at 14 MiB on MI350P). But `nt` is only
# a non-temporal *hint* -- it does not write through -- so a payload store can
# still be parked in L2 after `vmcnt(0)` while a peer spins on a flag it cannot
# see. That stall clears only when unrelated traffic evicts the line, so its cost
# scales inversely with how busy the kernel is: invisible at 448 blocks,
# 5.3 seconds at 1 block.
#
# So keep the payload cacheable and make the *release* explicit: write back L2
# after the payload drains, then publish the flag write-through so the peer's
# spin observes it immediately. Forcing the payload itself write-through
# (`sc0 sc1` on every store) also fixes visibility, but defeats the coalescing
# and gives back the entire bandwidth win.
# ``fanout`` picks which axis of the (peer, sector) fanout runs fastest across
# consecutive quads; see the layouts in the kernel body.
#
# ``recv`` is the cache modifier the *reader* uses on its payload loads. It is
# part of the same policy because it is the other half of the same decision: the
# more the writer is allowed to cache, the harder the reader has to work to
# avoid a stale line.
_INBOX_POLICY = {
    "uncached": {
        "payload": "nt",
        "flag": "nt",
        "writeback": None,
        "fanout": "sector",
        "recv": _CM_NT,
    },
    "finegrained": {
        "payload": "nt",
        "flag": "sc0 sc1 nt",
        "writeback": "buffer_wbl2 sc1",
        "fanout": "peer",
        "recv": _CM_NT,
    },
    # Coarse-grained: the only mode whose pages are marked cacheable, so the
    # only one where a peer write can actually sit in the writer's L2 and be
    # combined with its neighbours before going out on the wire.
    #
    # The payload stores are plain (no `nt`) so lines stay dirty in L2; 
    # `buffer_wbl2` at the publish point is what puts them on the wire; 
    # the flag goes out write-through so the peer's spin sees it after the payload; 
    # and the reader must bypass both its caches (`sc0 sc1`) rather than trust `nt`, 
    # which is only a hint and can be answered from a stale line.
    "default": {
        "payload": "",
        "flag": "sc0 sc1",
        "writeback": "buffer_wbl2 sc1",
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
    return _INBOX_POLICY[inbox_memory]["writeback"] is not None


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


def _store_v4i32_peer(addr_i64, data, policy):
    """Store 16 B to a peer through a per-lane global address.

    One instruction here sends 16 B to a different GPU in each lane of a
    4-wide group. A buffer-descriptor store wants the descriptor in scalar
    registers, so LLVM would serialize those lanes (one destination at a
    time). A flat global store takes the address from a vector register,
    so all destinations issue together.

    *policy* is the cache-policy suffix for the inbox memory type; see
    ``_INBOX_POLICY``. Not yet applied natively -- see TODO below.

    A batched variant lives in ``qr_1stage_kernel._store_v4i32_peer_multi``,
    its only consumer. Both emit ``global_store_dwordx4 ... {policy}``, so a
    change to that encoding has to be made in both places.
    """
    ptr_ty = ir.Type.parse("!llvm.ptr<1>")
    ptr = llvm.IntToPtrOp(ptr_ty, as_ir_value(addr_i64)).result
    llvm.InlineAsmOp(
        None,
        [ptr, as_ir_value(data)],
        f"global_store_dwordx4 $0, $1, off {policy}",
        "v,v",
        has_side_effects=True,
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


def _load_i32_uncached(rsrc):
    """One i32 that cannot be answered from this device's caches.

    ``sc0 sc1`` is what LLVM itself emits for a system-scope load, and it is
    what makes a spin loop safe without a fence in the loop body: the value
    is fetched past L1 and L2 every time, so no retry can see a stale line.
    ``sc1`` alone would only bypass L2.
    """
    val = buffer_ops.buffer_load(
        rsrc, 0, vec_width=1, dtype=T.i32, cache_modifier=_CM_SC0 | _CM_SC1
    )
    rocdl.s_waitcnt(vmcnt=0)
    return fx.Int32(val)


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
    :func:`qr_int4_kernel.make_qr_int4_kernel._wait_release`.
    """
    llvm.fence(llvm.AtomicOrdering.acquire, syncscope="one-as")


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

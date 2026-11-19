# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Lamport handshake parts that the mesh, ring, and hierarchical kernels
share.

An unwritten inbox dword holds the sentinel (``LAMPORT_SENTINEL``, the inbox
starts filled with ``LAMPORT_FILL_BYTE``), and a producer keeps its packets
off that value (``quick_allreduce_codec.lamport_fix``). Hence, a dword that is not
the sentinel has landed. Each dword is written once per round, so a torn 16 B
arrival needs no special handling: a dword that has not landed yet still reads
as the sentinel.

**Receive.** A thread reads every dword it decodes from a set of packets
(``lamport_offsets``) before it tests any, past L1 and L2 (``lamport_poll``),
then re-reads all of them together until none is the sentinel
(``lamport_spin``): a serial spin would pay one more round trip per packet. The
re-reads are relaxed system-scope atomics (``_load_flag``), which go past L1
and L2 and keep the spin from being deleted. Re-reading a dword that has
landed is harmless, since nobody writes it again this round.

**Re-arm.** A reader puts the sentinel back before the slot's writer can use
it again. Each kernel decides when, and keeps enough inbox buffers that its
re-arm is off the critical path:

* ``lamport_rearm_dwords``: the dwords this thread has just polled, right after
  the poll (mesh, ring). A dword that several threads read (scale, extra
  plane) is re-armed by one of them, which ``lamport_slots`` picks; all of
  them are lanes of one wave, which leave the spin together.
* ``lamport_rearm_span``: whole slots, 16 B per lane, a super-tile group late
  (hierarchical): on a fine-grained inbox a push can reach the reader by
  eviction before the writer's release writes the line back, and a sentinel
  put back at once could be overwritten by the same stale bytes.

**Send.** No drain and no flag: the packet words are their own flag. After its
pushes each wave writes them back out of L2 on a cacheable inbox
(``lamport_pushed``).
"""

import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, range_constexpr, rocdl

from ...allreduce_policy import HANDSHAKES
from .quick_allreduce_codec import lamport_slots, split_lamport_packet
from .transport import (
    _CM_NT,
    _CM_SC0,
    _CM_SC1,
    _INBOX_POLICY,
    _RECV_POLICY,
    LAMPORT_SENTINEL,
    _buffer_load,
    _buffer_store,
    _i32_to_bytes,
    _load_flag,
    _release_inbox,
)


def check_handshake(handshake: str) -> bool:
    if handshake not in HANDSHAKES:
        raise ValueError(f"handshake must be one of {HANDSHAKES}, got {handshake!r}")
    return handshake == "lamport"


def lamport_rearm_modifier(inbox_memory: str) -> int:
    """``_CM_*`` bits of the stores that put the sentinel back.

    On a cacheable inbox they write through (``sc0 sc1``): an ``nt`` store
    could leave the sentinel dirty in our own L2, where it would hide a peer's
    later push from the poll, and its eviction would overwrite that push. The
    stores are local, so writing through costs no fabric traffic.
    """
    if _INBOX_POLICY[inbox_memory]["release"] is None:
        return _CM_NT
    return _CM_SC0 | _CM_SC1


def _any_sentinel(vec):
    """Whether any i32 of *vec* is still the Lamport sentinel."""
    hit = vec[0] == fx.Int32(LAMPORT_SENTINEL)
    for e in range(1, vec.numel):
        hit = hit | (vec[e] == fx.Int32(LAMPORT_SENTINEL))
    return hit


def lamport_offsets(codec, tid, scale_slot, bases):
    """``(offs, rearm)`` of the packets of *codec* at the i32 offsets *bases*:
    every dword this thread reads, packet by packet in ``_codec_load`` order,
    and the same offsets with their re-arm predicates (``lamport_slots``).

    *bases* may be a generator: each base is taken just before its packet's
    dwords, so a caller keeps its own order of address arithmetic.
    """
    offs, rearm = [], []
    for base in bases:
        for off, pred in lamport_slots(codec, tid, scale_slot):
            offs.append(base + off)
            rearm.append((base + off, pred))
    return offs, rearm


def lamport_first_reads(inbox, offs):
    """The first read of every dword at *offs* of the buffer pointer *inbox*,
    all issued before any is tested, past L1 and L2."""
    return [_buffer_load(inbox, o, 1, fx.Int32, _RECV_POLICY)[0] for o in offs]


def lamport_addrs(inbox_base, offs):
    """The global addresses of the dwords at *offs*, for ``lamport_spin``."""
    return [inbox_base + _i32_to_bytes(o) for o in offs]


@ASTRewriter.transform
def lamport_spin(got, addrs, poll_sleep=0):
    """Re-read *addrs* until no dword of *got* is the sentinel; return them.

    *poll_sleep* is the mesh's ``poll_sleep`` knob: ``s_sleep`` is a pure
    issue-rate hint. It orders nothing and the loop exit still depends on the loaded
    values, so it cannot change which value a poll accepts.
    """
    while _any_sentinel(got):
        if const_expr(poll_sleep != 0):
            rocdl.s_sleep(poll_sleep)
        got = fx.Vector.from_elements([_load_flag(a) for a in addrs], fx.Int32)
    return got


def lamport_poll(inbox, inbox_base, offs, poll_sleep=0):
    """Every dword at *offs* of our inbox, once none is the sentinel."""
    got = fx.Vector.from_elements(lamport_first_reads(inbox, offs), fx.Int32)
    return lamport_spin(got, lamport_addrs(inbox_base, offs), poll_sleep)


def lamport_packets(codec, got, n_packets):
    """Yield ``(words, scale_word)``, raw as ``_codec_load`` returns them, for
    each of the *n_packets* packets of *codec* that the polled dwords *got* hold."""
    per = len(got) // n_packets
    for i in range_constexpr(n_packets):
        yield split_lamport_packet(
            codec, [got[i * per + j] for j in range_constexpr(per)]
        )


@ASTRewriter.transform
def lamport_rearm_dwords(inbox, rearm, cache_modifier):
    """Put the sentinel back in the polled dwords *rearm* (``lamport_offsets``)
    of the buffer pointer *inbox*, each by the one thread its predicate picks.

    ``vmcnt(0)`` first: it drains the re-arm of the round before this one,
    which was issued before these reads and must land before our next push.
    """
    rocdl.s_waitcnt(vmcnt=0)
    armed = fx.Vector.from_elements([fx.Int32(LAMPORT_SENTINEL)], fx.Int32)
    for off, pred in rearm:
        if pred:
            _buffer_store(inbox, off, armed, cache_modifier)


@ASTRewriter.transform
def lamport_rearm_span(inbox, base_i32, n_i32, tid, block, cache_modifier):
    """Put the sentinel back in *n_i32* dwords of the buffer pointer *inbox*
    from *base_i32*, 16 B per lane across the *block* threads."""
    armed = fx.Vector.from_elements([fx.Int32(LAMPORT_SENTINEL)] * 4, fx.Int32)
    n_vec = fx.Int32(n_i32 // 4)
    for v0 in range(fx.Int32(0), n_vec, fx.Int32(block)):
        v = v0 + tid
        if v < n_vec:
            _buffer_store(inbox, base_i32 + v * fx.Int32(4), armed, cache_modifier)


def lamport_pushed(release_scope):
    """After a wave's pushes: on a cacheable inbox (*release_scope* not None)
    write them back out of L2. No barrier and no flag."""
    if const_expr(release_scope is not None):
        _release_inbox(release_scope)

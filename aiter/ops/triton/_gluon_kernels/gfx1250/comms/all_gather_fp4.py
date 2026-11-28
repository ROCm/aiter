# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Byte-exact TP4 gather of FP4 codes and optional row metadata."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit(repr=make_kernel_repr("_fp4_ag_publish", ["BLOCK_WORDS"]))
def _publish_kernel(
    codes,
    metadata,
    qout,
    mout,
    peers,
    calls,
    qwords,
    mwords,
    RANK: gl.constexpr,
    SLOT_WORDS: gl.constexpr,
    BLOCK_WORDS: gl.constexpr,
    LOCAL_COPY: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
    i = gl.arange(0, BLOCK_WORDS, layout=layout)
    pid = gl.program_id(0)
    qblocks = gl.cdiv(qwords, BLOCK_WORDS)
    call = gl.load(calls) + 1
    base = peers[RANK].to(gl.pointer_type(gl.int32))
    bank = 64 + (call % 2) * SLOT_WORDS
    if pid < qblocks:
        offset = pid * BLOCK_WORDS + i
        values = gl.amd.cdna5.buffer_load(codes, offset, offset < qwords, other=0)
        gl.amd.cdna5.buffer_store(
            values, base, bank + offset, offset < qwords, cache=".wt"
        )
        if LOCAL_COPY:
            gl.amd.cdna5.buffer_store(
                values, qout, RANK * qwords + offset, offset < qwords
            )
    else:
        offset = (pid - qblocks) * BLOCK_WORDS + i
        values = gl.amd.cdna5.buffer_load(metadata, offset, offset < mwords, other=0)
        gl.amd.cdna5.buffer_store(
            values, base, bank + qwords + offset, offset < mwords, cache=".wt"
        )
        if LOCAL_COPY:
            gl.amd.cdna5.buffer_store(
                values, mout, RANK * mwords + offset, offset < mwords
            )


@gluon.jit(repr=make_kernel_repr("_fp4_ag_ready", ["RANK"]))
def _ready_kernel(peers, calls, errors, RANK: gl.constexpr, MAXIT: gl.constexpr):
    # This kernel starts after every local publish CTA has completed its stores.
    call = gl.load(calls) + 1
    gl.store(calls, call)
    layout: gl.constexpr = gl.BlockedLayout([1], [32], [1], [0])
    flags = gl.arange(0, 4, layout=layout)
    for peer in gl.static_range(4):
        base = peers[peer].to(gl.pointer_type(gl.int32))
        gl.amd.cdna5.buffer_store(
            call + flags * 0, base, flags, flags == RANK, cache=".wt"
        )
    local = peers[RANK].to(gl.pointer_type(gl.int32))
    late = 1
    it = 0
    while (late != 0) & (it < MAXIT):
        seen = gl.amd.cdna5.buffer_load(local, flags, cache=".cv")
        late = gl.sum((seen < call).to(gl.int32), 0)
        gl.inline_asm("", "~{memory}", [], [], is_pure=False)
        it += 1
    if late != 0:
        gl.atomic_add(errors, 1)


@gluon.jit(
    repr=make_kernel_repr("_fp4_ag_read", ["BLOCK_WORDS", "PEER_GROUP", "LOCAL_COPY"])
)
def _gather_kernel(
    peers,
    calls,
    qout,
    mout,
    qwords,
    mwords,
    RANK: gl.constexpr,
    SLOT_WORDS: gl.constexpr,
    BLOCK_WORDS: gl.constexpr,
    PEER_GROUP: gl.constexpr,
    LOCAL_COPY: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([4], [32], [gl.num_warps()], [0])
    i = gl.arange(0, BLOCK_WORDS, layout=layout)
    linear = gl.program_id(0)
    qblocks = gl.cdiv(qwords, BLOCK_WORDS)
    programs = qblocks + gl.cdiv(mwords, BLOCK_WORDS)
    peers_count: gl.constexpr = 3 if LOCAL_COPY else 4
    if PEER_GROUP == 0:
        pid = linear % programs
        peer_index = linear // programs
    else:
        pid = linear // (peers_count * PEER_GROUP) * PEER_GROUP + linear % PEER_GROUP
        peer_index = linear // PEER_GROUP % peers_count
    peer_id = (RANK + peer_index + (1 if LOCAL_COPY else 0)) % 4
    bank = 64 + (gl.load(calls) % 2) * SLOT_WORDS
    for peer in gl.static_range(4):
        if (peer_id == peer) & (pid < programs):
            base = peers[peer].to(gl.pointer_type(gl.int32))
            if pid < qblocks:
                offset = pid * BLOCK_WORDS + i
                values = gl.amd.cdna5.buffer_load(
                    base, bank + offset, offset < qwords, other=0, cache=".cv"
                )
                gl.amd.cdna5.buffer_store(
                    values, qout, peer * qwords + offset, offset < qwords
                )
            else:
                offset = (pid - qblocks) * BLOCK_WORDS + i
                values = gl.amd.cdna5.buffer_load(
                    base, bank + qwords + offset, offset < mwords, other=0, cache=".cv"
                )
                gl.amd.cdna5.buffer_store(
                    values, mout, peer * mwords + offset, offset < mwords
                )

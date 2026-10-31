# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from the Kimi-K3 mono mailbox query gather.

"""DCP query all-gather as one FlyDSL mailbox launch.

Each rank pushes its local heads. Each rank polls every source into a persistent
buffer. ``nope > 0`` quantizes cat(ql_nope, q_pe) to fp8 with one per-tensor scale.
``fold > 0`` stores pseudo-request ``req * fold + group`` as 16 heads.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.dcp_mailbox.combine import TAG_SLOTS, _finish
from aiter.ops.flydsl.dcp_mailbox.common import (
    KernelAbi,
    bf2_f32,
    fp8_pack4,
    key_tuple,
    ld_dev,
    load_ptr64,
    pair_layout,
    peer_bases,
    rsrc,
    uniform,
)
from aiter.ops.flydsl.dcp_mailbox.peer import PeerBuffer
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    TaggedMailbox,
    tagged_peer_address,
    tagged_shift,
)

THREADS = 128
FP8_MAX = 448.0

_STREAM = fx.Stream(None)
_BUILDS: dict = {}


@dataclass(frozen=True)
class GatherBuild:
    tokens: int
    ranks: int
    heads: int
    words: int
    pad_heads: int
    max_tokens: int
    nope: int = 0
    fold: int = 0
    qlen: int = 0
    phase: int = 0


def peer_layout(ranks, max_tokens, heads, words):
    """One parity: word pairs [src, token, head, words]."""
    return pair_layout([("q", ranks * max_tokens * heads * words)])


def peer_bytes(ranks, max_tokens, heads, words):
    lay = peer_layout(ranks, max_tokens, heads, words)
    return 2 * max(o + n for o, n in lay.values())


ABI = KernelAbi(
    (
        "q",
        "q_sb",
        "q_sh",
        "pe",
        "pe_sb",
        "pe_sh",
        "scale",
        "gq",
        "peers",
        "rank",
        "state",
    )
)


def build(key: GatherBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    n, hl, w, mt = key.ranks, key.heads, key.words, key.max_tokens
    assert w % 2 == 0 and w <= 2 * THREADS, w
    assert key.nope % 8 == 0
    nblk = key.tokens * hl
    lay = peer_layout(n, mt, hl, w)
    half = peer_bytes(n, mt, hl, w) // 2
    keyed = key_tuple(key)
    name = (
        f"aiter_dcp_qgather_t{key.tokens}_n{n}_h{hl}_w{w}_p{key.pad_heads}"
        f"_m{mt}_q{key.nope}_f{key.fold}x{key.qlen}_s{key.phase}"
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def gather(
        q: Int64,
        q_sb: Int32,
        q_sh: Int32,
        pe: Int64,
        pe_sb: Int32,
        pe_sh: Int32,
        scale: Int64,
        gq: Int64,
        peers: Int64,
        rank: Int32,
        state: Int64,
    ):
        _ = keyed
        tid = fx.Int32(fx.thread_idx.x)
        bid = fx.Int32(fx.block_idx.x)
        b = bid // hl
        lh = bid % hl
        e = uniform(fx.Int32(ld_dev(state, 0)))
        mb = TaggedMailbox((e % (1 << 22)) * TAG_SLOTS)
        put_words, poll = mb.put_words, mb.poll
        flip = fx.Int64(e % 2) * half
        bases = peer_bases(peers, n)
        own = load_ptr64(peers, rank)
        to = [
            tagged_shift(tagged_peer_address(peer, lay["q"][0]), flip) for peer in bases
        ]
        mine = tagged_shift(tagged_peer_address(own, lay["q"][0]), flip)
        slot = (rank * mt + b) * hl + lh
        j = fx.min(tid, w // 2 - 1) * 2
        push = key.phase != 2
        if const_expr(push and key.nope > 0):
            e0 = j * 4
            in_nope = e0 < key.nope
            vn = fx.Vector(
                bo.buffer_load(
                    rsrc(q),
                    (b * q_sb + lh * q_sh + fx.min(e0, key.nope - 8)) // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            vp = fx.Vector(
                bo.buffer_load(
                    rsrc(pe),
                    (b * pe_sb + lh * pe_sh + fx.max(e0 - key.nope, 0)) // 2,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            inv = fx.Float32(1.0) / fx.Float32(
                bo.buffer_load(rsrc(scale), 0, vec_width=1, dtype=T.f32)
            )
            xs = []
            for k in range_constexpr(4):
                lo, hi = bf2_f32(in_nope.select(vn[k], vp[k]))
                xs += [lo, hi]
            ys = [
                fx.max(fx.min(x * inv, fx.Float32(FP8_MAX)), fx.Float32(-FP8_MAX))
                for x in xs
            ]
            v = [fp8_pack4(*ys[0:4]), fp8_pack4(*ys[4:8])]
        if const_expr(push and key.nope == 0):
            vw = fx.Vector(
                bo.buffer_load(
                    rsrc(q), b * q_sb + lh * q_sh + j, vec_width=2, dtype=T.i32
                )
            )
            v = [vw[0], vw[1]]
        if const_expr(push):
            for dest in range_constexpr(n):
                put_words(to[dest], slot * w + j, [v[0], v[1]])
        if const_expr(key.phase == 1):
            return
        got = poll(
            [(mine, ((src * mt + b) * hl + lh) * w + j, 2) for src in range(n)],
            batch=n,
        )
        for src in range_constexpr(n):
            if const_expr(key.fold > 0):
                h = src * hl + lh
                row = ((b // key.qlen) * key.fold + h // 16) * key.qlen + b % key.qlen
                at = (row * 16 + h % 16) * w + j
            else:
                at = (b * key.pad_heads + src * hl + lh) * w + j
            bo.buffer_store(
                fx.Vector.from_elements([got[src][0], got[src][1]], fx.Int32),
                rsrc(gq),
                at,
            )
        _finish(tid, state, e, nblk)

    return _jit(key, gather, nblk, keyed)


def _jit(key, gather, nblk, keyed):
    @flyc.jit
    def launch(
        q: Int64,
        q_sb: Int32,
        q_sh: Int32,
        pe: Int64,
        pe_sb: Int32,
        pe_sh: Int32,
        scale: Int64,
        gq: Int64,
        peers: Int64,
        rank: Int32,
        state: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        gather(
            q,
            q_sb,
            q_sh,
            pe,
            pe_sb,
            pe_sh,
            scale,
            gq,
            peers,
            rank,
            state,
        ).launch(grid=(nblk,), block=(THREADS,), stream=stream)

    ABI.check(gather, launch)
    _BUILDS[key] = launch
    return launch


class GatherWorkspace:
    """Peer buffer, epoch, and the persistent gathered query."""

    def __init__(self, group, device, heads, dim, dtype, pad_heads, max_tokens):
        self.ranks = group.world_size
        self.rank = group.rank_in_group
        self.heads = heads
        self.dim = dim
        self.words = dim * torch.empty((), dtype=dtype).element_size() // 4
        self.pad_heads = pad_heads
        self.max_tokens = max_tokens
        nbytes = peer_bytes(self.ranks, max_tokens, heads, self.words)
        self.peers = PeerBuffer(nbytes, group.cpu_group, self.rank, self.ranks, device)
        self.peers.bytes.zero_()
        self.state = torch.zeros(2, dtype=torch.int32, device=device)
        self.gathered = torch.zeros(
            (max_tokens, pad_heads, dim), dtype=dtype, device=device
        )
        self.folded: dict = {}

    def folded_buffer(self, fold):
        buf = self.folded.get(fold)
        if buf is None:
            buf = torch.zeros(
                (self.max_tokens * fold, 16, self.dim),
                dtype=self.gathered.dtype,
                device=self.gathered.device,
            )
            self.folded[fold] = buf
        return buf


def _launch(ws, key, args):
    out = ws.folded_buffer(key.fold) if key.fold else ws.gathered
    build(key)(
        *args,
        out.data_ptr(),
        ws.peers.addresses.data_ptr(),
        ws.rank,
        ws.state.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    if key.phase == 1:
        return None
    if key.fold:
        return out[: key.tokens * key.fold].view(key.tokens, key.fold * 16, ws.dim)
    return ws.gathered[: key.tokens, : ws.ranks * ws.heads]


def dcp_query_gather(
    ws: GatherWorkspace, query: torch.Tensor, plan=None, phase: int = 0
):
    """Gather ``query`` [tokens, heads, dim] into [tokens, ranks * heads, dim].

    The result is a view of the workspace buffer. ``plan`` may set ``fold`` and
    ``qlen`` for the 16-head store layout. ``phase`` is 0 (push and poll),
    1 (push), or 2 (poll).
    """
    esz = query.element_size()
    key = GatherBuild(
        tokens=query.shape[0],
        ranks=ws.ranks,
        heads=ws.heads,
        words=ws.words,
        pad_heads=ws.pad_heads,
        max_tokens=ws.max_tokens,
        fold=0 if plan is None else plan.fold,
        qlen=0 if plan is None else plan.qlen,
        phase=phase,
    )
    args = (
        query.data_ptr(),
        query.stride(0) * esz // 4,
        query.stride(1) * esz // 4,
        0,
        0,
        0,
        0,
    )
    return _launch(ws, key, args)


def dcp_quant_query_gather(
    ws: GatherWorkspace,
    nope: torch.Tensor,
    pe: torch.Tensor,
    scale: torch.Tensor,
    plan=None,
    phase: int = 0,
):
    """Quantize bf16 ``nope`` [tokens, heads, L] and ``pe`` [tokens, heads, P].

    The output is fp8 [tokens, ranks * heads, L + P], scaled by the per-tensor
    ``scale``. ``plan`` and ``phase`` match ``dcp_query_gather``.
    """
    key = GatherBuild(
        tokens=nope.shape[0],
        ranks=ws.ranks,
        heads=ws.heads,
        words=ws.words,
        pad_heads=ws.pad_heads,
        max_tokens=ws.max_tokens,
        nope=nope.shape[2],
        fold=0 if plan is None else plan.fold,
        qlen=0 if plan is None else plan.qlen,
        phase=phase,
    )
    args = (
        nope.data_ptr(),
        nope.stride(0),
        nope.stride(1),
        pe.data_ptr(),
        pe.stride(0),
        pe.stride(1),
        scale.data_ptr(),
    )
    return _launch(ws, key, args)

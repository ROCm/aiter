# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from the Kimi-K3 mono mailbox combine.

"""DCP LSE combine as one FlyDSL mailbox launch.

Each rank pushes every peer's head and LSE as tagged pairs. Each rank polls
its own heads and applies the LSE weights. One CTA owns one (token, local head)
and each of its ``dim / 4`` threads owns four values of the head.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.dcp_mailbox.common import (
    CM_DEV,
    KernelAbi,
    bf2_f32,
    bf16_pair,
    hw_exp2,
    hw_f32,
    key_tuple,
    ld_dev,
    ld_i32,
    load_ptr64,
    pair_layout,
    peer_bases,
    rsrc,
    traced,
    uniform,
)
from aiter.ops.flydsl.dcp_mailbox.peer import PeerBuffer
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    TaggedMailbox,
    atomic_add_agent,
    tagged_peer_address,
    tagged_shift,
)

MAX_THREADS = 1024
TAG_SLOTS = 256
LOG2E = 1.4426950408889634
NEG_INF = float("-inf")
POS_INF = float("inf")
LN2 = 0.6931471805599453

_STREAM = fx.Stream(None)
_BUILDS: dict = {}


@dataclass(frozen=True)
class CombineBuild:
    tokens: int
    ranks: int
    heads: int
    dim: int
    max_tokens: int
    seqs: int
    base_e: bool
    ret_lse: bool


def peer_layout(ranks, max_tokens, heads, dim):
    """One parity: bf16 pairs, then one LSE pair per slot."""
    slots = ranks * max_tokens * heads
    return pair_layout([("data", slots * dim // 2), ("lse", slots)])


def peer_bytes(ranks, max_tokens, heads, dim):
    lay = peer_layout(ranks, max_tokens, heads, dim)
    return 2 * max(o + n for o, n in lay.values())


ABI = KernelAbi(
    (
        "out",
        "out_sb",
        "out_sh",
        "lse",
        "lse_sb",
        "lse_sh",
        "res",
        "res_sb",
        "res_sh",
        "res_lse",
        "seq_lens",
        "qsl",
        "peers",
        "rank",
        "state",
    )
)


def _log(x):
    f = getattr(fmath, "log", None)
    return f(x) if f is not None else hw_f32("llvm.amdgcn.log.f32", x) * LN2


def _log2(x):
    f = getattr(fmath, "log2", None)
    return f(x) if f is not None else hw_f32("llvm.amdgcn.log.f32", x)


def _row_empty(b, seqs, seq_lens, qsl):
    """Row ``b`` is graph padding, or its request has no KV on this rank."""
    rows = ld_i32(qsl, seqs)
    si = fx.Int32(0)
    for i in range_constexpr(seqs):
        si = si + (ld_i32(qsl, i + 1) <= b).select(1, 0)
    si = fx.min(si, seqs - 1)
    return (b >= rows) | (ld_i32(seq_lens, si) == 0)


@traced
def _poll_combine(
    poll,
    own_lse,
    own_data,
    n,
    mt,
    hl,
    words,
    b,
    lh,
    tid,
    base_e,
    ret_lse,
    res,
    res_sb,
    res_sh,
    res_lse,
):
    """Poll head ``lh`` of row ``b`` from every rank and store the LSE combine."""
    j = tid * 2
    own_slot = [(src * mt + b) * hl + lh for src in range(n)]
    got = poll(
        [(own_lse, own_slot[src], 1) for src in range(n)]
        + [(own_data, own_slot[src] * words + j, 2) for src in range(n)],
        batch=2 * n,
    )
    ls = []
    for src in range_constexpr(n):
        v = got[src][0].bitcast(fx.Float32)
        bad = (v != v) | (v == fx.Float32(POS_INF))  # noqa: PLR0124
        ls.append(bad.select(fx.Float32(NEG_INF), v))
    m = ls[0]
    for src in range_constexpr(1, n):
        m = fx.max(m, ls[src])
    m = (m == fx.Float32(NEG_INF)).select(fx.Float32(0.0), m)
    scale = LOG2E if base_e else 1.0
    ex = [hw_exp2((ls[src] - m) * fx.Float32(scale)) for src in range(n)]
    tot = ex[0]
    for src in range_constexpr(1, n):
        tot = tot + ex[src]
    live = tot > fx.Float32(0.0)
    inv = fx.Float32(1.0) / live.select(tot, fx.Float32(1.0))
    wt = [live.select(ex[src] * inv, fx.Float32(0.0)) for src in range(n)]
    acc = [fx.Float32(0.0)] * 4
    for src in range_constexpr(n):
        pair = got[n + src]
        zero = wt[src] == fx.Float32(0.0)
        for k in range_constexpr(2):
            lo, hi = bf2_f32(pair[k])
            acc[2 * k] = acc[2 * k] + zero.select(fx.Float32(0.0), lo * wt[src])
            acc[2 * k + 1] = acc[2 * k + 1] + zero.select(fx.Float32(0.0), hi * wt[src])
    bo.buffer_store(
        fx.Vector.from_elements(
            [
                bf16_pair(acc[0], acc[1]).bitcast(fx.Int32),
                bf16_pair(acc[2], acc[3]).bitcast(fx.Int32),
            ],
            fx.Int32,
        ),
        rsrc(res),
        (b * res_sb + lh * res_sh) // 2 + j,
    )
    if const_expr(ret_lse):  # noqa: SIM102
        if tid == 0:
            if const_expr(base_e):
                g = _log(tot) + m
            else:
                g = _log2(tot) + m
            bo.buffer_store(fx.Float32(g), rsrc(res_lse), b * hl + lh)


@traced
def _finish(tid, state, e, nblk):
    """The last CTA advances the epoch for the next launch."""
    gpu.barrier()
    if tid == 0:
        old = atomic_add_agent(state + fx.Int64(4), 1)
        if old == nblk - 1:
            bo.buffer_store(fx.Int32(0), rsrc(state), 1, cache_modifier=CM_DEV)
            bo.buffer_store(e + 1, rsrc(state), 0, cache_modifier=CM_DEV)


def build(key: CombineBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    n, hl, d, mt = key.ranks, key.heads, key.dim, key.max_tokens
    words = d // 2
    threads = d // 4
    nblk = key.tokens * hl
    lay = peer_layout(n, mt, hl, d)
    half = peer_bytes(n, mt, hl, d) // 2
    keyed = key_tuple(key)
    name = (
        f"aiter_dcp_lse_combine_t{key.tokens}_n{n}_h{hl}_d{d}_m{mt}"
        f"_s{key.seqs}_e{int(key.base_e)}_l{int(key.ret_lse)}"
    )

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def combine(
        out: Int64,
        out_sb: Int32,
        out_sh: Int32,
        lse: Int64,
        lse_sb: Int32,
        lse_sh: Int32,
        res: Int64,
        res_sb: Int32,
        res_sh: Int32,
        res_lse: Int64,
        seq_lens: Int64,
        qsl: Int64,
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
        put, put_words, poll = mb.put, mb.put_words, mb.poll
        flip = fx.Int64(e % 2) * half
        bases = peer_bases(peers, n)
        own = load_ptr64(peers, rank)
        data = [
            tagged_shift(tagged_peer_address(p, lay["data"][0]), flip) for p in bases
        ]
        lsep = [
            tagged_shift(tagged_peer_address(p, lay["lse"][0]), flip) for p in bases
        ]
        own_data = tagged_shift(tagged_peer_address(own, lay["data"][0]), flip)
        own_lse = tagged_shift(tagged_peer_address(own, lay["lse"][0]), flip)
        slot = (rank * mt + b) * hl + lh
        j = tid * 2
        ws = [
            fx.Vector(
                bo.buffer_load(
                    rsrc(out),
                    (b * out_sb + (dest * hl + lh) * out_sh) // 2 + j,
                    vec_width=2,
                    dtype=T.i32,
                )
            )
            for dest in range(n)
        ]
        lv = [
            fx.Float32(
                bo.buffer_load(
                    rsrc(lse),
                    b * lse_sb + (dest * hl + lh) * lse_sh,
                    vec_width=1,
                    dtype=T.f32,
                )
            )
            for dest in range(n)
        ]
        for dest in range_constexpr(n):
            put_words(data[dest], slot * words + j, [ws[dest][0], ws[dest][1]])
        if const_expr(key.seqs > 0):
            empty = _row_empty(b, key.seqs, seq_lens, qsl)
            for dest in range_constexpr(n):
                lv[dest] = empty.select(fx.Float32(NEG_INF), lv[dest])
        for dest in range_constexpr(n):
            if tid == dest:
                put(lsep[dest], slot, lv[dest])
        _poll_combine(
            poll,
            own_lse,
            own_data,
            n,
            mt,
            hl,
            words,
            b,
            lh,
            tid,
            key.base_e,
            key.ret_lse,
            res,
            res_sb,
            res_sh,
            res_lse,
        )
        _finish(tid, state, e, nblk)

    return _jit(key, combine, nblk, threads, keyed)


def _jit(key, combine, nblk, threads, keyed):
    @flyc.jit
    def launch(
        out: Int64,
        out_sb: Int32,
        out_sh: Int32,
        lse: Int64,
        lse_sb: Int32,
        lse_sh: Int32,
        res: Int64,
        res_sb: Int32,
        res_sh: Int32,
        res_lse: Int64,
        seq_lens: Int64,
        qsl: Int64,
        peers: Int64,
        rank: Int32,
        state: Int64,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        combine(
            out,
            out_sb,
            out_sh,
            lse,
            lse_sb,
            lse_sh,
            res,
            res_sb,
            res_sh,
            res_lse,
            seq_lens,
            qsl,
            peers,
            rank,
            state,
        ).launch(grid=(nblk,), block=(threads,), stream=stream)

    ABI.check(combine, launch)
    _BUILDS[key] = launch
    return launch


class CombineWorkspace:
    """Peer buffer and epoch for one (heads per rank, dim).

    ``state`` is [epoch, CTAs done]. Every rank launches in the same order.
    """

    def __init__(self, group, device, heads, dim, max_tokens):
        self.ranks = group.world_size
        self.rank = group.rank_in_group
        self.heads = heads
        self.dim = dim
        self.max_tokens = max_tokens
        nbytes = peer_bytes(self.ranks, max_tokens, heads, dim)
        self.peers = PeerBuffer(nbytes, group.cpu_group, self.rank, self.ranks, device)
        self.peers.bytes.zero_()
        self.state = torch.zeros(2, dtype=torch.int32, device=device)


def dcp_lse_combine(
    ws: CombineWorkspace,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    return_lse: bool = False,
    base_e: bool = True,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    res: torch.Tensor | None = None,
):
    """Combine ``out`` [tokens, ranks * heads, dim] bf16 and ``lse`` [tokens, ranks * heads].

    Returns this rank's heads [tokens, heads, dim]. With ``return_lse``, also
    returns the combined LSE [tokens, heads]. ``base_e`` selects natural-log LSE.
    ``seq_lens`` and ``query_start_loc`` mask empty local KV rows. ``dim`` must
    be a multiple of 4 with ``4 * ranks <= dim <= 4 * MAX_THREADS``.
    """
    tokens, heads, dim = out.shape[0], ws.heads, ws.dim
    assert out.shape[1] == ws.ranks * heads and out.shape[2] == dim
    assert lse.shape == (tokens, ws.ranks * heads)
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert dim % 4 == 0 and 4 * ws.ranks <= dim <= 4 * MAX_THREADS, dim
    if res is None:
        res = out.new_empty((tokens, heads, dim))
    res_lse = torch.empty(
        (tokens, heads) if return_lse else (1,), device=out.device, dtype=torch.float32
    )
    seqs = 0 if seq_lens is None else seq_lens.shape[0]
    key = CombineBuild(
        tokens=tokens,
        ranks=ws.ranks,
        heads=heads,
        dim=dim,
        max_tokens=ws.max_tokens,
        seqs=seqs,
        base_e=base_e,
        ret_lse=return_lse,
    )
    build(key)(
        out.data_ptr(),
        out.stride(0),
        out.stride(1),
        lse.data_ptr(),
        lse.stride(0),
        lse.stride(1),
        res.data_ptr(),
        res.stride(0),
        res.stride(1),
        res_lse.data_ptr(),
        0 if seq_lens is None else seq_lens.data_ptr(),
        0 if query_start_loc is None else query_start_loc.data_ptr(),
        ws.peers.addresses.data_ptr(),
        ws.rank,
        ws.state.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    if return_lse:
        return res, res_lse
    return res

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from the Kimi-K3 mono mailbox combine.

"""DCP LSE combine as one FlyDSL mailbox launch.

Each rank pushes every peer's head and LSE as tagged pairs. Each rank polls
its own heads and applies the LSE weights. One CTA owns one or more tokens for
one local head, and each of its ``dim / 4`` threads owns four head values.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.dcp_mailbox.common import (
    EPOCH_MASK,
    KernelAbi,
    bf2_f32,
    bf16_pair,
    epoch_offset,
    key_tuple,
    pair_layout,
    pair_span,
    token_tile,
)
from aiter.ops.flydsl.dcp_mailbox.common import (
    peer_bytes as peer_storage_bytes,
)
from aiter.ops.flydsl.dcp_mailbox.peer import PeerBuffer
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.communication_ops_utils import (
    CACHE_DEVICE,
    CACHE_SYSTEM,
    TaggedMailbox,
    atomic_add_agent,
    buffer_load_i32,
    buffer_resource,
    load_i64_from_buffer,
    tagged_peer_address,
    tagged_shift,
    wave_uniform_i32,
)
from aiter.ops.flydsl.kernels.kernels_common import LOG2E

MAX_THREADS = 1024
NEG_INF = float("-inf")
POS_INF = float("inf")
RANK_MASK = (1 << 31) - 1
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
    debug: bool
    token_tile: int
    row_mask: bool


def peer_layout(ranks, max_tokens, heads, dim):
    """One parity of 8-byte aligned bf16/LSE pairs.

    The untagged per-rank epoch slots follow both parities in the allocation.
    """
    slots = ranks * max_tokens * heads
    return pair_layout([("data", slots * dim // 2), ("lse", slots)])


def peer_bytes(ranks, max_tokens, heads, dim):
    lay = peer_layout(ranks, max_tokens, heads, dim)
    return peer_storage_bytes(lay, ranks)


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
        "row_counts",
        "peers",
        "rank",
        "state",
    )
)


def _row_empty(b, seqs, seq_lens, qsl):
    """Row ``b`` is graph padding, or its request has no KV on this rank."""
    rows = buffer_load_i32(qsl, seqs)
    si = fx.Int32(0)
    for i in range_constexpr(seqs):
        si = si + (buffer_load_i32(qsl, i + 1) <= b).select(1, 0)
    si = fx.min(si, seqs - 1)
    return (b >= rows) | (buffer_load_i32(seq_lens, si) == 0)


@flyc.jit
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
    ex = [fx.exp2((ls[src] - m) * fx.Float32(scale)) for src in range(n)]
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
        buffer_resource(res),
        (b * res_sb + lh * res_sh) // 2 + j,
    )
    if const_expr(ret_lse):  # noqa: SIM102
        if tid == 0:
            if const_expr(base_e):
                g = fx.log(tot) + m
            else:
                g = fx.log2(tot) + m
            bo.buffer_store(fx.Float32(g), buffer_resource(res_lse), b * hl + lh)


@flyc.jit
def _push_epoch(bid, tid, e, targets):
    """Debug builds publish ``e + 1``, so zero always means never arrived."""
    for dest in range_constexpr(len(targets)):
        if (bid == 0) & (tid == dest):
            bo.buffer_store(
                e + 1,
                buffer_resource(targets[dest]),
                0,
                cache_modifier=CACHE_SYSTEM,
            )


@flyc.jit
def _finish(tid, state, e, nblk):
    """The last CTA advances the epoch for the next launch."""
    gpu.barrier()
    if tid == 0:
        old = atomic_add_agent(state + fx.Int64(4), 1)
        if old == nblk - 1:
            bo.buffer_store(
                fx.Int32(0),
                buffer_resource(state),
                1,
                cache_modifier=CACHE_DEVICE,
            )
            bo.buffer_store(
                (e + 1) & fx.Int32(EPOCH_MASK),
                buffer_resource(state),
                0,
                cache_modifier=CACHE_DEVICE,
            )


def build(key: CombineBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    n, hl, d, mt = key.ranks, key.heads, key.dim, key.max_tokens
    words = d // 2
    threads = d // 4
    tile = key.token_tile
    nblk = ((key.tokens + tile - 1) // tile) * hl
    lay = peer_layout(n, mt, hl, d)
    half = pair_span(lay)
    keyed = key_tuple(key)
    name = (
        f"aiter_dcp_lse_combine_t{key.tokens}_n{n}_h{hl}_d{d}_m{mt}"
        f"_s{key.seqs}_e{int(key.base_e)}_l{int(key.ret_lse)}"
        f"_g{int(key.debug)}_r{int(key.row_mask)}_bt{tile}"
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
        row_counts: Int64,
        peers: Int64,
        rank: Int32,
        state: Int64,
    ):
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        first_b = (bid // hl) * tile
        lh = bid % hl
        e = wave_uniform_i32(buffer_load_i32(state, 0, cache_modifier=CACHE_DEVICE))
        flip = fx.Int64(e % 2) * half
        bases = [load_i64_from_buffer(peers, source) for source in range(n)]
        own = load_i64_from_buffer(peers, rank)
        epoch_base = epoch_offset(lay)
        epoch_targets = [
            tagged_peer_address(peer, epoch_base + rank * 4) for peer in bases
        ]
        own_epoch = tagged_peer_address(own, epoch_base)
        epoch_sources = [
            tagged_peer_address(own_epoch, source * 4) for source in range(n)
        ]
        if const_expr(key.debug):
            _push_epoch(bid, tid, e, epoch_targets)
        mb = TaggedMailbox(
            e % (1 << 30),
            debug=key.debug,
            epoch_addresses=epoch_sources,
            expected_epoch=e + 1,
            diagnostic=state,
        )
        put, put_words, poll = mb.put, mb.put_words, mb.poll
        data = [
            tagged_shift(tagged_peer_address(p, lay["data"][0]), flip) for p in bases
        ]
        lsep = [
            tagged_shift(tagged_peer_address(p, lay["lse"][0]), flip) for p in bases
        ]
        own_data = tagged_shift(tagged_peer_address(own, lay["data"][0]), flip)
        own_lse = tagged_shift(tagged_peer_address(own, lay["lse"][0]), flip)
        j = tid * 2
        for token_offset in range_constexpr(tile):
            b = first_b + token_offset
            if b < key.tokens:
                slot = (rank * mt + b) * hl + lh
                ws = [
                    fx.Vector(
                        bo.buffer_load(
                            buffer_resource(out),
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
                            buffer_resource(lse),
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
                if const_expr(key.row_mask):
                    empty = buffer_load_i32(row_counts, b) == 0
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
        row_counts: Int64,
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
            row_counts,
            peers,
            rank,
            state,
        ).launch(grid=(nblk,), block=(threads,), stream=stream)

    ABI.check(combine, launch)
    _BUILDS[key] = launch
    return launch


class CombineWorkspace:
    """Peer buffer and epoch for one (heads per rank, dim).

    Every rank must issue the identical call sequence, including warmup and
    capture. One extra call can silently misalign every later result.

    ``debug=True`` enables bounded eager diagnostics and cannot be graph-captured.
    """

    def __init__(self, group, device, heads, dim, max_tokens, debug=False):
        self.ranks = group.world_size
        self.rank = group.rank_in_group
        self.group = group.cpu_group
        self.device = device
        self.debug = bool(debug)
        if self.debug and self.ranks > 31:
            raise ValueError("debug diagnostics support at most 31 ranks")
        self.heads = heads
        self.dim = dim
        self.max_tokens = max_tokens
        nbytes = peer_bytes(self.ranks, max_tokens, heads, dim)
        self.peers = PeerBuffer(nbytes, self.group, self.rank, self.ranks, device)
        self.state = torch.zeros(3, dtype=torch.int32, device=device)

    def close(self) -> None:
        """Synchronize outstanding work, then release peer memory."""
        if self.peers is None:
            return
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        self.peers.close_peers()
        dist.barrier(group=self.group)
        self.peers.close()
        self.peers = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def dcp_lse_combine(
    ws: CombineWorkspace,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    return_lse: bool = False,
    base_e: bool = True,
    seq_lens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    row_counts: torch.Tensor | None = None,
    res: torch.Tensor | None = None,
):
    """Combine ``out`` [tokens, ranks * heads, dim] bf16 and ``lse`` [tokens, ranks * heads].

    Returns this rank's heads [tokens, heads, dim]. With ``return_lse``, also
    returns the combined LSE [tokens, heads]. ``base_e`` selects natural-log LSE.
    ``seq_lens`` with ``query_start_loc`` masks empty requests. ``row_counts``
    masks rows with no locally owned keys. ``dim`` must be a multiple of 4 with
    ``4 * ranks <= dim <= 4 * MAX_THREADS``.

    Every rank must issue the identical call sequence. One extra call can
    silently misalign every later result.
    """
    tokens, heads, dim = out.shape[0], ws.heads, ws.dim
    assert out.shape[1] == ws.ranks * heads and out.shape[2] == dim
    assert lse.shape == (tokens, ws.ranks * heads)
    assert out.dtype == torch.bfloat16 and lse.dtype == torch.float32
    if out.device != lse.device or out.device != ws.peers.bytes.device:
        raise ValueError("out, lse, and workspace must use the same device")
    assert dim % 4 == 0 and 4 * ws.ranks <= dim <= 4 * MAX_THREADS, dim
    if tokens > ws.max_tokens:
        raise ValueError(
            f"tokens ({tokens}) exceeds workspace capacity ({ws.max_tokens})"
        )
    if (seq_lens is None) != (query_start_loc is None):
        raise ValueError("seq_lens and query_start_loc must be provided together")
    for name, tensor in (
        ("seq_lens", seq_lens),
        ("query_start_loc", query_start_loc),
        ("row_counts", row_counts),
    ):
        if tensor is not None and tensor.device != out.device:
            raise ValueError(f"{name} must use the output device")
    if row_counts is not None and (
        row_counts.dtype != torch.int32
        or row_counts.shape != (tokens,)
        or not row_counts.is_contiguous()
    ):
        raise ValueError("row_counts must be contiguous int32 with shape [tokens]")
    _prepare_debug(ws)
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
        debug=ws.debug,
        token_tile=token_tile(tokens),
        row_mask=row_counts is not None,
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
        0 if row_counts is None else row_counts.data_ptr(),
        ws.peers.addresses.data_ptr(),
        ws.rank,
        ws.state.data_ptr(),
        stream=torch.cuda.current_stream(),
    )
    _check_debug(ws)
    if return_lse:
        return res, res_lse
    return res


def _prepare_debug(ws) -> None:
    if not ws.debug:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("DCP mailbox debug diagnostics do not support graph capture")
    ws.state[2].zero_()


def _check_debug(ws) -> None:
    if not ws.debug:
        return
    torch.cuda.synchronize(ws.device)
    diagnostic = int(ws.state[2].item()) & 0xFFFFFFFF
    if diagnostic:
        missing = diagnostic & RANK_MASK
        raise RuntimeError(f"DCP mailbox timeout: missing ranks=0x{missing:x}")

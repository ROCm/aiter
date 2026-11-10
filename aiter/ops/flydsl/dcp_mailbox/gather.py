# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from the Kimi-K3 mono mailbox query gather.

"""DCP query all-gather as one FlyDSL mailbox launch.

Each rank pushes its local heads. Each rank polls every source into a persistent
buffer. ``nope > 0`` quantizes cat(ql_nope, q_pe) to fp8 with one per-tensor scale.
``fold > 0`` stores pseudo-request ``req * fold + group`` as 16 heads. One CTA
handles one or more tokens for one local head.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Int32, Int64, T

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.dcp_mailbox.combine import (
    _check_debug,
    _finish,
    _prepare_debug,
    _push_epoch,
)
from aiter.ops.flydsl.dcp_mailbox.common import (
    KernelAbi,
    bf2_f32,
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
    TaggedMailbox,
    buffer_load_i32,
    buffer_resource,
    load_i64_from_buffer,
    tagged_peer_address,
    tagged_shift,
    wave_uniform_i32,
)
from aiter.ops.flydsl.kernels.quant_utils import emit_f32_to_fp8

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
    debug: bool = False
    token_tile: int = 1


def peer_layout(ranks, max_tokens, heads, words):
    """One parity of naturally aligned word/tag pairs.

    The untagged per-rank epoch slots follow both parities in the allocation.
    """
    return pair_layout([("q", ranks * max_tokens * heads * words)])


def peer_bytes(ranks, max_tokens, heads, words):
    lay = peer_layout(ranks, max_tokens, heads, words)
    return peer_storage_bytes(lay, ranks)


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
    if w % 2 or w > 2 * THREADS:
        raise ValueError(f"gather words per head must be even and <= 256, got {w}")
    if key.nope % 8:
        raise ValueError(f"quantized no-PE dimension must divide 8, got {key.nope}")
    tile = key.token_tile
    nblk = ((key.tokens + tile - 1) // tile) * hl
    lay = peer_layout(n, mt, hl, w)
    half = pair_span(lay)
    keyed = key_tuple(key)
    name = (
        f"aiter_dcp_qgather_t{key.tokens}_n{n}_h{hl}_w{w}_p{key.pad_heads}"
        f"_m{mt}_q{key.nope}_f{key.fold}x{key.qlen}_s{key.phase}"
        f"_g{int(key.debug)}_bt{tile}"
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
        put_words, poll = mb.put_words, mb.poll
        to = [
            tagged_shift(tagged_peer_address(peer, lay["q"][0]), flip) for peer in bases
        ]
        mine = tagged_shift(tagged_peer_address(own, lay["q"][0]), flip)
        j = fx.min(tid, w // 2 - 1) * 2
        push = key.phase != 2
        for token_offset in range_constexpr(tile):
            b = first_b + token_offset
            if b < key.tokens:
                slot = (rank * mt + b) * hl + lh
                if const_expr(push and key.nope > 0):
                    e0 = j * 4
                    in_nope = e0 < key.nope
                    vn = fx.Vector(
                        bo.buffer_load(
                            buffer_resource(q),
                            (b * q_sb + lh * q_sh + fx.min(e0, key.nope - 8)) // 2,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    vp = fx.Vector(
                        bo.buffer_load(
                            buffer_resource(pe),
                            (b * pe_sb + lh * pe_sh + fx.max(e0 - key.nope, 0)) // 2,
                            vec_width=4,
                            dtype=T.i32,
                        )
                    )
                    inv = fx.Float32(1.0) / fx.Float32(
                        bo.buffer_load(
                            buffer_resource(scale), 0, vec_width=1, dtype=T.f32
                        )
                    )
                    xs = []
                    for k in range_constexpr(4):
                        lo, hi = bf2_f32(in_nope.select(vn[k], vp[k]))
                        xs += [lo, hi]
                    ys = [
                        fx.max(
                            fx.min(x * inv, fx.Float32(FP8_MAX)),
                            fx.Float32(-FP8_MAX),
                        )
                        for x in xs
                    ]
                    v = [
                        emit_f32_to_fp8(ys[0], ys[1])
                        | (emit_f32_to_fp8(ys[2], ys[3]) << 16),
                        emit_f32_to_fp8(ys[4], ys[5])
                        | (emit_f32_to_fp8(ys[6], ys[7]) << 16),
                    ]
                if const_expr(push and key.nope == 0):
                    vw = fx.Vector(
                        bo.buffer_load(
                            buffer_resource(q),
                            b * q_sb + lh * q_sh + j,
                            vec_width=2,
                            dtype=T.i32,
                        )
                    )
                    v = [vw[0], vw[1]]
                if const_expr(push):
                    for dest in range_constexpr(n):
                        put_words(to[dest], slot * w + j, [v[0], v[1]])
                if const_expr(key.phase != 1):
                    got = poll(
                        [
                            (mine, ((src * mt + b) * hl + lh) * w + j, 2)
                            for src in range(n)
                        ],
                        batch=n,
                    )
                    for src in range_constexpr(n):
                        if const_expr(key.fold > 0):
                            h = src * hl + lh
                            row = (
                                (b // key.qlen) * key.fold + h // 16
                            ) * key.qlen + b % key.qlen
                            at = (row * 16 + h % 16) * w + j
                        else:
                            at = (b * key.pad_heads + src * hl + lh) * w + j
                        bo.buffer_store(
                            fx.Vector.from_elements(
                                [got[src][0], got[src][1]], fx.Int32
                            ),
                            buffer_resource(gq),
                            at,
                        )
        if const_expr(key.phase != 1):
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
    """Peer buffer, epoch, and the persistent gathered query.

    Every rank must issue the identical call sequence, including warmup and
    capture. One extra call can silently misalign every later result.

    ``debug=True`` enables bounded eager diagnostics and cannot be graph-captured.
    """

    def __init__(
        self, group, device, heads, dim, dtype, pad_heads, max_tokens, debug=False
    ):
        self.ranks = group.world_size
        self.rank = group.rank_in_group
        self.group = group.cpu_group
        self.device = device
        self.debug = bool(debug)
        if self.debug and self.ranks > 31:
            raise ValueError("debug diagnostics support at most 31 ranks")
        self.heads = heads
        self.dim = dim
        self.dtype = dtype
        self.words = dim * torch.empty((), dtype=dtype).element_size() // 4
        self.pad_heads = pad_heads
        self.max_tokens = max_tokens
        nbytes = peer_bytes(self.ranks, max_tokens, heads, self.words)
        self.peers = PeerBuffer(nbytes, self.group, self.rank, self.ranks, device)
        self.state = torch.zeros(3, dtype=torch.int32, device=device)
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


def _launch(ws, key, args):
    if key.tokens > ws.max_tokens:
        raise ValueError(
            f"tokens ({key.tokens}) exceeds workspace capacity ({ws.max_tokens})"
        )
    if key.fold and (key.qlen <= 0 or key.tokens % key.qlen):
        raise ValueError("folded gather requires qlen > 0 and tokens divisible by qlen")
    _prepare_debug(ws)
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
    _check_debug(ws)
    if key.fold:
        physical = out[: key.tokens * key.fold].view(
            key.tokens // key.qlen, key.fold, key.qlen, 16, ws.dim
        )
        return physical.permute(0, 2, 1, 3, 4).reshape(
            key.tokens, key.fold * 16, ws.dim
        )
    return ws.gathered[: key.tokens, : ws.ranks * ws.heads]


def dcp_query_gather(
    ws: GatherWorkspace, query: torch.Tensor, plan=None, phase: int = 0
):
    """Gather ``query`` [tokens, heads, dim] into [tokens, ranks * heads, dim].

    The plain result is a view of the workspace buffer. A folded result is
    reordered to token-major layout. ``plan`` may set ``fold`` and ``qlen`` for
    the 16-head store layout. ``phase`` is 0 (push and poll), 1 (push), or 2
    (poll). BF16 supports head dimensions through 512.

    Every rank must issue the identical call sequence. One extra call can
    silently misalign every later result.
    """
    if query.device != ws.gathered.device:
        raise ValueError("query and workspace must use the same device")
    if query.dtype != ws.dtype or query.shape[1:] != (ws.heads, ws.dim):
        raise ValueError("query dtype and [heads, dim] must match the workspace")
    if ws.words > 2 * THREADS:
        raise ValueError(
            f"{ws.dtype} query gather supports at most 256 words per head; "
            f"dim {ws.dim} needs {ws.words}"
        )
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
        debug=ws.debug,
        token_tile=token_tile(query.shape[0]),
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

    Every rank must issue the identical call sequence. One extra call can
    silently misalign every later result.
    """
    if any(tensor.device != ws.gathered.device for tensor in (nope, pe, scale)):
        raise ValueError("nope, pe, scale, and workspace must use the same device")
    if nope.shape[:2] != pe.shape[:2] or nope.shape[1] != ws.heads:
        raise ValueError("nope and pe must have matching [tokens, heads]")
    if nope.shape[2] + pe.shape[2] != ws.dim:
        raise ValueError("nope and pe dimensions must sum to the workspace dim")
    arch = get_gfx_runtime().split(":", 1)[0]
    if arch != "gfx950":
        raise RuntimeError(
            "dcp_quant_query_gather requires OCP-FN FP8 conversion on gfx950, "
            f"got {arch}"
        )
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
        debug=ws.debug,
        token_tile=token_tile(nope.shape[0]),
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

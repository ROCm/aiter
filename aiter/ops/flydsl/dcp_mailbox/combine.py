# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from the Kimi-K3 mono mailbox combine.

"""DCP LSE combine as one FlyDSL mailbox launch.

Each rank pushes every peer's head and LSE as untagged words into the peer's
sentinel-filled mailbox. Each rank polls its own heads, rearms their slots and
applies the LSE weights. One CTA owns one or more tokens for one local head,
and each of its ``dim / 4`` threads owns four head values.
"""

from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
import torch.distributed as dist
from flydsl.expr.typing import Int32, Int64

from aiter.ops.flydsl.dcp_mailbox.common import (
    KernelAbi,
    epoch_offset,
    pair_span,
    token_tile,
    word_layout,
)
from aiter.ops.flydsl.dcp_mailbox.common import (
    peer_bytes as peer_storage_bytes,
)
from aiter.ops.flydsl.dcp_mailbox.peer import PeerBuffer
from aiter.ops.flydsl.kernels.dcp_mailbox.combine import (
    make_kernel as make_combine_kernel,
)

MAX_THREADS = 1024
RANK_MASK = (1 << 31) - 1
_STREAM = fx.Stream(None)
_BUILDS: dict = {}


@dataclass(frozen=True)
class CombineBuild:
    """Compile-time shape of one combine kernel. The token count is a launch
    argument, so one build serves every decode width with the same tile."""

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
    """One parity of untagged 32-bit words: bf16 pairs, then fp32 LSEs.

    The per-rank epoch slots follow both parities in the allocation.
    """
    slots = ranks * max_tokens * heads
    return word_layout([("data", slots * dim // 2), ("lse", slots)])


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
        "tokens",
        "nblk",
    )
)


def build(key: CombineBuild):
    if key in _BUILDS:
        return _BUILDS[key]
    n, hl, d, mt = key.ranks, key.heads, key.dim, key.max_tokens
    threads = d // 4
    tile = key.token_tile
    lay = peer_layout(n, mt, hl, d)
    half = pair_span(lay)
    name = (
        f"aiter_dcp_lse_combine_n{n}_h{hl}_d{d}_m{mt}"
        f"_s{key.seqs}_e{int(key.base_e)}_l{int(key.ret_lse)}"
        f"_g{int(key.debug)}_r{int(key.row_mask)}_bt{tile}"
    )

    combine = make_combine_kernel(key, name, threads, lay, half, epoch_offset(lay))
    return _jit(key, combine, threads)


def _jit(key, combine, threads):
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
        tokens: Int32,
        nblk: Int32,
        stream: fx.Stream = _STREAM,
    ):
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
            tokens,
            nblk,
        ).launch(grid=(nblk, 1, 1), block=(threads, 1, 1), stream=stream)

    ABI.check(combine, launch)
    _BUILDS[key] = launch
    return launch


class CombineWorkspace:
    """Peer buffer and epoch for one (heads per rank, dim).

    Every rank must issue the identical call sequence, including warmup and
    capture. One extra call can silently misalign every later result.

    ``debug=True`` enables bounded eager diagnostics and cannot be graph-captured.
    A workspace that timed out is not reusable: its mailbox may hold payloads
    that a later launch of the same parity would read as fresh.
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
        # Both parities start as the sentinel; the epoch slots after them stay 0.
        lay = peer_layout(self.ranks, max_tokens, heads, dim)
        self.peers.bytes[: epoch_offset(lay)].fill_(0xFF)
        torch.cuda.synchronize(device)
        dist.barrier(group=self.group)
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
    tile = token_tile(tokens)
    key = CombineBuild(
        ranks=ws.ranks,
        heads=heads,
        dim=dim,
        max_tokens=ws.max_tokens,
        seqs=seqs,
        base_e=base_e,
        ret_lse=return_lse,
        debug=ws.debug,
        token_tile=tile,
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
        tokens,
        (tokens + tile - 1) // tile * heads,
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
        raise RuntimeError(
            f"DCP mailbox timeout: missing ranks=0x{missing:x}"
            + (
                " (every peer arrived: a payload never left the sentinel, or"
                " the ranks launched with different token counts)"
                if not missing
                else ""
            )
        )

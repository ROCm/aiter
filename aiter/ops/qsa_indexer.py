# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Complete gfx942 QSA indexers, from projected QK to token selections.

The caller owns projection, cache allocation, metadata and TP communication.
Prep consumes the caller's RoPE table, including MRoPE and YaRN scaling.
"""

from collections.abc import Sequence

import torch

from aiter.ops.flydsl import qsa_indexer as selection
from aiter.ops.flydsl.kernels.qsa.indexer_topk import LONG_KEYS, STAT_SLOTS
from aiter.ops.flydsl.qsa_indexer import _host_lengths
from aiter.ops.triton.attention.qsa_indexer import (
    _check,
    _no_overlap,
    _same_device,
    qsa_indexer_decode_prep,
    qsa_indexer_prefill_prep,
)

MAX_COMPRESSED_KEYS = selection.MAX_COMPRESSED_KEYS
_WIDTH = 2051
_PAD = 512


def qsa_prefill_indexer(
    qk: torch.Tensor,
    *,
    heads: int,
    positions: torch.Tensor,
    logical_positions: torch.Tensor,
    state_slots: torch.Tensor,
    key_state: torch.Tensor,
    rope_state: torch.Tensor,
    write_locs: torch.Tensor,
    member_rows: torch.Tensor,
    group_sequences: torch.Tensor,
    group_ends: torch.Tensor,
    rope_matrix: torch.Tensor,
    compressed: torch.Tensor,
    token_slot_table: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    axis_map: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    seq_lens: Sequence[int],
    extend_lens: Sequence[int],
    q_out: torch.Tensor | None = None,
    row_range: tuple[int, int] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update QSA caches and return int32 [rows,2051] request-local token IDs.

    QK is contiguous BF16 [sum(extend_lens),640], with heads=4 and 1 K head
    of D128. Host seq_lens/extend_lens describe packed requests; each prefix
    (seq_len - extend_len) must be a multiple of four. logical_positions
    must equal prefix+i. group_sequences/group_ends retain the original
    extend write-plan contract and are checked but not read by the kernels.
    member_rows identifies each complete four-row group in the current QK;
    write_locs holds its compressed slot (zero is inert). token_slot_table
    maps logical tokens to physical slots; key j is at table[request,4*j]//4.

    key_state [S,1,128], rope_state [S,3] and compressed [C,1,128] update
    in place. Slots are caller-owned, with eight ring slots per request.
    The norm/RoPE table may contain YaRN scaling. Prep updates all rows,
    even with row_range=(first,end). Caller-owned contiguous out [>=rows,2051]
    preserves unselected rows. Without out, unselected rows are unspecified.
    out and q_out must not overlap inputs. Capture requires a previous call
    with the same host layout and stream; captured plans remain pinned.
    """
    if qk.device.type != "cuda":
        raise ValueError("QSA prefill requires a GPU")
    if torch.compiler.is_compiling():
        raise RuntimeError("QSA prefill indexer does not support torch.compile")
    if heads != 4:
        raise ValueError("QSA prefill requires heads=4")
    seq_lens = _host_lengths(seq_lens, "seq_lens")
    extends = _host_lengths(extend_lens, "extend_lens")
    rows = sum(extends)
    if (
        not seq_lens
        or len(seq_lens) != len(extends)
        or rows == 0
        or any(s < e for s, e in zip(seq_lens, extends))
    ):
        raise ValueError("Invalid host sequence/extend lengths")
    if any((s - e) % 4 for s, e in zip(seq_lens, extends)):
        raise ValueError("QSA prefill requires prefixes aligned to compress_ratio=4")
    if max(seq_lens) // 4 > MAX_COMPRESSED_KEYS:
        raise ValueError(
            f"QSA prefill supports at most {MAX_COMPRESSED_KEYS} compressed keys"
        )
    _check(qk, "qk", torch.bfloat16, (rows, 640))
    _check(logical_positions, "logical_positions", torch.int64, (rows,))
    groups = write_locs.numel()
    _check(group_sequences, "group_sequences", torch.int64, (groups,))
    _check(group_ends, "group_ends", torch.int64, (groups,))
    if token_slot_table.ndim != 2 or token_slot_table.shape[1] < max(seq_lens):
        raise ValueError("token_slot_table must cover each request's full context")
    _check(
        token_slot_table,
        "token_slot_table",
        torch.int32,
        (len(extends), token_slot_table.shape[1]),
        contiguous=False,
    )
    if token_slot_table.stride(1) != 1:
        raise ValueError("token_slot_table must have unit column stride")
    metadata = (logical_positions, token_slot_table, group_sequences, group_ends)
    _same_device(qk.device, metadata)
    if row_range is not None:
        row_range = tuple(row_range)
        if (
            len(row_range) != 2
            or any(type(r) is not int for r in row_range)
            or not 0 <= row_range[0] <= row_range[1] <= rows
        ):
            raise ValueError("row_range must satisfy 0 <= first <= end <= rows")
        if row_range == (0, rows):
            row_range = None
    if out is not None:
        if out.ndim != 2 or out.shape[0] < rows:
            raise ValueError("out must be int32 [>=rows,2051]")
        _check(out, "out", torch.int32, (out.shape[0], _WIDTH))
        _same_device(qk.device, (out,))
        _no_overlap(
            (out,),
            (
                qk,
                positions,
                logical_positions,
                state_slots,
                key_state,
                rope_state,
                member_rows,
                write_locs,
                group_sequences,
                group_ends,
                rope_matrix,
                compressed,
                token_slot_table,
                cos_sin_cache,
                axis_map,
                q_weight,
                k_weight,
                *(() if q_out is None else (q_out,)),
            ),
        )
    _no_overlap((key_state, rope_state, compressed), metadata)
    if q_out is not None:
        _no_overlap((q_out,), metadata)
    layout = selection._layout(seq_lens, extends, qk.device, row_range)
    q, stats = qsa_indexer_prefill_prep(
        qk,
        positions=positions,
        state_slots=state_slots,
        key_state=key_state,
        rope_state=rope_state,
        member_rows=member_rows,
        write_locs=write_locs,
        rope_matrix=rope_matrix,
        compressed=compressed,
        cos_sin_cache=cos_sin_cache,
        axis_map=axis_map,
        q_weight=q_weight,
        k_weight=k_weight,
        q_eps=q_eps,
        k_eps=k_eps,
        q_out=q_out,
    )
    if out is None:
        out = torch.empty((rows, _WIDTH), dtype=torch.int32, device=qk.device)
    return selection.select_prefill(
        q, compressed, token_slot_table, logical_positions, layout, stats, out
    )


def qsa_decode_indexer(
    qk: torch.Tensor,
    *,
    positions: torch.Tensor,
    logical_positions: torch.Tensor,
    state_slots: torch.Tensor,
    key_state: torch.Tensor,
    rope_state: torch.Tensor,
    write_locs: torch.Tensor,
    group_locs: torch.Tensor,
    compressed: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    axis_map: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_eps: float,
    k_eps: float,
    seq_lens: torch.Tensor,
    verify: bool = False,
    q_out: torch.Tensor | None = None,
    logits_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Update QSA state and return int32 [query_rows,2051] token selections.

    All lengths, positions, slots and page tables are GPU tensors. page_table
    is int32 [rows,pages], one row per query, with 16 compressed keys per page.
    lengths is each query's visible compressed-key count; seq_lens is int32
    [rows], not a host request list. group_locs is int32 [rows,4], oldest
    member first. Inert/padded rows use valid dump slots and zero lengths.

    Set verify=True for verify/draft-extend windows (at most four
    rows per request in an eight-slot ring). Raw writes then precede a second
    compression launch. YaRN is encoded in the caller's cos/sin table.
    Warm before graph capture. logits_out, if supplied, is FP32 [rows,width]
    with 512 extra FP32 elements of accessible storage after the view.
    """
    if (
        qk.ndim != 2
        or page_table.ndim != 2
        or qk.shape[0] == 0
        or page_table.shape[1] == 0
    ):
        raise ValueError("QK and page table must have nonempty row/page dimensions")
    if torch.compiler.is_compiling():
        raise RuntimeError("QSA decode indexer does not support torch.compile")
    rows, width = qk.shape[0], page_table.shape[1] * 16
    if width > MAX_COMPRESSED_KEYS or (rows * width + _PAD) * 4 >= 2**31:
        raise ValueError(
            "QSA decode requires <=262144 keys and logits smaller than 2 GiB"
        )
    _check(page_table, "page_table", torch.int32, (rows, page_table.shape[1]))
    _check(lengths, "lengths", torch.int32, (rows,))
    _check(
        logical_positions,
        "logical_positions",
        (torch.int32, torch.int64),
        (rows,),
        contiguous=False,
    )
    _check(seq_lens, "seq_lens", torch.int32, (rows,), contiguous=False)
    extra = (page_table, lengths, logical_positions, seq_lens)
    if logits_out is not None:
        _check(logits_out, "logits_out", torch.float32, (rows, width))
        required = (logits_out.storage_offset() + rows * width + _PAD) * 4
        if logits_out.untyped_storage().nbytes() < required:
            raise ValueError(
                "logits_out storage must extend 512 FP32 values past the view"
            )
        _no_overlap(
            (logits_out,),
            (
                qk,
                positions,
                state_slots,
                key_state,
                rope_state,
                group_locs,
                write_locs,
                compressed,
                cos_sin_cache,
                axis_map,
                q_weight,
                k_weight,
                *extra,
                *(() if q_out is None else (q_out,)),
            ),
        )
        extra += (logits_out,)
    _same_device(qk.device, extra)
    _no_overlap((key_state, rope_state, compressed), extra)
    if q_out is not None:
        _no_overlap((q_out,), extra)
    stats = (
        torch.empty((rows, STAT_SLOTS, 2), dtype=torch.int32, device=qk.device)
        if width > LONG_KEYS
        else None
    )
    q = qsa_indexer_decode_prep(
        qk,
        positions=positions,
        state_slots=state_slots,
        key_state=key_state,
        rope_state=rope_state,
        group_locs=group_locs,
        write_locs=write_locs,
        compressed=compressed,
        cos_sin_cache=cos_sin_cache,
        axis_map=axis_map,
        q_weight=q_weight,
        k_weight=k_weight,
        q_eps=q_eps,
        k_eps=k_eps,
        verify=verify,
        q_out=q_out,
        stats_out=stats,
    )
    return selection.select_decode(
        q,
        compressed,
        page_table,
        lengths,
        logical_positions,
        seq_lens,
        stats=stats,
        logits_out=logits_out,
    )


__all__ = ["MAX_COMPRESSED_KEYS", "qsa_decode_indexer", "qsa_prefill_indexer"]

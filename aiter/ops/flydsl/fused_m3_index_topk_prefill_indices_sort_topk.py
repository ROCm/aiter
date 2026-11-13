# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MiniMax-M3 prefill index top-k fused with the MSA page sort_topk.

flydsl_fused_m3_index_topk_prefill_indices_sort_topk gives each prefill row's
top-16 index blocks (bit for bit vLLM's m3_index_topk_prefill_indices) and,
with sort_topk_q > 1, the page sort_topk of every sort_topk_q consecutive rows that
aiter's Gluon MSA walks at BLOCK_M = 16 * sort_topk_q
(aiter.ops.triton.attention.msa.msa_fwd(sort_topk=...)), bit for bit what its
msa_sort_topk_kernel pre-pass builds.
"""

import functools

import torch

from .kernels.fused_m3_index_topk_prefill_indices_sort_topk import (
    BLK,
    PREFILL_STRIPS,
    TOPK,
    build_fused_m3_index_topk_prefill_indices_sort_topk_module,
)
from .kernels.tensor_shim import _run_compiled

MAX_SLOTS = 128  # 8192 blocks: the block index bits the kernel keys carry


@functools.cache
def _get_launcher(strips: int, sort_topk_q: int):
    return build_fused_m3_index_topk_prefill_indices_sort_topk_module(strips, sort_topk_q)


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n >= 1 else 1


_PLANS: dict = {}


def _validate(
    score, topk_idx, max_seq_len, num_valid_pages, sort_topk_q, block_table, query_req,
    query_pos, sort_topk_out,
):  # fmt: skip
    """The call's checks; returns (scanned blocks, row groups, window strips)."""
    heads, rows, s_width = score.shape
    if score.dtype != torch.float32 or score.stride(2) != 1:
        raise ValueError("score must be fp32 [H, rows, S], contiguous in S")
    if topk_idx.dtype != torch.int32 or topk_idx.stride(2) != 1:
        raise ValueError("topk_idx must be int32 [H, rows, 16]")
    if tuple(topk_idx.shape) != (heads, rows, TOPK):
        raise ValueError(f"topk_idx must be [{heads}, {rows}, {TOPK}]")
    if num_valid_pages.dtype != torch.int32 or not num_valid_pages.is_contiguous():
        raise ValueError("num_valid_pages must be contiguous int32")
    if num_valid_pages.numel() < rows:
        raise ValueError("num_valid_pages must cover every row")
    slots = _pow2_ceil((max_seq_len + BLK * 64 - 1) // (BLK * 64))
    if slots > MAX_SLOTS or s_width < slots * 64:
        raise ValueError("score is narrower than max_seq_len's blocks (or > 8192 blocks)")
    if sort_topk_q not in (1, 2, 4):
        raise ValueError("sort_topk_q must be 1, 2 or 4")
    groups = (rows + sort_topk_q - 1) // sort_topk_q
    if sort_topk_q > 1:
        if block_table is None or query_req is None or query_pos is None:
            raise ValueError("sort_topk_q > 1 needs block_table, query_req and query_pos")
        for name, x in (("block_table", block_table), ("query_req", query_req),
                        ("query_pos", query_pos)):  # fmt: skip
            if x.dtype != torch.int32 or not x.is_contiguous():
                raise ValueError(f"{name} must be contiguous int32")
        if block_table.dim() != 2:
            raise ValueError("block_table must be [reqs, max_blocks]")
        if query_req.numel() < rows or query_pos.numel() < rows:
            raise ValueError("query_req / query_pos must cover every row")
        if sort_topk_out is not None:
            c = sort_topk_q * TOPK
            uniq, count, tokblk, pos = sort_topk_out
            if (tuple(uniq.shape) != (heads, groups, c) or tuple(count.shape) != (heads, groups)
                    or tuple(tokblk.shape) != (heads, groups, c, sort_topk_q)
                    or pos.numel() < groups * sort_topk_q):  # fmt: skip
                raise ValueError("sort_topk_out buffers have the wrong shapes")
            if not all(x.dtype == torch.int32 and x.is_contiguous() for x in sort_topk_out):
                raise ValueError("sort_topk_out buffers must be contiguous int32")
    strips = next((s for s in PREFILL_STRIPS if max_seq_len <= s * BLK * 64), PREFILL_STRIPS[-1])
    return slots * 64, groups, strips


def flydsl_fused_m3_index_topk_prefill_indices_sort_topk(
    score: torch.Tensor,
    topk_idx: torch.Tensor,
    *,
    max_seq_len: int,
    num_valid_pages: torch.Tensor,
    sort_topk_q: int = 1,
    block_table: torch.Tensor | None = None,
    query_req: torch.Tensor | None = None,
    query_pos: torch.Tensor | None = None,
    sort_topk_out: tuple | None = None,
) -> tuple | None:
    """Top-16 blocks of ragged prefill rows, and (sort_topk_q > 1) the MSA page
    sort_topk of every sort_topk_q consecutive rows.

    score            fp32 [H, rows, S], contiguous in S: index scores per block
    topk_idx         int32 [H, rows, 16] out: block ids in AITER's
                     pa_sparse_block_topk order, -1 padded
    max_seq_len      the longest sequence; the kernel scans
                     pow2(cdiv(max_seq_len, 128 * 64)) * 64 blocks of a row
                     (S must cover them)
    num_valid_pages  int32 [>= rows]: row n's visible blocks

    sort_topk_q in (2, 4) also needs block_table [reqs, max_blocks] and
    query_req / query_pos [rows] (int32, contiguous) and returns
    (uniq [H, groups, C], count [H, groups], tokblk [H, groups, C, sort_topk_q],
    pos [groups * sort_topk_q]), C = 16 * sort_topk_q, groups = cdiv(rows, sort_topk_q),
    for msa_fwd(sort_topk=...); sort_topk_out gives those buffers (e.g. for CUDA
    graphs). sort_topk_q = 1 returns None.

    The first call per (window size, sort_topk_q, device) JIT-compiles; warm up
    before capturing a CUDA graph.
    """
    heads, rows, _ = score.shape
    # Validation once per signature (shapes / strides / dtypes); a call then
    # only binds pointers and launches.
    sig = (
        score.shape, score.stride(), score.dtype, topk_idx.shape, topk_idx.stride(),
        topk_idx.dtype, num_valid_pages.shape, num_valid_pages.stride(), num_valid_pages.dtype,
        max_seq_len, sort_topk_q,
        None if block_table is None else (block_table.shape, block_table.stride(), block_table.dtype),
        None if query_req is None else (query_req.shape, query_req.stride(), query_req.dtype),
        None if query_pos is None else (query_pos.shape, query_pos.stride(), query_pos.dtype),
        None if sort_topk_out is None
        else tuple((x.shape, x.stride(), x.dtype) for x in sort_topk_out),
    )  # fmt: skip
    plan = _PLANS.get(sig)
    if plan is None:
        if len(_PLANS) > 4096:
            _PLANS.clear()
        plan = _PLANS[sig] = _validate(
            score, topk_idx, max_seq_len, num_valid_pages, sort_topk_q, block_table,
            query_req, query_pos, sort_topk_out,
        )  # fmt: skip
    cols, groups, strips = plan
    out = None
    sort_topk_args = (rows, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    if sort_topk_q > 1:
        if sort_topk_out is None:
            c = sort_topk_q * TOPK
            dev = score.device
            sort_topk_out = (
                torch.empty((heads, groups, c), dtype=torch.int32, device=dev),
                torch.empty((heads, groups), dtype=torch.int32, device=dev),
                torch.empty((heads, groups, c, sort_topk_q), dtype=torch.int32, device=dev),
                torch.empty((groups * sort_topk_q,), dtype=torch.int32, device=dev),
            )
        uniq, count, tokblk, pos = sort_topk_out
        out = sort_topk_out
        sort_topk_args = (
            groups,
            block_table.data_ptr(),
            block_table.stride(0),
            block_table.numel(),
            query_req.data_ptr(),
            query_pos.data_ptr(),
            uniq.data_ptr(),
            count.data_ptr(),
            tokblk.data_ptr(),
            pos.data_ptr(),
        )
    if rows == 0:
        return out
    _run_compiled(
        _get_launcher(strips, sort_topk_q),
        score.data_ptr(),
        score.stride(0),
        score.stride(1),
        cols,
        num_valid_pages.data_ptr(),
        topk_idx.data_ptr(),
        topk_idx.stride(0),
        topk_idx.stride(1),
        rows,
        heads,
        *sort_topk_args,
        # Bound to the tensors' device, not the ambient one.
        torch.cuda.current_stream(score.device),
        # A compiled handle launches on the device it was built for.
        specialization_key=(score.device.index,),
    )
    return out

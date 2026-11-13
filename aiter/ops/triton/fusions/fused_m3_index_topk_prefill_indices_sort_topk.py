# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

# MiniMax-M3 prefill index top-k (vLLM's m3_index_topk_prefill_indices) fused
# with the MSA page sort_topk (msa_sort_topk_kernel): one launch gives the attention
# both its top-k and the row groups' sort_topk. See
# _triton_kernels/fusions/fused_m3_index_topk_prefill_indices_sort_topk.py.

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.fused_m3_index_topk_prefill_indices_sort_topk import (
    fused_m3_index_topk_prefill_indices_sort_topk_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

BLK = 128  # tokens per block
TOPK = 16
MAX_SLOTS = 128  # 8192 blocks, as the FlyDSL top-k
DIRECT_COLS = 256  # up to this width: one top-k pass over the scores
GROUP = 16  # wider: group maxima first (two passes), groups of GROUP blocks
MAX_CHUNK = 1024  # score columns a pass holds at once


def _pow2_ceil(n: int) -> int:
    return 1 << (n - 1).bit_length() if n >= 1 else 1


def fused_m3_index_topk_prefill_indices_sort_topk(
    score: torch.Tensor,
    topk_idx: torch.Tensor,
    *,
    max_seq_len: int,
    num_valid_pages: torch.Tensor,
    block_q: int = 1,
    block_table: torch.Tensor | None = None,
    query_req: torch.Tensor | None = None,
    query_pos: torch.Tensor | None = None,
    sort_topk_out: tuple | None = None,
    num_warps: int | None = None,
    backend: str = "triton",
    strips: int | None = None,
):
    """Top-16 blocks of ragged prefill rows, and (block_q > 1) the MSA page
    sort_topk of every block_q consecutive rows.

    score            fp32 [H, rows, S], contiguous in S (index scores per block)
    topk_idx         int32 [H, rows, 16] out: block ids in AITER's order, -1 pad
    max_seq_len      longest sequence (sets the scanned width, as the FlyDSL
                     kernel: pow2(cdiv(max_seq_len, 128 * 64)) * 64 blocks)
    num_valid_pages  int32 [>= rows]: row n's visible blocks

    block_q > 1 also needs block_table [reqs, max_blocks], query_req and
    query_pos [rows] (int32), and returns msa_fwd's precomputed sort_topk
    (uniq [H, groups, C], count [H, groups], tokblk [H, groups, C, block_q],
    pos [groups * block_q]), C = block_q * 16, groups = cdiv(rows, block_q);
    sort_topk_out gives those buffers (e.g. for CUDA graphs). block_q = 1 returns
    None.

    backend: "triton" (tl.topk over every score, any arch) or "gluon" (gfx950:
    FlyDSL's bound-and-compact walk, one warp per row, windows of strips x 64
    blocks).
    """
    heads, rows, s_width = score.shape
    if score.dtype != torch.float32 or score.stride(2) != 1:
        raise ValueError("score must be fp32 [H, rows, S], contiguous in S")
    if topk_idx.dtype != torch.int32 or topk_idx.stride(2) != 1:
        raise ValueError("topk_idx must be int32 [H, rows, 16]")
    if topk_idx.shape != (heads, rows, TOPK):
        raise ValueError(f"topk_idx must be [{heads}, {rows}, {TOPK}]")
    if num_valid_pages.dtype != torch.int32 or not num_valid_pages.is_contiguous():
        raise ValueError("num_valid_pages must be contiguous int32")
    if num_valid_pages.numel() < rows:
        raise ValueError("num_valid_pages must cover every row")
    slots = _pow2_ceil(triton.cdiv(max_seq_len, BLK * 64))
    if slots > MAX_SLOTS or s_width < slots * 64:
        raise ValueError("score is narrower than max_seq_len's blocks")
    cols = slots * 64
    if block_q < 1 or block_q & (block_q - 1) or block_q > 8:
        raise ValueError("block_q must be 1, 2, 4 or 8")
    sort_topk = block_q > 1
    groups = triton.cdiv(rows, block_q)
    out = None
    if sort_topk:
        if block_table is None or query_req is None or query_pos is None:
            raise ValueError("block_q > 1 needs block_table, query_req and query_pos")
        for name, x in (("block_table", block_table), ("query_req", query_req),
                        ("query_pos", query_pos)):  # fmt: skip
            if x.dtype != torch.int32:
                raise ValueError(f"{name} must be int32")
        if query_req.numel() < rows or query_pos.numel() < rows or block_table.stride(1) != 1:
            raise ValueError("query_req / query_pos must cover every row")
        c = block_q * TOPK
        dev = score.device
        if sort_topk_out is None:
            sort_topk_out = (
                torch.empty((heads, groups, c), dtype=torch.int32, device=dev),
                torch.empty((heads, groups), dtype=torch.int32, device=dev),
                torch.empty((heads, groups, c, block_q), dtype=torch.int32, device=dev),
                torch.empty((groups * block_q,), dtype=torch.int32, device=dev),
            )
        uniq, count, tokblk, pos = sort_topk_out
        if (uniq.shape != (heads, groups, c) or count.shape != (heads, groups)
                or tokblk.shape != (heads, groups, c, block_q)
                or pos.numel() < groups * block_q):  # fmt: skip
            raise ValueError("sort_topk_out buffers have the wrong shapes")
        out = sort_topk_out
    else:
        block_table = query_req = query_pos = num_valid_pages  # unused
        uniq = count = tokblk = pos = topk_idx  # unused
    if rows == 0:
        return out
    if backend == "gluon":
        from aiter.ops.triton._gluon_kernels.gfx950.fusions.fused_m3_index_topk_prefill_indices_sort_topk import (
            fused_m3_index_topk_prefill_indices_sort_topk_gfx950_gluon_kernel as gk,
        )

        if block_q > 4:
            raise ValueError("the gluon backend takes block_q 1, 2 or 4")
        if strips is None:
            strips = min(max(cols // 64, 2), 16)
        gk[(groups, heads)](
            score, num_valid_pages, topk_idx, block_table, query_req, query_pos,
            uniq, count, tokblk, pos, rows, cols,
            score.stride(0), score.stride(1), topk_idx.stride(0), topk_idx.stride(1),
            block_table.stride(0),
            TQ=block_q, STRIPS=strips, SORT_TOPK=sort_topk, num_warps=block_q,
        )  # fmt: skip
        return out
    if backend != "triton":
        raise ValueError(f"unknown backend {backend!r}")
    chunk = min(cols, MAX_CHUNK)
    two_pass = cols > DIRECT_COLS
    if num_warps is None:
        # one wave per program unless a row group streams long rows (MI355X sweep)
        num_warps = 1 if cols <= DIRECT_COLS or block_q == 1 else 4
    _LOGGER.info(
        f"FUSED_M3_INDEX_TOPK_PREFILL_INDICES_SORT_TOPK: score={tuple(score.shape)} "
        f"block_q={block_q} chunk={chunk}"
    )
    fused_m3_index_topk_prefill_indices_sort_topk_kernel[(groups, heads)](
        score, num_valid_pages, topk_idx, block_table, query_req, query_pos,
        uniq, count, tokblk, pos, rows, cols,
        score.stride(0), score.stride(1), topk_idx.stride(0), topk_idx.stride(1),
        block_table.stride(0),
        TOPK=TOPK, TQ=block_q, CHUNK=chunk, TWO_PASS=two_pass, G=GROUP, SORT_TOPK=sort_topk,
        num_warps=num_warps,
    )  # fmt: skip
    return out

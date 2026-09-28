# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL DCP decode TopK merge interface."""

import functools

import torch

from .kernels.dcp_topk_merge import build_dcp_topk_merge_module
from .kernels.kernels_common import get_warp_size
from .kernels.tensor_shim import _run_compiled


@functools.cache
def _get_launcher(n_cand, k_loc, topk_tokens, page_size, world_size):
    return build_dcp_topk_merge_module(
        n_cand, k_loc, topk_tokens, page_size, world_size
    )


@functools.lru_cache(maxsize=128)
def _validate(
    sc_shape,
    sc_dtype,
    li_shape,
    li_dtype,
    bt_shape,
    bt_dtype,
    idx_numel,
    idx_dtype,
    indptr_shape,
    indptr_dtype,
    counts_shape,
    counts_dtype,
    stage_shape,
    stage_dtype,
    dcp_rank,
    world_size,
    topk_tokens,
    page_size,
    device_warp,
):
    if len(sc_shape) != 2 or sc_dtype != torch.float32:
        raise ValueError("gathered_scores must be a 2D float32 tensor")
    rows, n_cand = sc_shape
    if world_size <= 0 or n_cand % world_size:
        raise ValueError("gathered_scores.shape[1] must be divisible by world_size")
    k_loc = n_cand // world_size
    if not 0 <= dcp_rank < world_size:
        raise ValueError("dcp_rank out of range")
    if li_shape != (rows, k_loc) or li_dtype != torch.int32:
        raise ValueError("local_idx must be int32 [rows, n_cand // world_size]")
    # The kernel slices this 2D (fx.slice(block_table, (row, None))), so a 1D
    # tensor fails deep inside the JIT rather than here.
    if len(bt_shape) != 2:
        raise ValueError(f"block_table must be 2D [rows, max_blocks], got {bt_shape}")
    # rows == num_req: DCP+DSA is qlen=1 decode only.
    if bt_shape[0] != rows:
        raise ValueError("block_table.shape[0] must equal rows (qlen=1 decode only)")
    # The kernel reads every one of these as i32. torch builds page tables as
    # int64 by default, and an int64 block_table passes a shape-only check and
    # then yields silently wrong physical slots -- no crash, wrong KV.
    if bt_dtype != torch.int32:
        raise ValueError("block_table must be int32")
    if idx_dtype != torch.int32:
        raise ValueError("out_kv_indices must be int32")
    if indptr_dtype != torch.int32:
        raise ValueError("out_kv_indptr must be int32")
    if counts_dtype != torch.int32:
        raise ValueError("owned_counts must be int32")
    if stage_shape != (rows, k_loc):
        raise ValueError("staging must be int32 [rows, n_cand // world_size]")
    if stage_dtype != torch.int32:
        raise ValueError("staging must be int32 [rows, n_cand // world_size]")
    if indptr_shape[0] < rows + 1:
        raise ValueError("out_kv_indptr must hold rows + 1 entries")
    if counts_shape[0] < rows:
        raise ValueError("owned_counts must hold rows entries")
    # Per row a rank owns at most its own plane (k_loc) and at most the whole
    # global winner set (topk_tokens) -- whichever binds first. Requiring k_loc
    # unconditionally would reject a valid buffer whenever topk_tokens < k_loc.
    if idx_numel < rows * min(k_loc, topk_tokens):
        raise ValueError("out_kv_indices too small for the worst case")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if device_warp != 64:
        raise ValueError("the FlyDSL DCP merge kernel requires a wave64 GPU")


def flydsl_dcp_topk_merge(
    gathered_scores: torch.Tensor,
    local_idx: torch.Tensor,
    block_table: torch.Tensor,
    out_kv_indices: torch.Tensor,
    out_kv_indptr: torch.Tensor,
    owned_counts: torch.Tensor,
    staging: torch.Tensor,
    dcp_rank: int,
    world_size: int,
    topk_tokens: int,
    page_size: int,
) -> None:
    """Select this DCP rank's share of the global top-k and emit packed KV slots.

    `gathered_scores[:, c]` MUST come from rank `c // k_loc` at that rank's local
    top-k position `c % k_loc` -- the natural all_gather(dim=0) order. Ownership
    is positional, so a permuted layout silently produces wrong KV.

    PRECONDITION: unused candidate slots in `gathered_scores` must be -inf, not
    0.0. Padding is detected via `local_idx < 0` for THIS rank's plane only; the
    other W-1 planes carry no liveness signal, so their padding has to lose the
    global threshold comparison on score alone. Logits are routinely negative,
    so a 0.0 pad outranks real candidates and this rank silently under-emits.
    `top_k_per_row_decode(values=...)` pads with -inf for exactly this reason --
    an aiter predating that fix cannot feed this op.


    First call for a new (n_cand, k_loc, topk, page_size, world_size) JIT-compiles
    the kernels, so warm up each shape before capturing a CUDAGraph.
    """
    _validate(
        tuple(gathered_scores.shape),
        gathered_scores.dtype,
        tuple(local_idx.shape),
        local_idx.dtype,
        tuple(block_table.shape),
        block_table.dtype,
        out_kv_indices.numel(),
        out_kv_indices.dtype,
        tuple(out_kv_indptr.shape),
        out_kv_indptr.dtype,
        tuple(owned_counts.shape),
        owned_counts.dtype,
        tuple(staging.shape),
        staging.dtype,
        dcp_rank,
        world_size,
        topk_tokens,
        page_size,
        get_warp_size(),
    )
    rows, n_cand = gathered_scores.shape
    k_loc = n_cand // world_size
    launcher = _get_launcher(n_cand, k_loc, topk_tokens, page_size, world_size)
    _run_compiled(
        launcher,
        gathered_scores,
        local_idx,
        block_table,
        out_kv_indices,
        out_kv_indptr,
        staging,
        owned_counts,
        rows,
        dcp_rank,
        # Bound to the tensors' device: current_stream() with no argument
        # returns a stream on the *ambient* device, which is a different device
        # entirely if the caller did not set one -- an invalid launch, not a
        # slow one.
        torch.cuda.current_stream(gathered_scores.device),
    )

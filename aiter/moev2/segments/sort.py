# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Token sorting: route each token to its experts in blocks of `block_m` rows.

The backend is chosen as in the legacy `moe_sorting`, from the same environment variables:
Opus by default, CK with AITER_USE_CK_MOE_SORTING=1, FlyDSL with
AITER_USE_FLYDSL_MOE_SORTING=1. This module covers the plain sort (no local topk ids, no
auxiliary outputs, not FLAT); the legacy `moe_sorting` stays in aiter/fused_moe.py,
where tests patch its helpers. `hooks.set_sort_impl` replaces `moe_sorting` here.
"""

import os

import torch

import aiter
from aiter import dtypes, logger

from .. import hooks
from ..dispatch.buf import Buf
from ..dispatch.ops import Op
from .base import Cap, Segment

_USE_CK_MOE_SORTING = os.environ.get("AITER_USE_CK_MOE_SORTING", "0") == "1"
_USE_FLYDSL_MOE_SORTING = os.environ.get("AITER_USE_FLYDSL_MOE_SORTING", "0") == "1"


def _sorted_buffers(
    topk_ids,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    accumulate,
    expert_mask,
    output,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    max_num_tokens_padded = int(topk_ids.numel() + num_experts * block_size - topk)
    max_num_m_blocks = int((max_num_tokens_padded + block_size - 1) // block_size)
    max_num_tokens_padded = max_num_m_blocks * block_size
    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(
        max_num_tokens_padded, dtype=dtypes.fp32, device=device
    )
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    # Stage2 accumulates into [M, model_dim] (also under EP); otherwise the caller owns
    # the intermediate and this is a placeholder. A caller buffer can stand in: the
    # sort kernel zeroes what it is handed.
    if accumulate or expert_mask is not None:
        moe_buf = (
            output
            if output is not None
            else torch.empty((M, model_dim), dtype=moebuf_dtype, device=device)
        )
    else:
        moe_buf = torch.empty((0, 0), dtype=moebuf_dtype, device=device)
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf


def _flydsl_moe_sorting(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask,
    num_local_tokens,
    accumulate=True,
    output=None,
):
    from aiter.ops.flydsl.moe_sorting import flydsl_moe_sorting_fwd

    ret = _sorted_buffers(
        topk_ids,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size,
        accumulate,
        expert_mask,
        output,
    )
    flydsl_moe_sorting_fwd(
        topk_ids,
        topk_weights,
        *ret,
        num_experts,
        int(block_size),
        expert_mask,
        num_local_tokens,
    )
    return ret


def _moe_sorting_impl(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask,
    num_local_tokens,
    dispatch_policy,
    use_opus,
    accumulate=True,
    output=None,
):
    device = topk_ids.device
    M, topk = topk_ids.shape
    ret = _sorted_buffers(
        topk_ids,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size,
        accumulate,
        expert_mask,
        output,
    )
    if use_opus:
        ws_size = aiter.moe_sorting_opus_get_workspace_size(
            M, num_experts, topk, dispatch_policy, topk_ids.get_device()
        )
        workspace = (
            torch.empty(ws_size, dtype=torch.uint8, device=device)
            if ws_size > 0
            else None
        )
        aiter.moe_sorting_opus_fwd(
            topk_ids,
            topk_weights,
            *ret,
            num_experts,
            int(block_size),
            expert_mask,
            num_local_tokens,
            workspace,
            dispatch_policy,
            None,
            None,
            None,
        )
    else:
        aiter.moe_sorting_fwd(
            topk_ids,
            topk_weights,
            *ret,
            num_experts,
            int(block_size),
            expert_mask,
            num_local_tokens,
            dispatch_policy,
        )
    return ret


def moe_sorting(
    topk_ids,
    topk_weights,
    num_experts,
    model_dim,
    moebuf_dtype,
    block_size,
    expert_mask=None,
    num_local_tokens=None,
    dispatch_policy=0,
    accumulate=True,
    output=None,
):
    """`(sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf)`."""
    if not _USE_CK_MOE_SORTING and _USE_FLYDSL_MOE_SORTING and dispatch_policy == 0:
        return _flydsl_moe_sorting(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            accumulate=accumulate,
            output=output,
        )
    try:
        return _moe_sorting_impl(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            use_opus=not _USE_CK_MOE_SORTING,
            accumulate=accumulate,
            output=output,
        )
    except Exception as e:
        logger.error(f"Error in moe_sorting: {e}")
        max_num_tokens_padded = int(
            topk_ids.numel() + num_experts * block_size - topk_ids.shape[1]
        )
        topk = topk_ids.shape[1]
        logger.error(
            f"Moe_sorting info: {max_num_tokens_padded=} {block_size=} {num_experts=} {topk=} {topk_ids.shape=}"
        )
        raise


class Sort(Segment):
    name = "sort"
    covers = (Op.SORT, Op.SORT)
    structure_params = frozenset({"block_m"})
    capabilities = (
        Cap(
            features=frozenset(
                {
                    "ep_mask",
                    "num_local_tokens",
                    "doweight_stage1",
                    "act_scales",
                    "a1_scale",
                }
            )
        ),
    )

    def produces(self, p, params):
        return Buf("sorted", None, block_m=params.get("block_m"))

    def run(
        self,
        topk_ids,
        topk_weights,
        num_experts,
        model_dim,
        moebuf_dtype,
        block_size,
        expert_mask,
        num_local_tokens,
        dispatch_policy,
        output,
    ):
        return hooks.sort_impl(moe_sorting)(
            topk_ids,
            topk_weights,
            num_experts,
            model_dim,
            moebuf_dtype,
            block_size,
            expert_mask,
            num_local_tokens,
            dispatch_policy,
            accumulate=True,
            output=output,
        )

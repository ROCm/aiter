# SPDX-License-Identifier: MIT
"""Top-8 routing with padding and dispatch-counter reset."""

import os

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.fused_topk_dispatch import (
    _fused_topk_dispatch_kernel,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


def fused_topk_dispatch(
    logits: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    real_requests: torch.Tensor | None,
    pad_rows: torch.Tensor,
    recv_counter: torch.Tensor,
    *,
    batch_size: int,
    is_draft: bool,
    write_requests: bool,
    no_padding: bool = False,
    num_valid_rows: torch.Tensor | None = None,
) -> None:
    """Write normalized top-8 weights, expert IDs, padding, and a zero counter.

    Supports gfx1250 BF16 logits [N, 256] with contiguous expert columns.
    Outputs are contiguous FP32 weights [N, 8] and int32 IDs [N, 8].
    cu_seqlens holds at least batch_size + 1 int32 token offsets.
    real_requests is an int64 scalar; pad_rows is bool [N]; recv_counter is
    an int32 scalar. All tensors must be on the same GPU.

    Non-draft calls use cu_seqlens[batch_size] as the live token count and
    can update real_requests. Draft calls read the previously stored count
    and require N divisible by batch_size. They must not write_requests.
    no_padding marks all rows live, including eager drafts with padded batches.
    num_valid_rows can supply a device int32/int64 live-row scalar instead.
    In that mode, cu_seqlens and real_requests can be None; write_requests
    must be false. The same applies to no_padding with no request update.
    Invalid rows get expert ID -1. The function writes outputs in place.
    """
    n = logits.shape[0]
    assert logits.ndim == 2 and logits.shape[1] == 256
    assert logits.dtype == torch.bfloat16 and logits.stride(1) == 1
    assert n > 0 and batch_size > 0
    if num_valid_rows is not None:
        assert not write_requests and not no_padding
        assert num_valid_rows.numel() == 1
        assert num_valid_rows.dtype in (torch.int32, torch.int64)
    if write_requests or (num_valid_rows is None and not no_padding):
        assert cu_seqlens is not None and batch_size < cu_seqlens.numel()
        assert real_requests is not None
    assert not is_draft or (n % batch_size == 0 and not write_requests)
    assert weights.shape == ids.shape == (n, 8)
    assert weights.dtype == torch.float32 and ids.dtype == torch.int32
    assert weights.is_contiguous() and ids.is_contiguous()
    config = load_config_json(
        os.path.join(resolve_config_dir("moe", "FUSED-TOPK-DISPATCH"), "DEFAULT.json")
    )
    _fused_topk_dispatch_kernel[(n,)](
        logits,
        weights,
        ids,
        cu_seqlens,
        real_requests,
        pad_rows,
        recv_counter,
        num_valid_rows,
        STRIDE=logits.stride(0),
        BS=batch_size,
        Q=n // batch_size,
        DRAFT=is_draft,
        WRITE_REQUESTS=write_requests,
        BLOCK_BS=triton.next_power_of_2(batch_size),
        NO_PADDING=no_padding,
        HAS_VALID_ROWS=num_valid_rows is not None,
        **config,
    )

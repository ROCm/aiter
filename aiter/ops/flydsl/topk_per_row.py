# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Variable-length FlyDSL decode TopK interface."""

import torch

from .kernels.topk_per_row_decode import topk_per_row_decode_impl


def flydsl_top_k_per_row_decode(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    num_rows: int,
    stride0: int,
    stride1: int,
    k: int = 2048,
    stable: bool = False,
) -> None:
    """Write per-row TopK indices using each request's effective context length.

    For output row ``r``, the valid input length is
    ``seq_lens[r // next_n] - next_n + r % next_n + 1``.
    When ``stable`` is true, indices are emitted in ascending order and ties at
    the selection threshold prefer the smallest input indices.
    """
    if logits.ndim != 2 or logits.dtype != torch.float32 or not logits.is_cuda:
        raise ValueError("logits must be a 2D CUDA float32 tensor")
    if k <= 0 or k > logits.shape[1]:
        raise ValueError("k must be in the range [1, logits.shape[1]]")
    if logits.stride(1) != 1:
        raise ValueError("logits must have inner stride 1")
    if torch.cuda.get_device_properties(logits.device).warp_size != 64:
        raise ValueError("the FlyDSL decode TopK kernel requires a wave64 GPU")
    if seq_lens.ndim != 1 or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be a 1D int32 tensor")
    if not seq_lens.is_cuda or seq_lens.device != logits.device:
        raise ValueError("seq_lens must be on the same CUDA device as logits")
    if next_n <= 0:
        raise ValueError("next_n must be positive")
    rows = logits.shape[0]
    required_seq_lens = (rows + next_n - 1) // next_n
    if seq_lens.numel() < required_seq_lens:
        raise ValueError("seq_lens does not have enough entries for logits rows")
    if indices.shape != (rows, k) or indices.dtype != torch.int32:
        raise ValueError("indices must be an int32 tensor with shape [rows, k]")
    if indices.device != logits.device:
        raise ValueError("indices must be on the same CUDA device as logits")

    topk_per_row_decode_impl(
        logits,
        seq_lens,
        indices,
        k=k,
        next_n=next_n,
        stable=stable,
    )

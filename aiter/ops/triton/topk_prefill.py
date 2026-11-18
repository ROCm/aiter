# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.topk.topk_prefill import (
    _topk_prefill_kernel,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


def _get_config() -> dict:
    cfg_dir = resolve_config_dir("attention", "TOPK_PREFILL", backend="gluon")
    return load_config_json(f"{cfg_dir}/DEFAULT.json")


def top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    k: int = 2048,
) -> torch.Tensor:
    """Per-row top-k of ``logits[r, row_starts[r]:row_ends[r]]`` into ``indices[r, :k]``.

    int32 indices relative to ``row_starts``, unordered, -1 padded; NaN ranks lowest.
    """
    cfg = _get_config()
    cap = cfg["CAP"]
    assert logits.dtype == torch.float32 and logits.dim() == 2
    assert indices.dtype == torch.int32 and indices.stride(1) == 1
    assert k & (k - 1) == 0 and k <= cap, f"k must be a power of two <= {cap}"
    rows, width = logits.shape
    assert width < 1 << 18, "rows longer than 2**18 - 1 entries are not supported"
    if rows == 0:
        return indices
    vpt = cfg["VPT"]
    lanes = triton.next_power_of_2(max(1, triton.cdiv(width, vpt)))
    num_warps = min(cfg["MAX_WARPS"], max(cfg["MIN_WARPS"], lanes // 32))
    block = 32 * num_warps * vpt
    _topk_prefill_kernel[(rows,)](
        logits,
        row_starts,
        row_ends,
        indices,
        logits.stride(0),
        logits.stride(1),
        indices.stride(0),
        K=k,
        VPT=vpt,
        NUM_CHUNKS=triton.cdiv(width, block),
        CAP=cap,
        BUF=triton.next_power_of_2(2 * k + cap),
        SAMPLE_MIN=cfg["SAMPLE_MIN_TILES"] * block,
        CONTIG=logits.stride(1) == 1,
        SAMPLE_BLOCKS=cfg["SAMPLE_BLOCKS"],
        HI_NUM=cfg["HI_NUM"],
        LO_NUM=cfg["LO_NUM"],
        num_warps=num_warps,
    )
    return indices

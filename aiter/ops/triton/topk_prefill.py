# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import functools

import torch
import triton

from aiter.ops.triton._gluon_kernels.gfx1250.topk.topk_prefill import (
    _topk_classify_kernel,
    _topk_prefill_kernel,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


@functools.cache
def _config() -> dict:
    cfg_dir = resolve_config_dir("attention", "TOPK_PREFILL", backend="gluon")
    return load_config_json(f"{cfg_dir}/DEFAULT.json")


@functools.cache
def _classify_kwargs(k: int, width: int) -> dict:
    cfg = _config()["CLASSIFY"]
    cap = cfg["CAP"]
    assert k <= cap
    vpt = cfg["VPT"]
    lanes = triton.next_power_of_2(max(1, triton.cdiv(width, vpt)))
    num_warps = min(cfg["MAX_WARPS"], max(cfg["MIN_WARPS"], lanes // 32))
    block = 32 * num_warps * vpt
    return {
        "K": k,
        "VPT": vpt,
        "NUM_CHUNKS": triton.cdiv(width, block),
        "CAP": cap,
        "BUF": triton.next_power_of_2(2 * k + cap),
        "SAMPLE_MIN": cfg["SAMPLE_MIN_TILES"] * block,
        "SAMPLE_BLOCKS": cfg["SAMPLE_BLOCKS"],
        "HI_NUM": cfg["HI_NUM"],
        "LO_NUM": cfg["LO_NUM"],
        "num_warps": num_warps,
    }


@functools.cache
def _prefill_kwargs(k: int, width: int, route_few: bool, zero_agg: bool) -> dict:
    cfg = _config()
    ccfg = cfg["CLASSIFY"]
    c = _classify_kwargs(k, width)
    return {
        "K": k,
        "NBUF": cfg["NBUF"],
        "CAPW": cfg["CAPW"],
        "TGT_NUM": cfg["TGT_NUM"],
        "RMIN": cfg["RMIN"],
        "SLACK": cfg["SLACK"],
        "num_warps": cfg["NUM_WARPS"],
        "FEW_MAX": (
            ccfg["FEW_MAX_LEN"] if route_few and width <= ccfg["FEW_MAX_WIDTH"] else 0
        ),
        "RETRY_MAX": ccfg["RETRY_MAX_LEN"],
        "RING_ROW": width <= (cfg["NBUF"] - 1) * 32 * cfg["NUM_WARPS"] * c["VPT"],
        "CL_CAP": c["CAP"],
        "CL_SAMPLE_MIN": c["SAMPLE_MIN"],
        "CL_SAMPLE_BLOCKS": c["SAMPLE_BLOCKS"],
        "CL_HI": c["HI_NUM"],
        "CL_LO": c["LO_NUM"],
        "ZERO_AGG": zero_agg,
        "waves_per_eu": 8,
    }


def top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
    indices: torch.Tensor,
    k: int = 2048,
    short_rows_classify: bool = True,
    route_few: bool = True,
    zero_agg: bool = False,
) -> torch.Tensor:
    """Top-k indices of each row ``logits[r, row_starts[r]:row_ends[r]]``, unordered; NaN lowest, -0 == +0.

    logits: float32 [rows, width].
    row_starts: int32 [rows], first entry of each row.
    row_ends: int32 [rows], one past the last entry of each row.
    indices: int32 [rows, >= k] output, relative to row_starts, -1 padded.
    k: a power of two up to the config cap.
    short_rows_classify: launch widths <= CLASSIFY.MAX_WIDTH take the classify kernel.
    route_few: rows with few distinct values take the classify algorithm.
    zero_agg: classify histograms add a lane's exact zeros with one atomic (same results).
    """
    cfg = _config()
    cap = cfg["NUM_WARPS"] * (cfg["CAPW"] - 32) - cfg["SLACK"]
    assert k & (k - 1) == 0 and k <= cap, f"k must be a power of two <= {cap}"
    assert logits.dtype == torch.float32 and logits.dim() == 2
    assert indices.dtype == torch.int32 and indices.stride(1) == 1
    rows, width = logits.shape
    assert width < 1 << 24, "rows longer than 2**24 - 1 entries are not supported"
    if rows == 0:
        return indices
    if logits.stride(1) != 1:  # both kernels read unit-stride rows
        logits = logits.contiguous()
    # short rows: the classify kernel; launches of width <= k (copies only) use its width-k parameters
    if short_rows_classify and width <= cfg["CLASSIFY"]["MAX_WIDTH"]:
        _topk_classify_kernel[(rows,)](
            logits,
            row_starts,
            row_ends,
            indices,
            logits.stride(0),
            indices.stride(0),
            **_classify_kwargs(k, max(width, k)),
            ZERO_AGG=zero_agg,
        )
        return indices
    _topk_prefill_kernel[(rows,)](
        logits,
        row_starts,
        row_ends,
        indices,
        logits.stride(0),
        indices.stride(0),
        **_prefill_kwargs(k, width, route_few, zero_agg),
    )
    return indices

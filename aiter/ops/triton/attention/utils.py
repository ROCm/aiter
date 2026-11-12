# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.block_lut import (
    _block_attn_mask_to_lut_kernel,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import (
    AITER_TRITON_CONFIGS_PATH,
    load_config_json,
    resolve_config_dir,
)

_BLOCK_LUT_CONFIG_NAME = "BLOCK_LUT"
# Arches with no table of their own read gfx942's, as fav3_sage does for its tables.
_BLOCK_LUT_FALLBACK_ARCH = "gfx942"


def _load_block_kb_bounds() -> dict[str, tuple[int, int]]:
    """Every shipped BLOCK_LUT (BLOCK_KB_MIN, BLOCK_KB_MAX), keyed by architecture.

    Read once, at import, for the reason fav3_sage reads its tables then: mha_v4 builds the LUT
    inside torch.compile'd code, and Dynamo cannot trace the open() in load_config_json().
    """
    bounds = {}
    for entry in sorted(os.scandir(AITER_TRITON_CONFIGS_PATH), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        cfg_dir = resolve_config_dir(
            "attention", _BLOCK_LUT_CONFIG_NAME, backend="triton", arch=entry.name
        )
        table = load_config_json(f"{cfg_dir}/DEFAULT.json", required=False)
        if table is None:
            continue
        low, high = table["BLOCK_KB_MIN"], table["BLOCK_KB_MAX"]
        # tl.arange needs a power-of-two width, and clamping a power of two keeps one only when
        # both bounds are powers of two.
        assert all(b > 0 and b & (b - 1) == 0 for b in (low, high)) and low <= high, (
            f"{cfg_dir}/DEFAULT.json: BLOCK_KB_MIN={low} and BLOCK_KB_MAX={high} must be "
            "powers of two with BLOCK_KB_MIN <= BLOCK_KB_MAX"
        )
        bounds[entry.name] = (low, high)
    assert _BLOCK_LUT_FALLBACK_ARCH in bounds, (
        f"no {_BLOCK_LUT_CONFIG_NAME} table for the fallback architecture "
        f"{_BLOCK_LUT_FALLBACK_ARCH!r}: expected "
        f"{resolve_config_dir('attention', _BLOCK_LUT_CONFIG_NAME, backend='triton', arch=_BLOCK_LUT_FALLBACK_ARCH)}"
        "/DEFAULT.json"
    )
    return bounds


_BLOCK_KB_BOUNDS = _load_block_kb_bounds()


def block_attn_mask_to_ragged_lut(
    block_attn_mask: torch.Tensor,
    num_heads: int | None = None,
    return_none_if_dense: bool = False,
    BLOCK_KB: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """
    Convert a dense block attention mask to a ragged look-up table of KV block
    indices per (batch, head, q_block). Used for block-sparse attention with no
    per-iteration branching in the kernel.

    block_attn_mask: Either (batch, num_q_blocks, num_kv_blocks) boolean for
        same mask for all heads, or (batch, num_heads, num_q_blocks, num_kv_blocks)
        for per-head masks. True = may attend, False = must not attend.
    num_heads: Required when block_attn_mask is 3D (number of Q heads). Ignored when 4D.
    return_none_if_dense: If True and the mask is all True (dense), return None so the
        caller can pass block_lut=None to fav3_sage_wrapper_func and use the dense path.
        Avoids building a very large LUT that can trigger munmap_chunk on MI300X/ROCm.
    BLOCK_KB: Mask columns the LUT kernel scans per step. Defaults to the row length
        rounded up to a power of two and clamped to the arch's BLOCK_KB_MIN and
        BLOCK_KB_MAX from configs/<arch>/triton/attention/block_lut/DEFAULT.json:
        each step is a serial cumsum, so a long row wants few steps, while a width
        past the row wastes lanes. An explicit BLOCK_KB is used as given.
    Returns:
        kv_block_indices: 1D int32, concatenation of all KV block index lists.
        lut_start: 1D int32, length batch * num_heads * num_q_blocks. Index
            idx = batch_idx * (num_heads * num_q_blocks) + head_idx * num_q_blocks + q_block_idx.
        lut_count: 1D int32, same length as lut_start.
        When return_none_if_dense is True and the mask is all True, returns None instead.
    """
    device = block_attn_mask.device

    # 3D -> 4D: expand and fall through to 4D path
    if block_attn_mask.dim() == 3:
        if num_heads is None:
            raise ValueError("num_heads must be provided when block_attn_mask is 3D")
        batch, num_q_blocks, num_kv_blocks = block_attn_mask.shape
        if return_none_if_dense and block_attn_mask.all():
            return None
        block_attn_mask = block_attn_mask.unsqueeze(1).expand(
            batch, num_heads, num_q_blocks, num_kv_blocks
        )

    # 4D: (batch, num_heads, num_q_blocks, num_kv_blocks) — GPU vectorized path
    batch, num_heads, num_q_blocks, num_kv_blocks = block_attn_mask.shape
    if return_none_if_dense and block_attn_mask.all():
        return None

    counts = block_attn_mask.sum(dim=-1, dtype=torch.int32)
    lut_count = counts.reshape(-1)
    lut_start = torch.cumsum(lut_count, dim=0, dtype=torch.int32) - lut_count

    # NOTE: Overallocating the LUT is a waste of memory, but the
    # alternative lut_count.sum(), will cause graph break with torch compile.
    # TODO(sparse refresh): entries past the used range stay uninitialized, and an empty row's
    # lut_start points at them. Kernels that speculatively read LUT[lut_start] before testing
    # lut_count must clamp the derived offsets, as the FP8 prologue does, or this must be zeroed.
    max_count = batch * num_heads * num_q_blocks * num_kv_blocks
    kv_block_indices = torch.empty(max_count, dtype=torch.int32, device=device)
    if BLOCK_KB is None:
        low, high = _BLOCK_KB_BOUNDS.get(
            arch_info.get_arch(), _BLOCK_KB_BOUNDS[_BLOCK_LUT_FALLBACK_ARCH]
        )
        BLOCK_KB = min(high, max(low, triton.next_power_of_2(num_kv_blocks)))
    _block_attn_mask_to_lut_kernel[(batch * num_heads * num_q_blocks,)](
        block_attn_mask,
        lut_start,
        lut_count,
        kv_block_indices,
        stride_mask_b=block_attn_mask.stride(0),
        stride_mask_h=block_attn_mask.stride(1),
        stride_mask_qb=block_attn_mask.stride(2),
        stride_mask_kb=block_attn_mask.stride(3),
        num_heads=num_heads,
        num_q_blocks=num_q_blocks,
        num_kv_blocks=num_kv_blocks,
        BLOCK_KB=BLOCK_KB,
    )

    return kv_block_indices, lut_start, lut_count

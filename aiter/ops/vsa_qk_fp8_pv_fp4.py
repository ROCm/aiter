# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""QK=FP8 / PV=FP4 mixed-precision VSA block-sparse attention (gfx950).

Public API:
  vsa_qk_fp8_pv_fp4_dropB           # high-level convenience wrapper
  vsa_qk_fp8_pv_fp4                 # raw C++ binding (advanced use)
  build_l2_aware_lim_vsa_qk_fp8_pv_fp4  # GPU helper for task ordering
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ..jit.core import compile_ops

__all__ = [
    "vsa_qk_fp8_pv_fp4_dropB",
    "vsa_qk_fp8_pv_fp4",
    "build_l2_aware_lim_vsa_qk_fp8_pv_fp4",
]


# --------------------------------------------------------------------------- #
# Raw binding to the .co launcher (loaded from
# /opt/aiter/hsa/gfx950/vsa/vsa_qk_fp8_pv_fp4.co at first call).
# --------------------------------------------------------------------------- #
@compile_ops("module_vsa_qk_fp8_pv_fp4")
def vsa_qk_fp8_pv_fp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vscale: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    vbs: torch.Tensor,
    lim: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    counters: torch.Tensor,
    B: int,
    T: int,
    num_q_blks: int,
    max_kv: int,
    n_dense: int,
) -> None: ...


# --------------------------------------------------------------------------- #
# L2-aware task ordering — Triton-fused composite key + single GPU radix sort.
#
# Sort tasks by (first_kv_block // FKV_BAND_SIZE   ASC,   -- HBM/L2 band
#                -q2k_num                          ASC)   -- longest-job-first
# packed into a single int32 composite key:
#   high (32-QN_BITS) bits = (first_kv_block >> FKV_LOG2)  -- HBM-row band
#   low  QN_BITS      bits = (2*max_kv - q2k_num) clipped  -- LJF within band
#
# The kernel partitions the schedule into a "dense" head and a "sparse" tail
# using the host-side `n_dense` count (threshold = max_kv * 7/8 by default);
# inside each partition tiles are L2-banded so adjacent waves share HBM rows.
# --------------------------------------------------------------------------- #
_LIM_BLOCK = 256


@triton.jit
def _vsa_qk_fp8_pv_fp4_composite_key_kernel(
    q2k_num_ptr,        # *int32   (n_tasks,)
    q2k_idx_ptr,        # *int32   (n_tasks, idx_stride)  first col = first KV blk
    is_dense_ptr,       # *int32   (n_tasks,)             1 if q2k_num >= threshold
    out_key_ptr,        # *int32   (n_tasks,)
    n_tasks: tl.int32,
    max_kv_x2: tl.int32,
    threshold: tl.int32,
    idx_stride0: tl.int32,
    BLOCK: tl.constexpr,
    FKV_LOG2: tl.constexpr,
    QN_BITS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_tasks
    qn = tl.load(q2k_num_ptr + offs, mask=mask, other=0)
    fkv = tl.load(q2k_idx_ptr + offs * idx_stride0, mask=mask, other=0)
    band = fkv >> FKV_LOG2
    qn_neg = (max_kv_x2 - qn) & ((1 << QN_BITS) - 1)
    key = (band << QN_BITS) | qn_neg
    tl.store(out_key_ptr + offs, key, mask=mask)
    tl.store(is_dense_ptr + offs,
             tl.where(qn >= threshold, 1, 0), mask=mask)


def build_l2_aware_lim_vsa_qk_fp8_pv_fp4(
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    max_kv: int,
    fkv_band_size: int = 512,
    dense_ratio: float = 7.0 / 8.0,
) -> Tuple[torch.Tensor, int]:
    """Return ``(lim, n_dense)`` — L2-cache-aware task ordering for the
    kernel's outer scheduler.

    Two-level sort packed into a single int32 composite key:
      1. ``first_kv_block // fkv_band_size`` ASC  — adjacent tasks share an HBM row
      2. ``-q2k_num``                        ASC  — longest job first within band

    A leading **dense partition** of length ``n_dense`` is concatenated before
    the sparse tail (``q2k_num >= max_kv * dense_ratio`` -> dense).  The kernel
    consumes that partition through its dedicated dense-tile path which keeps
    per-K loop body tight when every q-row touches every KV block.

    Cost: O(n_tasks) Triton key build + O(n) radix sort + 2 small splits.
    """
    n = q2k_num.numel()
    threshold = int(max_kv * dense_ratio)

    key_buf      = torch.empty(n, dtype=torch.int32, device=q2k_num.device)
    is_dense_buf = torch.empty(n, dtype=torch.int32, device=q2k_num.device)
    grid = ((n + _LIM_BLOCK - 1) // _LIM_BLOCK,)
    _vsa_qk_fp8_pv_fp4_composite_key_kernel[grid](
        q2k_num, q2k_idx, is_dense_buf, key_buf,
        n, 2 * max_kv, threshold, q2k_idx.stride(0),
        BLOCK=_LIM_BLOCK,
        FKV_LOG2=fkv_band_size.bit_length() - 1,
        QN_BITS=13,  # max_kv < 4096 fits in 13 bits; with negation max_kv < 8192
    )

    # Sort the full task list by the composite key.
    order = torch.argsort(key_buf).to(torch.int32)
    # Partition that ordered list into dense vs sparse, preserving the
    # composite-key ordering within each partition.
    is_dense_ord = is_dense_buf[order.long()]
    d_mask = is_dense_ord != 0
    d_order = order[d_mask]
    s_order = order[~d_mask]
    lim = torch.cat([d_order, s_order]).contiguous()
    n_dense = int(d_order.numel())
    return lim, n_dense


# --------------------------------------------------------------------------- #
# High-level wrapper — auto-allocates out / lse / counters and zeros the
# atomic dispatch counters before each launch.  Mirrors the style of the
# stand-alone /home/vsa_qk_fp8_pv_fp4_hip/vsa_hybrid.py:vsa_qk_fp8_pv_fp4().
# --------------------------------------------------------------------------- #
_HEAD_DIM = 128


def vsa_qk_fp8_pv_fp4_dropB(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    vbs: torch.Tensor,
    lim: torch.Tensor,
    n_dense: int,
    B: int,
    T: int,
    num_q_blks: int,
    max_kv: int,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    counters: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch the QK=FP8 / PV=FP4 mixed-precision VSA kernel.

    Tensor layout (flattened heads, BH = B * H):
      q, k:     (BH, T, 128)        float8_e4m3fn  (per-32-element E8M0 scale)
      v:        (BH, T, 64)         uint8           (FP4 nibbles, 2 per byte)
      qs, ks:   (BH, T, 4)          uint8           (E8M0 per 32-element group)
      vs:       (BH, T/128, 128, 4) uint8           (E8M0 per K-block, HBM-coalesced)
      q2k_idx:  (BH * num_q_blks, max_kv) int32
      q2k_num:  (BH * num_q_blks,)        int32
      vbs:      (num_q_blks,)              int32
      lim:      (BH * num_q_blks,)         int32
      out:      (BH, T, 128)               bfloat16
      lse:      (BH, T)                    float32

    Numerical contract (sparsity 0.0846, seed-independent, T = 50k..1M):
      - cosine similarity vs FP32 ref: 0.9826 .. 0.9833
      - cos(LSE)                     : 1.000000
      - max |diff|                   : 6e-3 (T=1M) .. 2.4e-2 (T=50k)
      - no NaN / Inf at any tested size
    """
    BH = q.shape[0]
    assert BH % B == 0, f"q.shape[0]={BH} must be divisible by B={B}"
    if out is None:
        out = torch.empty((BH, T, _HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    if lse is None:
        lse = torch.empty((BH, T), dtype=torch.float32, device=q.device)
    if counters is None:
        counters = torch.zeros(2, dtype=torch.int32, device=q.device)

    vsa_qk_fp8_pv_fp4(
        q, k, v,
        qs, ks, vs,
        q2k_idx, q2k_num, vbs,
        lim, out, lse, counters,
        B, T, num_q_blks, max_kv, n_dense,
    )
    return out, lse

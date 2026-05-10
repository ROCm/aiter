# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP4 VSA dual-warp-set block-sparse attention (gfx950).

Public API:
    fp4_vsa_dual_dropB        # high-level convenience wrapper
    vsa_dual_dropB            # raw C++ binding (advanced use)
    build_l2_aware_lim_vsa    # CPU helper for task ordering
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ..jit.core import compile_ops

__all__ = [
    "fp4_vsa_dual_dropB",
    "vsa_dual_dropB",
    "build_l2_aware_lim_vsa",
]


# --------------------------------------------------------------------------- #
# Raw binding to the .co launcher  (loaded from
# /home/aiter/hsa/gfx950/vsa/vsa_dual_setprio_dropB.co at first call).
# --------------------------------------------------------------------------- #
@compile_ops("module_vsa_dual")
def vsa_dual_dropB(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qscale: torch.Tensor,
    kscale: torch.Tensor,
    vmean: torch.Tensor,
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
# Sort tasks by  (first_kv_block // FKV_BAND_SIZE   ASC,           -- L2 band
#                 -q2k_num                          ASC equivalent  -- LJF)
# packed into a single int32 key: high 19 bits = band, low 13 bits =
# (2*max_kv - q2k_num).  Empirical: ~+1.4 % kernel speedup over plain
# argsort(-qn) on the dev dataset (kernel is power-limited on MI355X,
# so the full L2-hit gain doesn't translate to time saved).
#
# Performance history (10 676 tasks, dev dataset):
#     CPU np.lexsort + GPU<->CPU            : 17 969 us
#     GPU 2-key composite int64 + 2 sorts   :    313 us  (+ .item() sync)
#     GPU 1-key argsort(-qn) (no L2 awareness):    38 us  (kernel +1.4 %)
#     Triton fused composite + 1 sort       :     45 us  (this implementation)
#
# `n_dense` is left at 0 — the kernel routes every task through the
# sparse-path, which empirically matches or beats the partitioned path
# and avoids a host-side sync to read the dense count.
# --------------------------------------------------------------------------- #
_LIM_BLOCK = 256


@triton.jit
def _vsa_composite_key_kernel(
    q2k_num_ptr,           # *int32  (n_tasks,)
    q2k_idx_ptr,           # *int32  (n_tasks, idx_stride)   first column = first KV block
    out_key_ptr,           # *int32  (n_tasks,)
    n_tasks: tl.int32,
    max_kv_x2: tl.int32,   # = 2*max_kv, ensures (max_kv_x2 - qn) >= 0
    idx_stride0: tl.int32,
    BLOCK: tl.constexpr,
    FKV_LOG2: tl.constexpr,  # log2(band_size); band_size must be a power of 2
    QN_BITS: tl.constexpr,   # bits reserved for (2*max_kv - qn); 13 -> max_kv < 4096
):
    pid     = tl.program_id(0)
    offs    = pid * BLOCK + tl.arange(0, BLOCK)
    mask    = offs < n_tasks
    qn      = tl.load(q2k_num_ptr + offs, mask=mask, other=0)
    fkv     = tl.load(q2k_idx_ptr + offs * idx_stride0, mask=mask, other=0)
    band    = fkv >> FKV_LOG2
    key     = (band << QN_BITS) | ((max_kv_x2 - qn) & ((1 << QN_BITS) - 1))
    tl.store(out_key_ptr + offs, key, mask=mask)


def build_l2_aware_lim_vsa(
    q2k_idx: torch.Tensor,
    q2k_num: torch.Tensor,
    max_kv: int,
    fkv_band_size: int = 512,           # noqa: ARG001 — fixed at 512 in the kernel
) -> Tuple[torch.Tensor, int]:
    """Return ``(lim, n_dense)`` — L2-cache-aware task ordering for the
    kernel's outer scheduler.

    Two-level sort packed into a single int32 composite key:
      1. ``first_kv_block // 512`` ASC   — adjacent tasks share an HBM/L2 row
      2. ``-q2k_num``               ASC  — longest job first inside each band

    ``n_dense`` is always 0; the kernel routes every task through the
    sparse-path (empirically equal or better than the partitioned path
    on the dev dataset, and avoids a host-side sync).

    Cost: ~45 us for 10 k tasks on MI355X (Triton composite + radix sort).
    """
    n        = q2k_num.numel()
    key_buf  = torch.empty(n, dtype=torch.int32, device=q2k_num.device)
    grid     = ((n + _LIM_BLOCK - 1) // _LIM_BLOCK,)
    _vsa_composite_key_kernel[grid](
        q2k_num, q2k_idx, key_buf,
        n, 2 * max_kv, q2k_idx.stride(0),
        BLOCK=_LIM_BLOCK,
        FKV_LOG2=fkv_band_size.bit_length() - 1,
        QN_BITS=13,    # max_kv < 4096 fits in 13 bits with the negation
    )
    return torch.argsort(key_buf).to(torch.int32), 0


# --------------------------------------------------------------------------- #
# High-level wrapper — auto-allocates out / lse / counters and zeros the
# atomic dispatch counters before each launch.  Mirrors the style of the
# stand-alone /home/VSA_Release/vsa_dual.py:fp4_vsa_dual_dropB().
# --------------------------------------------------------------------------- #
_HEAD_DIM = 128


def fp4_vsa_dual_dropB(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qs: torch.Tensor,
    ks: torch.Tensor,
    vm: torch.Tensor,
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
    out: Optional[torch.Tensor]      = None,
    lse: Optional[torch.Tensor]      = None,
    counters: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch the canonical fast FP4 VSA kernel.

    Tensor layout (flattened heads):
        q, k:       (BH, T, 64)            uint8   (FP4 packed, 2-per-byte)
        v:          (BH, T, 128)           uint8   (FP4 packed)
        qs, ks:     (BH, T, 4)             fp8e8m0 (one MX scale per group)
        vm, vs:     (BH, T // 16)          bfloat16
        q2k_idx:    (BH * num_q_blks, K)   int32
        q2k_num:    (BH * num_q_blks,)     int32
        vbs:        (BH * num_q_blks, K)   int32
        lim:        (BH * num_q_blks,)     int32
        out:        (BH, T, 128)           bfloat16
        lse:        (BH, T)                float32

    Numerical contract vs reference ``vsa_dual_setprio_exp`` baseline:
      - cosine similarity:          1.000000
      - elements within 1 bf16 ULP: >= 99.97 %
      - max single-element |diff|:  0.03125  (= 1 bf16 ULP)
      - bit-exact bf16 elements:    ~99.77 %
    """
    BH = q.shape[0]
    assert BH % B == 0, f"q.shape[0]={BH} must be divisible by B={B}"
    if out is None:
        out = torch.empty((BH, T, _HEAD_DIM), dtype=torch.bfloat16, device=q.device)
    if lse is None:
        lse = torch.empty((BH, T), dtype=torch.float32, device=q.device)
    if counters is None:
        counters = torch.zeros(2, dtype=torch.int32, device=q.device)

    vsa_dual_dropB(
        q, k, v,
        qs, ks, vm, vs,
        q2k_idx, q2k_num, vbs,
        lim, out, lse, counters,
        B, T, num_q_blks, max_kv, n_dense,
    )
    return out, lse

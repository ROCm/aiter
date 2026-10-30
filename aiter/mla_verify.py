# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA target verify (speculative decoding) kernel selection and the gfx950
HipKittens-style verify kernel.

Callers pick a kernel on the host with ``select_mla_verify_kernel`` and, under
CUDA graphs, capture one graph per returned kernel:

- ``"gluon"``: ``aiter.ops.triton.gluon.mla_gluon.mla_gluon`` (short contexts).
- ``"hk"``: ``mla_verify_hk`` below (long contexts; QLEN 8, 12 heads,
  latent 576 / value 512, FP8 e4m3 KV with page size 1, gfx950 only).
"""

import torch
import triton

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.attention import hk_mla_verify_fwd, hk_mla_verify_reduce
from aiter.ops.triton._triton_kernels.attention.mla_verify_reduce import (
    _mla_verify_reduce_kernel,
)

MLA_VERIFY_GLUON = "gluon"
MLA_VERIFY_HK = "hk"
MLA_VERIFY_KERNELS = (MLA_VERIFY_GLUON, MLA_VERIFY_HK)

_HK_HEADS = 12
_HK_QLEN = 8
_HK_D = 576
_HK_DV = 512
_HK_BLOCK_N = 64
_HK_MIN_CHUNK = 256
_HK_TARGET_CTAS = 256
_HK_MAX_SPLITS = 256
# Splits from which the partial O is stored as fp8 (HIP merge only).
_HK_PART8_MIN_SPLITS = 96
_LOG2E = 1.4426950408889634

# Gluon -> HK crossover per GPU: HK when batch * max_kv_len >= threshold, keyed
# by the largest batch size each threshold covers. Measured with CUDA graphs on
# MI355X at QLEN 8, 12 heads, FP8 KV.
_HK_THRESHOLDS = {
    "gfx950": ({1: 27000, 2: 28000, 4: 27000, 8: 29000, 16: 30000}, 30000),
}


def mla_verify_hk_supported(
    num_heads: int, qlen: int, kv_dtype: torch.dtype, page_size: int = 1
) -> bool:
    """Whether mla_verify_hk can run this verify configuration on this GPU."""
    return (
        get_gfx() in _HK_THRESHOLDS
        and num_heads == _HK_HEADS
        and qlen == _HK_QLEN
        and kv_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        and page_size == 1
    )


def mla_verify_hk_threshold(batch_size: int) -> int | None:
    """batch * max_kv_len from which HK beats Gluon, or None without HK."""
    entry = _HK_THRESHOLDS.get(get_gfx())
    if entry is None:
        return None
    table, default = entry
    for bs in sorted(table):
        if batch_size <= bs:
            return table[bs]
    return default


def select_mla_verify_kernel(
    batch_size: int,
    max_kv_len: int,
    num_heads: int = _HK_HEADS,
    qlen: int = _HK_QLEN,
    kv_dtype: torch.dtype = torch.float8_e4m3fn,
    page_size: int = 1,
) -> str:
    """Pick the verify kernel for a batch on the host.

    max_kv_len is the longest request's KV length including the draft tokens,
    e.g. ``max(seq_lens_cpu) + num_draft_tokens``. Returns one of
    MLA_VERIFY_KERNELS.
    """
    if batch_size <= 0 or not mla_verify_hk_supported(
        num_heads, qlen, kv_dtype, page_size
    ):
        return MLA_VERIFY_GLUON
    threshold = mla_verify_hk_threshold(batch_size)
    if threshold is not None and batch_size * max_kv_len >= threshold:
        return MLA_VERIFY_HK
    return MLA_VERIFY_GLUON


def mla_verify_hk_num_splits(batch_size: int) -> int:
    """Power-of-two KV splits per request so batch * splits ~ one CTA per CU."""
    n = max(1, -(-_HK_TARGET_CTAS // max(batch_size, 1)))
    return max(2, min(triton.next_power_of_2(n), _HK_MAX_SPLITS))


def _use_hip_reduce(nsplit: int) -> bool:
    # The one-round-trip HIP merge wins at nsplit >= 128 (batch <= 2); the
    # Triton merge is faster for larger batches.
    return 128 <= nsplit <= _HK_MAX_SPLITS


def mla_verify_hk(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    sm_scale: float,
    kv_scale=None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """MLA target verify on gfx950 for QLEN 8, 12 heads, FP8 KV, page size 1.

    q: bf16 [batch * 8, 12, 576]; kv_buffer: fp8 [N, ..., 576]; kv_indptr:
    int32 [batch + 1] (KV lengths include the draft tokens); kv_indices: int32;
    kv_scale: fp32 [1] tensor, float or None (1.0). Returns bf16 [batch * 8, 12, 512] (written to out).
    Allocates no host-dependent state, so it can be captured in a CUDA graph.
    """
    ntok = q.shape[0]
    assert ntok % _HK_QLEN == 0, "token count must be a multiple of 8"
    bs = ntok // _HK_QLEN
    if out is None:
        out = torch.empty(
            (ntok, _HK_HEADS, _HK_DV), dtype=torch.bfloat16, device=q.device
        )
    q = q.view(-1, _HK_HEADS, _HK_D)
    kv = kv_buffer.view(-1, _HK_D)
    if not isinstance(kv_scale, torch.Tensor):
        kv_scale = torch.full(
            (1,),
            1.0 if kv_scale is None else float(kv_scale),
            dtype=torch.float32,
            device=q.device,
        )
    nsplit = mla_verify_hk_num_splits(bs)
    rows = _HK_HEADS * _HK_QLEN
    o_part = torch.empty(
        (bs, nsplit, rows, _HK_DV), dtype=torch.bfloat16, device=q.device
    )
    lse_part = torch.empty((bs, nsplit, rows), dtype=torch.float32, device=q.device)
    hip_reduce = _use_hip_reduce(nsplit)
    part8_min = _HK_PART8_MIN_SPLITS if hip_reduce else 0
    hk_mla_verify_fwd(
        q,
        kv,
        kv_indptr,
        kv_indices,
        kv_scale,
        o_part,
        lse_part,
        out,
        nsplit,
        _HK_MIN_CHUNK,
        sm_scale * _LOG2E,
        part8_min,
    )
    if bs == 0:
        return out
    if hip_reduce:
        hk_mla_verify_reduce(
            o_part, lse_part, kv_indptr, kv_scale, out, nsplit, _HK_MIN_CHUNK, part8_min
        )
        return out
    nsplit_p2 = triton.next_power_of_2(nsplit)
    _mla_verify_reduce_kernel[(bs, rows, _HK_DV // 128)](
        kv_indptr,
        o_part,
        lse_part,
        out,
        out.stride(0),
        out.stride(1),
        H=_HK_HEADS,
        QLEN=_HK_QLEN,
        D_V=_HK_DV,
        NSPLIT=nsplit,
        NSPLIT_P2=nsplit_p2,
        BLOCK_N=_HK_BLOCK_N,
        MIN_CHUNK=_HK_MIN_CHUNK,
        S_BLK=min(64, nsplit_p2),
        D_BLK=128,
        num_warps=4 if nsplit >= 64 else 2,
    )
    return out

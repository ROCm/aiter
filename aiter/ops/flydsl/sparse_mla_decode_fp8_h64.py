# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse MLA decode for per-tensor fp8 KV, 64 heads x 512 dims, on gfx950.

Public entry for ``kernels/sparse_mla_decode_fp8_h64.py``. It serves the
DeepSeek-V4-style decode call of ``sparse_mla_fwd`` (aiter Triton) with a
per-tensor e4m3 ("fp8_scalar") cache, fp8 dots and the inverse-RoPE MXFP8
output epilogue: 64 query heads of 512 dims whose RoPE (64 dims) sits inside
the row, a sliding-window main list plus a top-k extra list per query row, an
attention sink. Both entry points take ``sparse_mla_fwd``'s keyword
arguments with its defaults (plus ``max_keys_per_row``), so a caller checks
``can_use_flydsl_sparse_mla_decode_fp8_h64`` with the arguments it would pass
there and falls back to it otherwise; a value this kernel does not implement
(``dot_precision="bf16"``, ``return_lse=True``, an appended RoPE, ...) is
declined.

Both checks read host metadata only (shapes, dtypes, strides, pointers), so
they are safe under CUDA graph capture.
"""

from __future__ import annotations

import functools

import numpy as np
import torch

from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

from .kernels.sparse_mla_decode_fp8_h64 import (
    HEAD_GROUPS,
    LOG2E,
    MAXK,
    ROPE,
    D,
    H,
    compile_sparse_mla_decode_fp8_h64,
)
from .kernels.tensor_shim import _run_compiled

__all__ = [
    "can_use_flydsl_sparse_mla_decode_fp8_h64",
    "flydsl_sparse_mla_decode_fp8_h64",
    "warmup_flydsl_sparse_mla_decode_fp8_h64",
]

# main + extra keys of one query row: the kernel stages a row's slot ids in LDS
MAX_KEYS_PER_ROW = MAXK
# query rows: keeps every output offset a 32-bit buffer offset (bf16 rows: 1 GiB)
MAX_ROWS = 16384
_FP8 = (torch.float8_e4m3fn, torch.uint8)
_XCDS = 8


@functools.cache
def _launcher(device_index: int, hg: int, p_terms: int, mxfp8: bool, extra: bool):
    # keyed by device: a compiled handle loaded on one device does not launch
    # on another
    del device_index
    return compile_sparse_mla_decode_fp8_h64(hg, p_terms, mxfp8, extra)


@functools.cache
def _cus_per_xcd() -> int:
    return max(1, get_cu_num() // _XCDS)


def _plan_hg(rows: int) -> int:
    """Heads per workgroup: the fewest whose workgroups fit one wave of the CUs.

    Measured on MI355X (256 CUs): 16 up to 64 rows, 32 up to 128, then 64 (the
    64-head workgroup runs a row's tiles at the L2 -> LDS gather rate).
    """
    cus = _cus_per_xcd() * _XCDS
    if rows * (H // 16) <= cus:
        return 16
    if rows * (H // 32) <= cus:
        return 32
    return 64


def _plan_wgs(units_x: int) -> int:
    """Workgroups per XCD: the fewest that keep the makespan of ``units_x``
    units, which then run back to back in a workgroup."""
    rounds = -(-units_x // _cus_per_xcd())
    return -(-units_x // rounds)


def _cache_geometry(cache: torch.Tensor) -> tuple[int, int] | None:
    """(log2 block size, block stride in rows) of an e4m3 cache
    ``[blocks, block, 512]`` or ``[rows, 512]``, or None if the kernel cannot
    address it."""
    if cache.dtype not in _FP8 or cache.shape[-1] != D or cache.stride(-1) != 1:
        return None
    if cache.dim() == 2:
        return (0, 1) if cache.stride(0) == D else None
    if cache.dim() != 3:
        return None
    block = cache.shape[1]
    if block <= 0 or block & (block - 1) or cache.stride(1) != D:
        return None
    if cache.stride(0) % D:
        return None
    return block.bit_length() - 1, cache.stride(0) // D


def _aligned(t: torch.Tensor, align: int = 16) -> bool:
    return t.data_ptr() % align == 0


def _index_ok(t: torch.Tensor | None, device: torch.device) -> bool:
    return (
        t is not None
        and t.device == device
        and t.dtype == torch.int32
        and t.dim() == 1
        and t.is_contiguous()
    )


def _scale_ok(t: torch.Tensor | None, device: torch.device) -> bool:
    return (
        t is not None
        and t.device == device
        and t.dtype == torch.float32
        and t.numel() == 1
        and t.is_contiguous()
    )


def _unsupported_reason(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    kv_scale: torch.Tensor | None,
    q_scale: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_kv: torch.Tensor | None,
    extra_indptr: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_kv_scale: torch.Tensor | None,
    inv_rope_positions: torch.Tensor | None,
    inv_rope_cos_sin_cache: torch.Tensor | None,
    out_mxfp8: tuple[torch.Tensor, torch.Tensor] | None,
    out: torch.Tensor | None,
    has_invalid: bool,
    max_keys_per_row: int | None,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    dot_precision: str,
    return_lse: bool,
    skip_reduce: bool,
) -> str | None:
    if not q.is_cuda or torch.version.hip is None:
        return "needs a ROCm device tensor"
    if get_gfx_runtime() != "gfx950":
        return "gfx950 only"
    if kv_lora_rank != D or qk_rope_head_dim != 0:
        return f"rope inside the row only: kv_lora_rank={D}, qk_rope_head_dim=0"
    if dot_precision != "fp8":
        return 'fp8 dots only (dot_precision="fp8")'
    if return_lse or skip_reduce:
        return "no LSE or split-K partial outputs"
    dev = q.device
    if q.dim() != 3 or tuple(q.shape[1:]) != (H, D):
        return f"q must be [T, {H}, {D}], got {tuple(q.shape)}"
    if q.dtype not in _FP8 or not q.is_contiguous() or not _aligned(q):
        return "q must be a contiguous, 16-byte aligned float8_e4m3fn tensor"
    if q.shape[0] > MAX_ROWS:
        return f"at most {MAX_ROWS} query rows"
    if has_invalid:
        return "-1 sentinels in the index lists are not supported"
    # The kernel clamps list positions to MAXK - 1, so a longer row would be
    # wrong without an error; the lengths live on the device, so the caller
    # states its bound (window + top-k).
    if max_keys_per_row is None or max_keys_per_row > MAXK:
        return f"pass max_keys_per_row (main + extra keys of a row), at most {MAXK}"
    if (
        kv_buffer.device != dev
        or _cache_geometry(kv_buffer) is None
        or not _aligned(kv_buffer)
    ):
        return (
            "kv_buffer must be an aligned e4m3 [blocks, 2^k, 512] or [rows, 512] cache"
        )
    if not (_index_ok(kv_indptr, dev) and _index_ok(kv_indices, dev)):
        return "kv_indptr / kv_indices must be contiguous 1-D int32 on q's device"
    if kv_indptr.shape[0] < q.shape[0] + 1:
        return "kv_indptr needs T + 1 entries"
    if not (_scale_ok(kv_scale, dev) and _scale_ok(q_scale, dev)):
        return "q_scale and kv_scale must be [1] float32 on q's device"
    if (
        attn_sink is None
        or attn_sink.device != dev
        or attn_sink.dtype != torch.float32
        or attn_sink.numel() != H
        or not attn_sink.is_contiguous()
    ):
        return f"attn_sink must be a contiguous [{H}] float32 tensor"
    extra = (extra_kv, extra_indptr, extra_indices)
    if any(t is not None for t in extra):
        if any(t is None for t in extra):
            return "extra_kv, extra_indptr and extra_indices go together"
        if (
            extra_kv.device != dev
            or _cache_geometry(extra_kv) is None
            or not _aligned(extra_kv)
        ):
            return "extra_kv must be an aligned e4m3 [blocks, 2^k, 512] or [rows, 512] cache"
        if not (_index_ok(extra_indptr, dev) and _index_ok(extra_indices, dev)):
            return "extra_indptr / extra_indices must be contiguous 1-D int32"
        if extra_indptr.shape[0] < q.shape[0] + 1:
            return "extra_indptr needs T + 1 entries"
        if extra_kv_scale is not None and extra_kv_scale.data_ptr() != (
            kv_scale.data_ptr()
        ):
            return "both caches must share kv_scale"
    if (out is None) == (out_mxfp8 is None):
        return "exactly one of out and out_mxfp8"
    T = q.shape[0]
    if out_mxfp8 is not None:
        if inv_rope_positions is None or inv_rope_cos_sin_cache is None:
            return "the MXFP8 output applies the inverse RoPE: pass its positions"
        pos, cs = inv_rope_positions, inv_rope_cos_sin_cache
        if (
            pos.device != dev
            or pos.dtype not in (torch.int32, torch.int64)
            or pos.dim() != 1
            or pos.shape[0] < T
            or not pos.is_contiguous()
        ):
            return "inv_rope_positions must be contiguous [T] int32 / int64"
        # read as 16-byte vectors: 16-byte aligned rows
        if (
            cs.device != dev
            or cs.dtype != torch.float32
            or cs.dim() != 2
            or cs.shape[1] != ROPE
            or cs.stride(1) != 1
            or cs.stride(0) % 4
            or not _aligned(cs)
        ):
            return (
                f"inv_rope_cos_sin_cache must be [P, {ROPE}] float32 (cos | sin) "
                "with 16-byte aligned rows"
            )
        data, scale = out_mxfp8
        if (
            data.device != dev
            or data.dtype not in _FP8
            or tuple(data.shape) != (T, H * D)
            or not data.is_contiguous()
            or not _aligned(data)
        ):
            return f"out_mxfp8[0] must be contiguous, aligned [T, {H * D}] e4m3"
        if (
            scale.device != dev
            or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu)
            or tuple(scale.shape) != (T, H * D // 32)
            or not scale.is_contiguous()
        ):
            return f"out_mxfp8[1] must be contiguous [T, {H * D // 32}] uint8 E8M0"
    else:
        if inv_rope_positions is not None or inv_rope_cos_sin_cache is not None:
            return "the inverse RoPE runs only with the MXFP8 output"
        if (
            out.device != dev
            or out.dtype != torch.bfloat16
            or tuple(out.shape) != (T, H, D)
            or not out.is_contiguous()
            or not _aligned(out)
        ):
            return f"out must be contiguous, aligned [T, {H}, {D}] bf16"
    return None


def can_use_flydsl_sparse_mla_decode_fp8_h64(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    *,
    kv_scale: torch.Tensor | None = None,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    kv_splits: int | None = None,
    skip_reduce: bool = False,
    has_invalid: bool = False,
    dot_precision: str = "bf16",
    q_scale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    attn_sink: torch.Tensor | None = None,
    extra_kv: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_kv_scale: torch.Tensor | None = None,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_cos_sin_cache: torch.Tensor | None = None,
    out_mxfp8: tuple[torch.Tensor, torch.Tensor] | None = None,
    max_keys_per_row: int | None = None,
) -> bool:
    """Whether ``flydsl_sparse_mla_decode_fp8_h64`` serves this call.

    Takes ``sparse_mla_fwd``'s keyword arguments, with its defaults, and
    ``max_keys_per_row``: the caller's bound on a row's main + extra keys
    (window + top-k). The lengths live on the device and the kernel clamps a
    row to ``MAX_KEYS_PER_ROW`` (1024) keys, so a call without the bound is
    declined. ``kv_splits`` (a split-K hint) is ignored.
    """
    del kv_splits
    return (
        _unsupported_reason(
            q,
            kv_buffer,
            kv_indptr,
            kv_indices,
            kv_scale=kv_scale,
            q_scale=q_scale,
            attn_sink=attn_sink,
            extra_kv=extra_kv,
            extra_indptr=extra_indptr,
            extra_indices=extra_indices,
            extra_kv_scale=extra_kv_scale,
            inv_rope_positions=inv_rope_positions,
            inv_rope_cos_sin_cache=inv_rope_cos_sin_cache,
            out_mxfp8=out_mxfp8,
            out=out,
            has_invalid=has_invalid,
            max_keys_per_row=max_keys_per_row,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            dot_precision=dot_precision,
            return_lse=return_lse,
            skip_reduce=skip_reduce,
        )
        is None
    )


def flydsl_sparse_mla_decode_fp8_h64(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    softmax_scale: float,
    *,
    kv_scale: torch.Tensor,
    kv_lora_rank: int = 512,
    qk_rope_head_dim: int = 64,
    kv_splits: int | None = None,
    skip_reduce: bool = False,
    has_invalid: bool = False,
    dot_precision: str = "bf16",
    q_scale: torch.Tensor,
    out: torch.Tensor | None = None,
    return_lse: bool = False,
    attn_sink: torch.Tensor,
    extra_kv: torch.Tensor | None = None,
    extra_indptr: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_kv_scale: torch.Tensor | None = None,
    inv_rope_positions: torch.Tensor | None = None,
    inv_rope_cos_sin_cache: torch.Tensor | None = None,
    out_mxfp8: tuple[torch.Tensor, torch.Tensor] | None = None,
    max_keys_per_row: int | None = None,
    p_terms: int = 2,
    heads_per_workgroup: int | None = None,
    workgroups_per_xcd: int | None = None,
) -> None:
    """Sparse MQA decode attention with the inverse-RoPE MXFP8 output.

    Every query row ``t`` attends the keys
    ``kv_indices[kv_indptr[t]:kv_indptr[t + 1]]`` of ``kv_buffer`` followed by
    ``extra_indices[extra_indptr[t]:extra_indptr[t + 1]]`` of ``extra_kv``
    (slot ``s`` of a ``[blocks, block, 512]`` cache is row ``s % block`` of
    block ``s // block``), at most ``max_keys_per_row`` <= 1024 keys in all,
    plus the sink ``exp(attn_sink[h])`` in the softmax denominator. A key row is
    K and V (448 nope + 64 RoPE dims, e4m3, one per-tensor ``kv_scale`` for both
    caches); scores are ``q k * q_scale * kv_scale * softmax_scale``. A row
    with no key gives zeros.

    The keyword arguments are ``sparse_mla_fwd``'s with its defaults; this
    kernel implements ``kv_lora_rank=512, qk_rope_head_dim=0,
    dot_precision="fp8"`` with ``return_lse=False``, ``skip_reduce=False`` and
    ``has_invalid=False`` (``kv_splits`` is ignored).

    Output, one of:
        out_mxfp8: (codes ``[T, 64 * 512]`` e4m3, scales ``[T, 64 * 512 // 32]``
            uint8 E8M0), row-major, one scale per 32 dims. The bf16-rounded
            rows get the inverse GPT-J RoPE on dims 448..511 at
            ``inv_rope_positions`` (``inv_rope_cos_sin_cache``: ``[P, 64]`` f32,
            32 cos then 32 sin) before the quantization; needs both.
        out: ``[T, 64, 512]`` bf16 rows, no RoPE.

    Args:
        max_keys_per_row: the caller's bound on a row's main + extra keys
            (window + top-k, e.g. 128 + 512); required, at most 1024.
        p_terms: 2 runs P x V with P as two e4m3 terms (hi + residual, cosine
            to an fp64 reference > 0.99999), 1 with one e4m3 P (~0.9997, like
            the Triton kernel's fp8 dots) and 5-15% faster.
        heads_per_workgroup, workgroups_per_xcd: dispatch overrides (16, 32 or
            64 heads per workgroup; the persistent 64-head kernel's workgroups
            per XCD); None picks them from the row count.

    The first call of each kernel variant compiles it; call
    ``warmup_flydsl_sparse_mla_decode_fp8_h64`` before CUDA graph capture.

    Raises:
        ValueError: if ``can_use_flydsl_sparse_mla_decode_fp8_h64`` would
            return False for these arguments.
    """
    del kv_splits
    reason = _unsupported_reason(
        q,
        kv_buffer,
        kv_indptr,
        kv_indices,
        kv_scale=kv_scale,
        q_scale=q_scale,
        attn_sink=attn_sink,
        extra_kv=extra_kv,
        extra_indptr=extra_indptr,
        extra_indices=extra_indices,
        extra_kv_scale=extra_kv_scale,
        inv_rope_positions=inv_rope_positions,
        inv_rope_cos_sin_cache=inv_rope_cos_sin_cache,
        out_mxfp8=out_mxfp8,
        out=out,
        has_invalid=has_invalid,
        max_keys_per_row=max_keys_per_row,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        dot_precision=dot_precision,
        return_lse=return_lse,
        skip_reduce=skip_reduce,
    )
    if reason is not None:
        raise ValueError(f"flydsl_sparse_mla_decode_fp8_h64: {reason}")
    if p_terms not in (1, 2):
        raise ValueError(f"p_terms must be 1 or 2, got {p_terms}")
    rows = q.shape[0]
    if rows == 0:
        return
    hg = _plan_hg(rows) if heads_per_workgroup is None else heads_per_workgroup
    if hg not in HEAD_GROUPS:
        raise ValueError(f"heads_per_workgroup must be one of {HEAD_GROUPS}")
    units_x = -(-rows * (H // hg) // _XCDS)
    wgs_x = (
        _plan_wgs(units_x)
        if workgroups_per_xcd is None
        else max(1, min(workgroups_per_xcd, units_x))
    )

    mxfp8 = out_mxfp8 is not None
    if mxfp8:
        out_ptr, osc_ptr = out_mxfp8[0].data_ptr(), out_mxfp8[1].data_ptr()
        pos_ptr = inv_rope_positions.data_ptr()
        pos_bytes = inv_rope_positions.element_size()
        cs_ptr = inv_rope_cos_sin_cache.data_ptr()
        cs_stride = inv_rope_cos_sin_cache.stride(0)
    else:
        out_ptr, osc_ptr = out.data_ptr(), 0
        pos_ptr = cs_ptr = cs_stride = 0
        pos_bytes = 4
    has_extra = extra_kv is not None
    m_lg, m_st = _cache_geometry(kv_buffer)
    if has_extra:
        x_lg, x_st = _cache_geometry(extra_kv)
        xc, xip, xid = (
            extra_kv.data_ptr(),
            extra_indptr.data_ptr(),
            extra_indices.data_ptr(),
        )
    else:
        x_lg = x_st = xc = xip = xid = 0
    # the kernel scores in log2 units
    sm_log2 = float(np.float32(softmax_scale) * np.float32(LOG2E))

    exe = _launcher(q.device.index or 0, hg, p_terms, mxfp8, has_extra)
    # FlyDSL loads and launches on the current HIP device, not the stream's.
    with torch.cuda.device(q.device):
        _run_compiled(
            exe,
            q.data_ptr(),
            kv_buffer.data_ptr(),
            xc,
            kv_indptr.data_ptr(),
            kv_indices.data_ptr(),
            xip,
            xid,
            attn_sink.data_ptr(),
            pos_ptr,
            cs_ptr,
            out_ptr,
            osc_ptr,
            q_scale.data_ptr(),
            kv_scale.data_ptr(),
            rows,
            units_x,
            wgs_x,
            m_lg,
            m_st,
            x_lg,
            x_st,
            cs_stride,
            pos_bytes,
            sm_log2,
            _XCDS * wgs_x,
            torch.cuda.current_stream(q.device),
        )


def warmup_flydsl_sparse_mla_decode_fp8_h64(
    device: torch.device | str | int | None = None,
    *,
    p_terms: int = 2,
    mxfp8: bool = True,
    has_extra: bool = True,
) -> None:
    """Compile the kernels the dispatch can pick for this configuration (one
    per head group) by running each once on a one-row input, so that no JIT
    compile happens during CUDA graph capture. ``mxfp8`` selects the output
    (MXFP8 with the inverse RoPE, or bf16 rows) and ``has_extra`` whether the
    calls pass the extra (top-k) list."""
    if device is None:
        device = torch.cuda.current_device()
    dev = (
        torch.device("cuda", device)
        if isinstance(device, int)
        else torch.device(device)
    )
    f32 = {"dtype": torch.float32, "device": dev}
    i32 = {"dtype": torch.int32, "device": dev}
    cache = torch.zeros((2, D), dtype=torch.float8_e4m3fn, device=dev)
    lists = {
        "kv_indptr": torch.tensor([0, 1], **i32),
        "kv_indices": torch.zeros(1, **i32),
    }
    if has_extra:
        lists["extra_kv"] = cache
        lists["extra_indptr"] = torch.tensor([0, 1], **i32)
        lists["extra_indices"] = torch.ones(1, **i32)
    if mxfp8:
        output = {
            "out_mxfp8": (
                torch.empty((1, H * D), dtype=torch.float8_e4m3fn, device=dev),
                torch.empty((1, H * D // 32), dtype=torch.uint8, device=dev),
            ),
            "inv_rope_positions": torch.zeros(1, **i32),
            "inv_rope_cos_sin_cache": torch.zeros((1, ROPE), **f32),
        }
    else:
        output = {"out": torch.empty((1, H, D), dtype=torch.bfloat16, device=dev)}
    for hg in HEAD_GROUPS:
        flydsl_sparse_mla_decode_fp8_h64(
            torch.zeros((1, H, D), dtype=torch.float8_e4m3fn, device=dev),
            cache,
            softmax_scale=D**-0.5,
            kv_scale=torch.ones(1, **f32),
            q_scale=torch.ones(1, **f32),
            attn_sink=torch.zeros(H, **f32),
            kv_lora_rank=D,
            qk_rope_head_dim=0,
            dot_precision="fp8",
            max_keys_per_row=2,
            p_terms=p_terms,
            heads_per_workgroup=hg,
            **lists,
            **output,
        )

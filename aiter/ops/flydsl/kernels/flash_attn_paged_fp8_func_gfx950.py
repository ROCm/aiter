# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Native gfx950 paged FP8 prefill with packed queries and BF16 output."""

from contextlib import nullcontext
from functools import lru_cache
from numbers import Integral, Real

import torch

from aiter.ops.flydsl.kernels.flash_attn_func_fp8_gfx950 import (
    _gpu_arch,
    _is_valid_softmax_scale,
)
from aiter.ops.flydsl.kernels.fmha_gfx950.paged_pipeline import (
    PAGED_FP8_BUFFER_LIMIT_BYTES,
)

_HEAD_DIMS = ((128, 128), (192, 128), (192, 192))
_MAX_FLAT_ELEMS = 1 << 31


def _cache_geometry(q, k, v):
    if k.ndim != v.ndim:
        raise ValueError("paged FP8 K/V must have matching ranks")
    if k.ndim == 3:
        pages, heads, dim = k.shape
        value_dim = v.shape[2]
        if v.shape[:2] != (pages, heads):
            raise ValueError("paged FP8 linear3d K/V page and head counts must match")
        page_size, layout = 1, "linear3d"
    elif k.ndim == 4:
        pages, page_size, heads, dim = k.shape
        value_dim = v.shape[3]
        if v.shape[:3] != (pages, page_size, heads):
            raise ValueError("paged FP8 linear K/V page and head counts must match")
        if page_size != 1:
            raise NotImplementedError("paged FP8 linear caches require page size 1")
        layout = "linear"
    elif k.ndim == 5:
        pages, heads, groups, page_size, pack = k.shape
        dim, value_dim = groups * pack, v.shape[3]
        if pack != 16 or page_size not in (16, 64, 1024):
            raise NotImplementedError(
                "paged FP8 vectorized caches require page 16/64/1024 and pack 16"
            )
        if v.shape != (pages, heads, page_size // 16, value_dim, 16):
            raise ValueError("paged FP8 vectorized V layout does not match K")
        layout = "vectorized"
    else:
        raise ValueError("paged FP8 K/V must be rank 3, 4 or 5")
    if dim != q.shape[-1]:
        raise ValueError("paged FP8 K width must match Q")
    if (dim, value_dim) not in _HEAD_DIMS:
        raise NotImplementedError(
            "paged FP8 supports D128/V128, D192/V128 and D192/V192"
        )
    if heads <= 0 or q.shape[1] <= 0 or q.shape[1] % heads:
        raise ValueError("paged FP8 requires positive heads with Hq divisible by Hkv")
    return pages, page_size, heads, dim, value_dim, layout


def _batch_interleave_group(batch, dims, paired):
    if paired:
        if dims == (128, 128) and batch in (2, 3, 5):
            return batch
        return 2 if dims == (192, 128) and batch == 2 else 1
    if dims[0] != 192 or batch <= 1 or (dims == (192, 192) and batch > 16):
        return 1
    for group in (8, 4, 2):
        if batch % group == 0:
            return group
    return 1


@lru_cache(maxsize=64)
def _build(
    num_heads,
    num_kv_heads,
    head_dim,
    value_dim,
    page_size,
    layout,
    paired,
    group,
    buffered,
    lazy,
):
    from aiter.ops.flydsl.kernels.fmha_gfx950.flash_attn_paged_fp8_gfx950 import (
        build_flash_attn_paged_fp8_module,
    )

    return build_flash_attn_paged_fp8_module(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        value_head_dim=value_dim,
        causal=True,
        dtype_str="fp8",
        varlen=True,
        cross_seqlen=True,
        paged=True,
        kv_cache_layout=layout,
        page_size=page_size,
        paged_bn128=paired,
        batch_interleave_group=group,
        cache_buffered=buffered,
        dualwave_swp_lazy_rescale=lazy,
    )


def flydsl_flash_attn_paged_fp8_func(
    q,
    k,
    v,
    cu_seqlens_q,
    max_seqlen_q,
    max_seqlen_k,
    *,
    block_table,
    seqlen_k,
    q_descale,
    k_descale,
    v_descale,
    softmax_scale=None,
    causal=True,
    out=None,
    dualwave_swp_lazy_rescale=True,
    stream=None,
):
    """Compute bottom-right causal prefill directly from native FP8 pages.

    Q is packed ``[total_q,Hq,Dqk]``. Page-1 K/V are linear rank 3/4;
    page-16/64/1024 K/V use the native vectorized rank-5 cache layouts.
    ``block_table`` contains physical page IDs and ``seqlen_k`` contains
    actual KV token lengths. Both maxima are nonnegative launch bounds.
    Metadata must be int32 on Q's device. Active IDs and lengths are the
    caller's valid cache metadata; unused table entries are never consumed.

    Q/K/V descales are single FP32 values on the same device. The independent
    softmax scale defaults to ``Dqk**-0.5`` and must be positive and finite.
    The output is contiguous BF16 and may be preallocated by the caller.
    """
    if not all(torch.is_tensor(tensor) for tensor in (q, k, v)):
        raise ValueError("paged FP8 Q/K/V must be tensors")
    if not (q.is_cuda and k.is_cuda and v.is_cuda and q.device == k.device == v.device):
        raise ValueError("paged FP8 Q/K/V must be on the same CUDA/HIP device")
    if q.ndim != 3:
        raise ValueError("paged FP8 Q must be packed [total_q,Hq,Dqk]")
    if not (q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn):
        raise ValueError("paged FP8 Q/K/V must use OCP float8_e4m3fn")
    if not causal:
        raise NotImplementedError("paged FP8 currently requires causal attention")
    if _gpu_arch(q.device) != "gfx950":
        raise NotImplementedError("paged FP8 requires gfx950")
    if softmax_scale is not None and not isinstance(softmax_scale, Real):
        raise ValueError("softmax_scale must be a positive finite Python scalar")
    if not _is_valid_softmax_scale(softmax_scale):
        raise ValueError("softmax_scale must be positive and finite")
    pages, page_size, kv_heads, dim, value_dim, layout = _cache_geometry(q, k, v)
    softmax_scale = dim**-0.5 if softmax_scale is None else float(softmax_scale)
    if (
        not torch.is_tensor(cu_seqlens_q)
        or cu_seqlens_q.ndim != 1
        or cu_seqlens_q.numel() < 1
    ):
        raise ValueError("cu_seqlens_q must be int32 [B+1] with at least one boundary")
    batch = cu_seqlens_q.numel() - 1
    for name, tensor, shape in (
        ("cu_seqlens_q", cu_seqlens_q, (batch + 1,)),
        ("seqlen_k", seqlen_k, (batch,)),
    ):
        if (
            not torch.is_tensor(tensor)
            or tensor.shape != shape
            or tensor.dtype != torch.int32
            or tensor.device != q.device
        ):
            raise ValueError(f"{name} must be int32 {shape} on {q.device}")
    if not isinstance(max_seqlen_q, Integral) or not isinstance(max_seqlen_k, Integral):
        raise TypeError("paged FP8 sequence maxima must be integers")
    if max_seqlen_q < 0 or max_seqlen_k < 0:
        raise ValueError("paged FP8 sequence maxima must be nonnegative")
    max_pages = (max_seqlen_k + page_size - 1) // page_size
    if (
        not torch.is_tensor(block_table)
        or block_table.ndim != 2
        or block_table.shape[0] != batch
        or block_table.shape[1] < max_pages
        or block_table.dtype != torch.int32
        or block_table.device != q.device
    ):
        raise ValueError(
            "block_table must be int32 [B,capacity] on Q's device and cover max_seqlen_k"
        )
    for name, tensor in (
        ("q_descale", q_descale),
        ("k_descale", k_descale),
        ("v_descale", v_descale),
    ):
        if (
            not torch.is_tensor(tensor)
            or tensor.device != q.device
            or tensor.dtype != torch.float32
            or tensor.numel() != 1
        ):
            raise ValueError(f"{name} must be one FP32 value on Q's device")
    output_shape = (q.shape[0], q.shape[1], value_dim)
    if max(q.numel(), q.shape[0] * q.shape[1] * value_dim) >= _MAX_FLAT_ELEMS:
        raise NotImplementedError(
            "paged FP8 flattened Q/O must each contain fewer than 2**31 elements"
        )
    if out is not None and (
        out.shape != output_shape
        or out.dtype != torch.bfloat16
        or out.device != q.device
        or not out.is_contiguous()
    ):
        raise ValueError(
            "paged FP8 out must be contiguous BF16 with the expected shape on Q's device"
        )

    with torch.cuda.device(q.device):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        if launch_stream.device != q.device:
            raise ValueError("paged FP8 stream must be on Q's device")
        with torch.cuda.stream(launch_stream) if stream is not None else nullcontext():
            if out is None:
                out = torch.empty(output_shape, dtype=torch.bfloat16, device=q.device)
            if q.numel() == 0 or pages == 0 or max_seqlen_k == 0:
                out.zero_()
                if stream is not None:
                    out.record_stream(launch_stream)
                return out
            paired = page_size == 64 and max_pages % 2 == 0
            group = (
                _batch_interleave_group(batch, (dim, value_dim), paired)
                if (q.shape[1], kv_heads) == (16, 1)
                else 1
            )
            buffered = (
                page_size in (1, 16)
                and max(k.numel(), v.numel()) <= PAGED_FP8_BUFFER_LIMIT_BYTES
            )
            launch = _build(
                num_heads=q.shape[1],
                num_kv_heads=kv_heads,
                head_dim=dim,
                value_dim=value_dim,
                page_size=page_size,
                layout=layout,
                paired=paired,
                group=group,
                buffered=buffered,
                lazy=dualwave_swp_lazy_rescale,
            )
            query = q.contiguous().view(-1)
            key, value = k.contiguous(), v.contiguous()
            table = block_table.contiguous().view(-1)
            cuq, lengths = cu_seqlens_q.contiguous(), seqlen_k.contiguous()
            scales = [
                s if s.stride() == (1,) else s.as_strided((1,), (1,))
                for s in (q_descale, k_descale, v_descale)
            ]
            launch(
                query,
                key,
                value,
                out.view(-1),
                batch,
                int(max_seqlen_q),
                seq_len_kv=int(max_seqlen_k),
                cu_seqlens_q=cuq,
                seqlen_k=lengths,
                block_table=table,
                block_table_stride=block_table.shape[1],
                q_descale=scales[0],
                k_descale=scales[1],
                v_descale=scales[2],
                softmax_scale=softmax_scale,
                stream=launch_stream,
            )
            if stream is not None:
                for tensor in (query, key, value, table, cuq, lengths, *scales, out):
                    tensor.record_stream(launch_stream)
    return out

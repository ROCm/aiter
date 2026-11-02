# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused QSA sparse decode on gfx950: attention straight from the KV pool.

Public entry for the kernels in ``kernels/qsa/``, serving Qwen sparse attention
(QSA) decode and speculative verify.

Every query row attends to its own top-k token positions. The kernel maps them
to pool slots through ``req_to_token`` itself, so there is no gather pass and no
packed scratch, and it splits each row's top-k list across CTAs. Speculative
verify rows use the same entry point: each row is its own length-one query.
"""

from __future__ import annotations

import functools

import flydsl.expr as fx
import torch

from .kernels.qsa.sparse_decode_attn import (
    HEAD_DIM,
    SPLIT_TOKEN_ALIGN,
    build_sparse_decode_attn,
)
from .kernels.qsa.sparse_decode_reduce import build_sparse_decode_reduce
from .kernels.tensor_shim import _run_compiled, ptr_arg

__all__ = [
    "can_use_flydsl_qsa_sparse_decode",
    "flydsl_qsa_sparse_decode",
    "recommended_num_splits",
    "workspace_partials",
]

_MAX_GROUP_SIZE = 16
# One CTA per CU. An extra split adds a prologue and a partial to merge, and
# below two 16-token tiles per wave the fixed per-CTA latency outweighs the
# extra parallelism.
_CTAS_PER_CU = 1
_MIN_TOKENS_PER_SPLIT = 2 * SPLIT_TOKEN_ALIGN
_KV_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)


@functools.cache
def _device_info(index: int) -> tuple[str, int]:
    props = torch.cuda.get_device_properties(index)
    return props.gcnArchName.split(":", 1)[0], props.multi_processor_count


def _split_layout(topk: int, num_splits: int) -> tuple[int, int]:
    """Tokens per split (whole wave rounds) and the split count that covers topk."""
    topk = max(1, topk)
    per_split = -(-topk // max(1, num_splits))
    per_split = -(-per_split // SPLIT_TOKEN_ALIGN) * SPLIT_TOKEN_ALIGN
    return per_split, -(-topk // per_split)


def recommended_num_splits(
    num_rows: int, num_kv_heads: int, topk: int, device: torch.device
) -> int:
    """KV splits that fill the GPU at this batch; fixed per CUDA-graph batch size."""
    _, num_cus = _device_info(torch.device(device).index or 0)
    wanted = -(-(_CTAS_PER_CU * num_cus) // max(1, num_rows * num_kv_heads))
    wanted = min(wanted, max(1, -(-topk // _MIN_TOKENS_PER_SPLIT)))
    return _split_layout(topk, wanted)[1]


@functools.cache
def workspace_partials(
    max_rows: int, num_q_heads: int, num_kv_heads: int, topk: int, device: torch.device
) -> int:
    """Partial rows covering every batch up to ``max_rows`` at its recommended split."""
    return max(
        rows * num_q_heads * recommended_num_splits(rows, num_kv_heads, topk, device)
        for rows in range(1, max(1, max_rows) + 1)
    )


def can_use_flydsl_qsa_sparse_decode(
    q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor
) -> bool:
    """Whether this gfx950 specialization can serve these operands."""
    if not (q.is_cuda and torch.version.hip is not None):
        return False
    if _device_info(q.device.index or 0)[0] != "gfx950":
        return False
    if k_cache.device != q.device or v_cache.device != q.device:
        return False
    if q.dim() != 3 or k_cache.dim() != 3 or v_cache.shape != k_cache.shape:
        return False
    if q.dtype != torch.bfloat16 or k_cache.dtype not in _KV_DTYPES:
        return False
    if v_cache.dtype != k_cache.dtype or q.shape[2] != HEAD_DIM:
        return False
    if q.shape[1] == 0 or k_cache.shape[1] == 0:
        return False
    if k_cache.shape[2] != HEAD_DIM or q.shape[1] % k_cache.shape[1]:
        return False
    if q.shape[1] // k_cache.shape[1] > _MAX_GROUP_SIZE:
        return False
    return k_cache.is_contiguous() and v_cache.is_contiguous()


def _check_device(name: str, tensor: torch.Tensor, device: torch.device) -> None:
    # ptr_arg forwards raw addresses, so a host tensor would be read as device memory.
    if tensor.device != device:
        raise ValueError(f"`{name}` must be on {device}, got {tensor.device}.")


def _check_index_tensor(name: str, tensor: torch.Tensor, rows: int) -> None:
    if tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"`{name}` must be int32 or int64, got {tensor.dtype}.")
    if tensor.dim() != 1 or tensor.shape[0] < rows or tensor.stride(0) != 1:
        raise ValueError(
            f"`{name}` must be a contiguous 1-D tensor with >= {rows} entries."
        )


def flydsl_qsa_sparse_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    row_req_pool_indices: torch.Tensor,
    indices: torch.Tensor,
    seq_lens: torch.Tensor,
    softmax_scale: float,
    *,
    num_splits: int,
    workspace: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Sparse attention of each query row over its selected KV pool tokens.

    ``q``: ``[rows, num_q_heads, 256]`` bf16. ``k_cache``/``v_cache``:
    ``[num_slots, num_kv_heads, 256]`` bf16 or fp8 e4m3fn (no scales).
    ``indices``: ``[rows, topk]`` int32 logical positions; an entry is attended iff
    ``0 <= pos < seq_lens[row]``, and maps to slot
    ``req_to_token[row_req_pool_indices[row], pos]``. ``workspace``: flat f32
    ``(part_m, part_l, part_o)`` holding at least ``rows * num_q_heads * splits``
    (times 256 for ``part_o``) entries, needed only when ``num_splits > 1``;
    allocated here when omitted. Returns ``[rows, num_q_heads, 256]`` bf16.
    """
    if not can_use_flydsl_qsa_sparse_decode(q, k_cache, v_cache):
        raise ValueError("unsupported operands for the gfx950 QSA sparse decode")
    rows, num_q_heads, _ = q.shape
    num_kv_heads = k_cache.shape[1]
    group_size = num_q_heads // num_kv_heads
    if not q.is_contiguous():
        raise ValueError("`q` must be contiguous.")
    for name, tensor in (
        ("req_to_token", req_to_token),
        ("row_req_pool_indices", row_req_pool_indices),
        ("indices", indices),
        ("seq_lens", seq_lens),
    ):
        _check_device(name, tensor, q.device)
    if indices.dtype != torch.int32 or indices.dim() != 2 or indices.stride(1) != 1:
        raise ValueError(
            "`indices` must be a [rows, topk] int32 tensor with inner stride 1."
        )
    if indices.shape[0] != rows:
        raise ValueError(f"`indices` must have {rows} rows, got {indices.shape[0]}.")
    if (
        req_to_token.dtype != torch.int32
        or req_to_token.dim() != 2
        or req_to_token.stride(1) != 1
    ):
        raise ValueError(
            "`req_to_token` must be a 2-D int32 tensor with inner stride 1."
        )
    _check_index_tensor("row_req_pool_indices", row_req_pool_indices, rows)
    _check_index_tensor("seq_lens", seq_lens, rows)

    out = torch.empty_like(q)
    topk = indices.shape[1]
    if rows == 0 or topk == 0:
        # No selected tokens: the same zeros the kernel writes for an empty row.
        return out.zero_()
    tokens_per_split, num_splits = _split_layout(topk, num_splits)
    single_split = num_splits == 1
    partial_rows = rows * num_q_heads * num_splits
    if single_split:
        part_m = part_l = part_o = out
    elif workspace is None:
        part_m = torch.empty(partial_rows, dtype=torch.float32, device=q.device)
        part_l = torch.empty_like(part_m)
        part_o = torch.empty(
            partial_rows * HEAD_DIM, dtype=torch.float32, device=q.device
        )
    else:
        part_m, part_l, part_o = workspace
        # The kernels index each buffer as a flat f32 array from its data pointer.
        for name, tensor, min_numel in (
            ("part_m", part_m, partial_rows),
            ("part_l", part_l, partial_rows),
            ("part_o", part_o, partial_rows * HEAD_DIM),
        ):
            _check_device(f"workspace.{name}", tensor, q.device)
            if tensor.dtype != torch.float32 or not tensor.is_contiguous():
                raise ValueError(
                    f"`workspace.{name}` must be a contiguous float32 tensor."
                )
            if tensor.numel() < min_numel:
                raise ValueError(
                    f"`workspace.{name}` needs >= {min_numel} elements for this "
                    f"batch and split count, got {tensor.numel()}."
                )

    attn = build_sparse_decode_attn(
        group_size=group_size,
        topk=topk,
        softmax_scale=float(softmax_scale),
        single_split=single_split,
        row_req_i64=row_req_pool_indices.dtype == torch.int64,
        seq_lens_i64=seq_lens.dtype == torch.int64,
        kv_fp8=k_cache.dtype == torch.float8_e4m3fn,
    )
    kv_ptr_dtype = fx.Int8 if k_cache.dtype == torch.float8_e4m3fn else fx.BFloat16
    # FlyDSL loads and launches on the current HIP device, not the stream's.
    with torch.cuda.device(q.device):
        stream = torch.cuda.current_stream(q.device)
        _run_compiled(
            attn,
            ptr_arg(q, fx.BFloat16),
            ptr_arg(k_cache, kv_ptr_dtype),
            ptr_arg(v_cache, kv_ptr_dtype),
            ptr_arg(req_to_token, fx.Int32),
            ptr_arg(row_req_pool_indices, fx.Int32),
            ptr_arg(indices, fx.Int32),
            ptr_arg(seq_lens, fx.Int32),
            ptr_arg(out, fx.BFloat16),
            ptr_arg(part_m, fx.Float32),
            ptr_arg(part_l, fx.Float32),
            ptr_arg(part_o, fx.Float32),
            rows,
            num_kv_heads,
            num_splits,
            tokens_per_split,
            indices.stride(0),
            req_to_token.stride(0),
            stream,
        )
        if not single_split:
            _run_compiled(
                build_sparse_decode_reduce(
                    group_size=group_size, num_splits=num_splits
                ),
                ptr_arg(part_m, fx.Float32),
                ptr_arg(part_l, fx.Float32),
                ptr_arg(part_o, fx.Float32),
                ptr_arg(out, fx.BFloat16),
                rows,
                num_kv_heads,
                stream,
            )
    return out

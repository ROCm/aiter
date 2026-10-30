# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""QSA operators for quantized cache writes, index scoring, selection and
sparse paged GQA."""

import functools

import torch
import triton

from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.triton._triton_kernels.attention.qsa import (
    _expand_selected_groups,
    _qsa_paged_mqa_logits_kernel,
    _qsa_paged_mqa_logits_mfma_kernel,
    _qsa_paged_mqa_logits_stacked_kernel,
    _qsa_sparse_paged_gqa_kernel,
    _qsa_store_rows_fp4_kernel,
    _qsa_store_rows_fp8_kernel,
)
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

# Quantized cache rows: FP8 with one FP32 scale per row, or MXFP4 with one
# E8M0 exponent for each FP4_GROUP dims (see the kernel module).
FP8_DTYPE = dtypes.fp8
# Without a visible GPU, aiter gives a uint8 placeholder that has no FP8 range.
FP8_MAX = float(torch.finfo(FP8_DTYPE).max) if FP8_DTYPE.is_floating_point else None
FP4_GROUP = 32
_FMT_BF16, _FMT_FP8, _FMT_FP4 = 0, 1, 2
QSA_CACHE_FORMATS = ("bf16", "fp8", "fp4")


def qsa_cache_row_layout(fmt: str, head_dim: int):
    """`(value dim, value dtype, scale tail, scale dtype)` of one cached row.

    The scale tail is the per-row scale shape after the row's own axes, None
    when the row is stored unquantized.
    """
    if fmt == "bf16":
        return head_dim, torch.bfloat16, None, None
    if fmt == "fp8":
        if FP8_MAX is None:
            raise ValueError("FP8 QSA rows need a GPU with an FP8 dtype")
        return head_dim, FP8_DTYPE, (), torch.float32
    if fmt == "fp4":
        # The FP4 kernels index the packed bytes with tl.arange(0, head_dim / 2).
        if head_dim < FP4_GROUP or head_dim & (head_dim - 1):
            raise ValueError(f"FP4 rows need a power-of-two head_dim >= {FP4_GROUP}")
        return head_dim // 2, torch.uint8, (head_dim // FP4_GROUP,), torch.uint8
    raise ValueError(f"unsupported QSA cache format {fmt!r}")


def _cache_format(cache: torch.Tensor, scale: torch.Tensor | None) -> int:
    """KEY_FMT of a cache. `_FMT_BF16` covers every unquantized float cache
    (BF16, FP16, FP32), which the kernels read in Q's dtype."""
    if FP8_MAX is not None and cache.dtype == FP8_DTYPE:
        fmt = _FMT_FP8
    elif cache.dtype == torch.uint8:
        fmt = _FMT_FP4
    elif cache.dtype.is_floating_point and cache.dtype.itemsize > 1:
        if scale is not None:
            raise ValueError("an unquantized QSA cache takes no scale plane")
        return _FMT_BF16
    else:
        raise ValueError(f"unsupported QSA cache dtype {cache.dtype}")
    if scale is None:
        raise ValueError("a quantized QSA cache needs its scale plane")
    return fmt


def _merge_slots(plane: torch.Tensor, name: str) -> torch.Tensor:
    """`[pages, page_size, ...]` viewed as `[pages * page_size, ...]`."""
    (pages, page_size, *rest), (page_stride, slot_stride, *_) = (
        plane.shape,
        plane.stride(),
    )
    if pages > 1 and page_size > 1 and page_stride != page_size * slot_stride:
        raise ValueError(
            f"{name} must hold its pages back to back (stride(0) == page_size * "
            "stride(1)); a slice of a larger page pool does not"
        )
    return plane.view(pages * page_size, *rest)


def qsa_store_quantized_rows(
    cache: torch.Tensor,
    scale: torch.Tensor,
    slots: torch.Tensor,
    values: torch.Tensor,
    second_cache: torch.Tensor | None = None,
    second_scale: torch.Tensor | None = None,
    second_values: torch.Tensor | None = None,
) -> None:
    """Write `values[i]` into slot `slots[i]` of an FP8 or FP4 cache plus its
    scale plane; negative or out-of-range slots are skipped.

    `cache` is `[pages, page_size, heads, stored dim]`, `scale` `[pages,
    page_size, heads(, groups)]` (see `qsa_cache_row_layout`), `values`
    `[rows, heads, head_dim]`. FP8 may take a second tensor (V beside K) that
    shares the slots and both planes' strides, written in the same launch.
    """
    fmt = _cache_format(cache, scale)
    if fmt == _FMT_BF16:
        raise ValueError("qsa_store_quantized_rows writes FP8 or FP4 caches")
    given = [t is not None for t in (second_cache, second_scale, second_values)]
    if any(given) and not all(given):
        raise ValueError("pass second_cache, second_scale and second_values together")
    second = all(given)
    if cache.ndim != 4 or values.ndim != 3:
        raise ValueError("cache must be [pages, page, heads, dim], values 3-D")
    rows, heads, head_dim = values.shape
    _check_vector("slots", slots)
    if rows == 0:
        return
    if slots.shape[0] < rows or heads != cache.shape[2]:
        raise ValueError("slots or heads disagree with the cache")
    _, _, tail, scale_dtype = qsa_cache_row_layout(QSA_CACHE_FORMATS[fmt], head_dim)
    if scale.shape != (*cache.shape[:3], *tail) or scale.dtype != scale_dtype:
        raise ValueError(
            f"scale must be the {scale_dtype} [pages, page, heads(, groups)] "
            "plane of the cache"
        )
    if (
        values.stride(-1) != 1
        or cache.stride(-1) != 1
        or (tail and scale.stride(-1) != 1)
    ):
        raise ValueError("quantized QSA rows must be contiguous in the head dim")
    flat = _merge_slots(cache, "cache")
    flat_scale = _merge_slots(scale, "scale")
    if fmt == _FMT_FP8:
        if cache.shape[3] != head_dim:
            raise ValueError("values and FP8 cache disagree on head_dim")
        if second and (
            second_cache.shape != cache.shape
            or second_cache.dtype != cache.dtype
            or second_scale.shape != scale.shape
            or second_scale.dtype != scale.dtype
            or second_cache.stride() != cache.stride()
            or second_scale.stride() != scale.stride()
            or second_values.shape != values.shape
            or second_values.stride(-1) != 1
        ):
            raise ValueError("the second FP8 tensor must mirror the first")
        _qsa_store_rows_fp8_kernel[(rows, heads)](
            flat,
            flat_scale,
            slots,
            values,
            flat.stride(0),
            flat.stride(1),
            flat_scale.stride(0),
            flat_scale.stride(1),
            values.stride(0),
            values.stride(1),
            second_cache if second else flat,
            second_scale if second else flat_scale,
            second_values if second else values,
            second_values.stride(0) if second else 0,
            second_values.stride(1) if second else 0,
            rows,
            flat.shape[0],
            HEAD_DIM=head_dim,
            BLOCK_D=triton.next_power_of_2(head_dim),
            FP8_MAX_C=FP8_MAX,
            HAS_SECOND=second,
            num_warps=4,
        )
        return
    if second:
        raise ValueError("only the FP8 writer takes a second tensor")
    if cache.shape[3] * 2 != head_dim:
        raise ValueError("values and FP4 cache disagree on head_dim")
    _qsa_store_rows_fp4_kernel[(rows, heads)](
        flat,
        flat_scale,
        slots,
        values,
        flat.stride(0),
        flat.stride(1),
        flat_scale.stride(0),
        flat_scale.stride(1),
        values.stride(0),
        values.stride(1),
        rows,
        flat.shape[0],
        HEAD_DIM=head_dim,
        GROUP=FP4_GROUP,
        num_warps=1,
    )


def _prev_pow2(n: int) -> int:
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _table_width_bucket(width: int) -> int:
    """Page-table width as seen by the kernels, rounded up to a power of two."""
    return 1 << (max(int(width), 1) - 1).bit_length()


def _kv_splits_heuristic(
    T: int,
    kv_heads: int,
    topk: int,
    num_cu: int | None = None,
    target_wg_per_cu: float = 4.0,
    max_kv_splits: int = 64,
) -> int:
    if topk < 512:
        return 1
    if num_cu is None:
        num_cu = get_cu_num()
    target_wg = max(1, int(target_wg_per_cu * num_cu))
    base_ctas = max(1, T * kv_heads)
    if base_ctas >= target_wg:
        return 1
    return _prev_pow2(min(target_wg // base_ctas, max_kv_splits))


def _kernel_config(
    T: int,
    kv_heads: int,
    kv_splits: int,
    group_size: int,
    num_cu: int | None = None,
) -> tuple[int, int, int, int, int]:
    """Pick (BLOCK_N, num_warps, num_stages, waves_per_eu, sub_group)."""
    if num_cu is None:
        num_cu = get_cu_num()
    sub_group = group_size
    if group_size > 16:
        sub_group = 8
    num_head_groups = (group_size + sub_group - 1) // sub_group
    grid_size = T * kv_heads * num_head_groups * kv_splits
    if grid_size <= num_cu:
        return 64, 4, 1, 0, group_size
    if grid_size >= num_cu * 8:
        return 32, 2, 1, 0, sub_group
    return 32, 4, 1, 3, sub_group


_SCORING_BLOCK_N = 32
_SCORING_COLUMN_PROGRAMS = 64


@functools.lru_cache(maxsize=1024)
def _tuned_tile(kernel: str, case: str) -> dict:
    """`kernel[case]` of `configs/<arch>/triton/attention/qsa/DEFAULT.json`."""
    cfg_dir = resolve_config_dir("attention", "QSA", backend="triton")
    return load_config_json(f"{cfg_dir}/DEFAULT.json")[kernel][case]


def _check_vector(name: str, tensor: torch.Tensor, length: int | None = None) -> None:
    if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{name} must be a one-dimensional int32/int64 tensor")
    if length is not None and tensor.shape[0] != length:
        raise ValueError(f"{name} must contain {length} entries")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def qsa_paged_mqa_logits(
    q: torch.Tensor,
    compressed_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    compress_ratio: int = 4,
    score_divisor: float | None = None,
    max_columns: int | None = None,
    row_starts: torch.Tensor | None = None,
    compressed_k_scale: torch.Tensor | None = None,
    stacked_decode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score compressed key groups.

    `q` is `[tokens, index_heads, head_dim]` and `compressed_k_cache` is
    `[pages, page_size, 1, head_dim]`. Returns FP32 logits `[tokens, columns]`
    and the per-row count of causally visible groups.

    An unquantized cache (BF16, FP16, FP32) has Q's dtype. A quantized cache
    comes with `compressed_k_scale`: FP8 (`head_dim` E4M3, scale `[pages,
    page_size, 1]` FP32) or MXFP4 (`head_dim / 2` bytes, scale `[pages,
    page_size, 1, head_dim / 32]` E8M0). Q stays in its own dtype.

    `stacked_decode` scores decode-sized calls (<= 256 rows) with an MFMA
    kernel that stacks each row's heads into M, so a request's keys are read
    once per call rather than once per row.
    """
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("q must have shape [tokens, heads, head_dim]")
    key_fmt = _cache_format(compressed_k_cache, compressed_k_scale)
    stored_dim = q.shape[2] // 2 if key_fmt == _FMT_FP4 else q.shape[2]
    if (
        compressed_k_cache.ndim != 4
        or compressed_k_cache.shape[2] != 1
        or compressed_k_cache.shape[3] != stored_dim
    ):
        raise ValueError(
            "compressed_k_cache must have shape [pages, page_size, 1, head_dim]"
        )
    if key_fmt == _FMT_BF16 and q.dtype != compressed_k_cache.dtype:
        raise ValueError("q and compressed_k_cache must have the same dtype")
    if key_fmt != _FMT_BF16:
        _, _, tail, scale_dtype = qsa_cache_row_layout(
            QSA_CACHE_FORMATS[key_fmt], q.shape[2]
        )
        if (
            compressed_k_scale.shape != (*compressed_k_cache.shape[:3], *tail)
            or compressed_k_scale.dtype != scale_dtype
            or compressed_k_scale.stride(-1) != 1
            or q.stride(2) != 1
            or compressed_k_cache.stride(3) != 1
        ):
            raise ValueError("compressed_k_scale must be the scale plane of the cache")
    # BF16 keys pass a None scale, which binds as a constexpr.
    scale = compressed_k_scale
    scale_strides = (0, 0) if scale is None else (scale.stride(0), scale.stride(1))
    if page_table.ndim != 2 or page_table.dtype not in (torch.int32, torch.int64):
        raise ValueError("page_table must be a two-dimensional integer tensor")
    _check_vector("token_to_request", token_to_request, q.shape[0])
    _check_vector("query_positions", query_positions, q.shape[0])
    _check_vector("context_lens", context_lens, page_table.shape[0])
    if row_starts is not None:
        _check_vector("row_starts", row_starts, q.shape[0])

    divisor = q.shape[2] ** 0.5 if score_divisor is None else score_divisor
    if divisor <= 0:
        raise ValueError("score_divisor must be positive")
    columns = page_table.shape[1] * compressed_k_cache.shape[1]
    if max_columns is not None:
        columns = max(1, min(columns, int(max_columns)))
    logits = torch.empty((q.shape[0], columns), dtype=torch.float32, device=q.device)
    visible_groups = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    if q.shape[0] == 0 or columns == 0:
        visible_groups.zero_()
        if row_starts is not None:
            row_starts.zero_()
        return logits, visible_groups

    contiguous = (
        q.stride(2) == 1
        and compressed_k_cache.stride(3) == 1
        and page_table.stride(1) == 1
    )
    heads, head_dim = q.shape[1], q.shape[2]
    tile = None
    if (
        q.shape[0] <= 256
        and stacked_decode
        and contiguous
        and heads & (heads - 1) == 0
        and head_dim & (head_dim - 1) == 0
        and head_dim >= 16
    ):
        tile = _tuned_tile("stacked_decode", QSA_CACHE_FORMATS[key_fmt])
    if tile is not None and tile["BMT"] * heads >= 16:
        bmt, bn = tile["BMT"], tile["BN"]
        _qsa_paged_mqa_logits_stacked_kernel[
            (
                triton.cdiv(q.shape[0], bmt),
                min(triton.cdiv(columns, bn), tile["column_programs"]),
            )
        ](
            q,
            compressed_k_cache,
            scale,
            page_table,
            token_to_request,
            query_positions,
            context_lens,
            visible_groups,
            row_starts,
            logits,
            q.stride(0),
            q.stride(1),
            compressed_k_cache.stride(0),
            compressed_k_cache.stride(1),
            *scale_strides,
            page_table.stride(0),
            logits.stride(0),
            q.shape[0],
            columns,
            compressed_k_cache.shape[0],
            page_table.shape[0],
            float(divisor),
            PAGE_SIZE=compressed_k_cache.shape[1],
            PAGE_TABLE_WIDTH=_table_width_bucket(page_table.shape[1]),
            NUM_HEADS=heads,
            HEAD_DIM=head_dim,
            COMPRESS_RATIO=compress_ratio,
            BMT=bmt,
            BN=bn,
            KEY_FMT=key_fmt,
            GROUP=FP4_GROUP,
            num_warps=tile["num_warps"],
        )
        return logits, visible_groups
    if q.shape[0] > 256 and contiguous and q.shape[2] >= 16:
        bmt, bn = 16, 128
        _qsa_paged_mqa_logits_mfma_kernel[
            (triton.cdiv(q.shape[0], bmt), triton.cdiv(columns, bn))
        ](
            q,
            compressed_k_cache,
            scale,
            page_table,
            token_to_request,
            query_positions,
            context_lens,
            visible_groups,
            row_starts,
            logits,
            q.stride(0),
            q.stride(1),
            compressed_k_cache.stride(0),
            compressed_k_cache.stride(1),
            *scale_strides,
            page_table.stride(0),
            logits.stride(0),
            q.shape[0],
            columns,
            compressed_k_cache.shape[0],
            page_table.shape[0],
            float(divisor),
            PAGE_SIZE=compressed_k_cache.shape[1],
            PAGE_TABLE_WIDTH=_table_width_bucket(page_table.shape[1]),
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            COMPRESS_RATIO=compress_ratio,
            BMT=bmt,
            BN=bn,
            KEY_FMT=key_fmt,
            GROUP=FP4_GROUP,
            num_warps=4,
        )
        return logits, visible_groups
    # Fallback calls use the CUDA-core kernel.
    column_programs = min(
        triton.cdiv(columns, _SCORING_BLOCK_N), _SCORING_COLUMN_PROGRAMS
    )
    _qsa_paged_mqa_logits_kernel[(q.shape[0], column_programs)](
        q,
        compressed_k_cache,
        scale,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        visible_groups,
        row_starts,
        logits,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        compressed_k_cache.stride(0),
        compressed_k_cache.stride(1),
        compressed_k_cache.stride(3),
        *scale_strides,
        page_table.stride(0),
        page_table.stride(1),
        logits.stride(0),
        q.shape[0],
        columns,
        compressed_k_cache.shape[0],
        page_table.shape[0],
        float(divisor),
        PAGE_SIZE=compressed_k_cache.shape[1],
        PAGE_TABLE_WIDTH=_table_width_bucket(page_table.shape[1]),
        NUM_HEADS=q.shape[1],
        HEAD_DIM=q.shape[2],
        COMPRESS_RATIO=compress_ratio,
        BLOCK_N=_SCORING_BLOCK_N,
        BLOCK_D=triton.next_power_of_2(q.shape[2]),
        KEY_FMT=key_fmt,
        GROUP=FP4_GROUP,
        num_warps=4,
    )
    return logits, visible_groups


def qsa_expand_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    context_lens: torch.Tensor,
    token_to_request: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
    output_width: int | None = None,
) -> torch.Tensor:
    """Expand selected groups to token ids and append the causal tail."""
    if compress_ratio <= 0 or token_topk <= 0:
        raise ValueError("compress_ratio and token_topk must be positive")
    if block_indices.ndim != 2 or block_indices.dtype != torch.int32:
        raise ValueError("block_indices must be a two-dimensional int32 tensor")
    if token_topk % compress_ratio:
        raise ValueError("token_topk must be divisible by compress_ratio")
    _check_vector("query_positions", query_positions, block_indices.shape[0])
    _check_vector("token_to_request", token_to_request, block_indices.shape[0])
    _check_vector("context_lens", context_lens)

    block_topk = block_indices.shape[1]
    if block_topk > token_topk // compress_ratio:
        raise ValueError(
            f"block_indices must have at most {token_topk // compress_ratio} columns"
        )
    if output_width is None:
        output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.dtype != torch.int32 or out.shape != (
        block_indices.shape[0],
        output_width,
    ):
        raise ValueError("out has an invalid shape")
    if block_indices.shape[0] == 0:
        return out

    _expand_selected_groups[(block_indices.shape[0], triton.cdiv(output_width, 128))](
        block_indices,
        context_lens,
        query_positions,
        token_to_request,
        out,
        context_lens.numel(),
        block_topk,
        output_width,
        block_indices.stride(0),
        block_indices.stride(1),
        out.stride(0),
        out.stride(1),
        RATIO=compress_ratio,
        BLOCK=128,
    )
    return out


def qsa_sparse_paged_gqa(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    softmax_scale: float | None = None,
    kv_splits: int | None = None,
    num_decode_requests: int | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Grouped-query attention restricted to `logical_indices` (-1 = padding).

    FP8 K/V come with `k_scale` / `v_scale`, FP32 `[pages, page, heads]`
    (see `qsa_store_quantized_rows`).
    """
    if q.ndim != 3:
        raise ValueError("q must be [tokens, query_heads, head_dim]")
    if k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("K/V caches must have matching [pages, page, heads, dim]")
    kv_fp8 = _cache_format(k_cache, k_scale) == _FMT_FP8
    if kv_fp8:
        if (
            v_cache.dtype != k_cache.dtype
            or v_scale is None
            or k_scale.shape != k_cache.shape[:3]
            or v_scale.shape != k_scale.shape
            or v_scale.stride() != k_scale.stride()
            or k_scale.dtype != torch.float32
            or v_scale.dtype != torch.float32
        ):
            raise ValueError(
                "FP8 K/V need matching float32 [pages, page, heads] scales"
            )
    elif v_scale is not None:
        raise ValueError("v_scale needs an FP8 cache and k_scale")
    elif q.dtype != k_cache.dtype or q.dtype != v_cache.dtype:
        raise ValueError("q, k_cache and v_cache must have the same dtype")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("query heads must form equal groups over KV heads")
    if (
        logical_indices.ndim != 2
        or logical_indices.shape[0] != q.shape[0]
        or logical_indices.dtype != torch.int32
    ):
        raise ValueError("logical_indices must be int32 [tokens, selection_width]")
    if block_table.ndim != 2:
        raise ValueError("block_table must be a two-dimensional integer tensor")
    if (
        token_to_request.shape != (q.shape[0],)
        or token_to_request.dtype not in (torch.int32, torch.int64)
        or not token_to_request.is_contiguous()
    ):
        raise ValueError("token_to_request must be contiguous int32/int64 [tokens]")

    scale = q.shape[2] ** -0.5 if softmax_scale is None else softmax_scale
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if q.shape[0] == 0:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_d = max(16, triton.next_power_of_2(q.shape[2]))
    if kv_splits is None:
        split_rows = q.shape[0]
        if split_rows <= 0:
            raise ValueError("num_decode_requests must be positive")
        kv_splits = _kv_splits_heuristic(
            split_rows, k_cache.shape[2], logical_indices.shape[1]
        )
    if kv_splits < 1 or kv_splits & (kv_splits - 1):
        raise ValueError("kv_splits must be a positive power of two")
    block_n, num_warps, num_stages, waves_per_eu, sub_group = _kernel_config(
        q.shape[0], k_cache.shape[2], kv_splits, group_size
    )
    if kv_fp8:
        phase = "prefill" if num_decode_requests is None else "decode"
        tile = _tuned_tile("gqa_fp8", phase)
        block_n, num_warps = tile["BLOCK_N"], tile["num_warps"]
        num_stages, waves_per_eu = tile["num_stages"], tile["waves_per_eu"]
    num_head_groups = (group_size + sub_group - 1) // sub_group
    block_m = max(16, triton.next_power_of_2(sub_group))
    if logical_indices.shape[1] == 0:
        return out.zero_()
    target = out
    partial_max = partial_sum = out
    widths = None
    if kv_splits > 1:
        shape = (q.shape[0], q.shape[1], kv_splits)
        partial_max = torch.empty(shape, device=q.device, dtype=torch.float32)
        partial_sum = torch.empty_like(partial_max)
        target = torch.empty((*shape, q.shape[2]), device=q.device, dtype=torch.float32)
        widths = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    grid_y = k_cache.shape[2] * num_head_groups
    _qsa_sparse_paged_gqa_kernel[(q.shape[0], grid_y, kv_splits)](
        q,
        k_cache,
        v_cache,
        k_scale if kv_fp8 else None,
        v_scale if kv_fp8 else None,
        logical_indices,
        block_table,
        token_to_request,
        target,
        partial_max,
        partial_sum,
        widths,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        *(k_scale.stride() if kv_fp8 else (0, 0, 0)),
        logical_indices.stride(0),
        logical_indices.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        target.stride(0),
        target.stride(1),
        target.stride(-1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        float(scale),
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=_table_width_bucket(block_table.shape[1]),
        NUM_KV_HEADS=k_cache.shape[2],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        KV_SPLITS=kv_splits,
        SUB_GROUP=sub_group if sub_group < group_size else 0,
        KV_FP8=kv_fp8,
        num_warps=num_warps,
        num_stages=num_stages,
        waves_per_eu=waves_per_eu,
    )
    if kv_splits > 1:
        from aiter.ops.triton._triton_kernels.attention.mla import (
            _mla_decode_fwd_reduce_kernel,
        )

        _mla_decode_fwd_reduce_kernel[(q.shape[0], q.shape[1])](
            out,
            target,
            partial_max,
            partial_sum,
            widths,
            None,
            q.shape[0],
            q.shape[1],
            out.stride(0),
            out.stride(1),
            1,
            1,
            q.shape[0],
            TILE_SIZE=block_n,
            KV_LORA_RANK=q.shape[2],
            query_start_len_ptr=None,
            BLOCK_Q=1,
            NUM_SEGMENTS_PER_SEQ=kv_splits,
            ALL_DECODE=True,
        )
    return out

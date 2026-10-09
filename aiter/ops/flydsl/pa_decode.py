# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""FP8 or BF16 paged-attention decode on gfx942/gfx950.

Cache layouts are logical, not preshuffled. K/V use E4M3 FNUZ on gfx942 and
OCP on gfx950. See ``kernels.pa_decode_kernel`` for Q/P quantization and
MFMA specialization details. Native BF16 uses unscaled, 8-element vectorized
K/V caches and BF16 queries.
"""

from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.autotune import Config, autotune

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.pa_decode import implementation_cache_tag
from .kernels.pa_decode_kernel import KV_COMPUTE_BLOCK, compile_pa_decode_tile
from .kernels.pa_decode_plan import PADecodePlan
from .kernels.pa_decode_plan import plan_pa_decode as plan_pa_decode  # noqa: PLC0414
from .kernels.pa_decode_reduce import compile_pa_decode_ps_reduce
from .kernels.tensor_shim import _run_compiled, get_dtype_str, ptr_arg


def _flydsl_pointer_dtype(dtype: torch.dtype):
    return {
        torch.float32: fx.Float32,
        torch.float16: fx.Float16,
        torch.bfloat16: fx.BFloat16,
        torch.float8_e4m3fn: fx.Float8E4M3FN,
        torch.float8_e4m3fnuz: fx.Float8E4M3FNUZ,
        torch.int32: fx.Int32,
    }[dtype]


def launch_pa_decode_ps_reduce(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    logits: torch.Tensor,
    sink_token: torch.Tensor | None,
    stride_output_bs: int,
    stride_output_len: int,
    stride_output_kv_head: int,
    stride_output_group_size: int,
    stride_exp_sums_head: int,
    stride_exp_sums_part: int,
    stride_logits_head: int,
    stride_logits_part: int,
    stride_logits_group: int,
    *,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    context_partition_num: int,
    stream: torch.cuda.Stream,
    reduce_info: torch.Tensor,
) -> None:
    partition_limit = torch.cuda.get_device_properties(
        output.device
    ).multi_processor_count
    if context_partition_num > partition_limit:
        raise ImportError(
            f"FlyDSL pa_decode reduce supports at most {partition_limit} partitions"
        )
    use_sinks = sink_token is not None
    # Bound all attempted i32 byte offsets, including inactive partitions,
    # so out-of-bounds reads cannot wrap back into valid data.
    bounded_plan_logits = (
        context_partition_num <= 64
        and query_seq_len > 0
        and query_group_size > 0
        and 0 <= stride_logits_head <= 2**31 - 1
        and stride_logits_group == head_size
        and stride_logits_part == query_seq_len * query_group_size * head_size
        and 0
        < context_partition_num * stride_logits_part * logits.element_size()
        <= 2**31 - 1
    )
    # Paired loads need dword alignment; scalar output stores do not.
    vectorize_plan_logits = (
        bounded_plan_logits
        and head_size == 128
        and logits.dtype in (torch.bfloat16, torch.float16)
        and logits.data_ptr() % 4 == 0
        and stride_logits_head % 2 == 0
        and stride_logits_part % 2 == 0
        and stride_logits_group % 2 == 0
    )
    compiled = compile_pa_decode_ps_reduce(
        max_context_partition_num=context_partition_num,
        head_size=head_size,
        output_dtype_str=get_dtype_str(output.dtype),
        logits_dtype_str=get_dtype_str(logits.dtype),
        sink_dtype_str=get_dtype_str(
            output.dtype if sink_token is None else sink_token.dtype
        ),
        use_sinks=use_sinks,
        query_group_size=query_group_size,
        bounded_plan_logits=bounded_plan_logits,
        vectorize_plan_logits=vectorize_plan_logits,
        compact_plan=(
            head_size == 128
            and context_partition_num > 64
            and exp_sums.ndim == 3
            and exp_sums.shape[1] <= 32 * output.shape[0]
        ),
    )
    sink_ptr = (
        ptr_arg(sink_token, _flydsl_pointer_dtype(sink_token.dtype))
        if use_sinks
        else flyc.from_c_void_p(_flydsl_pointer_dtype(output.dtype), 0)
    )
    _run_compiled(
        compiled["launch"],
        ptr_arg(output, _flydsl_pointer_dtype(output.dtype)),
        ptr_arg(exp_sums, fx.Float32),
        ptr_arg(max_logits, fx.Float32),
        ptr_arg(logits, _flydsl_pointer_dtype(logits.dtype)),
        sink_ptr,
        stride_output_bs,
        stride_output_len,
        stride_output_kv_head,
        stride_output_group_size,
        stride_exp_sums_head,
        stride_exp_sums_part,
        stride_logits_head,
        stride_logits_part,
        stride_logits_group,
        query_seq_len,
        query_group_size,
        output.shape[0],
        output.shape[2],
        ptr_arg(reduce_info, fx.Int32),
        stream,
    )


def pa_decode(
    output: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    query: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    softmax_scale: float,
    query_length: int,
    *,
    context_partition_size: int = 256,
    compute_type: torch.dtype = torch.bfloat16,
    query_scale: torch.Tensor = None,
    key_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    exp_sums: torch.Tensor = None,  # [num_kv_heads, capacity, query_length * query_group_size]
    max_logits: torch.Tensor = None,  # [num_kv_heads, capacity, query_length * query_group_size]
    temporary_output: torch.Tensor = None,  # [num_kv_heads, capacity, query_length * query_group_size, head_size]
    alibi_slopes: torch.Tensor = None,
    sinks: torch.Tensor = None,
    sliding_window: int = 0,
    work_plan: PADecodePlan,
    max_context_length: int | None = None,
) -> None:
    """Decode FP8 or BF16 K/V using 256-token compute tiles.

    Supports page sizes 16/64/128. BF16 head dimensions are multiples of 64
    up to 1024; FP8 supports 64 and multiples of 128 up to 1024.
    K/V scales are [1] or [num_blocks, num_kv_heads, block_size, 1].
    BF16 K/V requires BF16 queries, compute_type=torch.bfloat16, no K/V scales,
    and vectorized 5D caches with x=8; FP8 uses x=16. BF16 Q/P operands stay
    in BF16 and softmax statistics and MFMA accumulation stay in FP32.
    BF16 automatically splits query positions to keep each CTA within 64 KiB
    of LDS; a single query's GQA rows must fit that limit. Q/K and V/output
    head dimensions must match.
    ALiBi and externally quantized queries are unsupported.

    MTP lengths include the query tokens and use dense causal masking.
    Independently selected sparse queries need separate table rows and
    query_length=1. Positive ``sliding_window`` includes each query's own
    position; 0 and -1 disable it.

    ``work_plan`` is required and must be built with ``plan_pa_decode`` before
    calling decode. ``prepare_pa_decode_plan`` uses native FlyDSL autotune to
    select its budget before graph capture. A cache miss uses twice the device
    CU count; ``FLYDSL_AUTOTUNE=1`` forces a search during preparation. Decode
    and graph replay use the explicit plan without accessing autotune. Set its
    partition cap with ``plan_pa_decode(max_partitions=...)``; even a cap of one
    uses packed scratch and reduction.

    ``sinks`` is a contiguous [num_query_heads] BF16/FP16/FP32 tensor on the
    query device: unscaled zero-value logits shared across batch/MTP positions.
    Each contributes once, independently of the window; -inf disables it and
    +inf suppresses the head's output.

    Refresh ``work_plan`` on the current stream after changing lengths. Its
    max_partitions is bounded by device CUs. Match sliding_window and, for
    windowed plans, query_length. Scratch is [KV heads, capacity, query rows
    (, D)], with query rows = query_length * query_group_size.

    ``max_context_length`` selects the scheduling bound, rounded up to
    whole pages as in tuning. This keeps the selected specialization independent
    of extra block-table padding. Callers must ensure
    every GPU context length is at most ``max_context_length`` when supplying it.

    GPU lengths and table entries are not checked. Callers must ensure
    0 <= context_lengths[i] <= block_tables.shape[1] * block_size and used
    block indices in [0, min(key_cache.shape[0], value_cache.shape[0])).
    """
    if not isinstance(work_plan, PADecodePlan):
        raise TypeError("work_plan must be a PADecodePlan")
    if context_partition_size != KV_COMPUTE_BLOCK:
        raise NotImplementedError(
            "pa_decode only supports context_partition_size=256, "
            f"got {context_partition_size}"
        )
    if query_scale is not None:
        raise NotImplementedError(
            "pa_decode does not support externally quantized FP8 queries"
        )
    if alibi_slopes is not None:
        raise NotImplementedError("pa_decode does not support ALiBi")
    if not isinstance(sliding_window, int):
        raise TypeError("sliding_window must be an int")
    if sliding_window < -1:
        raise ValueError("sliding_window must be -1, 0, or positive")
    sliding_window = max(sliding_window, 0)
    if not isinstance(query_length, int):
        raise TypeError("query_length must be an int")
    if query_length < 1:
        raise ValueError(f"query_length must be positive, got {query_length}")

    required_tensors = (
        ("output", output),
        ("query", query),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("context_lengths", context_lengths),
        ("block_tables", block_tables),
    )
    for name, tensor in required_tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
            )

    expected_ranks = (
        ("output", output, (3,)),
        ("query", query, (3,)),
        ("key_cache", key_cache, (5,)),
        ("value_cache", value_cache, (4, 5)),
        ("context_lengths", context_lengths, (1,)),
        ("block_tables", block_tables, (2,)),
    )
    for name, tensor, ranks in expected_ranks:
        if tensor.dim() not in ranks:
            expected = " or ".join(f"{rank}D" for rank in ranks)
            raise ValueError(
                f"{name} must be {expected}, got shape {tuple(tensor.shape)}"
            )

    num_seqs = context_lengths.shape[0]
    total_q_rows, num_q_heads, head_dim = query.shape
    if query.device.type != "cuda":
        raise ValueError(f"query must be on a CUDA device, got {query.device}")
    if num_q_heads < 1:
        raise ValueError(f"query must contain at least one head, got {num_q_heads}")
    if total_q_rows != num_seqs * query_length:
        raise ValueError(
            f"query.shape[0] ({total_q_rows}) must equal "
            f"context_lengths.shape[0] * query_length ({num_seqs} * {query_length})"
        )
    if output.shape != query.shape:
        raise ValueError(
            f"output shape {tuple(output.shape)} must match "
            f"query shape {tuple(query.shape)}"
        )
    if block_tables.shape[0] != num_seqs:
        raise ValueError(
            f"block_tables.shape[0] ({block_tables.shape[0]}) must match "
            f"context_lengths.shape[0] ({num_seqs})"
        )

    num_blocks, num_kv_heads, num_hgroups, block_size, hgroup_width = key_cache.shape
    if num_kv_heads < 1:
        raise ValueError(
            f"key_cache must contain at least one KV head, got {num_kv_heads}"
        )

    arch = get_gfx_runtime()
    expected_fp8_dtype = {
        "gfx942": torch.float8_e4m3fnuz,
        "gfx950": torch.float8_e4m3fn,
    }.get(arch)
    if expected_fp8_dtype is None:
        raise NotImplementedError(
            f"pa_decode only supports gfx942 and gfx950, got {arch}"
        )
    is_bf16_kv = key_cache.dtype == torch.bfloat16
    kv_dtype = "bf16" if is_bf16_kv else "fp8"
    expected_kv_dtype = torch.bfloat16 if is_bf16_kv else expected_fp8_dtype
    if compute_type != expected_kv_dtype:
        raise NotImplementedError(
            f"pa_decode requires compute_type={expected_kv_dtype} for the KV cache "
            f"on {arch}, got {compute_type}"
        )
    vector_width = 8 if is_bf16_kv else 16

    # Enforce tile Q-load constraints and the reducer's 1024-thread limit.
    q_chunk = head_dim // 16
    if not (
        64 <= head_dim <= 1024
        and head_dim % 64 == 0
        and (is_bf16_kv or q_chunk <= 8 or q_chunk % 8 == 0)
    ):
        raise NotImplementedError(
            f"pa_decode does not support head_dim={head_dim}; supported values "
            + (
                "are multiples of 64 in [64, 1024] for BF16 KV"
                if is_bf16_kv
                else "are 64 and multiples of 128 in [128, 1024] for FP8 KV"
            )
        )
    if num_hgroups != head_dim // vector_width or hgroup_width != vector_width:
        raise ValueError(
            "key_cache shape must be "
            f"[num_blocks, num_kv_heads, head_dim // {vector_width}, "
            f"block_size, {vector_width}], "
            f"got {tuple(key_cache.shape)} for head_dim={head_dim}"
        )
    if block_size not in (16, 64, 128):
        raise ValueError(
            f"pa_decode only supports block_size in (16, 64, 128), got {block_size}"
        )

    trans_v = value_cache.dim() == 5
    if is_bf16_kv and not trans_v:
        raise ValueError("BF16 KV requires the vectorized 5D value_cache layout")
    if trans_v:
        v_num_blocks, v_num_kv_heads = value_cache.shape[:2]
        expected_v_tail = (block_size // vector_width, head_dim, vector_width)
        if tuple(value_cache.shape[2:]) != expected_v_tail:
            raise ValueError(
                "transposed value_cache shape must be "
                f"[num_blocks, num_kv_heads, block_size // {vector_width}, "
                f"head_dim, {vector_width}], "
                f"got {tuple(value_cache.shape)} for block_size={block_size}, "
                f"head_dim={head_dim}"
            )
    else:
        v_num_blocks, v_num_kv_heads, v_head_dim, v_block_size = value_cache.shape
        if v_head_dim != head_dim or v_block_size != block_size:
            raise ValueError(
                "value_cache shape must be "
                "[num_blocks, num_kv_heads, head_dim, block_size], "
                f"got {tuple(value_cache.shape)} for block_size={block_size}, "
                f"head_dim={head_dim}"
            )
    # Packed V is a shifted view and may span fewer blocks than K.
    if v_num_blocks > num_blocks:
        raise ValueError(
            f"value_cache must not span more blocks than key_cache, "
            f"got {num_blocks} and {v_num_blocks}"
        )
    if v_num_kv_heads != num_kv_heads:
        raise ValueError(
            "key_cache and value_cache must have the same number of KV heads, "
            f"got {num_kv_heads} and {v_num_kv_heads}"
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads ({num_q_heads}) must be divisible by "
            f"num_kv_heads ({num_kv_heads})"
        )

    if block_tables.dtype != torch.int32:
        raise TypeError(f"block_tables must be int32, got {block_tables.dtype}")
    if context_lengths.dtype != torch.int32:
        raise TypeError(f"context_lengths must be int32, got {context_lengths.dtype}")
    query_group_size = num_q_heads // num_kv_heads
    max_blocks_per_seq = block_tables.shape[1]
    if query.dtype == torch.bfloat16:
        query_dtype = "bf16"
    elif query.dtype == torch.float16:
        query_dtype = "f16"
    else:
        raise TypeError(f"pa_decode only supports f16/bf16 query, got {query.dtype}")
    if is_bf16_kv and query.dtype != torch.bfloat16:
        raise NotImplementedError("BF16 KV requires bfloat16 queries")
    if output.dtype != query.dtype:
        raise TypeError(
            "pa_decode requires output.dtype == query.dtype, "
            f"got {output.dtype} vs {query.dtype}"
        )

    if key_cache.dtype != expected_kv_dtype:
        raise TypeError(
            f"pa_decode requires {expected_kv_dtype} key cache on {arch}, "
            f"got {key_cache.dtype}"
        )
    if value_cache.dtype != expected_kv_dtype:
        raise TypeError(
            f"pa_decode requires {expected_kv_dtype} value cache on {arch}, "
            f"got {value_cache.dtype}"
        )

    if query.stride(2) != 1:
        raise ValueError(
            "pa_decode requires a contiguous query head_dim axis, "
            f"got strides {query.stride()}"
        )
    if output.stride(2) != 1:
        raise ValueError(
            "pa_decode requires a contiguous output head_dim axis, "
            f"got strides {output.stride()}"
        )

    dev = query.device
    if sinks is not None:
        if not isinstance(sinks, torch.Tensor):
            raise TypeError("sinks must be a torch.Tensor")
        if sinks.shape != (num_q_heads,):
            raise ValueError("sinks must have shape [num_query_heads]")
        if sinks.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError("sinks must have dtype bfloat16, float16, or float32")
        if sinks.device != dev:
            raise ValueError("sinks must be on the same device as query")
        if not sinks.is_contiguous():
            raise ValueError("sinks must be contiguous")
    for name, tensor in (
        ("output", output),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_tables", block_tables),
        ("context_lengths", context_lengths),
    ):
        if tensor.device != dev:
            raise ValueError(
                f"{name} must be on the same device as query ({dev}), "
                f"got {tensor.device}"
            )
    for name, tensor in (
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_tables", block_tables),
        ("context_lengths", context_lengths),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def normalize_scale(scale, name):
        if scale is None:
            return torch.ones(1, dtype=torch.float32, device=dev)
        if not isinstance(scale, torch.Tensor):
            return torch.tensor([float(scale)], dtype=torch.float32, device=dev)
        if scale.numel() == 1:
            return scale.reshape(1)
        if scale.dim() == 4:
            if scale.shape[-1] != 1:
                raise ValueError(
                    f"{name} must have a trailing singleton dimension, "
                    f"got shape {tuple(scale.shape)}"
                )
            return scale.squeeze(-1)
        if scale.dim() != 3:
            raise ValueError(
                f"{name} must be scalar or have shape "
                "[num_blocks, num_kv_heads, block_size, 1]"
            )
        return scale

    if is_bf16_kv:
        if key_scale is not None or value_scale is not None:
            raise ValueError(
                "BF16 KV is unscaled; key_scale and value_scale must be None"
            )
        # The native BF16 specialization never dereferences scale pointers.
        # Null pointers avoid allocating dummy scales during graph capture.
        key_scale_t = value_scale_t = None
        per_token_kv = False
        stride_ks_block = stride_ks_head = 0
    else:
        if (key_scale is None) != (value_scale is None):
            raise ValueError(
                "key_scale and value_scale must either both be provided or both be None"
            )
        key_scale_t = normalize_scale(key_scale, "key_scale")
        value_scale_t = normalize_scale(value_scale, "value_scale")
        per_token_kv = key_scale_t.numel() > 1
        if per_token_kv != (value_scale_t.numel() > 1):
            raise ValueError(
                "key_scale and value_scale must both be per-tensor or both be per-token"
            )
        if per_token_kv:
            if key_scale_t.shape != value_scale_t.shape:
                raise ValueError(
                    "key_scale/value_scale shape mismatch: "
                    f"{tuple(key_scale_t.shape)} vs {tuple(value_scale_t.shape)}"
                )
            expected_scale_shape = (num_blocks, num_kv_heads, block_size)
            if key_scale_t.shape != expected_scale_shape:
                raise ValueError(
                    "per-token key_scale/value_scale must be "
                    "[num_blocks, num_kv_heads, block_size] matching the KV cache, "
                    f"got {tuple(key_scale_t.shape)}"
                )
            stride_ks_block = int(key_scale_t.stride(0))
            stride_ks_head = int(key_scale_t.stride(1))
            if key_scale_t.stride(2) != 1:
                raise ValueError(
                    "per-token key_scale token dimension must be contiguous, "
                    f"got strides {key_scale_t.stride()}"
                )
            if value_scale_t.stride() != key_scale_t.stride():
                raise ValueError(
                    "per-token key_scale and value_scale must have matching strides, "
                    f"got {key_scale_t.stride()} vs {value_scale_t.stride()}"
                )
        else:
            stride_ks_block = 0
            stride_ks_head = 0
        for name, scale in (("key_scale", key_scale_t), ("value_scale", value_scale_t)):
            if scale.dtype != torch.float32:
                raise TypeError(f"{name} tensor must be float32, got {scale.dtype}")
            if scale.device != dev:
                raise ValueError(
                    f"{name} tensor must be on the same device as query ({dev}), "
                    f"got {scale.device}"
                )

    schedule_context_length = int(max_blocks_per_seq) * int(block_size)
    if max_context_length is not None:
        if (
            isinstance(max_context_length, bool)
            or not isinstance(max_context_length, int)
            or not 0 <= max_context_length <= max_blocks_per_seq * block_size
        ):
            raise ValueError(
                "max_context_length must be a nonnegative host integer within "
                "block-table capacity"
            )
        schedule_context_length = (
            (max_context_length + block_size - 1) // block_size * block_size
        )
    work_plan.validate(num_seqs, num_kv_heads, dev)
    if work_plan.sliding_window != sliding_window:
        raise ValueError("sliding_window must match work_plan.sliding_window")
    if sliding_window > 0 and work_plan.query_length != query_length:
        raise ValueError("query_length must match work_plan.query_length")
    num_partitions = work_plan.max_partitions
    pmax = max_logits
    psum = exp_sums
    pout = temporary_output

    # Bound byte offsets as well as element indices for 16-bit BF16 caches.
    cache_extent = max(
        key_cache.numel() * key_cache.element_size(),
        value_cache.numel() * value_cache.element_size(),
    )
    wide_kv_addressing = cache_extent >= 2**31
    kv_buffer_u32 = cache_extent < 2**32

    with torch.cuda.device(dev):
        compiled = compile_pa_decode_tile(
            head_dim=head_dim,
            query_group_size=query_group_size,
            block_size=int(block_size),
            num_seqs=num_seqs,
            num_kv_heads=num_kv_heads,
            num_compute_units=torch.cuda.get_device_properties(
                dev
            ).multi_processor_count,
            num_partitions=num_partitions,
            softmax_scale=softmax_scale,
            query_dtype=query_dtype,
            kv_dtype=kv_dtype,
            per_token_kv=per_token_kv,
            query_length=query_length,
            trans_v=trans_v,
            wide_kv_addressing=wide_kv_addressing,
            kv_buffer_u32=kv_buffer_u32,
            work_capacity=work_plan.capacity,
            max_context_length=schedule_context_length,
            sliding_window=sliding_window,
        )

    total_rows = query_length * query_group_size
    expected_scalar_shape = (num_kv_heads, work_plan.capacity, total_rows)
    if pmax is None:
        pmax = torch.empty(*expected_scalar_shape, dtype=torch.float32, device=dev)
    if psum is None:
        psum = torch.empty(*expected_scalar_shape, dtype=torch.float32, device=dev)
    if pout is None:
        pout = torch.empty(
            *expected_scalar_shape, head_dim, dtype=output.dtype, device=dev
        )
    for name, tensor in (
        ("max_logits", pmax),
        ("exp_sums", psum),
        ("temporary_output", pout),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, got {type(tensor).__name__}"
            )
    if pmax.shape != expected_scalar_shape:
        raise ValueError(
            f"max_logits shape {tuple(pmax.shape)} != {expected_scalar_shape}"
        )
    if psum.shape != expected_scalar_shape:
        raise ValueError(
            f"exp_sums shape {tuple(psum.shape)} != {expected_scalar_shape}"
        )
    expected_output_shape = (*expected_scalar_shape, head_dim)
    if pout.shape != expected_output_shape:
        raise ValueError(
            f"temporary_output shape {tuple(pout.shape)} != {expected_output_shape}"
        )
    if pmax.dtype != torch.float32:
        raise TypeError(f"max_logits must be float32, got {pmax.dtype}")
    if psum.dtype != torch.float32:
        raise TypeError(f"exp_sums must be float32, got {psum.dtype}")
    if pout.dtype != output.dtype:
        raise TypeError(
            f"temporary_output dtype {pout.dtype} must match output dtype "
            f"{output.dtype}"
        )
    for name, tensor in (
        ("max_logits", pmax),
        ("exp_sums", psum),
        ("temporary_output", pout),
    ):
        if tensor.device != dev:
            raise ValueError(
                f"{name} must be on the same device as query ({dev}), "
                f"got {tensor.device}"
            )
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    with torch.cuda.device(dev):
        s = torch.cuda.current_stream(dev)
        _run_compiled(
            compiled["launch"],
            ptr_arg(pmax, fx.Float32),
            ptr_arg(psum, fx.Float32),
            ptr_arg(pout, _flydsl_pointer_dtype(pout.dtype)),
            ptr_arg(query, _flydsl_pointer_dtype(query.dtype)),
            ptr_arg(key_cache, _flydsl_pointer_dtype(key_cache.dtype)),
            ptr_arg(value_cache, _flydsl_pointer_dtype(value_cache.dtype)),
            ptr_arg(block_tables, fx.Int32),
            (
                flyc.from_c_void_p(fx.Float32, 0)
                if is_bf16_kv
                else ptr_arg(key_scale_t, fx.Float32)
            ),
            (
                flyc.from_c_void_p(fx.Float32, 0)
                if is_bf16_kv
                else ptr_arg(value_scale_t, fx.Float32)
            ),
            int(max_blocks_per_seq),
            int(num_seqs),
            int(num_kv_heads),
            stride_ks_block,
            stride_ks_head,
            int(query.stride(0)),
            int(query.stride(1)),
            ptr_arg(work_plan.work_info, fx.Int32),
            work_plan.capacity,
            s,
        )
        output_5d = output.reshape(
            num_seqs, query_length, num_kv_heads, query_group_size, head_dim
        )
        reduce_partitions = num_partitions
        if sliding_window > 0:
            # Bound the unaligned MTP window without shrinking the reusable plan.
            window_tiles = (
                sliding_window + query_length - 2 + KV_COMPUTE_BLOCK - 1
            ) // KV_COMPUTE_BLOCK + 1
            reduce_partitions = min(reduce_partitions, window_tiles)
        launch_pa_decode_ps_reduce(
            output_5d,
            psum,
            pmax,
            pout,
            sinks,
            output_5d.stride(0),
            output_5d.stride(1),
            output_5d.stride(2),
            output_5d.stride(3),
            pmax.stride(0),
            pmax.stride(1),
            pout.stride(0),
            pout.stride(1),
            pout.stride(2),
            query_seq_len=query_length,
            query_group_size=query_group_size,
            head_size=head_dim,
            context_partition_num=reduce_partitions,
            stream=s,
            reduce_info=work_plan.reduce_info,
        )


class _PADecodeAutotuneResources:
    """Prepare candidate plans and graphs outside FlyDSL's timed replay."""

    def __init__(self, query, args, options, max_partitions):
        self.query = query
        self.args = args
        self.options = options
        self.max_partitions = max_partitions
        self.outputs = {}
        self.graphs = {}
        self.plans = {}
        self.workspaces = {}
        self.reference = None

    def prepare(self, workgroup_budget):
        if workgroup_budget in self.graphs:
            return
        key, value, lengths, table, scale, query_length = self.args
        if key.dtype != torch.bfloat16:
            if (self.options["key_scale"] is None) != (
                self.options["value_scale"] is None
            ):
                raise ValueError(
                    "key_scale and value_scale must either both be provided or both be None"
                )
            for name in ("key_scale", "value_scale"):
                kv_scale = self.options[name]
                if not isinstance(kv_scale, torch.Tensor):
                    # Scalar host-to-device copies must precede graph capture.
                    self.options[name] = torch.tensor(
                        [1.0 if kv_scale is None else float(kv_scale)],
                        dtype=torch.float32,
                        device=self.query.device,
                    )
        plan = plan_pa_decode(
            lengths,
            key.shape[1],
            max_partitions=self.max_partitions,
            workgroup_budget=workgroup_budget,
            sliding_window=self.options["sliding_window"],
            query_length=query_length,
        )
        rows = query_length * self.query.shape[1] // key.shape[1]
        scalar_shape = (key.shape[1], plan.capacity, rows)
        output = torch.full_like(
            self.query, float("nan"), memory_format=torch.contiguous_format
        )
        workspace = {
            "exp_sums": torch.empty(
                scalar_shape, dtype=torch.float32, device=output.device
            ),
            "max_logits": torch.empty(
                scalar_shape, dtype=torch.float32, device=output.device
            ),
            "temporary_output": torch.empty(
                (*scalar_shape, self.query.shape[-1]),
                dtype=output.dtype,
                device=output.device,
            ),
        }

        def launch():
            pa_decode(
                output,
                self.query,
                key,
                value,
                lengths,
                table,
                scale,
                query_length,
                work_plan=plan,
                **self.options,
                **workspace,
            )

        # Eager compilation and graph capture precede native autotune timing.
        stream = torch.cuda.current_stream()
        capture_stream = torch.cuda.Stream(device=output.device)
        capture_stream.wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            launch()
            with torch.cuda.graph(graph, stream=capture_stream):
                launch()
        stream.wait_stream(capture_stream)
        graph.replay()
        stream.synchronize()
        if not torch.isfinite(output).all().item():
            raise ArithmeticError("PA autotune candidate produced nonfinite output")
        self.outputs[workgroup_budget] = output
        self.graphs[workgroup_budget] = graph
        # Graph replay uses the original pointers, so retain all backing tensors.
        self.plans[workgroup_budget] = plan
        self.workspaces[workgroup_budget] = workspace
        if self.reference is None:
            self.reference = output.clone()
        else:
            torch.testing.assert_close(output, self.reference, atol=5e-3, rtol=5e-3)


def _pa_decode_configs(*args, resources, num_cu, **kwargs):
    # Keep a validated 2*CU baseline; deduplicate identical plan capacities.
    baseline = 2 * num_cu
    resources.prepare(baseline)
    batch = resources.args[2].numel()
    kv_heads = resources.args[0].shape[1]
    capacities = {}
    for budget in [baseline, 128, 256, 512, 1024, 2048, 4096]:
        capacity = min(
            batch * resources.max_partitions,
            max(batch, (budget + kv_heads - 1) // kv_heads),
        )
        capacities.setdefault(capacity, Config(workgroup_budget=budget))
    return list(capacities.values())


def _pa_decode_default(*args, num_cu, **kwargs):
    return Config(workgroup_budget=2 * num_cu)


@contextmanager
def _validate_pa_decode_config(arguments):
    resources = arguments["resources"]
    budget = arguments["workgroup_budget"]
    resources.prepare(budget)
    resources.outputs[budget].fill_(float("nan"))
    yield
    torch.cuda.current_stream().synchronize()
    torch.testing.assert_close(
        resources.outputs[budget], resources.reference, atol=5e-3, rtol=5e-3
    )


@autotune(
    configs=_pa_decode_configs,
    key=[
        "num_seqs",
        "num_kv_heads",
        "query_group_size",
        "head_dim",
        "block_size",
        "query_length",
        "query_dtype",
        "kv_dtype",
        "per_token_kv",
        "trans_v",
        "sink_dtype",
        "sliding_window",
        "max_partitions",
        "num_cu",
        "implementation",
    ],
    default=_pa_decode_default,
    validate_hook=_validate_pa_decode_config,
)
def _pa_decode_autotuner(
    num_seqs,
    num_kv_heads,
    query_group_size,
    head_dim,
    block_size,
    query_length,
    query_dtype,
    kv_dtype,
    per_token_kv,
    trans_v,
    sink_dtype,
    sliding_window,
    max_partitions,
    num_cu,
    implementation,
    resources,
    *,
    workgroup_budget,
):
    resources.graphs[workgroup_budget].replay()


def prepare_pa_decode_plan(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    query_length: int,
    *,
    compute_type: torch.dtype = torch.bfloat16,
    key_scale: torch.Tensor | float | None = None,
    value_scale: torch.Tensor | float | None = None,
    sinks: torch.Tensor | None = None,
    sliding_window: int = 0,
    max_partitions: int | None = None,
    max_context_length: int | None = None,
) -> PADecodePlan:
    """Select a work budget using native FlyDSL autotune, then build a plan.

    Call outside graph capture with the actual decode tensors/options. Native
    cache hits and misses with the 2*CU default do not launch candidates.
    Set FLYDSL_AUTOTUNE=1 to prepare candidate workspaces/graphs and search with
    FlyDSL's default benchmark and fastest-config selection. Candidates must
    agree with the 2*CU baseline; independent FP32 accuracy is covered by tests.
    Set FLYDSL_AUTOTUNE_CACHE_DIR before importing this module to choose the
    native cache directory.

    Cache by attention geometry and execution modes. Tensor strides, total KV
    cache pages, block-table padding, and max_context_length do not partition
    the tuning cache; max_context_length still bounds kernel scheduling.

    Retain the returned plan for decode/graph replay. Refresh it with
    plan_pa_decode(..., plan=plan) when context lengths change; decode never
    invokes autotune. Workspace sizes follow the returned plan.capacity.
    """
    with torch.cuda.device(query.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prepare_pa_decode_plan must run before graph capture")
        num_cu = torch.cuda.get_device_properties(query.device).multi_processor_count
        limit = num_cu if max_partitions is None else max_partitions
        # Validate plan geometry even when native autotune takes the default.
        default_plan = plan_pa_decode(
            context_lengths,
            key_cache.shape[1],
            max_partitions=limit,
            sliding_window=sliding_window,
            query_length=query_length,
        )
        options = {
            "compute_type": compute_type,
            "key_scale": key_scale,
            "value_scale": value_scale,
            "sinks": sinks,
            "sliding_window": default_plan.sliding_window,
            "max_context_length": max_context_length,
        }
        resources = _PADecodeAutotuneResources(
            query,
            (
                key_cache,
                value_cache,
                context_lengths,
                block_tables,
                softmax_scale,
                query_length,
            ),
            options,
            limit,
        )

        config = _pa_decode_autotuner.resolve_config(
            num_seqs=context_lengths.numel(),
            num_kv_heads=key_cache.shape[1],
            query_group_size=query.shape[1] // key_cache.shape[1],
            head_dim=query.shape[-1],
            block_size=key_cache.shape[3],
            query_length=query_length,
            query_dtype=str(query.dtype),
            kv_dtype=str(key_cache.dtype),
            per_token_kv=(
                isinstance(key_scale, torch.Tensor) and key_scale.numel() > 1
            ),
            trans_v=value_cache.ndim == 5,
            sink_dtype=None if sinks is None else str(sinks.dtype),
            sliding_window=default_plan.sliding_window,
            max_partitions=limit,
            num_cu=num_cu,
            implementation=implementation_cache_tag(),
            resources=resources,
        )
        budget = config.kwargs.get("workgroup_budget")
        if (
            set(config.all_kwargs()) != {"workgroup_budget"}
            or type(budget) is not int
            or budget < 1
            or config.compiler_opts()
            or config.pre_hook is not None
        ):
            raise ValueError("Invalid PA workgroup budget in FlyDSL autotune cache")
        if budget == 2 * num_cu:
            return default_plan
        return plan_pa_decode(
            context_lengths,
            key_cache.shape[1],
            max_partitions=limit,
            workgroup_budget=budget,
            sliding_window=default_plan.sliding_window,
            query_length=query_length,
        )

# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Paged-attention decode with BF16/FP16 queries on gfx942/gfx950.

Cache layouts are logical, not preshuffled. FP8 K/V use E4M3 FNUZ on gfx942
and OCP on gfx950. The supported BF16 K/V path uses native BF16 page-64
vectorized caches. See ``kernels.pa_decode_kernel`` for Q/P and MFMA details.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.pa_decode_common import pa_decode_pointer_dtype as _flydsl_pointer_dtype
from .kernels.pa_decode_kernel import (
    KV_COMPUTE_BLOCK,
    compile_pa_decode_fp8_small,
    compile_pa_decode_fp8_wave,
    compile_pa_decode_tile,
)
from .kernels.pa_decode_plan import PADecodePlan
from .kernels.pa_decode_plan import plan_pa_decode as plan_pa_decode  # noqa: PLC0414
from .kernels.pa_decode_reduce import (
    MAX_CONTEXT_PARTITIONS,
    launch_pa_decode_ps_reduce,
)
from .kernels.tensor_shim import _run_compiled, get_dtype_str, ptr_arg


def get_recommended_splits(
    num_sequences: int,
    num_kv_heads: int,
    split_kv_blocks: int = 1,
    max_partitions: int | None = None,
    *,
    max_context_length: int | None = None,
) -> int:
    """Recommend a uniform split count for scratch allocation and ``pa_decode``.

    Without ``max_context_length``, the default cap is eight. A host length
    hint targets two CTAs per CU, bounded by 256-token tiles and the reducer
    limit; short contexts and large grids retain the legacy recommendation.
    ``max_partitions`` caps either mode. Allocate scratch and call ``pa_decode``
    with the returned count for every sequence; no GPU lengths are read back.
    """
    if max_context_length is not None and max_context_length < 0:
        raise ValueError("max_context_length must be non-negative")
    if max_partitions is None:
        max_partitions = 8 if max_context_length is None else MAX_CONTEXT_PARTITIONS
    if not 4 <= max_partitions <= MAX_CONTEXT_PARTITIONS:
        raise ValueError(
            f"max_partitions must be in [4, {MAX_CONTEXT_PARTITIONS}], "
            f"got {max_partitions}"
        )
    props = torch.cuda.get_device_properties(torch.device("cuda"))
    num_sm = props.multi_processor_count * 2
    denom = max(1, num_sequences * num_kv_heads * split_kv_blocks)
    n = ((num_sm + denom - 1) // denom) * split_kv_blocks
    if max_context_length is not None:
        legacy = max(4, min(n, 8))
        # Count compute tiles, not pages; the floor preserves short-grid tuning.
        context_tiles = (max_context_length + KV_COMPUTE_BLOCK - 1) // KV_COMPUTE_BLOCK
        work_limit = max(8, context_tiles)
        sequence_heads = max(1, num_sequences * num_kv_heads)
        occupancy_limit = (num_sm + sequence_heads - 1) // sequence_heads
        n = max(legacy, min(work_limit, occupancy_limit))
    return max(4, min(n, max_partitions))


def _pa_decode_fp8_qlen8(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    num_partitions: int,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    max_logits: torch.Tensor | None = None,
    exp_sums: torch.Tensor | None = None,
    temporary_output: torch.Tensor | None = None,
) -> None:
    """Write [B*8, 16, 128] BF16 output from page64 vectorized FP8 KV."""
    if get_gfx_runtime() != "gfx950":
        raise NotImplementedError("FP8 Qlen8 decode requires gfx950")
    if context_lengths.ndim != 1 or context_lengths.numel() < 1:
        raise ValueError("context_lengths must be a nonempty [B] vector")
    batch = context_lengths.numel()
    head_dim = query.shape[-1]
    expected_query = (batch * 8, 16, head_dim)
    if (
        head_dim not in (128, 192)
        or query.shape != expected_query
        or query.dtype != torch.bfloat16
        or query.stride(-1) != 1
    ):
        raise ValueError(
            "FP8 Qlen8 requires BF16 query [B*8, 16, Dqk=128/192] "
            f"with a contiguous head axis, got {query.shape}/{query.dtype}"
        )
    if (
        output.shape != (batch * 8, 16, 128)
        or output.dtype != torch.bfloat16
        or output.stride(-1) != 1
    ):
        raise ValueError("FP8 Qlen8 requires BF16 output [B*8, 16, 128]")
    if (
        key_cache.ndim != 5
        or key_cache.shape[1:] != (1, head_dim // 16, 64, 16)
        or key_cache.dtype != torch.float8_e4m3fn
        or not key_cache.is_contiguous()
    ):
        raise ValueError("FP8 Qlen8 requires vectorized page64 FP8 K")
    if (
        value_cache.shape != (key_cache.shape[0], 1, 4, 128, 16)
        or value_cache.dtype != key_cache.dtype
        or not value_cache.is_contiguous()
    ):
        raise ValueError("FP8 Qlen8 requires vectorized page64 FP8 V128")
    if (
        block_tables.shape[0] != batch
        or block_tables.ndim != 2
        or block_tables.dtype != torch.int32
        or not block_tables.is_contiguous()
    ):
        raise ValueError("block_tables must be contiguous int32 [B, max_pages]")
    if context_lengths.dtype != torch.int32 or not context_lengths.is_contiguous():
        raise ValueError("context_lengths must be contiguous int32 [B]")
    if not isinstance(num_partitions, int) or not 1 <= num_partitions <= 64:
        raise ValueError("num_partitions must be in [1, 64]")
    if not 0 < softmax_scale < float("inf"):
        raise ValueError("softmax_scale must be finite and positive")
    device = query.device
    if device.type != "cuda" or any(
        tensor.device != device
        for tensor in (
            output,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
        )
    ):
        raise ValueError("all FP8 Qlen8 decode tensors must be on the query device")
    for name, scale in (("key_scale", key_scale), ("value_scale", value_scale)):
        if (
            not isinstance(scale, torch.Tensor)
            or scale.shape != (1,)
            or scale.dtype != torch.float32
            or scale.device != device
            or not scale.is_contiguous()
        ):
            raise ValueError(f"{name} must be a contiguous float32 GPU scalar")

    scalar_shape = (batch, 1, num_partitions, 128)
    if num_partitions == 1:
        max_logits = key_scale if max_logits is None else max_logits
        exp_sums = key_scale if exp_sums is None else exp_sums
        temporary_output = output if temporary_output is None else temporary_output
    else:
        supplied = (max_logits, exp_sums, temporary_output)
        if any(tensor is None for tensor in supplied):
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("preallocate FP8 Qlen8 partials before graph capture")
            if any(tensor is not None for tensor in supplied):
                raise ValueError("supply all partial buffers, or none")
            max_logits = torch.empty(scalar_shape, dtype=torch.float32, device=device)
            exp_sums = torch.empty_like(max_logits)
            temporary_output = torch.empty(
                (*scalar_shape, 128), dtype=torch.bfloat16, device=device
            )
        if (
            max_logits.shape != scalar_shape
            or exp_sums.shape != scalar_shape
            or temporary_output.shape != (*scalar_shape, 128)
            or max_logits.dtype != torch.float32
            or exp_sums.dtype != torch.float32
            or temporary_output.dtype != torch.bfloat16
            or any(
                not tensor.is_contiguous() or tensor.device != device
                for tensor in (max_logits, exp_sums, temporary_output)
            )
        ):
            raise ValueError(
                "FP8 Qlen8 partial buffers have incompatible shape or dtype"
            )

    compile_kernel = (
        compile_pa_decode_fp8_small
        if batch * num_partitions <= 64
        else compile_pa_decode_fp8_wave
    )
    compiled = compile_kernel(head_dim, num_partitions, softmax_scale)
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        _run_compiled(
            compiled["launch"],
            output,
            max_logits.view(-1),
            exp_sums.view(-1),
            temporary_output.view(-1),
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale,
            value_scale,
            int(block_tables.shape[1]),
            batch,
            1,
            0,
            0,
            int(query.stride(0)),
            int(query.stride(1)),
            stream,
        )
        if num_partitions > 1:
            output_5d = output.view(batch, 8, 1, 16, 128)
            launch_pa_decode_ps_reduce(
                output_5d,
                exp_sums,
                max_logits,
                temporary_output,
                None,
                output_5d.stride(0),
                output_5d.stride(1),
                output_5d.stride(2),
                output_5d.stride(3),
                exp_sums.stride(0),
                exp_sums.stride(1),
                exp_sums.stride(2),
                temporary_output.stride(0),
                temporary_output.stride(1),
                temporary_output.stride(2),
                temporary_output.stride(3),
                query_seq_len=8,
                query_group_size=16,
                head_size=128,
                context_partition_num=num_partitions,
                stream=stream,
            )


def pa_decode(
    output: torch.Tensor,  # [num_seqs * query_length, num_query_heads, value_head_size]
    query: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, value_head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, value_head_size, x]
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    softmax_scale: float,
    query_length: int,
    max_context_partition_num: int,
    context_partition_size: int = 256,
    compute_type: torch.dtype = torch.bfloat16,
    query_scale: torch.Tensor = None,
    key_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    exp_sums: torch.Tensor = None,  # [num_seqs, num_kv_heads, max_context_partition_num, query_length * query_group_size]
    max_logits: torch.Tensor = None,  # [num_seqs, num_kv_heads, max_context_partition_num, query_length * query_group_size]
    temporary_output: torch.Tensor = None,  # [num_seqs, num_kv_heads, max_context_partition_num, query_length * query_group_size, value_head_size]
    alibi_slopes: torch.Tensor = None,
    sinks: torch.Tensor = None,
    sliding_window: int = 0,
    work_plan: PADecodePlan | None = None,
) -> None:
    """Decode FP8 or BF16 K/V using 256-token compute tiles.

    Supports page sizes 16/64/128 and head_dim 64 or multiples of 128 up to 1024.
    gfx950 Qlen8/GQA16/page64 full attention also supports Dqk192/V128 via
    the optimized FP8 wave schedule.
    Planned gfx950 Qlen8/GQA16/page64 W128 also supports FP8 Dqk192/V128
    with optional sinks and scalar K/V scales.
    Native BF16 K/V supports BF16 Qlen8/GQA16/page64/D128 with W1024,
    vectorized-5D caches and a refreshed work plan. FP8 K/V scales are [1]
    or [num_blocks, num_kv_heads, block_size, 1]; BF16 K/V is unscaled.
    ALiBi and externally quantized queries are unsupported.

    MTP lengths include the query tokens and use dense causal masking.
    Independently selected sparse queries need separate table rows and
    query_length=1. Positive ``sliding_window`` requires ``work_plan`` and
    includes each query's own position; 0 and -1 disable it.

    ``sinks`` is a contiguous [num_query_heads] BF16/FP16/FP32 tensor on the
    query device: unscaled zero-value logits shared across batch/MTP positions.
    Each contributes once, independently of the window; -inf disables it and
    +inf suppresses the head's output.

    Refresh ``work_plan`` on the current stream after changing lengths. Its
    max_partitions must equal max_context_partition_num, bounded by device CUs
    (static limit: 256). Match sliding_window and, for windowed plans,
    query_length. Planned scratch is [KV heads, capacity, query rows (, D)];
    static scratch uses the per-sequence layouts shown in the signature.

    GPU lengths and table entries are not checked. Callers must ensure
    0 <= context_lengths[i] <= block_tables.shape[1] * block_size and used
    block indices in [0, min(key_cache.shape[0], value_cache.shape[0])).
    """
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
    if sliding_window > 0 and work_plan is None:
        raise ValueError("positive sliding_window requires work_plan")
    if not isinstance(query_length, int):
        raise TypeError("query_length must be an int")
    if query_length < 1:
        raise ValueError(f"query_length must be positive, got {query_length}")
    if (
        work_plan is None
        and not 1 <= max_context_partition_num <= MAX_CONTEXT_PARTITIONS
    ):
        raise ValueError(
            f"max_context_partition_num must be in [1, {MAX_CONTEXT_PARTITIONS}], "
            f"got {max_context_partition_num}"
        )

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

    # Use the Qlen8 wave kernels for this full-attention geometry. Keep SWA,
    # sinks, per-token scales, and other shapes on the existing dispatcher.
    if (
        query_length == 8
        and sliding_window == 0
        and sinks is None
        and work_plan is None
        and num_q_heads == 16
        and head_dim in (128, 192)
        and output.shape == (num_seqs * 8, 16, 128)
        and key_cache.shape[1] == 1
        and key_cache.shape[-2:] == (64, 16)
        and key_cache.dtype == torch.float8_e4m3fn
        and value_cache.shape == (key_cache.shape[0], 1, 4, 128, 16)
        and compute_type == key_cache.dtype
        and isinstance(key_scale, torch.Tensor)
        and isinstance(value_scale, torch.Tensor)
        and key_scale.numel() == value_scale.numel() == 1
        and 1 <= max_context_partition_num <= 64
    ):
        return _pa_decode_fp8_qlen8(
            output,
            query,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
            softmax_scale,
            max_context_partition_num,
            key_scale,
            value_scale,
            max_logits,
            exp_sums,
            temporary_output,
        )

    if output.shape[:2] != query.shape[:2]:
        raise ValueError(
            f"output token/head shape {tuple(output.shape[:2])} must match "
            f"query token/head shape {tuple(query.shape[:2])}"
        )
    value_dim = output.shape[-1]
    asymmetric_value = value_dim != head_dim
    if block_tables.shape[0] != num_seqs:
        raise ValueError(
            f"block_tables.shape[0] ({block_tables.shape[0]}) must match "
            f"context_lengths.shape[0] ({num_seqs})"
        )

    num_blocks, num_kv_heads, num_hgroups, block_size, hgroup_width = key_cache.shape
    is_bf16_kv = key_cache.dtype == torch.bfloat16
    kv_vector_width = 8 if is_bf16_kv else 16
    if num_kv_heads < 1:
        raise ValueError(
            f"key_cache must contain at least one KV head, got {num_kv_heads}"
        )

    # Enforce tile Q-load constraints and the reducer's 1024-thread limit,
    # including for direct output.
    q_chunk = head_dim // 16
    if not (
        64 <= head_dim <= 1024
        and head_dim % 64 == 0
        and (q_chunk <= 8 or q_chunk % 8 == 0 or (asymmetric_value and q_chunk == 12))
    ):
        raise NotImplementedError(
            f"pa_decode does not support head_dim={head_dim}; supported values "
            "are 64, multiples of 128 in [128, 1024], and qualified asymmetric D192/V128"
        )
    if num_hgroups != head_dim // kv_vector_width or hgroup_width != kv_vector_width:
        raise ValueError(
            "key_cache shape must be "
            "[num_blocks, num_kv_heads, head_dim // vector_width, block_size, vector_width], "
            f"got {tuple(key_cache.shape)} for head_dim={head_dim}"
        )
    if block_size not in (16, 64, 128):
        raise ValueError(
            f"pa_decode only supports block_size in (16, 64, 128), got {block_size}"
        )

    trans_v = value_cache.dim() == 5
    if trans_v:
        v_num_blocks, v_num_kv_heads = value_cache.shape[:2]
        expected_v_tail = (block_size // kv_vector_width, value_dim, kv_vector_width)
        if tuple(value_cache.shape[2:]) != expected_v_tail:
            raise ValueError(
                "transposed value_cache shape must be "
                "[num_blocks, num_kv_heads, block_size // vector_width, value_dim, vector_width], "
                f"got {tuple(value_cache.shape)} for block_size={block_size}, "
                f"value_dim={value_dim}"
            )
    else:
        if is_bf16_kv:
            raise ValueError("BF16 KV requires vectorized-5D value_cache")
        v_num_blocks, v_num_kv_heads, v_head_dim, v_block_size = value_cache.shape
        if v_head_dim != value_dim or v_block_size != block_size:
            raise ValueError(
                "value_cache shape must be "
                "[num_blocks, num_kv_heads, value_dim, block_size], "
                f"got {tuple(value_cache.shape)} for block_size={block_size}, "
                f"value_dim={value_dim}"
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

    arch = get_gfx_runtime()
    expected_fp8_dtype = {
        "gfx942": torch.float8_e4m3fnuz,
        "gfx950": torch.float8_e4m3fn,
    }.get(arch)
    if expected_fp8_dtype is None:
        raise NotImplementedError(
            f"pa_decode only supports gfx942 and gfx950, got {arch}"
        )
    required_compute_type = torch.bfloat16 if is_bf16_kv else expected_fp8_dtype
    if compute_type != required_compute_type:
        raise NotImplementedError(
            f"pa_decode requires {required_compute_type} compute for this KV cache on {arch}, "
            f"got {compute_type}"
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
        raise TypeError("BF16 KV requires a BF16 query")
    if output.dtype != query.dtype:
        raise TypeError(
            "pa_decode requires output.dtype == query.dtype, "
            f"got {output.dtype} vs {query.dtype}"
        )

    required_cache_dtype = torch.bfloat16 if is_bf16_kv else expected_fp8_dtype
    if key_cache.dtype != required_cache_dtype:
        raise TypeError(
            f"pa_decode requires {required_cache_dtype} key cache on {arch}, "
            f"got {key_cache.dtype}"
        )
    if value_cache.dtype != required_cache_dtype:
        raise TypeError(
            f"pa_decode requires {required_cache_dtype} value cache on {arch}, "
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
        # The scale pointer slots are unused by the BF16 specialization.
        key_scale_t = value_scale_t = query
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

    num_partitions = max_context_partition_num
    if work_plan is not None:
        if not isinstance(work_plan, PADecodePlan):
            raise TypeError("work_plan must be a PADecodePlan")
        work_plan.validate(num_seqs, num_kv_heads, dev)
        if work_plan.max_partitions != num_partitions:
            raise ValueError(
                "max_context_partition_num must match work_plan.max_partitions"
            )
        if work_plan.sliding_window != sliding_window:
            raise ValueError("sliding_window must match work_plan.sliding_window")
        if sliding_window > 0 and work_plan.query_length != query_length:
            raise ValueError("query_length must match work_plan.query_length")
    if asymmetric_value and not (
        arch == "gfx950"
        and query.dtype == torch.bfloat16
        and key_cache.dtype == torch.float8_e4m3fn
        and query_length == 8
        and sliding_window == 128
        and head_dim == 192
        and value_dim == 128
        and block_size == 64
        and query_group_size == 16
        and num_kv_heads == 1
        and trans_v
        and work_plan is not None
        and not per_token_kv
    ):
        raise NotImplementedError(
            "asymmetric value width requires gfx950 BF16 Q/O, FP8 K/V, "
            "Qlen8, GQA16, one KV head, page64, Q/K192-V128, W128, "
            "vectorized V, scalar scales, and a work plan"
        )
    if is_bf16_kv and (
        query_length != 8
        or sliding_window != 1024
        or head_dim != 128
        or block_size != 64
        or query_group_size != 16
        or not trans_v
    ):
        raise NotImplementedError(
            "BF16 KV currently supports BF16 Qlen8, GQA16, page64, "
            "D128/V128, and sliding_window=1024; got "
            f"qlen={query_length}, window={sliding_window}, D={head_dim}, "
            f"page={block_size}, GQA={query_group_size}, trans_v={trans_v}"
        )
    pmax = max_logits
    psum = exp_sums
    pout = temporary_output

    # Widen before either cache's i32 byte offsets can wrap.
    wide_kv_addressing = (
        max(
            key_cache.numel() * key_cache.element_size(),
            value_cache.numel() * value_cache.element_size(),
        )
        >= 2**31
    )

    # Add sinks once: here for static NP=1, otherwise in reduction.
    use_direct_sinks = sinks is not None and num_partitions == 1 and work_plan is None
    with torch.cuda.device(dev):
        compiled = compile_pa_decode_tile(
            head_dim=head_dim,
            value_dim=value_dim,
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
            kv_dtype="bf16" if is_bf16_kv else "fp8",
            per_token_kv=per_token_kv,
            query_length=query_length,
            trans_v=trans_v,
            wide_kv_addressing=wide_kv_addressing,
            use_work_plan=work_plan is not None,
            work_capacity=work_plan.capacity if work_plan is not None else None,
            sliding_window=sliding_window,
            use_sinks=use_direct_sinks,
            sink_dtype_str=get_dtype_str(sinks.dtype) if use_direct_sinks else "f32",
        )

    if num_partitions == 1 and work_plan is None:
        # Direct output ignores caller scratch; supply unused kernel arguments.
        dummy = torch.empty(1, dtype=torch.float32, device=dev)
        pmax = psum = pout = dummy
    else:
        total_rows = query_length * query_group_size
        expected_scalar_shape = (num_seqs, num_kv_heads, num_partitions, total_rows)
        if work_plan is not None:
            expected_scalar_shape = (num_kv_heads, work_plan.capacity, total_rows)
        if pmax is None or psum is None or pout is None:
            if pmax is None:
                pmax = torch.empty(
                    *expected_scalar_shape, dtype=torch.float32, device=dev
                )
            if psum is None:
                psum = torch.empty(
                    *expected_scalar_shape, dtype=torch.float32, device=dev
                )
            if pout is None:
                pout = torch.empty(
                    *expected_scalar_shape, value_dim, dtype=output.dtype, device=dev
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
        expected_output_shape = (*expected_scalar_shape, value_dim)
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
            ptr_arg(output, _flydsl_pointer_dtype(output.dtype)),
            ptr_arg(pmax, fx.Float32),
            ptr_arg(psum, fx.Float32),
            ptr_arg(pout, _flydsl_pointer_dtype(pout.dtype)),
            ptr_arg(query, _flydsl_pointer_dtype(query.dtype)),
            ptr_arg(key_cache, _flydsl_pointer_dtype(key_cache.dtype)),
            ptr_arg(value_cache, _flydsl_pointer_dtype(value_cache.dtype)),
            ptr_arg(block_tables, fx.Int32),
            ptr_arg(context_lengths, fx.Int32),
            ptr_arg(key_scale_t, fx.Float32),
            ptr_arg(value_scale_t, fx.Float32),
            (
                ptr_arg(sinks, _flydsl_pointer_dtype(sinks.dtype))
                if use_direct_sinks
                else flyc.from_c_void_p(fx.Float32, 0)
            ),
            int(max_blocks_per_seq),
            int(num_seqs),
            int(num_kv_heads),
            stride_ks_block,
            stride_ks_head,
            int(output.stride(0)),
            int(output.stride(1)),
            int(query.stride(0)),
            int(query.stride(1)),
            (
                ptr_arg(work_plan.work_info, fx.Int32)
                if work_plan is not None
                else flyc.from_c_void_p(fx.Int32, 0)
            ),
            work_plan.capacity if work_plan is not None else 0,
            s,
        )
        if num_partitions > 1 or work_plan is not None:
            output_5d = output.reshape(
                num_seqs, query_length, num_kv_heads, query_group_size, value_dim
            )
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
                pmax.stride(0) if work_plan is None else 0,
                pmax.stride(1) if work_plan is None else pmax.stride(0),
                pmax.stride(2) if work_plan is None else pmax.stride(1),
                pout.stride(0) if work_plan is None else 0,
                pout.stride(1) if work_plan is None else pout.stride(0),
                pout.stride(2) if work_plan is None else pout.stride(1),
                pout.stride(3) if work_plan is None else pout.stride(2),
                query_seq_len=query_length,
                query_group_size=query_group_size,
                head_size=value_dim,
                context_partition_num=num_partitions,
                stream=s,
                reduce_info=work_plan.reduce_info if work_plan is not None else None,
            )

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
import argparse
import random
from typing import List, Optional, Tuple, Union, Dict
import hashlib
import pandas as pd
import numpy as np
import pytest
import torch
import triton
import aiter
from aiter import dtypes
from aiter import pertoken_quant, per_tensor_quant
from aiter.test_common import benchmark, checkAllclose, perftest
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.attention import pa_decode_gluon
from aiter.ops.triton.gluon.pa_decode_gluon import (
    get_recommended_splits,
)
from aiter.ops.triton.gluon.pa_decode_flydsl import (
    pa_decode_flydsl,
)
from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot import (
    pa_decode_gluon_aot,
)
from csrc.cpp_itfs.pa_gluon_aot.pa_decode_gluon_aot_prebuild import (
    prebuild_normal_accuracy_cases_aot_so,
    prebuild_normal_performance_cases_aot_so,
    get_so_files_size_and_count,
)

try:
    from triton.experimental import gluon  # noqa: F401
    from triton.experimental.gluon import language as gl  # noqa: F401
except ImportError:
    print(
        "Warning: triton.experimental.gluon or triton.experimental.gluon.language not exists, only pa_decode_gluon_aot can be used!"
    )
    pa_decode_gluon = None
    pa_decode_flydsl = None


TRITON_VERSION = triton.__version__
TEST_NAME = "main.normal_accuracy_performance.jit"

# Global variables that will be set by command line arguments
USE_TORCH_FLASH_REF = True

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# Global configuration
UNIFORM_RANGE = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
}

# Test configuration parameters
USE_TORCH_FLASH_REF_OPTIONS = [True]
USE_AOT_IMPL_OPTIONS = [True, False]
KV_VARLEN_OPTIONS = [False, True]
TRANS_V_OPTIONS = [False, True]
QUANT_Q_AND_KV_OPTIONS = [[True, True]]
CONTEXT_PARTITION_SIZE_OPTIONS = [256]
COMPUTE_TYPE_OPTIONS = ["fp8", "bf16", "fp16"]
QUANT_MODE_OPTIONS = ["per_token", "per_tensor"]
HEAD_DIMENSION_OPTIONS = [128]
BLOCK_SIZE_OPTIONS = [16, 64, 1024]
HEAD_CONFIGURATIONS = [(5, 1), (8, 1), (10, 1), (16, 1), (64, 4)]
QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
CONTEXT_LENGTH_OPTIONS = [512, 4096, 4097]
BATCH_SIZE_OPTIONS = [4, 80, 128]
BATCH_SIZE_OPTIONS = [4, 80, 128]
SINKS_OPTIONS = [True, False]
SLIDING_WINDOW_OPTIONS = [0, 128]
COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = []
PS_OPTIONS = [True, False]

CASE_SET_NAME_OPTIONS = [
    "normal_accuracy",
    "normal_accuracy_aot",
    "sliding_window_accuracy",
    "sliding_window_performance",
    "normal_performance",
    "normal_performance_aot",
]


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def compare_arrays(
    arr1: np.ndarray,
    arr2: np.ndarray,
    k: int = 5,
    thresholds: List[float] = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1],
) -> Dict:
    """
    Compare two numpy arrays and compute various difference metrics.

    Args:
        arr1: First input array (float32)
        arr2: Second input array (float32)
        k: Number of top differences to return
        thresholds: List of thresholds for difference magnitude analysis

    Returns:
        Dictionary containing:
        - top_k_diff: Top k absolute differences with their positions
        - threshold_stats: Count and percentage of differences above each threshold
        - nan_info: Information about NaN values in input arrays
    """
    # Check input shapes
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")
    arr1 = arr1.astype(np.float32)
    arr2 = arr2.astype(np.float32)

    result = {"top_k_diff": [], "threshold_stats": [], "nan_info": {}}

    # Check for NaN values
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)

    if np.any(nan_mask1):
        result["nan_info"]["arr1_nan_count"] = np.sum(nan_mask1)
        result["nan_info"]["arr1_nan_positions"] = np.argwhere(nan_mask1)
        print(
            f"Warning: arr1 contains {result['nan_info']['arr1_nan_count']} NaN values"
        )

    if np.any(nan_mask2):
        result["nan_info"]["arr2_nan_count"] = np.sum(nan_mask2)
        result["nan_info"]["arr2_nan_positions"] = np.argwhere(nan_mask2)
        print(
            f"Warning: arr2 contains {result['nan_info']['arr2_nan_count']} NaN values"
        )

    # Compute absolute differences
    diff = np.abs(arr1 - arr2)
    total_elements = arr1.size

    max_diff_thr = diff / (1.0 + np.abs(arr2))
    max_diff_thr = max_diff_thr.max()
    print(f"diff.abs.max={diff.max()}")
    print(f"max_diff_thr={max_diff_thr}")
    result["max_diff"] = diff.max()
    result["max_diff_thr"] = max_diff_thr

    # Find top k differences
    flat_diff = diff.flatten()
    top_k_indices = np.argpartition(flat_diff, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(-flat_diff[top_k_indices])]

    # Convert flat indices to multi-dimensional indices
    orig_indices = np.unravel_index(top_k_indices, diff.shape)
    for i in range(k):
        idx = tuple(dim[i] for dim in orig_indices)
        result["top_k_diff"].append(
            {
                "value": diff[idx],
                "position": idx,
                "arr1_value": arr1[idx],
                "arr2_value": arr2[idx],
            }
        )

    # Compute threshold statistics
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]
        mask = (diff >= lower) & (diff < upper)
        count = np.sum(mask)
        result["threshold_stats"].append(
            {
                "range": f"[{lower:.1e}, {upper:.1e})",
                "count": count,
                "percentage": 100 * count / total_elements,
            }
        )

    # Handle values above the largest threshold
    mask = diff >= thresholds[-1]
    count = np.sum(mask)
    result["threshold_stats"].append(
        {
            "range": f">={thresholds[-1]:.1e}",
            "count": count,
            "percentage": 100 * count / total_elements,
        }
    )

    print("\nTop differences:")
    for item in result['top_k_diff']:
        print(f"Position {item['position']}: arr1 = {arr1[item['position']]:.6f}, arr2 = {arr2[item['position']]:.6f}, Diff = {item['value']:.6f}")

    print("\nThreshold statistics:")
    for stat in result['threshold_stats']:
        print(f"{stat['range']}: {stat['count']} ({stat['percentage']:.2f}%)")

    print("\nNaN info:")
    print(result["nan_info"])

    return result


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    """Convert cache dtype specification to torch dtype."""
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def create_kv_cache(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
    itemsize: int = 1,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Create key and value cache tensors."""
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    elements_per_vector = 16 // itemsize
    key_cache_shape = (
        num_blocks,
        num_heads,
        head_size // elements_per_vector,
        block_size,
        elements_per_vector,
    )

    key_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(*UNIFORM_RANGE)
            # key_cache.uniform_(-72, 88)
            # key_cache.uniform_(-28, 28)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(
            size=value_cache_shape, dtype=torch_dtype, device=device
        )
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(*UNIFORM_RANGE)
            # value_cache.uniform_(-10, 9)
            # value_cache.uniform_(-56, 56)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)

    return key_caches, value_caches


def reference_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    softmax_scale: float,
    output_dtype: torch.dtype,
    is_causal: bool = True,
    sinks=None,
    sliding_window=0,
) -> torch.Tensor:
    """Reference implementation of masked attention."""
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    s_q = query.shape[0]
    s_k = key.shape[0]
    key = key.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    value = value.repeat_interleave(num_query_heads // num_kv_heads, dim=1)

    attention_weights = torch.einsum("qhd,khd->hqk", query, key) * softmax_scale

    if is_causal:
        query_len = query.shape[0]
        key_len = key.shape[0]
        attention_bias = torch.zeros(
            query_len, key_len, dtype=torch.float32, device=query.device
        )
        causal_mask = torch.ones(
            query_len, key_len, dtype=torch.bool, device=query.device
        ).tril(diagonal=key_len - query_len)
        # attention_bias.masked_fill_(causal_mask.logical_not(), float(-3.4e38))
        attention_bias.masked_fill_(causal_mask.logical_not(), float(-3.4e38))
        attention_weights += attention_bias

    if sliding_window > 0:
        # Handle position calculation for both context and generation phases
        if s_q == s_k:
            # Context phase: standard position calculation
            query_positions = torch.arange(s_q, device=query.device)
            key_positions = torch.arange(s_k, device=query.device)
        else:
            # Generation phase: query is at position s_k (after the cache)
            query_positions = torch.arange(
                s_k, s_k + s_q, device=query.device
            )  # [s_k] for s_q=1
            key_positions = torch.arange(s_k, device=query.device)  # [0,1,2,...,s_k-1]

        # Create position difference matrix: query_pos - key_pos
        pos_diff = query_positions.unsqueeze(1) - key_positions.unsqueeze(
            0
        )  # [s_q, s_k]

        # Sliding window mask: allow attention only if 0 <= pos_diff < sliding_window_size
        sliding_window_mask = (pos_diff < 0) | (
            pos_diff >= sliding_window
        )  # [s_q, s_k]
        attention_weights.masked_fill_(sliding_window_mask.unsqueeze(0), float("-inf"))

    if sinks is not None:
        logits_max = torch.max(attention_weights, dim=-1, keepdim=True).values
        sinks = torch.exp(sinks[:, None, None] - logits_max)
        unnormalized_scores = torch.exp(attention_weights - logits_max)
        normalizer = unnormalized_scores.sum(dim=-1, keepdim=True) + sinks
        attention_weights = unnormalized_scores / normalizer
    else:
        attention_weights = torch.softmax(attention_weights, dim=-1)
    output = torch.einsum("hqk,khd->qhd", attention_weights, value)
    return output.to(output_dtype)


def torch_mha_extend(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    query_output_indptr: torch.Tensor,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    sinks=None,
    sliding_window=0,
) -> torch.Tensor:
    """PyTorch reference implementation of paged attention."""
    num_blocks, num_heads, head_size, block_size = value_cache.shape
    softmax_scale = 1.0 / (head_size**0.5)

    output_dtype = query.dtype
    kv_dtype = key_cache.dtype

    queries_split = torch.tensor_split(query, query_output_indptr.tolist()[1:])
    key_cache_flat = (
        key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, head_size)
    )
    value_cache_flat = (
        value_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_heads, head_size)
    )

    batch_size = query_output_indptr.shape[0] - 1
    outputs = []

    for batch_idx in range(batch_size):
        current_query = queries_split[batch_idx]
        current_block_table = block_tables[batch_idx]
        current_context_length = context_lengths[batch_idx].item()

        token_indices = (
            current_block_table.repeat_interleave(block_size)[:current_context_length]
            * block_size
            + torch.arange(current_context_length, device=current_block_table.device)
            % block_size
        )

        gathered_keys = (
            key_cache_flat.view(torch.int8)[token_indices]
            .view(kv_dtype)
            .to(torch.float)
        )
        if key_scale is not None:
            gathered_keys *= key_scale[:, token_indices].t().unsqueeze(-1)

        gathered_values = (
            value_cache_flat.view(torch.int8)[token_indices]
            .view(kv_dtype)
            .to(torch.float)
        )
        if value_scale is not None:
            gathered_values *= value_scale[:, token_indices].t().unsqueeze(-1)

        attention_output = reference_masked_attention(
            current_query,
            gathered_keys,
            gathered_values,
            softmax_scale,
            output_dtype,
            is_causal=True,
            sinks=sinks,
            sliding_window=sliding_window,
        )
        outputs.append(attention_output)

    return torch.cat(outputs)


def torch_attention_compute(
    query: torch.Tensor,  # [num_seqs, num_q_heads, head_size] - FP8
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size//x, block_size, x] - FP8
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size] or transposed - FP8
    block_tables: torch.Tensor,  # [num_seqs, max_blocks]
    context_lengths: torch.Tensor,  # [num_seqs]
    softmax_scale: float,
    q_seq_len: int,
    query_scale: Optional[
        torch.Tensor
    ] = None,  # per-tensor [1] or per-token [num_seqs, num_q_heads, 1]
    key_scale: Optional[
        torch.Tensor
    ] = None,  # per-tensor [1] or per-token [num_blocks, num_kv_heads, block_size, 1]
    value_scale: Optional[torch.Tensor] = None,  # same as key_scale
    alibi_slopes: Optional[torch.Tensor] = None,  # [num_kv_heads, query_group_size]
    compute_type: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype = torch.bfloat16,
    kv_block_size: int = 16,
    context_partition_size: int = 256,
    is_causal: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Main attention computation stage for Triton's two-stage paged attention decode with FP8.
    Returns intermediate tensors for reduce stage: exp_sums, max_logits, partial_output
    """
    assert query.dtype in [aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16]
    assert key_cache.dtype in [aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16]
    assert value_cache.dtype in [aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16]
    assert compute_type in [aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16]

    num_seqs, num_q_heads_total, head_size = query.shape
    num_blocks, num_kv_heads, _, _, _ = key_cache.shape
    query_group_size = num_q_heads_total // num_kv_heads
    query_group_size_ori = query_group_size // q_seq_len
    assert num_q_heads_total % num_kv_heads == 0

    # Determine value layout
    value_transposed = len(value_cache.shape) == 5

    compute_block_size = 256
    if kv_block_size > context_partition_size and value_transposed:
        compute_block_size = 128

    # Reconstruct full key/value per sequence
    max_seq_len = context_lengths.max().item()
    max_context_partition_num = (
        max_seq_len + context_partition_size - 1
    ) // context_partition_size

    # Output buffers (same as Triton)
    intermediate_shape = (
        num_seqs,
        num_kv_heads,
        max_context_partition_num,
        query_group_size,
    )
    max_logits = torch.full(
        intermediate_shape, -float("inf"), dtype=torch.float32, device=query.device
    )
    exp_sums = torch.zeros(intermediate_shape, dtype=torch.float32, device=query.device)
    partial_output = torch.zeros(
        (*intermediate_shape, head_size), dtype=output_dtype, device=query.device
    )

    FP8_MAX = torch.finfo(aiter.dtypes.fp8).max

    # Quant mode detection
    query_quant_mode = -1
    if query_scale is not None:
        query_quant_mode = 0 if query_scale.numel() == 1 else 1

    kv_quant_mode = -1
    if key_scale is not None and value_scale is not None:
        kv_quant_mode = 0 if key_scale.numel() == 1 else 1

    # Flatten caches for easy indexing
    # key_cache: [num_blocks, num_kv_heads, head_size//x, block_size, x] -> [num_blocks * block_size, num_kv_heads, head_size]
    key_cache_flat = (
        key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_kv_heads, head_size)
    )
    if value_transposed:
        # [num_blocks, num_kv_heads, block_size//x, head_size, x] -> [num_blocks * block_size, num_kv_heads, head_size]
        value_cache_flat = (
            value_cache.permute(0, 2, 4, 1, 3)
            .contiguous()
            .view(-1, num_kv_heads, head_size)
        )
    else:
        # [num_blocks, num_kv_heads, head_size, block_size] -> [num_blocks * block_size, num_kv_heads, head_size]
        value_cache_flat = (
            value_cache.permute(0, 3, 1, 2)
            .contiguous()
            .view(-1, num_kv_heads, head_size)
        )

    # Precompute block -> token mapping
    for seq_idx in range(num_seqs):
        seq_len = context_lengths[seq_idx].item()
        block_table = block_tables[seq_idx]  # [max_blocks]

        # Build token -> physical index mapping
        num_tokens = seq_len
        block_indices = torch.arange(num_tokens, device=query.device) // kv_block_size
        token_offsets = torch.arange(num_tokens, device=query.device) % kv_block_size
        physical_block_ids = block_table[block_indices]  # [num_tokens]
        physical_token_indices = (
            physical_block_ids * kv_block_size + token_offsets
        )  # [num_tokens]

        # Extract per-seq query
        q = query[seq_idx].view(
            num_kv_heads, query_group_size, head_size
        )  # [num_kv_heads, query_group_size, head_size]
        q_scale = 0
        if query_quant_mode >= 0:
            if query_quant_mode == 0:
                q_scale = query_scale.item()
            else:
                q_scale = query_scale[seq_idx].view(
                    num_kv_heads, query_group_size, 1
                )  # [num_kv_heads, query_group_size, 1]
        if q.dtype != compute_type:
            q_fp32 = q.to(compute_type).to(torch.float32)
        else:
            q_fp32 = q.to(torch.float32)
        # q_fp32 = q.to(torch.float32)

        # Process each partition
        for part_idx in range(max_context_partition_num):
            part_start = part_idx * context_partition_size
            part_end = min(part_start + context_partition_size, seq_len)
            if part_start >= seq_len:
                continue

            num_compute_blocks = (
                part_end - part_start + compute_block_size - 1
            ) // compute_block_size

            part_shape = (num_kv_heads, query_group_size)
            max_logits_part = torch.full(
                (*part_shape, 1),
                -float("inf"),
                dtype=torch.float32,
                device=query.device,
            )
            exp_sums_part = torch.zeros(
                *part_shape, 1, dtype=torch.float32, device=query.device
            )
            output_part = torch.zeros(
                (*part_shape, head_size), dtype=torch.float32, device=query.device
            )
            for cb in range(num_compute_blocks):
                cb_start = part_start + cb * compute_block_size
                cb_end = min(cb_start + compute_block_size, part_end)
                if cb_start >= cb_end:
                    break
                cb_len = cb_end - cb_start

                token_range = torch.arange(cb_start, cb_end, device=query.device)
                indices = physical_token_indices[token_range]  # [cb_len]

                # Gather K/V
                k = key_cache_flat.view(torch.int8)[indices].view(
                    key_cache.dtype
                )  # [cb_len, num_kv_heads, head_size]
                v = value_cache_flat.view(torch.int8)[indices].view(
                    value_cache.dtype
                )  # [cb_len, num_kv_heads, head_size]

                k_scale_vals = 0
                v_scale_vals = 0
                # Apply scales
                if kv_quant_mode >= 0:
                    if kv_quant_mode == 0:
                        k_scale_vals = key_scale.item()
                        v_scale_vals = value_scale.item()
                    else:
                        # key_scale: [num_blocks, num_kv_heads, block_size, 1]
                        block_ids = physical_block_ids[token_range]
                        offsets = token_offsets[token_range]
                        k_scale_vals = key_scale[
                            block_ids, :, offsets, 0
                        ]  # [cb_len, num_kv_heads]
                        v_scale_vals = value_scale[
                            block_ids, :, offsets, 0
                        ]  # [cb_len, num_kv_heads]
                        k_scale_vals = k_scale_vals.reshape(
                            cb_len, num_kv_heads, 1
                        ).permute(
                            1, 2, 0
                        )  # [num_kv_heads, 1, cb_len]
                        v_scale_vals = v_scale_vals.reshape(
                            cb_len, num_kv_heads, 1
                        ).permute(
                            1, 2, 0
                        )  # [num_kv_heads, 1, cb_len]

                qk_scale = 0
                if query_quant_mode >= 0:
                    # [num_kv_heads, query_group_size, cb_len]
                    if kv_quant_mode >= 0:
                        qk_scale = softmax_scale * q_scale * k_scale_vals
                    else:
                        qk_scale = softmax_scale * q_scale
                else:
                    if kv_quant_mode >= 0:
                        qk_scale = softmax_scale * k_scale_vals
                    else:
                        qk_scale = softmax_scale

                if k.dtype != compute_type:
                    k_fp32 = k.to(compute_type).to(torch.float32)
                else:
                    k_fp32 = k.to(torch.float32)
                if v.dtype != compute_type:
                    v_fp32 = v.to(compute_type).to(torch.float32)
                else:
                    v_fp32 = v.to(torch.float32)
                # k_fp32 = k.to(torch.float32)
                # v_fp32 = v.to(torch.float32)

                # Compute QK = q @ k^T  --> [num_kv_heads, query_group_size, cb_len]
                # q_fp32: [num_kv_heads, query_group_size, head], k_fp32: [cb_len, num_kv_heads, head]
                qk = torch.einsum(
                    "hqd,khd->hqk", q_fp32, k_fp32
                )  # [num_kv_heads, query_group_size, cb_len]

                qk = qk * qk_scale

                # ALiBi bias
                if alibi_slopes is not None:
                    # alibi_slopes: [num_kv_heads, query_group_size]
                    slopes = alibi_slopes[
                        :, :query_group_size
                    ]  # [num_kv_heads, query_group_size]
                    positions = token_range.unsqueeze(0).unsqueeze(0)  # [1,1,cb_len]
                    # In Triton: alibi_bias = slope * (col - kv_len + 1)
                    alibi_bias = slopes.unsqueeze(-1) * (
                        positions - seq_len + 1
                    )  # [num_kv_heads, query_group_size, cb_len]
                    qk += alibi_bias

                # Causal mask
                if is_causal:
                    q_positions = torch.arange(
                        query_group_size, device=query.device
                    )  # [query_group_size]
                    valid_mask = (
                        q_seq_len
                        - 1
                        - q_positions[:, None] // query_group_size_ori
                        + token_range[None, :]
                        < seq_len
                    )  # [query_group_size, cb_len]
                    valid_mask = valid_mask[None, :, :]  # [1, query_group_size, cb_len]
                    # valid_mask = token_range < seq_len  # [cb_len]
                    qk = qk.masked_fill(~valid_mask, -3.4e38)

                # Compute local max and exp
                current_max = qk.max(
                    dim=-1, keepdim=True
                ).values  # [num_kv_heads, query_group_size, 1]
                new_max = torch.maximum(
                    max_logits_part, current_max
                )  # [num_kv_heads, query_group_size, 1]
                acc_scale = torch.exp(max_logits_part - new_max)

                # Compute attention probs
                probs = torch.exp(
                    qk - new_max
                )  # [num_kv_heads, query_group_size, cb_len]
                exp_sums_part = acc_scale * exp_sums_part + probs.sum(
                    dim=-1, keepdim=True
                )  # [num_kv_heads, query_group_size, 1]

                # Handle value scaling for FP8 (Triton special logic)
                if kv_quant_mode == 1:
                    valid_v_scale = v_scale_vals
                    v_scale_max = valid_v_scale.max(
                        dim=-1, keepdim=True
                    ).values  # [num_kv_heads, 1, 1]
                    # Avoid division by zero
                    # v_scale_max = torch.clamp(v_scale_max, min=1e-12)
                    v_scale_vals = (
                        v_scale_vals * FP8_MAX / (v_scale_max + 1e-8)
                    )  # [num_kv_heads, 1, cb_len]
                    probs_scaled = (
                        v_scale_vals * probs
                    )  # [num_kv_heads, query_group_size, cb_len]
                    prob_scale = v_scale_max / FP8_MAX  # [num_kv_heads, 1, 1]
                elif kv_quant_mode == 0:
                    probs_scaled = FP8_MAX * probs
                    prob_scale = v_scale_vals / FP8_MAX
                else:
                    probs_scaled = probs

                # import pdb; pdb.set_trace()
                probs_scaled = probs_scaled.to(compute_type).to(torch.float32)
                # Compute PV = probs @ v
                # probs_scaled: [num_kv_heads, query_group_size, cb_len], v_fp32: [cb_len, num_kv_heads, head]
                pv = torch.einsum(
                    "hqk,khd->hqd", probs_scaled, v_fp32
                )  # [num_kv_heads, query_group_size, head]

                # Accumulate with rescaling
                output_part = (
                    acc_scale * output_part
                )  # [num_kv_heads, query_group_size, head]
                if kv_quant_mode >= 0:
                    output_part += prob_scale * pv
                else:
                    output_part += pv

                max_logits_part = new_max

            exp_sums_part_reciprocal = 1.0 / exp_sums_part
            output_part = output_part * exp_sums_part_reciprocal
            # Store back
            max_logits[seq_idx, :, part_idx, :] = max_logits_part.squeeze(-1)
            exp_sums[seq_idx, :, part_idx, :] = exp_sums_part.squeeze(-1)
            partial_output[seq_idx, :, part_idx, :, :] = output_part.to(output_dtype)

    return exp_sums, max_logits, partial_output


def torch_reduce_compute(
    output: torch.Tensor,  # [num_seqs, num_q_heads_total, head_size]
    exp_sums: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    max_logits: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    temporary_output: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]
    context_lengths: torch.Tensor,  # [num_seqs]
    context_partition_size: int = 256,
) -> torch.Tensor:
    """
    Reference implementation of the reduce kernel.
    This mimics the reduce stage from torch_mha_extend_flashattn_style function.
    """
    num_seqs = output.shape[0]
    num_q_heads_total = output.shape[1]
    head_size = output.shape[2]
    final_output = torch.empty_like(output)

    for seq_idx in range(num_seqs):
        seq_len = context_lengths[seq_idx].item()
        num_parts = (seq_len + context_partition_size - 1) // context_partition_size

        # Global max across partitions
        global_max = (
            max_logits[seq_idx, :, :num_parts, :].max(dim=1).values
        )  # [num_kv_heads, query_group_size]

        # Rescale exp_sums
        exp_sums_local = exp_sums[
            seq_idx, :, :num_parts, :
        ]  # [num_kv_heads, num_parts, query_group_size]
        max_local = max_logits[
            seq_idx, :, :num_parts, :
        ]  # [num_kv_heads, num_parts, query_group_size]
        exp_sums_rescaled = exp_sums_local * torch.exp(
            max_local - global_max.unsqueeze(1)
        )  # [num_kv_heads, num_parts, query_group_size]
        global_exp_sum = exp_sums_rescaled.sum(
            dim=1
        )  # [num_kv_heads, query_group_size]

        # Avoid division by zero
        global_exp_sum = torch.clamp(global_exp_sum, min=1e-12)

        # Weighted sum of partial outputs
        weights = exp_sums_rescaled / global_exp_sum.unsqueeze(
            1
        )  # [num_kv_heads, num_parts, query_group_size]
        partial_seq = temporary_output[
            seq_idx, :, :num_parts, :, :
        ]  # [num_kv_heads, num_parts, query_group_size, head]
        weighted = (partial_seq * weights.unsqueeze(-1)).sum(
            dim=1
        )  # [num_kv_heads, query_group_size, head]

        final_output[seq_idx] = weighted.view(num_q_heads_total, head_size)

    return final_output


def torch_mha_extend_flashattn_style(
    query: torch.Tensor,  # [num_seqs, num_q_heads, head_size] - FP8
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size//x, block_size, x] - FP8
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, block_size] or transposed - FP8
    block_tables: torch.Tensor,  # [num_seqs, max_blocks]
    context_lengths: torch.Tensor,  # [num_seqs]
    softmax_scale: float,
    q_seq_len: int,
    query_scale: Optional[
        torch.Tensor
    ] = None,  # per-tensor [1] or per-token [num_seqs, num_q_heads, 1]
    key_scale: Optional[
        torch.Tensor
    ] = None,  # per-tensor [1] or per-token [num_blocks, num_kv_heads, block_size, 1]
    value_scale: Optional[torch.Tensor] = None,  # same as key_scale
    alibi_slopes: Optional[torch.Tensor] = None,  # [num_kv_heads, query_group_size]
    compute_type: torch.dtype = torch.bfloat16,
    kv_block_size: int = 16,
    context_partition_size: int = 256,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Reference implementation mimicking Triton's two-stage paged attention decode with FP8.
    This function now calls torch_attention_compute for the main computation and torch_reduce_compute for reduction.
    """
    if compute_type == aiter.dtypes.fp8:
        output_dtype = torch.bfloat16
    else:
        output_dtype = compute_type
    # Main attention computation stage
    exp_sums, max_logits, partial_output = torch_attention_compute(
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        block_tables=block_tables,
        context_lengths=context_lengths,
        softmax_scale=softmax_scale,
        q_seq_len=q_seq_len,
        query_scale=query_scale,
        key_scale=key_scale,
        value_scale=value_scale,
        alibi_slopes=alibi_slopes,
        compute_type=compute_type,
        output_dtype=output_dtype,
        kv_block_size=kv_block_size,
        context_partition_size=context_partition_size,
        is_causal=is_causal,
    )

    num_seqs, num_q_heads_total, head_size = query.shape
    # Reduce stage
    final_output = torch_reduce_compute(
        torch.zeros(
            (num_seqs, num_q_heads_total, head_size),
            dtype=output_dtype,
            device=query.device,
        ),
        exp_sums,
        max_logits,
        partial_output,
        context_lengths,
        context_partition_size,
    )

    return final_output


def quantize_kv_cache_symmetric(
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    quant_dtype: torch.dtype,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Apply symmetric per-token quantization to KV cache."""
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    total_tokens = num_blocks * block_size

    key_cache_reshaped = (
        key_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )

    value_cache_reshaped = (
        value_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )

    quantized_keys, key_scales_original = pertoken_quant(
        key_cache_reshaped, quant_dtype=quant_dtype
    )
    quantized_values, value_scales_original = pertoken_quant(
        value_cache_reshaped, quant_dtype=quant_dtype
    )

    elements_per_vector = 16 // quant_dtype.itemsize

    quantized_keys = (
        quantized_keys.view(
            num_blocks,
            num_heads,
            block_size,
            head_dim // elements_per_vector,
            elements_per_vector,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )

    key_scales_flat = (
        key_scales_original.permute(1, 0, 2, 3)
        .contiguous()
        .view(num_heads, total_tokens)
    )

    quantized_values = (
        quantized_values.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )

    value_scales_flat = (
        value_scales_original.permute(1, 0, 2, 3)
        .contiguous()
        .view(num_heads, total_tokens)
    )

    return (
        quantized_keys,
        key_scales_flat,
        quantized_values,
        value_scales_flat,
        key_scales_original,
        value_scales_original,
    )


def quantize_kv_cache_per_tensor(
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    quant_dtype: torch.dtype,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Apply per-tensor quantization to KV cache."""
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    elements_per_vector = 16 // quant_dtype.itemsize

    key_cache_reshaped = (
        key_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    key_cache_reshaped = (
        key_cache_reshaped.view(
            num_blocks,
            num_heads,
            block_size,
            head_dim // elements_per_vector,
            elements_per_vector,
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )

    # Per-tensor quantization for keys
    quantized_keys, key_scales_original = per_tensor_quant(
        key_cache_reshaped, quant_dtype=quant_dtype
    )
    # Per-tensor quantization for values
    quantized_values, value_scales_original = per_tensor_quant(
        value_cache, quant_dtype=quant_dtype
    )

    # For per-tensor quantization, scales are scalars
    key_scales_flat = key_scales_original.expand(num_heads, num_blocks * block_size)
    value_scales_flat = value_scales_original.expand(num_heads, num_blocks * block_size)

    return (
        quantized_keys,
        key_scales_flat,
        quantized_values,
        value_scales_flat,
        key_scales_original,
        value_scales_original,
    )


@perftest()
def run_aiter_assembly_kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables_stride0: int,
    max_query_length: int,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    query_output_indptr: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run AIT assembly kernel for paged attention."""
    return aiter.pa_fwd_asm(
        query,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        block_tables_stride0,
        max_query_length,
        key_scale,
        value_scale,
        None,
        query_output_indptr,
        high_precision=0,
    )


def shuffle_value_cache_layout(value_cache: torch.Tensor) -> torch.Tensor:
    """Shuffle value cache layout for optimized memory access."""
    # value_cache: [num_blocks, num_kv_heads, head_size, kv_block_size]
    elements_per_vector = 16 // value_cache.element_size()
    num_blocks, num_kv_heads, head_size, block_size = value_cache.shape

    value_cache_reshaped = value_cache.view(
        num_blocks,
        num_kv_heads,
        head_size,
        block_size // elements_per_vector,
        elements_per_vector,
    )

    # [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    value_cache_shuffled = value_cache_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    return value_cache_shuffled


def prepare_gluon_query_and_scale(
    quantized_query: torch.Tensor,
    query_scale_factors: torch.Tensor,
    reference_output_quant: torch.Tensor,
    batch_size: int,
    query_length: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare inputs for Gluon kernel by reshaping and transposing tensors.

    Args:
        quantized_query: Quantized query tensor [batch_size * query_length, num_query_heads, head_size]
        query_scale_factors: Query scale factors [batch_size * query_length, num_query_heads, 1] or scalar
        reference_output_quant: Reference output tensor [batch_size * query_length, num_query_heads, head_size]
        batch_size: Batch size
        query_length: Query sequence length
        num_query_heads: Number of query heads
        num_kv_heads: Number of key-value heads
        head_size: Head dimension size

    Returns:
        Tuple of (quantized_query_gluon, query_scale_gluon, output_gluon)
    """
    quantized_query_gluon = quantized_query
    query_scale_gluon = query_scale_factors
    output_gluon = torch.empty_like(reference_output_quant)
    # output_gluon = torch.zeros_like(reference_output_quant)

    if query_length > 1:
        query_group_size = num_query_heads // num_kv_heads

        # Reshape and transpose query tensor for Gluon kernel
        quantized_query_gluon = quantized_query.reshape(
            batch_size, query_length, num_kv_heads, query_group_size, head_size
        )
        quantized_query_gluon = quantized_query_gluon.transpose(1, 2).reshape(
            batch_size, num_kv_heads * query_length * query_group_size, head_size
        )

        # Reshape and transpose output tensor for Gluon kernel
        output_gluon = output_gluon.reshape(
            batch_size, query_length, num_kv_heads, query_group_size, head_size
        )
        output_gluon = output_gluon.transpose(1, 2).reshape(
            batch_size, num_kv_heads * query_length * query_group_size, head_size
        )

        # Handle query scale factors based on quantization mode
        if (
            query_scale_factors is not None and len(query_scale_factors.shape) > 1
        ):  # per-token quantization
            query_scale_gluon = query_scale_factors.reshape(
                batch_size, query_length, num_kv_heads, query_group_size, 1
            )
            query_scale_gluon = query_scale_gluon.transpose(1, 2).reshape(
                batch_size, num_kv_heads * query_length * query_group_size, 1
            )

    return quantized_query_gluon, query_scale_gluon, output_gluon


@perftest()
def run_gluon_kernel(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    context_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    softmax_scale: float,
    query_length: int,
    max_context_partition_num: int,
    context_partition_size: int,
    compute_type: torch.dtype,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    temporary_output: torch.Tensor,
    alibi_slopes: Optional[torch.Tensor] = None,
    use_aot_impl: bool = False,
    sinks: Optional[torch.Tensor] = None,
    sliding_window: int = 0,
    ps=False,
) -> None:
    """Run Gluon FP8/BF16/FP16 kernel for paged attention.

    Args:
        output: Output tensor [num_seqs * query_length, num_query_heads, head_size]
        query: Query tensor [num_seqs * query_length, num_query_heads, head_size]
        key_cache: Key cache tensor [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
        value_cache: Value cache tensor [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
        context_lengths: Current context lengths for each sequence [num_seqs]
        block_tables: Mapping from sequences to physical cache blocks [num_seqs, max_num_blocks_per_seq]
        softmax_scale: Softmax scale factor, typically 1/sqrt(head_size)
        query_length: Query sequence length
        max_context_length: Maximum sequence length supported
        context_partition_size: Context partition size
        compute_type: Compute data type (torch.dtype)
        query_scale: Query scale tensor [num_seqs * query_length, num_query_heads, 1] or [1]
        key_scale: Key scale tensor [num_blocks, num_kv_heads, kv_block_size, 1]
        value_scale: Value scale tensor [num_blocks, num_kv_heads, kv_block_size, 1]
        exp_sums: Exponential sums tensor [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
        max_logits: Max logits tensor [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
        temporary_output: Temporary output tensor [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]
        alibi_slopes: Optional ALiBi slopes tensor
        use_aot_impl: Whether to use AOT implementation (default: False)
        sinks: Optional sinks tensor for attention sinks
        sliding_window: Sliding window size (default: 0, disabled)

    Returns:
        None (modifies output in-place)
        Note: The @perftest() decorator wraps this to return (None, avg_time)

    This function can run in aot or jit mode based on use_aot_impl flag.
    """
    # Run kernel
    if use_aot_impl and sliding_window == 0 and not ps:
        pa_decode_gluon_aot(
            output,
            query,
            key_cache,
            value_cache,
            context_lengths,
            block_tables,
            softmax_scale,
            query_length,
            max_context_partition_num,
            context_partition_size,
            compute_type,
            query_scale,
            key_scale,
            value_scale,
            exp_sums=exp_sums,
            max_logits=max_logits,
            temporary_output=temporary_output,
            alibi_slopes=alibi_slopes,
            sinks=sinks,
        )
    else:
        if pa_decode_gluon is not None:
            torch.ops.aiter.pa_decode_gluon(
            # pa_decode_flydsl(
                output,
                query,
                key_cache,
                value_cache,
                context_lengths,
                block_tables,
                softmax_scale,
                query_length,
                max_context_partition_num,
                context_partition_size,
                compute_type,
                query_scale,
                key_scale,
                value_scale,
                exp_sums=exp_sums,
                max_logits=max_logits,
                temporary_output=temporary_output,
                alibi_slopes=alibi_slopes,
                sinks=sinks,
                sliding_window=sliding_window,
                ps=ps,
            )
        else:
            raise RuntimeError(
                "This version triton is not support gluon, please upgrade to 3.5.0 or higher!"
            )


@benchmark()
def run_pa_gluon_test(
    context_length: int,
    batch_size: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    compute_type: torch.dtype,
    query_length: int,
    quant_mode: str,
    context_partition_size: int,
    trans_v: bool,
    kv_varlen: bool,
    use_aot_impl: bool,
    quant_q: bool,
    quant_kv: bool,
    use_sinks: bool,
    sliding_window: int,
    ps: bool,
) -> Dict[str, Union[float, str]]:
    """Test paged attention decode with assembly and gluon implementations."""
    data_type = compute_type
    if compute_type == aiter.dtypes.fp8:
        data_type = torch.bfloat16
    results = {}
    seed = 123
    # seed = 371
    setup_seed(seed)
    device = "cuda:0"
    torch.set_default_device(device)
    num_query_heads, num_kv_heads = num_heads
    assert (
        num_query_heads % num_kv_heads == 0
    ), "Query heads must be divisible by KV heads"

    max_context_length = max(16384, context_length)
    # max_context_length = max(65536, context_length)
    max_blocks_per_sequence = (max_context_length + block_size - 1) // block_size
    total_blocks = max_blocks_per_sequence * batch_size
    blocks_per_sequence = (context_length + block_size - 1) // block_size

    query_output_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    sequence_lengths_qo = torch.randint(
        1, 5, (batch_size,), dtype=torch.int32, device=device
    ).fill_(query_length)
    query_output_indptr[1 : batch_size + 1] = torch.cumsum(sequence_lengths_qo, dim=0)
    total_queries = query_output_indptr[-1].item()
    max_query_length = sequence_lengths_qo.max().item()

    qkv_tensor = torch.randn(
        total_queries, num_query_heads + 2 * num_kv_heads, head_size, dtype=data_type
    )
    query, key, value = torch.split(
        qkv_tensor, [num_query_heads, num_kv_heads, num_kv_heads], dim=1
    )
    query.uniform_(*UNIFORM_RANGE)
    # query.uniform_(-9.0625, 7.8125)
    # query.uniform_(-120, 85)

    if kv_varlen:
        random.seed(seed)
        # kv_len_list = [random.randint(1, context_length) for _ in range(batch_size)]
        kv_len_list = [
            random.randint(query_length, context_length) for _ in range(batch_size)
        ]
    else:
        kv_len_list = [context_length] * batch_size

    context_lengths = torch.tensor(kv_len_list, dtype=torch.int32, device=device)
    # print(f"context_lengths={context_lengths}")
    if use_sinks:
        sinks = torch.randn(num_query_heads, device=query.device, dtype=data_type)
    else:
        sinks = None
    random.seed(seed)
    block_tables_list = []
    for _ in range(batch_size):
        block_table = [
            random.randint(0, total_blocks - 1) for _ in range(blocks_per_sequence)
        ]
        block_tables_list.append(block_table)

    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

    key_caches, value_caches = create_kv_cache(
        total_blocks,
        block_size,
        1,
        num_kv_heads,
        head_size,
        "auto",
        data_type,
        seed,
        device,
        1 if quant_kv else 2,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    softmax_scale = 1.0 / (head_size**0.5)

    # Quantization based on mode and flags
    if quant_mode == "per_token":
        # Per-token quantization for query (if enabled)
        if quant_q:
            quantized_query, query_scale_factors = pertoken_quant(
                query, quant_dtype=aiter.dtypes.fp8
            )
        else:
            quantized_query = query
            query_scale_factors = None

        # Per-token quantization for KV cache (if enabled)
        if quant_kv:
            if compute_type not in [aiter.dtypes.fp8]:
                (
                    quantized_keys,
                    key_scale_factors_flat,
                    quantized_values,
                    value_scale_factors_flat,
                    key_scale_original,
                    value_scale_original,
                ) = quantize_kv_cache_symmetric(
                    key_cache, value_cache, quant_dtype=aiter.dtypes.fp8
                )
            else:
                quantized_keys = key_cache.to(aiter.dtypes.fp8)
                quantized_values = value_cache.to(aiter.dtypes.fp8)
                key_scale_factors_flat = None
                value_scale_factors_flat = None
                key_scale_original = torch.tensor(1, dtype=torch.float32, device=device)
                value_scale_original = torch.tensor(
                    1, dtype=torch.float32, device=device
                )
        else:
            quantized_keys = key_cache
            quantized_values = value_cache
            key_scale_factors_flat = None
            value_scale_factors_flat = None
            key_scale_original = None
            value_scale_original = None
    else:  # per_tensor
        # Per-tensor quantization for query (if enabled)
        if quant_q:
            quantized_query, query_scale_factors = per_tensor_quant(
                query, quant_dtype=aiter.dtypes.fp8
            )
        else:
            quantized_query = query
            query_scale_factors = None

        # Per-tensor quantization for KV cache (if enabled)
        if quant_kv:
            (
                quantized_keys,
                key_scale_factors_flat,
                quantized_values,
                value_scale_factors_flat,
                key_scale_original,
                value_scale_original,
            ) = quantize_kv_cache_per_tensor(
                key_cache, value_cache, quant_dtype=aiter.dtypes.fp8
            )
        else:
            quantized_keys = key_cache
            quantized_values = value_cache
            key_scale_factors_flat = None
            value_scale_factors_flat = None
            key_scale_original = None
            value_scale_original = None

    # Reference (original)
    reference_output_quant = torch_mha_extend(
        query,
        quantized_keys,
        quantized_values,
        block_tables,
        context_lengths,
        query_output_indptr,
        key_scale_factors_flat,
        value_scale_factors_flat,
        sinks=sinks,
        sliding_window=sliding_window,
    )
    reference_output_quant = reference_output_quant.to(data_type)
    kv_len_list = [
        min(context_length, sliding_window) if sliding_window > 0 else context_length
        for context_length in kv_len_list
    ]
    pa_rw_bytes = head_size * (
        2 * sum(kv_len_list) * num_kv_heads * quantized_keys.dtype.itemsize
        + 2 * query_length * num_query_heads * quantized_query.dtype.itemsize
    )

    if trans_v:
        quantized_values = shuffle_value_cache_layout(quantized_values)
        # print(f"Transformed quantized_values.shape={quantized_values.shape}")

    diff_tolerance = 5e-3
    if compute_type != aiter.dtypes.fp8 and not quant_q and not quant_kv:
        diff_tolerance = 5e-4
    if kv_varlen:
        diff_tolerance = 5e-2
        if compute_type != aiter.dtypes.fp8 and not quant_q and not quant_kv:
            diff_tolerance = 5e-3
    if sliding_window > 0:
        diff_tolerance = 8e-2

    flash_style_diff_tolerance = 5e-4
    if quant_mode == "per_token" and (quant_q or quant_kv):
        flash_style_diff_tolerance = 5e-3
    if kv_varlen:
        flash_style_diff_tolerance = 5e-3
        if quant_mode == "per_token" and (quant_q or quant_kv):
            flash_style_diff_tolerance = 5e-2

    quantized_query_gluon, query_scale_gluon, output_gluon = (
        prepare_gluon_query_and_scale(
            quantized_query,
            query_scale_factors,
            reference_output_quant,
            batch_size,
            query_length,
            num_query_heads,
            num_kv_heads,
            head_size,
        )
    )

    if USE_TORCH_FLASH_REF:
        # Reference (flash attention style - mimicking Triton kernel)
        reference_output_flashattn = torch_mha_extend_flashattn_style(
            quantized_query_gluon,
            quantized_keys,
            quantized_values,
            block_tables,
            context_lengths,
            softmax_scale,
            query_length,
            query_scale=query_scale_gluon,
            key_scale=key_scale_original,
            value_scale=value_scale_original,
            compute_type=compute_type,
            kv_block_size=block_size,
            context_partition_size=context_partition_size,
        )
        if query_length > 1:
            query_group_size = num_query_heads // num_kv_heads
            reference_output_flashattn = reference_output_flashattn.reshape(
                batch_size, num_kv_heads, query_length, query_group_size, head_size
            )
            reference_output_flashattn = reference_output_flashattn.transpose(
                1, 2
            ).reshape(
                batch_size * query_length, num_kv_heads * query_group_size, head_size
            )
        print("\n=== Comparing Two Reference Implementations ===")
        ref_diff = (
            (reference_output_quant - reference_output_flashattn).abs().max().item()
        )
        print(f"FlashAttn-style Ref vs Original Ref: max diff = {ref_diff:.6e}")
        compare_arrays(
            reference_output_flashattn.to(torch.float32).detach().cpu().numpy(),
            reference_output_quant.to(torch.float32).detach().cpu().numpy(),
        )
        out_flashattn_ref_md5 = hashlib.md5(
            reference_output_flashattn.contiguous()
            .view(torch.uint8)
            .detach()
            .cpu()
            .numpy()
            .tobytes()
        ).hexdigest()
        print(f"out_flashattn_ref_md5={out_flashattn_ref_md5}")

    # Create intermediate tensors for attention computation
    num_seqs = batch_size
    num_kv_heads_local = num_kv_heads  # Avoid shadowing
    max_context_length = (
        min(context_lengths.max().item(), sliding_window)
        if sliding_window > 0
        else context_lengths.max().item()
    )
    if sliding_window > 0:

        max_context_partition_num = 1
    elif ps:
        max_context_partition_num = get_recommended_splits(num_seqs, num_kv_heads)
    else:
        max_context_partition_num = triton.cdiv(
            max_context_length, context_partition_size
        )

    equivalent_query_group_size = query_length * (num_query_heads // num_kv_heads)
    intermediate_shape = (
        num_seqs,
        num_kv_heads_local,
        max_context_partition_num,
        equivalent_query_group_size,
    )

    exp_sums = torch.empty(
        intermediate_shape, dtype=torch.float32, device=reference_output_quant.device
    )
    max_logits = torch.empty(
        intermediate_shape, dtype=torch.float32, device=reference_output_quant.device
    )
    temporary_output = torch.empty(
        *intermediate_shape,
        head_size,
        dtype=reference_output_quant.dtype,
        device=reference_output_quant.device,
    )
    # Create output tensor with the same shape as reference
    final_output_gluon = torch.empty_like(reference_output_quant)

    _, gluon_time = run_gluon_kernel(
        final_output_gluon,
        quantized_query,
        quantized_keys,
        quantized_values,
        context_lengths,
        block_tables,
        softmax_scale,
        query_length,
        max_context_partition_num,
        context_partition_size,
        compute_type,
        query_scale=query_scale_factors,
        key_scale=key_scale_original,
        value_scale=value_scale_original,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        alibi_slopes=None,
        use_aot_impl=use_aot_impl,
        sinks=sinks,
        sliding_window=sliding_window,
        ps=ps,
    )

    # Compare with original reference
    err_gluon = checkAllclose(
        reference_output_quant,
        final_output_gluon,
        atol=diff_tolerance,
        rtol=diff_tolerance,
        msg=f"[PyTorch vs Gluon_FP8][{quant_mode}] (vs orig ref): {gluon_time:>8.2f} us......",
    )

    if err_gluon > 0:
        err_gluon = 1
    print("\n=== Detailed Error Analysis ===")
    print("Gluon vs Original Ref:")
    diff_result = compare_arrays(
        final_output_gluon.to(torch.float32).detach().cpu().numpy(),
        reference_output_quant.to(torch.float32).detach().cpu().numpy(),
    )
    if diff_result["max_diff_thr"] < diff_tolerance:
        print("gluon_vs_torch_ref PASSED")
    else:
        print("gluon_vs_torch_ref FAILED")
    # Track results based on implementation type
    results["us_gluon"] = gluon_time
    results["err_gluon"] = err_gluon

    if USE_TORCH_FLASH_REF:
        print("\nGluon vs FlashAttn-style Ref:")
        diff_result = compare_arrays(
            final_output_gluon.to(torch.float32).detach().cpu().numpy(),
            reference_output_flashattn.to(torch.float32).detach().cpu().numpy(),
        )
        if diff_result["max_diff_thr"] < flash_style_diff_tolerance:
            print("gluon_vs_torch_flash_ref PASSED")
        else:
            print("gluon_vs_torch_flash_ref FAILED")

    # MD5 hash
    out_ref_md5 = hashlib.md5(
        reference_output_quant.contiguous()
        .view(torch.uint8)
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    gluon_hash = hashlib.md5(
        final_output_gluon.contiguous()
        .view(torch.uint8)
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    print(f"out_ref_md5={out_ref_md5}")
    print(f"gluon_output_md5={gluon_hash}")

    # Bandwidth
    kernel_time_us = gluon_time
    bandwidth_tb_per_sec = pa_rw_bytes / (kernel_time_us * 1e6 * 1.024**4)
    results["gluon_bandwith(TB/s)"] = bandwidth_tb_per_sec

    # Test Assembly
    query_group_size = num_query_heads // num_kv_heads
    skip_assembly = (
        (block_size == 1024 and num_heads != (10, 1))
        or (block_size == 1024 and arch_info.get_arch() in ["gfx950"])
        or (block_size == 16 and query_group_size == 8 and query_length == 3)
        or (query_group_size == 5 and query_length == 3)
        or (block_size == 64)
        # or (not quant_kv)
        or (compute_type == torch.float16 and (quant_q or quant_kv))
        or (head_size not in [128])
        or (sliding_window > 0)
        or True
    )

    # aiter_assembly_kernel do not support per-tensor quantization, we always use per-token quantization here
    if quant_kv and quant_mode == "per_tensor":
        key_scale_original = key_scale_factors_flat.contiguous()
        value_scale_original = value_scale_factors_flat.contiguous()
    if not skip_assembly:
        # Assembly kernel always expects shuffled V layout [num_blocks, num_kv_heads, block_size/X, head_size, X]
        asm_values = quantized_values if trans_v else shuffle_value_cache_layout(quantized_values)
        assembly_output, assembly_time = run_aiter_assembly_kernel(
            query,
            quantized_keys,
            asm_values,
            block_tables,
            context_lengths,
            block_tables.size(1),
            max_query_length,
            key_scale_original,
            value_scale_original,
            query_output_indptr,
        )
        print("\nAIT_Assembly vs Original Ref:")
        compare_arrays(
            assembly_output.to(torch.float32).detach().cpu().numpy(),
            reference_output_quant.to(torch.float32).detach().cpu().numpy(),
        )
        assembly_md5 = hashlib.md5(
            assembly_output.contiguous()
            .view(torch.uint8)
            .detach()
            .cpu()
            .numpy()
            .tobytes()
        ).hexdigest()
        print(f"assembly_md5={assembly_md5}")

        results["us_asm"] = assembly_time
        assembly_bandwidth = pa_rw_bytes / (assembly_time * 1e6 * 1.024**4)
        results["asm_bandwith(TB/s)"] = assembly_bandwidth

    if "us_asm" in results:
        results["perf_gluon_vs_asm"] = f'{results["us_asm"] / results["us_gluon"]:.0%}'
    else:
        results["perf_gluon_vs_asm"] = "NaN"

    sys.stdout.flush()

    return results


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test paged attention decode gluon implementation",
    )

    parser.add_argument(
        "--compute_type",
        type=str,
        default=None,
        help="Compute type",
    )
    parser.add_argument(
        "-n",
        "--num_heads",
        type=dtypes.str2tuple,
        default=None,
        help="Number of heads (q_heads, kv_heads)",
    )
    parser.add_argument(
        "-q",
        "--query_length",
        type=int,
        choices=QUERY_LENGTH_OPTIONS,
        default=None,
        help="Query length",
    )
    parser.add_argument(
        "-c", "--context_length", type=int, default=None, help="Context length"
    )
    parser.add_argument("-b", "--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument(
        "-d", "--head_dim", type=int, default=None, help="Head dimension"
    )
    parser.add_argument("--block_size", type=int, default=None, help="Block size")
    parser.add_argument(
        "--quant_mode",
        type=str,
        choices=["per_token", "per_tensor", "both"],
        default=None,
        help="Quantization mode: per_token, per_tensor, or both",
    )
    parser.add_argument(
        "--quant_q_and_kv",
        type=dtypes.str2tuple,
        default=None,
        help=(
            "Tuple of bools specifying whether to quant_q and quant_kv, e.g. 0,0 "
            "First value is for quant_q, second for quant_kv."
        ),
    )
    parser.add_argument(
        "--trans_v",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Transpose value cache layout (True/False)",
    )
    parser.add_argument(
        "--kv_varlen",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="KV use varlen (True/False)",
    )
    parser.add_argument(
        "--use_torch_flash_ref",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Use torch flash reference implementation (True/False)",
    )
    parser.add_argument(
        "--use_aot_impl",
        type=lambda x: (str(x).lower() == "true"),
        default=None,
        help="Use gluon AOT implementation (True/False)",
    )
    parser.add_argument(
        "--context_partition_size",
        type=int,
        default=None,
        help="Sequence partition size",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=1.0,
        help="Sample rate for test cases (0.0-1.0). Each test case has sample_rate probability to be selected. Default: 1.0 (run all tests)",
    )

    return parser


def process_arguments(args: argparse.Namespace) -> tuple:
    """Process command line arguments."""
    compute_types = COMPUTE_TYPE_OPTIONS
    block_sizes = BLOCK_SIZE_OPTIONS
    head_configs = HEAD_CONFIGURATIONS
    context_lengths = CONTEXT_LENGTH_OPTIONS
    batch_sizes = BATCH_SIZE_OPTIONS
    head_sizes = HEAD_DIMENSION_OPTIONS
    query_lengths = QUERY_LENGTH_OPTIONS
    quant_mode = QUANT_MODE_OPTIONS
    trans_v = TRANS_V_OPTIONS
    kv_varlen = KV_VARLEN_OPTIONS
    quant_q_and_kv = QUANT_Q_AND_KV_OPTIONS
    use_torch_flash_ref_options = USE_TORCH_FLASH_REF_OPTIONS
    use_aot_impl_options = USE_AOT_IMPL_OPTIONS
    context_partition_size_options = CONTEXT_PARTITION_SIZE_OPTIONS
    sinks_options = SINKS_OPTIONS
    sliding_window_options = SLIDING_WINDOW_OPTIONS
    ps_options = PS_OPTIONS
    if args.compute_type is not None:
        compute_types = [dtypes.d_dtypes[args.compute_type]]
    else:
        compute_types = [dtypes.d_dtypes[key] for key in compute_types]

    if args.num_heads is not None:
        head_configs = [args.num_heads]
    if args.query_length is not None:
        query_lengths = [args.query_length]
    if args.context_length is not None:
        context_lengths = [args.context_length]
    if args.batch_size is not None:
        batch_sizes = [args.batch_size]
    if args.head_dim is not None:
        head_sizes = [args.head_dim]
    if args.block_size is not None:
        block_sizes = [args.block_size]
    if args.quant_mode is not None:
        quant_mode = [args.quant_mode]
    if args.trans_v is not None:
        trans_v = [args.trans_v]
    if args.kv_varlen is not None:
        kv_varlen = [args.kv_varlen]
    if args.quant_q_and_kv is not None:
        quant_q_and_kv = [args.quant_q_and_kv]
    if args.use_torch_flash_ref is not None:
        use_torch_flash_ref_options = [args.use_torch_flash_ref]
    if args.use_aot_impl is not None:
        use_aot_impl_options = [args.use_aot_impl]
    if args.context_partition_size is not None:
        context_partition_size_options = [args.context_partition_size]

    compute_types_quant_q_and_kv = []
    for ct in compute_types:
        for quant_q, quant_kv in quant_q_and_kv:
            compute_types_quant_q_and_kv.append([ct, quant_q, quant_kv])
    if len(COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS) > 0:
        compute_types_quant_q_and_kv = COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS
        for idx in range(len(compute_types_quant_q_and_kv)):
            if not isinstance(compute_types_quant_q_and_kv[idx][0], torch.dtype):
                compute_types_quant_q_and_kv[idx][0] = dtypes.d_dtypes[
                    compute_types_quant_q_and_kv[idx][0]
                ]

    # Process sample_rate argument
    sample_rate = args.sample_rate if hasattr(args, "sample_rate") else 1.0

    return (
        block_sizes,
        head_configs,
        context_lengths,
        batch_sizes,
        head_sizes,
        query_lengths,
        quant_mode,
        trans_v,
        kv_varlen,
        compute_types_quant_q_and_kv,
        use_torch_flash_ref_options,
        use_aot_impl_options,
        context_partition_size_options,
        sample_rate,
        sinks_options,
        sliding_window_options,
        ps_options,
    )


def _run_single_test(args):
    """
    Helper function to run a single test case.

    Args:
        args: Tuple containing (test_config, current, total)

    Returns:
        Dictionary containing test results
    """
    test_config, current, total = args

    print(
        f"\n[{current}/{total}] Testing: "
        f"use_torch_flash_ref={test_config['use_torch_flash_ref']}, "
        f"compute_type={test_config['compute_type']}, "
        f"quant_q_and_kv=({test_config['quant_q']}, {test_config['quant_kv']}), "
        f"use_aot_impl={test_config['use_aot_impl']}, "
        f"trans_v={test_config['trans_v']}, "
        f"kv_varlen={test_config['kv_varlen']}, "
        f"context_partition_size={test_config['context_partition_size']}, "
        f"quant_mode={test_config['quant_mode']}, "
        f"block_size={test_config['block_size']}, "
        f"num_heads={test_config['num_heads']}, "
        f"context_lengths={test_config['context_length']}, "
        f"batch_size={test_config['batch_size']}, "
        f"query_length={test_config['query_length']}, "
        f"head_size={test_config['head_size']}, "
        f"sinks={test_config['sinks']}, "
        f"sliding_window={test_config['sliding_window']},"
        f"ps={test_config['ps']}"
    )

    # Import global variables to modify them
    global USE_TORCH_FLASH_REF
    USE_TORCH_FLASH_REF = test_config["use_torch_flash_ref"]
    if test_config["sinks"] or test_config["sliding_window"] > 0:
        USE_TORCH_FLASH_REF = False

    result = run_pa_gluon_test(
        context_length=test_config["context_length"],
        batch_size=test_config["batch_size"],
        num_heads=test_config["num_heads"],
        head_size=test_config["head_size"],
        block_size=test_config["block_size"],
        compute_type=test_config["compute_type"],
        query_length=test_config["query_length"],
        quant_mode=test_config["quant_mode"],
        context_partition_size=test_config["context_partition_size"],
        trans_v=test_config["trans_v"],
        kv_varlen=test_config["kv_varlen"],
        use_aot_impl=test_config["use_aot_impl"],
        quant_q=test_config["quant_q"],
        quant_kv=test_config["quant_kv"],
        use_sinks=test_config["sinks"],
        sliding_window=test_config["sliding_window"],
        ps=test_config["ps"],
    )

    return result


def run_multi_pa_gluon_test(
    block_sizes,
    head_configs,
    context_lengths,
    batch_sizes,
    head_sizes,
    query_lengths,
    quant_mode,
    trans_v,
    kv_varlen,
    compute_types_quant_q_and_kv,
    use_torch_flash_ref_options,
    use_aot_impl_options,
    context_partition_size_options,
    sample_rate=1.0,
    sinks_options=[False],
    sliding_window_options=[0, 128],
    ps_options=[False],
) -> pd.DataFrame:
    """Run all tests."""
    # Generate all test configurations
    test_configs = []

    for use_torch_flash_ref in use_torch_flash_ref_options:
        for hc in head_configs:
            for ct, quant_q, quant_kv in compute_types_quant_q_and_kv:
                for trans_v_mode in trans_v:
                    for kv_varlen_mode in kv_varlen:
                        for context_partition_size in context_partition_size_options:
                            qm_cnt = 0
                            for qm in quant_mode:
                                qm_cnt += 1
                                if not quant_q and not quant_kv and qm_cnt > 1:
                                    continue
                                for bs in block_sizes:
                                    for head_size in head_sizes:
                                        for ql in query_lengths:
                                            for bsz in batch_sizes:
                                                for cl in context_lengths:
                                                    for (
                                                        use_aot_impl
                                                    ) in use_aot_impl_options:
                                                        for sinks in sinks_options:
                                                            for (
                                                                sliding_window
                                                            ) in sliding_window_options:
                                                                for ps in ps_options:
                                                                    test_config = {
                                                                        "use_torch_flash_ref": use_torch_flash_ref,
                                                                        "compute_type": ct,
                                                                        "quant_q": quant_q,
                                                                        "quant_kv": quant_kv,
                                                                        "trans_v": trans_v_mode,
                                                                        "kv_varlen": kv_varlen_mode,
                                                                        "context_partition_size": context_partition_size,
                                                                        "quant_mode": qm,
                                                                        "block_size": bs,
                                                                        "num_heads": hc,
                                                                        "context_length": cl,
                                                                        "batch_size": bsz,
                                                                        "query_length": ql,
                                                                        "head_size": head_size,
                                                                        "use_aot_impl": use_aot_impl,
                                                                        "sinks": sinks,
                                                                        "sliding_window": sliding_window,
                                                                        "ps": ps,
                                                                    }
                                                                    test_configs.append(
                                                                        test_config
                                                                    )
    total = len(test_configs)
    print(f"\nTotal test cases: {total}")

    # Run tests with random sampling
    if sample_rate < 1.0:
        # Random sampling: each test case has sample_rate probability to be selected
        test_configs_to_run = [
            config for config in test_configs if random.random() < sample_rate
        ]
        print(
            f"Using random sampling: running {len(test_configs_to_run)} out of {total} test cases (sample_rate={sample_rate:.2%})"
        )
    else:
        test_configs_to_run = test_configs
        print(f"Running all {total} test cases (sample_rate=100%)")

    results = []
    for idx, test_config in enumerate(test_configs_to_run):
        result = _run_single_test((test_config, idx + 1, len(test_configs_to_run)))
        results.append(result)

    return pd.DataFrame(results)


def parse_arg_and_run_test(sample_rate0: float = None):
    """Parse arguments and run tests."""
    print(f"Triton location: {triton}")
    print(f"Triton version: {triton.__version__}")

    parser = create_argument_parser()
    # When running via pytest, use empty args to avoid conflict with pytest's argv
    running_via_pytest = "pytest" in sys.argv[0] or sys.argv[0].endswith("py.test")
    if running_via_pytest:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    (
        block_sizes,
        head_configs,
        context_lengths,
        batch_sizes,
        head_sizes,
        query_lengths,
        quant_mode,
        trans_v,
        kv_varlen,
        compute_types_quant_q_and_kv,
        use_torch_flash_ref_options,
        use_aot_impl_options,
        context_partition_size_options,
        sample_rate1,
        sinks_options,
        sliding_window_options,
        ps_options,
    ) = process_arguments(args)
    if sample_rate0 is None:
        sample_rate = sample_rate1
    else:
        sample_rate = sample_rate0

    results_df = run_multi_pa_gluon_test(
        block_sizes,
        head_configs,
        context_lengths,
        batch_sizes,
        head_sizes,
        query_lengths,
        quant_mode,
        trans_v,
        kv_varlen,
        compute_types_quant_q_and_kv,
        use_torch_flash_ref_options,
        use_aot_impl_options,
        context_partition_size_options,
        sample_rate,
        sinks_options,
        sliding_window_options,
        ps_options,
    )

    output_file = f"run_pa_gluon_test.{TEST_NAME}.block_size_{block_sizes[0]}.triton.{TRITON_VERSION}.csv"
    results_df.to_csv(output_file, index=False)

    print(f"\nResults saved to {output_file}")
    print(f"\nSummary:\n{results_df}")

    # Print mean of selected columns grouped by compute_type
    columns_to_print_mean = [
        "us_gluon",
        "gluon_bandwith(TB/s)",
        "us_asm",
        "asm_bandwith(TB/s)",
        "perf_gluon_vs_asm",
    ]

    def compute_column_mean(col_data):
        """Compute mean for a column, handling NaN, empty values, and percentage strings."""
        if pd.api.types.is_numeric_dtype(col_data):
            valid_data = col_data.dropna()
            if len(valid_data) == 0:
                return None, False
            return valid_data.mean(), False
        else:
            # Handle string columns (e.g., "150%", "NaN")
            valid_data = col_data[
                col_data.notna() & (col_data != "") & (col_data.astype(str) != "NaN")
            ]
            if len(valid_data) == 0:
                return None, False
            try:
                numeric_vals = valid_data.astype(str).str.rstrip("%").astype(float)
                is_percentage = valid_data.iloc[0].endswith("%")
                return numeric_vals.mean(), is_percentage
            except (ValueError, AttributeError):
                return None, False

    # Check if DataFrame is empty or missing required column
    if results_df.empty or "compute_type" not in results_df.columns:
        print("\nNo test results to analyze (empty DataFrame).")
        return

    # Get unique compute_types
    compute_types = results_df["compute_type"].unique()

    # Filter columns that exist in the dataframe
    valid_columns = [col for col in columns_to_print_mean if col in results_df.columns]

    if valid_columns:
        print("\n=== Selected Columns Mean by Compute Type ===")

        # Build the mean table
        mean_table = {}
        is_percentage_col = {}

        for ct in compute_types:
            ct_data = results_df[results_df["compute_type"] == ct]
            mean_table[str(ct)] = {}
            for col in valid_columns:
                mean_val, is_pct = compute_column_mean(ct_data[col])
                mean_table[str(ct)][col] = mean_val
                if col not in is_percentage_col:
                    is_percentage_col[col] = is_pct

        # Print header
        col_widths = {col: max(len(col), 12) for col in valid_columns}
        ct_width = max(len(str(ct)) for ct in compute_types)
        ct_width = max(ct_width, len("compute_type"))

        header = f"{'compute_type':<{ct_width}}"
        for col in valid_columns:
            header += f"  {col:>{col_widths[col]}}"
        print(header)
        print("-" * len(header))

        # Print rows for each compute_type
        for ct in compute_types:
            row = f"{str(ct):<{ct_width}}"
            for col in valid_columns:
                val = mean_table[str(ct)][col]
                if val is None:
                    row += f"  {'N/A':>{col_widths[col]}}"
                elif is_percentage_col.get(col, False):
                    row += f"  {val:>{col_widths[col]}.2f}%"
                else:
                    row += f"  {val:>{col_widths[col]}.4f}"
            print(row)

    # Check if all tests passed
    total_errors = results_df["err_gluon"].sum()
    if total_errors > 0:
        print(
            f"\nTests failed! {total_errors} test case(s) exceeded the error threshold. "
        )
        print(f"Please check rows with non-zero err_gluon in {output_file}.")
        assert False, f"{total_errors} test case(s) exceeded the error threshold"
    else:
        print("\nAll tests passed!")


def normal_accuracy_test():
    """Run normal accuracy test."""
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPE_OPTIONS
    global QUANT_MODE_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global QUANT_Q_AND_KV_OPTIONS
    global USE_TORCH_FLASH_REF_OPTIONS
    global USE_AOT_IMPL_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global SINKS_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS
    global PS_OPTIONS

    USE_AOT_IMPL_OPTIONS = [False]
    SINKS_OPTIONS = [False]
    SLIDING_WINDOW_OPTIONS = [0]
    PS_OPTIONS = [False]
    USE_TORCH_FLASH_REF_OPTIONS = [False]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]

    HEAD_DIMENSION_OPTIONS = [128]
    HEAD_CONFIGURATIONS = [(5, 1), (8, 1), (10, 1), (16, 1)]
    QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = [["fp8", True, True], ["bf16", False, False]]
    QUANT_MODE_OPTIONS = ["per_token", "per_tensor"]
    CONTEXT_LENGTH_OPTIONS = [1027]
    BATCH_SIZE_OPTIONS = [3, 81]
    TRANS_V_OPTIONS = [False]
    KV_VARLEN_OPTIONS = [False, True]
    BLOCK_SIZE_OPTIONS = [16, 64, 1024]
    parse_arg_and_run_test()

    # Test for different head dimensions
    HEAD_DIMENSION_OPTIONS = [64, 192, 256]
    HEAD_CONFIGURATIONS = [(8, 1)]
    QUERY_LENGTH_OPTIONS = [1, 3]
    QUANT_MODE_OPTIONS = ["per_token"]
    BATCH_SIZE_OPTIONS = [81]
    KV_VARLEN_OPTIONS = [True]
    parse_arg_and_run_test()


def normal_accuracy_aot_test():
    """Run normal accuracy aot test."""
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPE_OPTIONS
    global QUANT_MODE_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global QUANT_Q_AND_KV_OPTIONS
    global USE_TORCH_FLASH_REF_OPTIONS
    global USE_AOT_IMPL_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global SINKS_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS
    global PS_OPTIONS

    prebuild_normal_accuracy_cases_aot_so()
    get_so_files_size_and_count()
    USE_AOT_IMPL_OPTIONS = [True]

    SINKS_OPTIONS = [False]
    SLIDING_WINDOW_OPTIONS = [0]
    PS_OPTIONS = [False]
    USE_TORCH_FLASH_REF_OPTIONS = [False]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]

    HEAD_DIMENSION_OPTIONS = [128]
    HEAD_CONFIGURATIONS = [(5, 1), (8, 1), (10, 1), (16, 1)]
    QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = [["fp8", True, True], ["bf16", False, False]]
    QUANT_MODE_OPTIONS = ["per_token", "per_tensor"]
    CONTEXT_LENGTH_OPTIONS = [1027]
    BATCH_SIZE_OPTIONS = [3, 81]
    TRANS_V_OPTIONS = [False]
    KV_VARLEN_OPTIONS = [False, True]
    BLOCK_SIZE_OPTIONS = [16, 64, 1024]
    parse_arg_and_run_test()

    # Test for different head dimensions
    HEAD_DIMENSION_OPTIONS = [64, 192, 256]
    HEAD_CONFIGURATIONS = [(8, 1)]
    QUERY_LENGTH_OPTIONS = [1, 3]
    QUANT_MODE_OPTIONS = ["per_token"]
    BATCH_SIZE_OPTIONS = [81]
    KV_VARLEN_OPTIONS = [True]
    parse_arg_and_run_test()


def normal_performance_test():
    """Run normal performance test."""
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPE_OPTIONS
    global QUANT_MODE_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global QUANT_Q_AND_KV_OPTIONS
    global USE_TORCH_FLASH_REF_OPTIONS
    global USE_AOT_IMPL_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global SINKS_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS
    global PS_OPTIONS

    USE_AOT_IMPL_OPTIONS = [False]
    SINKS_OPTIONS = [False]
    SLIDING_WINDOW_OPTIONS = [0]
    PS_OPTIONS = [False]
    USE_TORCH_FLASH_REF_OPTIONS = [False]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]

    HEAD_DIMENSION_OPTIONS = [128]
    CONTEXT_LENGTH_OPTIONS = [2048, 4096, 8192]
    BATCH_SIZE_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128]
    QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    # COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = [["fp8", True, True], ["bf16", False, False]]
    QUANT_MODE_OPTIONS = ["per_tensor"]
    TRANS_V_OPTIONS = [False]
    KV_VARLEN_OPTIONS = [False]
    HEAD_CONFIGURATIONS = [(64, 4), (64, 8)]
    BLOCK_SIZE_OPTIONS = [16]
    parse_arg_and_run_test()
    BLOCK_SIZE_OPTIONS = [64]
    # parse_arg_and_run_test()


def normal_performance_aot_test():
    """Run normal performance aot test."""
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPE_OPTIONS
    global QUANT_MODE_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global QUANT_Q_AND_KV_OPTIONS
    global USE_TORCH_FLASH_REF_OPTIONS
    global USE_AOT_IMPL_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global SINKS_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS
    global PS_OPTIONS

    prebuild_normal_performance_cases_aot_so()
    get_so_files_size_and_count()
    USE_AOT_IMPL_OPTIONS = [True]

    SINKS_OPTIONS = [False]
    SLIDING_WINDOW_OPTIONS = [0]
    PS_OPTIONS = [False]
    USE_TORCH_FLASH_REF_OPTIONS = [False]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]

    HEAD_DIMENSION_OPTIONS = [128]
    CONTEXT_LENGTH_OPTIONS = [2048, 4096, 8192]
    BATCH_SIZE_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128]
    QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = [["fp8", True, True], ["bf16", False, False]]
    QUANT_MODE_OPTIONS = ["per_tensor"]
    TRANS_V_OPTIONS = [False]
    KV_VARLEN_OPTIONS = [False]
    HEAD_CONFIGURATIONS = [(64, 4), (64, 8)]
    BLOCK_SIZE_OPTIONS = [16]
    parse_arg_and_run_test()
    BLOCK_SIZE_OPTIONS = [64]
    parse_arg_and_run_test()


def sliding_window_accuracy_test():
    """Run sliding window accuracy test."""
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global SINKS_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global QUANT_Q_AND_KV_OPTIONS
    global USE_TORCH_FLASH_REF_OPTIONS
    global USE_AOT_IMPL_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global PS_OPTIONS
    global QUANT_MODE_OPTIONS

    USE_TORCH_FLASH_REF_OPTIONS = [False]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]

    SINKS_OPTIONS = [True, False]
    SLIDING_WINDOW_OPTIONS = [0, 128]
    HEAD_DIMENSION_OPTIONS = [64]
    CONTEXT_LENGTH_OPTIONS = [1024, 8192]
    BATCH_SIZE_OPTIONS = [1, 4, 128]
    QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = [["bf16", False, True]]
    QUANT_MODE_OPTIONS = ["per_token"]
    TRANS_V_OPTIONS = [False]
    KV_VARLEN_OPTIONS = [True]
    HEAD_CONFIGURATIONS = [(64, 8)]
    USE_AOT_IMPL_OPTIONS = [False]
    PS_OPTIONS = [True]
    BLOCK_SIZE_OPTIONS = [16]
    parse_arg_and_run_test()
    # BLOCK_SIZE_OPTIONS = [64]
    # parse_arg_and_run_test()


def sliding_window_performance_test():
    """Run sliding window performance test."""
    global BLOCK_SIZE_OPTIONS
    global QUERY_LENGTH_OPTIONS
    global BATCH_SIZE_OPTIONS
    global HEAD_CONFIGURATIONS
    global CONTEXT_LENGTH_OPTIONS
    global COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS
    global QUANT_MODE_OPTIONS
    global HEAD_DIMENSION_OPTIONS
    global SINKS_OPTIONS
    global SLIDING_WINDOW_OPTIONS
    global TRANS_V_OPTIONS
    global KV_VARLEN_OPTIONS
    global USE_TORCH_FLASH_REF_OPTIONS
    global USE_AOT_IMPL_OPTIONS
    global CONTEXT_PARTITION_SIZE_OPTIONS
    global PS_OPTIONS

    SINKS_OPTIONS = [False, True]
    SLIDING_WINDOW_OPTIONS = [0, 128]

    USE_TORCH_FLASH_REF_OPTIONS = [False]
    CONTEXT_PARTITION_SIZE_OPTIONS = [256]
    HEAD_DIMENSION_OPTIONS = [64]
    HEAD_CONFIGURATIONS = [(64, 8)]
    QUERY_LENGTH_OPTIONS = [1]
    COMPUTE_TYPES_QUANT_Q_AND_KV_OPTIONS = [["fp8", True, True], ["bf16", False, False]]
    QUANT_MODE_OPTIONS = ["per_tensor"]
    CONTEXT_LENGTH_OPTIONS = [1024]
    BATCH_SIZE_OPTIONS = [4, 128]
    TRANS_V_OPTIONS = [False]
    KV_VARLEN_OPTIONS = [True]
    USE_AOT_IMPL_OPTIONS = [False]
    BLOCK_SIZE_OPTIONS = [16]
    PS_OPTIONS = [True]
    parse_arg_and_run_test()


@pytest.mark.parametrize("case_set_name", CASE_SET_NAME_OPTIONS)
def test_multi_case_set(case_set_name):
    if case_set_name == "normal_accuracy":
        normal_accuracy_test()
    elif case_set_name == "normal_accuracy_aot":
        normal_accuracy_aot_test()
    elif case_set_name == "normal_performance":
        normal_performance_test()
    elif case_set_name == "normal_performance_aot":
        normal_performance_aot_test()
    elif case_set_name == "sliding_window_accuracy":
        sliding_window_accuracy_test()
    elif case_set_name == "sliding_window_performance":
        sliding_window_performance_test()


if __name__ == "__main__":
    # normal_accuracy_test()
    # normal_accuracy_aot_test()
    normal_performance_test()
    # normal_performance_aot_test()
    # sliding_window_accuracy_test()
    # sliding_window_performance_test()

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import os
import sys
import argparse
import random
from typing import List, Optional, Tuple, Union, Dict
import hashlib

import pandas as pd
import numpy as np
import torch
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter import pertoken_quant, per_tensor_quant
from aiter.test_common import benchmark, checkAllclose, perftest

from aiter.ops.triton.gluon.pa_decode_triton_gluon_fp8 import (
    paged_attention_decode as paged_attention_decode_gluon_fp8,
)

TRITON_VERSION = triton.__version__

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

# Triton to PyTorch dtype mapping
TL_TO_TORCH_DTYPE = {tl.bfloat16: torch.bfloat16, tl.float16: torch.float16}
TORCH_TO_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


def setup_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(123)


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

    # print("\nTop differences:")
    # for item in result['top_k_diff']:
    #     print(f"Position {item['position']}: arr1 = {arr1[item['position']]:.6f}, arr2 = {arr2[item['position']]:.6f}, Diff = {item['value']:.6f}")

    # print("\nThreshold statistics:")
    # for stat in result['threshold_stats']:
    #     print(f"{stat['range']}: {stat['count']} ({stat['percentage']:.2f}%)")

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
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Create key and value cache tensors."""
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)
    elements_per_vector = 16 // torch_dtype.itemsize
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
) -> torch.Tensor:
    """Reference implementation of masked attention."""
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]

    key = key.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    value = value.repeat_interleave(num_query_heads // num_kv_heads, dim=1)

    attention_scores = (
        torch.einsum("qhd,khd->hqk", query.float(), key.float()) * softmax_scale
    )

    if is_causal:
        query_len = query.shape[0]
        key_len = key.shape[0]
        attention_bias = torch.zeros(
            query_len, key_len, dtype=query.dtype, device=query.device
        )
        causal_mask = torch.ones(
            query_len, key_len, dtype=torch.bool, device=query.device
        ).tril(diagonal=key_len - query_len)
        attention_bias.masked_fill_(causal_mask.logical_not(), float("-inf"))
        attention_scores += attention_bias

    attention_weights = torch.softmax(attention_scores, dim=-1)
    output = torch.einsum("hqk,khd->qhd", attention_weights.float(), value.float())
    return output.to(output_dtype)


def torch_mha_extend(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    query_output_indptr: torch.Tensor,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
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
        current_context_length = sequence_lengths[batch_idx].item()

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
        )
        outputs.append(attention_output)

    return torch.cat(outputs)


def quantize_kv_cache_symmetric(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
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
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    quant_dtype: torch.dtype,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Apply per-tensor quantization to KV cache."""
    num_blocks, num_heads, head_dim, block_size = value_cache.shape

    # Per-tensor quantization for keys
    quantized_keys, key_scale = per_tensor_quant(key_cache, quant_dtype=quant_dtype)
    # Per-tensor quantization for values
    quantized_values, value_scale = per_tensor_quant(
        value_cache, quant_dtype=quant_dtype
    )

    # For per-tensor quantization, scales are scalars
    key_scales_flat = key_scale.expand(num_heads, num_blocks * block_size)
    value_scales_flat = value_scale.expand(num_heads, num_blocks * block_size)

    # Create proper shape for original scales (per-tensor uses scalar broadcasted to appropriate shape)
    key_scales_original = key_scale.squeeze()
    value_scales_original = value_scale.squeeze()

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
    sequence_lengths: torch.Tensor,
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
        sequence_lengths,
        block_tables_stride0,
        max_query_length,
        key_scale,
        value_scale,
        None,
        query_output_indptr,
    )


def shuffle_value_cache_layout(value_cache: torch.Tensor) -> torch.Tensor:
    """Shuffle value cache layout for optimized memory access."""
    elements_per_vector = 16 // value_cache.element_size()
    num_blocks, num_kv_heads, head_size, block_size = value_cache.shape

    value_cache_reshaped = value_cache.view(
        num_blocks,
        num_kv_heads,
        head_size,
        block_size // elements_per_vector,
        elements_per_vector,
    )

    value_cache_shuffled = value_cache_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    return value_cache_shuffled


def run_gluon_fp8_kernel(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    sequence_lengths: torch.Tensor,
    block_tables: torch.Tensor,
    attention_scale: float,
    query_sequence_length: int,
    max_sequence_length: int,
    compute_type: tl.dtype,
    query_scale: torch.Tensor,
    key_scale: torch.Tensor,
    value_scale: torch.Tensor,
    num_sequence_partitions: int = 0,
    alibi_slopes: Optional[torch.Tensor] = None,
) -> Dict:
    """Run Gluon FP8 kernel for paged attention."""
    result = paged_attention_decode_gluon_fp8(
        output,
        query,
        key_cache,
        value_cache,
        sequence_lengths,
        block_tables,
        attention_scale,
        query_sequence_length,
        max_sequence_length,
        compute_type,
        query_scale,
        key_scale,
        value_scale,
        num_sequence_partitions=0,
        alibi_slopes=None,
    )
    return result


@benchmark()
def test_paged_attention_decode_gluon(
    context_lengths: int,
    batch_size: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    data_type: torch.dtype,
    query_length: int,
    trans_v: bool,
    quant_mode: str = "per_token",  # "per_token" or "per_tensor"
) -> Dict[str, Union[float, str]]:
    """Test paged attention decode with assembly and gluon implementations."""
    results = {}
    seed = 0
    device = "cuda:0"
    torch.set_default_device(device)
    num_query_heads, num_kv_heads = num_heads

    assert (
        num_query_heads % num_kv_heads == 0
    ), "Query heads must be divisible by KV heads"

    max_sequence_length = 16384
    max_blocks_per_sequence = (max_sequence_length + block_size - 1) // block_size
    total_blocks = max_blocks_per_sequence * batch_size
    blocks_per_sequence = (context_lengths + block_size - 1) // block_size

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

    sequence_lengths = torch.tensor(
        [context_lengths] * batch_size, dtype=torch.int32, device=device
    )

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
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    softmax_scale = 1.0 / (head_size**0.5)

    # Quantization based on mode
    if quant_mode == "per_token":
        # Per-token quantization for query
        quantized_query, query_scale_factors = pertoken_quant(
            query, quant_dtype=aiter.dtypes.fp8
        )

        # Per-token quantization for KV cache
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
    else:  # per_tensor
        # Per-tensor quantization for query
        quantized_query, query_scale = per_tensor_quant(
            query, quant_dtype=aiter.dtypes.fp8
        )
        query_scale_factors = query_scale.squeeze()

        # Per-tensor quantization for KV cache
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

    # Reference
    reference_output_quant = torch_mha_extend(
        query,
        quantized_keys,
        quantized_values,
        block_tables,
        sequence_lengths,
        query_output_indptr,
        key_scale_factors_flat,
        value_scale_factors_flat,
    )

    if trans_v:
        quantized_values = shuffle_value_cache_layout(quantized_values)
        print(f"Transformed quantized_values.shape={quantized_values.shape}")

    fp8_tolerance = 5e-2

    # Prepare for Gluon kernel
    quantized_query_gluon = quantized_query
    query_scale_gluon = query_scale_factors
    gluon_output = torch.empty_like(reference_output_quant)

    if query_length > 1:
        query_group_size = num_query_heads // num_kv_heads

        quantized_query_gluon = quantized_query.reshape(
            batch_size, query_length, num_kv_heads, query_group_size, head_size
        )
        quantized_query_gluon = quantized_query_gluon.transpose(1, 2).reshape(
            batch_size, num_kv_heads * query_length * query_group_size, head_size
        )

        gluon_output = gluon_output.reshape(
            batch_size, query_length, num_kv_heads, query_group_size, head_size
        )
        gluon_output = gluon_output.transpose(1, 2).reshape(
            batch_size, num_kv_heads * query_length * query_group_size, head_size
        )

        if len(query_scale_factors.shape) > 0:  # per-token quantization
            query_scale_gluon = query_scale_factors.reshape(
                batch_size, query_length, num_kv_heads, query_group_size, 1
            )
            query_scale_gluon = query_scale_gluon.transpose(1, 2).reshape(
                batch_size, num_kv_heads * query_length * query_group_size, 1
            )
        else:  # per-tensor quantization
            # For per-tensor, query_scale is already a scalar, no need to reshape
            pass

    # Test Gluon
    gluon_results = run_gluon_fp8_kernel(
        gluon_output,
        quantized_query_gluon,
        quantized_keys,
        quantized_values,
        sequence_lengths,
        block_tables,
        softmax_scale,
        query_length,
        sequence_lengths.max().item(),
        TORCH_TO_TL_DTYPE[data_type],
        query_scale=query_scale_gluon,
        key_scale=key_scale_original,
        value_scale=value_scale_original,
        num_sequence_partitions=0,
        alibi_slopes=None,
    )

    final_gluon_output = gluon_output
    if query_length > 1:
        final_gluon_output = gluon_output.reshape(
            batch_size, num_kv_heads, query_length, query_group_size, head_size
        )
        final_gluon_output = final_gluon_output.transpose(1, 2).reshape(
            batch_size * query_length, num_kv_heads * query_group_size, head_size
        )

    gluon_time = gluon_results["total_triton_time"]
    gluon_error = checkAllclose(
        reference_output_quant,
        final_gluon_output,
        atol=fp8_tolerance,
        rtol=fp8_tolerance,
        msg=f"[PyTorch vs Gluon_FP8][{quant_mode}]: {gluon_time:>8.2f} us......",
    )
    compare_arrays(
        final_gluon_output.to(torch.float32).detach().cpu().numpy(),
        reference_output_quant.to(torch.float32).detach().cpu().numpy(),
    )
    results[f"us_gluon_fp8"] = gluon_time
    results[f"err_gluon_fp8"] = gluon_error

    # MD5 hash
    reference_hash = hashlib.md5(
        reference_output_quant.contiguous()
        .view(torch.uint8)
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    gluon_hash = hashlib.md5(
        final_gluon_output.contiguous()
        .view(torch.uint8)
        .detach()
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()
    print(f"out_ref_md5={reference_hash}")
    print(f"gluon_fp8_output_md5={gluon_hash}")

    # Bandwidth
    kernel_time_us = gluon_time
    bandwidth_tb_per_sec = (
        batch_size
        * head_size
        * (
            2 * context_lengths * num_kv_heads * quantized_keys.dtype.itemsize
            + 2 * query_length * num_query_heads * quantized_query.dtype.itemsize
        )
        / (kernel_time_us * 1e6 * 1.024**4)
    )
    results[f"gluon_fp8_bandwith(TB/s)"] = bandwidth_tb_per_sec

    # Test Assembly
    query_group_size = num_query_heads // num_kv_heads
    skip_assembly = (
        (block_size == 1024 and num_heads != (10, 1))
        or (block_size == 16 and query_group_size == 8 and query_length == 3)
        or (context_lengths == 512 and query_group_size == 5 and query_length == 3)
        or (block_size == 64)
    )

    # aiter_assembly_kernel do not support per-tensor quantization, we always use per-token quantization here
    if not skip_assembly:
        assembly_output, assembly_time = run_aiter_assembly_kernel(
            query,
            quantized_keys,
            quantized_values,
            block_tables,
            sequence_lengths,
            block_tables.size(1),
            max_query_length,
            key_scale_factors_flat.contiguous(),
            value_scale_factors_flat.contiguous(),
            query_output_indptr,
        )
        assembly_error = checkAllclose(
            reference_output_quant,
            assembly_output,
            atol=fp8_tolerance,
            rtol=fp8_tolerance,
            msg=f"[PyTorch vs AIT_Assembly][{quant_mode}]: {assembly_time:>8.2f} us......",
        )
        compare_arrays(
            assembly_output.to(torch.float32).detach().cpu().numpy(),
            reference_output_quant.to(torch.float32).detach().cpu().numpy(),
        )
        results[f"us_asm_fp8"] = assembly_time
        assembly_bandwidth = (
            batch_size
            * head_size
            * (
                2 * context_lengths * num_kv_heads * quantized_keys.dtype.itemsize
                + 2 * query_length * num_query_heads * query.dtype.itemsize
            )
            / (assembly_time * 1e6 * 1.024**4)
        )
        results[f"asm_fp8_bandwith(TB/s)"] = assembly_bandwidth

    if f"us_asm_fp8" in results:
        results[f"perf_fp8_gluon_vs_asm"] = (
            f'{results[f"us_asm_fp8"] / results[f"us_gluon_fp8"]:.0%}'
        )
    else:
        results[f"perf_fp8_gluon_vs_asm"] = "NaN"

    print(f"Triton version: {triton.__version__}")
    print(f"Quantization mode: {quant_mode}")
    sys.stdout.flush()

    return results


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test paged attention decode gluon implementation",
    )

    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=DATA_TYPE_OPTIONS,
        default=None,
        help="Data type",
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
    parser.add_argument("--block_size", type=int, default=None, help="Block size")
    parser.add_argument(
        "--trans_v", action="store_true", help="Transpose value cache layout"
    )
    parser.add_argument(
        "--quant_mode",
        type=str,
        choices=["per_token", "per_tensor", "both"],
        default="both",
        help="Quantization mode: per_token, per_tensor, or both",
    )

    return parser


def process_arguments(args: argparse.Namespace) -> tuple:
    """Process command line arguments."""
    data_types = DATA_TYPE_OPTIONS
    block_sizes = BLOCK_SIZE_OPTIONS
    head_configs = HEAD_CONFIGURATIONS
    context_lengths = CONTEXT_LENGTH_OPTIONS
    batch_sizes = BATCH_SIZE_OPTIONS
    query_lengths = QUERY_LENGTH_OPTIONS

    if args.dtype is not None:
        data_types = [dtypes.d_dtypes[args.dtype]]
    else:
        data_types = [dtypes.d_dtypes[key] for key in data_types]

    if args.num_heads is not None:
        head_configs = [args.num_heads]
    if args.query_length is not None:
        query_lengths = [args.query_length]
    if args.context_length is not None:
        context_lengths = [args.context_length]
    if args.batch_size is not None:
        batch_sizes = [args.batch_size]
    if args.block_size is not None:
        block_sizes = [args.block_size]

    return (
        data_types,
        block_sizes,
        head_configs,
        context_lengths,
        batch_sizes,
        query_lengths,
        args.trans_v,
        args.quant_mode,
    )


def run_tests(
    data_types,
    block_sizes,
    head_configs,
    context_lengths,
    batch_sizes,
    query_lengths,
    trans_v,
    quant_mode,
) -> pd.DataFrame:
    """Run all tests."""
    results = []
    total = (
        len(data_types)
        * len(block_sizes)
        * len(head_configs)
        * len(context_lengths)
        * len(batch_sizes)
        * len(query_lengths)
    )
    current = 0

    # Determine which quantization modes to test
    if quant_mode == "both":
        quant_modes_to_test = ["per_token", "per_tensor"]
    else:
        quant_modes_to_test = [quant_mode]

    for qm in quant_modes_to_test:
        for dt in data_types:
            for bs in block_sizes:
                for hc in head_configs:
                    for cl in context_lengths:
                        for bsz in batch_sizes:
                            for ql in query_lengths:
                                current += 1
                                print(
                                    f"\n[{current}/{total * len(quant_modes_to_test)}] Testing: dtype={dt}, block_size={bs}, heads={hc}, ctx_len={cl}, batch={bsz}, qlen={ql}, quant_mode={qm}"
                                )
                                result = test_paged_attention_decode_gluon(
                                    context_lengths=cl,
                                    batch_size=bsz,
                                    num_heads=hc,
                                    head_size=HEAD_DIMENSION,
                                    block_size=bs,
                                    data_type=dt,
                                    query_length=ql,
                                    trans_v=trans_v,
                                    quant_mode=qm,
                                )
                                results.append(result)

    return pd.DataFrame(results)


def main():
    """Main function."""
    parser = create_argument_parser()
    args = parser.parse_args()

    (
        data_types,
        block_sizes,
        head_configs,
        context_lengths,
        batch_sizes,
        query_lengths,
        trans_v,
        quant_mode,
    ) = process_arguments(args)

    results_df = run_tests(
        data_types,
        block_sizes,
        head_configs,
        context_lengths,
        batch_sizes,
        query_lengths,
        trans_v,
        quant_mode,
    )

    output_file = f"test_paged_attention_decode_gluon{'_trans_v' if trans_v else ''}.triton.{TRITON_VERSION}.quant_{quant_mode}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    print(f"\nSummary:\n{results_df}")


if __name__ == "__main__":
    # Configuration parameters
    # HEAD_DIMENSION = 128
    # BLOCK_SIZE_OPTIONS = [16, 64, 1024]
    # DATA_TYPE_OPTIONS = ["bf16"]
    # HEAD_CONFIGURATIONS = [(5, 1), (8, 1), (10, 1), (16, 1), (64, 4)]
    # QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    # CONTEXT_LENGTH_OPTIONS = [512, 4096, 4097]
    # BATCH_SIZE_OPTIONS = [4, 80, 128]

    HEAD_DIMENSION = 128
    BLOCK_SIZE_OPTIONS = [16]
    # BLOCK_SIZE_OPTIONS = [1024]
    DATA_TYPE_OPTIONS = ["bf16"]
    HEAD_CONFIGURATIONS = [(16, 1)]
    # HEAD_CONFIGURATIONS = [(10, 1)]
    QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
    CONTEXT_LENGTH_OPTIONS = [4096]
    BATCH_SIZE_OPTIONS = [4, 128]

    main()

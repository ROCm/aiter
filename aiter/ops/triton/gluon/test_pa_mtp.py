# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import os
import sys
import argparse
import random
from typing import List, Optional, Tuple, Union, Dict
import hashlib
import numpy as np

import pandas as pd
import torch
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter import paged_attn as ops
from aiter import pertoken_quant
from aiter.test_common import benchmark, checkAllclose, perftest

from utils import compare_arrays
from pa_decode_triton_fp8 import paged_attention_decode as paged_attention_decode_triton_fp8
from pa_decode_triton_gluon_fp8 import paged_attention_decode as paged_attention_decode_gluon_fp8


TRITON_VERSION=triton.__version__

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# Global configuration
UNIFORM_RANGE = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": torch.bfloat16,
    "float": torch.float,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}

# Triton to PyTorch dtype mapping and vice versa
TL_TO_TORCH_DTYPE = {tl.bfloat16: torch.bfloat16, tl.float16: torch.float16}
TORCH_TO_TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}

# Configuration parameters for comprehensive testing
HEAD_DIMENSION = 128
BLOCK_SIZE_OPTIONS = [16, 64, 1024]
DATA_TYPE_OPTIONS = ["bf16"]
HEAD_CONFIGURATIONS = [(5, 1), (8, 1), (10, 1), (16, 1), (64, 4)]
QUERY_LENGTH_OPTIONS = [1, 2, 3, 4]
CONTEXT_LENGTH_OPTIONS = [512, 4096, 4097]
BATCH_SIZE_OPTIONS = [4, 80, 128]


def setup_seed(seed: int) -> None:
    """
    Set random seed for reproducibility across all random number generators.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(123)


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    """
    Convert cache dtype specification to actual torch dtype.
    
    Args:
        cache_dtype: Cache data type specification (string or torch.dtype)
        model_dtype: Model data type for 'auto' inference
        
    Returns:
        torch.dtype: The resolved torch data type
        
    Raises:
        ValueError: If cache_dtype or model_dtype is invalid
    """
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
    """
    Create key and value cache tensors for paged attention.
    
    Args:
        num_blocks: Number of memory blocks
        block_size: Size of each memory block
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        head_size: Dimension of each attention head
        cache_dtype: Data type for the cache
        model_dtype: Model data type for 'auto' inference
        seed: Random seed for initialization
        device: Device to place tensors on
        
    Returns:
        Tuple containing:
            - List of key caches for each layer
            - List of value caches for each layer
            
    Raises:
        ValueError: If cache_dtype is fp8 and head_size is not divisible by 16
    """
    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    # Calculate elements per vector for key cache layout optimization
    elements_per_vector = 16 // torch_dtype.itemsize
    key_cache_shape = (num_blocks, num_heads, head_size // elements_per_vector, block_size, elements_per_vector)
    
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
        value_cache = torch.empty(size=value_cache_shape, dtype=torch_dtype, device=device)
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
    """
    Reference implementation of masked attention for verification.
    
    Args:
        query: Query tensor [sequence_len, num_heads, head_dim]
        key: Key tensor [sequence_len, num_kv_heads, head_dim]
        value: Value tensor [sequence_len, num_kv_heads, head_dim]
        softmax_scale: Scaling factor for attention scores
        output_dtype: Data type for output tensor
        is_causal: Whether to apply causal masking
        
    Returns:
        Output tensor [sequence_len, num_heads, head_dim]
    """
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    
    # Repeat KV heads to match Q heads (for MQA/GQA)
    key = key.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    value = value.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    
    # Compute attention scores: (num_heads, query_len, key_len)
    attention_scores = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * softmax_scale
    
    if is_causal:
        query_len = query.shape[0]
        key_len = key.shape[0]
        attention_bias = torch.zeros(query_len, key_len, dtype=query.dtype, device=query.device)
        causal_mask = torch.ones(query_len, key_len, dtype=torch.bool, device=query.device).tril(diagonal=key_len - query_len)
        attention_bias.masked_fill_(causal_mask.logical_not(), float("-inf"))
        attention_scores += attention_bias
        
    # Apply softmax to get attention weights
    attention_weights = torch.softmax(attention_scores, dim=-1)

    # Compute weighted sum of values
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
    """
    PyTorch implementation of paged multi-head attention for verification.
    
    Args:
        query: Query tensor [total_queries, num_heads, head_dim]
        key_cache: Key cache [num_blocks, num_heads, head_size // x, block_size, x]
        value_cache: Value cache [num_blocks, num_heads, head_size, block_size]
        block_tables: Block allocation tables [batch_size, max_blocks_per_sequence]
        sequence_lengths: Sequence lengths for each batch [batch_size]
        query_output_indptr: Indices pointer for query/output splitting [batch_size + 1]
        key_scale: Optional scaling factors for quantized keys [num_heads, total_tokens]
        value_scale: Optional scaling factors for quantized values [num_heads, total_tokens]
        
    Returns:
        Output tensor [total_queries, num_heads, head_dim]
    """
    num_blocks, num_heads, head_size, block_size = value_cache.shape
    softmax_scale = 1.0 / (head_size**0.5)

    output_dtype = query.dtype
    kv_dtype = key_cache.dtype
    
    # Split queries by batch using indptr
    queries_split = torch.tensor_split(query, query_output_indptr.tolist()[1:])

    # Reshape key cache for efficient indexing: [total_tokens, num_heads, head_size]
    key_cache_flat = key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, head_size)
    value_cache_flat = value_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_heads, head_size)

    batch_size = query_output_indptr.shape[0] - 1
    outputs = []
    
    for batch_idx in range(batch_size):
        current_query = queries_split[batch_idx]
        current_block_table = block_tables[batch_idx]
        current_context_length = sequence_lengths[batch_idx].item()

        # Compute token indices from block table
        token_indices = (
            current_block_table.repeat_interleave(block_size)[:current_context_length] * block_size
            + torch.arange(current_context_length, device=current_block_table.device) % block_size
        )

        # Gather keys and values for current sequence
        gathered_keys = key_cache_flat.view(torch.int8)[token_indices].view(kv_dtype).to(torch.float)
        if key_scale is not None:
            gathered_keys *= key_scale[:, token_indices].t().unsqueeze(-1)

        gathered_values = value_cache_flat.view(torch.int8)[token_indices].view(kv_dtype).to(torch.float)
        if value_scale is not None:
            gathered_values *= value_scale[:, token_indices].t().unsqueeze(-1)
            
        # Compute attention output
        attention_output = reference_masked_attention(
            current_query, gathered_keys, gathered_values, softmax_scale, output_dtype, is_causal=True
        )
        outputs.append(attention_output)
        
    return torch.cat(outputs)


def quantize_kv_cache_symmetric(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    quant_dtype: torch.dtype,
    scale_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply symmetric per-token quantization to key-value cache.
    
    Args:
        key_cache: Key cache [num_blocks, num_heads, head_size // x, block_size, x]
        value_cache: Value cache [num_blocks, num_heads, head_size, block_size]
        quant_dtype: Quantization data type (e.g., torch.float8_e4m3fnuz)
        scale_dtype: Data type for scaling factors
        
    Returns:
        Tuple containing:
            - Quantized key cache
            - Key scaling factors (flattened)
            - Quantized value cache  
            - Value scaling factors (flattened)
            - Key scaling factors (original shape)
            - Value scaling factors (original shape)
    """
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    total_tokens = num_blocks * block_size

    # Reshape caches for per-token quantization
    # Key cache: [num_blocks, num_heads, block_size, head_dim]
    key_cache_reshaped = (
        key_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    
    # Value cache: [num_blocks, num_heads, block_size, head_dim]  
    value_cache_reshaped = (
        value_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )

    # Apply per-token quantization
    quantized_keys, key_scales_original = pertoken_quant(key_cache_reshaped, quant_dtype=quant_dtype)
    quantized_values, value_scales_original = pertoken_quant(value_cache_reshaped, quant_dtype=quant_dtype)

    # Calculate elements per vector for quantized layout
    elements_per_vector = 16 // quant_dtype.itemsize

    # Reshape quantized keys back to original layout
    quantized_keys = (
        quantized_keys.view(num_blocks, num_heads, block_size, head_dim // elements_per_vector, elements_per_vector)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    
    # Flatten scaling factors for efficient access
    key_scales_flat = key_scales_original.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    
    # Reshape quantized values back to original layout
    quantized_values = (
        quantized_values.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    
    value_scales_flat = value_scales_original.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)

    return (
        quantized_keys, 
        key_scales_flat, 
        quantized_values, 
        value_scales_flat, 
        key_scales_original, 
        value_scales_original
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
    """
    Run the AIT assembly kernel for paged attention.
    
    Args:
        query: Query tensor
        key_cache: Key cache tensor
        value_cache: Value cache tensor  
        block_tables: Block tables for paged access
        sequence_lengths: Length of each sequence
        block_tables_stride0: Stride for block tables
        max_query_length: Maximum query length in the batch
        key_scale: Optional scaling factors for quantized keys
        value_scale: Optional scaling factors for quantized values
        query_output_indptr: Optional indices pointer for query/output
        
    Returns:
        Attention output tensor
    """
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
        None,  # Additional options parameter
        query_output_indptr,
    )


@perftest()
def run_aiter_hip_kernel(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    sequence_lengths: torch.Tensor,
    max_sequence_length: int,
    max_query_length: int,
    kv_cache_data_type: str,
    num_kv_heads: int,
    softmax_scale: float,
    key_scale: Optional[torch.Tensor] = None,
    value_scale: Optional[torch.Tensor] = None,
    query_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, float]:
    """
    Run the AIT HIP kernel for paged attention.
    
    Args:
        query: Query tensor [total_queries, num_heads, head_dim]
        key_cache: Key cache tensor
        value_cache: Value cache tensor
        block_tables: Block allocation tables
        sequence_lengths: Sequence lengths for each batch
        max_sequence_length: Maximum sequence length in the batch
        max_query_length: Maximum query length in the batch
        kv_cache_data_type: Data type specification for KV cache
        num_kv_heads: Number of key-value heads
        softmax_scale: Scaling factor for attention scores
        key_scale: Optional scaling factors for quantized keys
        value_scale: Optional scaling factors for quantized values
        query_scale: Optional scaling factors for quantized queries
        
    Returns:
        Tuple of (output_tensor, execution_time_in_microseconds)
    """
    return aiter.paged_attn.PagedAttention.forward_decode(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        max_sequence_length,
        kv_cache_data_type,
        num_kv_heads,
        softmax_scale,
        None,  # Additional options
        key_scale,
        value_scale,
        mtp=max_query_length,  # Maximum tokens per page
    )


def shuffle_value_cache_layout(
    value_cache: torch.Tensor,
) -> torch.Tensor:
    """
    Shuffle value cache layout for optimized memory access pattern.
    
    Transforms value cache from [num_blocks, num_kv_heads, head_size, block_size]
    to [num_blocks, num_kv_heads, block_size//x, head_size, x] where x=16/element_size.
    
    Args:
        value_cache: Value cache tensor [num_blocks, num_kv_heads, head_size, block_size]
        
    Returns:
        Shuffled value cache tensor [num_blocks, num_kv_heads, block_size//x, head_size, x]
    """
    elements_per_vector = 16 // value_cache.element_size()
    num_blocks, num_kv_heads, head_size, block_size = value_cache.shape
    
    # Reshape to introduce vector dimension
    value_cache_reshaped = value_cache.view(
        num_blocks, num_kv_heads, head_size, block_size // elements_per_vector, elements_per_vector
    )
    
    # Permute dimensions for optimized access pattern
    value_cache_shuffled = value_cache_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    return value_cache_shuffled


def run_triton_fp8_kernel(
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
    max_logits: Optional[torch.Tensor] = None,
    exp_sums: Optional[torch.Tensor] = None,
    temp_output: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Run Triton FP8 kernel for paged attention.
    
    Args:
        output: Output tensor [num_sequences, num_kv_heads * query_group_size, head_size]
        query: Query tensor [num_sequences, num_kv_heads * query_group_size, head_size]
        key_cache: Key cache [num_blocks, num_kv_heads, head_size//x, kv_block_size, x]
        value_cache: Value cache [num_blocks, num_kv_heads, head_size, kv_block_size]
        sequence_lengths: Sequence lengths [num_sequences]
        block_tables: Block tables [num_sequences, max_blocks_per_sequence]
        attention_scale: Scaling factor for attention scores
        query_sequence_length: Length of query sequences
        max_sequence_length: Maximum sequence length
        compute_type: Triton compute data type
        query_scale: Query scaling factors for quantization
        key_scale: Key scaling factors for quantization
        value_scale: Value scaling factors for quantization
        num_sequence_partitions: Number of sequence partitions (unused)
        alibi_slopes: ALiBi attention slopes
        max_logits: Maximum logits for numerical stability
        exp_sums: Exponential sums for softmax
        temp_output: Temporary output buffer
        
    Returns:
        Tuple of (output_tensor, result_dictionary)
    """
    result = paged_attention_decode_triton_fp8(
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
        num_seq_partitions=0,
        alibi_slopes=None,
    )
    return result


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
    max_logits: Optional[torch.Tensor] = None,
    exp_sums: Optional[torch.Tensor] = None,
    temp_output: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Run Gluon FP8 kernel for paged attention.
    
    Args:
        output: Output tensor [num_sequences, num_kv_heads * query_group_size, head_size]
        query: Query tensor [num_sequences, num_kv_heads * query_group_size, head_size]
        key_cache: Key cache [num_blocks, num_kv_heads, head_size//x, kv_block_size, x]
        value_cache: Value cache [num_blocks, num_kv_heads, head_size, kv_block_size]
        sequence_lengths: Sequence lengths [num_sequences]
        block_tables: Block tables [num_sequences, max_blocks_per_sequence]
        attention_scale: Scaling factor for attention scores
        query_sequence_length: Length of query sequences
        max_sequence_length: Maximum sequence length
        compute_type: Triton compute data type
        query_scale: Query scaling factors for quantization
        key_scale: Key scaling factors for quantization
        value_scale: Value scaling factors for quantization
        num_sequence_partitions: Number of sequence partitions (unused)
        alibi_slopes: ALiBi attention slopes
        max_logits: Maximum logits for numerical stability
        exp_sums: Exponential sums for softmax
        temp_output: Temporary output buffer
        
    Returns:
        Tuple of (output_tensor, result_dictionary)
    """
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
def test_paged_attention_multi_token(
    context_lengths: int,
    batch_size: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    data_type: torch.dtype,
    query_length: int,
    trans_v: bool,
) -> Dict[str, Union[float, str]]:
    """
    Comprehensive test for paged attention with multiple token support.
    
    Tests various implementations (HIP, Triton, Gluon, Assembly) with and without quantization.
    
    Args:
        context_lengths: Length of context sequences
        batch_size: Number of sequences in batch
        num_heads: Tuple of (num_query_heads, num_kv_heads)
        head_size: Dimension of each attention head
        block_size: Size of memory blocks
        data_type: Data type for computations
        query_length: Length of query sequences
        trans_v: Whether to transpose value cache layout
        
    Returns:
        Dictionary containing performance metrics and error information
    """
    results = {}
    seed = 0
    device = "cuda:0"
    torch.set_default_device(device)
    num_query_heads, num_kv_heads = num_heads

    # Validate head configuration
    assert num_query_heads % num_kv_heads == 0, "Query heads must be divisible by KV heads"
    
    # Configuration parameters
    max_sequence_length = 16384
    max_blocks_per_sequence = (max_sequence_length + block_size - 1) // block_size
    total_blocks = max_blocks_per_sequence * batch_size
    blocks_per_sequence = (context_lengths + block_size - 1) // block_size

    # Create query/output index pointer
    query_output_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    sequence_lengths_qo = torch.randint(
        1, 5, (batch_size,), dtype=torch.int32, device=device
    ).fill_(query_length)
    query_output_indptr[1:batch_size + 1] = torch.cumsum(sequence_lengths_qo, dim=0)
    total_queries = query_output_indptr[-1].item()
    max_query_length = sequence_lengths_qo.max().item()

    # Generate random QKV tensors
    qkv_tensor = torch.randn(
        total_queries,
        num_query_heads + 2 * num_kv_heads,
        head_size,
        dtype=data_type,
    )
    query, key, value = torch.split(
        qkv_tensor, [num_query_heads, num_kv_heads, num_kv_heads], dim=1
    )
    query.uniform_(*UNIFORM_RANGE)

    # Create sequence lengths and block tables
    sequence_lengths = torch.tensor([context_lengths] * batch_size, dtype=torch.int32, device=device)

    # Generate random block tables
    block_tables_list = []
    for _ in range(batch_size):
        block_table = [
            random.randint(0, total_blocks - 1) for _ in range(blocks_per_sequence)
        ]
        block_tables_list.append(block_table)

    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

    # Create KV caches
    key_caches, value_caches = create_kv_cache(
        total_blocks,
        block_size,
        1,  # num_layers
        num_kv_heads,
        head_size,
        "auto",  # cache_dtype
        data_type,  # model_dtype
        seed,
        device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]
    softmax_scale = 1.0 / (head_size ** 0.5)

    # Reference implementation without quantization
    reference_output_no_quant = torch_mha_extend(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        query_output_indptr,
    )

    # Test HIP implementation without quantization
    hip_output_no_quant, hip_time_no_quant = run_aiter_hip_kernel(
        query,
        key_cache,
        value_cache,
        block_tables,
        sequence_lengths,
        context_lengths,
        max_query_length,
        "auto",  # kv_cache_data_type
        num_kv_heads,
        softmax_scale,
    )
    
    hip_error_no_quant = checkAllclose(
        reference_output_no_quant,
        hip_output_no_quant,
        msg=f"[PyTorch vs AIT_HIP][No Quant]: {hip_time_no_quant:>8.2f} us......",
    )
    
    # Compare arrays for numerical validation
    compare_arrays(
        hip_output_no_quant.to(torch.float32).detach().cpu().numpy(),
        reference_output_no_quant.to(torch.float32).detach().cpu().numpy()
    )
    results["us_hip_bf16"] = hip_time_no_quant

    # Quantization testing section
    quantized_query, query_scale_factors = pertoken_quant(query, quant_dtype=aiter.dtypes.fp8)
    quantized_keys, key_scale_factors_flat, quantized_values, value_scale_factors_flat, key_scale_original, value_scale_original = (
        quantize_kv_cache_symmetric(key_cache, value_cache, quant_dtype=aiter.dtypes.fp8)
    )

    # Reference implementation with quantization
    reference_output_quant = torch_mha_extend(
        query, 
        quantized_keys, 
        quantized_values, 
        block_tables, 
        sequence_lengths, 
        query_output_indptr, 
        key_scale_factors_flat, 
        value_scale_factors_flat
    )

    # Apply value cache layout transformation if requested
    if trans_v:
        quantized_values = shuffle_value_cache_layout(quantized_values)
        print(f"Transformed quantized_values.shape={quantized_values.shape}")

    fp8_tolerance = 5e-2  # Tolerance for FP8 comparisons

    # Prepare tensors for Triton/Gluon kernels
    quantized_query_triton = quantized_query
    query_scale_triton = query_scale_factors
    triton_output = torch.empty_like(reference_output_no_quant)
    
    # Reshape for multi-query scenarios
    if query_length > 1:
        query_group_size = num_query_heads // num_kv_heads
        assert len(quantized_query.shape) == 3, f"Expected 3D query tensor, but got shape {quantized_query.shape}"
        assert len(triton_output.shape) == 3, f"Expected 3D output tensor, but got shape {triton_output.shape}"
        
        # Reshape query for Triton kernel format
        quantized_query_triton = quantized_query.reshape(
            batch_size, query_length, num_kv_heads, query_group_size, head_size
        )
        quantized_query_triton = quantized_query_triton.transpose(1, 2).reshape(
            batch_size, num_kv_heads * query_length * query_group_size, head_size
        )
        
        # Reshape output for Triton kernel format
        triton_output = triton_output.reshape(
            batch_size, query_length, num_kv_heads, query_group_size, head_size
        )
        triton_output = triton_output.transpose(1, 2).reshape(
            batch_size, num_kv_heads * query_length * query_group_size, head_size
        )

        # Reshape query scales if present
        if len(query_scale_factors.shape) > 0:
            assert len(query_scale_factors.shape) == 3, f"Expected 3D scale tensor, but got shape {query_scale_factors.shape}"
            query_scale_triton = query_scale_factors.reshape(
                batch_size, query_length, num_kv_heads, query_group_size, 1
            )
            query_scale_triton = query_scale_triton.transpose(1, 2).reshape(
                batch_size, num_kv_heads * query_length * query_group_size, 1
            )

    # Test Triton FP8 kernel for short sequences
    if query_length <= 2:
        triton_results = run_triton_fp8_kernel(
            triton_output,
            quantized_query_triton,
            quantized_keys,
            quantized_values,
            sequence_lengths,
            block_tables,
            softmax_scale,
            query_length,
            sequence_lengths.max().item(),
            TORCH_TO_TL_DTYPE[data_type],
            query_scale=query_scale_triton,
            key_scale=key_scale_original,
            value_scale=value_scale_original,
            num_sequence_partitions=0,
            alibi_slopes=None,
        )
        
        # Restore original output shape if needed
        final_output = triton_output
        if query_length > 1:
            final_output = final_output.reshape(
                batch_size, num_kv_heads, query_length, query_group_size, head_size
            )
            final_output = final_output.transpose(1, 2).reshape(
                batch_size * query_length, num_kv_heads * query_group_size, head_size
            )
            
        triton_time = triton_results['triton']
        triton_error = checkAllclose(
            reference_output_quant,
            final_output,
            atol=fp8_tolerance,
            rtol=fp8_tolerance,
            msg=f"[PyTorch vs Triton_FP8][Quant]: {triton_time:>8.2f} us......",
        )
        
        compare_arrays(
            final_output.to(torch.float32).detach().cpu().numpy(),
            reference_output_quant.to(torch.float32).detach().cpu().numpy()
        )
        results["us_triton_fp8"] = triton_time
        
        # Compute MD5 hashes for binary validation
        reference_hash = hashlib.md5(
            reference_output_quant.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()
        ).hexdigest()
        triton_hash = hashlib.md5(
            final_output.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()
        ).hexdigest()
        print(f"out_ref_md5={reference_hash}")
        print(f"triton_output_md5={triton_hash}")

    # Test Gluon FP8 kernel
    gluon_output = torch.empty_like(triton_output)
    gluon_results = run_gluon_fp8_kernel(
        gluon_output,
        quantized_query_triton,
        quantized_keys,
        quantized_values,
        sequence_lengths,
        block_tables,
        softmax_scale,
        query_length,
        sequence_lengths.max().item(),
        TORCH_TO_TL_DTYPE[data_type],
        query_scale=query_scale_triton,
        key_scale=key_scale_original,
        value_scale=value_scale_original,
        # query_scale=query_scale_triton.reshape(-1)[0],
        # key_scale=key_scale_original.reshape(-1)[0],
        # value_scale=value_scale_original.reshape(-1)[0],
        num_sequence_partitions=0,
        alibi_slopes=None,
    )
    
    # Restore original output shape
    if query_length > 1:
        gluon_output = gluon_output.reshape(
            batch_size, num_kv_heads, query_length, query_group_size, head_size
        )
        gluon_output = gluon_output.transpose(1, 2).reshape(
            batch_size * query_length, num_kv_heads * query_group_size, head_size
        )

    gluon_time = gluon_results['total_triton_time']
    gluon_error = checkAllclose(
        reference_output_quant,
        gluon_output,
        atol=fp8_tolerance,
        rtol=fp8_tolerance,
        msg=f"[PyTorch vs Gluon_FP8][Quant]: {gluon_time:>8.2f} us......",
    )
    
    compare_arrays(
        gluon_output.to(torch.float32).detach().cpu().numpy(),
        reference_output_quant.to(torch.float32).detach().cpu().numpy()
    )
    results["us_gluon_fp8"] = gluon_time
    results["err_gluon_fp8"] = gluon_error
    
    # Compute MD5 hashes for binary validation
    reference_hash = hashlib.md5(
        reference_output_quant.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()
    ).hexdigest()
    gluon_hash = hashlib.md5(
        gluon_output.contiguous().view(torch.uint8).detach().cpu().numpy().tobytes()
    ).hexdigest()
    print(f"out_ref_md5={reference_hash}")
    print(f"gluon_fp8_output_md5={gluon_hash}")
    
    # Calculate bandwidth for performance analysis
    kernel_time_us = gluon_time
    bandwidth_tb_per_sec = (
        batch_size * head_size * (
            2 * context_lengths * num_kv_heads * quantized_keys.dtype.itemsize + 
            2 * query_length * num_query_heads * quantized_query.dtype.itemsize
        ) / (kernel_time_us * 1e6 * 1.024 ** 4)
    )
    results["gluon_fp8_bandwith(TB/s)"] = bandwidth_tb_per_sec

    # Test Assembly kernel for supported configurations
    query_group_size = num_query_heads // num_kv_heads
    skip_assembly_conditions = (
        (block_size == 1024 and num_heads != (10, 1)) or
        (block_size == 16 and query_group_size == 8 and query_length == 3) or
        (context_lengths == 512 and query_group_size == 5 and query_length == 3) or
        (block_size == 64)
    )
    
    if not skip_assembly_conditions:
        assembly_output, assembly_time = run_aiter_assembly_kernel(
            query,
            quantized_keys,
            quantized_values,
            block_tables,
            sequence_lengths,
            block_tables.size(1),  # block_tables_stride0
            max_query_length,
            key_scale_original,
            value_scale_original,
            query_output_indptr,
        )
        assembly_error = checkAllclose(
            reference_output_quant,
            assembly_output,
            atol=fp8_tolerance,
            rtol=fp8_tolerance,
            msg=f"[PyTorch vs AIT_Assembly][Quant]: {assembly_time:>8.2f} us......",
        )
        
        compare_arrays(
            assembly_output.to(torch.float32).detach().cpu().numpy(),
            reference_output_quant.to(torch.float32).detach().cpu().numpy()
        )
        results["us_asm_fp8"] = assembly_time
        
        # Calculate assembly kernel bandwidth
        assembly_bandwidth_tb_per_sec = (
            batch_size * head_size * (
                2 * context_lengths * num_kv_heads * quantized_keys.dtype.itemsize + 
                2 * query_length * num_query_heads * query.dtype.itemsize
            ) / (assembly_time * 1e6 * 1.024 ** 4)
        )
        results["asm_fp8_bandwith(TB/s)"] = assembly_bandwidth_tb_per_sec

    # Test HIP implementation with quantization
    query_scale_squeezed = query_scale_factors.squeeze(-1)
    hip_output_quant, hip_time_quant = run_aiter_hip_kernel(
        query,
        quantized_keys,
        quantized_values,
        block_tables,
        sequence_lengths,
        context_lengths,
        max_query_length,
        "fp8",  # kv_cache_data_type
        num_kv_heads,
        softmax_scale,
        key_scale_original,
        value_scale_original,
        query_scale_squeezed,
    )
    
    hip_error_quant = checkAllclose(
        reference_output_quant,
        hip_output_quant,
        atol=fp8_tolerance,
        rtol=fp8_tolerance,
        msg=f"[PyTorch vs AIT_HIP_FP8][Quant]: {hip_time_quant:>8.2f} us......",
    )
    
    compare_arrays(
        hip_output_quant.to(torch.float32).detach().cpu().numpy(),
        reference_output_quant.to(torch.float32).detach().cpu().numpy()
    )
    results["us_hip_fp8"] = hip_time_quant

    # Calculate performance ratios
    if "us_hip_fp8" in results:
        results["perf_fp8_gluon_vs_hip"] = f'{results["us_hip_fp8"] / results["us_gluon_fp8"]:.0%}'
    else:
        results["perf_fp8_gluon_vs_hip"] = 'NaN'
        
    if "us_asm_fp8" in results:
        results["perf_fp8_gluon_vs_asm"] = f'{results["us_asm_fp8"] / results["us_gluon_fp8"]:.0%}'
    else:
        results["perf_fp8_gluon_vs_asm"] = 'NaN'

    # Print Triton version information
    print(f"Triton location: {triton}")
    print(f"Triton version: {triton.__version__}")
    sys.stdout.flush()

    return results


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure command line argument parser for paged attention testing.
    
    Returns:
        Configured ArgumentParser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Configuration input for paged attention performance testing",
    )
    
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=DATA_TYPE_OPTIONS,
        nargs="?",
        const=None,
        default=None,
        help="""Data type for computation.
        Available options: bf16
        Example: -d bf16""",
    )
    
    parser.add_argument(
        "-n",
        "--num_heads",
        type=dtypes.str2tuple,
        default=None,
        help="""Number of attention heads in format (query_heads, kv_heads).
        Example: -n 8,1""",
    )
    
    parser.add_argument(
        "-q",
        "--query_length",
        type=int,
        choices=QUERY_LENGTH_OPTIONS,
        default=None,
        help="""Length of query sequences.
        Available options: 1, 2, 3, 4
        Example: -q 1""",
    )
    
    parser.add_argument(
        "-c",
        "--context_length",
        type=int,
        default=None,
        help="""Length of context sequences.
        Example: -c 128""",
    )
    
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=None,
        help="""Number of sequences in batch.
        Example: -b 128""",
    )
    
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="""Size of memory blocks for paged attention.
        Example: --block_size 16""",
    )
    
    parser.add_argument(
        "--trans_v",
        action="store_true",
        help="""Enable value cache layout transformation for optimized memory access.
        Example: --trans_v""",
    )
    
    return parser


def process_command_line_arguments(args: argparse.Namespace) -> tuple:
    """
    Process command line arguments and update test configuration accordingly.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Tuple containing updated configuration lists:
        (data_types, block_sizes, head_configs, context_lengths, batch_sizes, query_lengths, trans_v)
    """
    # Initialize with default values
    data_types = DATA_TYPE_OPTIONS
    block_sizes = BLOCK_SIZE_OPTIONS
    head_configs = HEAD_CONFIGURATIONS
    context_lengths = CONTEXT_LENGTH_OPTIONS
    batch_sizes = BATCH_SIZE_OPTIONS
    query_lengths = QUERY_LENGTH_OPTIONS
    
    # Override with command line arguments if provided
    if args.dtype is not None:
        # Convert string dtype to torch dtype using mapping
        data_types = [dtypes.d_dtypes[args.dtype]]
    else:
        # Convert all default string dtypes to torch dtypes
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
    
    return data_types, block_sizes, head_configs, context_lengths, batch_sizes, query_lengths, args.trans_v


def run_comprehensive_paged_attention_tests(
    data_types: list,
    block_sizes: list,
    head_configs: list,
    context_lengths: list,
    batch_sizes: list,
    query_lengths: list,
    trans_v: bool
) -> pd.DataFrame:
    """
    Run comprehensive paged attention tests across all specified configurations.
    
    Args:
        data_types: List of data types to test
        block_sizes: List of block sizes to test
        head_configs: List of (query_heads, kv_heads) configurations
        context_lengths: List of context lengths to test
        batch_sizes: List of batch sizes to test
        query_lengths: List of query lengths to test
        trans_v: Whether to transpose value cache layout
        
    Returns:
        DataFrame containing performance results for all test configurations
    """
    test_results = []
    total_combinations = (
        len(data_types) * len(block_sizes) * len(head_configs) * 
        len(context_lengths) * len(batch_sizes) * len(query_lengths)
    )
    current_combination = 0
    
    aiter.logger.info(f"Starting comprehensive paged attention testing with {total_combinations} configurations")
    
    # Iterate through all parameter combinations
    for data_type in data_types:
        for block_size in block_sizes:
            for head_config in head_configs:
                for context_length in context_lengths:
                    for batch_size in batch_sizes:
                        for query_length in query_lengths:
                            current_combination += 1
                            
                            aiter.logger.info(
                                f"Running test {current_combination}/{total_combinations}: "
                                f"dtype={data_type}, block_size={block_size}, heads={head_config}, "
                                f"ctx_len={context_length}, batch_size={batch_size}, qlen={query_length}"
                            )
                            
                            try:
                                # Execute paged attention test with current configuration
                                result = test_paged_attention_multi_token(
                                    context_lengths=context_length,
                                    batch_size=batch_size,
                                    num_heads=head_config,
                                    head_size=HEAD_DIMENSION,
                                    block_size=block_size,
                                    data_type=data_type,
                                    query_length=query_length,
                                    trans_v=trans_v,
                                )
                                
                                # # Add configuration metadata to result for tracking
                                # result.update({
                                #     'data_type': str(data_type),
                                #     'block_size': block_size,
                                # })
                                
                                test_results.append(result)
                                aiter.logger.info(f"Completed test {current_combination} successfully")
                                
                            except Exception as e:
                                aiter.logger.error(
                                    f"Test failed for configuration: "
                                    f"dtype={data_type}, block_size={block_size}, heads={head_config}, "
                                    f"ctx_len={context_length}, batch_size={batch_size}, qlen={query_length}"
                                )
                                aiter.logger.error(f"Error: {str(e)}")
                                # Continue with other tests even if one fails
    
    # Create DataFrame from results
    results_dataframe = pd.DataFrame(test_results)
    return results_dataframe


def save_results_to_file(results_dataframe: pd.DataFrame, trans_v: bool) -> None:
    """
    Save test results to CSV file with appropriate naming.
    
    Args:
        results_dataframe: DataFrame containing test results
        trans_v: Whether value cache transformation was used
    """
    if trans_v:
        output_filename = "pa_gluon_fp8_trans_v.triton." + str(TRITON_VERSION) + ".csv"
    else:
        output_filename = "pa_gluon_fp8.triton." + str(TRITON_VERSION) + ".csv"
    
    results_dataframe.to_csv(output_filename, index=False)
    aiter.logger.info(f"Results saved to {output_filename}")


def main() -> None:
    """
    Main function to execute paged attention performance testing.
    
    Parses command line arguments, runs comprehensive tests, and saves results.
    """
    # Parse command line arguments
    parser = create_argument_parser()
    command_line_args = parser.parse_args()
    
    # Process arguments and get test configuration
    (test_data_types, test_block_sizes, test_head_configs, 
     test_context_lengths, test_batch_sizes, test_query_lengths, 
     should_transpose_value_cache) = process_command_line_arguments(command_line_args)
    
    # # Log test configuration
    # aiter.logger.info("Test Configuration:")
    # aiter.logger.info(f"  Data Types: {test_data_types}")
    # aiter.logger.info(f"  Block Sizes: {test_block_sizes}")
    # aiter.logger.info(f"  Head Configurations: {test_head_configs}")
    # aiter.logger.info(f"  Context Lengths: {test_context_lengths}")
    # aiter.logger.info(f"  Batch Sizes: {test_batch_sizes}")
    # aiter.logger.info(f"  Query Lengths: {test_query_lengths}")
    # aiter.logger.info(f"  Transpose Value Cache: {should_transpose_value_cache}")
    
    # Run comprehensive tests
    results_df = run_comprehensive_paged_attention_tests(
        data_types=test_data_types,
        block_sizes=test_block_sizes,
        head_configs=test_head_configs,
        context_lengths=test_context_lengths,
        batch_sizes=test_batch_sizes,
        query_lengths=test_query_lengths,
        trans_v=should_transpose_value_cache
    )
    
    # Display summary and save results
    aiter.logger.info(f"Testing completed. Summary:\n{results_df}")
    save_results_to_file(results_df, should_transpose_value_cache)
    
    # Print key performance metrics if available
    if not results_df.empty:
        performance_columns = [col for col in results_df.columns if 'us_' in col or 'bandwith' in col]
        if performance_columns:
            aiter.logger.info("Performance Summary:")
            aiter.logger.info(results_df[performance_columns].describe())


if __name__ == "__main__":
    main()

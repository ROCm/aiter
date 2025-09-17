# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from typing import Tuple, List, Dict
import numpy as np
import triton
import triton.language as tl
import torch
import pytest
import random
from aiter import pertoken_quant
from aiter import dtypes
from aiter import logger
from aiter import paged_attn as ops
from aiter.test_common import (
    checkAllclose,
    benchmark,
    perftest,
)
import pandas as pd
import os
import itertools
import math
import hashlib
from pa_decode_gluon import paged_attention_decode as paged_attention_decode_gluon
from pa_decode_triton import paged_attention_decode as paged_attention_decode_triton


TEST_NUM_ITERS = 101

DEBUG_MODE = False
tl_to_torch_dtype = {tl.bfloat16: torch.bfloat16, tl.float16: torch.float16}
torch_to_tl_dtype = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


def setup_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(123)


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, 
                   k: int = 5, 
                   thresholds: List[float] = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]) -> Dict:
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

    result = {
        'top_k_diff': [],
        'threshold_stats': [],
        'nan_info': {}
    }

    # Check for NaN values
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    
    if np.any(nan_mask1):
        result['nan_info']['arr1_nan_count'] = np.sum(nan_mask1)
        result['nan_info']['arr1_nan_positions'] = np.argwhere(nan_mask1)
        print(f"Warning: arr1 contains {result['nan_info']['arr1_nan_count']} NaN values")
    
    if np.any(nan_mask2):
        result['nan_info']['arr2_nan_count'] = np.sum(nan_mask2)
        result['nan_info']['arr2_nan_positions'] = np.argwhere(nan_mask2)
        print(f"Warning: arr2 contains {result['nan_info']['arr2_nan_count']} NaN values")
    
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
        result['top_k_diff'].append({
            'value': diff[idx],
            'position': idx,
            'arr1_value': arr1[idx],
            'arr2_value': arr2[idx]
        })

    # Compute threshold statistics
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]
        mask = (diff >= lower) & (diff < upper)
        count = np.sum(mask)
        result['threshold_stats'].append({
            'range': f"[{lower:.1e}, {upper:.1e})",
            'count': count,
            'percentage': 100 * count / total_elements
        })
    
    # Handle values above the largest threshold
    mask = diff >= thresholds[-1]
    count = np.sum(mask)
    result['threshold_stats'].append({
        'range': f">={thresholds[-1]:.1e}",
        'count': count,
        'percentage': 100 * count / total_elements
    })

    print("\nTop differences:")
    for item in result['top_k_diff']:
        print(f"Position {item['position']}: arr1 = {arr1[item['position']]:.6f}, arr2 = {arr2[item['position']]:.6f}, Diff = {item['value']:.6f}")

    print("\nThreshold statistics:")
    for stat in result['threshold_stats']:
        print(f"{stat['range']}: {stat['count']} ({stat['percentage']:.2f}%)")

    print("\nNaN info:")
    print(result['nan_info'])

    return result


def input_helper(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    dtype,
    kv_cache_dtype,
    output_type,
    num_blocks=4,
):
    """Helper function to generate input tensors for paged attention testing."""
    # Query tensor generation
    if dtype not in (torch.bfloat16, torch.float16, torch.float32):
        query = torch.randn(
            B, H_Q, D, dtype=torch.float16, device="cuda"
        )  # assumption dtype is 8bits or lower
        query = query.to(dtype=dtype, device="cuda")
    else:
        query = torch.randn(B, H_Q, D, dtype=dtype, device="cuda")

    if kv_cache_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        x = min(D, 16 // torch.tensor([], dtype=torch.float16).element_size())
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=torch.float16, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=torch.float16, device="cuda"
        )
        key_cache = torch.clamp(
            key_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs
        value_cache = torch.clamp(
            value_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs

        # torch doesn't have randn for fp8 data type, so we convert here
        key_cache = key_cache.to(dtype=kv_cache_dtype)
        value_cache = value_cache.to(dtype=kv_cache_dtype)
    else:
        x = min(D, 16 // torch.tensor([], dtype=kv_cache_dtype).element_size())
        key_cache = torch.randn(
            num_blocks, H_KV, D // x, KV_BLK_SZ, x, dtype=kv_cache_dtype, device="cuda"
        )
        value_cache = torch.randn(
            num_blocks, H_KV, D, KV_BLK_SZ, dtype=kv_cache_dtype, device="cuda"
        )
        key_cache = torch.clamp(
            key_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs
        value_cache = torch.clamp(
            value_cache, min=1e-3
        )  # For FP8 case, this is needed to prevent NANs

    key_cache_tri = key_cache
    value_cache_tri = value_cache

    context_lens = torch.full((B,), SEQ_LEN, device="cuda").to(torch.int32)
    max_context_len = max(context_lens)
    max_num_blks_per_seq = (max_context_len + KV_BLK_SZ - 1) // KV_BLK_SZ

    block_tables = []
    for i in range(B):
        block_table = [
            random.randint(0, num_blocks - 1) for _ in range(max_num_blks_per_seq)
            # 0 for _ in range(max_num_blks_per_seq)
        ]
        block_tables.append(block_table)
    block_tables = torch.tensor(block_tables, dtype=torch.int32, device="cuda")

    output = torch.zeros(B, H_Q, D, dtype=output_type, device="cuda")

    return (
        query,
        output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    )

@perftest()
# @perftest(num_iters=TEST_NUM_ITERS)
def run_hip(
    query,          # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    k_cache,        # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    v_cache,        # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    block_tables,   # [num_seqs, max_num_blks_per_seq]
    seq_lens,       # [num_seqs]
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    k_scale,
    v_scale,
):
    return ops.PagedAttention.forward_decode(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )

def run_triton(
    output: torch.Tensor,       # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,        # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,    # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    seq_lens: torch.Tensor,     # [num_seqs]
    block_tables: torch.Tensor, # [num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    compute_type,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_seq_partitions: int = 0,  # TODO use this below
    alibi_slopes: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    exp_sums: torch.Tensor = None,
    tmp_output: torch.Tensor = None,
) -> None:
    result = paged_attention_decode_triton(
        output,
        query,
        key_cache,
        value_cache,
        seq_lens,
        block_tables,
        attn_scale,
        max_seq_len,
        compute_type,
        k_scale,
        v_scale,
        num_seq_partitions=0,
        alibi_slopes=None,
    )
    return output, result

def run_gluon(
    output: torch.Tensor,       # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    query: torch.Tensor,        # [num_seqs, num_kv_heads*query_grp_sz, head_sz]
    key_cache: torch.Tensor,    # [num_blks, num_kv_heads, head_sz/x, kv_blk_sz, x]
    value_cache: torch.Tensor,  # [num_blks, num_kv_heads, head_sz, kv_blk_sz]
    seq_lens: torch.Tensor,     # [num_seqs]
    block_tables: torch.Tensor, # [num_seqs, max_num_blks_per_seq]
    attn_scale: float,
    max_seq_len: int,
    compute_type,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    num_seq_partitions: int = 0,  # TODO use this below
    alibi_slopes: torch.Tensor = None,
    max_logits: torch.Tensor = None,
    exp_sums: torch.Tensor = None,
    tmp_output: torch.Tensor = None,
) -> None:
    result = paged_attention_decode_gluon(
        output,
        query,
        key_cache,
        value_cache,
        seq_lens,
        block_tables,
        attn_scale,
        max_seq_len,
        compute_type,
        k_scale,
        v_scale,
        num_seq_partitions=0,
        alibi_slopes=None,
    )
    return output, result

@benchmark()
def test_paged_attention(
    B,
    H_Q,
    H_KV,
    D,
    KV_BLK_SZ,
    SEQ_LEN,
    NUM_BLK,
    dtype,
    kv_cache_dtype,
    compute_type,
    output_type,
):

    if SEQ_LEN >= 8192 and B >= 16:
        pytest.skip("B>={16} and SEQ_LEN>={8192} tests are too slow")
    torch.set_printoptions(threshold=100000)
    num_blocks = NUM_BLK

    (
        query,
        triton_output,
        key_cache,
        value_cache,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        max_context_len,
    ) = input_helper(
        B,
        H_Q,
        H_KV,
        D,
        KV_BLK_SZ,
        SEQ_LEN,
        dtype,
        kv_cache_dtype,
        output_type,
        num_blocks,
    )

    attn_scale = 1.0 / (D**0.5)
    k_scale = v_scale = 1.0

    hip_output, time_hip = run_hip(
        query,
        key_cache_tri,
        value_cache_tri,
        block_tables,
        context_lens,
        max_context_len.item(),
        "auto",
        H_KV,
        attn_scale,
        alibi_slopes=None,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    time_hip = {"hip": time_hip}
    # time_hip = {"hip": 0.99999999}

    triton_output, triton_time = run_triton(
        triton_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        compute_type,
        k_scale=torch.tensor(k_scale),
        v_scale=torch.tensor(v_scale),
        num_seq_partitions=0,
        alibi_slopes=None,
    )

    gluon_output = torch.empty_like(triton_output)
    gluon_output, gluon_time = run_gluon(
        gluon_output,
        query,
        key_cache_tri,
        value_cache_tri,
        context_lens,
        block_tables,
        attn_scale,
        max_context_len,
        compute_type,
        k_scale=torch.tensor(k_scale),
        v_scale=torch.tensor(v_scale),
        num_seq_partitions=0,
        alibi_slopes=None,
    )

    def compare_tensor_in_dict(res_dict, ref_res_dict, key):
        res = res_dict[key]
        ref_res = ref_res_dict[key]
        res_md5 = hashlib.md5(res.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
        ref_res_md5 = hashlib.md5(ref_res.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
        print(f"******************compare {key}******************")
        compare_arrays(res.to(torch.float32).detach().cpu().numpy(), ref_res.to(torch.float32).detach().cpu().numpy())
        print(f"res_md5={res_md5}")
        print(f"ref_res_md5={ref_res_md5}")


    compare_arrays(triton_output.to(torch.float32).detach().cpu().numpy(), hip_output.to(torch.float32).detach().cpu().numpy())
    compare_arrays(gluon_output.to(torch.float32).detach().cpu().numpy(), triton_output.to(torch.float32).detach().cpu().numpy())
    compare_tensor_in_dict(gluon_time, triton_time, "tmp_output")
    compare_tensor_in_dict(gluon_time, triton_time, "exp_sums")
    compare_tensor_in_dict(gluon_time, triton_time, "max_logits")

    hip_output_md5 = hashlib.md5(hip_output.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    triton_output_md5 = hashlib.md5(triton_output.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()
    gluon_output_md5 = hashlib.md5(gluon_output.view(torch.uint8).detach().cpu().numpy().tobytes()).hexdigest()

    print(f"hip_output_md5={hip_output_md5}")
    print(f"triton_output_md5={triton_output_md5}")
    print(f"gluon_output_md5={gluon_output_md5}")

    checkAllclose(triton_output, hip_output)
    checkAllclose(gluon_output, hip_output)
    print("\033[92mPASSED\033[0m")

    return {**time_hip, **triton_time}

df = []
HEAD_SIZE = 128
KV_BLOCK_SIZE = 16
SEED = 0
# NUM_SEQS_LIST = [1, 2, 4, 8, 16, 32, 64, 128]
# SEQ_LEN_LIST = [128, 256, 512, 1024, 2048, 4096]
# NUM_HEADS_LIST = [(8, 1), (64, 8)]
# NUM_SEQS_LIST = [random.randint(1, 128) for _ in range(8)]
# SEQ_LEN_LIST = [random.randint(64, 4096) for _ in range(8)]
NUM_SEQS_LIST = [32]
SEQ_LEN_LIST = [4096]
NUM_HEADS_LIST = [(8, 1)]
# NUM_HEADS_LIST = [(16, 1)]
DTYPE_LIST = [torch.bfloat16]

for (num_seq, seq_len, num_heads, dtype) in itertools.product(
    NUM_SEQS_LIST, SEQ_LEN_LIST, NUM_HEADS_LIST, DTYPE_LIST):
    max_num_blocks_per_seq = (seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    num_blocks = max_num_blocks_per_seq * num_seq
    ret = test_paged_attention(
        num_seq, num_heads[0], num_heads[1], HEAD_SIZE, KV_BLOCK_SIZE, seq_len, 
        num_blocks, dtype, dtype, torch_to_tl_dtype[dtype], dtype
    )
#     df.append(ret)
# df = pd.DataFrame(df)
# logger.info(f"summary:\n{df}")

# cur_dir = os.path.dirname(os.path.abspath(__file__))
# result_dir = os.path.join(os.path.dirname(cur_dir), "result")
# if os.path.exists(result_dir) == False:
#     os.mkdir(result_dir)
# df.to_csv(os.path.join(result_dir, "pa_summary.csv"), index=False)

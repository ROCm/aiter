# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import random
from typing import List, Optional, Tuple, Union
import torch
import aiter
from aiter import dtypes
import pytest
import aiter as ops
from aiter.test_common import (
    perftest,
    tensor_dump,
    tensor_load,
)

uniform_range = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": dtypes.bf16,
    "float": dtypes.fp32,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}
ck_naive_quant_algo = [
    "NO",
    "KV_8BIT_PER_HEAD",
    # // FP8/INT8 quant for KVCache, per-token quant
    # // [num_tokens, nhead, hdim] -> [nhead, num_tokens]
    "KV_8BIT_PER_TOKEN",
    # // same as 8bit per token quant but 4 bit
    "KV_4BIT_PER_TOKEN",
    "KV_8BIT_PER_TENSOR",
]


def _generate_random_fp8(
    tensor: torch.Tensor,
    low: float,
    high: float,
) -> None:
    # NOTE(zhaoyang): Due to NaN and Inf representation for fp8 data type,
    # it may occur Inf or NaN if we directly use torch.randint
    # to generate random data for fp8 data.
    # For example, s.11111.00 in fp8e5m2 format represents Inf.
    #     | E4M3        | E5M2
    # -----|-------------|-------------------
    # Inf | N/A         | s.11111.00
    # NaN | s.1111.111  | s.11111.{01,10,11}
    from vllm import _custom_ops as ops

    tensor_tmp = torch.empty_like(tensor, dtype=torch.float16)
    tensor_tmp.uniform_(low, high)
    ops.convert_fp8(tensor, tensor_tmp)
    del tensor_tmp


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
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


def kv_cache_factory(
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

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    x = 16 // torch_dtype.itemsize
    k_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    k_caches: List[torch.Tensor] = []
    scale = head_size**-0.5
    for _ in range(num_layers):
        k_cache = torch.empty(size=k_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            k_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(k_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        k_caches.append(k_cache)

    v_cache_shape = (num_blocks, num_heads, head_size, block_size)
    v_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        v_cache = torch.empty(size=v_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            v_cache.uniform_(-scale, scale)
        elif cache_dtype == "fp8":
            _generate_random_fp8(v_cache, -scale, scale)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        v_caches.append(v_cache)
    return k_caches, v_caches


FLOAT32_BYTES = torch.finfo(dtypes.fp32).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = 65536
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 32768  # Arbitrary values for testing
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {dtypes.fp16, dtypes.bf16}
DTYPES = [torch.half, dtypes.bf16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 128]

BLOCK_SIZES = [16, 32]
USE_ALIBI = [False]
KV_CACHE_DTYPE = ["auto"]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]

# 0: no quant. 1: (ignore this), FP8, 2: K/V per-token(prefer this)
PA_QUANT = 0


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    num_queries_per_kv: int,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
) -> None:
    num_query_heads = query.shape[1]
    num_kv_heads = value_cache.shape[1]
    head_size = value_cache.shape[2]
    block_size = value_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: list[torch.Tensor] = []
        values_lst: list[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, :, :, block_offset]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)


@perftest(num_iters=2)
def run_native(
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
    k_scale_cache,
    v_scale_cache,
    num_queries_per_kv,
    dtype,
):
    output = torch.zeros_like(query).to(dtype)
    num_query_heads = query.shape[1]
    num_kv_heads = v_cache.shape[1]
    head_size = v_cache.shape[2]
    block_size = v_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()

    # (num_blocks, num_heads, head_size // x, block_size, x)
    k_cache = (
        k_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_kv_heads, head_size)
    )
    # (num_blocks, num_heads, head_size, block_size)
    v_cache = v_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_kv_heads, head_size)
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        ctx_len = int(seq_lens_lst[i])

        idx = [
            int(block_table[j // block_size]) * block_size + (j % block_size)
            for j in range(ctx_len)
        ]
        if k_cache.dtype == dtypes.fp8:
            keys = k_cache.view(dtypes.i8)[idx].view(dtypes.fp8)
            values = v_cache.view(dtypes.i8)[idx].view(dtypes.fp8)
        else:
            keys = k_cache[idx]
            values = v_cache[idx]
        if k_scale_cache.numel() > 1:
            k_scale = k_scale_cache[:, idx].contiguous().view(num_kv_heads, 1, ctx_len)
            v_scale = v_scale_cache[:, idx].contiguous().view(num_kv_heads, 1, ctx_len)
        else:
            k_scale = k_scale_cache  # [1]
            v_scale = v_scale_cache  # [1]

        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(ctx_len).int()
            alibi_bias = (position_ids - ctx_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(
            q, keys, values, scale, num_kv_heads, k_scale, v_scale, alibi_bias, dtype
        )
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    return output  # , 1


@perftest()
def run_aiter(
    output,
    exp_sums,
    max_logits,
    tmp_output,
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    block_size,
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    k_scale,
    v_scale,
):
    ops.paged_attention_rocm(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        num_kv_heads,
        scale,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        k_scale,
        v_scale,
        None,
        256,
    )


@perftest(num_iters=2)
def run_aiter_naive(
    query,
    k_cache,
    v_cache,
    block_tables,
    seq_lens,
    k_dequant_scales,
    v_dequant_scales,
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    k_scale,
    v_scale,
    block_size,
    quant_algo=0,
):
    return aiter.pa_fwd_naive(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        k_dequant_scales,
        v_dequant_scales,
        max_seq_len,
        num_kv_heads,
        scale,
        k_scale,
        v_scale,
        block_size,
        quant_algo,
    )


@perftest()
def run_aiter_asm(
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
    max_num_blocks,
    k_scale=None,
    v_scale=None,
    high_precision=0,
):
    return aiter.pa_fwd_asm(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        max_num_blocks,
        k_scale,
        v_scale,
        None,
        high_precision,
    )


def dump_input(
    path,
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: float,
    v_scale: float,
    out_golden,
    out_test,
):
    # path = '/mnt/raid0/ljin1/dk/ater/debug_ctx7'
    tensor_dump(query, "Q", path)
    # qbk = tensor_load('Q.bin')
    # checkAllclose(query, qbk)
    tensor_dump(k_cache, "K_cache", path)
    tensor_dump(v_cache, "V_cache", path)
    tensor_dump(block_tables, "block_tables", path)
    tensor_dump(seq_lens, "seq_lens", path)
    tensor_dump(k_scale, "k_scale", path)
    tensor_dump(v_scale, "v_scale", path)
    tensor_dump(out_golden, "out_golden", path)
    tensor_dump(out_test, "out_test", path)


def load_input():
    # return (tensor_load('Q.bin'),
    #         tensor_load('K_cache.bin'),
    #         tensor_load('V_cache.bin'),
    #         tensor_load('block_tables.bin'),
    #         tensor_load('seq_lens.bin'),
    #         tensor_load('out_aiter.bin'))
    # return (tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/Q_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/K_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/V_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/block_tables.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/seq_lens.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/OUT_16.bin'),
    #         )
    return (
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/Q_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/K_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/V_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/block_tables.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/seq_lens.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/OUT_BF16.bin"),
    )


@pytest.mark.parametrize("num_seqs", NUM_GEN_SEQS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("use_alibi", USE_ALIBI)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kv_cache_dtype", KV_CACHE_DTYPE)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_paged_attention(
    num_seqs: int,
    num_heads: tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    seed: int,
    device: str,
) -> None:

    torch.set_default_device(device)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
    query.uniform_(-scale, scale)

    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=torch.float)

    seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
    seq_lens[-1] = MAX_SEQ_LEN
    max_seq_len = max(seq_lens)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int)

    # Create the block tables.
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    block_tables_lst: list[list[int]] = []
    for _ in range(num_seqs):
        block_table = [
            random.randint(0, NUM_BLOCKS - 1) for _ in range(max_num_blocks_per_seq)
        ]
        block_tables_lst.append(block_table)

    block_tables = torch.tensor(block_tables_lst, dtype=torch.int)

    # Create the KV caches.
    key_caches, value_caches = kv_cache_factory(
        NUM_BLOCKS,
        block_size,
        1,
        num_kv_heads,
        head_size,
        kv_cache_dtype,
        dtype,
        seed,
        device,
    )
    key_cache, value_cache = key_caches[0], value_caches[0]

    # Using default kv_scale
    k_scale = v_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    # Call the paged attention kernel.
    output = torch.empty_like(query)

    PARTITION_SIZE = 256

    num_partitions = (max_seq_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    assert PARTITION_SIZE % block_size == 0
    num_seqs, num_heads, head_size = output.shape
    tmp_output = torch.empty(
        size=(num_seqs, num_heads, num_partitions, head_size),
        dtype=output.dtype,
    )
    exp_sums = torch.empty(
        size=(num_seqs, num_heads, num_partitions),
        dtype=torch.float32,
    )
    max_logits = torch.empty_like(exp_sums)

    query = torch.ones_like(query)
    key_cache = torch.ones_like(key_cache)
    value_cache = torch.ones_like(value_cache)

    # Run the reference implementation.
    if kv_cache_dtype == "fp8":
        # Convert cache data back to dtype.
        x = 16 // torch.tensor([], dtype=dtype).element_size()
        key_cache_shape = (NUM_BLOCKS, num_kv_heads, head_size // x, block_size, x)
        dequantized_key_cache = torch.empty(
            size=key_cache_shape, dtype=dtype, device=device
        )
        ops.convert_fp8(dequantized_key_cache, key_cache, 1.0, "fp8")
        key_cache = dequantized_key_cache

        value_cache_shape = value_cache.shape
        dequantized_value_cache = torch.empty(
            size=value_cache_shape, dtype=dtype, device=device
        )
        ops.convert_fp8(dequantized_value_cache, value_cache, 1.0, "fp8")
        value_cache = dequantized_value_cache

    output = torch.empty_like(query)
    run_aiter(
        output,
        exp_sums,
        max_logits,
        tmp_output,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        block_size,
        max_seq_len,
        kv_cache_dtype,
        num_kv_heads,
        scale,
        alibi_slopes,
        k_scale,
        v_scale,
    )

    ref_output = torch.empty_like(query)
    ref_single_query_cached_kv_attention(
        ref_output,
        query,
        num_queries_per_kv,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        scale,
        alibi_slopes,
    )

    # NOTE(zhaoyang): FP8 KV Cache will introduce quantization error,
    # so we use a relaxed tolerance for the test.
    atol, rtol = 1e-2, 1e-5
    torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)

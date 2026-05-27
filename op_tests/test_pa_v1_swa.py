# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Zyphra AI Inc. All rights reserved.
"""
Unit tests for sliding-window attention (SWA) support in
``torch.ops.aiter.paged_attention_v1``.

The kernel under test is the JIT-compiled HIP mfma16 decode path in
``csrc/cpp_itfs/pa/pa_v1.{cuh,cpp.jinja,py}``. We compare its output against
a from-scratch PyTorch reference (`run_torch_swa`) that builds the full
unmasked attention then applies the SWA mask explicitly.

This repo uses ``sliding_window`` (int, default 0 = disabled) at the API
layer. Inside the tests we keep ``window_size`` as the local variable name
(value > 0 = enabled, value <= 0 = full causal) and translate at the call
boundary.

Run::

    pytest op_tests/test_pa_v1_swa.py -v
"""

from __future__ import annotations

import argparse
import random
from typing import List, Optional, Tuple

import pytest
import torch
from einops import rearrange

import aiter  # noqa: F401  -- registers torch.ops.aiter.*
from aiter import dtypes
from aiter.test_common import checkAllclose

UNIFORM_RANGE = (-1, 1)
_PARTITION_SIZE_ROCM = 256


# ----------------------------- helpers --------------------------------- #


def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: torch.dtype,
    seed: int = 0,
    device: str = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Allocate paged-KV cache tensors in the internal 5D layout."""
    x = 16 // cache_dtype.itemsize
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        k = torch.empty(size=key_cache_shape, dtype=cache_dtype, device=device)
        k.uniform_(*UNIFORM_RANGE)
        key_caches.append(k)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        v = torch.empty(size=value_cache_shape, dtype=cache_dtype, device=device)
        v.uniform_(*UNIFORM_RANGE)
        value_caches.append(v)
    return key_caches, value_caches


def run_torch_swa(
    query: torch.Tensor,  # [num_seqs, num_q_heads, head_size]
    key_cache_nhd: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache_nhd: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_size]
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    window_size: int,  # <=0 means full causal (no mask)
    num_queries_per_kv: int,
) -> torch.Tensor:
    """PyTorch reference for masked decode attention.

    For each sequence:
      1. Gather K, V from blocks using `block_tables`.
      2. Compute logits = scale * Q @ K^T.
      3. Apply SWA mask: token positions < max(0, seq_len - window_size) are
         masked out (set to -inf). The query sits at position seq_len-1.
      4. Softmax, weighted sum of V.
    """
    num_seqs, num_q_heads, head_size = query.shape
    num_kv_heads = key_cache_nhd.shape[1]
    block_size = key_cache_nhd.shape[2]
    output = torch.zeros_like(query)

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()

    for i in range(num_seqs):
        q = query[i].unsqueeze(0)  # [1, num_q_heads, head_size]
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])
        if seq_len <= 0:
            continue

        keys_lst, values_lst = [], []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size
            k = key_cache_nhd[block_number, :, block_offset, :].reshape(
                num_kv_heads, head_size
            )
            v = value_cache_nhd[block_number, :, block_offset, :].reshape(
                num_kv_heads, head_size
            )
            keys_lst.append(k)
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)

        if num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        attn_weights = scale * torch.einsum("qhd,khd->hqk", q, keys).float()

        if window_size > 0:
            kv_lo = max(0, seq_len - window_size)
            if kv_lo > 0:
                mask = torch.arange(seq_len, device=q.device) < kv_lo
                attn_weights = attn_weights.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(attn_weights, dim=-1).to(values.dtype)
        out = torch.einsum("hqk,khd->qhd", attn_weights, values).view(
            num_q_heads, head_size
        )
        output[i].copy_(out)

    return output


def _api_sliding_window(window_size: int) -> int:
    """Translate test-local convention (-1 = disabled) to API convention
    (0 = disabled, target repo)."""
    return window_size if window_size > 0 else 0


def run_aiter_pa_v1_swa(
    query: torch.Tensor,
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, block_size, head_size]
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    window_size: int,
    cu_query_lens: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    logits_soft_cap: float = 0.0,
    kv_cache_dtype: str = "auto",
    kv_cache_layout: str = "NHD",
) -> torch.Tensor:
    """Wrapper for ``torch.ops.aiter.paged_attention_v1`` with SWA.

    ``window_size <= 0`` means full causal (passes ``sliding_window=0``).
    """
    num_seqs, num_heads, head_size = query.shape
    max_seq_len = int(seq_lens.max().item())
    # SWA-aware workspace sizing matches the in-op formula in pa_v1.py:
    # when ctx > sw the dispatcher rebases each seq's partition_idx 0 to
    # align_down(ctx_i - sw, partition_size), and grid_y caps at
    # cdiv(sw, partition_size) + 1 (the +1 absorbs the align-down boundary).
    sw_for_grid = _api_sliding_window(window_size)
    if sw_for_grid > 0 and max_seq_len > sw_for_grid:
        max_num_partitions = (
            sw_for_grid + _PARTITION_SIZE_ROCM - 1
        ) // _PARTITION_SIZE_ROCM + 1
    else:
        max_num_partitions = (
            max_seq_len + _PARTITION_SIZE_ROCM - 1
        ) // _PARTITION_SIZE_ROCM
    output = torch.empty_like(query)

    nbytes_per_qo_elem = torch.finfo(output.dtype).bits // 8
    workspace_buffer = torch.empty(
        (num_seqs * num_heads * max_num_partitions * head_size) * nbytes_per_qo_elem
        + 2 * (num_seqs * num_heads * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=output.device,
    )

    k_scale = torch.tensor([1.0], dtype=dtypes.fp32, device=output.device)
    v_scale = torch.tensor([1.0], dtype=dtypes.fp32, device=output.device)

    if cu_query_lens is None:
        cu_query_lens = torch.arange(
            0, num_seqs + 1, dtype=torch.int, device=output.device
        )

    torch.ops.aiter.paged_attention_v1(
        output,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        cu_query_lens,
        seq_lens,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        None,  # fp8_out_scale
        _PARTITION_SIZE_ROCM,  # partition_size
        1,  # mtp
        _api_sliding_window(window_size),  # NEW: sliding_window arg (0 = disabled)
    )
    return output


# ----------------------------- pytest -------------------------------- #


@pytest.mark.parametrize("dtype", [dtypes.bf16, dtypes.fp16])
@pytest.mark.parametrize("num_seqs", [1, 8, 16])
@pytest.mark.parametrize("num_heads", [(8, 1), (16, 4), (32, 4)])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize(
    "ctx_len,window_size",
    [
        (128, 512),
        (512, 512),
        (513, 512),
        (768, 512),
        (2048, 512),
        (2048, 1024),
        (4097, 2048),
        (9216, 8192),
        (256, 4096),
        (1024, -1),
        (4096, -1),
    ],
)
@pytest.mark.parametrize("seed", [0])
def test_pa_v1_swa_correctness(
    dtype: torch.dtype,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    block_size: int,
    ctx_len: int,
    window_size: int,
    seed: int,
) -> None:
    device = "cuda:0"
    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_default_device(device)

    num_q_heads, num_kv_heads = num_heads
    assert num_q_heads % num_kv_heads == 0
    num_queries_per_kv = num_q_heads // num_kv_heads
    scale = float(1.0 / (head_size**0.5))
    max_seq_len = ctx_len
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max(max_num_blocks_per_seq * num_seqs, 1)

    query = torch.empty(num_seqs, num_q_heads, head_size, dtype=dtype, device=device)
    query.uniform_(*UNIFORM_RANGE)

    key_caches, value_caches = kv_cache_factory(
        num_blocks, block_size, 1, num_kv_heads, head_size, dtype, seed, device
    )
    key_cache = key_caches[0]
    value_cache = value_caches[0]

    block_tables = rearrange(
        torch.randperm(num_blocks, dtype=dtypes.i32, device=device),
        "(b n) -> b n",
        b=num_seqs,
    )
    seq_lens = torch.full(
        size=(num_seqs,), fill_value=ctx_len, dtype=torch.int, device=device
    )

    key_cache_hbsd = rearrange(key_cache, "b h d1 s d2 -> b h s (d1 d2)").contiguous()
    value_cache_hbsd = rearrange(value_cache, "b h d s -> b h s d").contiguous()

    ref = run_torch_swa(
        query=query,
        key_cache_nhd=key_cache_hbsd,
        value_cache_nhd=value_cache_hbsd,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        window_size=window_size,
        num_queries_per_kv=num_queries_per_kv,
    )

    key_cache_nhd = rearrange(key_cache_hbsd, "b h s d -> b s h d").contiguous()
    value_cache_nhd = rearrange(value_cache_hbsd, "b h s d -> b s h d").contiguous()

    out = run_aiter_pa_v1_swa(
        query=query,
        key_cache=key_cache_nhd,
        value_cache=value_cache_nhd,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        window_size=window_size,
    )

    atol = 1e-2 if dtype is dtypes.bf16 else 5e-3
    rtol = 1e-2
    msg = (
        f"dtype={dtype} seqs={num_seqs} heads={num_heads} hs={head_size} "
        f"bs={block_size} ctx={ctx_len} window={window_size}"
    )
    err = checkAllclose(ref, out, atol=atol, rtol=rtol, msg=msg, printLog=False)
    assert err < 0.01, f"too many mismatches for {msg}: error_ratio={err}"


def test_pa_v1_swa_matches_full_causal_when_window_ge_context() -> None:
    """When window >= context, SWA output must equal full-causal output bit-exact."""
    device = "cuda:0"
    torch.manual_seed(123)
    torch.set_default_device(device)

    num_seqs, num_q_heads, num_kv_heads, head_size, block_size = 4, 16, 4, 128, 16
    ctx_len = 1024
    scale = float(1.0 / (head_size**0.5))
    max_num_blocks_per_seq = (ctx_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs

    query = torch.empty(
        num_seqs, num_q_heads, head_size, dtype=dtypes.bf16, device=device
    )
    query.uniform_(*UNIFORM_RANGE)
    key_caches, value_caches = kv_cache_factory(
        num_blocks, block_size, 1, num_kv_heads, head_size, dtypes.bf16, 0, device
    )
    key_cache_nhd = rearrange(
        rearrange(key_caches[0], "b h d1 s d2 -> b h s (d1 d2)"),
        "b h s d -> b s h d",
    ).contiguous()
    value_cache_nhd = rearrange(
        rearrange(value_caches[0], "b h d s -> b h s d"),
        "b h s d -> b s h d",
    ).contiguous()
    block_tables = rearrange(
        torch.randperm(num_blocks, dtype=dtypes.i32, device=device),
        "(b n) -> b n",
        b=num_seqs,
    )
    seq_lens = torch.full(
        size=(num_seqs,), fill_value=ctx_len, dtype=torch.int, device=device
    )

    out_full = run_aiter_pa_v1_swa(
        query=query,
        key_cache=key_cache_nhd,
        value_cache=value_cache_nhd,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        window_size=-1,  # disabled
    )
    out_swa_large = run_aiter_pa_v1_swa(
        query=query,
        key_cache=key_cache_nhd,
        value_cache=value_cache_nhd,
        block_tables=block_tables,
        seq_lens=seq_lens,
        scale=scale,
        window_size=ctx_len + 100,  # window > context (kv_lo <= 0, no masking)
    )
    err = checkAllclose(out_full, out_swa_large, atol=1e-3, rtol=1e-3, printLog=False)
    assert err < 1e-4, f"SWA(window>ctx) should match full causal but err={err}"


# ----------------------------- CLI ----------------------------------- #


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--window", type=int, default=1024)
    parser.add_argument("--num_seqs", type=int, default=8)
    parser.add_argument("--gqa", type=int, default=4, help="num_q_heads / num_kv_heads")
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--head_size", type=int, default=128)
    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dtype = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp16
    test_pa_v1_swa_correctness(
        dtype=dtype,
        num_seqs=args.num_seqs,
        num_heads=(args.num_kv_heads * args.gqa, args.num_kv_heads),
        head_size=args.head_size,
        block_size=args.block_size,
        ctx_len=args.ctx,
        window_size=args.window,
        seed=args.seed,
    )
    print(
        f"OK  ctx={args.ctx} window={args.window} seqs={args.num_seqs} "
        f"q_heads={args.num_kv_heads * args.gqa} kv_heads={args.num_kv_heads} "
        f"hs={args.head_size} bs={args.block_size} dtype={args.dtype}"
    )

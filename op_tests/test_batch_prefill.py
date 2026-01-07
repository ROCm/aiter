# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import math
import os
import pytest
import torch

import aiter
from aiter import dtypes
from aiter import per_tensor_quant
from einops import rearrange, repeat
import argparse

from aiter.test_common import (
    perftest,
)

def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(
        torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1"
    )
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
    window_left: int = -1,
    logits_soft_cap: float = 0.0,
) -> torch.Tensor:
    if causal:
        window_size = (window_left, 0)
    else:
        window_size = (-1, -1)

    head_dim = query.shape[2]
    seqlen_q = query.shape[0]
    seqlen_k = key.shape[0]
    scale = 1.0 / math.sqrt(head_dim)

    attn_weights = scale * torch.einsum("qhd,khd->hqk", query.float(), key.float())
    if 0 < logits_soft_cap:
        mode = int(os.environ.get("CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT", 0))
        if mode == 0:
            attn_weights = logits_soft_cap * torch.tanh(attn_weights / logits_soft_cap)
        else:
            attn_weights = attn_weights / (
                1.0 + torch.abs(attn_weights / logits_soft_cap)
            )

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            device=query.device,
        )
        attn_weights.masked_fill_(local_mask, float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    if window_size[0] >= 0 or window_size[1] >= 0:
        attn_weights = attn_weights.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    out = torch.einsum("hqk,khd->qhd", attn_weights, value.float())
    return out.to(query)


def make_scaled_rand(min_val, max_val, *shape, dtype, device="cuda"):
    x = torch.randn(*shape, device=device, dtype=dtype)
    x = (x - x.min()) / (x.max() - x.min())
    return min_val + (max_val - min_val) * x


def convert_lens_to_indptr(lens):
    return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()


def build_qo_lens(batch_size, qo_len, randomize=True):
    if randomize and batch_size > 1:
        return torch.randint(1, qo_len + 1, (batch_size,)).int()
    return torch.full((batch_size,), qo_len).int()


def build_kv_lens(batch_size, kv_len, qo_lens, randomize=True, ensure_at_least_q=True):
    if randomize and batch_size > 1:
        kv_lens = torch.randint(1, kv_len + 1, (batch_size,)).int()
        return torch.maximum(qo_lens, kv_lens) if ensure_at_least_q else kv_lens
    return torch.full((batch_size,), kv_len).int()


def build_q_tensor(
    total_q_tokens, num_qo_heads, head_dim, dtype, q_init_min, q_init_max
):
    return make_scaled_rand(
        q_init_min,
        q_init_max,
        total_q_tokens,
        num_qo_heads,
        head_dim,
        dtype=dtype,
    ).to(0)


def build_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    head_dim,
    kv_lens,
    kv_init_min,
    kv_init_max,
    dtype,
    use_uniform=False,
):
    max_num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = max_num_pages_per_seq * batch_size
    kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if use_uniform:
        kv_data_fp32 = torch.rand(*kv_shape, device="cuda", dtype=torch.float32)
        if kv_init_min is not None and kv_init_max is not None:
            kv_data_fp32 = kv_init_min + (kv_init_max - kv_init_min) * kv_data_fp32
    else:
        kv_data_fp32 = make_scaled_rand(
            kv_init_min, kv_init_max, *kv_shape, dtype=torch.float32
        ).to(0)
    kv_data = kv_data_fp32.to(dtype)
    kv_num_used_pages = (kv_lens + page_size - 1) // page_size
    kv_indptr_cpu = convert_lens_to_indptr(kv_num_used_pages)
    kv_indices_cpu = torch.nn.functional.pad(
        torch.randperm(total_num_pages).int(), (0, 128), value=0
    )
    kv_last_page_len_cpu = ((kv_lens - 1) % page_size + 1).int()
    return {
        "kv_data_fp32": kv_data_fp32,
        "kv_data": kv_data,
        "kv_indptr_cpu": kv_indptr_cpu,
        "kv_indices_cpu": kv_indices_cpu,
        "kv_last_page_len_cpu": kv_last_page_len_cpu,
        "max_num_pages_per_seq": max_num_pages_per_seq,
        "total_num_pages": total_num_pages,
    }


def split_kv_pages(kv_data):
    chunks = torch.chunk(kv_data, 2, dim=1)
    k_cache_ref = chunks[0].squeeze(1).contiguous()
    v_cache_ref = chunks[1].squeeze(1).contiguous()
    return k_cache_ref, v_cache_ref


def apply_kv_layout(
    k_cache_ref,
    v_cache_ref,
    num_kv_heads,
    head_dim,
    page_size,
    k_vector_size,
    layout,
):
    if layout == "vectorized":
        return vectorize_kv_cache(
            k_cache_ref,
            v_cache_ref,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size,
        )
    if layout == "linear":
        return k_cache_ref.contiguous(), v_cache_ref.contiguous()
    raise ValueError(f"Unsupported KV layout: {layout}")


def build_block_table(kv_indptr_cpu, kv_indices_cpu, batch_size, max_num_pages_per_seq):
    block_table_cpu = torch.zeros(
        (batch_size, max_num_pages_per_seq), dtype=torch.int32
    )
    for i in range(batch_size):
        start = kv_indptr_cpu[i].item()
        end = kv_indptr_cpu[i + 1].item()
        block_table_cpu[i, : (end - start)] = kv_indices_cpu[start:end]
    return block_table_cpu


def build_reference_output(
    q,
    q_indptr_cpu,
    kv_data_fp32,
    kv_indices_cpu,
    kv_indptr_cpu,
    kv_last_page_len_cpu,
    num_kv_heads,
    head_dim,
    dtype,
    causal,
    logits_soft_cap,
):
    o_ref_list = []
    for i in range(len(q_indptr_cpu) - 1):
        perm_dims = [0, 1, 2, 3]
        perm_dims_last = [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        used_kv_indices = kv_indices_cpu[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
        last_k = kv_data_fp32[used_kv_indices[-1], 0, : kv_last_page_len_cpu[i], :]
        last_v = kv_data_fp32[used_kv_indices[-1], 1, : kv_last_page_len_cpu[i], :]
        ki = torch.cat(
            [
                kv_data_fp32[used_kv_indices[:-1], 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                last_k.permute(*perm_dims_last).reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(dtype)
        vi = torch.cat(
            [
                kv_data_fp32[used_kv_indices[:-1], 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                last_v.permute(*perm_dims_last).reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(dtype)
        o_ref_list.append(
            ref_masked_attention(
                qi, ki, vi, causal=causal, logits_soft_cap=logits_soft_cap
            )
        )
    return torch.cat(o_ref_list, dim=0)


def assert_output_matches_reference(out, q_indptr_cpu, o_ref, rtol, atol):
    for i in range(len(q_indptr_cpu) - 1):
        start = q_indptr_cpu[i]
        end = q_indptr_cpu[i + 1]
        torch.testing.assert_close(
            out[start:end], o_ref[start:end], rtol=rtol, atol=atol
        )


@pytest.mark.parametrize("kvcache_layout", ["linear", "vectorized"])
@pytest.mark.parametrize("table_layout", ["sglang", "vllm"])
@pytest.mark.parametrize("input_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("batch_size", [1, 3, 7])
@pytest.mark.parametrize(
    "qo_len,kv_len",
    [
        (128, 128),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
        (8192, 8192),
    ],
)
@pytest.mark.parametrize("page_size", [128, 256, 1024])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (16, 1)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_init_min,q_init_max", [(-10, 10)])
@pytest.mark.parametrize("kv_init_min,kv_init_max", [(-5, 5)])
@pytest.mark.parametrize("seed", [19378])
def test_batch_prefill(
    kvcache_layout,
    table_layout,
    input_dtype,
    batch_size,
    qo_len,
    kv_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal,
    logits_soft_cap,
    dtype,
    q_init_min,
    q_init_max,
    kv_init_min,
    kv_init_max,
    seed,
    profile=False,
):
    if seed is not None:
        torch.manual_seed(seed)

    if input_dtype == "fp8" and dtype != torch.bfloat16:
        pytest.skip("FP8 tests use BF16 reference dtype only")

    if causal and kv_len < qo_len:
        pytest.skip("kv_len < qo_len is not allowed if causal=True")

    k_vector_size = 16 // torch.tensor([], dtype=dtype).element_size()
    k_vector_size_fp8 = 16 // torch.tensor([], dtype=dtypes.fp8).element_size()
    if kvcache_layout == "vectorized":
        if page_size % k_vector_size != 0 or head_dim % k_vector_size != 0:
            pytest.skip(
                "Vectorized layout requires page/head dim divisible by vector size"
            )
        if input_dtype == "fp8" and (
            page_size % k_vector_size_fp8 != 0 or head_dim % k_vector_size_fp8 != 0
        ):
            pytest.skip(
                "FP8 vectorized layout requires page/head dim divisible by vector size"
            )
    else:
        if head_dim % k_vector_size != 0:
            pytest.skip("Linear layout requires head dim divisible by vector size")
        if input_dtype == "fp8" and head_dim % k_vector_size_fp8 != 0:
            pytest.skip("FP8 linear layout requires head dim divisible by vector size")

    qo_lens = build_qo_lens(batch_size, qo_len, randomize=True)
    q_indptr_cpu = convert_lens_to_indptr(qo_lens)
    if input_dtype == "fp8":
        total_q_tokens = torch.sum(qo_lens).item()
        q = torch.rand(
            total_q_tokens, num_qo_heads, head_dim, device="cuda", dtype=dtype
        )
    else:
        q = build_q_tensor(
            batch_size * qo_len, num_qo_heads, head_dim, dtype, q_init_min, q_init_max
        )

    kv_lens = build_kv_lens(batch_size, kv_len, qo_lens, randomize=True)
    kv_init_min_use = None if input_dtype == "fp8" else kv_init_min
    kv_init_max_use = None if input_dtype == "fp8" else kv_init_max
    kv_cache = build_paged_kv_cache(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
        kv_lens,
        kv_init_min_use,
        kv_init_max_use,
        dtype,
        use_uniform=input_dtype == "fp8",
    )

    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_cache["kv_indptr_cpu"].to(0)
    kv_indices_gpu = kv_cache["kv_indices_cpu"].to(0)
    kv_last_page_len_gpu = kv_cache["kv_last_page_len_cpu"].to(0)

    k_cache_ref, v_cache_ref = split_kv_pages(kv_cache["kv_data"])

    block_table_gpu = None
    seqlen_k_gpu = None
    if table_layout == "vllm":
        block_table_cpu = build_block_table(
            kv_cache["kv_indptr_cpu"],
            kv_cache["kv_indices_cpu"],
            batch_size,
            kv_cache["max_num_pages_per_seq"],
        )
        block_table_gpu = block_table_cpu.to(0)
        seqlen_k_gpu = kv_lens.to(0).int()

    if input_dtype == "fp8":
        q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
        k_cache_quant, k_descale = per_tensor_quant(
            k_cache_ref.to(dtype), quant_dtype=dtypes.fp8
        )
        v_cache_quant, v_descale = per_tensor_quant(
            v_cache_ref.to(dtype), quant_dtype=dtypes.fp8
        )
        k_cache_quant, v_cache_quant = apply_kv_layout(
            k_cache_quant,
            v_cache_quant,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size_fp8,
            kvcache_layout,
        )
        k_cache_ref_layout, v_cache_ref_layout = apply_kv_layout(
            k_cache_ref.to(dtype),
            v_cache_ref.to(dtype),
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size,
            kvcache_layout,
        )

        out_fp8 = run_ck(
            batch_size,
            num_kv_heads,
            q_quant,
            k_cache_quant,
            v_cache_quant,
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            torch.max(qo_lens).item(),
            torch.max(kv_lens).item(),
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            kv_last_page_lens=kv_last_page_len_gpu,
            block_table=block_table_gpu,
            seqlen_k=seqlen_k_gpu,
            profile=profile,
        )
        # Reference using FP16/BF16
        out_ref = run_ck(
            batch_size,
            num_kv_heads,
            q,
            k_cache_ref_layout,
            v_cache_ref_layout,
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            torch.max(qo_lens).item(),
            torch.max(kv_lens).item(),
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            kv_last_page_lens=kv_last_page_len_gpu,
            block_table=block_table_gpu,
            seqlen_k=seqlen_k_gpu,
            profile=profile,
        )

        o_ref = build_reference_output(
            q,
            q_indptr_cpu,
            kv_cache["kv_data_fp32"],
            kv_cache["kv_indices_cpu"],
            kv_cache["kv_indptr_cpu"],
            kv_cache["kv_last_page_len_cpu"],
            num_kv_heads,
            head_dim,
            dtype,
            causal,
            logits_soft_cap,
        )

        max_diff = (out_fp8 - o_ref).abs().max().item()
        threshold = 0.055
        assert max_diff < threshold, (
            f"FP8 kernel vs reference difference too large: "
            f"{max_diff} (threshold: {threshold})"
        )
        rtol, atol = 2e-2, 1e-2
        torch.testing.assert_close(out_ref, o_ref, rtol=rtol, atol=atol)
    else:
        k_cache, v_cache = apply_kv_layout(
            k_cache_ref,
            v_cache_ref,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size,
            kvcache_layout,
        )
        out = aiter.mha_batch_prefill_func(
            q,
            k_cache,
            v_cache,
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            torch.max(qo_lens).item(),
            torch.max(kv_lens).item(),
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            kv_last_page_lens=kv_last_page_len_gpu,
            block_table=block_table_gpu,
            seqlen_k=seqlen_k_gpu,
        )
        o_ref = build_reference_output(
            q,
            q_indptr_cpu,
            kv_cache["kv_data_fp32"],
            kv_cache["kv_indices_cpu"],
            kv_cache["kv_indptr_cpu"],
            kv_cache["kv_last_page_len_cpu"],
            num_kv_heads,
            head_dim,
            dtype,
            causal,
            logits_soft_cap,
        )
        rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-2, 1e-2)
        assert_output_matches_reference(out, q_indptr_cpu, o_ref, rtol, atol)

@perftest()
def profile_func(target_func, *args, **kwargs):
    return target_func(*args, **kwargs)

def flops(
    batch,
    seqlen_q,
    seqlen_k,
    headdim_q,
    headdim_v,
    nheads_q,
    nheads_k,
    causal,
    mode="fwd",
):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    mask_area = seqlen_q * seqlen_k // (2 if causal else 1)
    qk = 2 * batch * mask_area * nheads_q * headdim_q
    # Match CK's fmha_fwd_runner.hpp which always scales PV by nheads_q,
    # even for MQA/GQA where KV heads are fewer than query heads.
    pv = 2 * batch * mask_area * nheads_q * headdim_v
    base = qk + pv
    if mode == "fwd":
        return base
    if mode == "bwd":
        return 2.5 * base
    return 3.5 * base


def efficiency(flop, time_in_us):
    return flop / time_in_us / 10**6


def run_ck(
    batch_size,
    num_kv_heads,
    q,
    k_cache,
    v_cache,
    cu_seqlens_q,
    kv_indptr,
    kv_page_indices,
    max_seqlen_q,
    max_seqlen_k,
    causal=False,
    logits_soft_cap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    kv_last_page_lens=None,
    block_table=None,
    seqlen_k=None,
    profile=False,
):
    kernel_args = (
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
    )
    kernel_kwargs = dict(
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        kv_last_page_lens=kv_last_page_lens,
        block_table=block_table,
        seqlen_k=seqlen_k,
    )

    if profile:
        out, time_us = profile_func(
            aiter.mha_batch_prefill_func, *kernel_args, **kernel_kwargs
        )
        nheads_q = q.shape[1]
        headdim = q.shape[2]
        seqlen_q = max_seqlen_q
        seqlen_k = max_seqlen_k
        total_flops = flops(
                batch_size,
                seqlen_q,
                seqlen_k,
                headdim,
                headdim,
                nheads_q,
                num_kv_heads,
                causal,
            )
        tflops = efficiency(
            total_flops,
            time_us,
        )
        print(f"time: {time_us:.2f} us, {tflops:.2f} TFlops")
    else:
        out = aiter.mha_batch_prefill_func(*kernel_args, **kernel_kwargs)

    return out


def vectorize_kv_cache(
    k_cache, v_cache, num_kv_heads, head_dim, page_size, k_vector_size
):
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    k_cache = (
        k_cache.view(
            -1, page_size, num_kv_heads, head_dim // k_vector_size, k_vector_size
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    v_cache = (
        v_cache.view(
            -1, page_size // k_vector_size, k_vector_size, num_kv_heads, head_dim
        )
        .permute(0, 3, 1, 4, 2)
        .contiguous()
    )
    return k_cache, v_cache


def varlen_to_paged_kv(k_varlen, v_varlen, kv_lens, page_size=1):
    """
    Convert varlen format K/V to paged KV cache format.

    Args:
        k_varlen: [total_tokens, num_kv_heads, head_dim]
        v_varlen: [total_tokens, num_kv_heads, head_dim]
        kv_lens: [batch_size] - length of each sequence
        page_size: tokens per page

    Returns:
        kv_data: [total_num_pages, 2, page_size, num_kv_heads, head_dim]
        kv_indptr: [batch_size + 1]
        kv_indices: [total_num_pages + padding]
    """
    batch_size = len(kv_lens)
    num_kv_heads = k_varlen.shape[1]
    head_dim = k_varlen.shape[2]
    dtype = k_varlen.dtype
    device = k_varlen.device

    # Calculate number of pages needed
    max_kv_len = kv_lens.max().item()
    max_num_pages_per_seq = (max_kv_len + page_size - 1) // page_size
    total_num_pages = max_num_pages_per_seq * batch_size

    # Create paged KV cache
    kv_data = torch.zeros(
        total_num_pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        dtype=dtype,
        device=device,
    )

    # Create page indices (identity mapping for simplicity)
    kv_indices = torch.arange(total_num_pages, dtype=torch.int32, device="cpu")
    kv_indices = torch.nn.functional.pad(kv_indices, (0, 128), value=0)

    # Fill in the data
    def convert_lens_to_indptr_local(lens):
        return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()

    kv_indptr = convert_lens_to_indptr_local(
        ((kv_lens + page_size - 1) // page_size).cpu()
    )
    cu_kv_lens = convert_lens_to_indptr_local(kv_lens.cpu())

    for batch_idx in range(batch_size):
        seq_start = cu_kv_lens[batch_idx].item()
        seq_end = cu_kv_lens[batch_idx + 1].item()
        seq_len = seq_end - seq_start

        page_start = kv_indptr[batch_idx].item()
        num_pages = kv_indptr[batch_idx + 1].item() - page_start

        # Copy K and V data into pages
        for page_idx in range(num_pages):
            global_page_idx = page_start + page_idx
            token_start = page_idx * page_size
            token_end = min(token_start + page_size, seq_len)
            tokens_in_page = token_end - token_start

            # K data
            kv_data[global_page_idx, 0, :tokens_in_page, :, :] = k_varlen[
                seq_start + token_start : seq_start + token_end, :, :
            ]

            # V data
            kv_data[global_page_idx, 1, :tokens_in_page, :, :] = v_varlen[
                seq_start + token_start : seq_start + token_end, :, :
            ]

    return kv_data, kv_indptr, kv_indices


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(6, 1), (8, 1)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize(
    "qo_len,kv_len",
    [
        (128, 128),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_batch_prefill_vs_varlen_fp8(
    batch_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    qo_len,
    kv_len,
    causal,
    logits_soft_cap,
):
    """
    Compare FP8 batch_prefill (paged KV) vs FP8 flash_attn_varlen.
    Both use qr_async pipeline with FP8, should produce identical results.
    """
    torch.manual_seed(42)
    dtype = torch.bfloat16
    quant_dtype = dtypes.fp8
    page_size = 128
    k_vector_size = 16 // torch.tensor([], dtype=quant_dtype).element_size()
    if page_size % k_vector_size != 0 or head_dim % k_vector_size != 0:
        pytest.skip("Vectorized layout requires page/head dim divisible by vector size")

    # Create Q, K, V in varlen format (BF16 first)
    if batch_size > 1:
        qo_lens = torch.randint(qo_len // 2, qo_len + 1, (batch_size,)).int()
        kv_lens = torch.maximum(
            qo_lens, torch.randint(kv_len // 2, kv_len + 1, (batch_size,))
        ).int()
    else:
        qo_lens = torch.full((batch_size,), qo_len).int()
        kv_lens = torch.full((batch_size,), kv_len).int()

    total_q_tokens = qo_lens.sum().item()
    total_kv_tokens = kv_lens.sum().item()

    q_bf16 = make_scaled_rand(
        -10, 10, total_q_tokens, num_qo_heads, head_dim, dtype=dtype
    )
    k_bf16 = make_scaled_rand(
        -5, 5, total_kv_tokens, num_kv_heads, head_dim, dtype=dtype
    )
    v_bf16 = make_scaled_rand(
        -5, 5, total_kv_tokens, num_kv_heads, head_dim, dtype=dtype
    )

    # Quantize to FP8
    q_fp8, q_descale = per_tensor_quant(q_bf16, quant_dtype=quant_dtype)
    k_fp8, k_descale = per_tensor_quant(k_bf16, quant_dtype=quant_dtype)
    v_fp8, v_descale = per_tensor_quant(v_bf16, quant_dtype=quant_dtype)

    cu_seqlens_q = convert_lens_to_indptr(qo_lens).cuda()
    cu_seqlens_k = convert_lens_to_indptr(kv_lens).cuda()

    # Run flash_attn_varlen FP8
    out_varlen = aiter.flash_attn_varlen_fp8_pertensor_func(
        q_fp8,
        k_fp8,
        v_fp8,
        q_descale,
        k_descale,
        v_descale,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=qo_lens.max().item(),
        max_seqlen_k=kv_lens.max().item(),
        min_seqlen_q=0,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        window_size=(-1, -1),
    )

    # Convert to paged KV cache format
    kv_data, kv_indptr, kv_indices = varlen_to_paged_kv(
        k_fp8, v_fp8, kv_lens, page_size=page_size
    )
    kv_last_page_len_cpu = ((kv_lens - 1) % page_size + 1).int()
    kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
    seqlen_k_gpu = kv_lens.to(0).int()
    max_num_pages_per_seq = (kv_lens.max().item() + page_size - 1) // page_size
    block_table_cpu = torch.zeros(
        (batch_size, max_num_pages_per_seq), dtype=torch.int32
    )
    for i in range(batch_size):
        start = kv_indptr[i].item()
        end = kv_indptr[i + 1].item()
        block_table_cpu[i, : (end - start)] = kv_indices[start:end]
    block_table_gpu = block_table_cpu.to(0)

    # Extract K and V from paged format
    chunks = torch.chunk(kv_data, 2, dim=1)
    k_paged, v_paged = vectorize_kv_cache(
        chunks[0].squeeze(1),
        chunks[1].squeeze(1),
        num_kv_heads,
        head_dim,
        page_size,
        k_vector_size,
    )

    # Run batch_prefill FP8
    out_batch_prefill = aiter.mha_batch_prefill_func(
        q_fp8,
        k_paged,
        v_paged,
        cu_seqlens_q,
        kv_indptr.cuda(),
        kv_indices.cuda(),
        max_seqlen_q=qo_lens.max().item(),
        max_seqlen_k=kv_lens.max().item(),
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        kv_last_page_lens=kv_last_page_len_gpu,
        block_table=block_table_gpu,
        seqlen_k=seqlen_k_gpu,
    )

    # Compare results (all tokens are valid, no padding)
    print("\n=== FP8 Comparison: batch_prefill vs varlen ===")
    print(
        f"batch_size={batch_size}, heads={num_qo_heads}/{num_kv_heads}, "
        f"dim={head_dim}, qo_len={qo_len}, kv_len={kv_len}"
    )
    print(f"causal={causal}, logits_soft_cap={logits_soft_cap}")

    # Sanity check: outputs should not be all zeros
    assert (
        out_varlen.abs().max().item() > 1e-6
    ), "Varlen output is all zeros - kernel may not have launched!"
    assert (
        out_batch_prefill.abs().max().item() > 1e-6
    ), "Batch_prefill output is all zeros - kernel may not have launched!"

    # Compute differences on entire tensor
    diff = (out_varlen - out_batch_prefill).abs()
    max_diff_all = diff.max().item()
    mean_diff_all = diff.mean().item()

    print(f"Max diff: {max_diff_all:.6e}")
    print(f"Mean diff: {mean_diff_all:.6e}")
    print(f"Varlen output max: {out_varlen.abs().max().item():.6e}")
    print(f"Batch_prefill output max: {out_batch_prefill.abs().max().item():.6e}")

    if out_varlen.abs().max().item() > 0:
        rel_error = max_diff_all / out_varlen.abs().max().item()
        print(f"Relative error: {rel_error * 100:.4f}%")

    # Should be nearly identical (same pipeline, same computation)
    # FP8 may have slightly larger tolerance
    rtol, atol = 1e-4, 1e-4
    torch.testing.assert_close(out_batch_prefill, out_varlen, rtol=rtol, atol=atol)


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-c",
    "--causal",
    type=dtypes.str2bool,
    nargs="*",
    default=[False, True],
    help="""Causal mask mode (False or True).
    e.g.: -c false""",
)
parser.add_argument(
    "-l",
    "--logits_soft_cap",
    type=float,
    choices=[0.0, 30.0],
    nargs="*",
    default=[0.0, 30.0],
    help="""Logits soft cap.
    e.g.: -l 30.0""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    choices=[dtypes.d_dtypes["fp16"], dtypes.d_dtypes["bf16"]],
    nargs="*",
    default="fp16, bf16",
    metavar="{fp16, bf16}",
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-s",
    "--seqlen",
    type=int,
    const=None,
    default=1024,
    help="""seqlen.
    e.g.: -s 1024""",
)
parser.add_argument(
    "-p",
    "--pagesize",
    type=int,
    const=None,
    default=1024,
    help="""page size.
    e.g.: -p 1024""",
)
parser.add_argument(
    "-q",
    "--headq",
    type=int,
    const=None,
    default=8,
    help="""number of q head.
    e.g.: -h 8""",
)
parser.add_argument(
    "-k",
    "--headk",
    type=int,
    const=None,
    default=8,
    help="""number of kv head.
    e.g.: -h_k 8""",
)
parser.add_argument(
    "-t",
    "--lookup_table",
    type=str,
    const=None,
    choices=["sglang", "vllm"],
    default=["sglang"],
    nargs="*",
    help="""lookup table.
    e.g.: -t sglang""",
)
parser.add_argument(
    "--kv_layout",
    type=str,
    const=None,
    choices=["vectorized", "linear"],
    default=["vectorized"],
    nargs="*",
    help="""kv cache table.
    e.g.: -o vectorized""",
)
parser.add_argument(
    "--input_dtype",
    type=str,
    const=None,
    choices=[dtypes.d_dtypes["fp16"], dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    default="bf16, fp8",
    nargs="*",
    help="""input dtype.
    e.g.: -o bf16""",
)
parser.add_argument(
    "--profile",
    action="store_true",
    help="Enable profiling mode",
)


if __name__ == "__main__":
    args = parser.parse_args()

    for (
        causal,
        logits_soft_cap,
        dtype,
        lookup_table,
        kv_layout,
        input_dtype,
    ) in itertools.product(
        args.causal,
        args.logits_soft_cap,
        args.dtype,
        args.lookup_table,
        args.kv_layout,
        args.input_dtype,
    ):
        print(
            f"causal={causal}, logits_soft_cap={logits_soft_cap}, dtype={dtype}, lookup_table={lookup_table}, kv_layout={kv_layout}, input_dtype={input_dtype}"
        )
        test_batch_prefill(
            kvcache_layout=kv_layout,
            table_layout=lookup_table,
            input_dtype=input_dtype,
            batch_size=1,
            qo_len=args.seqlen,
            kv_len=args.seqlen,
            page_size=args.pagesize,
            num_qo_heads=args.headq,
            num_kv_heads=args.headk,
            head_dim=128,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            dtype=dtype,
            q_init_min=-10,
            q_init_max=10,
            kv_init_min=-5,
            kv_init_max=5,
            seed=19378,
            profile=args.profile,
        )

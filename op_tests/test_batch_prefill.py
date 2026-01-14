# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import math
import os
import pytest
import torch

import pandas as pd

import aiter
from aiter import dtypes
from aiter import per_tensor_quant
from einops import rearrange, repeat
import argparse

from aiter.test_common import (
    perftest,
)


def skip_test_if(condition: bool, reason: str) -> bool:
    """
    Skip the test if condition is True.

    Works in both pytest and direct python execution:
    - pytest session: calls pytest.skip()
    - direct python: prints message and returns True

    Usage:
        if skip_test_if(causal and kv_len < qo_len, "reason"):
            return

    Returns:
        True if test should be skipped (caller should return early)
    """
    if not condition:
        return False

    # PYTEST_CURRENT_TEST is only set when pytest is actively running tests,
    # not when pytest is just imported. This is the reliable way to detect
    # if we're inside a pytest session.
    if "PYTEST_CURRENT_TEST" in os.environ:
        pytest.skip(reason)

    print(f"SKIP: {reason}")
    return True


def get_vector_size(dtype) -> int:
    """Calculate vector size for a given dtype (16 bytes / element_size)."""
    return 16 // torch.tensor([], dtype=dtype).element_size()


def check_common_skip_conditions(
    is_input_fp8: bool,
    dtype,
    causal: bool,
    kv_len: int,
    qo_len: int,
    contiguous_kv: bool,
) -> bool:
    """
    Check common skip conditions shared across test functions.
    Returns True if test should be skipped.
    """
    if skip_test_if(
        is_input_fp8 and dtype != torch.bfloat16,
        "FP8 tests use BF16 reference dtype only",
    ):
        return True

    if skip_test_if(
        causal and kv_len < qo_len,
        "kv_len < qo_len is not allowed if causal=True",
    ):
        return True

    if skip_test_if(
        not contiguous_kv and is_input_fp8,
        "Non-contiguous KV is only validated for non-FP8 path",
    ):
        return True

    return False


def check_layout_skip_conditions(
    kvcache_layout: str,
    head_dim: int,
    page_size: int,
    k_vector_size: int,
    k_vector_size_fp8: int,
    is_input_fp8: bool,
    contiguous_kv: bool,
) -> bool:
    """
    Check layout-specific skip conditions.
    Returns True if test should be skipped.
    """
    if kvcache_layout == "vectorized":
        if skip_test_if(
            not contiguous_kv,
            "Non-contiguous KV is only validated for linear layout",
        ):
            return True
        if skip_test_if(
            page_size % k_vector_size != 0 or head_dim % k_vector_size != 0,
            "Vectorized layout requires page/head dim divisible by vector size",
        ):
            return True
        if skip_test_if(
            is_input_fp8
            and (
                page_size % k_vector_size_fp8 != 0 or head_dim % k_vector_size_fp8 != 0
            ),
            "FP8 vectorized layout requires page/head dim divisible by vector size",
        ):
            return True
    else:
        if skip_test_if(
            head_dim % k_vector_size != 0,
            "Linear layout requires head dim divisible by vector size",
        ):
            return True
        if skip_test_if(
            is_input_fp8 and head_dim % k_vector_size_fp8 != 0,
            "FP8 linear layout requires head dim divisible by vector size",
        ):
            return True

    return False


def get_tolerances(dtype, is_fp8: bool = False) -> tuple[float, float]:
    """Return (rtol, atol) tolerances based on dtype and FP8 mode."""
    if is_fp8:
        return 2e-2, 1e-2
    if dtype == torch.float16:
        return 1e-3, 1e-3
    return 2e-2, 1e-2


def build_q_tensor_for_test(
    qo_lens,
    batch_size: int,
    qo_len: int,
    num_qo_heads: int,
    head_dim: int,
    dtype,
    q_init_min: float,
    q_init_max: float,
    is_input_fp8: bool,
):
    """Build Q tensor, handling both FP8 and non-FP8 cases."""
    if is_input_fp8:
        total_q_tokens = torch.sum(qo_lens).item()
        return torch.rand(
            total_q_tokens, num_qo_heads, head_dim, device="cuda", dtype=dtype
        )
    return build_q_tensor(
        batch_size * qo_len, num_qo_heads, head_dim, dtype, q_init_min, q_init_max
    )


def extract_kv_caches(kv_cache: dict, contiguous_kv: bool):
    """Extract K and V reference tensors from KV cache dict."""
    if contiguous_kv:
        return split_kv_pages(kv_cache["kv_data"])
    return kv_cache["kv_data"][:, 0], kv_cache["kv_data"][:, 1]


def verify_fp8_output(out_fp8, o_ref, threshold: float = 0.055):
    """Verify FP8 kernel output against reference."""
    max_diff = (out_fp8 - o_ref).abs().max().item()
    assert max_diff < threshold, (
        f"FP8 kernel vs reference difference too large: "
        f"{max_diff} (threshold: {threshold})"
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
    contiguous_kv=True,
):
    max_num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = max_num_pages_per_seq * batch_size
    kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if contiguous_kv:
        if use_uniform:
            kv_data_fp32 = torch.rand(*kv_shape, device="cuda", dtype=torch.float32)
            if kv_init_min is not None and kv_init_max is not None:
                kv_data_fp32 = kv_init_min + (kv_init_max - kv_init_min) * kv_data_fp32
        else:
            kv_data_fp32 = make_scaled_rand(
                kv_init_min, kv_init_max, *kv_shape, dtype=torch.float32
            ).to(0)
        kv_data = kv_data_fp32.to(dtype)
    else:
        kv_shape_nc = [kv_shape[0]]
        for dim in kv_shape[1:]:
            kv_shape_nc.append(2)
            kv_shape_nc.append(dim)
        if use_uniform:
            kv_data_fp32 = torch.rand(*kv_shape_nc, device="cuda", dtype=torch.float32)
            if kv_init_min is not None and kv_init_max is not None:
                kv_data_fp32 = kv_init_min + (kv_init_max - kv_init_min) * kv_data_fp32
        else:
            kv_data_fp32 = make_scaled_rand(
                kv_init_min, kv_init_max, *kv_shape_nc, dtype=torch.float32
            ).to(0)
        kv_data = kv_data_fp32.to(dtype)
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
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


@pytest.mark.parametrize("input_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("batch_size", [1, 3, 7])
@pytest.mark.parametrize(
    "qo_len,kv_len",
    [
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(6, 1), (3, 1)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_init_min,q_init_max", [(-10, 10)])
@pytest.mark.parametrize("kv_init_min,kv_init_max", [(-5, 5)])
@pytest.mark.parametrize("kv_dim", [4, 3])
@pytest.mark.parametrize("contiguous_kv", [True, False])
@pytest.mark.parametrize("seed", [19378])
def test_batch_prefill_page_size_1_linear_sglang(
    input_dtype,
    batch_size,
    kv_len,
    qo_len,
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
    kv_dim,
    contiguous_kv,
    seed,
):
    if seed is not None:
        torch.manual_seed(seed)

    is_input_fp8 = input_dtype == dtypes.fp8 or input_dtype == "fp8"
    k_vector_size = get_vector_size(dtype)
    k_vector_size_fp8 = get_vector_size(dtypes.fp8)
    page_size = 1

    # Skip conditions
    if check_common_skip_conditions(
        is_input_fp8, dtype, causal, kv_len, qo_len, contiguous_kv
    ):
        return
    if check_layout_skip_conditions(
        "linear",
        head_dim,
        page_size,
        k_vector_size,
        k_vector_size_fp8,
        is_input_fp8,
        contiguous_kv,
    ):
        return

    # Build test tensors
    qo_lens = build_qo_lens(batch_size, qo_len, randomize=True)
    q_indptr_cpu = convert_lens_to_indptr(qo_lens)
    q = build_q_tensor_for_test(
        qo_lens,
        batch_size,
        qo_len,
        num_qo_heads,
        head_dim,
        dtype,
        q_init_min,
        q_init_max,
        is_input_fp8,
    )

    kv_lens = build_kv_lens(batch_size, kv_len, qo_lens, randomize=True)
    kv_cache = build_paged_kv_cache(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
        kv_lens,
        None if is_input_fp8 else kv_init_min,
        None if is_input_fp8 else kv_init_max,
        dtype,
        use_uniform=is_input_fp8,
        contiguous_kv=contiguous_kv,
    )

    # Move to GPU
    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_cache["kv_indptr_cpu"].to(0)
    kv_indices_gpu = kv_cache["kv_indices_cpu"].to(0)
    kv_last_page_len_gpu = kv_cache["kv_last_page_len_cpu"].to(0)

    k_cache_ref, v_cache_ref = extract_kv_caches(kv_cache, contiguous_kv)
    max_qo_len = torch.max(qo_lens).item()
    max_kv_len = torch.max(kv_lens).item()

    # Build reference output (shared between FP8 and non-FP8)
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

    if is_input_fp8:
        q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
        k_cache_quant, k_descale = per_tensor_quant(
            k_cache_ref.to(dtype), quant_dtype=dtypes.fp8
        )
        v_cache_quant, v_descale = per_tensor_quant(
            v_cache_ref.to(dtype), quant_dtype=dtypes.fp8
        )

        # Apply layout based on kv_dim
        if kv_dim == 3:
            k_cache_fp8 = k_cache_quant.squeeze(1).contiguous()
            v_cache_fp8 = v_cache_quant.squeeze(1).contiguous()
            k_cache_ref_layout = k_cache_ref.squeeze(1).contiguous()
            v_cache_ref_layout = v_cache_ref.squeeze(1).contiguous()
        else:
            k_cache_fp8, v_cache_fp8 = apply_kv_layout(
                k_cache_quant,
                v_cache_quant,
                num_kv_heads,
                head_dim,
                page_size,
                k_vector_size_fp8,
                "linear",
            )
            k_cache_ref_layout, v_cache_ref_layout = apply_kv_layout(
                k_cache_ref.to(dtype),
                v_cache_ref.to(dtype),
                num_kv_heads,
                head_dim,
                page_size,
                k_vector_size,
                "linear",
            )

        out_fp8 = aiter.mha_batch_prefill_func(
            q_quant,
            k_cache_fp8,
            v_cache_fp8,
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            max_qo_len,
            max_kv_len,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            kv_last_page_lens=kv_last_page_len_gpu,
        )

        out_ref = aiter.mha_batch_prefill_func(
            q,
            k_cache_ref_layout,
            v_cache_ref_layout,
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            max_qo_len,
            max_kv_len,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            kv_last_page_lens=kv_last_page_len_gpu,
        )

        verify_fp8_output(out_fp8, o_ref)
        rtol, atol = get_tolerances(dtype, is_fp8=True)
        torch.testing.assert_close(out_ref, o_ref, rtol=rtol, atol=atol)
    else:
        # Prepare KV cache based on kv_dim and contiguity
        if kv_dim == 3:
            k_cache = k_cache_ref.squeeze(1)
            v_cache = v_cache_ref.squeeze(1)
            if contiguous_kv:
                k_cache = k_cache.contiguous()
                v_cache = v_cache.contiguous()
        elif contiguous_kv:
            k_cache, v_cache = apply_kv_layout(
                k_cache_ref,
                v_cache_ref,
                num_kv_heads,
                head_dim,
                page_size,
                k_vector_size,
                "linear",
            )
        else:
            k_cache, v_cache = k_cache_ref, v_cache_ref

        # Verify contiguity expectations
        assert k_cache.is_contiguous() == contiguous_kv
        assert v_cache.is_contiguous() == contiguous_kv

        out = aiter.mha_batch_prefill_func(
            q,
            k_cache,
            v_cache,
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            max_qo_len,
            max_kv_len,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            kv_last_page_lens=kv_last_page_len_gpu,
        )
        rtol, atol = get_tolerances(dtype)
        assert_output_matches_reference(out, q_indptr_cpu, o_ref, rtol, atol)


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
@pytest.mark.parametrize("contiguous_kv", [True, False])
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
    contiguous_kv,
    seed,
    profile=False,
):
    if seed is not None:
        torch.manual_seed(seed)

    is_input_fp8 = input_dtype == dtypes.fp8 or input_dtype == "fp8"
    k_vector_size = get_vector_size(dtype)
    k_vector_size_fp8 = get_vector_size(dtypes.fp8)

    # Skip conditions
    if check_common_skip_conditions(
        is_input_fp8, dtype, causal, kv_len, qo_len, contiguous_kv
    ):
        return {"status": "skipped"}
    if check_layout_skip_conditions(
        kvcache_layout,
        head_dim,
        page_size,
        k_vector_size,
        k_vector_size_fp8,
        is_input_fp8,
        contiguous_kv,
    ):
        return {"status": "skipped"}

    # Build test tensors
    qo_lens = build_qo_lens(batch_size, qo_len, randomize=True)
    q_indptr_cpu = convert_lens_to_indptr(qo_lens)
    q = build_q_tensor_for_test(
        qo_lens,
        batch_size,
        qo_len,
        num_qo_heads,
        head_dim,
        dtype,
        q_init_min,
        q_init_max,
        is_input_fp8,
    )

    kv_lens = build_kv_lens(batch_size, kv_len, qo_lens, randomize=True)
    kv_cache = build_paged_kv_cache(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
        kv_lens,
        None if is_input_fp8 else kv_init_min,
        None if is_input_fp8 else kv_init_max,
        dtype,
        use_uniform=is_input_fp8,
        contiguous_kv=contiguous_kv,
    )

    # Move to GPU
    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_cache["kv_indptr_cpu"].to(0)
    kv_indices_gpu = kv_cache["kv_indices_cpu"].to(0)
    kv_last_page_len_gpu = kv_cache["kv_last_page_len_cpu"].to(0)

    k_cache_ref, v_cache_ref = extract_kv_caches(kv_cache, contiguous_kv)
    max_qo_len = torch.max(qo_lens).item()
    max_kv_len = torch.max(kv_lens).item()

    # Build vLLM-style block table if needed
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

    # Build reference output (shared between FP8 and non-FP8)
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

    profile_result = {"status": "passed"}

    if is_input_fp8:
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

        # Run FP8 kernel (with optional profiling)
        fp8_result = run_ck(
            batch_size,
            num_kv_heads,
            q_quant,
            k_cache_quant,
            v_cache_quant,
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            max_qo_len,
            max_kv_len,
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
        if profile:
            out_fp8, time_us, tflops = fp8_result
            profile_result = {"status": "passed", "time_us": time_us, "tflops": tflops}
        else:
            out_fp8 = fp8_result

        # Run reference (BF16/FP16) - no profiling for reference
        out_ref = run_ck(
            batch_size,
            num_kv_heads,
            q,
            k_cache_ref_layout,
            v_cache_ref_layout,
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            max_qo_len,
            max_kv_len,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            kv_last_page_lens=kv_last_page_len_gpu,
            block_table=block_table_gpu,
            seqlen_k=seqlen_k_gpu,
            profile=False,
        )

        verify_fp8_output(out_fp8, o_ref)
        rtol, atol = get_tolerances(dtype, is_fp8=True)
        torch.testing.assert_close(out_ref, o_ref, rtol=rtol, atol=atol)
    else:
        # Prepare KV cache based on layout and contiguity
        if kvcache_layout == "linear" and not contiguous_kv:
            k_cache, v_cache = k_cache_ref, v_cache_ref
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

        # Verify contiguity for linear layout
        if kvcache_layout == "linear":
            assert k_cache.is_contiguous() == contiguous_kv
            assert v_cache.is_contiguous() == contiguous_kv

        # Run kernel (with optional profiling)
        run_result = run_ck(
            batch_size,
            num_kv_heads,
            q,
            k_cache,
            v_cache,
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            max_qo_len,
            max_kv_len,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            kv_last_page_lens=kv_last_page_len_gpu,
            block_table=block_table_gpu,
            seqlen_k=seqlen_k_gpu,
            profile=profile,
        )
        if profile:
            out, time_us, tflops = run_result
            profile_result = {"status": "passed", "time_us": time_us, "tflops": tflops}
        else:
            out = run_result

        rtol, atol = get_tolerances(dtype)
        assert_output_matches_reference(out, q_indptr_cpu, o_ref, rtol, atol)

    # Suppress return value in pytest to avoid PytestReturnNotNoneWarning
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    return profile_result


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
    """
    Run CK kernel with optional profiling.

    Returns:
        If profile=False: out tensor
        If profile=True: (out tensor, time_us, tflops)
    """
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
        total_flops = flops(
            batch_size,
            max_seqlen_q,
            max_seqlen_k,
            headdim,
            headdim,
            nheads_q,
            num_kv_heads,
            causal,
        )
        tflops = efficiency(total_flops, time_us)
        return out, time_us, tflops
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
    kv_indptr = convert_lens_to_indptr(((kv_lens + page_size - 1) // page_size).cpu())
    cu_kv_lens = convert_lens_to_indptr(kv_lens.cpu())

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
    k_vector_size = get_vector_size(quant_dtype)

    if skip_test_if(
        page_size % k_vector_size != 0 or head_dim % k_vector_size != 0,
        "Vectorized layout requires page/head dim divisible by vector size",
    ):
        return

    # Build sequence lengths
    qo_lens = build_qo_lens(batch_size, qo_len, randomize=batch_size > 1)
    kv_lens = build_kv_lens(batch_size, kv_len, qo_lens, randomize=batch_size > 1)
    total_q_tokens = qo_lens.sum().item()
    total_kv_tokens = kv_lens.sum().item()
    max_qo_len = qo_lens.max().item()
    max_kv_len = kv_lens.max().item()

    # Create Q, K, V in varlen format
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
        max_seqlen_q=max_qo_len,
        max_seqlen_k=max_kv_len,
        min_seqlen_q=0,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        window_size=(-1, -1),
    )

    # Convert to paged KV cache format
    kv_data, kv_indptr, kv_indices = varlen_to_paged_kv(
        k_fp8, v_fp8, kv_lens, page_size=page_size
    )
    kv_last_page_len_gpu = ((kv_lens - 1) % page_size + 1).int().to(0)
    seqlen_k_gpu = kv_lens.to(0).int()
    max_num_pages_per_seq = (max_kv_len + page_size - 1) // page_size

    # Build block table
    block_table_cpu = torch.zeros(
        (batch_size, max_num_pages_per_seq), dtype=torch.int32
    )
    for i in range(batch_size):
        start, end = kv_indptr[i].item(), kv_indptr[i + 1].item()
        block_table_cpu[i, : (end - start)] = kv_indices[start:end]
    block_table_gpu = block_table_cpu.to(0)

    # Extract and vectorize K/V from paged format
    k_cache_raw, v_cache_raw = split_kv_pages(kv_data)
    k_paged, v_paged = vectorize_kv_cache(
        k_cache_raw,
        v_cache_raw,
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
        max_seqlen_q=max_qo_len,
        max_seqlen_k=max_kv_len,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        kv_last_page_lens=kv_last_page_len_gpu,
        block_table=block_table_gpu,
        seqlen_k=seqlen_k_gpu,
    )

    # Sanity check: outputs should not be all zeros
    assert (
        out_varlen.abs().max().item() > 1e-6
    ), "Varlen output is all zeros - kernel may not have launched!"
    assert (
        out_batch_prefill.abs().max().item() > 1e-6
    ), "Batch_prefill output is all zeros - kernel may not have launched!"

    # Should be nearly identical (same pipeline, same computation)
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
    choices=[1, 1024],
    default=[1, 1024],
    nargs="*",
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
    default=["sglang", "vllm"],
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
    type=dtypes.str2Dtype,
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

    collected = []
    for (
        page_size,
        causal,
        logits_soft_cap,
        dtype,
        lookup_table,
        kv_layout,
        input_dtype,
        contiguous_kv,
    ) in itertools.product(
        args.pagesize,
        args.causal,
        args.logits_soft_cap,
        args.dtype,
        args.lookup_table,
        args.kv_layout,
        args.input_dtype,
        [True, False],  # contiguous_kv
    ):
        result = test_batch_prefill(
            kvcache_layout=kv_layout,
            table_layout=lookup_table,
            input_dtype=input_dtype,
            batch_size=1,
            qo_len=args.seqlen,
            kv_len=args.seqlen,
            page_size=page_size,
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
            contiguous_kv=contiguous_kv,
            seed=19378,
            profile=args.profile,
        )

        # Build result row
        time_us = result.get("time_us") if result else None
        tflops = result.get("tflops") if result else None
        row = {
            "seqlen": args.seqlen,
            "page_sz": page_size,
            "h_q": args.headq,
            "h_kv": args.headk,
            "hdim": 128,
            "input_dtype": str(input_dtype).split(".")[-1],
            "kv_layout": kv_layout,
            "table": lookup_table,
            "causal": causal,
            "soft_cap": logits_soft_cap,
            "contig": contiguous_kv,
            "status": result.get("status", "passed") if result else "passed",
            "time_us": f"{time_us:.2f}" if time_us is not None else "-",
            "tflops": f"{tflops:.2f}" if tflops is not None else "-",
        }

        collected.append(row)

    # Print summary
    df = pd.DataFrame(collected)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.float_format", lambda x: f"{x:.2f}")

    print("\n" + "=" * 100)
    aiter.logger.info(f"\n=== Batch Prefill Summary ===\n{df.to_string(index=False)}")

    # Print statistics
    passed = df[df["status"] == "passed"].shape[0]
    skipped = df[df["status"] == "skipped"].shape[0]
    total = len(collected)
    print(f"\nTotal: {total}, Passed: {passed}, Skipped: {skipped}")
    print("=" * 100)

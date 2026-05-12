# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import ctypes
import itertools
import math
import os
import weakref

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


def get_rocm_version():
    """
    Get ROCm version from PyTorch.

    Returns:
        tuple (major, minor) or None if not using ROCm

    Example:
        >>> get_rocm_version()
        (7, 2)  # ROCm 7.2
    """
    if not torch.version.hip:
        return None

    try:
        # torch.version.hip returns string like "6.2.41133" or "6.2.41133-rocm6.2.2"
        hip_version = torch.version.hip
        parts = hip_version.split(".")
        if len(parts) >= 2:
            return (int(parts[0]), int(parts[1]))
    except (ValueError, AttributeError):
        pass

    return None


def get_gpu_arch():
    """
    Get GPU architecture (gcnArchName).

    Returns:
        str like "gfx942", "gfx950", etc., or None if cannot determine

    Example:
        >>> get_gpu_arch()
        "gfx950"
    """
    if not torch.cuda.is_available():
        return None

    try:
        # Get device properties
        props = torch.cuda.get_device_properties(0)
        # gcnArchName property contains architecture like "gfx942:sramecc+:xnack-"
        if hasattr(props, "gcnArchName"):
            arch_name = props.gcnArchName
            # Extract base architecture (e.g., "gfx950" from "gfx950:sramecc+:xnack-")
            if ":" in arch_name:
                return arch_name.split(":")[0]
            return arch_name
    except (AttributeError, RuntimeError):
        pass

    return None


def should_skip_rocm72_issue(causal, logits_soft_cap):
    """
    Check if test should be skipped due to ROCm 7.2 + gfx950 compiler issue.

    FIXME: ROCm 7.2 on gfx950 has a compiler bug with causal=True + logits_soft_cap=0.0
    configuration. This workaround should be removed once the compiler is fixed.

    Args:
        causal: Whether causal masking is enabled
        logits_soft_cap: Soft cap value for logits

    Returns:
        True if test should be skipped on current ROCm version + GPU architecture
    """
    # Only check if the problematic configuration is used
    if not (causal and logits_soft_cap == 0.0):
        return False

    # Check ROCm version
    rocm_version = get_rocm_version()
    if rocm_version is None:
        return False  # Not ROCm, no need to skip

    # Check GPU architecture
    gpu_arch = get_gpu_arch()
    if gpu_arch is None:
        return False  # Cannot determine GPU, no need to skip

    # Only skip on ROCm 7.2.x + gfx950
    major, minor = rocm_version
    if (major, minor) == (7, 2) and gpu_arch == "gfx950":
        return True

    return False


def check_common_skip_conditions(
    is_input_fp8: bool,
    return_lse: bool = False,
) -> bool:
    """
    Check common skip conditions shared across test functions.
    Returns True if test should be skipped.
    """

    # FP8 is inference-only, no backward pass needed, so LSE is not required
    if skip_test_if(
        is_input_fp8 and return_lse,
        "FP8 is inference-only, LSE not needed for backward pass",
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
    # Use actual sum of qo_lens as total_q_tokens for correct shape
    total_q_tokens = torch.sum(qo_lens).item()
    if is_input_fp8:
        return torch.rand(
            total_q_tokens, num_qo_heads, head_dim, device="cuda", dtype=dtype
        )
    return build_q_tensor(
        total_q_tokens, num_qo_heads, head_dim, dtype, q_init_min, q_init_max
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
    return_lse: bool = False,
) -> torch.Tensor:
    """
    Reference implementation of masked attention.

    Args:
        query: [seqlen_q, num_heads, head_dim]
        key: [seqlen_k, num_heads, head_dim]
        value: [seqlen_k, num_heads, head_dim]
        causal: whether to use causal mask
        window_left: left window size for sliding window attention
        logits_soft_cap: soft cap for logits (0.0 = disabled)
        return_lse: whether to return log-sum-exp values

    Returns:
        If return_lse=False: output [seqlen_q, num_heads, head_dim]
        If return_lse=True: (output, lse) where lse is [num_heads, seqlen_q]
    """
    if causal:
        window_size = (window_left, 0)
    else:
        window_size = (-1, -1)

    head_dim = query.shape[2]
    seqlen_q = query.shape[0]
    seqlen_k = key.shape[0]
    scale = 1.0 / math.sqrt(head_dim)

    # Compute scaled attention scores: [num_heads, seqlen_q, seqlen_k]
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

    # Compute LSE before softmax using torch.logsumexp
    # This correctly handles fully-masked rows (all -inf) by returning -inf instead of nan
    if return_lse:
        # attn_weights: [num_heads, seqlen_q, seqlen_k]
        lse = torch.logsumexp(attn_weights, dim=-1)  # [H, Q]

    attn_weights = torch.softmax(attn_weights, dim=-1)
    if window_size[0] >= 0 or window_size[1] >= 0:
        attn_weights = attn_weights.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    out = torch.einsum("hqk,khd->qhd", attn_weights, value.float())

    if return_lse:
        return out.to(query), lse.float()
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
    return_lse=False,
):
    """
    Build reference output (and optionally LSE) for batch prefill.

    Args:
        return_lse: If True, also return LSE values.

    Returns:
        If return_lse=False: output tensor [total_q, num_heads, head_dim]
        If return_lse=True: (output, lse) where lse is [total_q, num_heads]
    """
    o_ref_list = []
    lse_ref_list = []
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
        if qi.shape[1] != num_kv_heads:
            assert qi.shape[1] % num_kv_heads == 0
            ratio = qi.shape[1] // num_kv_heads
            ki = ki.repeat_interleave(ratio, dim=1)
            vi = vi.repeat_interleave(ratio, dim=1)

        result = ref_masked_attention(
            qi,
            ki,
            vi,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            return_lse=return_lse,
        )
        if return_lse:
            o_ref_list.append(result[0])
            # ref_masked_attention returns lse as [num_heads, seqlen_q]
            # kernel also returns [num_heads, total_q], so no transpose needed
            lse_ref_list.append(result[1])
        else:
            o_ref_list.append(result)

    if return_lse:
        # Concatenate along the seqlen dimension (dim=1 for [num_heads, seqlen_q])
        return torch.cat(o_ref_list, dim=0), torch.cat(lse_ref_list, dim=1)
    return torch.cat(o_ref_list, dim=0)


def assert_output_matches_reference(out, q_indptr_cpu, o_ref, rtol, atol):
    for i in range(len(q_indptr_cpu) - 1):
        start = q_indptr_cpu[i]
        end = q_indptr_cpu[i + 1]
        torch.testing.assert_close(
            out[start:end], o_ref[start:end], rtol=rtol, atol=atol
        )


def assert_lse_matches_reference(
    lse_kernel: torch.Tensor,
    lse_ref: torch.Tensor,
    rtol: float = 1e-3,
    atol: float = 1e-3,
):
    """
    Compare kernel LSE output against reference LSE.

    Both should be [total_q, num_heads] and float32.
    Uses same tolerance logic as CK's fmha_fwd_runner.hpp.
    """
    assert (
        lse_kernel.shape == lse_ref.shape
    ), f"LSE shape mismatch: kernel={lse_kernel.shape}, ref={lse_ref.shape}"
    assert (
        lse_kernel.dtype == torch.float32
    ), f"Kernel LSE should be float32, got {lse_kernel.dtype}"
    assert (
        lse_ref.dtype == torch.float32
    ), f"Reference LSE should be float32, got {lse_ref.dtype}"

    # CK's check_err with allow_infinity_ref=true
    torch.testing.assert_close(
        lse_kernel,
        lse_ref,
        rtol=rtol,
        atol=atol,
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
@pytest.mark.parametrize("return_lse", [False, True])
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
    return_lse,
    seed,
):
    if seed is not None:
        torch.manual_seed(seed)

    is_input_fp8 = input_dtype == dtypes.fp8 or input_dtype == "fp8"
    k_vector_size = get_vector_size(dtype)
    k_vector_size_fp8 = get_vector_size(dtypes.fp8)
    page_size = 1

    # Skip conditions
    if check_common_skip_conditions(is_input_fp8, return_lse):
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

    if skip_test_if(
        should_skip_rocm72_issue(causal, logits_soft_cap),
        "ROCm 7.2 + gfx950 compiler issue with causal=True + logits_soft_cap=0.0",
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
    ref_result = build_reference_output(
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
        return_lse=return_lse,
    )
    if return_lse:
        o_ref, lse_ref = ref_result
    else:
        o_ref = ref_result
        lse_ref = None

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

        # Note: FP8 is inference-only, LSE not needed
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

        # Causal + kv_len < qo_len: rows with few valid K positions amplify
        # FP8 quantization error (not averaged over many attention targets).
        # Larger head_dim accumulates more rounding error in dot products
        # (CK's own FP8BF16 atol is 0.18 for reference).
        fp8_threshold = 0.06 if causal and kv_len < qo_len else 0.055
        if head_dim > 128:
            fp8_threshold = max(fp8_threshold, 0.06)
        verify_fp8_output(out_fp8, o_ref, threshold=fp8_threshold)
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

        kernel_result = aiter.mha_batch_prefill_func(
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
            return_lse=return_lse,
        )
        if return_lse:
            out, lse_kernel = kernel_result
        else:
            out = kernel_result
            lse_kernel = None

        rtol, atol = get_tolerances(dtype)
        assert_output_matches_reference(out, q_indptr_cpu, o_ref, rtol, atol)

        # Compare LSE if requested
        if return_lse:
            assert_lse_matches_reference(lse_kernel, lse_ref)


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
@pytest.mark.parametrize("page_size", [16, 1024])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (16, 1)])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_init_min,q_init_max", [(-10, 10)])
@pytest.mark.parametrize("kv_init_min,kv_init_max", [(-5, 5)])
@pytest.mark.parametrize("contiguous_kv", [True, False])
@pytest.mark.parametrize("return_lse", [False, True])
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
    return_lse,
    seed,
    profile=False,
):
    if seed is not None:
        torch.manual_seed(seed)

    is_input_fp8 = input_dtype == dtypes.fp8 or input_dtype == "fp8"
    k_vector_size = get_vector_size(dtype)
    k_vector_size_fp8 = get_vector_size(dtypes.fp8)

    # Skip conditions
    if check_common_skip_conditions(is_input_fp8, return_lse):
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

    if skip_test_if(
        should_skip_rocm72_issue(causal, logits_soft_cap),
        "ROCm 7.2 + gfx950 compiler issue with causal=True + logits_soft_cap=0.0",
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
    ref_result = build_reference_output(
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
        return_lse=return_lse,
    )
    if return_lse:
        o_ref, lse_ref = ref_result
    else:
        o_ref = ref_result
        lse_ref = None

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
        # Note: FP8 is inference-only, LSE not needed
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

        # Causal + kv_len < qo_len: rows with few valid K positions amplify
        # FP8 quantization error (not averaged over many attention targets).
        # Larger head_dim accumulates more rounding error in dot products
        # (CK's own FP8BF16 atol is 0.18 for reference).
        fp8_threshold = 0.06 if causal and kv_len < qo_len else 0.055
        if head_dim > 128:
            fp8_threshold = max(fp8_threshold, 0.06)
        verify_fp8_output(out_fp8, o_ref, threshold=fp8_threshold)
        rtol, atol = get_tolerances(dtype, is_fp8=False)
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

        # Run kernel (with optional profiling and LSE)
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
            return_lse=return_lse,
        )
        if profile:
            if return_lse:
                out, lse_kernel, time_us, tflops = run_result
            else:
                out, time_us, tflops = run_result
                lse_kernel = None
            profile_result = {"status": "passed", "time_us": time_us, "tflops": tflops}
        else:
            if return_lse:
                out, lse_kernel = run_result
            else:
                out = run_result
                lse_kernel = None

        rtol, atol = get_tolerances(dtype)
        assert_output_matches_reference(out, q_indptr_cpu, o_ref, rtol, atol)

        # Compare LSE if requested
        if return_lse:
            assert_lse_matches_reference(lse_kernel, lse_ref)

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
    kv_block_descale=None,
    kv_last_page_lens=None,
    block_table=None,
    seqlen_k=None,
    profile=False,
    return_lse=False,
):
    """
    Run CK kernel with optional profiling and LSE output.

    Returns:
        If profile=False and return_lse=False: out tensor
        If profile=False and return_lse=True: (out tensor, lse tensor)
        If profile=True and return_lse=False: (out tensor, time_us, tflops)
        If profile=True and return_lse=True: (out tensor, lse tensor, time_us, tflops)
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
        kv_block_descale=kv_block_descale,
        kv_last_page_lens=kv_last_page_lens,
        block_table=block_table,
        seqlen_k=seqlen_k,
        return_lse=return_lse,
    )

    if profile:
        result, time_us = profile_func(
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
        if return_lse:
            out, lse = result
            return out, lse, time_us, tflops
        else:
            return result, time_us, tflops
    else:
        result = aiter.mha_batch_prefill_func(*kernel_args, **kernel_kwargs)
        return result


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


@pytest.mark.parametrize("table_layout", ["sglang", "vllm"])
@pytest.mark.parametrize("input_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize(
    "qo_len,kv_len",
    [
        (128, 128),
        (1024, 1024),
        (2048, 2048),
        (4096, 4096),
    ],
)
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (16, 1)])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_batch_prefill_linear_vs_vectorized(
    table_layout,
    input_dtype,
    batch_size,
    qo_len,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal,
    logits_soft_cap,
):
    """
    Compare LINEAR vs VECTORIZED layout batch_prefill output.

    Both layouts represent the same logical KV data. Outputs should be
    consistent regardless of memory layout. Uses a tighter tolerance than
    the FP32 reference tests to catch layout-specific regressions.
    """
    torch.manual_seed(42)
    dtype = torch.bfloat16
    is_input_fp8 = input_dtype == dtypes.fp8 or input_dtype == "fp8"
    page_size = 1024
    k_vector_size = get_vector_size(dtype)
    k_vector_size_fp8 = get_vector_size(dtypes.fp8)

    if skip_test_if(
        should_skip_rocm72_issue(causal, logits_soft_cap),
        "ROCm 7.2 + gfx950 compiler issue with causal=True + logits_soft_cap=0.0",
    ):
        return

    # Build test tensors
    qo_lens = build_qo_lens(batch_size, qo_len, randomize=batch_size > 1)
    kv_lens = build_kv_lens(batch_size, kv_len, qo_lens, randomize=batch_size > 1)
    q_indptr_cpu = convert_lens_to_indptr(qo_lens)
    q = build_q_tensor_for_test(
        qo_lens,
        batch_size,
        qo_len,
        num_qo_heads,
        head_dim,
        dtype,
        -10,
        10,
        is_input_fp8,
    )

    kv_cache = build_paged_kv_cache(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
        kv_lens,
        None if is_input_fp8 else -5,
        None if is_input_fp8 else 5,
        dtype,
        use_uniform=is_input_fp8,
        contiguous_kv=True,
    )

    # Move to GPU
    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_cache["kv_indptr_cpu"].to(0)
    kv_indices_gpu = kv_cache["kv_indices_cpu"].to(0)
    kv_last_page_len_gpu = kv_cache["kv_last_page_len_cpu"].to(0)
    max_qo_len = torch.max(qo_lens).item()
    max_kv_len = torch.max(kv_lens).item()

    k_cache_ref, v_cache_ref = extract_kv_caches(kv_cache, contiguous_kv=True)

    # Build vLLM block table if needed
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

    if is_input_fp8:
        q_quant, q_descale = per_tensor_quant(q, quant_dtype=dtypes.fp8)
        k_cache_quant, k_descale = per_tensor_quant(
            k_cache_ref.to(dtype), quant_dtype=dtypes.fp8
        )
        v_cache_quant, v_descale = per_tensor_quant(
            v_cache_ref.to(dtype), quant_dtype=dtypes.fp8
        )

        # LINEAR layout (dispatches V3)
        k_linear = k_cache_quant.contiguous()
        v_linear = v_cache_quant.contiguous()

        # VECTORIZED layout (dispatches V2)
        k_vec, v_vec = vectorize_kv_cache(
            k_cache_quant,
            v_cache_quant,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size_fp8,
        )

        out_linear = run_ck(
            batch_size,
            num_kv_heads,
            q_quant,
            k_linear,
            v_linear,
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
        )
        out_vec = run_ck(
            batch_size,
            num_kv_heads,
            q_quant,
            k_vec,
            v_vec,
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
        )
    else:
        # LINEAR layout (dispatches V3)
        k_linear, v_linear = apply_kv_layout(
            k_cache_ref,
            v_cache_ref,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size,
            "linear",
        )

        # VECTORIZED layout (dispatches V2)
        k_vec, v_vec = apply_kv_layout(
            k_cache_ref,
            v_cache_ref,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size,
            "vectorized",
        )

        out_linear = run_ck(
            batch_size,
            num_kv_heads,
            q,
            k_linear,
            v_linear,
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
        )
        out_vec = run_ck(
            batch_size,
            num_kv_heads,
            q,
            k_vec,
            v_vec,
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
        )

    # Sanity checks
    assert out_linear.abs().max().item() > 1e-6, "LINEAR output is all zeros!"
    assert out_vec.abs().max().item() > 1e-6, "VECTORIZED output is all zeros!"

    # LINEAR and VECTORIZED should produce consistent results
    # Same data, same computation, only memory layout differs
    max_diff = (out_linear - out_vec).abs().max().item()
    threshold = 0.017
    assert max_diff < threshold, (
        f"LINEAR vs VECTORIZED difference too large: "
        f"{max_diff} (threshold: {threshold})"
    )


def per_page_quant(tensor, page_size, quant_dtype):
    """
    Quantize tensor with per-page scale.

    Args:
        tensor: [num_pages, page_size, num_heads, head_dim]
        page_size: tokens per page
        quant_dtype: target quantization dtype

    Returns:
        quantized: quantized tensor [num_pages, page_size, num_heads, head_dim]
        descales: [num_pages, num_heads] per-page descale factors
    """
    num_pages, ps, num_heads, head_dim = tensor.shape
    assert ps == page_size

    # Compute per-page max absolute value
    # [num_pages, page_size, num_heads, head_dim] -> [num_pages, num_heads]
    abs_max = tensor.abs().amax(dim=(1, 3))  # max over page_size and head_dim
    abs_max = abs_max.clamp(min=1e-12)

    # Get FP8 max value
    fp8_max = torch.finfo(quant_dtype).max

    # Compute descale = abs_max / fp8_max (must be float32 for kernel)
    descales = (abs_max / fp8_max).float()  # [num_pages, num_heads]

    # Quantize: q = round(x / descale)
    # Broadcast descales: [num_pages, 1, num_heads, 1]
    descales_broadcast = descales.unsqueeze(1).unsqueeze(-1)
    quantized = (tensor / descales_broadcast).to(quant_dtype)

    return quantized, descales


def reference_attention_kv_blockscale(
    q_fp8,
    k_fp8,
    v_fp8,
    q_descale,
    kv_block_descale,
    cu_seqlens_q,
    kv_indptr,
    kv_indices,
    kv_lens,
    page_size,
    causal=False,
    softmax_scale=None,
    logits_soft_cap=0.0,
):
    """
    Reference implementation of attention with per-page KV descale.

    Args:
        q_fp8: [total_q, num_heads, head_dim] FP8
        k_fp8: [num_pages, page_size, num_kv_heads, head_dim] FP8
        v_fp8: [num_pages, page_size, num_kv_heads, head_dim] FP8
        q_descale: [1] per-tensor Q descale
        kv_block_descale: [num_pages, num_kv_heads, 2] per-page K/V descales
        cu_seqlens_q: [batch_size + 1]
        kv_indptr: [batch_size + 1]
        kv_indices: page indices
        kv_lens: [batch_size] K/V sequence lengths
        page_size: tokens per page
        causal: whether to use causal mask
        softmax_scale: attention scale (default: 1/sqrt(head_dim))
        logits_soft_cap: soft cap for logits (0.0 = disabled)

    Returns:
        output: [total_q, num_heads, head_dim]
    """
    import math

    batch_size = len(kv_lens)
    num_heads = q_fp8.shape[1]
    num_kv_heads = k_fp8.shape[2]
    head_dim = q_fp8.shape[2]
    head_ratio = num_heads // num_kv_heads

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    # Dequantize Q
    q = q_fp8.float() * q_descale.item()

    # Build output tensor
    total_q = q_fp8.shape[0]
    output = torch.zeros(
        total_q, num_heads, head_dim, dtype=torch.float32, device=q_fp8.device
    )

    for batch_idx in range(batch_size):
        q_start = cu_seqlens_q[batch_idx].item()
        q_end = cu_seqlens_q[batch_idx + 1].item()
        q_len = q_end - q_start

        page_start = kv_indptr[batch_idx].item()
        page_end = kv_indptr[batch_idx + 1].item()
        kv_len = kv_lens[batch_idx].item()

        q_batch = q[q_start:q_end]

        # Gather and dequantize K/V from pages
        k_batch = []
        v_batch = []
        for page_offset in range(page_end - page_start):
            page_idx = kv_indices[page_start + page_offset].item()
            token_start = page_offset * page_size
            token_end = min(token_start + page_size, kv_len)
            num_tokens = token_end - token_start

            k_page = k_fp8[page_idx, :num_tokens].float()
            v_page = v_fp8[page_idx, :num_tokens].float()

            # Apply per-page descale
            k_descale = kv_block_descale[page_idx, :, 0]
            v_descale_page = kv_block_descale[page_idx, :, 1]

            k_page = k_page * k_descale.unsqueeze(0).unsqueeze(-1)
            v_page = v_page * v_descale_page.unsqueeze(0).unsqueeze(-1)

            k_batch.append(k_page)
            v_batch.append(v_page)

        k_batch = torch.cat(k_batch, dim=0)
        v_batch = torch.cat(v_batch, dim=0)

        # Expand K/V for GQA
        if head_ratio > 1:
            k_batch = k_batch.unsqueeze(2).expand(-1, -1, head_ratio, -1)
            k_batch = k_batch.reshape(kv_len, num_heads, head_dim)
            v_batch = v_batch.unsqueeze(2).expand(-1, -1, head_ratio, -1)
            v_batch = v_batch.reshape(kv_len, num_heads, head_dim)

        # Compute attention scores
        scores = torch.einsum("qhd,khd->hqk", q_batch, k_batch) * softmax_scale

        # Apply logits soft cap
        if logits_soft_cap > 0.0:
            scores = logits_soft_cap * torch.tanh(scores / logits_soft_cap)

        # Apply causal mask
        if causal:
            mask = torch.triu(
                torch.ones(q_len, kv_len, device=scores.device),
                diagonal=kv_len - q_len + 1,
            )
            scores = scores.masked_fill(mask.unsqueeze(0).bool(), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out_batch = torch.einsum("hqk,khd->qhd", attn, v_batch)
        output[q_start:q_end] = out_batch

    return output.to(torch.bfloat16)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_cache_size_gb", [4.5])
@pytest.mark.parametrize("page_size", [1, 16, 1024])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 8), (16, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("input_dtype", ["bf16", "fp8"])
# scatter_pages=True: adjacent logical tokens map to physically distant pages,
# stress-testing the paged KV cache addressing when pages span large physical distances.
@pytest.mark.parametrize("scatter_pages", [False, True])
@pytest.mark.parametrize("kv_layout", ["linear", "vectorized"])
# quant_mode: "pertensor" uses single Q/K/V scale (existing behavior);
# "kv_blockscale" uses per-page K/V scales.
@pytest.mark.parametrize("quant_mode", ["pertensor", "kv_blockscale"])
def test_batch_prefill_large_kvcache(
    batch_size,
    kv_cache_size_gb,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal,
    input_dtype,
    scatter_pages,
    kv_layout,
    quant_mode,
):
    """
    Test that batch prefill produces correct results with large KV caches
    whose element offsets exceed the INT32_MAX boundary.

    Uses the full KV cache for attention with pages spanning the overflow
    boundary, and compares kernel output against SDPA reference.
    For page_size < kN0 (128), this validates the per-tile SRD rebase path.

    Args:
        batch_size: Number of sequences. >1 partitions the >2GB page pool
            across batches, exercising the per-sequence SRD rebase path.
        scatter_pages: If True, interleave page indices so adjacent logical
            tokens map to physically distant pages (stress-tests rebase).
        kv_layout: "linear" or "vectorized" KV cache memory layout.
    """
    # page_size=1 only supports linear layout (3D tensor)
    if page_size == 1 and kv_layout == "vectorized":
        pytest.skip("page_size=1 does not support vectorized layout")

    # Skip otherwise so the parametrize matrix doesn't generate dead cells.
    if quant_mode == "kv_blockscale":
        if input_dtype != "fp8":
            pytest.skip("KV_BLOCKSCALE requires fp8 input")
        if page_size != 1024:
            pytest.skip("KV_BLOCKSCALE requires page_size=1024")

    torch.manual_seed(42)
    torch.cuda.empty_cache()

    is_fp8 = input_dtype == "fp8"
    dtype = torch.bfloat16

    # Compute num_blocks from target KV cache size
    elem_size = 1 if is_fp8 else 2  # fp8=1 byte, bf16=2 bytes
    elements_per_block = page_size * num_kv_heads * head_dim
    target_bytes = int(kv_cache_size_gb * 1024**3)
    num_blocks = target_bytes // (elements_per_block * elem_size)

    # Verify this config triggers overflow
    stride_per_page = elements_per_block
    max_offset = (num_blocks - 1) * stride_per_page
    INT32_MAX = 2**31 - 1
    if max_offset <= INT32_MAX:
        pytest.skip(
            f"max_offset {max_offset} doesn't exceed INT32_MAX, not an overflow test"
        )

    # Check available GPU memory
    free_mem = torch.cuda.mem_get_info()[0]
    # Per-batch page partition: uniform split, remainder absorbed by the last
    # sequence to keep all kv_indptr deltas > 0 (zero-length sequences would be
    # skipped by the kernel's per-batch dispatch and hide any rebase bug).
    blocks_per_seq = [num_blocks // batch_size] * batch_size
    blocks_per_seq[-1] += num_blocks % batch_size
    kv_lens_per_seq = [bps * page_size for bps in blocks_per_seq]
    max_kv_len_per_seq = max(kv_lens_per_seq)
    # Causal with attn_mask forces SDPA math backend which materializes
    # [H_q, qo_len, kv_len] score + mask tensors. Magnitudes empirically chosen:
    #   non-causal: 1024 -- flash backend, no full score matrix, headroom is large
    #   causal:      128 -- math backend cliff: 3x [H_q, qo, kv] fp32 buffers must
    #                      fit alongside K/V cache (kv_len up to ~5GB at this scale)
    # qo_len is per-batch; total qo tokens = batch_size * qo_len.
    qo_len = min(128, max_kv_len_per_seq) if causal else min(1024, max_kv_len_per_seq)
    total_qo_len = batch_size * qo_len
    # SDPA causal with attn_mask forces math backend: expanded mask + score matrix
    # + softmax intermediates, each [1, H_q, qo, kv_per_batch] fp32. ~3x overhead.
    # The per-batch SDPA loop allocates one batch's worth at a time (kv_len
    # divided by batch_size), then frees before the next iteration.
    sdpa_causal_mem = (
        3 * num_qo_heads * qo_len * max_kv_len_per_seq * 4 if causal else 0
    )
    # GQA expands K/V from H_kv to H_q heads for SDPA reference
    gqa_ratio = num_qo_heads // num_kv_heads
    # Sequential pages reuse K/V directly; scattered need a gathered copy
    gathered_mem = 2 * num_blocks * elements_per_block * 2 if scatter_pages else 0
    required_mem = (
        2 * num_blocks * elements_per_block * 2  # K/V bf16
        + 2 * num_blocks * elements_per_block * elem_size  # kernel K/V (fp8 or bf16)
        + gathered_mem
        + 2 * num_blocks * elements_per_block * 2 * (gqa_ratio - 1)  # GQA K/V expansion
        + sdpa_causal_mem
    )
    if free_mem < required_mem * 1.1:
        pytest.skip(
            f"Not enough GPU memory: need {required_mem / 1e9:.1f}GB, "
            f"have {free_mem / 1e9:.1f}GB"
        )

    # Allocate KV caches in bf16
    # page_size=1 uses 3D linear layout [num_tokens, num_kv_heads, head_dim]
    # page_size>1 uses 4D paged layout [num_blocks, page_size, num_kv_heads, head_dim]
    if page_size == 1:
        kv_shape = (num_blocks, num_kv_heads, head_dim)
    else:
        kv_shape = (num_blocks, page_size, num_kv_heads, head_dim)

    k_cache_bf16 = torch.randn(*kv_shape, device="cuda", dtype=dtype)
    if scatter_pages:
        # Use page-dependent V values to detect address wrapping bugs.
        # With random V, wrong addresses read statistically similar data -> false pass.
        # With V[page] ? page_index, wrapped addresses (low pages) give ~0 instead of
        # the correct ~1 for high pages, making the error detectable.
        page_vals = (
            torch.arange(num_blocks, device="cuda", dtype=torch.float32) / num_blocks
        )
        if page_size == 1:
            v_cache_bf16 = page_vals.view(-1, 1, 1).expand(*kv_shape).to(dtype)
        else:
            v_cache_bf16 = page_vals.view(-1, 1, 1, 1).expand(*kv_shape).to(dtype)
    else:
        v_cache_bf16 = torch.randn(*kv_shape, device="cuda", dtype=dtype)

    # Query: flat [total_qo_len, H_q, D] layout matching mha_batch_prefill_func
    # input contract. Per-batch slices recovered via cu_seqlens_q in the loop below.
    q_bf16 = torch.randn(
        total_qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype
    )

    # Page indices: since the buffer exceeds INT32_MAX elements, these pages
    # naturally span the overflow boundary.
    overflow_page = INT32_MAX // stride_per_page

    if scatter_pages:
        # Interleave: [0, N-1, 1, N-2, 2, N-3, ...] so adjacent logical tokens
        # map to physically distant pages (low <-> high, spanning >2GB gap).
        lo = torch.arange(0, num_blocks, 2, dtype=torch.int32)
        hi = torch.arange(num_blocks - 1, -1, -2, dtype=torch.int32)
        page_indices = torch.zeros(num_blocks, dtype=torch.int32)
        page_indices[0::2] = lo[: (num_blocks + 1) // 2]
        page_indices[1::2] = hi[: num_blocks // 2]
    else:
        # Sequential: [0, 1, 2, ..., N-1]
        page_indices = torch.arange(num_blocks, dtype=torch.int32)

    # --- Step 1: Compute SDPA reference FIRST (while bf16 data is alive) ---
    # Per-batch loop: each iteration gathers its slice of pages, runs SDPA,
    # and frees intermediates before the next batch. Keeps peak memory at
    # one batch's worth (vs. materializing the full multi-batch score tensor).
    o_ref_list = []
    page_offset = 0
    for b in range(batch_size):
        n_blocks_b = blocks_per_seq[b]
        page_slice_b = page_indices[page_offset : page_offset + n_blocks_b]
        page_offset += n_blocks_b
        kv_len_b = kv_lens_per_seq[b]

        # Always gather: even sequential pages need a per-batch slice to keep
        # the multi-batch SDPA references aligned with the kernel's per-batch
        # SRD rebase. (For batch_size=1 + sequential, this is just an alias
        # of the full cache via the index slice.)
        if page_size == 1:
            k_ref_b = k_cache_bf16[page_slice_b.long()]
            v_ref_b = v_cache_bf16[page_slice_b.long()]
        else:
            k_ref_b = k_cache_bf16[page_slice_b.long()].reshape(
                -1, num_kv_heads, head_dim
            )
            v_ref_b = v_cache_bf16[page_slice_b.long()].reshape(
                -1, num_kv_heads, head_dim
            )

        q_b = q_bf16[b * qo_len : (b + 1) * qo_len]

        # SDPA expects [batch, heads, seq, dim]
        q_sdpa = q_b.unsqueeze(0).transpose(1, 2)
        k_sdpa = k_ref_b.unsqueeze(0).transpose(1, 2)
        v_sdpa = v_ref_b.unsqueeze(0).transpose(1, 2)
        del k_ref_b, v_ref_b

        # GQA: manual K/V head expansion (see comment in non-multi-batch
        # equivalent removed in this commit -- using enable_gqa=True with
        # causal attn_mask forces SDPA math backend and OOMs for large kv_len).
        if num_qo_heads != num_kv_heads:
            ratio = num_qo_heads // num_kv_heads
            k_sdpa = k_sdpa.repeat_interleave(ratio, dim=1)
            v_sdpa = v_sdpa.repeat_interleave(ratio, dim=1)

        sdpa_kwargs = {}
        if causal:
            # CK batch prefill causal: Q is at the END of the KV context.
            # Q[i] can see K[j] where j <= (kv_len_b - qo_len) + i.
            offset = kv_len_b - qo_len
            row_idx = torch.arange(qo_len, device="cuda").unsqueeze(1)
            col_idx = torch.arange(kv_len_b, device="cuda").unsqueeze(0)
            sdpa_kwargs["attn_mask"] = col_idx <= (offset + row_idx)

        o_b = (
            torch.nn.functional.scaled_dot_product_attention(
                q_sdpa, k_sdpa, v_sdpa, **sdpa_kwargs
            )
            .squeeze(0)
            .transpose(0, 1)
        )
        o_ref_list.append(o_b)
        del q_sdpa, k_sdpa, v_sdpa, sdpa_kwargs
        torch.cuda.empty_cache()

    o_ref = torch.cat(o_ref_list, dim=0)
    del o_ref_list
    torch.cuda.empty_cache()

    # --- Step 2: Prepare kernel inputs (quantize for FP8, free bf16 after) ---
    if is_fp8:
        # Q is always per-tensor quantized (both quant modes).
        q_kernel, q_descale = per_tensor_quant(q_bf16, quant_dtype=dtypes.fp8)
        if quant_mode == "kv_blockscale":
            # KV_BLOCKSCALE: per-page K/V scales. Requires 4D paged shape
            # [num_blocks, page_size, num_kv_heads, head_dim] -- guaranteed by
            # the page_size=1024 skip above.
            k_cache_kernel, k_descales = per_page_quant(
                k_cache_bf16, page_size, dtypes.fp8
            )
            v_cache_kernel, v_descales = per_page_quant(
                v_cache_bf16, page_size, dtypes.fp8
            )
            # kv_block_descale: [num_blocks, num_kv_heads, 2] (K in [..,0], V in [..,1])
            kv_block_descale = torch.stack([k_descales, v_descales], dim=-1)
            k_descale = v_descale = None
        else:
            k_cache_kernel, k_descale = per_tensor_quant(
                k_cache_bf16, quant_dtype=dtypes.fp8
            )
            v_cache_kernel, v_descale = per_tensor_quant(
                v_cache_bf16, quant_dtype=dtypes.fp8
            )
            kv_block_descale = None
        del k_cache_bf16, v_cache_bf16, q_bf16
        torch.cuda.empty_cache()
    else:
        k_cache_kernel = k_cache_bf16
        v_cache_kernel = v_cache_bf16
        q_kernel = q_bf16
        kv_block_descale = None

    # Apply vectorized layout transformation if needed
    if kv_layout == "vectorized" and page_size > 1:
        kv_vector_size = 16 // k_cache_kernel.element_size()
        k_cache_kernel, v_cache_kernel = apply_kv_layout(
            k_cache_kernel,
            v_cache_kernel,
            num_kv_heads,
            head_dim,
            page_size,
            kv_vector_size,
            "vectorized",
        )

    # Multi-batch indptrs: cu_seqlens_q is the cumulative qo offset per batch
    # (uniform qo_len), kv_indptr is the cumulative page count per batch.
    cu_seqlens_q = torch.tensor(
        [0] + [(i + 1) * qo_len for i in range(batch_size)],
        device="cuda",
        dtype=torch.int32,
    )
    kv_indptr = torch.tensor(
        [0] + list(itertools.accumulate(blocks_per_seq)),
        device="cuda",
        dtype=torch.int32,
    )
    # +256 padding is a batch_prefill ABI requirement: the kernel may speculatively
    # read up to 256 entries past the last valid page index (one bn0=256 tile worth)
    # before the bounds check kicks in. Padding with 0 keeps reads in-bounds; the
    # values are masked out by causal/length logic and never affect the output.
    kv_page_indices = torch.nn.functional.pad(page_indices, (0, 256), value=0).to(
        "cuda"
    )
    kv_last_page_lens = torch.tensor(
        [page_size] * batch_size, device="cuda", dtype=torch.int32
    )

    # --- Step 3: Run CK kernel ---
    extra_kwargs = {}
    if is_fp8:
        if quant_mode == "kv_blockscale":
            extra_kwargs = dict(q_descale=q_descale, kv_block_descale=kv_block_descale)
        else:
            extra_kwargs = dict(
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale
            )

    result = aiter.mha_batch_prefill_func(
        q_kernel,
        k_cache_kernel,
        v_cache_kernel,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        qo_len,
        max_kv_len_per_seq,
        causal=causal,
        kv_last_page_lens=kv_last_page_lens,
        **extra_kwargs,
    )
    # Synchronize immediately to catch async GPU faults from CK kernel before
    # they cascade. Without this sync, an async fault can surface inside the
    # next test's torch.cuda.empty_cache() (or any other CUDA call), causing
    # the failure to be misattributed to that unrelated test -- and on bad
    # faults the cascade can trigger a GPU reset that wipes out subsequent
    # test results too.
    torch.cuda.synchronize()
    out = result[0] if isinstance(result, (list, tuple)) else result

    # Compare kernel output vs SDPA reference
    if is_fp8:
        verify_fp8_output(out, o_ref, threshold=0.055)
    else:
        rtol, atol = get_tolerances(dtype)
        torch.testing.assert_close(
            out,
            o_ref,
            rtol=rtol,
            atol=atol,
            msg=lambda msg: (
                f"[{input_dtype}] batch_size={batch_size} "
                f"page_size={page_size} num_pages={num_blocks} "
                f"(overflow at page {overflow_page}): {msg}"
            ),
        )


# Targeted boundary detector. Companion to test_batch_prefill_large_kvcache:
# the latter exercises *correctness under stress* with 4M-token sequences,
# but those long sequences dilute single-page-corruption bugs below the
# threshold (one bad page contributes ~2e-4 to attention output).
# This test picks exactly 2 contiguous pages at byte-offset boundaries
# (factor*overflow_page) so kv_len=2048 and ALL attention math runs through
# the suspect pages -- wrong reads produce max_diff well above threshold.
@pytest.mark.parametrize("input_dtype", ["bf16", "fp8"])
@pytest.mark.parametrize("quant_mode", ["pertensor", "kv_blockscale"])
@pytest.mark.parametrize("page_offset_factor", [1, 2])
@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_4gb_boundary_targeted(
    input_dtype, quant_mode, page_offset_factor, causal
):
    """Targeted 2-page boundary probe for >4GB KV cache offset bugs.

    Args:
        input_dtype:
            "bf16", "fp8"
        quant_mode (only meaningful for fp8; bf16 ignores and skips kv_blockscale):
            "pertensor"
            "kv_blockscale"
        page_offset_factor:
            1 - first page at overflow_page boundary (byte offset = 2^31)
            2 - first page at 2x overflow boundary (byte offset = 2^32 exactly,
                int32-wrap-to-zero edge)
        causal: standard causal attention toggle.
    """

    # bf16 has no descale -> quant_mode is meaningless. Run only the pertensor
    # combo to avoid duplicated bf16 runs across the matrix.
    if input_dtype == "bf16" and quant_mode == "kv_blockscale":
        pytest.skip("bf16 has no quant_mode (no descale); pertensor combo covers bf16")

    # Skip the known ROCm 7.2 + gfx950 compiler bug: causal=True + logits_soft_cap=0.0
    # produces wrong output for bf16 + multi-Q (qo>=2) due to SGPR spill in the
    # generated kernel. Cross-validated on gfx942 (MI308X, same ROCm 7.2.26015):
    # gfx942 PASSES with max_diff=0.001 vs gfx950 FAILS with max_diff=0.05-1.78.
    # Same skip rule used by 5 other batch_prefill tests in this file.
    # Logits soft cap is 0.0 by default in this targeted boundary test.
    if should_skip_rocm72_issue(causal, logits_soft_cap=0.0):
        pytest.skip(
            "ROCm 7.2 + gfx950 compiler bug (SGPR spill) with causal=True + "
            "logits_soft_cap=0.0; cross-validated PASS on gfx942 with same source"
        )

    torch.manual_seed(42)
    torch.cuda.empty_cache()

    is_fp8 = input_dtype == "fp8"
    elem_size = 1 if is_fp8 else 2  # fp8=1B, bf16=2B

    # bf16/PERTENSOR share the shape for parity / direct dispatch comparison.
    num_blocks = 5000
    page_size = 1024
    num_kv_heads = 8
    num_qo_heads = 8
    head_dim = 128
    qo_len = 128

    # bytes_per_page = page_size x num_kv_heads x head_dim x elem_size.
    # overflow_page = first page index where byte_offset >= 2^31.
    # fp8: 1MB/page -> overflow_page=2048. bf16: 2MB/page -> overflow_page=1024.
    bytes_per_page = page_size * num_kv_heads * head_dim * elem_size
    overflow_page = (2**31) // bytes_per_page

    vh_start = overflow_page * page_offset_factor
    if vh_start + 2 > num_blocks:
        pytest.skip(
            f"page_offset_factor={page_offset_factor} -> vh_start={vh_start} "
            f"out of range for num_blocks={num_blocks}"
        )

    # Memory budget check.
    # bf16: 2x bf16 KV (10GB each = 20GB total, kept for kernel + reference).
    # fp8:  bf16 source (2x10GB) + fp8 quantized (2x5GB) at peak = 30GB peak.
    free_mem = torch.cuda.mem_get_info()[0]
    bf16_kv_bytes = num_blocks * page_size * num_kv_heads * head_dim * 2  # per K or V
    if is_fp8:
        # peak: bf16 source (2x) + fp8 quantized (2x) before del bf16
        required_mem = 2 * bf16_kv_bytes + 2 * (bf16_kv_bytes // 2)  # bf16+fp8
    else:
        required_mem = 2 * bf16_kv_bytes  # K+V bf16
    if free_mem < required_mem * 1.2:
        pytest.skip(
            f"Not enough GPU memory: need {required_mem / 1e9:.1f}GB, "
            f"have {free_mem / 1e9:.1f}GB"
        )

    device = "cuda"
    dtype = torch.bfloat16
    quant_dtype = dtypes.fp8

    # Allocate bf16 source (always -- kept alive for fp8 dequant reference too)
    k_bf16 = torch.randn(
        num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    v_bf16 = torch.randn(
        num_blocks, page_size, num_kv_heads, head_dim, device=device, dtype=dtype
    )
    q_bf16 = torch.randn(qo_len, num_qo_heads, head_dim, device=device, dtype=dtype)

    # Build kernel inputs. bf16: pass tensors as-is. fp8: quantize per quant_mode.
    if is_fp8:
        if quant_mode == "kv_blockscale":
            k_kernel, k_descales = per_page_quant(k_bf16, page_size, quant_dtype)
            v_kernel, v_descales = per_page_quant(v_bf16, page_size, quant_dtype)
            kv_block_descale = torch.stack([k_descales, v_descales], dim=-1)
            k_descale = v_descale = None
        else:  # pertensor
            k_kernel, k_descale = per_tensor_quant(k_bf16, quant_dtype=quant_dtype)
            v_kernel, v_descale = per_tensor_quant(v_bf16, quant_dtype=quant_dtype)
            kv_block_descale = None
        q_kernel, q_descale = per_tensor_quant(q_bf16, quant_dtype=quant_dtype)
        del k_bf16, v_bf16, q_bf16
        torch.cuda.empty_cache()
    else:
        k_kernel = k_bf16
        v_kernel = v_bf16
        q_kernel = q_bf16

    # Page table: single batch, 2 pages [vh_start, vh_start+1]
    page_indices = [vh_start, vh_start + 1]
    kv_len = page_size * 2  # 2048 tokens -- short enough to avoid dilution

    cu_seqlens_q = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 2], dtype=torch.int32, device=device)
    # Kernel ABI: pad page_indices buffer to avoid OOB speculative reads.
    kv_page_indices = torch.nn.functional.pad(
        torch.tensor(page_indices, dtype=torch.int32), (0, 256), value=0
    ).to(device)
    kv_last_page_lens = torch.tensor([page_size], dtype=torch.int32, device=device)

    # Reference: gather the 2 selected pages and compute SDPA in float32.
    # For fp8: dequantize first (mirrors what the kernel does internally).
    # For bf16: use raw page data directly.
    if is_fp8:
        q_for_ref = q_kernel.float() * q_descale.item()
    else:
        q_for_ref = q_kernel.float()
    k_ref_pages, v_ref_pages = [], []
    for pidx in page_indices:
        if is_fp8:
            if quant_mode == "kv_blockscale":
                k_page = k_kernel[pidx].float() * k_descales[pidx].unsqueeze(
                    0
                ).unsqueeze(-1)
                v_page = v_kernel[pidx].float() * v_descales[pidx].unsqueeze(
                    0
                ).unsqueeze(-1)
            else:
                k_page = k_kernel[pidx].float() * k_descale.item()
                v_page = v_kernel[pidx].float() * v_descale.item()
        else:
            k_page = k_kernel[pidx].float()
            v_page = v_kernel[pidx].float()
        k_ref_pages.append(k_page)
        v_ref_pages.append(v_page)

    k_ref = torch.cat(k_ref_pages, dim=0)  # [2048, 8, 128]
    v_ref = torch.cat(v_ref_pages, dim=0)

    # Use PyTorch SDPA with explicit attn_mask -- same approach as
    # test_batch_prefill_large_kvcache. Manual softmax + torch.triu produces
    # subtle numerical differences from SDPA on small kv_len (off-by-one in the
    # boundary region), which would falsely fail the causal cases here even
    # though the kernel and the SDPA reference agree.
    # SDPA expects [B, H, S, D]; reshape from [S, H, D] -> [1, H, S, D].
    q_sdpa = q_for_ref.unsqueeze(0).transpose(1, 2)
    k_sdpa = k_ref.unsqueeze(0).transpose(1, 2)
    v_sdpa = v_ref.unsqueeze(0).transpose(1, 2)
    sdpa_kwargs = {}
    if causal:
        # CK batch prefill causal: Q is at the END of the KV context.
        # Q[i] sees K[j] when j <= (kv_len - qo_len) + i.
        sq = q_for_ref.shape[0]
        sk = k_ref.shape[0]
        offset = sk - sq
        row_idx = torch.arange(sq, device=device).unsqueeze(1)
        col_idx = torch.arange(sk, device=device).unsqueeze(0)
        sdpa_kwargs["attn_mask"] = col_idx <= (offset + row_idx)
    o_b = torch.nn.functional.scaled_dot_product_attention(
        q_sdpa, k_sdpa, v_sdpa, **sdpa_kwargs
    )
    o_ref = o_b.transpose(1, 2).squeeze(0).to(torch.bfloat16)  # [S, H, D]

    # Build descale kwargs for kernel call (bf16 has none).
    extra_kwargs = {}
    if is_fp8:
        if quant_mode == "kv_blockscale":
            extra_kwargs = dict(q_descale=q_descale, kv_block_descale=kv_block_descale)
        else:
            extra_kwargs = dict(
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale
            )

    out = aiter.mha_batch_prefill_func(
        q_kernel,
        k_kernel,
        v_kernel,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q=qo_len,
        max_seqlen_k=kv_len,
        causal=causal,
        kv_last_page_lens=kv_last_page_lens,
        **extra_kwargs,
    )
    torch.cuda.synchronize()  # surface async faults before the next test masks them

    if is_fp8:
        verify_fp8_output(out.float(), o_ref.float(), threshold=0.055)
    else:
        rtol, atol = get_tolerances(dtype)
        torch.testing.assert_close(out, o_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (16, 16)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("qo_len,kv_len", [(128, 1024), (512, 2048), (1024, 4096)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("table_layout", ["sglang", "vllm"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
def test_batch_prefill_kv_blockscale_pytest(
    batch_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    qo_len,
    kv_len,
    causal,
    table_layout,
    logits_soft_cap,
):
    """Pytest wrapper for KV_BLOCKSCALE test.

    Note: LSE testing is not included because FP8 is inference-only (no backward pass).
    """
    if skip_test_if(
        should_skip_rocm72_issue(causal, logits_soft_cap),
        "ROCm 7.2 + gfx950 compiler issue with causal=True + logits_soft_cap=0.0",
    ):
        return

    run_batch_prefill_kv_blockscale(
        kvcache_layout="linear",
        table_layout=table_layout,
        batch_size=batch_size,
        qo_len=qo_len,
        kv_len=kv_len,
        page_size=1024,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        dtype=torch.bfloat16,
        contiguous_kv=True,
        seed=42,
    )


def run_batch_prefill_kv_blockscale(
    kvcache_layout,
    table_layout,
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
    contiguous_kv,
    seed,
    profile=False,
):
    """
    Test FP8 batch prefill with per-page KV descale (KV_BLOCKSCALE mode).
    """
    if seed is not None:
        torch.manual_seed(seed)

    quant_dtype = dtypes.fp8
    # KV_BLOCKSCALE only supports page_size=1024
    if page_size != 1024:
        if skip_test_if(
            True, f"KV_BLOCKSCALE only supports page_size=1024, got {page_size}"
        ):
            return {"status": "skipped"}

    k_vector_size = get_vector_size(quant_dtype)

    if skip_test_if(
        should_skip_rocm72_issue(causal, logits_soft_cap),
        "ROCm 7.2 + gfx950 compiler issue with causal=True + logits_soft_cap=0.0",
    ):
        return {"status": "skipped"}

    # Build sequence lengths
    qo_lens = build_qo_lens(batch_size, qo_len, randomize=batch_size > 1)
    kv_lens = build_kv_lens(batch_size, kv_len, qo_lens, randomize=batch_size > 1)
    max_qo_len = qo_lens.max().item()
    max_kv_len = kv_lens.max().item()

    # Create Q in dtype (same as pertensor FP8 test - uses uniform [0, 1])
    q = build_q_tensor_for_test(
        qo_lens,
        batch_size,
        qo_len,
        num_qo_heads,
        head_dim,
        dtype,
        None,  # q_init_min = None (not used for FP8)
        None,  # q_init_max = None (not used for FP8)
        is_input_fp8=True,  # Use FP8 path: uniform [0, 1]
    )

    # Create paged KV cache with uniform [0, 1] data (same as pertensor FP8 test)
    # Use build_paged_kv_cache with use_uniform=True and no min/max scaling
    kv_cache = build_paged_kv_cache(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
        kv_lens,
        None,  # kv_init_min = None for uniform [0, 1]
        None,  # kv_init_max = None for uniform [0, 1]
        dtype,
        use_uniform=True,
        contiguous_kv=contiguous_kv,
    )

    # Extract tensors
    kv_data_fp32 = kv_cache["kv_data_fp32"]  # FP32 for reference calculation
    kv_data = kv_cache["kv_data"]  # dtype (BF16) version
    kv_indptr = kv_cache["kv_indptr_cpu"]
    kv_indices = kv_cache["kv_indices_cpu"]
    kv_last_page_len_cpu = kv_cache["kv_last_page_len_cpu"]

    # Split K/V from paged format
    k_paged_ref, v_paged_ref = split_kv_pages(kv_data)  # BF16 for BF16 kernel

    # Quantize Q with per-tensor scale
    q_fp8, q_descale = per_tensor_quant(q, quant_dtype=quant_dtype)

    cu_seqlens_q = convert_lens_to_indptr(qo_lens).cuda()
    q_indptr_cpu = convert_lens_to_indptr(qo_lens)

    # Build FP32 reference output (same as pertensor test)
    o_ref = build_reference_output(
        q,
        q_indptr_cpu,
        kv_data_fp32,
        kv_indices,
        kv_indptr,
        kv_last_page_len_cpu,
        num_kv_heads,
        head_dim,
        dtype,
        causal,
        logits_soft_cap,
    )

    # Quantize K/V with per-page scale
    # k_paged_ref is [num_pages, page_size, num_kv_heads, head_dim]
    k_paged_fp8, k_descales = per_page_quant(k_paged_ref, page_size, quant_dtype)
    v_paged_fp8, v_descales = per_page_quant(v_paged_ref, page_size, quant_dtype)

    # Build kv_block_descale: [num_pages, num_kv_heads, 2]
    kv_block_descale = torch.stack([k_descales, v_descales], dim=-1)

    # Apply KV layout for FP8 tensors
    if kvcache_layout == "vectorized":
        k_for_vec = k_paged_fp8.view(-1, num_kv_heads, head_dim)
        v_for_vec = v_paged_fp8.view(-1, num_kv_heads, head_dim)
        k_paged, v_paged = vectorize_kv_cache(
            k_for_vec,
            v_for_vec,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size,
        )
    else:
        # Linear layout: [num_pages, page_size, num_kv_heads, head_dim]
        k_paged = k_paged_fp8
        v_paged = v_paged_fp8

    # Apply KV layout for BF16 reference tensors (for BF16 kernel run)
    k_vector_size_bf16 = get_vector_size(dtype)
    if kvcache_layout == "vectorized":
        k_for_vec_bf16 = k_paged_ref.view(-1, num_kv_heads, head_dim)
        v_for_vec_bf16 = v_paged_ref.view(-1, num_kv_heads, head_dim)
        k_cache_bf16, v_cache_bf16 = vectorize_kv_cache(
            k_for_vec_bf16,
            v_for_vec_bf16,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size_bf16,
        )
    else:
        # Linear layout: [num_pages, page_size, num_kv_heads, head_dim]
        k_cache_bf16 = k_paged_ref
        v_cache_bf16 = v_paged_ref

    # Build block table
    max_num_pages_per_seq = (max_kv_len + page_size - 1) // page_size
    block_table_cpu = torch.zeros(
        (batch_size, max_num_pages_per_seq), dtype=torch.int32
    )
    for i in range(batch_size):
        start, end = kv_indptr[i].item(), kv_indptr[i + 1].item()
        block_table_cpu[i, : (end - start)] = kv_indices[start:end]
    block_table_gpu = block_table_cpu.cuda()

    kv_last_page_len_gpu = ((kv_lens - 1) % page_size + 1).int().cuda()
    seqlen_k_gpu = kv_lens.cuda().int()

    # Run kernel with KV_BLOCKSCALE using run_ck
    profile_result = {"status": "passed"}
    run_result = run_ck(
        batch_size,
        num_kv_heads,
        q_fp8,
        k_paged,
        v_paged,
        cu_seqlens_q,
        kv_indptr.cuda(),
        kv_indices.cuda(),
        max_qo_len,
        max_kv_len,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        q_descale=q_descale,
        kv_block_descale=kv_block_descale,
        kv_last_page_lens=kv_last_page_len_gpu,
        block_table=block_table_gpu,
        seqlen_k=seqlen_k_gpu,
        profile=profile,
    )
    if profile:
        out_fp8, time_us, tflops = run_result
        profile_result = {"status": "passed", "time_us": time_us, "tflops": tflops}
    else:
        out_fp8 = run_result

    # Run BF16 reference kernel (no quantization) - no profiling for reference
    out_bf16 = run_ck(
        batch_size,
        num_kv_heads,
        q.cuda(),
        k_cache_bf16.cuda(),
        v_cache_bf16.cuda(),
        cu_seqlens_q,
        kv_indptr.cuda(),
        kv_indices.cuda(),
        max_qo_len,
        max_kv_len,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        kv_last_page_lens=kv_last_page_len_gpu,
        block_table=block_table_gpu,
        seqlen_k=seqlen_k_gpu,
        profile=False,
    )

    # Sanity checks
    assert out_fp8.abs().max().item() > 1e-6, "FP8 kernel output is all zeros!"
    assert out_bf16.abs().max().item() > 1e-6, "BF16 kernel output is all zeros!"
    assert o_ref.abs().max().item() > 1e-6, "FP32 reference output is all zeros!"

    # Compare FP8 kernel vs FP32 reference (same as pertensor test)
    verify_fp8_output(out_fp8, o_ref)

    # Compare BF16 kernel vs FP32 reference (same as pertensor test)
    rtol, atol = get_tolerances(dtype, is_fp8=False)
    torch.testing.assert_close(out_bf16, o_ref, rtol=rtol, atol=atol)

    return profile_result


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
    choices=[1, 16, 1024],
    default=[1, 16, 1024],
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
    type=str,
    const=None,
    choices=["fp16", "bf16", "fp8"],
    default=["bf16", "fp8"],
    nargs="*",
    help="""input dtype.
    e.g.: --input_dtype bf16 fp8""",
)
parser.add_argument(
    "--quant_method",
    type=str,
    const=None,
    choices=["none", "pertensor", "kv_blockscale"],
    default=["none", "pertensor", "kv_blockscale"],
    nargs="*",
    help="""quantization method.
    none: no quantization (for fp16/bf16)
    pertensor: per-tensor Q/K/V descale (for fp8)
    kv_blockscale: per-tensor Q, per-page K/V descale (for fp8)
    e.g.: --quant_method pertensor kv_blockscale""",
)
parser.add_argument(
    "--head_dim",
    type=int,
    const=None,
    choices=[128, 256],
    default=[128, 256],
    nargs="*",
    help="""head dimension.
    e.g.: --head_dim 128 256""",
)
parser.add_argument(
    "--profile",
    action="store_true",
    help="Enable profiling mode",
)
parser.add_argument(
    "--return_lse",
    type=dtypes.str2bool,
    nargs="*",
    default=[True, False],
    help="""Enable LSE (log-sum-exp) output and comparison with reference.
    e.g.: --return_lse true""",
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
        quant_method,
        contiguous_kv,
        return_lse,
        head_dim,
    ) in itertools.product(
        args.pagesize,
        args.causal,
        args.logits_soft_cap,
        args.dtype,
        args.lookup_table,
        args.kv_layout,
        args.input_dtype,
        args.quant_method,
        [True, False],  # contiguous_kv
        args.return_lse,
        args.head_dim,
    ):
        # Validate quant_method and input_dtype combinations:
        # - fp16/bf16 must use quant_method="none"
        # - fp8 must use quant_method="pertensor" or "kv_blockscale"
        if input_dtype != "fp8" and quant_method != "none":
            continue
        if input_dtype == "fp8" and quant_method == "none":
            continue

        # Convert string input_dtype to torch dtype
        input_dtype_torch = dtypes.str2Dtype(input_dtype)

        # Choose test function based on input_dtype and quant_method
        if input_dtype == "fp8" and quant_method == "kv_blockscale":
            # KV_BLOCKSCALE: per-page K/V descale
            result = run_batch_prefill_kv_blockscale(
                kvcache_layout=kv_layout,
                table_layout=lookup_table,
                batch_size=1,
                qo_len=args.seqlen,
                kv_len=args.seqlen,
                page_size=page_size,
                num_qo_heads=args.headq,
                num_kv_heads=args.headk,
                head_dim=head_dim,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
                dtype=dtype,
                contiguous_kv=contiguous_kv,
                seed=19378,
                profile=args.profile,
            )
        else:
            result = test_batch_prefill(
                kvcache_layout=kv_layout,
                table_layout=lookup_table,
                input_dtype=input_dtype_torch,
                batch_size=1,
                qo_len=args.seqlen,
                kv_len=args.seqlen,
                page_size=page_size,
                num_qo_heads=args.headq,
                num_kv_heads=args.headk,
                head_dim=head_dim,
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
                return_lse=return_lse,
            )

        # Build result row
        time_us = result.get("time_us") if result else None
        tflops = result.get("tflops") if result else None
        row = {
            "seqlen": args.seqlen,
            "page_sz": page_size,
            "h_q": args.headq,
            "h_kv": args.headk,
            "hdim": head_dim,
            "input_dtype": input_dtype,
            "quant_method": quant_method if input_dtype == "fp8" else "-",
            "kv_layout": kv_layout,
            "table": lookup_table,
            "causal": causal,
            "soft_cap": logits_soft_cap,
            "contig": contiguous_kv,
            "lse": "-" if input_dtype == "fp8" else return_lse,
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


# =============================================================================
# StreamLLM Sink Token Tests
# =============================================================================


def ref_masked_attention_with_sink(
    query,
    key,
    value,
    window_left,
    sink_size,
    sink_ptr_value,
):
    """
    Reference attention with StreamLLM sink semantics.

    Args:
        query:          [seqlen_q, num_heads, head_dim]
        key:            [seqlen_k, num_heads, head_dim]
        value:          [seqlen_k, num_heads, head_dim]
        window_left:    left window size (-1 = infinite)
        sink_size:      number of KV tokens at start always attended
        sink_ptr_value: per-head float tensor [num_heads] or None.
                        When not None, a virtual sink token with this scaled
                        logit is appended to the attention matrix (it steals
                        probability mass but has no V contribution).

    Valid KV range for query at absolute position abs_q = seqlen_k - seqlen_q + i_q:
        k < sink_size   (sink region, always valid)
        OR
        abs_q - window_left <= k <= abs_q   (window region, window_left=-1 means k >= 0)
    """
    head_dim = query.shape[2]
    seqlen_q = query.shape[0]
    seqlen_k = key.shape[0]
    num_heads = query.shape[1]
    scale = 1.0 / math.sqrt(head_dim)

    # [num_heads, seqlen_q, seqlen_k]
    attn = scale * torch.einsum("qhd,khd->hqk", query.float(), key.float())

    # Build mask vectorized to avoid per-element GPU synchronization
    # i_q: [seqlen_q, 1], i_k: [1, seqlen_k]
    i_q = torch.arange(seqlen_q, device=query.device).unsqueeze(1)  # [sq, 1]
    i_k = torch.arange(seqlen_k, device=query.device).unsqueeze(0)  # [1, sk]
    abs_q = seqlen_k - seqlen_q + i_q  # [sq, 1]
    k_end = abs_q  # causal boundary
    if window_left < 0:
        k_start_window = torch.zeros_like(abs_q)
    else:
        k_start_window = torch.clamp(abs_q - window_left, min=sink_size)
    is_sink = i_k < sink_size  # [1, sk]
    is_window = (i_k >= k_start_window) & (i_k <= k_end)  # [sq, sk]
    valid = is_sink | is_window  # [sq, sk]
    # attn: [H, sq, sk] -- broadcast mask over heads
    attn.masked_fill_(~valid.unsqueeze(0), float("-inf"))

    if sink_ptr_value is not None:
        # Append virtual sink token column: logit = sink_ptr_value[h] (scaled space)
        # Shape: [num_heads, seqlen_q, 1]
        virt = sink_ptr_value.float().view(num_heads, 1, 1).expand(-1, seqlen_q, 1)
        attn_ext = torch.cat([attn, virt], dim=-1)  # [H, sq, sk+1]
        P_ext = torch.softmax(attn_ext, dim=-1)
        P = P_ext[:, :, :seqlen_k]  # drop virtual column (V contribution = 0)
    else:
        P = torch.softmax(attn, dim=-1)

    out = torch.einsum("hqk,khd->qhd", P, value.float())
    return out.to(query.dtype)


def run_batch_prefill_sink(
    batch_size,
    qo_len,
    kv_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    window_left,
    sink_size,
    sink_ptr_value,
    dtype,
    seed,
):
    """
    Run batch_prefill with sink tokens and compare against torch reference.

    sink_ptr_value: float or None. When float, a sink_ptr tensor of shape
                    [num_qo_heads] filled with this value is passed to the kernel.
    """
    if seed is not None:
        torch.manual_seed(seed)

    k_vector_size = get_vector_size(dtype)

    # kv_len must be large enough to create a real gap between sink and window
    if skip_test_if(
        kv_len <= sink_size + window_left + 1,
        f"kv_len={kv_len} too small for gap (need >{sink_size + window_left + 1})",
    ):
        return {"status": "skipped"}

    qo_lens = build_qo_lens(batch_size, qo_len, randomize=batch_size > 1)
    kv_lens = build_kv_lens(batch_size, kv_len, qo_lens, randomize=batch_size > 1)
    max_qo_len = qo_lens.max().item()
    max_kv_len = kv_lens.max().item()
    q_indptr_cpu = convert_lens_to_indptr(qo_lens)

    total_q = q_indptr_cpu[-1]
    q = build_q_tensor(total_q, num_qo_heads, head_dim, dtype, -5, 5)

    kv_cache = build_paged_kv_cache(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
        kv_lens,
        -5,
        5,
        dtype,
        contiguous_kv=True,
    )
    kv_data_fp32 = kv_cache["kv_data_fp32"]
    kv_indices_cpu = kv_cache["kv_indices_cpu"]
    kv_indptr_cpu_cache = kv_cache["kv_indptr_cpu"]
    kv_last_page_len_cpu = kv_cache["kv_last_page_len_cpu"]

    k_cache_ref, v_cache_ref = extract_kv_caches(kv_cache, contiguous_kv=True)
    k_cache, v_cache = apply_kv_layout(
        k_cache_ref,
        v_cache_ref,
        num_kv_heads,
        head_dim,
        page_size,
        k_vector_size,
        "vectorized",
    )

    # Build sink_ptr tensor
    sink_ptr = None
    if sink_ptr_value is not None:
        sink_ptr = torch.full(
            (num_qo_heads,), sink_ptr_value, dtype=torch.float32, device="cuda"
        )

    # -- Torch reference ------------------------------------------------------
    # kv_data_fp32: [total_pages, 2, page_size, num_kv_heads, head_dim]
    #   dim 1: 0=K, 1=V
    o_ref_list = []
    for i in range(batch_size):
        used_idx = kv_indices_cpu[kv_indptr_cpu_cache[i] : kv_indptr_cpu_cache[i + 1]]
        last_len = kv_last_page_len_cpu[i].item()

        # Full pages: [num_full_pages, page_size, num_kv_heads, head_dim]
        # Last page: [:last_len, num_kv_heads, head_dim]
        ki = torch.cat(
            [
                kv_data_fp32[used_idx[:-1], 0].reshape(-1, num_kv_heads, head_dim),
                kv_data_fp32[used_idx[-1], 0, :last_len].reshape(
                    -1, num_kv_heads, head_dim
                ),
            ],
            dim=0,
        ).to(dtype)
        vi = torch.cat(
            [
                kv_data_fp32[used_idx[:-1], 1].reshape(-1, num_kv_heads, head_dim),
                kv_data_fp32[used_idx[-1], 1, :last_len].reshape(
                    -1, num_kv_heads, head_dim
                ),
            ],
            dim=0,
        ).to(dtype)

        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

        if num_qo_heads != num_kv_heads:
            ratio = num_qo_heads // num_kv_heads
            ki = ki.repeat_interleave(ratio, dim=1)
            vi = vi.repeat_interleave(ratio, dim=1)

        o_ref_list.append(
            ref_masked_attention_with_sink(qi, ki, vi, window_left, sink_size, sink_ptr)
        )
    o_ref = torch.cat(o_ref_list, dim=0)

    # -- CK kernel -------------------------------------------------------------
    kv_indptr_gpu = kv_indptr_cpu_cache.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)
    kv_last_page_lens = kv_last_page_len_cpu.to(0)
    cu_seqlens_q = q_indptr_cpu.to(0)

    out = aiter.mha_batch_prefill_func(
        q,
        k_cache,
        v_cache,
        cu_seqlens_q,
        kv_indptr_gpu,
        kv_indices_gpu,
        max_seqlen_q=max_qo_len,
        max_seqlen_k=max_kv_len,
        causal=True,
        window_size=(window_left, -1),
        sink_size=sink_size,
        sink_ptr=sink_ptr,
        kv_last_page_lens=kv_last_page_lens,
        return_lse=False,
    )

    # -- Compare ---------------------------------------------------------------
    rtol, atol = get_tolerances(dtype)
    assert_output_matches_reference(out, q_indptr_cpu, o_ref, rtol, atol)
    return {"status": "passed"}


# ---------------------------------------------------------------------------
# AICK-1171 reproducer: load_physical_pages OOB read on V prefetch lookahead
#
# Ported from 3rdparty/composable_kernel/test_rocm_mha_attn.py --case crash1_r8
# (the bisect family that isolated the bug to total cache size, i.e. the page
# table is read past the valid region).
#
# Crash shape from Tencent Hunyuan / MI-308X:
#   prefill (q=2042, kv=2042), 10 q-heads, 1 kv-head, head_dim=128,
#   page_size=16, bf16, causal=True
#
# Trigger conditions the standard `build_paged_kv_cache` masks:
#   1. `kv_indices_cpu` here is built at EXACT length (no 128-element zero
#      padding), so an OOB `page_idx[N]` read no longer falls into a benign
#      pad region of value 0.
#   2. The cache tensor has unused trailing pages (n_used < total_blocks)
#      that we POISON with sentinel data -- if the kernel reads past the
#      page table and into one of those pages, the output diverges from
#      the reference and the assert fires.
# ---------------------------------------------------------------------------
def _build_aick1171_paged_kv_cache(
    kv_len, page_size, num_kv_heads, head_dim, dtype, total_blocks, seed
):
    """Build a paged KV cache shaped exactly like the AICK-1171 reproducer.

    Mirrors `build_paged_kv_cache`'s `make_scaled_rand` distribution (so the
    tolerance picture matches the rest of the suite), but with two trigger
    knobs that the standard helper masks:
      - `kv_indices_cpu` is exactly `n_used` entries (no 128-element zero pad),
        so an OOB `page_idx[N]` no longer falls into a benign 0-page.
      - Cache slots `[n_used .. total_blocks-1]` are filled with a sentinel
        value large enough to dominate softmax -- any OOB read of those pages
        causes a numerically detectable mismatch.
    """
    n_used = (kv_len + page_size - 1) // page_size
    assert total_blocks >= n_used

    # Valid region: same distribution as build_paged_kv_cache (-5, 5).
    valid_shape = [n_used, 2, page_size, num_kv_heads, head_dim]
    valid = make_scaled_rand(-5, 5, *valid_shape, dtype=torch.float32).to(0)

    kv_shape = [total_blocks, 2, page_size, num_kv_heads, head_dim]
    kv_data_fp32 = torch.empty(*kv_shape, device="cuda", dtype=torch.float32)
    kv_data_fp32[:n_used] = valid
    kv_data_fp32[n_used:] = 50.0  # sentinel -- any read of these dominates softmax

    kv_data = kv_data_fp32.to(dtype)

    # Logical pages 0..n_used-1 map to a permutation of physical slots inside
    # the valid region. kv_indices_cpu is exactly n_used long: the bug, if
    # present, dereferences whatever lies past the tensor's buffer end.
    page_perm = torch.randperm(
        n_used, generator=torch.Generator().manual_seed(seed)
    ).int()
    kv_indices_cpu = page_perm.contiguous()

    kv_indptr_cpu = torch.tensor([0, n_used], dtype=torch.int32)
    kv_last_page_len_cpu = torch.tensor(
        [(kv_len - 1) % page_size + 1], dtype=torch.int32
    )
    return {
        "kv_data_fp32": kv_data_fp32,
        "kv_data": kv_data,
        "kv_indptr_cpu": kv_indptr_cpu,
        "kv_indices_cpu": kv_indices_cpu,
        "kv_last_page_len_cpu": kv_last_page_len_cpu,
        "max_num_pages_per_seq": n_used,
        "total_num_pages": total_blocks,
    }


@pytest.mark.parametrize(
    "total_blocks",
    # Mirrors crash1_r8_blocks_{160,164,168,176,208,256}: 128 used + padding.
    # 160 was the smallest size that consistently faulted on MI-308X; 168
    # was the bisect boundary; >=256 silently passed under the bug because
    # OOB reads still landed in valid (zero) memory.
    [160, 164, 168, 176, 208, 256],
)
def test_batch_prefill_aick1171_oob_page_table_read(total_blocks):
    """AICK-1171: page-table OOB read in load_physical_pages V prefetch.

    With the fix in place (clamp_token_idx / max_page_table_idx), output must
    match the torch reference regardless of `total_blocks`. Without the fix,
    runs with `total_blocks` ? {160..167} fault on gfx942/MI-308X, and the
    larger sizes silently corrupt output by reading the sentinel pages.
    """
    torch.manual_seed(42)

    # Exact crash1_r8 shape
    qo_len = kv_len = 2042
    num_qo_heads, num_kv_heads = 10, 1
    head_dim = 128
    page_size = 16
    dtype = torch.bfloat16
    causal = True

    qo_lens = torch.tensor([qo_len], dtype=torch.int32)
    q_indptr_cpu = convert_lens_to_indptr(qo_lens)
    total_q = q_indptr_cpu[-1].item()
    q = build_q_tensor(total_q, num_qo_heads, head_dim, dtype, -10, 10)

    kv_cache = _build_aick1171_paged_kv_cache(
        kv_len=kv_len,
        page_size=page_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype,
        total_blocks=total_blocks,
        seed=42,
    )

    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_cache["kv_indptr_cpu"].to(0)
    kv_indices_gpu = kv_cache["kv_indices_cpu"].to(0)
    kv_last_page_len_gpu = kv_cache["kv_last_page_len_cpu"].to(0)

    k_cache_ref, v_cache_ref = extract_kv_caches(kv_cache, contiguous_kv=True)
    k_cache, v_cache = apply_kv_layout(
        k_cache_ref,
        v_cache_ref,
        num_kv_heads,
        head_dim,
        page_size,
        get_vector_size(dtype),
        "vectorized",
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
        logits_soft_cap=0.0,
    )

    out = run_ck(
        batch_size=1,
        num_kv_heads=num_kv_heads,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        cu_seqlens_q=q_indptr_gpu,
        kv_indptr=kv_indptr_gpu,
        kv_page_indices=kv_indices_gpu,
        max_seqlen_q=qo_len,
        max_seqlen_k=kv_len,
        causal=causal,
        kv_last_page_lens=kv_last_page_len_gpu,
    )

    # Use the project-standard bf16 tolerance (matches main test_batch_prefill).
    rtol, atol = get_tolerances(dtype)
    assert_output_matches_reference(out, q_indptr_cpu, o_ref, rtol, atol)


# ===========================================================================
# HIP Virtual Memory Management (VMM) bindings -- used by the AICK-1171
# guard-page test below.
#
# Allocates a GPU buffer that ends exactly at an unmapped virtual page boundary
# so any read past the buffer end deterministically triggers a GPU memory access
# fault. PyTorch's CUDACachingAllocator pads OOB reads with mapped pool memory
# and silently masks the fault -- we need raw HIP VMM API to control the page
# layout. This block is inlined (rather than a separate helper module) so the
# test file is self-contained for committing.
#
# ROCm 7.x VMM API surface used:
#   hipMemGetAllocationGranularity   - query min page size for VMM ops
#   hipMemAddressReserve / Free      - reserve / release VA range
#   hipMemCreate / Release           - allocate / release physical handle
#   hipMemMap / Unmap                - bind physical to VA / unbind
#   hipMemSetAccess                  - set RW permissions on mapped range
# ===========================================================================

_HIP_LIB = "libamdhip64.so"

# Enum constants (mirror hip/hip_runtime_api.h)
_HIP_MEM_LOCATION_TYPE_DEVICE = 1
_HIP_MEM_ALLOCATION_TYPE_PINNED = 1
_HIP_MEM_HANDLE_TYPE_NONE = 0
_HIP_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
_HIP_MEM_ALLOC_GRANULARITY_MINIMUM = 0
_HIP_MEMCPY_HOST_TO_DEVICE = 1
_HIP_MEMCPY_DEVICE_TO_HOST = 2


# Structures (must mirror hip/hip_runtime_api.h byte layout exactly)
class _HipMemLocation(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("id", ctypes.c_int),
    ]


class _HipMemAllocFlags(ctypes.Structure):
    _fields_ = [
        ("compressionType", ctypes.c_ubyte),
        ("gpuDirectRDMACapable", ctypes.c_ubyte),
        ("usage", ctypes.c_ushort),
        ("reserved", ctypes.c_ubyte * 4),
    ]


class _HipMemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleType", ctypes.c_int),
        ("location", _HipMemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", _HipMemAllocFlags),
    ]


class _HipMemAccessDesc(ctypes.Structure):
    _fields_ = [
        ("location", _HipMemLocation),
        ("flags", ctypes.c_int),
    ]


# Catch silent ROCm header drift at import time. If a future ROCm release adds
# a field to any of these structs without our binding being updated, ctypes
# would silently write garbage into the new field's bytes. These assertions
# fail loudly at module load instead. Sizes verified against ROCm 7.2.0.
assert ctypes.sizeof(_HipMemLocation) == 8, (
    f"_HipMemLocation size mismatch: {ctypes.sizeof(_HipMemLocation)} != 8 "
    f"(ROCm header changed?)"
)
assert ctypes.sizeof(_HipMemAllocFlags) == 8, (
    f"_HipMemAllocFlags size mismatch: {ctypes.sizeof(_HipMemAllocFlags)} != 8 "
    f"(ROCm header changed?)"
)
assert ctypes.sizeof(_HipMemAllocationProp) == 32, (
    f"_HipMemAllocationProp size mismatch: "
    f"{ctypes.sizeof(_HipMemAllocationProp)} != 32 (ROCm header changed?)"
)
assert ctypes.sizeof(_HipMemAccessDesc) == 12, (
    f"_HipMemAccessDesc size mismatch: {ctypes.sizeof(_HipMemAccessDesc)} != 12 "
    f"(ROCm header changed?)"
)


# Library binding (lazy -- first call to make_guarded_int32_tensor populates).
# Loading libamdhip64.so at module top would break test collection on non-ROCm
# CI machines that import this file for unrelated tests.
_hip = None


def _hip_lib():
    """Lazy CDLL loader + argtype/restype setup. Idempotent; subsequent calls
    return the cached handle."""
    global _hip
    if _hip is not None:
        return _hip
    lib = ctypes.CDLL(_HIP_LIB)

    lib.hipMemGetAllocationGranularity.argtypes = [
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(_HipMemAllocationProp),
        ctypes.c_int,
    ]
    lib.hipMemGetAllocationGranularity.restype = ctypes.c_int

    lib.hipMemAddressReserve.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_ulonglong,
    ]
    lib.hipMemAddressReserve.restype = ctypes.c_int

    lib.hipMemAddressFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.hipMemAddressFree.restype = ctypes.c_int

    lib.hipMemCreate.argtypes = [
        ctypes.POINTER(ctypes.c_ulonglong),
        ctypes.c_size_t,
        ctypes.POINTER(_HipMemAllocationProp),
        ctypes.c_ulonglong,
    ]
    lib.hipMemCreate.restype = ctypes.c_int

    lib.hipMemRelease.argtypes = [ctypes.c_ulonglong]
    lib.hipMemRelease.restype = ctypes.c_int

    lib.hipMemMap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_ulonglong,
        ctypes.c_ulonglong,
    ]
    lib.hipMemMap.restype = ctypes.c_int

    lib.hipMemUnmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.hipMemUnmap.restype = ctypes.c_int

    lib.hipMemSetAccess.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(_HipMemAccessDesc),
        ctypes.c_size_t,
    ]
    lib.hipMemSetAccess.restype = ctypes.c_int

    lib.hipMemcpy.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.hipMemcpy.restype = ctypes.c_int

    lib.hipGetErrorString.argtypes = [ctypes.c_int]
    lib.hipGetErrorString.restype = ctypes.c_char_p

    _hip = lib
    return _hip


def _hip_check(err: int, where: str):
    if err != 0:
        msg = _hip_lib().hipGetErrorString(err)
        msg_str = msg.decode() if msg else f"unknown_err_{err}"
        raise RuntimeError(f"HIP error in {where}: {err} ({msg_str})")


def _build_alloc_prop(device: int) -> _HipMemAllocationProp:
    prop = _HipMemAllocationProp()
    prop.type = _HIP_MEM_ALLOCATION_TYPE_PINNED
    prop.requestedHandleType = _HIP_MEM_HANDLE_TYPE_NONE
    prop.location.type = _HIP_MEM_LOCATION_TYPE_DEVICE
    prop.location.id = device
    prop.win32HandleMetaData = None
    return prop


def _alloc_int32_with_guard_page(num_indices: int, device: int = 0):
    """Allocate `num_indices * 4` bytes of int32 GPU memory with a guard page.

    Memory layout:
        [ mapped page (size=G) | UNMAPPED page (size=G) ]
        |<-- (G - num*4) -->|<--- num*4 bytes --->|
                             ^                    ^
                          raw_ptr              raw_ptr + num*4 = page boundary

    Any GPU access to addresses >= raw_ptr + num*4 hits the unmapped page and
    causes hardware MEMORY_VIOLATION.

    Returns: (raw_ptr, granularity, cleanup_handle)
    """
    if num_indices <= 0:
        raise ValueError(f"num_indices must be positive, got {num_indices}")

    hip = _hip_lib()
    prop = _build_alloc_prop(device)
    g = ctypes.c_size_t(0)
    _hip_check(
        hip.hipMemGetAllocationGranularity(
            ctypes.byref(g),
            ctypes.byref(prop),
            _HIP_MEM_ALLOC_GRANULARITY_MINIMUM,
        ),
        "hipMemGetAllocationGranularity",
    )
    G = g.value

    buf_size = num_indices * 4
    if buf_size > G:
        raise ValueError(
            f"Buffer size {buf_size} exceeds VMM page granularity {G}; "
            f"reduce num_indices to <= {G // 4}"
        )

    # 1. Reserve 2*G of contiguous VA. First half mapped, second half = guard.
    va_ptr = ctypes.c_void_p(0)
    _hip_check(
        hip.hipMemAddressReserve(
            ctypes.byref(va_ptr),
            2 * G,
            G,
            ctypes.c_void_p(0),
            ctypes.c_ulonglong(0),
        ),
        "hipMemAddressReserve",
    )

    # 2. Allocate physical handle of size G.
    phys_handle = ctypes.c_ulonglong(0)
    try:
        _hip_check(
            hip.hipMemCreate(
                ctypes.byref(phys_handle),
                G,
                ctypes.byref(prop),
                ctypes.c_ulonglong(0),
            ),
            "hipMemCreate",
        )
    except Exception:
        hip.hipMemAddressFree(va_ptr, 2 * G)
        raise

    # 3. Map physical handle to first G of VA. Second G remains UNMAPPED.
    try:
        _hip_check(
            hip.hipMemMap(va_ptr, G, 0, phys_handle, ctypes.c_ulonglong(0)),
            "hipMemMap",
        )
    except Exception:
        hip.hipMemRelease(phys_handle)
        hip.hipMemAddressFree(va_ptr, 2 * G)
        raise

    # 4. Grant device RW access on the mapped page.
    desc = _HipMemAccessDesc()
    desc.location.type = _HIP_MEM_LOCATION_TYPE_DEVICE
    desc.location.id = device
    desc.flags = _HIP_MEM_ACCESS_FLAGS_PROT_READWRITE
    try:
        _hip_check(
            hip.hipMemSetAccess(va_ptr, G, ctypes.byref(desc), 1),
            "hipMemSetAccess",
        )
    except Exception:
        hip.hipMemUnmap(va_ptr, G)
        hip.hipMemRelease(phys_handle)
        hip.hipMemAddressFree(va_ptr, 2 * G)
        raise

    # 5. Place buffer at the END of the mapped page so it touches the guard.
    raw_ptr = va_ptr.value + (G - buf_size)
    return raw_ptr, G, (va_ptr, phys_handle)


def _free_guard_page(handle, granularity: int):
    """Release VA + physical handle. Order matters: unmap, release, free VA."""
    hip = _hip_lib()
    va_ptr, phys_handle = handle
    hip.hipMemUnmap(va_ptr, granularity)
    hip.hipMemRelease(phys_handle)
    hip.hipMemAddressFree(va_ptr, 2 * granularity)


def make_guarded_int32_tensor(values, device: int = 0):
    """Wrap a guarded int32 buffer as a torch.Tensor (zero-copy via CAI).

    The tensor's data_ptr() points to memory with an unmapped page right
    after the buffer end. Any GPU read past the buffer faults.

    Used by test_batch_prefill_aick1171_hard_fault_via_guard_page below.
    """
    import numpy as np

    if isinstance(values, torch.Tensor):
        values_np = values.detach().cpu().to(torch.int32).contiguous().numpy()
    else:
        values_np = np.ascontiguousarray(np.asarray(values, dtype=np.int32))
    n = values_np.size

    raw_ptr, G, handle = _alloc_int32_with_guard_page(n, device=device)
    _hip_check(
        _hip_lib().hipMemcpy(
            ctypes.c_void_p(raw_ptr),
            ctypes.c_void_p(values_np.ctypes.data),
            n * 4,
            _HIP_MEMCPY_HOST_TO_DEVICE,
        ),
        "hipMemcpy H2D init",
    )

    cai_holder = type("_CAIHolder", (), {})()
    cai_holder.__cuda_array_interface__ = {
        "shape": (n,),
        "typestr": "<i4",
        "data": (raw_ptr, False),
        "version": 3,
        "strides": None,
        "stream": torch.cuda.current_stream(device).cuda_stream,
    }
    tensor = torch.as_tensor(cai_holder, device=f"cuda:{device}")

    # weakref.finalize survives interpreter-shutdown ordering (a naive __del__
    # can fail because module globals like _hip get None'd before tensor.__del__
    # runs, leaving VMM mappings leaked).
    tensor._guard_finalizer = weakref.finalize(
        tensor,
        _free_guard_page,
        handle,
        G,
    )
    return tensor


# ---------------------------------------------------------------------------
# AICK-1171 hard-fault regression test (companion to the sentinel-padding test
# above).
#
# Mechanism difference from test_batch_prefill_aick1171_oob_page_table_read:
#   Sentinel test  : poisons KV cache with sentinel values; depends on the
#                    OOB-read page index being CONSUMED by V load to detect
#                    numerical corruption. Defends against future
#                    "prefetch becomes consumed" regressions.
#   Guard-page test: places kv_page_indices buffer against an unmapped HIP
#                    VMM page. Any GPU read past the buffer end faults
#                    deterministically (HSA MEMORY_VIOLATION), regardless of
#                    whether the loaded value flows downstream. Catches the
#                    actual AICK-1171 manifestation: GPU coredump from
#                    speculative prefetch reading page_idx[128].
#
# Subprocess isolation is REQUIRED: GPU memory fault sends SIGABRT/SIGPIPE to
# the entire process; no try/except in pytest can catch it.
# ---------------------------------------------------------------------------
def _aick1171_run_in_subprocess(child_code: str, timeout: int = 180):
    """Run AICK-1171-shaped test body in a fresh Python subprocess.
    HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 prevents multi-GB GPU coredump files
    when the host has the coredump tool installed (verified env var name on
    ROCm 7.2.0 via `strings libhsa-runtime64.so | grep coredump`)."""
    import subprocess as _sub
    import sys as _sys

    env = dict(os.environ)
    env["HSA_DISABLE_COREDUMP_ON_EXCEPTION"] = "1"
    return _sub.run(
        [_sys.executable, "-c", child_code],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def _aick1171_fault_signature(rc: int, stderr: str) -> bool:
    """Detect HSA GPU memory-fault death.

    Empirically on ROCm 7.2.0 / MI300X the signal is SIGABRT (-6); on
    ROCm 7.x with different coredump-tool config it can be SIGPIPE (-13).
    Match on (process killed by ANY signal) AND (HSA fault keyword in
    stderr) -- both conditions must hold to avoid false positives from
    unrelated SIGPIPE / SIGABRT.
    """
    killed_by_signal = (rc < 0) or (128 <= rc < 256)
    if not killed_by_signal:
        return False
    msg = stderr.lower()
    return "memory access fault" in msg or "hsa_status_error_memory_fault" in msg


# The OOB read at page_idx[128] overruns the kv_page_indices buffer (not
# kv_data), so the fault trigger is independent of total_blocks. We still
# parametrize two values [160, 192] as a defensive coverage hedge: if a
# future kernel variant unexpectedly ties OOB-read consumption to a tile
# shape that depends on total_blocks, the second config catches it.
@pytest.mark.parametrize("total_blocks", [160, 192])
def test_batch_prefill_aick1171_hard_fault_via_guard_page(total_blocks):
    """AICK-1171: V prefetch reads page_idx[128] past valid range, GPU faults.

    Pre-fix:  child subprocess dies via signal (SIGABRT/SIGPIPE), stderr
              contains 'Memory access fault by GPU node-N on address 0x...'.
    Post-fix: clamp_token_idx / max_page_table_idx prevents the OOB load,
              kernel completes, output matches reference, child exits 0.

    Detection requires subprocess isolation (see module-level comment).
    """
    import textwrap as _textwrap

    aiter_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    child_code = _textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {aiter_root!r})
        import torch
        from op_tests.test_batch_prefill import (
            _build_aick1171_paged_kv_cache,
            build_q_tensor, convert_lens_to_indptr,
            extract_kv_caches, apply_kv_layout, get_vector_size,
            build_reference_output, run_ck,
            get_tolerances, assert_output_matches_reference,
        )
        from op_tests.test_batch_prefill import make_guarded_int32_tensor

        torch.manual_seed(42)

        # Exact crash1_r8 shape from the AICK-1171 reproducer
        qo_len = kv_len = 2042
        num_qo_heads, num_kv_heads = 10, 1
        head_dim = 128
        page_size = 16
        dtype = torch.bfloat16
        causal = True
        total_blocks = {total_blocks}

        qo_lens = torch.tensor([qo_len], dtype=torch.int32)
        q_indptr_cpu = convert_lens_to_indptr(qo_lens)
        total_q = q_indptr_cpu[-1].item()
        q = build_q_tensor(total_q, num_qo_heads, head_dim, dtype, -10, 10)

        kv_cache = _build_aick1171_paged_kv_cache(
            kv_len=kv_len, page_size=page_size, num_kv_heads=num_kv_heads,
            head_dim=head_dim, dtype=dtype,
            total_blocks=total_blocks, seed=42,
        )

        # KEY DIFFERENCE vs sentinel test: kv_page_indices is allocated with a
        # HIP VMM guard page right after the buffer. Any OOB read deterministically
        # triggers MEMORY_VIOLATION instead of relying on allocator-pool garbage
        # to leak into V loads.
        kv_indices_gpu = make_guarded_int32_tensor(
            kv_cache["kv_indices_cpu"], device=0,
        )

        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_cache["kv_indptr_cpu"].to(0)
        kv_last_page_len_gpu = kv_cache["kv_last_page_len_cpu"].to(0)

        k_cache_ref, v_cache_ref = extract_kv_caches(kv_cache, contiguous_kv=True)
        k_cache, v_cache = apply_kv_layout(
            k_cache_ref, v_cache_ref,
            num_kv_heads, head_dim, page_size,
            get_vector_size(dtype), 'vectorized',
        )

        o_ref = build_reference_output(
            q, q_indptr_cpu, kv_cache['kv_data_fp32'],
            kv_cache['kv_indices_cpu'], kv_cache['kv_indptr_cpu'],
            kv_cache['kv_last_page_len_cpu'],
            num_kv_heads, head_dim, dtype, causal,
            logits_soft_cap=0.0,
        )

        out = run_ck(
            batch_size=1,
            num_kv_heads=num_kv_heads,
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            cu_seqlens_q=q_indptr_gpu,
            kv_indptr=kv_indptr_gpu,
            kv_page_indices=kv_indices_gpu,
            max_seqlen_q=qo_len,
            max_seqlen_k=kv_len,
            causal=causal,
            kv_last_page_lens=kv_last_page_len_gpu,
        )
        torch.cuda.synchronize()  # surface async fault if any

        # Post-fix path only -- verify numerical correctness
        rtol, atol = get_tolerances(dtype)
        assert_output_matches_reference(out, q_indptr_cpu, o_ref, rtol, atol)
        print('AICK1171_GUARD_PAGE_OK', flush=True)
    """)

    result = _aick1171_run_in_subprocess(child_code)

    if _aick1171_fault_signature(result.returncode, result.stderr):
        pytest.fail(
            "AICK-1171 hard fault detected -- V prefetch read past "
            "kv_page_indices buffer end and hit guard page.\n"
            f"  rc={result.returncode}\n"
            f"  stderr (last 1KB):\n{result.stderr[-1024:]}\n"
            "Fix in load_physical_pages (clamp page_id to max_page_table_idx) "
            "is missing or has regressed."
        )

    if result.returncode != 0:
        pytest.fail(
            f"Child subprocess failed with non-fault error (rc={result.returncode}):\n"
            f"  stdout: {result.stdout[-500:]}\n"
            f"  stderr: {result.stderr[-1024:]}"
        )

    assert "AICK1171_GUARD_PAGE_OK" in result.stdout, (
        f"Subprocess didn't reach completion marker:\n"
        f"  stdout: {result.stdout!r}\n  stderr: {result.stderr!r}"
    )


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "sink_ptr_value",
    [None, 0.0, 2.0],
    ids=["ptr=None", "ptr=0.0", "ptr=2.0"],
)
@pytest.mark.parametrize("sink_size", [4, 16])
@pytest.mark.parametrize(
    "window_left,kv_len",
    [(128, 512), (1024, 2048)],
    ids=["win=128/kv=512", "win=1024/kv=2048"],
)
@pytest.mark.parametrize("qo_len", [32, 128])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(8, 1), (4, 4)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_batch_prefill_sink(
    batch_size,
    page_size,
    head_dim,
    num_qo_heads,
    num_kv_heads,
    qo_len,
    window_left,
    kv_len,
    sink_size,
    sink_ptr_value,
    dtype,
    seed,
):
    """
    Test batch_prefill with StreamLLM sink token support.

    Validates:
    - sink_size: first sink_size KV positions always attended (never window-masked)
    - sink_ptr: virtual sink token with fixed logit participates in softmax
    - window_left + sink_size creates a real gap; gap tokens are correctly masked
    """
    run_batch_prefill_sink(
        batch_size=batch_size,
        qo_len=qo_len,
        kv_len=kv_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        window_left=window_left,
        sink_size=sink_size,
        sink_ptr_value=sink_ptr_value,
        dtype=dtype,
        seed=seed,
    )


# CI runs `python3 test_batch_prefill.py` (no pytest), so the __main__ block
# above only executes the non-sink scenarios. Add a small representative sweep
# of the StreamLLM sink scenarios here so they actually exercise in CI.
if __name__ == "__main__":
    sink_cases = list(
        itertools.product(
            [(128, 512), (1024, 2048)],  # (window_left, kv_len)
            [4],  # sink_size
            [None, 2.0],  # sink_ptr_value
            [torch.bfloat16],  # dtype
        )
    )
    for (window_left, kv_len), sink_size, sink_ptr_value, dtype in sink_cases:
        run_batch_prefill_sink(
            batch_size=1,
            qo_len=128,
            kv_len=kv_len,
            page_size=16,
            num_qo_heads=8,
            num_kv_heads=1,
            head_dim=128,
            window_left=window_left,
            sink_size=sink_size,
            sink_ptr_value=sink_ptr_value,
            dtype=dtype,
            seed=42,
        )

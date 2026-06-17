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

    if kvcache_layout == "vec_k_col_v":
        # Decode-aligned col-V kernels are generated for the non-quant bf16/fp16
        # path and the separate fp8bf16 PER_TOKEN_HEAD path. The in-test fp8
        # pertensor path has no col-V kernel, so skip it here.
        if skip_test_if(
            is_input_fp8,
            "vec_k_col_v has no in-test FP8 pertensor kernel (bf16/fp16 only)",
        ):
            return True
        # ColumnMajor V is loaded as a single merged (D, TotalSeqK) view, so a V
        # tile must stay within one physical page. Supported page sizes mirror
        # the col-V kernel's single-page V-tile constraint.
        if skip_test_if(
            page_size not in (64, 1024),
            f"vec_k_col_v only supports page_size in (64, 1024), got {page_size}",
        ):
            return True
        if skip_test_if(
            page_size % k_vector_size != 0 or head_dim % k_vector_size != 0,
            "vec_k_col_v layout requires page/head dim divisible by vector size",
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


def _local_mask_rows(q_start, q_end, seqlen_q, seqlen_k, window_size, device):
    """
    Row-slice of construct_local_mask covering global Q indices [q_start, q_end).

    This reproduces construct_local_mask semantics with key_padding_mask=None
    and query_padding_mask=None (the only path the batch_prefill reference uses)
    but only materializes [q_end - q_start, seqlen_k] booleans instead of the
    full [seqlen_q, seqlen_k] mask, which would OOM at long sequences.
    """
    row_idx = torch.arange(q_start, q_end, device=device, dtype=torch.long).view(-1, 1)
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if window_size[0] < 0:
        return col_idx > row_idx + seqlen_k - seqlen_q + window_size[1]
    sk_t = torch.full_like(col_idx, seqlen_k)
    return torch.logical_or(
        col_idx > torch.minimum(row_idx + seqlen_k - seqlen_q + window_size[1], sk_t),
        col_idx < row_idx + seqlen_k - seqlen_q - window_size[0],
    )


def _ref_masked_attention_unchunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
    window_left: int = -1,
    logits_soft_cap: float = 0.0,
    return_lse: bool = False,
) -> torch.Tensor:
    """
    Canonical (pre-chunking) reference implementation, kept verbatim as the
    semantic anchor for the chunked path. Used by the equivalence test only;
    do not call from production paths — peak memory is H * Q * K * 4 bytes
    which OOMs at long sequences (e.g. 137 GB at H=8, Q=K=65536).
    """
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

    if return_lse:
        lse = torch.logsumexp(attn_weights, dim=-1)

    attn_weights = torch.softmax(attn_weights, dim=-1)
    if window_size[0] >= 0 or window_size[1] >= 0:
        attn_weights = attn_weights.masked_fill(
            torch.all(local_mask, dim=-1, keepdim=True), 0.0
        )
    out = torch.einsum("hqk,khd->qhd", attn_weights, value.float())

    if return_lse:
        return out.to(query), lse.float()
    return out.to(query)


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
    window_left: int = -1,
    logits_soft_cap: float = 0.0,
    return_lse: bool = False,
    q_chunk_size: int = None,
) -> torch.Tensor:
    """
    Reference implementation of masked attention with Q-dimension chunking.

    Args:
        query: [seqlen_q, num_heads, head_dim]
        key: [seqlen_k, num_heads, head_dim]
        value: [seqlen_k, num_heads, head_dim]
        causal: whether to use causal mask
        window_left: left window size for sliding window attention
        logits_soft_cap: soft cap for logits (0.0 = disabled)
        return_lse: whether to return log-sum-exp values
        q_chunk_size: rows of Q processed per iteration. Default reads
            BENCH_REF_Q_CHUNK env var (fallback 1024). Set <= 0 to disable
            chunking (single shot over the entire Q dim).

    Returns:
        If return_lse=False: output [seqlen_q, num_heads, head_dim]
        If return_lse=True: (output, lse) where lse is [num_heads, seqlen_q]

    Memory notes:
        Peak intermediate is the per-chunk attention matrix [H, Q_chunk, K] in
        fp32. For H=8, K=131072, Q_chunk=1024 that's ~4 GB — fits on a 192 GB
        MI308X with plenty of headroom. The unchunked version peaks at
        H * seqlen_q * seqlen_k * 4 bytes which OOMs at seq=65536+ on long
        contexts (137 GB at H=8, Q=K=65536; ~550 GB at Q=K=131072).
    """
    if causal:
        window_size = (window_left, 0)
    else:
        window_size = (-1, -1)

    head_dim = query.shape[2]
    seqlen_q = query.shape[0]
    seqlen_k = key.shape[0]
    num_heads = query.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    device = query.device

    has_local_mask = window_size[0] >= 0 or window_size[1] >= 0

    # Empty-query fast path: preserve the shape behavior of the original.
    if seqlen_q == 0:
        out = torch.empty(0, num_heads, head_dim, device=device, dtype=query.dtype)
        if return_lse:
            lse = torch.empty(num_heads, 0, device=device, dtype=torch.float32)
            return out, lse
        return out

    if q_chunk_size is None:
        q_chunk = int(os.environ.get("BENCH_REF_Q_CHUNK", "1024"))
    else:
        q_chunk = int(q_chunk_size)
    if q_chunk <= 0 or q_chunk > seqlen_q:
        q_chunk = seqlen_q

    # Materialize fp32 K/V once and reuse across all Q chunks. For H=8,
    # K=131072, D=128 that's ~537 MB per tensor in fp32, well within budget.
    key_f32 = key.float()
    value_f32 = value.float()

    out_chunks = []
    lse_chunks = [] if return_lse else None

    for q_start in range(0, seqlen_q, q_chunk):
        q_end = min(q_start + q_chunk, seqlen_q)
        q_block = query[q_start:q_end].float()

        # [num_heads, q_chunk, seqlen_k]
        attn = scale * torch.einsum("qhd,khd->hqk", q_block, key_f32)

        if 0 < logits_soft_cap:
            mode = int(os.environ.get("CK_TILE_ATTENTION_LOGITS_SOFT_CAP_DEFAULT", 0))
            if mode == 0:
                attn = logits_soft_cap * torch.tanh(attn / logits_soft_cap)
            else:
                attn = attn / (1.0 + torch.abs(attn / logits_soft_cap))

        local_mask = None
        if has_local_mask:
            # Construct only the rows we need. Uses global Q indices so the
            # causal/window boundary stays consistent with the unchunked path.
            local_mask = _local_mask_rows(
                q_start, q_end, seqlen_q, seqlen_k, window_size, device
            )
            attn.masked_fill_(local_mask, float("-inf"))

        # LSE is computed on the post-mask, pre-softmax logits — matches the
        # unchunked implementation. logsumexp of an all -inf row is -inf, which
        # propagates cleanly through downstream consumers.
        if return_lse:
            lse_chunks.append(torch.logsumexp(attn, dim=-1))

        attn = torch.softmax(attn, dim=-1)
        if has_local_mask:
            # Zero out fully-masked rows so softmax(all -inf) = nan becomes 0.
            attn = attn.masked_fill(
                torch.all(local_mask, dim=-1, keepdim=True), 0.0
            )

        # [q_chunk, num_heads, head_dim]
        out_chunks.append(torch.einsum("hqk,khd->qhd", attn, value_f32))

        # Drop chunk-local refs so the caching allocator can reuse the slot for
        # the next iteration's [H, Q_chunk, K] tensor.
        del attn
        if local_mask is not None:
            del local_mask

    out = torch.cat(out_chunks, dim=0)
    if return_lse:
        lse = torch.cat(lse_chunks, dim=1)
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


def vec_k_col_v_kv_cache(
    k_cache, v_cache, num_kv_heads, head_dim, page_size, k_vector_size
):
    """Decode-aligned VEC_K_COL_V layout: K is 5D vectorized (same as VECTORIZED),
    V is 4D ColumnMajor [Pages, Heads, HeadDim, PageSize]."""
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    k_cache = (
        k_cache.view(
            -1, page_size, num_kv_heads, head_dim // k_vector_size, k_vector_size
        )
        .permute(0, 2, 3, 1, 4)
        .contiguous()
    )
    # v_cache_ref shape: [Pages, PageSize, Heads, HeadDim] -> [Pages, Heads, HeadDim, PageSize]
    v_cache = v_cache.permute(0, 2, 3, 1).contiguous()
    return k_cache, v_cache


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
    if layout == "vec_k_col_v":
        return vec_k_col_v_kv_cache(
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


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("q_chunk_size", [256, 1000, 1024])
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.bfloat16])
def test_ref_masked_attention_chunking_equivalence(
    causal, logits_soft_cap, return_lse, q_chunk_size, input_dtype
):
    """
    Guard against silent semantic drift between the chunked
    ref_masked_attention and the canonical unchunked implementation.

    Runs at qlen=klen=4096, nhq=8, hd=128 — fits in either path. Exercises
    causal/non-causal, soft_cap on/off, return_lse on/off, plus a couple of
    chunk sizes including one (1000) that does not evenly divide seqlen_q so
    the final partial chunk gets covered too.

    Tolerances:
        - fp32 input: 1e-5 (true fp32 round-off scale; chunking and unchunked
          differ only because cuBLAS routes shape-dependent einsum kernels
          with different reduction order in the head_dim sum).
        - bf16 input: 1e-3 (~1 bf16 ULP at output magnitudes near 0.1, set by
          the final out.to(query) cast back to bf16). The pre-cast fp32 math
          is bit-identical to the fp32-input case.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device for fp32 attention")

    torch.manual_seed(0)
    seqlen_q = seqlen_k = 4096
    num_heads = 8
    head_dim = 128
    device = "cuda"

    q = torch.randn(seqlen_q, num_heads, head_dim, device=device, dtype=input_dtype)
    k = torch.randn(seqlen_k, num_heads, head_dim, device=device, dtype=input_dtype)
    v = torch.randn(seqlen_k, num_heads, head_dim, device=device, dtype=input_dtype)

    unchunked = _ref_masked_attention_unchunked(
        q, k, v,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        return_lse=return_lse,
    )
    chunked = ref_masked_attention(
        q, k, v,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        return_lse=return_lse,
        q_chunk_size=q_chunk_size,
    )

    out_atol = 1e-5 if input_dtype == torch.float32 else 1e-3
    # LSE is always fp32 in both paths (no output cast); use the fp32 bound.
    lse_atol = 1e-5

    if return_lse:
        out_u, lse_u = unchunked
        out_c, lse_c = chunked
        out_diff = (out_u.float() - out_c.float()).abs().max().item()
        lse_diff = (lse_u - lse_c).abs().max().item()
        assert out_diff <= out_atol, (
            f"chunked output diverged from unchunked: max-abs={out_diff} "
            f"(dtype={input_dtype}, causal={causal}, "
            f"soft_cap={logits_soft_cap}, return_lse={return_lse}, "
            f"q_chunk_size={q_chunk_size})"
        )
        assert lse_diff <= lse_atol, (
            f"chunked LSE diverged from unchunked: max-abs={lse_diff} "
            f"(dtype={input_dtype}, causal={causal}, "
            f"soft_cap={logits_soft_cap}, return_lse={return_lse}, "
            f"q_chunk_size={q_chunk_size})"
        )
    else:
        out_diff = (unchunked.float() - chunked.float()).abs().max().item()
        assert out_diff <= out_atol, (
            f"chunked output diverged from unchunked: max-abs={out_diff} "
            f"(dtype={input_dtype}, causal={causal}, "
            f"soft_cap={logits_soft_cap}, return_lse={return_lse}, "
            f"q_chunk_size={q_chunk_size})"
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


@pytest.mark.parametrize("kvcache_layout", ["linear", "vectorized", "vec_k_col_v"])
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
    q_descale_per_token=None,
    k_descale_per_token=None,
    v_descale_per_head=None,
    kv_last_page_lens=None,
    block_table=None,
    seqlen_k=None,
    p_scale=None,
    p_scale_inv=None,
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
        q_descale_per_token=q_descale_per_token,
        k_descale_per_token=k_descale_per_token,
        v_descale_per_head=v_descale_per_head,
        kv_last_page_lens=kv_last_page_lens,
        block_table=block_table,
        seqlen_k=seqlen_k,
        p_scale=p_scale,
        p_scale_inv=p_scale_inv,
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


def per_token_per_head_quant_q(tensor, quant_dtype):
    """
    Quantize Q with per-token-per-head scale (PER_TOKEN_HEAD mode).

    Args:
        tensor: [total_q, num_heads, head_dim]
        quant_dtype: target quantization dtype (fp8)

    Returns:
        quantized: [total_q, num_heads, head_dim]
        descales:  [total_q, num_heads] fp32 per-token-per-head descale
    """
    fp8_max = torch.finfo(quant_dtype).max
    abs_max = tensor.abs().amax(dim=-1).clamp(min=1e-12)  # [total_q, num_heads]
    descales = (abs_max / fp8_max).float()
    quantized = (tensor / descales.unsqueeze(-1)).to(quant_dtype)
    return quantized, descales


def per_token_per_head_quant_k_paged(tensor, quant_dtype):
    """
    Quantize paged K with per-token-per-head scale (PER_TOKEN_HEAD mode).

    Args:
        tensor: [num_pages, page_size, num_kv_heads, head_dim]
        quant_dtype: target quantization dtype (fp8)

    Returns:
        quantized: [num_pages, page_size, num_kv_heads, head_dim]
        descales:  [num_pages, page_size, num_kv_heads] fp32
    """
    fp8_max = torch.finfo(quant_dtype).max
    abs_max = tensor.abs().amax(dim=-1).clamp(min=1e-12)
    descales = (abs_max / fp8_max).float()
    quantized = (tensor / descales.unsqueeze(-1)).to(quant_dtype)
    return quantized, descales


def per_head_quant_v_paged(tensor, quant_dtype):
    """
    Quantize paged V with per-head scale (PER_TOKEN_HEAD mode V scaling).

    Args:
        tensor: [num_pages, page_size, num_kv_heads, head_dim]
        quant_dtype: target quantization dtype (fp8)

    Returns:
        quantized: [num_pages, page_size, num_kv_heads, head_dim]
        descales:  [num_kv_heads] fp32
    """
    fp8_max = torch.finfo(quant_dtype).max
    # Max over (num_pages, page_size, head_dim) -> [num_kv_heads]
    abs_max = tensor.abs().amax(dim=(0, 1, 3)).clamp(min=1e-12)
    descales = (abs_max / fp8_max).float()
    # Broadcast: [1, 1, num_kv_heads, 1]
    quantized = (tensor / descales.view(1, 1, -1, 1)).to(quant_dtype)
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


@pytest.mark.parametrize(
    "num_blocks,page_size",
    [
        (5000, 1024),  # ~10GB KV cache
        (10000, 1024),  # ~20GB KV cache
    ],
)
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("input_dtype", ["bf16", "fp8"])
def test_batch_prefill_large_kvcache(
    num_blocks,
    page_size,
    num_kv_heads,
    head_dim,
    causal,
    input_dtype,
):
    """
    Test that batch prefill produces correct results with large KV caches
    whose element offsets exceed the INT32_MAX boundary (~4GB for bf16).
    """
    torch.manual_seed(42)

    is_fp8 = input_dtype == "fp8"
    dtype = torch.bfloat16
    num_qo_heads = num_kv_heads  # MHA (no GQA) for simplicity

    stride_per_page = page_size * num_kv_heads * head_dim  # elements per page block
    max_offset = (num_blocks - 1) * stride_per_page
    INT32_MAX = 2**31 - 1

    if max_offset <= INT32_MAX:
        pytest.skip(
            f"max_offset {max_offset} doesn't exceed INT32_MAX, not an overflow test"
        )

    # Check available GPU memory -- skip if not enough
    free_mem = torch.cuda.mem_get_info()[0]
    elem_size = 1 if is_fp8 else 2  # fp8=1 byte, bf16=2 bytes
    required_mem = 2 * num_blocks * page_size * num_kv_heads * head_dim * elem_size
    if free_mem < required_mem * 1.1:  # 10% headroom
        pytest.skip(
            f"Not enough GPU memory: need {required_mem / 1e9:.1f}GB, "
            f"have {free_mem / 1e9:.1f}GB"
        )

    # Allocate KV caches in linear layout: [num_blocks, page_size, num_kv_heads, head_dim]
    k_cache_bf16 = torch.randn(
        num_blocks, page_size, num_kv_heads, head_dim, device="cuda", dtype=dtype
    )
    v_cache_bf16 = torch.randn(
        num_blocks, page_size, num_kv_heads, head_dim, device="cuda", dtype=dtype
    )

    if is_fp8:
        k_cache, k_descale = per_tensor_quant(k_cache_bf16, quant_dtype=dtypes.fp8)
        v_cache, v_descale = per_tensor_quant(v_cache_bf16, quant_dtype=dtypes.fp8)
    else:
        k_cache = k_cache_bf16
        v_cache = v_cache_bf16

    # Test pages that span the overflow boundary
    qo_len = 1
    kv_len = page_size  # one full page

    q_bf16 = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype)
    if is_fp8:
        q, q_descale = per_tensor_quant(q_bf16, quant_dtype=dtypes.fp8)
    else:
        q = q_bf16
    cu_seqlens_q = torch.tensor([0, qo_len], device="cuda", dtype=torch.int32)

    # Test at several page indices: before, at, and after the overflow boundary
    overflow_page = INT32_MAX // stride_per_page
    test_pages = [
        0,
        overflow_page - 1,
        overflow_page,
        overflow_page + 1,
        num_blocks - 1,
    ]
    test_pages = [p for p in test_pages if 0 <= p < num_blocks]
    # Remove duplicates while preserving order
    test_pages = list(dict.fromkeys(test_pages))

    threshold = 0.055 if is_fp8 else 0.01

    for page_idx in test_pages:
        offset = page_idx * stride_per_page
        label = "OVERFLOW" if offset > INT32_MAX else "safe"

        kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
        kv_page_indices = torch.tensor([page_idx], device="cuda", dtype=torch.int32)
        kv_last_page_lens = torch.tensor([page_size], device="cuda", dtype=torch.int32)

        extra_kwargs = {}
        if is_fp8:
            extra_kwargs = dict(
                q_descale=q_descale, k_descale=k_descale, v_descale=v_descale
            )

        result = aiter.mha_batch_prefill_func(
            q,
            k_cache,
            v_cache,
            cu_seqlens_q,
            kv_indptr,
            kv_page_indices,
            qo_len,
            kv_len,
            causal=causal,
            kv_last_page_lens=kv_last_page_lens,
            **extra_kwargs,
        )
        out = result[0] if isinstance(result, (list, tuple)) else result

        # Reference: direct attention on the original bf16 data
        k_page = k_cache_bf16[page_idx]  # [page_size, num_kv_heads, head_dim]
        v_page = v_cache_bf16[page_idx]
        o_ref = ref_masked_attention(q_bf16, k_page, v_page, causal=causal)

        max_diff = (out - o_ref).abs().max().item()
        assert max_diff < threshold, (
            f"[{input_dtype}] page {page_idx} (offset={offset}, {label}): "
            f"max_diff={max_diff} exceeds threshold {threshold}"
        )


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
    skip_reference=False,
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
    if skip_reference:
        o_ref = None
    else:
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
    if skip_reference:
        return profile_result
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


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (16, 16)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("qo_len,kv_len", [(128, 1024), (512, 2048), (1024, 4096)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("table_layout", ["sglang", "vllm"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("page_size", [64, 1024])
def test_batch_prefill_per_token_head_pytest(
    batch_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    qo_len,
    kv_len,
    causal,
    table_layout,
    logits_soft_cap,
    page_size,
):
    """Pytest wrapper for PER_TOKEN_HEAD FP8 batch prefill test.

    Q/K: per-token-per-head fp32 descale. V: per-head fp32 descale.
    """
    if skip_test_if(
        should_skip_rocm72_issue(causal, logits_soft_cap),
        "ROCm 7.2 + gfx950 compiler issue with causal=True + logits_soft_cap=0.0",
    ):
        return

    run_batch_prefill_per_token_head(
        kvcache_layout="linear",
        table_layout=table_layout,
        batch_size=batch_size,
        qo_len=qo_len,
        kv_len=kv_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        dtype=torch.bfloat16,
        contiguous_kv=True,
        seed=42,
    )


# Small-seqlen regression for the PER_TOKEN_HEAD q_descale coredump. When
# qo_len <= kM0 (128), the query tile is mostly padding rows; before the fix the
# kernel read q_descale_per_token for those padding rows (past the [total_q]
# tensor) and faulted. Exercises the generated PER_TOKEN_HEAD path on the
# layout-independent linear cache.
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("qo_len,kv_len", [(1, 1), (4, 4), (16, 16), (16, 64)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [64, 1024])
def test_batch_prefill_per_token_head_small_seqlen_pytest(
    batch_size,
    qo_len,
    kv_len,
    causal,
    page_size,
):
    """Regression: PER_TOKEN_HEAD must not OOB-read q_descale on padding rows
    when the query tile (kM0=128) is larger than the valid sequence length."""
    if skip_test_if(
        should_skip_rocm72_issue(causal, 0.0),
        "ROCm 7.2 + gfx950 compiler issue with causal=True + logits_soft_cap=0.0",
    ):
        return

    run_batch_prefill_per_token_head(
        kvcache_layout="linear",
        table_layout="sglang",
        batch_size=batch_size,
        qo_len=qo_len,
        kv_len=kv_len,
        page_size=page_size,
        num_qo_heads=32,
        num_kv_heads=8,
        head_dim=128,
        causal=causal,
        logits_soft_cap=0.0,
        dtype=torch.bfloat16,
        contiguous_kv=True,
        seed=42,
    )


def _make_cross_layer_5d_view(
    num_blocks: int,
    num_kv_heads: int,
    num_layers: int,
    page_size: int,
    head_dim: int,
    layer_idx: int,
    dtype: torch.dtype,
    device: str = "cuda",
    fill: torch.Tensor | None = None,
):
    """Build a Cross-Layer 5D KV cache and return the per-layer
    non-contiguous 5D view ``(2, num_blocks, num_kv_heads, page_size, head_dim)``.

    Mirrors the construction in
    ``Cross-Layer_5D_KV_Cache_Operator_Adaptation_Plan_EN.md``:

      1. Allocate a contiguous 6D physical tensor of shape
         ``(num_blocks, num_kv_heads, num_layers, 2, page_size, head_dim)``.
      2. Permute it into the logical 6D order
         ``(num_layers, 2, num_blocks, num_kv_heads, page_size, head_dim)`` -- this
         view is non-contiguous.
      3. Select layer ``layer_idx`` along the leading num_layers dim, yielding the
         per-layer 5D view ``(2, num_blocks, num_kv_heads, page_size, head_dim)``
         with the cross-layer stride pattern.

    Optionally fills the underlying buffer first; the returned view shares storage
    with that buffer so subsequent K/V reads observe the same data.
    """
    physical = torch.empty(
        num_blocks,
        num_kv_heads,
        num_layers,
        2,
        page_size,
        head_dim,
        dtype=dtype,
        device=device,
    )
    if fill is not None:
        physical.copy_(fill)
    # Logical 6D view (num_layers, 2, num_blocks, num_kv_heads, page_size, head_dim)
    logical6d = physical.permute(2, 3, 0, 1, 4, 5)
    # Per-layer 5D view (2, num_blocks, num_kv_heads, page_size, head_dim)
    return physical, logical6d[layer_idx]


@pytest.mark.parametrize("num_layers", [4, 80])
@pytest.mark.parametrize("layer_idx", [0, 1])
@pytest.mark.parametrize("num_blocks", [4])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_kv_heads,num_qo_heads", [(1, 4)])
@pytest.mark.parametrize("head_dim", [128])
def test_batch_prefill_cross_layer_5d_layout_strides(
    num_layers,
    layer_idx,
    num_blocks,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    """Metadata-only test for Cross-Layer 5D KV cache.

    Verifies that the per-layer 5D view produced by the framework-side construction
    has exactly the stride pattern documented in section 5.3 ("Stride in
    Non-Contiguous Mode, Cross-Layer Mode") of
    ``Cross-Layer_5D_KV_Cache_Operator_Adaptation_Plan_EN.md``::

        stride[0] = page_size * head_dim
        stride[1] = num_layers * 2 * num_kv_heads * page_size * head_dim
        stride[2] = num_layers * 2 * page_size * head_dim
        stride[3] = head_dim
        stride[4] = 1

    This is the precondition the kernel relies on; running it does not require a
    GPU build of the prefill kernel itself.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    _, kv_cache = _make_cross_layer_5d_view(
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
        page_size=page_size,
        head_dim=head_dim,
        layer_idx=layer_idx,
        dtype=torch.bfloat16,
    )

    B = page_size
    D = head_dim
    H = num_kv_heads
    L = num_layers

    assert kv_cache.shape == (2, num_blocks, H, B, D)
    assert not kv_cache.is_contiguous(), (
        "Cross-Layer 5D KV cache per-layer view must be non-contiguous"
    )
    expected_strides = (B * D, L * 2 * H * B * D, L * 2 * B * D, D, 1)
    assert kv_cache.stride() == expected_strides, (
        f"Per-layer 5D view stride mismatch:\n"
        f"  got      = {kv_cache.stride()}\n"
        f"  expected = {expected_strides}\n"
        f"  (B={B}, D={D}, H={H}, L={L})"
    )

    # The K and V 4D slices must also be non-contiguous with the cross-layer
    # stride pattern and still have innermost stride == 1 (head_dim is contiguous,
    # which is what lets CK Tile keep its 16B vectorized loads).
    k_slice = kv_cache[0]
    v_slice = kv_cache[1]
    expected_kv_strides = (L * 2 * H * B * D, L * 2 * B * D, D, 1)
    assert k_slice.shape == (num_blocks, H, B, D)
    assert v_slice.shape == (num_blocks, H, B, D)
    assert k_slice.stride() == expected_kv_strides
    assert v_slice.stride() == expected_kv_strides
    assert k_slice.stride(-1) == 1
    assert v_slice.stride(-1) == 1


def test_batch_prefill_cross_layer_5d_layout_rejects_layout_mismatch():
    """`kv_layout` must agree with explicit `kv_cache` input."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    _, kv_cache = _make_cross_layer_5d_view(
        num_blocks=4,
        num_kv_heads=1,
        num_layers=4,
        page_size=16,
        head_dim=128,
        layer_idx=0,
        dtype=torch.bfloat16,
    )
    q = torch.zeros(4, 4, 128, dtype=torch.bfloat16, device="cuda")
    cu_seqlens_q = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
    kv_indptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    kv_page_indices = torch.tensor([0], device="cuda", dtype=torch.int32)
    kv_last_page_lens = torch.tensor([16], device="cuda", dtype=torch.int32)

    with pytest.raises(ValueError, match="kv_layout"):
        aiter.mha_batch_prefill_func(
            q,
            cu_seqlens_q=cu_seqlens_q,
            kv_indptr=kv_indptr,
            kv_page_indices=kv_page_indices,
            max_seqlen_q=4,
            max_seqlen_k=16,
            kv_cache=kv_cache,
            kv_layout=aiter.KV_LAYOUT_LINEAR,
            kv_last_page_lens=kv_last_page_lens,
        )

    with pytest.raises(ValueError, match="not both"):
        aiter.mha_batch_prefill_func(
            q,
            k=kv_cache[0],
            v=kv_cache[1],
            cu_seqlens_q=cu_seqlens_q,
            kv_indptr=kv_indptr,
            kv_page_indices=kv_page_indices,
            max_seqlen_q=4,
            max_seqlen_k=16,
            kv_cache=kv_cache,
            kv_last_page_lens=kv_last_page_lens,
        )


@pytest.mark.parametrize("num_layers", [4])
@pytest.mark.parametrize("layer_idx", [0, 2])
@pytest.mark.parametrize("num_blocks", [4])
@pytest.mark.parametrize("page_size", [16])
@pytest.mark.parametrize("num_kv_heads,num_qo_heads", [(1, 4), (2, 8)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("qo_len", [16, 32])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_batch_prefill_cross_layer_5d_layout_matches_contiguous(
    num_layers,
    layer_idx,
    num_blocks,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    batch_size,
    qo_len,
    causal,
    dtype,
):
    """Functional test for Cross-Layer 5D KV cache prefill.

    Builds a 6D physical KV buffer, fills the layer-of-interest slot with random
    data, then runs ``mha_batch_prefill_func`` two ways:

      * legacy contiguous LINEAR_LAYOUT path on a packed ``[N, B, H, D]`` copy,
      * cross-layer non-contiguous LINEAR_HEADS_FIRST_LAYOUT path on the
        ``[N, H, B, D]`` per-layer view.

    Both must produce numerically equivalent attention outputs, which validates
    that the wrapper plumbs the correct strides for the non-contiguous layout
    into the CK Tile kernel.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")
    if num_qo_heads % num_kv_heads != 0:
        pytest.skip("GQA group size must divide num_qo_heads")

    torch.manual_seed(0xC8055)
    device = "cuda"

    # Build the 6D physical buffer and fill the *layer-of-interest* with random
    # K/V data; other layers stay zeroed so a cross-talk bug would be obvious.
    physical = torch.zeros(
        num_blocks,
        num_kv_heads,
        num_layers,
        2,
        page_size,
        head_dim,
        dtype=dtype,
        device=device,
    )
    layer_data = torch.randn(
        2,
        num_blocks,
        num_kv_heads,
        page_size,
        head_dim,
        dtype=dtype,
        device=device,
    )
    # Logical 6D view shares storage with `physical`; assign into the per-layer slot.
    logical6d = physical.permute(2, 3, 0, 1, 4, 5)
    logical6d[layer_idx].copy_(layer_data)
    kv_cache_view = logical6d[layer_idx]  # 5D (2, N, H, B, D) non-contiguous view

    assert not kv_cache_view.is_contiguous()
    assert kv_cache_view.stride(-1) == 1

    # Packed reference: a contiguous [N, B, H, D] for K and V each, copied out of
    # the cross-layer view so the kernel under both paths sees the same numeric
    # data, just via different stride patterns.
    k_xlayer = kv_cache_view[0]  # [N, H, B, D]
    v_xlayer = kv_cache_view[1]
    k_packed = k_xlayer.permute(0, 2, 1, 3).contiguous()  # [N, B, H, D]
    v_packed = v_xlayer.permute(0, 2, 1, 3).contiguous()

    # Build a single-page-per-batch scenario: kv_indptr [0,1,2,...]. Each batch
    # element points to a distinct page so we exercise the per-block stride.
    assert batch_size <= num_blocks
    kv_indptr = torch.arange(
        batch_size + 1, device=device, dtype=torch.int32
    )
    kv_page_indices = torch.arange(
        batch_size, device=device, dtype=torch.int32
    )
    kv_last_page_lens = torch.full(
        (batch_size,), page_size, device=device, dtype=torch.int32
    )

    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * qo_len, qo_len, device=device, dtype=torch.int32
    )
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, dtype=dtype, device=device
    ) * 0.1

    # Path A: packed LINEAR_LAYOUT (existing fast path).
    out_packed = aiter.mha_batch_prefill_func(
        q,
        k_packed,
        v_packed,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        qo_len,
        page_size,
        causal=causal,
        kv_last_page_lens=kv_last_page_lens,
    )
    out_packed_t = out_packed[0] if isinstance(out_packed, (list, tuple)) else out_packed

    # Path B: cross-layer LINEAR_HEADS_FIRST_LAYOUT on the non-contiguous view.
    out_xlayer = aiter.mha_batch_prefill_func(
        q,
        cu_seqlens_q=cu_seqlens_q,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        max_seqlen_q=qo_len,
        max_seqlen_k=page_size,
        causal=causal,
        kv_cache=kv_cache_view,
        kv_last_page_lens=kv_last_page_lens,
    )
    out_xlayer_t = out_xlayer[0] if isinstance(out_xlayer, (list, tuple)) else out_xlayer

    rtol, atol = get_tolerances(dtype, is_fp8=False)
    torch.testing.assert_close(out_xlayer_t, out_packed_t, rtol=rtol, atol=atol)


@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_cross_layer_5d_matches_torch_reference(causal):
    """End-to-end correctness test: cross-layer 5D prefill vs pure-torch reference.

    Companion to ``test_batch_prefill_cross_layer_5d_layout_matches_contiguous``,
    which only validates kernel-vs-kernel self-consistency between the packed
    and cross-layer paths. This one feeds the cross-layer kernel output through
    ``ref_masked_attention`` (einsum + softmax) so the actual attention math
    is independently validated on the non-contiguous ``[N, H, B, D]`` view.

    Single hard-coded scenario chosen to cover the gaps in the existing
    cross-layer tests in one shot:

      - ``num_layers = 80``  -- nominal cross-layer value; exercises the
        pathological per-head / per-block stride magnitudes
        (``L*2*B*D = 327_680`` elements).
      - ``layer_idx = 2``    -- non-zero, catches a layer-select offset bug.
      - ``batch_size = 2``   -- per-batch page counts ``[3, 2]`` exercise the
        multi-page-per-sequence path that the existing tests skip.
      - ``last_page_lens = [5, 11]`` -- partial last pages of different sizes,
        which the existing tests likewise skip (they all use full last pages).
      - ``GQA num_kv_heads=1 -> num_qo_heads=4``.
    """
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/HIP device")

    torch.manual_seed(0xCA5CADE)
    device = "cuda"
    dtype = torch.bfloat16

    num_layers = 80
    layer_idx = 2
    num_blocks = 8
    page_size = 16
    head_dim = 128
    num_kv_heads = 1
    num_qo_heads = 4
    pages_per_seq = [3, 2]
    last_page_lens = [5, 11]
    qo_lens = [10, 7]
    batch_size = len(pages_per_seq)

    # Build the 6D physical buffer; only fill the layer of interest with random
    # K/V. Other layers stay zeroed so a stride-leak bug into another layer
    # would corrupt the attention output and trip the reference comparison.
    physical = torch.zeros(
        num_blocks, num_kv_heads, num_layers, 2, page_size, head_dim,
        dtype=dtype, device=device,
    )
    layer_data = torch.randn(
        2, num_blocks, num_kv_heads, page_size, head_dim,
        dtype=dtype, device=device,
    ) * 0.1
    logical6d = physical.permute(2, 3, 0, 1, 4, 5)  # (L, 2, N, H, B, D)
    logical6d[layer_idx].copy_(layer_data)
    kv_cache_view = logical6d[layer_idx]  # (2, N, H, B, D), non-contiguous

    # Sanity: assert the layout we fed the kernel matches the documented spec.
    H, B, D, L = num_kv_heads, page_size, head_dim, num_layers
    assert kv_cache_view.shape == (2, num_blocks, H, B, D)
    assert kv_cache_view.stride() == (B * D, L * 2 * H * B * D, L * 2 * B * D, D, 1)
    assert not kv_cache_view.is_contiguous()

    total_q = sum(qo_lens)
    cu_seqlens_q = torch.tensor(
        [0] + list(itertools.accumulate(qo_lens)),
        dtype=torch.int32, device=device,
    )
    kv_indptr = torch.tensor(
        [0] + list(itertools.accumulate(pages_per_seq)),
        dtype=torch.int32, device=device,
    )
    # Distinct pages per batch out of the available [0, num_blocks).
    kv_page_indices_list: list[int] = []
    cursor = 0
    for npages in pages_per_seq:
        kv_page_indices_list.extend(range(cursor, cursor + npages))
        cursor += npages
    assert cursor <= num_blocks
    kv_page_indices = torch.tensor(
        kv_page_indices_list, dtype=torch.int32, device=device
    )
    kv_last_page_lens = torch.tensor(
        last_page_lens, dtype=torch.int32, device=device
    )
    seqlen_k_per_batch = [
        (pages_per_seq[b] - 1) * page_size + last_page_lens[b]
        for b in range(batch_size)
    ]
    max_seqlen_k = max(seqlen_k_per_batch)

    q = torch.randn(
        total_q, num_qo_heads, head_dim, dtype=dtype, device=device,
    ) * 0.1

    out = aiter.mha_batch_prefill_func(
        q,
        cu_seqlens_q=cu_seqlens_q,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        max_seqlen_q=max(qo_lens),
        max_seqlen_k=max_seqlen_k,
        causal=causal,
        kv_cache=kv_cache_view,
        kv_last_page_lens=kv_last_page_lens,
    )
    out_t = out[0] if isinstance(out, (list, tuple)) else out

    # Per-batch torch reference: gather K/V tokens for this batch's pages out
    # of the cross-layer view, repeat for GQA, hand to ref_masked_attention.
    k_view = kv_cache_view[0]  # [N, H, B, D]
    v_view = kv_cache_view[1]
    cu_q_cpu = cu_seqlens_q.cpu().tolist()
    iptr_cpu = kv_indptr.cpu().tolist()
    pidx_cpu = kv_page_indices.cpu().tolist()
    gqa_ratio = num_qo_heads // num_kv_heads
    rtol, atol = get_tolerances(dtype, is_fp8=False)

    for b in range(batch_size):
        pages_b = pidx_cpu[iptr_cpu[b]:iptr_cpu[b + 1]]
        last_len_b = last_page_lens[b]
        k_segs, v_segs = [], []
        for j, p in enumerate(pages_b):
            tokens_in_page = page_size if j < len(pages_b) - 1 else last_len_b
            # k_view[p] is [H, B, D]; transpose to [B, H, D] then truncate to
            # the valid token range for the last page.
            k_segs.append(
                k_view[p, :, :tokens_in_page, :].permute(1, 0, 2).contiguous()
            )
            v_segs.append(
                v_view[p, :, :tokens_in_page, :].permute(1, 0, 2).contiguous()
            )
        ki = torch.cat(k_segs, dim=0)  # [seqlen_k, H, D]
        vi = torch.cat(v_segs, dim=0)
        if gqa_ratio > 1:
            ki = ki.repeat_interleave(gqa_ratio, dim=1)
            vi = vi.repeat_interleave(gqa_ratio, dim=1)
        qi = q[cu_q_cpu[b]:cu_q_cpu[b + 1]]
        ref_b = ref_masked_attention(qi, ki, vi, causal=causal)
        torch.testing.assert_close(
            out_t[cu_q_cpu[b]:cu_q_cpu[b + 1]],
            ref_b,
            rtol=rtol,
            atol=atol,
        )

# Decode-aligned VEC_K_COL_V_LAYOUT pytest. Codegen only emits this variant for
# fp8bf16 PER_TOKEN_HEAD with the sglang lookup table (see fmha_batch_prefill.py).
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8), (16, 16)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("qo_len,kv_len", [(512, 2048), (1024, 4096)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("page_size", [64, 1024])
def test_batch_prefill_per_token_head_vec_k_col_v_pytest(
    batch_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    qo_len,
    kv_len,
    causal,
    logits_soft_cap,
    page_size,
):
    """Pytest wrapper for PER_TOKEN_HEAD FP8 batch prefill on the decode-aligned
    VEC_K_COL_V layout (5D vectorized K + 4D ColumnMajor V)."""
    if skip_test_if(
        should_skip_rocm72_issue(causal, logits_soft_cap),
        "ROCm 7.2 + gfx950 compiler issue with causal=True + logits_soft_cap=0.0",
    ):
        return

    run_batch_prefill_per_token_head(
        kvcache_layout="vec_k_col_v",
        table_layout="sglang",
        batch_size=batch_size,
        qo_len=qo_len,
        kv_len=kv_len,
        page_size=page_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        dtype=torch.bfloat16,
        contiguous_kv=True,
        seed=42,
    )


# Caller-supplied softmax-P scale variants for PER_TOKEN_HEAD.
# p_scale is per-q-head fp32 multiplier folded into the exp2 shift before fp8 P
# quantization; it cancels mathematically in the final output, so the FP32
# reference comparison must hold regardless of the value. Values kept <= 1.0 to
# stay clear of FP8 e4m3 saturation (internal P quantization scale is 256).
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(32, 8)])
@pytest.mark.parametrize("qo_len,kv_len", [(512, 2048)])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "p_scale_mode", ["none", "ones", "half", "per_head_random"]
)
def test_batch_prefill_per_token_head_p_scale_pytest(
    batch_size,
    num_qo_heads,
    num_kv_heads,
    qo_len,
    kv_len,
    causal,
    p_scale_mode,
):
    """PER_TOKEN_HEAD FP8 batch prefill with caller-supplied softmax-P scale."""
    if skip_test_if(
        should_skip_rocm72_issue(causal, 0.0),
        "ROCm 7.2 + gfx950 compiler issue with causal=True + logits_soft_cap=0.0",
    ):
        return

    if p_scale_mode == "none":
        p_scale = None
    elif p_scale_mode == "ones":
        p_scale = torch.ones(num_qo_heads, dtype=torch.float32, device="cuda")
    elif p_scale_mode == "half":
        p_scale = torch.full(
            (num_qo_heads,), 0.5, dtype=torch.float32, device="cuda"
        )
    else:  # per_head_random in [0.25, 1.0]
        g = torch.Generator(device="cuda").manual_seed(123)
        p_scale = (
            0.25
            + 0.75
            * torch.rand(
                num_qo_heads, dtype=torch.float32, device="cuda", generator=g
            )
        )

    run_batch_prefill_per_token_head(
        kvcache_layout="linear",
        table_layout="sglang",
        batch_size=batch_size,
        qo_len=qo_len,
        kv_len=kv_len,
        page_size=64,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=128,
        causal=causal,
        logits_soft_cap=0.0,
        dtype=torch.bfloat16,
        contiguous_kv=True,
        seed=42,
        p_scale=p_scale,
    )


def run_batch_prefill_per_token_head(
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
    skip_reference=False,
    p_scale=None,
    p_scale_inv=None,
):
    """
    FP8 batch prefill with PER_TOKEN_HEAD quantization:
      Q descale: [total_q, nhead_q] fp32
      K descale: [num_total_pages, page_block_size, nhead_k] fp32
      V descale: [nhead_k] fp32

    p_scale (optional, [num_qo_heads] fp32 on device): folded into the softmax
    exp2 shift before fp8 P quantization. p_scale cancels in the final output,
    so the FP32 reference comparison is unaffected.
    """
    if seed is not None:
        torch.manual_seed(seed)

    quant_dtype = dtypes.fp8
    SUPPORTED_PAGE_SIZES = (16, 64, 1024)
    if page_size not in SUPPORTED_PAGE_SIZES:
        if skip_test_if(
            True,
            f"PER_TOKEN_HEAD only supports page_size in {SUPPORTED_PAGE_SIZES}, got {page_size}",
        ):
            return {"status": "skipped"}

    k_vector_size = get_vector_size(quant_dtype)

    if skip_test_if(
        should_skip_rocm72_issue(causal, logits_soft_cap),
        "ROCm 7.2 + gfx950 compiler issue with causal=True + logits_soft_cap=0.0",
    ):
        return {"status": "skipped"}

    qo_lens = build_qo_lens(batch_size, qo_len, randomize=batch_size > 1)
    kv_lens = build_kv_lens(batch_size, kv_len, qo_lens, randomize=batch_size > 1)
    max_qo_len = qo_lens.max().item()
    max_kv_len = kv_lens.max().item()

    q = build_q_tensor_for_test(
        qo_lens,
        batch_size,
        qo_len,
        num_qo_heads,
        head_dim,
        dtype,
        None,
        None,
        is_input_fp8=True,
    )

    kv_cache = build_paged_kv_cache(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        head_dim,
        kv_lens,
        None,
        None,
        dtype,
        use_uniform=True,
        contiguous_kv=contiguous_kv,
    )

    kv_data_fp32 = kv_cache["kv_data_fp32"]
    kv_data = kv_cache["kv_data"]
    kv_indptr = kv_cache["kv_indptr_cpu"]
    kv_indices = kv_cache["kv_indices_cpu"]
    kv_last_page_len_cpu = kv_cache["kv_last_page_len_cpu"]

    k_paged_ref, v_paged_ref = split_kv_pages(kv_data)

    cu_seqlens_q = convert_lens_to_indptr(qo_lens).cuda()
    q_indptr_cpu = convert_lens_to_indptr(qo_lens)

    # FP32 reference (bf16 ground truth)
    if skip_reference:
        o_ref = None
    else:
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

    # PER_TOKEN_HEAD quantization
    q_fp8, q_descale_per_token = per_token_per_head_quant_q(q, quant_dtype)
    q_descale_per_token = q_descale_per_token.cuda()

    k_paged_fp8, k_descale_per_token = per_token_per_head_quant_k_paged(
        k_paged_ref, quant_dtype
    )
    k_descale_per_token = k_descale_per_token.cuda()

    v_paged_fp8, v_descale_per_head = per_head_quant_v_paged(v_paged_ref, quant_dtype)
    v_descale_per_head = v_descale_per_head.cuda()

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
    elif kvcache_layout == "vec_k_col_v":
        # k_paged_fp8 / v_paged_fp8 currently have shape [Pages, PageSize, Heads, HeadDim].
        # vec_k_col_v_kv_cache expects the same layout, so feed them directly without flattening.
        k_paged, v_paged = vec_k_col_v_kv_cache(
            k_paged_fp8,
            v_paged_fp8,
            num_kv_heads,
            head_dim,
            page_size,
            k_vector_size,
        )
    else:
        k_paged = k_paged_fp8
        v_paged = v_paged_fp8

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
    elif kvcache_layout == "vec_k_col_v":
        # bf16 reference also runs through the prefill kernel for correctness; bf16 has no
        # PER_TOKEN_HEAD codegen variant for vec_k_col_v, so we keep the bf16 reference
        # in plain linear layout (the prefill bf16 path supports linear K/V natively).
        k_cache_bf16 = k_paged_ref
        v_cache_bf16 = v_paged_ref
    else:
        k_cache_bf16 = k_paged_ref
        v_cache_bf16 = v_paged_ref

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
        q_descale_per_token=q_descale_per_token,
        k_descale_per_token=k_descale_per_token,
        v_descale_per_head=v_descale_per_head,
        kv_last_page_lens=kv_last_page_len_gpu,
        block_table=block_table_gpu,
        seqlen_k=seqlen_k_gpu,
        p_scale=p_scale,
        p_scale_inv=p_scale_inv,
        profile=profile,
    )
    if profile:
        out_fp8, time_us, tflops = run_result
        profile_result = {"status": "passed", "time_us": time_us, "tflops": tflops}
    else:
        out_fp8 = run_result

    if skip_reference:
        return profile_result

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

    assert out_fp8.abs().max().item() > 1e-6, "FP8 kernel output is all zeros!"
    assert out_bf16.abs().max().item() > 1e-6, "BF16 kernel output is all zeros!"
    assert o_ref.abs().max().item() > 1e-6, "FP32 reference output is all zeros!"

    verify_fp8_output(out_fp8, o_ref)

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


# ===========================================================================
# ASM qkptph/vph FP8 causal paged batch-prefill kernel (module
# module_mha_batch_prefill_asm). Self-contained: builds paged inputs in the
# exact layout the asm launcher consumes (packed-Q via cu_seqlens, vec_k_col_v
# K, column-major V, per-token-head q/k descales, per-head v descale, optional
# p_scale, SGLang 1D page table) and validates against an fp32 reference whose
# math mirrors scripts/f8_fmha_prefill/fwd_fp8.cpp (poc_kl).
# ===========================================================================
from aiter.ops.mha_batch_prefill_asm import mha_batch_prefill_asm  # noqa: E402

_ASM_FP8 = dtypes.fp8
_ASM_FP8_MAX = float(torch.finfo(_ASM_FP8).max)
_ASM_PAGE = 16
_ASM_VECX = 16  # FP8 128-bit vector width; equals page size


def _asm_quant_per_row(x):
    """Per-(row) symmetric fp8 quant over the last dim. Returns (fp8, descale)."""
    amax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    descale = (amax / _ASM_FP8_MAX).to(torch.float32)
    xq = (x / descale).to(_ASM_FP8)
    return xq, descale.squeeze(-1)


def _asm_reference(qf8, kf8, vf8, q_desc, k_desc, v_desc, p_scale, scale, gqa):
    """fp32 reference for one batch element.

    Shapes: qf8 [sq,hq,d], kf8/vf8 [sk,hk,d], q_desc [sq,hq],
    k_desc [sk,hk], v_desc [hk], p_scale [hq] or None. Returns out
    [sq,hq,d] fp32.
    """
    sq, hq, d = qf8.shape
    sk, hk, _ = kf8.shape
    qd = qf8.float() * q_desc[..., None]  # [sq,hq,d]
    kd = kf8.float() * k_desc[..., None]  # [sk,hk,d]
    vd = vf8.float() * v_desc[None, :, None]  # [sk,hk,d]
    out = torch.empty(sq, hq, d, dtype=torch.float32, device=qf8.device)
    # Bottom-right causal mask (sq==sk here): key k visible to query q if k<=q.
    qidx = torch.arange(sq, device=qf8.device)[:, None]
    kidx = torch.arange(sk, device=qf8.device)[None, :]
    mask = kidx > (qidx + (sk - sq))
    for h in range(hq):
        hk_i = h // gqa
        scores = (qd[:, h] @ kd[:, hk_i].transpose(0, 1)) * scale  # [sq,sk]
        scores = scores.masked_fill(mask, float("-inf"))
        m = scores.amax(dim=-1, keepdim=True)
        p = torch.exp(scores - m)
        if p_scale is not None:
            p = p * p_scale[h]
        denom = p.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        p_deq = p.to(_ASM_FP8).float()  # P quantized to fp8 for PV
        acc = p_deq @ vd[:, hk_i]  # [sq,d]
        out[:, h] = acc / denom
    return out


def _asm_pack_paged_kv(kf8_list, vf8_list, kdesc_list, num_heads_k, head_dim):
    """Scatter per-batch BHSD-style fp8 K/V and K-descale into paged pools."""
    dev = kf8_list[0].device
    page, x = _ASM_PAGE, _ASM_VECX
    ppb = [(kf8.shape[0] + page - 1) // page for kf8 in kf8_list]
    num_pages = sum(ppb)
    kv_indptr = torch.tensor(
        [0, *list(itertools.accumulate(ppb))], dtype=torch.int32, device=dev
    )
    kv_page_indices = torch.arange(
        num_pages, dtype=torch.int32, device=dev
    )  # identity LTD
    seqlens_kvcache = torch.tensor(
        [kf8.shape[0] for kf8 in kf8_list], dtype=torch.int32, device=dev
    )

    k_pool = torch.zeros(
        num_pages, num_heads_k, head_dim // x, page, x, dtype=_ASM_FP8, device=dev
    )
    v_pool = torch.zeros(
        num_pages, num_heads_k, head_dim, page, dtype=_ASM_FP8, device=dev
    )
    kdesc_pool = torch.zeros(
        num_pages, page, num_heads_k, dtype=torch.float32, device=dev
    )

    for b, (kf8, vf8, kdesc) in enumerate(zip(kf8_list, vf8_list, kdesc_list)):
        sk = kf8.shape[0]
        base = kv_indptr[b].item()
        for t in range(sk):
            phys = base + t // page  # identity LTD
            row = t % page
            # K: [phys, hk, d//x, row, d%x]
            kt = kf8[t].view(num_heads_k, head_dim // x, x)  # [hk, d/x, x]
            k_pool[phys, :, :, row, :] = kt
            # V: [phys, hk, d, row]
            v_pool[phys, :, :, row] = vf8[t]  # [hk, d]
            # K-descale: [phys, row, hk]
            kdesc_pool[phys, row, :] = kdesc[t]
    return dict(
        k_pool=k_pool,
        v_pool=v_pool,
        kdesc_pool=kdesc_pool,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        seqlens_kvcache=seqlens_kvcache,
        num_pages=num_pages,
    )


def run_batch_prefill_asm(
    seqlens,
    num_qo_heads=8,
    num_kv_heads=1,
    head_dim=128,
    use_p_scale=True,
    seed=0,
    bench=False,
    warmup=10,
    iters=50,
):
    """Drive mha_batch_prefill_asm and validate against the fp32 reference."""
    torch.manual_seed(seed)
    dev = "cuda"
    batch = len(seqlens)
    gqa = num_qo_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    q_packed, qdesc_packed, out_ref_packed = [], [], []
    kf8_list, vf8_list, kdesc_list = [], [], []
    v_desc = (torch.rand(num_kv_heads, device=dev) * 0.5 + 0.5).to(torch.float32)
    p_scale = (
        (torch.rand(num_qo_heads, device=dev) * 0.5 + 0.75).to(torch.float32)
        if use_p_scale
        else None
    )

    for s in seqlens:
        q = torch.randn(s, num_qo_heads, head_dim, device=dev)
        k = torch.randn(s, num_kv_heads, head_dim, device=dev)
        v = torch.randn(s, num_kv_heads, head_dim, device=dev)
        qf8, qdesc = _asm_quant_per_row(q)  # [s,hq,d],[s,hq]
        kf8, kdesc = _asm_quant_per_row(k)  # [s,hk,d],[s,hk]
        # V descale is per-kv-head only (launcher v_descale [hk]).
        vf8 = (v / v_desc[None, :, None]).to(_ASM_FP8)
        q_packed.append(qf8)
        qdesc_packed.append(qdesc)
        kf8_list.append(kf8)
        vf8_list.append(vf8)
        kdesc_list.append(kdesc)
        out_ref_packed.append(
            _asm_reference(qf8, kf8, vf8, qdesc, kdesc, v_desc, p_scale, scale, gqa)
        )

    q_packed = torch.cat(q_packed, dim=0).contiguous()  # [total_q,hq,d]
    qdesc_packed = torch.cat(qdesc_packed, dim=0).contiguous()  # [total_q,hq]
    out_ref = torch.cat(out_ref_packed, dim=0)  # [total_q,hq,d]
    cu_seqlens_q = torch.tensor(
        [0, *list(itertools.accumulate(seqlens))], dtype=torch.int32, device=dev
    )
    paged = _asm_pack_paged_kv(kf8_list, vf8_list, kdesc_list, num_kv_heads, head_dim)

    out = torch.empty_like(q_packed, dtype=torch.bfloat16)

    def _launch():
        mha_batch_prefill_asm(
            q_packed,
            paged["k_pool"],
            paged["v_pool"],
            cu_seqlens_q,
            paged["kv_indptr"],
            paged["kv_page_indices"],
            paged["seqlens_kvcache"],
            out,
            qdesc_packed,
            paged["kdesc_pool"],
            v_desc,
            batch,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            head_dim,
            _ASM_PAGE,
            paged["num_pages"],
            max(seqlens),
            scale,
            p_scale=p_scale,
        )

    _launch()

    out_f = out.float()
    diff = out_f - out_ref
    nrms = (diff.pow(2).sum() / out_ref.pow(2).sum().clamp(min=1e-20)).sqrt().item()
    max_abs = diff.abs().max().item()

    latency_ms = None
    tflops = None
    if bench:
        for _ in range(warmup):
            _launch()
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            _launch()
        end.record()
        torch.cuda.synchronize()
        latency_ms = start.elapsed_time(end) / iters
        # Causal FLOPs: per batch sum_q (q+1) score pairs * 2*(d_qk+d_v) (QK + PV).
        score_pairs = sum(s * (s + 1) / 2 for s in seqlens)
        flops = num_qo_heads * score_pairs * (head_dim + head_dim) * 2.0
        tflops = flops / (latency_ms * 1e-3) / 1e12

    msg = (
        f"[asm] seqlens={seqlens} hq={num_qo_heads} hk={num_kv_heads} "
        f"p_scale={use_p_scale} NRMS={nrms:.4e} max_abs={max_abs:.4e}"
    )
    if bench:
        msg += f" | latency={latency_ms:.4f} ms  {tflops:.1f} TFLOPS"
    print(msg)
    assert nrms < 0.06, f"asm kernel vs reference NRMS too high: {nrms:.4e}"
    return dict(nrms=nrms, max_abs=max_abs, latency_ms=latency_ms, tflops=tflops)


@pytest.mark.skipif(
    get_gpu_arch() != "gfx942", reason="asm qkptph/vph kernel is gfx942-only"
)
@pytest.mark.parametrize("use_p_scale", [False, True])
@pytest.mark.parametrize("seqlen", [256, 512])
def test_batch_prefill_asm_batched(seqlen, use_p_scale):
    # Batched / b=1 path (varlen kernel with a single [0, S] segment).
    run_batch_prefill_asm([seqlen], use_p_scale=use_p_scale, seed=11)


@pytest.mark.skipif(
    get_gpu_arch() != "gfx942", reason="asm qkptph/vph kernel is gfx942-only"
)
@pytest.mark.parametrize("use_p_scale", [False, True])
@pytest.mark.parametrize("seqlens", [[256, 512], [512, 256, 384]])
def test_batch_prefill_asm_varlen(seqlens, use_p_scale):
    # Varlen: distinct per-batch sequence lengths.
    run_batch_prefill_asm(seqlens, use_p_scale=use_p_scale, seed=23)


# Perf sweep: b=1 latency/TFLOPS at increasing context, reporting NRMS + runtime.
# Opt-in (it is a benchmark, not a correctness gate) via AITER_ASM_PERF=1.
@pytest.mark.skipif(
    get_gpu_arch() != "gfx942", reason="asm qkptph/vph kernel is gfx942-only"
)
@pytest.mark.skipif(
    os.environ.get("AITER_ASM_PERF") != "1",
    reason="perf sweep; set AITER_ASM_PERF=1 to run",
)
@pytest.mark.parametrize("seqlen", [512, 1024, 2048, 4096, 8192, 16384, 32768])
def test_batch_prefill_asm_perf(seqlen):
    run_batch_prefill_asm([seqlen], use_p_scale=True, seed=31, bench=True)


if __name__ == "__main__" and get_gpu_arch() == "gfx942":
    if os.environ.get("AITER_ASM_PERF") == "1":
        # Perf sweep: b=1 latency/TFLOPS (+ NRMS) at increasing context.
        for s in (512, 1024, 2048, 4096, 8192, 16384, 32768):
            run_batch_prefill_asm([s], use_p_scale=True, seed=31, bench=True)
    elif os.environ.get("AITER_ASM_TEST") == "1":
        for sl in ([512], [256, 512], [512, 256, 384]):
            for ps in (False, True):
                run_batch_prefill_asm(sl, use_p_scale=ps, seed=7)

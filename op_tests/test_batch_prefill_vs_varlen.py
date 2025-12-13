# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Test to compare batch_prefill (paged KV cache) with flash_attn_varlen (varlen format).
Both should produce identical results since they use the same qr_async pipeline.
"""

import torch
import pytest
import aiter
from aiter import dtypes
from aiter import per_tensor_quant


def convert_lens_to_indptr(lens):
    """Convert sequence lengths to cumulative index pointer."""
    return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()


def create_tensor(min_val, max_val, *args, **kwargs):
    """Create a tensor with values uniformly distributed in [min_val, max_val]."""
    x = torch.randn(*args, **kwargs)
    x = (x - x.min()) / (x.max() - x.min())
    return min_val + (max_val - min_val) * x


def varlen_to_paged_kv(k_varlen, v_varlen, kv_lens, page_size=1):
    """
    Convert varlen format K/V to paged KV cache format.

    Args:
        k_varlen: [total_tokens, num_kv_heads, head_dim]
        v_varlen: [total_tokens, num_kv_heads, head_dim]
        kv_lens: [batch_size] - length of each sequence
        page_size: tokens per page

    Returns:
        kv_data: [total_num_pages, 2, num_kv_heads, page_size, head_dim]
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
        num_kv_heads,
        page_size,
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
            kv_data[global_page_idx, 0, :, :tokens_in_page, :] = k_varlen[
                seq_start + token_start : seq_start + token_end, :, :
            ]

            # V data
            kv_data[global_page_idx, 1, :, :tokens_in_page, :] = v_varlen[
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
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_batch_prefill_vs_varlen_bf16(
    batch_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    qo_len,
    kv_len,
    causal,
    logits_soft_cap,
    dtype,
):
    """
    Compare BF16 batch_prefill (paged KV) vs BF16 flash_attn_varlen.
    Both use qr_async pipeline, should produce identical results.
    """
    torch.manual_seed(42)

    # Create Q, K, V in varlen format
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

    q = create_tensor(
        -10, 10, total_q_tokens, num_qo_heads, head_dim, dtype=dtype
    ).cuda()
    k = create_tensor(
        -5, 5, total_kv_tokens, num_kv_heads, head_dim, dtype=dtype
    ).cuda()
    v = create_tensor(
        -5, 5, total_kv_tokens, num_kv_heads, head_dim, dtype=dtype
    ).cuda()

    cu_seqlens_q = convert_lens_to_indptr(qo_lens).cuda()
    cu_seqlens_k = convert_lens_to_indptr(kv_lens).cuda()

    # Run flash_attn_varlen
    out_varlen = aiter.flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q=qo_lens.max().item(),
        max_seqlen_k=kv_lens.max().item(),
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        return_lse=False,
    )

    # Convert to paged KV cache format
    kv_data, kv_indptr, kv_indices = varlen_to_paged_kv(k, v, kv_lens, page_size=1)

    # Extract K and V from paged format
    k_paged = kv_data[:, 0, :, :, :].squeeze(2)  # [num_pages, num_kv_heads, head_dim]
    v_paged = kv_data[:, 1, :, :, :].squeeze(2)  # [num_pages, num_kv_heads, head_dim]

    # Run batch_prefill
    out_batch_prefill = aiter.mha_batch_prefill_func(
        q,
        k_paged,
        v_paged,
        cu_seqlens_q,
        kv_indptr.cuda(),
        kv_indices.cuda(),
        max_seqlen_q=qo_lens.max().item(),
        max_seqlen_k=kv_lens.max().item(),
        causal=causal,
        logits_soft_cap=logits_soft_cap,
    )

    # Compare results
    print(f"\n=== BF16 Comparison ===")
    print(
        f"batch_size={batch_size}, heads={num_qo_heads}/{num_kv_heads}, "
        f"dim={head_dim}, qo_len={qo_len}, kv_len={kv_len}"
    )
    print(f"causal={causal}, logits_soft_cap={logits_soft_cap}")

    diff = (out_varlen - out_batch_prefill).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max diff: {max_diff:.6e}")
    print(f"Mean diff: {mean_diff:.6e}")

    if out_varlen.abs().max().item() > 0:
        rel_error = max_diff / out_varlen.abs().max().item()
        print(f"Relative error: {rel_error * 100:.4f}%")

    # Should be nearly identical (same pipeline, same computation)
    rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-2, 1e-2)
    torch.testing.assert_close(out_batch_prefill, out_varlen, rtol=rtol, atol=atol)


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

    q_bf16 = create_tensor(
        -10, 10, total_q_tokens, num_qo_heads, head_dim, dtype=dtype
    ).cuda()
    k_bf16 = create_tensor(
        -5, 5, total_kv_tokens, num_kv_heads, head_dim, dtype=dtype
    ).cuda()
    v_bf16 = create_tensor(
        -5, 5, total_kv_tokens, num_kv_heads, head_dim, dtype=dtype
    ).cuda()

    # Quantize to FP8
    scale = torch.tensor([1.0], dtype=torch.float32).cuda()
    q_fp8, q_descale = per_tensor_quant(q_bf16, scale, quant_dtype=quant_dtype)
    k_fp8, k_descale = per_tensor_quant(k_bf16, scale, quant_dtype=quant_dtype)
    v_fp8, v_descale = per_tensor_quant(v_bf16, scale, quant_dtype=quant_dtype)

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
        window_size=(-1, -1),
    )

    # Convert to paged KV cache format
    kv_data, kv_indptr, kv_indices = varlen_to_paged_kv(
        k_fp8, v_fp8, kv_lens, page_size=1
    )

    # Extract K and V from paged format
    k_paged = kv_data[:, 0, :, :, :].squeeze(2)
    v_paged = kv_data[:, 1, :, :, :].squeeze(2)

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
    )

    # Compare results
    print(f"\n=== FP8 Comparison ===")
    print(
        f"batch_size={batch_size}, heads={num_qo_heads}/{num_kv_heads}, "
        f"dim={head_dim}, qo_len={qo_len}, kv_len={kv_len}"
    )
    print(f"causal={causal}, logits_soft_cap={logits_soft_cap}")

    diff = (out_varlen - out_batch_prefill).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"Max diff: {max_diff:.6e}")
    print(f"Mean diff: {mean_diff:.6e}")

    if out_varlen.abs().max().item() > 0:
        rel_error = max_diff / out_varlen.abs().max().item()
        print(f"Relative error: {rel_error * 100:.4f}%")

    # Should be nearly identical (same pipeline, same computation)
    # FP8 may have slightly larger tolerance
    torch.testing.assert_close(out_batch_prefill, out_varlen, rtol=5e-2, atol=5e-2)


if __name__ == "__main__":
    # Run BF16 tests
    print("=" * 80)
    print("Testing BF16: batch_prefill vs flash_attn_varlen")
    print("=" * 80)
    for causal in [False, True]:
        for logits_soft_cap in [0.0, 30.0]:
            test_batch_prefill_vs_varlen_bf16(
                batch_size=1,
                num_qo_heads=6,
                num_kv_heads=1,
                head_dim=128,
                qo_len=128,
                kv_len=128,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
                dtype=torch.bfloat16,
            )

    # Run FP8 tests
    print("\n" + "=" * 80)
    print("Testing FP8: batch_prefill vs flash_attn_varlen")
    print("=" * 80)
    for causal in [False, True]:
        for logits_soft_cap in [0.0, 30.0]:
            test_batch_prefill_vs_varlen_fp8(
                batch_size=1,
                num_qo_heads=8,
                num_kv_heads=1,
                head_dim=128,
                qo_len=128,
                kv_len=128,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
            )

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)

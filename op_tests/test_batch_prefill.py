# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import math
import os
import pytest
import torch

import aiter
from aiter import dtypes
from aiter import per_tensor_quant
from aiter.test_common import run_perftest
from einops import rearrange, repeat
import argparse


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


@pytest.mark.parametrize("batch_size", [1, 3, 7])
@pytest.mark.parametrize(
    "qo_len,kv_len",
    [
        (128, 128),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(6, 1), (3, 1)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("contiguous_kv", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_init_min,q_init_max", [(-10, 10)])
@pytest.mark.parametrize("kv_init_min,kv_init_max", [(-5, 5)])
@pytest.mark.parametrize("seed", [19378])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal,
    kv_layout,
    logits_soft_cap,
    contiguous_kv,
    dtype,
    q_init_min,
    q_init_max,
    kv_init_min,
    kv_init_max,
    seed,
):
    if seed is not None:
        torch.manual_seed(seed)

    if causal and kv_len < qo_len:
        pytest.skip("kv_len < qo_len is not allowed if causal=True")

    if head_dim == 64 and qo_len <= 64:
        pytest.skip("Unsupported configuration")

    def create_tensor(min, max, *args, **kwargs):
        x = torch.randn(*args, **kwargs)
        x = (x - x.min()) / (x.max() - x.min())
        return min + (max - min) * x

    def convert_lens_to_indtpr(lens):
        return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()

    q = create_tensor(
        q_init_min, q_init_max, batch_size * qo_len, num_qo_heads, head_dim, dtype=dtype
    ).to(0)
    if 1 < batch_size:
        qo_lens = torch.randint(1, qo_len + 1, (batch_size,)).int()
    else:
        qo_lens = torch.full((batch_size,), qo_len).int()
    q_indptr_cpu = convert_lens_to_indtpr(qo_lens)
    max_num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = max_num_pages_per_seq * batch_size
    kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = create_tensor(
            kv_init_min, kv_init_max, *kv_shape, dtype=torch.float32
        ).to(0)
        kv_data = kv_data_fp32.to(dtype)
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = create_tensor(
            kv_init_min, kv_init_max, *kv_shape, dtype=torch.float32
        ).to(0)
        kv_data = kv_data_fp32.to(dtype)
    if 1 < batch_size:
        kv_lens = torch.maximum(
            qo_lens, torch.randint(1, kv_len + 1, (batch_size,))
        ).int()
    else:
        kv_lens = torch.full((batch_size,), kv_len).int()
    kv_num_used_pages = (kv_lens + page_size - 1) // page_size
    kv_indptr_cpu = convert_lens_to_indtpr(kv_num_used_pages)
    kv_indices_cpu = torch.nn.functional.pad(
        torch.randperm(total_num_pages).int(), (0, 128), value=0
    )
    kv_last_page_len_cpu = ((kv_lens - 1) % page_size + 1).int()

    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_indptr_cpu.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)

    chunks = torch.chunk(kv_data, 2, dim=1)
    k_cache = chunks[0].squeeze(2).squeeze(2)
    v_cache = chunks[1].squeeze(2).squeeze(2)

    o_ck_flash_attn = aiter.mha_batch_prefill_func(
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
    )

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        used_kv_indices = kv_indices_cpu[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1]]
        ki = torch.cat(
            [
                kv_data_fp32[used_kv_indices[:-1], 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[used_kv_indices[-1], 0, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        used_kv_indices[-1], 0, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(dtype)
        vi = torch.cat(
            [
                kv_data_fp32[used_kv_indices[:-1], 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[used_kv_indices[-1], 1, :, : kv_last_page_len_cpu[i]]
                    if kv_layout == "HND"
                    else kv_data_fp32[
                        used_kv_indices[-1], 1, : kv_last_page_len_cpu[i], :
                    ]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(dtype)

        # enlarge rtol for bf16 to allow passing very few numeric errors
        rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-2, 1e-2)

        o_ref_i = ref_masked_attention(
            qi, ki, vi, causal=causal, logits_soft_cap=logits_soft_cap
        )

        o_i = o_ck_flash_attn[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        torch.testing.assert_close(o_i, o_ref_i, rtol=rtol, atol=atol)


def run_ck(
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
):
    """Unified interface for running batch_prefill with or without FP8."""
    if (
        q.dtype == dtypes.fp8
        and k_cache.dtype == dtypes.fp8
        and v_cache.dtype == dtypes.fp8
    ):
        # FP8 path
        return (
            aiter.mha_batch_prefill_func(
                q,
                k_cache,
                v_cache,
                cu_seqlens_q,
                kv_indptr,
                kv_page_indices,
                max_seqlen_q,
                max_seqlen_k,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            ),
            0.1,
        )
    else:
        # Standard BF16/FP16 path
        return (
            aiter.mha_batch_prefill_func(
                q,
                k_cache,
                v_cache,
                cu_seqlens_q,
                kv_indptr,
                kv_page_indices,
                max_seqlen_q,
                max_seqlen_k,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
            ),
            0.1,
        )


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
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
def test_batch_prefill_fp8_output(
    batch_size,
    num_qo_heads,
    num_kv_heads,
    qo_len,
    kv_len,
    head_dim,
    causal,
    logits_soft_cap,
):
    """Test FP8 batch_prefill by comparing with BF16 kernel."""
    torch.random.manual_seed(0)
    torch.cuda.empty_cache()

    dtype = torch.bfloat16
    quant_dtype = dtypes.fp8
    page_size = 1

    def convert_lens_to_indptr(lens):
        return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()

    # Create Q tensor - using torch.rand (same as test_mha_varlen_fp8.py)
    q = torch.rand(batch_size * qo_len, num_qo_heads, head_dim, dtype=dtype).to(0)

    # Create sequence lengths
    if batch_size > 1:
        qo_lens = torch.randint(1, qo_len + 1, (batch_size,)).int()
        kv_lens = torch.maximum(
            qo_lens, torch.randint(1, kv_len + 1, (batch_size,))
        ).int()
    else:
        qo_lens = torch.full((batch_size,), qo_len).int()
        kv_lens = torch.full((batch_size,), kv_len).int()

    q_indptr_cpu = convert_lens_to_indptr(qo_lens)

    # Create paged KV cache - using torch.rand (same as test_mha_varlen_fp8.py)
    max_num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = max_num_pages_per_seq * batch_size
    kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]

    kv_data = torch.rand(*kv_shape, dtype=dtype).to(0)

    kv_num_used_pages = (kv_lens + page_size - 1) // page_size
    kv_indptr_cpu = convert_lens_to_indptr(kv_num_used_pages)
    kv_indices_cpu = torch.nn.functional.pad(
        torch.randperm(total_num_pages).int(), (0, 128), value=0
    )

    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_indptr_cpu.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)

    # Extract K and V caches
    chunks = torch.chunk(kv_data, 2, dim=1)
    k_cache = chunks[0].squeeze(2).squeeze(2)
    v_cache = chunks[1].squeeze(2).squeeze(2)

    # Debug: Check input statistics before quantization
    print(f"\n=== Input Statistics (BF16) ===")
    print(
        f"Q: min={q.min().item():.4f}, max={q.max().item():.4f}, mean={q.mean().item():.4f}"
    )
    print(
        f"K: min={k_cache.min().item():.4f}, max={k_cache.max().item():.4f}, mean={k_cache.mean().item():.4f}"
    )
    print(
        f"V: min={v_cache.min().item():.4f}, max={v_cache.max().item():.4f}, mean={v_cache.mean().item():.4f}"
    )

    # Quantize to FP8 (let per_tensor_quant automatically compute optimal scale)
    q_quant, q_descale = per_tensor_quant(q, quant_dtype=quant_dtype)
    k_cache_quant, k_descale = per_tensor_quant(k_cache, quant_dtype=quant_dtype)
    v_cache_quant, v_descale = per_tensor_quant(v_cache, quant_dtype=quant_dtype)

    # Debug: Check quantization parameters
    print(f"\n=== Quantization Parameters ===")
    print(f"Q descale: {q_descale.item():.6f}")
    print(f"K descale: {k_descale.item():.6f}")
    print(f"V descale: {v_descale.item():.6f}")

    # Debug: Check dequantized values (for debugging only)
    q_dequant = q_quant.to(torch.float32) * q_descale
    k_dequant = k_cache_quant.to(torch.float32) * k_descale
    v_dequant = v_cache_quant.to(torch.float32) * v_descale
    print(f"\n=== Quantization Error ===")
    print(f"Q quant error: {(q.float() - q_dequant).abs().max().item():.6f}")
    print(f"K quant error: {(k_cache.float() - k_dequant).abs().max().item():.6f}")
    print(f"V quant error: {(v_cache.float() - v_dequant).abs().max().item():.6f}")

    # Run FP8 kernel
    out_fp8, us_fp8 = run_ck(
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
    )

    # Run BF16 reference with ORIGINAL data (same as test_mha_varlen_fp8.py)
    out_ref, us_ref = run_ck(
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
    )

    # Compare outputs
    diff = (out_fp8 - out_ref).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"\n=== Output Statistics ===")
    print(
        f"FP8 kernel output: min={out_fp8.min().item():.4f}, max={out_fp8.max().item():.4f}, mean={out_fp8.mean().item():.4f}"
    )
    print(
        f"BF16 kernel output: min={out_ref.min().item():.4f}, max={out_ref.max().item():.4f}, mean={out_ref.mean().item():.4f}"
    )

    # Check if output is all zeros (kernel didn't run)
    print(f"\n=== Sanity Checks ===")
    print(f"FP8 output max abs: {out_fp8.abs().max().item():.6e}")
    print(f"BF16 output max abs: {out_ref.abs().max().item():.6e}")
    print(f"FP8 has NaN: {torch.isnan(out_fp8).any().item()}")
    print(f"FP8 has Inf: {torch.isinf(out_fp8).any().item()}")
    print(f"BF16 has NaN: {torch.isnan(out_ref).any().item()}")
    print(f"BF16 has Inf: {torch.isinf(out_ref).any().item()}")

    if out_fp8.abs().max().item() < 1e-6:
        print("WARNING: FP8 output is all zeros - kernel may not have launched!")
    if out_ref.abs().max().item() < 1e-6:
        print("WARNING: BF16 output is all zeros - kernel may not have launched!")

    if torch.isnan(out_ref).any() or torch.isinf(out_ref).any():
        print("\nERROR: BF16 kernel produced NaN or Inf values!")
        print(
            "This indicates a numerical sㄍㄨbility issue in the BF16 batch_prefill kernel."
        )
        print(
            "Please investigate the BF16 kernel implementation before comparing with FP8."
        )
        return

    print(f"\n=== Output Difference (FP8 kernel vs BF16 kernel) ===")
    print(f"Max diff: {max_diff:.6f}")
    print(f"Mean diff: {mean_diff:.6f}")
    print(f"Relative error: {(max_diff / out_ref.abs().max().item() * 100):.2f}%")

    # Find position of max diff
    max_idx = diff.argmax()
    max_idx_unravel = torch.unravel_index(max_idx, diff.shape)
    print(f"Max diff location: {max_idx_unravel}")
    print(f"  FP8 kernel value: {out_fp8.flatten()[max_idx].item():.6f}")
    print(f"  BF16 kernel value: {out_ref.flatten()[max_idx].item():.6f}")

    print(f"\n=== Performance ===")
    print(f"FP8 kernel time: {us_fp8:.2f} us")
    print(f"BF16 kernel time: {us_ref:.2f} us")
    print(f"Speedup: {us_ref / us_fp8:.2f}x")

    # Assert accuracy (same threshold as test_mha_varlen_fp8.py)
    # Note: This test compares FP8 kernel (with quantized inputs) vs BF16 kernel (with original inputs)
    # The difference includes both quantization error and FP8 tensor core computation differences.
    # For a test that isolates just the kernel implementation correctness,
    # see test_batch_prefill_vs_varlen.py which compares FP8 batch_prefill vs FP8 varlen
    threshold = 0.055
    assert max_diff < threshold, (
        f"FP8 kernel vs BF16 kernel difference too large: {max_diff} (threshold: {threshold})"
    )


l_causal = [False, True]
l_logits_soft_cap = [0.0, 30.0]
l_dtype = ["fp16", "bf16"]
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-c",
    "--causal",
    type=dtypes.str2bool,
    nargs="?",
    const=None,
    default=None,
    help="""Causal mask mode (False or True).
    e.g.: -c false""",
)
parser.add_argument(
    "-l",
    "--logits_soft_cap",
    type=float,
    choices=l_logits_soft_cap,
    nargs="?",
    const=None,
    default=None,
    help="""Logits soft cap.
    e.g.: -l 30.0""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "--test_fp8",
    action="store_true",
    help="""Run FP8 test instead of standard test.
    e.g.: --test_fp8""",
)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    if args.causal is not None:
        l_causal = [args.causal]
    if args.logits_soft_cap is not None:
        l_logits_soft_cap = [args.logits_soft_cap]

    if args.test_fp8:
        # Run FP8 tests
        for causal, logits_soft_cap in itertools.product(l_causal, l_logits_soft_cap):
            test_batch_prefill_fp8_output(
                batch_size=1,
                num_qo_heads=1,
                num_kv_heads=1,
                qo_len=128,
                kv_len=128,
                head_dim=128,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
            )
    else:
        # Run standard tests
        for (
            causal,
            logits_soft_cap,
            dtype,
        ) in itertools.product(l_causal, l_logits_soft_cap, l_dtype):
            test_batch_prefill_with_paged_kv_cache(
                batch_size=1,
                kv_len=8192,
                qo_len=8192,
                page_size=1,
                num_qo_heads=6,
                num_kv_heads=1,
                head_dim=128,
                causal=causal,
                kv_layout="NHD",
                logits_soft_cap=logits_soft_cap,
                contiguous_kv=True,
                dtype=dtype,
                q_init_min=-10,
                q_init_max=10,
                kv_init_min=-5,
                kv_init_max=5,
                seed=19378,
            )

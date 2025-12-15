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


@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(6, 1), (3, 1)])
@pytest.mark.parametrize("head_dim", [128])
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
@pytest.mark.parametrize("seed", [19378])
def test_batch_prefill_fp8_output(
    batch_size,
    num_qo_heads,
    num_kv_heads,
    qo_len,
    kv_len,
    head_dim,
    page_size,
    causal,
    logits_soft_cap,
    seed,
):
    """Test FP8 batch_prefill by comparing with BF16 kernel, following test_mha_varlen_fp8 pattern."""
    if seed is not None:
        torch.manual_seed(seed)

    torch.cuda.empty_cache()

    if causal and kv_len < qo_len:
        pytest.skip("kv_len < qo_len is not allowed if causal=True")

    if head_dim == 64 and qo_len <= 64:
        pytest.skip("Unsupported configuration")

    dtype = torch.bfloat16
    quant_dtype = dtypes.fp8

    def convert_lens_to_indptr(lens):
        return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()

    # Create Q tensor following test_mha_varlen_fp8 pattern (using torch.rand for [0, 1) range)
    q = torch.rand(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda", dtype=dtype
    )

    # Create variable sequence lengths
    if batch_size > 1:
        qo_lens = torch.randint(1, qo_len + 1, (batch_size,)).int()
    else:
        qo_lens = torch.full((batch_size,), qo_len).int()
    q_indptr_cpu = convert_lens_to_indptr(qo_lens)

    # Create paged KV cache following test_batch_prefill_with_paged_kv_cache pattern
    # Generate in FP32 first for accurate reference computation
    max_num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = max_num_pages_per_seq * batch_size
    kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]

    kv_data_fp32 = torch.rand(*kv_shape, device="cuda", dtype=torch.float32)
    kv_data = kv_data_fp32.to(dtype)

    if batch_size > 1:
        kv_lens = torch.maximum(
            qo_lens, torch.randint(1, kv_len + 1, (batch_size,))
        ).int()
    else:
        kv_lens = torch.full((batch_size,), kv_len).int()

    kv_num_used_pages = (kv_lens + page_size - 1) // page_size
    kv_indptr_cpu = convert_lens_to_indptr(kv_num_used_pages)
    kv_indices_cpu = torch.nn.functional.pad(
        torch.randperm(total_num_pages).int(), (0, 128), value=0
    )
    kv_last_page_len_cpu = ((kv_lens - 1) % page_size + 1).int()

    q_indptr_gpu = q_indptr_cpu.to(0)
    kv_indptr_gpu = kv_indptr_cpu.to(0)
    kv_indices_gpu = kv_indices_cpu.to(0)

    # Extract K and V caches
    chunks = torch.chunk(kv_data, 2, dim=1)
    k_cache = chunks[0].squeeze(2).squeeze(2)
    v_cache = chunks[1].squeeze(2).squeeze(2)

    # Quantize to FP8 following test_mha_varlen_fp8 pattern
    q_quant, q_descale = per_tensor_quant(q, quant_dtype=quant_dtype)
    k_cache_quant, k_descale = per_tensor_quant(k_cache, quant_dtype=quant_dtype)
    v_cache_quant, v_descale = per_tensor_quant(v_cache, quant_dtype=quant_dtype)

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

    # Run BF16 reference kernel
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

    # Compare outputs per-sequence (following test_batch_prefill_with_paged_kv_cache pattern)
    # This ensures we only check valid tokens and avoid false-positive NaN detection
    rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (2e-2, 1e-2)
    kv_layout = "NHD"

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]

        # Extract valid Q for this sequence
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

        # Extract valid K and V for this sequence
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

        # Compute reference attention for this sequence
        o_ref_i = ref_masked_attention(
            qi, ki, vi, causal=causal, logits_soft_cap=logits_soft_cap
        )

        # Extract outputs for this sequence (only valid tokens)
        o_fp8_i = out_fp8[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]
        o_bf16_i = out_ref[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

        # Compare FP8 output with reference
        # Following test_mha_varlen_fp8 threshold
        max_diff = (o_fp8_i - o_ref_i).abs().max().item()
        threshold = 0.055
        assert max_diff < threshold, (
            f"Sequence {i}: FP8 kernel vs reference difference too large: "
            f"{max_diff} (threshold: {threshold})"
        )

        # Also verify BF16 kernel matches reference (sanity check)
        torch.testing.assert_close(o_bf16_i, o_ref_i, rtol=rtol, atol=atol)


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
                qo_len=8192,
                kv_len=8192,
                page_size=1,
                num_qo_heads=6,
                num_kv_heads=1,
                head_dim=128,
                causal=causal,
                logits_soft_cap=logits_soft_cap,
                seed=19378,
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

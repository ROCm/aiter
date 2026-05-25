# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import random
import argparse

import pytest
import torch

from aiter.test_common import checkAllclose, run_perftest
from aiter.ops.triton.gather_kv_b_proj import gather_kv_b_proj
from aiter.ops.shuffle import shuffle_weight
from aiter import dtypes


def ref_gather_kv_b_proj(
    k_buffer: torch.Tensor,  # [num_block, block_size, hidden_dim]
    k_scale: torch.Tensor,  # [1]
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indices: torch.Tensor,  # len(kv_indices) = kv_indptr[-1]
    kv_prefix_sum_context_lens: torch.Tensor,  # [batch_size + 1]
    kv_proj_weight: torch.Tensor,  # [tp_heads * (qk_nope_head_dim + v_head_dim), kv_c_dim]
    kv_proj_scale: torch.Tensor,  # [weight_n] per-row or [N//128, K//128] block
    qk_nope_head_dim: int = 128,
    v_head_dim: int = None,
):
    if v_head_dim is None:
        v_head_dim = qk_nope_head_dim

    batch_size = kv_indptr.shape[0] - 1

    kv_c_dim = 512
    kv_pe_dim = 64

    num_block, block_size, hidden_dim = k_buffer.shape
    weight_n, weight_k = kv_proj_weight.shape
    per_row_scale = kv_proj_scale.dim() == 1 or (
        kv_proj_scale.dim() == 2 and kv_proj_scale.shape[1] == 1
    )

    assert hidden_dim == kv_c_dim + kv_pe_dim
    assert weight_k == kv_c_dim
    if per_row_scale:
        assert kv_proj_scale.numel() == weight_n
    else:
        scale_granularity_n = weight_n // kv_proj_scale.shape[0]
        scale_granularity_k = weight_k // kv_proj_scale.shape[1]
        assert scale_granularity_k == 128

    tp_k_head_num = weight_n // (qk_nope_head_dim + v_head_dim)

    kv_c, k_pe = k_buffer.split(
        [kv_c_dim, kv_pe_dim], dim=-1
    )  # [num_block, block_size, C_dim / Pe_dim]

    total_kv = kv_prefix_sum_context_lens[-1].item()
    k_prefix = torch.zeros(
        (total_kv, tp_k_head_num * (qk_nope_head_dim + kv_pe_dim)),
        device=k_buffer.device,
        dtype=torch.bfloat16,
    )
    v_prefix = torch.zeros(
        (total_kv, tp_k_head_num * v_head_dim),
        device=k_buffer.device,
        dtype=torch.bfloat16,
    )
    k_prefix_tp = k_prefix.view(total_kv, tp_k_head_num, qk_nope_head_dim + kv_pe_dim)
    v_prefix_tp = v_prefix.view(total_kv, tp_k_head_num, v_head_dim)

    if not per_row_scale:
        kv_proj_scale_repeat = kv_proj_scale.repeat_interleave(
            scale_granularity_n, dim=0
        )

    kv_indptr_list = kv_indptr.tolist()
    for b in range(batch_size):
        kv_indice_start = kv_indptr_list[b]
        kv_indice_end = kv_indptr_list[b + 1]

        context_start = kv_prefix_sum_context_lens[b].item()
        context_end = kv_prefix_sum_context_lens[b + 1].item()

        # broadcast k_pe to all tp
        k_prefix_block = k_pe[kv_indices[kv_indice_start:kv_indice_end], :, :].reshape(
            -1, kv_pe_dim
        )
        k_prefix_tp[context_start:context_end, :, qk_nope_head_dim:] = (
            k_prefix_block[: context_end - context_start, :]
            .unsqueeze(1)
            .broadcast_to(-1, tp_k_head_num, kv_pe_dim)
        )
        if k_buffer.dtype != torch.bfloat16:
            k_prefix_tp[
                context_start:context_end, :, qk_nope_head_dim:
            ] *= k_scale.unsqueeze(0).unsqueeze(1)

        k_data = kv_c[kv_indices[kv_indice_start:kv_indice_end], :, :].reshape(
            -1, kv_c_dim
        )[: context_end - context_start, :]

        if per_row_scale:
            kv_proj = (
                k_data.to(torch.float32) @ kv_proj_weight.to(torch.float32).T
            ) * (kv_proj_scale.to(torch.float32).reshape(1, -1))
        else:
            kv_proj = torch.zeros(
                (context_end - context_start, weight_n),
                device=k_buffer.device,
                dtype=torch.float32,
            )
            for i in range(weight_k // scale_granularity_k):
                kv_proj_tmp = (
                    k_data[
                        :, i * scale_granularity_k : (i + 1) * scale_granularity_k
                    ].to(torch.float32)
                    @ kv_proj_weight[
                        :, i * scale_granularity_k : (i + 1) * scale_granularity_k
                    ]
                    .to(torch.float32)
                    .T
                )
                kv_proj += kv_proj_tmp * kv_proj_scale_repeat[:, i].unsqueeze(0)

        kv_proj_tp = kv_proj.view(
            context_end - context_start, tp_k_head_num, qk_nope_head_dim + v_head_dim
        )

        if k_buffer.dtype != torch.bfloat16:
            kv_proj_tp *= k_scale.unsqueeze(0).unsqueeze(1)

        k_proj_tp, v_proj_tp = kv_proj_tp.split([qk_nope_head_dim, v_head_dim], dim=-1)

        k_prefix_tp[
            context_start:context_end,
            :,
            :qk_nope_head_dim,
        ] = k_proj_tp
        v_prefix_tp[context_start:context_end, :] = v_proj_tp

    return (k_prefix, v_prefix)


def _make_kv_test_data(
    batch_size,
    block_size,
    avg_kv_length,
    kv_c_dim,
    kv_pe_dim,
    k_buffer_type,
    device="cuda",
):
    """Create common test data: k_buffer, k_scale, kv_indptr, kv_indices, etc."""
    num_block = 2 * avg_kv_length // block_size

    k_buffer = torch.randn(
        (num_block, block_size, kv_c_dim + kv_pe_dim),
        device=device,
        dtype=torch.float32,
    ).to(k_buffer_type)
    k_scale = torch.randn(1, device=device, dtype=torch.float32).abs()

    var_ratio = 0.2
    context_lens = (
        torch.randint(
            int((1 - var_ratio) * avg_kv_length),
            int(((1 + var_ratio)) * avg_kv_length) + 1,
            (batch_size,),
        )
        .cuda()
        .to(torch.int32)
    )
    context_blocks = torch.div(
        context_lens + block_size - 1, block_size, rounding_mode="trunc"
    )

    kv_indptr = torch.zeros((batch_size + 1,), device="cuda", dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(context_blocks, dim=0)

    kv_prefix_sum_context_lens = torch.zeros(
        (batch_size + 1,), device="cuda", dtype=torch.int32
    )
    kv_prefix_sum_context_lens[1:] = torch.cumsum(context_lens, dim=0)

    kv_indices = torch.zeros(kv_indptr[-1], device="cuda", dtype=torch.int32)
    for b in range(batch_size):
        ctx_len = int(context_blocks[b].item())
        kv_indices[kv_indptr[b] : kv_indptr[b + 1]] = torch.randperm(
            num_block, device="cuda"
        )[:ctx_len]

    return (
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        context_lens,
        num_block,
    )


@pytest.mark.parametrize(
    "batch_size, block_size, num_tp, k_buffer_type, avg_kv_length",
    [
        (4, 1, 4, dtypes.fp8, 512),
        (8, 16, 4, dtypes.fp8, 1024),
        (32, 32, 4, dtypes.fp8, 2048),
        (64, 1, 4, torch.bfloat16, 2048),
        (1, 1, 4, torch.bfloat16, 512),
    ],
)
def test_gather_kv_b_proj(
    batch_size, block_size, num_tp, k_buffer_type, avg_kv_length, perf=False
):
    torch.manual_seed(0)
    random.seed(0)
    # Configuration
    kv_c_dim = 512
    kv_pe_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128
    tp_k_head_num = 128 // num_tp
    num_block = 2 * avg_kv_length // block_size

    weight_preshuffle = True

    device = "cuda"
    weight_dtype = dtypes.fp8

    # Generate random k_buffer
    k_buffer = torch.randn(
        (num_block, block_size, kv_c_dim + kv_pe_dim),
        device=device,
        dtype=torch.float32,
    ).to(k_buffer_type)
    k_scale = torch.randn(1, device=device, dtype=torch.float32).abs()

    # Generate random kv_indptr and kv_indices
    var_ratio = 0.2
    context_lens = (
        torch.randint(
            int((1 - var_ratio) * avg_kv_length),
            int(((1 + var_ratio)) * avg_kv_length) + 1,
            (batch_size,),
        )
        .cuda()
        .to(torch.int32)
    )
    context_blocks = torch.div(
        context_lens + block_size - 1, block_size, rounding_mode="trunc"
    )

    kv_indptr = torch.zeros((batch_size + 1,), device="cuda", dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(context_blocks, dim=0)

    kv_prefix_sum_context_lens = torch.zeros(
        (batch_size + 1,), device="cuda", dtype=torch.int32
    )
    kv_prefix_sum_context_lens[1:] = torch.cumsum(context_lens, dim=0)

    kv_indices = torch.zeros(kv_indptr[-1], device="cuda", dtype=torch.int32)
    for b in range(batch_size):
        ctx_len = int(context_blocks[b].item())
        kv_indices[kv_indptr[b] : kv_indptr[b + 1]] = torch.randperm(
            num_block, device="cuda"
        )[:ctx_len]

    # Generate random kv_proj_weight and kv_proj_scale
    weight_n = tp_k_head_num * (qk_nope_head_dim + v_head_dim)
    kv_proj_weight = torch.randn(
        (weight_n, kv_c_dim),
        device=device,
        dtype=torch.float32,
    ).to(weight_dtype)
    kv_proj_scale = torch.randn(
        (weight_n // 128, 4), device=device, dtype=torch.float32
    ).abs()

    # Reference implementation
    k_ref, v_ref = ref_gather_kv_b_proj(
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        kv_proj_weight,
        kv_proj_scale,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
    )

    k_prefix = torch.zeros(
        (
            kv_prefix_sum_context_lens[-1].item(),
            tp_k_head_num * (qk_nope_head_dim + kv_pe_dim),
        ),
        device=device,
        dtype=torch.bfloat16,
    )
    v_prefix = torch.zeros(
        (kv_prefix_sum_context_lens[-1].item(), tp_k_head_num * v_head_dim),
        device=device,
        dtype=torch.bfloat16,
    )

    if weight_preshuffle:
        kv_proj_weight = shuffle_weight(kv_proj_weight)

    gather_kv_b_proj(
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        kv_proj_weight,
        kv_proj_scale,
        k_prefix.view(-1, tp_k_head_num, qk_nope_head_dim + kv_pe_dim),
        v_prefix.view(-1, tp_k_head_num, v_head_dim),
        weight_preshuffle=weight_preshuffle,
    )

    # Validate results
    checkAllclose(k_ref, k_prefix, atol=1e-2, rtol=1e-2)
    checkAllclose(v_ref, v_prefix, atol=1e-2, rtol=1e-2)

    if perf:
        _, elapsed_us = run_perftest(
            gather_kv_b_proj,
            k_buffer,
            k_scale,
            kv_indptr,
            kv_indices,
            kv_prefix_sum_context_lens,
            kv_proj_weight,
            kv_proj_scale,
            k_prefix.view(-1, tp_k_head_num, qk_nope_head_dim + kv_pe_dim),
            v_prefix.view(-1, tp_k_head_num, v_head_dim),
            weight_preshuffle=weight_preshuffle,
        )
        total_float_operations = (
            2
            * context_lens.float().sum().item()
            * (tp_k_head_num * (qk_nope_head_dim + v_head_dim))
            * kv_c_dim
        )
        tflops = total_float_operations / elapsed_us * 1e-6

        print(">>> Performance gather_kv_b_proj:")
        print(
            f">>>   batch {batch_size}, block_size {block_size}, tp_k_head_num {tp_k_head_num}, kv_c_dim {kv_c_dim}, qk_nope_head_dim {qk_nope_head_dim}, kv_length {avg_kv_length}\n"
            f">>>       elapsed={elapsed_us:.2f}us, TFLOPS={tflops:.2f}"
        )


@pytest.mark.parametrize(
    "batch_size, block_size, num_tp, k_buffer_type, avg_kv_length",
    [
        (4, 1, 4, dtypes.fp8, 512),
        (8, 16, 4, dtypes.fp8, 1024),
        (8, 16, 4, torch.bfloat16, 1024),
    ],
)
def test_gather_kv_b_proj_per_row_scale(
    batch_size, block_size, num_tp, k_buffer_type, avg_kv_length, perf=False
):
    """kv_proj_scale [weight_n]: one BF16/FP32 scale per output row (e.g. F8_E4M3 + per-row)."""
    torch.manual_seed(0)
    random.seed(0)
    kv_c_dim = 512
    kv_pe_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128
    tp_k_head_num = 128 // num_tp
    num_block = 2 * avg_kv_length // block_size
    weight_preshuffle = True
    device = "cuda"
    weight_dtype = dtypes.fp8
    weight_n = tp_k_head_num * (qk_nope_head_dim + v_head_dim)

    k_buffer = torch.randn(
        (num_block, block_size, kv_c_dim + kv_pe_dim),
        device=device,
        dtype=torch.float32,
    ).to(k_buffer_type)
    k_scale = torch.randn(1, device=device, dtype=torch.float32).abs()

    var_ratio = 0.2
    context_lens = (
        torch.randint(
            int((1 - var_ratio) * avg_kv_length),
            int(((1 + var_ratio)) * avg_kv_length) + 1,
            (batch_size,),
        )
        .cuda()
        .to(torch.int32)
    )
    context_blocks = torch.div(
        context_lens + block_size - 1, block_size, rounding_mode="trunc"
    )

    kv_indptr = torch.zeros((batch_size + 1,), device="cuda", dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(context_blocks, dim=0)

    kv_prefix_sum_context_lens = torch.zeros(
        (batch_size + 1,), device="cuda", dtype=torch.int32
    )
    kv_prefix_sum_context_lens[1:] = torch.cumsum(context_lens, dim=0)

    kv_indices = torch.zeros(kv_indptr[-1], device="cuda", dtype=torch.int32)
    for b in range(batch_size):
        ctx_len = int(context_blocks[b].item())
        kv_indices[kv_indptr[b] : kv_indptr[b + 1]] = torch.randperm(
            num_block, device="cuda"
        )[:ctx_len]

    kv_proj_weight = torch.randn(
        (weight_n, kv_c_dim),
        device=device,
        dtype=torch.float32,
    ).to(weight_dtype)
    kv_proj_scale = torch.randn(
        (weight_n, 1), device=device, dtype=torch.bfloat16
    ).abs()

    k_ref, v_ref = ref_gather_kv_b_proj(
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        kv_proj_weight,
        kv_proj_scale,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
    )

    k_prefix = torch.zeros(
        (
            kv_prefix_sum_context_lens[-1].item(),
            tp_k_head_num * (qk_nope_head_dim + kv_pe_dim),
        ),
        device=device,
        dtype=torch.bfloat16,
    )
    v_prefix = torch.zeros(
        (kv_prefix_sum_context_lens[-1].item(), tp_k_head_num * v_head_dim),
        device=device,
        dtype=torch.bfloat16,
    )

    if weight_preshuffle:
        kv_proj_weight = shuffle_weight(kv_proj_weight)

    gather_kv_b_proj(
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        kv_proj_weight,
        kv_proj_scale,
        k_prefix.view(-1, tp_k_head_num, qk_nope_head_dim + kv_pe_dim),
        v_prefix.view(-1, tp_k_head_num, v_head_dim),
        weight_preshuffle=weight_preshuffle,
    )

    checkAllclose(k_ref, k_prefix, atol=1e-2, rtol=1e-2)
    checkAllclose(v_ref, v_prefix, atol=1e-2, rtol=1e-2)

    if perf:
        _, elapsed_us = run_perftest(
            gather_kv_b_proj,
            k_buffer,
            k_scale,
            kv_indptr,
            kv_indices,
            kv_prefix_sum_context_lens,
            kv_proj_weight,
            kv_proj_scale,
            k_prefix.view(-1, tp_k_head_num, qk_nope_head_dim + kv_pe_dim),
            v_prefix.view(-1, tp_k_head_num, v_head_dim),
            weight_preshuffle=weight_preshuffle,
        )
        total_float_operations = (
            2
            * context_lens.float().sum().item()
            * (tp_k_head_num * (qk_nope_head_dim + v_head_dim))
            * kv_c_dim
        )
        tflops = total_float_operations / elapsed_us * 1e-6

        print(">>> Performance gather_kv_b_proj_per_row_scale:")
        print(
            f">>>   batch {batch_size}, block_size {block_size}, tp_k_head_num {tp_k_head_num}, kv_c_dim {kv_c_dim}, qk_nope_head_dim {qk_nope_head_dim}, kv_length {avg_kv_length}\n"
            f">>>       elapsed={elapsed_us:.2f}us, TFLOPS={tflops:.2f}"
        )


@pytest.mark.parametrize(
    "batch_size, block_size, num_tp, k_buffer_type, avg_kv_length, scale_mode",
    [
        (4, 1, 4, torch.bfloat16, 512, "block"),
        (8, 16, 4, torch.bfloat16, 1024, "block"),
        (4, 1, 4, dtypes.fp8, 512, "block"),
        (8, 16, 4, dtypes.fp8, 1024, "block"),
        (4, 1, 4, torch.bfloat16, 512, "per_row"),
        (8, 16, 4, torch.bfloat16, 1024, "per_row"),
        (4, 1, 4, dtypes.fp8, 512, "per_row"),
        (8, 16, 4, dtypes.fp8, 1024, "per_row"),
    ],
)
def test_gather_kv_b_proj_bf16_weight(
    batch_size, block_size, num_tp, k_buffer_type, avg_kv_length, scale_mode, perf=False
):
    """Test gather_kv_b_proj with bf16 weight (no quantization on weight).

    When weight is bf16, weight_scale is set to all-ones so the matmul result
    is not scaled — matching the behavior of an unquantized kv_b_proj.
    """
    torch.manual_seed(0)
    random.seed(0)
    kv_c_dim = 512
    kv_pe_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128
    tp_k_head_num = 128 // num_tp
    num_block = 2 * avg_kv_length // block_size
    weight_preshuffle = True
    device = "cuda"
    weight_dtype = torch.bfloat16
    weight_n = tp_k_head_num * (qk_nope_head_dim + v_head_dim)

    k_buffer = torch.randn(
        (num_block, block_size, kv_c_dim + kv_pe_dim),
        device=device,
        dtype=torch.float32,
    ).to(k_buffer_type)
    k_scale = torch.randn(1, device=device, dtype=torch.float32).abs()

    var_ratio = 0.2
    context_lens = (
        torch.randint(
            int((1 - var_ratio) * avg_kv_length),
            int(((1 + var_ratio)) * avg_kv_length) + 1,
            (batch_size,),
        )
        .cuda()
        .to(torch.int32)
    )
    context_blocks = torch.div(
        context_lens + block_size - 1, block_size, rounding_mode="trunc"
    )

    kv_indptr = torch.zeros((batch_size + 1,), device="cuda", dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(context_blocks, dim=0)

    kv_prefix_sum_context_lens = torch.zeros(
        (batch_size + 1,), device="cuda", dtype=torch.int32
    )
    kv_prefix_sum_context_lens[1:] = torch.cumsum(context_lens, dim=0)

    kv_indices = torch.zeros(kv_indptr[-1], device="cuda", dtype=torch.int32)
    for b in range(batch_size):
        ctx_len = int(context_blocks[b].item())
        kv_indices[kv_indptr[b] : kv_indptr[b + 1]] = torch.randperm(
            num_block, device="cuda"
        )[:ctx_len]

    # bf16 weight — no quantization
    kv_proj_weight = torch.randn(
        (weight_n, kv_c_dim),
        device=device,
        dtype=torch.float32,
    ).to(weight_dtype)

    # Use all-ones scale to simulate no weight quantization
    if scale_mode == "per_row":
        kv_proj_scale = torch.ones((weight_n, 1), device=device, dtype=torch.float32)
    else:
        kv_proj_scale = torch.ones(
            (weight_n // 128, kv_c_dim // 128), device=device, dtype=torch.float32
        )

    k_ref, v_ref = ref_gather_kv_b_proj(
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        kv_proj_weight,
        kv_proj_scale,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
    )

    total_kv = kv_prefix_sum_context_lens[-1].item()
    k_prefix = torch.zeros(
        (total_kv, tp_k_head_num * (qk_nope_head_dim + kv_pe_dim)),
        device=device,
        dtype=torch.bfloat16,
    )
    v_prefix = torch.zeros(
        (total_kv, tp_k_head_num * v_head_dim),
        device=device,
        dtype=torch.bfloat16,
    )

    if weight_preshuffle:
        kv_proj_weight = shuffle_weight(kv_proj_weight)

    gather_kv_b_proj(
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        kv_proj_weight,
        kv_proj_scale,
        k_prefix.view(-1, tp_k_head_num, qk_nope_head_dim + kv_pe_dim),
        v_prefix.view(-1, tp_k_head_num, v_head_dim),
        weight_preshuffle=weight_preshuffle,
    )

    checkAllclose(k_ref, k_prefix, atol=1e-2, rtol=1e-2)
    checkAllclose(v_ref, v_prefix, atol=1e-2, rtol=1e-2)

    if perf:
        _, elapsed_us = run_perftest(
            gather_kv_b_proj,
            k_buffer,
            k_scale,
            kv_indptr,
            kv_indices,
            kv_prefix_sum_context_lens,
            kv_proj_weight,
            kv_proj_scale,
            k_prefix.view(-1, tp_k_head_num, qk_nope_head_dim + kv_pe_dim),
            v_prefix.view(-1, tp_k_head_num, v_head_dim),
            weight_preshuffle=weight_preshuffle,
        )
        total_float_operations = (
            2
            * context_lens.float().sum().item()
            * (tp_k_head_num * (qk_nope_head_dim + v_head_dim))
            * kv_c_dim
        )
        tflops = total_float_operations / elapsed_us * 1e-6

        print(">>> Performance gather_kv_b_proj_bf16_weight:")
        print(
            f">>>   batch {batch_size}, block_size {block_size}, tp_k_head_num {tp_k_head_num}, "
            f"kv_c_dim {kv_c_dim}, qk_nope_head_dim {qk_nope_head_dim}, kv_length {avg_kv_length}, "
            f"scale_mode {scale_mode}\n"
            f">>>       elapsed={elapsed_us:.2f}us, TFLOPS={tflops:.2f}"
        )


@pytest.mark.parametrize(
    "batch_size, block_size, k_buffer_type, avg_kv_length, qk_nope_head_dim, v_head_dim, scale_mode",
    [
        # GLM-5 dims (192/256), per-row scale, tp=4 → 32 heads
        (4, 1, dtypes.fp8, 512, 192, 256, "per_row"),
        (8, 16, dtypes.fp8, 1024, 192, 256, "per_row"),
        (4, 1, torch.bfloat16, 512, 192, 256, "per_row"),
        # GLM-5 dims, bf16 weight + per-row all-ones scale
        (4, 1, dtypes.fp8, 512, 192, 256, "bf16_weight"),
        (8, 16, torch.bfloat16, 1024, 192, 256, "bf16_weight"),
        # Symmetric (DeepSeek-like) as sanity check
        (4, 1, dtypes.fp8, 512, 128, 128, "per_row"),
        (4, 1, dtypes.fp8, 512, 128, 128, "bf16_weight"),
    ],
)
def test_gather_kv_b_proj_asymmetric_dims(
    batch_size,
    block_size,
    k_buffer_type,
    avg_kv_length,
    qk_nope_head_dim,
    v_head_dim,
    scale_mode,
):
    """Test gather_kv_b_proj with qk_nope_head_dim != v_head_dim (e.g. GLM-5: 192/256)."""
    torch.manual_seed(0)
    random.seed(0)
    kv_c_dim = 512
    kv_pe_dim = 64
    num_tp = 4
    tp_k_head_num = 128 // num_tp
    weight_preshuffle = True
    device = "cuda"
    weight_n = tp_k_head_num * (qk_nope_head_dim + v_head_dim)

    (
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        context_lens,
        num_block,
    ) = _make_kv_test_data(
        batch_size,
        block_size,
        avg_kv_length,
        kv_c_dim,
        kv_pe_dim,
        k_buffer_type,
        device,
    )

    if scale_mode == "bf16_weight":
        weight_dtype = torch.bfloat16
        kv_proj_scale = torch.ones(weight_n, device=device, dtype=torch.float32)
    else:
        weight_dtype = dtypes.fp8
        kv_proj_scale = torch.randn(
            (weight_n, 1), device=device, dtype=torch.float32
        ).abs()

    kv_proj_weight = torch.randn(
        (weight_n, kv_c_dim),
        device=device,
        dtype=torch.float32,
    ).to(weight_dtype)

    k_ref, v_ref = ref_gather_kv_b_proj(
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        kv_proj_weight,
        kv_proj_scale,
        qk_nope_head_dim=qk_nope_head_dim,
        v_head_dim=v_head_dim,
    )

    total_kv = kv_prefix_sum_context_lens[-1].item()
    k_prefix = torch.zeros(
        (total_kv, tp_k_head_num * (qk_nope_head_dim + kv_pe_dim)),
        device=device,
        dtype=torch.bfloat16,
    )
    v_prefix = torch.zeros(
        (total_kv, tp_k_head_num * v_head_dim),
        device=device,
        dtype=torch.bfloat16,
    )

    if weight_preshuffle:
        kv_proj_weight = shuffle_weight(kv_proj_weight)

    gather_kv_b_proj(
        k_buffer,
        k_scale,
        kv_indptr,
        kv_indices,
        kv_prefix_sum_context_lens,
        kv_proj_weight,
        kv_proj_scale,
        k_prefix.view(-1, tp_k_head_num, qk_nope_head_dim + kv_pe_dim),
        v_prefix.view(-1, tp_k_head_num, v_head_dim),
        weight_preshuffle=weight_preshuffle,
    )

    checkAllclose(k_ref, k_prefix, atol=1e-2, rtol=1e-2)
    checkAllclose(v_ref, v_prefix, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-B", "--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=1,
        help="KVCache block size, only used when kv_preshuffle is enabled, must be multiple of 16",
    )
    parser.add_argument(
        "-num_tp",
        type=int,
        default=4,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "-kv_length",
        type=int,
        default=1024,
        help="Sequence length of K buffer",
    )
    parser.add_argument(
        "-ktype",
        type=str,
        default="fp8",
        help="Tensor type of K buffer, should be fp8 or bf16",
    )
    parser.add_argument(
        "--kv_preshuffle",
        action="store_true",
        help="Enable KV cache preshuffle, also change blocksize to 16",
    )

    args = parser.parse_args()

    assert (
        args.ktype == "fp8" or args.ktype == "bf16"
    ), "Only fp8 and bfloat16 are supported"
    k_buffer_type = dtypes.fp8 if args.ktype == "fp8" else torch.bfloat16

    test_gather_kv_b_proj_per_row_scale(
        args.batch,
        args.blocksize,
        args.num_tp,
        k_buffer_type,
        args.kv_length,
        perf=True,
    )
    test_gather_kv_b_proj(
        args.batch,
        args.blocksize,
        args.num_tp,
        k_buffer_type,
        args.kv_length,
        perf=True,
    )

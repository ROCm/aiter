# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import math
import random
import argparse

import pytest
import torch

from aiter.test_common import checkAllclose, run_perftest
from aiter.ops.triton.gather_kv_b_proj import gather_kv_b_proj
from aiter.ops.shuffle import shuffle_weight


def ref_gather_kv_b_proj(
    k_buffer: torch.Tensor,  # [num_block, block_size, hidden_dim]
    k_scale: torch.Tensor,  # [1]
    kv_indptr: torch.Tensor,  # [batch_size + 1]
    kv_indices: torch.Tensor,  # len(kv_indices) = kv_indptr[-1]
    kv_prefix_sum_context_lens: torch.Tensor,  # [batch_size + 1]
    kv_proj_weight: torch.Tensor,  # [2 * 128 // TP * 128, 512]
    kv_proj_scale: torch.Tensor,  # [2 * 128 // TP, 4], blockscale=128 x 128
):
    batch_size = kv_indptr.shape[0] - 1

    kv_c_dim = 512
    kv_pe_dim = 64

    num_block, block_size, hidden_dim = k_buffer.shape
    weight_n, weight_k = kv_proj_weight.shape

    scale_granularity_n = weight_n // kv_proj_scale.shape[0]
    scale_granularity_k = weight_k // kv_proj_scale.shape[1]

    assert hidden_dim == kv_c_dim + kv_pe_dim
    assert weight_k == kv_c_dim
    assert scale_granularity_k == 128

    num_tp = 2 * 128 * 128 // weight_n
    tp_k_head_num = 128 // num_tp
    qk_nope_head_dim = 128

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
        (total_kv, tp_k_head_num * qk_nope_head_dim),
        device=k_buffer.device,
        dtype=torch.bfloat16,
    )
    k_prefix_tp = k_prefix.view(total_kv, tp_k_head_num, qk_nope_head_dim + kv_pe_dim)
    v_prefix_tp = v_prefix.view(total_kv, tp_k_head_num, qk_nope_head_dim)

    kv_proj_scale_repeat = kv_proj_scale.repeat_interleave(scale_granularity_n, dim=0)

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

        kv_proj = torch.zeros(
            (context_end - context_start, weight_n),
            device=k_buffer.device,
            dtype=torch.float32,
        )
        for i in range(weight_k // scale_granularity_k):
            kv_proj_tmp = (
                k_data[:, i * scale_granularity_k : (i + 1) * scale_granularity_k].to(
                    torch.float32
                )
                @ kv_proj_weight[
                    :, i * scale_granularity_k : (i + 1) * scale_granularity_k
                ]
                .to(torch.float32)
                .T
            )  # [batch_kv, 2 * 128 // TP * 128]
            kv_proj += kv_proj_tmp * kv_proj_scale_repeat[:, i].unsqueeze(0)

        kv_proj_tp = kv_proj.view(
            context_end - context_start, tp_k_head_num, qk_nope_head_dim * 2
        )  # [batch_kv, tp_k_head_num, 2 * 128 // TP * 128]

        if k_buffer.dtype != torch.bfloat16:
            kv_proj_tp *= k_scale.unsqueeze(0).unsqueeze(1)

        k_proj_tp, v_proj_tp = kv_proj_tp.split(
            [qk_nope_head_dim, qk_nope_head_dim], dim=-1
        )  # [batch_kv, tp_k_head_num, 128 // TP * 128]

        k_prefix_tp[
            context_start:context_end,
            :,
            :qk_nope_head_dim,
        ] = k_proj_tp
        v_prefix_tp[context_start:context_end, :] = v_proj_tp

    return (k_prefix, v_prefix)


@pytest.mark.parametrize(
    "batch_size, block_size, num_tp, k_buffer_type, avg_kv_length",
    [
        (4, 1, 4, torch.float8_e4m3fnuz, 512),
        (8, 16, 4, torch.float8_e4m3fnuz, 1024),
        (32, 32, 4, torch.float8_e4m3fnuz, 2048),
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
    tp_k_head_num = 128 // num_tp
    num_block = 2 * avg_kv_length // block_size

    weight_preshuffle = False  # block_size % 16 == 0

    device = "cuda"
    weight_dtype = torch.float8_e4m3fnuz

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
    kv_proj_weight = torch.randn(
        (2 * 128 // num_tp * qk_nope_head_dim, kv_c_dim),
        device=device,
        dtype=torch.float32,
    ).to(weight_dtype)
    kv_proj_scale = torch.randn(
        (2 * 128 // num_tp, 4), device=device, dtype=torch.float32
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
        (kv_prefix_sum_context_lens[-1].item(), tp_k_head_num * qk_nope_head_dim),
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
        v_prefix.view(-1, tp_k_head_num, qk_nope_head_dim),
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
            v_prefix.view(-1, tp_k_head_num, qk_nope_head_dim),
        )
        total_float_operations = (
            2
            * context_lens.float().sum().item()
            * (2 * tp_k_head_num * qk_nope_head_dim)
            * kv_c_dim  # gemm_m  # gemm_n  # gemm_k
        )
        tflops = total_float_operations / elapsed_us * 1e-6

        print(f">>> Performance gather_kv_b_proj:")
        print(
            f">>>   batch {batch_size}, block_size {block_size}, tp_k_head_num {tp_k_head_num}, kv_c_dim {kv_c_dim}, qk_nope_head_dim {qk_nope_head_dim}, kv_length {avg_kv_length}\n"
            f">>>       elapsed={elapsed_us:.2f}us, TFLOPS={tflops:.2f}"
        )


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
    test_gather_kv_b_proj(
        args.batch,
        args.blocksize,
        args.num_tp,
        torch.float8_e4m3fnuz if args.ktype == "fp8" else torch.bfloat16,
        args.kv_length,
        perf=True,
    )

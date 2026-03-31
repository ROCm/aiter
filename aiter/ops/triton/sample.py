# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _mixed_sample_outer_exponential_kernel(
    output_ptr,
    input_ptr,
    exponentials_ptr,
    temperatures_ptr,
    N,
    stride_input_m,
    stride_exp_m,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    temperature = tl.load(temperatures_ptr + row_idx)

    input_row_start = input_ptr + row_idx * stride_input_m
    exp_row_start = exponentials_ptr + row_idx * stride_exp_m

    is_greedy = temperature == 0.0
    safe_temp = tl.where(is_greedy, 1.0, tl.maximum(temperature, 1e-5))
    inv_temp = 1.0 / safe_temp

    # Pass 1: find max for numerical stability
    global_max = float("-inf")
    for block_start in range(0, N, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        logits = tl.load(input_row_start + cols, mask=mask, other=float("-inf"))
        scaled = (logits * inv_temp).to(tl.float32)
        block_max = tl.max(scaled, axis=0)
        global_max = tl.maximum(global_max, block_max)

    # Pass 2: compute sum of exp(scaled - max)
    global_sum = 0.0
    for block_start in range(0, N, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        logits = tl.load(input_row_start + cols, mask=mask, other=float("-inf"))
        scaled = (logits * inv_temp).to(tl.float32)
        exp_vals = tl.exp(scaled - global_max)
        exp_vals = tl.where(mask, exp_vals, 0.0)
        global_sum += tl.sum(exp_vals, axis=0)

    # Pass 3: compute scores and find argmax
    best_val = float("-inf")
    best_idx: tl.int32 = 0
    for block_start in range(0, N, BLOCK_SIZE):
        cols = block_start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N

        logits = tl.load(input_row_start + cols, mask=mask, other=float("-inf"))
        scaled = (logits * inv_temp).to(tl.float32)

        # softmax prob
        probs = tl.exp(scaled - global_max) / global_sum

        # Gumbel-max: prob / exponential
        exponentials = tl.load(exp_row_start + cols, mask=mask, other=1.0)
        gumbel_vals = probs / (exponentials + eps)

        # For greedy, use raw logits for argmax
        score = tl.where(is_greedy, logits.to(tl.float32), gumbel_vals)
        score = tl.where(mask, score, float("-inf"))

        block_best_val = tl.max(score, axis=0)
        block_best_idx = tl.argmax(score, axis=0) + block_start

        better = block_best_val > best_val
        best_val = tl.where(better, block_best_val, best_val)
        best_idx = tl.where(better, block_best_idx, best_idx)

    tl.store(output_ptr + row_idx, best_idx)


def mixed_sample_outer_exponential_triton(
    out: torch.Tensor,
    input: torch.Tensor,
    exponentials: torch.Tensor,
    temperature: torch.Tensor,
    eps: float = 1e-10,
) -> None:
    M, N = input.shape
    BLOCK_SIZE = min(triton.next_power_of_2(N), 8192)

    _mixed_sample_outer_exponential_kernel[(M,)](
        out,
        input,
        exponentials,
        temperature,
        N=N,
        stride_input_m=input.stride(0),
        stride_exp_m=exponentials.stride(0),
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

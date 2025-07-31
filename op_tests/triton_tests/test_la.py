# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import sys
import pytest
import torch
from bisect import bisect_right
from typing import Union, List
from aiter.ops.triton.lean_atten import (
    _persistent_lean_attention,
    persistent_lean_attention,
    _get_config,
)
import aiter.ops.triton.utils.arch_info as arch_info


def get_lean_attn_inputs(
    batch: int,
    n_ctx_q: int,
    n_ctx: List[int],
    block_n: int,
    h: int,
    d: int,
    total_programs: int,
    init_dtype: Union[torch.dtype, str],
):
    assert batch == len(n_ctx)
    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")
    # Allocate Tensors
    q = torch.empty((n_ctx_q * batch, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((sum_n_ctx, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((sum_n_ctx, h, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # LeanAttention Specific Parameters
    Mp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, n_ctx_q, d), device=q.device, dtype=torch.float32)

    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)

    # N_CTX is a list of context lengthes for all the req in a batch
    # First, calculate #BLOCK_N for each context length "list_num_block_n"
    # Second, Convert it to a list of assumulative lengthes "list_sum_block_n"
    # Third, convert list to a tensor "batch_num_block_n"
    for s in n_ctx:
        # print(f"s={s}")
        list_num_block_n = [
            (int(str(s).strip()) + block_n - 1) // block_n for s in n_ctx
        ]
    len_sum = 0
    list_sum_block_n = []
    for i in range(batch):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, device="cuda", dtype=torch.int32)

    return q, k, v, Mp, Lp, Op, locks, batch_num_block_n


def get_lean_attention_params(
    causal, batch_size, max_seqlen_q, max_seqlen_k, num_heads, BLOCK_M, BLOCK_N, num_SMs
):
    """
    Mirrors the get_num_splits_and_buffer_sizes logic from the host code.
    """
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        # The total number of lean tiles for a single head/batch item
        for i in range(num_m_blocks):
            tiles_per_head += (i + 1) * (BLOCK_M // BLOCK_N)
        tiles_per_head *= batch_size
    else:
        num_n_blocks_total = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N
        tiles_per_head = num_m_blocks * num_n_blocks_total

    total_tiles = tiles_per_head * num_heads
    lean_griddimz = num_SMs

    if lean_griddimz == 0:
        return 0, 0, 0, 0, 0

    max_tiles_per_wg = (total_tiles + lean_griddimz - 1) // lean_griddimz
    high_load_wgs = total_tiles % lean_griddimz
    if high_load_wgs == 0 and total_tiles > 0:
        high_load_wgs = lean_griddimz

    return (
        tiles_per_head,
        num_m_blocks,
        lean_griddimz,
        high_load_wgs,
        max_tiles_per_wg,
    )


def calculate_max_output_tiles_analytically(
    causal: bool,
    batch_size: int,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_SMs: int,
    BLOCK_M: int,
    BLOCK_N: int,
):
    """
    Calculates the maximum number of output tiles any single workgroup will process
    using a fast, analytical method with binary search.
    """
    MASKED_BLOCKS = BLOCK_M // BLOCK_N
    if causal and BLOCK_M % BLOCK_N != 0:
        raise ValueError("For causal, BLOCK_M must be a multiple of BLOCK_N")

    (
        tiles_per_head,
        num_m_blocks,
        num_wgs,
        high_load_wgs,
        max_tiles_per_wg,
    ) = get_lean_attention_params(
        causal,
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads,
        BLOCK_M,
        BLOCK_N,
        num_SMs,
    )

    if num_wgs == 0:
        return 0

    m_block_boundaries = []
    if causal:
        # Pre-compute the boundaries of each M-block's workload for a single head.
        # This list will be used for binary searches.
        total_blocks = 0
        for i in range(num_m_blocks):
            pair_idx = i // 2
            q_block_idx = pair_idx if (i % 2) == 0 else num_m_blocks - 1 - pair_idx
            task_size = (q_block_idx + 1) * MASKED_BLOCKS
            total_blocks += task_size
            m_block_boundaries.append(total_blocks)

    max_total_output_tiles = 0
    # Loop through each workgroup to find the one that spans the most output tiles.
    for wg_id in range(num_wgs):
        total_output_tiles_for_wg = 0

        # Determine the range of global lean tile indices for this WG
        if wg_id < high_load_wgs:
            start_iter = max_tiles_per_wg * wg_id
            end_iter = start_iter + max_tiles_per_wg
        else:
            start_iter = (max_tiles_per_wg - 1) * (
                wg_id - high_load_wgs
            ) + high_load_wgs * max_tiles_per_wg
            end_iter = start_iter + (max_tiles_per_wg - 1)

        start_head = start_iter // tiles_per_head
        end_head = (end_iter - 1) // tiles_per_head

        # Loop through each head this workgroup touches
        for head_idx in range(start_head, end_head + 1):
            head_start_iter = head_idx * tiles_per_head

            # Find the intersection of the WG's range and the current head's range
            wg_start_in_head = max(start_iter, head_start_iter)
            wg_end_in_head = min(end_iter, head_start_iter + tiles_per_head)

            if not causal:
                # For non-causal, each head is one output tile.
                total_output_tiles_for_wg += 1
                continue

            # --- Causal Logic using Binary Search ---
            # Convert to indices relative to the start of the head's workload
            relative_start = wg_start_in_head - head_start_iter
            relative_end = wg_end_in_head - head_start_iter

            # Use binary search to find which M-block the start and end tiles fall into
            start_m_idx = bisect_right(m_block_boundaries, relative_start)
            end_m_idx = bisect_right(m_block_boundaries, relative_end - 1)

            # The number of output tiles is the number of boundaries crossed
            tiles_in_this_head = (end_m_idx - start_m_idx) + 1
            total_output_tiles_for_wg += tiles_in_this_head

        max_total_output_tiles = max(max_total_output_tiles, total_output_tiles_for_wg)

    return max_total_output_tiles


@pytest.mark.parametrize(
    "causal, batch, h, n_ctx_q, n_ctx, d, total_programs, init_dtype, BLOCK_M, BLOCK_N, waves_per_eu, num_warps ",
    [
        (False, 2, 64, 128, [65536, 65536], 128, 304, torch.float16, 128, 64, 1, 4),
        (False, 2, 64, 16, [65536, 65536], 128, 912, torch.float16, 16, 128, 3, 4),
        (False, 1, 64, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 64, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 64, 16, [524288], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 2, 96, 16, [32768, 32768], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 96, 16, [65536], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 96, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 96, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 96, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
        (False, 1, 96, 16, [1048576], 16, 912, torch.float16, 16, 256, 1, 4),  #
        (False, 1, 128, 16, [32768], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 128, 16, [65536], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 128, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
        (False, 1, 128, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 128, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
        (
            False,
            3,
            64,
            16,
            [4096, 32768, 65536],
            128,
            912,
            torch.float16,
            16,
            128,
            2,
            4,
        ),
        (
            False,
            8,
            64,
            16,
            [1024, 1024, 2048, 2048, 4096, 4096, 32768, 65536],
            128,
            912,
            torch.float16,
            16,
            64,
            2,
            4,
        ),
        (
            True,
            1,
            64,
            8192,
            [8192],
            128,
            304,
            torch.float16,
            128,
            64,
            2,
            4,
        ),  # Causal=1,
        (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 128, 64, 2, 4),
        # These test cases fail:
        # (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 128, 64, 2, 4),
        # (True, 1, 64, 4096, [4096], 128, 304, torch.float16, 128, 16, 3, 4),
        # (False, 1, 64, 4096, [4096], 128, 304, torch.float16, 128, 16, 3, 4),
    ],
)
def test_persistent_lean_attention(
    request,
    batch,
    h,
    n_ctx_q,
    n_ctx,
    d,
    total_programs,
    init_dtype,
    BLOCK_M,
    BLOCK_N,
    waves_per_eu,
    num_warps,
    causal,
):

    torch.manual_seed(20)
    # Long seqlen (>512K) can hit memory access fault. Suspect compiler issue
    # WA with shorter d and longer BLOCK_N
    if any(item > 524288 for item in n_ctx):
        BLOCK_N = 256
        d = 16

    assert batch == len(n_ctx)
    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")

    max_output_tile_cnt = calculate_max_output_tiles_analytically(
        causal=causal,
        batch_size=batch,
        max_seqlen_q=n_ctx_q,
        max_seqlen_k=sum_n_ctx,
        num_heads=h,
        num_SMs=total_programs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    max_output_tile_cnt = max_output_tile_cnt + 4

    # N_CTX is a list of context lengthes for all the req in a batch
    # First, calculate #BLOCK_N for each context length "list_num_block_n"
    # Second, Convert it to a list of assumulative lengthes "list_sum_block_n"
    # Third, convert list to a tensor "batch_num_block_n"
    for s in n_ctx:
        # print(f"s={s}")
        list_num_block_n = [
            (int(str(s).strip()) + BLOCK_N - 1) // BLOCK_N for s in n_ctx
        ]
    len_sum = 0
    list_sum_block_n = []
    for i in range(batch):
        len_sum += list_num_block_n[i]
        list_sum_block_n.append(len_sum)
    batch_num_block_n = torch.tensor(list_sum_block_n, device="cuda", dtype=torch.int32)

    sm_scale = 0.5

    q, k, v, Mp, Lp, Op, locks, batch_num_block_n = get_lean_attn_inputs(
        batch,
        n_ctx_q,
        n_ctx,
        BLOCK_N,
        h,
        d,
        total_programs,
        init_dtype,
    )

    # Triton LeanAttention output
    la_out, ms = _persistent_lean_attention(
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        total_programs,
        BLOCK_M,
        BLOCK_N,
        causal,
        batch,
        sm_scale,
        num_warps,
        waves_per_eu,
        max_output_tile_cnt,
    )

    # Calculate Pytorch refence output
    ref_out = torch.empty_like(q, dtype=v.dtype)
    start = 0
    start_q = 0

    for b in n_ctx:
        qb = q[start_q : (start_q + int(n_ctx_q)), :, :]
        qb_reshaped = qb.transpose(0, 1)
        kb = k[start : (start + int(b)), :, :]
        kb_reshaped = kb.transpose(0, 1)
        vb = v[start : (start + int(b)), :, :]
        vb_reshaped = vb.transpose(0, 1)
        p = torch.matmul(qb_reshaped, kb_reshaped.transpose(-2, -1)) * sm_scale
        if causal:
            M = torch.tril(torch.ones((n_ctx_q, b), device="cuda"))
            mask = M == 0
            p[:, mask] = float("-inf")
        # print(f"p shape: {p.shape}")
        p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        refb = torch.matmul(p, vb_reshaped)
        ref_out[start_q : (start_q + int(n_ctx_q)), :, :] = refb.transpose(0, 1)
        start += b
        start_q += n_ctx_q

    # Compare result
    atol = 1.4e-1 if init_dtype == "fp8" else 1e-2
    rtol = 1e-2 if init_dtype == "fp8" else 3e-3
    torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)


# NOTE: Tests where the workload < num_sms currently fail.
# You can elicit this behavior by decreasing `h` and `n_ctx`.
# Tests also appear to fail when n_ctx_q != n_ctx when causal=True.
@pytest.mark.skip(
    "Known issue with lean attention causes these tests to fail. La is a WIP."
)
@pytest.mark.parametrize("batch", [1])
@pytest.mark.parametrize("h", [16])
@pytest.mark.parametrize("n_ctx_q", [8192])
@pytest.mark.parametrize("n_ctx", [[8192]])
@pytest.mark.parametrize("d", [32])
@pytest.mark.parametrize("causal", [(True), (False)])
@pytest.mark.parametrize("init_dtype", [torch.float16])
def test_persistent_lean_attention_outer(
    batch,
    h,
    n_ctx_q,
    n_ctx,
    d,
    init_dtype,
    causal,
):
    torch.manual_seed(20)

    sm_scale = 0.5
    config = _get_config(
        batch_size=batch,
        causal=causal,
    )
    sm_count = arch_info.get_num_sms()

    # Long seqlen (>512K) can hit memory access fault. Suspect compiler issue
    # WA with shorter d and longer BLOCK_N
    if any(item > 524288 for item in n_ctx):
        config["BLOCK_SIZE_N"] = 256
        d = 16

    q, k, v, Mp, Lp, Op, locks, batch_num_block_n = get_lean_attn_inputs(
        batch,
        n_ctx_q,
        n_ctx,
        config["BLOCK_SIZE_N"],
        h,
        d,
        sm_count,
        init_dtype,
    )

    # Triton LeanAttention output
    la_out = persistent_lean_attention(
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        batch,
        sm_scale,
        causal=causal,
        config=config,
    )

    # Calculate Pytorch refence output
    ref_out = torch.empty_like(q, dtype=v.dtype)
    start = 0
    start_q = 0

    for b in n_ctx:
        qb = q[start_q : (start_q + int(n_ctx_q)), :, :]
        qb_reshaped = qb.transpose(0, 1)
        kb = k[start : (start + int(b)), :, :]
        kb_reshaped = kb.transpose(0, 1)
        vb = v[start : (start + int(b)), :, :]
        vb_reshaped = vb.transpose(0, 1)
        p = torch.matmul(qb_reshaped, kb_reshaped.transpose(-2, -1)) * sm_scale
        if causal:
            M = torch.tril(torch.ones((n_ctx_q, b), device="cuda"))
            mask = M == 0
            p[:, mask] = float("-inf")
        # print(f"p shape: {p.shape}")
        p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        refb = torch.matmul(p, vb_reshaped)
        ref_out[start_q : (start_q + int(n_ctx_q)), :, :] = refb.transpose(0, 1)
        start += b
        start_q += n_ctx_q

    # Compare result
    atol = 1.4e-1 if init_dtype == "fp8" else 1e-2
    rtol = 1e-2 if init_dtype == "fp8" else 3e-3
    torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)


def print_mismatches(ref_out, la_out, atol=1e-8, rtol=1e-5):
    # Check if shapes match first
    if ref_out.shape != la_out.shape:
        print(f"Shape mismatch! Reference: {ref_out.shape}, Actual: {la_out.shape}")
        return

    # Find mismatches using absolute and relative tolerance
    abs_diff = torch.abs(ref_out - la_out)
    rel_diff = abs_diff / (
        torch.abs(ref_out) + 1e-8
    )  # Add small epsilon to avoid division by zero

    mismatch_mask = (abs_diff > atol) & (rel_diff > rtol)

    if not mismatch_mask.any():
        print("Tensors match within tolerances!")
        return

    # Get indices of mismatches
    mismatched_indices = torch.nonzero(mismatch_mask)

    print(f"Found {len(mismatched_indices)} mismatches:")
    for idx in mismatched_indices:
        idx_tuple = tuple(idx.tolist())
        print(f"At index {idx_tuple}:")
        print(f"  Reference: {ref_out[idx_tuple].item()}")
        print(f"  Actual:    {la_out[idx_tuple].item()}")
        print(f"  Abs diff:  {abs_diff[idx_tuple].item()}")
        print(f"  Rel diff:  {rel_diff[idx_tuple].item()}\n")


def main():
    batch = 1
    causal = True
    h = 64
    n_ctx_q = 8192
    n_ctx = [8192]
    d = 128
    total_programs = 304
    init_dtype = torch.float16
    BLOCK_M = 128
    BLOCK_N = 64
    waves_per_eu = 2
    num_warps = 4
    assert batch == len(n_ctx)

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")

    max_output_tile_cnt = calculate_max_output_tiles_analytically(
        causal=causal,
        batch_size=batch,
        max_seqlen_q=n_ctx_q,
        max_seqlen_k=sum_n_ctx,
        num_heads=h,
        num_SMs=total_programs,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    print(f"max_output_tile_cnt={max_output_tile_cnt}")
    max_output_tile_cnt = max_output_tile_cnt + 4
    print(f"causal={causal}, batch={batch}")

    sm_scale = 0.5

    q, k, v, Mp, Lp, Op, locks, batch_num_block_n = get_lean_attn_inputs(
        batch,
        n_ctx_q,
        n_ctx,
        BLOCK_N,
        h,
        d,
        total_programs,
        init_dtype,
    )

    # Triton LeanAttention output
    la_out, ms = _persistent_lean_attention(
        q,
        k,
        v,
        Mp,
        Lp,
        Op,
        locks,
        batch_num_block_n,
        total_programs,
        BLOCK_M,
        BLOCK_N,
        causal,
        batch,
        sm_scale,
        num_warps,
        waves_per_eu,
        max_output_tile_cnt,
    )
    # print(f"ms={ms}")
    # Calculate Pytorch refence output
    ref_out = torch.empty_like(q, dtype=v.dtype)
    start = 0
    start_q = 0

    for b in n_ctx:
        qb = q[start_q : (start_q + int(n_ctx_q)), :, :]
        # print(f"qb shape: {qb.shape}")
        qb_reshaped = qb.transpose(0, 1)
        # print(f"qb reshaped shape: {qb_reshaped.shape}")
        kb = k[start : (start + int(b)), :, :]
        kb_reshaped = kb.transpose(0, 1)
        vb = v[start : (start + int(b)), :, :]
        vb_reshaped = vb.transpose(0, 1)
        p = torch.matmul(qb_reshaped, kb_reshaped.transpose(-2, -1)) * sm_scale
        if causal:
            M = torch.tril(torch.ones((n_ctx_q, b), device="cuda"))
            mask = M == 0
            p[:, mask] = float("-inf")
        # print(f"p shape: {p.shape}")
        p = torch.softmax(p.float(), dim=-1).to(q.dtype)
        refb = torch.matmul(p, vb_reshaped)
        ref_out[start_q : (start_q + int(n_ctx_q)), :, :] = refb.transpose(0, 1)
        start += b
        start_q += n_ctx_q

    # Compare result
    atol = 1.4e-1 if init_dtype == "fp8" else 1e-2
    rtol = 1e-2 if init_dtype == "fp8" else 3e-3
    try:
        torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)
    except AssertionError:
        print("Assertion failed! Showing mismatches:")
        print_mismatches(ref_out, la_out, atol, rtol)
        raise  # Re-raise the exception after printing mismatches

    # torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(main())
#     benchmark_params = BenchmarkArgs()
#     args = benchmark_params.parse_args()
#     bench_streamk(args.m, args.n, args.k, args.total_programs_streamk, str_to_dtype(args.in_dtype), str_to_dtype(args.out_dtype), args.BLK_M, args.BLK_N, args.BLK_K, args.gsize_m)

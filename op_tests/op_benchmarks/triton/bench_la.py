# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import sys
import torch
import triton

from aiter.ops.triton.lean_atten import persistent_lean_attention

from bisect import bisect_right


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


configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=[
            "causal",
            "batch",
            "h",
            "n_ctx_q",
            "n_ctx",
            "d",
            "total_programs",
            "init_dtype",
            "BLOCK_M",
            "BLOCK_N",
            "waves_per_eu",
            "num_warps",
        ],
        x_vals=[
            (False, 2, 64, 16, [65536, 65536], 128, 912, torch.float16, 16, 128, 2, 4),
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
                128,
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
        ],
        line_arg="provider",
        line_vals=["triton"],
        line_names=["Triton(ms)"],
        # styles=[('red', '-'), ('blue', '-')],
        ylabel="ms",
        plot_name="lean-attention-",
        args={
            # "causal": causal,
        },
    )
)


@triton.testing.perf_report(configs)
def bench_lean_attention(
    causal,
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
    provider,
    device="cuda",
):

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

    # Triton LeanAttention output
    fn = lambda: persistent_lean_attention(  # noqa: E731
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

    warmup = 1
    rep = 1

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms


def main():
    bench_lean_attention.run(save_path=".", print_data=True)


if __name__ == "__main__":
    sys.exit(main())

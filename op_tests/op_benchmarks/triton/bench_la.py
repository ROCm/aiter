# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import sys
import torch
import triton

from aiter.ops.triton.lean_atten import _persistent_lean_attention


configs = []
configs.append(
    triton.testing.Benchmark(
        x_names=[
            "causal",
            "batch",
            "hq",
            "hk",
            "n_ctx_q",
            "n_ctx_k",
            "d",
            "total_programs",
            "init_dtype",
            "BLOCK_M",
            "BLOCK_N",
            "RAGGED_BATCH",
            "waves_per_eu",
            "num_warps",
        ],
        x_vals=[
            # (False, 2, 64, 64, 16, [65536, 65536], 128, 912, torch.float16, 16, 128, 2, 4),
            # (False, 1, 64, 64, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
            # (False, 1, 64, 64, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
            # (False, 1, 64, 64, 16, [524288], 64, 912, torch.float16, 16, 64, 2, 4),
            # (False, 2, 96, 96, 16, [32768, 32768], 128, 912, torch.float16, 16, 128, 2, 4),
            # (False, 1, 96, 96, 16, [65536], 128, 912, torch.float16, 16, 128, 2, 4),
            # (False, 1, 96, 96, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
            # (False, 1, 96, 96, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
            # (False, 1, 96, 96, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
            # (False, 1, 96, 96, 16, [1048576], 16, 912, torch.float16, 16, 256, 1, 4),  #
            # (False, 1, 128, 128, 16, [32768], 128, 912, torch.float16, 16, 128, 2, 4),
            # (False, 1, 128, 128, 16, [65536], 128, 912, torch.float16, 16, 128, 2, 4),
            # (False, 1, 128, 128, 16, [131072], 128, 912, torch.float16, 16, 128, 2, 4),
            # (False, 1, 128, 128, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
            # (False, 1, 128, 128, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
            # (
            #     False,
            #     3,
            #     64,
            #     64,
            #     16,
            #     [4096, 32768, 65536],
            #     128,
            #     912,
            #     torch.float16,
            #     16,
            #     128,
            #     2,
            #     4,
            # ),
            # (
            #     False,
            #     8,
            #     64,
            #     64,
            #     16,
            #     [1024, 1024, 2048, 2048, 4096, 4096, 32768, 65536],
            #     128,
            #     912,
            #     torch.float16,
            #     16,
            #     128,
            #     2,
            #     4,
            # ),
            # (
            #     True,
            #     1,
            #     64,
            #     64,
            #     8192,
            #     [8192],
            #     128,
            #     912,
            #     torch.float16,
            #     128,
            #     64,
            #     2,
            #     4,
            # ),  # Causal=1,
            # (True, 2, 64, 64, 2048, [2048, 2048], 128, 608, torch.float16, 128, 64, 2, 4),
            # Diff here
            (
                True,
                1,
                32,
                8,
                8192,
                [8192],
                128,
                608,
                torch.float16,
                128,
                64,
                False,
                2,
                4,
            ),
            (
                True,
                1,
                64,
                8,
                8192,
                [8192],
                128,
                608,
                torch.float16,
                128,
                64,
                False,
                2,
                4,
            ),
            (
                True,
                1,
                128,
                8,
                8192,
                [8192],
                128,
                608,
                torch.float16,
                128,
                64,
                False,
                2,
                4,
            ),
            (
                True,
                1,
                32,
                16,
                1024,
                [1024],
                128,
                608,
                torch.float16,
                128,
                64,
                False,
                2,
                4,
            ),
            (
                True,
                1,
                64,
                16,
                1024,
                [1024],
                128,
                608,
                torch.float16,
                128,
                64,
                False,
                2,
                4,
            ),
            (
                True,
                1,
                128,
                16,
                1024,
                [1024],
                128,
                608,
                torch.float16,
                128,
                64,
                False,
                2,
                4,
            ),
            (
                True,
                1,
                32,
                32,
                2048,
                [2048],
                128,
                608,
                torch.float16,
                128,
                64,
                False,
                2,
                4,
            ),
            (
                True,
                1,
                64,
                32,
                2048,
                [2048],
                128,
                608,
                torch.float16,
                128,
                64,
                False,
                2,
                4,
            ),
            (
                True,
                1,
                128,
                32,
                2048,
                [2048],
                128,
                608,
                torch.float16,
                128,
                64,
                False,
                2,
                4,
            ),
            (
                False,
                512,
                32,
                8,
                16,
                [8192],
                128,
                608,
                torch.float16,
                16,
                64,
                False,
                2,
                4,
            ),
            (
                False,
                512,
                64,
                8,
                16,
                [8192],
                128,
                608,
                torch.float16,
                16,
                128,
                False,
                2,
                4,
            ),
            (
                False,
                512,
                128,
                8,
                16,
                [8192],
                128,
                608,
                torch.float16,
                16,
                128,
                False,
                2,
                4,
            ),
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
    hq,
    hk,
    n_ctx_q,
    n_ctx_k,
    d,
    total_programs,
    init_dtype,
    BLOCK_M,
    BLOCK_N,
    RAGGED_BATCH,
    waves_per_eu,
    num_warps,
    provider,
    device="cuda",
):
    n_ctx = n_ctx_k * batch
    assert batch == len(n_ctx)

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")

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

    # Allocate Tensors
    q = torch.empty((n_ctx_q * batch, hq, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    k = torch.empty((sum_n_ctx, hk, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )
    v = torch.empty((sum_n_ctx, hk, d), dtype=init_dtype, device="cuda").normal_(
        mean=0.0, std=0.5
    )

    # LeanAttention Specific Parameters
    Mp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Lp = torch.empty((total_programs, n_ctx_q), device=q.device, dtype=torch.float32)
    Op = torch.empty((total_programs, n_ctx_q, d), device=q.device, dtype=torch.float32)

    locks = torch.zeros((total_programs,), device=q.device, dtype=torch.int32)
    XCD_REMAP = True

    # Triton LeanAttention output
    fn = lambda: _persistent_lean_attention(  # noqa: E731
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
        XCD_REMAP,
        causal,
        batch,
        RAGGED_BATCH,
        num_warps,
        waves_per_eu,
    )

    warmup = 1
    rep = 1

    ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)

    return ms


def main():
    bench_lean_attention.run(save_path=".", print_data=True)


if __name__ == "__main__":
    sys.exit(main())

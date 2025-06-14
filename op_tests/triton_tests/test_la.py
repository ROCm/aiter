# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import sys
import pytest
import torch

from aiter.ops.triton.lean_atten import persistent_lean_attention


@pytest.mark.parametrize(
    "causal, batch, h, n_ctx_q, n_ctx, d, total_programs, init_dtype, BLOCK_M, BLOCK_N, waves_per_eu, num_warps ",
    [
        (False, 1, 64, 16, [65536], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 64, 16, [131072], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 64, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 64, 16, [524288], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 96, 16, [32768], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 96, 16, [65536], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 96, 16, [131072], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 96, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 96, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
        (False, 1, 96, 16, [1048576], 16, 912, torch.float16, 16, 256, 1, 4),  #
        (False, 1, 128, 16, [32768], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 128, 16, [65536], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 128, 16, [131072], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 128, 16, [262144], 64, 912, torch.float16, 16, 64, 2, 4),
        (False, 1, 128, 16, [524288], 16, 912, torch.float16, 16, 256, 1, 4),  #
        (False, 3, 64, 16, [4096, 32768, 65536], 64, 912, torch.float16, 16, 64, 2, 4),
        (
            False,
            8,
            64,
            16,
            [1024, 1024, 2048, 2048, 4096, 4096, 32768, 65536],
            64,
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
            64,
            304,
            torch.float16,
            64,
            64,
            1,
            4,
        ),  # Causal=1,
        (True, 2, 64, 2048, [2048, 2048], 64, 304, torch.float16, 64, 64, 1, 4),
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
    la_out = persistent_lean_attention(
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


def main():
    batch = 2
    causal = True
    h = 4
    n_ctx_q = 1024
    n_ctx = [1024, 1024]
    d = 64
    total_programs = 32
    init_dtype = torch.float16
    BLOCK_M = 64
    BLOCK_N = 64
    waves_per_eu = 1
    num_warps = 4
    assert batch == len(n_ctx)

    try:
        sum_n_ctx = sum(int(n) for n in n_ctx)
    except ValueError:
        print(f"N_CTX contains non-numeric values: {n_ctx}")

    print(f"causal={causal}, batch={batch}")
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
    la_out = persistent_lean_attention(
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
    )

    # Calculate Pytorch refence output
    ref_out = torch.empty_like(q, dtype=v.dtype)
    start = 0
    start_q = 0

    for b in n_ctx:
        qb = q[start_q : (start_q + int(n_ctx_q)), :, :]
        print(f"qb shape: {qb.shape}")
        qb_reshaped = qb.transpose(0, 1)
        print(f"qb reshaped shape: {qb_reshaped.shape}")
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


if __name__ == "__main__":
    sys.exit(main())
#     benchmark_params = BenchmarkArgs()
#     args = benchmark_params.parse_args()
#     bench_streamk(args.m, args.n, args.k, args.total_programs_streamk, str_to_dtype(args.in_dtype), str_to_dtype(args.out_dtype), args.BLK_M, args.BLK_N, args.BLK_K, args.gsize_m)

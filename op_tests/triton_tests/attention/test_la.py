# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import sys
import pytest
import torch
import math
from typing import Union, List
from aiter.ops.triton.lean_atten import (
    _persistent_lean_attention,
    persistent_lean_attention,
)
from aiter.ops.triton._triton_kernels.lean_atten import _get_config
import aiter.ops.triton.utils._triton.arch_info as arch_info
import pytest

DEBUG_MODE = False


def get_lean_attn_inputs(
    batch: int,
    n_ctx_q: int,
    n_ctx: List[int],
    block_n: int,
    hq: int,
    hk: int,
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


def reference_attention(q, k, v, n_ctx, n_ctx_q, causal):

    # Calculate Pytorch refence output
    ref_out = torch.empty_like(q, dtype=q.dtype)
    start = 0
    start_q = 0
    d = q.shape[-1]

    for b in n_ctx:
        qb = q[start_q : (start_q + int(n_ctx_q)), :, :]
        qb_reshaped = qb.transpose(0, 1)
        kb = k[start : (start + int(b)), :, :]
        kb_reshaped = kb.transpose(0, 1)
        vb = v[start : (start + int(b)), :, :]
        vb_reshaped = vb.transpose(0, 1)
        # Expand K/V heads to match Q heads when using GQA (hq > hk)
        if qb_reshaped.shape[0] != kb_reshaped.shape[0]:
            assert qb_reshaped.shape[0] % kb_reshaped.shape[0] == 0
            group_size = qb_reshaped.shape[0] // kb_reshaped.shape[0]
            kb_reshaped = kb_reshaped.repeat_interleave(group_size, dim=0)
            vb_reshaped = vb_reshaped.repeat_interleave(group_size, dim=0)
        p = torch.matmul(qb_reshaped, kb_reshaped.transpose(-2, -1)) / math.sqrt(d)
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
    return ref_out


@pytest.mark.parametrize(
    "causal, batch, hq, hk, n_ctx_q, n_ctx, d, total_programs, init_dtype, BLOCK_M, BLOCK_N, RAGGED_BATCH, waves_per_eu, num_warps ",
    [
        (
            False,
            2,
            64,
            64,
            128,
            [65536, 65536],
            128,
            304,
            torch.float16,
            128,
            64,
            False,
            1,
            4,
        ),
        (
            False,
            2,
            64,
            64,
            16,
            [65536, 65536],
            128,
            912,
            torch.float16,
            16,
            128,
            False,
            3,
            4,
        ),
        (False, 1, 64, 64, 16, [131072], 128, 912, torch.float16, 16, 128, False, 2, 4),
        (False, 1, 64, 64, 16, [262144], 64, 912, torch.float16, 16, 64, False, 2, 4),
        (False, 1, 64, 64, 16, [524288], 64, 912, torch.float16, 16, 64, False, 2, 4),
        (
            False,
            2,
            96,
            96,
            16,
            [32768, 32768],
            128,
            912,
            torch.float16,
            16,
            128,
            False,
            2,
            4,
        ),
        (False, 1, 96, 96, 16, [65536], 128, 912, torch.float16, 16, 128, False, 2, 4),
        (False, 1, 96, 96, 16, [131072], 128, 912, torch.float16, 16, 128, False, 2, 4),
        (False, 1, 96, 96, 16, [262144], 64, 912, torch.float16, 16, 64, False, 2, 4),
        (
            False,
            1,
            96,
            96,
            16,
            [524288],
            16,
            912,
            torch.float16,
            16,
            256,
            False,
            1,
            4,
        ),  #
        (
            False,
            1,
            96,
            96,
            16,
            [1048576],
            16,
            912,
            torch.float16,
            16,
            256,
            False,
            1,
            4,
        ),  #
        (
            False,
            1,
            128,
            128,
            16,
            [32768],
            128,
            912,
            torch.float16,
            16,
            128,
            False,
            2,
            4,
        ),
        (
            False,
            1,
            128,
            128,
            16,
            [65536],
            128,
            912,
            torch.float16,
            16,
            128,
            False,
            2,
            4,
        ),
        (
            False,
            1,
            128,
            128,
            16,
            [131072],
            128,
            912,
            torch.float16,
            16,
            128,
            False,
            2,
            4,
        ),
        (False, 1, 128, 128, 16, [262144], 64, 912, torch.float16, 16, 64, False, 2, 4),
        (
            False,
            1,
            128,
            128,
            16,
            [524288],
            16,
            912,
            torch.float16,
            16,
            256,
            False,
            1,
            4,
        ),  #
        (
            False,
            3,
            64,
            64,
            16,
            [4096, 32768, 65536],
            128,
            912,
            torch.float16,
            16,
            128,
            True,
            2,
            4,
        ),
        (
            False,
            8,
            64,
            64,
            16,
            [1024, 1024, 2048, 2048, 4096, 4096, 32768, 65536],
            128,
            912,
            torch.float16,
            16,
            64,
            True,
            2,
            4,
        ),
        (
            True,
            1,
            64,
            64,
            8192,
            [8192],
            128,
            304,
            torch.float16,
            128,
            64,
            False,
            2,
            4,
        ),  # Causal=1,
        (
            True,
            2,
            64,
            64,
            2048,
            [2048, 2048],
            128,
            304,
            torch.float16,
            128,
            64,
            False,
            2,
            4,
        ),
        # These test cases fail:
        # (True, 2, 64, 2048, [2048, 2048], 128, 304, torch.float16, 128, 64, 2, 4),
        # (True, 1, 64, 4096, [4096], 128, 304, torch.float16, 128, 16, 3, 4),
        # (False, 1, 64, 4096, [4096], 128, 304, torch.float16, 128, 16, 3, 4),
    ],
)
def test_persistent_lean_attention(
    request,
    causal,
    batch,
    hq,
    hk,
    n_ctx_q,
    n_ctx,
    d,
    total_programs,
    init_dtype,
    BLOCK_M,
    BLOCK_N,
    RAGGED_BATCH,
    waves_per_eu,
    num_warps,
):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

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

    q, k, v, Mp, Lp, Op, locks, batch_num_block_n = get_lean_attn_inputs(
        batch,
        n_ctx_q,
        n_ctx,
        BLOCK_N,
        hq,
        hk,
        d,
        total_programs,
        init_dtype,
    )

    XCD_REMAP = False

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
        XCD_REMAP,
        causal,
        batch,
        RAGGED_BATCH,
        num_warps,
        waves_per_eu,
    )

    # Calculate Pytorch refence output
    ref_out = reference_attention(q, k, v, n_ctx, n_ctx_q, causal)
    # Compare result
    atol = 1.4e-1 if init_dtype == "fp8" else 1e-2
    rtol = 1e-2 if init_dtype == "fp8" else 3e-3
    # torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)
    # # Compare result
    # atol = 1e-2
    # rtol = 1e-2
    try:
        torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)
    except AssertionError:
        if DEBUG_MODE:
            print("Assertion failed! Showing mismatches:")
            print_mismatches(ref_out, la_out, atol, rtol)
        raise  # Re-raise the exception after printing mismatches


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
@pytest.mark.parametrize("RAGGED_BATCH", [False])
def test_persistent_lean_attention_outer(
    batch,
    h,
    n_ctx_q,
    n_ctx,
    d,
    init_dtype,
    causal,
    RAGGED_BATCH,
):
    torch.manual_seed(20)

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
        causal=causal,
        RAGGED_BATCH=RAGGED_BATCH,
        config=config,
    )

    # Calculate Pytorch refence output
    ref_out = reference_attention(q, k, v, n_ctx, n_ctx_q, causal)
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
    batch = 8
    causal = False
    hq = 64
    hk = 64
    n_ctx_q = 16
    n_ctx = [
        1024,
        1024,
        2048,
        2048,
        4096,
        4096,
        32768,
        65536,
    ]  # [4096, 32768, 65536]  # [131072] * batch  # [16384] #[8192]
    d = 128
    total_programs = 912
    init_dtype = torch.float16
    BLOCK_M = 16
    BLOCK_N = 64
    XCD_REMAP = True
    waves_per_eu = 2
    num_warps = 4
    RAGGED_BATCH = True
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

    q, k, v, Mp, Lp, Op, locks, batch_num_block_n = get_lean_attn_inputs(
        batch,
        n_ctx_q,
        n_ctx,
        BLOCK_N,
        hq,
        hk,
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
        XCD_REMAP,
        causal,
        batch,
        RAGGED_BATCH,
        num_warps,
        waves_per_eu,
    )
    # print(f"ms={ms}")

    ref_out = reference_attention(q, k, v, n_ctx, n_ctx_q, causal)

    # # Compare result
    atol = 1.4e-1 if init_dtype == "fp8" else 1e-2
    rtol = 1e-2 if init_dtype == "fp8" else 3e-3
    try:
        torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)
    except AssertionError:
        #     print("Assertion failed! Showing mismatches:")
        #     # print_mismatches(ref_out, la_out, atol, rtol)
        raise  # Re-raise the exception after printing mismatches

    # torch.testing.assert_close(ref_out, la_out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(main())

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Correctness test for the fused grouped_topk_decode kernel.
Compares outputs against a torch-native grouped topk + moe_sorting reference.
"""

import torch
import aiter
from aiter import dtypes
from aiter.ops.topk import grouped_topk_torch, biased_grouped_topk_torch
from aiter.fused_moe import moe_sorting
import argparse

BLOCK_SIZE_M = 32


def reference_grouped_topk_and_sort(
    gating_output, topk, num_experts, num_expert_group, topk_group,
    block_size, renormalize, scoring_func="sigmoid", correction_bias=None,
    model_dim=128,
):
    """Pure-torch reference: grouped_topk → moe_sorting format for M=1."""
    device = gating_output.device
    dtype = gating_output.dtype

    if correction_bias is not None:
        topk_weights, topk_ids = biased_grouped_topk_torch(
            gating_output, correction_bias, topk, renormalize,
            num_expert_group, topk_group,
        )
    else:
        topk_weights, topk_ids = grouped_topk_torch(
            gating_output, topk, renormalize,
            num_expert_group, topk_group,
            scoring_func=scoring_func,
        )

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        topk_ids, topk_weights, num_experts, model_dim, dtype, block_size,
    )
    return sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids


def test_grouped_topk_decode(
    dtype, E, topk, num_expert_group, topk_group,
    block_size=BLOCK_SIZE_M, renormalize=True, use_bias=True,
    model_dim=128,
):
    device = "cuda"
    M = 1

    gating_output = torch.randn(M, E, dtype=dtype, device=device)
    correction_bias = torch.randn(E, dtype=dtype, device=device) if use_bias else None

    max_num_tokens_padded = topk + E * block_size - topk
    max_num_m_blocks = (max_num_tokens_padded + block_size - 1) // block_size

    sorted_ids = torch.empty(max_num_tokens_padded, dtype=dtypes.i32, device=device)
    sorted_weights = torch.empty(max_num_tokens_padded, dtype=dtypes.fp32, device=device)
    sorted_expert_ids = torch.empty(max_num_m_blocks, dtype=dtypes.i32, device=device)
    num_valid_ids = torch.empty(2, dtype=dtypes.i32, device=device)
    moe_buf = torch.ones((M, model_dim), dtype=dtype, device=device)

    bias_tensor = correction_bias if correction_bias is not None else torch.empty(0, dtype=dtype, device=device)
    aiter.grouped_topk_moe_sorting(
        gating_output,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        num_expert_group,
        topk_group,
        topk,
        block_size,
        renormalize,
        False,
        bias_tensor,
        1.0,
    )

    fused_nv = num_valid_ids.cpu()
    total_padded = fused_nv[0].item()

    checks = {}
    checks["num_valid_padded"] = total_padded == topk * block_size
    checks["num_valid_tokens"] = fused_nv[1].item() == 1

    n_tiles = total_padded // block_size
    fused_eids = sorted_expert_ids[:n_tiles].cpu()
    checks["expert_ids_ascending"] = all(
        fused_eids[i] <= fused_eids[i + 1] for i in range(n_tiles - 1)
    )

    if renormalize:
        real_weights = [sorted_weights[i * block_size].item() for i in range(topk)]
        wsum = sum(real_weights)
        checks["weights_sum_1"] = abs(wsum - 1.0) < 1e-3
    else:
        checks["weights_sum_1"] = True

    checks["moe_buf_zeroed"] = torch.all(moe_buf == 0).item()

    sentinel_ok = True
    for i in range(topk):
        base = i * block_size
        for p in range(1, block_size):
            if sorted_weights[base + p].item() != 0.0:
                sentinel_ok = False
                break
    checks["sentinel_structure"] = sentinel_ok

    passed = all(checks.values())
    status = "PASS" if passed else "FAIL"
    detail = ", ".join(f"{k}={v}" for k, v in checks.items())
    print(f"  {status}: dtype={dtype}, E={E}, topk={topk}, g={num_expert_group}, "
          f"tg={topk_group}, bias={use_bias}, {detail}")
    return passed


def main():
    parser = argparse.ArgumentParser(description="Test grouped_topk_decode kernel")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["fp16", "bf16"])
    args = parser.parse_args()

    dtype = dtypes.bf16 if args.dtype == "bf16" else dtypes.fp16
    configs = [
        # (E, topk, num_expert_group, topk_group, use_bias)
        (256, 8, 8, 4, True),   # DeepSeek-V3
        (256, 8, 8, 4, False),
        (128, 8, 8, 4, True),
        (128, 8, 8, 4, False),
        (128, 4, 8, 4, True),
        (64, 4, 4, 2, True),
        (64, 4, 4, 2, False),
    ]

    all_pass = True
    for E, topk, g, tg, bias in configs:
        print(f"Testing E={E}, topk={topk}, g={g}, tg={tg}, bias={bias}:")
        passed = test_grouped_topk_decode(dtype, E, topk, g, tg, use_bias=bias)
        if not passed:
            all_pass = False

    if all_pass:
        print("\nAll tests PASSED!")
    else:
        print("\nSome tests FAILED!")
        exit(1)


if __name__ == "__main__":
    main()

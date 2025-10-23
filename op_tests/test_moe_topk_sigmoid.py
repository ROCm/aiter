# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import (
    checkAllclose,
    perftest,
)


@perftest(num_iters=10, num_warmup=1)
def run_torch(gating_output: torch.Tensor, topk: int):
    # llama4 maverick custom routing function
    router_scores, router_indices = torch.topk(gating_output, topk, dim=-1)
    router_scores = torch.sigmoid(router_scores.float())
    return router_scores, router_indices.to(torch.int32)


@perftest(num_iters=10, num_warmup=1)
def run_fused(gating_output: torch.Tensor, topk: int):
    tokens, _ = gating_output.shape
    router_scores = torch.empty(
        (tokens, topk), dtype=torch.float32, device=gating_output.device
    )
    router_indices = torch.empty(
        (tokens, topk), dtype=torch.int32, device=gating_output.device
    )
    aiter.topk_sigmoid(router_scores, router_indices, gating_output)
    return router_scores, router_indices


def test_topk_sigmoid(
    num_experts: int = 128,
    num_tokens: int = 1024,
    topk: int = 4,
    dtype: torch.dtype = torch.float16,
):
    # generate data - each row has only unique values
    gating_output = (
        torch.arange(-1, 1, 2.0 / num_experts)
        .repeat((num_tokens, 1))
        .to(dtype=dtype, device="cuda")
    )
    permutation = torch.argsort(torch.rand_like(gating_output), dim=-1)
    gating_output = torch.gather(gating_output, dim=-1, index=permutation)
    assert gating_output.is_contiguous()
    # run benchmarks
    (scores_torch, indices_torch), avg_torch = run_torch(gating_output.clone(), topk)
    (scores_fused, indices_fused), avg_fused = run_fused(gating_output.clone(), topk)
    # check correctness
    score_errors = checkAllclose(scores_torch, scores_fused, tol_err_ratio=0.01)
    index_errors = checkAllclose(indices_torch, indices_fused, tol_err_ratio=0.01)
    # print some failed rows
    if score_errors > 0.01 or index_errors > 0.01:
        failed_rows = (indices_torch != indices_fused).sum(dim=-1) > 0
        print("Wrong scores:")
        print(scores_torch[failed_rows][:5])
        print(scores_fused[failed_rows][:5])
        print("Wrong indices:")
        print(indices_torch[failed_rows][:5])
        print(indices_fused[failed_rows][:5])
        print("Gating outputs:")
        failed_values = gating_output[failed_rows][:5]
        failed_values, _ = failed_values.sort(dim=-1, descending=True)
        print(failed_values[:, :10])
        print(
            f"Number of wrong tokens: {sum(failed_rows)} / {len(failed_rows)}, {100 * sum(failed_rows) / len(failed_rows):.2f} %"
        )
    # print run times
    print(f"Runtime (torch baseline):     {avg_torch}")
    print(f"Runtime (fused topk sigmoid): {avg_fused}")
    print(f"Uplift:                       {avg_torch / avg_fused:.2f}x")


if __name__ == "__main__":
    test_topk_sigmoid(dtype=torch.float16)
    test_topk_sigmoid(dtype=torch.bfloat16)

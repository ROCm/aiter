# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Inputs and FP32 reference shared by HK sparse MLA tests and benchmarks."""

import torch

from aiter.ops.hk_sparse_mla import DIM, HEADS


def rand_fp8(shape):
    return (torch.randn(shape, device="cuda") * 0.4).to(torch.float8_e4m3fn)


def make_inputs(T, num_slots, topk_real, topk, row_bytes):
    """Random fp8 q / kv; topk_real random indices, then 3 live tail slots and -1 padding up to topk."""
    q = rand_fp8((T, HEADS, DIM))
    kv = rand_fp8((num_slots, 1, row_bytes))
    idx = torch.randint(0, num_slots, (T, topk), device="cuda", dtype=torch.int32)
    idx[:, topk_real:] = -1
    if topk > topk_real:
        idx[:, topk_real : topk_real + 3] = torch.randint(
            0, num_slots, (T, 3), device="cuda", dtype=torch.int32
        )
    idx[:, ::7] = -1  # scattered masked keys
    return q, kv, idx


def reference(q, kv, idx, sm_scale, rows):
    out = []
    for t in rows:
        valid = (idx[t] >= 0) & (idx[t] < kv.shape[0])
        k = kv[idx[t].clamp(0, kv.shape[0] - 1).long(), 0, :DIM].float()
        s = (q[t].float() @ k.T) * sm_scale
        s[:, ~valid] = -float("inf")
        out.append(torch.softmax(s, -1).nan_to_num() @ k)
    return torch.stack(out)


def rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()

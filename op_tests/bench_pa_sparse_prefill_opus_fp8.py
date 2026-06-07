# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark: bf16 baseline vs fp8 dequant-prepass vs fp8 fused, same problem."""
from __future__ import annotations
import math
import sys
import pandas as pd
import torch

import aiter  # noqa
from aiter.ops.pa_sparse_prefill_opus import (
    pa_sparse_prefill_opus,
    pa_sparse_prefill_opus_fp8,
    pa_sparse_prefill_opus_fp8_fused,
)
from aiter.test_common import run_perftest

sys.path.insert(0, "op_tests")
from test_pa_sparse_prefill_opus_fp8 import _make_fp8_inputs  # noqa: E402
from pa_sparse_prefill_opus_fp8_quant import dequantize_v4_fp8, D_FULL  # noqa: E402


def bench_point(n, h, pages, tokens, mode):
    inp = _make_fp8_inputs(n, h, pages, tokens, mode=mode, seed=0)
    ss = 1.0 / math.sqrt(D_FULL)
    # bf16 baseline operates on dequanted inputs.
    q = dequantize_v4_fp8(inp["q_nope"], inp["q_rope"], inp["q_scale"])
    ukv = dequantize_v4_fp8(inp["ukv_nope"], inp["ukv_rope"], inp["ukv_scale"])
    kv = dequantize_v4_fp8(inp["kv_nope"], inp["kv_rope"], inp["kv_scale"])

    _, us_bf16 = run_perftest(
        pa_sparse_prefill_opus, q, ukv, inp["kv_indices_prefix"], inp["kv_indptr_prefix"],
        kv, inp["kv_indices_extend"], inp["kv_indptr_extend"], inp["attn_sink"], ss,
        num_iters=50, num_warmup=5,
    )
    _, us_fused = run_perftest(
        pa_sparse_prefill_opus_fp8_fused,
        inp["q_nope"], inp["q_rope"], inp["q_scale"],
        inp["ukv_nope"], inp["ukv_rope"], inp["ukv_scale"],
        inp["kv_nope"], inp["kv_rope"], inp["kv_scale"],
        inp["kv_indices_prefix"], inp["kv_indptr_prefix"],
        inp["kv_indices_extend"], inp["kv_indptr_extend"], inp["attn_sink"], ss,
        num_iters=50, num_warmup=5,
    )
    nnz = int(inp["kv_indices_prefix"].numel()) + int(inp["kv_indices_extend"].numel())
    return {
        "N": n, "H": h, "mode": mode, "nnz": nnz,
        "bf16_us": round(us_bf16, 1), "fused_us": round(us_fused, 1),
        "fused/bf16": round(us_fused / us_bf16, 2),
    }


if __name__ == "__main__":
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx950":
        print(f"SKIP: needs gfx950, got {arch}"); sys.exit(0)
    rows = []
    for (n, h, pages, tokens) in [(1024, 128, 4096, 1024), (4096, 128, 16384, 4096),
                                  (1024, 16, 4096, 1024)]:
        for mode in ["dense", "sparse"]:
            rows.append(bench_point(n, h, pages, tokens, mode))
            print(rows[-1])
    print()
    print(pd.DataFrame(rows).to_string(index=False))

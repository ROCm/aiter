# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runnable correctness test for the gfx1250-only split-precision
(NoPE fp8 / RoPE bf16) sparse paged prefill attention kernel.

Reuses the reference + input packing from the gfx950 fp8 test.

Run:
  GPU=0 aiter_run 'cd /app/aiter && PYTHONPATH=. python3 \
      op_tests/test_pa_sparse_prefill_fp8_opus_gfx1250.py'
"""

import sys
import torch

import aiter.ops.pa_sparse_prefill_fp8_opus_gfx1250 as mod
from op_tests.test_pa_sparse_prefill_opus import (
    _ref_pa_sparse_prefill_fp8,
    _make_inputs_fp8,
)


def run_case(n, h, total_pages, total_tokens, mode, seed=0):
    data = _make_inputs_fp8(
        n, h, total_pages, total_tokens, mode=mode, seed=seed, device="cuda"
    )
    kern = data["kernel"]
    ref_inputs = data["ref"]
    softmax_scale = 1.0 / (512.0**0.5)

    out = mod.pa_sparse_prefill_fp8_opus_gfx1250(
        kern["q_nope"],
        kern["q_rope"],
        kern["unified_kv_nope"],
        kern["unified_kv_rope"],
        kern["kv_indices_prefix"],
        kern["kv_indptr_prefix"],
        kern["kv_nope"],
        kern["kv_rope"],
        kern["kv_indices_extend"],
        kern["kv_indptr_extend"],
        kern["attn_sink"],
        softmax_scale,
    )

    ref = _ref_pa_sparse_prefill_fp8(
        ref_inputs["q_fp32"],
        ref_inputs["ukv_fp32"],
        ref_inputs["kv_fp32"],
        ref_inputs["kv_indices_prefix"],
        ref_inputs["kv_indptr_prefix"],
        ref_inputs["kv_indices_extend"],
        ref_inputs["kv_indptr_extend"],
        ref_inputs["attn_sink"],
        softmax_scale,
    )

    out_f = out.to(torch.float32)
    ref_f = ref.to(torch.float32)
    diff = (out_f - ref_f).abs()
    max_abs = diff.max().item()
    atol = rtol = 3e-2
    ok = torch.allclose(out_f, ref_f, atol=atol, rtol=rtol)
    print(
        f"[{mode:6s}] N={n} H={h} pages={total_pages} tokens={total_tokens} "
        f"max_abs_diff={max_abs:.4e}  {'PASS' if ok else 'FAIL'}"
    )
    return ok


def main():
    torch.cuda.init()
    cases = [
        # (n, h, total_pages, total_tokens, mode)
        (4, 16, 64, 64, "empty"),  # sink-only output
        (4, 16, 64, 64, "dense"),  # single head-block, single tile
        (8, 16, 64, 64, "dense"),
        (4, 32, 64, 64, "dense"),  # H=32 -> 2 head-blocks
        (4, 16, 100, 100, "sparse"),  # multi-tile + OOB-token masking
        (8, 32, 256, 256, "sparse"),  # larger: multi head-block + multi-tile sparse
    ]
    if len(sys.argv) > 1:
        # quick override: mode only
        cases = [(4, 16, 64, 64, sys.argv[1])]
    all_ok = True
    for c in cases:
        try:
            all_ok &= run_case(*c)
        except Exception as e:
            print(f"[{c[4]:6s}] N={c[0]} H={c[1]} EXCEPTION: {e}")
            all_ok = False
    print("ALL PASS" if all_ok else "SOME FAILED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

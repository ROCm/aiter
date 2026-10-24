# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One FA forward call in a fresh process.

ck_tile caches CK_TILE_FMHA_* in function-local statics, so grouped vs
DISABLE=1 must be separate processes. Callers compare --save tensors.
"""

import argparse  # noqa: I001
import sys

# triton-first justification: aiter/__init__.py getLogger() ->
# torch._dynamo.config.ignore_logger_methods lazily imports torch._dynamo,
# which imports triton via has_triton_package(). On ROCm 10 + torch 2.13 +
# triton 3.8 that SIGSEGVs (139) in triton._C.libtriton (knobs.py create_module)
# unless triton is already loaded. Reproduced without aiter:
# `import torch; torch._dynamo` is also 139. gh unavailable; aiter/__init__.py
# is out of scope for this C4 test-only change.
import triton  # noqa: F401
import torch

import aiter
from aiter import dtypes, per_tensor_quant


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", choices=["fp8bf16", "bf16", "fp16"], required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--nheads", type=int, required=True)
    p.add_argument("--nheads-k", type=int, required=True)
    p.add_argument("--seqlen", type=int, required=True)
    p.add_argument("--hdim", type=int, default=128)
    p.add_argument("--causal", action="store_true")
    p.add_argument(
        "--lse",
        action="store_true",
        help="also request softmax_lse (bf16/fp16 only, run_fwd_head_grouped "
        "does per-head lse_ptr offset math that no other flag exercises)",
    )
    p.add_argument("--save", default=None)
    a = p.parse_args()
    if a.lse and a.dtype == "fp8bf16":
        p.error(
            "--lse is not supported with --dtype fp8bf16 (flash_attn_fp8_pertensor_func "
            "has no return_lse parameter)"
        )
    return a


def _make_inputs(a):
    torch.manual_seed(0)
    base = dtypes.bf16 if a.dtype == "fp8bf16" else getattr(dtypes, a.dtype)
    q = torch.randn(a.batch, a.seqlen, a.nheads, a.hdim, dtype=base, device="cuda")
    k = torch.randn(a.batch, a.seqlen, a.nheads_k, a.hdim, dtype=base, device="cuda")
    v = torch.randn(a.batch, a.seqlen, a.nheads_k, a.hdim, dtype=base, device="cuda")
    return q, k, v


def _build_call(a, q, k, v):
    if a.dtype == "fp8bf16":
        q8, qs = per_tensor_quant(q, quant_dtype=dtypes.fp8)
        k8, ks = per_tensor_quant(k, quant_dtype=dtypes.fp8)
        v8, vs = per_tensor_quant(v, quant_dtype=dtypes.fp8)
        return lambda: aiter.flash_attn_fp8_pertensor_func(
            q8, k8, v8, qs, ks, vs, causal=a.causal
        )

    return lambda: aiter.flash_attn_func(q, k, v, causal=a.causal, return_lse=a.lse)


def main():
    a = _parse()
    q, k, v = _make_inputs(a)
    call = _build_call(a, q, k, v)

    out = call()
    torch.cuda.synchronize()

    if a.save:
        if a.lse:
            out_t, lse_t = out
            torch.save((out_t.detach().cpu(), lse_t.detach().cpu()), a.save)
        else:
            torch.save(out.detach().cpu(), a.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())

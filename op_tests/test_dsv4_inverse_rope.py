# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit test for the DeepSeek-V4 inverse-RoPE kernel (aiter port of ATOM's
atom/model_ops/v4_kernels/inverse_rope.py)."""

import argparse
import sys

import torch

from aiter import dtypes
from aiter.ops.triton.dsv4.inverse_rope import (
    inverse_rope_inplace,
    inverse_rope_reference,
)
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")
torch.manual_seed(0)


def _build_cos_sin(max_pos, rd):
    # arbitrary smooth cos/sin cache shaped [max_pos, 1, 1, rd//2]
    inv = 1.0 / (10000.0 ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    t = torch.arange(max_pos, dtype=torch.float32)
    freqs = torch.outer(t, inv)
    cos = freqs.cos().to(dtypes.bf16).unsqueeze(-2).unsqueeze(-2)
    sin = freqs.sin().to(dtypes.bf16).unsqueeze(-2).unsqueeze(-2)
    return cos, sin


def run(num_tokens, n_heads, rd, max_pos, atol, rtol):
    cos, sin = _build_cos_sin(max_pos, rd)
    x = (torch.randn(num_tokens, n_heads, rd) * 0.3).to(dtypes.bf16)
    positions = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64)

    ref = inverse_rope_reference(x, cos, sin, positions)
    out = x.clone()
    inverse_rope_inplace(out, cos, sin, positions)
    err = checkAllclose(
        ref,
        out,
        rtol=rtol,
        atol=atol,
        msg=f"inverse_rope T{num_tokens} H{n_heads} rd{rd}",
    )
    return err


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 17, 64])
    p.add_argument("--heads", type=int, default=128)
    p.add_argument("--rope-dim", type=int, default=64)
    p.add_argument("--max-pos", type=int, default=4096)
    p.add_argument("--atol", type=float, default=2e-2)
    p.add_argument("--rtol", type=float, default=2e-2)
    args = p.parse_args()
    fail = 0
    for t in args.tokens:
        err = run(t, args.heads, args.rope_dim, args.max_pos, args.atol, args.rtol)
        fail += 0 if (err == 0 or (isinstance(err, float) and err < 0.02)) else 1
    if fail:
        print(f"{fail} case(s) FAILED")
        sys.exit(1)
    print("All cases passed.")


if __name__ == "__main__":
    main()

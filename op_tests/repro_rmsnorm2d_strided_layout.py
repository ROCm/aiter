#!/usr/bin/env python3
"""
Reproducer for Aiter rmsnorm2d_fwd layout handling on gfx942.

The public rmsnorm2d_fwd API should accept higher-rank strided views such as
Q/K tensors sliced from a packed QKV projection. Older behavior passed those
views directly to the low-level 2D HIP kernel and could memory-fault on MI325
(gfx942).

Safe fixed path:
    python op_tests/repro_rmsnorm2d_strided_layout.py --mode fixed

Compare fixed public API against the old low-level path in an isolated child:
    python op_tests/repro_rmsnorm2d_strided_layout.py --mode both
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys


GPUCORE_GLOBS = (
    "/sgl-workspace/sglang/gpucore.*",
    "/sgl-workspace/gpucore.*",
    "/tmp/gpucore.*",
    "/root/gpucore.*",
)


def cleanup_gpucore() -> None:
    for pattern in GPUCORE_GLOBS:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass


def make_strided_qk():
    import torch

    tokens = 2048
    q_dim = 4096
    kv_dim = 1024
    head_dim = 128
    packed_dim = q_dim + 2 * kv_dim

    qkv = torch.randn((tokens, packed_dim), device="cuda", dtype=torch.bfloat16)
    q = qkv[:, :q_dim].view(tokens, q_dim // head_dim, head_dim)
    k = qkv[:, q_dim : q_dim + kv_dim].view(tokens, kv_dim // head_dim, head_dim)
    return q, k


def describe(name, tensor) -> None:
    print(
        f"{name}: shape={tuple(tensor.shape)} stride={tuple(tensor.stride())} "
        f"contiguous={tensor.is_contiguous()} dtype={tensor.dtype}",
        flush=True,
    )


def run_fixed() -> None:
    import torch
    from aiter.ops.rmsnorm import rmsnorm2d_fwd

    q, k = make_strided_qk()
    describe("q_input", q)
    describe("k_input", k)

    q_weight = torch.ones((q.shape[-1],), device="cuda", dtype=torch.bfloat16)
    k_weight = torch.ones((k.shape[-1],), device="cuda", dtype=torch.bfloat16)

    q_out = rmsnorm2d_fwd(q, q_weight, 1e-6)
    k_out = rmsnorm2d_fwd(k, k_weight, 1e-6)
    torch.cuda.synchronize()

    describe("q_output", q_out)
    describe("k_output", k_out)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    print("FIXED_PUBLIC_API_OK", flush=True)


def run_old_low_level_path() -> None:
    import torch
    from aiter.ops.rmsnorm import rmsnorm

    q, k = make_strided_qk()
    describe("q_input", q)
    describe("k_input", k)

    q_weight = torch.ones((q.shape[-1],), device="cuda", dtype=torch.bfloat16)
    k_weight = torch.ones((k.shape[-1],), device="cuda", dtype=torch.bfloat16)
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    print(
        "Calling low-level rmsnorm(out, input, weight, eps) directly on strided "
        "3D views. This represents the pre-fix public-wrapper behavior and may "
        "abort on gfx942.",
        flush=True,
    )
    rmsnorm(q_out, q, q_weight, 1e-6)
    rmsnorm(k_out, k, k_weight, 1e-6)
    torch.cuda.synchronize()
    print("OLD_LOW_LEVEL_PATH_DID_NOT_CRASH", flush=True)


def run_both() -> int:
    print("=== Fixed public API path ===", flush=True)
    run_fixed()

    print("\n=== Old low-level path in subprocess ===", flush=True)
    proc = subprocess.run([sys.executable, __file__, "--mode", "old-low-level"])
    cleanup_gpucore()
    print(f"OLD_LOW_LEVEL_SUBPROCESS_EXIT_CODE={proc.returncode}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("fixed", "old-low-level", "both"),
        default="fixed",
        help="'old-low-level' may crash on gfx942; 'both' isolates it in a child process.",
    )
    args = parser.parse_args()

    if args.mode == "fixed":
        run_fixed()
    elif args.mode == "old-low-level":
        run_old_low_level_path()
    else:
        run_both()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

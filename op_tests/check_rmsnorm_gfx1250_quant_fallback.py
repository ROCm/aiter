#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare FP8 fallback outputs from two source revisions.

Collect against each checkout with PYTHONPATH and a separate AITER_JIT_DIR:
  python check_rmsnorm_gfx1250_quant_fallback.py --out before
  python check_rmsnorm_gfx1250_quant_fallback.py --out after
  python check_rmsnorm_gfx1250_quant_fallback.py --compare before after

Cases and paths are isolated in subprocesses. The fused path calls raw pybind,
using aiter's canonical tensor conversion/stream; this is kernel regression
coverage, not a test of the higher-level Python wrapper.
"""
import argparse
import itertools
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("AITER_AOT_IMPORT", "1")
os.environ.setdefault("GPU_ARCHS", "gfx1250")


def run_case(case, path, output):
    import torch
    import aiter
    from aiter.utility import dtypes
    from aiter.jit import core
    from aiter.ops.rmsnorm import rmsnorm_quant, add_rmsnorm_quant

    aiter.dtypes = dtypes
    n, add, gemma, group = list(
        itertools.product((128, 6144), (False, True), (False, True), (32, 128))
    )[case]
    assert torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] == "gfx1250"
    torch.manual_seed(1950 + n + add + gemma + group)
    x = torch.randn(7, n, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(n, device="cuda", dtype=torch.bfloat16)
    r = torch.randn_like(x)
    y = torch.empty(7, n, device="cuda", dtype=dtypes.fp8)
    scale = torch.empty(7, n // group, device="cuda", dtype=torch.float32)
    ro = torch.empty_like(x)
    if path == "generic":
        if add:
            add_rmsnorm_quant(y, x, r, ro, scale, w, 1e-6, group, False, gemma)
        else:
            rmsnorm_quant(y, x, scale, w, 1e-6, group, False, gemma)
    else:
        # This collector expects the same prebuilt JIT modules as the benchmark.
        module = core.get_module("module_fused_qk_rmsnorm_group_quant")
        convert, _, raw_stream, current_device = core._pybind_develop_hooks()
        module._set_current_hip_stream(raw_stream(current_device()))
        module.fused_qk_rmsnorm_group_quant(
            convert(y),
            convert(scale),
            convert(x),
            convert(w),
            1e-6,
            None,
            None,
            convert(ro) if add else None,
            None,
            None,
            None,
            convert(r) if add else None,
            group,
            False,
            gemma,
        )
    torch.cuda.synchronize()
    torch.save(
        {
            "y": y.view(torch.uint8).cpu(),
            "scale": scale.cpu(),
            "residual": ro.cpu() if add else None,
        },
        output,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--case", type=int, choices=range(16))
    parser.add_argument("--path", choices=("generic", "atom"))
    args = parser.parse_args()
    if args.compare:
        import torch

        for i in range(16):
            for path in ("generic", "atom"):
                name = f"{path}_{i}.pt"
                a, b = [torch.load(p / name, weights_only=True) for p in args.compare]
                for key in a:
                    if a[key] is None:
                        assert b[key] is None
                    else:
                        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        print("32 quantized fallback cases: exact match")
    elif args.out is None:
        parser.error("--out or --compare is required")
    elif args.case is not None:
        if args.path is None:
            parser.error("--case requires --path")
        run_case(args.case, args.path, args.out)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        for i in range(16):
            for path in ("generic", "atom"):
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "--case",
                        str(i),
                        "--path",
                        path,
                        "--out",
                        str(args.out / f"{path}_{i}.pt"),
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()

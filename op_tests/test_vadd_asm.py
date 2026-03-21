# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pytest
import torch

import aiter
from aiter.ops.vadd_asm import vadd_asm
from aiter.test_common import run_perftest

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="vadd asm requires ROCm/HIP GPU"
)


def test_vadd_correctness():
    height, width = 128, 128
    a = torch.randn(height, width, device="cuda", dtype=torch.float32)
    b = torch.randn(height, width, device="cuda", dtype=torch.float32)
    c_asm = torch.empty_like(a)
    c_ref = a + b

    vadd_asm(a, b, c_asm)
    torch.cuda.synchronize()
    assert torch.allclose(c_ref, c_asm, atol=1e-5, rtol=1e-5)

    out = aiter.fused_vadd(a, b)
    assert torch.allclose(c_ref, out, atol=1e-5, rtol=1e-5)


def test_vadd_perf():
    height, width = 4096, 4096
    a = torch.randn(height, width, device="cuda", dtype=torch.float32)
    b = torch.randn(height, width, device="cuda", dtype=torch.float32)
    c = torch.empty_like(a)

    def run_asm():
        vadd_asm(a, b, c)

    # 仅作示例：打印平均耗时（微秒级），需 GPU
    run_perftest(run_asm, num_iters=20, num_warmup=5)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--perf", action="store_true", help="run perf smoke test")
    args = p.parse_args()
    test_vadd_correctness()
    print("vadd correctness ok")
    if args.perf:
        test_vadd_perf()
        print("vadd perf done")

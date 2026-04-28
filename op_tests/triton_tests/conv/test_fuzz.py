# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations
import random
import traceback
import torch
from aiter.ops.triton.conv._utils import _out_hw
from .suite import TestSuite, TestResult, run_all_methods


def test_random_fuzzing(
    suite: TestSuite,
    num_tests=200,
    activation="none",
    method: str = "default",
    seed: int = 42,
):
    print("\n" + "=" * 80)
    print(
        f"RANDOM FUZZING TESTS (n={num_tests}, activation={activation}, method={method})"
    )
    print("=" * 80)
    random.seed(seed)
    for i in range(num_tests):
        N = random.randint(1, 8)
        C = random.choice([1, 3, 16, 32, 64, 128, 256])
        H = random.randint(4, 64)
        W = random.randint(4, 64)
        K_out = random.choice([16, 32, 64, 128, 256])
        R = random.randint(1, min(7, H))
        S = random.randint(1, min(7, W))
        sh = random.randint(1, 3)
        sw = random.randint(1, 3)
        ph = random.randint(0, R // 2)
        pw = random.randint(0, S // 2)
        dh = random.randint(1, 2)
        dw = random.randint(1, 2)
        P, Q = _out_hw(H, W, R, S, (sh, sw), (ph, pw), (dh, dw))
        if P < 1 or Q < 1:
            continue
        try:
            x = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
            w = torch.randn((K_out, C, R, S), device=suite.device, dtype=suite.dtype)
            b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
            tag = f"Random[{i}] ({N},{C},{H},{W})->({N},{K_out},{P},{Q})"
            run_all_methods(
                suite,
                x,
                w,
                b,
                (sh, sw),
                (ph, pw),
                (dh, dw),
                name=tag,
                method=method,
                activation=activation,
            )
        except Exception as e:
            tb = traceback.format_exc()
            print(f"  ✗ Random[{i}] EXCEPTION: {type(e).__name__}: {e}")
            for line in tb.rstrip().splitlines():
                print(f"      {line}")
            suite.results.append(
                TestResult(f"Random[{i}]", False, float("inf"), float("inf"), tb)
            )

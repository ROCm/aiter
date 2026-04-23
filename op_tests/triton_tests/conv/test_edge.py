# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations
import torch
from aiter.ops.triton.conv._utils import _out_hw
from .suite import TestSuite, run_all_methods


def get_edge_case_shapes():
    return [
        (1, 3, 7, 7, 8, 3, 3, (1, 1), (1, 1), (1, 1), "3x3 same padding"),
        (1, 3, 8, 8, 16, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 stride1"),
        (2, 16, 32, 32, 32, 3, 3, (2, 2), (1, 1), (1, 1), "stride2"),
        (2, 32, 17, 23, 64, 5, 5, (2, 2), (2, 2), (1, 1), "odd dims + pad"),
        (4, 64, 28, 28, 128, 3, 3, (1, 1), (0, 0), (2, 2), "dilation2"),
        (2, 512, 7, 7, 1024, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 large channels"),
        (1, 3, 112, 112, 64, 7, 7, (2, 2), (3, 3), (1, 1), "7x7 large spatial"),
        (1, 1, 16, 16, 16, 3, 3, (1, 1), (1, 1), (1, 1), "single input channel"),
        (2, 64, 8, 8, 64, 3, 3, (1, 1), (1, 1), (1, 1), "small spatial 3x3"),
        (1, 128, 4, 4, 256, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 tiny spatial"),
        (2, 32, 32, 32, 32, 3, 3, (1, 1), (0, 0), (1, 1), "3x3 no padding"),
        (2, 64, 28, 28, 128, 3, 3, (2, 2), (1, 1), (1, 1), "3x3 stride2 standard"),
    ]


def test_edge_cases(
    suite: TestSuite, activation: str = "none", method: str = "default"
):
    print("\n" + "=" * 80)
    print(f"EDGE CASE TESTS (activation={activation}, method={method})")
    print("=" * 80)
    for (
        N,
        C,
        H,
        W,
        K_out,
        R,
        S,
        stride,
        padding,
        dilation,
        desc,
    ) in get_edge_case_shapes():
        P, Q = _out_hw(H, W, R, S, stride, padding, dilation)
        if P < 1 or Q < 1:
            continue
        x = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
        w = torch.randn((K_out, C, R, S), device=suite.device, dtype=suite.dtype)
        b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
        run_all_methods(
            suite,
            x,
            w,
            b,
            stride,
            padding,
            dilation,
            name=desc,
            method=method,
            activation=activation,
        )


def test_activations(
    suite: TestSuite, method: str = "default", activation: str = "relu"
):
    print("\n" + "=" * 80)
    print(f"ACTIVATION TEST (method={method}, activation={activation})")
    print("=" * 80)
    N, C, H, W, K_out = 2, 32, 16, 16, 64
    R, S = 3, 3
    stride, padding, dilation = (1, 1), (1, 1), (1, 1)
    x = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
    w = torch.randn((K_out, C, R, S), device=suite.device, dtype=suite.dtype)
    b = torch.randn((K_out,), device=suite.device, dtype=suite.dtype)
    run_all_methods(
        suite,
        x,
        w,
        b,
        stride,
        padding,
        dilation,
        name=f"activation_{activation}_{method}",
        method=method,
        activation=activation,
    )


def test_no_bias(suite: TestSuite, method: str = "default"):
    print("\n" + "=" * 80)
    print(f"NO-BIAS TEST (method={method})")
    print("=" * 80)
    shapes = [
        (1, 64, 8, 8, 128, 1, 1, (1, 1), (0, 0), (1, 1), "1x1 no bias"),
        (2, 32, 16, 16, 64, 3, 3, (1, 1), (1, 1), (1, 1), "3x3 no bias"),
        (1, 16, 8, 8, 32, 5, 5, (1, 1), (2, 2), (1, 1), "5x5 no bias"),
    ]
    for N, C, H, W, K_out, R, S, stride, padding, dilation, desc in shapes:
        x = torch.randn((N, C, H, W), device=suite.device, dtype=suite.dtype)
        w = torch.randn((K_out, C, R, S), device=suite.device, dtype=suite.dtype)
        run_all_methods(
            suite, x, w, None, stride, padding, dilation, name=desc, method=method
        )

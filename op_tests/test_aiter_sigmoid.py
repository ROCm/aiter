# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
from dataclasses import dataclass

import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.aiter_operator import _unary_tile_supported


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}

OPERATIONS = {
    "sigmoid": (aiter.sigmoid, torch.sigmoid),
    "tanh": (aiter.tanh, torch.tanh),
}


@dataclass(frozen=True)
class Case:
    shape: tuple[int, ...]
    native: bool
    noncontiguous: bool = False


CASES = (
    # Native tile paths, including the 3-D M dimension.
    Case((8, 8), True),
    Case((32, 1024), True),
    Case((2, 16, 128), True),
    # Public-API fallbacks for unsupported dimensions, tails, and layouts.
    Case((17,), False),
    Case((7, 9), False),
    Case((2, 8, 8, 8), False),
    Case((2, 8, 8), False, noncontiguous=True),
)


def make_input(case: Case, dtype: torch.dtype) -> torch.Tensor:
    x = torch.randn(case.shape, dtype=dtype, device="cuda") * 3
    if case.noncontiguous:
        x = x.transpose(-1, -2)
        assert not x.is_contiguous()
    return x


def run_case(op_name: str, dtype: torch.dtype, case: Case) -> float:
    aiter_op, torch_op = OPERATIONS[op_name]
    x = make_input(case, dtype)
    native = _unary_tile_supported(x)
    assert native == case.native, (
        f"routing mismatch for {op_name}, dtype={dtype}, shape={case.shape}, "
        f"noncontiguous={case.noncontiguous}: expected native={case.native}, got {native}"
    )

    expected = torch_op(x)
    actual = aiter_op(x)
    tolerance = 1e-2 if dtype in (torch.float16, torch.bfloat16) else 1e-4
    torch.testing.assert_close(
        actual, expected, atol=tolerance, rtol=tolerance
    )
    return float((actual.float() - expected.float()).abs().max().item())


def run_guarded_fallbacks() -> int:
    inputs = {
        "cpu": torch.randn((8, 8), dtype=torch.float32),
        "unsupported_dtype": torch.randn(
            (8, 8), dtype=torch.float64, device="cuda"
        ),
        "empty": torch.empty((0, 8), dtype=torch.float32, device="cuda"),
    }
    passed = 0
    for case_name, x in inputs.items():
        assert not _unary_tile_supported(x), (
            f"guarded case {case_name} unexpectedly selected the native kernel"
        )
        for op_name, (aiter_op, torch_op) in OPERATIONS.items():
            torch.testing.assert_close(aiter_op(x), torch_op(x))
            passed += 1
            print(f"PASS op={op_name} guarded_fallback={case_name}")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate AITER native unary tiles and public fallbacks"
    )
    parser.add_argument(
        "-d",
        "--dtype",
        nargs="+",
        choices=DTYPES,
        default=["bf16"],
    )
    parser.add_argument(
        "--op",
        nargs="+",
        choices=OPERATIONS,
        default=list(OPERATIONS),
    )
    parser.add_argument(
        "--mode",
        choices=("all", "native", "fallback"),
        default="all",
    )
    args = parser.parse_args()

    selected_cases = [
        case
        for case in CASES
        if args.mode == "all" or case.native == (args.mode == "native")
    ]
    failures = []
    passed = run_guarded_fallbacks()
    print(f"GPU: {torch.cuda.get_device_name()} ({get_gfx()})")
    for dtype_name in args.dtype:
        for op_name in args.op:
            for case in selected_cases:
                try:
                    max_abs = run_case(op_name, DTYPES[dtype_name], case)
                except Exception as error:
                    failures.append((op_name, dtype_name, case, error))
                    print(
                        f"FAIL op={op_name} dtype={dtype_name} shape={case.shape} "
                        f"native={case.native}: {error}"
                    )
                else:
                    passed += 1
                    print(
                        f"PASS op={op_name} dtype={dtype_name} shape={case.shape} "
                        f"native={case.native} max_abs={max_abs:.6g}"
                    )

    print(f"unary summary: {passed} passed, {len(failures)} failed")
    if failures:
        raise AssertionError(f"AITER unary correctness failures: {len(failures)}")


if __name__ == "__main__":
    main()

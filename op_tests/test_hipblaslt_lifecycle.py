# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run in a fresh process on an idle GPU; --other-device opts into two GPUs."""

import argparse

import torch

import aiter
from aiter.test_common import checkAllclose

MIB = 1024 * 1024
WORKSPACE_BYTES = 256 * MIB
MEMORY_TOLERANCE = 16 * MIB


def free_memory() -> int:
    torch.cuda.synchronize()
    return torch.cuda.mem_get_info()[0]


def check_gemm() -> None:
    for dtype in (torch.float16, torch.bfloat16):
        for m, n, k in ((16, 64, 64), (33, 128, 256)):
            x = torch.randint(-2, 3, (m, k), device="cuda").to(dtype)
            w = torch.randint(-2, 3, (n, k), device="cuda").to(dtype)
            for use_bias in (False, True):
                bias = torch.ones(n, dtype=dtype, device="cuda") if use_bias else None
                ref = x.float() @ w.float().t()
                if bias is not None:
                    ref += bias.float()
                ref = ref.to(dtype)
                solutions = aiter.hipb_findallsols(x, w.t(), bias, dtype)
                assert solutions, f"No hipBLASLt solutions for {(m, n, k, dtype)}"
                for solution in (-1, solutions[0]):
                    result = aiter.hipb_mm(x, w.t(), solution, bias, dtype)
                    # Integer inputs make the accumulation exact before output casting.
                    error = checkAllclose(ref, result, rtol=0, atol=0, printLog=False)
                    assert error == 0, (m, n, k, dtype, use_bias, solution, error)
    torch.cuda.synchronize()


def test_repeated_init(iterations: int) -> None:
    aiter.hipb_create_extension()
    try:
        # Exclude first-use HIP/hipBLASLt and torch allocator overhead.
        check_gemm()
        torch.cuda.empty_cache()
        before = free_memory()
        for _ in range(iterations):
            aiter.hipb_create_extension()
        after = free_memory()
        print(f"Repeated init: before={before}, after={after}, loss={before - after}")
        assert before - after < MEMORY_TOLERANCE, "Repeated init leaked device memory"
        check_gemm()
    finally:
        torch.cuda.synchronize()
        aiter.hipb_destroy_extension()


def test_destroy_and_reinit() -> None:
    # The preceding test has already destroyed the resource set.
    aiter.hipb_destroy_extension()
    torch.cuda.empty_cache()
    for cycle in range(3):
        before = free_memory()
        aiter.hipb_create_extension()
        initialized = free_memory()
        assert before - initialized >= WORKSPACE_BYTES - MEMORY_TOLERANCE
        try:
            check_gemm()
        finally:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            before_destroy = free_memory()
            aiter.hipb_destroy_extension()
        after_destroy = free_memory()
        assert after_destroy - before_destroy >= WORKSPACE_BYTES - MEMORY_TOLERANCE
        aiter.hipb_destroy_extension()
        assert abs(free_memory() - after_destroy) < MEMORY_TOLERANCE
        print(f"Destroy/re-init cycle {cycle + 1}: PASS")


def test_device_change(device: int, other_device: int) -> None:
    aiter.hipb_create_extension()
    try:
        check_gemm()
        torch.cuda.set_device(other_device)
        try:
            aiter.hipb_create_extension()
        except RuntimeError as error:
            assert "destroy it before initializing on another device" in str(error)
        else:
            raise AssertionError("Changing devices with live resources must fail")
        # Rejection must preserve the original resources.
        torch.cuda.set_device(device)
        check_gemm()
        torch.cuda.set_device(other_device)
    finally:
        aiter.hipb_destroy_extension()
    assert torch.cuda.current_device() == other_device
    aiter.hipb_create_extension()
    try:
        check_gemm()
    finally:
        aiter.hipb_destroy_extension()
        torch.cuda.set_device(device)
    print("Device rejection, owner-device destruction, and device migration: PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    # Keep the regression bounded even when run against an unfixed build.
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--other-device", type=int, default=None)
    args = parser.parse_args()
    if args.iterations < 2:
        parser.error("--iterations must be at least 2")
    if args.other_device == args.device:
        parser.error("--other-device must differ from --device")
    torch.cuda.set_device(args.device)
    torch.manual_seed(42)
    test_repeated_init(args.iterations)
    test_destroy_and_reinit()
    if args.other_device is not None:
        test_device_change(args.device, args.other_device)
    print("hipBLASLt lifecycle: PASS")

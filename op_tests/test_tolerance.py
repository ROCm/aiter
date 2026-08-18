# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the derived-tolerance helper. CPU-only."""

import torch

from aiter.utility.tolerance import derive_tolerance, tolerance_for, unit_roundoff


def test_unit_roundoff_float_formats():
    assert unit_roundoff(torch.float32) == 2**-24
    assert unit_roundoff(torch.float16) == 2**-11
    assert unit_roundoff(torch.bfloat16) == 2**-8
    assert unit_roundoff(torch.float8_e4m3fnuz) == 2**-4
    assert unit_roundoff(torch.float8_e5m2) == 2**-3


def test_integer_compute_dtype_is_refused():
    # An integer holds no rounding error of its own.
    assert unit_roundoff(torch.int32) == 0.0
    # But an integer used as a quantization container has no validated model
    # here, and asking for one is an error rather than a silently wrong number.
    for container in (torch.int8, torch.uint8):
        try:
            derive_tolerance(container, torch.bfloat16, max_value=2048.0)
        except ValueError:
            continue
        raise AssertionError(f"{container} should have been refused")


def test_atol_tracks_the_binary_exponent_of_the_data():
    # bf16 output, bf16 compute: the full-ULP output-cast term dominates, so
    # atol is 2**-7 of the exponent band the data sits in.
    for magnitude, expected in ((1.0, 2**-7), (2048.0, 16.0), (4096.0, 32.0)):
        _, atol = derive_tolerance(
            torch.bfloat16,
            torch.bfloat16,
            num_accumulations=8192,
            max_value=magnitude,
        )
        assert atol == expected, f"{magnitude}: got {atol}, want {expected}"


def test_quantized_compute_dominates_the_bf16_output_cast():
    # fp8 e4m3 has 3 mantissa bits: u = 2**-4 = 0.0625, far above bf16's
    # 2**-8, so the quantization error sets the tolerance.
    rtol, atol = derive_tolerance(
        torch.float8_e4m3fnuz,
        torch.bfloat16,
        num_accumulations=8192,
        max_value=4096.0,
    )
    assert rtol == 2**-4
    assert atol == 256.0


def test_derived_atol_differs_across_the_float_quant_algos():
    # The float-container quant algorithms in test_moe.py, at one fixed
    # magnitude. One constant cannot sit in the right place for both.
    atols = {
        dt: derive_tolerance(
            dt,
            torch.bfloat16,
            num_accumulations=8192,
            max_value=2048.0,
        )[1]
        for dt in (torch.bfloat16, torch.float8_e4m3fnuz)
    }
    assert atols[torch.bfloat16] == 16.0, atols
    assert atols[torch.float8_e4m3fnuz] == 128.0, atols


def test_exact_pipeline_is_bit_exact():
    assert derive_tolerance(torch.int32, torch.int32) == (0.0, 0.0)


def test_safety_factor_only_loosens():
    base = derive_tolerance(torch.bfloat16, torch.bfloat16, max_value=1024.0)
    loose = derive_tolerance(
        torch.bfloat16, torch.bfloat16, max_value=1024.0, safety_factor=2.0
    )
    assert loose == (base[0] * 2, base[1] * 2)
    for bad in (0.5, 0.0):
        try:
            derive_tolerance(torch.bfloat16, torch.bfloat16, safety_factor=bad)
        except ValueError:
            continue
        raise AssertionError(f"safety_factor={bad} should have been rejected")


def test_tolerance_for_reads_the_reference():
    ref = torch.zeros(4, 4, dtype=torch.bfloat16)
    ref[0, 0] = 4096.0
    ref[0, 1] = float("inf")  # ignored when sizing the magnitude
    rtol, atol = tolerance_for(ref, compute_dtype=torch.float8_e4m3fnuz)
    assert rtol == 2**-4
    assert atol == 256.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tolerance tests passed")

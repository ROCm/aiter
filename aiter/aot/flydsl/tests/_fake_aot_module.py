# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Importable CPU-only compile target for spawn-pool tests."""


def get_aot_jobs():
    return [{"kernel_name": "fake_kernel", "value": 7}]


def compile_one_config(kernel_name: str, value: int):
    if value < 0:
        raise ValueError("negative test value")
    return {
        "kernel_name": kernel_name,
        "value": value,
        "compile_time": 0.01,
    }

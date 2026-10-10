# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared acceptance helpers retain raw kernel metrics and exact input hashes."""

import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "replay_common", Path(__file__).with_name("replay_common.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_trace_aggregation_excludes_annotations_and_keeps_span_separate() -> None:
    events = [
        {"cat": "kernel", "ph": "X", "ts": 100, "dur": 4},
        {"cat": "gpu_user_annotation", "ph": "X", "ts": 0, "dur": 1000},
        {"cat": "kernel", "ph": "X", "ts": 110, "dur": 6},
        {"cat": "kernel", "ph": "i", "ts": 120, "dur": 100},
    ]
    kernels, kernel_sum, span = module.trace_kernel_metrics(events, 2)
    assert len(kernels) == 2 and kernel_sum == 5 and span == 8


def test_fingerprint_hashes_contiguous_raw_bytes_of_cpu_tensor() -> None:
    # The SHA256 of independent known literal bytes 00 01 02 03.
    tensor = torch.tensor([0, 1, 2, 3], dtype=torch.uint8, device="cpu")
    assert (
        module.fingerprint(tensor)
        == "054edec1d0211f624fed0cbca9d4f9400b0e491c43742af2c5b0abebf0c990d8"
    )

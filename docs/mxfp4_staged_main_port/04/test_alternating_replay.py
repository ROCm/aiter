# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Targeted replay repeats selected pairs in balanced order on one fixed fixture."""

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "alternating_replay", Path(__file__).with_name("alternating_replay.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_alternating_pairs_keep_every_sample_and_reverse_order_between_cycles() -> (
    "None"
):
    calls = []
    values = {"left": iter([10.0, 12.0, 11.0]), "right": iter([9.0, 8.0, 10.0])}

    def measure(label: "str") -> "dict[str, float]":
        calls.append(label)
        return {"kernel_sum_us": next(values[label]), "cosine_diff": 0.011}

    report = module.alternating_samples(measure, 3)
    assert calls == ["left", "right", "right", "left", "left", "right"]
    assert [sample["label"] for sample in report["samples"]] == calls
    assert report["left"]["kernel_sum_median_us"] == 11.0
    assert report["right"]["kernel_sum_median_us"] == 9.0
    assert report["left"]["kernel_sum_min_us"] == 10.0
    assert report["left"]["kernel_sum_max_us"] == 12.0

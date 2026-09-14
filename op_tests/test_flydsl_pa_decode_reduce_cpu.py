# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only coverage for the FlyDSL PA partition reducer."""

import importlib.util
import math
import random
import sys
from pathlib import Path

import pytest

pytest.importorskip("flydsl")


def _load_reduce_module():
    """Load the self-contained kernel module without importing GPU-facing aiter."""
    path = (
        Path(__file__).parents[1]
        / "aiter"
        / "ops"
        / "flydsl"
        / "kernels"
        / "pa_decode_reduce.py"
    )
    name = "_test_flydsl_pa_decode_reduce"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


REDUCE = _load_reduce_module()


@pytest.mark.parametrize("num_partitions", [1, 63, 64, 65, 128, 192, 256])
def test_reduce_kernel_accepts_supported_partition_counts(num_partitions):
    """Kernel construction is CPU-only and covers both generated code paths."""
    compiled = REDUCE.compile_pa_decode_ps_reduce(
        max_context_partition_num=num_partitions,
        head_size=128,
        output_dtype_str="bf16",
        logits_dtype_str="bf16",
        sink_dtype_str="bf16",
        use_sinks=False,
    )
    assert set(compiled) == {"launch", "kernel"}


@pytest.mark.parametrize("num_partitions", [0, 257])
def test_reduce_kernel_rejects_unsupported_partition_counts(num_partitions):
    with pytest.raises(ValueError, match=r"must be in \[1, 256\]"):
        REDUCE.compile_pa_decode_ps_reduce(
            max_context_partition_num=num_partitions,
            head_size=128,
            output_dtype_str="bf16",
            logits_dtype_str="bf16",
            sink_dtype_str="bf16",
            use_sinks=False,
        )


def _flat_partition_reduce(exp_sums, max_logits, partials, sink=None):
    global_max = max(max_logits)
    scaled_sums = [
        part_sum * math.exp(part_max - global_max) if part_max != -math.inf else 0.0
        for part_sum, part_max in zip(exp_sums, max_logits)
    ]
    denominator = sum(scaled_sums)
    if sink is not None and global_max != -math.inf:
        denominator += math.exp(sink - global_max)
    if denominator == 0.0:
        return [0.0] * len(partials[0])
    return [
        sum(
            partials[part][element] * scaled_sums[part] for part in range(len(exp_sums))
        )
        / denominator
        for element in range(len(partials[0]))
    ]


def _lane_striped_partition_reduce(exp_sums, max_logits, partials):
    """Model the >64 kernel: lane owns p, p+64, p+128, and p+192."""
    wave_size = 64
    lane_maxes = [-math.inf] * wave_size
    for part, part_max in enumerate(max_logits):
        lane = part % wave_size
        lane_maxes[lane] = max(lane_maxes[lane], part_max)
    global_max = max(lane_maxes)

    scaled_sums = []
    lane_sums = [0.0] * wave_size
    for part, (part_sum, part_max) in enumerate(zip(exp_sums, max_logits)):
        scaled_sum = (
            part_sum * math.exp(part_max - global_max) if part_max != -math.inf else 0.0
        )
        scaled_sums.append(scaled_sum)
        lane_sums[part % wave_size] += scaled_sum

    denominator = sum(lane_sums)
    if denominator == 0.0:
        return [0.0] * len(partials[0])
    accumulators = [0.0] * len(partials[0])
    for chunk_base in range(0, len(exp_sums), wave_size):
        chunk_end = min(chunk_base + wave_size, len(exp_sums))
        for part in range(chunk_base, chunk_end):
            weight = scaled_sums[part] / denominator
            for element, value in enumerate(partials[part]):
                accumulators[element] += value * weight
    return accumulators


def _parallel_lds_partition_reduce(exp_sums, max_logits, partials, sink=None):
    """Model the D=128 large-NP workgroup and its cross-wave LDS merge."""
    num_partitions = len(exp_sums)
    parallel_groups = 2 if num_partitions <= 96 else 8
    parts_per_group = (num_partitions + parallel_groups - 1) // parallel_groups
    global_max = max(max_logits)
    scaled_sums = [
        part_sum * math.exp(part_max - global_max) if part_max != -math.inf else 0.0
        for part_sum, part_max in zip(exp_sums, max_logits)
    ]
    denominator = sum(scaled_sums)
    if sink is not None and global_max != -math.inf:
        denominator += math.exp(sink - global_max)
    if denominator == 0.0:
        return [0.0] * len(partials[0])
    weights = [value / denominator for value in scaled_sums]

    group_accumulators = []
    for group in range(parallel_groups):
        begin = group * parts_per_group
        end = min(begin + parts_per_group, num_partitions)
        group_accumulators.append(
            [
                sum(
                    partials[part][element] * weights[part]
                    for part in range(begin, end)
                )
                for element in range(len(partials[0]))
            ]
        )
    return [
        sum(group[element] for group in group_accumulators)
        for element in range(len(partials[0]))
    ]


@pytest.mark.parametrize(
    "num_partitions",
    [26, 30, 32, 34, 36, 40, 64, 65, 96, 127, 128, 129, 160, 192, 255, 256],
)
def test_lane_striped_reduce_matches_flat_reference(num_partitions):
    rng = random.Random(num_partitions)
    max_logits = [rng.uniform(-20.0, 5.0) for _ in range(num_partitions)]
    exp_sums = [rng.uniform(0.01, 10.0) for _ in range(num_partitions)]
    partials = [
        [rng.uniform(-2.0, 2.0) for _ in range(7)] for _ in range(num_partitions)
    ]

    # Exercise masked lanes in the final chunk and neutral empty partitions.
    for part in range(11, num_partitions, 37):
        max_logits[part] = -math.inf
        exp_sums[part] = 0.0

    expected = _flat_partition_reduce(exp_sums, max_logits, partials)
    actual = _lane_striped_partition_reduce(exp_sums, max_logits, partials)
    assert actual == pytest.approx(expected, rel=2e-15, abs=2e-15)


@pytest.mark.parametrize("num_partitions", [128, 192, 256])
def test_lane_striped_reduce_all_empty_is_zero(num_partitions):
    exp_sums = [0.0] * num_partitions
    max_logits = [-math.inf] * num_partitions
    # The tile kernel writes finite neutral scratch for empty partitions.
    partials = [[0.0] * 7 for _ in range(num_partitions)]
    assert _lane_striped_partition_reduce(exp_sums, max_logits, partials) == [0.0] * 7


@pytest.mark.parametrize("num_partitions", [65, 96, 97, 128, 160, 192, 255, 256])
@pytest.mark.parametrize("use_sink", [False, True])
def test_parallel_lds_reduce_matches_flat_reference(num_partitions, use_sink):
    rng = random.Random(10_000 + num_partitions)
    max_logits = [rng.uniform(-20.0, 5.0) for _ in range(num_partitions)]
    exp_sums = [rng.uniform(0.01, 10.0) for _ in range(num_partitions)]
    partials = [
        [rng.uniform(-2.0, 2.0) for _ in range(128)] for _ in range(num_partitions)
    ]
    for part in range(7, num_partitions, 37):
        max_logits[part] = -math.inf
        exp_sums[part] = 0.0
    sink = rng.uniform(-10.0, 4.0) if use_sink else None

    expected = _flat_partition_reduce(exp_sums, max_logits, partials, sink)
    actual = _parallel_lds_partition_reduce(exp_sums, max_logits, partials, sink)
    assert actual == pytest.approx(expected, rel=2e-15, abs=2e-15)


@pytest.mark.parametrize("num_partitions", [65, 96, 128, 160, 192, 256])
@pytest.mark.parametrize("sink", [None, -2.0])
def test_parallel_lds_reduce_all_empty_is_zero(num_partitions, sink):
    exp_sums = [0.0] * num_partitions
    max_logits = [-math.inf] * num_partitions
    partials = [[0.0] * 128 for _ in range(num_partitions)]
    assert (
        _parallel_lds_partition_reduce(exp_sums, max_logits, partials, sink)
        == [0.0] * 128
    )

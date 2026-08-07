#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test and benchmark for FlyDSL AttnRes flag specializations.

Usage:
    python op_tests/test_flydsl_attn_res.py
    python op_tests/test_flydsl_attn_res.py --tokens 17 320

Delta update, canonical append snapshot, and output RMSNorm are independently
specialized. Snapshot writes remain limited to ``block_write_idx == num_blocks``.
"""

import argparse
import itertools
import math

import pandas as pd
import pytest
import torch

from aiter.ops.flydsl import flydsl_attn_res
from aiter.ops.torch_ref.attn_res import attn_res as attn_res_reference
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

_D = 7168
_MAX_BLOCKS = 8
_EPS = 1e-5
_ATOL = 8e-2
_RTOL = 3e-2
_PEAK_BW_GBPS_GFX950 = 8000.0


def _make_inputs(tokens: int):
    torch.manual_seed(20260804 + tokens)
    prefix = torch.randn(tokens, _D, dtype=torch.bfloat16)
    delta = torch.randn(tokens, _D, dtype=torch.bfloat16)
    blocks = torch.randn(tokens, _MAX_BLOCKS, _D, dtype=torch.bfloat16)
    norm_weight = (1.0 + 0.1 * torch.randn(_D)).to(torch.bfloat16)
    qk_weight = (torch.randn(_D) / math.sqrt(_D)).to(torch.bfloat16)
    output_norm_weight = (1.0 + 0.1 * torch.randn(_D)).to(torch.bfloat16)
    return prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight


def _run_and_check(
    tokens: int,
    num_blocks: int,
    has_delta: bool,
    write_block: bool,
    apply_output_norm: bool,
):
    block_write_idx = num_blocks if write_block else -1
    prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight = _make_inputs(
        tokens
    )
    prefix_before = prefix.clone()
    delta_before = delta.clone()
    blocks_before = blocks.clone()
    delta_arg = delta if has_delta else None
    output_norm_weight_arg = output_norm_weight if apply_output_norm else None

    reference, prefix_expected = attn_res_reference(
        prefix.clone(),
        delta_arg,
        blocks.clone(),
        norm_weight,
        qk_weight,
        output_norm_weight_arg,
        num_blocks,
        _EPS,
        _EPS,
    )
    blocks_expected = blocks_before.clone()
    if write_block:
        blocks_expected[:, block_write_idx] = prefix_expected

    output = flydsl_attn_res(
        prefix,
        delta_arg,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight_arg,
        num_blocks,
        block_write_idx,
        _EPS,
        _EPS,
    )
    torch.cuda.synchronize()

    error = checkAllclose(
        reference,
        output,
        atol=_ATOL,
        rtol=_RTOL,
        msg=(
            "AttnRes "
            f"delta={has_delta}, write={write_block}, "
            f"output_norm={apply_output_norm}, "
            f"T={tokens}, num_blocks={num_blocks}: "
        ),
    )
    assert error == 0, f"AttnRes output mismatch for T={tokens}: error ratio={error}"
    expected_prefix = prefix_expected if has_delta else prefix_before
    assert torch.equal(prefix, expected_prefix), "prefix side effect mismatch"
    assert torch.equal(blocks, blocks_expected), "blocks side effect mismatch"
    assert torch.equal(delta, delta_before), "kernel mutated delta"
    assert output.is_contiguous(), "output must be contiguous"


def test_flydsl_attn_res_correctness():
    """Cover the flag surface and source-count endpoints at the prefill grid."""
    for flags in itertools.product((False, True), repeat=3):
        _run_and_check(17, _MAX_BLOCKS - 1, *flags)

    _run_and_check(17, 0, False, False, False)
    _run_and_check(17, 0, True, True, True)
    _run_and_check(17, _MAX_BLOCKS, False, False, False)
    _run_and_check(17, _MAX_BLOCKS, True, False, True)

    for tokens in (1, 320):
        _run_and_check(tokens, _MAX_BLOCKS, False, False, False)
        _run_and_check(tokens, _MAX_BLOCKS - 1, True, True, True)


def test_flydsl_attn_res_empty_returns_without_side_effects():
    """T=0 returns an empty contiguous tensor without launching a kernel."""
    prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight = _make_inputs(
        0
    )
    prefix_before = prefix.clone()
    delta_before = delta.clone()
    blocks_before = blocks.clone()

    output = flydsl_attn_res(
        prefix,
        delta,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight,
        0,
        0,
        _EPS,
        _EPS,
    )

    assert output.shape == prefix.shape
    assert output.is_contiguous(), "empty output must be contiguous"
    assert torch.equal(prefix, prefix_before), "empty path mutated prefix"
    assert torch.equal(delta, delta_before), "empty path mutated delta"
    assert torch.equal(blocks, blocks_before), "empty path mutated blocks"


@pytest.mark.parametrize(
    ("num_blocks", "block_write_idx", "error_type"),
    (
        (5, 2, NotImplementedError),
        (_MAX_BLOCKS, 0, NotImplementedError),
        (0, _MAX_BLOCKS, ValueError),
        (0, -2, ValueError),
        (_MAX_BLOCKS + 1, -1, ValueError),
        (-1, -1, ValueError),
    ),
)
def test_flydsl_attn_res_rejects_invalid_specializations(
    num_blocks: int, block_write_idx: int, error_type: type[Exception]
):
    """Reject unsupported writes before compiling a specialization."""
    prefix, _, blocks, norm_weight, qk_weight, _ = _make_inputs(1)

    with pytest.raises(error_type):
        flydsl_attn_res(
            prefix,
            None,
            blocks,
            norm_weight,
            qk_weight,
            None,
            num_blocks,
            block_write_idx,
            _EPS,
            _EPS,
        )


@benchmark()
def benchmark_flydsl_attn_res(
    tokens: int,
    num_blocks: int,
    has_delta: bool,
    write_block: bool,
    apply_output_norm: bool,
):
    """Validate and measure one supported prefill flag specialization."""
    block_write_idx = num_blocks if write_block else -1
    _run_and_check(
        tokens,
        num_blocks,
        has_delta,
        write_block,
        apply_output_norm,
    )

    # Allocate fresh buffers for timing: delta and snapshot paths mutate inputs
    # during run_perftest, so validation inputs cannot be reused.
    prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight = _make_inputs(
        tokens
    )
    _, us = run_perftest(
        flydsl_attn_res,
        prefix,
        delta if has_delta else None,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight if apply_output_norm else None,
        num_blocks,
        block_write_idx,
        _EPS,
        _EPS,
    )

    # k source reads, optional delta read + prefix write, optional snapshot
    # write, and one output write; all counted rows are BF16.
    k = num_blocks + 1
    rows = k + 2 * int(has_delta) + int(write_block) + 1
    moved_bytes = rows * tokens * _D * 2
    gbps = moved_bytes / (us * 1e-6) / 1e9
    floor_us = moved_bytes / (_PEAK_BW_GBPS_GFX950 * 1e9) * 1e6
    return {
        "k": k,
        "has_delta": has_delta,
        "write_block": write_block,
        "apply_output_norm": apply_output_norm,
        "us": round(us, 3),
        "GB/s": round(gbps, 0),
        "%peak": round(gbps / _PEAK_BW_GBPS_GFX950 * 100, 1),
        "floor_us": round(floor_us, 3),
        "efficiency": round(floor_us / us, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 17, 320],
        help="Prefill token counts to test and benchmark",
    )
    args = parser.parse_args()
    configs = (
        (_MAX_BLOCKS, False, False, False),
        (_MAX_BLOCKS - 1, True, True, True),
        (0, True, True, True),
        (_MAX_BLOCKS, True, False, False),
        (_MAX_BLOCKS, False, False, True),
    )
    results = [
        benchmark_flydsl_attn_res(
            tokens,
            num_blocks,
            has_delta,
            write_block,
            apply_output_norm,
        )
        for num_blocks, has_delta, write_block, apply_output_norm in configs
        for tokens in args.tokens
    ]
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()

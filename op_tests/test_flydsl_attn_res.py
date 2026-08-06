#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test and benchmark for the FlyDSL AttnRes append specializations.

Usage:
    python op_tests/test_flydsl_attn_res.py
    python op_tests/test_flydsl_attn_res.py --tokens 17 320

The supported prefill paths are the original nine-source mix-only path and the
canonical all-on append path, where delta, snapshot write, and output RMSNorm
are all enabled together.
"""

import argparse
import math

import pandas as pd
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


def _run_and_check(tokens: int, num_blocks: int, block_write_idx: int):
    fused = block_write_idx >= 0
    prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight = _make_inputs(
        tokens
    )
    prefix_before = prefix.clone()
    delta_before = delta.clone()
    blocks_before = blocks.clone()
    delta_arg = delta if fused else None
    output_norm_weight_arg = output_norm_weight if fused else None

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
    if fused:
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
            f"AttnRes {'all-on append' if fused else 'mix-only'} "
            f"T={tokens}, num_blocks={num_blocks}: "
        ),
    )
    assert error == 0, f"AttnRes output mismatch for T={tokens}: error ratio={error}"
    assert torch.equal(prefix, prefix_expected), "prefix side effect mismatch"
    assert torch.equal(blocks, blocks_expected), "blocks side effect mismatch"
    assert torch.equal(delta, delta_before), "kernel mutated delta"
    if not fused:
        assert torch.equal(prefix, prefix_before), "mix-only kernel mutated prefix"
        assert torch.equal(blocks, blocks_before), "mix-only kernel mutated blocks"
    assert output.is_contiguous(), "output must be contiguous"


def test_flydsl_attn_res_correctness():
    """Cover mix-only plus canonical all-on append at the prefill grid."""
    for tokens in (1, 17, 320):
        _run_and_check(tokens, _MAX_BLOCKS, -1)
        _run_and_check(tokens, 0, 0)
        _run_and_check(tokens, _MAX_BLOCKS - 1, _MAX_BLOCKS - 1)


@benchmark()
def benchmark_flydsl_attn_res(tokens: int, num_blocks: int, fused: bool):
    """Validate and measure one supported prefill configuration."""
    block_write_idx = num_blocks if fused else -1
    _run_and_check(tokens, num_blocks, block_write_idx)

    # Allocate fresh buffers for timing: the all-on path mutates prefix on every
    # iteration run_perftest performs, so validation inputs cannot be reused.
    prefix, delta, blocks, norm_weight, qk_weight, output_norm_weight = _make_inputs(
        tokens
    )
    _, us = run_perftest(
        flydsl_attn_res,
        prefix,
        delta if fused else None,
        blocks,
        norm_weight,
        qk_weight,
        output_norm_weight if fused else None,
        num_blocks,
        block_write_idx,
        _EPS,
        _EPS,
    )

    # k source reads, optional delta read + prefix write, optional snapshot
    # write, and one output write; all counted rows are BF16.
    k = num_blocks + 1
    rows = k + 2 * int(fused) + int(fused) + 1
    moved_bytes = rows * tokens * _D * 2
    gbps = moved_bytes / (us * 1e-6) / 1e9
    floor_us = moved_bytes / (_PEAK_BW_GBPS_GFX950 * 1e9) * 1e6
    return {
        "k": k,
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
        (_MAX_BLOCKS, False),
        (0, True),
        (_MAX_BLOCKS - 1, True),
    )
    results = [
        benchmark_flydsl_attn_res(tokens, num_blocks, fused)
        for num_blocks, fused in configs
        for tokens in args.tokens
    ]
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()

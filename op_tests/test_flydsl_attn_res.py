#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness test and benchmark for the item-2 FlyDSL AttnRes mix kernel.

Usage:
    python op_tests/test_flydsl_attn_res.py
    python op_tests/test_flydsl_attn_res.py --tokens 17 320

The initial kernel intentionally covers only the contiguous, nine-source,
mix-only prefill path.  Later build-order items add the optional side effects,
smaller source counts, and padded row strides.
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
_NUM_BLOCKS = 8
_EPS = 1e-5
_ATOL = 8e-2
_RTOL = 3e-2
_PEAK_BW_GBPS_GFX950 = 8000.0


def _make_inputs(tokens: int):
    torch.manual_seed(20260804 + tokens)
    prefix = torch.randn(tokens, _D, dtype=torch.bfloat16)
    blocks = torch.randn(tokens, _NUM_BLOCKS, _D, dtype=torch.bfloat16)
    norm_weight = (1.0 + 0.1 * torch.randn(_D)).to(torch.bfloat16)
    qk_weight = (torch.randn(_D) / math.sqrt(_D)).to(torch.bfloat16)
    return prefix, blocks, norm_weight, qk_weight


def _run_and_check(tokens: int):
    prefix, blocks, norm_weight, qk_weight = _make_inputs(tokens)
    prefix_before = prefix.clone()
    blocks_before = blocks.clone()

    reference, _ = attn_res_reference(
        prefix.clone(),
        None,
        blocks.clone(),
        norm_weight,
        qk_weight,
        None,
        _NUM_BLOCKS,
        _EPS,
        _EPS,
    )
    output = flydsl_attn_res(
        prefix,
        None,
        blocks,
        norm_weight,
        qk_weight,
        None,
        _NUM_BLOCKS,
        -1,
        _EPS,
        _EPS,
    )
    torch.cuda.synchronize()

    error = checkAllclose(
        reference,
        output,
        atol=_ATOL,
        rtol=_RTOL,
        msg=f"AttnRes mix-only T={tokens}: ",
    )
    assert error == 0, f"AttnRes output mismatch for T={tokens}: error ratio={error}"
    assert torch.equal(prefix, prefix_before), "mix-only kernel mutated prefix"
    assert torch.equal(blocks, blocks_before), "mix-only kernel mutated blocks"
    assert output.is_contiguous(), "output must be contiguous"
    return prefix, blocks, norm_weight, qk_weight


def test_flydsl_attn_res_mix_correctness():
    """Cover the narrow build-item-2 prefill grid."""
    for tokens in (1, 17, 320):
        _run_and_check(tokens)


@benchmark()
def benchmark_flydsl_attn_res_mix(tokens: int):
    """Validate and measure one prefill shape."""
    prefix, blocks, norm_weight, qk_weight = _run_and_check(tokens)
    _, us = run_perftest(
        flydsl_attn_res,
        prefix,
        None,
        blocks,
        norm_weight,
        qk_weight,
        None,
        _NUM_BLOCKS,
        -1,
        _EPS,
        _EPS,
    )

    # k source reads plus one output write; all tensors are BF16.
    moved_bytes = (_NUM_BLOCKS + 1 + 1) * tokens * _D * 2
    gbps = moved_bytes / (us * 1e-6) / 1e9
    floor_us = moved_bytes / (_PEAK_BW_GBPS_GFX950 * 1e9) * 1e6
    return {
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
    results = [benchmark_flydsl_attn_res_mix(tokens) for tokens in args.tokens]
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()

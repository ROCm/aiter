# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""per_1x32_f4_quant_hip(hadamard_rotate=True): fused 32-point Hadamard + MXFP4."""

import argparse
import itertools
import math
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import dynamic_per_group_scaled_quant, per_1x32_f4_quant_hip
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")


def sylvester_hadamard(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h


def ref_hadamard32(x: torch.Tensor) -> torch.Tensor:
    """Butterfly in the kernel's order, so the fp32 result is bit-identical."""
    v = x.float().reshape(-1, 32).clone()
    h = 1
    while h < 32:
        v = v.view(-1, 32 // (2 * h), 2, h)
        a, b = v[:, :, 0], v[:, :, 1]
        v = torch.stack([a + b, a - b], dim=2)
        h *= 2
    v = v.reshape(-1, 32) * 0.17677669529663687
    return v.to(x.dtype).view(x.shape)


def quant_into(x, shuffle, hadamard_rotate):
    m, n = x.shape
    rows = (m + 255) // 256 * 256 if shuffle else m
    cols = ((n // 32 + 7) // 8 * 8) if shuffle else n // 32
    scale = torch.zeros(rows, cols, dtype=torch.uint8).view(dtypes.fp8_e8m0)
    y = torch.empty(m, n // 2, dtype=dtypes.fp4x2)
    dynamic_per_group_scaled_quant(
        y, x, scale, 32, shuffle_scale=shuffle, hadamard_rotate=hadamard_rotate
    )
    return y, scale


@benchmark()
def test_fused_matches_rotate_then_quant(m, n, float_dtype, shuffle):
    torch.manual_seed(42)
    x = torch.randn((m, n), dtype=float_dtype) * 3.0
    x[:, :: max(1, n // 7)] *= 40.0  # outlier channels

    h = sylvester_hadamard(32).cuda() / math.sqrt(32)
    torch.testing.assert_close(
        ref_hadamard32(x.float()),
        (x.float().view(-1, 32) @ h).view(x.shape),
        rtol=1e-5,
        atol=1e-4,
    )

    y, s = quant_into(x, shuffle, hadamard_rotate=True)
    y_ref, s_ref = quant_into(ref_hadamard32(x), shuffle, hadamard_rotate=False)
    assert torch.equal(
        s.view(torch.uint8), s_ref.view(torch.uint8)
    ), f"scale mismatch ({m},{n},{float_dtype},{shuffle=})"
    assert torch.equal(
        y.view(torch.uint8), y_ref.view(torch.uint8)
    ), f"packed mismatch ({m},{n},{float_dtype},{shuffle=})"

    # The wrapper allocates its scale with torch.empty, so only the packed
    # output is compared.
    y_w, _ = per_1x32_f4_quant_hip(x, shuffle=shuffle, hadamard_rotate=True)
    assert torch.equal(y_w.view(torch.uint8), y.view(torch.uint8))
    return {"result": "PASS"}


@benchmark()
def test_perf(m, n):
    x = torch.randn((m, n), dtype=torch.bfloat16)
    shuffle = m >= 32
    _, us_plain = run_perftest(per_1x32_f4_quant_hip, x, shuffle=shuffle)
    _, us_rot = run_perftest(
        per_1x32_f4_quant_hip, x, shuffle=shuffle, hadamard_rotate=True
    )
    return {"us_plain": us_plain, "us_hadamard": us_rot, "ratio": us_rot / us_plain}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true")
    args = parser.parse_args()
    if get_gfx() != "gfx950":
        aiter.logger.info("hadamard_rotate requires gfx950, skipped on %s", get_gfx())
        sys.exit(0)

    df = [
        test_fused_matches_rotate_then_quant(m, n, dt, sh)
        for (m, n), dt, sh in itertools.product(
            [(1, 32), (125, 2048), (4097, 8192)],
            [torch.bfloat16, torch.float16],
            [False, True],
        )
    ]
    if args.perf:
        df += [
            test_perf(m, n)
            for m, n in itertools.product([1, 32, 256, 2048, 8192], [2048, 8192])
        ]
    aiter.logger.info(
        "quant_mxfp4_hadamard summary:\n%s", pd.DataFrame(df).to_markdown(index=False)
    )

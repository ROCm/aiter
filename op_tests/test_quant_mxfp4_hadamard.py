# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""per_1x32_f4_quant_hip(hadamard_rotate=True): fused 32-point Hadamard + MXFP4."""

import argparse
import itertools
import math
import subprocess
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.ops.quant import dynamic_per_group_scaled_quant, per_1x32_f4_quant_hip
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")

E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


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


def dequant_mxfp4(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """(m, n // 2) fp4x2 low-nibble-first + unshuffled (m, n // 32) e8m0 -> fp32."""
    u8 = packed.view(torch.uint8)
    codes = torch.stack([u8 & 0xF, u8 >> 4], dim=-1).flatten(-2).long()
    table = torch.tensor(E2M1_VALUES + [-v for v in E2M1_VALUES], device=u8.device)
    vals = table[codes]
    exp = scale.view(torch.uint8)[:, : vals.shape[1] // 32].float() - 127.0
    return (vals.view(vals.shape[0], -1, 32) * torch.exp2(exp)[..., None]).flatten(-2)


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


def test_reference_is_hadamard_matmul():
    torch.manual_seed(0)
    x = torch.randn(64, 256, dtype=torch.float32)
    h = sylvester_hadamard(32).cuda() / math.sqrt(32)
    want = (x.view(-1, 32) @ h).view(x.shape)
    v = x.reshape(-1, 32).clone()
    step = 1
    while step < 32:
        v = v.view(-1, 32 // (2 * step), 2, step)
        v = torch.stack([v[:, :, 0] + v[:, :, 1], v[:, :, 0] - v[:, :, 1]], dim=2)
        step *= 2
    got = v.reshape(x.shape) * 0.17677669529663687
    torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


@benchmark()
def test_fused_matches_rotate_then_quant(m, n, float_dtype, shuffle):
    torch.manual_seed(42)
    x = torch.randn((m, n), dtype=float_dtype) * 3.0
    x[:, :: max(1, n // 7)] *= 40.0  # outlier channels

    y, s = quant_into(x, shuffle, hadamard_rotate=True)
    y_ref, s_ref = quant_into(ref_hadamard32(x), shuffle, hadamard_rotate=False)

    assert torch.equal(
        s.view(torch.uint8), s_ref.view(torch.uint8)
    ), f"scale mismatch ({m},{n},{float_dtype},{shuffle=})"
    assert torch.equal(
        y.view(torch.uint8), y_ref.view(torch.uint8)
    ), f"packed mismatch ({m},{n},{float_dtype},{shuffle=})"

    y_w, _ = per_1x32_f4_quant_hip(x, shuffle=shuffle, hadamard_rotate=True)
    assert torch.equal(y_w.view(torch.uint8), y.view(torch.uint8))
    return {"result": "PASS"}


@benchmark()
def test_gemm_error(m, n, k):
    """Rotating both operands leaves x @ w.T unchanged and shrinks the FP4 error."""
    torch.manual_seed(7)
    x = (torch.distributions.StudentT(3.0).sample((m, k)).cuda()).to(torch.bfloat16)
    x[:, torch.randperm(k)[: k // 64]] *= 20.0
    w = (torch.randn(n, k) * 0.02).to(torch.bfloat16)
    want = x.float() @ w.float().T

    def err(xq, xs, wq, ws):
        got = dequant_mxfp4(xq, xs) @ dequant_mxfp4(wq, ws).T
        return ((got - want).norm() / want.norm()).item()

    plain = err(*per_1x32_f4_quant_hip(x), *per_1x32_f4_quant_hip(w))
    rot = err(
        *per_1x32_f4_quant_hip(x, hadamard_rotate=True),
        *per_1x32_f4_quant_hip(w, hadamard_rotate=True),
    )
    assert rot < plain, f"rotation did not reduce error: {rot=} {plain=}"
    return {"rel_err_plain": plain, "rel_err_hadamard": rot}


@benchmark()
def test_perf(m, n):
    x = torch.randn((m, n), dtype=torch.bfloat16)
    shuffle = m >= 32
    _, us_plain = run_perftest(per_1x32_f4_quant_hip, x, shuffle=shuffle)
    _, us_rot = run_perftest(
        per_1x32_f4_quant_hip, x, shuffle=shuffle, hadamard_rotate=True
    )
    return {"us_plain": us_plain, "us_hadamard": us_rot, "ratio": us_rot / us_plain}


def test_rejects_unsupported():
    x = torch.randn(32, 256, dtype=torch.bfloat16)
    try:
        aiter.per_1x32_mx_quant_hip(x, quant_dtype=dtypes.fp8, hadamard_rotate=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("fp8 hadamard_rotate was not rejected")

    # AITER_CHECK aborts the process, so the C++ check runs in a child.
    cmd = [
        sys.executable,
        "-c",
        (
            "import torch\n"
            "from aiter import dtypes\n"
            "from aiter.ops.quant import dynamic_per_group_scaled_quant\n"
            "x = torch.randn(32, 256, dtype=torch.bfloat16, device='cuda')\n"
            "dynamic_per_group_scaled_quant(\n"
            "    torch.empty(32, 128, dtype=dtypes.fp4x2, device='cuda'), x,\n"
            "    torch.empty(32, 2, dtype=torch.uint8, device='cuda'), 128,\n"
            "    hadamard_rotate=True)\n"
            "torch.cuda.synchronize()\n"
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    if (
        proc.returncode == 0
        or "hadamard_rotate requires" not in proc.stdout + proc.stderr
    ):
        raise AssertionError(
            "group_size 128 hadamard_rotate was not rejected by AITER_CHECK "
            f"(exit {proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--perf", action="store_true")
    args = parser.parse_args()

    test_reference_is_hadamard_matmul()
    test_rejects_unsupported()

    shapes = [(1, 32), (3, 256), (31, 2048), (32, 8192), (125, 2048), (4097, 8192)]
    df = [
        test_fused_matches_rotate_then_quant(m, n, dt, sh)
        for (m, n), dt, sh in itertools.product(
            shapes, [torch.bfloat16, torch.float16], [False, True]
        )
    ]
    df += [test_gemm_error(m, n, k) for m, n, k in [(256, 512, 2048), (64, 256, 8192)]]
    if args.perf:
        df += [
            test_perf(m, n)
            for m, n in itertools.product([1, 32, 256, 2048, 8192], [2048, 8192])
        ]
    aiter.logger.info(
        "quant_mxfp4_hadamard summary:\n%s", pd.DataFrame(df).to_markdown(index=False)
    )

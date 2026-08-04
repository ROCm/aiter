# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FlyDSL warp-decode MoE primitives (SILOTIGER-667, Phase 1).

Validates the three low-level primitives the warp-decode gate_up / down_reduce
kernels are built from, each in isolation on real gfx950 hardware:

    1. ``v_dot2_f32_bf16``      local inline-asm dot helper (2 bf16 MACs/lane).
    2. ``cvt_scalef32_pk_bf16_fp8``  scaled FP8(e4m3) -> BF16 pair convert.
    3. 64-lane ``shuffle_xor`` butterfly reduce.

Usage:
    python op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py
    pytest -q op_tests/flydsl_tests/test_flydsl_warp_decode_moe.py
"""

from __future__ import annotations

import argparse

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL warp-decode tests.",
        allow_module_level=True,
    )

import flydsl.compiler as flyc  # noqa: E402

from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg  # noqa: E402
from aiter.ops.flydsl.kernels.warp_decode_moe import (  # noqa: E402
    WARP_SIZE,
    build_warp_decode_primitives_module,
)

torch.set_default_device("cuda")

_HAS_FP8 = hasattr(torch, "float8_e4m3fn")


def _run_primitives(serialize_dot2: bool = True):
    """Launch the primitive kernel once and return (inputs, outputs) dicts."""
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(20260404)
    n = WARP_SIZE

    # 1. dot2 inputs: two bf16 lanes packed per i32.
    a_bf16 = (torch.rand((n, 2), generator=gen, device=device) * 4 - 2).to(
        torch.bfloat16
    )
    b_bf16 = (torch.rand((n, 2), generator=gen, device=device) * 4 - 2).to(
        torch.bfloat16
    )
    a_i32 = a_bf16.contiguous().view(torch.int32).reshape(n).contiguous()
    b_i32 = b_bf16.contiguous().view(torch.int32).reshape(n).contiguous()
    out_dot = torch.zeros(n, dtype=torch.float32, device=device)

    # 2. convert inputs: four e4m3 bytes packed per i32, one f32 scale per lane.
    f8 = (torch.rand((n, 4), generator=gen, device=device) * 8 - 4).to(
        torch.float8_e4m3fn
    )
    f8_i32 = f8.contiguous().view(torch.int32).reshape(n).contiguous()
    scale = torch.empty(n, dtype=torch.float32, device=device)
    # Power-of-two scales keep the reference exact regardless of whether the HW
    # applies the full f32 or only its exponent.
    scale[0::3] = 1.0
    scale[1::3] = 2.0
    scale[2::3] = 0.5
    out_cvt = torch.zeros(n * 4, dtype=torch.bfloat16, device=device)

    # 3. reduce inputs.
    red_in = (torch.rand(n, generator=gen, device=device) * 2 - 1).to(torch.float32)
    out_red = torch.zeros(n, dtype=torch.float32, device=device)

    launcher = build_warp_decode_primitives_module(serialize_dot2=serialize_dot2)
    cf = flyc.compile(
        launcher,
        ptr_arg(a_i32),
        ptr_arg(b_i32),
        ptr_arg(out_dot),
        ptr_arg(f8_i32),
        ptr_arg(scale),
        ptr_arg(out_cvt),
        ptr_arg(red_in),
        ptr_arg(out_red),
        torch.cuda.current_stream(),
    )
    del cf
    torch.cuda.synchronize()

    inputs = {
        "a_bf16": a_bf16,
        "b_bf16": b_bf16,
        "f8": f8,
        "scale": scale,
        "red_in": red_in,
    }
    outputs = {
        "dot": out_dot,
        "cvt": out_cvt.reshape(n, 4),
        "red": out_red,
    }
    return inputs, outputs


def _report(label, ref, out, *, atol, rtol):
    ref_f = ref.float()
    out_f = out.float()
    max_delta = (ref_f - out_f).abs().max().item()
    close = torch.isclose(ref_f, out_f, atol=atol, rtol=rtol)
    pct = close.float().mean().item() * 100.0
    passed = bool(close.all())
    print(f"  [{label}] max_delta={max_delta:.5f}, {pct:.2f}% close (atol={atol})")
    print(f"    ref  sample: {ref_f.reshape(-1)[:6].tolist()}")
    print(f"    test sample: {out_f.reshape(-1)[:6].tolist()}")
    print(f"    --> {'PASS' if passed else 'FAIL'}")
    return passed, max_delta


def _check_dot2(inputs, outputs):
    ref = (inputs["a_bf16"].float() * inputs["b_bf16"].float()).sum(dim=1)
    return _report("dot2_f32_bf16", ref, outputs["dot"], atol=2e-2, rtol=2e-2)


def _check_convert(inputs, outputs):
    ref = inputs["f8"].float() * inputs["scale"][:, None]
    return _report(
        "cvt_scalef32_pk_bf16_fp8", ref, outputs["cvt"], atol=1e-1, rtol=2e-2
    )


def _check_reduce(inputs, outputs):
    ref = inputs["red_in"].float().sum().expand(WARP_SIZE)
    return _report("butterfly_reduce", ref, outputs["red"], atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("serialize_dot2", [True, False])
def test_dot2_f32_bf16(serialize_dot2):
    inputs, outputs = _run_primitives(serialize_dot2=serialize_dot2)
    passed, _ = _check_dot2(inputs, outputs)
    assert passed


@pytest.mark.skipif(not _HAS_FP8, reason="torch build lacks float8_e4m3fn")
def test_cvt_scalef32_pk_bf16_fp8():
    inputs, outputs = _run_primitives()
    passed, _ = _check_convert(inputs, outputs)
    assert passed


def test_butterfly_reduce():
    inputs, outputs = _run_primitives()
    passed, _ = _check_reduce(inputs, outputs)
    assert passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-serialize",
        action="store_true",
        help="disable the s_nop 2 dot2 hazard guard",
    )
    args = parser.parse_args()

    print("=" * 78)
    print("[flydsl] warp-decode MoE primitives (Phase 1)")
    print("=" * 78)
    inputs, outputs = _run_primitives(serialize_dot2=not args.no_serialize)

    results = [
        _check_dot2(inputs, outputs),
        _check_convert(inputs, outputs) if _HAS_FP8 else (True, 0.0),
        _check_reduce(inputs, outputs),
    ]
    n_pass = sum(1 for p, _ in results if p)
    print(f"\n  {n_pass}/{len(results)} primitive checks passed")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

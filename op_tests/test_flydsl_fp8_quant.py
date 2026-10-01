# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Public FlyDSL Q/K/V FP8 quantizer op test."""

import argparse
import itertools
import math
import warnings

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_fp8_quant
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx1201"]
FP8_MAX = 448.0
SEED = 0

# D64 same-shape selects the fused producer; D64 cross-attention selects three
# VEC2 producers; D96 exercises the padded, unrotated generic fallback; D512
# and D1024 exercise the split dwordx4 loads used by wider lane fragments.
SHAPES = [
    ("d64_fused", 1, 128, 128, 4, 64, True),
    ("d64_cross", 1, 192, 128, 4, 64, False),
    ("d96_padded", 1, 128, 160, 4, 96, False),
    ("d512", 1, 32, 32, 2, 512, True),
    ("d1024", 1, 16, 16, 1, 1024, True),
]


def _make_qkv(batch, seq_q, seq_kv, heads, head_dim, *, device="cuda"):
    generator = torch.Generator(device=device).manual_seed(SEED)
    return (
        torch.randn(
            (batch, seq_q, heads, head_dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.randn(
            (batch, seq_kv, heads, head_dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ),
        torch.randn(
            (batch, seq_kv, heads, head_dim),
            generator=generator,
            dtype=torch.bfloat16,
            device=device,
        ),
    )


def _fwht(x):
    """Orthonormal FWHT over the last dimension, matching Q/K rotation."""
    head_dim = x.shape[-1]
    result = x.float().reshape(-1, head_dim)
    width = 1
    while width < head_dim:
        pairs = result.reshape(-1, head_dim // (2 * width), 2 * width)
        left, right = pairs[..., :width], pairs[..., width:]
        result = torch.cat((left + right, left - right), dim=-1).reshape(-1, head_dim)
        width *= 2
    return result.reshape_as(x) / math.sqrt(head_dim)


def _quant_reference(x, *, rotation):
    transformed = _fwht(x) if rotation else x.float()
    scale = (transformed.abs().amax() / FP8_MAX).clamp(min=1e-12).reshape(1)
    return transformed, scale


def _check_quantized(actual, scale, reference, reference_scale, *, exact_scale, name):
    if exact_scale:
        assert torch.equal(scale, reference_scale), f"{name}: scale is not exact"
    else:
        assert torch.allclose(scale, reference_scale, rtol=2e-5, atol=1e-7), name
    err = checkAllclose(
        reference.to(dtypes.fp32),
        (actual.float() * scale).to(dtypes.fp32),
        rtol=1.3e-1,
        atol=scale.item() * 0.125,
        msg=f"{name}: fp8 dequant",
    )
    assert err == 0, f"{name}: fp8 dequant mismatch ratio={err}"
    return err


def _exercise_public_contracts():
    """Validate materialization, padding, stream, and device ownership."""
    storage = torch.zeros((1, 32, 2, 128), dtype=torch.bfloat16, device="cuda")
    noncontiguous = storage[..., ::2]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q8, k8, v8, sq, sk, sv = flydsl_fp8_quant(
            noncontiguous, noncontiguous, noncontiguous
        )
    assert any(
        "materializes non-contiguous Q/K/V" in str(warning.message)
        for warning in caught
    )
    for quantized, scale in ((q8, sq), (k8, sk), (v8, sv)):
        assert torch.isfinite(quantized.float()).all()
        assert torch.all(quantized == 0)
        assert torch.equal(scale, torch.tensor([1e-12], device=scale.device))

    padded = torch.zeros((1, 32, 2, 96), dtype=torch.bfloat16, device="cuda")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        flydsl_fp8_quant(padded, padded, padded, rotation=False)
    assert any("pads head_dim=96 to 128" in str(warning.message) for warning in caught)

    stream = torch.cuda.Stream(device="cuda")
    with torch.cuda.stream(stream):
        q, k, v = _make_qkv(1, 64, 64, 2, 64)
        q8, k8, v8, sq, sk, sv = flydsl_fp8_quant(q, k, v, rotation=False)
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()
    references = [_quant_reference(x, rotation=False) for x in (q, k, v)]
    for actual, scale, (reference, reference_scale), name in zip(
        (q8, k8, v8), (sq, sk, sv), references, ("q", "k", "v"), strict=True
    ):
        _check_quantized(
            actual, scale, reference, reference_scale, exact_scale=True, name=name
        )

    if torch.cuda.device_count() < 2:
        aiter.logger.warning("two-GPU FP8 quant check skipped: fewer than two GPUs")
        return
    previous_device = torch.cuda.current_device()
    try:
        device1 = torch.device("cuda", 1)
        with torch.cuda.device(device1):
            q, k, v = _make_qkv(1, 64, 64, 2, 64, device=device1)
        torch.cuda.set_device(0)
        outputs = flydsl_fp8_quant(q, k, v, rotation=False)
        torch.cuda.synchronize(device1)
        assert all(x.device == device1 for x in outputs)
    finally:
        torch.cuda.set_device(previous_device)


@benchmark()
def run_flydsl_fp8_quant(shape, batch, seq_q, seq_kv, heads, head_dim, rotation):
    """Time the public producer and compare it to a torch/FWHT reference."""
    q, k, v = _make_qkv(batch, seq_q, seq_kv, heads, head_dim)
    references = [
        _quant_reference(q, rotation=rotation),
        _quant_reference(k, rotation=rotation),
        _quant_reference(v, rotation=False),
    ]
    outputs, us = run_perftest(flydsl_fp8_quant, q, k, v, rotation=rotation)
    q8, k8, v8, sq, sk, sv = outputs
    errors = [
        _check_quantized(
            actual,
            scale,
            reference,
            reference_scale,
            exact_scale=not rotation or name == "v",
            name=f"{shape}_{name}",
        )
        for actual, scale, (reference, reference_scale), name in zip(
            (q8, k8, v8), (sq, sk, sv), references, ("q", "k", "v"), strict=True
        )
    ]
    elements = q.numel() + k.numel() + v.numel()
    flops = elements * (math.log2(head_dim) if rotation else 1)
    nbytes = elements * (2 * q.element_size() + 1)
    return {
        "gfx": get_gfx(),
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": max(errors),
    }


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "native FlyDSL FP8 quant unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(description="Native FlyDSL FP8 quant config")
    parser.add_argument(
        "-s",
        "--shape",
        choices=[config[0] for config in SHAPES],
        nargs="*",
        default=[config[0] for config in SHAPES],
        help="Quantizer shape groups to sweep.",
    )
    args = parser.parse_args()
    _exercise_public_contracts()
    rows = [
        run_flydsl_fp8_quant(*config)
        for (config,) in itertools.product(SHAPES)
        if config[0] in args.shape
    ]
    aiter.logger.info(
        "flydsl_fp8_quant summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()

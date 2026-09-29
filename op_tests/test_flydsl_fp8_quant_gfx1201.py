# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native gfx1201 FlyDSL Q/K/V FP8 quantizer op test."""

from __future__ import annotations

import argparse
import itertools
import math

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

# (label, batch, seq_q, seq_kv, heads, head_dim, rotation). D64 same-shape
# selects the fused producer; D64 cross-attention selects three VEC2 producers;
# D96 exercises the padded, unrotated generic fallback.
SHAPES = [
    ("d64_fused", 1, 128, 128, 4, 64, True),
    ("d64_cross", 1, 192, 128, 4, 64, False),
    ("d96_padded", 1, 128, 160, 4, 96, False),
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


def _fwht(x: torch.Tensor) -> torch.Tensor:
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


def _quant_reference(x: torch.Tensor, *, rotation: bool):
    transformed = _fwht(x) if rotation else x.float()
    scale = (transformed.abs().amax() / FP8_MAX).clamp(min=1e-12).reshape(1)
    return transformed, scale


def _check_quantized(
    actual: torch.Tensor,
    scale: torch.Tensor,
    reference: torch.Tensor,
    reference_scale: torch.Tensor,
    *,
    exact_scale: bool,
    name: str,
) -> float:
    if exact_scale:
        assert torch.equal(scale, reference_scale), f"{name}: scale is not exact"
    else:
        assert torch.allclose(scale, reference_scale, rtol=2e-5, atol=1e-7), name
    return checkAllclose(
        reference.to(dtypes.fp32),
        (actual.float() * scale).to(dtypes.fp32),
        rtol=1.3e-1,
        atol=scale.item() * 0.125,
        msg=f"{name}: fp8 dequant",
    )


def _exercise_public_contracts() -> None:
    """Validate zero, stream, and input-device ownership contracts."""
    zero = torch.zeros((1, 32, 2, 64), dtype=torch.bfloat16, device="cuda")
    q8, k8, v8, sq, sk, sv = flydsl_fp8_quant(zero, zero, zero)
    for quantized, scale in ((q8, sq), (k8, sk), (v8, sv)):
        assert torch.isfinite(quantized.float()).all()
        assert torch.all(quantized == 0)
        assert torch.equal(scale, torch.tensor([1e-12], device=scale.device))

    # An event after the call on this stream is sufficient to read all outputs
    # only when the public wrapper enqueues its work on q.device's current stream.
    stream = torch.cuda.Stream(device="cuda")
    with torch.cuda.stream(stream):
        q, k, v = _make_qkv(1, 64, 64, 2, 64)
        q8, k8, v8, sq, sk, sv = flydsl_fp8_quant(q, k, v, rotation=False)
        done = torch.cuda.Event()
        done.record(stream)
    done.synchronize()
    refs = [_quant_reference(x, rotation=False) for x in (q, k, v)]
    for actual, scale, (reference, reference_scale), name in zip(
        (q8, k8, v8), (sq, sk, sv), refs, ("q", "k", "v"), strict=True
    ):
        _check_quantized(
            actual, scale, reference, reference_scale, exact_scale=True, name=name
        )

    if torch.cuda.device_count() < 2:
        aiter.logger.warning(
            "two-GPU FlyDSL FP8 quant check skipped: fewer than two GPUs"
        )
        return

    device1 = torch.device("cuda", 1)
    with torch.cuda.device(device1):
        q, k, v = _make_qkv(1, 64, 64, 2, 64, device=device1)
    torch.cuda.set_device(0)
    outputs = flydsl_fp8_quant(q, k, v, rotation=False)
    torch.cuda.synchronize(device1)
    assert all(x.device == device1 for x in outputs)


@benchmark()
def run_flydsl_fp8_quant(shape, batch, seq_q, seq_kv, heads, head_dim, rotation):
    """Time the public Q/K/V producer and compare to a torch/FWHT reference."""
    q, k, v = _make_qkv(batch, seq_q, seq_kv, heads, head_dim)
    # The public producer rotates Q/K only; V remains in its original basis.
    references = [
        _quant_reference(q, rotation=rotation),
        _quant_reference(k, rotation=rotation),
        _quant_reference(v, rotation=False),
    ]
    outputs, us = run_perftest(flydsl_fp8_quant, q, k, v, rotation=rotation)
    q8, k8, v8, sq, sk, sv = outputs

    # A raw amax has identical FP32 semantics for the unrotated producer paths.
    # The rotated path is checked against the mathematically equivalent FWHT.
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
    nbytes = elements * (2 * q.element_size() + 1)  # two BF16 passes + FP8 write
    return {
        "gfx": get_gfx(),
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": max(errors),
    }


def main() -> None:
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "native FlyDSL FP8 quant unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(description="Native gfx1201 FlyDSL FP8 quant")
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
        "flydsl_fp8_quant_gfx1201 summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()

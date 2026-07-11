# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Bench-style static DATA/SCALE init for gfx1250 FP4/FP8 GEMM tests.

Two *independent* axes -- data and scale are sampled independently, **not**
"make the data then derive the scale by quantizing it". This matches the
AAI/mblas ``bench_init`` guideline (canonical realistic pattern for MAF/TAF
and competitive compares); distribution names mirror mblas-bench
``generic_init.h`` (normal / uniform / pow2_binomial / trig).

DATA (per element, iid) -- ``--data-init``::

    uniform  : FP4 -> U(-3,3), FP8 -> U(-6,6)      [default]
    gaussian : N(0,1)                              [norm-dist / LLM-like]
    trig     : trig_float in [-2,2]                [optimistic pattern]
    random   : pure random on-wire codes           [overly pessimistic baseline]

then round-to-nearest into the on-wire low-precision format (FP4 e2m1 packed
2/byte, FP8 e4m3).

SCALE (per block) -- ``--scale-init``::

    pow2_binomial : E8M0 value = 2^(Binomial(2n+1,0.5) - (n+1)), n=10
    gaussian      : N(0.34375, 0.08) -> E4M3
    random        : random on-wire byte in a modest range
    auto          : format-recommended default (E8M0 -> pow2_binomial,
                    E4M3 -> gaussian(0.34375, 0.08))  [default]

Every sampler takes a ``torch.Generator`` so a fixed ``--seed`` reproduces the
buffers bit-for-bit. ``constant`` init is operator-specific (fixed representable
bytes) and stays in the individual test files.
"""
import math

import torch

FP8_E4M3 = torch.float8_e4m3fn  # gfx1250 fp8 data & E4M3 block-scale encoding

# Selectable distributions (kept in sync with the test-file argparse choices).
DATA_DISTS = ("constant", "uniform", "gaussian", "trig", "random")
E8M0_SCALE_DISTS = ("constant", "pow2_binomial", "random", "auto")
E4M3_SCALE_DISTS = ("constant", "gaussian", "random", "auto")

# Per-format uniform ranges for scaled low-precision DATA.
FP4_UNIFORM = (-3.0, 3.0)  # E2M1 max is 6.0; keep headroom
FP8_UNIFORM = (-6.0, 6.0)  # a touch wider than FP4

# E4M3 block-scale gaussian: recognisable non-2^ center + spread.
E4M3_SCALE_MEAN, E4M3_SCALE_STD = 0.34375, 0.08

# pow2_binomial exponent recenter: e = Binomial(2n+1, 0.5) - (n+1).
# n=10 -> Binomial(21,0.5)-11, exponents centred at -0.5. Lower n narrows the
# scale dynamic range (useful if a correctness ref hits the allclose tolerance).
POW2_BINOMIAL_N = 10

# e4m3fn NaN encodings (OCP): S.1111.111 -> 0x7F / 0xFF. Everything else is a
# finite representable code, so "random codes" only needs to avoid these two.
_E4M3_NAN_POS, _E4M3_NAN_NEG = 0x7F, 0xFF


def make_generator(seed, device="cuda"):
    """Seeded ``torch.Generator`` -- same seed => bit-identical buffers."""
    return torch.Generator(device=device).manual_seed(int(seed))


# --------------------------------------------------------------------------- #
# DATA
# --------------------------------------------------------------------------- #
def _sample_f32(shape, dist, gen, *, lo, hi, device):
    """Sample an f32 tensor for the requested continuous DATA distribution."""
    if dist == "uniform":
        return torch.empty(shape, dtype=torch.float32, device=device).uniform_(
            lo, hi, generator=gen
        )
    if dist == "gaussian":
        return torch.empty(shape, dtype=torch.float32, device=device).normal_(
            0.0, 1.0, generator=gen
        )
    if dist == "trig":
        # Deterministic trig_float in [-2,2]; the generator only jitters the
        # phase so --seed still varies the pattern without breaking repro.
        n = 1
        for s in shape:
            n *= s
        idx = torch.arange(n, dtype=torch.float32, device=device)
        phase = torch.rand(1, generator=gen, device=device).item() * (2.0 * math.pi)
        return (2.0 * torch.sin(0.017 * idx + phase)).reshape(shape)
    raise ValueError(f"data dist {dist!r} is not continuous; use fill_* dispatch")


def fill_fp4(shape, dist, gen, *, uniform=FP4_UNIFORM, device="cuda"):
    """Return packed e2m1 ``uint8`` of shape ``(rows, cols // 2)``.

    ``dist`` in {uniform, gaussian, trig} -> sample f32 then round to e2m1.
    ``dist == "random"`` -> uniform over all e2m1 codes (every byte is a valid
    pair of e2m1 nibbles).
    """
    rows, cols = shape
    assert cols % 2 == 0, f"FP4 needs even columns, got {cols}"
    if dist == "random":
        return torch.randint(
            0, 256, (rows, cols // 2), dtype=torch.uint8, device=device, generator=gen
        )
    # Local import: fp4_utils pulls in triton; keep module import cheap.
    from aiter.utility import fp4_utils

    v = _sample_f32(shape, dist, gen, lo=uniform[0], hi=uniform[1], device=device)
    return fp4_utils.f32_to_mxfp4(v).view(torch.uint8)


def fill_fp8(shape, dist, gen, *, uniform=FP8_UNIFORM, device="cuda"):
    """Return an e4m3 tensor of ``shape``.

    ``dist`` in {uniform, gaussian, trig} -> sample f32 then cast to e4m3.
    ``dist == "random"`` -> uniform over finite e4m3 codes (NaN bytes remapped).
    """
    if dist == "random":
        b = torch.randint(
            0, 256, shape, dtype=torch.uint8, device=device, generator=gen
        )
        b[b == _E4M3_NAN_POS] = 0x00  # +NaN -> +0
        b[b == _E4M3_NAN_NEG] = 0x80  # -NaN -> -0
        return b.view(FP8_E4M3)
    v = _sample_f32(shape, dist, gen, lo=uniform[0], hi=uniform[1], device=device)
    return v.to(FP8_E4M3)


# --------------------------------------------------------------------------- #
# SCALE
# --------------------------------------------------------------------------- #
def fill_scale_e8m0(shape, dist="auto", gen=None, *, device="cuda", n=POW2_BINOMIAL_N):
    """Return E8M0 on-wire ``uint8`` (biased exponent, bias 127).

    ``auto``/``pow2_binomial`` -> value = 2^(Binomial(2n+1, 0.5) - (n+1)).
    ``random`` -> uniform exponent in [-2, 2] (modest, ref-friendly).
    """
    if dist not in E8M0_SCALE_DISTS:
        raise ValueError(f"E8M0 scale dist {dist!r}; choose from {E8M0_SCALE_DISTS}")
    if dist == "random":
        return torch.randint(
            125, 130, shape, dtype=torch.uint8, device=device, generator=gen
        )
    # auto / pow2_binomial: accumulate 2n+1 Bernoulli(0.5) draws (reproducible,
    # far less memory than materialising the trials dim at once).
    e = torch.zeros(shape, dtype=torch.int32, device=device)
    for _ in range(2 * n + 1):
        e += (torch.rand(shape, generator=gen, device=device) < 0.5).to(torch.int32)
    e -= n + 1
    return (e + 127).clamp_(0, 255).to(torch.uint8)


def fill_scale_e4m3(shape, dist="auto", gen=None, *, device="cuda"):
    """Return E4M3 on-wire ``uint8``.

    ``auto``/``gaussian`` -> N(0.34375, 0.08), clamped non-negative, cast e4m3.
    ``random`` -> uniform over e4m3 bytes in [0x20, 0x50) (legacy NVFP4 range).
    """
    if dist not in E4M3_SCALE_DISTS:
        raise ValueError(f"E4M3 scale dist {dist!r}; choose from {E4M3_SCALE_DISTS}")
    if dist == "random":
        return torch.randint(
            0x20, 0x50, shape, dtype=torch.uint8, device=device, generator=gen
        )
    v = torch.empty(shape, dtype=torch.float32, device=device).normal_(
        E4M3_SCALE_MEAN, E4M3_SCALE_STD, generator=gen
    )
    v.clamp_(min=0.0)  # block scales are non-negative
    return v.to(FP8_E4M3).view(torch.uint8)

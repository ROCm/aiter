# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared benchmark data and scale initialization helpers."""

import torch

# --------------------------------------------------------------------------- #
# DATA / SCALE init.
#   gen = make_generator(seed)
#   x  = fill(shape, dist, gen, dtype=...)            # bf16 / fp32 / float8
#   s  = fill_scale(shape, dist, gen)                 # float32 block scale
#   xq = fill_fp4(shape, dist, gen)                   # MXFP4 packed e2m1
#   x8 = fill_fp8(shape, dist, gen)                   # MXFP8 e4m3
#   s8 = fill_scale_e8m0(shape, dist, gen)            # MX E8M0 on-wire
#   s4 = fill_scale_e4m3(shape, dist, gen)            # NVFP4 E4M3 on-wire
#
# OCP (Open Compute Project) published the MX microscaling formats: e2m1/e4m3
# data plus a tiny E8M0 (or E4M3) scale per block. fill_fp* emit those on-wire
# buffers. Large 2-D tensors are filled in row chunks (~1 GiB f32 staging).
# --------------------------------------------------------------------------- #
DATA_DISTS = ("zero", "constant", "uniform", "norm")
SCALE_DISTS = DATA_DISTS
SCALE_UNIFORM = (0.5, 2.0)
SCALE_NORM_MEAN, SCALE_NORM_STD = 1.0, 0.25
FP8_E4M3 = torch.float8_e4m3fn
FP4_UNIFORM = (-3.0, 3.0)  # e2m1 max is 6.0; keep headroom
FP8_UNIFORM = (-6.0, 6.0)
E8M0_BIAS = 127
E8M0_NEUTRAL = 0x7F  # 2^0 = 1.0
E4M3_NEUTRAL = 0x38  # e4m3 exp bias -> 1.0
E4M3_SCALE_MEAN, E4M3_SCALE_STD = 0.34375, 0.08
POW2_BINOMIAL_N = 10
E8M0_SCALE_DISTS = ("zero", "constant", "uniform", "norm", "auto", "pow2_binomial")
E4M3_SCALE_DISTS = ("zero", "constant", "uniform", "norm", "auto")
_STAGE_ELEMS = 1 << 28  # 256M f32 = 1 GiB per chunk


def make_generator(seed, device="cuda"):
    """Seeded ``torch.Generator`` -- same seed => bit-identical buffers."""
    return torch.Generator(device=device).manual_seed(int(seed))


def add_data_init_args(
    parser,
    *,
    default_dist="uniform",
    default_scale="constant",
    default_seed=0,
    include_scale=True,
):
    """Attach shared data initialization arguments to a benchmark parser."""
    parser.add_argument(
        "--data-init",
        dest="data_init",
        nargs="+",
        choices=list(DATA_DISTS),
        default=[default_dist],
        help="DATA init: zero | constant | uniform | norm (N(0,1)). "
        "e.g.: --data-init uniform norm",
    )
    if include_scale:
        parser.add_argument(
            "--scale-init",
            dest="scale_init",
            nargs="+",
            choices=list(SCALE_DISTS),
            default=[default_scale],
            help="SCALE init (non-negative float): zero | constant(=1) | "
            "uniform U(0.5,2) | norm N(1,0.25). Independent of --data-init.",
        )
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed,
        help="RNG seed; same seed -> bit-identical uniform/norm buffers",
    )
    return parser


def _row_chunks(rows, cols):
    """Row slices whose f32 staging stays around _STAGE_ELEMS elements."""
    step = max(_STAGE_ELEMS // max(cols, 1), 1)
    for start in range(0, rows, step):
        yield start, min(start + step, rows)


def _canon_dist(dist, allowed):
    if dist == "gaussian":
        dist = "norm"
    if dist not in allowed:
        raise ValueError(f"dist {dist!r}; choose from {allowed}")
    return dist


def _sample_data_f32(shape, dist, gen, *, lo, hi, device):
    if dist == "uniform":
        return torch.empty(shape, dtype=torch.float32, device=device).uniform_(
            lo, hi, generator=gen
        )
    if dist == "norm":
        return torch.empty(shape, dtype=torch.float32, device=device).normal_(
            0.0, 1.0, generator=gen
        )
    raise ValueError(f"data dist {dist!r} is not continuous; use fill dispatch")


def _sample_scale_f32(shape, dist, gen, *, lo, hi, device):
    if dist == "uniform":
        v = torch.empty(shape, dtype=torch.float32, device=device).uniform_(
            lo, hi, generator=gen
        )
    elif dist == "norm":
        v = torch.empty(shape, dtype=torch.float32, device=device).normal_(
            SCALE_NORM_MEAN, SCALE_NORM_STD, generator=gen
        )
    else:
        raise ValueError(f"scale dist {dist!r} is not continuous; use fill_scale")
    v.clamp_(min=0.0)
    return v


def _fill_sampled(shape, dist, gen, *, dtype, device, uniform, constant, sample_fn):
    if dist == "zero":
        return torch.zeros(shape, dtype=dtype, device=device)
    if dist == "constant":
        return torch.full(shape, constant, dtype=dtype, device=device)
    lo, hi = uniform
    if len(shape) != 2:
        return sample_fn(shape, dist, gen, lo=lo, hi=hi, device=device).to(dtype)
    rows, cols = shape
    out = torch.empty(shape, dtype=dtype, device=device)
    for r0, r1 in _row_chunks(rows, cols):
        v = sample_fn((r1 - r0, cols), dist, gen, lo=lo, hi=hi, device=device)
        out[r0:r1] = v.to(dtype)
        del v
    return out


def fill(
    shape,
    dist,
    gen,
    *,
    dtype=torch.float32,
    device="cuda",
    uniform=(-1.0, 1.0),
    constant=1.0,
):
    """Return a ``dtype`` DATA tensor of ``shape``.

    ``dist`` in {zero, constant, uniform, norm}. ``uniform`` is U(lo, hi);
    ``norm`` / ``gaussian`` is N(0, 1). ``zero`` / ``constant`` ignore ``gen``.
    """
    dist = _canon_dist(dist, DATA_DISTS)
    return _fill_sampled(
        shape,
        dist,
        gen,
        dtype=dtype,
        device=device,
        uniform=uniform,
        constant=constant,
        sample_fn=_sample_data_f32,
    )


def fill_scale(
    shape,
    dist,
    gen,
    *,
    dtype=torch.float32,
    device="cuda",
    uniform=SCALE_UNIFORM,
    constant=1.0,
):
    """Return a non-negative float SCALE tensor of ``shape``.

    Same dist names as ``fill``, sampled independently. ``constant`` defaults
    to 1.0 (neutral). ``norm`` is N(1, 0.25) clamped >= 0 -- not DATA's N(0,1).
    For MX on-wire scales use ``fill_scale_e8m0`` / ``fill_scale_e4m3``.
    """
    dist = _canon_dist(dist, SCALE_DISTS)
    return _fill_sampled(
        shape,
        dist,
        gen,
        dtype=dtype,
        device=device,
        uniform=uniform,
        constant=constant,
        sample_fn=_sample_scale_f32,
    )


def _f32_to_e8m0(v: torch.Tensor) -> torch.Tensor:
    """Round positive floats to the nearest E8M0 on-wire byte (bias 127)."""
    e = torch.zeros_like(v, dtype=torch.int32)
    pos = v > 0
    e[pos] = v[pos].log2().round().to(torch.int32) + E8M0_BIAS
    return e.clamp_(0, 255).to(torch.uint8)


def _popcount64(x: torch.Tensor) -> torch.Tensor:
    """Population count for a non-negative int64 tensor (SWAR bit-hack)."""
    x = x - ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0F
    return (x * 0x0101010101010101) >> 56


def fill_fp4(shape, dist, gen, *, uniform=FP4_UNIFORM, device="cuda", constant=0):
    """MXFP4 on-wire: packed e2m1 ``uint8`` of shape ``(rows, cols // 2)``.

    Samples with the same DATA dists as ``fill``, then round-to-nearest e2m1.
    ``shape`` is the logical ``(rows, cols)``; ``cols`` must be even.
    """
    dist = _canon_dist(dist, DATA_DISTS)
    rows, cols = shape
    assert cols % 2 == 0, f"FP4 needs even columns, got {cols}"
    packed = (rows, cols // 2)
    if dist == "zero":
        return torch.zeros(packed, dtype=torch.uint8, device=device)
    if dist == "constant":
        return torch.full(packed, int(constant), dtype=torch.uint8, device=device)

    from aiter.utility import fp4_utils  # local: fp4_utils pulls in triton

    out = torch.empty(packed, dtype=torch.uint8, device=device)
    for r0, r1 in _row_chunks(rows, cols):
        v = _sample_data_f32(
            (r1 - r0, cols),
            dist,
            gen,
            lo=uniform[0],
            hi=uniform[1],
            device=device,
        )
        out[r0:r1] = fp4_utils.f32_to_mxfp4(v).view(torch.uint8)
        del v
    return out


def fill_fp8(
    shape,
    dist,
    gen,
    *,
    dtype=FP8_E4M3,
    uniform=FP8_UNIFORM,
    device="cuda",
    constant=0.5,
):
    """MXFP8 on-wire: e4m3 tensor of ``shape``."""
    dist = _canon_dist(dist, DATA_DISTS)
    if dist == "zero":
        return torch.zeros(shape, dtype=dtype, device=device)
    if dist == "constant":
        return torch.full(
            shape, float(constant), dtype=torch.float32, device=device
        ).to(dtype)
    if len(shape) != 2:
        v = _sample_data_f32(
            shape,
            dist,
            gen,
            lo=uniform[0],
            hi=uniform[1],
            device=device,
        )
        return v.to(dtype)
    rows, cols = shape
    out = torch.empty(shape, dtype=dtype, device=device)
    for r0, r1 in _row_chunks(rows, cols):
        v = _sample_data_f32(
            (r1 - r0, cols),
            dist,
            gen,
            lo=uniform[0],
            hi=uniform[1],
            device=device,
        )
        out[r0:r1] = v.to(dtype)
        del v
    return out


def fill_scale_e8m0(
    shape,
    dist="auto",
    gen=None,
    *,
    device="cuda",
    n=POW2_BINOMIAL_N,
    constant=E8M0_NEUTRAL,
):
    """MX E8M0 on-wire ``uint8`` (biased exponent, bias 127).

    ``zero`` / ``constant`` / ``uniform`` / ``norm`` map from our SCALE dists
    (float then round to nearest power-of-two byte). ``auto`` /
    ``pow2_binomial`` match the MX GEMM default: 2^(Binomial(21,0.5)-11).
    """
    if dist == "gaussian":
        dist = "norm"
    if dist not in E8M0_SCALE_DISTS:
        raise ValueError(f"E8M0 scale dist {dist!r}; choose from {E8M0_SCALE_DISTS}")
    if dist == "zero":
        return torch.zeros(shape, dtype=torch.uint8, device=device)
    if dist == "constant":
        return torch.full(shape, int(constant), dtype=torch.uint8, device=device)
    if dist in ("uniform", "norm"):
        v = fill_scale(shape, dist, gen, device=device)
        return _f32_to_e8m0(v)
    # auto / pow2_binomial: Binomial(k, 0.5) == popcount of a uniform k-bit int
    trials = 2 * n + 1
    assert trials <= 24, "pow2_binomial popcount path assumes <= 24 trials"
    bits = torch.randint(
        0, 1 << trials, shape, dtype=torch.int64, device=device, generator=gen
    )
    e = _popcount64(bits).to(torch.int32) - (n + 1)
    return (e + E8M0_BIAS).clamp_(0, 255).to(torch.uint8)


def fill_scale_e4m3(
    shape, dist="auto", gen=None, *, device="cuda", constant=E4M3_NEUTRAL
):
    """NVFP4 / E4M3 on-wire ``uint8``.

    ``auto`` -> N(0.34375, 0.08) clamped >= 0, then cast e4m3 (MX GEMM default).
    ``uniform`` / ``norm`` use ``fill_scale`` then cast. ``constant`` is 0x38
    (1.0).
    """
    if dist == "gaussian":
        dist = "auto"
    if dist not in E4M3_SCALE_DISTS:
        raise ValueError(f"E4M3 scale dist {dist!r}; choose from {E4M3_SCALE_DISTS}")
    if dist == "zero":
        return torch.zeros(shape, dtype=torch.uint8, device=device)
    if dist == "constant":
        return torch.full(shape, int(constant), dtype=torch.uint8, device=device)
    if dist in ("uniform", "norm"):
        v = fill_scale(shape, dist, gen, device=device)
        return v.to(FP8_E4M3).view(torch.uint8)
    v = torch.empty(shape, dtype=torch.float32, device=device).normal_(
        E4M3_SCALE_MEAN, E4M3_SCALE_STD, generator=gen
    )
    v.clamp_(min=0.0)
    return v.to(FP8_E4M3).view(torch.uint8)

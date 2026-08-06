# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``flydsl_rot_quant`` -- fused online Hadamard rotation + MXFP4 quant.

Three independent things are checked, because each has its own failure mode and
all three fail *silently* (plausible-looking numbers, no exception):

1. ``test_rot_quant_bitexact`` -- packed fp4 and e8m0 scales byte-for-byte against
   a pure-torch reference (``_ref_rot_quant`` below). Byte equality, not allclose:
   the kernel's claim is bit-exactness, and allclose would hide a systematic
   off-by-one-binade scale bug.
2. ``test_rot_quant_fused_shuffle`` -- the in-kernel CK scale swizzle against
   ``aiter.ops.shuffle.shuffle_scale`` applied to the natural-layout output.
   Compared only on the unpadded region; the padding cells are ``torch.empty``.
3. ``test_rot_quant_gemm_a4w4`` -- end-to-end through ``gemm_a4w4``, the consumer
   the swizzle exists for. A layout mismatch between quantizer and gemm produces
   garbage output with no error, so unit-level parity alone is not sufficient
   evidence that the op works.

Plus ``test_rot_quant_rejects_bad_input``: the input contract is load-bearing
(fp16 or a strided view would be read as contiguous bf16 and silently yield
garbage), so the guards must raise rather than compute.
"""

import argparse
import math

import torch

from aiter import dtypes
from aiter.ops.flydsl import flydsl_rot_quant
from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import (
    f32_to_mx_e8m0_scale,
    f32_to_mxfp4,
    mxfp4_to_f32,
)
from aiter.utility.mx_types import MX_DEFAULT_ROUND_MODE, MxDtypeInt

QG = 32  # mxfp4 group size

# aiter's CK asm gemm_a4w4 is numerically wrong below this N (rel err ~0.94 at
# N=64, exact at N>=512), so it is not usable as a reference there. The op itself
# is fine at any N -- this bound is a property of the gemm, not the quantizer.
CK_MIN_N = 512


def _ref_rot_quant(x, RS):
    """Pure-torch reference for ``flydsl_rot_quant(..., shuffle_scales=False)``.

    Mirrors the kernel step for step so the comparison can be exact:

    * Sylvester FWHT over each RS-wide block, stage order ``h = 1, 2, ... RS/2``,
      pairing ``(lo, lo+h) -> (a+b, a-b)``. Same op order => same f32 rounding.
    * The ``1/sqrt(RS)`` normalization is folded into the e8m0 exponent when
      ``log2(RS)/2`` is an integer (i.e. RS=64), and applied as an f32 multiply
      otherwise (RS=32, 128). Either way the result is round-tripped through
      bf16, matching the kernel's ``.to(BFloat16).to(Float32)``.
    * The block scale is aiter's default MXFP4 convention, taken from the shared
      ``f32_to_mx_e8m0_scale(mode=RoundUp, dtype=FP4_E2M1)`` = ``ceil_pow2(amax/6)``
      helper -- the CPU mirror of the ``emit_mx_e8m0_scale`` IR builder the kernel
      itself uses. ``fold_k`` is then subtracted from the exponent (clamped at 0),
      which is exact because RoundUp is exponent-linear.
    """
    M, K = x.shape
    v = x.float().view(M, K // RS, RS)
    h = 1
    while h < RS:
        t = v.reshape(M, K // RS, RS // (2 * h), 2, h)
        a, b = t[..., 0, :], t[..., 1, :]
        v = torch.stack([a + b, a - b], dim=-2).reshape(M, K // RS, RS)
        h *= 2

    hl = 0.5 * math.log2(RS)
    fold_k = int(hl) if float(hl).is_integer() else 0
    if not fold_k:
        v = v * (1.0 / math.sqrt(RS))
    v = v.to(torch.bfloat16).float().reshape(M, K)

    g = v.reshape(M, K // QG, QG)
    amax = g.abs().amax(dim=-1).float()
    e8 = (
        f32_to_mx_e8m0_scale(
            amax, mode=MX_DEFAULT_ROUND_MODE, dtype=MxDtypeInt.FP4_E2M1
        )
        .view(torch.uint8)
        .to(torch.int32)
    )
    if fold_k:
        e8 = torch.clamp(e8 - fold_k, min=0)
    hw = ((e8 + fold_k) << 23).view(torch.float32)

    fp4 = f32_to_mxfp4((g / hw.unsqueeze(-1)).reshape(M, K)).view(torch.uint8)
    return fp4, e8.to(torch.uint8)


def _dequant(fp4, scales):
    """Unpack (fp4, natural-layout e8m0) back to f32 for the gemm reference."""
    vals = mxfp4_to_f32(fp4).float()
    exp = scales.to(torch.int32).repeat_interleave(QG, dim=-1)
    return vals * torch.exp2((exp - 127).float())


def _inputs(M, K, seed=0, outliers=False):
    torch.manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.5
    if outliers:
        # Force the top of the fp4 range: RoundUp puts amax at (3, 6], so a group
        # whose rounding lands high drives elements right up against the 6.0
        # saturation point, which is where an encoder that clamps differently
        # from the hardware would diverge.
        x[:, ::37] *= 64
    return x


@benchmark()
def test_rot_quant_bitexact(M, K, RS, outliers=False):
    """fp4 bytes and e8m0 scales must match the torch reference exactly."""
    x = _inputs(M, K, outliers=outliers)
    (fp4, scales), us = run_perftest(flydsl_rot_quant, x, RS)
    ref_fp4, ref_scales = _ref_rot_quant(x, RS)

    n_fp4 = (fp4 != ref_fp4).sum().item()
    n_sc = (scales != ref_scales).sum().item()
    assert n_fp4 == 0, f"{n_fp4}/{fp4.numel()} fp4 bytes differ (M={M} K={K} RS={RS})"
    assert (
        n_sc == 0
    ), f"{n_sc}/{scales.numel()} scale bytes differ (M={M} K={K} RS={RS})"
    assert fp4.shape == (M, K // 2) and scales.shape == (M, K // QG)

    gbps = (x.numel() * 2 + fp4.numel() + scales.numel()) / us / 1e3
    return {"us": us, "GB/s": gbps}


def test_rot_quant_fused_shuffle(M, K, RS):
    """The in-kernel CK swizzle must equal shuffle_scale() of the natural layout.

    Only the unpadded ``[M, K//QG]`` region is compared: the kernel allocates the
    padded buffer with ``torch.empty`` and never writes the pad cells (the gemm
    slices its output to ``[:M]``), so a full-tensor compare would read
    uninitialized memory and flake.
    """
    x = _inputs(M, K)
    fp4_nat, sc_nat = flydsl_rot_quant(x, RS, shuffle_scales=False)
    fp4_shf, sc_shf = flydsl_rot_quant(x, RS, shuffle_scales=True)

    assert torch.equal(
        fp4_nat, fp4_shf
    ), "fp4 output must not depend on the scale layout"

    n = K // QG
    sm, sn = (M + 255) // 256 * 256, (n + 7) // 8 * 8
    assert sc_shf.shape == (
        sm,
        sn,
    ), f"expected padded {(sm, sn)}, got {tuple(sc_shf.shape)}"

    ref = shuffle_scale(sc_nat).view(sm, sn)
    # Which destination cells carry real data. ``shuffle_scale`` pads with
    # ``torch.empty``, so the mask cannot be recovered from its output -- push a
    # deterministic 0/1 mask through the same reshape/permute instead.
    mask = torch.zeros((sm, sn), dtype=torch.uint8, device=sc_nat.device)
    mask[:M, :n] = 1
    live = (
        mask.view(sm // 32, 2, 16, sn // 8, 2, 4)
        .permute(0, 3, 5, 2, 4, 1)
        .reshape(sm, sn)
        != 0
    )
    assert live.sum().item() == M * n, "mask permutation does not preserve the payload"
    n_diff = ((sc_shf != ref) & live).sum().item()
    assert n_diff == 0, f"{n_diff}/{live.sum().item()} live scale bytes differ"
    print(f"✓ fused swizzle == shuffle_scale (M={M} K={K} RS={RS})")


def test_rot_quant_gemm_a4w4(M, N, K, RS):
    """End-to-end: quantize with the fused swizzle, feed the CK asm gemm."""
    if N < CK_MIN_N:
        print(f"skip N={N}: gemm_a4w4 is not a valid reference below N={CK_MIN_N}")
        return
    from aiter import gemm_a4w4
    from aiter.utility.fp4_utils import dynamic_mxfp4_quant

    x = _inputs(M, K)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    wq, ws = dynamic_mxfp4_quant(w)

    xq, xs = flydsl_rot_quant(x, RS, shuffle_scales=True)
    out = gemm_a4w4(
        xq,
        shuffle_weight(wq, layout=(16, 16)),
        xs,
        shuffle_scale(ws),
        dtype=dtypes.bf16,
    )[:M]

    # Reference: dequantize the same bytes the gemm consumed and matmul in f32.
    # This isolates the gemm/layout contract -- quantization error is already
    # accounted for because both sides start from the identical fp4 payload.
    xq_nat, xs_nat = flydsl_rot_quant(x, RS, shuffle_scales=False)
    ref = (
        _dequant(xq_nat, xs_nat)
        @ _dequant(wq.view(torch.uint8), ws.view(torch.uint8)).T
    )

    checkAllclose(
        out.float(), ref, rtol=1e-2, atol=1e-2, msg=f"M={M} N={N} K={K} RS={RS}"
    )


def test_rot_quant_rejects_bad_input():
    """Guards must raise. Each of these would otherwise be read as contiguous
    bf16 and produce plausible garbage."""
    K = 4096
    good = torch.randn(64, K, device="cuda", dtype=torch.bfloat16)
    cases = [
        ("fp16 input", lambda: flydsl_rot_quant(good.half(), 64)),
        ("fp32 input", lambda: flydsl_rot_quant(good.float(), 64)),
        ("non-contiguous", lambda: flydsl_rot_quant(good.t().contiguous().t(), 64)),
        ("3D input", lambda: flydsl_rot_quant(good.view(2, 32, K), 64)),
        ("K % RS != 0", lambda: flydsl_rot_quant(good[:, :100].contiguous(), 64)),
        ("unsupported RS", lambda: flydsl_rot_quant(good, 16)),
        ("bad AMAX", lambda: flydsl_rot_quant(good, 64, AMAX="fastest")),
    ]
    for name, fn in cases:
        try:
            fn()
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"{name}: expected a raise, got a result")
    print(f"✓ all {len(cases)} input guards raise")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="flydsl_rot_quant: fused Hadamard rotation + MXFP4 quant"
    )
    parser.add_argument("-m", type=int, default=None, help="M dimension (rows)")
    parser.add_argument("-n", type=int, default=None, help="N dimension (gemm test)")
    parser.add_argument("-k", type=int, default=None, help="K dimension (cols)")
    parser.add_argument("-r", "--rs", type=int, default=None, help="rotation size")
    args = parser.parse_args()

    l_m = [args.m] if args.m else [1, 32, 256, 1024, 4096, 16384]
    l_k = [args.k] if args.k else [2048, 4096]
    l_rs = [args.rs] if args.rs else [32, 64, 128]
    l_n = [args.n] if args.n else [512, 4096, 4608]

    for RS in l_rs:
        for K in l_k:
            for M in l_m:
                test_rot_quant_bitexact(M, K, RS)
    # Outlier pass: drives the saturating top of the fp4 range.
    for RS in l_rs:
        test_rot_quant_bitexact(1024, 4096, RS, outliers=True)

    for RS in l_rs:
        for M in [1, 32, 256, 1024, 4096]:
            test_rot_quant_fused_shuffle(M, 4096, RS)

    for N in l_n:
        for M in [1, 32, 1024]:
            test_rot_quant_gemm_a4w4(M, N, 4096, 64)

    test_rot_quant_rejects_bad_input()

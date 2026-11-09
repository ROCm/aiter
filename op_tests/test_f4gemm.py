# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# ============================================================================
# gfx1250 F4GEMM ASM Support Matrix (256x256 persistent kernels)
# ----------------------------------------------------------------------------
#  OUTTYPE | A_PRESHUFFLE | B_PRESHUFFLE | INTYPE |   M    |   N    |   K
# ---------+--------------+--------------+--------+--------+--------+--------
#  BF16    |      0       |      1       | MXFP4  | %1==0  | %16==0 | %32==0
#  BF16    |      0       |      1       | NVFP4  | %1==0  | %16==0 | %32==0
#  BF16    |      1       |      1       | MXFP4  | %16==0 | %16==0 | %32==0
#  BF16    |      1       |      1       | NVFP4  | %16==0 | %16==0 | %32==0
#  FP8     |      0       |      1       | MXFP4  | %1==0  | %16==0 | %32==0
#  FP8     |      0       |      1       | NVFP4  | %1==0  | %16==0 | %32==0
#  FP8     |      1       |      1       | MXFP4  | %16==0 | %16==0 | %32==0
#  FP8     |      1       |      1       | NVFP4  | %16==0 | %16==0 | %32==0
# ----------------------------------------------------------------------------
# Notes:
#  - B_PRESHUFFLE is always 1 (B is always pre-shuffled).
#  - A_PRESHUFFLE=1 tightens the M constraint from %1==0 to %16==0.
#  - K is always a multiple of 32.
#  - The current 192x256 NPS path requires MXFP4/BF16/AP0, M%768==0,
#    N%1024==0 and K%256==0, with a fixed 4x4 cluster. Automatic NPS
#    selection is limited to 48 clusters; larger shapes use persistent kernels.
# ============================================================================

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.benchmark_data_init import (
    fill_fp4,
    fill_scale_e4m3,
    fill_scale_e8m0,
    make_generator,
)
from aiter.jit.utils.asm_guard import is_gfx1250_asm_supported
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a4w4 import (
    MXFP8_OUT_SCALE_BLOCK,
    unpack_mxfp8_out_scale,
)
from aiter.ops.shuffle import (
    shuffle_scale,
    shuffle_scale_f4,
    shuffle_weight,
    shuffle_weight_f4,
)
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.utility import fp4_utils

# Run this CLI directly; --mode func includes the API regression checks.
__test__ = False

SUPPORTED_GFX = ["gfx1250"]

_OUT_DTYPE = {"bf16": dtypes.bf16, "fp8": dtypes.fp8}
_NOT_SUPPORTED_MARKERS = (
    "cannot get heuristic kernel",
    "kernel not in cfg_f4gemm",
    "no kernel registered for f4gemm",
)
_PS_KERNEL = "_ZN5aiter44f4gemm_bf16_mxfp4_BpreShuffle_256x256_4x4_psE"


def _observed_kernel(fn, args):
    """Observe one warmed-up launch outside timing; selection stays in C++."""
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        fn(*args)
        torch.cuda.synchronize()
    names = {
        event.name.removeprefix("aiter::")
        for event in prof.events()
        if str(event.device_type).endswith("CUDA") and "f4gemm_" in event.name
    }
    assert len(names) == 1, f"expected one F4GEMM kernel in the trace, got {names}"
    return names.pop()


PERF_SHAPES = [
    (1536, 24576, 1536),
    (1536, 7168, 16384),
    (12288, 16384, 16384),  # Large aligned shape checks persistent fallback.
    (16384, 16384, 16384),  # Persistent throughput baseline.
]
FUNC_SHAPES = [
    # Small/partial tiles and runtime cluster shapes along both axes.
    (1, 256, 256),
    (17, 256, 256),
    (32, 256, 256),
    (768, 256, 256),
    (512, 512, 256),
    (1024, 256, 256),
    (512, 1024, 256),
    (1024, 512, 256),
    (1024, 1024, 256),
    # NPS: one cluster, the K6 boundary, and the production shape.
    (768, 1024, 256),
    (768, 1024, 1536),
    (768, 1024, 1792),
    (1536, 24576, 1536),
    (1536, 7168, 16384),
    # Alignment and automatic-selection size boundaries.
    (767, 1024, 256),
    (768, 1008, 256),
    (768, 1024, 288),
    (768, 49152, 256),
    (768, 50176, 256),
    (12288, 16384, 16384),
    # pure_compute
    (256, 2048, 8192),
    (2048, 8192, 8192),
    (16384, 16384, 16384),
    # (32768, 106496, 16384),
    # (32768, 16384, 53248),
    # (32768, 18432, 16384),
    # (32768, 16384, 16384),
    (128, 106496, 16384),
    (128, 16384, 53248),
    (128, 18432, 16384),
    (128, 16384, 16384),
    (64, 106496, 16384),
    (64, 16384, 53248),
    (64, 18432, 16384),
    (64, 16384, 16384),
    (32, 106496, 16384),
    (32, 16384, 53248),
    (32, 18432, 16384),
    (32, 16384, 16384),
    # qkv_proj
    (1, 1280, 8192),
    (64, 1280, 8192),
    (127, 1280, 8192),
    (129, 1280, 8192),
    (65, 1280, 8192),
    (32, 1280, 8192),
    (128, 1280, 8192),
    (192, 1280, 8192),
    (256, 1280, 8192),
    (320, 1280, 8192),
    (512, 1280, 8192),
    (1024, 1280, 8192),
    (2048, 1280, 8192),
    (4096, 1280, 8192),
    (8192, 1280, 8192),
    # attn_out
    (1, 8192, 1024),
    (32, 8192, 1024),
    (64, 8192, 1024),
    (128, 8192, 1024),
    (192, 8192, 1024),
    (256, 8192, 1024),
    (320, 8192, 1024),
    (512, 8192, 1024),
    (1024, 8192, 1024),
    (2048, 8192, 1024),
    (4096, 8192, 1024),
    (8192, 8192, 1024),
    (16384, 8192, 1024),
    # tune
    (1552, 8192, 8192),
    (1664, 8192, 8192),
    (1792, 8192, 8192),
    (1920, 8192, 8192),
    (3072, 8192, 8192),
    (1552, 10240, 8192),
    (1664, 10240, 8192),
    (1792, 10240, 8192),
    (1920, 10240, 8192),
    (3072, 10240, 8192),
    (1552, 57344, 8192),
    (1664, 57344, 8192),
    (1792, 57344, 8192),
    (1920, 57344, 8192),
    (3072, 57344, 8192),
    (1552, 8192, 28672),
    (1664, 8192, 28672),
    (1792, 8192, 28672),
    (1920, 8192, 28672),
    (3072, 8192, 28672),
    (128, 1280, 8224),
    # partial_tile
    (128, 384, 8192),
    (128, 272, 8192),
    (65, 384, 8192),
]

MXFP4_SCALE_BLOCK = 32
NVFP4_SCALE_BLOCK = 16
# MXFP8_OUT_SCALE_BLOCK (=128) is imported from gemm_op_a4w4.


# mxfp8 output E8M0 scale, mirrors the fp8out kernel: amax/256, then
# exp[30:23] + guard[22] (single guard-bit round, no RNE). Compared with a
# +-1 e8m0-step tolerance (see the scale checkAllclose below).
def _e8m0_out_scale(amax):
    u = (amax / 256.0).view(torch.int32)
    return (((u >> 23) & 0xFF) + ((u >> 22) & 1)).to(torch.uint8)


def _quant_mxfp8_blockN(x_f32, block=MXFP8_OUT_SCALE_BLOCK):
    """Golden mxfp8 output quant: per-128-col block amax -> E8M0 scale (kernel-
    exact) + e4m3 data. Returns (fp8 [M,N], e8m0 row-major [M, N/block])."""
    M, N = x_f32.shape
    assert N % block == 0, f"mxfp8 golden requires N % {block} == 0"
    xb = x_f32.reshape(M, N // block, block)
    amax = xb.abs().amax(dim=-1).clamp(min=torch.finfo(torch.float32).tiny)
    scale_e8m0 = _e8m0_out_scale(amax)
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0).unsqueeze(-1)
    q_fp8 = (xb / scale_f32).reshape(M, N).to(dtypes.fp8)
    return q_fp8, scale_e8m0


def _dequant_mxfp8_blockN(q_fp8, scale_e8m0, block=MXFP8_OUT_SCALE_BLOCK):
    """Inverse of :func:`_quant_mxfp8_blockN`; ``scale_e8m0`` must be row-major."""
    M, N = q_fp8.shape
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0).unsqueeze(-1)
    return (q_fp8.float().reshape(M, N // block, block) * scale_f32).reshape(M, N)


# checkAllclose returns 0 when all-close, else the mismatch fraction. Its own
# verdict thresholds: pass (0) / warning (<= tol_err_ratio) / failed (above).
_TOL_ERR_RATIO = 0.05  # matches checkAllclose default tol_err_ratio


def _verdict(err):
    if err == 0:
        return "pass"
    return "warning" if err <= _TOL_ERR_RATIO else "failed"


def _support_reason(outtype, apre, M, N, K):
    """Check data-preparation and reference constraints before allocating tensors.
    Kernel selection remains in the C++ entrypoint."""
    if M <= 0 or N <= 0 or K <= 0:
        return "M/N/K must be positive"
    if outtype not in _OUT_DTYPE:
        return f"outtype {outtype}"  # no kernel for this output format yet
    if K % 32 != 0:
        return "K%32"  # B 16x16 preshuffle
    if N % 16 != 0:
        return "N%16"  # B 16x16 preshuffle
    if apre and M % 16 != 0:
        return "apre M%16"  # A 16x16 preshuffle
    if outtype == "fp8" and N % MXFP8_OUT_SCALE_BLOCK != 0:
        return "fp8 N%128"  # per-128 output-scale golden limitation
    return None


def _e4m3_to_f32(s: torch.Tensor) -> torch.Tensor:
    return s.view(torch.float8_e4m3fn).to(torch.float32)


def run_torch_mxfp4(xq, wq, xs, ws):
    # Reference only: fp32 math. Returns fp32; the caller casts to bf16 or
    # quantizes per outtype. Not timed, not in the table.
    x_f32 = fp4_utils.mxfp4_to_f32(xq)
    w_f32 = fp4_utils.mxfp4_to_f32(wq)
    xs = fp4_utils.e8m0_to_f32(xs).repeat_interleave(MXFP4_SCALE_BLOCK, dim=1)
    ws = fp4_utils.e8m0_to_f32(ws).repeat_interleave(MXFP4_SCALE_BLOCK, dim=1)
    return (x_f32 * xs) @ (w_f32 * ws).T


def run_torch_nvfp4(xq, wq, xs, ws, gA, gB):
    # Reference only: fp32 math. Returns fp32 (see run_torch_mxfp4).
    x_f32 = fp4_utils.mxfp4_to_f32(xq)
    w_f32 = fp4_utils.mxfp4_to_f32(wq)
    xs = _e4m3_to_f32(xs).repeat_interleave(NVFP4_SCALE_BLOCK, dim=1)
    ws = _e4m3_to_f32(ws).repeat_interleave(NVFP4_SCALE_BLOCK, dim=1)
    return float(gA) * float(gB) * (x_f32 * xs) @ (w_f32 * ws).T


def _gluon_support_reason(intype, outtype, apre, M, N, K):
    if get_gfx() != "gfx1250":
        return "Gluon comparison requires gfx1250"
    if intype != "mxfp4" or outtype != "bf16":
        return "Gluon comparison requires --intype mxfp4 --outtype bf16"
    if K % 256:
        return f"Gluon preshuffled scales require K % 256 == 0, got K={K}"
    return _support_reason(outtype, apre, M, N, K)


def run_gluon_mxfp4(xq, wq, xs, ws):
    """Run the benchmark's Gluon entry on the same logical data and scales.

    ASM's shuffle_scale_f4 layout differs from Gluon's shuffle_scale layout.
    Start from the original buffers, including row-major A for --apre 1.
    """
    from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import (
        gemm_afp4wfp4_preshuffle,
    )

    M = xq.shape[0]
    N, K_bytes = wq.shape
    w_shuf = shuffle_weight(wq).reshape(N // 16, K_bytes * 16)
    if M >= 32:
        xs_shuf = shuffle_scale(xs)
        xs_shuf = xs_shuf.view(-1, xs_shuf.shape[1] * 32)
    else:
        xs_shuf = xs.contiguous()
    ws_shuf = shuffle_scale(ws)
    ws_shuf = ws_shuf.view(-1, ws_shuf.shape[1] * 32)
    y = torch.empty((M, N), dtype=dtypes.bf16, device=xq.device)
    return gemm_afp4wfp4_preshuffle(xq, w_shuf, xs_shuf, ws_shuf, dtypes.bf16, y)


def _compare_gluon_output(out, gluon_out):
    """Compare BF16 storage bits, counting mismatched output elements.

    Float equality would miss +0/-0 differences and discard NaN payloads.
    Numeric error is diagnostic only; every bit mismatch fails this check.
    """
    assert (
        out.shape == gluon_out.shape
    ), f"ASM/Gluon shape mismatch: {tuple(out.shape)} vs {tuple(gluon_out.shape)}"
    assert (
        out.dtype == gluon_out.dtype == dtypes.bf16
    ), f"ASM/Gluon bit-match requires BF16 outputs: {out.dtype} vs {gluon_out.dtype}"
    out_bits = out.contiguous().view(torch.int16)
    gluon_bits = gluon_out.contiguous().view(torch.int16)
    mismatch = out_bits != gluon_bits
    count = int(mismatch.count_nonzero().item())
    total = out.numel()
    ratio = count / total
    max_abs = (out.float() - gluon_out.float()).abs().max().item() if count else 0.0
    result = "failed" if count else "pass"
    log = aiter.logger.warning if count else aiter.logger.info
    log(
        "ASM/Gluon bit-match %s: %d/%d BF16 elements differ (%.8f%%), max_abs=%g",
        result,
        count,
        total,
        100 * ratio,
        max_abs,
    )
    if count:
        # Limit diagnostic indexing to the first bad row: materializing every
        # mismatch coordinate can consume hundreds of MB for benchmark shapes.
        row = int(mismatch.any(dim=1).to(torch.int8).argmax().item())
        cols = mismatch[row].nonzero().flatten()[:5].tolist()
        for col in cols:
            log(
                "  [%d, %d]: ASM=%g (0x%04x), Gluon=%g (0x%04x)",
                row,
                col,
                out[row, col].item(),
                int(out_bits[row, col].item()) & 0xFFFF,
                gluon_out[row, col].item(),
                int(gluon_bits[row, col].item()) & 0xFFFF,
            )
    return {
        "gluon bitmatch": result,
        "gluon mismatches": count,
        "gluon mismatch ratio": ratio,
        "gluon max abs": max_abs,
    }


def _prep_mxfp4(M, N, K, apre, data_init, scale_init, gen, check_gluon=False):
    # DATA (fp4 e2m1, packed 2/byte). data & scale are sampled *independently*.
    # constant is built inside fill_fp4 via constant=; f4gemm.cpp data_init=0
    # keeps A=0x22, B=0x33 (fixed representable e2m1).
    xq = fill_fp4((M, K), data_init, gen, constant=0x22)
    wq = fill_fp4((N, K), data_init, gen, constant=0x33)
    # SCALE (e8m0 per-32). auto -> pow2_binomial for E8M0; constant default is
    # the neutral e8m0 byte 0x7F (exp 0 -> 2^0 = 1.0).
    xs = fill_scale_e8m0((M, K // MXFP4_SCALE_BLOCK), scale_init, gen)
    ws = fill_scale_e8m0((N, K // MXFP4_SCALE_BLOCK), scale_init, gen)
    ref = run_torch_mxfp4(xq, wq, xs, ws)
    inp = {
        "A": shuffle_weight_f4(xq) if apre else xq,
        "B": shuffle_weight_f4(wq),
        "sA": shuffle_scale_f4(xs, 7),
        "sB": shuffle_scale_f4(ws, 7),
        "gA": None,
        "gB": None,
    }
    if check_gluon:
        # Preserve the exact sampled buffers; do not regenerate from a seed or
        # reuse ASM's shuffled scales as if they had Gluon's layout.
        inp["gluon_args"] = (xq, wq, xs, ws)
    return inp, ref


def _prep_nvfp4(M, N, K, apre, data_init, scale_init, gen):
    # DATA (fp4 e2m1). data & scale sampled independently (test_common).
    # constant built inside fill_fp4 via constant=; f4gemm.cpp data_init=0 keeps
    # A=0x22, B=0x33 (fixed representable e2m1).
    xq = fill_fp4((M, K), data_init, gen, constant=0x22)
    wq = fill_fp4((N, K), data_init, gen, constant=0x33)
    # SCALE (e4m3 per-16). auto -> gaussian(0.34375,0.08) for E4M3; constant
    # default is the neutral e4m3 byte 0x38 (exp 7 = bias -> 1.0).
    xs = fill_scale_e4m3((M, K // NVFP4_SCALE_BLOCK), scale_init, gen)
    ws = fill_scale_e4m3((N, K // NVFP4_SCALE_BLOCK), scale_init, gen)
    # Per-tensor global scale is NOT part of the init helpers: keep neutral.
    gA = gB = 1.0
    ref = run_torch_nvfp4(xq, wq, xs, ws, gA, gB)
    inp = {
        "A": shuffle_weight_f4(xq) if apre else xq,
        "B": shuffle_weight_f4(wq),
        "sA": shuffle_scale_f4(xs, 8),
        "sB": shuffle_scale_f4(ws, 8),
        "gA": gA,  # NVFP4 per-tensor global scales (floats)
        "gB": gB,
    }
    return inp, ref


def _check_output_contracts(intypes, outtypes, apres, seed):
    """Exercise output reuse, invalid buffers and empty batches through the public API."""
    M, N, K = 32, 128, 256
    checked = 0
    for intype, apre in itertools.product(intypes, apres):
        prep = _prep_mxfp4 if intype == "mxfp4" else _prep_nvfp4
        gemm = getattr(aiter, f"gemm_{intype}_asm")
        inp, _ = prep(M, N, K, apre, "uniform", "auto", make_generator(seed))
        args = (inp["A"], inp["B"], inp["sA"], inp["sB"])
        if intype == "nvfp4":
            args += (inp["gA"], inp["gB"])
        device = inp["A"].device
        for outtype in outtypes:
            dtype = _OUT_DTYPE[outtype]
            kwargs = {"dtype": dtype, "a_preshuffle": bool(apre)}
            expected = gemm(*args, **kwargs)
            y = torch.empty((M, N), dtype=dtype, device=device)
            y.view(torch.uint8).fill_(0xFF)
            result = gemm(*args, **kwargs, y=y)
            if outtype == "fp8":
                out, scale = result
                expected, expected_scale = expected
                assert torch.equal(
                    unpack_mxfp8_out_scale(scale, M, N).view(torch.uint8),
                    unpack_mxfp8_out_scale(expected_scale, M, N).view(torch.uint8),
                ), "preallocated FP8 output changed its scales"
            else:
                out = result
            assert out is y, "GEMM must return the supplied y"
            assert torch.equal(out.view(torch.uint8), expected.view(torch.uint8))

            for invalid, bad_y in (
                ("shape", torch.empty((M, N + 1), dtype=dtype, device=device)),
                ("dtype", torch.empty((M, N), dtype=torch.float32, device=device)),
                ("device", torch.empty((M, N), dtype=dtype, device="cpu")),
                ("strides", y.t().contiguous().t()),
            ):
                try:
                    gemm(*args, **kwargs, y=bad_y)
                except ValueError as error:
                    assert "y must be contiguous" in str(error), error
                else:
                    raise AssertionError(f"GEMM accepted an invalid y: {invalid}")

            empty_args = (args[0][:0], args[1], args[2][:0], args[3], *args[4:])
            for empty_y in (None, torch.empty((0, N), dtype=dtype, device=device)):
                result = gemm(*empty_args, **kwargs, y=empty_y)
                empty_out = result[0] if outtype == "fp8" else result
                assert empty_out.shape == (0, N) and empty_out.dtype == dtype
                assert empty_out.device == device
                if empty_y is not None:
                    assert empty_out is empty_y
                if outtype == "fp8":
                    assert result[1].shape == (0, 1)
                    assert result[1].dtype == dtypes.fp8_e8m0

            public_args = (
                args[0][:0].view(2, 0, K // 2),
                args[1],
                args[2][:0],
                args[3],
            )
            public_kwargs = {"apreshuffle": bool(apre)}
            if intype == "nvfp4":
                public_kwargs["global_A_scale"] = torch.tensor(1.0, device=device)
            if outtype == "fp8":
                empty_out, scale = aiter.gemm_a4w4o8(*public_args, **public_kwargs)
                assert scale.shape == (0, 1)
            else:
                empty_out = aiter.gemm_a4w4(*public_args, **public_kwargs)
            assert empty_out.shape == (2, 0, N) and empty_out.dtype == dtype
            checked += 1
    aiter.logger.info(
        "F4GEMM output and empty-batch contracts passed (%d cases)", checked
    )


@benchmark()  # intype, M, N, K, apre, outtype, data_init, ... -> table columns
def test_gemm(
    intype,
    M,
    N,
    K,
    apre,
    outtype="bf16",
    data_init="uniform",
    scale_init="auto",
    seed=0,
    mode="perf",
    knl_name=None,
    num_warmup=2,
    num_iters=None,
    test_graph=False,
    num_rotate=0,
    check_gluon=False,
):
    if check_gluon:
        reason = _gluon_support_reason(intype, outtype, apre, M, N, K)
        if reason is not None:
            raise ValueError(f"--check-gluon: {reason}")

    def not_supported(reason):
        aiter.logger.warning(
            "f4gemm not supported (%s): intype=%s outtype=%s apre=%s M=%s N=%s K=%s",
            reason,
            intype,
            outtype,
            apre,
            M,
            N,
            K,
        )
        result = {
            "gfx": get_gfx(),
            "knl_name": knl_name or "auto",
            "asm us": float("nan"),
            "asm TFLOPS": float("nan"),
            "asm TB/s": float("nan"),
            "asm err": float("nan"),
            "asm result": f"not support ({reason})",
        }
        if check_gluon:
            result["gluon bitmatch"] = "not checked"
        return result

    # Check only the preparation/reference contract here. C++ selects the kernel.
    reason = _support_reason(outtype, apre, M, N, K)
    if reason is not None:
        return not_supported(reason)
    knl = "" if knl_name in (None, "", "auto") else knl_name
    compare_gluon = check_gluon or (
        mode == "func" and _gluon_support_reason(intype, outtype, apre, M, N, K) is None
    )

    block = MXFP4_SCALE_BLOCK if intype == "mxfp4" else NVFP4_SCALE_BLOCK
    assert K % block == 0, f"K must be a multiple of {block}"
    out_fp8 = outtype == "fp8"
    out_dtype = _OUT_DTYPE[outtype]
    gen = make_generator(seed)  # fixed seed -> bit-identical buffers
    if intype == "mxfp4":
        inp, ref_f32 = _prep_mxfp4(
            M, N, K, apre, data_init, scale_init, gen, check_gluon=compare_gluon
        )
    else:
        inp, ref_f32 = _prep_nvfp4(M, N, K, apre, data_init, scale_init, gen)
    # Reference in the kernel's output form: block-scaled (fp8 e4m3 data + e8m0
    # scale) tuple for fp8, else bf16.
    if out_fp8:
        ref = _quant_mxfp8_blockN(ref_f32)  # (ref_fp8, ref_scale_e8m0)
    else:
        ref = ref_f32.to(out_dtype)
    needTrace = mode == "profile"
    # --iters overrides; unset keeps the mode default (func=5, perf/profile=101).
    num_iters = num_iters if num_iters is not None else (5 if mode == "func" else 101)

    # Pass inputs as ARGS so run_perftest can rotate them (defeats the L2 hot-cache).
    y = torch.empty((M, N), dtype=out_dtype, device=inp["A"].device)
    if intype == "nvfp4":

        def run_asm(A, B, sA, sB, gA, gB, y):
            return aiter.gemm_nvfp4_asm(
                A,
                B,
                sA,
                sB,
                gA,
                gB,
                dtype=out_dtype,
                a_preshuffle=bool(apre),
                kernelName=knl,
                y=y,
            )

        asm_args = (inp["A"], inp["B"], inp["sA"], inp["sB"], inp["gA"], inp["gB"], y)
    else:

        def run_asm(A, B, sA, sB, y):
            return aiter.gemm_mxfp4_asm(
                A,
                B,
                sA,
                sB,
                dtype=out_dtype,
                a_preshuffle=bool(apre),
                kernelName=knl,
                y=y,
            )

        asm_args = (inp["A"], inp["B"], inp["sA"], inp["sB"], y)

    # Only the low-level asm entry is timed/tabled. (fn, args); args are rotated.
    candidates = {"asm": (run_asm, asm_args)}

    flops = 2 * M * N * K
    # Output bytes: fp8 = M*N (fp8) + M*N/128 (e8m0 scale); bf16 = M*N*itemsize.
    if out_fp8:
        out_bytes = M * N + M * (N // MXFP8_OUT_SCALE_BLOCK)
    else:
        out_bytes = M * N * out_dtype.itemsize
    # Scale bytes use the LOGICAL (unpadded) size: shuffle_scale_f4 pads scale rows
    # to fill the preshuffle tile, but the shader clamps its scale dim and never
    # reads the padding, so the padded buffer's .nbytes would inflate bandwidth.
    # e8m0 (MXFP4) and e4m3 (NVFP4) are both 1 byte/elem; `block` is set above.
    scale_bytes = (M + N) * (K // block)
    nbytes = inp["A"].nbytes + inp["B"].nbytes + scale_bytes + out_bytes

    ret = {"gfx": get_gfx(), "knl_name": knl or "auto"}
    if compare_gluon:
        # Missing ASM kernels must not silently pass a requested comparison.
        ret["gluon bitmatch"] = "not checked"
    # Only a missing .co is reported as "not support"; any other failure (OOM,
    # memory fault, shape assert, ...) must propagate, not show as a green cell.
    for name, (fn, fn_args) in candidates.items():
        try:
            out, us = run_perftest(
                fn,
                *fn_args,
                num_iters=num_iters,
                num_warmup=num_warmup,
                testGraph=test_graph,
                num_rotate_args=num_rotate,
                needTrace=needTrace,
            )
        except Exception as e:
            if not any(m in str(e) for m in _NOT_SUPPORTED_MARKERS):
                raise
            # No .co for this config; mark unsupported, keep going.
            aiter.logger.warning(
                "f4gemm not supported: intype=%s outtype=%s apre=%s "
                "M=%s N=%s K=%s [%s]: %s",
                intype,
                outtype,
                apre,
                M,
                N,
                K,
                knl or "auto",
                e,
            )
            ret[f"{name} us"] = float("nan")
            ret[f"{name} TFLOPS"] = float("nan")
            ret[f"{name} TB/s"] = float("nan")
            ret[f"{name} err"] = float("nan")
            ret[f"{name} result"] = "not support"
            continue
        ret["knl_name"] = _observed_kernel(fn, fn_args)
        # Func-mode only: check the high-level op contracts by outtype -- bf16 ->
        # gemm_a4w4 (single tensor), fp8 -> gemm_a4w4o8 ((data, scale) tuple).
        # Not timed/tabled.
        if mode == "func":
            if "f4gemm_bf16_mxfp4_BpreShuffle_192x256" in ret["knl_name"]:
                persistent = aiter.gemm_mxfp4_asm(
                    inp["A"],
                    inp["B"],
                    inp["sA"],
                    inp["sB"],
                    a_preshuffle=False,
                    kernelName=_PS_KERNEL,
                )
                assert torch.equal(
                    out.view(torch.uint8), persistent.view(torch.uint8)
                ), f"NPS/PS bit mismatch for M,N,K={M},{N},{K}"
            a4_kwargs = {"apreshuffle": bool(apre)}
            if intype == "nvfp4":
                # global scales are Optional[Tensor] (schema); a non-None value
                # selects the NVFP4 path -- wrap the float scalars as tensors.
                a4_kwargs.update(
                    global_A_scale=torch.tensor(inp["gA"], device=inp["A"].device),
                    global_B_scale=torch.tensor(inp["gB"], device=inp["A"].device),
                )
            args = (inp["A"], inp["B"], inp["sA"], inp["sB"])
            if out_fp8:
                o, s = aiter.gemm_a4w4o8(*args, **a4_kwargs)
                assert o.shape == out[0].shape and s.shape == out[1].shape, (
                    f"gemm_a4w4o8 shape mismatch: {tuple(o.shape)}/{tuple(s.shape)} "
                    f"vs {tuple(out[0].shape)}/{tuple(out[1].shape)}"
                )
            else:  # bf16
                res = aiter.gemm_a4w4(*args, dtype=out_dtype, **a4_kwargs)
                assert not isinstance(res, tuple), "gemm_a4w4 must return a tensor"
                assert (
                    res.shape == out.shape
                ), f"gemm_a4w4 shape mismatch: {tuple(res.shape)} vs {tuple(out.shape)}"
        if out_fp8:
            # (fp8 data, packed e8m0). Unpack scale to row-major; e8m0 compared
            # with a +-1 step tolerance, data dequant with tolerance.
            ref_fp8, ref_scale = ref
            o_fp8, o_scale = out  # o_* avoids shadowing the out_fp8 flag
            M_out, N_out = o_fp8.shape
            out_scale_rm = unpack_mxfp8_out_scale(o_scale, M_out, N_out)
            err_s = checkAllclose(
                ref_scale.view(torch.uint8).float(),
                out_scale_rm.view(torch.uint8).float(),
                rtol=0,
                # e8m0 out-scale rounds with a single guard bit (no RNE); allow a
                # +-1 e8m0-step slack for rounding-mode drift across shapes/kernels.
                atol=1,
                msg=f"{intype} {name} fp8 e8m0 (+-1)",
            )
            err_d = checkAllclose(
                _dequant_mxfp8_blockN(ref_fp8, ref_scale),
                _dequant_mxfp8_blockN(o_fp8, out_scale_rm),
                rtol=1e-1,
                atol=1.0,
                msg=f"{intype} {name} fp8",
            )
            err = max(err_s, err_d)
        else:
            # Compare in fp32: checkAllclose does no dtype promotion, so a bf16
            # comparison evaluates atol + rtol*|b| at 8-bit mantissa too. That
            # makes the threshold itself jitter ~0.2% and systematically
            # under-reports borderline elements.
            err = checkAllclose(
                ref.to(dtypes.fp32),
                out.to(dtypes.fp32),
                rtol=1e-1,
                atol=1.0,
                msg=f"{intype} {name}",
            )
        ret[f"{name} us"] = round(us, 2)
        ret[f"{name} TFLOPS"] = round(flops / us / 1e6, 1)
        ret[f"{name} TB/s"] = round(nbytes / us / 1e6, 2)
        ret[f"{name} err"] = err
        ret[f"{name} result"] = _verdict(err)
        if compare_gluon:
            # Gluon compilation, shuffling, execution and comparison are outside
            # ASM's timed region. This uses the same entry/config as the Gluon
            # benchmark, with the UT's --data-init/--scale-init/--seed buffers.
            gluon_out = run_gluon_mxfp4(*inp["gluon_args"])
            ret.update(_compare_gluon_output(out, gluon_out))
        if needTrace:
            ret[f"{name} trace"] = f"./aiter_logs/gpu_id_{torch.cuda.current_device()}"
    return ret


def main():
    torch.set_default_device("cuda")
    torch.set_printoptions(sci_mode=False)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.width", 1000)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test/benchmark gfx1250 A4W4 (F4GEMM) via the low-level asm entry",
    )
    parser.add_argument(
        "--mode",
        choices=["func", "perf", "profile"],
        default="perf",
        help="func=API checks, accuracy, timing and supported MXFP4/Gluon bit-match; "
        "perf=accuracy+timing; profile=perf+trace",
    )
    parser.add_argument(
        "--check-gluon",
        action="store_true",
        help="compare ASM output bit-for-bit with the gfx1250 Gluon kernel on "
        "the SAME data/scales (outside timing). Requires --intype mxfp4 "
        "--outtype bf16 and K%%256==0; supports --apre 0 and 1. "
        "Any mismatch or unchecked case exits nonzero AFTER the summary/JSON.",
    )
    parser.add_argument(
        "--intype",
        nargs="*",
        choices=["mxfp4", "nvfp4"],
        default=["mxfp4", "nvfp4"],
        help="fp4 input format(s) to sweep, e.g. --intype nvfp4",
    )
    parser.add_argument(
        "--apre",
        type=int,
        nargs="*",
        choices=[1, 0],
        default=None,
        help="A-preshuffle sweep list: 1 preshuffles A (M%%16), 0 sends it "
        "row-major (M%%1). Default: [1, 0], covering persistent and NPS dispatch.",
    )
    parser.add_argument(
        "--outtype",
        nargs="*",
        choices=sorted(_OUT_DTYPE),
        default=["bf16", "fp8"],
        help="output-format sweep list (default: bf16 fp8):\n"
        "  bf16 = bf16 [M,N]\n"
        "  fp8  = fp8 e4m3 [M,N] + per-128 E8M0 scale (mxfp8)",
    )
    parser.add_argument(
        "--data-init",
        dest="data_init",
        nargs="+",
        choices=["zero", "constant", "uniform", "norm"],
        default=None,
        help="DATA init distribution(s) (sampled independently of scale).\n"
        "Paired position-wise with --scale-init (length-1 broadcasts).\n"
        "Default (unset): perf/profile = 'constant uniform', func = 'uniform'\n"
        "(func drops constant: its exact-boundary values trigger e8m0/e4m3\n"
        "edge rounding that shows as spurious warnings).\n"
        "  zero     = all-zero e2m1 codes\n"
        "  constant = A=0x22, B=0x33 (deterministic)\n"
        "  uniform  = FP4 U(-3,3)\n"
        "  norm     = N(0,1)                 [norm-dist / LLM-like]",
    )
    parser.add_argument(
        "--scale-init",
        dest="scale_init",
        nargs="+",
        choices=["auto", "pow2_binomial", "zero", "constant", "uniform", "norm"],
        default=None,
        help="SCALE init distribution(s) (by scale format)\n"
        "Default (unset): perf/profile = 'constant auto', func = 'auto'\n"
        "  auto          = format-recommended: mxfp4/E8M0 -> pow2_binomial,\n"
        "                  nvfp4/E4M3 -> gaussian(0.34375,0.08)\n"
        "  pow2_binomial = 2^(Binomial(21,0.5)-11)   [E8M0 only]\n"
        "  zero          = all-zero scale bytes\n"
        "  constant      = neutral scale (2^0 = 1.0)\n"
        "  uniform       = U(0.5,2) -> nearest on-wire byte\n"
        "  norm          = N(1,0.25) -> nearest on-wire byte",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed; same seed -> bit-identical data/scale buffers",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="warmup iterations before timing (run_perftest num_warmup)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="timed iterations (run_perftest num_iters); unset -> mode default "
        "(func=5, perf/profile=101)",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help="also time via HIP graph replay (run_perftest testGraph), "
        "minimizing inter-kernel gaps",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        default=0,
        help="rotating input-buffer copies to defeat the L2 hot-cache "
        "(run_perftest num_rotate_args); 0 = auto-size from L2",
    )
    parser.add_argument(
        "--json",
        dest="json_out",
        default=None,
        help="also write the full summary table (all columns) as JSON records "
        "to this path, for CI/regression",
    )
    parser.add_argument(
        "--knl-name",
        dest="knl_name",
        default=None,
        help="dispatch mode. Default (unset) = heuristic: the aiter op picks the "
        ".co from f4gemm.csv by shape and input/output formats, validating dispatch. "
        "'auto' also uses the heuristic. Any other value "
        "= use that exact mangled knl_name for all runs (developer experiment/debug).",
    )
    parser.add_argument(
        "-s",
        "-mnk",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        # Unset -> per-mode defaults: perf=PERF_SHAPES (NPS and PS throughput),
        # func=FUNC_SHAPES (many small/odd shapes, correctness). K must be %32.
        default=None,
        help="(M,N,K) tuples, e.g. -mnk 2048,2048,2048 16384,16384,16384; "
        "unset uses PERF_SHAPES (perf/profile) or FUNC_SHAPES (func)",
    )
    args = parser.parse_args()

    if args.iters is not None and args.iters < 2:
        parser.error("--iters must be at least 2 for profiler-based timing")

    # @benchmark always returns the call-args dict, so gate before emitting rows.
    if get_gfx() not in SUPPORTED_GFX:
        if args.check_gluon:
            parser.error("--check-gluon requires gfx1250")
        aiter.logger.warning(
            "gemm_a4w4 (F4GEMM) unsupported on %s; skipping", get_gfx()
        )
        return
    if not torch.cuda.is_available() or not is_gfx1250_asm_supported():
        if args.check_gluon:
            parser.error("--check-gluon requires an available gfx1250 B0+ GPU")
        aiter.logger.warning("F4GEMM requires an available gfx1250 B0+ GPU; skipping")
        return

    # DATA and SCALE init are paired position-wise (NOT crossed). Mode-aware
    # defaults when unset: perf/profile run constant+constant and uniform+auto;
    # func drops the constant pair (its exact-boundary values trigger e8m0/e4m3
    # edge rounding -> spurious warnings) and runs just uniform+auto. A length-1
    # list broadcasts against the other axis.
    if args.mode == "func":
        default_di, default_si = ["uniform"], ["auto"]
    else:
        default_di, default_si = ["constant", "uniform"], ["constant", "auto"]
    di_list = args.data_init if args.data_init is not None else default_di
    si_list = args.scale_init if args.scale_init is not None else default_si
    if len(di_list) == 1:
        di_list = di_list * len(si_list)
    if len(si_list) == 1:
        si_list = si_list * len(di_list)
    if len(di_list) != len(si_list):
        parser.error(
            "--data-init and --scale-init must have equal length "
            "(or length 1 to broadcast)"
        )
    init_pairs = list(zip(di_list, si_list))

    # Shapes: explicit --shape wins; otherwise func sweeps the many small/odd
    # correctness shapes and perf/profile cover aligned NPS shapes and PS.
    if args.shape is not None:
        shapes = args.shape
    else:
        shapes = FUNC_SHAPES if args.mode == "func" else PERF_SHAPES

    # Both layouts are needed to exercise the NPS and persistent choices.
    apre_list = args.apre if args.apre is not None else [1, 0]

    if args.check_gluon:
        if not (shapes and apre_list and args.intype and args.outtype):
            parser.error("--check-gluon requires at least one case")
        for intype, outtype, apre, (M, N, K) in itertools.product(
            args.intype, args.outtype, apre_list, shapes
        ):
            reason = _gluon_support_reason(intype, outtype, apre, M, N, K)
            if reason is not None:
                parser.error(f"--check-gluon: {reason} (M,N,K={M},{N},{K})")

    if args.mode == "func":
        _check_output_contracts(args.intype, args.outtype, apre_list, args.seed)

    rows = [
        test_gemm(
            intype,
            M,
            N,
            K,
            apre,
            outtype,
            di,
            si,
            seed=args.seed,
            mode=args.mode,
            knl_name=args.knl_name,
            num_warmup=args.warmup,
            num_iters=args.iters,
            test_graph=args.graph,
            num_rotate=args.rotate,
            check_gluon=args.check_gluon,
        )
        for apre, (di, si), intype, outtype, (M, N, K) in itertools.product(
            apre_list, init_pairs, args.intype, args.outtype, shapes
        )
    ]
    df_full = pd.DataFrame(rows)
    # JSON keeps every column (config + algo details + results) so each record is
    # self-describing for CI/regression; the markdown table below drops columns
    # that are constant within a run for readability.
    if args.json_out:
        df_full.to_json(args.json_out, orient="records", indent=2)
        aiter.logger.info(
            "wrote JSON summary (%d rows) to %s", len(df_full), args.json_out
        )
    # Keep the observed kernel name; no Python copy of dispatch/geometry rules.
    df = df_full.drop(
        columns=[
            "seed",
            "gfx",
            "mode",
            "num_warmup",
            "num_iters",
            "test_graph",
            "num_rotate",
            "check_gluon",
        ],
        errors="ignore",
    )
    aiter.logger.info(
        "gemm_a4w4 (F4GEMM) summary (markdown):\n%s",
        df.to_markdown(index=False),
    )
    if args.mode == "profile":
        aiter.logger.info("profiler traces written under ./aiter_logs/")
    failed = sum(
        row.get("asm result") == "failed"
        or row.get("gluon bitmatch") == "failed"
        or (args.check_gluon and row.get("gluon bitmatch") != "pass")
        for row in rows
    )
    if failed:
        aiter.logger.error("F4GEMM validation failed for %d case(s)", failed)
        raise SystemExit(1)


if __name__ == "__main__":
    main()

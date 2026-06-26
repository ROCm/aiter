# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness test for the FlyDSL mxfp8 a8w8 bpreshuffle GEMM (gfx950).

mxfp8 = fp8 (E4M3) activations AND fp8 weights, both quantized per-1x32 with
e8m0 microscales applied inside the scaled MFMA (mfma_scale_f32_16x16x128_f8f6f4).

Run:  PYTHONPATH=/data/FlyDSL/build-fly/python_packages python op_tests/test_gemm_mxfp8_bpreshuffle.py
"""
import sys

import torch

import aiter
from aiter import dtypes
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils
from aiter.test_common import checkAllclose


def _ref_mxfp8(x_fp8, w_fp8, x_scale_e8m0, w_scale_e8m0, out_dtype):
    """fp32 reference: dequant A and W with e8m0 scales, then matmul."""
    block = 32
    M, K = x_fp8.shape
    N = w_fp8.shape[0]
    x_sf = fp4_utils.e8m0_to_f32(x_scale_e8m0).float()  # [M, K//32]
    w_sf = fp4_utils.e8m0_to_f32(w_scale_e8m0).float()  # [N, K//32]
    a = x_fp8.float().view(M, K // block, block) * x_sf.view(M, K // block, 1)
    b = w_fp8.float().view(N, K // block, block) * w_sf.view(N, K // block, 1)
    a = a.view(M, K)
    b = b.view(N, K)
    return (a @ b.t()).to(out_dtype)


def test_mxfp8_bpreshuffle(
    m, n, k, tile_m, tile_n, tile_k, out_dtype=torch.bfloat16, split_k=1
):
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_mxfp8

    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")

    x_fp8, x_scale = per_1x32_f8_scale_f8_quant(
        x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w_fp8, w_scale = per_1x32_f8_scale_f8_quant(
        w, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )

    ref = _ref_mxfp8(x_fp8, w_fp8, x_scale, w_scale, out_dtype)

    w_shuffled = shuffle_weight(w_fp8, layout=(16, 16))
    x_scale_shuf = fp4_utils.e8m0_shuffle(x_scale)
    w_scale_shuf = fp4_utils.e8m0_shuffle(w_scale)

    out = torch.empty((m, n), dtype=out_dtype, device="cuda")
    flydsl_preshuffle_gemm_mxfp8(
        x_fp8,
        w_shuffled,
        x_scale_shuf,
        w_scale_shuf,
        out,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        split_k=split_k,
    )

    err = checkAllclose(
        ref,
        out,
        msg=f"mxfp8 bpreshuffle m={m} n={n} k={k} split_k={split_k}: ",
        rtol=1e-2,
        atol=1e-2,
    )
    return err


def test_mxfp8_public_op(
    m, n, k, out_dtype=torch.bfloat16, split_k=1, tile=(0, 0, 0)
):
    """Same correctness check, but through the public API
    ``aiter.gemm_a8w8_bpreshuffle_mxfp8`` instead of the FlyDSL dispatcher.

    Verifies the public-op wiring: pre-shuffled inputs (Option A convention),
    auto tile selection (tile=(0,0,0)) and the splitK passthrough.
    """
    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")

    x_fp8, x_scale = per_1x32_f8_scale_f8_quant(
        x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w_fp8, w_scale = per_1x32_f8_scale_f8_quant(
        w, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )

    ref = _ref_mxfp8(x_fp8, w_fp8, x_scale, w_scale, out_dtype)

    w_shuffled = shuffle_weight(w_fp8, layout=(16, 16))
    x_scale_shuf = fp4_utils.e8m0_shuffle(x_scale)
    w_scale_shuf = fp4_utils.e8m0_shuffle(w_scale)

    tm, tn, tk = tile
    out = aiter.gemm_a8w8_bpreshuffle_mxfp8(
        x_fp8,
        w_shuffled,
        x_scale_shuf,
        w_scale_shuf,
        dtype=out_dtype,
        splitK=split_k,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
    )

    return checkAllclose(
        ref,
        out,
        msg=f"mxfp8 public-op m={m} n={n} k={k} split_k={split_k}: ",
        rtol=1e-2,
        atol=1e-2,
    )


def test_mxfp8_splitk_correctness(m, n, k, tile_m, tile_n, tile_k, split_k):
    """Verify split_k > 1 matches split_k == 1 (within fp tolerance).

    Mirrors ck_gemm_a8w8_blockscale's test_splitk_correctness: both sides are
    the same mxfp8 kernel, so this isolates the split-K bf16 atomic reduction
    error from quantization error. Tolerance is aligned with the blockscale
    operator's split-K test: rtol=atol=1e-2 (checkAllclose default 5% err ratio).
    """
    from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_mxfp8

    torch.manual_seed(0)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    x_fp8, x_scale = per_1x32_f8_scale_f8_quant(
        x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w_fp8, w_scale = per_1x32_f8_scale_f8_quant(
        w, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    w_shuffled = shuffle_weight(w_fp8, layout=(16, 16))
    x_scale_shuf = fp4_utils.e8m0_shuffle(x_scale)
    w_scale_shuf = fp4_utils.e8m0_shuffle(w_scale)

    base = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    split = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    flydsl_preshuffle_gemm_mxfp8(
        x_fp8, w_shuffled, x_scale_shuf, w_scale_shuf, base,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, split_k=1,
    )
    flydsl_preshuffle_gemm_mxfp8(
        x_fp8, w_shuffled, x_scale_shuf, w_scale_shuf, split,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k, split_k=split_k,
    )
    return checkAllclose(
        base,
        split,
        msg=f"mxfp8 split_k={split_k} vs 1  m={m} n={n} k={k}: ",
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
    arch = aiter.get_gfx() if hasattr(aiter, "get_gfx") else None
    try:
        from aiter.utility.base_tuner import get_gfx as _ggfx  # noqa

        arch = _ggfx()
    except Exception:
        pass
    if arch is not None and "gfx950" not in str(arch):
        print(f"SKIP: mxfp8 MFMA GEMM requires gfx950, got {arch}")
        sys.exit(0)

    # ── single-pass correctness vs torch fp32 reference (rtol=atol=1e-2) ──
    shapes = [
        # (m, n, k, tile_m, tile_n, tile_k, out_dtype)
        (256, 256, 512, 128, 128, 256, torch.bfloat16),
        (256, 256, 1024, 128, 128, 256, torch.bfloat16),  # multiple K-tiles
        (256, 256, 1024, 128, 128, 512, torch.bfloat16),  # tile_k=512
        (256, 256, 512, 128, 128, 256, torch.float16),    # fp16 out
        (512, 512, 1024, 256, 128, 256, torch.bfloat16),  # bigger tile_m
        (128, 256, 768, 64, 128, 256, torch.bfloat16),    # tile_m=64, K=768
        (384, 512, 512, 128, 256, 256, torch.bfloat16),   # M not mult of 256
        # ── tile_k=128: K need only be a multiple of 128 (not 256). The two
        #    128-K tiles of each 256-K scale group share one e8m0 scale i32
        #    (opsel picks the half); K=384 exercises the odd trailing tile. ──
        (256, 256, 384, 128, 128, 128, torch.bfloat16),   # K=384 (3 tiles, odd)
        (64, 256, 384, 64, 128, 128, torch.float16),      # K=384 fp16 out
        (256, 256, 512, 128, 128, 128, torch.bfloat16),   # tile_k=128, K mult 256
    ]
    # ── split-K self-consistency vs split_k=1 (aligned with blockscale's
    #    test_splitk_correctness: rtol=atol=1e-2) ──
    splitk_shapes = [
        # (m, n, k, tile_m, tile_n, tile_k, split_k)
        (256, 256, 512, 128, 128, 256, 2),
        (256, 256, 1024, 128, 128, 256, 2),
        (256, 256, 2048, 128, 128, 512, 2),  # tile_k=512
        (512, 512, 1024, 256, 128, 256, 2),  # bigger tile_m
    ]
    fails = 0
    for (m, n, k, tm, tn, tk, od) in shapes:
        try:
            test_mxfp8_bpreshuffle(m, n, k, tm, tn, tk, out_dtype=od, split_k=1)
        except Exception as e:
            print(f"FAIL m={m} n={n} k={k} t={tm}x{tn}x{tk} {od}: {e}")
            fails += 1
    for (m, n, k, tm, tn, tk, spk) in splitk_shapes:
        try:
            test_mxfp8_splitk_correctness(m, n, k, tm, tn, tk, split_k=spk)
        except Exception as e:
            print(f"FAIL splitk m={m} n={n} k={k} t={tm}x{tn}x{tk} spk={spk}: {e}")
            fails += 1
    # ── public-op wiring through aiter.gemm_a8w8_bpreshuffle_mxfp8 (auto tile) ──
    public_shapes = [
        # (m, n, k, out_dtype, split_k)
        (256, 256, 512, torch.bfloat16, 1),
        (256, 256, 1024, torch.float16, 1),
        (384, 512, 512, torch.bfloat16, 1),  # M not mult of 256, auto tile
        (256, 256, 1024, torch.bfloat16, 2),  # splitK passthrough
        (256, 256, 384, torch.bfloat16, 1),  # K=384, auto tile -> tile_k=128
        (32, 7168, 384, torch.bfloat16, 1),  # dsv4 K=384 shape, auto tile
    ]
    for (m, n, k, od, spk) in public_shapes:
        try:
            test_mxfp8_public_op(m, n, k, out_dtype=od, split_k=spk)
        except Exception as e:
            print(f"FAIL public-op m={m} n={n} k={k} {od} spk={spk}: {e}")
            fails += 1
    print("ALL PASSED" if fails == 0 else f"{fails} FAILED")

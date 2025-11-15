# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, perftest, run_perftest
from aiter import dtypes
from aiter.ops.triton.fused_mxfp4_quant import (
    fused_rms_mxfp4_quant,
)
from op_tests.triton_tests.test_quant_mxfp4 import torch_dynamic_mxfp4_quant
from op_tests.triton_tests.test_gemm_afp4wfp4 import (
    mxfp4_to_f32,
    e8m0_to_f32,
    SCALE_GROUP_SIZE,
)
import argparse
import pandas as pd
from quark.torch.algorithm.rotation.rotation_utils import get_rotation_matrix


torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", 30)


def rmsnorm(input, weight, eps=1e-6):
    """Reference RMSNorm implementation"""
    row_norm = input * input
    row_norm = torch.sum(row_norm, dim=-1)
    norm_factor = torch.rsqrt((row_norm / input.shape[1]) + eps).reshape(-1, 1)
    rms_norm = input * norm_factor * weight.reshape(1, -1)
    return rms_norm


def rotate(input, rot):
    """Reference rotation implementation"""
    mat_shape = input.shape
    return torch.matmul(input.view(-1, rot.shape[0]), rot).view(mat_shape)


@perftest(num_iters=100)
def run_unfused_single(mat1, rms1_w, eps, rot1=None):
    """Unfused version: RMS + optional rotation + quantization"""
    mat1_normalized = rmsnorm(mat1, rms1_w, eps)
    if rot1 is not None:
        mat1_normalized = rotate(mat1_normalized, rot1)
    q_fp4, q_scales = torch_dynamic_mxfp4_quant(mat1_normalized)
    return q_fp4, q_scales


@perftest(num_iters=100)
def run_unfused_single_with_residual(mat1, rms1_w, eps, resid1, rot1=None):
    """Unfused version with residual: add + RMS + optional rotation + quantization"""
    mat1_with_residual = mat1 + resid1
    mat1_normalized = rmsnorm(mat1_with_residual, rms1_w, eps)
    if rot1 is not None:
        mat1_normalized = rotate(mat1_normalized, rot1)
    q_fp4, q_scales = torch_dynamic_mxfp4_quant(mat1_normalized)
    return q_fp4, q_scales, mat1_with_residual


@perftest(num_iters=100)
def run_unfused_dual(mat1, rms1_w, eps1, mat2, rms2_w, eps2, resid1=None, rot1=None):
    """Unfused version: optional add + RMS + optional rotation + quantization for mat1, RMS for mat2"""
    if resid1 is not None:
        mat1 = mat1 + resid1
        out_res1 = mat1.clone()
    else:
        out_res1 = None
    
    mat1_normalized = rmsnorm(mat1, rms1_w, eps1)
    if rot1 is not None:
        mat1_normalized = rotate(mat1_normalized, rot1)
    q_fp4, q_scales = torch_dynamic_mxfp4_quant(mat1_normalized)
    
    mat2_normalized = rmsnorm(mat2, rms2_w, eps2)
    
    if out_res1 is not None:
        return (q_fp4, q_scales), mat2_normalized, out_res1
    else:
        return (q_fp4, q_scales), mat2_normalized


@perftest(num_iters=100)
def run_fused_single(mat1, rms1_w, eps, rot1=None):
    """Fused version: RMS + optional rotation + quantization"""
    return fused_rms_mxfp4_quant(
        mat1, rms1_w, eps, None, None, None, None, rot1
    )


@perftest(num_iters=100)
def run_fused_single_with_residual(mat1, rms1_w, eps, resid1, rot1=None):
    """Fused version with residual: add + RMS + optional rotation + quantization"""
    return fused_rms_mxfp4_quant(
        mat1, rms1_w, eps, None, None, None, resid1, rot1
    )


@perftest(num_iters=100)
def run_fused_dual(mat1, rms1_w, eps1, mat2, rms2_w, eps2, resid1=None, rot1=None):
    """Fused version: optional add + RMS + optional rotation + quantization for mat1, RMS for mat2"""
    return fused_rms_mxfp4_quant(
        mat1, rms1_w, eps1, mat2, rms2_w, eps2, resid1, rot1
    )


def convert_mxfp4_to_fp32(x, x_scales):
    """Convert mxfp4 quantized data back to fp32 for comparison"""
    x_f32 = mxfp4_to_f32(x)
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)[:, : x_f32.shape[1]]
    x_f32 = x_f32 * x_scales_f32
    return x_f32


@benchmark()
def bench_fused_rms_quant_single(dtype, M, N, rotation_size=0, with_residual=False):
    """Benchmark single matrix: RMS + optional rotation + quantization"""
    ret = {}
    ret["M"] = M
    ret["N"] = N
    ret["dtype"] = str(dtype)

    # Generate inputs
    mat1 = torch.randn((M, N), dtype=dtype, device="cuda")
    rms1_w = torch.randn(N, dtype=dtype, device="cuda")
    eps = 1e-6

    rot1 = None
    if rotation_size > 0:
        if N % rotation_size != 0:
            aiter.logger.warning(f"Skipping rotation for N={N} (not divisible by {rotation_size})")
            return None
        rot1 = get_rotation_matrix(rotation_size, random=False).to(dtype=dtype, device="cuda")
    resid1 = None
    if with_residual:
        resid1 = torch.randn((M, N), dtype=dtype, device="cuda")

    # Run unfused version
    # if with_residual:
    #     unfused_out, us_unfused = run_unfused_single_with_residual(
    #         mat1.clone(), rms1_w, eps, resid1, rot1
    #     )
    #     q_fp4_unfused, q_scales_unfused, _ = unfused_out
    # else:
    #     q_fp4_unfused, q_scales_unfused, us_unfused = run_unfused_single(
    #         mat1.clone(), rms1_w, eps, rot1
    #     )

    # Run fused version
    if with_residual:
        fused_out, us_fused = run_fused_single_with_residual(
            mat1.clone(), rms1_w, eps, resid1, rot1
        )
        q_fp4_fused, q_scales_fused = fused_out[0]
    else:
        fused_out, us_fused = run_fused_single(
            mat1.clone(), rms1_w, eps, rot1
        )
        q_fp4_fused, q_scales_fused = fused_out

    # Convert back to fp32 for comparison
    # unfused_fp32 = convert_mxfp4_to_fp32(q_fp4_unfused, q_scales_unfused)
    # fused_fp32 = convert_mxfp4_to_fp32(q_fp4_fused, q_scales_fused)

    # Check correctness
    # err = checkAllclose(unfused_fp32, fused_fp32, msg="fused vs unfused")

    # Calculate performance metrics
    # Approximate memory access: read mat1, rms1_w, write quantized output
    mem_bytes = mat1.nbytes + rms1_w.nbytes + q_fp4_fused.nbytes + q_scales_fused.nbytes
    if with_residual:
        mem_bytes += resid1.nbytes

    # ret["us_unfused"] = us_unfused
    ret["us_fused"] = us_fused
    # ret["speedup"] = us_unfused / us_fused if us_fused > 0 else float('nan')
    # ret["GB/s_unfused"] = mem_bytes / us_unfused / 1e3
    ret["GB/s_fused"] = mem_bytes / us_fused / 1e3
    # ret["err"] = err

    return ret


# ============================================================================
# Benchmark configurations
# ============================================================================

l_dtype = ["bf16"]  # "fp16"
_l_m = [1, 64, 16384, 16384 * 4]
# Single matrix shapes (typical attention dimension scenarios)
l_single_shapes = [
    # (M, N)
    # dsr1
    # qkv_a_proj
    *((i, 7168) for i in _l_m),
    # q_b_proj
    *((i, 1536) for i in _l_m),
    # out of kv_a_nope
    *((i, 512) for i in _l_m),
]

# l_single_shapes = [l_single_shapes[-1]]  # For dump triton codegen kernel
# Dual matrix shapes (typical DeepSeek scenarios)
l_dual_shapes = [
    # (M, N1, N2) - mat1 gets quantized, mat2 gets RMSNorm
    # DeepSeek R1 typical shapes
    (1, 7168, 2048),
    (1, 1536, 512),
    (1, 2112, 512),
    (32, 7168, 2048),
    (32, 1536, 512),
    (32, 2112, 512),
    (64, 7168, 2048),
    (64, 1536, 512),
    (64, 2112, 512),
    (128, 7168, 2048),
    (128, 1536, 512),
    (128, 2112, 512),
    (256, 7168, 2048),
    (256, 1536, 512),
    (512, 7168, 2048),
    (1024, 7168, 2048),
]


# l_rotation_sizes = [0, 32, 64, 128]
# l_single_shapes = [
#     (16384 * 4, 512),
# ]
l_rotation_sizes = [0, 32]
# l_rotation_sizes = [32]

# Parse command line arguments
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Benchmark fused_rms_mxfp4_quant kernel",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "--single",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Single matrix shape (M,N).
    e.g. --single 128,4096""",
)
parser.add_argument(
    "--dual",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Dual matrix shape (M,N1,N2).
    e.g. --dual 128,7168,2048""",
)

args = parser.parse_args()

# Setup dtype list
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]

# Override shapes if provided
if args.single is not None:
    l_single_shapes = [args.single]
if args.dual is not None:
    l_dual_shapes = [args.dual]


# ============================================================================
# Run benchmarks
# ============================================================================


aiter.logger.info("=" * 80)
aiter.logger.info("BENCHMARKING SINGLE MATRIX (RMS + QUANT)")
aiter.logger.info("=" * 80)

df_single_rotation = []
df_single_residual_rotation = []

for dtype in l_dtype:
    for M, N in l_single_shapes:
        for rotation_size in l_rotation_sizes:
            aiter.logger.info(f"Running single: dtype={dtype}, M={M}, N={N}, rotation_size={rotation_size}")

            # With rotation: RMS + Rotate + Quant
            if N != 7168:
                ret = bench_fused_rms_quant_single(dtype, M, N, rotation_size=rotation_size, with_residual=False)
                if ret:
                    df_single_rotation.append(ret)

            # With both: Add + RMS + Rotate + Quant
            if N == 7168:
                ret = bench_fused_rms_quant_single(dtype, M, N, rotation_size=rotation_size, with_residual=True)
                if ret:
                    df_single_residual_rotation.append(ret)

# Print results
if df_single_rotation:
    df = pd.DataFrame(df_single_rotation)
    aiter.logger.info(f"SINGLE MATRIX - (RMS + ROTATE + QUANT)\n{df}")

if df_single_residual_rotation:
    df = pd.DataFrame(df_single_residual_rotation)
    aiter.logger.info(f"SINGLE MATRIX - WITH RESIDUAL (ADD + RMS + ROTATE + QUANT)\n{df}")

from typing import Optional
import functools
import json

import torch

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd, pid_grid
import aiter.ops.triton.gluon.triton_version as tv

_LOGGER = AiterTritonLogger()

# Pre-compute version check as constexpr for use in JIT kernels
TRITON_VERSION_GE_3_6_0 = gl.constexpr(tv.TRITON_VERSION_GE_3_6_0)

@gluon.jit
def _mxfp4_quant_op_gluon(
    x,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
):
    """
    Converts given x (in fp32) to mxfp4 format.
    x: [BLOCK_SIZE_M, BLOCK_SIZE_N], fp32

    """
    EXP_BIAS_FP32: gl.constexpr = 127
    EXP_BIAS_FP4: gl.constexpr = 1
    EBITS_F32: gl.constexpr = 8
    EBITS_FP4: gl.constexpr = 2
    MBITS_F32: gl.constexpr = 23
    MBITS_FP4: gl.constexpr = 1

    max_normal: gl.constexpr = 6
    min_normal: gl.constexpr = 1

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    # Calculate scale
    amax = gl.max(gl.abs(x), axis=-1, keep_dims=True)
    amax = amax.to(gl.int32, bitcast=True)
    amax = (amax + 0x200000).to(gl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(gl.float32, bitcast=True)
    scale_e8m0_unbiased = gl.log2(amax).floor() - 2
    scale_e8m0_unbiased = gl.clamp(scale_e8m0_unbiased, min=-127, max=127)

    # blockscale_e8m0
    bs_e8m0 = scale_e8m0_unbiased.to(gl.uint8) + 127  # in fp32, we have 2&(e - 127)

    quant_scale = gl.exp2(-scale_e8m0_unbiased)

    # Compute quantized x
    qx = x * quant_scale

    # Convert quantized fp32 tensor to uint32 before converting to mxfp4 format
    # Note: MXFP4  S:1-bit, E:2-bit, M:1-bit
    #   Zeros: S000 -> +/-0
    #   Denormal Numbers: S001 -> +/- 0.5
    #   Normal Numbers:
    #           S010 -> +/- 1.0
    #           S011 -> +/- 1.5
    #           S100 -> +/- 2.0
    #           S101 -> +/- 3.0
    #           S110 -> +/- 4.0
    #           S111 -> +/- 6.0
    qx = qx.to(gl.uint32, bitcast=True)

    # Extract sign
    s = qx & 0x80000000
    # Set everything to positive, will add sign back at the end
    qx = qx ^ s

    qx_fp32 = qx.to(gl.float32, bitcast=True)
    saturate_mask = qx_fp32 >= max_normal
    denormal_mask = (~saturate_mask) & (qx_fp32 < min_normal)
    normal_mask = ~(saturate_mask | denormal_mask)

    # Denormal numbers
    denorm_exp: gl.constexpr = (
        (EXP_BIAS_FP32 - EXP_BIAS_FP4) + (MBITS_F32 - MBITS_FP4) + 1
    )
    denorm_mask_int: gl.constexpr = denorm_exp << MBITS_F32
    denorm_mask_float: gl.constexpr = gl.cast(denorm_mask_int, gl.float32, bitcast=True)

    denormal_x = qx_fp32 + denorm_mask_float
    denormal_x = denormal_x.to(gl.uint32, bitcast=True)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(gl.uint8)

    # Normal numbers
    normal_x = qx
    # resulting mantissa is odd
    mant_odd = (normal_x >> (MBITS_F32 - MBITS_FP4)) & 1
    # update exponent, rounding bias part 1
    val_to_add = ((EXP_BIAS_FP4 - EXP_BIAS_FP32) << MBITS_F32) + (1 << 21) - 1
    normal_x += val_to_add
    # rounding bias part 2
    normal_x += mant_odd
    # take the bits!
    normal_x = normal_x >> (MBITS_F32 - MBITS_FP4)
    normal_x = normal_x.to(gl.uint8)

    # Merge results
    e2m1_value = gl.full(qx.type.get_block_shapes(), 0x7, dtype=gl.uint8)
    e2m1_value = gl.where(normal_mask, normal_x, e2m1_value)
    e2m1_value = gl.where(denormal_mask, denormal_x, e2m1_value)
    # add sign back
    sign_lp = s >> (MBITS_F32 + EBITS_F32 - MBITS_FP4 - EBITS_FP4)
    sign_lp = sign_lp.to(gl.uint8)
    e2m1_value = e2m1_value | sign_lp
    e2m1_value = gl.reshape(
        e2m1_value, [BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2]
    )
    evens, odds = gl.split(e2m1_value)
    x_fp4 = evens | (odds << 4)
    x_fp4 = x_fp4.reshape(BLOCK_SIZE_M, BLOCK_SIZE_N // 2)

    return x_fp4, bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

@gluon.jit
def _dynamic_mxfp4_quant_kernel_gluon(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    NUM_ITER: gl.constexpr,
    NUM_STAGES: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
    #EVEN_M_N: gl.constexpr,
    SCALING_MODE: gl.constexpr,
):
    pid_m = gl.program_id(0)
    start_n = gl.program_id(1) * NUM_ITER
    # cast strides to int64, in case M*N > max int32
    stride_x_m = gl.cast(stride_x_m_in, gl.int64)
    stride_x_n = gl.cast(stride_x_n_in, gl.int64)
    stride_x_fp4_m = gl.cast(stride_x_fp4_m_in, gl.int64)
    stride_x_fp4_n = gl.cast(stride_x_fp4_n_in, gl.int64)
    stride_bs_m = gl.cast(stride_bs_m_in, gl.int64)
    stride_bs_n = gl.cast(stride_bs_n_in, gl.int64)

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    for pid_n in range(start_n, min(start_n + NUM_ITER, N), num_stages=NUM_STAGES):
        x_offs_m = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
        x_offs_n = pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N)
        x_offs = x_offs_m[:, None] * stride_x_m + x_offs_n[None, :] * stride_x_n

        # if EVEN_M_N:
        #     x = gl.load(x_ptr + x_offs, cache_modifier=".cg").to(gl.float32)
        # else:
        x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
        x = gl.load(x_ptr + x_offs, mask=x_mask, cache_modifier=".cg").to(
        gl.float32
        )

        out_tensor, bs_e8m0 = _mxfp4_quant_op_gluon(
            x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        out_offs_m = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
        out_offs_n = pid_n * BLOCK_SIZE_N // 2 + gl.arange(0, BLOCK_SIZE_N // 2)
        out_offs = (
            out_offs_m[:, None] * stride_x_fp4_m + out_offs_n[None, :] * stride_x_fp4_n
        )

        # if EVEN_M_N:
        #     gl.store(x_fp4_ptr + out_offs, out_tensor)
        # else:
        out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
        gl.store(x_fp4_ptr + out_offs, out_tensor, mask=out_mask)

        bs_offs_m = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
        bs_offs_n = pid_n * NUM_QUANT_BLOCKS + gl.arange(0, NUM_QUANT_BLOCKS)
        bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
        # if EVEN_M_N:
        #     gl.store(bs_ptr + bs_offs, bs_e8m0)
        # else:
        bs_mask = (bs_offs_m < M)[:, None] & (
            bs_offs_n < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
        )[None, :]
        gl.store(
            bs_ptr + bs_offs,
            bs_e8m0,
            mask=bs_mask,
        )

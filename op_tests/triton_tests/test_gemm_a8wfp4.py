# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from enum import Enum
from aiter.ops.triton.gemm_a8wfp4 import gemm_a8wfp4
import aiter.ops.triton.utils.arch_info as arch_info

# Debug
DEBUG = False
ZERO_OUTPUT = True


class INPUT_TYPE(Enum):
    ONES = "ones"  # generate all ones
    RANDOM = "random"  # generate random values
    INCREMENTAL = "incremental"  # generate incremental pattern: row i contains value i


INPUT_TYPE = INPUT_TYPE.RANDOM

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32

# FP4 look up table
MXFP4_TABLE = [
    0.0,  # 0000
    0.5,  # 0001
    1.0,  # 0010
    1.5,  # 0011
    2.0,  # 0100
    3.0,  # 0101
    4.0,  # 0110
    6.0,  # 0111
    -0.0,  # 1000
    -0.5,  # 1001
    -1.0,  # 1010
    -1.5,  # 1011
    -2.0,  # 1100
    -3.0,  # 1101
    -4.0,  # 1110
    -6.0,  # 1111
]


def generate_fp32_tensors(M, N, K, debug_type):
    """Generate fp32 tensors based on debug input type"""
    if debug_type == INPUT_TYPE.ONES:
        x_fp32 = torch.ones((M, K), dtype=torch.float32, device="cuda")
        w_fp32 = torch.ones((N, K), dtype=torch.float32, device="cuda")
    elif debug_type == INPUT_TYPE.RANDOM:
        # default to random
        x_fp32 = torch.randn((M, K), dtype=torch.float32, device="cuda")
        w_fp32 = torch.randn((N, K), dtype=torch.float32, device="cuda")
    elif debug_type == INPUT_TYPE.INCREMENTAL:
        # generate incremental pattern: row i contains value i
        x_fp32 = (
            torch.arange(M, dtype=torch.float32, device="cuda")
            .unsqueeze(1)
            .expand(M, K)
        )
        w_fp32 = (
            torch.arange(N, dtype=torch.float32, device="cuda")
            .unsqueeze(1)
            .expand(N, K)
        )
    else:
        raise ValueError("Unknown Input Type")

    return x_fp32, w_fp32


def generate_random_fp4_inputs(N, K):
    """Generate random fp4 inputs"""
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    return w.T, w_scales.T


def quantize_to_8bit(x_fp32, dtype):
    """Convert fp32 tensor to 8-bit quantized format with scales"""
    max_x = x_fp32.abs().float().amax(dim=1, keepdim=True)
    dtype_max = (
        torch.iinfo(dtype).max if dtype == torch.int8 else torch.finfo(dtype).max
    )
    x_scale = max_x / dtype_max
    x_quantized = x_fp32 / x_scale
    x_quantized = x_quantized.to(dtype)
    return x_quantized, x_scale


def quantize_to_fp4(w_fp32):
    """Convert fp32 tensor to packed fp4 format with e8m0 scales

    Args:
        w_fp32: fp32 tensor [N, K]

    Returns:
        w_packed: packed fp4 tensor [N, K//2]
        w_scales: e8m0 scale factors [N, K//SCALE_GROUP_SIZE]
    """
    N, K = w_fp32.shape

    # scale to fit in fp4 range
    max_w = w_fp32.abs().float().amax(dim=1, keepdim=True)  # [N, 1]
    mxfp4_max = 6.0

    # handle zero rows to avoid NaN
    w_scale = torch.where(
        max_w == 0,
        torch.ones_like(max_w),  # use scale of 1.0 for zero rows
        max_w / mxfp4_max,
    )

    w_scaled = torch.where(
        max_w == 0, torch.zeros_like(w_fp32), w_fp32 / w_scale  # keep zeros as zeros
    )

    if DEBUG:
        print("w_scaled:", w_scaled)

    # find nearest MXFP4 value for each element
    mxfp4_values = torch.tensor(MXFP4_TABLE, device="cuda", dtype=torch.float32)
    diffs = (w_scaled.unsqueeze(-1) - mxfp4_values.view(1, 1, -1)).abs()
    w_fp4_indices = diffs.argmin(dim=-1).to(torch.uint8)

    if DEBUG:
        print("w_fp4_indices:", w_fp4_indices)

    # pack two FP4 values into one uint8
    w_packed = (w_fp4_indices[:, 1::2] << 4) | w_fp4_indices[:, ::2]

    if DEBUG:
        print("w_packed:", w_packed)

    # convert scale factor to e8m0 format
    # for zero rows, use scale that gives 0 when decoded (very small exponent)
    w_scales_e8m0 = torch.where(
        max_w.squeeze(-1) == 0,
        torch.zeros_like(
            max_w.squeeze(-1), dtype=torch.uint8
        ),  # 0 in e8m0 = 2^(-127) ? 0
        (torch.log2(w_scale.squeeze(-1)) + 127)
        .round()
        .clamp(0, 127)
        .to(torch.uint8),  # clamp to 127 to avoid NaN
    )

    # repeat for each scale group: [N,] -> [N, K//SCALE_GROUP_SIZE]
    w_scales_e8m0 = w_scales_e8m0.unsqueeze(-1).repeat(1, K // SCALE_GROUP_SIZE)

    if DEBUG:
        print("w_scales_e8m0:", w_scales_e8m0)

    return w_packed, w_scales_e8m0


def get_x_vals():

    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 9)]
    x_vals += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]
    x_vals += [
        (1, 1280, 8192),
        (32, 1280, 8192),
        (64, 1280, 8192),
        (128, 1280, 8192),
        (192, 1280, 8192),
        (256, 1280, 8192),
        (320, 1280, 8192),
        (512, 1280, 8192),
        (1024, 1280, 8192),
        (2048, 1280, 8192),
        (4096, 1280, 8192),
        (8192, 1280, 8192),
        (16384, 1280, 8192),
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
    ]
    x_vals += [(2 ** (v - 1), 4096 * v, 4096 * v) for v in range(1, 6)]
    # x_vals = [(128, 1024, 4096)]
    x_vals += [(16, 16384, 3328 * 2), (128, 16384, 3328 * 2)]
    return x_vals


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_in_f32 = torch.tensor(MXFP4_TABLE, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x.to(torch.float32) - 127))
    x_f32[x == 128] = float("nan")
    return x_f32


def dequantize_fp8(x_quantized, x_scales, dtype=torch.float32):
    """dequantize fp8/int8 tensor to fp32

    Args:
        x_quantized: quantized tensor in fp8/int8 format [M, K]
        x_scales: scale factors in fp32 [M, 1]
        dtype: output dtype (default: torch.float32)

    Returns:
        dequantized tensor in specified dtype [M, K]
    """
    x_fp32 = x_quantized.to(torch.float32)
    x_fp32 = x_fp32 * x_scales
    return x_fp32.to(dtype)


def dequantize_fp4(w_packed, w_scales, dtype=torch.float32):
    """dequantize packed fp4 tensor to fp32

    Args:
        w_packed: packed fp4 tensor where 2 fp4 values are packed in each uint8 [N, K//2]
        w_scales: scale factors in e8m0 format [N, K//SCALE_GROUP_SIZE]
        dtype: output dtype (default: torch.float32)

    Returns:
        dequantized tensor in specified dtype [N, K]
    """
    # unpack fp4 values: [N, K//2] -> [N, K]
    w_fp32 = mxfp4_to_f32(w_packed)

    # convert e8m0 scales to fp32
    w_scales_fp32 = e8m0_to_f32(w_scales)  # [N, K//SCALE_GROUP_SIZE]

    # apply scales per group
    # w_scales shape: [N, K//SCALE_GROUP_SIZE]
    # w_fp32 shape: [N, K]
    N, K = w_fp32.shape

    # reshape w_fp32 to group by scale: [N, K] -> [N, K//SCALE_GROUP_SIZE, SCALE_GROUP_SIZE]
    w_fp32_grouped = w_fp32.view(N, K // SCALE_GROUP_SIZE, SCALE_GROUP_SIZE)

    # apply scales: w_scales_fp32 has shape [N, K//SCALE_GROUP_SIZE]
    # expand to [N, K//SCALE_GROUP_SIZE, 1] to broadcast correctly
    w_scales_expanded = w_scales_fp32.unsqueeze(-1)

    # apply scales to each group
    w_fp32_scaled = w_fp32_grouped * w_scales_expanded

    # reshape back: [N, K//SCALE_GROUP_SIZE, SCALE_GROUP_SIZE] -> [N, K]
    w_fp32 = w_fp32_scaled.view(N, K)

    return w_fp32.to(dtype)


def run_torch_emulation(x, w, x_scales, w_scales, dtype):
    """run torch emulation using dequantize functions

    Args:
        x: quantized A matrix [M, K]
        w: packed fp4 B matrix [N, K//2]
        x_scales: A scales [M, 1]
        w_scales: B scales [N, K//SCALE_GROUP_SIZE]
        dtype: output dtype

    Returns:
        matmul result [M, N]
    """
    # dequantize int8/fp8 A to fp32: [M, K]
    x_f32 = dequantize_fp8(x, x_scales, dtype=torch.float32)

    # dequantize fp4 B to fp32: [N, K]
    w_f32 = dequantize_fp4(w, w_scales, dtype=torch.float32)

    # compute matmul: [M, K] @ [K, N] = [M, N]
    return torch.mm(x_f32, w_f32.T).to(dtype)


e5m2_type, e4m3_type = arch_info.get_fp8_dtypes()


@pytest.mark.parametrize("M, N, K", get_x_vals())
# @pytest.mark.parametrize("M, N, K", [
#     (2, 2, 32),
#     (4, 4, 32),
#     (8, 8, 32),
#     (16, 16, 32),
#     (32, 32, 32),
#     (48, 48, 32),
#     (64, 64, 32),
#     (512, 512, 512),
#     (1024, 1024, 1024),
#     (9728,8192,65536),
#     (1,1280,8192)
# ])
@pytest.mark.parametrize("a_dtype", [e4m3_type])  # [e4m3_type, e5m2_type, torch.int8]
@pytest.mark.parametrize("out_dtype", [torch.float16])
def test_gemm_a8wfp4(M: int, N: int, K: int, a_dtype, out_dtype, CLEAR_GPUS=True):
    torch.manual_seed(42)  # for reproducibility
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    # clean up to avoid hangs in large tests
    if CLEAR_GPUS:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # generate fp32 tensors first
    x_fp32, w_fp32 = generate_fp32_tensors(M, N, K, INPUT_TYPE)
    torch_ref_out = torch.mm(x_fp32, w_fp32.T).to(out_dtype)
    if DEBUG:
        print()
        print("x_fp32:", x_fp32, x_fp32.shape)
        print("w_fp32:", w_fp32, w_fp32.shape)
        print("torch_ref_out:", torch_ref_out, torch_ref_out.shape)

    # quantize to 8-bit and fp4
    x, x_scales = quantize_to_8bit(x_fp32, a_dtype)
    w, w_scales = quantize_to_fp4(w_fp32)  # generate_random_fp4_inputs(N, K)
    assert x.shape[1] == w.shape[1] * 2

    if DEBUG:
        print()
        print("x", x, x.shape)
        print("x_scales", x_scales, x_scales.shape)
        print("w", w, w.shape)
        print("w_scales", w_scales, w_scales.shape)
        print(
            f"NOTE: we have shape {M}x{K} for A (fp8) and {N}x{K//2} for B (fp4). 2 fp4 values are packed into each uint8 value in the B tensor."
        )
        print("=== Debug: Matrix Values  ===")
        x_f32 = dequantize_fp8(x, x_scales)
        print(x_f32, x_f32.shape)
        w_f32 = dequantize_fp4(w, w_scales)
        print(w_f32, w_f32.shape)
        print(f"Expected result: each element should be {K} (sum of {K} ones)")

        print("=== What Triton Kernel Will See ===")
        print("A matrix raw bytes (what tl.load will return):")
        x_uint8 = x.view(torch.uint8)
        print(f"x as uint8: {x_uint8}")
        print(
            f"These are the raw byte values - 448 in fp8_e4m3fn is encoded as byte value {x_uint8[0, 0]}"
        )

        print("B matrix raw bytes:")
        print(f"w as uint8: {w}")
        print(
            f"0x22 = {0x22} = two packed fp4 values: lower nibble = 2 (1.0), upper nibble = 2 (1.0)"
        )

        print("Scale values:")
        print(f"a_scales (fp32): {x_scales.flatten()}")
        print(f"b_scales (e8m0 as uint8): {w_scales.flatten()}")
        print(f"b_scales decoded to fp32: {e8m0_to_f32(w_scales).flatten()}")
    torch_emulated_out = run_torch_emulation(x, w, x_scales, w_scales, out_dtype).to(
        out_dtype
    )
    if DEBUG:
        print("torch_emulated_out", torch_emulated_out, torch_emulated_out.shape)

    if ZERO_OUTPUT:
        triton_out = torch.zeros(
            x.shape[0], w.shape[0], device=x.device, dtype=out_dtype
        )
    else:
        triton_out = torch.empty(
            x.shape[0], w.shape[0], device=x.device, dtype=out_dtype
        )
    gemm_a8wfp4(x, w.T, triton_out, x_scales, w_scales, out_dtype)
    if DEBUG:
        print("triton_out:", triton_out, triton_out.shape)

    torch.testing.assert_close(
        torch_emulated_out, triton_out, atol=0.01, rtol=1e-2, equal_nan=True
    )

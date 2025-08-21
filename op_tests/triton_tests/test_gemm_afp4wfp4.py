# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import os
import torch
from aiter.ops.triton.gemm_afp4wfp4 import (
    gemm_afp4wfp4,
    gemm_afp4wfp4_preshuffled_scales,
)
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.types import str_to_torch_dtype

TRITON_HIP_PRESHUFFLE_SCALES = (
    os.environ.get("TRITON_HIP_PRESHUFFLE_SCALES", "0") == "1"
)


def shuffle_scales(scales: torch.Tensor):
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape
    scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled


# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32


def generate_gemm_afp4wfp4_inputs(M, N, K, dtype, layout="TN", output=True):
    torch.manual_seed(5)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    if layout[0] == "T":
        # 34 is two packed e2m1 values 0010 which is 1.0.
        x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
        x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8)
    else:
        x_low = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8).T
        x_high = torch.randint(0, 16, (K // 2, M), dtype=torch.uint8).T

    if layout[1] == "N":
        w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
        w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    else:
        w_low = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T
        w_high = torch.randint(0, 16, (K // 2, N), dtype=torch.uint8, device="cuda").T

    x = (
        x_high << 4 | x_low
    )  # Doing this computation on GPU tensors results in NaNs, so move it to GPU afterwards
    x = x.to(device="cuda")

    w = w_low | w_high << 4
    # Scale of 1.0 in e8m0, bias 127.
    if M >= 32 and TRITON_HIP_PRESHUFFLE_SCALES:
        M_pad = (M + 255) // 256 * 256
    else:
        M_pad = M
    x_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, M_pad), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda"
    )
    x_scales = x_scales.T
    w_scales = w_scales.T
    if TRITON_HIP_PRESHUFFLE_SCALES:
        if M >= 32:
            x_scales_shuffled = shuffle_scales(x_scales)
        else:
            x_scales_shuffled = x_scales
        w_scales_shuffled = shuffle_scales(w_scales)
    else:
        x_scales_shuffled = x_scales
        w_scales_shuffled = w_scales

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return (
        x,
        w,
        x_scales[:M],
        w_scales,
        x_scales_shuffled,
        w_scales_shuffled,
        out_dtype,
        y,
    )


def mxfp4_to_f32(x):
    # 2 because we pack fp4 in uint8.
    x = x.repeat_interleave(2, dim=1)
    x[:, ::2] = x[:, ::2] & 0xF
    x[:, 1::2] = x[:, 1::2] >> 4
    mxfp4_list = [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        -0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def run_torch(x, w, x_scales, w_scales, dtype):
    # First convert the x and w inputs to f32.
    x_f32 = mxfp4_to_f32(x)
    w_f32 = mxfp4_to_f32(w)
    # Next convert the e8m0 scales to f32.
    x_scales = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales)
    x_f32 = x_f32 * x_scales_f32
    w_scales = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales)
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)


def basic_shape_set():
    shapes = [(1, 1, SCALE_GROUP_SIZE)]  # minimal case
    shapes += [
        (32, 32, 32),
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (4864, 4096, 8192),
    ]
    shapes += [(2**i, 256, 7168) for i in range(1, 4)]
    return shapes


def extended_shape_set():
    shapes = [(2**i, 256, 7168) for i in range(5, 9)]
    shapes += [(1024 * v, 1024 * v, 1024 * v) for v in range(2, 9)]
    shapes += [(9728, 8192, 65536), (4864, 8192, 4160)]
    shapes += [
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
    shapes += [
        (256, 8192, 1024),
        (256, 1024, 8192),
        (256, 32768, 8192),
        (256, 8192, 32768),
    ]
    shapes += [(16, 16384, 3328 * 2), (128, 16384, 3328 * 2)]
    return shapes


@pytest.mark.parametrize(
    "M, N, K, dtype, output, layout",
    [
        (*shape, dtype_str, output, layout)
        for shape in basic_shape_set()
        for dtype_str in ["bf16"]
        for output in [True]
        for layout in ["TN", "TT", "NN", "NT"]
    ]
    + [
        pytest.param(*shape, dtype_str, output, layout, marks=pytest.mark.extended)
        for shape in extended_shape_set()
        for dtype_str in ["bf16", "fp16"]
        for output in [True]  # TODO: Debug False fails
        for layout in ["TN", "TT", "NN", "NT"]
    ],
)
def test_gemm_afp4_wfp4(M: int, N: int, K: int, dtype, output, layout):
    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    dtype = str_to_torch_dtype[dtype]
    if TRITON_HIP_PRESHUFFLE_SCALES:
        if N % 32 > 0:
            pytest.skip(
                f"N = {N} is not divisible by 32, skip this test for preshuffled scales tests"
            )
        elif K % 256 > 0:
            pytest.skip(
                f"K = {K} is not divisible by 256, skip this test for preshuffled scales tests"
            )

    (
        x,
        w,
        x_scales,
        w_scales,
        x_scales_triton,
        w_scales_triton,
        out_dtype,
        y,
    ) = generate_gemm_afp4wfp4_inputs(M, N, K, dtype, layout=layout, output=output)

    torch_out = run_torch(x, w, x_scales, w_scales, dtype).to(dtype)

    if TRITON_HIP_PRESHUFFLE_SCALES:
        if output:
            triton_out = gemm_afp4wfp4_preshuffled_scales(
                x, w, x_scales_triton, w_scales_triton, dtype, y
            )
        else:
            triton_out = gemm_afp4wfp4_preshuffled_scales(
                x, w, x_scales_triton, w_scales_triton, dtype
            )
    else:
        if output:
            triton_out = gemm_afp4wfp4(x, w, x_scales_triton, w_scales_triton, dtype, y)
        else:
            triton_out = gemm_afp4wfp4(x, w, x_scales_triton, w_scales_triton, dtype)

    torch.testing.assert_close(torch_out, triton_out)

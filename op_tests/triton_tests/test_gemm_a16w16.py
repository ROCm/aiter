# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import triton
import pytest
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.utils.types import str_to_torch_dtype


class TestGemmA16W16:
    def generate_gemm_a16w16_inputs(M, N, K, dtype, layout="TN", output=True):
        if isinstance(dtype, str):
            dtype = str_to_torch_dtype[dtype]

        # TN is default layout
        if layout[0] == "T":
            x = torch.randn((M, K), dtype=dtype).cuda()
        else:
            x = torch.randn((K, M), dtype=dtype).cuda().T

        if layout[1] == "T":
            weight = torch.randn((K, N), dtype=dtype).cuda().T
        else:
            weight = torch.randn((N, K), dtype=dtype).cuda()

        y = None
        if output:
            y = torch.empty((M, N), dtype=dtype).cuda()
            out_dtype = (None,)
        else:
            out_dtype = dtype

        return x, weight, out_dtype, y

    # Test Parameters
    basic_shape_set = [(1024 * v, 1024 * v, 1024 * v) for v in range(1, 6)]
    basic_shape_set += [(4864, 4096, 8192), (9728, 8192, 65536), (4864, 8192, 4160)]
    basic_set = [
        pytest.param(*shape, dtype, output)
        for shape in basic_shape_set
        for dtype in ["bfloat16", "float16"]
        for output in [True, False]
    ]

    extended_shape_set = [(1024 * v, 1024 * v, 1024 * v) for v in range(6, 9)]
    extended_shape_set += [(2**i, 256, 7168) for i in range(5, 9)]
    extended_shape_set += [
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
    extended_shape_set += [(1, 1, 1)]  # minimal case
    extended_set = [
        pytest.param(*shape, dtype, output, marks=pytest.mark.extended)
        for shape in extended_shape_set
        for dtype in ["bfloat16", "float16"]
        for output in [True, False]
    ]

    test_params = basic_set + extended_set

    @pytest.mark.parametrize("M, N, K, dtype_str, output", test_params)
    def test_gemm_a16_w16(self, M: int, N: int, K: int, dtype_str, output):
        x, w, out_dtype, y = TestGemmA16W16.generate_gemm_a16w16_inputs(
            M, N, K, str_to_torch_dtype[dtype_str], output=output
        )

        torch_out = F.linear(x, w, bias=None)

        if output:
            triton_out = gemm_a16w16(x, w, out_dtype, y)
        else:
            triton_out = gemm_a16w16(x, w, out_dtype)

        triton.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)

    @pytest.mark.parametrize("M, N, K, dtype_str, output", test_params)
    def test_gemm_a16_w16_atomic(self, M: int, N: int, K: int, dtype_str, output):
        dtype = str_to_torch_dtype[dtype_str]
        x, w, out_dtype, y = TestGemmA16W16.generate_gemm_a16w16_inputs(
            M, N, K, dtype, output=output
        )

        torch_out = F.linear(x, w, bias=None)

        # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
        if output:
            y = y.to(torch.float32).zero_()
            triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(dtype)
        else:
            triton_out = gemm_a16w16_atomic(x, w, dtype=torch.float32).to(dtype)

        triton.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)

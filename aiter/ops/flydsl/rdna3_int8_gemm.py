# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL RDNA3 INT8 GEMM launcher."""

from functools import lru_cache

import torch

from aiter.ops.flydsl.kernels.rdna3_int8_gemm import create_wmma_int8_gemm_module


@lru_cache(maxsize=64)
def _get_kernel(arch, m, n, k, lda, ldb, out_dtype):
    reg_n, waves_n = (4, 2) if n % 128 == 0 else (2, 2)
    return create_wmma_int8_gemm_module(
        m,
        n,
        k,
        in_dtype="int8",
        out_dtype=out_dtype,
        scale_mode="row_col",
        reg_m=2,
        reg_n=reg_n,
        reg_k=4,
        waves_m=2,
        waves_n=waves_n,
        lda=lda,
        ldb=ldb,
        ldc=n,
    )[0]


def gemm_a8w8_rdna3(x, w, x_scale, w_scale, out):
    """Run the scaled RDNA3 WMMA kernel for the AITER A8W8 contract."""
    out_dtype = {
        torch.float32: "f32",
        torch.bfloat16: "bf16",
        torch.float16: "f16",
    }[out.dtype]
    with torch.cuda.device(x.device):
        arch = str(torch.cuda.get_device_properties(x.device).gcnArchName).split(":")[0]
        kernel = _get_kernel(
            arch,
            x.shape[0],
            w.shape[0],
            x.shape[1],
            x.stride(0),
            w.stride(0),
            out_dtype,
        )
        kernel(
            out,
            x,
            w,
            torch.cuda.current_stream(x.device),
            x_scale.reshape(-1),
            w_scale.reshape(-1),
        )
    return out

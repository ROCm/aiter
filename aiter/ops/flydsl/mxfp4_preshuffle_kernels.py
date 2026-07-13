# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host dispatcher for the FlyDSL MXFP4/MXFP6/MXFP8 preshuffle GEMM (gfx950 MFMA).

Wraps the self-contained ``@flyc.jit launch_gemm`` in
``aiter/ops/flydsl/kernels/mxfp4_preshuffle.py``. ``launch_gemm`` already caches
and dispatches per distinct Constexpr config (N, K, tile_*, a_dtype, b_dtype,
out_dtype, batch, strides, waves_per_eu), so this layer only marshals torch
tensors to raw pointers via ``tensor_shim.ptr_arg`` and forwards the config.

Operand convention (matches the kernel's host preshuffle):
    A       : [M, K]   row-major, NOT preshuffled  (fp8/fp6 = 1 byte/code, fp4 = 2 codes/byte)
    B       : preshuffled via shuffle_weight_w4(., 16)  (fp4 or fp8 weight)
    a_scale : per-1x32 E8M0, shuffle_scale_w4'd
    b_scale : per-1x32 E8M0, shuffle_scale_w4'd
    Out     : [M, N]   bf16 / fp16
"""

from __future__ import annotations

import torch

from aiter.ops.flydsl.utils import is_flydsl_available

_OUT_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "fp16"}


def flydsl_mxfp4_preshuffle_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    Out: torch.Tensor,
    *,
    a_dtype: str,
    b_dtype: str = "fp4",
    tile_m: int,
    tile_n: int,
    tile_k: int,
    waves_per_eu: int = 0,
    stream=None,
) -> torch.Tensor:
    """Run the gfx950 MXFP4/6/8 preshuffle GEMM. a8w8 = a_dtype="fp8", b_dtype="fp8".

    A is [M, K]; N is taken from Out ([M, N]); K from A. Returns Out.
    """
    if not is_flydsl_available():
        raise RuntimeError("flydsl is not available; cannot run mxfp4_preshuffle GEMM")

    from .kernels.mxfp4_preshuffle import launch_gemm
    from .kernels.tensor_shim import ptr_arg

    # Logical K: fp4 A packs 2 codes/byte (A last dim = K//2); fp6/fp8 A = 1 byte/code.
    M = int(A.shape[0])
    K = int(A.shape[-1]) * (2 if a_dtype == "fp4" else 1)
    N = int(Out.shape[-1])
    out_dtype = _OUT_DTYPE_STR.get(Out.dtype)
    if out_dtype is None:
        raise ValueError(f"unsupported Out dtype {Out.dtype}; expected bfloat16 or float16")

    st = stream if stream is not None else torch.cuda.current_stream()

    launch_gemm(
        ptr_arg(Out),
        ptr_arg(A),
        ptr_arg(B),
        ptr_arg(a_scale),
        ptr_arg(b_scale),
        M,
        N,
        st,
        N,
        K,
        int(tile_m),
        int(tile_n),
        int(tile_k),
        a_dtype,
        out_dtype,
        b_dtype,
        1,  # batch
        -1,  # a_row_stride
        -1,  # a_batch_stride
        -1,  # sca_row_stride
        -1,  # sca_batch_stride
        -1,  # c_row_stride
        -1,  # c_batch_stride
        int(waves_per_eu),
    )
    return Out

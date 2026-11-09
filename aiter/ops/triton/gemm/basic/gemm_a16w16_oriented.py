# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
    _gemm_splitk_reduce_kernel,
)
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a16w16 import (
    _gemm_a16_w16_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_xcds
from aiter.ops.triton.utils.gemm_config_utils import (
    compute_splitk_params,
    get_gemm_config,
)


def gemm_a16w16_oriented(x, w, bias=None, dtype=None, out=None):
    """Compute X[M,K] @ W[N,K].T, with optional bias[N].

    Inputs must be FP16 or BF16 on gfx950. Input views may have positive strides.
    Output is contiguous [M,N], in the input dtype or FP32. If out is supplied,
    write into that buffer. K partials are FP32 and are reduced before casting.
    """
    if get_arch() != "gfx950":
        raise ValueError("oriented A16W16 requires gfx950")
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[1]:
        raise ValueError("Expected X[M,K] and W[N,K]")
    if not x.is_cuda or w.device != x.device or w.dtype != x.dtype:
        raise ValueError("X and W must have the same GPU device and dtype")
    m, k = x.shape
    n = w.shape[0]
    dtype = x.dtype if dtype is None else dtype
    if min(m, n, k) <= 0 or min(*x.stride(), *w.stride()) <= 0:
        raise ValueError("Positive dimensions and strides are required")
    if x.dtype not in (torch.float16, torch.bfloat16) or dtype not in (
        x.dtype,
        torch.float32,
    ):
        raise ValueError("Expected FP16/BF16 inputs and same-dtype or FP32 output")
    if bias is not None:
        if bias.device != x.device or bias.shape != (n,):
            raise ValueError("bias must be an N-vector on the input device")
        bias = bias.contiguous()
    if out is not None and (
        out.device != x.device
        or out.dtype != dtype
        or out.shape != (m, n)
        or not out.is_contiguous()
    ):
        raise ValueError("out must be contiguous [M,N] on the input device and dtype")
    config, _ = get_gemm_config("GEMM-A16W16-ORIENTED", m, n, k)
    reverse = config.pop("REVERSE")
    use_xcd = config.pop("USE_XCD")
    reduce_block = config.pop("REDUCE_BLOCK")
    config = compute_splitk_params(config, k)
    sk = config["NUM_KSPLIT"]
    y = torch.empty((m, n), dtype=dtype, device=x.device) if out is None else out
    partials = (
        torch.empty((sk, m, n), dtype=torch.float32, device=x.device) if sk > 1 else y
    )
    # Swap operands and output strides together; no tensor transpose is materialized.
    left, right = (w, x.t()) if reverse else (x, w.t())
    rows, cols = (n, m) if reverse else (m, n)
    stride_m, stride_n = (1, n) if reverse else (n, 1)
    grid = (
        sk
        * triton.cdiv(rows, config["BLOCK_SIZE_M"])
        * triton.cdiv(cols, config["BLOCK_SIZE_N"])
    )
    _gemm_a16_w16_kernel[(grid,)](
        left,
        right,
        bias,
        partials,
        rows,
        cols,
        k,
        *left.stride(),
        *right.stride(),
        m * n,
        stride_m,
        stride_n,
        NUM_XCDS=get_num_xcds() if use_xcd else 1,
        ADD_BIAS=bias is not None,
        BIAS_ON_ROWS=reverse,
        GROUP_SPLIT_K=True,
        SKIP_REDUCE=False,
        activation="",
        use_activation=False,
        **config,
    )
    if sk > 1:
        # The shared reducer uses modulo coordinates. Exact N tiles avoid
        # duplicate stores for an output tail.
        reduce_block = min(reduce_block, n & -n)
        _gemm_splitk_reduce_kernel[(m, triton.cdiv(n, reduce_block))](
            partials,
            y,
            bias,
            m,
            n,
            m * n,
            n,
            1,
            n,
            1,
            1,
            reduce_block,
            sk,
            triton.next_power_of_2(sk),
            ADD_BIAS=bias is not None,
            activation="",
            use_activation=False,
        )
    return y

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the FlyDSL fn-resident mHC pre GEMM (gfx950).

Same arguments and outputs as ``mhc_pre_gemm_sqrsum`` with ``tile_k == 1024``: ``out``
(split_k, m, 32-padded) and ``sqrsum`` (split_k, m), both fp32. See
``aiter/ops/flydsl/kernels/mhc_pre_gemm.py`` for the kernel.
"""

import torch

from aiter.jit.utils.chip_info import get_cu_num

__all__ = ["flydsl_mhc_pre_gemm_sqrsum"]


def flydsl_mhc_pre_gemm_sqrsum(
    out: torch.Tensor,
    sqrsum: torch.Tensor,
    x: torch.Tensor,
    fn: torch.Tensor,
    w_preshuffle_bf16: bool,
) -> None:
    from aiter.ops.flydsl.kernels.mhc_pre_gemm import KS, build_mhc_pre_gemm
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled, ptr_arg

    m = x.size(0)
    k = fn.size(1)
    split_k = k // KS
    assert out.size(0) == split_k and sqrsum.size(0) == split_k
    assert x.is_contiguous() and x.numel() == m * k
    assert fn.is_contiguous()
    out_stride = out.stride(1)

    # One workgroup per CU: split_k column groups, rows divided over the rest.
    groups = max(1, get_cu_num() // split_k)
    rows_per_group = ((m + groups - 1) // groups + 15) // 16 * 16
    groups = (m + rows_per_group - 1) // rows_per_group
    num_waves = 16 if m >= 8192 else 8

    launcher = build_mhc_pre_gemm(k, num_waves, bool(w_preshuffle_bf16))
    _run_compiled(
        launcher,
        ptr_arg(fn),
        ptr_arg(x),
        ptr_arg(out),
        ptr_arg(sqrsum),
        m,
        rows_per_group,
        out_stride,
        groups,
        torch.cuda.current_stream(x.device),
    )

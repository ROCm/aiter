# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dispatch-side helper shared by the a8w8 FlyDSL split-K bpreshuffle paths:
``dispatch_flydsl_splitk`` (the parsed-kernelName ->
``flydsl_preshuffle_gemm_splitk_a8`` call, shared by
``gemm_op_a8w8.gemm_a8w8_bpreshuffle_flydsl`` and the inline split-K branch in
``gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle``).

Callers still own kernelName parsing (the format differs per family) and the
parse-failure fallback (differs per call site); this only covers the part
that is byte-identical across them -- kwarg construction and the actual
kernel call.
"""

import torch


def dispatch_flydsl_splitk(
    XQ: torch.Tensor,
    WQ: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    Out: torch.Tensor,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    split_k: int,
    *,
    use_async_copy: int,
    waves_per_eu: int,
    xcd_swizzle: int,
    lds_stage: int,
    scheduler: str,
    scale_mode: str,
    use_m_bounded_store: bool,
    in_dtype: str | None = None,
) -> torch.Tensor:
    """Run one parsed FlyDSL split-K candidate and return ``Out``.

    ``in_dtype=None`` resolves to int8/fp8 by ``XQ.dtype`` inside
    ``flydsl_preshuffle_gemm_splitk_a8``.
    """
    from .kernels.preshuffle_gemm_splitk_op import flydsl_preshuffle_gemm_splitk_a8

    XQ, WQ = XQ.contiguous(), WQ.contiguous()

    flydsl_preshuffle_gemm_splitk_a8(
        XQ,
        WQ,
        x_scale,
        w_scale,
        Out,
        tile_m,
        tile_n,
        tile_k,
        split_k,
        use_async_copy=use_async_copy,
        waves_per_eu=waves_per_eu,
        xcd_swizzle=xcd_swizzle,
        lds_stage=lds_stage,
        enable_scheduler=str(scheduler).lower() != "off",
        scale_mode=scale_mode,
        use_m_bounded_store=use_m_bounded_store,
        in_dtype=in_dtype,
    )
    return Out

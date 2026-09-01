# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dispatch-side helpers shared by the a8w8 and a4w4 FlyDSL split-K bpreshuffle
paths: ``is_flydsl_available`` (was copy-defined verbatim in
``gemm_op_a8w8.py`` and ``gemm_op_a4w4.py``) and ``dispatch_flydsl_splitk``
(the parsed-kernelName -> ``flydsl_preshuffle_gemm_splitk_a8`` call, shared by
``gemm_op_a8w8.gemm_a8w8_bpreshuffle_flydsl``, the inline split-K branch in
``gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle``, and
``gemm_op_a4w4.gemm_a4w4``).

Callers still own kernelName parsing (the format differs per family) and the
parse-failure fallback (differs per call site); this only covers the part
that was byte-identical across all three -- kwarg construction and the actual
kernel call. The a4w4 family additionally preshuffles B and both scales before
the call, which a8w8 does not need; that difference is a ``preshuffle`` hook,
not a second copy of the dispatch body.
"""

from collections.abc import Callable

import torch


def is_flydsl_available() -> bool:
    try:
        from .utils import is_flydsl_available as _is_flydsl_available
    except ImportError:
        return False
    return _is_flydsl_available()


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
    stage_a_scales: bool = False,
    preshuffle: (
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ]
        | None
    ) = None,
) -> torch.Tensor:
    """Run one parsed FlyDSL split-K candidate and return ``Out``.

    ``preshuffle``, when given, replaces the default ``XQ.contiguous(),
    WQ.contiguous()`` prep -- it receives ``(XQ, WQ, x_scale, w_scale)`` and
    must return the four tensors to actually pass to the kernel (the a4w4
    caller uses this to shuffle B and both scales into the CDNA4 scaled-MFMA
    preshuffle layout). ``in_dtype=None`` matches every pre-refactor a8w8 call
    site (none passed it), which resolves to int8/fp8 by ``XQ.dtype`` inside
    ``flydsl_preshuffle_gemm_splitk_a8``; a4w4 passes ``in_dtype="fp4"``.
    """
    from .kernels.preshuffle_gemm_splitk_op import flydsl_preshuffle_gemm_splitk_a8

    if preshuffle is not None:
        XQ, WQ, x_scale, w_scale = preshuffle(XQ, WQ, x_scale, w_scale)
    else:
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
        stage_a_scales=stage_a_scales,
    )
    return Out

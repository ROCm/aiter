# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL batched mxfp8 GEMM on a (16, 16)-preshuffled weight, on whichever
arch has a kernel for the operands' w_scale block.

``XQ [M, B, K]`` fp8, ``WQ [B, N, K]`` fp8 preshuffled, ``x_scale`` and
``w_scale`` e8m0 blocks, ``Out [M, B, N]`` bf16. ``x_scale`` is row-major, or
with x_scale_transposed (B = 1, 1x128 blocks) column-major [K / 128, M] under
the same shape, the blockscale convention. The per-arch modules
(batched_gemm_a8w8_gfx950, batched_gemm_a8w8_gfx1250) hold the kernels; a tuned
kernelName names one of them, so it is only valid on its arch.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable

from torch import Tensor

from aiter.jit.utils.chip_info import get_gfx

from ..gemm_op_common import mxscale_w_scale_block

_GFX950 = ("batched_gemm_a8w8_gfx950", "run_bmm_a8w8_mxfp8_gfx950")
_GFX1250 = ("batched_gemm_a8w8_gfx1250", "run_bmm_a8w8_mxfp8_128_gfx1250")
# (gfx, w_scale block, x_scale column-major) -> (module, runner, its layout
# argument). Only the running arch's module is ever imported.
_RUNNERS = {
    ("gfx950", "32x32", False): (*_GFX950, {}),
    ("gfx950", "1x32", False): (*_GFX950, {}),
    ("gfx950", "128x128", False): (*_GFX950, {}),
    ("gfx950", "128x128", True): (*_GFX950, {"x_scale_transposed": True}),
    ("gfx1250", "128x128", False): (*_GFX1250, {}),
}


@functools.cache
def _runner(gfx: str, w_scale_block: str, x_scale_transposed: bool) -> Callable | None:
    entry = _RUNNERS.get((gfx, w_scale_block, x_scale_transposed))
    if entry is None:
        return None
    module, name, layout = entry
    run = getattr(importlib.import_module(f".{module}", __package__), name)
    return functools.partial(run, **layout) if layout else run


def bmm_a8w8_mxfp8_supported(
    w_scale_block: str, x_scale_transposed: bool = False
) -> bool:
    """Whether this arch has a kernel for a w_scale block and x_scale layout."""
    return _runner(get_gfx(), w_scale_block, x_scale_transposed) is not None


def run_bmm_a8w8_mxfp8(
    XQ: Tensor,
    WQ: Tensor,
    x_scale: Tensor,
    w_scale: Tensor,
    Out: Tensor,
    kernel_name: str | None = None,
    x_scale_transposed: bool = False,
) -> Tensor:
    """``Out[m, b] = dequant(XQ[m, b]) @ dequant(WQ[b]).T`` on this arch's
    kernel for the w_scale block and x_scale layout; writes into ``Out`` and
    returns it. ``kernel_name`` is a tuned row's kernelName, else the arch's
    heuristic picks one."""
    n, k = WQ.shape[-2], WQ.shape[-1]
    w_scale_block = mxscale_w_scale_block(tuple(w_scale.shape), n, k)
    run = _runner(get_gfx(), w_scale_block, x_scale_transposed)
    if run is None:
        layout = "column-major" if x_scale_transposed else "row-major"
        raise NotImplementedError(
            f"no preshuffled mxfp8 BMM kernel reads a {w_scale_block} w_scale "
            f"with a {layout} x_scale on {get_gfx()}"
        )
    return run(XQ, WQ, x_scale, w_scale, Out, kernel_name=kernel_name)


__all__ = ["bmm_a8w8_mxfp8_supported", "run_bmm_a8w8_mxfp8"]
